"""Shared utilities: config loading, seeding, device, paths."""

import math
import os
import random
import statistics
from collections import deque
from pathlib import Path

import numpy as np
import yaml

# torch is imported lazily inside the two functions that need it. Importing it
# at module scope costs ~0.37 GiB RSS, which data.download_era5 would pay for
# nothing — and that matters against the 4 GiB cgroup cap on a BriCS login
# node, where the download has no GPU work at all.

PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(path: str | os.PathLike = "config/default.yaml") -> dict:
    """Load a YAML config, resolving relative paths against the project root."""
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    # Resolve all paths.* entries relative to project root.
    for key, val in cfg.get("paths", {}).items():
        p = Path(val)
        cfg["paths"][key] = str(p if p.is_absolute() else PROJECT_ROOT / p)
    return cfg


def add_perf_args(ap):
    """Register the hardware-dependent CLI knobs shared by every trainer.

    The checked-in config values are tuned for one machine (a 24 GB card, 4
    dataloader workers). On different hardware they are simply wrong — a 96 GB
    GH200 wants a far larger batch, a 288-core node wants more workers.
    Overriding from the CLI keeps one canonical config instead of a fork per
    cluster. Pair with `apply_perf_overrides`.
    """
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Override the training batch size. PER PROCESS under "
                         "torchrun, so the global batch is this x world size.")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="Override train.num_workers (dataloader subprocesses per "
                         "process). Too few starves the GPU, which is wasted spend "
                         "on a node-hour-billed cluster.")
    ap.add_argument("--amp-dtype", choices=["off", "fp16", "bf16"], default=None,
                    help="Mixed-precision mode. bf16 is the one to want on "
                         "GH200/H100: same exponent range as fp32, so no loss "
                         "scaling and none of fp16's overflow risk on "
                         "noise-prediction targets, at ~2x throughput and ~half "
                         "the activation memory. fp16 is the legacy behaviour of "
                         "`amp: true`. Default off — precision is not changed "
                         "silently.")
    return ap


def resolve_amp(section: dict, device_type: str):
    """(enabled, dtype) for torch.amp, from a config section's `amp_dtype`.

    `section` is the block that owns the setting — train for the generative
    trainers, directmap for the regression one — matching where each trainer
    already reads `amp`.

    Back-compat: a legacy `amp: true` with no `amp_dtype` means fp16, because
    that is what `autocast("cuda")` defaulted to before this existed. Reading
    it as bf16 would silently change the numerics of existing configs.
    """
    import torch  # noqa: PLC0415 - see the module-scope note

    mode = section.get("amp_dtype")
    if mode is None:
        mode = "fp16" if section.get("amp", False) else "off"
    if mode == "off" or device_type != "cuda":
        return False, None
    if mode == "fp16":
        return True, torch.float16
    if mode == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "amp_dtype: bf16 requested but this GPU reports no bf16 support")
        return True, torch.bfloat16
    raise ValueError(f"unknown amp_dtype {mode!r} (expected off, fp16 or bf16)")


def apply_perf_overrides(cfg: dict, args, batch_section: str = "train") -> dict:
    """Apply --batch-size / --num-workers onto a loaded config, in place.

    `batch_section` is the config section owning batch_size — "train" for the
    generative trainers, "directmap" for the regression one. num_workers always
    lives under train.
    """
    if getattr(args, "batch_size", None) is not None:
        cfg[batch_section]["batch_size"] = args.batch_size
    if getattr(args, "num_workers", None) is not None:
        cfg["train"]["num_workers"] = args.num_workers
    if getattr(args, "amp_dtype", None) is not None:
        # Same section that owns batch_size, which is also where each trainer
        # already reads `amp`.
        cfg[batch_section]["amp_dtype"] = args.amp_dtype
    return cfg


class DivergenceGuard:
    """Stop a run whose validation loss has blown up and is not recovering.

    Motivated by a real 200-epoch run (job 6169725, the 20-var no-geo baseline):
    it trained healthily to epoch 89 (val 0.01261, better than either surviving
    arm at that point), collapsed to the trivial solution by epoch 92
    (val 0.99980), then trained 108 more epochs at full cost producing a
    worthless checkpoint. Nothing in the trainer noticed. That is ~1.3 wasted
    node-hours per occurrence on a cluster billed by the node-hour.

    Fires only after `patience` CONSECUTIVE epochs above `factor` x the best
    validation loss seen so far. The consecutive requirement matters: the same
    run spiked to 0.18781 at epoch 10 (6.4x the then-best 0.02914) and recovered
    completely, so a single-epoch trigger would have killed a healthy run.

    `min_epochs` skips the early phase, where the loss is still falling fast and
    "best so far" is not yet meaningful.
    """

    def __init__(self, factor: float = 10.0, patience: int = 3,
                 min_epochs: int = 5, ceiling=None):
        self.ceiling = None if ceiling is None else float(ceiling)
        self.factor = float(factor)
        self.patience = int(patience)
        self.min_epochs = int(min_epochs)
        self.best = float("inf")
        self.strikes = 0

    def update(self, val_loss: float, epoch: int):
        """Feed one epoch's validation loss.

        Returns a human-readable reason string when the run should abort, else
        None. Call on the rank that actually computes validation, then agree
        across ranks with distributed.broadcast_flag — aborting on one rank
        alone deadlocks the others at the next collective.
        """
        if val_loss is None or not math.isfinite(val_loss):
            self.strikes += 1
            if self.strikes >= self.patience:
                return f"non-finite validation loss for {self.strikes} epochs"
            return None
        if epoch < self.min_epochs:
            self.best = min(self.best, val_loss)
            return None
        threshold = self.factor * self.best
        # Absolute ceiling. The trivial epsilon-prediction solution scores ~1.0 in
        # EVERY arm, but the relative factor cannot see it from a high loss
        # floor: the residual arm's best is 0.123, so its collapse to 1.002 is
        # 8.1x -- under 10x -- and job 6369451 ran 165 epochs collapsed with this
        # guard watching. Gated on the run having been healthy first, so a slow
        # starter still above the ceiling after min_epochs is not rolled back.
        over_ceiling = (self.ceiling is not None and self.best < self.ceiling / 2
                        and val_loss > self.ceiling)
        if val_loss > threshold or over_ceiling:
            self.strikes += 1
            if self.strikes >= self.patience:
                if over_ceiling and not val_loss > threshold:
                    return (f"val loss {val_loss:.5f} above the absolute ceiling "
                            f"{self.ceiling:g} (trivial-solution signature; best was "
                            f"{self.best:.5f}) for {self.strikes} consecutive epochs")
                return (f"val loss {val_loss:.5f} exceeded {self.factor:g}x the best "
                        f"({self.best:.5f}) for {self.strikes} consecutive epochs")
        else:
            self.strikes = 0
            self.best = min(self.best, val_loss)
        return None


def build_divergence_guard(train_cfg: dict):
    """DivergenceGuard from a config's `divergence` block; None when disabled."""
    d = train_cfg.get("divergence", {})
    if not d.get("enabled", True):
        return None
    return DivergenceGuard(factor=d.get("factor", 10.0),
                           patience=d.get("patience", 3),
                           min_epochs=d.get("min_epochs", 5),
                           ceiling=d.get("ceiling"))


class SpikeGuard:
    """Skip the optimizer step on a wildly anomalous gradient.

    DivergenceGuard is a post-mortem: it detects a collapse that has already
    happened and salvages the run's cost. This is the prevention.

    The 20-var no-geo baseline collapsed TWICE — job 6169725 at epoch 92 and
    job 6189791 at epoch 82, from different RNG states — while both geo arms ran
    200 epochs untouched (max gradient norm over the whole run: hpx 0.374,
    static 1.578, against the no-geo p99.9 of 0.548 and a collapse-window mean
    of 3.167). Both collapses have the same shape: one 50-step window where the
    mean gradient norm jumps ~30x and the loss goes from ~0.014 to ~1.0, the
    trivial eps_hat = 0 solution, which is absorbing — gradients there are
    ~0.03, too small to climb back out.

    grad_clip does NOT prevent this, which is why it kept happening with
    clip_grad_norm_(1.0) in force. Clipping rescales the gradient, but Adam
    divides by its own second-moment estimate, so the parameter step is ~lr per
    element almost regardless of the gradient's magnitude — what clipping
    changes is the DIRECTION's weight in the moment estimates, not the distance
    travelled. A single pathological batch still moves 244M parameters by ~lr
    each, and a handful in a row is enough to leave the basin.

    So the only effective intervention is to not take the step at all. Skipping
    before opt.step() also leaves Adam's moments untouched, so the bad direction
    does not persist through subsequent steps the way clipping would allow.

    The threshold is RELATIVE to a running median of recent gradient norms
    rather than absolute, because the healthy scale differs per arm and drifts
    over training (early epochs run ~10x the late-epoch norm). `warmup` steps
    accumulate that history before the guard can fire, and the guard stays off
    while the history is short.

    Under DDP no cross-rank agreement is needed: gradients are all-reduced in
    the backward pass, so every rank computes an identical norm and reaches an
    identical decision. Do not add a broadcast here — it would be a collective
    inside a branch, which is exactly how ranks deadlock.
    """

    def __init__(self, factor: float = 50.0, window: int = 200,
                 warmup: int = 200, max_consecutive: int = 20):
        self.factor = float(factor)
        self.window = int(window)
        self.warmup = int(warmup)
        self.max_consecutive = int(max_consecutive)
        self.history: deque = deque(maxlen=self.window)
        self.skipped = 0
        self.consecutive = 0
        self.last_reason = None

    def check(self, grad_norm: float) -> bool:
        """True = skip this optimizer step. Feed the PRE-clip gradient norm.

        A non-finite norm is always skipped, at any point in training: there is
        no history for which inf is a reasonable step.
        """
        if grad_norm is None or not math.isfinite(grad_norm):
            self.skipped += 1
            self.consecutive += 1
            self.last_reason = f"non-finite gradient norm ({grad_norm})"
            return True
        if len(self.history) < self.warmup:
            self.history.append(grad_norm)
            self.consecutive = 0
            return False
        med = statistics.median(self.history)
        threshold = self.factor * med
        # med can be ~0 late in a converged run; the floor keeps the threshold
        # from collapsing onto the noise and skipping every step.
        if med > 0 and grad_norm > threshold:
            self.skipped += 1
            self.consecutive += 1
            self.last_reason = (f"gradient norm {grad_norm:.3f} exceeded "
                                f"{self.factor:g}x the running median "
                                f"({med:.4f}) over the last {len(self.history)} steps")
            return True
        # Only healthy steps enter the history, so a run that is genuinely
        # diverging cannot raise its own threshold to accommodate itself.
        self.history.append(grad_norm)
        self.consecutive = 0
        return False

    def reset(self, scale: float = None) -> None:
        """Re-arm the threshold at `scale`. NEVER leaves the guard unarmed.

        A bare history.clear() DISARMS the guard: check() returns False
        unconditionally while len(history) < warmup, so clearing accepts
        `warmup` steps of ANY magnitude. Measured on this class: 200/200 steps
        accepted at gradient norm 1e6. Job 6369451 collapsed to the trivial
        solution 104 steps into that window; jobs 6238082 and 6362657, which
        never called reset, deadlocked but kept a healthy model. Every collapse
        in this project followed a re-calibration.

        Passing `scale` seeds a full window at that value, so the guard is armed
        on the very next step at a threshold of factor*scale. Passing nothing
        keeps the old clear-and-rewarm behaviour and is only safe when the model
        has just been restored to a known-good state.
        """
        self.consecutive = 0
        self.history.clear()
        if scale is not None and math.isfinite(scale) and scale > 0:
            self.history.extend([float(scale)] * self.window)

    def scale(self):
        """Current healthy gradient scale (median of accepted steps), or None."""
        return statistics.median(self.history) if self.history else None

    def exhausted(self) -> bool:
        """True when skipping has stopped being a rescue and become a stall.

        Consecutive skips mean every step looks anomalous against a history
        that is no longer representative — the run is not recoverable by
        skipping and should fall through to DivergenceGuard.
        """
        return self.consecutive >= self.max_consecutive


def build_spike_guard(train_cfg: dict):
    """SpikeGuard from a config's `spike` block; None when disabled.

    Off by default: it changes the optimizer trajectory when it fires, so a
    study that has already trained arms without it must opt in deliberately
    rather than inherit it from a config bump.
    """
    s = train_cfg.get("spike", {})
    if not s.get("enabled", False):
        return None
    return SpikeGuard(factor=s.get("factor", 50.0),
                      window=s.get("window", 200),
                      warmup=s.get("warmup", 200),
                      max_consecutive=s.get("max_consecutive", 20))


def set_seed(seed: int) -> None:
    import torch  # noqa: PLC0415 - see the module-scope note
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    import torch  # noqa: PLC0415 - see the module-scope note
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


_VAR_SHORT = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "msl",
    "surface_pressure": "sp",
    "total_column_water_vapour": "tcwv",
    "geopotential": "z",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "specific_humidity": "q",
    "vertical_velocity": "w",
}


def channel_specs(dcfg: dict) -> list[dict]:
    """Per-channel {name, level} specs, in channel order.

    Multi-channel configs list them under data.variables; a legacy config's
    single data.variable/level becomes a one-element list."""
    if dcfg.get("variables"):
        return [{"name": v["name"], "level": v.get("level")} for v in dcfg["variables"]]
    return [{"name": dcfg["variable"], "level": dcfg.get("level")}]


def channel_label(name: str, level=None) -> str:
    """Short channel label, e.g. 2m_temperature -> t2m, geopotential@500 -> z500."""
    short = _VAR_SHORT.get(name, name)
    return f"{short}{int(level)}" if level is not None else short


def channel_labels(dcfg: dict) -> list[str]:
    return [channel_label(s["name"], s["level"]) for s in channel_specs(dcfg)]


def display_channel(cfg: dict) -> int:
    """Channel index used for figures and headline (physical-unit) metrics."""
    return int(cfg.get("eval", {}).get("display_channel", 0))


# Checkpoint-name tag per geo encoder ("" for the default hash grid, so
# existing checkpoint names like diffusion_geo.pt / diffusion_geo_hpx.pt are
# unchanged).
_ENCODER_TAG = {"hash": "", "healpix": "_hpx", "hash2d": "_hash2d",
                "hash_compact": "_hashcompact",
                "hash_compact_static": "_compactcombo",
                "xyz": "_xyz", "sinusoidal": "_sin", "static": "_static",
                "hash_static": "_combo",

    "xyz_static": "_xyzstatic",
    "sinusoidal_static": "_sinstatic",
}


def geo_suffix(cfg: dict) -> str:
    """Checkpoint-name suffix identifying the geo conditioning: '' when geo is
    disabled, else '_geo' + the encoder tag (e.g. '_geo', '_geo_hpx',
    '_geo_static'), plus '_gated' when noise-dependent level gating is on."""
    g = cfg.get("geo", {})
    if not g.get("enabled", False):
        return ""
    encoder = g.get("encoder", "hash")
    if encoder not in _ENCODER_TAG:
        raise ValueError(f"unknown geo encoder: {encoder}")
    suffix = "_geo" + _ENCODER_TAG[encoder]
    if g.get("level_gating", False):
        suffix += "_gated"
    return suffix


def _var_tag(dcfg: dict) -> str:
    """Short dataset tag: channel label for single-variable runs, data.name
    (or '<C>ch') for multi-channel runs."""
    specs = channel_specs(dcfg) if (dcfg.get("variables") or dcfg.get("variable")) else []
    if len(specs) > 1:
        return dcfg.get("name") or f"{len(specs)}ch"
    if specs:
        return channel_label(specs[0]["name"], specs[0]["level"])
    return ""


def run_name(cfg: dict, *parts: str) -> str:
    """Canonical wandb run name: short variable + identity parts.

    Callers pass the checkpoint stem (which already encodes model kind, geo,
    encoder, mean type, and seed) plus any extra tags; empty parts are
    skipped. Example: run_name(cfg, 'diffusion_geo_hpx', 'resumed')
    -> 't2m-diffusion_geo_hpx-resumed'."""
    return "-".join(p for p in (_var_tag(cfg.get("data", {})), *parts) if p)


def init_wandb(cfg: dict, job_type: str, extra_config: dict | None = None,
               name: str | None = None):
    """Start a wandb run when cfg['wandb'].enabled is true.

    Opt-in: returns (None, None) when disabled so callers can guard with
    `if run is not None`. Returns (run, wandb_module) when enabled — the module
    is handed back so callers can build wandb.Image() etc. without re-importing.
    Name precedence: explicit wandb.name in the config > `name` argument >
    auto default (variable-geo-job)."""
    wcfg = cfg.get("wandb", {})
    if not wcfg.get("enabled"):
        return None, None
    import wandb
    config = {**cfg, **(extra_config or {})}
    name = wcfg.get("name") or name
    if not name and "data" in cfg:
        geo_tag = "geo" if cfg.get("geo", {}).get("enabled") else "base"
        name = f"{_var_tag(cfg['data'])}-{geo_tag}-{job_type}"
    run = wandb.init(
        project=wcfg.get("project", "era5-diffusion-downscaling"),
        entity=wcfg.get("entity"),
        name=name,
        job_type=job_type,
        config=config,
    )
    return run, wandb
