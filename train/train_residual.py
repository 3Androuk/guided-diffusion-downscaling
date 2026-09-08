"""Train the conditional RESIDUAL diffusion model (split-model Phase A).

Deterministic mean = bicubic upsampling of the coarse field (no training);
the diffusion learns the residual HF - bicubic, conditioned on the bicubic
field (+ geo embedding with --geo). The degradation ratio is RANDOMIZED per
batch over residual.train_ratios, so one model serves all ratios — test at a
held-out ratio (e.g. 6) for the generalization claim.

Run:
    python -m train.train_residual --config config/t2m.yaml --wandb [--geo] [--seed N]

Optionally, the same run can be split across several GPUs/nodes (e.g. 4 nodes)
by launching under torchrun — see train/distributed.py and
scripts/train_multinode.sh. Single-process behavior is unchanged.
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import PatchDataset, load_norm_stats  # noqa: E402
from data.degrade import coarsen, degrade as degrade_nearest  # noqa: E402
from eval.metrics import spectrum_log_l1  # noqa: E402
from models.diffusion import build_diffusion  # noqa: E402
from models.residual import build_residual_model, res_scale  # noqa: E402
from train.distributed import (barrier, broadcast_flag, cleanup,  # noqa: E402
                               init_distributed,
                               make_train_loader, set_epoch, wrap_model)
from train.ema import EMA  # noqa: E402
from train.guards import restore_best, spike_skip  # noqa: E402
from utils import (add_perf_args, apply_perf_overrides,  # noqa: E402
                   build_divergence_guard, build_spike_guard, resolve_amp,
                   display_channel, ensure_dir, geo_suffix, init_wandb,
                   load_config, run_name, set_seed)



def _centre_crop(y, coords, size):
    """Centre-crop patch and coords to `size`. No-op when size is falsy."""
    if not size or y.shape[-1] == size:
        return y, coords
    o = (y.shape[-1] - size) // 2
    p = (y.shape[-2] - size) // 2
    y = y[..., p:p + size, o:o + size]
    if coords is not None:
        # coords are (N, H, W, d) -- crop the spatial axes, not the feature axis
        coords = coords[:, p:p + size, o:o + size, :]
    return y, coords

def _bicubic_mean(y: torch.Tensor, ratio: int) -> torch.Tensor:
    """Phase-A deterministic mean: bicubic upsample of the coarse field."""
    lo = coarsen(y, ratio)
    return F.interpolate(lo, size=y.shape[-2:], mode="bicubic", align_corners=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=None,
                    help="centre-crop each patch to this size before degrading. "
                         "Needed for ratios that do not divide the stored patch: "
                         "128 %% 6 == 2, so ratio 6 requires --crop 120 (=6*20=8*15). "
                         "Default None = no crop, existing runs unchanged.")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--wandb", action="store_true",
                    help="Enable wandb logging (overrides config wandb.enabled).")
    ap.add_argument("--per-channel-res-std", dest="per_channel_res_std",
                    action="store_true",
                    help="scale the residual PER CHANNEL instead of by one global "
                         "scalar. DDPM samples x0 from unit-variance noise, so a "
                         "channel whose true residual is far below the global scalar "
                         "gets its signal buried and comes back over-energised; "
                         "measured Spearman -0.73 between s_c/res_std and the damage. "
                         "Exact and irrelevant for a 1-channel model, essential for 20.")
    ap.add_argument("--ratios", type=int, nargs="+", default=None,
                    help="override residual.train_ratios. Omitted = read the "
                         "config, so existing runs are unchanged. Use --ratios 6 "
                         "to train AT the deployment ratio.")
    ap.add_argument("--tag", default="",
                    help="suffix for the checkpoint name, so a variant does not "
                         "overwrite the arm it is being compared against.")
    ap.add_argument("--lr", type=float, default=None,
                    help="override train.lr for THIS arm only. At the ladder's 1e-4 the "
                         "residual arm collapses to the trivial solution from epoch ~33 "
                         "(4 times in 51 epochs, job 6372044); the config's own note "
                         "records 2e-4 doing the same by epoch ~2. Breaks the ladder "
                         "constant -- say so in --tag.")
    ap.add_argument("--beta2", type=float, default=None,
                    help="override AdamW beta2 (PyTorch default 0.999). Adam's worst-case "
                         "step ratio scales ~1/sqrt(1-beta2): 32x at 0.999, 10x at 0.99. "
                         "Targets the instability without touching lr.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from the residual checkpoint if it exists.")
    ap.add_argument("--geo", action="store_true",
                    help="Condition the residual model on the hash-grid location "
                         "embedding as well.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Override config seed; suffixes the checkpoint name.")
    ap.add_argument("--encoder", choices=["hash", "healpix", "xyz", "sinusoidal", "static", "hash_static", "xyz_static", "sinusoidal_static"], default=None,
                    help="Override geo.encoder from the CLI so the config can "
                         "keep its default.")
    ap.add_argument("--gated", action="store_true",
                    help="Force geo.level_gating: true — noise-dependent gating "
                         "of the embedding levels; checkpoint gains _gated.")
    ap.add_argument("--mean-ckpt", default=None,
                    help="Checkpoint name (in paths.ckpt_dir) of a frozen learned "
                         "regression mean (train_directmap --random-ratio -> "
                         "meanmap*.pt) to use instead of bicubic; the residual "
                         "checkpoint gets an _lm suffix and remembers the mean.")
    add_perf_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.wandb:
        cfg.setdefault("wandb", {})["enabled"] = True
    if args.geo:
        cfg.setdefault("geo", {})["enabled"] = True
    if args.encoder is not None:
        cfg.setdefault("geo", {})["encoder"] = args.encoder
    if args.gated:
        cfg.setdefault("geo", {})["level_gating"] = True
    if args.seed is not None:
        cfg["seed"] = args.seed
    apply_perf_overrides(cfg, args, "train")
    set_seed(cfg["seed"])
    dist = init_distributed()  # no-op unless launched under torchrun
    device = dist.device

    tc = cfg["train"]
    rcfg = cfg.get("residual", {})
    ratios = args.ratios or rcfg.get("train_ratios", [2, 4, 8])
    patch_dir = Path(cfg["paths"]["patch_dir"])
    ckpt_dir = ensure_dir(cfg["paths"]["ckpt_dir"])
    results_dir = ensure_dir(cfg["paths"]["results_dir"])

    geo_on = cfg.get("geo", {}).get("enabled", False)
    seed_suffix = f"_s{cfg['seed']}" if args.seed is not None else ""
    lm = "_lm" if args.mean_ckpt else ""
    tag = f"_{args.tag}" if args.tag else ""
    ckpt_name = f"residual{geo_suffix(cfg)}{lm}{tag}{seed_suffix}.pt"

    # ── The deterministic mean: bicubic (Phase A) or a frozen learned
    # regression (Phase B, --mean-ckpt). The mean may be geo-conditioned even
    # when the residual model is not, in which case coords are still needed.
    device_early = device
    mean_geo = False
    if args.mean_ckpt:
        from sample.reconstruct import load_directmap
        mean_model, mean_cfg = load_directmap(ckpt_dir / args.mean_ckpt, device_early)
        for p in mean_model.parameters():
            p.requires_grad_(False)
        mean_geo = mean_cfg.get("geo", {}).get("enabled", False)
        print(f"Learned mean: {args.mean_ckpt} (geo={mean_geo}, frozen)")

        def mean_fn(y, r, coords=None):
            x = degrade_nearest(y, r)
            with torch.no_grad():
                return mean_model(x, None, coords) if mean_geo else mean_model(x)
    else:
        def mean_fn(y, r, coords=None):
            return _bicubic_mean(y, r)

    need_coords = geo_on or mean_geo
    if args.mean_ckpt and mean_geo and geo_on:
        assert (mean_cfg["geo"].get("encoder", "hash")
                == cfg["geo"].get("encoder", "hash")), \
            "mean and residual geo encoders must match (they share one coords payload)"

    normalizer = load_norm_stats(patch_dir)
    gkw = {}
    if need_coords:
        gcfg = (cfg if geo_on else mean_cfg)["geo"]
        gkw = dict(origins_path=patch_dir / "train_origins.npy",
                   coords_full_path=patch_dir / "coords_full.npz",
                   geo_input_dim=gcfg["input_dim"], altitude=gcfg["altitude"],
                   geo_encoder=gcfg.get("encoder", "hash"),
                   healpix_index_path=((patch_dir / gcfg["healpix_index"])
                                       if gcfg.get("healpix_index") else None))
    ds = PatchDataset(patch_dir / "train_patches.npy", normalizer, **gkw)
    loader = make_train_loader(ds, tc["batch_size"], tc["num_workers"], dist,
                               seed=cfg["seed"])
    print(f"Residual diffusion | ratios {ratios} | patches {len(ds)} | geo={geo_on}")

    # ── Residual normalization: one scalar std over ratios (estimated once
    # against the ACTUAL mean in use). The mean-field channel tells the model
    # which regime it is in, so a shared scale is sufficient; the value is
    # stored in the checkpoint.
    with torch.no_grad():
        chunks = []
        # min(): a hardcoded 64 raises IndexError on any dataset smaller
        # than that, which is every synthetic test fixture.
        items = [ds[i] for i in range(min(64, len(ds)))]
        if need_coords:
            y64 = torch.stack([it[0] for it in items]).to(device)
            c64 = torch.stack([it[1] for it in items]).to(device)
        else:
            y64, c64 = torch.stack(items).to(device), None
        y64, c64 = _centre_crop(y64, c64, args.crop)
        for r in ratios:
            chunks.append((y64 - mean_fn(y64, r, c64)).cpu())
        if args.per_channel_res_std:
            # (C,) -- every channel's x0 becomes unit-variance, which is what
            # the diffusion forward/reverse process assumes.
            res_std = torch.cat(chunks, dim=0).std(dim=(0, 2, 3)).tolist()
        else:
            res_std = float(torch.cat([c.flatten() for c in chunks]).std())
        del y64, c64
    if isinstance(res_std, float):
        print(f"Residual std (normalized units): {res_std:.4f}  [ONE global scalar]")
    else:
        print(f"Residual std per channel (normalized units): "
              f"min {min(res_std):.4f} max {max(res_std):.4f} "
              f"ratio {max(res_std)/min(res_std):.1f}x")
        print("  " + " ".join(f"{v:.4f}" for v in res_std))

    # ── Val: fixed-RNG residual noise-prediction loss at the middle ratio.
    # Rank 0 only under DDP: weights are identical on every rank, so one
    # process scoring the full val set reproduces single-process values.
    val_loader, val_ratio = None, ratios[len(ratios) // 2]
    test_path = patch_dir / "test_patches.npy"
    if dist.is_main and test_path.exists():
        vkw = dict(gkw)
        if need_coords:
            vkw["origins_path"] = patch_dir / "test_origins.npy"
        val_ds = PatchDataset(test_path, normalizer, **vkw)
        n_val = min(int(tc.get("val_patches", 256)), len(val_ds))
        # Spread over the WHOLE test split: patches are time-ordered at 8
        # per field, so range(n_val) was ~32 consecutive January days.
        val_idx = np.linspace(0, len(val_ds) - 1, n_val).astype(int).tolist()
        val_loader = DataLoader(Subset(val_ds, val_idx),
                                batch_size=tc["batch_size"], shuffle=False, num_workers=0)
        print(f"Val patches: {n_val} (ratio {val_ratio}x)")

    model = build_residual_model(cfg).to(device)
    diffusion = build_diffusion(cfg).to(device)
    ema = EMA(model, decay=tc["ema_decay"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet params: {n_params:,}")

    lr_eff = args.lr if args.lr is not None else tc["lr"]
    beta2_eff = args.beta2 if args.beta2 is not None else 0.999   # PyTorch default, bit-identical
    opt = torch.optim.AdamW(model.parameters(), lr=lr_eff, betas=(0.9, beta2_eff),
                            weight_decay=tc["weight_decay"])
    if args.lr is not None or args.beta2 is not None:
        # Provenance: the checkpoint saves `cfg`, so the override travels with it.
        cfg.setdefault("train_overrides", {}).update(
            {k: v for k, v in (("lr", args.lr), ("beta2", args.beta2)) if v is not None})
        print(f"OPTIMIZER OVERRIDE: lr={lr_eff:g} beta2={beta2_eff:g} "
              f"(ladder constant is lr={tc['lr']:g} beta2=0.999) -- NOT ladder-comparable",
              flush=True)
    # Collapse protection (train/guards.py). Four of six 20-var arms collapsed
    # mid-run; this trainer had none.
    spike = build_spike_guard(tc)
    guard = build_divergence_guard(tc)
    rollbacks_left = int(tc.get("divergence", {}).get("rollbacks", 0))
    reshuffle_offset = 0
    best_val = float("inf")
    use_amp, amp_dtype = resolve_amp(tc, device.type)
    # A GradScaler exists for fp16's narrow exponent range; bf16 needs none.
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and amp_dtype is torch.float16)

    start_epoch, step = 1, 0
    ckpt_path = ckpt_dir / ckpt_name
    best_ckpt_path = ckpt_path.with_name(f"{ckpt_path.stem}_best{ckpt_path.suffix}")
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        res_std = ck["res_std"]
        start_epoch = ck["epoch"] + 1
        step = ck["step"]
        print(f"Resumed from {ckpt_path} at epoch {ck['epoch']} (step {step})")
    elif args.resume:
        print(f"(no checkpoint at {ckpt_path} — starting fresh)")

    # DDP wrap AFTER resume so state loads into the raw module; raw_model stays
    # the handle for EMA/val/checkpointing (its state_dict keeps plain keys).
    raw_model = model
    model = wrap_model(model, dist, cfg)
    if dist.enabled:
        # Same weights everywhere (DDP broadcast); per-rank seed decorrelates
        # the noise/timestep draws and per-batch ratio choices.
        set_seed(cfg["seed"] + dist.rank)

    wb_run, wandb = (None, None)
    if dist.is_main:
        wb_run, wandb = init_wandb(cfg, job_type="train_residual",
                                   extra_config={"unet_params": n_params,
                                                 "n_train_patches": len(ds),
                                                 "train_ratios": ratios,
                                                 "res_std": res_std,
                                                 "mean_ckpt": args.mean_ckpt,
                                                 "world_size": dist.world_size},
                                   name=run_name(cfg, Path(ckpt_name).stem,
                                                 "resumed" if start_epoch > 1 else ""))
    if wb_run is not None:
        print(f"wandb: logging to {wb_run.url}")

    running, running_n, grad_sum = 0.0, 0, 0.0
    t_last_log = time.time()
    for epoch in range(start_epoch, tc["epochs"] + 1):
        # Offset so a post-rollback replay draws a DIFFERENT shard order.
        set_epoch(loader, epoch + 10_000 * reshuffle_offset)
        model.train()
        epoch_loss, epoch_batches = 0.0, 0
        accepted_epoch = skipped_epoch = 0   # a stall must be visible per epoch
        spike_stalled = False
        epoch_start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for batch in loader:
            if need_coords:
                y, coords = batch
                y = y.to(device, non_blocking=True)
                coords = coords.to(device, non_blocking=True)
            else:
                y, coords = batch.to(device, non_blocking=True), None
            y, coords = _centre_crop(y, coords, args.crop)
            ratio = random.choice(ratios)
            mean_f = mean_fn(y, ratio, coords)
            x0 = (y - mean_f) / res_scale(res_std, y)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                loss = diffusion.training_loss(model, x0, cond=(mean_f, coords))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                tc["grad_clip"] if tc["grad_clip"] > 0 else float("inf"))
            if spike_skip(spike, grad_norm, scaler, step, dist.is_main):
                # No opt.step() -> Adam's moments never see the bad direction;
                # no ema.update() -> EMA keeps the pre-spike weights.
                # scaler.update() is still required or the next unscale_ raises.
                scaler.update()
                step += 1
                skipped_epoch += 1
                if spike is not None and spike.exhausted():
                    # Persistent high-gradient region, not a transient spike.
                    # Do NOT re-calibrate and push through: every run that did
                    # collapsed to the trivial solution within ~100 steps
                    # (6369451, 6372044 x3). Abandon the epoch; the rollback
                    # below returns to the last good state and reshuffles.
                    spike_stalled = True
                    if dist.is_main:
                        print(f"  spike guard: {spike.consecutive} consecutive skips at "
                              f"step {step} — persistent high-gradient region, "
                              f"abandoning the epoch for a rollback", flush=True)
                    break
                continue
            accepted_epoch += 1
            scaler.step(opt)
            scaler.update()
            ema.update(raw_model)

            running += loss.item()
            running_n += 1
            grad_sum += grad_norm.item()
            epoch_loss += loss.item()
            epoch_batches += 1
            step += 1
            if step % tc["log_every"] == 0:
                now = time.time()
                metrics = {
                    "train/loss": running / running_n,
                    "train/grad_norm": grad_sum / running_n,
                    # Global throughput: ranks step in lockstep, so rank 0's
                    # window x world_size counts every rank's images.
                    "train/imgs_per_sec": (running_n * y.shape[0] * dist.world_size
                                           / (now - t_last_log)),
                    "epoch": epoch,
                }
                print(f"epoch {epoch:03d} step {step:07d} | "
                      f"loss {metrics['train/loss']:.5f} | "
                      f"grad {metrics['train/grad_norm']:.3f} | "
                      f"{metrics['train/imgs_per_sec']:.1f} img/s")
                if wb_run is not None:
                    wb_run.log(metrics, step=step)
                running, running_n, grad_sum = 0.0, 0, 0.0
                t_last_log = now

        epoch_metrics = {
            "train/epoch_loss": epoch_loss / max(epoch_batches, 1),
            "train/epoch_time_s": time.time() - epoch_start,
            "epoch": epoch,
        }
        if device.type == "cuda":
            epoch_metrics["train/gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
        # A spike stall is handled BEFORE validation: the model is still healthy
        # (every anomalous step was skipped), so there is nothing to measure and
        # nothing to salvage -- just go back and take a different path.
        if broadcast_flag(spike_stalled, dist):
            healthy = spike.scale() if spike is not None else None
            if rollbacks_left > 0 and best_ckpt_path.exists():
                rollbacks_left -= 1
                reshuffle_offset += 1
                if dist.is_main:
                    print(f"SPIKE STALL at epoch {epoch}: rolling back to "
                          f"{best_ckpt_path} (val {best_val:.5f}) and reshuffling; "
                          f"{rollbacks_left} rollback(s) left.", flush=True)
                barrier(dist)
                restore_best(best_ckpt_path, raw_model, ema, opt, scaler, device)
                set_seed(cfg["seed"] + dist.rank + 1000 * reshuffle_offset)
                # Re-arm at the pre-spike healthy scale, NOT unarmed.
                spike = build_spike_guard(tc)
                if spike is not None and healthy:
                    spike.reset(healthy)
                guard = build_divergence_guard(tc)
                if guard is not None:
                    guard.best = best_val
                continue
            if dist.is_main:
                print(f"SPIKE STALL at epoch {epoch} with no rollbacks left. "
                      f"Stopping. The best state (val {best_val:.5f}) is at "
                      f"{best_ckpt_path} and is HEALTHY -- every anomalous step "
                      f"was skipped, so this is a converged-as-far-as-it-goes "
                      f"arm, not a collapsed one.", flush=True)
            break

        if val_loader is not None:
            epoch_metrics["val/loss"] = _val_loss(
                diffusion, ema.shadow, val_loader, device, val_ratio, res_std,
                need_coords, mean_fn, args.crop)
            print(f"epoch {epoch:03d} done | val loss {epoch_metrics['val/loss']:.5f} | "
                  f"steps {accepted_epoch}+{skipped_epoch}skip | "
                  f"{epoch_metrics['train/epoch_time_s']:.0f}s")
        if wb_run is not None:
            wb_run.log(epoch_metrics, step=step)

        # ── Best checkpoint + divergence guard ─────────────────────────────
        # val_loader exists on rank 0 ONLY, so "val/loss" is absent elsewhere.
        # The SAVE is rank-0 file I/O and may be conditional; the BROADCAST
        # must not be — a collective reached by one rank hangs the others,
        # which is exactly how job 6236360 died (NCCL ALLREDUCE timeout).
        val_now = epoch_metrics.get("val/loss")
        # An epoch that skipped steps is not a trustworthy rollback target:
        # val runs on the lagging EMA, so it can still improve while the raw
        # weights are already damaged (job 6372044 epoch 42, 185+20skip).
        if (dist.is_main and val_now is not None and val_now < best_val
                and skipped_epoch == 0):
            best_val = val_now
            tmpb = best_ckpt_path.with_suffix(".pt.tmp")
            torch.save({
                "model": raw_model.state_dict(), "ema": ema.state_dict(),
                "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                "config": cfg, "res_std": res_std, "mean_ckpt": args.mean_ckpt,
                "norm_mean": normalizer.mean, "norm_std": normalizer.std,
                "epoch": epoch, "step": step,
            }, tmpb)
            tmpb.replace(best_ckpt_path)

        reason = (guard.update(val_now, epoch)
                  if guard is not None and val_now is not None else None)
        if guard is not None and broadcast_flag(reason is not None, dist):
            if rollbacks_left > 0 and best_ckpt_path.exists():
                rollbacks_left -= 1
                reshuffle_offset += 1
                if dist.is_main:
                    print(f"DIVERGED at epoch {epoch}: {reason}\n"
                          f"  rolling back to {best_ckpt_path} (val {best_val:.5f}) "
                          f"and continuing with a reshuffled order; "
                          f"{rollbacks_left} rollback(s) left.", flush=True)
                barrier(dist)
                restore_best(best_ckpt_path, raw_model, ema, opt, scaler, device)
                set_seed(cfg["seed"] + dist.rank + 1000 * reshuffle_offset)
                guard = build_divergence_guard(tc)
                if guard is not None:
                    # Seed the rebuilt guard with the restored checkpoint's val. A fresh
                    # guard starts at best=inf and takes the FIRST post-rollback val as
                    # best; if the model re-collapses immediately that val is ~1.0, the
                    # ceiling gate (best < ceiling/2) never arms, and with 0 rollbacks
                    # left the run coasts instead of stopping. Job 6372044: 110 dead epochs.
                    guard.best = best_val
                if spike is not None:
                    spike = build_spike_guard(tc)
                continue
            if dist.is_main:
                print(f"DIVERGED at epoch {epoch}: {reason}\nStopping early. The "
                      f"rolling checkpoint holds the diverged weights; the "
                      f"pre-collapse state is at {best_ckpt_path} "
                      f"(val {best_val:.5f}).", flush=True)
            break
        t_last_log = time.time()

        if epoch % tc["sample_every_epochs"] == 0 and val_loader is not None:
            fig_path = results_dir / f"residual_epoch{epoch:03d}.png"
            spec = _save_recons(diffusion, ema.shadow, val_loader.dataset, normalizer,
                                device, res_std, rcfg.get("n_steps", 100), need_coords,
                                mean_fn, fig_path, disp=display_channel(cfg), crop=args.crop)
            if wb_run is not None:
                log = {"recons": wandb.Image(str(fig_path))}
                if spec is not None:
                    log["samples/spectrum_log_l1"] = spec
                wb_run.log(log, step=step)

        if epoch % tc["ckpt_every_epochs"] == 0 or epoch == tc["epochs"]:
            # All ranks check — weights are identical, so all stop together.
            if not (_weights_finite(raw_model) and _weights_finite(ema.shadow)):
                raise RuntimeError(
                    f"non-finite weights at epoch {epoch} — training has diverged. "
                    f"Checkpoint NOT overwritten; last good state kept at {ckpt_path}.")
            if dist.is_main:
                tmp = ckpt_path.with_suffix(".pt.tmp")
                torch.save({
                    "model": raw_model.state_dict(), "ema": ema.state_dict(),
                    "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "config": cfg, "res_std": res_std, "mean_ckpt": args.mean_ckpt,
                    "norm_mean": normalizer.mean, "norm_std": normalizer.std,
                    "epoch": epoch, "step": step,
                }, tmp)
                tmp.replace(ckpt_path)

    if wb_run is not None:
        wb_run.finish()
    cleanup(dist)
    print(f"Done. Checkpoint -> {ckpt_path}")
    if best_ckpt_path.exists():
        print(f"Best (val {best_val:.5f}) -> {best_ckpt_path}")


@torch.no_grad()
def _weights_finite(model) -> bool:
    return all(p.isfinite().all() for p in model.parameters())


@torch.no_grad()
def _val_loss(diffusion, model, val_loader, device, ratio, res_std, need_coords, mean_fn, crop=None):
    """Fixed-RNG residual noise-prediction loss at one ratio (comparable
    across epochs)."""
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(0)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(0)
        for batch in val_loader:
            if need_coords:
                y, coords = batch
                y = y.to(device, non_blocking=True)
                coords = coords.to(device, non_blocking=True)
            else:
                y, coords = batch.to(device, non_blocking=True), None
            y, coords = _centre_crop(y, coords, crop)
            mean_f = mean_fn(y, ratio, coords)
            x0 = (y - mean_f) / res_scale(res_std, y)
            loss = diffusion.training_loss(model, x0, cond=(mean_f, coords))
            total += loss.item() * y.shape[0]
            n += y.shape[0]
    if was_training:
        model.train()
    return total / max(n, 1)


@torch.no_grad()
def _save_recons(diffusion, model, val_subset, normalizer, device, res_std,
                 n_steps, need_coords, mean_fn, path, disp=0, crop=None):
    """2 fixed val patches reconstructed at 4x and 8x: mean | mean+residual |
    target, shared color scale (display channel). Returns spectrum_log_l1 of
    the recons vs targets (both ratios pooled, display channel) or None."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = [val_subset[i] for i in range(2)]
    if need_coords:
        y = torch.stack([it[0] for it in items]).to(device)
        coords = torch.stack([it[1] for it in items]).to(device)
    else:
        y = torch.stack(items).to(device)
        coords = None
    y, coords = _centre_crop(y, coords, crop)

    recs, panels = [], []
    for i in range(len(y)):
        yi = y[i:i + 1]
        ci = None if coords is None else coords[i:i + 1]
        row = []
        for r in (4, 8):
            mean_f = mean_fn(yi, r, ci)
            res = diffusion.sample_unconditional(
                model, yi.shape, device, n_steps=n_steps, cond=(mean_f, ci))
            rec = mean_f + res_scale(res_std, mean_f) * res
            recs.append(rec)
            row += [(f"Mean {r}x", mean_f), (f"Recon {r}x", rec)]
        row.append(("Target", yi))
        panels.append(row)

    fig, axes = plt.subplots(len(panels), 5, figsize=(20, 4 * len(panels)))
    axes = axes.reshape(len(panels), 5)
    for r_i, row in enumerate(panels):
        ref = normalizer.decode(row[-1][1].cpu())[0, disp].numpy()
        vmin, vmax = float(ref.min()), float(ref.max())
        for ax, (title, t) in zip(axes[r_i], row):
            ax.imshow(normalizer.decode(t.cpu())[0, disp].numpy(), cmap="RdBu_r",
                      vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.axis("off")
    fig.suptitle("Residual model reconstructions (fixed val patches)")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved recons -> {path}")

    rec_phys = normalizer.decode(torch.cat(recs).cpu())
    tgt_phys = normalizer.decode(torch.cat([y, y]).cpu())
    return spectrum_log_l1(rec_phys[:, disp:disp + 1], tgt_phys[:, disp:disp + 1])


if __name__ == "__main__":
    main()
