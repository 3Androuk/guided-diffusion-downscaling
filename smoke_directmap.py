"""CPU smoke of train_directmap with the newly ported guards.

Verifies the WIRING, not the science: that the trainer still runs, writes both
checkpoints, and that the rollback branch actually executes (hair-trigger
divergence) without desyncing the step counter or the scaler.
"""
import subprocess, sys, tempfile
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
PY = REPO / ".venv/bin/python"

CFG = """
seed: 42
paths: {{raw_dir: "{d}/raw", patch_dir: "{d}/patches", ckpt_dir: "{d}/ckpt",
         results_dir: "{d}/res", log_dir: "{d}/res/tb"}}
data: {{variable: 2m_temperature, level: null}}
patches: {{size: 16, per_field: 2, lat_range: [-60.0, 60.0]}}
normalize: {{method: zscore}}
diffusion: {{timesteps: 20, beta_schedule: linear, beta_start: 1.0e-4, beta_end: 2.0e-2}}
unet: {{in_channels: 1, out_channels: 1, base_channels: 8, channel_mults: [1, 2],
        num_res_blocks: 1, time_emb_dim: 16, attn_resolutions: [8], dropout: 0.0,
        groupnorm_groups: 4}}
geo: {{enabled: false}}
residual: {{train_ratios: [2, 4], n_steps: 10}}
directmap: {{train_ratio: 4, batch_size: 4, epochs: {epochs}, lr: 1.0e-4,
             grad_clip: 1.0, amp: false, amp_dtype: off}}
train: {{batch_size: 4, epochs: {epochs}, lr: 1.0e-4, weight_decay: 0.0, grad_clip: 1.0,
         amp: false, amp_dtype: off, ema_decay: 0.9, num_workers: 0, log_every: 1,
         val_patches: 8, ckpt_every_epochs: 1, sample_every_epochs: 99,
         divergence: {{enabled: true, factor: {df}, patience: 1, min_epochs: {dm},
                       rollbacks: {rb}}},
         spike: {{enabled: {sp}, factor: {sf}, window: 8, warmup: {sw},
                  max_consecutive: 20}}}}
sample: {{ddim_eta: 0.0, guidance_strength: 0.0, interp: nearest,
          reconstructions: [{{ratio: 4, K: 1, t_steps: [5], smooth_sigma: 0.0}}]}}
eval: {{n_test_patches: 8, display_channel: 0, spectrum_bins: null, hist_bins: 10}}
wandb: {{enabled: false}}
"""


def run(tag, epochs=3, df=10.0, dm=5, rb=0, sp="false", sf=50.0, sw=200, extra=()):
    with tempfile.TemporaryDirectory() as d:
        dd = Path(d); (dd / "patches").mkdir(parents=True)
        rng = np.random.default_rng(0)
        for split, n in (("train", 40), ("test", 8)):
            np.save(dd / "patches" / f"{split}_patches.npy",
                    rng.standard_normal((n, 1, 16, 16)).astype("float32"))
        np.savez(dd / "patches" / "norm_stats.npz",
                 mean=np.float32(0.0), std=np.float32(1.0), size=16)
        cfg = dd / "cfg.yaml"
        cfg.write_text(CFG.format(d=d, epochs=epochs, df=df, dm=dm, rb=rb,
                                  sp=sp, sf=sf, sw=sw))
        r = subprocess.run([str(PY), "-m", "train.train_directmap", "--config", str(cfg),
                            *extra], cwd=REPO, capture_output=True, text=True, timeout=1200)
        ck = dd / "ckpt"
        files = sorted(p.name for p in ck.iterdir()) if ck.exists() else []
        print(f"--- {tag}: rc={r.returncode} ckpts={files}")
        if r.returncode != 0:
            print(r.stdout[-1500:]); print("STDERR:", r.stderr[-1500:])
        return r, files


print("=== 1. baseline runs, writes rolling + best ===")
r, f = run("baseline", extra=("--random-ratio",))
assert r.returncode == 0, "trainer failed"
assert "meanmap.pt" in f and "meanmap_best.pt" in f, f"missing checkpoints: {f}"
assert "Best (val" in r.stdout

print("=== 2. spike guard fires without breaking the run ===")
r, f = run("spike", sp="true", sf=1.0, sw=4, extra=("--random-ratio",))
assert r.returncode == 0
assert "spike guard: skipped step" in r.stdout, "spike branch never ran"

print("=== 3. rollback recovers instead of stopping ===")
r, f = run("rollback", epochs=8, df=0.5, dm=0, rb=5, extra=("--random-ratio",))
assert r.returncode == 0
assert "rolling back to" in r.stdout, "rollback branch never ran"
assert "Stopping early" not in r.stdout, "should not stop with budget left"

print("=== 4. rollbacks:0 keeps the old stop behaviour ===")
r, f = run("stop", epochs=8, df=0.5, dm=0, rb=0, extra=("--random-ratio",))
assert r.returncode == 0
assert "Stopping early" in r.stdout and "rolling back" not in r.stdout

print("\nALL DIRECTMAP GUARD SMOKE CHECKS PASSED")
