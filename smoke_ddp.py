"""Multi-rank smoke: catches collectives placed inside rank-0-only blocks.

Job 6236360 died as an NCCL ALLREDUCE timeout -- ranks 1-3 stuck 600 s at one
sequence number while rank 0 sat in a broadcast_flag that only it reached,
because the guard had been written inside `if val_x is not None:` and val_x is
built on rank 0 alone. A single-process smoke test CANNOT see this: with one
rank every collective trivially completes.

So this runs the trainers under torchrun with 2 ranks on CPU/gloo. Any rank
divergence shows up as a hang, which the timeout converts into a failure.
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
directmap: {{train_ratio: 4, batch_size: 2, epochs: {epochs}, lr: 1.0e-4,
             grad_clip: 1.0, amp: false, amp_dtype: off}}
train: {{batch_size: 2, epochs: {epochs}, lr: 1.0e-4, weight_decay: 0.0, grad_clip: 1.0,
         amp: false, amp_dtype: off, ema_decay: 0.9, num_workers: 0, log_every: 5,
         val_patches: 8, ckpt_every_epochs: 1, sample_every_epochs: 99,
         divergence: {{enabled: true, factor: {df}, patience: 1, min_epochs: {dm},
                       rollbacks: {rb}}},
         spike: {{enabled: false, factor: 50.0, window: 8, warmup: 200,
                  max_consecutive: 20}}}}
sample: {{ddim_eta: 0.0, guidance_strength: 0.0, interp: nearest,
          reconstructions: [{{ratio: 4, K: 1, t_steps: [5], smooth_sigma: 0.0}}]}}
eval: {{n_test_patches: 8, display_channel: 0, spectrum_bins: null, hist_bins: 10}}
wandb: {{enabled: false}}
"""


def run(tag, module, epochs, df, dm, rb, extra=(), timeout=900):
    with tempfile.TemporaryDirectory() as d:
        dd = Path(d); (dd / "patches").mkdir(parents=True)
        rng = np.random.default_rng(0)
        for split, n in (("train", 80), ("test", 16)):
            np.save(dd / "patches" / f"{split}_patches.npy",
                    rng.standard_normal((n, 1, 16, 16)).astype("float32"))
        np.savez(dd / "patches" / "norm_stats.npz",
                 mean=np.float32(0.0), std=np.float32(1.0), size=16)
        cfg = dd / "cfg.yaml"
        cfg.write_text(CFG.format(d=d, epochs=epochs, df=df, dm=dm, rb=rb))
        cmd = [str(PY), "-m", "torch.distributed.run", "--nproc_per_node=2",
               "--master_port=29677", "-m", module, "--config", str(cfg), *extra]
        env = {"OMP_NUM_THREADS": "1", "PATH": "/usr/bin:/bin",
               "TORCH_DISTRIBUTED_DEFAULT_BACKEND": "gloo",
               "MASTER_ADDR": "127.0.0.1", "GLOO_SOCKET_IFNAME": "lo",
               "HOME": str(Path.home()), "CUDA_VISIBLE_DEVICES": ""}
        try:
            r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                               timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            print(f"--- {tag}: HANG (deadlock) after {timeout}s"); return False
        ok = r.returncode == 0
        print(f"--- {tag}: rc={r.returncode} {'OK' if ok else 'FAIL'}")
        if not ok:
            print(r.stdout[-1200:]); print("STDERR:", r.stderr[-1200:])
        return ok


ok = True
ok &= run("directmap 2-rank, guard quiet", "train.train_directmap",
          epochs=3, df=10.0, dm=5, rb=0, extra=("--random-ratio",))
ok &= run("directmap 2-rank, guard FIRES + rollback", "train.train_directmap",
          epochs=6, df=0.5, dm=0, rb=3, extra=("--random-ratio",))
ok &= run("directmap 2-rank, guard FIRES + stop", "train.train_directmap",
          epochs=6, df=0.5, dm=0, rb=0, extra=("--random-ratio",))
# The residual trainer has the same rank-0-only val block, so the same trap.
ok &= run("residual 2-rank, guard quiet", "train.train_residual",
          epochs=3, df=10.0, dm=5, rb=0)
ok &= run("residual 2-rank, guard FIRES + rollback", "train.train_residual",
          epochs=6, df=0.5, dm=0, rb=3)
ok &= run("residual 2-rank, guard FIRES + stop", "train.train_residual",
          epochs=6, df=0.5, dm=0, rb=0)
# train_transport backs BOTH flow matching and the stochastic interpolant, and
# its val_loader is likewise rank-0 only.
ok &= run("SI 2-rank, guard quiet", "train.train_stochastic_interpolant",
          epochs=3, df=10.0, dm=5, rb=0)
ok &= run("SI 2-rank, guard FIRES + rollback", "train.train_stochastic_interpolant",
          epochs=6, df=0.5, dm=0, rb=3)
ok &= run("flow 2-rank, guard quiet", "train.train_flow_matching",
          epochs=3, df=10.0, dm=5, rb=0)
print("\n" + ("ALL DDP SMOKE CHECKS PASSED" if ok else "DDP SMOKE FAILED"))
sys.exit(0 if ok else 1)
