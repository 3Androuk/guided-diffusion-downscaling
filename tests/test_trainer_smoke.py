"""End-to-end CPU smoke run of train_diffusion.

Exists because of a real failure: a NameError on `stem` in main() killed a
4-GPU job 37 seconds in. py_compile does not catch unresolved names, the
divergence-guard unit tests exercise the class in isolation, and
test_distributed.py never invokes a trainer — so nothing in the suite executed
the code path that broke.

This runs the actual module as a subprocess on a tiny synthetic dataset for a
couple of epochs. It is slower than the rest of the suite and worth it: it
catches NameErrors, bad config keys, checkpoint-path mistakes and wiring
regressions before they reach a billed GPU node.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

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
train: {{batch_size: 4, epochs: {epochs}, lr: 1.0e-4, weight_decay: 0.0, grad_clip: 1.0,
         amp: false, amp_dtype: off, ema_decay: 0.9, num_workers: 0, log_every: 1,
         val_patches: 4, ckpt_every_epochs: 1, sample_every_epochs: 99,
         divergence: {{enabled: true, factor: {div_factor}, patience: 1,
                      min_epochs: {div_min_epochs}, rollbacks: {rollbacks}}},
         spike: {{enabled: {spike}, factor: {spike_factor}, window: 8,
                  warmup: {spike_warmup}, max_consecutive: 20}}}}
sample: {{ddim_eta: 0.0, guidance_strength: 0.0, interp: nearest,
          reconstructions: [{{ratio: 4, K: 1, t_steps: [5], smooth_sigma: 0.0}}]}}
eval: {{n_test_patches: 4, display_channel: 0, spectrum_bins: null, hist_bins: 10}}
wandb: {{enabled: false}}
"""


def _cfg(d, spike="false", spike_factor=50.0, spike_warmup=200,
         div_factor=10.0, div_min_epochs=5, rollbacks=0, epochs=2):
    """Render CFG. Guards inert by default so existing cases are unchanged."""
    return CFG.format(d=d, spike=spike, spike_factor=spike_factor,
                      spike_warmup=spike_warmup, div_factor=div_factor,
                      div_min_epochs=div_min_epochs, rollbacks=rollbacks,
                      epochs=epochs)


class TrainerSmokeTest(unittest.TestCase):
    def test_train_diffusion_runs_and_writes_both_checkpoints(self):
        with tempfile.TemporaryDirectory() as d:
            dd = Path(d)
            (dd / "patches").mkdir(parents=True)
            rng = np.random.default_rng(0)
            for split, n in (("train", 8), ("test", 4)):
                np.save(dd / "patches" / f"{split}_patches.npy",
                        rng.standard_normal((n, 1, 16, 16)).astype("float32"))
            np.savez(dd / "patches" / "norm_stats.npz",
                     mean=np.float32(0.0), std=np.float32(1.0), size=16)
            cfg = dd / "cfg.yaml"
            cfg.write_text(_cfg(d))

            r = subprocess.run(
                [sys.executable, "-m", "train.train_diffusion", "--config", str(cfg)],
                cwd=REPO, capture_output=True, text=True, timeout=900)
            self.assertEqual(r.returncode, 0,
                             f"trainer failed\nSTDOUT:\n{r.stdout[-2500:]}"
                             f"\nSTDERR:\n{r.stderr[-2500:]}")

            ck = dd / "ckpt"
            rolling = ck / "diffusion.pt"
            best = ck / "diffusion_best.pt"
            self.assertTrue(rolling.exists(), f"no rolling checkpoint; got {list(ck.iterdir())}")
            self.assertTrue(best.exists(), f"no best checkpoint; got {list(ck.iterdir())}")
            self.assertIn("Best (val", r.stdout)

    def test_resume_from_the_rolling_checkpoint(self):
        """--resume is passed unconditionally by scripts/isambard_train.sbatch."""
        with tempfile.TemporaryDirectory() as d:
            dd = Path(d)
            (dd / "patches").mkdir(parents=True)
            rng = np.random.default_rng(1)
            for split, n in (("train", 8), ("test", 4)):
                np.save(dd / "patches" / f"{split}_patches.npy",
                        rng.standard_normal((n, 1, 16, 16)).astype("float32"))
            np.savez(dd / "patches" / "norm_stats.npz",
                     mean=np.float32(0.0), std=np.float32(1.0), size=16)
            cfg = dd / "cfg.yaml"
            cfg.write_text(_cfg(d))
            base = [sys.executable, "-m", "train.train_diffusion", "--config", str(cfg)]

            first = subprocess.run(base, cwd=REPO, capture_output=True, text=True, timeout=900)
            self.assertEqual(first.returncode, 0, first.stderr[-2000:])
            # ... and again with --resume, which must not crash on an existing ckpt
            second = subprocess.run(base + ["--resume"], cwd=REPO,
                                    capture_output=True, text=True, timeout=900)
            self.assertEqual(second.returncode, 0,
                             f"--resume failed\nSTDERR:\n{second.stderr[-2500:]}")

    def test_spike_guard_skips_steps_without_breaking_the_run(self):
        """Exercise the skip branch in the real training loop, not in isolation.

        tests/test_spike_guard.py checks the decision logic; this checks the
        wiring around it — that `continue` does not desync the step counter or
        the scaler, that a skipped step still leaves a trainable run, and that
        the trainer exits 0 with checkpoints written. factor 1.0 makes roughly
        every above-median step fire, which is nothing like production settings
        and exactly what makes it a useful stress of the branch.
        """
        with tempfile.TemporaryDirectory() as d:
            dd = Path(d)
            (dd / "patches").mkdir(parents=True)
            rng = np.random.default_rng(3)
            for split, n in (("train", 40), ("test", 4)):
                np.save(dd / "patches" / f"{split}_patches.npy",
                        rng.standard_normal((n, 1, 16, 16)).astype("float32"))
            np.savez(dd / "patches" / "norm_stats.npz",
                     mean=np.float32(0.0), std=np.float32(1.0), size=16)
            cfg = dd / "cfg.yaml"
            cfg.write_text(_cfg(d, spike="true", spike_factor=1.0, spike_warmup=4))

            r = subprocess.run(
                [sys.executable, "-m", "train.train_diffusion", "--config", str(cfg)],
                cwd=REPO, capture_output=True, text=True, timeout=900)
            self.assertEqual(r.returncode, 0,
                             f"spike guard broke the trainer\nSTDOUT:\n{r.stdout[-2500:]}"
                             f"\nSTDERR:\n{r.stderr[-2500:]}")
            self.assertIn("spike guard: skipped step", r.stdout,
                          "guard never fired, so the branch was not exercised")
            self.assertTrue((dd / "ckpt" / "diffusion.pt").exists())

    def test_spike_guard_off_by_default_leaves_no_trace(self):
        """The completed arms trained without it; the default must stay inert."""
        with tempfile.TemporaryDirectory() as d:
            dd = Path(d)
            (dd / "patches").mkdir(parents=True)
            rng = np.random.default_rng(4)
            for split, n in (("train", 40), ("test", 4)):
                np.save(dd / "patches" / f"{split}_patches.npy",
                        rng.standard_normal((n, 1, 16, 16)).astype("float32"))
            np.savez(dd / "patches" / "norm_stats.npz",
                     mean=np.float32(0.0), std=np.float32(1.0), size=16)
            cfg = dd / "cfg.yaml"
            cfg.write_text(_cfg(d))          # spike: enabled false
            r = subprocess.run(
                [sys.executable, "-m", "train.train_diffusion", "--config", str(cfg)],
                cwd=REPO, capture_output=True, text=True, timeout=900)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            self.assertNotIn("spike guard", r.stdout)

    def _tiny_dataset(self, dd, seed, n_train=40):
        (dd / "patches").mkdir(parents=True)
        rng = np.random.default_rng(seed)
        for split, n in (("train", n_train), ("test", 4)):
            np.save(dd / "patches" / f"{split}_patches.npy",
                    rng.standard_normal((n, 1, 16, 16)).astype("float32"))
        np.savez(dd / "patches" / "norm_stats.npz",
                 mean=np.float32(0.0), std=np.float32(1.0), size=16)

    def test_divergence_rollback_continues_instead_of_stopping(self):
        """The second line of defence: recover from a collapse without a resubmit.

        factor 0.5 makes the trigger DETERMINISTIC: it fires whenever val
        exceeds HALF the best, which is every epoch after the first. (A first
        version used factor 1.0 — "any epoch that is not a new best" — and
        flaked, because on this tiny problem val improves monotonically for all
        8 epochs and the guard never fired at all.) What matters is the wiring:
        the best checkpoint reloads, training carries on past the trigger
        instead of breaking out, and the process still exits 0 with a
        checkpoint on disk. Without `rollbacks` the same config stops at the
        first trigger — the next test pins that contrast.
        """
        with tempfile.TemporaryDirectory() as d:
            dd = Path(d)
            self._tiny_dataset(dd, 5)
            cfg = dd / "cfg.yaml"
            cfg.write_text(_cfg(d, div_factor=0.5, div_min_epochs=0,
                                rollbacks=5, epochs=8))
            r = subprocess.run(
                [sys.executable, "-m", "train.train_diffusion", "--config", str(cfg)],
                cwd=REPO, capture_output=True, text=True, timeout=900)
            self.assertEqual(r.returncode, 0,
                             f"rollback broke the trainer\nSTDOUT:\n{r.stdout[-3000:]}"
                             f"\nSTDERR:\n{r.stderr[-3000:]}")
            self.assertIn("rolling back to", r.stdout,
                          "the rollback branch never ran")
            self.assertNotIn("Stopping early", r.stdout,
                             "budget was not exhausted, so it must not stop")
            self.assertTrue((dd / "ckpt" / "diffusion.pt").exists())

    def test_rollbacks_zero_keeps_the_old_stop_behaviour(self):
        """Default is 0, so runs that predate this keep stopping on divergence."""
        with tempfile.TemporaryDirectory() as d:
            dd = Path(d)
            self._tiny_dataset(dd, 6)
            cfg = dd / "cfg.yaml"
            cfg.write_text(_cfg(d, div_factor=0.5, div_min_epochs=0,
                                rollbacks=0, epochs=8))
            r = subprocess.run(
                [sys.executable, "-m", "train.train_diffusion", "--config", str(cfg)],
                cwd=REPO, capture_output=True, text=True, timeout=900)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            self.assertIn("Stopping early", r.stdout)
            self.assertNotIn("rolling back", r.stdout)

    def test_resume_on_a_fresh_run_is_not_an_error(self):
        """The launcher always passes --resume, including the very first run."""
        with tempfile.TemporaryDirectory() as d:
            dd = Path(d)
            (dd / "patches").mkdir(parents=True)
            rng = np.random.default_rng(2)
            for split, n in (("train", 8), ("test", 4)):
                np.save(dd / "patches" / f"{split}_patches.npy",
                        rng.standard_normal((n, 1, 16, 16)).astype("float32"))
            np.savez(dd / "patches" / "norm_stats.npz",
                     mean=np.float32(0.0), std=np.float32(1.0), size=16)
            cfg = dd / "cfg.yaml"
            cfg.write_text(_cfg(d))
            r = subprocess.run(
                [sys.executable, "-m", "train.train_diffusion", "--config", str(cfg), "--resume"],
                cwd=REPO, capture_output=True, text=True, timeout=900)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            self.assertIn("starting fresh", r.stdout)


if __name__ == "__main__":
    unittest.main()
