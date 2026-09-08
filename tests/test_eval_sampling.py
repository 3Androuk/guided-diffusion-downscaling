"""Which test patches the eval scores, and that payloads stay aligned to them.

Patches are time-ordered at 8 per field, so the previous `range(n)` scored the
first n/8 FIELDS -- at n=256 that is ~32 consecutive days of January 2016 out
of two years available. Rankings were still valid (every arm saw identical
patches) but the scores were winter-only, resting on ~32 independent weather
states rather than 256.

The alignment test is the important one: the truth stack and each model's geo
payload are built by SEPARATE code paths, so an index change in one and not
the other silently scores every patch against another patch's coordinates --
a wrong number with no error.
"""

import unittest

import numpy as np

from eval.compare_geo import test_indices


class TestIndexTests(unittest.TestCase):
    def test_spans_the_whole_split_not_the_head(self):
        idx = test_indices(5848, 256)
        self.assertEqual(len(idx), 256)
        self.assertEqual(idx[0], 0)
        self.assertEqual(idx[-1], 5847)
        # fields covered, at 8 patches per field
        fields = {int(i) // 8 for i in idx}
        self.assertGreater(len(fields), 200,
                           "should span most of the 731 fields, not ~32")

    def test_old_behaviour_covered_only_32_fields(self):
        """Pins what was wrong, so a revert is visible."""
        old = np.arange(256)
        self.assertEqual(len({int(i) // 8 for i in old}), 32)

    def test_indices_are_unique_and_sorted(self):
        idx = test_indices(5848, 256)
        self.assertEqual(len(set(idx.tolist())), len(idx))
        self.assertTrue((np.diff(idx) > 0).all())

    def test_is_deterministic(self):
        """Two runs must score identical patches or nothing is comparable."""
        self.assertTrue(np.array_equal(test_indices(5848, 256),
                                       test_indices(5848, 256)))

    def test_asking_for_more_than_exists_is_clamped(self):
        idx = test_indices(100, 4096)
        self.assertEqual(len(idx), 100)
        self.assertEqual(idx[-1], 99)

    def test_larger_n_keeps_full_coverage(self):
        """Upping the count must add samples, not narrow the window."""
        small, big = test_indices(5848, 256), test_indices(5848, 1024)
        self.assertEqual(small[-1], big[-1])
        self.assertGreater(len({int(i) // 8 for i in big}),
                           len({int(i) // 8 for i in small}))

    def test_truth_and_payload_use_the_same_indices(self):
        """The alignment invariant, checked on a real PatchDataset pair."""
        import tempfile
        from pathlib import Path
        from data.dataset import Normalizer, PatchDataset
        with tempfile.TemporaryDirectory() as d:
            dd = Path(d)
            rng = np.random.default_rng(0)
            n_patch = 40
            np.save(dd / "test_patches.npy",
                    rng.standard_normal((n_patch, 1, 8, 8)).astype("float32"))
            # origin row encodes the patch index, so a mismatch is detectable
            np.save(dd / "test_origins.npy",
                    np.stack([np.arange(n_patch), np.zeros(n_patch)], 1).astype(np.int64))
            np.savez(dd / "coords_full.npz",
                     lat=np.linspace(-60, 60, n_patch + 8).astype("float32"),
                     lon=np.linspace(0, 15, 16).astype("float32"))
            norm = Normalizer(0.0, 1.0)
            plain = PatchDataset(dd / "test_patches.npy", norm)
            geo = PatchDataset(dd / "test_patches.npy", norm,
                               origins_path=dd / "test_origins.npy",
                               coords_full_path=dd / "coords_full.npz",
                               geo_input_dim=3, geo_encoder="hash")
            sel = test_indices(n_patch, 10)
            for i in sel:
                truth = plain[int(i)]
                patch, coords = geo[int(i)]
                self.assertTrue(np.allclose(truth.numpy(), patch.numpy()),
                                f"patch {i}: truth and geo path disagree")
                # coords must start at the row this patch's origin names
                lat0 = np.linspace(-60, 60, n_patch + 8).astype("float32")[int(i)]
                self.assertAlmostEqual(
                    float(coords[0, 0, 2]), float((np.sin(np.deg2rad(lat0)) + 1) / 2),
                    places=5, msg=f"patch {i}: coords are from another patch")


if __name__ == "__main__":
    unittest.main()
