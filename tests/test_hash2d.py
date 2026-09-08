"""The 2-D plate-carree hash arm (--encoder hash2d).

The experiment this enables: the 3-D xyz hash spends ~84% of its 3.3M entries
on volume cells the 2-D sphere surface never touches (measured on the trained
wb220 UCL checkpoint). hash2d parameterizes the same angular band with a 2-D
(lat, lon) grid — every level dense, ~178k params, zero collisions — at the
cost of a prime-meridian seam and no equal-area pole treatment. These tests
pin the properties the comparison depends on: the angular match to the 3-D
ladder, the density claim, the payload plumbing, and a distinct checkpoint
name so the arm cannot overwrite the 3-D hash run.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from data.dataset import Normalizer, PatchDataset
from models.geo_encoding import (MultiResHashGrid, build_geo_encoder,
                                 build_level_gate, build_patch_coords)
from utils import geo_suffix

# Mirrors config/wb2_20var.yaml's geo block where it matters here.
GEO = {"enabled": True, "encoder": "hash2d", "input_dim": 3, "n_levels": 8,
       "n_features_per_level": 2, "log2_hashmap_size": 19,
       "base_resolution": 16, "finest_resolution": 128,
       "hash2d_base_resolution": 25, "hash2d_finest_resolution": 200,
       "altitude": None}


class Hash2dCoordTests(unittest.TestCase):
    def test_shape_range_and_mapping(self):
        lat = np.linspace(-60.0, 60.0, 9)
        lon = np.linspace(0.0, 90.0, 7)
        c = build_patch_coords(lat, lon, input_dim=2)
        self.assertEqual(c.shape, (9, 7, 2))
        self.assertEqual(c.dtype, np.float32)
        self.assertTrue((c >= 0.0).all() and (c <= 1.0).all())
        # (lat + 90)/180 and lon/360: the equator row and lon 180 hit 0.5.
        self.assertAlmostEqual(c[4, 0, 0], 0.5, places=6)     # lat 0
        self.assertAlmostEqual(float(build_patch_coords(
            np.array([0.0]), np.array([180.0]), input_dim=2)[0, 0, 1]), 0.5,
            places=6)

    def test_monotone_in_both_axes(self):
        lat = np.linspace(-60.0, 60.0, 5)
        lon = np.linspace(10.0, 50.0, 5)
        c = build_patch_coords(lat, lon, input_dim=2)
        self.assertTrue((np.diff(c[:, 0, 0]) > 0).all())
        self.assertTrue((np.diff(c[0, :, 1]) > 0).all())

    def test_meridian_seam_is_real_and_documented(self):
        """Adjacent columns across lon 0 get DISTANT coordinates.

        This is the accepted cost of the 2-D parameterization (the xyz mode is
        seamless); the test documents it so a silent 'fix' that changes the
        parameterization mid-study shows up as a failure to be discussed, and
        pins that the seam can only separate patches, never split one (patch
        columns are a contiguous slice of the 0..360 grid, tested implicitly
        by monotonicity above).
        """
        c = build_patch_coords(np.array([0.0]), np.array([359.75, 0.0]),
                               input_dim=2)
        self.assertGreater(abs(c[0, 0, 1] - c[0, 1, 1]), 0.99)

    def test_altitude_rejected_in_2d_mode(self):
        with self.assertRaises(AssertionError):
            build_patch_coords(np.array([0.0]), np.array([0.0]),
                               altitude=0.5, input_dim=2)

    def test_3d_default_unchanged(self):
        """input_dim defaults to 3 so every existing caller is untouched."""
        lat, lon = np.linspace(-30, 30, 4), np.linspace(0, 40, 4)
        old = build_patch_coords(lat, lon)
        self.assertEqual(old.shape, (4, 4, 3))
        with_alt = build_patch_coords(lat, lon, altitude=0.25)
        self.assertEqual(with_alt.shape, (4, 4, 4))


class Hash2dEncoderTests(unittest.TestCase):
    def setUp(self):
        self.enc = build_geo_encoder({"geo": GEO})

    def test_ladder_is_angular_matched(self):
        """base 25 = 7.2 deg of latitude, the 3-D grid's 2/16 rad base cell."""
        self.assertIsInstance(self.enc, MultiResHashGrid)
        self.assertEqual(self.enc.d, 2)
        r = self.enc.resolutions
        self.assertEqual(len(r), 8)
        self.assertEqual(r[0], 25)
        self.assertAlmostEqual(180.0 / r[0], 7.2, places=6)
        # floor(base * b^7) can land on 199 from float rounding; both are the
        # intended ~0.9 deg finest cell.
        self.assertIn(r[-1], (199, 200))
        self.assertTrue(all(b > a for a, b in zip(r, r[1:])))

    def test_every_level_is_dense(self):
        """The point of the arm: no hashing, no collisions, at any level."""
        self.assertTrue(all(self.enc.is_dense))
        for n, table in zip(self.enc.resolutions, self.enc.tables):
            self.assertEqual(table.shape[0], (n + 1) ** 2)

    def test_param_count_is_178k_not_3m(self):
        n = sum(p.numel() for p in self.enc.parameters())
        self.assertGreater(n, 170_000)
        self.assertLess(n, 185_000)

    def test_output_dim_matches_the_3d_hash(self):
        """Same 16-channel conditioning width, so the UNet is identical across
        arms and the comparison isolates the parameterization."""
        g3 = dict(GEO, encoder="hash")
        self.assertEqual(self.enc.output_dim,
                         build_geo_encoder({"geo": g3}).output_dim)

    def test_forward_shapes_and_gradients_reach_every_level(self):
        coords = torch.rand(2, 8, 8, 2, requires_grad=False)
        out = self.enc(coords)
        self.assertEqual(out.shape, (2, 8, 8, self.enc.output_dim))
        out.square().sum().backward()
        for l, table in enumerate(self.enc.tables):
            self.assertIsNotNone(table.grad, f"level {l} got no gradient")
            self.assertGreater(table.grad.abs().sum().item(), 0.0,
                               f"level {l} gradient is all zero")

    def test_level_gate_accepts_hash2d(self):
        g = dict(GEO, level_gating=True)
        gate = build_level_gate({"geo": g})
        self.assertIsNotNone(gate)
        self.assertEqual(gate.gated_dim, self.enc.output_dim)


class Hash2dPlumbingTests(unittest.TestCase):
    def test_dataset_derives_dim_2_from_the_encoder_name(self):
        """Call sites pass the config's input_dim (3); hash2d must override it,
        or the encoder would be fed xyz coordinates it interprets as (lat, lon).
        """
        with tempfile.TemporaryDirectory() as d:
            dd = Path(d)
            rng = np.random.default_rng(0)
            np.save(dd / "p.npy", rng.standard_normal((2, 1, 8, 8)).astype("float32"))
            np.save(dd / "o.npy", np.array([[0, 0], [4, 4]], dtype=np.int64))
            np.savez(dd / "cf.npz", lat=np.linspace(-60, 60, 16).astype("float32"),
                     lon=np.linspace(0, 15, 16).astype("float32"))
            ds = PatchDataset(dd / "p.npy", Normalizer(0.0, 1.0),
                              origins_path=dd / "o.npy",
                              coords_full_path=dd / "cf.npz",
                              geo_input_dim=3, geo_encoder="hash2d")
            _, coords = ds[1]
            self.assertEqual(tuple(coords.shape), (8, 8, 2))
            # ... and the crop matches the origin: row 4 of the full grid.
            self.assertAlmostEqual(float(coords[0, 0, 0]),
                                   (np.linspace(-60, 60, 16)[4] + 90.0) / 180.0,
                                   places=5)

    def test_checkpoint_suffix_cannot_collide_with_the_3d_hash(self):
        s2 = geo_suffix({"geo": dict(GEO)})
        s3 = geo_suffix({"geo": dict(GEO, encoder="hash")})
        self.assertEqual(s2, "_geo_hash2d")
        self.assertEqual(s3, "_geo")
        self.assertNotEqual(s2, s3)


if __name__ == "__main__":
    unittest.main()
