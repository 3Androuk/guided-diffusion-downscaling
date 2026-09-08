"""CompactMultiResHashGrid, ported from claude/wb2-20var-downscaling 85f1ee6.

The queried locus is a thin voxelised spherical shell (the +-60 deg band on the
unit sphere) inside a 3-D cube, so most lattice vertices can never be reached by
any ERA5 coordinate. This subclass enumerates the reachable vertices per level
from the fixed grid at CONSTRUCTION time and sizes each table to exactly that
count — before training, not by pruning after it. Dropping unreachable entries
removes nothing the model could have used, which is why the compact grid
reproduces the plain grid's interpolation exactly on the support.

Measured on config/wb2_20var.yaml (8 levels, base 16, finest 128, log2T 19):
3,318,628 -> 577,504 params (-82.6%), every level collision-free. The upstream
commit message quotes -43% (7.85M -> 4.45M) from the 16-level/finest-512 config
instead, so the numbers below are re-derived rather than copied.

These tests use a small synthetic support so they stay fast; the ladder-specific
counts are asserted as properties (dense, collision-free, exact agreement)
rather than as the wb220 magic numbers, which depend on the real grid.
"""

import unittest

import numpy as np
import torch

from models.geo_encoding import (CompactMultiResHashGrid, MultiResHashGrid,
                                 build_geo_encoder, build_latlon_support,
                                 build_level_gate, build_patch_coords)
from utils import geo_suffix

KW = dict(input_dim=3, n_levels=6, n_features_per_level=2,
          log2_hashmap_size=14, base_resolution=8, finest_resolution=64)


def _support(nlat=60, nlon=120):
    """A band grid shaped like the real one: dense in lon, +-60 deg in lat."""
    lat = np.linspace(-60.0, 60.0, nlat).astype(np.float32)
    lon = np.linspace(0.0, 359.0, nlon).astype(np.float32)
    return torch.from_numpy(build_patch_coords(lat, lon).reshape(-1, 3)), lat, lon


class CompactGridTests(unittest.TestCase):
    def setUp(self):
        self.sup, self.lat, self.lon = _support()
        torch.manual_seed(0)
        self.plain = MultiResHashGrid(**KW)
        self.compact = CompactMultiResHashGrid(support=self.sup, **KW)

    def test_compaction_happens_at_construction_not_by_pruning(self):
        """Tables are BORN small: no level may exceed its touched-vertex count."""
        for l in range(self.compact.L):
            if self.compact.compact[l]:
                n_touched = self.compact.get_buffer(f"support_{l}").numel()
                self.assertEqual(self.compact.tables[l].shape[0], n_touched)

    def test_saves_parameters_against_the_plain_grid(self):
        p = sum(x.numel() for x in self.plain.parameters())
        c = sum(x.numel() for x in self.compact.parameters())
        self.assertLess(c, p)

    def test_exact_agreement_with_the_plain_grid_on_support(self):
        """The property the whole design rests on: identical function, fewer
        slots. Copy plain's values into compact's slots, then compare."""
        self._copy_plain_into_compact()
        probe = self.sup[::37]
        with torch.no_grad():
            d = (self.plain(probe) - self.compact(probe)).abs().max().item()
        self.assertEqual(d, 0.0, "compact must not approximate — it re-indexes")

    def _copy_plain_into_compact(self):
        with torch.no_grad():
            for l in range(self.compact.L):
                if not self.compact.compact[l]:
                    self.compact.tables[l].copy_(self.plain.tables[l])
                    continue
                ids = self.compact.get_buffer(f"support_{l}")
                n = self.compact.resolutions[l]
                if self.plain.is_dense[l]:
                    self.compact.tables[l].copy_(self.plain.tables[l][ids])
                    continue
                rem = ids.clone()
                corner = torch.zeros(ids.numel(), 3, dtype=torch.long)
                for i in range(3):
                    corner[:, i] = rem % (n + 1)
                    rem = rem // (n + 1)
                h = torch.zeros(ids.numel(), dtype=torch.long)
                for i in range(3):
                    h = torch.bitwise_xor(h, corner[:, i] * self.plain.primes[i])
                self.compact.tables[l].copy_(
                    self.plain.tables[l][h % self.plain.tables[l].shape[0]])

    def test_every_compact_level_is_collision_free(self):
        for l in range(self.compact.L):
            if not self.compact.compact[l]:
                continue
            ids = self.compact.get_buffer(f"support_{l}")
            self.assertEqual(ids.numel(), torch.unique(ids).numel())
            self.assertTrue(bool((ids[1:] > ids[:-1]).all()),
                            "searchsorted needs a strictly sorted support")

    def test_no_real_coordinate_ever_misses_the_table(self):
        """If a training query fell off support it would hash into an occupied
        slot and corrupt a learned vertex. Assert that cannot happen."""
        for l in range(self.compact.L):
            if not self.compact.compact[l]:
                continue
            n = self.compact.resolutions[l]
            sup = self.compact.get_buffer(f"support_{l}")
            base = torch.floor(self.sup * n).long()
            for off in self.compact.corner_offsets:
                lin = self.compact._linear_id((base + off).clamp(0, n), n)
                pos = torch.searchsorted(sup, lin).clamp(max=sup.numel() - 1)
                self.assertTrue(bool((sup[pos] == lin).all()),
                                f"level {l} has off-support queries")

    def test_gradients_reach_every_level(self):
        out = self.compact(self.sup[:500])
        out.square().sum().backward()
        for l, t in enumerate(self.compact.tables):
            self.assertIsNotNone(t.grad, f"level {l} got no gradient")
            self.assertGreater(t.grad.abs().sum().item(), 0.0)

    def test_off_support_input_is_finite_not_a_crash(self):
        with torch.no_grad():
            o = self.compact(torch.rand(200, 3))
        self.assertTrue(torch.isfinite(o).all())

    def test_checkpoint_round_trip_through_a_rebuilt_encoder(self):
        """Eval reconstructs the encoder from the same support, so shapes must
        match at load time — this is what makes the arm evaluable at all."""
        sd = self.compact.state_dict()
        torch.manual_seed(99)
        rebuilt = CompactMultiResHashGrid(support=self.sup, **KW)
        rebuilt.load_state_dict(sd)
        probe = self.sup[::53]
        with torch.no_grad():
            self.assertEqual(
                (rebuilt(probe) - self.compact(probe)).abs().max().item(), 0.0)

    def test_a_different_grid_fails_loudly_rather_than_silently(self):
        """The checkpoint is tied to the grid it was built on. A lat_range
        change must raise at load, not quietly mis-index."""
        other, _, _ = _support(nlat=30, nlon=60)
        rebuilt = CompactMultiResHashGrid(support=other, **KW)
        with self.assertRaises(RuntimeError):
            rebuilt.load_state_dict(self.compact.state_dict())


class WiringTests(unittest.TestCase):
    def test_build_geo_encoder_dispatches_and_reads_the_real_grid(self):
        sup, lat, lon = _support()
        cfg = {"geo": {"enabled": True, "encoder": "hash_compact",
                       "altitude": None, **KW}}
        enc = build_geo_encoder(cfg, support=sup)
        self.assertIsInstance(enc, CompactMultiResHashGrid)
        self.assertEqual(enc.output_dim, KW["n_levels"] * KW["n_features_per_level"])

    def test_output_dim_matches_the_other_arms(self):
        """Same conditioning width, so the UNet is identical across arms and the
        comparison isolates the encoder."""
        sup, _, _ = _support()
        a = build_geo_encoder({"geo": {"encoder": "hash_compact",
                                       "altitude": None, **KW}}, support=sup)
        b = build_geo_encoder({"geo": {"encoder": "hash", **KW}})
        self.assertEqual(a.output_dim, b.output_dim)

    def test_checkpoint_suffix_is_distinct(self):
        s = geo_suffix({"geo": {"enabled": True, "encoder": "hash_compact"}})
        self.assertEqual(s, "_geo_hashcompact")
        for other in ("hash", "hash2d", "healpix", "static"):
            self.assertNotEqual(
                s, geo_suffix({"geo": {"enabled": True, "encoder": other}}))

    def test_level_gating_accepts_it(self):
        g = build_level_gate({"geo": {"encoder": "hash_compact",
                                      "level_gating": True, **KW}})
        self.assertIsNotNone(g)

    def test_support_builder_matches_the_dataset_coordinates(self):
        """build_latlon_support must produce exactly what PatchDataset feeds the
        encoder, or training queries would fall off support."""
        sup, lat, lon = _support()
        built = build_latlon_support(lat, lon, altitude=None)
        self.assertEqual(built.shape, sup.shape)
        self.assertEqual((built - sup).abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()


class IndexMapTests(unittest.TestCase):
    """The dense linear-id -> slot map that replaced torch.searchsorted.

    Measured on a GH200 at the real training shape (524,288 points,
    forward+backward): the binary search ran at 1.44x the plain grid and the
    map at 1.01x. The map must therefore be exercised — and must stay exactly
    equivalent to the search it replaced, or the arm is no longer the encoder
    it claims to be.
    """

    def setUp(self):
        self.sup, _, _ = _support()
        torch.manual_seed(0)
        self.m = CompactMultiResHashGrid(support=self.sup, **KW)

    def test_the_map_path_is_actually_taken(self):
        self.assertTrue(any(self.m.use_map),
                        "no level used the map, so the fast path is untested")

    def test_map_and_search_agree_exactly(self):
        """Same indices from both paths, on support and off it."""
        for l in range(self.m.L):
            if not self.m.use_map[l]:
                continue
            n = self.m.resolutions[l]
            n_entries = self.m.tables[l].shape[0]
            on = torch.floor(self.sup[:400] * n).long().clamp(0, n)
            off = torch.randint(0, n + 1, (400, 3))
            for corner in (on, off):
                viamap = self.m._index_level(corner, l, n, n_entries)
                self.m.use_map[l] = False
                viasearch = self.m._index_level(corner, l, n, n_entries)
                self.m.use_map[l] = True
                self.assertTrue(torch.equal(viamap, viasearch),
                                f"level {l}: map and search disagree")

    def test_indices_stay_in_range(self):
        for l in range(self.m.L):
            if not self.m.use_map[l]:
                continue
            mp = self.m.get_buffer(f"map_{l}")
            self.assertGreaterEqual(int(mp.min()), 0)
            self.assertLess(int(mp.max()), self.m.tables[l].shape[0])

    def test_maps_are_not_persisted_into_checkpoints(self):
        """They are pure functions of the support: rebuilding is ~2 s, while
        persisting them would add tens of MiB to every 3.9 GB checkpoint AND
        break checkpoints written before this change."""
        keys = self.m.state_dict().keys()
        self.assertFalse([k for k in keys if k.startswith("map_")])
        self.assertTrue([k for k in keys if k.startswith("support_")])

    def test_loads_a_checkpoint_written_before_the_map_existed(self):
        """The running arm's checkpoints have tables + support_ only."""
        sd = {k: v for k, v in self.m.state_dict().items()
              if not k.startswith("map_")}
        torch.manual_seed(7)
        fresh = CompactMultiResHashGrid(support=self.sup, **KW)
        fresh.load_state_dict(sd)          # strict=True must succeed
        probe = self.sup[::41]
        with torch.no_grad():
            self.assertEqual((fresh(probe) - self.m(probe)).abs().max().item(), 0.0)

    def test_oversized_levels_fall_back_to_the_search(self):
        """A finer ladder must not try to materialize a huge map."""
        torch.manual_seed(0)
        small = CompactMultiResHashGrid(support=self.sup, **KW)
        small.MAX_MAP = 100          # force every level over the cap
        small._build_index_maps()
        self.assertFalse(any(small.use_map))
        probe = self.sup[::41]
        with torch.no_grad():
            self.assertEqual((small(probe) - self.m(probe)).abs().max().item(), 0.0)


class CompactStaticComboTests(unittest.TestCase):
    """hash_compact_static: the compact grid CONCATENATED with static fields.

    The discriminating arm of the geo ablation. Motivated by the paired 5-way
    eval's per-channel split at 4x vs bicubic: static leads the surface
    (t2m -26.3% vs -21.7%, msl -50.9% vs -43.1%) while the compact grid leads
    the free atmosphere (z500 -28.7% vs -27.8%). If the combo only matches
    static, the learned tables were physiography proxies.
    """

    N_STATIC = 3        # geopotential_at_surface, land_sea_mask, slope

    def setUp(self):
        self.sup, self.lat, self.lon = _support()
        self.cfg = {"geo": {"enabled": True, "encoder": "hash_compact_static",
                            "altitude": None,
                            "static_fields": ["a", "b", "c"], **KW}}
        torch.manual_seed(0)
        self.enc = build_geo_encoder(self.cfg, support=self.sup)

    def test_output_is_learned_plus_static_channels(self):
        plain = build_geo_encoder(
            {"geo": dict(self.cfg["geo"], encoder="hash_compact")},
            support=self.sup)
        self.assertEqual(self.enc.output_dim, plain.output_dim + self.N_STATIC)

    def test_forward_splits_the_payload_at_the_grid_dim(self):
        """Payload is [coords | static]; the grid must see only the coords."""
        coords = self.sup[:256].reshape(1, 16, 16, 3)
        static = torch.randn(1, 16, 16, self.N_STATIC)
        out = self.enc(torch.cat([coords, static], dim=-1))
        self.assertEqual(tuple(out.shape), (1, 16, 16, self.enc.output_dim))
        # the static tail must pass through untouched
        self.assertTrue(torch.equal(out[..., -self.N_STATIC:], static))
        # ... and the learned head must equal the bare grid on those coords
        with torch.no_grad():
            self.assertTrue(torch.equal(out[..., :self.enc.hash.output_dim],
                                        self.enc.hash(coords)))

    def test_uses_the_compact_grid_not_the_plain_one(self):
        self.assertIsInstance(self.enc.hash, CompactMultiResHashGrid)
        self.assertTrue(any(self.enc.hash.use_map))

    def test_gradients_reach_the_learned_tables(self):
        coords = self.sup[:256].reshape(1, 16, 16, 3)
        static = torch.randn(1, 16, 16, self.N_STATIC)
        self.enc(torch.cat([coords, static], dim=-1)).square().sum().backward()
        for l, t in enumerate(self.enc.hash.tables):
            self.assertIsNotNone(t.grad, f"level {l} got no gradient")
            self.assertGreater(t.grad.abs().sum().item(), 0.0)

    def test_checkpoint_name_is_distinct_from_both_parents(self):
        s = geo_suffix({"geo": {"enabled": True, "encoder": "hash_compact_static"}})
        self.assertEqual(s, "_geo_compactcombo")
        for other in ("hash_compact", "static", "hash_static", "hash2d"):
            self.assertNotEqual(
                s, geo_suffix({"geo": {"enabled": True, "encoder": other}}))

    def test_dataset_emits_coords_concatenated_with_static(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            dd = Path(d)
            rng = np.random.default_rng(0)
            np.save(dd / "p.npy", rng.standard_normal((2, 1, 8, 8)).astype("float32"))
            np.save(dd / "o.npy", np.array([[0, 0], [4, 4]], dtype=np.int64))
            np.savez(dd / "cf.npz", lat=np.linspace(-60, 60, 16).astype("float32"),
                     lon=np.linspace(0, 15, 16).astype("float32"))
            np.savez(dd / "static_fields.npz",
                     fields=rng.standard_normal((self.N_STATIC, 16, 16)).astype("float32"))
            from data.dataset import Normalizer, PatchDataset
            ds = PatchDataset(dd / "p.npy", Normalizer(0.0, 1.0),
                              origins_path=dd / "o.npy",
                              coords_full_path=dd / "cf.npz",
                              geo_input_dim=3,
                              geo_encoder="hash_compact_static")
            _, payload = ds[1]
            self.assertEqual(tuple(payload.shape), (8, 8, 3 + self.N_STATIC))


from pathlib import Path  # noqa: E402  (used by the dataset test above)
