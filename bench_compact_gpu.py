"""What actually costs the 48% in CompactMultiResHashGrid — on a GPU.

A CPU benchmark said the `bool(hit.all())` early-exit was a WIN (removing it
was slower, because the hash fallback then always runs). On GPU that call is a
device->host sync, and there is a second suspect the CPU test could not
separate: torch.searchsorted does a binary search over up to 129k sorted ids
for every one of ~524k query points, 8 corners x 8 levels per forward.

Variants, all required to produce bit-identical output:
  plain     - MultiResHashGrid (the 44 s/epoch reference)
  as_pushed - searchsorted + bool(hit.all()) early exit
  no_sync   - searchsorted, always torch.where (no host sync)
  map_gather- dense int64 linear_id -> slot map, prefilled so off-support ids
              already hold their hash fallback: lookup is ONE gather, no
              search, no branch, no sync.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.geo_encoding import (CompactMultiResHashGrid,  # noqa: E402
                                 MultiResHashGrid, build_latlon_support)
from utils import load_config  # noqa: E402

dev = torch.device("cuda")
cfg = load_config("config/wb2_20var.yaml")
g = cfg["geo"]
cf = np.load(Path(cfg["paths"]["patch_dir"]) / "coords_full.npz")
sup = build_latlon_support(cf["lat"], cf["lon"])
KW = dict(input_dim=g["input_dim"], n_levels=g["n_levels"],
          n_features_per_level=g["n_features_per_level"],
          log2_hashmap_size=g["log2_hashmap_size"],
          base_resolution=g["base_resolution"],
          finest_resolution=g["finest_resolution"])


class NoSync(CompactMultiResHashGrid):
    def _index_level(self, corner, l, n, n_entries):
        if not self.compact[l]:
            return MultiResHashGrid._index(self, corner, n, self.is_dense[l], n_entries)
        s = self.get_buffer(f"support_{l}")
        lin = self._linear_id(corner, n)
        pos = torch.searchsorted(s, lin).clamp(max=s.numel() - 1)
        h = torch.zeros_like(lin)
        for i in range(self.d):
            h = torch.bitwise_xor(h, corner[:, i] * self.primes[i])
        return torch.where(s[pos] == lin, pos, h % n_entries)


class MapGather(CompactMultiResHashGrid):
    """Precompute linear_id -> slot for every lattice vertex, once."""
    MAX_MAP = 1 << 24          # 16.7M entries = 134 MiB at int64; above this,
                               # fall back to searchsorted rather than blow up.

    def __init__(self, support, **kw):
        super().__init__(support, **kw)
        self.use_map = []
        for l in range(self.L):
            n = self.resolutions[l]
            cells = (n + 1) ** self.d
            if not self.compact[l] or cells > self.MAX_MAP:
                self.register_buffer(f"map_{l}", torch.zeros(0, dtype=torch.long))
                self.use_map.append(False)
                continue
            ids = self.get_buffer(f"support_{l}")
            n_entries = self.tables[l].shape[0]
            # Off-support entries pre-filled with their hash slot, so the gather
            # needs no branch and stays semantically identical to the original.
            lin_all = torch.arange(cells, dtype=torch.long)
            rem, corner = lin_all.clone(), []
            for _ in range(self.d):
                corner.append(rem % (n + 1))
                rem = rem // (n + 1)
            h = torch.zeros(cells, dtype=torch.long)
            for i in range(self.d):
                h = torch.bitwise_xor(h, corner[i] * self.primes[i])
            m = h % n_entries
            m[ids] = torch.arange(ids.numel(), dtype=torch.long)
            self.register_buffer(f"map_{l}", m)
            self.use_map.append(True)

    def _index_level(self, corner, l, n, n_entries):
        if not self.use_map[l]:
            return super()._index_level(corner, l, n, n_entries)
        return self.get_buffer(f"map_{l}")[self._linear_id(corner, n)]


torch.manual_seed(0)
mods = {}
mods["plain"] = MultiResHashGrid(**KW).to(dev)
mods["as_pushed"] = CompactMultiResHashGrid(support=sup, **KW).to(dev)
mods["no_sync"] = NoSync(support=sup, **KW).to(dev)
mods["map_gather"] = MapGather(support=sup, **KW).to(dev)
with torch.no_grad():
    for k in ("no_sync", "map_gather"):
        for l in range(mods[k].L):
            mods[k].tables[l].copy_(mods["as_pushed"].tables[l])

buf_mib = sum(v.numel() * v.element_size() for k, v in
              mods["map_gather"].state_dict().items() if k.startswith("map_")) / 2**20
print(f"map_gather extra buffers: {buf_mib:.1f} MiB "
      f"(levels mapped: {sum(mods['map_gather'].use_map)}/{mods['map_gather'].L})")

# Realistic training shape: one rank's batch of 128px patches.
q = sup[torch.randint(0, len(sup), (32 * 128 * 128,))].to(dev)
print(f"query points: {len(q):,}\n")

print("=== identical output (required) ===")
with torch.no_grad():
    ref = mods["as_pushed"](q[:50000])
    for k in ("no_sync", "map_gather"):
        d = (mods[k](q[:50000]) - ref).abs().max().item()
        print(f"  {k:>11}: max|diff| = {d:.3e}  {'OK' if d == 0 else 'MISMATCH'}")


def bench(m, x, train, reps=6):
    for _ in range(2):
        if train:
            m(x).square().sum().backward(); m.zero_grad(set_to_none=True)
        else:
            with torch.no_grad():
                m(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        if train:
            m(x).square().sum().backward(); m.zero_grad(set_to_none=True)
        else:
            with torch.no_grad():
                m(x)
    torch.cuda.synchronize()
    return (time.time() - t0) / reps * 1000


for train in (False, True):
    print(f"\n=== {'forward+backward' if train else 'forward'} ===")
    base = None
    for k, m in mods.items():
        ms = bench(m, q, train)
        base = ms if base is None else base
        print(f"  {k:>11}: {ms:8.2f} ms   ({ms/base:5.2f}x plain)")
