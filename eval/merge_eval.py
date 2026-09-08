"""Merge per-arm eval JSONs from scripts/eval_sharded.sh into one table.

Each shard writes results_wb220/compare_<stem>_proj.json holding that arm plus
Bicubic. Merging is a dict union rather than any kind of averaging: the arms
were scored on identical patches with identical per-arm noise seeds, so the
numbers are the ones a single N-way job would have produced.

Bicubic is deterministic and appears in every shard, so it doubles as a
CONSISTENCY CHECK -- if two shards disagree on it, they did not score the same
patches and the merge must not proceed.
"""
import json, sys
from pathlib import Path

RES = Path(sys.argv[1] if len(sys.argv) > 1
           else "/path/to/physics-informed-weather/"
                "era5-diffusion-downscaling/results_wb220")
# Single-arm shard files ONLY. The bare glob also matches leftovers from
# earlier protocols -- a stale 256-patch "_vs_" file tripped the Bicubic check
# on the first run, which is precisely what that check exists to prevent, but
# excluding them up front is cleaner than relying on the abort.
files = sorted(f for f in RES.glob("compare_diffusion*_proj.json")
               if "_vs_" not in f.name and "way" not in f.name)
if not files:
    sys.exit("no single-arm shard files found")
merged, bicubic = {}, {}
for f in files:
    d = json.load(open(f))
    for tag, row in d.items():
        for name, v in row.items():
            if name == "Bicubic":
                prev = bicubic.get(tag)
                if prev is not None and abs(prev - v["l2_normalized"]) > 1e-12:
                    sys.exit(f"ABORT: {f.name} scored Bicubic {v['l2_normalized']:.10f} "
                             f"at {tag} but a previous shard got {prev:.10f} — "
                             f"the shards did not use the same patches.")
                bicubic[tag] = v["l2_normalized"]
            merged.setdefault(tag, {})[name] = v
print(f"merged {len(files)} shard files; Bicubic consistent across all "
      f"({', '.join(f'{k} {v:.7f}' for k, v in bicubic.items())})\n")

for tag in sorted(merged, key=lambda t: int(t.rstrip('x'))):
    print(f"--- {tag} ---")
    for name, v in sorted(merged[tag].items(), key=lambda kv: kv[1]["l2_normalized"]):
        print(f"  {name:30s} l2n {v['l2_normalized']:.7f}  "
              f"spec {v['spectrum_log_l1']:.6f}")
out = RES / "compare_FINAL_merged.json"
json.dump(merged, open(out, "w"), indent=2)
print(f"\n-> {out}")
