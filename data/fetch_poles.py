"""Download ONLY the polar rows the +-60 deg raw download is missing.

The HEALPix mesh is global, but datasets/raw_wb220 covers 481 of the store's
721 latitude rows. Rather than re-streaming 208 GB, this fetches the two
missing caps -- lat <= -60.25 and lat >= 60.25, 120 rows each -- for the same
years, stride, and 20 channels, so they can be stitched onto the existing band
before remapping to HPX faces.

Everything expensive is reused from data.download_era5: _download_year streams
one year straight into an on-disk memmap (a 20-channel year does not fit the
4 GiB login-node cgroup), groups channels by variable so WB2's all-levels-in-
one-chunk layout is read once per variable rather than once per level, and
retries each batch on a fresh connection.

Resumable: a year already on disk is skipped, so a dropped SSH session costs
at most the batch in flight. Run on a LOGIN node -- it is pure network I/O and
a GPU node would bill node-hours with the GPUs idle.

    python -m data.fetch_poles --config config/wb2_20var.yaml \
        --out /path/to/datasets/poles_wb220 --batch 16
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.download_era5 import _download_year  # noqa: E402
from utils import ensure_dir, load_config  # noqa: E402

# Store grid is 0.25 deg, 721 rows from -90 to 90; the band keeps [-60, 60].
CAPS = {"south": (-90.0, -60.25), "north": (60.25, 90.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/wb2_20var.yaml")
    ap.add_argument("--out", required=True, help="directory for the cap caches")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--chunk-time", type=int, default=8)
    ap.add_argument("--max-retries", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0,
                    help="this process handles cap-years [shard::nshards]. "
                         "Sharding rather than a shared queue keeps the "
                         "processes race-free without a lock: no two ever "
                         "touch the same .tmp file.")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--dask-threads", type=int, default=4,
                    help="dask worker threads. The default pool is sized to the "
                         "node's CPU count (144 here), which multiplies the "
                         "in-flight chunk memory and OOM-kills against the 4 GiB "
                         "login cgroup.")
    args = ap.parse_args()

    import dask
    dask.config.set(scheduler="threads", num_workers=args.dask_threads)
    dask.config.set({"array.chunk-size": "32MiB"})

    cfg = load_config(args.config)
    dcfg = cfg["data"]
    stride = dcfg.get("time_stride", 1)
    out_dir = ensure_dir(args.out)
    splits = {
        "train": list(range(dcfg["train_years"][0], dcfg["train_years"][1] + 1)),
        "test": list(range(dcfg["test_years"][0], dcfg["test_years"][1] + 1)),
    }

    todo = [(s, y, c) for s, ys in splits.items() for y in ys for c in CAPS]
    todo = todo[args.shard::args.nshards]
    done = [t for t in todo if (out_dir / f"{t[0]}_{t[1]}_{t[2]}.npy").exists()]
    print(f"shard {args.shard}/{args.nshards}: {len(todo)} cap-years, "
          f"{len(done)} cached, {len(todo) - len(done)} to fetch", flush=True)

    for split, year, cap in todo:
        path = out_dir / f"{split}_{year}_{cap}.npy"
        if path.exists():
            print(f"[skip] {path.name}", flush=True)
            continue
        lo, hi = CAPS[cap]
        print(f"[fetch] {path.name}  lat [{lo}, {hi}]", flush=True)
        tmp = path.with_suffix(".npy.tmp")
        shape, lat, lon = _download_year(
            dcfg, (lo, hi), year, stride, args.batch, args.timeout,
            args.chunk_time, args.max_retries, tmp)
        tmp.replace(path)
        if args.shard == 0:      # one writer is enough; all shards agree
            np.savez(out_dir / f"{cap}_coords.npz", lat=lat, lon=lon)
        print(f"[done] {path.name}: {shape}", flush=True)

    print("all cap-years cached", flush=True)


if __name__ == "__main__":
    main()
