"""Merge the shard JSONs of one eval.downscale_nwp run into the unsharded file.

Every score in these files is a MEAN over the shard's inits of a per-init
statistic (lat-weighted RMSE, CRPS, spread, spectrum, balance residual), so the
merged value is the n_inits_done-weighted mean of the shard values — exactly what
a single unsharded run would have produced. Lists (per-channel) are merged
element-wise. Metadata is taken from the first shard with the shard key dropped.

    python -m eval.merge_nwp_shards results_wb220/nwp_<stem>_ens4.json
    python -m eval.merge_nwp_shards --all results_wb220     # every stem with shards
"""
import argparse, glob, json, re, sys
from pathlib import Path

import numpy as np


def _merge_numeric(vals, weights):
    w = np.asarray(weights, dtype=np.float64); w = w / w.sum()
    if isinstance(vals[0], list):
        return (np.average(np.asarray(vals, dtype=np.float64), axis=0, weights=w)).tolist()
    return float(np.average(np.asarray(vals, dtype=np.float64), weights=w))


def merge(target: Path) -> Path:
    stem = target.with_suffix("").name
    shards = sorted(glob.glob(str(target.parent / f"{stem}_shard*of*.json")))
    if not shards:
        raise SystemExit(f"no shard files for {target.name}")
    docs = [json.load(open(f)) for f in shards]
    ns = {int(re.search(r"_shard(\d+)of(\d+)", f).group(2)) for f in shards}
    if len(ns) != 1:
        raise SystemExit(f"mixed shard counts {ns} for {target.name}")
    n_expected = ns.pop()
    if len(shards) != n_expected:
        print(f"  WARNING {target.name}: {len(shards)} of {n_expected} shards present — partial merge", file=sys.stderr)
    weights = [d["n_inits_done"] for d in docs]
    out = {k: v for k, v in docs[0].items() if k not in ("arms",)}
    out["shard"] = None
    out["merged_from"] = [Path(f).name for f in shards]
    out["shard_inits_done"] = weights
    out["n_inits_done"] = int(sum(weights))
    # each shard's n_inits is ITS slice length; the run's total is their sum
    out["n_inits"] = int(sum(d["n_inits"] for d in docs))
    out["complete"] = bool(len(shards) == n_expected and all(d["complete"] for d in docs)
                           and out["n_inits_done"] == out["n_inits"])
    out["arms"] = {}
    for arm in docs[0]["arms"]:
        out["arms"][arm] = {}
        for entry in docs[0]["arms"][arm]:               # bicubic / native / model
            merged = {}
            for key in docs[0]["arms"][arm][entry]:
                vals = [d["arms"][arm][entry][key] for d in docs if entry in d["arms"][arm]]
                merged[key] = _merge_numeric(vals, weights[: len(vals)]) if isinstance(vals[0], (int, float, list)) else vals[0]
            out["arms"][arm][entry] = merged
    tmp = target.with_suffix(".json.tmp"); tmp.write_text(json.dumps(out, indent=1)); tmp.replace(target)
    print(f"  merged {len(shards)} shards -> {target.name}  (inits {out['n_inits_done']}/{out['n_inits']}, complete={out['complete']})")
    return target


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="?", help="unsharded output path, e.g. results_wb220/nwp_<stem>_ens4.json")
    ap.add_argument("--all", metavar="DIR", help="merge every stem in DIR that has shard files")
    a = ap.parse_args()
    if a.all:
        stems = sorted({re.sub(r"_shard\d+of\d+\.json$", ".json", f) for f in glob.glob(str(Path(a.all) / "nwp_*_shard*of*.json"))})
        for t in stems: merge(Path(t))
    elif a.target:
        merge(Path(a.target))
    else:
        ap.error("give a target path or --all DIR")


if __name__ == "__main__":
    main()
