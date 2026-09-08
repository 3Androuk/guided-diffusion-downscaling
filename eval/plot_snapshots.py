"""Multi-variable maps from --snapshot dumps: the method ladder side by side for
any of the 20 channels, on one shared colour scale.

Two ablations are visible in every figure: GEO (no-geo vs static vs compactcombo)
and PROJECTION (compactcombo vs compactcombo UNPROJECTED), against three
references (bicubic floor, native ceiling, truth).

    python -m eval.plot_snapshots                        # fields, all vars, both regimes
    python -m eval.plot_snapshots --error                # pred - truth (small differences)
    python -m eval.plot_snapshots --zoom 25 55 240 300   # lat0 lat1 lon0 lon1
"""
import argparse, os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(os.environ.get("PROJECTDIR", "")) / "results_wb220"
SNAP = R / "nwp_snapshots"

# (panel label, snapshot stem, array key). Order = the method ladder.
PANELS = [
    ("Bicubic from 1.5°",        "diffusion_geo_compactcombo",        "bicubic"),
    ("No-geo diffusion",         "diffusion_best",                    "model_member0"),
    ("static (geo)",             "diffusion_geo_static",              "model_member0"),
    ("compactcombo (projected)", "diffusion_geo_compactcombo",        "model_member0"),
    ("compactcombo UNPROJECTED", "diffusion_geo_compactcombo_noproj", "model_member0"),
    ("residual_lm (amortized)",  "residual_lm_best",                  "model_member0"),
    ("Native 0.25° forecast",    "diffusion_geo_compactcombo",        "native"),
    ("Truth (0.25° analysis)",   "diffusion_geo_compactcombo",        "truth"),
]
TRUTH = "Truth (0.25° analysis)"
CMAP = {"t2m": "RdYlBu_r", "t850": "RdYlBu_r", "t700": "RdYlBu_r", "t500": "RdYlBu_r",
        "msl": "viridis", "tcwv": "YlGnBu",
        "z500": "viridis", "z700": "viridis", "z850": "viridis",
        "q500": "BrBG", "q700": "BrBG", "q850": "BrBG"}
DEFAULT_VARS = ["t2m", "msl", "z500", "q850", "u10", "tcwv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["forecast", "control"])
    ap.add_argument("--vars", nargs="+", default=DEFAULT_VARS)
    ap.add_argument("--error", action="store_true", help="plot pred - truth instead of the field")
    ap.add_argument("--zoom", nargs=4, type=float, default=None,
                    metavar=("LAT0", "LAT1", "LON0", "LON1"))
    ap.add_argument("--out", default=str(R / "figs_nwp"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    made = 0
    for arm in a.arms:
        got = {}
        for _, stem, _ in PANELS:
            hits = sorted(SNAP.glob(f"{stem}_{arm}_*.npz"))
            if hits and stem not in got:
                got[stem] = np.load(hits[0], allow_pickle=True)
        if not got:
            print(f"  no snapshots for arm={arm}"); continue
        ref = got["diffusion_geo_compactcombo"]
        ch = [str(c) for c in ref["channels"]]
        lat = ref["lat"]; H, W = ref["truth"].shape[-2:]
        lon = np.linspace(0, 360, W, endpoint=False)
        sl = (slice(None), slice(None))
        if a.zoom:
            la0, la1, lo0, lo1 = a.zoom
            ri = np.where((lat >= min(la0, la1)) & (lat <= max(la0, la1)))[0]
            ci = np.where((lon >= lo0) & (lon <= lo1))[0]
            sl = (slice(ri[0], ri[-1] + 1), slice(ci[0], ci[-1] + 1))
        ext = [lon[sl[1]][0], lon[sl[1]][-1], lat[sl[0]][-1], lat[sl[0]][0]]

        for v in a.vars:
            if v not in ch:
                print(f"  skip {v}: not a channel"); continue
            k = ch.index(v)
            fields = []
            for label, stem, key in PANELS:
                d = got.get(stem)
                if d is None or key not in d:     # e.g. native has no control counterpart
                    continue
                fields.append((label, d[key][k][sl].astype(np.float32)))
            truth = dict(fields).get(TRUTH)
            if a.error:
                fields = [(l, f - truth) for l, f in fields if l != TRUTH]
                lim = np.percentile(np.abs(np.stack([f for _, f in fields])), 99)
                vmin, vmax, cmap = -lim, lim, "RdBu_r"
            else:
                vmin, vmax = np.percentile(truth, [1, 99])
                cmap = CMAP.get(v, "RdBu_r")

            n = len(fields); cols = int(np.ceil(n / 2)); rows = 2
            aspect = (sl[1].stop - sl[1].start if sl[1].start is not None else W) / \
                     max(1, (sl[0].stop - sl[0].start if sl[0].start is not None else H))
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.4, rows * 4.4 / max(aspect, .85)),
                                     constrained_layout=True)
            axes = np.atleast_1d(axes).ravel()
            for ax, (label, f) in zip(axes, fields):
                im = ax.imshow(f, extent=ext, origin="upper", aspect="auto",
                               cmap=cmap, vmin=vmin, vmax=vmax)
                t = label
                if truth is not None and label != TRUTH:
                    e = f if a.error else f - truth
                    t += f"   RMSE {np.sqrt(np.mean(e ** 2)):.4g}"
                ax.set_title(t, fontsize=9.5)
                ax.set_xticks([]); ax.set_yticks([])
            for ax in axes[n:]:
                ax.set_axis_off()
            fig.colorbar(im, ax=axes.tolist(), shrink=0.85, pad=0.012)
            fig.suptitle(f"{v} — {arm} regime{' — ERROR (pred − truth)' if a.error else ''}"
                         f" — HRES 1.5° → 0.25°, +24 h"
                         + ("  (zoom)" if a.zoom else "  (60°N–60°S)"), fontsize=12, y=1.03)
            p = out / f"nwp_{'err' if a.error else 'map'}_{v}_{arm}{'_zoom' if a.zoom else ''}.png"
            fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
            made += 1; print(f"  wrote {p.name}  ({n} panels)")
    print(f"{made} figures -> {out}")


if __name__ == "__main__":
    main()
