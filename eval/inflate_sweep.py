"""Does variance inflation on the PROJECTED ensemble recover the unprojected
arm's CRPS advantage — for free?

Inflating about the ensemble mean, x_i <- mu + alpha (x_i - mu), leaves mu
EXACTLY unchanged, so RMSE of the ensemble mean is invariant. Only CRPS and
spread move. If an alpha exists where the projected arm's CRPS matches the
unprojected arm's, then the unprojected arm's only genuine win is purchasable
without giving up any RMSE skill.

t2m only (that is the channel the driver saves members for). Unweighted, to
match eval/metrics.py.
"""
import os, re
from pathlib import Path
import numpy as np

R = Path(os.environ["PROJECTDIR"]) / "results_wb220/nwp_members"
ALPHAS = np.concatenate([np.arange(1.0, 3.01, 0.1), [3.5, 4.0]])
PAIRS = [("compactcombo", "diffusion_geo_compactcombo", "diffusion_geo_compactcombo_noproj"),
         ("static",       "diffusion_geo_static",       "diffusion_geo_static_noproj"),
         ("no-geo",       "diffusion_best",             "diffusion_best_noproj"),
         ("sinstatic",    "diffusion_geo_sinstatic_best", "diffusion_geo_sinstatic_best_noproj")]


def crps(p, t):
    """p (M,...) members, t (...) truth. Fair ensemble CRPS, same formula as
    eval/metrics.crps_ensemble."""
    m = p.shape[0]
    term1 = np.abs(p - t[None]).mean()
    s = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            s += np.abs(p[i] - p[j]).mean()
    return float(term1 - s / (m * (m - 1)))


def load(stem, regime):
    """tag -> (members, truth), keeping only genuine ensembles.

    nwp_members/<stem>/ is keyed by STEM, not by ensemble size, so an earlier
    --ensemble 1 run of the same checkpoint overwrote 8 of compactcombo's 32
    member files with single-member arrays. Those cannot carry a CRPS, so they
    are dropped here and the two arms of a pair are then intersected on tags,
    keeping the comparison paired on identical inits."""
    d = R / stem
    if not d.is_dir():
        return {}
    out = {}
    for f in sorted(d.glob(f"{regime}_*_members_t2m.npy")):
        tag = re.match(rf"{regime}_(.+)_members_t2m\.npy", f.name).group(1)
        tf = d / f"truth_{tag}_t2m.npy"
        if not tf.exists():
            continue
        m = np.load(f)
        if m.shape[0] < 2:            # an --ensemble 1 leftover
            continue
        out[tag] = (m.astype(np.float64), np.load(tf).astype(np.float64))
    return out


def stats(pairs, alpha=1.0):
    se = n = 0.0; cr = []; sp = []
    for p, t in pairs:
        mu = p.mean(0)
        q = mu + alpha * (p - mu) if alpha != 1.0 else p
        se += ((mu - t) ** 2).sum(); n += mu.size      # mu is alpha-invariant
        cr.append(crps(q, t)); sp.append(q.std(0).mean())
    return np.sqrt(se / n), float(np.mean(cr)), float(np.mean(sp))


for name, s_proj, s_nop in PAIRS:
    Pd, Nd = load(s_proj, "forecast"), load(s_nop, "forecast")
    common = sorted(set(Pd) & set(Nd))
    if not common:
        print(f"{name}: no paired inits (proj {len(Pd)}, noproj {len(Nd)})")
        continue
    P = [Pd[t] for t in common]; N = [Nd[t] for t in common]
    if len(common) < max(len(Pd), len(Nd)):
        print(f"  ({name}: paired on {len(common)} inits of {max(len(Pd), len(Nd))} "
              f"- the rest lack a 4-member file on one side)")
    rp, cp, spp = stats(P)
    rn, cn, spn = stats(N)
    print(f"\n=== {name} — 24 h forecast, t2m, {len(P)} inits ===")
    print(f"  projected     RMSE {rp:.4f}  CRPS {cp:.4f}  spread {spp:.4f}")
    print(f"  UNPROJECTED   RMSE {rn:.4f}  CRPS {cn:.4f}  spread {spn:.4f}")
    print(f"  -> dropping the projection: RMSE {100*(rn/rp-1):+.1f}%, CRPS {100*(cn/cp-1):+.1f}%")
    print(f"  inflation sweep on the PROJECTED ensemble (RMSE fixed at {rp:.4f} throughout):")
    best = (1e9, None); cross = None
    for a in ALPHAS:
        _, c, s = stats(P, a)
        if c < best[0]:
            best = (c, a)
        if cross is None and c <= cn:
            cross = a
        if abs(a - round(a, 1)) < 1e-9 and (abs(a * 10) % 3 == 0 or a in (1.0, 2.0, 3.0, 4.0)):
            print(f"    alpha {a:4.1f}  CRPS {c:.4f} ({100*(c/cp-1):+5.1f}%)  spread {s:.4f}")
    print(f"  BEST alpha {best[1]:.1f} -> CRPS {best[0]:.4f} "
          f"({100*(best[0]/cp-1):+.1f}% vs projected, {100*(best[0]/cn-1):+.1f}% vs UNPROJECTED)")
    print(f"  matches the unprojected CRPS at alpha "
          f"{cross:.1f}" if cross else "  never reaches the unprojected CRPS")
