"""Summarise the 20-variable NWP arms with a DEFENSIBLE multi-channel score.

Why this exists. The first summary used RMS-pooled normalized error,
    RMS(norm) = sqrt( mean_c (rmse_c/std_c)^2 ),
which weights each channel by the SQUARE of how hard it is, not by how much the
method changed it. Measured shares of that pool (bicubic, 24 h forecast):
q500 11.6%, v850 11.6%, v10 10.8%, v700 10.5% ... against t2m 2.0%, msl 1.3%,
z700 0.3%, z500 0.1%. Specific humidity and the v-winds are ~60% of it; the
surface and mass fields that the study is actually about are ~5%.

That inverts conclusions. Dropping the DDNM projection moves t2m +9.1% WORSE and
q500 2.4% better; under RMS pooling the q500 move alone is -24% of the total
change (its variance share is 100x z500's) and the pooled number IMPROVES 0.38%,
while under any per-channel-symmetric score the same run is clearly worse.

So the headline is now MEAN SKILL vs a common baseline,
    skill = mean_c ( 1 - rmse_c / rmse_bicubic_c ),
where every channel contributes its own relative improvement equally — the
WeatherBench-style framing, and the one that answers "did the method help?".
MEAN(norm) is kept as an absolute error level, RMS(norm) is kept but LABELLED as
variance-weighted, and surface/aloft are always split because the two groups
behave oppositely at 24 h (aloft, even the native forecast loses to bicubic).

    python -m eval.nwp_summary                 # print
    python -m eval.nwp_summary --write         # also prepend a section to nwp_full_table.md
"""
import argparse, json, os
from pathlib import Path

import numpy as np

ARMS = [("meanmap (regressor)", "nwp_meanmap_best_ens1.json"),
        ("residual_lm", "nwp_residual_lm_best_ens4.json"),
        ("residual_lm PCSTD (trained)", "nwp_residual_lm_pcstd_best_mtile_rtile_ens4.json"),
        ("SI_res_lm (transport)", "nwp_stochastic_interpolant_res_lm_best_ens4.json"),
        ("no-geo", "nwp_diffusion_best_ens4.json"),
        ("no-geo · NOPROJ", "nwp_diffusion_best_noproj_ens4.json"),
        ("no-geo · Weather-DDNM", "nwp_diffusion_best_wddnm_ens4.json"),
        ("sinusoidal_static*", "nwp_diffusion_geo_sinstatic_best_ens4.json"),
        ("sinusoidal_static* · NOPROJ", "nwp_diffusion_geo_sinstatic_best_noproj_ens4.json"),
        ("static", "nwp_diffusion_geo_static_ens4.json"),
        ("static · NOPROJ", "nwp_diffusion_geo_static_noproj_ens4.json"),
        ("hashcompact", "nwp_diffusion_geo_hashcompact_ens4.json"),
        ("compactcombo", "nwp_diffusion_geo_compactcombo_ens4.json"),
        ("compactcombo · NOPROJ", "nwp_diffusion_geo_compactcombo_noproj_ens4.json"),
        ("compactcombo · Weather-DDNM", "nwp_diffusion_geo_compactcombo_wddnm_ens4.json"),
        ("compactcombo · eta 1.0", "nwp_diffusion_geo_compactcombo_eta1_ens4.json"),
        ("compactcombo · eta 1.5", "nwp_diffusion_geo_compactcombo_eta1.5_ens4.json"),
        ("compactcombo · eta 1.0 NOPROJ", "nwp_diffusion_geo_compactcombo_eta1_noproj_ens4.json"),
        ("compactcombo · eta 1.5 NOPROJ", "nwp_diffusion_geo_compactcombo_eta1.5_noproj_ens4.json"),

        ("compactcombo · hydrostatic 0.5", "nwp_diffusion_geo_compactcombo_hyd0.5_ens4.json")]
SURFACE = ("t2m", "u10", "v10", "msl", "tcwv", "z850", "t850")
MARKER = "## Multi-channel summary"   # the section --write owns and replaces


def summarise(root: Path):
    std = np.load(root.parent / "datasets/patches_wb220/norm_stats.npz")["std"].reshape(-1)
    D, missing = {}, []
    for name, f in ARMS:
        p = root / f
        if p.exists():
            D[name] = json.load(open(p))
        else:
            missing.append(f)
    if not D:
        raise SystemExit(f"no result files under {root}")
    if missing:
        print(f"  (skipped {len(missing)} absent: {', '.join(missing)})")
    ref = next(iter(D.values())); ch = list(ref["channels"])
    si = [ch.index(c) for c in SURFACE]; ai = [i for i in range(len(ch)) if i not in si]
    out = []
    for regime in ("forecast", "control"):
        bic = np.asarray(ref["arms"][regime]["bicubic"]["rmse_latweighted"])
        rows = []

        bic_mae = np.asarray(ref["arms"][regime]["bicubic"]["mae_latweighted"])
        nat_e = ref["arms"][regime].get("native")
        nat_mae = np.asarray(nat_e["mae_latweighted"]) if nat_e else None

        def row(label, r, prob=None, spread=None, ens=None):
            """r = lat-weighted RMSE per channel; prob = CRPS (or MAE for a
            deterministic arm, which is what CRPS reduces to); spread = ensemble
            spread. All skills are mean_c(1 - x_c / baseline_c) so every channel
            counts equally by its own relative improvement."""
            n = r / std
            # CRPS is reported as a VALUE, not a skill ratio. It is computed
            # UNWEIGHTED in eval/metrics.py (plain pixel mean) whereas rmse/mae are
            # cos(lat)-weighted, so a skill ratio against mae_latweighted would mix
            # two spatial weightings. Measured effect of the weighting is <=5% per
            # channel (q500 0.949, z500 1.045), so it does not reorder arms, but a
            # ratio is not clean and a value is.
            cs = float(np.mean(prob / std)) if prob is not None else None   # pooled, normalized
            cn = float(prob[ch.index("t2m")]) if prob is not None else None  # t2m, Kelvin
            # calibration: a reliable M-member ensemble has spread = err * sqrt(1+1/M)
            cal = (float(np.mean(spread / (r * np.sqrt(1 + 1 / ens))))
                   if (spread is not None and ens and ens > 1) else None)
            rows.append((label,
                         100 * float(np.mean(1 - r / bic)),          # RMSE skill vs bicubic
                         100 * float(np.mean(1 - r[si] / bic[si])),  # ... surface
                         100 * float(np.mean(1 - r[ai] / bic[ai])),  # ... aloft
                         cs, cn, cal,
                         float(np.mean(n)),                          # mean normalized
                         float(np.sqrt(np.mean(n ** 2))),            # RMS normalized (variance-weighted)
                         ens))
        row("bicubic (from 1.5°)", bic, bic_mae)
        if nat_e:
            row("native 0.25° forecast", np.asarray(nat_e["rmse_latweighted"]), nat_mae)
        for name, d in D.items():
            m = d["arms"][regime]["model"]
            ens = d["ensemble"]
            prob = np.asarray(m["crps"]) if ens > 1 else np.asarray(m["mae_latweighted"])
            row(name, np.asarray(m["rmse_latweighted"]), prob,
                np.asarray(m["spread"]) if ens > 1 else None, ens)
        out.append((regime, rows))
    return out


def render(out):
    L = []
    L.append(f"{MARKER} — mean skill vs bicubic (headline)\n")
    L.append("### How to read this table\n")
    L.append("One **row** is one *arm*: one reconstruction method, scored on the same 32 initialisation")
    L.append("times, with the same tiles, seeds, ensemble size and sampler settings as every other row.")
    L.append("Rows therefore differ only by the thing named in the label.\n")
    L.append("Label suffixes: **NOPROJ** = the DDNM projection removed entirely, so the coarse field")
    L.append("reaches the sampler only through the noise-mixing initialisation. **Weather-DDNM** =")
    L.append("the covariance-weighted projection `C Aᵀ(A C Aᵀ)⁻¹` in place of the plain pseudo-inverse")
    L.append("`A†`. **hydrostatic C** = a hypsometric-balance correction applied to each x₀ estimate")
    L.append("before the projection. An arm with no suffix uses the plain per-step DDNM projection.\n")
    L.append("Two **regimes**, reported as separate blocks:\n")
    L.append("* **Forecast** — the +24 h HRES forecast, coarsened to 1.5°, downscaled to 0.25°, and")
    L.append("  scored against the 0.25° HRES *analysis*. This is the deployment test: the input")
    L.append("  carries genuine forecast error, not just a loss of resolution.")
    L.append("* **Control** — the 0.25° analysis coarsened to 1.5° and downscaled straight back. Same")
    L.append("  machinery, no forecast error, so it isolates *downscaling* skill on its own.\n")
    L.append("Two rows are references rather than methods. **bicubic (from 1.5°)** is the interpolation")
    L.append("baseline every skill score is measured against, so it is +0.0% by construction.")
    L.append("**native 0.25° forecast** is the raw HRES forecast at full resolution — the target to beat,")
    L.append("and the honest ceiling for this task.\n")
    L.append("### What each column means\n")
    L.append("| column | definition | how to read it |")
    L.append("|---|---|---|")
    L.append("| **RMSE skill** | `mean_c (1 − rmse_c / rmse_bicubic_c)`, over all 20 channels | **The headline number.** Positive = better than bicubic. Each channel contributes its own *relative* improvement and all 20 count equally. |")
    L.append("| surface | the same average over the 7 surface/near-surface channels: t2m, u10, v10, msl, tcwv, z850, t850 | Where downscaling skill actually lives. Quote this alongside the headline. |")
    L.append("| aloft | the same average over the other 13 channels (500/700 hPa fields) | At +24 h even the **native** forecast scores negative here: its fine scales are physically real but misplaced, so RMSE double-penalises them. A ≈0% aloft score is the expected result, not a failure. |")
    L.append("| **CRPS (norm)** | `mean_c (crps_c / std_c)` — a **value**, not a skill ratio | Lower is better. Probabilistic accuracy of the 4-member ensemble, each channel divided by its own standard deviation so they are comparable. For the deterministic rows (bicubic, native, meanmap) CRPS reduces exactly to MAE, so every row is on one footing. |")
    L.append("| CRPS t2m (K) | the same quantity for 2-m temperature, left in kelvin | The one column in physical units you can sanity-check by eye. Note it can **disagree in sign** with the pooled CRPS — see the caveat below. |")
    L.append("| spread/reliable | `mean_c spread_c / (rmse_c·√(1+1/M))` with M = 4 members | **1.0 = calibrated.** Below 1 the ensemble is underdispersed (too confident); above 1, overdispersed. Every projected arm sits near 0.16 — roughly 6× too narrow — because members differ only through the noise-mixing draw at η=0 and the projection then pins them all to the same coarse field. |")
    L.append("| MEAN(norm) | `mean_c (rmse_c / std_c)` | Absolute error level, in units of each channel's own variability. Lower is better. Needs no baseline, so it is the one column that stays meaningful if the bicubic reference ever changes. |")
    L.append("| RMS(norm) | `sqrt(mean_c (rmse_c / std_c)²)` | **Variance-weighted — never use this as a headline.** Squaring means each channel is weighted by how *hard* it is, not by how much the method moved it: ~60% of this number is specific humidity and the v-winds, while t2m + msl + z together are ~5%. It once inverted the projection conclusion outright. Retained only for continuity with earlier tables. |")
    L.append("")
    L.append("**Which number to quote.** RMSE skill with the surface/aloft split for accuracy, CRPS (norm)")
    L.append("for probabilistic quality, spread/reliable for calibration. They answer different questions and")
    L.append("routinely disagree: dropping the projection *costs* RMSE skill but *gains* CRPS, because what")
    L.append("it buys is dispersion, not accuracy. Report all three rather than picking the flattering one.\n")
    L.append("**Caveat — one known inconsistency.** RMSE and MAE are cos(lat)-weighted; CRPS and spread are")
    L.append("**not** (plain pixel mean). Measured, the weighting moves a per-channel value by at most ~5%")
    L.append("(q500 ×0.95, z500 ×1.05) and reorders no arm, but it means a CRPS *ratio* against the weighted")
    L.append("MAE would mix two spatial weightings — which is why CRPS is reported as a value. Do not form")
    L.append("CRPS skill scores until `crps_per_channel` accepts latitude weights.\n")
    L.append("**Caveat — pooled and t2m CRPS can disagree.** On the projection question the pooled value")
    L.append("prefers the unprojected arms while t2m prefers the projected ones. That is the same")
    L.append("channel-weighting split as the RMSE story, so quote both columns, never one alone.\n")
    for regime, rows in out:
        L.append(f"\n### {regime.capitalize()}\n")
        L.append("| arm | **RMSE skill** | surface | aloft | **CRPS (norm)** | CRPS t2m (K) | spread/reliable | MEAN(norm) | RMS(norm) |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for label, skill, ss, sa, cs, cn, cal, mn, rms, ens in rows:
            tag = "" if ens in (None, 4) else f" (ens {ens})"
            b = "**" if label.startswith(("compactcombo", "native")) and "NOPROJ" not in label else ""
            g = lambda v, n=4: "—" if v is None else f"{v:.{n}f}"
            calf = "—" if cal is None else f"{cal:.2f}"
            L.append(f"| {label}{tag} | {b}{skill:+.1f}%{b} | {ss:+.1f}% | {sa:+.1f}% | {b}{g(cs)}{b} | {g(cn)} | {calf} | {mn:.4f} | {rms:.4f} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("PROJECTDIR", "") + "/results_wb220")
    ap.add_argument("--write", action="store_true",
                    help="replace the summary section in nwp_full_table.md (idempotent)")
    a = ap.parse_args()
    root = Path(a.root)
    text = render(summarise(root))
    print(text)
    if a.write:
        p = root / "nwp_full_table.md"
        lines = p.read_text().split("\n")
        # Drop EVERY existing summary section, then insert one. The first version
        # of this prepended blindly, so three stale copies accumulated in the file
        # and a reader could land on an obsolete one (they differed: skill-%, then
        # CRPS-as-%, then CRPS-as-value). Idempotent now: run it as often as you like.
        out, skipping, dropped = [], False, 0
        for ln in lines:
            if ln.startswith(MARKER):
                skipping = True
                dropped += 1
                continue
            if skipping:
                # a summary section ends at the next top-level "## " that is not ours
                if ln.startswith("## "):
                    skipping = False
                else:
                    continue
            out.append(ln)
        # insert after the H1 title block, before the first remaining section
        i = next((k for k, ln in enumerate(out) if ln.startswith("## ")), len(out))
        body = out[:i] + text.split("\n") + [""] + out[i:]
        p.write_text("\n".join(body).rstrip("\n") + "\n")
        print(f"\n  -> summary written into {p} (replaced {dropped} stale cop"
              f"{'y' if dropped == 1 else 'ies'})")


if __name__ == "__main__":
    main()
