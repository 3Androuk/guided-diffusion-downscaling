import json, os
R = os.environ["PROJECTDIR"] + "/results_wb220"
ARMS = [("meanmap · regressor", "nwp_meanmap_best_ens1.json"),
        ("residual_lm · residual diffusion", "nwp_residual_lm_best_ens4.json"),
        ("residual_lm PCSTD (trained)", "nwp_residual_lm_pcstd_best_mtile_rtile_ens4.json"),
        ("SI_res_lm · transport", "nwp_stochastic_interpolant_res_lm_best_ens4.json"),
        ("no-geo · plain DDNM", "nwp_diffusion_best_ens4.json"),
        ("sinusoidal_static (best-effort) · plain DDNM", "nwp_diffusion_geo_sinstatic_best_ens4.json"),
        ("static · plain DDNM", "nwp_diffusion_geo_static_ens4.json"),
        ("hashcompact · plain DDNM", "nwp_diffusion_geo_hashcompact_ens4.json"),
        ("compactcombo · plain DDNM", "nwp_diffusion_geo_compactcombo_ens4.json"),
        ("no-geo · UNPROJECTED", "nwp_diffusion_best_noproj_ens4.json"),
        ("no-geo · Weather-DDNM", "nwp_diffusion_best_wddnm_ens4.json"),
        ("sinusoidal_static (best-effort) · UNPROJECTED", "nwp_diffusion_geo_sinstatic_best_noproj_ens4.json"),
        ("static · UNPROJECTED", "nwp_diffusion_geo_static_noproj_ens4.json"),
        ("compactcombo · UNPROJECTED", "nwp_diffusion_geo_compactcombo_noproj_ens4.json"),
        ("compactcombo · Weather-DDNM", "nwp_diffusion_geo_compactcombo_wddnm_ens4.json"),
        ("compactcombo · eta 1.0", "nwp_diffusion_geo_compactcombo_eta1_ens4.json"),
        ("compactcombo · eta 1.5", "nwp_diffusion_geo_compactcombo_eta1.5_ens4.json"),
        ("compactcombo · eta 1.0 NOPROJ", "nwp_diffusion_geo_compactcombo_eta1_noproj_ens4.json"),
        ("compactcombo · eta 1.5 NOPROJ", "nwp_diffusion_geo_compactcombo_eta1.5_noproj_ens4.json"),

        ("compactcombo · hydrostatic 0.5", "nwp_diffusion_geo_compactcombo_hyd0.5_ens4.json")]
# Skip arms whose run has not landed yet, so a queued arm can be registered above
# without breaking the rebuild.
D = {k: json.load(open(f"{R}/{f}")) for k, f in ARMS if os.path.exists(f"{R}/{f}")}
_absent = [k for k, f in ARMS if not os.path.exists(f"{R}/{f}")]
if _absent:
    print("  (not yet on disk, skipped: " + ", ".join(_absent) + ")")
ref = D["compactcombo · plain DDNM"]; ch = ref["channels"]
UNITS = {"t2m":"K","u10":"m/s","v10":"m/s","msl":"Pa","tcwv":"kg/m²","z500":"m²/s²","z700":"m²/s²","z850":"m²/s²",
         "t500":"K","t700":"K","t850":"K","u500":"m/s","u700":"m/s","u850":"m/s","v500":"m/s","v700":"m/s","v850":"m/s",
         "q500":"kg/kg","q700":"kg/kg","q850":"kg/kg"}
def fmt(v, c):
    return f"{v:.2e}" if c.startswith("q") else (f"{v:.1f}" if c.startswith(("z","msl")) else f"{v:.4f}")
out = []
out.append(f"# 20-var real-NWP test — full per-channel results, complete method ladder ({len(D)} arms)\n")
out.append(f"BriCS, `eval/downscale_nwp.py`. HRES 1.5° → 0.25° (ratio 6), {ref['n_inits']} inits at stride {ref['stride']} over 2016–17, "
           f"{ref['ensemble']} sampler members, tile {ref['tile']}/{ref['overlap']}, K={ref['recon']['K']} t_steps={ref['recon']['t_steps']}, η={ref['eta']}, fp32. "
           "Latitude-weighted RMSE in physical units. **Forecast** = coarsened +24 h HRES forecast → downscaled, scored vs the 0.25° HRES analysis; "
           "**native** = the 0.25° forecast scored directly; **control** = coarsened analysis → downscaled (pure downscaling error, no forecast error). "
           "Every arm shares inits, seeds, tiles and sampler settings; the three compactcombo rows differ *only* in the projection.\n")
for regime in ("forecast", "control"):
    out.append(f"\n## {regime.capitalize()} — lat-weighted RMSE\n")
    out.append("One row per **channel**, one column per **arm**. Values are root-mean-square error in the "
               "channel's own physical unit, averaged over the grid with cos(latitude) weights (so the "
               "poles do not count more than the tropics) and then over all inits. **Lower is better; "
               "the best arm in each row is bold.** `bicubic` is the interpolation baseline"
               + (" and `native` is the raw 0.25° HRES forecast scored directly — the target to beat."
                  if regime == "forecast" else
                  ". There is no `native` column here: the control regime starts from the analysis, so "
                  "the full-resolution analysis *is* the truth.")
               + " Compare channels only within a row — the units differ (z is m²/s², q is kg/kg).\n")
    hdr = "| channel | unit | bicubic |" + (" native |" if regime == "forecast" else "") + "".join(f" {k} |" for k in D)
    out.append(hdr); out.append("|---|---|---|" + ("---|" if regime == "forecast" else "") + "---|" * len(D))
    for i, c in enumerate(ch):
        a0 = ref["arms"][regime]; b = a0["bicubic"]["rmse_latweighted"][i]; n = a0.get("native", {}).get("rmse_latweighted", [None]*20)[i]
        row = f"| {c} | {UNITS[c]} | {fmt(b,c)} |" + (f" {fmt(n,c)} |" if regime == "forecast" else "")
        for k, d in D.items():
            m = d["arms"][regime]["model"]["rmse_latweighted"][i]
            best = min(dd["arms"][regime]["model"]["rmse_latweighted"][i] for dd in D.values())
            row += f" {'**' if abs(m-best)<1e-12 else ''}{fmt(m,c)}{'**' if abs(m-best)<1e-12 else ''} |"
        out.append(row)
    out.append(f"\n### {regime.capitalize()} — model vs bicubic (Δ%, negative = better) and vs native\n")
    out.append("The same numbers as percentage differences, which is what makes channels comparable. "
               "**Sign convention: negative = the arm has LESS error than the reference, i.e. better.** "
               "`vs bicubic` is `100·(rmse_arm/rmse_bicubic − 1)`"
               + (" and `vs native` the same against the full-resolution forecast; a negative `vs native` "
                  "means the downscaled 1.5° field beat the real 0.25° product on that channel.\n"
                  if regime == "forecast" else ".\n"))
    out.append("| channel |" + "".join(f" {k} vs bicubic | vs native |" if regime == "forecast" else f" {k} vs bicubic |" for k in D))
    out.append("|---|" + ("---|---|" if regime == "forecast" else "---|") * len(D))
    for i, c in enumerate(ch):
        a0 = ref["arms"][regime]; b = a0["bicubic"]["rmse_latweighted"][i]; n = a0.get("native", {}).get("rmse_latweighted", [None]*20)[i]
        row = f"| {c} |"
        for k, d in D.items():
            m = d["arms"][regime]["model"]["rmse_latweighted"][i]
            row += f" {100*(m/b-1):+.1f}% |" + (f" {100*(m/n-1):+.1f}% |" if regime == "forecast" else "")
        out.append(row)
    out.append(f"\n### {regime.capitalize()} — CRPS (models) vs MAE (deterministic references) and ensemble spread\n")
    out.append("**CRPS** (continuous ranked probability score) scores the whole 4-member ensemble rather "
               "than its mean: lower is better, same physical unit as the channel. For a single "
               "deterministic field CRPS collapses to MAE, so the reference columns and the ens-1 arms "
               "show MAE and the comparison stays fair. **spread** is the standard deviation across the "
               "4 members, the model's own claim about its uncertainty; a well-calibrated 4-member "
               "ensemble would have spread ≈ rmse·√(1+1/4) ≈ 1.12·rmse, so spread far below the RMSE "
               "column above means overconfident. Deterministic arms have spread 0 by construction. "
               "**Note:** unlike the RMSE tables, CRPS and spread are NOT latitude-weighted (≤5% per "
               "channel, reorders nothing, but do not mix the two into a ratio).\n")
    out.append("| channel | bicubic MAE |" + (" native MAE |" if regime == "forecast" else "") + "".join(f" {k} CRPS | spread |" for k in D))
    out.append("|---|---|" + ("---|" if regime == "forecast" else "") + "---|---|" * len(D))
    for i, c in enumerate(ch):
        a0 = ref["arms"][regime]; b = a0["bicubic"]["mae_latweighted"][i]; n = a0.get("native", {}).get("mae_latweighted", [None]*20)[i]
        row = f"| {c} | {fmt(b,c)} |" + (f" {fmt(n,c)} |" if regime == "forecast" else "")
        for k, d in D.items():
            m = d["arms"][regime]["model"]; row += (f" {fmt(m['crps'][i],c)} | {fmt(m['spread'][i],c)} |" if d['ensemble'] > 1 else f" {fmt(m['mae_latweighted'][i],c)} (MAE) | 0 |")
        out.append(row)
out.append("\n## Hydrostatic balance residual (RMS, m²/s²; read as |R − R_truth|)\n")
out.append("A **physics-consistency** check, not an accuracy score. The hypsometric equation ties the "
           "geopotential thickness between two pressure levels to the mean temperature of the layer; the "
           "residual is how far a field departs from that relation. **The target is the truth column, "
           "not zero** — the real atmosphere has a non-zero residual on this grid, so a field that "
           "scores far BELOW truth is over-smoothed rather than well-balanced. Bicubic sits low for "
           "exactly that reason. Read every column as distance from truth in either direction.\n")
out.append("| regime | layer | truth | bicubic | native | plain DDNM (8-init run) | Weather-DDNM | hydrostatic 0.5 | combo UNPROJ | static UNPROJ | sinstatic UNPROJ | no-geo UNPROJ | no-geo | sinusoidal_static | meanmap | residual_lm | SI_res_lm |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
E1 = json.load(open(f"{R}/nwp_diffusion_geo_compactcombo_ens1.json"))
for regime in ("control", "forecast"):
    for lay in ("850_700", "700_500"):
        h = D["compactcombo · hydrostatic 0.5"]["arms"][regime]; w = D["compactcombo · Weather-DDNM"]["arms"][regime]; e = E1["arms"][regime]
        nat = h.get("native", {}).get(f"hyd_rms_{lay}")
        out.append(f"| {regime} | {lay.replace('_','–')} | {h['bicubic']['truth_hyd_rms_'+lay]:.2f} | {h['bicubic']['hyd_rms_'+lay]:.2f} | "
                   f"{nat:.2f} |" if nat else f"| {regime} | {lay.replace('_','–')} | {h['bicubic']['truth_hyd_rms_'+lay]:.2f} | {h['bicubic']['hyd_rms_'+lay]:.2f} | — |")
        out[-1] += f" {e['model']['hyd_rms_'+lay]:.2f} (truth {e['bicubic']['truth_hyd_rms_'+lay]:.2f}) | {w['model']['hyd_rms_'+lay]:.2f} | {h['model']['hyd_rms_'+lay]:.2f} |"
        for k in ("compactcombo · UNPROJECTED", "static · UNPROJECTED", "sinusoidal_static (best-effort) · UNPROJECTED", "no-geo · UNPROJECTED", "no-geo · plain DDNM", "sinusoidal_static (best-effort) · plain DDNM", "meanmap · regressor", "residual_lm · residual diffusion", "SI_res_lm · transport"):
            _d = D.get(k)
            v = _d["arms"][regime]["model"].get("hyd_rms_" + lay) if _d else None
            out[-1] += f" {v:.2f} |" if v is not None else " — |"
out.append("\n## t2m spectrum error (`spectrum_log_l1`, display channel)\n")
out.append("Mean absolute difference between the arm's radially-averaged log power spectrum and the "
           "truth's, for 2-m temperature. **Lower is better; it measures fine-scale realism, which RMSE "
           "actively penalises.** An arm can have good RMSE and a poor spectrum by being smooth. Compare "
           "against `native`, which is a genuine 0.25° product and therefore the realistic floor.\n")
out.append("| regime | bicubic | native |" + "".join(f" {k} |" for k in D)); out.append("|---|---|---|" + "---|" * len(D))
for regime in ("forecast", "control"):
    a0 = ref["arms"][regime]; nat = a0.get("native", {}).get("spectrum_display")
    out.append(f"| {regime} | {a0['bicubic']['spectrum_display']:.4f} | {nat:.4f} |" if nat else f"| {regime} | {a0['bicubic']['spectrum_display']:.4f} | — |")
    out[-1] += "".join(f" {d['arms'][regime]['model']['spectrum_display']:.4f} |" for d in D.values())
# Compose the summary here so this file has exactly ONE writer (a separate
# --write pass used to race this rebuild and lose the section).
from eval.nwp_summary import render as _render_summary, summarise as _summarise
from pathlib import Path as _Path
_summary = _render_summary(_summarise(_Path(R))).split("\n")
# NB: entries are multi-line and several start with "\n", so lstrip before
# testing — the naive startswith matched nothing and appended the summary
# to the END of the document.
_i = next((k for k, ln in enumerate(out) if ln.lstrip("\n").startswith("## ")), len(out))
out = out[:_i] + _summary + [""] + out[_i:]
path = f"{R}/nwp_full_table.md"; open(path, "w").write("\n".join(out) + "\n"); print("wrote", path, len(out), "lines (summary included)")
# compact terminal view: forecast RMSE, all channels, 3 projection arms + refs
print("\nchannel   bicubic    native   plainDDNM   WeatherDDNM   hydro0.5   | static   hashcompact")
for i, c in enumerate(ch):
    a0 = ref["arms"]["forecast"]
    vals = [a0["bicubic"]["rmse_latweighted"][i], a0["native"]["rmse_latweighted"][i]] + [d["arms"]["forecast"]["model"]["rmse_latweighted"][i] for d in D.values()]
    f = (lambda v: f"{v:.2e}") if c.startswith("q") else ((lambda v: f"{v:8.1f}") if c.startswith(("z","msl")) else (lambda v: f"{v:8.4f}"))
    print(f"{c:6s} {f(vals[0]):>9s} {f(vals[1]):>9s} {f(vals[2]):>11s} {f(vals[3]):>13s} {f(vals[4]):>10s}   | {f(vals[5]):>8s} {f(vals[6]):>11s}")
