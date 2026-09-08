"""Deployment test on REAL NWP fields (20-var): downscale HRES, score vs HRES analysis.

Consumes data/fetch_nwp.py output (datasets/nwp_hres). Two arms, both built
with OUR block-average operator so the per-step projection is exact and no
grid registration is assumed:

  control    coarsen(truth_fine)      -> downscale -> vs truth_fine
             downscaling error only; also measures the representation shift
             (models were trained on ERA5, this is the ECMWF analysis)
  forecast   coarsen(forecast24_fine) -> downscale -> vs truth_fine
             + 24 h forecast error on top

The WB2 1.5 deg products (analysis_coarse / forecast24_coarse) are NOT used as
inputs here: their cell centres sit at multiples of 1.5 deg, half a coarse cell
off the block centres of the 0.25 deg grid, so they are not a block average of
the fine field on any registration and the exact-consistency constraint would
be enforcing something the data does not satisfy. That arm needs a deliberate
registration choice first.

ORIENTATION: the WB2 hres_t0 0.25 deg store is latitude-ASCENDING (-60..60);
every model, coords_full.npz, static_fields.npz and the HEALPix index are
latitude-DESCENDING (60..-60, ERA5 order). The loader flips and then ASSERTS
against coords_full — feeding the arrays unflipped runs silently and scores
garbage (verified 2026-09-02: 60N seasonal std 9.2 K sat at row 480).

Run (1 GPU):
    python -m eval.downscale_nwp --config config/wb2_20var.yaml \
        --ckpt diffusion_geo_static.pt --limit 32 --ensemble 4
    python -m eval.downscale_nwp --ckpt bicubic --limit 32      # baseline, CPU ok
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import load_norm_stats  # noqa: E402
from data.degrade import coarsen, upsample_nearest  # noqa: E402
from eval.metrics import (crps_per_channel, spectrum_log_l1,  # noqa: E402
                          spectrum_log_l1_per_channel)
from utils import channel_specs, ensure_dir, load_config  # noqa: E402

ARMS = {"control": "truth_fine.npy", "forecast": "forecast24_fine.npy"}


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_oriented(data_dir: Path, patch_dir: Path, ratio: int):
    """Memmaps + the row/col crop, with latitude flipped to training order.

    Returns dict(arrays..., lat, lon, flip, rows, cols). Arrays stay memmapped;
    callers index one timestep at a time and apply `orient()`."""
    tc = np.load(data_dir / "truth_coords.npz")
    cf = np.load(patch_dir / "coords_full.npz")
    lat, lon = tc["lat"], tc["lon"]
    flip = bool(lat[0] < lat[-1])                      # ascending -> needs flip
    if flip:
        lat = lat[::-1]
    rows = (len(lat) // ratio) * ratio                 # 481 -> 480
    cols = (len(lon) // ratio) * ratio                 # 1440
    lat, lon = lat[:rows], lon[:cols]
    # The models' geo assets are cropped [:rows, :cols] from the TOP of the
    # training grid, so the oriented NWP grid must match it exactly there.
    if not (np.allclose(lat, cf["lat"][:rows]) and np.allclose(lon, cf["lon"][:cols])):
        raise SystemExit(
            f"NWP grid does not match the training grid after orientation: "
            f"nwp lat {lat[0]}..{lat[-1]} vs train {cf['lat'][0]}..{cf['lat'][rows-1]}")
    arrays = {k: np.load(data_dir / v, mmap_mode="r") for k, v in ARMS.items()}
    times = np.load(data_dir / "times.npy")
    return dict(arrays=arrays, times=times, lat=lat, lon=lon, flip=flip,
                rows=rows, cols=cols)


def orient(field: np.ndarray, d: dict) -> np.ndarray:
    """(C, H, W) memmap slice -> contiguous float32 in training orientation."""
    x = np.asarray(field, dtype=np.float32)
    if d["flip"]:
        x = x[:, ::-1, :]
    return np.ascontiguousarray(x[:, : d["rows"], : d["cols"]])


# ----------------------------------------------------------------------------
# geo payload — covers the BriCS encoders eval.full_field._geo_full predates
# ----------------------------------------------------------------------------
def geo_payload(cfg_ck: dict, patch_dir: Path, hw, device):
    g = cfg_ck.get("geo", {})
    if not g.get("enabled", False):
        return None
    enc = g.get("encoder", "hash")
    if enc in ("healpix", "static", "hash_static", "hash", "xyz", "sinusoidal"):
        from eval.full_field import _geo_full
        return _geo_full(cfg_ck, patch_dir, hw, device)
    # hash2d / hash_compact / hash_compact_static: same coordinate payload as
    # the hash family, honouring input_dim (2 for the plate-carree chart), with
    # the static fields concatenated for the *_static combo.
    from models.geo_encoding import build_patch_coords
    h, w = hw
    cf = np.load(patch_dir / "coords_full.npz")
    # hash2d is 2-D by definition; the config's input_dim stays 3 (dataset.py does the same).
    d = 2 if enc == "hash2d" else int(g.get("input_dim", 3))
    alt = g.get("altitude") if d == 4 else None
    coords = torch.from_numpy(build_patch_coords(
        cf["lat"][:h], cf["lon"][:w], altitude=alt, input_dim=d))
    if enc.endswith("_static"):
        sf = np.load(patch_dir / "static_fields.npz")
        static = torch.from_numpy(np.ascontiguousarray(sf["fields"][:, :h, :w]))
        coords = torch.cat([coords, static.permute(1, 2, 0)], dim=-1)
    return coords.to(device)


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
R_D, EPS_Q = 287.05, 0.608


def hydro_residual_rms(field_phys: torch.Tensor, labels) -> dict:
    """Hypsometric residual Phi_u - Phi_l - R_d*Tv_bar*ln(p_l/p_u), RMS in m2/s2,
    for the 850-700 and 700-500 layers of a (C,H,W) field in PHYSICAL units.
    Same formula as models/hydrostatic_constraint.py. Read as |R - R_truth|:
    ERA5/HRES carry their own ~40-60 m2/s2 floor from the two-level Tv_bar."""
    out = {}
    for lo, up in ((850, 700), (700, 500)):
        try:
            zl, zu = field_phys[labels.index(f"z{lo}")], field_phys[labels.index(f"z{up}")]
            tl, tu = field_phys[labels.index(f"t{lo}")], field_phys[labels.index(f"t{up}")]
            ql, qu = field_phys[labels.index(f"q{lo}")], field_phys[labels.index(f"q{up}")]
        except ValueError:
            return out
        tv = 0.5 * (tl * (1 + EPS_Q * ql) + tu * (1 + EPS_Q * qu))
        r = (zu - zl) - R_D * tv * float(np.log(lo / up))
        out[f"hyd_rms_{lo}_{up}"] = float(torch.sqrt((r ** 2).mean()))
    return out


def lat_weights(lat_deg: np.ndarray, device) -> torch.Tensor:
    w = np.cos(np.deg2rad(lat_deg)).astype(np.float32)
    w = w / w.mean()
    return torch.from_numpy(w).to(device)[None, None, :, None]   # (1,1,H,1)


def per_channel_scores(pred: torch.Tensor, truth: torch.Tensor, w: torch.Tensor):
    """pred/truth (C,H,W) physical units -> dict of per-channel lists."""
    err = (pred - truth)[None]
    rmse_w = torch.sqrt((w * err ** 2).mean(dim=(0, 2, 3)))
    rmse = torch.sqrt((err ** 2).mean(dim=(0, 2, 3)))
    mae_w = (w * err.abs()).mean(dim=(0, 2, 3))
    return {"rmse_latweighted": rmse_w.tolist(), "rmse": rmse.tolist(),
            "mae_latweighted": mae_w.tolist()}


class Accum:
    def __init__(self):
        self.sums, self.n = {}, 0

    def add(self, d: dict):
        for k, v in d.items():
            v = np.asarray(v, dtype=np.float64)
            self.sums[k] = self.sums.get(k, 0.0) + v
        self.n += 1

    def mean(self) -> dict:
        return {k: (v / self.n).tolist() if np.ndim(v) else float(v / self.n)
                for k, v in self.sums.items()}


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/wb2_20var.yaml")
    ap.add_argument("--ckpt", default="diffusion.pt",
                    help="checkpoint name in paths.ckpt_dir, or 'bicubic'")
    ap.add_argument("--data-dir", default=None,
                    help="fetch_nwp.py output (default: <raw_dir>/../nwp_hres)")
    ap.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    ap.add_argument("--limit", type=int, default=None, help="max inits (of 256)")
    ap.add_argument("--stride", type=int, default=1,
                    help="take every k-th init (spreads --limit over the period)")
    ap.add_argument("--shard", default=None,
                    help="I/N: run only every N-th of the selected inits, starting at I "
                         "(0-based). N separate 1-GPU jobs finish in wall/N at the same "
                         "total cost; the JSON gains _shardIofN and is merged exactly by "
                         "eval.merge_nwp_shards. Members are saved per init, unsharded.")
    ap.add_argument("--ensemble", type=int, default=1, help="sampler members")
    ap.add_argument("--eta", type=float, default=None)
    ap.add_argument("--ratio", type=int, default=6, help="1.5 deg / 0.25 deg")
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--t-steps", type=int, nargs="+", default=None,
                    help="ratio 6 is not in config.sample.reconstructions; "
                         "default K=1 t_steps=[240] (between the r4 and r8 entries)")
    ap.add_argument("--smooth-sigma", type=float, default=0.0)
    # Tile origins are snapped to multiples of the ratio so each tile's coarse
    # observation is an exact crop of the global one; tile and stride must
    # therefore be multiples of 6. 128 is not (the t2m demo hit the same
    # wall and used 144); 144 = 6*24 = 16*9 also divides the UNet's stride.
    ap.add_argument("--tile", type=int, default=144)
    ap.add_argument("--overlap", type=int, default=36)
    ap.add_argument("--batch-tiles", type=int, default=75,
                    help="tiles per forward (cap); 480x1440 at tile 144/overlap 36 is 5x13=65, "
                         "so 75 = one forward per diffusion step (fits a 96 GB GH200 in fp32)")
    ap.add_argument("--hydrostatic", type=float, default=0.0,
                    help="inference-time hypsometric (hydrostatic) correction of x0 at "
                         "every DDIM step, applied BEFORE the projection so it stays in "
                         "ker A (0 = off; 0.5 = the operating point measured on patches: "
                         "-14.5%% spectrum for +1%% L2, saturating by 1.0). Output stem "
                         "gains _hyd<coef>.")
    ap.add_argument("--covariance", default=None,
                    help="Weather-DDNM: spectral covariance npz whose grid equals "
                         "--tile. Replaces the pixel-space DDNM correction A^T with "
                         "C A^T (A C A^T)^-1 — the observed block means are still "
                         "exact, only the spread of the correction changes. "
                         "Output stem gains _wddnm.")
    ap.add_argument("--localization-radius", type=float, default=None,
                    help="Gaspari-Cohn taper half-width in pixels applied at load "
                         "time, so the periodic embedding cannot wrap the "
                         "correction across the tile edge.")
    ap.add_argument("--no-project", dest="no_project", action="store_true",
                    help="Drop the DDNM projection entirely (no per-step and no final "
                         "block-mean re-pin): the coarse field steers the chain only "
                         "through the noise-mixing initialization. lambda=0 in the DDNM+ "
                         "family. Output stem gains _noproj.")
    ap.add_argument("--unconditional", action="store_true",
                    help="UNCONDITIONAL DDNM: start the chain from pure noise at "
                         "t=T-1 with a ZERO guidance field, so the coarse input "
                         "enters ONLY through the per-step projection. This is the "
                         "original zero-shot DDNM setting (Wang et al. 2023); the "
                         "default arms instead seed the chain by noise-mixing the "
                         "upsampled coarse field at t=240 AND project. Isolates how "
                         "much of the skill comes from the projection alone. "
                         "Output stem gains _uncond.")
    ap.add_argument("--ddim-stride", dest="ddim_stride", type=int, default=1,
                    help="take every k-th timestep of the DDIM subsequence. The "
                         "guided arms run t=240..0 at stride 1 = 240 net function "
                         "evaluations; an unconditional chain spans t=999..0, so "
                         "--ddim-stride 4 gives 250 NFE and matches the guided arms' "
                         "compute budget instead of costing 4.2x.")
    ap.add_argument("--directmap-full", dest="directmap_full", action="store_true",
                    help="apply a directmap checkpoint to the whole field in one "
                         "forward pass instead of tiling it. The residual and "
                         "transport arms call their frozen mean this way, so this "
                         "is the setting that makes meanmap comparable to the mean "
                         "INSIDE them. Output stem gains _full.")
    ap.add_argument("--mean-tiled", dest="mean_tiled", action="store_true",
                    help="apply the frozen learned mean of the residual and "
                         "transport arms TILED, the way the standalone regression "
                         "arm is applied, instead of one full-field forward pass. "
                         "The mean was trained on 128px patches, so the full-field "
                         "pass is off-distribution for it AND for the residual "
                         "model that was trained to correct it. Off by default so "
                         "existing results reproduce. Output stem gains _mtile.")
    ap.add_argument("--inflation-alphas", dest="inflation_alphas", type=float,
                    nargs="+", default=None,
                    help="variance-inflation factors to score, x_i <- mu + a(x_i - mu). "
                         "The ensemble MEAN is unchanged by this, so rmse is identical "
                         "at every alpha; only crps and spread move. Records "
                         "crps_infl<a> / spread_infl<a> per channel. Needs --ensemble>1. "
                         "Output stem gains _infl so the un-swept JSON is untouched.")
    ap.add_argument("--residual-tiled", dest="residual_tiled", action="store_true",
                    help="sample the residual arm's residual TILED (the transport "
                         "arm already is), instead of one unconditional pass over "
                         "the whole 480x1440 field. The residual model is trained "
                         "on 128px patches, so the full-field pass is 42x its "
                         "training area. Off by default so existing results "
                         "reproduce. Output stem gains _rtile.")
    ap.add_argument("--project-directmap", dest="project_directmap",
                    action="store_true",
                    help="apply the final range-null projection to the directmap "
                         "output, so the regression arm satisfies coarsen(x) == y "
                         "like every other arm does. Off by default so existing "
                         "results reproduce. Output stem gains _dmproj.")
    ap.add_argument("--snapshot", action="store_true",
                    help="dump the full 20-channel fields of the FIRST init to "
                         "<results_dir>/nwp_snapshots/<stem>_<arm>_<time>.npz "
                         "(truth, bicubic, native, ensemble mean, member 0; fp16) "
                         "and stop. The routine --save-members path keeps only the "
                         "display channel, so this is what any non-t2m figure needs. "
                         "Pair with --limit 1 --ensemble 1 --arms forecast: one "
                         "reconstruction, ~1 min of GPU.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-members", dest="save_members", action="store_true", default=True,
                    help="save the display-channel ensemble members per init (fp16 .npy, "
                         "~5 MB per init per arm) under <results_dir>/nwp_members/<stem>/ so "
                         "CRPS and variance-inflation sweeps can be run offline")
    ap.add_argument("--no-save-members", dest="save_members", action="store_false")
    args = ap.parse_args()

    cfg = load_config(args.config)
    specs = channel_specs(cfg["data"])
    labels = [f"{s['name']}@{s['level']}" if s["level"] else s["name"] for s in specs]
    short = {"2m_temperature": "t2m", "10m_u_component_of_wind": "u10",
             "10m_v_component_of_wind": "v10", "mean_sea_level_pressure": "msl",
             "total_column_water_vapour": "tcwv", "geopotential": "z",
             "temperature": "t", "u_component_of_wind": "u",
             "v_component_of_wind": "v", "specific_humidity": "q"}
    labels = [short.get(s["name"], s["name"]) + (str(s["level"]) if s["level"] else "")
              for s in specs]
    disp = labels.index("t2m") if "t2m" in labels else 0

    patch_dir = Path(cfg["paths"]["patch_dir"])
    data_dir = Path(args.data_dir) if args.data_dir else \
        Path(cfg["paths"]["raw_dir"]).parent / "nwp_hres"
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ratio = args.ratio

    d = load_oriented(data_dir, patch_dir, ratio)
    H, W = d["rows"], d["cols"]
    idx = np.arange(0, len(d["times"]), args.stride)
    if args.limit:
        idx = idx[: args.limit]
    shard_tag = ""
    if args.shard:
        si, sn = (int(x) for x in args.shard.split("/"))
        if not (0 <= si < sn):
            raise SystemExit(f"--shard {args.shard}: need 0 <= I < N")
        n_all = len(idx)
        idx = idx[si::sn]
        shard_tag = f"_shard{si}of{sn}"
        print(f"  shard {si}/{sn}: {len(idx)} of {n_all} selected inits", flush=True)
    normalizer = load_norm_stats(patch_dir)
    w = lat_weights(d["lat"], device)
    print(f"NWP data {data_dir} | {len(idx)} inits (stride {args.stride}) | "
          f"grid {H}x{W} (lat {d['lat'][0]:.2f}..{d['lat'][-1]:.2f}, "
          f"flipped={d['flip']}) | ratio {ratio} | arms {args.arms}", flush=True)

    # ---- model ------------------------------------------------------------
    use_model = args.ckpt != "bicubic"
    if args.no_project and (args.covariance or args.hydrostatic):
        raise SystemExit("--no-project cannot be combined with --covariance or "
                         "--hydrostatic: both act inside the projection step")
    if args.unconditional and args.no_project:
        raise SystemExit("--unconditional with --no-project has NO data channel at "
                         "all: pure noise in, an unconditional prior sample out. "
                         "The unconditional arm exists to isolate the projection, "
                         "so the projection must stay on.")
    # eta MUST be in the stem: it is a sampler setting, not a model, so without
    # it an --eta run silently overwrites the eta=0 result file of the same
    # checkpoint. Every arm in the 20-var table was produced at eta=0.
    # _snap MUST be in the stem. --snapshot runs one init at --ensemble 1, and
    # without a tag it writes to nwp_<ckpt>_ens1.json -- which is where the
    # 8-init hydrostatic balance runs live. Measured 2026-09-05: a snapshot
    # clobbered compactcombo's balance run (43.9 over 8 inits -> 38.88 over one
    # January day) and the full table silently reported the wrong number.
    stem = (Path(args.ckpt).stem
            + (f"_r{ratio}" if ratio != 6 else "")   # 6 = the native 1.5deg->0.25deg case
            + ("_full" if args.directmap_full else "")
            + ("_mtile" if args.mean_tiled else "")
            + ("_rtile" if args.residual_tiled else "")
            + ("_infl" if args.inflation_alphas else "")
            + ("_dmproj" if args.project_directmap else "")
            + ("_snap" if args.snapshot else "")
            + (f"_eta{args.eta:g}" if args.eta is not None else "")
            + ("_uncond" if args.unconditional else "")
            + (f"_hyd{args.hydrostatic:g}" if args.hydrostatic else "")
            + ("_wddnm" if args.covariance else "")
            + ("_noproj" if args.no_project else ""))
    if use_model:
        from sample.full_field import (reconstruct_full_tiled_diffusion,
                                       reconstruct_full_tiled_directmap,
                                       reconstruct_full_tiled_residual,
                                       reconstruct_full_tiled_transport)
        from sample.reconstruct import load_diffusion, load_directmap, load_residual
        from models.residual import res_scale
        from sample.transport import load_transport
        ck_path = Path(cfg["paths"]["ckpt_dir"]) / args.ckpt
        # ---- dispatch on checkpoint kind (ported from eval/downscale_forecast.py)
        process = method = None
        residual_t = None                      # residual-transport payload
        res_std = res_mean_model = None; res_mean_geo = False
        if stem.startswith(("flow_matching", "stochastic_interpolant")):
            kind = "transport"
            model, process, cfg_ck, method, residual_t = load_transport(ck_path, device)
            diffusion = None
            if residual_t is not None and residual_t["mean_geo"]:
                raise SystemExit(f"{args.ckpt} uses a GEO-conditioned frozen mean; "
                                 "its coords payload is not built here.")
        elif stem.startswith("residual"):
            kind = "residual"
            (model, diffusion, cfg_ck, res_std,
             res_mean_model, res_mean_geo) = load_residual(ck_path, device)
            if res_mean_geo:
                raise SystemExit(f"{args.ckpt} has a GEO-conditioned mean; coords "
                                 "payload for it is not built here.")
        elif stem.startswith(("directmap", "meanmap")):
            kind = "directmap"
            model, cfg_ck = load_directmap(ck_path, device)
            diffusion = None
            if args.ensemble > 1:
                print("  direct map is deterministic -> forcing --ensemble 1", flush=True)
                args.ensemble = 1
        else:
            kind = "diffusion"
            model, diffusion, cfg_ck = load_diffusion(ck_path, device)
        if kind != "diffusion" and (args.covariance or args.hydrostatic):
            raise SystemExit("--covariance / --hydrostatic act inside the DDIM projection "
                             f"and only apply to guided diffusion, not {kind}")
        print(f"  checkpoint kind: {kind}", flush=True)
        geo = geo_payload(cfg_ck, patch_dir, (H, W), device)
        cov_projector = None
        if args.covariance:
            from sample.weather_ddnm import SpectralCovarianceProjector  # noqa: PLC0415
            cov_projector = SpectralCovarianceProjector.from_npz(
                args.covariance, localization_radius=args.localization_radius)
            if tuple(cov_projector.image_size) != (args.tile, args.tile):
                raise SystemExit(
                    f"covariance grid {cov_projector.image_size} != tile "
                    f"{(args.tile, args.tile)}: run "
                    f"data.resample_covariance --size {args.tile}")
            cov_projector = cov_projector.to(device)
            print(f"  Weather-DDNM covariance ON: {args.covariance} "
                  f"grid {cov_projector.image_size} channels {cov_projector.channels}"
                  + (f", localization {args.localization_radius}px"
                     if args.localization_radius else ""), flush=True)
        if args.no_project:
            print("  projection OFF: noise-mixing guidance only (lambda=0)", flush=True)
        if args.unconditional:
            print("  UNCONDITIONAL: zero guidance, pure-noise start; the coarse "
                  "field enters only through the projection", flush=True)
        hydro = None
        if args.hydrostatic:
            from models.hydrostatic_constraint import HydrostaticProjection
            hydro = HydrostaticProjection(cfg_ck, normalizer, coef=args.hydrostatic,
                                          device=device)
            print(f"  hydrostatic constraint ON, coef={args.hydrostatic:g} "
                  f"(physics-first, then projection)", flush=True)
        recons = cfg_ck["sample"]["reconstructions"]
        rc = next((r for r in recons if r["ratio"] == ratio), None)
        if rc is None:
            rc = {"ratio": ratio, "K": 1, "t_steps": [240], "smooth_sigma": 0.0}
            print(f"  ratio {ratio} not in sample.reconstructions -> using {rc}")
        rc = dict(rc)
        if args.K is not None:
            rc["K"] = args.K
        if args.unconditional:
            # The whole reverse chain, not the paper's partial one. guided_reconstruct
            # starts at x = sqrt(abar_t0)*x_g + sqrt(1-abar_t0)*eps; with x_g zeroed
            # below and t0 = T-1 (sqrt(abar_999) = 6.3e-3) that is a pure-noise start.
            rc["K"] = 1
            rc["t_steps"] = [int(cfg_ck["diffusion"]["timesteps"]) - 1]
        if args.t_steps is not None:
            rc["t_steps"] = args.t_steps
        rc["smooth_sigma"] = args.smooth_sigma
        eta = float(cfg_ck["sample"].get("ddim_eta", 0.0)) if args.eta is None else args.eta
        enc = cfg_ck.get("geo", {}).get("encoder", "-") if cfg_ck.get("geo", {}).get("enabled") else "-"
        if kind == "diffusion":
            print(f"  {stem}: geo={enc} | K={rc['K']} t_steps={rc['t_steps']} eta={eta:g} "
                  f"| tile {args.tile}/{args.overlap} | ensemble {args.ensemble}", flush=True)
        elif kind == "residual":
            res_steps = int(cfg_ck.get("residual", {}).get("n_steps", 100))
            _rs = (f"{res_std:.4f}" if isinstance(res_std, float)
                   else f"per-channel {min(res_std):.4f}-{max(res_std):.4f}")
            print(f"  {stem}: residual diffusion | geo={enc} | res_std={_rs} | "
                  f"mean={'learned' if res_mean_model is not None else 'bicubic'} | "
                  f"{res_steps} DDIM steps on the FULL field (direct mode, as the reference) "
                  f"| ensemble {args.ensemble}", flush=True)
        elif kind == "transport":
            tcfg = cfg_ck.get("transport", {})
            print(f"  {stem}: {method} | geo={enc} | steps={tcfg.get('sample_steps', 100)} "
                  f"solver={tcfg.get('solver', 'heun')} | residual="
                  f"{'no' if residual_t is None else 'res_std=%.4f' % residual_t['res_std']} "
                  f"| tile {args.tile}/{args.overlap} | ensemble {args.ensemble}", flush=True)
        else:
            print(f"  {stem}: direct map (deterministic) | geo={enc} | tile {args.tile}/{args.overlap}", flush=True)

        @torch.no_grad()
        def reconstruct(coarse_n: torch.Tensor, member: int) -> torch.Tensor:
            # sample.full_field._global_noise draws on the CPU and moves the
            # noise to the field's device, so the generator must be a CPU one.
            gen = torch.Generator().manual_seed(args.seed + 1000 * member)
            lf = upsample_nearest(coarse_n, (H, W))
            if args.unconditional:
                # Zero guidance: the chain sees the observation ONLY via the
                # projection. Kept as a full-shape tensor because the tiled
                # driver uses it for shape/device/dtype and for the global noise.
                lf = torch.zeros_like(lf)
            if rc["smooth_sigma"] > 0:
                from data.degrade import degrade
                lf = degrade(upsample_nearest(coarse_n, (H, W)), 1, rc["smooth_sigma"])
            def compose_and_project(mean_f, res, std):
                # coarsen(x) == y holds for the COMPOSED field, not the residual,
                # so the projection is applied once here, never inside the sampler.
                out = mean_f + res_scale(std, mean_f) * res
                if not args.no_project:
                    out = out + upsample_nearest(coarse_n - coarsen(out, ratio), (H, W))
                return out

            def mean_from_coarse(mean_model, mean_geo=False):
                # The frozen mean was trained on degrade(hf) = the coarse field
                # nearest-upsampled; at deployment that same object is `lf`, built
                # from the forecast instead of the truth.
                if mean_model is not None:
                    if args.mean_tiled:
                        return reconstruct_full_tiled_directmap(
                            mean_model, lf, tile=args.tile, overlap=args.overlap,
                            batch=args.batch_tiles,
                            geo_full=(geo if mean_geo else None))
                    return mean_model(lf)
                return torch.nn.functional.interpolate(
                    coarse_n, size=(H, W), mode="bicubic", align_corners=False)

            if kind == "residual":
                mean_f = mean_from_coarse(res_mean_model, res_mean_geo)
                n_res = int(cfg_ck.get("residual", {}).get("n_steps", 100))
                if args.residual_tiled:
                    res = reconstruct_full_tiled_residual(
                        diffusion, model, mean_f, n_steps=n_res, tile=args.tile,
                        overlap=args.overlap, batch=args.batch_tiles,
                        geo_full=(geo if cfg_ck.get("geo", {}).get("enabled") else None),
                        align=ratio, generator=gen)
                else:
                    torch.manual_seed(args.seed + 1000 * member)   # sample_unconditional has no generator arg
                    res = diffusion.sample_unconditional(
                        model, mean_f.shape, mean_f.device, n_steps=n_res,
                        cond=(mean_f, None))
                return compose_and_project(mean_f, res, res_std)
            if kind == "directmap":
                if args.directmap_full:
                    out = model(lf)           # exactly what mean_from_coarse does
                else:
                    out = reconstruct_full_tiled_directmap(
                        model, lf, tile=args.tile, overlap=args.overlap,
                        batch=args.batch_tiles, geo_full=geo)
                if args.project_directmap and not args.no_project:
                    out = out + upsample_nearest(coarse_n - coarsen(out, ratio), (H, W))
                return out
            if kind == "transport":
                tcfg = cfg_ck.get("transport", {})
                kw = {}
                if method == "stochastic_interpolant":
                    si = tcfg.get("stochastic_interpolant", {})
                    kw = dict(sampler=si.get("sampler", "ode"),
                              stochasticity=si.get("stochasticity", 0.1))
                cond_full = (lf if residual_t is None else
                             mean_from_coarse(residual_t["mean_model"],
                                              residual_t["mean_geo"]))
                inner = (not args.no_project) and residual_t is None
                out = reconstruct_full_tiled_transport(
                    model, process, cond_full, coarse_n, ratio, cfg_ck, method,
                    tile=args.tile, overlap=args.overlap, batch=args.batch_tiles,
                    geo_full=geo, project_final=inner, project_each=inner,
                    generator=gen, **kw)
                if residual_t is not None:
                    out = compose_and_project(cond_full, out, residual_t["res_std"])
                return out
            out = reconstruct_full_tiled_diffusion(
                diffusion, model, lf, coarse_n, ratio, rc, eta=eta, tile=args.tile,
                overlap=args.overlap, batch=args.batch_tiles, geo_full=geo,
                project_steps=not args.no_project, project_final=not args.no_project,
                generator=gen, post_x0=hydro, covariance_projector=cov_projector,
                ddim_stride=args.ddim_stride)
            return out
    else:
        eta, rc = 0.0, None

    # ---- incremental results -----------------------------------------------
    # Saved after EVERY init: 32 inits x 2 arms x 4 members is hours of GPU, and
    # a walltime kill must leave the running mean on disk, not nothing. The
    # JSON carries n_inits_done so a partial file is never mistaken for a full one.
    out_path = Path(args.out) if args.out else \
        ensure_dir(cfg["paths"]["results_dir"]) / f"nwp_{stem}_ens{args.ensemble}{shard_tag}.json"

    def write_results(n_done: int, final: bool = False):
        res = {"ckpt": stem, "data_dir": str(data_dir), "n_inits": int(len(idx)),
               "n_inits_done": int(n_done), "complete": bool(final),
               "stride": args.stride, "ratio": ratio, "eta": eta, "recon": rc,
               "ensemble": args.ensemble, "tile": args.tile, "overlap": args.overlap,
               "hydrostatic": float(args.hydrostatic),
               "projection": (not args.no_project),
               "unconditional": bool(args.unconditional),
               "directmap_full": bool(args.directmap_full),
               "residual_tiled": bool(args.residual_tiled),
               "inflation_alphas": args.inflation_alphas,
               "ddim_stride": int(args.ddim_stride),
               "shard": args.shard,
               "kind": (kind if use_model else "bicubic"),
               "covariance": args.covariance,
               "localization_radius": args.localization_radius,
               "grid": [H, W], "lat_flipped_on_load": d["flip"], "channels": labels,
               "arms": {a: {k: v.mean() for k, v in acc[a].items() if v.n}
                        for a in args.arms}}
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(res, indent=1))
        tmp.replace(out_path)
        return res

    # ---- loop -------------------------------------------------------------
    acc = {a: {"model": Accum(), "bicubic": Accum(), "native": Accum()} for a in args.arms}
    t_start = time.time()
    for n, i in enumerate(idx):
        truth = torch.from_numpy(orient(d["arrays"]["control"][i], d)).to(device)
        truth_n = normalizer.encode(truth)
        for arm in args.arms:
            src = truth if arm == "control" else \
                torch.from_numpy(orient(d["arrays"]["forecast"][i], d)).to(device)
            src_n = normalizer.encode(src)
            if arm == "forecast":
                # The NATIVE 0.25 deg forecast scored directly, no downscaling:
                # the expensive product the downscaled 1.5 deg field is trying to
                # attain. Its MAE is the CRPS of a deterministic field, so the
                # ensemble's CRPS compares against it on one footing.
                sn = per_channel_scores(src, truth, w)
                sn.update(hydro_residual_rms(src, labels))
                sn["spectrum_display"] = spectrum_log_l1(
                    src[disp][None, None].cpu(), truth[disp][None, None].cpu())
                acc[arm]["native"].add(sn)
            coarse_n = coarsen(src_n[None], ratio)                # (1,C,H/r,W/r)
            coarse = normalizer.decode(coarse_n[0])
            bic = F.interpolate(coarse[None], size=(H, W), mode="bicubic",
                                align_corners=False)[0]
            sb = per_channel_scores(bic, truth, w)
            sb.update(hydro_residual_rms(bic, labels))
            sb.update({k.replace("hyd_", "truth_hyd_"): v
                       for k, v in hydro_residual_rms(truth, labels).items()})
            sb["spectrum_display"] = spectrum_log_l1(bic[disp][None, None].cpu(),
                                                     truth[disp][None, None].cpu())
            acc[arm]["bicubic"].add(sb)
            if not use_model:
                continue
            members = []
            for m in range(args.ensemble):
                out_n = reconstruct(coarse_n, m)
                members.append(normalizer.decode(out_n[0]))
            stack = torch.stack(members)                            # (M,C,H,W)
            if args.save_members:
                mdir = ensure_dir(Path(cfg["paths"]["results_dir"]) / "nwp_members" / stem)
                tag = str(d["times"][i])[:13].replace(":", "")
                np.save(mdir / f"{arm}_{tag}_members_{labels[disp]}.npy",
                        stack[:, disp].cpu().numpy().astype(np.float16))
                if arm == args.arms[0]:
                    np.save(mdir / f"truth_{tag}_{labels[disp]}.npy",
                            truth[disp].cpu().numpy().astype(np.float16))
            pred = stack.mean(0)
            if args.snapshot:
                sdir = ensure_dir(Path(cfg["paths"]["results_dir"]) / "nwp_snapshots")
                tag = str(d["times"][i])[:13].replace(":", "")
                # fp32, NOT fp16: msl is ~101325 Pa and fp16 tops out at 65504,
                # so a half-precision dump silently turns the whole channel to inf
                # (measured 2026-09-05: 691200/691200 non-finite).
                f16 = lambda t: t.detach().cpu().numpy().astype(np.float32)
                pay = dict(truth=f16(truth), bicubic=f16(bic), model_mean=f16(pred),
                           model_member0=f16(members[0]),
                           lat=np.asarray(d["lat"], dtype=np.float32),
                           channels=np.asarray(labels))
                if arm == "forecast":
                    pay["native"] = f16(src)     # the 0.25 deg forecast, undownscaled
                p_out = sdir / f"{stem}_{arm}_{tag}.npz"
                np.savez_compressed(p_out, **pay)
                print(f"  snapshot -> {p_out.name} "
                      f"({p_out.stat().st_size/1e6:.0f} MB, {len(labels)} channels)",
                      flush=True)
            sm = per_channel_scores(pred, truth, w)
            sm.update(hydro_residual_rms(members[0], labels))   # single member, not the smoothed mean
            sm["rmse_normalized"] = torch.sqrt(
                ((normalizer.encode(pred) - truth_n) ** 2).mean(dim=(1, 2))).tolist()
            sm["spectrum_display"] = spectrum_log_l1(
                members[0][disp][None, None].cpu(), truth[disp][None, None].cpu())
            if args.inflation_alphas:
                # the realism half of the inflation story, for all 20 channels
                sm["spectrum"] = spectrum_log_l1_per_channel(
                    members[0][None].cpu(), truth[None].cpu())
            if args.ensemble > 1:
                # crps_per_channel wants a SEQUENCE of M members, each (N,C,H,W),
                # and an (N,C,H,W) truth; a bare (M,C,H,W) tensor is misread by
                # _as_nchw (verified on synthetic data 2026-09-03).
                sm["crps"] = crps_per_channel([m.cpu()[None] for m in members],
                                              truth.cpu()[None])
                sm["spread"] = stack.std(0).mean(dim=(1, 2)).tolist()
                if args.inflation_alphas:
                    # Inflation about the ensemble mean leaves the mean exactly
                    # unchanged, so rmse_* above are already the inflated values.
                    mu = stack.mean(0, keepdim=True)
                    dev = stack - mu
                    for a in args.inflation_alphas:
                        q = mu + float(a) * dev
                        sm[f"crps_infl{a:g}"] = crps_per_channel(
                            [q[k].cpu()[None] for k in range(q.shape[0])], truth.cpu()[None])
                        sm[f"spread_infl{a:g}"] = q.std(0).mean(dim=(1, 2)).tolist()
                        sm[f"spectrum_infl{a:g}"] = spectrum_log_l1_per_channel(
                            q[0][None].cpu(), truth[None].cpu())
                    del mu, dev, q
                sm["rmse_single_latweighted"] = per_channel_scores(
                    members[0], truth, w)["rmse_latweighted"]
            acc[arm]["model"].add(sm)
        el = time.time() - t_start
        print(f"  init {n+1}/{len(idx)} ({str(d['times'][i])[:13]}) done | "
              f"{el/(n+1):.0f}s/init", flush=True)
        write_results(n + 1)
        if args.snapshot:
            print("  snapshot mode: stopping after one init (scores in this JSON "
                  "are a 1-init sample, NOT a run - do not quote them)", flush=True)
            break

    # ---- report -----------------------------------------------------------
    res = write_results(len(idx), final=True)
    print("\n" + "=" * 78)
    print(f"{stem} | {len(idx)} inits | ratio {ratio} | lat-weighted RMSE (physical units)")
    for arm in args.arms:
        a = res["arms"][arm]
        print(f"[{arm}]")
        nat = a.get("native")
        hdr = f"  {'channel':8s} {'bicubic':>10s}" + (f" {'native':>10s}" if nat else "")
        if "model" in a:
            hdr += f" {stem[:18]:>18s} {'vs bicubic':>10s}" + (f" {'vs native':>9s}" if nat else "")
            if args.ensemble > 1:
                hdr += f" {'CRPS':>9s} {'spread':>8s}"
        print(hdr)
        for c, lab in enumerate(labels):
            b = a["bicubic"]["rmse_latweighted"][c]
            line = f"  {lab:8s} {b:10.4f}"
            if nat:
                line += f" {nat['rmse_latweighted'][c]:10.4f}"
            if "model" in a:
                m = a["model"]["rmse_latweighted"][c]
                line += f" {m:18.4f} {100*(m-b)/b:+9.1f}%"
                if nat:
                    line += f" {100*(m-nat['rmse_latweighted'][c])/nat['rmse_latweighted'][c]:+8.1f}%"
                if args.ensemble > 1:
                    line += f" {a['model']['crps'][c]:9.4f} {a['model']['spread'][c]:8.4f}"
            print(line)
        for k in ("hyd_rms_850_700", "hyd_rms_700_500"):
            tk = k.replace("hyd_", "truth_hyd_")
            if tk in a["bicubic"]:
                line = f"  {k:16s} truth {a['bicubic'][tk]:7.2f}  bicubic {a['bicubic'][k]:7.2f}"
                if nat: line += f"  native {nat[k]:7.2f}"
                if "model" in a: line += f"  model {a['model'][k]:7.2f}   (m2/s2; read as |x - truth|)"
                print(line)
        if nat:
            # CRPS footing on the display channel: a deterministic field's CRPS is its MAE.
            line = (f"  CRPS({labels[disp]}): native(=MAE) {nat['mae_latweighted'][disp]:.4f}"
                    f"  bicubic(=MAE) {a['bicubic']['mae_latweighted'][disp]:.4f}")
            if "model" in a and args.ensemble > 1:
                line += f"  ensemble {a['model']['crps'][disp]:.4f}"
            print(line)
        if "model" in a:
            mn = float(np.mean(a["model"]["rmse_normalized"]))
            print(f"  {'mean(norm)':8s} {'':>10s} {mn:18.4f}   "
                  f"spectrum(t2m) model {a['model']['spectrum_display']:.5f} "
                  f"bicubic {a['bicubic']['spectrum_display']:.5f}")
    print(f"\nSaved -> {out_path} (complete)")


if __name__ == "__main__":
    main()
