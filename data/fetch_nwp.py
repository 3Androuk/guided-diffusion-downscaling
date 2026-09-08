"""Fetch real NWP fields to test the downscalers outside their training regime.

Everything so far was trained and scored on ERA5 degraded by our own coarsen().
This pulls genuinely different inputs from WeatherBench2:

  hres_t0  ECMWF operational ANALYSIS (the HRES forecast at lead 0). Different
           model, different assimilation, no forecast error -- isolates
           representation mismatch.
  hres     the HRES FORECAST at +24 h. Adds forecast error on top.

Both at 1.5 deg (240x121) as the coarse input, with hres_t0 at 0.25 deg over the
+-60 band as the verification truth. Times are linspace-sampled across 2016-2017
rather than taken from the head, for the same reason eval.compare_geo samples
that way: patches are time-ordered and a contiguous block is one season.

tcwv is absent from both products and is RECONSTRUCTED from q on 13 pressure
levels: mask levels below the surface, integrate, add the sub-1000 hPa layer,
then apply the affine correction fitted against ERA5's own tcwv. Measured on
ERA5: 6.7% of std (r=0.9978), against ~4-5% for the models' own error on that
channel. The naive trapezoid is 16.5% and cannot be corrected (r=0.9863).
"""
import argparse
import sys
import time as _time
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import channel_specs, ensure_dir, load_config  # noqa: E402

SO = dict(token="anon")
G = 9.80665
# Fitted against ERA5's total_column_water_vapour (see fetch docstring).
TCWV_A, TCWV_B = 0.883, -0.25
COARSE = {
    "hres_t0": "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-240x121_equiangular_with_poles_conservative.zarr",
    "hres": "gs://weatherbench2/datasets/hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr",
}
TRUTH = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"
FINE_FC = "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr"


def reconstruct_tcwv(q, levels_hpa, sp):
    """(T,L,H,W) q + (T,H,W) surface pressure -> (T,H,W) tcwv, kg/m^2."""
    p = levels_hpa * 100.0
    P = p[None, :, None, None]
    under = P > sp[:, None]
    qm = np.where(under, 0.0, q)
    w = np.where(under, 0.0, np.gradient(p)[None, :, None, None])
    col = (qm * w).sum(axis=1) / G
    idx = np.argmax(~under[:, ::-1], axis=1)
    low = np.take_along_axis(qm, (len(p) - 1 - idx)[:, None], axis=1)[:, 0]
    plow = p[(len(p) - 1 - idx)]
    col = col + low * np.clip(sp - plow, 0, None) / G
    return TCWV_A * col + TCWV_B


def build(ds, times, specs, path, lat_slice=None, lead=None, chunk=8):
    """Stream (T, C, H, W) in config channel order into `path`, tcwv rebuilt.

    Written straight into an on-disk memmap in time chunks rather than
    assembled in RAM: at 0.25 deg over the +-60 band, 256 times x 20 channels is
    ~14 GB, which the 4 GiB login-node cgroup OOM-kills (exit 137). A 4-time
    smoke test passes regardless, which is exactly how this slipped through.
    Peak here is one chunk, ~0.7 GiB at chunk=8.
    """
    from numpy.lib.format import open_memmap
    probe = ds.isel(time=slice(0, 1))
    if lat_slice is not None:
        probe = probe.isel(latitude=lat_slice)
    H, W = probe.sizes["latitude"], probe.sizes["longitude"]
    tmp = Path(str(path) + ".tmp")
    out = open_memmap(tmp, mode="w+", dtype=np.float32,
                      shape=(len(times), len(specs), H, W))
    try:
        for a in range(0, len(times), chunk):
            b = min(a + chunk, len(times))
            out[a:b] = _chunk(ds, times[a:b], specs, lat_slice, lead)
            out.flush()
            print(f"    {b}/{len(times)}", flush=True)
    finally:
        del out
    tmp.replace(path)
    return (len(times), len(specs), H, W)


def _chunk(ds, times, specs, lat_slice=None, lead=None):
    """One in-memory chunk: (t, C, H, W)."""
    sub = ds.sel(time=times)
    if lead is not None:
        # WB2 stores prediction_timedelta as int64 HOURS in some products and
        # as timedelta64 in others; xarray's .sel does not cast between them,
        # so pick the selector from the coordinate's own dtype. (A comparison
        # like `values == np.timedelta64(24,"h")` misleadingly reports True on
        # the int64 form, so dtype must be checked rather than membership.)
        ptd = ds["prediction_timedelta"]
        sel = int(lead) if np.issubdtype(ptd.dtype, np.integer) \
            else np.timedelta64(lead, "h").astype(ptd.dtype)
        sub = sub.sel(prediction_timedelta=sel)
    if lat_slice is not None:
        sub = sub.isel(latitude=lat_slice)
    lev = ds["level"].values.astype(float)
    H = sub.sizes["latitude"]; W = sub.sizes["longitude"]
    out = np.empty((len(times), len(specs), H, W), dtype=np.float32)
    # EVERY read is transposed explicitly. The two stores disagree on dim
    # order -- the 1.5 deg product is (time, longitude, latitude) while the
    # 0.25 deg one is (time, latitude, longitude) -- so relying on the native
    # order silently transposes fields on one of them.
    def flat(da):
        return da.transpose("time", "latitude", "longitude").values

    qfull = sub["specific_humidity"].transpose(
        "time", "level", "latitude", "longitude").values
    sp = flat(sub["surface_pressure"])
    tcwv = reconstruct_tcwv(qfull, lev, sp)
    for ci, sp_ in enumerate(specs):
        name, lv = sp_["name"], sp_["level"]
        if name == "total_column_water_vapour":
            out[:, ci] = tcwv
        elif lv is None:
            out[:, ci] = flat(sub[name])
        else:
            out[:, ci] = flat(sub[name].sel(level=lv))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/wb2_20var.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-times", type=int, default=512)
    ap.add_argument("--lead-hours", type=int, default=24)
    ap.add_argument("--lat-min", type=float, default=-60.0)
    ap.add_argument("--lat-max", type=float, default=60.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    specs = channel_specs(cfg["data"])
    out = ensure_dir(args.out)

    a0 = xr.open_zarr(COARSE["hres_t0"], storage_options=SO, chunks={"time": 1})
    win = a0.sel(time=slice("2016-01-01", "2017-12-31"))["time"].values
    # hres is initialised at 00/12Z ONLY, while hres_t0 is 6-hourly. Restrict
    # the verification times to 00/12Z so that (valid - 24 h) is always a real
    # init and the analysis and forecast cases score the SAME timestamps --
    # otherwise half the forecast cases silently drop and the two are no longer
    # comparable.
    hh = win.astype("datetime64[h]").astype(int) % 24
    win = win[(hh == 0) | (hh == 12)]
    # Drop the first day so (valid - lead) always lands inside the hres store;
    # otherwise the earliest verification times lose their forecast counterpart
    # and the two cases stop scoring the same timestamps.
    win = win[win >= win[0] + np.timedelta64(args.lead_hours, "h")]
    times = win[np.linspace(0, len(win) - 1, args.n_times).astype(int)]
    np.save(out / "times.npy", times)
    print(f"{len(times)} times, {times[0]} .. {times[-1]}", flush=True)

    for tag, key, lead in [("analysis_coarse", "hres_t0", None),
                           ("forecast24_coarse", "hres", args.lead_hours)]:
        path = out / f"{tag}.npy"
        if path.exists():
            print(f"[skip] {path.name}"); continue
        ds = xr.open_zarr(COARSE[key], storage_options=SO, chunks={"time": 1})
        t = times if lead is None else times - np.timedelta64(lead, "h")
        keep = np.isin(t, ds["time"].values)
        if not keep.all():
            print(f"  {tag}: {(~keep).sum()} of {len(t)} init times absent — dropping")
        t0 = _time.time()
        shape = build(ds, t[keep], specs, path, lead=lead)
        np.save(out / f"{tag}_valid_times.npy", times[keep])
        print(f"[done] {tag}: {shape} in {_time.time()-t0:.0f}s", flush=True)

    path = out / "truth_fine.npy"
    if not path.exists():
        ds = xr.open_zarr(TRUTH, storage_options=SO, chunks={"time": 1})
        lat = ds["latitude"].values
        sl = np.where((lat >= args.lat_min) & (lat <= args.lat_max))[0]
        sl = slice(int(sl.min()), int(sl.max()) + 1)
        t0 = _time.time()
        shape = build(ds, times, specs, path, lat_slice=sl)
        np.savez(out / "truth_coords.npz", lat=ds["latitude"].values[sl],
                 lon=ds["longitude"].values)
        print(f"[done] truth_fine: {shape} in {_time.time()-t0:.0f}s", flush=True)
    # 0.25 deg forecast, so the forecast can also be coarsened with OUR operator.
    # That separates the two shifts a deployment test otherwise conflates:
    #   fine forecast + our coarsen()  -> source-model shift alone (HRES vs ERA5)
    #   WB2's 1.5 deg forecast         -> + a different coarsening operator
    path = out / "forecast24_fine.npy"
    if not path.exists():
        ds = xr.open_zarr(FINE_FC, storage_options=SO, chunks={"time": 1})
        lat = ds["latitude"].values
        sl = np.where((lat >= args.lat_min) & (lat <= args.lat_max))[0]
        sl = slice(int(sl.min()), int(sl.max()) + 1)
        t = times - np.timedelta64(args.lead_hours, "h")
        keep = np.isin(t, ds["time"].values)
        t0 = _time.time()
        shape = build(ds, t[keep], specs, path, lat_slice=sl, lead=args.lead_hours)
        np.save(out / "forecast24_fine_valid_times.npy", times[keep])
        print(f"[done] forecast24_fine: {shape} in {_time.time()-t0:.0f}s", flush=True)

    print("all NWP fields cached", flush=True)


if __name__ == "__main__":
    main()
