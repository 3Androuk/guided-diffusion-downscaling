"""Does surface-pressure-aware integration rescue the tcwv reconstruction?

The naive 13-level trapezoid gives RMSE 16.5% of std -- 3-4x the model's own
error on that channel. Two known defects:
  (a) everything BELOW 1000 hPa is ignored, yet where ps > 1000 hPa (most ocean
      and lowland) that layer holds a large share of the column's moisture;
  (b) where ps < 1000 hPa (orography) the 1000 hPa level is UNDERGROUND and its
      extrapolated q is counted anyway.
Both are fixable with surface_pressure, which ERA5 and hres_t0 both carry.
"""
import numpy as np
import xarray as xr

SO = dict(token="anon"); G = 9.80665
ds = xr.open_zarr("gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr",
                  storage_options=SO, chunks={"time": 1})
t = ds.sel(time=slice("2016-01-01", "2016-12-31")).isel(time=slice(0, 240, 40))
sl = dict(latitude=slice(None, None, 4), longitude=slice(None, None, 4))
lev = t["level"].values.astype(float)
q = t["specific_humidity"].isel(**sl).transpose("time", "level", "latitude", "longitude").load().values
tv = t["total_column_water_vapour"].isel(**sl).load().values
ps = t["surface_pressure"].isel(**sl).load().values          # Pa
p = lev * 100.0

def score(name, est):
    rmse = float(np.sqrt(np.mean((est - tv) ** 2)))
    r = float(np.corrcoef(est.ravel(), tv.ravel())[0, 1])
    # Best achievable after an affine correction is std*sqrt(1-r^2): a large
    # RMSE with a high r is pure bias/scale and is fixable; a low r is not.
    A = np.vstack([est.ravel(), np.ones(est.size)]).T
    (a, b), *_ = np.linalg.lstsq(A, tv.ravel(), rcond=None)
    rc = float(np.sqrt(np.mean((a * est + b - tv) ** 2)))
    print(f"  {name:34s} RMSE/std {100*rmse/tv.std():5.1f}% | r {r:.4f} "
          f"| AFTER affine fit: {100*rc/tv.std():4.1f}%  (a={a:.3f}, b={b:+.2f})")

score("naive trapezoid (13 levels)", np.trapezoid(q, x=p, axis=1) / G)

# Surface-aware: mask underground, then add the surface layer using the lowest
# valid level's q, integrated from that level down to ps.
P = p[None, :, None, None]
under = P > ps[:, None]                       # level is below ground
qm = np.where(under, 0.0, q)
w = np.zeros_like(qm)
dp = np.gradient(p)
w[:] = dp[None, :, None, None]
w = np.where(under, 0.0, w)
col = (qm * w).sum(axis=1) / G

# lowest valid level per column, extended to the true surface
idx = np.argmax(~under[:, ::-1], axis=1)      # from the bottom up
low = np.take_along_axis(qm, (len(p) - 1 - idx)[:, None], axis=1)[:, 0]
plow = p[(len(p) - 1 - idx)]
surf = low * np.clip(ps - plow, 0, None) / G
score("surface-aware + surface layer", col + surf)
score("surface-aware, no surface layer", col)
print(f"\n  truth: mean {tv.mean():.2f} std {tv.std():.2f} kg/m2")
print("  model's own tcwv error is ~4-5% of std for comparison")
