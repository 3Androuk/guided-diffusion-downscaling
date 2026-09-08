"""Can tcwv be reconstructed from 13-level specific humidity?

hres/hres_t0 carry no total_column_water_vapour, but do carry q on 13 pressure
levels. TCWV = (1/g) int q dp. The models were TRAINED on ERA5's own tcwv,
which integrates ~137 model levels down to the true surface -- so a 13-level
pressure integration to 1000 hPa is an approximation, and its error is a
distribution shift on that channel at deployment.

This measures that error on ERA5 itself, where both the ingredients and the
ground truth exist, so the bias is known before it is relied on.
"""
import numpy as np
import xarray as xr

SO = dict(token="anon")
ERA5 = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"
G = 9.80665

ds = xr.open_zarr(ERA5, storage_options=SO, chunks={"time": 1})
print("surface_pressure present:", "surface_pressure" in ds.data_vars)

t = ds.sel(time=slice("2016-01-01", "2016-12-31")).isel(time=slice(0, 240, 40))
lev = t["level"].values.astype(float)          # hPa, ascending 50..1000
print("levels:", lev.tolist())

q = t["specific_humidity"].isel(latitude=slice(None, None, 4),
                                longitude=slice(None, None, 4)).load()
truth = t["total_column_water_vapour"].isel(latitude=slice(None, None, 4),
                                            longitude=slice(None, None, 4)).load()
sp = (t["surface_pressure"].isel(latitude=slice(None, None, 4),
                                 longitude=slice(None, None, 4)).load()
      if "surface_pressure" in t else None)

p = lev * 100.0                                 # Pa
qv = q.transpose("time", "level", "latitude", "longitude").values

# trapezoidal integral (numpy 2 renamed trapz -> trapezoid)
tcwv_trap = np.trapezoid(qv, x=p, axis=1) / G

print(f"\nsamples {qv.shape[0]}  grid {qv.shape[-2]}x{qv.shape[-1]}")
tv = truth.values
for name, est in [("trapz 13-level", tcwv_trap)]:
    bias = float(np.mean(est - tv))
    rmse = float(np.sqrt(np.mean((est - tv) ** 2)))
    r = float(np.corrcoef(est.ravel(), tv.ravel())[0, 1])
    rel = 100.0 * rmse / float(np.std(tv))
    print(f"{name}: bias {bias:+.3f} kg/m2 | RMSE {rmse:.3f} | "
          f"r {r:.4f} | RMSE/std {rel:.1f}%")
    # a single global scale+offset correction, fit here
    A = np.vstack([est.ravel(), np.ones(est.size)]).T
    coef, *_ = np.linalg.lstsq(A, tv.ravel(), rcond=None)
    corr = coef[0] * est + coef[1]
    rmse_c = float(np.sqrt(np.mean((corr - tv) ** 2)))
    print(f"   after linear correction (a={coef[0]:.4f}, b={coef[1]:+.3f}): "
          f"RMSE {rmse_c:.3f} | RMSE/std {100*rmse_c/float(np.std(tv)):.1f}%")
print(f"\ntruth tcwv: mean {tv.mean():.2f} std {tv.std():.2f} "
      f"range [{tv.min():.2f}, {tv.max():.2f}] kg/m2")
