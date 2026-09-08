"""Per-channel lat-lon -> HPX -> lat-lon round-trip floor, bilinear both ways.

The error a PERFECT mesh model would still be charged when scored on the
lat-lon benchmark, because its prediction lives on the mesh. eval/
measure_remap_floor.py does this with an exact SHT backward transform, but
ducc0 has no aarch64 wheel and will not build here, so this measures the
BILINEAR round trip -- the one we would actually be stuck with.

Reported in the same normalized units as compare_geo's l2_normalized, so the
floor is directly comparable to the models' scores (best arm: 0.0433 at 4x,
0.0967 at 8x). Restricted to the +-60 deg band, which is the region the patch
arms are scored on.
"""
import sys, time
from pathlib import Path
import numpy as np

H = Path.home() / "physics-informed-weather"
sys.path.insert(0, str(H / "dlwp-hpx-sr"))
sys.path.insert(0, str(H / "era5-diffusion-downscaling"))
from hpx.remap import LatLonToHPX, hpx_to_latlon      # noqa: E402
from utils import channel_labels, load_config         # noqa: E402

NSIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 256
NF = int(sys.argv[2]) if len(sys.argv) > 2 else 3

cfg = load_config(str(H / "era5-diffusion-downscaling/config/wb2_20var_global.yaml"))
labels = channel_labels(cfg["data"])
G = Path("/path/to/datasets/raw_wb220_global")
arr = np.load(G / "_years/train_2007.npy", mmap_mode="r")     # (T, C, 721, 1440)
co = np.load(G.parent / "raw_wb220/coords.npz")               # band coords (for lon)
full = np.load(H / "era5-diffusion-downscaling/datasets/patches_wb220/norm_stats.npz")
std = np.asarray(full["std"], dtype=np.float64).reshape(-1)   # per-channel, physical

T, C, NLAT, NLON = arr.shape
print(f"source {arr.shape}  nside={NSIDE}  fields={NF}")

# The store's latitude is DESCENDING; LatLonToHPX requires strictly ascending.
lat = np.linspace(-90.0, 90.0, NLAT)
lon = np.linspace(0.0, 360.0 - 360.0 / NLON, NLON)
band = (lat >= -60.0) & (lat <= 60.0)
print(f"band rows {band.sum()} of {NLAT}")

t0 = time.time()
fwd = LatLonToHPX(lat, lon, NSIDE)
print(f"forward remap built in {time.time()-t0:.1f}s")

idx = np.linspace(0, T - 1, NF).astype(int)
se = np.zeros(C); n = 0
for k, i in enumerate(idx):
    f = np.asarray(arr[i], dtype=np.float32)[:, ::-1]   # flip to ascending lat
    faces = fwd(f)                                      # (C, 12, nside, nside)
    back = hpx_to_latlon(faces, lat, lon)               # (C, NLAT, NLON)
    d = (back[:, band] - f[:, band]).astype(np.float64)
    se += (d ** 2).mean(axis=(-2, -1))
    n += 1
    print(f"  field {k+1}/{NF} (t={i}) done  {time.time()-t0:.0f}s", flush=True)

rmse = np.sqrt(se / n)                 # physical units, per channel
norm = rmse / std                      # same units as l2_normalized

print(f"\n{'channel':<8} {'floor (phys)':>14} {'floor (norm)':>13} {'vs 4x best 0.0433':>19}")
for c in range(C):
    lab = labels[c] if c < len(labels) else f"ch{c}"
    print(f"{lab:<8} {rmse[c]:>14.5g} {norm[c]:>13.5f} {100*norm[c]/0.0433417:>18.1f}%")
print(f"\npooled normalized floor: {norm.mean():.5f}")
print(f"best model 4x 0.04334 | 8x 0.09669  -> floor is "
      f"{100*norm.mean()/0.0433417:.1f}% of the 4x score, "
      f"{100*norm.mean()/0.0966865:.1f}% of the 8x score")
