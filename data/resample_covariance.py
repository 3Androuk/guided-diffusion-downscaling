"""Re-sample a saved isotropic spectral covariance onto a different grid size.

`data.estimate_spectral_covariance` derives its grid from the training patches
(128x128 here), and `SpectralCovarianceProjector` raises unless the covariance
grid equals the reconstruction grid. The forecast demo tiles at 144 (ratio 6
needs `tile % ratio == 0`, and 128 % 6 = 2), so the 128 artifact cannot be used
there as-is.

The saved spectrum is RADIALLY AVERAGED (`_isotropize` in the estimator), i.e.
a stationary ISOTROPIC covariance: its content is a 1-D profile P(f) of spatial
frequency in cycles/pixel. Re-sampling that profile onto another grid is
therefore exact, not an approximation — the same physical covariance, sampled
differently. Two details make it safe:

  * `np.fft.fftfreq`/`rfftfreq` already return cycles per PIXEL, so the profile
    transfers between grid sizes without rescaling the frequency axis (a naive
    transfer of the integer wavenumber index would shift every scale by
    144/128 = 12.5%).
  * Overall scale is irrelevant: the projector uses C only through
    C A^T (A C A^T)^-1, in which any constant factor cancels — which is also
    why a scalar diagonal covariance is exactly ordinary DDNM.

Run:
    python -m data.resample_covariance \
        --input datasets/patches_t2m/spectral_covariance.npz --size 144
"""

import argparse
from pathlib import Path

import numpy as np


def radial_profile(power: np.ndarray, h: int, w: int):
    """(C,H,W//2+1) rFFT power -> (freqs, profile) sorted by cycles/pixel."""
    fy = np.fft.fftfreq(h)                    # cycles per pixel
    fx = np.fft.rfftfreq(w)
    f = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2).ravel()
    order = np.argsort(f)
    fs = f[order]
    prof = np.stack([c.ravel()[order] for c in power])          # (C, P)
    # collapse duplicate radii (the stored spectrum is constant on rings)
    uniq, inv = np.unique(np.round(fs, 12), return_inverse=True)
    out = np.empty((prof.shape[0], len(uniq)), dtype=np.float64)
    counts = np.bincount(inv)
    for c in range(prof.shape[0]):
        out[c] = np.bincount(inv, weights=prof[c]) / counts
    return uniq, out


def resample(power: np.ndarray, src_hw, dst_hw) -> np.ndarray:
    sh, sw = int(src_hw[0]), int(src_hw[1])
    dh, dw = int(dst_hw[0]), int(dst_hw[1])
    freqs, prof = radial_profile(power, sh, sw)
    fy = np.fft.fftfreq(dh)
    fx = np.fft.rfftfreq(dw)
    ftgt = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    out = np.empty((power.shape[0], dh, dw // 2 + 1), dtype=np.float32)
    for c in range(power.shape[0]):
        # np.interp clamps outside the source range; the corner frequency
        # sqrt(2)/2 is identical for both grids, so no extrapolation occurs.
        out[c] = np.interp(ftgt, freqs, prof[c]).astype(np.float32)
    floor = float(out[out > 0].min()) * 1e-3 if (out > 0).any() else 1e-12
    return np.maximum(out, floor)            # projector requires strictly > 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="spectral_covariance.npz")
    ap.add_argument("--size", type=int, required=True, help="target square grid (e.g. 144)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    src = Path(args.input)
    with np.load(src) as d:
        power = np.array(d["power"], dtype=np.float64)
        image_size = tuple(int(v) for v in d["image_size"])
        extra = {k: d[k] for k in d.files if k not in ("power", "image_size")}
    if power.ndim == 2:
        power = power[None]
    dst_hw = (args.size, args.size)
    out = resample(power, image_size, dst_hw)

    # Sanity: the profile must be reproduced at the shared corner frequency and
    # stay monotone-ish in magnitude; report the band the two grids share.
    f_src, p_src = radial_profile(power, *image_size)
    f_dst, p_dst = radial_profile(out.astype(np.float64), *dst_hw)
    print(f"  source {image_size} power {power.shape} -> target {dst_hw} power {out.shape}")
    print(f"  freq range src {f_src[0]:.5f}..{f_src[-1]:.5f}  dst {f_dst[0]:.5f}..{f_dst[-1]:.5f} (cycles/px)")
    print(f"  P(f) src [{p_src.min():.4g}, {p_src.max():.4g}]  dst [{p_dst.min():.4g}, {p_dst.max():.4g}]")
    for probe in (0.05, 0.1, 0.2, 0.4):
        a = np.interp(probe, f_src, p_src[0]); b = np.interp(probe, f_dst, p_dst[0])
        print(f"    P({probe:.2f} cyc/px): src {a:.5g}  dst {b:.5g}  ratio {b/a:.4f}")
    out_path = Path(args.output) if args.output else \
        src.with_name(f"{src.stem}_{args.size}{src.suffix}")
    np.savez(out_path, power=out, image_size=np.array(dst_hw), **extra)
    print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
