#!/bin/bash
# Score one 20-var checkpoint on the real-NWP fetch (datasets/nwp_hres):
#
#   cd era5-diffusion-downscaling
#   sbatch scripts/nwp_eval.sbatch diffusion_geo_static.pt --limit 32 --stride 8 --ensemble 4
#   sbatch scripts/nwp_eval.sbatch diffusion.pt            --limit 32 --stride 8 --ensemble 4
#
# ONE GPU: this is tiled inference on 480x1440 fields, one field at a time —
# a single GH200 is the right shape and bills a quarter node (billing=72).
# MEASURED 2026-09-03: ~430 s per init (2 arms x 4 members x 240 DDIM steps,
# 65 tiles per forward, 244M-param UNet, fp32, GPU at 100%), i.e. 32 inits is
# ~4 h. The first attempt asked for 2 h off a t2m-scale guess and would have
# been killed with nothing saved; the driver now writes an atomic running-mean
# JSON after every init, so a kill costs at most one init. 5 h is the fit for
# 32 inits — pass --time=... to override for other --limit values.
#
#SBATCH --job-name=nwp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out

set -uo pipefail
CKPT=${1:?usage: sbatch $0 <checkpoint.pt|bicubic> [downscale_nwp args...]}
shift
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
source .venv/bin/activate
export OMP_NUM_THREADS=8
echo "[node] $(hostname) job $SLURM_JOB_ID | $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[run]  eval.downscale_nwp --ckpt $CKPT $*"
python -u -m eval.downscale_nwp --config config/wb2_20var.yaml --ckpt "$CKPT" "$@"
echo "[exit $?]"
