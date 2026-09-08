#!/bin/bash
#SBATCH --job-name=evalgeo
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=evalgeo-%j.out
cd "$SLURM_SUBMIT_DIR"; source .venv/bin/activate
# Head-to-head on IDENTICAL test patches. Projection on (the DDNM contribution,
# and what every prior table in this project reports).
python -u -m eval.compare_geo --config config/wb2_20var.yaml \
    --ckpts diffusion_geo_hpx.pt diffusion_geo_static.pt \
    --project --batch 16
