#!/bin/bash
#SBATCH --job-name=bench
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:25:00
#SBATCH --output=bench-%j.out
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "### flow, no geo ###"
python -u scripts/bench_batch.py --config config/wb2_20var.yaml --model flow \
    --batches 8,16,24,32,48,64,96,128 --amp off,bf16
echo
echo "### flow + geo (what we would actually train) ###"
python -u scripts/bench_batch.py --config config/wb2_20var.yaml --model flow --geo \
    --batches 8,16,24,32,48,64,96 --amp bf16
