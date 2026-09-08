#!/bin/bash
# Fan an N-arm comparison out to N one-arm jobs, one GPU each.
#
# EXACT, not an approximation: eval/compare_geo._recon reseeds per arm
# (seed = cfg.seed + 1000*ratio), so an arm's noise draw does not depend on
# which other arms share the job. Verified — diffusion_geo_static scored
# 0.0436386 both in the 5-way run and alone alongside the combo.
#
# Same node-hours as one sequential job (billing is proportional: 1 GPU =
# billing 72 = a quarter node, 4 GPUs = 288), but wall time is ONE arm's
# runtime instead of N arms'. Patch-level sharding would instead need
# per-batch reseeding and would change every number; this needs no code
# change at all, which is why it is the shape used here.
#
#   scripts/eval_sharded.sh config/wb2_20var.yaml TAG ckptA.pt ckptB.pt ...
set -eu
CFG="$1"; TAG="$2"; shift 2
cd "$(dirname "$0")/.."
for CK in "$@"; do
  STEM="${CK%.pt}"
  sbatch --job-name="ev_${STEM}" --nodes=1 --gpus=1 --time=04:00:00 \
    --output="evalshard-${TAG}-${STEM}-%j.out" \
    --wrap="cd $PWD && .venv/bin/python -m eval.compare_geo --config ${CFG} --ckpts ${CK} --batch 32 --project"
done
