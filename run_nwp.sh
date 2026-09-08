#!/bin/bash
# Real-NWP fetch: hres_t0 analysis + hres +24h forecast at 1.5 deg, with
# hres_t0 at 0.25 deg over the +-60 band as truth. Login node -- pure network
# I/O, no billing. Thread caps for the same RLIMIT_NPROC reason as run_global.sh.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd "$(dirname "$0")"
.venv/bin/python -u -m data.fetch_nwp --config config/wb2_20var.yaml \
    --out $PROJECTDIR/datasets/nwp_hres --n-times 256 --lead-hours 24 \
    >> $PROJECTDIR/datasets/nwp_fetch.log 2>&1
