#!/bin/bash
# Full-GLOBAL 20-var download for the HEALPix mesh arm.
#
# ONE process, not several. Measured 2026-09-01: a single process ran ~10.4
# timesteps/min; three concurrent shards managed ~2.3/min EACH (~4.6 total),
# i.e. parallelism was ~2.4x counterproductive -- this is bandwidth-bound, and
# splitting it just divides one pipe while multiplying overhead and stalls.
#
# GLOBAL rather than poles-only because the store chunk is (1, 13, 721, 1440):
# one read returns ALL 721 latitudes, so fetching the two polar caps as
# separate passes downloaded every chunk twice and discarded 83% each time.
# Reading the whole field once is half the wire cost of two cap passes AND
# removes the band/cap stitching, the descending-lat flip, and the grid-subset
# match from the remap step entirely.
#
# Thread caps: OpenBLAS sizes its pool to the 144-core node and blew
# RLIMIT_NPROC (1900) when processes ran in parallel; harmless here, kept so a
# future parallel variant cannot regress.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd "$(dirname "$0")"
LOG=$PROJECTDIR/datasets/global_download.log
.venv/bin/python -u -m data.download_era5 --config config/wb2_20var_global.yaml \
    --batch 8 --dask-threads 4 --max-retries 8 >> $LOG 2>&1
