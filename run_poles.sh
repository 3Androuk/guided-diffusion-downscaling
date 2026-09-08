#!/bin/bash
# Sharded polar fetch on a login node.
#
# THREAD CAPS ARE LOAD-BEARING. A first attempt with 4 shards and no caps died
# instantly: OpenBLAS sizes its pool to the node's core count (144 here), so
# four processes blew RLIMIT_NPROC (1900 per user), and once threads could not
# be spawned every GCS read failed with RuntimeError. It was never an OOM --
# memory.events showed oom_kill 0. The single-process smoke test worked purely
# because one pool of 144 fit.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd "$(dirname "$0")"
OUT=$PROJECTDIR/datasets/poles_wb220
mkdir -p $OUT
N=${NSHARDS:-3}
for ((i=0;i<N;i++)); do
  .venv/bin/python -u -m data.fetch_poles --config config/wb2_20var.yaml \
      --out $OUT --batch 8 --max-retries 8 --dask-threads 2 \
      --shard $i --nshards $N >> $OUT/shard$i.log 2>&1 &
done
wait
