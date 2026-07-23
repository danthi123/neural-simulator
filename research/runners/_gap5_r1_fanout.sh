#!/bin/bash
# gap#5 R1 hetero-encode fan-out. Usage: _gap5_r1_fanout.sh "<seeds>" "<heteros>"
#   e.g.  _gap5_r1_fanout.sh "42 43 44 100 101 102" "0.2"     (multi-seed confirm of a winner)
#         _gap5_r1_fanout.sh "42" "0.0 0.1 0.2 0.3 0.5"        (single-seed lever-find)
# Runs each (seed,hetero) as a concurrent numpy proc (CPU; leaves the GPU for training).
cd /home/dant123/Projects/sim
export SIM_BACKEND=numpy OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
SEEDS="${1:-42}"; HETS="${2:-0.0 0.1 0.2 0.3 0.5}"
mkdir -p research/findings/raw/gap5_r1
pids=()
for s in $SEEDS; do for h in $HETS; do
  tag=$(echo $h | tr -d '.')
  .venv/bin/python -m research.runners._gap5_R1_hetero_encode_sweep --seed $s --hetero $h \
     --out research/findings/raw/gap5_r1/s${s}_h${tag}.json > research/findings/raw/gap5_r1/s${s}_h${tag}.log 2>&1 &
  pids+=($!)
done; done
echo "launched ${#pids[@]} R1 procs (seeds='$SEEDS' hets='$HETS')"
wait
echo "ALL_R1_DONE"
for s in $SEEDS; do for h in $HETS; do
  tag=$(echo $h | tr -d '.')
  grep -h "^\[R1\]" research/findings/raw/gap5_r1/s${s}_h${tag}.log 2>/dev/null || echo "  s=$s h=$h FAILED (see log)"
done; done
