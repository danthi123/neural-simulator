#!/usr/bin/env bash
# Serial handler-level runs (seed 101, group emo) from the pin+patch worktree. Each job waits for tools/mem_ok.sh 10, then runs under memcap 11.
WT=/home/dant123/Projects/sim-rev/wt-emobc-pin
S=/home/dant123/Projects/sim-rev/emobc_scratch/handler
PY=/home/dant123/Projects/sim/.venv/bin/python
PREREG=research/findings/2026-09-30-stage2-exit-gate-PREREGISTRATION.md
cd $WT || exit 1
for job in "on intact_a" "on permuted" "off intact_a" "off permuted"; do
  set -- $job; side=$1; arm=$2
  out=$S/$side/s101/$arm.json
  [ -s "$out" ] && { echo "$(date +%T) skip $side $arm (exists)"; continue; }
  mkdir -p $S/$side/s101
  until bash /home/dant123/Projects/sim/tools/mem_ok.sh 10 >/dev/null 2>&1; do sleep 20; done
  echo "$(date +%T) start $side $arm"
  if [ "$side" = on ]; then FLAG="BRAIN_EMOTION_CATEGORY_BOUNDED=1"; else FLAG="EMOBC_UNUSED=1"; fi
  bash /home/dant123/Projects/sim/tools/memcap.sh 11 -- env $FLAG SIM_NO_PROVENANCE=1 SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
    $PY -u -m research.runners.stage2_exit_gate --shard --seed 101 --group emo --arm $arm --allow-gate-seed --prereg $PREREG --out $out > $S/${side}_${arm}.log 2>&1
  echo "$(date +%T) end $side $arm rc=$?"
done
echo QUEUE_DONE
