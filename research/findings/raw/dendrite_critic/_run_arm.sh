#!/usr/bin/env bash
# Run one nav-RL arm across a seed set, OOM-guarded (wait + retry, never crash the batch).
# Usage: _run_arm.sh <arm_name> <extra_flags> <seed1,seed2,...>
set -u
ARM="$1"; EXTRA="$2"; SEEDS="$3"
OUTDIR="research/findings/raw/dendrite_critic"
mkdir -p "$OUTDIR"
COMMON="--moving-goal --goal-schedule multi --deterministic \
  --enable-neural-critic --spiking-snc --spiking-reward-us --enable-critic-homeostasis \
  --grid-size 32 --n-steps 1800"
IFS=',' read -ra SS <<< "$SEEDS"
for s in "${SS[@]}"; do
  OUT="$OUTDIR/${ARM}_seed${s}.json"
  if [ -f "$OUT" ]; then echo "[$ARM seed=$s] exists, skip"; continue; fi
  for attempt in 1 2 3 4 5; do
    echo "[$ARM seed=$s] attempt $attempt"
    PYTHONIOENCODING=utf-8 SIM_BACKEND=cupy python -m research.runners.g11_bg_runner \
      $COMMON $EXTRA --seed "$s" --out "$OUT" > "$OUTDIR/${ARM}_seed${s}.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && [ -f "$OUT" ]; then echo "[$ARM seed=$s] DONE rc=0"; break; fi
    if grep -qiE "OutOfMemory|cudaErrorMemoryAllocation|cudaError|out of memory" "$OUTDIR/${ARM}_seed${s}.log"; then
      echo "[$ARM seed=$s] OOM (attempt $attempt) -> wait 120s + retry"; sleep 120; continue
    fi
    echo "[$ARM seed=$s] FAILED rc=$rc (non-OOM); tail:"; tail -8 "$OUTDIR/${ARM}_seed${s}.log"; break
  done
done
echo "[$ARM] ALL SEEDS DONE: $SEEDS"
