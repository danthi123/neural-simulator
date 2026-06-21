#!/usr/bin/env bash
# Serial driver: run the 3 arms x 3 seeds ONE AT A TIME (the GPU is shared with other jobs;
# adding one job at a time minimizes contention + gives incremental, committable results).
# Order seed-first so a full seed-triple (dendcritic/baseline/lesion) completes before the next
# seed -> a complete comparison ROW can be committed as each seed finishes.
set -u
OUTDIR="research/findings/raw/dendrite_critic"
mkdir -p "$OUTDIR"
COMMON="--moving-goal --goal-schedule multi --deterministic \
  --enable-neural-critic --spiking-snc --spiking-reward-us --enable-critic-homeostasis \
  --grid-size 32 --n-steps 1800"

run_one () {  # arm  extra_flags  seed
  local arm="$1"; local extra="$2"; local s="$3"
  local out="$OUTDIR/${arm}_seed${s}.json"
  if [ -f "$out" ]; then echo "[$arm seed=$s] exists, skip"; return 0; fi
  for attempt in 1 2 3 4 5; do
    echo "[$arm seed=$s] attempt $attempt @ $(date '+%H:%M:%S')"
    PYTHONIOENCODING=utf-8 SIM_BACKEND=cupy python -m research.runners.g11_bg_runner \
      $COMMON $extra --seed "$s" --out "$out" > "$OUTDIR/${arm}_seed${s}.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ] && [ -f "$out" ]; then echo "[$arm seed=$s] DONE @ $(date '+%H:%M:%S')"; return 0; fi
    if grep -qiE "OutOfMemory|cudaErrorMemoryAllocation|out of memory" "$OUTDIR/${arm}_seed${s}.log"; then
      echo "[$arm seed=$s] OOM attempt $attempt -> wait 120s + retry"; sleep 120; continue
    fi
    echo "[$arm seed=$s] FAILED rc=$rc (non-OOM); tail:"; tail -8 "$OUTDIR/${arm}_seed${s}.log"; return 1
  done
  echo "[$arm seed=$s] GAVE UP after retries"; return 1
}

for s in 42 43 44; do
  run_one dendcritic "--dendrite-critic" "$s"
  run_one baseline   ""                  "$s"
  run_one lesion     "--dendrite-critic --dendrite-critic-graded-strength 0" "$s"
  echo "=== SEED ${s} TRIPLE COMPLETE @ $(date '+%H:%M:%S') ==="
done
echo "=== SERIAL ALL DONE @ $(date '+%H:%M:%S') ==="
