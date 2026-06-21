#!/usr/bin/env bash
# DECISIVE attribution control: baseline point-neuron critic + NMDA-on (no dendrite plateau).
# If this recovers the dendcritic ~8.5 nav score WITHOUT any graded dendrite value, it proves the
# dendcritic-vs-baseline gap is driven by NMDA-on-the-critic-slice, NOT by the learned dendrite value
# (which the lesion already showed is non-load-bearing: striov 263->0, nav unchanged).
set -u
OUTDIR="research/findings/raw/dendrite_critic"
COMMON="--moving-goal --goal-schedule multi --deterministic \
  --enable-neural-critic --spiking-snc --spiking-reward-us --enable-critic-homeostasis \
  --grid-size 32 --n-steps 1800"
run_one () {  # arm  extra  seed
  local arm="$1"; local extra="$2"; local s="$3"
  local OUT="$OUTDIR/${arm}_seed${s}.json"
  if [ -s "$OUT" ]; then echo "[$arm seed=$s] exists, skip"; return 0; fi
  for attempt in 1 2 3 4 5; do
    echo "[$arm seed=$s] attempt $attempt @ $(date '+%H:%M:%S')"
    PYTHONIOENCODING=utf-8 SIM_BACKEND=cupy python -m research.runners.g11_bg_runner \
      $COMMON $extra --seed "$s" --out "$OUT" > "$OUTDIR/${arm}_seed${s}.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && [ -s "$OUT" ]; then echo "[$arm seed=$s] DONE @ $(date '+%H:%M:%S')"; return 0; fi
    if grep -qiE "OutOfMemory|out of memory|cudaErrorMemoryAllocation" "$OUTDIR/${arm}_seed${s}.log"; then
      echo "[$arm seed=$s] OOM attempt $attempt -> wait 120s"; sleep 120; continue
    fi
    echo "[$arm seed=$s] FAILED rc=$rc; tail:"; tail -12 "$OUTDIR/${arm}_seed${s}.log"; return 1
  done
}
# baseline + global NMDA (the cause-attribution control)
run_one ctrl_nmda "--enable-pfc-nmda" 42
echo "[ctrl] DONE @ $(date '+%H:%M:%S')"
