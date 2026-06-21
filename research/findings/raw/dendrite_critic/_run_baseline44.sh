#!/usr/bin/env bash
set -u
OUTDIR="research/findings/raw/dendrite_critic"
COMMON="--moving-goal --goal-schedule multi --deterministic \
  --enable-neural-critic --spiking-snc --spiking-reward-us --enable-critic-homeostasis \
  --grid-size 32 --n-steps 1800"
OUT="$OUTDIR/baseline_seed44.json"
rm -f "$OUT"  # 0-byte stale
for attempt in 1 2 3 4 5; do
  echo "[baseline seed=44] attempt $attempt @ $(date '+%H:%M:%S')"
  PYTHONIOENCODING=utf-8 SIM_BACKEND=cupy python -m research.runners.g11_bg_runner \
    $COMMON --seed 44 --out "$OUT" > "$OUTDIR/baseline_seed44.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -s "$OUT" ]; then echo "[baseline seed=44] DONE @ $(date '+%H:%M:%S')"; break; fi
  if grep -qiE "OutOfMemory|out of memory|cudaErrorMemoryAllocation" "$OUTDIR/baseline_seed44.log"; then
    echo "[baseline seed=44] OOM attempt $attempt -> wait 120s"; sleep 120; continue
  fi
  echo "[baseline seed=44] FAILED rc=$rc; tail:"; tail -12 "$OUTDIR/baseline_seed44.log"; break
done
echo "[baseline seed=44] EXIT @ $(date '+%H:%M:%S')"
