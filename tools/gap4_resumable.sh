#!/usr/bin/env bash
# Resumable, durable driver for the gap#4 crux. Runs ONE (seed, arm) cell per process.
#
# WHY (2026-07-30): the runner writes its JSON only AFTER all arms finish, so a 15-arm run that is interrupted
# at hour 300 loses everything, and for a multi-day job the only record of a completed arm was a line in a
# /tmp log that does not survive a reboot. The owner reboots into Windows for gaming periodically. The compute
# was never the fragile part -- the bookkeeping was.
#
# THIS CHANGES THE LAUNCHER ONLY. The runner is untouched and stays byte-identical (it is validated code and
# the crux depends on it); all that changes is that it is invoked per-cell with --arms/--seeds.
#
#   bash tools/gap4_resumable.sh [epochs] [settle] [seeds...]
#   bash tools/gap4_resumable.sh 10 20 42 43 44
#
# RESUME IS AUTOMATIC: a cell whose output JSON already exists is SKIPPED. Kill it, reboot, rerun the same
# command -- it picks up at the first unfinished cell. Worst case loss is ONE cell, not the whole run.
set -uo pipefail
cd /home/dant123/Projects/sim || exit 1
EPOCHS="${1:-10}"; SETTLE="${2:-20}"; shift 2 2>/dev/null || true
SEEDS=("${@:-42}")
ARMS=(reservoir fixed_fa kp micro transport_ceiling)
OUTD=research/findings/raw/gap4/resumable
LOGD=research/findings/raw/gap4/logs          # in-repo => survives a reboot, unlike /tmp
mkdir -p "$OUTD" "$LOGD"

echo "[gap4-resumable] epochs=$EPOCHS settle=$SETTLE seeds=${SEEDS[*]}"
echo "[gap4-resumable] $(( ${#SEEDS[@]} * ${#ARMS[@]} )) cells; existing outputs are SKIPPED (this is the resume)"
for seed in "${SEEDS[@]}"; do
  for arm in "${ARMS[@]}"; do
    OUT="$OUTD/s${seed}_${arm}_e${EPOCHS}s${SETTLE}.json"
    if [ -s "$OUT" ]; then
      echo "[gap4-resumable] SKIP  seed=$seed arm=$arm (already done: $(basename "$OUT"))"
      continue
    fi
    echo "[gap4-resumable] RUN   seed=$seed arm=$arm  $(date '+%H:%M:%S')"
    T0=$(date +%s)
    SIM_BACKEND=cupy .venv/bin/python -m research.runners._gap4_onbridge_spiking_selfpredict_derisk \
      --full --seeds "$seed" --arms "$arm" --epochs "$EPOCHS" --settle-steps "$SETTLE" \
      --out "$OUT" >> "$LOGD/resumable_s${seed}_${arm}.log" 2>&1
    RC=$?
    T1=$(date +%s)
    if [ "$RC" -eq 0 ] && [ -s "$OUT" ]; then
      echo "[gap4-resumable] DONE  seed=$seed arm=$arm in $(( (T1-T0)/60 )) min  -> $(basename "$OUT")"
    else
      # A failed cell leaves NO output, so the next run retries exactly this cell and nothing else.
      echo "[gap4-resumable] FAIL  seed=$seed arm=$arm rc=$RC after $(( (T1-T0)/60 )) min (will retry on rerun)"
    fi
  done
done
echo "[gap4-resumable] all cells attempted. Completed: $(ls -1 "$OUTD" 2>/dev/null | wc -l)"
