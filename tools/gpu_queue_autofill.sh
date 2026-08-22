#!/bin/bash
# gpu_queue_autofill.sh — keep the GPU queue TOPPED UP with genuine LONG jobs so the 3090 never drains before the
# Tue 2026-08-26 usage reset. This ENQUEUES work (the sibling gpu-queue-refill.timer only RESTARTS the dispatcher,
# which is why the queue sat idle after 22 cheap jobs cleared in ~30 min on 2026-08-22).
#
# WHY genuine, not padding: the batch is dominated by the LONGITUDINAL CONTINUOUS-LIFE loop
# (_longitudinal_develop_loop_gpu) at a FRESH seed each cycle — a different simulated developmental life per seed,
# which is exactly the owner's #1 strategic priority (make the brain CONTINUOUS: learn-through-use / novelty /
# trains-of-thought over long wall-clock). A rotating seed space means no exact duplicates. A vision #75 cell adds
# lane diversity so lane_check does not read monoculture.
#
# Runs from a systemd timer (~every 15 min). If the queue is at/above the watermark it does nothing.
set -uo pipefail
MAIN=/home/dant123/Projects/sim
WT=/home/dant123/Projects/sim-worktrees/four-day-queue
PY=/home/dant123/Projects/sim/.venv/bin/python
QUEUE=$MAIN/research/queue/gpu.queue
STATE=$MAIN/research/queue/.autofill_cycle
LOG=$MAIN/research/queue/autofill.log
LOW=${GPU_QUEUE_LOW_WATERMARK:-8}
OUT=research/findings/raw/four_day            # relative to $WT (jobs cd there)

depth=$(wc -l < "$QUEUE" 2>/dev/null || echo 0)
if [ "${depth:-0}" -ge "$LOW" ]; then
  echo "$(date '+%F %T') depth=$depth >= $LOW — no refill" >> "$LOG"; exit 0
fi

cycle=$(cat "$STATE" 2>/dev/null || echo 0); cycle=$((cycle + 1)); echo "$cycle" > "$STATE"
base=$((900 + cycle * 10))                      # fresh seed block per cycle -> genuinely new lives, no dup

add() { bash "$MAIN/tools/gpu_queue.sh" add "$1" >/dev/null; }

# 3 continuous-life longitudinal loops at fresh seeds (the long, high-value backstop)
for i in 0 1 2; do
  s=$((base + i))
  add "cd $WT && SIM_BACKEND=cupy $PY -u -m research.runners._longitudinal_develop_loop_gpu --n-days 60 --seed $s --out $OUT/longi_nd60_s${s}.json"
done
# 1 continuous-life multi-seed persistent-loop soak (distinct mechanism, also long)
add "cd $WT && SIM_BACKEND=cupy $PY -u -m research.runners.persistent_living_loop_derisk --seeds $base $((base+1)) $((base+2)) --segment 6000 --out $OUT/persistent_living_seg6000_c${cycle}.json"
# 1 vision #75 cell for lane diversity (fresh 6-seed at a deeper epoch)
add "cd $WT && SIM_BACKEND=cupy $PY -u -m research.runners._vision_rstdp_readout_derisk --seeds 42 43 44 100 101 102 --n-s2 96 --epochs 150 --out $OUT/vrstdp_ns296_ep150_c${cycle}.json"

echo "$(date '+%F %T') cycle=$cycle depth_was=$depth enqueued 5 long jobs (seed base $base)" >> "$LOG"
