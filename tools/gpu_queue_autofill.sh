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

# SYNC-BACK (added 2026-08-26): each cycle, PULL completed pool-node results back to the local repo BEFORE refilling,
# so finished 0-token compute never strands on the nodes (the gap that left ~2433 results unpulled overnight). Safe
# rsync -au (never clobbers a newer local copy); non-fatal so a sync hiccup never blocks the refill.
bash "$MAIN/tools/pool_sync.sh" >> "$LOG" 2>&1 || echo "$(date '+%F %T') pool_sync non-fatal error" >> "$LOG"

depth=$(wc -l < "$QUEUE" 2>/dev/null || echo 0)
POOL_QUEUE=$MAIN/research/queue/pool.queue
pdepth=$(awk 'NF>1 && $0 !~ /^[[:space:]]*#/' "$POOL_QUEUE" 2>/dev/null | wc -l)

# BUGFIX 2026-08-26: the old GPU-queue-full check did `exit 0` BEFORE the pool refill, so the 5 CPU lanes STARVED
# whenever the GPU queue was topped up — which is almost always, since the longi loops are long. Now each queue is
# refilled INDEPENDENTLY on its OWN depth; the cycle advances if EITHER needs work, and we exit only when BOTH are healthy.
if [ "${depth:-0}" -ge "$LOW" ] && [ "${pdepth:-0}" -ge 3 ]; then
  echo "$(date '+%F %T') gpu_depth=$depth>=$LOW AND pool_depth=$pdepth>=3 — both healthy, no refill" >> "$LOG"; exit 0
fi

cycle=$(cat "$STATE" 2>/dev/null || echo 0); cycle=$((cycle + 1)); echo "$cycle" > "$STATE"
base=$((900 + cycle * 10))                      # fresh seed block per cycle -> genuinely new lives, no dup

add() { bash "$MAIN/tools/gpu_queue.sh" add "$1" >/dev/null; }

if [ "${depth:-0}" -lt "$LOW" ]; then
  # 3 continuous-life longitudinal loops at fresh seeds (the long, high-value backstop)
  for i in 0 1 2; do
    s=$((base + i))
    add "cd $WT && SIM_BACKEND=cupy $PY -u -m research.runners._longitudinal_develop_loop_gpu --n-days 60 --seed $s --out $OUT/longi_nd60_s${s}.json"
  done
  # 1 continuous-life multi-seed persistent-loop soak (distinct mechanism, also long)
  add "cd $WT && SIM_BACKEND=cupy $PY -u -m research.runners.persistent_living_loop_derisk --seeds $base $((base+1)) $((base+2)) --segment 6000 --out $OUT/persistent_living_seg6000_c${cycle}.json"
  # 1 vision #75 cell for lane diversity (fresh 6-seed at a deeper epoch)
  add "cd $WT && SIM_BACKEND=cupy $PY -u -m research.runners._vision_rstdp_readout_derisk --seeds 42 43 44 100 101 102 --n-s2 96 --epochs 150 --out $OUT/vrstdp_ns296_ep150_c${cycle}.json"
  echo "$(date '+%F %T') cycle=$cycle gpu_depth_was=$depth enqueued 5 long GPU jobs (seed base $base)" >> "$LOG"
else
  echo "$(date '+%F %T') cycle=$cycle gpu_depth=$depth>=$LOW — GPU queue healthy, skipped GPU refill" >> "$LOG"
fi

# --- POOL refill: keep the 5 disjoint CPU lanes served (they drain in minutes; lane_check gate blocks on empty) ---
if [ "${pdepth:-0}" -lt 3 ]; then
  PQ="$WT/tools/pool_queue.sh"
  PS="$base $((base+1)) $((base+2)) $((base+3)) $((base+4)) $((base+5))"
  padd() { bash "$PQ" add "SIM_BACKEND=numpy OMP_NUM_THREADS=4 .venv/bin/python -u -m research.runners.$1" --checked "$2" >/dev/null 2>&1; }
  padd "_affect_state_region_derisk --seeds $PS --out research/findings/raw/four_day/affect_c${cycle}.json" "laneA affect: multi-seed robustness (continuous-substrate faculty), autofill c$cycle"
  padd "_laneB_curiosity_learning_progress_slope_derisk --seeds $PS --out research/findings/raw/four_day/curiosity_c${cycle}.json" "laneB curiosity: LP-slope drive multi-seed, autofill c$cycle"
  padd "_gnw_coincidence_integrator_derisk --seeds $PS --json research/findings/raw/four_day/gnw_c${cycle}.json" "laneC self/workspace: GNW coincidence integrator multi-seed, autofill c$cycle"
  padd "_b1_v1_selforg_rf_derisk --seeds $PS --out research/findings/raw/four_day/b1_c${cycle}.json" "laneD perception: V1 self-org RF multi-seed, autofill c$cycle"
  padd "_emerge63_corpus_taught_slot_order_derisk --derisk --seeds $PS" "laneE language: slot-order emergence multi-seed, autofill c$cycle"
  echo "$(date '+%F %T') cycle=$cycle pool refill: staged 5 lane jobs (seeds ${base}..)" >> "$LOG"
fi
