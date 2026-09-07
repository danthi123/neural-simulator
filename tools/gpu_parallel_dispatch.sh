#!/bin/bash
# gpu_parallel_dispatch.sh — bounded-concurrency dispatcher for SMALL GPU jobs (<~1GB VRAM each).
#
# WHY THIS EXISTS (and why it does NOT replace gpu_queue.sh): the strict-serial gpu_queue.sh protects
# the BIG ~20GB production brain from stacking — two concurrent brain loads OOM the 24GB 3090 / take the
# card off the bus (reboot-only). But the agi-fork substrate runs are tiny rate-recurrent nets: measured
# ~0.9GB VRAM each with 20GB free, and one job leaves the GPU at ~44% util. Serial there wastes the card.
# Owner directive 2026-09-07: "optimally use compute within RAM and VRAM limits." This runs N jobs
# concurrently, VRAM-GATED (won't start a job unless free VRAM >= MIN_FREE_MIB), sharing gpu_queue's QLOCK
# for atomic pops so there is NO double-run even if the daemon is (accidentally) not paused. It logs
# START/DONE lines to gpu_queue.log so the existing completion Monitor keeps catching finishes.
#
# USAGE: pause the serial daemon first, then launch this as a background controller:
#   bash tools/gpu_queue.sh pause                 # graceful: current job finishes, no new serial pops
#   bash tools/gpu_parallel_dispatch.sh [N] [MIN_FREE_MIB]   # default N=3, MIN_FREE_MIB=4000
# It drains research/queue/gpu.queue and exits when the queue is empty AND all launched jobs have exited.
# ⛔ Do NOT point this at the production brain — that stays strictly serial on gpu_queue.sh.
set -u
QDIR=research/queue
QUEUE=$QDIR/gpu.queue
QLOCK=$QDIR/.gpu_queue.lock
LOG=$QDIR/gpu_queue.log
N=${1:-3}
MIN_FREE_MIB=${2:-4000}

mkdir -p "$QDIR"; touch "$QUEUE" "$QLOCK"

pop() {  # atomically pop the first queue line under the SAME lock the serial daemon uses (fd 9)
  ( flock 9; [ -s "$QUEUE" ] || exit 1; head -1 "$QUEUE"; tail -n +2 "$QUEUE" > "$QUEUE.pd.tmp" && mv "$QUEUE.pd.tmp" "$QUEUE" ) 9>>"$QLOCK"
}
freevram() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }

echo "$(date '+%F %T') PARALLEL-DISPATCH START (N=$N, min_free=${MIN_FREE_MIB}MiB)" >> "$LOG"
pids=()
while true; do
  live=(); for p in "${pids[@]:-}"; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && live+=("$p"); done; pids=("${live[@]:-}")
  nrun=0; for p in "${pids[@]:-}"; do [ -n "$p" ] && nrun=$((nrun+1)); done
  if [ ! -s "$QUEUE" ] && [ "$nrun" -eq 0 ]; then echo "$(date '+%F %T') PARALLEL-DISPATCH DONE (queue drained)" >> "$LOG"; break; fi
  if [ "$nrun" -lt "$N" ] && [ -s "$QUEUE" ]; then
    fv=$(freevram); fv=${fv:-0}
    if [ "$fv" -ge "$MIN_FREE_MIB" ]; then
      job=$(pop) || { sleep 6; continue; }
      [ -z "$job" ] && { sleep 6; continue; }
      echo "$(date '+%F %T') START(parallel): $job" >> "$LOG"
      ( setsid bash -c "$job" >> "$LOG" 2>&1; rc=$?; echo "$(date '+%F %T') DONE(rc=$rc): $job" >> "$LOG" ) &
      pids+=("$!")
      sleep 10   # stagger cupy inits so simultaneous context allocs don't spike VRAM together
      continue
    fi
  fi
  sleep 12
done
