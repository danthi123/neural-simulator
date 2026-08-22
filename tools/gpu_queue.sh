#!/bin/bash
# gpu_queue.sh — LOCAL single-GPU job QUEUE. Headless (0 Claude/agent tokens), contention-safe, pausable for gaming.
#
# WHY: the mini-PC pool is CPU; heavy GPU sweeps/training must run on the local 3090 without (a) thrashing the one GPU
# with concurrent jobs, or (b) fighting a game for VRAM. This runs queued GPU jobs ONE AT A TIME and only starts a job
# when there is VRAM headroom — so it auto-yields whenever a run or a GAME already holds the card. Pause reclaims the
# GPU on demand and re-queues the killed job, so at most the current job's progress is lost ("not much work").
#
#   tools/gpu_queue.sh start                 # launch the dispatcher daemon (nohup, survives the shell)
#   tools/gpu_queue.sh add '<full cmd>'      # queue a job, e.g. 'SIM_BACKEND=cupy .venv/bin/python -u -m research.runners.X --json raw/o.json'
#   tools/gpu_queue.sh pause [--now]         # stop starting new jobs; --now also KILLS the current job (frees VRAM to game) + re-queues it
#   tools/gpu_queue.sh resume                # clear pause
#   tools/gpu_queue.sh status                # running job + queue depth + pause state + GPU VRAM
#   tools/gpu_queue.sh stop                  # stop the daemon (current job keeps running)
#
# Contention knobs (env): GPU_MIN_FREE_MIB (default 3000) — required free VRAM before starting a job.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
QDIR=research/queue; QUEUE=$QDIR/gpu.queue; PAUSE=$QDIR/GPU_PAUSE
RUNNING=$QDIR/gpu.running; DPID=$QDIR/gpu_queue.dpid; LOG=$QDIR/gpu_queue.log
QLOCK=$QDIR/.gpu_queue.lock
mkdir -p "$QDIR"; touch "$QUEUE" "$QLOCK"
MIN_FREE=${GPU_MIN_FREE_MIB:-3000}
# `timeout` is load-bearing: when the 3090 falls off the bus (a known failure here) `nvidia-smi` HANGS rather than
# erroring, which would block the dispatcher forever inside the contention guard (the "alive but not dequeuing" wedge,
# hit twice 2026-08-20). With a timeout it returns empty -> the guard sleeps + retries until the GPU recovers.
freevram() { timeout 8 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }
# Serialise every queue read-modify-write: `add` (>> append) racing the daemon's pop (tail>tmp;mv) could clobber a
# concurrently-added job (the "queued job vanished without a START line" wedge). flock makes add + pop mutually exclusive.

daemon() {
  set +e   # a long-running dispatcher must NOT die on a single non-zero (e.g. `[ -f PAUSE ] && continue`)
  echo "$(date '+%F %T') dispatcher up (min_free=${MIN_FREE}MiB)" >> "$LOG"
  while true; do
    if [ -f "$PAUSE" ]; then sleep 8; continue; fi
    # RESTART-SAFETY (2026-08-22): never run two GPU jobs at once. A job launched by a PREVIOUS daemon survives
    # that daemon dying (own process group via setsid) — after a crash + systemd Restart=, or a stop+restart
    # migration. If $RUNNING names a still-alive pid, WAIT for it; only clear $RUNNING once its pid is gone.
    # Without this, a daemon restart double-starts a job while the old one is still on the GPU.
    if [ -f "$RUNNING" ]; then
      rpid=$(cut -f1 "$RUNNING" 2>/dev/null)
      if [ -n "$rpid" ] && kill -0 "$rpid" 2>/dev/null; then sleep 12; continue; fi
      rm -f "$RUNNING"
    fi
    job=$(head -1 "$QUEUE" 2>/dev/null || true)
    if [ -z "$job" ]; then sleep 12; continue; fi
    # contention guard: wait for VRAM headroom (auto-yields to a game / another run) and respect pause
    while :; do [ -f "$PAUSE" ] && break; f=$(freevram); [ "${f:-0}" -ge "$MIN_FREE" ] && break; sleep 12; done
    [ -f "$PAUSE" ] && continue
    # pop the job atomically (flock so a concurrent `add` append is not clobbered by this rewrite)
    ( flock 9; tail -n +2 "$QUEUE" > "$QUEUE.tmp" 2>/dev/null && mv "$QUEUE.tmp" "$QUEUE" ) 9>"$QLOCK"
    echo "$(date '+%F %T') START: $job" >> "$LOG"
    setsid bash -c "$job" >> "$LOG" 2>&1 & jpid=$!   # own process GROUP so pause --now can kill the whole job tree (frees VRAM)
    printf '%s\t%s\n' "$jpid" "$job" > "$RUNNING"
    wait "$jpid" 2>/dev/null; rc=$?
    rm -f "$RUNNING"
    echo "$(date '+%F %T') DONE(rc=$rc): $job" >> "$LOG"
  done
}

case "${1:-}" in
  start)
    if [ -f "$DPID" ] && kill -0 "$(cat "$DPID")" 2>/dev/null; then echo "already running (pid $(cat "$DPID"))"; exit 0; fi
    setsid bash "$0" __daemon </dev/null >>"$LOG" 2>&1 & echo $! > "$DPID"; disown 2>/dev/null || true
    echo "gpu_queue dispatcher started (pid $(cat "$DPID")); log=$LOG" ;;
  __daemon) echo $$ > "$DPID"; daemon ;;   # record pid so `status`/`stop` work even when launched by systemd
  add)
    [ -z "${2:-}" ] && { echo 'usage: add "<full gpu command incl. --json out>"' >&2; exit 1; }
    ( flock 9; printf '%s\n' "$2" >> "$QUEUE" ) 9>"$QLOCK"; echo "queued (depth $(wc -l < "$QUEUE")): ${2:0:80}" ;;
  pause)
    touch "$PAUSE"
    # --now: reclaim the GPU immediately. Retry the running-job lookup briefly (the daemon writes $RUNNING just
    # after launch, so a job started sub-second ago may not be recorded yet).
    if [ "${2:-}" = "--now" ]; then for _ in 1 2 3 4 5; do [ -f "$RUNNING" ] && break; sleep 0.5; done; fi
    if [ "${2:-}" = "--now" ] && [ -f "$RUNNING" ]; then
      p=$(cut -f1 "$RUNNING"); j=$(cut -f2- "$RUNNING")
      pg=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')       # the job's ACTUAL process group (job + python children)
      [ -n "$pg" ] && kill -TERM -"$pg" 2>/dev/null; kill -TERM "$p" 2>/dev/null; sleep 2
      [ -n "$pg" ] && kill -KILL -"$pg" 2>/dev/null; kill -KILL "$p" 2>/dev/null || true
      # re-queue the killed job at the FRONT so resume re-runs it
      tmp=$(mktemp); printf '%s\n' "$j" > "$tmp"; cat "$QUEUE" >> "$tmp"; mv "$tmp" "$QUEUE"; rm -f "$RUNNING"
      echo "PAUSED + killed current job (VRAM freed); re-queued it at the front for resume."
    else echo "PAUSED (current job finishes; no new jobs start). Use --now to reclaim the GPU immediately."; fi ;;
  resume) rm -f "$PAUSE"; echo "RESUMED." ;;
  status)
    echo "== gpu_queue =="; [ -f "$PAUSE" ] && echo "state: PAUSED" || echo "state: running"
    if [ -f "$RUNNING" ]; then echo "current: $(cut -f2- "$RUNNING" | cut -c1-100) (pid $(cut -f1 "$RUNNING"))"; else echo "current: (idle)"; fi
    echo "queued: $(wc -l < "$QUEUE" 2>/dev/null || echo 0) | VRAM free: $(freevram)MiB (min_free ${MIN_FREE})"
    [ -f "$DPID" ] && kill -0 "$(cat "$DPID")" 2>/dev/null && echo "dispatcher: up (pid $(cat "$DPID"))" || echo "dispatcher: DOWN (run: tools/gpu_queue.sh start)" ;;
  stop) [ -f "$DPID" ] && kill "$(cat "$DPID")" 2>/dev/null && rm -f "$DPID" && echo "dispatcher stopped" || echo "not running" ;;
  *) grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -18 ;;
esac
