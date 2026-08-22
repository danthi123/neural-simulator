#!/bin/bash
# gpu_queue.sh — LOCAL single-GPU job QUEUE. Headless (0 Claude/agent tokens), contention-safe, pausable for gaming.
#
# WHY: the mini-PC pool is CPU; heavy GPU sweeps/training must run on the local 3090 without (a) thrashing the one GPU
# with concurrent jobs, or (b) fighting a game for VRAM. This runs queued GPU jobs ONE AT A TIME and only starts a job
# when there is VRAM headroom — so it auto-yields whenever a run or a GAME already holds the card. Pause reclaims the
# GPU on demand and re-queues the killed job, so at most the current job's progress is lost ("not much work").
#
# SINGLETON across worktrees (2026-08-21): the queue + dpid + lock + daemon are ONE per repo, shared by EVERY git
# worktree (resolved at the git-common-dir root, not per-checkout). `start` from any worktree either adopts the one
# live daemon or refuses; two daemons on the one physical 3090 = concurrent brain loads = card off the bus (reboot).
#
#   tools/gpu_queue.sh start                 # launch the ONE dispatcher daemon (no-op if one is already live anywhere)
#   tools/gpu_queue.sh add '<full cmd>'      # queue a job, e.g. 'SIM_BACKEND=cupy .venv/bin/python -u -m research.runners.X --json raw/o.json'
#   tools/gpu_queue.sh pause [--now]         # stop starting new jobs; --now also KILLS the current job (frees VRAM to game) + re-queues it
#   tools/gpu_queue.sh resume                # clear pause
#   tools/gpu_queue.sh status                # running job + queue depth + pause state + GPU VRAM + shared-queue path
#   tools/gpu_queue.sh stop                  # stop the daemon (current job keeps running)
#   tools/gpu_queue.sh --selftest            # prove the singleton guard holds (isolated scratch dir; never touches the live queue)
#
# Contention knobs (env): GPU_MIN_FREE_MIB (default 3000) — required free VRAM before starting a job.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
# SINGLETON across worktrees: resolve the SHARED repo root (the parent of the ONE git-common-dir every worktree
# shares) so the queue + dpid + lock + daemon are ONE, not per-checkout. Before 2026-08-21 QDIR was relative to each
# worktree's cwd, so N worktrees each ran their OWN daemon against the ONE physical 3090 -> concurrent brain loads ->
# the card falls off the bus (reboot-only). GPU_QUEUE_DIR overrides the location; it is used ONLY by --selftest to run
# fully isolated in a scratch dir — NEVER set it in production.
GIT_COMMON=$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)
if [ -n "$GIT_COMMON" ] && [ -d "$(dirname "$GIT_COMMON")" ]; then SHARED_ROOT="$(dirname "$GIT_COMMON")"; else SHARED_ROOT="$ROOT"; fi
QDIR=${GPU_QUEUE_DIR:-"$SHARED_ROOT/research/queue"}
QUEUE=$QDIR/gpu.queue; PAUSE=$QDIR/GPU_PAUSE
RUNNING=$QDIR/gpu.running; DPID=$QDIR/gpu_queue.dpid; LOG=$QDIR/gpu_queue.log
QLOCK=$QDIR/.gpu_queue.lock                  # fd 9: serialises queue read-modify-write (add vs pop)
DLOCK=$QDIR/.gpu_daemon.lock                 # fd 8: the SINGLETON daemon lock — held for the daemon's whole life
[ "${1:-}" = "--selftest" ] || { mkdir -p "$QDIR"; touch "$QUEUE" "$QLOCK" "$DLOCK"; }
MIN_FREE=${GPU_MIN_FREE_MIB:-3000}
# `timeout` is load-bearing: when the 3090 falls off the bus (a known failure here) `nvidia-smi` HANGS rather than
# erroring, which would block the dispatcher forever inside the contention guard (the "alive but not dequeuing" wedge,
# hit twice 2026-08-20). With a timeout it returns empty -> the guard sleeps + retries until the GPU recovers.
freevram() { timeout 8 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }
# Serialise every queue read-modify-write: `add` (>> append) racing the daemon's pop (tail>tmp;mv) could clobber a
# concurrently-added job (the "queued job vanished without a START line" wedge). flock makes add + pop mutually exclusive.

# A daemon is ALIVE if its recorded pid is live OR the singleton lock is held by someone. The lock is the
# authoritative signal (it survives a lost/stale dpid — a SIGKILLed daemon frees fd 8, so the lock is honest even when
# the dpid file is not); the pid check is a fast path that also gives us a number to report. Called only in `if`
# conditions, so a `return 1` here never trips `set -e`.
daemon_alive() {
  if [ -f "$DPID" ] && kill -0 "$(cat "$DPID" 2>/dev/null)" 2>/dev/null; then return 0; fi
  if ( exec 8>"$DLOCK"; flock -n 8 ) 2>/dev/null; then return 1; fi   # acquired freely => nobody holds it => no daemon
  return 0                                                            # could not acquire => a live daemon holds it
}

daemon() {
  set +e   # a long-running dispatcher must NOT die on a single non-zero (e.g. `[ -f PAUSE ] && continue`)
  echo "$(date '+%F %T') dispatcher up (min_free=${MIN_FREE}MiB)" >> "$LOG"
  while true; do
    if [ -f "$PAUSE" ]; then sleep 8; continue; fi
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

# EXIT-trap cleanup for --selftest. It keys off the GLOBAL scratch path (not the function's locals, which are already
# out of scope when the trap fires at shell exit) and finds stray daemons by their env, so it works on BOTH the normal
# and the abnormal (FAIL / interrupt) exit path. The exact-match on GPU_QUEUE_DIR=<scratch> can NEVER match the live
# production daemon (it has no GPU_QUEUE_DIR env and a different dir), so this is safe.
_gpu_selftest_cleanup() {
  [ -n "${_GPU_SELFTEST_T:-}" ] || return 0
  local p
  for p in $(pgrep -f "gpu_queue.sh __daemon" 2>/dev/null); do
    tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx "GPU_QUEUE_DIR=$_GPU_SELFTEST_T" && kill -KILL "$p" 2>/dev/null
  done
  rm -rf "$_GPU_SELFTEST_T"
}

# --selftest: prove the singleton guard holds AND that this check fails in its failing direction. Fully isolated in a
# scratch dir (GPU_QUEUE_DIR override) with an EMPTY queue, so the spawned daemons never call nvidia-smi, never
# dispatch, never load a brain — and the cleanup safety-net only ever kills processes bound to THIS scratch dir.
selftest() {
  set +e
  # scratch dir is GLOBAL (not local) so the EXIT-trap cleanup still sees it after this function returns.
  _GPU_SELFTEST_T=$(mktemp -d "${TMPDIR:-/tmp}/gpu_queue_selftest.XXXXXX")
  trap _gpu_selftest_cleanup EXIT
  local T="$_GPU_SELFTEST_T" SELF="$0" A="" B="" C="" D="" rc=0 out
  live()    { kill -0 "$1" 2>/dev/null; }
  dpid_is() { [ "$(cat "$T/gpu_queue.dpid" 2>/dev/null)" = "$1" ]; }
  waitfor() { local i; for i in $(seq 1 40); do "$@" && return 0; sleep 0.1; done; return 1; }

  echo "== gpu_queue --selftest =="
  echo "isolated scratch dir: $T   (the live production queue + daemon are NEVER touched)"

  # ---- TEST A: the singleton REFUSES a second daemon --------------------------------------------------------------
  echo
  echo "-- TEST A: with a daemon holding the singleton lock, a 2nd daemon must NOT run --"
  GPU_QUEUE_DIR="$T" bash "$SELF" __daemon >>"$T/log" 2>&1 & A=$!
  if ! waitfor dpid_is "$A"; then echo "  FAIL(A0): daemon #1 (pid $A) never claimed the singleton lock"; return 1; fi
  echo "  daemon #1 up and holding the lock (pid $A)"
  # (A1) a direct 2nd __daemon (bypassing start's pre-check) must LOSE the lock and exit — the lock is the real guard.
  GPU_QUEUE_DIR="$T" bash "$SELF" __daemon >>"$T/log" 2>&1 & B=$!
  sleep 1
  if live "$B"; then echo "  FAIL(A1): 2nd daemon (pid $B) is STILL ALIVE -> singleton lock did NOT hold"; rc=1
  else echo "  PASS(A1): 2nd daemon lost the lock and exited (pid $B gone); #1 (pid $A) still up"; fi
  # (A2) `start` from a would-be second worktree must refuse with "already running" and spawn nothing.
  out=$(GPU_QUEUE_DIR="$T" bash "$SELF" start 2>&1)
  if echo "$out" | grep -q "already running"; then echo "  PASS(A2): start refused a 2nd daemon -> \"$out\""
  else echo "  FAIL(A2): start did NOT refuse -> \"$out\""; rc=1; fi
  # (A3) exactly one daemon survives and it is the recorded dpid.
  if dpid_is "$A" && live "$A"; then echo "  PASS(A3): recorded dpid == the one live daemon (pid $A)"
  else echo "  FAIL(A3): dpid=$(cat "$T/gpu_queue.dpid" 2>/dev/null) daemon#1_alive=$(live "$A" && echo yes || echo no)"; rc=1; fi
  kill -KILL "$A" 2>/dev/null; A=""; sleep 0.3

  # ---- TEST B: the FAILING DIRECTION (guard removed) is DETECTED --------------------------------------------------
  echo
  echo "-- TEST B (failing direction): with the singleton guard BYPASSED, 2 daemons must coexist --"
  echo "   (this proves TEST A is meaningful: it detects exactly the 2-daemon state the guard prevents)"
  GPU_QUEUE_NO_SINGLETON=1 GPU_QUEUE_DIR="$T" bash "$SELF" __daemon >>"$T/log" 2>&1 & C=$!
  GPU_QUEUE_NO_SINGLETON=1 GPU_QUEUE_DIR="$T" bash "$SELF" __daemon >>"$T/log" 2>&1 & D=$!
  sleep 1
  if live "$C" && live "$D"; then
    echo "  CONFIRMED(B): guard-bypassed daemons BOTH alive (pids $C, $D) = the double-dispatch regression."
  else
    echo "  FAIL(B): expected 2 coexisting daemons with the guard bypassed (C_alive=$(live "$C" && echo yes || echo no) D_alive=$(live "$D" && echo yes || echo no))"; rc=1
  fi
  kill -KILL "$C" "$D" 2>/dev/null; C=""; D=""

  echo
  if [ "$rc" -eq 0 ]; then echo "SELFTEST: PASS — singleton holds, and the failing direction is detectable."
  else echo "SELFTEST: FAIL"; fi
  return "$rc"
}

case "${1:-}" in
  start)
    # Refuse if a daemon is already live ANYWHERE (dpid alive OR singleton lock held) — the cross-worktree check.
    # It is advisory: the AUTHORITATIVE guard is the DLOCK acquired inside __daemon, so even a TOCTOU race here
    # (two starts both passing this check) still cannot produce two daemons.
    if daemon_alive; then echo "already running (pid $(cat "$DPID" 2>/dev/null)); shared queue: $QDIR"; exit 0; fi
    setsid bash "$0" __daemon </dev/null >>"$LOG" 2>&1 & disown 2>/dev/null || true
    # Confirm via the dpid the daemon writes AFTER it owns the lock (dpid-only: no lock probe here, so start never
    # races the lock away from the daemon it just spawned).
    for _ in $(seq 1 30); do
      if [ -f "$DPID" ] && kill -0 "$(cat "$DPID" 2>/dev/null)" 2>/dev/null; then
        echo "gpu_queue dispatcher started (pid $(cat "$DPID")); shared queue: $QDIR; log=$LOG"; exit 0
      fi
      sleep 0.1
    done
    echo "gpu_queue dispatcher start attempted but did not confirm within 3s — check: tools/gpu_queue.sh status (log=$LOG)"; exit 1 ;;
  __daemon)
    # SINGLETON: hold DLOCK on fd 8 for the whole daemon lifetime. If another daemon (from ANY worktree) already
    # holds it, this loser exits immediately WITHOUT dispatching — the lock, not start's pre-check, is the real guard
    # (two `start`s can both pass the pre-check and spawn; only one wins the lock here). When the daemon dies the fd
    # closes and the lock frees automatically. GPU_QUEUE_NO_SINGLETON is TEST-ONLY: it removes the guard so
    # --selftest can demonstrate the failing direction; it must NEVER be set in production.
    if [ -z "${GPU_QUEUE_NO_SINGLETON:-}" ]; then
      exec 8>"$DLOCK"
      if ! flock -n 8; then echo "$(date '+%F %T') __daemon: singleton lock held by a live daemon -> exiting" >> "$LOG"; exit 0; fi
    fi
    echo $$ > "$DPID"                 # I own the singleton; record MY pid at the shared root
    trap 'rm -f "$DPID"' EXIT
    daemon ;;
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
    echo "== gpu_queue =="; echo "shared queue: $QDIR"; [ -f "$PAUSE" ] && echo "state: PAUSED" || echo "state: running"
    if [ -f "$RUNNING" ]; then echo "current: $(cut -f2- "$RUNNING" | cut -c1-100) (pid $(cut -f1 "$RUNNING"))"; else echo "current: (idle)"; fi
    echo "queued: $(wc -l < "$QUEUE" 2>/dev/null || echo 0) | VRAM free: $(freevram)MiB (min_free ${MIN_FREE})"
    if daemon_alive; then echo "dispatcher: up (pid $(cat "$DPID" 2>/dev/null))"; else echo "dispatcher: DOWN (run: tools/gpu_queue.sh start)"; fi ;;
  stop) [ -f "$DPID" ] && kill "$(cat "$DPID")" 2>/dev/null && rm -f "$DPID" && echo "dispatcher stopped" || echo "not running" ;;
  --selftest) selftest; exit $? ;;
  *) grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -20 ;;
esac
