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
#
# ONE-BRAIN-PROC-AT-A-TIME (2026-09-04): the dispatcher's OWN bookkeeping (gpu.running's recorded pid,
# gpu_queue.dpid) can silently diverge from reality — a prior daemon incarnation can die mid-job (crash,
# manual stop+start, systemd Restart=always) without ever cleaning up gpu.running or killing the job it
# launched (the EXIT trap removes only the dpid); the orphaned job keeps running + holding VRAM, invisible
# to a freshly-started daemon that trusts only its own (empty) record — and MIN_FREE is far too low
# relative to this workload's typical per-job footprint (300MB-3.4GB seen in gpu_queue.log vs a 24GB card)
# to ever catch a double-start on raw VRAM headroom alone. So the dispatch guard ALSO checks actual GPU
# residency (nvidia-smi's compute-apps list, ground truth) before ever starting a new job, not just its own
# record; `pause --now` does the same before deciding what to kill. See
# research/findings/2026-09-04-gpu-queue-dispatcher-tracking-fix.md. Test-only env seams (NEVER set in
# production): GPU_QUEUE_NVIDIA_SMI (override the nvidia-smi binary — a fake script for --selftest),
# GPU_QUEUE_POLL_SEC (contention-guard poll cadence, default 12 — fast --selftest), and
# GPU_QUEUE_NO_RESIDENCY_GUARD (bypass the residency check — proves the failing direction in --selftest).
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
# shellcheck source=tools/queue_job_shape_check.sh
source "$ROOT/tools/queue_job_shape_check.sh"
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
NVIDIA_SMI=${GPU_QUEUE_NVIDIA_SMI:-nvidia-smi}     # TEST-ONLY override (a fake script for --selftest); NEVER set in production
POLL_SEC=${GPU_QUEUE_POLL_SEC:-12}                 # contention-guard poll cadence; TEST-ONLY override for a fast --selftest
# `timeout` is load-bearing: when the 3090 falls off the bus (a known failure here) `nvidia-smi` HANGS rather than
# erroring, which would block the dispatcher forever inside the contention guard (the "alive but not dequeuing" wedge,
# hit twice 2026-08-20). With a timeout it returns empty -> the guard sleeps + retries until the GPU recovers.
freevram() { timeout 8 "$NVIDIA_SMI" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null 8>&- | head -1; }
# Ground truth for "is a brain-loading GPU process resident RIGHT NOW?", independent of our own bookkeeping
# (gpu.running / $!). Same process-pattern as game.sh's gpu_python_procs() so both tools agree on what
# counts as "a brain" (a `research.runners` or `webapp` python invocation). Cross-references nvidia-smi's
# compute-apps pid list against each pid's own /proc cmdline, since nvidia-smi's own process_name field is
# just the interpreter path (".../python"), not the full argv needed to match the pattern. `timeout` guards
# the same hung-nvidia-smi failure mode as freevram() above.
gpu_resident_brain_pids() {
  local p
  for p in $(timeout 8 "$NVIDIA_SMI" --query-compute-apps=pid --format=csv,noheader 2>/dev/null 8>&- | tr -d ' \r'); do
    [ -n "$p" ] || continue
    tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -qE 'python.*(research\.runners|webapp)' && echo "$p"
  done
}

# LOCAL-LLM AUTOSWAP (2026-09-25, fix-round same day): the interactive local-llm service (tools/local_llm/llm.sh,
# ~22GB of the 24GB 3090) and this queue are NOT coordinated on their own -- a queued job starting while it is
# loaded exhausts the card (this machine has fallen off the bus on such hangs before, see
# docs/GPU_CRASH_RECOVERY.md). Mirrors the older Hermes-era tools/qwen_supervisor.sh pattern (stop the model for
# a job, reload it once idle), inlined here because nothing else drives the owner's own interactive model.
# `.local_llm_was_on` is a ONE-SHOT marker whose CONTENTS are the profile to restore, written only when THIS code
# actually VERIFIES it stopped a running unit (never fabricates an "it was on" memory, and never claims success
# on a stop that didn't take) and consumed the moment a real restore attempt is made, win or lose -- so a
# failed/hung `llm on` can never retry forever, and llm.sh's own `off` also removes it (a manual off, even
# mid-job, always cancels the pending restore -- see llm.sh's cmd_off).
#
# BLOCKER fixed here (2026-09-25 fix-round): the stop path used to write the marker and THEN call `llm.sh off`
# for the actual stop -- but `off`'s own cmd_off unconditionally deletes that SAME marker file (its own "a
# manual off cancels the pending restore" behavior), so the marker this code had just written was destroyed a
# few lines later, and the model was never auto-restored. Stopping is now done DIRECTLY via
# $LOCAL_LLM_SYSTEMCTL (never through llm.sh's `off`), and the marker is written only AFTER a verified stop.
# `llm on`'s own profile resolution / systemd-run / health-wait are still reused for the RESTORE side (that path
# never touches the marker).
LLM_SH="${GPU_QUEUE_LLM_SH:-$ROOT/tools/local_llm/llm.sh}"   # override is TEST-ONLY (a fake script); NEVER set in production
LLM_UNIT="local-llm"
LLM_WAS_ON="$QDIR/.local_llm_was_on"                         # CONTENTS = the profile to restore; presence == "gpu_queue stopped it; restore when idle"
LOCAL_LLM_SYSTEMCTL=${LOCAL_LLM_SYSTEMCTL:-systemctl}        # TEST-ONLY override (shared name with llm.sh); NEVER set in production
export LOCAL_LLM_SYSTEMCTL                                   # llm.sh (invoked as a subprocess below) must see the same stub

llm_is_active() { "$LOCAL_LLM_SYSTEMCTL" --user is-active --quiet "$LLM_UNIT" 2>/dev/null; }

# Call right before a queued job is allowed to compete for VRAM (before the freevram wait, so the wait can
# actually succeed), AND from the dispatcher's own contention loop below as a retry while the unit is still
# active. A no-op unless the unit is genuinely active -- never writes the marker for a model that was already
# down, which would fabricate a restore the owner never asked for. Stops the unit DIRECTLY via
# $LOCAL_LLM_SYSTEMCTL (see the BLOCKER note above) and only writes the marker -- carrying the profile name, so
# the restore brings back the SAME profile -- once the stop is VERIFIED (still active afterward => log the
# failure and return 1 instead of ever claiming "stopped"; the caller's own retry loop tries again).
llm_stop_for_job() {
  llm_is_active || return 0
  local prof; prof=$(bash "$LLM_SH" __current_profile 2>/dev/null || true)
  "$LOCAL_LLM_SYSTEMCTL" --user stop "$LLM_UNIT" >> "$LOG" 2>&1
  "$LOCAL_LLM_SYSTEMCTL" --user reset-failed "$LLM_UNIT" >/dev/null 2>&1 || true
  if llm_is_active; then
    echo "$(date '+%F %T') LLM-AUTOSWAP: FAILED to stop $LLM_UNIT for a queued job -- still active, will retry" >> "$LOG"
    return 1
  fi
  printf '%s\n' "$prof" > "$LLM_WAS_ON"
  echo "$(date '+%F %T') LLM-AUTOSWAP: stopped $LLM_UNIT for a queued GPU job" >> "$LOG"
}

# Call on every idle poll (queue empty). Cheap no-op unless llm_stop_for_job set the marker. Two conditions
# DEFER (leave the marker for the next idle poll, retried every ~12s) rather than give up: GPU_PAUSE (gaming --
# reloading a ~22GB model mid-game would be exactly backwards) and a brain process still GPU-resident (a job
# that outlived a dead daemon incarnation, or a truly-standalone launch -- restoring now would double-load the
# card). Only the restart attempt itself is one-shot: the marker is consumed immediately before it, so a crash
# or a failed `llm on` is never retried forever.
llm_restore_if_idle() {
  [ -f "$LLM_WAS_ON" ] || return 0
  if [ -f "$PAUSE" ]; then return 0; fi
  if [ -n "$(gpu_resident_brain_pids)" ]; then
    echo "$(date '+%F %T') LLM-AUTOSWAP: queue drained but a brain process is still GPU-resident -- deferring restore" >> "$LOG"
    return 0
  fi
  local prof; prof=$(cat "$LLM_WAS_ON" 2>/dev/null || true)
  rm -f "$LLM_WAS_ON"                       # consumed HERE, before the one attempt below
  llm_is_active && return 0                 # owner (or a race) already reloaded it manually
  echo "$(date '+%F %T') LLM-AUTOSWAP: queue drained -- restoring $LLM_UNIT${prof:+ ($prof)}" >> "$LOG"
  # backgrounded (loading can take a while, must not block dispatch) + 8>&- so this child can never keep the
  # daemon's singleton lock held after the daemon itself dies (same fd-inheritance hazard as the dispatched job).
  bash "$LLM_SH" on ${prof:+"$prof"} >> "$LOG" 2>&1 8>&- &
}

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

# Reconcile gpu.running against reality when a brain-loading process is GPU-resident but our own
# bookkeeping doesn't already (transitively) account for it — an orphan from a dead daemon incarnation, or
# a truly-standalone launch outside the queue entirely. Overwrites $RUNNING with the discovered pid (so
# `status` and `pause --now` see the truth) and logs loudly; a silent no-op when the existing record
# already covers it, so this never spams the log while a normally-tracked job is simply still running.
_adopt_resident() {
  local resident="$1" first rp
  first=$(echo "$resident" | head -1)
  if [ -f "$RUNNING" ]; then
    rp=$(cut -f1 "$RUNNING" 2>/dev/null)
    if [ -n "$rp" ] && { [ "$rp" = "$first" ] || pgrep -P "$rp" 2>/dev/null | grep -qx "$first"; }; then
      return 0
    fi
  fi
  printf '%s\t%s\n' "$first" "<adopted: GPU-resident brain process with no matching queue record (dead dispatcher incarnation, or launched outside the queue) -- see gpu_queue.log RECONCILE line>" > "$RUNNING"
  echo "$(date '+%F %T') RECONCILE: adopted untracked resident GPU pid $first into gpu.running (previous record: ${rp:-<none>}) -- refusing to start a new job until it clears" >> "$LOG"
}

daemon() {
  set +e   # a long-running dispatcher must NOT die on a single non-zero (e.g. `[ -f PAUSE ] && continue`)
  echo "$(date '+%F %T') dispatcher up (min_free=${MIN_FREE}MiB)" >> "$LOG"
  while true; do
    # `8>&-` on every forked child below: fd 8 is the singleton lock (exec'd open at __daemon startup) and
    # a fork INHERITS open fds by default, so any child that outlives the daemon keeps the flock held even
    # after the daemon itself dies -- discovered 2026-09-04 via --selftest: a stray `sleep 12` from an empty
    # queue kept a KILLED daemon's lock "held" for the rest of that sleep. The dispatched JOB is the far
    # more serious case (an ordinary fork, not the setsid-then-exec chain that later replaces its own
    # image -- exec preserves open fds too, so the inheritance survives all the way into the running job):
    # if the daemon dies mid-job WITHOUT this, the job would keep the DLOCK "held" for its entire remaining
    # runtime (hours, for the multi-day campaigns this queue runs), so a freshly-(re)started daemon could
    # never even WIN the singleton lock to reach the residency guard above -- silently defeating the whole
    # tracking-loss fix in exactly the scenario it exists for.
    if [ -f "$PAUSE" ]; then sleep 8 8>&-; continue; fi
    job=$(head -1 "$QUEUE" 2>/dev/null || true)
    if [ -z "$job" ]; then llm_restore_if_idle; sleep 12 8>&-; continue; fi
    llm_stop_for_job   # free the model's VRAM BEFORE waiting for headroom below -- retried in the loop right below if it failed
    # Contention guard: wait for (1) no PAUSE, (2) local-llm to be GENUINELY stopped (never dispatch over a
    # loaded model just because raw VRAM headroom happens to clear -- the ~22GB unit alone can still leave
    # ~3-4GB free, comfortably above MIN_FREE's default 3000MiB while very much still holding the card; a
    # stop attempt above can also simply fail, e.g. a wedged unit -- HIGH finding, 2026-09-25 fix-round), (3)
    # the GPU to be genuinely free of any OTHER brain-loading process — GROUND TRUTH via nvidia-smi, not just
    # "does our own gpu.running say something is running" — and (4) raw VRAM headroom (auto-yields to a game
    # / another run). (3) is what closes the tracking-loss bug: a prior daemon incarnation can die mid-job
    # without ever cleaning up gpu.running or killing the job it launched; the orphan keeps running+holding
    # VRAM, invisible to a freshly-started daemon that only trusts its own (empty) bookkeeping, and MIN_FREE
    # alone would never catch it at this workload's typical per-job VRAM footprint. Checking residency before
    # EVERY dispatch (not just at startup) also catches a truly-standalone brain process launched outside the
    # queue entirely. GPU_QUEUE_NO_RESIDENCY_GUARD is TEST-ONLY (proves the failing direction in --selftest);
    # it must NEVER be set in production.
    while :; do
      [ -f "$PAUSE" ] && break
      if llm_is_active; then
        # a stop attempt failed (or the owner/something else reloaded it mid-wait) -- retry rather than EVER
        # proceed to dispatch over a loaded model.
        llm_stop_for_job
        sleep "$POLL_SEC" 8>&-
        continue
      fi
      if [ -z "${GPU_QUEUE_NO_RESIDENCY_GUARD:-}" ]; then
        resident=$(gpu_resident_brain_pids)
        if [ -n "$resident" ]; then _adopt_resident "$resident"; sleep "$POLL_SEC" 8>&-; continue; fi
      fi
      f=$(freevram); [ "${f:-0}" -ge "$MIN_FREE" ] && break
      sleep "$POLL_SEC" 8>&-
    done
    [ -f "$PAUSE" ] && continue
    # pop the job atomically (flock so a concurrent `add` append is not clobbered by this rewrite)
    ( flock 9; tail -n +2 "$QUEUE" > "$QUEUE.tmp" 2>/dev/null && mv "$QUEUE.tmp" "$QUEUE" ) 9>"$QLOCK" 8>&-
    echo "$(date '+%F %T') START: $job" >> "$LOG"
    start_s=$(date +%s)
    setsid bash -c "$job" >> "$LOG" 2>&1 8>&- & jpid=$!   # own process GROUP so pause --now can kill the whole job tree (frees VRAM); 8>&- so the job never holds the daemon's singleton lock (see note above)
    printf '%s\t%s\n' "$jpid" "$job" > "$RUNNING"
    wait "$jpid" 2>/dev/null; rc=$?
    rm -f "$RUNNING"
    dur=$(( $(date +%s) - start_s ))
    echo "$(date '+%F %T') DONE(rc=$rc): $job" >> "$LOG"
    # FAST-FAIL, LOUDLY (2026-09-25). rc=127 ("command not found") or rc=2 (a shell syntax/usage error) inside
    # FAST_FAIL_S of START means the job never actually ran -- it died on argv[0]/syntax, exactly like the
    # historical `status` job (2026-08-31/09-01, three cycles, rc=127 in under a second each time) that sat in
    # this very log as an ordinary, unremarkable DONE(rc=127) line for weeks. A plain DONE line is easy to miss
    # in a log this size; a distinct marker is not. tools/queue_job_shape_check.sh now refuses that SHAPE at
    # enqueue time, but this catches whatever it cannot see (e.g. a module importable locally but not on the
    # box actually running the job, or a bug in the shape check itself) -- belt and suspenders.
    if [ "$rc" -eq 127 ] || [ "$rc" -eq 2 ]; then
      if [ "$dur" -le "${GPU_QUEUE_FAST_FAIL_S:-10}" ]; then
        echo "$(date '+%F %T') ⛔ FAST-FAIL: rc=$rc after ${dur}s (died on argv[0]/syntax, not a real run): $job" >> "$LOG"
      fi
    fi
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
  # TEST C/D spawn fake "brain" processes (argv0 rewritten via `exec -a`) to simulate an orphaned/
  # standalone GPU job; kill any still alive if the selftest aborted before its own cleanup ran. Only ever
  # a pid this same selftest spawned, recorded inside the isolated scratch dir.
  [ -f "$_GPU_SELFTEST_T/fake_resident_pid" ] && kill -KILL "$(cat "$_GPU_SELFTEST_T/fake_resident_pid" 2>/dev/null)" 2>/dev/null
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
  local T="$_GPU_SELFTEST_T" SELF="$0" A="" B="" C="" D="" E="" F="" G="" H="" I="" J="" rc=0 out dead_pid
  live()    { kill -0 "$1" 2>/dev/null; }
  dpid_is() { [ "$(cat "$T/gpu_queue.dpid" 2>/dev/null)" = "$1" ]; }
  lock_free() { ( exec 8>"$T/.gpu_daemon.lock"; flock -n 8 ) 2>/dev/null; }
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
  kill -KILL "$C" "$D" 2>/dev/null; C=""; D=""; sleep 0.3

  # ---- TEST C: an untracked GPU-resident brain process BLOCKS a new dispatch (the tracking-loss / --------
  # ---- double-start bug: a job survives a dead dispatcher incarnation with no record of it) -------------
  echo
  echo "-- TEST C: a GPU-resident brain process with NO queue record must block a new dispatch --"
  cat > "$T/fake_nvidia_smi.sh" <<FAKEEOF
#!/bin/bash
case "\$*" in
  *--query-compute-apps*)
    if [ -s "$T/fake_resident_pid" ]; then
      p=\$(cat "$T/fake_resident_pid")
      if [ -n "\$p" ] && kill -0 "\$p" 2>/dev/null; then echo "\$p"; fi
    fi ;;
  *--query-gpu*) echo "99999" ;;
esac
FAKEEOF
  chmod +x "$T/fake_nvidia_smi.sh"
  # setsid'd, like a REAL queued job would be (own process group -- this is what makes group-kills safe to
  # test; a bare, non-setsid background job here would share ITS group with the selftest script itself).
  setsid bash -c 'exec -a "python -u -m research.runners.faketest_orphan" sleep 60' >/dev/null 2>&1 & E=$!
  echo "$E" > "$T/fake_resident_pid"
  echo "  simulated orphan (untracked, GPU-\"resident\" per the fake nvidia-smi): pid $E"
  GPU_QUEUE_DIR="$T" bash "$SELF" add "touch $T/second_job_ran" >/dev/null
  # (C1) with the residency guard ACTIVE, the queued job must NOT run while the orphan is "resident".
  GPU_QUEUE_DIR="$T" GPU_QUEUE_NVIDIA_SMI="$T/fake_nvidia_smi.sh" GPU_QUEUE_POLL_SEC=1 \
    bash "$SELF" __daemon >>"$T/log" 2>&1 & F=$!
  sleep 3
  if [ -f "$T/second_job_ran" ]; then
    echo "  FAIL(C1): the second job ran WHILE an untracked resident process was still \"on the GPU\" -> double-start"; rc=1
  else
    echo "  PASS(C1): second job correctly held back while the untracked resident process persists"
  fi
  if [ -s "$T/gpu.queue" ]; then echo "  PASS(C1b): the queued job was never popped (peeks-don't-pops held under the guard)"
  else echo "  FAIL(C1b): the queued job was popped despite the guard"; rc=1; fi
  kill -KILL "$F" 2>/dev/null
  # Wait for the singleton lock to be GENUINELY free (not a fixed sleep) before starting G -- a SIGKILLed
  # daemon's fd-close-driven flock release can lag under load, and a blind short sleep here would make G
  # itself lose the (real, working) singleton race and exit immediately, misreported as C2 refuting the
  # residency guard when it's really just testing infra flakiness.
  if ! waitfor lock_free; then echo "  FAIL(C-setup): daemon #1's singleton lock never freed after kill -- test infra issue, not the guard"; rc=1; fi
  # (C2) the FAILING DIRECTION: with the residency guard bypassed, the same orphan does NOT block the
  # second job -- proves C1 is meaningful (it detects exactly the double-start the guard prevents).
  GPU_QUEUE_DIR="$T" GPU_QUEUE_NVIDIA_SMI="$T/fake_nvidia_smi.sh" GPU_QUEUE_POLL_SEC=1 GPU_QUEUE_NO_RESIDENCY_GUARD=1 \
    bash "$SELF" __daemon >>"$T/log" 2>&1 & G=$!
  if waitfor test -f "$T/second_job_ran"; then
    echo "  CONFIRMED(C2): guard-bypassed daemon double-started the second job over the untracked resident process."
  else
    echo "  FAIL(C2): expected the second job to run once the guard is bypassed (it didn't -- C1 may not be testing what we think)"; rc=1
  fi
  kill -KILL "$G" 2>/dev/null; kill -KILL "$E" 2>/dev/null
  rm -f "$T/fake_resident_pid" "$T/second_job_ran"; sleep 0.3

  # ---- TEST D: pause --now must reach a GPU-resident process even when gpu.running's own record is ------
  # ---- stale/wrong (proves the "standalone job a normal pause can't stop" symptom is fixed) -------------
  echo
  echo "-- TEST D: pause --now must kill a GPU-resident brain process even with a STALE gpu.running record --"
  # setsid'd, like a REAL standalone job would be (own process group -- see TEST C's spawn for why this
  # matters: a non-isolated bare background job here would share ITS group with the selftest script itself,
  # and pause --now's group-kill safety check would then correctly refuse to touch it, defeating the test).
  setsid bash -c 'exec -a "python -u -m research.runners.faketest_standalone" sleep 60' >/dev/null 2>&1 & H=$!
  echo "$H" > "$T/fake_resident_pid"
  ( : ) & dead_pid=$!; wait "$dead_pid" 2>/dev/null   # a pid guaranteed to be dead (just spawned + reaped)
  printf '%s\t%s\n' "$dead_pid" "stale-record-of-a-job-that-no-longer-corresponds-to-anything-real" > "$T/gpu.running"
  echo "  simulated standalone process (GPU-\"resident\", NOT reachable via gpu.running's stale pid $dead_pid): pid $H"
  out=$(GPU_QUEUE_DIR="$T" GPU_QUEUE_NVIDIA_SMI="$T/fake_nvidia_smi.sh" bash "$SELF" pause --now 2>&1)
  echo "  pause --now said: \"$out\""
  sleep 0.5
  if kill -0 "$H" 2>/dev/null; then
    echo "  FAIL(D): pid $H is STILL ALIVE after pause --now -> the stale record hid the real job from pause"; rc=1
  else
    echo "  PASS(D): pause --now killed the genuinely-resident process ($H) despite gpu.running naming an unrelated dead pid"
  fi
  rm -f "$T/fake_resident_pid" "$T/gpu.running" "$T/GPU_PAUSE"

  # ---- TEST E: a job that dies almost instantly with rc=127/2 is logged as a LOUD FAST-FAIL, not buried in --
  # ---- an ordinary DONE line (2026-09-25: the historical `status` job did exactly this, unflagged, 3x) ------
  echo
  echo "-- TEST E: a fast rc=127 job is logged as a loud FAST-FAIL, an rc=0 job is NOT --"
  # Dispatch messages (START/DONE/FAST-FAIL) are written to \$LOG (\$T/gpu_queue.log), NOT the ">>...  2>&1"
  # redirection target below (that one only catches the daemon subprocess's own raw stdout/stderr).
  DLOG="$T/gpu_queue.log"
  if ! waitfor lock_free; then echo "  FAIL(E-setup): singleton lock not free before TEST E (prior daemon's kill hadn't released it yet)"; rc=1; fi
  rm -f "$DLOG"
  # A job whose FIRST WORD resolves (so the enqueue-time shape check accepts it -- see
  # tools/queue_job_shape_check.sh) but whose TARGET does not -- the exact `status`-job shape: `env` is a real
  # command, the thing it tries to exec is not, so the real dispatch still dies with rc=127 in well under a
  # second.
  GPU_QUEUE_DIR="$T" bash "$SELF" add "env this_command_does_not_exist_xyz_12345" >/dev/null
  GPU_QUEUE_DIR="$T" GPU_QUEUE_NVIDIA_SMI="$T/fake_nvidia_smi.sh" GPU_QUEUE_POLL_SEC=1 \
    bash "$SELF" __daemon >/dev/null 2>&1 & I=$!
  if waitfor grep -q 'DONE(rc=127)' "$DLOG"; then
    if grep -q 'FAST-FAIL: rc=127' "$DLOG"; then
      echo "  PASS(E1): a fast rc=127 job is logged as a loud FAST-FAIL"
    else
      echo "  FAIL(E1): job died rc=127 fast but NO FAST-FAIL line was logged -- exactly the \`status\` bug"; rc=1
    fi
  else
    echo "  FAIL(E-setup): the rc=127 job never completed (log tail: $(tail -3 "$DLOG" 2>/dev/null))"; rc=1
  fi
  kill -KILL "$I" 2>/dev/null
  if ! waitfor lock_free; then echo "  FAIL(E-setup): daemon #E1's singleton lock never freed after kill"; rc=1; fi
  # (E2) the FAILING DIRECTION: an ordinary rc=0 job must NEVER be flagged -- proves E1 is a real signal, not a
  # marker stamped on every completion regardless of outcome.
  rm -f "$DLOG"
  GPU_QUEUE_DIR="$T" bash "$SELF" add "true" >/dev/null
  GPU_QUEUE_DIR="$T" GPU_QUEUE_NVIDIA_SMI="$T/fake_nvidia_smi.sh" GPU_QUEUE_POLL_SEC=1 \
    bash "$SELF" __daemon >/dev/null 2>&1 & J=$!
  if waitfor grep -q 'DONE(rc=0)' "$DLOG"; then
    if grep -q 'FAST-FAIL' "$DLOG"; then
      echo "  FAIL(E2): a normal rc=0 job was wrongly flagged FAST-FAIL"; rc=1
    else
      echo "  PASS(E2): a normal rc=0 job is correctly never flagged"
    fi
  else
    echo "  FAIL(E2-setup): the rc=0 job never completed (log tail: $(tail -3 "$DLOG" 2>/dev/null))"; rc=1
  fi
  kill -KILL "$J" 2>/dev/null

  echo
  if [ "$rc" -eq 0 ]; then echo "SELFTEST: PASS — singleton holds, the residency guard holds (+ both failing directions are detectable), and pause --now reaches a genuinely-resident job even with a stale record."
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
    # SHAPE GATE (2026-09-25, tools/queue_job_shape_check.sh): refuse a line whose first word could not
    # possibly run (a prose label, a torn line, a syntax error) BEFORE it is queued -- see that file's header
    # for the real historical failures (the SETTLE A2 pool lines; a bare `status` job in this very log,
    # 2026-08-31/09-01, three separate cycles, rc=127 each time, never flagged).
    if ! SHAPE_MSG=$(queue_job_runnable_check "$2"); then echo "$SHAPE_MSG" >&2; exit 1; fi
    ( flock 9; printf '%s\n' "$2" >> "$QUEUE" ) 9>"$QLOCK"; echo "queued (depth $(wc -l < "$QUEUE")): ${2:0:80}" ;;
  pause)
    touch "$PAUSE"
    # --now: reclaim the GPU immediately. Retry the running-job lookup briefly (the daemon writes $RUNNING just
    # after launch, so a job started sub-second ago may not be recorded yet).
    if [ "${2:-}" = "--now" ]; then for _ in 1 2 3 4 5; do [ -f "$RUNNING" ] && break; sleep 0.5; done; fi
    if [ "${2:-}" = "--now" ]; then
      # Kill the UNION of every plausible target, not just the recorded pid: its own process group (the
      # common, verified case — the exec chain collapses so the recorded pid already IS the compute proc),
      # its live descendants (defense-in-depth for a job shape that forks instead of exec'ing), AND every
      # GPU-resident brain pid nvidia-smi reports (the fully-lost/orphaned case — a job our own bookkeeping
      # never saw, e.g. because a prior daemon incarnation died mid-job). This is what lets a plain
      # `pause --now` (game.sh's non-force path) reach a "standalone" job without needing game.sh's own
      # --force sweep: tracking loss here no longer means pause can't reach the real process.
      targets=""; j=""
      if [ -f "$RUNNING" ]; then
        p=$(cut -f1 "$RUNNING"); j=$(cut -f2- "$RUNNING")
        targets="$p $(pgrep -P "$p" 2>/dev/null | tr '\n' ' ')"
        for p2 in $(pgrep -P "$p" 2>/dev/null); do targets="$targets $(pgrep -P "$p2" 2>/dev/null | tr '\n' ' ')"; done
      fi
      targets="$targets $(gpu_resident_brain_pids | tr '\n' ' ')"
      targets=$(echo "$targets" | xargs -n1 2>/dev/null | sort -u | xargs)
      if [ -n "$targets" ]; then
        # Group-kill (-pgid) ONLY when the target IS its own process-group leader (pgid == its own pid) --
        # true for anything descended from our own `setsid bash -c "$job"` launch (setsid makes the job its
        # own leader), which is what makes a group-wide kill safe there (catches python's own children in
        # one shot). A pid surfaced by gpu_resident_brain_pids() alone has UNKNOWN provenance -- it could be
        # a bare process sharing its group with an unrelated shell/session (never launched through this
        # queue at all) -- so for anything that ISN'T confirmed to be its own leader, kill ONLY that specific
        # pid. Blindly doing `-pgid` for every discovered target risks signalling a whole unrelated session.
        # Every kill is `|| true`: a target can legitimately already be dead (a stale/adopted-placeholder
        # record, or one that finished between discovery and this loop) -- `2>/dev/null` alone only hides
        # the "No such process" message, NOT kill's nonzero exit, and this whole case runs under `set -e`
        # (unlike daemon(), which explicitly turns it off) -- an unguarded bare `kill ... "$t"` on an
        # already-dead pid aborts the WHOLE sweep right there, silently skipping every later target
        # (discovered 2026-09-04 via --selftest TEST D's deliberately-stale record).
        for t in $targets; do
          pg=$(ps -o pgid= -p "$t" 2>/dev/null | tr -d ' ')
          if [ -n "$pg" ] && [ "$pg" = "$t" ]; then kill -TERM -"$pg" 2>/dev/null || true; fi
          kill -TERM "$t" 2>/dev/null || true
        done
        sleep 2
        for t in $targets; do
          pg=$(ps -o pgid= -p "$t" 2>/dev/null | tr -d ' ')
          if [ -n "$pg" ] && [ "$pg" = "$t" ]; then kill -KILL -"$pg" 2>/dev/null || true; fi
          kill -KILL "$t" 2>/dev/null || true
        done
        if [ -f "$RUNNING" ] && [ -n "$j" ] && [ "${j#<adopted:}" = "$j" ]; then
          # re-queue the killed job at the FRONT so resume re-runs it -- but only when $j is a REAL command
          # (skip a stale-pid record with no matching job string, and skip _adopt_resident's own
          # "<adopted: ...>" placeholder text, which is a description for humans, not a runnable command).
          tmp=$(mktemp); printf '%s\n' "$j" > "$tmp"; cat "$QUEUE" >> "$tmp"; mv "$tmp" "$QUEUE"
        fi
        rm -f "$RUNNING"
        echo "PAUSED + killed current job/process(es) (VRAM freed) [pids: $targets]; re-queued it at the front for resume if it was a tracked job."
      else
        echo "PAUSED (nothing was recorded as running, and no GPU-resident brain process was found)."
      fi
    else echo "PAUSED (current job finishes; no new jobs start). Use --now to reclaim the GPU immediately."; fi ;;
  resume) rm -f "$PAUSE"; echo "RESUMED." ;;
  status)
    echo "== gpu_queue =="; echo "shared queue: $QDIR"; [ -f "$PAUSE" ] && echo "state: PAUSED" || echo "state: running"
    if [ -f "$RUNNING" ]; then echo "current: $(cut -f2- "$RUNNING" | cut -c1-100) (pid $(cut -f1 "$RUNNING"))"; else echo "current: (idle)"; fi
    echo "queued: $(wc -l < "$QUEUE" 2>/dev/null || echo 0) | VRAM free: $(freevram)MiB (min_free ${MIN_FREE})"
    if [ -f "$DPID" ] && kill -0 "$(cat "$DPID" 2>/dev/null)" 2>/dev/null; then
      echo "dispatcher: up (pid $(cat "$DPID" 2>/dev/null))"
    elif daemon_alive; then
      echo "dispatcher: up (lock-held; recorded dpid $(cat "$DPID" 2>/dev/null) is STALE/dead -- a live daemon holds the singleton lock but its actual pid is unknown from here; harmless but worth a look)"
    else
      echo "dispatcher: DOWN (run: tools/gpu_queue.sh start)"
    fi
    # Ground-truth cross-check: a pid GPU-resident RIGHT NOW that gpu.running doesn't (transitively) cover
    # is exactly the tracking-loss bug this file guards against (an orphan from a dead dispatcher
    # incarnation, or a truly-standalone launch outside the queue) -- surface it loudly rather than
    # silently trusting the record. `pause --now` sweeps these regardless of whether status is ever checked.
    rp=$([ -f "$RUNNING" ] && cut -f1 "$RUNNING" 2>/dev/null || echo "")
    untracked=""
    for rpid in $(gpu_resident_brain_pids); do
      if [ "$rpid" != "$rp" ] && ! pgrep -P "$rp" 2>/dev/null | grep -qx "$rpid"; then untracked="$untracked $rpid"; fi
    done
    # `if` (not `test && echo`) deliberately: the healthy/common case (untracked empty) must exit 0, not fall
    # through to the LAST command's own exit status. `[ -n "$untracked" ] && echo ...` returns the TEST's
    # failure (1) whenever there is nothing to warn about, so `gpu_queue.sh status` reported rc=1 on every
    # healthy call -- a read-only diagnostic that could never signal success via its own exit code (caught
    # 2026-09-09 by `tools/tool_health.py` marking gpu-queue permanently ROTTED although the printed status
    # was fine; `tests/test_gpu_queue_status_exit.py::test_status_healthy_exits_zero` pins this).
    if [ -n "$untracked" ]; then
      echo "⛔ UNTRACKED GPU-resident brain process(es), not covered by gpu.running:$untracked  -- pause --now (or game.sh on) will still stop these."
    fi
    exit 0 ;;
  stop) [ -f "$DPID" ] && kill "$(cat "$DPID")" 2>/dev/null && rm -f "$DPID" && echo "dispatcher stopped" || echo "not running" ;;
  # TEST-ONLY hidden entry points: run one autoswap decision synchronously, without the daemon's infinite loop,
  # so tests can drive it deterministically with stubbed LOCAL_LLM_SYSTEMCTL / GPU_QUEUE_NVIDIA_SMI /
  # GPU_QUEUE_LLM_SH. The real daemon() calls the same functions in-process (see above).
  __llm_stop_for_job) llm_stop_for_job ;;
  __llm_restore_if_idle) llm_restore_if_idle ;;
  --selftest) selftest; exit $? ;;
  *) grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -20 ;;
esac
