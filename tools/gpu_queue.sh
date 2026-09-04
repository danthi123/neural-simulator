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
    if [ -z "$job" ]; then sleep 12 8>&-; continue; fi
    # Contention guard: wait for (1) no PAUSE, (2) the GPU to be genuinely free of any brain-loading
    # process — GROUND TRUTH via nvidia-smi, not just "does our own gpu.running say something is running"
    # — and (3) raw VRAM headroom (auto-yields to a game / another run). (2) is what closes the
    # tracking-loss bug: a prior daemon incarnation can die mid-job without ever cleaning up gpu.running or
    # killing the job it launched; the orphan keeps running+holding VRAM, invisible to a freshly-started
    # daemon that only trusts its own (empty) bookkeeping, and MIN_FREE alone would never catch it at this
    # workload's typical per-job VRAM footprint. Checking residency before EVERY dispatch (not just at
    # startup) also catches a truly-standalone brain process launched outside the queue entirely.
    # GPU_QUEUE_NO_RESIDENCY_GUARD is TEST-ONLY (proves the failing direction in --selftest); it must NEVER
    # be set in production.
    while :; do
      [ -f "$PAUSE" ] && break
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
    setsid bash -c "$job" >> "$LOG" 2>&1 8>&- & jpid=$!   # own process GROUP so pause --now can kill the whole job tree (frees VRAM); 8>&- so the job never holds the daemon's singleton lock (see note above)
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
  local T="$_GPU_SELFTEST_T" SELF="$0" A="" B="" C="" D="" E="" F="" G="" H="" rc=0 out dead_pid
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
    [ -n "$untracked" ] && echo "⛔ UNTRACKED GPU-resident brain process(es), not covered by gpu.running:$untracked  -- pause --now (or game.sh on) will still stop these." ;;
  stop) [ -f "$DPID" ] && kill "$(cat "$DPID")" 2>/dev/null && rm -f "$DPID" && echo "dispatcher stopped" || echo "not running" ;;
  --selftest) selftest; exit $? ;;
  *) grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -20 ;;
esac
