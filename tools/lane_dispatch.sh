#!/usr/bin/env bash
# lane_dispatch.sh — keep a compute lane FULL from a persistent job queue.
#
# WHY THIS EXISTS. Parallelization was being enforced by REMEMBERING to launch things, and it failed
# repeatedly: the owner flagged under-use of the compute pool TWICE on 2026-07-29, on a day when the
# heartbeat was already printing "UNDER-FILLED-GPU" every 15 minutes and the 36-core pool sat at load 0.00.
# The warning was not the missing piece. The missing piece was that acting on it required INVENTING A JOB
# on the spot, so the cheap response was always to launch one or two things and move on.
#
# CLAUDE.md: "Drift prevention is MECHANICAL, not remembered... prefer converting a rule into a check that
# can FAIL LOUDLY." This converts "parallelize" from a decision into a loop: the dispatcher keeps N slots
# busy from a queue file, so the only thing a session must do is KEEP THE QUEUE STOCKED — which is the
# standing "build de-risks ahead of time so idle compute always has a ready job" directive, made executable.
#
#   tools/lane_dispatch.sh gpu  4      # keep 4 GPU jobs running, pulling from research/queue/gpu.queue
#   tools/lane_dispatch.sh pool 12     # keep 12 jobs per pool node from research/queue/pool.queue
#
# Queue format: one shell command per line. Blank lines and #-comments ignored. A dispatched line is moved
# to <queue>.running, then to <queue>.done on completion, so state survives a restart and nothing is lost
# or double-run.
set -uo pipefail
ROOT=/home/dant123/Projects/sim
LANE="${1:?usage: lane_dispatch.sh <gpu|pool> <slots>}"
SLOTS="${2:?need slot count}"
Q="$ROOT/research/queue/${LANE}.queue"
RUN="$Q.running"; DONE="$Q.done"; LOGD=/tmp/claude-1000/lane_$LANE
mkdir -p "$(dirname "$Q")" "$LOGD"; touch "$Q" "$RUN" "$DONE"

# Count only real jobs, never this script or its own shells (a past monitor counted its own shells and
# reported a false negative).
running_count() {
  # `pgrep -c` already prints 0 and EXITS 1 when nothing matches, so a `|| echo 0` fallback double-emits
  # "0\n0" and crashes the `$(( SLOTS - N ))` arithmetic at startup (0-running IS the startup case). Capture
  # into a var and default-if-empty instead. Pattern is lenient (`python .*-m research`) so it still matches
  # commands with interpreter flags like `-u` (`python -u -m research...`); a strict `python -m` missed those
  # and always counted 0 -> over-dispatch.
  local n
  if [ "$LANE" = gpu ]; then n=$(pgrep -fc '[.]venv/bin/python .*-m research' 2>/dev/null)
  else n=$(pgrep -fc '[l]ane_worker' 2>/dev/null); fi
  printf '%s' "${n:-0}"
}

# STARTUP SELF-CHECK (2026-08-01). A dispatcher that cannot dispatch must fail LOUDLY, not idle the fleet
# silently -- the same principle the gate registry applies to gates (refuse one whose selftest cannot fail).
# On 2026-08-01 running_count emitted "0\n0" at 0-running, crashing the arithmetic below on the FIRST
# iteration, so nothing ever dispatched and the failure was invisible. Assert the invariant that broke.
_sc=$(running_count)
if ! [ "$_sc" -ge 0 ] 2>/dev/null; then
  echo "⛔ DISPATCHER SELF-CHECK FAILED: running_count returned a non-integer [$_sc] -- FREE arithmetic would" >&2
  echo "   crash and NOTHING would dispatch. Aborting loudly instead of idling the fleet. (see running_count)" >&2
  exit 1
fi
echo "[$LANE-dispatch] self-check ok: running_count=$_sc, holding $SLOTS slots"

while true; do
  N=$(running_count)
  FREE=$(( SLOTS - N ))
  while [ "$FREE" -gt 0 ]; do
    LINE=$(grep -vE '^\s*(#|$)' "$Q" 2>/dev/null | head -1)
    [ -z "$LINE" ] && break
    # THE RECORD-CHECK GATE (2026-07-29). A line may only run if it entered via tools/queue_add.sh,
    # which forces a look at existing findings FIRST. This sits on the EXECUTION PATH deliberately:
    # `before_you_build.sh` already existed and was skipped, because running it was a thing to REMEMBER,
    # and that skip cost crux-lane slots re-running a result banked five days earlier. Unmarked lines are
    # set aside in $Q.unchecked, never silently dropped.
    case "$LINE" in
      *"#checked:"*) ;;
      *) echo "[BLOCKED $LANE] unchecked job -- requeue via tools/queue_add.sh (it reads the record first):"
         echo "               $(echo "$LINE" | cut -c1-100)"
         grep -vxF "$LINE" "$Q" > "$Q.tmp" 2>/dev/null && mv "$Q.tmp" "$Q"
         echo "$LINE" >> "$Q.unchecked"
         continue ;;
    esac
    # atomically remove that line from the queue before launching (no double-dispatch on restart)
    # ⛔ BUG FIXED 2026-07-30: this was `grep -vxF ... && mv`, and `grep -vxF` EXITS 1 WHEN IT OUTPUTS
    # NOTHING -- which is exactly the case when the queue holds only this one line. The `&&` then
    # short-circuited, `mv` never ran, the line SURVIVED, and it was re-dispatched every 60 s cycle.
    # Observed live: ONE queued job launched NINE times (staggered ages 417/294/291/288/225/222/219/36/33 s,
    # distinct parents), `.running` grown to 16 entries against 13 real processes, and the heartbeat showing a
    # stuck `queue=1` for hours. Do NOT gate the write on grep's exit status; grep -v legitimately returns 1.
    grep -vxF "$LINE" "$Q" > "$Q.tmp" 2>/dev/null || true
    mv "$Q.tmp" "$Q"
    echo "$LINE" >> "$RUN"
    TAG=$(echo "$LINE" | md5sum | cut -c1-8)
    ( cd "$ROOT" && eval "$LINE" > "$LOGD/$TAG.log" 2>&1
      echo "$LINE" >> "$DONE"
      grep -vxF "$LINE" "$RUN" > "$RUN.tmp" 2>/dev/null && mv "$RUN.tmp" "$RUN" ) &
    echo "[dispatch $LANE] -> $TAG  $(echo "$LINE" | cut -c1-90)"
    FREE=$(( FREE - 1 )); sleep 3
  done
  PEND=$(grep -cvE '^\s*(#|$)' "$Q" 2>/dev/null || echo 0)
  # THE ALARM THAT MATTERS: an empty queue, not an idle lane. An idle lane with a full queue self-heals in
  # seconds; an empty queue means the session has stopped preparing work and the lane WILL go idle.
  if [ "$PEND" -eq 0 ] && [ "$(running_count)" -eq 0 ]; then
    echo "[QUEUE EMPTY] $LANE lane drained — STOCK THE QUEUE ($Q) or this lane sits idle"
  fi
  sleep 60
done
