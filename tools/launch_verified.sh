#!/usr/bin/env bash
# launch_verified.sh — launch a background job and PROVE it is running before reporting success.
#
#   bash tools/launch_verified.sh <logfile> <command...>
#   bash tools/launch_verified.sh /path/run.log env SIM_BACKEND=cupy .venv/bin/python -m research.runners.foo --seeds 42
#
# Exits 0 only when the process is alive AND its CPU time has ADVANCED between two samples. Exits 1 otherwise,
# printing the log tail so the cause is on screen instead of needing a separate hunt.
#
# WHY (2026-07-30, after this failed FOUR times in one session, three of them caught only by a heartbeat):
#   * Six crux cells were launched with bash's `set -- $PAIR` idiom. THIS SHELL IS FISH, where `set` is a
#     different builtin with no positional-splitting semantics, so every cell ran as `--seeds "43 fixed_fa"
#     --arms ""`, died instantly on argument parsing, and wrote its log to a path containing a literal space.
#     I reported the crux as "fully parallel" while six of nine cells did not exist.
#   * The bug was VISIBLE IN MY OWN LAUNCH OUTPUT (`s43 fixed_fa/` instead of `s43/fixed_fa`) and I read past it.
#   * Earlier the same day: a runner re-scoped and reported as started that was never started; a return-arity
#     change that killed a run at a third call site; and a "1 cell completed" read off a stamp file I had
#     written myself minutes earlier.
# The common shape is NOT bad science — it is treating "the command returned" as "the work is running".
# `nohup ... &` ALWAYS succeeds: the shell forks, then the child dies on its own. $! is a valid pid for a process
# that is already dead. So the launch can never fail loudly on its own; something has to check.
#
# WHY CPU-TIME-ADVANCED RATHER THAN JUST ALIVE: a process can be alive and doing nothing (the 2026-07-24 failure
# was a live-but-stalled run, not an idle one), and a freshly forked process is briefly alive even when doomed.
# Requiring CPU to ADVANCE between samples distinguishes computing from merely existing — the same distinction
# the session heartbeat exists to enforce.
set -uo pipefail

LOG="${1:?usage: launch_verified.sh <logfile> <command...>}"; shift
[ "$#" -ge 1 ] || { echo "launch_verified: no command given" >&2; exit 2; }

WAIT_A="${LAUNCH_VERIFY_WAIT_A:-8}"     # first sample delay
WAIT_B="${LAUNCH_VERIFY_WAIT_B:-12}"    # second sample delay

mkdir -p "$(dirname "$LOG")"
case "$LOG" in *" "*) echo "launch_verified: REFUSING — log path contains a space: [$LOG]" >&2
  echo "  A space here almost always means a shell-splitting bug upstream (fish vs bash \`set --\`)." >&2
  exit 1 ;;
esac

nohup "$@" > "$LOG" 2>&1 &
PID=$!

cpu_ticks() { awk '{print $14+$15}' /proc/"$1"/stat 2>/dev/null; }

sleep "$WAIT_A"
if ! kill -0 "$PID" 2>/dev/null; then
  echo "⛔ launch FAILED — pid $PID is already dead after ${WAIT_A}s."
  echo "   command: $*"
  echo "   --- log tail ---"; tail -15 "$LOG" 2>/dev/null | sed 's/^/   /'
  exit 1
fi
T1=$(cpu_ticks "$PID")

sleep "$WAIT_B"
if ! kill -0 "$PID" 2>/dev/null; then
  echo "⛔ launch FAILED — pid $PID died between ${WAIT_A}s and $((WAIT_A+WAIT_B))s."
  echo "   command: $*"
  echo "   --- log tail ---"; tail -15 "$LOG" 2>/dev/null | sed 's/^/   /'
  exit 1
fi
T2=$(cpu_ticks "$PID")

if [ -z "${T1:-}" ] || [ -z "${T2:-}" ] || [ "$T2" -le "${T1:-0}" ]; then
  echo "⚠️  pid $PID is ALIVE but its CPU time did not advance (${T1:-?} -> ${T2:-?} ticks) over ${WAIT_B}s."
  echo "   Alive is not the same as computing. Check before trusting this job."
  echo "   command: $*"
  echo "   --- log tail ---"; tail -10 "$LOG" 2>/dev/null | sed 's/^/   /'
  exit 1
fi

echo "✅ VERIFIED running: pid=$PID cpu $((T2-T1)) ticks in ${WAIT_B}s"
echo "   log: $LOG"
