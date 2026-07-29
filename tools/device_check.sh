#!/usr/bin/env bash
# device_check.sh — is each RUNNING job actually on the device you think? Exit 1 if any is on CPU.
#
# WHY (2026-07-29). The roadmap's crux ran **47 minutes on the CPU** while the GPU was free. Everything
# else was correct: the construct smoke passed, lane coverage passed, queue position was right, slots had
# been raised so it could start, and CPU-time tracked elapsed-time at ~99% — so every liveness and
# scheduling indicator read healthy. The job was genuinely computing, on the wrong device. The runner
# printed the diagnosis in its FIRST LINE and it was scrolled past while reading for a verdict.
#
# Today's other mechanisms are all SCHEDULING checks (is the right work queued, in the right lane, in the
# right order, checked against the record). None of them asks **"is the running job doing what I think it
# is doing"**. This does.
#
#   bash tools/device_check.sh          # table + exit 1 on any CPU-bound job
#   bash tools/device_check.sh --quiet  # one line, for the heartbeat
#
# STALENESS IS HANDLED DELIBERATELY: a log older than its process is a LEFTOVER from a previous run and
# must not be read as the current device. (Immediately after killing the CPU crux runs, their 62-minute-old
# logs still said "numpy" and were nearly misread as a repeat failure.)
set -uo pipefail
QUIET=0; [ "${1:-}" = "--quiet" ] && QUIET=1
BAD=0; N=0; LINES=""

while read -r PID ETIME ARGS; do
  [ -z "${PID:-}" ] && continue
  N=$((N+1))
  RUNNER=$(echo "$ARGS" | grep -oE 'research\.runners\.[._a-zA-Z0-9]+' | sed 's/research\.runners\.//' | head -1)
  # ps ARGS does NOT contain shell redirects (the shell consumes them), so the log path must come from
  # the process's actual stdout fd. Reading it from ARGS produced "unknown" for every job — and the first
  # version then reported OK, a check that PASSES having determined NOTHING.
  LOG=$(readlink -f /proc/"$PID"/fd/1 2>/dev/null)
  case "$LOG" in *.log) ;; *) LOG=$(echo "$ARGS" | grep -oE '/tmp/[^ ]+\.log' | head -1) ;; esac
  DEV="⚠️ UNDETERMINED"
  # AUTHORITATIVE FIRST: the process's own SIM_BACKEND env var. Log parsing depends on a runner choosing
  # to print a device line, and the crux runner does not — it read UNDETERMINED for jobs whose backend was
  # verified by hand via /proc/<pid>/environ. Use the env; fall back to the log only when it is absent.
  ENVB=$(tr '\0' '\n' < /proc/"$PID"/environ 2>/dev/null | grep -oE '^SIM_BACKEND=.*' | cut -d= -f2)
  case "$ENVB" in
    cupy) DEV="GPU (env SIM_BACKEND=cupy)" ;;
    numpy) DEV="⛔ CPU (env SIM_BACKEND=numpy)"; BAD=$((BAD+1)) ;;
  esac
  if [ -n "$ENVB" ]; then
    LINES="$LINES  $(printf '%-42s' "${RUNNER:-?}") $(printf '%-8s' "$ETIME") $DEV\n"
    continue
  fi
  if [ -n "$LOG" ] && [ -f "$LOG" ]; then
    # process age in seconds (etime is [[dd-]hh:]mm:ss)
    PSTART=$(ps -o lstart= -p "$PID" 2>/dev/null)
    PAGE=$(( $(date +%s) - $(date -d "${PSTART:-now}" +%s 2>/dev/null || date +%s) ))
    LAGE=$(( $(date +%s) - $(stat -c%Y "$LOG" 2>/dev/null || date +%s) ))
    UNK=0
    if [ "$LAGE" -gt "$((PAGE + 60))" ]; then
      DEV="STALE-LOG(not this run)"
    elif grep -qa "SIM_BACKEND=numpy selected" "$LOG" 2>/dev/null; then
      DEV="⛔ CPU"; BAD=$((BAD+1))
    elif grep -qaE "GPU memory: [0-9.]+GB" "$LOG" 2>/dev/null; then
      DEV="GPU $(grep -oaE "GPU memory: [0-9.]+GB" "$LOG" | tail -1 | grep -oE '[0-9.]+GB')"
    fi
  fi
  LINES="$LINES  $(printf '%-42s' "${RUNNER:-?}") $(printf '%-8s' "$ETIME") $DEV\n"
done < <(ps -eo pid,etime,args | grep '[r]esearch\.runners' | awk '{pid=$1; et=$2; $1="";$2=""; print pid, et, $0}')

if [ "$QUIET" -eq 1 ]; then
  UND=$(printf "%b" "$LINES" | grep -c "UNDETERMINED" || true)
  if [ "$BAD" -gt 0 ]; then echo "device=⛔${BAD}-ON-CPU"; exit 1; fi
  if [ "$UND" -gt 0 ]; then echo "device=⚠️${UND}-UNDETERMINED(of $N)"; exit 1; fi
  echo "device=ok($N)"; exit 0
fi
echo "RUNNING JOBS AND THEIR DEVICE ($N)"
printf "%b" "$LINES"
UND=$(printf "%b" "$LINES" | grep -c "UNDETERMINED" || true)
if [ "$UND" -gt 0 ]; then
  echo
  echo "  ⚠️  $UND job(s) UNDETERMINED — the device could not be read. This is NOT a pass:"
  echo "     a check that reports OK having determined nothing is worse than no check."
fi
if [ "$BAD" -gt 0 ]; then
  echo
  echo "  ⛔ $BAD job(s) on CPU while a GPU is available."
  echo "     Cause is usually os.environ.setdefault('SIM_BACKEND','numpy') inside the runner, which"
  echo "     SILENTLY WINS unless the caller overrides it. Fix: prefix the command SIM_BACKEND=cupy."
  exit 1
fi
[ "$UND" -gt 0 ] && exit 1
echo "  OK — every job's device was READ and none is on the CPU."
