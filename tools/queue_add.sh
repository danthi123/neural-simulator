#!/usr/bin/env bash
# queue_add.sh — the ONLY sanctioned way to add a job to a compute queue. Checks the record FIRST.
#
# WHY (2026-07-29). "Read the record before spending compute" was the single most expensive lapse of the
# day and the LAST one still living in memory rather than in a mechanism. `tools/before_you_build.sh`
# already existed and was skipped — because running it was a thing to REMEMBER, and urgency (an owner
# critique about prioritization) made speed feel appropriate. The result: crux-lane GPU slots spent
# re-running a rate reference that had been banked five days earlier, while the actual open work (the
# on-bridge SPIKING port) sat build-ahead ready the whole time.
#
# A rule you must remember is not a mechanism. So the check now lives ON THE PATH the work must travel:
# jobs enter the queue through here, and `lane_dispatch.sh` REFUSES to run a line that did not.
#
#   tools/queue_add.sh gpu ".venv/bin/python -m research.runners.foo --seeds 42 > /tmp/x.log 2>&1"
#
# It prints every finding that already mentions the runner, then requires a one-word reason recorded
# inline. If the runner is already banked and you proceed anyway, the reason is in the queue file forever.
set -uo pipefail
ROOT=/home/dant123/Projects/sim
LANE="${1:?usage: queue_add.sh <gpu|pool> \"<command>\" [reason]}"
CMD="${2:?need the command}"
REASON="${3:-}"
Q="$ROOT/research/queue/${LANE}.queue"
mkdir -p "$(dirname "$Q")"; touch "$Q"

# INTERPRETER GUARD (2026-08-01, see pool_queue.sh + dispatcher_selftest.sh). Every research runner must go
# through .venv/bin/python: bare `python` is absent on the pool nodes and is not the sanctioned local
# interpreter either. A bare-python job validates fine (the checks shell out to .venv/bin/python) and then
# produces NOTHING -- the silent no-output failure measured on the affect + brain-quench sweeps.
case "$CMD" in *"-m research"*)
  case "$CMD" in *".venv/bin/python"*) ;;
    *) echo "⛔ REFUSED: runs a research module without .venv/bin/python -- bare 'python' produces silent no-output." >&2
       echo "   Use: SIM_BACKEND=<numpy|cupy> .venv/bin/python -u -m research.runners.X ..." >&2; exit 2 ;;
  esac ;;
esac

RUNNER=$(echo "$CMD" | grep -oE '[-_a-zA-Z0-9]+' | grep -E '^_?[a-z0-9]+([_a-z0-9]+)+$' | grep -vE '^(venv|bin|python|research|runners|tmp|claude|log|seeds|out|full|smoke)$' | head -1)
echo "── queue_add: checking the record for runner '${RUNNER:-?}' ──"

HITS=""
if [ -n "$RUNNER" ]; then
  # RECURSIVE (2026-07-31): the flat glob missed 42 findings one directory down in findings/raw/, so a runner
  # whose only prior record lived in a nested scoping doc looked like it had never been run.
  HITS=$(grep -rl --include='*.md' -- "$RUNNER" "$ROOT/research/findings/" 2>/dev/null | head -6)
fi

if [ -n "$HITS" ]; then
  echo "  ⚠️  ALREADY IN THE RECORD — read these BEFORE spending compute:"
  echo "$HITS" | sed 's|.*/|      |'
  echo "      (a GO/banked verdict here means the job may be a RE-RUN; that is what cost crux slots today)"
  if [ -z "$REASON" ]; then
    echo "  ⛔ NOT QUEUED. Re-run with a third argument giving the reason, e.g.:"
    echo "     tools/queue_add.sh $LANE \"<cmd>\" 'new-seeds' | 'new-config' | 'on-bridge-not-rate' | 'reproducing-deliberately'"
    exit 2
  fi
  echo "  ✔ proceeding with reason: $REASON"
else
  echo "  ✔ no prior findings mention this runner — genuinely new work"
  REASON="${REASON:-new}"
fi

printf '%s  #checked:%s\n' "$CMD" "$REASON" >> "$Q"
echo "  queued to $Q (marked #checked:$REASON)"
