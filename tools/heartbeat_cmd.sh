#!/usr/bin/env bash
# The heartbeat body. Folds workflow_check into every cycle so the three rules the owner kept enforcing
# by hand (parallelism, unserved lanes, primary-source reading) are CHECKED automatically and reported
# with literal commands — not left as prose I have to remember.
cd /home/dant123/Projects/sim
gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr '\n' '|')
procs=$(pgrep -fc "research.runners" 2>/dev/null || echo 0)
wc_out=$(bash tools/workflow_check.sh 2>&1); wc_rc=$?
echo "⚓ HB $(date +%H:%M) gpu=[$gpu] procs=$procs"
if [ "$wc_rc" -ne 0 ]; then
  echo "⛔ WORKFLOW RULES VIOLATED — copy-paste commands below (do NOT rationalise past this):"
  echo "$wc_out" | grep -E "⛔|run:" | head -8
else
  echo "✅ workflow_check: parallelism + lane coverage + primary-source reading all satisfied"
fi
