#!/usr/bin/env bash
# The heartbeat body. Folds workflow_check into every cycle so the three rules the owner kept enforcing
# by hand (parallelism, unserved lanes, primary-source reading) are CHECKED automatically and reported
# with literal commands — not left as prose I have to remember.
cd /home/dant123/Projects/sim
gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr '\n' '|')
procs=$(pgrep -fc "research.runners" 2>/dev/null | head -1); procs=${procs:-0}   # -c prints 0 AND exits 1
wc_out=$(bash tools/workflow_check.sh 2>&1); wc_rc=$?
# UNPUSHED COMMITS (2026-07-31). There is NO secondary local backup — the E: drive was wiped in the Linux
# migration, so origin+gitea are the ONLY copies and an unpushed commit lives on exactly one disk, on a box
# with a documented GPU crash that needs a hard reboot. Pushing was an arc-boundary habit rather than a
# policy; the owner caught two commits sitting unpushed for ~35 min. Reported every cycle so the drift
# cannot be silent. Uses ls-remote, not the remote-tracking ref: a cached ref will happily agree with a
# push that never happened.
ahead=$(git rev-list --count "$(git ls-remote origin refs/heads/main 2>/dev/null | cut -f1)"..HEAD 2>/dev/null || echo "?")
echo "⚓ HB $(date +%H:%M) gpu=[$gpu] procs=$procs unpushed=$ahead"
if [ "$ahead" != "0" ] && [ "$ahead" != "?" ]; then
  echo "⛔ $ahead COMMIT(S) UNPUSHED — the only copy is this disk. run: bash tools/push_both.sh"
fi
if [ "$wc_rc" -ne 0 ]; then
  echo "⛔ WORKFLOW RULES VIOLATED — copy-paste commands below (do NOT rationalise past this):"
  echo "$wc_out" | grep -E "⛔|run:" | head -8
else
  echo "✅ workflow_check: parallelism + lanes + sources + cluster all satisfied"
fi
