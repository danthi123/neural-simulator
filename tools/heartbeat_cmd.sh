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
# BRANCH-AWARE (2026-08-06). Measuring "unpushed" against origin/main is WRONG when this worktree is checked
# out on any other branch: a branch whose commits are fully pushed to its OWN remote ref then reads as N
# unpushed forever (the archived codex/replay-consolidation-v3 read "3 unpushed" every cycle for a whole
# session while being safely on origin — a textbook corrosive false alarm). Compare HEAD to the CURRENT
# branch's own upstream ref; only fall back to main when the branch has no remote (genuinely local-only work).
cur_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
up_sha=$(git ls-remote origin "refs/heads/${cur_branch}" 2>/dev/null | cut -f1)
[ -z "$up_sha" ] && up_sha=$(git ls-remote origin refs/heads/main 2>/dev/null | cut -f1)
ahead=$(git rev-list --count "${up_sha}"..HEAD 2>/dev/null || echo "?")
# AWS state, QUERIED not remembered. The previous heartbeat carried a hardcoded "AWS BILLS while running"
# reminder that stayed on the screen after the instance was stopped — a false alarm, which the project's own
# rule says is as corrosive as a missed one because it trains the reader to skip the line.
aws_state=""
if [ -f research/queue/.aws_gpu ]; then
  inst=$(grep -m1 '^instance=' research/queue/.aws_gpu 2>/dev/null | cut -d= -f2)
  if [ -n "$inst" ]; then
    st=$(timeout 25 aws ec2 describe-instances --instance-ids "$inst" \
           --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null)
    case "$st" in
      running) aws_state=" ⛔ AWS $inst RUNNING — BILLING. stop: bash tools/aws_gpu.sh stop" ;;
      ""|None) aws_state="" ;;
      *)       aws_state=" aws=$st" ;;
    esac
  fi
fi
crux=$(ls research/findings/raw/gap4/resumable/*.json 2>/dev/null | wc -l)
echo "⚓ HB $(date +%H:%M) gpu=[$gpu] procs=$procs unpushed=$ahead crux-banked=$crux$aws_state"
if [ "$ahead" != "0" ] && [ "$ahead" != "?" ]; then
  echo "⛔ $ahead COMMIT(S) UNPUSHED — the only copy is this disk. run: bash tools/push_both.sh"
fi
if [ "$wc_rc" -ne 0 ]; then
  echo "⛔ WORKFLOW RULES VIOLATED — copy-paste commands below (do NOT rationalise past this):"
  echo "$wc_out" | grep -E "⛔|run:" | head -8
else
  echo "✅ workflow_check: parallelism + lanes + sources + cluster all satisfied"
fi
# PARALLEL AUDIT — call the LIVE file, never a snapshot (2026-08-26). The audit output used to come from a
# SEPARATE Monitor command that carried its own copy of the logic; when tools/parallel_audit.py was fixed to
# actually fire, the running Monitor kept printing the OLD verdict because it never re-read the file. Folding the
# call into THIS one canonical script means the heartbeat always executes whatever tools/parallel_audit.py
# currently is — a code fix is live on the next cycle with no re-arm. Selftest-guarded (a check unable to fire is
# the bug it exists to catch), so a broken audit is loud, not silently-passing.
.venv/bin/python tools/parallel_audit.py 2>/dev/null || echo "⚠ parallel_audit FAILED to run — check tools/parallel_audit.py"
