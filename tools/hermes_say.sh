#!/usr/bin/env bash
# hermes_say.sh — queue feedback for Hermes WITHOUT interrupting its current turn/run.
#
#   bash tools/hermes_say.sh "<feedback message>"
#
# WHY A FILE, NOT hermes -z: an interrupt-driven "-z" nudge competes with whatever Hermes is
# mid-doing (a GPU job's Qwen-unload wait, a long tool call) and either queues behind it or forces
# a context switch. The owner asked for a way to leave a note that surfaces on its own schedule --
# so this just appends a timestamped line to research/coordination/.hermes_feedback_queue.
# tools/hermes/hook_live_state_context.py (the pre_llm_call hook) drains this file and injects its
# contents into the NEXT turn's context automatically, exactly once (same drain-then-delete
# pattern as the .hermes_pending_advisory file tools/hermes/hook_post_edit.py writes). Works
# whether Hermes or Claude currently drives, and needs no GPU, no daemon, no hermes CLI call --
# it is one `printf >>`.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUEUE="$ROOT/research/coordination/.hermes_feedback_queue"
MSG="${1:-}"

if [ -z "$MSG" ]; then
  echo 'usage: bash tools/hermes_say.sh "<feedback message>"' >&2
  exit 2
fi

mkdir -p "$(dirname "$QUEUE")"
printf '[%s] %s\n' "$(date '+%F %T')" "$MSG" >> "$QUEUE"
echo "[hermes-say] queued -- will surface once in Hermes's next turn's context (via pre_llm_call): $MSG"
