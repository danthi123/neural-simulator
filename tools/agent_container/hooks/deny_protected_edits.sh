#!/usr/bin/env bash
# tools/local_llm/hooks/deny_protected_edits.sh -- Claude Code PreToolUse hook for the local-model driver
# (docs/plans/2026-09-29-local-model-transition-PLAN.md section 4, "RUNNER-EDIT GUARD", plus a 2026-09-29
# loophole-review addition covering `Bash`). The actual matching logic lives in the co-located
# deny_protected_edits.py (see its module docstring for the three guards it implements); this wrapper only
# owns the hook's stdin/argv contract.
#
# CONTRACT (Claude Code PreToolUse hooks, see /docs/en/hooks and /docs/en/hooks-guide): the hook receives one
# JSON object on stdin -- {"tool_name": "...", "tool_input": {...}, ...} -- for every tool call. It signals a
# block with exit code 2 and a message on stderr (shown back to Claude as the reason); exit 0 allows the call.
# The payload is read into a shell variable first, then handed to python3 as an ARGUMENT (never as python3's
# own stdin), so there is no fd conflict between "read the hook payload" and "run the checker".
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$(cd "$HERE/../../.." && pwd)}"

INPUT=$(cat)

OUT=$(python3 "$HERE/deny_protected_edits.py" "$INPUT" "$PROJECT_ROOT")

case "$OUT" in
  ALLOW) exit 0 ;;
  BLOCK\|*)
    printf '\xe2\x9b\x94 BLOCKED: %s\n' "${OUT#BLOCK|}" >&2
    exit 2 ;;
  *) exit 0 ;;   # anything unrecognized from the python step -- fail open rather than block on our own bug
esac
