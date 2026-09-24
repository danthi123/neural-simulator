#!/usr/bin/env bash
# Install the protected-delete guard (.claude/hooks/guard_protected_delete.py) at USER level, so it covers every
# project and every subagent on this machine, not just sessions opened in this repo. Idempotent: re-running
# refreshes the installed copy and never duplicates the settings entry. Ends by PROVING the installed copy blocks
# the 2026-09-24 incident and allows an ordinary delete; exits non-zero if either check fails.
#
#   bash tools/install_delete_guard.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO/.claude/hooks/guard_protected_delete.py"
DST_DIR="$HOME/.claude/hooks"
DST="$DST_DIR/guard_protected_delete.py"
SETTINGS="$HOME/.claude/settings.json"

mkdir -p "$DST_DIR"
install -m 0755 "$SRC" "$DST"
[ -f "$SETTINGS" ] || echo '{}' > "$SETTINGS"

if jq -e --arg d "$DST" '[.hooks.PreToolUse[]?.hooks[]?.command] | any(contains($d))' "$SETTINGS" >/dev/null; then
  echo "already registered in $SETTINGS"
else
  TMP="$(mktemp)"
  jq --arg cmd "python3 \"$DST\"" \
     '.hooks.PreToolUse = ((.hooks.PreToolUse // []) + [{"matcher": "Bash", "hooks": [{"type": "command", "command": $cmd}]}])' \
     "$SETTINGS" > "$TMP"
  jq empty "$TMP"
  cp "$SETTINGS" "$SETTINGS.pre-delete-guard.bak"
  cat "$TMP" > "$SETTINGS"
  echo "registered in $SETTINGS (previous version: $SETTINGS.pre-delete-guard.bak)"
fi

probe() {
  local rc=0
  jq -cn --arg c "$1" --arg cwd "$HOME" '{tool_name: "Bash", tool_input: {command: $c}, cwd: $cwd}' \
    | python3 "$DST" 2>/dev/null || rc=$?
  echo "$rc"
}
[ "$(probe "rm -rf /tmp/scratch_clone $HOME/.claude/projects")" = 2 ] || { echo "FAIL: the installed guard did not block the incident"; exit 1; }
[ "$(probe "rm -rf /tmp/some-scratch-dir")" = 0 ] || { echo "FAIL: the installed guard blocked an ordinary delete"; exit 1; }
echo "OK: $DST blocks the 2026-09-24 incident and allows an ordinary delete"
