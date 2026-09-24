#!/usr/bin/env bash
# Install this repo's USER-LEVEL PreToolUse guards into ~/.claude, so they cover every project, session and subagent on
# the machine (a project-level hook only binds sessions whose checkout already carries it; agents in older worktrees
# would slip past). Idempotent: re-running refreshes each installed copy and never duplicates a settings entry. Ends by
# PROVING each installed guard blocks its incident and allows an ordinary command; exits non-zero if any check fails.
#
#   bash tools/install_user_hooks.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST_DIR="$HOME/.claude/hooks"
SETTINGS="$HOME/.claude/settings.json"
HOOKS=(guard_protected_delete.py require_memcap_for_brain_builds.py)

mkdir -p "$DST_DIR"
[ -f "$SETTINGS" ] || echo '{}' > "$SETTINGS"
for h in "${HOOKS[@]}"; do
  install -m 0755 "$REPO/.claude/hooks/$h" "$DST_DIR/$h"
  if jq -e --arg d "$DST_DIR/$h" '[.hooks.PreToolUse[]?.hooks[]?.command] | any(contains($d))' "$SETTINGS" >/dev/null; then
    echo "already registered: $h"
  else
    TMP="$(mktemp)"
    jq --arg cmd "python3 \"$DST_DIR/$h\"" \
       '.hooks.PreToolUse = ((.hooks.PreToolUse // []) + [{"matcher": "Bash", "hooks": [{"type": "command", "command": $cmd}]}])' \
       "$SETTINGS" > "$TMP"
    jq empty "$TMP"
    cp "$SETTINGS" "$SETTINGS.pre-user-hooks.bak"
    cat "$TMP" > "$SETTINGS"
    echo "registered: $h (previous settings: $SETTINGS.pre-user-hooks.bak)"
  fi
done

probe() {  # probe <hook> <command> -> prints the hook's exit code
  local rc=0
  jq -cn --arg c "$2" --arg cwd "$HOME" '{tool_name: "Bash", tool_input: {command: $c}, cwd: $cwd}' \
    | python3 "$DST_DIR/$1" 2>/dev/null || rc=$?
  echo "$rc"
}
fail=0
check() { [ "$(probe "$1" "$2")" = "$3" ] || { echo "FAIL: $1 returned the wrong verdict for: $2"; fail=1; }; }
check guard_protected_delete.py "rm -rf /tmp/scratch_clone $HOME/.claude/projects" 2
check guard_protected_delete.py "rm -rf /tmp/some-scratch-dir" 0
check require_memcap_for_brain_builds.py ".venv/bin/python -m pytest -q tests/test_webapp_server.py" 2
check require_memcap_for_brain_builds.py "bash tools/memcap.sh 12 -- .venv/bin/python -m pytest -q tests/test_webapp_server.py" 0
[ "$fail" = 0 ] && echo "OK: every installed guard blocks its incident and allows an ordinary command"
exit "$fail"
