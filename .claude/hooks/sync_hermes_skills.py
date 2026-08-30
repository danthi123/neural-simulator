#!/usr/bin/env python3
"""PostToolUse hook: keep .hermes/skills/ in sync with .claude/skills/ automatically.

When Claude edits a project skill under .claude/skills/, Hermes must see the SAME
procedure. The two dirs cannot be symlinked (Hermes's project-skill scanner quarantines
a symlink that resolves outside the skill dir — see tools/hermes/sync_skills.sh), so they
are kept as byte copies. This hook runs that sync the instant a skill file changes, so the
two never drift (the hermes_parity_check remains the belt-and-suspenders backstop).

Fail-open + silent on non-skills edits: never blocks or noises up an unrelated edit.
"""
import json
import os
import subprocess
import sys


def _code_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return
    tool_input = data.get("tool_input") or {}
    tool_response = data.get("tool_response") or {}
    path = (tool_input.get("file_path") or tool_response.get("filePath") or "").replace("\\", "/")
    if "/.claude/skills/" not in path and not path.startswith(".claude/skills/"):
        return
    root = _code_root()
    script = os.path.join(root, "tools", "hermes", "sync_skills.sh")
    if not os.path.isfile(script):
        return
    try:
        out = subprocess.run(
            ["bash", script], cwd=root, capture_output=True, text=True, timeout=30
        )
    except Exception:
        return
    synced = [ln for ln in (out.stdout or "").splitlines() if "synced:" in ln]
    if synced:
        print(json.dumps({"systemMessage":
            "Hermes skills synced (.claude/skills -> .hermes/skills): %d file(s) updated so "
            "the Hermes agent sees the same procedure. Commit the .hermes/skills change alongside "
            "the .claude/skills edit." % len(synced)}))


if __name__ == "__main__":
    main()
