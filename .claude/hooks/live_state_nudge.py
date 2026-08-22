#!/usr/bin/env python3
"""PostToolUse nudge: keep LIVE-STATE fresh on a landing (spec §5 "updated on every landing").

Fires on Write|Edit (wired alongside the existing check_doc_drift nudge). When the edited file is
one of the DURABLE state sources — the board, the production-integration ledger, or a findings doc —
this REGENERATES research/coordination/live_state.md from those sources (fast mode: board parse only,
no gpu/pool subprocess, so it is a few milliseconds and never a completion dump) and emits a one-line
systemMessage confirming it. This is the "enforced, not remembered" half: the file that gets
re-injected each turn tracks the durable state automatically, instead of relying on me to remember to
regenerate it.

PostToolUse CANNOT add to the model's context (confirmed against the hooks docs — only SessionStart /
UserPromptSubmit can), so the actual re-injection is done by live_state_inject.py on those events;
this hook only refreshes the FILE and shows the user a short note.

Fail-open: any error exits 0 silently.
"""
import json
import os
import re
import sys

# Editing one of these changes the frontier / next-actions / last-decision that LIVE-STATE carries.
DURABLE_SOURCE = re.compile(
    r"(?:^|/)GAP_CLOSURE_MISSION\.md$"
    r"|(?:^|/)docs/PRODUCTION_INTEGRATION_LEDGER\.yaml$"
    r"|(?:^|/)research/coordination/backlog\.json$"
    r"|(?:^|/)research/findings/[^/]+\.md$"
)


def _code_root() -> str:
    """Physical location of this hook (correct in a worktree). The data root is resolved by
    tools.live_state._root() (honours LIVE_STATE_ROOT); in production the two coincide."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return
    tool_input = data.get("tool_input") or {}
    tool_response = data.get("tool_response") or {}
    path = tool_input.get("file_path") or tool_response.get("filePath") or ""
    if not path:
        return
    normalized = path.replace("\\", "/")
    if not DURABLE_SOURCE.search(normalized):
        return

    root = _code_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import tools.live_state as ls
        os.environ["LIVE_STATE_FAST"] = "1"          # board parse only — no gpu/pool subprocess
        text = ls.write(fast=True)
        nb = len(text.encode("utf-8"))
        print(json.dumps({"systemMessage":
            "LIVE-STATE regenerated (%d B / cap %d B) from the durable sources after editing %s. "
            "It re-injects at turn-start + post-compaction; if a top next-action changed, that is now "
            "what the heartbeat and the next turn will carry." % (nb, ls.CAP_BYTES,
                                                                  os.path.basename(normalized))}))
    except Exception:
        return


if __name__ == "__main__":
    main()
