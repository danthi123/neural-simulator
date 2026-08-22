#!/usr/bin/env python3
"""LIVE-STATE re-injection hook — the anti-compaction-loss + anti-dilution mechanism (spec §5).

Wired in .claude/settings.json for TWO events (the only two that actually add text to context — a
claude-code-guide check of the official hooks docs confirmed PreCompact does NOT inject post-compaction
and PostToolUse cannot inject at all):

  - SessionStart  (all sources incl. `compact`): fires at session start, resume, AND immediately
    AFTER A COMPACTION (source="compact"). This is THE post-compaction re-injection point — the
    documented way to restore state that must survive compaction. Here it REGENERATES the LIVE-STATE
    file from the DURABLE sources (the board + backlog.json + live-run status, NEVER the lossy
    summary) and prints it to stdout, which Claude Code adds to the fresh context.

  - UserPromptSubmit (every user turn): prints the LIVE-STATE file to stdout so the frontier, the
    ordered next-actions and the hard constraints re-appear at the END of context every turn, where
    attention is strongest — countering dilution as the window fills. Pure read: no regeneration, no
    subprocess, so it never adds latency to a turn (durable state only changes at session start and
    on landings, which is where the file is regenerated).

FAIL-OPEN, ALWAYS. A hook that hangs or crashes the session/turn is far worse than one that no-ops,
so every path is wrapped and exits 0 with (at worst) nothing. Output goes to stdout as plain text,
which both events treat as a context addition on exit 0.
"""
import json
import os
import sys


def _code_root() -> str:
    """Where the code lives, for the import path — the physical location of this hook file, so it is
    correct in a git worktree too. The DATA root (which board/live_state.md to read/write) is left to
    tools.live_state._root(), which honours LIVE_STATE_ROOT; in production the two coincide."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        data = {}

    event = data.get("hook_event_name") or data.get("hookEventName") or ""

    root = _code_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import tools.live_state as ls
    except Exception:
        return  # generator unavailable: emit nothing rather than break the session

    try:
        if event == "SessionStart":
            # session start / resume / clear / compact — regenerate from the durable sources and emit.
            # source="compact" lands here right after a compaction: this is the re-injection that
            # survives it (re-read from the FILE, not from the summary the compactor produced).
            text = ls.write()
        else:
            # UserPromptSubmit (and any other wiring): read the file, regenerating only if missing.
            text = ls.read()
    except Exception:
        try:
            text = ls.read()
        except Exception:
            return

    if text and text.strip():
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
