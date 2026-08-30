#!/usr/bin/env python3
"""Hermes `pre_llm_call` shell hook — the Hermes equivalent of Claude Code's
UserPromptSubmit half of .claude/hooks/live_state_inject.py.

WHY THIS EVENT (not on_session_start). Hermes's shell-hook wire protocol only lets a hook inject
context into the running turn via `pre_llm_call` -- the Hermes docs name this explicitly as the
Claude-Code `UserPromptSubmit` equivalent ("Claude Code's UserPromptSubmit event is intentionally
not a separate Hermes event -- pre_llm_call fires at the same place and already supports context
injection", website/docs/user-guide/features/hooks.md, worked example 3). `on_session_start`'s
return value is never consumed by the caller (agent/conversation_loop.py:1049-1061 fires it
fire-and-forget), so it CANNOT re-inject text -- only pre_llm_call can. This hook therefore does
three things Claude Code splits across separate mechanisms: (1) prints the current
research/coordination/live_state.md (regenerating it if missing, via `tools/live_state.py --emit`)
as ephemeral per-turn context, (2) drains any PENDING advisory left by hook_post_edit.py (the
post_tool_call sibling, whose own return value is an observer-only no-op in Hermes and so cannot
inject anything itself) so a doc-drift / sync-documentation nudge surfaces on the NEXT LLM call
instead of being silently dropped, and (3) drains any owner feedback queued via
`tools/hermes_say.sh` -- the AUTONOMOUS-MODE side channel that lets the owner leave Hermes a note
WITHOUT interrupting whatever it is mid-doing (a GPU-job wait, a long tool call): the note surfaces
here, once, on Hermes's own next turn.

Registration (~/.hermes/config.yaml, see hermes-parity/config.hooks.snippet.yaml):

    hooks:
      pre_llm_call:
        - command: "python3 /home/dant123/Projects/sim/tools/hermes/hook_live_state_context.py"
          timeout: 15

Fail-open: any error prints nothing (silent no-op), matching the Claude Code hook's fail-open
posture and Hermes's own default shell-hook failure semantics.
"""
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PENDING_ADVISORY = os.path.join(REPO, "research", "coordination", ".hermes_pending_advisory")
FEEDBACK_QUEUE = os.path.join(REPO, "research", "coordination", ".hermes_feedback_queue")


def _live_state_text() -> str:
    try:
        r = subprocess.run(
            [sys.executable, os.path.join(REPO, "tools", "live_state.py"), "--emit"],
            cwd=REPO, capture_output=True, text=True, timeout=10,
        )
        return r.stdout or ""
    except Exception:
        return ""


def _drain_pending_advisory() -> str:
    """Read-then-delete the advisory file hook_post_edit.py may have left. Surfacing it exactly
    once (on the next turn) mirrors Claude Code's PostToolUse systemMessage, which Hermes's
    observer-only post_tool_call cannot emit directly."""
    try:
        with open(PENDING_ADVISORY, "r", encoding="utf-8") as fh:
            text = fh.read().strip()
        os.remove(PENDING_ADVISORY)
        return text
    except FileNotFoundError:
        return ""
    except Exception:
        return ""


def _drain_feedback_queue() -> str:
    """Read-then-delete research/coordination/.hermes_feedback_queue -- owner notes appended by
    `tools/hermes_say.sh`. Same drain-once contract as the advisory file above: this is the
    AUTONOMOUS-MODE answer to "queue feedback without interrupting" -- the owner leaves a note any
    time (Hermes may be mid-GPU-job-wait, mid-tool-call, or simply not listening), and it surfaces
    exactly once, on whichever turn reads it next."""
    try:
        with open(FEEDBACK_QUEUE, "r", encoding="utf-8") as fh:
            text = fh.read().strip()
        os.remove(FEEDBACK_QUEUE)
        return text
    except FileNotFoundError:
        return ""
    except Exception:
        return ""


def main() -> None:
    try:
        sys.stdin.read()  # discard the pre_llm_call payload -- this hook is unconditional
    except Exception:
        pass

    # VRAM-supervisor heartbeat: mark that an LLM turn is active RIGHT NOW. The supervisor refuses
    # to unload Qwen while this is fresh, so it can never cut a Hermes turn mid-generation to start a
    # queued run (the overnight failure: the turn was killed and lost its work). Touched on every LLM
    # call; goes stale a few minutes after the turn's last call, at which point the supervisor may
    # unload for the pending run. Fail-open.
    try:
        import time as _t
        with open(os.path.join(REPO, "research", "queue", ".qwen_llm_active"), "w") as _f:
            _f.write(str(int(_t.time())))
    except Exception:
        pass

    parts = []
    live_state = _live_state_text()
    if live_state.strip():
        parts.append(live_state.strip())
    advisory = _drain_pending_advisory()
    if advisory:
        parts.append("PENDING ADVISORY (from a recent edit, surfaced once):\n" + advisory)

    feedback = _drain_feedback_queue()
    if feedback:
        parts.append(
            "OWNER FEEDBACK (queued via tools/hermes_say.sh, surfaced once -- read it, act on it "
            "if it calls for action, do not wait for anything else before doing so):\n" + feedback
        )

    if not parts:
        return  # silent no-op -- matches Hermes's "empty output is fine" contract
    print(json.dumps({"context": "\n\n".join(parts)}))


if __name__ == "__main__":
    main()
