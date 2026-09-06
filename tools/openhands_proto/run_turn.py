#!/usr/bin/env python3
"""run_turn.py — drive ONE turn of the persisted OpenHands conversation against the local endpoint.

PROTOTYPE. Requires the local llama.cpp endpoint (tools/qwen_serve.sh) to be UP — refuses to start
otherwise (fail fast, don't let litellm's own retry/backoff silently eat minutes). This is the
script the CONTROLLER should run for the live end-to-end test once the GPU frees + qwen is up (see
the write-up doc for the exact invocation). It is also what `openhands_loop.py` calls internally for
each turn once the offload wrapper confirms the endpoint is healthy.

Usage:
    tools/openhands_proto/.venv/bin/python tools/openhands_proto/run_turn.py \
        [--prompt "custom instruction"] [--workspace /path/to/repo] [--max-iterations N]

With no --prompt, sends agent_config.TURN_PROMPT (the "continue the session" prompt).
"""
from __future__ import annotations

import argparse
import sys

import agent_config as cfg


def _print_event(event):
    # Minimal visible transcript so a human watching stdout can follow the ONE continuous session —
    # this is the "scrollable" part of the owner's requirement made concrete.
    kind = type(event).__name__
    text = getattr(event, "to_llm_content", None)
    if callable(text):
        try:
            text = "".join(getattr(c, "text", str(c)) for c in event.to_llm_content())
        except Exception:
            text = None
    snippet = (text or str(event))[:500]
    print("\n[%s]\n%s" % (kind, snippet), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--max-iterations", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="skip the qwen_up() health check (debugging only — normally refuse to run)")
    args = ap.parse_args()

    if not args.force and not cfg.qwen_up():
        print("[run_turn] REFUSING: local endpoint %s is not reachable. Start it with "
              "`bash tools/qwen_serve.sh up` (or wait for the offload supervisor) before running a "
              "turn. Use --force to override (not recommended)." % cfg.QWEN_BASE, file=sys.stderr)
        return 1

    conversation = cfg.build_conversation(callbacks=[_print_event])
    if args.max_iterations:
        conversation.state.max_iterations = args.max_iterations

    prompt = args.prompt or cfg.TURN_PROMPT
    print("[run_turn] conversation_id=%s persistence_dir=%s workspace=%s" % (
        cfg.CONVERSATION_ID, cfg.PERSISTENCE_DIR, args.workspace or cfg.DEFAULT_WORKSPACE))
    print("[run_turn] sending: %s" % prompt[:200])
    conversation.send_message(prompt)
    conversation.run()
    print("\n[run_turn] turn complete. execution_status=%s" %
          getattr(conversation.state, "execution_status", "?"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
