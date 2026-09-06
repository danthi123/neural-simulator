#!/usr/bin/env python3
"""agent_config.py — shared factory for the OpenHands-SDK prototype (2026-09-06).

PROTOTYPE / EVALUATION ONLY. This directory (`tools/openhands_proto/`) is an isolated sandbox for
evaluating OpenHands as a possible replacement for the Hermes harness (`tools/hermes/loop.py`) — it
is NOT wired into the live Hermes/gpu_queue infra and installs into its OWN venv
(`tools/openhands_proto/.venv/`, Python 3.12 via `uv`), never the repo's main `.venv`. Nothing here
runs unless invoked directly.

WHY OpenHands (owner's decisive requirement, 2026-09-06): ONE continuous, scrollable session (like
Claude Code), not Hermes v2's fresh-session-per-turn design. The `openhands-sdk` package
(`pip install openhands-sdk openhands-tools`, PyPI, actively maintained — NOT the deprecated
`OpenHands-CLI` / `openhands` CLI package, which the upstream repo now points at "Agent Canvas"
instead, see the write-up doc) gives this via `Conversation(persistence_dir=..., conversation_id=...)`
— resuming the SAME event history (compacted by `LLMSummarizingCondenser`) across process restarts,
which is exactly the offload boundary our GPU-sharing setup needs to cross.

This module builds the Agent/Conversation pointed at our local llama.cpp OpenAI-compatible endpoint
(`tools/qwen_serve.sh`, default `http://127.0.0.1:8033/v1`). Construction here makes NO network call
(verified empirically 2026-09-06 while qwen was offloaded) — safe to import/construct any time,
including while the model is down for a GPU experiment.
"""
from __future__ import annotations

import os
import subprocess
import uuid

os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")

# --- repo root discovery (works from the worktree during prototyping AND the canonical checkout
# during the live controller test — never hardcode a path) --------------------------------------
def _git_root(start):
    try:
        out = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=start,
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return start


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WORKSPACE = os.environ.get("OPENHANDS_WORKSPACE") or _git_root(HERE)

# --- local endpoint config -----------------------------------------------------------------------
# qwen_serve.sh defaults: HOSTADDR=127.0.0.1 PORT=8033 (see tools/qwen_serve.sh). The endpoint is
# only UP while Qwen is loaded — it offloads during GPU research jobs (tools/gpu_queue.sh). Callers
# of this module MUST health-check before constructing a Conversation that will actually call the
# LLM; construction itself is always safe (no network I/O), only conversation.run() hits the network.
QWEN_BASE = os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:8033")
LLM_BASE_URL = os.environ.get("OPENHANDS_LLM_BASE_URL", QWEN_BASE + "/v1")
# litellm's custom-OpenAI-endpoint convention is `openai/<model-id>` (docs.openhands.dev/openhands/
# usage/llms/local-llms, verified 2026-09-06). llama.cpp's `llama-server` accepts any string in the
# `model` field of a request when only one model is loaded (it does not exact-match), so this default
# works without inspecting `/v1/models` first — override via OPENHANDS_MODEL if a future backend
# (e.g. vLLM serving multiple models) requires an exact match.
LLM_MODEL = os.environ.get("OPENHANDS_MODEL", "openai/qwen-local")
LLM_API_KEY = os.environ.get("OPENHANDS_API_KEY", "local-no-key-required")

# --- one continuous session: a STABLE conversation_id so re-running this module (across process
# restarts, across the GPU-offload boundary) resumes the SAME persisted history rather than starting
# fresh. This is the whole point of the prototype (see module docstring). Override via
# OPENHANDS_CONVO_ID for a deliberate fresh session (e.g. a clean test run). -----------------------
_STABLE_NAMESPACE = uuid.UUID("6f9c9c1a-8b1e-4b9a-9b0a-3a6a2b6a9c00")  # arbitrary fixed namespace
DEFAULT_CONVERSATION_ID = uuid.uuid5(_STABLE_NAMESPACE, "neural-simulator-openhands-prototype")
CONVERSATION_ID = uuid.UUID(os.environ["OPENHANDS_CONVO_ID"]) if os.environ.get("OPENHANDS_CONVO_ID") \
    else DEFAULT_CONVERSATION_ID

PERSISTENCE_DIR = os.environ.get(
    "OPENHANDS_PERSIST_DIR", os.path.join(HERE, "state", "conversations")
)

# --- retry posture -------------------------------------------------------------------------------
# These are a SAFETY NET, not the primary offload-tolerance mechanism. The primary mechanism is
# architectural (see openhands_loop.py): the wrapper health-checks the endpoint and only calls
# conversation.run() while qwen is confirmed up, exactly mirroring tools/hermes/loop.py's
# qwen_up()/vram_handoff() pattern. A short local retry budget just absorbs a sub-minute race (e.g.
# the model finishing its reload a few seconds after the health check passed).
NUM_RETRIES = int(os.environ.get("OPENHANDS_NUM_RETRIES", "3"))
RETRY_MIN_WAIT = int(os.environ.get("OPENHANDS_RETRY_MIN_WAIT", "5"))
RETRY_MAX_WAIT = int(os.environ.get("OPENHANDS_RETRY_MAX_WAIT", "30"))

# --- condenser ------------------------------------------------------------------------------------
# max_size/keep_first are counted in EVENTS (message/action/observation), not tokens. Qwen3.8-27B
# here runs an 80-160k token context (tools/qwen_serve.sh --ctx-size, docs/2026-09-06-local-agent-
# stack-review.md) so these are generous defaults for a long dev session; tune down if turns are
# large (e.g. big diffs/log dumps) and the context still fills before condensation kicks in.
CONDENSER_MAX_SIZE = int(os.environ.get("OPENHANDS_CONDENSER_MAX_SIZE", "120"))
CONDENSER_KEEP_FIRST = int(os.environ.get("OPENHANDS_CONDENSER_KEEP_FIRST", "6"))


def qwen_up(timeout=4):
    """Health-check the local endpoint WITHOUT importing requests/httpx at module scope (keep this
    importable with zero network deps). Mirrors tools/hermes/loop.py's qwen_up()."""
    import urllib.request
    import urllib.error
    try:
        urllib.request.urlopen(QWEN_BASE + "/health", timeout=timeout)
        return True
    except Exception:
        return False


def build_llm(usage_id="qwen-local"):
    from openhands.sdk import LLM
    return LLM(
        usage_id=usage_id,
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        num_retries=NUM_RETRIES,
        retry_min_wait=RETRY_MIN_WAIT,
        retry_max_wait=RETRY_MAX_WAIT,
    )


def build_condenser(llm):
    from openhands.sdk.context.condenser import LLMSummarizingCondenser
    return LLMSummarizingCondenser(
        llm=llm.model_copy(update={"usage_id": "condenser"}),
        max_size=CONDENSER_MAX_SIZE,
        keep_first=CONDENSER_KEEP_FIRST,
    )


def build_agent(llm=None):
    from openhands.sdk import Agent, Tool
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.task_tracker import TaskTrackerTool
    from openhands.tools.terminal import TerminalTool
    from openhands.tools.grep import GrepTool
    from openhands.tools.glob import GlobTool

    llm = llm or build_llm()
    condenser = build_condenser(llm)
    return Agent(
        llm=llm,
        tools=[
            Tool(name=TerminalTool.name),
            Tool(name=FileEditorTool.name),
            Tool(name=TaskTrackerTool.name),
            Tool(name=GrepTool.name),
            Tool(name=GlobTool.name),
        ],
        condenser=condenser,
    )


def build_conversation(agent=None, workspace=None, callbacks=None):
    """Construct (or resume, if PERSISTENCE_DIR/CONVERSATION_ID already has state) the ONE
    continuous conversation. Safe to call while qwen is down — no network I/O happens here."""
    from openhands.sdk import Conversation

    agent = agent or build_agent()
    workspace = workspace or DEFAULT_WORKSPACE
    os.makedirs(PERSISTENCE_DIR, exist_ok=True)
    return Conversation(
        agent=agent,
        workspace=workspace,
        persistence_dir=PERSISTENCE_DIR,
        conversation_id=CONVERSATION_ID,
        callbacks=callbacks or [],
    )


# The turn prompt Hermes uses (tools/hermes/loop.py TURN_PROMPT) re-anchors from durable state each
# turn because Hermes' sessions are NOT continuous. Here the conversation IS continuous (that's the
# whole point), so this prompt is deliberately lighter: re-reading live_state.md is still correct
# (ground truth can change between turns — a GPU job may have completed), but there is no need to
# re-explain the whole mission every turn the way a fresh-session harness must.
TURN_PROMPT = (
    "Continue the neural-simulator research session. Re-read "
    "research/coordination/live_state.md and the CURRENT STATE atop GAP_CLOSURE_MISSION.md for "
    "anything that changed since your last turn (a GPU job may have completed). Take the next "
    "concrete action from the ordered NEXT ACTIONS: edit files, run tests, commit via "
    "tools/push_both.sh (NEVER --no-verify). If a GPU experiment is the next step, enqueue exactly "
    "ONE via `bash tools/gpu_queue.sh add '<cmd>'` and then stop this turn — an external supervisor "
    "unloads you and reloads you once the job completes (see openhands_loop.py). Obey CLAUDE.md "
    "(brain-based-only, one-brain, no-defer, 6-seed, gates authoritative). One concrete step per "
    "turn, then stop."
)


if __name__ == "__main__":
    print("workspace       :", DEFAULT_WORKSPACE)
    print("llm base_url    :", LLM_BASE_URL)
    print("llm model       :", LLM_MODEL)
    print("conversation_id :", CONVERSATION_ID)
    print("persistence_dir :", PERSISTENCE_DIR)
    print("qwen_up()       :", qwen_up())
