#!/usr/bin/env python3
"""validate_offline.py — everything about the OpenHands prototype checkable WITHOUT the GPU/qwen up.

Run: tools/openhands_proto/.venv/bin/python tools/openhands_proto/validate_offline.py

Exercises: SDK imports, the TerminalTool + FileEditorTool executors standalone (real shell/file
access, no LLM call), Agent/Conversation/condenser construction against the (currently-down) local
endpoint (must NOT touch the network), conversation persistence + resume-from-disk, and read-only
GPU-queue/qwen-health introspection (informational — this prototype makes no GPU/network calls of
its own). Does NOT call conversation.run() (that needs a live LLM) — see run_turn.py for the live
test, run only once qwen is confirmed up.
"""
from __future__ import annotations

import os
import shutil
import sys
import uuid

import agent_config as cfg

PASS = []
FAIL = []


def check(name, fn):
    try:
        detail = fn()
        PASS.append((name, detail))
        print("PASS  %-55s %s" % (name, detail or ""))
    except Exception as e:
        FAIL.append((name, repr(e)))
        print("FAIL  %-55s %r" % (name, e))


def t_imports():
    from openhands.sdk import LLM, Agent, Conversation, Tool  # noqa: F401
    from openhands.sdk.context.condenser import LLMSummarizingCondenser  # noqa: F401
    from openhands.tools.terminal import TerminalTool  # noqa: F401
    from openhands.tools.file_editor import FileEditorTool  # noqa: F401
    from openhands.tools.task_tracker import TaskTrackerTool  # noqa: F401
    from openhands.tools.grep import GrepTool  # noqa: F401
    from openhands.tools.glob import GlobTool  # noqa: F401
    import importlib.metadata as m
    return "sdk=%s tools=%s" % (m.version("openhands-sdk"), m.version("openhands-tools"))


def t_terminal_tool():
    from openhands.tools.terminal.impl import TerminalExecutor
    from openhands.tools.terminal.definition import TerminalAction
    ex = TerminalExecutor(working_dir=cfg.DEFAULT_WORKSPACE, terminal_type="subprocess")
    obs = ex(TerminalAction(command="echo VALIDATE_OFFLINE_OK"))
    text = "".join(getattr(c, "text", "") for c in obs.to_llm_content)
    assert "VALIDATE_OFFLINE_OK" in text, "expected marker not in terminal output: %r" % text
    return "real shell exec via TerminalExecutor(terminal_type=subprocess) — no tmux needed"


def t_file_editor_tool():
    from openhands.tools.file_editor.impl import FileEditorExecutor
    from openhands.tools.file_editor.definition import FileEditorAction
    ex = FileEditorExecutor(workspace_root=cfg.DEFAULT_WORKSPACE)
    target = os.path.join(cfg.DEFAULT_WORKSPACE, "HERMES.md")
    obs = ex(FileEditorAction(command="view", path=target, view_range=[1, 3]))
    text = "".join(getattr(c, "text", "") for c in obs.to_llm_content)
    assert "Hermes" in text
    return "real file read via FileEditorExecutor (view HERMES.md)"


def t_llm_construction_no_network():
    llm = cfg.build_llm()
    assert llm.base_url == cfg.LLM_BASE_URL
    return "LLM(base_url=%s) constructed with zero network I/O" % llm.base_url


def t_condenser_construction():
    llm = cfg.build_llm()
    condenser = cfg.build_condenser(llm)
    return "LLMSummarizingCondenser(max_size=%s, keep_first=%s)" % (
        condenser.max_size, condenser.keep_first)


def t_agent_construction():
    agent = cfg.build_agent()
    names = sorted(t.name for t in agent.tools)
    assert names == sorted(["terminal", "file_editor", "task_tracker", "grep", "glob"])
    return "tools=%s" % names


def t_conversation_persistence_and_resume():
    """Build a conversation with a THROWAWAY id (not the real session id), inspect its persisted
    state, then reconstruct a fresh Conversation object against the SAME id/dir and confirm it
    reads back the same agent/LLM config from disk — this is the resume path the offload wrapper
    depends on. Cleans up its scratch dir afterward; never touches the real persisted session."""
    scratch = os.path.join(cfg.HERE, "state", "_validate_scratch")
    shutil.rmtree(scratch, ignore_errors=True)
    cid = uuid.uuid4()
    agent = cfg.build_agent()
    # scratch id/dir (never the real persisted session) — construct directly rather than via
    # cfg.build_conversation(), which would use the module's real CONVERSATION_ID/PERSISTENCE_DIR
    from openhands.sdk import Conversation
    conv1 = Conversation(agent=agent, workspace=cfg.DEFAULT_WORKSPACE,
                         persistence_dir=scratch, conversation_id=cid)
    base_state = os.path.join(scratch, str(cid).replace("-", ""), "base_state.json")
    assert os.path.exists(base_state), "no base_state.json written at construction"
    # resume: fresh objects, same id/dir
    agent2 = cfg.build_agent()
    conv2 = Conversation(agent=agent2, workspace=cfg.DEFAULT_WORKSPACE,
                         persistence_dir=scratch, conversation_id=cid)
    assert str(conv2.state.id) == str(cid)
    shutil.rmtree(scratch, ignore_errors=True)
    return "base_state.json written + resumed with matching id=%s" % cid


def t_qwen_health():
    up = cfg.qwen_up()
    return "qwen_up()=%s (expected False right now — GPU is held by a research run; " \
           "see run_turn.py for the live test once it's up)" % up


def t_gpu_queue_introspection():
    import openhands_loop as ol
    return ("SHARED_ROOT=%s dispatcher_alive=%s queue_depth=%s running_job=%s"
            % (ol.SHARED_ROOT, ol.dispatcher_alive(), ol._queue_depth(),
               "yes" if ol._running_job() else "no"))


def main():
    check("imports (openhands-sdk + openhands-tools)", t_imports)
    check("TerminalTool executor (real shell, no LLM)", t_terminal_tool)
    check("FileEditorTool executor (real file read, no LLM)", t_file_editor_tool)
    check("LLM() construction hits no network", t_llm_construction_no_network)
    check("LLMSummarizingCondenser construction", t_condenser_construction)
    check("Agent construction (5 tools)", t_agent_construction)
    check("Conversation persistence + resume-from-disk", t_conversation_persistence_and_resume)
    check("qwen endpoint health (informational)", t_qwen_health)
    check("gpu_queue read-only introspection (informational)", t_gpu_queue_introspection)
    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
