"""Tests for tools/local_llm/hooks/deny_protected_edits.sh, a Claude Code PreToolUse hook for the local-model
driver (docs/plans/2026-09-29-local-model-transition-PLAN.md section 4, "RUNNER-EDIT GUARD").

Drives the hook exactly as Claude Code would: one JSON payload on stdin per call
({"tool_name": ..., "tool_input": {"file_path": ...}}), asserting the exit-code contract (0 = allow, 2 = block
with a reason on stderr) rather than importing any Python from the script (it is a standalone shell script).

Every path-matching test sets CLAUDE_PROJECT_DIR to a FAKE root ("/fake/project") rather than relying on the
real repo location, both to stay hermetic and because the hook does pure string prefix-matching (never touches
the filesystem), so a nonexistent root is fine. This also exercises the exact mechanism Claude Code itself uses
in production (it sets CLAUDE_PROJECT_DIR for every hook invocation) rather than only the BASH_SOURCE-derived
fallback, which gets its own dedicated test below.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / "tools" / "local_llm" / "hooks" / "deny_protected_edits.sh"
FAKE_ROOT = "/fake/project"


def _run(tool_name, file_path=None, extra_tool_input=None, project_dir=FAKE_ROOT):
    payload = {"hook_event_name": "PreToolUse", "tool_name": tool_name, "tool_input": {}}
    if file_path is not None:
        payload["tool_input"]["file_path"] = file_path
    if extra_tool_input:
        payload["tool_input"].update(extra_tool_input)
    env = dict(os.environ)
    if project_dir is not None:
        env["CLAUDE_PROJECT_DIR"] = project_dir
    else:
        env.pop("CLAUDE_PROJECT_DIR", None)
    return subprocess.run(["bash", str(HOOK)], input=json.dumps(payload), capture_output=True,
                           text=True, timeout=15, env=env)


def _run_raw(raw_stdin, project_dir=FAKE_ROOT):
    env = dict(os.environ)
    if project_dir is not None:
        env["CLAUDE_PROJECT_DIR"] = project_dir
    return subprocess.run(["bash", str(HOOK)], input=raw_stdin, capture_output=True, text=True,
                           timeout=15, env=env)


def test_hook_script_exists_and_is_executable():
    assert HOOK.is_file()
    assert os.access(HOOK, os.X_OK), "the hook must be executable (Claude Code invokes it directly)"


# --------------------------------------------------------------------- research/runners/*.py (non-draft) blocked

def test_blocks_edit_on_an_existing_runner():
    res = _run("Edit", f"{FAKE_ROOT}/research/runners/_tone_bg_nchannel_selector.py")
    assert res.returncode == 2
    assert "BLOCKED" in res.stderr
    assert "_draft_" in res.stderr


def test_blocks_write_on_an_existing_runner():
    res = _run("Write", f"{FAKE_ROOT}/research/runners/some_runner.py")
    assert res.returncode == 2
    assert "BLOCKED" in res.stderr


def test_blocks_multiedit_on_an_existing_runner():
    res = _run("MultiEdit", f"{FAKE_ROOT}/research/runners/load_bearing_fraction.py",
                extra_tool_input={"edits": [{"old_string": "a", "new_string": "b"}]})
    assert res.returncode == 2
    assert "BLOCKED" in res.stderr


def test_blocks_notebookedit_on_an_existing_runner_path():
    # NotebookEdit's file_path is still the key field carrying the target path.
    res = _run("NotebookEdit", f"{FAKE_ROOT}/research/runners/some_runner.py",
                extra_tool_input={"cell_id": "1", "source": "x"})
    assert res.returncode == 2
    assert "BLOCKED" in res.stderr


# ------------------------------------------------------------------------------ _draft_*.py runners are allowed

def test_allows_edit_on_a_draft_runner():
    res = _run("Edit", f"{FAKE_ROOT}/research/runners/_draft_new_thing.py")
    assert res.returncode == 0, res.stderr
    assert res.stderr == ""


def test_allows_write_on_a_new_draft_runner_inside_a_worktree():
    res = _run("Write", f"{FAKE_ROOT}/.claude/worktrees/draft-x/research/runners/_draft_x.py")
    assert res.returncode == 0, res.stderr


def test_blocks_edit_on_an_existing_runner_inside_a_worktree():
    # The worktree-unwrap logic must still find and block a non-draft runner path.
    res = _run("Edit", f"{FAKE_ROOT}/.claude/worktrees/draft-x/research/runners/existing_runner.py")
    assert res.returncode == 2
    assert "BLOCKED" in res.stderr


# ---------------------------------------------------------------------------- tools/gates/**, sim/**, webapp/**

def test_blocks_edit_on_tools_gates():
    res = _run("Edit", f"{FAKE_ROOT}/tools/gates/single_seed.py")
    assert res.returncode == 2
    assert "tools/gates/**" in res.stderr


def test_blocks_edit_on_sim_nested_path():
    res = _run("Edit", f"{FAKE_ROOT}/sim/core/engine.py")
    assert res.returncode == 2
    assert "sim/**" in res.stderr


def test_blocks_write_on_webapp_nested_path():
    res = _run("Write", f"{FAKE_ROOT}/webapp/server.py")
    assert res.returncode == 2
    assert "webapp/**" in res.stderr


def test_blocks_edit_on_sim_inside_a_worktree():
    res = _run("Edit", f"{FAKE_ROOT}/.claude/worktrees/draft-x/sim/core/engine.py")
    assert res.returncode == 2
    assert "sim/**" in res.stderr


# -------------------------------------------------------------------------------------------------- safe paths

def test_allows_edit_on_an_unrelated_doc():
    res = _run("Edit", f"{FAKE_ROOT}/docs/plans/2026-09-29-local-model-transition-PLAN.md")
    assert res.returncode == 0, res.stderr


def test_allows_edit_on_a_nested_runners_subdirectory_file():
    # Literal "research/runners/*.py" (single star) is a NON-recursive glob: a file one level deeper than
    # research/runners/ itself is out of scope for this guard (there are no such subdirectories in the real
    # tree today; documented here so the boundary is a deliberate choice, not an accident).
    res = _run("Edit", f"{FAKE_ROOT}/research/runners/subdir/nested.py")
    assert res.returncode == 0, res.stderr


def test_allows_edit_on_a_lookalike_path_that_only_contains_the_word_sim():
    # The project's OWN directory is itself named "sim" -- this must never trip the sim/** rule for paths that
    # merely happen to contain "sim" as a substring, only a real repo-relative top-level "sim/" component.
    res = _run("Edit", f"{FAKE_ROOT}/simulacrum/docs/notes.py")
    assert res.returncode == 0, res.stderr


def test_allows_edit_on_a_path_outside_the_project_root():
    res = _run("Edit", "/tmp/scratch/sim/webapp/whatever.py")
    assert res.returncode == 0, res.stderr


def test_allows_edit_on_a_lookalike_runners_path():
    res = _run("Edit", f"{FAKE_ROOT}/other_research/runners_extra/foo.py")
    assert res.returncode == 0, res.stderr


def test_ignores_non_edit_tools():
    res = _run("Read", f"{FAKE_ROOT}/research/runners/some_runner.py")
    assert res.returncode == 0, res.stderr
    res2 = _run("Bash", None, extra_tool_input={"command": "cat research/runners/some_runner.py"})
    assert res2.returncode == 0, res2.stderr


# ------------------------------------------------------------------------------------------------ fails open

def test_fails_open_on_malformed_json():
    res = _run_raw("not json at all {{{")
    assert res.returncode == 0
    assert res.stderr == ""


def test_fails_open_on_missing_file_path():
    res = _run("Edit", file_path=None)
    assert res.returncode == 0, res.stderr


def test_fails_open_on_empty_stdin():
    res = _run_raw("")
    assert res.returncode == 0


# ---------------------------------------------------------------------- CLAUDE_PROJECT_DIR fallback (unset)

def test_falls_back_to_the_scripts_own_location_when_claude_project_dir_is_unset():
    # With no CLAUDE_PROJECT_DIR, the hook derives its root from its own path (three levels up from
    # tools/local_llm/hooks/), which for THIS checkout is ROOT itself -- so a real repo-relative path under
    # this checkout must still be caught.
    res = _run("Edit", str(ROOT / "sim" / "core" / "engine.py"), project_dir=None)
    assert res.returncode == 2
    assert "sim/**" in res.stderr

    res_allow = _run("Edit", str(ROOT / "docs" / "plans" / "whatever.md"), project_dir=None)
    assert res_allow.returncode == 0, res_allow.stderr
