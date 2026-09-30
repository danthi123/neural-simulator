"""Tests for the Bash-command guards in tools/local_llm/hooks/deny_protected_edits.py (2026-09-29, loophole
review of claude_local_settings.json's `Bash(git -C .claude/worktrees/draft-:*)` allow rule).

That prefix rule permits ANY git subcommand run as `git -C <draft-worktree-path> <anything>`, not just the
intended status/add/commit -- including gate/hook bypasses (--no-verify, core.hooksPath), history rewrites
(reset --hard, rebase, cherry-pick, commit-tree, hash-object -w, update-ref), and forced pushes. A second guard
in the same module blocks Bash commands that write to known gate-read state files (research/coordination/
*.jsonl, research/queue/pool.queue, research/queue/gpu.queue, research/queue/.waiver_history.jsonl,
tools/gates/**) through redirection/tee/sed-i/cp/mv/python-open, while trusting the sanctioned wrapper CLIs.

Drives the hook script exactly as Claude Code would (JSON on stdin, exit-code contract), not the Python module
directly, so a change to the stdin/argv plumbing in the .sh wrapper is exercised too.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / "tools" / "local_llm" / "hooks" / "deny_protected_edits.sh"
FAKE_ROOT = "/fake/project"


def _run_bash(command, project_dir=FAKE_ROOT):
    payload = {"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": command}}
    env = dict(os.environ)
    if project_dir is not None:
        env["CLAUDE_PROJECT_DIR"] = project_dir
    return subprocess.run(["bash", str(HOOK)], input=json.dumps(payload), capture_output=True,
                           text=True, timeout=15, env=env)


# ------------------------------------------------------------------------------------- git-escape guard: blocked

def test_blocks_bare_no_verify():
    res = _run_bash("git commit --no-verify -m 'x'")
    assert res.returncode == 2
    assert "BLOCKED" in res.stderr
    assert "--no-verify" in res.stderr


def test_blocks_git_dash_c_worktree_no_verify():
    res = _run_bash("git -C .claude/worktrees/draft-x commit --no-verify -m 'x'")
    assert res.returncode == 2
    assert "BLOCKED" in res.stderr


def test_blocks_git_commit_short_n_flag():
    res = _run_bash("git -C .claude/worktrees/draft-x commit -n -m 'x'")
    assert res.returncode == 2
    assert "-n" in res.stderr


def test_blocks_git_commit_combined_short_flag_containing_n():
    res = _run_bash("git -C .claude/worktrees/draft-x commit -an -m 'x'")
    assert res.returncode == 2


def test_blocks_core_hookspath():
    res = _run_bash("git -c core.hooksPath=/tmp/evil -C .claude/worktrees/draft-x commit -m x")
    assert res.returncode == 2
    assert "core.hooksPath" in res.stderr


def test_blocks_git_replace():
    res = _run_bash("git -C .claude/worktrees/draft-x replace HEAD abc123")
    assert res.returncode == 2


def test_blocks_update_ref():
    res = _run_bash("git -C .claude/worktrees/draft-x update-ref refs/heads/main abc123")
    assert res.returncode == 2


def test_blocks_commit_tree():
    res = _run_bash("git -C .claude/worktrees/draft-x commit-tree abc123 -m x")
    assert res.returncode == 2


def test_blocks_hash_object_dash_w():
    res = _run_bash("git -C .claude/worktrees/draft-x hash-object -w evil.py")
    assert res.returncode == 2


def test_allows_hash_object_without_write():
    # -w is what makes hash-object dangerous (writes a loose object); without it, it's a read-only hash.
    res = _run_bash("git -C .claude/worktrees/draft-x hash-object some.py")
    assert res.returncode == 0, res.stderr


def test_blocks_cherry_pick():
    res = _run_bash("git -C .claude/worktrees/draft-x cherry-pick abc123")
    assert res.returncode == 2


def test_blocks_git_stash():
    res = _run_bash("git -C .claude/worktrees/draft-x stash")
    assert res.returncode == 2


def test_blocks_git_merge():
    res = _run_bash("git -C .claude/worktrees/draft-x merge main")
    assert res.returncode == 2


def test_blocks_git_rebase():
    res = _run_bash("git -C .claude/worktrees/draft-x rebase main")
    assert res.returncode == 2


def test_blocks_push_force():
    res = _run_bash("git -C .claude/worktrees/draft-x push --force origin research/draft-x")
    assert res.returncode == 2


def test_blocks_push_dash_f():
    res = _run_bash("git -C .claude/worktrees/draft-x push -f origin research/draft-x")
    assert res.returncode == 2


def test_blocks_push_force_with_lease_standalone():
    res = _run_bash("git push --force-with-lease origin research/draft-x")
    assert res.returncode == 2


def test_blocks_reset_hard():
    res = _run_bash("git -C .claude/worktrees/draft-x reset --hard origin/main")
    assert res.returncode == 2


def test_blocks_git_no_verify_env_var():
    res = _run_bash("GIT_NO_VERIFY=1 git -C .claude/worktrees/draft-x commit -m x")
    assert res.returncode == 2


def test_blocks_skip_env_var():
    res = _run_bash("SKIP=gates/coverage git -C .claude/worktrees/draft-x commit -m x")
    assert res.returncode == 2


def test_blocks_no_verify_inside_a_bash_dash_c_wrapper():
    res = _run_bash('bash -c "git -C .claude/worktrees/draft-x commit --no-verify -m x"')
    assert res.returncode == 2
    assert "--no-verify" in res.stderr


def test_blocks_reset_hard_after_a_benign_chained_command():
    res = _run_bash("git -C .claude/worktrees/draft-x status && git -C .claude/worktrees/draft-x reset --hard HEAD~1")
    assert res.returncode == 2


# ------------------------------------------------------------------------------------- git-escape guard: allowed

def test_allows_ordinary_status_add_commit_in_a_draft_worktree():
    res = _run_bash("git -C .claude/worktrees/draft-x status")
    assert res.returncode == 0, res.stderr
    res2 = _run_bash("git -C .claude/worktrees/draft-x add research/runners/_draft_x.py")
    assert res2.returncode == 0, res2.stderr
    res3 = _run_bash('git -C .claude/worktrees/draft-x commit -m "add draft runner"')
    assert res3.returncode == 0, res3.stderr


def test_does_not_false_positive_on_grep_merge():
    # The concrete case this guard must not break: --grep=merge must never look like `git merge`.
    res = _run_bash("git -C .claude/worktrees/draft-x log --grep=merge --oneline")
    assert res.returncode == 0, res.stderr


def test_does_not_false_positive_on_reset_without_hard():
    res = _run_bash("git -C .claude/worktrees/draft-x reset HEAD~1")
    assert res.returncode == 0, res.stderr


def test_does_not_false_positive_on_push_without_force():
    res = _run_bash("git -C .claude/worktrees/draft-x push origin research/draft-x")
    assert res.returncode == 0, res.stderr


def test_does_not_false_positive_on_unrelated_command_with_dash_n():
    res = _run_bash("grep -n TODO research/runners/_draft_x.py")
    assert res.returncode == 0, res.stderr


# ------------------------------------------------------------------------------------- gate-state write guard

def test_blocks_redirect_into_pool_queue():
    res = _run_bash("echo bogus >> research/queue/pool.queue")
    assert res.returncode == 2
    assert "pool.queue" in res.stderr


def test_blocks_redirect_into_gpu_queue_no_space():
    res = _run_bash("echo bogus>research/queue/gpu.queue")
    assert res.returncode == 2


def test_blocks_redirect_into_coordination_jsonl():
    res = _run_bash("echo '{}' >> research/coordination/lane_builds.jsonl")
    assert res.returncode == 2
    assert "lane_builds.jsonl" in res.stderr


def test_blocks_redirect_into_waiver_history():
    res = _run_bash("echo '{}' >> research/queue/.waiver_history.jsonl")
    assert res.returncode == 2


def test_blocks_redirect_into_tools_gates():
    res = _run_bash("echo 'return True' >> tools/gates/single_seed.py")
    assert res.returncode == 2


def test_blocks_tee_into_pool_queue():
    res = _run_bash("echo bogus | tee -a research/queue/pool.queue")
    assert res.returncode == 2


def test_blocks_sed_inplace_on_gates():
    res = _run_bash("sed -i 's/True/False/' tools/gates/single_seed.py")
    assert res.returncode == 2


def test_blocks_cp_onto_pool_queue():
    res = _run_bash("cp /tmp/fake.queue research/queue/pool.queue")
    assert res.returncode == 2


def test_blocks_mv_onto_gpu_queue():
    res = _run_bash("mv /tmp/fake.queue research/queue/gpu.queue")
    assert res.returncode == 2


def test_blocks_python_open_write_on_coordination_jsonl():
    res = _run_bash(
        """python3 -c "open('research/coordination/lane_builds.jsonl','a').write('{}')" """
    )
    assert res.returncode == 2


def test_allows_sanctioned_pool_queue_add():
    res = _run_bash(
        "bash tools/pool_queue.sh add 'true --seed 7' --checked 'dev smoke'"
    )
    assert res.returncode == 0, res.stderr


def test_still_blocks_when_sanctioned_call_is_paired_with_a_raw_redirect():
    res = _run_bash(
        "bash tools/pool_queue.sh add 'true --seed 7' --checked 'x' > research/queue/pool.queue"
    )
    assert res.returncode == 2


def test_allows_sanctioned_waiver_history_clear():
    res = _run_bash("python3 tools/waiver_history.py clear")
    assert res.returncode == 0, res.stderr


def test_allows_redirect_to_an_unrelated_log_file():
    res = _run_bash("bash tools/status.sh >> /tmp/scratch/status.log")
    assert res.returncode == 0, res.stderr


def test_allows_cp_of_unrelated_files():
    res = _run_bash("cp research/runners/_draft_x.py /tmp/backup.py")
    assert res.returncode == 0, res.stderr
