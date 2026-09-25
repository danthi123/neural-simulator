"""Hermetic tests for tools/before_you_build.sh's shared corpus-check log (research/corpus-check-shared-log,
2026-09-25): the script now resolves its OWN location before any `cd` and writes to the ONE log at the git
COMMON dir root, so a check made in any worktree of a repo is visible to a run started in any other.

Every test runs a COPY of the real script inside a throwaway `git init` repo under `tmp_path`, never the real
project's `.git` or `research/queue/`. `GIT_CEILING_DIRECTORIES=tmp_path` stops git from ever walking up past
the fixture (belt-and-braces against a `tmp_path` that happens to sit inside some ancestor's git tree --
`research/corpus-check-propagation`'s own review flagged tests that silently assumed tmp_path was outside
git). `stdin=subprocess.DEVNULL` on every call: with no `research/findings/*.md` in the fixture,
`before_you_build.sh`'s section-3 `grep` can receive an empty file list and fall through to reading its own
stdin -- a real hang risk this repo's own real corpus (thousands of findings) never triggers, unrelated to
this fix, and irrelevant to what these tests check, so stdin is closed rather than left to inherit whatever
fd the test runner happens to have open.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

REPO_ROOT = "/home/dant123/Projects/sim/.claude/worktrees/wf_87826bf5-9db-1"
SCRIPT_SRC = os.path.join(REPO_ROOT, "tools", "before_you_build.sh")


def _env(tmp_path, extra=None):
    env = dict(os.environ)
    env.pop("SIM_CORPUS_CHECK_LOG", None)
    env["GIT_CEILING_DIRECTORIES"] = str(tmp_path)
    if extra:
        env.update(extra)
    return env


def _init_repo(tmp_path, name="repo"):
    repo = tmp_path / name
    (repo / "tools").mkdir(parents=True)
    shutil.copy(SCRIPT_SRC, repo / "tools" / "before_you_build.sh")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, env=_env(tmp_path))
    return repo


def _run_script(script_path, args, cwd, env):
    return subprocess.run(
        ["bash", str(script_path)] + args, cwd=str(cwd), env=env,
        capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL,
    )


def _entries(path):
    with open(path) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def test_writes_to_the_git_common_dir_root(tmp_path):
    repo = _init_repo(tmp_path)
    res = _run_script(repo / "tools" / "before_you_build.sh", ["a defect"], repo, _env(tmp_path))
    assert res.returncode == 0, res.stderr

    log = repo / ".git" / "corpus_checks_shared.jsonl"
    assert log.exists(), "expected the shared log directly under the repo's .git"
    entries = _entries(log)
    assert len(entries) == 1
    assert entries[0]["query"] == "a defect"
    assert entries[0]["cwd"] == str(repo)
    assert "[recorded] corpus check logged to" in res.stdout


def test_relative_invocation_from_a_nested_cwd_still_finds_the_same_log(tmp_path):
    """The script's own location must be resolved BEFORE any `cd` -- a RELATIVE invocation from a
    subdirectory must not silently land the log somewhere else, or fail to find it at all.

    Mutation-verify: move the `_SCRIPT_DIR`/`_ORIG_CWD` capture to AFTER the `cd "$_SCRIPT_DIR/.."` line (the
    exact shape of the propagation-branch defect this replaces) and this test fails."""
    repo = _init_repo(tmp_path)
    sub = repo / "some" / "nested" / "dir"
    sub.mkdir(parents=True)
    res = subprocess.run(
        ["bash", "../../../tools/before_you_build.sh", "relative call"],
        cwd=str(sub), env=_env(tmp_path), capture_output=True, text=True, timeout=30,
        stdin=subprocess.DEVNULL,
    )
    assert res.returncode == 0, res.stderr

    log = repo / ".git" / "corpus_checks_shared.jsonl"
    entries = _entries(log)
    assert entries[-1]["query"] == "relative call"
    assert entries[-1]["cwd"] == str(sub), "recorded cwd should be the CALLER's cwd, not the repo root"


def test_shared_across_two_worktrees_of_the_same_repo(tmp_path):
    """The end-to-end property the whole fix exists for: a check made from one worktree lands in the exact
    same file a run started from a DIFFERENT worktree of the same repo will read."""
    repo = _init_repo(tmp_path)
    env = _env(tmp_path)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "add", "tools/before_you_build.sh"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True, env=env)
    wt = tmp_path / "wt"
    subprocess.run(["git", "worktree", "add", "-q", str(wt), "-b", "side"], cwd=repo, check=True, env=env)

    res_main = _run_script(repo / "tools" / "before_you_build.sh", ["from main checkout"], repo, env)
    res_wt = _run_script(wt / "tools" / "before_you_build.sh", ["from the worktree"], wt, env)
    assert res_main.returncode == 0, res_main.stderr
    assert res_wt.returncode == 0, res_wt.stderr

    log = repo / ".git" / "corpus_checks_shared.jsonl"           # the COMMON dir -- same physical file
    queries = [e["query"] for e in _entries(log)]
    assert "from main checkout" in queries
    assert "from the worktree" in queries, \
        "the worktree's check did not land in the same shared log as the main checkout's"


def test_exits_nonzero_and_logs_nothing_when_not_inside_any_git_repo(tmp_path):
    """`exit non-zero if the log path is empty` -- outside any git repo, git-common-dir resolution fails, and
    the script must refuse to log a check nowhere rather than silently swallow it."""
    plain = tmp_path / "plain"
    (plain / "tools").mkdir(parents=True)
    shutil.copy(SCRIPT_SRC, plain / "tools" / "before_you_build.sh")

    res = _run_script(plain / "tools" / "before_you_build.sh", ["no repo here"], plain, _env(tmp_path))
    assert res.returncode != 0
    assert "could not resolve the git common dir" in res.stderr


def test_sim_corpus_check_log_override_is_honored_and_records_the_caller_cwd(tmp_path):
    repo = _init_repo(tmp_path)
    override = tmp_path / "custom.jsonl"
    env = _env(tmp_path, {"SIM_CORPUS_CHECK_LOG": str(override)})

    res = _run_script(repo / "tools" / "before_you_build.sh", ["overridden"], repo, env)
    assert res.returncode == 0, res.stderr
    assert override.exists()
    assert not (repo / ".git" / "corpus_checks_shared.jsonl").exists()
    assert _entries(override)[0]["cwd"] == str(repo)


def test_missing_query_exits_with_usage_and_logs_nothing(tmp_path):
    repo = _init_repo(tmp_path)
    res = _run_script(repo / "tools" / "before_you_build.sh", [], repo, _env(tmp_path))
    assert res.returncode == 2
    assert not (repo / ".git" / "corpus_checks_shared.jsonl").exists()
