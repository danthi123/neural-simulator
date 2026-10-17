"""tools/agent_ff_branch.sh: the driver agent fast-forwards its own research branch, never main (owner 2026-10-16)."""
import os
import shutil
import subprocess

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_ROOT, "tools", "agent_ff_branch.sh")


def _repo(tmp_path):
    def git(*a, cwd=tmp_path, check=True):
        return subprocess.run(["git", *a], cwd=cwd, check=check, capture_output=True, text=True)
    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@t"); git("config", "user.name", "t")
    (tmp_path / "tools").mkdir(); shutil.copy(_SCRIPT, tmp_path / "tools" / "agent_ff_branch.sh")
    (tmp_path / "a.txt").write_text("a\n")
    git("add", "-A"); git("commit", "-q", "-m", "base")
    return git


def _run(tmp_path, *a):
    return subprocess.run(["bash", "tools/agent_ff_branch.sh", *a], cwd=tmp_path, capture_output=True, text=True)


def _fix_on_top(git, tmp_path, branch, name="claude-build/fix"):
    git("branch", name, branch)
    wt = tmp_path / "fixwt"; git("worktree", "add", "-q", str(wt), name)
    (wt / "a.txt").write_text("fixed\n"); git("commit", "-q", "-am", "fix", cwd=wt)
    tip = git("rev-parse", name).stdout.strip()
    git("worktree", "remove", str(wt))
    return tip


def test_fast_forwards_a_branch_checked_out_in_a_worktree(tmp_path):
    git = _repo(tmp_path)
    git("branch", "research/x")
    wt = tmp_path / "wt"; git("worktree", "add", "-q", str(wt), "research/x")
    tip = _fix_on_top(git, tmp_path, "research/x")
    r = _run(tmp_path, "research/x", tip)
    assert r.returncode == 0, r.stderr
    assert git("rev-parse", "research/x").stdout.strip() == tip
    assert (wt / "a.txt").read_text() == "fixed\n"                  # the worktree moved with the ref
    assert _run(tmp_path, "research/x", tip).stdout.startswith("research/x already at")


def test_fast_forwards_a_branch_checked_out_nowhere(tmp_path):
    git = _repo(tmp_path)
    git("branch", "research/y")
    tip = _fix_on_top(git, tmp_path, "research/y")
    r = _run(tmp_path, "research/y", tip)
    assert r.returncode == 0, r.stderr
    assert git("rev-parse", "research/y").stdout.strip() == tip


def test_refuses_main_and_non_research_branches(tmp_path):
    git = _repo(tmp_path)
    tip = _fix_on_top(git, tmp_path, "main")
    for b in ("main", "claude-build/fix", "research/../main", "research/"):
        r = _run(tmp_path, b, tip)
        assert r.returncode == 2, (b, r.stdout, r.stderr)
    assert git("rev-parse", "main").stdout.strip() != tip


def test_refuses_a_target_that_is_not_a_fast_forward(tmp_path):
    git = _repo(tmp_path)
    git("checkout", "-q", "-b", "research/z")
    (tmp_path / "a.txt").write_text("branch\n"); git("commit", "-q", "-am", "branch work")
    git("checkout", "-q", "main")
    (tmp_path / "a.txt").write_text("main\n"); git("commit", "-q", "-am", "main work")
    before = git("rev-parse", "research/z").stdout
    r = _run(tmp_path, "research/z", "main")
    assert r.returncode == 2 and "not a fast-forward" in r.stderr
    assert git("rev-parse", "research/z").stdout == before


def test_refuses_a_dirty_worktree(tmp_path):
    git = _repo(tmp_path)
    git("branch", "research/d")
    wt = tmp_path / "wt"; git("worktree", "add", "-q", str(wt), "research/d")
    tip = _fix_on_top(git, tmp_path, "research/d")
    (wt / "a.txt").write_text("local edit\n")
    before = git("rev-parse", "research/d").stdout
    r = _run(tmp_path, "research/d", tip)
    assert r.returncode == 2 and "uncommitted" in r.stderr
    assert git("rev-parse", "research/d").stdout == before
    assert (wt / "a.txt").read_text() == "local edit\n"
