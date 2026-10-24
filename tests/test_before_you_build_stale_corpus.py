"""before_you_build.sh refuses to record a corpus check when local main lacks published main (stale-corpus guard,
2026-10-21): the workstation main sat 1,184 commits behind and every RAG search there read a 15-day-old corpus."""
import os
import shutil
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "before_you_build.sh"


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _repo_pair(tmp_path):
    seed = tmp_path / "seed"
    (seed / "tools").mkdir(parents=True)
    shutil.copy(SCRIPT, seed / "tools" / "before_you_build.sh")
    (seed / "research" / "findings").mkdir(parents=True)
    (seed / "research" / "findings" / "a.md").write_text("first\n")
    _git(seed, "init", "-q", "-b", "main")
    _git(seed, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    _git(seed, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c1")
    remote = tmp_path / "remote.git"
    _git(tmp_path, "clone", "-q", "--bare", str(seed), str(remote))
    local = tmp_path / "local"
    _git(tmp_path, "clone", "-q", str(remote), str(local))
    (seed / "research" / "findings" / "b.md").write_text("second\n")
    _git(seed, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    _git(seed, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c2")
    _git(seed, "push", "-q", str(remote), "main")
    return remote, local


def _run(local, log):
    env = dict(os.environ, SIM_CORPUS_CHECK_LOG=str(log))
    return subprocess.run(["bash", "tools/before_you_build.sh", "a defect"], cwd=local, env=env,
                          capture_output=True, text=True, timeout=120)


def test_stale_local_main_refuses_and_records_nothing(tmp_path):
    _remote, local = _repo_pair(tmp_path)
    log = tmp_path / "checks.jsonl"
    r = _run(local, log)
    assert r.returncode == 4, r.stdout + r.stderr
    assert "STALE CORPUS" in r.stdout
    assert not log.exists() or log.read_text() == ""


def test_current_local_main_passes_and_records(tmp_path):
    _remote, local = _repo_pair(tmp_path)
    _git(local, "pull", "-q", "--ff-only", "origin", "main")
    log = tmp_path / "checks.jsonl"
    r = _run(local, log)
    assert "STALE CORPUS" not in r.stdout
    assert "local main contains published main" in r.stdout
    assert r.returncode == 0, r.stdout + r.stderr
    assert '"query": "a defect"' in log.read_text()


def test_feature_branch_worktree_reads_main_not_its_own_branch(tmp_path):
    _remote, local = _repo_pair(tmp_path)
    _git(local, "pull", "-q", "--ff-only", "origin", "main")
    _git(local, "checkout", "-q", "-b", "old-branch", "HEAD~1")
    r = _run(local, tmp_path / "checks.jsonl")
    assert "STALE CORPUS" not in r.stdout, r.stdout
