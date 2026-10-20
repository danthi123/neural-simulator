"""tools/agent_sync_branch.sh: the driver agent brings main INTO its own research branch (owner 2026-10-01)."""
import os
import shutil
import subprocess

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_ROOT, "tools", "agent_sync_branch.sh")
RB = "research/coordination/LOCAL_LLM_RUNBOOK.md"
FL = "research/FAILURE_LOG.md"


def _repo(tmp_path):
    def git(*a, cwd=tmp_path, check=True):
        return subprocess.run(["git", *a], cwd=cwd, check=check, capture_output=True, text=True)
    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@t"); git("config", "user.name", "t")
    (tmp_path / "tools").mkdir(); shutil.copy(_SCRIPT, tmp_path / "tools" / "agent_sync_branch.sh")
    (tmp_path / "research" / "coordination").mkdir(parents=True)
    (tmp_path / RB).write_text("# runbook\n- line\n")
    (tmp_path / FL).write_text("| date | failure |\n| d0 | base |\n")
    (tmp_path / "tools" / "gate.py").write_text("v = 1\n")
    (tmp_path / "sim").mkdir(); (tmp_path / "sim" / "core.py").write_text("x = 1\n")
    git("add", "-A"); git("commit", "-q", "-m", "base")
    return git


def _run(tmp_path, b):
    return subprocess.run(["bash", "tools/agent_sync_branch.sh", b], cwd=tmp_path, capture_output=True, text=True)


def test_syncs_main_into_a_worktree_branch_and_keeps_staged_files(tmp_path):
    git = _repo(tmp_path)
    git("branch", "research/old")
    wt = tmp_path / "wt"; git("worktree", "add", "-q", str(wt), "research/old")
    (wt / RB).write_text("# runbook\n- branch edit\n"); (wt / FL).write_text("| date | failure |\n| d0 | base |\n| d1 | branch row |\n")
    git("commit", "-q", "-am", "branch work", cwd=wt)
    (tmp_path / "tools" / "gate.py").write_text("v = 2\n"); (tmp_path / RB).write_text("# runbook\n- main edit\n")
    (tmp_path / FL).write_text("| date | failure |\n| d0 | base |\n| d2 | main row |\n")
    git("commit", "-q", "-am", "main fix")
    (wt / "new.json").write_text("{}\n"); git("add", "new.json", cwd=wt)

    r = _run(tmp_path, "research/old")
    assert r.returncode == 0, r.stderr
    assert git("merge-base", "--is-ancestor", "main", "research/old", check=False).returncode == 0
    assert (wt / "tools" / "gate.py").read_text() == "v = 2\n"
    assert (wt / RB).read_text() == "# runbook\n- main edit\n"              # main's runbook wins
    fl = (wt / FL).read_text(); assert "branch row" in fl and "main row" in fl and "<<<" not in fl
    assert git("diff", "--cached", "--name-only", cwd=wt).stdout.split() == ["new.json"]   # still staged
    assert len(git("rev-list", "--parents", "-n", "1", "research/old").stdout.split()) == 3
    r2 = _run(tmp_path, "research/old")
    assert r2.returncode == 0 and "already contains main" in r2.stdout


def test_other_conflicts_abort_and_leave_the_branch_unchanged(tmp_path):
    git = _repo(tmp_path)
    git("checkout", "-q", "-b", "research/clash")
    (tmp_path / "sim" / "core.py").write_text("x = 2\n"); git("commit", "-q", "-am", "branch code")
    git("checkout", "-q", "main")
    (tmp_path / "sim" / "core.py").write_text("x = 3\n"); git("commit", "-q", "-am", "main code")
    before = git("rev-parse", "research/clash").stdout

    r = _run(tmp_path, "research/clash")                                       # no worktree: uses a temporary one
    assert r.returncode == 2 and "sim/core.py" in r.stderr
    assert git("rev-parse", "research/clash").stdout == before
    assert "_sync-clash" not in git("worktree", "list").stdout
    assert not (tmp_path / ".claude" / "worktrees" / "_sync-clash").exists()
    assert git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "main"
    assert git("status", "--porcelain").stdout == ""


def test_refuses_main_and_non_research_branches(tmp_path):
    git = _repo(tmp_path)
    git("branch", "feature/x")
    for b in ("main", "feature/x", "research/../main", "research/nope"):
        r = _run(tmp_path, b)
        assert r.returncode == 2, b
    assert git("log", "--oneline", "main").stdout.count("\n") == 1


def test_a_staged_rename_and_deletion_stay_staged(tmp_path):
    git = _repo(tmp_path)
    (tmp_path / "keep.txt").write_text("k\n"); (tmp_path / "gone.txt").write_text("g\n")
    git("add", "-A"); git("commit", "-q", "-m", "files")
    git("branch", "research/mv")
    wt = tmp_path / "wt"; git("worktree", "add", "-q", str(wt), "research/mv")
    (tmp_path / "tools" / "gate.py").write_text("v = 2\n"); git("commit", "-q", "-am", "main fix")
    git("mv", "keep.txt", "kept.txt", cwd=wt); git("rm", "-q", "gone.txt", cwd=wt)
    before = git("diff", "--cached", "--name-status", "--no-renames", cwd=wt).stdout

    r = _run(tmp_path, "research/mv")
    assert r.returncode == 0, r.stderr
    assert git("diff", "--cached", "--name-status", "--no-renames", cwd=wt).stdout == before
    assert git("diff", "--name-only", cwd=wt).stdout == ""                    # nothing left unstaged


def test_worklog_conflicts_keep_both_sides(tmp_path):
    """Claude 2026-10-10: *WORKLOG.md files are append-only packet logs; a conflict keeps both sides' lines."""
    git = _repo(tmp_path)
    wl = tmp_path / "docs" / "drafts" / "pkt" / "SPIKING-WORKLOG.md"
    wl.parent.mkdir(parents=True); wl.write_text("# log\n- base\n"); git("add", "-A"); git("commit", "-q", "-m", "wl")
    git("checkout", "-q", "-b", "research/wl")
    wl.write_text("# log\n- base\n- branch entry\n"); git("commit", "-q", "-am", "branch log")
    git("checkout", "-q", "main")
    wl.write_text("# log\n- base\n- main entry\n"); git("commit", "-q", "-am", "main log")

    r = _run(tmp_path, "research/wl")
    assert r.returncode == 0, r.stderr
    out = git("show", "research/wl:docs/drafts/pkt/SPIKING-WORKLOG.md").stdout
    assert "branch entry" in out and "main entry" in out and "<<<" not in out


def test_lane_builds_conflicts_keep_both_sides(tmp_path):
    """Claude 2026-10-20 ([bg-tonic-loop-0] STOP): lane_builds.jsonl is an append-only registration log that
    gates/lane_starvation reads row by row, so a conflict keeps both sides' rows and every row stays valid JSON."""
    import json
    git = _repo(tmp_path)
    lb = tmp_path / "research" / "coordination" / "lane_builds.jsonl"
    row = lambda w, t: json.dumps({"lane": "C", "workflow": w, "branch": "research/x", "started": t}) + "\n"
    lb.write_text(row("base", 1.0)); git("add", "-A"); git("commit", "-q", "-m", "lanes")
    git("checkout", "-q", "-b", "research/lanes")
    lb.write_text(row("base", 1.0) + row("branch", 2.0)); git("commit", "-q", "-am", "branch lane")
    git("checkout", "-q", "main")
    lb.write_text(row("base", 1.0) + row("main", 3.0)); git("commit", "-q", "-am", "main lane")

    r = _run(tmp_path, "research/lanes")
    assert r.returncode == 0, r.stderr
    out = git("show", "research/lanes:research/coordination/lane_builds.jsonl").stdout
    rows = [json.loads(ln) for ln in out.splitlines() if ln.strip()]
    assert sorted(x["workflow"] for x in rows) == ["base", "branch", "main"]
