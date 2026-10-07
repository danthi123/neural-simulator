"""tools/needs_line.py: one NEEDS-CLAUDE line in the runbook's format, at most 300 characters, and never filed past an
existing section-6 ruling for the same key without --rulings-read (session 6: T12 re-filed after its 10-01 ruling)."""
import importlib.util
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("needs_line", os.path.join(_ROOT, "tools", "needs_line.py"))
N = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(N)

RUNBOOK = """# runbook
## 6. RULINGS (Claude writes)

- T12-seam-map: done. T12 is closed. (2026-10-01)
- other: about T120 and T1-A0, not this key. (2026-10-02)
  continued line
- handled: [T12] (2026-10-02)

## NEEDS CLAUDE

- 2026-10-01 00:00 [x] INFO: seeded
"""


def _rb(tmp_path):
    p = tmp_path / "rb.md"
    p.write_text(RUNBOOK)
    return str(p)


def test_compose_format_and_whitespace():
    import datetime
    line = N.compose("T24", "STOP", "a  b\n c", "#1", datetime.datetime(2026, 10, 4, 4, 0))
    assert line == "- 2026-10-04 04:00 [T24] STOP #1: a b c"


def test_rulings_for_matches_the_key_only():
    found = N.rulings_for("T12", RUNBOOK)
    assert len(found) == 2 and found[0].startswith("- T12-seam-map")       # T12-seam-map and handled: [T12]
    assert N.rulings_for("T1", RUNBOOK) == []                              # T12, T120, T1-A0 are other keys
    assert len(N.rulings_for("T120", RUNBOOK)) == 1


def test_refuses_long_lines_and_unread_rulings(tmp_path):
    rb = _rb(tmp_path)
    assert N.main(["--key", "T9", "--kind", "INFO", "--text", "x" * 400, "--runbook", rb, "--write"]) == 1
    assert N.main(["--key", "T12", "--kind", "STOP", "--text", "file missing", "--runbook", rb, "--write"]) == 1
    assert open(rb).read() == RUNBOOK                                       # nothing written on a refusal
    assert N.main(["--key", "T12", "--kind", "STOP", "--text", "file missing", "--runbook", rb, "--write",
                   "--rulings-read"]) == 0
    assert open(rb).read().rstrip().endswith("[T12] STOP: file missing")
    assert N.main(["--key", "T9", "--kind", "INFO", "--text", "ok", "--runbook", rb]) == 0   # print only
    assert open(rb).read().count("\n- 20") == 2


# ── a branch this line hands Claude must not sit in a dirty worktree (Claude review 2026-10-06) ──────────────────────
import subprocess  # noqa: E402


def _g(repo, *a):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],
                          capture_output=True, text=True, check=True).stdout


def _repo_with_worktree(tmp_path, branch="research/foo"):
    repo = tmp_path / "repo"
    repo.mkdir()
    _g(repo, "init", "-b", "main")
    (repo / ".gitignore").write_text("ignored.log\n")
    (repo / "a.txt").write_text("a\n")
    _g(repo, "add", "-A")
    _g(repo, "commit", "-m", "init")
    wt = tmp_path / "wt"
    _g(repo, "worktree", "add", "-b", branch, str(wt))
    return repo, wt


def _run(tmp_path, repo, text, kind="RESULT", extra=()):
    return N.main(["--key", "k", "--kind", kind, "--text", text, "--runbook", _rb(tmp_path), "--repo", str(repo), *extra])


def test_dirty_worktree_of_a_named_branch_refuses_asks_and_results(tmp_path, capsys):
    repo, wt = _repo_with_worktree(tmp_path)
    assert _run(tmp_path, repo, "built on research/foo, please merge") == 0                    # clean: fine
    (wt / "a.txt").write_text("changed\n")                                                     # tracked change
    assert _run(tmp_path, repo, "built on research/foo, please merge") == 1
    err = capsys.readouterr()
    assert "research/foo" in err.out and "a.txt" in err.out and "REFUSED" in err.err
    _g(wt, "checkout", "--", "a.txt")
    (wt / "new.txt").write_text("x\n")                                                         # untracked, not ignored
    assert _run(tmp_path, repo, "ASK merge research/foo", kind="ASK") == 1
    (wt / "new.txt").unlink()
    (wt / "ignored.log").write_text("x\n")                                                     # ignored: not dirty
    assert _run(tmp_path, repo, "ASK merge research/foo", kind="ASK") == 0


def test_dirty_check_only_for_ask_and_result_and_only_for_real_branches(tmp_path):
    repo, wt = _repo_with_worktree(tmp_path)
    (wt / "a.txt").write_text("changed\n")
    assert _run(tmp_path, repo, "stuck on research/foo", kind="STOP") == 0
    assert _run(tmp_path, repo, "fyi research/foo", kind="INFO") == 0
    assert _run(tmp_path, repo, "see research/findings/raw/x.json and docs/y.md, merge please") == 0   # paths, not branches
    assert _run(tmp_path, repo, "merge research/nonexistent") == 0
    assert _run(tmp_path, repo, "merge it", extra=["--branch", "research/foo"]) == 1                    # explicit --branch


def test_dirty_ok_files_with_the_reason_in_the_line(tmp_path, capsys):
    repo, wt = _repo_with_worktree(tmp_path)
    (wt / "a.txt").write_text("changed\n")
    rb = _rb(tmp_path)
    assert N.main(["--key", "k", "--kind", "RESULT", "--text", "merge research/foo", "--runbook", rb, "--repo", str(repo),
                   "--dirty-ok", "state file regenerated by the heartbeat", "--write"]) == 0
    assert open(rb).read().rstrip().endswith("[dirty-ok: state file regenerated by the heartbeat]")
    assert N.main(["--key", "k", "--kind", "RESULT", "--text", "merge research/foo", "--runbook", rb, "--repo", str(repo),
                   "--dirty-ok", " "]) == 2


def test_a_branch_without_a_worktree_is_not_refused(tmp_path):
    repo, wt = _repo_with_worktree(tmp_path)
    _g(repo, "branch", "research/bare")
    assert _run(tmp_path, repo, "merge research/bare") == 0
