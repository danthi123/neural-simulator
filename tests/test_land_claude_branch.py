"""tools/land_claude_branch.sh: the owner lands a cloud Claude branch on main, pinned to the reported sha."""
import os
import shutil
import subprocess

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_lands_only_the_reported_sha(tmp_path):
    main = tmp_path / "main"; main.mkdir()
    def git(*a, cwd=main, check=True):
        return subprocess.run(["git", *a], cwd=cwd, check=check, capture_output=True, text=True)
    git("init", "-q", "-b", "main"); git("config", "user.email", "t@t"); git("config", "user.name", "t")
    (main / "tools").mkdir(); shutil.copy(os.path.join(_ROOT, "tools", "land_claude_branch.sh"), main / "tools")
    (main / "a.txt").write_text("a\n"); git("add", "-A"); git("commit", "-q", "-m", "base")
    imp = tmp_path / "import.git"
    git("clone", "-q", "--bare", str(main), str(imp), cwd=tmp_path)
    cloud = tmp_path / "cloud"; git("clone", "-q", str(imp), str(cloud), cwd=tmp_path)
    git("config", "user.email", "c@c", cwd=cloud); git("config", "user.name", "c", cwd=cloud)
    git("checkout", "-q", "-b", "claude/flip", cwd=cloud)
    (cloud / "b.txt").write_text("b\n"); git("add", "-A", cwd=cloud); git("commit", "-q", "-m", "work", cwd=cloud)
    git("push", "-q", "origin", "claude/flip", cwd=cloud)
    sha = git("rev-parse", "HEAD", cwd=cloud).stdout.strip()
    (main / "c.txt").write_text("c\n"); git("add", "-A"); git("commit", "-q", "-m", "main moved")
    env = dict(os.environ, CLAUDE_IMPORT_REPO=str(imp))
    run = lambda *a: subprocess.run(["bash", "tools/land_claude_branch.sh", *a], cwd=main, env=env,
                                    capture_output=True, text=True)

    head = git("rev-parse", "HEAD").stdout
    for bad in (("claude/flip", "deadbeef"), ("research/x", sha), ("claude/flip",), ("claude/../main", sha)):
        r = run(*bad)
        assert r.returncode == 2, bad
    assert git("rev-parse", "HEAD").stdout == head
    r = run("claude/flip", sha[:10])
    assert r.returncode == 0, r.stderr
    assert (main / "b.txt").exists() and len(git("rev-list", "--parents", "-n", "1", "HEAD").stdout.split()) == 3
    assert "already on main" in run("claude/flip", sha).stdout


# ── stray state in the live checkout (owner 2026-10-01: the first two landings of claude/session1-backlog failed on
#    377 untracked iw2 raw files the branch adds, and on runs.jsonl lines appended by running jobs) ─────────────────
LEDGER = "research/findings/raw/_provenance/runs.jsonl"


def _setup(tmp_path, fail_gate=False):
    main = tmp_path / "main"; main.mkdir()

    def git(*a, cwd=main, check=True):
        return subprocess.run(["git", *a], cwd=cwd, check=check, capture_output=True, text=True)
    git("init", "-q", "-b", "main"); git("config", "user.email", "t@t"); git("config", "user.name", "t")
    (main / "tools").mkdir(); shutil.copy(os.path.join(_ROOT, "tools", "land_claude_branch.sh"), main / "tools")
    (main / ".gitattributes").write_text("research/findings/raw/_provenance/*.jsonl merge=union\n")
    led = main / LEDGER; led.parent.mkdir(parents=True); led.write_text('{"run": 1}\n')
    git("add", "-A"); git("commit", "-q", "-m", "base")
    imp = tmp_path / "import.git"
    git("clone", "-q", "--bare", str(main), str(imp), cwd=tmp_path)
    cloud = tmp_path / "cloud"; git("clone", "-q", str(imp), str(cloud), cwd=tmp_path)
    git("config", "user.email", "c@c", cwd=cloud); git("config", "user.name", "c", cwd=cloud)
    git("checkout", "-q", "-b", "claude/work", cwd=cloud)
    raw = cloud / "research/findings/raw/_iw2"; raw.mkdir(parents=True)
    for i in range(3):
        (raw / ("r%d.json" % i)).write_text('{"cell": %d}\n' % i)
    with open(cloud / LEDGER, "a") as fh:
        fh.write('{"run": "cloud"}\n')
    git("add", "-A", cwd=cloud); git("commit", "-q", "-m", "work", cwd=cloud)
    git("push", "-q", "origin", "claude/work", cwd=cloud)
    sha = git("rev-parse", "HEAD", cwd=cloud).stdout.strip()
    # the live checkout: the branch's raw files already synced in (untracked), and a running job appending a line
    (main / "research/findings/raw/_iw2").mkdir(parents=True)
    for i in range(3):
        (main / "research/findings/raw/_iw2" / ("r%d.json" % i)).write_text('{"cell": %d}\n' % i)
    with open(main / LEDGER, "a") as fh:
        fh.write('{"run": "live job"}\n')
    if fail_gate:
        hook = main / ".git/hooks/pre-commit"; hook.write_text("#!/bin/sh\necho 'gate says no' >&2\nexit 1\n")
        hook.chmod(0o755)
    env = dict(os.environ, CLAUDE_IMPORT_REPO=str(imp), LAND_BACKUP_ROOT=str(tmp_path / "scratch"))
    run = lambda *a: subprocess.run(["bash", "tools/land_claude_branch.sh", *a], cwd=main, env=env,
                                    capture_output=True, text=True)
    return main, git, run, sha


def test_identical_untracked_files_move_aside_and_ledger_appends_survive_the_landing(tmp_path):
    main, git, run, sha = _setup(tmp_path)
    r = run("claude/work", sha)
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(git("rev-list", "--parents", "-n", "1", "HEAD").stdout.split()) == 3          # landed as a merge
    backups = list((tmp_path / "scratch").glob("land-backup-*/files/research/findings/raw/_iw2/r*.json"))
    assert len(backups) == 3, r.stdout
    # the committed ledger is the union merge WITHOUT the live job's line; the line is back, appended and unstaged
    assert git("show", "HEAD:" + LEDGER).stdout == '{"run": 1}\n{"run": "cloud"}\n'
    assert (main / LEDGER).read_text() == '{"run": 1}\n{"run": "cloud"}\n{"run": "live job"}\n'
    assert git("diff", "--cached", "--name-only").stdout == ""
    assert git("diff", "--name-only").stdout.split() == [LEDGER]


def test_untracked_file_with_different_content_refuses_and_touches_nothing(tmp_path):
    main, git, run, sha = _setup(tmp_path)
    (main / "research/findings/raw/_iw2/r1.json").write_text('{"cell": "LOCAL EDIT"}\n')
    head = git("rev-parse", "HEAD").stdout
    r = run("claude/work", sha)
    assert r.returncode == 2 and "r1.json" in r.stderr and "DIFFERENT" in r.stderr, r.stderr
    assert git("rev-parse", "HEAD").stdout == head
    assert (main / "research/findings/raw/_iw2/r1.json").read_text() == '{"cell": "LOCAL EDIT"}\n'
    assert (main / "research/findings/raw/_iw2/r0.json").exists()                         # nothing moved
    assert (main / LEDGER).read_text() == '{"run": 1}\n{"run": "live job"}\n'


def test_a_gate_refusal_puts_the_moved_files_and_ledger_lines_back(tmp_path):
    main, git, run, sha = _setup(tmp_path, fail_gate=True)
    head = git("rev-parse", "HEAD").stdout
    r = run("claude/work", sha)
    assert r.returncode == 2 and "gates refused" in r.stderr, r.stdout + r.stderr
    assert git("rev-parse", "HEAD").stdout == head and not (main / ".git/MERGE_HEAD").exists()
    for i in range(3):
        assert (main / "research/findings/raw/_iw2" / ("r%d.json" % i)).read_text() == '{"cell": %d}\n' % i
    assert (main / LEDGER).read_text() == '{"run": 1}\n{"run": "live job"}\n'
    assert git("diff", "--cached", "--name-only").stdout == ""


def test_a_ledger_change_that_is_not_a_pure_append_refuses(tmp_path):
    main, git, run, sha = _setup(tmp_path)
    (main / LEDGER).write_text('{"run": "REWRITTEN"}\n')
    head = git("rev-parse", "HEAD").stdout
    r = run("claude/work", sha)
    assert r.returncode == 2 and "not pure appends" in r.stderr, r.stderr
    assert git("rev-parse", "HEAD").stdout == head
    assert (main / LEDGER).read_text() == '{"run": "REWRITTEN"}\n'
    assert (main / "research/findings/raw/_iw2/r0.json").exists()


def test_fetches_an_import_repo_owned_by_another_uid(tmp_path):
    """The server's import repo belongs to another uid. `git -c safe.directory=...` does not reach the upload-pack a
    local fetch spawns (git 2.43: 'detected dubious ownership'); the script must still fetch it."""
    import pytest
    if os.geteuid() != 0:
        pytest.skip("needs root to give the import repo another owner")
    main, git, run, sha = _setup(tmp_path)
    subprocess.run(["chown", "-R", "nobody", str(tmp_path / "import.git")], check=True)
    r = run("claude/work", sha)
    assert "dubious ownership" not in r.stderr and r.returncode == 0, r.stdout + r.stderr
