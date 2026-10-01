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
