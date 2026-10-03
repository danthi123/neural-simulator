"""tools/gpu_remote.sh: the server -> workstation GPU bridge (docs/plans/2026-09-30-home-to-server-MIGRATION.md).

Hermetic: a stub `ssh` on PATH runs the remote script with `bash -s` (and git's `git-receive-pack`) locally, with HOME
set to a temp "workstation" home whose ~/Projects/sim is a clone of a temp "server" repo, with NO remote back to it.
The server repo commits tools/gpu_remote.sh itself plus a stub tools/local_llm/gpu_add.sh that records the job it was
handed (and whether its checkout was clean). No real ssh, queue, GPU or workstation is touched.

Pins: the job's revision reaches the workstation by PUSH over the same ssh connection (refs/server-jobs/<sha>), so the
workstation needs no credential to the server (2026-09-30 security re-review, CRITICAL); the job survives base64
transport byte for byte (the workstation's login shell is fish); the refusals on uncommitted code and on a HEAD that is
on no branch; the sparse per-job worktree (no research/findings/raw, clean to `git status`); the cleanup of a refused
job's worktree; and --dry-run running nothing.
"""
import os
import shutil
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GPU_REMOTE = ROOT / "tools" / "gpu_remote.sh"

GIT_ENV = {
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
    "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
}

STUB_GPU_ADD = """#!/usr/bin/env bash
# test stub: record the job and the cleanliness of the checkout it cds into
job="$1"; wt=$(printf '%s' "$job" | sed -E 's/^cd ([^ ]+) .*/\\1/')
printf '%s' "$job" > "$HOME/gpu_add_job"
git -C "$wt" status --porcelain > "$HOME/gpu_add_status"
exit "${STUB_GPU_ADD_RC:-0}"
"""

STUB_SSH = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$SSH_LOG"
case "${@: -1}" in "git-receive-pack "*) cd "$FAKE_HOME" && HOME="$FAKE_HOME" exec sh -c "${@: -1}" ;; esac
[ "${@: -2:1}" = bash ] && [ "${@: -1}" = -s ] || { echo "stub ssh: unexpected args: $*" >&2; exit 97; }
HOME="$FAKE_HOME" exec bash -s
"""


def _git(args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, env={**os.environ, **GIT_ENV})


def _exe(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _setup(tmp_path):
    server = tmp_path / "server"
    (server / "tools" / "local_llm").mkdir(parents=True)
    shutil.copy(GPU_REMOTE, server / "tools" / "gpu_remote.sh")
    _exe(server / "tools" / "local_llm" / "gpu_add.sh", STUB_GPU_ADD)
    raw = server / "research" / "findings" / "raw" / "old"
    raw.mkdir(parents=True)
    (raw / "result.json").write_text("{}")
    (server / "research" / "coordination").mkdir(parents=True)
    (server / "research" / "coordination" / "state.json").write_text("{}")
    (server / "code.py").write_text("x = 1\n")
    _git(["init", "-q", "-b", "main"], server)
    _git(["add", "-A"], server)
    _git(["commit", "-q", "-m", "init"], server)

    home = tmp_path / "home"
    wrepo = home / "Projects" / "sim"
    wrepo.parent.mkdir(parents=True)
    _git(["clone", "-q", str(server), str(wrepo)], tmp_path)
    _git(["remote", "remove", "origin"], wrepo)      # the workstation has no way to fetch from the server
    (wrepo / ".venv").mkdir()
    (wrepo / "data").mkdir()
    with open(wrepo / ".git" / "info" / "exclude", "a") as fh:   # as on the workstation
        fh.write(".venv\ndata\n")

    bindir = tmp_path / "bin"
    _exe(bindir / "ssh", STUB_SSH)
    env = {**os.environ, **GIT_ENV, "PATH": "%s:%s" % (bindir, os.environ["PATH"]), "FAKE_HOME": str(home),
           "SSH_LOG": str(tmp_path / "ssh.log")}
    env.pop("SIM_GPU_HOST", None)
    env.pop("SIM_GPU_HOST_REPO", None)
    return server, home, env


def _run(server, env, *args):
    return subprocess.run(["bash", str(server / "tools" / "gpu_remote.sh"), *args], cwd=server, env=env,
                          capture_output=True, text=True, timeout=60)


def test_add_hands_the_exact_job_to_the_worktrees_own_gpu_add(tmp_path):
    server, home, env = _setup(tmp_path)
    (server / "code.py").write_text("x = 'new on the server only'\n")   # a commit the workstation does not have
    _git(["commit", "-q", "-am", "server-only"], server)
    job = """echo "a  b" 'c$d' && x=$(date) ; printf '%s\\n' "`q`" | cat # ü"""
    res = _run(server, env, "add", job)
    assert res.returncode == 0, res.stderr
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=server, capture_output=True, text=True).stdout.strip()
    calls = (tmp_path / "ssh.log").read_text().splitlines()
    assert len(calls) == 2 and "git-receive-pack 'Projects/sim'" in calls[0], calls
    assert calls[1].split() == ["-o", "BatchMode=yes", "workstation", "bash", "-s"]
    wrepo = home / "Projects" / "sim"
    pushed = subprocess.run(["git", "rev-parse", "refs/server-jobs/" + sha], cwd=wrepo, capture_output=True, text=True)
    assert pushed.stdout.strip() == sha, "the job's revision must arrive by push, not by a fetch from the server"
    handed = (home / "gpu_add_job").read_text()
    prefix, _, rest = handed.partition(" && ")
    assert rest == job, "the job must reach gpu_add.sh byte for byte"
    wt = Path(prefix[len("cd "):])
    assert "new on the server only" in (wt / "code.py").read_text()
    assert wt.parent == home / "derisk-pool" / "revisions" and wt.name.startswith(sha + "-")
    assert (home / "gpu_add_status").read_text() == "", "the per-job worktree must be clean for gpu_add.sh"
    assert (wt / "code.py").exists() and (wt / "research" / "coordination" / "state.json").exists()
    assert not (wt / "research" / "findings" / "raw").exists(), "research/findings/raw must not be checked out"
    assert os.readlink(wt / ".venv") == str(home / "Projects" / "sim" / ".venv")
    assert os.readlink(wt / "data") == str(home / "Projects" / "sim" / "data")


def test_refused_job_leaves_no_worktree(tmp_path):
    server, home, env = _setup(tmp_path)
    res = _run(server, {**env, "STUB_GPU_ADD_RC": "1"}, "add", "echo hi")
    assert res.returncode == 1
    assert list((home / "derisk-pool" / "revisions").iterdir()) == []


def test_refuses_uncommitted_code_but_not_results_or_state(tmp_path):
    server, home, env = _setup(tmp_path)
    (server / "research" / "coordination" / "state.json").write_text('{"changed": 1}')
    (server / "research" / "findings" / "raw" / "old" / "result.json").write_text('{"changed": 1}')
    assert _run(server, env, "add", "echo ok").returncode == 0
    (server / "code.py").write_text("x = 2\n")
    res = _run(server, env, "add", "echo no")
    assert res.returncode == 1 and "uncommitted" in res.stderr
    assert len((tmp_path / "ssh.log").read_text().splitlines()) == 2, "a refused add must not reach the workstation"


def test_refuses_a_head_on_no_branch(tmp_path):
    server, home, env = _setup(tmp_path)
    (server / "code.py").write_text("x = 3\n")
    _git(["commit", "-q", "-am", "second"], server)
    _git(["checkout", "-q", "--detach", "HEAD"], server)
    _git(["reset", "-q", "--soft", "HEAD~1"], server)
    _git(["commit", "-q", "-m", "orphan"], server)
    res = _run(server, env, "add", "echo hi")
    assert res.returncode == 1 and "no branch" in res.stderr
    assert not (tmp_path / "ssh.log").exists()


def test_dry_run_prints_the_remote_script_and_runs_nothing(tmp_path):
    server, home, env = _setup(tmp_path)
    res = _run(server, env, "--dry-run", "status")
    assert res.returncode == 0
    assert 'bash "$HOME/Projects/sim/tools/gpu_queue.sh" status' in res.stdout
    res = _run(server, env, "--dry-run", "add", "echo hi")
    assert res.returncode == 0 and "sparse-checkout set --no-cone" in res.stdout and "gpu_add.sh" in res.stdout
    assert "# git push workstation:Projects/sim " in res.stdout and "fetch" not in res.stdout
    assert not (tmp_path / "ssh.log").exists()
    assert not (home / "derisk-pool").exists()


def test_status_log_yield_provision_argument_checks(tmp_path):
    server, home, env = _setup(tmp_path)
    assert _run(server, env, "log", "x").returncode == 2
    assert _run(server, env, "yield", "--max-hours", "1;id").returncode == 2
    assert _run(server, env, "provision", "not-a-sha").returncode == 2
    res = _run(server, env, "--dry-run", "yield", "--max-hours", "1.5")
    assert res.returncode == 0 and "llm.sh\" yield --max-hours 1.5" in res.stdout
    assert not (tmp_path / "ssh.log").exists()


def test_provision_pushes_the_revision_and_keeps_the_given_sha(tmp_path):
    server, home, env = _setup(tmp_path)
    (server / "code.py").write_text("x = 'provision me'\n")
    _git(["commit", "-q", "-am", "to-provision"], server)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=server, capture_output=True, text=True).stdout.strip()
    res = _run(server, env, "--dry-run", "provision", sha[:10])
    assert res.returncode == 0, res.stderr
    assert "# git push workstation:Projects/sim %s:refs/server-jobs/%s" % (sha, sha) in res.stdout
    assert "--revision %s\n" % sha[:10] in res.stdout, "the remote script must keep the sha as given"
    assert _run(server, env, "provision", "deadbeef00").returncode == 2, "an unknown commit is refused before any ssh"
    assert not (tmp_path / "ssh.log").exists()


def test_add_rev_runs_a_pinned_older_revision_not_head(tmp_path):
    """--rev <sha> (session 4): a gate pin older than HEAD runs at that pin; the working tree does not matter."""
    server, home, env = _setup(tmp_path)
    pin = subprocess.run(["git", "rev-parse", "HEAD"], cwd=server, capture_output=True, text=True).stdout.strip()
    (server / "code.py").write_text("x = 'newer than the pin'\n")
    _git(["commit", "-q", "-am", "newer"], server)
    (server / "code.py").write_text("x = 'uncommitted'\n")      # irrelevant under --rev
    res = _run(server, env, "add", "--rev", pin[:9], "echo pinned")
    assert res.returncode == 0, res.stderr
    wrepo = home / "Projects" / "sim"
    pushed = subprocess.run(["git", "rev-parse", "refs/server-jobs/" + pin], cwd=wrepo, capture_output=True, text=True)
    assert pushed.stdout.strip() == pin
    prefix, _, rest = (home / "gpu_add_job").read_text().partition(" && ")
    assert rest == "echo pinned"
    wt = Path(prefix[len("cd "):])
    assert wt.name.startswith(pin + "-") and (wt / "code.py").read_text() == "x = 1\n"


def test_add_rev_refusals(tmp_path):
    server, home, env = _setup(tmp_path)
    assert _run(server, env, "add", "--rev", "zz", "echo x").returncode == 2
    assert _run(server, env, "add", "--rev", "deadbeef00", "echo x").returncode == 2
    _git(["checkout", "-q", "--detach", "HEAD"], server)          # a commit on no branch
    (server / "code.py").write_text("x = 9\n")
    _git(["commit", "-q", "-am", "orphan"], server)
    orphan = subprocess.run(["git", "rev-parse", "HEAD"], cwd=server, capture_output=True, text=True).stdout.strip()
    _git(["checkout", "-q", "main"], server)
    res = _run(server, env, "add", "--rev", orphan, "echo x")
    assert res.returncode == 1 and "no branch" in res.stderr
    _git(["rm", "-q", "tools/local_llm/gpu_add.sh"], server)        # a revision without the guard
    _git(["commit", "-q", "-m", "no guard"], server)
    res = _run(server, env, "add", "--rev", "HEAD", "echo x")
    assert res.returncode == 2   # not a hex sha
    noguard = subprocess.run(["git", "rev-parse", "HEAD"], cwd=server, capture_output=True, text=True).stdout.strip()
    res = _run(server, env, "add", "--rev", noguard, "echo x")
    assert res.returncode == 1 and "gpu_add.sh" in res.stderr
    assert not (tmp_path / "ssh.log").exists()
