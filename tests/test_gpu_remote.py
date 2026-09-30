"""tools/gpu_remote.sh: the server -> workstation GPU bridge (docs/plans/2026-09-30-home-to-server-MIGRATION.md).

Hermetic: a stub `ssh` on PATH runs the remote script with `bash -s` locally, with HOME set to a temp "workstation"
home whose ~/Projects/sim is a clone with a `server` remote pointing at a temp "server" repo. The server repo commits
tools/gpu_remote.sh itself plus a stub tools/local_llm/gpu_add.sh that records the job it was handed (and whether its
checkout was clean). No real ssh, queue, GPU or workstation is touched.

Pins: the job survives base64 transport byte for byte (the workstation's login shell is fish), the refusals on
uncommitted code and on a HEAD that is on no branch, the sparse per-job worktree (no research/findings/raw, clean to
`git status`), the cleanup of a refused job's worktree, and --dry-run running nothing.
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
    _git(["remote", "add", "server", str(server)], wrepo)
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
    job = """echo "a  b" 'c$d' && x=$(date) ; printf '%s\\n' "`q`" | cat # ü"""
    res = _run(server, env, "add", job)
    assert res.returncode == 0, res.stderr
    assert (tmp_path / "ssh.log").read_text().split() == ["-o", "BatchMode=yes", "workstation", "bash", "-s"]
    handed = (home / "gpu_add_job").read_text()
    prefix, _, rest = handed.partition(" && ")
    assert rest == job, "the job must reach gpu_add.sh byte for byte"
    wt = Path(prefix[len("cd "):])
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=server, capture_output=True, text=True).stdout.strip()
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
    assert len((tmp_path / "ssh.log").read_text().splitlines()) == 1, "a refused add must not reach the workstation"


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
