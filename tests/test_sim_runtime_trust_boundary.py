"""The server's trust boundary (2026-09-30, docs/plans/2026-09-30-home-to-server-MIGRATION.md section 10).

On srv-host-0 the pool dispatcher and the result pull run as the unprivileged user `sim` from a reviewed checkout the
agent cannot write; the agent-writable workspace is only DATA (SIM_WORKSPACE). These tests pin what that requires of
the scripts themselves:

  * tools/pool_autodispatch.sh in runtime mode reads the queue from the workspace, runs no git, and refuses a queue
    that is not a regular file on the workspace's own filesystem (a symlink to a secret would otherwise be copied into
    agent-readable quarantine files);
  * a node's capacity-probe answer never reaches bash arithmetic unchecked (`a[$(cmd)]` would run cmd locally);
  * node names from files are plain host aliases only (`-oProxyCommand=...` is an ssh option);
  * tools/pool_sync.sh in runtime mode pulls into a private staging dir and tools/safe_copy_into.py moves the files
    into the workspace without following a planted symlink; a host that is reachable but has no main results dir
    (the workstation) no longer reads as UNREACHABLE, and its revisions are still pulled.

The pool_sync tests run REAL rsync against a local fake node: the stub `ssh` drops its options and host and runs the
remote command with the fake node's HOME.
"""
from __future__ import annotations

import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DISPATCHER = ROOT / "tools" / "pool_autodispatch.sh"
POOL_SYNC = ROOT / "tools" / "pool_sync.sh"
SAFE_COPY = ROOT / "tools" / "safe_copy_into.py"


def _exe(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _run(script: Path, args: list[str], env: dict[str, str], cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(script), *args], cwd=cwd, env={**os.environ, **env}, capture_output=True,
                          text=True, timeout=120)


# ------------------------------------------------------------------ the dispatcher

def _workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "projects" / "neural-simulator"
    (ws / "research" / "queue").mkdir(parents=True)
    return ws


def test_runtime_mode_pops_from_the_workspace_queue_and_runs_no_git(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    now = int(time.time())
    (ws / "research" / "queue" / "pool.queue").write_text(f"{now}\techo hi  #checked:r\n")
    git_log = tmp_path / "git.log"
    bin_dir = tmp_path / "bin"
    _exe(bin_dir / "git", f'echo "$*" >> "{git_log}"; exit 1\n')
    res = _run(DISPATCHER, ["--pop-once"], {"SIM_WORKSPACE": str(ws), "PATH": f"{bin_dir}:{os.environ['PATH']}"})
    assert res.returncode == 0, res.stderr
    assert res.stdout.endswith("echo hi")
    assert (ws / "research" / "queue" / "pool.queue").read_text() == ""
    assert "echo hi" in (ws / "research" / "queue" / "pool.queue.claims").read_text()
    assert not git_log.exists(), "runtime mode must not run git (it would execute the workspace's .git/config)"


def test_runtime_mode_refuses_a_queue_symlinked_off_the_workspace_filesystem(tmp_path: Path) -> None:
    secret = Path("/etc/passwd")
    if os.stat(secret).st_dev == os.stat(tmp_path).st_dev:
        pytest.skip("needs /etc and the pytest tmp dir on different filesystems")
    ws = _workspace(tmp_path)
    q = ws / "research" / "queue" / "pool.queue"
    q.symlink_to(secret)
    res = _run(DISPATCHER, ["--pop-once"], {"SIM_WORKSPACE": str(ws)})
    assert "REFUSING the queue" in res.stderr
    qd = ws / "research" / "queue"
    for name in ("pool.queue.malformed", "pool.queue.claims", "pool.queue.unchecked"):
        assert not (qd / name).exists(), f"{name} was written from a queue that points off the workspace"
    assert q.is_symlink(), "the refused queue must be left as it was"


def test_runtime_mode_accepts_a_same_filesystem_symlink_and_legacy_mode_is_unchanged(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    now = int(time.time())
    real = ws / "elsewhere.queue"
    real.write_text(f"{now}\techo same-fs  #checked:r\n")
    (ws / "research" / "queue" / "pool.queue").symlink_to(real)
    res = _run(DISPATCHER, ["--pop-once"], {"SIM_WORKSPACE": str(ws)})
    assert res.returncode == 0 and res.stdout.endswith("echo same-fs"), res.stderr
    # legacy mode (no SIM_WORKSPACE, explicit POOL_QUEUE_PATH): the malformed quarantine still rewrites the queue
    q = tmp_path / "legacy.queue"
    q.write_text(f"not-a-record\n{now}\techo legacy  #checked:r\n")
    res = _run(DISPATCHER, ["--pop-once"], {"POOL_QUEUE_PATH": str(q), "POOL_RUNNING_PATH": str(tmp_path / "r")})
    assert res.stdout.endswith("echo legacy") and "not-a-record" in Path(str(q) + ".malformed").read_text()
    assert q.read_text() == ""


def test_runtime_mode_read_only_seams_go_through_the_snapshot_guard(tmp_path: Path) -> None:
    # --peek-est-gb / --min-queued-est-gb used to read $QUEUE directly, a second reader path around queue_snapshot's
    # device check (2026-09-30 re-review, LOW). A queue symlinked off the workspace filesystem is now refused there too.
    shm = Path("/dev/shm")
    if not (shm.is_dir() and os.access(shm, os.W_OK)) or os.stat(shm).st_dev == os.stat(tmp_path).st_dev:
        pytest.skip("needs a writable /dev/shm on a different filesystem from the pytest tmp dir")
    now = int(time.time())
    ws = _workspace(tmp_path)
    snapdir = tmp_path / "snaps"
    snapdir.mkdir()
    env = {"SIM_WORKSPACE": str(ws), "TMPDIR": str(snapdir), "POOL_JOB_EST_GB": "1"}
    off = shm / f"pool_queue_offfs_{os.getpid()}_{now}"
    off.write_text(f"{now}\tpython x mem_gb=9  #checked:r\n")
    try:
        q = ws / "research" / "queue" / "pool.queue"
        q.symlink_to(off)
        res = _run(DISPATCHER, ["--peek-est-gb"], env)
        assert res.stdout.strip() == "1" and "REFUSING the queue" in res.stderr, (res.stdout, res.stderr)
        res = _run(DISPATCHER, ["--min-queued-est-gb"], env)
        assert res.stdout.strip() == "" and "REFUSING the queue" in res.stderr, (res.stdout, res.stderr)
        q.unlink()
        q.write_text(f"{now}\tpython x mem_gb=9  #checked:r\n")     # a regular file on the workspace: read as before
        assert _run(DISPATCHER, ["--peek-est-gb"], env).stdout.strip() == "9"
        assert _run(DISPATCHER, ["--min-queued-est-gb"], env).stdout.strip() == "9"
        assert list(snapdir.iterdir()) == [], "a seam left its private snapshot behind"
    finally:
        off.unlink()


def test_runtime_mode_loop_refuses_to_start_unconfined(tmp_path: Path) -> None:
    # The worktree running this test is writable, exactly like /home/sim/runtime outside its unit's ProtectHome.
    ws = _workspace(tmp_path)
    res = _run(DISPATCHER, [], {"SIM_WORKSPACE": str(ws)})
    assert res.returncode == 1 and "REFUSING to start" in res.stderr and "writable code root" in res.stderr
    assert not (ws / "research" / "queue" / "pool.queue").exists(), "runtime mode must create nothing up front"


@pytest.mark.parametrize("answer,busy", [
    ("4 0.10 0 10 0 ROOT[$(touch${IFS}PWNED)]", True),   # total_gb reached $(( )) unchecked (ran on the old code)
    ("ROOT[$(touch${IFS}PWNED)] 0.10 0 10 0 15", True),  # cores reached $(( )) unchecked
    ("4 0.10 0 10 0 15\nPOOL_JOB_ID=x POOL_JOB_MEM_GB=a[$(touch${IFS}PWNED)]", False),   # awk sums it as 0
])
def test_a_node_probe_answer_never_reaches_arithmetic(tmp_path: Path, answer: str, busy: bool) -> None:
    bin_dir = tmp_path / "bin"
    _exe(bin_dir / "ssh", f"cat <<'EOF'\n{answer}\nEOF\n")
    res = _run(DISPATCHER, ["--node-idle", "pool41"],
               {"PATH": f"{bin_dir}:{os.environ['PATH']}", "POOL_SSH_CONFIG": str(tmp_path / "none"),
                "POOL_QUEUE_PATH": str(tmp_path / "q" / "pool.queue"), "POOL_RESERVATIONS_PATH": str(tmp_path / "resv")},
               cwd=tmp_path)
    assert not (tmp_path / "PWNED").exists(), "a node's answer ran a command on the dispatcher host"
    assert (res.returncode == 1) is busy


def test_a_plain_numeric_probe_answer_still_reads_idle(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    _exe(bin_dir / "ssh", 'echo "12 0.50 2 10 1 15"\n')
    res = _run(DISPATCHER, ["--node-idle", "pool41"],
               {"PATH": f"{bin_dir}:{os.environ['PATH']}", "POOL_SSH_CONFIG": str(tmp_path / "none"),
                "POOL_RESERVATIONS_PATH": str(tmp_path / "resv"), "POOL_QUEUE_PATH": str(tmp_path / "q" / "pool.queue")})
    assert res.returncode == 0, res.stderr


def test_node_names_from_files_must_be_plain_host_aliases(tmp_path: Path) -> None:
    extra = tmp_path / "extra"
    extra.write_text("pool1\n-oProxyCommand=touch${IFS}PWNED\n*\npool2.lan\n")
    res = _run(DISPATCHER, ["--nodes-this-cycle"], {"POOL_NODES": "pool41", "POOL_EXTRA_NODES_FILE": str(extra),
                                                     "POOL_QUEUE_PATH": str(tmp_path / "q" / "pool.queue")})
    assert res.stdout.split() == ["pool41", "pool1", "pool2.lan"], res.stdout
    assert "ignoring node name" in res.stderr


# ------------------------------------------------------------------ safe_copy_into.py

def _copy(anchor: Path, dest: str, *srcs: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", str(SAFE_COPY), "--anchor", str(anchor), "--dest", dest,
                           *map(str, srcs)], capture_output=True, text=True, timeout=60)


def test_safe_copy_refuses_every_planted_symlink(tmp_path: Path) -> None:
    anchor = tmp_path / "projects"
    raw = anchor / "ws" / "research" / "findings" / "raw"
    raw.mkdir(parents=True)
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.json").write_text("victim\n")
    stage = tmp_path / "stage"
    (stage / "sub").mkdir(parents=True)
    (stage / "a.json").write_text("new-a\n")
    (stage / "keep.json").write_text("new-keep\n")
    (stage / "sub" / "b.json").write_text("new-b\n")
    (raw / "sub").symlink_to(victim)                   # a symlinked subdirectory
    (raw / "keep.json").symlink_to(victim / "keep.json")  # a symlinked file
    res = _copy(anchor, "ws/research/findings/raw", stage)
    assert res.returncode == 0, res.stderr
    assert (victim / "keep.json").read_text() == "victim\n" and not (victim / "b.json").exists()
    assert (raw / "a.json").read_text() == "new-a\n"
    assert "copied 1," in res.stdout and "refused 2" in res.stdout
    # an ancestor of the destination replaced by a symlink: nothing is written anywhere
    (anchor / "ws" / "research" / "findings").rename(tmp_path / "moved")
    (anchor / "ws" / "research" / "findings").symlink_to(victim)
    res = _copy(anchor, "ws/research/findings/raw", stage)
    assert res.returncode == 2 and "without following a symlink" in res.stderr
    assert sorted(p.name for p in victim.iterdir()) == ["keep.json"]


def test_safe_copy_is_newer_wins_keeps_mtime_strips_x_and_replaces_hard_links(tmp_path: Path) -> None:
    anchor = tmp_path / "projects"
    raw = anchor / "ws" / "raw"
    raw.mkdir(parents=True)
    stage = tmp_path / "stage"
    stage.mkdir()
    for name, text in (("old.json", "stage-old\n"), ("new.json", "stage-new\n"), ("x.json", "x\n"),
                       ("linked.json", "stage-linked\n")):
        (stage / name).write_text(text)
    t = time.time()
    os.utime(stage / "old.json", (t - 100, t - 100))
    os.utime(stage / "new.json", (t - 100, t - 100))
    (stage / "x.json").chmod(0o755)
    (raw / "old.json").write_text("workspace-newer\n")          # newer on the destination: kept
    (raw / "new.json").write_text("workspace-older\n")
    os.utime(raw / "new.json", (t - 500, t - 500))             # older on the destination: replaced
    outside = tmp_path / "outside.json"
    outside.write_text("outside\n")
    os.utime(outside, (t - 500, t - 500))
    os.link(outside, raw / "linked.json")                      # a hard link to a file outside
    res = _copy(anchor, "ws/raw", stage)
    assert res.returncode == 0, res.stderr
    assert (raw / "old.json").read_text() == "workspace-newer\n"
    assert (raw / "new.json").read_text() == "stage-new\n"
    assert int(os.stat(raw / "new.json").st_mtime) == int(t - 100)
    assert not os.stat(raw / "x.json").st_mode & 0o111
    assert outside.read_text() == "outside\n" and (raw / "linked.json").read_text() == "stage-linked\n"
    assert "copied 3, unchanged 1, refused 0" in res.stdout


def test_safe_copy_never_clobbers_a_settled_result(tmp_path: Path) -> None:
    """2026-10-10: a target older than KEEP_AGE_H with different content is kept (a re-run of a landed cell twice
    overwrote committed results); an identical one is unchanged; a fresh, still-refreshing one stays newer-wins."""
    anchor = tmp_path / "projects"
    raw = anchor / "ws" / "raw"
    raw.mkdir(parents=True)
    stage = tmp_path / "stage"
    stage.mkdir()
    t = time.time()
    for name, ws_text, st_text, ws_age in (("settled.json", "committed\n", "rerun\n", 2 * 86400),
                                           ("same.json", "same\n", "same\n", 2 * 86400),
                                           ("fresh.json", "partial\n", "final\n", 600)):
        (raw / name).write_text(ws_text)
        os.utime(raw / name, (t - ws_age, t - ws_age))
        (stage / name).write_text(st_text)
    res = _copy(anchor, "ws/raw", stage)
    assert res.returncode == 0, res.stderr
    assert (raw / "settled.json").read_text() == "committed\n"
    assert (raw / "fresh.json").read_text() == "final\n"
    assert "KEPT settled 'settled.json'" in res.stderr
    assert "copied 1, unchanged 1, refused 0, kept 1" in res.stdout


# ------------------------------------------------------------------ pool_sync.sh against a local fake node

def _fake_node(tmp_path: Path) -> tuple[Path, Path]:
    """A stub ssh that runs the remote command locally with HOME=<node home>; returns (bin dir, node home)."""
    home = tmp_path / "nodehome"
    home.mkdir()
    bin_dir = tmp_path / "bin"
    _exe(bin_dir / "ssh", f"""while [ $# -gt 0 ]; do
  case "$1" in -o|-F) shift 2 ;; -*) shift ;; *) break ;; esac
done
shift   # the host
cd "{home}" && HOME="{home}" exec bash -c "$*"
""")
    return bin_dir, home


def _node_file(home: Path, rel: str, text: str) -> None:
    p = home / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _sync_env(bin_dir: Path, tmp_path: Path, ws: Path, node: str = "pool41") -> dict[str, str]:
    return {"PATH": f"{bin_dir}:{os.environ['PATH']}", "POOL_NODES": node, "SIM_WORKSPACE": str(ws),
            "POOL_SYNC_STAGE_DIR": str(tmp_path / "stage"), "POOL_SSH_CONFIG": str(tmp_path / "none")}


def test_runtime_pull_lands_through_the_stage_and_ignores_a_planted_symlink(tmp_path: Path) -> None:
    bin_dir, home = _fake_node(tmp_path)
    _node_file(home, "derisk-pool/sim/research/findings/raw/main.json", "main\n")
    _node_file(home, "derisk-pool/sim/research/findings/raw/lane/deep.json", "deep\n")
    _node_file(home, "derisk-pool/revisions/abc1234/research/findings/raw/rev.json", "rev\n")
    ws = tmp_path / "projects" / "neural-simulator"
    raw = ws / "research" / "findings" / "raw"
    raw.mkdir(parents=True)
    victim = tmp_path / "victim"
    victim.mkdir()
    (raw / "lane").symlink_to(victim)
    res = _run(POOL_SYNC, [], _sync_env(bin_dir, tmp_path, ws))
    assert res.returncode == 0, res.stderr
    assert (raw / "main.json").read_text() == "main\n" and (raw / "rev.json").read_text() == "rev\n"
    assert list(victim.iterdir()) == [], "the pull followed a symlink planted in the destination"
    assert "REFUSED" in res.stdout and (tmp_path / "stage" / "pool41" / "main" / "lane" / "deep.json").exists()


def test_runtime_pull_writes_nothing_when_the_destination_path_is_a_symlink(tmp_path: Path) -> None:
    bin_dir, home = _fake_node(tmp_path)
    _node_file(home, "derisk-pool/sim/research/findings/raw/main.json", "main\n")
    ws = tmp_path / "projects" / "neural-simulator"
    (ws / "research").mkdir(parents=True)
    victim = tmp_path / "victim"
    victim.mkdir()
    (ws / "research" / "findings").symlink_to(victim)
    (victim / "raw").mkdir()
    res = _run(POOL_SYNC, [], _sync_env(bin_dir, tmp_path, ws))
    assert list((victim / "raw").iterdir()) == []
    assert "NOT copied into the workspace" in res.stdout


def test_a_reachable_host_without_a_main_results_dir_still_has_its_revisions_pulled(tmp_path: Path) -> None:
    # The server's first workstation pull logged "workstation: UNREACHABLE (skipped)": ssh worked, but the
    # workstation has no ~/derisk-pool/sim/research/findings/raw, rsync exited 23, and the revisions were skipped too.
    bin_dir, home = _fake_node(tmp_path)
    _node_file(home, "derisk-pool/revisions/abc1234-1759270000/research/findings/raw/gpu.json", "gpu\n")
    ws = tmp_path / "projects" / "neural-simulator"
    raw = ws / "research" / "findings" / "raw"
    raw.mkdir(parents=True)
    res = _run(POOL_SYNC, [], _sync_env(bin_dir, tmp_path, ws, node="workstation"))
    assert "UNREACHABLE" not in res.stdout, res.stdout
    assert "reachable, so its revisions are still pulled" in res.stdout
    assert (raw / "gpu.json").read_text() == "gpu\n"


def test_revisions_are_listed_even_when_the_login_shell_wraps_ls(tmp_path: Path) -> None:
    # The workstation's login shell is fish with `ls` wrapped in a long, coloured listing; the old bare `ls -d` there
    # returned permission/owner/date tokens instead of paths, so none of its GPU-job revisions could be pulled.
    bin_dir, home = _fake_node(tmp_path)
    bashenv = tmp_path / "bashenv"
    bashenv.write_text("ls() { command ls -l --color=always \"$@\"; }\n")
    _node_file(home, "derisk-pool/sim/research/findings/raw/.keep.json", "{}\n")
    _node_file(home, "derisk-pool/revisions/abc1234-1759270000/research/findings/raw/gpu.json", "gpu\n")
    ws = tmp_path / "projects" / "neural-simulator"
    raw = ws / "research" / "findings" / "raw"
    raw.mkdir(parents=True)
    res = _run(POOL_SYNC, [], {**_sync_env(bin_dir, tmp_path, ws, node="workstation"), "BASH_ENV": str(bashenv)})
    assert "ignoring unexpected revision path" not in res.stdout, res.stdout
    assert (raw / "gpu.json").read_text() == "gpu\n"


def test_runtime_dry_run_writes_nothing(tmp_path: Path) -> None:
    bin_dir, home = _fake_node(tmp_path)
    _node_file(home, "derisk-pool/sim/research/findings/raw/main.json", "main\n")
    ws = tmp_path / "projects" / "neural-simulator"
    raw = ws / "research" / "findings" / "raw"
    raw.mkdir(parents=True)
    env = _sync_env(bin_dir, tmp_path, ws)
    del env["POOL_SYNC_STAGE_DIR"]                   # a dry run needs no staging dir
    res = _run(POOL_SYNC, ["--dry-run"], env)
    assert res.returncode == 0, res.stderr
    assert "would pull 1 file(s)" in res.stdout and list(raw.iterdir()) == []
    assert not (tmp_path / "stage").exists()


RRSYNC = Path("/usr/bin/rrsync")


def _rrsync_node(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A stub ssh that behaves like sshd with `command="rrsync -ro ~/derisk-pool"`: the REAL rrsync judges every remote
    command (a shell command is refused). Returns (bin dir, node home, command log)."""
    home = tmp_path / "nodehome"
    (home / "derisk-pool").mkdir(parents=True)
    log = tmp_path / "remote_commands.log"
    bin_dir = tmp_path / "bin"
    _exe(bin_dir / "ssh", f"""while [ $# -gt 0 ]; do
  case "$1" in -o|-F) shift 2 ;; -*) shift ;; *) break ;; esac
done
shift   # the host
printf '%s\\n' "$*" >> "{log}"
cd "{home}" && HOME="{home}" SSH_ORIGINAL_COMMAND="$*" exec python3 {RRSYNC} -ro "{home}/derisk-pool"
""")
    return bin_dir, home, log


@pytest.mark.skipif(not RRSYNC.exists(), reason="needs rsync's rrsync script")
def test_a_read_only_rrsync_host_still_has_its_results_and_revisions_pulled(tmp_path: Path) -> None:
    # sim's key on the workstation is restricted to `rrsync -ro ~/derisk-pool` (no shell): the pull must work through
    # it, listing revisions with rsync itself and skipping a revision whose job has written nothing yet.
    bin_dir, home, log = _rrsync_node(tmp_path)
    _node_file(home, "derisk-pool/sim/research/findings/raw/.keep.json", "{}\n")
    _node_file(home, "derisk-pool/revisions/abc1234-1759270000/research/findings/raw/gpu.json", "gpu\n")
    (home / "derisk-pool" / "revisions" / "def5678-1759270001" / "tools").mkdir(parents=True)   # no results yet
    ws = tmp_path / "projects" / "neural-simulator"
    raw = ws / "research" / "findings" / "raw"
    raw.mkdir(parents=True)
    env = {**_sync_env(bin_dir, tmp_path, ws, node="workstation"), "POOL_SYNC_RRSYNC_NODES": "workstation"}
    res = _run(POOL_SYNC, ["--strict"], env)
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert (raw / "gpu.json").read_text() == "gpu\n"
    cmds = log.read_text().splitlines()
    assert cmds and all(c.startswith("rsync --server --sender ") for c in cmds), cmds
    # The forced command refuses a shell, so without the mode nothing is pulled at all (fresh stage and workspace).
    shell = subprocess.run([str(bin_dir / "ssh"), "-o", "BatchMode=yes", "workstation", "sh -c 'id'"],
                           capture_output=True, text=True, timeout=30)
    assert shell.returncode != 0 and "uid=" not in shell.stdout
    raw2 = tmp_path / "projects2" / "neural-simulator" / "research" / "findings" / "raw"
    raw2.mkdir(parents=True)
    res = _run(POOL_SYNC, [], {**env, "POOL_SYNC_RRSYNC_NODES": "", "SIM_WORKSPACE": str(raw2.parents[2]),
                               "POOL_SYNC_STAGE_DIR": str(tmp_path / "stage2")})
    assert "0 file(s) pulled" in res.stdout and list(raw2.iterdir()) == [], res.stdout


def test_runtime_pull_skips_a_file_over_the_size_cap(tmp_path: Path) -> None:
    bin_dir, home = _fake_node(tmp_path)
    _node_file(home, "derisk-pool/sim/research/findings/raw/small.json", "{}\n")
    _node_file(home, "derisk-pool/sim/research/findings/raw/huge.npz", "x" * 5000)
    ws = tmp_path / "projects" / "neural-simulator"
    raw = ws / "research" / "findings" / "raw"
    raw.mkdir(parents=True)
    res = _run(POOL_SYNC, [], {**_sync_env(bin_dir, tmp_path, ws), "POOL_SYNC_MAX_FILE_SIZE": "1K"})
    assert res.returncode == 0, res.stderr
    assert (raw / "small.json").exists() and not (raw / "huge.npz").exists()
    assert not (tmp_path / "stage" / "pool41" / "main" / "huge.npz").exists()


def test_runtime_pull_stops_and_fails_loudly_at_the_stage_cap(tmp_path: Path) -> None:
    # The stage is on the server's root fs and the nodes are agent-controlled: past its cap no pull starts, and the
    # run exits 1 (the unit fails, ntfy fires) whatever --strict says.
    bin_dir, home = _fake_node(tmp_path)
    _node_file(home, "derisk-pool/sim/research/findings/raw/main.json", "main\n")
    ws = tmp_path / "projects" / "neural-simulator"
    raw = ws / "research" / "findings" / "raw"
    raw.mkdir(parents=True)
    filler = tmp_path / "stage" / "pool41" / "main" / "earlier.npz"
    filler.parent.mkdir(parents=True)
    filler.write_bytes(b"\0" * (2 * 1024 * 1024))
    res = _run(POOL_SYNC, [], {**_sync_env(bin_dir, tmp_path, ws), "POOL_SYNC_STAGE_MAX_MB": "1"})
    assert res.returncode == 1 and "no further pull" in res.stdout, (res.stdout, res.stderr)
    assert not (raw / "main.json").exists(), "a pull started although the stage was over its cap"
    assert (raw / "earlier.npz").exists(), "what was already staged is still copied into the workspace"
    res = _run(POOL_SYNC, [], {**_sync_env(bin_dir, tmp_path, ws), "POOL_SYNC_STAGE_MAX_MB": "64"})
    assert res.returncode == 0 and (raw / "main.json").read_text() == "main\n", res.stdout


# ------------------------------------------------------------------ lane_starvation reads pool.running

def test_lane_starvation_counts_recent_pool_running_records(tmp_path: Path, monkeypatch) -> None:
    import importlib.util   # loaded by path: tools/gates on sys.path would shadow real packages (gates/coverage.py)
    spec = importlib.util.spec_from_file_location("lane_starvation_under_test", ROOT / "tools" / "gates" / "lane_starvation.py")
    ls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ls)
    qd = tmp_path / "research" / "queue"
    qd.mkdir(parents=True)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    old = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - 7200))
    (qd / "pool.running").write_text(f"{old}\tpool41\tpython -m research.runners.old_job\n"
                                     f"{now}\tpool42\tpython -m research.runners.fresh_job  #checked:r\n")
    monkeypatch.setattr(ls, "_queue_dir", lambda: str(qd))
    got = ls._work_lines()
    assert any("fresh_job" in l for l in got) and not any("old_job" in l for l in got)
