"""Subprocess-level tests for tools/aws_pool_node.sh -- treating ONE AWS r7i.4xlarge as an extra mini-PC-pool
node (2026-09-23 build). NO real AWS instance is ever launched here (the build-lane rule for this feature); a
stubbed `aws`/`ssh`/`rsync` on PATH stands in, mirroring tests/test_aws_budget_guard_workflow.py's approach.

Covers exactly the three things this feature's own build-lane checklist scoped for testing without a real
launch: (1) the ssh-config Host-block helpers `up`/`down` rely on, (2) `down`'s ordering (unregister -> pull
results -> terminate), and (3) that a budget refusal short-circuits `up` before any `run-instances` call.
"""
from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "aws_pool_node.sh"


def _run(args, bin_dir=None, env=None, tmp_path=None):
    full_env = dict(os.environ)
    if bin_dir is not None:
        full_env["PATH"] = f"{bin_dir}:{full_env.get('PATH', '')}"
    # Always isolate the spend ledger + budget log, even for callers that pass no tmp_path: several tests here
    # call _run(...) without one, and on 2026-09-24/25 their stub instance "i-existing" was written into the
    # PRODUCTION ledger (research/queue/.aws_spend_ledger.jsonl), adding ~$14.8 of phantom spend to "today".
    iso = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp(prefix="aws_pool_node_test_"))
    full_env.setdefault("AWS_BUDGET_LOG", str(iso / "aws_budget.log"))
    full_env.setdefault("AWS_SPEND_LEDGER", str(iso / "aws_spend_ledger.jsonl"))
    if env:
        full_env.update(env)
    return subprocess.run(["bash", str(SCRIPT), *args], cwd=ROOT, env=full_env,
                           capture_output=True, text=True, timeout=30)


# ------------------------------------------------------------------------------- ssh Host-block helpers

def test_write_host_block_appends_a_new_entry(tmp_path):
    cfg = tmp_path / "ssh_config"
    cfg.write_text("Include ~/.ssh/config\n")
    res = _run(["--write-host-block", str(cfg), "pool1", "1.2.3.4", "/tmp/key.pem"])
    assert res.returncode == 0, res.stderr
    text = cfg.read_text()
    assert "Host pool1" in text
    assert "HostName 1.2.3.4" in text
    assert "IdentityFile /tmp/key.pem" in text
    assert "User ubuntu" in text


def test_write_host_block_replaces_same_alias_without_touching_others(tmp_path):
    cfg = tmp_path / "ssh_config"
    cfg.write_text("Include ~/.ssh/config\n")
    _run(["--write-host-block", str(cfg), "pool1", "1.1.1.1", "/tmp/k1.pem"])
    _run(["--write-host-block", str(cfg), "pool2", "2.2.2.2", "/tmp/k2.pem"])
    res = _run(["--write-host-block", str(cfg), "pool1", "9.9.9.9", "/tmp/k1.pem"])
    assert res.returncode == 0, res.stderr
    text = cfg.read_text()
    assert text.count("Host pool1") == 1   # replaced, not duplicated
    assert "HostName 9.9.9.9" in text
    assert "1.1.1.1" not in text            # stale IP is gone
    assert "Host pool2" in text and "HostName 2.2.2.2" in text   # untouched


def test_remove_host_block_drops_only_the_named_alias(tmp_path):
    cfg = tmp_path / "ssh_config"
    cfg.write_text("Include ~/.ssh/config\n")
    _run(["--write-host-block", str(cfg), "pool1", "1.1.1.1", "/tmp/k1.pem"])
    _run(["--write-host-block", str(cfg), "pool2", "2.2.2.2", "/tmp/k2.pem"])
    res = _run(["--remove-host-block", str(cfg), "pool1"])
    assert res.returncode == 0, res.stderr
    text = cfg.read_text()
    assert "Host pool1" not in text
    assert "Host pool2" in text and "HostName 2.2.2.2" in text


def test_remove_host_block_on_missing_file_is_a_noop(tmp_path):
    res = _run(["--remove-host-block", str(tmp_path / "does-not-exist"), "pool1"])
    assert res.returncode == 0, res.stderr


# ------------------------------------------------------------------------------- _write_host_block: atomicity

def test_write_host_block_is_serialized_by_flock_against_a_concurrent_holder(tmp_path):
    # MEDIUM (2026-09-25 review): "_write_host_block not atomic" -- it must hold `<file>.lock` for its ENTIRE
    # read-modify-write, so anything else holding that same lock makes it WAIT rather than interleave. Hold the
    # lock externally (via the real `flock` CLI) for a measured duration, then time how long
    # `--write-host-block` takes to return: it must take AT LEAST that long (proving it genuinely waited for the
    # lock), not return near-instantly (which the OLD code -- no flock call anywhere -- always did).
    cfg = tmp_path / "ssh_config"
    cfg.write_text("Include ~/.ssh/config\n")
    lock = tmp_path / "ssh_config.lock"
    hold_s = 2.0
    holder = subprocess.Popen(["flock", str(lock), "sleep", str(hold_s)])
    try:
        time.sleep(0.4)   # give the holder a head start so it has genuinely acquired the lock first
        t0 = time.monotonic()
        res = _run(["--write-host-block", str(cfg), "pool1", "1.2.3.4", "/tmp/key.pem"])
        elapsed = time.monotonic() - t0
    finally:
        holder.wait(timeout=10)
    assert res.returncode == 0, res.stderr
    assert elapsed >= 1.0, (
        f"--write-host-block returned after only {elapsed:.2f}s while an external holder had the SAME lock "
        f"for {hold_s}s -- it did not actually wait for the lock (not serialized)")
    assert "Host pool1" in cfg.read_text()
    assert "HostName 1.2.3.4" in cfg.read_text()


def test_remove_host_block_is_serialized_by_flock_against_a_concurrent_holder(tmp_path):
    # Same guarantee, for the removal path -- the two must share the SAME lock file so a write and a remove
    # against the same config can never race each other either.
    cfg = tmp_path / "ssh_config"
    cfg.write_text("Include ~/.ssh/config\nHost pool1\n  HostName 1.2.3.4\n")
    lock = tmp_path / "ssh_config.lock"
    hold_s = 2.0
    holder = subprocess.Popen(["flock", str(lock), "sleep", str(hold_s)])
    try:
        time.sleep(0.4)
        t0 = time.monotonic()
        res = _run(["--remove-host-block", str(cfg), "pool1"])
        elapsed = time.monotonic() - t0
    finally:
        holder.wait(timeout=10)
    assert res.returncode == 0, res.stderr
    assert elapsed >= 1.0, f"--remove-host-block did not wait for the lock (returned after {elapsed:.2f}s)"
    assert "Host pool1" not in cfg.read_text()


def test_write_host_block_concurrent_writers_never_lose_or_duplicate_a_block(tmp_path):
    # MEDIUM: without the flock+single-mv fix, concurrent writers could each read the SAME pre-edit content and
    # race their own `mv`s -- the review's own repro found a DUPLICATE block (stale ip listed FIRST, so ssh
    # picks it) in 27/30 trials, and a concurrent READER saw the block MISSING entirely in 10/1482 snapshots.
    # Fire many real concurrent writer PROCESSES, each adding its own distinct alias to the SAME file, and
    # require every single one to land exactly once, fully formed, in the final content.
    cfg = tmp_path / "ssh_config"
    cfg.write_text("Include ~/.ssh/config\n")
    n = 15
    procs = [
        subprocess.Popen(["bash", str(SCRIPT), "--write-host-block", str(cfg), f"node{i}",
                           f"10.0.0.{i}", f"/tmp/k{i}.pem"], cwd=ROOT, env=dict(os.environ))
        for i in range(n)
    ]
    for p in procs:
        assert p.wait(timeout=30) == 0

    lines = cfg.read_text().splitlines()
    for i in range(n):
        # Exact LINE match (not substring): "Host node1" is a substring of "Host node10".."Host node14" too.
        assert lines.count(f"Host node{i}") == 1, f"node{i}'s block is missing or duplicated:\n{lines}"
        assert f"  HostName 10.0.0.{i}" in lines, f"node{i}'s block is malformed/incomplete:\n{lines}"


# --------------------------------------------------------------------------------------------- down ordering

_STUB = r"""#!/usr/bin/env bash
echo "{tag} $*" >> "{log}"
{body}
"""


def _make_down_stub_bin(tmp_path):
    """One shared, order-preserving log for aws/ssh/rsync: `down` calls them strictly sequentially in one
    process, so line order in a single file IS call order -- the load-bearing assertion below is that a
    pool_sync (ssh/rsync) line appears BEFORE the `ec2 terminate-instances` line."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "combined.log"
    log.write_text("")

    aws_stub = bin_dir / "aws"
    aws_stub.write_text(_STUB.format(tag="AWS", log=log, body='echo ok\nexit 0'))
    aws_stub.chmod(aws_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    ssh_stub = bin_dir / "ssh"
    # Answers the drain loop's pgrep-style reachability probe with "0" (reachable, idle) -- a plain `exit 0`
    # with no stdout would read as UNREACHABLE/UNKNOWN under the fix-round-#2 drain guard, hanging any test
    # whose ssh_config actually has a Host entry for the node (tests that use a non-existent config skip the
    # whole drain block and never notice either way).
    ssh_stub.write_text(_STUB.format(tag="SSH", log=log, body='case "$*" in *pgrep*) echo 0 ;; esac\nexit 0'))
    ssh_stub.chmod(ssh_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    rsync_stub = bin_dir / "rsync"
    rsync_stub.write_text(_STUB.format(tag="RSYNC", log=log, body='exit 0'))
    rsync_stub.chmod(rsync_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    return bin_dir, log


def _write_state(tmp_path, instance="i-aaa", region="us-east-1", key=None, sg="sg-aaa"):
    key = key or (tmp_path / "key.pem")
    Path(key).write_text("fake key\n")
    state = tmp_path / ".aws_testnode"
    state.write_text(f"instance={instance}\nregion={region}\nkey={key}\nsg={sg}\n")
    return state


def test_down_unregisters_before_pulling_results_before_terminating(tmp_path):
    bin_dir, log = _make_down_stub_bin(tmp_path)
    state = _write_state(tmp_path)
    extra = tmp_path / "extra_nodes"
    extra.write_text("otherpool\ntestnode\n")

    res = _run(["down", "testnode"], bin_dir=bin_dir, tmp_path=tmp_path, env={
        "AWS_POOL_NODE_STATE_FILE": str(state),
        "POOL_EXTRA_NODES_FILE": str(extra),
        "POOL_SSH_CONFIG": str(tmp_path / "no-such-config"),
    })
    assert res.returncode == 0, res.stderr

    # (1) unregistered -- the node the dispatcher will never target again.
    remaining = extra.read_text().split()
    assert "testnode" not in remaining
    assert "otherpool" in remaining   # untouched sibling entry

    # (2) + (3) ORDER: a pool_sync ssh/rsync call happened, and it happened BEFORE terminate-instances.
    lines = log.read_text().splitlines()
    sync_idx = next((i for i, ln in enumerate(lines) if ln.startswith("SSH ") or ln.startswith("RSYNC ")), None)
    terminate_idx = next((i for i, ln in enumerate(lines) if "ec2 terminate-instances" in ln), None)
    assert sync_idx is not None, f"no pool_sync ssh/rsync call observed: {lines}"
    assert terminate_idx is not None, f"no terminate-instances call observed: {lines}"
    assert sync_idx < terminate_idx, f"terminate happened before/without a prior pool_sync: {lines}"

    # delete-security-group also happened.
    assert any("ec2 delete-security-group" in ln and "sg-aaa" in ln for ln in lines), lines

    # (4) state file marked torn down, never deleted.
    assert state.exists()
    assert state.read_text().startswith("# TORN DOWN")
    assert "instance=i-aaa" in state.read_text()   # the original record survives underneath


def test_down_is_idempotent_and_never_re_terminates(tmp_path):
    bin_dir, log = _make_down_stub_bin(tmp_path)
    state = _write_state(tmp_path)
    extra = tmp_path / "extra_nodes"
    extra.write_text("testnode\n")
    env = {
        "AWS_POOL_NODE_STATE_FILE": str(state),
        "POOL_EXTRA_NODES_FILE": str(extra),
        "POOL_SSH_CONFIG": str(tmp_path / "no-such-config"),
    }
    res1 = _run(["down", "testnode"], bin_dir=bin_dir, tmp_path=tmp_path, env=env)
    assert res1.returncode == 0, res1.stderr
    n_terminate_calls_1 = log.read_text().count("ec2 terminate-instances")
    assert n_terminate_calls_1 == 1

    res2 = _run(["down", "testnode"], bin_dir=bin_dir, tmp_path=tmp_path, env=env)
    assert res2.returncode == 0, res2.stderr
    assert log.read_text().count("ec2 terminate-instances") == 1   # NOT called again
    assert "already torn down" in (res2.stdout + res2.stderr)


def _make_down_stub_bin_with_running_runners(tmp_path, n_running=3):
    """Like _make_down_stub_bin, but the ssh stub ALWAYS reports `n_running` runners (matching
    _running_runners's pgrep-style probe) -- so `down`'s drain wait / refuse-while-running guard has something
    to refuse against, deterministically, with no real node."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "combined.log"
    log.write_text("")

    aws_stub = bin_dir / "aws"
    aws_stub.write_text(_STUB.format(tag="AWS", log=log, body='echo ok\nexit 0'))
    aws_stub.chmod(aws_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    ssh_stub = bin_dir / "ssh"
    ssh_stub.write_text(f"""#!/usr/bin/env bash
echo "SSH $*" >> "{log}"
case "$*" in
  *pgrep*) echo {n_running} ;;
esac
exit 0
""")
    ssh_stub.chmod(ssh_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    rsync_stub = bin_dir / "rsync"
    rsync_stub.write_text(_STUB.format(tag="RSYNC", log=log, body='exit 0'))
    rsync_stub.chmod(rsync_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    return bin_dir, log


def test_down_refuses_to_terminate_while_runners_are_still_running(tmp_path):
    # HIGH (2026-09-23 fix round): `down` used to unregister then IMMEDIATELY pool_sync + terminate, with NO
    # wait for (or check of) an in-flight runner -- the root volume is DeleteOnTermination=true, so a job killed
    # mid-run loses its output with no requeue. It must now DRAIN first and REFUSE to terminate if a runner is
    # still running after the (bounded) drain wait, unless --force.
    bin_dir, log = _make_down_stub_bin_with_running_runners(tmp_path, n_running=3)
    state = _write_state(tmp_path)
    ssh_config = tmp_path / "ssh_config"
    ssh_config.write_text("Include ~/.ssh/config\nHost testnode\n  HostName 1.2.3.4\n")
    env = {
        "AWS_POOL_NODE_STATE_FILE": str(state),
        "POOL_EXTRA_NODES_FILE": str(tmp_path / "extra_nodes"),
        "POOL_SSH_CONFIG": str(ssh_config),
        "AWS_POOL_DRAIN_TIMEOUT_S": "1",   # bounded -- this test must not hang for the real 30 min default
        "AWS_POOL_DRAIN_POLL_S": "0",
    }
    (tmp_path / "extra_nodes").write_text("testnode\n")

    res = _run(["down", "testnode"], bin_dir=bin_dir, tmp_path=tmp_path, env=env)
    assert res.returncode == 1
    assert "Refusing to terminate" in (res.stdout + res.stderr)
    assert "ec2 terminate-instances" not in log.read_text()   # never reached -- refused before it
    assert state.read_text().startswith("instance=")          # NOT marked torn down -- teardown never happened


def test_down_force_terminates_anyway_despite_running_runners(tmp_path):
    # --force is the documented override for the guard above.
    bin_dir, log = _make_down_stub_bin_with_running_runners(tmp_path, n_running=2)
    state = _write_state(tmp_path)
    ssh_config = tmp_path / "ssh_config"
    ssh_config.write_text("Include ~/.ssh/config\nHost testnode\n  HostName 1.2.3.4\n")
    env = {
        "AWS_POOL_NODE_STATE_FILE": str(state),
        "POOL_EXTRA_NODES_FILE": str(tmp_path / "extra_nodes"),
        "POOL_SSH_CONFIG": str(ssh_config),
        "AWS_POOL_DRAIN_TIMEOUT_S": "1",
        "AWS_POOL_DRAIN_POLL_S": "0",
    }
    (tmp_path / "extra_nodes").write_text("testnode\n")

    res = _run(["down", "testnode", "--force"], bin_dir=bin_dir, tmp_path=tmp_path, env=env)
    assert res.returncode == 0, res.stderr
    assert "ec2 terminate-instances" in log.read_text()
    assert state.read_text().startswith("# TORN DOWN")


def _make_down_stub_bin_unreachable(tmp_path):
    """ssh AND rsync both exit 255 unconditionally (an unreachable node), aws behaves normally otherwise (so the
    describe-instances EC2-state probe reads some generic non-'stopped'/non-terminated state and does not
    short-circuit) -- reproduces the re-review's literal repro of HIGH #4."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "combined.log"
    log.write_text("")

    aws_stub = bin_dir / "aws"
    aws_stub.write_text(_STUB.format(tag="AWS", log=log, body='echo ok\nexit 0'))
    aws_stub.chmod(aws_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    for name in ("ssh", "rsync", "scp"):
        stub = bin_dir / name
        stub.write_text(f"""#!/usr/bin/env bash
echo "{name.upper()} $*" >> "{log}"
exit 255
""")
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir, log


def test_down_refuses_to_terminate_an_unreachable_node_without_force(tmp_path):
    # THE ACTUAL DEFECT (re-review, HIGH #4, "prior HIGH #4 only partly resolved"): pool_sync.sh's PLAIN/default
    # mode always exits 0 (`|| { echo UNREACHABLE; continue; }`, then unconditional success) -- so `down`'s old
    # SYNC_OK check was structurally unable to ever read failure for "unreachable ... or an instance already
    # STOPPED", which the guard's own comment claims to refuse on. Repro: ssh/rsync both fail (255); `down` must
    # now refuse (via pool_sync's new --strict mode) rather than terminate an unsynced node.
    bin_dir, log = _make_down_stub_bin_unreachable(tmp_path)
    state = _write_state(tmp_path)
    ssh_config = tmp_path / "ssh_config"
    ssh_config.write_text("Include ~/.ssh/config\nHost testnode\n  HostName 1.2.3.4\n")
    env = {
        "AWS_POOL_NODE_STATE_FILE": str(state),
        "POOL_EXTRA_NODES_FILE": str(tmp_path / "extra_nodes"),
        "POOL_SSH_CONFIG": str(ssh_config),
        "AWS_POOL_DRAIN_TIMEOUT_S": "1",
        "AWS_POOL_DRAIN_POLL_S": "0",
    }
    (tmp_path / "extra_nodes").write_text("testnode\n")

    res = _run(["down", "testnode"], bin_dir=bin_dir, tmp_path=tmp_path, env=env)
    assert res.returncode == 1
    assert "Refusing to terminate" in (res.stdout + res.stderr)
    assert "ec2 terminate-instances" not in log.read_text()   # never reached -- refused before it
    assert state.read_text().startswith("instance=")          # NOT marked torn down


def test_down_force_terminates_an_unreachable_node_anyway(tmp_path):
    bin_dir, log = _make_down_stub_bin_unreachable(tmp_path)
    state = _write_state(tmp_path)
    ssh_config = tmp_path / "ssh_config"
    ssh_config.write_text("Include ~/.ssh/config\nHost testnode\n  HostName 1.2.3.4\n")
    env = {
        "AWS_POOL_NODE_STATE_FILE": str(state),
        "POOL_EXTRA_NODES_FILE": str(tmp_path / "extra_nodes"),
        "POOL_SSH_CONFIG": str(ssh_config),
        "AWS_POOL_DRAIN_TIMEOUT_S": "1",
        "AWS_POOL_DRAIN_POLL_S": "0",
    }
    (tmp_path / "extra_nodes").write_text("testnode\n")

    res = _run(["down", "testnode", "--force"], bin_dir=bin_dir, tmp_path=tmp_path, env=env)
    assert res.returncode == 0, res.stderr
    assert "ec2 terminate-instances" in log.read_text()
    assert state.read_text().startswith("# TORN DOWN")


def _make_down_stub_bin_stopped_then_reachable(tmp_path):
    """describe-instances reports 'stopped' on the FIRST call and 'running' after start-instances; ssh becomes
    reachable once a marker file (written by the stubbed start-instances call) exists -- simulates a node that
    aws_idle_stop.sh had stopped, coming back up after `down` starts it."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "combined.log"
    log.write_text("")
    started_marker = tmp_path / "started"

    aws_stub = bin_dir / "aws"
    aws_stub.write_text(f"""#!/usr/bin/env bash
echo "AWS $*" >> "{log}"
case "$*" in
  *"ec2 start-instances"*) touch "{started_marker}"; echo ok; exit 0 ;;
  *"describe-instances"*"State.Name"*)
    if [ -f "{started_marker}" ]; then echo running; else echo stopped; fi
    exit 0 ;;
esac
echo ok
exit 0
""")
    aws_stub.chmod(aws_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    ssh_stub = bin_dir / "ssh"
    ssh_stub.write_text(f"""#!/usr/bin/env bash
echo "SSH $*" >> "{log}"
[ -f "{started_marker}" ] || exit 255
case "$*" in *pgrep*) echo 0 ;; esac
exit 0
""")
    ssh_stub.chmod(ssh_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    for name in ("rsync", "scp"):
        stub = bin_dir / name
        stub.write_text(f"""#!/usr/bin/env bash
echo "{name.upper()} $*" >> "{log}"
[ -f "{started_marker}" ] || exit 255
exit 0
""")
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir, log, started_marker


def test_down_starts_a_stopped_instance_syncs_then_terminates(tmp_path):
    # THE FIX (fix round #2, LOW/design item): "a STOPPED instance is started, synced, then terminated" -- not
    # just refused as unreachable-forever.
    bin_dir, log, started_marker = _make_down_stub_bin_stopped_then_reachable(tmp_path)
    state = _write_state(tmp_path)
    ssh_config = tmp_path / "ssh_config"
    ssh_config.write_text("Include ~/.ssh/config\nHost testnode\n  HostName 1.2.3.4\n")
    env = {
        "AWS_POOL_NODE_STATE_FILE": str(state),
        "POOL_EXTRA_NODES_FILE": str(tmp_path / "extra_nodes"),
        "POOL_SSH_CONFIG": str(ssh_config),
        "AWS_POOL_START_TIMEOUT_S": "5",
        "AWS_POOL_START_POLL_S": "0",
    }
    (tmp_path / "extra_nodes").write_text("testnode\n")

    res = _run(["down", "testnode"], bin_dir=bin_dir, tmp_path=tmp_path, env=env)
    assert res.returncode == 0, res.stderr
    assert started_marker.exists()                              # start-instances was called
    assert "ec2 terminate-instances" in log.read_text()          # ...and it WAS eventually torn down
    lines = log.read_text().splitlines()
    start_idx = next(i for i, ln in enumerate(lines) if "ec2 start-instances" in ln)
    terminate_idx = next(i for i, ln in enumerate(lines) if "ec2 terminate-instances" in ln)
    assert start_idx < terminate_idx
    assert state.read_text().startswith("# TORN DOWN")


def _make_down_stub_bin_stopped_with_new_ip(tmp_path):
    """Like _make_down_stub_bin_stopped_then_reachable, but describe-instances' PublicIpAddress query answers a
    DIFFERENT ip AFTER start-instances than before -- exactly what EC2 does on every stop/start (there is no
    Elastic IP in this feature). ssh only succeeds once it is invoked against the NEW ip (embedded in its own
    -F ssh_config argument), so a `down` that never rewrites the Host block would time out here exactly as it
    did in production before the fix, instead of the old stub's lucky-because-content-agnostic reachability."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "combined.log"
    log.write_text("")
    started_marker = tmp_path / "started"
    OLD_IP, NEW_IP = "1.2.3.4", "9.9.9.9"

    aws_stub = bin_dir / "aws"
    aws_stub.write_text(f"""#!/usr/bin/env bash
echo "AWS $*" >> "{log}"
case "$*" in
  *"ec2 start-instances"*) touch "{started_marker}"; echo ok; exit 0 ;;
  *"describe-instances"*"State.Name"*)
    if [ -f "{started_marker}" ]; then echo running; else echo stopped; fi
    exit 0 ;;
  *"describe-instances"*"PublicIpAddress"*)
    if [ -f "{started_marker}" ]; then echo "{NEW_IP}"; else echo "{OLD_IP}"; fi
    exit 0 ;;
esac
echo ok
exit 0
""")
    aws_stub.chmod(aws_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    ssh_stub = bin_dir / "ssh"
    # Reachable ONLY when invoked with a config file whose testnode Host block carries the NEW ip -- proves the
    # probe actually ran against the rewritten block, not merely that some ssh call happened to succeed.
    ssh_stub.write_text(f"""#!/usr/bin/env bash
echo "SSH $*" >> "{log}"
cfg=""
prev=""
for a in "$@"; do
  if [ "$prev" = "-F" ]; then cfg="$a"; fi
  prev="$a"
done
[ -n "$cfg" ] && grep -q "{NEW_IP}" "$cfg" 2>/dev/null || exit 255
case "$*" in *pgrep*) echo 0 ;; esac
exit 0
""")
    ssh_stub.chmod(ssh_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    for name in ("rsync", "scp"):
        stub = bin_dir / name
        stub.write_text(f"""#!/usr/bin/env bash
echo "{name.upper()} $*" >> "{log}"
exit 0
""")
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir, log, started_marker, OLD_IP, NEW_IP


def test_down_rewrites_the_host_block_to_the_new_ip_after_restarting_a_stopped_instance(tmp_path):
    # THE ACTUAL DEFECT (re-review, HIGH, "the prior HIGH #4 stopped-instance branch does not work in practice"):
    # EC2 assigns a NEW public IPv4 on every stop/start (no Elastic IP here). The old code started the instance
    # then probed ssh against the SAME persistent Host block written at the LAST `up`/`down` -- i.e. the STALE
    # ip -- so the probe always timed out and `down` exited 1 leaving the instance RUNNING (a cost leak `down`
    # itself created). Fix: re-read the current PublicIpAddress and rewrite the Host block before probing.
    bin_dir, log, started_marker, old_ip, new_ip = _make_down_stub_bin_stopped_with_new_ip(tmp_path)
    state = _write_state(tmp_path)
    ssh_config = tmp_path / "ssh_config"
    ssh_config.write_text(f"Include ~/.ssh/config\nHost testnode\n  HostName {old_ip}\n")
    env = {
        "AWS_POOL_NODE_STATE_FILE": str(state),
        "POOL_EXTRA_NODES_FILE": str(tmp_path / "extra_nodes"),
        "POOL_SSH_CONFIG": str(ssh_config),
        "AWS_POOL_START_TIMEOUT_S": "5",
        "AWS_POOL_START_POLL_S": "0",
        "AWS_POOL_DRAIN_TIMEOUT_S": "5",
        "AWS_POOL_DRAIN_POLL_S": "0",
    }
    (tmp_path / "extra_nodes").write_text("testnode\n")

    res = _run(["down", "testnode"], bin_dir=bin_dir, tmp_path=tmp_path, env=env)
    assert res.returncode == 0, res.stderr
    assert started_marker.exists()
    assert "ec2 terminate-instances" in log.read_text()   # reached termination -- the reachability probe passed
    assert state.read_text().startswith("# TORN DOWN")
    # The Host block is removed as the LAST step (after terminate) -- while `down` was running, it must have
    # carried the NEW ip (asserted by the ssh stub itself refusing the OLD one); this re-confirms via the log
    # that at least one ssh call was made once the marker (and therefore the new ip) existed.
    ssh_calls_after_restart = [ln for ln in log.read_text().splitlines() if ln.startswith("SSH ")]
    assert ssh_calls_after_restart, "expected at least one ssh probe after restarting the instance"


def test_down_leaves_empty_describe_instances_state_unknown_not_gone(tmp_path):
    # THE ACTUAL DEFECT (re-review, MEDIUM, cost leak + false record): an EMPTY describe-instances result (a
    # transient AWS API/credential/throttle failure) was treated identically to a CONFIRMED terminated/gone
    # instance. With --force (which the old refusal text itself invited), `down` marked the state file
    # "# TORN DOWN" and removed the Host block WITHOUT ever calling terminate-instances -- a still-live instance
    # recorded as gone, its EBS volume and SG leaking indefinitely. Without --force it must simply refuse (not
    # crash, not silently proceed); it must never take the "already gone" shortcut on an empty read.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "combined.log"
    log.write_text("")
    aws_stub = bin_dir / "aws"
    aws_stub.write_text(f"""#!/usr/bin/env bash
echo "AWS $*" >> "{log}"
case "$*" in
  *"describe-instances"*) echo ""; exit 0 ;;
  *"terminate-instances"*) echo "shutting-down"; exit 0 ;;
esac
echo ok
exit 0
""")
    aws_stub.chmod(aws_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    for name in ("ssh", "rsync", "scp"):
        stub = bin_dir / name
        stub.write_text(f'#!/usr/bin/env bash\necho "{name.upper()} $*" >> "{log}"\nexit 255\n')
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    state = _write_state(tmp_path)
    ssh_config = tmp_path / "ssh_config"
    ssh_config.write_text("Include ~/.ssh/config\nHost testnode\n  HostName 1.2.3.4\n")
    env = {
        "AWS_POOL_NODE_STATE_FILE": str(state),
        "POOL_EXTRA_NODES_FILE": str(tmp_path / "extra_nodes"),
        "POOL_SSH_CONFIG": str(ssh_config),
        "AWS_POOL_DRAIN_TIMEOUT_S": "1",
        "AWS_POOL_DRAIN_POLL_S": "0",
    }
    (tmp_path / "extra_nodes").write_text("testnode\n")

    res = _run(["down", "testnode"], bin_dir=bin_dir, tmp_path=tmp_path, env=env)
    assert res.returncode == 1
    assert "ec2 terminate-instances" not in log.read_text()
    assert not state.read_text().startswith("# TORN DOWN")   # never marked gone on an unconfirmed read
    assert "HostName 1.2.3.4" in ssh_config.read_text()       # Host block untouched -- never removed either


def test_down_force_still_calls_terminate_on_empty_describe_instances(tmp_path):
    # --force may skip the REFUSAL above, but must never skip the actual terminate-instances CALL, and must
    # never mark the state file torn down as a substitute for attempting it (re-review, MEDIUM).
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "combined.log"
    log.write_text("")
    aws_stub = bin_dir / "aws"
    aws_stub.write_text(f"""#!/usr/bin/env bash
echo "AWS $*" >> "{log}"
case "$*" in
  *"describe-instances"*) echo ""; exit 0 ;;
esac
echo ok
exit 0
""")
    aws_stub.chmod(aws_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    for name in ("ssh", "rsync", "scp"):
        stub = bin_dir / name
        stub.write_text(f'#!/usr/bin/env bash\necho "{name.upper()} $*" >> "{log}"\nexit 255\n')
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    state = _write_state(tmp_path)
    ssh_config = tmp_path / "ssh_config"
    ssh_config.write_text("Include ~/.ssh/config\nHost testnode\n  HostName 1.2.3.4\n")
    env = {
        "AWS_POOL_NODE_STATE_FILE": str(state),
        "POOL_EXTRA_NODES_FILE": str(tmp_path / "extra_nodes"),
        "POOL_SSH_CONFIG": str(ssh_config),
        "AWS_POOL_DRAIN_TIMEOUT_S": "1",
        "AWS_POOL_DRAIN_POLL_S": "0",
    }
    (tmp_path / "extra_nodes").write_text("testnode\n")

    res = _run(["down", "testnode", "--force"], bin_dir=bin_dir, tmp_path=tmp_path, env=env)
    assert res.returncode == 0, res.stderr
    assert "ec2 terminate-instances" in log.read_text()   # the real terminate call was actually made
    assert state.read_text().startswith("# TORN DOWN")


def test_down_marks_torn_down_when_describe_instances_confirms_terminated(tmp_path):
    # The mirror/baseline case: a genuinely CONFIRMED terminated/shutting-down state (as opposed to an EMPTY,
    # unconfirmed read) is still the safe, cheap "already gone" shortcut -- unaffected by the empty-is-UNKNOWN
    # fix, and still requires --force like before (only the EMPTY case's handling changed).
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "combined.log"
    log.write_text("")
    aws_stub = bin_dir / "aws"
    aws_stub.write_text(f"""#!/usr/bin/env bash
echo "AWS $*" >> "{log}"
case "$*" in
  *"describe-instances"*) echo "terminated"; exit 0 ;;
esac
echo ok
exit 0
""")
    aws_stub.chmod(aws_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    state = _write_state(tmp_path)
    ssh_config = tmp_path / "ssh_config"
    ssh_config.write_text("Include ~/.ssh/config\nHost testnode\n  HostName 1.2.3.4\n")
    env = {
        "AWS_POOL_NODE_STATE_FILE": str(state),
        "POOL_EXTRA_NODES_FILE": str(tmp_path / "extra_nodes"),
        "POOL_SSH_CONFIG": str(ssh_config),
    }
    (tmp_path / "extra_nodes").write_text("testnode\n")

    res = _run(["down", "testnode", "--force"], bin_dir=bin_dir, tmp_path=tmp_path, env=env)
    assert res.returncode == 0, res.stderr
    assert "ec2 terminate-instances" not in log.read_text()   # nothing left to terminate -- the cheap shortcut
    assert state.read_text().startswith("# TORN DOWN")
    assert "Host testnode" not in ssh_config.read_text()


def test_down_pulls_job_status_log_before_terminating(tmp_path):
    # MEDIUM (re-review, lost-job observability): a crash on this node must be surfaced, not destroyed with the
    # root volume on terminate.
    bin_dir, log = _make_down_stub_bin(tmp_path)
    state = _write_state(tmp_path)
    ssh_config = tmp_path / "ssh_config"
    ssh_config.write_text("Include ~/.ssh/config\nHost testnode\n  HostName 1.2.3.4\n")
    env = {
        "AWS_POOL_NODE_STATE_FILE": str(state),
        "POOL_EXTRA_NODES_FILE": str(tmp_path / "extra_nodes"),
        "POOL_SSH_CONFIG": str(ssh_config),
    }
    (tmp_path / "extra_nodes").write_text("testnode\n")
    # _make_down_stub_bin's scp is not stubbed -- add one that logs + succeeds, matching its ssh/rsync stubs.
    scp_stub = bin_dir / "scp"
    scp_stub.write_text(f'#!/usr/bin/env bash\necho "SCP $*" >> "{log}"\nexit 0\n')
    scp_stub.chmod(scp_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    res = _run(["down", "testnode"], bin_dir=bin_dir, tmp_path=tmp_path, env=env)
    assert res.returncode == 0, res.stderr
    lines = log.read_text().splitlines()
    scp_idx = next((i for i, ln in enumerate(lines) if ln.startswith("SCP ")), None)
    terminate_idx = next((i for i, ln in enumerate(lines) if "ec2 terminate-instances" in ln), None)
    assert scp_idx is not None, f"job_status.log was never pulled: {lines}"
    assert terminate_idx is not None
    assert scp_idx < terminate_idx


def test_down_with_no_state_file_is_a_clean_noop(tmp_path):
    res = _run(["down", "ghost"], env={
        "AWS_POOL_NODE_STATE_FILE": str(tmp_path / "does-not-exist"),
        "POOL_EXTRA_NODES_FILE": str(tmp_path / "extra_nodes"),
    })
    assert res.returncode == 0, res.stderr


# --------------------------------------------------------------------------------------- budget gate on `up`

def test_up_refused_by_budget_never_calls_run_instances(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "aws.log"
    log.write_text("")
    describe = tmp_path / "describe.json"
    # One already-running, expensive-enough project instance so a tiny cap refuses immediately.
    describe.write_text(
        '{"Reservations": [{"Instances": [{"InstanceId": "i-existing", "InstanceType": "r7i.4xlarge", '
        '"State": {"Name": "running"}, "LaunchTime": "2020-01-01T00:00:00+00:00", '
        '"Tags": [{"Key": "Project", "Value": "neural-sim"}]}]}]}'
    )
    aws_stub = bin_dir / "aws"
    aws_stub.write_text(f"""#!/usr/bin/env bash
echo "$*" >> "{log}"
if [[ "$*" == *"ec2 describe-instances"* ]]; then cat "{describe}"; exit 0; fi
if [[ "$*" == *"ec2 run-instances"* ]]; then echo "i-should-not-happen"; exit 0; fi
echo ok
exit 0
""")
    aws_stub.chmod(aws_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    res = _run(["up", "testnode"], bin_dir=bin_dir, tmp_path=tmp_path, env={
        "AWS_POOL_NODE_STATE_FILE": str(tmp_path / ".aws_testnode"),
        "AWS_DAILY_CAP_USD": "1",
    })
    assert res.returncode == 1
    assert "refused by tools/aws_budget.sh" in res.stdout + res.stderr
    assert "ec2 run-instances" not in log.read_text()
    assert not (tmp_path / ".aws_testnode").exists()   # aws_cpu_launch.sh never even ran


def test_up_refuses_when_a_live_state_file_already_exists(tmp_path):
    state = _write_state(tmp_path, instance="i-live")
    res = _run(["up", "testnode"], env={"AWS_POOL_NODE_STATE_FILE": str(state)})
    assert res.returncode == 1
    assert "already recorded live" in res.stdout + res.stderr


def _make_expensive_running_instance_aws_stub(tmp_path):
    """A stub `aws` whose describe-instances reports one already-running, expensive project instance -- enough
    for a $1 cap to refuse cleanly at the budget-check gate (no real network, no run-instances)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "aws.log"
    log.write_text("")
    describe = tmp_path / "describe.json"
    describe.write_text(
        '{"Reservations": [{"Instances": [{"InstanceId": "i-existing", "InstanceType": "r7i.4xlarge", '
        '"State": {"Name": "running"}, "LaunchTime": "2020-01-01T00:00:00+00:00", '
        '"Tags": [{"Key": "Project", "Value": "neural-sim"}]}]}]}'
    )
    aws_stub = bin_dir / "aws"
    aws_stub.write_text(f"""#!/usr/bin/env bash
echo "$*" >> "{log}"
if [[ "$*" == *"ec2 describe-instances"* ]]; then cat "{describe}"; exit 0; fi
if [[ "$*" == *"ec2 run-instances"* ]]; then echo "i-should-not-happen"; exit 0; fi
echo ok
exit 0
""")
    aws_stub.chmod(aws_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir, log


def test_up_does_not_refuse_on_a_torn_down_state_file_reaching_the_next_gate(tmp_path):
    # aws_pool_node.sh's OWN "already recorded live" guard already special-cased a '# TORN DOWN' marker -- this
    # exercises that it really does let a re-`up` PAST that guard, all the way into aws_cpu_launch.sh (down
    # never deletes the state file, by design, so its stale `instance=` line must not permanently block the
    # node-name). The budget stub below refuses at the NEXT gate instead, so no real AWS/run-instances call
    # happens either way -- this test is only about which gate stops the run.
    state = _write_state(tmp_path, instance="i-old")
    state.write_text("# TORN DOWN 2026-09-23 00:00:00 UTC\n" + state.read_text())
    bin_dir, aws_log = _make_expensive_running_instance_aws_stub(tmp_path)
    res = _run(["up", "testnode"], bin_dir=bin_dir, tmp_path=tmp_path,
                env={"AWS_POOL_NODE_STATE_FILE": str(state), "AWS_DAILY_CAP_USD": "1"})
    assert "already recorded live" not in (res.stdout + res.stderr)
    assert res.returncode == 1
    assert "refused by tools/aws_budget.sh" in (res.stdout + res.stderr)
    assert "ec2 run-instances" not in aws_log.read_text()


def test_aws_cpu_launch_still_refuses_a_genuinely_live_state_file(tmp_path):
    # The mirror case, directly against tools/aws_cpu_launch.sh (the script whose OWN separate "live=" check was
    # the actual 2026-09-23 fix-round bug -- it read `instance=` blindly, ignoring '# TORN DOWN'). A state file
    # WITHOUT the marker must still refuse re-launch, unchanged.
    root = ROOT
    key = tmp_path / "key.pem"; key.write_text("fake\n")
    state = tmp_path / ".aws_gpu"
    state.write_text(f"instance=i-live\nregion=us-east-1\nkey={key}\nsg=sg-live\n")
    res = subprocess.run(["bash", str(root / "tools" / "aws_cpu_launch.sh")], cwd=root,
                          env={**os.environ, "AWS_CPU_STATE_FILE": str(state)},
                          capture_output=True, text=True, timeout=30)
    assert res.returncode == 1
    assert "already recorded" in (res.stdout + res.stderr)


def test_aws_cpu_launch_honors_the_torn_down_marker_and_proceeds_past_its_own_live_check(tmp_path):
    # The actual 2026-09-23 fix: the SAME state file, but WITH the '# TORN DOWN' marker prepended (exactly what
    # aws_pool_node.sh's `down` leaves behind) -- aws_cpu_launch.sh's own live-check must not refuse, and the run
    # must instead reach (and stop at) the budget-check gate.
    key = tmp_path / "key.pem"; key.write_text("fake\n")
    state = tmp_path / ".aws_gpu"
    state.write_text(f"# TORN DOWN 2026-09-23 00:00:00 UTC\ninstance=i-old\nregion=us-east-1\nkey={key}\nsg=sg-old\n")
    bin_dir, aws_log = _make_expensive_running_instance_aws_stub(tmp_path)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["AWS_CPU_STATE_FILE"] = str(state)
    env["AWS_DAILY_CAP_USD"] = "1"
    res = subprocess.run(["bash", str(ROOT / "tools" / "aws_cpu_launch.sh")], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=30)
    assert "already recorded" not in (res.stdout + res.stderr)
    assert res.returncode == 1
    assert "refused by tools/aws_budget.sh" in (res.stdout + res.stderr)
    assert "ec2 run-instances" not in aws_log.read_text()


# --------------------------------------------------------------------- up auto-teardown (fix round #2, LOW)

def test_up_auto_tears_down_when_the_instance_has_no_public_ip(tmp_path):
    # THE DEFECT: `up` used to `exit 1` directly on a missing public IP, BEFORE `_up_failed` was even defined --
    # skipping its terminate+delete-SG teardown entirely (an instance with no public IP still bills and still
    # leaves an SG behind). aws_cpu_launch.sh's own stub below succeeds (writes a state file with instance+sg),
    # and describe-instances answers with an empty/None PublicIpAddress.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "aws.log"
    log.write_text("")
    state = tmp_path / ".aws_testnode"

    aws_stub = bin_dir / "aws"
    aws_stub.write_text(f"""#!/usr/bin/env bash
echo "$*" >> "{log}"
case "$*" in
  *"describe-instances"*"--filters"*) echo '{{"Reservations": []}}'; exit 0 ;;   # aws_budget.sh's fetch: no project instances -> $0 spend
  *"describe-instances"*"PublicIpAddress"*) echo "None"; exit 0 ;;
  *"ec2 run-instances"*) echo "i-noip"; exit 0 ;;
esac
echo ok
exit 0
""")
    aws_stub.chmod(aws_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    # aws_cpu_launch.sh (the REAL script `up` calls) shells out to `curl` for its own public IP -- stub it so
    # the test makes no real network call.
    curl_stub = bin_dir / "curl"
    curl_stub.write_text("#!/usr/bin/env bash\necho 203.0.113.1\n")
    curl_stub.chmod(curl_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    res = _run(["up", "testnode"], bin_dir=bin_dir, tmp_path=tmp_path, env={
        "AWS_POOL_NODE_STATE_FILE": str(state),
        "AWS_DAILY_CAP_USD": "10000",
    })
    # aws_cpu_launch.sh is the REAL script here (not stubbed) -- it calls the stubbed `aws` itself and writes a
    # real-shaped state file (instance id from run-instances, sg from create-security-group, etc.), so by the
    # time `up` reaches the public-IP check, IID/SG are genuinely populated from that state file.
    assert res.returncode == 1
    assert "no public IP" in (res.stdout + res.stderr)
    assert "ec2 terminate-instances" in log.read_text(), "no-public-IP must still auto-teardown, not just exit"
    assert state.exists() and state.read_text().startswith("# TORN DOWN"), \
        "no-public-IP must mark the state file torn down so a retried `up` is not blocked"


# ----------------------------------------------------------------------------------------------------- status

def test_status_reports_not_launched_when_no_state_file(tmp_path):
    res = _run(["status", "testnode"], env={"AWS_POOL_NODE_STATE_FILE": str(tmp_path / "nope")})
    assert res.returncode == 0, res.stderr
    assert "not launched" in res.stdout


def test_status_reports_torn_down(tmp_path):
    state = _write_state(tmp_path)
    state.write_text("# TORN DOWN 2026-09-23 00:00:00 UTC\n" + state.read_text())
    res = _run(["status", "testnode"], env={"AWS_POOL_NODE_STATE_FILE": str(state)})
    assert res.returncode == 0, res.stderr
    assert "TORN DOWN" in res.stdout


def test_down_does_not_mark_torn_down_when_terminate_fails(tmp_path):
    # Re-review round 4 (MEDIUM): every aws call fails (RequestLimitExceeded, exit 255). down must NOT write
    # '# TORN DOWN', must keep the Host block, and must exit non-zero -- a still-running instance is not torn down.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "combined.log"
    log.write_text("")
    aws_stub = bin_dir / "aws"
    aws_stub.write_text(f"""#!/usr/bin/env bash
echo "AWS $*" >> "{log}"
echo "An error occurred (RequestLimitExceeded)" >&2
exit 255
""")
    aws_stub.chmod(aws_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    for name in ("ssh", "rsync", "scp"):
        stub = bin_dir / name
        stub.write_text(f'#!/usr/bin/env bash\necho "{name.upper()} $*" >> "{log}"\nexit 255\n')
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    state = _write_state(tmp_path)
    ssh_config = tmp_path / "ssh_config"
    ssh_config.write_text("Include ~/.ssh/config\nHost testnode\n  HostName 1.2.3.4\n")
    env = {
        "AWS_POOL_NODE_STATE_FILE": str(state),
        "POOL_EXTRA_NODES_FILE": str(tmp_path / "extra_nodes"),
        "POOL_SSH_CONFIG": str(ssh_config),
        "AWS_POOL_DRAIN_TIMEOUT_S": "1",
        "AWS_POOL_DRAIN_POLL_S": "0",
    }
    (tmp_path / "extra_nodes").write_text("testnode\n")
    res = _run(["down", "testnode", "--force"], bin_dir=bin_dir, tmp_path=tmp_path, env=env)
    assert res.returncode != 0
    assert not state.read_text().startswith("# TORN DOWN")
    assert "Host testnode" in ssh_config.read_text()


# --------------------------------------------------------------------------------------------------- `start`

def _make_start_stub_bin(tmp_path, initial_state="stopped", instance_type="r7i.4xlarge",
                          old_ip="1.2.3.4", new_ip="9.9.9.9", ssh_ok=True):
    """Stub `aws`+`ssh` for `start`: describe-instances answers State.Name/PublicIpAddress/InstanceType
    distinctly by --query content, transitioning stopped->running the instant start-instances is called
    (marked by a sentinel file so the poll loop below observes it on its NEXT describe-instances call, exactly
    like the real API). Budget's own describe-instances --filters fetch always reports zero project instances
    (so AWS_DAILY_CAP_USD alone controls whether `check` allows or refuses)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "combined.log"
    log.write_text("")
    started_marker = tmp_path / "started"
    if initial_state == "running":
        started_marker.write_text("")   # already running from the start -- no start-instances call expected

    aws_stub = bin_dir / "aws"
    aws_stub.write_text(f"""#!/usr/bin/env bash
echo "AWS $*" >> "{log}"
case "$*" in
  *"--filters"*) echo '{{"Reservations": []}}'; exit 0 ;;
  *"ec2 start-instances"*) touch "{started_marker}"; echo ok; exit 0 ;;
  *"State.Name"*)
    if [ -f "{started_marker}" ]; then echo running; else echo {initial_state}; fi
    exit 0 ;;
  *"PublicIpAddress"*)
    if [ -f "{started_marker}" ]; then echo "{new_ip}"; else echo "{old_ip}"; fi
    exit 0 ;;
  *"InstanceType"*) echo "{instance_type}"; exit 0 ;;
esac
echo ok
exit 0
""")
    aws_stub.chmod(aws_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    ssh_stub = bin_dir / "ssh"
    if ssh_ok:
        ssh_stub.write_text(f"""#!/usr/bin/env bash
echo "SSH $*" >> "{log}"
exit 0
""")
    else:
        ssh_stub.write_text(f"""#!/usr/bin/env bash
echo "SSH $*" >> "{log}"
exit 255
""")
    ssh_stub.chmod(ssh_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir, log, started_marker


def test_start_starts_a_stopped_instance_waits_and_rewrites_the_host_block(tmp_path):
    bin_dir, log, started_marker = _make_start_stub_bin(tmp_path, initial_state="stopped",
                                                          old_ip="1.2.3.4", new_ip="9.9.9.9")
    state = _write_state(tmp_path, instance="i-aaa")
    ssh_config = tmp_path / "ssh_config"
    ssh_config.write_text("Include ~/.ssh/config\nHost testnode\n  HostName 1.2.3.4\n")
    env = {
        "AWS_POOL_NODE_STATE_FILE": str(state),
        "POOL_SSH_CONFIG": str(ssh_config),
        "AWS_POOL_START_TIMEOUT_S": "5",
        "AWS_POOL_START_POLL_S": "0",
        "AWS_DAILY_CAP_USD": "10000",
    }
    res = _run(["start", "testnode"], bin_dir=bin_dir, tmp_path=tmp_path, env=env)
    assert res.returncode == 0, res.stderr
    assert started_marker.exists(), "start-instances was never called"
    assert "9.9.9.9" in ssh_config.read_text()
    assert "1.2.3.4" not in ssh_config.read_text()   # stale ip replaced, not merely appended alongside
    assert (tmp_path / "ssh_config.bak").exists(), "no backup of the prior config was kept"
    assert "1.2.3.4" in (tmp_path / "ssh_config.bak").read_text()   # the backup holds the PRIOR content
    assert "reachable" in res.stdout


def test_start_refused_by_budget_never_calls_start_instances(tmp_path):
    bin_dir, log, started_marker = _make_start_stub_bin(tmp_path, initial_state="stopped")
    state = _write_state(tmp_path, instance="i-aaa")
    ssh_config = tmp_path / "ssh_config"
    ssh_config.write_text("Include ~/.ssh/config\nHost testnode\n  HostName 1.2.3.4\n")
    env = {
        "AWS_POOL_NODE_STATE_FILE": str(state),
        "POOL_SSH_CONFIG": str(ssh_config),
        "AWS_DAILY_CAP_USD": "0",   # any positive-cost launch refuses
    }
    res = _run(["start", "testnode"], bin_dir=bin_dir, tmp_path=tmp_path, env=env)
    assert res.returncode == 1
    assert "refused by tools/aws_budget.sh" in (res.stdout + res.stderr)
    assert not started_marker.exists(), "start-instances must never be called once budget refuses"
    assert "1.2.3.4" in ssh_config.read_text()   # untouched


def test_start_on_an_already_running_node_only_refreshes_never_calls_start_instances(tmp_path):
    bin_dir, log, started_marker = _make_start_stub_bin(tmp_path, initial_state="running", new_ip="9.9.9.9")
    state = _write_state(tmp_path, instance="i-aaa")
    ssh_config = tmp_path / "ssh_config"
    ssh_config.write_text("Include ~/.ssh/config\nHost testnode\n  HostName 1.2.3.4\n")
    env = {
        "AWS_POOL_NODE_STATE_FILE": str(state),
        "POOL_SSH_CONFIG": str(ssh_config),
        "AWS_DAILY_CAP_USD": "0",   # would refuse a real launch -- proves start-instances truly never happens
    }
    res = _run(["start", "testnode"], bin_dir=bin_dir, tmp_path=tmp_path, env=env)
    assert res.returncode == 0, res.stderr
    assert "ec2 start-instances" not in log.read_text()
    assert "9.9.9.9" in ssh_config.read_text()


def test_start_refuses_a_terminated_instance(tmp_path):
    bin_dir, log, started_marker = _make_start_stub_bin(tmp_path, initial_state="terminated")
    state = _write_state(tmp_path, instance="i-aaa")
    env = {"AWS_POOL_NODE_STATE_FILE": str(state), "POOL_SSH_CONFIG": str(tmp_path / "no-cfg")}
    res = _run(["start", "testnode"], bin_dir=bin_dir, tmp_path=tmp_path, env=env)
    assert res.returncode == 1
    assert "cannot start" in (res.stdout + res.stderr)
    assert "ec2 start-instances" not in log.read_text()


def test_start_refuses_with_no_state_file(tmp_path):
    res = _run(["start", "testnode"], env={"AWS_POOL_NODE_STATE_FILE": str(tmp_path / "nope")})
    assert res.returncode == 1
    assert "no state file" in (res.stdout + res.stderr)


def test_start_refuses_a_torn_down_node(tmp_path):
    state = _write_state(tmp_path, instance="i-old")
    state.write_text("# TORN DOWN 2026-09-23 00:00:00 UTC\n" + state.read_text())
    res = _run(["start", "testnode"], env={"AWS_POOL_NODE_STATE_FILE": str(state)})
    assert res.returncode == 1
    assert "TORN DOWN" in (res.stdout + res.stderr)


def test_start_surfaces_start_instances_failure_instead_of_silently_timing_out(tmp_path):
    # LOW (2026-09-25 review): the OLD code discarded start-instances' own rc/stderr (`>/dev/null 2>&1`) and
    # then waited the FULL poll timeout with no explanation for why the instance never reached 'running'. A
    # real failure (throttled/credential/quota error) must be surfaced IMMEDIATELY, with the real reason.
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    log = tmp_path / "combined.log"; log.write_text("")
    aws_stub = bin_dir / "aws"
    aws_stub.write_text(f"""#!/usr/bin/env bash
echo "AWS $*" >> "{log}"
case "$*" in
  *"--filters"*) echo '{{"Reservations": []}}'; exit 0 ;;
  *"ec2 start-instances"*) echo "An error occurred (RequestLimitExceeded)" >&2; exit 255 ;;
  *"State.Name"*) echo stopped; exit 0 ;;
  *"InstanceType"*) echo r7i.4xlarge; exit 0 ;;
esac
echo ok; exit 0
""")
    aws_stub.chmod(aws_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    state = _write_state(tmp_path, instance="i-aaa")
    ssh_config = tmp_path / "ssh_config"
    ssh_config.write_text("Include ~/.ssh/config\nHost testnode\n  HostName 1.2.3.4\n")
    env = {
        "AWS_POOL_NODE_STATE_FILE": str(state),
        "POOL_SSH_CONFIG": str(ssh_config),
        "AWS_POOL_START_TIMEOUT_S": "3",
        "AWS_POOL_START_POLL_S": "0",
        "AWS_DAILY_CAP_USD": "10000",
    }
    t0 = time.monotonic()
    res = _run(["start", "testnode"], bin_dir=bin_dir, tmp_path=tmp_path, env=env)
    elapsed = time.monotonic() - t0
    assert res.returncode == 1
    assert "start-instances failed" in (res.stdout + res.stderr)
    assert "RequestLimitExceeded" in (res.stdout + res.stderr)
    assert elapsed < 3, "must fail immediately on a real start-instances error, not wait out the whole poll timeout"
    assert "1.2.3.4" in ssh_config.read_text(), "never touched -- failed before any Host-block rewrite"


def test_start_waits_out_a_stopping_instance_then_starts_it(tmp_path):
    # LOW: 'stopping' (e.g. aws_idle_stop.sh's own stop-instances call landed moments ago) must be WAITED OUT,
    # not refused like 'terminated' -- it resolves to 'stopped' on its own, from which `start` CAN proceed.
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    log = tmp_path / "combined.log"; log.write_text("")
    poll_counter = tmp_path / "polls"
    started_marker = tmp_path / "started"
    aws_stub = bin_dir / "aws"
    aws_stub.write_text(f"""#!/usr/bin/env bash
echo "AWS $*" >> "{log}"
case "$*" in
  *"--filters"*) echo '{{"Reservations": []}}'; exit 0 ;;
  *"ec2 start-instances"*) touch "{started_marker}"; echo ok; exit 0 ;;
  *"State.Name"*)
    if [ -f "{started_marker}" ]; then echo running; exit 0; fi
    n=$(cat "{poll_counter}" 2>/dev/null || echo 0); n=$((n+1)); echo "$n" > "{poll_counter}"
    if [ "$n" -ge 2 ]; then echo stopped; else echo stopping; fi
    exit 0 ;;
  *"PublicIpAddress"*) echo "9.9.9.9"; exit 0 ;;
  *"InstanceType"*) echo r7i.4xlarge; exit 0 ;;
esac
echo ok; exit 0
""")
    aws_stub.chmod(aws_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    ssh_stub = bin_dir / "ssh"
    ssh_stub.write_text(f'#!/usr/bin/env bash\necho "SSH $*" >> "{log}"\nexit 0\n')
    ssh_stub.chmod(ssh_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    state = _write_state(tmp_path, instance="i-aaa")
    ssh_config = tmp_path / "ssh_config"
    ssh_config.write_text("Include ~/.ssh/config\nHost testnode\n  HostName 1.2.3.4\n")
    env = {
        "AWS_POOL_NODE_STATE_FILE": str(state),
        "POOL_SSH_CONFIG": str(ssh_config),
        "AWS_POOL_START_TIMEOUT_S": "5",
        "AWS_POOL_START_POLL_S": "0",
        "AWS_POOL_STOP_WAIT_TIMEOUT_S": "5",
        "AWS_DAILY_CAP_USD": "10000",
    }
    res = _run(["start", "testnode"], bin_dir=bin_dir, tmp_path=tmp_path, env=env)
    assert res.returncode == 0, res.stderr
    assert started_marker.exists(), "start-instances was never called once the instance genuinely stopped"
    assert "9.9.9.9" in ssh_config.read_text()


def test_start_waits_out_a_pending_instance_without_calling_start_instances_again(tmp_path):
    # LOW: 'pending' means a start is ALREADY in flight (e.g. a concurrent `start` call, or the owner's own
    # console action) -- must be waited out to 'running', never treated as an unhandled state, and NEVER call
    # start-instances a second time on top of the one already in flight.
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    log = tmp_path / "combined.log"; log.write_text("")
    poll_counter = tmp_path / "polls"
    aws_stub = bin_dir / "aws"
    aws_stub.write_text(f"""#!/usr/bin/env bash
echo "AWS $*" >> "{log}"
case "$*" in
  *"--filters"*) echo '{{"Reservations": []}}'; exit 0 ;;
  *"ec2 start-instances"*) echo "SHOULD NEVER BE CALLED" >&2; exit 1 ;;
  *"State.Name"*)
    n=$(cat "{poll_counter}" 2>/dev/null || echo 0); n=$((n+1)); echo "$n" > "{poll_counter}"
    if [ "$n" -ge 2 ]; then echo running; else echo pending; fi
    exit 0 ;;
  *"PublicIpAddress"*) echo "9.9.9.9"; exit 0 ;;
esac
echo ok; exit 0
""")
    aws_stub.chmod(aws_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    ssh_stub = bin_dir / "ssh"
    ssh_stub.write_text(f'#!/usr/bin/env bash\necho "SSH $*" >> "{log}"\nexit 0\n')
    ssh_stub.chmod(ssh_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    state = _write_state(tmp_path, instance="i-aaa")
    ssh_config = tmp_path / "ssh_config"
    ssh_config.write_text("Include ~/.ssh/config\nHost testnode\n  HostName 1.2.3.4\n")
    env = {
        "AWS_POOL_NODE_STATE_FILE": str(state),
        "POOL_SSH_CONFIG": str(ssh_config),
        "AWS_POOL_START_TIMEOUT_S": "5",
        "AWS_POOL_START_POLL_S": "0",
        "AWS_DAILY_CAP_USD": "10000",
    }
    res = _run(["start", "testnode"], bin_dir=bin_dir, tmp_path=tmp_path, env=env)
    assert res.returncode == 0, res.stderr
    assert "ec2 start-instances" not in log.read_text()
    assert "9.9.9.9" in ssh_config.read_text()


# -------------------------------------------------------------------------------------------------- `refresh`

def _make_refresh_stub_bin(tmp_path, ec2_state="running", ip="9.9.9.9"):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "combined.log"
    log.write_text("")
    aws_stub = bin_dir / "aws"
    aws_stub.write_text(f"""#!/usr/bin/env bash
echo "AWS $*" >> "{log}"
case "$*" in
  *"State.Name"*) echo "{ec2_state}"; exit 0 ;;
  *"PublicIpAddress"*) echo "{ip}"; exit 0 ;;
esac
echo ok
exit 0
""")
    aws_stub.chmod(aws_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir, log


def test_refresh_rewrites_a_stale_host_block_for_a_running_node(tmp_path):
    bin_dir, log = _make_refresh_stub_bin(tmp_path, ec2_state="running", ip="9.9.9.9")
    state = _write_state(tmp_path, instance="i-aaa")
    ssh_config = tmp_path / "ssh_config"
    ssh_config.write_text("Include ~/.ssh/config\nHost testnode\n  HostName 1.2.3.4\n")
    env = {"AWS_POOL_NODE_STATE_FILE": str(state), "POOL_SSH_CONFIG": str(ssh_config)}
    res = _run(["refresh", "testnode"], bin_dir=bin_dir, tmp_path=tmp_path, env=env)
    assert res.returncode == 0, res.stderr
    assert "9.9.9.9" in ssh_config.read_text()
    assert "1.2.3.4" not in ssh_config.read_text()
    assert (tmp_path / "ssh_config.bak").exists()
    assert "refreshed" in res.stdout


def test_refresh_is_a_noop_when_the_recorded_ip_is_already_current(tmp_path):
    bin_dir, log = _make_refresh_stub_bin(tmp_path, ec2_state="running", ip="1.2.3.4")
    state = _write_state(tmp_path, instance="i-aaa")
    ssh_config = tmp_path / "ssh_config"
    ssh_config.write_text("Include ~/.ssh/config\nHost testnode\n  HostName 1.2.3.4\n")
    env = {"AWS_POOL_NODE_STATE_FILE": str(state), "POOL_SSH_CONFIG": str(ssh_config)}
    res = _run(["refresh", "testnode"], bin_dir=bin_dir, tmp_path=tmp_path, env=env)
    assert res.returncode == 0, res.stderr
    assert "already current" in res.stdout
    assert not (tmp_path / "ssh_config.bak").exists(), "a no-op refresh must not rewrite (or back up) anything"


def test_refresh_never_starts_a_stopped_node(tmp_path):
    bin_dir, log = _make_refresh_stub_bin(tmp_path, ec2_state="stopped")
    state = _write_state(tmp_path, instance="i-aaa")
    ssh_config = tmp_path / "ssh_config"
    ssh_config.write_text("Include ~/.ssh/config\nHost testnode\n  HostName 1.2.3.4\n")
    env = {"AWS_POOL_NODE_STATE_FILE": str(state), "POOL_SSH_CONFIG": str(ssh_config)}
    res = _run(["refresh", "testnode"], bin_dir=bin_dir, tmp_path=tmp_path, env=env)
    assert res.returncode == 0, res.stderr
    assert "ec2 start-instances" not in log.read_text()
    assert "1.2.3.4" in ssh_config.read_text()   # untouched


def test_refresh_with_no_state_file_is_a_noop(tmp_path):
    res = _run(["refresh", "testnode"], env={"AWS_POOL_NODE_STATE_FILE": str(tmp_path / "nope")})
    assert res.returncode == 0, res.stderr
    assert "nothing to refresh" in res.stdout


def test_refresh_never_touches_the_real_ssh_config_path():
    # A cheap static guard against a regression that would make `refresh` (or `start`) fall back to the
    # default POOL_SSH_CONFIG resolution and accidentally reference the user's real ~/.ssh/config path
    # directly (it must only ever be `Include`d, never written).
    src = (ROOT / "tools" / "aws_pool_node.sh").read_text()
    for line in src.splitlines():
        if "~/.ssh/config" in line:
            assert "Include" in line or "NEVER" in line or line.strip().startswith("#"), \
                f"a non-Include, non-comment reference to ~/.ssh/config: {line!r}"
