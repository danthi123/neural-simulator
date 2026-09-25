"""Tests for the DURABLE pause/resume restore (2026-09-25 review, fix round 4, MEDIUM: "the pause/resume pair
has no durable restore" -- reproduced by sending SIGTERM to tools/aws_idle_stop.sh mid-sync: .pool_extra_nodes
was left empty and nothing ever restored it).

Covers, in order:
  (A) the marker primitives (pause writes one, resume clears it) via tools/aws_stop_safety_lib.sh's CLI seam;
  (B) reregister_stale_paused_nodes -- leaves a fresh marker alone, restores + clears a stale one;
  (C) the actual "kill the script mid-sync" scenarios the review asked for, for BOTH callers
      (tools/aws_idle_stop.sh, tools/aws_budget.sh enforce): SIGKILL (untrappable -- proves the DURABLE marker
      mechanism, not just the trap, is doing the work) and SIGTERM (proves the trap resumes immediately, without
      waiting for the staleness window);
  (D) every OTHER entry point re-registering a stale pause (tools/aws_pool_node.sh start/refresh, the
      dispatcher's own cycle_setup, and fill_node stopping mid-cycle on a live pause marker);
  (E) the shared flock across every .pool_extra_nodes writer (pause/resume/up/down).

Every test isolates POOL_EXTRA_NODES_FILE and POOL_PAUSE_MARK_DIR to tmp_path -- never the shared production
research/queue/ files (other sessions/agents touch them concurrently).
"""
from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "tools" / "aws_stop_safety_lib.sh"
AWS_IDLE_STOP = ROOT / "tools" / "aws_idle_stop.sh"
AWS_BUDGET = ROOT / "tools" / "aws_budget.sh"
AWS_POOL_NODE = ROOT / "tools" / "aws_pool_node.sh"
DISPATCHER = ROOT / "tools" / "pool_autodispatch.sh"


def _lib_run(args, extra_env=None, tmp_path=None, timeout=15):
    env = dict(os.environ)
    if tmp_path is not None:
        env.setdefault("AWS_SYNC_LOG", str(tmp_path / "sync.log"))
        env.setdefault("POOL_SSH_CONFIG", str(tmp_path / "no-such-pool-ssh-config"))
        env.setdefault("POOL_EXTRA_NODES_FILE", str(tmp_path / "extra_nodes"))
        env.setdefault("POOL_PAUSE_MARK_DIR", str(tmp_path / "paused"))
    if extra_env:
        env.update(extra_env)
    return subprocess.run(["bash", str(LIB), *args], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, timeout=timeout)


# =================================================================================== (A) marker primitives

def test_pause_writes_a_marker_only_when_it_actually_removes_a_registration(tmp_path):
    extra = tmp_path / "extra_nodes"
    extra.write_text("gpu\n")
    mark_dir = tmp_path / "paused"
    res = _lib_run(["--pause", "gpu"],
                    extra_env={"POOL_EXTRA_NODES_FILE": str(extra), "POOL_PAUSE_MARK_DIR": str(mark_dir)},
                    tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "1"
    assert (mark_dir / "gpu").exists(), "pause removed a registration but wrote no durable marker"


def test_pause_writes_no_marker_on_a_noop(tmp_path):
    extra = tmp_path / "extra_nodes"
    extra.write_text("other-node\n")
    mark_dir = tmp_path / "paused"
    res = _lib_run(["--pause", "gpu"],
                    extra_env={"POOL_EXTRA_NODES_FILE": str(extra), "POOL_PAUSE_MARK_DIR": str(mark_dir)},
                    tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "0"
    assert not mark_dir.exists() or not (mark_dir / "gpu").exists()


def test_resume_clears_the_marker_pause_wrote(tmp_path):
    extra = tmp_path / "extra_nodes"
    extra.write_text("gpu\n")
    mark_dir = tmp_path / "paused"
    env = {"POOL_EXTRA_NODES_FILE": str(extra), "POOL_PAUSE_MARK_DIR": str(mark_dir)}
    r1 = _lib_run(["--pause", "gpu"], extra_env=env, tmp_path=tmp_path)
    assert r1.stdout.strip() == "1"
    assert (mark_dir / "gpu").exists()
    r2 = _lib_run(["--resume", "gpu", "1"], extra_env=env, tmp_path=tmp_path)
    assert r2.returncode == 0, r2.stderr
    assert not (mark_dir / "gpu").exists(), "resume left the marker behind after successfully restoring"
    assert extra.read_text().splitlines() == ["gpu"]


def test_resume_with_was_registered_zero_never_touches_a_marker(tmp_path):
    # A stale marker from an EARLIER pause must survive an unrelated resume(was=0) call for the same node --
    # otherwise the durable record that a resume is still owed would be silently erased without ever restoring
    # the registration (this exact bug would defeat reregister_stale_paused_nodes entirely).
    extra = tmp_path / "extra_nodes"
    extra.write_text("")
    mark_dir = tmp_path / "paused"
    mark_dir.mkdir()
    (mark_dir / "gpu").write_text("123")
    env = {"POOL_EXTRA_NODES_FILE": str(extra), "POOL_PAUSE_MARK_DIR": str(mark_dir)}
    res = _lib_run(["--resume", "gpu", "0"], extra_env=env, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert (mark_dir / "gpu").exists(), "a was=0 resume must never remove a marker it did not itself own"
    assert extra.read_text() == ""


# =================================================================================== (B) reregister_stale_paused_nodes

def test_reregister_stale_leaves_a_fresh_marker_alone(tmp_path):
    extra = tmp_path / "extra_nodes"
    extra.write_text("")
    mark_dir = tmp_path / "paused"
    mark_dir.mkdir()
    (mark_dir / "gpu").write_text(str(int(time.time())))   # just paused
    env = {"POOL_EXTRA_NODES_FILE": str(extra), "POOL_PAUSE_MARK_DIR": str(mark_dir),
           "POOL_PAUSE_MAX_WINDOW_S": "600"}
    res = _lib_run(["--reregister-stale"], extra_env=env, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert extra.read_text() == "", "a FRESH pause (still legitimately in progress) must not be reregistered"
    assert (mark_dir / "gpu").exists(), "a fresh marker must not be cleared"


def test_reregister_stale_restores_and_clears_an_old_marker(tmp_path):
    # THE MEDIUM ITSELF: a marker older than the window means the pausing process died before it could resume
    # the node -- every entry point calling this must self-heal.
    extra = tmp_path / "extra_nodes"
    extra.write_text("")
    mark_dir = tmp_path / "paused"
    mark_dir.mkdir()
    (mark_dir / "gpu").write_text(str(int(time.time()) - 9999))   # ancient
    env = {"POOL_EXTRA_NODES_FILE": str(extra), "POOL_PAUSE_MARK_DIR": str(mark_dir),
           "POOL_PAUSE_MAX_WINDOW_S": "600"}
    res = _lib_run(["--reregister-stale"], extra_env=env, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert extra.read_text().splitlines() == ["gpu"], "a stale pause must be re-registered for dispatch"
    assert not (mark_dir / "gpu").exists(), "the stale marker must be cleared once handled"
    log = (tmp_path / "sync.log").read_text()
    assert "Re-registering for dispatch" in log


def test_reregister_stale_is_a_noop_with_no_marker_directory(tmp_path):
    env = {"POOL_EXTRA_NODES_FILE": str(tmp_path / "extra_nodes"),
           "POOL_PAUSE_MARK_DIR": str(tmp_path / "does-not-exist")}
    res = _lib_run(["--reregister-stale"], extra_env=env, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr


def test_reregister_stale_multiple_nodes_only_touches_the_stale_one(tmp_path):
    extra = tmp_path / "extra_nodes"
    extra.write_text("")
    mark_dir = tmp_path / "paused"
    mark_dir.mkdir()
    (mark_dir / "stale_node").write_text(str(int(time.time()) - 9999))
    (mark_dir / "fresh_node").write_text(str(int(time.time())))
    env = {"POOL_EXTRA_NODES_FILE": str(extra), "POOL_PAUSE_MARK_DIR": str(mark_dir),
           "POOL_PAUSE_MAX_WINDOW_S": "600"}
    res = _lib_run(["--reregister-stale"], extra_env=env, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert extra.read_text().splitlines() == ["stale_node"]
    assert not (mark_dir / "stale_node").exists()
    assert (mark_dir / "fresh_node").exists()


# =================================================================================== (C) kill-mid-sync (the ask)

def _describe_json(instances):
    return json.dumps({"Reservations": [{"Instances": instances}]})


def _instance(instance_id, itype, state, hours_ago):
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    return {
        "InstanceId": instance_id, "InstanceType": itype, "State": {"Name": state},
        "LaunchTime": (now - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "Tags": [{"Key": "Project", "Value": "neural-sim"}],
    }


def _chmod_x(p):
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _make_slow_sync_stub_bin(tmp_path, sleep_s=4):
    """aws/ssh answer instantly (one running, idle, no-runner instance); rsync SLEEPS `sleep_s` seconds after
    recording its own pid -- giving a test a real window to observe the pause, then kill the script mid-sync."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    describe_fixture = tmp_path / "describe.json"
    cw_fixture = tmp_path / "cw.json"
    cw_fixture.write_text(json.dumps({"Datapoints": [{"Average": 1.0}, {"Average": 2.0}]}))
    aws_log = tmp_path / "aws.log"
    aws_log.write_text("")

    aws_stub = bin_dir / "aws"
    aws_stub.write_text(f"""#!/usr/bin/env bash
echo "$*" >> "{aws_log}"
if [[ "$*" == *"cloudwatch get-metric-statistics"* ]]; then cat "{cw_fixture}"; exit 0; fi
if [[ "$*" == *"ec2 describe-instances"* ]]; then cat "{describe_fixture}"; exit 0; fi
if [[ "$*" == *"PublicIpAddress"* ]]; then echo "5.6.7.8"; exit 0; fi
if [[ "$*" == *"ec2 stop-instances"* ]]; then echo stopped; exit 0; fi
echo "sg-stub"; exit 0
""")
    _chmod_x(aws_stub)

    ssh_stub = bin_dir / "ssh"
    ssh_stub.write_text("""#!/usr/bin/env bash
case "$*" in
  *pgrep*) exit 1 ;;
  *"if [ -d "*) echo has_raw ;;
  *) exit 0 ;;
esac
""")
    _chmod_x(ssh_stub)

    pidfile = tmp_path / "rsync.pid"
    rsync_stub = bin_dir / "rsync"
    rsync_stub.write_text(f"""#!/usr/bin/env bash
echo $$ > "{pidfile}"
sleep {sleep_s}
exit 0
""")
    _chmod_x(rsync_stub)
    return bin_dir, describe_fixture, aws_log, pidfile


def _write_gpu_state(tmp_path, instance_id, key_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir(exist_ok=True)
    Path(key_path).write_text("fake key\n")
    (state_dir / ".aws_gpu").write_text(f"instance={instance_id}\nregion=us-east-1\nkey={key_path}\nsg=sg-stub\n")
    return state_dir


def _wait_until(predicate, timeout_s=10, interval_s=0.05):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return predicate()


def test_idle_stop_sigkill_mid_sync_leaves_a_stale_marker_a_later_cycle_reregisters(tmp_path):
    # THE INCIDENT, REPRODUCED (2026-09-25 review, MEDIUM): SIGKILL cannot be caught by ANY trap, so this proves
    # the DURABLE MARKER mechanism specifically -- not the trap -- is what recovers the node.
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=1)
    bin_dir, describe_fixture, aws_log, rsync_pidfile = _make_slow_sync_stub_bin(tmp_path, sleep_s=5)
    describe_fixture.write_text(_describe_json([inst]))
    state_dir = _write_gpu_state(tmp_path, "i-aaa", tmp_path / "key.pem")
    extra = tmp_path / "extra_nodes"
    extra.write_text("gpu\n")
    mark_dir = tmp_path / "paused"

    env = {
        **os.environ, "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}", "PYTHON": sys.executable,
        "AWS_IDLE_STOP_LOG": str(tmp_path / "idle.log"), "AWS_GPU_STATE_FILE": str(state_dir / ".aws_gpu"),
        "POOL_SSH_CONFIG": str(tmp_path / "no-such-pool-ssh-config"),
        "POOL_EXTRA_NODES_FILE": str(extra), "POOL_PAUSE_MARK_DIR": str(mark_dir),
    }
    proc = subprocess.Popen(["bash", str(AWS_IDLE_STOP)], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        # Wait for the pause to actually take effect (node removed from dispatch) AND for the sync (our slow
        # rsync) to genuinely be in flight -- both must be true before "mid-sync" means anything.
        paused = _wait_until(lambda: "gpu" not in extra.read_text().split(), timeout_s=10)
        assert paused, "the node was never paused before the kill -- test setup is wrong"
        in_sync = _wait_until(rsync_pidfile.exists, timeout_s=10)
        assert in_sync, "the sync (rsync) never started -- test setup is wrong"
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)

    # The process is dead WITHOUT ever running its own resume: the node stays out of dispatch...
    assert "gpu" not in extra.read_text().split(), "node must still be unregistered right after the kill"
    # ...but a DURABLE marker was left behind recording that a resume is still owed.
    assert (mark_dir / "gpu").exists(), "no durable marker survived the kill -- the node is now stuck forever"

    # A LATER cycle (any entry point) notices the stale marker and self-heals. POOL_PAUSE_MAX_WINDOW_S=0 makes
    # ANY marker read as stale without needing to actually wait out the real window in a test.
    describe_fixture.write_text(_describe_json([]))   # nothing running this cycle -- isolates the assertion
    env2 = {**env, "POOL_PAUSE_MAX_WINDOW_S": "0"}
    res2 = subprocess.run(["bash", str(AWS_IDLE_STOP)], cwd=ROOT, env=env2, stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, timeout=15)
    assert res2.returncode == 0, res2.stderr
    assert extra.read_text().splitlines() == ["gpu"], "the next cycle must re-register the orphaned node"
    assert not (mark_dir / "gpu").exists(), "the marker must be cleared once the node is re-registered"


def test_idle_stop_sigterm_mid_sync_trap_resumes_immediately(tmp_path):
    # The GRACEFUL-kill half of the fix: SIGTERM IS trappable, so the trap should restore registration and clear
    # the marker itself, immediately -- with no need to wait for a later cycle's staleness check at all.
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=1)
    bin_dir, describe_fixture, aws_log, rsync_pidfile = _make_slow_sync_stub_bin(tmp_path, sleep_s=5)
    describe_fixture.write_text(_describe_json([inst]))
    state_dir = _write_gpu_state(tmp_path, "i-aaa", tmp_path / "key.pem")
    extra = tmp_path / "extra_nodes"
    extra.write_text("gpu\n")
    mark_dir = tmp_path / "paused"

    env = {
        **os.environ, "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}", "PYTHON": sys.executable,
        "AWS_IDLE_STOP_LOG": str(tmp_path / "idle.log"), "AWS_GPU_STATE_FILE": str(state_dir / ".aws_gpu"),
        "POOL_SSH_CONFIG": str(tmp_path / "no-such-pool-ssh-config"),
        "POOL_EXTRA_NODES_FILE": str(extra), "POOL_PAUSE_MARK_DIR": str(mark_dir),
    }
    proc = subprocess.Popen(["bash", str(AWS_IDLE_STOP)], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        paused = _wait_until(lambda: "gpu" not in extra.read_text().split(), timeout_s=10)
        assert paused, "the node was never paused before the signal -- test setup is wrong"
        in_sync = _wait_until(rsync_pidfile.exists, timeout_s=10)
        assert in_sync, "the sync (rsync) never started -- test setup is wrong"
        os.kill(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)

    assert extra.read_text().splitlines() == ["gpu"], (
        "the TRAP must restore registration immediately on SIGTERM, not leave it for a later stale-check")
    assert not (mark_dir / "gpu").exists(), "the trap must clear the marker it resumed, immediately"


def _budget_slow_sync_stub_bin(tmp_path, sleep_s=5):
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    describe_fixture = tmp_path / "describe.json"
    aws_stub = bin_dir / "aws"
    aws_stub.write_text(f"""#!/usr/bin/env bash
if [[ "$*" == *"PublicIpAddress"* ]]; then echo "5.6.7.8"; exit 0; fi
if [[ "$*" == *"ec2 describe-instances"* ]]; then cat "{describe_fixture}"; exit 0; fi
if [[ "$*" == *"ec2 stop-instances"* ]]; then echo stopped; exit 0; fi
echo "sg-stub"; exit 0
""")
    _chmod_x(aws_stub)
    ssh_stub = bin_dir / "ssh"
    ssh_stub.write_text("""#!/usr/bin/env bash
case "$*" in
  *pgrep*) exit 1 ;;
  *"if [ -d "*) echo has_raw ;;
  *) exit 0 ;;
esac
""")
    _chmod_x(ssh_stub)
    pidfile = tmp_path / "rsync.pid"
    rsync_stub = bin_dir / "rsync"
    rsync_stub.write_text(f"""#!/usr/bin/env bash
echo $$ > "{pidfile}"
sleep {sleep_s}
exit 0
""")
    _chmod_x(rsync_stub)
    return bin_dir, describe_fixture, pidfile


def test_budget_enforce_sigkill_mid_sync_leaves_a_stale_marker_a_later_call_reregisters(tmp_path):
    # Same durable-restore requirement, the OTHER caller (tools/aws_budget.sh enforce) -- the review named it
    # explicitly ("every entry point ... aws_budget enforce").
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=40)   # well over any small cap
    bin_dir, describe_fixture, rsync_pidfile = _budget_slow_sync_stub_bin(tmp_path, sleep_s=5)
    describe_fixture.write_text(_describe_json([inst]))
    state_dir = tmp_path / "state"; state_dir.mkdir()
    key = tmp_path / "key.pem"; key.write_text("fake key\n")
    (state_dir / ".aws_gpu").write_text(f"instance=i-aaa\nregion=us-east-1\nkey={key}\nsg=sg-stub\n")
    extra = tmp_path / "extra_nodes"; extra.write_text("gpu\n")
    mark_dir = tmp_path / "paused"

    env = {
        **os.environ, "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}", "PYTHON": sys.executable,
        "AWS_DAILY_CAP_USD": "1", "AWS_BUDGET_LOG": str(tmp_path / "budget.log"),
        "AWS_SPEND_LEDGER": str(tmp_path / "ledger.jsonl"), "AWS_NODE_STATE_DIR": str(state_dir),
        "POOL_SSH_CONFIG": str(tmp_path / "no-such-pool-ssh-config"),
        "POOL_EXTRA_NODES_FILE": str(extra), "POOL_PAUSE_MARK_DIR": str(mark_dir),
        "AWS_BUDGET_SYNC_TIMEOUT_S": "120",
    }
    proc = subprocess.Popen(["bash", str(AWS_BUDGET), "enforce"], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        paused = _wait_until(lambda: "gpu" not in extra.read_text().split(), timeout_s=10)
        assert paused, "the node was never paused before the kill -- test setup is wrong"
        in_sync = _wait_until(rsync_pidfile.exists, timeout_s=10)
        assert in_sync, "the sync (rsync) never started -- test setup is wrong"
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)

    assert "gpu" not in extra.read_text().split()
    assert (mark_dir / "gpu").exists(), "no durable marker survived the kill"

    describe_fixture.write_text(_describe_json([]))
    env2 = {**env, "POOL_PAUSE_MAX_WINDOW_S": "0"}
    res2 = subprocess.run(["bash", str(AWS_BUDGET), "status"], cwd=ROOT, env=env2, stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, timeout=15)
    assert res2.returncode == 0, res2.stderr
    assert extra.read_text().splitlines() == ["gpu"]
    assert not (mark_dir / "gpu").exists()


# =================================================================================== (D) other entry points

def test_aws_pool_node_start_reregisters_a_stale_paused_node(tmp_path):
    # `start` refreshes ONLY when the instance is already running (no budget check, no real spend) -- this
    # exercises that cheap path while proving the stale-pause self-heal runs before it.
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    aws_stub = bin_dir / "aws"
    aws_stub.write_text("""#!/usr/bin/env bash
case "$*" in
  *"State.Name"*) echo running ;;
  *"PublicIpAddress"*) echo "9.9.9.9" ;;
esac
exit 0
""")
    _chmod_x(aws_stub)
    ssh_stub = bin_dir / "ssh"
    ssh_stub.write_text("#!/usr/bin/env bash\nexit 0\n")
    _chmod_x(ssh_stub)
    key = tmp_path / "key.pem"; key.write_text("fake key\n")
    state = tmp_path / ".aws_pool1"
    state.write_text(f"instance=i-aaa\nregion=us-east-1\nkey={key}\nsg=sg-stub\n")
    ssh_cfg = tmp_path / "ssh_config"; ssh_cfg.write_text("Include ~/.ssh/config\n")
    extra = tmp_path / "extra_nodes"; extra.write_text("")
    mark_dir = tmp_path / "paused"; mark_dir.mkdir()
    (mark_dir / "pool1").write_text(str(int(time.time()) - 9999))   # ancient -- stale

    env = {
        **os.environ, "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        "AWS_POOL_NODE_STATE_FILE": str(state), "POOL_SSH_CONFIG": str(ssh_cfg),
        "POOL_EXTRA_NODES_FILE": str(extra), "POOL_PAUSE_MARK_DIR": str(mark_dir),
        "POOL_PAUSE_MAX_WINDOW_S": "0",
        "AWS_BUDGET_LOG": str(tmp_path / "budget.log"), "AWS_SPEND_LEDGER": str(tmp_path / "ledger.jsonl"),
    }
    res = subprocess.run(["bash", str(AWS_POOL_NODE), "start", "pool1"], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    assert extra.read_text().splitlines() == ["pool1"], "start must re-register a stale-paused node"
    assert not (mark_dir / "pool1").exists()


def test_aws_pool_node_refresh_reregisters_a_stale_paused_node(tmp_path):
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    aws_stub = bin_dir / "aws"
    aws_stub.write_text("""#!/usr/bin/env bash
case "$*" in
  *"State.Name"*) echo running ;;
  *"PublicIpAddress"*) echo "9.9.9.9" ;;
esac
exit 0
""")
    _chmod_x(aws_stub)
    key = tmp_path / "key.pem"; key.write_text("fake key\n")
    state = tmp_path / ".aws_pool1"
    state.write_text(f"instance=i-aaa\nregion=us-east-1\nkey={key}\nsg=sg-stub\n")
    ssh_cfg = tmp_path / "ssh_config"; ssh_cfg.write_text("Include ~/.ssh/config\n")
    extra = tmp_path / "extra_nodes"; extra.write_text("")
    mark_dir = tmp_path / "paused"; mark_dir.mkdir()
    (mark_dir / "pool1").write_text(str(int(time.time()) - 9999))

    env = {
        **os.environ, "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        "AWS_POOL_NODE_STATE_FILE": str(state), "POOL_SSH_CONFIG": str(ssh_cfg),
        "POOL_EXTRA_NODES_FILE": str(extra), "POOL_PAUSE_MARK_DIR": str(mark_dir),
        "POOL_PAUSE_MAX_WINDOW_S": "0",
        "AWS_BUDGET_LOG": str(tmp_path / "budget.log"), "AWS_SPEND_LEDGER": str(tmp_path / "ledger.jsonl"),
    }
    res = subprocess.run(["bash", str(AWS_POOL_NODE), "refresh", "pool1"], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    assert extra.read_text().splitlines() == ["pool1"]
    assert not (mark_dir / "pool1").exists()


def test_dispatcher_cycle_setup_reregisters_a_stale_paused_node(tmp_path):
    # cycle_setup (run every dispatch cycle, via --fill-node/--print-ssh-f-loop/the real loop) must self-heal
    # a stale pause on its own, with no aws_idle_stop/aws_budget cycle involved at all.
    queue = tmp_path / "pool.queue"
    queue.write_text("")   # empty -- nothing to dispatch, isolates the assertion to registration alone
    extra = tmp_path / "extra_nodes"; extra.write_text("")
    mark_dir = tmp_path / "paused"; mark_dir.mkdir()
    (mark_dir / "pool1").write_text(str(int(time.time()) - 9999))
    env = {
        **os.environ, "POOL_QUEUE_PATH": str(queue), "POOL_RUNNING_PATH": str(tmp_path / "pool.running"),
        "POOL_RESERVATIONS_PATH": str(tmp_path / "resv"), "POOL_SSH_CONFIG": str(tmp_path / "does-not-exist"),
        "POOL_EXTRA_NODES_FILE": str(extra), "POOL_PAUSE_MARK_DIR": str(mark_dir), "POOL_PAUSE_MAX_WINDOW_S": "0",
        "PATH": f"{tmp_path}/bin:{os.environ.get('PATH', '')}",
    }
    (tmp_path / "bin").mkdir()
    res = subprocess.run(["bash", str(DISPATCHER), "--print-ssh-f-loop", "1", "0"], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=15)
    assert res.returncode == 0, res.stderr
    assert extra.read_text().splitlines() == ["pool1"]
    assert not (mark_dir / "pool1").exists()


def test_fill_node_stops_filling_the_instant_a_pause_marker_appears(tmp_path):
    # THE RACE (2026-09-25 review, LOW: "fill_node never re-reads registration -- a cycle already running keeps
    # filling the node through the whole pause"). CYCLE_NODES is fixed for the whole cycle, but the pause MARKER
    # is checked on every iteration -- a node paused mid-cycle must stop being filled immediately, without
    # needing node_is_idle's ssh probe to run at all (proven by an ssh stub that would otherwise report idle).
    now = int(time.time())
    queue = tmp_path / "pool.queue"
    queue.write_text(f"{now}\techo job1  #checked:r mem_gb=1\n{now}\techo job2  #checked:r mem_gb=1\n")
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    ssh_log = tmp_path / "ssh.log"; ssh_log.write_text("")
    ssh_stub = bin_dir / "ssh"
    # Would ALWAYS report idle+roomy if actually called -- the pause-marker check must short-circuit BEFORE this.
    ssh_stub.write_text(f"""#!/usr/bin/env bash
echo "$*" >> "{ssh_log}"
case "$*" in
  *MemAvailable*) echo "8 1 0 20 1 32" ;;
esac
exit 0
""")
    _chmod_x(ssh_stub)
    mark_dir = tmp_path / "paused"; mark_dir.mkdir()
    (mark_dir / "pool1").write_text(str(int(time.time())))   # FRESH -- a real pause in progress right now

    env = {
        **os.environ, "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        "POOL_QUEUE_PATH": str(queue), "POOL_RUNNING_PATH": str(tmp_path / "pool.running"),
        "POOL_RESERVATIONS_PATH": str(tmp_path / "resv"), "POOL_SSH_CONFIG": str(tmp_path / "does-not-exist"),
        "POOL_EXTRA_NODES_FILE": str(tmp_path / "extra_nodes"), "POOL_PAUSE_MARK_DIR": str(mark_dir),
        "POOL_PAUSE_MAX_WINDOW_S": "600", "POOL_DISPATCH_LAUNCH_SLEEP": "0",
    }
    res = subprocess.run(["bash", str(DISPATCHER), "--fill-node", "pool1"], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=15)
    assert res.returncode == 0, res.stderr
    assert queue.read_text().count("#checked:") == 2, "both jobs must remain queued -- the paused node must not be filled"
    assert ssh_log.read_text() == "", "node_is_idle's ssh probe must never even run once the pause marker is present"


# =================================================================================== (E) shared lock

def test_pause_removal_is_serialized_by_the_shared_extra_nodes_lock(tmp_path):
    # "Lock .pool_extra_nodes writes with one shared flock across all writers" -- an external holder of
    # "$EXTRA_NODES_FILE.lock" must make pause_dispatch_for_node genuinely WAIT, exactly like
    # tests/test_aws_pool_node_workflow.py's own ssh_config.lock tests prove for _write_host_block.
    extra = tmp_path / "extra_nodes"
    extra.write_text("gpu\n")
    lock = tmp_path / "extra_nodes.lock"
    hold_s = 2.0
    holder = subprocess.Popen(["flock", str(lock), "sleep", str(hold_s)])
    try:
        time.sleep(0.4)
        t0 = time.monotonic()
        res = _lib_run(["--pause", "gpu"], extra_env={"POOL_EXTRA_NODES_FILE": str(extra)}, tmp_path=tmp_path)
        elapsed = time.monotonic() - t0
    finally:
        holder.wait(timeout=10)
    assert res.returncode == 0, res.stderr
    assert elapsed >= 1.0, (
        f"--pause returned after only {elapsed:.2f}s while an external holder had the SAME lock for {hold_s}s")
    assert res.stdout.strip() == "1"
    assert extra.read_text() == ""


def test_aws_pool_node_down_unregister_is_serialized_by_the_shared_extra_nodes_lock(tmp_path):
    # aws_pool_node.sh's `down` now goes through the SAME _extra_nodes_remove (sourced from
    # tools/aws_stop_safety_lib.sh) pause/resume use -- proves the "one shared flock across ALL writers" ask,
    # not just the pause/resume pair.
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    aws_stub = bin_dir / "aws"
    aws_stub.write_text("#!/usr/bin/env bash\necho ok\nexit 0\n")
    _chmod_x(aws_stub)
    ssh_stub = bin_dir / "ssh"
    ssh_stub.write_text('#!/usr/bin/env bash\ncase "$*" in *pgrep*) echo 0 ;; esac\nexit 0\n')
    _chmod_x(ssh_stub)
    rsync_stub = bin_dir / "rsync"
    rsync_stub.write_text("#!/usr/bin/env bash\nexit 0\n")
    _chmod_x(rsync_stub)
    key = tmp_path / "key.pem"; key.write_text("fake key\n")
    state = tmp_path / ".aws_testnode"
    state.write_text(f"instance=i-aaa\nregion=us-east-1\nkey={key}\nsg=sg-aaa\n")
    extra = tmp_path / "extra_nodes"; extra.write_text("testnode\n")
    lock = tmp_path / "extra_nodes.lock"
    hold_s = 2.0
    holder = subprocess.Popen(["flock", str(lock), "sleep", str(hold_s)])
    try:
        time.sleep(0.4)
        t0 = time.monotonic()
        env = {
            **os.environ, "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
            "AWS_POOL_NODE_STATE_FILE": str(state), "POOL_EXTRA_NODES_FILE": str(extra),
            "POOL_SSH_CONFIG": str(tmp_path / "no-such-config"),
            # Never let aws_pool_node.sh's startup reregister_stale_paused_nodes call scan the SHARED production
            # research/queue/.pool_paused (other sessions/agents may have live markers there).
            "POOL_PAUSE_MARK_DIR": str(tmp_path / "paused"),
            "AWS_BUDGET_LOG": str(tmp_path / "budget.log"), "AWS_SPEND_LEDGER": str(tmp_path / "ledger.jsonl"),
        }
        res = subprocess.run(["bash", str(AWS_POOL_NODE), "down", "testnode"], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=30)
        elapsed = time.monotonic() - t0
    finally:
        holder.wait(timeout=10)
    assert res.returncode == 0, res.stderr
    assert elapsed >= 1.0, (
        f"`down` returned after only {elapsed:.2f}s while an external holder had the SAME .pool_extra_nodes.lock "
        f"for {hold_s}s -- its unregister step did not wait for the shared lock")
    assert "testnode" not in extra.read_text().split()
