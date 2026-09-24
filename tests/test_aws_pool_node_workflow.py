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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "aws_pool_node.sh"


def _run(args, bin_dir=None, env=None, tmp_path=None):
    full_env = dict(os.environ)
    if bin_dir is not None:
        full_env["PATH"] = f"{bin_dir}:{full_env.get('PATH', '')}"
    if tmp_path is not None:
        full_env.setdefault("AWS_BUDGET_LOG", str(tmp_path / "aws_budget.log"))
        full_env.setdefault("AWS_SPEND_LEDGER", str(tmp_path / "aws_spend_ledger.jsonl"))
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
    ssh_stub.write_text(_STUB.format(tag="SSH", log=log, body='exit 0'))
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
