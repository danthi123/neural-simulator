"""Subprocess-level tests for the AWS budget guard's bash wrappers (tools/aws_budget.sh,
tools/aws_idle_stop.sh, and the pre-launch gate wired into tools/aws_cpu_launch.sh / tools/aws_gpu.sh
`launch`), with a STUBBED `aws` and `ssh` on PATH -- no real AWS/network calls. Complements
tests/test_aws_cost_lib.py, which covers the pure arithmetic/decision functions directly.

The stub `aws` binary is a small dispatcher: it reads its own argv, returns canned JSON for the
`describe-instances` / `get-metric-statistics` calls the guard makes, and LOGS every invocation (one line per
call) to a file the test can inspect afterwards -- that log is how we prove a refused launch never reached
`run-instances`, and how we prove `enforce`/`aws_idle_stop.sh` really called `stop-instances` (or didn't).
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AWS_BUDGET = ROOT / "tools" / "aws_budget.sh"
AWS_IDLE_STOP = ROOT / "tools" / "aws_idle_stop.sh"


def _describe_json(instances):
    return json.dumps({"Reservations": [{"Instances": instances}]})


def _instance(instance_id, itype, state, hours_ago, project=True, name_tag=None):
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    tags = []
    if project:
        tags.append({"Key": "Project", "Value": "neural-sim"})
    if name_tag:
        tags.append({"Key": "Name", "Value": name_tag})
    return {
        "InstanceId": instance_id,
        "InstanceType": itype,
        "State": {"Name": state},
        "LaunchTime": (now - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "Tags": tags,
    }


_AWS_STUB_TEMPLATE = r"""#!/usr/bin/env bash
# Stub `aws` CLI for tests: logs every invocation, answers describe-instances/get-metric-statistics from
# fixture files, and answers everything else (stop-instances, run-instances, ...) with a canned OK so the
# calling script's happy path completes -- the point is to observe WHICH calls were made, via the log.
echo "$*" >> "{log}"
case "$*" in
  *"describe-instances"*"CPUUtilization"*) ;; # unreachable; get-metric-statistics matched below first
esac
if [[ "$*" == *"cloudwatch get-metric-statistics"* ]]; then
  cat "{cw_fixture}"
  exit 0
fi
if [[ "$*" == *"ec2 describe-instances"* ]]; then
  cat "{describe_fixture}"
  exit 0
fi
if [[ "$*" == *"ec2 stop-instances"* ]]; then
  echo "stopped"
  exit 0
fi
if [[ "$*" == *"ec2 run-instances"* ]]; then
  echo "i-shouldnothappen"
  exit 0
fi
echo "sg-stub-000"
exit 0
"""

_SSH_STUB_NOT_NEEDED = r"""#!/usr/bin/env bash
echo "$*" >> "{log}"
exit 1
"""


def _make_stub_bin(tmp_path, describe_instances, cw_datapoints=None, ssh_pgrep_finds_runner=False,
                    ssh_reachable=True):
    """Build a temp bin/ dir with stub `aws` (+ `ssh`) executables and return (bin_dir, aws_log, ssh_log)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    describe_fixture = tmp_path / "describe.json"
    describe_fixture.write_text(_describe_json(describe_instances))
    cw_fixture = tmp_path / "cw.json"
    cw_fixture.write_text(json.dumps({"Datapoints": [{"Average": a} for a in (cw_datapoints or [])]}))
    aws_log = tmp_path / "aws.log"
    aws_log.write_text("")
    ssh_log = tmp_path / "ssh.log"
    ssh_log.write_text("")

    aws_stub = bin_dir / "aws"
    aws_stub.write_text(_AWS_STUB_TEMPLATE.format(
        log=aws_log, cw_fixture=cw_fixture, describe_fixture=describe_fixture))
    aws_stub.chmod(aws_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    ssh_stub = bin_dir / "ssh"
    if ssh_reachable:
        # `uptime` -> low load; `nproc` -> 8; pgrep -> exit 0 (found) or 1 (not found) per ssh_pgrep_finds_runner
        pgrep_rc = 0 if ssh_pgrep_finds_runner else 1
        ssh_stub.write_text(f"""#!/usr/bin/env bash
echo "$*" >> "{ssh_log}"
case "$*" in
  *uptime*) echo " 12:00:00 up 1 day,  1 user,  load average: 0.05, 0.10, 0.10" ;;
  *nproc*) echo 8 ;;
  *pgrep*) exit {pgrep_rc} ;;
  *) exit 0 ;;
esac
""")
    else:
        ssh_stub.write_text(_SSH_STUB_NOT_NEEDED.format(log=ssh_log))
    ssh_stub.chmod(ssh_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    return bin_dir, aws_log, ssh_log


def _run(script, args, bin_dir, extra_env=None, tmp_path=None):
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["PYTHON"] = sys.executable
    if tmp_path is not None:
        # Never let a test write into the SHARED production research/queue/ (other sessions touch it
        # concurrently) -- redirect every log/state path these scripts support overriding.
        env.setdefault("AWS_BUDGET_LOG", str(tmp_path / "aws_budget.log"))
        env.setdefault("AWS_IDLE_STOP_LOG", str(tmp_path / "aws_idle_stop.log"))
        env.setdefault("AWS_GPU_STATE_FILE", str(tmp_path / "state" / ".aws_gpu"))
    if extra_env:
        env.update(extra_env)
    return subprocess.run(["bash", str(script), *args], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=30)


def _write_gpu_state(tmp_path, instance_id, key_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir(exist_ok=True)
    key_path.write_text("fake key material\n")
    (state_dir / ".aws_gpu").write_text(
        f"instance={instance_id}\nregion=us-east-1\nkey={key_path}\nsg=sg-stub-000\n"
    )


# --------------------------------------------------------------------------------------------- aws_budget.sh

def test_budget_status_reports_a_running_project_instance(tmp_path):
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=1)
    bin_dir, _aws_log, _ = _make_stub_bin(tmp_path, [inst])
    res = _run(AWS_BUDGET, ["status"], bin_dir, {"AWS_DAILY_CAP_USD": "50"}, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "i-aaa" in res.stdout
    assert "cap=$50.00" in res.stdout


def test_budget_check_allows_under_cap(tmp_path):
    bin_dir, _aws_log, _ = _make_stub_bin(tmp_path, [])
    res = _run(AWS_BUDGET, ["check", "r7i.4xlarge"], bin_dir, {"AWS_DAILY_CAP_USD": "50"}, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr


def test_budget_check_refuses_over_cap(tmp_path):
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=40)  # big accrued cost
    bin_dir, _aws_log, _ = _make_stub_bin(tmp_path, [inst])
    res = _run(AWS_BUDGET, ["check", "r7i.4xlarge"], bin_dir, {"AWS_DAILY_CAP_USD": "1"}, tmp_path=tmp_path)
    assert res.returncode == 1
    assert "refusing" in res.stderr


def test_budget_enforce_stops_running_instance_over_cap(tmp_path):
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=40)
    bin_dir, aws_log, _ = _make_stub_bin(tmp_path, [inst])
    res = _run(AWS_BUDGET, ["enforce"], bin_dir, {"AWS_DAILY_CAP_USD": "1"}, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    log_text = aws_log.read_text()
    assert "ec2 stop-instances" in log_text
    assert "i-aaa" in log_text


def test_budget_enforce_does_not_stop_when_under_cap(tmp_path):
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=1)
    bin_dir, aws_log, _ = _make_stub_bin(tmp_path, [inst])
    res = _run(AWS_BUDGET, ["enforce"], bin_dir, {"AWS_DAILY_CAP_USD": "1000"}, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "ec2 stop-instances" not in aws_log.read_text()


# --------------------------------------------------------------------------------- launch-path gate wiring

def test_aws_cpu_launch_refused_by_budget_never_calls_run_instances(tmp_path):
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=40)
    bin_dir, aws_log, _ = _make_stub_bin(tmp_path, [inst])
    # aws_cpu_launch.sh's OWN "already recorded" check reads the SHARED $ROOT/research/queue/.aws_gpu
    # (it has no override, by design -- that anti-leak guard must always look at the real, single lane
    # state file). This test doesn't touch that file; it only proves the budget gate -- which DOES accept
    # an override -- refuses before `run-instances` is ever called.
    res = _run(ROOT / "tools" / "aws_cpu_launch.sh", [], bin_dir, {"AWS_DAILY_CAP_USD": "1"}, tmp_path=tmp_path)
    assert res.returncode == 1
    assert "refused by tools/aws_budget.sh" in res.stdout + res.stderr
    assert "ec2 run-instances" not in aws_log.read_text()


def test_aws_gpu_launch_refused_by_budget_never_calls_run_instances(tmp_path):
    inst = _instance("i-aaa", "g5.xlarge", "running", hours_ago=40, project=False, name_tag="claude-gpu-verify")
    bin_dir, aws_log, _ = _make_stub_bin(tmp_path, [inst])
    res = _run(ROOT / "tools" / "aws_gpu.sh", ["launch"], bin_dir, {"AWS_DAILY_CAP_USD": "1"}, tmp_path=tmp_path)
    assert res.returncode == 1
    assert "refused by tools/aws_budget.sh" in res.stdout + res.stderr
    assert "ec2 run-instances" not in aws_log.read_text()


# ----------------------------------------------------------------------------------------- aws_idle_stop.sh

def test_idle_stop_stops_idle_instance_with_no_runner(tmp_path):
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=1)
    bin_dir, aws_log, ssh_log = _make_stub_bin(tmp_path, [inst], cw_datapoints=[1.0, 2.0],
                                                ssh_pgrep_finds_runner=False)
    _write_gpu_state(tmp_path, "i-aaa", tmp_path / "aws_key.pem")
    res = _run(AWS_IDLE_STOP, [], bin_dir, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "ec2 stop-instances" in aws_log.read_text()
    assert "pgrep" in ssh_log.read_text()   # the no-runner check really ran over SSH, not assumed


def test_idle_stop_keeps_instance_when_runner_active(tmp_path):
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=1)
    bin_dir, aws_log, ssh_log = _make_stub_bin(tmp_path, [inst], cw_datapoints=[1.0, 2.0],
                                                ssh_pgrep_finds_runner=True)
    _write_gpu_state(tmp_path, "i-aaa", tmp_path / "aws_key.pem")
    res = _run(AWS_IDLE_STOP, [], bin_dir, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "ec2 stop-instances" not in aws_log.read_text()


def test_idle_stop_keeps_instance_when_cpu_not_idle(tmp_path):
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=1)
    bin_dir, aws_log, ssh_log = _make_stub_bin(tmp_path, [inst], cw_datapoints=[55.0, 60.0],
                                                ssh_pgrep_finds_runner=False)
    _write_gpu_state(tmp_path, "i-aaa", tmp_path / "aws_key.pem")
    res = _run(AWS_IDLE_STOP, [], bin_dir, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "ec2 stop-instances" not in aws_log.read_text()


def test_idle_stop_keeps_instance_when_no_state_file_key_to_verify_runner(tmp_path):
    # CPU reads idle via CloudWatch, but this instance ISN'T the one recorded in the (missing) GPU state
    # file, so there is no known SSH key -> the no-runner check is inconclusive -> the conservative default
    # (assume a runner IS active) must win, and the instance must be kept running.
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=1)
    bin_dir, aws_log, ssh_log = _make_stub_bin(tmp_path, [inst], cw_datapoints=[1.0, 2.0],
                                                ssh_pgrep_finds_runner=False)
    res = _run(AWS_IDLE_STOP, [], bin_dir, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "ec2 stop-instances" not in aws_log.read_text()
    assert ssh_log.read_text() == ""   # never even attempted SSH -- no known key for this instance


def test_idle_stop_no_running_instances_is_a_noop(tmp_path):
    bin_dir, aws_log, _ = _make_stub_bin(tmp_path, [])
    res = _run(AWS_IDLE_STOP, [], bin_dir, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "ec2 stop-instances" not in aws_log.read_text()
