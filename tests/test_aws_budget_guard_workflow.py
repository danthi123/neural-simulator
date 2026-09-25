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

sys.path.insert(0, str(ROOT / "tools"))
import aws_cost_lib as cost_lib  # noqa: E402  (computes EXPECTED spend dynamically for the ledger test below,
# so that test's assertions never depend on what real-world time-of-day it happens to run at)


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
                    ssh_reachable=True, rsync_ok=True):
    """Build a temp bin/ dir with stub `aws` (+ `ssh` + `rsync`) executables and return (bin_dir, aws_log,
    ssh_log, rsync_log). `rsync_ok` governs aws_idle_stop.sh's sync-before-stop fallback pull -- `_run` below
    points POOL_SSH_CONFIG at a nonexistent file by default, so every test here exercises the direct-rsync
    fallback path (never the pool_sync.sh/.pool_ssh_config-alias path, covered directly in
    tests/test_aws_pool_node_workflow.py and tests/test_pool_ssh_config_plumbing.py)."""
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
    rsync_log = tmp_path / "rsync.log"
    rsync_log.write_text("")

    aws_stub = bin_dir / "aws"
    aws_stub.write_text(_AWS_STUB_TEMPLATE.format(
        log=aws_log, cw_fixture=cw_fixture, describe_fixture=describe_fixture))
    aws_stub.chmod(aws_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    ssh_stub = bin_dir / "ssh"
    if ssh_reachable:
        # `uptime` -> low load; `nproc` -> 8; pgrep -> exit 0 (found) or 1 (not found) per ssh_pgrep_finds_runner;
        # the `if [ -d ...` remote probe -> sync_node_before_stop's HIGH #1 fallback directory check (default:
        # answer "has_raw" for EVERY candidate, i.e. every known project layout exists and has results -- tests
        # that care about a SPECIFIC layout build their own stub instead, see test_idle_stop_fallback_* below).
        pgrep_rc = 0 if ssh_pgrep_finds_runner else 1
        ssh_stub.write_text(f"""#!/usr/bin/env bash
echo "$*" >> "{ssh_log}"
case "$*" in
  *uptime*) echo " 12:00:00 up 1 day,  1 user,  load average: 0.05, 0.10, 0.10" ;;
  *nproc*) echo 8 ;;
  *pgrep*) exit {pgrep_rc} ;;
  *"if [ -d "*) echo has_raw ;;
  *) exit 0 ;;
esac
""")
    else:
        ssh_stub.write_text(_SSH_STUB_NOT_NEEDED.format(log=ssh_log))
    ssh_stub.chmod(ssh_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    rsync_stub = bin_dir / "rsync"
    rsync_rc = 0 if rsync_ok else 1
    rsync_stub.write_text(f"""#!/usr/bin/env bash
echo "$*" >> "{rsync_log}"
exit {rsync_rc}
""")
    rsync_stub.chmod(rsync_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    return bin_dir, aws_log, ssh_log, rsync_log


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
        # aws_budget.sh's status/check/enforce now RECORD to the spend ledger (tools/aws_spend_ledger.py) --
        # isolate it too, same reasoning as the three overrides above.
        env.setdefault("AWS_SPEND_LEDGER", str(tmp_path / "aws_spend_ledger.jsonl"))
        # aws_idle_stop.sh's sync-before-stop (2026-09-25) prefers a pool_sync.sh/.pool_ssh_config-alias pull
        # when the stopping node has a registered dispatch Host entry -- point at a file that never exists so
        # every test here takes the direct-rsync fallback path instead (deterministic, no real ssh config to
        # accidentally match against the SHARED production one).
        env.setdefault("POOL_SSH_CONFIG", str(tmp_path / "no-such-pool-ssh-config"))
        # pause_dispatch_for_node/resume_dispatch_for_node (2026-09-25 fix round 3, tools/aws_stop_safety_lib.sh)
        # read/write this file every idle-stop/enforce cycle -- must never touch the SHARED production
        # research/queue/.pool_extra_nodes (other sessions/agents read it too).
        env.setdefault("POOL_EXTRA_NODES_FILE", str(tmp_path / "no-such-pool-extra-nodes"))
        # aws_budget.sh enforce's per-instance node/key lookup (2026-09-25 fix round 3) globs this directory's
        # `.aws_*` state files -- must never resolve to the SHARED production research/queue/.
        env.setdefault("AWS_NODE_STATE_DIR", str(tmp_path / "state"))
    if extra_env:
        env.update(extra_env)
    # stdin=DEVNULL (2026-09-25, HIGH #2 test coverage): matches how this script actually runs in production
    # (systemd --user, stdin /dev/null) and keeps a stub that deliberately reads stdin (see
    # _make_stdin_draining_stub_bin) from hanging on the test runner's own inherited stdin, which is not
    # guaranteed to hit EOF.
    return subprocess.run(["bash", str(script), *args], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
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
    bin_dir, _aws_log, _, _ = _make_stub_bin(tmp_path, [inst])
    res = _run(AWS_BUDGET, ["status"], bin_dir, {"AWS_DAILY_CAP_USD": "50"}, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "i-aaa" in res.stdout
    assert "cap=$50.00" in res.stdout


def test_budget_check_allows_under_cap(tmp_path):
    bin_dir, _aws_log, _, _ = _make_stub_bin(tmp_path, [])
    res = _run(AWS_BUDGET, ["check", "r7i.4xlarge"], bin_dir, {"AWS_DAILY_CAP_USD": "50"}, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr


def test_budget_check_refuses_over_cap(tmp_path):
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=40)  # big accrued cost
    bin_dir, _aws_log, _, _ = _make_stub_bin(tmp_path, [inst])
    res = _run(AWS_BUDGET, ["check", "r7i.4xlarge"], bin_dir, {"AWS_DAILY_CAP_USD": "1"}, tmp_path=tmp_path)
    assert res.returncode == 1
    assert "refusing" in res.stderr


def test_budget_enforce_stops_running_instance_over_cap(tmp_path):
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=40)
    bin_dir, aws_log, _, _ = _make_stub_bin(tmp_path, [inst])
    res = _run(AWS_BUDGET, ["enforce"], bin_dir, {"AWS_DAILY_CAP_USD": "1"}, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    log_text = aws_log.read_text()
    assert "ec2 stop-instances" in log_text
    assert "i-aaa" in log_text


def test_budget_enforce_does_not_stop_when_under_cap(tmp_path):
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=1)
    bin_dir, aws_log, _, _ = _make_stub_bin(tmp_path, [inst])
    res = _run(AWS_BUDGET, ["enforce"], bin_dir, {"AWS_DAILY_CAP_USD": "1000"}, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "ec2 stop-instances" not in aws_log.read_text()


def _budget_stop_stub_bin(tmp_path, ip="5.6.7.8", pgrep_finds_runner=False, rsync_ok=True):
    """Custom aws/ssh/rsync stubs for aws_budget.sh `enforce`'s sync-before-stop path (2026-09-25 fix round 3):
    `aws` answers a REAL PublicIpAddress for the ip-resolution call `enforce` now makes -- the generic
    `_AWS_STUB_TEMPLATE` above ignores `--query`/`--output` entirely (it just dumps the whole describe fixture),
    which is fine for the pre-existing cap tests but not for exercising an actual sync. All three log to ONE
    shared file so ordering assertions (sync strictly before stop) are meaningful."""
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    log = tmp_path / "shared.log"; log.write_text("")
    describe_fixture = tmp_path / "describe.json"; describe_fixture.write_text("{}")
    aws_stub = bin_dir / "aws"
    aws_stub.write_text(f"""#!/usr/bin/env bash
echo "AWS $*" >> "{log}"
if [[ "$*" == *"PublicIpAddress"* ]]; then echo "{ip}"; exit 0; fi
if [[ "$*" == *"ec2 describe-instances"* ]]; then cat "{describe_fixture}"; exit 0; fi
if [[ "$*" == *"ec2 stop-instances"* ]]; then echo stopped; exit 0; fi
echo "sg-stub"; exit 0
""")
    aws_stub.chmod(aws_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    pgrep_rc = 0 if pgrep_finds_runner else 1
    ssh_stub = bin_dir / "ssh"
    ssh_stub.write_text(f"""#!/usr/bin/env bash
echo "SSH $*" >> "{log}"
case "$*" in
  *pgrep*) exit {pgrep_rc} ;;
  *"if [ -d "*) echo has_raw ;;
  *) exit 0 ;;
esac
""")
    ssh_stub.chmod(ssh_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    rsync_rc = 0 if rsync_ok else 1
    rsync_stub = bin_dir / "rsync"
    rsync_stub.write_text(f'#!/usr/bin/env bash\necho "RSYNC $*" >> "{log}"\nexit {rsync_rc}\n')
    rsync_stub.chmod(rsync_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir, log, describe_fixture


def test_budget_enforce_syncs_before_stopping_and_the_stop_happens_after_it(tmp_path):
    # STILL-OPEN ITEM (2026-09-25 review, fix round 3): "aws_budget.sh:72-76 still stops a node at the cap
    # without syncing first". `enforce` now resolves the node's ip/key (from its research/queue/.aws_* state
    # file) and calls the SAME sync_node_before_stop tools/aws_idle_stop.sh uses, strictly before the stop.
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=40)
    bin_dir, shared_log, describe_fixture = _budget_stop_stub_bin(tmp_path)
    describe_fixture.write_text(_describe_json([inst]))
    _write_gpu_state(tmp_path, "i-aaa", tmp_path / "aws_key.pem")
    res = _run(AWS_BUDGET, ["enforce"], bin_dir, {"AWS_DAILY_CAP_USD": "1"}, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    lines = shared_log.read_text().splitlines()
    rsync_idx = next((i for i, ln in enumerate(lines) if ln.startswith("RSYNC ")), None)
    stop_idx = next((i for i, ln in enumerate(lines) if "ec2 stop-instances" in ln), None)
    assert rsync_idx is not None, f"a sync was never attempted before stopping at the cap: {lines}"
    assert stop_idx is not None, f"stop-instances was never called: {lines}"
    assert rsync_idx < stop_idx, f"stop-instances happened before/without a prior sync attempt: {lines}"


def test_budget_enforce_stops_even_when_the_sync_fails(tmp_path):
    # THE DELIBERATE DIFFERENCE from tools/aws_idle_stop.sh's own (blocking) use of sync_node_before_stop: a
    # hard SPEND CAP must never be defeated by a stuck/failing sync -- continuing to run past the cap is exactly
    # what `enforce` exists to prevent. The sync is still ATTEMPTED and its failure is still LOGGED, but the
    # stop proceeds regardless.
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=40)
    bin_dir, shared_log, describe_fixture = _budget_stop_stub_bin(tmp_path, rsync_ok=False)
    describe_fixture.write_text(_describe_json([inst]))
    _write_gpu_state(tmp_path, "i-aaa", tmp_path / "aws_key.pem")
    res = _run(AWS_BUDGET, ["enforce"], bin_dir, {"AWS_DAILY_CAP_USD": "1"}, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    log_text = shared_log.read_text()
    assert "RSYNC " in log_text, "the sync must still have been ATTEMPTED"
    assert "ec2 stop-instances" in log_text, "a hard cap must stop the instance REGARDLESS of a failed sync"
    combined_log = (tmp_path / "aws_budget.log").read_text()
    assert "sync-before-stop failed" in combined_log


def test_budget_enforce_stops_even_with_no_verified_key_for_the_instance(tmp_path):
    # No research/queue/.aws_* state file names this instance at all -- `enforce` must still stop it (the SAME
    # "no verified ssh key/ip on hand" case tools/aws_idle_stop.sh treats as inconclusive-keep, but here there
    # is nothing FOR the cap to keep: the instance is over budget regardless of whether it can be synced).
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=40)
    bin_dir, shared_log, describe_fixture = _budget_stop_stub_bin(tmp_path)
    describe_fixture.write_text(_describe_json([inst]))
    res = _run(AWS_BUDGET, ["enforce"], bin_dir, {"AWS_DAILY_CAP_USD": "1"}, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "ec2 stop-instances" in shared_log.read_text()
    assert "RSYNC " not in shared_log.read_text(), "no key on hand -- must never even attempt a sync"


def test_budget_enforce_takes_node_out_of_dispatch_during_the_sync_and_restores_it(tmp_path):
    # Same still-open item as tools/aws_idle_stop.sh's own fix: the node must be taken OUT of
    # tools/pool_autodispatch.sh's pool for the duration of the sync, and restored once the decision is made.
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=40)
    bin_dir, shared_log, describe_fixture = _budget_stop_stub_bin(tmp_path)
    describe_fixture.write_text(_describe_json([inst]))
    extra_nodes = tmp_path / "extra_nodes"
    extra_nodes.write_text("gpu\n")
    marker = tmp_path / "rsync_saw_extra_nodes"
    rsync_stub = bin_dir / "rsync"
    rsync_stub.write_text(f"""#!/usr/bin/env bash
echo "RSYNC $*" >> "{shared_log}"
if grep -qxF gpu "{extra_nodes}" 2>/dev/null; then echo PRESENT >> "{marker}"; else echo ABSENT >> "{marker}"; fi
exit 0
""")
    rsync_stub.chmod(rsync_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    _write_gpu_state(tmp_path, "i-aaa", tmp_path / "aws_key.pem")
    res = _run(AWS_BUDGET, ["enforce"], bin_dir,
               {"AWS_DAILY_CAP_USD": "1", "POOL_EXTRA_NODES_FILE": str(extra_nodes)}, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    seen = marker.read_text().split()
    assert seen, "no rsync call observed the extra-nodes file -- test setup is wrong"
    assert all(s == "ABSENT" for s in seen), f"node was still registered for dispatch during the sync: {seen}"
    assert extra_nodes.read_text().splitlines() == ["gpu"], "registration was not restored after the cycle"


def test_budget_check_refuses_after_earlier_instance_vanished_from_live_snapshot(tmp_path):
    # Regression for the 2026-09-23 undercount: two SEPARATE `aws_budget.sh check` invocations (as real usage
    # is -- once per launch attempt), sharing one ledger. The first sees TWO instances still live (mirrors the
    # real report: two r7i.4xlarge running ~$12 combined); the second sees only the survivor (i-old has since
    # terminated and dropped out of the describe-instances snapshot entirely). A live-only check would then
    # see only the survivor's spend and wrongly allow a new launch past the cap.
    #
    # The cap is computed DYNAMICALLY from aws_cost_lib's own formula (rather than a number derived assuming a
    # fixed hours-ago) so this test cannot go flaky depending on what time of day it happens to run -- an
    # `hours_ago` near a UTC-midnight boundary would otherwise change the accrued cost out from under a fixed
    # expected constant (this exact mistake was caught in review before this test was committed).
    ledger_path = tmp_path / "aws_spend_ledger.jsonl"
    call1 = tmp_path / "call1"; call1.mkdir()
    call2 = tmp_path / "call2"; call2.mkdir()

    hours_ago = 2
    old = _instance("i-old", "r7i.4xlarge", "running", hours_ago=hours_ago)
    keep = _instance("i-keep", "r7i.4xlarge", "running", hours_ago=hours_ago)
    each_cost, _hrs = cost_lib.instance_cost_today(old)   # ground truth, from the same UTC-day-clamped formula
    combined = 2 * each_cost
    extra = cost_lib.price_per_hour("r7i.4xlarge")
    cap = combined + extra / 2.0   # strictly between "combined" and "combined + a new instance's first hour"

    bin_dir1, _aws_log1, _, _ = _make_stub_bin(call1, [old, keep])
    res1 = _run(AWS_BUDGET, ["check"], bin_dir1,
                {"AWS_DAILY_CAP_USD": f"{cap:.4f}", "AWS_SPEND_LEDGER": str(ledger_path)}, tmp_path=call1)
    assert res1.returncode == 0, res1.stderr   # both still live, combined cost is under cap

    bin_dir2, _aws_log2, _, _ = _make_stub_bin(call2, [keep])  # i-old is now gone -- terminated
    res2 = _run(AWS_BUDGET, ["check", "r7i.4xlarge"], bin_dir2,
                {"AWS_DAILY_CAP_USD": f"{cap:.4f}", "AWS_SPEND_LEDGER": str(ledger_path)}, tmp_path=call2)
    assert res2.returncode == 1, res2.stderr
    assert "refusing" in res2.stderr


# --------------------------------------------------------------------------------- launch-path gate wiring

def test_aws_cpu_launch_refused_by_budget_never_calls_run_instances(tmp_path):
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=40)
    bin_dir, aws_log, _, _ = _make_stub_bin(tmp_path, [inst])
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
    bin_dir, aws_log, _, _ = _make_stub_bin(tmp_path, [inst])
    res = _run(ROOT / "tools" / "aws_gpu.sh", ["launch"], bin_dir, {"AWS_DAILY_CAP_USD": "1"}, tmp_path=tmp_path)
    assert res.returncode == 1
    assert "refused by tools/aws_budget.sh" in res.stdout + res.stderr
    assert "ec2 run-instances" not in aws_log.read_text()


# ----------------------------------------------------------------------------------------- aws_idle_stop.sh

def test_idle_stop_stops_idle_instance_with_no_runner(tmp_path):
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=1)
    bin_dir, aws_log, ssh_log, _ = _make_stub_bin(tmp_path, [inst], cw_datapoints=[1.0, 2.0],
                                                ssh_pgrep_finds_runner=False)
    _write_gpu_state(tmp_path, "i-aaa", tmp_path / "aws_key.pem")
    res = _run(AWS_IDLE_STOP, [], bin_dir, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "ec2 stop-instances" in aws_log.read_text()
    assert "pgrep" in ssh_log.read_text()   # the no-runner check really ran over SSH, not assumed


def test_idle_stop_keeps_a_freshly_launched_instance(tmp_path):
    # 2026-09-24: a new instance (no CloudWatch data, low SSH load while provisioning, no runner yet) was STOPPED
    # minutes after launch. An instance younger than the idle window is never judged.
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=0.05)   # launched 3 minutes ago
    bin_dir, aws_log, ssh_log, _ = _make_stub_bin(tmp_path, [inst], cw_datapoints=[], ssh_pgrep_finds_runner=False)
    _write_gpu_state(tmp_path, "i-aaa", tmp_path / "aws_key.pem")
    res = _run(AWS_IDLE_STOP, [], bin_dir, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "ec2 stop-instances" not in aws_log.read_text()


def test_idle_stop_keeps_instance_when_runner_active(tmp_path):
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=1)
    bin_dir, aws_log, ssh_log, _ = _make_stub_bin(tmp_path, [inst], cw_datapoints=[1.0, 2.0],
                                                ssh_pgrep_finds_runner=True)
    _write_gpu_state(tmp_path, "i-aaa", tmp_path / "aws_key.pem")
    res = _run(AWS_IDLE_STOP, [], bin_dir, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "ec2 stop-instances" not in aws_log.read_text()


def test_idle_stop_keeps_instance_when_cpu_not_idle(tmp_path):
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=1)
    bin_dir, aws_log, ssh_log, _ = _make_stub_bin(tmp_path, [inst], cw_datapoints=[55.0, 60.0],
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
    bin_dir, aws_log, ssh_log, _ = _make_stub_bin(tmp_path, [inst], cw_datapoints=[1.0, 2.0],
                                                ssh_pgrep_finds_runner=False)
    res = _run(AWS_IDLE_STOP, [], bin_dir, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "ec2 stop-instances" not in aws_log.read_text()
    assert ssh_log.read_text() == ""   # never even attempted SSH -- no known key for this instance


# ------------------------------------------------------------------------- aws_idle_stop.sh: sync-before-stop

def test_idle_stop_syncs_before_stopping_and_stop_happens_after_the_sync(tmp_path):
    # 2026-09-25, incident-driven: pool1 was idle-stopped with two finished DA LTM-on seeds' last artifacts
    # written after the last routine pool_sync -- stranded until the owner restarted it by hand. A pull must
    # happen, and STOP must come strictly after it, never before/without it.
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=1)
    bin_dir, aws_log, ssh_log, rsync_log = _make_stub_bin(tmp_path, [inst], cw_datapoints=[1.0, 2.0],
                                                           ssh_pgrep_finds_runner=False, rsync_ok=True)
    _write_gpu_state(tmp_path, "i-aaa", tmp_path / "aws_key.pem")
    res = _run(AWS_IDLE_STOP, [], bin_dir, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert rsync_log.read_text().strip() != "", "sync-before-stop never called rsync"
    assert "ec2 stop-instances" in aws_log.read_text()

    combined_log = tmp_path / "aws_idle_stop.log"
    assert combined_log.exists()
    log_text = combined_log.read_text()
    sync_idx = log_text.find("rsync")   # the sync-before-stop function tees rsync's own output into this log
    stop_idx = log_text.find("STOPPING")
    # rsync's stdout is empty in this stub (it only appends to its OWN log), so fall back to proving order via
    # aws.log/rsync.log mtimes is unreliable under fast local test I/O -- instead assert STOPPING appears at
    # all (proving the gate let it through) and that a bare "sync-before-stop FAILED" never appears alongside it.
    assert stop_idx != -1
    assert "sync-before-stop FAILED" not in log_text


def test_idle_stop_does_not_stop_when_the_fallback_sync_fails(tmp_path):
    # THE ACTUAL GUARD: an idle, no-runner instance whose result-pull FAILS must NOT be stopped this cycle.
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=1)
    bin_dir, aws_log, ssh_log, rsync_log = _make_stub_bin(tmp_path, [inst], cw_datapoints=[1.0, 2.0],
                                                           ssh_pgrep_finds_runner=False, rsync_ok=False)
    _write_gpu_state(tmp_path, "i-aaa", tmp_path / "aws_key.pem")
    res = _run(AWS_IDLE_STOP, [], bin_dir, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert rsync_log.read_text().strip() != "", "the sync must still have been ATTEMPTED"
    assert "ec2 stop-instances" not in aws_log.read_text()
    combined_log = (tmp_path / "aws_idle_stop.log").read_text()
    assert "sync-before-stop FAILED" in combined_log


def test_idle_stop_does_not_stop_when_no_ssh_key_is_on_hand_to_sync(tmp_path):
    # Mirrors test_idle_stop_keeps_instance_when_no_state_file_key_to_verify_runner's reasoning, but for the
    # sync gate specifically: even if cpu_idle/runner_active somehow both read favorably with no verified key,
    # sync_node_before_stop itself must refuse (no ip/key -> cannot pull -> cannot safely stop).
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=1)
    bin_dir, aws_log, ssh_log, rsync_log = _make_stub_bin(tmp_path, [inst], cw_datapoints=[1.0, 2.0],
                                                           ssh_pgrep_finds_runner=False)
    # No _write_gpu_state call -- no state file, so have_ssh=0 and runner_flag stays "true" (inconclusive ->
    # keep) REGARDLESS of the sync gate; this proves the pre-existing conservative default still wins and the
    # sync path is never even reached when there is nothing to sync with.
    res = _run(AWS_IDLE_STOP, [], bin_dir, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "ec2 stop-instances" not in aws_log.read_text()
    assert rsync_log.read_text() == ""


def test_idle_stop_prefers_pool_sync_strict_for_a_registered_pool_node(tmp_path):
    # A node with an .aws_<name> state file that IS a registered pool-dispatch alias (.pool_ssh_config has a
    # Host block for it) must go through tools/pool_sync.sh --strict, not the raw fallback rsync -- identical
    # exclusions/isolated-revision handling to the routine cadence sync (see sync_node_before_stop's comment).
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=1)
    bin_dir, aws_log, ssh_log, rsync_log = _make_stub_bin(tmp_path, [inst], cw_datapoints=[1.0, 2.0],
                                                           ssh_pgrep_finds_runner=False, rsync_ok=True)
    state_dir = tmp_path / "state"; state_dir.mkdir(exist_ok=True)
    key_path = tmp_path / "aws_key.pem"; key_path.write_text("fake key\n")
    (state_dir / ".aws_pool1").write_text(f"instance=i-aaa\nregion=us-east-1\nkey={key_path}\nsg=sg-x\n")
    ssh_config = tmp_path / "pool_ssh_config"
    ssh_config.write_text(f"Include ~/.ssh/config\nHost pool1\n  HostName 1.2.3.4\n  User ubuntu\n  IdentityFile {key_path}\n")
    res = _run(AWS_IDLE_STOP, [], bin_dir, extra_env={
        "AWS_GPU_STATE_FILE": str(state_dir / ".aws_pool1"),
        "POOL_SSH_CONFIG": str(ssh_config),
    }, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "ec2 stop-instances" in aws_log.read_text()
    # pool_sync.sh's own rsync call targets the alias "pool1" (not a raw ip) when routed through the config.
    assert "pool1:" in rsync_log.read_text()


def test_idle_stop_no_running_instances_is_a_noop(tmp_path):
    bin_dir, aws_log, _, _ = _make_stub_bin(tmp_path, [])
    res = _run(AWS_IDLE_STOP, [], bin_dir, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "ec2 stop-instances" not in aws_log.read_text()


# ---------------------------------------------------------- aws_idle_stop.sh: 2026-09-25 fix-round review items

def test_idle_stop_fallback_pulls_from_the_sim_layout_when_derisk_pool_does_not_exist(tmp_path):
    # HIGH #1 (2026-09-25 review): the .aws_gpu/.aws_cpuN lanes (tools/aws_cpu_launch.sh + tools/
    # aws_cpu_provision.sh / tools/aws_provision.sh) rsync CODE to ~/sim, never ~/derisk-pool/sim -- that's the
    # pool-node (tools/aws_pool_node.sh) layout only. The OLD single-path fallback always assumed
    # ~/derisk-pool/sim; on these lanes that source dir never exists, rsync exits 23 every cycle, and the
    # instance is NEVER idle-stopped (only the $50/day cap ever stops it) -- its results under ~/sim are never
    # pulled either. Stub ssh: ~/derisk-pool/sim does not exist, ~/sim does (and has results) -- exactly this
    # node's real layout.
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=1)
    bin_dir, aws_log, ssh_log, rsync_log = _make_stub_bin(tmp_path, [inst], cw_datapoints=[1.0, 2.0],
                                                           ssh_pgrep_finds_runner=False, rsync_ok=True)
    ssh_stub = bin_dir / "ssh"
    ssh_stub.write_text(f"""#!/usr/bin/env bash
echo "$*" >> "{ssh_log}"
case "$*" in
  *uptime*) echo " 12:00:00 up 1 day,  1 user,  load average: 0.05, 0.10, 0.10" ;;
  *nproc*) echo 8 ;;
  *pgrep*) exit 1 ;;
  *"if [ -d ~/derisk-pool"*) echo no_root ;;
  *"if [ -d "*) echo has_raw ;;
  *) exit 0 ;;
esac
""")
    ssh_stub.chmod(ssh_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    _write_gpu_state(tmp_path, "i-aaa", tmp_path / "aws_key.pem")
    res = _run(AWS_IDLE_STOP, [], bin_dir, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "ec2 stop-instances" in aws_log.read_text()
    assert "sim/research/findings/raw" in rsync_log.read_text()
    assert "derisk-pool" not in rsync_log.read_text(), "must never have rsync'd the layout that doesn't exist"


def test_idle_stop_fallback_refuses_when_neither_known_layout_exists(tmp_path):
    # HIGH #1, the flip side: a node whose code layout matches NEITHER known convention must be treated as
    # unverifiable (conservative: keep running), never silently skipped as if it had nothing to sync. Mutation
    # check: the OLD code never asked `ssh test -d` at all -- it just rsync'd ~/derisk-pool/sim unconditionally,
    # which the stub rsync below would happily report as a SUCCESS regardless of what's really there, and the
    # instance would wrongly get stopped.
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=1)
    bin_dir, aws_log, ssh_log, rsync_log = _make_stub_bin(tmp_path, [inst], cw_datapoints=[1.0, 2.0],
                                                           ssh_pgrep_finds_runner=False, rsync_ok=True)
    ssh_stub = bin_dir / "ssh"
    ssh_stub.write_text(f"""#!/usr/bin/env bash
echo "$*" >> "{ssh_log}"
case "$*" in
  *uptime*) echo " 12:00:00 up 1 day,  1 user,  load average: 0.05, 0.10, 0.10" ;;
  *nproc*) echo 8 ;;
  *pgrep*) exit 1 ;;
  *"if [ -d "*) echo no_root ;;
  *) exit 0 ;;
esac
""")
    ssh_stub.chmod(ssh_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    _write_gpu_state(tmp_path, "i-aaa", tmp_path / "aws_key.pem")
    res = _run(AWS_IDLE_STOP, [], bin_dir, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "ec2 stop-instances" not in aws_log.read_text()
    assert rsync_log.read_text() == "", "must never have attempted an rsync with no confirmed source dir"
    combined_log = (tmp_path / "aws_idle_stop.log").read_text()
    assert "no known project layout found" in combined_log


def _make_stdin_draining_stub_bin(tmp_path, describe_instances, cw_datapoints=None):
    """Like _make_stub_bin, but its `ssh` stub ALWAYS drains whatever is sitting on its OWN stdin (`cat
    >/dev/null`) before answering -- regardless of whether it was invoked with `-n` -- so a test built on this
    stub isolates the fd-3 loop-read fix (HIGH #2) from the `-n` flag: even if some remote call inside the loop
    somehow failed to behave as `-n` promises, the loop's OWN read of the instance-id list must still be immune,
    because it now lives on a completely separate fd."""
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    describe_fixture = tmp_path / "describe.json"
    describe_fixture.write_text(_describe_json(describe_instances))
    cw_fixture = tmp_path / "cw.json"
    cw_fixture.write_text(json.dumps({"Datapoints": [{"Average": a} for a in (cw_datapoints or [])]}))
    aws_log = tmp_path / "aws.log"; aws_log.write_text("")
    ssh_log = tmp_path / "ssh.log"; ssh_log.write_text("")

    aws_stub = bin_dir / "aws"
    aws_stub.write_text(_AWS_STUB_TEMPLATE.format(
        log=aws_log, cw_fixture=cw_fixture, describe_fixture=describe_fixture))
    aws_stub.chmod(aws_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    ssh_stub = bin_dir / "ssh"
    ssh_stub.write_text(f"""#!/usr/bin/env bash
cat >/dev/null 2>&1   # DRAIN local stdin unconditionally -- simulates real ssh's stdin-forwarding, `-n` or not
echo "$*" >> "{ssh_log}"
case "$*" in
  *uptime*) echo " 12:00:00 up 1 day,  1 user,  load average: 0.05, 0.10, 0.10" ;;
  *nproc*) echo 8 ;;
  *pgrep*) exit 1 ;;
  *"if [ -d "*) echo has_raw ;;
  *) exit 0 ;;
esac
""")
    ssh_stub.chmod(ssh_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    rsync_stub = bin_dir / "rsync"
    rsync_stub.write_text("#!/usr/bin/env bash\nexit 0\n")   # never inspected by this test -- just must not hang
    rsync_stub.chmod(rsync_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir, aws_log, ssh_log


def test_idle_stop_checks_the_second_instance_even_when_ssh_drains_local_stdin(tmp_path):
    # HIGH #2 (2026-09-25 review): `while read iid; do ... ssh ...; done <<<"$ids"` put the id list on fd 0 for
    # the WHOLE loop body -- ssh, even when its remote command never reads stdin itself, still drains whatever
    # local stdin it inherits (the same bug class 096dfdae0 fixed in the dispatcher's revision_available
    # probe). With 2+ running instances, the FIRST one's ssh calls drained the REST of the id list off fd 0, so
    # `read -r iid` hit EOF and the loop silently ended after ONE instance -- matching the production log
    # (exactly one load1 line per cycle while pool1 AND pool2 both ran). Two running instances here; i-aaa has
    # a verified ssh key (so its uptime/nproc/pgrep calls actually run and can drain), i-bbb has none. The
    # assertion is that i-bbb's OWN CloudWatch check (an `aws` call carrying its instance id) still happens --
    # i.e. the loop reached a SECOND iteration at all.
    inst_a = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=1)
    inst_b = _instance("i-bbb", "r7i.4xlarge", "running", hours_ago=1)
    bin_dir, aws_log, ssh_log = _make_stdin_draining_stub_bin(tmp_path, [inst_a, inst_b], cw_datapoints=[])
    _write_gpu_state(tmp_path, "i-aaa", tmp_path / "aws_key.pem")
    res = _run(AWS_IDLE_STOP, [], bin_dir, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "uptime" in ssh_log.read_text(), "i-aaa's ssh fallback never even ran -- test setup is wrong"
    assert "Value=i-bbb" in aws_log.read_text(), (
        "the loop never reached the second instance -- fd-3 read did not protect it from ssh draining fd 0")


def test_idle_stop_remote_ssh_calls_all_pass_dash_n(tmp_path):
    # HIGH #2, the belt-and-suspenders half: every remote ssh call this script makes (uptime/nproc/pgrep) must
    # ALSO carry `-n` on its own, independent of the fd-3 structural fix above.
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=1)
    bin_dir, aws_log, ssh_log, rsync_log = _make_stub_bin(tmp_path, [inst], cw_datapoints=[],
                                                           ssh_pgrep_finds_runner=False, rsync_ok=True)
    _write_gpu_state(tmp_path, "i-aaa", tmp_path / "aws_key.pem")
    res = _run(AWS_IDLE_STOP, [], bin_dir, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    lines = [ln for ln in ssh_log.read_text().splitlines() if ln.strip()]
    assert lines, "no ssh calls observed -- test setup is wrong"
    for ln in lines:
        assert ln.startswith("-n "), f"an ssh call did not pass -n first: {ln!r}"


def test_idle_stop_rechecks_runner_right_before_stop_and_aborts_if_one_appeared(tmp_path):
    # MEDIUM (2026-09-25 review): the strict sync between the ORIGINAL no-runner check and the stop call can
    # take from ~1s to several minutes (main pull + ssh ls + one rsync per isolated revision, up to 180s each),
    # widening the check-to-stop window enough for a job to land in it (5b5ea1b7 did, at 09:59:54). Stop must
    # re-verify no-runner immediately before the stop-instances call, not rely on a reading that may now be
    # stale. Stub pgrep: NOT found on the FIRST call (the original check, which is why the sync even started),
    # FOUND on every call after (the re-check right before stop).
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=1)
    bin_dir, aws_log, ssh_log, rsync_log = _make_stub_bin(tmp_path, [inst], cw_datapoints=[1.0, 2.0],
                                                           ssh_pgrep_finds_runner=False, rsync_ok=True)
    counter = tmp_path / "pgrep_calls"
    ssh_stub = bin_dir / "ssh"
    ssh_stub.write_text(f"""#!/usr/bin/env bash
echo "$*" >> "{ssh_log}"
case "$*" in
  *uptime*) echo " 12:00:00 up 1 day,  1 user,  load average: 0.05, 0.10, 0.10" ;;
  *nproc*) echo 8 ;;
  *pgrep*)
    n=$(cat "{counter}" 2>/dev/null || echo 0); n=$((n+1)); echo "$n" > "{counter}"
    [ "$n" -eq 1 ] && exit 1   # first call (the original check): no runner found
    exit 0                    # every call after (the re-check): a runner IS now running
    ;;
  *"if [ -d "*) echo has_raw ;;
  *) exit 0 ;;
esac
""")
    ssh_stub.chmod(ssh_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    _write_gpu_state(tmp_path, "i-aaa", tmp_path / "aws_key.pem")
    res = _run(AWS_IDLE_STOP, [], bin_dir, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "ec2 stop-instances" not in aws_log.read_text(), "stopped despite a runner appearing during the sync"
    assert rsync_log.read_text().strip() != "", "the sync must still have run (it's what widened the window)"
    combined_log = (tmp_path / "aws_idle_stop.log").read_text()
    assert "a runner appeared during the sync" in combined_log


def test_idle_stop_log_states_cloudwatch_signal_is_sustained(tmp_path):
    # MEDIUM (2026-09-25 review): "state the idle-signal strength honestly" -- a CONCLUSIVE CloudWatch read
    # spans the whole idle window (multiple 5-min datapoints); this must read differently in the log than a
    # single SSH load sample (below).
    #
    # LOW, re-review (2026-09-25 fix round 3): the original fix's own label ("CloudWatch (sustained, >= 20m of
    # 5-min datapoints)") itself overstated the evidence -- cw-has-data accepts a SINGLE datapoint, and
    # CloudWatch lags 5-10 minutes, so "sustained >= 20m" was not actually proven by the data. The label must
    # name the REAL sample count instead of claiming a duration it cannot support -- here 2 datapoints, not
    # "sustained".
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=1)
    bin_dir, aws_log, ssh_log, rsync_log = _make_stub_bin(tmp_path, [inst], cw_datapoints=[1.0, 2.0],
                                                           ssh_pgrep_finds_runner=False, rsync_ok=True)
    _write_gpu_state(tmp_path, "i-aaa", tmp_path / "aws_key.pem")
    res = _run(AWS_IDLE_STOP, [], bin_dir, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "ec2 stop-instances" in aws_log.read_text()
    combined_log = (tmp_path / "aws_idle_stop.log").read_text()
    assert "CloudWatch (2 x 5-min average" in combined_log, combined_log
    assert "sustained" not in combined_log, "must not claim more than the 2 real datapoints support"


def test_idle_stop_log_states_ssh_loadavg_signal_is_a_single_sample(tmp_path):
    # MEDIUM, the flip side: the SSH-loadavg fallback (used when CloudWatch has no datapoints yet) is a SINGLE
    # 1-minute sample, not the sustained signal the CloudWatch path gives -- the STOPPING line must say so, not
    # imply the same strength of evidence as the CloudWatch path.
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=1)
    bin_dir, aws_log, ssh_log, rsync_log = _make_stub_bin(tmp_path, [inst], cw_datapoints=[],
                                                           ssh_pgrep_finds_runner=False, rsync_ok=True)
    _write_gpu_state(tmp_path, "i-aaa", tmp_path / "aws_key.pem")
    res = _run(AWS_IDLE_STOP, [], bin_dir, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "ec2 stop-instances" in aws_log.read_text()
    combined_log = (tmp_path / "aws_idle_stop.log").read_text()
    assert "single 1-minute sample" in combined_log
    assert "NOT a sustained" in combined_log


# --------------------------------------------------------- aws_idle_stop.sh: 2026-09-25 fix round 3 review items

def test_idle_stop_keeps_instance_when_initial_runner_check_ssh_is_unreachable(tmp_path):
    # MEDIUM (2026-09-25 review, fix round 3): rc=255 (ssh cannot connect) used to fall into the SAME "else"
    # branch as rc=1 ("pgrep ran and found nothing") and read identically as "no runner" -- an UNREACHABLE check
    # was able to stop an instance this script could not actually verify was idle. pgrep here always exits 255
    # (a connection failure), never 0 or 1.
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=1)
    bin_dir, aws_log, ssh_log, rsync_log = _make_stub_bin(tmp_path, [inst], cw_datapoints=[1.0, 2.0],
                                                           ssh_pgrep_finds_runner=False, rsync_ok=True)
    ssh_stub = bin_dir / "ssh"
    ssh_stub.write_text(f"""#!/usr/bin/env bash
echo "$*" >> "{ssh_log}"
case "$*" in
  *pgrep*) exit 255 ;;
  *"if [ -d "*) echo has_raw ;;
  *) exit 0 ;;
esac
""")
    ssh_stub.chmod(ssh_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    _write_gpu_state(tmp_path, "i-aaa", tmp_path / "aws_key.pem")
    res = _run(AWS_IDLE_STOP, [], bin_dir, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "ec2 stop-instances" not in aws_log.read_text(), (
        "stopped despite an INCONCLUSIVE (unreachable) runner check")
    combined_log = (tmp_path / "aws_idle_stop.log").read_text()
    assert "INCONCLUSIVE, treated as active" in combined_log
    assert rsync_log.read_text() == "", (
        "sync must never even have been attempted -- the outer idle() call must read 'keep'")


def test_idle_stop_keeps_instance_when_recheck_runner_probe_is_unreachable(tmp_path):
    # Same rc=255-vs-rc=1 fix, applied to the RE-CHECK right before stop. The first pgrep call (the original
    # check, which is why the sync even starts) finds no runner (rc=1); every call after (the re-check) is
    # UNREACHABLE (rc=255) -- must be treated as inconclusive (keep), never as "definitely no runner".
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=1)
    bin_dir, aws_log, ssh_log, rsync_log = _make_stub_bin(tmp_path, [inst], cw_datapoints=[1.0, 2.0],
                                                           ssh_pgrep_finds_runner=False, rsync_ok=True)
    counter = tmp_path / "pgrep_calls"
    ssh_stub = bin_dir / "ssh"
    ssh_stub.write_text(f"""#!/usr/bin/env bash
echo "$*" >> "{ssh_log}"
case "$*" in
  *pgrep*)
    n=$(cat "{counter}" 2>/dev/null || echo 0); n=$((n+1)); echo "$n" > "{counter}"
    [ "$n" -eq 1 ] && exit 1   # first call (the original check): no runner found
    exit 255                  # every call after (the re-check): ssh cannot connect
    ;;
  *"if [ -d "*) echo has_raw ;;
  *) exit 0 ;;
esac
""")
    ssh_stub.chmod(ssh_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    _write_gpu_state(tmp_path, "i-aaa", tmp_path / "aws_key.pem")
    res = _run(AWS_IDLE_STOP, [], bin_dir, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "ec2 stop-instances" not in aws_log.read_text(), (
        "stopped despite an INCONCLUSIVE (unreachable) re-check")
    assert rsync_log.read_text().strip() != "", "the sync must still have run (the original check said no runner)"
    combined_log = (tmp_path / "aws_idle_stop.log").read_text()
    assert "the re-check's ssh was INCONCLUSIVE (rc=255" in combined_log


def test_idle_stop_takes_node_out_of_dispatch_during_the_sync_and_restores_it_after(tmp_path):
    # STILL-OPEN ITEM (2026-09-25 review, fix round 3): "the node was not taken out of dispatch before the
    # sync" -- the re-check right before stop only DETECTS a job dispatched during the sync window; this proves
    # the node is actually taken OUT of tools/pool_autodispatch.sh's pool for the duration of the sync call
    # itself (the rsync stub below snapshots the extra-nodes file the INSTANT it runs, so this is a genuine
    # mid-sync observation, not an end-of-run inference), and that registration is restored once the decision
    # is made.
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=1)
    bin_dir, aws_log, ssh_log, rsync_log = _make_stub_bin(tmp_path, [inst], cw_datapoints=[1.0, 2.0],
                                                           ssh_pgrep_finds_runner=False, rsync_ok=True)
    extra_nodes = tmp_path / "extra_nodes"
    extra_nodes.write_text("gpu\n")
    marker = tmp_path / "rsync_saw_extra_nodes"
    rsync_stub = bin_dir / "rsync"
    rsync_stub.write_text(f"""#!/usr/bin/env bash
echo "$*" >> "{rsync_log}"
if grep -qxF gpu "{extra_nodes}" 2>/dev/null; then echo PRESENT >> "{marker}"; else echo ABSENT >> "{marker}"; fi
exit 0
""")
    rsync_stub.chmod(rsync_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    _write_gpu_state(tmp_path, "i-aaa", tmp_path / "aws_key.pem")
    res = _run(AWS_IDLE_STOP, [], bin_dir, extra_env={"POOL_EXTRA_NODES_FILE": str(extra_nodes)}, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "ec2 stop-instances" in aws_log.read_text()
    assert marker.exists(), "rsync (the sync step) never ran -- test setup is wrong"
    seen = marker.read_text().split()   # one "PRESENT"/"ABSENT" per candidate-directory rsync call
    assert seen, "no rsync call observed the extra-nodes file -- test setup is wrong"
    assert all(s == "ABSENT" for s in seen), (
        f"the node was still registered for dispatch WHILE the sync was running -- never taken out of dispatch: {seen}")
    assert extra_nodes.read_text().splitlines() == ["gpu"], "registration was not restored after the cycle"


def test_idle_stop_restores_dispatch_registration_even_when_sync_fails(tmp_path):
    # The pause/resume pair must run on EVERY exit from the sync+decision block, not just the "stopped" one --
    # otherwise a failed sync would permanently strand the node out of dispatch until someone noticed by hand.
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=1)
    bin_dir, aws_log, ssh_log, rsync_log = _make_stub_bin(tmp_path, [inst], cw_datapoints=[1.0, 2.0],
                                                           ssh_pgrep_finds_runner=False, rsync_ok=False)
    extra_nodes = tmp_path / "extra_nodes"
    extra_nodes.write_text("gpu\n")
    _write_gpu_state(tmp_path, "i-aaa", tmp_path / "aws_key.pem")
    res = _run(AWS_IDLE_STOP, [], bin_dir, extra_env={"POOL_EXTRA_NODES_FILE": str(extra_nodes)}, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "ec2 stop-instances" not in aws_log.read_text()
    assert extra_nodes.read_text().splitlines() == ["gpu"], (
        "registration must be restored even when the sync fails")


def test_idle_stop_order_rsync_strictly_before_stop_instances_via_one_shared_call_log(tmp_path):
    # LOW (2026-09-25 review): the prior version of this test only checked that "STOPPING" appeared in the log
    # and that rsync was called SOMEWHERE -- never the actual ORDER, so mutation-deleting the
    # `if sync_node_before_stop ...; then stop` gating still passed. `aws` and `rsync` now append to ONE SHARED
    # log (same idiom as tests/test_aws_pool_node_workflow.py's `down`-ordering tests) so line order really is
    # call order.
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=1)
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    shared_log = tmp_path / "shared.log"; shared_log.write_text("")
    describe_fixture = tmp_path / "describe.json"; describe_fixture.write_text(_describe_json([inst]))
    cw_fixture = tmp_path / "cw.json"
    cw_fixture.write_text(json.dumps({"Datapoints": [{"Average": a} for a in (1.0, 2.0)]}))

    aws_stub = bin_dir / "aws"
    aws_stub.write_text(f"""#!/usr/bin/env bash
echo "AWS $*" >> "{shared_log}"
if [[ "$*" == *"cloudwatch get-metric-statistics"* ]]; then cat "{cw_fixture}"; exit 0; fi
if [[ "$*" == *"ec2 describe-instances"* ]]; then cat "{describe_fixture}"; exit 0; fi
if [[ "$*" == *"ec2 stop-instances"* ]]; then echo stopped; exit 0; fi
echo "sg-stub-000"; exit 0
""")
    aws_stub.chmod(aws_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    ssh_stub = bin_dir / "ssh"
    ssh_stub.write_text(f"""#!/usr/bin/env bash
echo "SSH $*" >> "{shared_log}"
case "$*" in
  *pgrep*) exit 1 ;;
  *"if [ -d "*) echo has_raw ;;
  *) exit 0 ;;
esac
""")
    ssh_stub.chmod(ssh_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    rsync_stub = bin_dir / "rsync"
    rsync_stub.write_text(f'#!/usr/bin/env bash\necho "RSYNC $*" >> "{shared_log}"\nexit 0\n')
    rsync_stub.chmod(rsync_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    _write_gpu_state(tmp_path, "i-aaa", tmp_path / "aws_key.pem")
    res = _run(AWS_IDLE_STOP, [], bin_dir, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    lines = shared_log.read_text().splitlines()
    rsync_idx = next((i for i, ln in enumerate(lines) if ln.startswith("RSYNC ")), None)
    stop_idx = next((i for i, ln in enumerate(lines) if "ec2 stop-instances" in ln), None)
    assert rsync_idx is not None, f"rsync was never called: {lines}"
    assert stop_idx is not None, f"stop-instances was never called: {lines}"
    assert rsync_idx < stop_idx, f"stop-instances happened before/without a prior rsync: {lines}"


def test_idle_stop_does_not_stop_when_recorded_key_file_is_missing_even_with_cpu_idle(tmp_path):
    # LOW (2026-09-25 review): mutation-deleting sync_node_before_stop's own `[ ! -f "$key" ]` no-key guard
    # passed 19/19 because no existing test's state file pointed at a key path that does not exist while CPU
    # read idle -- every prior case either had a real key file or CPU that wasn't idle. A missing key file
    # (recorded, but the file itself is gone -- e.g. cleaned up, or the state file copied without it) must
    # never be treated as "have ssh access" anywhere in this script, end to end.
    inst = _instance("i-aaa", "r7i.4xlarge", "running", hours_ago=1)
    bin_dir, aws_log, ssh_log, rsync_log = _make_stub_bin(tmp_path, [inst], cw_datapoints=[1.0, 2.0],
                                                           ssh_pgrep_finds_runner=False, rsync_ok=True)
    state_dir = tmp_path / "state"; state_dir.mkdir(exist_ok=True)
    missing_key = tmp_path / "does-not-exist.pem"   # recorded path, but the file was never written
    (state_dir / ".aws_gpu").write_text(f"instance=i-aaa\nregion=us-east-1\nkey={missing_key}\nsg=sg-stub-000\n")
    res = _run(AWS_IDLE_STOP, [], bin_dir, extra_env={"AWS_GPU_STATE_FILE": str(state_dir / ".aws_gpu")},
               tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "ec2 stop-instances" not in aws_log.read_text()
    assert rsync_log.read_text() == ""
    assert ssh_log.read_text() == "", "no ssh key on hand -- must never even attempt an ssh call"


# --------------------------------------------------------------------------------------- guard timer ordering

def test_guard_service_records_spend_before_idle_stop_can_act():
    # 2026-09-23 review, item 7: `aws_budget.sh enforce` is what RECORDS this cycle's spend to the ledger. If
    # aws_idle_stop.sh ran FIRST and stopped an instance, enforce would then observe it already stopped
    # (hours_running_today -> 0) and silently lose up to this cycle's ~10 minutes of accrued compute. The
    # ExecStart must run `aws_budget.sh enforce` before `aws_idle_stop.sh`.
    service = (ROOT / "tools" / "systemd" / "aws-guard.service").read_text()
    exec_line = next(ln for ln in service.splitlines() if ln.strip().startswith("ExecStart="))
    enforce_pos = exec_line.find("aws_budget.sh enforce")
    idle_stop_pos = exec_line.find("aws_idle_stop.sh")
    assert enforce_pos != -1 and idle_stop_pos != -1, f"ExecStart is missing one of the two calls: {exec_line!r}"
    assert enforce_pos < idle_stop_pos, (
        f"aws_idle_stop.sh runs BEFORE aws_budget.sh enforce records this cycle's spend: {exec_line!r}")
