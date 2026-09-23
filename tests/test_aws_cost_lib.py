"""Unit tests for tools/aws_cost_lib.py — the pure spend-arithmetic + idle-decision logic behind the AWS
budget guard (tools/aws_budget.sh, tools/aws_idle_stop.sh; owner-approved 2026-09-23, cap raised same day
$15 -> $50/day). Every function here is side-effect-free (no real `aws`/`ssh` calls), so these run hermetic
and fast against synthetic instance/CloudWatch JSON. Subprocess-level tests against the bash wrappers with a
stubbed `aws`/`ssh` on PATH live in tests/test_aws_budget_guard_workflow.py.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import aws_cost_lib as lib  # noqa: E402


NOW = datetime(2026, 9, 23, 18, 0, 0, tzinfo=timezone.utc)


def _instance(instance_id="i-aaa", itype="r7i.4xlarge", state="running", launch=None,
              project_tag=True, name_tag=None):
    tags = []
    if project_tag:
        tags.append({"Key": "Project", "Value": "neural-sim"})
    if name_tag:
        tags.append({"Key": "Name", "Value": name_tag})
    return {
        "InstanceId": instance_id,
        "InstanceType": itype,
        "State": {"Name": state},
        "LaunchTime": (launch or NOW).isoformat().replace("+00:00", "Z"),
        "Tags": tags,
    }


# --------------------------------------------------------------------------------------------------- tags

def test_is_project_instance_via_project_tag():
    inst = _instance(project_tag=True)
    assert lib.is_project_instance(inst) is True


def test_is_project_instance_via_legacy_name_tag():
    inst = _instance(project_tag=False, name_tag="claude-cpu-verify")
    assert lib.is_project_instance(inst) is True
    inst2 = _instance(project_tag=False, name_tag="claude-gpu-verify")
    assert lib.is_project_instance(inst2) is True


def test_is_project_instance_false_for_unrelated_instance():
    inst = _instance(project_tag=False, name_tag="someone-elses-box")
    assert lib.is_project_instance(inst) is False


# ------------------------------------------------------------------------------------------ hours_running

def test_hours_running_today_two_hours_in():
    inst = _instance(state="running", launch=NOW - timedelta(hours=2))
    hours = lib.hours_running_today(inst, now=NOW)
    assert abs(hours - 2.0) < 1e-6


def test_hours_running_today_capped_at_midnight_utc():
    # launched yesterday -> only counts from today's UTC midnight, not the full elapsed wall time
    inst = _instance(state="running", launch=NOW - timedelta(days=1, hours=3))
    hours = lib.hours_running_today(inst, now=NOW)
    assert abs(hours - 18.0) < 1e-6   # NOW is 18:00 UTC -> 18h since 00:00 UTC today


def test_hours_running_today_zero_when_stopped():
    inst = _instance(state="stopped", launch=NOW - timedelta(hours=5))
    assert lib.hours_running_today(inst, now=NOW) == 0.0


def test_hours_running_today_zero_missing_launch_time():
    inst = _instance(state="running")
    del inst["LaunchTime"]
    assert lib.hours_running_today(inst, now=NOW) == 0.0


# --------------------------------------------------------------------------------------------------- price

def test_price_per_hour_known_type():
    assert lib.price_per_hour("r7i.4xlarge") == lib.PRICES["r7i.4xlarge"]


def test_price_per_hour_unknown_type_falls_back_to_max_known_never_undercounts():
    assert lib.price_per_hour("some-brand-new-instance-type") == max(lib.PRICES.values())


# --------------------------------------------------------------------------------------------- estimate_spend

def test_estimate_spend_excludes_non_project_instances():
    project = _instance("i-aaa", "r7i.4xlarge", "running", NOW - timedelta(hours=1))
    other = _instance("i-bbb", "r7i.4xlarge", "running", NOW - timedelta(hours=1), project_tag=False)
    total, rows = lib.estimate_spend([project, other], now=NOW)
    assert len(rows) == 1
    assert rows[0]["id"] == "i-aaa"
    expected = 1.0 * lib.PRICES["r7i.4xlarge"] + lib.EBS_DAILY_USD
    assert abs(total - round(expected, 4)) < 1e-4


def test_estimate_spend_excludes_terminated_instances():
    terminated = _instance("i-ccc", "r7i.4xlarge", "terminated", NOW - timedelta(hours=1))
    total, rows = lib.estimate_spend([terminated], now=NOW)
    assert total == 0.0
    assert rows == []


def test_estimate_spend_includes_stopped_instance_ebs_but_no_compute():
    stopped = _instance("i-ddd", "g5.xlarge", "stopped", NOW - timedelta(hours=5))
    total, rows = lib.estimate_spend([stopped], now=NOW)
    assert len(rows) == 1
    assert rows[0]["hours_today"] == 0.0
    assert abs(total - lib.EBS_DAILY_USD) < 1e-4


def test_estimate_spend_sums_multiple_project_instances():
    a = _instance("i-aaa", "r7i.4xlarge", "running", NOW - timedelta(hours=1))
    b = _instance("i-bbb", "g5.xlarge", "running", NOW - timedelta(hours=2), project_tag=False,
                  name_tag="claude-gpu-verify")
    total, rows = lib.estimate_spend([a, b], now=NOW)
    assert len(rows) == 2
    expected = (1.0 * lib.PRICES["r7i.4xlarge"] + lib.EBS_DAILY_USD) + (2.0 * lib.PRICES["g5.xlarge"] + lib.EBS_DAILY_USD)
    assert abs(total - round(expected, 4)) < 1e-3


# --------------------------------------------------------------------------------------------- stop_candidates

def test_stop_candidates_empty_under_cap():
    inst = _instance("i-aaa", "r7i.4xlarge", "running", NOW - timedelta(hours=1))
    ids, total, _rows = lib.stop_candidates([inst], cap=1000.0, now=NOW)
    assert ids == []


def test_stop_candidates_over_cap_lists_running_only():
    running = _instance("i-aaa", "r7i.4xlarge", "running", NOW - timedelta(hours=40))  # big hours -> big cost
    stopped = _instance("i-bbb", "g5.xlarge", "stopped", NOW - timedelta(hours=1), name_tag="claude-gpu-verify")
    ids, total, _rows = lib.stop_candidates([running, stopped], cap=1.0, now=NOW)
    assert ids == ["i-aaa"]
    assert total >= 1.0


# --------------------------------------------------------------------------------------------- idle decision

def test_is_idle_by_cpu_samples_all_low():
    assert lib.is_idle_by_cpu_samples([1.0, 2.5, 0.0, 4.9], threshold_pct=10.0) is True


def test_is_idle_by_cpu_samples_one_high_not_idle():
    assert lib.is_idle_by_cpu_samples([1.0, 2.5, 55.0], threshold_pct=10.0) is False


def test_is_idle_by_cpu_samples_empty_is_inconclusive_not_idle():
    assert lib.is_idle_by_cpu_samples([], threshold_pct=10.0) is False


def test_is_idle_by_loadavg_low_load():
    # load1=0.3 on 8 cpus -> 3.75% < 10%
    assert lib.is_idle_by_loadavg(0.3, 8, threshold_pct=10.0) is True


def test_is_idle_by_loadavg_high_load():
    assert lib.is_idle_by_loadavg(6.0, 8, threshold_pct=10.0) is False


def test_is_idle_by_loadavg_zero_cpu_inconclusive():
    assert lib.is_idle_by_loadavg(0.1, 0, threshold_pct=10.0) is False


def test_idle_decision_requires_both_idle_cpu_and_no_runner():
    assert lib.idle_decision(cpu_idle=True, runner_active=False) is True
    assert lib.idle_decision(cpu_idle=True, runner_active=True) is False
    assert lib.idle_decision(cpu_idle=False, runner_active=False) is False


def test_idle_decision_unknown_runner_state_defaults_to_keep():
    # couldn't determine whether a runner is active (e.g. SSH unreachable) -> never stop
    assert lib.idle_decision(cpu_idle=True, runner_active=None) is False


# --------------------------------------------------------------------------------------------------- CLI

def _run_cli(args, stdin_text=""):
    return subprocess.run(
        [sys.executable, str(ROOT / "tools" / "aws_cost_lib.py"), *args],
        input=stdin_text, capture_output=True, text=True, timeout=15,
    )


def _describe_json(instances):
    return json.dumps({"Reservations": [{"Instances": instances}]})


def test_cli_check_exits_zero_under_cap():
    inst = _instance("i-aaa", "r7i.4xlarge", "running", NOW - timedelta(hours=1))
    res = _run_cli(["check", "--cap", "1000"], _describe_json([inst]))
    assert res.returncode == 0, res.stderr


def test_cli_check_exits_nonzero_over_cap():
    inst = _instance("i-aaa", "r7i.4xlarge", "running", NOW - timedelta(hours=40))
    res = _run_cli(["check", "--cap", "1.0"], _describe_json([inst]))
    assert res.returncode == 1
    assert "refusing" in res.stderr


def test_cli_check_accounts_for_the_type_about_to_launch():
    # nothing running yet, but the FIRST HOUR of a pricey new instance alone would blow a tiny cap
    res = _run_cli(["check", "--cap", "0.5", "--type", "r7i.4xlarge"], _describe_json([]))
    assert res.returncode == 1


def test_cli_enforce_lists_running_instances_over_cap():
    inst = _instance("i-aaa", "r7i.4xlarge", "running", NOW - timedelta(hours=40))
    res = _run_cli(["enforce", "--cap", "1.0"], _describe_json([inst]))
    assert res.returncode == 0
    assert "i-aaa" in res.stdout.split()


def test_cli_enforce_empty_under_cap():
    inst = _instance("i-aaa", "r7i.4xlarge", "running", NOW - timedelta(hours=1))
    res = _run_cli(["enforce", "--cap", "1000"], _describe_json([inst]))
    assert res.returncode == 0
    assert res.stdout.strip() == ""


def test_cli_project_ids_filters_non_project_and_non_running():
    running_project = _instance("i-aaa", "r7i.4xlarge", "running")
    stopped_project = _instance("i-bbb", "r7i.4xlarge", "stopped")
    running_other = _instance("i-ccc", "r7i.4xlarge", "running", project_tag=False)
    res = _run_cli(["project-ids"], _describe_json([running_project, stopped_project, running_other]))
    assert res.returncode == 0
    ids = res.stdout.split()
    assert ids == ["i-aaa"]


def test_cli_cpu_idle_exits_zero_when_all_datapoints_low():
    cw = json.dumps({"Datapoints": [{"Average": 1.0}, {"Average": 3.0}]})
    res = _run_cli(["cpu-idle", "--threshold", "10"], cw)
    assert res.returncode == 0


def test_cli_cpu_idle_exits_nonzero_when_no_datapoints():
    cw = json.dumps({"Datapoints": []})
    res = _run_cli(["cpu-idle", "--threshold", "10"], cw)
    assert res.returncode == 1


def test_cli_cw_has_data_distinguishes_conclusive_busy_from_no_data():
    # A CONCLUSIVE "busy" verdict (has datapoints, all high) must NOT be treated the same as "no data yet" --
    # this is what lets tools/aws_idle_stop.sh skip the SSH load-average fallback when CloudWatch already
    # gave a real (even if busy) answer.
    busy = json.dumps({"Datapoints": [{"Average": 90.0}]})
    empty = json.dumps({"Datapoints": []})
    assert _run_cli(["cw-has-data"], busy).returncode == 0
    assert _run_cli(["cw-has-data"], empty).returncode == 1


def test_cli_loadavg_idle():
    res_idle = _run_cli(["loadavg-idle", "--load1", "0.2", "--ncpu", "8", "--threshold", "10"])
    assert res_idle.returncode == 0
    res_busy = _run_cli(["loadavg-idle", "--load1", "7.0", "--ncpu", "8", "--threshold", "10"])
    assert res_busy.returncode == 1


def test_cli_idle_combines_cpu_and_runner():
    res_stop = _run_cli(["idle", "1", "--runner-active", "false"])
    assert res_stop.returncode == 0
    res_keep_runner = _run_cli(["idle", "1", "--runner-active", "true"])
    assert res_keep_runner.returncode == 1
    res_keep_cpu = _run_cli(["idle", "0", "--runner-active", "false"])
    assert res_keep_cpu.returncode == 1
