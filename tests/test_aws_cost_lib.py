"""Unit tests for tools/aws_cost_lib.py — the pure spend-arithmetic + idle-decision logic behind the AWS
budget guard (tools/aws_budget.sh, tools/aws_idle_stop.sh; owner-approved 2026-09-23, cap raised same day
$15 -> $50/day). Every function here is side-effect-free (no real `aws`/`ssh` calls), so these run hermetic
and fast against synthetic instance/CloudWatch JSON. Subprocess-level tests against the bash wrappers with a
stubbed `aws`/`ssh` on PATH live in tests/test_aws_budget_guard_workflow.py.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import aws_cost_lib as lib  # noqa: E402
import aws_spend_ledger as ledger  # noqa: E402


NOW = datetime(2026, 9, 23, 18, 0, 0, tzinfo=timezone.utc)
DAY0 = datetime(2026, 9, 23, 0, 0, 0, tzinfo=timezone.utc)   # NOW's UTC calendar-day start


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


# ------------------------------------------------------------------------------ ledger-backed spend (the fix)
# Regression coverage for the 2026-09-23 undercount: `estimate_spend`'s live-only read loses an instance's
# accrued cost the moment it stops or terminates. `estimate_spend_with_ledger` must not. Every test here
# passes its OWN `ledger_file` (a pytest tmp_path file) so it never touches the shared production ledger.

def test_ledger_five_hour_run_then_terminated_still_counts_its_spend(tmp_path):
    lf = str(tmp_path / "ledger.jsonl")
    # Observed once while running (5h in) -> ledger records ~5h x price + EBS.
    running = _instance("i-aaa", "r7i.4xlarge", "running", NOW - timedelta(hours=5))
    total1, rows1, ok1 = lib.estimate_spend_with_ledger([running], now=NOW, ledger_file=lf)
    assert ok1 is True
    expected = 5.0 * lib.PRICES["r7i.4xlarge"] + lib.EBS_DAILY_USD
    assert abs(total1 - round(expected, 4)) < 1e-3
    assert abs(expected - 5.2355) < 1e-3   # "~$5.2", per the build brief's example

    # Next observation: the instance is GONE from the live snapshot entirely (terminated + fallen out of the
    # describe-instances filter) -- exactly what tools/aws_budget.sh's _fetch_instances does today.
    total2, rows2, ok2 = lib.estimate_spend_with_ledger([], now=NOW + timedelta(minutes=10), ledger_file=lf)
    assert ok2 is True
    assert abs(total2 - total1) < 1e-3, "spend vanished after the instance disappeared from the live snapshot"
    row = next(r for r in rows2 if r["id"] == "i-aaa")
    assert row["from_ledger"] is True
    assert row["hours_today"] is None   # honestly reports "we don't know the live hours anymore"


def test_ledger_stopped_instance_keeps_its_larger_recorded_spend():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ledger.jsonl")
        running = _instance("i-bbb", "g5.xlarge", "running", NOW - timedelta(hours=8))
        total_running, _rows, _ok = lib.estimate_spend_with_ledger([running], now=NOW, ledger_file=path)

        # Now observed as `stopped` (SAME LaunchTime -- it was not restarted, just stopped; current live cost
        # recomputes to JUST the EBS constant, per `estimate_spend`'s stopped-instance rule) -- the ledger's
        # earlier, larger recorded period max must win.
        stopped = _instance("i-bbb", "g5.xlarge", "stopped", NOW - timedelta(hours=8))
        total_stopped, rows_stopped, _ok = lib.estimate_spend_with_ledger(
            [stopped], now=NOW + timedelta(minutes=5), ledger_file=path)
        assert abs(total_stopped - total_running) < 1e-3
        row = next(r for r in rows_stopped if r["id"] == "i-bbb")
        assert row["from_ledger"] is True
        assert row["cost_today_usd"] > lib.EBS_DAILY_USD + 1e-6   # not reset down to just the EBS constant


def test_ledger_monotone_max_never_decreases_across_repeated_observations():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ledger.jsonl")
        totals = []
        for hrs_ago, hrs_after in [(1, None), (3, None), (6, None)]:
            inst = _instance("i-ccc", "r7i.4xlarge", "running", NOW - timedelta(hours=hrs_ago))
            t, _rows, _ok = lib.estimate_spend_with_ledger([inst], now=NOW, ledger_file=path)
            totals.append(t)
        assert totals == sorted(totals), f"spend must be non-decreasing as observed hours grow: {totals}"
        # A final observation reporting a SMALLER cost-right-now (e.g. a clock/API hiccup) must not pull the
        # recorded max back down.
        tiny = _instance("i-ccc", "r7i.4xlarge", "running", NOW - timedelta(minutes=1))
        t_final, rows_final, _ok = lib.estimate_spend_with_ledger([tiny], now=NOW, ledger_file=path)
        assert t_final >= totals[-1] - 1e-9
        row = next(r for r in rows_final if r["id"] == "i-ccc")
        assert row["cost_today_usd"] >= totals[-1] - 1e-9


def test_ledger_cap_refuses_launch_after_earlier_instance_terminated():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ledger.jsonl")
        # Mirrors the real 2026-09-23 report: two r7i.4xlarge instances running most of the day (~$12.5
        # together: 6h x $1.0071 + $0.20 EBS, each), then one (i-old) terminates.
        old = _instance("i-old", "r7i.4xlarge", "running", NOW - timedelta(hours=6))
        keep = _instance("i-keep", "r7i.4xlarge", "running", NOW - timedelta(hours=6))
        lib.estimate_spend_with_ledger([old, keep], now=NOW, ledger_file=path)

        # i-old has since terminated (vanished from the live snapshot entirely); i-keep is still running. A
        # naive live-only read now sees only i-keep's ~$6.24 -- the ledgered total must still reflect both.
        total_after, _rows, _ok = lib.estimate_spend_with_ledger([keep], now=NOW + timedelta(minutes=1),
                                                                   ledger_file=path)
        assert total_after > 12.0, "i-old's spend vanished once it dropped out of the live snapshot"

        ids, total, _rows, _ok = lib.stop_candidates_with_ledger([keep], cap=13.0, now=NOW + timedelta(minutes=1),
                                                                    ledger_file=path)
        # The cap math (what `aws_budget.sh check` uses) must see the TRUE total, which is what refuses the
        # launch of a new ~$1/h instance that would cross the $13 cap -- a live-only read of just i-keep
        # (~$6.24 + the new instance's first hour, well under $13) would wrongly allow it.
        extra = lib.price_per_hour("r7i.4xlarge")
        assert total + extra > 13.0, "the launch-time cap check would wrongly allow a launch past the cap"


def test_ledger_seed_backfills_an_instance_that_ended_before_the_ledger_existed():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ledger.jsonl")
        day = ledger.day_key(NOW.timestamp())
        ledger.seed("i-08992a81da0e7f4a6", 7.1652, day, itype="r7i.4xlarge", state="terminated",
                    ledger_file=path)
        total, rows, _ok = lib.estimate_spend_with_ledger([], now=NOW, ledger_file=path)
        assert abs(total - 7.1652) < 1e-3
        row = next(r for r in rows if r["id"] == "i-08992a81da0e7f4a6")
        assert row["from_ledger"] is True


# ------------------------------------------------------------------ 2026-09-23 adversarial-review fix-round
# Four more bugs found reviewing the ledger above, each with a test that FAILS against the pre-fix code and
# PASSES after. See the commit that introduces `period_totals_today` / `record()`'s bool return / the
# state="gone" fix / the try/except in row parsing for the corresponding code change.

def test_ledger_sums_distinct_stop_restart_periods_instead_of_maxing():
    # BLOCKING UNDERCOUNT: a stop+restart resets AWS's LaunchTime, so the OLD max-only ledger compared the two
    # run periods against each other instead of ADDING them. Probe (from the review): a 5h run, stop, restart,
    # then 10h more must total ~$15.31 (5h + 10h of compute, EBS counted ONCE), not ~$10.27 (just the larger
    # of the two periods).
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ledger.jsonl")
        period_a_launch = DAY0 + timedelta(hours=1)
        obs_a_time = period_a_launch + timedelta(hours=5)              # 5h into period A
        inst_a = _instance("i-x", "r7i.4xlarge", "running", period_a_launch)
        lib.estimate_spend_with_ledger([inst_a], now=obs_a_time, ledger_file=path)

        # Stopped, then RESTARTED -> AWS assigns a NEW LaunchTime -> a distinct run period.
        period_b_launch = obs_a_time + timedelta(hours=1)
        obs_b_time = period_b_launch + timedelta(hours=10)             # 10h into period B
        inst_b = _instance("i-x", "r7i.4xlarge", "running", period_b_launch)
        lib.estimate_spend_with_ledger([inst_b], now=obs_b_time, ledger_file=path)

        # Now terminated -- gone from the live snapshot entirely.
        total, _rows, _ok = lib.estimate_spend_with_ledger([], now=obs_b_time + timedelta(minutes=5),
                                                             ledger_file=path)
        expected = 5.0 * lib.PRICES["r7i.4xlarge"] + 10.0 * lib.PRICES["r7i.4xlarge"] + lib.EBS_DAILY_USD
        assert abs(expected - 15.3065) < 1e-3   # "~$15.31" per the review's probe
        assert abs(total - round(expected, 4)) < 1e-2, (
            f"expected ~${expected:.2f} (SUM of both run periods, EBS counted once), got ${total} -- "
            "periods are being MAXED against each other instead of summed")


def test_ledger_upgrade_mid_run_does_not_double_count_legacy_rows_as_a_phantom_period():
    # Found while fixing the stop/restart-period bug above: a instance that keeps running STRAIGHT THROUGH a
    # code upgrade of this ledger writes legacy rows (no launch_time) before the upgrade and tagged rows (with
    # launch_time) after it -- for the SAME real run, never having stopped. Naively treating "no launch_time"
    # as its own period would double-count that instance's spend the moment tagged rows start appearing next
    # to its pre-upgrade rows.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ledger.jsonl")
        launch = DAY0 + timedelta(hours=1)
        inst = _instance("i-mixed", "r7i.4xlarge", "running", launch)

        # Simulate a pre-upgrade observation: write a LEGACY row directly (no launch_time/compute_usd), as the
        # old code would have.
        with open(path, "a") as fh:
            fh.write(json.dumps({"ts": (launch + timedelta(hours=2)).timestamp(),
                                  "day": ledger.day_key((launch + timedelta(hours=2)).timestamp()),
                                  "id": "i-mixed", "type": "r7i.4xlarge", "state": "running",
                                  "accrued_usd": 2.2142}) + "\n")

        # Post-upgrade: the SAME still-running instance observed again, now with the new tagged schema.
        total, rows, _ok = lib.estimate_spend_with_ledger([inst], now=launch + timedelta(hours=5),
                                                           ledger_file=path)
        expected = 5.0 * lib.PRICES["r7i.4xlarge"] + lib.EBS_DAILY_USD   # ONE real run, not two
        assert abs(total - round(expected, 4)) < 1e-2, (
            f"expected ~${expected:.2f} (one continuous run), got ${total} -- the pre-upgrade legacy row was "
            "counted as an extra phantom period")


def test_ledger_skips_malformed_row_instead_of_crashing_and_blinding_other_instances():
    # A single non-numeric `accrued_usd` row must not raise (which would make `check` block ALL launches for
    # the day and `enforce` stop NOTHING, per the review) -- it must be skipped, with a warning, while every
    # OTHER instance's spend is still reported correctly.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ledger.jsonl")
        good = _instance("i-good", "r7i.4xlarge", "running", NOW - timedelta(hours=2))
        lib.estimate_spend_with_ledger([good], now=NOW, ledger_file=path)
        with open(path, "a") as fh:
            fh.write(json.dumps({"ts": NOW.timestamp(), "day": ledger.day_key(NOW.timestamp()),
                                  "id": "i-bad", "type": "r7i.4xlarge", "state": "running",
                                  "accrued_usd": "not-a-number", "compute_usd": "also-not-a-number"}) + "\n")
        total, rows, _ok = lib.estimate_spend_with_ledger([good], now=NOW + timedelta(minutes=1),
                                                           ledger_file=path)
        assert total > 0, "a malformed row for a DIFFERENT instance blinded the whole summary to real spend"
        good_row = next((r for r in rows if r["id"] == "i-good"), None)
        assert good_row is not None and good_row["cost_today_usd"] > 0


def test_ledger_only_instance_marked_gone_not_stale_running_state():
    # An instance that has vanished from the live snapshot (terminated) must not keep displaying whatever
    # state it was LAST recorded in (typically "running") -- `enforce` would try to stop it every cycle
    # forever, and `status` would mislabel it as still live.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ledger.jsonl")
        inst = _instance("i-y", "r7i.4xlarge", "running", NOW - timedelta(hours=3))
        lib.estimate_spend_with_ledger([inst], now=NOW, ledger_file=path)
        _total, rows, _ok = lib.estimate_spend_with_ledger([], now=NOW + timedelta(minutes=1), ledger_file=path)
        row = next(r for r in rows if r["id"] == "i-y")
        assert row["state"] not in lib.RUNNING_STATES, (
            f"ledger-only instance kept a stale LIVE state {row['state']!r} -- enforce would try to stop it "
            "forever")


def test_stop_candidates_with_ledger_excludes_ledger_only_gone_instances():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ledger.jsonl")
        inst = _instance("i-z", "r7i.4xlarge", "running", NOW - timedelta(hours=60))  # push total over any cap
        lib.estimate_spend_with_ledger([inst], now=NOW, ledger_file=path)
        ids, _total, _rows, _ok = lib.stop_candidates_with_ledger([], cap=0.01, now=NOW + timedelta(minutes=1),
                                                                     ledger_file=path)
        assert "i-z" not in ids, "enforce would try to stop an instance that no longer exists"


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

def _run_cli(args, stdin_text="", env=None):
    # `check`/`status`/`enforce` all RECORD to the spend ledger now (see aws_cost_lib.estimate_spend_with_ledger)
    # -- isolate every CLI test's ledger to a throwaway temp file so a test run never writes into the SHARED
    # production research/queue/.aws_spend_ledger.jsonl (mirrors AWS_BUDGET_LOG's test-isolation purpose).
    full_env = dict(os.environ)
    with tempfile.TemporaryDirectory() as td:
        full_env["AWS_SPEND_LEDGER"] = os.path.join(td, "ledger.jsonl")
        if env:
            full_env.update(env)
        return subprocess.run(
            [sys.executable, str(ROOT / "tools" / "aws_cost_lib.py"), *args],
            input=stdin_text, capture_output=True, text=True, timeout=15, env=full_env,
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


def test_cli_status_shows_ledger_only_row_for_vanished_instance(tmp_path):
    # Two separate subprocess invocations sharing ONE ledger path -- proves the ledger really persists
    # across process invocations (the real-world case: two separate `aws_budget.sh status` calls).
    env = {"AWS_SPEND_LEDGER": str(tmp_path / "ledger.jsonl")}
    inst = _instance("i-aaa", "r7i.4xlarge", "running", NOW - timedelta(hours=5))
    res1 = _run_cli(["status", "--cap", "50"], _describe_json([inst]), env=env)
    assert res1.returncode == 0, res1.stderr
    assert "i-aaa" in res1.stdout

    # Instance has now vanished from the live snapshot entirely (terminated, excluded by the
    # describe-instances state filter) -- status must still show its spend, sourced from the ledger.
    res2 = _run_cli(["status", "--cap", "50"], _describe_json([]), env=env)
    assert res2.returncode == 0, res2.stderr
    assert "i-aaa" in res2.stdout
    assert "ledger" in res2.stdout.lower()


def test_cli_check_refuses_after_instance_vanishes_from_live_snapshot(tmp_path):
    # Mirrors the real 2026-09-23 report: two r7i.4xlarge instances running combined, then one (i-old)
    # terminates -- a naive live-only check on just the survivor would wrongly allow another launch.
    # The CLI's `now` is the REAL wall clock (no injection point), so the cap is computed DYNAMICALLY from
    # aws_cost_lib's own formula rather than a number assuming a fixed hours-ago -- otherwise this test could
    # go flaky if it happens to run near a UTC-midnight boundary (hours_running_today clamps to the UTC
    # calendar day).
    real_now = datetime.now(timezone.utc)
    hours_ago = 2
    old = _instance("i-old", "r7i.4xlarge", "running", real_now - timedelta(hours=hours_ago))
    keep = _instance("i-keep", "r7i.4xlarge", "running", real_now - timedelta(hours=hours_ago))
    each_cost, _hrs = lib.instance_cost_today(old, now=real_now)
    combined = 2 * each_cost
    extra = lib.price_per_hour("r7i.4xlarge")
    cap = combined + extra / 2.0   # strictly between "combined" and "combined + a new instance's first hour"

    env = {"AWS_SPEND_LEDGER": str(tmp_path / "ledger.jsonl")}
    res1 = _run_cli(["check", "--cap", f"{cap:.4f}"], _describe_json([old, keep]), env=env)
    assert res1.returncode == 0, res1.stderr   # both still live, combined cost is under cap

    # i-old is now gone from the live snapshot (terminated); i-keep is still live and alone accrues only
    # half the combined cost -- the ledgered check must still see both, and refuse the new instance's
    # first-hour addition.
    res2 = _run_cli(["check", "--cap", f"{cap:.4f}", "--type", "r7i.4xlarge"], _describe_json([keep]), env=env)
    assert res2.returncode == 1, res2.stderr


def test_cli_check_fails_closed_when_ledger_write_fails(tmp_path):
    # A cap CHECK must never proceed on spend it could not durably record -- that is the whole point of the
    # ledger. Force a write failure by pointing AWS_SPEND_LEDGER at a path whose PARENT component is a plain
    # file (so `os.makedirs` cannot create it), and require `check` to refuse the launch (exit 1, stderr).
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("this is a file, not a directory")
    env = {"AWS_SPEND_LEDGER": str(blocker / "ledger.jsonl")}
    inst = _instance("i-aaa", "r7i.4xlarge", "running", NOW - timedelta(hours=1))
    res = _run_cli(["check", "--cap", "1000"], _describe_json([inst]), env=env)
    assert res.returncode == 1, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert res.stderr.strip() != "", "a fail-closed refusal must say why on stderr"


def test_cli_status_warns_but_does_not_crash_when_ledger_write_fails(tmp_path):
    # `status` is informational, not a launch gate -- it must WARN (not silently succeed, not crash) rather
    # than fail closed like `check`.
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("this is a file, not a directory")
    env = {"AWS_SPEND_LEDGER": str(blocker / "ledger.jsonl")}
    inst = _instance("i-aaa", "r7i.4xlarge", "running", NOW - timedelta(hours=1))
    res = _run_cli(["status", "--cap", "1000"], _describe_json([inst]), env=env)
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert res.stderr.strip() != "", "status must still surface a warning that the ledger write failed"


def test_cli_enforce_stays_best_effort_when_ledger_write_fails(tmp_path):
    # `enforce`'s job is to stop instances over cap; it must not let a ledger-write failure stop it from doing
    # that (best-effort, per the review).
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("this is a file, not a directory")
    env = {"AWS_SPEND_LEDGER": str(blocker / "ledger.jsonl")}
    inst = _instance("i-aaa", "r7i.4xlarge", "running", NOW - timedelta(hours=40))  # big accrued cost
    res = _run_cli(["enforce", "--cap", "1.0"], _describe_json([inst]), env=env)
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert "i-aaa" in res.stdout.split(), "enforce must still identify the over-cap instance to stop"


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
