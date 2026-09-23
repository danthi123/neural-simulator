#!/usr/bin/env python3
"""aws_cost_lib.py — pure spend-arithmetic + idle-decision logic behind the AWS budget guard
(tools/aws_budget.sh, tools/aws_idle_stop.sh). Owner-approved 2026-09-23: on-demand AWS instances for 6-seed
CPU batteries, capped at a daily USD ceiling enforced by TOOLING, not remembered.

Kept dependency-free (stdlib only) and side-effect-free (no `aws`/`ssh` calls in here — those live in the
bash wrappers) so every decision this module makes can be unit-tested with synthetic JSON, per
tests/test_aws_cost_lib.py. The bash wrappers pipe real `aws ec2 describe-instances` /
`aws cloudwatch get-metric-statistics` JSON in on stdin and act on this module's verdict.

SPEND MODEL (a documented approximation — NOT AWS Cost Explorer, which lags ~24h and costs per call):
  For every EC2 instance that is a "project instance" (see `is_project_instance`) and not terminated:
    hours_running_today (UTC calendar day, LaunchTime -> now, counted only while State in running/pending)
      x on-demand hourly price (PRICES below; us-east-1, approximate — edit to update)
    + EBS_DAILY_USD (a small flat constant for the attached root volume, added once per still-existing instance)
  KNOWN LIMITATION: `describe-instances` only reports the CURRENT LaunchTime. If an instance stopped and was
  restarted earlier the same UTC day, the hours it ran before that restart are not recoverable from this API
  alone and are undercounted. This is a conservative, documented trade-off (see the module CLAUDE.md's stance
  on `docs/TERMS.md` unchecked-term discipline) — not a silent bug. `enforce`'s job (stop running instances
  once the cap is hit) is unaffected by it.
"""
import json
import sys
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------------------------------------
# Tagging: how project instances are identified. New launches (tools/aws_cpu_launch.sh, tools/aws_gpu.sh
# `launch`) tag Project=neural-sim going forward. LEGACY_NAME_TAGS covers instances launched before that tag
# existed (aws_cpu_launch.sh has always tagged Name=claude-cpu-verify; the GPU lane's convention added here
# is Name=claude-gpu-verify) so old + new instances are both covered by the same spend estimate.
PROJECT_TAG_KEY = "Project"
PROJECT_TAG_VALUE = "neural-sim"
LEGACY_NAME_TAGS = {"claude-cpu-verify", "claude-gpu-verify"}

# On-demand, us-east-1, approximate as of 2026-09 — verify/update via
# https://aws.amazon.com/ec2/pricing/on-demand/ (or `aws pricing get-products`) and edit this table; it is
# the single source of truth the budget guard prices against.
PRICES = {
    "r7i.4xlarge": 1.0071,   # 16 vCPU / 128 GiB — tools/aws_cpu_launch.sh (the RAM-blocked CPU verify batteries)
    "r7i.2xlarge": 0.5036,
    "g5.xlarge": 1.006,      # 1x A10G 24GB, 4 vCPU / 16 GiB — the GPU type tools/aws_gpu.sh's `launch` uses
    "g5.2xlarge": 1.212,     # 1x A10G 24GB, 8 vCPU / 32 GiB
    "g4dn.xlarge": 0.526,    # 1x T4 16GB — cheaper GPU fallback referenced in past AWS lane specs
    "t3.micro": 0.0104,
}
# An unknown/new instance type must never be silently undercounted -> price it as the most expensive KNOWN
# type until PRICES is updated with its real rate.
_FALLBACK_PRICE = max(PRICES.values())

# Small flat daily allowance for the attached EBS root volume (all launch scripts in this repo request a
# 60GB gp3 DeleteOnTermination volume; gp3 is ~$0.08/GB-month -> 60*0.08/30 ≈ $0.16/day). Rounded up as a
# deliberately conservative constant per the build brief ("+ EBS a small constant").
EBS_DAILY_USD = 0.20

DEFAULT_CAP_USD = 50.0  # owner-approved ceiling, raised 2026-09-23 from an initial $15/day (AWS_DAILY_CAP_USD overrides)

RUNNING_STATES = ("pending", "running")
LIVE_STATES = ("pending", "running", "stopping", "stopped")


# ---------------------------------------------------------------------------------------------------------
# Instance-JSON helpers (operate on one `Instance` dict as returned by `aws ec2 describe-instances`)

def get_tag(instance, key):
    for t in instance.get("Tags") or []:
        if t.get("Key") == key:
            return t.get("Value")
    return None


def is_project_instance(instance):
    """True iff this instance belongs to the neural-sim project (new Project tag, or a known legacy Name)."""
    if get_tag(instance, PROJECT_TAG_KEY) == PROJECT_TAG_VALUE:
        return True
    return get_tag(instance, "Name") in LEGACY_NAME_TAGS


def _parse_iso(ts):
    # AWS emits e.g. '2026-09-23T10:15:32+00:00' or with a trailing 'Z'.
    ts = ts.replace("Z", "+00:00")
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def hours_running_today(instance, now=None):
    """Hours this instance has been running so far today (UTC calendar day), 0.0 if not currently
    running/pending or if LaunchTime is missing/unparseable. See the module docstring for the known
    stop/restart-same-day undercount limitation."""
    now = now or datetime.now(timezone.utc)
    state = (instance.get("State") or {}).get("Name")
    if state not in RUNNING_STATES:
        return 0.0
    launch = instance.get("LaunchTime")
    if not launch:
        return 0.0
    try:
        launch_dt = _parse_iso(launch)
    except (ValueError, TypeError):
        return 0.0
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start = max(launch_dt, day_start)
    if start >= now:
        return 0.0
    return (now - start).total_seconds() / 3600.0


def price_per_hour(instance_type):
    return PRICES.get(instance_type, _FALLBACK_PRICE)


def instance_cost_today(instance, now=None):
    """Return (cost_usd, hours_running_today) for one instance."""
    state = (instance.get("State") or {}).get("Name")
    hours = hours_running_today(instance, now=now)
    compute = hours * price_per_hour(instance.get("InstanceType", ""))
    ebs = EBS_DAILY_USD if state in LIVE_STATES else 0.0
    return compute + ebs, hours


def load_instances(raw_json_text):
    """Flatten a raw `aws ec2 describe-instances --output json` document into a list of Instance dicts."""
    text = (raw_json_text or "").strip()
    if not text:
        return []
    data = json.loads(text)
    out = []
    for res in data.get("Reservations", []):
        out.extend(res.get("Instances", []))
    return out


def estimate_spend(instances, now=None):
    """Return (total_usd, rows) over every project instance in `instances` (any non-terminated state)."""
    rows = []
    total = 0.0
    for inst in instances:
        if not is_project_instance(inst):
            continue
        state = (inst.get("State") or {}).get("Name")
        if state not in LIVE_STATES:
            continue
        cost, hours = instance_cost_today(inst, now=now)
        rows.append({
            "id": inst.get("InstanceId"),
            "type": inst.get("InstanceType"),
            "state": state,
            "hours_today": round(hours, 3),
            "cost_today_usd": round(cost, 4),
        })
        total += cost
    return round(total, 4), rows


def stop_candidates(instances, cap, now=None):
    """Instance IDs to stop right now because today's project spend is at/over `cap`. Only currently
    running/pending instances are candidates (a stopped instance is already not accruing compute cost)."""
    total, rows = estimate_spend(instances, now=now)
    if total < cap:
        return [], total, rows
    ids = [r["id"] for r in rows if r["state"] in RUNNING_STATES]
    return ids, total, rows


# ---------------------------------------------------------------------------------------------------------
# Idle-stop decision logic (tools/aws_idle_stop.sh). Split into small pure pieces so each signal is
# independently testable: CPU-idle (from CloudWatch datapoints OR an SSH load-average fallback) AND
# no-runner-active must BOTH be true before we call an instance idle. Any inconclusive signal -> not idle
# (never stop on missing information; stopping is reversible but a false-positive stop can interrupt a live
# run mid-flight, which is the worse failure mode here).

def is_idle_by_cpu_samples(samples_pct, threshold_pct=10.0):
    """`samples_pct`: CPU utilization percentages (e.g. CloudWatch Datapoints[].Average) covering the idle
    window. Idle iff there is at least one sample AND every sample is below `threshold_pct`. An EMPTY list
    (no CloudWatch data yet) is NOT evidence of idleness -> False."""
    if not samples_pct:
        return False
    return all(s < threshold_pct for s in samples_pct)


def is_idle_by_loadavg(load1, ncpu, threshold_pct=10.0):
    """SSH `uptime` fallback when CloudWatch has no datapoints. `load1` = 1-minute load average, `ncpu` =
    vCPU count. Idle iff (load1/ncpu)*100 < threshold_pct. `ncpu <= 0` is inconclusive -> False."""
    if ncpu is None or ncpu <= 0 or load1 is None:
        return False
    return (load1 / ncpu) * 100.0 < threshold_pct


def idle_decision(cpu_idle, runner_active):
    """Final stop/keep call: idle CPU AND no research runner process. `runner_active=None` (couldn't check,
    e.g. SSH unreachable) is treated as "assume active" -> keep, the conservative default."""
    if runner_active is not False:
        return False
    return bool(cpu_idle)


# ---------------------------------------------------------------------------------------------------------
# CLI — thin argv/stdin/stdout wrapper around the pure functions above, invoked by the bash scripts.

def _read_stdin():
    try:
        return sys.stdin.read()
    except Exception:
        return ""


def _parse_args(rest):
    cap = DEFAULT_CAP_USD
    itype = None
    threshold = 10.0
    load1 = None
    ncpu = None
    runner_active = None
    i = 0
    while i < len(rest):
        a = rest[i]
        if a == "--cap" and i + 1 < len(rest):
            cap = float(rest[i + 1]); i += 2
        elif a == "--type" and i + 1 < len(rest):
            itype = rest[i + 1]; i += 2
        elif a == "--threshold" and i + 1 < len(rest):
            threshold = float(rest[i + 1]); i += 2
        elif a == "--load1" and i + 1 < len(rest):
            load1 = float(rest[i + 1]); i += 2
        elif a == "--ncpu" and i + 1 < len(rest):
            ncpu = int(rest[i + 1]); i += 2
        elif a == "--runner-active" and i + 1 < len(rest):
            v = rest[i + 1].strip().lower()
            runner_active = v in ("1", "true", "yes"); i += 2
        else:
            i += 1
    return {"cap": cap, "type": itype, "threshold": threshold, "load1": load1, "ncpu": ncpu,
            "runner_active": runner_active}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: aws_cost_lib.py {check|status|enforce|project-ids|cpu-idle|loadavg-idle|idle} [opts]",
              file=sys.stderr)
        return 2
    cmd, rest = argv[0], argv[1:]
    opts = _parse_args(rest)

    if cmd in ("check", "status", "enforce", "project-ids"):
        instances = load_instances(_read_stdin())

    if cmd == "status":
        total, rows = estimate_spend(instances)
        print(f"─ AWS BUDGET ─ cap=${opts['cap']:.2f}/day  spend_today≈${total:.2f}")
        if not rows:
            print(f"  (no project instances: Project={PROJECT_TAG_VALUE} tag, or legacy Name in "
                  f"{sorted(LEGACY_NAME_TAGS)})")
        for r in rows:
            print(f"  {r['id']}  {r['type']:<14} {r['state']:<10} {r['hours_today']:>6.2f}h  "
                  f"~${r['cost_today_usd']:.2f}")
        return 0

    if cmd == "check":
        total, _rows = estimate_spend(instances)
        extra = price_per_hour(opts["type"]) if opts["type"] else 0.0
        projected = total + extra
        if projected > opts["cap"]:
            print(f"⛔ aws_budget: refusing — spend_today≈${total:.2f}"
                  f"{f' + first-hour(${extra:.2f})' if extra else ''} = ${projected:.2f} > cap ${opts['cap']:.2f}",
                  file=sys.stderr)
            return 1
        print(f"✓ aws_budget: spend_today≈${total:.2f}"
              f"{f' (+${extra:.2f} if launched)' if extra else ''} <= cap ${opts['cap']:.2f}")
        return 0

    if cmd == "enforce":
        ids, _total, _rows = stop_candidates(instances, opts["cap"])
        for iid in ids:
            print(iid)
        return 0

    if cmd == "project-ids":
        for inst in instances:
            state = (inst.get("State") or {}).get("Name")
            if is_project_instance(inst) and state in RUNNING_STATES:
                print(inst.get("InstanceId"))
        return 0

    if cmd == "cpu-idle":
        raw = _read_stdin()
        data = json.loads(raw) if raw.strip() else {}
        samples = [dp.get("Average") for dp in data.get("Datapoints", []) if "Average" in dp]
        idle = is_idle_by_cpu_samples(samples, threshold_pct=opts["threshold"])
        print(f"samples={len(samples)} idle={idle}", file=sys.stderr)
        return 0 if idle else 1

    if cmd == "cw-has-data":
        # Separates "CloudWatch answered CONCLUSIVELY (idle or busy, either way trust it)" from "CloudWatch
        # has NO datapoints yet (inconclusive, the caller should fall back to SSH load-average)". Without
        # this, a bash `if cpu-idle; then ... elif <ssh fallback> ...` would wrongly treat a CONCLUSIVE
        # "busy" verdict (cpu-idle exits 1) the same as "no data" and needlessly (and, worse, sometimes
        # wrongly) re-ask over SSH.
        raw = _read_stdin()
        data = json.loads(raw) if raw.strip() else {}
        return 0 if data.get("Datapoints") else 1

    if cmd == "loadavg-idle":
        idle = is_idle_by_loadavg(opts["load1"], opts["ncpu"], threshold_pct=opts["threshold"])
        print(f"load1={opts['load1']} ncpu={opts['ncpu']} idle={idle}", file=sys.stderr)
        return 0 if idle else 1

    if cmd == "idle":
        # combine a precomputed cpu_idle (0/1 as argv[1]) with --runner-active
        cpu_idle = bool(rest and rest[0] in ("1", "true", "True"))
        stop = idle_decision(cpu_idle, opts["runner_active"])
        return 0 if stop else 1

    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
