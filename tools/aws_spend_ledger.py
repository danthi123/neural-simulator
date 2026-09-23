#!/usr/bin/env python3
"""aws_spend_ledger.py — persistent, append-only, per-instance MONOTONE-MAX spend ledger that closes the
"spend vanishes when an instance stops/terminates" gap in tools/aws_cost_lib.py's live-only estimate (see its
module docstring's former KNOWN LIMITATION). Measured 2026-09-23: two r7i.4xlarge instances had run ~$12
today; once one (i-08992a81da0e7f4a6) was stopped and terminated, `aws_budget.sh status` read $5.31 — its
already-accrued hours vanished because the live formula only ever reads the CURRENT LaunchTime of instances
that still exist in a `describe-instances` response filtered to pending/running/stopping/stopped (a
stop+restart shortly before termination resets LaunchTime too, compounding the loss). Under the owner-approved
$50/day cap, that undercount is a real escape hatch: launching new instances after old ones end could exceed
the cap while `check` still reports comfortable headroom.

Modeled on tools/waiver_history.py's append-only-ledger discipline: same `shared_root()` resolution (so every
worktree agrees on ONE physical file — git-common-dir based, SIM_QUEUE_ROOT-overridable), same fail-open-on-
missing/corrupt-file posture (a ledger read/write failure must never crash or block the budget guard), same
best-effort append.

MODEL. Every `aws_cost_lib.py status|check|enforce` invocation (tools/aws_budget.sh) observes the live
`describe-instances` JSON and RECORDS one row per project instance via `record()`: {ts, day (UTC calendar
day), id, type, state, accrued_usd}, where accrued_usd is that instance's CURRENT-observation
cost-so-far-today (tools.aws_cost_lib.instance_cost_today — compute-hours x price, + the flat EBS constant
while the instance still exists). `summary_today()` then reports, for every instance-id seen TODAY, the MAX
accrued_usd ever recorded for it — monotone, so a later observation of a stopped/terminated instance (whose
live-right-now cost recomputes to a smaller number, or zero once terminated/vanished) can never erase an
earlier, larger recorded max. `tools.aws_cost_lib.estimate_spend_with_ledger()` is the caller that combines
this with the live snapshot.

This is a documented approximation, not exact continuous accounting: it only knows what it was told between
polls (the aws-guard.service timer polls every 10 min via `tools/aws_idle_stop.sh` + `aws_budget.sh enforce`,
and any `status`/`check` call records too), so an instance created AND torn down entirely between two polls is
undercounted by up to one polling interval's cost. That residual gap is small, bounded, and can only
UNDER-count — never over-count — unlike the bug this closes, which could lose 100% of an instance's accrued
cost outright. Also unlike the pre-fix code, it needs no config change if the polling interval changes; a
tighter interval just tightens the bound.

Known residual (same as `tools/waiver_history.py`'s): the ledger file is not tamper-evident, and an agent that
deletes it resets today's recorded history for whatever hasn't been re-observed since.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from waiver_history import shared_root  # noqa: E402  (the one git-common-dir resolver; see its docstring)

# AWS_SPEND_LEDGER overrides the path — tests use this to keep from writing into the SHARED production queue
# dir (other sessions/agents touch research/queue/ concurrently; see the worktree environment note on this,
# and tools/aws_budget.sh's AWS_BUDGET_LOG for the identical pattern).
LEDGER_FILE = os.environ.get("AWS_SPEND_LEDGER") or os.path.join(
    shared_root(), "research", "queue", ".aws_spend_ledger.jsonl")


def day_key(ts=None):
    """UTC calendar day string 'YYYY-MM-DD' for `ts` (unix seconds; default: now). Matches
    tools.aws_cost_lib.hours_running_today's UTC-calendar-day convention so a "today" in one module is the
    same "today" in the other."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc) if ts is not None else datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%d")


def _load_rows(ledger_file):
    if not os.path.exists(ledger_file):
        return []
    try:
        lines = open(ledger_file, errors="ignore").read().splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("id") and row.get("day") and "accrued_usd" in row:
            rows.append(row)
    return rows


def _append(row, ledger_file):
    """Best-effort append — a ledger-write failure must never block or crash the caller (mirrors
    waiver_history._append)."""
    try:
        os.makedirs(os.path.dirname(ledger_file), exist_ok=True)
        with open(ledger_file, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    except OSError:
        pass


def record(rows, ts=None, ledger_file=None):
    """Append one ledger row per instance in `rows` (the per-instance dicts `aws_cost_lib.estimate_spend`
    returns, or any dict of the same shape — needs at least "id" and "cost_today_usd"). `ts` is injectable for
    tests; `ledger_file` defaults to `LEDGER_FILE`. Best-effort: never raises."""
    ledger_file = ledger_file if ledger_file is not None else LEDGER_FILE
    ts = ts if ts is not None else time.time()
    day = day_key(ts)
    for r in rows:
        iid = r.get("id")
        if not iid:
            continue
        _append({"ts": ts, "day": day, "id": iid, "type": r.get("type"), "state": r.get("state"),
                 "accrued_usd": round(float(r.get("cost_today_usd") or 0.0), 4)}, ledger_file)


def summary_today(day=None, ledger_file=None):
    """{instance_id: {"accrued_usd": <MAX ever recorded for `day`>, "type":, "state":, "ts": <of that max
    row>}} for `day` (default: today, UTC). The max-holding row's type/state are kept for display — it is
    typically the last observation while the instance was still running, so they are meaningful (not a
    "vanished"/None placeholder)."""
    ledger_file = ledger_file if ledger_file is not None else LEDGER_FILE
    day = day or day_key()
    out = {}
    for row in _load_rows(ledger_file):
        if row.get("day") != day:
            continue
        iid = row["id"]
        val = float(row.get("accrued_usd") or 0.0)
        cur = out.get(iid)
        if cur is None or val >= cur["accrued_usd"]:
            out[iid] = {"accrued_usd": val, "type": row.get("type"), "state": row.get("state"),
                        "ts": row.get("ts")}
    return out


def seed(instance_id, accrued_usd, day, itype=None, state=None, ts=None, ledger_file=None, note=None):
    """Manually seed one historical observation — used for a backfill of an instance whose spend today ended
    (stopped/terminated) BEFORE this ledger existed to observe it, so it would otherwise be lost forever (see
    the one-time seed of i-08992a81da0e7f4a6 recorded in the FIX commit that introduced this module). Not
    part of the normal `status`/`check`/`enforce` recording path."""
    ledger_file = ledger_file if ledger_file is not None else LEDGER_FILE
    ts = ts if ts is not None else time.time()
    row = {"ts": ts, "day": day, "id": instance_id, "type": itype, "state": state,
           "accrued_usd": round(float(accrued_usd), 4), "seeded": True}
    if note:
        row["note"] = note
    _append(row, ledger_file)
    return row


def _cli(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--selftest" in argv:
        bad = selftest()
        if bad:
            print("aws_spend_ledger selftest FAILED:\n  - " + "\n  - ".join(bad), file=sys.stderr)
            return 1
        print("aws_spend_ledger selftest PASSED (monotone max, day isolation, seed round-trip, "
              "missing/corrupt-file handling).")
        return 0
    p = argparse.ArgumentParser(prog="aws_spend_ledger.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    s_seed = sub.add_parser("seed", help="manually backfill one historical observation")
    s_seed.add_argument("--id", required=True, dest="instance_id")
    s_seed.add_argument("--usd", required=True, type=float, dest="accrued_usd")
    s_seed.add_argument("--day", required=True, help="UTC calendar day, YYYY-MM-DD")
    s_seed.add_argument("--type", dest="itype", default=None)
    s_seed.add_argument("--state", default=None)
    s_seed.add_argument("--note", default=None)

    s_show = sub.add_parser("show", help="print today's (or --day's) recorded max per instance")
    s_show.add_argument("--day", default=None)

    args = p.parse_args(argv)
    if args.cmd == "seed":
        row = seed(args.instance_id, args.accrued_usd, args.day, itype=args.itype, state=args.state,
                   note=args.note)
        print(f"seeded {row['id']} day={row['day']} accrued_usd={row['accrued_usd']:.4f} "
              f"-> {LEDGER_FILE}")
        return 0
    if args.cmd == "show":
        summ = summary_today(day=args.day)
        if not summ:
            print(f"(no ledger rows for day={args.day or day_key()} in {LEDGER_FILE})")
            return 0
        for iid, row in sorted(summ.items()):
            print(f"  {iid}  {row.get('type') or '?':<14} {row.get('state') or '?':<12} "
                  f"~${row['accrued_usd']:.2f}")
        return 0
    return 2


def selftest():
    """FAILING DIRECTION FIRST: the max must never decrease on a smaller/absent later observation, a seed
    must be readable back, and rows outside the requested day must not leak in."""
    import tempfile
    bad = []
    with tempfile.TemporaryDirectory() as td:
        lf = os.path.join(td, ".ledger.jsonl")
        t0 = 1_800_000_000.0  # an arbitrary fixed epoch, well inside a known UTC day
        day = day_key(t0)

        # (a) monotone max: an instance's recorded cost rises while running, then a SMALLER later observation
        # (as if freshly stopped, or terminated and recomputed as $0 live) must not erase the peak.
        record([{"id": "i-aaa", "type": "r7i.4xlarge", "state": "running", "cost_today_usd": 1.0}],
               ts=t0, ledger_file=lf)
        record([{"id": "i-aaa", "type": "r7i.4xlarge", "state": "running", "cost_today_usd": 5.0}],
               ts=t0 + 3600, ledger_file=lf)
        record([{"id": "i-aaa", "type": "r7i.4xlarge", "state": "stopped", "cost_today_usd": 0.20}],
               ts=t0 + 7200, ledger_file=lf)
        record([{"id": "i-aaa", "type": "r7i.4xlarge", "state": "terminated", "cost_today_usd": 0.0}],
               ts=t0 + 10800, ledger_file=lf)
        summ = summary_today(day=day, ledger_file=lf)
        if abs(summ["i-aaa"]["accrued_usd"] - 5.0) > 1e-9:
            bad.append("MONOTONE MAX VIOLATED: a smaller later observation changed the recorded max "
                       f"(got {summ.get('i-aaa')})")

        # (b) a day this instance was never observed on must not see its spend.
        other_day_summ = summary_today(day="1999-01-01", ledger_file=lf)
        if "i-aaa" in other_day_summ:
            bad.append("DAY LEAKAGE: a row from one UTC day appeared under a different day's summary")

        # (c) seed() must be readable back exactly like a normal observation.
        seed("i-seeded", 7.1652, day, itype="r7i.4xlarge", state="terminated", ts=t0, ledger_file=lf)
        summ2 = summary_today(day=day, ledger_file=lf)
        if "i-seeded" not in summ2 or abs(summ2["i-seeded"]["accrued_usd"] - 7.1652) > 1e-6:
            bad.append("SEED NOT READABLE: a seeded row did not round-trip through summary_today")

        # (d) a missing ledger file must fail OPEN (empty summary, not a crash).
        missing = os.path.join(td, "does-not-exist.jsonl")
        if summary_today(day=day, ledger_file=missing):
            bad.append("FALSE POSITIVE: a missing ledger file produced non-empty spend")

        # (e) corrupt lines must be skipped, not crash the reader.
        with open(lf, "a") as fh:
            fh.write("{not json\n")
        try:
            summary_today(day=day, ledger_file=lf)
        except Exception as e:  # noqa: BLE001
            bad.append(f"CORRUPT-LINE CRASH: reading a ledger with one malformed line raised {e!r}")
    return bad


if __name__ == "__main__":
    sys.exit(_cli())
