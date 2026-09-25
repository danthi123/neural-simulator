#!/usr/bin/env python3
"""aws_spend_ledger.py — persistent, append-only spend ledger that closes the "spend vanishes when an
instance stops/terminates" gap in tools/aws_cost_lib.py's live-only estimate (see its module docstring's
former KNOWN LIMITATION). Measured 2026-09-23: two r7i.4xlarge instances had run ~$12 today; once one
(i-08992a81da0e7f4a6) was stopped and terminated, `aws_budget.sh status` read $5.31 — its already-accrued
hours vanished because the live formula only ever reads the CURRENT LaunchTime of instances that still exist
in a `describe-instances` response filtered to pending/running/stopping/stopped.

Modeled on tools/waiver_history.py's append-only-ledger discipline: same `shared_root()` resolution (so every
worktree agrees on ONE physical file — git-common-dir based, SIM_QUEUE_ROOT-overridable), same fail-open-on-
missing/corrupt-file read posture. Unlike waiver_history, a WRITE failure is surfaced to the caller (`record()`
returns bool) rather than swallowed — see "FAIL-CLOSED WRITES" below.

RUN PERIODS, not just instances (2026-09-23 review round 2 — BLOCKING). A stop+restart resets AWS's own
`LaunchTime`, so a ledger keyed by instance-id ALONE and taking a single running MAX across all of today's rows
compares two separate run periods against each other instead of ADDING them: a probe of a 5h run, stop,
restart, then 10h more read $10.27 (just the larger of the two periods) against a true ~$15.31 (5h + 10h of
compute, EBS counted once). Every recorded row now carries the instance's `launch_time` (AWS's own field, raw)
so a genuinely NEW run period is distinguishable from a repeated observation of the SAME period. `record()`
also carries `compute_usd` (hours x price, EBS excluded) SEPARATELY from `accrued_usd` (the old combined
field, kept for display/back-compat) so a per-period max can be summed across periods without multiplying the
once-per-day EBS constant by the number of stop/restart cycles. `period_totals_today()` is the aggregator:
per instance, MAX within each (instance, launch_time) period, then SUM across periods, then the caller
(aws_cost_lib, which owns EBS_DAILY_USD) adds EBS once.

KNOWN LIMITATION (rows written before this fix, or by any other caller that omits `launch_time`/`compute_usd`):
such a row cannot be assigned to a real run period, so it is bucketed into one synthetic "unknown period" per
instrument, and its "compute" is approximated from its own `accrued_usd` (which already includes that
observation's own EBS bump). Multiple genuinely-distinct STOP/RESTART periods that both lack `launch_time`
will therefore still be MAXED against each other, not summed — the exact pre-fix behavior — deliberately,
because assuming they're different periods when they might be the same one would be an OVER-count, and this
module's invariant is "can only under-count, never over-count" (see below). This can only affect rows written
by a pre-fix build of this module; every row written by the current `record()` carries both fields.

FAIL-CLOSED WRITES (2026-09-23 review round 2). `_append`/`record()` now RETURN whether the write succeeded,
instead of the old best-effort-and-silently-swallow. `tools/aws_cost_lib.py`'s `check` (a launch-time cap
decision) fails CLOSED — refuses the launch — when a write fails, because a cap check must never proceed on
spend it could not durably record. `status` (informational) surfaces a warning instead of failing. `enforce`
(stop instances already over cap) stays best-effort: a ledger write failure must not stop it from acting on
whatever it already knows.

MALFORMED ROWS (2026-09-23 review round 2). A single non-numeric field in one row used to raise inside the
aggregator, uncaught — which meant `check` blocked EVERY launch for the rest of the day and `enforce` stopped
NOTHING, no matter how obviously over cap other instances were. `period_totals_today()` now wraps each row's
numeric parsing in try/except: a bad row is skipped (with a warning to stderr), every other row is still
aggregated normally.

GONE, not a stale state (2026-09-23 review round 2). An instance with no row in the CURRENT live snapshot used
to keep whatever state its highest-cost historical row carried (typically "running") when synthesized into a
ledger-only display row. `tools.aws_cost_lib.estimate_spend_with_ledger` now hard-codes `state="gone"` for
those rows regardless of ledger history, so `enforce` never treats a long-terminated instance as a stop
candidate and `status` never mislabels it as live.

MODEL (aggregate). Every `aws_cost_lib.py status|check|enforce` invocation (tools/aws_budget.sh) observes the
live `describe-instances` JSON and RECORDS one row per project instance via `record()`. `period_totals_today()`
reports, for every instance-id seen TODAY, the SUM across today's distinct run periods of each period's peak
recorded compute cost — monotone WITHIN a period (a later, smaller observation of the SAME period can never
erase that period's peak), summed ACROSS periods (so a stop+restart adds to, rather than replaces, the
already-accrued total). `tools.aws_cost_lib.estimate_spend_with_ledger()` is the caller that combines this
with the live snapshot and adds the once-per-day EBS constant.

This is a documented approximation, not exact continuous accounting: it only knows what it was told between
polls (the aws-guard.service timer polls every 10 min), so an instance created AND torn down entirely between
two polls is undercounted by up to one polling interval's cost. That residual gap is small, bounded, and can
only UNDER-count — never over-count — unlike the bug this module exists to close, which could lose 100% of an
instance's accrued cost outright.

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


PRODUCTION_LEDGER_FILE = os.path.join(shared_root(), "research", "queue", ".aws_spend_ledger.jsonl")


def _is_production_ledger(path):
    try:
        return os.path.realpath(path) == os.path.realpath(PRODUCTION_LEDGER_FILE)
    except (OSError, TypeError, ValueError):
        return True   # cannot tell: treat as production (fail closed for the guard)


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
    """Append one row. Returns True on success, False on any OSError (disk full, unwritable path, a path
    component that is a plain file, ...). A write FAILURE must never raise — the caller decides what a
    failed write means for it (fail closed / warn / best-effort; see the module docstring)."""
    try:
        os.makedirs(os.path.dirname(ledger_file), exist_ok=True)
        with open(ledger_file, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        return True
    except OSError:
        return False


def record(rows, ts=None, ledger_file=None):
    """Append one ledger row per instance in `rows` (dicts with at least "id" and "cost_today_usd"; "type",
    "state", "launch_time", "hours_today", "compute_usd" are recorded when present — `launch_time`/
    `compute_usd` are what let `period_totals_today` distinguish and correctly sum stop/restart periods, see
    the module docstring). `ts` is injectable for tests; `ledger_file` defaults to `LEDGER_FILE`.

    Returns True iff EVERY row was durably written, False if ANY write failed (including "no rows to write"
    trivially returning True). Never raises."""
    ledger_file = ledger_file if ledger_file is not None else LEDGER_FILE
    # TEST-ISOLATION GUARD (2026-09-25): a test helper that forgot AWS_SPEND_LEDGER wrote a stub instance
    # ("i-existing", launch 2020-01-01) into the PRODUCTION ledger on 2026-09-24 and again on 2026-09-25; it added
    # ~$14.8 of phantom spend to "today" and would have made aws-guard stop both real pool nodes at the cap
    # mid-job. Under pytest (PYTEST_CURRENT_TEST is inherited by every subprocess a test starts) the production
    # ledger is never written; a test that needs a ledger must point AWS_SPEND_LEDGER at its own tmp file.
    if os.environ.get("PYTEST_CURRENT_TEST") and _is_production_ledger(ledger_file):
        print("[aws_spend_ledger] refusing to write the production spend ledger from a test "
              "(set AWS_SPEND_LEDGER to a tmp file)", file=sys.stderr)
        return False
    ts = ts if ts is not None else time.time()
    day = day_key(ts)
    ok = True
    for r in rows:
        iid = r.get("id")
        if not iid:
            continue
        row = {
            "ts": ts, "day": day, "id": iid, "type": r.get("type"), "state": r.get("state"),
            "launch_time": r.get("launch_time"),
            "hours_today": r.get("hours_today"),
            "compute_usd": r.get("compute_usd"),
            "accrued_usd": round(float(r.get("cost_today_usd") or 0.0), 4),
        }
        if not _append(row, ledger_file):
            ok = False
    return ok


def period_totals_today(day=None, ledger_file=None, warn=True):
    """{instance_id: {"compute_usd": <SUM across today's distinct run periods of each period's PEAK recorded
    compute cost, EBS excluded>, "has_compute_field": <bool>, "type":, "state":, "ts": <of the most recent
    contributing row>}} for `day` (default: today, UTC).

    A "run period" is keyed by (instance_id, launch_time) — see the module docstring's BLOCKING fix. Within
    one period, later/smaller observations never pull the period's recorded peak back down (the original
    monotone-max property, now scoped to a period instead of the whole day). Across DIFFERENT periods for the
    same instance, peaks are SUMMED, not maxed — a stop+restart's cost adds to what came before it.

    `has_compute_field` is False only when NONE of an instance's rows today carry an explicit `compute_usd`
    (i.e. every row is legacy/pre-fix) — the caller (aws_cost_lib) must not add its once-per-day EBS constant
    on top in that case, because the legacy fallback already folds an EBS bump into its "compute" estimate
    (see the module docstring's KNOWN LIMITATION); it MUST add EBS once when `has_compute_field` is True.

    A LEGACY row (no `launch_time`, e.g. written by a pre-2026-09-23 build of this module) cannot be assigned
    a real period key. If the SAME instance also has at least one properly-tagged row today, each legacy row
    is folded into whichever tagged period is CLOSEST IN TIME (as just another candidate for that period's
    peak) rather than treated as its own extra period — otherwise a still-running instance whose ledger rows
    started before a code upgrade and continued after it would have its pre-upgrade rows double-counted as a
    phantom SECOND period, purely from the schema change, even though it never actually stopped. Legacy rows
    are chronologically first in practice (a legacy row can only be written by code older than any tagged
    row), so "closest in time" reduces to "the earliest tagged period" — but is computed generally rather than
    assumed, in case rows are read out of write order. If an instance has NO tagged rows at all today (a purely
    historical/never-migrated instance), its legacy rows fall back to the OLD single whole-day-max behavior
    among themselves — this can only ever UNDER-count two genuinely distinct untagged periods as one (folding
    what might be 2+ periods' peaks together, never inventing an extra one), preserving this module's
    can-only-under-count invariant in both directions.

    A row with a non-numeric numeric field is SKIPPED (with a warning to stderr unless `warn=False`) rather
    than raising — see the module docstring's MALFORMED ROWS fix. `check`/`enforce` must never be blinded to
    every OTHER instance's spend by one corrupt row."""
    ledger_file = ledger_file if ledger_file is not None else LEDGER_FILE
    day = day or day_key()
    by_instance = {}
    for row in _load_rows(ledger_file):
        if row.get("day") != day:
            continue
        iid = row.get("id")
        if not iid:
            continue
        try:
            launch_time = row.get("launch_time")
            raw_compute = row.get("compute_usd")
            has_compute = raw_compute is not None
            if raw_compute is None:
                raw_compute = row.get("accrued_usd")   # legacy fallback -- see KNOWN LIMITATION
            compute = float(raw_compute or 0.0)
            ts = float(row.get("ts") or 0.0)
        except (TypeError, ValueError) as exc:
            if warn:
                print(f"⚠️  aws_spend_ledger: skipping malformed row for instance {iid!r} in "
                      f"{ledger_file} ({exc.__class__.__name__}: {exc}) — its OTHER rows/instances are "
                      f"still counted normally", file=sys.stderr)
            continue
        entry = by_instance.setdefault(
            iid, {"tagged": {}, "legacy": [], "has_compute_field": False,
                  "type": None, "state": None, "ts": 0.0})
        if launch_time:
            period = entry["tagged"].setdefault(launch_time, {"max": 0.0, "ts_list": []})
            if compute > period["max"]:
                period["max"] = compute
            period["ts_list"].append(ts)
        else:
            entry["legacy"].append((ts, compute))
        if has_compute:
            entry["has_compute_field"] = True
        if ts >= entry["ts"]:
            entry["ts"] = ts
            entry["type"] = row.get("type") or entry["type"]
            entry["state"] = row.get("state") or entry["state"]

    out = {}
    for iid, entry in by_instance.items():
        tagged, legacy = entry["tagged"], entry["legacy"]
        if tagged:
            # Fold each legacy row into its nearest-in-time TAGGED period -- see the docstring above.
            for ts, compute in legacy:
                nearest = min(tagged, key=lambda k: min(abs(ts - t) for t in tagged[k]["ts_list"]))
                if compute > tagged[nearest]["max"]:
                    tagged[nearest]["max"] = compute
            total_compute = sum(period["max"] for period in tagged.values())
        elif legacy:
            total_compute = max(compute for _ts, compute in legacy)
        else:
            total_compute = 0.0
        out[iid] = {
            "compute_usd": round(total_compute, 6),
            "has_compute_field": entry["has_compute_field"],
            "type": entry["type"], "state": entry["state"], "ts": entry["ts"],
        }
    return out


def seed(instance_id, accrued_usd, day, itype=None, state=None, ts=None, ledger_file=None, note=None,
         launch_time=None, compute_usd=None):
    """Manually seed one historical observation — used for a backfill of an instance whose spend today ended
    (stopped/terminated) BEFORE this ledger existed to observe it, so it would otherwise be lost forever (see
    the one-time seed of i-08992a81da0e7f4a6 recorded in the commit that introduced this module). Not part of
    the normal `status`/`check`/`enforce` recording path.

    `launch_time`/`compute_usd` are optional but recommended (see `period_totals_today`'s docstring): without
    them the seeded row falls into the legacy "unknown period" bucket for this instance."""
    ledger_file = ledger_file if ledger_file is not None else LEDGER_FILE
    ts = ts if ts is not None else time.time()
    row = {"ts": ts, "day": day, "id": instance_id, "type": itype, "state": state,
           "launch_time": launch_time, "compute_usd": compute_usd,
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
        print("aws_spend_ledger selftest PASSED (period summing across stop/restart, malformed-row "
              "skip, seed round-trip, missing/corrupt-file handling, fail-closed write signaling).")
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
    s_seed.add_argument("--launch-time", dest="launch_time", default=None,
                         help="the instance's raw AWS LaunchTime (ISO8601), if known")
    s_seed.add_argument("--compute-usd", dest="compute_usd", default=None, type=float,
                         help="the compute-only cost (hours x price, EBS excluded), if known")

    s_show = sub.add_parser("show", help="print today's (or --day's) recorded total per instance")
    s_show.add_argument("--day", default=None)

    args = p.parse_args(argv)
    if args.cmd == "seed":
        row = seed(args.instance_id, args.accrued_usd, args.day, itype=args.itype, state=args.state,
                   note=args.note, launch_time=args.launch_time, compute_usd=args.compute_usd)
        print(f"seeded {row['id']} day={row['day']} accrued_usd={row['accrued_usd']:.4f} "
              f"-> {LEDGER_FILE}")
        return 0
    if args.cmd == "show":
        totals = period_totals_today(day=args.day)
        if not totals:
            print(f"(no ledger rows for day={args.day or day_key()} in {LEDGER_FILE})")
            return 0
        for iid, row in sorted(totals.items()):
            print(f"  {iid}  {row.get('type') or '?':<14} {row.get('state') or '?':<12} "
                  f"~${row['compute_usd']:.2f} compute (has_compute_field={row['has_compute_field']})")
        return 0
    return 2


def selftest():
    """FAILING DIRECTION FIRST: stop/restart periods must SUM not MAX, a malformed row must not crash and must
    not blind other instances, a write failure must be reported (not swallowed), a seed must round-trip, and
    rows outside the requested day must not leak in."""
    import tempfile
    bad = []
    with tempfile.TemporaryDirectory() as td:
        lf = os.path.join(td, ".ledger.jsonl")
        t0 = 1_800_000_000.0  # an arbitrary fixed epoch, well inside a known UTC day
        day = day_key(t0)

        # (a) WITHIN one run period (same launch_time): monotone max, a smaller later observation of the SAME
        # period must not erase the peak.
        record([{"id": "i-aaa", "type": "r7i.4xlarge", "state": "running", "launch_time": "L1",
                 "hours_today": 1.0, "compute_usd": 1.0071, "cost_today_usd": 1.2071}],
               ts=t0, ledger_file=lf)
        record([{"id": "i-aaa", "type": "r7i.4xlarge", "state": "running", "launch_time": "L1",
                 "hours_today": 5.0, "compute_usd": 5.0355, "cost_today_usd": 5.2355}],
               ts=t0 + 3600, ledger_file=lf)
        record([{"id": "i-aaa", "type": "r7i.4xlarge", "state": "stopped", "launch_time": "L1",
                 "hours_today": 0.0, "compute_usd": 0.0, "cost_today_usd": 0.20}],
               ts=t0 + 7200, ledger_file=lf)
        totals = period_totals_today(day=day, ledger_file=lf)
        if abs(totals["i-aaa"]["compute_usd"] - 5.0355) > 1e-6:
            bad.append("WITHIN-PERIOD MONOTONE MAX VIOLATED: a smaller later observation of the SAME period "
                       f"changed the recorded peak (got {totals.get('i-aaa')})")

        # (b) ACROSS a stop+restart (a NEW launch_time): the two periods must SUM, not max.
        record([{"id": "i-aaa", "type": "r7i.4xlarge", "state": "running", "launch_time": "L2",
                 "hours_today": 10.0, "compute_usd": 10.071, "cost_today_usd": 10.271}],
               ts=t0 + 10800, ledger_file=lf)
        totals = period_totals_today(day=day, ledger_file=lf)
        expected = 5.0355 + 10.071   # period L1's peak + period L2's peak
        if abs(totals["i-aaa"]["compute_usd"] - expected) > 1e-6:
            bad.append("CROSS-PERIOD SUM VIOLATED: expected periods L1+L2 to SUM to %.4f, got %s"
                       % (expected, totals.get("i-aaa")))
        if not totals["i-aaa"]["has_compute_field"]:
            bad.append("has_compute_field FALSE POSITIVE-NEGATIVE: rows with compute_usd were not detected")

        # (b2) MIXED legacy + tagged rows for a CONTINUOUSLY-RUNNING instance (the exact scenario of upgrading
        # this module's code mid-run): pre-fix rows with no launch_time, then post-fix tagged rows for the
        # SAME real run. These must fold into ONE period, not double-count as legacy-period + tagged-period.
        record([{"id": "i-mixed", "type": "r7i.4xlarge", "state": "running",
                 "hours_today": 2.0, "cost_today_usd": 2.2142}],   # legacy: no launch_time/compute_usd at all
               ts=t0, ledger_file=lf)
        record([{"id": "i-mixed", "type": "r7i.4xlarge", "state": "running", "launch_time": "LM",
                 "hours_today": 5.0, "compute_usd": 5.0355, "cost_today_usd": 5.2355}],
               ts=t0 + 3600, ledger_file=lf)   # SAME real run, observed again after a code upgrade
        totals = period_totals_today(day=day, ledger_file=lf)
        if abs(totals["i-mixed"]["compute_usd"] - 5.0355) > 1e-6:
            bad.append("LEGACY-TO-TAGGED DOUBLE COUNT: a continuously-running instance's pre-upgrade legacy "
                       f"row was treated as an extra phantom period instead of folded into the same real run "
                       f"(expected 5.0355, got {totals.get('i-mixed')})")

        # (c) a day this instance was never observed on must not see its spend.
        other_day = period_totals_today(day="1999-01-01", ledger_file=lf)
        if "i-aaa" in other_day:
            bad.append("DAY LEAKAGE: a row from one UTC day appeared under a different day's totals")

        # (d) seed() must be readable back exactly like a normal observation.
        seed("i-seeded", 7.1652, day, itype="r7i.4xlarge", state="terminated", ts=t0,
             ledger_file=lf, launch_time="LS", compute_usd=6.9652)
        totals = period_totals_today(day=day, ledger_file=lf)
        if "i-seeded" not in totals or abs(totals["i-seeded"]["compute_usd"] - 6.9652) > 1e-6:
            bad.append("SEED NOT READABLE: a seeded row did not round-trip through period_totals_today")

        # (e) a missing ledger file must fail OPEN (empty totals, not a crash).
        missing = os.path.join(td, "does-not-exist.jsonl")
        if period_totals_today(day=day, ledger_file=missing):
            bad.append("FALSE POSITIVE: a missing ledger file produced non-empty spend")

        # (f) corrupt JSON lines must be skipped, not crash the reader.
        with open(lf, "a") as fh:
            fh.write("{not json\n")
        try:
            period_totals_today(day=day, ledger_file=lf)
        except Exception as e:  # noqa: BLE001
            bad.append(f"CORRUPT-LINE CRASH: reading a ledger with one malformed JSON line raised {e!r}")

        # (g) a MALFORMED NUMERIC FIELD (valid JSON, bad value) must be skipped without crashing, and must not
        # blind other instances' totals.
        with open(lf, "a") as fh:
            fh.write(json.dumps({"ts": t0, "day": day, "id": "i-bad", "type": "r7i.4xlarge",
                                  "state": "running", "accrued_usd": "not-a-number"}) + "\n")
        try:
            totals = period_totals_today(day=day, ledger_file=lf, warn=False)
        except Exception as e:  # noqa: BLE001
            bad.append(f"MALFORMED-FIELD CRASH: a non-numeric accrued_usd raised {e!r} instead of being "
                       "skipped")
        else:
            if "i-bad" in totals:
                bad.append("a malformed row was somehow still aggregated instead of skipped")
            if "i-aaa" not in totals:
                bad.append("BLINDED: a malformed row for one instance erased totals for OTHER instances")

        # (h) a write failure must be REPORTED, not silently swallowed (fail-closed writes).
        blocker_dir = os.path.join(td, "blocker")
        open(blocker_dir, "w").close()   # a FILE, not a directory -- os.makedirs must fail under it
        bad_path = os.path.join(blocker_dir, "ledger.jsonl")
        ok = record([{"id": "i-x", "cost_today_usd": 1.0}], ts=t0, ledger_file=bad_path)
        if ok is not False:
            bad.append("FAIL-CLOSED VIOLATION: record() returned truthy for a write that could not succeed")
        ok2 = record([{"id": "i-x", "cost_today_usd": 1.0}], ts=t0, ledger_file=lf)
        if ok2 is not True:
            bad.append("record() returned falsy for a write that should have succeeded")
    return bad


if __name__ == "__main__":
    sys.exit(_cli())
