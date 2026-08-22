"""CLASS TH — a free lane/tool ROTTED (or its readiness went unchecked) and nothing surfaced it.

WHY (owner, 2026-08-21: "tools rot … and only get revived on manual prompting — readiness must be a checked
property"). `tools/tool_health.py` smoke-tests every free lane (experiment engine, pool, gpu_queue, --auto-tune,
cloud) against CURRENT state and writes research/coordination/tool_health.json. This gate turns that artifact
into an ENFORCED check: every commit it re-reads the file and SURFACES (a) any tool the last smoke found ROTTED
— which `tool_health.py --emit` has already turned into a repair backlog item — and (b) whether the smoke is
STALE (periodicity), nudging a re-run.

REPORT-ONLY (BLOCKING=False) BY DESIGN. A hard block would wedge every commit on a network-down pool or an
un-run smoke — the opposite of frugality. Rot is caught by the smoke (run on a free lane by the heartbeat /
scheduled task, per the plan) and becomes a backlog item automatically; this gate makes the STATE visible in
the same place every other gate reports, so it cannot be forgotten. It is still registry-discovered and
selftest-backed, so it "fires" (is reported) on every commit — readiness is now a checked property, not a memory.

WHAT IT CANNOT CATCH: rot in a tool the smoke does not cover, or a smoke that was never run (the file is absent
-> silent, to avoid nagging repos that have not adopted it). Coverage of the smoke itself is `tool_health.py`'s
job, not this gate's.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

NAME = "tool-health-fresh"
CLASS_ID = "TH"
BLOCKING = False

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FILE = os.path.join(_ROOT, "research", "coordination", "tool_health.json")
STALE_DAYS = 7


def _decide(exists, age_days, rotted_names, stale_days=STALE_DAYS):
    """Pure decision (testable without files/clock). Report each ROTTED tool + staleness; silent if absent."""
    if not exists:
        return []
    problems = []
    for name in rotted_names:
        problems.append("⛔ TOOL ROTTED: %s — see research/coordination/tool_health.json (a repair backlog "
                        "item was emitted). Fix, or re-run `python tools/tool_health.py --only %s`." % (name, name))
    if age_days is not None and age_days > stale_days:
        problems.append("tool-health smoke is %d days stale (> %d): re-run `python tools/tool_health.py --emit` "
                        "so free-lane readiness stays a checked property." % (int(age_days), stale_days))
    return problems


def _load():
    """Return (exists, age_days, rotted_names)."""
    if not os.path.exists(_FILE):
        return False, None, []
    try:
        d = json.load(open(_FILE, errors="ignore"))
    except (ValueError, OSError):
        # a present-but-unparseable artifact IS a problem to surface
        return True, None, ["<tool_health.json unparseable>"]
    rotted = [r.get("tool", "?") for r in d.get("results", []) if r.get("status") == "ROTTED"]
    age = None
    ts = d.get("generated_at")
    if ts:
        try:
            gen = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - gen).total_seconds() / 86400.0
        except (ValueError, TypeError):
            age = None
    return True, age, rotted


def check(paths):
    exists, age_days, rotted = _load()
    return _decide(exists, age_days, rotted)


def selftest():
    """FAILING DIRECTION FIRST: a ROTTED tool and a stale smoke MUST be reported."""
    bad = []
    if not _decide(True, 0.0, ["experiment-engine"]):
        bad.append("did NOT report a ROTTED tool")
    if not _decide(True, 30.0, []):
        bad.append("did NOT report a stale (30d) smoke")
    if not _decide(True, 30.0, ["pool"]):
        bad.append("did NOT report BOTH a rotted tool and staleness")
    # NEGATIVE — fresh + clean -> silent.
    if _decide(True, 0.5, []):
        bad.append("FALSE POSITIVE: reported on a fresh, all-ready smoke")
    # NEGATIVE — file absent -> silent (don't nag un-adopted repos).
    if _decide(False, None, ["x"]):
        bad.append("FALSE POSITIVE: reported when tool_health.json is absent")
    # boundary — exactly at the budget is not yet stale.
    if _decide(True, float(STALE_DAYS), []):
        bad.append("FALSE POSITIVE: flagged staleness AT the budget (should be strictly over)")
    return bad
