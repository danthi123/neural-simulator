"""CLASS UA — fewer than the agent floor running, PERSISTENTLY. Report-only (why NOT blocking, below).

RELATIONSHIP TO `gates/compute_idle_persistent` (CLASS UC). `tools/parallel_audit.py` computes TWO
independent under-parallelization triggers each heartbeat cycle: `under_compute` (a dedicated pool node / the
GPU idle with ready work — UC blocks on that, persisted) and `under_agents` (fewer than `AGENT_FLOOR`
concurrent build/research agents running). Both used to be equally advisory (printed only); this closes the
LOWER-RISK half by blocking (UC) and closes THIS half by making it commit-visible without blocking, because
the two signals have genuinely different risk profiles — see below.

WHY NOT BLOCKING (this is the judgement call, not a corner cut). `under_compute` fires on a genuinely idle,
zero-marginal-cost DEDICATED resource — the same shape `gates/lane_starvation` already blocks on for CPU
lanes, so extending that precedent to GPU/pool idleness is low-risk. `under_agents` is different: it also
fires on completely ROUTINE serial work — a one-line fix, a small bugfix commit, a single de-risk that
genuinely needs no fan-out — where fewer than the floor is not a failure at all. Forcing every commit below
the floor to BLOCK risks becoming exactly the naggy, `--no-verify`-trained gate this project's own gate
philosophy warns against (a check that gets bypassed once disables every OTHER gate on the same commit,
docs/FAILURE_GATE_MATRIX.md's own "loop that keeps this file honest"). What floor, what persistence duration,
and what waiver would make blocking safe here is a genuine judgement call, written up as a proposal for
owner review rather than decided unilaterally in this gate. Until that is resolved, this REPORTS the
persisted condition (visible on every commit, not just a heartbeat line that can be read past) without
risking a false-positive block on routine work.

STATE SOURCE + STALENESS GUARD: identical to `gates/compute_idle_persistent` — reads
`research/coordination/parallel_audit_state.json` (`tools/parallel_state.py`), and treats an absent or
stale (`> tools.parallel_state.STALE_S`) reading as NO SIGNAL rather than "still under-parallelized" (a
quiet repo must never brick — or here, nag on every future commit forever).

WHAT IT CANNOT CATCH: whether the routing choice (agents vs. compute) was the right one for the work at
hand — that remains judgement, same as every other gate here.
"""
from __future__ import annotations

import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(_ROOT, "tools") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "tools"))
import parallel_state  # noqa: E402  (shared writer/reader schema, see tools/parallel_state.py)

NAME = "agent-floor-persistent"
CLASS_ID = "UA"
BLOCKING = False

REPORT_S = 20 * 60   # under_agents must hold CONTINUOUSLY this long before it is even reported


def _decide(state, now_ts, report_s=REPORT_S):
    """Pure decision (testable without files/clock)."""
    if not parallel_state.is_fresh(state, now_ts):
        return []                       # no signal: absent/stale state never reports (see module docstring)
    since = state.get("since_under_agents")
    if not (state.get("under_agents") and since is not None and (now_ts - since) >= report_s):
        return []
    mins = int((now_ts - since) / 60)
    return ["fewer than the agent floor has been running for %d min straight (agents=%s currently). Consider "
            "fanning out more build/research agents, or hold deliberately — this reports rather than blocks; "
            "see this module's docstring for why escalating it to BLOCKING is an open proposal for owner "
            "review, not decided here." % (mins, state.get("agents"))]


def check(paths=None):
    del paths
    return _decide(parallel_state.load(), time.time())


def selftest():
    """FAILING DIRECTION FIRST: a fresh state with under_agents persisted past REPORT_S must be reported."""
    bad = []
    now = 3_000_000.0
    fresh_over = {"generated_at": now, "under_agents": True, "since_under_agents": now - REPORT_S - 60, "agents": 1}
    if not _decide(fresh_over, now):
        bad.append("did NOT report a persisted-past-budget agent-floor reading")
    # NEGATIVE — not yet past budget.
    fresh_under = {"generated_at": now, "under_agents": True, "since_under_agents": now - 60, "agents": 1}
    if _decide(fresh_under, now):
        bad.append("FALSE POSITIVE: reported before the persistence budget was reached")
    # NEGATIVE — currently healthy.
    healthy = {"generated_at": now, "under_agents": False, "since_under_agents": None, "agents": 5}
    if _decide(healthy, now):
        bad.append("FALSE POSITIVE: reported while under_agents currently reads healthy")
    # NEGATIVE — stale state must be silent.
    stale = {"generated_at": now - parallel_state.STALE_S - 1, "under_agents": True,
              "since_under_agents": now - REPORT_S - 3600, "agents": 1}
    if _decide(stale, now):
        bad.append("FALSE POSITIVE: reported on a STALE state (heartbeat may not be running)")
    # NEGATIVE — absent state.
    if _decide(None, now):
        bad.append("FALSE POSITIVE: reported with no state file at all")
    return bad
