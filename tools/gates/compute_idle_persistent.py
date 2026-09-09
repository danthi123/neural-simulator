"""CLASS UC — DEDICATED compute (a pool node / the GPU) reads idle-with-ready-work, PERSISTENTLY. BLOCKING.

WHY THIS IS BLOCKING AND NOT JUST ANOTHER REPORT (owner, 2026-09-08, on `tools/parallel_audit.py`'s existing
UNDER-PARALLELIZED verdict: "past fixes failed being manual/advisory/passive"). `parallel_audit.py` already
computes this exact condition every heartbeat cycle and PRINTS it — and printing was exactly the failure
mode caught this session: one arc ran all session while 8 roadmap lanes + the mini-PC pool sat idle ~2 days,
with the heartbeat alarming correctly the entire time. `gates/lane_starvation` already solved the identical
shape of problem for idle CPU LANES (owner, 2026-07-31: "what's the point of a gate that doesn't block
non-adherence to our workflow?") by moving the check onto an unavoidable path with a waiver escape. This is
the SAME fix for the SAME resource-waste shape — a genuinely idle, zero-marginal-cost DEDICATED resource
sitting unused next to ready work — just fed by a different signal: a pool node or the GPU instead of a CPU
lane. The owner has already accepted this risk profile for lane_starvation; this gate mirrors it rather than
inventing a new one.

WHY PERSISTENCE, NOT A SINGLE READING. `under_compute` is a SNAPSHOT (is a dedicated lane idle RIGHT NOW)
that can legitimately flicker — a pool job just finished, the next hasn't been dispatched yet. Blocking on a
single snapshot would nag on completely normal gaps between jobs and get bypassed with `--no-verify`, which
disables every OTHER gate on the same commit (the exact "check that gets disabled" failure this project's
gate philosophy warns against). So this reads `research/coordination/parallel_audit_state.json` (written
every heartbeat cycle by `parallel_audit.py` via `tools/parallel_state.py`, which tracks WHEN the current
unbroken "true" streak started) and blocks only once `since_under_compute` shows the condition has held
CONTINUOUSLY past `PERSIST_S`.

THE ESCAPE, because a gate with no legitimate exit gets bypassed and then ignored entirely (the same
rationale `gates/lane_starvation` documents for its own `.lane_waiver`): write
`research/queue/.parallel_compute_waiver` with a REASON. It expires after `WAIVER_MAX_H` hours, so a waiver
cannot silently become permanent. A waiver that excuses the idle lane by PRIORITY/FOCUS ("focused on the
crux") rather than naming a genuine per-lane BLOCKER is REJECTED — reusing lane_starvation's own
rationalisation regex verbatim, because pool/GPU idleness is exactly as zero-cost to fill as an idle CPU
lane, so "I'm prioritising something else" can never excuse leaving it idle either (the 2026-08-01 abuse
lane_starvation's own docstring records).

STALENESS GUARD, AND WHY AN ABSENT/OLD STATE FILE PASSES RATHER THAN BLOCKS. The heartbeat is a Claude-session
habit, not a daemon (CLAUDE.md: "Cross-session continuation is MANUAL ... no watchdog/daemon"). A commit made
before the heartbeat has run this session, or long after the last agent process exited, must not be blocked
on data that no longer describes the world — that would eventually brick every commit in a repo that goes
quiet, a worse failure than the advisory-only gap this closes. So a MISSING file, or one older than
`tools.parallel_state.STALE_S`, is treated as "no signal" and passes silently — the same shape
`gates/tool_health_fresh` uses for an absent smoke.

WHAT IT CANNOT CATCH: whether the launched work is WORTH running (the same limit `lane_starvation` names for
itself), and it is blind between heartbeat cycles — a session that never runs the heartbeat never persists a
streak, so this gate stays silent for it (a coverage gap shared with every heartbeat-fed check here, not a
defect unique to this one). It also does not cover the AGENT-FLOOR half of `parallel_audit`'s verdict — see
`gates/agent_floor_persistent` (report-only) and this session's write-up for why escalating that half to
BLOCKING is left as a proposal for owner review rather than decided here.
"""
from __future__ import annotations

import os
import re
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(_ROOT, "tools") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "tools"))
import parallel_state  # noqa: E402  (shared writer/reader schema, see tools/parallel_state.py)

NAME = "compute-idle-persistent"
CLASS_ID = "UC"
BLOCKING = True

WAIVER_FILE = os.path.join(_ROOT, "research", "queue", ".parallel_compute_waiver")
PERSIST_S = 45 * 60     # under_compute must hold CONTINUOUSLY this long before it blocks
WAIVER_MAX_H = 6

# Reused verbatim from gates/lane_starvation.py's own hard-earned lesson (2026-08-01): a waiver that excuses
# idle DEDICATED compute by priority/focus rather than a genuine blocker is the exact abuse to reject, and
# pool/GPU idleness is exactly as zero-cost-to-fill as an idle CPU lane.
_RATIONALISATION = re.compile(r"crux|priorit|focus|deprioriti|momentum|behind the|saturated with", re.I)


def _waiver_reason():
    if not os.path.exists(WAIVER_FILE):
        return None
    age_h = (time.time() - os.path.getmtime(WAIVER_FILE)) / 3600.0
    if age_h > WAIVER_MAX_H:
        return None
    try:
        return open(WAIVER_FILE, errors="ignore").read().strip()[:120] or "(no reason given)"
    except OSError:
        return None


def _decide(state, now_ts, waiver, persist_s=PERSIST_S):
    """Pure decision (testable without files/clock/waiver disk state)."""
    if not parallel_state.is_fresh(state, now_ts):
        return []                       # no signal: absent/stale state never blocks (see module docstring)
    since = state.get("since_under_compute")
    if not (state.get("under_compute") and since is not None and (now_ts - since) >= persist_s):
        return []
    mins = int((now_ts - since) / 60)
    if waiver:
        if _RATIONALISATION.search(waiver):
            return ["DEDICATED compute (a pool node / the GPU) has read idle-with-ready-work for %d min "
                    "straight, and research/queue/.parallel_compute_waiver excuses it by PRIORITY/FOCUS "
                    "(\"%s\"), which is REJECTED (the same 2026-08-01 lane_starvation abuse, same shape: "
                    "these lanes cost NOTHING beside whatever else is running). Queue something instead: "
                    "`tools/sweep_pool.sh` / `tools/gpu_queue.sh add '<cmd>'`, or waive a REAL blocker."
                    % (mins, waiver)]
        return []
    return ["DEDICATED compute (a pool node / the GPU) has read idle-with-ready-work for %d min straight "
            "(budget %d) — this is the same zero-marginal-cost waste `gates/lane_starvation` already blocks "
            "on for CPU lanes, just for GPU/pool idleness. FIX: queue one job — `tools/sweep_pool.sh` (CPU) "
            "or `tools/gpu_queue.sh add '<cmd>'` (GPU) — or waive a genuine blocker (auto-expires in %dh):\n"
            "          echo 'why' > research/queue/.parallel_compute_waiver"
            % (mins, persist_s // 60, WAIVER_MAX_H)]


def check(paths=None):
    del paths
    return _decide(parallel_state.load(), time.time(), _waiver_reason())


def selftest():
    """FAILING DIRECTION FIRST: a fresh state with under_compute persisted past budget, no waiver, MUST block."""
    bad = []
    now = 2_000_000.0
    fresh_over = {"generated_at": now, "under_compute": True, "since_under_compute": now - PERSIST_S - 60}
    if not _decide(fresh_over, now, None):
        bad.append("did NOT block a persisted-past-budget idle-compute reading with no waiver")
    # NEGATIVE — not yet past budget.
    fresh_under = {"generated_at": now, "under_compute": True, "since_under_compute": now - 60}
    if _decide(fresh_under, now, None):
        bad.append("FALSE POSITIVE: blocked before the persistence budget was reached")
    # NEGATIVE — currently healthy.
    healthy = {"generated_at": now, "under_compute": False, "since_under_compute": None}
    if _decide(healthy, now, None):
        bad.append("FALSE POSITIVE: blocked while under_compute currently reads healthy")
    # NEGATIVE — stale state (heartbeat not running recently) must pass SILENTLY, never block.
    stale = {"generated_at": now - parallel_state.STALE_S - 1, "under_compute": True,
             "since_under_compute": now - PERSIST_S - 3600}
    if _decide(stale, now, None):
        bad.append("FALSE POSITIVE: blocked on a STALE state (heartbeat may not be running)")
    # NEGATIVE — absent state.
    if _decide(None, now, None):
        bad.append("FALSE POSITIVE: blocked with no state file at all")
    # waiver: a genuine blocker excuses it.
    if _decide(fresh_over, now, "no ready de-risk for the pool: the cache build is blocked on X"):
        bad.append("FALSE POSITIVE: a genuine per-blocker waiver was still blocked")
    # waiver: a priority/focus rationalisation must still block (the 2026-08-01 abuse).
    if not _decide(fresh_over, now, "focused on the crux, deprioritized behind it"):
        bad.append("did NOT reject a priority/focus rationalisation waiver (the 2026-08-01 abuse would pass)")
    return bad
