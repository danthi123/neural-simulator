#!/usr/bin/env python3
"""parallel_state.py — shared schema + pure logic for the under-parallelization PERSISTENCE record.

`tools/parallel_audit.py` (the heartbeat check) is the sole WRITER of
`research/coordination/parallel_audit_state.json`, once per cycle. `tools/gates/compute_idle_persistent.py`
and `tools/gates/agent_floor_persistent.py` are its READERS, at commit time — turning "printed every
heartbeat, read past" into an actually-enforced persistence signal (owner, 2026-09-08, on parallel_audit's
existing UNDER-PARALLELIZED verdict: "past fixes failed being manual/advisory/passive").

Kept in ONE file so the writer and both readers cannot drift on field names or the streak-tracking rule —
the same reason `tools/cost_audit.py` exposes `find_untiered_agents()` for
`tools/gates/workflow_cost_tiering.py` to import rather than re-implementing the detector.

WHY A "STREAK START" AND NOT JUST THE LATEST READING. `parallel_audit.py`'s under_agents/under_compute are
SNAPSHOTS (how many agents are running / is a lane idle RIGHT NOW) — they can flicker true/false between
heartbeat cycles for completely legitimate reasons (an agent just finished, a pool job just got dispatched).
A single flicker is not the failure this exists to catch; a condition that has read true for the LAST 45
minutes straight is. `since_under_agents` / `since_under_compute` record when the CURRENT unbroken streak of
"true" started, reset to `None` the instant the reading goes healthy, so a reader can compute "how long has
this been continuously true" without re-deriving it from raw process/queue state itself.
"""
from __future__ import annotations

import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_FILE = os.path.join(_ROOT, "research", "coordination", "parallel_audit_state.json")
# A reading older than this describes a world that may no longer hold (the heartbeat is a Claude-session
# habit, not a daemon — CLAUDE.md: "Cross-session continuation is MANUAL ... no watchdog/daemon"). Readers
# MUST treat a stale/absent state as NO SIGNAL, never as "still under-parallelized" — the failure mode this
# guards against is a repo that goes quiet (heartbeat not running) permanently blocking every future commit,
# which would be worse than the advisory-only gap being fixed here.
STALE_S = 40 * 60


def next_state(prev, now_ts, under_agents, under_compute, agents, idle_pool, gpu_free, n_open):
    """Pure state transition (testable without the filesystem/clock). `prev` is the previously-persisted
    dict, or None/{} on the very first run or after the file was removed. Returns the new dict to persist.
    """
    prev = prev or {}

    def _since(now_true, flag_key, since_key):
        if not now_true:
            return None                                            # healthy now -> no streak
        if prev.get(flag_key) and prev.get(since_key) is not None:
            return prev[since_key]                                  # already true last cycle -> keep start
        return now_ts                                               # just became true -> streak starts now

    return {
        # Lets tools/gates/device_and_cost.py recognize this as coordination/operational STATE, not a
        # scientific result — the same exemption pattern as "sim-autonomous-workboard-v1" / "board-sync-v1" /
        # "tool-health-v1" (see that gate's _is_structural_record()). Without this, a freshly-written record
        # with no backend/device field reads as an unaudited experiment artifact and wrongly blocks a commit.
        "schema": "parallel-audit-state-v1",
        "generated_at": now_ts,
        "under_agents": bool(under_agents),
        "under_compute": bool(under_compute),
        "since_under_agents": _since(under_agents, "under_agents", "since_under_agents"),
        "since_under_compute": _since(under_compute, "under_compute", "since_under_compute"),
        "agents": agents,
        "idle_pool": idle_pool,
        "gpu_free": bool(gpu_free),
        "n_open": n_open,
    }


def persist(now_ts, **kwargs):
    """Load STATE_FILE (best-effort), compute the next state, write it back atomically (best-effort — this
    must never raise: the caller is parallel_audit.py, an exit-0-always heartbeat script). Returns the dict
    actually computed (even if the write itself failed), so callers can still print/inspect it."""
    prev = None
    try:
        prev = json.load(open(STATE_FILE, errors="ignore"))
    except (OSError, ValueError):
        prev = None
    new = next_state(prev, now_ts, **kwargs)
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(new, f, indent=1)
        os.replace(tmp, STATE_FILE)
    except OSError:
        pass
    return new


def load():
    """Return the persisted state dict, or None if absent/unparseable (treated as no-signal by readers)."""
    try:
        d = json.load(open(STATE_FILE, errors="ignore"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def is_fresh(state, now_ts, stale_s=STALE_S):
    """A state is fresh only if present AND generated within stale_s of now — an absent or old file is
    NO SIGNAL, never treated as "still under-parallelized" (see the STALE_S docstring above)."""
    if not state:
        return False
    gen = state.get("generated_at")
    return gen is not None and 0 <= (now_ts - gen) <= stale_s
