"""ONE process-level TURN LOCK for the brain (one-brain shared-memory plan, step M0).

docs/plans/2026-10-13-one-brain-shared-memory-DESIGN.md section 2: `brain_chat` is a sync FastAPI route, so uvicorn
runs it in a threadpool, and before this module there was no turn lock (only `_WARM_QWEN_LOCK` around the Qwen
model). Two conversations could therefore run turns at the same time against the PROCESS-SHARED organs (affect,
comprehension, surprise, metacog, world-model, the xedge pool, the bridge backend cache), and the between-turn idle
tick (`webapp/continuous_engine.tick_idle_sessions`, run in a thread executor) could run against those organs in
the middle of a turn. This lock serializes them: one turn, or one session's idle-tick step, at a time.

Biology, not only safety: one global workspace ignites one content at a time, so a serial turn is the GNW's own
constraint (design section 2, "Concurrency"). It is pure engineering here: no cognition claim, and a single caller
is byte-identical (the lock adds no work inside a turn; it only orders turns).

    with brain_turn("brain_chat"):
        ...  # the brain-processing part of a turn

The lock is RE-ENTRANT (`threading.RLock`): a turn that calls back into a locked path on the same thread (the OpenAI
shim calls `brain_chat`; a tick inside a world step) never deadlocks.

Escape: `BRAIN_TURN_LOCK=0` (also false/off/no) -> `brain_turn` takes no lock at all, the pre-M0 behaviour. Read on
every call, so a test or an operator can flip it without a restart. Default ON.

Wait time is recorded here (`stats()`, and the per-thread `last_wait_s()`), not in the reply: the `/api/brain-chat`
response has no timing or debug field today, and adding one would change every response body.
"""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager

TURN_LOCK_ENV = "BRAIN_TURN_LOCK"
_TURN_LOCK_DEFAULT_ON = True

_TURN_LOCK = threading.RLock()
_STATS_LOCK = threading.Lock()   # guards _STATS only (never held while waiting for the turn lock)
_LOCAL = threading.local()


def _fresh_stats() -> dict:
    return {"n_acquired": 0, "n_waited": 0, "total_wait_s": 0.0, "max_wait_s": 0.0, "last_wait_s": 0.0,
            "last_where": None, "by_where": {}}


_STATS = _fresh_stats()

# A wait shorter than this is an uncontended acquire (the cost of the call itself), not a queue.
WAITED_THRESHOLD_S = 0.001


def turn_lock_enabled(env=None) -> bool:
    """True unless `BRAIN_TURN_LOCK` is one of {0, false, off, no} (case-insensitive). Unset or empty -> the default
    (ON)."""
    env = os.environ if env is None else env
    v = env.get(TURN_LOCK_ENV)
    if v is None or not v.strip():
        return _TURN_LOCK_DEFAULT_ON
    return v.strip().lower() not in ("0", "false", "off", "no")


def _record(where: str, wait_s: float) -> None:
    with _STATS_LOCK:
        s = _STATS
        s["n_acquired"] += 1
        if wait_s >= WAITED_THRESHOLD_S:
            s["n_waited"] += 1
        s["total_wait_s"] += wait_s
        s["max_wait_s"] = max(s["max_wait_s"], wait_s)
        s["last_wait_s"] = wait_s
        s["last_where"] = where
        w = s["by_where"].setdefault(where, {"n": 0, "total_wait_s": 0.0, "max_wait_s": 0.0})
        w["n"] += 1
        w["total_wait_s"] += wait_s
        w["max_wait_s"] = max(w["max_wait_s"], wait_s)


@contextmanager
def brain_turn(where: str = "turn"):
    """Hold the process turn lock for the block (unless `BRAIN_TURN_LOCK=0`). Yields the seconds spent waiting for
    it (0.0 when the escape is set). A nested acquire on the same thread is free and is not counted."""
    if not turn_lock_enabled():
        _LOCAL.last_wait_s = 0.0
        yield 0.0
        return
    depth = getattr(_LOCAL, "depth", 0)
    t0 = time.perf_counter()
    _TURN_LOCK.acquire()
    wait_s = time.perf_counter() - t0
    _LOCAL.depth = depth + 1
    try:
        if depth == 0:   # count the outermost acquire only: a re-entrant one never waits
            _LOCAL.last_wait_s = wait_s
            _record(where, wait_s)
        yield wait_s
    finally:
        _LOCAL.depth = depth
        _TURN_LOCK.release()


def held_by_current_thread() -> bool:
    """True when this thread holds the turn lock (for tests and assertions; never used to decide a reply)."""
    return getattr(_LOCAL, "depth", 0) > 0


def last_wait_s() -> float:
    """Seconds the current thread last waited for the turn lock (0.0 if it never took it)."""
    return float(getattr(_LOCAL, "last_wait_s", 0.0))


def stats() -> dict:
    """A copy of the process-wide wait statistics."""
    with _STATS_LOCK:
        out = dict(_STATS)
        out["by_where"] = {k: dict(v) for k, v in _STATS["by_where"].items()}
        return out


def reset_stats() -> None:
    """Zero the statistics (tests and probes only)."""
    global _STATS
    with _STATS_LOCK:
        _STATS = _fresh_stats()
