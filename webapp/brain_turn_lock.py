"""ONE process-level TURN LOCK for the brain (one-brain shared-memory plan, step M0).

docs/plans/2026-10-13-one-brain-shared-memory-DESIGN.md section 2: `brain_chat` is a sync FastAPI route, so uvicorn
runs it in a threadpool, and before this module there was no turn lock (only `_WARM_QWEN_LOCK` around the Qwen
model). Two conversations could therefore run turns at the same time against the PROCESS-SHARED organs (affect,
comprehension, surprise, metacog, world-model, the xedge pool, the bridge backend cache), and the between-turn idle
tick (`webapp/continuous_engine.tick_idle_sessions`, run in a thread executor) could run against those organs in
the middle of a turn. This lock serializes them: one turn, or one cheap idle-tick step, at a time.

Biology, not only safety: one global workspace ignites one content at a time, so a serial turn is the GNW's own
constraint (design section 2, "Concurrency"). It is pure engineering here: no cognition claim.

    with brain_turn("brain_chat", timeout=turn_wait_timeout_s()):
        ...  # the brain-processing part of a turn

FIFO. Waiters are served in arrival order (a ticket queue on a Condition). `threading.RLock` is not fair: a thread
that releases and re-acquires at once (the idle tick moving to its next step) can starve a waiting turn. With the
queue, a re-acquire while anyone waits goes to the back, so the tick yields to a waiting turn between any two of its
steps. Re-entrant for the thread that holds it (a nested `brain_turn` on that thread never waits).

BOUNDED WAIT. `brain_chat` waits at most `BRAIN_TURN_WAIT_TIMEOUT_S` seconds (default 300; a value <= 0 waits
without limit), then `TurnBusy` is raised and the handler answers with `busy_payload` (the normal response shape,
`busy: true`) instead of stacking waits that hold anyio threadpool tokens other routes need. The idle tick waits
without limit (it runs in its own executor thread, not on a request).

Escape: `BRAIN_TURN_LOCK=0` (also false/off/no) -> `brain_turn` takes no lock at all, the pre-M0 behaviour.

Wait time is recorded here (`stats()`, and the per-thread `last_wait_s()`), not in the reply: the `/api/brain-chat`
response has no timing or debug field, and adding one would change every response body.

WHAT WAS MEASURED (2026-10-13; do not read more into it). Single-caller identity: the real handler called directly
(no uvicorn, no idle tick running), stub renderer, numpy, production LTM default, one session, five turns, dev seed
7, three fresh processes (lock ON, OFF, ON again as the null): `research/runners/turn_lock_identity_check.py`,
`research/findings/raw/_one_brain_probe/turn_lock_identity_s7*.json`. Ordering and fairness: brain-free tests with
stubbed bodies (`tests/test_brain_turn_lock.py`); they show the lock's behaviour, not reply identity.
"""
from __future__ import annotations

import collections
import os
import threading
import time
from contextlib import contextmanager

TURN_LOCK_ENV = "BRAIN_TURN_LOCK"
TIMEOUT_ENV = "BRAIN_TURN_WAIT_TIMEOUT_S"
_TURN_LOCK_DEFAULT_ON = True
DEFAULT_TIMEOUT_S = 300.0

# A wait shorter than this is an uncontended acquire (the cost of the call itself), not a queue.
WAITED_THRESHOLD_S = 0.001


class TurnBusy(Exception):
    """The turn lock was not granted within the wait bound."""

    def __init__(self, where, waited_s, timeout_s):
        super().__init__("turn lock busy after %.1fs (bound %.1fs) at %s" % (waited_s, timeout_s, where))
        self.where, self.waited_s, self.timeout_s = where, waited_s, timeout_s


class FifoRLock:
    """A re-entrant lock that grants waiters in arrival order. `acquire(timeout)` returns False on timeout and
    removes the caller from the queue (it never strands the line)."""

    def __init__(self):
        self._cv = threading.Condition(threading.Lock())
        self._queue = collections.deque()
        self._owner = None
        self._depth = 0

    def acquire(self, timeout=None) -> bool:
        me = threading.get_ident()
        with self._cv:
            if self._owner == me:
                self._depth += 1
                return True
            if self._owner is None and not self._queue:
                self._owner, self._depth = me, 1
                return True
            ticket = object()
            self._queue.append(ticket)
            deadline = None if timeout is None else time.monotonic() + timeout
            while not (self._owner is None and self._queue[0] is ticket):
                if deadline is None:
                    self._cv.wait()
                    continue
                rem = deadline - time.monotonic()
                if rem <= 0:
                    self._queue.remove(ticket)
                    self._cv.notify_all()   # the next in line may now be at the head
                    return False
                self._cv.wait(rem)
            self._queue.popleft()
            self._owner, self._depth = me, 1
            return True

    def release(self) -> None:
        with self._cv:
            if self._owner != threading.get_ident():
                raise RuntimeError("release of a turn lock this thread does not hold")
            self._depth -= 1
            if self._depth == 0:
                self._owner = None
                self._cv.notify_all()

    def held_by_current_thread(self) -> bool:
        return self._owner == threading.get_ident()

    def waiters(self) -> int:
        with self._cv:
            return len(self._queue)


_TURN_LOCK = FifoRLock()
_STATS_LOCK = threading.Lock()   # guards _STATS only (never held while waiting for the turn lock)
_LOCAL = threading.local()


def _fresh_stats() -> dict:
    return {"n_acquired": 0, "n_waited": 0, "n_timeouts": 0, "total_wait_s": 0.0, "max_wait_s": 0.0,
            "last_wait_s": 0.0, "last_where": None, "by_where": {}}


_STATS = _fresh_stats()


def turn_lock_enabled(env=None) -> bool:
    """True unless `BRAIN_TURN_LOCK` is one of {0, false, off, no} (case-insensitive). Unset or empty -> ON.
    AT ONE_BRAIN ON (PACKET-m2-fixes item 3) the lock is FORCED: the escape is ignored there — one shared brain
    carries every conversation and the lock is what serialises a single reply, so ON + `BRAIN_TURN_LOCK=0` would
    ship the shared turn lock off in production. Off-ON the escape is byte-identical to HEAD (default untouched)."""
    env = os.environ if env is None else env
    v = env.get(TURN_LOCK_ENV)
    if v is None or not v.strip():
        return _TURN_LOCK_DEFAULT_ON
    if v.strip().lower() in ("0", "false", "off", "no"):
        from webapp import one_brain as _OB   # lazy: webapp/one_brain imports this module inside Brain.__init__
        if _OB.one_brain_on(env):            # forced at ON — the escape does not apply
            return True
        return False
    return True


def turn_wait_timeout_s(env=None):
    """The request-side wait bound in seconds: `BRAIN_TURN_WAIT_TIMEOUT_S` (default 300). A value <= 0 means no bound
    (None); a malformed value falls back to the default."""
    env = os.environ if env is None else env
    v = env.get(TIMEOUT_ENV)
    if v is None or not v.strip():
        return DEFAULT_TIMEOUT_S
    try:
        t = float(v.strip())
    except ValueError:
        return DEFAULT_TIMEOUT_S
    return None if t <= 0 else t


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
def brain_turn(where: str = "turn", timeout=None):
    """Hold the process turn lock for the block (unless `BRAIN_TURN_LOCK=0`). Yields the seconds spent waiting.
    `timeout` (seconds, None = no bound) -> `TurnBusy` when the lock is not granted in time. A nested acquire on the
    holding thread is free and is not counted."""
    if not turn_lock_enabled():
        _LOCAL.last_wait_s = 0.0
        yield 0.0
        return
    nested = _TURN_LOCK.held_by_current_thread()
    t0 = time.perf_counter()
    if not _TURN_LOCK.acquire(timeout=timeout):
        waited = time.perf_counter() - t0
        with _STATS_LOCK:
            _STATS["n_timeouts"] += 1
        raise TurnBusy(where, waited, timeout)
    wait_s = time.perf_counter() - t0
    try:
        if not nested:
            _LOCAL.last_wait_s = wait_s
            _record(where, wait_s)
        yield wait_s
    finally:
        _TURN_LOCK.release()


def busy_payload(req, renderer, busy: TurnBusy) -> dict:
    """The `/api/brain-chat` body for a turn that was not admitted: the normal response keys, nothing recalled,
    `busy: true` and the wait it gave up after. A transport status, not a brain decision: `abstained` is False
    because the moat did not fire (the brain never saw the message)."""
    return {"answer": "I'm still busy with another conversation; please send that again in a moment.",
            "abstained": False, "busy": True, "recalled_svo": None, "derived": False, "verified": False,
            "renderer": renderer, "brain": getattr(req, "brain", None), "source": None, "rich": False,
            "activity": None,
            "turn_lock": {"waited_s": round(float(busy.waited_s), 3), "timeout_s": busy.timeout_s}}


def held_by_current_thread() -> bool:
    """True when this thread holds the turn lock (for tests and assertions; never used to decide a reply)."""
    return _TURN_LOCK.held_by_current_thread()


def waiters() -> int:
    """How many threads are queued for the turn lock right now."""
    return _TURN_LOCK.waiters()


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
