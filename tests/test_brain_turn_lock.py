"""M0 of the one-brain shared-memory plan: ONE process-level FIFO turn lock (webapp/brain_turn_lock.py).

docs/plans/2026-10-13-one-brain-shared-memory-DESIGN.md section 2. `brain_chat` is a sync FastAPI route (threadpool),
so before M0 two conversations could run turns at once against the process-shared organs, and the idle tick
(continuous_engine.tick_idle_sessions, run in a thread executor) could run in the middle of a turn. CPU-only, no
brain build: the real `brain_chat` handler runs with its brain build and `brain_reply` stubbed, and the real
`tick_idle_sessions` runs with stub organs. These tests pin the lock's BEHAVIOUR (ordering, fairness, preemption,
bounded wait); reply byte-identity is measured on the real brain by research/runners/turn_lock_identity_check.py.

Gate test for the research/FAILURE_LOG.md line "brain_chat has no turn lock: concurrent sessions race on shared
organs": `test_two_sessions_never_overlap_in_the_real_handler` (and its escape-flag control, which shows the
instrument does see an overlap when the lock is off).
"""
import json
import threading
import time

import pytest

from webapp import brain_turn_lock as L


# ── the lock module ─────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("val,expected", [(None, True), ("", True), ("1", True), ("on", True), ("yes", True),
                                          ("0", False), ("false", False), (" OFF ", False), ("No", False)])
def test_escape_flag_parse(val, expected):
    env = {} if val is None else {L.TURN_LOCK_ENV: val}
    assert L.turn_lock_enabled(env) is expected


@pytest.mark.parametrize("val,expected", [(None, 300.0), ("", 300.0), ("2.5", 2.5), ("0", None), ("-1", None),
                                          ("junk", 300.0)])
def test_wait_timeout_parse(val, expected):
    env = {} if val is None else {L.TIMEOUT_ENV: val}
    assert L.turn_wait_timeout_s(env) == expected


def test_reentrant_on_one_thread(monkeypatch):
    monkeypatch.delenv(L.TURN_LOCK_ENV, raising=False)
    L.reset_stats()
    with L.brain_turn("outer"):
        assert L.held_by_current_thread()
        with L.brain_turn("inner") as w:      # would deadlock with a plain Lock
            assert w < 0.5
            assert L.held_by_current_thread()
        assert L.held_by_current_thread()
    assert not L.held_by_current_thread()
    s = L.stats()
    assert s["n_acquired"] == 1 and list(s["by_where"]) == ["outer"]   # only the outermost acquire is counted


def test_wait_time_is_recorded(monkeypatch):
    monkeypatch.delenv(L.TURN_LOCK_ENV, raising=False)
    L.reset_stats()
    inside, waits = threading.Event(), []

    def holder():
        with L.brain_turn("holder"):
            inside.set()
            time.sleep(0.2)

    t = threading.Thread(target=holder)
    t.start()
    assert inside.wait(2.0)
    with L.brain_turn("waiter") as w:
        waits.append(w)
        assert L.last_wait_s() == w
    t.join()
    assert waits[0] >= 0.1
    s = L.stats()
    assert s["n_acquired"] == 2 and s["n_waited"] >= 1 and s["by_where"]["waiter"]["max_wait_s"] >= 0.1


def test_escape_takes_no_lock(monkeypatch):
    monkeypatch.setenv(L.TURN_LOCK_ENV, "0")
    L.reset_stats()
    with L.brain_turn("x") as w:
        assert w == 0.0 and not L.held_by_current_thread()
        # another thread can take the raw lock while we are "inside": nothing is held
        got = []
        t = threading.Thread(target=lambda: got.append(L._TURN_LOCK.acquire(timeout=1.0)) or L._TURN_LOCK.release())
        t.start()
        t.join()
        assert got == [True]
    assert L.stats()["n_acquired"] == 0


def _queue_in_order(lock, names, order, timeouts=None):
    """Start one thread per name, each queued on `lock` before the next starts (while the caller holds it)."""
    threads = []
    for i, nm in enumerate(names):
        to = (timeouts or {}).get(nm)

        def run(nm=nm, to=to):
            if lock.acquire(timeout=to):
                order.append(nm)
                time.sleep(0.01)
                lock.release()
            else:
                order.append(nm + ":timeout")

        th = threading.Thread(target=run)
        th.start()
        deadline = time.time() + 2.0
        while lock.waiters() < i + 1 - sum(1 for x in order if x.endswith(":timeout")) and time.time() < deadline:
            time.sleep(0.005)
        threads.append(th)
    return threads


def test_fifo_grants_in_arrival_order():
    lock = L.FifoRLock()
    order = []
    assert lock.acquire()
    ths = _queue_in_order(lock, ["a", "b", "c", "d", "e"], order)
    assert lock.waiters() == 5
    lock.release()
    for th in ths:
        th.join(5.0)
    assert order == ["a", "b", "c", "d", "e"]


def test_fifo_releaser_cannot_jump_the_queue():
    """The RLock unfairness the review measured: a holder that releases and re-acquires at once must go behind the
    thread already waiting."""
    lock = L.FifoRLock()
    order = []
    assert lock.acquire()
    ths = _queue_in_order(lock, ["turn"], order)
    lock.release()
    assert lock.acquire(timeout=2.0)          # the releaser re-acquires: it is served AFTER the waiting turn
    order.append("releaser")
    lock.release()
    ths[0].join(5.0)
    assert order == ["turn", "releaser"]


def test_fifo_timeout_leaves_the_queue_clean():
    lock = L.FifoRLock()
    order = []
    assert lock.acquire()
    ths = _queue_in_order(lock, ["b"], order, timeouts={"b": 0.1})
    ths[0].join(2.0)                          # b gives up while the line is held
    assert order == ["b:timeout"] and lock.waiters() == 0
    ths += _queue_in_order(lock, ["c"], order)
    lock.release()
    ths[1].join(2.0)
    assert order == ["b:timeout", "c"]        # the abandoned ticket never strands the next waiter


# ── the real brain_chat handler, brain build + reply stubbed ────────────────────────────────────────────────────────

class _Chat:
    """Stand-in for a ChatBrain."""


@pytest.fixture
def stubbed_handler(monkeypatch):
    from fastapi.responses import JSONResponse
    from webapp import server as S
    from webapp import continuous_engine as CE

    monkeypatch.setenv("BRAIN_CHAT_RENDERER", "stub")
    monkeypatch.delenv(L.TIMEOUT_ENV, raising=False)
    monkeypatch.setattr(S, "_BRAIN_CHATS", {})
    monkeypatch.setattr(CE, "_LAST_REQUEST", {})
    monkeypatch.setattr(CE, "_LAST_TURN_END", {})
    monkeypatch.setattr(CE, "_REQ_SEQ", {})
    monkeypatch.setattr(CE, "_WANDER_BUDGET", {})
    monkeypatch.setattr(S, "_pin_bridge_backend", lambda: None)
    monkeypatch.setattr(S, "_build_chat_brain", lambda brain, r: (_Chat(), "stub-build"))
    monkeypatch.setattr(S, "_reload_persisted_learning", lambda key, chat: None)
    st = {"active": 0, "max_active": 0, "lock": threading.Lock(), "both_in": threading.Event(), "held": []}

    def fake_reply(chat, req, source, key):
        with st["lock"]:
            st["active"] += 1
            st["max_active"] = max(st["max_active"], st["active"])
            if st["active"] >= 2:
                st["both_in"].set()
        st["held"].append(L.held_by_current_thread())
        st["both_in"].wait(0.3)        # give the other thread every chance to enter (it can, only without the lock)
        with st["lock"]:
            st["active"] -= 1
        return JSONResponse({"answer": "ok:%s:%s" % (req.session, req.message), "key": list(key)})

    monkeypatch.setattr(S, "brain_reply", fake_reply)
    return S, st


def _two_sessions(S, n_turns=3):
    barrier = threading.Barrier(2)
    out, errs = {"A": [], "B": []}, []

    def run(sess):
        try:
            for i in range(n_turns):
                barrier.wait(5.0)
                r = S.brain_chat(S.BrainChatRequest(session=sess, message="m%d" % i, brain="tiny-demo",
                                                    renderer="stub", rich=False))
                out[sess].append(json.loads(r.body)["answer"])
        except Exception as e:   # surfaced below, never swallowed
            errs.append(e)

    ts = [threading.Thread(target=run, args=(s,)) for s in ("A", "B")]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30.0)
    assert not errs, errs
    return out


def test_two_sessions_never_overlap_in_the_real_handler(stubbed_handler, monkeypatch):
    S, st = stubbed_handler
    monkeypatch.delenv(L.TURN_LOCK_ENV, raising=False)       # the default: ON
    out = _two_sessions(S)
    assert st["max_active"] == 1                              # never two turns inside the brain at once
    assert all(st["held"])                                    # brain_reply always ran under the lock
    assert out == {"A": ["ok:A:m0", "ok:A:m1", "ok:A:m2"], "B": ["ok:B:m0", "ok:B:m1", "ok:B:m2"]}
    assert sorted(k[0] for k in S._BRAIN_CHATS) == ["A", "B"]


def test_escape_flag_restores_concurrent_turns(stubbed_handler, monkeypatch):
    """The control: the same instrument DOES see two turns overlap when the lock is off (pre-M0 behaviour)."""
    S, st = stubbed_handler
    monkeypatch.setenv(L.TURN_LOCK_ENV, "0")
    _two_sessions(S, n_turns=2)
    assert st["max_active"] == 2
    assert not any(st["held"])


def test_lock_adds_nothing_to_a_stub_body(stubbed_handler, monkeypatch):
    """Plumbing only (the reply is a stub): the lock path returns the handler's body unchanged. NOT reply-identity
    evidence; that is turn_lock_identity_check on the real brain."""
    S, _ = stubbed_handler
    bodies = []
    for val in (None, "0"):
        if val is None:
            monkeypatch.delenv(L.TURN_LOCK_ENV, raising=False)
        else:
            monkeypatch.setenv(L.TURN_LOCK_ENV, val)
        S._BRAIN_CHATS.clear()
        bodies.append([bytes(S.brain_chat(S.BrainChatRequest(session="s", message=m, brain="tiny-demo",
                                                             renderer="stub", rich=False)).body)
                       for m in ("hello", "what does the dog chase")])
    assert bodies[0] == bodies[1]


def test_a_failed_build_releases_the_lock(stubbed_handler, monkeypatch):
    from fastapi import HTTPException
    S, _ = stubbed_handler
    monkeypatch.delenv(L.TURN_LOCK_ENV, raising=False)

    def boom(brain, r):
        raise RuntimeError("no bundle")

    monkeypatch.setattr(S, "_build_chat_brain", boom)
    with pytest.raises(HTTPException):
        S.brain_chat(S.BrainChatRequest(session="s", message="hi", brain="tiny-demo", renderer="stub"))
    got = []
    t = threading.Thread(target=lambda: got.append(L._TURN_LOCK.acquire(timeout=1.0)) or L._TURN_LOCK.release())
    t.start()
    t.join()
    assert got == [True]


def test_turn_end_is_stamped_without_refilling_the_budget(stubbed_handler, monkeypatch):
    from webapp import continuous_engine as CE
    S, _ = stubbed_handler
    monkeypatch.delenv(L.TURN_LOCK_ENV, raising=False)
    t0 = time.time()
    S.brain_chat(S.BrainChatRequest(session="s", message="hi", brain="tiny-demo", renderer="stub"))
    key = ("s", "tiny-demo", "stub")
    assert CE._LAST_TURN_END[key] >= CE._LAST_REQUEST[key] >= t0
    CE._WANDER_BUDGET[key] = 0                 # the tick spent it
    CE.mark_turn_end(key)
    assert CE._WANDER_BUDGET[key] == 0         # only mark_request (a new turn) refills


def test_turn_end_is_stamped_while_the_lock_is_held(stubbed_handler, monkeypatch):
    """Re-review item 3: the real handler writes the end stamp INSIDE the turn lock (try/finally around the turn
    body), so a tick admitted right after the turn always reads it; also when the turn body raises."""
    from fastapi import HTTPException
    from webapp import continuous_engine as CE
    S, _ = stubbed_handler
    monkeypatch.delenv(L.TURN_LOCK_ENV, raising=False)
    held = []
    real = CE.mark_turn_end
    monkeypatch.setattr(CE, "mark_turn_end", lambda key: (held.append(L.held_by_current_thread()), real(key)))
    S.brain_chat(S.BrainChatRequest(session="s", message="hi", brain="tiny-demo", renderer="stub"))

    def boom(brain, r):
        raise RuntimeError("no bundle")

    monkeypatch.setattr(S, "_build_chat_brain", boom)
    with pytest.raises(HTTPException):
        S.brain_chat(S.BrainChatRequest(session="t", message="hi", brain="tiny-demo", renderer="stub"))
    assert held == [True, True]
    assert ("t", "tiny-demo", "stub") in CE._LAST_TURN_END


def test_a_turn_not_admitted_in_time_gets_busy(stubbed_handler, monkeypatch):
    S, _ = stubbed_handler
    monkeypatch.delenv(L.TURN_LOCK_ENV, raising=False)
    monkeypatch.setenv(L.TIMEOUT_ENV, "0.2")
    L.reset_stats()
    inside, release = threading.Event(), threading.Event()

    def holder():
        with L.brain_turn("long_turn"):
            inside.set()
            release.wait(5.0)

    th = threading.Thread(target=holder)
    th.start()
    assert inside.wait(2.0)
    t0 = time.time()
    r = S.brain_chat(S.BrainChatRequest(session="s", message="hi", brain="tiny-demo", renderer="stub"))
    waited = time.time() - t0
    body = json.loads(r.body)
    assert r.status_code == 200 and body["busy"] is True
    assert {"answer", "abstained", "recalled_svo", "verified", "renderer", "brain", "rich"} <= set(body)
    assert body["recalled_svo"] is None and body["turn_lock"]["timeout_s"] == 0.2
    assert 0.15 <= waited < 2.0
    assert L.stats()["n_timeouts"] == 1 and L.waiters() == 0
    assert ("s", "tiny-demo", "stub") not in S._BRAIN_CHATS          # the brain never saw the message
    release.set()
    th.join(5.0)
    r2 = json.loads(S.brain_chat(S.BrainChatRequest(session="s", message="hi", brain="tiny-demo",
                                                    renderer="stub")).body)
    assert r2["answer"] == "ok:s:hi" and "busy" not in r2               # the next turn is served normally


# ── the idle tick (continuous_engine.tick_idle_sessions; the battery's world steps run the same function) ──────────

class _Affect:
    def __init__(self, log, sleep_s=0.0):
        self.log, self.sleep_s = log, sleep_s

    def read_differential(self, v, lesion=False):
        self.log.append(("read", L.held_by_current_thread(), time.perf_counter()))
        if self.sleep_s:
            time.sleep(self.sleep_s)
        return {"differential": 0.0}


class _SelfInit:
    """Stand-in self-init organ: `speak` is the wander; it records whether the turn lock / organ lock were held."""

    def __init__(self, key, log, wander_s=0.0, entered=None):
        self.key, self.log, self.wander_s, self.entered = key, log, wander_s, entered
        self.agents, self.gains_on, self.novelties = ["cat", "dog"], [1.0, 1.0], [0.1, 0.1]

    def _ensure_mouth(self):
        pass

    def speak(self, lesion=False):
        from webapp import continuous_engine as CE
        self.log.append(("wander", L.held_by_current_thread(), CE.selfinit_lock(self.key).locked()))
        if self.entered is not None:
            self.entered.set()
        time.sleep(self.wander_s)
        return {"concept": "dog"}


@pytest.fixture
def tick_env(monkeypatch):
    from webapp import continuous_engine as CE
    monkeypatch.delenv("BRAIN_CONTINUOUS", raising=False)            # default ON
    monkeypatch.delenv("BRAIN_PERSIST_LEARNING", raising=False)
    monkeypatch.delenv("BRAIN_CONTINUOUS_IDEATE", raising=False)
    monkeypatch.setenv("BRAIN_CONTINUOUS_WANDER_MULL", "0")
    for name in ("_LAST_REQUEST", "_LAST_TURN_END", "_REQ_SEQ", "_WANDER_BUDGET", "_INNER_LIFE", "_WANDER_ADAPT",
                 "_SELFINIT_LOCKS"):
        monkeypatch.setattr(CE, name, {})
    assert CE.continuous_enabled()
    return CE


def _mood(n=1):
    return {("s%d" % i, "tiny-demo", "stub"): {"valence": 0.5, "arousal": 0.2} for i in range(n)}


def test_idle_tick_takes_the_turn_lock(tick_env, monkeypatch):
    CE = tick_env
    monkeypatch.delenv(L.TURN_LOCK_ENV, raising=False)
    log = []
    assert CE.tick_idle_sessions(_mood(), lambda: _Affect(log)) == 1
    assert [h for _, h, _ in log] == [True]                          # the shared affect organ was read under the lock


def test_idle_tick_waits_for_a_turn_in_flight(tick_env, monkeypatch):
    CE = tick_env
    monkeypatch.delenv(L.TURN_LOCK_ENV, raising=False)
    log, released = [], []
    mood = _mood()
    with L.brain_turn("turn"):
        t = threading.Thread(target=lambda: CE.tick_idle_sessions(mood, lambda: _Affect(log)))
        t.start()
        time.sleep(0.2)
        assert log == []                                             # blocked: the turn holds the lock
        released.append(time.perf_counter())
    t.join(5.0)
    assert len(log) == 1 and log[0][2] >= released[0]


def test_idle_tick_skips_a_session_marked_while_it_waited(tick_env, monkeypatch):
    """brain_chat marks its session BEFORE taking the lock, and the tick checks idleness INSIDE the lock."""
    CE = tick_env
    monkeypatch.delenv(L.TURN_LOCK_ENV, raising=False)
    key = ("s0", "tiny-demo", "stub")
    log, n = [], []
    mood = _mood()
    with L.brain_turn("turn"):
        t = threading.Thread(target=lambda: n.append(CE.tick_idle_sessions(mood, lambda: _Affect(log))))
        t.start()
        time.sleep(0.1)
        CE.mark_request(key)                                        # the next turn on this session arrives
    t.join(5.0)
    assert n == [0] and log == []


def test_tick_behind_a_same_session_turn_does_not_tick_it_when_the_turn_ends(tick_env, monkeypatch):
    """Review item 3: the turn STARTED 30 s ago (idle by its start stamp), the tick waits behind it; when the turn
    ends (end stamp, then release) the tick reads the clock inside the lock and measures from the end: not idle.
    The stamp-then-release order is the real handler's (pinned by test_turn_end_is_stamped_while_the_lock_is_held)."""
    CE = tick_env
    monkeypatch.delenv(L.TURN_LOCK_ENV, raising=False)
    key = ("s0", "tiny-demo", "stub")
    log, n = [], []
    mood = _mood()
    with L.brain_turn("turn"):
        CE._LAST_REQUEST[key] = time.time() - 30.0                  # a long turn: started 30 s ago
        t = threading.Thread(target=lambda: n.append(CE.tick_idle_sessions(mood, lambda: _Affect(log))))
        t.start()
        time.sleep(0.1)
        CE.mark_turn_end(key)
    t.join(5.0)
    assert n == [0] and log == []


def test_battery_clock_ignores_the_end_stamp(tick_env, monkeypatch):
    """A caller-supplied `now` (the battery's world steps) keeps the pre-M0 reference: the turn-start stamp."""
    CE = tick_env
    key = ("s0", "tiny-demo", "stub")
    CE._LAST_REQUEST[key] = 1000.0
    CE._LAST_TURN_END[key] = time.time()                             # a wall-clock end stamp, far "later"
    log = []
    assert CE.tick_idle_sessions(_mood(), lambda: _Affect(log), now=1000.0 + CE.IDLE_SEC) == 1


def test_wander_runs_outside_the_turn_lock_and_a_turn_gets_in(tick_env, monkeypatch):
    """Review item 1: the ~55 s wander must not hold a turn. A turn (another session) arriving mid-wander is admitted
    within one cheap step, and that other session's turn does not drop this session's wander."""
    CE = tick_env
    monkeypatch.delenv(L.TURN_LOCK_ENV, raising=False)
    log, entered = [], threading.Event()
    key = ("s0", "tiny-demo", "stub")
    CE._WANDER_BUDGET[key] = 1
    org = _SelfInit(key, log, wander_s=1.0, entered=entered)
    th = threading.Thread(target=lambda: CE.tick_idle_sessions(_mood(), lambda: _Affect(log),
                                                               selfinit_getter=lambda k: org))
    th.start()
    assert entered.wait(2.0)
    t0 = time.perf_counter()
    with L.brain_turn("turn"):
        waited = time.perf_counter() - t0
        CE.mark_request(("other", "tiny-demo", "stub"))
    th.join(5.0)
    assert waited < 0.2
    assert ("wander", False, True) in log                            # no turn lock, organ lock held
    rec = CE._INNER_LIFE[key][-1]
    assert rec["wandered"] == "dog" and "wander_dropped" not in rec
    assert CE._WANDER_BUDGET[key] == 0                               # the wander was spent


def test_a_same_session_turn_preempts_the_wander(tick_env, monkeypatch):
    """A turn for THIS session during the wander: the result is dropped and the drop is recorded; the rest of the
    session's tick waits for the next idle period."""
    CE = tick_env
    monkeypatch.delenv(L.TURN_LOCK_ENV, raising=False)
    log, entered = [], threading.Event()
    key = ("s0", "tiny-demo", "stub")
    CE._WANDER_BUDGET[key] = 1
    org = _SelfInit(key, log, wander_s=0.5, entered=entered)
    steps = []
    monkeypatch.setattr(CE, "consolidate_substrate_homeostasis", lambda k, c, trigger="": steps.append(k))
    th = threading.Thread(target=lambda: CE.tick_idle_sessions(_mood(), lambda: _Affect(log),
                                                               selfinit_getter=lambda k: org,
                                                               chat_getter=lambda k: object()))
    th.start()
    assert entered.wait(2.0)
    CE.mark_request(key)                                             # the user speaks mid-wander
    with L.brain_turn("turn"):
        pass
    th.join(5.0)
    rec = CE._INNER_LIFE[key][-1]
    assert rec["wandered"] is None and rec["wander_dropped"]["concept"] == "dog"
    assert "interrupted by your message" in rec["note"]
    assert steps == []                                               # no consolidation step ran for the session
    assert CE.recent_wander(key) is None


def test_a_waiting_turn_is_served_within_one_cheap_step(tick_env, monkeypatch):
    """Review item 2: the measured RLock starvation. Five sessions, 0.3 s per cheap step; a turn arriving at any
    point waits at most about one step (FIFO + the tick re-queues between steps)."""
    CE = tick_env
    monkeypatch.delenv(L.TURN_LOCK_ENV, raising=False)
    log, waits = [], []
    mood = _mood(5)
    first = threading.Event()

    class _A(_Affect):
        def read_differential(self, v, lesion=False):
            first.set()
            return super().read_differential(v, lesion)

    th = threading.Thread(target=lambda: CE.tick_idle_sessions(mood, lambda: _A(log, sleep_s=0.3)))
    th.start()
    assert first.wait(2.0)
    for delay in (0.05, 0.2, 0.1):
        time.sleep(delay)
        t0 = time.perf_counter()
        with L.brain_turn("turn"):
            waits.append(time.perf_counter() - t0)
    th.join(10.0)
    assert len(log) == 5
    assert max(waits) < 0.45, waits                                  # one 0.3 s step plus scheduling slack


def test_idle_tick_escape_takes_no_lock(tick_env, monkeypatch):
    CE = tick_env
    monkeypatch.setenv(L.TURN_LOCK_ENV, "0")
    log = []
    assert CE.tick_idle_sessions(_mood(), lambda: _Affect(log)) == 1
    assert [h for _, h, _ in log] == [False]
