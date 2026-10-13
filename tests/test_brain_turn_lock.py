"""M0 of the one-brain shared-memory plan: ONE process-level turn lock (webapp/brain_turn_lock.py).

docs/plans/2026-10-13-one-brain-shared-memory-DESIGN.md section 2. `brain_chat` is a sync FastAPI route (threadpool),
so before M0 two conversations could run turns at once against the process-shared organs, and the idle tick
(continuous_engine.tick_idle_sessions, run in a thread executor) could run in the middle of a turn. CPU-only, no
brain build: the real `brain_chat` handler runs with its brain build and `brain_reply` stubbed.

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


# ── the real brain_chat handler, brain build + reply stubbed ────────────────────────────────────────────────────────

class _Chat:
    """Stand-in for a ChatBrain."""


@pytest.fixture
def stubbed_handler(monkeypatch):
    from fastapi.responses import JSONResponse
    from webapp import server as S
    from webapp import continuous_engine as CE

    monkeypatch.setenv("BRAIN_CHAT_RENDERER", "stub")
    monkeypatch.setattr(S, "_BRAIN_CHATS", {})
    monkeypatch.setattr(CE, "_LAST_REQUEST", {})
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


def test_single_caller_response_is_identical_with_and_without_the_lock(stubbed_handler, monkeypatch):
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


# ── the idle tick (continuous_engine.tick_idle_sessions; the battery's world steps run the same function) ──────────

class _Affect:
    def __init__(self, log):
        self.log = log

    def read_differential(self, v, lesion=False):
        self.log.append(("read", L.held_by_current_thread(), time.perf_counter()))
        return {"differential": 0.0}


@pytest.fixture
def tick_env(monkeypatch):
    from webapp import continuous_engine as CE
    monkeypatch.delenv("BRAIN_CONTINUOUS", raising=False)            # default ON
    monkeypatch.delenv("BRAIN_PERSIST_LEARNING", raising=False)
    monkeypatch.setattr(CE, "_LAST_REQUEST", {})
    monkeypatch.setattr(CE, "_WANDER_BUDGET", {})
    monkeypatch.setattr(CE, "_INNER_LIFE", {})
    assert CE.continuous_enabled()
    return CE


def test_idle_tick_takes_the_turn_lock(tick_env, monkeypatch):
    CE = tick_env
    monkeypatch.delenv(L.TURN_LOCK_ENV, raising=False)
    log = []
    mood = {("s", "tiny-demo", "stub"): {"valence": 0.5, "arousal": 0.2}}
    assert CE.tick_idle_sessions(mood, lambda: _Affect(log)) == 1
    assert [h for _, h, _ in log] == [True]                          # the shared affect organ was read under the lock


def test_idle_tick_waits_for_a_turn_in_flight(tick_env, monkeypatch):
    CE = tick_env
    monkeypatch.delenv(L.TURN_LOCK_ENV, raising=False)
    log, released = [], []
    mood = {("s", "tiny-demo", "stub"): {"valence": 0.5, "arousal": 0.2}}
    with L.brain_turn("turn"):
        t = threading.Thread(target=lambda: CE.tick_idle_sessions(mood, lambda: _Affect(log)))
        t.start()
        time.sleep(0.2)
        assert log == []                                             # blocked: the turn holds the lock
        released.append(time.perf_counter())
    t.join(5.0)
    assert len(log) == 1 and log[0][2] >= released[0]


def test_idle_tick_skips_a_session_marked_while_it_waited(tick_env, monkeypatch):
    """brain_chat marks its session BEFORE taking the lock, and the tick checks idleness INSIDE the lock, so a
    turn that is waiting (or about to run) on that session is never ticked under it."""
    CE = tick_env
    monkeypatch.delenv(L.TURN_LOCK_ENV, raising=False)
    key = ("s", "tiny-demo", "stub")
    log, n = [], []
    mood = {key: {"valence": 0.5, "arousal": 0.2}}
    with L.brain_turn("turn"):
        t = threading.Thread(target=lambda: n.append(CE.tick_idle_sessions(mood, lambda: _Affect(log))))
        t.start()
        time.sleep(0.1)
        CE.mark_request(key)                                        # the next turn on this session arrives
    t.join(5.0)
    assert n == [0] and log == []


def test_idle_tick_escape_takes_no_lock(tick_env, monkeypatch):
    CE = tick_env
    monkeypatch.setenv(L.TURN_LOCK_ENV, "0")
    log = []
    mood = {("s", "tiny-demo", "stub"): {"valence": 0.5, "arousal": 0.2}}
    assert CE.tick_idle_sessions(mood, lambda: _Affect(log)) == 1
    assert [h for _, h, _ in log] == [False]
