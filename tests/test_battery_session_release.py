"""c5-next(b): the regression-battery worker releases each session's server state after its last use.

brain_chat caches a full ChatBrain + ~20 per-session registries per (session, brain, renderer) key and frees them only
on reset, so a worker that kept every session held ~5.3 GB per session (finding
2026-10-12-battery-worker-per-kind-retention). The worker now evicts a session through the SAME reset teardown once
nothing later in its turn list can read it. CPU-only, no brain build: brain_chat is replaced by a fake that caches a
stand-in chat object the way the real handler does.
"""
import gc
import json
import weakref

import pytest

from research.runners import onebrain_regression_battery as B


# ── the plan (pure: labels only) ─────────────────────────────────────────────────────────────────────────────────────

def test_plan_releases_each_session_after_its_last_turn():
    # emo has one turn; bc_a/bc_b share session 'bc'; rich_well is its own session
    assert B.session_release_plan(["emo", "bc_a", "bc_b", "rich_well"]) == {0: ["emo"], 2: ["bc"], 3: ["richw"]}


def test_plan_keeps_a_session_live_through_a_later_world_step():
    world = next(iter(B._WORLD_STEPS))      # any label-only environment step
    plan = B.session_release_plan(["emo", world, "bc_a", "bc_b"])
    # world steps tick every live session, so 'emo' must survive the step at index 1 and is released after it;
    # 'bc' starts after the step, so it is released after its own last turn
    assert plan == {1: ["emo"], 3: ["bc"]}


def test_plan_without_world_steps_never_releases_before_the_last_use():
    labels = ["dr_a", "emo", "dr_b", "dr_c"]   # session 'dr' is interleaved with 'emo'
    plan = B.session_release_plan(labels)
    assert plan == {1: ["emo"], 3: ["dr"]}


# ── the worker, with a fake brain_chat ──────────────────────────────────────────────────────────────────────────────

class _Chat:
    """Stand-in for a ChatBrain (only identity and liveness matter here)."""


@pytest.fixture
def fake_server(monkeypatch):
    from fastapi.responses import JSONResponse
    from webapp import server as S

    for k in ("SIM_BACKEND", "BRAIN_CHAT_RENDERER", "SIM_DISABLE_LLM"):   # the worker setdefaults these: restore
        monkeypatch.setenv(k, {"SIM_BACKEND": "numpy", "BRAIN_CHAT_RENDERER": "stub", "SIM_DISABLE_LLM": "1"}[k])
    monkeypatch.delenv("BRAIN_IDLE_REPLAY_LEAKLOG", raising=False)
    monkeypatch.delenv(B.KEEP_SESSIONS_ENV, raising=False)
    monkeypatch.setattr(S, "_BRAIN_CHATS", {})
    monkeypatch.setattr(S, "_SESSION_MOOD", {})
    refs, seen_live = {}, []

    def fake_brain_chat(req):
        key = S.brain_chat_cache_key(req)
        seen_live.append((req.session, sorted(k[0] for k in S._BRAIN_CHATS)))   # what is cached when this turn starts
        if req.reset:
            S._evict_brain_chat_session(key)
        chat = S._BRAIN_CHATS.get(key)
        if chat is None:
            chat = _Chat()
            S._BRAIN_CHATS[key] = chat
            refs[req.session] = weakref.ref(chat)
        S._SESSION_MOOD[key] = {"v": 0.0}
        return JSONResponse({"answer": "ok:" + req.message})

    monkeypatch.setattr(S, "brain_chat", fake_brain_chat)
    return S, refs, seen_live


def test_worker_frees_a_session_before_the_next_session_builds(fake_server, tmp_path):
    S, refs, seen_live = fake_server
    out = tmp_path / "arm.json"
    assert B._collect_worker("{}", ["emo", "bc_a", "bc_b"], str(out)) == 0
    # 'emo' is gone when 'bc' starts, and 'bc_b' still sees its own session from bc_a
    assert seen_live == [("emo", []), ("bc", []), ("bc", ["bc"])]
    gc.collect()
    assert refs["emo"]() is None and refs["bc"]() is None      # nothing else pins the released chats
    assert S._BRAIN_CHATS == {} and S._SESSION_MOOD == {}
    body = json.loads(out.read_text())
    assert [body[k]["answer"] for k in ("emo", "bc_a", "bc_b")] == [
        "ok:" + B._TURN_BY_LABEL[k][1] for k in ("emo", "bc_a", "bc_b")]


def test_keep_sessions_env_restores_the_old_behaviour(fake_server, tmp_path, monkeypatch):
    S, refs, seen_live = fake_server
    monkeypatch.setenv(B.KEEP_SESSIONS_ENV, "1")
    assert B._collect_worker("{}", ["emo", "bc_a", "bc_b"], str(tmp_path / "arm.json")) == 0
    assert seen_live == [("emo", []), ("bc", ["emo"]), ("bc", ["bc", "emo"])]
    assert sorted(k[0] for k in S._BRAIN_CHATS) == ["bc", "emo"]


def test_cache_key_matches_the_handler_expression():
    from webapp import server as S
    req = S.BrainChatRequest(session="emo", message="hi", brain="tiny-demo", renderer="STUB", rich=False, reset=True)
    assert S.brain_chat_cache_key(req) == ("emo", "tiny-demo", "stub")


def test_evict_drops_every_server_registry_for_the_key(monkeypatch):
    from webapp import server as S
    key = ("s", "tiny-demo", "stub")
    names = ("_BRAIN_CHATS", "_BRAIN_RICH", "_SESSION_MOOD", "_SESSION_WORLDVIEW", "_SESSION_MULTIREF",
             "_SESSION_SILENT_WM", "_SESSION_SELFINIT", "_SESSION_DISCOURSE", "_SESSION_PMEM")
    other = ("t", "tiny-demo", "stub")
    for n in names:
        monkeypatch.setattr(S, n, {key: object(), other: object()})
    S._evict_brain_chat_session(key)
    for n in names:
        assert list(getattr(S, n)) == [other], n
