"""c5-next(b): the regression-battery worker's OPT-IN session release (LB_BATTERY_RELEASE_SESSIONS).

brain_chat caches a full ChatBrain + ~20 per-session registries per (session, brain, renderer) key and frees them only
on reset, so a worker that keeps every session holds ~5.4 GiB per session (finding
2026-10-12-battery-worker-per-kind-retention). Default OFF keeps every session (the registered-instrument
behaviour); opt-in evicts a session through the SAME reset teardown once nothing later in its turn list can read it,
and fails loudly when a key it should free was not live. CPU-only, no brain build: brain_chat is replaced by a fake
that caches a stand-in chat object the way the real handler does (except the handler-path key test, which runs the
real handler with its brain build stubbed).
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
    assert B.session_release_plan(labels) == {1: ["emo"], 3: ["dr"]}


@pytest.mark.parametrize("val,expected", [("1", True), ("TRUE", True), ("On", True), (" yes ", True),
                                          ("0", False), ("", False), ("off", False), ("2", False)])
def test_release_env_is_parsed_case_insensitively(val, expected):
    assert B.session_release_enabled({B.RELEASE_SESSIONS_ENV: val}) is expected
    assert B.session_release_enabled({}) is False         # unset -> OFF (the default)


def test_release_raises_when_a_key_was_not_live():
    evicted = []
    with pytest.raises(RuntimeError, match="not live"):
        B._release_sessions({"a": [("a", "b", "c")], "x": [("x", "b", "c")]}, evicted.append,
                            {("a", "b", "c"): object()})
    assert evicted == [("a", "b", "c"), ("x", "b", "c")]   # every key is still evicted before the raise


def test_no_recorded_key_is_checked_per_session():
    # one planned session has its key, the other recorded none: the release must name the empty one, not pass
    # because the release as a whole had a key -- and the session that has a key is still evicted
    evicted = []
    with pytest.raises(RuntimeError, match=r"no cache key was recorded for session\(s\) \['bc'\]"):
        B._release_sessions({"emo": [("emo", "b", "c")], "bc": []}, evicted.append, {("emo", "b", "c"): object()})
    assert evicted == [("emo", "b", "c")]


def test_released_multiref_organs_stay_alive_for_id_uniqueness():
    # the wave3 pool tracks its d6 slice owner by id(MultiReferentWMOrgan): a released session's organ is kept
    # alive so its address can never be reused by a later session's organ
    class _Org:
        pass
    key = ("emo", "tiny-demo", "stub")
    org = _Org()
    multiref = {key: org}
    ref = weakref.ref(org)
    del org
    keep = []
    B._release_sessions({"emo": [key]}, multiref.pop, {key: object()}, keep_alive=keep, multiref=multiref)
    gc.collect()
    assert key not in multiref and ref() is not None and keep == [ref()]


# ── the worker, with a fake brain_chat ──────────────────────────────────────────────────────────────────────────────

class _Chat:
    """Stand-in for a ChatBrain (only identity and liveness matter here)."""


@pytest.fixture
def fake_server(monkeypatch, tmp_path):
    from fastapi.responses import JSONResponse
    from webapp import server as S

    for k, v in (("SIM_BACKEND", "numpy"), ("BRAIN_CHAT_RENDERER", "stub"), ("SIM_DISABLE_LLM", "1")):
        monkeypatch.setenv(k, v)              # the worker setdefaults these: monkeypatch restores them
    side = tmp_path / "side.jsonl"
    monkeypatch.setenv("BRAIN_IDLE_REPLAY_LEAKLOG", str(side))
    monkeypatch.delenv(B.RELEASE_SESSIONS_ENV, raising=False)
    monkeypatch.setattr(S, "_BRAIN_CHATS", {})
    monkeypatch.setattr(S, "_SESSION_MOOD", {})
    st = {"refs": {}, "seen_live": [], "skip_cache": set(), "side": side}
    handler_reset = S._evict_brain_chat_session   # the handler's own reset path (a test may break the WORKER's evict)

    def fake_brain_chat(req):
        key = S.brain_chat_cache_key(req)
        st["seen_live"].append((req.session, sorted(k[0] for k in S._BRAIN_CHATS)))   # cached when this turn starts
        if req.reset:
            handler_reset(key)
        if req.session not in st["skip_cache"]:
            chat = S._BRAIN_CHATS.get(key)
            if chat is None:
                chat = _Chat()
                S._BRAIN_CHATS[key] = chat
                st["refs"][req.session] = weakref.ref(chat)
            S._SESSION_MOOD[key] = {"v": 0.0}
        return JSONResponse({"answer": "ok:" + req.message})

    monkeypatch.setattr(S, "brain_chat", fake_brain_chat)
    return S, st


def _side(st):
    return [json.loads(line) for line in st["side"].read_text().splitlines()]


def test_default_keeps_every_session(fake_server, tmp_path):
    S, st = fake_server
    assert B._collect_worker("{}", ["emo", "bc_a", "bc_b"], str(tmp_path / "arm.json")) == 0
    assert st["seen_live"] == [("emo", []), ("bc", ["emo"]), ("bc", ["bc", "emo"])]
    assert sorted(k[0] for k in S._BRAIN_CHATS) == ["bc", "emo"]
    recs = _side(st)
    assert [r["event"] for r in recs] == ["battery_turn"] * 3          # no release records when OFF
    assert all(r["session_release"] is False for r in recs)


def test_opt_in_frees_a_session_before_the_next_session_builds(fake_server, tmp_path, monkeypatch):
    S, st = fake_server
    monkeypatch.setenv(B.RELEASE_SESSIONS_ENV, "True")
    out = tmp_path / "arm.json"
    assert B._collect_worker("{}", ["emo", "bc_a", "bc_b"], str(out)) == 0
    # 'emo' is gone when 'bc' starts, and 'bc_b' still sees its own session from bc_a
    assert st["seen_live"] == [("emo", []), ("bc", []), ("bc", ["bc"])]
    gc.collect()
    assert st["refs"]["emo"]() is None and st["refs"]["bc"]() is None      # nothing else pins the released chats
    assert S._BRAIN_CHATS == {} and S._SESSION_MOOD == {}
    body = json.loads(out.read_text())
    assert [body[k]["answer"] for k in ("emo", "bc_a", "bc_b")] == [
        "ok:" + B._TURN_BY_LABEL[k][1] for k in ("emo", "bc_a", "bc_b")]
    assert not any("_battery_release_error" in v for v in body.values())
    recs = _side(st)
    assert [(r["event"], r["turn"]) for r in recs] == [
        ("battery_turn", "emo"), ("battery_release", "emo"), ("battery_turn", "bc_a"),
        ("battery_turn", "bc_b"), ("battery_release", "bc_b")]
    assert all(r["session_release"] is True for r in recs if r["event"] == "battery_turn")
    rel = [r for r in recs if r["event"] == "battery_release"]
    assert [(r["sessions"], r["n_keys"], r["error"]) for r in rel] == [(["emo"], 1, None), (["bc"], 1, None)]


def test_a_key_that_was_not_live_is_recorded_loudly(fake_server, tmp_path, monkeypatch):
    S, st = fake_server
    monkeypatch.setenv(B.RELEASE_SESSIONS_ENV, "1")
    st["skip_cache"].add("emo")               # the handler never cached 'emo' -> releasing it frees nothing
    out = tmp_path / "arm.json"
    assert B._collect_worker("{}", ["emo", "bc_a"], str(out)) == 0
    body = json.loads(out.read_text())
    assert "not live" in body["emo"]["_battery_release_error"]
    assert "_battery_release_error" not in body["bc_a"]
    rel = [r for r in _side(st) if r["event"] == "battery_release"]
    assert "not live" in rel[0]["error"] and rel[1]["error"] is None


def test_a_reset_hook_exception_is_recorded_not_raised(fake_server, tmp_path, monkeypatch):
    S, st = fake_server
    monkeypatch.setenv(B.RELEASE_SESSIONS_ENV, "1")

    def boom(key):
        raise ValueError("reset hook failed")

    monkeypatch.setattr(S, "_evict_brain_chat_session", boom)
    out = tmp_path / "arm.json"
    assert B._collect_worker("{}", ["emo", "bc_a"], str(out)) == 0      # the worker completes
    body = json.loads(out.read_text())
    assert body["emo"]["_battery_release_error"] == "ValueError: reset hook failed"
    assert body["emo"]["answer"] == "ok:" + B._TURN_BY_LABEL["emo"][1]


# ── the server side ─────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("renderer", ["STUB", None])
def test_cache_key_is_the_key_the_handler_caches_under(monkeypatch, renderer):
    """Runs the REAL brain_chat handler (brain build + reply stubbed) and checks the key it cached under equals
    brain_chat_cache_key(req), including the default-renderer path (renderer=None)."""
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
    monkeypatch.setattr(S, "brain_reply", lambda chat, req, source, key: JSONResponse({"key": list(key)}))
    req = S.BrainChatRequest(session="emo", message="hi", brain="tiny-demo", renderer=renderer, rich=False,
                             reset=False)
    resp = S.brain_chat(req)
    assert list(S._BRAIN_CHATS) == [("emo", "tiny-demo", "stub")]           # the literal key, both renderers
    assert S.brain_chat_cache_key(req) == ("emo", "tiny-demo", "stub")
    assert tuple(json.loads(resp.body)["key"]) == S.brain_chat_cache_key(req)


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
