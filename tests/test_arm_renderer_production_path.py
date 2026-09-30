"""The load-bearing battery's arm renderer (research/runners/onebrain_regression_battery.py, LB_ARM_RENDERER).

Governing text: research/findings/2026-09-29-production-default-battery-B2d-PREREGISTRATION.md, "Mouth-channel arm"
and its Amendment 1 (research/b2d-mouth-renderer). Two claims are pinned here, without building a brain:

1. DEFAULT (stub) PATH UNCHANGED. With LB_ARM_RENDERER unset, the worker sets the same three environment defaults,
   sends the same BrainChatRequest on every probe turn, writes byte-identical arm files, and `_spawn_arm` launches the
   same argv, as the worker did before the renderer mode existed (the pre-change worker is kept verbatim below as
   `_old_collect_worker`, copied from origin/main f5e3732fe).
2. PRODUCTION PATH EQUALS PRODUCTION. With LB_ARM_RENDERER=production, the worker calls the very handler the
   `/api/brain-chat` route serves, with the request the console sends for its default 'auto' renderer (renderer
   omitted), sets none of the three defaults, and the handler then resolves the renderer with
   `_default_brain_renderer()` exactly as it does for a console turn. The per-turn Qwen call counter is a transparent
   wrapper.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("SIM_BACKEND", "numpy")

import research.runners.onebrain_regression_battery as B  # noqa: E402
# Imported once HERE, with SIM_BACKEND set: on a CUDA host, importing webapp.server with SIM_BACKEND unset defaults
# it to cupy (the production behavior), which would otherwise happen inside whichever test imports it first.
import webapp.server  # noqa: E402,F401

_ENV_KEYS = ("SIM_BACKEND", "BRAIN_CHAT_RENDERER", "SIM_DISABLE_LLM", "LB_ARM_RENDERER")


@pytest.fixture
def clean_env():
    """Snapshot and restore the environment keys the worker reads or setdefaults (setdefault writes keys that
    monkeypatch would not know to remove)."""
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    for k in _ENV_KEYS:
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


class _Resp:
    def __init__(self, payload):
        self.body = json.dumps(payload).encode()


def _spy_handler(record, payload_fn=None, side_effect=None):
    def brain_chat(req):
        record.append(req)
        if side_effect is not None:
            side_effect(req)
        payload = payload_fn(req) if payload_fn else {"answer": "ok:%s" % req.message, "abstained": False,
                                                      "recalled_svo": None, "renderer": "spy"}
        return _Resp(payload)
    return brain_chat


# The worker exactly as it was on origin/main f5e3732fe (research/runners/onebrain_regression_battery.py:882-912),
# kept verbatim so the default path can be compared against it byte for byte.
def _old_collect_worker(env_json, turn_labels, out_path):
    os.environ.setdefault("SIM_BACKEND", "numpy")
    os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")
    os.environ.setdefault("SIM_DISABLE_LLM", "1")
    env = json.loads(env_json)
    for k, v in env.items():
        os.environ[k] = v
    from webapp.server import brain_chat, BrainChatRequest
    responses = {}
    for label in turn_labels:
        if label in B._WORLD_STEPS:
            try:
                responses[label] = B._run_world_step(B._WORLD_STEPS[label])
            except Exception as e:
                responses[label] = {"_error": "%s: %s" % (type(e).__name__, e)}
            continue
        _, msg, session, reset, percept, rich = B._TURN_BY_LABEL[label]
        try:
            kwargs = dict(session=session, message=msg, brain="tiny-demo",
                          renderer="stub", rich=bool(rich), reset=reset)
            if percept is not None:
                kwargs["percept"] = percept
            r = brain_chat(BrainChatRequest(**kwargs))
            responses[label] = json.loads(r.body)
        except Exception as e:
            responses[label] = {"_error": "%s: %s" % (type(e).__name__, e)}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(responses, open(out_path, "w"), indent=2, default=str)
    return 0


def _brain_turn_labels():
    return [lab for lab in B._TURN_BY_LABEL if lab not in B._WORLD_STEPS]


# ─────────────────────────────────────────── mode parsing ───────────────────────────────────────────

def test_mode_defaults_to_stub_and_rejects_unknown_values():
    assert B.arm_renderer_mode({}) == "stub"
    assert B.arm_renderer_mode({"LB_ARM_RENDERER": ""}) == "stub"
    assert B.arm_renderer_mode({"LB_ARM_RENDERER": " Stub "}) == "stub"
    assert B.arm_renderer_mode({"LB_ARM_RENDERER": "production"}) == "production"
    for bad in ("qwen", "auto", "prod", "1"):
        with pytest.raises(ValueError):
            B.arm_renderer_mode({"LB_ARM_RENDERER": bad})


def test_stub_request_kwargs_are_the_pre_change_literal():
    for label in _brain_turn_labels():
        _, msg, session, reset, percept, rich = B._TURN_BY_LABEL[label]
        want = dict(session=session, message=msg, brain="tiny-demo", renderer="stub", rich=bool(rich), reset=reset)
        if percept is not None:
            want["percept"] = percept
        got = B._arm_request_kwargs("stub", session, msg, reset, percept, rich)
        assert got == want and list(got) == list(want), label


# ─────────────────────────────────────── 1. default path unchanged ───────────────────────────────────────

def test_stub_worker_is_byte_identical_to_the_pre_change_worker(clean_env, monkeypatch, tmp_path):
    import webapp.server as S
    labels = _brain_turn_labels()
    old_reqs, new_reqs = [], []

    monkeypatch.setattr(S, "brain_chat", _spy_handler(old_reqs))
    assert _old_collect_worker("{}", labels, str(tmp_path / "old.json")) == 0
    old_env = {k: os.environ.get(k) for k in _ENV_KEYS}
    for k in _ENV_KEYS:
        os.environ.pop(k, None)

    monkeypatch.setattr(S, "brain_chat", _spy_handler(new_reqs))
    assert B._collect_worker("{}", labels, str(tmp_path / "new.json")) == 0
    new_env = {k: os.environ.get(k) for k in _ENV_KEYS}

    assert (tmp_path / "old.json").read_bytes() == (tmp_path / "new.json").read_bytes()
    assert [r.model_dump() for r in old_reqs] == [r.model_dump() for r in new_reqs]
    assert all(r.renderer == "stub" for r in new_reqs)
    assert new_env == old_env == {"SIM_BACKEND": "numpy", "BRAIN_CHAT_RENDERER": "stub", "SIM_DISABLE_LLM": "1",
                                  "LB_ARM_RENDERER": None}
    assert "_arm_renderer" not in (tmp_path / "new.json").read_text()


def test_explicit_stub_value_equals_unset(clean_env, monkeypatch, tmp_path):
    import webapp.server as S
    a, b = [], []
    monkeypatch.setattr(S, "brain_chat", _spy_handler(a))
    B._collect_worker("{}", ["well", "emo"], str(tmp_path / "unset.json"))
    for k in _ENV_KEYS:
        os.environ.pop(k, None)
    os.environ["LB_ARM_RENDERER"] = "stub"
    monkeypatch.setattr(S, "brain_chat", _spy_handler(b))
    B._collect_worker("{}", ["well", "emo"], str(tmp_path / "stub.json"))
    assert (tmp_path / "unset.json").read_bytes() == (tmp_path / "stub.json").read_bytes()
    assert [r.model_dump() for r in a] == [r.model_dump() for r in b]


def test_spawn_arm_argv_does_not_depend_on_the_mode(clean_env, monkeypatch, tmp_path):
    seen = []

    class _P:
        returncode = 1

    def fake_run(argv, env=None, **kw):
        seen.append((list(argv), dict(env or {})))
        return _P()

    monkeypatch.setattr(B.subprocess, "run", fake_run)
    out = str(tmp_path / "arm.json")
    B._spawn_arm({"X": "1"}, ["well"], out)
    os.environ["LB_ARM_RENDERER"] = "production"
    B._spawn_arm({"X": "1"}, ["well"], out)
    (argv0, env0), (argv1, env1) = seen
    assert argv0 == argv1 == [sys.executable, "-u", "-m", "research.runners.onebrain_regression_battery", "--worker",
                              "--env", json.dumps({"X": "1"}), "--turns", "well", "--out", out]
    assert "LB_ARM_RENDERER" not in env0 and env1["LB_ARM_RENDERER"] == "production"


# ─────────────────────────────────────── 2. production path = production ───────────────────────────────────────

def _install_fake_generate(monkeypatch):
    """Replace SpikingQwenFaculty's two generation entry points with cheap fakes (monkeypatch restores the real ones,
    and with them anything the worker's counter wrapped on top)."""
    from research.runners._grounded_lang_integration_derisk import SpikingQwenFaculty

    def fake_generate(self, user_msg, seed=None):
        return ("gen:" + user_msg, "gen:" + user_msg, 0.0)

    def fake_generate_batch(self, user_msgs, seed=None):
        return ([("gen:" + m, "gen:" + m) for m in user_msgs], 0.0)

    monkeypatch.setattr(SpikingQwenFaculty, "_generate", fake_generate)
    monkeypatch.setattr(SpikingQwenFaculty, "_generate_batch", fake_generate_batch)
    return SpikingQwenFaculty


def test_production_worker_sends_the_console_auto_request(clean_env, monkeypatch, tmp_path):
    import webapp.server as S
    _install_fake_generate(monkeypatch)
    os.environ["LB_ARM_RENDERER"] = "production"
    os.environ["SIM_BACKEND"] = "cupy"
    reqs = []
    monkeypatch.setattr(S, "brain_chat", _spy_handler(reqs))
    labels = ["well", "emo"]
    assert B._collect_worker("{}", labels, str(tmp_path / "p.json")) == 0
    # the console's default 'auto' option: app.js rendererParam() -> null; everything else from the probe turn
    for req, label in zip(reqs, labels):
        _, msg, session, reset, percept, rich = B._TURN_BY_LABEL[label]
        console_body = {"session": session, "message": msg, "brain": "tiny-demo", "renderer": None,
                        "rich": bool(rich), "reset": reset}
        if percept is not None:
            console_body["percept"] = percept
        assert req.model_dump() == S.BrainChatRequest(**console_body).model_dump(), label
        assert req.renderer is None
    # production sets none of the stub defaults (and the worker did not touch the backend it was given)
    assert "BRAIN_CHAT_RENDERER" not in os.environ and "SIM_DISABLE_LLM" not in os.environ
    assert os.environ["SIM_BACKEND"] == "cupy"
    out = json.loads((tmp_path / "p.json").read_text())
    assert out["well"]["_arm_renderer"] == {"mode": "production", "qwen_generate_calls": 0}


def test_production_worker_counts_qwen_calls_per_turn_transparently(clean_env, monkeypatch, tmp_path):
    import webapp.server as S
    fac_cls = _install_fake_generate(monkeypatch)
    os.environ["LB_ARM_RENDERER"] = "production"
    returned = []

    def payload(req):
        # the handler would render through the warm faculty: 2 single calls + 1 batch on 'well', none on 'emo'
        if req.message == B._TURN_BY_LABEL["well"][1]:
            returned.append(fac_cls._generate(object(), "p1"))
            returned.append(fac_cls._generate(object(), "p2"))
            returned.append(fac_cls._generate_batch(object(), ["b1", "b2"]))
        return {"answer": "x", "abstained": False, "recalled_svo": None}

    monkeypatch.setattr(S, "brain_chat", _spy_handler([], payload_fn=payload))
    assert B._collect_worker("{}", ["well", "emo"], str(tmp_path / "c.json")) == 0
    out = json.loads((tmp_path / "c.json").read_text())
    assert out["well"]["_arm_renderer"]["qwen_generate_calls"] == 3
    assert out["emo"]["_arm_renderer"]["qwen_generate_calls"] == 0
    # transparent: the wrapped entry points returned exactly what the originals return
    assert returned == [("gen:p1", "gen:p1", 0.0), ("gen:p2", "gen:p2", 0.0),
                        ([("gen:b1", "gen:b1"), ("gen:b2", "gen:b2")], 0.0)]


@pytest.mark.parametrize("forced", ["BRAIN_CHAT_RENDERER", "SIM_DISABLE_LLM"])
def test_production_worker_refuses_a_forced_resolution(clean_env, monkeypatch, tmp_path, forced):
    import webapp.server as S
    os.environ["LB_ARM_RENDERER"] = "production"
    os.environ[forced] = "qwen" if forced == "BRAIN_CHAT_RENDERER" else "1"
    reqs = []
    monkeypatch.setattr(S, "brain_chat", _spy_handler(reqs))
    assert B._collect_worker("{}", ["well"], str(tmp_path / "r.json")) == 3
    assert reqs == [] and not (tmp_path / "r.json").exists()


def test_the_worker_calls_the_handler_the_production_route_serves():
    import webapp.server as S
    endpoints = [r.endpoint for r in S.app.routes if getattr(r, "path", None) == "/api/brain-chat"]
    assert endpoints == [S.brain_chat]


class _Sentinel(RuntimeError):
    pass


@pytest.mark.parametrize("mode,env,want", [
    ("production", {"SIM_BACKEND": "numpy"}, "stub"),          # a GPU-free host: auto -> stub, as for the console
    ("production", {"SIM_BACKEND": "cupy", "BRAIN_CHAT_RENDERER": "qwen"}, "qwen"),   # a host the resolver sends to qwen
    ("stub", {"SIM_BACKEND": "cupy", "BRAIN_CHAT_RENDERER": "qwen"}, "stub"),         # stub mode stays stub regardless
])
def test_the_handler_resolves_the_omitted_renderer_with_the_production_resolver(clean_env, monkeypatch, mode, env,
                                                                                 want):
    """brain_chat itself (not a spy): for the request the worker builds, the renderer handed to _build_chat_brain is
    `_default_brain_renderer()` (production mode) or 'stub' (stub mode). _build_chat_brain is stubbed to record and
    raise, so no brain is built."""
    import webapp.server as S
    from fastapi import HTTPException
    for k, v in env.items():
        os.environ[k] = v
    seen = []

    def fake_build(brain, renderer):
        seen.append((brain, renderer))
        raise _Sentinel("recorded")

    monkeypatch.setattr(S, "_build_chat_brain", fake_build)
    _, msg, session, reset, percept, rich = B._TURN_BY_LABEL["well"]
    kwargs = B._arm_request_kwargs(mode, "armtest-%s-%s" % (mode, want), msg, reset, percept, rich)
    with pytest.raises(HTTPException):
        S.brain_chat(S.BrainChatRequest(**kwargs))
    assert seen == [("tiny-demo", want)]
    if mode == "production":
        assert S._default_brain_renderer() == want


def test_counter_wrapper_is_transparent_and_uninstalls():
    class K:
        def _generate(self, x, seed=None):
            return ("g", x, seed)

        def _generate_batch(self, xs, seed=None):
            return (list(xs), seed)

    orig_g, orig_b = K._generate, K._generate_batch
    c = B._QwenGenerateCounter().install(K)
    k = K()
    assert k._generate("a", seed=3) == ("g", "a", 3)
    assert k._generate_batch(["a", "b"]) == (["a", "b"], None)
    assert c.calls == 2
    c.install(K)                      # a second install never double-wraps
    k._generate("z")
    assert c.calls == 3
    c.uninstall(K)
    assert K._generate is orig_g and K._generate_batch is orig_b



def test_qwen_renderer_name_is_the_one_the_mouth_scorer_requires():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import b2d_mouth_score as M
    src = open(os.path.join(ROOT, "research", "runners", "brain_chat_tui.py")).read()
    assert 'name = "%s"' % M.QWEN_RENDERER_NAME in src
