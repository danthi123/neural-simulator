"""Regression pin for the one-brain Stage-2 BUILD-AHEAD flag, `BRAIN_TOUCHPOINT_A_FACT_CLAUSE` in
`research/runners/rich_answer_composer.py::RichAnswerComposer._render_one_verified` /
`_touchpoint_a_fact_clause_enabled`. See that file's flag-block comment (just above
`_touchpoint_a_fact_clause_enabled`) for the full measured context and citations; this file pins the WIRING
behaviour with fast, deterministic, mocked fixtures -- no GPU, no heavy model, no real ChatBrain build.

THIS FLAG IS PREP, NOT A LANDED RESULT (see the paired de-risk runner
`research/runners/_touchpoint_a_fact_clause_derisk.py`'s own module docstring: the full measure+retire battery
is DEFERRED to compute-availability). What THIS file proves, independent of that deferred measurement:
  1. Default OFF, and OFF is byte-identical to the pre-flag `_render_one_verified` -- the fact-clause render is
     never even imported/called (a poison-pill on `render_fact_sentence` never trips).
  2. Flag ON + a relation the fact-clause lexicon covers: the fact-clause render is used and the Qwen/template
     renderer is genuinely SKIPPED (a poison-pill on `renderer.render_svo` never trips) -- not merely that its
     output is discarded.
  3. Flag ON + an uncovered relation (or a raised exception inside the fact-clause attempt): falls straight
     through to the pre-existing renderer, UNCHANGED -- a pure additive safety net, never a replacement that
     could drop content.
  4. Scope guard: a `spiking_recall_surface` HIT (the bounded transitive-SVO spiking Broca mouth already
     covered this SVO) takes precedence, unchanged -- the fact-clause attempt is never even reached.
  5. SVO-identity guard: `render_fact_sentence` is called with EXACTLY a one-item list containing THIS gathered
     svo, never the composer's broader gathered set -- so it can only ever render this fact or return None,
     never substitute a different one `pick_covered_fact` might otherwise have picked from a longer list.
  6. A fact-clause HIT is trusted directly (verified=True), the SAME trust model the pre-existing
     `spiking_recall_surface` hit already uses -- `chat._verify`/`_verify_claim_set` are not additionally
     invoked for it.
  7. `_touchpoint_a_fact_clause_enabled()` itself: the standard truthy/falsy env-parsing contract this repo's
     other default-OFF flags use (unset/0/false/off/no -> False; 1/true/on/yes, case-insensitive -> True).

Run: CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy .venv/bin/python -m pytest tests/test_touchpoint_a_fact_clause_flag.py -q
"""
from __future__ import annotations

import os
import types

os.environ.setdefault("SIM_BACKEND", "numpy")

import pytest

from research.runners import rich_answer_composer as RAC
from webapp import wkv_mouth_generator as WKV

pytestmark = pytest.mark.filterwarnings("ignore")

_SVO = ("angora_turkey", "located_in_time_zone", "kaliningrad_time")
_RENDERER_REPLY = ("The angora_turkey located_in_time_zones kaliningrad_time.",
                    ["angora_turkey", "located_in_time_zone", "kaliningrad_time"])
_FACT_CLAUSE_REPLY = "The Angora Turkey is located in the Kaliningrad Time zone."


class _PoisonPill:
    """Raises if ever CALLED -- proves a code path was genuinely never reached, not merely that its return
    value went unused. Mirrors `_open_ended_qwen_fact_clause_fallback_verify._PoisonPillRenderFactSentence`."""

    def __init__(self, name):
        self._name = name

    def __call__(self, *a, **k):
        raise AssertionError(f"{self._name} must not be called on this path")


class _FakeRenderer:
    def __init__(self, reply=_RENDERER_REPLY, has_regen=False):
        self._reply = reply
        self.calls = 0
        if has_regen:
            self.render_svo_regen = self._regen

    def render_svo(self, a, v, p):
        self.calls += 1
        return self._reply

    def _regen(self, a, v, p):
        self.calls += 1
        return self._reply


class _FakeChat:
    """Exposes exactly what `_render_one_verified`/`_verify_rendered` touch -- no real ChatBrain/agent build."""

    def __init__(self, spiking_hit=None, renderer=None, verify_result=True, inner_seed=42):
        self.raw_mode = False
        self.renderer = renderer if renderer is not None else _FakeRenderer()
        self._spiking_hit = spiking_hit
        self._verify_result = verify_result
        self.verify_calls = 0
        self.inner = types.SimpleNamespace(seed=inner_seed)

    def spiking_recall_surface(self, a, v, p):
        return self._spiking_hit

    def _verify(self, surface, asserted, svo):
        self.verify_calls += 1
        return self._verify_result

    def _verify_claim_set(self, surface, gated):
        return None, None            # claim moat inactive -> _verify_rendered falls to self.chat._verify


def _composer(chat):
    """Build a `RichAnswerComposer` around a `_FakeChat` WITHOUT running `__init__` (which requires a real
    `chat.inner.composer`/`chat.is_multiturn` this test has no need to build) -- `_render_one_verified` and
    `_verify_rendered` only ever touch `self.chat`."""
    c = object.__new__(RAC.RichAnswerComposer)
    c.chat = chat
    return c


@pytest.fixture()
def clean_env(monkeypatch):
    monkeypatch.delenv("BRAIN_TOUCHPOINT_A_FACT_CLAUSE", raising=False)
    yield monkeypatch


# ---------------------------------------------------------------------------------------------------------------
# 1. Default OFF is byte-identical to pre-flag behaviour.
# ---------------------------------------------------------------------------------------------------------------
class TestFlagOffIsByteIdenticalToPreFlag:
    def test_default_unset_never_imports_or_calls_fact_clause(self, clean_env, monkeypatch):
        monkeypatch.setattr(WKV, "render_fact_sentence", _PoisonPill("render_fact_sentence"))
        chat = _FakeChat(spiking_hit=None)
        surface, verified = _composer(chat)._render_one_verified(_SVO)
        assert (surface, verified) == (_RENDERER_REPLY[0], True)
        assert chat.renderer.calls == 1

    def test_explicit_off_matches_default(self, clean_env, monkeypatch):
        clean_env.setenv("BRAIN_TOUCHPOINT_A_FACT_CLAUSE", "0")
        monkeypatch.setattr(WKV, "render_fact_sentence", _PoisonPill("render_fact_sentence"))
        chat = _FakeChat(spiking_hit=None)
        surface, verified = _composer(chat)._render_one_verified(_SVO)
        assert (surface, verified) == (_RENDERER_REPLY[0], True)


# ---------------------------------------------------------------------------------------------------------------
# 2-3. Flag ON: covered relation skips the renderer genuinely; uncovered/exception falls through unchanged.
# ---------------------------------------------------------------------------------------------------------------
class TestFlagOnRoutesThroughFactClauseFirst:
    def test_covered_relation_uses_fact_clause_and_skips_renderer_genuinely(self, clean_env, monkeypatch):
        clean_env.setenv("BRAIN_TOUCHPOINT_A_FACT_CLAUSE", "1")
        calls = []

        def _fake_render_fact_sentence(facts, seed=42):
            calls.append((list(facts), seed))
            return _FACT_CLAUSE_REPLY
        monkeypatch.setattr(WKV, "render_fact_sentence", _fake_render_fact_sentence)
        chat = _FakeChat(spiking_hit=None, renderer=_FakeRenderer())
        chat.renderer.render_svo = _PoisonPill("renderer.render_svo")  # must never fire
        surface, verified = _composer(chat)._render_one_verified(_SVO)
        assert (surface, verified) == (_FACT_CLAUSE_REPLY, True)
        assert calls == [([list(_SVO)], 42)] or calls == [([_SVO], 42)], calls

    def test_uncovered_relation_falls_through_to_renderer_unchanged(self, clean_env, monkeypatch):
        clean_env.setenv("BRAIN_TOUCHPOINT_A_FACT_CLAUSE", "1")
        monkeypatch.setattr(WKV, "render_fact_sentence", lambda facts, seed=42: None)  # not lexicon-covered
        chat_off = _FakeChat(spiking_hit=None)
        off_surface, off_verified = _composer(chat_off)._render_one_verified(_SVO)
        clean_env.setenv("BRAIN_TOUCHPOINT_A_FACT_CLAUSE", "0")
        chat_on_uncovered = _FakeChat(spiking_hit=None)
        on_surface, on_verified = _composer(chat_on_uncovered)._render_one_verified(_SVO)
        assert (off_surface, off_verified) == (on_surface, on_verified) == (_RENDERER_REPLY[0], True)

    def test_exception_inside_fact_clause_degrades_safely_to_renderer(self, clean_env, monkeypatch):
        clean_env.setenv("BRAIN_TOUCHPOINT_A_FACT_CLAUSE", "1")

        def _boom(facts, seed=42):
            raise RuntimeError("simulated fact-clause failure")
        monkeypatch.setattr(WKV, "render_fact_sentence", _boom)
        chat = _FakeChat(spiking_hit=None)
        surface, verified = _composer(chat)._render_one_verified(_SVO)
        assert (surface, verified) == (_RENDERER_REPLY[0], True), "an exception must never crash the turn"
        assert chat.renderer.calls == 1


# ---------------------------------------------------------------------------------------------------------------
# 4. Scope guard: a spiking-mouth HIT takes precedence, unchanged -- the fact-clause attempt is never reached.
# ---------------------------------------------------------------------------------------------------------------
def test_spiking_recall_surface_hit_precedence_is_unchanged(clean_env, monkeypatch):
    clean_env.setenv("BRAIN_TOUCHPOINT_A_FACT_CLAUSE", "1")
    monkeypatch.setattr(WKV, "render_fact_sentence", _PoisonPill("render_fact_sentence"))
    chat = _FakeChat(spiking_hit="the Angora Turkey follows Kaliningrad Time.")
    chat.renderer.render_svo = _PoisonPill("renderer.render_svo")
    surface, verified = _composer(chat)._render_one_verified(_SVO)
    assert (surface, verified) == ("the Angora Turkey follows Kaliningrad Time.", True)


# ---------------------------------------------------------------------------------------------------------------
# 5. SVO-identity guard: called with exactly [svo], one item -- never the broader gathered set.
# ---------------------------------------------------------------------------------------------------------------
def test_fact_clause_called_with_exactly_one_item_list_of_this_svo(clean_env, monkeypatch):
    clean_env.setenv("BRAIN_TOUCHPOINT_A_FACT_CLAUSE", "1")
    seen = []

    def _capture(facts, seed=42):
        seen.append(facts)
        return None
    monkeypatch.setattr(WKV, "render_fact_sentence", _capture)
    chat = _FakeChat(spiking_hit=None)
    _composer(chat)._render_one_verified(_SVO)
    assert len(seen) == 1
    assert len(seen[0]) == 1, "must be a ONE-item list -- never the composer's broader gathered set"
    assert tuple(seen[0][0]) == tuple(_SVO)


# ---------------------------------------------------------------------------------------------------------------
# 6. A fact-clause HIT is trusted directly (moat-safe by construction), same as the spiking-mouth precedent.
# ---------------------------------------------------------------------------------------------------------------
def test_fact_clause_hit_is_trusted_without_an_extra_verify_call(clean_env, monkeypatch):
    clean_env.setenv("BRAIN_TOUCHPOINT_A_FACT_CLAUSE", "1")
    monkeypatch.setattr(WKV, "render_fact_sentence", lambda facts, seed=42: _FACT_CLAUSE_REPLY)
    chat = _FakeChat(spiking_hit=None)
    _composer(chat)._render_one_verified(_SVO)
    assert chat.verify_calls == 0, "a fact-clause hit is moat-safe by construction, trusted like spiking_recall_surface"


# ---------------------------------------------------------------------------------------------------------------
# 7. Flag-parsing contract.
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("value,expected", [
    (None, False), ("0", False), ("false", False), ("off", False), ("no", False), ("", False),
    ("1", True), ("true", True), ("on", True), ("yes", True), ("TRUE", True), ("Yes", True),
])
def test_flag_parsing_truthy_falsy_contract(clean_env, value, expected):
    if value is None:
        clean_env.delenv("BRAIN_TOUCHPOINT_A_FACT_CLAUSE", raising=False)
    else:
        clean_env.setenv("BRAIN_TOUCHPOINT_A_FACT_CLAUSE", value)
    assert RAC._touchpoint_a_fact_clause_enabled() is expected
