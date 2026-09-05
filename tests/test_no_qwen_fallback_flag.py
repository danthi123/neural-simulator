"""Regression pin for the one-brain Stage-1 Qwen-fallback retirement (2026-09-04,
research/findings/2026-09-04-onebrain-stage1-qwen-fallback-retire-GO.md), `BRAIN_OPEN_ENDED_NO_QWEN_FALLBACK`
in `webapp/open_ended_chat.py::answer_turn`/`no_qwen_fallback_enabled`.

THE MEASURED CONTEXT (see the finding for the full real-traffic numbers). After the 2026-09-04 linattn flip's
`BRAIN_WKV_MOUTH_SCOPE=broad` default, the literal one-shot Qwen fallback in `answer_turn`'s final `else`
branch already fires on 0/15 forked turns of the project's own 16-probe battery (was 9/15 = 60% pre-flip,
research/findings/2026-09-04-per-touchpoint-qwen-call-share.md) -- the WKV mouth's own free generation now
covers every known=False turn that battery samples. This flag closes the narrower residual: a known=False
turn that STILL reaches the literal Qwen branch (WKV mouth disabled, reverted to narrow vocab scope, or a
genuine exception -- `not wkv_attempted`) skips the `gen.generate(...)` forward pass entirely instead of
paying for a reply `post_filter`'s unknown-topic branch discards anyway (every sampled known=False reply,
Qwen or wkv_mouth, converges to the SAME fixed honest-abstain string regardless of what was generated).

THIS FILE PINS, end-to-end through the REAL `answer_turn` (a `_FakeQwenGenerator` stands in for the heavy
off-bridge Qwen-0.5B load, mirroring `tests/test_generator_trace_matches_producer.py`'s own isolation
strategy -- these tests never need CUDA or a live model):
  1. Flag OFF (default, unset): a known=False turn that cannot reach the WKV mouth (out-of-vocab) or the
     fact-clause fallback (no facts at all) still calls the fake Qwen exactly once, traced "qwen" -- the
     pre-existing path, BYTE-IDENTICAL to before this flag existed.
  2. Flag ON: the IDENTICAL turn calls the fake Qwen ZERO times (the forward pass is genuinely skipped, not
     just its output discarded), traces "no_qwen_fallback", and produces the EXACT SAME `answer` text as the
     flag-off run -- the moat's fixed honest-abstain string, because the fake reply's own text contains no
     hedge language and gets reduced to it anyway (see `_base_post_filter`'s unknown-topic branch). This is
     the load-bearing claim: retiring the call is compute-free, not answer-changing.
  3. Scope guard: a known=True turn that reaches this SAME literal-Qwen branch (the WKV mouth AND the
     fact-clause fallback both declined -- an out-of-vocab topic whose relation is NOT lexicon-covered) is
     UNCHANGED by the flag -- still calls Qwen, still traces "qwen". The flag never touches known=True.
  4. `no_qwen_fallback_enabled()` itself: the standard truthy/falsy env-parsing contract this repo's other
     flags use (unset/0/false/off/no -> False; 1/true/on/yes, case-insensitive -> True).

Run: CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy .venv/bin/python -m pytest tests/test_no_qwen_fallback_flag.py -q
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import pytest

from webapp import open_ended_chat as OEC
from webapp import wkv_mouth_generator as WKV

pytestmark = pytest.mark.filterwarnings("ignore")

# Mirrors tests/test_generator_trace_matches_producer.py's own fixtures -- duplicated rather than imported so
# this file stays independently readable/runnable, matching that file's own "mirroring the isolation strategy"
# convention (it duplicated _open_ended_qwen_fact_clause_fallback_verify.py's _FakeQwenGenerator rather than
# importing it). Values must stay in sync with that file's own `test_routing_preconditions`.
_OOV_TOPIC = "isaac_asimov"
_OOV_KNOWN_MSG = f"tell me about {_OOV_TOPIC}"
_OOV_UNKNOWN_MSG = "what do you think about quantum chromodynamics superconductivity"
_UNCOVERED_ACTION = "zzz_totally_uncovered_relation_xyz"


@pytest.fixture()
def clean_env(monkeypatch):
    for k in ("BRAIN_WKV_MOUTH_SCOPE", "BRAIN_OPEN_ENDED_WKV_MOUTH", "BRAIN_OPEN_ENDED_FACT_CLAUSE_FALLBACK",
              "BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_SENTENCE", "BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_GROUND",
              "BRAIN_OPEN_ENDED_GEN_TIME_HONESTY", "BRAIN_OPEN_ENDED_NO_QWEN_FALLBACK"):
        monkeypatch.delenv(k, raising=False)
    # pinned to the ssm family for the SAME reason test_generator_trace_matches_producer.py pins it: the
    # OOV/in-vocab fixtures above are calibrated against the ssm/V=1000 word-level checkpoint's vocabulary,
    # not the linattn family's general BPE vocabulary (where scope defaults to 'broad' and admits everything).
    monkeypatch.setenv("BRAIN_WKV_MOUTH_RECURRENCE", "ssm")
    yield monkeypatch


def _make_bundle(tmp_path, facts: list[dict]) -> str:
    d = tmp_path / "bundle"
    d.mkdir(exist_ok=True)
    (d / "facts.json").write_text(json.dumps({"schema_version": 1, "facts": facts}), encoding="utf-8")
    return str(d)


class _FakeQwenGenerator:
    """Drop-in replacement for `OpenEndedGenerator` -- records call COUNT (not just content), so a test can
    assert the forward pass was genuinely skipped, not merely that its output was discarded downstream."""

    def __init__(self):
        self.calls = 0

    def generate(self, system, user, seed=42, max_new_tokens=None):
        self.calls += 1
        return "FAKE QWEN REPLY -- this generator must never be reached by the fact-clause/WKV paths", 0.01


@pytest.fixture()
def fake_qwen(monkeypatch):
    fake = _FakeQwenGenerator()
    monkeypatch.setattr(OEC, "get_generator", lambda warm_faculty: fake)
    return fake


# ---------------------------------------------------------------------------------------------------------------
# Sanity: the routing precondition this whole file depends on (mirrors the sibling file's own precondition test).
# ---------------------------------------------------------------------------------------------------------------
def test_routing_precondition(clean_env):
    assert WKV.in_vocab_scope(_OOV_UNKNOWN_MSG, seed=42) is False


# ---------------------------------------------------------------------------------------------------------------
# 1-2: flag OFF is byte-identical to before it existed; flag ON skips the call but not the answer text.
# ---------------------------------------------------------------------------------------------------------------
class TestFlagTogglesTheForwardPassNotTheAnswer:
    def test_flag_off_default_still_calls_qwen_once(self, clean_env, fake_qwen):
        """Default (unset) -- the pre-existing path, unaffected by this flag's mere existence."""
        res = OEC.answer_turn(_OOV_UNKNOWN_MSG, None, valence=0.0, arousal=0.5, ltm_bundle=None,
                              brain_bundle=None, seed=42, max_new_tokens=25)
        assert res["known"] is False
        assert res["generator"] == "qwen"
        assert fake_qwen.calls == 1
        assert res["raw"].startswith("FAKE QWEN REPLY")

    def test_flag_explicit_off_matches_default(self, clean_env, fake_qwen):
        clean_env.setenv("BRAIN_OPEN_ENDED_NO_QWEN_FALLBACK", "0")
        res = OEC.answer_turn(_OOV_UNKNOWN_MSG, None, valence=0.0, arousal=0.5, ltm_bundle=None,
                              brain_bundle=None, seed=42, max_new_tokens=25)
        assert res["generator"] == "qwen"
        assert fake_qwen.calls == 1

    def test_flag_on_skips_the_call_and_traces_no_qwen_fallback(self, clean_env, fake_qwen):
        clean_env.setenv("BRAIN_OPEN_ENDED_NO_QWEN_FALLBACK", "1")
        res = OEC.answer_turn(_OOV_UNKNOWN_MSG, None, valence=0.0, arousal=0.5, ltm_bundle=None,
                              brain_bundle=None, seed=42, max_new_tokens=25)
        assert res["known"] is False
        assert res["generator"] == "no_qwen_fallback"
        assert fake_qwen.calls == 0, "the Qwen forward pass must be genuinely SKIPPED, not merely discarded"
        assert res["raw"] == ""
        assert res["wkv_mouth_used"] is False
        assert res["fact_clause_used"] is False

    def test_flag_on_produces_the_identical_answer_text_as_flag_off(self, clean_env):
        """The load-bearing equivalence claim: retiring the call must not change what the user sees. Two
        independent answer_turn calls (fresh fake generator each, since fake_qwen is per-test) -- one per
        flag state -- must agree on `answer` (the post_filter-visible surface) byte-for-byte."""
        off_fake = _FakeQwenGenerator()
        orig = OEC.get_generator
        try:
            OEC.get_generator = lambda warm_faculty: off_fake
            res_off = OEC.answer_turn(_OOV_UNKNOWN_MSG, None, valence=0.0, arousal=0.5, ltm_bundle=None,
                                      brain_bundle=None, seed=42, max_new_tokens=25)
        finally:
            OEC.get_generator = orig
        assert res_off["generator"] == "qwen" and off_fake.calls == 1

        on_fake = _FakeQwenGenerator()
        clean_env.setenv("BRAIN_OPEN_ENDED_NO_QWEN_FALLBACK", "1")
        try:
            OEC.get_generator = lambda warm_faculty: on_fake
            res_on = OEC.answer_turn(_OOV_UNKNOWN_MSG, None, valence=0.0, arousal=0.5, ltm_bundle=None,
                                     brain_bundle=None, seed=42, max_new_tokens=25)
        finally:
            OEC.get_generator = orig
        assert res_on["generator"] == "no_qwen_fallback" and on_fake.calls == 0

        assert res_on["answer"] == res_off["answer"], (
            f"retirement must be answer-preserving: on={res_on['answer']!r} off={res_off['answer']!r}"
        )
        assert res_on["known"] == res_off["known"] is False


# ---------------------------------------------------------------------------------------------------------------
# 3: scope guard -- a known=True turn on this same branch is never touched by the flag.
# ---------------------------------------------------------------------------------------------------------------
def test_known_topic_on_the_same_branch_is_unaffected_by_the_flag(clean_env, fake_qwen, tmp_path):
    """An out-of-vocab known topic whose relation the fact-clause lexicon does NOT cover reaches the literal
    Qwen branch exactly like an unknown topic would -- but this flag is scoped to known=False only, so it
    must NOT intercept this turn even when ON."""
    assert WKV.in_vocab_scope(_OOV_KNOWN_MSG, seed=42) is False
    bundle = _make_bundle(tmp_path, [{"agent": _OOV_TOPIC, "action": _UNCOVERED_ACTION,
                                      "patient": "something", "polarity": "AFFIRM"}])
    clean_env.setenv("BRAIN_OPEN_ENDED_NO_QWEN_FALLBACK", "1")
    res = OEC.answer_turn(_OOV_KNOWN_MSG, None, valence=0.0, arousal=0.5, ltm_bundle=bundle,
                          brain_bundle=None, seed=42, max_new_tokens=25)
    assert res["known"] is True
    assert res["generator"] == "qwen", res
    assert fake_qwen.calls == 1
    assert res["wkv_mouth_used"] is False
    assert res["fact_clause_used"] is False


# ---------------------------------------------------------------------------------------------------------------
# 4: the flag-parsing contract itself, isolated.
# ---------------------------------------------------------------------------------------------------------------
class TestFlagParsing:
    @pytest.mark.parametrize("val", ["0", "false", "False", "off", "no", "", "garbage"])
    def test_falsy_values(self, monkeypatch, val):
        monkeypatch.setenv("BRAIN_OPEN_ENDED_NO_QWEN_FALLBACK", val)
        assert OEC.no_qwen_fallback_enabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "True", "on", "ON", "yes", "Yes"])
    def test_truthy_values(self, monkeypatch, val):
        monkeypatch.setenv("BRAIN_OPEN_ENDED_NO_QWEN_FALLBACK", val)
        assert OEC.no_qwen_fallback_enabled() is True

    def test_unset_is_falsy(self, monkeypatch):
        monkeypatch.delenv("BRAIN_OPEN_ENDED_NO_QWEN_FALLBACK", raising=False)
        assert OEC.no_qwen_fallback_enabled() is False
