"""BRAIN_OPEN_ENDED_AFFECT_CONDITIONED (default-off) — data-level checks with a fake Qwen generator (no model load).

Off: answer_turn hands the generator EXACTLY the production (system, user) and never imports the module.
prompt: only the MOOD line changes, and the live +0.16 valence is no longer rendered as the neutral dead-zone text.
"""
import sys

import pytest


class _FakeGen:
    def __init__(self):
        self.calls = []
        self.fac = None

    def generate(self, system, user, seed=42, max_new_tokens=None):
        self.calls.append((system, user, seed, max_new_tokens))
        return "fake reply.", 0.0


@pytest.fixture
def oe(monkeypatch):
    from webapp import open_ended_chat as OE
    for k in ("BRAIN_OPEN_ENDED_AFFECT_CONDITIONED", "BRAIN_AFFECT_COND_VALENCE_FS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("BRAIN_OPEN_ENDED_WKV_MOUTH", "0")
    monkeypatch.setenv("BRAIN_OPEN_ENDED_NO_QWEN_FALLBACK", "0")
    monkeypatch.setenv("BRAIN_OPEN_ENDED_FACT_CLAUSE_FALLBACK", "0")
    monkeypatch.setenv("BRAIN_OPEN_ENDED_GEN_TIME_HONESTY", "0")
    fake = _FakeGen()
    monkeypatch.setattr(OE, "get_generator", lambda _w: fake)
    monkeypatch.setattr(OE, "build_index", lambda *_a, **_k: {})
    return OE, fake


def _prod_prompt(OE, msg, valence, arousal):
    st = OE.StateContext(topic=msg.strip(), facts=[], valence=float(valence), arousal=float(arousal),
                         familiarity=0.1, confidence=0.1, novelty=0.9, curiosity=0.5 + 0.3 * 0.9,
                         self_model=OE.SELF_MODEL, affect_source="real-organ")
    return OE.build_prompt(st)


def test_off_path_hands_generator_the_production_prompt_and_never_imports(oe):
    OE, fake = oe
    sys.modules.pop("webapp.affect_conditioned_mouth", None)
    out = OE.answer_turn("Tell me about the ocean", None, 0.16, 0.4, ltm_bundle=None, brain_bundle=None)
    assert out["generator"] == "qwen"
    system, user, seed, mnt = fake.calls[-1]
    assert (system, user) == _prod_prompt(OE, "Tell me about the ocean", 0.16, 0.4)
    assert "webapp.affect_conditioned_mouth" not in sys.modules


def test_prompt_mode_changes_only_the_mood_line(oe, monkeypatch):
    OE, fake = oe
    monkeypatch.setenv("BRAIN_OPEN_ENDED_AFFECT_CONDITIONED", "prompt")
    monkeypatch.setenv("BRAIN_AFFECT_COND_VALENCE_FS", "0.34")
    OE.answer_turn("Tell me about the ocean", None, 0.16, 0.4, ltm_bundle=None, brain_bundle=None)
    system, user, _s, _m = fake.calls[-1]
    prod_sys, prod_user = _prod_prompt(OE, "Tell me about the ocean", 0.16, 0.4)
    assert user == prod_user
    changed = [(a, b) for a, b in zip(prod_sys.split("\n"), system.split("\n")) if a != b]
    assert len(changed) == 1 and changed[0][0].startswith("MOOD: ")
    assert "even and steady" in changed[0][0]          # the production dead zone at +0.16
    assert "warm" in changed[0][1] and "even and steady" not in changed[0][1]


def test_prompt_mode_lesion_is_neutral(oe, monkeypatch):
    OE, fake = oe
    monkeypatch.setenv("BRAIN_OPEN_ENDED_AFFECT_CONDITIONED", "prompt")
    OE.answer_turn("Tell me about the ocean", None, 0.0, 0.3, ltm_bundle=None, brain_bundle=None)
    system = fake.calls[-1][0]
    mood = [ln for ln in system.split("\n") if ln.startswith("MOOD: ")][0]
    assert "even and steady" in mood
