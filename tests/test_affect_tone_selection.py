"""BRAIN_OPEN_ENDED_AFFECT_TONE_SELECT (default-off) — data-level checks with a fake Qwen generator and a fake organ
evaluator (no model load, no brain build).

Off: answer_turn hands the generator EXACTLY the production (system, user), makes ONE call, never imports the module,
and returns byte-identically to the pinned pre-change module (cc0ded327).
On: the draft prompt is affect-free; restyles are identical across held valences; the lock rejects content drift;
the released candidate follows the held valence; held valence 0 releases the most neutral candidate.
"""
import importlib.util
import subprocess
import sys
import types

import pytest

PINNED = "cc0ded3270d6e7e35d076b5672e1e6db31de541d"

RESTYLES = {
    "very warm, joyful and cheerful": "Oh, what a joy: the ocean is a vast body of water covering 71 percent of Earth!",
    "slightly warmer and friendlier": "The ocean is a lovely vast body of water covering 71 percent of Earth.",
    "slightly more subdued and wistful": "The ocean is a vast, quiet body of water covering 71 percent of Earth.",
    "very sad, somber and melancholy": "Sadly, the ocean is a vast body of water covering 72 percent of Mars.",
}
DRAFT = "The ocean is a vast body of water covering 71 percent of Earth."
# fake organ: valence by style marker words
VAL = {"joy": 0.3, "lovely": 0.15, "quiet": -0.1, "Sadly": -0.3}


class _FakeGen:
    def __init__(self):
        self.calls = []
        self.fac = None

    def generate(self, system, user, seed=42, max_new_tokens=None):
        self.calls.append((system, user, seed, max_new_tokens))
        if user.startswith("Rewrite this reply in a "):
            desc = user[len("Rewrite this reply in a "):].split(" tone:")[0]
            return RESTYLES[desc], 0.0
        return DRAFT, 0.0


def _fake_eval(text):
    v = 0.0
    for k, x in VAL.items():
        if k in text:
            v = x
    return {"appraisal": v, "n_hits": int(v != 0), "differential": v / 4.0, "valence": v}


@pytest.fixture
def oe(monkeypatch):
    from webapp import open_ended_chat as OE
    for k in ("BRAIN_OPEN_ENDED_AFFECT_CONDITIONED", "BRAIN_OPEN_ENDED_AFFECT_TONE_SELECT"):
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


def test_off_path_single_production_call_and_never_imports(oe):
    OE, fake = oe
    sys.modules.pop("webapp.affect_tone_selection", None)
    OE.answer_turn("Tell me about the ocean", None, 0.16, 0.4, ltm_bundle=None, brain_bundle=None)
    assert len(fake.calls) == 1
    assert fake.calls[-1][:2] == _prod_prompt(OE, "Tell me about the ocean", 0.16, 0.4)
    assert "webapp.affect_tone_selection" not in sys.modules


def test_off_path_byte_identical_to_pinned_pre_change_module(oe, monkeypatch):
    OE, fake = oe
    src = subprocess.run(["git", "show", "%s:webapp/open_ended_chat.py" % PINNED], capture_output=True, text=True,
                         check=True).stdout
    mod = types.ModuleType("_pinned_open_ended_chat")
    mod.__file__ = OE.__file__
    exec(compile(src, "pinned_open_ended_chat.py", "exec"), mod.__dict__)
    monkeypatch.setattr(mod, "get_generator", lambda _w: fake)
    monkeypatch.setattr(mod, "build_index", lambda *_a, **_k: {})
    for msg, v in (("Tell me about the ocean", 0.16), ("Describe the city at night", -0.08), ("hi", 0.0)):
        a = OE.answer_turn(msg, None, v, 0.4, ltm_bundle=None, brain_bundle=None)
        b = mod.answer_turn(msg, None, v, 0.4, ltm_bundle=None, brain_bundle=None)
        assert a == b
        assert fake.calls[-1] == fake.calls[-2]


def _on(oe, monkeypatch):
    OE, fake = oe
    monkeypatch.setenv("BRAIN_OPEN_ENDED_AFFECT_TONE_SELECT", "1")
    from webapp import affect_tone_selection as ATS
    monkeypatch.setattr(ATS, "EVALUATOR", _fake_eval)
    return OE, fake, ATS


def test_draft_prompt_is_affect_free_and_restyles_do_not_see_the_mood(oe, monkeypatch):
    OE, fake, ATS = _on(oe, monkeypatch)
    OE.answer_turn("Tell me about the ocean", None, 0.16, 0.4, ltm_bundle=None, brain_bundle=None)
    calls_a = list(fake.calls)
    fake.calls.clear()
    OE.answer_turn("Tell me about the ocean", None, -0.3, 0.9, ltm_bundle=None, brain_bundle=None)
    calls_b = list(fake.calls)
    assert len(calls_a) == 1 + len(ATS.STYLES)
    assert calls_a == calls_b                      # nothing the mouth is asked depends on the held affect
    mood = [ln for ln in calls_a[0][0].split("\n") if ln.startswith("MOOD: ")]
    assert mood == [ATS.NEUTRAL_MOOD_LINE]


def test_lock_rejects_changed_number_and_new_name(oe, monkeypatch):
    OE, fake, ATS = _on(oe, monkeypatch)
    ok, det = ATS.content_lock(DRAFT, RESTYLES["very sad, somber and melancholy"])
    assert not ok and det["missing_numbers"] == ["71"] and "mar" in det["new_names"]
    ok, _ = ATS.content_lock(DRAFT, RESTYLES["slightly warmer and friendlier"])
    assert ok


def test_selection_follows_held_valence(oe, monkeypatch):
    OE, fake, ATS = _on(oe, monkeypatch)
    out = OE.answer_turn("Tell me about the ocean", None, 0.16, 0.4, ltm_bundle=None, brain_bundle=None)
    assert out["raw"] == RESTYLES["slightly warmer and friendlier"] and ATS.LAST_TRACE["selected_style"] == "pos1"
    out = OE.answer_turn("Tell me about the ocean", None, 0.4, 0.4, ltm_bundle=None, brain_bundle=None)
    assert ATS.LAST_TRACE["selected_style"] == "pos2"
    out = OE.answer_turn("Tell me about the ocean", None, -0.3, 0.4, ltm_bundle=None, brain_bundle=None)
    # the strongly negative restyle broke the lock (72 / Mars), so the most negative ADMISSIBLE one is released
    assert ATS.LAST_TRACE["selected_style"] == "neg1"
    assert 4 not in ATS.LAST_TRACE["admissible"]


def test_lesion_releases_most_neutral_and_ties_go_to_draft(oe, monkeypatch):
    OE, fake, ATS = _on(oe, monkeypatch)
    out = OE.answer_turn("Tell me about the ocean", None, 0.0, 0.3, ltm_bundle=None, brain_bundle=None)
    assert out["raw"] == DRAFT and ATS.LAST_TRACE["selected"] == 0
    assert ATS.select_by_mood([0.1, -0.1, 0.1], 0.0) == 0
