"""OPEN-ENDED GENERATION (#3E) rendered as a FLUENT, clearly-FLAGGED guess.

The DEFAULT ``/api/brain-chat`` turn runs the fluent RichAnswerComposer path. When the brain VOLUNTEERS a
generated novel proposition (a ``HypothesisSVO`` from ``ChatBrain.gate``), it must be spoken as FLUENT prose via
the mouth BUT framed as an explicit guess ("Maybe ... -- that's a guess ..."), SVO-VERIFIED so the mouth cannot
swap the content, with the raw template ("perhaps a v p") as the fallback. It must NEVER be chained/elaborated
into asserted stored facts (the pre-fix leak), and an unknown/ungrounded topic must still ABSTAIN (the no-confab
moat). These tests run GPU-FREE on the deterministic template-stub renderer (numpy backend); the live Qwen
confirmation of the fluent SURFACE needs a CUDA launch and is covered by the webapp verify harness.
"""
import os

import pytest


def _imports():
    os.environ["SIM_BACKEND"] = "numpy"
    os.environ.setdefault("BRAIN_COMPOSER_KIND", "rf")
    try:
        from research.runners.brain_chat_tui import (  # noqa: F401
            ChatBrain, StubRenderer, DEFAULT_SELF_ALIASES, HypothesisSVO,
        )
        from research.runners.multi_turn_agent import MultiTurnAgent  # noqa: F401
        from research.runners.rich_answer_composer import RichAnswerComposer  # noqa: F401
    except Exception as e:  # pragma: no cover - host without the conversational stack
        pytest.skip(f"conversational stack not importable here: {e}")


# A small INTERLINKED graph -- dense enough that the #3E generator can find plausible NOVEL triples (the 5-fact
# tiny-demo is too sparse and only ever abstains, so it cannot exercise the fluent-guess render).
_FACTS = [
    ("dog", "chase", "cat"), ("cat", "chase", "mouse"), ("dog", "eat", "bone"),
    ("cat", "eat", "fish"), ("mouse", "eat", "cheese"), ("bird", "eat", "worm"),
    ("dog", "like", "bone"), ("cat", "like", "fish"), ("bird", "chase", "worm"),
    ("dog", "chase", "bird"), ("cat", "chase", "bird"),
]


@pytest.fixture(scope="module")
def chat():
    _imports()
    from research.runners.brain_chat_tui import ChatBrain, StubRenderer, DEFAULT_SELF_ALIASES
    from research.runners.multi_turn_agent import MultiTurnAgent
    vocab = sorted({w for f in _FACTS for w in f})
    actions = {v for _a, v, _p in _FACTS}
    referents = [w for w in vocab if w not in actions]
    agent = MultiTurnAgent(referent_concepts=referents, concepts={w: None for w in vocab}, seed=42,
                           enable_neural_render=False, composer_kind="rf",
                           enable_biased_competition=False, defer_planner=True, event_register=None)
    inner = getattr(agent, "agent", agent)
    for a, v, p in _FACTS:
        inner.hear(f"{a} {v} {p}", polarity="AFFIRM")
    return ChatBrain(agent, self_aliases=DEFAULT_SELF_ALIASES, renderer=StubRenderer())


def _fresh_rich(chat):
    from research.runners.rich_answer_composer import RichAnswerComposer
    return RichAnswerComposer(chat, max_sentences=4, neural_planner=False, planner_seed=42)


_NOT_TAUGHT = "[a guess from what i've learned -- not something i was taught]"


def test_render_hypothesis_fluent_flagged_guess_stub(chat):
    """DEFAULT (brain-native SPIKING Broca mouth, ON unless BRAIN_SPIKING_MOUTH=0): a generated HYPOTHESIS renders
    as a grammatical sentence produced on spikes ('perhaps the <S> <V-3sg> the <O>'), flagged as a guess, VERIFIED to
    carry the SAME (a,v,p). The spiking Broca carries its own epistemic hedge 'perhaps' (its CONN slot) and the shared
    not-taught disclaimer, so it is NOT the raw template ('perhaps dog like fish') and `fluent` is True.
    `render()` dispatches HypothesisSVO here. (This test originally pinned the host 'Maybe ...' frame; that is now
    the BRAIN_SPIKING_MOUTH=0 escape, pinned in the next test.)"""
    from research.runners.brain_chat_tui import HypothesisSVO
    hyp = HypothesisSVO(["dog", "like", "fish"])   # each word known; the TRIPLE is novel (not stored)
    surface, fluent = chat.render_hypothesis_verified(hyp)
    assert fluent is True
    low = surface.lower()
    assert low.startswith("perhaps the dog likes the fish")   # spiking Broca: determiners + 3sg verb = fluent, not raw
    assert "guess" in low
    assert "not something i was taught" in low
    assert not low.startswith("perhaps dog like fish")        # not the raw template (no determiners / 3sg verb)
    assert not low.startswith("maybe ")                       # not the pre-spiking host frame
    # content-swap guard: `fluent is True` means the internal VERIFY re-parse confirmed the surface carries the
    # hypothesis's exact (a,v,p); sanity-check the subject + object appear in the fluent prose.
    assert "dog" in surface and "fish" in surface
    # render() dispatches a HypothesisSVO to the flagged-guess renderer
    assert chat.render(hyp) == surface


def test_render_hypothesis_prespiking_mouth_escape_maybe_frame(chat, monkeypatch):
    """ESCAPE (BRAIN_SPIKING_MOUTH=0): the pre-spiking mouth renders the guess FLUENTLY via the renderer (template-stub
    here), framed 'Maybe ... -- that's a guess ...', SVO-verified, `fluent` True, and NOT the raw template. This is the
    original (pre-spiking) contract of this test, kept alive behind the documented escape flag."""
    from research.runners.brain_chat_tui import HypothesisSVO
    monkeypatch.setenv("BRAIN_SPIKING_MOUTH", "0")
    hyp = HypothesisSVO(["dog", "like", "fish"])
    surface, fluent = chat.render_hypothesis_verified(hyp)
    assert fluent is True
    assert surface.lower().startswith("maybe ")
    assert "guess" in surface.lower()
    assert "not something i was taught" in surface.lower()
    assert not surface.lower().startswith("perhaps ")     # the fluent path, not the raw template
    assert "dog" in surface and "fish" in surface


def _raw_chat():
    """A ChatBrain with NO fluent renderer (raw mode / --no-renderer) over the same small graph."""
    _imports()
    from research.runners.brain_chat_tui import ChatBrain, DEFAULT_SELF_ALIASES
    from research.runners.multi_turn_agent import MultiTurnAgent
    vocab = sorted({w for f in _FACTS for w in f})
    actions = {v for _a, v, _p in _FACTS}
    referents = [w for w in vocab if w not in actions]
    agent = MultiTurnAgent(referent_concepts=referents, concepts={w: None for w in vocab}, seed=42,
                           enable_neural_render=False, composer_kind="rf",
                           enable_biased_competition=False, defer_planner=True, event_register=None)
    inner = getattr(agent, "agent", agent)
    for a, v, p in _FACTS:
        inner.hear(f"{a} {v} {p}", polarity="AFFIRM")
    return ChatBrain(agent, self_aliases=DEFAULT_SELF_ALIASES, renderer=None)   # raw: no mouth


def test_render_hypothesis_template_fallback_without_mouth(monkeypatch):
    """With NO fluent renderer (raw mode / --no-renderer) AND the pre-spiking path (BRAIN_SPIKING_MOUTH=0) the guess
    falls back to the raw FLAGGED template (byte-identical to the pre-fluent surface), NEVER an asserted fact."""
    from research.runners.brain_chat_tui import HypothesisSVO
    monkeypatch.setenv("BRAIN_SPIKING_MOUTH", "0")
    c = _raw_chat()
    hyp = HypothesisSVO(["dog", "like", "fish"])
    surface, fluent = c.render_hypothesis_verified(hyp)
    assert fluent is False
    assert surface == "perhaps dog like fish  [a guess from what I've learned -- not something I was taught]"


def test_render_hypothesis_spiking_mouth_needs_no_renderer():
    """DEFAULT: the spiking Broca is transformer-FREE and runs BEFORE the renderer-None / raw-mode check, so a
    renderer-less ChatBrain still speaks a grammatical, verified, flagged guess ON SPIKES (fluent True) rather than the
    raw template. The raw template is reached only via BRAIN_SPIKING_MOUTH=0 (above) or un-frameable content
    (multi-word roles), pinned below."""
    from research.runners.brain_chat_tui import HypothesisSVO
    c = _raw_chat()
    surface, fluent = c.render_hypothesis_verified(HypothesisSVO(["dog", "like", "fish"]))
    assert fluent is True
    assert surface.lower().startswith("perhaps the dog likes the fish")
    assert surface.lower().endswith(_NOT_TAUGHT)
    # content the spiking Broca cannot frame (multi-word role) falls through to the raw flagged template, no leak
    surface2, fluent2 = c.render_hypothesis_verified(["dog", "like", "big fish"])
    assert fluent2 is False
    assert surface2 == "perhaps dog like big fish  [a guess from what I've learned -- not something I was taught]"


def test_rich_open_ended_flagged_guess_no_leak(chat):
    """The RICH (default) path on an open-ended prompt returns a SINGLE clearly-FLAGGED guess -- NEVER an
    unflagged asserted fact, and never chains/elaborates the guess into stored recall (the pre-fix leak). Over a
    battery of open-ended prompts: 0 leaks, and at least one real guess is produced (the path is exercised)."""
    prompts = ["what might dog do", "what might cat eat", "guess about dog", "guess",
               "what might mouse eat", "what might bird chase", "tell me something new about dog",
               "what else about cat"]
    leaks, guesses = 0, 0
    for q in prompts:
        rc = _fresh_rich(chat)
        r = rc.answer(q)
        if r["abstained"]:
            continue
        flagged = ("guess" in r["answer"].lower()) or ("perhaps" in r["answer"].lower())
        if not flagged:
            leaks += 1
            continue
        if r.get("hypothesis"):
            guesses += 1
            assert r["n_sentences"] == 1
            assert r["fluent_hypothesis"] is True         # the stub verifies -> fluent surface
            svo = r["hypothesis_svo"]
            assert isinstance(svo, list) and len(svo) == 3
            # the guess's (a,v,p) is NOT a stored fact (it is genuinely novel)
            assert tuple(svo) not in {tuple(f) for f in _FACTS}
            # content-swap guard: `fluent_hypothesis is True` means the internal VERIFY re-parse already confirmed
            # the fluent sentence carries exactly `svo`; sanity-check the subject + object surface in the prose.
            assert svo[0] in r["answer"] and svo[2] in r["answer"]
    assert leaks == 0, f"{leaks} open-ended answers leaked as unflagged asserted facts"
    assert guesses >= 1, "no hypothesis was generated -- the fluent-guess path was not exercised"


def test_rich_open_ended_moat_abstains_on_unknown(chat):
    """An open-ended prompt about an UNKNOWN/ungrounded subject the brain never heard of must ABSTAIN -- the brain
    does not invent about what it has never learned (the no-confab moat)."""
    for q in ["what might dragon do", "tell me something new about unicorn",
              "guess about wizard", "what might xyzzy do"]:
        rc = _fresh_rich(chat)
        r = rc.answer(q)
        assert r["abstained"] is True, f"{q!r} should abstain (unknown subject), got {r['answer']!r}"
        assert not r.get("hypothesis")


def test_rich_recall_and_abstain_unregressed(chat):
    """The RICH RECALL path is unchanged: a taught cue answers with a multi-sentence grounded reply; an untaught
    cue still ABSTAINS. (Guards that the hypothesis interception did not disturb the recall/abstain paths.)"""
    rc = _fresh_rich(chat)
    r = rc.answer("what does dog chase")
    assert r["abstained"] is False
    assert r["n_sentences"] >= 1
    assert not r.get("hypothesis")
    assert "dog" in r["answer"].lower() and "cat" in r["answer"].lower()
    rc2 = _fresh_rich(chat)
    r2 = rc2.answer("what does the dragon breathe")
    assert r2["abstained"] is True
    assert "don't know" in r2["answer"].lower()
