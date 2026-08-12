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


def test_render_hypothesis_fluent_flagged_guess_stub(chat):
    """A generated HYPOTHESIS renders as a FLUENT sentence framed as a guess, VERIFIED to carry the SAME (a,v,p).
    On the stub the fluent surface is deterministic + SVO-verified, so `fluent` is True and it is NOT the raw
    template. `render()` dispatches HypothesisSVO here."""
    from research.runners.brain_chat_tui import HypothesisSVO
    hyp = HypothesisSVO(["dog", "like", "fish"])   # each word known; the TRIPLE is novel (not stored)
    surface, fluent = chat.render_hypothesis_verified(hyp)
    assert fluent is True
    assert surface.lower().startswith("maybe ")
    assert "guess" in surface.lower()
    assert "not something i was taught" in surface.lower()
    assert not surface.lower().startswith("perhaps ")     # the fluent path, not the raw template
    # content-swap guard: `fluent is True` means the internal VERIFY re-parse confirmed the surface carries the
    # hypothesis's exact (a,v,p); sanity-check the subject + object appear in the fluent prose.
    assert "dog" in surface and "fish" in surface
    # render() dispatches a HypothesisSVO to the flagged-guess renderer
    assert chat.render(hyp) == surface


def test_render_hypothesis_template_fallback_without_mouth():
    """With NO fluent renderer (raw mode / --no-renderer) the guess falls back to the raw FLAGGED template
    (byte-identical to the pre-fluent surface), NEVER an asserted fact."""
    _imports()
    from research.runners.brain_chat_tui import (
        ChatBrain, DEFAULT_SELF_ALIASES, HypothesisSVO,
    )
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
    c = ChatBrain(agent, self_aliases=DEFAULT_SELF_ALIASES, renderer=None)   # raw: no mouth
    hyp = HypothesisSVO(["dog", "like", "fish"])
    surface, fluent = c.render_hypothesis_verified(hyp)
    assert fluent is False
    assert surface == "perhaps dog like fish  [a guess from what I've learned -- not something I was taught]"


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
