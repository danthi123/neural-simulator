"""Tests for the learned voice-invariant SVO parser (conjunctive position*voice coding).

The load-bearing property (validated in _vsa_parser_voice_probe): conjunctive position*voice coding assigns
roles VOICE-INVARIANTLY -- "dog chase cat" (active) and "cat is chased by dog" (passive) parse to the SAME
meaning {agent: dog, action: chase, patient: cat}. A position-only parser cannot do this.
"""

from research.runners.conjunctive_parser import ConjunctiveParser

VOCAB = {"dog", "cat", "child", "ball", "chase", "hold", "see", "eat"}


def test_active_svo():
    p = ConjunctiveParser()
    assert p.parse("dog chase cat", VOCAB) == {"agent": "dog", "action": "chase", "patient": "cat"}


def test_passive_is_voice_invariant():
    p = ConjunctiveParser()
    # passive form must yield the SAME roles as the active form (dog is the agent in both)
    assert p.parse("cat is chased by dog", VOCAB) == {"agent": "dog", "action": "chase", "patient": "cat"}


def test_active_and_passive_agree():
    p = ConjunctiveParser()
    active = p.parse("child hold ball", VOCAB)
    passive = p.parse("ball is held by child", VOCAB)
    assert active == passive == {"agent": "child", "action": "hold", "patient": "ball"}


def test_morphology_recovers_base_verb():
    p = ConjunctiveParser()
    # "chased"/"held"/"seen" must normalize to the base vocab verb
    assert p.parse("cat is chased by dog", VOCAB)["action"] == "chase"


def test_non_triple_returns_none():
    p = ConjunctiveParser()
    assert p.parse("dog", VOCAB) is None
    assert p.parse("hello there", VOCAB) is None


def test_detect_voice():
    from research.runners.conjunctive_parser import detect_voice
    assert detect_voice("cat is chased by dog".split()) is True
    assert detect_voice("dog chase cat".split()) is False
