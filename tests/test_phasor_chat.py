"""Tests for PhasorChat -- the conversational agent on the phasor learned-code substrate.

Pins the conversational behavior: learn facts (flat / attributed / two-attribute / embedded clause) from
typed statements, answer what/who questions, abstain on the untold, reject unknown words. Cognition runs on
STDP-learned codes; the parser is the simple language front-end.
"""

from research.runners.phasor_chat import PhasorChat

NOUNS = ["dog", "cat", "ball", "river", "bird", "child"]
VERBS = ["chase", "see", "eat", "hold", "want"]
ADJS = ["big", "red", "cold", "small"]


def _chat(seed=42):
    return PhasorChat(NOUNS, VERBS, ADJS, seed=seed)


def test_flat_fact_and_query():
    c = _chat()
    assert c.say("dog chase cat") == "ok"
    assert c.say("what does dog chase") == "cat"


def test_attributed_entity():
    c = _chat()
    c.say("bird see cold river")
    assert c.say("what does bird see") == "cold river"


def test_two_attribute_entity():
    c = _chat()
    c.say("child hold big red ball")
    assert c.say("what does child hold") == "big red ball"


def test_embedded_clause():
    c = _chat()
    c.say("dog eat cat chase ball")
    assert c.say("what does dog eat") == "cat chase ball"


def test_who_question():
    c = _chat()
    c.say("dog chase cat")
    assert c.say("who chase cat") == "dog"


def test_abstain_on_untold_fact():
    c = _chat()
    c.say("dog chase cat")
    assert "don't know" in c.say("what does cat want")          # never told -> abstain


def test_reject_unknown_word():
    c = _chat()
    assert "don't know the word" in c.say("dog chase zebra")     # zebra not in vocab


def test_multi_seed_core_facts():
    ok = 0
    for seed in (42, 43, 44):
        c = _chat(seed)
        c.say("dog chase cat")
        c.say("bird see cold river")
        ok += int(c.say("what does dog chase") == "cat" and c.say("what does bird see") == "cold river")
    assert ok == 3
