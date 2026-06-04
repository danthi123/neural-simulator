"""Smoke test for the (iv) grounded multi-turn conversation demo: the agent built on real-V1-grounded codes
composes correctly (flat / one-attribute / two-attribute / embedded clause) and ABSTAINS on the unknown."""
import numpy as np

from research.runners.unified_agent_conversation_demo import _grounded_codes
from research.runners.nested_composition_agent import NestedCompositionAgent, Clause


def _agent():
    nouns = ["dog", "cat", "ball", "bird", "river", "child", "apple", "bread"]
    verbs = ["chase", "hold", "see", "eat", "want", "give"]
    adjs = ["big", "small", "red", "cold", "fast", "soft"]
    ext = _grounded_codes(nouns + verbs + adjs)
    return NestedCompositionAgent(nouns, verbs, adjs, D=2048, seed=42, external_codes=ext)


def test_grounded_conversation_composition_and_abstention():
    a = _agent()
    a.learn("dog", "chase", "cat")                                   # flat
    a.learn("child", "hold", ("red", "ball"))                        # one attribute
    a.learn("cat", "want", (("big", "red"), "ball"))                 # two attributes
    a.learn("bird", "see", Clause("cat", "chase", ("cold", "river")))  # clause (attributed inner arg)
    a.learn("dog", "see", Clause("child", "give", "bread"))          # clause (flat inner args)

    assert a.query_patient("dog", "chase") == "cat"
    assert a.query_patient("child", "hold") == "red ball"
    assert a.query_patient("cat", "want") == "big red ball"
    assert a.query_patient("bird", "see") == "cat chase cold river"
    assert a.query_patient("dog", "see") == "child give bread"
    assert a.query_agent("chase", "cat") == "dog"
    # abstention: in-vocab pairs never stored -> None (no confabulation)
    assert a.query_patient("dog", "eat") is None
    assert a.query_patient("river", "chase") is None


def test_grounded_topic_elaboration():
    a = _agent()
    a.learn("dog", "chase", "cat")
    a.learn("dog", "see", Clause("child", "give", "bread"))
    a.set_topic("dog")
    said = []
    while True:
        e = a.elaborate()
        if e is None:
            break
        said.append(e)
    assert "dog chase cat" in said
    assert "dog see child give bread" in said
