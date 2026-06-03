"""Tests for the nested-composition conversational agent on phasor FHRR.

The capability the flat-distinct 320 substrate could NOT do: store + answer facts whose slots are themselves
STRUCTURED ENTITIES (an attributed patient, "big cat"), decoded by the resonator. Validated cheap-first
(_resonator_nested_fact_probe RESOLVES); these tests pin the agent that exposes it as learn/query.
"""

from research.runners.nested_composition_agent import NestedCompositionAgent

NOUNS = ["dog", "cat", "ball", "bird", "river"]
VERBS = ["chase", "hold", "see", "eat"]
ADJS = ["big", "small", "red", "cold"]


def _agent(seed=42):
    return NestedCompositionAgent(NOUNS, VERBS, ADJS, seed=seed)


def test_flat_fact_round_trip():
    a = _agent()
    a.learn("dog", "chase", "cat")
    assert a.query_patient("dog", "chase") == "cat"        # flat patient


def test_nested_fact_attributed_patient():
    a = _agent()
    a.learn("dog", "chase", ("big", "cat"))                # patient is an attributed entity
    assert a.query_patient("dog", "chase") == "big cat"    # recovered as BOTH adjective and noun


def test_mixed_flat_and_nested_facts():
    a = _agent()
    a.learn("dog", "chase", "cat")                         # flat
    a.learn("bird", "hold", ("red", "ball"))               # nested
    assert a.query_patient("dog", "chase") == "cat"
    assert a.query_patient("bird", "hold") == "red ball"


def test_nested_recovers_both_components():
    a = _agent()
    a.learn("dog", "see", ("small", "river"))
    assert a.query_patient("dog", "see") == "small river"


def test_abstain_on_unknown_query():
    a = _agent()
    a.learn("dog", "chase", "cat")
    assert a.query_patient("cat", "eat") is None           # no such fact -> abstain (no confabulation)


def test_flat_vs_nested_distinguished_automatically():
    # the agent must NOT need to be told which patient is nested -- the abstention threshold detects it
    a = _agent()
    a.learn("dog", "chase", "cat")                         # flat
    a.learn("dog", "eat", ("cold", "river"))               # nested, same agent
    assert a.query_patient("dog", "chase") == "cat"        # flat decoded as a flat concept
    assert a.query_patient("dog", "eat") == "cold river"   # nested decoded via the resonator
