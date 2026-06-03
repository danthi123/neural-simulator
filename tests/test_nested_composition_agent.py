"""Tests for the nested-composition conversational agent on phasor FHRR.

The capability the flat-distinct 320 substrate could NOT do: store + answer facts whose slots are themselves
STRUCTURED ENTITIES (an attributed patient, "big cat"), decoded by the resonator. Validated cheap-first
(_resonator_nested_fact_probe RESOLVES); these tests pin the agent that exposes it as learn/query.
"""

from research.runners.nested_composition_agent import NestedCompositionAgent, Clause

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


def test_two_modifier_attributed_patient():
    a = _agent()
    a.learn("cat", "see", (("big", "red"), "ball"))         # patient has TWO attributes
    assert a.query_patient("cat", "see") == "big red ball"   # both adjectives + noun recovered


def test_single_and_two_modifier_distinguished_automatically():
    # the agent is NOT told how many modifiers a patient has -- the 2-factor residual selects it
    a = _agent()
    a.learn("bird", "hold", ("red", "ball"))                 # ONE modifier
    a.learn("cat", "see", (("big", "red"), "ball"))          # TWO modifiers, same noun
    assert a.query_patient("bird", "hold") == "red ball"     # one attribute
    assert a.query_patient("cat", "see") == "big red ball"   # two attributes


def test_two_modifier_canonical_vocab_order():
    # binding is commutative -> adjective ORDER is not recoverable; rendered in vocabulary order regardless of how stored
    a = _agent()
    a.learn("cat", "see", (("red", "big"), "ball"))          # stored reversed
    assert a.query_patient("cat", "see") == "big red ball"   # still canonical vocab order ("big" before "red")


def test_two_modifier_multi_seed_robust():
    # the multi-modifier (repeated-codebook) decode must be robust across seeds (restart + residual selection)
    ok = 0
    for seed in (42, 43, 44, 45, 46, 47):
        a = _agent(seed)
        a.learn("dog", "eat", (("small", "cold"), "river"))
        ok += int(a.query_patient("dog", "eat") == "small cold river")
    assert ok == 6, f"multi-modifier decoded {ok}/6 seeds"


def test_embedded_clause_patient():
    # a clause as an argument -- "dog see (cat chase bird)" -- the real syntactic recursion
    a = _agent()
    a.learn("dog", "see", Clause("cat", "chase", "bird"))
    assert a.query_patient("dog", "see") == "cat chase bird"   # the whole embedded clause recovered


def test_embedded_clause_multi_seed_robust():
    # single embedded clause with flat arguments is the robust agent capability (auto-detected, no flag)
    ok = 0
    for seed in (42, 43, 44, 45, 46, 47):
        a = _agent(seed)
        a.learn("dog", "see", Clause("cat", "chase", "bird"))
        ok += int(a.query_patient("dog", "see") == "cat chase bird")
    assert ok == 6, f"embedded clause decoded {ok}/6 seeds"


def test_clause_vs_flat_and_attributed_distinguished():
    # the agent is NOT told the kind -- a verb-presence signal marks a clause; flat/attributed have no verb
    a = _agent()
    a.learn("dog", "chase", "cat")                          # flat
    a.learn("bird", "hold", ("red", "ball"))               # attributed entity
    a.learn("dog", "see", Clause("cat", "chase", "bird"))  # embedded clause
    assert a.query_patient("dog", "chase") == "cat"        # flat stays flat (not mistaken for a clause)
    assert a.query_patient("bird", "hold") == "red ball"   # attributed stays attributed
    assert a.query_patient("dog", "see") == "cat chase bird"   # clause decoded as a clause


def test_clause_patient_abstains_on_unknown():
    a = _agent()
    a.learn("dog", "see", Clause("cat", "chase", "bird"))
    assert a.query_patient("dog", "chase") is None         # no such fact -> abstain (no confabulation)


def test_abstain_on_unknown_query():
    a = _agent()
    a.learn("dog", "chase", "cat")
    assert a.query_patient("cat", "eat") is None           # no such fact -> abstain (no confabulation)


def test_who_query():
    a = _agent()
    a.learn("dog", "chase", "cat")
    a.learn("bird", "hold", "ball")
    assert a.query_agent("chase", "cat") == "dog"          # who chases cat -> dog
    assert a.query_agent("hold", "ball") == "bird"
    assert a.query_agent("eat", "ball") is None            # no such fact -> abstain


def test_flat_vs_nested_distinguished_automatically():
    # the agent must NOT need to be told which patient is nested -- the abstention threshold detects it
    a = _agent()
    a.learn("dog", "chase", "cat")                         # flat
    a.learn("dog", "eat", ("cold", "river"))               # nested, same agent
    assert a.query_patient("dog", "chase") == "cat"        # flat decoded as a flat concept
    assert a.query_patient("dog", "eat") == "cold river"   # nested decoded via the resonator


def test_tell_about_includes_flat_and_nested():
    a = _agent()
    a.learn("dog", "chase", "cat")
    a.learn("dog", "eat", ("red", "ball"))
    facts = a.tell_about("dog")
    assert "dog chase cat" in facts                        # flat fact rendered
    assert "dog eat red ball" in facts                     # nested fact rendered (attributed patient)


def test_elaborate_brings_up_topic_facts_non_repeating():
    # the unified capability: dialogue planning (Control) over the agent's nested facts
    a = _agent()
    a.learn("dog", "chase", "cat")
    a.learn("dog", "eat", ("red", "ball"))
    a.set_topic("dog")
    e1 = a.elaborate()
    e2 = a.elaborate()
    assert {e1, e2} == {"dog chase cat", "dog eat red ball"}   # both dog facts, coherent
    assert e1 != e2                                            # non-repeating
    assert a.elaborate() is None                              # exhausted
