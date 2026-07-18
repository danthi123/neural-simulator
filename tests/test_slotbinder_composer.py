"""CI guard for the gap-#2 SlotBinderComposer (2026-07-17): the fully-spiking competitive-slot binder wired into
the conversational pipeline as a selectable composer.

Covers (1) the composer contract directly (store/query both directions, negation via the 4th polarity slot, the
no-confab moat, describe, multi-hop chain) and (2) the full BrainConversationalAgent path through it
(comprehend -> store -> who/what/yes-no/describe/abstain). CPU/numpy, fast (small vocab, ~a few s).

The binder itself is 6-seed GO + adversarially confirmed (no-teach->chance, scramble-teach->0.00) in
research/findings/2026-07-17-*keystone-2-*.md + *gap2-adversarial-verify-*.md. These tests pin the WIRE-IN.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")   # CPU, fast + deterministic for CI

import pytest

from research.runners.slotbinder_composer import SlotBinderComposer
from research.runners.brain_conversational_agent import BrainConversationalAgent

_VOCAB = ["dog", "cat", "river", "apple", "go", "come", "stop", "look", "north", "south", "east", "west",
          "chase", "eat", "see", "fish", "bird"]


def _composer(seed=42):
    return SlotBinderComposer(seed=seed, vocab=list(_VOCAB), max_facts=8)


def test_composer_contract_store_query_both_directions():
    c = _composer()
    c.store("dog", "chase", "cat")
    c.store("bird", "see", "fish")
    assert c.query_patient("dog", "chase") == "cat"      # who -> what
    assert c.query_agent("see", "fish") == "bird"        # what -> who
    assert c.render_fact("dog") == "dog chase cat"       # describe
    assert c.query_chain("dog", ["chase"]) == "cat"      # multi-hop (1 hop)


def test_composer_no_confab_moat():
    c = _composer()
    c.store("dog", "chase", "cat")
    assert c.query_patient("river", "look") is None       # never-stored cue -> abstain
    assert c.query_patient("dog", "eat") is None           # real agent + WRONG verb -> abstain (content-addressed)
    assert c.ask_yes_no("apple", "stop", "west") == "unknown"
    assert c.render_fact("river") is None


def test_composer_negation_polarity_slot():
    c = _composer()
    c.store("dog", "chase", "cat", polarity="AFFIRM")
    c.store("cat", "eat", "fish", polarity="NEGATE")
    assert c.ask_yes_no("dog", "chase", "cat") == "yes"
    assert c.ask_yes_no("cat", "eat", "fish") == "no"      # the 4th (polarity) slot reads NEGATE
    assert c.ask_yes_no("dog", "chase", "fish") == "unknown"  # patient mismatch -> unknown (moat)


@pytest.mark.parametrize("seed", [42, 43])
def test_composer_multiseed_robust(seed):
    c = _composer(seed)
    facts = [("dog", "chase", "cat"), ("cat", "eat", "fish"), ("bird", "see", "dog")]
    for a, v, p in facts:
        c.store(a, v, p)
    assert all(c.query_patient(a, v) == p for a, v, p in facts)
    assert all(c.query_agent(v, p) == a for a, v, p in facts)
    assert c.query_patient("fish", "hear") is None          # moat


def test_agent_path_who_what_moat_negation():
    """The whole BrainConversationalAgent loop runs through the SlotBinderComposer: comprehend a sentence ->
    store on the slot-binder -> answer who/what/yes-no/describe + abstain. This is the gap-#2 wire-in."""
    comp = SlotBinderComposer(seed=42, vocab=list(_VOCAB), max_facts=8)
    try:
        ag = BrainConversationalAgent(seed=42, composer=comp)
    except FileNotFoundError:
        pytest.skip("agent parser cache not present")
    ag.hear("dog go north", polarity="AFFIRM")
    ag.hear("cat come south", polarity="NEGATE")
    ag.hear("river stop west")
    assert ag.what_does("dog", "go") == "north"
    assert ag.what_does("cat", "come") == "south"
    assert ag.who_does("stop", "west") == "river"
    assert ag.what_does("apple", "look") is None            # no-confab moat
    assert ag.is_it_true("dog", "go", "north") == "yes"
    assert ag.is_it_true("cat", "come", "south") == "no"    # negated
    assert ag.is_it_true("apple", "stop", "west") == "unknown"
    assert ag.describe("river") == "river stop west"
    assert ag.describe("apple") is None                     # moat
