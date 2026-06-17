"""Regression guard for the production MultiTurnAgent: a persistent spiking WM holds discourse referents across
turns so a pronoun resolves, a multi-hop chain can be cued by a pronoun, and the no-confab moat survives an
unresolved pronoun (empty WM -> None). De-risked GO: 2026-06-17-multiturn-anaphora-derisk-GO.md.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.multi_turn_agent import MultiTurnAgent

NOUNS = ["dog", "cat", "fish", "bird", "worm", "ball"]
VOCAB = NOUNS + ["chase", "eat"]


def _agent():
    a = MultiTurnAgent(referent_concepts=NOUNS, concepts={w: None for w in VOCAB}, seed=42)
    a.agent.composer.store("cat", "eat", "fish")     # the fact the turn-2 answer needs
    return a


def test_multiturn_anaphora_resolves():
    a = _agent()
    a.hear("dog chase cat")                          # turn 1: writes the object referent 'cat' to the WM
    assert a.what_does("it", "eat") == "fish"         # turn 2: 'it' -> cat -> (cat eat fish)


def test_pronoun_cued_multihop():
    a = _agent()
    a.agent.composer.store("fish", "eat", "worm")     # extend the chain: cat eat fish eat worm
    a.hear("dog chase cat")
    assert a.reason_chain("it", ["eat", "eat"]) == "worm"   # 'it'->cat, then cat eat fish eat worm


def test_moat_empty_wm_abstains():
    a = _agent()
    # no turn-1 referent -> empty WM -> 'it' unresolved -> abstain (no confabulated antecedent)
    assert a.what_does("it", "eat") is None
    assert a.reason_chain("it", ["eat"]) is None
