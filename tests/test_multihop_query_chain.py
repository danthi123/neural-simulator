"""Regression guard for multi-hop relational reasoning (query_chain / reason_chain) on the production composer.

De-risked GO 3 seeds x 3 D (2026-06-17-multihop-query-chain-GO.md). This pins the production method's load-bearing
behaviour: a 2-hop relational chain returns the correct terminal concept, AND the no-confab moat holds at every
hop (a broken or over-run or unstored chain abstains, never confabulates).
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.rf_phasor_composer import RFPhasorComposer
from research.runners.brain_conversational_agent import BrainConversationalAgent

VOCAB = ["dog", "cat", "mouse", "bug", "lion", "deer", "grass", "eat", "play", "ball"]


def _composer():
    c = RFPhasorComposer(seed=42, D=128, vocab=VOCAB)
    # a relational chain dog -eat-> cat -eat-> mouse -eat-> bug, plus a distractor (different relation).
    c.store("dog", "eat", "cat")
    c.store("cat", "eat", "mouse")
    c.store("mouse", "eat", "bug")
    c.store("cat", "play", "ball")     # distractor: pollutes co-occurrence, NOT the eat-relation
    return c


def test_two_and_three_hop_chase_correct():
    c = _composer()
    assert c.query_chain("dog", ["eat"]) == "cat"            # 1-hop (direct)
    assert c.query_chain("dog", ["eat", "eat"]) == "mouse"   # 2-hop (held-out composition)
    assert c.query_chain("dog", ["eat", "eat", "eat"]) == "bug"  # 3-hop


def test_mixed_relation_chain():
    # the chain can follow DIFFERENT relations per hop: dog -eat-> cat -play-> ball.
    c = _composer()
    assert c.query_chain("dog", ["eat", "play"]) == "ball"


def test_moat_holds_at_every_hop():
    c = _composer()
    # unstored cue (ball is never an agent) -> abstain
    assert c.query_chain("ball", ["eat", "eat"]) is None
    # over-run the chain end (bug has no eat-fact) -> abstain, no confabulation
    assert c.query_chain("dog", ["eat", "eat", "eat", "eat"]) is None
    # a first hop with no matching relation (dog has no 'play' fact) -> abstain immediately
    assert c.query_chain("dog", ["play"]) is None
    # a mid-chain dead end (mouse has no 'play' fact) -> abstain mid-chain
    assert c.query_chain("dog", ["eat", "eat", "play"]) is None


def test_agent_surface_delegates():
    agent = BrainConversationalAgent(seed=42, concepts={w: None for w in VOCAB})
    agent.composer.store("dog", "eat", "cat")
    agent.composer.store("cat", "eat", "mouse")
    assert agent.reason_chain("dog", ["eat", "eat"]) == "mouse"
    assert agent.reason_chain("ball", ["eat"]) is None        # moat preserved on the agent surface
