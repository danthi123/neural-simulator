"""Regression guard for Tier 2.2 SELF-CUED associative chain-of-thought on the production composer.

De-risked GO numpy 3 seeds x 3 D (2026-06-27-tier2.2-chain-of-thought-GO.md). The ONE difference vs query_chain:
the agent SELECTS each next relation by LEARNED association strength over its own stored facts, NOT a caller-
supplied action list. This pins the load-bearing behaviour + the anti-cheats the 2026-05-14 transitive-inference
RETRACTION demands: the self-cued chain reaches the held-out multi-hop target, LESION-the-association collapses
it, the moat holds at every hop, and a 3-4 hop chain does not compound error.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np

from research.runners.rf_phasor_composer import RFPhasorComposer
from research.runners.brain_conversational_agent import BrainConversationalAgent

VOCAB = ["dog", "cat", "mouse", "bug", "leaf", "lion", "deer", "eat", "see", "ball"]


def _composer(seed=42, D=128):
    """A relational chain dog -eat-> cat -eat-> mouse -eat-> bug -eat-> leaf, with `eat` REINFORCED (stored 3x)
    above a distractor `see` relation (1x) so the SELF-CUED selector PICKS `eat` by learned strength. The
    distractor pollutes the relation-blind co-occurrence (so a spreading baseline can't chase) but not the eat
    relation."""
    c = RFPhasorComposer(seed=seed, D=D, vocab=VOCAB)
    chain = ["dog", "cat", "mouse", "bug", "leaf"]
    for a, p in zip(chain[:-1], chain[1:]):
        for _ in range(3):                 # reinforce the chain relation -> the selector prefers it
            c.store(a, "eat", p)
        c.store(a, "see", "ball")          # a weaker distractor relation
    return c


def test_self_cued_chain_reaches_held_out_target():
    # NO caller-supplied relation list -- the agent SELECTS `eat` at each hop by learned association.
    c = _composer()
    term, path = c.chain_of_thought("dog", goal="leaf", max_hops=4, return_path=True)
    assert term == "leaf", f"self-cued chain did not reach the held-out 4-hop target: {path}"
    assert path == ["dog", "cat", "mouse", "bug", "leaf"], path   # the agent chose every hop
    # the selected relation at each start IS the reinforced chain relation
    assoc = c._relation_assoc()
    assert c._select_next_relation("dog", assoc) == "eat"


def test_self_cued_two_hop_with_goal_stop():
    c = _composer()
    assert c.chain_of_thought("dog", goal="mouse", max_hops=4) == "mouse"   # stops at the goal (2 hops)


def test_lesion_the_association_collapses_the_chain():
    # ZERO the learned association the selector reads -> it has no signal -> abstains (the chain dies). This is the
    # load-bearing anti-cheat (proves the LEARNED association, not co-occurrence smearing, drives the chain).
    c = _composer()
    assert c.chain_of_thought("dog", goal="leaf", max_hops=4, lesion="zero") is None


def test_moat_holds_at_every_hop():
    c = _composer()
    assert c.chain_of_thought("ball", max_hops=3) is None          # unstored start (ball is never an agent)
    # past the chain end: leaf has no out-fact -> the chain stops at leaf and does not fabricate a further hop
    term, path = c.chain_of_thought("dog", max_hops=8, return_path=True)
    assert term == "leaf" and path[-1] == "leaf", path
    assert "ball" not in path[1:]                                  # never wandered onto the distractor object


def test_no_error_compounding_to_four_hops():
    # the cleanup re-discretizes between hops, so a deeper chain does not degrade: every prefix target is correct.
    c = _composer()
    for k, want in [(1, "cat"), (2, "mouse"), (3, "bug"), (4, "leaf")]:
        assert c.chain_of_thought("dog", goal=want, max_hops=k) == want, f"{k}-hop self-cued target wrong"


def test_permuted_graph_collapses():
    # scramble which patient binds each (agent, eat) -> the selector still picks `eat`, but the role-structured
    # chase lands on the WRONG patient -> the held-out 2-hop target is (almost surely) not reached.
    c = RFPhasorComposer(seed=42, D=128, vocab=VOCAB)
    chain = ["dog", "cat", "mouse", "bug", "leaf"]
    edges = list(zip(chain[:-1], chain[1:]))
    patients = [p for _, p in edges]
    np.random.default_rng(123).shuffle(patients)
    for (a, _), p in zip(edges, patients):
        for _ in range(3):
            c.store(a, "eat", p)
    # with a deranged mapping the 2-hop self-cued chase should miss the true 2-hop target 'mouse'
    assert c.chain_of_thought("dog", goal="mouse", max_hops=2) != "mouse"


def test_agent_surface_delegates():
    agent = BrainConversationalAgent(seed=42, concepts={w: None for w in VOCAB})
    for a, p in zip(["dog", "cat", "mouse"], ["cat", "mouse", "bug"]):
        for _ in range(3):
            agent.composer.store(a, "eat", p)
        agent.composer.store(a, "see", "ball")
    assert agent.chain_of_thought("dog", goal="bug", max_hops=3) == "bug"
    assert agent.chain_of_thought("ball", max_hops=2) is None      # moat preserved on the agent surface
