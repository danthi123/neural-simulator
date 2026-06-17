"""Production gate for the ORDER-ENCODED discourse buffer (MultiTurnAgentV2) -- multi-referent disambiguation on
the spiking resonate-and-fire phasor substrate, the production version of the CYCLE-135 de-risk
(2026-06-17-ordered-wm-position-binding-derisk.md).

The four load-bearing tests (ALL required for GO), each across 6 seeds 42 43 44 100 101 102:
  1. MULTI-REFERENT RESOLUTION  -- two referents introduced, a turn-2 bare pronoun resolves to the MOST-RECENT.
  2. ORDER-CONTROL (the wall)   -- swap which referent is introduced last -> the resolution FLIPS (slot-addressing,
                                   not a fixed answer -- the exact control the three rate-buffer negatives failed).
  3. NO-CONFAB MOAT             -- a pronoun with NO referent held (empty discourse) -> ABSTAIN (None).
  4. SINGLE-REFERENT REGRESSION -- the existing single-referent anaphora (the production MultiTurnAgent capability)
                                   still resolves -- no regression.

CPU/numpy backend (the spiking RF composer runs there; each op is a small SimulationBridge). NO `sim/` edit.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np
import pytest

from research.runners.multi_turn_agent_v2 import MultiTurnAgentV2
from research.runners.multi_turn_agent import MultiTurnAgent

SEEDS = [42, 43, 44, 100, 101, 102]
NOUNS = ["dog", "cat", "fish", "bird", "worm", "ball"]
VOCAB = NOUNS + ["chase", "eat", "see"]


def _v2(seed):
    """A V2 agent with the order-encoded discourse buffer + a small food-web so the Q&A path has facts to read."""
    a = MultiTurnAgentV2(referent_concepts=NOUNS, concepts={w: None for w in VOCAB}, seed=seed)
    c = a.agent.composer
    # who-eats-what (the patient each referent eats), used by the multi-referent resolution check.
    for ag, ob in [("cat", "fish"), ("dog", "worm"), ("fish", "worm"), ("bird", "ball"), ("hawk_skip", "x")][:4]:
        c.store(ag, "eat", ob)
    return a


# ---------------------------------------------------------------------------
# Test 1: multi-referent resolution -- the capability.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_multireferent_resolution(seed):
    """Two referents introduced ("the dog saw the cat") then a turn-2 bare pronoun ("it") -> resolves to the
    most-recent referent (the cat), and the Q&A reads the cat's fact."""
    a = _v2(seed)
    a.hear("dog see cat")                       # surface order: dog (slot0), cat (slot1=most-recent)
    assert a.most_recent_referent() == "cat", f"seed {seed}: pronoun should foreground cat"
    assert a.what_does("it", "eat") == "fish", f"seed {seed}: 'it' (cat) eats fish"


# ---------------------------------------------------------------------------
# Test 2: ORDER-CONTROL -- the load-bearing flip the rate buffer could not do.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_order_control_flips(seed):
    """Swap which referent is introduced LAST -> the resolution must FLIP to the new most-recent one. Natural:
    [dog, cat] -> 'it'=cat. Order-control: [cat, dog] -> 'it'=dog. Proves slot-addressing (which slot you read),
    not a fixed answer (the exact failure mode of recency / salience-boost / biased-competition-WTA)."""
    nat = _v2(seed)
    nat.hear("dog see cat")                     # cat most-recent
    rec_nat = nat.most_recent_referent()

    flip = _v2(seed)
    flip.hear("cat see dog")                    # dog now most-recent (order swapped)
    rec_flip = flip.most_recent_referent()

    assert rec_nat == "cat", f"seed {seed}: natural -> cat, got {rec_nat}"
    assert rec_flip == "dog", f"seed {seed}: order-control -> dog (FLIPPED), got {rec_flip}"
    assert rec_nat != rec_flip, f"seed {seed}: resolution must flip with discourse order"
    # And the downstream Q&A flips too (cat eats fish; dog eats worm).
    assert nat.what_does("it", "eat") == "fish", f"seed {seed}: natural 'it'(cat) eats fish"
    assert flip.what_does("it", "eat") == "worm", f"seed {seed}: flipped 'it'(dog) eats worm"


# ---------------------------------------------------------------------------
# Test 3: NO-CONFAB MOAT -- empty discourse -> abstain.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_moat_empty_discourse_abstains(seed):
    """A pronoun with NO referent held (empty discourse) -> ABSTAIN (None), not a confabulated referent. The
    abstention is FREE from the composer's familiarity gate (reading an unoccupied buffer grounds nothing)."""
    a = _v2(seed)                               # no hear() -> empty discourse buffer
    assert a.most_recent_referent() is None, f"seed {seed}: empty discourse -> no referent"
    assert a.what_does("it", "eat") is None, f"seed {seed}: unresolved pronoun -> abstain (no fact invented)"
    assert a.reason_chain("it", ["eat"]) is None, f"seed {seed}: unresolved cue -> abstain"
    assert a.is_it_true("it", "eat", "fish") == "unknown", f"seed {seed}: unresolved -> unknown"


# ---------------------------------------------------------------------------
# Test 4: SINGLE-REFERENT REGRESSION -- V2 is a strict superset of V1.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_single_referent_regression(seed):
    """The single-referent anaphora capability (the production MultiTurnAgent) still resolves under V2: one held
    referent -> the most-recent slot IS that referent -> 'it' resolves to it."""
    a = MultiTurnAgentV2(referent_concepts=NOUNS, concepts={w: None for w in VOCAB}, seed=seed)
    a.agent.composer.store("cat", "eat", "fish")
    # (a) genuinely single referent: introduce ONLY 'cat' -> the most-recent (and only) slot holds cat -> 'it'=cat.
    a._write_referent("cat")
    assert a.most_recent_referent() == "cat", f"seed {seed}: single referent -> cat"
    assert a.what_does("it", "eat") == "fish", f"seed {seed}: single-referent anaphora 'it'(cat) eats fish"
    # (b) the exact V1 production contract (hear a full sentence -> pronoun resolves to the salient object).
    a2 = MultiTurnAgentV2(referent_concepts=NOUNS, concepts={w: None for w in VOCAB}, seed=seed)
    a2.agent.composer.store("cat", "eat", "fish")
    a2.hear("dog chase cat")                    # object 'cat' most-recent -- the V1 single-antecedent contract
    assert a2.what_does("it", "eat") == "fish", f"seed {seed}: V1-contract anaphora 'it'(cat) eats fish"
    a = a2
    # And a pronoun-cued multi-hop still works (cat eat fish eat worm).
    a.agent.composer.store("fish", "eat", "worm")
    a.hear("dog chase cat")
    assert a.reason_chain("it", ["eat", "eat"]) == "worm", f"seed {seed}: pronoun-cued 2-hop chain -> worm"


# ---------------------------------------------------------------------------
# Cross-check: the production MultiTurnAgent (V1, rate buffer) still imports + its single-referent path works
# (guards that promoting the module + adding V2 did not perturb V1).
# ---------------------------------------------------------------------------
def test_v1_single_referent_unbroken():
    a = MultiTurnAgent(referent_concepts=NOUNS, concepts={w: None for w in VOCAB}, seed=42)
    a.agent.composer.store("cat", "eat", "fish")
    a.hear("dog chase cat")
    assert a.what_does("it", "eat") == "fish"
    # V1's moat (empty WM -> abstain) still holds.
    b = MultiTurnAgent(referent_concepts=NOUNS, concepts={w: None for w in VOCAB}, seed=42)
    b.agent.composer.store("cat", "eat", "fish")
    assert b.what_does("it", "eat") is None


# ---------------------------------------------------------------------------
# Code-parity guard: the WM's concept codes MUST equal the agent composer's (so a slot read is a real concept).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_wm_code_parity_with_composer(seed):
    a = MultiTurnAgentV2(referent_concepts=NOUNS, concepts={w: None for w in VOCAB}, seed=seed)
    comp = a.agent.composer
    assert all(np.allclose(a.wm.concepts[w], comp.concepts[w]) for w in comp.words), \
        f"seed {seed}: discourse-WM concept codes must match the composer's"
    # The calibrated familiarity threshold must sit in a sane band (the principled separation midpoint), not the
    # de-risk's marginal frozen 0.15.
    assert 0.15 < a.wm.match_threshold < 0.55, f"seed {seed}: calibrated threshold {a.wm.match_threshold}"
