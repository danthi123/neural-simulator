"""Phase-1 regression tests for the consolidated core-sim composition (the conversational pipeline ON the core
SimulationBridge, not the bolted-on numpy phasor simulators). Pins the frozen bars carried from the validated
_insubstrate probes: who/what Q&A, abstention (no-confab moat), negation, and the multi-trial recovery >= 0.80.

These build a real SimulationBridge (~6400 neurons at D=800) and run spiking bind/unbind, so they are heavier than
a pure-numpy unit test; they run on the available backend (GPU when present) and skip gracefully if the substrate
concept-code cache is absent."""
import numpy as np
import pytest

from research.runners.core_sim_composition import CoreSimComposer


def _composer(seed=42, proj_dim=800):
    try:
        return CoreSimComposer(seed=seed, proj_dim=proj_dim)
    except FileNotFoundError:
        pytest.skip("denoise64 concept-code cache not present (run activity_level_integration to build it)")


def test_who_what_qa_and_abstention_on_the_bridge():
    """KB + who/what Q&A + the no-confab moat, realized in spiking on the bridge."""
    c = _composer()
    c.store("dog", "go", "north")
    c.store("cat", "come", "south")
    assert c.query_patient("dog", "go") == "north"
    assert c.query_patient("cat", "come") == "south"
    assert c.query_agent("go", "north") == "dog"
    # abstention: an in-vocabulary agent+action pair that was never stored -> None (no confabulation)
    assert c.query_patient("river", "look") is None


def test_negation_yes_no_on_the_bridge():
    """Negation via a bound POLARITY tag: affirmed -> yes, negated -> no, unstored -> unknown."""
    c = _composer()
    c.store("dog", "go", "north", polarity="AFFIRM")
    c.store("cat", "come", "south", polarity="NEGATE")
    assert c.ask_yes_no("dog", "go", "north") == "yes"
    assert c.ask_yes_no("cat", "come", "south") == "no"
    assert c.ask_yes_no("apple", "stop", "west") == "unknown"


def test_recovery_rate_clears_frozen_bar():
    """Multi-trial single-fact recovery >= 0.80 (the frozen bar), reusing one bridge across trials."""
    c = _composer()
    rng = np.random.default_rng(7)
    ok = tot = 0
    for _ in range(6):
        a, ac, p = (str(x) for x in rng.choice(c.words, size=3, replace=False))
        c.kb = []
        c.store(a, ac, p)
        ok += int(c.query_patient(a, ac) == p)
        tot += 1
    assert ok / tot >= 0.80, f"recovery {ok}/{tot} below the frozen 0.80 bar"
