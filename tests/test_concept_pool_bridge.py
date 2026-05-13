"""Integration test for the concept pool architecture on CPU.

Verifies the bridge actually INITIALIZES with noun/verb pool regions
and that a few training steps run without errors. Does NOT verify
actual binding (that requires 200 events on GPU, too expensive for
unit tests).

Use SIM_BACKEND=numpy for portability + CI; no GPU required.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def concept_bridge():
    """Tiny concept-pool bridge for fast CPU tests."""
    os.environ["SIM_BACKEND"] = "numpy"
    from research.runners.concept_pool_demo import build_concept_bridge
    bridge = build_concept_bridge(
        seed=42,
        n_lang_input=64,
        n_per_pool=16,
        n_fs_per_pool=4,
        verbose=False,
    )
    return bridge


def test_bridge_has_all_12_pools(concept_bridge):
    """Bridge initialization succeeds with 12 distinct output pools.

    2026-05-13 v2: expanded from 10 to 12 (4 verb pools: GO/COME/STOP/LOOK
    instead of just GO/COME) to fix FS within-kind asymmetry that caused
    v1 seed 42 0/10 PASS via verb_pool dominance.
    """
    rm = concept_bridge.region_manager
    motor_pools = [f"motor_{a}" for a in ["N", "E", "S", "W"]]
    noun_pools = [f"noun_pool_{n}" for n in ["APPLE", "RIVER", "DOG", "CAT"]]
    verb_pools = [f"verb_pool_{v}" for v in ["GO", "COME", "STOP", "LOOK"]]

    for pool in motor_pools + noun_pools + verb_pools:
        indices = list(rm.indices(pool))
        assert len(indices) == 16, (
            f"Pool {pool} should have 16 neurons, got {len(indices)}"
        )


def test_bridge_has_language_io(concept_bridge):
    """language_input and language_output regions exist with expected size."""
    rm = concept_bridge.region_manager
    lang_in = list(rm.indices("language_input"))
    lang_out = list(rm.indices("language_output"))
    assert len(lang_in) == 64
    assert len(lang_out) == 64


def test_plasticity_gates_exist(concept_bridge):
    """All concept-pool plasticity gates can be set/get."""
    # Try setting all expected gates to 1.0 and 0.0
    expected_gates = [
        "language_input_to_motor",
        "language_input_to_noun_pool",
        "language_input_to_verb_pool",
        "motor_to_language_output",
        "noun_pool_to_language_output",
        "verb_pool_to_language_output",
    ]
    for g in expected_gates:
        # Should not raise
        concept_bridge.set_plasticity_gate(g, 1.0)
        concept_bridge.set_plasticity_gate(g, 0.0)


def test_train_word_to_pool_basic(concept_bridge):
    """train_word_to_pool runs without errors for one word."""
    from research.runners.concept_pool_demo import train_word_to_pool
    result = train_word_to_pool(
        concept_bridge, word="apple",
        target_pool_region="noun_pool_APPLE",
        n_events=2,
        stim_steps_per_event=20,
        reset_steps=10,
        n_lang_input=64,
        n_lang_output=64,
        verbose=False,
    )
    assert result["word"] == "apple"
    assert result["target"] == "noun_pool_APPLE"
    assert result["n_events"] == 2
    # Should have opened plasticity gates
    assert "language_input_to_noun_pool" in result["gates_opened"]


def test_measure_pool_firing_returns_all_pools(concept_bridge):
    """measure_pool_firing returns rate for each requested pool."""
    from research.runners.concept_pool_demo import measure_pool_firing
    all_pools = (
        [f"motor_{a}" for a in ["N", "E", "S", "W"]]
        + [f"noun_pool_{n}" for n in ["APPLE", "RIVER", "DOG", "CAT"]]
        + [f"verb_pool_{v}" for v in ["GO", "COME", "STOP", "LOOK"]]
    )
    rates = measure_pool_firing(
        concept_bridge, word="apple",
        all_pool_regions=all_pools,
        stim_steps=20, reset_steps=10,
        n_lang_input=64,
    )
    assert set(rates.keys()) == set(all_pools)
    for v in rates.values():
        assert v >= 0  # non-negative firing rate


def test_apply_concept_topographic_bias_does_not_crash(concept_bridge):
    """Topographic bias application succeeds across ALL pools.

    Post-v9 fix (2026-05-13): reciprocal pool -> language_output bias
    also applied. Summary contains:
    - 12 forward target entries (word -> target_pool, boost)
    - 132 forward off-target entries (word -> off-target pool, dampen)
    - 2 reciprocal summary entries (total target + off-target edges)
    Total 146 entries.
    """
    from research.runners.concept_pool_demo import apply_concept_topographic_bias
    summary = apply_concept_topographic_bias(
        concept_bridge,
        n_lang_input=64,
        topographic_factor=2.0,
        off_target_factor=0.5,
        apply_reciprocal=True,
        n_lang_output=64,
        verbose=False,
    )
    assert len(summary) == 146
