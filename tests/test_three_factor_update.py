"""Tests for the three-factor eligibility/weight-update primitive.

The pure function `update_eligibility_and_weights` is the GPU-port hot
path: when called with cupy arrays, the entire update happens on GPU
with no host round-trip. These tests verify the numpy-backend correctness;
the cupy-backend path is byte-identical algorithmically (xp swap only).

GPU equivalence is validated by running the same test with cupy arrays
when GPU is free — see `tests/test_three_factor_gpu.py` (added separately
once GPU work isn't competing with the running chain).
"""
from __future__ import annotations

import numpy as np
import pytest


def _make_state(n_edges=12, n_neurons=20, seed=0):
    """Synthetic state for a tiny three-factor scenario.

    Builds:
      4 motor pools × 3 motor neurons each = 12 motor neurons
      8 language input neurons
      12 lang->motor edges (3 per motor pool)
    """
    rng = np.random.default_rng(seed)
    # Motor pools: neuron IDs 0..11, partitioned 0-2=N, 3-5=E, 6-8=S, 9-11=W
    # Lang input: neuron IDs 12..19
    # Edges: each motor neuron gets 1 edge from a random lang neuron
    edge_src = rng.integers(12, 20, size=n_edges).astype(np.int32)
    edge_dst = np.arange(n_edges, dtype=np.int32)  # 0..11 = motor IDs
    edge_off = np.arange(n_edges, dtype=np.int64)  # CSR offsets
    # 0=N, 1=E, 2=S, 3=W (3 edges each, since 3 motor neurons per pool)
    edge_action = (np.arange(n_edges, dtype=np.int8) // 3)

    eligibility = np.zeros(n_edges, dtype=np.float32)
    weights_data = rng.uniform(0.0, 5.0, size=n_edges).astype(np.float32)

    return {
        "n_edges": n_edges, "n_neurons": n_neurons,
        "edge_src": edge_src, "edge_dst": edge_dst,
        "edge_off": edge_off, "edge_action": edge_action,
        "eligibility": eligibility, "weights_data": weights_data,
    }


def test_eligibility_decays_per_step():
    """When no neurons fire, eligibility just decays exponentially."""
    from research.runners.bio_three_factor import update_eligibility_and_weights
    s = _make_state()
    s["eligibility"][:] = 1.0  # all eligible

    lang_active = np.zeros(s["n_neurons"], dtype=bool)
    post_active = np.zeros(s["n_neurons"], dtype=bool)
    da = np.zeros(4, dtype=np.float32)  # no DA → no weight update

    update_eligibility_and_weights(
        s["eligibility"], s["weights_data"],
        s["edge_src"], s["edge_dst"], s["edge_off"], s["edge_action"],
        lang_active, post_active, da,
        decay_per_step=0.9, learning_rate=1e-3,
        weight_min=0.0, weight_max=5.0, xp=np,
    )

    # All eligibility should have decayed by 0.9
    assert np.allclose(s["eligibility"], 0.9, atol=1e-6)


def test_pre_post_coincidence_increments_eligibility():
    """When pre and post both fire, eligibility for that edge gets +1."""
    from research.runners.bio_three_factor import update_eligibility_and_weights
    s = _make_state()

    # Edge 0 is from src=edge_src[0] to dst=0 (motor_N). Make both active.
    lang_active = np.zeros(s["n_neurons"], dtype=bool)
    lang_active[s["edge_src"][0]] = True
    post_active = np.zeros(s["n_neurons"], dtype=bool)
    post_active[s["edge_dst"][0]] = True
    da = np.zeros(4, dtype=np.float32)

    update_eligibility_and_weights(
        s["eligibility"], s["weights_data"],
        s["edge_src"], s["edge_dst"], s["edge_off"], s["edge_action"],
        lang_active, post_active, da,
        decay_per_step=1.0,  # no decay so we can isolate the +1
        learning_rate=1e-3, weight_min=0.0, weight_max=5.0, xp=np,
    )

    # Edge 0: pre×post coincidence → eligibility = 1.0
    assert s["eligibility"][0] == pytest.approx(1.0)
    # All other edges where src is the SAME and dst is also active should
    # also have eligibility 1.0 (vectorized fancy indexing applies to all)
    same_src_active = (s["edge_src"] == s["edge_src"][0])
    same_dst_active = (s["edge_dst"] == s["edge_dst"][0])
    coincident = same_src_active & same_dst_active
    assert np.all(s["eligibility"][coincident] == 1.0)
    # Edges with no coincidence stay 0
    assert np.all(s["eligibility"][~coincident] == 0.0)


def test_weight_update_only_when_da_and_eligibility_both_nonzero():
    """Weight changes only for edges where eligibility>0 AND da[action]!=0."""
    from research.runners.bio_three_factor import update_eligibility_and_weights
    s = _make_state(seed=42)
    initial_weights = s["weights_data"].copy()

    # Set ALL edges to eligibility=1.0
    s["eligibility"][:] = 1.0
    # DA only for action 0 (motor_N) — edges 0,1,2 should update
    da = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    lang_active = np.zeros(s["n_neurons"], dtype=bool)  # not used by this test path
    post_active = np.zeros(s["n_neurons"], dtype=bool)

    update_eligibility_and_weights(
        s["eligibility"], s["weights_data"],
        s["edge_src"], s["edge_dst"], s["edge_off"], s["edge_action"],
        lang_active, post_active, da,
        decay_per_step=1.0, learning_rate=1.0,
        weight_min=0.0, weight_max=10.0, xp=np,
    )

    # Edges with action=0 (N): weights changed by +1 × eligibility × DA = 1.0
    motor_n_edges = (s["edge_action"] == 0)
    assert np.all(s["weights_data"][motor_n_edges] >
                  initial_weights[motor_n_edges] - 1e-6)
    # Other edges unchanged
    other_edges = ~motor_n_edges
    assert np.allclose(s["weights_data"][other_edges],
                       initial_weights[other_edges])


def test_weight_clipping_to_bounds():
    """Updated weights are clipped to [weight_min, weight_max]."""
    from research.runners.bio_three_factor import update_eligibility_and_weights
    s = _make_state(seed=7)
    s["weights_data"][:] = 4.5  # near max
    s["eligibility"][:] = 100.0  # huge eligibility
    da = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)  # update all

    lang_active = np.zeros(s["n_neurons"], dtype=bool)
    post_active = np.zeros(s["n_neurons"], dtype=bool)

    update_eligibility_and_weights(
        s["eligibility"], s["weights_data"],
        s["edge_src"], s["edge_dst"], s["edge_off"], s["edge_action"],
        lang_active, post_active, da,
        decay_per_step=1.0, learning_rate=1.0,
        weight_min=0.0, weight_max=5.0, xp=np,
    )

    # All weights clipped to weight_max=5.0
    assert np.all(s["weights_data"] <= 5.0)
    assert np.all(s["weights_data"] >= 0.0)
    # And specifically pinned to max since 4.5 + 100*1 >> 5
    assert np.all(s["weights_data"] == 5.0)


def test_negative_da_decreases_weights():
    """When DA is -1 (false-positive penalty), weights go down."""
    from research.runners.bio_three_factor import update_eligibility_and_weights
    s = _make_state(seed=99)
    s["weights_data"][:] = 3.0
    s["eligibility"][:] = 1.0
    da = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    lang_active = np.zeros(s["n_neurons"], dtype=bool)
    post_active = np.zeros(s["n_neurons"], dtype=bool)

    update_eligibility_and_weights(
        s["eligibility"], s["weights_data"],
        s["edge_src"], s["edge_dst"], s["edge_off"], s["edge_action"],
        lang_active, post_active, da,
        decay_per_step=1.0, learning_rate=0.5,
        weight_min=0.0, weight_max=5.0, xp=np,
    )

    # Action 0 edges: 3.0 + 1.0 * (-1.0) * 0.5 = 2.5
    motor_n_edges = (s["edge_action"] == 0)
    assert np.allclose(s["weights_data"][motor_n_edges], 2.5)
    # Others stay at 3.0
    other_edges = ~motor_n_edges
    assert np.allclose(s["weights_data"][other_edges], 3.0)


def test_combined_decay_and_coincidence():
    """Decay applies BEFORE coincidence accumulation."""
    from research.runners.bio_three_factor import update_eligibility_and_weights
    s = _make_state()
    s["eligibility"][:] = 2.0  # start at 2.0

    # Edge 0 is coincident
    lang_active = np.zeros(s["n_neurons"], dtype=bool)
    lang_active[s["edge_src"][0]] = True
    post_active = np.zeros(s["n_neurons"], dtype=bool)
    post_active[s["edge_dst"][0]] = True
    da = np.zeros(4, dtype=np.float32)

    update_eligibility_and_weights(
        s["eligibility"], s["weights_data"],
        s["edge_src"], s["edge_dst"], s["edge_off"], s["edge_action"],
        lang_active, post_active, da,
        decay_per_step=0.5, learning_rate=1e-3,
        weight_min=0.0, weight_max=5.0, xp=np,
    )

    # Edge 0: 2.0 * 0.5 = 1.0, then +1.0 (coincidence) = 2.0
    assert s["eligibility"][0] == pytest.approx(2.0)
    # Other edges of same src/dst pair: same logic
    # Edges with no coincidence: 2.0 * 0.5 = 1.0
    same_pair = (s["edge_src"] == s["edge_src"][0]) & (s["edge_dst"] == s["edge_dst"][0])
    assert np.all(s["eligibility"][same_pair] == 2.0)
    assert np.all(s["eligibility"][~same_pair] == 1.0)


def test_function_handles_empty_active_masks():
    """When no neurons fire, eligibility just decays — no errors."""
    from research.runners.bio_three_factor import update_eligibility_and_weights
    s = _make_state()
    s["eligibility"][:] = 1.0
    initial_weights = s["weights_data"].copy()

    lang_active = np.zeros(s["n_neurons"], dtype=bool)
    post_active = np.zeros(s["n_neurons"], dtype=bool)
    da = np.zeros(4, dtype=np.float32)

    update_eligibility_and_weights(
        s["eligibility"], s["weights_data"],
        s["edge_src"], s["edge_dst"], s["edge_off"], s["edge_action"],
        lang_active, post_active, da,
        decay_per_step=0.9, learning_rate=1e-3,
        weight_min=0.0, weight_max=5.0, xp=np,
    )

    # Eligibility decayed; weights unchanged (no DA)
    assert np.allclose(s["eligibility"], 0.9)
    assert np.allclose(s["weights_data"], initial_weights)
