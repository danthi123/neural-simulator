"""Tests for Cluster B.1 — D1/D2 plasticity asymmetry."""
from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_bg_bridge(enable_d1_d2: bool):
    """Build a small BG bridge using the runner's region builder."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import build_bg_brain_regions
    from sim import (
        SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig,
    )

    regions, pathways = build_bg_brain_regions(n_cortex=20)  # small for speed
    cfg = CoreSimConfig(
        num_neurons=1,  # placeholder; region_manager will override
        enable_brain_region_framework=True,
        brain_regions=regions,
        region_pathways=pathways,
        enable_d1_d2_asymmetry=enable_d1_d2,
    )
    # Cortex→D1 weights are weight_mean=25 with Gaussian jitter sigma=0.2,
    # so initial values can hit ~40+ in the tail. Set bounds well above
    # that so clipping doesn't dominate the small reward delta in tests
    # exercising the reward-modulated update path. See CLAUDE.md
    # "STDP bounds gotcha".
    cfg.stdp_w_max = 100.0
    cfg.hebbian_max_weight = 100.0
    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


def test_d1_d2_sign_array_allocated_when_enabled():
    """cp_d1_d2_sign exists with shape (nnz,) and dtype float32 when enable=True."""
    pytest.importorskip("cupy")
    import cupy as cp
    bridge = _build_bg_bridge(enable_d1_d2=True)
    assert hasattr(bridge, "cp_d1_d2_sign") and bridge.cp_d1_d2_sign is not None
    nnz = int(bridge.cp_connections.nnz)
    assert bridge.cp_d1_d2_sign.shape == (nnz,)
    assert bridge.cp_d1_d2_sign.dtype == cp.float32
    bridge.clear_simulation_state_and_gpu_memory()


def test_d1_d2_sign_default_off():
    """When flag off, cp_d1_d2_sign is None — flagship is bit-identical."""
    pytest.importorskip("cupy")
    bridge = _build_bg_bridge(enable_d1_d2=False)
    assert getattr(bridge, "cp_d1_d2_sign", None) is None
    bridge.clear_simulation_state_and_gpu_memory()


def test_d1_targeted_synapses_have_sign_plus_one():
    """Synapses whose post-region is str_D1_* get sign=+1."""
    pytest.importorskip("cupy")
    bridge = _build_bg_bridge(enable_d1_d2=True)
    # Find synapses targeting str_D1_N region. region_manager exposes indices
    # for each named region.
    d1_n_neurons = bridge.region_manager.indices("str_D1_N")
    # cp_connections is CSR; for each synapse, post-neuron is at row index.
    # sparse storage: data[i] is the weight, indices[i] is the column = post,
    # indptr defines row boundaries (= pre-neuron). For each i, the post is
    # cp_connections.indices[i].
    import cupy as cp
    post = bridge.cp_connections.indices
    d1_n_set = cp.asarray(list(d1_n_neurons), dtype=cp.int64)
    # Mask: which synapses post into str_D1_N
    mask = cp.isin(post, d1_n_set)
    if int(mask.sum()) == 0:
        pytest.skip("No synapses target str_D1_N in this build (probably no D1 inputs)")
    signs_at_d1 = bridge.cp_d1_d2_sign[mask]
    assert (signs_at_d1 == 1.0).all(), \
        f"D1-targeted synapses must have sign=+1, got {signs_at_d1}"
    bridge.clear_simulation_state_and_gpu_memory()


def test_d2_targeted_synapses_have_sign_minus_one():
    """Synapses whose post-region is str_D2_* get sign=-1."""
    pytest.importorskip("cupy")
    bridge = _build_bg_bridge(enable_d1_d2=True)
    d2_n_neurons = bridge.region_manager.indices("str_D2_N")
    import cupy as cp
    post = bridge.cp_connections.indices
    d2_n_set = cp.asarray(list(d2_n_neurons), dtype=cp.int64)
    mask = cp.isin(post, d2_n_set)
    if int(mask.sum()) == 0:
        pytest.skip("No synapses target str_D2_N in this build")
    signs_at_d2 = bridge.cp_d1_d2_sign[mask]
    assert (signs_at_d2 == -1.0).all(), \
        f"D2-targeted synapses must have sign=-1, got {signs_at_d2}"
    bridge.clear_simulation_state_and_gpu_memory()


def test_non_d1_d2_synapses_have_sign_plus_one():
    """Synapses NOT targeting D1 or D2 (e.g. cortex→cortex, gpe→gpi) keep sign=+1.
    This ensures the rest of the network is unaffected by the asymmetry."""
    pytest.importorskip("cupy")
    bridge = _build_bg_bridge(enable_d1_d2=True)
    import cupy as cp
    post = bridge.cp_connections.indices
    # Compute the union of all D1+D2 neuron indices
    d1_d2_neurons = []
    for action in ("N", "E", "S", "W"):
        d1_d2_neurons.extend(bridge.region_manager.indices(f"str_D1_{action}"))
        d1_d2_neurons.extend(bridge.region_manager.indices(f"str_D2_{action}"))
    d1_d2_set = cp.asarray(d1_d2_neurons, dtype=cp.int64)
    mask_outside = ~cp.isin(post, d1_d2_set)
    if int(mask_outside.sum()) == 0:
        pytest.skip("No non-D1/D2-targeted synapses in this build")
    signs_outside = bridge.cp_d1_d2_sign[mask_outside]
    assert (signs_outside == 1.0).all(), \
        f"Non-D1/D2 synapses must keep sign=+1, got {signs_outside.unique()}"
    bridge.clear_simulation_state_and_gpu_memory()


def test_d1_d2_sign_inverts_weight_change_under_reward():
    """With enable_d1_d2_asymmetry on:
       - D1-targeting synapses' weights move in the SAME direction as reward
       - D2-targeting synapses' weights move in the OPPOSITE direction
    With a fixed positive eligibility trace and positive reward, D1 weights
    grow and D2 weights shrink."""
    pytest.importorskip("cupy")
    import cupy as cp
    bridge = _build_bg_bridge(enable_d1_d2=True)
    nnz = int(bridge.cp_connections.nnz)

    # Isolate the reward-modulated weight update path: disable other
    # plasticity rules so the only weight change comes from
    # effective_lr * RPE * eligibility * cp_d1_d2_sign. STDP, Hebbian,
    # homeostasis, and structural plasticity all write to cp_connections
    # and would mask the small reward delta we care about here.
    bridge.core_config.enable_stdp = False
    bridge.core_config.enable_hebbian_learning = False
    bridge.core_config.enable_homeostasis = False
    bridge.core_config.enable_structural_plasticity = False
    bridge.core_config.enable_synaptic_scaling = False

    # Set uniform positive eligibility on all synapses
    bridge.cp_eligibility_trace[:nnz] = 1.0
    # Save initial weights
    w_before = bridge.cp_connections.data.copy()
    # Apply reward (positive)
    bridge.core_config.current_reward_signal = 1.0
    bridge.core_config.reward_baseline = 0.0
    bridge.core_config.reward_learning_rate = 0.01
    bridge.core_config.enable_reward_modulation = True
    bridge._run_one_simulation_step()
    w_after = bridge.cp_connections.data
    delta = w_after - w_before

    # Find D1- and D2-targeted synapse indices
    post = bridge.cp_connections.indices
    d1_set = cp.asarray(
        [n for action in ("N", "E", "S", "W") for n in bridge.region_manager.indices(f"str_D1_{action}")],
        dtype=cp.int64,
    )
    d2_set = cp.asarray(
        [n for action in ("N", "E", "S", "W") for n in bridge.region_manager.indices(f"str_D2_{action}")],
        dtype=cp.int64,
    )
    d1_mask = cp.isin(post, d1_set)
    d2_mask = cp.isin(post, d2_set)

    if int(d1_mask.sum()) > 0:
        assert (delta[d1_mask] >= 0).all(), \
            f"D1 weights should grow under +reward; saw deltas {delta[d1_mask].min().get():.4f} to {delta[d1_mask].max().get():.4f}"
    if int(d2_mask.sum()) > 0:
        assert (delta[d2_mask] <= 0).all(), \
            f"D2 weights should shrink under +reward; saw deltas {delta[d2_mask].min().get():.4f} to {delta[d2_mask].max().get():.4f}"
    bridge.clear_simulation_state_and_gpu_memory()
