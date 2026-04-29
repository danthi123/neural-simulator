"""Unit tests for per-region GABA_A reversal potential override (R1.1).

Catalog reference: Kandel PBR-160 ch 6 (striatum) and ch 11 (SNc DA).

Striatal MSNs measured ECl ~−60 mV via gramicidin perforated patch (vs the
−75 mV cortical-pyramidal default). SNc DA neurons lack KCC2 chloride
exporter → ECl ~−55 mV. Both deviate enough from a global cfg default
that misusing −75 mV silently distorts dendritic integration in the BG
cascade.

This module verifies:
  * BrainRegion.syn_reversal_potential_i_override defaults to None.
  * The bridge allocates a per-neuron E_inh array initialized to global config.
  * Region-level overrides are applied to the correct contiguous slice.
  * Regions without an override keep the global value.
  * Determinism: re-init produces identical per-neuron E_inh arrays.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------- BrainRegion field default ----------

def test_brain_region_e_inh_override_defaults_to_none():
    from sim.regions import BrainRegion

    r = BrainRegion(name="cortex", n_neurons=10)
    assert r.syn_reversal_potential_i_override is None


def test_brain_region_e_inh_override_accepts_float():
    from sim.regions import BrainRegion

    r_msn = BrainRegion(name="str_D1", n_neurons=10,
                         syn_reversal_potential_i_override=-60.0)
    r_da = BrainRegion(name="snc", n_neurons=5,
                        syn_reversal_potential_i_override=-55.0)
    assert r_msn.syn_reversal_potential_i_override == -60.0
    assert r_da.syn_reversal_potential_i_override == -55.0


# ---------- Bridge integration ----------

def _make_bridge_with_regions(brain_regions, region_pathways=None, seed=42):
    """Helper: minimal Izhikevich bridge with the region framework on."""
    pytest.importorskip("cupy")
    from sim import (
        SimulationBridge, CoreSimConfig, VisualizationConfig,
        RuntimeState, GPUConfig,
    )
    from sim.enums import NeuronModel

    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = seed
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(brain_regions)
    cfg.region_pathways = list(region_pathways or [])

    sb = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    return sb, cfg


def test_e_inh_array_defaults_to_global_when_no_overrides():
    pytest.importorskip("cupy")
    import cupy as cp
    from sim.regions import BrainRegion

    sb, cfg = _make_bridge_with_regions(
        brain_regions=[
            BrainRegion(name="cortex", n_neurons=20, internal_density=0.0),
            BrainRegion(name="motor", n_neurons=10, internal_density=0.0),
        ],
    )
    try:
        assert sb.cp_syn_reversal_potential_i_per_neuron is not None
        arr = cp.asnumpy(sb.cp_syn_reversal_potential_i_per_neuron)
        assert arr.shape == (30,)
        # All 30 neurons should hold the global default.
        global_e_inh = cfg.syn_reversal_potential_i
        assert (arr == global_e_inh).all(), (
            f"expected all values == {global_e_inh}, got unique={set(arr.tolist())}"
        )
    finally:
        sb.clear_simulation_state_and_gpu_memory()


def test_e_inh_array_applies_per_region_override():
    """Regions with override get the override; others get the global default."""
    pytest.importorskip("cupy")
    import cupy as cp
    from sim.regions import BrainRegion

    sb, cfg = _make_bridge_with_regions(
        brain_regions=[
            BrainRegion(name="cortex", n_neurons=20, internal_density=0.0),
            # Striatal MSN per PBR-160 ch 6 (gramicidin perforated patch).
            BrainRegion(name="str_D1", n_neurons=15, internal_density=0.0,
                         syn_reversal_potential_i_override=-60.0),
            BrainRegion(name="motor", n_neurons=10, internal_density=0.0),
            # SNc DA per PBR-160 ch 11 (lacks KCC2).
            BrainRegion(name="dopamine", n_neurons=5, internal_density=0.0,
                         syn_reversal_potential_i_override=-55.0),
        ],
    )
    try:
        arr = cp.asnumpy(sb.cp_syn_reversal_potential_i_per_neuron)
        global_e_inh = cfg.syn_reversal_potential_i  # -75 mV by default
        assert arr.shape == (50,)

        # cortex: indices [0, 20) -> global default
        assert (arr[0:20] == global_e_inh).all()
        # str_D1: indices [20, 35) -> -60.0
        assert (arr[20:35] == -60.0).all()
        # motor: indices [35, 45) -> global default
        assert (arr[35:45] == global_e_inh).all()
        # dopamine: indices [45, 50) -> -55.0
        assert (arr[45:50] == -55.0).all()
    finally:
        sb.clear_simulation_state_and_gpu_memory()


def test_e_inh_array_deterministic_across_inits():
    """Two independent inits with the same seed produce identical E_inh arrays."""
    pytest.importorskip("cupy")
    import cupy as cp
    from sim.regions import BrainRegion

    regions = [
        BrainRegion(name="cortex", n_neurons=20, internal_density=0.0),
        BrainRegion(name="str_D1", n_neurons=10, internal_density=0.0,
                     syn_reversal_potential_i_override=-60.0),
    ]

    sb1, _ = _make_bridge_with_regions(regions, seed=42)
    arr1 = cp.asnumpy(sb1.cp_syn_reversal_potential_i_per_neuron).copy()
    sb1.clear_simulation_state_and_gpu_memory()

    sb2, _ = _make_bridge_with_regions(regions, seed=42)
    arr2 = cp.asnumpy(sb2.cp_syn_reversal_potential_i_per_neuron).copy()
    sb2.clear_simulation_state_and_gpu_memory()

    assert (arr1 == arr2).all()


def test_e_inh_global_when_region_framework_disabled():
    """Without the region framework, the array still allocates with the global value."""
    pytest.importorskip("cupy")
    import cupy as cp
    from sim import (
        SimulationBridge, CoreSimConfig, VisualizationConfig,
        RuntimeState, GPUConfig,
    )
    from sim.enums import NeuronModel

    cfg = CoreSimConfig()
    cfg.num_neurons = 25
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = 42
    # default: enable_brain_region_framework = False

    sb = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    try:
        assert sb.region_manager is None
        assert sb.cp_syn_reversal_potential_i_per_neuron is not None
        arr = cp.asnumpy(sb.cp_syn_reversal_potential_i_per_neuron)
        assert arr.shape == (25,)
        assert (arr == cfg.syn_reversal_potential_i).all()
    finally:
        sb.clear_simulation_state_and_gpu_memory()


# ---------- End-to-end: kernel accepts the per-neuron array ----------

def test_fused_conductance_kernel_accepts_per_neuron_e_inh():
    """The fused kernel broadcasts a CuPy array E_i element-wise against v.

    This guards against a future @cp.fuse() breaking change that might
    reject a non-scalar E_i argument.
    """
    pytest.importorskip("cupy")
    import cupy as cp
    from sim.kernels import fused_conductance_decay_and_current

    n = 8
    g_e = cp.full(n, 0.5, dtype=cp.float32)
    g_i = cp.full(n, 0.3, dtype=cp.float32)
    decay_e = cp.float32(0.9)
    decay_i = cp.float32(0.85)
    v = cp.full(n, -65.0, dtype=cp.float32)
    E_e = cp.float32(0.0)
    # Mixed E_i: half neurons at −75 mV, half at −60 mV.
    E_i = cp.array([-75.0, -75.0, -75.0, -75.0,
                    -60.0, -60.0, -60.0, -60.0], dtype=cp.float32)

    g_e_new, g_i_new, I_syn = fused_conductance_decay_and_current(
        g_e, g_i, decay_e, decay_i, v, E_e, E_i
    )
    # Decay correctness
    assert cp.allclose(g_e_new, g_e * decay_e)
    assert cp.allclose(g_i_new, g_i * decay_i)
    # Current uses per-neuron E_i: I = g_e_new*(E_e-v) + g_i_new*(E_i-v)
    expected = g_e_new * (E_e - v) + g_i_new * (E_i - v)
    assert cp.allclose(I_syn, expected)
    # Sanity: neurons with E_i=-60 receive less inhibitory drive than -75
    # (because |E_i - v| is smaller when E_i is closer to v).
    I_syn_np = cp.asnumpy(I_syn)
    # The first half (E_i=-75) has more negative inhib current contribution.
    assert I_syn_np[0] < I_syn_np[4]


def test_e_inh_override_runner_assignment():
    """The g11_bg_runner builder correctly tags D1, D2, and dopamine."""
    from research.runners.g11_bg_runner import build_bg_brain_regions

    regions, _ = build_bg_brain_regions()
    by_name = {r.name: r for r in regions}

    # All four D1 regions
    for action in ("N", "E", "S", "W"):
        d1 = by_name[f"str_D1_{action}"]
        d2 = by_name[f"str_D2_{action}"]
        assert d1.syn_reversal_potential_i_override == -60.0, (
            f"str_D1_{action}: expected -60, got {d1.syn_reversal_potential_i_override}"
        )
        assert d2.syn_reversal_potential_i_override == -60.0, (
            f"str_D2_{action}: expected -60, got {d2.syn_reversal_potential_i_override}"
        )
    # SNc DA
    da = by_name["dopamine"]
    assert da.syn_reversal_potential_i_override == -55.0
    # Spot-check a region that should NOT have the override
    cortex_n = by_name["cortex_N"]
    assert cortex_n.syn_reversal_potential_i_override is None
    # GPi pallidum: cortical-pyramidal default applies
    gpi_n = by_name["gpi_N"]
    assert gpi_n.syn_reversal_potential_i_override is None
    # STN: glutamatergic, no override
    stn = by_name["stn"]
    assert stn.syn_reversal_potential_i_override is None
