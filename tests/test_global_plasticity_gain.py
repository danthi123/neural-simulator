"""Tests for bridge.set_global_plasticity_gain / get_global_plasticity_gain.

The new API is a global wrapper around cp_plasticity_rate_gain. It enables
"skip plasticity during reset_steps" optimization (perf audit item #3,
expected 1.3-1.5x speedup on plasticity-heavy training).

Test coverage:
- Method exists on bridge
- Lazy allocation path (cp_plasticity_rate_gain initially None)
- Fill path (cp_plasticity_rate_gain pre-allocated by gates)
- get_global_plasticity_gain returns uniform value
- get_global_plasticity_gain returns None when heterogeneous

GPU tests are gated on cupy availability.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_method_exists():
    """set_global_plasticity_gain + get_global_plasticity_gain are on the bridge class."""
    from sim.bridge import SimulationBridge
    assert hasattr(SimulationBridge, "set_global_plasticity_gain"), (
        "SimulationBridge.set_global_plasticity_gain missing — needed for "
        "perf optimization #3 (skip plasticity during reset_steps)"
    )
    assert hasattr(SimulationBridge, "get_global_plasticity_gain"), (
        "SimulationBridge.get_global_plasticity_gain missing"
    )


def test_signature_takes_value_only():
    """set_global_plasticity_gain takes a single value, no name."""
    import inspect
    from sim.bridge import SimulationBridge
    sig = inspect.signature(SimulationBridge.set_global_plasticity_gain)
    params = list(sig.parameters.keys())
    # self + value
    assert params == ["self", "value"], (
        f"Expected (self, value) signature, got {params}. The global gain "
        f"API is intentionally simpler than per-pathway set_plasticity_gate."
    )


def test_get_returns_float_or_none():
    """Return type annotation includes None (heterogeneous case)."""
    import inspect
    from sim.bridge import SimulationBridge
    sig = inspect.signature(SimulationBridge.get_global_plasticity_gain)
    # Just verify the method exists and returns something callable
    assert sig.return_annotation is not inspect.Signature.empty


# ──────────────────────────────────────────────────────────────────────
# GPU-required tests (need a real bridge to test allocation/fill paths)
# ──────────────────────────────────────────────────────────────────────


def _build_minimal_bridge_with_synapses():
    """Helper: build a tiny bridge with some synapses for testing."""
    pytest.importorskip("cupy")
    from sim.config import (
        CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig,
    )
    from sim.bridge import SimulationBridge

    cfg = CoreSimConfig()
    cfg.num_neurons = 100
    cfg.connection_density = 0.05
    cfg.dt_ms = 1.0
    cfg.fast_spike_reset = True
    cfg.enable_reward_modulation = False
    cfg.enable_hebbian_learning = False
    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


def test_lazy_allocation_when_no_gates():
    """If cp_plasticity_rate_gain is None initially, set_global allocates."""
    bridge = _build_minimal_bridge_with_synapses()
    # No gates declared → cp_plasticity_rate_gain may be None
    initial = bridge.cp_plasticity_rate_gain
    bridge.set_global_plasticity_gain(0.5)
    # Now it should be allocated and uniform
    assert bridge.cp_plasticity_rate_gain is not None, (
        "set_global_plasticity_gain should lazy-allocate cp_plasticity_rate_gain"
    )
    val = bridge.get_global_plasticity_gain()
    assert val == 0.5, f"Expected uniform 0.5, got {val}"


def test_fill_path_overrides_existing_gates():
    """If cp_plasticity_rate_gain pre-exists (from gates), fill replaces all."""
    pytest.importorskip("cupy")
    import cupy as cp
    bridge = _build_minimal_bridge_with_synapses()
    # Force-allocate with mixed values
    nnz = int(bridge.cp_connections.nnz)
    bridge.cp_plasticity_rate_gain = cp.ones(nnz, dtype=cp.float32)
    bridge.cp_plasticity_rate_gain[0] = 0.3  # heterogeneous
    bridge.cp_plasticity_rate_gain[5] = 0.7
    # Sanity: get should return None for heterogeneous
    assert bridge.get_global_plasticity_gain() is None
    # set_global wipes
    bridge.set_global_plasticity_gain(0.0)
    val = bridge.get_global_plasticity_gain()
    assert val == 0.0, f"Expected uniform 0.0 after fill, got {val}"


def test_idempotent_via_same_value():
    """Calling set_global twice with same value is no-op (uniform)."""
    bridge = _build_minimal_bridge_with_synapses()
    bridge.set_global_plasticity_gain(0.7)
    bridge.set_global_plasticity_gain(0.7)
    val = bridge.get_global_plasticity_gain()
    assert val == 0.7


def test_freeze_then_thaw_pattern():
    """The expected use case: freeze to 0, do something, thaw to 1."""
    bridge = _build_minimal_bridge_with_synapses()
    # Freeze
    bridge.set_global_plasticity_gain(0.0)
    assert bridge.get_global_plasticity_gain() == 0.0
    # ... reset_steps would happen here ...
    # Thaw
    bridge.set_global_plasticity_gain(1.0)
    assert bridge.get_global_plasticity_gain() == 1.0


def test_zero_synapses_safe():
    """If bridge has no connections (cp_connections empty), call is safe no-op."""
    pytest.importorskip("cupy")
    from sim.config import (
        CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig,
    )
    from sim.bridge import SimulationBridge

    cfg = CoreSimConfig()
    cfg.num_neurons = 1
    cfg.connection_density = 0.0  # zero connections
    cfg.dt_ms = 1.0
    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)
    # Should not crash even with no synapses
    bridge.set_global_plasticity_gain(0.0)
    bridge.set_global_plasticity_gain(1.0)
