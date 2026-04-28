"""Smoke tests for the structural-plasticity (axon pruning) machinery."""
from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_bridge(cfg):
    """Helper that mirrors the codebase's existing bridge-init test pattern
    (see tests/test_regions.py::_make_bridge_with_regions). Calls
    `_initialize_simulation_data` directly because that is the central entry
    point that allocates synapse-indexed arrays for both the region and
    non-region paths."""
    pytest.importorskip("cupy")
    from sim.bridge import SimulationBridge
    from sim.config import VisualizationConfig, RuntimeState, GPUConfig

    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


def test_structural_pruning_arrays_allocated_when_enabled():
    """When enable_structural_pruning is True on a bridge with synapses,
    `cp_synapse_alive` (bool) and `cp_synapse_survival` (float32) arrays
    are allocated and have shape (nnz,)."""
    pytest.importorskip("cupy")
    import cupy as cp
    from sim.config import CoreSimConfig

    # Defaults: enable_watts_strogatz=True, connectivity_k=10. With 20 neurons
    # this yields a non-empty connection matrix (the plan's
    # connection_density_input/_recurrent kwargs aren't real CoreSimConfig
    # fields; W-S defaults serve the same role for this smoke test).
    cfg = CoreSimConfig(num_neurons=20, enable_structural_pruning=True)
    bridge = _build_bridge(cfg)
    nnz = int(bridge.cp_connections.nnz)
    assert nnz > 0, "test config must produce a non-empty connection matrix"
    assert hasattr(bridge, "cp_synapse_alive"), "cp_synapse_alive must be allocated"
    assert hasattr(bridge, "cp_synapse_survival"), "cp_synapse_survival must be allocated"
    assert bridge.cp_synapse_alive is not None
    assert bridge.cp_synapse_survival is not None
    assert bridge.cp_synapse_alive.shape == (nnz,)
    assert bridge.cp_synapse_survival.shape == (nnz,)
    assert bridge.cp_synapse_alive.dtype == cp.bool_
    assert bridge.cp_synapse_survival.dtype == cp.float32
    # All synapses start alive and with zero survival score
    assert bool(bridge.cp_synapse_alive.all())
    assert float(bridge.cp_synapse_survival.sum()) == 0.0
    bridge.clear_simulation_state_and_gpu_memory()


def test_structural_pruning_default_off():
    """When the flag is not set, the arrays are not allocated. Flagship is bit-identical."""
    pytest.importorskip("cupy")
    from sim.config import CoreSimConfig

    cfg = CoreSimConfig(num_neurons=20)
    bridge = _build_bridge(cfg)
    assert not hasattr(bridge, "cp_synapse_alive") or bridge.cp_synapse_alive is None
    assert not hasattr(bridge, "cp_synapse_survival") or bridge.cp_synapse_survival is None
    bridge.clear_simulation_state_and_gpu_memory()
