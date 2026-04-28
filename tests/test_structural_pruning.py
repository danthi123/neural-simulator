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


def test_update_pruning_increments_survival():
    """update_pruning(eligibility, reward) updates survival in place by
    alpha * eligibility * reward. Synapses with positive eligibility
    when reward is positive accumulate positive survival; opposite for
    negative reward."""
    pytest.importorskip("cupy")
    import cupy as cp
    from sim.config import CoreSimConfig

    # alpha=1.0 for crisp test signal; W-S defaults yield non-empty connections.
    cfg = CoreSimConfig(
        num_neurons=20, enable_structural_pruning=True, pruning_alpha=1.0
    )
    bridge = _build_bridge(cfg)
    nnz = int(bridge.cp_connections.nnz)
    assert nnz > 0, "test config must produce a non-empty connection matrix"
    # Set first half of synapses to eligibility +1, second half to -1
    eligibility = cp.zeros(nnz, dtype=cp.float32)
    eligibility[: nnz // 2] = 1.0
    eligibility[nnz // 2 :] = -1.0
    bridge.update_pruning(eligibility, reward_signal=1.0, prunable_indices=None)
    # First half should now have positive survival; second half negative
    surv = bridge.cp_synapse_survival.get()
    assert (surv[: nnz // 2] == 1.0).all()
    assert (surv[nnz // 2 :] == -1.0).all()
    bridge.clear_simulation_state_and_gpu_memory()


def test_update_pruning_eliminates_low_survival_low_weight():
    """When survival is below threshold AND weight is below floor, the
    synapse gets pruned: alive=False, weight=0."""
    pytest.importorskip("cupy")
    import cupy as cp
    from sim.config import CoreSimConfig

    cfg = CoreSimConfig(
        num_neurons=20,
        enable_structural_pruning=True,
        pruning_threshold=-0.5,
        pruning_weight_floor=0.5,
    )
    bridge = _build_bridge(cfg)
    nnz = int(bridge.cp_connections.nnz)
    assert nnz >= 4, "test config must produce at least 4 synapses for quadrant split"
    # Set first quarter to (low survival, low weight) — should prune
    # Set second quarter to (low survival, high weight) — should NOT prune
    # Set third quarter to (high survival, low weight) — should NOT prune
    # Set fourth quarter to (high survival, high weight) — should NOT prune
    bridge.cp_synapse_survival[: nnz // 4] = -1.0
    bridge.cp_synapse_survival[nnz // 4 : nnz // 2] = -1.0
    bridge.cp_synapse_survival[nnz // 2 : 3 * nnz // 4] = 1.0
    bridge.cp_synapse_survival[3 * nnz // 4 :] = 1.0
    bridge.cp_connections.data[: nnz // 4] = 0.1
    bridge.cp_connections.data[nnz // 4 : nnz // 2] = 1.0
    bridge.cp_connections.data[nnz // 2 : 3 * nnz // 4] = 0.1
    bridge.cp_connections.data[3 * nnz // 4 :] = 1.0
    bridge.update_pruning(
        eligibility_trace=cp.zeros(nnz, dtype=cp.float32),
        reward_signal=0.0,
        prunable_indices=None,
    )
    alive = bridge.cp_synapse_alive.get()
    assert not alive[: nnz // 4].any(), (
        "first quarter (low surv + low weight) should be pruned"
    )
    assert alive[nnz // 4 : nnz // 2].all(), (
        "second quarter (low surv + high weight) should survive"
    )
    assert alive[nnz // 2 : 3 * nnz // 4].all(), (
        "third quarter (high surv + low weight) should survive"
    )
    assert alive[3 * nnz // 4 :].all(), "fourth quarter should survive"
    weights = bridge.cp_connections.data.get()
    assert (weights[: nnz // 4] == 0.0).all(), "pruned synapses must have weight==0"
    bridge.clear_simulation_state_and_gpu_memory()


def test_update_pruning_respects_prunable_indices():
    """When prunable_indices is provided, only those synapses are eligible
    for pruning; others are left alone even if they meet the criteria."""
    pytest.importorskip("cupy")
    import cupy as cp
    from sim.config import CoreSimConfig

    cfg = CoreSimConfig(
        num_neurons=20,
        enable_structural_pruning=True,
        pruning_threshold=-0.5,
        pruning_weight_floor=0.5,
    )
    bridge = _build_bridge(cfg)
    nnz = int(bridge.cp_connections.nnz)
    assert nnz >= 2, "test config must produce at least 2 synapses for half split"
    # Set all synapses to (low survival, low weight) — would prune everything if unprotected
    bridge.cp_synapse_survival[:] = -1.0
    bridge.cp_connections.data[:] = 0.1
    # But only allow pruning of the first half
    prunable = cp.arange(nnz // 2, dtype=cp.int64)
    bridge.update_pruning(
        eligibility_trace=cp.zeros(nnz, dtype=cp.float32),
        reward_signal=0.0,
        prunable_indices=prunable,
    )
    alive = bridge.cp_synapse_alive.get()
    assert not alive[: nnz // 2].any(), "first half (in prunable set) pruned"
    assert alive[nnz // 2 :].all(), "second half (not in prunable set) protected"
    bridge.clear_simulation_state_and_gpu_memory()


def test_pruned_synapse_stays_at_zero_after_simulation_steps():
    """After pruning, even if other forces would push the weight up
    (or down), the alive mask keeps it at zero across many sim steps.

    Acts as a regression guard for the pruning invariant: once a synapse
    is alive=False with weight=0 and plasticity_gain=0, no plasticity
    pathway should reintroduce non-zero weight."""
    pytest.importorskip("cupy")
    from sim.config import CoreSimConfig

    cfg = CoreSimConfig(
        num_neurons=20,
        enable_structural_pruning=True,
        pruning_threshold=-0.5,
        pruning_weight_floor=0.5,
        enable_stdp=True,
    )
    bridge = _build_bridge(cfg)
    nnz = int(bridge.cp_connections.nnz)
    assert nnz >= 10, "test config must produce at least 10 synapses"
    # Force-prune the first 10 synapses (alive=False, weight=0,
    # plasticity_gain=0 — same end state as update_pruning would produce).
    bridge.cp_synapse_alive[:10] = False
    bridge.cp_connections.data[:10] = 0.0
    if bridge.cp_plasticity_gain is not None:
        bridge.cp_plasticity_gain[:10] = 0.0
    # Run several sim steps; pruned weights must stay 0 even though the
    # broader simulation pipeline (dynamics, STDP, etc.) is active.
    for _ in range(20):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    weights = bridge.cp_connections.data.get()
    assert (weights[:10] == 0.0).all(), (
        f"pruned synapse weights diverged from zero; "
        f"max |w[:10]|={float(abs(weights[:10]).max()):.6e}"
    )
    bridge.clear_simulation_state_and_gpu_memory()
