"""Tests for the fast_spike_reset optimization in the Izhikevich path.

Validates that the cp.where masked-update path produces numerically
identical (v, u, refractory) trajectories to the legacy fancy-index
path.

The optimization avoids a GPU-CPU sync (`if fired_indices.size > 0`)
and reduces ~5 kernel launches per sub-step. Numerical equivalence
must hold to within float32 precision.
"""
from __future__ import annotations

import pytest


def _build_bridge(seed: int, fast_spike_reset: bool):
    """Build a small Izhikevich bridge with deterministic state for
    side-by-side equivalence testing.

    NOTE: OU noise is disabled (sigma=0) so two bridges sharing a CuPy
    context produce identical trajectories from same seed. Without
    this, the global GPU RNG state interleaves across the two bridges
    and they diverge from random noise alone, masking the actual
    optimization equivalence.
    """
    import cupy as cp
    from sim.config import (
        CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig,
    )
    from sim.bridge import SimulationBridge
    from research.runners.text_minimal_isolation import build_minimal_brain_regions

    regions, pathways = build_minimal_brain_regions(
        n_lang_input=128,
        n_motor_per_action=10,
    )
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5
    cfg.seed = seed
    cfg.enable_nmda = False
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = False
    cfg.stdp_w_max = 5.0
    cfg.fast_spike_reset = fast_spike_reset
    # Disable OU noise for deterministic side-by-side comparison
    cfg.ou_std_current_pA = 0.0

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


def _drive_some_neurons(bridge, n_drive=20, current_pA=300.0):
    """Inject a constant current into the first N neurons so they
    actually fire — gives the spike-reset path real work to do."""
    import cupy as cp
    bridge.cp_external_input_current[:n_drive] = cp.float32(current_pA)


@pytest.mark.parametrize("seed", [42, 123, 7])
def test_fast_spike_reset_numerically_equivalent_to_legacy(seed):
    """Run the same seed with fast_spike_reset=True and =False, step
    the bridge identically, compare state every 10 steps. Must match
    to within float32 precision."""
    import cupy as cp

    bridge_old = _build_bridge(seed, fast_spike_reset=False)
    bridge_new = _build_bridge(seed, fast_spike_reset=True)

    _drive_some_neurons(bridge_old)
    _drive_some_neurons(bridge_new)

    n_steps = 50
    for step in range(n_steps):
        bridge_old._run_one_simulation_step()
        bridge_new._run_one_simulation_step()
        bridge_old.runtime_state.current_time_step += 1
        bridge_new.runtime_state.current_time_step += 1

        # Spot-check state every 10 steps
        if (step + 1) % 10 == 0:
            assert cp.allclose(
                bridge_old.cp_membrane_potential_v,
                bridge_new.cp_membrane_potential_v,
                atol=1e-4,
            ), f"step {step+1}, seed {seed}: v diverged"

            assert cp.allclose(
                bridge_old.cp_recovery_variable_u,
                bridge_new.cp_recovery_variable_u,
                atol=1e-4,
            ), f"step {step+1}, seed {seed}: u diverged"

            # refractory timers are integers — must match exactly
            assert cp.array_equal(
                bridge_old.cp_refractory_timers,
                bridge_new.cp_refractory_timers,
            ), f"step {step+1}, seed {seed}: refractory diverged"


def test_fast_spike_reset_default_off():
    """fast_spike_reset must default to False so existing runs are
    bit-identical. Opt-in via cfg.fast_spike_reset = True."""
    from sim.config import CoreSimConfig
    cfg = CoreSimConfig()
    assert cfg.fast_spike_reset is False


def test_fast_spike_reset_handles_no_firings(seed=42):
    """When NO neurons fire in a step (no input current), both paths
    must produce identical state. Tests the empty-fired_indices case
    where fancy-index does nothing — cp.where must also do nothing."""
    import cupy as cp

    bridge_old = _build_bridge(seed, fast_spike_reset=False)
    bridge_new = _build_bridge(seed, fast_spike_reset=True)
    # NO drive — neurons stay at rest

    for _ in range(20):
        bridge_old._run_one_simulation_step()
        bridge_new._run_one_simulation_step()
        bridge_old.runtime_state.current_time_step += 1
        bridge_new.runtime_state.current_time_step += 1

    assert cp.allclose(
        bridge_old.cp_membrane_potential_v,
        bridge_new.cp_membrane_potential_v,
        atol=1e-5,
    )
    assert cp.array_equal(
        bridge_old.cp_refractory_timers,
        bridge_new.cp_refractory_timers,
    )


def test_fast_spike_reset_handles_high_firing(seed=42):
    """When MANY neurons fire simultaneously, both paths must produce
    identical state. Tests the heavy-fired-indices case where fancy-
    index touches many entries — cp.where must handle them all."""
    import cupy as cp

    bridge_old = _build_bridge(seed, fast_spike_reset=False)
    bridge_new = _build_bridge(seed, fast_spike_reset=True)
    # HEAVY drive on all language input neurons — many will fire each step
    _drive_some_neurons(bridge_old, n_drive=128, current_pA=500.0)
    _drive_some_neurons(bridge_new, n_drive=128, current_pA=500.0)

    for _ in range(30):
        bridge_old._run_one_simulation_step()
        bridge_new._run_one_simulation_step()
        bridge_old.runtime_state.current_time_step += 1
        bridge_new.runtime_state.current_time_step += 1

    assert cp.allclose(
        bridge_old.cp_membrane_potential_v,
        bridge_new.cp_membrane_potential_v,
        atol=1e-4,
    ), "v diverged under high firing"
    assert cp.array_equal(
        bridge_old.cp_refractory_timers,
        bridge_new.cp_refractory_timers,
    ), "refractory diverged under high firing"
