"""Tests for bridge.enable_stp_runtime() — late-enable STP after STP-off training.

Per user (2026-05-10): "I'd say select the option you think makes most sense.
We definitely don't want to abandon biological realism as a core foundation
of the sim, but I'm okay with temporarily disabling things as needed when
it gets us significant performance (and other metric) boosts that would
persist even after reenabling STP (such as initial language training)."

This test verifies the API: train fast with STP-off, then re-enable for
inference/eval. CPU-only tests check method exists + signature. GPU tests
gated on cupy.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_method_exists():
    """SimulationBridge.enable_stp_runtime is callable."""
    from sim.bridge import SimulationBridge
    assert hasattr(SimulationBridge, "enable_stp_runtime")
    assert callable(SimulationBridge.enable_stp_runtime)


def test_signature_no_required_args():
    """enable_stp_runtime takes only self."""
    import inspect
    from sim.bridge import SimulationBridge
    sig = inspect.signature(SimulationBridge.enable_stp_runtime)
    params = [p for p in sig.parameters.values()
              if p.name != "self" and p.default is inspect.Parameter.empty]
    assert len(params) == 0, "enable_stp_runtime should take no required args"


def _build_bridge_with_stp_off():
    """Helper: build a tiny bridge with STP disabled."""
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
    cfg.enable_short_term_plasticity = False  # KEY: train without STP
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


def test_enable_after_init_with_stp_off():
    """Bridge built with STP-off has cp_stp_x=None; enable_stp_runtime allocates."""
    bridge = _build_bridge_with_stp_off()
    # Pre-condition: STP off, arrays not allocated
    assert bridge.core_config.enable_short_term_plasticity is False
    assert bridge.cp_stp_x is None
    assert bridge.cp_stp_u is None
    # Re-enable
    newly = bridge.enable_stp_runtime()
    assert newly is True, "should newly-allocate STP arrays"
    # Post-condition: STP on, arrays allocated
    assert bridge.core_config.enable_short_term_plasticity is True
    assert bridge.cp_stp_x is not None
    assert bridge.cp_stp_u is not None
    # Sanity: cp_stp_x should be ones (initial vesicle pool)
    import cupy as cp
    assert bool(cp.all(bridge.cp_stp_x == 1.0).get())


def test_enable_idempotent():
    """Calling twice doesn't crash; second call returns False (already active)."""
    bridge = _build_bridge_with_stp_off()
    first = bridge.enable_stp_runtime()
    second = bridge.enable_stp_runtime()
    assert first is True
    assert second is False  # already active


def test_does_not_clobber_existing_stp():
    """If bridge was built with STP-on, calling again is a no-op."""
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
    cfg.enable_short_term_plasticity = True  # ALREADY ON
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
    # Modify stp_x to non-default values
    import cupy as cp
    bridge.cp_stp_x[:] = 0.7
    # Re-enable — should be no-op
    result = bridge.enable_stp_runtime()
    assert result is False  # already active
    # Verify values not clobbered
    assert bool(cp.all(bridge.cp_stp_x == 0.7).get()), (
        "enable_stp_runtime should not clobber existing STP state"
    )


def test_step_works_after_runtime_enable():
    """After enabling STP at runtime, bridge step shouldn't crash."""
    bridge = _build_bridge_with_stp_off()
    bridge.enable_stp_runtime()
    # Run a few steps — should not raise
    for _ in range(10):
        bridge._run_one_simulation_step()
    # Sanity: STP arrays still exist
    assert bridge.cp_stp_x is not None
