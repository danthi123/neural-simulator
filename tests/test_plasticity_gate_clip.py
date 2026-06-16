"""The Hebbian/reward/homeostasis weight CLIP must be GATED by cp_plasticity_rate_gain.

Regression test for the recurring "ungated-clip" foot-gun (sim/bridge.py): the per-step weight DECAY was already
gated by cp_plasticity_rate_gain (gain 0 = a frozen pathway does not decay), but the immediately-following CLIP to
[hebbian_min_weight, hebbian_max_weight] was UNGATED. When a sub-routine TEMPORARILY lowers hebbian_max_weight for
its own pathway's potentiation (e.g. a generalization-convergence pass that sets hebbian_max_weight=20), the ungated
clip crushed EVERY other (frozen) pathway's weights down to that bound -- which silenced the co-resident
conversational parser on the merged nav+conv bridge (its load-bearing conj->role edges, legitimately ~40-60, were
clipped to <=20). The fix gates the clip the same way the decay is gated: clip only gain>0 synapses; leave frozen
(gain 0) weights verbatim. gain=None (the default for non-gated configs) keeps the byte-identical un-gated clip.

CPU-friendly: forces SIM_BACKEND=numpy so the test runs without a GPU (the bug is structural, not GPU-specific).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FROZEN_W = 50.0      # a frozen pathway's legitimate weight, ABOVE the lowered clip max
LOW_MAX = 20.0       # a sub-routine temporarily lowers hebbian_max_weight to this
DRIVE_PA = 6000.0    # strong external current so every neuron fires every step (triggers the Hebbian clip block)


def _build_bridge():
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    cfg = CoreSimConfig()
    cfg.num_neurons = 60
    cfg.connection_density = 0.1
    cfg.dt_ms = 1.0
    cfg.fast_spike_reset = True
    cfg.enable_reward_modulation = False
    cfg.enable_hebbian_learning = True
    cfg.enable_stdp = False
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    cfg.hebbian_min_weight = 0.0
    cfg.hebbian_max_weight = LOW_MAX
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


def _drive_and_step(bridge, n_steps=4):
    for _ in range(n_steps):
        bridge.cp_external_input_current[:] = DRIVE_PA
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0


def _to_host(a):
    return np.asarray(a.get() if hasattr(a, "get") else a)


def test_frozen_pathway_survives_lowered_clip():
    """A gain-0 (frozen) synapse at weight 50 must SURVIVE the clip when hebbian_max_weight is 20."""
    bridge = _build_bridge()
    nnz = int(bridge.cp_connections.nnz)
    assert nnz >= 4, "need a few synapses"
    bridge.set_global_plasticity_gain(1.0)            # allocate cp_plasticity_rate_gain (all plastic)
    bridge.cp_plasticity_rate_gain[0] = 0.0           # synapse 0 = FROZEN
    bridge.cp_connections.data[0] = FROZEN_W          # frozen synapse legitimately above the lowered max
    bridge.cp_connections.data[1] = FROZEN_W          # synapse 1 = PLASTIC, also above the lowered max
    _drive_and_step(bridge)
    w = _to_host(bridge.cp_connections.data)
    assert w[0] == pytest.approx(FROZEN_W, abs=1e-3), (
        f"FROZEN synapse (gain 0) was clipped: {w[0]} != {FROZEN_W} — the ungated-clip foot-gun")
    assert w[1] <= LOW_MAX + 1e-3, (
        f"PLASTIC synapse (gain 1) should be clipped to <= {LOW_MAX}, got {w[1]}")


def test_gain_none_clips_normally_byte_identical_default():
    """The default path (cp_plasticity_rate_gain is None) keeps the un-gated clip: a synapse at 50 -> clipped to 20."""
    bridge = _build_bridge()
    assert bridge.cp_plasticity_rate_gain is None, (
        "this minimal bridge declares no gates, so the gain array must stay None (the default un-gated path)")
    bridge.cp_connections.data[0] = FROZEN_W
    bridge.cp_connections.data[1] = FROZEN_W
    _drive_and_step(bridge)
    w = _to_host(bridge.cp_connections.data)
    assert w[0] <= LOW_MAX + 1e-3, f"gain None must clip normally: {w[0]} should be <= {LOW_MAX}"
    assert w[1] <= LOW_MAX + 1e-3, f"gain None must clip normally: {w[1]} should be <= {LOW_MAX}"


if __name__ == "__main__":
    test_frozen_pathway_survives_lowered_clip()
    test_gain_none_clips_normally_byte_identical_default()
    print("PASS: clip-gate regression tests")
