"""Focused guards for opt-in Vogels-style inhibitory STDP."""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

from sim.backend import get_backend, to_host
from sim.bridge import SimulationBridge
from sim.config import CoreSimConfig, GPUConfig, RuntimeState, VisualizationConfig
from sim.kernels import (
    fused_inhibitory_stdp_trace_update,
    fused_inhibitory_stdp_weight_update,
)


def _bridge(*, enabled: bool, include_gabab: bool = False) -> SimulationBridge:
    cfg = CoreSimConfig(
        num_neurons=6,
        connections_per_neuron=0,
        num_traits=1,
        seed=700,
        dt_ms=1.0,
        enable_watts_strogatz=False,
        enable_parameter_heterogeneity=False,
        enable_conductance_noise=False,
        enable_ou_process=False,
        enable_short_term_plasticity=False,
        enable_homeostasis=False,
        enable_stdp=False,
        enable_hebbian_learning=False,
        enable_reward_modulation=False,
        enable_structural_plasticity=False,
        enable_gabab=include_gabab,
        enable_inhibitory_stdp=enabled,
    )
    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


def _weight_by_edge(bridge: SimulationBridge) -> dict[tuple[int, int], float]:
    coo = bridge.cp_connections.tocoo(copy=False)
    rows = np.asarray(to_host(coo.row), dtype=np.int64)
    cols = np.asarray(to_host(coo.col), dtype=np.int64)
    data = np.asarray(to_host(coo.data), dtype=np.float64)
    return {(int(row), int(col)): float(weight) for row, col, weight in zip(rows, cols, data)}


def test_inhibitory_stdp_defaults_off_and_validates_parameters():
    cfg = CoreSimConfig()
    assert cfg.enable_inhibitory_stdp is False
    assert cfg.inhibitory_stdp_tau_ms == 20.0
    assert cfg.inhibitory_stdp_target_rate_per_step == 0.02
    assert cfg.inhibitory_stdp_eta == 0.001
    assert (cfg.inhibitory_stdp_w_min, cfg.inhibitory_stdp_w_max) == (0.0, 6.0)

    with pytest.raises(ValueError, match="inhibitory_stdp_tau_ms"):
        CoreSimConfig(inhibitory_stdp_tau_ms=0.0)
    with pytest.raises(ValueError, match="target_rate_per_step"):
        CoreSimConfig(inhibitory_stdp_target_rate_per_step=1.1)
    with pytest.raises(ValueError, match="inhibitory_stdp_eta"):
        CoreSimConfig(inhibitory_stdp_eta=-0.1)
    with pytest.raises(ValueError, match="inhibitory_stdp_w_min"):
        CoreSimConfig(inhibitory_stdp_w_min=7.0, inhibitory_stdp_w_max=6.0)


def test_fused_rule_matches_vogels_update_and_clips():
    xp, _ = get_backend()
    trace = fused_inhibitory_stdp_trace_update(
        xp.asarray([2.0, 0.0], dtype=xp.float32),
        xp.asarray([1.0, 1.0], dtype=xp.float32),
        xp.float32(0.5),
    )
    assert np.array_equal(
        np.asarray(to_host(trace)), np.asarray([2.0, 1.0], dtype=np.float32)
    )

    updated = fused_inhibitory_stdp_weight_update(
        xp.asarray([3.0, 5.9999, 0.0001], dtype=xp.float32),
        xp.asarray([2.0, 2.0, 0.0], dtype=xp.float32),
        xp.asarray([1.0, 3.0, 0.0], dtype=xp.float32),
        xp.asarray([True, True, True]),
        xp.asarray([True, True, False]),
        xp.float32(0.001),
        xp.float32(0.8),
        xp.float32(0.0),
        xp.float32(6.0),
    )
    assert np.allclose(
        np.asarray(to_host(updated)),
        np.asarray([3.0022, 6.0, 0.0], dtype=np.float32),
    )


def test_bridge_scopes_updates_to_plastic_gabaa_inhibitory_open_routes():
    bridge = _bridge(enabled=True, include_gabab=True)
    bridge.inject_explicit_wiring(
        {
            "eligible_up": {
                "pre_indices": [0],
                "post_indices": [2],
                "initial_weights": [3.0],
                "plastic": True,
                "plasticity_gate": "istdp_open",
                "receptor": "gaba_a",
            },
            "eligible_down": {
                "pre_indices": [0],
                "post_indices": [3],
                "initial_weights": [3.0],
                "plastic": True,
                "plasticity_gate": "istdp_open",
                "receptor": "gaba_a",
            },
            "fixed": {
                "pre_indices": [1],
                "post_indices": [2],
                "initial_weights": [3.0],
                "plastic": False,
                "receptor": "gaba_a",
            },
            "excitatory_pre": {
                "pre_indices": [2],
                "post_indices": [3],
                "initial_weights": [3.0],
                "plastic": True,
                "receptor": "gaba_a",
            },
            "closed_gate": {
                "pre_indices": [1],
                "post_indices": [3],
                "initial_weights": [3.0],
                "plastic": True,
                "plasticity_gate": "istdp_closed",
                "receptor": "gaba_a",
            },
            "gabab": {
                "pre_indices": [1],
                "post_indices": [4],
                "initial_weights": [3.0],
                "plastic": True,
                "receptor": "gaba_b",
            },
        },
        output_inhibitory_indices=[0, 1],
    )
    bridge.set_plasticity_gate("istdp_open", 1.0)
    bridge.set_plasticity_gate("istdp_closed", 0.0)
    before = _weight_by_edge(bridge)
    xp, _ = get_backend()

    bridge._apply_inhibitory_stdp(
        xp.asarray([True, True, True, False, True, False]),
        plasticity_gated=True,
    )
    after = _weight_by_edge(bridge)

    assert after[(0, 2)] > before[(0, 2)]
    assert after[(0, 3)] < before[(0, 3)]
    for edge in ((1, 2), (2, 3), (1, 3), (1, 4)):
        assert after[edge] == before[edge]
    assert np.array_equal(
        np.asarray(to_host(bridge.cp_inhibitory_stdp_trace)),
        np.asarray([1.0, 1.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32),
    )


def test_disabled_rule_allocates_no_trace_and_step_never_calls_rule(monkeypatch):
    bridge = _bridge(enabled=False)
    assert bridge.cp_inhibitory_stdp_trace is None
    weights_before = np.asarray(to_host(bridge.cp_connections.data)).tobytes()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("disabled inhibitory STDP path was reached")

    monkeypatch.setattr(bridge, "_apply_inhibitory_stdp", fail_if_called)
    for _ in range(3):
        bridge._run_one_simulation_step()
    assert np.asarray(to_host(bridge.cp_connections.data)).tobytes() == weights_before
    assert bridge.cp_inhibitory_stdp_trace is None


def test_trace_checkpoint_roundtrip_and_clear_reinitialize(tmp_path):
    pytest.importorskip("h5py")
    bridge = _bridge(enabled=True)
    xp, _ = get_backend()
    bridge.cp_inhibitory_stdp_trace[:] = xp.asarray(
        np.arange(6, dtype=np.float32)
    )
    checkpoint = tmp_path / "inhibitory-stdp.simstate.h5"
    assert bridge.save_checkpoint(str(checkpoint)) is not False

    restored = _bridge(enabled=False)
    assert restored.load_checkpoint(str(checkpoint)) is not False
    assert bool(restored.core_config.enable_inhibitory_stdp)
    assert np.array_equal(
        np.asarray(to_host(restored.cp_inhibitory_stdp_trace)),
        np.arange(6, dtype=np.float32),
    )

    restored.clear_simulation_state_and_gpu_memory()
    assert restored.cp_inhibitory_stdp_trace is None
    restored._initialize_simulation_data(called_from_playback_init=False)
    assert np.array_equal(
        np.asarray(to_host(restored.cp_inhibitory_stdp_trace)),
        np.zeros(6, dtype=np.float32),
    )
