"""Checkpoint coverage for optional slow synaptic conductance state."""
from __future__ import annotations

import importlib
import sys

import h5py
import numpy as np
import pytest


SLOW_CONDUCTANCE_ARRAYS = (
    "cp_conductance_g_nmda",
    "cp_conductance_g_nmda_rise",
    "cp_conductance_g_nmda_recurrent",
    "cp_conductance_g_nmda_recurrent_rise",
    "cp_conductance_g_gabab",
    "cp_conductance_g_gabab_slow",
)


@pytest.fixture
def numpy_backend(monkeypatch):
    monkeypatch.setenv("SIM_BACKEND", "numpy")
    from sim.backend import _reset_cache_for_tests, get_backend

    _reset_cache_for_tests()
    xp, name = get_backend("numpy")
    assert name == "numpy"
    for module_name in ("sim.kernels", "sim.connectivity", "sim.bridge"):
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])

    yield xp

    _reset_cache_for_tests()
    for module_name in ("sim.kernels", "sim.connectivity", "sim.bridge"):
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])


def _build_bridge(*, enable_slow_conductances: bool):
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, GPUConfig, RuntimeState, VisualizationConfig

    config = CoreSimConfig(
        num_neurons=12,
        enable_ou_process=False,
        enable_nmda=enable_slow_conductances,
        enable_nmda_recurrent=enable_slow_conductances,
        enable_gabab=enable_slow_conductances,
        enable_td_value_derivative=enable_slow_conductances,
    )
    bridge = SimulationBridge(
        core_config=config,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    bridge._initialize_simulation_data()
    assert bridge.is_initialized
    return bridge


def test_allocated_slow_conductances_round_trip_exactly(numpy_backend, tmp_path):
    bridge = _build_bridge(enable_slow_conductances=True)
    expected = {}
    for offset, attr_name in enumerate(SLOW_CONDUCTANCE_ARRAYS, start=1):
        values = np.linspace(offset, offset + 0.75, bridge.core_config.num_neurons).astype(
            np.float32
        )
        getattr(bridge, attr_name)[:] = values
        expected[attr_name] = values.copy()

    checkpoint = tmp_path / "slow-conductance.simstate.h5"
    assert bridge.save_checkpoint(str(checkpoint)) is True

    restored = _build_bridge(enable_slow_conductances=False)
    assert restored.load_checkpoint(str(checkpoint)) is True
    for attr_name, expected_values in expected.items():
        np.testing.assert_array_equal(getattr(restored, attr_name), expected_values)


def test_legacy_checkpoint_rebuilds_enabled_slow_conductances(numpy_backend, tmp_path):
    bridge = _build_bridge(enable_slow_conductances=True)
    checkpoint = tmp_path / "legacy.simstate.h5"
    assert bridge.save_checkpoint(str(checkpoint)) is True

    with h5py.File(checkpoint, "r+") as h5f:
        for attr_name in SLOW_CONDUCTANCE_ARRAYS:
            del h5f[attr_name]

    restored = _build_bridge(enable_slow_conductances=False)
    assert restored.load_checkpoint(str(checkpoint)) is True
    for attr_name in SLOW_CONDUCTANCE_ARRAYS:
        value = getattr(restored, attr_name)
        assert value is not None
        np.testing.assert_array_equal(value, np.zeros(bridge.core_config.num_neurons, np.float32))


def test_default_checkpoint_keeps_slow_conductances_unallocated(numpy_backend, tmp_path):
    bridge = _build_bridge(enable_slow_conductances=False)
    checkpoint = tmp_path / "default.simstate.h5"
    assert bridge.save_checkpoint(str(checkpoint)) is True

    with h5py.File(checkpoint, "r") as h5f:
        assert "cp_conductance_g_nmda" in h5f
        assert "cp_conductance_g_nmda_rise" in h5f
        assert all(attr_name not in h5f for attr_name in SLOW_CONDUCTANCE_ARRAYS[2:])

    restored = _build_bridge(enable_slow_conductances=True)
    assert restored.load_checkpoint(str(checkpoint)) is True
    np.testing.assert_array_equal(
        restored.cp_conductance_g_nmda,
        np.zeros(bridge.core_config.num_neurons, np.float32),
    )
    np.testing.assert_array_equal(
        restored.cp_conductance_g_nmda_rise,
        np.zeros(bridge.core_config.num_neurons, np.float32),
    )
    assert all(getattr(restored, attr_name) is None for attr_name in SLOW_CONDUCTANCE_ARRAYS[2:])
