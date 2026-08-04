"""Focused coverage for population-scoped HH membrane parameters."""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from sim import (  # noqa: E402
    CoreSimConfig,
    GPUConfig,
    NeuronModel,
    RuntimeState,
    SimulationBridge,
    VisualizationConfig,
)
from sim.backend import to_host  # noqa: E402
from sim.enums import DefaultHodgkinHuxleyParams, NeuronType  # noqa: E402
from sim.regions import BrainRegion  # noqa: E402


_READ_ONLY = {
    "enable_parameter_heterogeneity": False,
    "enable_ou_process": False,
    "enable_conductance_noise": False,
    "enable_hebbian_learning": False,
    "enable_short_term_plasticity": False,
    "enable_homeostasis": False,
    "enable_stdp": False,
    "enable_structural_plasticity": False,
    "enable_reward_modulation": False,
    "enable_inhibitory_stdp": False,
    "enable_nmda": False,
    "enable_gabab": False,
    "enable_step_megakernel_v2": False,
}

_ARRAYS = {
    "hh_C_m_override": "cp_hh_C_m",
    "hh_g_Na_max_override": "cp_hh_g_Na_max",
    "hh_g_K_max_override": "cp_hh_g_K_max",
    "hh_g_L_override": "cp_hh_g_L",
    "hh_E_Na_override": "cp_hh_E_Na",
    "hh_E_K_override": "cp_hh_E_K",
    "hh_E_L_override": "cp_hh_E_L",
}


def _region(name: str, n: int, **overrides) -> BrainRegion:
    return BrainRegion(
        name=name,
        n_neurons=n,
        internal_density=0.0,
        **overrides,
    )


def _build(regions, *, model=NeuronModel.HODGKIN_HUXLEY.name):
    regions = list(regions)
    cfg = CoreSimConfig(
        num_neurons=sum(region.n_neurons for region in regions),
        connections_per_neuron=0,
        seed=1741,
        neuron_model_type=model,
        default_neuron_type_hh="HH_EXCITATORY_DEFAULT_LEGACY",
        dt_ms=0.05,
        enable_brain_region_framework=True,
        brain_regions=regions,
        region_pathways=[],
        hh_external_drive_scale=0.0,
        **_READ_ONLY,
    )
    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(enable_profiling=False),
    )
    bridge._initialize_simulation_data()
    assert bridge.is_initialized
    return bridge


def _capture(bridge, steps):
    raster = []
    for _ in range(steps):
        bridge._run_one_simulation_step()
        raster.append(to_host(bridge.cp_firing_states).copy())
    return {
        "raster": np.stack(raster),
        "voltage": to_host(bridge.cp_membrane_potential_v).copy(),
        "m": to_host(bridge.cp_gating_variable_m).copy(),
        "h": to_host(bridge.cp_gating_variable_h).copy(),
        "n": to_host(bridge.cp_gating_variable_n).copy(),
    }


def test_absent_overrides_preserve_selected_hh_preset_exactly():
    bridge = _build([_region("control", 7)])
    params = DefaultHodgkinHuxleyParams.get_params(
        NeuronType.HH_EXCITATORY_DEFAULT_LEGACY
    )
    parameter_names = {
        "cp_hh_C_m": "C_m",
        "cp_hh_g_Na_max": "g_Na_max",
        "cp_hh_g_K_max": "g_K_max",
        "cp_hh_g_L": "g_L",
        "cp_hh_E_Na": "E_Na",
        "cp_hh_E_K": "E_K",
        "cp_hh_E_L": "E_L",
    }
    for attr_name, parameter_name in parameter_names.items():
        expected = np.full(7, params[parameter_name], dtype=np.float32)
        np.testing.assert_array_equal(to_host(getattr(bridge, attr_name)), expected)


def test_explicit_hh_overrides_are_region_scoped():
    values = {
        "hh_C_m_override": 1.4,
        "hh_g_Na_max_override": 88.0,
        "hh_g_K_max_override": 24.0,
        "hh_g_L_override": 0.12,
        "hh_E_Na_override": 52.0,
        "hh_E_K_override": -91.0,
        "hh_E_L_override": -67.0,
    }
    bridge = _build([
        _region("control", 3),
        _region("snr", 4, **values),
        _region("second-control", 2),
    ])
    defaults = DefaultHodgkinHuxleyParams.get_params(
        NeuronType.HH_EXCITATORY_DEFAULT_LEGACY
    )
    default_names = {
        "hh_C_m_override": "C_m",
        "hh_g_Na_max_override": "g_Na_max",
        "hh_g_K_max_override": "g_K_max",
        "hh_g_L_override": "g_L",
        "hh_E_Na_override": "E_Na",
        "hh_E_K_override": "E_K",
        "hh_E_L_override": "E_L",
    }
    for field_name, attr_name in _ARRAYS.items():
        actual = to_host(getattr(bridge, attr_name))
        expected = np.full(9, defaults[default_names[field_name]], dtype=np.float32)
        expected[3:7] = np.float32(values[field_name])
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "field_name",
    [
        "hh_C_m_override",
        "hh_g_Na_max_override",
        "hh_g_K_max_override",
        "hh_g_L_override",
    ],
)
@pytest.mark.parametrize("bad_value", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_positive_hh_overrides_reject_invalid_values(field_name, bad_value):
    with pytest.raises(ValueError, match=f"{field_name} must be finite and positive"):
        _region("snr", 2, **{field_name: bad_value})


@pytest.mark.parametrize(
    "field_name", ["hh_E_Na_override", "hh_E_K_override", "hh_E_L_override"]
)
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_reversal_overrides_reject_nonfinite_values(field_name, bad_value):
    with pytest.raises(ValueError, match=f"{field_name} must be finite"):
        _region("snr", 2, **{field_name: bad_value})


def test_hh_overrides_require_hh_model_even_after_region_mutation():
    region = _region("snr", 2)
    region.hh_C_m_override = 1.2
    with pytest.raises(ValueError, match="require HODGKIN_HUXLEY"):
        _build([region], model=NeuronModel.IZHIKEVICH.name)


def test_bridge_rejects_unrepresentable_override_after_region_mutation():
    region = _region("snr", 2)
    region.hh_E_Na_override = 1e40
    with pytest.raises(ValueError, match="representable as float32"):
        _build([region])


def test_checkpoint_round_trip_preserves_overridden_arrays(tmp_path):
    values = {
        "hh_C_m_override": 1.4,
        "hh_g_Na_max_override": 88.0,
        "hh_g_K_max_override": 24.0,
        "hh_g_L_override": 0.12,
        "hh_E_Na_override": 52.0,
        "hh_E_K_override": -91.0,
        "hh_E_L_override": -67.0,
    }
    bridge = _build([_region("snr", 4, **values)])
    expected = {
        attr_name: to_host(getattr(bridge, attr_name)).copy()
        for attr_name in _ARRAYS.values()
    }
    checkpoint = tmp_path / "hh-region-overrides.simstate.h5"
    _capture(bridge, 5)
    assert bridge.save_checkpoint(str(checkpoint)) is True
    uninterrupted = _capture(bridge, 10)

    restored = _build([_region("snr", 4)])
    assert restored.load_checkpoint(str(checkpoint)) is True
    for attr_name, expected_array in expected.items():
        np.testing.assert_array_equal(
            to_host(getattr(restored, attr_name)), expected_array
        )
    continued = _capture(restored, 10)
    for name, expected_array in uninterrupted.items():
        np.testing.assert_array_equal(continued[name], expected_array)

    with h5py.File(checkpoint, "r") as h5f:
        for attr_name, expected_array in expected.items():
            np.testing.assert_array_equal(h5f[attr_name], expected_array)
