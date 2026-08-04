"""Focused simulator integration tests for exact inhibitory conductance clamp."""
from __future__ import annotations

import inspect
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from experiment.inhibitory_conductance_clamp import (  # noqa: E402
    BiexponentialInhibitoryEvent,
    EventSchedule,
    InhibitoryBarrage,
)
from sim.backend import get_backend, to_host  # noqa: E402
from sim.bridge import (  # noqa: E402
    SimulationBridge,
    _INHIBITORY_CLAMP_STATE_ARRAYS,
)
from sim.config import (  # noqa: E402
    CoreSimConfig,
    GPUConfig,
    InhibitoryConductanceClampConfig,
    RuntimeState,
    VisualizationConfig,
)
from sim.enums import NeuronModel  # noqa: E402
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
}


def _channel(pathway, *, peak, rise, decay, events):
    return InhibitoryConductanceClampConfig(
        pathway=pathway,
        target_region="snr",
        tau_rise_ms=rise,
        tau_decay_ms=decay,
        reversal_mV=-70.0,
        event_peak_nS=peak,
        membrane_area_um2=2000.0,
        event_times_ms=events,
    )


def _channels():
    return [
        _channel(
            "direct_striatonigral",
            peak=0.9,
            rise=0.9,
            decay=6.2,
            events=[0.0, 0.15, 0.4],
        ),
        _channel(
            "pallidonigral",
            peak=2.0,
            rise=0.4,
            decay=2.1,
            events=[0.0, 0.2, 0.45],
        ),
    ]


def _build(*, enabled=True, channels=None):
    channels = _channels() if channels is None else channels
    regions = [
        BrainRegion(name="control", n_neurons=2, internal_density=0.0),
        BrainRegion(name="snr", n_neurons=3, internal_density=0.0),
    ]
    cfg = CoreSimConfig(
        num_neurons=5,
        connections_per_neuron=0,
        seed=1741,
        neuron_model_type=NeuronModel.HODGKIN_HUXLEY.name,
        default_neuron_type_hh="HH_EXCITATORY_DEFAULT_LEGACY",
        dt_ms=0.05,
        total_simulation_time_ms=2.0,
        enable_brain_region_framework=True,
        brain_regions=regions,
        region_pathways=[],
        hh_external_drive_scale=0.0,
        enable_inhibitory_conductance_clamp=enabled,
        inhibitory_conductance_clamps=channels,
        **_READ_ONLY,
    )
    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(enable_profiling=False),
    )
    bridge.strict_step_errors = True
    bridge._initialize_simulation_data()
    assert bridge.is_initialized
    return bridge


def _advance(bridge, steps):
    for _ in range(steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        bridge.runtime_state.current_time_step += 1


def _capture(bridge):
    names = (
        "cp_membrane_potential_v",
        "cp_gating_variable_m",
        "cp_gating_variable_h",
        "cp_gating_variable_n",
        "cp_conductance_g_e",
        "cp_conductance_g_i",
        *_INHIBITORY_CLAMP_STATE_ARRAYS,
        "cp_inhibitory_clamp_current_pA_equivalent",
    )
    return {
        name: None if getattr(bridge, name) is None else to_host(
            getattr(bridge, name)
        ).copy()
        for name in names
    }


def test_default_off_is_byte_identical_with_dormant_channel_declarations():
    empty = _build(enabled=False, channels=[])
    declared = _build(enabled=False, channels=_channels())

    assert empty.cp_inhibitory_clamp_decay_state_nS is None
    assert declared.cp_inhibitory_clamp_decay_state_nS is None
    _advance(empty, 8)
    _advance(declared, 8)
    for name, expected in _capture(empty).items():
        actual = _capture(declared)[name]
        if expected is None:
            assert actual is None
        else:
            np.testing.assert_array_equal(actual, expected)


def test_paths_remain_distinct_and_current_is_region_scoped_and_area_scaled():
    bridge = _build()
    bridge.cp_membrane_potential_v[:] = -50.0
    generic_before = to_host(bridge.cp_conductance_g_i).copy()

    first = to_host(bridge._update_inhibitory_conductance_clamp_current()).copy()
    bridge.runtime_state.current_time_step = 1
    second = to_host(bridge._update_inhibitory_conductance_clamp_current()).copy()

    assert bridge.inhibitory_clamp_pathways == (
        "direct_striatonigral",
        "pallidonigral",
    )
    assert bridge.inhibitory_clamp_target_regions == ("snr", "snr")
    np.testing.assert_array_equal(first, np.zeros(5, dtype=np.float32))
    np.testing.assert_array_equal(second[:2], np.zeros(2, dtype=np.float32))
    assert np.all(second[2:] < 0.0)

    expected_density = 0.0
    for channel in _channels():
        event = BiexponentialInhibitoryEvent(
            pathway=channel.pathway,
            tau_rise_ms=channel.tau_rise_ms,
            tau_decay_ms=channel.tau_decay_ms,
            reversal_mV=channel.reversal_mV,
            event_peak_nS=channel.event_peak_nS,
            membrane_area_um2=channel.membrane_area_um2,
        )
        g_nS = InhibitoryBarrage(
            event, EventSchedule.exact(channel.event_times_ms)
        ).conductance_nS(np.asarray([0.05]), np)[0]
        expected_density += (
            g_nS * 100.0 / channel.membrane_area_um2
            * (channel.reversal_mV - (-50.0)) * 1.0e6
        )
    np.testing.assert_allclose(second[2:], expected_density, rtol=2e-6)
    np.testing.assert_array_equal(
        to_host(bridge.cp_conductance_g_i), generic_before
    )


def test_checkpoint_continuation_preserves_clamp_and_hh_state(tmp_path):
    bridge = _build()
    _advance(bridge, 6)
    checkpoint = tmp_path / "inhibitory-clamp.simstate.h5"
    assert bridge.save_checkpoint(str(checkpoint)) is True

    _advance(bridge, 10)
    uninterrupted = _capture(bridge)

    restored = _build()
    assert restored.load_checkpoint(str(checkpoint)) is True
    assert restored.inhibitory_clamp_pathways == bridge.inhibitory_clamp_pathways
    _advance(restored, 10)
    continued = _capture(restored)
    for name, expected in uninterrupted.items():
        np.testing.assert_array_equal(continued[name], expected)

    with h5py.File(checkpoint, "r") as h5f:
        assert h5f.attrs["inhibitory_clamp_state_schema"] == 1
        assert all(name in h5f for name in _INHIBITORY_CLAMP_STATE_ARRAYS)


def test_incomplete_checkpoint_clamp_state_is_rejected(tmp_path):
    bridge = _build()
    checkpoint = tmp_path / "incomplete-clamp.simstate.h5"
    assert bridge.save_checkpoint(str(checkpoint)) is True
    with h5py.File(checkpoint, "r+") as h5f:
        del h5f[_INHIBITORY_CLAMP_STATE_ARRAYS[1]]

    restored = _build()
    assert restored.load_checkpoint(str(checkpoint)) is False


def test_runtime_path_has_no_event_loop_or_host_synchronization():
    source = inspect.getsource(
        SimulationBridge._update_inhibitory_conductance_clamp_current
    )
    assert "for " not in source
    assert ".get(" not in source
    assert "asnumpy" not in source
    assert "_backend_to_host" not in source


def test_active_backend_keeps_runtime_state_and_current_resident():
    xp, backend_name = get_backend()
    bridge = _build()
    bridge._update_inhibitory_conductance_clamp_current()

    assert backend_name in {"numpy", "cupy"}
    for name in (
        *_INHIBITORY_CLAMP_STATE_ARRAYS,
        "cp_inhibitory_clamp_current_pA_equivalent",
        "cp_inhibitory_clamp_event_decay_increments",
        "cp_inhibitory_clamp_target_density",
    ):
        assert isinstance(getattr(bridge, name), xp.ndarray)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"pathway": ""}, "pathway"),
        ({"target_region": ""}, "target_region"),
        ({"tau_rise_ms": 3.0}, "less than"),
        ({"event_times_ms": [0.2, 0.1]}, "sorted"),
    ],
)
def test_channel_config_rejects_ambiguous_or_nonphysical_values(
    overrides, message
):
    values = {
        "pathway": "pallidonigral",
        "target_region": "snr",
        "tau_rise_ms": 0.4,
        "tau_decay_ms": 2.1,
        "reversal_mV": -70.0,
        "event_peak_nS": 2.0,
        "membrane_area_um2": 2000.0,
        "event_times_ms": [0.0],
    }
    values.update(overrides)
    with pytest.raises((TypeError, ValueError), match=message):
        InhibitoryConductanceClampConfig(**values)
