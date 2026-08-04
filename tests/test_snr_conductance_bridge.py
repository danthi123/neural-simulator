import h5py
import numpy as np

from sim.backend import to_host
from sim.bridge import SimulationBridge, _SNR_CONDUCTANCE_ARRAYS
from sim.config import CoreSimConfig, GPUConfig, RuntimeState, VisualizationConfig
from sim.enums import NeuronModel
from sim.regions import BrainRegion


def _region(name, n, **conductances):
    return BrainRegion(
        name=name,
        n_neurons=n,
        internal_density=0.0,
        **conductances,
    )


def _build(regions, seed=193883):
    config = CoreSimConfig(
        num_neurons=sum(region.n_neurons for region in regions),
        connections_per_neuron=0,
        seed=seed,
        neuron_model_type=NeuronModel.HODGKIN_HUXLEY.name,
        default_neuron_type_hh="HH_EXCITATORY_DEFAULT_LEGACY",
        dt_ms=0.05,
        enable_brain_region_framework=True,
        brain_regions=regions,
        enable_parameter_heterogeneity=False,
        enable_hebbian_learning=False,
        enable_short_term_plasticity=False,
        enable_structural_plasticity=False,
        enable_ou_process=False,
        enable_conductance_noise=False,
        hh_external_drive_scale=0.0,
    )
    bridge = SimulationBridge(
        core_config=config,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(enable_profiling=False),
    )
    bridge._initialize_simulation_data()
    assert bridge.is_initialized
    return bridge


def _active_region():
    return _region(
        "snr",
        4,
        snr_g_nalcn_max=0.01,
        snr_g_nap_max=0.02,
        snr_g_ca_max=0.03,
        snr_g_sk_max=0.04,
        snr_g_h_max=0.005,
    )


def test_bundle_default_off_allocates_no_arrays():
    bridge = _build([_region("control", 6)])
    assert all(getattr(bridge, name) is None for name in _SNR_CONDUCTANCE_ARRAYS)


def test_bundle_maxima_are_region_scoped_and_states_start_at_equilibrium():
    bridge = _build([_region("control", 3), _active_region()])
    assert sum(
        getattr(bridge, name).nbytes for name in _SNR_CONDUCTANCE_ARRAYS
    ) == 48 * bridge.core_config.num_neurons
    expected = {
        "cp_snr_g_nalcn_max": 0.01,
        "cp_snr_g_nap_max": 0.02,
        "cp_snr_g_ca_max": 0.03,
        "cp_snr_g_sk_max": 0.04,
        "cp_snr_g_h_max": 0.005,
    }
    for name, value in expected.items():
        array = to_host(getattr(bridge, name))
        np.testing.assert_array_equal(array[:3], np.zeros(3, dtype=np.float32))
        np.testing.assert_allclose(array[3:], value, rtol=0.0, atol=1e-7)

    before = {
        name: to_host(getattr(bridge, name)).copy()
        for name in _SNR_CONDUCTANCE_ARRAYS
    }
    bridge._run_one_simulation_step()
    for name in _SNR_CONDUCTANCE_MAX_ARRAY_NAMES:
        np.testing.assert_array_equal(to_host(getattr(bridge, name)), before[name])
    assert all(
        np.all(np.isfinite(to_host(getattr(bridge, name))))
        for name in _SNR_CONDUCTANCE_ARRAYS
    )


_SNR_CONDUCTANCE_MAX_ARRAY_NAMES = (
    "cp_snr_g_nalcn_max",
    "cp_snr_g_nap_max",
    "cp_snr_g_ca_max",
    "cp_snr_g_sk_max",
    "cp_snr_g_h_max",
)


def _capture(bridge, steps):
    raster = []
    for _ in range(steps):
        bridge._run_one_simulation_step()
        raster.append(to_host(bridge.cp_firing_states).copy())
    return {
        "raster": np.stack(raster),
        "voltage": to_host(bridge.cp_membrane_potential_v).copy(),
        **{
            name: to_host(getattr(bridge, name)).copy()
            for name in _SNR_CONDUCTANCE_ARRAYS
        },
    }


def test_bundle_checkpoint_continuation_and_incomplete_state_rejection(tmp_path):
    bridge = _build([_active_region()])
    _capture(bridge, 5)
    checkpoint = tmp_path / "snr-bundle.simstate.h5"
    assert bridge.save_checkpoint(str(checkpoint)) is True
    uninterrupted = _capture(bridge, 10)

    restored = _build([_active_region()])
    assert restored.load_checkpoint(str(checkpoint)) is True
    continued = _capture(restored, 10)
    for name in uninterrupted:
        np.testing.assert_array_equal(continued[name], uninterrupted[name])

    with h5py.File(checkpoint, "r+") as h5f:
        del h5f["cp_snr_sk_activation"]
    malformed = _build([_active_region()])
    assert malformed.load_checkpoint(str(checkpoint)) is False
