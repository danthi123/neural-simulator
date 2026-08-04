"""End-to-end exactness tests for direct-output HH+SNr fusion."""

import os

os.environ.setdefault("SIM_BACKEND", "cupy")

import numpy as np
import pytest

from sim.backend import get_backend, to_host
from sim.bridge import SimulationBridge, _SNR_CONDUCTANCE_ARRAYS
from sim.config import CoreSimConfig, GPUConfig, RuntimeState, VisualizationConfig
from sim.enums import NeuronModel
from sim.regions import BrainRegion


cp, _BACKEND_NAME = get_backend()
pytestmark = pytest.mark.skipif(_BACKEND_NAME != "cupy", reason="CuPy-only bridge path")


def _build(*, fused: bool):
    region = BrainRegion(
        name="snr",
        n_neurons=32,
        internal_density=0.0,
        snr_g_nalcn_max=0.01,
        snr_g_nap_max=0.02,
        snr_g_ca_max=0.03,
        snr_g_sk_max=0.04,
        snr_g_h_max=0.005,
    )
    config = CoreSimConfig(
        num_neurons=32,
        connections_per_neuron=0,
        seed=20260804,
        neuron_model_type=NeuronModel.HODGKIN_HUXLEY.name,
        default_neuron_type_hh="HH_EXCITATORY_DEFAULT_LEGACY",
        dt_ms=0.05,
        enable_brain_region_framework=True,
        brain_regions=[region],
        enable_parameter_heterogeneity=False,
        enable_conductance_noise=False,
        enable_hebbian_learning=False,
        enable_short_term_plasticity=False,
        enable_structural_plasticity=False,
        enable_ou_process=False,
        hh_external_drive_scale=0.0,
        enable_snr_direct_outputs=fused,
    )
    bridge = SimulationBridge(
        core_config=config,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(enable_profiling=False, stats_sync_interval_steps=10_000),
    )
    bridge._initialize_simulation_data()
    assert bridge.is_initialized
    assert bridge._snr_direct_outputs_can_dispatch(config) is fused
    return bridge


def _capture(bridge, steps):
    raster = []
    for _ in range(steps):
        bridge._run_one_simulation_step()
        raster.append(to_host(bridge.cp_firing_states).copy())
    return {
        "raster": np.stack(raster),
        "V": to_host(bridge.cp_membrane_potential_v).copy(),
        "m": to_host(bridge.cp_gating_variable_m).copy(),
        "h": to_host(bridge.cp_gating_variable_h).copy(),
        "n": to_host(bridge.cp_gating_variable_n).copy(),
        **{
            name: to_host(getattr(bridge, name)).copy()
            for name in _SNR_CONDUCTANCE_ARRAYS
        },
    }


def _assert_exact(reference, candidate):
    assert reference.keys() == candidate.keys()
    for name in reference:
        np.testing.assert_array_equal(candidate[name], reference[name], err_msg=name)


def test_bridge_path_and_checkpoint_continuation_are_byte_identical(tmp_path):
    reference = _build(fused=False)
    candidate = _build(fused=True)
    try:
        _assert_exact(_capture(reference, 64), _capture(candidate, 64))

        checkpoint = tmp_path / "hh-snr-megakernel.simstate.h5"
        assert candidate.save_checkpoint(str(checkpoint)) is True
        uninterrupted = _capture(candidate, 32)

        restored = _build(fused=True)
        try:
            assert restored.load_checkpoint(str(checkpoint)) is True
            continued = _capture(restored, 32)
            _assert_exact(uninterrupted, continued)
        finally:
            restored.clear_simulation_state_and_gpu_memory()
    finally:
        reference.clear_simulation_state_and_gpu_memory()
        candidate.clear_simulation_state_and_gpu_memory()
        cp.get_default_memory_pool().free_all_blocks()
