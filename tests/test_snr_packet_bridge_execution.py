"""Mixed-region bridge execution tests for authenticated SNr packets."""

from types import MappingProxyType

import h5py
import numpy as np
import pytest

import sim.bridge as bridge_module
from sim.backend import get_backend, to_host
from sim.bridge import SimulationBridge
from sim.config import CoreSimConfig, GPUConfig, RuntimeState, VisualizationConfig
from sim.enums import NeuronModel
from sim.kernels import fused_snr_conductance_update
from sim.regions import BrainRegion
from tests.snr_packet_fixtures import runtime_binding


cp, BACKEND_NAME = get_backend()
_PACKET_PATH = "packets/snr.json"
_PACKET_DIGEST = "a" * 64


def _region(name, n, **overrides):
    return BrainRegion(name=name, n_neurons=n, internal_density=0.0, **overrides)


def _config(*, direct=False):
    regions = [
        _region("control", 2),
        _region(
            "legacy", 2,
            snr_g_nalcn_max=0.01,
            snr_g_nap_max=0.02,
            snr_g_ca_max=0.03,
            snr_g_sk_max=0.04,
            snr_g_h_max=0.005,
        ),
        _region(
            "packet", 3,
            snr_executable_packet_path=_PACKET_PATH,
            snr_executable_packet_sha256=_PACKET_DIGEST,
        ),
    ]
    return CoreSimConfig(
        num_neurons=7,
        connections_per_neuron=0,
        seed=20260804,
        neuron_model_type=NeuronModel.HODGKIN_HUXLEY.name,
        default_neuron_type_hh="HH_EXCITATORY_DEFAULT_LEGACY",
        dt_ms=0.05,
        enable_brain_region_framework=True,
        brain_regions=regions,
        region_pathways=[],
        snr_authority_policy_path="packets/policy.json",
        snr_authority_policy_sha256="b" * 64,
        enable_parameter_heterogeneity=False,
        enable_conductance_noise=False,
        enable_hebbian_learning=False,
        enable_short_term_plasticity=False,
        enable_structural_plasticity=False,
        enable_ou_process=False,
        hh_external_drive_scale=0.0,
        enable_snr_direct_outputs=direct,
    )


def _build(monkeypatch, *, direct=False):
    bindings = MappingProxyType({"packet": runtime_binding(label="live")})
    monkeypatch.setattr(
        bridge_module,
        "load_runtime_snr_packet_bindings",
        lambda config, *, source_root: bindings,
    )
    bridge = SimulationBridge(
        core_config=_config(direct=direct),
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(enable_profiling=False),
        simulation_source_root="/unused/authenticated-root",
    )
    bridge._initialize_simulation_data()
    assert bridge.is_initialized
    return bridge


def _host(value):
    return np.asarray(to_host(value))


def test_mixed_bridge_materializes_packet_arrays_and_preserves_region_authority(monkeypatch):
    bridge = _build(monkeypatch)
    parameters = runtime_binding().runtime_parameters
    packet = slice(4, 7)
    legacy = slice(2, 4)

    assert bridge.cp_snr_packet_parameter_matrix.shape == (36, 7)
    assert set(bridge.snr_packet_kernel_parameters) == set(
        bridge_module._SNR_PACKET_KERNEL_PARAMETER_ORDER
    )
    np.testing.assert_array_equal(
        _host(bridge.cp_snr_g_nalcn_max)[:2], np.zeros(2, np.float32)
    )
    np.testing.assert_allclose(_host(bridge.cp_snr_g_nalcn_max)[legacy], 0.01)
    np.testing.assert_allclose(
        _host(bridge.cp_snr_g_nalcn_max)[packet],
        parameters.nalcn.conductance_density_ms_per_cm2,
    )
    np.testing.assert_allclose(
        _host(bridge.snr_packet_kernel_parameters["nap_activation_half_mv"])[legacy],
        -50.0,
    )
    np.testing.assert_allclose(
        _host(bridge.snr_packet_kernel_parameters["nap_activation_half_mv"])[packet],
        parameters.nap.activation_half_mv,
    )
    np.testing.assert_allclose(
        _host(bridge.snr_packet_kernel_parameters["sk_deactivation_tau_ms"])[packet],
        parameters.sk.deactivation_tau_ms / parameters.q10_factors.sk,
    )
    np.testing.assert_allclose(
        _host(bridge.cp_membrane_potential_v)[packet],
        parameters.fast_hh.initial_voltage_mv,
    )
    np.testing.assert_allclose(
        _host(bridge.cp_hh_E_L)[packet], parameters.ionic_env.leak_reversal_mv
    )
    np.testing.assert_allclose(
        _host(bridge.snr_packet_hh_phi["hh_phi_m"])[packet],
        parameters.q10_factors.fast_hh_sodium_activation,
    )


def test_legacy_slice_of_packet_kernel_matches_existing_legacy_equations(monkeypatch):
    bridge = _build(monkeypatch)
    dt = bridge.core_config.dt_ms
    packet_result = bridge_module.fused_snr_packet_conductance_update(
        *bridge._snr_packet_conductance_inputs(dt)
    )
    expected = fused_snr_conductance_update(
        bridge.cp_membrane_potential_v,
        bridge.cp_snr_nap_activation,
        bridge.cp_snr_nap_inactivation,
        bridge.cp_snr_ca_activation,
        bridge.cp_snr_ca_inactivation,
        bridge.cp_snr_calcium,
        bridge.cp_snr_sk_activation,
        bridge.cp_snr_h_activation,
        dt,
        bridge.cp_snr_g_nalcn_max,
        bridge.cp_snr_g_nap_max,
        bridge.cp_snr_g_ca_max,
        bridge.cp_snr_g_sk_max,
        bridge.cp_snr_g_h_max,
        bridge.core_config.snr_E_nalcn,
        bridge.core_config.snr_E_nap,
        bridge.core_config.snr_E_ca,
        bridge.core_config.snr_E_sk,
        bridge.core_config.snr_E_h,
        bridge.core_config.snr_calcium_baseline,
        bridge.core_config.snr_calcium_influx_scale,
        bridge.core_config.snr_calcium_decay_tau_ms,
        bridge.core_config.snr_sk_calcium_half,
        bridge.core_config.snr_sk_hill_coefficient,
        bridge.core_config.snr_sk_activation_tau_ms,
    )
    for actual, reference in zip(packet_result, expected):
        np.testing.assert_allclose(
            _host(actual)[2:4], _host(reference)[2:4], rtol=2e-6, atol=1e-7
        )


def _capture(bridge, steps):
    raster = []
    for _ in range(steps):
        bridge._run_one_simulation_step()
        raster.append(_host(bridge.cp_firing_states).copy())
    return (
        np.stack(raster),
        _host(bridge.cp_membrane_potential_v).copy(),
        *(
            _host(getattr(bridge, name)).copy()
            for name in bridge_module._SNR_CONDUCTANCE_STATE_ARRAYS
        ),
    )


def test_packet_standard_path_advances_finite_state(monkeypatch):
    bridge = _build(monkeypatch)
    result = _capture(bridge, 8)
    assert all(np.all(np.isfinite(value)) for value in result)


def test_packet_reset_drops_all_authenticated_device_views(monkeypatch):
    bridge = _build(monkeypatch)
    bridge.clear_simulation_state_and_gpu_memory()
    assert bridge.cp_snr_packet_parameter_matrix is None
    assert bridge.cp_snr_packet_hh_phi_matrix is None
    assert bridge.snr_packet_kernel_parameters == {}
    assert bridge.snr_packet_hh_phi == {}
    assert bridge.snr_packet_bindings == {}


def test_packet_checkpoint_regenerates_immutable_arrays_and_continues_exactly(
    monkeypatch, tmp_path
):
    manifest = b"authenticated-packet-manifest-v1"
    monkeypatch.setattr(
        bridge_module, "runtime_binding_manifest_bytes", lambda bindings: manifest
    )
    bridge = _build(monkeypatch)
    _capture(bridge, 5)
    immutable_before = {
        name: _host(value).copy()
        for name, value in bridge.snr_packet_kernel_parameters.items()
    }
    checkpoint = tmp_path / "packet-v2.simstate.h5"
    assert bridge.save_checkpoint(str(checkpoint)) is True

    with h5py.File(checkpoint, "r") as h5f:
        assert h5f.attrs["snr_conductance_state_schema"] == 2
        assert set(bridge_module._SNR_CONDUCTANCE_STATE_ARRAYS) <= set(h5f)
        assert not (set(bridge_module._SNR_CONDUCTANCE_MAX_ARRAYS) & set(h5f))
        assert "cp_snr_packet_parameter_matrix" not in h5f
        assert h5f.attrs["snr_packet_hh_parameters_masked"]
        for name in (
            "cp_hh_C_m", "cp_hh_g_Na_max", "cp_hh_g_K_max", "cp_hh_g_L",
            "cp_hh_E_Na", "cp_hh_E_K", "cp_hh_E_L", "cp_hh_v_peak",
        ):
            np.testing.assert_array_equal(h5f[name][4:7], np.zeros(3, np.float32))

    uninterrupted = _capture(bridge, 12)
    restored = _build(monkeypatch)
    assert restored.load_checkpoint(str(checkpoint)) is True
    for name, expected in immutable_before.items():
        np.testing.assert_array_equal(
            _host(restored.snr_packet_kernel_parameters[name]), expected
        )
    continued = _capture(restored, 12)
    for reference, candidate in zip(uninterrupted, continued):
        np.testing.assert_array_equal(candidate, reference)


@pytest.mark.parametrize(
    "tamper",
    ("missing_dynamic", "wrong_dtype", "gate_domain", "immutable_injection", "hh_exposure"),
)
def test_packet_checkpoint_v2_tampering_fails_closed(monkeypatch, tmp_path, tamper):
    manifest = b"authenticated-packet-manifest-v1"
    monkeypatch.setattr(
        bridge_module, "runtime_binding_manifest_bytes", lambda bindings: manifest
    )
    source = _build(monkeypatch)
    checkpoint = tmp_path / f"packet-v2-{tamper}.simstate.h5"
    assert source.save_checkpoint(str(checkpoint)) is True

    with h5py.File(checkpoint, "r+") as h5f:
        if tamper == "missing_dynamic":
            del h5f["cp_snr_sk_activation"]
        elif tamper == "wrong_dtype":
            values = h5f["cp_snr_calcium"][:].astype(np.float64)
            del h5f["cp_snr_calcium"]
            h5f.create_dataset("cp_snr_calcium", data=values)
        elif tamper == "gate_domain":
            h5f["cp_snr_h_activation"][0] = np.float32(1.5)
        elif tamper == "immutable_injection":
            h5f.create_dataset(
                "cp_snr_g_nalcn_max", data=np.zeros(7, dtype=np.float32)
            )
        else:
            h5f["cp_hh_C_m"][4] = np.float32(1.2)

    restored = _build(monkeypatch)
    assert restored.load_checkpoint(str(checkpoint)) is False


def test_pre_dynamics_packet_checkpoint_migrates_without_continuation_claim(
    monkeypatch, tmp_path
):
    manifest = b"authenticated-packet-manifest-v1"
    monkeypatch.setattr(
        bridge_module, "runtime_binding_manifest_bytes", lambda bindings: manifest
    )
    source = _build(monkeypatch)
    checkpoint = tmp_path / "packet-pre-dynamics.simstate.h5"
    assert source.save_checkpoint(str(checkpoint)) is True
    with h5py.File(checkpoint, "r+") as h5f:
        for name in bridge_module._SNR_CONDUCTANCE_STATE_ARRAYS:
            del h5f[name]
        del h5f.attrs["snr_conductance_state_schema"]
        del h5f.attrs["snr_packet_hh_parameters_masked"]

    restored = _build(monkeypatch)
    assert restored.load_checkpoint(str(checkpoint)) is True
    assert restored.cp_snr_packet_parameter_matrix.shape == (36, 7)
    assert all(
        np.all(np.isfinite(_host(getattr(restored, name))))
        for name in bridge_module._SNR_CONDUCTANCE_STATE_ARRAYS
    )


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="CuPy direct-output path")
def test_packet_standard_and_direct_output_paths_are_exact(monkeypatch):
    standard = _build(monkeypatch, direct=False)
    direct = _build(monkeypatch, direct=True)
    try:
        for reference, candidate in zip(_capture(standard, 32), _capture(direct, 32)):
            np.testing.assert_array_equal(candidate, reference)
    finally:
        standard.clear_simulation_state_and_gpu_memory()
        direct.clear_simulation_state_and_gpu_memory()
        cp.get_default_memory_pool().free_all_blocks()
