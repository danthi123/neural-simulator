"""Fail-closed dispatch tests for direct-output HH+SNr fusion."""

from types import SimpleNamespace

import numpy as np
import pytest

import sim.bridge as bridge_module
from sim.bridge import SimulationBridge
from sim.config import CoreSimConfig
from sim.enums import NeuronModel


def _config(**overrides):
    values = {
        "enable_snr_direct_outputs": True,
        "neuron_model_type": NeuronModel.HODGKIN_HUXLEY.name,
        "enable_conductance_noise": False,
        "hh_g_M_max": 0.0,
        "hh_g_CaT_max": 0.0,
        "hh_g_h_max": 0.0,
        "hh_g_NaP_max": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _bridge():
    bridge = object.__new__(SimulationBridge)
    bridge.cp_snr_g_nalcn_max = object()
    return bridge


def test_snr_direct_outputs_is_opt_in():
    assert CoreSimConfig().enable_snr_direct_outputs is False


def test_dispatch_accepts_only_the_complete_supported_regime(monkeypatch):
    monkeypatch.setattr(bridge_module, "is_gpu_backend", lambda: True)
    assert _bridge()._snr_direct_outputs_can_dispatch(_config()) is True


@pytest.mark.parametrize(
    "override",
    [
        {"enable_snr_direct_outputs": False},
        {"neuron_model_type": NeuronModel.IZHIKEVICH.name},
        {"enable_conductance_noise": True},
        {"hh_g_M_max": 0.01},
        {"hh_g_CaT_max": 0.01},
        {"hh_g_h_max": 0.01},
        {"hh_g_NaP_max": 0.01},
    ],
)
def test_dispatch_rejects_unfused_or_stochastic_regimes(monkeypatch, override):
    monkeypatch.setattr(bridge_module, "is_gpu_backend", lambda: True)
    assert _bridge()._snr_direct_outputs_can_dispatch(_config(**override)) is False


def test_dispatch_rejects_cpu_or_missing_snr_bundle(monkeypatch):
    bridge = _bridge()
    monkeypatch.setattr(bridge_module, "is_gpu_backend", lambda: False)
    assert bridge._snr_direct_outputs_can_dispatch(_config()) is False

    monkeypatch.setattr(bridge_module, "is_gpu_backend", lambda: True)
    bridge.cp_snr_g_nalcn_max = None
    assert bridge._snr_direct_outputs_can_dispatch(_config()) is False


def test_bridge_helper_passes_persistent_state_in_place_and_detects_crossing(monkeypatch):
    bridge = object.__new__(SimulationBridge)
    bridge.cp_membrane_potential_v = np.array([-1.0, 1.0], dtype=np.float32)
    bridge.cp_gating_variable_m = np.zeros(2, dtype=np.float32)
    bridge.cp_gating_variable_h = np.zeros(2, dtype=np.float32)
    bridge.cp_gating_variable_n = np.zeros(2, dtype=np.float32)
    bridge.cp_firing_states = np.zeros(2, dtype=bool)
    bridge.cp_snr_ionic_current_scratch = np.zeros(2, dtype=np.float32)
    for name in (
        "nap_activation", "nap_inactivation", "ca_activation", "ca_inactivation",
        "calcium", "sk_activation", "h_activation",
    ):
        setattr(bridge, f"cp_snr_{name}", np.zeros(2, dtype=np.float32))
    for name in ("C_m", "g_Na_max", "g_K_max", "g_L", "E_Na", "E_K", "E_L", "v_peak"):
        setattr(bridge, f"cp_hh_{name}", np.ones(2, dtype=np.float32))
    for name in ("g_nalcn_max", "g_nap_max", "g_ca_max", "g_sk_max", "g_h_max"):
        setattr(bridge, f"cp_snr_{name}", np.ones(2, dtype=np.float32))
    bridge._cached_hh_phi_m = bridge._cached_hh_phi_h = bridge._cached_hh_phi_n = 1.0
    cfg = SimpleNamespace(
        snr_E_nalcn=0.0, snr_E_nap=50.0, snr_E_ca=120.0,
        snr_E_sk=-90.0, snr_E_h=-30.0, snr_calcium_baseline=0.0,
        snr_calcium_influx_scale=0.01, snr_calcium_decay_tau_ms=80.0,
        snr_sk_calcium_half=0.5, snr_sk_hill_coefficient=4.0,
        snr_sk_activation_tau_ms=5.0,
    )
    observed = {}

    def snr_kernel(inputs, outputs):
        observed["snr_inputs"] = inputs
        outputs[-1][:] = 0.0

    def hh_kernel(*inputs):
        observed["hh_inputs"] = inputs
        out_v, out_m, out_h, out_n, out_fired = inputs[-5:]
        out_v[:] = np.array([2.0, 0.0], dtype=np.float32)
        out_m[:] = bridge.cp_gating_variable_m
        out_h[:] = bridge.cp_gating_variable_h
        out_n[:] = bridge.cp_gating_variable_n
        out_fired[:] = np.array([True, False])

    monkeypatch.setattr(bridge_module, "fused_snr_conductance_update_into", snr_kernel)
    monkeypatch.setattr(bridge_module, "fused_hh_state_and_spike_update_into", hh_kernel)
    current = np.zeros(2, dtype=np.float32)
    fired = bridge._run_snr_direct_outputs(cfg, current, 0.05)

    assert observed["snr_inputs"][0] is bridge.cp_membrane_potential_v
    assert observed["hh_inputs"][4] is not current
    assert observed["hh_inputs"][-1] is bridge.cp_firing_states
    np.testing.assert_array_equal(fired, np.array([True, False]))
