"""Oracle tests for the fully parameterized packet SNr fused kernel."""

import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np
import pytest

from sim.backend import get_backend, to_host
from sim.kernels import (
    fused_snr_packet_conductance_update,
    fused_snr_packet_conductance_update_into,
)
from sim.snr_channel_parameters import (
    CalciumParameters,
    EvidenceClass,
    ParameterProvenance,
    SKParameters,
    _phillips_tau,
    _sigmoid,
    calcium_concentration_step,
    calcium_influx_delta_um,
    first_order_gate_step,
    sk_activation_step,
)


xp, BACKEND_NAME = get_backend()


def _array(values):
    return xp.asarray(values, dtype=xp.float32)


def _packet_inputs(size=19):
    """Return an input tuple with independently varying packet parameters."""

    rng = np.random.default_rng(20260804)
    linspace = lambda low, high: _array(np.linspace(low, high, size, dtype=np.float32))
    uniform = lambda low, high: _array(rng.uniform(low, high, size).astype(np.float32))
    states = (
        uniform(0.05, 0.95),
        uniform(0.05, 0.95),
        uniform(0.05, 0.95),
        uniform(0.05, 0.95),
        uniform(0.05, 1.25),
        uniform(0.05, 0.95),
        uniform(0.05, 0.95),
    )
    parameters = (
        linspace(-54.0, -46.0),  # NaP activation half/slope.
        linspace(2.8, 4.1),
        linspace(0.02, 0.05),
        linspace(0.12, 0.19),
        linspace(-45.0, -40.0),
        linspace(12.0, 16.0),
        linspace(-16.0, -12.0),
        linspace(-60.0, -54.0),  # NaP inactivation half/slope.
        linspace(-5.2, -3.6),
        linspace(8.0, 11.0),
        linspace(16.0, 21.0),
        linspace(-36.0, -31.0),
        linspace(24.0, 29.0),
        linspace(-34.0, -29.0),
        linspace(-30.0, -24.0),  # Cav2.2 activation half/slope/tau.
        linspace(2.4, 3.8),
        linspace(0.35, 0.8),
        linspace(-56.0, -49.0),  # Cav2.2 inactivation half/slope/tau.
        linspace(-6.2, -4.4),
        linspace(12.0, 24.0),
        linspace(2.0, 4.0),
        linspace(-79.0, -71.0),  # HCN half/slope/tau.
        linspace(-6.6, -4.8),
        linspace(65.0, 130.0),
        linspace(0.04, 0.10),  # Physical calcium parameters.
        linspace(35.0, 75.0),
        linspace(0.001, 0.006),
        linspace(0.25, 0.7),  # SK half/hill/open/close taus.
        linspace(2.0, 5.0),
        linspace(1.0, 3.0),
        linspace(5.0, 11.0),
        linspace(0.01, 0.08),  # Conductances.
        linspace(0.02, 0.15),
        linspace(0.03, 0.22),
        linspace(0.01, 0.12),
        linspace(0.01, 0.08),
        linspace(-15.0, -5.0),  # Reversals.
        linspace(45.0, 55.0),
        linspace(105.0, 125.0),
        linspace(-95.0, -85.0),
        linspace(-38.0, -24.0),
    )
    return (linspace(-95.0, 28.0), *states, xp.float32(0.075), *parameters)


def _host_inputs(inputs):
    return tuple(np.asarray(to_host(value), dtype=np.float32) for value in inputs)


def _reference_packet_step(inputs):
    """Independent NumPy oracle composed from channel-parameter equations."""

    (
        voltage,
        nap_m_old,
        nap_h_old,
        cav_m_old,
        cav_h_old,
        calcium_old,
        sk_old,
        hcn_old,
        dt,
        nap_m_half,
        nap_m_slope,
        nap_m_tau_min,
        nap_m_tau_max,
        nap_m_tau_half,
        nap_m_tau_sigma0,
        nap_m_tau_sigma1,
        nap_h_half,
        nap_h_slope,
        nap_h_tau_min,
        nap_h_tau_max,
        nap_h_tau_half,
        nap_h_tau_sigma0,
        nap_h_tau_sigma1,
        cav_m_half,
        cav_m_slope,
        cav_m_tau,
        cav_h_half,
        cav_h_slope,
        cav_h_tau,
        cav_power,
        hcn_half,
        hcn_slope,
        hcn_tau,
        calcium_baseline,
        calcium_decay_tau,
        calcium_influx,
        sk_half,
        sk_hill,
        sk_open_tau,
        sk_close_tau,
        g_nalcn,
        g_nap,
        g_cav,
        g_sk,
        g_hcn,
        e_nalcn,
        e_na,
        e_ca,
        e_k,
        e_hcn,
    ) = _host_inputs(inputs)

    nap_m_inf = _sigmoid(voltage, nap_m_half, nap_m_slope, np)
    nap_h_inf = _sigmoid(voltage, nap_h_half, nap_h_slope, np)
    nap_m_tau = _phillips_tau(
        voltage, nap_m_tau_min, nap_m_tau_max, nap_m_tau_half,
        nap_m_tau_sigma0, nap_m_tau_sigma1, np,
    )
    nap_h_tau = _phillips_tau(
        voltage, nap_h_tau_min, nap_h_tau_max, nap_h_tau_half,
        nap_h_tau_sigma0, nap_h_tau_sigma1, np,
    )
    nap_m = np.clip(first_order_gate_step(nap_m_old, nap_m_inf, float(dt), nap_m_tau), 0.0, 1.0)
    nap_h = np.clip(first_order_gate_step(nap_h_old, nap_h_inf, float(dt), nap_h_tau), 0.0, 1.0)

    cav_m_inf = _sigmoid(voltage, cav_m_half, cav_m_slope, np)
    cav_h_inf = _sigmoid(voltage, cav_h_half, cav_h_slope, np)
    cav_m = np.clip(first_order_gate_step(cav_m_old, cav_m_inf, float(dt), cav_m_tau), 0.0, 1.0)
    cav_h = np.clip(first_order_gate_step(cav_h_old, cav_h_inf, float(dt), cav_h_tau), 0.0, 1.0)
    hcn_inf = _sigmoid(voltage, hcn_half, hcn_slope, np)
    hcn = np.clip(first_order_gate_step(hcn_old, hcn_inf, float(dt), hcn_tau), 0.0, 1.0)

    i_nalcn = g_nalcn * (voltage - e_nalcn)
    i_nap = g_nap * nap_m * nap_h * (voltage - e_na)
    i_cav = g_cav * np.power(cav_m, cav_power) * cav_h * (voltage - e_ca)
    calcium_target = calcium_baseline + calcium_decay_tau * calcium_influx * np.maximum(-i_cav, 0.0)
    calcium = calcium_target + (calcium_old - calcium_target) * np.exp(-float(dt) / calcium_decay_tau)
    calcium = np.maximum(calcium, 0.0)

    calcium_hill = np.power(calcium, sk_hill)
    sk_inf = calcium_hill / (calcium_hill + np.power(sk_half, sk_hill))
    sk_tau = np.where(sk_inf >= sk_old, sk_open_tau, sk_close_tau)
    sk = np.clip(first_order_gate_step(sk_old, sk_inf, float(dt), sk_tau), 0.0, 1.0)
    current = i_nalcn + i_nap + i_cav + g_sk * sk * (voltage - e_k) + g_hcn * hcn * (voltage - e_hcn)
    return nap_m, nap_h, cav_m, cav_h, calcium, sk, hcn, current


def test_packet_kernel_matches_channel_parameter_equation_oracle_with_array_parameters():
    inputs = _packet_inputs()
    actual = fused_snr_packet_conductance_update(*inputs)
    expected = _reference_packet_step(inputs)

    for expected_value, actual_value in zip(expected, actual):
        np.testing.assert_allclose(
            to_host(actual_value), expected_value, rtol=5e-6, atol=2e-6
        )
    for gate in (*actual[:4], actual[5], actual[6]):
        host_gate = to_host(gate)
        assert np.all(np.isfinite(host_gate))
        assert np.all((host_gate >= 0.0) & (host_gate <= 1.0))
    assert np.all(np.isfinite(to_host(actual[4])))
    assert np.all(to_host(actual[4]) >= 0.0)


def test_packet_kernel_clamps_stale_dynamic_states_to_finite_domains():
    inputs = list(_packet_inputs(size=4))
    inputs[1] = _array([-4.0, 5.0, np.nan, 0.5])
    inputs[2] = _array([5.0, -4.0, np.nan, 0.5])
    inputs[3] = _array([-4.0, 5.0, np.nan, 0.5])
    inputs[4] = _array([5.0, -4.0, np.nan, 0.5])
    inputs[5] = _array([-1.0, np.nan, 4.0, 0.5])
    inputs[6] = _array([-4.0, 5.0, np.nan, 0.5])
    inputs[7] = _array([5.0, -4.0, np.nan, 0.5])

    actual = fused_snr_packet_conductance_update(*inputs)

    for gate in (*actual[:4], actual[5], actual[6]):
        host_gate = to_host(gate)
        assert np.all(np.isfinite(host_gate))
        assert np.all((host_gate >= 0.0) & (host_gate <= 1.0))
    calcium = to_host(actual[4])
    assert np.all(np.isfinite(calcium))
    assert np.all(calcium >= 0.0)


def _physical_provenance():
    return ParameterProvenance(
        parameter_set_id="packet-kernel-test-v1",
        evidence_class=EvidenceClass.MEASURED,
        source_locator="test fixture",
        reference_temperature_celsius=37.0,
    )


def test_packet_kernel_uses_physical_calcium_oracle_asymmetric_sk_and_cav_power():
    inputs = list(_packet_inputs(size=2))
    area_um2, volume_um3 = 875.0, 150.0
    coefficient = float(
        calcium_influx_delta_um(
            1.0, 1.0, membrane_area_um2=area_um2, accessible_volume_um3=volume_um3
        )
    )
    # A depolarized opening case and hyperpolarized closing case force distinct
    # SK tau branches.  The Cav2.2 activation power is deliberately three.
    inputs[0] = _array([-15.0, -100.0])
    inputs[5] = _array([0.08, 0.45])
    inputs[6] = _array([0.10, 0.90])
    inputs[8] = xp.float32(0.8)
    inputs[29] = _array([3.0, 3.0])
    inputs[33] = _array([0.04, 0.04])
    inputs[34] = _array([40.0, 40.0])
    inputs[35] = _array([coefficient, coefficient])
    inputs[36] = _array([0.45, 0.45])
    inputs[37] = _array([4.0, 4.0])
    inputs[38] = _array([1.25, 1.25])
    inputs[39] = _array([8.5, 8.5])
    inputs[40] = _array([0.0, 0.0])
    inputs[41] = _array([0.0, 0.0])
    inputs[42] = _array([0.45, 0.45])
    inputs[43] = _array([0.9, 0.9])
    inputs[44] = _array([0.0, 0.0])
    inputs[45] = _array([0.0, 0.0])
    actual = fused_snr_packet_conductance_update(*inputs)
    expected = _reference_packet_step(tuple(inputs))

    host = _host_inputs(tuple(inputs))
    voltage, calcium_old, sk_old, dt = host[0], host[5], host[6], float(host[8])
    i_cav = (
        host[42]
        * np.power(expected[2], host[29])
        * expected[3]
        * (voltage - host[47])
    )
    calcium_parameters = CalciumParameters(
        _physical_provenance(), 0.04, 40.0, 2000.0
    )
    physical_calcium = calcium_concentration_step(
        calcium_old,
        np.maximum(-i_cav, 0.0),
        dt,
        calcium_parameters,
        membrane_area_um2=area_um2,
        accessible_volume_um3=volume_um3,
    )
    sk_parameters = SKParameters(_physical_provenance(), 0.45, 4.0, 1.25, 8.5)
    physical_sk = sk_activation_step(
        sk_old, physical_calcium, dt, sk_parameters, calcium_units="micromolar"
    )

    # The direct physical oracle requires the same conversion coefficient the
    # packet supplies.  Make that equality explicit before testing the kernel.
    np.testing.assert_allclose(host[35], coefficient, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(to_host(actual[4]), physical_calcium, rtol=5e-6, atol=2e-6)
    np.testing.assert_allclose(to_host(actual[5]), physical_sk, rtol=5e-6, atol=2e-6)
    np.testing.assert_allclose(to_host(actual[2]), expected[2], rtol=5e-6, atol=2e-6)
    np.testing.assert_allclose(to_host(actual[7]), expected[7], rtol=5e-6, atol=2e-6)

    expected_sk_inf = np.power(physical_calcium, 4.0) / (
        np.power(physical_calcium, 4.0) + 0.45 ** 4.0
    )
    assert expected_sk_inf[0] > sk_old[0]
    assert expected_sk_inf[1] < sk_old[1]
    assert not np.isclose(expected[7][0], (
        host[40][0] * (voltage[0] - host[45][0])
        + host[41][0] * expected[0][0] * expected[1][0] * (voltage[0] - host[46][0])
        + host[42][0] * expected[2][0] ** 2 * expected[3][0] * (voltage[0] - host[47][0])
        + host[43][0] * expected[5][0] * (voltage[0] - host[48][0])
        + host[44][0] * expected[6][0] * (voltage[0] - host[49][0])
    ))


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="CuPy-only direct output reuse")
def test_packet_kernel_direct_outputs_reuse_the_compiled_fusion_graph():
    inputs = _packet_inputs(size=23)
    outputs = tuple(xp.empty_like(inputs[0]) for _ in range(8))
    fused_snr_packet_conductance_update.clear_cache()

    assert fused_snr_packet_conductance_update_into(inputs, outputs) is False
    expected = fused_snr_packet_conductance_update(*inputs)
    for expected_value, output in zip(expected, outputs):
        np.testing.assert_array_equal(to_host(output), to_host(expected_value))

    assert fused_snr_packet_conductance_update_into(inputs, outputs) is True
    expected = fused_snr_packet_conductance_update(*inputs)
    for expected_value, output in zip(expected, outputs):
        np.testing.assert_array_equal(to_host(output), to_host(expected_value))
