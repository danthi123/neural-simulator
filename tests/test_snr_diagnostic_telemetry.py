import numpy as np

from sim.kernels import (
    fused_snr_diagnostic_capture_into,
    fused_snr_packet_diagnostic_currents,
)


def test_diagnostic_capture_writes_every_state_without_aliasing():
    width = 3
    states = [np.arange(width, dtype=np.float32) + index for index in range(14)]
    states[2] = np.array([False, True, False], dtype=bool)
    outputs = [np.full(width, np.nan, dtype=np.float32) for _ in range(14)]
    outputs[2] = np.zeros(width, dtype=bool)

    fused_snr_diagnostic_capture_into(*states, *outputs)

    for source, output in zip(states, outputs, strict=True):
        np.testing.assert_array_equal(output, source)
        assert not np.shares_memory(output, source)


def test_diagnostic_current_decomposition_matches_hand_calculation():
    voltage = np.array([-50.0, -40.0], dtype=np.float32)
    m = np.array([0.2, 0.3], dtype=np.float32)
    h = np.array([0.8, 0.7], dtype=np.float32)
    n = np.array([0.4, 0.5], dtype=np.float32)
    nap_m = np.array([0.25, 0.35], dtype=np.float32)
    nap_h = np.array([0.75, 0.65], dtype=np.float32)
    cav_m = np.array([0.1, 0.2], dtype=np.float32)
    cav_h = np.array([0.9, 0.8], dtype=np.float32)
    sk = np.array([0.2, 0.4], dtype=np.float32)
    hcn = np.array([0.3, 0.1], dtype=np.float32)
    conductances = [
        np.array([100.0, 110.0], dtype=np.float32),
        np.array([20.0, 22.0], dtype=np.float32),
        np.array([0.1, 0.2], dtype=np.float32),
        np.array([0.03, 0.04], dtype=np.float32),
        np.array([0.05, 0.06], dtype=np.float32),
        np.array([0.07, 0.08], dtype=np.float32),
        np.array([0.09, 0.10], dtype=np.float32),
        np.array([0.11, 0.12], dtype=np.float32),
    ]
    reversals = [
        np.array([55.0, 55.0], dtype=np.float32),
        np.array([-90.0, -90.0], dtype=np.float32),
        np.array([-65.0, -65.0], dtype=np.float32),
        np.array([-20.0, -20.0], dtype=np.float32),
        np.array([120.0, 120.0], dtype=np.float32),
        np.array([-35.0, -35.0], dtype=np.float32),
    ]
    power = np.array([1.0, 2.0], dtype=np.float32)

    observed = fused_snr_packet_diagnostic_currents(
        voltage, m, h, n, nap_m, nap_h, cav_m, cav_h, sk, hcn,
        *conductances, *reversals, power,
    )
    expected = (
        conductances[0] * m**3 * h * (voltage - reversals[0]),
        conductances[1] * n**4 * (voltage - reversals[1]),
        conductances[2] * (voltage - reversals[2]),
        conductances[3] * (voltage - reversals[3]),
        conductances[4] * nap_m * nap_h * (voltage - reversals[0]),
        conductances[5] * cav_m**power * cav_h * (voltage - reversals[4]),
        conductances[6] * sk * (voltage - reversals[1]),
        conductances[7] * hcn * (voltage - reversals[5]),
    )
    expected_total = sum(expected)
    for actual, wanted in zip(observed[:-1], expected, strict=True):
        np.testing.assert_allclose(actual, wanted, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(observed[-1], expected_total, rtol=1e-6, atol=1e-6)


def test_zeroed_nap_conductance_produces_exact_zero_current():
    one = np.ones(4, dtype=np.float32)
    zero = np.zeros(4, dtype=np.float32)
    currents = fused_snr_packet_diagnostic_currents(
        -50.0 * one, one, one, one, one, one, one, one, one, one,
        one, one, one, one, zero, one, one, one,
        50.0 * one, -90.0 * one, -65.0 * one, -20.0 * one,
        120.0 * one, -35.0 * one, one,
    )
    np.testing.assert_array_equal(currents[4], zero)
