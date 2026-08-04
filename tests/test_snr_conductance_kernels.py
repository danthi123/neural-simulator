"""Focused equation tests for the V14 Stage-A SNr conductance bundle."""

import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np
import pytest

from sim.kernels import fused_snr_conductance_update
from sim.backend import get_backend, to_host


xp, _BACKEND_NAME = get_backend()


def _state(size):
    return tuple(xp.zeros(size, dtype=xp.float32) for _ in range(7))


def _update(
    voltage,
    state=None,
    *,
    dt=0.05,
    g_nalcn=0.0,
    g_nap=0.0,
    g_ca=0.0,
    g_sk=0.0,
    g_h=0.0,
    calcium_baseline=0.0,
    calcium_influx_scale=0.01,
    calcium_decay_tau_ms=50.0,
    sk_half_activation=0.5,
    sk_tau_ms=5.0,
):
    voltage = xp.asarray(voltage, dtype=xp.float32)
    if state is None:
        state = _state(voltage.size)
    return fused_snr_conductance_update(
        voltage,
        *state,
        xp.float32(dt),
        xp.float32(g_nalcn),
        xp.float32(g_nap),
        xp.float32(g_ca),
        xp.float32(g_sk),
        xp.float32(g_h),
        xp.float32(-10.0),
        xp.float32(50.0),
        xp.float32(120.0),
        xp.float32(-90.0),
        xp.float32(-30.0),
        xp.float32(calcium_baseline),
        xp.float32(calcium_influx_scale),
        xp.float32(calcium_decay_tau_ms),
        xp.float32(sk_half_activation),
        xp.float32(4.0),
        xp.float32(sk_tau_ms),
    )


def _steady_state(voltage, steps=5000, **kwargs):
    state = _state(np.asarray(voltage).size)
    result = None
    for _ in range(steps):
        result = _update(voltage, state, **kwargs)
        state = result[:7]
    return result


def test_nalcn_is_ohmic_and_inward_below_its_reversal():
    current = to_host(_update([-80.0, -40.0, 0.0], g_nalcn=1.0)[-1])

    assert current[0] < current[1] < 0.0
    assert current[2] > 0.0


def test_nap_and_cav22_are_inward_and_voltage_activated():
    voltages = np.array([-70.0, -50.0, -30.0], dtype=np.float32)
    nap = to_host(_steady_state(voltages, g_nap=1.0)[-1])
    calcium = to_host(_steady_state(voltages, g_ca=1.0)[-1])

    assert np.all(nap < 0.0)
    assert abs(nap[1]) > abs(nap[0])
    assert np.all(calcium < 0.0)
    assert abs(calcium[2]) > abs(calcium[1]) > abs(calcium[0])


def test_sk_is_outward_and_ih_activates_with_hyperpolarization():
    sk_state = list(_state(2))
    sk_state[4][:] = 2.0
    sk_state[5][:] = 1.0
    sk_current = to_host(_update([-80.0, -40.0], tuple(sk_state), g_sk=1.0)[-1])
    ih_current = to_host(_steady_state([-100.0, -50.0], g_h=1.0)[-1])

    assert np.all(sk_current > 0.0)
    assert sk_current[1] > sk_current[0]
    assert ih_current[0] < 0.0
    assert abs(ih_current[0]) > abs(ih_current[1])


def test_gate_updates_use_exact_first_order_solution():
    voltage = np.array([-50.0], dtype=np.float32)
    state = _state(1)
    dt = 0.05

    result = _update(voltage, state, dt=dt)

    nap_activation_inf = 1.0 / (1.0 + np.exp(-(voltage + 50.0) / 4.5))
    expected_nap_activation = nap_activation_inf * (1.0 - np.exp(-dt / 0.1))
    np.testing.assert_allclose(
        to_host(result[0]), expected_nap_activation, rtol=2e-6, atol=1e-7
    )


@pytest.mark.parametrize("dt", [0.01, 0.05, 0.1])
def test_states_remain_float32_bounded_and_finite(dt):
    voltages = np.linspace(-120.0, 60.0, 37, dtype=np.float32)
    state = tuple(xp.full(voltages.size, 0.5, dtype=xp.float32) for _ in range(7))

    for _ in range(1000):
        result = _update(
            voltages,
            state,
            dt=dt,
            g_nalcn=0.1,
            g_nap=0.25,
            g_ca=0.7,
            g_sk=0.05,
            g_h=0.05,
        )
        state = result[:7]

    for gate in (*state[:4], state[5], state[6]):
        gate = to_host(gate)
        assert gate.dtype == np.float32
        assert np.all(np.isfinite(gate))
        assert np.all((gate >= 0.0) & (gate <= 1.0))
    calcium_state = to_host(state[4])
    current = to_host(result[-1])
    assert calcium_state.dtype == np.float32
    assert np.all(np.isfinite(calcium_state))
    assert np.all(calcium_state >= 0.0)
    assert current.dtype == np.float32
    assert np.all(np.isfinite(current))


def test_calcium_accumulates_decays_and_recruits_sk():
    voltage = np.array([-20.0], dtype=np.float32)
    state = _state(1)
    initial_sk = to_host(state[5]).copy()

    for _ in range(500):
        result = _update(
            voltage,
            state,
            dt=0.1,
            g_ca=1.0,
            g_sk=1.0,
            calcium_influx_scale=0.02,
            calcium_decay_tau_ms=20.0,
            sk_half_activation=0.2,
            sk_tau_ms=2.0,
        )
        state = result[:7]

    accumulated_calcium = to_host(state[4]).copy()
    recruited_sk = to_host(state[5]).copy()
    assert accumulated_calcium[0] > 0.0
    assert recruited_sk[0] > initial_sk[0]
    assert to_host(result[-1])[0] > 0.0

    for _ in range(500):
        result = _update(
            voltage,
            state,
            dt=0.1,
            g_ca=0.0,
            g_sk=1.0,
            calcium_influx_scale=0.02,
            calcium_decay_tau_ms=20.0,
            sk_half_activation=0.2,
            sk_tau_ms=2.0,
        )
        state = result[:7]

    assert to_host(state[4])[0] < accumulated_calcium[0]
    assert to_host(state[5])[0] < recruited_sk[0]


def test_zero_conductances_produce_exactly_zero_current():
    voltages = np.linspace(-100.0, 40.0, 15, dtype=np.float32)
    result = _update(voltages)

    np.testing.assert_array_equal(to_host(result[-1]), np.zeros_like(voltages))
    assert len(result) == 8
