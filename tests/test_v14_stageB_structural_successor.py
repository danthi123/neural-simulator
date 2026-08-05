from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from sim.backend import get_backend
from sim.kernels import fused_snr_fast_channel_clamp_update
from sim.snr_structural_successor import (
    DEFAULT_FAST_CHANNEL_PARAMETERS as PARAMETERS,
    EVIDENCE_CLASSES,
    PROTOCOL_PATH,
    PROTOCOL_SHA256,
    UNITS,
    power_gate_rise_factor,
)


ROOT = Path(__file__).resolve().parents[1]
_, BACKEND = get_backend()
NUMPY_ONLY = pytest.mark.skipif(BACKEND != "numpy", reason="NumPy authority test")


def _inf_activation(voltage, half, slope):
    return 1.0 / (1.0 + np.exp(-(voltage - half) / slope))


def _inf_inactivation(voltage, half, slope):
    return 1.0 / (1.0 + np.exp((voltage - half) / slope))


def _inputs(voltage, states, dt):
    p = PARAMETERS
    return (
        voltage, *states, np.float32(dt),
        np.float32(p.na_activation_half_mv),
        np.float32(p.na_activation_slope_mv),
        np.float32(p.na_inactivation_half_mv),
        np.float32(p.na_inactivation_slope_mv),
        np.float32(p.na_activation_gate_tau_at_zero_ms),
        np.float32(p.na_deactivation_gate_tau_at_minus_40_ms),
        np.float32(p.na_recovery_fast_tau_ms),
        np.float32(p.na_recovery_slow_tau_ms),
        np.float32(p.na_inactivation_gate_tau_at_zero_ms),
        np.float32(p.na_recovery_fast_fraction),
        np.float32(p.kv3_activation_half_mv),
        np.float32(p.kv3_activation_slope_mv),
        np.float32(p.kv3_inactivation_half_mv),
        np.float32(p.kv3_inactivation_slope_mv),
        np.float32(p.kv3_activation_gate_tau_at_plus_40_ms),
        *(np.float32(value) for value in p.kv3_deactivation_gate_taus_ms),
        np.float32(p.kv3_inactivation_tau_prior_ms),
        np.float32(p.na_reversal_mv),
        np.float32(p.potassium_reversal_mv),
    )


def _step(voltage, states, dt=0.01):
    return fused_snr_fast_channel_clamp_update(*_inputs(voltage, states, dt))


def test_stage0_contract_digest_units_and_evidence_classes_are_frozen():
    protocol = ROOT / PROTOCOL_PATH
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == PROTOCOL_SHA256
    assert UNITS == {
        "voltage": "mV", "time": "ms", "conductance": "normalized",
        "current": "normalized_conductance_times_mV", "gate": "dimensionless",
    }
    assert EVIDENCE_CLASSES["steady_state_and_reported_kinetics"].startswith("direct_measured")
    assert EVIDENCE_CLASSES["tau_between_voltage_endpoints"].endswith("model_prior")
    assert EVIDENCE_CLASSES["conductance_scale"].startswith("unavailable")


def test_powered_gate_time_conversions_reproduce_reported_current_times():
    assert math.isclose(
        PARAMETERS.na_activation_gate_tau_at_zero_ms
        * power_gate_rise_factor(3, 0.1, 0.9),
        PARAMETERS.na_activation_current_rise_10_90_ms,
    )
    assert math.isclose(
        PARAMETERS.kv3_activation_gate_tau_at_plus_40_ms
        * power_gate_rise_factor(4, 0.2, 0.8),
        PARAMETERS.kv3_activation_current_rise_20_80_plus_40_ms,
    )
    assert PARAMETERS.na_deactivation_gate_tau_at_minus_40_ms == 3 * 0.099
    assert PARAMETERS.kv3_deactivation_gate_taus_ms == (4 * 0.82, 4 * 1.35, 4 * 1.87)


@NUMPY_ONLY
def test_clamp_kernel_converges_to_source_measured_steady_state_curves():
    voltage = np.asarray([-120, -63.3, -30.2, -8.5, 0, 40], dtype=np.float32)
    states = tuple(np.zeros_like(voltage) for _ in range(5))
    result = _step(voltage, states, dt=20_000.0)
    np.testing.assert_allclose(
        result[0] ** 3,
        _inf_activation(voltage, PARAMETERS.na_activation_half_mv, PARAMETERS.na_activation_slope_mv),
        atol=2e-6,
    )
    np.testing.assert_allclose(
        result[1],
        _inf_inactivation(voltage, PARAMETERS.na_inactivation_half_mv, PARAMETERS.na_inactivation_slope_mv),
        atol=2e-6,
    )
    np.testing.assert_allclose(result[2], result[1], atol=2e-6)
    np.testing.assert_allclose(
        result[3] ** 4,
        _inf_activation(voltage, PARAMETERS.kv3_activation_half_mv, PARAMETERS.kv3_activation_slope_mv),
        atol=2e-6,
    )
    np.testing.assert_allclose(
        result[4],
        _inf_inactivation(voltage, PARAMETERS.kv3_inactivation_half_mv, PARAMETERS.kv3_inactivation_slope_mv),
        atol=2e-6,
    )


@NUMPY_ONLY
def test_recovery_components_remain_separate_at_minus_120_mv():
    voltage = np.asarray([-120.0], dtype=np.float32)
    states = tuple(np.zeros_like(voltage) for _ in range(5))
    duration = 5.0
    result = _step(voltage, states, dt=duration)
    h_inf = _inf_inactivation(
        voltage, PARAMETERS.na_inactivation_half_mv, PARAMETERS.na_inactivation_slope_mv
    )[0]
    expected_fast = h_inf * (1.0 - math.exp(-duration / PARAMETERS.na_recovery_fast_tau_ms))
    expected_slow = h_inf * (1.0 - math.exp(-duration / PARAMETERS.na_recovery_slow_tau_ms))
    assert math.isclose(float(result[1][0]), expected_fast, rel_tol=2e-6)
    assert math.isclose(float(result[2][0]), expected_slow, rel_tol=2e-6)
    assert float(result[1][0]) > float(result[2][0])


@NUMPY_ONLY
def test_kernel_is_float32_finite_and_does_not_reset_or_mutate_inputs():
    voltage = np.asarray([-120.0, -40.0, 0.0, 40.0], dtype=np.float32)
    states = tuple(np.full_like(voltage, 0.5) for _ in range(5))
    snapshots = tuple(value.copy() for value in states)
    result = _step(voltage, states, dt=0.025)
    assert len(result) == 7
    assert all(value.dtype == np.float32 for value in result)
    assert all(np.isfinite(value).all() for value in result)
    for original, snapshot in zip(states, snapshots):
        np.testing.assert_array_equal(original, snapshot)
