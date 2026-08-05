from __future__ import annotations

import numpy as np
import pytest

from sim.backend import get_backend
from sim.kernels import fused_snr_fast_channel_clamp_update
from sim.snr_structural_successor import DEFAULT_FAST_CHANNEL_PARAMETERS as P


cp, BACKEND = get_backend()
pytestmark = pytest.mark.skipif(BACKEND != "cupy", reason="requires SIM_BACKEND=cupy")


def test_cupy_fast_channel_kernel_preserves_measured_conductance_curves():
    voltage = cp.asarray([-120, -63.3, -30.2, -8.5, 0, 40], dtype=cp.float32)
    states = tuple(cp.zeros_like(voltage) for _ in range(5))
    values = (
        P.na_activation_half_mv, P.na_activation_slope_mv,
        P.na_inactivation_half_mv, P.na_inactivation_slope_mv,
        P.na_activation_gate_tau_at_zero_ms,
        P.na_deactivation_gate_tau_at_minus_40_ms,
        P.na_recovery_fast_tau_ms, P.na_recovery_slow_tau_ms,
        P.na_inactivation_gate_tau_at_zero_ms, P.na_recovery_fast_fraction,
        P.kv3_activation_half_mv, P.kv3_activation_slope_mv,
        P.kv3_inactivation_half_mv, P.kv3_inactivation_slope_mv,
        P.kv3_activation_gate_tau_at_plus_40_ms,
        *P.kv3_deactivation_gate_taus_ms, P.kv3_inactivation_tau_prior_ms,
        P.na_reversal_mv, P.potassium_reversal_mv,
    )
    result = fused_snr_fast_channel_clamp_update(
        voltage, *states, cp.float32(20_000.0),
        *(cp.float32(value) for value in values),
    )
    cp.cuda.Stream.null.synchronize()
    observed_na = cp.asnumpy(result[0] ** 3)
    observed_kv3 = cp.asnumpy(result[3] ** 4)
    host_voltage = cp.asnumpy(voltage)
    expected_na = 1.0 / (
        1.0 + np.exp(-(host_voltage - P.na_activation_half_mv) / P.na_activation_slope_mv)
    )
    expected_kv3 = 1.0 / (
        1.0 + np.exp(-(host_voltage - P.kv3_activation_half_mv) / P.kv3_activation_slope_mv)
    )
    np.testing.assert_allclose(observed_na, expected_na, atol=2e-6)
    np.testing.assert_allclose(observed_kv3, expected_kv3, atol=2e-6)
    assert all(value.dtype == cp.float32 for value in result)
    assert all(bool(cp.isfinite(value).all()) for value in result)

