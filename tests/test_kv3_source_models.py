import numpy as np
import pytest
from scipy.linalg import expm

from sim.kv3_source_models import (
    DESAI_2008_CONTROL,
    LABRO_2015,
    MODEL_METADATA,
    advance,
    equilibrium,
    open_probability,
    rates,
)


LABRO_TEMPERATURE_C = 22.5
DESAI_TEMPERATURE_C = None


def test_metadata_keeps_models_separate_and_source_bound():
    assert set(MODEL_METADATA) == {LABRO_2015, DESAI_2008_CONTROL}
    assert MODEL_METADATA[LABRO_2015]["state_names"] == (
        "resting_closed",
        "pre_active_closed",
        "relaxed_pre_active_closed",
        "relaxed_active_open",
    )
    assert MODEL_METADATA[LABRO_2015]["has_inactivation_state"] is False
    assert "inactiv" not in " ".join(
        MODEL_METADATA[LABRO_2015]["state_names"]
    ).lower()
    assert MODEL_METADATA[DESAI_2008_CONTROL]["state_names"] == (
        "n_activation",
        "p_availability",
    )
    assert MODEL_METADATA[DESAI_2008_CONTROL]["open_probability"] == (
        "n^3 * (0.23 + 0.77*p)"
    )
    assert MODEL_METADATA[LABRO_2015]["source_temperature_c"] is None
    assert MODEL_METADATA[LABRO_2015]["reported_temperature"] == (
        "room temperature; exact value unresolved"
    )
    assert "prospectively sealed" in MODEL_METADATA[LABRO_2015][
        "temperature_convention"
    ]
    assert MODEL_METADATA[DESAI_2008_CONTROL]["source_temperature_c"] is None
    assert MODEL_METADATA[DESAI_2008_CONTROL]["reported_temperature"] is None
    assert "requires None" in MODEL_METADATA[DESAI_2008_CONTROL][
        "temperature_convention"
    ]


def test_labro_rates_match_supplement_table_and_unit_explicit_voltage_law():
    voltage = np.array([6.2, -20.0, 40.0])
    alpha, beta = rates(LABRO_2015, voltage, LABRO_TEMPERATURE_C, np)

    alpha0 = np.array([0.05, 6.0, 1.0])
    beta0 = np.array([0.15, 0.6, 0.8])
    z = np.array([3.5, 0.4, 0.001])
    thermal_mv = (
        1000.0
        * 1.380649e-23
        * (LABRO_TEMPERATURE_C + 273.15)
        / 1.602176634e-19
    )
    exponent = (voltage[:, None] - 6.2) * z / thermal_mv

    np.testing.assert_allclose(alpha, alpha0 * np.exp(exponent), rtol=2e-15)
    np.testing.assert_allclose(beta, beta0 * np.exp(-exponent), rtol=2e-15)
    np.testing.assert_array_equal(alpha[0], alpha0)
    np.testing.assert_array_equal(beta[0], beta0)


def test_labro_accepts_execution_spec_temperature_sensitivity_envelope():
    voltage = -20.0
    rates_by_temperature = [
        rates(LABRO_2015, voltage, temperature_c, np)[0]
        for temperature_c in (20.0, 22.5, 25.0)
    ]
    assert all(np.all(np.isfinite(value)) for value in rates_by_temperature)
    assert not np.array_equal(rates_by_temperature[0], rates_by_temperature[-1])


def test_desai_rates_match_published_control_constants_exactly():
    voltage = np.array([-40.0, 0.0, 30.0])
    alpha, beta = rates(DESAI_2008_CONTROL, voltage, DESAI_TEMPERATURE_C, np)

    expected_alpha = np.array([0.039, 0.000045]) * np.exp(
        voltage[:, None] * np.array([0.0467, -0.18925])
    )
    expected_beta = np.array([0.0868, 0.00246]) * np.exp(
        voltage[:, None] * np.array([0.0067, 0.01075])
    )
    np.testing.assert_allclose(alpha, expected_alpha, rtol=2e-15)
    np.testing.assert_allclose(beta, expected_beta, rtol=2e-15)
    np.testing.assert_array_equal(alpha[1], np.array([0.039, 0.000045]))
    np.testing.assert_array_equal(beta[1], np.array([0.0868, 0.00246]))


@pytest.mark.parametrize(
    "model_id,width,temperature_c",
    [(LABRO_2015, 4, LABRO_TEMPERATURE_C), (DESAI_2008_CONTROL, 2, None)],
)
def test_equilibrium_is_stationary_under_finite_advance(
    model_id, width, temperature_c
):
    voltage = np.array([-60.0, -20.0, 20.0, 50.0])
    state = equilibrium(model_id, voltage, temperature_c, np)
    assert state.shape == (4, width)

    result = advance(model_id, state, voltage, 7.25, temperature_c, np)
    np.testing.assert_allclose(result, state, atol=2e-13, rtol=2e-13)


def test_labro_conservative_solve_preserves_probability_and_nonnegativity():
    initial = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.1, 0.2, 0.3, 0.4],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    result = advance(
        LABRO_2015,
        initial,
        np.array([-60.0, -10.0, 40.0]),
        11.0,
        LABRO_TEMPERATURE_C,
        np,
    )
    assert np.all(result >= 0.0)
    np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=2e-15)
    np.testing.assert_allclose(open_probability(LABRO_2015, result, np), result[:, 3])


def test_labro_advance_matches_independent_scipy_matrix_exponential():
    voltage = -40.0
    duration = 17.5
    initial = equilibrium(LABRO_2015, -90.0, 22.5, np)
    alpha, beta = rates(LABRO_2015, voltage, 22.5, np)
    generator = np.zeros((4, 4), dtype=np.float64)
    generator[0, 0], generator[1, 0] = -alpha[0], alpha[0]
    generator[0, 1], generator[1, 1], generator[2, 1] = (
        beta[0], -(beta[0] + alpha[1]), alpha[1]
    )
    generator[1, 2], generator[2, 2], generator[3, 2] = (
        beta[1], -(beta[1] + alpha[2]), alpha[2]
    )
    generator[2, 3], generator[3, 3] = beta[2], -beta[2]
    expected = expm(generator * duration) @ initial
    actual = advance(LABRO_2015, initial, voltage, duration, 22.5, np)

    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_desai_rush_larsen_stays_bounded_and_uses_published_open_probability():
    initial = np.array([[0.0, 1.0], [1.0, 0.0]])
    result = advance(
        DESAI_2008_CONTROL,
        initial,
        np.array([-70.0, 30.0]),
        25.0,
        DESAI_TEMPERATURE_C,
        np,
    )
    assert np.all((0.0 <= result) & (result <= 1.0))
    expected = result[:, 0] ** 3 * (0.23 + 0.77 * result[:, 1])
    np.testing.assert_allclose(
        open_probability(DESAI_2008_CONTROL, result, np), expected
    )


@pytest.mark.parametrize(
    "model_id,temperature_c",
    [(LABRO_2015, LABRO_TEMPERATURE_C), (DESAI_2008_CONTROL, None)],
)
def test_duration_array_broadcasts_a_complete_fixed_voltage_trace(model_id, temperature_c):
    initial = equilibrium(model_id, -70.0, temperature_c, np)
    durations = np.array([0.0, 0.05, 0.5, 2.0, 10.0])
    trace = advance(model_id, initial, 20.0, durations, temperature_c, np)

    expected = np.stack(
        [
            advance(model_id, initial, 20.0, float(duration), temperature_c, np)
            for duration in durations
        ]
    )
    assert trace.shape == (durations.size, initial.size)
    np.testing.assert_array_equal(trace[0], initial)
    np.testing.assert_allclose(trace, expected, atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize(
    "model_id,temperature_c",
    [(LABRO_2015, LABRO_TEMPERATURE_C), (DESAI_2008_CONTROL, None)],
)
def test_state_voltage_and_mixed_duration_arrays_broadcast_together(
    model_id, temperature_c
):
    state = equilibrium(model_id, np.array([-70.0, -50.0]), temperature_c, np)
    voltage = np.array([[-30.0], [10.0], [40.0]])
    duration = np.array([[0.0], [0.75], [3.0]])
    result = advance(model_id, state, voltage, duration, temperature_c, np)

    assert result.shape == (3, 2, state.shape[-1])
    np.testing.assert_array_equal(result[0], state)
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize(
    "model_id,temperature_c",
    [(LABRO_2015, LABRO_TEMPERATURE_C), (DESAI_2008_CONTROL, None)],
)
def test_zero_duration_is_exact_identity(model_id, temperature_c):
    state = equilibrium(model_id, np.array([-50.0, 10.0]), temperature_c, np)
    result = advance(
        model_id, state, np.array([-50.0, 10.0]), 0.0, temperature_c, np
    )
    np.testing.assert_array_equal(result, state)
    assert result is not state


@pytest.mark.parametrize(
    "call",
    [
        lambda: equilibrium("unknown", -20.0, LABRO_TEMPERATURE_C, np),
        lambda: equilibrium(LABRO_2015, -20.0, None, np),
        lambda: equilibrium(LABRO_2015, -20.0, -273.15, np),
        lambda: equilibrium(DESAI_2008_CONTROL, -20.0, 22.5, np),
        lambda: equilibrium(DESAI_2008_CONTROL, np.nan, None, np),
        lambda: advance(
            LABRO_2015,
            [1.0, 0.0, 0.0],
            -20.0,
            1.0,
            LABRO_TEMPERATURE_C,
            np,
        ),
        lambda: advance(
            LABRO_2015,
            [0.5, 0.0, 0.0, 0.0],
            -20.0,
            1.0,
            LABRO_TEMPERATURE_C,
            np,
        ),
        lambda: advance(
            DESAI_2008_CONTROL,
            np.ones((2, 2)),
            np.ones(3),
            1.0,
            None,
            np,
        ),
        lambda: advance(
            DESAI_2008_CONTROL, [0.5, 0.5], -20.0, -1.0, None, np
        ),
    ],
)
def test_invalid_model_temperature_shape_or_values_fail_closed(call):
    with pytest.raises((TypeError, ValueError)):
        call()


def test_numpy_cupy_backend_parity_when_cupy_is_available():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")

    voltage_np = np.array([-55.0, -5.0, 35.0])
    for model_id in (LABRO_2015, DESAI_2008_CONTROL):
        temperature_c = LABRO_TEMPERATURE_C if model_id == LABRO_2015 else None
        state_np = equilibrium(model_id, voltage_np, temperature_c, np)
        expected = advance(model_id, state_np, voltage_np, 3.5, temperature_c, np)
        state_cp = equilibrium(
            model_id, cupy.asarray(voltage_np), temperature_c, cupy
        )
        actual = advance(
            model_id,
            state_cp,
            cupy.asarray(voltage_np),
            3.5,
            temperature_c,
            cupy,
        )
        np.testing.assert_allclose(cupy.asnumpy(actual), expected, atol=2e-12, rtol=2e-12)


def test_numpy_cupy_broadcast_trace_parity_when_cupy_is_available():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")

    durations = np.array([0.0, 0.1, 0.7, 4.0])[:, None]
    voltages = np.array([-35.0, 15.0])[None, :]
    for model_id, temperature_c in (
        (LABRO_2015, LABRO_TEMPERATURE_C),
        (DESAI_2008_CONTROL, None),
    ):
        initial = equilibrium(model_id, -70.0, temperature_c, np)
        expected = advance(
            model_id, initial, voltages, durations, temperature_c, np
        )
        actual = advance(
            model_id,
            cupy.asarray(initial),
            cupy.asarray(voltages),
            cupy.asarray(durations),
            temperature_c,
            cupy,
        )
        np.testing.assert_allclose(
            cupy.asnumpy(actual), expected, atol=2e-12, rtol=2e-12
        )
