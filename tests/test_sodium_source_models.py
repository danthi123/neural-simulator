import numpy as np
import pytest
from scipy.linalg import expm

import sim.sodium_source_models as sodium


KHALIQ = sodium.KHALIQ_RAMAN_13_STATE
BALBI = sodium.BALBI_NAV16_SIX_STATE


def _temperature(model_id, balbi_temperature=20.0):
    return None if model_id == KHALIQ else balbi_temperature


def _edge(matrix, source, target, forward, reverse):
    matrix[target, source] += forward
    matrix[source, source] -= forward
    matrix[source, target] += reverse
    matrix[target, target] -= reverse


def _expected_khaliq_generator(voltage_mv):
    matrix = np.zeros((13, 13), dtype=np.float64)
    alpha = 150.0 * np.exp(voltage_mv / 20.0)
    beta = 3.0 * np.exp(voltage_mv / -20.0)
    alfac = (0.75 / 0.005) ** 0.25
    btfac = (0.005 / 0.5) ** 0.25

    _edge(matrix, 0, 1, 4 * alpha, beta)
    _edge(matrix, 1, 2, 3 * alpha, 2 * beta)
    _edge(matrix, 2, 3, 2 * alpha, 3 * beta)
    _edge(matrix, 3, 4, alpha, 4 * beta)
    _edge(
        matrix,
        4,
        10,
        150.0 * np.exp(voltage_mv / 1e12),
        40.0 * np.exp(voltage_mv / -1e12),
    )
    _edge(
        matrix,
        10,
        11,
        1.75 * np.exp(voltage_mv / 1e12),
        0.03 * np.exp(voltage_mv / -25.0),
    )
    _edge(matrix, 10, 12, 0.75, 0.005)
    _edge(matrix, 5, 6, 4 * alpha * alfac, beta * btfac)
    _edge(matrix, 6, 7, 3 * alpha * alfac, 2 * beta * btfac)
    _edge(matrix, 7, 8, 2 * alpha * alfac, 3 * beta * btfac)
    _edge(matrix, 8, 9, alpha * alfac, 4 * beta * btfac)
    _edge(
        matrix,
        9,
        12,
        150.0 * np.exp(voltage_mv / 1e12),
        40.0 * np.exp(voltage_mv / -1e12),
    )
    _edge(matrix, 0, 5, 0.005, 0.5)
    _edge(matrix, 1, 6, 0.005 * alfac, 0.5 * btfac)
    _edge(matrix, 2, 7, 0.005 * alfac**2, 0.5 * btfac**2)
    _edge(matrix, 3, 8, 0.005 * alfac**3, 0.5 * btfac**3)
    _edge(matrix, 4, 9, 0.005 * alfac**4, 0.5 * btfac**4)
    return matrix


def _rate2(voltage_mv, b, midpoint, slope):
    return b / (1.0 + np.exp((voltage_mv - midpoint) / slope))


def _expected_balbi_generator(voltage_mv, temperature_c):
    matrix = np.zeros((6, 6), dtype=np.float64)
    q10 = 3.0 ** ((temperature_c - 20.0) / 10.0)
    c1c2 = q10 * _rate2(voltage_mv, 14, -8, -10)
    c2c1 = q10 * (
        _rate2(voltage_mv, 2, -38, 9) + _rate2(voltage_mv, 14, -8, -10)
    )
    c2o1 = q10 * _rate2(voltage_mv, 14, -18, -10)
    o1c2 = q10 * (
        _rate2(voltage_mv, 4, -48, 9) + _rate2(voltage_mv, 14, -18, -10)
    )
    c2o2 = q10 * _rate2(voltage_mv, 0.0001, -10, -8)
    o2c2 = q10 * (
        _rate2(voltage_mv, 0.0001, -55, 10)
        + _rate2(voltage_mv, 0.0001, -20, -5)
    )
    o1i1 = q10 * (
        _rate2(voltage_mv, 6, -40, 13) + _rate2(voltage_mv, 10, 15, -18)
    )
    i1o1 = q10 * _rate2(voltage_mv, 0.00001, -40, 10)
    i1c1 = q10 * _rate2(voltage_mv, 0.1, -86, 9)
    c1i1 = q10 * _rate2(voltage_mv, 0.08, -55, -12)
    i1i2 = q10 * _rate2(voltage_mv, 0.00022, -50, -5)
    i2i1 = q10 * _rate2(voltage_mv, 0.0018, -90, 30)

    _edge(matrix, 0, 1, c1c2, c2c1)
    _edge(matrix, 1, 2, c2o1, o1c2)
    _edge(matrix, 1, 3, c2o2, o2c2)
    _edge(matrix, 2, 4, o1i1, i1o1)
    _edge(matrix, 4, 0, i1c1, c1i1)
    _edge(matrix, 4, 5, i1i2, i2i1)
    return matrix


def test_source_metadata_and_state_order_are_exact():
    assert set(sodium.MODEL_METADATA) == {KHALIQ, BALBI}

    khaliq = sodium.model_metadata(KHALIQ)
    assert khaliq["source_file"] == "rsg.mod"
    assert khaliq["source_sha256"] == (
        "1a3382714bd0962665ec31f7dfac2aa3a9e403a5e3d23e29851afec232c4543e"
    )
    assert khaliq["states"] == (
        "C1", "C2", "C3", "C4", "C5", "I1", "I2", "I3", "I4", "I5", "O", "B", "I6"
    )
    assert khaliq["open_states"] == ("O",)
    assert khaliq["q10"] is None
    assert "not equivalent to source seqinitial" in khaliq["transfer_initialization"]

    balbi = sodium.model_metadata(BALBI)
    assert balbi["source_file"] == "Nav16_a.mod"
    assert balbi["source_sha256"] == (
        "69931ced1587944070edb3169a865e9e3e2a42f715b19a8b7b57e72e831ba71d"
    )
    assert balbi["states"] == ("C1", "C2", "O1", "O2", "I1", "I2")
    assert balbi["open_states"] == ("O1", "O2")
    assert balbi["q10"] == 3.0
    assert balbi["q10_reference_temperature_c"] == 20.0
    assert balbi["source_package_temperature_c"] == 22.0


@pytest.mark.parametrize("voltage_mv", [-80.0, -20.0, 20.0])
def test_khaliq_transition_rates_match_rsg_mod(voltage_mv):
    actual = sodium._generator(KHALIQ, np.asarray(voltage_mv), None, np)
    np.testing.assert_allclose(actual, _expected_khaliq_generator(voltage_mv), rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("voltage_mv,temperature_c", [(-80.0, 20.0), (-20.0, 30.0), (20.0, 10.0)])
def test_balbi_transition_rates_match_nav16_mod(voltage_mv, temperature_c):
    actual = sodium._generator(BALBI, np.asarray(voltage_mv), np.asarray(temperature_c), np)
    expected = _expected_balbi_generator(voltage_mv, temperature_c)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_source_temperature_behavior():
    voltage = np.asarray(-20.0)
    with pytest.raises(ValueError, match="must be None"):
        sodium.equilibrium(KHALIQ, voltage, 20.0, np)
    sodium.equilibrium(KHALIQ, voltage, None, np)

    balbi_20 = sodium._generator(BALBI, voltage, np.asarray(20.0), np)
    balbi_30 = sodium._generator(BALBI, voltage, np.asarray(30.0), np)
    np.testing.assert_allclose(balbi_30, 3.0 * balbi_20, rtol=1e-14, atol=1e-14)
    with pytest.raises(ValueError, match="must be explicit"):
        sodium.equilibrium(BALBI, voltage, None, np)


@pytest.mark.parametrize("model_id", [KHALIQ, BALBI])
def test_equilibrium_is_stationary_conserved_and_nonnegative(model_id):
    temperature = _temperature(model_id)
    equilibria = sodium.equilibrium(
        model_id,
        np.asarray([-100.0, -60.0, 0.0]),
        temperature,
        np,
    )
    assert np.all(equilibria >= 0.0)
    np.testing.assert_allclose(equilibria.sum(axis=-1), 1.0, rtol=0.0, atol=1e-12)

    advanced = sodium.advance(
        model_id,
        np.asarray([-100.0, -60.0, 0.0]),
        equilibria,
        100.0,
        temperature,
        np,
    )
    np.testing.assert_allclose(advanced, equilibria, rtol=2e-8, atol=2e-10)


@pytest.mark.parametrize("model_id", [KHALIQ, BALBI])
def test_advance_conserves_nonnegative_probabilities(model_id):
    count = len(sodium.model_metadata(model_id)["states"])
    initial = np.eye(count)
    advanced = sodium.advance(model_id, -20.0, initial, 2.0, _temperature(model_id), np)
    assert np.all(advanced >= -1e-12)
    np.testing.assert_allclose(advanced.sum(axis=-1), 1.0, rtol=0.0, atol=1e-10)


@pytest.mark.parametrize("model_id,temperature", [(KHALIQ, None), (BALBI, 22.0)])
@pytest.mark.parametrize("voltage,duration", [(-100.0, 0.005), (-40.0, 0.5), (30.0, 20.0)])
def test_advance_matches_independent_scipy_matrix_exponential(
    model_id, temperature, voltage, duration
):
    initial = sodium.equilibrium(model_id, -90.0, temperature, np)
    generator = sodium._generator(
        model_id, np.asarray(voltage),
        None if temperature is None else np.asarray(temperature), np,
    )
    expected = expm(generator * duration) @ initial
    actual = sodium.advance(model_id, voltage, initial, duration, temperature, np)

    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-11)


@pytest.mark.parametrize("model_id,temperature", [(KHALIQ, None), (BALBI, 22.0)])
def test_advance_obeys_fixed_voltage_semigroup(model_id, temperature):
    initial = sodium.equilibrium(model_id, -100.0, temperature, np)
    direct = sodium.advance(model_id, -20.0, initial, 3.75, temperature, np)
    first = sodium.advance(model_id, -20.0, initial, 1.25, temperature, np)
    composed = sodium.advance(model_id, -20.0, first, 2.5, temperature, np)

    np.testing.assert_allclose(composed, direct, rtol=2e-9, atol=2e-11)


@pytest.mark.parametrize("model_id", [KHALIQ, BALBI])
def test_zero_duration_is_exact_identity(model_id):
    temperature = _temperature(model_id, 37.0)
    initial = sodium.equilibrium(model_id, -80.0, temperature, np)
    actual = sodium.advance(model_id, 40.0, initial, 0.0, temperature, np)
    np.testing.assert_array_equal(actual, initial)


@pytest.mark.parametrize("model_id", [KHALIQ, BALBI])
def test_cpu_broadcasts_voltage_and_elapsed_time_to_complete_trace(model_id):
    temperature = _temperature(model_id, 30.0)
    holding_voltages = np.asarray([-100.0, -90.0, -80.0])[:, None]
    clamp_voltages = np.asarray([-60.0, -20.0, 20.0])[:, None]
    elapsed_ms = np.asarray([0.0, 0.01, 0.1, 1.0])[None, :]
    initial = sodium.equilibrium(model_id, holding_voltages, temperature, np)

    trace = sodium.advance(
        model_id,
        clamp_voltages,
        initial,
        elapsed_ms,
        temperature,
        np,
    )

    assert trace.shape == (3, 4, initial.shape[-1])
    np.testing.assert_array_equal(trace[:, 0, :], initial[:, 0, :])
    np.testing.assert_allclose(trace.sum(axis=-1), 1.0, rtol=0.0, atol=1e-10)
    assert np.all(trace >= -1e-12)
    scalar = sodium.advance(
        model_id,
        clamp_voltages[1, 0],
        initial[1, 0],
        elapsed_ms[0, 2],
        temperature,
        np,
    )
    np.testing.assert_allclose(trace[1, 2], scalar, rtol=2e-11, atol=2e-13)

    all_zero = sodium.advance(
        model_id,
        clamp_voltages,
        initial,
        np.zeros((1, 4)),
        temperature,
        np,
    )
    expected_identity = np.broadcast_to(initial, (3, 4, initial.shape[-1]))
    np.testing.assert_array_equal(all_zero, expected_identity)


def test_open_probability_uses_source_open_states():
    khaliq = np.zeros(13)
    khaliq[10] = 1.0
    assert sodium.open_probability(KHALIQ, khaliq, np) == 1.0

    balbi = np.zeros(6)
    balbi[2] = 0.25
    balbi[3] = 0.75
    assert sodium.open_probability(BALBI, balbi, np) == 1.0


@pytest.mark.parametrize(
    "call",
    [
        lambda: sodium.equilibrium("not-a-model", -80.0, 20.0, np),
        lambda: sodium.equilibrium(KHALIQ, -80.0, 20.0, np),
        lambda: sodium.equilibrium(BALBI, -80.0, None, np),
        lambda: sodium.equilibrium(BALBI, -80.0, np.nan, np),
        lambda: sodium.equilibrium(BALBI, -80.0, [20.0], np),
        lambda: sodium.advance(KHALIQ, -20.0, np.ones(12) / 12, 1.0, None, np),
        lambda: sodium.advance(BALBI, np.ones(2), np.ones((3, 6)) / 6, 1.0, 20.0, np),
        lambda: sodium.advance(BALBI, -20.0, np.ones(6) / 6, -1.0, 20.0, np),
        lambda: sodium.advance(BALBI, -20.0, np.ones(6) / 6, [0.0, np.nan], 20.0, np),
        lambda: sodium.open_probability(KHALIQ, np.zeros(13), np),
    ],
)
def test_invalid_inputs_fail_closed(call):
    with pytest.raises(ValueError):
        call()


def test_cupy_backend_matches_numpy_when_available():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("no CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")

    for model_id in (KHALIQ, BALBI):
        temperature = _temperature(model_id, 30.0)
        holding_voltages = np.asarray([-100.0, -90.0, -80.0])[:, None]
        clamp_voltages = np.asarray([-60.0, -20.0, 20.0])[:, None]
        elapsed_ms = np.asarray([0.0, 0.01, 0.1, 0.5])[None, :]
        expected_eq = sodium.equilibrium(model_id, holding_voltages, temperature, np)
        actual_eq = sodium.equilibrium(model_id, cp.asarray(holding_voltages), temperature, cp)
        np.testing.assert_allclose(cp.asnumpy(actual_eq), expected_eq, rtol=2e-11, atol=2e-13)

        expected = sodium.advance(
            model_id,
            clamp_voltages,
            expected_eq,
            elapsed_ms,
            temperature,
            np,
        )
        actual = sodium.advance(
            model_id,
            cp.asarray(clamp_voltages),
            actual_eq,
            cp.asarray(elapsed_ms),
            temperature,
            cp,
        )
        assert actual.shape == (3, 4, expected_eq.shape[-1])
        cp.testing.assert_array_equal(actual[:, 0, :], actual_eq[:, 0, :])
        np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=2e-9, atol=2e-11)
