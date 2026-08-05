"""Oracle-parity tests for candidate-first source-model kinetic batches."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from sim import kv3_source_models as kv3
from sim import sodium_source_models as sodium
from sim import source_model_candidate_batch as batch


MODELS = (
    (batch.KHALIQ_RAMAN_13_STATE, sodium, None),
    (batch.BALBI_NAV16_SIX_STATE, sodium, np.array([20.0, 24.0, 31.0])),
    (batch.LABRO_2015, kv3, np.array([20.0, 22.5, 25.0])),
    (batch.DESAI_2008_CONTROL, kv3, None),
)


def _parameters(model_id: str, module, count: int) -> list[dict]:
    result = []
    for index in range(count):
        document = dict(module.source_parameters(model_id))
        if model_id == batch.KHALIQ_RAMAN_13_STATE:
            document["alpha_per_ms"] = 120.0 + 15.0 * index
            document["x6_mv"] = -22.0 - index
        elif model_id == batch.BALBI_NAV16_SIX_STATE:
            document["c1c2"] = (11.0 + index, -6.0 + index, -8.0 - index)
            document["q10"] = 2.5 + 0.25 * index
        elif model_id == batch.LABRO_2015:
            document["vhalf_mv"] = 2.0 + 2.0 * index
            document["alpha0_per_ms"] = (0.04 + 0.01 * index, 5.5 + index, 0.8 + 0.1 * index)
        else:
            document["k_alpha_per_ms"] = (0.035 + 0.005 * index, 0.00004 + 0.00001 * index)
            document["eta_beta_per_mv"] = (0.005 + 0.0005 * index, 0.009 + 0.00025 * index)
        result.append(document)
    return result


def _temperature(temperatures, index: int):
    return None if temperatures is None else float(temperatures[index])


def _oracle_equilibrium(model_id, module, parameters, voltage, temperatures):
    rows = []
    for index, document in enumerate(parameters):
        rows.append(
            module.equilibrium(
                model_id, voltage, _temperature(temperatures, index), np, parameters=document
            )
        )
    return np.stack(rows)


def _oracle_advance(model_id, module, parameters, voltage, states, duration, temperatures):
    rows = []
    for index, document in enumerate(parameters):
        if module is sodium:
            row = sodium.advance(
                model_id, voltage, states[index], duration, _temperature(temperatures, index), np,
                parameters=document,
            )
        else:
            row = kv3.advance(
                model_id, states[index], voltage, duration, _temperature(temperatures, index), np,
                parameters=document,
            )
        rows.append(row)
    return np.stack(rows)


def _oracle_trace(model_id, module, parameters, voltage, states, elapsed, temperatures):
    rows = []
    for index, document in enumerate(parameters):
        if module is sodium:
            row = sodium.trace(
                model_id, voltage, states[index], elapsed, _temperature(temperatures, index), np,
                parameters=document,
            )
        else:
            row = np.stack(
                [
                    kv3.advance(
                        model_id, states[index], voltage, duration,
                        _temperature(temperatures, index), np, parameters=document,
                    )
                    for duration in elapsed
                ],
                axis=-2,
            )
        rows.append(row)
    return np.stack(rows)


@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("model_id,module,temperatures", MODELS)
def test_equilibrium_batch_matches_heterogeneous_single_candidate_oracle(
    model_id, module, temperatures, count
):
    parameters = _parameters(model_id, module, count)
    temperature = None if temperatures is None else temperatures[:count]
    voltage = np.array([[-85.0, -45.0], [-10.0, 25.0]])

    actual = batch.equilibrium_batch(model_id, parameters, voltage, temperature, np)
    expected = _oracle_equilibrium(model_id, module, parameters, voltage, temperature)

    assert actual.dtype == np.float64
    assert actual.shape == (count, 2, 2, expected.shape[-1])
    np.testing.assert_allclose(actual, expected, rtol=5e-8, atol=5e-10)
    assert np.all(np.isfinite(actual))
    if model_id != batch.DESAI_2008_CONTROL:
        assert np.all(actual >= -1e-10)
        np.testing.assert_allclose(actual.sum(axis=-1), 1.0, rtol=0.0, atol=1e-8)


@pytest.mark.parametrize("model_id,module,temperatures", MODELS)
def test_chained_advances_and_open_probability_match_single_candidate_oracle(
    model_id, module, temperatures
):
    count = 3
    parameters = _parameters(model_id, module, count)
    temperature = None if temperatures is None else temperatures[:count]
    holding = np.array([-95.0, -80.0])
    initial = batch.equilibrium_batch(model_id, parameters, holding, temperature, np)
    command_voltage = np.array([[-50.0], [-15.0], [25.0]])
    first_duration = np.array([[0.05], [0.35], [1.25]])
    second_duration = np.array([[0.1, 0.9]])

    first = batch.advance_batch(
        model_id, parameters, command_voltage, initial, first_duration, temperature, np
    )
    first_expected = _oracle_advance(
        model_id, module, parameters, command_voltage, initial, first_duration, temperature
    )
    second = batch.advance_batch(
        model_id, parameters, command_voltage, first, second_duration, temperature, np
    )
    second_expected = _oracle_advance(
        model_id, module, parameters, command_voltage, first_expected, second_duration, temperature
    )

    np.testing.assert_allclose(first, first_expected, rtol=5e-8, atol=5e-10)
    np.testing.assert_allclose(second, second_expected, rtol=5e-8, atol=5e-10)
    open_actual = batch.open_probability_batch(model_id, second, np)
    open_expected = np.stack(
        [
            module.open_probability(model_id, second_expected[index], np)
            for index in range(count)
        ]
    )
    np.testing.assert_allclose(open_actual, open_expected, rtol=5e-8, atol=5e-10)
    if model_id != batch.DESAI_2008_CONTROL:
        np.testing.assert_allclose(second.sum(axis=-1), 1.0, rtol=0.0, atol=1e-8)
        assert np.all(second >= -1e-10)
    else:
        assert np.all((second >= -1e-10) & (second <= 1.0 + 1e-10))


@pytest.mark.parametrize("model_id,module,temperatures", MODELS)
def test_trace_batch_matches_oracle_across_multiple_voltages_and_durations(
    model_id, module, temperatures
):
    count = 3
    parameters = _parameters(model_id, module, count)
    temperature = None if temperatures is None else temperatures[:count]
    voltage = np.array([-70.0, -20.0, 35.0])
    initial = batch.equilibrium_batch(model_id, parameters, -90.0, temperature, np)
    elapsed = np.array([0.0, 0.01, 0.2, 1.5])

    actual = batch.trace_batch(model_id, parameters, voltage, initial, elapsed, temperature, np)
    expected = _oracle_trace(model_id, module, parameters, voltage, initial, elapsed, temperature)

    assert actual.shape == (count, 3, 4, initial.shape[-1])
    np.testing.assert_allclose(actual, expected, rtol=5e-8, atol=5e-10)
    np.testing.assert_array_equal(actual[..., 0, :], np.broadcast_to(initial[:, None, :], (count, 3, initial.shape[-1])))


@pytest.mark.parametrize("model_id,module,temperatures", MODELS)
def test_candidate_batch_is_deterministic(model_id, module, temperatures):
    parameters = _parameters(model_id, module, 3)
    temperature = None if temperatures is None else temperatures
    voltage = np.array([-60.0, -10.0, 30.0])
    first = batch.equilibrium_batch(model_id, parameters, voltage, temperature, np)
    second = batch.equilibrium_batch(model_id, copy.deepcopy(parameters), voltage, temperature, np)
    np.testing.assert_array_equal(first, second)
    advanced_first = batch.advance_batch(model_id, parameters, voltage, first, 0.75, temperature, np)
    advanced_second = batch.advance_batch(model_id, parameters, voltage, second, 0.75, temperature, np)
    np.testing.assert_array_equal(advanced_first, advanced_second)


@pytest.mark.parametrize(
    "model_id,module,temperature",
    [
        (batch.KHALIQ_RAMAN_13_STATE, sodium, np.array([20.0])),
        (batch.BALBI_NAV16_SIX_STATE, sodium, None),
        (batch.LABRO_2015, kv3, np.array([-273.15])),
        (batch.DESAI_2008_CONTROL, kv3, np.array([22.5])),
    ],
)
def test_source_temperature_rules_fail_closed(model_id, module, temperature):
    parameters = _parameters(model_id, module, 1)
    with pytest.raises((TypeError, ValueError)):
        batch.equilibrium_batch(model_id, parameters, -40.0, temperature, np)


def test_invalid_candidate_documents_and_shapes_fail_closed():
    valid = _parameters(batch.KHALIQ_RAMAN_13_STATE, sodium, 3)
    incomplete = dict(valid[0])
    incomplete.pop("alpha_per_ms")
    with pytest.raises(ValueError, match="missing keys"):
        batch.equilibrium_batch(batch.KHALIQ_RAMAN_13_STATE, [incomplete], -40.0, None, np)
    with pytest.raises(TypeError, match="sequence"):
        batch.equilibrium_batch(batch.KHALIQ_RAMAN_13_STATE, valid[0], -40.0, None, np)
    with pytest.raises(ValueError, match="candidate axis"):
        batch.advance_batch(
            batch.KHALIQ_RAMAN_13_STATE, valid, -40.0,
            np.ones((13,), dtype=np.float64) / 13.0, 0.2, None, np,
        )
    states = batch.equilibrium_batch(batch.KHALIQ_RAMAN_13_STATE, valid, -80.0, None, np)
    with pytest.raises(ValueError, match="candidate axis"):
        batch.advance_batch(batch.KHALIQ_RAMAN_13_STATE, valid, -40.0, states[:2], 0.2, None, np)
    with pytest.raises(ValueError, match="nonempty one-dimensional"):
        batch.trace_batch(batch.KHALIQ_RAMAN_13_STATE, valid, -40.0, states, np.ones((1, 2)), None, np)
    with pytest.raises(ValueError, match="must sum to one"):
        batch.open_probability_batch(batch.KHALIQ_RAMAN_13_STATE, np.zeros((3, 13)), np)


def test_optional_cupy_candidate_batch_parity():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")

    for model_id, module, temperatures in MODELS:
        parameters = _parameters(model_id, module, 3)
        temperature = None if temperatures is None else temperatures
        voltage = np.array([-70.0, -20.0, 25.0])
        expected_equilibrium = batch.equilibrium_batch(model_id, parameters, voltage, temperature, np)
        actual_equilibrium = batch.equilibrium_batch(model_id, parameters, cupy.asarray(voltage), temperature, cupy)
        np.testing.assert_allclose(cupy.asnumpy(actual_equilibrium), expected_equilibrium, rtol=5e-8, atol=5e-10)
        expected = batch.trace_batch(model_id, parameters, voltage, expected_equilibrium, np.array([0.0, 0.02, 0.5]), temperature, np)
        actual = batch.trace_batch(model_id, parameters, cupy.asarray(voltage), actual_equilibrium, cupy.asarray([0.0, 0.02, 0.5]), temperature, cupy)
        np.testing.assert_allclose(cupy.asnumpy(actual), expected, rtol=5e-8, atol=5e-10)
