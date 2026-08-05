from types import MappingProxyType

import numpy as np
import pytest

import sim.sodium_source_models as sodium


KHALIQ = sodium.KHALIQ_RAMAN_13_STATE
BALBI = sodium.BALBI_NAV16_SIX_STATE


def test_exact_source_defaults_are_public_and_immutable():
    assert isinstance(sodium.SOURCE_PARAMETER_DEFAULTS, MappingProxyType)
    assert sodium.source_parameters(KHALIQ) is sodium.SOURCE_PARAMETER_DEFAULTS[KHALIQ]
    assert sodium.source_parameters(BALBI) is sodium.SOURCE_PARAMETER_DEFAULTS[BALBI]
    assert dict(sodium.source_parameters(KHALIQ)) == {
        "alpha_per_ms": 150.0,
        "beta_per_ms": 3.0,
        "gamma_per_ms": 150.0,
        "delta_per_ms": 40.0,
        "epsilon_per_ms": 1.75,
        "zeta_per_ms": 0.03,
        "con_per_ms": 0.005,
        "coff_per_ms": 0.5,
        "oon_per_ms": 0.75,
        "ooff_per_ms": 0.005,
        "x1_mv": 20.0,
        "x2_mv": -20.0,
        "x3_mv": 1e12,
        "x4_mv": -1e12,
        "x5_mv": 1e12,
        "x6_mv": -25.0,
    }
    assert sodium.source_parameters(BALBI)["c1c2"] == (14.0, -8.0, -10.0)
    assert sodium.source_parameters(BALBI)["i2i1"] == (0.0018, -90.0, 30.0)
    with pytest.raises(TypeError):
        sodium.SOURCE_PARAMETER_DEFAULTS[KHALIQ]["alpha_per_ms"] = 1.0


@pytest.mark.parametrize("model_id,temperature", [(KHALIQ, None), (BALBI, 22.0)])
def test_explicit_defaults_preserve_exact_outputs(model_id, temperature):
    parameters = sodium.source_parameters(model_id)
    voltage = np.array([-80.0, -20.0, 20.0])
    default_equilibrium = sodium.equilibrium(model_id, voltage, temperature, np)
    explicit_equilibrium = sodium.equilibrium(
        model_id, voltage, temperature, np, parameters=parameters
    )
    np.testing.assert_array_equal(explicit_equilibrium, default_equilibrium)

    default_advance = sodium.advance(
        model_id, voltage, default_equilibrium, 0.25, temperature, np
    )
    explicit_advance = sodium.advance(
        model_id,
        voltage,
        default_equilibrium,
        0.25,
        temperature,
        np,
        parameters=parameters,
    )
    np.testing.assert_array_equal(explicit_advance, default_advance)

    elapsed = np.array([0.0, 0.01, 0.25])
    np.testing.assert_array_equal(
        sodium.trace(
            model_id,
            voltage,
            default_equilibrium,
            elapsed,
            temperature,
            np,
            parameters=parameters,
        ),
        sodium.trace(
            model_id, voltage, default_equilibrium, elapsed, temperature, np
        ),
    )


def test_khaliq_override_changes_equilibrium_advance_and_trace():
    parameters = sodium.source_parameters(KHALIQ, {"alpha_per_ms": 120.0})
    initial = sodium.equilibrium(KHALIQ, -90.0, None, np)
    voltage = np.array([-40.0, 10.0])
    assert not np.array_equal(
        sodium.equilibrium(KHALIQ, voltage, None, np, parameters=parameters),
        sodium.equilibrium(KHALIQ, voltage, None, np),
    )
    assert not np.array_equal(
        sodium.advance(
            KHALIQ, voltage, initial, 0.5, None, np, parameters=parameters
        ),
        sodium.advance(KHALIQ, voltage, initial, 0.5, None, np),
    )
    assert not np.array_equal(
        sodium.trace(
            KHALIQ,
            voltage,
            initial,
            np.array([0.1, 0.5]),
            None,
            np,
            parameters=parameters,
        ),
        sodium.trace(KHALIQ, voltage, initial, np.array([0.1, 0.5]), None, np),
    )


def test_balbi_override_changes_equilibrium_advance_and_trace():
    parameters = sodium.source_parameters(BALBI, {"c1c2": (11.0, -4.0, -8.0)})
    initial = sodium.equilibrium(BALBI, -90.0, 22.0, np)
    voltage = np.array([-40.0, 10.0])
    assert not np.array_equal(
        sodium.equilibrium(BALBI, voltage, 22.0, np, parameters=parameters),
        sodium.equilibrium(BALBI, voltage, 22.0, np),
    )
    assert not np.array_equal(
        sodium.advance(
            BALBI, voltage, initial, 0.5, 22.0, np, parameters=parameters
        ),
        sodium.advance(BALBI, voltage, initial, 0.5, 22.0, np),
    )
    assert not np.array_equal(
        sodium.trace(
            BALBI,
            voltage,
            initial,
            np.array([0.1, 0.5]),
            22.0,
            np,
            parameters=parameters,
        ),
        sodium.trace(BALBI, voltage, initial, np.array([0.1, 0.5]), 22.0, np),
    )


@pytest.mark.parametrize(
    "model_id,parameters,match",
    [
        (KHALIQ, {"alpha_per_ms": 150.0}, "missing keys"),
        (
            KHALIQ,
            {**dict(sodium.SOURCE_PARAMETER_DEFAULTS[KHALIQ]), "unknown": 1.0},
            "extra keys",
        ),
        (
            KHALIQ,
            {**dict(sodium.SOURCE_PARAMETER_DEFAULTS[KHALIQ]), "alpha_per_ms": 0.0},
            "positive",
        ),
        (
            KHALIQ,
            {**dict(sodium.SOURCE_PARAMETER_DEFAULTS[KHALIQ]), "x1_mv": 0.0},
            "nonzero",
        ),
        (
            BALBI,
            {**dict(sodium.SOURCE_PARAMETER_DEFAULTS[BALBI]), "c1c2": (1.0, 2.0)},
            "exactly 3",
        ),
        (
            BALBI,
            {**dict(sodium.SOURCE_PARAMETER_DEFAULTS[BALBI]), "c1c2": (1.0, 2.0, 0.0)},
            "slope must be nonzero",
        ),
        (
            BALBI,
            {**dict(sodium.SOURCE_PARAMETER_DEFAULTS[BALBI]), "q10": np.nan},
            "finite",
        ),
    ],
)
def test_complete_parameter_documents_fail_closed(model_id, parameters, match):
    with pytest.raises((TypeError, ValueError), match=match):
        sodium.validate_source_parameters(model_id, parameters)


def test_override_builder_and_behavior_api_fail_closed():
    with pytest.raises(ValueError, match="extra keys"):
        sodium.source_parameters(KHALIQ, {"unknown": 1.0})
    with pytest.raises(TypeError, match="mapping"):
        sodium.source_parameters(BALBI, [1.0])
    with pytest.raises(ValueError, match="missing keys"):
        sodium.advance(
            BALBI,
            -20.0,
            np.ones(6) / 6.0,
            0.0,
            22.0,
            np,
            parameters={"q10": 3.0},
        )


def test_parameterized_numpy_cupy_parity_when_cupy_is_available():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")

    for model_id, temperature, parameters in (
        (KHALIQ, None, sodium.source_parameters(KHALIQ, {"alpha_per_ms": 120.0})),
        (BALBI, 22.0, sodium.source_parameters(BALBI, {"c1c2": (11.0, -4.0, -8.0)})),
    ):
        voltage = np.array([-35.0, 15.0])
        initial = sodium.equilibrium(
            model_id, -80.0, temperature, np, parameters=parameters
        )
        expected = sodium.advance(
            model_id,
            voltage,
            initial,
            0.2,
            temperature,
            np,
            parameters=parameters,
        )
        actual = sodium.advance(
            model_id,
            cupy.asarray(voltage),
            cupy.asarray(initial),
            0.2,
            temperature,
            cupy,
            parameters=parameters,
        )
        np.testing.assert_allclose(
            cupy.asnumpy(actual), expected, rtol=2e-9, atol=2e-11
        )
