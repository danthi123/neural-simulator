from types import MappingProxyType

import numpy as np
import pytest

from sim.kv3_source_models import (
    DESAI_2008_CONTROL,
    LABRO_2015,
    SOURCE_PARAMETER_DEFAULTS,
    advance,
    equilibrium,
    rates,
    source_parameters,
    validate_source_parameters,
)


def test_exact_source_defaults_are_public_and_immutable():
    assert isinstance(SOURCE_PARAMETER_DEFAULTS, MappingProxyType)
    assert source_parameters(LABRO_2015) is SOURCE_PARAMETER_DEFAULTS[LABRO_2015]
    assert source_parameters(DESAI_2008_CONTROL) is SOURCE_PARAMETER_DEFAULTS[
        DESAI_2008_CONTROL
    ]
    assert dict(source_parameters(LABRO_2015)) == {
        "alpha0_per_ms": (0.05, 6.0, 1.0),
        "beta0_per_ms": (0.15, 0.6, 0.8),
        "z": (3.5, 0.4, 0.001),
        "vhalf_mv": 6.2,
    }
    assert dict(source_parameters(DESAI_2008_CONTROL)) == {
        "k_alpha_per_ms": (0.039, 0.000045),
        "eta_alpha_per_mv": (0.0467, -0.18925),
        "k_beta_per_ms": (0.0868, 0.00246),
        "eta_beta_per_mv": (0.0067, 0.01075),
    }
    with pytest.raises(TypeError):
        SOURCE_PARAMETER_DEFAULTS[LABRO_2015]["vhalf_mv"] = 0.0


@pytest.mark.parametrize(
    "model_id,temperature_c",
    [(LABRO_2015, 22.5), (DESAI_2008_CONTROL, None)],
)
def test_explicit_defaults_preserve_exact_outputs(model_id, temperature_c):
    voltage = np.array([-60.0, -10.0, 30.0])
    defaults = source_parameters(model_id)
    default_rates = rates(model_id, voltage, temperature_c, np)
    explicit_rates = rates(
        model_id, voltage, temperature_c, np, parameters=defaults
    )
    for actual, expected in zip(explicit_rates, default_rates, strict=True):
        np.testing.assert_array_equal(actual, expected)

    default_equilibrium = equilibrium(model_id, voltage, temperature_c, np)
    explicit_equilibrium = equilibrium(
        model_id, voltage, temperature_c, np, parameters=defaults
    )
    np.testing.assert_array_equal(explicit_equilibrium, default_equilibrium)
    np.testing.assert_array_equal(
        advance(
            model_id,
            default_equilibrium,
            voltage,
            2.5,
            temperature_c,
            np,
            parameters=defaults,
        ),
        advance(
            model_id, default_equilibrium, voltage, 2.5, temperature_c, np
        ),
    )


def test_labro_override_changes_rates_equilibrium_and_advance():
    parameters = source_parameters(LABRO_2015, {"vhalf_mv": -3.0})
    voltage = np.array([-20.0, 20.0])
    default_rates = rates(LABRO_2015, voltage, 22.5, np)
    overridden_rates = rates(
        LABRO_2015, voltage, 22.5, np, parameters=parameters
    )
    assert not np.array_equal(overridden_rates[0], default_rates[0])
    assert not np.array_equal(overridden_rates[1], default_rates[1])

    initial = equilibrium(LABRO_2015, -80.0, 22.5, np)
    assert not np.array_equal(
        equilibrium(LABRO_2015, voltage, 22.5, np, parameters=parameters),
        equilibrium(LABRO_2015, voltage, 22.5, np),
    )
    assert not np.array_equal(
        advance(
            LABRO_2015,
            initial,
            voltage,
            1.0,
            22.5,
            np,
            parameters=parameters,
        ),
        advance(LABRO_2015, initial, voltage, 1.0, 22.5, np),
    )


def test_desai_override_changes_rates_equilibrium_and_advance():
    parameters = source_parameters(
        DESAI_2008_CONTROL, {"k_alpha_per_ms": (0.078, 0.00009)}
    )
    voltage = np.array([-30.0, 25.0])
    default_rates = rates(DESAI_2008_CONTROL, voltage, None, np)
    overridden_rates = rates(
        DESAI_2008_CONTROL, voltage, None, np, parameters=parameters
    )
    assert not np.array_equal(overridden_rates[0], default_rates[0])

    initial = equilibrium(DESAI_2008_CONTROL, -80.0, None, np)
    assert not np.array_equal(
        equilibrium(
            DESAI_2008_CONTROL, voltage, None, np, parameters=parameters
        ),
        equilibrium(DESAI_2008_CONTROL, voltage, None, np),
    )
    assert not np.array_equal(
        advance(
            DESAI_2008_CONTROL,
            initial,
            voltage,
            1.0,
            None,
            np,
            parameters=parameters,
        ),
        advance(DESAI_2008_CONTROL, initial, voltage, 1.0, None, np),
    )


@pytest.mark.parametrize(
    "parameters,match",
    [
        ({"alpha0_per_ms": (0.05, 6.0, 1.0)}, "missing keys"),
        (
            {
                **dict(SOURCE_PARAMETER_DEFAULTS[LABRO_2015]),
                "unknown": 1.0,
            },
            "extra keys",
        ),
        (
            {
                **dict(SOURCE_PARAMETER_DEFAULTS[LABRO_2015]),
                "alpha0_per_ms": (0.05, np.nan, 1.0),
            },
            "finite",
        ),
        (
            {
                **dict(SOURCE_PARAMETER_DEFAULTS[LABRO_2015]),
                "z": (3.5, 0.0, 0.001),
            },
            "positive",
        ),
        (
            {
                **dict(SOURCE_PARAMETER_DEFAULTS[LABRO_2015]),
                "beta0_per_ms": (0.15, 0.6),
            },
            "exactly 3",
        ),
    ],
)
def test_complete_parameter_documents_fail_closed(parameters, match):
    with pytest.raises((TypeError, ValueError), match=match):
        validate_source_parameters(LABRO_2015, parameters)


def test_override_builder_rejects_unknown_or_invalid_values():
    with pytest.raises(ValueError, match="extra keys"):
        source_parameters(DESAI_2008_CONTROL, {"unknown": 1.0})
    with pytest.raises(ValueError, match="positive"):
        source_parameters(
            DESAI_2008_CONTROL, {"k_beta_per_ms": (0.0, 0.00246)}
        )
    with pytest.raises(ValueError, match="finite"):
        source_parameters(DESAI_2008_CONTROL, {"eta_alpha_per_mv": (np.inf, 0.1)})


def test_behavior_api_rejects_incomplete_parameters_even_for_zero_duration():
    with pytest.raises(ValueError, match="missing keys"):
        advance(
            LABRO_2015,
            [1.0, 0.0, 0.0, 0.0],
            -20.0,
            0.0,
            22.5,
            np,
            parameters={"vhalf_mv": 6.2},
        )


def test_parameterized_numpy_cupy_parity_when_cupy_is_available():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")

    parameters = source_parameters(LABRO_2015, {"vhalf_mv": 2.5})
    voltage = np.array([-35.0, 15.0])
    initial = equilibrium(LABRO_2015, -70.0, 22.5, np, parameters=parameters)
    expected = advance(
        LABRO_2015, initial, voltage, 2.0, 22.5, np, parameters=parameters
    )
    actual = advance(
        LABRO_2015,
        cupy.asarray(initial),
        cupy.asarray(voltage),
        2.0,
        22.5,
        cupy,
        parameters=parameters,
    )
    np.testing.assert_allclose(cupy.asnumpy(actual), expected, atol=2e-12, rtol=2e-12)
