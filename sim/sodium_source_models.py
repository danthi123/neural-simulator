"""Source-faithful fixed-voltage sodium-channel comparators.

The transition equations and state order are transcribed from immutable
author-supplied NMODL files.  States are probabilities in the final axis.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any


KHALIQ_RAMAN_13_STATE = "khaliq_raman_13_state"
BALBI_NAV16_SIX_STATE = "balbi_nav16_six_state"

MODEL_METADATA = MappingProxyType(
    {
        KHALIQ_RAMAN_13_STATE: MappingProxyType(
            {
                "model_id": KHALIQ_RAMAN_13_STATE,
                "source_file": "rsg.mod",
                "source_sha256": "1a3382714bd0962665ec31f7dfac2aa3a9e403a5e3d23e29851afec232c4543e",
                "source_commit": "c96405173a17d18999d2a8d63d40899a76d02bdf",
                "source_url": (
                    "https://raw.githubusercontent.com/ModelDBRepository/48332/"
                    "c96405173a17d18999d2a8d63d40899a76d02bdf/rsg.mod"
                ),
                "states": (
                    "C1",
                    "C2",
                    "C3",
                    "C4",
                    "C5",
                    "I1",
                    "I2",
                    "I3",
                    "I4",
                    "I5",
                    "O",
                    "B",
                    "I6",
                ),
                "open_states": ("O",),
                "q10": None,
                "q10_reference_temperature_c": None,
                "source_native_initialization": "custom seqinitial equations in rsg.mod",
                "transfer_initialization": (
                    "stationary distribution of the declared kinetic graph; "
                    "assay-defined and not equivalent to source seqinitial"
                ),
            }
        ),
        BALBI_NAV16_SIX_STATE: MappingProxyType(
            {
                "model_id": BALBI_NAV16_SIX_STATE,
                "source_file": "Nav16_a.mod",
                "source_sha256": "69931ced1587944070edb3169a865e9e3e2a42f715b19a8b7b57e72e831ba71d",
                "source_commit": "815a1d7762d0cdccc3a3c6e6bed3a678d15888e4",
                "source_url": (
                    "https://raw.githubusercontent.com/ModelDBRepository/230137/"
                    "815a1d7762d0cdccc3a3c6e6bed3a678d15888e4/Nav16_a.mod"
                ),
                "states": ("C1", "C2", "O1", "O2", "I1", "I2"),
                "open_states": ("O1", "O2"),
                "q10": 3.0,
                "q10_reference_temperature_c": 20.0,
                "source_package_temperature_c": 22.0,
                "source_native_initialization": "stationary distribution of the kinetic graph",
                "transfer_initialization": "stationary distribution of the kinetic graph",
            }
        ),
    }
)

__all__ = [
    "BALBI_NAV16_SIX_STATE",
    "KHALIQ_RAMAN_13_STATE",
    "MODEL_METADATA",
    "advance",
    "equilibrium",
    "model_metadata",
    "open_probability",
]


def model_metadata(model_id: str) -> MappingProxyType:
    """Return immutable provenance and state-layout metadata for one model."""

    return MODEL_METADATA[_validated_model_id(model_id)]


def equilibrium(model_id: str, voltage_mv: Any, temperature_c: Any, xp: Any) -> Any:
    """Return the stationary state probabilities at a fixed voltage."""

    model_id = _validated_model_id(model_id)
    voltage = _validated_voltage(voltage_mv, xp)
    temperature = _validated_temperature(model_id, temperature_c, xp)
    generator = _generator(model_id, voltage, temperature, xp)

    system = generator.copy()
    system[..., -1, :] = 1.0
    target = xp.zeros(voltage.shape + (_state_count(model_id),), dtype=xp.float64)
    target[..., -1] = 1.0
    result = xp.linalg.solve(system, target[..., None])[..., 0]
    return _validated_result(result, xp)


def advance(
    model_id: str,
    voltage_mv: Any,
    states: Any,
    duration_ms: Any,
    temperature_c: Any,
    xp: Any,
) -> Any:
    """Advance probabilities exactly over one constant-voltage interval.

    A batched eigendecomposition evaluates the matrix exponential.  This has
    no Python loop over simulation time and leaves the array backend in charge
    of all numerical work.
    """

    model_id = _validated_model_id(model_id)
    voltage = _validated_voltage(voltage_mv, xp)
    temperature = _validated_temperature(model_id, temperature_c, xp)
    duration = _validated_duration(duration_ms, xp)

    state_array = _validated_states(model_id, states, xp)
    try:
        voltage, duration, state_probe = xp.broadcast_arrays(
            voltage,
            duration,
            state_array[..., 0],
        )
    except ValueError as exc:
        raise ValueError(
            "voltage_mv, duration_ms, and states batch dimensions are incompatible"
        ) from exc
    state_array = xp.broadcast_to(state_array, state_probe.shape + (state_array.shape[-1],))

    if _scalar_bool(xp.all(duration == 0.0)):
        return state_array.copy()

    generator = _generator(model_id, voltage, temperature, xp)
    eigenvalues, eigenvectors = xp.linalg.eig(generator)
    coefficients = xp.linalg.solve(eigenvectors, state_array[..., :, None])[..., 0]
    evolved_coefficients = xp.exp(eigenvalues * duration[..., None]) * coefficients
    evolved = xp.matmul(eigenvectors, evolved_coefficients[..., None])[..., 0]

    imaginary_scale = xp.max(xp.abs(xp.imag(evolved)))
    if _scalar_bool(imaginary_scale > 1e-9):
        raise FloatingPointError("transition-matrix exponential produced a complex state")
    evolved = xp.real(evolved)
    evolved = xp.where((duration == 0.0)[..., None], state_array, evolved)
    return _validated_result(evolved, xp)


def open_probability(model_id: str, states: Any, xp: Any) -> Any:
    """Return source-defined open probability from state probabilities."""

    model_id = _validated_model_id(model_id)
    state_array = _validated_states(model_id, states, xp)
    if model_id == KHALIQ_RAMAN_13_STATE:
        return state_array[..., 10]
    return state_array[..., 2] + state_array[..., 3]


def _validated_model_id(model_id: str) -> str:
    if not isinstance(model_id, str) or model_id not in MODEL_METADATA:
        raise ValueError(f"unknown sodium source model: {model_id!r}")
    return model_id


def _state_count(model_id: str) -> int:
    return len(MODEL_METADATA[model_id]["states"])


def _scalar_bool(value: Any) -> bool:
    return bool(value)


def _validated_scalar(value: Any, name: str, xp: Any) -> Any:
    array = xp.asarray(value, dtype=xp.float64)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    if not _scalar_bool(xp.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _validated_temperature(model_id: str, temperature_c: Any, xp: Any) -> Any:
    if model_id == KHALIQ_RAMAN_13_STATE:
        if temperature_c is not None:
            raise ValueError("temperature_c must be None for the Khaliq/Raman model")
        return None
    if temperature_c is None:
        raise ValueError("temperature_c must be explicit for the Balbi Nav1.6 model")
    return _validated_scalar(temperature_c, "temperature_c", xp)


def _validated_duration(duration_ms: Any, xp: Any) -> Any:
    duration = xp.asarray(duration_ms, dtype=xp.float64)
    if not _scalar_bool(xp.all(xp.isfinite(duration))):
        raise ValueError("duration_ms must be finite")
    if _scalar_bool(xp.any(duration < 0.0)):
        raise ValueError("duration_ms must be nonnegative")
    return duration


def _validated_voltage(voltage_mv: Any, xp: Any) -> Any:
    voltage = xp.asarray(voltage_mv, dtype=xp.float64)
    if not _scalar_bool(xp.all(xp.isfinite(voltage))):
        raise ValueError("voltage_mv must be finite")
    return voltage


def _validated_states(model_id: str, states: Any, xp: Any) -> Any:
    state_array = xp.asarray(states, dtype=xp.float64)
    expected = _state_count(model_id)
    if state_array.ndim < 1 or state_array.shape[-1] != expected:
        raise ValueError(f"states must have final dimension {expected}")
    if not _scalar_bool(xp.all(xp.isfinite(state_array))):
        raise ValueError("states must be finite")
    if _scalar_bool(xp.any(state_array < -1e-12)):
        raise ValueError("states must be nonnegative probabilities")
    if not _scalar_bool(xp.all(xp.abs(xp.sum(state_array, axis=-1) - 1.0) <= 1e-10)):
        raise ValueError("states must sum to one")
    return state_array


def _validated_result(states: Any, xp: Any) -> Any:
    if not _scalar_bool(xp.all(xp.isfinite(states))):
        raise FloatingPointError("non-finite sodium-model state")
    if _scalar_bool(xp.any(states < -1e-9)):
        raise FloatingPointError("negative sodium-model state")
    if not _scalar_bool(xp.all(xp.abs(xp.sum(states, axis=-1) - 1.0) <= 1e-8)):
        raise FloatingPointError("sodium-model state does not conserve probability")
    return states


def _add_reversible_edge(
    generator: Any,
    source: int,
    target: int,
    forward: Any,
    reverse: Any,
) -> None:
    generator[..., target, source] += forward
    generator[..., source, source] -= forward
    generator[..., source, target] += reverse
    generator[..., target, target] -= reverse


def _generator(model_id: str, voltage: Any, temperature: Any, xp: Any) -> Any:
    if model_id == KHALIQ_RAMAN_13_STATE:
        return _khaliq_generator(voltage, xp)
    return _balbi_generator(voltage, temperature, xp)


def _khaliq_generator(voltage: Any, xp: Any) -> Any:
    generator = xp.zeros(voltage.shape + (13, 13), dtype=xp.float64)
    activation = 150.0 * xp.exp(voltage / 20.0)
    deactivation = 3.0 * xp.exp(voltage / -20.0)
    alfac = (0.75 / 0.005) ** 0.25
    btfac = (0.005 / 0.5) ** 0.25

    # C1-C5-O activation chain.
    _add_reversible_edge(generator, 0, 1, 4.0 * activation, 1.0 * deactivation)
    _add_reversible_edge(generator, 1, 2, 3.0 * activation, 2.0 * deactivation)
    _add_reversible_edge(generator, 2, 3, 2.0 * activation, 3.0 * deactivation)
    _add_reversible_edge(generator, 3, 4, 1.0 * activation, 4.0 * deactivation)
    _add_reversible_edge(
        generator,
        4,
        10,
        150.0 * xp.exp(voltage / 1e12),
        40.0 * xp.exp(voltage / -1e12),
    )

    # Open-channel block and open-state inactivation.
    _add_reversible_edge(
        generator,
        10,
        11,
        1.75 * xp.exp(voltage / 1e12),
        0.03 * xp.exp(voltage / -25.0),
    )
    _add_reversible_edge(generator, 10, 12, 0.75, 0.005)

    # I1-I6 activation chain.
    _add_reversible_edge(generator, 5, 6, 4.0 * activation * alfac, deactivation * btfac)
    _add_reversible_edge(generator, 6, 7, 3.0 * activation * alfac, 2.0 * deactivation * btfac)
    _add_reversible_edge(generator, 7, 8, 2.0 * activation * alfac, 3.0 * deactivation * btfac)
    _add_reversible_edge(generator, 8, 9, activation * alfac, 4.0 * deactivation * btfac)
    _add_reversible_edge(
        generator,
        9,
        12,
        150.0 * xp.exp(voltage / 1e12),
        40.0 * xp.exp(voltage / -1e12),
    )

    # Closed-state inactivation couplings.
    _add_reversible_edge(generator, 0, 5, 0.005, 0.5)
    _add_reversible_edge(generator, 1, 6, 0.005 * alfac, 0.5 * btfac)
    _add_reversible_edge(generator, 2, 7, 0.005 * alfac**2, 0.5 * btfac**2)
    _add_reversible_edge(generator, 3, 8, 0.005 * alfac**3, 0.5 * btfac**3)
    _add_reversible_edge(generator, 4, 9, 0.005 * alfac**4, 0.5 * btfac**4)
    return generator


def _rates2(voltage: Any, b: float, midpoint: float, slope: float, xp: Any) -> Any:
    return b / (1.0 + xp.exp((voltage - midpoint) / slope))


def _balbi_generator(voltage: Any, temperature: Any, xp: Any) -> Any:
    generator = xp.zeros(voltage.shape + (6, 6), dtype=xp.float64)
    q10 = 3.0 ** ((temperature - 20.0) / 10.0)

    c1c2 = q10 * _rates2(voltage, 14.0, -8.0, -10.0, xp)
    c2c1 = q10 * (
        _rates2(voltage, 2.0, -38.0, 9.0, xp)
        + _rates2(voltage, 14.0, -8.0, -10.0, xp)
    )
    c2o1 = q10 * _rates2(voltage, 14.0, -18.0, -10.0, xp)
    o1c2 = q10 * (
        _rates2(voltage, 4.0, -48.0, 9.0, xp)
        + _rates2(voltage, 14.0, -18.0, -10.0, xp)
    )
    c2o2 = q10 * _rates2(voltage, 0.0001, -10.0, -8.0, xp)
    o2c2 = q10 * (
        _rates2(voltage, 0.0001, -55.0, 10.0, xp)
        + _rates2(voltage, 0.0001, -20.0, -5.0, xp)
    )
    o1i1 = q10 * (
        _rates2(voltage, 6.0, -40.0, 13.0, xp)
        + _rates2(voltage, 10.0, 15.0, -18.0, xp)
    )
    i1o1 = q10 * _rates2(voltage, 0.00001, -40.0, 10.0, xp)
    i1c1 = q10 * _rates2(voltage, 0.1, -86.0, 9.0, xp)
    c1i1 = q10 * _rates2(voltage, 0.08, -55.0, -12.0, xp)
    i1i2 = q10 * _rates2(voltage, 0.00022, -50.0, -5.0, xp)
    i2i1 = q10 * _rates2(voltage, 0.0018, -90.0, 30.0, xp)

    _add_reversible_edge(generator, 0, 1, c1c2, c2c1)
    _add_reversible_edge(generator, 1, 2, c2o1, o1c2)
    _add_reversible_edge(generator, 1, 3, c2o2, o2c2)
    _add_reversible_edge(generator, 2, 4, o1i1, i1o1)
    _add_reversible_edge(generator, 4, 0, i1c1, c1i1)
    _add_reversible_edge(generator, 4, 5, i1i2, i2i1)
    return generator
