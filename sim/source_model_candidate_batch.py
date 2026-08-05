"""Candidate-batched evaluators for immutable Stage B source-model oracles.

The source modules remain the authoritative single-candidate implementation.
This module reproduces their published kinetic graphs under a candidate-first
array contract so a scientific screen can evaluate a complete candidate batch
without Python execution loops over candidates, commands, or elapsed times.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from sim import kv3_source_models as kv3
from sim import sodium_source_models as sodium


KHALIQ_RAMAN_13_STATE = sodium.KHALIQ_RAMAN_13_STATE
BALBI_NAV16_SIX_STATE = sodium.BALBI_NAV16_SIX_STATE
LABRO_2015 = kv3.LABRO_2015
DESAI_2008_CONTROL = kv3.DESAI_2008_CONTROL

_SODIUM_MODELS = frozenset((KHALIQ_RAMAN_13_STATE, BALBI_NAV16_SIX_STATE))
_KV3_MODELS = frozenset((LABRO_2015, DESAI_2008_CONTROL))
_MODEL_WIDTHS = {
    KHALIQ_RAMAN_13_STATE: 13,
    BALBI_NAV16_SIX_STATE: 6,
    LABRO_2015: 4,
    DESAI_2008_CONTROL: 2,
}
_BOLTZMANN_J_PER_K = 1.380649e-23
_ELEMENTARY_CHARGE_C = 1.602176634e-19

__all__ = [
    "BALBI_NAV16_SIX_STATE",
    "DESAI_2008_CONTROL",
    "KHALIQ_RAMAN_13_STATE",
    "LABRO_2015",
    "advance_batch",
    "equilibrium_batch",
    "open_probability_batch",
    "trace_batch",
]


def equilibrium_batch(
    model_id: str,
    candidate_parameters: Sequence[Mapping[str, Any]],
    voltage_mv: Any,
    temperature_c: Any,
    xp: Any,
) -> Any:
    """Return float64 stationary states with shape ``(candidate, *command, state)``.

    ``candidate_parameters`` must contain one complete source-parameter mapping
    per candidate. ``voltage_mv`` supplies command axes only, so every
    candidate is evaluated against the same command set. Temperature is a
    one-dimensional candidate vector only for models whose source permits it.
    """

    _validate_xp(xp)
    model_id = _validated_model_id(model_id)
    parameters = _pack_parameters(model_id, candidate_parameters, xp)
    temperature = _candidate_temperatures(
        model_id, temperature_c, parameters["_count"], xp
    )
    voltage = _finite_array(voltage_mv, "voltage_mv", xp)

    if model_id in _SODIUM_MODELS:
        result = _sodium_equilibrium(model_id, parameters, voltage, temperature, xp)
    else:
        alpha, beta = _kv3_rates(model_id, parameters, voltage, temperature, xp)
        result = _kv3_equilibrium(model_id, alpha, beta, xp)
    return _validated_result(model_id, result, xp)


def advance_batch(
    model_id: str,
    candidate_parameters: Sequence[Mapping[str, Any]],
    voltage_mv: Any,
    states: Any,
    duration_ms: Any,
    temperature_c: Any,
    xp: Any,
) -> Any:
    """Advance complete candidate batches across fixed-voltage command axes.

    ``states`` must have shape ``(candidate, *state_command, state)``. Its
    command dimensions, ``voltage_mv``, and ``duration_ms`` broadcast together;
    the candidate axis is never broadcast or treated as a command axis.
    """

    _validate_xp(xp)
    model_id = _validated_model_id(model_id)
    parameters = _pack_parameters(model_id, candidate_parameters, xp)
    count = parameters["_count"]
    temperature = _candidate_temperatures(model_id, temperature_c, count, xp)
    voltage = _finite_array(voltage_mv, "voltage_mv", xp)
    duration = _duration_array(duration_ms, xp)
    state_array, voltage, duration = _prepared_advance_inputs(
        model_id, states, count, voltage, duration, xp
    )

    if _scalar_bool(xp.all(duration == 0.0)):
        return state_array.copy()

    if model_id in _SODIUM_MODELS:
        result = _sodium_advance(
            model_id, parameters, voltage, state_array, duration, temperature, xp
        )
    else:
        alpha, beta = _kv3_rates(model_id, parameters, voltage, temperature, xp)
        result = _kv3_advance(model_id, state_array, alpha, beta, duration, xp)
    result = xp.where((duration == 0.0)[None, ..., None], state_array, result)
    return _validated_result(model_id, result, xp)


def trace_batch(
    model_id: str,
    candidate_parameters: Sequence[Mapping[str, Any]],
    voltage_mv: Any,
    states: Any,
    elapsed_ms: Any,
    temperature_c: Any,
    xp: Any,
) -> Any:
    """Evaluate all elapsed times without a Python loop over time points.

    The result has shape ``(candidate, *command, elapsed, state)``. This is a
    vectorized fixed-voltage trace, not sequential time stepping.
    """

    _validate_xp(xp)
    voltage = _finite_array(voltage_mv, "voltage_mv", xp)
    elapsed = _duration_array(elapsed_ms, xp)
    if elapsed.ndim != 1 or elapsed.size == 0:
        raise ValueError("elapsed_ms must be a nonempty one-dimensional array")

    expanded_voltage = xp.expand_dims(voltage, axis=-1)
    expanded_states = xp.expand_dims(xp.asarray(states, dtype=xp.float64), axis=-2)
    expanded_elapsed = elapsed.reshape((1,) * voltage.ndim + (elapsed.size,))
    return advance_batch(
        model_id,
        candidate_parameters,
        expanded_voltage,
        expanded_states,
        expanded_elapsed,
        temperature_c,
        xp,
    )


def open_probability_batch(model_id: str, states: Any, xp: Any) -> Any:
    """Return source-defined float64 open probability for candidate-batched states."""

    _validate_xp(xp)
    model_id = _validated_model_id(model_id)
    state_array = _validated_states(model_id, states, None, xp)
    if model_id == KHALIQ_RAMAN_13_STATE:
        return state_array[..., 10]
    if model_id == BALBI_NAV16_SIX_STATE:
        return state_array[..., 2] + state_array[..., 3]
    if model_id == LABRO_2015:
        return state_array[..., 3]
    return state_array[..., 0] ** 3 * (0.23 + 0.77 * state_array[..., 1])


def _validated_model_id(model_id: str) -> str:
    if not isinstance(model_id, str) or model_id not in _MODEL_WIDTHS:
        raise ValueError(f"unknown source model: {model_id!r}")
    return model_id


def _validate_xp(xp: Any) -> None:
    required = (
        "all",
        "any",
        "asarray",
        "broadcast_to",
        "concatenate",
        "cumsum",
        "einsum",
        "exp",
        "expand_dims",
        "eye",
        "imag",
        "isfinite",
        "linalg",
        "log",
        "matmul",
        "maximum",
        "real",
        "stack",
        "sum",
        "where",
        "zeros",
    )
    if xp is None or any(not hasattr(xp, name) for name in required):
        raise TypeError("xp must be a NumPy/CuPy-compatible array module")


def _scalar_bool(value: Any) -> bool:
    return bool(value)


def _finite_array(value: Any, name: str, xp: Any) -> Any:
    array = xp.asarray(value, dtype=xp.float64)
    if not _scalar_bool(xp.all(xp.isfinite(array))):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _duration_array(value: Any, xp: Any) -> Any:
    duration = _finite_array(value, "duration_ms", xp)
    if _scalar_bool(xp.any(duration < 0.0)):
        raise ValueError("duration_ms must be nonnegative")
    return duration


def _pack_parameters(
    model_id: str, candidate_parameters: Sequence[Mapping[str, Any]], xp: Any
) -> dict[str, Any]:
    if isinstance(candidate_parameters, (str, bytes, Mapping)) or not isinstance(
        candidate_parameters, Sequence
    ):
        raise TypeError("candidate_parameters must be a nonempty sequence of mappings")
    if not candidate_parameters:
        raise ValueError("candidate_parameters must not be empty")

    validator = (
        sodium.validate_source_parameters
        if model_id in _SODIUM_MODELS
        else kv3.validate_source_parameters
    )
    # This is host-side receipt validation and packing only. All kinetic
    # calculations below operate on the resulting candidate-batched arrays.
    documents = tuple(validator(model_id, item) for item in candidate_parameters)
    defaults = (
        sodium.SOURCE_PARAMETER_DEFAULTS[model_id]
        if model_id in _SODIUM_MODELS
        else kv3.SOURCE_PARAMETER_DEFAULTS[model_id]
    )
    packed: dict[str, Any] = {"_count": len(documents)}
    for name in defaults:
        packed[name] = xp.asarray(
            [document[name] for document in documents], dtype=xp.float64
        )
    return packed


def _candidate_temperatures(
    model_id: str, temperature_c: Any, count: int, xp: Any
) -> Any:
    if model_id in (KHALIQ_RAMAN_13_STATE, DESAI_2008_CONTROL):
        if temperature_c is not None:
            raise ValueError(f"temperature_c must be None for {model_id}")
        return None
    if isinstance(temperature_c, (str, bytes)):
        raise TypeError("temperature_c must be a finite candidate vector")
    try:
        host_temperature = np.asarray(temperature_c, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("temperature_c must be a finite candidate vector") from exc
    if host_temperature.shape != (count,):
        raise ValueError("temperature_c must have shape (candidate_count,)")
    if not np.all(np.isfinite(host_temperature)):
        raise ValueError("temperature_c must contain only finite values")
    if model_id == LABRO_2015 and np.any(host_temperature <= -273.15):
        raise ValueError("Labro temperature_c must be above absolute zero")
    return xp.asarray(host_temperature, dtype=xp.float64)


def _prepared_advance_inputs(
    model_id: str,
    states: Any,
    count: int,
    voltage: Any,
    duration: Any,
    xp: Any,
) -> tuple[Any, Any, Any]:
    state_array = _validated_states(model_id, states, count, xp)
    try:
        command_shape = np.broadcast_shapes(
            state_array.shape[1:-1], voltage.shape, duration.shape
        )
    except ValueError as exc:
        raise ValueError(
            "state command dimensions, voltage_mv, and duration_ms must broadcast"
        ) from exc
    state_command_shape = state_array.shape[1:-1]
    aligned_shape = (
        (count,)
        + (1,) * (len(command_shape) - len(state_command_shape))
        + state_command_shape
        + (_MODEL_WIDTHS[model_id],)
    )
    state_array = xp.broadcast_to(
        state_array.reshape(aligned_shape),
        (count,) + command_shape + (_MODEL_WIDTHS[model_id],),
    )
    return (
        state_array,
        xp.broadcast_to(voltage, command_shape),
        xp.broadcast_to(duration, command_shape),
    )


def _validated_states(model_id: str, states: Any, count: int | None, xp: Any) -> Any:
    state_array = _finite_array(states, "states", xp)
    width = _MODEL_WIDTHS[model_id]
    if state_array.ndim < 2 or state_array.shape[-1] != width:
        raise ValueError(
            f"states must have shape (candidate, ..., {width}) with a candidate axis"
        )
    if count is not None and state_array.shape[0] != count:
        raise ValueError("states candidate axis must match candidate_parameters")
    if model_id in _SODIUM_MODELS:
        if _scalar_bool(xp.any(state_array < -1e-9)):
            raise ValueError("sodium states must be nonnegative probabilities")
        if not _scalar_bool(
            xp.all(xp.abs(xp.sum(state_array, axis=-1) - 1.0) <= 1e-8)
        ):
            raise ValueError("sodium states must sum to one")
        return state_array

    tolerance = 1e-10
    if _scalar_bool(xp.any(state_array < -tolerance)) or _scalar_bool(
        xp.any(state_array > 1.0 + tolerance)
    ):
        raise ValueError("Kv3 state values must lie in [0, 1]")
    if model_id == LABRO_2015 and not _scalar_bool(
        xp.all(xp.abs(xp.sum(state_array, axis=-1) - 1.0) <= tolerance)
    ):
        raise ValueError("Labro occupancies must sum to one")
    return state_array


def _validated_result(model_id: str, states: Any, xp: Any) -> Any:
    if not _scalar_bool(xp.all(xp.isfinite(states))):
        raise FloatingPointError("source-model state contains non-finite values")
    if model_id in _SODIUM_MODELS:
        if _scalar_bool(xp.any(states < -1e-9)):
            raise FloatingPointError("sodium-model state contains negative probabilities")
        if not _scalar_bool(
            xp.all(xp.abs(xp.sum(states, axis=-1) - 1.0) <= 1e-8)
        ):
            raise FloatingPointError("sodium-model state does not conserve probability")
    elif model_id == LABRO_2015:
        if _scalar_bool(xp.any(states < -1e-10)):
            raise FloatingPointError("Labro state contains negative probabilities")
        if not _scalar_bool(
            xp.all(xp.abs(xp.sum(states, axis=-1) - 1.0) <= 1e-10)
        ):
            raise FloatingPointError("Labro state does not conserve probability")
    elif _scalar_bool(xp.any(states < -1e-10)) or _scalar_bool(
        xp.any(states > 1.0 + 1e-10)
    ):
        raise FloatingPointError("Desai gate values leave [0, 1]")
    return states


def _candidate_scalar(parameters: Mapping[str, Any], name: str, command_ndim: int) -> Any:
    return parameters[name].reshape((parameters["_count"],) + (1,) * command_ndim)


def _candidate_vector(parameters: Mapping[str, Any], name: str, command_ndim: int) -> Any:
    width = parameters[name].shape[-1]
    return parameters[name].reshape(
        (parameters["_count"],) + (1,) * command_ndim + (width,)
    )


def _add_reversible_edge(
    generator: Any, source: int, target: int, forward: Any, reverse: Any
) -> None:
    generator[..., target, source] += forward
    generator[..., source, source] -= forward
    generator[..., source, target] += reverse
    generator[..., target, target] -= reverse


def _sodium_generator(
    model_id: str, parameters: Mapping[str, Any], voltage: Any, temperature: Any, xp: Any
) -> Any:
    command_ndim = voltage.ndim
    candidate_voltage = voltage[None, ...]
    leading_shape = (parameters["_count"],) + voltage.shape
    if model_id == KHALIQ_RAMAN_13_STATE:
        generator = xp.zeros(leading_shape + (13, 13), dtype=xp.float64)
        alpha = _candidate_scalar(parameters, "alpha_per_ms", command_ndim) * xp.exp(
            candidate_voltage / _candidate_scalar(parameters, "x1_mv", command_ndim)
        )
        beta = _candidate_scalar(parameters, "beta_per_ms", command_ndim) * xp.exp(
            candidate_voltage / _candidate_scalar(parameters, "x2_mv", command_ndim)
        )
        alfac = (
            _candidate_scalar(parameters, "oon_per_ms", command_ndim)
            / _candidate_scalar(parameters, "con_per_ms", command_ndim)
        ) ** 0.25
        btfac = (
            _candidate_scalar(parameters, "ooff_per_ms", command_ndim)
            / _candidate_scalar(parameters, "coff_per_ms", command_ndim)
        ) ** 0.25

        _add_reversible_edge(generator, 0, 1, 4.0 * alpha, beta)
        _add_reversible_edge(generator, 1, 2, 3.0 * alpha, 2.0 * beta)
        _add_reversible_edge(generator, 2, 3, 2.0 * alpha, 3.0 * beta)
        _add_reversible_edge(generator, 3, 4, alpha, 4.0 * beta)
        _add_reversible_edge(
            generator,
            4,
            10,
            _candidate_scalar(parameters, "gamma_per_ms", command_ndim)
            * xp.exp(candidate_voltage / _candidate_scalar(parameters, "x3_mv", command_ndim)),
            _candidate_scalar(parameters, "delta_per_ms", command_ndim)
            * xp.exp(candidate_voltage / _candidate_scalar(parameters, "x4_mv", command_ndim)),
        )
        _add_reversible_edge(
            generator,
            10,
            11,
            _candidate_scalar(parameters, "epsilon_per_ms", command_ndim)
            * xp.exp(candidate_voltage / _candidate_scalar(parameters, "x5_mv", command_ndim)),
            _candidate_scalar(parameters, "zeta_per_ms", command_ndim)
            * xp.exp(candidate_voltage / _candidate_scalar(parameters, "x6_mv", command_ndim)),
        )
        _add_reversible_edge(
            generator,
            10,
            12,
            _candidate_scalar(parameters, "oon_per_ms", command_ndim),
            _candidate_scalar(parameters, "ooff_per_ms", command_ndim),
        )
        _add_reversible_edge(generator, 5, 6, 4.0 * alpha * alfac, beta * btfac)
        _add_reversible_edge(generator, 6, 7, 3.0 * alpha * alfac, 2.0 * beta * btfac)
        _add_reversible_edge(generator, 7, 8, 2.0 * alpha * alfac, 3.0 * beta * btfac)
        _add_reversible_edge(generator, 8, 9, alpha * alfac, 4.0 * beta * btfac)
        _add_reversible_edge(
            generator,
            9,
            12,
            _candidate_scalar(parameters, "gamma_per_ms", command_ndim)
            * xp.exp(candidate_voltage / _candidate_scalar(parameters, "x3_mv", command_ndim)),
            _candidate_scalar(parameters, "delta_per_ms", command_ndim)
            * xp.exp(candidate_voltage / _candidate_scalar(parameters, "x4_mv", command_ndim)),
        )
        con = _candidate_scalar(parameters, "con_per_ms", command_ndim)
        coff = _candidate_scalar(parameters, "coff_per_ms", command_ndim)
        _add_reversible_edge(generator, 0, 5, con, coff)
        _add_reversible_edge(generator, 1, 6, con * alfac, coff * btfac)
        _add_reversible_edge(generator, 2, 7, con * alfac**2, coff * btfac**2)
        _add_reversible_edge(generator, 3, 8, con * alfac**3, coff * btfac**3)
        _add_reversible_edge(generator, 4, 9, con * alfac**4, coff * btfac**4)
        return generator

    generator = xp.zeros(leading_shape + (6, 6), dtype=xp.float64)
    q10 = _candidate_scalar(parameters, "q10", command_ndim) ** (
        (temperature.reshape((parameters["_count"],) + (1,) * command_ndim)
        - _candidate_scalar(parameters, "q10_reference_temperature_c", command_ndim))
        / 10.0
    )

    def rate(name: str) -> Any:
        values = _candidate_vector(parameters, name, command_ndim)
        return values[..., 0] / (1.0 + xp.exp((candidate_voltage - values[..., 1]) / values[..., 2]))

    c1c2 = q10 * rate("c1c2")
    c2c1 = q10 * (rate("c2c1_extra") + rate("c1c2"))
    c2o1 = q10 * rate("c2o1")
    o1c2 = q10 * (rate("o1c2_extra") + rate("c2o1"))
    c2o2 = q10 * rate("c2o2")
    o2c2 = q10 * (rate("o2c2_first") + rate("o2c2_second"))
    o1i1 = q10 * (rate("o1i1_first") + rate("o1i1_second"))
    i1o1 = q10 * rate("i1o1")
    i1c1 = q10 * rate("i1c1")
    c1i1 = q10 * rate("c1i1")
    i1i2 = q10 * rate("i1i2")
    i2i1 = q10 * rate("i2i1")
    _add_reversible_edge(generator, 0, 1, c1c2, c2c1)
    _add_reversible_edge(generator, 1, 2, c2o1, o1c2)
    _add_reversible_edge(generator, 1, 3, c2o2, o2c2)
    _add_reversible_edge(generator, 2, 4, o1i1, i1o1)
    _add_reversible_edge(generator, 4, 0, i1c1, c1i1)
    _add_reversible_edge(generator, 4, 5, i1i2, i2i1)
    return generator


def _sodium_equilibrium(
    model_id: str, parameters: Mapping[str, Any], voltage: Any, temperature: Any, xp: Any
) -> Any:
    generator = _sodium_generator(model_id, parameters, voltage, temperature, xp)
    system = generator.copy()
    system[..., -1, :] = 1.0
    target = xp.zeros(system.shape[:-1], dtype=xp.float64)
    target[..., -1] = 1.0
    return xp.linalg.solve(system, target[..., None])[..., 0]


def _sodium_advance(
    model_id: str,
    parameters: Mapping[str, Any],
    voltage: Any,
    states: Any,
    duration: Any,
    temperature: Any,
    xp: Any,
) -> Any:
    generator = _sodium_generator(model_id, parameters, voltage, temperature, xp)
    eigenvalues, eigenvectors = xp.linalg.eig(generator)
    coefficients = xp.linalg.solve(eigenvectors, states[..., :, None])[..., 0]
    evolved = xp.matmul(
        eigenvectors,
        (xp.exp(eigenvalues * duration[None, ..., None]) * coefficients)[..., None],
    )[..., 0]
    imaginary_scale = xp.max(xp.abs(xp.imag(evolved)))
    if _scalar_bool(imaginary_scale > 1e-9):
        raise FloatingPointError("transition-matrix exponential produced a complex state")
    return xp.real(evolved)


def _kv3_rates(
    model_id: str, parameters: Mapping[str, Any], voltage: Any, temperature: Any, xp: Any
) -> tuple[Any, Any]:
    command_ndim = voltage.ndim
    candidate_voltage = voltage[None, ...]
    if model_id == LABRO_2015:
        thermal_voltage = (
            1000.0
            * _BOLTZMANN_J_PER_K
            * (temperature.reshape((parameters["_count"],) + (1,) * command_ndim) + 273.15)
            / _ELEMENTARY_CHARGE_C
        )
        exponent = (
            (candidate_voltage - _candidate_scalar(parameters, "vhalf_mv", command_ndim))[..., None]
            * _candidate_vector(parameters, "z", command_ndim)
            / thermal_voltage[..., None]
        )
        alpha = _candidate_vector(parameters, "alpha0_per_ms", command_ndim) * xp.exp(exponent)
        beta = _candidate_vector(parameters, "beta0_per_ms", command_ndim) * xp.exp(-exponent)
    else:
        alpha = _candidate_vector(parameters, "k_alpha_per_ms", command_ndim) * xp.exp(
            candidate_voltage[..., None]
            * _candidate_vector(parameters, "eta_alpha_per_mv", command_ndim)
        )
        beta = _candidate_vector(parameters, "k_beta_per_ms", command_ndim) * xp.exp(
            candidate_voltage[..., None]
            * _candidate_vector(parameters, "eta_beta_per_mv", command_ndim)
        )
    if not _scalar_bool(xp.all(xp.isfinite(alpha))) or not _scalar_bool(
        xp.all(xp.isfinite(beta))
    ):
        raise ValueError("voltage_mv produces non-finite source rates")
    return alpha, beta


def _kv3_equilibrium(model_id: str, alpha: Any, beta: Any, xp: Any) -> Any:
    if model_id == LABRO_2015:
        log_ratio = xp.log(alpha) - xp.log(beta)
        log_weight = xp.concatenate(
            (
                xp.zeros(log_ratio.shape[:-1] + (1,), dtype=xp.float64),
                xp.cumsum(log_ratio, axis=-1),
            ),
            axis=-1,
        )
        log_weight = log_weight - xp.max(log_weight, axis=-1, keepdims=True)
        weight = xp.exp(log_weight)
        return weight / xp.sum(weight, axis=-1, keepdims=True)
    return alpha / (alpha + beta)


def _kv3_advance(
    model_id: str, states: Any, alpha: Any, beta: Any, duration: Any, xp: Any
) -> Any:
    if model_id == DESAI_2008_CONTROL:
        steady_state = alpha / (alpha + beta)
        decay = xp.exp(-(alpha + beta) * duration[None, ..., None])
        return steady_state + (states - steady_state) * decay
    return _labro_advance(states, alpha, beta, duration, xp)


def _labro_advance(states: Any, alpha: Any, beta: Any, duration: Any, xp: Any) -> Any:
    generator = xp.zeros(states.shape[:-1] + (4, 4), dtype=xp.float64)
    generator[..., 0, 0] = -alpha[..., 0]
    generator[..., 1, 0] = alpha[..., 0]
    generator[..., 0, 1] = beta[..., 0]
    generator[..., 1, 1] = -(beta[..., 0] + alpha[..., 1])
    generator[..., 2, 1] = alpha[..., 1]
    generator[..., 1, 2] = beta[..., 1]
    generator[..., 2, 2] = -(beta[..., 1] + alpha[..., 2])
    generator[..., 3, 2] = alpha[..., 2]
    generator[..., 2, 3] = beta[..., 2]
    generator[..., 3, 3] = -beta[..., 2]

    exit_rate = xp.stack(
        (
            alpha[..., 0],
            beta[..., 0] + alpha[..., 1],
            beta[..., 1] + alpha[..., 2],
            beta[..., 2],
        ),
        axis=-1,
    )
    uniform_rate = xp.max(exit_rate, axis=-1)
    largest_mu = float(xp.max(uniform_rate * duration[None, ...]))
    if not math.isfinite(largest_mu):
        raise ValueError("duration_ms and voltage_mv produce non-finite Labro scale")
    squarings = max(0, math.ceil(math.log2(largest_mu / 0.5))) if largest_mu > 0.5 else 0
    if squarings > 60:
        raise ValueError("duration_ms and voltage_mv make the Labro solve ill-conditioned")

    scaled_duration = duration[None, ...] / (2**squarings)
    mu = uniform_rate * scaled_duration
    identity = xp.eye(4, dtype=xp.float64)
    stochastic = identity + generator / uniform_rate[..., None, None]
    term = xp.broadcast_to(identity, generator.shape).copy()
    weight = xp.exp(-mu)
    propagator = weight[..., None, None] * term
    # Fixed-order source algorithm: this is uniformization, not time stepping.
    for order in range(1, 33):
        term = xp.matmul(term, stochastic)
        weight = weight * mu / order
        propagator = propagator + weight[..., None, None] * term
    for _ in range(squarings):
        propagator = xp.matmul(propagator, propagator)
    result = xp.einsum("...ij,...j->...i", propagator, states)
    result = xp.maximum(result, 0.0)
    return result / xp.sum(result, axis=-1, keepdims=True)
