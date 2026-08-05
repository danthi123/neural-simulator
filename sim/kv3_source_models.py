"""Unmodified source comparators for Kv3 channel kinetics.

The Labro and Desai models are intentionally separate.  This module does not
add Desai availability to the Labro graph or otherwise complete either source
with transitions from the other.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any
import math


LABRO_2015 = "labro_2015_four_state"
DESAI_2008_CONTROL = "desai_2008_control"

# CODATA 2018 exact SI definitions, converted from volts to millivolts.
_BOLTZMANN_J_PER_K = 1.380649e-23
_ELEMENTARY_CHARGE_C = 1.602176634e-19

MODEL_METADATA = MappingProxyType(
    {
        LABRO_2015: MappingProxyType(
            {
                "citation": (
                    "Labro AJ et al. Nature Communications 6:10173 (2015), "
                    "Figure 7, Methods Simulations, Supplementary Table 1"
                ),
                "channel": "human Kv3.1b",
                "preparation": "Xenopus oocyte",
                "source_temperature_c": None,
                "reported_temperature": "room temperature; exact value unresolved",
                "temperature_convention": (
                    "explicit prospectively sealed execution temperature; no Q10"
                ),
                "state_names": (
                    "resting_closed",
                    "pre_active_closed",
                    "relaxed_pre_active_closed",
                    "relaxed_active_open",
                ),
                "open_state": "relaxed_active_open",
                "has_inactivation_state": False,
                "open_probability": "P(relaxed_active_open)",
                "voltage_unit": "mV",
                "time_unit": "ms",
            }
        ),
        DESAI_2008_CONTROL: MappingProxyType(
            {
                "citation": (
                    "Desai R et al. Journal of Biological Chemistry "
                    "283:22283-22294 (2008), numerical simulation section"
                ),
                "channel": "mouse Kv3.3 control",
                "preparation": "CHO cell",
                "source_temperature_c": None,
                "reported_temperature": None,
                "temperature_convention": (
                    "published equations contain no temperature term; API requires None"
                ),
                "state_names": ("n_activation", "p_availability"),
                "has_inactivation_state": True,
                "open_probability": "n^3 * (0.23 + 0.77*p)",
                "voltage_unit": "mV",
                "time_unit": "ms",
            }
        ),
    }
)

_LABRO_ALPHA0_PER_MS = (0.05, 6.0, 1.0)
_LABRO_BETA0_PER_MS = (0.15, 0.6, 0.8)
_LABRO_Z = (3.5, 0.4, 0.001)
_LABRO_VHALF_MV = 6.2

_DESAI_K_ALPHA_PER_MS = (0.039, 0.000045)
_DESAI_ETA_ALPHA_PER_MV = (0.0467, -0.18925)
_DESAI_K_BETA_PER_MS = (0.0868, 0.00246)
_DESAI_ETA_BETA_PER_MV = (0.0067, 0.01075)


def _metadata(model_id: str) -> MappingProxyType:
    try:
        return MODEL_METADATA[model_id]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unknown Kv3 source model: {model_id!r}") from exc


def _validate_xp(xp: Any) -> None:
    required = ("asarray", "exp", "isfinite", "stack", "zeros", "linalg")
    if xp is None or any(not hasattr(xp, name) for name in required):
        raise TypeError("xp must be a NumPy/CuPy-compatible array module")


def _validate_temperature(model_id: str, temperature_c: float | None) -> float | None:
    _metadata(model_id)
    if model_id == DESAI_2008_CONTROL:
        if temperature_c is not None:
            raise ValueError(
                "Desai published no temperature term; temperature_c must be None"
            )
        return None
    if temperature_c is None:
        raise TypeError("Labro temperature_c must be an explicit finite scalar")
    try:
        temperature = float(temperature_c)
    except (TypeError, ValueError) as exc:
        raise TypeError("temperature_c must be a finite scalar") from exc
    if not math.isfinite(temperature):
        raise ValueError("temperature_c must be finite")
    if temperature <= -273.15:
        raise ValueError("temperature_c must be above absolute zero")
    return temperature


def _finite_array(value: Any, name: str, xp: Any):
    array = xp.asarray(value, dtype=xp.float64)
    if not bool(xp.all(xp.isfinite(array))):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _thermal_voltage_mv(temperature_c: float) -> float:
    kelvin = temperature_c + 273.15
    return 1000.0 * _BOLTZMANN_J_PER_K * kelvin / _ELEMENTARY_CHARGE_C


def rates(model_id: str, voltage_mv: Any, temperature_c: float | None, xp: Any):
    """Return source rates as ``(alpha, beta)`` with final axis by transition."""

    _validate_xp(xp)
    temperature = _validate_temperature(model_id, temperature_c)
    voltage = _finite_array(voltage_mv, "voltage_mv", xp)

    if model_id == LABRO_2015:
        alpha0 = xp.asarray(_LABRO_ALPHA0_PER_MS, dtype=xp.float64)
        beta0 = xp.asarray(_LABRO_BETA0_PER_MS, dtype=xp.float64)
        z = xp.asarray(_LABRO_Z, dtype=xp.float64)
        exponent = (
            (voltage[..., None] - _LABRO_VHALF_MV)
            * z
            / _thermal_voltage_mv(temperature)
        )
        alpha = alpha0 * xp.exp(exponent)
        beta = beta0 * xp.exp(-exponent)
    else:
        k_alpha = xp.asarray(_DESAI_K_ALPHA_PER_MS, dtype=xp.float64)
        eta_alpha = xp.asarray(_DESAI_ETA_ALPHA_PER_MV, dtype=xp.float64)
        k_beta = xp.asarray(_DESAI_K_BETA_PER_MS, dtype=xp.float64)
        eta_beta = xp.asarray(_DESAI_ETA_BETA_PER_MV, dtype=xp.float64)
        alpha = k_alpha * xp.exp(voltage[..., None] * eta_alpha)
        beta = k_beta * xp.exp(voltage[..., None] * eta_beta)

    if not bool(xp.all(xp.isfinite(alpha))) or not bool(xp.all(xp.isfinite(beta))):
        raise ValueError("voltage_mv produces non-finite source rates")
    return alpha, beta


def equilibrium(
    model_id: str, voltage_mv: Any, temperature_c: float | None, xp: Any
):
    """Return the fixed-voltage equilibrium state for one source model."""

    alpha, beta = rates(model_id, voltage_mv, temperature_c, xp)
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

    total = alpha + beta
    return alpha / total


def _validate_state(model_id: str, state: Any, xp: Any):
    state_array = _finite_array(state, "state", xp)
    width = 4 if model_id == LABRO_2015 else 2
    if state_array.ndim < 1 or state_array.shape[-1] != width:
        raise ValueError(f"{model_id} state must have shape (..., {width})")
    tolerance = 1e-10
    if bool(xp.any(state_array < -tolerance)) or bool(xp.any(state_array > 1.0 + tolerance)):
        raise ValueError("state values must lie in [0, 1]")
    if model_id == LABRO_2015:
        total = xp.sum(state_array, axis=-1)
        if not bool(xp.all(xp.abs(total - 1.0) <= tolerance)):
            raise ValueError("Labro occupancies must sum to one")
    return state_array


def _labro_advance(state: Any, alpha: Any, beta: Any, duration_ms: Any, xp: Any):
    """Apply a batched conservative uniformization matrix exponential."""

    generator = xp.zeros(state.shape[:-1] + (4, 4), dtype=xp.float64)
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
    largest_mu = float(xp.max(uniform_rate * duration_ms))
    if not math.isfinite(largest_mu):
        raise ValueError("duration_ms and voltage_mv produce non-finite Labro scale")
    squarings = max(0, math.ceil(math.log2(largest_mu / 0.5))) if largest_mu > 0.5 else 0
    if squarings > 60:
        raise ValueError("duration_ms and voltage_mv make the Labro solve ill-conditioned")

    scaled_duration = duration_ms / (2**squarings)
    mu = uniform_rate * scaled_duration
    identity = xp.eye(4, dtype=xp.float64)
    stochastic = identity + generator / uniform_rate[..., None, None]

    # With mu <= 0.5, 32 Poisson terms are well beyond float64 precision.
    # This is a fixed-order matrix-exponential algorithm, not time stepping.
    term = xp.broadcast_to(identity, generator.shape).copy()
    weight = xp.exp(-mu)
    propagator = weight[..., None, None] * term
    for order in range(1, 33):
        term = xp.matmul(term, stochastic)
        weight = weight * mu / order
        propagator = propagator + weight[..., None, None] * term
    for _ in range(squarings):
        propagator = xp.matmul(propagator, propagator)

    result = xp.einsum("...ij,...j->...i", propagator, state)
    # Preserve exact probability invariants against final floating-point drift.
    result = xp.maximum(result, 0.0)
    return result / xp.sum(result, axis=-1, keepdims=True)


def _labro_equilibrium_from_rates(alpha: Any, beta: Any, xp: Any):
    log_ratio = xp.log(alpha) - xp.log(beta)
    log_weight = xp.concatenate(
        (
            xp.zeros(log_ratio.shape[:-1] + (1,), dtype=xp.float64),
            xp.cumsum(log_ratio, axis=-1),
        ),
        axis=-1,
    )
    log_weight -= xp.max(log_weight, axis=-1, keepdims=True)
    weight = xp.exp(log_weight)
    return weight / xp.sum(weight, axis=-1, keepdims=True)


def advance(
    model_id: str,
    state: Any,
    voltage_mv: Any,
    duration_ms: Any,
    temperature_c: float | None,
    xp: Any,
):
    """Advance broadcast fixed-voltage intervals without Python time stepping."""

    _validate_xp(xp)
    _validate_temperature(model_id, temperature_c)
    state_array = _validate_state(model_id, state, xp)
    voltage = _finite_array(voltage_mv, "voltage_mv", xp)
    duration = _finite_array(duration_ms, "duration_ms", xp)
    if bool(xp.any(duration < 0.0)):
        raise ValueError("duration_ms must be nonnegative")
    try:
        leading_shape = xp.broadcast_shapes(
            state_array.shape[:-1], voltage.shape, duration.shape
        )
    except ValueError as exc:
        raise ValueError(
            "state, voltage_mv, and duration_ms leading dimensions must broadcast"
        ) from exc
    state_array = xp.broadcast_to(
        state_array, leading_shape + (state_array.shape[-1],)
    )
    voltage = xp.broadcast_to(voltage, leading_shape)
    duration = xp.broadcast_to(duration, leading_shape)
    if bool(xp.all(duration == 0.0)):
        return state_array.copy()

    alpha, beta = rates(model_id, voltage, temperature_c, xp)
    if model_id == LABRO_2015:
        result = _labro_advance(state_array, alpha, beta, duration, xp)
        return xp.where((duration == 0.0)[..., None], state_array, result)

    steady_state = alpha / (alpha + beta)
    decay = xp.exp(-(alpha + beta) * duration[..., None])
    result = steady_state + (state_array - steady_state) * decay
    return xp.where((duration == 0.0)[..., None], state_array, result)


def open_probability(model_id: str, state: Any, xp: Any):
    """Return the source-defined conducting fraction for a valid state array."""

    _validate_xp(xp)
    _metadata(model_id)
    state_array = _validate_state(model_id, state, xp)
    if model_id == LABRO_2015:
        return state_array[..., 3]
    n_gate = state_array[..., 0]
    p_gate = state_array[..., 1]
    return n_gate**3 * (0.23 + 0.77 * p_gate)


__all__ = [
    "LABRO_2015",
    "DESAI_2008_CONTROL",
    "MODEL_METADATA",
    "rates",
    "equilibrium",
    "advance",
    "open_probability",
]
