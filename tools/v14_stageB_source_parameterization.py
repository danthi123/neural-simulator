"""Source-centered, constraint-preserving Stage B fitted-arm coordinates.

These coordinates define no biological search bounds. They only map finite
optimizer contrasts to valid constants while preserving each source model's
fixed graph, algebra, signs, and source-only sentinels.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from sim import kv3_source_models as kv3
from sim import sodium_source_models as sodium


SCHEMA = "v14-snr-stageB-source-centered-parameterization-v1"
LOG_POSITIVE = "source_centered_log_positive"
SIGNED_LOG = "source_centered_signed_log_magnitude"
ADDITIVE = "source_centered_additive"


class SourceParameterizationError(ValueError):
    """Raised when fitted-arm coordinates violate their source constraints."""


@dataclass(frozen=True)
class Coordinate:
    id: str
    parameter: str
    component: int | None
    transform: str
    source_value: float


_SODIUM_MODELS = frozenset(sodium.SOURCE_PARAMETER_DEFAULTS)
_KV3_MODELS = frozenset(kv3.SOURCE_PARAMETER_DEFAULTS)
_MODELS = _SODIUM_MODELS | _KV3_MODELS

_KHALIQ_POSITIVE = frozenset(
    {
        "alpha_per_ms", "beta_per_ms", "gamma_per_ms", "delta_per_ms",
        "epsilon_per_ms", "zeta_per_ms", "con_per_ms", "coff_per_ms",
        "oon_per_ms", "ooff_per_ms",
    }
)
_KHALIQ_SIGNED = frozenset({"x1_mv", "x2_mv", "x6_mv"})
_KHALIQ_FIXED = frozenset({"x3_mv", "x4_mv", "x5_mv"})
_BALBI_FIXED = frozenset({"q10", "q10_reference_temperature_c"})


def _defaults(model_id: str) -> Mapping[str, Any]:
    if model_id in _SODIUM_MODELS:
        return sodium.SOURCE_PARAMETER_DEFAULTS[model_id]
    if model_id in _KV3_MODELS:
        return kv3.SOURCE_PARAMETER_DEFAULTS[model_id]
    raise SourceParameterizationError(f"unknown source model: {model_id!r}")


def _coordinate(
    parameter: str, component: int | None, transform: str, source_value: Any
) -> Coordinate:
    value = float(source_value)
    if not math.isfinite(value):
        raise SourceParameterizationError("source coordinate is not finite")
    suffix = "" if component is None else f"[{component}]"
    return Coordinate(
        id=f"{parameter}{suffix}", parameter=parameter, component=component,
        transform=transform, source_value=value,
    )


def coordinates(model_id: str) -> tuple[Coordinate, ...]:
    """Return the exact fitted coordinates for one immutable source graph."""

    defaults = _defaults(model_id)
    result: list[Coordinate] = []
    if model_id == sodium.KHALIQ_RAMAN_13_STATE:
        if set(defaults) != _KHALIQ_POSITIVE | _KHALIQ_SIGNED | _KHALIQ_FIXED:
            raise SourceParameterizationError("Khaliq source parameter set changed")
        for name, value in defaults.items():
            if name in _KHALIQ_FIXED:
                continue
            transform = LOG_POSITIVE if name in _KHALIQ_POSITIVE else SIGNED_LOG
            result.append(_coordinate(name, None, transform, value))
    elif model_id == sodium.BALBI_NAV16_SIX_STATE:
        if not _BALBI_FIXED < set(defaults):
            raise SourceParameterizationError("Balbi fixed temperature law changed")
        for name, value in defaults.items():
            if name in _BALBI_FIXED:
                continue
            if not isinstance(value, tuple) or len(value) != 3:
                raise SourceParameterizationError("Balbi kinetic component shape changed")
            result.extend(
                (
                    _coordinate(name, 0, LOG_POSITIVE, value[0]),
                    _coordinate(name, 1, ADDITIVE, value[1]),
                    _coordinate(name, 2, SIGNED_LOG, value[2]),
                )
            )
    elif model_id == kv3.LABRO_2015:
        for name, value in defaults.items():
            if name == "vhalf_mv":
                result.append(_coordinate(name, None, ADDITIVE, value))
                continue
            if not isinstance(value, tuple) or len(value) != 3:
                raise SourceParameterizationError("Labro kinetic component shape changed")
            result.extend(
                _coordinate(name, index, LOG_POSITIVE, item)
                for index, item in enumerate(value)
            )
    else:
        for name, value in defaults.items():
            if not isinstance(value, tuple) or len(value) != 2:
                raise SourceParameterizationError("Desai kinetic component shape changed")
            transform = LOG_POSITIVE if name.startswith("k_") else SIGNED_LOG
            result.extend(
                _coordinate(name, index, transform, item)
                for index, item in enumerate(value)
            )
    identifiers = [item.id for item in result]
    if len(identifiers) != len(set(identifiers)):
        raise SourceParameterizationError("source coordinate ids are not unique")
    return tuple(result)


def parameterization_document(model_id: str) -> dict[str, Any]:
    """Describe fitted coordinates without implying numeric biological bounds."""

    rows = [
        {
            "id": item.id,
            "parameter": item.parameter,
            "component": item.component,
            "transform": item.transform,
            "source_value": item.source_value,
            "lower_bound": None,
            "upper_bound": None,
            "bound_authority": "unresolved_not_invented",
        }
        for item in coordinates(model_id)
    ]
    fixed = sorted(set(_defaults(model_id)) - {item.parameter for item in coordinates(model_id)})
    return {
        "schema": SCHEMA,
        "model_id": model_id,
        "coordinates": rows,
        "fixed_parameters": fixed,
        "graph_changes_allowed": False,
        "numeric_biological_bounds_available": False,
    }


def _validated_parameters(model_id: str, parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    validator = (
        sodium.validate_source_parameters if model_id in _SODIUM_MODELS
        else kv3.validate_source_parameters
    )
    try:
        return validator(model_id, parameters)
    except (TypeError, ValueError) as exc:
        raise SourceParameterizationError(str(exc)) from exc


def _value(parameters: Mapping[str, Any], coordinate: Coordinate) -> float:
    value = parameters[coordinate.parameter]
    if coordinate.component is not None:
        value = value[coordinate.component]
    return float(value)


def encode(model_id: str, parameters: Mapping[str, Any]) -> np.ndarray:
    """Encode a valid source-graph parameter document as source contrasts."""

    checked = _validated_parameters(model_id, parameters)
    result: list[float] = []
    for item in coordinates(model_id):
        value = _value(checked, item)
        if item.transform == ADDITIVE:
            contrast = value - item.source_value
        elif item.transform == LOG_POSITIVE:
            if value <= 0.0:
                raise SourceParameterizationError(f"{item.id} must remain positive")
            contrast = math.log(value / item.source_value)
        else:
            if value == 0.0 or math.copysign(1.0, value) != math.copysign(1.0, item.source_value):
                raise SourceParameterizationError(f"{item.id} changed source sign")
            contrast = math.log(abs(value) / abs(item.source_value))
        result.append(contrast)
    return np.asarray(result, dtype=np.float64)


def decode(model_id: str, contrasts: Any) -> Mapping[str, Any]:
    """Decode finite unbounded contrasts into a valid source-graph document."""

    try:
        vector = np.asarray(contrasts, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SourceParameterizationError("contrasts must be finite real values") from exc
    declared = coordinates(model_id)
    if vector.shape != (len(declared),) or not np.all(np.isfinite(vector)):
        raise SourceParameterizationError(
            f"contrasts must have finite shape ({len(declared)},)"
        )
    result: dict[str, Any] = {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in _defaults(model_id).items()
    }
    for contrast, item in zip(vector, declared, strict=True):
        try:
            if item.transform == ADDITIVE:
                value = item.source_value + float(contrast)
            elif item.transform == LOG_POSITIVE:
                value = item.source_value * math.exp(float(contrast))
            else:
                value = math.copysign(
                    abs(item.source_value) * math.exp(float(contrast)), item.source_value
                )
        except OverflowError as exc:
            raise SourceParameterizationError(
                f"contrast for {item.id} exceeds finite source coordinates"
            ) from exc
        if item.component is None:
            result[item.parameter] = value
        else:
            result[item.parameter][item.component] = value
    frozen = {
        name: tuple(value) if isinstance(value, list) else value
        for name, value in result.items()
    }
    return _validated_parameters(model_id, frozen)


def decode_batch(model_id: str, contrasts: Any) -> list[Mapping[str, Any]]:
    """Decode a host-side candidate matrix for the vectorized kinetic engine."""

    try:
        matrix = np.asarray(contrasts, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SourceParameterizationError("contrast matrix must contain real values") from exc
    width = len(coordinates(model_id))
    if matrix.ndim != 2 or matrix.shape[1] != width or matrix.shape[0] == 0:
        raise SourceParameterizationError(
            f"contrast matrix must have nonempty shape (candidate, {width})"
        )
    if not np.all(np.isfinite(matrix)):
        raise SourceParameterizationError("contrast matrix must be finite")
    return [decode(model_id, row) for row in matrix]
