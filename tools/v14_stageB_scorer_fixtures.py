"""Pure validation and scoring for bounded V14 Stage B physiology fixtures."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


class StageBFixtureError(ValueError):
    """Raised when a fixture or observation is not safe to score."""


_PREPARATION_FIELDS = (
    "species_age",
    "slice",
    "temperature",
    "recording_modes",
    "solution",
    "blockers",
)
_UNRESOLVED_MARKERS = (
    "unresolved",
    "remains to be transcribed",
    "see source",
    "must be verified",
    "pending exact",
)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StageBFixtureError(f"{field} must be non-empty text")
    return value.strip()


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise StageBFixtureError(f"{field} must be a finite number")
    return float(value)


def _validate_preparation(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StageBFixtureError("evidence.preparation must be an object")
    preparation = dict(value)
    for field in _PREPARATION_FIELDS:
        if field not in preparation:
            raise StageBFixtureError(f"evidence.preparation.{field} is required")
        item = preparation[field]
        if field == "recording_modes":
            if not isinstance(item, list) or not item:
                raise StageBFixtureError("evidence.preparation.recording_modes must be a non-empty list")
            texts = [_text(entry, "evidence.preparation.recording_modes[]") for entry in item]
        elif field == "blockers" and isinstance(item, list):
            if not item:
                raise StageBFixtureError("evidence.preparation.blockers must not be empty")
            texts = [_text(entry, "evidence.preparation.blockers[]") for entry in item]
        else:
            texts = [_text(item, f"evidence.preparation.{field}")]
        if any(marker in text.lower() for text in texts for marker in _UNRESOLVED_MARKERS):
            raise StageBFixtureError(f"evidence.preparation.{field} is unresolved")
    return preparation


def validate_fixture(fixture: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one immutable scoring contract without reading files or global state."""
    if not isinstance(fixture, Mapping):
        raise StageBFixtureError("fixture must be an object")
    normalized = dict(fixture)
    for field in ("id", "target_id", "source_id", "cohort", "pathway", "metric", "units"):
        normalized[field] = _text(normalized.get(field), field)

    evidence = normalized.get("evidence")
    if not isinstance(evidence, Mapping):
        raise StageBFixtureError("evidence must be an object")
    evidence = dict(evidence)
    provenance = _text(evidence.get("interval_provenance"), "evidence.interval_provenance")
    if provenance not in {"source-derived", "model-derived", "not-an-interval"}:
        raise StageBFixtureError("evidence.interval_provenance is invalid")
    _text(evidence.get("uncertainty"), "evidence.uncertainty")
    locator = _text(evidence.get("source_locator"), "evidence.source_locator")
    if any(marker in locator.lower() for marker in _UNRESOLVED_MARKERS):
        raise StageBFixtureError("evidence.source_locator is unresolved")
    evidence["preparation"] = _validate_preparation(evidence.get("preparation"))
    normalized["evidence"] = evidence

    kind = _text(normalized.get("score_kind"), "score_kind")
    if kind == "bounded-interval":
        if provenance == "not-an-interval":
            raise StageBFixtureError("bounded interval requires source-derived or model-derived provenance")
        interval = normalized.get("interval")
        if not isinstance(interval, Mapping):
            raise StageBFixtureError("interval must be an object")
        low = _number(interval.get("low"), "interval.low")
        high = _number(interval.get("high"), "interval.high")
        if low > high:
            raise StageBFixtureError("interval.low must not exceed interval.high")
        normalized["interval"] = {"low": low, "high": high}
        if provenance == "model-derived":
            _text(evidence.get("derivation_method"), "evidence.derivation_method")
            measurements = evidence.get("source_measurements")
            if not isinstance(measurements, Mapping) or not measurements:
                raise StageBFixtureError("model-derived interval requires source_measurements")
        elif "derivation_method" in evidence or "source_measurements" in evidence:
            raise StageBFixtureError("source-derived interval cannot carry model-derivation fields")
    elif kind == "non-significance-boundary":
        if provenance != "not-an-interval":
            raise StageBFixtureError("non-significance cannot define an interval")
        if "interval" in normalized:
            raise StageBFixtureError("non-significance cannot carry equivalence bounds")
        if normalized.get("interpretation") != "no-equivalence-claim":
            raise StageBFixtureError("non-significance must preserve the no-equivalence-claim boundary")
    else:
        raise StageBFixtureError(f"unsupported score_kind: {kind}")
    return normalized


def score_observation(fixture: Mapping[str, Any], observation: Mapping[str, Any]) -> dict[str, Any]:
    """Score one observation while enforcing its exact cohort and pathway contract."""
    contract = validate_fixture(fixture)
    if not isinstance(observation, Mapping):
        raise StageBFixtureError("observation must be an object")
    for field in ("cohort", "pathway", "metric", "units"):
        actual = _text(observation.get(field), f"observation.{field}")
        if actual != contract[field]:
            raise StageBFixtureError(f"observation.{field} does not match fixture")

    provenance = contract["evidence"]["interval_provenance"]
    if contract["score_kind"] == "non-significance-boundary":
        return {
            "fixture_id": contract["id"],
            "status": "not-scorable-as-equivalence",
            "passed": None,
            "interval_provenance": provenance,
        }

    value = _number(observation.get("value"), "observation.value")
    interval = contract["interval"]
    return {
        "fixture_id": contract["id"],
        "status": "scored",
        "passed": interval["low"] <= value <= interval["high"],
        "value": value,
        "interval": interval,
        "interval_provenance": provenance,
    }
