"""Bounded NumPy identifiability diagnostics for sealed V14 Stage B fits.

This module is deliberately analysis-only.  It may describe what the supplied
calibration evidence supports, but cannot promote a fit, read held-out data, or
provide a scientific verdict.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


SCHEMA = "v14-snr-stageB-identifiability-diagnostic-v1"
PARAMETER_SPACE_SCHEMA = "v14-snr-stageB-identifiability-parameter-space-v1"
FIT_SCHEMA = "v14-snr-stageB-completed-fit-v1"
PERTURBATION_SCHEMA = "v14-snr-stageB-identifiability-perturbations-v1"
CANONICALIZATION = "json-sort-keys-compact-ascii-v1"
_SHA256 = frozenset("0123456789abcdef")


class StageBIdentifiabilityError(ValueError):
    """Raised when supplied evidence is not sealed, finite, or calibration-only."""


def canonical_bytes(value: Any) -> bytes:
    """Return the repository's deterministic JSON representation."""
    try:
        return json.dumps(
            value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise StageBIdentifiabilityError("value is not canonical JSON data") from exc


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _fail(message: str) -> None:
    raise StageBIdentifiabilityError(message)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or "\x00" in value:
        _fail(f"{field} must be non-empty trimmed text")
    return value


def _finite(value: Any, field: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{field} must be a finite number")
    if positive and result <= 0.0:
        _fail(f"{field} must be positive")
    if nonnegative and result < 0.0:
        _fail(f"{field} must be non-negative")
    return result


def _integer(value: Any, field: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{field} must be an integer at least {minimum}")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - _SHA256:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _self_digest(document: Mapping[str, Any], field: str) -> str:
    value = _sha256(document.get("sha256"), f"{field} sha256")
    if value != digest({key: item for key, item in document.items() if key != "sha256"}):
        _fail(f"{field} self digest is invalid")
    return value


def _binding(value: Any, field: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"id", "sha256"}:
        _fail(f"{field} must contain exactly id and sha256")
    return {"id": _text(value["id"], f"{field} id"), "sha256": _sha256(value["sha256"], f"{field} sha256")}


def _contains_held_out(value: Any) -> bool:
    if isinstance(value, str):
        return "held" in value.lower() and "out" in value.lower()
    if isinstance(value, Mapping):
        return any(_contains_held_out(key) or _contains_held_out(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_held_out(item) for item in value)
    return False


def _parameter_space(document: Any) -> tuple[dict[str, Any], tuple[str, ...], np.ndarray, dict[str, tuple[float, float]], dict[str, Any]]:
    if not isinstance(document, Mapping):
        _fail("sealed parameter space must be an object")
    required = {
        "schema", "id", "status", "parameters", "allowed_fit_partitions", "thresholds",
        "scientific_verdict", "optimization_allowed", "sha256",
    }
    if set(document) != required or document.get("schema") != PARAMETER_SPACE_SCHEMA:
        _fail("sealed parameter space has an invalid shape")
    if document.get("status") != "sealed":
        _fail("sealed parameter space status must be sealed")
    if document.get("scientific_verdict") is not None or document.get("optimization_allowed") is not False:
        _fail("sealed parameter space must remain analysis-only")
    _self_digest(document, "sealed parameter space")
    identifier = _text(document.get("id"), "parameter space id")
    allowed = document.get("allowed_fit_partitions")
    if (not isinstance(allowed, list) or not allowed or any(not isinstance(item, str) for item in allowed)
            or len(set(allowed)) != len(allowed) or _contains_held_out(allowed)):
        _fail("allowed_fit_partitions must be a non-empty unique calibration-only list")
    rows = document.get("parameters")
    if not isinstance(rows, list) or not rows:
        _fail("parameter space parameters must be a non-empty list")
    parameter_ids: list[str] = []
    scales: list[float] = []
    bounds: dict[str, tuple[float, float]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {"id", "low", "high", "scale"}:
            _fail(f"parameter {index} has an invalid shape")
        name = _text(row["id"], f"parameter {index} id")
        low = _finite(row["low"], f"parameter {name} low")
        high = _finite(row["high"], f"parameter {name} high")
        scale = _finite(row["scale"], f"parameter {name} scale", positive=True)
        if low >= high:
            _fail(f"parameter {name} must have low < high")
        parameter_ids.append(name)
        scales.append(scale)
        bounds[name] = (low, high)
    if len(set(parameter_ids)) != len(parameter_ids) or parameter_ids != sorted(parameter_ids):
        _fail("parameter ids must be unique and sorted")
    thresholds = document.get("thresholds")
    expected_thresholds = {
        "min_completed_fits", "min_near_equivalent_fits", "min_perturbation_pairs",
        "near_equivalent_objective_l2", "svd_rank_relative_tolerance", "max_condition_number_identified",
        "max_relative_span_identified", "max_abs_correlation_identified",
        "max_scaled_perturbation_l2",
    }
    if not isinstance(thresholds, Mapping) or set(thresholds) != expected_thresholds:
        _fail("parameter space thresholds have an invalid shape")
    checked = {
        "min_completed_fits": _integer(thresholds["min_completed_fits"], "min_completed_fits", minimum=2),
        "min_near_equivalent_fits": _integer(thresholds["min_near_equivalent_fits"], "min_near_equivalent_fits", minimum=2),
        "min_perturbation_pairs": _integer(thresholds["min_perturbation_pairs"], "min_perturbation_pairs", minimum=1),
        "near_equivalent_objective_l2": _finite(thresholds["near_equivalent_objective_l2"], "near_equivalent_objective_l2", nonnegative=True),
        "svd_rank_relative_tolerance": _finite(thresholds["svd_rank_relative_tolerance"], "svd_rank_relative_tolerance", positive=True),
        "max_condition_number_identified": _finite(thresholds["max_condition_number_identified"], "max_condition_number_identified", positive=True),
        "max_relative_span_identified": _finite(thresholds["max_relative_span_identified"], "max_relative_span_identified", nonnegative=True),
        "max_abs_correlation_identified": _finite(thresholds["max_abs_correlation_identified"], "max_abs_correlation_identified", nonnegative=True),
        "max_scaled_perturbation_l2": _finite(
            thresholds["max_scaled_perturbation_l2"],
            "max_scaled_perturbation_l2",
            positive=True,
        ),
    }
    if checked["max_abs_correlation_identified"] > 1.0:
        _fail("max_abs_correlation_identified cannot exceed one")
    return dict(document), tuple(parameter_ids), np.asarray(scales, dtype=np.float64), bounds, checked


def _fit_records(
    records: Sequence[Any], parameter_space: Mapping[str, Any], parameter_ids: tuple[str, ...],
    bounds: Mapping[str, tuple[float, float]],
) -> tuple[dict[str, dict[str, Any]], tuple[str, ...]]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        _fail("completed fit records must be a sequence")
    loaded: dict[str, dict[str, Any]] = {}
    residual_names: tuple[str, ...] | None = None
    space_binding = {"id": parameter_space["id"], "sha256": parameter_space["sha256"]}
    allowed = set(parameter_space["allowed_fit_partitions"])
    if allowed != {"calibration"}:
        _fail("allowed_fit_partitions must contain calibration only")
    for index, document in enumerate(records):
        if not isinstance(document, Mapping):
            _fail(f"fit record {index} must be an object")
        required = {
            "schema", "id", "status", "parameter_space", "partition", "parameters",
            "objective_residuals", "scientific_verdict", "optimization_allowed", "sha256",
        }
        if set(document) != required or document.get("schema") != FIT_SCHEMA:
            _fail(f"fit record {index} has an invalid shape")
        if document.get("status") != "completed":
            _fail(f"fit record {index} is not completed")
        if document.get("scientific_verdict") is not None or document.get("optimization_allowed") is not False:
            _fail(f"fit record {index} is not analysis-only")
        _self_digest(document, f"fit record {index}")
        identifier = _text(document.get("id"), f"fit record {index} id")
        if identifier in loaded:
            _fail("fit record ids must be unique")
        if _binding(document.get("parameter_space"), f"fit record {identifier} parameter_space") != space_binding:
            _fail(f"fit record {identifier} is not bound to the sealed parameter space")
        partition = _text(document.get("partition"), f"fit record {identifier} partition")
        if partition not in allowed or _contains_held_out(document):
            _fail(f"fit record {identifier} contains held-out or unauthorized data")
        parameters = document.get("parameters")
        if not isinstance(parameters, Mapping) or tuple(sorted(parameters)) != parameter_ids:
            _fail(f"fit record {identifier} parameters do not equal the sealed parameter space")
        checked_parameters = {name: _finite(parameters[name], f"fit record {identifier} parameter {name}") for name in parameter_ids}
        for name, value in checked_parameters.items():
            low, high = bounds[name]
            if value < low or value > high:
                _fail(f"fit record {identifier} parameter {name} is outside sealed bounds")
        residuals = document.get("objective_residuals")
        if not isinstance(residuals, Mapping) or not residuals:
            _fail(f"fit record {identifier} objective_residuals must be non-empty")
        names = tuple(sorted(residuals))
        if any(not isinstance(name, str) or not name for name in names):
            _fail(f"fit record {identifier} residual names are invalid")
        if residual_names is None:
            residual_names = names
        elif names != residual_names:
            _fail("fit records must have exactly the same objective residual names")
        checked_residuals = {name: _finite(residuals[name], f"fit record {identifier} residual {name}") for name in names}
        loaded[identifier] = {
            "id": identifier, "sha256": document["sha256"], "parameters": checked_parameters,
            "objective_residuals": checked_residuals,
        }
    if not loaded:
        _fail("completed fit records must be non-empty")
    assert residual_names is not None
    return loaded, residual_names


def _perturbations(
    document: Any, parameter_space: Mapping[str, Any], fits: Mapping[str, Any],
) -> tuple[list[tuple[str, str]], str]:
    if not isinstance(document, Mapping):
        _fail("perturbation pairs must be an object")
    required = {"schema", "parameter_space", "pairs", "scientific_verdict", "optimization_allowed", "sha256"}
    if set(document) != required or document.get("schema") != PERTURBATION_SCHEMA:
        _fail("perturbation pairs has an invalid shape")
    if document.get("scientific_verdict") is not None or document.get("optimization_allowed") is not False:
        _fail("perturbation pairs must remain analysis-only")
    value = _self_digest(document, "perturbation pairs")
    expected = {"id": parameter_space["id"], "sha256": parameter_space["sha256"]}
    if _binding(document.get("parameter_space"), "perturbation pairs parameter_space") != expected:
        _fail("perturbation pairs is not bound to the sealed parameter space")
    pairs = document.get("pairs")
    if not isinstance(pairs, list):
        _fail("perturbation pairs must be a list")
    loaded: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, pair in enumerate(pairs):
        if not isinstance(pair, Mapping) or set(pair) != {"baseline_fit_id", "perturbed_fit_id"}:
            _fail(f"perturbation pair {index} has an invalid shape")
        baseline = _text(pair["baseline_fit_id"], f"perturbation pair {index} baseline_fit_id")
        perturbed = _text(pair["perturbed_fit_id"], f"perturbation pair {index} perturbed_fit_id")
        if baseline == perturbed or baseline not in fits or perturbed not in fits:
            _fail(f"perturbation pair {index} does not reference distinct completed fits")
        item = (baseline, perturbed)
        if item in seen:
            _fail("perturbation pairs must be unique")
        seen.add(item)
        loaded.append(item)
    return sorted(loaded), value


def _json_number(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        _fail("internal diagnostic produced a non-finite value")
    return result


def _ensemble(
    fits: Mapping[str, Mapping[str, Any]], parameter_ids: tuple[str, ...], scales: np.ndarray,
    threshold: float,
) -> tuple[list[str], dict[str, Any], np.ndarray]:
    ordered = sorted(fits)
    residuals = np.asarray(
        [[fits[item]["objective_residuals"][name] for name in sorted(fits[item]["objective_residuals"])] for item in ordered],
        dtype=np.float64,
    )
    scores = np.linalg.norm(residuals, axis=1)
    minimum = float(np.min(scores))
    equivalent = [item for item, score in zip(ordered, scores, strict=True) if score <= minimum + threshold]
    values = np.asarray([[fits[item]["parameters"][name] for name in parameter_ids] for item in equivalent], dtype=np.float64)
    spans = np.max(values, axis=0) - np.min(values, axis=0)
    relative_spans = spans / scales
    if len(equivalent) >= 2 and len(parameter_ids) >= 2:
        correlations = np.corrcoef(values, rowvar=False)
        correlations = np.nan_to_num(correlations, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(correlations, 1.0)
    else:
        correlations = np.eye(len(parameter_ids), dtype=np.float64)
    output = {
        "count": len(equivalent),
        "objective_l2_minimum": _json_number(minimum),
        "objective_l2_tolerance": _json_number(threshold),
        "parameter_spans": [
            {"id": name, "absolute": _json_number(spans[index]), "relative_to_scale": _json_number(relative_spans[index])}
            for index, name in enumerate(parameter_ids)
        ],
        "parameter_correlation": [
            [_json_number(item) for item in row] for row in correlations.tolist()
        ],
    }
    return equivalent, output, correlations


def _jacobian(
    pairs: Sequence[tuple[str, str]], fits: Mapping[str, Mapping[str, Any]],
    parameter_ids: tuple[str, ...], residual_names: tuple[str, ...], scales: np.ndarray, thresholds: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray | None]:
    if len(pairs) < thresholds["min_perturbation_pairs"]:
        return {
            "status": "insufficient_evidence", "reason": "fewer_than_preregistered_perturbation_pairs",
            "method": "scaled_finite_difference_local_least_squares", "pair_count": len(pairs),
            "input_rank": None, "rank": None, "condition_number": None, "singular_values": [], "matrix": None,
        }, None
    baselines = {baseline_id for baseline_id, _ in pairs}
    if len(baselines) != 1:
        return {
            "status": "insufficient_evidence", "reason": "perturbation_pairs_do_not_share_one_local_baseline",
            "method": "scaled_finite_difference_local_least_squares", "pair_count": len(pairs),
            "input_rank": None, "rank": None, "condition_number": None, "singular_values": [], "matrix": None,
        }, None
    directions: list[list[float]] = []
    changes: list[list[float]] = []
    for baseline_id, perturbed_id in pairs:
        baseline = fits[baseline_id]
        perturbed = fits[perturbed_id]
        direction = np.asarray(
            [(perturbed["parameters"][name] - baseline["parameters"][name]) / scales[index]
             for index, name in enumerate(parameter_ids)], dtype=np.float64,
        )
        if not np.any(direction):
            _fail(f"perturbation pair {baseline_id!r}->{perturbed_id!r} has zero parameter displacement")
        if float(np.linalg.norm(direction)) > thresholds["max_scaled_perturbation_l2"]:
            return {
                "status": "insufficient_evidence", "reason": "perturbation_exceeds_preregistered_local_radius",
                "method": "scaled_finite_difference_local_least_squares", "pair_count": len(pairs),
                "input_rank": None, "rank": None, "condition_number": None, "singular_values": [], "matrix": None,
            }, None
        response = np.asarray(
            [perturbed["objective_residuals"][name] - baseline["objective_residuals"][name] for name in residual_names],
            dtype=np.float64,
        )
        directions.append(direction.tolist())
        changes.append(response.tolist())
    design = np.asarray(directions, dtype=np.float64)
    responses = np.asarray(changes, dtype=np.float64)
    input_singular = np.linalg.svd(design, compute_uv=False)
    input_largest = float(input_singular[0]) if input_singular.size else 0.0
    input_rank = int(np.sum(input_singular > input_largest * thresholds["svd_rank_relative_tolerance"])) if input_largest else 0
    if input_rank < len(parameter_ids):
        return {
            "status": "insufficient_evidence", "reason": "perturbation_directions_do_not_span_parameter_space",
            "method": "scaled_finite_difference_local_least_squares", "pair_count": len(pairs),
            "input_rank": input_rank, "rank": None, "condition_number": None, "singular_values": [], "matrix": None,
        }, None
    matrix = np.linalg.lstsq(design, responses, rcond=thresholds["svd_rank_relative_tolerance"])[0].T
    singular = np.linalg.svd(matrix, compute_uv=False)
    largest = float(singular[0]) if singular.size else 0.0
    rank = int(np.sum(singular > largest * thresholds["svd_rank_relative_tolerance"])) if largest else 0
    condition = float(largest / singular[-1]) if rank == min(matrix.shape) and singular[-1] > 0.0 else None
    return {
        "status": "available", "reason": None, "method": "scaled_finite_difference_local_least_squares",
        "pair_count": len(pairs), "input_rank": input_rank, "rank": rank,
        "condition_number": _json_number(condition) if condition is not None else None,
        "singular_values": [_json_number(item) for item in singular.tolist()],
        "matrix": [[_json_number(item) for item in row] for row in matrix.tolist()],
    }, matrix


def diagnose_identifiability(
    sealed_parameter_space: Mapping[str, Any],
    completed_fit_records: Sequence[Mapping[str, Any]],
    perturbation_pairs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a deterministic, calibration-only identifiability diagnostic.

    A result marked ``insufficient_evidence`` is intentionally not upgraded by
    ensemble variation alone: local sensitivity requires declared perturbation
    pairs, and the diagnostic requires more than one completed fit.
    """
    parameter_space, parameter_ids, scales, bounds, thresholds = _parameter_space(sealed_parameter_space)
    fits, residual_names = _fit_records(completed_fit_records, parameter_space, parameter_ids, bounds)
    pairs, perturbation_sha = _perturbations(perturbation_pairs, parameter_space, fits)
    equivalent, ensemble, correlations = _ensemble(
        fits, parameter_ids, scales, thresholds["near_equivalent_objective_l2"],
    )
    jacobian, matrix = _jacobian(pairs, fits, parameter_ids, residual_names, scales, thresholds)
    enough_fits = len(fits) >= thresholds["min_completed_fits"]
    enough_ensemble = len(equivalent) >= thresholds["min_near_equivalent_fits"]
    sufficient = enough_fits and enough_ensemble and matrix is not None
    if not sufficient:
        status = "insufficient_evidence"
    else:
        status = "sufficient_evidence"
    rank_full = matrix is not None and jacobian["rank"] == len(parameter_ids)
    condition_ok = jacobian["condition_number"] is not None and jacobian["condition_number"] <= thresholds["max_condition_number_identified"]
    spans = ensemble["parameter_spans"]
    parameter_rows: list[dict[str, Any]] = []
    for index, identifier in enumerate(parameter_ids):
        correlation = max((abs(correlations[index, other]) for other in range(len(parameter_ids)) if other != index), default=0.0)
        span_ok = spans[index]["relative_to_scale"] <= thresholds["max_relative_span_identified"]
        correlation_ok = correlation <= thresholds["max_abs_correlation_identified"]
        if not sufficient:
            classification = "unresolved"
            reason = "insufficient_evidence"
        elif rank_full and condition_ok and span_ok and correlation_ok:
            classification = "identified"
            reason = "all_preregistered_criteria_met"
        elif rank_full:
            classification = "weak"
            reason = "ensemble_span_correlation_or_condition_threshold_not_met"
        else:
            classification = "unresolved"
            reason = "rank_deficient_local_jacobian"
        parameter_rows.append({
            "id": identifier, "classification": classification, "reason": reason,
            "relative_span": spans[index]["relative_to_scale"], "max_abs_correlation": _json_number(correlation),
        })
    body = {
        "schema": SCHEMA,
        "canonicalization": CANONICALIZATION,
        "parameter_space": {"id": parameter_space["id"], "sha256": parameter_space["sha256"]},
        "fit_records": [{"id": fits[item]["id"], "sha256": fits[item]["sha256"]} for item in sorted(fits)],
        "perturbation_pairs": {"sha256": perturbation_sha},
        "objective_residual_names": list(residual_names),
        "thresholds": thresholds,
        "diagnostic_status": status,
        "near_equivalent_fit_ids": equivalent,
        "near_equivalent_ensemble": ensemble,
        "jacobian": jacobian,
        "parameters": parameter_rows,
        "scientific_verdict": None,
        "optimization_allowed": False,
    }
    return {**body, "sha256": digest(body)}
