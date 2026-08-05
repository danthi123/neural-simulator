"""Score one sealed candidate observation against calibration-only population targets.

This is a bounded analysis sidecar.  It cannot read validation or held-out
packets, emit an optimisation objective, or make a scientific verdict.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

from tools import v14_stageB_population_targets as population_targets


SCHEMA = "v14-snr-stageB-population-calibration-score-v1"
OBSERVATION_SCHEMA = "v14-snr-stageB-population-candidate-observation-v1"
CANONICALIZATION = "json-sort-keys-compact-ascii-v1"
_SHA256 = frozenset("0123456789abcdef")


class PopulationScorerError(ValueError):
    """Raised when custody-bound target or candidate evidence is invalid."""


def canonical_bytes(value: Any) -> bytes:
    """Return the deterministic JSON representation used by sealed evidence."""
    try:
        return json.dumps(
            value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise PopulationScorerError("value is not canonical JSON data") from exc


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _fail(message: str) -> None:
    raise PopulationScorerError(message)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or "\x00" in value:
        _fail(f"{field} must be non-empty trimmed text")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - _SHA256:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _finite(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{field} must be a finite number")
    if positive and result <= 0.0:
        _fail(f"{field} must be positive")
    return result


def _self_digest(document: Mapping[str, Any], field: str) -> str:
    value = _sha256(document.get("sha256"), f"{field} sha256")
    if value != digest({key: item for key, item in document.items() if key != "sha256"}):
        _fail(f"{field} self digest is invalid")
    return value


def _binding(value: Any, field: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"id", "sha256"}:
        _fail(f"{field} must contain exactly id and sha256")
    return {
        "id": _text(value["id"], f"{field} id"),
        "sha256": _sha256(value["sha256"], f"{field} sha256"),
    }


def _measurement(value: Any, field: str, *, positive_uncertainty: bool) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != {
        "median", "standard_uncertainty", "q025", "q975",
    }:
        _fail(f"{field} has an invalid measurement shape")
    result = {key: _finite(value[key], f"{field} {key}") for key in value}
    if positive_uncertainty and result["standard_uncertainty"] <= 0.0:
        _fail(f"{field} standard_uncertainty must be positive")
    if result["q025"] > result["median"] or result["median"] > result["q975"]:
        _fail(f"{field} quantiles are not ordered")
    return result


def _biological_sem(value: Any, y: Mapping[str, float], target_id: str) -> float | None:
    """Return a conservative SEM only when both bounded SEM endpoints exist."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        _fail(f"target {target_id} biological_error is invalid")
    status = value.get("status")
    if status in {"unavailable", "not_reported", "one_sided"}:
        return None
    if status != "available" or set(value) != {
        "status", "kind", "lower_endpoint_digitization", "upper_endpoint_digitization",
    }:
        _fail(f"target {target_id} biological_error is invalid")
    if value["kind"] != "standard_error":
        _fail(f"target {target_id} biological_error is not a standard error")
    lower = _measurement(
        value["lower_endpoint_digitization"],
        f"target {target_id} lower SEM endpoint",
        positive_uncertainty=False,
    )
    upper = _measurement(
        value["upper_endpoint_digitization"],
        f"target {target_id} upper SEM endpoint",
        positive_uncertainty=False,
    )
    mean = y["median"]
    if lower["median"] > mean or mean > upper["median"]:
        _fail(f"target {target_id} SEM endpoints do not bound its mean")
    # Error bars can be slightly asymmetric after digitisation.  The larger
    # bounded distance is conservative and does not invent a missing endpoint.
    sem = max(mean - lower["median"], upper["median"] - mean)
    if sem < 0.0 or not math.isfinite(sem):
        _fail(f"target {target_id} biological SEM is invalid")
    return sem


def _packet_targets(packet: Any) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(packet, Mapping):
        _fail("target packet must be an object")
    required = {
        "schema", "scientific_verdict", "optimization_command", "optimization_allowed", "status",
        "partition", "proposal_visible", "measurement_protocol", "partition_protocol", "evidence",
        "targets", "sha256",
    }
    if set(packet) != required or packet.get("schema") != population_targets.PACKET_SCHEMA:
        _fail("target packet has an invalid shape")
    packet_sha = _self_digest(packet, "target packet")
    if (
        packet.get("partition") != "calibration"
        or packet.get("proposal_visible") is not True
        or packet.get("status") != "sealed_source_measurements"
        or packet.get("scientific_verdict") is not None
        or packet.get("optimization_command") is not None
        or packet.get("optimization_allowed") is not False
    ):
        _fail("target packet must be the proposal-visible calibration partition")
    rows = packet.get("targets")
    if not isinstance(rows, list) or not rows:
        _fail("target packet targets must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            _fail(f"target {index} must be an object")
        required_target = {
            "target_id", "target_family", "asset_id", "panel", "series_identity", "command_id",
            "partition", "x_quantity", "x_unit", "y_quantity", "y_unit", "sample_size",
            "measurement_limitation", "status", "unavailable_reason", "x", "y",
            "digitization_uncertainty", "biological_error",
        }
        if set(row) != required_target:
            _fail(f"target {index} has an invalid shape")
        target_id = _text(row["target_id"], f"target {index} id")
        if target_id in seen:
            _fail("target ids must be unique")
        seen.add(target_id)
        family = _text(row["target_family"], f"target {target_id} family")
        for field in (
            "asset_id", "panel", "series_identity", "command_id", "x_quantity", "x_unit",
            "y_quantity", "y_unit",
        ):
            _text(row[field], f"target {target_id} {field}")
        if row["partition"] != "calibration":
            _fail(f"target {target_id} is not calibration evidence")
        status = row["status"]
        if status == "unavailable":
            if (
                not isinstance(row["unavailable_reason"], str)
                or any(row[field] is not None for field in ("x", "y", "digitization_uncertainty", "biological_error"))
            ):
                _fail(f"unavailable target {target_id} has numeric evidence")
            normalized.append({"target_id": target_id, "target_family": family, "status": status})
            continue
        if status != "available" or row["unavailable_reason"] is not None:
            _fail(f"target {target_id} availability is invalid")
        x = _measurement(row["x"], f"target {target_id} x", positive_uncertainty=False)
        y = _measurement(row["y"], f"target {target_id} y", positive_uncertainty=True)
        if not isinstance(row["digitization_uncertainty"], Mapping):
            _fail(f"target {target_id} digitization uncertainty is invalid")
        sem = _biological_sem(row["biological_error"], y, target_id)
        normalized.append({
            "target_id": target_id,
            "target_family": family,
            "status": status,
            "x": x["median"],
            "y": y["median"],
            "digitization_standard_uncertainty": y["standard_uncertainty"],
            "biological_standard_error": sem,
        })
    if [row["target_id"] for row in normalized] != sorted(row["target_id"] for row in normalized):
        _fail("target packet targets must be sorted by target_id")
    return packet_sha, normalized


def _observation(document: Any, packet_sha: str, available_target_ids: set[str]) -> tuple[dict[str, str], dict[str, dict[str, float]], str]:
    if not isinstance(document, Mapping):
        _fail("candidate observation must be an object")
    required = {
        "schema", "status", "target_packet", "candidate", "predictions", "scientific_verdict",
        "optimization_allowed", "sha256",
    }
    if set(document) != required or document.get("schema") != OBSERVATION_SCHEMA:
        _fail("candidate observation has an invalid shape")
    observation_sha = _self_digest(document, "candidate observation")
    if document.get("status") != "completed" or document.get("scientific_verdict") is not None or document.get("optimization_allowed") is not False:
        _fail("candidate observation must be completed analysis-only evidence")
    packet = document.get("target_packet")
    if not isinstance(packet, Mapping) or set(packet) != {"sha256"} or _sha256(packet["sha256"], "candidate observation target_packet sha256") != packet_sha:
        _fail("candidate observation is not bound to this target packet")
    candidate = _binding(document.get("candidate"), "candidate observation candidate")
    predictions = document.get("predictions")
    if not isinstance(predictions, list):
        _fail("candidate observation predictions must be a list")
    checked: dict[str, dict[str, float]] = {}
    for index, row in enumerate(predictions):
        if not isinstance(row, Mapping) or set(row) != {"target_id", "x", "y"}:
            _fail(f"prediction {index} has an invalid shape")
        target_id = _text(row["target_id"], f"prediction {index} target_id")
        if target_id in checked:
            _fail("candidate observation has duplicate predictions")
        if target_id not in available_target_ids:
            _fail(f"candidate observation predicts unavailable or extra target {target_id}")
        checked[target_id] = {
            "x": _finite(row["x"], f"prediction {target_id} x"),
            "y": _finite(row["y"], f"prediction {target_id} y"),
        }
    if set(checked) != available_target_ids:
        _fail("candidate observation predictions do not exactly cover available targets")
    return candidate, checked, observation_sha


def score_population_calibration(target_packet: Mapping[str, Any], candidate_observation: Mapping[str, Any]) -> dict[str, Any]:
    """Return deterministic per-family residual analysis for one candidate."""
    packet_sha, targets = _packet_targets(target_packet)
    available = [row for row in targets if row["status"] == "available"]
    candidate, predictions, observation_sha = _observation(
        candidate_observation, packet_sha, {row["target_id"] for row in available},
    )
    per_target: list[dict[str, Any]] = []
    for target in available:
        prediction = predictions[target["target_id"]]
        if prediction["x"] != target["x"]:
            _fail(f"prediction {target['target_id']} x does not exactly match the target")
        signed = prediction["y"] - target["y"]
        biological_sem = target["biological_standard_error"]
        combined = (
            math.hypot(target["digitization_standard_uncertainty"], biological_sem)
            if biological_sem is not None
            else None
        )
        per_target.append({
            "target_id": target["target_id"],
            "target_family": target["target_family"],
            "x": target["x"],
            "target_y": target["y"],
            "prediction_y": prediction["y"],
            "signed_residual": signed,
            "absolute_residual": abs(signed),
            "digitization_standard_uncertainty": target["digitization_standard_uncertainty"],
            "digitization_standardized_residual": signed / target["digitization_standard_uncertainty"],
            "biological_standard_error": biological_sem,
            "combined_standard_uncertainty": combined,
        })
    per_target.sort(key=lambda row: (row["target_family"], row["target_id"]))
    families: list[dict[str, Any]] = []
    for family in sorted({row["target_family"] for row in per_target}):
        rows = [row for row in per_target if row["target_family"] == family]
        absolute = [row["absolute_residual"] for row in rows]
        standardized = [row["digitization_standardized_residual"] for row in rows]
        families.append({
            "target_family": family,
            "count": len(rows),
            "rmse_absolute": math.sqrt(sum(value * value for value in absolute) / len(absolute)),
            "mae": sum(absolute) / len(absolute),
            "max_absolute": max(absolute),
            "digitization_standardized_rmse": math.sqrt(
                sum(value * value for value in standardized) / len(standardized)
            ),
        })
    core = {
        "schema": SCHEMA,
        "canonicalization": CANONICALIZATION,
        "status": "analysis_only",
        "scientific_verdict": None,
        "optimization_allowed": False,
        "target_packet": {"sha256": packet_sha},
        "candidate": candidate,
        "candidate_observation_sha256": observation_sha,
        "per_target": per_target,
        "target_families": families,
    }
    return {**core, "sha256": digest(core)}
