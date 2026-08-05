"""Aggregate the resolved, non-scientific V14 Stage B screen subgates.

This module is deliberately narrower than the Stage B scorer.  The scorer
recomputes observations and source-bound hard gates; this module only joins
digest-bound candidate results and evaluates the five subgates that are
currently resolved enough for a bounded candidate screen.  It never produces
a scientific verdict, a GO/NO-GO decision, a score, or a ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.snr_executable_packet import PacketError, canonical_bytes
from tools.v14_stageB_candidate_batch import (
    StageBCandidateBatchError,
    build_candidate_manifest,
)
from tools.v14_stageB_scorer import (
    INTRINSIC_LESION_SCHEMA,
    StageBScorerError,
    score_intrinsic_lesion_observations,
)


RESULT_SCHEMA = "v14-snr-stageB-resolved-screen-aggregate-v1"
MANIFEST_SCHEMA = "v14-snr-stageB-sobol-candidate-manifest-v1"
SCORE_SCHEMA = "v14-snr-stageB-intrinsic-lesion-score-v1"
CAUSAL_GATE_SCHEMA = "v14-snr-stageB-causal-gates-v1"
INTRINSIC_GATE_IDS = frozenset(
    {
        "nap-complete-lesion",
        "cav2.2-complete-lesion",
        "sk-complete-lesion",
        "hcn-complete-lesion",
    }
)

# Order is part of the output contract.  These are the only metrics allowed
# to influence screen classification.
REQUIRED_METRICS = (
    ("nap-complete-lesion", "spike_count"),
    ("cav2.2-complete-lesion", "isi_cv"),
    ("sk-complete-lesion", "isi_cv"),
    ("hcn-complete-lesion", "lesion_spike_count"),
    ("hcn-complete-lesion", "absolute_baseline_rate_change_fraction"),
)
_REQUIRED_SET = frozenset(REQUIRED_METRICS)
_SCORE_KEYS = {
    "schema",
    "process_status",
    "scientific_verdict",
    "readiness_only",
    "adaptive_candidate",
    "causal_gate_packet",
    "runner_observations",
    "all_intrinsic_lesion_gates_passed",
    "readiness_contract_result",
    "source_equivalence_claimed",
    "results",
}
_MANIFEST_KEYS = {
    "schema", "status", "device", "provenance_exempt", "template", "design",
    "search_space", "candidates", "sha256",
}
_CANDIDATE_ROW_KEYS = {"point_index", "candidate_sha256", "candidate"}
_CANDIDATE_KEYS = {"schema", "candidate_id", "parameters"}
_SCORE_GATE_KEYS = {"gate_id", "source", "preparation", "passed", "hard_gates"}
_SCORE_METRIC_KEYS = {
    "metric", "operator", "evidence_class", "value", "source_equivalence_claimed",
    "status", "passed", "observed", "threshold", "window_s", "cohort_n", "reason",
}


class StageBScreenAggregatorError(ValueError):
    """Raised when a screen input is not an authenticated Stage B artifact."""


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any) -> str:
    try:
        return _digest_bytes(canonical_bytes(value))
    except (PacketError, TypeError, ValueError) as exc:
        raise StageBScreenAggregatorError(f"value cannot be canonically digested: {exc}") from exc


def _valid_digest(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StageBScreenAggregatorError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _repository_relative(root: Path, value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise StageBScreenAggregatorError(f"{context} path must be repository-relative")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.name
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise StageBScreenAggregatorError(f"{context} path must be repository-relative")
    path = root.joinpath(*relative.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise StageBScreenAggregatorError(f"{context} path escapes the repository") from exc
    if path.is_symlink() or not path.is_file():
        raise StageBScreenAggregatorError(f"{context} must be a regular file")
    return path


def _repository_output(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise StageBScreenAggregatorError("output path must be repository-relative")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.name
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise StageBScreenAggregatorError("output path must be repository-relative")
    path = root.joinpath(*relative.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise StageBScreenAggregatorError("output path escapes the repository") from exc
    if path.exists() or path.is_symlink():
        raise StageBScreenAggregatorError("refusing to replace an existing output")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise StageBScreenAggregatorError("output parent must be an existing regular directory")
    return path


def _load_bound_json(root: Path, declaration: Any, context: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(declaration, Mapping) or set(declaration) != {"path", "sha256"}:
        raise StageBScreenAggregatorError(f"{context} must declare only path and sha256")
    expected = _valid_digest(declaration["sha256"], f"{context} digest")
    path = _repository_relative(root, declaration["path"], context)
    raw = path.read_bytes()
    if _digest_bytes(raw) != expected:
        raise StageBScreenAggregatorError(f"{context} digest does not match")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StageBScreenAggregatorError(f"{context} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise StageBScreenAggregatorError(f"{context} must contain a JSON object")
    return path, value


def _finite_number(value: Any, context: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StageBScreenAggregatorError(f"{context} must be numeric")
    if not math.isfinite(float(value)):
        raise StageBScreenAggregatorError(f"{context} must be finite")


def _validate_seed_free(value: Any, context: str) -> None:
    """Reject actual seed/held-out material while allowing null readiness fields."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower().replace("-", "_")
            if "held_out" in lowered or lowered in {"heldout", "held_out_partition"}:
                if item not in (None, [], {}, False, ""):
                    raise StageBScreenAggregatorError(f"{context} contains held-out data")
            if "seed" in lowered:
                if key == "scientific_seed" and item is None:
                    continue
                if key == "reserved_seed_count" and item == 0:
                    continue
                raise StageBScreenAggregatorError(f"{context} contains seed data")
            _validate_seed_free(item, f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_seed_free(item, f"{context}[{index}]")


def _numeric_parameters(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise StageBScreenAggregatorError(f"{context} parameters must be a nonempty object")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise StageBScreenAggregatorError(f"{context} has an invalid parameter key")
        _finite_number(item, f"{context}.{key}")
        result[key] = item
    return result


def _regenerate_candidate_manifest(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    template = manifest.get("template")
    if not isinstance(template, Mapping) or set(template) != {"path", "sha256", "template_id"}:
        raise StageBScreenAggregatorError("candidate manifest has an invalid template binding")
    try:
        return build_candidate_manifest(
            root / str(template["path"]), str(template["sha256"]), root=root,
        )
    except (StageBCandidateBatchError, OSError, TypeError, ValueError) as exc:
        raise StageBScreenAggregatorError(
            f"candidate manifest cannot be regenerated from its template: {exc}"
        ) from exc


def _recompute_scorer(root: Path, score: Mapping[str, Any]) -> dict[str, Any]:
    request = {
        "schema": INTRINSIC_LESION_SCHEMA,
        "readiness_only": score.get("readiness_only"),
        "causal_gate_packet": score.get("causal_gate_packet"),
        "runner_observations": score.get("runner_observations"),
    }
    try:
        return score_intrinsic_lesion_observations(request, root=root)
    except (StageBScorerError, OSError, TypeError, ValueError) as exc:
        raise StageBScreenAggregatorError(
            f"scorer JSON cannot be reproduced from authenticated runner observations: {exc}"
        ) from exc


def _validate_candidate_manifest(root: Path, declaration: Any) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    path, manifest = _load_bound_json(root, declaration, "candidate manifest")
    if set(manifest) != _MANIFEST_KEYS or manifest.get("schema") != MANIFEST_SCHEMA:
        raise StageBScreenAggregatorError("candidate manifest has an invalid schema or shape")
    self_digest = _valid_digest(manifest["sha256"], "candidate manifest self digest")
    if self_digest != _digest({key: value for key, value in manifest.items() if key != "sha256"}):
        raise StageBScreenAggregatorError("candidate manifest self digest is invalid")
    if manifest.get("status") != "preregistered-seed-free-candidate-generation":
        raise StageBScreenAggregatorError("candidate manifest is not the seed-free preregistered design")
    if (
        manifest.get("device") != "not_applicable_non_executed_candidate_design"
        or manifest.get("provenance_exempt")
        != "deterministic non-executed Sobol candidate design; contains no measured result"
    ):
        raise StageBScreenAggregatorError("candidate manifest changed its non-executed evidence boundary")
    if manifest != _regenerate_candidate_manifest(root, manifest):
        raise StageBScreenAggregatorError(
            "candidate manifest does not equal the exact regenerated Sobol design"
        )
    design = manifest.get("design")
    if not isinstance(design, Mapping) or design.get("scientific_seed") is not None:
        raise StageBScreenAggregatorError("candidate manifest contains seed data")
    _validate_seed_free(manifest, "candidate manifest")
    rows = manifest.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise StageBScreenAggregatorError("candidate manifest has no candidates")

    candidates: dict[str, dict[str, Any]] = {}
    digests: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != _CANDIDATE_ROW_KEYS:
            raise StageBScreenAggregatorError(f"candidate manifest row {index} has an invalid shape")
        candidate_digest = _valid_digest(row["candidate_sha256"], f"candidate {index} digest")
        candidate = row["candidate"]
        if not isinstance(candidate, Mapping) or set(candidate) != _CANDIDATE_KEYS:
            raise StageBScreenAggregatorError(f"candidate {index} has an invalid shape")
        if candidate.get("schema") != "sim-adaptive-candidate-v1":
            raise StageBScreenAggregatorError(f"candidate {index} has the wrong schema")
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id != candidate_id.strip():
            raise StageBScreenAggregatorError(f"candidate {index} has an invalid candidate_id")
        parameters = _numeric_parameters(candidate["parameters"], f"candidate {candidate_id}")
        canonical_candidate = {
            "schema": candidate["schema"],
            "candidate_id": candidate_id,
            "parameters": parameters,
        }
        if candidate_digest != _digest(canonical_candidate):
            raise StageBScreenAggregatorError(f"candidate {candidate_id} parameter identity does not match its digest")
        if candidate_id in candidates or candidate_digest in digests:
            raise StageBScreenAggregatorError("candidate manifest contains duplicate candidates")
        candidates[candidate_id] = {
            "candidate_id": candidate_id,
            "candidate_sha256": candidate_digest,
            "parameters": parameters,
        }
        digests.add(candidate_digest)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _digest_bytes(path.read_bytes()),
    }, candidates


def _contract_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: value[key]
        for key in ("metric", "operator", "evidence_class", "window_s", "cohort_n")
        if key in value
    }
    if "value" in value:
        result["value"] = value["value"]
    elif "threshold" in value:
        result["value"] = value["threshold"]
    return result


def _contract_matches_score(hard_result: Mapping[str, Any], contract: Mapping[str, Any]) -> bool:
    # Unavailable scorer entries intentionally omit thresholds and cohort
    # details. Their identity is still authenticated, while unavailable status
    # prevents them affecting classification.
    if hard_result.get("status") == "unavailable":
        base = {
            key: hard_result[key]
            for key in ("metric", "operator", "evidence_class")
            if key in hard_result
        }
        return base == {
            key: contract[key]
            for key in ("metric", "operator", "evidence_class")
            if key in contract
        }
    return _contract_fields(hard_result) == _contract_fields(contract)


def _validate_gate_packet(root: Path, binding: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, Mapping[str, Any]]]:
    path, packet = _load_bound_json(root, binding, "causal gate packet")
    if packet.get("schema") != CAUSAL_GATE_SCHEMA:
        raise StageBScreenAggregatorError("causal gate packet has the wrong schema")
    gates = packet.get("causal_gates")
    if not isinstance(gates, list):
        raise StageBScreenAggregatorError("causal gate packet has no gate list")
    selected: dict[str, Mapping[str, Any]] = {}
    for gate in gates:
        if not isinstance(gate, Mapping) or not isinstance(gate.get("id"), str):
            raise StageBScreenAggregatorError("causal gate packet contains an invalid gate")
        gate_id = str(gate["id"])
        if gate_id in selected:
            raise StageBScreenAggregatorError(f"causal gate packet duplicates {gate_id}")
        if gate_id in INTRINSIC_GATE_IDS:
            selected[gate_id] = gate
    if set(selected) != INTRINSIC_GATE_IDS:
        raise StageBScreenAggregatorError("causal gate packet does not contain exactly the intrinsic gates")
    for gate_id, gate in selected.items():
        hard_gates = gate.get("hard_gates")
        if not isinstance(hard_gates, list) or not hard_gates:
            raise StageBScreenAggregatorError(f"{gate_id} has no hard-gate contract")
        seen: set[str] = set()
        for hard_gate in hard_gates:
            if not isinstance(hard_gate, Mapping) or not isinstance(hard_gate.get("metric"), str):
                raise StageBScreenAggregatorError(f"{gate_id} has an invalid hard-gate contract")
            metric = str(hard_gate["metric"])
            if metric in seen:
                raise StageBScreenAggregatorError(f"{gate_id} duplicates hard-gate metric {metric}")
            seen.add(metric)
    filed_metrics = {
        (gate_id, hard_gate.get("metric"))
        for gate_id, gate in selected.items()
        for hard_gate in gate.get("hard_gates", [])
    }
    if not _REQUIRED_SET.issubset(filed_metrics):
        raise StageBScreenAggregatorError("causal gate packet is missing a required resolved subgate")
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _digest_bytes(path.read_bytes()),
    }, selected


def _validate_score(
    root: Path,
    declaration: Any,
    manifest_candidate: Mapping[str, Any],
    causal_binding: dict[str, str] | None,
    causal_gates: dict[str, Mapping[str, Any]] | None,
) -> tuple[dict[str, str], dict[str, Any], dict[str, str], dict[str, Mapping[str, Any]]]:
    path, score = _load_bound_json(root, declaration, "scorer JSON")
    if set(score) != _SCORE_KEYS or score.get("schema") != SCORE_SCHEMA:
        raise StageBScreenAggregatorError("scorer JSON has an invalid schema or shape")
    if score.get("process_status") != "completed":
        raise StageBScreenAggregatorError("scorer JSON is not a completed score")
    if score.get("scientific_verdict") is not None:
        raise StageBScreenAggregatorError("scorer scientific verdicts are forbidden")
    if score.get("source_equivalence_claimed") is not False:
        raise StageBScreenAggregatorError("scorer source-equivalence claims are forbidden")
    if score.get("readiness_only") != {
        "enabled": True, "reserved_seed_count": 0, "scientific_seed": None,
    }:
        raise StageBScreenAggregatorError("scorer is not seed-free readiness-only data")
    _validate_seed_free(score, "scorer JSON")

    candidate = score.get("adaptive_candidate")
    if not isinstance(candidate, Mapping) or set(candidate) != {
        "candidate_id", "candidate_sha256", "effective_parameters"
    }:
        raise StageBScreenAggregatorError("scorer candidate identity has an invalid shape")
    if (
        candidate.get("candidate_id") != manifest_candidate["candidate_id"]
        or candidate.get("candidate_sha256") != manifest_candidate["candidate_sha256"]
        or candidate.get("effective_parameters") != manifest_candidate["parameters"]
    ):
        raise StageBScreenAggregatorError("scorer candidate parameter identity does not match the manifest")
    _valid_digest(candidate["candidate_sha256"], "scorer candidate digest")

    binding_value = score.get("causal_gate_packet")
    if not isinstance(binding_value, Mapping) or set(binding_value) != {"path", "sha256"}:
        raise StageBScreenAggregatorError("scorer has no exact causal-gate binding")
    actual_binding, gates = _validate_gate_packet(root, binding_value)
    if causal_binding is not None and actual_binding != causal_binding:
        raise StageBScreenAggregatorError("scorer causal-gate bindings do not match")
    if causal_gates is not None and {
        key: dict(value) for key, value in gates.items()
    } != {key: dict(value) for key, value in causal_gates.items()}:
        raise StageBScreenAggregatorError("scorer causal-gate contracts do not match")

    results = score.get("results")
    if not isinstance(results, list) or len(results) != len(INTRINSIC_GATE_IDS):
        raise StageBScreenAggregatorError("scorer contains unknown or extra gates")
    by_gate: dict[str, Mapping[str, Any]] = {}
    for result in results:
        if not isinstance(result, Mapping) or set(result) != _SCORE_GATE_KEYS:
            raise StageBScreenAggregatorError("scorer contains an invalid or extra gate")
        gate_id = result.get("gate_id")
        if gate_id not in INTRINSIC_GATE_IDS or gate_id in by_gate:
            raise StageBScreenAggregatorError("scorer contains an unknown or duplicate gate")
        if result.get("source") != gates[gate_id].get("source") or result.get("preparation") != gates[gate_id].get("preparation"):
            raise StageBScreenAggregatorError(f"scorer gate {gate_id} does not match the causal contract")
        by_gate[gate_id] = result
    if set(by_gate) != INTRINSIC_GATE_IDS:
        raise StageBScreenAggregatorError("scorer does not contain exactly the intrinsic gates")

    for gate_id, gate in gates.items():
        contract = gate.get("hard_gates")
        hard_results = by_gate[gate_id].get("hard_gates")
        if not isinstance(contract, list) or not isinstance(hard_results, list) or len(contract) != len(hard_results):
            raise StageBScreenAggregatorError(f"scorer gate {gate_id} has unknown or missing hard gates")
        expected_metrics = [item.get("metric") for item in contract]
        actual_metrics: list[Any] = []
        for hard_result in hard_results:
            if not isinstance(hard_result, Mapping) or set(hard_result) - _SCORE_METRIC_KEYS:
                raise StageBScreenAggregatorError(f"scorer gate {gate_id} contains an unknown hard gate field")
            metric = hard_result.get("metric")
            if metric in actual_metrics:
                raise StageBScreenAggregatorError(f"scorer gate {gate_id} duplicates a hard gate")
            actual_metrics.append(metric)
            matching = next((item for item in contract if item.get("metric") == metric), None)
            if matching is None or not _contract_matches_score(hard_result, matching):
                raise StageBScreenAggregatorError(f"scorer gate {gate_id} hard-gate contract mismatch")
            if hard_result.get("source_equivalence_claimed") is not False:
                raise StageBScreenAggregatorError("scorer hard-gate source-equivalence claims are forbidden")
            if hard_result.get("status") not in {"scored", "unavailable"}:
                raise StageBScreenAggregatorError(f"scorer gate {gate_id} has an invalid hard-gate status")
            passed = hard_result.get("passed")
            if hard_result.get("status") == "unavailable" and passed is not None:
                raise StageBScreenAggregatorError("unavailable hard gates must have passed=null")
            if hard_result.get("status") == "scored" and not isinstance(passed, bool):
                raise StageBScreenAggregatorError("scored hard gates must have a boolean passed value")
        if actual_metrics != expected_metrics:
            raise StageBScreenAggregatorError(f"scorer gate {gate_id} hard-gate ordering or identity mismatch")
        expected_passed = (
            False if any(item.get("passed") is False for item in hard_results)
            else None if any(item.get("passed") is None for item in hard_results)
            else True
        )
        if by_gate[gate_id].get("passed") is not expected_passed:
            raise StageBScreenAggregatorError(f"scorer gate {gate_id} summary is inconsistent with its hard gates")

    expected_all_passed = (
        False if any(item.get("passed") is False for item in by_gate.values())
        else None if any(item.get("passed") is None for item in by_gate.values())
        else True
    )
    if score.get("all_intrinsic_lesion_gates_passed") is not expected_all_passed:
        raise StageBScreenAggregatorError("scorer aggregate summary is inconsistent with its gates")
    expected_contract_result = (
        "FAIL" if expected_all_passed is False
        else "UNAVAILABLE" if expected_all_passed is None
        else "PASS"
    )
    if score.get("readiness_contract_result") != expected_contract_result:
        raise StageBScreenAggregatorError("scorer readiness summary is inconsistent with its gates")

    if score != _recompute_scorer(root, score):
        raise StageBScreenAggregatorError(
            "scorer JSON does not equal the score recomputed from authenticated runner observations"
        )

    return (
        {"path": path.relative_to(root).as_posix(), "sha256": _digest_bytes(path.read_bytes())},
        score,
        actual_binding,
        gates,
    )


def _classification(score: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    resolved: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    by_gate = {item["gate_id"]: item for item in score["results"]}
    for gate_id, metric in REQUIRED_METRICS:
        hard_gate = next(item for item in by_gate[gate_id]["hard_gates"] if item["metric"] == metric)
        record = {
            "gate_id": gate_id,
            "metric": metric,
            "status": hard_gate["status"],
            "passed": hard_gate["passed"],
        }
        resolved.append(record)
    for result in score["results"]:
        for hard_gate in result["hard_gates"]:
            key = (result["gate_id"], hard_gate["metric"])
            if key not in _REQUIRED_SET:
                missing.append({
                    "gate_id": result["gate_id"],
                    "metric": hard_gate["metric"],
                    "status": hard_gate["status"],
                    "passed": None,
                    "reason": hard_gate.get("reason"),
                })
    if any(item["passed"] is False for item in resolved):
        classification = "screen_fail"
    elif any(item["passed"] is None for item in resolved):
        classification = "screen_inconclusive"
    elif all(item["passed"] is True for item in resolved):
        classification = "screen_pass"
    else:
        classification = "screen_invalid"
    return classification, resolved, missing


def aggregate_stageB_screen(
    candidate_manifest: Mapping[str, Any],
    scorer_jsons: Sequence[Mapping[str, Any]],
    *,
    root: str | Path,
) -> dict[str, Any]:
    """Aggregate one digest-bound scorer JSON per manifest candidate.

    ``candidate_manifest`` and each item in ``scorer_jsons`` are binding
    objects of the form ``{"path": ..., "sha256": ...}``.  All structural,
    identity, authority, and provenance failures raise
    :class:`StageBScreenAggregatorError`; candidate classifications are only
    emitted after those checks pass.
    """
    root_path = Path(root).expanduser().resolve(strict=True)
    manifest_binding, candidates = _validate_candidate_manifest(root_path, candidate_manifest)
    if not isinstance(scorer_jsons, Sequence) or isinstance(scorer_jsons, (str, bytes)):
        raise StageBScreenAggregatorError("scorer_jsons must be a sequence of digest bindings")
    if len(scorer_jsons) != len(candidates):
        raise StageBScreenAggregatorError("there must be exactly one scorer JSON per candidate")

    scores: dict[str, tuple[dict[str, str], dict[str, Any]]] = {}
    causal_binding: dict[str, str] | None = None
    causal_gates: dict[str, Mapping[str, Any]] | None = None
    for declaration in scorer_jsons:
        candidate_id = None
        _, raw_score = _load_bound_json(root_path, declaration, "scorer JSON")
        candidate_value = raw_score.get("adaptive_candidate")
        if isinstance(candidate_value, Mapping) and isinstance(candidate_value.get("candidate_id"), str):
            candidate_id = candidate_value["candidate_id"]
        if candidate_id not in candidates:
            raise StageBScreenAggregatorError("scorer references an unknown candidate")
        if candidate_id in scores:
            raise StageBScreenAggregatorError("duplicate scorer candidates are forbidden")
        score_binding, score, actual_binding, actual_gates = _validate_score(
            root_path, declaration, candidates[candidate_id], causal_binding, causal_gates
        )
        causal_binding = actual_binding
        causal_gates = actual_gates
        scores[candidate_id] = (score_binding, score)
    if set(scores) != set(candidates):
        raise StageBScreenAggregatorError("scorer set does not exactly match the candidate manifest")

    output_candidates = []
    for candidate_id, candidate in candidates.items():
        score_binding, score = scores[candidate_id]
        classification, resolved, missing = _classification(score)
        output_candidates.append({
            "candidate_id": candidate_id,
            "candidate_sha256": candidate["candidate_sha256"],
            "scorer_json": score_binding,
            "classification": classification,
            "resolved_metrics": resolved,
            "missing_contract_metrics": missing,
        })

    body = {
        "schema": RESULT_SCHEMA,
        "candidate_manifest": manifest_binding,
        "causal_gate_packet": causal_binding,
        "required_metrics": [
            {"gate_id": gate_id, "metric": metric} for gate_id, metric in REQUIRED_METRICS
        ],
        "candidates": output_candidates,
        "scientific_verdict": None,
        "source_equivalence_claimed": False,
    }
    # The two explicit null/false fields make the non-scientific boundary
    # machine-checkable without introducing a GO-like aggregate result.
    return {**body, "sha256": _digest(body)}


aggregate_screen = aggregate_stageB_screen
run_screen_aggregation = aggregate_stageB_screen


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--candidate-manifest-sha256", required=True)
    parser.add_argument("--scorer", action="append", nargs=2, metavar=("PATH", "SHA256"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    binding = {"path": args.candidate_manifest, "sha256": args.candidate_manifest_sha256}
    scorers = [{"path": path, "sha256": digest} for path, digest in args.scorer]
    try:
        result = aggregate_stageB_screen(binding, scorers, root=args.root)
        output = _repository_output(
            Path(args.root).expanduser().resolve(strict=True), args.output,
        )
        try:
            with output.open("xb") as handle:
                handle.write(canonical_bytes(result) + b"\n")
        except FileExistsError as exc:
            raise StageBScreenAggregatorError(
                "refusing to replace an existing output"
            ) from exc
    except (StageBScreenAggregatorError, OSError, PacketError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
