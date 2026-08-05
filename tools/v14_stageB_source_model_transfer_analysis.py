#!/usr/bin/env python3
"""Independently authenticate and score the Stage B source-model transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

# These are deliberately imported, not copied or modified.  The source transfer
# therefore uses exactly the estimators preregistered for the original assay.
from tools.v14_stageB_fast_channel_clamp_analysis import (
    _fit_boltzmann,
    _fit_recovery,
    _fit_tail,
    _rise_and_decay,
    _targets,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC_SCHEMA = "v14-snr-stageB-source-model-transfer-v1"
OBSERVATION_SCHEMA = "v14-snr-stageB-source-model-observation-v1"
PROVENANCE_SCHEMA = "sim-run-provenance-v2"
RECEIPT_SCHEMA = "sim-execution-receipt-v2"
OUTPUT_SCHEMA = "v14-snr-stageB-source-model-transfer-analysis-v1"
LEDGER_SCHEMA = "v14-snr-stageB-source-model-transfer-consumption-v1"
RUNNER = "research/runners/v14_stageB_source_model_transfer.py"
PARITY_RTOL = 5e-8
PARITY_ATOL = 5e-10

KHALIQ = "khaliq_raman_13_state"
BALBI = "balbi_nav16_six_state"
LABRO = "labro_2015_four_state"
DESAI = "desai_2008_control"
MODEL_IDS = frozenset({KHALIQ, BALBI, LABRO, DESAI})

SODIUM_METRICS = (
    "fast_na.activation_vhalf_mV",
    "fast_na.activation_slope_mV",
    "fast_na.inactivation_vhalf_mV",
    "fast_na.inactivation_slope_mV",
    "fast_na.activation_10_90_at_0_mV_ms",
    "fast_na.inactivation_10_90_at_0_mV_ms",
    "fast_na.recovery_fast_tau_ms",
    "fast_na.recovery_slow_tau_ms",
    "fast_na.recovery_fast_fraction",
    "fast_na.deactivation_at_minus_40_mV_ms",
)
LABRO_METRICS = (
    "kv3_like.activation_vhalf_mV",
    "kv3_like.activation_slope_mV",
    "kv3_like.rise_20_80_at_plus_40_mV_ms",
    "kv3_like.deactivation_at_minus_60_mV_ms",
    "kv3_like.deactivation_at_minus_50_mV_ms",
    "kv3_like.deactivation_at_minus_40_mV_ms",
)
DESAI_METRICS = (
    "kv3_like.activation_vhalf_mV",
    "kv3_like.activation_slope_mV",
    "kv3_like.inactivation_vhalf_mV",
    "kv3_like.inactivation_slope_mV",
    "kv3_like.rise_20_80_at_plus_40_mV_ms",
    "kv3_like.deactivation_at_minus_60_mV_ms",
    "kv3_like.deactivation_at_minus_50_mV_ms",
    "kv3_like.deactivation_at_minus_40_mV_ms",
)


class SourceModelTransferAnalysisError(ValueError):
    """Raised when evidence is incomplete, altered, or outside the seal."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceModelTransferAnalysisError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256(value: Any, context: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{context} must be a lowercase SHA-256 digest",
    )
    return value


def _inside_file(root: Path, value: Any, context: str) -> tuple[str, Path]:
    _require(
        isinstance(value, str) and value and "\\" not in value and "\x00" not in value,
        f"{context} path is invalid",
    )
    relative = PurePosixPath(value)
    _require(
        not relative.is_absolute()
        and str(relative) == value
        and all(part not in {"", ".", ".."} for part in relative.parts),
        f"{context} path is not canonical",
    )
    unresolved = root.joinpath(*relative.parts)
    _require(not unresolved.is_symlink(), f"{context} must not be a symbolic link")
    path = unresolved.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SourceModelTransferAnalysisError(f"{context} escapes repository") from exc
    _require(path.is_file(), f"{context} must be a regular file")
    return value, path


def _load_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SourceModelTransferAnalysisError(f"{context} is not valid JSON") from exc
    _require(isinstance(value, dict), f"{context} must contain an object")
    return value


def _load_binding(
    root: Path, binding: Any, context: str
) -> tuple[dict[str, str], Path]:
    _require(
        isinstance(binding, Mapping) and set(binding) == {"path", "sha256"},
        f"{context} binding is invalid",
    )
    relative, path = _inside_file(root, binding["path"], context)
    observed = _file_digest(path)
    _require(observed == _sha256(binding["sha256"], f"{context} sha256"), f"{context} digest mismatch")
    return {"path": relative, "sha256": observed}, path


def _load_bound_json(
    root: Path, binding: Any, context: str
) -> tuple[dict[str, str], dict[str, Any]]:
    authenticated, path = _load_binding(root, binding, context)
    return authenticated, _load_json(path, context)


def _load_declared_json(
    root: Path, relative: Any, context: str
) -> tuple[dict[str, str], dict[str, Any]]:
    declared, path = _inside_file(root, relative, context)
    binding = {"path": declared, "sha256": _file_digest(path)}
    return binding, _load_json(path, context)


def _semantic_digest(document: Mapping[str, Any]) -> str:
    return _digest({key: value for key, value in document.items() if key != "sha256"})


def _models(spec: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = spec.get("models")
    _require(isinstance(rows, list) and len(rows) == 4, "exactly four source models are required")
    result = {
        row.get("model_id"): row for row in rows if isinstance(row, Mapping)
    }
    _require(set(result) == MODEL_IDS, "source model set changed")
    condition_pairs: set[tuple[str, str]] = set()
    for model_id, row in result.items():
        _require(row.get("calibration_allowed") is False, f"{model_id} permits calibration")
        _require(row.get("conductance_fitting_allowed") is False, f"{model_id} permits conductance fitting")
        _sha256(row.get("model_fingerprint"), f"{model_id} fingerprint")
        conditions = row.get("conditions")
        _require(isinstance(conditions, list) and conditions, f"{model_id} has no conditions")
        for condition in conditions:
            _require(isinstance(condition, Mapping), f"{model_id} condition is invalid")
            condition_id = condition.get("condition_id")
            _require(isinstance(condition_id, str) and condition_id, f"{model_id} condition identity is invalid")
            pair = (model_id, condition_id)
            _require(pair not in condition_pairs, "source condition is duplicated")
            condition_pairs.add(pair)
    _require(len(condition_pairs) == 6, "exactly six source-model conditions are required")
    labro_temperatures = sorted(
        float(condition.get("temperature_c"))
        for condition in result[LABRO]["conditions"]
    )
    _require(labro_temperatures == [20.0, 22.5, 25.0], "Labro temperature envelope changed")
    _require(len(result[KHALIQ]["conditions"]) == 1, "Khaliq condition count changed")
    _require(len(result[BALBI]["conditions"]) == 1, "Balbi condition count changed")
    _require(len(result[DESAI]["conditions"]) == 1, "Desai condition count changed")
    return result


def _expected_argv(
    root: Path, spec_binding: Mapping[str, str], row: Mapping[str, Any]
) -> list[str]:
    return [
        str(root / RUNNER),
        "--spec", spec_binding["path"],
        "--spec-sha256", spec_binding["sha256"],
        "--model-id", row["model_id"],
        "--condition-id", row["condition_id"],
        "--backend", row["backend"],
        "--out", row["output"],
    ]


def _command_suffix(argv: Any, expected: Sequence[str]) -> bool:
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        return False
    # execution_receipt may invoke the runner as a module or script.  The
    # entire sealed argument sequence from --spec onward must still be exact.
    return len(argv) >= len(expected) - 1 and argv[-(len(expected) - 1):] == list(expected[1:])


def _authenticate_job(
    root: Path,
    spec: Mapping[str, Any],
    spec_binding: Mapping[str, str],
    model: Mapping[str, Any],
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = f"{row['model_id']}/{row['condition_id']}/{row['backend']}"
    artifact_relative, artifact_path = _inside_file(root, row.get("output"), f"{identity} observation")
    artifact_sha = _file_digest(artifact_path)
    document = _load_json(artifact_path, f"{identity} observation")
    _require(document.get("sha256") == _semantic_digest(document), f"{identity} observation self digest is invalid")

    provenance_relative = row.get("provenance", f"{row['output']}.prov.json")
    provenance_binding, provenance = _load_declared_json(
        root, provenance_relative, f"{identity} provenance"
    )
    receipt_binding, receipt = _load_declared_json(
        root, row.get("receipt"), f"{identity} receipt"
    )
    expected_argv = _expected_argv(root, spec_binding, row)

    _require(provenance.get("schema") == PROVENANCE_SCHEMA, f"{identity} provenance schema mismatch")
    _require(provenance.get("runner") == RUNNER, f"{identity} provenance runner mismatch")
    _require(provenance.get("argv") == expected_argv, f"{identity} provenance argv mismatch")
    _require(provenance.get("artifact") == artifact_relative, f"{identity} provenance artifact mismatch")
    _require(provenance.get("sim_backend_requested") == row["backend"], f"{identity} requested backend mismatch")
    _require(provenance.get("sim_backend") == row["backend"], f"{identity} actual backend mismatch")
    _require(isinstance(provenance.get("run_id"), str) and provenance["run_id"], f"{identity} run id is invalid")
    _require(
        isinstance(provenance.get("git_sha"), str)
        and 7 <= len(provenance["git_sha"]) <= 64
        and all(character in "0123456789abcdefABCDEF" for character in provenance["git_sha"]),
        f"{identity} provenance Git identity is invalid",
    )
    _require(provenance.get("source_kind") in {"git", "git_archive"}, f"{identity} provenance source kind is invalid")
    _sha256(provenance.get("source_manifest_sha256"), f"{identity} provenance source manifest")
    _require(isinstance(provenance.get("started_utc_ns"), int), f"{identity} provenance start is invalid")
    _require(isinstance(provenance.get("ended_utc_ns"), int) and provenance["ended_utc_ns"] >= provenance["started_utc_ns"], f"{identity} provenance end is invalid")

    _require(receipt.get("schema") == RECEIPT_SCHEMA, f"{identity} receipt schema mismatch")
    _require(receipt.get("status") == "success" and receipt.get("exit_code") == 0, f"{identity} receipt is not successful")
    _require(receipt.get("device") == row.get("device"), f"{identity} receipt device mismatch")
    receipt_artifact = receipt.get("artifact")
    _require(
        isinstance(receipt_artifact, Mapping)
        and receipt_artifact.get("path") == artifact_relative
        and receipt_artifact.get("sha256") == artifact_sha
        and receipt_artifact.get("size_bytes") == artifact_path.stat().st_size,
        f"{identity} receipt artifact mismatch",
    )
    receipt_provenance = receipt.get("provenance")
    _require(
        isinstance(receipt_provenance, Mapping)
        and receipt_provenance.get("path") == provenance_binding["path"]
        and receipt_provenance.get("sha256") == provenance_binding["sha256"]
        and receipt_provenance.get("run_id") == provenance["run_id"]
        and receipt_provenance.get("started_utc_ns") == provenance["started_utc_ns"]
        and receipt_provenance.get("ended_utc_ns") == provenance["ended_utc_ns"],
        f"{identity} receipt provenance mismatch",
    )
    _require(_command_suffix(receipt.get("argv"), expected_argv), f"{identity} receipt argv mismatch")
    _require(
        isinstance(receipt.get("source"), Mapping)
        and receipt["source"].get("git_sha") == provenance.get("git_sha")
        and receipt["source"].get("kind") == provenance.get("source_kind")
        and receipt["source"].get("manifest_sha256") == provenance.get("source_manifest_sha256"),
        f"{identity} receipt source identity mismatch",
    )

    expected_authority = "cpu_reference" if row["backend"] == "numpy" else "gpu_parity_only"
    condition = next(
        item for item in model["conditions"] if item["condition_id"] == row["condition_id"]
    )
    _require(
        document.get("schema") == OBSERVATION_SCHEMA
        and document.get("execution_spec") == spec_binding
        and document.get("command_authority") == spec["command_authority"]
        and document.get("implementation") == spec["implementation"]
        and document.get("model_id") == row["model_id"]
        and document.get("condition_id") == row["condition_id"]
        and document.get("condition") == condition
        and document.get("model_fingerprint") == model["model_fingerprint"]
        and document.get("backend") == row["backend"]
        and document.get("device") == row["device"]
        and document.get("authority") == expected_authority == row.get("authority")
        and document.get("dtype") == "float64"
        and document.get("analysis_status") == "raw_unanalyzed"
        and document.get("scientific_verdict") is None
        and document.get("candidate_calibration_allowed") is False
        and document.get("conductance_fitting_allowed") is False,
        f"{identity} observation identity or authority is invalid",
    )
    _validate_assays(document, spec["_commands"])
    binding = {
        "artifact": {"path": artifact_relative, "sha256": artifact_sha, "self_sha256": document["sha256"]},
        "provenance": provenance_binding,
        "receipt": receipt_binding,
        "run_id": provenance["run_id"],
    }
    return binding, document


def _validate_assays(document: Mapping[str, Any], commands_document: Mapping[str, Any]) -> None:
    assays = document.get("assays")
    _require(isinstance(assays, Mapping), "observation assays are invalid")
    model_id = document["model_id"]
    if model_id in {KHALIQ, BALBI}:
        expected = {"fast_na_activation", "fast_na_inactivation", "fast_na_composite_zero", "fast_na_recovery", "fast_na_deactivation"}
        unavailable: list[str] = []
    elif model_id == LABRO:
        expected = {"kv3_activation", "kv3_rise", "kv3_deactivation"}
        unavailable = ["kv3_inactivation"]
    else:
        expected = {"kv3_activation", "kv3_rise", "kv3_inactivation", "kv3_deactivation"}
        unavailable = []
    _require(set(assays) == expected, f"{model_id} assay set changed")
    _require(document.get("unavailable_assays") == unavailable, f"{model_id} unavailable assay declaration changed")
    interval = commands_document["numeric_contract"]["clamp_sample_interval_ms"]
    _require(document.get("sample_interval_ms") == interval, "sample interval changed")
    command_map = commands_document["commands"]
    ladders = {
        "fast_na_activation": ("command_voltage_mv", "fast_na_activation", "test_mV"),
        "fast_na_inactivation": ("prepulse_voltage_mv", "fast_na_inactivation", "prepulse_mV"),
        "fast_na_recovery": ("recovery_duration_ms", "fast_na_recovery", "recovery_duration_ms"),
        "fast_na_deactivation": ("command_voltage_mv", "fast_na_deactivation", "test_mV"),
        "kv3_activation": ("command_voltage_mv", "kv3_activation", "test_mV"),
        "kv3_inactivation": ("prepulse_voltage_mv", "kv3_inactivation", "prepulse_mV"),
        "kv3_deactivation": ("command_voltage_mv", "kv3_deactivation", "test_mV"),
    }
    for assay_name, assay in assays.items():
        _require(isinstance(assay, Mapping), f"{assay_name} assay is invalid")
        elapsed = np.asarray(assay.get("elapsed_ms"), dtype=np.float64)
        _require(elapsed.ndim == 1 and elapsed.size and np.all(np.isfinite(elapsed)) and np.all(np.diff(elapsed) > 0), f"{assay_name} elapsed grid is invalid")
        _require(np.isclose(elapsed[0], interval, rtol=0.0, atol=1e-12), f"{assay_name} first sample changed")
        if assay_name in ladders:
            field, command_name, command_field = ladders[assay_name]
            observed = np.asarray(assay.get(field), dtype=np.float64)
            expected_ladder = np.asarray(command_map[command_name][command_field], dtype=np.float64)
            _require(observed.shape == expected_ladder.shape and np.allclose(observed, expected_ladder, rtol=0.0, atol=1e-12), f"{assay_name} command ladder changed")
        for field, value in assay.items():
            if isinstance(value, list):
                array = np.asarray(value)
                if np.issubdtype(array.dtype, np.number):
                    _require(np.all(np.isfinite(array)), f"{assay_name}.{field} contains non-finite values")


def _condition_metrics(document: Mapping[str, Any]) -> dict[str, float]:
    assays = document["assays"]
    model_id = document["model_id"]
    if model_id in {KHALIQ, BALBI}:
        activation_vhalf, activation_slope = _fit_boltzmann(
            assays["fast_na_activation"]["command_voltage_mv"],
            assays["fast_na_activation"]["ladder_normalized_peak_conductance"],
            activation=True,
        )
        inactivation_vhalf, inactivation_slope = _fit_boltzmann(
            assays["fast_na_inactivation"]["prepulse_voltage_mv"],
            assays["fast_na_inactivation"]["ladder_normalized_peak_test_current"],
            activation=False,
        )
        rise, decay = _rise_and_decay(assays["fast_na_composite_zero"], 0.1, 0.9)
        fraction, fast_tau, slow_tau = _fit_recovery(
            assays["fast_na_recovery"]["recovery_duration_ms"],
            assays["fast_na_recovery"]["ladder_normalized_peak_test_current"],
        )
        return {
            "fast_na.activation_vhalf_mV": activation_vhalf,
            "fast_na.activation_slope_mV": activation_slope,
            "fast_na.inactivation_vhalf_mV": inactivation_vhalf,
            "fast_na.inactivation_slope_mV": inactivation_slope,
            "fast_na.activation_10_90_at_0_mV_ms": rise,
            "fast_na.inactivation_10_90_at_0_mV_ms": float(decay),
            "fast_na.recovery_fast_tau_ms": fast_tau,
            "fast_na.recovery_slow_tau_ms": slow_tau,
            "fast_na.recovery_fast_fraction": fraction,
            "fast_na.deactivation_at_minus_40_mV_ms": _fit_tail(assays["fast_na_deactivation"], "normalized_na_current", -40.0),
        }
    activation_vhalf, activation_slope = _fit_boltzmann(
        assays["kv3_activation"]["command_voltage_mv"],
        assays["kv3_activation"]["ladder_normalized_peak_conductance"],
        activation=True,
    )
    rise, _ = _rise_and_decay(assays["kv3_rise"], 0.2, 0.8)
    result = {
        "kv3_like.activation_vhalf_mV": activation_vhalf,
        "kv3_like.activation_slope_mV": activation_slope,
        "kv3_like.rise_20_80_at_plus_40_mV_ms": rise,
        "kv3_like.deactivation_at_minus_60_mV_ms": _fit_tail(assays["kv3_deactivation"], "normalized_kv3_current", -60.0),
        "kv3_like.deactivation_at_minus_50_mV_ms": _fit_tail(assays["kv3_deactivation"], "normalized_kv3_current", -50.0),
        "kv3_like.deactivation_at_minus_40_mV_ms": _fit_tail(assays["kv3_deactivation"], "normalized_kv3_current", -40.0),
    }
    if model_id == DESAI:
        inactivation_vhalf, inactivation_slope = _fit_boltzmann(
            assays["kv3_inactivation"]["prepulse_voltage_mv"],
            assays["kv3_inactivation"]["ladder_normalized_peak_test_current"],
            activation=False,
        )
        result["kv3_like.inactivation_vhalf_mV"] = inactivation_vhalf
        result["kv3_like.inactivation_slope_mV"] = inactivation_slope
    return result


def _target_map(authority: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    _require(
        authority.get("schema") == "v14-snr-stageB-structural-successor-protocol-v2",
        "target authority schema mismatch",
    )
    try:
        raw_targets = _targets(authority)
    except (KeyError, TypeError, ValueError) as exc:
        raise SourceModelTransferAnalysisError("target authority targets are invalid") from exc
    targets = {
        name: {"mean": target["mean"], "sem": target["sem"]}
        for name, target in raw_targets.items()
    }
    expected = set(SODIUM_METRICS) | set(DESAI_METRICS)
    _require(set(targets) == expected, "target metric set changed")
    for name, target in targets.items():
        _require(isinstance(target, Mapping) and set(target) == {"mean", "sem"}, f"{name} target is invalid")
        mean, sem = target["mean"], target["sem"]
        _require(isinstance(mean, (int, float)) and np.isfinite(mean), f"{name} mean is invalid")
        _require(isinstance(sem, (int, float)) and np.isfinite(sem) and sem >= 0.0, f"{name} SEM is invalid")
    return targets


def _pointwise_parity(cpu: Mapping[str, Any], gpu: Mapping[str, Any]) -> dict[str, Any]:
    cpu_assays, gpu_assays = cpu["assays"], gpu["assays"]
    _require(set(cpu_assays) == set(gpu_assays), "CPU/GPU assay sets differ")
    failures: list[dict[str, Any]] = []
    compared_points = 0
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for assay_name in sorted(cpu_assays):
        _require(set(cpu_assays[assay_name]) == set(gpu_assays[assay_name]), f"{assay_name} CPU/GPU fields differ")
        for field in sorted(cpu_assays[assay_name]):
            left_value, right_value = cpu_assays[assay_name][field], gpu_assays[assay_name][field]
            if not isinstance(left_value, list) and not isinstance(right_value, list):
                _require(left_value == right_value, f"{assay_name}.{field} CPU/GPU metadata differ")
                continue
            left, right = np.asarray(left_value), np.asarray(right_value)
            _require(left.shape == right.shape, f"{assay_name}.{field} CPU/GPU shapes differ")
            if not np.issubdtype(left.dtype, np.number) or not np.issubdtype(right.dtype, np.number):
                _require(left.tolist() == right.tolist(), f"{assay_name}.{field} CPU/GPU values differ")
                continue
            left, right = left.astype(np.float64), right.astype(np.float64)
            compared_points += left.size
            absolute = np.abs(left - right)
            denominator = np.maximum(np.abs(left), PARITY_ATOL)
            relative = absolute / denominator
            if absolute.size:
                maximum_absolute = max(maximum_absolute, float(np.max(absolute)))
                maximum_relative = max(maximum_relative, float(np.max(relative)))
            passed = np.isclose(left, right, rtol=PARITY_RTOL, atol=PARITY_ATOL)
            if not np.all(passed):
                index = tuple(int(value) for value in np.argwhere(~passed)[0])
                failures.append({
                    "assay": assay_name,
                    "field": field,
                    "first_failed_index": list(index),
                    "cpu": float(left[index]),
                    "gpu": float(right[index]),
                    "absolute_difference": float(absolute[index]),
                })
    return {
        "rtol": PARITY_RTOL,
        "atol": PARITY_ATOL,
        "compared_points": compared_points,
        "maximum_absolute_difference": maximum_absolute,
        "maximum_relative_difference": maximum_relative,
        "failure_count": len(failures),
        "failures": failures,
        "pointwise_passed": not failures,
    }


def analyze(
    spec_path: str | Path, spec_sha256: str, *, repository_root: str | Path = ROOT
) -> dict[str, Any]:
    root = Path(repository_root).expanduser().resolve(strict=True)
    supplied = Path(spec_path).expanduser()
    path = (supplied if supplied.is_absolute() else root / supplied).resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise SourceModelTransferAnalysisError("execution spec escapes repository") from exc
    _, path = _inside_file(root, relative, "execution spec")
    observed_spec_sha = _file_digest(path)
    _require(observed_spec_sha == _sha256(spec_sha256, "execution spec sha256"), "execution spec digest mismatch")
    spec = _load_json(path, "execution spec")
    spec_binding = {"path": relative, "sha256": observed_spec_sha}
    _require(spec.get("schema") == SPEC_SCHEMA, "execution spec schema mismatch")
    _require(spec.get("status") == "preregistered_not_executed", "execution spec is not prospective")
    _require(spec.get("scientific_verdict") is None, "execution spec contains a verdict")
    _require(spec.get("candidate_calibration_allowed") is False, "calibration is not forbidden")
    _require(spec.get("hybridization_allowed") is False, "hybridization is not forbidden")
    _require(spec.get("stage2_integration_allowed") is False, "Stage 2 integration is not forbidden")
    _require(isinstance(spec.get("analysis_output"), str), "analysis output is not sealed")
    _require(isinstance(spec.get("consumption_ledger"), str), "consumption ledger is not sealed")

    contract = spec.get("analysis_contract")
    _require(isinstance(contract, Mapping), "analysis contract is invalid")
    _require(contract.get("compensation_allowed") is False, "analysis permits compensation")
    _require(contract.get("stage2_verdict_allowed") is False, "analysis permits a Stage 2 verdict")
    parity_contract = contract.get("cpu_gpu_parity")
    _require(
        parity_contract == {"rtol": PARITY_RTOL, "atol": PARITY_ATOL, "pointwise": True},
        "CPU/GPU parity contract changed",
    )
    estimator_binding, estimator_path = _load_binding(
        root, contract.get("estimator_authority"), "estimator authority"
    )
    _require(
        estimator_path == (root / "tools/v14_stageB_fast_channel_clamp_analysis.py").resolve(),
        "estimator authority path changed",
    )
    analyzer_binding, _ = _load_binding(
        root, contract.get("analyzer_authority"), "analyzer authority"
    )
    _require(
        analyzer_binding["path"] == "tools/v14_stageB_source_model_transfer_analysis.py"
        and analyzer_binding["sha256"] == _file_digest(Path(__file__).resolve()),
        "loaded analyzer differs from authority",
    )

    command_binding, commands = _load_bound_json(root, spec.get("command_authority"), "command authority")
    _require(commands.get("schema") == "v14-snr-stageB-fast-channel-clamp-execution-v1", "command authority schema mismatch")
    _require(commands.get("numeric_contract", {}).get("clamp_sample_interval_ms") == 0.005, "command sample interval changed")
    _require(set(commands.get("commands", {})) == {
        "fast_na_activation", "fast_na_inactivation", "fast_na_recovery",
        "fast_na_deactivation", "kv3_activation", "kv3_inactivation",
        "kv3_deactivation",
    }, "command authority assay set changed")
    spec["_commands"] = commands
    _require(command_binding == spec["command_authority"], "command authority binding changed")
    target_binding, target_document = _load_bound_json(root, spec.get("target_authority"), "target authority")
    targets = _target_map(target_document)

    implementation = spec.get("implementation")
    _require(isinstance(implementation, Mapping) and set(implementation) == {"sodium_models", "kv3_models", "runner"}, "implementation binding set changed")
    implementation_hashes: dict[str, str] = {}
    for label, binding in implementation.items():
        authenticated, _ = _load_binding(root, binding, label)
        _require(authenticated == binding, f"{label} implementation binding changed")
        implementation_hashes[label] = authenticated["sha256"]

    models = _models(spec)
    matrix = spec.get("execution_matrix")
    _require(isinstance(matrix, list) and len(matrix) == 12, "execution matrix must contain 12 jobs")
    identities: set[tuple[str, str, str]] = set()
    outputs: set[str] = set()
    documents: dict[tuple[str, str, str], dict[str, Any]] = {}
    input_bindings: dict[str, Any] = {}
    for row in matrix:
        _require(isinstance(row, Mapping), "execution row is invalid")
        model_id, condition_id, backend = row.get("model_id"), row.get("condition_id"), row.get("backend")
        _require(model_id in models and backend in {"numpy", "cupy"}, "execution identity is invalid")
        condition_ids = {item["condition_id"] for item in models[model_id]["conditions"]}
        _require(condition_id in condition_ids, "execution condition is not sealed")
        identity = (model_id, condition_id, backend)
        _require(identity not in identities, "execution matrix duplicates a job")
        _require(row.get("output") not in outputs, "execution matrix duplicates an output")
        identities.add(identity)
        outputs.add(row["output"])
        binding, document = _authenticate_job(root, spec, spec_binding, models[model_id], row)
        documents[identity] = document
        input_bindings["/".join(identity)] = binding
    expected_conditions = {
        (model_id, condition["condition_id"])
        for model_id, model in models.items()
        for condition in model["conditions"]
    }
    _require(identities == {pair + (backend,) for pair in expected_conditions for backend in ("numpy", "cupy")}, "execution matrix coverage changed")

    condition_results: list[dict[str, Any]] = []
    for model_id, condition_id in sorted(expected_conditions):
        cpu = documents[(model_id, condition_id, "numpy")]
        gpu = documents[(model_id, condition_id, "cupy")]
        cpu_metrics = _condition_metrics(cpu)
        gpu_metrics = _condition_metrics(gpu)
        metric_names = SODIUM_METRICS if model_id in {KHALIQ, BALBI} else LABRO_METRICS if model_id == LABRO else DESAI_METRICS
        _require(set(cpu_metrics) == set(metric_names) and set(gpu_metrics) == set(metric_names), f"{model_id}/{condition_id} metric set changed")
        gates = []
        for name in metric_names:
            target = targets[name]
            lower = float(target["mean"] - 2.0 * target["sem"])
            upper = float(target["mean"] + 2.0 * target["sem"])
            value = float(cpu_metrics[name])
            gates.append({
                "metric": name,
                "value": value,
                "source_mean": float(target["mean"]),
                "source_sem": float(target["sem"]),
                "lower_2sem": lower,
                "upper_2sem": upper,
                "passed": lower <= value <= upper,
            })
        parity = _pointwise_parity(cpu, gpu)
        cpu_decisions = {row["metric"]: row["passed"] for row in gates}
        gpu_decisions = {
            name: float(targets[name]["mean"] - 2.0 * targets[name]["sem"])
            <= float(gpu_metrics[name])
            <= float(targets[name]["mean"] + 2.0 * targets[name]["sem"])
            for name in metric_names
        }
        parity["same_metric_gate_decisions"] = cpu_decisions == gpu_decisions
        parity["metric_absolute_differences"] = {
            name: abs(float(cpu_metrics[name]) - float(gpu_metrics[name])) for name in metric_names
        }
        parity["passed"] = parity["pointwise_passed"] and parity["same_metric_gate_decisions"]
        gates_passed = all(row["passed"] for row in gates)
        candidate = gates_passed and parity["passed"]
        condition_results.append({
            "model_id": model_id,
            "condition_id": condition_id,
            "condition": cpu["condition"],
            "cpu_authority_metrics": cpu_metrics,
            "gpu_parity_metrics": gpu_metrics,
            "gates": gates,
            "failed_gate_count": sum(not row["passed"] for row in gates),
            "aggregate_compensation_allowed": False,
            "backend_parity": parity,
            "inactivation_assay": "STRUCTURALLY_UNAVAILABLE" if model_id == LABRO else "AVAILABLE_AND_SCORED",
            "source_transfer_candidate": candidate,
            "verdict": "SOURCE_TRANSFER_GO" if candidate else ("INCONCLUSIVE_INVALID_EVIDENCE" if not parity["passed"] else "SOURCE_TRANSFER_NO_GO"),
        })

    labro_rows = [row for row in condition_results if row["model_id"] == LABRO]
    endpoint_coverage = {
        name: {
            "passing_conditions": [
                row["condition_id"]
                for row in labro_rows
                if next(gate for gate in row["gates"] if gate["metric"] == name)["passed"]
            ],
        }
        for name in LABRO_METRICS
    }
    labro_envelope = {
        "temperatures_c": [float(row["condition"]["temperature_c"]) for row in labro_rows],
        "condition_verdicts": {row["condition_id"]: row["verdict"] for row in labro_rows},
        "metric_min_max": {
            name: {
                "minimum": min(float(row["cpu_authority_metrics"][name]) for row in labro_rows),
                "maximum": max(float(row["cpu_authority_metrics"][name]) for row in labro_rows),
            }
            for name in LABRO_METRICS
        },
        "post_hoc_temperature_selection_allowed": False,
        "available_endpoint_coverage": endpoint_coverage,
        "family_candidate_policy": "each available endpoint passes somewhere in the sealed envelope; no temperature is selected",
        "family_candidate": all(
            endpoint["passing_conditions"] for endpoint in endpoint_coverage.values()
        ) and all(row["backend_parity"]["passed"] for row in labro_rows),
        "inactivation": "STRUCTURALLY_UNAVAILABLE_NOT_SCORED",
    }
    candidate_families = [
        {"model_id": row["model_id"], "condition_id": row["condition_id"]}
        for row in condition_results
        if row["model_id"] != LABRO and row["source_transfer_candidate"]
    ]
    if labro_envelope["family_candidate"]:
        candidate_families.append({"model_id": LABRO, "condition_id": "full_20_22.5_25_c_envelope"})
    output: dict[str, Any] = {
        "schema": OUTPUT_SCHEMA,
        "execution_spec": spec_binding,
        "command_authority": command_binding,
        "target_authority": target_binding,
        "analysis_contract": dict(contract),
        "estimator_authority": estimator_binding,
        "implementation_hashes": implementation_hashes,
        "model_fingerprints": {model_id: model["model_fingerprint"] for model_id, model in sorted(models.items())},
        "analyzer": analyzer_binding,
        "inputs": input_bindings,
        "conditions": condition_results,
        "labro_temperature_envelope": labro_envelope,
        "candidate_families_requiring_new_integration_preregistration": candidate_families,
        "scientific_verdict": "SOURCE_TRANSFER_CANDIDATES_REQUIRE_NEW_INTEGRATION_PREREGISTRATION" if candidate_families else "NO_SOURCE_TRANSFER_CANDIDATE",
        "stage2_go_issued": False,
        "stage2_integration_allowed": False,
        "aggregate_compensation_allowed": False,
    }
    output["sha256"] = _semantic_digest(output)
    return output


def _exclusive_link(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise SourceModelTransferAnalysisError(f"refusing existing file: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def write_analysis_bundle(
    root: str | Path,
    output_relative: str,
    ledger_relative: str,
    output: Mapping[str, Any],
) -> dict[str, Any]:
    repository = Path(root).resolve(strict=True)
    _require(output.get("sha256") == _semantic_digest(output), "analysis output self digest is invalid")
    output_path = repository / output_relative
    ledger_path = repository / ledger_relative
    for relative, path, label in (
        (output_relative, output_path, "analysis output"),
        (ledger_relative, ledger_path, "consumption ledger"),
    ):
        pure = PurePosixPath(relative)
        _require(not pure.is_absolute() and all(part not in {"", ".", ".."} for part in pure.parts), f"{label} path is invalid")
        _require(path.resolve().is_relative_to(repository), f"{label} escapes repository")
        _require(not os.path.lexists(path), f"refusing existing {label}: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    analyzer = output["analyzer"]
    ledger: dict[str, Any] = {
        "schema": LEDGER_SCHEMA,
        "execution_spec": output["execution_spec"],
        "model_fingerprints": output["model_fingerprints"],
        "analysis_output": {"path": output_relative, "sha256": output["sha256"]},
        "analyzer": analyzer,
        "consumption": "one_shot_complete",
    }
    ledger["sha256"] = _semantic_digest(ledger)
    output_payload = (json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("ascii")
    ledger_payload = (json.dumps(ledger, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("ascii")
    created_output = False
    try:
        _exclusive_link(output_path, output_payload)
        created_output = True
        _exclusive_link(ledger_path, ledger_payload)
    except Exception:
        if created_output:
            output_path.unlink(missing_ok=True)
        raise
    return ledger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ledger", required=True)
    arguments = parser.parse_args(argv)
    result = analyze(arguments.spec, arguments.spec_sha256)
    spec_path = ROOT / result["execution_spec"]["path"]
    spec = _load_json(spec_path, "execution spec")
    for supplied, expected, label in (
        (arguments.out, spec["analysis_output"], "analysis output"),
        (arguments.ledger, spec["consumption_ledger"], "consumption ledger"),
    ):
        resolved = (Path(supplied) if Path(supplied).is_absolute() else ROOT / supplied).resolve()
        _require(resolved == (ROOT / expected).resolve(), f"{label} path differs from seal")
    ledger = write_analysis_bundle(ROOT, spec["analysis_output"], spec["consumption_ledger"], result)
    print(json.dumps({"output": spec["analysis_output"], "sha256": result["sha256"], "ledger": spec["consumption_ledger"], "ledger_sha256": ledger["sha256"], "verdict": result["scientific_verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
