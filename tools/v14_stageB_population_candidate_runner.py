"""Candidate-batched predictions for sealed Stage B population targets."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sim import kv3_source_models as kv3
from sim import sodium_source_models as sodium
from sim import source_model_candidate_batch as batch
from tools import v14_stageB_batch_curve_metrics as curve_metrics
from tools import v14_stageB_population_scorer as scorer
from tools import v14_stageB_population_targets as target_authority


ROOT = Path(__file__).resolve().parents[1]
COMMAND_SCHEMA = "v14-snr-stageB-fast-channel-clamp-execution-v1"


class PopulationCandidateRunnerError(ValueError):
    """Raised when a candidate run leaves its sealed calibration boundary."""


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise PopulationCandidateRunnerError(message)


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_commands(
    path: str | Path,
    expected_sha256: str,
    *,
    repository_root: str | Path = ROOT,
) -> dict[str, Any]:
    """Load the frozen command authority by exact file digest."""

    root = Path(repository_root).expanduser().resolve(strict=True)
    supplied = Path(path)
    candidate = (supplied if supplied.is_absolute() else root / supplied).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PopulationCandidateRunnerError("command authority escapes repository") from exc
    _fail(candidate.is_file() and not candidate.is_symlink(), "command authority is unavailable")
    _fail(_file_digest(candidate) == expected_sha256, "command authority digest mismatch")
    try:
        document = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PopulationCandidateRunnerError("command authority is not valid JSON") from exc
    _fail(document.get("schema") == COMMAND_SCHEMA, "command authority schema changed")
    _fail(
        document.get("numeric_contract", {}).get("clamp_sample_interval_ms") == 0.005,
        "command sample interval changed",
    )
    _fail(
        set(document.get("commands", {}))
        == {
            "fast_na_activation", "fast_na_inactivation", "fast_na_recovery",
            "fast_na_deactivation", "kv3_activation", "kv3_inactivation",
            "kv3_deactivation",
        },
        "command assay set changed",
    )
    return document


def _packet(packet: Any) -> tuple[str, list[dict[str, Any]]]:
    _fail(isinstance(packet, Mapping), "target packet must be an object")
    _fail(packet.get("schema") == target_authority.PACKET_SCHEMA, "target packet schema changed")
    _fail(packet.get("partition") == "calibration", "runner accepts calibration only")
    _fail(packet.get("proposal_visible") is True, "calibration packet is not proposal-visible")
    _fail(packet.get("optimization_allowed") is False, "target packet cannot authorize optimization")
    supplied_sha = packet.get("sha256")
    _fail(
        isinstance(supplied_sha, str)
        and supplied_sha
        == target_authority._digest({key: value for key, value in packet.items() if key != "sha256"}),
        "target packet self digest is invalid",
    )
    rows = packet.get("targets")
    _fail(isinstance(rows, list), "target packet targets must be a list")
    available: list[dict[str, Any]] = []
    for row in rows:
        _fail(isinstance(row, Mapping), "target packet contains an invalid target")
        _fail(row.get("partition") == "calibration", "target packet contains withheld data")
        if row.get("status") == "available":
            _fail(
                isinstance(row.get("target_id"), str)
                and isinstance(row.get("target_family"), str)
                and isinstance(row.get("x"), Mapping)
                and isinstance(row["x"].get("median"), (int, float))
                and math.isfinite(float(row["x"]["median"])),
                "available target is incomplete",
            )
            available.append(dict(row))
        else:
            _fail(row.get("status") == "unavailable", "target availability is invalid")
    _fail(
        [row["target_id"] for row in available]
        == sorted(row["target_id"] for row in available),
        "available target ids must remain sorted",
    )
    return supplied_sha, available


def _times(duration_ms: float, interval_ms: float, xp: Any) -> Any:
    count = round(float(duration_ms) / interval_ms)
    _fail(
        count > 0 and math.isclose(count * interval_ms, float(duration_ms), abs_tol=1e-9),
        "command duration is not an integral sample count",
    )
    return xp.arange(1, count + 1, dtype=xp.float64) * xp.float64(interval_ms)


def _candidate_count(parameters: Any, label: str) -> int:
    _fail(
        isinstance(parameters, Sequence)
        and not isinstance(parameters, (str, bytes, Mapping))
        and len(parameters) > 0,
        f"{label} parameters must be a nonempty sequence",
    )
    return len(parameters)


def _target_x(rows: Sequence[Mapping[str, Any]], family: str, xp: Any) -> Any:
    return xp.asarray(
        [float(row["x"]["median"]) for row in rows if row["target_family"] == family],
        dtype=xp.float64,
    )


def _families(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["target_family"], []).append(dict(row))
    return grouped


def _peak_open(
    model_id: str,
    parameters: Sequence[Mapping[str, Any]],
    temperature_c: Any,
    voltage: Any,
    states: Any,
    duration_ms: float,
    interval_ms: float,
    xp: Any,
) -> Any:
    traced = batch.trace_batch(
        model_id, parameters, voltage, states,
        _times(duration_ms, interval_ms, xp), temperature_c, xp,
    )
    return xp.max(batch.open_probability_batch(model_id, traced, xp), axis=-1)


def _normalized_target_values(full_values: Any, target_values: Any, xp: Any) -> Any:
    scale = xp.max(xp.abs(full_values), axis=1, keepdims=True)
    if bool(xp.any(~xp.isfinite(scale))) or bool(xp.any(scale <= 0.0)):
        raise PopulationCandidateRunnerError("full command ladder has no finite normalization scale")
    return target_values / scale


def _sodium_predictions(
    model_id: str,
    parameters: Sequence[Mapping[str, Any]],
    temperature_c: Any,
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    commands: Mapping[str, Any],
    interval_ms: float,
    xp: Any,
) -> dict[str, Any]:
    _fail(model_id in sodium.MODEL_METADATA, "sodium model id is invalid")
    result: dict[str, Any] = {}

    family = "fast_na_activation"
    if family in grouped:
        command = commands[family]
        full_x = xp.asarray(command["test_mV"], dtype=xp.float64)
        target_x = _target_x(grouped[family], family, xp)
        all_x = xp.concatenate((full_x, target_x))
        states = batch.equilibrium_batch(
            model_id, parameters, xp.full_like(all_x, command["hold_mV"]), temperature_c, xp
        )
        peak = _peak_open(
            model_id, parameters, temperature_c, all_x, states,
            command["test_duration_ms"], interval_ms, xp,
        )
        result[family] = _normalized_target_values(
            peak[:, : full_x.size], peak[:, full_x.size :], xp
        )

    family = "fast_na_steady_state_inactivation"
    if family in grouped:
        command = commands["fast_na_inactivation"]
        full_x = xp.asarray(command["prepulse_mV"], dtype=xp.float64)
        target_x = _target_x(grouped[family], family, xp)
        all_x = xp.concatenate((full_x, target_x))
        states = batch.equilibrium_batch(
            model_id, parameters, xp.full_like(all_x, command["hold_mV"]), temperature_c, xp
        )
        states = batch.advance_batch(
            model_id, parameters, all_x, states, command["prepulse_duration_ms"], temperature_c, xp
        )
        test_voltage = xp.full_like(all_x, command["test_mV"])
        peak = _peak_open(
            model_id, parameters, temperature_c, test_voltage, states,
            command["test_duration_ms"], interval_ms, xp,
        )
        result[family] = _normalized_target_values(
            peak[:, : full_x.size], peak[:, full_x.size :], xp
        )

    family = "fast_na_recovery"
    if family in grouped:
        command = commands[family]
        full_x = xp.asarray(command["recovery_duration_ms"], dtype=xp.float64)
        target_x = _target_x(grouped[family], family, xp)
        all_x = xp.concatenate((full_x, target_x))
        states = batch.equilibrium_batch(
            model_id, parameters, xp.full_like(all_x, command["hold_mV"]), temperature_c, xp
        )
        for voltage_key, duration_key in (
            ("recovery_prepulse_mV", "recovery_prepulse_duration_ms"),
            ("inactivation_mV", "inactivation_duration_ms"),
        ):
            states = batch.advance_batch(
                model_id, parameters, xp.full_like(all_x, command[voltage_key]), states,
                command[duration_key], temperature_c, xp,
            )
        states = batch.advance_batch(
            model_id, parameters, xp.full_like(all_x, command["recovery_mV"]), states,
            all_x, temperature_c, xp,
        )
        peak = _peak_open(
            model_id, parameters, temperature_c,
            xp.full_like(all_x, command["test_mV"]), states,
            command["test_duration_ms"], interval_ms, xp,
        )
        result[family] = _normalized_target_values(
            peak[:, : full_x.size], peak[:, full_x.size :], xp
        )

    family = "fast_na_deactivation"
    if family in grouped:
        command = commands[family]
        target_x = _target_x(grouped[family], family, xp)
        states = batch.equilibrium_batch(
            model_id, parameters, xp.full_like(target_x, command["hold_mV"]), temperature_c, xp
        )
        for voltage_key, duration_key in (
            ("prepulse_mV", "prepulse_duration_ms"),
            ("activation_mV", "activation_duration_ms"),
        ):
            states = batch.advance_batch(
                model_id, parameters, xp.full_like(target_x, command[voltage_key]), states,
                command[duration_key], temperature_c, xp,
            )
        elapsed = _times(command["test_duration_ms"], interval_ms, xp)
        traced = batch.trace_batch(
            model_id, parameters, target_x, states, elapsed, temperature_c, xp
        )
        result[family] = curve_metrics.fit_deactivation_tails(
            elapsed, batch.open_probability_batch(model_id, traced, xp), xp
        )
    return result


def _kv3_predictions(
    model_id: str,
    parameters: Sequence[Mapping[str, Any]],
    temperature_c: Any,
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    commands: Mapping[str, Any],
    interval_ms: float,
    xp: Any,
) -> dict[str, Any]:
    _fail(model_id in kv3.MODEL_METADATA, "Kv3 model id is invalid")
    result: dict[str, Any] = {}

    family = "kv3_activation"
    if family in grouped:
        command = commands[family]
        full_x = xp.asarray(command["test_mV"], dtype=xp.float64)
        target_x = _target_x(grouped[family], family, xp)
        all_x = xp.concatenate((full_x, target_x))
        states = batch.equilibrium_batch(
            model_id, parameters, xp.full_like(all_x, command["hold_mV"]), temperature_c, xp
        )
        peak = _peak_open(
            model_id, parameters, temperature_c, all_x, states,
            command["test_duration_ms"], interval_ms, xp,
        )
        result[family] = _normalized_target_values(
            peak[:, : full_x.size], peak[:, full_x.size :], xp
        )

    family = "kv3_steady_state_inactivation"
    if family in grouped:
        _fail(
            kv3.MODEL_METADATA[model_id]["has_inactivation_state"] is True,
            f"{model_id} cannot predict Kv3 inactivation without changing its source graph",
        )
        command = commands["kv3_inactivation"]
        full_x = xp.asarray(command["prepulse_mV"], dtype=xp.float64)
        target_x = _target_x(grouped[family], family, xp)
        all_x = xp.concatenate((full_x, target_x))
        states = batch.equilibrium_batch(
            model_id, parameters, xp.full_like(all_x, command["hold_mV"]), temperature_c, xp
        )
        states = batch.advance_batch(
            model_id, parameters, all_x, states, command["prepulse_duration_ms"], temperature_c, xp
        )
        peak = _peak_open(
            model_id, parameters, temperature_c,
            xp.full_like(all_x, command["test_mV"]), states,
            command["test_duration_ms"], interval_ms, xp,
        )
        result[family] = _normalized_target_values(
            peak[:, : full_x.size], peak[:, full_x.size :], xp
        )

    family = "kv3_deactivation"
    if family in grouped:
        command = commands[family]
        target_x = _target_x(grouped[family], family, xp)
        states = batch.equilibrium_batch(
            model_id, parameters, xp.full_like(target_x, command["hold_mV"]), temperature_c, xp
        )
        states = batch.advance_batch(
            model_id, parameters, xp.full_like(target_x, command["activation_mV"]), states,
            command["activation_duration_ms"], temperature_c, xp,
        )
        elapsed = _times(command["test_duration_ms"], interval_ms, xp)
        traced = batch.trace_batch(
            model_id, parameters, target_x, states, elapsed, temperature_c, xp
        )
        result[family] = curve_metrics.fit_deactivation_tails(
            elapsed, batch.open_probability_batch(model_id, traced, xp), xp
        )
    return result


def predict_population_targets(
    target_packet: Mapping[str, Any],
    *,
    sodium_model_id: str,
    sodium_parameters: Sequence[Mapping[str, Any]],
    sodium_temperature_c: Any,
    kv3_model_id: str,
    kv3_parameters: Sequence[Mapping[str, Any]],
    kv3_temperature_c: Any,
    command_authority: Mapping[str, Any],
    xp: Any,
) -> tuple[list[str], Any]:
    """Predict every available calibration target for paired candidate rows."""

    _, rows = _packet(target_packet)
    sodium_count = _candidate_count(sodium_parameters, "sodium")
    _fail(
        _candidate_count(kv3_parameters, "Kv3") == sodium_count,
        "sodium and Kv3 candidate counts differ",
    )
    _fail(command_authority.get("schema") == COMMAND_SCHEMA, "command authority schema changed")
    interval = command_authority.get("numeric_contract", {}).get("clamp_sample_interval_ms")
    _fail(interval == 0.005, "command sample interval changed")
    commands = command_authority.get("commands")
    _fail(isinstance(commands, Mapping), "command authority lacks commands")
    grouped = _families(rows)
    allowed = {
        "fast_na_activation", "fast_na_steady_state_inactivation", "fast_na_recovery",
        "fast_na_deactivation", "kv3_activation", "kv3_steady_state_inactivation",
        "kv3_deactivation",
    }
    _fail(set(grouped) <= allowed, "target packet contains an unsupported family")
    predictions = {
        **_sodium_predictions(
            sodium_model_id, sodium_parameters, sodium_temperature_c,
            grouped, commands, interval, xp,
        ),
        **_kv3_predictions(
            kv3_model_id, kv3_parameters, kv3_temperature_c,
            grouped, commands, interval, xp,
        ),
    }
    ids: list[str] = []
    columns: list[Any] = []
    family_offsets: dict[str, int] = {}
    for row in rows:
        family = row["target_family"]
        offset = family_offsets.get(family, 0)
        _fail(family in predictions, f"target family {family} was not predicted")
        ids.append(row["target_id"])
        columns.append(predictions[family][:, offset])
        family_offsets[family] = offset + 1
    matrix = xp.stack(columns, axis=1) if columns else xp.empty((sodium_count, 0), dtype=xp.float64)
    if bool(xp.any(~xp.isfinite(matrix))):
        raise PopulationCandidateRunnerError("candidate prediction contains non-finite values")
    return ids, matrix


def build_candidate_observations(
    target_packet: Mapping[str, Any],
    candidate_bindings: Sequence[Mapping[str, str]],
    target_ids: Sequence[str],
    predictions: Any,
) -> list[dict[str, Any]]:
    """Build scorer-compatible analysis-only observations on the host."""

    packet_sha, rows = _packet(target_packet)
    expected_ids = [row["target_id"] for row in rows]
    _fail(list(target_ids) == expected_ids, "prediction target ids differ from packet")
    matrix = np.asarray(predictions, dtype=np.float64)
    _fail(
        matrix.shape == (len(candidate_bindings), len(expected_ids))
        and np.all(np.isfinite(matrix)),
        "prediction matrix shape or values are invalid",
    )
    x_by_id = {row["target_id"]: float(row["x"]["median"]) for row in rows}
    output: list[dict[str, Any]] = []
    for index, binding in enumerate(candidate_bindings):
        _fail(
            isinstance(binding, Mapping)
            and set(binding) == {"id", "sha256"}
            and isinstance(binding["id"], str)
            and isinstance(binding["sha256"], str)
            and len(binding["sha256"]) == 64,
            "candidate binding is invalid",
        )
        core = {
            "schema": scorer.OBSERVATION_SCHEMA,
            "status": "completed",
            "target_packet": {"sha256": packet_sha},
            "candidate": dict(binding),
            "predictions": [
                {"target_id": target_id, "x": x_by_id[target_id], "y": float(matrix[index, column])}
                for column, target_id in enumerate(expected_ids)
            ],
            "scientific_verdict": None,
            "optimization_allowed": False,
        }
        output.append({**core, "sha256": scorer.digest(core)})
    return output

