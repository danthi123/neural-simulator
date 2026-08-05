"""Source-bound raw-trace scorer for V14 SNr Stage B."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from tools.compact_trace import CompactTraceError, load_compact_trace
from tools.v14_stageB_physiology_metrics import (
    ahp_depth,
    interspike_voltage_nadirs,
    peak_conductance,
    spike_train_metrics,
)
from tools.v14_stageB_scorer_fixtures import StageBFixtureError, score_observation, validate_fixture


SCHEMA = "v14-snr-stageB-raw-observations-v1"
RESULT_SCHEMA = "v14-snr-stageB-score-v1"
INTRINSIC_LESION_SCHEMA = "v14-snr-stageB-intrinsic-lesion-observations-v1"
INTRINSIC_LESION_RESULT_SCHEMA = "v14-snr-stageB-intrinsic-lesion-score-v1"

_INTRINSIC_LESION_IDS = (
    "nap-complete-lesion",
    "cav2.2-complete-lesion",
    "sk-complete-lesion",
    "hcn-complete-lesion",
)
_INTRINSIC_ARMS = {
    "intact_autonomous": None,
    "nap_lesion": ("nap", "cp_snr_g_nap_max"),
    "cav2_2_lesion": ("cav2.2", "cp_snr_g_ca_max"),
    "sk_lesion": ("sk", "cp_snr_g_sk_max"),
    "hcn_baseline_lesion": ("hcn", "cp_snr_g_h_max"),
}
_GATE_ARMS = {
    "nap-complete-lesion": "nap_lesion",
    "cav2.2-complete-lesion": "cav2_2_lesion",
    "sk-complete-lesion": "sk_lesion",
    "hcn-complete-lesion": "hcn_baseline_lesion",
}


class StageBScorerError(ValueError):
    """Raised when raw observations cannot support a Stage B score."""


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_bound_json(root: Path, declaration: Any, context: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(declaration, Mapping) or set(declaration) != {"path", "sha256"}:
        raise StageBScorerError(f"{context} must declare only path and sha256")
    relative = PurePosixPath(str(declaration.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts or not relative.name:
        raise StageBScorerError(f"{context} path must be repository-relative")
    path = root.joinpath(*relative.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise StageBScorerError(f"{context} path escapes the repository") from exc
    try:
        data = path.read_bytes()
        value = json.loads(data)
    except (OSError, json.JSONDecodeError) as exc:
        raise StageBScorerError(f"cannot load {context}: {exc}") from exc
    if _digest_bytes(data) != declaration.get("sha256"):
        raise StageBScorerError(f"{context} digest does not match")
    if not isinstance(value, dict):
        raise StageBScorerError(f"{context} must contain a JSON object")
    return path, value


def _candidate_echo(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "candidate_id", "candidate_sha256", "effective_parameters"
    }:
        raise StageBScorerError("adaptive_candidate has an invalid shape")
    identifier = value.get("candidate_id")
    digest = value.get("candidate_sha256")
    parameters = value.get("effective_parameters")
    if (not isinstance(identifier, str) or not identifier or not isinstance(digest, str)
            or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(parameters, Mapping)):
        raise StageBScorerError("adaptive_candidate is malformed")
    return {"candidate_id": identifier, "candidate_sha256": digest,
            "effective_parameters": dict(parameters)}


def _contains_seed(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any("seed" in str(key).lower() or _contains_seed(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_seed(item) for item in value)
    return False


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StageBScorerError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise StageBScorerError(f"{context} must be a finite number")
    return result


def _validate_runner_intervention(arm: str, value: Any) -> dict[str, Any]:
    expected_keys = {
        "kind", "operation", "target", "runtime_conductance_field",
        "conductance_density_unit", "before", "after",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise StageBScorerError(f"runner artifact {arm!r} has an invalid intervention shape")
    intervention = dict(value)
    lesion = _INTRINSIC_ARMS[arm]
    if lesion is None:
        expected = {
            "kind": "none", "operation": "authenticated_packet_intact", "target": None,
            "runtime_conductance_field": None, "conductance_density_unit": "mS/cm^2",
            "before": None, "after": None,
        }
        if intervention != expected:
            raise StageBScorerError("intact runner artifact carries an intervention")
        return intervention

    target, field = lesion
    fixed = {
        "kind": "complete_intrinsic_current_lesion",
        "operation": "set_conductance_density_to_zero_after_authenticated_packet_initialization",
        "target": target,
        "runtime_conductance_field": field,
        "conductance_density_unit": "mS/cm^2",
    }
    if any(intervention.get(key) != expected for key, expected in fixed.items()):
        raise StageBScorerError(f"runner artifact {arm!r} does not carry the filed complete lesion")
    before = intervention.get("before")
    after = intervention.get("after")
    if (not isinstance(before, list) or len(before) != 1
            or _finite_number(before[0], f"{arm}.intervention.before") < 0.0
            or after != [0.0]):
        raise StageBScorerError(f"runner artifact {arm!r} did not prove zero conductance")
    return intervention


def _load_compact_trace_vectors(
    root: Path, declaration: Any, arm: str,
) -> tuple[list[float], list[float], list[bool]]:
    if not isinstance(declaration, Mapping) or set(declaration) != {"path", "sha256"}:
        raise StageBScorerError(f"runner artifact {arm!r} compact trace must bind only path and sha256")
    relative = PurePosixPath(str(declaration.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts or not relative.name:
        raise StageBScorerError(f"runner artifact {arm!r} compact trace path must be repository-relative")
    archive = root.joinpath(*relative.parts)
    if archive.is_symlink() or not archive.is_file():
        raise StageBScorerError(f"runner artifact {arm!r} compact trace must be a regular file")
    resolved = archive.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise StageBScorerError(
            f"runner artifact {arm!r} compact trace path escapes the repository"
        ) from exc
    try:
        arrays = load_compact_trace(resolved, expected_sha256=declaration["sha256"])
    except (CompactTraceError, OSError, TypeError, ValueError) as exc:
        raise StageBScorerError(f"runner artifact {arm!r} compact trace is invalid: {exc}") from exc
    return (
        [float(value) for value in arrays["time"]],
        [float(value) for value in arrays["voltage"]],
        [bool(value) for value in arrays["spikes"]],
    )


def _trace_vectors(
    raw: Any, arm: str, *, root: Path,
) -> tuple[list[float], list[float], list[float], dict[str, Any] | None]:
    base_fields = {
        "kind", "time_unit", "voltage_unit", "sample_interval_s", "recording_start_s",
        "recording_end_s", "uncropped", "sample_semantics",
    }
    raw_fields = set(raw) if isinstance(raw, Mapping) else set()
    inline_fields = base_fields | {"time_s", "voltage_mV", "spike_states"}
    compact_fields = base_fields | {"compact_trace"}
    allowed_shapes = {
        frozenset(inline_fields),
        frozenset(inline_fields | {"analysis_protocol"}),
        frozenset(compact_fields),
        frozenset(compact_fields | {"analysis_protocol"}),
    }
    if not isinstance(raw, Mapping) or frozenset(raw_fields) not in allowed_shapes:
        raise StageBScorerError(f"runner artifact {arm!r} raw trace has an invalid shape")
    if (raw.get("kind") != "packet_voltage_spike_trace" or raw.get("time_unit") != "s"
            or raw.get("voltage_unit") != "mV" or raw.get("uncropped") is not True
            or raw.get("sample_semantics") != "post-update state at the declared time"):
        raise StageBScorerError(f"runner artifact {arm!r} changed the raw trace contract")
    dt = _finite_number(raw.get("sample_interval_s"), f"{arm}.sample_interval_s")
    start = _finite_number(raw.get("recording_start_s"), f"{arm}.recording_start_s")
    end = _finite_number(raw.get("recording_end_s"), f"{arm}.recording_end_s")
    if dt <= 0.0 or end <= start:
        raise StageBScorerError(f"runner artifact {arm!r} has invalid trace timing")
    compact = "compact_trace" in raw
    if compact:
        times, voltages, spikes = _load_compact_trace_vectors(
            root, raw["compact_trace"], arm
        )
    else:
        times_raw = raw.get("time_s")
        voltages_raw = raw.get("voltage_mV")
        spikes_raw = raw.get("spike_states")
        if (not isinstance(times_raw, list) or not times_raw or not isinstance(voltages_raw, list)
                or not isinstance(spikes_raw, list) or len(times_raw) != len(voltages_raw)
                or len(times_raw) != len(spikes_raw)):
            raise StageBScorerError(f"runner artifact {arm!r} has incomplete trace vectors")
        times = [_finite_number(value, f"{arm}.time_s[]") for value in times_raw]
        voltages = []
        spikes = []
        for voltage_row, spike_row in zip(voltages_raw, spikes_raw, strict=True):
            if not isinstance(voltage_row, list) or len(voltage_row) != 1:
                raise StageBScorerError(f"runner artifact {arm!r} voltage trace is not single-cell")
            if (not isinstance(spike_row, list) or len(spike_row) != 1
                    or not isinstance(spike_row[0], bool)):
                raise StageBScorerError(f"runner artifact {arm!r} spike trace is not single-cell boolean")
            voltages.append(_finite_number(voltage_row[0], f"{arm}.voltage_mV[]"))
            spikes.append(spike_row[0])
    if not times or len(times) != len(voltages) or len(times) != len(spikes):
        raise StageBScorerError(f"runner artifact {arm!r} has incomplete trace vectors")
    times = [_finite_number(value, f"{arm}.time_s[]") for value in times]
    voltages = [_finite_number(value, f"{arm}.voltage_mV[]") for value in voltages]
    spike_times: list[float] = []
    for time, spike in zip(times, spikes, strict=True):
        if not isinstance(spike, bool):
            raise StageBScorerError(f"runner artifact {arm!r} spike trace is not single-cell boolean")
        if spike:
            spike_times.append(time)
    tolerance = max(1e-12, dt * 1e-9)
    if (not math.isclose(times[0], start, abs_tol=tolerance)
            or not math.isclose(times[-1] + dt, end, abs_tol=tolerance)
            or any(not math.isclose(right - left, dt, abs_tol=tolerance)
                   for left, right in zip(times, times[1:]))):
        raise StageBScorerError(f"runner artifact {arm!r} time vector does not match its protocol")
    protocol = raw.get("analysis_protocol")
    if protocol is not None and not isinstance(protocol, Mapping):
        raise StageBScorerError(f"runner artifact {arm!r} analysis_protocol must be an object")
    return times, voltages, spike_times, dict(protocol) if protocol is not None else None


def _load_trace_protocol(
    root: Path, protocol: Mapping[str, Any], arm: str,
    causal_gate_packet: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if set(protocol) != {"binding", "termination"}:
        raise StageBScorerError(f"runner artifact {arm!r} has an invalid analysis protocol")
    path, document = _load_bound_json(root, protocol["binding"], f"{arm} analysis protocol")
    required = {
        "device", "provenance_exempt",
        "schema", "protocol_id", "status", "causal_gate_authority",
        "target_packet", "primary_source", "analysis_conventions", "execution",
        "arms", "scientific_boundaries",
    }
    if (set(document) != required
            or document.get("schema") not in {
                "v14-snr-stageB-intrinsic-protocol-v1",
                "v14-snr-stageB-intrinsic-protocol-v2",
            }
            or document.get("status") != "production-measurement-partial"):
        raise StageBScorerError(f"runner artifact {arm!r} analysis protocol changed schema or status")
    if document.get("analysis_conventions") != {
        "cv_method": "population standard deviation of the 100 complete interspike intervals divided by their mean",
        "cv_method_evidence_class": "project_analysis_convention",
        "frequency_method": "100 divided by the elapsed time from the first through the 101st spike",
        "frequency_method_evidence_class": "project_analysis_convention",
    }:
        raise StageBScorerError(f"runner artifact {arm!r} changed the project analysis formulas")
    gate_path, gate_document = _load_bound_json(root, causal_gate_packet, "causal gate authority")
    authority = document.get("causal_gate_authority")
    if (not isinstance(authority, Mapping)
            or authority.get("path") != gate_path.relative_to(root).as_posix()
            or gate_document.get("authorized_analysis_protocol") != {
                "path": path.relative_to(root).as_posix(),
                "sha256": _digest_bytes(path.read_bytes()),
            }):
        raise StageBScorerError(f"runner artifact {arm!r} protocol is not authorized by the causal gate")
    execution = document.get("execution")
    if execution != {
        "dt_ms": 0.05,
        "dt_status": "project_operational_discretization_requires_timestep_convergence_before_waveform_claims",
        "trace_policy": "uncropped_post_update_voltage_and_spike_state",
    }:
        raise StageBScorerError(f"runner artifact {arm!r} protocol changed execution settings")
    arms = document.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(_INTRINSIC_ARMS):
        raise StageBScorerError("analysis protocol does not define exactly the intrinsic arms")
    arm_protocol = arms.get(arm)
    if not isinstance(arm_protocol, Mapping):
        raise StageBScorerError(f"analysis protocol has no valid arm {arm!r}")
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _digest_bytes(path.read_bytes()),
    }, dict(arm_protocol)


def _event_count_spike_metrics(spike_times: list[float]) -> dict[str, Any]:
    if len(spike_times) != 101:
        raise StageBScorerError("completed event-count trace must contain exactly 101 spikes")
    intervals = [right - left for left, right in zip(spike_times, spike_times[1:])]
    if len(intervals) != 100 or any(interval <= 0.0 for interval in intervals):
        raise StageBScorerError("event-count trace does not contain 100 complete positive ISIs")
    mean_isi = sum(intervals) / len(intervals)
    variance = sum((interval - mean_isi) ** 2 for interval in intervals) / len(intervals)
    return {
        "window_convention": "first-through-101st-spike",
        "spike_count": 101,
        "complete_isi_count": 100,
        "firing_rate_hz": 1.0 / mean_isi,
        "isi_cv": math.sqrt(variance) / mean_isi,
        "isi_cv2": (
            sum(
                2.0 * abs(right - left) / (right + left)
                for left, right in zip(intervals, intervals[1:])
            ) / (len(intervals) - 1)
        ),
    }


def _recompute_trace_metrics(
    raw: Mapping[str, Any], arm: str, *, root: Path,
    causal_gate_packet: Mapping[str, str],
) -> dict[str, Any]:
    times, voltages, spike_times, protocol = _trace_vectors(raw, arm, root=root)
    if protocol is None:
        return {"status": "unavailable", "reason": "runner trace has no sealed analysis protocol"}
    binding, arm_protocol = _load_trace_protocol(root, protocol, arm, causal_gate_packet)
    termination = protocol["termination"]
    termination_fields = {
        "mode", "reason", "steps_executed", "spikes_observed", "target_spike_count",
        "maximum_steps", "timeout_is_physiology_failure",
    }
    if not isinstance(termination, Mapping) or set(termination) != termination_fields:
        raise StageBScorerError(f"runner artifact {arm!r} has invalid termination evidence")
    observed_spikes = len(spike_times)
    if (termination.get("steps_executed") != len(times)
            or termination.get("spikes_observed") != observed_spikes
            or termination.get("timeout_is_physiology_failure") is not False):
        raise StageBScorerError(f"runner artifact {arm!r} termination does not match its trace")
    filed_termination = arm_protocol.get("termination")
    filed_spikes = arm_protocol.get("spike_metrics")
    if not isinstance(filed_termination, Mapping) or not isinstance(filed_spikes, Mapping):
        raise StageBScorerError(f"runner artifact {arm!r} protocol arm is incomplete")

    dt = float(raw["sample_interval_s"])
    if arm == "nap_lesion":
        maximum_steps = int(round(float(filed_termination.get("duration_s", 0.0)) / dt))
        if (filed_termination.get("mode") != "fixed_duration"
                or filed_spikes.get("window_s") != 1.0
                or termination.get("mode") != "fixed_duration"
                or termination.get("reason") != "fixed_duration_complete"
                or termination.get("target_spike_count") is not None
                or termination.get("maximum_steps") != maximum_steps
                or len(times) != maximum_steps):
            raise StageBScorerError("Nap trace does not implement the filed one-second protocol")
        window_start = float(raw["recording_start_s"])
        window_end = float(raw["recording_end_s"])
        spikes = {
            "window_convention": "full-filed-one-second-window",
            "spike_count": len(spike_times),
            "firing_rate_hz": len(spike_times) / (window_end - window_start),
            "isi_cv": None,
            "isi_cv2": None,
        }
    else:
        target = filed_spikes.get("target_spike_count")
        maximum_steps = int(round(
            float(filed_termination.get("maximum_duration_s", 0.0)) / dt
        ))
        if (filed_termination.get("mode") != "event_count_or_timeout"
                or target != 101
                or filed_spikes.get("target_spike_count_evidence_class") != "source_reported"
                or termination.get("mode") != "event_count_or_timeout"
                or termination.get("target_spike_count") != 101
                or termination.get("maximum_steps") != maximum_steps):
            raise StageBScorerError(f"runner artifact {arm!r} changed the 101-spike contract")
        window_start = float(raw["recording_start_s"])
        window_end = float(raw["recording_end_s"])
        if termination.get("reason") == "target_spike_count_reached":
            if len(times) > maximum_steps:
                raise StageBScorerError("event-count trace exceeded its operational maximum")
            spikes = _event_count_spike_metrics(spike_times)
        elif termination.get("reason") == "maximum_duration_reached":
            if len(times) != maximum_steps or observed_spikes >= 101:
                raise StageBScorerError("event-count timeout does not match its trace")
            spikes = {
                "window_convention": "event-count-timeout",
                "spike_count": observed_spikes,
                "complete_isi_count": max(0, observed_spikes - 1),
                "firing_rate_hz": None,
                "isi_cv": None,
                "isi_cv2": None,
                "unavailable_reason": "101-spike source-bound event count was not reached before the operational timeout",
            }
        else:
            raise StageBScorerError(f"runner artifact {arm!r} has an invalid termination reason")
    selected_voltage = [
        voltage for time, voltage in zip(times, voltages, strict=True)
        if window_start <= time < window_end
    ]
    if not selected_voltage:
        raise StageBScorerError(f"runner artifact {arm!r} scoring window contains no voltage samples")
    nadir_contract = arm_protocol.get("interspike_voltage_nadir")
    if nadir_contract is None:
        nadir = {
            "status": "unavailable",
            "median_interspike_voltage_nadir_mV": None,
            "reason": "the filed arm does not require the total-AHP directional assay",
        }
    elif (
        not isinstance(nadir_contract, Mapping)
        or nadir_contract.get("status") != "production-project-analysis-convention"
    ):
        raise StageBScorerError(f"runner artifact {arm!r} has an invalid voltage-nadir contract")
    elif len(spike_times) != 101:
        nadir = {
            "status": "unavailable",
            "median_interspike_voltage_nadir_mV": None,
            "reason": "the 101-spike trace required for 100 complete interspike intervals is unavailable",
        }
    else:
        measured = interspike_voltage_nadirs(
            times,
            voltages,
            spike_times,
            time_unit="s",
            voltage_unit="mV",
            sample_interval_s=dt,
            recording_start_s=float(raw["recording_start_s"]),
            recording_end_s=float(raw["recording_end_s"]),
            burn_in_start_s=float(raw["recording_start_s"]),
            burn_in_end_s=float(spike_times[0]),
        )
        nadir = {
            "status": "recomputed",
            **measured,
            "source_equivalence_claimed": False,
        }
    return {
        "status": "recomputed",
        "spike_metrics": spikes,
        "mean_membrane_voltage_mV": sum(selected_voltage) / len(selected_voltage),
        "medium_ahp": {
            "status": "unavailable",
            "ahp_depth_mV": None,
            "reason": str(arm_protocol.get("medium_ahp", {}).get("reason", "not specified")),
        },
        "interspike_voltage_nadir": nadir,
        "analysis_protocol": {
            "binding": binding,
            "termination": dict(termination),
            "spike_metrics": {
                "window_start_s": window_start,
                "window_end_s": window_end,
            },
        },
    }


def _load_intrinsic_runner_artifacts(
    root: Path, declarations: Any
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]], dict[str, Any]]:
    if not isinstance(declarations, Mapping) or set(declarations) != set(_INTRINSIC_ARMS):
        raise StageBScorerError("runner_observations must bind exactly the five intrinsic readiness arms")
    documents: dict[str, dict[str, Any]] = {}
    bindings: dict[str, dict[str, str]] = {}
    expected_top = {
        "schema", "process_status", "readiness_only", "backend", "device", "arm",
        "runtime_intervention", "adaptive_candidate", "raw_observation", "provenance",
    }
    readiness = {
        "enabled": True,
        "reserved_seed_count": 0,
        "scientific_seed": None,
        "engine_seed": 0,
        "engine_seed_effect": "none; connectivity, heterogeneity, noise, and plasticity are disabled",
    }
    for arm in _INTRINSIC_ARMS:
        path, document = _load_bound_json(root, declarations[arm], f"runner observation {arm}")
        if set(document) != expected_top or document.get("schema") != "v14-snr-stageB-physiology-observation-v1":
            raise StageBScorerError(f"runner artifact {arm!r} has an invalid shape or schema")
        if (document.get("process_status") != "completed" or document.get("backend") != "numpy"
                or document.get("device") != "cpu" or document.get("arm") != arm
                or document.get("readiness_only") != readiness):
            raise StageBScorerError(f"runner artifact {arm!r} is not a completed seed-free readiness trace")
        seed_checked = {key: value for key, value in document.items() if key != "readiness_only"}
        if _contains_seed(seed_checked):
            raise StageBScorerError(f"runner artifact {arm!r} contains seed data outside readiness metadata")
        _validate_runner_intervention(arm, document.get("runtime_intervention"))
        documents[arm] = document
        bindings[arm] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": _digest_bytes(path.read_bytes()),
        }

    intact = documents["intact_autonomous"]
    candidate = _candidate_echo(intact.get("adaptive_candidate"))
    provenance = intact.get("provenance")
    if not isinstance(provenance, Mapping):
        raise StageBScorerError("intact runner artifact has invalid provenance")
    identity = {
        "adaptive_candidate": intact.get("adaptive_candidate"),
        "candidate_release": provenance.get("candidate_release"),
        "bindings": provenance.get("bindings"),
        "raw_protocol": {
            "time_unit": intact["raw_observation"].get("time_unit"),
            "voltage_unit": intact["raw_observation"].get("voltage_unit"),
            "sample_interval_s": intact["raw_observation"].get("sample_interval_s"),
            "recording_start_s": intact["raw_observation"].get("recording_start_s"),
            "uncropped": intact["raw_observation"].get("uncropped"),
            "sample_semantics": intact["raw_observation"].get("sample_semantics"),
            "analysis_protocol_binding": (
                intact["raw_observation"].get("analysis_protocol") or {}
            ).get("binding"),
        },
    }
    for arm, document in documents.items():
        arm_provenance = document.get("provenance")
        if not isinstance(arm_provenance, Mapping):
            raise StageBScorerError(f"runner artifact {arm!r} has invalid provenance")
        comparison = {
            "adaptive_candidate": document.get("adaptive_candidate"),
            "candidate_release": arm_provenance.get("candidate_release"),
            "bindings": arm_provenance.get("bindings"),
            "raw_protocol": {
                "time_unit": document["raw_observation"].get("time_unit"),
                "voltage_unit": document["raw_observation"].get("voltage_unit"),
                "sample_interval_s": document["raw_observation"].get("sample_interval_s"),
                "recording_start_s": document["raw_observation"].get("recording_start_s"),
                "uncropped": document["raw_observation"].get("uncropped"),
                "sample_semantics": document["raw_observation"].get("sample_semantics"),
                "analysis_protocol_binding": (
                    document["raw_observation"].get("analysis_protocol") or {}
                ).get("binding"),
            },
        }
        if comparison != identity:
            raise StageBScorerError(
                f"runner artifact {arm!r} does not match the intact candidate/protocol/release identity"
            )
    return documents, bindings, candidate


def _unavailable_hard_gate(contract: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        "metric": contract["metric"],
        "operator": contract["operator"],
        "evidence_class": contract["evidence_class"],
        "source_equivalence_claimed": False,
        "status": "unavailable",
        "passed": None,
        "reason": reason,
    }


def _score_intrinsic_hard_gate(
    gate_id: str,
    contract: Mapping[str, Any],
    intact: Mapping[str, Any],
    lesion: Mapping[str, Any],
) -> dict[str, Any]:
    required = {"metric", "operator", "evidence_class"}
    if not isinstance(contract, Mapping) or not required.issubset(contract):
        raise StageBScorerError(f"{gate_id} contains a malformed hard gate")
    if set(contract) - {"metric", "operator", "evidence_class", "value", "window_s", "cohort_n"}:
        raise StageBScorerError(f"{gate_id} hard gate contains unsupported fields")
    metric = contract["metric"]
    operator = contract["operator"]
    evidence_class = contract["evidence_class"]
    if not isinstance(metric, str) or not isinstance(operator, str):
        raise StageBScorerError(f"{gate_id} hard gate metric/operator must be text")
    if evidence_class not in {"source_reported_direction", "project_operational"}:
        raise StageBScorerError(f"{gate_id} hard gate has an unsupported evidence boundary")

    result: dict[str, Any] = {
        "metric": metric,
        "operator": operator,
        "evidence_class": evidence_class,
        "source_equivalence_claimed": False,
    }
    if metric == "depolarization_block_count":
        return _unavailable_hard_gate(
            contract, "sealed 12-cell SK cohort depolarization-block traces are not available"
        )
    if metric == "hyperpolarized_input_resistance_MOhm":
        return _unavailable_hard_gate(
            contract, "sealed HCN hyperpolarized intact/lesion current-step traces are not available"
        )
    if gate_id == "nap-complete-lesion" and metric == "mean_membrane_voltage_change_mV":
        return _unavailable_hard_gate(
            contract,
            "the source does not define a matched stable-baseline voltage estimator for this trace",
        )
    if metric == "medium_ahp_depth_mV":
        return _unavailable_hard_gate(
            contract,
            "the source-bound event-aligned medium-AHP measurement window is not specified",
        )
    if gate_id != "nap-complete-lesion" and any(
        trace.get("analysis_protocol", {}).get("termination", {}).get("reason")
        == "maximum_duration_reached"
        for trace in (intact, lesion)
    ):
        return _unavailable_hard_gate(
            contract,
            "the source-bound 101-spike event count was not completed before the operational timeout",
        )
    if intact.get("status") != "recomputed" or lesion.get("status") != "recomputed":
        return _unavailable_hard_gate(
            contract, "paired runner traces do not contain a sealed analysis protocol"
        )

    if operator in {"lesion_greater_than_intact", "lesion_less_than_intact"}:
        metric_paths = {
            "firing_rate_hz": ("spike_metrics", "firing_rate_hz"),
            "isi_cv": ("spike_metrics", "isi_cv"),
            "medium_ahp_depth_mV": ("medium_ahp", "ahp_depth_mV"),
            "median_interspike_voltage_nadir_mV": (
                "interspike_voltage_nadir",
                "median_interspike_voltage_nadir_mV",
            ),
        }
        if metric not in metric_paths:
            raise StageBScorerError(f"{gate_id} has unsupported directional metric {metric!r}")
        group, name = metric_paths[metric]
        intact_value = intact[group].get(name)
        lesion_value = lesion[group].get(name)
        if intact_value is None or lesion_value is None:
            return _unavailable_hard_gate(
                contract, f"paired raw traces cannot resolve {metric} with the recorded spike count"
            )
        observed = {"intact": float(intact_value), "lesion": float(lesion_value)}
        passed = (
            observed["lesion"] > observed["intact"]
            if operator == "lesion_greater_than_intact"
            else observed["lesion"] < observed["intact"]
        )
        return {**result, "status": "scored", "observed": observed, "passed": passed}

    if metric == "absolute_baseline_rate_change_fraction":
        intact_rate_raw = intact["spike_metrics"].get("firing_rate_hz")
        lesion_rate_raw = lesion["spike_metrics"].get("firing_rate_hz")
        if intact_rate_raw is None or lesion_rate_raw is None:
            return _unavailable_hard_gate(
                contract, "paired raw traces did not complete the 101-spike rate protocol"
            )
        intact_rate = float(intact_rate_raw)
        lesion_rate = float(lesion_rate_raw)
        if intact_rate <= 0.0:
            raise StageBScorerError("HCN baseline rate change is undefined when intact rate is zero")
        observed_value = abs(lesion_rate - intact_rate) / intact_rate
    elif metric in {"spike_count", "lesion_spike_count"}:
        observed_value = float(lesion["spike_metrics"]["spike_count"])
    elif metric == "mean_membrane_voltage_change_mV":
        observed_value = (
            float(lesion["mean_membrane_voltage_mV"])
            - float(intact["mean_membrane_voltage_mV"])
        )
    else:
        raise StageBScorerError(f"{gate_id} has unsupported scalar metric {metric!r}")

    if "value" not in contract:
        raise StageBScorerError(f"{gate_id} scalar hard gate has no filed value")
    threshold = _finite_number(contract["value"], f"{gate_id}.{metric}.value")
    comparisons = {
        "equal": observed_value == threshold,
        "less_than": observed_value < threshold,
        "greater_than": observed_value > threshold,
        "less_than_or_equal": observed_value <= threshold,
        "greater_than_or_equal": observed_value >= threshold,
    }
    if operator not in comparisons:
        raise StageBScorerError(f"{gate_id} uses unsupported operator {operator!r}")

    if "window_s" in contract:
        filed_window = _finite_number(contract["window_s"], f"{gate_id}.{metric}.window_s")
        spike_protocol = lesion["analysis_protocol"]["spike_metrics"]
        observed_window = float(spike_protocol["window_end_s"]) - float(
            spike_protocol["window_start_s"]
        )
        if not math.isclose(observed_window, filed_window, rel_tol=0.0, abs_tol=1e-12):
            return _unavailable_hard_gate(
                contract, "lesion runner trace does not contain the filed one-second scoring window"
            )
        result["window_s"] = filed_window
    return {**result, "status": "scored", "observed": observed_value, "threshold": threshold,
            "passed": comparisons[operator]}


def score_intrinsic_lesion_observations(
    document: Mapping[str, Any], *, root: str | Path
) -> dict[str, Any]:
    """Recompute filed intrinsic-lesion gates from digest-bound runner traces."""

    required = {
        "schema", "readiness_only", "causal_gate_packet", "runner_observations",
    }
    if (not isinstance(document, Mapping) or set(document) != required
            or document.get("schema") != INTRINSIC_LESION_SCHEMA):
        raise StageBScorerError("intrinsic lesion observation document has an invalid shape or schema")
    readiness = document.get("readiness_only")
    if readiness != {"enabled": True, "reserved_seed_count": 0, "scientific_seed": None}:
        raise StageBScorerError("intrinsic lesion scoring is seed-free readiness-only")
    seed_checked = {key: value for key, value in document.items() if key != "readiness_only"}
    if _contains_seed(seed_checked):
        raise StageBScorerError("intrinsic lesion observations must not contain seed data")

    root_path = Path(root).resolve()
    gate_path, packet = _load_bound_json(
        root_path, document.get("causal_gate_packet"), "causal gate packet"
    )
    if packet.get("schema") not in {
        "v14-snr-stageB-causal-gates-v1",
        "v14-snr-stageB-causal-gates-v2",
    }:
        raise StageBScorerError("causal gate packet has the wrong schema")
    _, target = _load_bound_json(root_path, packet.get("target_packet"), "causal target packet")
    if target.get("schema") != "v14-snr-stageB-target-packet-v1":
        raise StageBScorerError("causal target packet has the wrong schema")

    gates = packet.get("causal_gates")
    if not isinstance(gates, list):
        raise StageBScorerError("causal gate packet has no gate list")
    selected: dict[str, Mapping[str, Any]] = {}
    for gate in gates:
        if isinstance(gate, Mapping) and gate.get("id") in _INTRINSIC_LESION_IDS:
            gate_id = str(gate["id"])
            if gate_id in selected:
                raise StageBScorerError(f"causal gate packet duplicates {gate_id}")
            selected[gate_id] = gate
    if set(selected) != set(_INTRINSIC_LESION_IDS):
        raise StageBScorerError("causal gate packet does not contain exactly the required intrinsic gates")

    artifacts, artifact_bindings, candidate = _load_intrinsic_runner_artifacts(
        root_path, document.get("runner_observations")
    )
    causal_gate_binding = {
        "path": gate_path.relative_to(root_path).as_posix(),
        "sha256": _digest_bytes(gate_path.read_bytes()),
    }
    recomputed = {
        arm: _recompute_trace_metrics(
            artifact["raw_observation"], arm, root=root_path,
            causal_gate_packet=causal_gate_binding,
        )
        for arm, artifact in artifacts.items()
    }

    results = []
    for gate_id in _INTRINSIC_LESION_IDS:
        gate = selected[gate_id]
        hard_gates = gate.get("hard_gates")
        if not isinstance(hard_gates, list) or not hard_gates:
            raise StageBScorerError(f"{gate_id} has no filed hard gates")
        scored = [
            _score_intrinsic_hard_gate(
                gate_id,
                contract,
                recomputed["intact_autonomous"],
                recomputed[_GATE_ARMS[gate_id]],
            )
            for contract in hard_gates
        ]
        failed = any(item["passed"] is False for item in scored)
        unavailable = any(item["passed"] is None for item in scored)
        results.append({
            "gate_id": gate_id,
            "source": gate.get("source"),
            "preparation": gate.get("preparation"),
            "passed": False if failed else None if unavailable else True,
            "hard_gates": scored,
        })
    any_failed = any(item["passed"] is False for item in results)
    any_unavailable = any(item["passed"] is None for item in results)
    all_passed = False if any_failed else None if any_unavailable else True
    return {
        "schema": INTRINSIC_LESION_RESULT_SCHEMA,
        "process_status": "completed",
        "scientific_verdict": None,
        "readiness_only": readiness,
        "adaptive_candidate": candidate,
        "causal_gate_packet": {
            "path": gate_path.relative_to(root_path).as_posix(),
            "sha256": _digest_bytes(gate_path.read_bytes()),
        },
        "runner_observations": artifact_bindings,
        "all_intrinsic_lesion_gates_passed": all_passed,
        "readiness_contract_result": (
            "FAIL" if any_failed else "UNAVAILABLE" if any_unavailable else "PASS"
        ),
        "source_equivalence_claimed": False,
        "results": results,
    }


def _spike_metrics(raw: Mapping[str, Any]) -> dict[str, Any]:
    return spike_train_metrics(
        raw.get("spike_times_s", []),
        **{key: raw[key] for key in (
            "time_unit", "sample_interval_s", "recording_start_s", "recording_end_s",
            "burn_in_start_s", "burn_in_end_s", "window_start_s", "window_end_s",
        )},
    )


def _raw_value(fixture: Mapping[str, Any], raw: Any) -> tuple[float, dict[str, Any]]:
    if not isinstance(raw, Mapping):
        raise StageBScorerError(f"fixture {fixture['id']!r} has no raw observation")
    kind = raw.get("kind")
    try:
        if kind == "spike_train":
            if fixture["units"] != "spikes/s":
                raise StageBScorerError("spike-train raw data cannot score this fixture unit")
            metrics = _spike_metrics(raw)
            return float(metrics["firing_rate_hz"]), metrics
        if kind == "paired_spike_rate_ratio":
            if fixture["units"] != "dimensionless ratio":
                raise StageBScorerError("paired spike-rate raw data cannot score this fixture unit")
            intact = raw.get("intact")
            lesion = raw.get("lesion")
            if not isinstance(intact, Mapping) or not isinstance(lesion, Mapping):
                raise StageBScorerError("paired spike-rate raw data requires intact and lesion traces")
            protocol_fields = (
                "time_unit", "sample_interval_s", "recording_start_s", "recording_end_s",
                "burn_in_start_s", "burn_in_end_s", "window_start_s", "window_end_s",
            )
            if any(intact.get(field) != lesion.get(field) for field in protocol_fields):
                raise StageBScorerError("intact and lesion spike-rate protocols must match exactly")
            intact_metrics = _spike_metrics(intact)
            lesion_metrics = _spike_metrics(lesion)
            intact_rate = float(intact_metrics["firing_rate_hz"])
            lesion_rate = float(lesion_metrics["firing_rate_hz"])
            ratio = lesion_rate / intact_rate if intact_rate > 0.0 else 0.0
            return ratio, {
                "intact": intact_metrics,
                "lesion": lesion_metrics,
                "lesion_over_intact": ratio,
                "persistent_intact_firing": intact_rate > 0.0,
                "persistent_lesion_firing": lesion_rate > 0.0,
            }
        if kind == "conductance_trace":
            if fixture["units"] != "nS":
                raise StageBScorerError("conductance raw data cannot score this fixture unit")
            metrics = peak_conductance(
                raw.get("time_s", []), raw.get("conductance_nS", []),
                **{key: raw[key] for key in (
                    "time_unit", "conductance_unit", "sample_interval_s", "recording_start_s",
                    "burn_in_start_s", "burn_in_end_s", "window_start_s", "window_end_s",
                )},
            )
            return float(metrics["peak_conductance_nS"]), metrics
    except KeyError as exc:
        raise StageBScorerError(f"fixture {fixture['id']!r} raw protocol is missing {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise StageBScorerError(f"fixture {fixture['id']!r} raw protocol is invalid: {exc}") from exc
    raise StageBScorerError(f"fixture {fixture['id']!r} has unsupported raw kind {kind!r}")


def score_raw_observations(document: Mapping[str, Any], *, root: str | Path) -> dict[str, Any]:
    """Recompute every bounded fixture observation from digest-bound raw data."""
    if not isinstance(document, Mapping) or document.get("schema") != SCHEMA:
        raise StageBScorerError("raw observation document has the wrong schema")
    root_path = Path(root).resolve()
    candidate = _candidate_echo(document.get("adaptive_candidate"))
    fixture_path, packet = _load_bound_json(root_path, document.get("fixture_packet"), "fixture packet")
    if packet.get("schema") != "v14-snr-stageB-scorer-fixtures-v1":
        raise StageBScorerError("fixture packet has the wrong schema")
    _, target = _load_bound_json(root_path, packet.get("source_target_packet"), "source target packet")
    if target.get("schema") != "v14-snr-stageB-target-packet-v1":
        raise StageBScorerError("source target packet has the wrong schema")

    fixtures = packet.get("fixtures")
    raw_entries = document.get("observations")
    if not isinstance(fixtures, list) or not isinstance(raw_entries, list):
        raise StageBScorerError("fixtures and observations must be lists")
    contracts = [validate_fixture(item) for item in fixtures]
    bounded = {item["id"]: item for item in contracts if item["score_kind"] == "bounded-interval"}
    supplied: dict[str, Any] = {}
    for entry in raw_entries:
        if not isinstance(entry, Mapping) or set(entry) != {"fixture_id", "raw"}:
            raise StageBScorerError("each raw observation must contain only fixture_id and raw")
        fixture_id = entry.get("fixture_id")
        if fixture_id not in bounded or fixture_id in supplied:
            raise StageBScorerError(f"raw observation has unknown or duplicate fixture {fixture_id!r}")
        supplied[str(fixture_id)] = entry["raw"]
    if set(supplied) != set(bounded):
        missing = sorted(set(bounded) - set(supplied))
        raise StageBScorerError(f"raw observations do not cover every bounded fixture: {missing}")

    results = []
    for fixture_id in sorted(bounded):
        fixture = bounded[fixture_id]
        value, metrics = _raw_value(fixture, supplied[fixture_id])
        scored = score_observation(fixture, {
            "cohort": fixture["cohort"], "pathway": fixture["pathway"],
            "metric": fixture["metric"], "units": fixture["units"], "value": value,
        })
        results.append({**scored, "raw_metrics": metrics})
    all_passed = all(item["passed"] is True for item in results)
    return {
        "schema": RESULT_SCHEMA,
        "process_status": "completed",
        "scientific_verdict": "GO" if all_passed else "NO_GO",
        "adaptive_candidate": candidate,
        "fixture_packet": {
            "path": fixture_path.relative_to(root_path).as_posix(),
            "sha256": _digest_bytes(fixture_path.read_bytes()),
        },
        "all_bounded_fixtures_passed": all_passed,
        "results": results,
        "unscored_boundaries": sorted(
            item["id"] for item in contracts if item["score_kind"] != "bounded-interval"
        ),
    }


def score_raw_observation_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    root: str | Path,
) -> dict[str, Any]:
    """Validate one raw artifact and create its result without overwriting evidence."""
    source = Path(input_path)
    destination = Path(output_path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageBScorerError(f"cannot load raw observation file: {exc}") from exc
    result = score_raw_observations(document, root=root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="ascii") as handle:
            json.dump(result, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise StageBScorerError(f"refusing to replace existing score: {destination}") from exc
    return result


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="raw observation JSON")
    parser.add_argument("--output", required=True, help="new score JSON")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args(argv)
    try:
        score_raw_observation_file(args.input, args.output, root=args.root)
    except StageBScorerError as exc:
        parser.exit(2, f"Stage B scorer infrastructure failure: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
