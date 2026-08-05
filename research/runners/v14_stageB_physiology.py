#!/usr/bin/env python3
"""Readiness-only NumPy execution runner for one authenticated SNr packet.

This runner intentionally has no packet compiler and no scoring policy.  Its
only responsibility is to execute the already-authenticated packet referenced
by a sealed adaptive document, then preserve the uncropped voltage and spike
trace with the runtime binding evidence that produced it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# This is a reference-equation readiness runner.  Set the backend before any
# simulator imports so a direct CLI invocation cannot silently select CuPy.
if os.environ.get("SIM_BACKEND") not in {None, "numpy"}:
    raise RuntimeError("V14 Stage B readiness runner requires SIM_BACKEND=numpy")
os.environ["SIM_BACKEND"] = "numpy"

import numpy as np

from sim.backend import get_backend, to_host
from sim.bridge import SimulationBridge
from sim.config import CoreSimConfig, GPUConfig, RuntimeState, VisualizationConfig
from sim.enums import NeuronModel
from sim.regions import BrainRegion
from sim.snr_packet_runtime import (
    RuntimeSNrPacketBinding,
    load_runtime_snr_packet_bindings,
    runtime_binding_manifest_bytes,
)


PARAMETER_SCHEMA = "sim-adaptive-run-parameters-v1"
OUTPUT_SCHEMA = "v14-snr-stageB-physiology-observation-v1"
ANALYSIS_PROTOCOL_SCHEMA = "v14-snr-stageB-intrinsic-protocol-v1"
READINESS_ARM = "intact_autonomous"
READINESS_ARMS = frozenset(
    {
        READINESS_ARM,
        "nap_lesion",
        "cav2_2_lesion",
        "sk_lesion",
        "hcn_baseline_lesion",
    }
)
_LESION_RUNTIME_FIELDS = {
    "nap_lesion": ("nap", "cp_snr_g_nap_max"),
    "cav2_2_lesion": ("cav2.2", "cp_snr_g_ca_max"),
    "sk_lesion": ("sk", "cp_snr_g_sk_max"),
    "hcn_baseline_lesion": ("hcn", "cp_snr_g_h_max"),
}
_REFERENCE_KEYS = frozenset(
    {
        "snr_candidate_release_path",
        "snr_candidate_release_sha256",
        "snr_executable_packet_path",
        "snr_executable_packet_sha256",
        "snr_authority_policy_path",
        "snr_authority_policy_sha256",
    }
)
_RELEASE_ARTIFACT_KEYS = frozenset(
    {
        "compilation_request_sha256",
        "evidence_claims_sha256",
        "authority_claims_sha256",
        "structural_packet_sha256",
        "artifacts_verified_packet_sha256",
        "adjudication_sha256",
        "authority_policy_sha256",
        "sealed_packet_sha256",
        "materialized_sha256",
    }
)


class StageBPhysiologyRunnerError(ValueError):
    """Raised before a raw physiology artifact can be truthfully written."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_relative_path(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise StageBPhysiologyRunnerError(f"{context} must be nonempty trimmed text")
    if "\\" in value or "\x00" in value or any(ord(character) > 127 for character in value):
        raise StageBPhysiologyRunnerError(f"{context} must be ASCII POSIX-relative text")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(part in {"", ".", ".."} for part in path.parts):
        raise StageBPhysiologyRunnerError(f"{context} must be canonical and repository-relative")
    return value


def _sha256(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StageBPhysiologyRunnerError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _contains_seed(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any("seed" in str(key).lower() or _contains_seed(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_seed(item) for item in value)
    return False


def _load_parameter_document(value: str) -> dict[str, Any]:
    try:
        document = json.loads(value)
    except json.JSONDecodeError as exc:
        raise StageBPhysiologyRunnerError("adaptive parameter document is not JSON") from exc
    required = {
        "schema", "candidate_id", "candidate_sha256", "candidate_parameters",
        "arm", "arm_parameters", "effective_parameters",
    }
    if not isinstance(document, dict) or set(document) != required:
        raise StageBPhysiologyRunnerError("adaptive parameter document has an invalid shape")
    if document["schema"] != PARAMETER_SCHEMA:
        raise StageBPhysiologyRunnerError("adaptive parameter document has the wrong schema")
    if not isinstance(document["candidate_id"], str) or not document["candidate_id"]:
        raise StageBPhysiologyRunnerError("adaptive parameter document has an invalid candidate_id")
    if not isinstance(document["arm"], str) or document["arm"] not in READINESS_ARMS:
        raise StageBPhysiologyRunnerError(
            f"unsupported Stage B arm {document['arm']!r}; implemented readiness arms are "
            f"{sorted(READINESS_ARMS)!r}"
        )
    candidate_parameters = document["candidate_parameters"]
    arm_parameters = document["arm_parameters"]
    effective_parameters = document["effective_parameters"]
    if not isinstance(candidate_parameters, Mapping) or not isinstance(arm_parameters, Mapping):
        raise StageBPhysiologyRunnerError("adaptive parameter document parameters must be objects")
    if not isinstance(effective_parameters, Mapping):
        raise StageBPhysiologyRunnerError("effective_parameters must be an object")
    expected_candidate = {
        "schema": "sim-adaptive-candidate-v1",
        "candidate_id": document["candidate_id"],
        "parameters": dict(candidate_parameters),
    }
    if _sha256_bytes(_canonical_bytes(expected_candidate)) != _sha256(
        document["candidate_sha256"], "candidate_sha256"
    ):
        raise StageBPhysiologyRunnerError("candidate_sha256 does not bind candidate_parameters")
    if dict(effective_parameters) != {**dict(candidate_parameters), **dict(arm_parameters)}:
        raise StageBPhysiologyRunnerError(
            "effective_parameters does not exactly merge candidate and arm parameters"
        )
    if not candidate_parameters or set(candidate_parameters) & _REFERENCE_KEYS:
        raise StageBPhysiologyRunnerError(
            "candidate_parameters must contain only the numeric compiled candidate"
        )
    if set(arm_parameters) != _REFERENCE_KEYS:
        raise StageBPhysiologyRunnerError(
            "readiness arm parameters must contain only packet, policy, and release references"
        )
    if set(effective_parameters) != set(candidate_parameters) | _REFERENCE_KEYS:
        raise StageBPhysiologyRunnerError(
            "effective_parameters must contain only candidate values and authenticated references"
        )
    if _contains_seed(document):
        raise StageBPhysiologyRunnerError("readiness parameter documents must not contain seed data")
    references = {
        "release_path": _canonical_relative_path(
            arm_parameters["snr_candidate_release_path"], "snr_candidate_release_path"
        ),
        "release_sha256": _sha256(
            arm_parameters["snr_candidate_release_sha256"], "snr_candidate_release_sha256"
        ),
        "packet_path": _canonical_relative_path(
            arm_parameters["snr_executable_packet_path"], "snr_executable_packet_path"
        ),
        "packet_sha256": _sha256(
            arm_parameters["snr_executable_packet_sha256"], "snr_executable_packet_sha256"
        ),
        "policy_path": _canonical_relative_path(
            arm_parameters["snr_authority_policy_path"], "snr_authority_policy_path"
        ),
        "policy_sha256": _sha256(
            arm_parameters["snr_authority_policy_sha256"], "snr_authority_policy_sha256"
        ),
    }
    return {**document, "candidate_parameters": dict(candidate_parameters),
            "arm_parameters": dict(arm_parameters),
            "effective_parameters": dict(effective_parameters), "references": references}


def _candidate_echo(document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": document["candidate_id"],
        "candidate_sha256": document["candidate_sha256"],
        "effective_parameters": dict(document["effective_parameters"]),
    }


def _load_candidate_release(
    root: Path, references: Mapping[str, str], document: Mapping[str, Any]
) -> dict[str, Any]:
    release_path = root.joinpath(*PurePosixPath(references["release_path"]).parts).resolve()
    try:
        release_path.relative_to(root)
        raw = release_path.read_bytes()
        release = json.loads(raw)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise StageBPhysiologyRunnerError(f"cannot load candidate release: {exc}") from exc
    if _sha256_bytes(raw) != references["release_sha256"]:
        raise StageBPhysiologyRunnerError("candidate release digest does not match")
    if raw != _canonical_bytes(release):
        raise StageBPhysiologyRunnerError("candidate release is not canonical JSON")
    required = {"schema", "template", "candidate", "artifacts", "fitted_value_status"}
    if not isinstance(release, Mapping) or set(release) != required:
        raise StageBPhysiologyRunnerError("candidate release has an invalid shape")
    if release.get("schema") != "v14-snr-stageB-candidate-release-v1":
        raise StageBPhysiologyRunnerError("candidate release has the wrong schema")
    template = release.get("template")
    if (not isinstance(template, Mapping) or set(template) != {"template_id", "sha256"}
            or not isinstance(template.get("template_id"), str) or not template["template_id"]):
        raise StageBPhysiologyRunnerError("candidate release has an invalid template binding")
    _sha256(template.get("sha256"), "candidate release template sha256")
    candidate = release.get("candidate")
    if not isinstance(candidate, Mapping) or set(candidate) != {"candidate_id", "sha256"}:
        raise StageBPhysiologyRunnerError("candidate release has an invalid candidate binding")
    if (candidate["candidate_id"] != document["candidate_id"]
            or candidate["sha256"] != document["candidate_sha256"]):
        raise StageBPhysiologyRunnerError("candidate release does not bind the adaptive candidate")
    artifacts = release.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != _RELEASE_ARTIFACT_KEYS:
        raise StageBPhysiologyRunnerError("candidate release has invalid artifact bindings")
    for name, digest in artifacts.items():
        _sha256(digest, f"candidate release {name}")
    if (artifacts.get("sealed_packet_sha256") != references["packet_sha256"]
            or artifacts.get("authority_policy_sha256") != references["policy_sha256"]):
        raise StageBPhysiologyRunnerError("candidate release does not bind the packet and policy")
    if release.get("fitted_value_status") != (
        "Fitted values remain derived/model priors, never measurements."
    ):
        raise StageBPhysiologyRunnerError("candidate release changed the fitted-value evidence boundary")
    parent = PurePosixPath(references["release_path"]).parent
    if (PurePosixPath(references["packet_path"]) != parent / "packet.sealed.json"
            or PurePosixPath(references["policy_path"]) != parent / "authority-policy.json"):
        raise StageBPhysiologyRunnerError("candidate release, packet, and policy must be verifier siblings")
    return dict(release)


def _load_analysis_protocol(
    root: Path, path_value: str | Path, digest_value: str, arm: str,
) -> dict[str, Any]:
    """Load one digest-bound production protocol and validate its execution boundary."""

    digest = _sha256(digest_value, "analysis protocol sha256")
    path = Path(path_value).expanduser().resolve()
    try:
        relative = path.relative_to(root)
        raw = path.read_bytes()
        protocol = json.loads(raw)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise StageBPhysiologyRunnerError(f"cannot load analysis protocol: {exc}") from exc
    if path.is_symlink() or not path.is_file():
        raise StageBPhysiologyRunnerError("analysis protocol must be a regular file")
    if _sha256_bytes(raw) != digest:
        raise StageBPhysiologyRunnerError("analysis protocol digest does not match")
    required = {
        "device", "provenance_exempt",
        "schema", "protocol_id", "status", "causal_gate_authority",
        "target_packet", "primary_source", "analysis_conventions", "execution",
        "arms", "scientific_boundaries",
    }
    if not isinstance(protocol, Mapping) or set(protocol) != required:
        raise StageBPhysiologyRunnerError("analysis protocol has an invalid shape")
    if protocol.get("schema") != ANALYSIS_PROTOCOL_SCHEMA:
        raise StageBPhysiologyRunnerError("analysis protocol has the wrong schema")
    if protocol.get("status") != "production-measurement-partial":
        raise StageBPhysiologyRunnerError("analysis protocol changed its scientific status")
    if protocol.get("analysis_conventions") != {
        "cv_method": "population standard deviation of the 100 complete interspike intervals divided by their mean",
        "cv_method_evidence_class": "project_analysis_convention",
        "frequency_method": "100 divided by the elapsed time from the first through the 101st spike",
        "frequency_method_evidence_class": "project_analysis_convention",
    }:
        raise StageBPhysiologyRunnerError("analysis protocol changed the preregistered project formulas")
    execution = protocol.get("execution")
    if not isinstance(execution, Mapping) or set(execution) != {
        "dt_ms", "dt_status", "trace_policy",
    }:
        raise StageBPhysiologyRunnerError("analysis protocol has invalid execution settings")
    if execution != {
        "dt_ms": 0.05,
        "dt_status": "project_operational_discretization_requires_timestep_convergence_before_waveform_claims",
        "trace_policy": "uncropped_post_update_voltage_and_spike_state",
    }:
        raise StageBPhysiologyRunnerError("analysis protocol changed the filed execution settings")
    authority = protocol.get("causal_gate_authority")
    if not isinstance(authority, Mapping) or set(authority) != {"path", "role"}:
        raise StageBPhysiologyRunnerError("analysis protocol has no causal-gate authority")
    gate_relative = _canonical_relative_path(authority.get("path"), "causal gate path")
    gate_path = root.joinpath(*PurePosixPath(gate_relative).parts).resolve()
    try:
        gate_path.relative_to(root)
    except ValueError as exc:
        raise StageBPhysiologyRunnerError("causal gate path escapes repository_root") from exc
    if gate_path.is_symlink() or not gate_path.is_file():
        raise StageBPhysiologyRunnerError("analysis protocol causal-gate authority does not verify")
    try:
        gate_document = json.loads(gate_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise StageBPhysiologyRunnerError(f"cannot load causal-gate authority: {exc}") from exc
    authorized = gate_document.get("authorized_analysis_protocol") if isinstance(gate_document, Mapping) else None
    expected_authorization = {"path": PurePosixPath(*relative.parts).as_posix(), "sha256": digest}
    if authorized != expected_authorization:
        raise StageBPhysiologyRunnerError("causal gate does not authorize this analysis protocol")
    target = protocol.get("target_packet")
    if not isinstance(target, Mapping) or set(target) != {"path", "sha256"}:
        raise StageBPhysiologyRunnerError("analysis protocol has no target-packet binding")
    target_relative = _canonical_relative_path(target.get("path"), "target packet path")
    target_digest = _sha256(target.get("sha256"), "target packet sha256")
    target_path = root.joinpath(*PurePosixPath(target_relative).parts).resolve()
    try:
        target_path.relative_to(root)
    except ValueError as exc:
        raise StageBPhysiologyRunnerError("target packet path escapes repository_root") from exc
    if target_path.is_symlink() or not target_path.is_file() or _sha256_file(target_path) != target_digest:
        raise StageBPhysiologyRunnerError("analysis protocol target-packet binding does not verify")
    arms = protocol.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != READINESS_ARMS:
        raise StageBPhysiologyRunnerError("analysis protocol must define exactly the readiness arms")
    arm_protocol = arms.get(arm)
    if not isinstance(arm_protocol, Mapping):
        raise StageBPhysiologyRunnerError(f"analysis protocol has no valid arm {arm!r}")
    termination = arm_protocol.get("termination")
    spike_metrics = arm_protocol.get("spike_metrics")
    if not isinstance(termination, Mapping) or not isinstance(spike_metrics, Mapping):
        raise StageBPhysiologyRunnerError("analysis protocol arm lacks termination or spike metrics")
    if arm == "nap_lesion":
        if termination != {
            "duration_s": 1.0,
            "duration_evidence_class": "project_operational_from_filed_causal_gate",
            "mode": "fixed_duration",
        } or spike_metrics != {
            "source_evidence_class": "project_operational", "window_s": 1.0,
        }:
            raise StageBPhysiologyRunnerError("Nap arm changed the filed one-second protocol")
    else:
        if termination != {
            "maximum_duration_s": 20.0,
            "maximum_duration_evidence_class": (
                "project_operational_resource_bound_not_a_physiology_gate"
            ),
            "mode": "event_count_or_timeout",
        }:
            raise StageBPhysiologyRunnerError("event-count arm changed the operational timeout")
        if (
            set(spike_metrics) != {
                "source_locator", "target_spike_count",
                "target_spike_count_evidence_class",
            }
            or spike_metrics.get("target_spike_count_evidence_class") != "source_reported"
            or spike_metrics.get("target_spike_count") != 101
        ):
            raise StageBPhysiologyRunnerError("event-count arm changed the source-bound 101-spike contract")
    return {
        "path": PurePosixPath(*relative.parts).as_posix(),
        "sha256": digest,
        "causal_gate_authority": dict(authority),
        "arm": dict(arm_protocol),
    }


def _build_config(references: Mapping[str, str]) -> CoreSimConfig:
    return CoreSimConfig(
        total_simulation_time_ms=1.0,
        dt_ms=0.05,
        num_neurons=1,
        connections_per_neuron=0,
        seed=0,
        neuron_model_type=NeuronModel.HODGKIN_HUXLEY.name,
        default_neuron_type_hh="HH_EXCITATORY_DEFAULT_LEGACY",
        enable_brain_region_framework=True,
        brain_regions=[
            BrainRegion(
                name="snr",
                n_neurons=1,
                internal_density=0.0,
                snr_executable_packet_path=references["packet_path"],
                snr_executable_packet_sha256=references["packet_sha256"],
            )
        ],
        region_pathways=[],
        snr_authority_policy_path=references["policy_path"],
        snr_authority_policy_sha256=references["policy_sha256"],
        enable_parameter_heterogeneity=False,
        enable_conductance_noise=False,
        enable_hebbian_learning=False,
        enable_short_term_plasticity=False,
        enable_structural_plasticity=False,
        enable_ou_process=False,
        hh_external_drive_scale=0.0,
        enable_snr_direct_outputs=False,
    )


def _binding_provenance(binding: RuntimeSNrPacketBinding) -> dict[str, str]:
    return {
        "region_name": binding.region_name,
        "packet_path": binding.packet_path,
        "packet_file_sha256": binding.packet_file_sha256,
        "packet_sha256": binding.packet_sha256,
        "structural_sha256": binding.structural_sha256,
        "materialized_sha256": binding.materialized_sha256,
        "authority_policy_sha256": binding.authority_policy_sha256,
        "config_sha256": binding.config_sha256,
    }


def _apply_runtime_intervention(bridge: SimulationBridge, arm: str) -> dict[str, Any]:
    if arm == READINESS_ARM:
        return {
            "kind": "none",
            "operation": "authenticated_packet_intact",
            "target": None,
            "runtime_conductance_field": None,
            "conductance_density_unit": "mS/cm^2",
            "before": None,
            "after": None,
        }

    try:
        target, field = _LESION_RUNTIME_FIELDS[arm]
    except KeyError as exc:  # Defensive: document validation should make this unreachable.
        raise StageBPhysiologyRunnerError(
            f"no runtime intervention is defined for {arm!r}"
        ) from exc
    conductance = getattr(bridge, field, None)
    if conductance is None:
        raise StageBPhysiologyRunnerError(f"lesion target {field} was not initialized")
    before_array = np.asarray(to_host(conductance), dtype=np.float64)
    if before_array.shape != (1,) or not np.all(np.isfinite(before_array)):
        raise StageBPhysiologyRunnerError(
            f"lesion target {field} has an invalid runtime shape or value"
        )
    if np.any(before_array < 0.0):
        raise StageBPhysiologyRunnerError(f"lesion target {field} has a negative conductance")

    conductance[...] = 0.0
    after_array = np.asarray(to_host(conductance), dtype=np.float64)
    if after_array.shape != (1,) or not np.array_equal(after_array, np.zeros(1)):
        raise StageBPhysiologyRunnerError(f"complete lesion did not zero {field}")
    return {
        "kind": "complete_intrinsic_current_lesion",
        "operation": "set_conductance_density_to_zero_after_authenticated_packet_initialization",
        "target": target,
        "runtime_conductance_field": field,
        "conductance_density_unit": "mS/cm^2",
        "before": [float(value) for value in before_array],
        "after": [float(value) for value in after_array],
    }


def _output_document(
    document: Mapping[str, Any],
    bindings: Mapping[str, RuntimeSNrPacketBinding],
    *,
    voltage_millivolts: list[list[float]],
    spike_states: list[list[bool]],
    dt_ms: float,
    root: Path,
    candidate_release: Mapping[str, Any],
    runtime_intervention: Mapping[str, Any],
    analysis_protocol: Mapping[str, Any] | None,
    termination: Mapping[str, Any] | None,
) -> dict[str, Any]:
    steps = len(voltage_millivolts)
    if steps != len(spike_states) or steps == 0:
        raise StageBPhysiologyRunnerError("runner did not capture a complete raw trace")
    binding = bindings.get("snr")
    if binding is None or len(bindings) != 1:
        raise StageBPhysiologyRunnerError("runner requires exactly one authenticated SNr binding")
    raw_observation: dict[str, Any] = {
        "kind": "packet_voltage_spike_trace",
        "time_unit": "s",
        "voltage_unit": "mV",
        "sample_interval_s": dt_ms / 1000.0,
        "recording_start_s": dt_ms / 1000.0,
        "recording_end_s": (steps + 1) * dt_ms / 1000.0,
        "uncropped": True,
        "time_s": [(index + 1) * dt_ms / 1000.0 for index in range(steps)],
        "sample_semantics": "post-update state at the declared time",
        "voltage_mV": voltage_millivolts,
        "spike_states": spike_states,
    }
    if analysis_protocol is not None:
        if termination is None:
            raise StageBPhysiologyRunnerError("production trace has no termination record")
        raw_observation["analysis_protocol"] = {
            "binding": {
                "path": analysis_protocol["path"],
                "sha256": analysis_protocol["sha256"],
            },
            "termination": dict(termination),
        }
    return {
        "schema": OUTPUT_SCHEMA,
        "process_status": "completed",
        "readiness_only": {
            "enabled": True,
            "reserved_seed_count": 0,
            "scientific_seed": None,
            "engine_seed": 0,
            "engine_seed_effect": "none; connectivity, heterogeneity, noise, and plasticity are disabled",
        },
        "backend": "numpy",
        "device": "cpu",
        "arm": document["arm"],
        "runtime_intervention": dict(runtime_intervention),
        "adaptive_candidate": _candidate_echo(document),
        "raw_observation": raw_observation,
        "provenance": {
            "runner": "research/runners/v14_stageB_physiology.py",
            "repository_root": str(root),
            "runtime_binding_manifest_sha256": _sha256_bytes(
                runtime_binding_manifest_bytes(bindings)
            ),
            "bindings": [_binding_provenance(binding)],
            "candidate_release": {
                "path": document["references"]["release_path"],
                "sha256": document["references"]["release_sha256"],
                "candidate_sha256": candidate_release["candidate"]["sha256"],
            },
        },
    }


def run_readiness_arm(
    adaptive_parameter_document: str,
    output: str | Path,
    *,
    repository_root: str | Path,
    analysis_protocol_path: str | Path | None = None,
    analysis_protocol_sha256: str | None = None,
) -> dict[str, Any]:
    """Execute one authenticated SNr readiness arm and write one new raw artifact."""

    document = _load_parameter_document(adaptive_parameter_document)
    root = Path(repository_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise StageBPhysiologyRunnerError("repository_root must be a directory")
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise StageBPhysiologyRunnerError("refusing to replace an existing raw observation")
    if (analysis_protocol_path is None) != (analysis_protocol_sha256 is None):
        raise StageBPhysiologyRunnerError(
            "analysis protocol path and sha256 must be supplied together"
        )
    backend, backend_name = get_backend()
    if backend_name != "numpy" or backend is not np:
        raise StageBPhysiologyRunnerError("V14 Stage B readiness runner did not acquire NumPy")

    candidate_release = _load_candidate_release(root, document["references"], document)
    analysis_protocol = None
    if analysis_protocol_path is not None and analysis_protocol_sha256 is not None:
        analysis_protocol = _load_analysis_protocol(
            root, analysis_protocol_path, analysis_protocol_sha256, document["arm"]
        )
    config = _build_config(document["references"])
    bindings = load_runtime_snr_packet_bindings(config, source_root=root)
    binding = bindings.get("snr")
    if binding is None or len(bindings) != 1:
        raise StageBPhysiologyRunnerError("authenticated references did not produce one SNr binding")
    references = document["references"]
    if (
        binding.packet_path != references["packet_path"]
        or binding.packet_file_sha256 != references["packet_sha256"]
        or binding.authority_policy_sha256 != references["policy_sha256"]
    ):
        raise StageBPhysiologyRunnerError("runtime binding does not match sealed packet/policy references")

    bridge = SimulationBridge(
        core_config=config,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(enable_profiling=False),
        simulation_source_root=str(root),
    )
    voltage_millivolts: list[list[float]] = []
    spike_states: list[list[bool]] = []
    termination: dict[str, Any] | None = None
    try:
        bridge._initialize_simulation_data()
        if not bridge.is_initialized:
            raise StageBPhysiologyRunnerError("authenticated SNr bridge initialization failed")
        if set(bridge.snr_packet_bindings) != {"snr"}:
            raise StageBPhysiologyRunnerError("bridge did not retain the authenticated SNr binding")
        runtime_intervention = _apply_runtime_intervention(bridge, document["arm"])
        if analysis_protocol is None:
            maximum_steps = int(round(config.total_simulation_time_ms / config.dt_ms))
            target_spikes = None
            mode = "readiness_fixed_duration"
        else:
            filed_termination = analysis_protocol["arm"]["termination"]
            mode = str(filed_termination["mode"])
            if mode == "fixed_duration":
                maximum_steps = int(round(
                    float(filed_termination["duration_s"]) * 1000.0 / config.dt_ms
                ))
                target_spikes = None
            else:
                maximum_steps = int(round(
                    float(filed_termination["maximum_duration_s"]) * 1000.0 / config.dt_ms
                ))
                target_spikes = int(
                    analysis_protocol["arm"]["spike_metrics"]["target_spike_count"]
                )
        observed_spikes = 0
        for _ in range(maximum_steps):
            bridge._run_one_simulation_step()
            voltage = np.asarray(to_host(bridge.cp_membrane_potential_v), dtype=np.float64)
            spikes = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
            if voltage.shape != (1,) or spikes.shape != (1,):
                raise StageBPhysiologyRunnerError("SNr bridge changed the single-cell trace shape")
            if not np.all(np.isfinite(voltage)):
                raise StageBPhysiologyRunnerError("SNr bridge produced non-finite voltage")
            voltage_millivolts.append([float(value) for value in voltage])
            spike_states.append([bool(value) for value in spikes])
            observed_spikes += int(spikes[0])
            if target_spikes is not None and observed_spikes >= target_spikes:
                break
        if analysis_protocol is not None:
            if mode == "fixed_duration":
                reason = "fixed_duration_complete"
            elif observed_spikes >= int(target_spikes):
                reason = "target_spike_count_reached"
            else:
                reason = "maximum_duration_reached"
            termination = {
                "mode": mode,
                "reason": reason,
                "steps_executed": len(spike_states),
                "spikes_observed": observed_spikes,
                "target_spike_count": target_spikes,
                "maximum_steps": maximum_steps,
                "timeout_is_physiology_failure": False,
            }
    finally:
        bridge.clear_simulation_state_and_gpu_memory()

    result = _output_document(
        document,
        bindings,
        voltage_millivolts=voltage_millivolts,
        spike_states=spike_states,
        dt_ms=config.dt_ms,
        root=root,
        candidate_release=candidate_release,
        runtime_intervention=runtime_intervention,
        analysis_protocol=analysis_protocol,
        termination=termination,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="ascii") as handle:
            json.dump(result, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise StageBPhysiologyRunnerError("refusing to replace an existing raw observation") from exc
    return result


def run_readiness_intact(
    adaptive_parameter_document: str,
    output: str | Path,
    *,
    repository_root: str | Path,
    analysis_protocol_path: str | Path | None = None,
    analysis_protocol_sha256: str | None = None,
) -> dict[str, Any]:
    """Compatibility entry point for the authenticated intact readiness arm."""

    document = _load_parameter_document(adaptive_parameter_document)
    if document["arm"] != READINESS_ARM:
        raise StageBPhysiologyRunnerError(
            f"run_readiness_intact requires arm {READINESS_ARM!r}"
        )
    return run_readiness_arm(
        adaptive_parameter_document,
        output,
        repository_root=repository_root,
        analysis_protocol_path=analysis_protocol_path,
        analysis_protocol_sha256=analysis_protocol_sha256,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness", action="store_true", help="required seed-free readiness mode")
    parser.add_argument("--adaptive-parameter-document", required=True, help="sealed adaptive parameter JSON")
    parser.add_argument("--output", required=True, help="new raw-observation JSON path")
    parser.add_argument("--repository-root", required=True, help="root that owns packet and policy artifacts")
    parser.add_argument("--analysis-protocol-path", help="optional digest-bound production protocol")
    parser.add_argument("--analysis-protocol-sha256", help="expected production protocol digest")
    args = parser.parse_args(argv)
    if not args.readiness:
        parser.exit(2, "Stage B runner infrastructure failure: --readiness is required\n")
    try:
        result = run_readiness_arm(
            args.adaptive_parameter_document,
            args.output,
            repository_root=args.repository_root,
            analysis_protocol_path=args.analysis_protocol_path,
            analysis_protocol_sha256=args.analysis_protocol_sha256,
        )
    except (OSError, StageBPhysiologyRunnerError, ValueError, TypeError) as exc:
        parser.exit(2, f"Stage B runner infrastructure failure: {exc}\n")
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
