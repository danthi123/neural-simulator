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
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

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
READINESS_ARM = "intact_autonomous"
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
    if document["arm"] != READINESS_ARM:
        raise StageBPhysiologyRunnerError(
            f"unsupported Stage B arm {document['arm']!r}; only {READINESS_ARM!r} is implemented"
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
            "intact autonomous arm parameters must contain packet, policy, and release references"
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


def _output_document(
    document: Mapping[str, Any],
    bindings: Mapping[str, RuntimeSNrPacketBinding],
    *,
    voltage_millivolts: list[list[float]],
    spike_states: list[list[bool]],
    dt_ms: float,
    root: Path,
    candidate_release: Mapping[str, Any],
) -> dict[str, Any]:
    steps = len(voltage_millivolts)
    if steps != len(spike_states) or steps == 0:
        raise StageBPhysiologyRunnerError("runner did not capture a complete raw trace")
    binding = bindings.get("snr")
    if binding is None or len(bindings) != 1:
        raise StageBPhysiologyRunnerError("runner requires exactly one authenticated SNr binding")
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
        "arm": READINESS_ARM,
        "adaptive_candidate": _candidate_echo(document),
        "raw_observation": {
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
        },
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


def run_readiness_intact(
    adaptive_parameter_document: str,
    output: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Execute an authenticated intact SNr packet and write one new raw artifact."""

    document = _load_parameter_document(adaptive_parameter_document)
    root = Path(repository_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise StageBPhysiologyRunnerError("repository_root must be a directory")
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise StageBPhysiologyRunnerError("refusing to replace an existing raw observation")
    backend, backend_name = get_backend()
    if backend_name != "numpy" or backend is not np:
        raise StageBPhysiologyRunnerError("V14 Stage B readiness runner did not acquire NumPy")

    candidate_release = _load_candidate_release(root, document["references"], document)
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
    try:
        bridge._initialize_simulation_data()
        if not bridge.is_initialized:
            raise StageBPhysiologyRunnerError("authenticated SNr bridge initialization failed")
        if set(bridge.snr_packet_bindings) != {"snr"}:
            raise StageBPhysiologyRunnerError("bridge did not retain the authenticated SNr binding")
        steps = int(round(config.total_simulation_time_ms / config.dt_ms))
        for _ in range(steps):
            bridge._run_one_simulation_step()
            voltage = np.asarray(to_host(bridge.cp_membrane_potential_v), dtype=np.float64)
            spikes = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
            if voltage.shape != (1,) or spikes.shape != (1,):
                raise StageBPhysiologyRunnerError("SNr bridge changed the single-cell trace shape")
            if not np.all(np.isfinite(voltage)):
                raise StageBPhysiologyRunnerError("SNr bridge produced non-finite voltage")
            voltage_millivolts.append([float(value) for value in voltage])
            spike_states.append([bool(value) for value in spikes])
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
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="ascii") as handle:
            json.dump(result, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise StageBPhysiologyRunnerError("refusing to replace an existing raw observation") from exc
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness", action="store_true", help="required seed-free readiness mode")
    parser.add_argument("--adaptive-parameter-document", required=True, help="sealed adaptive parameter JSON")
    parser.add_argument("--output", required=True, help="new raw-observation JSON path")
    parser.add_argument("--repository-root", required=True, help="root that owns packet and policy artifacts")
    args = parser.parse_args(argv)
    if not args.readiness:
        parser.exit(2, "Stage B runner infrastructure failure: --readiness is required\n")
    try:
        result = run_readiness_intact(
            args.adaptive_parameter_document,
            args.output,
            repository_root=args.repository_root,
        )
    except (OSError, StageBPhysiologyRunnerError, ValueError, TypeError) as exc:
        parser.exit(2, f"Stage B runner infrastructure failure: {exc}\n")
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
