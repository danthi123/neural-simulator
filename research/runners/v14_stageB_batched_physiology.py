#!/usr/bin/env python3
"""Authenticated GPU-batched engineering screen for V14 Stage B intrinsic arms.

The runner executes one one-neuron, packet-backed region per declared candidate
inside one CuPy bridge.  It produces compact in-memory traces for a later
``tools/compact_trace`` boundary.  Results are screening evidence only: every
candidate requires confirmation with the NumPy reference runner before it can
support a scientific claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DECLARATION_SCHEMA = "v14-snr-stageB-authenticated-gpu-batch-v1"
OUTPUT_SCHEMA = "v14-snr-stageB-gpu-batched-physiology-v1"
PHASED_OUTPUT_SCHEMA = "v14-snr-stageB-gpu-batched-physiology-v2"
ANALYSIS_PROTOCOL_SCHEMA = "v14-snr-stageB-intrinsic-protocol-v1"
PHASED_ANALYSIS_PROTOCOL_SCHEMA = "v14-snr-stageB-intrinsic-protocol-v3"
READINESS_ARMS = frozenset(
    {
        "intact_autonomous",
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
_DECLARATION_KEYS = {"schema", "arm", "analysis_protocol", "candidates", "sha256"}
_CANDIDATE_KEYS = {"candidate_id", "candidate_sha256", "release", "packet", "policy"}
_REFERENCE_KEYS = {"path", "sha256"}
DEFAULT_CHUNK_STEPS = 4096
MAX_CHUNK_STEPS = 65536
DT_MS = 0.05
EVENT_TARGET_SPIKES = 101
EVENT_TIMEOUT_STEPS = 400_000
NAP_STEPS = 20_000
NAP_PHASED_BASELINE_STEPS = 40_000
NAP_PHASED_POST_STEPS = 20_000


class StageBBatchedPhysiologyError(ValueError):
    """Raised before unauthenticated or scientifically unsafe work can run."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise StageBBatchedPhysiologyError(f"value is not canonical JSON: {exc}") from exc


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any) -> str:
    return _digest_bytes(_canonical_bytes(value))


def _sha256(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StageBBatchedPhysiologyError(
            f"{context} must be a lowercase SHA-256 digest"
        )
    return value


def _candidate_id(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 33 or ord(character) > 126 for character in value)
    ):
        raise StageBBatchedPhysiologyError(
            f"{context} must be nonempty trimmed printable ASCII text"
        )
    return value


def _reject_seed_or_held_out(value: Any, context: str) -> None:
    """Reject any attempt to route seed or held-out material into screening."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if "seed" in normalized:
                raise StageBBatchedPhysiologyError(f"{context} contains seed data")
            if "held_out" in normalized or "heldout" in normalized:
                raise StageBBatchedPhysiologyError(f"{context} contains held-out data")
            _reject_seed_or_held_out(item, f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_seed_or_held_out(item, f"{context}[{index}]")


def _canonical_relative_path(root: Path, value: Any, context: str) -> tuple[str, Path]:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\\" in value
        or "\x00" in value
        or any(ord(character) > 127 for character in value)
    ):
        raise StageBBatchedPhysiologyError(
            f"{context} must be canonical repository-relative POSIX text"
        )
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or str(relative) != value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise StageBBatchedPhysiologyError(
            f"{context} must be canonical repository-relative POSIX text"
        )
    unresolved = root.joinpath(*relative.parts)
    if unresolved.is_symlink():
        raise StageBBatchedPhysiologyError(f"{context} must not be a symbolic link")
    path = unresolved.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise StageBBatchedPhysiologyError(f"{context} escapes repository root") from exc
    if not path.is_file():
        raise StageBBatchedPhysiologyError(f"{context} must be a regular file")
    return value, path


def _load_bound_json(
    root: Path,
    reference: Any,
    context: str,
    *,
    require_canonical: bool = True,
) -> tuple[dict[str, str], dict[str, Any]]:
    if not isinstance(reference, Mapping) or set(reference) != _REFERENCE_KEYS:
        raise StageBBatchedPhysiologyError(f"{context} must contain only path and sha256")
    relative, path = _canonical_relative_path(root, reference.get("path"), f"{context} path")
    expected = _sha256(reference.get("sha256"), f"{context} sha256")
    raw = path.read_bytes()
    if _digest_bytes(raw) != expected:
        raise StageBBatchedPhysiologyError(f"{context} digest does not match")
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StageBBatchedPhysiologyError(f"{context} is not valid JSON") from exc
    if not isinstance(document, dict):
        raise StageBBatchedPhysiologyError(f"{context} must contain a JSON object")
    if require_canonical and raw != _canonical_bytes(document):
        raise StageBBatchedPhysiologyError(f"{context} must be canonical JSON")
    return {"path": relative, "sha256": expected}, document


def _validate_release(
    root: Path, candidate: Mapping[str, Any]
) -> tuple[dict[str, str], dict[str, Any]]:
    release_ref, release = _load_bound_json(root, candidate["release"], "candidate release")
    required = {"schema", "template", "candidate", "artifacts", "fitted_value_status"}
    if set(release) != required or release.get("schema") != "v14-snr-stageB-candidate-release-v1":
        raise StageBBatchedPhysiologyError("candidate release has an invalid schema or shape")
    _reject_seed_or_held_out(release, "candidate release")

    template = release.get("template")
    if (
        not isinstance(template, Mapping)
        or set(template) != {"template_id", "sha256"}
        or not isinstance(template.get("template_id"), str)
        or not template.get("template_id")
    ):
        raise StageBBatchedPhysiologyError("candidate release has an invalid template binding")
    _sha256(template.get("sha256"), "candidate release template sha256")

    identity = release.get("candidate")
    if not isinstance(identity, Mapping) or set(identity) != {"candidate_id", "sha256"}:
        raise StageBBatchedPhysiologyError("candidate release has an invalid candidate binding")
    if (
        identity.get("candidate_id") != candidate["candidate_id"]
        or identity.get("sha256") != candidate["candidate_sha256"]
    ):
        raise StageBBatchedPhysiologyError("candidate release does not bind the declared candidate")

    artifacts = release.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != _RELEASE_ARTIFACT_KEYS:
        raise StageBBatchedPhysiologyError("candidate release has invalid artifact bindings")
    for key, digest in artifacts.items():
        _sha256(digest, f"candidate release {key}")
    if (
        artifacts.get("sealed_packet_sha256") != candidate["packet"]["sha256"]
        or artifacts.get("authority_policy_sha256") != candidate["policy"]["sha256"]
    ):
        raise StageBBatchedPhysiologyError("candidate release does not bind the packet and policy")
    if release.get("fitted_value_status") != (
        "Fitted values remain derived/model priors, never measurements."
    ):
        raise StageBBatchedPhysiologyError(
            "candidate release changed the fitted-value evidence boundary"
        )

    parent = PurePosixPath(release_ref["path"]).parent
    if (
        PurePosixPath(candidate["packet"]["path"]) != parent / "packet.sealed.json"
        or PurePosixPath(candidate["policy"]["path"]) != parent / "authority-policy.json"
    ):
        raise StageBBatchedPhysiologyError(
            "candidate release, packet, and policy must be verifier siblings"
        )
    return release_ref, release


def _validate_analysis_protocol(
    root: Path, reference: Mapping[str, Any], arm: str
) -> tuple[dict[str, str], dict[str, Any]]:
    binding, protocol = _load_bound_json(
        root, reference, "analysis protocol", require_canonical=False
    )
    required = {
        "device",
        "provenance_exempt",
        "schema",
        "protocol_id",
        "status",
        "causal_gate_authority",
        "target_packet",
        "primary_source",
        "analysis_conventions",
        "execution",
        "arms",
        "scientific_boundaries",
    }
    protocol_schema = protocol.get("schema")
    if set(protocol) != required or protocol_schema not in {
        ANALYSIS_PROTOCOL_SCHEMA,
        PHASED_ANALYSIS_PROTOCOL_SCHEMA,
    }:
        raise StageBBatchedPhysiologyError("analysis protocol has an invalid schema or shape")
    if protocol.get("status") != "production-measurement-partial":
        raise StageBBatchedPhysiologyError("analysis protocol changed its scientific status")
    if protocol.get("execution") != {
        "dt_ms": 0.05,
        "dt_status": "project_operational_discretization_requires_timestep_convergence_before_waveform_claims",
        "trace_policy": "uncropped_post_update_voltage_and_spike_state",
    }:
        raise StageBBatchedPhysiologyError("analysis protocol changed the filed execution settings")
    if protocol.get("analysis_conventions") != {
        "cv_method": "population standard deviation of the 100 complete interspike intervals divided by their mean",
        "cv_method_evidence_class": "project_analysis_convention",
        "frequency_method": "100 divided by the elapsed time from the first through the 101st spike",
        "frequency_method_evidence_class": "project_analysis_convention",
    }:
        raise StageBBatchedPhysiologyError("analysis protocol changed the preregistered formulas")

    authority = protocol.get("causal_gate_authority")
    if not isinstance(authority, Mapping) or set(authority) != {"path", "role"}:
        raise StageBBatchedPhysiologyError("analysis protocol lacks its causal-gate authority")
    authority_path, gate_path = _canonical_relative_path(
        root, authority.get("path"), "causal-gate authority path"
    )
    try:
        gate = json.loads(gate_path.read_bytes())
    except json.JSONDecodeError as exc:
        raise StageBBatchedPhysiologyError("causal-gate authority is not JSON") from exc
    expected_authorization = {"path": binding["path"], "sha256": binding["sha256"]}
    if not isinstance(gate, Mapping) or gate.get("authorized_analysis_protocol") != expected_authorization:
        raise StageBBatchedPhysiologyError("causal gate does not authorize this analysis protocol")

    target_binding, _ = _load_bound_json(
        root, protocol.get("target_packet"), "target packet", require_canonical=False
    )
    arms = protocol.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != READINESS_ARMS:
        raise StageBBatchedPhysiologyError("analysis protocol must define exactly the Stage B arms")
    arm_protocol = arms.get(arm)
    if not isinstance(arm_protocol, Mapping):
        raise StageBBatchedPhysiologyError(f"analysis protocol has no arm {arm!r}")
    termination = arm_protocol.get("termination")
    spike_metrics = arm_protocol.get("spike_metrics")
    if arm == "nap_lesion":
        expected_termination = {
            "duration_s": 1.0,
            "duration_evidence_class": "project_operational_from_filed_causal_gate",
            "mode": "fixed_duration",
        }
        if termination != expected_termination or spike_metrics != {
            "source_evidence_class": "project_operational",
            "window_s": 1.0,
        }:
            raise StageBBatchedPhysiologyError("Nap arm changed the filed one-second protocol")
        if protocol_schema == PHASED_ANALYSIS_PROTOCOL_SCHEMA:
            phased = arm_protocol.get("mean_voltage_change")
            if (
                not isinstance(phased, Mapping)
                or phased.get("same_cell_requirement")
                != "one continuously simulated cell; do not substitute independently initialized intact and lesion traces"
                or phased.get("phase_schedule")
                != {
                    "intact_baseline_duration_s": 2.0,
                    "lesion_onset_s": 2.0,
                    "post_lesion_duration_s": 1.0,
                    "total_duration_s": 3.0,
                }
            ):
                raise StageBBatchedPhysiologyError(
                    "V3 Nap arm changed the filed same-cell phase schedule"
                )
    elif termination != {
        "maximum_duration_s": 20.0,
        "maximum_duration_evidence_class": (
            "project_operational_resource_bound_not_a_physiology_gate"
        ),
        "mode": "event_count_or_timeout",
    } or not isinstance(spike_metrics, Mapping) or (
        set(spike_metrics)
        != {"source_locator", "target_spike_count", "target_spike_count_evidence_class"}
        or spike_metrics.get("target_spike_count") != EVENT_TARGET_SPIKES
        or spike_metrics.get("target_spike_count_evidence_class") != "source_reported"
    ):
        raise StageBBatchedPhysiologyError("event-count arm changed the 101-spike timeout protocol")
    return binding, {
        "schema": protocol_schema,
        "arm": dict(arm_protocol),
        "causal_gate_authority": {"path": authority_path, "role": authority["role"]},
        "target_packet": target_binding,
    }


def load_batch_declaration(
    declaration_path: str | Path,
    declaration_sha256: str,
    *,
    repository_root: str | Path = ROOT,
) -> dict[str, Any]:
    """Authenticate and normalize one exact seed-free batch declaration."""

    root = Path(repository_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise StageBBatchedPhysiologyError("repository_root must be a directory")
    path = Path(declaration_path).expanduser().resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise StageBBatchedPhysiologyError("batch declaration escapes repository root") from exc
    _, regular_path = _canonical_relative_path(root, relative, "batch declaration path")
    expected_file_digest = _sha256(declaration_sha256, "batch declaration sha256")
    raw = regular_path.read_bytes()
    if _digest_bytes(raw) != expected_file_digest:
        raise StageBBatchedPhysiologyError("batch declaration digest does not match")
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StageBBatchedPhysiologyError("batch declaration is not JSON") from exc
    if not isinstance(document, dict) or raw != _canonical_bytes(document):
        raise StageBBatchedPhysiologyError("batch declaration must be a canonical JSON object")
    _reject_seed_or_held_out(document, "batch declaration")
    if set(document) != _DECLARATION_KEYS or document.get("schema") != DECLARATION_SCHEMA:
        raise StageBBatchedPhysiologyError("batch declaration has an invalid schema or shape")
    if document.get("sha256") != _digest({key: value for key, value in document.items() if key != "sha256"}):
        raise StageBBatchedPhysiologyError("batch declaration self digest is invalid")
    arm = document.get("arm")
    if arm not in READINESS_ARMS:
        raise StageBBatchedPhysiologyError(f"unsupported Stage B arm {arm!r}")

    protocol_binding, protocol = _validate_analysis_protocol(root, document["analysis_protocol"], arm)
    rows = document.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise StageBBatchedPhysiologyError("batch declaration must contain candidates")
    candidate_ids: set[str] = set()
    candidate_digests: set[str] = set()
    normalized_candidates: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != _CANDIDATE_KEYS:
            raise StageBBatchedPhysiologyError(f"candidate row {index} has an invalid shape")
        candidate_id = _candidate_id(row.get("candidate_id"), f"candidate row {index} id")
        candidate_digest = _sha256(row.get("candidate_sha256"), f"candidate {candidate_id} sha256")
        if candidate_id in candidate_ids or candidate_digest in candidate_digests:
            raise StageBBatchedPhysiologyError("batch declaration contains duplicate candidates")
        refs: dict[str, dict[str, str]] = {}
        for kind in ("release", "packet", "policy"):
            ref = row.get(kind)
            if not isinstance(ref, Mapping) or set(ref) != _REFERENCE_KEYS:
                raise StageBBatchedPhysiologyError(
                    f"candidate {candidate_id} {kind} must contain only path and sha256"
                )
            relative_path, artifact_path = _canonical_relative_path(
                root, ref.get("path"), f"candidate {candidate_id} {kind} path"
            )
            digest = _sha256(ref.get("sha256"), f"candidate {candidate_id} {kind} sha256")
            if _digest_bytes(artifact_path.read_bytes()) != digest:
                raise StageBBatchedPhysiologyError(
                    f"candidate {candidate_id} {kind} digest does not match"
                )
            refs[kind] = {"path": relative_path, "sha256": digest}
        normalized = {
            "candidate_id": candidate_id,
            "candidate_sha256": candidate_digest,
            **refs,
        }
        release_binding, release = _validate_release(root, normalized)
        normalized["release"] = release_binding
        normalized["release_document"] = release
        normalized["region_name"] = f"snr_candidate_{index:04d}"
        normalized_candidates.append(normalized)
        candidate_ids.add(candidate_id)
        candidate_digests.add(candidate_digest)

    return {
        "schema": document["schema"],
        "path": relative,
        "sha256": expected_file_digest,
        "self_sha256": document["sha256"],
        "arm": arm,
        "analysis_protocol": protocol_binding,
        "protocol": protocol,
        "candidates": normalized_candidates,
        "repository_root": root,
    }


def _runtime_components() -> SimpleNamespace:
    requested = os.environ.get("SIM_BACKEND")
    if requested not in {None, "cupy"}:
        raise StageBBatchedPhysiologyError("GPU batch runner requires SIM_BACKEND=cupy")
    os.environ["SIM_BACKEND"] = "cupy"
    if "sim.backend" in sys.modules:
        backend_module = sys.modules["sim.backend"]
    else:
        from sim import backend as backend_module
    xp, backend_name = backend_module.get_backend()
    if backend_name != "cupy" or xp.__name__ != "cupy":
        raise StageBBatchedPhysiologyError("GPU batch runner did not acquire CuPy")

    from sim import bridge as bridge_module
    if bridge_module.cp is not xp:
        raise StageBBatchedPhysiologyError(
            "sim.bridge was imported under another backend; run the GPU batch in a fresh "
            "SIM_BACKEND=cupy process"
        )
    SimulationBridge = bridge_module.SimulationBridge
    from sim.config import CoreSimConfig, GPUConfig, RuntimeState, VisualizationConfig
    from sim.enums import NeuronModel
    from sim.regions import BrainRegion
    from sim.snr_packet_runtime import runtime_binding_manifest_bytes

    return SimpleNamespace(
        xp=xp,
        SimulationBridge=SimulationBridge,
        CoreSimConfig=CoreSimConfig,
        GPUConfig=GPUConfig,
        RuntimeState=RuntimeState,
        VisualizationConfig=VisualizationConfig,
        NeuronModel=NeuronModel,
        BrainRegion=BrainRegion,
        runtime_binding_manifest_bytes=runtime_binding_manifest_bytes,
    )


def _build_config(
    candidates: list[Mapping[str, Any]], maximum_steps: int, runtime: SimpleNamespace
) -> Any:
    regions = [
        runtime.BrainRegion(
            name=candidate["region_name"],
            n_neurons=1,
            internal_density=0.0,
            plastic_internal=False,
            snr_executable_packet_path=candidate["packet"]["path"],
            snr_executable_packet_sha256=candidate["packet"]["sha256"],
            snr_authority_policy_path=candidate["policy"]["path"],
            snr_authority_policy_sha256=candidate["policy"]["sha256"],
        )
        for candidate in candidates
    ]
    return runtime.CoreSimConfig(
        total_simulation_time_ms=maximum_steps * DT_MS,
        dt_ms=DT_MS,
        num_neurons=len(regions),
        connections_per_neuron=0,
        seed=0,
        neuron_model_type=runtime.NeuronModel.HODGKIN_HUXLEY.name,
        default_neuron_type_hh="HH_EXCITATORY_DEFAULT_LEGACY",
        enable_brain_region_framework=True,
        brain_regions=regions,
        region_pathways=[],
        snr_authority_policy_path=None,
        snr_authority_policy_sha256=None,
        enable_parameter_heterogeneity=False,
        enable_conductance_noise=False,
        enable_hebbian_learning=False,
        enable_short_term_plasticity=False,
        enable_structural_plasticity=False,
        enable_homeostasis=False,
        enable_stdp=False,
        enable_inhibitory_stdp=False,
        enable_reward_modulation=False,
        enable_ou_process=False,
        hh_external_drive_scale=0.0,
        enable_snr_direct_outputs=True,
        read_only_fast_step=True,
    )


def _binding_provenance(binding: Any) -> dict[str, str]:
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


def _synchronize(xp: Any) -> None:
    xp.cuda.Stream.null.synchronize()


def _to_host(xp: Any, value: Any) -> np.ndarray:
    if hasattr(xp, "asnumpy"):
        return np.asarray(xp.asnumpy(value))
    return np.asarray(value)


def _apply_intervention(
    bridge: Any, arm: str, candidate_count: int, xp: Any
) -> list[dict[str, Any]]:
    if arm == "intact_autonomous":
        return [
            {
                "kind": "none",
                "operation": "authenticated_packet_intact",
                "target": None,
                "runtime_conductance_field": None,
                "conductance_density_unit": "mS/cm^2",
                "before": None,
                "after": None,
            }
            for _ in range(candidate_count)
        ]
    target, field = _LESION_RUNTIME_FIELDS[arm]
    conductance = getattr(bridge, field, None)
    if conductance is None or tuple(conductance.shape) != (candidate_count,):
        raise StageBBatchedPhysiologyError(f"lesion target {field} was not initialized")
    _synchronize(xp)
    before = _to_host(xp, conductance).astype(np.float64, copy=False)
    if not np.all(np.isfinite(before)) or np.any(before < 0.0):
        raise StageBBatchedPhysiologyError(f"lesion target {field} is invalid")
    conductance[...] = 0.0
    _synchronize(xp)
    after = _to_host(xp, conductance).astype(np.float64, copy=False)
    if not np.array_equal(after, np.zeros(candidate_count, dtype=np.float64)):
        raise StageBBatchedPhysiologyError(f"complete lesion did not zero {field}")
    return [
        {
            "kind": "complete_intrinsic_current_lesion",
            "operation": "set_conductance_density_to_zero_after_authenticated_packet_initialization",
            "target": target,
            "runtime_conductance_field": field,
            "conductance_density_unit": "mS/cm^2",
            "before": float(before[index]),
            "after": float(after[index]),
        }
        for index in range(candidate_count)
    ]


def _run_trace_chunks(
    bridge: Any,
    *,
    xp: Any,
    candidate_count: int,
    maximum_steps: int,
    target_spikes: int | None,
    chunk_steps: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]], int, int]:
    voltage_chunks: list[list[np.ndarray]] = [[] for _ in range(candidate_count)]
    spike_chunks: list[list[np.ndarray]] = [[] for _ in range(candidate_count)]
    terminal_steps: list[int | None] = [None] * candidate_count
    observed_spikes = np.zeros(candidate_count, dtype=np.int64)
    bridge_steps = 0
    synchronization_boundaries = 0

    for chunk_start in range(0, maximum_steps, chunk_steps):
        width = min(chunk_steps, maximum_steps - chunk_start)
        device_voltage = xp.empty((width, candidate_count), dtype=xp.float32)
        device_spikes = xp.empty((width, candidate_count), dtype=xp.bool_)
        for offset in range(width):
            bridge._run_one_simulation_step()
            device_voltage[offset, :] = bridge.cp_membrane_potential_v
            device_spikes[offset, :] = bridge.cp_firing_states
        bridge_steps += width

        _synchronize(xp)
        synchronization_boundaries += 1
        voltage = _to_host(xp, device_voltage).astype(np.float32, copy=False)
        spikes = _to_host(xp, device_spikes).astype(bool, copy=False)
        if voltage.shape != (width, candidate_count) or spikes.shape != voltage.shape:
            raise StageBBatchedPhysiologyError("bridge changed the batched trace shape")
        if not np.all(np.isfinite(voltage)):
            raise StageBBatchedPhysiologyError("bridge produced non-finite voltage")

        for index in range(candidate_count):
            if terminal_steps[index] is not None:
                continue
            take = width
            if target_spikes is not None:
                cumulative = observed_spikes[index] + np.cumsum(
                    spikes[:, index], dtype=np.int64
                )
                reached = np.flatnonzero(cumulative >= target_spikes)
                if reached.size:
                    take = int(reached[0]) + 1
                    terminal_steps[index] = chunk_start + take
            selected_spikes = spikes[:take, index].copy()
            voltage_chunks[index].append(voltage[:take, index].copy())
            spike_chunks[index].append(selected_spikes)
            observed_spikes[index] += int(np.count_nonzero(selected_spikes))
        if target_spikes is not None and all(step is not None for step in terminal_steps):
            break

    voltages: list[np.ndarray] = []
    spike_states: list[np.ndarray] = []
    terminations: list[dict[str, Any]] = []
    for index in range(candidate_count):
        candidate_voltage = np.concatenate(voltage_chunks[index]).astype(np.float32, copy=False)
        candidate_spikes = np.concatenate(spike_chunks[index]).astype(bool, copy=False)
        if candidate_voltage.size != candidate_spikes.size or candidate_voltage.size == 0:
            raise StageBBatchedPhysiologyError("runner did not capture a complete candidate trace")
        count = int(np.count_nonzero(candidate_spikes))
        if target_spikes is None:
            reason = "fixed_duration_complete"
            mode = "fixed_duration"
        elif count >= target_spikes:
            if count != target_spikes:
                raise StageBBatchedPhysiologyError("event trace exceeded its exact spike target")
            reason = "target_spike_count_reached"
            mode = "event_count_or_timeout"
        else:
            reason = "maximum_duration_reached"
            mode = "event_count_or_timeout"
        terminations.append(
            {
                "mode": mode,
                "reason": reason,
                "steps_executed": int(candidate_voltage.size),
                "spikes_observed": count,
                "target_spike_count": target_spikes,
                "maximum_steps": maximum_steps,
                "timeout_is_physiology_failure": False,
            }
        )
        voltages.append(candidate_voltage)
        spike_states.append(candidate_spikes)
    return voltages, spike_states, terminations, bridge_steps, synchronization_boundaries


def run_authenticated_gpu_batch(
    declaration_path: str | Path,
    declaration_sha256: str,
    *,
    repository_root: str | Path = ROOT,
    chunk_steps: int = DEFAULT_CHUNK_STEPS,
    _runtime: SimpleNamespace | None = None,
) -> dict[str, Any]:
    """Execute one exact authenticated batch and return compact in-memory traces."""

    if isinstance(chunk_steps, bool) or not isinstance(chunk_steps, int):
        raise StageBBatchedPhysiologyError("chunk_steps must be an integer")
    if not 1 <= chunk_steps <= MAX_CHUNK_STEPS:
        raise StageBBatchedPhysiologyError(
            f"chunk_steps must be in [1, {MAX_CHUNK_STEPS}]"
        )
    declaration = load_batch_declaration(
        declaration_path,
        declaration_sha256,
        repository_root=repository_root,
    )
    runtime = _runtime if _runtime is not None else _runtime_components()
    xp = runtime.xp
    candidates = declaration["candidates"]
    arm = declaration["arm"]
    protocol_v3 = declaration["protocol"]["schema"] == PHASED_ANALYSIS_PROTOCOL_SCHEMA
    phased_nap = arm == "nap_lesion" and protocol_v3
    if phased_nap:
        maximum_steps = NAP_PHASED_BASELINE_STEPS + NAP_PHASED_POST_STEPS
        target_spikes = None
    elif arm == "nap_lesion":
        maximum_steps = NAP_STEPS
        target_spikes = None
    else:
        maximum_steps = EVENT_TIMEOUT_STEPS
        target_spikes = EVENT_TARGET_SPIKES
    config = _build_config(candidates, maximum_steps, runtime)
    bridge = runtime.SimulationBridge(
        core_config=config,
        viz_config=runtime.VisualizationConfig(),
        runtime_state=runtime.RuntimeState(),
        gpu_config=runtime.GPUConfig(
            enable_profiling=False,
            stats_sync_interval_steps=maximum_steps + 1,
        ),
        simulation_source_root=str(declaration["repository_root"]),
    )

    try:
        bridge._initialize_simulation_data()
        if not bridge.is_initialized:
            raise StageBBatchedPhysiologyError("authenticated batched bridge initialization failed")
        dispatch_check = getattr(bridge, "_snr_direct_outputs_can_dispatch", None)
        if not callable(dispatch_check) or not dispatch_check(config):
            raise StageBBatchedPhysiologyError(
                "authenticated batched bridge cannot dispatch SNr direct outputs"
            )
        bindings = bridge.snr_packet_bindings
        if set(bindings) != {candidate["region_name"] for candidate in candidates}:
            raise StageBBatchedPhysiologyError("runtime bindings do not match declared candidate regions")
        binding_provenance: dict[str, dict[str, str]] = {}
        for candidate in candidates:
            binding = bindings[candidate["region_name"]]
            if (
                binding.packet_path != candidate["packet"]["path"]
                or binding.packet_file_sha256 != candidate["packet"]["sha256"]
                or binding.authority_policy_sha256 != candidate["policy"]["sha256"]
            ):
                raise StageBBatchedPhysiologyError(
                    f"runtime binding does not match candidate {candidate['candidate_id']}"
                )
            binding_provenance[candidate["region_name"]] = _binding_provenance(binding)
        manifest_sha256 = _digest_bytes(runtime.runtime_binding_manifest_bytes(bindings))
        if phased_nap:
            baseline_v, baseline_s, _, baseline_steps, baseline_syncs = _run_trace_chunks(
                bridge,
                xp=xp,
                candidate_count=len(candidates),
                maximum_steps=NAP_PHASED_BASELINE_STEPS - 1,
                target_spikes=None,
                chunk_steps=chunk_steps,
            )
            interventions = _apply_intervention(bridge, arm, len(candidates), xp)
            post_v, post_s, _, post_steps, post_syncs = _run_trace_chunks(
                bridge,
                xp=xp,
                candidate_count=len(candidates),
                maximum_steps=NAP_PHASED_POST_STEPS + 1,
                target_spikes=None,
                chunk_steps=chunk_steps,
            )
            voltages = [
                np.concatenate((before, after)).astype(np.float32, copy=False)
                for before, after in zip(baseline_v, post_v, strict=True)
            ]
            spikes = [
                np.concatenate((before, after)).astype(bool, copy=False)
                for before, after in zip(baseline_s, post_s, strict=True)
            ]
            bridge_steps = baseline_steps + post_steps
            sync_boundaries = baseline_syncs + post_syncs
            terminations = [
                {
                    "mode": "fixed_duration",
                    "reason": "same_cell_phased_duration_complete",
                    "steps_executed": int(trace.size),
                    "spikes_observed": int(np.count_nonzero(trace)),
                    "target_spike_count": None,
                    "maximum_steps": maximum_steps,
                    "timeout_is_physiology_failure": False,
                }
                for trace in spikes
            ]
            for intervention in interventions:
                intervention.update(
                    {
                        "operation": "set_conductance_density_to_zero_between_post_update_samples",
                        "timestamp_s": 2.0,
                        "lesion_onset_sample_index": NAP_PHASED_BASELINE_STEPS - 1,
                        "lesion_onset_sample_number": NAP_PHASED_BASELINE_STEPS,
                        "last_intact_sample_s": 2.0 - DT_MS / 1000.0,
                        "first_lesion_sample_s": 2.0,
                    }
                )
        else:
            interventions = _apply_intervention(bridge, arm, len(candidates), xp)
            voltages, spikes, terminations, bridge_steps, sync_boundaries = _run_trace_chunks(
                bridge,
                xp=xp,
                candidate_count=len(candidates),
                maximum_steps=maximum_steps,
                target_spikes=target_spikes,
                chunk_steps=chunk_steps,
            )
    finally:
        bridge.clear_simulation_state_and_gpu_memory()

    candidate_results = []
    for index, candidate in enumerate(candidates):
        sample_count = int(voltages[index].size)
        candidate_results.append(
            {
                "candidate_id": candidate["candidate_id"],
                "candidate_sha256": candidate["candidate_sha256"],
                "arm": arm,
                "runtime_intervention": interventions[index],
                "termination": terminations[index],
                "trace": {
                    "encoding": "numpy_float32_voltage_and_little_bitpacked_spikes",
                    "sample_interval_s": DT_MS / 1000.0,
                    "sample_semantics": "post-update state at the declared time",
                    "sample_count": sample_count,
                    "recording_start_s": DT_MS / 1000.0,
                    "recording_end_s": (sample_count + 1) * DT_MS / 1000.0,
                    "uncropped": True,
                    "voltage_mV": voltages[index],
                    "spike_states_packed": np.packbits(spikes[index], bitorder="little"),
                    "spike_bitorder": "little",
                },
                "provenance": {
                    "release": candidate["release"],
                    "packet": candidate["packet"],
                    "policy": candidate["policy"],
                    "runtime_binding": binding_provenance[candidate["region_name"]],
                },
            }
        )

    return {
        "schema": PHASED_OUTPUT_SCHEMA if protocol_v3 else OUTPUT_SCHEMA,
        "process_status": "completed",
        "backend": "cupy",
        "device": "cuda",
        "engineering_screening_only": True,
        "scientific_verdict": None,
        "source_equivalence_claimed": False,
        "numpy_confirmation_required": True,
        "confirmation_requirement": (
            "Every selected candidate and arm must be rerun by the authenticated NumPy "
            "reference runner before scientific scoring or interpretation."
        ),
        "arm": arm,
        "batch_declaration": {
            "path": declaration["path"],
            "sha256": declaration["sha256"],
            "self_sha256": declaration["self_sha256"],
        },
        "analysis_protocol": declaration["analysis_protocol"],
        "execution": {
            "candidate_count": len(candidates),
            "one_neuron_region_per_candidate": True,
            "connections": 0,
            "plasticity": False,
            "noise": False,
            "enable_snr_direct_outputs": True,
            "read_only_fast_step": True,
            "chunk_steps": chunk_steps,
            "bridge_steps_executed": bridge_steps,
            "trace_synchronization_boundaries": sync_boundaries,
            "trace_transfer_policy": "device-resident chunks copied only at bounded chunk boundaries",
            **(
                {
                    "same_cell_phased_nap": True,
                    "intact_baseline_s": [0.0, 2.0],
                    "post_lesion_s": [2.0, 3.0],
                }
                if phased_nap
                else {}
            ),
        },
        "candidates": candidate_results,
        "provenance": {
            "runner": "research/runners/v14_stageB_batched_physiology.py",
            "runtime_binding_manifest_sha256": manifest_sha256,
            "runtime_bindings": [
                binding_provenance[candidate["region_name"]] for candidate in candidates
            ],
        },
    }


__all__ = [
    "ANALYSIS_PROTOCOL_SCHEMA",
    "DECLARATION_SCHEMA",
    "OUTPUT_SCHEMA",
    "PHASED_ANALYSIS_PROTOCOL_SCHEMA",
    "PHASED_OUTPUT_SCHEMA",
    "READINESS_ARMS",
    "StageBBatchedPhysiologyError",
    "load_batch_declaration",
    "run_authenticated_gpu_batch",
]
