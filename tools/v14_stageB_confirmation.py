#!/usr/bin/env python3
"""Preregister and execute authoritative NumPy confirmation of GPU survivors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import socket
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.snr_executable_packet import canonical_bytes
from tools.v14_stageB_gpu_triage import TRIAGE_SCHEMA
from tools.v14_stageB_intrinsic_readiness import run_intrinsic_readiness
from tools.v14_stageB_scorer import score_intrinsic_lesion_observations


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SCHEMA = "v14-snr-stageB-numpy-confirmation-manifest-v1"
RECEIPT_SCHEMA = "v14-snr-stageB-numpy-confirmation-receipt-v1"
JOB_PLAN_SCHEMA = "v14-snr-stageB-numpy-confirmation-job-plan-v1"
JOB_PLAN_SCHEMA_V2 = "v14-snr-stageB-numpy-confirmation-job-plan-v2"
ARMS = (
    "intact_autonomous",
    "nap_lesion",
    "cav2_2_lesion",
    "sk_lesion",
    "hcn_baseline_lesion",
)
EXPECTED_SELECTION_COUNT = 2
EXPECTED_ENVIRONMENT = {
    "python_major_minor": "3.10",
    "numpy": "2.2.6",
    "scipy": "1.15.3",
    "h5py": "3.16.0",
    "pyyaml": "6.0.3",
}
RECOMPUTED_FLOAT_REL_TOL = 1e-12
RECOMPUTED_FLOAT_ABS_TOL = 1e-15


class StageBConfirmationError(ValueError):
    """Raised when confirmation identity, source, or execution is invalid."""


def _recomputed_score_matches(stored: Any, recomputed: Any) -> bool:
    """Compare replayed scores while tolerating only machine-scale float drift."""
    if isinstance(stored, bool) or isinstance(recomputed, bool):
        return type(stored) is type(recomputed) and stored == recomputed
    if isinstance(stored, float) or isinstance(recomputed, float):
        if not isinstance(stored, (int, float)) or not isinstance(recomputed, (int, float)):
            return False
        return math.isclose(
            float(stored),
            float(recomputed),
            rel_tol=RECOMPUTED_FLOAT_REL_TOL,
            abs_tol=RECOMPUTED_FLOAT_ABS_TOL,
        )
    if isinstance(stored, Mapping) or isinstance(recomputed, Mapping):
        if not isinstance(stored, Mapping) or not isinstance(recomputed, Mapping):
            return False
        return set(stored) == set(recomputed) and all(
            _recomputed_score_matches(stored[key], recomputed[key]) for key in stored
        )
    if isinstance(stored, list) or isinstance(recomputed, list):
        if not isinstance(stored, list) or not isinstance(recomputed, list):
            return False
        return len(stored) == len(recomputed) and all(
            _recomputed_score_matches(left, right)
            for left, right in zip(stored, recomputed, strict=True)
        )
    return type(stored) is type(recomputed) and stored == recomputed


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any) -> str:
    return _digest_bytes(canonical_bytes(value))


def _sha(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StageBConfirmationError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _inside(root: Path, value: str | Path, context: str, *, require_file: bool) -> Path:
    supplied = Path(value).expanduser()
    path = (supplied if supplied.is_absolute() else root / supplied).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise StageBConfirmationError(f"{context} must be inside repository_root") from exc
    if require_file and (path.is_symlink() or not path.is_file()):
        raise StageBConfirmationError(f"{context} must be a regular file")
    return path


def _relative(root: Path, path: Path) -> str:
    value = PurePosixPath(path.relative_to(root).as_posix())
    if value.is_absolute() or any(part in {"", ".", ".."} for part in value.parts):
        raise StageBConfirmationError("artifact path is not canonical repository-relative text")
    return value.as_posix()


def _load_bound_json(
    root: Path, path_value: str | Path, expected_sha256: str, context: str
) -> tuple[Path, dict[str, Any]]:
    path = _inside(root, path_value, context, require_file=True)
    expected = _sha(expected_sha256, f"{context} digest")
    raw = path.read_bytes()
    if _digest_bytes(raw) != expected:
        raise StageBConfirmationError(f"{context} digest does not match")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StageBConfirmationError(f"{context} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise StageBConfirmationError(f"{context} must contain an object")
    return path, value


def _runtime_environment() -> dict[str, str]:
    import h5py
    import numpy
    import scipy
    import yaml

    return {
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "h5py": h5py.__version__,
        "pyyaml": yaml.__version__,
    }


def build_confirmation_manifest(
    candidate_manifest_path: str | Path,
    candidate_manifest_sha256: str,
    triage_path: str | Path,
    triage_sha256: str,
    *,
    repository_root: str | Path = ROOT,
) -> dict[str, Any]:
    """Freeze exactly the engineering-pass candidates for NumPy confirmation."""

    root = Path(repository_root).expanduser().resolve(strict=True)
    candidates_path, candidates = _load_bound_json(
        root, candidate_manifest_path, candidate_manifest_sha256, "candidate manifest"
    )
    triage_file, triage = _load_bound_json(root, triage_path, triage_sha256, "GPU triage")
    if triage.get("schema") != TRIAGE_SCHEMA or triage.get("sha256") != _digest({
        key: value for key, value in triage.items() if key != "sha256"
    }):
        raise StageBConfirmationError("GPU triage has an invalid schema or self digest")
    if (
        triage.get("process_status") != "completed"
        or triage.get("engineering_screening_only") is not True
        or triage.get("scientific_verdict") is not None
        or triage.get("numpy_confirmation_required") is not True
    ):
        raise StageBConfirmationError("GPU triage changed its engineering-only boundary")
    triage_candidates = triage.get("candidates")
    if (
        triage.get("candidate_count") != 512
        or not isinstance(triage_candidates, list)
        or len(triage_candidates) != 512
    ):
        raise StageBConfirmationError("GPU triage does not cover the exact 512-candidate design")
    observed_counts: dict[str, int] = {}
    triage_identities: dict[str, str] = {}
    for item in triage_candidates:
        if not isinstance(item, Mapping):
            raise StageBConfirmationError("GPU triage contains an invalid candidate row")
        classification = item.get("classification")
        candidate_id = item.get("candidate_id")
        candidate_sha = item.get("candidate_sha256")
        if (
            classification not in {
                "engineering_pass", "engineering_fail", "engineering_inconclusive"
            }
            or not isinstance(candidate_id, str)
            or candidate_id in triage_identities
        ):
            raise StageBConfirmationError("GPU triage candidate identity is invalid or duplicated")
        triage_identities[candidate_id] = _sha(candidate_sha, "GPU triage candidate digest")
        observed_counts[classification] = observed_counts.get(classification, 0) + 1
    if triage.get("classification_counts") != dict(sorted(observed_counts.items())):
        raise StageBConfirmationError("GPU triage classification counts are inconsistent")
    selected = [
        item for item in triage.get("candidates", [])
        if isinstance(item, Mapping) and item.get("classification") == "engineering_pass"
    ]
    if len(selected) != EXPECTED_SELECTION_COUNT:
        raise StageBConfirmationError("GPU triage must select exactly two engineering passes")
    source_rows = {
        row.get("candidate", {}).get("candidate_id"): row
        for row in candidates.get("candidates", [])
        if isinstance(row, Mapping) and isinstance(row.get("candidate"), Mapping)
    }
    source_identities = {
        candidate_id: row.get("candidate_sha256") for candidate_id, row in source_rows.items()
    }
    if triage_identities != source_identities:
        raise StageBConfirmationError("GPU triage does not exactly match the filed candidate identities")
    selection = []
    for item in sorted(selected, key=lambda value: value["candidate_id"]):
        row = source_rows.get(item.get("candidate_id"))
        if (
            not isinstance(row, Mapping)
            or row.get("candidate_sha256") != item.get("candidate_sha256")
            or _digest(row.get("candidate")) != item.get("candidate_sha256")
        ):
            raise StageBConfirmationError("selected candidate does not match the filed source design")
        selection.append({
            "point_index": row["point_index"],
            "candidate_id": item["candidate_id"],
            "candidate_sha256": item["candidate_sha256"],
            "candidate": row["candidate"],
        })
    body = {
        "schema": MANIFEST_SCHEMA,
        "status": "preregistered-authoritative-numpy-confirmation",
        "device": "not_applicable_non_executed_confirmation_design",
        "provenance_exempt": (
            "non-executed confirmation selection; contains no NumPy result"
        ),
        "candidate_manifest": {
            "path": _relative(root, candidates_path),
            "sha256": _digest_bytes(candidates_path.read_bytes()),
            "self_sha256": candidates.get("sha256"),
        },
        "gpu_triage": {
            "path": _relative(root, triage_file),
            "sha256": _digest_bytes(triage_file.read_bytes()),
            "self_sha256": triage["sha256"],
            "classification_counts": triage.get("classification_counts"),
        },
        "selection_rule": "all_and_only_engineering_pass_candidates",
        "selected_count": len(selection),
        "arms": list(ARMS),
        "backend": "numpy",
        "device_required": "cpu",
        "environment_required": EXPECTED_ENVIRONMENT,
        "authoritative_source_required": True,
        "scientific_seed": None,
        "held_out_access": False,
        "selected_candidates": selection,
    }
    return {**body, "sha256": _digest(body)}


def write_manifest(
    manifest: Mapping[str, Any], destination: str | Path, *, repository_root: str | Path = ROOT
) -> Path:
    root = Path(repository_root).expanduser().resolve(strict=True)
    path = _inside(root, destination, "confirmation manifest output", require_file=False)
    if path.exists() or path.is_symlink():
        raise StageBConfirmationError("refusing to replace confirmation manifest")
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("sha256") != _digest({
        key: value for key, value in manifest.items() if key != "sha256"
    }):
        raise StageBConfirmationError("confirmation manifest is invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_bytes(manifest))
    return path


def _validate_confirmation_manifest(
    root: Path, path_value: str | Path, expected_sha256: str
) -> tuple[Path, dict[str, Any]]:
    path, manifest = _load_bound_json(root, path_value, expected_sha256, "confirmation manifest")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("sha256") != _digest({
            key: value for key, value in manifest.items() if key != "sha256"
        })
        or manifest.get("status") != "preregistered-authoritative-numpy-confirmation"
        or manifest.get("selected_count") != EXPECTED_SELECTION_COUNT
        or manifest.get("arms") != list(ARMS)
        or manifest.get("backend") != "numpy"
        or manifest.get("device_required") != "cpu"
        or manifest.get("environment_required") != EXPECTED_ENVIRONMENT
        or manifest.get("authoritative_source_required") is not True
        or manifest.get("scientific_seed") is not None
        or manifest.get("held_out_access") is not False
    ):
        raise StageBConfirmationError("confirmation manifest changed its execution boundary")
    return path, manifest


def build_job_plan(
    confirmation_manifest_path: str | Path,
    confirmation_manifest_sha256: str,
    expected_source_revision: str,
    expected_source_manifest_sha256: str,
    assignments: Sequence[Mapping[str, str]],
    *,
    repository_root: str | Path = ROOT,
    causal_gate_path: str | Path | None = None,
    causal_gate_sha256: str | None = None,
    analysis_protocol_path: str | Path | None = None,
    analysis_protocol_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind every confirmation survivor to one primary and one recovery host."""

    root = Path(repository_root).expanduser().resolve(strict=True)
    manifest_path, manifest = _validate_confirmation_manifest(
        root, confirmation_manifest_path, confirmation_manifest_sha256
    )
    if len(expected_source_revision) != 40 or any(
        character not in "0123456789abcdef" for character in expected_source_revision
    ):
        raise StageBConfirmationError("expected source revision must be a full Git SHA-1")
    source_manifest_sha = _sha(
        expected_source_manifest_sha256, "expected source manifest digest"
    )
    selected = {
        item["candidate_id"]: item
        for item in manifest["selected_candidates"]
    }
    jobs = []
    observed: set[str] = set()
    for assignment in assignments:
        candidate_id = assignment.get("candidate_id")
        primary_host = assignment.get("primary_host")
        recovery_host = assignment.get("recovery_host")
        if (
            candidate_id not in selected or candidate_id in observed
            or not isinstance(primary_host, str) or not primary_host
            or not isinstance(recovery_host, str) or not recovery_host
            or primary_host == recovery_host
        ):
            raise StageBConfirmationError("confirmation job assignment is invalid")
        observed.add(candidate_id)
        row = selected[candidate_id]
        jobs.append({
            "job_id": f"confirm-{row['point_index']}",
            "candidate_id": candidate_id,
            "candidate_sha256": row["candidate_sha256"],
            "primary_host": primary_host,
            "recovery_host": recovery_host,
            "max_attempts": 3,
            "retry_policy": "infrastructure_failure_only",
        })
    if observed != set(selected):
        raise StageBConfirmationError("job plan must assign all and only selected candidates")
    contract_values = (
        causal_gate_path,
        causal_gate_sha256,
        analysis_protocol_path,
        analysis_protocol_sha256,
    )
    if any(value is not None for value in contract_values) and not all(
        value is not None for value in contract_values
    ):
        raise StageBConfirmationError(
            "versioned contract requires both paths and both digests"
        )
    contract = None
    if all(value is not None for value in contract_values):
        causal_path, causal = _load_bound_json(
            root, causal_gate_path, str(causal_gate_sha256), "causal gate contract"
        )
        protocol_path, _ = _load_bound_json(
            root, analysis_protocol_path, str(analysis_protocol_sha256),
            "intrinsic analysis protocol",
        )
        protocol_binding = {
            "path": _relative(root, protocol_path),
            "sha256": _digest_bytes(protocol_path.read_bytes()),
        }
        if causal.get("authorized_analysis_protocol") != protocol_binding:
            raise StageBConfirmationError(
                "causal gate contract does not authorize the selected analysis protocol"
            )
        contract = {
            "causal_gate": {
                "path": _relative(root, causal_path),
                "sha256": _digest_bytes(causal_path.read_bytes()),
            },
            "analysis_protocol": protocol_binding,
        }
    body = {
        "schema": JOB_PLAN_SCHEMA_V2 if contract is not None else JOB_PLAN_SCHEMA,
        "status": "ready-for-dispatch",
        "confirmation_manifest": {
            "path": _relative(root, manifest_path),
            "sha256": _digest_bytes(manifest_path.read_bytes()),
            "self_sha256": manifest["sha256"],
        },
        "expected_source": {
            "revision": expected_source_revision,
            "source_manifest_sha256": source_manifest_sha,
        },
        "environment_required": EXPECTED_ENVIRONMENT,
        "candidate_atomic": True,
        "arm_splitting_permitted": False,
        "jobs": sorted(jobs, key=lambda item: item["job_id"]),
    }
    if contract is not None:
        body["contract"] = contract
    return {**body, "sha256": _digest(body)}


def _validate_job_plan(
    root: Path, path_value: str | Path, expected_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    path, plan = _load_bound_json(root, path_value, expected_sha256, "confirmation job plan")
    if (
        plan.get("schema") not in {JOB_PLAN_SCHEMA, JOB_PLAN_SCHEMA_V2}
        or plan.get("sha256") != _digest({
            key: value for key, value in plan.items() if key != "sha256"
        })
        or plan.get("status") != "ready-for-dispatch"
        or plan.get("environment_required") != EXPECTED_ENVIRONMENT
        or plan.get("candidate_atomic") is not True
        or plan.get("arm_splitting_permitted") is not False
    ):
        raise StageBConfirmationError("confirmation job plan changed its execution boundary")
    if plan.get("schema") == JOB_PLAN_SCHEMA_V2:
        contract = plan.get("contract")
        if not isinstance(contract, Mapping) or set(contract) != {
            "causal_gate", "analysis_protocol"
        }:
            raise StageBConfirmationError("versioned job plan has no exact contract binding")
        causal_path, causal = _load_bound_json(
            root, contract["causal_gate"].get("path"),
            contract["causal_gate"].get("sha256"), "job-plan causal gate contract",
        )
        protocol_path, _ = _load_bound_json(
            root, contract["analysis_protocol"].get("path"),
            contract["analysis_protocol"].get("sha256"),
            "job-plan intrinsic analysis protocol",
        )
        if causal.get("authorized_analysis_protocol") != {
            "path": _relative(root, protocol_path),
            "sha256": _digest_bytes(protocol_path.read_bytes()),
        }:
            raise StageBConfirmationError(
                "job-plan causal gate does not authorize its analysis protocol"
            )
        if _relative(root, causal_path) != contract["causal_gate"].get("path"):
            raise StageBConfirmationError("job-plan causal gate path is not canonical")
    elif "contract" in plan:
        raise StageBConfirmationError("V1 job plan cannot contain a versioned contract")
    return path, plan


def write_job_plan(
    plan: Mapping[str, Any], destination: str | Path, *, repository_root: str | Path = ROOT
) -> Path:
    root = Path(repository_root).expanduser().resolve(strict=True)
    path = _inside(root, destination, "confirmation job plan output", require_file=False)
    if path.exists() or path.is_symlink():
        raise StageBConfirmationError("refusing to replace confirmation job plan")
    if plan.get("schema") not in {JOB_PLAN_SCHEMA, JOB_PLAN_SCHEMA_V2} or plan.get("sha256") != _digest({
        key: value for key, value in plan.items() if key != "sha256"
    }):
        raise StageBConfirmationError("confirmation job plan is invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_bytes(plan))
    return path


def run_confirmation_candidate(
    confirmation_manifest_path: str | Path,
    confirmation_manifest_sha256: str,
    candidate_id: str,
    output_dir: str | Path,
    *,
    repository_root: str | Path = ROOT,
    execution_argv: Sequence[str] | None = None,
    expected_source_revision: str,
    expected_source_manifest_sha256: str,
    job_plan_path: str | Path,
    job_plan_sha256: str,
    job_id: str,
) -> dict[str, Any]:
    """Execute one exact selected candidate through the existing five-arm authority path."""

    root = Path(repository_root).expanduser().resolve(strict=True)
    manifest_path, manifest = _validate_confirmation_manifest(
        root, confirmation_manifest_path, confirmation_manifest_sha256
    )
    matches = [
        item for item in manifest["selected_candidates"]
        if isinstance(item, Mapping) and item.get("candidate_id") == candidate_id
    ]
    if len(matches) != 1:
        raise StageBConfirmationError("candidate is not an exact selected confirmation member")
    selected = matches[0]
    plan_path, plan = _validate_job_plan(root, job_plan_path, job_plan_sha256)
    jobs = [item for item in plan.get("jobs", []) if item.get("job_id") == job_id]
    if len(jobs) != 1:
        raise StageBConfirmationError("confirmation job is absent or duplicated")
    job = jobs[0]
    if (
        job.get("candidate_id") != candidate_id
        or job.get("candidate_sha256") != selected["candidate_sha256"]
        or plan.get("confirmation_manifest", {}).get("sha256")
        != _digest_bytes(manifest_path.read_bytes())
        or plan.get("expected_source") != {
            "revision": expected_source_revision,
            "source_manifest_sha256": expected_source_manifest_sha256,
        }
        or socket.gethostname() not in {job.get("primary_host"), job.get("recovery_host")}
    ):
        raise StageBConfirmationError("runtime host or identity does not match confirmation job")
    output = _inside(root, output_dir, "confirmation output", require_file=False)
    if output.exists() or output.is_symlink():
        raise StageBConfirmationError("confirmation output must not already exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    candidate_input = output.parent / f".{candidate_id}.candidate.json"
    if candidate_input.exists() or candidate_input.is_symlink():
        raise StageBConfirmationError("candidate staging input already exists")
    candidate_input.write_bytes(canonical_bytes(selected["candidate"]))
    if _digest_bytes(candidate_input.read_bytes()) != selected["candidate_sha256"]:
        candidate_input.unlink(missing_ok=True)
        raise StageBConfirmationError("staged candidate digest does not match selection")
    template = root / "research/specs/v14_snr_stageB_packet_template.json"
    contract = plan.get("contract")
    if contract is None:
        causal = root / "research/specs/v14_snr_stageB_causal_gates.json"
        protocol = root / "research/specs/v14_snr_stageB_intrinsic_protocol.json"
    else:
        causal = _inside(
            root, contract["causal_gate"]["path"], "job-plan causal gate", require_file=True
        )
        protocol = _inside(
            root, contract["analysis_protocol"]["path"],
            "job-plan analysis protocol", require_file=True,
        )
    argv = list(sys.argv if execution_argv is None else execution_argv)
    expected_revision = expected_source_revision
    if len(expected_revision) != 40 or any(
        character not in "0123456789abcdef" for character in expected_revision
    ):
        raise StageBConfirmationError("expected source revision must be a full Git SHA-1")
    expected_manifest = _sha(
        expected_source_manifest_sha256, "expected source manifest digest"
    )
    environment = _runtime_environment()
    if environment != manifest["environment_required"]:
        raise StageBConfirmationError(
            f"runtime environment does not match confirmation manifest: {environment}"
        )
    try:
        inner = run_intrinsic_readiness(
            template, _digest_bytes(template.read_bytes()),
            candidate_input, selected["candidate_sha256"],
            causal, _digest_bytes(causal.read_bytes()),
            output, repository_root=root,
            analysis_protocol_path=protocol,
            analysis_protocol_sha256=_digest_bytes(protocol.read_bytes()),
            execution_argv=argv,
            require_authoritative_source=True,
            expected_source_revision=expected_revision,
            expected_source_manifest_sha256=expected_manifest,
        )
        source_identity = inner.get("provenance", {}).get("source_identity")
        provenance = inner.get("provenance", {})
        if (
            inner.get("backend") != "numpy"
            or inner.get("device") != "cpu"
            or inner.get("scientific_verdict") is not None
            or not isinstance(source_identity, Mapping)
            or source_identity.get("authoritative") is not True
            or source_identity.get("revision") != expected_revision
            or source_identity.get("source_manifest_sha256") != expected_manifest
            or provenance.get("source_verified_at_start") is not True
            or provenance.get("source_verified_at_exit") is not True
            or inner.get("candidate", {}).get("candidate_id") != candidate_id
            or inner.get("candidate", {}).get("sha256") != selected["candidate_sha256"]
        ):
            raise StageBConfirmationError("inner NumPy receipt changed identity or authority boundary")
        inner_path = output / "readiness-receipt.json"
        artifact_rows = []
        for artifact in sorted(path for path in output.rglob("*") if path.is_file()):
            if artifact.is_symlink():
                raise StageBConfirmationError("confirmation output contains a symbolic link")
            artifact_rows.append(
                f"{_digest_bytes(artifact.read_bytes())}  {artifact.relative_to(output).as_posix()}\n"
            )
        artifact_manifest_path = output / "artifact-manifest.sha256"
        artifact_manifest_path.write_text("".join(artifact_rows), encoding="ascii")
        body = {
            "schema": RECEIPT_SCHEMA,
            "process_status": "completed",
            "scientific_verdict": None,
            "backend": "numpy",
            "device": "cpu",
            "confirmation_manifest": {
                "path": _relative(root, manifest_path),
                "sha256": _digest_bytes(manifest_path.read_bytes()),
                "self_sha256": manifest["sha256"],
            },
            "job_plan": {
                "path": _relative(root, plan_path),
                "sha256": _digest_bytes(plan_path.read_bytes()),
                "self_sha256": plan["sha256"],
                "job_id": job_id,
                "execution_host": socket.gethostname(),
            },
            "candidate": {
                "point_index": selected["point_index"],
                "candidate_id": candidate_id,
                "candidate_sha256": selected["candidate_sha256"],
            },
            "source_identity": dict(source_identity),
            "environment": environment,
            "expected_source": {
                "revision": expected_revision,
                "source_manifest_sha256": expected_manifest,
            },
            "artifact_manifest": {
                "path": _relative(root, artifact_manifest_path),
                "sha256": _digest_bytes(artifact_manifest_path.read_bytes()),
                "artifact_count": len(artifact_rows),
            },
            "inner_receipt": {
                "path": _relative(root, inner_path),
                "sha256": _digest_bytes(inner_path.read_bytes()),
            },
            "score": dict(inner["score"]),
        }
        if contract is not None:
            body["contract"] = dict(contract)
        receipt = {**body, "sha256": _digest(body)}
        receipt_path = output / "confirmation-receipt.json"
        with receipt_path.open("xb") as handle:
            handle.write(canonical_bytes(receipt))
        return receipt
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
    finally:
        candidate_input.unlink(missing_ok=True)


def verify_collected_confirmation(
    receipt_path: str | Path,
    expected_receipt_sha256: str,
    *,
    repository_root: str | Path = ROOT,
) -> dict[str, Any]:
    """Authenticate a collected result and locally recompute its strict score."""

    root = Path(repository_root).expanduser().resolve(strict=True)
    path, receipt = _load_bound_json(
        root, receipt_path, expected_receipt_sha256, "confirmation receipt"
    )
    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("sha256") != _digest({
        key: value for key, value in receipt.items() if key != "sha256"
    }):
        raise StageBConfirmationError("confirmation receipt has an invalid self digest")
    if (
        receipt.get("backend") != "numpy" or receipt.get("device") != "cpu"
        or receipt.get("scientific_verdict") is not None
        or receipt.get("environment") != EXPECTED_ENVIRONMENT
        or receipt.get("source_identity", {}).get("authoritative") is not True
        or receipt.get("expected_source") != {
            "revision": receipt.get("source_identity", {}).get("revision"),
            "source_manifest_sha256": receipt.get("source_identity", {}).get(
                "source_manifest_sha256"
            ),
        }
    ):
        raise StageBConfirmationError("confirmation receipt changed execution authority")
    job_binding = receipt.get("job_plan")
    if not isinstance(job_binding, Mapping):
        raise StageBConfirmationError("confirmation receipt has no job-plan binding")
    _, plan = _validate_job_plan(
        root, job_binding.get("path"), job_binding.get("sha256")
    )
    jobs = [
        item for item in plan.get("jobs", [])
        if item.get("job_id") == job_binding.get("job_id")
    ]
    if (
        len(jobs) != 1
        or jobs[0].get("candidate_id") != receipt.get("candidate", {}).get("candidate_id")
        or jobs[0].get("candidate_sha256")
        != receipt.get("candidate", {}).get("candidate_sha256")
        or job_binding.get("execution_host")
        not in {jobs[0].get("primary_host"), jobs[0].get("recovery_host")}
        or plan.get("expected_source") != receipt.get("expected_source")
        or (
            plan.get("schema") == JOB_PLAN_SCHEMA_V2
            and receipt.get("contract") != plan.get("contract")
        )
        or (
            plan.get("schema") == JOB_PLAN_SCHEMA
            and "contract" in receipt
        )
    ):
        raise StageBConfirmationError("collected receipt does not match its confirmation job")
    output = path.parent
    artifact_binding = receipt.get("artifact_manifest")
    if not isinstance(artifact_binding, Mapping):
        raise StageBConfirmationError("confirmation receipt has no artifact manifest")
    manifest = _inside(root, artifact_binding.get("path"), "artifact manifest", require_file=True)
    if _digest_bytes(manifest.read_bytes()) != artifact_binding.get("sha256"):
        raise StageBConfirmationError("collected artifact manifest digest does not match")
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, separator, relative = line.partition("  ")
        pure = PurePosixPath(relative)
        if (
            not separator or len(digest) != 64 or pure.is_absolute() or not pure.name
            or any(part in {"", ".", ".."} for part in pure.parts)
            or relative in expected
        ):
            raise StageBConfirmationError("collected artifact manifest is malformed")
        expected[relative] = _sha(digest, "collected artifact digest")
    actual = {
        item.relative_to(output).as_posix(): item
        for item in output.rglob("*")
        if item.is_file() and item not in {path, manifest}
    }
    if set(actual) != set(expected) or len(expected) != artifact_binding.get("artifact_count"):
        raise StageBConfirmationError("collected artifact file set is incomplete or widened")
    for relative, digest in expected.items():
        artifact = actual[relative]
        if artifact.is_symlink() or _digest_bytes(artifact.read_bytes()) != digest:
            raise StageBConfirmationError(f"collected artifact is corrupt: {relative}")
    inner_binding = receipt.get("inner_receipt")
    inner_path, inner = _load_bound_json(
        root, inner_binding.get("path"), inner_binding.get("sha256"), "inner receipt"
    )
    scorer_input_binding = inner.get("scorer_input")
    score_binding = inner.get("score")
    _, scorer_input = _load_bound_json(
        root, scorer_input_binding.get("path"), scorer_input_binding.get("sha256"),
        "collected scorer input",
    )
    _, stored_score = _load_bound_json(
        root, score_binding.get("path"), score_binding.get("sha256"), "collected score"
    )
    recomputed = score_intrinsic_lesion_observations(scorer_input, root=root)
    if not _recomputed_score_matches(stored_score, recomputed):
        raise StageBConfirmationError("collected score does not equal local recomputation")
    return {
        "verified": True,
        "candidate": receipt["candidate"],
        "source_identity": receipt["source_identity"],
        "environment": receipt["environment"],
        "score": stored_score,
        "receipt_sha256": _digest_bytes(path.read_bytes()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--candidate-manifest", required=True)
    prepare.add_argument("--candidate-manifest-sha256", required=True)
    prepare.add_argument("--triage", required=True)
    prepare.add_argument("--triage-sha256", required=True)
    prepare.add_argument("--output", required=True)
    run = sub.add_parser("run")
    run.add_argument("--confirmation-manifest", required=True)
    run.add_argument("--confirmation-manifest-sha256", required=True)
    run.add_argument("--candidate-id", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--expected-source-revision", required=True)
    run.add_argument("--expected-source-manifest-sha256", required=True)
    run.add_argument("--job-plan", required=True)
    run.add_argument("--job-plan-sha256", required=True)
    run.add_argument("--job-id", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--confirmation-manifest", required=True)
    plan.add_argument("--confirmation-manifest-sha256", required=True)
    plan.add_argument("--expected-source-revision", required=True)
    plan.add_argument("--expected-source-manifest-sha256", required=True)
    plan.add_argument("--causal-gate")
    plan.add_argument("--causal-gate-sha256")
    plan.add_argument("--analysis-protocol")
    plan.add_argument("--analysis-protocol-sha256")
    plan.add_argument(
        "--assignment", action="append", required=True,
        help="candidate_id,primary_hostname,recovery_hostname",
    )
    plan.add_argument("--output", required=True)
    verify = sub.add_parser("verify-collected")
    verify.add_argument("--receipt", required=True)
    verify.add_argument("--receipt-sha256", required=True)
    for command in (prepare, plan, run, verify):
        command.add_argument("--repository-root", default=str(ROOT))
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare":
            result = build_confirmation_manifest(
                args.candidate_manifest, args.candidate_manifest_sha256,
                args.triage, args.triage_sha256,
                repository_root=args.repository_root,
            )
            write_manifest(result, args.output, repository_root=args.repository_root)
        elif args.action == "plan":
            assignments = []
            for value in args.assignment:
                fields = value.split(",")
                if len(fields) != 3:
                    raise StageBConfirmationError("assignment must have exactly three fields")
                assignments.append({
                    "candidate_id": fields[0],
                    "primary_host": fields[1],
                    "recovery_host": fields[2],
                })
            result = build_job_plan(
                args.confirmation_manifest, args.confirmation_manifest_sha256,
                args.expected_source_revision, args.expected_source_manifest_sha256,
                assignments, repository_root=args.repository_root,
                causal_gate_path=args.causal_gate,
                causal_gate_sha256=args.causal_gate_sha256,
                analysis_protocol_path=args.analysis_protocol,
                analysis_protocol_sha256=args.analysis_protocol_sha256,
            )
            write_job_plan(result, args.output, repository_root=args.repository_root)
        elif args.action == "run":
            result = run_confirmation_candidate(
                args.confirmation_manifest, args.confirmation_manifest_sha256,
                args.candidate_id, args.output_dir,
                repository_root=args.repository_root,
                expected_source_revision=args.expected_source_revision,
                expected_source_manifest_sha256=args.expected_source_manifest_sha256,
                job_plan_path=args.job_plan,
                job_plan_sha256=args.job_plan_sha256,
                job_id=args.job_id,
            )
        else:
            result = verify_collected_confirmation(
                args.receipt, args.receipt_sha256,
                repository_root=args.repository_root,
            )
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"Stage B NumPy confirmation failure: {exc}\n")
    print(canonical_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
