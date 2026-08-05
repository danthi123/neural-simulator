"""Run one authenticated, seed-free V14 Stage B intrinsic-lesion readiness pass.

This controller owns orchestration only.  The packet compiler and verifier
remain the authority path, the physiology runner owns execution, and the
intrinsic scorer owns all recomputation and readiness gates.  No scientific
analysis window or verdict is created here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.snr_executable_packet import PacketError, canonical_bytes
from tools.compact_trace import CompactTraceError, load_compact_trace
from tools.v14_stageB_packet_compiler import (
    StageBPacketCompilerError,
    compile_candidate,
)
from tools.v14_stageB_packet_verifier import (
    StageBPacketVerifierError,
    verify_candidate,
)
from tools.pool.provisioning.source_manifest import SourceManifestError, verify_manifest
from tools.v14_stageB_scorer import (
    INTRINSIC_LESION_SCHEMA,
    StageBScorerError,
    score_intrinsic_lesion_observations,
)


RECEIPT_SCHEMA = "v14-snr-stageB-intrinsic-readiness-v1"
PARAMETER_SCHEMA = "sim-adaptive-run-parameters-v1"
CANDIDATE_SCHEMA = "sim-adaptive-candidate-v1"
RUNNER_SCHEMA = "v14-snr-stageB-physiology-observation-v1"
RELEASE_SCHEMA = "v14-snr-stageB-candidate-release-v1"
ARMS = (
    "intact_autonomous",
    "nap_lesion",
    "cav2_2_lesion",
    "sk_lesion",
    "hcn_baseline_lesion",
)
REFERENCE_KEYS = frozenset(
    {
        "snr_candidate_release_path",
        "snr_candidate_release_sha256",
        "snr_executable_packet_path",
        "snr_executable_packet_sha256",
        "snr_authority_policy_path",
        "snr_authority_policy_sha256",
    }
)
READINESS_ONLY = {
    "enabled": True,
    "reserved_seed_count": 0,
    "scientific_seed": None,
}


class StageBIntrinsicReadinessError(ValueError):
    """The one-candidate intrinsic readiness pass could not complete."""


def run_readiness_arm(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Load the production runner only after the controller has fixed identity."""
    from research.runners.v14_stageB_physiology import (
        run_readiness_arm as execute,
    )

    return execute(*args, **kwargs)


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any) -> str:
    return _digest_bytes(canonical_bytes(value))


def _sha256(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StageBIntrinsicReadinessError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _inside_root(path: str | Path, root: Path, context: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise StageBIntrinsicReadinessError(f"{context} must be inside repository_root") from exc
    return resolved


def _relative(root: Path, path: Path, context: str) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise StageBIntrinsicReadinessError(f"{context} must be inside repository_root") from exc
    rendered = PurePosixPath(*relative.parts).as_posix()
    if not rendered or any(part in {"", ".", ".."} for part in PurePosixPath(rendered).parts):
        raise StageBIntrinsicReadinessError(f"{context} is not a canonical repository-relative path")
    return rendered


def _canonical_object(path: Path, context: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise StageBIntrinsicReadinessError(f"cannot load {context}: {exc}") from exc
    if not isinstance(value, dict):
        raise StageBIntrinsicReadinessError(f"{context} must contain an object")
    try:
        canonical = canonical_bytes(value)
    except PacketError as exc:
        raise StageBIntrinsicReadinessError(f"{context} is not canonical JSON: {exc}") from exc
    if raw != canonical:
        raise StageBIntrinsicReadinessError(f"{context} is not canonical JSON")
    return value


def _contains_seed(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any("seed" in str(key).lower() or _contains_seed(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_seed(item) for item in value)
    return False


def _load_pinned_candidate(
    path: str | Path, expected_sha256: str, root: Path,
) -> tuple[Path, str, dict[str, Any], dict[str, Any]]:
    _sha256(expected_sha256, "candidate expected digest")
    source = _inside_root(path, root, "candidate")
    if not source.is_file() or source.is_symlink():
        raise StageBIntrinsicReadinessError("candidate must be a regular file")
    document = _canonical_object(source, "candidate")
    digest = _digest_bytes(source.read_bytes())
    if digest != expected_sha256:
        raise StageBIntrinsicReadinessError("candidate digest does not match")
    if _contains_seed(document):
        raise StageBIntrinsicReadinessError("candidate contains seed data")
    if set(document) != {"schema", "candidate_id", "parameters"}:
        raise StageBIntrinsicReadinessError("candidate declaration has an invalid shape")
    if document.get("schema") != CANDIDATE_SCHEMA:
        raise StageBIntrinsicReadinessError("candidate has the wrong schema")
    candidate_id = document.get("candidate_id")
    parameters = document.get("parameters")
    if not isinstance(candidate_id, str) or not candidate_id or candidate_id != candidate_id.strip():
        raise StageBIntrinsicReadinessError("candidate has an invalid candidate_id")
    if not isinstance(parameters, Mapping) or not parameters:
        raise StageBIntrinsicReadinessError("candidate parameters must be a nonempty object")
    numeric: dict[str, Any] = {}
    for key, value in parameters.items():
        if not isinstance(key, str) or not key or key in REFERENCE_KEYS:
            raise StageBIntrinsicReadinessError("candidate has a reserved or invalid parameter key")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise StageBIntrinsicReadinessError("candidate parameters must contain only numeric values")
        if isinstance(value, float) and not math.isfinite(value):
            raise StageBIntrinsicReadinessError("candidate contains a non-finite parameter")
        numeric[key] = value
    return source, digest, document, {"candidate_id": candidate_id, "parameters": numeric}


def _valid_source_digest(value: str, context: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise StageBIntrinsicReadinessError(f"{context} is not a lowercase SHA-256 digest")
    return value


def _archive_source_identity(root: Path) -> dict[str, Any] | None:
    revision_path = root / ".source_revision"
    manifest_path = root / ".source_manifest.sha256"
    if not revision_path.exists() and not manifest_path.exists():
        return None
    if any(path.is_symlink() or not path.is_file() for path in (revision_path, manifest_path)):
        raise StageBIntrinsicReadinessError("archive source attestation must use regular files")
    fields: dict[str, str] = {}
    for line in revision_path.read_text(encoding="ascii").splitlines():
        if not line or "=" not in line:
            raise StageBIntrinsicReadinessError("archive source revision has malformed fields")
        key, value = line.split("=", 1)
        if not key or key in fields or not value:
            raise StageBIntrinsicReadinessError("archive source revision has duplicate or empty fields")
        fields[key] = value
    required = {
        "git_sha", "source_kind", "source_manifest_sha256",
        "source_ancestry_sha256", "excluded_worktree_paths", "created_utc",
    }
    if set(fields) != required or fields["source_kind"] != "git_archive":
        raise StageBIntrinsicReadinessError("archive source revision has an invalid shape")
    revision = fields["git_sha"]
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise StageBIntrinsicReadinessError("archive source revision has an invalid Git identity")
    if fields["excluded_worktree_paths"] != "0":
        raise StageBIntrinsicReadinessError("archive source includes excluded dirty worktree paths")
    manifest_sha = _valid_source_digest(
        fields["source_manifest_sha256"], "archive source manifest digest"
    )
    ancestry_sha = _valid_source_digest(
        fields["source_ancestry_sha256"], "archive source ancestry digest"
    )
    if _digest_bytes(manifest_path.read_bytes()) != manifest_sha:
        raise StageBIntrinsicReadinessError("archive source manifest file digest does not match")
    try:
        verify_manifest(root, manifest_path, manifest_sha)
    except (SourceManifestError, OSError, ValueError) as exc:
        raise StageBIntrinsicReadinessError(f"archive source verification failed: {exc}") from exc
    ancestry_path = root / ".source_ancestry.json"
    if (
        ancestry_path.is_symlink() or not ancestry_path.is_file()
        or _digest_bytes(ancestry_path.read_bytes()) != ancestry_sha
    ):
        raise StageBIntrinsicReadinessError("archive source ancestry is absent or mismatched")
    return {
        "kind": "git_archive",
        "revision": revision,
        "source_manifest_sha256": manifest_sha,
        "source_ancestry_sha256": ancestry_sha,
        "authoritative": True,
    }


def _source_identity(root: Path, *, require_authoritative: bool) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"], cwd=root,
        capture_output=True, text=True, check=False,
    )
    git_root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=root,
        capture_output=True, text=True, check=False,
    )
    is_exact_git_root = (
        git_root.returncode == 0
        and Path(git_root.stdout.strip()).expanduser().resolve() == root
    )
    if revision.returncode != 0 or not is_exact_git_root:
        identity = _archive_source_identity(root)
        if identity is not None:
            return identity
        if require_authoritative:
            raise StageBIntrinsicReadinessError(
                "authoritative readiness requires a clean Git checkout or verified archive source"
            )
        return {
            "kind": "test_fixture", "revision": "non-git-test-fixture",
            "source_manifest_sha256": None, "source_ancestry_sha256": None,
            "authoritative": False,
        }
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root,
        capture_output=True, text=True, check=False,
    )
    if status.returncode != 0:
        raise StageBIntrinsicReadinessError("cannot verify repository source state")
    if status.stdout.strip():
        raise StageBIntrinsicReadinessError(
            "intrinsic readiness requires a clean committed repository source"
        )
    value = revision.stdout.strip()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise StageBIntrinsicReadinessError("repository revision is not a full Git SHA-1")
    return {
        "kind": "git_checkout", "revision": value,
        "source_manifest_sha256": None, "source_ancestry_sha256": None,
        "authoritative": True,
    }


def _load_release_bindings(
    root: Path, authentication: Path, candidate_id: str, candidate_sha256: str,
    template_id: str, template_sha256: str, verification: Mapping[str, Any],
) -> dict[str, str]:
    release_path = authentication / "candidate-release.json"
    packet_path = authentication / "packet.sealed.json"
    policy_path = authentication / "authority-policy.json"
    release = _canonical_object(release_path, "candidate release")
    release_sha256 = _digest_bytes(release_path.read_bytes())
    if release.get("schema") != RELEASE_SCHEMA:
        raise StageBIntrinsicReadinessError("candidate release has the wrong schema")
    if release.get("template") != {"template_id": template_id, "sha256": template_sha256}:
        raise StageBIntrinsicReadinessError("candidate release does not echo the pinned template")
    if release.get("candidate") != {"candidate_id": candidate_id, "sha256": candidate_sha256}:
        raise StageBIntrinsicReadinessError("candidate release does not echo the pinned candidate")
    artifacts = release.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise StageBIntrinsicReadinessError("candidate release has no artifact bindings")
    packet_sha256 = _digest_bytes(packet_path.read_bytes())
    policy_sha256 = _digest_bytes(policy_path.read_bytes())
    if (
        artifacts.get("sealed_packet_sha256") != packet_sha256
        or artifacts.get("authority_policy_sha256") != policy_sha256
    ):
        raise StageBIntrinsicReadinessError("candidate release packet or policy identity mismatch")
    if verification.get("candidate_release_sha256") != release_sha256:
        raise StageBIntrinsicReadinessError("verifier release digest does not match the release file")
    if verification.get("packet_sha256") != packet_sha256:
        raise StageBIntrinsicReadinessError("verifier packet digest does not match the packet file")
    if verification.get("policy_sha256") != policy_sha256:
        raise StageBIntrinsicReadinessError("verifier policy digest does not match the policy file")
    return {
        "snr_candidate_release_path": _relative(root, release_path, "candidate release"),
        "snr_candidate_release_sha256": release_sha256,
        "snr_executable_packet_path": _relative(root, packet_path, "sealed packet"),
        "snr_executable_packet_sha256": packet_sha256,
        "snr_authority_policy_path": _relative(root, policy_path, "authority policy"),
        "snr_authority_policy_sha256": policy_sha256,
    }


def _parameter_document(
    candidate_id: str, candidate_sha256: str, candidate_parameters: Mapping[str, Any],
    references: Mapping[str, str], arm: str,
) -> dict[str, Any]:
    if set(references) != REFERENCE_KEYS:
        raise StageBIntrinsicReadinessError("authenticated references are partial or broadened")
    candidate = {"schema": CANDIDATE_SCHEMA, "candidate_id": candidate_id, "parameters": dict(candidate_parameters)}
    if _digest(candidate) != candidate_sha256:
        raise StageBIntrinsicReadinessError("candidate parameter identity does not match the pinned digest")
    return {
        "schema": PARAMETER_SCHEMA,
        "candidate_id": candidate_id,
        "candidate_sha256": candidate_sha256,
        "candidate_parameters": dict(candidate_parameters),
        "arm": arm,
        "arm_parameters": dict(references),
        "effective_parameters": {**dict(candidate_parameters), **dict(references)},
    }


def _write_once(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(canonical_bytes(value))
    except FileExistsError as exc:
        raise StageBIntrinsicReadinessError(f"refusing to replace existing artifact: {path}") from exc
    return _digest_bytes(path.read_bytes())


def _verify_runner_artifact(
    result: Mapping[str, Any], parameter_document: Mapping[str, Any],
    references: Mapping[str, str], arm: str, raw_path: Path, root: Path,
    analysis_protocol: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if result.get("schema") != RUNNER_SCHEMA or result.get("process_status") != "completed":
        raise StageBIntrinsicReadinessError(f"{arm} runner did not return a completed readiness artifact")
    if result.get("backend") != "numpy" or result.get("device") != "cpu":
        raise StageBIntrinsicReadinessError(f"{arm} runner did not report the required NumPy CPU backend")
    if result.get("readiness_only", {}).get("scientific_seed") is not None:
        raise StageBIntrinsicReadinessError(f"{arm} runner returned a scientific seed")
    expected_candidate = {
        "candidate_id": parameter_document["candidate_id"],
        "candidate_sha256": parameter_document["candidate_sha256"],
        "effective_parameters": parameter_document["effective_parameters"],
    }
    if result.get("adaptive_candidate") != expected_candidate:
        raise StageBIntrinsicReadinessError(f"{arm} runner candidate identity does not match")
    provenance = result.get("provenance")
    if not isinstance(provenance, Mapping):
        raise StageBIntrinsicReadinessError(f"{arm} runner returned no provenance")
    release = provenance.get("candidate_release")
    if not isinstance(release, Mapping) or (
        release.get("path") != references["snr_candidate_release_path"]
        or release.get("sha256") != references["snr_candidate_release_sha256"]
        or release.get("candidate_sha256") != parameter_document["candidate_sha256"]
    ):
        raise StageBIntrinsicReadinessError(f"{arm} runner release identity does not match")
    bindings = provenance.get("bindings")
    if not isinstance(bindings, list) or len(bindings) != 1:
        raise StageBIntrinsicReadinessError(f"{arm} runner must contain one runtime binding")
    binding = bindings[0]
    if not isinstance(binding, Mapping) or (
        binding.get("packet_path") != references["snr_executable_packet_path"]
        or binding.get("packet_file_sha256") != references["snr_executable_packet_sha256"]
        or binding.get("authority_policy_sha256") != references["snr_authority_policy_sha256"]
    ):
        raise StageBIntrinsicReadinessError(f"{arm} runner packet or policy identity does not match")
    if not raw_path.is_file() or raw_path.is_symlink():
        raise StageBIntrinsicReadinessError(f"{arm} runner did not write a regular raw artifact")
    try:
        persisted = json.loads(raw_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise StageBIntrinsicReadinessError(f"cannot reload {arm} raw artifact: {exc}") from exc
    if persisted != dict(result):
        raise StageBIntrinsicReadinessError(f"{arm} returned trace does not match persisted raw artifact")
    raw_protocol = result.get("raw_observation", {}).get("analysis_protocol")
    if analysis_protocol is None:
        if raw_protocol is not None:
            raise StageBIntrinsicReadinessError(f"{arm} runner injected an undeclared analysis protocol")
    elif (
        not isinstance(raw_protocol, Mapping)
        or raw_protocol.get("binding") != analysis_protocol
        or not isinstance(raw_protocol.get("termination"), Mapping)
    ):
        raise StageBIntrinsicReadinessError(f"{arm} runner protocol identity does not match")
    raw_observation = result.get("raw_observation")
    if not isinstance(raw_observation, Mapping):
        raise StageBIntrinsicReadinessError(f"{arm} runner returned no raw observation")
    trace_samples = len(raw_observation.get("time_s", []))
    compact_receipt: dict[str, str | int] | None = None
    compact_binding = raw_observation.get("compact_trace")
    if compact_binding is not None:
        if not isinstance(compact_binding, Mapping) or set(compact_binding) != {"path", "sha256"}:
            raise StageBIntrinsicReadinessError(f"{arm} runner compact trace binding is invalid")
        relative = PurePosixPath(str(compact_binding.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts or not relative.name:
            raise StageBIntrinsicReadinessError(f"{arm} runner compact trace path is not repository-relative")
        archive = root.joinpath(*relative.parts)
        if archive.is_symlink() or not archive.is_file():
            raise StageBIntrinsicReadinessError(f"{arm} runner compact trace is not a regular file")
        try:
            archive.resolve().relative_to(root)
            arrays = load_compact_trace(archive, expected_sha256=compact_binding["sha256"])
        except (CompactTraceError, OSError, TypeError, ValueError) as exc:
            raise StageBIntrinsicReadinessError(
                f"{arm} runner compact trace cannot be authenticated: {exc}"
            ) from exc
        trace_samples = len(arrays["time"])
        compact_receipt = {
            "path": _relative(root, archive, f"{arm} compact trace"),
            "sha256": str(compact_binding["sha256"]),
            "trace_samples": trace_samples,
        }
    if analysis_protocol is not None and compact_receipt is None:
        raise StageBIntrinsicReadinessError(f"{arm} production runner did not write a compact trace")
    if compact_receipt is not None and any(
        key in raw_observation for key in ("time_s", "voltage_mV", "spike_states")
    ):
        raise StageBIntrinsicReadinessError(f"{arm} compact trace duplicated inline vectors")
    receipt: dict[str, Any] = {
        "path": _relative(root, raw_path, f"{arm} raw artifact"),
        "sha256": _digest_bytes(raw_path.read_bytes()),
        "trace_samples": trace_samples,
    }
    if compact_receipt is not None:
        receipt["compact_trace"] = compact_receipt
    return receipt


def _write_sidecars(
    root: Path, *, repository_root: Path, candidate_sha256: str,
    source_identity: Mapping[str, Any], argv: Sequence[str], compact_artifacts: Sequence[Mapping[str, Any]],
) -> None:
    for artifact in sorted(root.rglob("*.json")):
        if artifact.name.endswith(".prov.json"):
            continue
        sidecar = artifact.with_name(f"{artifact.name}.prov.json")
        sidecar_document = {
            "schema": "v14-snr-stageB-intrinsic-readiness-provenance-v1",
            "artifact": _relative(repository_root, artifact, "provenance artifact"),
            "runner": "tools/v14_stageB_intrinsic_readiness.py",
            "argv": list(argv),
            "source_revision": source_identity["revision"],
            "source_identity": dict(source_identity),
            "backend": "numpy",
            "device": "cpu",
            "candidate_sha256": candidate_sha256,
            "scientific_seed": None,
            "scientific_verdict": None,
        }
        if compact_artifacts:
            sidecar_document["compact_trace_artifacts"] = [dict(item) for item in compact_artifacts]
        _write_once(sidecar, sidecar_document)


def run_intrinsic_readiness(
    template_path: str | Path, template_sha256: str,
    candidate_path: str | Path, candidate_sha256: str,
    causal_gate_path: str | Path, causal_gate_sha256: str,
    output_dir: str | Path, *, repository_root: str | Path,
    analysis_protocol_path: str | Path | None = None,
    analysis_protocol_sha256: str | None = None,
    execution_argv: Sequence[str] | None = None,
    require_authoritative_source: bool = False,
    expected_source_revision: str | None = None,
    expected_source_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Run exactly one pinned candidate through five authenticated arms and score it."""

    root = Path(repository_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise StageBIntrinsicReadinessError("repository_root must be a directory")
    _sha256(template_sha256, "template expected digest")
    _sha256(causal_gate_sha256, "causal gate expected digest")
    if (analysis_protocol_path is None) != (analysis_protocol_sha256 is None):
        raise StageBIntrinsicReadinessError(
            "analysis protocol path and sha256 must be supplied together"
        )
    template = _inside_root(template_path, root, "packet template")
    causal_gate = _inside_root(causal_gate_path, root, "causal gate packet")
    if not template.is_file() or template.is_symlink():
        raise StageBIntrinsicReadinessError("packet template must be a regular file")
    if not causal_gate.is_file() or causal_gate.is_symlink():
        raise StageBIntrinsicReadinessError("causal gate packet must be a regular file")
    template_document = _canonical_object(template, "packet template")
    gate_document = _canonical_object(causal_gate, "causal gate packet")
    if _digest_bytes(template.read_bytes()) != template_sha256:
        raise StageBIntrinsicReadinessError("packet template digest does not match")
    if _digest_bytes(causal_gate.read_bytes()) != causal_gate_sha256:
        raise StageBIntrinsicReadinessError("causal gate packet digest does not match")
    if gate_document.get("schema") not in {
        "v14-snr-stageB-causal-gates-v1",
        "v14-snr-stageB-causal-gates-v2",
    }:
        raise StageBIntrinsicReadinessError("causal gate packet has the wrong schema")
    analysis_protocol: dict[str, str] | None = None
    protocol_file: Path | None = None
    if analysis_protocol_path is not None and analysis_protocol_sha256 is not None:
        _sha256(analysis_protocol_sha256, "analysis protocol expected digest")
        protocol_file = _inside_root(analysis_protocol_path, root, "analysis protocol")
        if not protocol_file.is_file() or protocol_file.is_symlink():
            raise StageBIntrinsicReadinessError("analysis protocol must be a regular file")
        try:
            protocol_document = json.loads(protocol_file.read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise StageBIntrinsicReadinessError(
                f"cannot load analysis protocol: {exc}"
            ) from exc
        if not isinstance(protocol_document, dict):
            raise StageBIntrinsicReadinessError("analysis protocol must contain an object")
        if _digest_bytes(protocol_file.read_bytes()) != analysis_protocol_sha256:
            raise StageBIntrinsicReadinessError("analysis protocol digest does not match")
        if protocol_document.get("schema") not in {
            "v14-snr-stageB-intrinsic-protocol-v1",
            "v14-snr-stageB-intrinsic-protocol-v2",
        }:
            raise StageBIntrinsicReadinessError("analysis protocol has the wrong schema")
        analysis_protocol = {
            "path": _relative(root, protocol_file, "analysis protocol"),
            "sha256": analysis_protocol_sha256,
        }
        authority = protocol_document.get("causal_gate_authority")
        if not isinstance(authority, Mapping) or authority.get("path") != _relative(
            root, causal_gate, "causal gate packet"
        ):
            raise StageBIntrinsicReadinessError("analysis protocol names a different causal authority")
        if gate_document.get("authorized_analysis_protocol") != analysis_protocol:
            raise StageBIntrinsicReadinessError("causal gate does not authorize this analysis protocol")
    template_id = template_document.get("template_id")
    if not isinstance(template_id, str) or not template_id:
        raise StageBIntrinsicReadinessError("packet template has no valid template_id")
    source_identity = _source_identity(
        root, require_authoritative=require_authoritative_source,
    )
    if require_authoritative_source:
        if expected_source_revision is None or expected_source_manifest_sha256 is None:
            raise StageBIntrinsicReadinessError(
                "authoritative readiness requires expected revision and source manifest digests"
            )
        if source_identity.get("revision") != expected_source_revision:
            raise StageBIntrinsicReadinessError("source revision does not match the confirmation job")
        if source_identity.get("source_manifest_sha256") != expected_source_manifest_sha256:
            raise StageBIntrinsicReadinessError("source manifest does not match the confirmation job")
    argv = list(sys.argv if execution_argv is None else execution_argv)
    if not argv or any(not isinstance(item, str) or not item for item in argv):
        raise StageBIntrinsicReadinessError("execution_argv must contain nonempty text arguments")
    source, pinned_sha, candidate_document, candidate = _load_pinned_candidate(
        candidate_path, candidate_sha256, root
    )
    output = _inside_root(output_dir, root, "output directory")
    if output == root or output.exists():
        raise StageBIntrinsicReadinessError("output directory must be a new child of repository_root")
    if require_authoritative_source:
        runtime_root = (root / "research/experiment-runtime").resolve()
        try:
            output.relative_to(runtime_root)
        except ValueError as exc:
            raise StageBIntrinsicReadinessError(
                "authoritative readiness output must remain in research/experiment-runtime"
            ) from exc
    output.mkdir(parents=True)
    candidate_dir = output / pinned_sha
    authentication = candidate_dir / "authentication"
    try:
        compile_candidate(
            template, template_sha256, source, pinned_sha, authentication,
            repository_root=root,
        )
        verification = verify_candidate(
            template, template_sha256, authentication, repository_root=root
        )
        references = _load_release_bindings(
            root, authentication, candidate["candidate_id"], pinned_sha,
            template_id, template_sha256, verification,
        )
        runner_observations: dict[str, dict[str, str]] = {}
        arm_receipts: dict[str, dict[str, str | int]] = {}
        for arm in ARMS:
            arm_dir = candidate_dir / "arms" / arm
            parameter_path = arm_dir / "adaptive-parameters.json"
            raw_path = arm_dir / "raw-observation.json"
            parameter_document = _parameter_document(
                candidate["candidate_id"], pinned_sha, candidate["parameters"], references, arm
            )
            _write_once(parameter_path, parameter_document)
            result = run_readiness_arm(
                canonical_bytes(parameter_document).decode("ascii"),
                raw_path,
                repository_root=root,
                analysis_protocol_path=protocol_file,
                analysis_protocol_sha256=(
                    analysis_protocol["sha256"] if analysis_protocol is not None else None
                ),
                compact_trace=analysis_protocol is not None,
            )
            arm_receipt = _verify_runner_artifact(
                result, parameter_document, references, arm, raw_path, root,
                analysis_protocol=analysis_protocol,
            )
            runner_observations[arm] = {
                "path": str(arm_receipt["path"]),
                "sha256": str(arm_receipt["sha256"]),
            }
            arm_receipts[arm] = arm_receipt
        scorer_input = {
            "schema": INTRINSIC_LESION_SCHEMA,
            "readiness_only": dict(READINESS_ONLY),
            "causal_gate_packet": {
                "path": _relative(root, causal_gate, "causal gate packet"),
                "sha256": causal_gate_sha256,
            },
            "runner_observations": runner_observations,
        }
        scorer_input_path = candidate_dir / "intrinsic-lesion-observations.json"
        scorer_input_sha256 = _write_once(scorer_input_path, scorer_input)
        score = score_intrinsic_lesion_observations(scorer_input, root=root)
        if score.get("scientific_verdict") is not None:
            raise StageBIntrinsicReadinessError("intrinsic readiness scorer returned a scientific verdict")
        score_path = candidate_dir / "intrinsic-lesion-score.json"
        score_sha256 = _write_once(score_path, score)
        compact_artifacts = [
            {"arm": arm, **dict(arm_receipts[arm]["compact_trace"])}
            for arm in ARMS
            if "compact_trace" in arm_receipts[arm]
        ]
        _write_sidecars(
            candidate_dir, repository_root=root, candidate_sha256=pinned_sha,
            source_identity=source_identity, argv=argv,
            compact_artifacts=compact_artifacts,
        )
        exit_source_identity = _source_identity(
            root, require_authoritative=require_authoritative_source,
        )
        if exit_source_identity != source_identity:
            raise StageBIntrinsicReadinessError("source identity changed during readiness execution")
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "process_status": "completed",
            "scientific_verdict": None,
            "backend": "numpy",
            "device": "cpu",
            "provenance": {
                "runner": "tools/v14_stageB_intrinsic_readiness.py",
                "argv": argv,
                "source_revision": source_identity["revision"],
                "source_identity": source_identity,
                "source_verified_at_start": source_identity.get("authoritative") is True,
                "source_verified_at_exit": exit_source_identity.get("authoritative") is True,
            },
            "readiness_only": {
                **READINESS_ONLY,
                "scorer_invoked": True,
                "scientific_scoring": False,
                "claim": "engineering readiness transport only; no physiology claim",
            },
            "template": {
                "path": _relative(root, template, "packet template"),
                "sha256": template_sha256,
                "template_id": template_id,
            },
            "candidate_count": 1,
            "candidate": {
                "candidate_id": candidate["candidate_id"],
                "path": _relative(root, source, "candidate"),
                "sha256": pinned_sha,
                "directory": _relative(root, candidate_dir, "candidate directory"),
            },
            "authentication": {
                "compilation_directory": _relative(root, authentication, "authentication directory"),
                "release": {
                    "path": references["snr_candidate_release_path"],
                    "sha256": references["snr_candidate_release_sha256"],
                },
                "packet": {
                    "path": references["snr_executable_packet_path"],
                    "sha256": references["snr_executable_packet_sha256"],
                },
                "policy": {
                    "path": references["snr_authority_policy_path"],
                    "sha256": references["snr_authority_policy_sha256"],
            },
            },
            "arms": arm_receipts,
            "compact_trace_artifacts": compact_artifacts,
            "scorer_input": {
                "path": _relative(root, scorer_input_path, "scorer input"),
                "sha256": scorer_input_sha256,
            },
            "score": {
                "path": _relative(root, score_path, "score result"),
                "sha256": score_sha256,
                "readiness_contract_result": score["readiness_contract_result"],
                "all_intrinsic_lesion_gates_passed": score["all_intrinsic_lesion_gates_passed"],
            },
        }
        if analysis_protocol is not None:
            receipt["analysis_protocol"] = dict(analysis_protocol)
        receipt_path = output / "readiness-receipt.json"
        _write_once(receipt_path, receipt)
        _write_once(
            receipt_path.with_name("readiness-receipt.json.prov.json"),
            {
                "schema": "v14-snr-stageB-intrinsic-readiness-provenance-v1",
                "artifact": _relative(root, receipt_path, "receipt"),
                "runner": "tools/v14_stageB_intrinsic_readiness.py",
                "argv": argv,
                "source_revision": source_identity["revision"],
                "source_identity": source_identity,
                "source_verified_at_start": source_identity.get("authoritative") is True,
                "source_verified_at_exit": exit_source_identity.get("authoritative") is True,
                "backend": "numpy",
                "device": "cpu",
                "candidate_sha256": pinned_sha,
                "scientific_seed": None,
                "scientific_verdict": None,
                "compact_trace_artifacts": compact_artifacts,
            },
        )
        return receipt
    except Exception as exc:
        shutil.rmtree(output, ignore_errors=True)
        if isinstance(exc, StageBIntrinsicReadinessError):
            raise
        raise StageBIntrinsicReadinessError(f"intrinsic readiness failed: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--template-sha256", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--causal-gate-packet", required=True)
    parser.add_argument("--causal-gate-packet-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--analysis-protocol")
    parser.add_argument("--analysis-protocol-sha256")
    parser.add_argument("--require-authoritative-source", action="store_true")
    parser.add_argument("--expected-source-revision")
    parser.add_argument("--expected-source-manifest-sha256")
    args = parser.parse_args(argv)
    try:
        result = run_intrinsic_readiness(
            args.template, args.template_sha256, args.candidate, args.candidate_sha256,
            args.causal_gate_packet, args.causal_gate_packet_sha256, args.output_dir,
            repository_root=args.repository_root,
            analysis_protocol_path=args.analysis_protocol,
            analysis_protocol_sha256=args.analysis_protocol_sha256,
            require_authoritative_source=args.require_authoritative_source,
            expected_source_revision=args.expected_source_revision,
            expected_source_manifest_sha256=args.expected_source_manifest_sha256,
        )
    except (
        OSError, PacketError, StageBPacketCompilerError, StageBPacketVerifierError,
        StageBScorerError, StageBIntrinsicReadinessError, ValueError, TypeError,
    ) as exc:
        parser.exit(2, f"Stage B intrinsic readiness failure: {exc}\n")
    print(canonical_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
