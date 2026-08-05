"""Run the seed-free two-candidate V14 Stage B readiness transport check.

This module is an engineering orchestrator.  It compiles and independently
authorizes exactly two pinned candidates, executes each through the real
readiness-only NumPy runner, verifies identity echoes, and publishes one
top-level receipt.  It does not score traces or make a physiology claim.
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

from research.runners.v14_stageB_physiology import run_readiness_intact
from sim.snr_executable_packet import PacketError, canonical_bytes
from tools.v14_stageB_packet_compiler import StageBPacketCompilerError, compile_candidate
from tools.v14_stageB_packet_verifier import StageBPacketVerifierError, verify_candidate


RECEIPT_SCHEMA = "v14-snr-stageB-real-readiness-v1"
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


class StageBRealReadinessError(ValueError):
    """The two-candidate readiness transport check could not complete."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise StageBRealReadinessError(f"cannot read {path}: {exc}") from exc


def _digest(value: Any) -> str:
    return _sha256_bytes(canonical_bytes(value))


def _sha256(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StageBRealReadinessError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _contains_seed(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            "seed" in str(key).lower() or _contains_seed(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_seed(item) for item in value)
    return False


def _inside_root(path: str | Path, root: Path, context: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise StageBRealReadinessError(
            f"{context} must be inside repository_root"
        ) from exc
    return resolved


def _canonical_object(path: Path, context: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise StageBRealReadinessError(f"cannot load {context}: {exc}") from exc
    if not isinstance(document, dict):
        raise StageBRealReadinessError(f"{context} must contain an object")
    try:
        canonical = canonical_bytes(document)
    except PacketError as exc:
        raise StageBRealReadinessError(f"{context} is not canonical JSON: {exc}") from exc
    if raw != canonical:
        raise StageBRealReadinessError(f"{context} is not canonical JSON")
    return document


def _relative(root: Path, path: Path, context: str) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise StageBRealReadinessError(f"{context} must be inside repository_root") from exc
    rendered = PurePosixPath(*relative.parts).as_posix()
    if not rendered or rendered in {".", ".."} or any(
        part in {"", ".", ".."} for part in PurePosixPath(rendered).parts
    ):
        raise StageBRealReadinessError(f"{context} is not a canonical repository-relative path")
    return rendered


def _load_pinned_candidate(
    path: str | Path, expected_sha256: str, root: Path, index: int
) -> tuple[Path, str, dict[str, Any]]:
    _sha256(expected_sha256, f"candidate {index} expected digest")
    source = _inside_root(path, root, f"candidate {index}")
    if not source.is_file() or source.is_symlink():
        raise StageBRealReadinessError(f"candidate {index} must be a regular file")
    document = _canonical_object(source, f"candidate {index}")
    digest = _sha256_file(source)
    if digest != expected_sha256:
        raise StageBRealReadinessError(f"candidate {index} digest does not match")
    if _contains_seed(document):
        raise StageBRealReadinessError(f"candidate {index} contains seed data")
    return source, digest, document


def _numeric_candidate_parameters(document: Mapping[str, Any], index: int) -> tuple[str, dict[str, Any]]:
    if set(document) != {"schema", "candidate_id", "parameters"}:
        raise StageBRealReadinessError(f"candidate {index} has an invalid declaration shape")
    if document.get("schema") != "sim-adaptive-candidate-v1":
        raise StageBRealReadinessError(f"candidate {index} has the wrong schema")
    candidate_id = document.get("candidate_id")
    parameters = document.get("parameters")
    if not isinstance(candidate_id, str) or not candidate_id or candidate_id != candidate_id.strip():
        raise StageBRealReadinessError(f"candidate {index} has an invalid candidate_id")
    if not isinstance(parameters, Mapping) or not parameters:
        raise StageBRealReadinessError(f"candidate {index} parameters must be a nonempty object")
    numeric: dict[str, Any] = {}
    for key, value in parameters.items():
        if not isinstance(key, str) or not key or key in _REFERENCE_KEYS:
            raise StageBRealReadinessError(
                f"candidate {index} has a broadened or reserved parameter declaration"
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise StageBRealReadinessError(
                f"candidate {index} parameters must contain only numeric values"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise StageBRealReadinessError(f"candidate {index} contains a non-finite parameter")
        numeric[key] = value
    return candidate_id, numeric


def _new_output_root(output_dir: str | Path, root: Path) -> Path:
    destination = _inside_root(output_dir, root, "output directory")
    if destination == root:
        raise StageBRealReadinessError("output directory must be a new child of repository_root")
    if destination.exists():
        raise StageBRealReadinessError("output directory must not already exist")
    return destination


def _source_revision(root: Path) -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"], cwd=root,
        capture_output=True, text=True, check=False,
    )
    if revision.returncode != 0:
        return "non-git-test-fixture"
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root,
        capture_output=True, text=True, check=False,
    )
    if status.returncode != 0:
        raise StageBRealReadinessError("cannot verify repository source state")
    if status.stdout.strip():
        raise StageBRealReadinessError(
            "real readiness requires a clean committed repository source"
        )
    value = revision.stdout.strip()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise StageBRealReadinessError("repository revision is not a full Git SHA-1")
    return value


def _load_release_and_bindings(
    root: Path,
    candidate_dir: Path,
    candidate_id: str,
    candidate_sha256: str,
    template_id: str,
    template_sha256: str,
    verification: Mapping[str, Any],
) -> dict[str, str]:
    release_path = candidate_dir / "candidate-release.json"
    packet_path = candidate_dir / "packet.sealed.json"
    policy_path = candidate_dir / "authority-policy.json"
    release = _canonical_object(release_path, "candidate release")
    release_sha256 = _sha256_file(release_path)
    if release.get("schema") != "v14-snr-stageB-candidate-release-v1":
        raise StageBRealReadinessError("candidate release has the wrong schema")
    if release.get("template") != {"template_id": template_id, "sha256": template_sha256}:
        raise StageBRealReadinessError("candidate release does not echo the pinned template")
    if release.get("candidate") != {"candidate_id": candidate_id, "sha256": candidate_sha256}:
        raise StageBRealReadinessError("candidate release does not echo the candidate identity")
    artifacts = release.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise StageBRealReadinessError("candidate release has no artifact bindings")
    expected_packet_sha256 = _sha256_file(packet_path)
    expected_policy_sha256 = _sha256_file(policy_path)
    if (
        artifacts.get("sealed_packet_sha256") != expected_packet_sha256
        or artifacts.get("authority_policy_sha256") != expected_policy_sha256
    ):
        raise StageBRealReadinessError("candidate release packet or policy identity mismatch")
    if verification.get("candidate_release_sha256") != release_sha256:
        raise StageBRealReadinessError("verifier release digest does not match the release file")
    if verification.get("packet_sha256") != expected_packet_sha256:
        raise StageBRealReadinessError("verifier packet digest does not match the packet file")
    if verification.get("policy_sha256") != expected_policy_sha256:
        raise StageBRealReadinessError("verifier policy digest does not match the policy file")
    return {
        "snr_candidate_release_path": _relative(root, release_path, "candidate release"),
        "snr_candidate_release_sha256": release_sha256,
        "snr_executable_packet_path": _relative(root, packet_path, "sealed packet"),
        "snr_executable_packet_sha256": expected_packet_sha256,
        "snr_authority_policy_path": _relative(root, policy_path, "authority policy"),
        "snr_authority_policy_sha256": expected_policy_sha256,
    }


def _parameter_document(
    candidate_id: str,
    candidate_sha256: str,
    candidate_parameters: Mapping[str, Any],
    arm_parameters: Mapping[str, str],
) -> dict[str, Any]:
    if set(arm_parameters) != _REFERENCE_KEYS:
        raise StageBRealReadinessError("readiness arm references are partial or broadened")
    candidate = {
        "schema": "sim-adaptive-candidate-v1",
        "candidate_id": candidate_id,
        "parameters": dict(candidate_parameters),
    }
    if _digest(candidate) != candidate_sha256:
        raise StageBRealReadinessError("candidate parameter identity does not match the pinned digest")
    return {
        "schema": "sim-adaptive-run-parameters-v1",
        "candidate_id": candidate_id,
        "candidate_sha256": candidate_sha256,
        "candidate_parameters": dict(candidate_parameters),
        "arm": READINESS_ARM,
        "arm_parameters": dict(arm_parameters),
        "effective_parameters": {**dict(candidate_parameters), **dict(arm_parameters)},
    }


def _write_provenance_sidecars(
    candidate_dir: Path, *, candidate_sha256: str, argv: Sequence[str]
) -> None:
    provenance = {
        "schema": "v14-snr-stageB-real-readiness-provenance-v1",
        "runner": "tools/v14_stageB_real_readiness.py",
        "argv": list(argv),
        "backend": "numpy",
        "device": "cpu",
        "candidate_sha256": candidate_sha256,
        "scientific_seed": None,
        "scientific_verdict": None,
    }
    for artifact in sorted(candidate_dir.glob("*.json")):
        if artifact.name.endswith((".prov.json", ".cmd.json", ".provenance.json")):
            continue
        sidecar = artifact.with_name(f"{artifact.name}.prov.json")
        with sidecar.open("xb") as handle:
            handle.write(canonical_bytes(provenance))


def _verify_observation(
    result: Mapping[str, Any],
    parameter_document: Mapping[str, Any],
    references: Mapping[str, str],
    candidate_id: str,
    candidate_sha256: str,
    observation_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    if result.get("schema") != "v14-snr-stageB-physiology-observation-v1":
        raise StageBRealReadinessError("readiness runner returned the wrong observation schema")
    if result.get("process_status") != "completed":
        raise StageBRealReadinessError("readiness runner did not complete")
    if result.get("scientific_verdict") is not None:
        raise StageBRealReadinessError("readiness observation must not contain a scientific verdict")
    if result.get("readiness_only", {}).get("scientific_seed") is not None:
        raise StageBRealReadinessError("readiness observation contains a scientific seed")
    echo = result.get("adaptive_candidate")
    expected_echo = {
        "candidate_id": candidate_id,
        "candidate_sha256": candidate_sha256,
        "effective_parameters": parameter_document["effective_parameters"],
    }
    if echo != expected_echo:
        raise StageBRealReadinessError("runner candidate identity echo does not match the parameter document")
    provenance = result.get("provenance")
    if not isinstance(provenance, Mapping):
        raise StageBRealReadinessError("readiness observation has no provenance")
    release_echo = provenance.get("candidate_release")
    if not isinstance(release_echo, Mapping) or (
        release_echo.get("path") != references["snr_candidate_release_path"]
        or release_echo.get("sha256") != references["snr_candidate_release_sha256"]
        or release_echo.get("candidate_sha256") != candidate_sha256
    ):
        raise StageBRealReadinessError("runner candidate-release identity echo does not match")
    bindings = provenance.get("bindings")
    if not isinstance(bindings, list) or len(bindings) != 1:
        raise StageBRealReadinessError("readiness observation must contain one runtime binding")
    binding = bindings[0]
    if not isinstance(binding, Mapping) or (
        binding.get("packet_path") != references["snr_executable_packet_path"]
        or binding.get("packet_file_sha256") != references["snr_executable_packet_sha256"]
        or binding.get("authority_policy_sha256") != references["snr_authority_policy_sha256"]
    ):
        raise StageBRealReadinessError("runner packet or policy identity echo does not match")
    if not observation_path.is_file() or observation_path.is_symlink():
        raise StageBRealReadinessError("readiness runner did not write a regular observation")
    try:
        persisted = json.loads(observation_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise StageBRealReadinessError(f"cannot reload raw observation: {exc}") from exc
    if persisted != dict(result):
        raise StageBRealReadinessError("returned trace does not match persisted raw observation")
    return {
        "path": _relative(repository_root, observation_path, "raw observation"),
        "sha256": _sha256_file(observation_path),
        "trace_samples": len(result.get("raw_observation", {}).get("time_s", [])),
    }


def run_real_readiness(
    template_path: str | Path,
    template_sha256: str,
    candidate_a_path: str | Path,
    candidate_a_sha256: str,
    candidate_b_path: str | Path,
    candidate_b_sha256: str,
    output_dir: str | Path,
    *,
    repository_root: str | Path,
    execution_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run exactly two isolated, authenticated, seed-free readiness arms."""

    root = Path(repository_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise StageBRealReadinessError("repository_root must be a directory")
    _sha256(template_sha256, "template expected digest")
    template = _inside_root(template_path, root, "packet template")
    if not template.is_file() or template.is_symlink():
        raise StageBRealReadinessError("packet template must be a regular file")
    template_document = _canonical_object(template, "packet template")
    if _sha256_file(template) != template_sha256:
        raise StageBRealReadinessError("packet template digest does not match")
    template_id = template_document.get("template_id")
    if not isinstance(template_id, str) or not template_id:
        raise StageBRealReadinessError("packet template has no valid template_id")
    destination = _new_output_root(output_dir, root)
    source_revision = _source_revision(root)
    argv = list(sys.argv if execution_argv is None else execution_argv)
    if not argv or any(not isinstance(item, str) or not item for item in argv):
        raise StageBRealReadinessError("execution_argv must contain nonempty text arguments")
    candidates = [
        _load_pinned_candidate(candidate_a_path, candidate_a_sha256, root, 1),
        _load_pinned_candidate(candidate_b_path, candidate_b_sha256, root, 2),
    ]
    if candidates[0][1] == candidates[1][1]:
        raise StageBRealReadinessError("the two candidates must have distinct pinned digests")
    if candidates[0][0] == candidates[1][0]:
        raise StageBRealReadinessError("the two candidates must have distinct pinned paths")
    candidate_descriptors = []
    for index, (source, candidate_sha256, source_document) in enumerate(candidates, 1):
        candidate_id, candidate_parameters = _numeric_candidate_parameters(source_document, index)
        candidate_descriptors.append((source, candidate_sha256, candidate_id, candidate_parameters))
    if candidate_descriptors[0][2] == candidate_descriptors[1][2]:
        raise StageBRealReadinessError("the two candidates must have distinct candidate identities")

    destination.mkdir(parents=True)
    completed: list[dict[str, Any]] = []
    try:
        for source, candidate_sha256, candidate_id, candidate_parameters in candidate_descriptors:
            candidate_dir = destination / candidate_sha256
            compile_candidate(
                template, template_sha256, source, candidate_sha256, candidate_dir,
                repository_root=root,
            )
            verification = verify_candidate(
                template, template_sha256, candidate_dir, repository_root=root
            )
            references = _load_release_and_bindings(
                root, candidate_dir, candidate_id, candidate_sha256,
                template_id, template_sha256, verification,
            )
            parameter_document = _parameter_document(
                candidate_id, candidate_sha256, candidate_parameters, references
            )
            parameter_path = candidate_dir / "adaptive-parameters.json"
            with parameter_path.open("xb") as handle:
                handle.write(canonical_bytes(parameter_document))
            observation_path = candidate_dir / "raw-observation.json"
            result = run_readiness_intact(
                canonical_bytes(parameter_document).decode("ascii"),
                observation_path,
                repository_root=root,
            )
            observation = _verify_observation(
                result, parameter_document, references, candidate_id, candidate_sha256,
                observation_path, root,
            )
            completed.append({
                "candidate_id": candidate_id,
                "candidate_sha256": candidate_sha256,
                "candidate_path": _relative(root, source, "candidate"),
                "compilation_directory": _relative(root, candidate_dir, "candidate directory"),
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
                "adaptive_parameter_document": {
                    "path": _relative(root, parameter_path, "adaptive parameter document"),
                    "sha256": _sha256_file(parameter_path),
                },
                "observation": observation,
            })
            _write_provenance_sidecars(
                candidate_dir, candidate_sha256=candidate_sha256, argv=argv
            )
    except Exception as exc:
        shutil.rmtree(destination, ignore_errors=True)
        if isinstance(exc, StageBRealReadinessError):
            raise
        raise StageBRealReadinessError(f"two-candidate readiness failed: {exc}") from exc

    if len(completed) != 2:
        shutil.rmtree(destination, ignore_errors=True)
        raise StageBRealReadinessError("readiness receipt requires two completed candidates")
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "process_status": "completed",
        "scientific_verdict": None,
        "backend": "numpy",
        "device": "cpu",
        "provenance": {
            "runner": "tools/v14_stageB_real_readiness.py",
            "argv": argv,
            "template_sha256": template_sha256,
            "source_revision": source_revision,
        },
        "readiness_only": {
            "enabled": True,
            "reserved_seed_count": 0,
            "scientific_seed": None,
            "scored": False,
            "claim": "engineering transport only; no physiology claim",
        },
        "template": {
            "path": _relative(root, template, "packet template"),
            "sha256": template_sha256,
            "template_id": template_id,
        },
        "candidate_count": 2,
        "candidate_isolation": "one output directory named by each pinned candidate SHA-256",
        "candidates": completed,
    }
    receipt_path = destination / "readiness-receipt.json"
    try:
        with receipt_path.open("xb") as handle:
            handle.write(canonical_bytes(receipt))
    except OSError as exc:
        shutil.rmtree(destination, ignore_errors=True)
        raise StageBRealReadinessError(f"cannot write final readiness receipt: {exc}") from exc
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--template-sha256", required=True)
    parser.add_argument("--candidate-a", required=True)
    parser.add_argument("--candidate-a-sha256", required=True)
    parser.add_argument("--candidate-b", required=True)
    parser.add_argument("--candidate-b-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repository-root", required=True)
    args = parser.parse_args(argv)
    try:
        result = run_real_readiness(
            args.template, args.template_sha256,
            args.candidate_a, args.candidate_a_sha256,
            args.candidate_b, args.candidate_b_sha256,
            args.output_dir, repository_root=args.repository_root,
        )
    except (
        OSError, PacketError, StageBPacketCompilerError, StageBPacketVerifierError,
        StageBRealReadinessError, ValueError, TypeError,
    ) as exc:
        parser.exit(2, f"Stage B real readiness failure: {exc}\n")
    print(canonical_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
