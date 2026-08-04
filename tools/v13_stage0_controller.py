#!/usr/bin/env python3
"""Fail-closed external execution controller for Gate B V13 Stage 0.

The controller validates a frozen correction configuration and completed evidence,
then emits a create-only JSON command envelope.  It never executes the scientific
runner and never overrides a seed at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

try:
    from tools import execution_receipt
except ModuleNotFoundError:  # Direct ``python tools/...`` invocation.
    import execution_receipt  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "v13-stage0-controller-config-v1"
MANIFEST_SCHEMA = "v13-stage0-artifact-manifest-v1"
COMMAND_SCHEMA = "v13-stage0-command-v1"
MANIFEST_FIELDS = {
    "schema", "kind", "config_sha256", "source_revision", "artifact",
    "command_envelope", "execution_receipt", "sha256",
}
MANIFEST_ARTIFACT_FIELDS = {"path", "sha256"}
MANIFEST_COMMAND_REFERENCE_FIELDS = {"path", "sha256"}
MANIFEST_RECEIPT_REFERENCE_FIELDS = {
    "path", "sha256", "host", "device", "started_utc_ns", "ended_utc_ns",
}
COMMAND_FIELDS = {
    "schema", "action", "correction_id", "config", "source_revision", "cwd",
    "env", "argv", "output", "prerequisites", "execution",
}
MANIFEST_ACTIONS = {
    "calibration_numpy": "calibration_numpy",
    "calibration_cupy": "calibration_cupy",
    "calibration_selection": "merge_calibration",
    "replication_numpy": "replication_numpy",
    "replication_cupy": "replication_cupy",
    "held_out_cupy": "held_out_cupy",
    "held_out_numpy": "held_out_numpy",
    "performance_baseline": "performance_baseline",
    "performance_candidate": "performance_candidate",
    "final_stage0": "final_stage0_merge",
}
RUNNER_MODULE = "research.runners._vocal_action_credit_gate_v13_tonic_output"
SEED_SPEC_PATH = "research/specs/v13_tonic_output_substrate.json"
COMPATIBILITY_PATH = (
    "research/findings/raw/v13_deterministic_compatibility/"
    "comparison-baseline-vs-candidate.json"
)
CRITICAL_SOURCE_PATHS = (
    "research/runners/_vocal_action_credit_gate_v13_tonic_output.py",
    "sim/bridge.py",
    "sim/regions.py",
    "sim/kernels.py",
    SEED_SPEC_PATH,
)
OLD_CALIBRATION_SEED = 1013
OLD_REPLICATION_SEED = 1019
CALIBRATION_LADDER_PA = (75, 100, 125, 150, 175)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")


class ControllerError(RuntimeError):
    """A sealed prerequisite or state transition is invalid."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _canonical_digest(value: dict[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "sha256"}
    return hashlib.sha256(_canonical_bytes(body)).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ControllerError(f"{label} does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ControllerError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ControllerError(f"{label} must be a JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ControllerError(message)


def _require_digest(value: Any, label: str) -> str:
    _require(isinstance(value, str) and _HEX64.fullmatch(value) is not None,
             f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_revision(value: Any, label: str) -> str:
    _require(isinstance(value, str) and _REVISION.fullmatch(value) is not None,
             f"{label} must be a full lowercase 40-character Git revision")
    return value


def _repo_path(root: Path, value: Any, label: str) -> tuple[str, Path]:
    _require(isinstance(value, str) and value, f"{label} must be a repository-relative path")
    relative = Path(value)
    _require(not relative.is_absolute() and ".." not in relative.parts,
             f"{label} must be a safe repository-relative path")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    _require(resolved == resolved_root or resolved_root in resolved.parents,
             f"{label} escapes the source root")
    return relative.as_posix(), resolved


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode != 0:
        raise ControllerError(f"cannot resolve source revision for {root}: {result.stderr.strip()}")
    return result.stdout.strip()


def _revision_file_digest(root: Path, revision: str, relative: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{revision}:{relative}"], cwd=root,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", "replace").strip()
        raise ControllerError(f"cannot read {relative} from frozen revision: {message}")
    return hashlib.sha256(result.stdout).hexdigest()


def _artifact_paths(config: dict[str, Any], root: Path) -> dict[str, Path]:
    artifacts = config.get("artifacts")
    _require(isinstance(artifacts, dict), "config artifacts must be an object")
    required = {
        "calibration_numpy", "calibration_cupy", "calibration_selection",
        "replication_numpy", "replication_cupy", "held_out_cupy",
        "held_out_numpy", "performance_baseline", "performance_candidate",
        "final_stage0",
    }
    _require(set(artifacts) == required,
             f"config artifacts must contain exactly: {', '.join(sorted(required))}")
    paths: dict[str, Path] = {}
    seen: set[Path] = set()
    for kind, value in artifacts.items():
        _, path = _repo_path(root, value, f"artifacts.{kind}")
        _require(path not in seen, f"artifact destinations must be unique: {path}")
        seen.add(path)
        paths[kind] = path
    return paths


def _verify_source_binding(config: dict[str, Any], root: Path) -> None:
    expected_revision = _require_revision(
        config.get("candidate_source_revision"), "candidate_source_revision"
    )
    _require(_git_head(root) == expected_revision,
             "candidate source checkout is not at the frozen correction revision")
    source_identity = config["candidate_source_identity"]
    for relative in CRITICAL_SOURCE_PATHS:
        _, working_path = _repo_path(root, relative, f"candidate source {relative}")
        expected_digest = source_identity[relative]
        _require(working_path.is_file() and _file_digest(working_path) == expected_digest,
                 f"candidate working source has changed: {relative}")
        _require(_revision_file_digest(root, expected_revision, relative) == expected_digest,
                 f"candidate source is not bound to frozen revision: {relative}")
    binding = config.get("seed_binding")
    _require(isinstance(binding, dict), "seed_binding must be an object")
    _require(binding.get("path") == SEED_SPEC_PATH,
             f"seed_binding.path must be canonical: {SEED_SPEC_PATH}")
    _, spec_path = _repo_path(root, binding.get("path"), "seed_binding.path")
    expected_digest = _require_digest(binding.get("sha256"), "seed_binding.sha256")
    _require(spec_path.is_file() and _file_digest(spec_path) == expected_digest,
             "locked seed specification is missing or has the wrong digest")
    spec = _load_json(spec_path, "locked seed specification")
    expected = config["seeds"]
    partitions = spec.get("partitions")
    _require(isinstance(partitions, dict), "locked seed specification has no partitions object")
    for name in ("calibration", "replication", "held_out"):
        _require(partitions.get(name) == [expected[name]],
                 f"locked seed specification does not bind replacement {name} seed")


def load_config(path: Path, *, root: Path = ROOT, verify_source: bool = True) -> dict[str, Any]:
    config = _load_json(path, "correction config")
    _require(config.get("schema") == CONFIG_SCHEMA, f"unsupported config schema: {config.get('schema')!r}")
    _require(config.get("status") == "frozen", "correction config status must be frozen")
    supplied_digest = _require_digest(config.get("sha256"), "config sha256")
    _require(supplied_digest == _canonical_digest(config), "correction config self-digest is invalid")
    _require(isinstance(config.get("correction_id"), str) and config["correction_id"].strip(),
             "correction_id must be non-empty")
    _require(config.get("runner_module") == RUNNER_MODULE,
             f"runner_module must be {RUNNER_MODULE}")
    _require(isinstance(config.get("python"), str) and Path(config["python"]).is_absolute(),
             "python must be an absolute executable path")
    _require(Path(config["python"]).is_file() and os.access(config["python"], os.X_OK),
             "python executable is missing or not executable")
    _require_revision(config.get("candidate_source_revision"), "candidate_source_revision")
    source_identity = config.get("candidate_source_identity")
    _require(isinstance(source_identity, dict)
             and set(source_identity) == set(CRITICAL_SOURCE_PATHS),
             "candidate_source_identity must contain exactly the critical scientific paths")
    for relative, digest in source_identity.items():
        _require_digest(digest, f"candidate_source_identity.{relative}")

    seeds = config.get("seeds")
    _require(isinstance(seeds, dict) and set(seeds) == {"calibration", "replication", "held_out"},
             "seeds must contain exactly calibration, replication, and held_out")
    _require(all(type(seeds[name]) is int and seeds[name] >= 0 for name in seeds),
             "all Stage-0 seeds must be non-negative integers")
    _require(len(set(seeds.values())) == 3, "Stage-0 seeds must be distinct")
    _require(seeds["calibration"] != OLD_CALIBRATION_SEED,
             "calibration seed must replace consumed seed 1013")
    _require(seeds["replication"] != OLD_REPLICATION_SEED,
             "replication seed must replace consumed seed 1019")

    compatibility = config.get("compatibility")
    _require(isinstance(compatibility, dict), "compatibility must be an object")
    _require(compatibility.get("path") == COMPATIBILITY_PATH,
             f"compatibility.path must be canonical: {COMPATIBILITY_PATH}")
    compatibility_digest = _require_digest(compatibility.get("sha256"), "compatibility.sha256")
    _, compatibility_path = _repo_path(root, compatibility.get("path"), "compatibility.path")
    _require(compatibility_path.is_file() and _file_digest(compatibility_path) == compatibility_digest,
             "canonical compatibility artifact is missing or has the wrong digest")
    compatibility_artifact = _load_json(compatibility_path, "compatibility artifact")
    _require(compatibility_artifact.get("outcome") == "DETERMINISTIC_COMPATIBILITY_GO"
             and compatibility_artifact.get("go") is True,
             "canonical compatibility artifact has not earned GO")

    legacy = config.get("legacy_performance")
    _require(isinstance(legacy, dict), "legacy_performance must be an object")
    _require_revision(legacy.get("source_revision"), "legacy_performance.source_revision")
    _require(legacy.get("runner_path") == "research/runners/_vocal_action_credit_gate_v13_tonic_output.py",
             "legacy_performance.runner_path is not canonical")
    _require_digest(legacy.get("runner_sha256"), "legacy_performance.runner_sha256")
    _require(config["seed_binding"]["sha256"] == source_identity[SEED_SPEC_PATH],
             "seed binding digest differs from the frozen candidate source identity")
    _artifact_paths(config, root)
    if verify_source:
        _verify_source_binding(config, root)
    return config


def _manifest_digest(manifest: dict[str, Any]) -> str:
    return _canonical_digest(manifest)


def _expected_manifest_source(config: dict[str, Any], kind: str) -> str:
    if kind == "performance_baseline":
        return config["legacy_performance"]["source_revision"]
    return config["candidate_source_revision"]


def _expected_manifest_env(kind: str) -> dict[str, str]:
    if kind.endswith("_numpy"):
        return {"SIM_BACKEND": "numpy"}
    if kind in {"calibration_selection", "final_stage0"}:
        return {}
    return {"SIM_BACKEND": "cupy"}


def _expected_manifest_argv(
    *, config: dict[str, Any], kind: str, root: Path, output: Path,
) -> list[str]:
    paths = _artifact_paths(config, root)
    prefix = [config["python"], "-m", config["runner_module"]]
    if kind.startswith("calibration_") and kind != "calibration_selection":
        return [
            *prefix, "--calibration", "--compatibility-correction",
            str((root / COMPATIBILITY_PATH).resolve()), "--out", str(output),
        ]
    if kind == "calibration_selection":
        return [
            *prefix, "--merge-calibration", str(paths["calibration_numpy"]),
            str(paths["calibration_cupy"]), "--out", str(output),
        ]
    if kind.startswith("replication_"):
        return [
            *prefix, "--replication", str(paths["calibration_selection"]),
            "--out", str(output),
        ]
    if kind.startswith("held_out_"):
        return [
            *prefix, "--held-out", str(paths["calibration_selection"]),
            "--out", str(output),
        ]
    if kind == "performance_baseline":
        return [*prefix, "--legacy-performance-baseline", "--out", str(output)]
    if kind == "performance_candidate":
        return [
            *prefix, "--performance", "--old-baseline",
            str(paths["performance_baseline"]), "--out", str(output),
        ]
    compatibility = (root / COMPATIBILITY_PATH).resolve()
    return [
        *prefix, "--merge-final", str(compatibility),
        str(paths["replication_numpy"]), str(paths["replication_cupy"]),
        str(paths["held_out_cupy"]), str(paths["held_out_numpy"]),
        str(paths["performance_candidate"]), "--out", str(output),
    ]


def _validate_manifest_envelope(
    envelope: dict[str, Any], *, config: dict[str, Any], kind: str,
    root: Path, artifact_path: Path,
) -> Path:
    expected_fields = set(COMMAND_FIELDS)
    if kind == "final_stage0":
        expected_fields.add("expected_result")
    _require(set(envelope) == expected_fields,
             f"{kind} command envelope has missing or extra fields")
    _require(envelope.get("schema") == COMMAND_SCHEMA,
             f"{kind} command envelope schema is invalid")
    _require(envelope.get("action") == MANIFEST_ACTIONS[kind],
             f"{kind} command envelope action is invalid")
    _require(envelope.get("correction_id") == config["correction_id"],
             f"{kind} command envelope correction ID differs from config")
    config_ref = envelope.get("config")
    _require(isinstance(config_ref, dict) and set(config_ref) == {"path", "sha256"},
             f"{kind} command envelope config reference is invalid")
    _require(config_ref.get("sha256") == config["sha256"],
             f"{kind} command envelope config digest differs from config")
    config_path_value = config_ref.get("path")
    _require(isinstance(config_path_value, str) and Path(config_path_value).is_absolute(),
             f"{kind} command envelope config path must be absolute")
    try:
        config_path = Path(config_path_value).resolve(strict=True)
        config_path.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ControllerError(
            f"{kind} command envelope config path is outside the source root"
        ) from exc
    _require(_load_json(config_path, "envelope correction config") == config,
             f"{kind} command envelope names different config bytes")

    expected_source = _expected_manifest_source(config, kind)
    _require(envelope.get("source_revision") == expected_source,
             f"{kind} command envelope source revision is invalid")
    _require(envelope.get("execution") == "not_executed",
             f"{kind} command envelope execution marker is invalid")
    _require(isinstance(envelope.get("prerequisites"), list),
             f"{kind} command envelope prerequisites must be a list")
    cwd_value = envelope.get("cwd")
    _require(isinstance(cwd_value, str) and Path(cwd_value).is_absolute(),
             f"{kind} command envelope cwd must be absolute")
    try:
        cwd = Path(cwd_value).resolve(strict=True)
    except OSError as exc:
        raise ControllerError(f"{kind} command envelope cwd is invalid") from exc
    _require(cwd.is_dir(), f"{kind} command envelope cwd is not a directory")
    if kind != "performance_baseline":
        _require(cwd == root, f"{kind} command envelope cwd differs from source root")
    else:
        legacy = config["legacy_performance"]
        _, runner = _repo_path(cwd, legacy["runner_path"], "legacy runner path")
        _require(runner.is_file() and _file_digest(runner) == legacy["runner_sha256"],
                 "performance baseline command envelope has the wrong legacy runner")

    _require(envelope.get("output") == str(artifact_path),
             f"{kind} command envelope output differs from canonical artifact")
    expected_argv = _expected_manifest_argv(
        config=config, kind=kind, root=root, output=artifact_path
    )
    _require(envelope.get("argv") == expected_argv,
             f"{kind} command envelope argv differs from frozen command")
    _require(envelope.get("env") == _expected_manifest_env(kind),
             f"{kind} command envelope environment differs from frozen command")
    if kind == "final_stage0":
        _require(envelope.get("expected_result") == {
            "stage": "final_cross_backend", "outcome": "TONIC_OUTPUT_GO", "go": True,
        }, "final_stage0 command envelope expected result is invalid")
    return cwd


def load_manifest(
    path: Path,
    *,
    config: dict[str, Any],
    kind: str,
    root: Path = ROOT,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    _require(kind in MANIFEST_ACTIONS, f"unsupported manifest kind: {kind}")
    manifest_input = path if path.is_absolute() else root / path
    _require(manifest_input.is_file(), f"{kind} manifest does not exist: {path}")
    try:
        manifest_path = manifest_input.resolve(strict=True)
        manifest_path.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ControllerError(f"{kind} manifest path is outside the source root") from exc
    manifest = _load_json(manifest_path, f"{kind} manifest")
    _require(set(manifest) == MANIFEST_FIELDS,
             f"{kind} manifest has missing or extra fields")
    _require(manifest.get("schema") == MANIFEST_SCHEMA, f"{kind} manifest schema is invalid")
    _require(manifest.get("kind") == kind, f"manifest kind is not {kind}")
    _require(manifest.get("config_sha256") == config["sha256"],
             f"{kind} manifest is bound to a different correction config")
    expected_source = _expected_manifest_source(config, kind)
    _require(manifest.get("source_revision") == expected_source,
             f"{kind} manifest is bound to the wrong source revision")
    supplied_manifest_digest = _require_digest(manifest.get("sha256"), f"{kind} manifest sha256")
    _require(supplied_manifest_digest == _manifest_digest(manifest),
             f"{kind} manifest self-digest is invalid")
    artifact_ref = manifest.get("artifact")
    _require(isinstance(artifact_ref, dict) and set(artifact_ref) == MANIFEST_ARTIFACT_FIELDS,
             f"{kind} manifest artifact reference is invalid")
    expected_path = _artifact_paths(config, root)[kind]
    _, artifact_path = _repo_path(root, artifact_ref.get("path"), f"{kind} artifact path")
    _require(artifact_path == expected_path, f"{kind} manifest names a non-canonical artifact path")
    artifact_digest = _require_digest(artifact_ref.get("sha256"), f"{kind} artifact sha256")
    _require(artifact_path.is_file() and _file_digest(artifact_path) == artifact_digest,
             f"{kind} artifact is missing or its digest changed")

    envelope_ref = manifest.get("command_envelope")
    _require(
        isinstance(envelope_ref, dict)
        and set(envelope_ref) == MANIFEST_COMMAND_REFERENCE_FIELDS,
        f"{kind} command envelope reference is invalid",
    )
    envelope_relative, envelope_path = _repo_path(
        root, envelope_ref.get("path"), f"{kind} command envelope path"
    )
    envelope_digest = _require_digest(
        envelope_ref.get("sha256"), f"{kind} command envelope sha256"
    )
    _require(envelope_path.is_file() and _file_digest(envelope_path) == envelope_digest,
             f"{kind} command envelope is missing or its digest changed")
    envelope = _load_json(envelope_path, f"{kind} command envelope")
    cwd = _validate_manifest_envelope(
        envelope, config=config, kind=kind, root=root, artifact_path=artifact_path
    )
    _require(_file_digest(envelope_path) == envelope_digest,
             f"{kind} command envelope changed while being validated")

    receipt_ref = manifest.get("execution_receipt")
    _require(
        isinstance(receipt_ref, dict)
        and set(receipt_ref) == MANIFEST_RECEIPT_REFERENCE_FIELDS,
        f"{kind} execution receipt reference is invalid",
    )
    receipt_relative, receipt_path = _repo_path(
        cwd, receipt_ref.get("path"), f"{kind} execution receipt path"
    )
    receipt_digest = _require_digest(
        receipt_ref.get("sha256"), f"{kind} execution receipt sha256"
    )
    _require(receipt_path.is_file() and _file_digest(receipt_path) == receipt_digest,
             f"{kind} execution receipt is missing or its digest changed")
    host = receipt_ref.get("host")
    device = receipt_ref.get("device")
    _require(isinstance(host, str) and host.strip(),
             f"{kind} execution receipt host must be explicit")
    _require(isinstance(device, str) and device.strip(),
             f"{kind} execution receipt device must be explicit")
    started = receipt_ref.get("started_utc_ns")
    ended = receipt_ref.get("ended_utc_ns")
    _require(type(started) is int and type(ended) is int and started <= ended,
             f"{kind} execution receipt timestamps are invalid")
    try:
        receipt = execution_receipt.verify_receipt(cwd, receipt_relative)
    except execution_receipt.ReceiptError as exc:
        raise ControllerError(f"{kind} execution receipt is invalid: {exc}") from exc
    _require(_file_digest(receipt_path) == receipt_digest,
             f"{kind} execution receipt changed while being validated")
    _require(receipt.get("argv") == envelope["argv"],
             f"{kind} receipt argv differs from command envelope")
    _require(receipt.get("env_allowlist") == envelope["env"],
             f"{kind} receipt environment differs from command envelope")
    _require(receipt.get("source", {}).get("git_sha") == expected_source,
             f"{kind} receipt source revision is invalid")
    _require(
        receipt.get("host") == host and receipt.get("device") == device,
        f"{kind} receipt host or device differs from manifest",
    )
    _require(
        receipt.get("started_utc_ns") == started and receipt.get("ended_utc_ns") == ended,
        f"{kind} receipt timestamps differ from manifest",
    )
    receipt_artifact = receipt.get("artifact")
    _require(isinstance(receipt_artifact, dict), f"{kind} receipt artifact is invalid")
    _, receipt_artifact_path = _repo_path(
        cwd, receipt_artifact.get("path"), f"{kind} receipt artifact path"
    )
    _require(receipt_artifact_path == artifact_path,
             f"{kind} receipt names a non-canonical artifact")
    _require(receipt_artifact.get("sha256") == artifact_digest,
             f"{kind} receipt artifact digest differs from manifest")
    _require(_file_digest(artifact_path) == artifact_digest,
             f"{kind} artifact changed while evidence was being validated")
    artifact = _load_json(artifact_path, f"{kind} artifact")
    reference = {
        "kind": kind,
        "manifest_path": str(manifest_path),
        "manifest_sha256": supplied_manifest_digest,
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_digest,
        "command_envelope_path": envelope_relative,
        "command_envelope_sha256": envelope_digest,
        "execution_receipt_path": receipt_relative,
        "execution_receipt_sha256": receipt_digest,
    }
    return artifact, manifest, reference


def _compatibility_binding(artifact: dict[str, Any]) -> dict[str, Any]:
    value = artifact.get("compatibility_correction")
    _require(isinstance(value, dict), "calibration artifact lacks compatibility binding")
    return value


def _validate_calibration_backend(
    artifact: dict[str, Any], *, config: dict[str, Any], backend: str
) -> None:
    _require(artifact.get("stage") == "calibration_backend", "artifact is not a calibration backend result")
    _require(artifact.get("backend") == backend, f"calibration artifact backend is not {backend}")
    _require(artifact.get("seed") == config["seeds"]["calibration"],
             "calibration artifact used the wrong replacement seed")
    _require(artifact.get("source_sha") == config["candidate_source_revision"],
             "calibration artifact used the wrong source revision")
    _require(artifact.get("source_identity") == config["candidate_source_identity"],
             "calibration artifact source identities differ from the frozen config")
    _require(isinstance(artifact.get("spec_sha256"), str) and artifact["spec_sha256"],
             "calibration artifact lacks a spec digest")
    expected = config["compatibility"]
    binding = _compatibility_binding(artifact)
    _require(binding.get("path") == expected["path"] and binding.get("sha256") == expected["sha256"],
             "calibration artifact is not bound to the canonical compatibility artifact")
    rows = artifact.get("rows")
    _require(isinstance(rows, list) and len(rows) == 5,
             "calibration artifact does not contain the complete five-point ladder")
    _require(all(isinstance(row, dict) for row in rows),
             "calibration ladder rows must be objects")
    _require([row.get("current_pA") for row in rows] == [75, 100, 125, 150, 175],
             "calibration artifact does not contain the locked ordered current ladder")
    for row in rows:
        _require(isinstance(row.get("audit"), dict) and row["audit"].get("pass") in {True, False},
                 "calibration row lacks a completed population audit")
        _require(isinstance(row.get("physiology"), dict)
                 and row["physiology"].get("pass") in {True, False},
                 "calibration row lacks a completed physiology result")
        _require(row.get("pass") is (
            row["audit"]["pass"] is True and row["physiology"]["pass"] is True
        ), "calibration row verdict is inconsistent with its measured checks")
    earned_passing = [row["current_pA"] for row in rows if row["pass"]]
    _require(artifact.get("passing_currents_pA") == earned_passing,
             "calibration passing-current list disagrees with row verdicts")


def _validate_calibration_pair(
    numpy_artifact: dict[str, Any], cupy_artifact: dict[str, Any], config: dict[str, Any]
) -> None:
    _validate_calibration_backend(numpy_artifact, config=config, backend="numpy")
    _validate_calibration_backend(cupy_artifact, config=config, backend="cupy")
    for field in ("seed", "source_sha", "source_identity", "spec_sha256", "compatibility_correction"):
        _require(numpy_artifact.get(field) == cupy_artifact.get(field),
                 f"calibration backends disagree on {field}")


def _validate_selection(artifact: dict[str, Any], config: dict[str, Any]) -> None:
    _require(artifact.get("stage") == "calibration_cross_backend", "artifact is not a calibration selection")
    _require(artifact.get("seed") == config["seeds"]["calibration"],
             "calibration selection used the wrong replacement seed")
    _require(artifact.get("outcome") == "CALIBRATION_GO"
             and artifact.get("calibration_go") is True and artifact.get("go") is True,
             "calibration selection has not earned GO")
    _require(artifact.get("source_identity") == config["candidate_source_identity"],
             "calibration selection source identities differ from the frozen config")
    binding = _compatibility_binding(artifact)
    expected = config["compatibility"]
    _require(binding.get("path") == expected["path"] and binding.get("sha256") == expected["sha256"],
             "calibration selection is not bound to canonical compatibility evidence")
    _require(artifact.get("selected_current_pA") in CALIBRATION_LADDER_PA,
             "calibration selection is not on the locked current ladder")


def _selection_fingerprint(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_current_pA": artifact.get("selected_current_pA"),
        "source_identity": artifact.get("source_identity"),
        "compatibility_correction": artifact.get("compatibility_correction"),
    }


def _validate_stage_artifact(
    artifact: dict[str, Any], *, stage: str, backend: str,
    seed: int, config: dict[str, Any], selection: dict[str, Any],
) -> None:
    _require(artifact.get("stage") == stage, f"artifact is not a {stage} result")
    _require(artifact.get("backend") == backend, f"{stage} artifact backend is not {backend}")
    _require(artifact.get("seed") == seed, f"{stage} artifact used the wrong seed")
    _require(artifact.get("source_sha") == config["candidate_source_revision"],
             f"{stage} artifact used the wrong source revision")
    _require(artifact.get("go") is True and artifact.get("outcome") == f"{stage.upper()}_GO",
             f"{stage} artifact has not earned GO")
    _require(artifact.get("source_identity") == selection.get("source_identity"),
             f"{stage} artifact source identities differ from calibration")
    _require(artifact.get("selected_current_pA") == selection.get("selected_current_pA"),
             f"{stage} artifact selected current differs from calibration")
    embedded = artifact.get("selection")
    _require(isinstance(embedded, dict) and _selection_fingerprint(embedded) == _selection_fingerprint(selection),
             f"{stage} artifact embeds a different calibration selection")


def _validate_backend_pair(
    numpy_artifact: dict[str, Any], cupy_artifact: dict[str, Any], *,
    stage: str, seed: int, config: dict[str, Any], selection: dict[str, Any],
) -> None:
    _validate_stage_artifact(
        numpy_artifact, stage=stage, backend="numpy", seed=seed,
        config=config, selection=selection,
    )
    _validate_stage_artifact(
        cupy_artifact, stage=stage, backend="cupy", seed=seed,
        config=config, selection=selection,
    )
    for field in ("source_sha", "source_identity", "selected_current_pA"):
        _require(numpy_artifact.get(field) == cupy_artifact.get(field),
                 f"{stage} backends disagree on {field}")


def _require_earned_go(
    artifact: dict[str, Any], *, outcome: str, label: str,
) -> None:
    preconditions = artifact.get("preconditions")
    _require(
        artifact.get("go") is True
        and artifact.get("outcome") == outcome
        and artifact.get("verdict_status") == "GO"
        and isinstance(preconditions, list)
        and bool(preconditions)
        and all(isinstance(item, dict) and item.get("ok") is True for item in preconditions)
        and artifact.get("undefined_reasons") == [],
        f"{label} is missing, no-go, undefined, or lacks an earned {outcome} verdict",
    )


def _ensure_new_artifact(path: Path) -> None:
    _require(not path.exists(), f"refusing to target an existing artifact: {path}")


def _envelope(
    *, action: str, config_path: Path, config: dict[str, Any], root: Path,
    cwd: Path, argv: list[str], env: dict[str, str], output: Path,
    prerequisites: list[dict[str, Any]], source_revision: str | None = None,
) -> dict[str, Any]:
    return {
        "schema": COMMAND_SCHEMA,
        "action": action,
        "correction_id": config["correction_id"],
        "config": {"path": str(config_path.resolve()), "sha256": config["sha256"]},
        "source_revision": source_revision or config["candidate_source_revision"],
        "cwd": str(cwd.resolve()),
        "env": env,
        "argv": argv,
        "output": str(output),
        "prerequisites": prerequisites,
        "execution": "not_executed",
    }


def _emit_create_only(path: Path, envelope: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(envelope, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise ControllerError(f"refusing to overwrite command envelope: {path}") from exc
    with os.fdopen(descriptor, "w") as handle:
        handle.write(data)


def _runner_argv(config: dict[str, Any], *arguments: str) -> list[str]:
    return [config["python"], "-m", config["runner_module"], *arguments]


def emit_calibration(
    *, config_path: Path, backend: str, emit: Path,
    numpy_manifest: Path | None = None, root: Path = ROOT,
) -> dict[str, Any]:
    config = load_config(config_path, root=root)
    _require(backend in {"numpy", "cupy"}, "calibration backend must be numpy or cupy")
    prerequisites: list[dict[str, Any]] = [{
        "kind": "compatibility", "artifact_path": str((root / COMPATIBILITY_PATH).resolve()),
        "artifact_sha256": config["compatibility"]["sha256"],
    }]
    if backend == "cupy":
        _require(numpy_manifest is not None, "CuPy calibration requires a digested NumPy artifact")
        artifact, _, reference = load_manifest(
            numpy_manifest, config=config, kind="calibration_numpy", root=root
        )
        _validate_calibration_backend(artifact, config=config, backend="numpy")
        prerequisites.append(reference)
    else:
        _require(numpy_manifest is None, "NumPy calibration cannot consume a prior NumPy manifest")
    output = _artifact_paths(config, root)[f"calibration_{backend}"]
    _ensure_new_artifact(output)
    argv = _runner_argv(
        config, "--calibration", "--compatibility-correction",
        str((root / COMPATIBILITY_PATH).resolve()), "--out", str(output),
    )
    envelope = _envelope(
        action=f"calibration_{backend}", config_path=config_path, config=config,
        root=root, cwd=root, argv=argv, env={"SIM_BACKEND": backend},
        output=output, prerequisites=prerequisites,
    )
    _emit_create_only(emit, envelope)
    return envelope


def emit_merge_calibration(
    *, config_path: Path, numpy_manifest: Path, cupy_manifest: Path,
    emit: Path, root: Path = ROOT,
) -> dict[str, Any]:
    config = load_config(config_path, root=root)
    numpy_artifact, _, numpy_ref = load_manifest(
        numpy_manifest, config=config, kind="calibration_numpy", root=root
    )
    cupy_artifact, _, cupy_ref = load_manifest(
        cupy_manifest, config=config, kind="calibration_cupy", root=root
    )
    _validate_calibration_pair(numpy_artifact, cupy_artifact, config)
    output = _artifact_paths(config, root)["calibration_selection"]
    _ensure_new_artifact(output)
    argv = _runner_argv(
        config, "--merge-calibration", str(_artifact_paths(config, root)["calibration_numpy"]),
        str(_artifact_paths(config, root)["calibration_cupy"]), "--out", str(output),
    )
    envelope = _envelope(
        action="merge_calibration", config_path=config_path, config=config,
        root=root, cwd=root, argv=argv, env={}, output=output,
        prerequisites=[numpy_ref, cupy_ref],
    )
    _emit_create_only(emit, envelope)
    return envelope


def emit_replication(
    *, config_path: Path, backend: str, selection_manifest: Path,
    emit: Path, root: Path = ROOT,
) -> dict[str, Any]:
    config = load_config(config_path, root=root)
    _require(backend in {"numpy", "cupy"}, "replication backend must be numpy or cupy")
    selection, _, selection_ref = load_manifest(
        selection_manifest, config=config, kind="calibration_selection", root=root
    )
    _validate_selection(selection, config)
    output = _artifact_paths(config, root)[f"replication_{backend}"]
    _ensure_new_artifact(output)
    argv = _runner_argv(
        config, "--replication", str(_artifact_paths(config, root)["calibration_selection"]),
        "--out", str(output),
    )
    envelope = _envelope(
        action=f"replication_{backend}", config_path=config_path, config=config,
        root=root, cwd=root, argv=argv, env={"SIM_BACKEND": backend}, output=output,
        prerequisites=[selection_ref],
    )
    _emit_create_only(emit, envelope)
    return envelope


def emit_held_out(
    *, config_path: Path, backend: str, selection_manifest: Path,
    replication_numpy_manifest: Path, replication_cupy_manifest: Path,
    emit: Path, cupy_held_out_manifest: Path | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    config = load_config(config_path, root=root)
    _require(backend in {"numpy", "cupy"}, "held-out backend must be numpy or cupy")
    selection, _, selection_ref = load_manifest(
        selection_manifest, config=config, kind="calibration_selection", root=root
    )
    _validate_selection(selection, config)
    repl_numpy, _, repl_numpy_ref = load_manifest(
        replication_numpy_manifest, config=config, kind="replication_numpy", root=root
    )
    repl_cupy, _, repl_cupy_ref = load_manifest(
        replication_cupy_manifest, config=config, kind="replication_cupy", root=root
    )
    _validate_backend_pair(
        repl_numpy, repl_cupy, stage="replication",
        seed=config["seeds"]["replication"], config=config, selection=selection,
    )
    prerequisites = [selection_ref, repl_numpy_ref, repl_cupy_ref]
    if backend == "numpy":
        _require(cupy_held_out_manifest is not None,
                 "NumPy held-out requires a completed digested CuPy held-out GO")
        held_cupy, _, held_cupy_ref = load_manifest(
            cupy_held_out_manifest, config=config, kind="held_out_cupy", root=root
        )
        _validate_stage_artifact(
            held_cupy, stage="held_out", backend="cupy",
            seed=config["seeds"]["held_out"], config=config, selection=selection,
        )
        prerequisites.append(held_cupy_ref)
    else:
        _require(cupy_held_out_manifest is None,
                 "CuPy held-out cannot consume a prior held-out artifact")
    output = _artifact_paths(config, root)[f"held_out_{backend}"]
    _ensure_new_artifact(output)
    argv = _runner_argv(
        config, "--held-out", str(_artifact_paths(config, root)["calibration_selection"]),
        "--out", str(output),
    )
    envelope = _envelope(
        action=f"held_out_{backend}", config_path=config_path, config=config,
        root=root, cwd=root, argv=argv, env={"SIM_BACKEND": backend}, output=output,
        prerequisites=prerequisites,
    )
    _emit_create_only(emit, envelope)
    return envelope


def emit_performance_baseline(
    *, config_path: Path, source_root: Path, emit: Path,
    root: Path = ROOT,
) -> dict[str, Any]:
    config = load_config(config_path, root=root, verify_source=False)
    legacy = config["legacy_performance"]
    _require(_git_head(source_root) == legacy["source_revision"],
             "legacy performance checkout is not at the exact required old revision")
    _, runner = _repo_path(source_root, legacy["runner_path"], "legacy runner path")
    _require(runner.is_file() and _file_digest(runner) == legacy["runner_sha256"],
             "legacy performance runner is missing or has the wrong digest")
    output = _artifact_paths(config, root)["performance_baseline"]
    _ensure_new_artifact(output)
    argv = _runner_argv(config, "--legacy-performance-baseline", "--out", str(output))
    envelope = _envelope(
        action="performance_baseline", config_path=config_path, config=config,
        root=root, cwd=source_root, argv=argv, env={"SIM_BACKEND": "cupy"},
        output=output, prerequisites=[{
            "kind": "legacy_source", "source_revision": legacy["source_revision"],
            "runner_path": str(runner), "runner_sha256": legacy["runner_sha256"],
        }], source_revision=legacy["source_revision"],
    )
    _emit_create_only(emit, envelope)
    return envelope


def _validate_baseline(artifact: dict[str, Any], config: dict[str, Any]) -> None:
    legacy = config["legacy_performance"]
    _require(artifact.get("stage") == "legacy_performance_baseline",
             "baseline artifact has the wrong stage")
    _require(artifact.get("outcome") == "BASELINE_RECORDED",
             "legacy performance baseline was not completed")
    _require(artifact.get("source_sha") == legacy["source_revision"],
             "baseline artifact used the wrong old source revision")
    _require(artifact.get("backend") == "cupy" and "3090" in str(artifact.get("device", "")),
             "baseline artifact was not produced on the RTX 3090 CuPy lane")
    _require(isinstance(artifact.get("median_seconds"), (int, float))
             and artifact["median_seconds"] > 0,
             "baseline artifact lacks a positive median duration")


def emit_performance_candidate(
    *, config_path: Path, baseline_manifest: Path, selection_manifest: Path,
    held_out_cupy_manifest: Path, held_out_numpy_manifest: Path,
    emit: Path, root: Path = ROOT,
) -> dict[str, Any]:
    config = load_config(config_path, root=root)
    baseline, _, baseline_ref = load_manifest(
        baseline_manifest, config=config, kind="performance_baseline", root=root
    )
    _validate_baseline(baseline, config)
    selection, _, selection_ref = load_manifest(
        selection_manifest, config=config, kind="calibration_selection", root=root
    )
    _validate_selection(selection, config)
    held_cupy, _, held_cupy_ref = load_manifest(
        held_out_cupy_manifest, config=config, kind="held_out_cupy", root=root
    )
    held_numpy, _, held_numpy_ref = load_manifest(
        held_out_numpy_manifest, config=config, kind="held_out_numpy", root=root
    )
    _validate_backend_pair(
        held_numpy, held_cupy, stage="held_out", seed=config["seeds"]["held_out"],
        config=config, selection=selection,
    )
    output = _artifact_paths(config, root)["performance_candidate"]
    _ensure_new_artifact(output)
    argv = _runner_argv(
        config, "--performance", "--old-baseline",
        str(_artifact_paths(config, root)["performance_baseline"]), "--out", str(output),
    )
    envelope = _envelope(
        action="performance_candidate", config_path=config_path, config=config,
        root=root, cwd=root, argv=argv, env={"SIM_BACKEND": "cupy"}, output=output,
        prerequisites=[baseline_ref, selection_ref, held_cupy_ref, held_numpy_ref],
    )
    _emit_create_only(emit, envelope)
    return envelope


def _validate_performance_candidate(
    artifact: dict[str, Any], manifest: dict[str, Any], *,
    config: dict[str, Any], selection: dict[str, Any],
    selection_reference: dict[str, Any], root: Path,
) -> None:
    _require(artifact.get("stage") == "performance",
             "performance candidate artifact has the wrong stage")
    _require_earned_go(
        artifact, outcome="PERFORMANCE_GO", label="performance candidate",
    )
    _require(artifact.get("backend") == "cupy"
             and "3090" in str(artifact.get("device", "")),
             "performance candidate was not measured on the RTX 3090 CuPy lane")
    _require(artifact.get("source_sha") == config["candidate_source_revision"],
             "performance candidate used the wrong source revision")
    _require(artifact.get("source_identity") == config["candidate_source_identity"],
             "performance candidate source identities differ from the frozen config")
    baseline = artifact.get("old_baseline")
    _require(isinstance(baseline, dict),
             "performance candidate does not embed its legacy baseline")
    _validate_baseline(baseline, config)
    baseline_path = artifact.get("old_baseline_artifact")
    _require(isinstance(baseline_path, str) and baseline_path,
             "performance candidate lacks its baseline artifact path")
    supplied_baseline = Path(baseline_path)
    if not supplied_baseline.is_absolute():
        supplied_baseline = root / supplied_baseline
    _require(supplied_baseline.resolve() == _artifact_paths(config, root)["performance_baseline"],
             "performance candidate names a non-canonical baseline artifact")

    _, command_path = _repo_path(
        root, manifest["command_envelope"]["path"],
        "performance command envelope path",
    )
    command = _load_json(command_path, "performance command envelope")
    prerequisites = command.get("prerequisites")
    _require(
        isinstance(prerequisites, list) and selection_reference in prerequisites,
        "performance command envelope is not bound to the selected calibration manifest",
    )


def emit_final_merge(
    *, config_path: Path, selection_manifest: Path,
    replication_numpy_manifest: Path, replication_cupy_manifest: Path,
    held_out_cupy_manifest: Path, held_out_numpy_manifest: Path,
    performance_manifest: Path, emit: Path, root: Path = ROOT,
) -> dict[str, Any]:
    config = load_config(config_path, root=root)
    compatibility_path = (root / config["compatibility"]["path"]).resolve()
    compatibility = _load_json(compatibility_path, "canonical compatibility artifact")
    _require_earned_go(
        compatibility, outcome="DETERMINISTIC_COMPATIBILITY_GO",
        label="canonical compatibility artifact",
    )
    compatibility_ref = {
        "kind": "compatibility",
        "artifact_path": str(compatibility_path),
        "artifact_sha256": config["compatibility"]["sha256"],
    }

    selection, _, selection_ref = load_manifest(
        selection_manifest, config=config, kind="calibration_selection", root=root
    )
    _validate_selection(selection, config)
    _require_earned_go(selection, outcome="CALIBRATION_GO", label="calibration selection")

    repl_numpy, _, repl_numpy_ref = load_manifest(
        replication_numpy_manifest, config=config, kind="replication_numpy", root=root
    )
    repl_cupy, _, repl_cupy_ref = load_manifest(
        replication_cupy_manifest, config=config, kind="replication_cupy", root=root
    )
    _validate_backend_pair(
        repl_numpy, repl_cupy, stage="replication",
        seed=config["seeds"]["replication"], config=config, selection=selection,
    )
    _require_earned_go(repl_numpy, outcome="REPLICATION_GO", label="NumPy replication")
    _require_earned_go(repl_cupy, outcome="REPLICATION_GO", label="CuPy replication")

    held_cupy, _, held_cupy_ref = load_manifest(
        held_out_cupy_manifest, config=config, kind="held_out_cupy", root=root
    )
    held_numpy, _, held_numpy_ref = load_manifest(
        held_out_numpy_manifest, config=config, kind="held_out_numpy", root=root
    )
    _validate_backend_pair(
        held_numpy, held_cupy, stage="held_out", seed=config["seeds"]["held_out"],
        config=config, selection=selection,
    )
    _require_earned_go(held_cupy, outcome="HELD_OUT_GO", label="CuPy held-out")
    _require_earned_go(held_numpy, outcome="HELD_OUT_GO", label="NumPy held-out")

    performance, performance_seal, performance_ref = load_manifest(
        performance_manifest, config=config, kind="performance_candidate", root=root
    )
    _validate_performance_candidate(
        performance, performance_seal, config=config, selection=selection,
        selection_reference=selection_ref, root=root,
    )

    paths = _artifact_paths(config, root)
    output = paths["final_stage0"]
    _ensure_new_artifact(output)
    argv = _runner_argv(
        config, "--merge-final", str(compatibility_path),
        str(paths["replication_numpy"]), str(paths["replication_cupy"]),
        str(paths["held_out_cupy"]), str(paths["held_out_numpy"]),
        str(paths["performance_candidate"]), "--out", str(output),
    )
    envelope = _envelope(
        action="final_stage0_merge", config_path=config_path, config=config,
        root=root, cwd=root, argv=argv, env={}, output=output,
        prerequisites=[
            compatibility_ref, selection_ref, repl_numpy_ref, repl_cupy_ref,
            held_cupy_ref, held_numpy_ref, performance_ref,
        ],
    )
    envelope["expected_result"] = {
        "stage": "final_cross_backend",
        "outcome": "TONIC_OUTPUT_GO",
        "go": True,
    }
    _emit_create_only(emit, envelope)
    return envelope


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--emit", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)

    calibration = commands.add_parser("calibration")
    calibration.add_argument("--backend", choices=("numpy", "cupy"), required=True)
    calibration.add_argument("--numpy-manifest", type=Path)

    merge = commands.add_parser("merge-calibration")
    merge.add_argument("--numpy-manifest", type=Path, required=True)
    merge.add_argument("--cupy-manifest", type=Path, required=True)

    replication = commands.add_parser("replication")
    replication.add_argument("--backend", choices=("numpy", "cupy"), required=True)
    replication.add_argument("--selection-manifest", type=Path, required=True)

    held = commands.add_parser("held-out")
    held.add_argument("--backend", choices=("numpy", "cupy"), required=True)
    held.add_argument("--selection-manifest", type=Path, required=True)
    held.add_argument("--replication-numpy-manifest", type=Path, required=True)
    held.add_argument("--replication-cupy-manifest", type=Path, required=True)
    held.add_argument("--cupy-held-out-manifest", type=Path)

    baseline = commands.add_parser("performance-baseline")
    baseline.add_argument("--source-root", type=Path, required=True)

    performance = commands.add_parser("performance-candidate")
    performance.add_argument("--baseline-manifest", type=Path, required=True)
    performance.add_argument("--selection-manifest", type=Path, required=True)
    performance.add_argument("--held-out-cupy-manifest", type=Path, required=True)
    performance.add_argument("--held-out-numpy-manifest", type=Path, required=True)

    final = commands.add_parser("final-merge")
    final.add_argument("--selection-manifest", type=Path, required=True)
    final.add_argument("--replication-numpy-manifest", type=Path, required=True)
    final.add_argument("--replication-cupy-manifest", type=Path, required=True)
    final.add_argument("--held-out-cupy-manifest", type=Path, required=True)
    final.add_argument("--held-out-numpy-manifest", type=Path, required=True)
    final.add_argument("--performance-manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    try:
        if args.command == "calibration":
            envelope = emit_calibration(
                config_path=args.config, backend=args.backend, emit=args.emit,
                numpy_manifest=args.numpy_manifest, root=root,
            )
        elif args.command == "merge-calibration":
            envelope = emit_merge_calibration(
                config_path=args.config, numpy_manifest=args.numpy_manifest,
                cupy_manifest=args.cupy_manifest, emit=args.emit, root=root,
            )
        elif args.command == "replication":
            envelope = emit_replication(
                config_path=args.config, backend=args.backend,
                selection_manifest=args.selection_manifest, emit=args.emit, root=root,
            )
        elif args.command == "held-out":
            envelope = emit_held_out(
                config_path=args.config, backend=args.backend,
                selection_manifest=args.selection_manifest,
                replication_numpy_manifest=args.replication_numpy_manifest,
                replication_cupy_manifest=args.replication_cupy_manifest,
                cupy_held_out_manifest=args.cupy_held_out_manifest,
                emit=args.emit, root=root,
            )
        elif args.command == "performance-baseline":
            envelope = emit_performance_baseline(
                config_path=args.config, source_root=args.source_root.resolve(),
                emit=args.emit, root=root,
            )
        elif args.command == "performance-candidate":
            envelope = emit_performance_candidate(
                config_path=args.config, baseline_manifest=args.baseline_manifest,
                selection_manifest=args.selection_manifest,
                held_out_cupy_manifest=args.held_out_cupy_manifest,
                held_out_numpy_manifest=args.held_out_numpy_manifest,
                emit=args.emit, root=root,
            )
        else:
            envelope = emit_final_merge(
                config_path=args.config,
                selection_manifest=args.selection_manifest,
                replication_numpy_manifest=args.replication_numpy_manifest,
                replication_cupy_manifest=args.replication_cupy_manifest,
                held_out_cupy_manifest=args.held_out_cupy_manifest,
                held_out_numpy_manifest=args.held_out_numpy_manifest,
                performance_manifest=args.performance_manifest,
                emit=args.emit, root=root,
            )
    except ControllerError as exc:
        print(f"v13-stage0-controller: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "action": envelope["action"], "command": str(args.emit),
        "output": envelope["output"], "execution": "not_executed",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
