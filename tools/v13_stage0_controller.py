#!/usr/bin/env python3
"""Fail-closed external execution controller for Gate B V13 Stage 0.

The controller validates a frozen correction configuration and completed evidence,
then emits a create-only JSON command envelope.  It never executes the scientific
runner and never overrides a seed at runtime.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from typing import Any

try:
    from tools import execution_receipt
    from tools import stable_json_evidence
except ModuleNotFoundError:  # Direct ``python tools/...`` invocation.
    import execution_receipt  # type: ignore[no-redef]
    import stable_json_evidence  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "v13-stage0-controller-config-v3"
MANIFEST_SCHEMA = "v13-stage0-artifact-manifest-v2"
COMMAND_SCHEMA = "v13-stage0-command-v1"
READINESS_SCHEMA = "v13-stage0-readiness-v1"
CONFIG_FIELDS = {
    "schema", "status", "correction_id", "candidate_source_revision",
    "candidate_source_identity", "candidate_source_manifest", "python",
    "runner_module", "seeds", "seed_derivation", "seed_binding",
    "strict_arithmetic_replay", "compatibility", "legacy_performance",
    "artifacts", "sha256",
}
MANIFEST_FIELDS = {
    "schema", "kind", "config_sha256", "source_revision", "artifact",
    "command_envelope", "execution_receipt", "provenance_sidecar",
    "controller_config", "process_correction_spec", "candidate_source_manifest",
    "compatibility", "sha256",
}
MANIFEST_ARTIFACT_FIELDS = {"path", "sha256"}
MANIFEST_CONFIG_REFERENCE_FIELDS = {"path", "file_sha256", "canonical_sha256"}
MANIFEST_SOURCE_REFERENCE_FIELDS = {
    "path", "sha256", "tree_sha256", "file_count",
}
MANIFEST_COMPATIBILITY_FIELDS = {
    "path", "file_sha256", "canonical_json_sha256", "canonicalization",
}
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
BASE_SPEC_PATH = "research/specs/v13_tonic_output_substrate.json"
SEED_SPEC_PATH = (
    "research/specs/v13_tonic_output_stage0_process_correction_v2.json"
)
COMPATIBILITY_PATH = (
    "research/findings/raw/v13_deterministic_compatibility/"
    "comparison-baseline-vs-candidate.json"
)
STRICT_REPLAY_PATH = (
    "research/findings/raw/"
    "v13_backend_neutral_izh_arithmetic_replay_diagnostic_v2/"
    "evidence-manifest.json"
)
CRITICAL_SOURCE_PATHS = (
    "research/runners/_vocal_action_credit_gate_v13_tonic_output.py",
    "sim/bridge.py",
    "sim/regions.py",
    "sim/kernels.py",
    BASE_SPEC_PATH,
    SEED_SPEC_PATH,
)
SOURCE_CLOSURE_ROOTS = (
    *CRITICAL_SOURCE_PATHS[:-2],
    "tools/execution_receipt.py",
    "tools/v13_stage0_controller.py",
    "tools/v13_stage0_manifest.py",
)
SOURCE_CLOSURE_DATA_PATHS = (BASE_SPEC_PATH, SEED_SPEC_PATH)
REQUIRED_SOURCE_MANIFEST_PATHS = tuple(sorted(set(
    SOURCE_CLOSURE_ROOTS + SOURCE_CLOSURE_DATA_PATHS
)))
REPLAY_RUNNER_MODULE = (
    "research.runners._v13_backend_neutral_izh_arithmetic_replay_v2"
)
REPLAY_SENSITIVE_PREFIX = "sim/"
FORBIDDEN_CONSUMED_SEEDS = frozenset((1013, 1019, 840860))
RETIRED_UNEXECUTED_SEEDS = frozenset((687979,))
PRIOR_PARTITION_SEEDS = {"calibration": 840860, "replication": 687979}
LOCKED_HELD_OUT_SEED = 1021
SEED_DERIVATION_ALGORITHM = "sha256-first-12-mod-900000-plus-100000-v2"
SEED_DERIVATION_NAMESPACE = "V13_STAGE0_PROCESS_CORRECTION_V2"
SEED_DERIVATION_SOURCE_REVISION = "d091fa6692bdf8115c8073af6fd31fc9626921a8"
SEED_DERIVATION_SOURCE_COMMITTED_AT = "2026-08-04T05:45:13-04:00"
SEED_DERIVATION_SOURCE_RELATION = "fixed_before_observation"
SEED_DERIVATION_RESULT_EXCLUSION = (
    "no measured result, current, verdict, raster, state hash, or tested "
    "candidate is an input"
)
PROCESS_CORRECTION_SCHEMA = "v13-stage0-process-correction-v2"
PROCESS_CORRECTION_STATUS = "preregistered-not-executed"
COMPATIBILITY_CANONICALIZATION = "python-json-sort-keys-compact-separators-utf8-v1"
CALIBRATION_LADDER_PA = (75, 100, 125, 150, 175)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")


class ControllerError(RuntimeError):
    """A sealed prerequisite or state transition is invalid."""


class FrozenConfig(dict[str, Any]):
    """Validated config value with its exact file identity kept out of JSON."""

    def __init__(self, value: dict[str, Any], *, path: Path, file_sha256: str):
        super().__init__(value)
        self.path = path
        self.file_sha256 = file_sha256


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


def _load_json_evidence(
    path: Path, label: str,
) -> stable_json_evidence.StableJsonEvidence:
    try:
        return stable_json_evidence.read_stable_json_evidence(
            path, require_object=True
        )
    except stable_json_evidence.StableJsonEvidenceError as exc:
        raise ControllerError(f"cannot read {label}: {path}: {exc}") from exc


def _load_json(path: Path, label: str) -> dict[str, Any]:
    return _load_json_evidence(path, label).value


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
    relative = PurePosixPath(value)
    _require(
        not relative.is_absolute()
        and bool(relative.name)
        and "." not in relative.parts
        and ".." not in relative.parts,
             f"{label} must be a safe repository-relative path")
    resolved_root = root.resolve(strict=True)
    candidate = resolved_root.joinpath(*relative.parts)
    current = resolved_root
    for part in relative.parts:
        current = current / part
        if not os.path.lexists(current):
            continue
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise ControllerError(f"cannot inspect {label}: {current}: {exc}") from exc
        _require(not stat.S_ISLNK(mode), f"{label} cannot contain a symlink")
    return relative.as_posix(), candidate


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


def _module_source_paths(root: Path, module: str) -> tuple[str, ...]:
    """Return local files Python may execute while importing one module."""
    if not module or any(not part.isidentifier() for part in module.split(".")):
        return ()
    parts = module.split(".")
    found: set[str] = set()
    for depth in range(1, len(parts) + 1):
        package = root.joinpath(*parts[:depth], "__init__.py")
        if package.is_file():
            found.add(package.relative_to(root).as_posix())
    module_file = root.joinpath(*parts).with_suffix(".py")
    if module_file.is_file():
        found.add(module_file.relative_to(root).as_posix())
    return tuple(sorted(found))


def _imported_local_sources(root: Path, relative: str) -> tuple[str, ...]:
    path = root / relative
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise ControllerError(f"cannot inspect candidate Python source {relative}: {exc}") from exc

    module_parts = relative.removesuffix(".py").split("/")
    package_parts = module_parts if module_parts[-1] == "__init__" else module_parts[:-1]
    if package_parts[-1:] == ["__init__"]:
        package_parts = package_parts[:-1]

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            keep = len(package_parts) - (node.level - 1)
            if keep < 0:
                continue
            base_parts = package_parts[:keep]
            if node.module:
                base_parts.extend(node.module.split("."))
            base = ".".join(base_parts)
        else:
            base = node.module or ""
        if base:
            modules.add(base)
        for alias in node.names:
            if alias.name != "*":
                modules.add(".".join(part for part in (base, alias.name) if part))

    imported: set[str] = set()
    for module in modules:
        imported.update(_module_source_paths(root, module))
    return tuple(sorted(imported))


def _required_candidate_source_paths(root: Path) -> tuple[str, ...]:
    """Compute the exact conservative local import closure for Stage-0 execution."""
    root = root.resolve(strict=True)
    closure = set(SOURCE_CLOSURE_DATA_PATHS)
    closure.update(
        path.relative_to(root).as_posix()
        for path in (root / "sim").rglob("*.py")
        if path.is_file()
    )
    pending = list(SOURCE_CLOSURE_ROOTS)
    while pending:
        relative = pending.pop()
        if relative in closure:
            continue
        path = root / relative
        _require(path.is_file(), f"required candidate source is missing: {relative}")
        closure.add(relative)
        if path.suffix == ".py":
            pending.extend(
                imported for imported in _imported_local_sources(root, relative)
                if imported not in closure
            )
    for relative in SOURCE_CLOSURE_DATA_PATHS:
        _require((root / relative).is_file(), f"required candidate data is missing: {relative}")
    return tuple(sorted(closure))


def _parse_sha256_manifest(path: Path, label: str) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ControllerError(f"cannot read {label}: {path}: {exc}") from exc
    _require(bool(lines), f"{label} is empty")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(lines, 1):
        digest, separator, relative_text = line.partition("  ")
        relative = PurePosixPath(relative_text)
        valid = (
            bool(separator) and _HEX64.fullmatch(digest) is not None
            and not relative.is_absolute() and bool(relative.name)
            and "." not in relative.parts and ".." not in relative.parts
        )
        _require(valid, f"{label} has an invalid entry on line {line_number}")
        normalized = relative.as_posix()
        _require(normalized not in entries, f"{label} has duplicate entry: {normalized}")
        entries[normalized] = digest
    return entries


def _derive_replacement_seed(*, role: str, prior_seed: int) -> int:
    material = (
        f"{SEED_DERIVATION_NAMESPACE}|{SEED_DERIVATION_SOURCE_REVISION}|"
        f"role={role}|prior_seed={prior_seed}"
    )
    prefix = hashlib.sha256(material.encode("ascii")).hexdigest()[:12]
    return 100000 + (int(prefix, 16) % 900000)


def _expected_seed_derivation() -> dict[str, Any]:
    return {
        "algorithm": SEED_DERIVATION_ALGORITHM,
        "namespace": SEED_DERIVATION_NAMESPACE,
        "source_anchor": {
            "revision": SEED_DERIVATION_SOURCE_REVISION,
            "committed_at": SEED_DERIVATION_SOURCE_COMMITTED_AT,
            "relation_to_v1_observation": SEED_DERIVATION_SOURCE_RELATION,
        },
        "material_template": (
            "{namespace}|{source_anchor_revision}|role={role}|prior_seed={prior_seed}"
        ),
        "prior_partition_seeds": PRIOR_PARTITION_SEEDS,
        "result_exclusion": SEED_DERIVATION_RESULT_EXCLUSION,
    }


def _validate_seed_derivation(config: dict[str, Any]) -> None:
    derivation = config.get("seed_derivation")
    _require(
        derivation == _expected_seed_derivation(),
        "seed_derivation differs from the locked result-independent derivation",
    )
    seeds = config["seeds"]
    for role, prior in PRIOR_PARTITION_SEEDS.items():
        expected = _derive_replacement_seed(role=role, prior_seed=prior)
        _require(seeds[role] == expected,
                 f"{role} seed is not the mechanically derived replacement")


def _validate_source_manifest_binding(
    config: dict[str, Any], root: Path, revision: str,
) -> dict[str, Any]:
    binding = config.get("candidate_source_manifest")
    _require(
        isinstance(binding, dict)
        and set(binding) == {"path", "sha256", "tree_sha256", "file_count"},
        "candidate_source_manifest must contain exactly path, sha256, "
        "tree_sha256, and file_count",
    )
    relative, manifest_path = _repo_path(
        root, binding.get("path"), "candidate_source_manifest.path"
    )
    expected_manifest = _require_digest(
        binding.get("sha256"), "candidate_source_manifest.sha256"
    )
    expected_tree = _require_digest(
        binding.get("tree_sha256"), "candidate_source_manifest.tree_sha256"
    )
    _require(type(binding.get("file_count")) is int and binding["file_count"] > 0,
             "candidate_source_manifest.file_count must be a positive integer")
    try:
        source = execution_receipt.verify_source_manifest(root, relative)
    except execution_receipt.ReceiptError as exc:
        raise ControllerError(f"candidate source manifest is invalid: {exc}") from exc
    _require(source["manifest_sha256"] == expected_manifest,
             "candidate source manifest digest differs from frozen config")
    _require(source["tree_sha256"] == expected_tree,
             "candidate source tree digest differs from frozen config")
    _require(source["file_count"] == binding["file_count"],
             "candidate source manifest file count differs from frozen config")
    _require(_revision_file_digest(root, revision, relative) == expected_manifest,
             "candidate source manifest is not bound to frozen revision")
    files = source["files"]
    required = set(_required_candidate_source_paths(root))
    _require(
        set(files) == required,
        "candidate source manifest must exactly match the deterministic local "
        "import and data closure",
    )
    for source_relative, metadata in files.items():
        _require(
            _revision_file_digest(root, revision, source_relative)
            == metadata["sha256"],
            f"candidate source manifest entry is not bound to frozen revision: "
            f"{source_relative}",
        )
    _require(_file_digest(manifest_path) == expected_manifest,
             "candidate source manifest changed while being validated")
    return source


def _historical_replay_source(
    root: Path, source: Any, revision: str,
) -> dict[str, str]:
    _require(
        isinstance(source, dict)
        and set(source) == {
            "file_count", "git_sha", "kind", "manifest",
            "manifest_sha256", "tree_sha256",
        }
        and source.get("git_sha") == revision
        and source.get("kind") == "git",
        "strict arithmetic replay source binding is invalid",
    )
    relative, manifest_path = _repo_path(
        root, source.get("manifest"), "strict arithmetic replay source manifest"
    )
    manifest_sha = _require_digest(
        source.get("manifest_sha256"), "strict replay source manifest sha256"
    )
    tree_sha = _require_digest(source.get("tree_sha256"), "strict replay source tree sha256")
    _require(
        manifest_path.is_file() and _file_digest(manifest_path) == manifest_sha,
        "strict arithmetic replay source manifest is missing or has the wrong digest",
    )
    entries = _parse_sha256_manifest(manifest_path, "strict replay source manifest")
    _require(
        type(source.get("file_count")) is int
        and source["file_count"] == len(entries),
        "strict arithmetic replay source file count is invalid",
    )
    tree = hashlib.sha256()
    for source_relative, digest in sorted(entries.items()):
        tree.update(f"{digest}  {source_relative}\n".encode("utf-8"))
        _require(
            _revision_file_digest(root, revision, source_relative) == digest,
            "strict arithmetic replay source manifest is not bound to its frozen "
            f"revision: {source_relative}",
        )
    _require(tree.hexdigest() == tree_sha,
             "strict arithmetic replay source tree digest is invalid")
    _require(relative == source["manifest"],
             "strict arithmetic replay source manifest path is not canonical")
    return entries


def _validate_replay_receipt(
    root: Path, *, receipt_path: Any, source: dict[str, Any],
    artifact_path: str, artifact_sha256: str, mode: str,
    backend: str | None = None, receipt_sha256: str | None = None,
) -> None:
    relative, path = _repo_path(root, receipt_path, "strict replay receipt path")
    if receipt_sha256 is not None:
        _require(
            _file_digest(path) == _require_digest(receipt_sha256, "strict replay receipt sha256"),
            "strict replay receipt has the wrong digest",
        )
    receipt = _load_json(path, "strict replay receipt")
    _require(
        receipt.get("schema") == execution_receipt.SCHEMA
        and receipt.get("status") == "success"
        and receipt.get("exit_code") == 0
        and receipt.get("source") == source,
        "strict replay receipt does not prove a successful source-bound execution",
    )
    artifact = receipt.get("artifact")
    _, expected_artifact = _repo_path(root, artifact_path, "strict replay artifact path")
    _require(
        isinstance(artifact, dict)
        and artifact.get("path") == artifact_path
        and artifact.get("sha256") == artifact_sha256
        and type(artifact.get("size_bytes")) is int
        and expected_artifact.is_file()
        and expected_artifact.stat().st_size == artifact["size_bytes"]
        and _file_digest(expected_artifact) == artifact_sha256,
        "strict replay receipt does not bind its artifact",
    )
    argv = receipt.get("argv")
    output_argument = Path(argv[-1]) if isinstance(argv, list) and argv else Path()
    artifact_parts = Path(artifact_path).parts
    output_parts = output_argument.parts
    _require(
        isinstance(argv, list) and len(argv) >= 4
        and argv[1:3] == ["-m", REPLAY_RUNNER_MODULE]
        and mode in argv
        and argv[-2] == "--out"
        and output_argument.is_absolute()
        and len(output_parts) >= len(artifact_parts)
        and output_parts[-len(artifact_parts):] == artifact_parts,
        "strict replay receipt command differs from the frozen replay workflow",
    )
    expected_env = {"SIM_BACKEND": backend or "numpy"}
    _require(receipt.get("env_allowlist") == expected_env,
             "strict replay receipt environment is invalid")
    _require(relative == receipt_path, "strict replay receipt path is not canonical")


def _validate_strict_replay_binding(
    config: dict[str, Any], root: Path, candidate_source: dict[str, Any],
) -> None:
    binding = config.get("strict_arithmetic_replay")
    _require(
        isinstance(binding, dict)
        and set(binding) == {"path", "sha256", "source_revision"},
        "strict_arithmetic_replay must contain exactly path, sha256, and source_revision",
    )
    _require(binding.get("path") == STRICT_REPLAY_PATH,
             f"strict_arithmetic_replay.path must be canonical: {STRICT_REPLAY_PATH}")
    replay_revision = _require_revision(
        binding.get("source_revision"), "strict_arithmetic_replay.source_revision"
    )
    _, replay_path = _repo_path(root, binding.get("path"), "strict_arithmetic_replay.path")
    expected_digest = _require_digest(
        binding.get("sha256"), "strict_arithmetic_replay.sha256"
    )
    _require(replay_path.is_file() and _file_digest(replay_path) == expected_digest,
             "strict arithmetic replay evidence is missing or has the wrong digest")
    replay = _load_json(replay_path, "strict arithmetic replay evidence")
    _require(
        replay.get("schema")
        == "v13-backend-neutral-izh-arithmetic-replay-evidence-manifest-v2"
        and replay.get("outcome") == "DIAGNOSTIC_PASS"
        and replay.get("diagnostic_only") is True
        and replay.get("scientific_verdict") is None,
        "strict arithmetic replay v2 has not earned its diagnostic pass",
    )
    supplied = _require_digest(replay.get("sha256"), "strict replay artifact sha256")
    _require(supplied == _canonical_digest(replay),
             "strict arithmetic replay evidence self-digest is invalid")
    source = replay.get("source")
    replay_files = _historical_replay_source(root, source, replay_revision)
    candidate_sensitive = {
        path: metadata["sha256"]
        for path, metadata in candidate_source["files"].items()
        if path.startswith(REPLAY_SENSITIVE_PREFIX) and path.endswith(".py")
    }
    _require(bool(candidate_sensitive),
             "candidate source closure contains no replay-sensitive simulator code")
    for relative, digest in sorted(candidate_sensitive.items()):
        _require(
            replay_files.get(relative) == digest,
            "candidate replay-sensitive source differs from the strict replay v2 "
            f"revision: {relative}",
        )
    comparison = replay.get("comparison")
    _require(isinstance(comparison, dict),
             "strict arithmetic replay evidence lacks its comparison binding")
    comparison_relative, comparison_path = _repo_path(
        root, comparison.get("path"), "strict arithmetic replay comparison path"
    )
    del comparison_relative
    comparison_file_digest = _require_digest(
        comparison.get("sha256"), "strict arithmetic replay comparison file sha256"
    )
    _require(
        comparison_path.is_file()
        and _file_digest(comparison_path) == comparison_file_digest,
        "strict arithmetic replay comparison is missing or has the wrong digest",
    )
    comparison_artifact = _load_json(comparison_path, "strict arithmetic replay comparison")
    _require(
        comparison_artifact.get("outcome") == "DIAGNOSTIC_PASS"
        and comparison_artifact.get("all_required_trajectories_exact") is True
        and all(
            comparison_artifact.get("trajectory_comparisons", {})
            .get(name, {}).get("all_1200_rows_exact") is True
            for name in ("v", "u", "spikes")
        ),
        "strict arithmetic replay comparison does not prove exact required trajectories",
    )
    _require(
        comparison_artifact.get("sha256") == comparison.get("artifact_sha256")
        and comparison_artifact.get("source") == source,
        "strict arithmetic replay comparison differs from its evidence manifest",
    )
    cells = replay.get("cells")
    _require(
        isinstance(cells, dict) and set(cells) == {"numpy", "cupy"}
        and comparison_artifact.get("cell_artifacts") == cells,
        "strict arithmetic replay cell chain is incomplete",
    )
    for backend, record in cells.items():
        _require(
            isinstance(record, dict)
            and set(record) == {
                "artifact_sha256", "file_sha256", "path", "receipt_path",
            },
            f"strict replay {backend} cell binding is invalid",
        )
        _, cell_path = _repo_path(root, record.get("path"), f"strict replay {backend} cell")
        file_sha = _require_digest(record.get("file_sha256"), f"strict replay {backend} file")
        _require(cell_path.is_file() and _file_digest(cell_path) == file_sha,
                 f"strict replay {backend} cell file has the wrong digest")
        cell = _load_json(cell_path, f"strict replay {backend} cell")
        _require(
            cell.get("schema") == "v13-backend-neutral-izh-arithmetic-replay-cell-v2"
            and cell.get("backend") == backend
            and cell.get("source") == source
            and cell.get("sha256") == record.get("artifact_sha256")
            and cell.get("sha256") == _canonical_digest(cell),
            f"strict replay {backend} cell artifact binding is invalid",
        )
        _validate_replay_receipt(
            root, receipt_path=record.get("receipt_path"), source=source,
            artifact_path=record["path"], artifact_sha256=file_sha,
            mode="--run", backend=backend,
        )
    _validate_replay_receipt(
        root, receipt_path=comparison.get("receipt_path"), source=source,
        artifact_path=comparison["path"], artifact_sha256=comparison_file_digest,
        mode="--compare", receipt_sha256=comparison.get("receipt_sha256"),
    )


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


def _validate_process_correction_spec(
    spec: dict[str, Any], *, config: dict[str, Any],
) -> None:
    _require(spec.get("schema") == PROCESS_CORRECTION_SCHEMA,
             "locked process-correction schema is invalid")
    _require(spec.get("status") == PROCESS_CORRECTION_STATUS,
             "locked process-correction status is invalid")
    _require(
        spec.get("authority")
        == "research/findings/2026-08-04-neural-vocal-credit-gateB-v13-stage0-"
           "process-correction-v2-PREREGISTRATION.md",
        "locked process-correction authority is invalid",
    )
    base = spec.get("base_spec")
    _require(
        base == {
            "path": BASE_SPEC_PATH,
            "sha256": config["candidate_source_identity"][BASE_SPEC_PATH],
        },
        "locked process-correction base specification is invalid",
    )
    replay = spec.get("strict_arithmetic_replay")
    _require(
        replay == {
            "path": config["strict_arithmetic_replay"]["path"],
            "sha256": config["strict_arithmetic_replay"]["sha256"],
            "outcome": "DIAGNOSTIC_PASS",
        },
        "locked process-correction replay prerequisite is invalid",
    )
    _require(spec.get("seed_derivation") == _expected_seed_derivation(),
             "locked process-correction seed derivation is invalid")
    _require(
        spec.get("partitions")
        == {name: [seed] for name, seed in config["seeds"].items()},
        "locked process-correction partitions differ from the frozen config",
    )
    _require(
        spec.get("forbidden_consumed_seeds") == sorted(FORBIDDEN_CONSUMED_SEEDS),
        "locked process-correction consumed-seed list is invalid",
    )
    _require(
        spec.get("retired_unexecuted_seeds") == sorted(RETIRED_UNEXECUTED_SEEDS),
        "locked process-correction retired-seed list is invalid",
    )
    _require(
        spec.get("sealed_future_seeds")
        == {"held_out": LOCKED_HELD_OUT_SEED, "stage_1": 1031},
        "locked process-correction future seed seals are invalid",
    )
    _require(
        spec.get("calibration") == {
            "ladder_pA": list(CALIBRATION_LADDER_PA),
            "fresh_brain_per_point": True,
            "selection": "lowest_common_passing_point_after_both_backends_sealed",
            "forbid_v1_current_preference": True,
        },
        "locked process-correction calibration contract is invalid",
    )
    compatibility = spec.get("compatibility")
    _require(isinstance(compatibility, dict),
             "locked process-correction compatibility contract is invalid")
    _require(
        {field: compatibility.get(field) for field in config["compatibility"]}
        == config["compatibility"],
        "locked process-correction compatibility digests differ from the config",
    )
    _require(
        compatibility.get("verification") == {
            "same_opened_byte_buffer": True,
            "regular_file_only": True,
            "reject_symlink": True,
            "verify_bytes_before_and_after_parse": True,
            "require_both_named_digests_in_scientific_artifacts": True,
        },
        "locked process-correction compatibility verification contract is invalid",
    )
    manifest = spec.get("artifact_manifest")
    _require(isinstance(manifest, dict),
             "locked process-correction manifest contract is invalid")
    _require(manifest.get("schema") == MANIFEST_SCHEMA,
             "locked process-correction manifest schema is invalid")
    _require(manifest.get("create_only") is True,
             "locked process-correction manifests must be create-only")
    _require(manifest.get("manifest_v1_accepted_for_promotion") is False,
             "locked process-correction cannot promote manifest v1")
    _require(
        manifest.get("required_sealed_entries") == [
            "scientific_artifact", "provenance_sidecar", "command_envelope",
            "execution_receipt", "controller_config", "process_correction_spec",
            "candidate_source_manifest", "source_revision",
            "compatibility_file_sha256", "compatibility_canonical_json_sha256",
            "compatibility_canonicalization",
        ],
        "locked process-correction manifest seals are invalid",
    )
    _require(
        manifest.get("required_cross_checks") == [
            "artifact path and sha256 agree across sidecar receipt and manifest",
            "run identity arguments backend source and destination agree",
            "all sealed inputs are regular files and unchanged during validation",
            "provenance sidecar is sealed in the same validation step as the artifact",
        ],
        "locked process-correction manifest cross-checks are invalid",
    )
    _require(
        spec.get("execution_order") == [
            "calibration_numpy", "seal_calibration_numpy_manifest_v2",
            "calibration_cupy_after_numpy_manifest_v2_sealed",
            "seal_calibration_cupy_manifest_v2",
            "merge_calibration_lowest_common_pass",
            "replication_numpy_and_cupy_after_selection",
            "held_out_cupy_after_both_replication_go",
            "held_out_numpy_after_cupy_sealed",
        ],
        "locked process-correction execution order is invalid",
    )
    _require(
        spec.get("stop_rules") == [
            "v1 evidence cannot unlock any v2 stage",
            "any forbidden or retired seed blocks command emission",
            "any digest-domain substitution is UNDEFINED",
            "a missing or changed provenance sidecar is UNDEFINED",
            "any source receipt order ladder or rerun violation is UNDEFINED",
        ],
        "locked process-correction stop rules are invalid",
    )


def _verify_source_binding(
    config: dict[str, Any], root: Path, source: dict[str, Any],
) -> None:
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
    _validate_process_correction_spec(spec, config=config)
    expected = config["seeds"]
    partitions = spec.get("partitions")
    _require(isinstance(partitions, dict), "locked seed specification has no partitions object")
    for name in ("calibration", "replication", "held_out"):
        _require(partitions.get(name) == [expected[name]],
                 f"locked seed specification does not bind replacement {name} seed")
    _require(
        spec.get("seed_derivation") == config["seed_derivation"],
        "locked process-correction specification does not bind seed derivation metadata",
    )
    _require(
        source["files"].get(SEED_SPEC_PATH, {}).get("sha256") == expected_digest,
        "seed binding digest differs from the frozen candidate source manifest",
    )


def load_config(path: Path, *, root: Path = ROOT, verify_source: bool = True) -> dict[str, Any]:
    config_evidence = _load_json_evidence(path, "correction config")
    config = FrozenConfig(
        config_evidence.value,
        path=path.resolve(strict=True),
        file_sha256=config_evidence.file_sha256,
    )
    _require(set(config) == CONFIG_FIELDS,
             "correction config has missing or unknown top-level fields")
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
    _require(
        not ({seeds["calibration"], seeds["replication"]}
             & (FORBIDDEN_CONSUMED_SEEDS | RETIRED_UNEXECUTED_SEEDS)),
        "calibration and replication seeds must exclude consumed and retired partitions",
    )
    _require(seeds["held_out"] == LOCKED_HELD_OUT_SEED,
             "held-out seed must remain the original sealed Stage-0 seed")
    _validate_seed_derivation(config)

    source_manifest = config.get("candidate_source_manifest")
    _require(isinstance(source_manifest, dict),
             "candidate_source_manifest must be an object")
    _require_digest(source_manifest.get("sha256"), "candidate_source_manifest.sha256")
    _require_digest(
        source_manifest.get("tree_sha256"), "candidate_source_manifest.tree_sha256"
    )

    candidate_source = _validate_source_manifest_binding(
        config, root, config["candidate_source_revision"]
    )
    _validate_strict_replay_binding(config, root, candidate_source)

    compatibility = config.get("compatibility")
    _require(
        isinstance(compatibility, dict)
        and set(compatibility) == {
            "path", "file_sha256", "canonical_json_sha256", "canonicalization",
        },
        "compatibility must contain exactly the named byte and canonical digest fields",
    )
    _require(compatibility.get("path") == COMPATIBILITY_PATH,
             f"compatibility.path must be canonical: {COMPATIBILITY_PATH}")
    compatibility_digest = _require_digest(
        compatibility.get("file_sha256"), "compatibility.file_sha256"
    )
    canonical_digest = _require_digest(
        compatibility.get("canonical_json_sha256"),
        "compatibility.canonical_json_sha256",
    )
    _require(
        compatibility.get("canonicalization") == COMPATIBILITY_CANONICALIZATION,
        "compatibility canonicalization algorithm is not locked",
    )
    _, compatibility_path = _repo_path(root, compatibility.get("path"), "compatibility.path")
    compatibility_evidence = _load_json_evidence(
        compatibility_path, "compatibility artifact"
    )
    compatibility_artifact = compatibility_evidence.value
    _require(
        compatibility_evidence.file_sha256 == compatibility_digest,
        "canonical compatibility artifact byte digest is wrong",
    )
    _require(
        compatibility_evidence.canonical_json_sha256 == canonical_digest,
        "canonical compatibility JSON digest is wrong",
    )
    _require(
        compatibility_evidence.canonicalization
        == compatibility["canonicalization"],
        "canonical compatibility algorithm differs from the stable reader",
    )
    _require(compatibility_artifact.get("outcome") == "DETERMINISTIC_COMPATIBILITY_GO"
             and compatibility_artifact.get("go") is True,
             "canonical compatibility artifact has not earned GO")

    legacy = config.get("legacy_performance")
    _require(isinstance(legacy, dict), "legacy_performance must be an object")
    _require_revision(legacy.get("source_revision"), "legacy_performance.source_revision")
    _require(legacy.get("runner_path") == "research/runners/_vocal_action_credit_gate_v13_tonic_output.py",
             "legacy_performance.runner_path is not canonical")
    _require_digest(legacy.get("runner_sha256"), "legacy_performance.runner_sha256")
    _artifact_paths(config, root)
    if verify_source:
        _verify_source_binding(config, root, candidate_source)
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
        return {"SIM_BACKEND": "numpy"}
    return {"SIM_BACKEND": "cupy"}


def _artifact_backend(artifact: dict[str, Any]) -> str | None:
    backend = artifact.get("backend")
    if isinstance(backend, str) and backend:
        return backend
    backend_info = artifact.get("backend_info")
    if isinstance(backend_info, dict):
        backend = backend_info.get("backend")
        if isinstance(backend, str) and backend:
            return backend
    return None


def _artifact_source_revisions(artifact: dict[str, Any]) -> set[str]:
    revisions: set[str] = set()
    source = artifact.get("source_sha")
    if isinstance(source, str) and source:
        revisions.add(source.lower())
    sources = artifact.get("source_shas")
    if isinstance(sources, dict):
        revisions.update(
            value.lower() for value in sources.values()
            if isinstance(value, str) and value
        )
    return revisions


def _validate_v13_provenance_sidecar(
    *, root: Path, cwd: Path, artifact_path: Path, artifact: dict[str, Any],
    envelope: dict[str, Any], receipt: dict[str, Any], kind: str,
) -> tuple[str, str]:
    _require(receipt.get("schema") == execution_receipt.SCHEMA_V2,
             f"{kind} requires a provenance-binding execution receipt v2")
    try:
        sidecar_relative = Path(f"{artifact_path}.prov.json").relative_to(root).as_posix()
    except ValueError as exc:
        raise ControllerError(f"{kind} provenance sidecar is outside the source root") from exc
    _, sidecar_path = _repo_path(
        root, sidecar_relative, f"{kind} provenance sidecar"
    )
    sidecar_evidence = _load_json_evidence(
        sidecar_path, f"{kind} provenance sidecar"
    )
    sidecar = sidecar_evidence.value
    provenance = receipt.get("provenance")
    _require(
        isinstance(provenance, dict)
        and provenance.get("path") == sidecar_relative
        and provenance.get("sha256") == sidecar_evidence.file_sha256,
        f"{kind} receipt does not seal the canonical provenance sidecar",
    )
    _require(sidecar.get("schema") == execution_receipt.PROVENANCE_SCHEMA_V2,
             f"{kind} provenance sidecar schema is invalid")
    _require(sidecar.get("run_id") == provenance.get("run_id"),
             f"{kind} provenance sidecar run ID differs from receipt")
    _require(
        sidecar.get("started_utc_ns") == provenance.get("started_utc_ns")
        and sidecar.get("ended_utc_ns") == provenance.get("ended_utc_ns"),
        f"{kind} provenance timing differs from receipt",
    )
    artifact_relative, sidecar_artifact = _repo_path(
        root, sidecar.get("artifact"), f"{kind} provenance artifact"
    )
    _require(sidecar_artifact == artifact_path,
             f"{kind} provenance artifact differs from canonical destination")
    del artifact_relative

    runner_relative = RUNNER_MODULE.replace(".", "/") + ".py"
    _require(sidecar.get("runner") == runner_relative,
             f"{kind} provenance runner differs from frozen runner")
    sidecar_argv = sidecar.get("argv")
    _require(
        isinstance(sidecar_argv, list)
        and sidecar_argv
        and all(isinstance(item, str) and item for item in sidecar_argv),
        f"{kind} provenance argv is invalid",
    )
    runner_input = Path(sidecar_argv[0])
    runner_path = runner_input if runner_input.is_absolute() else cwd / runner_input
    try:
        runner_path = runner_path.resolve(strict=True)
        expected_runner = (cwd / runner_relative).resolve(strict=True)
    except OSError as exc:
        raise ControllerError(f"{kind} provenance runner path is invalid") from exc
    _require(runner_path == expected_runner,
             f"{kind} provenance argv names a different runner")
    _require(sidecar_argv[1:] == envelope["argv"][3:],
             f"{kind} provenance argv differs from command envelope")

    expected_source = envelope["source_revision"]
    _require(sidecar.get("git_sha") == expected_source,
             f"{kind} provenance source revision differs from command envelope")
    receipt_source = receipt.get("source")
    _require(isinstance(receipt_source, dict), f"{kind} receipt source is invalid")
    _require(
        sidecar.get("source_kind") == receipt_source.get("kind")
        and sidecar.get("source_manifest_sha256")
        == receipt_source.get("manifest_sha256"),
        f"{kind} provenance source identity differs from receipt",
    )
    artifact_sources = _artifact_source_revisions(artifact)
    _require(artifact_sources or kind == "final_stage0",
             f"{kind} artifact lacks a source revision binding")
    _require(all(source == expected_source for source in artifact_sources),
             f"{kind} artifact source differs from provenance")

    execution_backend = envelope["env"].get("SIM_BACKEND")
    _require(execution_backend in {"numpy", "cupy"},
             f"{kind} command lacks an explicit execution backend")
    _require(
        receipt.get("env_allowlist", {}).get("SIM_BACKEND") == execution_backend
        and sidecar.get("env", {}).get("SIM_BACKEND") == execution_backend
        and sidecar.get("sim_backend_requested") == execution_backend
        and sidecar.get("sim_backend") == execution_backend,
        f"{kind} provenance backend differs from command or receipt",
    )
    if execution_backend == "cupy":
        _require(sidecar.get("sim_backend_cupy_importable") is True,
                 f"{kind} provenance does not confirm CuPy availability")
    artifact_backend = _artifact_backend(artifact)
    expected_artifact_backend = (
        "cross_backend"
        if kind in {"calibration_selection", "final_stage0"}
        else execution_backend
    )
    _require(artifact_backend == expected_artifact_backend,
             f"{kind} artifact backend differs from the sealed execution")
    return sidecar_relative, sidecar_evidence.file_sha256


def _require_candidate_receipt_source(
    source: Any, *, config: dict[str, Any], label: str,
) -> None:
    binding = config["candidate_source_manifest"]
    _require(isinstance(source, dict), f"{label} source binding is invalid")
    _require(
        source.get("git_sha") == config["candidate_source_revision"]
        and source.get("kind") in {"git", "git_archive"}
        and source.get("manifest") == binding["path"]
        and source.get("manifest_sha256") == binding["sha256"]
        and source.get("tree_sha256") == binding["tree_sha256"]
        and source.get("file_count") == binding["file_count"],
        f"{label} source manifest differs from the frozen candidate source",
    )


def _expected_manifest_argv(
    *, config: dict[str, Any], kind: str, root: Path, output: Path,
) -> list[str]:
    paths = _artifact_paths(config, root)
    prefix = [config["python"], "-m", config["runner_module"]]
    corrected_prefix = [
        *prefix, "--process-correction-spec", str((root / SEED_SPEC_PATH).resolve()),
    ]
    if kind.startswith("calibration_") and kind != "calibration_selection":
        return [
            *corrected_prefix, "--calibration", "--compatibility-correction",
            str((root / COMPATIBILITY_PATH).resolve()), "--out", str(output),
        ]
    if kind == "calibration_selection":
        return [
            *corrected_prefix, "--merge-calibration", str(paths["calibration_numpy"]),
            str(paths["calibration_cupy"]), "--out", str(output),
        ]
    if kind.startswith("replication_"):
        return [
            *corrected_prefix, "--replication", str(paths["calibration_selection"]),
            "--out", str(output),
        ]
    if kind.startswith("held_out_"):
        return [
            *corrected_prefix, "--held-out", str(paths["calibration_selection"]),
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
        *corrected_prefix, "--merge-final", str(compatibility),
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
    manifest = _load_json_evidence(manifest_path, f"{kind} manifest").value
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

    config_ref = manifest.get("controller_config")
    _require(
        isinstance(config_ref, dict)
        and set(config_ref) == MANIFEST_CONFIG_REFERENCE_FIELDS,
        f"{kind} controller config reference is invalid",
    )
    _require(isinstance(config, FrozenConfig),
             f"{kind} config lacks exact-file validation metadata")
    _, referenced_config = _repo_path(
        root, config_ref.get("path"), f"{kind} controller config path"
    )
    _require(referenced_config == config.path,
             f"{kind} manifest names a different controller config")
    _require(config_ref.get("file_sha256") == config.file_sha256,
             f"{kind} controller config exact-byte digest differs")
    _require(config_ref.get("canonical_sha256") == config["sha256"],
             f"{kind} controller config canonical digest differs")
    _require(
        _load_json_evidence(referenced_config, f"{kind} controller config").file_sha256
        == config.file_sha256,
        f"{kind} controller config changed after validation",
    )

    process_ref = manifest.get("process_correction_spec")
    _require(
        isinstance(process_ref, dict)
        and set(process_ref) == MANIFEST_ARTIFACT_FIELDS
        and process_ref == config["seed_binding"],
        f"{kind} process-correction specification reference is invalid",
    )
    _, process_path = _repo_path(
        root, process_ref.get("path"), f"{kind} process-correction specification"
    )
    _require(
        _load_json_evidence(
            process_path, f"{kind} process-correction specification"
        ).file_sha256 == process_ref["sha256"],
        f"{kind} process-correction specification changed",
    )

    source_ref = manifest.get("candidate_source_manifest")
    _require(
        isinstance(source_ref, dict)
        and set(source_ref) == MANIFEST_SOURCE_REFERENCE_FIELDS
        and source_ref == config["candidate_source_manifest"],
        f"{kind} candidate source manifest reference is invalid",
    )
    source = execution_receipt.verify_source_manifest(root, source_ref["path"])
    _require(
        source["manifest_sha256"] == source_ref["sha256"]
        and source["tree_sha256"] == source_ref["tree_sha256"]
        and source["file_count"] == source_ref["file_count"],
        f"{kind} candidate source manifest changed",
    )

    compatibility_ref = manifest.get("compatibility")
    _require(
        isinstance(compatibility_ref, dict)
        and set(compatibility_ref) == MANIFEST_COMPATIBILITY_FIELDS
        and compatibility_ref == config["compatibility"],
        f"{kind} compatibility reference is invalid",
    )
    _, compatibility_path = _repo_path(
        root, compatibility_ref["path"], f"{kind} compatibility evidence"
    )
    compatibility_evidence = _load_json_evidence(
        compatibility_path, f"{kind} compatibility evidence"
    )
    _require(
        compatibility_evidence.file_sha256 == compatibility_ref["file_sha256"]
        and compatibility_evidence.canonical_json_sha256
        == compatibility_ref["canonical_json_sha256"]
        and compatibility_evidence.canonicalization
        == compatibility_ref["canonicalization"],
        f"{kind} compatibility evidence changed or swapped digest domains",
    )

    artifact_ref = manifest.get("artifact")
    _require(isinstance(artifact_ref, dict) and set(artifact_ref) == MANIFEST_ARTIFACT_FIELDS,
             f"{kind} manifest artifact reference is invalid")
    expected_path = _artifact_paths(config, root)[kind]
    _, artifact_path = _repo_path(root, artifact_ref.get("path"), f"{kind} artifact path")
    _require(artifact_path == expected_path, f"{kind} manifest names a non-canonical artifact path")
    artifact_digest = _require_digest(artifact_ref.get("sha256"), f"{kind} artifact sha256")
    artifact_evidence = _load_json_evidence(artifact_path, f"{kind} artifact")
    _require(artifact_evidence.file_sha256 == artifact_digest,
             f"{kind} artifact is missing or its digest changed")
    artifact = artifact_evidence.value

    sidecar_ref = manifest.get("provenance_sidecar")
    _require(
        isinstance(sidecar_ref, dict)
        and set(sidecar_ref) == MANIFEST_ARTIFACT_FIELDS,
        f"{kind} provenance sidecar reference is invalid",
    )
    sidecar_relative, sidecar_path = _repo_path(
        root, sidecar_ref.get("path"), f"{kind} provenance sidecar path"
    )
    expected_sidecar = Path(f"{artifact_path}.prov.json")
    _require(sidecar_path == expected_sidecar,
             f"{kind} manifest names a non-canonical provenance sidecar")
    sidecar_digest = _require_digest(
        sidecar_ref.get("sha256"), f"{kind} provenance sidecar sha256"
    )
    sidecar_evidence = _load_json_evidence(sidecar_path, f"{kind} provenance sidecar")
    _require(sidecar_evidence.file_sha256 == sidecar_digest,
             f"{kind} provenance sidecar is missing or its digest changed")

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
    envelope_evidence = _load_json_evidence(
        envelope_path, f"{kind} command envelope"
    )
    _require(envelope_evidence.file_sha256 == envelope_digest,
             f"{kind} command envelope is missing or its digest changed")
    envelope = envelope_evidence.value
    cwd = _validate_manifest_envelope(
        envelope, config=config, kind=kind, root=root, artifact_path=artifact_path
    )
    _require(
        _load_json_evidence(
            envelope_path, f"{kind} command envelope"
        ).file_sha256 == envelope_digest,
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
    receipt_evidence = _load_json_evidence(
        receipt_path, f"{kind} execution receipt"
    )
    _require(receipt_evidence.file_sha256 == receipt_digest,
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
    _require(
        _load_json_evidence(
            receipt_path, f"{kind} execution receipt"
        ).file_sha256 == receipt_digest,
             f"{kind} execution receipt changed while being validated")
    _require(receipt.get("argv") == envelope["argv"],
             f"{kind} receipt argv differs from command envelope")
    _require(receipt.get("env_allowlist") == envelope["env"],
             f"{kind} receipt environment differs from command envelope")
    _require(receipt.get("source", {}).get("git_sha") == expected_source,
             f"{kind} receipt source revision is invalid")
    if kind != "performance_baseline":
        _require_candidate_receipt_source(
            receipt.get("source"), config=config, label=f"{kind} receipt"
        )
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
    _require(
        _load_json_evidence(artifact_path, f"{kind} artifact").file_sha256
        == artifact_digest,
             f"{kind} artifact changed while evidence was being validated")
    validated_sidecar_relative, validated_sidecar_digest = (
        _validate_v13_provenance_sidecar(
            root=root,
            cwd=cwd,
            artifact_path=artifact_path,
            artifact=artifact,
            envelope=envelope,
            receipt=receipt,
            kind=kind,
        )
    )
    _require(
        (validated_sidecar_relative, validated_sidecar_digest)
        == (sidecar_relative, sidecar_digest),
        f"{kind} manifest sidecar seal differs from validated receipt provenance",
    )
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
        "provenance_sidecar_path": sidecar_relative,
        "provenance_sidecar_sha256": sidecar_digest,
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
    _require(all(binding.get(field) == expected[field] for field in (
        "path", "file_sha256", "canonical_json_sha256", "canonicalization",
    )),
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
    _require(all(binding.get(field) == expected[field] for field in (
        "path", "file_sha256", "canonical_json_sha256", "canonicalization",
    )),
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


def _runner_argv(
    config: dict[str, Any], *arguments: str, root: Path | None = None,
    process_correction: bool = False,
) -> list[str]:
    prefix = [config["python"], "-m", config["runner_module"]]
    if process_correction:
        _require(root is not None, "process-correction commands require a source root")
        prefix.extend([
            "--process-correction-spec", str((root / SEED_SPEC_PATH).resolve()),
        ])
    return [*prefix, *arguments]


def _calibration_inputs(
    *, config_path: Path, backend: str, numpy_manifest: Path | None,
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    config = load_config(config_path, root=root)
    _require(backend in {"numpy", "cupy"}, "calibration backend must be numpy or cupy")
    replay = config["strict_arithmetic_replay"]
    prerequisites: list[dict[str, Any]] = [
        {
            "kind": "strict_arithmetic_replay_v2",
            "artifact_path": str((root / replay["path"]).resolve()),
            "artifact_sha256": replay["sha256"],
            "source_revision": replay["source_revision"],
        },
        {
            "kind": "compatibility",
            "artifact_path": str((root / COMPATIBILITY_PATH).resolve()),
            "file_sha256": config["compatibility"]["file_sha256"],
            "canonical_json_sha256": config["compatibility"]["canonical_json_sha256"],
            "canonicalization": config["compatibility"]["canonicalization"],
        },
    ]
    if backend == "cupy":
        _require(numpy_manifest is not None,
                 "CuPy calibration requires a digested NumPy artifact")
        artifact, _, reference = load_manifest(
            numpy_manifest, config=config, kind="calibration_numpy", root=root
        )
        _validate_calibration_backend(artifact, config=config, backend="numpy")
        prerequisites.append(reference)
    else:
        _require(numpy_manifest is None,
                 "NumPy calibration cannot consume a prior NumPy manifest")
    output = _artifact_paths(config, root)[f"calibration_{backend}"]
    _ensure_new_artifact(output)
    return config, prerequisites, output


def check_calibration_readiness(
    *, config_path: Path, backend: str,
    numpy_manifest: Path | None = None, root: Path = ROOT,
) -> dict[str, Any]:
    """Validate calibration prerequisites without writing or exposing seed values."""
    config, prerequisites, output = _calibration_inputs(
        config_path=config_path, backend=backend,
        numpy_manifest=numpy_manifest, root=root,
    )
    source = config["candidate_source_manifest"]
    return {
        "schema": READINESS_SCHEMA,
        "ready": True,
        "action": f"calibration_{backend}",
        "backend": backend,
        "source_revision": config["candidate_source_revision"],
        "source_manifest": {
            "path": source["path"],
            "sha256": source["sha256"],
            "tree_sha256": source["tree_sha256"],
            "file_count": source["file_count"],
        },
        "prerequisite_kinds": [item["kind"] for item in prerequisites],
        "output_available": not output.exists(),
        "command_emitted": False,
        "execution": "not_executed",
    }


def emit_calibration(
    *, config_path: Path, backend: str, emit: Path,
    numpy_manifest: Path | None = None, root: Path = ROOT,
) -> dict[str, Any]:
    config, prerequisites, output = _calibration_inputs(
        config_path=config_path, backend=backend,
        numpy_manifest=numpy_manifest, root=root,
    )
    argv = _runner_argv(
        config, "--calibration", "--compatibility-correction",
        str((root / COMPATIBILITY_PATH).resolve()), "--out", str(output),
        root=root, process_correction=True,
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
        root=root, process_correction=True,
    )
    envelope = _envelope(
        action="merge_calibration", config_path=config_path, config=config,
        root=root, cwd=root, argv=argv,
        env=_expected_manifest_env("calibration_selection"), output=output,
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
        "--out", str(output), root=root, process_correction=True,
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
        "--out", str(output), root=root, process_correction=True,
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
        "file_sha256": config["compatibility"]["file_sha256"],
        "canonical_json_sha256": config["compatibility"]["canonical_json_sha256"],
        "canonicalization": config["compatibility"]["canonicalization"],
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
        root=root, process_correction=True,
    )
    envelope = _envelope(
        action="final_stage0_merge", config_path=config_path, config=config,
        root=root, cwd=root, argv=argv,
        env=_expected_manifest_env("final_stage0"), output=output,
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
    parser.add_argument("--emit", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)

    calibration = commands.add_parser("calibration")
    calibration.add_argument("--backend", choices=("numpy", "cupy"), required=True)
    calibration.add_argument("--numpy-manifest", type=Path)

    readiness = commands.add_parser("calibration-readiness")
    readiness.add_argument("--backend", choices=("numpy", "cupy"), required=True)
    readiness.add_argument("--numpy-manifest", type=Path)

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
        if args.command == "calibration-readiness":
            readiness = check_calibration_readiness(
                config_path=args.config, backend=args.backend,
                numpy_manifest=args.numpy_manifest, root=root,
            )
            print(json.dumps(readiness, sort_keys=True))
            return 0
        _require(args.emit is not None, "--emit is required for command-envelope creation")
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
