#!/usr/bin/env python3
"""Create the frozen V13 Stage-0 source manifest and controller config.

This helper never emits or executes a scientific command.  It only converts
already committed repository state into create-only control-plane artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

try:
    from tools import execution_receipt
    from tools import stable_json_evidence
    from tools import v13_stage0_controller as controller
except ModuleNotFoundError:  # Direct ``python tools/...`` invocation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools import execution_receipt
    from tools import stable_json_evidence
    from tools import v13_stage0_controller as controller


ROOT = Path(__file__).resolve().parents[1]
LEGACY_BASE_REVISION = "8994b5102"
CORRECTION_ID = "v13-stage0-process-correction-v3"
ARTIFACT_ROOT = "research/findings/raw/v13_tonic_output_stage0_process_correction_v3"


class FreezeError(RuntimeError):
    """Raised when committed inputs cannot produce a frozen artifact."""


def _run_git(root: Path, *args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=root, capture_output=True,
        text=not binary, check=False,
    )
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", "replace") if binary else result.stderr
        raise FreezeError(f"git {' '.join(args)} failed: {error.strip()}")
    return result.stdout


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _compatibility_binding(path: Path) -> dict[str, str]:
    try:
        evidence = stable_json_evidence.read_stable_json_evidence(
            path, require_object=True
        )
    except stable_json_evidence.StableJsonEvidenceError as exc:
        raise FreezeError(str(exc)) from exc
    return {
        "path": controller.COMPATIBILITY_PATH,
        "file_sha256": evidence.file_sha256,
        "canonical_json_sha256": evidence.canonical_json_sha256,
        "canonicalization": evidence.canonicalization,
    }


def _create_only(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise FreezeError(f"refusing to overwrite frozen artifact: {path}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


def _relative(root: Path, path: Path, label: str) -> str:
    root = root.resolve(strict=True)
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise FreezeError(f"{label} must be inside the repository") from exc


def _head(root: Path) -> str:
    return str(_run_git(root, "rev-parse", "HEAD")).strip()


def freeze_source_manifest(*, root: Path, emit: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    relative_emit = _relative(root, emit, "source manifest")
    paths = tuple(sorted(controller._required_candidate_source_paths(root)))
    status = str(_run_git(root, "status", "--porcelain", "--", *paths)).strip()
    if status:
        raise FreezeError("candidate source closure is dirty or untracked")
    revision = _head(root)
    lines = []
    for relative in paths:
        working = root / relative
        if not working.is_file():
            raise FreezeError(f"candidate source is missing: {relative}")
        committed = controller._revision_file_digest(root, revision, relative)
        if committed != _digest(working):
            raise FreezeError(f"working bytes differ from committed source: {relative}")
        lines.append(f"{committed}  {relative}\n")
    payload = "".join(lines).encode("ascii")
    _create_only(root / relative_emit, payload)
    snapshot = execution_receipt.verify_source_manifest(root, relative_emit)
    return {
        "kind": "source_manifest",
        "path": relative_emit,
        "revision": revision,
        "manifest_sha256": snapshot["manifest_sha256"],
        "tree_sha256": snapshot["tree_sha256"],
        "file_count": snapshot["file_count"],
    }


def _require_committed_manifest(
    *, root: Path, relative: str, revision: str,
) -> dict[str, Any]:
    path = root / relative
    snapshot = execution_receipt.verify_source_manifest(root, relative)
    committed = controller._revision_file_digest(root, revision, relative)
    if committed != snapshot["manifest_sha256"] or committed != _digest(path):
        raise FreezeError("source manifest is not committed at the candidate revision")
    required = set(controller._required_candidate_source_paths(root))
    if set(snapshot["files"]) != required:
        raise FreezeError("source manifest differs from the deterministic source closure")
    for source_path, metadata in snapshot["files"].items():
        if controller._revision_file_digest(root, revision, source_path) != metadata["sha256"]:
            raise FreezeError(f"manifest entry differs from candidate revision: {source_path}")
    return snapshot


def _legacy_revision_and_runner(root: Path) -> tuple[str, str]:
    revision = str(_run_git(root, "rev-parse", LEGACY_BASE_REVISION)).strip()
    runner = root / controller.CRITICAL_SOURCE_PATHS[0]
    return revision, _digest(runner)


def _artifact_paths() -> dict[str, str]:
    names = {
        "calibration_numpy": "calibration-numpy.json",
        "calibration_cupy": "calibration-cupy.json",
        "calibration_selection": "calibration-selection.json",
        "replication_numpy": "replication-numpy.json",
        "replication_cupy": "replication-cupy.json",
        "held_out_cupy": "held-out-cupy.json",
        "held_out_numpy": "held-out-numpy.json",
        "performance_baseline": "performance-baseline.json",
        "performance_candidate": "performance-candidate.json",
        "final_stage0": "final-stage0.json",
    }
    return {name: f"{ARTIFACT_ROOT}/{filename}" for name, filename in names.items()}


def build_config(
    *, root: Path, source_manifest: Path, python: Path,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    revision = _head(root)
    relative_manifest = _relative(root, source_manifest, "source manifest")
    snapshot = _require_committed_manifest(
        root=root, relative=relative_manifest, revision=revision,
    )
    python = python.expanduser().absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise FreezeError("configured Python is not executable")

    seed_spec = json.loads((root / controller.SEED_SPEC_PATH).read_text())
    partitions = seed_spec.get("partitions", {})
    seeds = {
        name: partitions[name][0]
        for name in ("calibration", "replication", "held_out")
    }
    derivation = seed_spec.get("seed_derivation")
    replay_path = root / controller.STRICT_REPLAY_PATH
    replay = json.loads(replay_path.read_text())
    compatibility_path = root / controller.COMPATIBILITY_PATH
    legacy_revision, legacy_runner_sha = _legacy_revision_and_runner(root)
    identity = {
        relative: snapshot["files"][relative]["sha256"]
        for relative in controller.CRITICAL_SOURCE_PATHS
    }
    body = {
        "schema": controller.CONFIG_SCHEMA,
        "status": "frozen",
        "correction_id": CORRECTION_ID,
        "candidate_source_revision": revision,
        "candidate_source_identity": identity,
        "candidate_source_manifest": {
            "path": relative_manifest,
            "sha256": snapshot["manifest_sha256"],
            "tree_sha256": snapshot["tree_sha256"],
            "file_count": snapshot["file_count"],
        },
        "python": str(python),
        "runner_module": controller.RUNNER_MODULE,
        "seeds": seeds,
        "seed_derivation": derivation,
        "seed_binding": {
            "path": controller.SEED_SPEC_PATH,
            "sha256": _digest(root / controller.SEED_SPEC_PATH),
        },
        "strict_arithmetic_replay": {
            "path": controller.STRICT_REPLAY_PATH,
            "sha256": _digest(replay_path),
            "source_revision": replay["source"]["git_sha"],
        },
        "compatibility": _compatibility_binding(compatibility_path),
        "legacy_performance": {
            "source_revision": legacy_revision,
            "runner_path": controller.CRITICAL_SOURCE_PATHS[0],
            "runner_sha256": legacy_runner_sha,
        },
        "artifacts": _artifact_paths(),
    }
    config = dict(body)
    config["sha256"] = controller._canonical_digest(config)
    return config


def freeze_config(
    *, root: Path, source_manifest: Path, python: Path, emit: Path,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    relative_emit = _relative(root, emit, "controller config")
    config = build_config(root=root, source_manifest=source_manifest, python=python)
    data = (json.dumps(config, indent=2, sort_keys=True) + "\n").encode("ascii")
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=root, prefix=".v13-stage0-config-", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(data)
    try:
        controller.load_config(temporary, root=root)
    except controller.ControllerError as exc:
        raise FreezeError(f"generated config is invalid: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    _create_only(root / relative_emit, data)
    return {
        "kind": "controller_config",
        "path": relative_emit,
        "revision": config["candidate_source_revision"],
        "config_sha256": config["sha256"],
        "source_file_count": config["candidate_source_manifest"]["file_count"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    source = commands.add_parser("source-manifest")
    source.add_argument("--emit", type=Path, required=True)
    config = commands.add_parser("config")
    config.add_argument("--source-manifest", type=Path, required=True)
    config.add_argument("--python", type=Path, required=True)
    config.add_argument("--emit", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    try:
        if args.command == "source-manifest":
            result = freeze_source_manifest(root=root, emit=args.emit)
        else:
            result = freeze_config(
                root=root, source_manifest=args.source_manifest,
                python=args.python, emit=args.emit,
            )
    except (FreezeError, OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"v13-stage0-freeze: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
