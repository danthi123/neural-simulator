#!/usr/bin/env python3
"""Freeze source and emit create-only commands for the V13 initialization diagnostic."""
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

from research.runners import _v13_backend_neutral_izh_initialization_diagnostic as diagnostic
from tools import execution_receipt


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "v13-backend-neutral-izh-initialization-command-v1"
FINAL_MANIFEST_SCHEMA = "v13-backend-neutral-izh-initialization-evidence-manifest-v1"
ACTIONS = ("capture_numpy", "capture_cupy", "compare")
OUTPUT_DIR = "research/findings/raw/v13_backend_neutral_izh_initialization_diagnostic"
_REVISION = re.compile(r"^[0-9a-f]{40}$")


class EvidenceError(ValueError):
    """Raised when a diagnostic evidence command cannot be frozen."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=False,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise EvidenceError("cannot resolve source revision")
    return result.stdout.strip()


def _git_file(root: Path, revision: str, relative: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{revision}:{relative}"], cwd=root, check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise EvidenceError(f"source path is not committed at {revision}: {relative}")
    return result.stdout


def _tracked_sim_python(root: Path, revision: str) -> set[str]:
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", revision, "--", "sim"],
        cwd=root, check=False, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise EvidenceError("cannot enumerate simulator inputs from the frozen revision")
    return {
        line for line in result.stdout.splitlines()
        if line.startswith("sim/") and line.endswith(".py")
    }


def _safe_relative(root: Path, value: str | Path, label: str) -> tuple[str, Path]:
    path = Path(value)
    if path.is_absolute() or not path.name or ".." in path.parts:
        raise EvidenceError(f"{label} must be a safe repository-relative path")
    resolved_root = root.resolve(strict=True)
    resolved = (resolved_root / path).resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise EvidenceError(f"{label} escapes the repository") from exc
    return path.as_posix(), resolved


def _write_create_only(path: Path, payload: bytes, label: str) -> None:
    if not path.parent.is_dir():
        raise EvidenceError(f"{label} parent directory does not exist")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise EvidenceError(f"refusing to overwrite existing {label}: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def freeze_source_manifest(
    *, root: Path, revision: str, out: str | Path,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if _REVISION.fullmatch(revision) is None:
        raise EvidenceError("revision must be a full lowercase Git SHA")
    if _git_head(root) != revision:
        raise EvidenceError("source checkout is not at the requested revision")
    relative_out, output = _safe_relative(root, out, "source manifest")
    entries: list[str] = []
    paths = set(diagnostic.source_paths(root))
    local_sim = {
        relative for relative in paths
        if relative.startswith("sim/") and relative.endswith(".py")
    }
    tracked_sim = _tracked_sim_python(root, revision)
    if local_sim != tracked_sim:
        raise EvidenceError(
            "working simulator Python set differs from the frozen tracked simulator set"
        )
    for relative in sorted(paths):
        _, path = _safe_relative(root, relative, "source path")
        if not path.is_file():
            raise EvidenceError(f"source path is missing: {relative}")
        working = path.read_bytes()
        committed = _git_file(root, revision, relative)
        if working != committed:
            raise EvidenceError(f"source path differs from committed revision: {relative}")
        entries.append(f"{hashlib.sha256(working).hexdigest()}  {relative}\n")
    payload = "".join(entries).encode("ascii")
    _write_create_only(output, payload, "source manifest")
    snapshot = execution_receipt.verify_source_manifest(root, relative_out)
    return {
        "path": relative_out,
        "sha256": snapshot["manifest_sha256"],
        "tree_sha256": snapshot["tree_sha256"],
        "file_count": snapshot["file_count"],
        "revision": revision,
    }


def _paths(root: Path) -> dict[str, str]:
    base = OUTPUT_DIR
    return {
        "source_manifest": f"{base}/source.sha256",
        "numpy_artifact": f"{base}/cell-numpy.json",
        "numpy_receipt": f"{base}/cell-numpy.receipt.json",
        "cupy_artifact": f"{base}/cell-cupy.json",
        "cupy_receipt": f"{base}/cell-cupy.receipt.json",
        "comparison_artifact": f"{base}/comparison.json",
        "comparison_receipt": f"{base}/comparison.receipt.json",
        "evidence_manifest": f"{base}/evidence-manifest.json",
    }


def _inner_command(
    *, root: Path, action: str, revision: str, paths: dict[str, str], python: str,
) -> list[str]:
    module = "research.runners._v13_backend_neutral_izh_initialization_diagnostic"
    common = [
        python, "-m", module,
        "--spec", str((root / diagnostic.SPEC_PATH.relative_to(diagnostic.ROOT)).resolve()),
        "--spec-sha256", diagnostic.SPEC_SHA256,
    ]
    if action in {"capture_numpy", "capture_cupy"}:
        backend = action.removeprefix("capture_")
        return [
            *common, "--capture", "--backend", backend,
            "--source-manifest", str((root / paths["source_manifest"]).resolve()),
            "--source-revision", revision,
            "--out", str((root / paths[f"{backend}_artifact"]).resolve()),
        ]
    return [
        *common, "--compare",
        "--numpy-artifact", str((root / paths["numpy_artifact"]).resolve()),
        "--numpy-receipt", str((root / paths["numpy_receipt"]).resolve()),
        "--cupy-artifact", str((root / paths["cupy_artifact"]).resolve()),
        "--cupy-receipt", str((root / paths["cupy_receipt"]).resolve()),
        "--out", str((root / paths["comparison_artifact"]).resolve()),
    ]


def emit_command(
    *, root: Path, action: str, revision: str, host: str, device: str,
    out: str | Path, python: str = sys.executable,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if action not in ACTIONS:
        raise EvidenceError(f"unsupported action: {action}")
    if _REVISION.fullmatch(revision) is None or _git_head(root) != revision:
        raise EvidenceError("command source revision is not the current full Git SHA")
    if not host.strip() or not device.strip():
        raise EvidenceError("host and device must be explicit")
    if not Path(python).is_absolute() or not Path(python).is_file() or not os.access(python, os.X_OK):
        raise EvidenceError("python must be an absolute executable path")
    relative_out, output = _safe_relative(root, out, "command envelope")
    paths = _paths(root)
    manifest = root / paths["source_manifest"]
    if not manifest.is_file():
        raise EvidenceError("frozen source manifest is missing")
    try:
        snapshot = execution_receipt.verify_source_manifest(root, paths["source_manifest"])
        execution_receipt._source_revision(root, revision, snapshot["manifest_sha256"])
    except execution_receipt.ReceiptError as exc:
        raise EvidenceError(f"source manifest verification failed: {exc}") from exc
    if set(snapshot["files"]) != set(diagnostic.source_paths(root)):
        raise EvidenceError("source manifest contains the wrong source set")

    inner = _inner_command(
        root=root, action=action, revision=revision, paths=paths, python=python,
    )
    backend = action.removeprefix("capture_") if action.startswith("capture_") else "numpy"
    artifact_key = (
        f"{backend}_artifact" if action.startswith("capture_") else "comparison_artifact"
    )
    receipt_key = (
        f"{backend}_receipt" if action.startswith("capture_") else "comparison_receipt"
    )
    prerequisites = [paths["source_manifest"]]
    if action == "compare":
        prerequisites.extend([
            paths["numpy_artifact"], paths["numpy_receipt"],
            paths["cupy_artifact"], paths["cupy_receipt"],
        ])
    for relative in prerequisites:
        if relative != paths["source_manifest"] and not (root / relative).is_file():
            raise EvidenceError(f"partial evidence: missing prerequisite {relative}")
    for relative in (paths[artifact_key], paths[receipt_key]):
        if os.path.lexists(root / relative):
            raise EvidenceError(f"refusing stale output path: {relative}")

    command = [
        python, "-m", "tools.execution_receipt", "run",
        "--root", str(root), "--artifact", paths[artifact_key],
        "--receipt", paths[receipt_key], "--source-manifest", paths["source_manifest"],
        "--git-sha", revision, "--host", host, "--device", device,
        "--env", "SIM_BACKEND", "--", *inner,
    ]
    envelope = {
        "schema": SCHEMA,
        "action": action,
        "promotion_value": diagnostic.PROMOTION_VALUE,
        "execution": "not_executed",
        "source_revision": revision,
        "source_manifest": {
            "path": paths["source_manifest"],
            "sha256": snapshot["manifest_sha256"],
            "tree_sha256": snapshot["tree_sha256"],
        },
        "artifact": paths[artifact_key],
        "receipt": paths[receipt_key],
        "prerequisites": prerequisites,
        "env": {"SIM_BACKEND": backend},
        "argv": command,
    }
    envelope["sha256"] = hashlib.sha256(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    _write_create_only(
        output, (json.dumps(envelope, indent=2, sort_keys=True) + "\n").encode("ascii"),
        "command envelope",
    )
    envelope["path"] = relative_out
    return envelope


def _comparison_receipt(
    *, root: Path, artifact_path: str | Path, receipt_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact_relative, artifact_file = _safe_relative(root, artifact_path, "comparison artifact")
    receipt_relative, _ = _safe_relative(root, receipt_path, "comparison receipt")
    try:
        artifact = json.loads(artifact_file.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError("comparison artifact is missing or invalid") from exc
    if (
        not isinstance(artifact, dict)
        or artifact.get("schema") != diagnostic.SCHEMA_COMPARISON
        or artifact.get("sha256") != diagnostic._artifact_digest(artifact)
        or artifact.get("promotion_value") != "none"
        or artifact.get("diagnostic_only") is not True
        or artifact.get("scientific_verdict") is not None
        or artifact.get("seed") != diagnostic.LOCKED_SEED
        or artifact.get("spec_sha256") != diagnostic.SPEC_SHA256
    ):
        raise EvidenceError("comparison artifact contract is invalid")
    comparisons = artifact.get("array_comparisons")
    if not isinstance(comparisons, dict) or set(comparisons) != set(diagnostic.ARRAYS):
        raise EvidenceError("comparison array evidence is incomplete")
    for name, row in comparisons.items():
        if (
            not isinstance(row, dict)
            or set(row) != {"dtype_equal", "shape_equal", "bytes_equal", "exact"}
            or any(type(row[field]) is not bool for field in row)
            or row["exact"] != (
                row["dtype_equal"] and row["shape_equal"] and row["bytes_equal"]
            )
        ):
            raise EvidenceError(f"comparison array evidence is invalid: {name}")
    passed = all(row["exact"] for row in comparisons.values())
    if (
        artifact.get("all_required_arrays_exact") is not passed
        or artifact.get("outcome") != ("DIAGNOSTIC_PASS" if passed else "DIAGNOSTIC_FAIL")
        or artifact.get("simulation_steps_executed") != {"numpy": 0, "cupy": 0}
        or not isinstance(artifact.get("config_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", artifact["config_sha256"]) is None
    ):
        raise EvidenceError("comparison outcome is inconsistent with its evidence")
    try:
        receipt = execution_receipt.verify_receipt(root, receipt_relative)
    except execution_receipt.ReceiptError as exc:
        raise EvidenceError(f"comparison receipt is invalid: {exc}") from exc
    if receipt["artifact"]["path"] != artifact_relative:
        raise EvidenceError("comparison receipt names a different artifact")
    if receipt["artifact"]["sha256"] != _sha256(artifact_file):
        raise EvidenceError("comparison receipt artifact digest mismatch")
    if receipt["env_allowlist"] != {"SIM_BACKEND": "numpy"}:
        raise EvidenceError("comparison receipt environment mismatch")
    source = artifact.get("source")
    if not isinstance(source, dict):
        raise EvidenceError("comparison source identity is missing")
    for key in ("git_sha", "kind", "manifest", "manifest_sha256", "tree_sha256", "file_count"):
        if source.get(key) != receipt.get("source", {}).get(key):
            raise EvidenceError("comparison source differs from its receipt")
    cells = artifact.get("cell_artifacts")
    if not isinstance(cells, dict) or set(cells) != {"numpy", "cupy"}:
        raise EvidenceError("comparison cell references are incomplete")
    argv = receipt.get("argv")
    if not isinstance(argv, list) or not argv or not Path(argv[0]).is_absolute():
        raise EvidenceError("comparison receipt command is invalid")
    expected = [
        argv[0], "-m",
        "research.runners._v13_backend_neutral_izh_initialization_diagnostic",
        "--spec", str((root / diagnostic.SPEC_PATH.relative_to(diagnostic.ROOT)).resolve()),
        "--spec-sha256", diagnostic.SPEC_SHA256,
        "--compare",
        "--numpy-artifact", str((root / cells["numpy"]["path"]).resolve()),
        "--numpy-receipt", str((root / cells["numpy"]["receipt_path"]).resolve()),
        "--cupy-artifact", str((root / cells["cupy"]["path"]).resolve()),
        "--cupy-receipt", str((root / cells["cupy"]["receipt_path"]).resolve()),
        "--out", str(artifact_file.resolve()),
    ]
    if argv != expected:
        raise EvidenceError("comparison receipt command differs from the frozen compare command")
    return artifact, receipt


def finalize_evidence(
    *, root: Path, artifact_path: str | Path, receipt_path: str | Path,
    out: str | Path,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    artifact, receipt = _comparison_receipt(
        root=root, artifact_path=artifact_path, receipt_path=receipt_path,
    )
    artifact_relative, artifact_file = _safe_relative(root, artifact_path, "comparison artifact")
    receipt_relative, receipt_file = _safe_relative(root, receipt_path, "comparison receipt")
    _, output = _safe_relative(root, out, "evidence manifest")
    manifest = {
        "schema": FINAL_MANIFEST_SCHEMA,
        "promotion_value": "none",
        "diagnostic_only": True,
        "scientific_verdict": None,
        "outcome": artifact["outcome"],
        "source": artifact["source"],
        "config_sha256": artifact["config_sha256"],
        "comparison_artifact": {
            "path": artifact_relative, "sha256": _sha256(artifact_file),
        },
        "comparison_receipt": {
            "path": receipt_relative, "sha256": _sha256(receipt_file),
            "host": receipt["host"], "device": receipt["device"],
            "started_utc_ns": receipt["started_utc_ns"],
            "ended_utc_ns": receipt["ended_utc_ns"],
        },
        "cell_artifacts": artifact["cell_artifacts"],
    }
    manifest["sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    _write_create_only(
        output, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("ascii"),
        "evidence manifest",
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    modes = parser.add_subparsers(dest="mode", required=True)
    source = modes.add_parser("freeze-source")
    source.add_argument("--revision", required=True)
    source.add_argument("--out", required=True)
    command = modes.add_parser("emit-command")
    command.add_argument("--action", choices=ACTIONS, required=True)
    command.add_argument("--revision", required=True)
    command.add_argument("--host", required=True)
    command.add_argument("--device", required=True)
    command.add_argument("--out", required=True)
    command.add_argument("--python", default=sys.executable)
    finalize = modes.add_parser("finalize")
    finalize.add_argument("--artifact", required=True)
    finalize.add_argument("--receipt", required=True)
    finalize.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.mode == "freeze-source":
            result = freeze_source_manifest(
                root=args.root, revision=args.revision, out=args.out,
            )
        elif args.mode == "emit-command":
            result = emit_command(
                root=args.root, action=args.action, revision=args.revision,
                host=args.host, device=args.device, out=args.out, python=args.python,
            )
        else:
            result = finalize_evidence(
                root=args.root, artifact_path=args.artifact,
                receipt_path=args.receipt, out=args.out,
            )
    except EvidenceError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
