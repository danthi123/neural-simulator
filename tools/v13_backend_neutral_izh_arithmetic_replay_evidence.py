#!/usr/bin/env python3
"""Freeze and receipt the create-only V13 strict-arithmetic replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

from research.runners import _v13_backend_neutral_izh_arithmetic_replay as replay
from tools import execution_receipt
from tools import v13_backend_neutral_izh_initialization_evidence as shared


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "v13-backend-neutral-izh-arithmetic-replay-command-v1"
FINAL_MANIFEST_SCHEMA = "v13-backend-neutral-izh-arithmetic-replay-evidence-manifest-v1"
ACTIONS = ("run_numpy", "run_cupy", "compare")
OUTPUT_DIR = "research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic"
_REVISION = re.compile(r"^[0-9a-f]{40}$")


EvidenceError = shared.EvidenceError


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_source_manifest(
    *, root: Path, revision: str, out: str | Path,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if _REVISION.fullmatch(revision) is None:
        raise EvidenceError("revision must be a full lowercase Git SHA")
    if shared._git_head(root) != revision:
        raise EvidenceError("source checkout is not at the requested revision")
    relative_out, output = shared._safe_relative(root, out, "source manifest")
    paths = set(replay.source_paths(root))
    local_sim = {
        relative for relative in paths
        if relative.startswith("sim/") and relative.endswith(".py")
    }
    if local_sim != shared._tracked_sim_python(root, revision):
        raise EvidenceError(
            "working simulator Python set differs from the frozen tracked simulator set"
        )
    entries: list[str] = []
    for relative in sorted(paths):
        _, path = shared._safe_relative(root, relative, "source path")
        if not path.is_file():
            raise EvidenceError(f"source path is missing: {relative}")
        working = path.read_bytes()
        if working != shared._git_file(root, revision, relative):
            raise EvidenceError(f"source path differs from committed revision: {relative}")
        entries.append(f"{hashlib.sha256(working).hexdigest()}  {relative}\n")
    shared._write_create_only(
        output, "".join(entries).encode("ascii"), "source manifest"
    )
    snapshot = execution_receipt.verify_source_manifest(root, relative_out)
    return {
        "path": relative_out,
        "sha256": snapshot["manifest_sha256"],
        "tree_sha256": snapshot["tree_sha256"],
        "file_count": snapshot["file_count"],
        "revision": revision,
    }


def _paths() -> dict[str, str]:
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
    module = "research.runners._v13_backend_neutral_izh_arithmetic_replay"
    common = [
        python, "-m", module,
        "--spec", str((root / replay.SPEC_RELATIVE_PATH).resolve()),
        "--spec-sha256", replay.SPEC_SHA256,
    ]
    if action.startswith("run_"):
        backend = action.removeprefix("run_")
        return [
            *common, "--run", "--backend", backend,
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
    if _REVISION.fullmatch(revision) is None or shared._git_head(root) != revision:
        raise EvidenceError("command source revision is not the current full Git SHA")
    if not host.strip() or not device.strip():
        raise EvidenceError("host and device must be explicit")
    if not Path(python).is_absolute() or not Path(python).is_file() or not os.access(
        python, os.X_OK
    ):
        raise EvidenceError("python must be an absolute executable path")
    relative_out, output = shared._safe_relative(root, out, "command envelope")
    if os.path.lexists(output):
        raise EvidenceError(f"refusing stale output path: {relative_out}")
    paths = _paths()
    manifest = root / paths["source_manifest"]
    if not manifest.is_file():
        raise EvidenceError("frozen source manifest is missing")
    try:
        snapshot = execution_receipt.verify_source_manifest(
            root, paths["source_manifest"]
        )
        execution_receipt._source_revision(root, revision, snapshot["manifest_sha256"])
    except execution_receipt.ReceiptError as exc:
        raise EvidenceError(f"source manifest verification failed: {exc}") from exc
    if set(snapshot["files"]) != set(replay.source_paths(root)):
        raise EvidenceError("source manifest contains the wrong source set")

    inner = _inner_command(
        root=root, action=action, revision=revision, paths=paths, python=python,
    )
    backend = action.removeprefix("run_") if action.startswith("run_") else "numpy"
    artifact_key = (
        f"{backend}_artifact" if action.startswith("run_") else "comparison_artifact"
    )
    receipt_key = (
        f"{backend}_receipt" if action.startswith("run_") else "comparison_receipt"
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
        "promotion_value": replay.PROMOTION_VALUE,
        "diagnostic_only": True,
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
    shared._write_create_only(
        output, (json.dumps(envelope, indent=2, sort_keys=True) + "\n").encode("ascii"),
        "command envelope",
    )
    envelope["path"] = relative_out
    return envelope


def _expected_comparison_argv(
    *, root: Path, artifact_file: Path, artifact: dict[str, Any], python: str,
) -> list[str]:
    cells = artifact["cell_artifacts"]
    return [
        python, "-m", "research.runners._v13_backend_neutral_izh_arithmetic_replay",
        "--spec", str((root / replay.SPEC_RELATIVE_PATH).resolve()),
        "--spec-sha256", replay.SPEC_SHA256,
        "--compare",
        "--numpy-artifact", str((root / cells["numpy"]["path"]).resolve()),
        "--numpy-receipt", str((root / cells["numpy"]["receipt_path"]).resolve()),
        "--cupy-artifact", str((root / cells["cupy"]["path"]).resolve()),
        "--cupy-receipt", str((root / cells["cupy"]["receipt_path"]).resolve()),
        "--out", str(artifact_file.resolve()),
    ]


def finalize_evidence(
    *, root: Path, artifact_path: str | Path, receipt_path: str | Path,
    out: str | Path,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    artifact_relative, artifact_file = shared._safe_relative(
        root, artifact_path, "comparison artifact"
    )
    receipt_relative, receipt_file = shared._safe_relative(
        root, receipt_path, "comparison receipt"
    )
    _, output = shared._safe_relative(root, out, "evidence manifest")
    try:
        artifact = json.loads(artifact_file.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError("comparison artifact is missing or invalid") from exc
    if (
        not isinstance(artifact, dict)
        or artifact.get("schema") != replay.SCHEMA_COMPARISON
        or artifact.get("sha256") != replay._artifact_digest(artifact)
        or artifact.get("promotion_value") != "none"
        or artifact.get("diagnostic_only") is not True
        or artifact.get("scientific_verdict") is not None
        or artifact.get("spec_sha256") != replay.SPEC_SHA256
        or artifact.get("simulation_steps_compared")
        != {"numpy": replay.TOTAL_STEPS, "cupy": replay.TOTAL_STEPS}
    ):
        raise EvidenceError("comparison artifact contract is invalid")
    comparisons = artifact.get("trajectory_comparisons")
    if not isinstance(comparisons, dict) or set(comparisons) != set(replay.TRAJECTORIES):
        raise EvidenceError("comparison trajectory evidence is incomplete")
    passed = all(row.get("exact") is True for row in comparisons.values())
    if (
        artifact.get("all_required_trajectories_exact") is not passed
        or artifact.get("outcome")
        != ("DIAGNOSTIC_PASS" if passed else "DIAGNOSTIC_FAIL")
    ):
        raise EvidenceError("comparison outcome is inconsistent with its evidence")
    try:
        receipt = execution_receipt.verify_receipt(root, receipt_relative)
    except execution_receipt.ReceiptError as exc:
        raise EvidenceError(f"comparison receipt is invalid: {exc}") from exc
    if (
        receipt["artifact"]["path"] != artifact_relative
        or receipt["artifact"]["sha256"] != _sha256(artifact_file)
        or receipt["env_allowlist"] != {"SIM_BACKEND": "numpy"}
        or receipt["source"] != artifact.get("source")
    ):
        raise EvidenceError("comparison receipt does not bind the comparison")

    cells = artifact.get("cell_artifacts", {})
    if not isinstance(cells, dict) or set(cells) != set(replay.BACKENDS):
        raise EvidenceError("comparison cell references are incomplete")
    argv = receipt.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or not Path(argv[0]).is_absolute()
        or argv != _expected_comparison_argv(
            root=root, artifact_file=artifact_file, artifact=artifact, python=argv[0],
        )
    ):
        raise EvidenceError(
            "comparison receipt command differs from the frozen compare command"
        )
    for backend in replay.BACKENDS:
        record = cells[backend]
        replay._load_cell(
            root / record["path"], root / record["receipt_path"], backend
        )
    manifest = {
        "schema": FINAL_MANIFEST_SCHEMA,
        "promotion_value": "none",
        "diagnostic_only": True,
        "scientific_verdict": None,
        "outcome": artifact["outcome"],
        "source": artifact["source"],
        "completed_input": artifact["completed_input"],
        "comparison": {
            "path": artifact_relative,
            "sha256": _sha256(artifact_file),
            "artifact_sha256": artifact["sha256"],
            "receipt_path": receipt_relative,
            "receipt_sha256": _sha256(receipt_file),
        },
        "cells": cells,
    }
    manifest["sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    shared._write_create_only(
        output, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("ascii"),
        "evidence manifest",
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-source")
    freeze.add_argument("--root", type=Path, default=ROOT)
    freeze.add_argument("--revision", required=True)
    freeze.add_argument("--out", default=f"{OUTPUT_DIR}/source.sha256")
    emit = commands.add_parser("emit")
    emit.add_argument("--root", type=Path, default=ROOT)
    emit.add_argument("--action", choices=ACTIONS, required=True)
    emit.add_argument("--revision", required=True)
    emit.add_argument("--host", required=True)
    emit.add_argument("--device", required=True)
    emit.add_argument("--out", required=True)
    emit.add_argument("--python", default=sys.executable)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--root", type=Path, default=ROOT)
    finalize.add_argument(
        "--artifact", default=f"{OUTPUT_DIR}/comparison.json"
    )
    finalize.add_argument(
        "--receipt", default=f"{OUTPUT_DIR}/comparison.receipt.json"
    )
    finalize.add_argument(
        "--out", default=f"{OUTPUT_DIR}/evidence-manifest.json"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze-source":
        result = freeze_source_manifest(
            root=args.root, revision=args.revision, out=args.out,
        )
    elif args.command == "emit":
        result = emit_command(
            root=args.root, action=args.action, revision=args.revision,
            host=args.host, device=args.device, out=args.out, python=args.python,
        )
    else:
        result = finalize_evidence(
            root=args.root, artifact_path=args.artifact,
            receipt_path=args.receipt, out=args.out,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
