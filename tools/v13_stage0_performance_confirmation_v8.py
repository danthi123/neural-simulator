#!/usr/bin/env python3
"""Confirm the V13 Stage-0 performance boundary after a process-only correction."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

try:
    from tools import v13_stage0_performance_continuation as v7
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools import v13_stage0_performance_continuation as v7


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = "research/specs/v13_tonic_output_stage0_performance_confirmation_v8.json"
RAW_PATH = "research/findings/raw/v13_tonic_output_stage0_performance_confirmation_v8"
V6_RAW = "research/findings/raw/v13_tonic_output_stage0_process_correction_v6"
V6_CONFIG = "research/specs/v13_stage0_controller_config_v8.json"
V6_CONFIG_FILE_SHA256 = "6367b975df3ae1664cad5ea5a7d4ed747c44ec7c0ea21f6d36d5af1f43364b93"
CANDIDATE_REVISION = "1ecc85cd698539a6ef92e112d2c49092cfa21f1e"
CANDIDATE_MANIFEST = "research/specs/v13_stage0_candidate_source_v10.sha256"
CANDIDATE_MANIFEST_SHA256 = "04214978e26efbe3a7014fa2ced6a52f8600419ef4b8ffd6df40bd466cc05a73"
LEGACY_TRANSFER = "legacy-baseline"
CANDIDATE_RUNTIME = f"{RAW_PATH}/candidate-runtime"


def _candidate_paths(candidate_root: Path) -> dict[str, Path]:
    runtime = candidate_root / CANDIDATE_RUNTIME
    return {
        "runtime": runtime,
        "baseline": runtime / v7.legacy.LEGACY_OUTPUT_NAME,
        "artifact": runtime / "performance-candidate.json",
        "receipt": runtime / "performance-candidate.receipt.json",
        "sidecar": runtime / "performance-candidate.json.prov.json",
    }


def _bind_v7_helpers() -> None:
    """Reuse the tested V7 validators while binding every mutable identity to V8."""
    v7.SPEC_PATH = SPEC_PATH
    v7.RAW_PATH = RAW_PATH
    v7.V6_RAW = V6_RAW
    v7.V6_CONFIG = V6_CONFIG
    v7.V6_CONFIG_FILE_SHA256 = V6_CONFIG_FILE_SHA256
    v7.CANDIDATE_REVISION = CANDIDATE_REVISION
    v7.CANDIDATE_MANIFEST = CANDIDATE_MANIFEST
    v7.CANDIDATE_MANIFEST_SHA256 = CANDIDATE_MANIFEST_SHA256
    v7.LEGACY_TRANSFER = LEGACY_TRANSFER
    v7._candidate_paths = _candidate_paths


_bind_v7_helpers()


def readiness(*, root: Path = ROOT) -> dict:
    return v7.readiness(root=root)


def run_candidate(*, root: Path, candidate_root: Path) -> dict:
    return v7.run_candidate(root=root, candidate_root=candidate_root)


def transfer_candidate(*, root: Path, candidate_root: Path) -> dict:
    readiness(root=root)
    candidate_root = candidate_root.resolve(strict=True)
    if v7._git_head(candidate_root) != CANDIDATE_REVISION:
        raise v7.ContinuationError("candidate checkout identity changed before transfer")
    paths = _candidate_paths(candidate_root)
    relative_receipt = paths["receipt"].relative_to(candidate_root).as_posix()
    try:
        receipt = v7.execution_receipt.verify_receipt(candidate_root, relative_receipt)
    except v7.execution_receipt.ReceiptError as exc:
        raise v7.ContinuationError(f"candidate receipt is invalid: {exc}") from exc
    destination = root / RAW_PATH / "candidate-performance"
    if os.path.lexists(destination):
        raise v7.ContinuationError(f"refusing existing candidate transfer: {destination}")
    destination.mkdir(parents=True, mode=0o755)
    records: dict[str, dict[str, int | str]] = {}
    try:
        for name, source in (
            ("performance-candidate.json", paths["artifact"]),
            ("performance-candidate.receipt.json", paths["receipt"]),
            ("performance-candidate.json.prov.json", paths["sidecar"]),
        ):
            data = v7._regular(source, f"candidate {name}")
            (destination / name).write_bytes(data)
            records[name] = {"sha256": v7._sha256(data), "size_bytes": len(data)}
        artifact = json.loads(v7._regular(paths["artifact"], "candidate performance artifact"))
        env = receipt.get("env_allowlist", {})
        backend = env.get("SIM_BACKEND")
        if not isinstance(backend, str) or not isinstance(receipt.get("device"), str):
            raise v7.ContinuationError("candidate receipt lacks backend or device identity")
        value = {
            "schema": "v13-stage0-candidate-performance-transfer-v1",
            "status": "transferred",
            "backend": backend,
            "device": receipt["device"],
            "seed": artifact.get("seed"),
            "runner": "tools/v13_stage0_performance_confirmation_v8.py",
            "source_revision": CANDIDATE_REVISION,
            "source_manifest_sha256": CANDIDATE_MANIFEST_SHA256,
            "receipt_source": receipt["source"],
            "provenance": {
                "receipt_sha256": records["performance-candidate.receipt.json"]["sha256"],
                "sidecar_sha256": records["performance-candidate.json.prov.json"]["sha256"],
            },
            "files": records,
        }
        value["sha256"] = v7._canonical_digest(value)
        v7._write_exclusive(destination / "transfer-manifest.json", value)
    except Exception:
        shutil.rmtree(destination)
        raise
    return value


def finalize(*, root: Path = ROOT) -> dict:
    """Finalize only a complete, receipt-backed V8 measurement."""
    accepted = v7.validate_v6(root=root)
    transfer_root = root / RAW_PATH / "candidate-performance"
    transfer = v7._read_json(transfer_root / "transfer-manifest.json")
    if transfer.get("sha256") != v7._canonical_digest(transfer):
        raise v7.ContinuationError("candidate transfer manifest self-digest is invalid")
    if (
        transfer.get("schema") != "v13-stage0-candidate-performance-transfer-v1"
        or transfer.get("status") != "transferred"
    ):
        raise v7.ContinuationError("V8 requires a complete execution receipt transfer")
    for name, binding in transfer.get("files", {}).items():
        data = v7._regular(transfer_root / name, f"transferred candidate {name}")
        if binding != {"sha256": v7._sha256(data), "size_bytes": len(data)}:
            raise v7.ContinuationError(f"candidate transfer file changed: {name}")
    artifact_bytes = v7._regular(
        transfer_root / "performance-candidate.json", "performance artifact"
    )
    artifact = json.loads(artifact_bytes)
    performance_go = v7._validate_performance(artifact)
    final = {
        "schema": "v13-stage0-performance-confirmation-final-v1",
        "stage": "performance_confirmation",
        "status": "complete",
        "go": performance_go,
        "outcome": "TONIC_OUTPUT_GO" if performance_go else "TONIC_OUTPUT_NO_GO",
        "measured_performance_outcome": artifact["outcome"],
        "promotion_eligible": performance_go,
        "candidate_receipt_complete": True,
        "backend": "cross_backend",
        "device": "NumPy CPU and NVIDIA GeForce RTX 3090",
        "runner": "tools/v13_stage0_performance_confirmation_v8.py",
        "config": SPEC_PATH,
        "preconditions": [
            {"name": "sealed_v6_physiology", "ok": True},
            {"name": "candidate_measurement_complete", "ok": True},
            {"name": "candidate_receipt_complete", "ok": True},
        ],
        "selected_current_pA": 100.0,
        "checks": {"sealed_v6_physiology": True, "performance": performance_go},
        "v6_inputs": accepted,
        "performance_artifact_sha256": v7._sha256(artifact_bytes),
        "performance_transfer_sha256": transfer["sha256"],
        "stage1_seed_1031": "sealed-not-read-or-executed",
    }
    final["sha256"] = v7._canonical_digest(final)
    v7._write_exclusive(root / RAW_PATH / "final-stage0-v8.json", final)
    return final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("readiness")
    candidate = commands.add_parser("run-candidate")
    candidate.add_argument("--candidate-root", type=Path, required=True)
    transfer = commands.add_parser("transfer-candidate")
    transfer.add_argument("--candidate-root", type=Path, required=True)
    commands.add_parser("finalize")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "readiness":
            result = readiness(root=args.root)
        elif args.command == "run-candidate":
            result = run_candidate(root=args.root, candidate_root=args.candidate_root)
        elif args.command == "transfer-candidate":
            result = transfer_candidate(root=args.root, candidate_root=args.candidate_root)
        else:
            result = finalize(root=args.root)
    except (v7.ContinuationError, OSError, ValueError) as exc:
        print(f"v13-stage0-performance-confirmation-v8: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
