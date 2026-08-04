#!/usr/bin/env python3
"""Continue V13 Stage 0 through its two frozen performance measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import sys
from typing import Any

try:
    from tools import execution_receipt
    from tools import v13_legacy_source_package as legacy
except ModuleNotFoundError:
    import execution_receipt  # type: ignore[no-redef]
    import v13_legacy_source_package as legacy  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = "research/specs/v13_tonic_output_stage0_performance_continuation_v7.json"
RAW_PATH = "research/findings/raw/v13_tonic_output_stage0_performance_continuation_v7"
V6_RAW = "research/findings/raw/v13_tonic_output_stage0_process_correction_v6"
V6_CONFIG = "research/specs/v13_stage0_controller_config_v8.json"
V6_CONFIG_FILE_SHA256 = "6367b975df3ae1664cad5ea5a7d4ed747c44ec7c0ea21f6d36d5af1f43364b93"
CANDIDATE_REVISION = "1bec3c22ad7c535a2cbb27860e5bf4cfd51d6d6f"
CANDIDATE_MANIFEST = "research/specs/v13_stage0_candidate_source_v9.sha256"
CANDIDATE_MANIFEST_SHA256 = "e4d34d40e5176e52b42621635a18f40e9e6783c326ee61d93275e91efb330003"
PYTHON = "/home/dant123/Projects/sim/.venv/bin/python"
RUNNER_MODULE = "research.runners._vocal_action_credit_gate_v13_tonic_output"
LEGACY_TRANSFER = "legacy-baseline"
CANDIDATE_RUNTIME = ".v13-stage0-performance-v7"
MANIFEST_SHA256 = {
    "calibration_selection": "ef2c8d3b20a3e3e8a99be18a45763b02a45901851884c4ef5c8371fbd5a8eb62",
    "replication_numpy": "0dc33e6b312c4831eaa3ffe94beafc825cf678499c673db6bf1ff7095775fcf0",
    "replication_cupy": "8b117a009f605b1bc4805c9e885250688f56e5b1c833bfac3ea3f68b1bff2c8e",
    "held_out_cupy": "600da09c5db47cb61be4051d720717994b98338d37fab3bef874841fb3c101cd",
    "held_out_numpy": "d203c7561049d3a2cace03e3f0e4ba1d0d50272b14171e5af25a368f915d3017",
}
ARTIFACTS = {
    "calibration_selection": "calibration-selection.json",
    "replication_numpy": "replication-numpy.json",
    "replication_cupy": "replication-cupy.json",
    "held_out_cupy": "held-out-cupy.json",
    "held_out_numpy": "held-out-numpy.json",
}
OUTCOMES = {
    "calibration_selection": "CALIBRATION_GO",
    "replication_numpy": "REPLICATION_GO",
    "replication_cupy": "REPLICATION_GO",
    "held_out_cupy": "HELD_OUT_GO",
    "held_out_numpy": "HELD_OUT_GO",
}


class ContinuationError(RuntimeError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_digest(value: dict[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "sha256"}
    return _sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContinuationError(f"cannot read stable JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContinuationError(f"evidence is not a JSON object: {path}")
    return value


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _regular(path: Path, label: str) -> bytes:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ContinuationError(f"missing {label}: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ContinuationError(f"{label} is not a regular file: {path}")
    return path.read_bytes()


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ContinuationError(f"cannot resolve Git identity: {result.stderr.strip()}")
    return result.stdout.strip()


def validate_v6(*, root: Path = ROOT) -> dict[str, Any]:
    config_bytes = _regular(root / V6_CONFIG, "V6 controller config")
    if _sha256(config_bytes) != V6_CONFIG_FILE_SHA256:
        raise ContinuationError("V6 controller config bytes changed")
    accepted: dict[str, Any] = {}
    for kind, digest in MANIFEST_SHA256.items():
        manifest_path = root / V6_RAW / "manifests" / f"{ARTIFACTS[kind]}"
        manifest_bytes = _regular(manifest_path, f"V6 {kind} manifest")
        if _sha256(manifest_bytes) != digest:
            raise ContinuationError(f"V6 {kind} manifest digest changed")
        manifest = json.loads(manifest_bytes)
        if manifest.get("sha256") != _canonical_digest(manifest):
            raise ContinuationError(f"V6 {kind} manifest self-digest is invalid")
        artifact_path = root / V6_RAW / ARTIFACTS[kind]
        artifact_bytes = _regular(artifact_path, f"V6 {kind} artifact")
        binding = manifest.get("artifact", {})
        if binding.get("sha256") != _sha256(artifact_bytes):
            raise ContinuationError(f"V6 {kind} artifact differs from its sealed manifest")
        artifact = json.loads(artifact_bytes)
        if artifact.get("go") is not True or artifact.get("outcome") != OUTCOMES[kind]:
            raise ContinuationError(f"V6 {kind} did not earn its required GO")
        accepted[kind] = {"manifest_sha256": digest, "artifact_sha256": _sha256(artifact_bytes)}
    return accepted


def readiness(*, root: Path = ROOT) -> dict[str, Any]:
    accepted = validate_v6(root=root)
    spec = _read_json(root / SPEC_PATH)
    if spec.get("sha256") != _canonical_digest(spec):
        raise ContinuationError("V7 preregistration self-digest is invalid")
    return {
        "schema": "v13-stage0-performance-continuation-readiness-v1",
        "status": "READY",
        "v6_inputs": accepted,
        "candidate_revision": CANDIDATE_REVISION,
        "legacy_base_revision": legacy.BASE_REVISION,
        "legacy_overlay_revision": legacy.CANDIDATE_REVISION,
        "stage1_seed_1031": "sealed-not-read-or-executed",
    }


def run_legacy(*, root: Path, package_path: Path) -> dict[str, Any]:
    readiness(root=root)
    raw = root / RAW_PATH
    raw.mkdir(parents=True, exist_ok=True)
    legacy.build_package(root=root, output=package_path)
    legacy.probe_execution_package(package_path, python=PYTHON)
    execution = legacy.execute_legacy_baseline(package_path, python=PYTHON)
    transfer = legacy.transfer_legacy_artifact(
        package_path, candidate_evidence=raw, transfer_name=LEGACY_TRANSFER,
    )
    return {"status": "LEGACY_BASELINE_RECORDED", "execution": execution, "transfer": transfer}


def _candidate_paths(candidate_root: Path) -> dict[str, Path]:
    runtime = candidate_root / CANDIDATE_RUNTIME
    return {
        "runtime": runtime,
        "baseline": runtime / legacy.LEGACY_OUTPUT_NAME,
        "artifact": runtime / "performance-candidate.json",
        "receipt": runtime / "performance-candidate.receipt.json",
        "sidecar": runtime / "performance-candidate.json.prov.json",
    }


def run_candidate(*, root: Path, candidate_root: Path) -> dict[str, Any]:
    readiness(root=root)
    candidate_root = candidate_root.resolve(strict=True)
    if _git_head(candidate_root) != CANDIDATE_REVISION:
        raise ContinuationError("candidate checkout is not the frozen V6 source revision")
    manifest = _regular(candidate_root / CANDIDATE_MANIFEST, "candidate source manifest")
    if _sha256(manifest) != CANDIDATE_MANIFEST_SHA256:
        raise ContinuationError("candidate source manifest changed")
    legacy_root = root / RAW_PATH / LEGACY_TRANSFER
    transfer = _read_json(legacy_root / legacy.TRANSFER_MANIFEST_NAME)
    if transfer.get("sha256") != legacy._canonical_digest(transfer):
        raise ContinuationError("legacy transfer manifest self-digest is invalid")
    source = legacy_root / legacy.LEGACY_OUTPUT_NAME
    source_bytes = _regular(source, "transferred legacy baseline")
    receipt_bytes = _regular(
        legacy_root / legacy.EXECUTION_RECEIPT_NAME,
        "transferred legacy execution receipt",
    )
    if (
        transfer.get("artifact", {}).get("sha256") != _sha256(source_bytes)
        or transfer.get("execution_receipt", {}).get("sha256")
        != _sha256(receipt_bytes)
    ):
        raise ContinuationError("legacy transfer evidence changed after sealing")
    paths = _candidate_paths(candidate_root)
    if os.path.lexists(paths["runtime"]):
        raise ContinuationError(f"refusing existing candidate runtime: {paths['runtime']}")
    paths["runtime"].mkdir(mode=0o700)
    paths["baseline"].write_bytes(source_bytes)
    env = os.environ.copy()
    env["SIM_BACKEND"] = "cupy"
    command = [
        PYTHON, "-m", RUNNER_MODULE, "--performance", "--old-baseline",
        str(paths["baseline"]), "--out", str(paths["artifact"]),
    ]
    try:
        receipt = execution_receipt.run_and_receipt(
            root=candidate_root,
            artifact_path=paths["artifact"].relative_to(candidate_root).as_posix(),
            receipt_path=paths["receipt"].relative_to(candidate_root).as_posix(),
            source_manifest=CANDIDATE_MANIFEST,
            git_sha=CANDIDATE_REVISION,
            host=socket.gethostname(),
            device="NVIDIA GeForce RTX 3090",
            argv=command,
            env_allowlist=["SIM_BACKEND"],
            environ=env,
            provenance_v2=True,
        )
    except execution_receipt.ReceiptError as exc:
        raise ContinuationError(f"candidate performance execution failed: {exc}") from exc
    return {"status": "CANDIDATE_PERFORMANCE_RECORDED", "receipt": receipt}


def transfer_candidate(*, root: Path, candidate_root: Path) -> dict[str, Any]:
    readiness(root=root)
    candidate_root = candidate_root.resolve(strict=True)
    if _git_head(candidate_root) != CANDIDATE_REVISION:
        raise ContinuationError("candidate checkout identity changed before transfer")
    paths = _candidate_paths(candidate_root)
    relative_receipt = paths["receipt"].relative_to(candidate_root).as_posix()
    try:
        receipt = execution_receipt.verify_receipt(candidate_root, relative_receipt)
    except execution_receipt.ReceiptError as exc:
        raise ContinuationError(f"candidate receipt is invalid: {exc}") from exc
    destination = root / RAW_PATH / "candidate-performance"
    if os.path.lexists(destination):
        raise ContinuationError(f"refusing existing candidate transfer: {destination}")
    destination.mkdir(parents=True, mode=0o755)
    records: dict[str, Any] = {}
    try:
        for name, source in (
            ("performance-candidate.json", paths["artifact"]),
            ("performance-candidate.receipt.json", paths["receipt"]),
            ("performance-candidate.json.prov.json", paths["sidecar"]),
        ):
            data = _regular(source, f"candidate {name}")
            target = destination / name
            target.write_bytes(data)
            records[name] = {"sha256": _sha256(data), "size_bytes": len(data)}
        value = {
            "schema": "v13-stage0-candidate-performance-transfer-v1",
            "status": "transferred",
            "source_revision": CANDIDATE_REVISION,
            "source_manifest_sha256": CANDIDATE_MANIFEST_SHA256,
            "receipt_source": receipt["source"],
            "files": records,
        }
        value["sha256"] = _canonical_digest(value)
        _write_exclusive(destination / "transfer-manifest.json", value)
    except Exception:
        shutil.rmtree(destination)
        raise
    return value


def preserve_failed_candidate(*, root: Path, candidate_root: Path) -> dict[str, Any]:
    """Preserve only a measured NO-GO after the sidecar/receipt path failed."""
    readiness(root=root)
    candidate_root = candidate_root.resolve(strict=True)
    if _git_head(candidate_root) != CANDIDATE_REVISION:
        raise ContinuationError("candidate checkout identity changed before preservation")
    paths = _candidate_paths(candidate_root)
    if paths["receipt"].exists() or paths["sidecar"].exists():
        raise ContinuationError("failure preservation refuses receipt or sidecar evidence")
    artifact_bytes = _regular(paths["artifact"], "unsealed candidate artifact")
    artifact = json.loads(artifact_bytes)
    if _validate_performance(artifact) is not False:
        raise ContinuationError("failure preservation accepts only measured PERFORMANCE_NO_GO")
    config = _read_json(root / V6_CONFIG)
    if artifact.get("source_identity") != config.get("candidate_source_identity"):
        raise ContinuationError("unsealed artifact source identities differ from V6")
    baseline_bytes = _regular(
        root / RAW_PATH / LEGACY_TRANSFER / legacy.LEGACY_OUTPUT_NAME,
        "sealed legacy baseline",
    )
    if artifact.get("old_baseline") != json.loads(baseline_bytes):
        raise ContinuationError("unsealed artifact embeds a different legacy baseline")

    run_log = candidate_root / "research/findings/raw/_provenance/runs.jsonl"
    records = []
    for line in _regular(run_log, "candidate provenance start log").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            record.get("provenance_schema") == "sim-run-provenance-v2"
            and record.get("git_sha") == CANDIDATE_REVISION
            and record.get("source_manifest_sha256") == CANDIDATE_MANIFEST_SHA256
            and record.get("env") == {"SIM_BACKEND": "cupy"}
            and str(paths["artifact"]) in record.get("argv", [])
        ):
            records.append(record)
    if len(records) != 1:
        raise ContinuationError(
            f"expected one matching failed-receipt run record, found {len(records)}"
        )
    run_record = records[0]
    if (
        not isinstance(run_record.get("run_id"), str)
        or len(run_record["run_id"]) != 64
        or type(run_record.get("started_utc_ns")) is not int
        or run_record["started_utc_ns"] <= 0
    ):
        raise ContinuationError("failed-receipt run identity or timing is invalid")

    destination = root / RAW_PATH / "candidate-performance"
    if os.path.lexists(destination):
        raise ContinuationError(f"refusing existing candidate transfer: {destination}")
    destination.mkdir(parents=True, mode=0o755)
    run_bytes = (json.dumps(run_record, indent=2, sort_keys=True) + "\n").encode("ascii")
    try:
        (destination / "performance-candidate.json").write_bytes(artifact_bytes)
        (destination / "provenance-start-record.json").write_bytes(run_bytes)
        records_out = {
            "performance-candidate.json": {
                "sha256": _sha256(artifact_bytes), "size_bytes": len(artifact_bytes),
            },
            "provenance-start-record.json": {
                "sha256": _sha256(run_bytes), "size_bytes": len(run_bytes),
            },
        }
        value = {
            "schema": "v13-stage0-candidate-performance-failed-receipt-v1",
            "status": "measured_no_go_receipt_failed",
            "source_revision": CANDIDATE_REVISION,
            "source_manifest_sha256": CANDIDATE_MANIFEST_SHA256,
            "run_id": run_record["run_id"],
            "receipt_failure": "artifact path outside provenance scanner raw root",
            "scientific_disposition": "PERFORMANCE_NO_GO",
            "rerun_authorized": False,
            "files": records_out,
        }
        value["sha256"] = _canonical_digest(value)
        _write_exclusive(destination / "transfer-manifest.json", value)
    except Exception:
        shutil.rmtree(destination)
        raise
    return value


def _validate_performance(artifact: dict[str, Any]) -> bool:
    checks = artifact.get("checks")
    ratios = artifact.get("ratios")
    cells = artifact.get("cells")
    baseline = artifact.get("old_baseline")
    if (
        artifact.get("stage") != "performance"
        or artifact.get("source_sha") != CANDIDATE_REVISION
        or artifact.get("backend") != "cupy"
        or "3090" not in str(artifact.get("device", ""))
        or not isinstance(checks, dict)
        or set(checks) != {
            "old_baseline_supplied", "default_off_ratio", "normal_active_ratio",
            "v1_active_ratio", "v2_active_ratio", "feature_storage",
            "default_does_not_allocate", "v1_dispatches", "v2_dispatches",
        }
        or not isinstance(ratios, dict)
        or set(ratios) != {"default_vs_old", "normal_active", "v1_active", "v2_active"}
        or not isinstance(cells, dict)
        or set(cells) != {
            "normal_default", "normal_active", "v1_default", "v1_active",
            "v2_default", "v2_active",
        }
        or not isinstance(baseline, dict)
        or baseline.get("source_sha") != legacy.BASE_REVISION
        or baseline.get("outcome") != "BASELINE_RECORDED"
    ):
        raise ContinuationError("candidate performance artifact has an invalid structure")
    expected_ratios = {
        "default_vs_old": cells["normal_default"]["median_seconds"] / baseline["median_seconds"],
        "normal_active": cells["normal_active"]["median_seconds"] / cells["normal_default"]["median_seconds"],
        "v1_active": cells["v1_active"]["median_seconds"] / cells["v1_default"]["median_seconds"],
        "v2_active": cells["v2_active"]["median_seconds"] / cells["v2_default"]["median_seconds"],
    }
    if any(
        not isinstance(ratios[name], (int, float))
        or isinstance(ratios[name], bool)
        or not math.isclose(float(ratios[name]), expected, rel_tol=1e-12, abs_tol=1e-12)
        for name, expected in expected_ratios.items()
    ):
        raise ContinuationError("candidate performance ratios differ from measured medians")
    expected_checks = {
        "old_baseline_supplied": True,
        "default_off_ratio": expected_ratios["default_vs_old"] <= 1.02,
        "normal_active_ratio": expected_ratios["normal_active"] <= 1.10,
        "v1_active_ratio": expected_ratios["v1_active"] <= 1.10,
        "v2_active_ratio": expected_ratios["v2_active"] <= 1.10,
        "feature_storage": all(
            all(value <= 4 * 600 for value in row["feature_bytes"])
            for row in cells.values()
        ),
        "default_does_not_allocate": all(
            value == 0 for name, row in cells.items() if name.endswith("_default")
            for value in row["feature_bytes"]
        ),
        "v1_dispatches": all(cells["v1_active"]["megakernel_dispatch"]),
        "v2_dispatches": all(cells["v2_active"]["megakernel_dispatch"]),
    }
    if checks != expected_checks:
        raise ContinuationError("candidate performance checks differ from raw measurements")
    measured_go = all(expected_checks.values())
    expected_outcome = "PERFORMANCE_GO" if measured_go else "PERFORMANCE_NO_GO"
    if artifact.get("go") is not measured_go or artifact.get("outcome") != expected_outcome:
        raise ContinuationError("candidate performance verdict differs from its measurements")
    return measured_go


def finalize(*, root: Path = ROOT) -> dict[str, Any]:
    accepted = validate_v6(root=root)
    transfer_root = root / RAW_PATH / "candidate-performance"
    transfer = _read_json(transfer_root / "transfer-manifest.json")
    if transfer.get("sha256") != _canonical_digest(transfer):
        raise ContinuationError("candidate transfer manifest self-digest is invalid")
    for name, binding in transfer.get("files", {}).items():
        data = _regular(transfer_root / name, f"transferred candidate {name}")
        if binding != {"sha256": _sha256(data), "size_bytes": len(data)}:
            raise ContinuationError(f"candidate transfer file changed: {name}")
    artifact_bytes = _regular(transfer_root / "performance-candidate.json", "performance artifact")
    if transfer.get("files", {}).get("performance-candidate.json", {}).get("sha256") != _sha256(artifact_bytes):
        raise ContinuationError("candidate transfer no longer binds the performance artifact")
    artifact = json.loads(artifact_bytes)
    performance_go = _validate_performance(artifact)
    receipt_complete = True
    if transfer.get("schema") == "v13-stage0-candidate-performance-failed-receipt-v1":
        receipt_complete = False
        if (
            performance_go
            or transfer.get("status") != "measured_no_go_receipt_failed"
            or transfer.get("scientific_disposition") != "PERFORMANCE_NO_GO"
            or transfer.get("rerun_authorized") is not False
        ):
            raise ContinuationError("failed-receipt transfer cannot support this verdict")
    elif (
        transfer.get("schema") != "v13-stage0-candidate-performance-transfer-v1"
        or transfer.get("status") != "transferred"
    ):
        raise ContinuationError("candidate transfer schema or status is invalid")
    outcome = (
        "TONIC_OUTPUT_GO" if performance_go
        else "TONIC_OUTPUT_NO_GO" if receipt_complete
        else "TONIC_OUTPUT_UNDEFINED"
    )
    final = {
        "schema": "v13-stage0-performance-continuation-final-v1",
        "stage": "final_cross_backend",
        "status": (
            "complete" if receipt_complete
            else "undefined-receipt-failed"
        ),
        "go": performance_go,
        "outcome": outcome,
        "measured_performance_outcome": artifact["outcome"],
        "promotion_eligible": performance_go and receipt_complete,
        "candidate_receipt_complete": receipt_complete,
        "backend": "cross_backend",
        "device": "NumPy CPU and NVIDIA GeForce RTX 3090",
        "runner": "tools/v13_stage0_performance_continuation.py",
        "config": SPEC_PATH,
        "preconditions": {
            "sealed_v6_physiology": True,
            "candidate_measurement_complete": True,
            "candidate_receipt_complete": receipt_complete,
        },
        "selected_current_pA": 100.0,
        "checks": {"sealed_v6_physiology": True, "performance": performance_go},
        "v6_inputs": accepted,
        "performance_artifact_sha256": _sha256(artifact_bytes),
        "performance_transfer_sha256": transfer["sha256"],
        "stage1_seed_1031": "sealed-not-read-or-executed",
    }
    final["sha256"] = _canonical_digest(final)
    _write_exclusive(root / RAW_PATH / "final-stage0-v7.json", final)
    return final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("readiness")
    old = commands.add_parser("run-legacy")
    old.add_argument("--package", type=Path, required=True)
    candidate = commands.add_parser("run-candidate")
    candidate.add_argument("--candidate-root", type=Path, required=True)
    transfer = commands.add_parser("transfer-candidate")
    transfer.add_argument("--candidate-root", type=Path, required=True)
    failed = commands.add_parser("preserve-failed-candidate")
    failed.add_argument("--candidate-root", type=Path, required=True)
    commands.add_parser("finalize")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "readiness":
            result = readiness(root=args.root)
        elif args.command == "run-legacy":
            result = run_legacy(root=args.root, package_path=args.package)
        elif args.command == "run-candidate":
            result = run_candidate(root=args.root, candidate_root=args.candidate_root)
        elif args.command == "transfer-candidate":
            result = transfer_candidate(root=args.root, candidate_root=args.candidate_root)
        elif args.command == "preserve-failed-candidate":
            result = preserve_failed_candidate(
                root=args.root, candidate_root=args.candidate_root,
            )
        else:
            result = finalize(root=args.root)
    except (ContinuationError, legacy.PackageError, OSError, ValueError) as exc:
        print(f"v13-stage0-performance-continuation: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
