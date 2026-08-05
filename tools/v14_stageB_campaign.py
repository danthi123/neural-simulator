#!/usr/bin/env python3
"""Materialize the exact V14 Stage B candidate screen into GPU batch declarations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.runners.v14_stageB_batched_physiology import (
    DECLARATION_SCHEMA,
    PHASED_ANALYSIS_PROTOCOL_SCHEMA,
    PHASED_OUTPUT_SCHEMA,
    READINESS_ARMS,
    run_authenticated_gpu_batch,
)
import numpy as np
from sim.snr_executable_packet import canonical_bytes
from tools.v14_stageB_candidate_batch import (
    EXACT_SCREEN_COUNT,
    MANIFEST_SCHEMA as CANDIDATE_MANIFEST_SCHEMA,
    SUCCESSOR_MANIFEST_SCHEMA as SUCCESSOR_CANDIDATE_MANIFEST_SCHEMA,
)
from tools.v14_stageB_packet_compiler import compile_candidate
from tools.v14_stageB_packet_verifier import verify_candidate
from tools.compact_trace import save_compact_trace


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_SCHEMA = "v14-snr-stageB-screen-campaign-v1"
PHASED_CAMPAIGN_SCHEMA = "v14-snr-stageB-screen-campaign-v2"
GPU_BATCH_RECEIPT_SCHEMA = "v14-snr-stageB-gpu-batch-receipt-v1"
PHASED_GPU_BATCH_RECEIPT_SCHEMA = "v14-snr-stageB-gpu-batch-receipt-v2"
DEFAULT_BATCH_SIZE = 64


class StageBCampaignError(ValueError):
    """Raised when an exact campaign cannot be authenticated or materialized."""


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
        raise StageBCampaignError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _inside_regular(root: Path, value: str | Path, context: str) -> tuple[str, Path]:
    supplied = Path(value).expanduser()
    path = (supplied if supplied.is_absolute() else root / supplied).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise StageBCampaignError(f"{context} must be inside repository_root") from exc
    if path.is_symlink() or not path.is_file():
        raise StageBCampaignError(f"{context} must be a regular file")
    pure = PurePosixPath(relative.as_posix())
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise StageBCampaignError(f"{context} path is not canonical")
    return pure.as_posix(), path


def _load_bound_json(
    root: Path, value: str | Path, expected_sha256: str, context: str
) -> tuple[dict[str, str], dict[str, Any]]:
    relative, path = _inside_regular(root, value, context)
    expected = _sha256(expected_sha256, f"{context} sha256")
    raw = path.read_bytes()
    if _digest_bytes(raw) != expected:
        raise StageBCampaignError(f"{context} digest does not match")
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StageBCampaignError(f"{context} is not valid JSON") from exc
    if not isinstance(document, dict):
        raise StageBCampaignError(f"{context} must contain an object")
    return {"path": relative, "sha256": expected}, document


def _validated_candidates(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest_schema = manifest.get("schema")
    if manifest_schema not in {
        CANDIDATE_MANIFEST_SCHEMA,
        SUCCESSOR_CANDIDATE_MANIFEST_SCHEMA,
    }:
        raise StageBCampaignError("candidate manifest has the wrong schema")
    body = {key: value for key, value in manifest.items() if key != "sha256"}
    if manifest.get("sha256") != _digest(body):
        raise StageBCampaignError("candidate manifest self digest is invalid")
    design = manifest.get("design")
    rows = manifest.get("candidates")
    if (
        manifest.get("status")
        not in {
            "preregistered-seed-free-candidate-generation",
            "preregistered-seed-free-successor-candidate-generation",
        }
        or not isinstance(design, Mapping)
        or design.get("scientific_seed") is not None
        or design.get("exact_count") != EXACT_SCREEN_COUNT
        or not isinstance(rows, list)
        or len(rows) != EXACT_SCREEN_COUNT
    ):
        raise StageBCampaignError("candidate manifest is not the exact seed-free screen")
    checked = []
    identifiers: set[str] = set()
    digests: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "point_index", "candidate_sha256", "candidate"
        }:
            raise StageBCampaignError("candidate row has an invalid shape")
        candidate = row["candidate"]
        if (
            row["point_index"]
            != (index + (EXACT_SCREEN_COUNT if manifest_schema == SUCCESSOR_CANDIDATE_MANIFEST_SCHEMA else 0))
            or not isinstance(candidate, Mapping)
            or set(candidate) != {"schema", "candidate_id", "parameters"}
            or candidate.get("schema") != "sim-adaptive-candidate-v1"
            or row["candidate_sha256"] != _digest(candidate)
        ):
            raise StageBCampaignError(f"candidate row {index} is not digest-bound")
        identifier = candidate.get("candidate_id")
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier != identifier.strip()
            or "/" in identifier
            or "\\" in identifier
        ):
            raise StageBCampaignError(f"candidate row {index} has an invalid identifier")
        identifiers.add(identifier)
        digests.add(row["candidate_sha256"])
        checked.append(dict(row))
    if len(identifiers) != EXACT_SCREEN_COUNT or len(digests) != EXACT_SCREEN_COUNT:
        raise StageBCampaignError("candidate manifest contains duplicate identities")
    return checked


def _write_once(path: Path, value: Mapping[str, Any]) -> str:
    raw = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return _digest_bytes(raw)


def _release_reference(root: Path, directory: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    def reference(name: str) -> dict[str, str]:
        path = directory / name
        return {
            "path": path.relative_to(root).as_posix(),
            "sha256": _digest_bytes(path.read_bytes()),
        }

    return {
        "candidate_id": row["candidate"]["candidate_id"],
        "candidate_sha256": row["candidate_sha256"],
        "release": reference("candidate-release.json"),
        "packet": reference("packet.sealed.json"),
        "policy": reference("authority-policy.json"),
    }


def materialize_campaign(
    candidate_manifest_path: str | Path,
    candidate_manifest_sha256: str,
    analysis_protocol_path: str | Path,
    analysis_protocol_sha256: str,
    output_dir: str | Path,
    *,
    repository_root: str | Path = ROOT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    workers: int | None = None,
) -> dict[str, Any]:
    """Compile, verify, and batch every preregistered candidate exactly once."""

    root = Path(repository_root).expanduser().resolve(strict=True)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise StageBCampaignError("batch_size must be a positive integer")
    manifest_ref, manifest = _load_bound_json(
        root, candidate_manifest_path, candidate_manifest_sha256, "candidate manifest"
    )
    candidates = _validated_candidates(manifest)
    protocol_ref, protocol = _load_bound_json(
        root, analysis_protocol_path, analysis_protocol_sha256, "analysis protocol"
    )
    phased_protocol = protocol.get("schema") == PHASED_ANALYSIS_PROTOCOL_SCHEMA
    template = manifest.get("template")
    if not isinstance(template, Mapping) or set(template) != {"path", "sha256", "template_id"}:
        raise StageBCampaignError("candidate manifest has an invalid template binding")
    template_ref, _ = _load_bound_json(
        root, template["path"], template["sha256"], "packet template"
    )

    destination = Path(output_dir).expanduser().resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise StageBCampaignError("campaign output must be inside repository_root") from exc
    if destination.exists() or destination.is_symlink():
        raise StageBCampaignError("campaign output directory must not already exist")
    destination.mkdir(parents=True)
    staging = destination / ".candidate-inputs"
    staging.mkdir()

    def materialize(row: Mapping[str, Any]) -> dict[str, Any]:
        identifier = row["candidate"]["candidate_id"]
        candidate_path = staging / f"{identifier}.json"
        _write_once(candidate_path, row["candidate"])
        release_dir = destination / "releases" / identifier
        compile_candidate(
            root / template_ref["path"],
            template_ref["sha256"],
            candidate_path,
            row["candidate_sha256"],
            release_dir,
            repository_root=root,
        )
        verify_candidate(
            root / template_ref["path"],
            template_ref["sha256"],
            release_dir,
            repository_root=root,
        )
        return _release_reference(root, release_dir, row)

    try:
        worker_count = workers if workers is not None else min(32, (os.cpu_count() or 1))
        if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count < 1:
            raise StageBCampaignError("workers must be a positive integer")
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            releases = list(executor.map(materialize, candidates))
        shutil.rmtree(staging)

        declarations = []
        for arm in sorted(READINESS_ARMS):
            for start in range(0, len(releases), batch_size):
                rows = releases[start : start + batch_size]
                body = {
                    "schema": DECLARATION_SCHEMA,
                    "arm": arm,
                    "analysis_protocol": protocol_ref,
                    "candidates": rows,
                }
                declaration = {**body, "sha256": _digest(body)}
                batch_index = start // batch_size
                path = destination / "declarations" / arm / f"batch-{batch_index:03d}.json"
                file_sha = _write_once(path, declaration)
                declarations.append({
                    "arm": arm,
                    "batch_index": batch_index,
                    "candidate_count": len(rows),
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_sha,
                    "declaration_sha256": declaration["sha256"],
                })

        body = {
            "schema": PHASED_CAMPAIGN_SCHEMA if phased_protocol else CAMPAIGN_SCHEMA,
            "status": "materialized-not-executed",
            "engineering_screening_only": True,
            "scientific_verdict": None,
            "numpy_confirmation_required": True,
            "candidate_manifest": manifest_ref,
            "analysis_protocol": protocol_ref,
            "packet_template": template_ref,
            "candidate_count": len(releases),
            "arm_count": len(READINESS_ARMS),
            "batch_size": batch_size,
            "batch_count": len(declarations),
            "declarations": declarations,
        }
        campaign = {**body, "sha256": _digest(body)}
        _write_once(destination / "campaign.json", campaign)
        return campaign
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def run_gpu_batch(
    campaign_path: str | Path,
    campaign_sha256: str,
    arm: str,
    batch_index: int,
    output_dir: str | Path,
    *,
    repository_root: str | Path = ROOT,
    chunk_steps: int = 4096,
) -> dict[str, Any]:
    """Run one exact campaign batch and publish one compact trace per candidate."""

    root = Path(repository_root).expanduser().resolve(strict=True)
    campaign_ref, campaign = _load_bound_json(
        root, campaign_path, campaign_sha256, "campaign manifest"
    )
    body = {key: value for key, value in campaign.items() if key != "sha256"}
    if (
        campaign.get("schema") not in {CAMPAIGN_SCHEMA, PHASED_CAMPAIGN_SCHEMA}
        or campaign.get("sha256") != _digest(body)
        or campaign.get("status") != "materialized-not-executed"
        or campaign.get("scientific_verdict") is not None
        or campaign.get("engineering_screening_only") is not True
    ):
        raise StageBCampaignError("campaign manifest is invalid or not executable screening state")
    if arm not in READINESS_ARMS:
        raise StageBCampaignError("arm is not an authorized readiness arm")
    if isinstance(batch_index, bool) or not isinstance(batch_index, int) or batch_index < 0:
        raise StageBCampaignError("batch_index must be a nonnegative integer")
    matches = [
        item for item in campaign.get("declarations", [])
        if isinstance(item, Mapping)
        and item.get("arm") == arm
        and item.get("batch_index") == batch_index
    ]
    if len(matches) != 1:
        raise StageBCampaignError("campaign does not contain exactly one requested batch")
    declaration = matches[0]
    output = Path(output_dir).expanduser().resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise StageBCampaignError("GPU batch output must be inside repository_root") from exc
    if output.exists() or output.is_symlink():
        raise StageBCampaignError("GPU batch output directory must not already exist")
    output.mkdir(parents=True)
    try:
        result = run_authenticated_gpu_batch(
            root / declaration["path"],
            declaration["sha256"],
            repository_root=root,
            chunk_steps=chunk_steps,
        )
        if (
            result.get("engineering_screening_only") is not True
            or result.get("scientific_verdict") is not None
            or result.get("numpy_confirmation_required") is not True
        ):
            raise StageBCampaignError("GPU runner changed its engineering-only boundary")
        phased_result = result.get("schema") == PHASED_OUTPUT_SCHEMA
        traces = []
        for candidate in result["candidates"]:
            trace = candidate["trace"]
            sample_count = int(trace["sample_count"])
            spikes = np.unpackbits(
                trace["spike_states_packed"], bitorder=trace["spike_bitorder"]
            )[:sample_count].astype(np.bool_, copy=False)
            dt = float(trace["sample_interval_s"])
            times = np.arange(1, sample_count + 1, dtype=np.float64) * dt
            voltage = np.asarray(trace["voltage_mV"], dtype=np.float64)
            archive = output / f"{candidate['candidate_id']}.trace.zip"
            archive_sha = save_compact_trace(archive, times, voltage, spikes)
            trace_receipt = {
                "candidate_id": candidate["candidate_id"],
                "candidate_sha256": candidate["candidate_sha256"],
                "termination": candidate["termination"],
                "compact_trace": {
                    "path": archive.relative_to(root).as_posix(),
                    "sha256": archive_sha,
                    "sample_count": sample_count,
                },
            }
            if phased_result:
                trace_receipt["runtime_intervention"] = candidate["runtime_intervention"]
            traces.append(trace_receipt)
        if phased_result != (campaign.get("schema") == PHASED_CAMPAIGN_SCHEMA):
            raise StageBCampaignError("GPU result schema does not match campaign protocol revision")
        receipt_body = {
            "schema": (
                PHASED_GPU_BATCH_RECEIPT_SCHEMA
                if phased_result
                else GPU_BATCH_RECEIPT_SCHEMA
            ),
            "process_status": "completed",
            "engineering_screening_only": True,
            "scientific_verdict": None,
            "numpy_confirmation_required": True,
            "campaign": campaign_ref,
            "declaration": {
                "path": declaration["path"],
                "sha256": declaration["sha256"],
                "declaration_sha256": declaration["declaration_sha256"],
            },
            "arm": arm,
            "batch_index": batch_index,
            "execution": result["execution"],
            "provenance": result["provenance"],
            "traces": traces,
        }
        receipt = {**receipt_body, "sha256": _digest(receipt_body)}
        _write_once(output / "receipt.json", receipt)
        return receipt
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-gpu-batch", action="store_true")
    parser.add_argument("--campaign")
    parser.add_argument("--campaign-sha256")
    parser.add_argument("--arm")
    parser.add_argument("--batch-index", type=int)
    parser.add_argument("--chunk-steps", type=int, default=4096)
    parser.add_argument("--candidate-manifest")
    parser.add_argument("--candidate-manifest-sha256")
    parser.add_argument("--analysis-protocol")
    parser.add_argument("--analysis-protocol-sha256")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(argv)
    try:
        if args.run_gpu_batch:
            if not all((args.campaign, args.campaign_sha256, args.arm, args.output_dir)) or args.batch_index is None:
                raise StageBCampaignError("GPU batch run requires campaign, digest, arm, batch index, and output")
            result = run_gpu_batch(
                args.campaign, args.campaign_sha256, args.arm, args.batch_index,
                args.output_dir, repository_root=args.repository_root,
                chunk_steps=args.chunk_steps,
            )
        else:
            required = (
                args.candidate_manifest, args.candidate_manifest_sha256,
                args.analysis_protocol, args.analysis_protocol_sha256, args.output_dir,
            )
            if not all(required):
                raise StageBCampaignError("materialization requires candidate manifest, protocol, digests, and output")
            result = materialize_campaign(
                args.candidate_manifest,
                args.candidate_manifest_sha256,
                args.analysis_protocol,
                args.analysis_protocol_sha256,
                args.output_dir,
                repository_root=args.repository_root,
                batch_size=args.batch_size,
                workers=args.workers,
            )
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"Stage B campaign materialization failure: {exc}\n")
    print(canonical_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
