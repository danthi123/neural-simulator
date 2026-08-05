#!/usr/bin/env python3
"""Benchmark V3 Stage B GPU batch width using consumed candidates only."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import numpy as np

from research.runners.v14_stageB_batched_physiology import (
    DECLARATION_SCHEMA,
    PHASED_OUTPUT_SCHEMA,
    run_authenticated_gpu_batch,
)
from sim.snr_executable_packet import canonical_bytes
from tools.v14_stageB_campaign import CAMPAIGN_SCHEMA


ROOT = Path(__file__).resolve().parents[1]
SPEC_SCHEMA = "v14-snr-stageB-v3-batch-width-benchmark-v1"
RESULT_SCHEMA = "v14-snr-stageB-v3-batch-width-benchmark-result-v1"


class StageBBatchWidthBenchmarkError(ValueError):
    """Raised when benchmark identity, execution, or selection is invalid."""


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any) -> str:
    return _digest_bytes(canonical_bytes(value))


def _sha(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StageBBatchWidthBenchmarkError(f"{context} must be a SHA-256 digest")
    return value


def _repo_file(root: Path, value: Any, context: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise StageBBatchWidthBenchmarkError(f"{context} must be repository-relative")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise StageBBatchWidthBenchmarkError(f"{context} path is not canonical")
    path = root.joinpath(*relative.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise StageBBatchWidthBenchmarkError(f"{context} escapes repository root") from exc
    if path.is_symlink() or not path.is_file():
        raise StageBBatchWidthBenchmarkError(f"{context} must be a regular file")
    return relative.as_posix(), path


def _load_bound_json(
    root: Path, reference: Any, context: str
) -> tuple[dict[str, str], dict[str, Any]]:
    if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
        raise StageBBatchWidthBenchmarkError(f"{context} binding is malformed")
    relative, path = _repo_file(root, reference.get("path"), context)
    expected = _sha(reference.get("sha256"), f"{context} sha256")
    raw = path.read_bytes()
    if _digest_bytes(raw) != expected:
        raise StageBBatchWidthBenchmarkError(f"{context} digest does not match")
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StageBBatchWidthBenchmarkError(f"{context} is not JSON") from exc
    if not isinstance(document, dict):
        raise StageBBatchWidthBenchmarkError(f"{context} must contain an object")
    return {"path": relative, "sha256": expected}, document


def load_benchmark_spec(
    path: str | Path, expected_sha256: str, *, repository_root: str | Path = ROOT
) -> tuple[dict[str, Any], Path]:
    root = Path(repository_root).expanduser().resolve(strict=True)
    supplied = Path(path).expanduser()
    spec_path = (supplied if supplied.is_absolute() else root / supplied).resolve()
    try:
        spec_path.relative_to(root)
    except ValueError as exc:
        raise StageBBatchWidthBenchmarkError("benchmark spec escapes repository root") from exc
    raw = spec_path.read_bytes()
    if _digest_bytes(raw) != _sha(expected_sha256, "benchmark spec sha256"):
        raise StageBBatchWidthBenchmarkError("benchmark spec digest does not match")
    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StageBBatchWidthBenchmarkError("benchmark spec is not JSON") from exc
    body = {key: value for key, value in spec.items() if key != "sha256"}
    required = {
        "schema", "status", "device", "provenance_exempt", "consumed_campaign",
        "analysis_protocol", "arm", "warmup_width", "run_order", "selection", "sha256",
    }
    widths = spec.get("run_order") if isinstance(spec, Mapping) else None
    selection = spec.get("selection") if isinstance(spec, Mapping) else None
    if (
        not isinstance(spec, dict)
        or set(spec) != required
        or spec.get("schema") != SPEC_SCHEMA
        or spec.get("sha256") != _digest(body)
        or spec.get("status") != "preregistered-consumed-candidate-performance-only"
        or spec.get("arm") != "nap_lesion"
        or spec.get("warmup_width") != 64
        or widths != [64, 128, 256, 512, 512, 256, 128, 64]
        or not isinstance(selection, Mapping)
        or selection
        != {
            "metric": "median_candidate_steps_per_second",
            "near_tie_fraction": 0.05,
            "near_tie_rule": "choose_smallest_width_within_fraction_of_best",
        }
    ):
        raise StageBBatchWidthBenchmarkError("benchmark spec changed its filed design")
    _load_bound_json(root, spec["consumed_campaign"], "consumed campaign")
    _load_bound_json(root, spec["analysis_protocol"], "analysis protocol")
    return spec, root


def _consumed_candidates(
    root: Path, campaign_ref: Mapping[str, str]
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    binding, campaign = _load_bound_json(root, campaign_ref, "consumed campaign")
    body = {key: value for key, value in campaign.items() if key != "sha256"}
    if (
        campaign.get("schema") != CAMPAIGN_SCHEMA
        or campaign.get("sha256") != _digest(body)
        or campaign.get("candidate_count") != 512
    ):
        raise StageBBatchWidthBenchmarkError("benchmark campaign is not the consumed V1 screen")
    rows: list[dict[str, Any]] = []
    for declaration in sorted(
        (row for row in campaign.get("declarations", []) if row.get("arm") == "nap_lesion"),
        key=lambda row: row["batch_index"],
    ):
        _, document = _load_bound_json(
            root,
            {"path": declaration["path"], "sha256": declaration["sha256"]},
            "consumed NaP declaration",
        )
        rows.extend(document.get("candidates", []))
    identities = [row.get("candidate_id") for row in rows if isinstance(row, Mapping)]
    if len(rows) != 512 or len(set(identities)) != 512:
        raise StageBBatchWidthBenchmarkError("consumed campaign does not cover 512 unique candidates")
    return binding, rows


def _write_declaration(
    root: Path,
    directory: Path,
    width: int,
    candidates: list[dict[str, Any]],
    protocol: Mapping[str, str],
) -> tuple[Path, str]:
    body = {
        "schema": DECLARATION_SCHEMA,
        "arm": "nap_lesion",
        "analysis_protocol": dict(protocol),
        "candidates": candidates[:width],
    }
    document = {**body, "sha256": _digest(body)}
    path = directory / f"width-{width:03d}.json"
    raw = canonical_bytes(document)
    if path.exists():
        if path.read_bytes() != raw:
            raise StageBBatchWidthBenchmarkError("existing benchmark declaration changed identity")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    path.relative_to(root)
    return path, _digest_bytes(raw)


def _select_width(rows: list[Mapping[str, Any]]) -> tuple[int, dict[str, Any]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["width"])].append(float(row["candidate_steps_per_second"]))
    summary = {
        str(width): {
            "replicates": len(values),
            "median_candidate_steps_per_second": float(np.median(values)),
            "minimum_candidate_steps_per_second": float(min(values)),
            "maximum_candidate_steps_per_second": float(max(values)),
        }
        for width, values in sorted(grouped.items())
    }
    if set(grouped) != {64, 128, 256, 512} or any(len(values) != 2 for values in grouped.values()):
        raise StageBBatchWidthBenchmarkError("benchmark did not complete the filed paired width matrix")
    best = max(item["median_candidate_steps_per_second"] for item in summary.values())
    eligible = [
        int(width) for width, item in summary.items()
        if item["median_candidate_steps_per_second"] >= best * 0.95
    ]
    return min(eligible), summary


def run_benchmark(
    spec_path: str | Path,
    spec_sha256: str,
    output_dir: str | Path,
    *,
    repository_root: str | Path = ROOT,
    _runner: Callable[..., Mapping[str, Any]] = run_authenticated_gpu_batch,
    _clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    spec, root = load_benchmark_spec(
        spec_path, spec_sha256, repository_root=repository_root
    )
    destination = Path(output_dir).expanduser()
    destination = (destination if destination.is_absolute() else root / destination).resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise StageBBatchWidthBenchmarkError("benchmark output escapes repository root") from exc
    if destination.exists() and not destination.is_dir():
        raise StageBBatchWidthBenchmarkError("benchmark output is not a directory")
    destination.mkdir(parents=True, exist_ok=True)

    campaign_binding, candidates = _consumed_candidates(root, spec["consumed_campaign"])
    protocol_binding, _ = _load_bound_json(root, spec["analysis_protocol"], "analysis protocol")
    declarations = {
        width: _write_declaration(
            root, destination / "declarations", width, candidates, protocol_binding
        )
        for width in {64, 128, 256, 512}
    }

    warmup_path, warmup_sha = declarations[spec["warmup_width"]]
    warmup = _runner(
        warmup_path, warmup_sha, repository_root=root, chunk_steps=4096
    )
    if warmup.get("schema") != PHASED_OUTPUT_SCHEMA:
        raise StageBBatchWidthBenchmarkError("warmup did not execute the V3 runner")
    del warmup
    gc.collect()

    rows = []
    for order_index, width in enumerate(spec["run_order"]):
        declaration_path, declaration_sha = declarations[width]
        start = _clock()
        result = _runner(
            declaration_path,
            declaration_sha,
            repository_root=root,
            chunk_steps=4096,
        )
        elapsed = _clock() - start
        if (
            result.get("schema") != PHASED_OUTPUT_SCHEMA
            or result.get("scientific_verdict") is not None
            or result.get("execution", {}).get("candidate_count") != width
            or result.get("execution", {}).get("bridge_steps_executed") != 60_000
            or elapsed <= 0.0
        ):
            raise StageBBatchWidthBenchmarkError("benchmark runner result is invalid")
        rows.append(
            {
                "order_index": order_index,
                "width": width,
                "elapsed_s": elapsed,
                "candidate_steps_per_second": width * 60_000 / elapsed,
                "declaration": {
                    "path": declaration_path.relative_to(root).as_posix(),
                    "sha256": declaration_sha,
                },
            }
        )
        del result
        gc.collect()

    selected, summary = _select_width(rows)
    body = {
        "schema": RESULT_SCHEMA,
        "process_status": "completed",
        "engineering_performance_only": True,
        "scientific_verdict": None,
        "fresh_candidate_data_used": False,
        "spec": {
            "path": Path(spec_path).as_posix(),
            "sha256": spec_sha256,
            "self_sha256": spec["sha256"],
        },
        "consumed_campaign": campaign_binding,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "runs": rows,
        "summary": summary,
        "selected_batch_width": selected,
        "selection_rule": dict(spec["selection"]),
    }
    result = {**body, "sha256": _digest(body)}
    output = destination / "result.json"
    if output.exists():
        raise StageBBatchWidthBenchmarkError("refusing to replace benchmark result")
    output.write_bytes(canonical_bytes(result))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repository-root", default=str(ROOT))
    args = parser.parse_args(argv)
    try:
        result = run_benchmark(
            args.spec,
            args.spec_sha256,
            args.output_dir,
            repository_root=args.repository_root,
        )
    except (OSError, TypeError, ValueError) as exc:
        parser.exit(2, f"Stage B batch-width benchmark failure: {exc}\n")
    print(canonical_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
