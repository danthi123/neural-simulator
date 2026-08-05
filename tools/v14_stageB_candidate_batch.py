#!/usr/bin/env python3
"""Generate the exact deterministic V14 Stage B intrinsic-screen candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import scipy
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_SCHEMA = "v14-snr-stageB-packet-template-v1"
CANDIDATE_SCHEMA = "sim-adaptive-candidate-v1"
MANIFEST_SCHEMA = "v14-snr-stageB-sobol-candidate-manifest-v1"
SUCCESSOR_MANIFEST_SCHEMA = "v14-snr-stageB-sobol-candidate-manifest-v2"
EXACT_SCREEN_COUNT = 512
SOBOL_EXPONENT = 9


class StageBCandidateBatchError(ValueError):
    """Raised when the filed candidate screen cannot be reproduced exactly."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    ).encode("ascii")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StageBCandidateBatchError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _relative_file(root: Path, value: str | Path, context: str) -> tuple[str, Path]:
    path = Path(value).expanduser().resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise StageBCandidateBatchError(f"{context} must be inside the repository") from exc
    if path.is_symlink() or not path.is_file():
        raise StageBCandidateBatchError(f"{context} must be a regular file")
    return relative.as_posix(), path


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise StageBCandidateBatchError(f"{context} must be numeric")
    if isinstance(value, str) and (not value or value != value.strip()):
        raise StageBCandidateBatchError(f"{context} must be a canonical decimal string")
    try:
        decimal = Decimal(str(value))
    except InvalidOperation as exc:
        raise StageBCandidateBatchError(f"{context} must be numeric") from exc
    result = float(decimal)
    if not math.isfinite(result):
        raise StageBCandidateBatchError(f"{context} must be finite")
    return result


def _search_space(template: Mapping[str, Any]) -> list[dict[str, Any]]:
    if template.get("schema") != TEMPLATE_SCHEMA:
        raise StageBCandidateBatchError("packet template has the wrong schema")
    groups = template.get("parameter_leaves")
    if not isinstance(groups, Mapping) or not groups:
        raise StageBCandidateBatchError("packet template has no parameter leaves")
    result = []
    keys = set()
    for group_name in sorted(groups):
        leaves = groups[group_name]
        if not isinstance(leaves, Mapping):
            raise StageBCandidateBatchError(f"template group {group_name!r} is malformed")
        for leaf_name in sorted(leaves):
            leaf = leaves[leaf_name]
            if not isinstance(leaf, Mapping) or leaf.get("mode") != "searched":
                continue
            required = {
                "mode", "unit", "uncertainty", "evidence", "authority",
                "candidate_key", "bounds", "transform",
            }
            if set(leaf) != required:
                raise StageBCandidateBatchError(
                    f"searched template leaf {group_name}.{leaf_name} has an invalid shape"
                )
            key = leaf.get("candidate_key")
            if not isinstance(key, str) or not key or key in keys:
                raise StageBCandidateBatchError("searched candidate keys must be unique nonempty text")
            keys.add(key)
            bounds = leaf.get("bounds")
            if not isinstance(bounds, Mapping) or set(bounds) != {"low", "high"}:
                raise StageBCandidateBatchError(f"searched candidate key {key!r} has invalid bounds")
            low = _number(bounds["low"], f"{key}.low")
            high = _number(bounds["high"], f"{key}.high")
            transform = leaf.get("transform")
            if low >= high or transform not in {"linear", "log"} or (transform == "log" and low <= 0):
                raise StageBCandidateBatchError(f"searched candidate key {key!r} has invalid range")
            if leaf.get("evidence") != "derived" or leaf.get("authority") != "project_decision":
                raise StageBCandidateBatchError(
                    f"searched candidate key {key!r} changed its evidence boundary"
                )
            result.append({
                "candidate_key": key,
                "group": str(group_name),
                "leaf": str(leaf_name),
                "low": low,
                "high": high,
                "transform": transform,
                "unit": leaf.get("unit"),
            })
    if not result:
        raise StageBCandidateBatchError("packet template contains no searched parameters")
    return sorted(result, key=lambda item: item["candidate_key"])


def _map_point(unit_point: np.ndarray, space: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    parameters = {}
    for coordinate, specification in zip(unit_point, space, strict=True):
        low = float(specification["low"])
        high = float(specification["high"])
        if float(coordinate) == 0.0:
            value = low
        elif float(coordinate) == 1.0:
            value = high
        elif specification["transform"] == "log":
            value = math.exp(math.log(low) + float(coordinate) * (math.log(high) - math.log(low)))
        else:
            value = low + float(coordinate) * (high - low)
        parameters[str(specification["candidate_key"])] = value
    return parameters


def build_candidate_manifest(
    template_path: str | Path,
    template_sha256: str,
    *,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Build the exact 512-point seed-free Sobol screen without executing it."""

    root_path = Path(root).expanduser().resolve(strict=True)
    expected_digest = _sha256(template_sha256, "template_sha256")
    relative, path = _relative_file(root_path, template_path, "packet template")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        raise StageBCandidateBatchError("packet template digest does not match")
    try:
        template = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StageBCandidateBatchError("packet template is not JSON") from exc
    if not isinstance(template, Mapping) or raw != _canonical(template):
        raise StageBCandidateBatchError("packet template must be canonical JSON")
    space = _search_space(template)

    points = qmc.Sobol(d=len(space), scramble=False).random_base2(SOBOL_EXPONENT)
    if points.shape != (EXACT_SCREEN_COUNT, len(space)):
        raise StageBCandidateBatchError("Sobol implementation did not produce the filed exact budget")
    candidates = []
    for point_index, point in enumerate(points):
        parameters = _map_point(point, space)
        identity = {
            "algorithm": "scipy.stats.qmc.Sobol(scramble=False).random_base2(9)",
            "point_index": point_index,
            "template_sha256": expected_digest,
            "parameters": parameters,
        }
        candidate = {
            "schema": CANDIDATE_SCHEMA,
            "candidate_id": f"v14-stageB-sobol-{point_index:03d}-{_digest(identity)[:12]}",
            "parameters": parameters,
        }
        candidates.append({
            "point_index": point_index,
            "candidate_sha256": _digest(candidate),
            "candidate": candidate,
        })

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "preregistered-seed-free-candidate-generation",
        "device": "not_applicable_non_executed_candidate_design",
        "provenance_exempt": (
            "deterministic non-executed Sobol candidate design; contains no measured result"
        ),
        "template": {
            "path": relative,
            "sha256": expected_digest,
            "template_id": template.get("template_id"),
        },
        "design": {
            "method": "unscrambled_sobol_base2",
            "implementation": "scipy.stats.qmc.Sobol(scramble=False).random_base2(9)",
            "scientific_seed": None,
            "skip": 0,
            "dimension": len(space),
            "exact_count": EXACT_SCREEN_COUNT,
            "ordering": "Sobol point index ascending",
            "duplicate_policy": "duplicates are infrastructure failure",
            "library_versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        },
        "search_space": space,
        "candidates": candidates,
    }
    if len({_digest(item["candidate"]) for item in candidates}) != EXACT_SCREEN_COUNT:
        raise StageBCandidateBatchError("candidate generation produced duplicate parameter points")
    manifest["sha256"] = _digest(manifest)
    return manifest


def build_successor_candidate_manifest(
    template_path: str | Path,
    template_sha256: str,
    predecessor_manifest_path: str | Path,
    predecessor_manifest_sha256: str,
    *,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Build the fresh Sobol 512..1023 partition bound to the consumed V1 screen."""

    root_path = Path(root).expanduser().resolve(strict=True)
    expected_template = _sha256(template_sha256, "template_sha256")
    template_relative, template_file = _relative_file(
        root_path, template_path, "packet template"
    )
    template_raw = template_file.read_bytes()
    if hashlib.sha256(template_raw).hexdigest() != expected_template:
        raise StageBCandidateBatchError("packet template digest does not match")
    try:
        template = json.loads(template_raw)
    except json.JSONDecodeError as exc:
        raise StageBCandidateBatchError("packet template is not JSON") from exc
    if not isinstance(template, Mapping) or template_raw != _canonical(template):
        raise StageBCandidateBatchError("packet template must be canonical JSON")

    predecessor_expected = _sha256(
        predecessor_manifest_sha256, "predecessor_manifest_sha256"
    )
    predecessor_relative, predecessor_file = _relative_file(
        root_path, predecessor_manifest_path, "predecessor candidate manifest"
    )
    predecessor_raw = predecessor_file.read_bytes()
    if hashlib.sha256(predecessor_raw).hexdigest() != predecessor_expected:
        raise StageBCandidateBatchError("predecessor candidate manifest digest does not match")
    try:
        predecessor = json.loads(predecessor_raw)
    except json.JSONDecodeError as exc:
        raise StageBCandidateBatchError("predecessor candidate manifest is not JSON") from exc
    predecessor_body = {
        key: value for key, value in predecessor.items() if key != "sha256"
    } if isinstance(predecessor, Mapping) else {}
    predecessor_rows = predecessor.get("candidates") if isinstance(predecessor, Mapping) else None
    if (
        not isinstance(predecessor, Mapping)
        or predecessor.get("schema") != MANIFEST_SCHEMA
        or predecessor.get("sha256") != _digest(predecessor_body)
        or predecessor.get("template", {}).get("sha256") != expected_template
        or predecessor.get("design", {}).get("skip") != 0
        or predecessor.get("design", {}).get("exact_count") != EXACT_SCREEN_COUNT
        or not isinstance(predecessor_rows, list)
        or len(predecessor_rows) != EXACT_SCREEN_COUNT
        or [row.get("point_index") for row in predecessor_rows]
        != list(range(EXACT_SCREEN_COUNT))
    ):
        raise StageBCandidateBatchError(
            "predecessor is not the exact consumed V1 Sobol partition"
        )

    space = _search_space(template)
    all_points = qmc.Sobol(d=len(space), scramble=False).random_base2(
        SOBOL_EXPONENT + 1
    )
    points = all_points[EXACT_SCREEN_COUNT : 2 * EXACT_SCREEN_COUNT]
    if points.shape != (EXACT_SCREEN_COUNT, len(space)):
        raise StageBCandidateBatchError("Sobol implementation did not produce the successor partition")
    algorithm = (
        "scipy.stats.qmc.Sobol(scramble=False).random_base2(10)[512:1024]"
    )
    candidates = []
    for offset, point in enumerate(points):
        point_index = EXACT_SCREEN_COUNT + offset
        parameters = _map_point(point, space)
        identity = {
            "algorithm": algorithm,
            "point_index": point_index,
            "template_sha256": expected_template,
            "parameters": parameters,
        }
        candidate = {
            "schema": CANDIDATE_SCHEMA,
            "candidate_id": (
                f"v14-stageB-v3-sobol-{point_index:04d}-{_digest(identity)[:12]}"
            ),
            "parameters": parameters,
        }
        candidates.append(
            {
                "point_index": point_index,
                "candidate_sha256": _digest(candidate),
                "candidate": candidate,
            }
        )

    predecessor_digests = {
        row.get("candidate_sha256") for row in predecessor_rows if isinstance(row, Mapping)
    }
    successor_digests = {row["candidate_sha256"] for row in candidates}
    if (
        len(predecessor_digests) != EXACT_SCREEN_COUNT
        or len(successor_digests) != EXACT_SCREEN_COUNT
        or predecessor_digests & successor_digests
    ):
        raise StageBCandidateBatchError(
            "successor candidate generation overlaps the consumed partition"
        )

    manifest = {
        "schema": SUCCESSOR_MANIFEST_SCHEMA,
        "status": "preregistered-seed-free-successor-candidate-generation",
        "device": "not_applicable_non_executed_candidate_design",
        "provenance_exempt": (
            "deterministic non-executed successor Sobol design; contains no measured result"
        ),
        "predecessor": {
            "path": predecessor_relative,
            "sha256": predecessor_expected,
            "self_sha256": predecessor["sha256"],
            "consumed_point_start": 0,
            "consumed_point_stop_exclusive": EXACT_SCREEN_COUNT,
        },
        "template": {
            "path": template_relative,
            "sha256": expected_template,
            "template_id": template.get("template_id"),
        },
        "design": {
            "method": "unscrambled_sobol_base2_successor_partition",
            "implementation": algorithm,
            "scientific_seed": None,
            "skip": EXACT_SCREEN_COUNT,
            "block_index": 1,
            "global_point_start": EXACT_SCREEN_COUNT,
            "global_point_stop_exclusive": 2 * EXACT_SCREEN_COUNT,
            "dimension": len(space),
            "exact_count": EXACT_SCREEN_COUNT,
            "ordering": "global Sobol point index ascending",
            "duplicate_policy": "overlap with predecessor or within successor is infrastructure failure",
            "library_versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        },
        "search_space": space,
        "candidates": candidates,
    }
    manifest["sha256"] = _digest(manifest)
    return manifest


def write_manifest(manifest: Mapping[str, Any], destination: str | Path, *, root: str | Path = ROOT) -> Path:
    root_path = Path(root).expanduser().resolve(strict=True)
    if manifest.get("schema") not in {MANIFEST_SCHEMA, SUCCESSOR_MANIFEST_SCHEMA}:
        raise StageBCandidateBatchError("candidate manifest has the wrong schema")
    body = {key: value for key, value in manifest.items() if key != "sha256"}
    if manifest.get("sha256") != _digest(body):
        raise StageBCandidateBatchError("candidate manifest digest is invalid")
    path = Path(destination).expanduser().resolve()
    try:
        relative = path.relative_to(root_path)
    except ValueError as exc:
        raise StageBCandidateBatchError("candidate manifest output must remain inside the repository") from exc
    pure = PurePosixPath(relative.as_posix())
    if not pure.name or any(part in {"", ".", ".."} for part in pure.parts):
        raise StageBCandidateBatchError("candidate manifest output path is invalid")
    if path.exists():
        raise StageBCandidateBatchError("refusing to replace an existing candidate manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(_canonical(manifest) + b"\n")
    except FileExistsError as exc:
        raise StageBCandidateBatchError("refusing to replace an existing candidate manifest") from exc
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--template-sha256", required=True)
    parser.add_argument("--predecessor-manifest")
    parser.add_argument("--predecessor-manifest-sha256")
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args(argv)
    try:
        if bool(args.predecessor_manifest) != bool(args.predecessor_manifest_sha256):
            raise StageBCandidateBatchError(
                "predecessor manifest path and digest must be supplied together"
            )
        manifest = (
            build_successor_candidate_manifest(
                args.template,
                args.template_sha256,
                args.predecessor_manifest,
                args.predecessor_manifest_sha256,
                root=args.root,
            )
            if args.predecessor_manifest
            else build_candidate_manifest(args.template, args.template_sha256, root=args.root)
        )
        write_manifest(manifest, args.output, root=args.root)
    except (StageBCandidateBatchError, OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
