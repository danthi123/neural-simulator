#!/usr/bin/env python3
"""Validate a fail-closed, read-only experiment lifecycle manifest.

This module deliberately does not execute commands.  It answers the narrower
question that the existing point tools cannot: does one experiment have a
continuous, hash-bound record from scope through roadmap update, with every
compute lane explicitly considered?
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from typing import Any, Mapping, Sequence


SCHEMA = "sim-experiment-automation-lifecycle-v1"
STAGES = (
    "scope",
    "research",
    "preregister",
    "execute",
    "validate",
    "compare",
    "archive",
    "roadmap",
)
REQUIRED_ROLES = {
    "scope": "scope_definition",
    "research": "research_record",
    "preregister": "preregistration",
    "execute": "execution_receipt",
    "validate": "validation_report",
    "compare": "comparison_report",
    "archive": "archive_manifest",
    "roadmap": "roadmap_update",
}
STATUSES = frozenset(("pending", "ready", "complete", "blocked"))
COMPUTE_TARGETS = ("local_cpu", "local_gpu", "mini_pc_cluster")
COMPUTE_DISPOSITIONS = frozenset(("planned", "complete", "not_applicable"))
_SHA256 = frozenset("0123456789abcdef")


class LifecycleError(ValueError):
    """Raised when a lifecycle manifest cannot earn a readiness result."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LifecycleError(message)


def _required_text(value: Any, field: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()),
             f"{field} must be a non-empty string")
    return value.strip()


def _safe_file(root: Path, value: Any, field: str) -> tuple[str, Path]:
    text = _required_text(value, field)
    relative = PurePosixPath(text)
    _require(
        not relative.is_absolute()
        and bool(relative.name)
        and "." not in relative.parts
        and ".." not in relative.parts,
        f"{field} must be a safe repository-relative path",
    )
    candidate = root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
        info = candidate.lstat()
    except (OSError, ValueError) as exc:
        raise LifecycleError(f"{field} is missing or escapes the repository: {text}") from exc
    _require(not stat.S_ISLNK(info.st_mode), f"{field} cannot be a symlink: {text}")
    _require(stat.S_ISREG(info.st_mode), f"{field} must be a regular file: {text}")
    return relative.as_posix(), resolved


def _hash_regular_file(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise LifecycleError(f"cannot open evidence file {path}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "rb") as handle:
            before = os.fstat(handle.fileno())
            _require(stat.S_ISREG(before.st_mode),
                     f"evidence is not a regular file: {path}")
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise LifecycleError(f"cannot hash evidence file {path}: {exc}") from exc
    state = lambda item: (  # noqa: E731 - compact immutable stat projection
        item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns
    )
    _require(state(before) == state(after), f"evidence changed while hashing: {path}")
    try:
        named = path.lstat()
    except OSError as exc:
        raise LifecycleError(f"evidence disappeared after hashing: {path}") from exc
    _require(not stat.S_ISLNK(named.st_mode) and state(named) == state(after),
             f"evidence changed while hashing: {path}")
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LifecycleError(f"cannot read lifecycle manifest {path}: {exc}") from exc
    _require(isinstance(value, dict), "lifecycle manifest must be a JSON object")
    return value


def _validate_evidence(
    *, root: Path, stage: str, records: Any, seen: set[str]
) -> list[dict[str, str]]:
    _require(isinstance(records, list), f"stage {stage} evidence must be a list")
    validated: list[dict[str, str]] = []
    for index, record in enumerate(records):
        label = f"stage {stage} evidence[{index}]"
        _require(isinstance(record, dict), f"{label} must be an object")
        _require(set(record) == {"path", "role", "sha256"},
                 f"{label} must contain exactly path, role, and sha256")
        relative, path = _safe_file(root, record.get("path"), f"{label} path")
        _require(relative not in seen,
                 f"evidence file is assigned to more than one stage: {relative}")
        seen.add(relative)
        role = _required_text(record.get("role"), f"{label} role")
        expected = _required_text(record.get("sha256"), f"{label} sha256")
        _require(len(expected) == 64 and set(expected) <= _SHA256,
                 f"{label} sha256 must be 64 lowercase hexadecimal characters")
        actual = _hash_regular_file(path)
        _require(actual == expected,
                 f"evidence digest mismatch for {relative}: expected {expected}, got {actual}")
        validated.append({"path": relative, "role": role, "sha256": actual})
    return validated


def _validate_compute_targets(value: Any, execute_status: str) -> dict[str, dict[str, str]]:
    _require(isinstance(value, dict), "compute_targets must be an object")
    _require(set(value) == set(COMPUTE_TARGETS),
             "compute_targets must name exactly local_cpu, local_gpu, and mini_pc_cluster")
    result: dict[str, dict[str, str]] = {}
    for target in COMPUTE_TARGETS:
        record = value[target]
        _require(isinstance(record, dict), f"compute target {target} must be an object")
        _require(set(record) == {"disposition", "reason"},
                 f"compute target {target} must contain exactly disposition and reason")
        disposition = record.get("disposition")
        _require(disposition in COMPUTE_DISPOSITIONS,
                 f"compute target {target} has invalid disposition {disposition!r}")
        reason = _required_text(record.get("reason"), f"compute target {target} reason")
        result[target] = {"disposition": disposition, "reason": reason}

    dispositions = {record["disposition"] for record in result.values()}
    _require("complete" in dispositions or "planned" in dispositions,
             "at least one compute target must be planned or complete")
    if execute_status == "complete":
        _require("planned" not in dispositions,
                 "execute is complete while a compute target is still planned")
        _require("complete" in dispositions,
                 "execute is complete without a completed compute target")
    return result


def validate_manifest(manifest: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    """Validate *manifest* against *root* and return a deterministic status report."""
    root = root.resolve(strict=True)
    _require(root.is_dir(), "repository root must be a directory")
    _require(isinstance(manifest, Mapping), "lifecycle manifest must be an object")
    _require(set(manifest) == {"schema", "experiment_id", "stages", "compute_targets"},
             "manifest must contain exactly schema, experiment_id, stages, and compute_targets")
    _require(manifest.get("schema") == SCHEMA, f"unsupported schema {manifest.get('schema')!r}")
    experiment_id = _required_text(manifest.get("experiment_id"), "experiment_id")

    stages = manifest.get("stages")
    _require(isinstance(stages, dict), "stages must be an object")
    _require(set(stages) == set(STAGES),
             f"stages must name exactly: {', '.join(STAGES)}")

    seen: set[str] = set()
    stage_results: dict[str, dict[str, Any]] = {}
    statuses: list[str] = []
    frontier_seen = False
    for stage in STAGES:
        record = stages[stage]
        _require(isinstance(record, dict), f"stage {stage} must be an object")
        _require(set(record) == {"status", "evidence", "blockers"},
                 f"stage {stage} must contain exactly status, evidence, and blockers")
        status = record.get("status")
        _require(status in STATUSES, f"stage {stage} has invalid status {status!r}")
        blockers = record.get("blockers")
        _require(isinstance(blockers, list)
                 and all(isinstance(item, str) and item.strip() for item in blockers),
                 f"stage {stage} blockers must be a list of non-empty strings")
        _require(len(blockers) == len(set(blockers)), f"stage {stage} has duplicate blockers")
        if status == "blocked":
            _require(bool(blockers), f"blocked stage {stage} must name at least one blocker")
        else:
            _require(not blockers, f"non-blocked stage {stage} cannot name blockers")

        evidence = _validate_evidence(root=root, stage=stage,
                                      records=record.get("evidence"), seen=seen)
        if status == "complete":
            _require(not frontier_seen,
                     f"stage {stage} is complete after the lifecycle frontier")
            roles = {item["role"] for item in evidence}
            _require(REQUIRED_ROLES[stage] in roles,
                     f"complete stage {stage} lacks required {REQUIRED_ROLES[stage]!r} evidence")
        else:
            if status in ("ready", "blocked"):
                _require(not frontier_seen, "only one stage may be ready or blocked")
                frontier_seen = True
            elif status == "pending":
                frontier_seen = True
                _require(not evidence, f"pending stage {stage} cannot claim evidence")
        statuses.append(status)
        stage_results[stage] = {
            "status": status,
            "evidence": evidence,
            "blockers": list(blockers),
        }

    active = [name for name, status in zip(STAGES, statuses)
              if status in ("ready", "blocked")]
    _require(len(active) <= 1, "only one stage may be ready or blocked")
    first_pending = next((index for index, status in enumerate(statuses)
                          if status == "pending"), len(statuses))
    _require(not any(status in ("ready", "blocked", "complete")
                     for status in statuses[first_pending + 1:]),
             "no stage after a pending stage may be active or complete")

    compute = _validate_compute_targets(manifest.get("compute_targets"),
                                        stage_results["execute"]["status"])
    if all(status == "complete" for status in statuses):
        state = "complete"
        frontier = None
    elif active:
        frontier = active[0]
        state = stage_results[frontier]["status"]
    else:
        frontier = STAGES[first_pending]
        state = "pending"
    return {
        "schema": SCHEMA,
        "experiment_id": experiment_id,
        "valid": True,
        "state": state,
        "frontier": frontier,
        "stages": stage_results,
        "compute_targets": compute,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a read-only experiment lifecycle manifest."
    )
    parser.add_argument("manifest", type=Path, help="JSON lifecycle manifest")
    parser.add_argument("--root", type=Path, default=Path.cwd(),
                        help="repository root used to resolve evidence paths")
    parser.add_argument("--json", action="store_true", help="print the full JSON report")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = validate_manifest(_load_manifest(args.manifest), root=args.root)
    except LifecycleError as exc:
        print(f"NOT READY: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        frontier = result["frontier"] or "none"
        print(f"READY: {result['experiment_id']} state={result['state']} frontier={frontier}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
