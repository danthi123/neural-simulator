#!/usr/bin/env python3
"""Dry-run orchestration for adaptive experiment proposals.

This module is intentionally a planning layer.  It validates an adaptive design,
asks ``adaptive_experiment`` for its deterministic proposal, and describes the
later harness steps without sealing, expanding, dispatching, or executing them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from tools.adaptive_experiment import (
    AdaptiveExperimentError,
    load_adaptive_design,
    propose_next_batch,
)
from tools.experiment import HarnessError, load_experiment_spec


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "sim-experiment-controller-dry-run-v1"
LANE_NAMES = {"local": "local_cpu", "gpu": "local_gpu", "pool": "mini_pc_cluster"}
REQUIRED_TARGET_FIELDS = ("device", "lane")


class ControllerError(ValueError):
    """Raised when a dry-run cannot be made trustworthy or owner-safe."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _reject_symlink_parents(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if os.path.lexists(current) and current.is_symlink():
            raise ControllerError(f"output path cannot contain a symlink: {current}")


def _execution_targets(spec: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    execution = spec.get("execution")
    if not isinstance(execution, dict):
        raise ControllerError("experiment spec requires an execution contract for lane planning")
    targets = execution.get("targets")
    if not isinstance(targets, dict):
        raise ControllerError("experiment spec execution requires targets for every backend")

    result: dict[str, dict[str, str]] = {}
    for backend in spec["backends"]:
        target = targets.get(backend)
        if not isinstance(target, dict):
            raise ControllerError(f"backend {backend!r} has no execution target")
        missing = [field for field in REQUIRED_TARGET_FIELDS if not isinstance(target.get(field), str)
                   or not target[field].strip()]
        if missing:
            raise ControllerError(f"backend {backend!r} target is missing {missing}")
        lane = target["lane"].strip()
        if lane not in LANE_NAMES:
            raise ControllerError(f"backend {backend!r} has unsupported lane {lane!r}")
        result[backend] = {"device": target["device"].strip(), "lane": LANE_NAMES[lane]}
    return result


def build_dry_run_plan(design_path: str | Path, *, root: str | Path = ROOT) -> dict[str, Any]:
    """Build a deterministic, non-executing plan from a validated adaptive design."""
    root = Path(root).resolve()
    design_path = Path(design_path).expanduser().resolve()
    if not _inside(design_path, root) or not design_path.is_file() or design_path.is_symlink():
        raise ControllerError("adaptive design must be a regular file inside the repository")

    try:
        design = load_adaptive_design(design_path, root=root)
        batch = propose_next_batch(design, root=root)
        spec_path = (root / design["experiment"]["spec_path"]).resolve()
        spec = load_experiment_spec(spec_path)
    except (AdaptiveExperimentError, HarnessError, OSError, KeyError, TypeError) as exc:
        raise ControllerError(f"cannot build dry-run plan: {exc}") from exc

    targets = _execution_targets(spec)
    candidates = []
    for candidate in batch["candidates"]:
        partition = candidate["partition"]
        if "heldout" in "".join(character for character in partition.lower() if character.isalnum()):
            raise ControllerError("proposal unexpectedly contains a held-out partition")
        backend = candidate["backend"]
        target = targets.get(backend)
        if target is None:
            raise ControllerError(f"proposal backend {backend!r} has no target")
        candidates.append({
            "candidate_id": candidate["candidate_id"],
            "order": candidate["order"],
            "parameters": candidate["parameters"],
            "fidelity": candidate["fidelity"],
            "fidelity_kind": candidate["fidelity_kind"],
            "backend": backend,
            "partition": partition,
            "resource_lane": target["lane"],
            "device": target["device"],
            "reason": candidate["reason"],
        })

    candidates.sort(key=lambda item: item["order"])
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "design": {
            "path": design_path.relative_to(root).as_posix(),
            "id": design["id"],
            "sha256": batch["design_sha256"],
        },
        "decision": batch["decision"],
        "reasons": batch["reasons"],
        "candidate_materialization": {
            "count": len(candidates),
            "candidates": candidates,
            "seeds_selected": False,
            "held_out_partitions_accessed": [],
        },
        "harness_sequence": [
            {"step": "materialize_candidates", "status": "described_only"},
            {"step": "preregister_candidate_arms", "authority": "tools/experiment.py", "status": "required"},
            {"step": "create_experiment_seal", "authority": "tools/experiment.py", "status": "required"},
            {"step": "expand_experiment_jobs", "authority": "tools/experiment.py", "status": "required"},
            {"step": "validate_receipts_controls_and_provenance", "status": "required"},
        ],
        "resource_lanes": {backend: targets[backend] for backend in sorted(targets)},
        "dispatch": {"performed": False, "reason": "dry-run controller never dispatches or executes jobs"},
        "validation_required": [
            "candidate parameters are read back by the runner",
            "calibration and replication seeds remain disjoint",
            "controls and lesions accompany any treatment arms",
            "source/config digests match the sealed execution contract",
            "malformed or incomplete receipts produce no scientific result",
            "held-out partitions remain untouched until an explicit later gate",
        ],
        "experiment_handoff": batch["experiment_handoff"],
    }
    body["sha256"] = _digest(body)
    return body


def write_dry_run_plan(plan: Mapping[str, Any], destination: str | Path, *, owner_root: str | Path) -> Path:
    """Write once, only beneath the caller's explicitly owned output root."""
    if plan.get("schema") != SCHEMA or plan.get("dispatch", {}).get("performed") is not False:
        raise ControllerError("refusing to write an invalid or executable controller plan")
    body = {key: value for key, value in plan.items() if key != "sha256"}
    if plan.get("sha256") != _digest(body):
        raise ControllerError("controller plan digest is invalid")
    owner_root = Path(owner_root).expanduser().resolve()
    destination = Path(destination).expanduser().absolute()
    if not _inside(destination, owner_root):
        raise ControllerError("dry-run output is outside the explicitly owned output root")
    _reject_symlink_parents(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(plan, handle, sort_keys=True, indent=2, ensure_ascii=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise ControllerError(f"refusing to replace existing dry-run output: {destination}") from exc
    destination.chmod(0o444)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a non-executing adaptive experiment controller plan.")
    parser.add_argument("design")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--output")
    parser.add_argument("--owner-root")
    args = parser.parse_args(argv)
    try:
        plan = build_dry_run_plan(args.design, root=args.root)
        if args.output:
            if not args.owner_root:
                raise ControllerError("--owner-root is required when --output is supplied")
            write_dry_run_plan(plan, args.output, owner_root=args.owner_root)
        print(json.dumps(plan, sort_keys=True, indent=2))
        return 0
    except ControllerError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
