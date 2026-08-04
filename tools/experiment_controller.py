#!/usr/bin/env python3
"""Dry-run orchestration for adaptive experiment proposals.

This module is intentionally a planning layer.  It validates an adaptive design,
asks ``adaptive_experiment`` for its deterministic proposal, and describes the
later harness steps without sealing, dispatching, or executing them.  An optional
handoff check validates an already-sealed, non-held-out expansion through the
existing experiment API without creating or running jobs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from tools.adaptive_experiment import (
    AdaptiveExperimentError,
    load_adaptive_design,
    propose_next_batch,
)
from tools.experiment import HarnessError, expand_experiment_jobs, load_experiment_spec


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "sim-experiment-controller-dry-run-v1"
HANDOFF_SCHEMA = "sim-experiment-controller-handoff-v1"
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


def validate_experiment_handoff(
    plan: Mapping[str, Any],
    *,
    seal_path: str | Path | None = None,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Validate a dry-run plan against a pre-existing sealed, non-held-out job matrix."""
    if not isinstance(plan, Mapping) or plan.get("schema") != SCHEMA:
        raise ControllerError("controller plan has the wrong schema")
    try:
        body = {key: value for key, value in plan.items() if key != "sha256"}
        valid_digest = plan.get("sha256") == _digest(body)
    except (TypeError, ValueError) as exc:
        raise ControllerError(f"controller plan digest cannot be validated: {exc}") from exc
    if not valid_digest:
        raise ControllerError("controller plan digest is invalid")
    if plan.get("decision") != "propose":
        raise ControllerError("only a proposing plan can be handed off to the experiment harness")

    dispatch = plan.get("dispatch")
    if not isinstance(dispatch, Mapping) or dispatch.get("performed") is not False:
        raise ControllerError("experiment handoff must remain non-dispatching")
    expected_harness_sequence = [
        {"step": "materialize_candidates", "status": "described_only"},
        {"step": "preregister_candidate_arms", "authority": "tools/experiment.py", "status": "required"},
        {"step": "create_experiment_seal", "authority": "tools/experiment.py", "status": "required"},
        {"step": "expand_experiment_jobs", "authority": "tools/experiment.py", "status": "required"},
        {"step": "validate_receipts_controls_and_provenance", "status": "required"},
    ]
    if plan.get("harness_sequence") != expected_harness_sequence:
        raise ControllerError("controller plan has an invalid harness sequence")
    required_validation = {
        "candidate parameters are read back by the runner",
        "calibration and replication seeds remain disjoint",
        "controls and lesions accompany any treatment arms",
        "source/config digests match the sealed execution contract",
        "malformed or incomplete receipts produce no scientific result",
        "held-out partitions remain untouched until an explicit later gate",
    }
    if (not isinstance(plan.get("validation_required"), list)
            or not required_validation.issubset(plan["validation_required"])):
        raise ControllerError("controller plan omits a required validation gate")
    materialization = plan.get("candidate_materialization")
    if not isinstance(materialization, Mapping):
        raise ControllerError("controller plan is missing candidate materialization")
    if materialization.get("seeds_selected") is not False:
        raise ControllerError("controller handoff cannot select seeds")
    if materialization.get("held_out_partitions_accessed") != []:
        raise ControllerError("controller handoff has touched a held-out partition")
    candidates = materialization.get("candidates")
    if (not isinstance(candidates, list) or materialization.get("count") != len(candidates)
            or not candidates):
        raise ControllerError("controller plan must contain a non-empty candidate list")

    handoff = plan.get("experiment_handoff")
    required_sequence = [
        "materialize candidate parameters as preregistered experiment arms",
        "create_experiment_seal",
        "expand_experiment_jobs for the candidate's named non-held-out partition",
        "execute only the emitted digest-bound job contract",
    ]
    if (not isinstance(handoff, Mapping)
            or handoff.get("authority") != "tools/experiment.py"
            or handoff.get("required_sequence") != required_sequence
            or handoff.get("direct_runner_commands_emitted") is not False
            or handoff.get("seal_required") is not True
            or handoff.get("digest_bound_job_expansion_required") is not True
            or handoff.get("held_out_partitions_accessed") != []):
        raise ControllerError("experiment handoff does not preserve the sealed harness sequence")

    root = Path(root).resolve()
    spec_relative = handoff.get("spec_path")
    if not isinstance(spec_relative, str) or not spec_relative.strip():
        raise ControllerError("experiment handoff requires a repository-relative spec path")
    relative = PurePosixPath(spec_relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.name:
        raise ControllerError("experiment handoff spec path is not repository-relative")
    spec_path = root.joinpath(*relative.parts)
    if (not _inside(spec_path, root) or not spec_path.is_file() or spec_path.is_symlink()):
        raise ControllerError("experiment handoff spec must be a regular file inside the repository")
    try:
        spec = load_experiment_spec(spec_path)
        targets = _execution_targets(spec)
    except (HarnessError, OSError, KeyError, TypeError) as exc:
        raise ControllerError(f"cannot validate experiment handoff spec: {exc}") from exc
    if handoff.get("experiment_id") != spec["id"]:
        raise ControllerError("experiment handoff experiment id does not match the preregistered spec")
    if plan.get("resource_lanes") != {backend: targets[backend] for backend in sorted(targets)}:
        raise ControllerError("controller plan resource lanes do not match the execution contract")

    pairs = set()
    candidate_ids = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise ControllerError(f"candidate {index} is not an object")
        identifier = candidate.get("candidate_id")
        if (not isinstance(identifier, str) or not identifier.strip()
                or identifier in candidate_ids or candidate.get("order") != index):
            raise ControllerError(f"candidate {index} has an invalid or duplicate identity")
        candidate_ids.add(identifier)
        if any(key in candidate for key in ("seed", "command", "enqueue_command", "execution_contract")):
            raise ControllerError(f"candidate {identifier!r} contains execution or seed selection")
        parameters = candidate.get("parameters")
        if not isinstance(parameters, Mapping) or "seed" in parameters:
            raise ControllerError(f"candidate {identifier!r} contains an invalid parameter payload")
        backend = candidate.get("backend")
        partition = candidate.get("partition")
        if not isinstance(backend, str) or backend not in targets:
            raise ControllerError(f"candidate {identifier!r} uses an undeclared backend")
        if (not isinstance(partition, str) or partition not in spec["partitions"]
                or "".join(character for character in partition.lower() if character.isalnum()).startswith("heldout")):
            raise ControllerError(f"candidate {identifier!r} uses a held-out or undeclared partition")
        target = targets[backend]
        if (candidate.get("resource_lane") != target["lane"]
                or candidate.get("device") != target["device"]):
            raise ControllerError(f"candidate {identifier!r} does not match its sealed execution target")
        pairs.add((backend, partition))

    if seal_path is None:
        raise ControllerError("an existing experiment seal is required for handoff validation")
    try:
        seal = Path(seal_path).expanduser().absolute()
    except (TypeError, ValueError) as exc:
        raise ControllerError(f"experiment seal path is invalid: {exc}") from exc
    if not seal.is_file() or seal.is_symlink():
        raise ControllerError("experiment handoff requires an existing regular experiment seal")

    partitions = sorted({partition for _, partition in pairs})
    try:
        jobs = expand_experiment_jobs(spec_path, partitions, seal_path=seal, root=root)
    except (HarnessError, OSError, KeyError, TypeError, ValueError) as exc:
        raise ControllerError(f"sealed experiment expansion rejected the handoff: {exc}") from exc
    if not jobs or any(not isinstance(job, Mapping) or job.get("sealed") is not True for job in jobs):
        raise ControllerError("sealed experiment expansion did not produce sealed job contracts")
    if any("".join(character for character in str(job.get("partition", "")).lower() if character.isalnum()).startswith("heldout")
           for job in jobs):
        raise ControllerError("sealed experiment expansion unexpectedly included a held-out partition")
    expanded_pairs = {(job.get("backend"), job.get("partition")) for job in jobs}
    missing = sorted(pairs - expanded_pairs)
    if missing:
        raise ControllerError(f"sealed experiment expansion omitted candidate mappings: {missing}")
    extra = sorted(expanded_pairs - pairs)
    if extra:
        raise ControllerError(
            "sealed experiment expansion widened the dry-run candidate set with undeclared backend/partition "
            f"pairs: {extra}"
        )

    return {
        "schema": HANDOFF_SCHEMA,
        "plan_sha256": plan["sha256"],
        "experiment_id": spec["id"],
        "spec_path": relative.as_posix(),
        "sealed": True,
        "candidate_count": len(candidates),
        "backend_partition_pairs": [
            {"backend": backend, "partition": partition}
            for backend, partition in sorted(pairs)
        ],
        "expanded_job_count": len(jobs),
        "seeds_selected": False,
        "held_out_partitions_accessed": [],
    }


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
