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
import subprocess
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
EXECUTION_MANIFEST_SCHEMA = "sim-experiment-controller-execution-manifest-v1"
EXECUTION_RECEIPT_SCHEMA = "sim-experiment-controller-execution-receipt-v1"
LANE_NAMES = {"local": "local_cpu", "gpu": "local_gpu", "pool": "mini_pc_cluster"}
REQUIRED_TARGET_FIELDS = ("device", "lane")
ARM_ROLES = ("treatment", "control", "lesion")


class ControllerError(ValueError):
    """Raised when a dry-run cannot be made trustworthy or owner-safe."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _without_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "sha256"}


def _valid_digest(value: Mapping[str, Any]) -> bool:
    try:
        return value.get("sha256") == _digest(_without_digest(value))
    except (TypeError, ValueError):
        return False


def _contains_forbidden_data(value: Any) -> bool:
    """Reject seed and held-out payloads at every depth of a controller contract."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = "".join(character for character in str(key).lower() if character.isalnum())
            if normalized.endswith(("seed", "seeds")) or normalized.startswith("heldout"):
                return True
            if _contains_forbidden_data(item):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_data(item) for item in value)
    return False


def _arm_contract(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize explicit treatment/control/lesion definitions without expanding jobs."""
    execution = spec.get("execution")
    raw = execution.get("arms") if isinstance(execution, Mapping) else None
    if not isinstance(raw, Mapping) or not raw:
        raise ControllerError("arm materialization requires explicit structured arm definitions")

    arms = []
    role_counts = {role: 0 for role in ARM_ROLES}
    for name in sorted(raw):
        definition = raw[name]
        if not isinstance(name, str) or not name.strip() or not isinstance(definition, Mapping):
            raise ControllerError("arm definitions must map non-empty names to objects")
        role = definition.get("role")
        if role not in ARM_ROLES:
            raise ControllerError(f"arm {name!r} must declare one of the roles {list(ARM_ROLES)}")
        expected_fields = {"role", "parameters", "target"} if role == "lesion" else {"role", "parameters"}
        if set(definition) != expected_fields:
            raise ControllerError(f"arm {name!r} has missing or extra role fields")
        parameters = definition.get("parameters")
        if not isinstance(parameters, Mapping) or _contains_forbidden_data(parameters):
            raise ControllerError(f"arm {name!r} has invalid, seed-bearing, or held-out parameters")
        arm = {"name": name, "role": role, "parameters": dict(parameters)}
        if role == "lesion":
            target = definition.get("target")
            if not isinstance(target, str) or not target.strip():
                raise ControllerError(f"lesion arm {name!r} requires a non-empty target")
            arm["target"] = target.strip()
        role_counts[role] += 1
        arms.append(arm)
    missing = [role for role, count in role_counts.items() if count == 0]
    if missing:
        raise ControllerError(f"arm materialization is missing required roles: {missing}")
    return arms


def _require_clean_sealed_source(source: Any, root: Path) -> None:
    if not isinstance(source, Mapping) or source.get("kind") not in {"git", "git_archive"}:
        raise ControllerError("execution materialization requires a sealed source identity")
    revision = source.get("revision")
    if not isinstance(revision, str) or not revision.strip():
        raise ControllerError("sealed source identity has no revision")
    if source["kind"] == "git_archive":
        revision_file = root / ".source_revision"
        manifest = root / ".source_manifest.sha256"
        if not revision_file.is_file() or not manifest.is_file():
            raise ControllerError("sealed archive source identity is missing")
        values = {}
        for line in revision_file.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator:
                values[key] = value
        if (values.get("source_kind") != "git_archive" or values.get("git_sha") != revision
                or values.get("source_manifest_sha256") != source.get("source_manifest_sha256")
                or _digest_file(manifest) != source.get("source_manifest_sha256")):
            raise ControllerError("sealed archive source identity changed after handoff")
        declared = {}
        for line in manifest.read_text(encoding="utf-8").splitlines():
            digest, separator, relative = line.partition("  ")
            path = PurePosixPath(relative)
            if (not separator or len(digest) != 64 or path.is_absolute() or ".." in path.parts
                    or relative in declared):
                raise ControllerError("sealed archive source manifest is malformed")
            try:
                int(digest, 16)
            except ValueError as exc:
                raise ControllerError("sealed archive source manifest has an invalid digest") from exc
            declared[relative] = digest
        actual = set()
        for relative_root in ("sim", "research/runners", "experiment", "tools"):
            source_root = root / relative_root
            if source_root.is_dir():
                actual.update(
                    path.relative_to(root).as_posix()
                    for path in source_root.rglob("*")
                    if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".sh"}
                )
        if (root / "research/__init__.py").is_file():
            actual.add("research/__init__.py")
        if set(declared) != actual:
            raise ControllerError("sealed archive source file set changed after handoff")
        for relative, digest in declared.items():
            path = root.joinpath(*PurePosixPath(relative).parts)
            if not path.is_file() or _digest_file(path) != digest:
                raise ControllerError(f"sealed archive source file changed after handoff: {relative}")
        return

    try:
        current = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, check=True,
            capture_output=True, text=True, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ControllerError(f"cannot verify sealed Git source: {exc}") from exc
    if current != revision:
        raise ControllerError("source revision changed after sealed handoff")
    if dirty:
        raise ControllerError("execution materialization requires a clean sealed source")


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
        arms = _arm_contract(spec)
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
    expected_arms = {arm["name"] for arm in arms}
    expanded_arms = {
        pair: {job.get("arm") for job in jobs if (job.get("backend"), job.get("partition")) == pair}
        for pair in expanded_pairs
    }
    invalid_arm_sets = sorted(pair for pair, names in expanded_arms.items() if names != expected_arms)
    if invalid_arm_sets:
        raise ControllerError(f"sealed expansion has extra or missing arms for mappings: {invalid_arm_sets}")
    identity_fields = ("spec_sha256", "execution_manifest_sha256", "source")
    identities = {
        field: {_canonical(job.get(field)) for job in jobs}
        for field in identity_fields
    }
    if any(len(values) != 1 for values in identities.values()):
        raise ControllerError("sealed expansion contains inconsistent source or manifest identities")

    result = {
        "schema": HANDOFF_SCHEMA,
        "plan_sha256": plan["sha256"],
        "experiment_id": spec["id"],
        "spec_path": relative.as_posix(),
        "spec_sha256": jobs[0]["spec_sha256"],
        "execution_manifest_sha256": jobs[0]["execution_manifest_sha256"],
        "source": jobs[0]["source"],
        "sealed": True,
        "candidate_count": len(candidates),
        "arms": arms,
        "backend_partition_pairs": [
            {"backend": backend, "partition": partition}
            for backend, partition in sorted(pairs)
        ],
        "expanded_job_count": len(jobs),
        "seeds_selected": False,
        "held_out_partitions_accessed": [],
    }
    result["sha256"] = _digest(result)
    return result


def materialize_execution_manifest(
    plan: Mapping[str, Any],
    handoff: Mapping[str, Any],
    *,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Materialize exact candidate/arm cells without selecting seeds or emitting commands."""
    if (not isinstance(plan, Mapping) or plan.get("schema") != SCHEMA or not _valid_digest(plan)):
        raise ControllerError("execution materialization requires a valid controller plan")
    if (not isinstance(handoff, Mapping) or handoff.get("schema") != HANDOFF_SCHEMA
            or not _valid_digest(handoff)):
        raise ControllerError("execution materialization requires a valid sealed handoff")
    if (handoff.get("sealed") is not True or handoff.get("plan_sha256") != plan["sha256"]
            or handoff.get("seeds_selected") is not False
            or handoff.get("held_out_partitions_accessed") != []):
        raise ControllerError("execution materialization requires the matching seed-free sealed handoff")
    for field in ("spec_sha256", "execution_manifest_sha256"):
        value = handoff.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ControllerError(f"sealed handoff has no valid {field}")
        try:
            int(value, 16)
        except ValueError as exc:
            raise ControllerError(f"sealed handoff has no valid {field}") from exc
    if _contains_forbidden_data({
        "arms": handoff.get("arms"),
        "backend_partition_pairs": handoff.get("backend_partition_pairs"),
    }):
        raise ControllerError("sealed handoff exposes seed or held-out data")

    root = Path(root).resolve()
    _require_clean_sealed_source(handoff.get("source"), root)
    materialization = plan.get("candidate_materialization")
    candidates = materialization.get("candidates") if isinstance(materialization, Mapping) else None
    if (not isinstance(candidates, list) or not candidates
            or materialization.get("count") != len(candidates)
            or handoff.get("candidate_count") != len(candidates)):
        raise ControllerError("sealed handoff has extra or missing candidates")

    raw_arms = handoff.get("arms")
    if not isinstance(raw_arms, list) or not raw_arms:
        raise ControllerError("sealed handoff has no arm contract")
    arm_names = set()
    roles = {role: [] for role in ARM_ROLES}
    arms = []
    for arm in raw_arms:
        if not isinstance(arm, Mapping):
            raise ControllerError("sealed handoff contains an invalid arm")
        name = arm.get("name")
        role = arm.get("role")
        expected = {"name", "role", "parameters", "target"} if role == "lesion" else {
            "name", "role", "parameters"
        }
        if (not isinstance(name, str) or not name.strip() or name in arm_names or role not in ARM_ROLES
                or set(arm) != expected or not isinstance(arm.get("parameters"), Mapping)
                or _contains_forbidden_data(arm)):
            raise ControllerError("sealed handoff has extra, duplicate, or malformed arms")
        if role == "lesion" and (not isinstance(arm.get("target"), str) or not arm["target"].strip()):
            raise ControllerError(f"lesion arm {name!r} has no target")
        arm_names.add(name)
        roles[role].append(name)
        arms.append(dict(arm))
    missing_roles = [role for role, names in roles.items() if not names]
    if missing_roles:
        raise ControllerError(f"sealed handoff is missing controls or lesions: {missing_roles}")
    arms.sort(key=lambda item: item["name"])
    for names in roles.values():
        names.sort()

    candidate_ids = set()
    for index, candidate in enumerate(candidates):
        identifier = candidate.get("candidate_id") if isinstance(candidate, Mapping) else None
        if (not isinstance(identifier, str) or not identifier.strip() or identifier in candidate_ids
                or candidate.get("order") != index):
            raise ControllerError("candidate materialization has an invalid order or duplicate identity")
        candidate_ids.add(identifier)
    actual_pairs = {(candidate.get("backend"), candidate.get("partition")) for candidate in candidates}
    if any(not isinstance(backend, str) or not isinstance(partition, str)
           or "".join(character for character in partition.lower() if character.isalnum()).startswith("heldout")
           for backend, partition in actual_pairs):
        raise ControllerError("candidate materialization contains an invalid or held-out mapping")
    declared_pairs = handoff.get("backend_partition_pairs")
    if not isinstance(declared_pairs, list):
        raise ControllerError("sealed handoff has no backend/partition contract")
    pair_rows = []
    for pair in declared_pairs:
        if not isinstance(pair, Mapping) or set(pair) != {"backend", "partition"}:
            raise ControllerError("sealed handoff contains an invalid backend/partition mapping")
        pair_rows.append((pair.get("backend"), pair.get("partition")))
    if len(set(pair_rows)) != len(pair_rows) or set(pair_rows) != actual_pairs:
        raise ControllerError("sealed handoff expands or omits a backend/partition mapping")

    cells = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or _contains_forbidden_data(candidate):
            raise ControllerError("candidate materialization contains seed or held-out data")
        identifier = candidate.get("candidate_id")
        parameters = candidate.get("parameters")
        if not isinstance(identifier, str) or not identifier.strip() or not isinstance(parameters, Mapping):
            raise ControllerError("candidate materialization is malformed")
        for arm in arms:
            cell = {
                "candidate_id": identifier,
                "candidate_order": candidate.get("order"),
                "arm": arm["name"],
                "role": arm["role"],
                "candidate_parameters": dict(parameters),
                "arm_parameters": dict(arm["parameters"]),
                "backend": candidate.get("backend"),
                "partition": candidate.get("partition"),
                "resource_lane": candidate.get("resource_lane"),
                "device": candidate.get("device"),
            }
            if arm["role"] == "lesion":
                cell["lesion_target"] = arm["target"]
            cell["materialization_id"] = _digest(cell)[:24]
            cells.append(cell)
    cells.sort(key=lambda item: (item["candidate_order"], item["arm"]))

    manifest = {
        "schema": EXECUTION_MANIFEST_SCHEMA,
        "plan_sha256": plan["sha256"],
        "handoff_sha256": handoff["sha256"],
        "experiment_id": handoff.get("experiment_id"),
        "spec_path": handoff.get("spec_path"),
        "spec_sha256": handoff.get("spec_sha256"),
        "sealed_execution_manifest_sha256": handoff.get("execution_manifest_sha256"),
        "sealed_expanded_job_count": handoff.get("expanded_job_count"),
        "source": handoff.get("source"),
        "arms": arms,
        "required_roles": roles,
        "backend_partition_pairs": [
            {"backend": backend, "partition": partition} for backend, partition in sorted(actual_pairs)
        ],
        "materialization_count": len(cells),
        "materializations": cells,
        "receipt_contract": {
            "schema": EXECUTION_RECEIPT_SCHEMA,
            "required_materialization_ids": [cell["materialization_id"] for cell in cells],
            "status": "accepted_non_dispatching",
            "exact_set_required": True,
        },
        "dispatch": {"performed": False, "commands_emitted": False},
        "seeds_selected": False,
        "held_out_partitions_accessed": [],
    }
    manifest["sha256"] = _digest(manifest)
    return manifest


def validate_execution_receipt(manifest: Mapping[str, Any], receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an exact, seed-free acknowledgement of the non-dispatching materialization."""
    if (not isinstance(manifest, Mapping) or manifest.get("schema") != EXECUTION_MANIFEST_SCHEMA
            or not _valid_digest(manifest)):
        raise ControllerError("execution receipt requires a valid execution manifest")
    if not isinstance(receipt, Mapping) or set(receipt) != {
        "schema", "execution_manifest_sha256", "materializations", "dispatch", "seeds_selected",
        "held_out_partitions_accessed", "sha256",
    } or receipt.get("schema") != EXECUTION_RECEIPT_SCHEMA or not _valid_digest(receipt):
        raise ControllerError("execution receipt has an invalid schema, shape, or digest")
    if (receipt.get("execution_manifest_sha256") != manifest["sha256"]
            or receipt.get("dispatch") != {"performed": False, "commands_emitted": False}
            or receipt.get("seeds_selected") is not False
            or receipt.get("held_out_partitions_accessed") != []
            or _contains_forbidden_data(receipt.get("materializations"))):
        raise ControllerError("execution receipt violates the sealed non-dispatching boundary")

    expected = {
        cell["materialization_id"]: {
            "materialization_id": cell["materialization_id"],
            "candidate_id": cell["candidate_id"],
            "arm": cell["arm"],
            "role": cell["role"],
            "backend": cell["backend"],
            "partition": cell["partition"],
            "status": "accepted_non_dispatching",
        }
        for cell in manifest["materializations"]
    }
    rows = receipt.get("materializations")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ControllerError("execution receipt materializations must be a list of objects")
    expected_fields = {"materialization_id", "candidate_id", "arm", "role", "backend", "partition", "status"}
    if any(set(row) != expected_fields or not isinstance(row.get("materialization_id"), str)
           for row in rows):
        raise ControllerError("execution receipt materializations have invalid fields or identities")
    observed = {row["materialization_id"]: dict(row) for row in rows}
    if len(observed) != len(rows) or observed != expected:
        raise ControllerError("execution receipt has extra, missing, or expanded arms, controls, or lesions")
    return {
        "schema": EXECUTION_RECEIPT_SCHEMA,
        "execution_manifest_sha256": manifest["sha256"],
        "accepted": True,
        "materialization_count": len(rows),
        "dispatch_performed": False,
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
