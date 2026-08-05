#!/usr/bin/env python3
"""Advance an adaptive campaign by one fail-closed lifecycle transition.

The supervisor never runs experiment jobs, opens held-out partitions, or makes a
scientific judgement.  It creates preparation artifacts through the existing
experiment APIs and otherwise emits one exact command that an operator or worker
may execute.  Every transition is an immutable, self-digested chain record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
from typing import Any, Mapping, Sequence

from tools.adaptive_experiment import load_adaptive_design
from tools.experiment_controller import (
    build_dry_run_plan,
    materialize_candidate_spec,
    materialize_execution_manifest,
    validate_experiment_handoff,
    write_dry_run_plan,
)
from tools.experiment_executor import (
    _read_receipt,
    _validate_plan,
    build_executor_manifest,
    initialize_state,
)
from tools.experiment_observation import digest


SCHEMA = "sim-adaptive-campaign-supervisor-state-v1"
CANONICALIZATION = "json-sort-keys-compact-ascii-v1"


class CampaignSupervisorError(ValueError):
    """Raised when campaign custody or prerequisites are incomplete."""


def _fail(message: str) -> None:
    raise CampaignSupervisorError(message)


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.absolute().relative_to(root)
    except ValueError:
        return False
    return True


def _safe_path(root: Path, value: str | Path, label: str, *, exists: bool) -> Path:
    supplied = Path(value).expanduser()
    path = supplied if supplied.is_absolute() else root / supplied
    path = path.absolute()
    if not _inside(path, root):
        _fail(f"{label} must be inside repository root")
    current = root
    relative = path.relative_to(root)
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        _fail(f"{label} is not a canonical repository-relative path")
    for part in relative.parts:
        current /= part
        if os.path.lexists(current) and current.is_symlink():
            _fail(f"{label} cannot contain a symlink")
    if exists and (not path.is_file() or path.is_symlink()):
        _fail(f"{label} must be an existing regular file")
    return path


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_bytes(),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CampaignSupervisorError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{label} must contain a JSON object")
    return value


def _write_once(path: Path, value: Mapping[str, Any], mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False,
    ).encode("ascii") + b"\n"
    try:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise CampaignSupervisorError(f"refusing to replace existing artifact: {path}") from exc
    path.chmod(mode)


def _quote(*parts: object) -> str:
    return shlex.join([str(part) for part in parts])


def _binding(path: Path, root: Path) -> dict[str, str]:
    return {"path": path.relative_to(root).as_posix(), "sha256": _file_sha(path)}


def _state_files(campaign_dir: Path) -> list[Path]:
    state_dir = campaign_dir / "state"
    if not state_dir.exists():
        return []
    files = sorted(state_dir.glob("*.json"))
    expected = [state_dir / f"{index:06d}.json" for index in range(1, len(files) + 1)]
    if files != expected:
        _fail("supervisor state sequence has gaps or unexpected JSON files")
    return files


def _load_chain(campaign_dir: Path) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    previous = None
    for index, path in enumerate(_state_files(campaign_dir), 1):
        state = _load_json(path, "supervisor state")
        body = {key: value for key, value in state.items() if key != "sha256"}
        if (state.get("schema") != SCHEMA or state.get("sequence") != index
                or state.get("previous_sha256") != previous
                or state.get("sha256") != digest(body)
                or state.get("scientific_verdict") is not None
                or state.get("held_out_partitions_accessed") != []):
            _fail(f"supervisor state chain is invalid at sequence {index}")
        chain.append(state)
        previous = state["sha256"]
    return chain


def _record(
    campaign_dir: Path,
    chain: Sequence[Mapping[str, Any]],
    transition: str,
    *,
    observed: Sequence[Mapping[str, str]] = (),
    created: Sequence[Mapping[str, str]] = (),
    requirements: Sequence[str] = (),
    command: str | None = None,
) -> dict[str, Any]:
    body = {
        "schema": SCHEMA,
        "canonicalization": CANONICALIZATION,
        "sequence": len(chain) + 1,
        "previous_sha256": chain[-1]["sha256"] if chain else None,
        "transition": transition,
        "observed": list(observed),
        "created": list(created),
        "requirements": list(requirements),
        "authorized_command": command,
        "execution_performed": False,
        "held_out_partitions_accessed": [],
        "scientific_verdict": None,
    }
    state = {**body, "sha256": digest(body)}
    _write_once(campaign_dir / "state" / f"{len(chain) + 1:06d}.json", state)
    return state


def _same_snapshot(
    latest: Mapping[str, Any] | None,
    transition: str,
    observed: Sequence[Mapping[str, str]],
) -> bool:
    return bool(latest and latest.get("transition") == transition
                and latest.get("observed") == list(observed))


def _partitions_from_plan(plan: Mapping[str, Any]) -> list[str]:
    candidates = plan.get("candidate_materialization", {}).get("candidates", [])
    partitions = sorted({row.get("partition") for row in candidates if isinstance(row, Mapping)})
    if not partitions or any(not isinstance(item, str) or "heldout" in "".join(
            character for character in item.lower() if character.isalnum()) for item in partitions):
        _fail("controller plan has no exclusively non-held-out candidate partitions")
    return partitions


def _read_jobs(plan_dir: Path) -> list[dict[str, Any]]:
    index = _load_json(plan_dir / "plan.json", "sealed plan index")
    job_ids = index.get("job_ids")
    if not isinstance(job_ids, list) or not job_ids:
        _fail("sealed plan index contains no jobs")
    jobs = []
    for path in sorted(plan_dir.glob("*.json")):
        if path.name == "plan.json":
            continue
        job = _load_json(path, "sealed job")
        if job.get("job_id") in job_ids:
            jobs.append(job)
    if len(jobs) != len(job_ids) or {job.get("job_id") for job in jobs} != set(job_ids):
        _fail("sealed plan has extra or missing job contracts")
    return jobs


def _receipt_snapshot(state_dir: Path, manifest: Mapping[str, Any], root: Path) -> list[dict[str, str]]:
    result = [_binding(state_dir / "manifest.json", root)]
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs or any(
            not isinstance(job, Mapping) or not isinstance(job.get("job_id"), str) for job in jobs):
        _fail("executor manifest has no exact job set")
    for job in jobs:
        result.append(_binding(state_dir / "receipts" / f"{job['job_id']}.json", root))
    return result


def advance_campaign(
    design_path: str | Path,
    campaign_dir: str | Path,
    *,
    repository_root: str | Path,
    observation_contract_path: str | Path | None = None,
    next_design_id: str | None = None,
) -> dict[str, Any]:
    """Perform at most one deterministic transition and return its immutable state."""
    root = Path(repository_root).expanduser().resolve(strict=True)
    design_file = _safe_path(root, design_path, "adaptive design", exists=True)
    campaign = _safe_path(root, campaign_dir, "campaign directory", exists=False)
    if campaign.exists() and not campaign.is_dir():
        _fail("campaign directory must be a directory")
    campaign.mkdir(parents=True, exist_ok=True)
    chain = _load_chain(campaign)
    latest = chain[-1] if chain else None

    # Validation here also proves that the base spec and proposed partitions are admissible.
    design = load_adaptive_design(design_file, root=root)
    plan_path = campaign / "controller-plan.json"
    candidate_spec = campaign / "candidate-spec.json"
    seal_path = campaign / "experiment-seal.json"
    handoff_path = campaign / "sealed-handoff.json"
    materialization_path = campaign / "materialization.json"
    plan_dir = campaign / "sealed-jobs"
    executor_manifest_path = campaign / "executor-manifest.json"
    executor_state = campaign / "executor-state"
    observations_path = campaign / "observations.json"
    updated_design_path = campaign / "next-design.json"
    update_receipt_path = campaign / "next-design.update.json"

    if not plan_path.exists():
        plan = build_dry_run_plan(design_file, root=root)
        if plan.get("decision") != "propose":
            observed = [_binding(design_file, root)]
            transition = "blocked_non_proposal"
            if _same_snapshot(latest, transition, observed):
                return dict(latest)
            return _record(campaign, chain, transition, observed=observed,
                           requirements=("manual review of the non-proposal adaptive decision",))
        write_dry_run_plan(plan, plan_path, owner_root=campaign)
        return _record(campaign, chain, "controller_plan_created",
                       observed=(_binding(design_file, root),),
                       created=(_binding(plan_path, root),))

    plan = _load_json(plan_path, "controller plan")
    expected_plan = build_dry_run_plan(design_file, root=root)
    if plan != expected_plan:
        _fail("controller plan no longer equals the deterministic design proposal")
    if not candidate_spec.exists():
        materialize_candidate_spec(plan, candidate_spec, root=root)
        return _record(campaign, chain, "candidate_spec_created",
                       observed=(_binding(plan_path, root),),
                       created=(_binding(candidate_spec, root),))

    if not seal_path.exists():
        observed = [_binding(candidate_spec, root)]
        transition = "seal_authorized"
        if _same_snapshot(latest, transition, observed):
            return dict(latest)
        command = _quote("python", "tools/experiment.py", "seal", "--spec",
                         candidate_spec.relative_to(root), "--seal", seal_path.relative_to(root))
        return _record(campaign, chain, transition, observed=observed,
                       requirements=("create the experiment seal with the existing harness",), command=command)

    if not handoff_path.exists():
        handoff = validate_experiment_handoff(
            plan, seal_path=seal_path, materialized_spec_path=candidate_spec, root=root,
        )
        _write_once(handoff_path, handoff)
        return _record(campaign, chain, "sealed_handoff_validated",
                       observed=(_binding(plan_path, root), _binding(candidate_spec, root),
                                 _binding(seal_path, root)),
                       created=(_binding(handoff_path, root),))

    handoff = _load_json(handoff_path, "sealed handoff")
    expected_handoff = validate_experiment_handoff(
        plan, seal_path=seal_path, materialized_spec_path=candidate_spec, root=root,
    )
    if handoff != expected_handoff:
        _fail("sealed handoff is invalid or no longer matches its plan and seal")
    if not materialization_path.exists():
        materialization = materialize_execution_manifest(plan, handoff, root=root)
        _write_once(materialization_path, materialization)
        return _record(campaign, chain, "execution_materialization_created",
                       observed=(_binding(handoff_path, root),),
                       created=(_binding(materialization_path, root),))

    materialization = _load_json(materialization_path, "execution materialization")
    expected_materialization = materialize_execution_manifest(plan, handoff, root=root)
    if materialization != expected_materialization:
        _fail("execution materialization is invalid or no longer matches its handoff")
    if not plan_dir.exists():
        observed = [_binding(materialization_path, root)]
        transition = "sealed_expansion_authorized"
        if _same_snapshot(latest, transition, observed):
            return dict(latest)
        command_parts: list[object] = ["python", "tools/experiment.py", "plan", "--spec",
                                       candidate_spec.relative_to(root)]
        for partition in _partitions_from_plan(plan):
            command_parts.extend(("--partition", partition))
        command_parts.extend(("--seal", seal_path.relative_to(root), "--plan-dir",
                              plan_dir.relative_to(root)))
        return _record(campaign, chain, transition, observed=observed,
                       requirements=("expand only the sealed non-held-out job partitions",),
                       command=_quote(*command_parts))

    if not executor_manifest_path.exists():
        jobs = _read_jobs(plan_dir)
        executor_manifest = build_executor_manifest(jobs, materialization)
        _write_once(executor_manifest_path, executor_manifest)
        return _record(campaign, chain, "executor_manifest_created",
                       observed=(_binding(plan_dir / "plan.json", root),
                                 _binding(materialization_path, root)),
                       created=(_binding(executor_manifest_path, root),))

    executor_manifest = _load_json(executor_manifest_path, "executor manifest")
    try:
        _validate_plan(executor_manifest)
    except Exception as exc:
        raise CampaignSupervisorError(f"executor manifest is invalid: {exc}") from exc
    if not executor_state.exists():
        initialize_state(executor_manifest, executor_state, owner_root=campaign)
        return _record(campaign, chain, "executor_state_initialized",
                       observed=(_binding(executor_manifest_path, root),),
                       created=(_binding(executor_state / "manifest.json", root),))

    receipt_snapshot = _receipt_snapshot(executor_state, executor_manifest, root)
    try:
        receipts = [
            _read_receipt(executor_state, job["job_id"], executor_manifest)
            for job in executor_manifest["jobs"]
        ]
    except Exception as exc:
        raise CampaignSupervisorError(f"executor receipt is invalid: {exc}") from exc
    running = [row for row in receipts if row.get("status") == "running"]
    failed = [row for row in receipts if row.get("status") == "failed"]
    queued = [row for row in receipts if row.get("status") == "queued"]
    pending = [row for row in receipts if row.get("status") == "pending"]
    if failed:
        transition = "blocked_failed_job"
        if _same_snapshot(latest, transition, receipt_snapshot):
            return dict(latest)
        return _record(campaign, chain, transition, observed=receipt_snapshot,
                       requirements=("investigate failed exact jobs; no automatic retry is authorized",))
    if running:
        transition = "recovery_check_authorized"
        if _same_snapshot(latest, transition, receipt_snapshot):
            return dict(latest)
        command = _quote("python", "tools/experiment_executor.py", "recover", "--state-dir",
                         executor_state.relative_to(root))
        return _record(campaign, chain, transition, observed=receipt_snapshot,
                       requirements=("recover only claims proven stale by the executor",), command=command)
    if queued:
        transition = "blocked_queued_reconciliation"
        if _same_snapshot(latest, transition, receipt_snapshot):
            return dict(latest)
        return _record(campaign, chain, transition, observed=receipt_snapshot,
                       requirements=("reconcile queued worker results into authenticated executor receipts",))
    if pending:
        job_id = sorted(row["job_id"] for row in pending)[0]
        transition = "job_authorized"
        if _same_snapshot(latest, transition, receipt_snapshot):
            return dict(latest)
        command = _quote("python", "tools/experiment_executor.py", "run-job", "--state-dir",
                         executor_state.relative_to(root), "--job-id", job_id, "--worker-id",
                         "adaptive-campaign-supervisor", "--root", root)
        return _record(campaign, chain, transition, observed=receipt_snapshot,
                       requirements=(f"execute exact pending job {job_id}",), command=command)

    if any(row.get("status") != "succeeded" for row in receipts):
        _fail("executor receipts contain an unsupported terminal state")
    if observation_contract_path is None:
        transition = "observation_contract_required"
        if _same_snapshot(latest, transition, receipt_snapshot):
            return dict(latest)
        return _record(campaign, chain, transition, observed=receipt_snapshot,
                       requirements=("provide a preregistered observation contract bound to this design and manifest",))

    contract_path = _safe_path(root, observation_contract_path, "observation contract", exists=True)
    contract = _load_json(contract_path, "observation contract")
    declared_output = contract.get("output_path")
    if declared_output != observations_path.relative_to(root).as_posix():
        _fail("observation contract output_path does not name this campaign's observations artifact")
    if not observations_path.exists():
        observed = [*receipt_snapshot, _binding(contract_path, root)]
        transition = "observation_compilation_authorized"
        if _same_snapshot(latest, transition, observed):
            return dict(latest)
        command_parts = ["python", "tools/experiment_observation.py", "--contract", contract_path.relative_to(root),
                         "--executor-manifest", executor_manifest_path.relative_to(root)]
        for job in sorted(executor_manifest["jobs"], key=lambda row: row["job_id"]):
            command_parts.extend(("--receipt", (executor_state / "receipts" /
                                  f"{job['job_id']}.json").relative_to(root)))
        command_parts.extend(("--output", observations_path.relative_to(root),
                              "--repository-root", root))
        return _record(campaign, chain, transition, observed=observed,
                       requirements=("compile authenticated non-held-out receipts only",),
                       command=_quote(*command_parts))

    if not next_design_id:
        transition = "next_design_id_required"
        observed = [_binding(observations_path, root)]
        if _same_snapshot(latest, transition, observed):
            return dict(latest)
        return _record(campaign, chain, transition, observed=observed,
                       requirements=("provide a new immutable adaptive design id",))
    if not updated_design_path.exists():
        observed = [_binding(observations_path, root)]
        transition = "design_update_authorized"
        if _same_snapshot(latest, transition, observed):
            return dict(latest)
        command = _quote("python", "tools/adaptive_design_update.py", "--design",
                         design_file.relative_to(root), "--observations", observations_path.relative_to(root),
                         "--output", updated_design_path.relative_to(root), "--receipt-output",
                         update_receipt_path.relative_to(root), "--new-id", next_design_id,
                         "--repository-root", root)
        return _record(campaign, chain, transition, observed=observed,
                       requirements=("append authenticated observations to a new design version",), command=command)

    observed = [_binding(updated_design_path, root), _binding(update_receipt_path, root)]
    transition = "cycle_complete"
    if _same_snapshot(latest, transition, observed):
        return dict(latest)
    return _record(campaign, chain, transition, observed=observed,
                   requirements=("start a new campaign directory from the updated design",))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", required=True)
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--observation-contract")
    parser.add_argument("--next-design-id")
    args = parser.parse_args(argv)
    try:
        state = advance_campaign(
            args.design, args.campaign_dir, repository_root=args.repository_root,
            observation_contract_path=args.observation_contract,
            next_design_id=args.next_design_id,
        )
    except (CampaignSupervisorError, ValueError, OSError) as exc:
        print(f"adaptive-campaign-supervisor: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(state, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
