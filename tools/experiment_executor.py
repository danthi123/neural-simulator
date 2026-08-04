#!/usr/bin/env python3
"""Execute only jobs already authorized by sealed experiment manifests.

This layer does not load an experiment specification, expand partitions, choose
seeds, or alter arms. It validates an exact pre-expanded job set, creates
durable per-job state, and either runs a local CPU job or submits an existing
GPU/pool job to the sanctioned queue.
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import socket
import stat
import subprocess
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "sim-experiment-executor-manifest-v1"
RECEIPT_SCHEMA = "sim-experiment-executor-receipt-v1"
MATERIALIZATION_SCHEMA = "sim-experiment-controller-execution-manifest-v1"
JOB_SCHEMA = "sim-experiment-plan-v1"
CONTRACT_SCHEMA = "sim-experiment-job-contract-v1"
GPU_LEASE = "/tmp/sim-local-model-gpu0.lock"
LANES = {"local": "local_cpu", "gpu": "local_gpu", "pool": "mini_pc_cluster"}
RESOURCE_LANES = set(LANES.values())
HEX_20 = re.compile(r"^[0-9a-f]{20}$")
HEX_24 = re.compile(r"^[0-9a-f]{24}$")
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
JOB_FIELDS = {
    "schema", "job_id", "experiment_id", "spec_sha256", "execution_manifest_sha256",
    "corpus_check_sha256", "source", "partition", "backend", "device", "arm", "seed",
    "output", "sealed", "lane", "command", "enqueue_command", "output_claim",
    "execution_contract",
}
CONTRACT_FIELDS = {
    "schema", "job_id", "experiment_id", "spec_sha256", "execution_manifest_sha256",
    "corpus_check_sha256", "source", "partition", "backend", "device", "arm", "seed",
    "output", "execution_snapshot", "runner_command", "environment", "claim_stale_seconds",
}
IDENTITY_FIELDS = (
    "job_id", "experiment_id", "spec_sha256", "execution_manifest_sha256",
    "corpus_check_sha256", "source", "partition", "backend", "device", "arm", "seed", "output",
)


class ExecutorError(ValueError):
    """Raised when execution would exceed a sealed authorization."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _valid_digest(value: Mapping[str, Any]) -> bool:
    try:
        return value.get("sha256") == _digest({key: item for key, item in value.items() if key != "sha256"})
    except (TypeError, ValueError):
        return False


def _held_out(value: Any) -> bool:
    return isinstance(value, str) and "".join(ch for ch in value.lower() if ch.isalnum()).startswith("heldout")


def _safe_relative(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ExecutorError(f"{field} must be a repository-relative path")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or not path.name or "." in path.parts or ".." in path.parts:
        raise ExecutorError(f"{field} must be a safe repository-relative path")
    return path.as_posix()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _regular_no_symlink(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def _validate_provenance(path: Path, job: Mapping[str, Any]) -> dict[str, Any]:
    sidecar = _load_json(path, "job provenance")
    if not isinstance(sidecar, Mapping):
        raise ExecutorError("job provenance must be a JSON object")
    if sidecar.get("artifact") != job["output"]:
        raise ExecutorError("job provenance artifact does not match the exact output")
    revision = job["source_revision"]
    recorded_revision = sidecar.get("git_sha")
    if (not isinstance(recorded_revision, str) or recorded_revision == "unknown"
            or not revision.startswith(recorded_revision)):
        raise ExecutorError("job provenance source revision does not match the sealed source")
    requested_backend = job["backend"]
    expected_runtime_backend = "cupy" if requested_backend == "cupy" else "numpy"
    if (sidecar.get("sim_backend_requested") != requested_backend
            or sidecar.get("sim_backend") != expected_runtime_backend):
        raise ExecutorError("job provenance backend does not match the sealed backend")
    if not isinstance(sidecar.get("run_id"), str) or not sidecar["run_id"]:
        raise ExecutorError("job provenance has no run identity")
    if job.get("source_kind") == "git_archive":
        if (sidecar.get("source_kind") != "git_archive"
                or sidecar.get("source_manifest_sha256") != job.get("source_manifest_sha256")
                or sidecar.get("source_manifest_verified_at_start") is not True
                or sidecar.get("source_manifest_verified_at_exit") is not True):
            raise ExecutorError("job provenance does not verify the sealed source archive")
    return dict(sidecar)


def _load_json(path: str | Path, label: str) -> Any:
    path = Path(path)
    if not _regular_no_symlink(path):
        raise ExecutorError(f"{label} must be an existing regular file")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExecutorError(f"cannot load {label}: {exc}") from exc


def _expected_wrapper(contract: Mapping[str, Any]) -> str:
    encoded = base64.urlsafe_b64encode(_canonical(contract)).decode("ascii")
    tokens = [
        ".venv/bin/python", "tools/experiment.py", "execute-job", "--contract", encoded,
        "--", *contract["runner_command"],
    ]
    return shlex.join(tokens)


def _expected_queue(lane: str, command: str, corpus_digest: str) -> str | None:
    if lane == "local":
        return None
    reason = "corpus-check:" + corpus_digest[:16]
    return shlex.join(["bash", "tools/queue_add.sh", lane, command, reason])


def _validate_materialization(manifest: Mapping[str, Any]) -> None:
    if (manifest.get("schema") != MATERIALIZATION_SCHEMA or not _valid_digest(manifest)
            or manifest.get("dispatch") != {"performed": False, "commands_emitted": False}
            or manifest.get("seeds_selected") is not False
            or manifest.get("held_out_partitions_accessed") != []):
        raise ExecutorError("executor requires a valid non-dispatching materialization manifest")
    if not isinstance(manifest.get("sealed_expanded_job_count"), int) or manifest["sealed_expanded_job_count"] <= 0:
        raise ExecutorError("materialization manifest has no sealed expanded-job count")
    if not isinstance(manifest.get("materializations"), list) or not manifest["materializations"]:
        raise ExecutorError("materialization manifest has no authorized cells")
    if manifest.get("materialization_count") != len(manifest["materializations"]):
        raise ExecutorError("materialization manifest has an invalid cell count")
    arms = manifest.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ExecutorError("materialization manifest has no arm contract")
    arm_roles: dict[str, str] = {}
    for arm in arms:
        if (not isinstance(arm, Mapping) or not isinstance(arm.get("name"), str)
                or arm.get("role") not in {"treatment", "control", "lesion"}
                or arm["name"] in arm_roles):
            raise ExecutorError("materialization manifest has malformed or duplicate arms")
        arm_roles[arm["name"]] = arm["role"]
    if set(arm_roles.values()) != {"treatment", "control", "lesion"}:
        raise ExecutorError("materialization manifest is missing treatment, control, or lesion coverage")
    cell_ids = set()
    for cell in manifest["materializations"]:
        if (not isinstance(cell, Mapping) or not HEX_24.fullmatch(str(cell.get("materialization_id", "")))
                or cell["materialization_id"] in cell_ids
                or arm_roles.get(cell.get("arm")) != cell.get("role")
                or cell.get("resource_lane") not in RESOURCE_LANES
                or _held_out(cell.get("partition"))):
            raise ExecutorError("materialization manifest has malformed, widened, or duplicate cells")
        cell_ids.add(cell["materialization_id"])
    pairs = manifest.get("backend_partition_pairs")
    if (not isinstance(pairs, list) or any(not isinstance(row, Mapping)
            or set(row) != {"backend", "partition"} or _held_out(row.get("partition")) for row in pairs)):
        raise ExecutorError("materialization manifest has invalid backend/partition pairs")
    pair_values = [(row["backend"], row["partition"]) for row in pairs]
    if len(set(pair_values)) != len(pair_values):
        raise ExecutorError("materialization manifest has duplicate backend/partition pairs")


def _validate_job(job: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    if set(job) != JOB_FIELDS or job.get("schema") != JOB_SCHEMA or job.get("sealed") is not True:
        raise ExecutorError("expanded job has an invalid shape or is not sealed")
    contract = job.get("execution_contract")
    if not isinstance(contract, Mapping) or set(contract) != CONTRACT_FIELDS or contract.get("schema") != CONTRACT_SCHEMA:
        raise ExecutorError(f"job {job.get('job_id')!r} has an invalid execution contract")
    if (not HEX_20.fullmatch(str(job.get("job_id", "")))
            or any(not HEX_64.fullmatch(str(job.get(field, ""))) for field in (
                "spec_sha256", "execution_manifest_sha256", "corpus_check_sha256"))
            or isinstance(job.get("seed"), bool) or not isinstance(job.get("seed"), int)
            or not isinstance(contract.get("runner_command"), list) or not contract["runner_command"]
            or any(not isinstance(token, str) or not token for token in contract["runner_command"])
            or isinstance(contract.get("claim_stale_seconds"), bool)
            or not isinstance(contract.get("claim_stale_seconds"), int)
            or contract["claim_stale_seconds"] < 60):
        raise ExecutorError(f"job {job.get('job_id')!r} has invalid sealed identities or runtime fields")
    source = job.get("source")
    if (not isinstance(source, Mapping) or source.get("kind") not in {"git", "git_archive"}
            or not isinstance(source.get("revision"), str) or not source["revision"]):
        raise ExecutorError(f"job {job.get('job_id')!r} has no sealed source revision")
    if any(job.get(field) != contract.get(field) for field in IDENTITY_FIELDS):
        raise ExecutorError(f"job {job.get('job_id')!r} differs from its execution contract")
    if (job.get("experiment_id") != manifest.get("experiment_id")
            or job.get("spec_sha256") != manifest.get("spec_sha256")
            or job.get("execution_manifest_sha256") != manifest.get("sealed_execution_manifest_sha256")
            or job.get("source") != manifest.get("source")):
        raise ExecutorError(f"job {job.get('job_id')!r} differs from the sealed materialization identity")
    if _held_out(job.get("partition")):
        raise ExecutorError("executor cannot authorize a held-out partition")
    lane = job.get("lane")
    if lane not in LANES:
        raise ExecutorError(f"job {job.get('job_id')!r} has an unsupported resource lane")
    if job.get("command") != _expected_wrapper(contract):
        raise ExecutorError(f"job {job.get('job_id')!r} command is not its exact digest-bound wrapper")
    expected_queue = _expected_queue(lane, job["command"], job["corpus_check_sha256"])
    if job.get("enqueue_command") != expected_queue:
        raise ExecutorError(f"job {job.get('job_id')!r} queue command was widened or changed")
    output = _safe_relative(job.get("output"), "job output")
    if job.get("output_claim") != output + ".claim":
        raise ExecutorError(f"job {job.get('job_id')!r} output claim does not match its output")
    environment = contract.get("environment")
    if (not isinstance(environment, Mapping) or environment.get("SIM_BACKEND") != job.get("backend")
            or any(not isinstance(key, str) or not isinstance(value, str) for key, value in environment.items())):
        raise ExecutorError(f"job {job.get('job_id')!r} backend environment is inconsistent")

    matches = [
        cell for cell in manifest["materializations"]
        if isinstance(cell, Mapping) and cell.get("arm") == job.get("arm")
        and cell.get("backend") == job.get("backend") and cell.get("partition") == job.get("partition")
        and cell.get("resource_lane") == LANES[lane] and cell.get("device") == job.get("device")
    ]
    if len(matches) != 1:
        raise ExecutorError(
            f"job {job.get('job_id')!r} must map to exactly one candidate/arm cell; found {len(matches)}"
        )
    cell = matches[0]
    return {
        "job_id": job["job_id"],
        "job_sha256": _digest(job),
        "materialization_id": cell["materialization_id"],
        "candidate_id": cell["candidate_id"],
        "arm": job["arm"],
        "role": cell["role"],
        "backend": job["backend"],
        "partition": job["partition"],
        "resource_lane": LANES[lane],
        "device": job["device"],
        "source_revision": job["source"].get("revision") if isinstance(job["source"], Mapping) else None,
        "source_kind": job["source"].get("kind") if isinstance(job["source"], Mapping) else None,
        "source_manifest_sha256": (
            job["source"].get("source_manifest_sha256")
            if isinstance(job["source"], Mapping) else None
        ),
        "command": job["command"],
        "queue_command": job["enqueue_command"],
        "output": output,
        "provenance": output + ".prov.json",
        "output_claim": job["output_claim"],
        "claim_stale_seconds": contract["claim_stale_seconds"],
    }


def build_executor_manifest(
    jobs: Sequence[Mapping[str, Any]], materialization: Mapping[str, Any], *,
    gpu_lease_path: str | Path = GPU_LEASE,
) -> dict[str, Any]:
    """Validate and bind an already expanded sealed job set without reopening its spec."""
    _validate_materialization(materialization)
    if not isinstance(jobs, list) or not jobs:
        raise ExecutorError("executor requires a non-empty sealed job list")
    if len(jobs) != materialization["sealed_expanded_job_count"]:
        raise ExecutorError("expanded job set has extra or missing jobs")
    authorized = [_validate_job(job, materialization) for job in jobs]
    job_ids = [item["job_id"] for item in authorized]
    outputs = [item["output"] for item in authorized]
    if len(set(job_ids)) != len(job_ids) or len(set(outputs)) != len(outputs):
        raise ExecutorError("expanded job set has duplicate identities or outputs")

    expected_pairs = {
        (row.get("backend"), row.get("partition"))
        for row in materialization.get("backend_partition_pairs", []) if isinstance(row, Mapping)
    }
    actual_pairs = {(item["backend"], item["partition"]) for item in authorized}
    if actual_pairs != expected_pairs:
        raise ExecutorError("expanded job set widens or omits a backend/partition mapping")
    expected_arms = {arm.get("name") for arm in materialization.get("arms", []) if isinstance(arm, Mapping)}
    for pair in actual_pairs:
        observed = {item["arm"] for item in authorized if (item["backend"], item["partition"]) == pair}
        if observed != expected_arms:
            raise ExecutorError(f"expanded job set has extra or missing controls/lesions for {pair}")

    lease = Path(gpu_lease_path)
    if not lease.is_absolute() or not lease.name:
        raise ExecutorError("shared GPU lease path must be absolute")
    authorized.sort(key=lambda item: item["job_id"])
    body = {
        "schema": SCHEMA,
        "experiment_id": materialization["experiment_id"],
        "materialization_manifest_sha256": materialization["sha256"],
        "sealed_execution_manifest_sha256": materialization["sealed_execution_manifest_sha256"],
        "source": materialization["source"],
        "gpu_lease_path": str(lease),
        "job_count": len(authorized),
        "jobs": authorized,
        "held_out_partitions_accessed": [],
        "seed_selection_performed": False,
    }
    body["sha256"] = _digest(body)
    return body


def _validate_plan(plan: Mapping[str, Any]) -> None:
    if (plan.get("schema") != SCHEMA or not _valid_digest(plan)
            or plan.get("held_out_partitions_accessed") != []
            or plan.get("seed_selection_performed") is not False
            or plan.get("job_count") != len(plan.get("jobs", []))):
        raise ExecutorError("executor manifest is invalid")


def _atomic_json(path: Path, value: Mapping[str, Any], mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.chmod(mode)
    os.replace(temporary, path)


def initialize_state(plan: Mapping[str, Any], state_dir: str | Path, *, owner_root: str | Path) -> Path:
    """Create immutable authorization and one durable pending receipt per exact job."""
    _validate_plan(plan)
    owner = Path(owner_root).expanduser().resolve()
    state = Path(state_dir).expanduser().resolve()
    if not _inside(state, owner) or state.exists():
        raise ExecutorError("state directory must be a new path inside its explicit owner root")
    state.mkdir(parents=True)
    _atomic_json(state / "manifest.json", plan, 0o444)
    now = time.time()
    for job in plan["jobs"]:
        receipt = {
            "schema": RECEIPT_SCHEMA, "executor_manifest_sha256": plan["sha256"],
            "job_id": job["job_id"], "job_sha256": job["job_sha256"], "status": "pending",
            "attempt": 0, "updated_at": now, "claim": None, "result": None,
        }
        receipt["sha256"] = _digest(receipt)
        _atomic_json(state / "receipts" / f"{job['job_id']}.json", receipt)
    (state / ".lock").touch(mode=0o600)
    return state


def _job(plan: Mapping[str, Any], job_id: str) -> Mapping[str, Any]:
    matches = [job for job in plan["jobs"] if job.get("job_id") == job_id]
    if len(matches) != 1:
        raise ExecutorError("job id is not an exact member of the executor manifest")
    return matches[0]


def _receipt_path(state: Path, job_id: str) -> Path:
    return state / "receipts" / f"{job_id}.json"


def _pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except PermissionError:
        return True
    except (ProcessLookupError, TypeError, ValueError):
        return False
    return True


def _read_receipt(state: Path, job_id: str, plan: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _load_json(_receipt_path(state, job_id), "job receipt")
    if (not isinstance(receipt, dict) or receipt.get("schema") != RECEIPT_SCHEMA
            or not _valid_digest(receipt) or receipt.get("executor_manifest_sha256") != plan["sha256"]
            or receipt.get("job_sha256") != _job(plan, job_id)["job_sha256"]):
        raise ExecutorError("job receipt is invalid or belongs to another exact manifest")
    return receipt


def claim_job(
    plan: Mapping[str, Any], state_dir: str | Path, job_id: str, *, worker_id: str,
    now: float | None = None,
) -> dict[str, Any]:
    """Atomically claim one exact job and return only its authorized command."""
    _validate_plan(plan)
    if not isinstance(worker_id, str) or not worker_id.strip():
        raise ExecutorError("worker id must be non-empty")
    state = Path(state_dir).resolve()
    with (state / ".lock").open("r+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        receipt = _read_receipt(state, job_id, plan)
        if receipt["status"] != "pending":
            raise ExecutorError(f"job {job_id} is not pending")
        timestamp = time.time() if now is None else float(now)
        token = os.urandom(16).hex()
        receipt.update({
            "status": "running", "attempt": receipt["attempt"] + 1, "updated_at": timestamp,
            "claim": {"token": token, "worker_id": worker_id.strip(), "hostname": socket.gethostname(),
                      "pid": os.getpid(), "started_at": timestamp}, "result": None,
        })
        receipt["sha256"] = _digest({key: value for key, value in receipt.items() if key != "sha256"})
        _atomic_json(_receipt_path(state, job_id), receipt)
    job = _job(plan, job_id)
    return {
        "job_id": job_id, "claim_token": token, "resource_lane": job["resource_lane"],
        "command": job["command"] if job["resource_lane"] == "local_cpu" else job["queue_command"],
        "gpu_lease_path": plan["gpu_lease_path"] if job["resource_lane"] == "local_gpu" else None,
    }


def finish_job(
    plan: Mapping[str, Any], state_dir: str | Path, job_id: str, claim_token: str, *,
    status: str, exit_code: int | None = None, root: str | Path = ROOT,
    now: float | None = None,
) -> dict[str, Any]:
    """Finish a claim, verifying exact output and provenance before success."""
    if status not in {"queued", "succeeded", "failed"}:
        raise ExecutorError("finish status must be queued, succeeded, or failed")
    _validate_plan(plan)
    state = Path(state_dir).resolve()
    job = _job(plan, job_id)
    root = Path(root).resolve()
    with (state / ".lock").open("r+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        receipt = _read_receipt(state, job_id, plan)
        if receipt["status"] != "running" or receipt.get("claim", {}).get("token") != claim_token:
            raise ExecutorError("finish does not own the active exact-job claim")
        if status == "queued" and job["resource_lane"] == "local_cpu":
            raise ExecutorError("local CPU jobs cannot be marked queued")
        result: dict[str, Any] = {"exit_code": exit_code}
        if status == "succeeded":
            if exit_code != 0:
                raise ExecutorError("a successful receipt requires exit code zero")
            for field in ("output", "provenance"):
                path = root.joinpath(*PurePosixPath(job[field]).parts)
                if not _inside(path, root) or not _regular_no_symlink(path):
                    raise ExecutorError(f"successful job is missing exact {field} artifact")
                result[field + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            provenance_path = root.joinpath(*PurePosixPath(job["provenance"]).parts)
            provenance = _validate_provenance(provenance_path, job)
            result["provenance_run_id"] = provenance["run_id"]
        receipt.update({"status": status, "updated_at": time.time() if now is None else float(now),
                        "claim": None, "result": result})
        receipt["sha256"] = _digest({key: value for key, value in receipt.items() if key != "sha256"})
        _atomic_json(_receipt_path(state, job_id), receipt)
    return receipt


def recover_stale(
    plan: Mapping[str, Any], state_dir: str | Path, *, now: float | None = None,
) -> list[str]:
    """Return stale running claims to pending; queued jobs are never guessed stale."""
    _validate_plan(plan)
    state = Path(state_dir).resolve()
    timestamp = time.time() if now is None else float(now)
    recovered = []
    with (state / ".lock").open("r+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for job in plan["jobs"]:
            receipt = _read_receipt(state, job["job_id"], plan)
            claim = receipt.get("claim")
            if receipt["status"] != "running" or not isinstance(claim, Mapping):
                continue
            if timestamp - float(claim.get("started_at", timestamp)) <= job["claim_stale_seconds"]:
                continue
            if claim.get("hostname") == socket.gethostname() and _pid_alive(claim.get("pid")):
                continue
            receipt.update({"status": "pending", "updated_at": timestamp, "claim": None,
                            "result": {"recovered_stale_attempt": receipt["attempt"]}})
            receipt["sha256"] = _digest({key: value for key, value in receipt.items() if key != "sha256"})
            _atomic_json(_receipt_path(state, job["job_id"]), receipt)
            recovered.append(job["job_id"])
    return recovered


def execute_claimed(
    plan: Mapping[str, Any], state_dir: str | Path, job_id: str, *, worker_id: str,
    root: str | Path = ROOT, dry_run: bool = False,
) -> dict[str, Any]:
    """Claim and run/queue one exact job. Dry-run has no durable side effects."""
    _validate_plan(plan)
    job = _job(plan, job_id)
    command = job["command"] if job["resource_lane"] == "local_cpu" else job["queue_command"]
    if dry_run:
        return {"job_id": job_id, "resource_lane": job["resource_lane"], "command": command,
                "performed": False}
    claim = claim_job(plan, state_dir, job_id, worker_id=worker_id)
    try:
        result = subprocess.run(shlex.split(command), cwd=Path(root).resolve(), check=False)
        if job["resource_lane"] == "local_cpu":
            status = "succeeded" if result.returncode == 0 else "failed"
        else:
            status = "queued" if result.returncode == 0 else "failed"
        return finish_job(plan, state_dir, job_id, claim["claim_token"], status=status,
                          exit_code=result.returncode, root=root)
    except BaseException:
        finish_job(plan, state_dir, job_id, claim["claim_token"], status="failed", root=root)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute only an exact, pre-expanded sealed experiment job set.")
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--jobs", required=True)
    prepare.add_argument("--materialization", required=True)
    prepare.add_argument("--state-dir")
    prepare.add_argument("--owner-root")
    prepare.add_argument("--gpu-lease", default=GPU_LEASE)
    prepare.add_argument("--dry-run", action="store_true")
    run = sub.add_parser("run-job")
    run.add_argument("--state-dir", required=True)
    run.add_argument("--job-id", required=True)
    run.add_argument("--worker-id", required=True)
    run.add_argument("--root", default=str(ROOT))
    run.add_argument("--dry-run", action="store_true")
    recover = sub.add_parser("recover")
    recover.add_argument("--state-dir", required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare":
            jobs = _load_json(args.jobs, "sealed job expansion")
            materialization = _load_json(args.materialization, "materialization manifest")
            plan = build_executor_manifest(jobs, materialization, gpu_lease_path=args.gpu_lease)
            if not args.dry_run:
                if not args.state_dir or not args.owner_root:
                    raise ExecutorError("non-dry-run prepare requires --state-dir and --owner-root")
                initialize_state(plan, args.state_dir, owner_root=args.owner_root)
            print(json.dumps(plan, sort_keys=True, indent=2))
        else:
            state = Path(args.state_dir)
            plan = _load_json(state / "manifest.json", "executor manifest")
            if args.action == "run-job":
                print(json.dumps(execute_claimed(plan, state, args.job_id, worker_id=args.worker_id,
                                                root=args.root, dry_run=args.dry_run), sort_keys=True))
            else:
                print(json.dumps({"recovered": recover_stale(plan, state)}, sort_keys=True))
    except (ExecutorError, OSError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
