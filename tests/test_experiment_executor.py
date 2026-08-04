"""Focused tests for exact-manifest experiment execution."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import shlex

import pytest

import tools.experiment_executor as executor
from tools.experiment_executor import (
    ExecutorError,
    build_executor_manifest,
    claim_job,
    execute_claimed,
    finish_job,
    initialize_state,
    recover_stale,
)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")).hexdigest()


def _materialization() -> dict:
    source = {"kind": "git", "revision": "a" * 40}
    arms = [
        {"name": "candidate", "role": "treatment", "parameters": {"gain": 0.4}},
        {"name": "control", "role": "control", "parameters": {"enabled": False}},
        {"name": "lesion", "role": "lesion", "parameters": {"gain": 0}, "target": "gain"},
    ]
    mappings = [
        ("numpy", "calibration", "local_cpu", "cpu", "cpu-candidate"),
        ("cupy", "calibration", "local_gpu", "cuda:0", "gpu-candidate"),
        ("numpy-pool", "replication", "mini_pc_cluster", "cpu", "pool-candidate"),
    ]
    cells = []
    for order, (backend, partition, lane, device, candidate_id) in enumerate(mappings):
        for arm in arms:
            cell = {
                "candidate_id": candidate_id, "candidate_order": order, "arm": arm["name"],
                "role": arm["role"], "candidate_parameters": {"gain": 0.4},
                "arm_parameters": arm["parameters"], "backend": backend, "partition": partition,
                "resource_lane": lane, "device": device,
            }
            if arm["role"] == "lesion":
                cell["lesion_target"] = arm["target"]
            cell["materialization_id"] = _digest(cell)[:24]
            cells.append(cell)
    manifest = {
        "schema": "sim-experiment-controller-execution-manifest-v1",
        "plan_sha256": "1" * 64, "handoff_sha256": "2" * 64,
        "experiment_id": "executor-fixture", "spec_path": "research/specs/fixture.json",
        "spec_sha256": "3" * 64, "sealed_execution_manifest_sha256": "4" * 64,
        "sealed_expanded_job_count": 9, "source": source, "arms": arms,
        "required_roles": {"treatment": ["candidate"], "control": ["control"], "lesion": ["lesion"]},
        "backend_partition_pairs": [
            {"backend": backend, "partition": partition}
            for backend, partition, *_ in mappings
        ],
        "materialization_count": len(cells), "materializations": cells,
        "receipt_contract": {"schema": "sim-experiment-controller-execution-receipt-v1",
                             "required_materialization_ids": [cell["materialization_id"] for cell in cells],
                             "status": "accepted_non_dispatching", "exact_set_required": True},
        "dispatch": {"performed": False, "commands_emitted": False},
        "seeds_selected": False, "held_out_partitions_accessed": [],
    }
    manifest["sha256"] = _digest(manifest)
    return manifest


def _job(manifest: dict, backend: str, partition: str, lane: str, device: str, arm: str, seed: int) -> dict:
    output = f"research/findings/raw/executor/{partition}/{backend}/{arm}-{seed}.json"
    identity = {
        "experiment_id": manifest["experiment_id"], "spec_sha256": manifest["spec_sha256"],
        "execution_manifest_sha256": manifest["sealed_execution_manifest_sha256"],
        "corpus_check_sha256": "5" * 64, "source": manifest["source"], "partition": partition,
        "backend": backend, "device": device, "arm": arm, "seed": seed, "output": output,
    }
    job_id = _digest(identity)[:20]
    runner = [".venv/bin/python", "-m", "research.runners.fixture", "--seed", str(seed),
              "--phase", partition, "--arm", arm, "--out", output]
    contract = {
        "schema": "sim-experiment-job-contract-v1", "job_id": job_id, **identity,
        "execution_snapshot": {"manifest_sha256": manifest["sealed_execution_manifest_sha256"]},
        "runner_command": runner, "environment": {"SIM_BACKEND": backend},
        "claim_stale_seconds": 60,
    }
    encoded = base64.urlsafe_b64encode(json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")).decode("ascii")
    command = shlex.join([
        ".venv/bin/python", "tools/experiment.py", "execute-job", "--contract", encoded, "--", *runner,
    ])
    enqueue = None if lane == "local" else shlex.join([
        "bash", "tools/queue_add.sh", lane, command, "corpus-check:" + ("5" * 16),
    ])
    return {
        "schema": "sim-experiment-plan-v1", "job_id": job_id, **identity, "sealed": True,
        "lane": lane, "command": command, "enqueue_command": enqueue,
        "output_claim": output + ".claim", "execution_contract": contract,
    }


def _jobs(manifest: dict) -> list[dict]:
    targets = [
        ("numpy", "calibration", "local", "cpu"),
        ("cupy", "calibration", "gpu", "cuda:0"),
        ("numpy-pool", "replication", "pool", "cpu"),
    ]
    return [
        _job(manifest, backend, partition, lane, device, arm, seed)
        for backend, partition, lane, device in targets
        for arm in ("candidate", "control", "lesion")
        for seed in [11]
    ]


def _adaptive_same_pair() -> tuple[dict, list[dict]]:
    manifest = _materialization()
    arms = manifest["arms"]
    cells = []
    for order, (candidate_id, parameters) in enumerate((
        ("candidate-alpha", {"gain": 0.4}),
        ("candidate-beta", {"gain": 0.7}),
    )):
        candidate_document = {
            "schema": "sim-adaptive-candidate-v1",
            "candidate_id": candidate_id,
            "parameters": parameters,
        }
        candidate_sha256 = _digest(candidate_document)
        for arm in arms:
            cell = {
                "candidate_id": candidate_id,
                "candidate_order": order,
                "candidate_document": candidate_document,
                "candidate_sha256": candidate_sha256,
                "arm": arm["name"],
                "role": arm["role"],
                "candidate_parameters": parameters,
                "arm_parameters": arm["parameters"],
                "effective_parameters": {**parameters, **arm["parameters"]},
                "backend": "numpy",
                "partition": "calibration",
                "resource_lane": "local_cpu",
                "device": "cpu",
            }
            if arm["role"] == "lesion":
                cell["lesion_target"] = arm["target"]
            cell["materialization_id"] = _digest(cell)[:24]
            cells.append(cell)
    manifest.update({
        "sealed_expanded_job_count": len(cells),
        "backend_partition_pairs": [{"backend": "numpy", "partition": "calibration"}],
        "materialization_count": len(cells),
        "materializations": cells,
        "receipt_contract": {
            "schema": "sim-experiment-controller-execution-receipt-v1",
            "required_materialization_ids": [cell["materialization_id"] for cell in cells],
            "status": "accepted_non_dispatching",
            "exact_set_required": True,
        },
    })
    manifest["sha256"] = _digest({key: value for key, value in manifest.items() if key != "sha256"})
    return manifest, [_adaptive_job(manifest, cell) for cell in cells]


def _adaptive_job(manifest: dict, cell: dict, seed: int = 11) -> dict:
    output = (
        "research/findings/raw/executor/calibration/numpy/"
        f"{cell['candidate_id']}-{cell['arm']}-{seed}.json"
    )
    parameter_document = {
        "schema": "sim-adaptive-run-parameters-v1",
        "candidate_id": cell["candidate_id"],
        "candidate_sha256": cell["candidate_sha256"],
        "candidate_parameters": cell["candidate_parameters"],
        "arm": cell["arm"],
        "arm_parameters": cell["arm_parameters"],
        "effective_parameters": cell["effective_parameters"],
    }
    identity = {
        "experiment_id": manifest["experiment_id"],
        "spec_sha256": manifest["spec_sha256"],
        "execution_manifest_sha256": manifest["sealed_execution_manifest_sha256"],
        "corpus_check_sha256": "5" * 64,
        "source": manifest["source"],
        "partition": "calibration",
        "backend": "numpy",
        "device": "cpu",
        "arm": cell["arm"],
        "seed": seed,
        "output": output,
        "candidate_id": cell["candidate_id"],
        "candidate_sha256": cell["candidate_sha256"],
        "parameter_document": parameter_document,
    }
    job_id = _digest(identity)[:20]
    runner = [
        ".venv/bin/python", "-m", "research.runners.fixture",
        "--seed", str(seed), "--phase", "calibration", "--arm", cell["arm"], "--out", output,
        "--adaptive-parameter-document",
        json.dumps(parameter_document, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
    ]
    contract = {
        "schema": "sim-experiment-job-contract-v1",
        "job_id": job_id,
        **identity,
        "execution_snapshot": {"manifest_sha256": manifest["sealed_execution_manifest_sha256"]},
        "runner_command": runner,
        "environment": {"SIM_BACKEND": "numpy"},
        "claim_stale_seconds": 60,
    }
    encoded = base64.urlsafe_b64encode(json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")).decode("ascii")
    command = shlex.join([
        ".venv/bin/python", "tools/experiment.py", "execute-job", "--contract", encoded, "--", *runner,
    ])
    return {
        "schema": "sim-experiment-plan-v1",
        "job_id": job_id,
        **identity,
        "sealed": True,
        "lane": "local",
        "command": command,
        "enqueue_command": None,
        "output_claim": output + ".claim",
        "execution_contract": contract,
    }
@pytest.fixture
def exact() -> tuple[dict, list[dict], dict]:
    materialization = _materialization()
    jobs = _jobs(materialization)
    return materialization, jobs, build_executor_manifest(jobs, materialization)


def _receipt(state: Path, job_id: str) -> dict:
    return json.loads((state / "receipts" / f"{job_id}.json").read_text())


def _provenance(job: dict) -> dict:
    requested = job["backend"]
    return {
        "run_id": "fixture-run",
        "artifact": job["output"],
        "git_sha": job["source_revision"],
        "sim_backend_requested": requested,
        "sim_backend": "cupy" if requested == "cupy" else "numpy",
    }


def test_builds_exact_three_lane_manifest_without_selecting_seeds(exact) -> None:
    materialization, jobs, plan = exact

    assert plan["job_count"] == len(jobs) == materialization["sealed_expanded_job_count"]
    assert {job["resource_lane"] for job in plan["jobs"]} == {
        "local_cpu", "local_gpu", "mini_pc_cluster",
    }
    assert plan["seed_selection_performed"] is False
    assert plan["held_out_partitions_accessed"] == []
    gpu = next(job for job in plan["jobs"] if job["resource_lane"] == "local_gpu")
    assert gpu["queue_command"].startswith("bash tools/queue_add.sh gpu ")
    assert plan["gpu_lease_path"] == "/tmp/sim-local-model-gpu0.lock"


@pytest.mark.parametrize("mutation,match", [
    ("drop", "extra or missing"),
    ("command", "digest-bound wrapper"),
    ("queue", "queue command"),
    ("lane", "resource lane"),
    ("heldout", "held-out"),
])
def test_fails_closed_on_any_job_set_widening(mutation: str, match: str) -> None:
    materialization = _materialization()
    jobs = _jobs(materialization)
    if mutation == "drop":
        jobs.pop()
    elif mutation == "command":
        jobs[0]["command"] += " --extra"
    elif mutation == "queue":
        jobs[3]["enqueue_command"] += " --extra"
    elif mutation == "lane":
        jobs[0]["lane"] = "unknown"
    else:
        jobs[0]["partition"] = jobs[0]["execution_contract"]["partition"] = "held_out"
        identity = {key: jobs[0]["execution_contract"][key] for key in executor.IDENTITY_FIELDS[1:]}
        jobs[0]["job_id"] = jobs[0]["execution_contract"]["job_id"] = _digest(identity)[:20]
    with pytest.raises(ExecutorError, match=match):
        build_executor_manifest(jobs, materialization)


def test_ambiguous_candidate_binding_fails_closed() -> None:
    materialization = _materialization()
    duplicate = dict(materialization["materializations"][0])
    duplicate["candidate_id"] = "second-candidate"
    duplicate["materialization_id"] = _digest(duplicate)[:24]
    materialization["materializations"].append(duplicate)
    materialization["materialization_count"] += 1
    materialization["sha256"] = _digest({k: v for k, v in materialization.items() if k != "sha256"})

    with pytest.raises(ExecutorError, match="exactly one candidate/arm cell"):
        build_executor_manifest(_jobs(materialization), materialization)


def test_digest_binds_multiple_candidates_on_same_backend_partition() -> None:
    materialization, jobs = _adaptive_same_pair()

    plan = build_executor_manifest(jobs, materialization)

    assert plan["job_count"] == 6
    assert {job["candidate_id"] for job in plan["jobs"]} == {"candidate-alpha", "candidate-beta"}
    assert {job["materialization_id"] for job in plan["jobs"]} == {
        cell["materialization_id"] for cell in materialization["materializations"]
    }
    for job in plan["jobs"]:
        command = shlex.split(job["command"])
        assert "--adaptive-parameter-document" in command
        encoded = command[command.index("--adaptive-parameter-document") + 1]
        assert json.loads(encoded) == job["parameter_document"]


def test_adaptive_success_requires_exact_runner_echo(tmp_path: Path) -> None:
    materialization, jobs = _adaptive_same_pair()
    plan = build_executor_manifest(jobs, materialization)
    state = initialize_state(plan, tmp_path / "state", owner_root=tmp_path)
    job = plan["jobs"][0]
    claim = claim_job(plan, state, job["job_id"], worker_id="adaptive-worker")
    output = tmp_path / job["output"]
    output.parent.mkdir(parents=True)
    output.write_text(json.dumps({"adaptive_candidate": {
        "candidate_id": job["candidate_id"],
        "candidate_sha256": job["candidate_sha256"],
        "effective_parameters": job["parameter_document"]["effective_parameters"],
    }}), encoding="utf-8")
    Path(str(output) + ".prov.json").write_text(json.dumps(_provenance(job)), encoding="utf-8")

    receipt = finish_job(
        plan, state, job["job_id"], claim["claim_token"], status="succeeded", exit_code=0, root=tmp_path,
    )

    assert receipt["result"]["candidate_sha256"] == job["candidate_sha256"]


def test_adaptive_success_rejects_wrong_effective_parameters(tmp_path: Path) -> None:
    materialization, jobs = _adaptive_same_pair()
    plan = build_executor_manifest(jobs, materialization)
    state = initialize_state(plan, tmp_path / "state", owner_root=tmp_path)
    job = plan["jobs"][0]
    claim = claim_job(plan, state, job["job_id"], worker_id="adaptive-worker")
    output = tmp_path / job["output"]
    output.parent.mkdir(parents=True)
    output.write_text(json.dumps({"adaptive_candidate": {
        "candidate_id": job["candidate_id"],
        "candidate_sha256": job["candidate_sha256"],
        "effective_parameters": {"gain": 999},
    }}), encoding="utf-8")
    Path(str(output) + ".prov.json").write_text(json.dumps(_provenance(job)), encoding="utf-8")

    with pytest.raises(ExecutorError, match="does not echo"):
        finish_job(
            plan, state, job["job_id"], claim["claim_token"], status="succeeded", exit_code=0,
            root=tmp_path,
        )


def test_durable_claim_success_and_provenance_verification(exact, tmp_path: Path) -> None:
    _, _, plan = exact
    state = initialize_state(plan, tmp_path / "owned/state", owner_root=tmp_path / "owned")
    job = next(item for item in plan["jobs"] if item["resource_lane"] == "local_cpu")
    claim = claim_job(plan, state, job["job_id"], worker_id="worker-1", now=100)
    assert _receipt(state, job["job_id"])["status"] == "running"
    with pytest.raises(ExecutorError, match="missing exact output"):
        finish_job(plan, state, job["job_id"], claim["claim_token"], status="succeeded",
                   exit_code=0, root=tmp_path)

    output = tmp_path / job["output"]
    output.parent.mkdir(parents=True)
    output.write_text("{}", encoding="utf-8")
    Path(str(output) + ".prov.json").write_text(json.dumps(_provenance(job)), encoding="utf-8")
    receipt = finish_job(plan, state, job["job_id"], claim["claim_token"], status="succeeded",
                         exit_code=0, root=tmp_path)
    assert receipt["status"] == "succeeded"
    assert receipt["result"]["output_sha256"] == hashlib.sha256(b"{}").hexdigest()
    assert receipt["result"]["provenance_run_id"] == "fixture-run"


@pytest.mark.parametrize("mutation,match", [
    ("artifact", "artifact does not match"),
    ("revision", "source revision does not match"),
    ("backend", "backend does not match"),
    ("run_id", "no run identity"),
])
def test_success_rejects_semantically_false_provenance(exact, tmp_path: Path, mutation: str, match: str) -> None:
    _, _, plan = exact
    state = initialize_state(plan, tmp_path / "state", owner_root=tmp_path)
    job = next(item for item in plan["jobs"] if item["resource_lane"] == "local_cpu")
    claim = claim_job(plan, state, job["job_id"], worker_id="worker")
    output = tmp_path / job["output"]
    output.parent.mkdir(parents=True)
    output.write_text("{}", encoding="utf-8")
    sidecar = _provenance(job)
    replacements = {
        "artifact": ("artifact", "research/findings/raw/wrong.json"),
        "revision": ("git_sha", "b" * 40),
        "backend": ("sim_backend", "cupy"),
        "run_id": ("run_id", ""),
    }
    key, value = replacements[mutation]
    sidecar[key] = value
    Path(str(output) + ".prov.json").write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(ExecutorError, match=match):
        finish_job(plan, state, job["job_id"], claim["claim_token"], status="succeeded",
                   exit_code=0, root=tmp_path)


def test_recovery_only_requeues_stale_dead_running_claims(exact, tmp_path: Path, monkeypatch) -> None:
    _, _, plan = exact
    state = initialize_state(plan, tmp_path / "state", owner_root=tmp_path)
    cpu = next(job for job in plan["jobs"] if job["resource_lane"] == "local_cpu")
    gpu = next(job for job in plan["jobs"] if job["resource_lane"] == "local_gpu")
    monkeypatch.setattr(executor.socket, "gethostname", lambda: "claim-host")
    claim_job(plan, state, cpu["job_id"], worker_id="cpu", now=100)
    gpu_claim = claim_job(plan, state, gpu["job_id"], worker_id="gpu", now=100)
    finish_job(plan, state, gpu["job_id"], gpu_claim["claim_token"], status="queued", exit_code=0, now=101)

    assert recover_stale(plan, state, now=159) == []
    monkeypatch.setattr(executor.socket, "gethostname", lambda: "recovery-host")
    assert recover_stale(plan, state, now=161) == [cpu["job_id"]]
    assert _receipt(state, cpu["job_id"])["status"] == "pending"
    assert _receipt(state, gpu["job_id"])["status"] == "queued"


def test_recovery_preserves_stale_claim_while_local_worker_is_alive(exact, tmp_path: Path) -> None:
    _, _, plan = exact
    state = initialize_state(plan, tmp_path / "state", owner_root=tmp_path)
    cpu = next(job for job in plan["jobs"] if job["resource_lane"] == "local_cpu")
    claim_job(plan, state, cpu["job_id"], worker_id="cpu", now=100)

    assert recover_stale(plan, state, now=1000) == []
    assert _receipt(state, cpu["job_id"])["status"] == "running"


def test_dry_run_returns_exact_command_without_state_or_subprocess(exact, tmp_path: Path, monkeypatch) -> None:
    _, _, plan = exact
    job = next(item for item in plan["jobs"] if item["resource_lane"] == "mini_pc_cluster")
    monkeypatch.setattr(executor.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("subprocess ran"))

    result = execute_claimed(plan, tmp_path / "absent", job["job_id"], worker_id="dry",
                             root=tmp_path, dry_run=True)

    assert result == {"job_id": job["job_id"], "resource_lane": "mini_pc_cluster",
                      "command": job["queue_command"], "performed": False}
    assert not (tmp_path / "absent").exists()


def test_queue_execution_records_durable_queued_receipt(exact, tmp_path: Path, monkeypatch) -> None:
    _, _, plan = exact
    state = initialize_state(plan, tmp_path / "state", owner_root=tmp_path)
    gpu = next(job for job in plan["jobs"] if job["resource_lane"] == "local_gpu")
    observed = []

    class Completed:
        returncode = 0

    monkeypatch.setattr(executor.subprocess, "run", lambda command, **kwargs: observed.append(command) or Completed())
    receipt = execute_claimed(plan, state, gpu["job_id"], worker_id="gpu-dispatch", root=tmp_path)

    assert observed == [shlex.split(gpu["queue_command"])]
    assert receipt["status"] == "queued"
    assert receipt["attempt"] == 1


def test_success_rejects_symlinked_provenance(exact, tmp_path: Path) -> None:
    _, _, plan = exact
    state = initialize_state(plan, tmp_path / "state", owner_root=tmp_path)
    job = next(item for item in plan["jobs"] if item["resource_lane"] == "local_cpu")
    claim = claim_job(plan, state, job["job_id"], worker_id="worker")
    output = tmp_path / job["output"]
    output.parent.mkdir(parents=True)
    output.write_text("{}", encoding="utf-8")
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    Path(str(output) + ".prov.json").symlink_to(target)

    with pytest.raises(ExecutorError, match="missing exact provenance"):
        finish_job(plan, state, job["job_id"], claim["claim_token"], status="succeeded",
                   exit_code=0, root=tmp_path)
