"""Tests for compiling durable executor receipts into adaptive observations."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import shlex

import pytest

import tools.experiment_observation as observation
from tools.experiment_executor import (
    build_executor_manifest,
    claim_job,
    finish_job,
    initialize_state,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="ascii")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_digest(value: dict) -> dict:
    body = {key: item for key, item in value.items() if key != "sha256"}
    return {**body, "sha256": _digest(body)}


def _design(*, partition: str = "calibration") -> dict:
    return {
        "schema": "sim-adaptive-experiment-v1",
        "id": "observation-compiler-fixture",
        "experiment": {"spec_path": "research/specs/observation-fixture.json"},
        "parameter_space": {
            "gain": {"type": "continuous", "low": 0.1, "high": 1.0},
            "fan_in": {"type": "discrete", "values": [8, 16, 32]},
        },
        "constraints": [{
            "id": "bounded-effective-drive",
            "source": "fixture biological bound",
            "predicate": {
                "op": "le",
                "left": {
                    "op": "mul",
                    "args": [{"param": "gain"}, {"param": "fan_in"}],
                },
                "right": {"value": 10.0},
            },
        }],
        "objectives": [
            {"name": "candidate_loss", "category": "physiology", "direction": "minimize",
             "weight": 1.0, "range": [0.0, 100.0]},
            {"name": "control_loss", "category": "robustness", "direction": "minimize",
             "weight": 1.0, "range": [0.0, 100.0]},
            {"name": "lesion_loss", "category": "behavior", "direction": "minimize",
             "weight": 1.0, "range": [0.0, 100.0]},
            {"name": "seconds", "category": "compute", "direction": "minimize",
             "weight": 1.0, "range": [0.0, 100.0]},
            {"name": "scaffold_count", "category": "scaffold_penalty",
             "direction": "minimize", "weight": 1.0, "range": [0.0, 100.0]},
        ],
        "fidelity_tiers": [
            {"name": "screen", "kind": "cpu_screen", "backend": "numpy",
             "partition": partition, "cost": 1.0},
            {"name": "gpu", "kind": "gpu", "backend": "cupy",
             "partition": "gpu_calibration", "cost": 5.0},
            {"name": "confirm", "kind": "replication", "backend": "cupy",
             "partition": "replication", "cost": 10.0},
        ],
        "observations": [],
        "policy": {
            "seed": 17,
            "batch_size": 2,
            "candidate_pool_size": 16,
            "initial_design_size": 2,
            "min_surrogate_observations": 3,
            "exploration_weight": 0.2,
            "promotion_slots": 1,
            "promotion_quantile": 0.5,
            "min_completed_for_promotion": 2,
            "max_completed_observations": 20,
            "plateau_window": 6,
            "min_improvement": 0.001,
            "min_feasible_fraction": 0.01,
            "research_after_observations": 4,
            "max_model_uncertainty": 0.01,
            "stop_on_replicated_targets": True,
        },
    }


def _materialization(parameters: dict, *, partition: str, job_count: int) -> dict:
    source = {"kind": "git", "revision": "a" * 40}
    arms = [
        {"name": "candidate", "role": "treatment", "parameters": {}},
        {"name": "control", "role": "control", "parameters": {"enabled": False}},
        {"name": "lesion", "role": "lesion", "parameters": {"gain": 0.0},
         "target": "gain"},
    ]
    candidate = {
        "schema": "sim-adaptive-candidate-v1",
        "candidate_id": "candidate-alpha",
        "parameters": parameters,
    }
    candidate_sha = _digest(candidate)
    cells = []
    for arm in arms:
        cell = {
            "candidate_id": candidate["candidate_id"],
            "candidate_order": 0,
            "candidate_document": candidate,
            "candidate_sha256": candidate_sha,
            "arm": arm["name"],
            "role": arm["role"],
            "candidate_parameters": parameters,
            "arm_parameters": arm["parameters"],
            "effective_parameters": {**parameters, **arm["parameters"]},
            "backend": "numpy",
            "partition": partition,
            "resource_lane": "local_cpu",
            "device": "cpu",
        }
        if arm["role"] == "lesion":
            cell["lesion_target"] = arm["target"]
        cell["materialization_id"] = _digest(cell)[:24]
        cells.append(cell)
    manifest = {
        "schema": "sim-experiment-controller-execution-manifest-v1",
        "plan_sha256": "1" * 64,
        "handoff_sha256": "2" * 64,
        "experiment_id": "observation-compiler-fixture",
        "spec_path": "research/specs/observation-fixture.json",
        "spec_sha256": "3" * 64,
        "sealed_execution_manifest_sha256": "4" * 64,
        "sealed_expanded_job_count": job_count,
        "source": source,
        "arms": arms,
        "required_roles": {
            "treatment": ["candidate"], "control": ["control"], "lesion": ["lesion"],
        },
        "backend_partition_pairs": [{"backend": "numpy", "partition": partition}],
        "materialization_count": len(cells),
        "materializations": cells,
        "receipt_contract": {
            "schema": "sim-experiment-controller-execution-receipt-v1",
            "required_materialization_ids": [cell["materialization_id"] for cell in cells],
            "status": "accepted_non_dispatching",
            "exact_set_required": True,
        },
        "dispatch": {"performed": False, "commands_emitted": False},
        "seeds_selected": False,
        "held_out_partitions_accessed": [],
    }
    return _self_digest(manifest)


def _job(materialization: dict, cell: dict, seed: int) -> dict:
    output = (
        "research/findings/raw/observation-fixture/"
        f"{cell['arm']}-{seed}.json"
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
        "experiment_id": materialization["experiment_id"],
        "spec_sha256": materialization["spec_sha256"],
        "execution_manifest_sha256": materialization["sealed_execution_manifest_sha256"],
        "corpus_check_sha256": "5" * 64,
        "source": materialization["source"],
        "partition": cell["partition"],
        "backend": cell["backend"],
        "device": cell["device"],
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
        "--seed", str(seed), "--phase", cell["partition"], "--arm", cell["arm"],
        "--out", output, "--adaptive-parameter-document",
        _canonical(parameter_document).decode("ascii"),
    ]
    contract = {
        "schema": "sim-experiment-job-contract-v1",
        "job_id": job_id,
        **identity,
        "execution_snapshot": {
            "manifest_sha256": materialization["sealed_execution_manifest_sha256"],
        },
        "runner_command": runner,
        "environment": {"SIM_BACKEND": "numpy"},
        "claim_stale_seconds": 60,
    }
    encoded = base64.urlsafe_b64encode(_canonical(contract)).decode("ascii")
    command = shlex.join([
        ".venv/bin/python", "tools/experiment.py", "execute-job", "--contract", encoded,
        "--", *runner,
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


def _provenance(job: dict) -> dict:
    return {
        "run_id": f"fixture-{job['arm']}-{job['seed']}",
        "artifact": job["output"],
        "git_sha": job["source_revision"],
        "sim_backend_requested": "numpy",
        "sim_backend": "numpy",
    }


def _compile(*args, **kwargs):
    compiler = getattr(observation, "compile_executor_receipts_to_observations", None)
    if compiler is None:
        pytest.fail("production executor-receipt observation compiler is not implemented")
    return compiler(*args, **kwargs)


def _fixture(
    tmp_path: Path,
    *,
    parameters: dict | None = None,
    seeds: tuple[int, ...] = (11, 12),
    held_out_seeds: tuple[int, ...] = (99,),
    partition: str = "calibration",
) -> dict:
    root = tmp_path / "repo"
    root.mkdir(parents=True)
    parameters = parameters or {"gain": 0.4, "fan_in": 16}
    materialization = _materialization(
        parameters, partition=partition, job_count=3 * len(seeds),
    )
    jobs = [
        _job(materialization, cell, seed)
        for cell in materialization["materializations"]
        for seed in seeds
    ]
    executor_manifest = build_executor_manifest(jobs, materialization)
    sealed_seeds = {job["job_id"]: job["seed"] for job in jobs}
    for authorized_job in executor_manifest["jobs"]:
        authorized_job["seed"] = sealed_seeds[authorized_job["job_id"]]
    executor_manifest = _self_digest(executor_manifest)
    state = initialize_state(executor_manifest, root / "state", owner_root=root)

    metric_bases = {"candidate": 1.0, "control": 10.0, "lesion": 20.0}
    for job in executor_manifest["jobs"]:
        claim = claim_job(
            executor_manifest, state, job["job_id"], worker_id="fixture", now=100.0,
        )
        output = root / job["output"]
        output.parent.mkdir(parents=True, exist_ok=True)
        seed_offset = 2.0 * seeds.index(job["seed"])
        _write_json(output, {
            "adaptive_candidate": {
                "candidate_id": job["candidate_id"],
                "candidate_sha256": job["candidate_sha256"],
                "effective_parameters": job["parameter_document"]["effective_parameters"],
            },
            "metrics": {
                "loss": metric_bases[job["arm"]] + seed_offset,
                "seconds": 4.0 + seed_offset,
                "scaffold_count": seed_offset / 2.0,
            },
        })
        _write_json(Path(str(output) + ".prov.json"), _provenance(job))
        finish_job(
            executor_manifest,
            state,
            job["job_id"],
            claim["claim_token"],
            status="succeeded",
            exit_code=0,
            root=root,
            now=200.0,
        )

    partitions = {
        "calibration": list(seeds) if partition == "calibration" else [1, 2],
        "gpu_calibration": [31, 32],
        "replication": [41, 42],
        "held_out": [901, 902],
    }
    if partition != "calibration":
        partitions[partition] = list(seeds)
    _write_json(root / "research/specs/observation-fixture.json", {
        "schema": "sim-experiment-spec-v0",
        "id": "observation-compiler-fixture",
        "partitions": partitions,
        "backends": ["numpy", "cupy"],
    })
    design_path = root / "design.json"
    design_sha = _write_json(design_path, _design(partition=partition))
    executor_path = state / "manifest.json"
    executor_sha = hashlib.sha256(executor_path.read_bytes()).hexdigest()
    contract = _self_digest({
        "schema": "sim-observation-contract-v1",
        "id": "observation-compiler-contract",
        "status": "preregistered",
        "bindings": {
            "adaptive_design": {"path": "design.json", "sha256": design_sha},
            "executor_manifest": {"path": "state/manifest.json", "sha256": executor_sha},
        },
        "objectives": {
            "candidate_loss": {"arm": "candidate", "path": ["metrics", "loss"],
                               "reducer": "mean"},
            "control_loss": {"arm": "control", "path": ["metrics", "loss"],
                             "reducer": "mean"},
            "lesion_loss": {"arm": "lesion", "path": ["metrics", "loss"],
                            "reducer": "mean"},
            "seconds": {"arm": "candidate", "path": ["metrics", "seconds"],
                        "reducer": "mean"},
            "scaffold_count": {"arm": "candidate", "path": ["metrics", "scaffold_count"],
                               "reducer": "mean"},
        },
        "fidelity_mapping": [{
            "backend": "numpy", "partition": partition, "fidelity": "screen",
        }],
        "required_seeds": list(seeds),
        "held_out_seeds": list(held_out_seeds),
        "output_path": "compiled.json",
    })
    contract_path = root / "contract.json"
    _write_json(contract_path, contract)
    receipts = sorted((state / "receipts").glob("*.json"))
    return {
        "root": root,
        "contract": contract_path,
        "executor_manifest": executor_path,
        "receipts": receipts,
        "output": root / "compiled.json",
        "plan": executor_manifest,
    }


def _run(fixture: dict) -> dict:
    return _compile(
        fixture["contract"],
        fixture["executor_manifest"],
        fixture["receipts"],
        fixture["output"],
        repository_root=fixture["root"],
    )


def test_groups_required_arms_and_seeds_into_one_adaptive_observation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    result = _run(fixture)

    assert len(result["observations"]) == 1
    compiled = result["observations"][0]
    assert compiled["status"] == "complete"
    assert compiled["parameters"] == {"gain": 0.4, "fan_in": 16}
    assert compiled["fidelity"] == "screen"
    assert compiled["partition"] == "calibration"
    assert compiled["objectives"] == {
        "candidate_loss": 2.0,
        "control_loss": 11.0,
        "lesion_loss": 21.0,
        "seconds": 5.0,
        "scaffold_count": 0.5,
    }
    assert result["blocked"] == []
    assert result["scientific_verdict"] is None


@pytest.mark.parametrize("target", ["output", "receipt"])
def test_rejects_tampered_authenticated_evidence(tmp_path: Path, target: str) -> None:
    fixture = _fixture(tmp_path)
    if target == "output":
        job = fixture["plan"]["jobs"][0]
        output = fixture["root"] / job["output"]
        value = json.loads(output.read_text())
        value["metrics"]["loss"] = 999.0
        _write_json(output, value)
    else:
        receipt = fixture["receipts"][0]
        value = json.loads(receipt.read_text())
        value["status"] = "failed"
        _write_json(receipt, value)

    with pytest.raises(observation.ObservationCompilerError, match="digest|receipt|tamper"):
        _run(fixture)


@pytest.mark.parametrize("missing", ["arm", "seed"])
def test_incomplete_required_matrix_blocks_or_fails_closed(tmp_path: Path, missing: str) -> None:
    fixture = _fixture(tmp_path)
    jobs = {job["job_id"]: job for job in fixture["plan"]["jobs"]}
    if missing == "arm":
        receipts = [
            path for path in fixture["receipts"]
            if jobs[path.stem]["arm"] != "lesion"
        ]
    else:
        receipts = [
            path for path in fixture["receipts"]
            if jobs[path.stem]["seed"] != 12
        ]
    fixture["receipts"] = receipts

    try:
        result = _run(fixture)
    except observation.ObservationCompilerError:
        return
    assert result["observations"] == []
    assert result["blocked"]
    assert any("missing" in row["reason"] or "incomplete" in row["reason"]
               for row in result["blocked"])


@pytest.mark.parametrize("eligibility", ["heldout", "engineering"])
def test_heldout_and_engineering_evidence_cannot_be_observations(
    tmp_path: Path, eligibility: str,
) -> None:
    fixture = (
        _fixture(tmp_path, seeds=(99,), held_out_seeds=(99,))
        if eligibility == "heldout"
        else _fixture(tmp_path, partition="engineering")
    )

    try:
        result = _run(fixture)
    except observation.ObservationCompilerError:
        return
    assert result["observations"] == []
    assert result["blocked"]
    assert all(eligibility in row["reason"].lower() for row in result["blocked"])


def test_candidate_violating_adaptive_hard_constraint_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, parameters={"gain": 0.8, "fan_in": 16})

    try:
        result = _run(fixture)
    except observation.ObservationCompilerError:
        return
    assert result["observations"] == []
    assert result["blocked"]
    assert any("constraint" in row["reason"].lower() for row in result["blocked"])


def test_output_is_deterministic_self_digested_and_create_only(tmp_path: Path) -> None:
    first_fixture = _fixture(tmp_path / "first")
    second_fixture = _fixture(tmp_path / "second")

    first = _run(first_fixture)
    second = _run(second_fixture)

    assert first == second
    assert first["sha256"] == _digest({
        key: value for key, value in first.items() if key != "sha256"
    })
    original = first_fixture["output"].read_bytes()
    with pytest.raises(observation.ObservationCompilerError, match="replace|exists|create"):
        _run(first_fixture)
    assert first_fixture["output"].read_bytes() == original
