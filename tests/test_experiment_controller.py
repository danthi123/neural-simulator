"""Focused dry-run tests for the adaptive experiment controller."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import pytest

import tools.experiment_controller as controller
from tools.experiment import create_experiment_seal, expand_experiment_jobs
from tools.experiment_controller import (
    ControllerError,
    EXECUTION_MANIFEST_SCHEMA,
    EXECUTION_RECEIPT_SCHEMA,
    SCHEMA,
    build_dry_run_plan,
    materialize_candidate_spec,
    materialize_execution_manifest,
    validate_experiment_handoff,
    validate_execution_receipt,
    write_dry_run_plan,
)
from tools.experiment_executor import build_executor_manifest


def _spec() -> dict:
    return {
        "schema": "sim-experiment-spec-v0",
        "id": "controller-fixture",
        "partitions": {"calibration": [11], "replication": [21], "held_out": [99]},
        "backends": ["numpy", "cupy"],
        "execution": {
            "command": [
                ".venv/bin/python", "-m", "research.runners.fixture",
                "--seed", "{seed}", "--phase", "{partition}", "--arm", "{arm}", "--out", "{output}",
            ],
            "output": "research/findings/raw/controller/{partition}/{backend}/{arm}-{seed}.json",
            "arms": {
                "candidate": {"role": "treatment", "parameters": {}},
                "matched-control": {"role": "control", "parameters": {"adaptive_enabled": False}},
                "gain-lesion": {
                    "role": "lesion", "target": "adaptive_gain", "parameters": {"adaptive_gain": 0.0},
                },
            },
            "targets": {
                "numpy": {"device": "cpu", "lane": "local"},
                "cupy": {"device": "cuda:0", "lane": "gpu"},
            },
            "corpus_check": {
                "path": "research/specs/corpus-check.json",
                "sha256": "pending",
                "query": "Has this controller fixture already been run?",
                "max_age_seconds": 3600,
            },
            "claim_stale_seconds": 60,
        },
    }


def _design() -> dict:
    return {
        "schema": "sim-adaptive-experiment-v1",
        "id": "controller-design",
        "experiment": {"spec_path": "research/specs/experiment.json"},
        "parameter_space": {"gain": {"type": "continuous", "low": 0.1, "high": 1.0, "transform": "linear"}},
        "constraints": [],
        "objectives": [
            {"name": "score", "category": "behavior", "direction": "maximize",
             "weight": 1, "range": [0, 1]},
            {"name": "cost", "category": "compute", "direction": "minimize",
             "weight": 1, "range": [0, 10]},
            {"name": "rate_error", "category": "physiology", "direction": "minimize",
             "weight": 1, "range": [0, 10]},
            {"name": "stability", "category": "robustness", "direction": "maximize",
             "weight": 1, "range": [0, 1]},
            {"name": "scaffolds", "category": "scaffold_penalty", "direction": "minimize",
             "weight": 1, "range": [0, 10]},
        ],
        "fidelity_tiers": [
            {"name": "cpu", "kind": "cpu_screen", "backend": "numpy", "partition": "calibration", "cost": 1},
            {"name": "gpu", "kind": "gpu", "backend": "cupy", "partition": "calibration", "cost": 2},
            {"name": "confirm", "kind": "replication", "backend": "cupy", "partition": "replication", "cost": 3},
        ],
        "observations": [],
        "policy": {
            "seed": 17, "batch_size": 2, "candidate_pool_size": 16, "initial_design_size": 3,
            "min_surrogate_observations": 3, "exploration_weight": 0.2, "promotion_slots": 1,
            "promotion_quantile": 0.5, "min_completed_for_promotion": 3,
            "max_completed_observations": 10, "plateau_window": 20, "min_improvement": 0.001,
            "min_feasible_fraction": 0.01, "research_after_observations": 6,
            "max_model_uncertainty": 0.01, "stop_on_replicated_targets": True,
        },
    }


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    spec_path = root / "research/specs/experiment.json"
    design_path = root / "research/specs/adaptive.json"
    spec_path.parent.mkdir(parents=True)
    spec = _spec()
    corpus_path = root / spec["execution"]["corpus_check"]["path"]
    corpus_path.write_text(json.dumps({
        "schema": "sim-corpus-check-v1",
        "experiment_id": spec["id"],
        "query": spec["execution"]["corpus_check"]["query"],
        "status": "success",
        "completed_at": time.time(),
        "rag": {"status": "success", "index_digest": "fixture-index"},
    }, sort_keys=True), encoding="utf-8")
    spec["execution"]["corpus_check"]["sha256"] = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    design_path.write_text(json.dumps(_design()), encoding="utf-8")
    manifest = root / ".source_manifest.sha256"
    manifest.write_text("", encoding="utf-8")
    manifest_digest = hashlib.sha256(b"").hexdigest()
    (root / ".source_revision").write_text(
        f"git_sha=controller-fixture\nsource_kind=git_archive\nsource_manifest_sha256={manifest_digest}\n",
        encoding="utf-8",
    )
    return root


def _plan_with_gpu_candidate(fixture_repo: Path) -> dict:
    plan = build_dry_run_plan(fixture_repo / "research/specs/adaptive.json", root=fixture_repo)
    candidate = dict(plan["candidate_materialization"]["candidates"][0])
    candidate.update({
        "candidate_id": "fixture-cupy-candidate",
        "order": len(plan["candidate_materialization"]["candidates"]),
        "backend": "cupy",
        "resource_lane": "local_gpu",
        "device": "cuda:0",
        "reason": "fixture coverage for the declared GPU pair",
    })
    plan["candidate_materialization"]["candidates"].append(candidate)
    plan["candidate_materialization"]["count"] = len(plan["candidate_materialization"]["candidates"])
    plan["sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in plan.items() if key != "sha256"},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return plan


def _sealed_handoff(fixture_repo: Path, tmp_path: Path) -> tuple[dict, dict]:
    plan = _plan_with_gpu_candidate(fixture_repo)
    materialized = materialize_candidate_spec(
        plan, fixture_repo / "research/specs/materialized.json", root=fixture_repo,
    )
    seal_path = tmp_path / "controller.seal.json"
    create_experiment_seal(materialized, seal_path, root=fixture_repo)
    return plan, validate_experiment_handoff(
        plan, seal_path=seal_path, materialized_spec_path=materialized, root=fixture_repo,
    )


def _receipt(manifest: dict) -> dict:
    receipt = {
        "schema": EXECUTION_RECEIPT_SCHEMA,
        "execution_manifest_sha256": manifest["sha256"],
        "materializations": [{
            "materialization_id": cell["materialization_id"],
            "candidate_id": cell["candidate_id"],
            "arm": cell["arm"],
            "role": cell["role"],
            "backend": cell["backend"],
            "partition": cell["partition"],
            "status": "accepted_non_dispatching",
        } for cell in manifest["materializations"]],
        "dispatch": {"performed": False, "commands_emitted": False},
        "seeds_selected": False,
        "held_out_partitions_accessed": [],
    }
    receipt["sha256"] = hashlib.sha256(json.dumps(
        receipt, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return receipt


def test_dry_run_is_deterministic_and_describes_lanes_without_dispatch(fixture_repo: Path) -> None:
    design = fixture_repo / "research/specs/adaptive.json"
    first = build_dry_run_plan(design, root=fixture_repo)
    second = build_dry_run_plan(design, root=fixture_repo)

    assert first == second
    assert first["schema"] == SCHEMA
    assert first["dispatch"]["performed"] is False
    assert first["candidate_materialization"]["seeds_selected"] is False
    assert first["candidate_materialization"]["held_out_partitions_accessed"] == []
    assert {item["resource_lane"] for item in first["candidate_materialization"]["candidates"]} == {"local_cpu"}
    assert all(item["partition"] == "calibration" for item in first["candidate_materialization"]["candidates"])
    assert not list(fixture_repo.rglob("*.claim"))


def test_missing_execution_target_fails_closed(fixture_repo: Path) -> None:
    spec_path = fixture_repo / "research/specs/experiment.json"
    spec = json.loads(spec_path.read_text())
    del spec["execution"]["targets"]["numpy"]
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(ControllerError, match="no execution target"):
        build_dry_run_plan(fixture_repo / "research/specs/adaptive.json", root=fixture_repo)


def test_writer_requires_explicit_owner_root_and_refuses_overwrite(fixture_repo: Path, tmp_path: Path) -> None:
    plan = build_dry_run_plan(fixture_repo / "research/specs/adaptive.json", root=fixture_repo)
    destination = tmp_path / "owned" / "plan.json"
    with pytest.raises(ControllerError, match="outside"):
        write_dry_run_plan(plan, destination, owner_root=tmp_path / "different")

    assert write_dry_run_plan(plan, destination, owner_root=tmp_path / "owned") == destination
    with pytest.raises(ControllerError, match="refusing to replace"):
        write_dry_run_plan(plan, destination, owner_root=tmp_path / "owned")


def test_controller_rejects_design_outside_repository(fixture_repo: Path, tmp_path: Path) -> None:
    outside = tmp_path / "adaptive.json"
    outside.write_text(json.dumps(_design()), encoding="utf-8")

    with pytest.raises(ControllerError, match="inside the repository"):
        build_dry_run_plan(outside, root=fixture_repo)


def test_valid_plan_maps_to_existing_sealed_expansion_contract(fixture_repo: Path, tmp_path: Path) -> None:
    plan = _plan_with_gpu_candidate(fixture_repo)
    materialized = materialize_candidate_spec(
        plan, fixture_repo / "research/specs/materialized.json", root=fixture_repo,
    )
    seal_path = tmp_path / "controller.seal.json"
    create_experiment_seal(materialized, seal_path, root=fixture_repo)

    handoff = validate_experiment_handoff(
        plan, seal_path=seal_path, materialized_spec_path=materialized, root=fixture_repo,
    )

    assert handoff["schema"] == "sim-experiment-controller-handoff-v1"
    assert handoff["experiment_id"] == "controller-fixture"
    assert handoff["sealed"] is True
    assert handoff["backend_partition_pairs"] == [
        {"backend": "cupy", "partition": "calibration"},
        {"backend": "numpy", "partition": "calibration"},
    ]
    assert handoff["expanded_job_count"] == 9
    assert handoff["adaptive_expanded_job_count"] == 9
    assert [arm["role"] for arm in handoff["arms"]] == ["treatment", "lesion", "control"]
    assert len(handoff["sha256"]) == 64
    assert handoff["seeds_selected"] is False
    assert handoff["held_out_partitions_accessed"] == []
    assert not list(fixture_repo.rglob("*.claim"))
    assert not (fixture_repo / "research/findings/raw/controller").exists()


def test_expansion_cannot_widen_candidate_backend_set(fixture_repo: Path, tmp_path: Path) -> None:
    plan = build_dry_run_plan(fixture_repo / "research/specs/adaptive.json", root=fixture_repo)
    materialized = materialize_candidate_spec(
        plan, fixture_repo / "research/specs/materialized.json", root=fixture_repo,
    )
    materialized.chmod(0o644)
    spec = json.loads(materialized.read_text())
    spec["execution"]["candidates"]["unexpected"] = {
        "order": len(spec["execution"]["candidates"]),
        "parameters": {"gain": 0.9},
        "backend": "cupy",
        "partition": "calibration",
    }
    materialized.write_text(json.dumps(spec), encoding="utf-8")
    seal_path = tmp_path / "controller.seal.json"
    create_experiment_seal(materialized, seal_path, root=fixture_repo)

    with pytest.raises(ControllerError, match="exactly bind"):
        validate_experiment_handoff(
            plan, seal_path=seal_path, materialized_spec_path=materialized, root=fixture_repo,
        )


def test_malformed_plan_fails_closed_before_expansion(fixture_repo: Path, tmp_path: Path) -> None:
    plan = build_dry_run_plan(fixture_repo / "research/specs/adaptive.json", root=fixture_repo)
    malformed = json.loads(json.dumps(plan))
    malformed["candidate_materialization"]["candidates"][0]["seed"] = 123
    malformed["sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in malformed.items() if key != "sha256"},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()

    with pytest.raises(ControllerError, match="execution or seed selection"):
        validate_experiment_handoff(malformed, seal_path=tmp_path / "missing-seal.json", root=fixture_repo)
    assert not list(fixture_repo.rglob("*.claim"))


def test_unsealed_plan_fails_closed(fixture_repo: Path) -> None:
    plan = build_dry_run_plan(fixture_repo / "research/specs/adaptive.json", root=fixture_repo)

    with pytest.raises(ControllerError, match="existing experiment seal"):
        validate_experiment_handoff(plan, root=fixture_repo)


def test_materialization_is_deterministic_exact_and_seed_free(fixture_repo: Path, tmp_path: Path) -> None:
    plan, handoff = _sealed_handoff(fixture_repo, tmp_path)

    first = materialize_execution_manifest(plan, handoff, root=fixture_repo)
    second = materialize_execution_manifest(plan, handoff, root=fixture_repo)

    assert first == second
    assert first["schema"] == EXECUTION_MANIFEST_SCHEMA
    assert first["materialization_count"] == 9
    assert first["sealed_expanded_job_count"] == 9
    assert first["required_roles"] == {
        "treatment": ["candidate"], "control": ["matched-control"], "lesion": ["gain-lesion"],
    }
    assert first["dispatch"] == {"performed": False, "commands_emitted": False}
    assert first["seeds_selected"] is False
    assert first["held_out_partitions_accessed"] == []
    encoded = json.dumps(first, sort_keys=True)
    assert '"seed"' not in encoded and "held_out\": [99]" not in encoded
    assert all("command" not in cell and "output" not in cell for cell in first["materializations"])
    assert len({
        (cell["candidate_id"], cell["candidate_sha256"])
        for cell in first["materializations"]
    }) == 3
    assert all(cell["candidate_document"]["parameters"] == cell["candidate_parameters"]
               for cell in first["materializations"])
    assert next(cell for cell in first["materializations"] if cell["arm"] == "gain-lesion")[
        "effective_parameters"
    ]["adaptive_gain"] == 0.0


def test_materialized_spec_expands_into_executor_accepted_adaptive_jobs(
    fixture_repo: Path, tmp_path: Path,
) -> None:
    plan = _plan_with_gpu_candidate(fixture_repo)
    materialized = materialize_candidate_spec(
        plan, fixture_repo / "research/specs/materialized.json", root=fixture_repo,
    )
    seal_path = tmp_path / "controller.seal.json"
    create_experiment_seal(materialized, seal_path, root=fixture_repo)
    handoff = validate_experiment_handoff(
        plan, seal_path=seal_path, materialized_spec_path=materialized, root=fixture_repo,
    )
    manifest = materialize_execution_manifest(plan, handoff, root=fixture_repo)
    jobs = expand_experiment_jobs(
        materialized, ["calibration"], seal_path=seal_path, root=fixture_repo,
    )

    executor = build_executor_manifest(jobs, manifest)

    assert executor["job_count"] == 9
    assert {job["candidate_id"] for job in executor["jobs"]} == {
        candidate["candidate_id"] for candidate in plan["candidate_materialization"]["candidates"]
    }
    assert all(job["parameter_document"] is not None for job in executor["jobs"])


def test_materialization_does_not_reopen_spec_or_expand_seed_jobs(
    fixture_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, handoff = _sealed_handoff(fixture_repo, tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("seed-bearing experiment API was called")

    monkeypatch.setattr(controller, "load_experiment_spec", forbidden)
    monkeypatch.setattr(controller, "expand_experiment_jobs", forbidden)

    manifest = materialize_execution_manifest(plan, handoff, root=fixture_repo)

    assert manifest["seeds_selected"] is False
    assert manifest["held_out_partitions_accessed"] == []


def test_exact_non_dispatching_receipt_is_accepted(fixture_repo: Path, tmp_path: Path) -> None:
    plan, handoff = _sealed_handoff(fixture_repo, tmp_path)
    manifest = materialize_execution_manifest(plan, handoff, root=fixture_repo)

    result = validate_execution_receipt(manifest, _receipt(manifest))

    assert result["accepted"] is True
    assert result["materialization_count"] == 9
    assert result["dispatch_performed"] is False
    assert result["seeds_selected"] is False


@pytest.mark.parametrize("role", ["treatment", "control", "lesion"])
def test_receipt_rejects_missing_arm_role(fixture_repo: Path, tmp_path: Path, role: str) -> None:
    plan, handoff = _sealed_handoff(fixture_repo, tmp_path)
    manifest = materialize_execution_manifest(plan, handoff, root=fixture_repo)
    receipt = _receipt(manifest)
    receipt["materializations"] = [row for row in receipt["materializations"] if row["role"] != role]
    receipt["sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in receipt.items() if key != "sha256"},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()

    with pytest.raises(ControllerError, match="extra, missing, or expanded"):
        validate_execution_receipt(manifest, receipt)


def test_receipt_rejects_backend_partition_or_arm_expansion(fixture_repo: Path, tmp_path: Path) -> None:
    plan, handoff = _sealed_handoff(fixture_repo, tmp_path)
    manifest = materialize_execution_manifest(plan, handoff, root=fixture_repo)
    receipt = _receipt(manifest)
    extra = dict(receipt["materializations"][0])
    extra.update({"materialization_id": "extra", "backend": "expanded", "partition": "replication"})
    receipt["materializations"].append(extra)
    receipt["sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in receipt.items() if key != "sha256"},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()

    with pytest.raises(ControllerError, match="extra, missing, or expanded"):
        validate_execution_receipt(manifest, receipt)


def test_materialization_rejects_unsealed_or_dirty_archive_source(fixture_repo: Path, tmp_path: Path) -> None:
    plan, handoff = _sealed_handoff(fixture_repo, tmp_path)
    unsealed = json.loads(json.dumps(handoff))
    unsealed["sealed"] = False
    unsealed["sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in unsealed.items() if key != "sha256"},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    with pytest.raises(ControllerError, match="seed-free sealed handoff"):
        materialize_execution_manifest(plan, unsealed, root=fixture_repo)

    (fixture_repo / ".source_manifest.sha256").write_text("changed", encoding="utf-8")
    with pytest.raises(ControllerError, match="source identity changed"):
        materialize_execution_manifest(plan, handoff, root=fixture_repo)


def test_handoff_rejects_missing_control_or_lesion_role(fixture_repo: Path, tmp_path: Path) -> None:
    spec_path = fixture_repo / "research/specs/experiment.json"
    spec = json.loads(spec_path.read_text())
    del spec["execution"]["arms"]["gain-lesion"]
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    plan = build_dry_run_plan(fixture_repo / "research/specs/adaptive.json", root=fixture_repo)
    seal_path = tmp_path / "controller.seal.json"
    create_experiment_seal(spec_path, seal_path, root=fixture_repo)

    with pytest.raises(ControllerError, match="missing required roles"):
        validate_experiment_handoff(plan, seal_path=seal_path, root=fixture_repo)
