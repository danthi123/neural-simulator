"""Focused dry-run tests for the adaptive experiment controller."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import pytest

from tools.experiment import create_experiment_seal
from tools.experiment_controller import (
    ControllerError,
    SCHEMA,
    build_dry_run_plan,
    validate_experiment_handoff,
    write_dry_run_plan,
)


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
            "arms": ["default"],
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
    plan = build_dry_run_plan(fixture_repo / "research/specs/adaptive.json", root=fixture_repo)
    # The adaptive screen initially proposes only the CPU pair. Materialize a second declared candidate in
    # this fixture so the sealed expansion covers exactly the pairs named by the handoff.
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
    seal_path = tmp_path / "controller.seal.json"
    create_experiment_seal(fixture_repo / "research/specs/experiment.json", seal_path, root=fixture_repo)

    handoff = validate_experiment_handoff(plan, seal_path=seal_path, root=fixture_repo)

    assert handoff["schema"] == "sim-experiment-controller-handoff-v1"
    assert handoff["experiment_id"] == "controller-fixture"
    assert handoff["sealed"] is True
    assert handoff["backend_partition_pairs"] == [
        {"backend": "cupy", "partition": "calibration"},
        {"backend": "numpy", "partition": "calibration"},
    ]
    assert handoff["expanded_job_count"] == 2
    assert handoff["seeds_selected"] is False
    assert handoff["held_out_partitions_accessed"] == []
    assert not list(fixture_repo.rglob("*.claim"))
    assert not (fixture_repo / "research/findings/raw/controller").exists()


def test_expansion_cannot_widen_candidate_backend_set(fixture_repo: Path, tmp_path: Path) -> None:
    plan = build_dry_run_plan(fixture_repo / "research/specs/adaptive.json", root=fixture_repo)
    seal_path = tmp_path / "controller.seal.json"
    create_experiment_seal(fixture_repo / "research/specs/experiment.json", seal_path, root=fixture_repo)

    with pytest.raises(ControllerError, match="widened"):
        validate_experiment_handoff(plan, seal_path=seal_path, root=fixture_repo)


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
