"""Focused dry-run tests for the adaptive experiment controller."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.experiment_controller import (
    ControllerError,
    SCHEMA,
    build_dry_run_plan,
    write_dry_run_plan,
)


def _spec() -> dict:
    return {
        "schema": "sim-experiment-spec-v0",
        "id": "controller-fixture",
        "partitions": {"calibration": [11], "replication": [21], "held_out": [99]},
        "backends": ["numpy", "cupy"],
        "execution": {
            "targets": {
                "numpy": {"device": "cpu", "lane": "local"},
                "cupy": {"device": "cuda:0", "lane": "gpu"},
            }
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
    spec_path.write_text(json.dumps(_spec()), encoding="utf-8")
    design_path.write_text(json.dumps(_design()), encoding="utf-8")
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
