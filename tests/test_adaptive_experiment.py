"""Offline tests for deterministic adaptive experiment design."""
from __future__ import annotations

import copy
import json
import os

import pytest

from tools.adaptive_experiment import (
    AdaptiveExperimentError,
    BATCH_SCHEMA,
    load_adaptive_design,
    propose_next_batch,
    write_next_batch,
)
from tools import adaptive_experiment


def _experiment_spec():
    return {
        "schema": "sim-experiment-spec-v0",
        "id": "adaptive-fixture",
        "partitions": {"calibration": [11, 12], "replication": [21, 22], "held_out": [99]},
        "backends": ["numpy", "cupy"],
    }


def _objectives():
    return [
        {"name": "rate_error", "category": "physiology", "direction": "minimize", "weight": 3,
         "range": [0, 20], "target": 4},
        {"name": "task_score", "category": "behavior", "direction": "maximize", "weight": 4,
         "range": [0, 1], "target": 0.75},
        {"name": "seed_stability", "category": "robustness", "direction": "maximize", "weight": 2,
         "range": [0, 1], "target": 0.7},
        {"name": "seconds", "category": "compute", "direction": "minimize", "weight": 1,
         "range": [0, 100]},
        {"name": "scaffold_count", "category": "scaffold_penalty", "direction": "minimize", "weight": 2,
         "range": [0, 10]},
    ]


def _design(spec_path="research/specs/adaptive.json"):
    return {
        "schema": "sim-adaptive-experiment-v1",
        "id": "mixed-biological-search",
        "experiment": {"spec_path": spec_path},
        "parameter_space": {
            "gain": {"type": "continuous", "low": 0.1, "high": 2.0, "transform": "log"},
            "fan_in": {"type": "discrete", "values": [8, 16, 32]},
            "rule": {"type": "categorical", "values": ["pair", "triplet"]},
        },
        "constraints": [
            {
                "id": "bounded-effective-drive",
                "source": "Fixture biology reference, section 2",
                "predicate": {
                    "op": "le",
                    "left": {"op": "mul", "args": [{"param": "gain"}, {"param": "fan_in"}]},
                    "right": {"value": 40},
                },
            }
        ],
        "objectives": _objectives(),
        "fidelity_tiers": [
            {"name": "cpu", "kind": "cpu_screen", "backend": "numpy", "partition": "calibration", "cost": 1},
            {"name": "gpu", "kind": "gpu", "backend": "cupy", "partition": "calibration", "cost": 5},
            {"name": "confirm", "kind": "replication", "backend": "cupy", "partition": "replication", "cost": 10},
        ],
        "observations": [],
        "policy": {
            "seed": 17,
            "batch_size": 4,
            "candidate_pool_size": 64,
            "initial_design_size": 4,
            "min_surrogate_observations": 4,
            "exploration_weight": 0.2,
            "promotion_slots": 1,
            "promotion_quantile": 0.5,
            "min_completed_for_promotion": 3,
            "max_completed_observations": 50,
            "plateau_window": 20,
            "min_improvement": 0.001,
            "min_feasible_fraction": 0.01,
            "research_after_observations": 6,
            "max_model_uncertainty": 0.01,
            "stop_on_replicated_targets": True,
        },
    }


@pytest.fixture
def adaptive_repo(tmp_path):
    root = tmp_path / "repo"
    path = root / "research/specs/adaptive.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_experiment_spec()), encoding="utf-8")
    return root


def _metrics(score, *, rate=3, stability=0.8, seconds=20, scaffolds=1):
    return {"rate_error": rate, "task_score": score, "seed_stability": stability,
            "seconds": seconds, "scaffold_count": scaffolds}


def _observation(identifier, parameters, fidelity="cpu", partition="calibration", score=0.8, **metrics):
    return {
        "id": identifier,
        "status": "complete",
        "parameters": parameters,
        "fidelity": fidelity,
        "partition": partition,
        "objectives": _metrics(score, **metrics),
    }


def _screen_points():
    return [
        {"gain": 0.12, "fan_in": 8, "rule": "pair"},
        {"gain": 0.2, "fan_in": 16, "rule": "triplet"},
        {"gain": 0.35, "fan_in": 32, "rule": "pair"},
        {"gain": 0.6, "fan_in": 16, "rule": "pair"},
        {"gain": 0.9, "fan_in": 32, "rule": "triplet"},
        {"gain": 1.4, "fan_in": 16, "rule": "triplet"},
    ]


def test_initial_batch_is_deterministic_constrained_and_nonexecuting(adaptive_repo):
    design = _design()
    first = propose_next_batch(design, root=adaptive_repo)
    second = propose_next_batch(copy.deepcopy(design), root=adaptive_repo)

    assert first == second
    assert first["schema"] == BATCH_SCHEMA
    assert first["decision"] == "propose"
    assert len(first["candidates"]) == 4
    assert first["algorithm"]["candidate_design"] == "scipy.stats.qmc.Sobol-scrambled"
    assert first["experiment_handoff"]["authority"] == "tools/experiment.py"
    assert first["experiment_handoff"]["direct_runner_commands_emitted"] is False
    assert first["experiment_handoff"]["seal_required"] is True
    assert first["experiment_handoff"]["digest_bound_job_expansion_required"] is True
    assert first["experiment_handoff"]["held_out_partitions_accessed"] == []
    assert "command" not in json.dumps(first["candidates"])
    fingerprints = set()
    for candidate in first["candidates"]:
        parameters = candidate["parameters"]
        assert parameters["gain"] * parameters["fan_in"] <= 40
        assert candidate["fidelity_kind"] == "cpu_screen"
        assert candidate["partition"] == "calibration"
        fingerprint = json.dumps(parameters, sort_keys=True)
        assert fingerprint not in fingerprints
        fingerprints.add(fingerprint)


def test_create_only_batch_is_read_only_and_refuses_overwrite(adaptive_repo, tmp_path):
    batch = propose_next_batch(_design(), root=adaptive_repo)
    destination = tmp_path / "plans/next.json"
    assert write_next_batch(destination, batch) == destination
    assert json.loads(destination.read_text()) == batch
    assert os.stat(destination).st_mode & 0o222 == 0
    with pytest.raises(AdaptiveExperimentError, match="refusing to replace"):
        write_next_batch(destination, batch)


def test_batch_writer_and_spec_loader_reject_symlinked_parents(adaptive_repo, tmp_path):
    batch = propose_next_batch(_design(), root=adaptive_repo)
    real_output = tmp_path / "real-output"
    real_output.mkdir()
    linked_output = tmp_path / "linked-output"
    linked_output.symlink_to(real_output, target_is_directory=True)
    with pytest.raises(AdaptiveExperimentError, match="cannot contain a symlink"):
        write_next_batch(linked_output / "next.json", batch)
    assert not (real_output / "next.json").exists()

    real_specs = adaptive_repo / "real-specs"
    real_specs.mkdir()
    (real_specs / "adaptive.json").write_text(json.dumps(_experiment_spec()))
    linked_specs = adaptive_repo / "linked-specs"
    linked_specs.symlink_to(real_specs, target_is_directory=True)
    with pytest.raises(AdaptiveExperimentError, match="cannot contain a symlink"):
        propose_next_batch(_design("linked-specs/adaptive.json"), root=adaptive_repo)


def test_loaded_design_can_be_passed_directly_to_proposer(adaptive_repo, tmp_path):
    path = tmp_path / "design.json"
    path.write_text(json.dumps(_design()), encoding="utf-8")
    loaded = load_adaptive_design(path, root=adaptive_repo)
    assert propose_next_batch(loaded, root=adaptive_repo)["decision"] == "propose"


@pytest.mark.parametrize("location", ["tier", "observation"])
def test_held_out_data_is_rejected_before_tuning(adaptive_repo, location):
    design = _design()
    if location == "tier":
        design["fidelity_tiers"][0]["partition"] = "held_out"
    else:
        design["observations"] = [
            _observation("leak", _screen_points()[0], partition="held_out")
        ]
    with pytest.raises(AdaptiveExperimentError, match="held-out"):
        propose_next_batch(design, root=adaptive_repo)


def test_observation_outside_hard_biological_constraint_is_rejected(adaptive_repo):
    design = _design()
    design["observations"] = [
        _observation("invalid", {"gain": 2.0, "fan_in": 32, "rule": "pair"})
    ]
    with pytest.raises(AdaptiveExperimentError, match="hard biological constraint"):
        propose_next_batch(design, root=adaptive_repo)


def test_categorical_hard_constraint_is_applied_to_proposals(adaptive_repo):
    design = _design()
    design["constraints"].append({
        "id": "plasticity-family",
        "source": "Fixture plasticity reference",
        "predicate": {"op": "eq", "left": {"param": "rule"}, "right": {"value": "triplet"}},
    })
    batch = propose_next_batch(design, root=adaptive_repo)
    assert batch["decision"] == "propose"
    assert all(candidate["parameters"]["rule"] == "triplet" for candidate in batch["candidates"])


def test_surrogate_batch_avoids_completed_points_and_reports_diagnostics(adaptive_repo):
    design = _design()
    points = _screen_points()
    design["observations"] = [
        _observation(f"cpu-{index}", point, score=0.35 + 0.08 * index,
                     rate=10 - index, stability=0.4 + 0.08 * index,
                     seconds=35 - index, scaffolds=3)
        for index, point in enumerate(points)
    ]
    # Prevent promotions so this test isolates new-point acquisition.
    design["policy"]["min_completed_for_promotion"] = 20

    batch = propose_next_batch(design, root=adaptive_repo)
    observed = {json.dumps(point, sort_keys=True) for point in points}
    assert batch["decision"] == "propose"
    assert all(item["reason"] == "surrogate_acquisition" for item in batch["candidates"])
    assert not observed.intersection(json.dumps(item["parameters"], sort_keys=True)
                                     for item in batch["candidates"])
    assert batch["diagnostics"]["status"] == "available"
    assert {item["parameter"] for item in batch["diagnostics"]["sensitivity"]} == {"gain", "fan_in", "rule"}
    assert len(batch["diagnostics"]["interactions"]) == 3
    assert batch["diagnostics"]["pareto_observation_ids_by_fidelity"]["cpu"]
    assert len(batch["decision_conditions"]["stop"]) == 3
    assert len(batch["decision_conditions"]["escalate_to_research"]) == 4


def test_best_cpu_point_is_promoted_before_new_screen_points(adaptive_repo):
    design = _design()
    points = _screen_points()[:3]
    design["observations"] = [
        _observation("cpu-low", points[0], score=0.4, rate=8, stability=0.5),
        _observation("cpu-mid", points[1], score=0.76, rate=3, stability=0.75),
        _observation("cpu-best", points[2], score=0.95, rate=1, stability=0.95),
    ]
    batch = propose_next_batch(design, root=adaptive_repo)
    first = batch["candidates"][0]
    assert first["reason"] == "promote"
    assert first["source_observation"] == "cpu-best"
    assert first["fidelity_kind"] == "gpu"
    assert first["parameters"] == points[2]


def test_gpu_point_can_promote_to_replication_without_using_held_out(adaptive_repo):
    design = _design()
    points = _screen_points()[:3]
    design["observations"] = [
        _observation(f"cpu-{index}", point, score=0.8 + index * 0.05, rate=2, stability=0.8)
        for index, point in enumerate(points)
    ] + [
        _observation(f"gpu-{index}", point, fidelity="gpu", score=0.8 + index * 0.05,
                     rate=2, stability=0.8)
        for index, point in enumerate(points)
    ]
    batch = propose_next_batch(design, root=adaptive_repo)
    assert batch["candidates"][0]["fidelity_kind"] == "replication"
    assert batch["candidates"][0]["partition"] == "replication"
    assert all(item["partition"] != "held_out" for item in batch["candidates"])


def test_replicated_target_causes_explicit_stop(adaptive_repo):
    design = _design()
    point = _screen_points()[0]
    design["observations"] = [
        _observation("replicated", point, fidelity="confirm", partition="replication",
                     score=0.9, rate=2, stability=0.9)
    ]
    batch = propose_next_batch(design, root=adaptive_repo)
    assert batch["decision"] == "stop"
    assert batch["candidates"] == []
    assert "replication fidelity" in batch["reasons"][0]


def test_infeasible_bounded_space_escalates_to_research(adaptive_repo):
    design = _design()
    design["constraints"] = [{
        "id": "impossible",
        "source": "A cited biological range that conflicts with this parameterization",
        "predicate": {"op": "gt", "left": {"param": "gain"}, "right": {"value": 10}},
    }]
    batch = propose_next_batch(design, root=adaptive_repo)
    assert batch["decision"] == "escalate_to_research"
    assert batch["candidates"] == []
    assert "no feasible candidate" in batch["reasons"][0]


def test_all_required_objective_classes_must_be_present(adaptive_repo):
    design = _design()
    design["objectives"] = design["objectives"][:-1]
    with pytest.raises(AdaptiveExperimentError, match="missing.*scaffold_penalty"):
        propose_next_batch(design, root=adaptive_repo)


def test_duplicate_completed_cell_is_rejected(adaptive_repo):
    design = _design()
    point = _screen_points()[0]
    design["observations"] = [
        _observation("one", point),
        _observation("two", point),
    ]
    with pytest.raises(AdaptiveExperimentError, match="duplicate completed observation"):
        propose_next_batch(design, root=adaptive_repo)


def test_finite_space_exhaustion_escalates_when_targets_are_not_met(adaptive_repo):
    design = _design()
    design["parameter_space"] = {
        "gain": {"type": "discrete", "values": [0.1, 0.2]},
        "fan_in": {"type": "discrete", "values": [8, 16]},
        "rule": {"type": "categorical", "values": ["pair", "triplet"]},
    }
    points = [
        {"gain": gain, "fan_in": fan_in, "rule": rule}
        for gain in (0.1, 0.2) for fan_in in (8, 16) for rule in ("pair", "triplet")
    ]
    design["observations"] = [
        _observation(f"cell-{index}", point, score=0.2, rate=15, stability=0.2)
        for index, point in enumerate(points)
    ]
    design["policy"]["min_completed_for_promotion"] = 20
    batch = propose_next_batch(design, root=adaptive_repo)
    assert batch["decision"] == "escalate_to_research"
    assert any("exhausted" in reason for reason in batch["reasons"])


def test_continuous_candidate_pool_exhaustion_escalates(adaptive_repo):
    design = _design()
    design["policy"]["candidate_pool_size"] = 16
    design["policy"]["min_completed_for_promotion"] = 20
    design["policy"]["max_completed_observations"] = 100
    validated = adaptive_experiment._validate_design(design, root=adaptive_repo)
    points, _, _ = adaptive_experiment._candidate_pool(validated)
    design["observations"] = [
        _observation(f"cell-{index}", point, score=0.2, rate=15, stability=0.2)
        for index, point in enumerate(points)
    ]

    batch = propose_next_batch(design, root=adaptive_repo)
    assert batch["decision"] == "escalate_to_research"
    assert batch["candidates"] == []
    assert any("exhausted" in reason for reason in batch["reasons"])


def test_plateau_does_not_preempt_available_promotion(adaptive_repo):
    design = _design()
    design["policy"]["plateau_window"] = 2
    points = _screen_points()[:4]
    design["observations"] = [
        _observation(f"cpu-{index}", point, score=0.8, rate=2, stability=0.8)
        for index, point in enumerate(points)
    ]

    batch = propose_next_batch(design, root=adaptive_repo)
    assert batch["decision"] == "propose"
    assert batch["candidates"][0]["reason"] == "promote"
    plateau = next(
        condition for condition in batch["decision_conditions"]["stop"]
        if condition["id"] == "utility_plateau"
    )
    assert plateau["triggered"] is False


def test_maximum_observation_budget_stops_even_with_available_points(adaptive_repo):
    design = _design()
    design["policy"]["max_completed_observations"] = 1
    design["observations"] = [_observation("one", _screen_points()[0], score=0.2, rate=15, stability=0.2)]
    batch = propose_next_batch(design, root=adaptive_repo)
    assert batch["decision"] == "stop"
    assert batch["candidates"] == []
    assert "budget" in batch["reasons"][0]
