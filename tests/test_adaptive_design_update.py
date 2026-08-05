from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.adaptive_design_update import (
    AdaptiveDesignUpdateError,
    update_adaptive_design,
)
from tools.experiment_observation import digest


def _write(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="ascii")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _design() -> dict:
    return {
        "schema": "sim-adaptive-experiment-v1",
        "id": "design-v1",
        "experiment": {"spec_path": "spec.json"},
        "parameter_space": {
            "gain": {"type": "continuous", "low": 0.1, "high": 1.0},
        },
        "constraints": [{
            "id": "positive", "source": "fixture", "predicate": {
                "op": "ge", "left": {"param": "gain"}, "right": {"value": 0.1},
            },
        }],
        "objectives": [
            {"name": "loss", "category": "mechanism", "direction": "minimize",
             "weight": 1.0, "range": [0.0, 10.0]},
            {"name": "physiology_loss", "category": "physiology", "direction": "minimize",
             "weight": 1.0, "range": [0.0, 10.0]},
            {"name": "robustness_loss", "category": "robustness", "direction": "minimize",
             "weight": 1.0, "range": [0.0, 10.0]},
            {"name": "seconds", "category": "compute", "direction": "minimize",
             "weight": 0.1, "range": [0.0, 10.0]},
            {"name": "scaffolds", "category": "scaffold_penalty", "direction": "minimize",
             "weight": 0.1, "range": [0.0, 2.0]},
        ],
        "fidelity_tiers": [
            {"name": "screen", "kind": "cpu_screen", "backend": "numpy",
             "partition": "calibration", "cost": 1.0},
            {"name": "gpu", "kind": "gpu", "backend": "cupy",
             "partition": "gpu_calibration", "cost": 2.0},
            {"name": "confirm", "kind": "replication", "backend": "cupy",
             "partition": "replication", "cost": 3.0},
        ],
        "observations": [],
        "policy": {
            "seed": 1, "batch_size": 2, "candidate_pool_size": 16,
            "initial_design_size": 2, "min_surrogate_observations": 3,
            "exploration_weight": 0.2, "promotion_slots": 1,
            "promotion_quantile": 0.5, "min_completed_for_promotion": 2,
            "max_completed_observations": 20, "plateau_window": 6,
            "min_improvement": 0.001, "min_feasible_fraction": 0.01,
            "research_after_observations": 4, "max_model_uncertainty": 0.01,
            "stop_on_replicated_targets": True,
        },
    }


def _fixture(tmp_path: Path) -> dict:
    root = tmp_path / "repo"
    root.mkdir()
    _write(root / "spec.json", {
        "schema": "sim-experiment-spec-v0", "id": "fixture",
        "partitions": {"calibration": [1], "gpu_calibration": [2],
                       "replication": [3], "held_out": [99]},
        "backends": ["numpy", "cupy"],
    })
    design = _design()
    design_sha = _write(root / "design-v1.json", design)
    row = {
        "id": "candidate-a--screen", "status": "complete",
        "parameters": {"gain": 0.4}, "fidelity": "screen",
        "partition": "calibration",
        "objectives": {"loss": 1.0, "physiology_loss": 1.5,
                       "robustness_loss": 1.25, "seconds": 2.0,
                       "scaffolds": 0.0},
    }
    body = {
        "schema": "sim-observation-output-v1",
        "canonicalization": "json-sort-keys-compact-ascii-v1",
        "contract": {"path": "contract.json", "sha256": "1" * 64},
        "adaptive_design": {"path": "design-v1.json", "sha256": design_sha},
        "executor_manifest": {"path": "manifest.json", "sha256": "2" * 64},
        "observations": [row],
        "evidence": [{"observation_id": row["id"], "receipts": [
            {"path": "receipt.json", "sha256": "3" * 64},
        ]}],
        "blocked": [], "scientific_verdict": None,
    }
    observations = {**body, "sha256": digest(body)}
    _write(root / "observations.json", observations)
    return {"root": root, "design": design, "observations": observations}


def _run(fixture: dict):
    root = fixture["root"]
    return update_adaptive_design(
        root / "design-v1.json", root / "observations.json",
        root / "design-v2.json", root / "design-v2.update.json",
        new_id="design-v2", repository_root=root,
    )


def test_appends_authenticated_rows_and_writes_lineage_receipt(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    updated, receipt = _run(fixture)

    assert updated["id"] == "design-v2"
    assert updated["observations"] == fixture["observations"]["observations"]
    assert receipt["appended_observation_count"] == 1
    assert receipt["scientific_verdict"] is None
    assert receipt["sha256"] == digest({k: v for k, v in receipt.items() if k != "sha256"})
    assert (fixture["root"] / "design-v2.json").stat().st_mode & 0o222 == 0


@pytest.mark.parametrize("tamper", ["binding", "self_digest", "evidence", "row"])
def test_rejects_unbound_or_internally_inconsistent_observations(
    tmp_path: Path, tamper: str,
) -> None:
    fixture = _fixture(tmp_path)
    value = fixture["observations"]
    if tamper == "binding":
        value["adaptive_design"]["sha256"] = "9" * 64
    elif tamper == "self_digest":
        value["blocked"].append({"reason": "changed"})
    elif tamper == "evidence":
        value["evidence"] = []
        value["sha256"] = digest({k: v for k, v in value.items() if k != "sha256"})
    else:
        value["observations"][0]["parameters"]["gain"] = 99.0
        value["sha256"] = digest({k: v for k, v in value.items() if k != "sha256"})
    _write(fixture["root"] / "observations.json", value)

    with pytest.raises(AdaptiveDesignUpdateError):
        _run(fixture)


def test_refuses_overwrite_and_duplicate_cell(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _run(fixture)
    with pytest.raises(AdaptiveDesignUpdateError, match="replace"):
        _run(fixture)

    root = fixture["root"]
    first = fixture["observations"]["observations"][0]
    old = _design()
    old["observations"] = [first]
    design_sha = _write(root / "design-existing.json", old)
    value = fixture["observations"]
    value["observations"][0]["id"] = "different-id"
    value["evidence"][0]["observation_id"] = "different-id"
    value["adaptive_design"] = {"path": "design-existing.json", "sha256": design_sha}
    value["sha256"] = digest({k: v for k, v in value.items() if k != "sha256"})
    _write(root / "observations-existing.json", value)
    with pytest.raises(AdaptiveDesignUpdateError, match="cell"):
        update_adaptive_design(
            root / "design-existing.json", root / "observations-existing.json",
            root / "design-v3.json", root / "design-v3.update.json",
            new_id="design-v3", repository_root=root,
        )
