from __future__ import annotations

import json
from pathlib import Path

from tools import assign_vocal_credit_v5_learning_seeds as seeds


def test_ast_scan_covers_prior_vocal_and_active_gate_partitions():
    declarations = seeds.scan_seed_declarations()
    used = {declaration.seed for declaration in declarations}
    assert {7, 11, 401, 409, 70001, 70003} <= used
    assert {503, 509, 601, 607} <= used


def test_assignment_is_deterministic_complete_and_collision_free():
    declarations = seeds.scan_seed_declarations()
    first = seeds.assignment(declarations)
    second = seeds.assignment(declarations)
    assert first == second
    assert {name: len(values) for name, values in first.items()} == {
        "calibration": 2,
        "development": 4,
        "held_out": 2,
    }
    assigned = {seed for values in first.values() for seed in values}
    used = {declaration.seed for declaration in declarations}
    assert len(assigned) == 8
    assert not assigned & used


def test_committed_manifest_is_valid_and_keeps_formal_execution_sealed():
    manifest_path = Path(
        "tools/seed_manifests/vocal_action_credit_gate_v5_learning.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seeds.validate_manifest(manifest)
    assert manifest["formal_execution_open"] is False
    assert manifest["collisions"] == []
