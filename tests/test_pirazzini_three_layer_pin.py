"""Grounding pin for the Pirazzini-reference three-layer theta-gamma stage.

Intentionally RED until Task 2 lands the runner. This IS the
Task-1/Task-2 completion gate (see
docs/plans/2026-05-19-pirazzini-reference-three-layer-implementation.md).
"""

import importlib


def test_pirazzini_runner_importable():
    m = importlib.import_module("research.runners.pirazzini_three_layer_runner")
    assert hasattr(m, "run_pirazzini_three_layer")


def test_pirazzini_core_importable():
    m = importlib.import_module("research.runners.pirazzini_three_layer_core")
    assert hasattr(m, "pirazzini_three_layer_verdict")
