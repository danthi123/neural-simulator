"""Grounding pin for the shared-rhythm SPEAR conversational stage.

Intentionally RED until Task 2 lands the runner. This IS the
Task-1/Task-2 completion gate (see
docs/plans/2026-05-19-shared-rhythm-SPEAR-conversational-implementation.md).
"""

import importlib


def test_spear_runner_importable():
    m = importlib.import_module("research.runners.spear_conversational_runner")
    assert hasattr(m, "run_spear_conversational")


def test_spear_core_importable():
    m = importlib.import_module("research.runners.spear_conversational_core")
    assert hasattr(m, "spear_conversational_verdict")
