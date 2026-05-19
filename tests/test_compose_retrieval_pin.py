"""Grounding pin for regime-correct compositional retrieval.

Intentionally RED until Task 2 lands the runner. This IS the Task-1/Task-2
completion gate (see
docs/plans/2026-05-19-regime-correct-compositional-retrieval-implementation.md).
"""

import importlib


def test_compose_retrieval_runner_importable():
    mod = importlib.import_module("research.runners.compose_retrieval_runner")
    assert hasattr(mod, "run_compose_retrieval")


def test_compose_retrieval_core_importable():
    mod = importlib.import_module("research.runners.compose_retrieval_core")
    assert hasattr(mod, "compose_retrieval_verdict")
