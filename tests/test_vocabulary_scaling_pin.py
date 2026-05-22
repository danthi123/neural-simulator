"""Grounding pin for the vocabulary-scaling arc (Task 0 of
docs/plans/2026-05-22-vocabulary-scaling-implementation.md).

This test imports the vocabulary-scaling runner and pins its
pre-registered constants -- the 64-concept vocabulary, the frozen 0.80
compositional bar, and the {2,3,5} load set. It FAILS until Task 2 of
the implementation plan creates the runner; that is intentional -- this
pin IS the Task-2 gate, and it goes green only when the runner exists
with exactly the pre-registered constants. The constants are frozen
here in advance so a later task cannot quietly move them.
"""
import importlib

import pytest


def test_vocabulary_scaling_runner_pinned_constants():
    """The vocabulary-scaling runner exists and carries the
    pre-registered constants (fails until Task 2 builds the runner)."""
    try:
        mod = importlib.import_module(
            "research.findings.raw.vocabulary_scaling_run")
    except ModuleNotFoundError:
        pytest.fail(
            "research/findings/raw/vocabulary_scaling_run.py does not "
            "exist yet -- Task 2 of the vocabulary-scaling plan creates "
            "it; this pin goes green then.")
    assert mod.N_CONCEPTS == 64, "vocabulary pinned at 64 concepts"
    assert mod.BAR == 0.80, "frozen compositional bar pinned at 0.80"
    assert list(mod.LOADS) == [2, 3, 5], "load set pinned at {2,3,5}"
