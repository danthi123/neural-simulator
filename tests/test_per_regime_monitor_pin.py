"""Grounding pin for the per-regime metacognitive-monitor stage.

Intentionally RED until Tasks 2 + 3 land. This IS the Tasks 1-3
completion gate (see
docs/plans/2026-05-20-per-regime-metacognitive-monitor-implementation.md).
"""
import importlib


def test_per_regime_core_importable():
    m = importlib.import_module("research.runners.per_regime_monitor_core")
    assert hasattr(m, "per_regime_monitor_verdict")


def test_compositional_gate_importable():
    m = importlib.import_module(
        "research.runners.abstention_gate_compositional"
    )
    assert hasattr(m, "gate")
    assert hasattr(m, "COMPOSITIONAL_THRESHOLD")


def test_per_regime_runner_importable():
    m = importlib.import_module("research.runners.per_regime_monitor_runner")
    assert hasattr(m, "run_per_regime_monitor")
