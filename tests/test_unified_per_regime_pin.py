"""Grounding pin for the unified per-regime-monitor + per-regime-encoding
stage. Intentionally RED until Task 1 lands the runner. This IS the
Task-1 completion gate (see
docs/plans/2026-05-20-unified-per-regime-monitor-with-per-regime-encoding-implementation.md).
"""
import importlib


def test_unified_per_regime_runner_importable():
    m = importlib.import_module(
        "research.runners.unified_per_regime_monitor_runner"
    )
    assert hasattr(m, "run_unified_per_regime_monitor")
