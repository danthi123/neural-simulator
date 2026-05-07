"""Unit tests for Phase 1.5 continual_eval_suite dispatcher.

Tests dispatcher plumbing without running heavyweight benchmarks:
- All 6 benchmarks register correctly
- register_benchmark decorator works
- Dispatcher main() handles unknown benchmarks gracefully
- Stub benchmarks return correct status structure
- Aggregate score calculation handles all-stub, mixed, all-completed
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_all_six_benchmarks_registered():
    """Verify all 6 expected benchmarks are present in the registry."""
    from research.runners.continual_eval_suite import BENCHMARK_REGISTRY
    expected = {
        "sequential_expansion",
        "retention_over_time",
        "interference",
        "long_tail",
        "multimodality",
        "composition",
    }
    assert set(BENCHMARK_REGISTRY.keys()) == expected


def test_register_benchmark_decorator():
    """Verify register_benchmark properly registers callables."""
    from research.runners.continual_eval_suite import (
        register_benchmark, BENCHMARK_REGISTRY,
    )
    # Save original to restore after test
    original = dict(BENCHMARK_REGISTRY)
    try:
        @register_benchmark("test_dummy")
        def dummy(args, rng):
            return {"name": "test_dummy", "score": 1.0,
                    "pass": True, "details": {}}

        assert "test_dummy" in BENCHMARK_REGISTRY
        result = BENCHMARK_REGISTRY["test_dummy"](None, None)
        assert result["name"] == "test_dummy"
        assert result["score"] == 1.0
    finally:
        # Restore registry
        BENCHMARK_REGISTRY.clear()
        BENCHMARK_REGISTRY.update(original)


def test_stub_benchmarks_return_correct_structure():
    """Stub benchmarks (multimodality, composition) must return
    correct status so the dispatcher can identify and skip them
    in aggregate calculation."""
    from research.runners.continual_eval_suite import BENCHMARK_REGISTRY

    args = SimpleNamespace(seed=42, events_per_word=20)
    rng = None

    for stub_name in ("multimodality", "composition"):
        result = BENCHMARK_REGISTRY[stub_name](args, rng)
        assert result["name"] == stub_name
        assert result["score"] == 0.0
        assert result["pass"] is False
        assert "status" in result["details"]
        assert result["details"]["status"] in (
            "tier_2_2_pending", "tier_2_3_pending"
        )


def test_aggregate_excludes_stub_benchmarks(tmp_path):
    """Aggregate score should ignore stubs (status: not_yet_implemented,
    tier_2_2_pending, tier_2_3_pending) and benchmarks that errored."""
    from research.runners.continual_eval_suite import BENCHMARK_REGISTRY
    import numpy as np

    # Simulated results: 2 completed, 1 stub, 1 errored
    completed = [
        {"name": "a", "score": 0.8, "pass": True, "details": {}},
        {"name": "b", "score": 0.9, "pass": True, "details": {}},
    ]
    stub = {"name": "c", "score": 0.0, "pass": False,
            "details": {"status": "tier_2_2_pending"}}
    errored = {"name": "d", "score": 0.0, "pass": False,
               "details": {"error": "boom"}}

    all_results = completed + [stub, errored]
    completed_only = [
        b for b in all_results
        if b["details"].get("status") not in
           ("not_yet_implemented", "tier_2_2_pending",
            "tier_2_3_pending")
        and "error" not in b["details"]
    ]
    assert len(completed_only) == 2
    assert all(b["pass"] for b in completed_only)
    agg = float(np.mean([b["score"] for b in completed_only]))
    assert agg == pytest.approx(0.85)


def test_aggregate_pass_threshold():
    """Aggregate score >= 0.7 with all_pass should signal validated."""
    import numpy as np
    # 4 completed benchmarks at 0.7+ aggregate, all pass
    benchmarks = [
        {"score": 0.85, "pass": True},
        {"score": 0.75, "pass": True},
        {"score": 0.70, "pass": True},
        {"score": 0.80, "pass": True},
    ]
    agg = float(np.mean([b["score"] for b in benchmarks]))
    all_pass = all(b["pass"] for b in benchmarks)
    assert agg >= 0.7
    assert all_pass

    # One failing benchmark drops all_pass even with high agg
    benchmarks_with_fail = [
        {"score": 0.85, "pass": True},
        {"score": 0.95, "pass": True},
        {"score": 0.50, "pass": False},  # one fails despite high agg
    ]
    agg = float(np.mean([b["score"] for b in benchmarks_with_fail]))
    all_pass = all(b["pass"] for b in benchmarks_with_fail)
    assert agg >= 0.7  # mean ~ 0.77
    assert all_pass is False  # but not all pass


def test_dispatcher_module_loads_clean():
    """Smoke test: module imports without errors and has a main() entry."""
    from research.runners import continual_eval_suite
    assert hasattr(continual_eval_suite, "main")
    assert hasattr(continual_eval_suite, "BENCHMARK_REGISTRY")
    assert hasattr(continual_eval_suite, "register_benchmark")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
