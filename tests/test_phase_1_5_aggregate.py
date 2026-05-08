"""Smoke tests for phase_1_5_aggregate."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.runners.phase_1_5_aggregate import aggregate


def _write(tmp_path, name, payload):
    p = tmp_path / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return str(p)


def _make_seed_payload(seed, sequential=1.0, retention=1.0,
                       interference=0.5, long_tail=0.4):
    return {
        "seed": seed,
        "benchmarks": [
            {"name": "sequential_expansion", "score": sequential,
             "pass": sequential >= 0.7, "details": {}},
            {"name": "retention_over_time", "score": retention,
             "pass": retention >= 0.7, "details": {}},
            {"name": "interference", "score": interference,
             "pass": interference >= 0.5, "details": {}},
            {"name": "long_tail", "score": long_tail,
             "pass": long_tail >= 0.5, "details": {}},
        ],
        "aggregate": {
            "score": (sequential + retention + interference + long_tail) / 4,
            "all_pass": all(s >= 0.5 for s in
                            [sequential, retention, interference, long_tail]),
        },
    }


def test_aggregate_basic(tmp_path):
    """Three seeds with all 4 benchmarks active."""
    paths = [
        _write(tmp_path, "g11_seed42_phase_1_5.json",
               _make_seed_payload(42, 1.0, 1.0, 0.6, 0.5)),
        _write(tmp_path, "g11_seed43_phase_1_5.json",
               _make_seed_payload(43, 1.0, 1.0, 0.55, 0.45)),
        _write(tmp_path, "g11_seed44_phase_1_5.json",
               _make_seed_payload(44, 1.0, 1.0, 0.5, 0.4)),
    ]
    s = aggregate(paths)
    assert s["n_seeds"] == 3
    assert "sequential_expansion" in s["benchmarks"]
    assert s["benchmarks"]["sequential_expansion"]["score_mean"] == 1.0
    assert s["benchmarks"]["sequential_expansion"]["pass_rate"] == 1.0
    assert s["benchmarks"]["interference"]["score_mean"] == pytest.approx(0.55)
    # Overall mean is mean across benchmarks (not raw scores)
    assert s["overall"]["aggregate_score_mean"] == pytest.approx(
        (1.0 + 1.0 + 0.55 + 0.45) / 4
    )


def test_aggregate_skips_pending_tier_2(tmp_path):
    """Benchmarks with tier_2_2_pending / tier_2_3_pending get skipped."""
    paths = [
        _write(tmp_path, "g11_seed42_phase_1_5.json", {
            "seed": 42,
            "benchmarks": [
                {"name": "sequential_expansion", "score": 1.0,
                 "pass": True, "details": {}},
                {"name": "multimodality", "score": 0.0, "pass": False,
                 "details": {"status": "tier_2_2_pending"}},
                {"name": "composition", "score": 0.0, "pass": False,
                 "details": {"status": "tier_2_3_pending"}},
            ],
            "aggregate": {"score": 0.33, "all_pass": False},
        }),
    ]
    s = aggregate(paths)
    assert s["benchmarks"]["sequential_expansion"]["n_seeds"] == 1
    assert s["benchmarks"]["multimodality"]["skipped"] is True
    assert s["benchmarks"]["composition"]["skipped"] is True
    # Overall mean ignores skipped benchmarks
    assert s["overall"]["aggregate_score_mean"] == 1.0
    assert s["overall"]["n_active_benchmarks"] == 1


def test_aggregate_master_plan_threshold(tmp_path):
    """master_plan_pass = aggregate_score_mean >= 0.70."""
    # Below threshold
    paths = [
        _write(tmp_path, "g11_seed42_phase_1_5.json",
               _make_seed_payload(42, 0.5, 0.5, 0.5, 0.5)),
    ]
    s = aggregate(paths)
    assert s["overall"]["aggregate_score_mean"] == 0.5
    assert s["overall"]["master_plan_pass"] is False

    # Above threshold (use a fresh subdir to avoid name clashes)
    above_dir = tmp_path / "above"
    above_dir.mkdir()
    paths = [
        _write(above_dir, "g11_seed42_phase_1_5.json",
               _make_seed_payload(42, 1.0, 1.0, 0.7, 0.5)),
    ]
    s2 = aggregate(paths)
    assert s2["overall"]["aggregate_score_mean"] == pytest.approx(0.8)
    assert s2["overall"]["master_plan_pass"] is True


def test_aggregate_filters_cmd_json(tmp_path):
    """Launcher .cmd.json sidecars are filtered out."""
    paths = [
        _write(tmp_path, "g11_seed42_phase_1_5.json",
               _make_seed_payload(42, 1.0, 1.0, 0.5, 0.5)),
        _write(tmp_path, "g11_seed42_phase_1_5.cmd.json",
               {"this_is_a_sidecar": True}),  # has no 'benchmarks'
    ]
    # Aggregate should ignore .cmd.json
    s = aggregate(paths)
    assert s["n_seeds"] == 1


def test_aggregate_empty_raises(tmp_path):
    with pytest.raises(ValueError, match="No result files"):
        aggregate([])
