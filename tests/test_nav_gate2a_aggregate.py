"""Tests for the nav-gate(a) aggregator.

Gate (a) of roadmap step 2a asks one question: does adding the frozen,
index-disjoint conversational half to the navigation bridge change the
navigation score? The aggregator reads the 12 per-seed run files
(standalone vs merged, seeds 42-47), computes each run's navigation score
(the sum over the four goal phases of the final-quarter mean distance to
goal), and renders a pass/fail verdict.

Lower navigation score = better (the agent sits closer to the goal). A
"merged minus standalone" delta of ~0 at a matched seed means the
conversational half is inert (the intended result). A delta beyond the
deterministic run-to-run noise floor is a real finding (the measured cost
of merging), not something to paper over.

These tests use synthetic fixtures written to a temporary directory, so
they do not depend on any raw run output being committed.
"""

import json
import os

import pytest

from research.runners.nav_gate2a_aggregate import (
    aggregate_gate2a,
    score_from_data,
    verdict,
)


def _phase_stats(final_quarter_values):
    """Build a minimal phase_stats list with the given per-phase scores."""
    return {
        "phase_stats": [
            {"phase": i, "final_quarter_mean_distance": v}
            for i, v in enumerate(final_quarter_values)
        ]
    }


def _write_run(raw_dir, seed, arm, final_quarter_values):
    path = os.path.join(raw_dir, f"gate6_{arm}_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(_phase_stats(final_quarter_values), f)
    return path


# ---------------------------------------------------------------------------
# score_from_data
# ---------------------------------------------------------------------------

def test_score_is_sum_of_phase_final_quarters():
    data = _phase_stats([0.5, 0.5, 0.5, 0.5])
    assert score_from_data(data) == pytest.approx(2.0)


def test_score_raises_on_missing_phase_stats():
    with pytest.raises(ValueError):
        score_from_data({"not_phase_stats": []})


def test_score_raises_on_empty_phase_stats():
    with pytest.raises(ValueError):
        score_from_data({"phase_stats": []})


# ---------------------------------------------------------------------------
# aggregate_gate2a
# ---------------------------------------------------------------------------

def test_aggregate_byte_identical_arms_is_green_inert(tmp_path):
    raw = str(tmp_path)
    for seed in range(42, 48):
        vals = [0.4956, 0.5044, 0.4956, 0.5044]  # sums to 2.0
        _write_run(raw, seed, "standalone", vals)
        _write_run(raw, seed, "merged", vals)  # identical -> delta 0

    agg = aggregate_gate2a(raw, seeds=range(42, 48))

    assert agg["n_complete"] == 6
    assert agg["max_abs_delta"] == pytest.approx(0.0, abs=1e-9)
    assert agg["mean_delta"] == pytest.approx(0.0, abs=1e-9)
    v = verdict(agg)
    assert v["label"] == "GREEN_INERT", v


def test_aggregate_reports_missing_files_as_incomplete(tmp_path):
    raw = str(tmp_path)
    # Only seed 42 present; 43-47 absent.
    _write_run(raw, 42, "standalone", [0.5, 0.5, 0.5, 0.5])
    _write_run(raw, 42, "merged", [0.5, 0.5, 0.5, 0.5])

    agg = aggregate_gate2a(raw, seeds=range(42, 48))

    assert agg["n_complete"] == 1
    assert len(agg["missing"]) == 10  # 5 seeds x 2 arms
    v = verdict(agg)
    assert v["label"] == "INCOMPLETE", v


def test_aggregate_regression_when_merged_worse_beyond_noise(tmp_path):
    raw = str(tmp_path)
    for seed in range(42, 48):
        _write_run(raw, seed, "standalone", [0.5, 0.5, 0.5, 0.5])  # 2.0
        # merged consistently +1.0 per phase worse -> sum 6.0, delta +4.0
        _write_run(raw, seed, "merged", [1.5, 1.5, 1.5, 1.5])

    agg = aggregate_gate2a(raw, seeds=range(42, 48))

    assert agg["mean_delta"] == pytest.approx(4.0)
    v = verdict(agg)
    assert v["label"] == "REGRESS", v


def test_aggregate_small_delta_within_noise_is_green_noise(tmp_path):
    raw = str(tmp_path)
    for seed in range(42, 48):
        _write_run(raw, seed, "standalone", [0.5, 0.5, 0.5, 0.5])  # 2.0
        # merged +0.05 per phase -> sum 2.2, delta +0.2 (< 0.7 noise floor,
        # > 0.05 inert epsilon)
        _write_run(raw, seed, "merged", [0.55, 0.55, 0.55, 0.55])

    agg = aggregate_gate2a(raw, seeds=range(42, 48))

    assert agg["max_abs_delta"] == pytest.approx(0.2, abs=1e-6)
    v = verdict(agg)
    assert v["label"] == "GREEN_WITHIN_NOISE", v
