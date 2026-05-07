"""Tests for the Phase 1.4 forgetting eval result summarizer."""
import json
import pytest
from pathlib import Path


def test_extract_metrics_from_metrics_field():
    """Modern JSON output: metrics field contains primary_a_acc etc."""
    from research.runners.forgetting_summarize import extract_metrics

    data = {
        "metrics": {
            "primary_a_acc": 0.40,
            "primary_b_acc": 0.32,
            "synonym_b_acc": 0.30,
            "retention_pct": 80.0,
        }
    }
    m = extract_metrics(data)
    assert m["phase_a_acc"] == 0.40
    assert m["phase_b_acc"] == 0.32
    assert m["retention_pct"] == 80.0
    assert m["synonym_b_acc"] == 0.30
    assert m["sanity_failed"] is False


def test_extract_metrics_from_checkpoints_fallback():
    """Older JSON without metrics field: pull from checkpoints array."""
    from research.runners.forgetting_summarize import extract_metrics

    data = {
        "checkpoints": [
            {"name": "after_phase_a",
             "primary_wa": {"accuracy": 0.40}},
            {"name": "after_phase_b",
             "primary_wa": {"accuracy": 0.32},
             "synonym_wa": {"accuracy": 0.30}},
        ]
    }
    m = extract_metrics(data)
    assert m["phase_a_acc"] == 0.40
    assert m["phase_b_acc"] == 0.32
    assert m["retention_pct"] == pytest.approx(80.0, abs=0.1)


def test_extract_metrics_sanity_failed():
    """Sanity-failed runs (Phase A below chance) should be flagged."""
    from research.runners.forgetting_summarize import extract_metrics

    data = {
        "sanity_check_failed": True,
        "metrics": {
            "primary_a_acc": 0.14,
            "primary_b_acc": None,
            "retention_pct": None,
            "abort_reason": "phase_a_below_chance",
        },
    }
    m = extract_metrics(data)
    assert m["sanity_failed"] is True
    assert m["phase_a_acc"] == 0.14
    assert m["retention_pct"] is None


def test_grade_thresholds():
    """grade() returns correct branch label per master plan thresholds."""
    from research.runners.forgetting_summarize import grade

    assert "PASS" in grade(95.0)
    assert "PASS" in grade(80.0)  # exactly 80 threshold = PASS
    assert "MODERATE" in grade(79.9)
    assert "MODERATE" in grade(50.0)  # exactly 50 threshold = MODERATE
    assert "FAIL" in grade(49.9)
    assert "FAIL" in grade(10.0)
    assert "?" in grade(None)


def test_summarize_with_real_files(tmp_path):
    """End-to-end: write fake forgetting JSONs, summarize, parse output."""
    from research.runners.forgetting_summarize import (
        load_seed, extract_metrics, grade,
    )

    # Seed 42: passing
    (tmp_path / "forgetting_seed42.json").write_text(json.dumps({
        "metrics": {
            "primary_a_acc": 0.40,
            "primary_b_acc": 0.36,  # 90% retention
            "synonym_b_acc": 0.30,
            "retention_pct": 90.0,
        }
    }))

    # Seed 43: moderate
    (tmp_path / "forgetting_seed43.json").write_text(json.dumps({
        "metrics": {
            "primary_a_acc": 0.40,
            "primary_b_acc": 0.24,  # 60% retention
            "synonym_b_acc": 0.28,
            "retention_pct": 60.0,
        }
    }))

    # Seed 44: failing
    (tmp_path / "forgetting_seed44.json").write_text(json.dumps({
        "metrics": {
            "primary_a_acc": 0.40,
            "primary_b_acc": 0.10,  # 25% retention
            "synonym_b_acc": 0.20,
            "retention_pct": 25.0,
        }
    }))

    # Verify each loads + parses correctly
    for seed, expected_grade in [(42, "PASS"),
                                  (43, "MODERATE"),
                                  (44, "FAIL")]:
        path = tmp_path / f"forgetting_seed{seed}.json"
        data = load_seed(path)
        assert data is not None
        m = extract_metrics(data)
        assert expected_grade in grade(m["retention_pct"])


def test_load_seed_missing_file(tmp_path):
    """load_seed returns None for missing files (not an error)."""
    from research.runners.forgetting_summarize import load_seed
    assert load_seed(tmp_path / "does_not_exist.json") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
