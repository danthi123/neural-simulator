"""Smoke tests for chat_demo_aggregate.

Tests the multi-seed aggregator handles the three demo types
(tier1, synonym, continual) cleanly + filters out launcher sidecars.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.runners.chat_demo_aggregate import aggregate


def _write(tmp_path, name, payload):
    p = tmp_path / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return str(p)


def test_aggregate_tier1_basic(tmp_path):
    """Three Tier 1 seed JSONs with overall accuracy."""
    paths = [
        _write(tmp_path, "g11_seed42_chat_demo.json",
               {"seed": 42, "accuracy": 0.16, "correct": 2, "total": 12}),
        _write(tmp_path, "g11_seed43_chat_demo.json",
               {"seed": 43, "accuracy": 0.42, "correct": 5, "total": 12}),
        _write(tmp_path, "g11_seed44_chat_demo.json",
               {"seed": 44, "accuracy": 0.33, "correct": 4, "total": 12}),
    ]
    s = aggregate(paths)
    assert s["n_seeds"] == 3
    assert s["demo_types"] == ["tier1"]
    assert abs(s["accuracy_mean"] - 0.303) < 0.01  # mean of 16, 42, 33
    assert s["accuracy_min"] == 0.16
    assert s["accuracy_max"] == 0.42


def test_aggregate_tier1_per_direction(tmp_path):
    """Tier 1 seed JSON with per-direction breakdown."""
    paths = [
        _write(tmp_path, "g11_seed42_chat_demo.json",
               {"seed": 42, "accuracy": 0.50, "correct": 6, "total": 12,
                "per_word_accuracy": {"north": 0.67, "east": 0.33,
                                      "south": 1.0, "west": 0.0}}),
        _write(tmp_path, "g11_seed43_chat_demo.json",
               {"seed": 43, "accuracy": 0.42, "correct": 5, "total": 12,
                "per_word_accuracy": {"north": 0.33, "east": 0.67,
                                      "south": 0.33, "west": 0.33}}),
    ]
    s = aggregate(paths)
    assert "tier1_per_direction" in s
    pd = s["tier1_per_direction"]
    assert "north" in pd and "east" in pd and "south" in pd and "west" in pd
    assert abs(pd["north"]["mean"] - 0.50) < 0.01  # (0.67+0.33)/2
    assert pd["south"]["n_seeds"] == 2


def test_aggregate_synonym(tmp_path):
    """Synonym demo seed JSON has primary/synonym split."""
    paths = [
        _write(tmp_path, "g11_seed42_chat_synonym_demo.json",
               {"seed": 42, "accuracy": 0.25, "correct": 4, "total": 16,
                "primary_accuracy": 0.5, "synonym_accuracy": 0.0,
                "per_action_correct": {"N": 0, "E": 1, "S": 1, "W": 2},
                "per_action_total": {"N": 4, "E": 4, "S": 4, "W": 4}}),
        _write(tmp_path, "g11_seed43_chat_synonym_demo.json",
               {"seed": 43, "accuracy": 0.50, "correct": 8, "total": 16,
                "primary_accuracy": 0.75, "synonym_accuracy": 0.25,
                "per_action_correct": {"N": 2, "E": 2, "S": 2, "W": 2},
                "per_action_total": {"N": 4, "E": 4, "S": 4, "W": 4}}),
    ]
    s = aggregate(paths)
    assert s["demo_types"] == ["synonym"]
    assert "synonym_demo" in s
    sd = s["synonym_demo"]
    assert abs(sd["primary_acc_mean"] - 0.625) < 0.01  # (0.5+0.75)/2
    assert abs(sd["synonym_acc_mean"] - 0.125) < 0.01  # (0+0.25)/2


def test_aggregate_continual_field_aliases(tmp_path):
    """Continual demo accepts both old + new field naming."""
    paths = [
        # Old naming (consolidation_trainer-style)
        _write(tmp_path, "old_seed42.json",
               {"seed": 42, "primary_post_a": 0.4, "primary_post_b": 0.3,
                "retention_ratio": 0.75, "synonym_learning": 0.25}),
        # New naming (chat_continual_demo-style)
        _write(tmp_path, "new_seed43.json",
               {"seed": 43, "primary_a_acc": 0.5, "primary_b_acc": 0.4,
                "retention": 0.8, "synonym_acc": 0.3}),
    ]
    s = aggregate(paths)
    assert s["demo_types"] == ["continual"]
    assert "continual_demo" in s
    cd = s["continual_demo"]
    assert abs(cd["retention_mean"] - 0.775) < 0.01  # (0.75+0.8)/2
    assert cd["n_pass_above_80"] == 1  # only 0.8 passes >= 0.80


def test_aggregate_empty_raises(tmp_path):
    """Empty input raises ValueError."""
    with pytest.raises(ValueError, match="No result files"):
        aggregate([])
