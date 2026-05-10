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


# ─────────────────────────────────────────────────────────────────────────
# chat_learn_demo branch (Track 3 online vocab learning, 2026-05-09)
# ─────────────────────────────────────────────────────────────────────────


def test_aggregate_chat_learn_demo_basic(tmp_path):
    """chat_learn_demo JSONs route to the chat_learn aggregation block."""
    paths = [
        _write(tmp_path, "g11_seed42_chat_learn.json", {
            "seed": 42,
            "demo_kind": "chat_learn_demo",
            "accuracy": 0.875,
            "primary_baseline_accuracy": 1.00,
            "primary_post_learn_accuracy": 0.875,
            "primary_retention_ratio": 0.875,
            "learn_binding_rate": 1.0,
            "go": True,
            "verdict": "GO",
            "new_words": [["ahead", "N"], ["back", "S"]],
            "learn_results": [
                {"word": "ahead", "target": "N", "predicted": "N",
                 "bound_ok": True, "confidence": 4.2,
                 "delta_counts": {"N": 12, "E": 1, "S": 0, "W": 1}},
                {"word": "back", "target": "S", "predicted": "S",
                 "bound_ok": True, "confidence": 3.7,
                 "delta_counts": {"N": 0, "E": 0, "S": 8, "W": 1}},
            ],
        }),
        _write(tmp_path, "g11_seed43_chat_learn.json", {
            "seed": 43,
            "demo_kind": "chat_learn_demo",
            "accuracy": 0.625,
            "primary_baseline_accuracy": 0.75,
            "primary_post_learn_accuracy": 0.625,
            "primary_retention_ratio": 0.83,
            "learn_binding_rate": 0.5,
            "go": True,
            "verdict": "GO",
            "new_words": [["ahead", "N"], ["back", "S"]],
            "learn_results": [
                {"word": "ahead", "target": "N", "predicted": "N",
                 "bound_ok": True},
                {"word": "back", "target": "S", "predicted": "E",
                 "bound_ok": False},
            ],
        }),
        _write(tmp_path, "g11_seed44_chat_learn.json", {
            "seed": 44,
            "demo_kind": "chat_learn_demo",
            "accuracy": 0.50,
            "primary_baseline_accuracy": 0.875,
            "primary_post_learn_accuracy": 0.50,
            "primary_retention_ratio": 0.57,
            "learn_binding_rate": 1.0,
            "go": False,
            "verdict": "NO-GO",
        }),
    ]
    s = aggregate(paths)
    assert s["n_seeds"] == 3
    assert s["demo_types"] == ["chat_learn"]
    assert "chat_learn_demo" in s
    cl = s["chat_learn_demo"]
    assert cl["n_seeds"] == 3
    # Binding rate mean: (1.0 + 0.5 + 1.0) / 3 = 0.833
    assert abs(cl["binding_rate_mean"] - 0.833) < 0.01
    # Primary retention mean: (0.875 + 0.83 + 0.57) / 3 = 0.758
    assert abs(cl["primary_retention_mean"] - 0.758) < 0.01
    # 3 seeds passed binding (>=0.5), 2 passed retention (>=0.8)
    assert cl["n_binding_pass"] == 3
    assert cl["n_retention_pass"] == 2
    assert cl["n_go_verdict"] == 2  # seeds 42 + 43


def test_chat_learn_demo_distinguishes_from_tier1(tmp_path):
    """A run with demo_kind set goes to chat_learn, NOT tier1."""
    paths = [
        _write(tmp_path, "g11_seed42_chat_learn.json", {
            "seed": 42,
            "demo_kind": "chat_learn_demo",
            "accuracy": 0.5,
            "primary_baseline_accuracy": 0.5,
            "primary_post_learn_accuracy": 0.5,
            "primary_retention_ratio": 1.0,
            "learn_binding_rate": 0.5,
            "go": True,
            "verdict": "GO",
        }),
    ]
    s = aggregate(paths)
    assert s["demo_types"] == ["chat_learn"]
    # Make sure it didn't accidentally route to tier1 demo type
    assert "chat_learn_demo" in s
    assert "tier1_per_direction" not in s


def test_chat_learn_demo_handles_missing_optional_fields(tmp_path):
    """Old chat_learn JSONs without all fields don't crash the aggregator."""
    paths = [
        _write(tmp_path, "g11_seed42_chat_learn.json", {
            "seed": 42,
            "demo_kind": "chat_learn_demo",
            # Only the required-for-tag field; everything else default-0
        }),
    ]
    s = aggregate(paths)
    assert s["n_seeds"] == 1
    cl = s["chat_learn_demo"]
    assert cl["binding_rate_mean"] == 0.0
    assert cl["primary_retention_mean"] == 0.0
    assert cl["n_go_verdict"] == 0


# ─────────────────────────────────────────────────────────────────────────
# chat_speak_demo branch (Track 3 layer 4 :speak generative decoder, 2026-05-09)
# ─────────────────────────────────────────────────────────────────────────


def test_aggregate_chat_speak_demo_basic(tmp_path):
    """chat_speak_demo JSONs route to the chat_speak aggregation block.

    Reproduces the 6-seed Track 3 v2 multi-seed validation result
    (A2W mean 58.3%, 5/6 above-50%) at three-seed scale.
    """
    paths = [
        _write(tmp_path, "g11_seed42_chat_speak_demo.json", {
            "seed": 42,
            "demo_kind": "chat_speak_demo",
            "accuracy": 0.125,            # W2A regression
            "speak_accuracy": 0.75,       # A2W
            "speak_correct": 3,
            "speak_total": 4,
            "per_word_accuracy": {"north": 0.0, "east": 0.5,
                                   "south": 0.0, "west": 0.0},
            "speak_results": [
                {"target_action": "N", "predicted_word": "north",
                 "correct": True},
                {"target_action": "E", "predicted_word": "east",
                 "correct": True},
                {"target_action": "S", "predicted_word": "south",
                 "correct": True},
                {"target_action": "W", "predicted_word": "east",
                 "correct": False},
            ],
        }),
        _write(tmp_path, "g11_seed43_chat_speak_demo.json", {
            "seed": 43,
            "demo_kind": "chat_speak_demo",
            "accuracy": 0.50,
            "speak_accuracy": 0.75,
            "speak_correct": 3,
            "speak_total": 4,
            "speak_results": [
                {"target_action": "N", "predicted_word": "north",
                 "correct": True},
                {"target_action": "E", "predicted_word": "east",
                 "correct": True},
                {"target_action": "S", "predicted_word": "south",
                 "correct": True},
                {"target_action": "W", "predicted_word": "east",
                 "correct": False},
            ],
        }),
        _write(tmp_path, "g11_seed102_chat_speak_demo.json", {
            "seed": 102,
            "demo_kind": "chat_speak_demo",
            "accuracy": 0.25,
            "speak_accuracy": 0.25,         # only outlier in real run
            "speak_correct": 1,
            "speak_total": 4,
            "speak_results": [
                {"target_action": "N", "predicted_word": "west",
                 "correct": False},
                {"target_action": "E", "predicted_word": "south",
                 "correct": False},
                {"target_action": "S", "predicted_word": "north",
                 "correct": False},
                {"target_action": "W", "predicted_word": "west",
                 "correct": True},
            ],
        }),
    ]
    s = aggregate(paths)
    assert s["n_seeds"] == 3
    assert s["demo_types"] == ["chat_speak"]
    assert "chat_speak_demo" in s
    cs = s["chat_speak_demo"]
    # A2W mean: (0.75 + 0.75 + 0.25) / 3 = 0.583
    assert abs(cs["speak_accuracy_mean"] - 0.583) < 0.01
    # 2/3 seeds above chance (>0.25), 2/3 at >=50%
    assert cs["n_speak_above_chance"] == 2
    assert cs["n_speak_above_50pct"] == 2
    # W2A surfaced separately
    assert abs(cs["w2a_accuracy_mean"] - 0.292) < 0.01  # (0.125+0.5+0.25)/3
    # Per-direction A2W: N hits 2/3, E hits 2/3, S hits 2/3, W hits 1/3
    pd = cs["per_direction_a2w_mean"]
    assert abs(pd["N"] - 2/3) < 0.01
    assert abs(pd["W"] - 1/3) < 0.01


def test_chat_speak_demo_distinguishes_from_tier1(tmp_path):
    """chat_speak_demo with demo_kind goes to chat_speak, NOT tier1."""
    paths = [
        _write(tmp_path, "g11_seed42_chat_speak_demo.json", {
            "seed": 42,
            "demo_kind": "chat_speak_demo",
            "accuracy": 0.5,
            "speak_accuracy": 0.5,
            "speak_correct": 2,
            "speak_total": 4,
            "speak_results": [
                {"target_action": "N", "predicted_word": "north",
                 "correct": True},
            ],
        }),
    ]
    s = aggregate(paths)
    assert s["demo_types"] == ["chat_speak"]
    assert "chat_speak_demo" in s
    # Should NOT have leaked into tier1 (no per_word_accuracy aggregation)
    assert "tier1_per_direction" not in s


def test_chat_speak_demo_handles_missing_speak_results(tmp_path):
    """Older chat_speak JSONs missing speak_results don't crash."""
    paths = [
        _write(tmp_path, "g11_seed42_chat_speak_demo.json", {
            "seed": 42,
            "demo_kind": "chat_speak_demo",
            "accuracy": 0.25,
            "speak_accuracy": 0.50,
            "speak_correct": 2,
            "speak_total": 4,
            # No speak_results field
        }),
    ]
    s = aggregate(paths)
    assert s["n_seeds"] == 1
    cs = s["chat_speak_demo"]
    assert cs["speak_accuracy_mean"] == 0.5
    # per_direction_a2w_mean should be empty when no speak_results
    assert cs["per_direction_a2w_mean"] == {}
