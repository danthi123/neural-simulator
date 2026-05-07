"""Tests for Phase 1.2 Tier 2.3 phrase_trainer module.

Tests the training buffer construction (deterministic, schema-correct)
without exercising the GPU training loop. The training loop itself is
exercised by the smoke runner.
"""
import numpy as np
import pytest


def test_buffer_counts_match_args():
    """Buffer should have exactly the requested number of each trial type."""
    from research.runners.phrase_trainer import build_phrase_buffer
    rng = np.random.default_rng(42)
    buf = build_phrase_buffer(
        n_phrase_events=200,
        n_direction_only_events=100,
        n_verb_only_events=30,
        rng=rng,
    )
    counts = {}
    for t in buf:
        counts[t["type"]] = counts.get(t["type"], 0) + 1
    assert counts.get("phrase") == 200
    assert counts.get("direction_only") == 100
    assert counts.get("verb_only") == 30
    assert len(buf) == 330


def test_buffer_phrase_trial_schema():
    """Phrase trials must have verb + direction + action keys."""
    from research.runners.phrase_trainer import build_phrase_buffer
    rng = np.random.default_rng(42)
    buf = build_phrase_buffer(40, 0, 0, rng)
    for trial in buf:
        if trial["type"] == "phrase":
            assert "verb" in trial
            assert "direction" in trial
            assert "action" in trial
            # Action must match direction
            from research.runners.phrase_trainer import DIRECTION_TO_ACTION
            assert trial["action"] == DIRECTION_TO_ACTION[trial["direction"]]


def test_buffer_direction_only_schema():
    """Direction-only trials must have direction + action, no verb."""
    from research.runners.phrase_trainer import build_phrase_buffer
    rng = np.random.default_rng(42)
    buf = build_phrase_buffer(0, 40, 0, rng)
    for trial in buf:
        assert trial["type"] == "direction_only"
        assert "direction" in trial
        assert "action" in trial
        assert "verb" not in trial


def test_buffer_verb_only_schema():
    """Verb-only trials have verb but action=None (no motor target)."""
    from research.runners.phrase_trainer import build_phrase_buffer
    rng = np.random.default_rng(42)
    buf = build_phrase_buffer(0, 0, 30, rng)
    for trial in buf:
        assert trial["type"] == "verb_only"
        assert trial["verb"] == "go"
        assert trial["action"] is None


def test_buffer_phrase_trials_distribute_across_directions():
    """The 4 directions should get roughly equal phrase trial counts."""
    from research.runners.phrase_trainer import (
        build_phrase_buffer, DIRECTIONS,
    )
    rng = np.random.default_rng(42)
    buf = build_phrase_buffer(200, 0, 0, rng)
    dir_counts = {d: 0 for d in DIRECTIONS}
    for t in buf:
        if t["type"] == "phrase":
            dir_counts[t["direction"]] += 1
    # Should be exactly 50 each (200 / 4)
    for d, c in dir_counts.items():
        assert c == 50


def test_buffer_deterministic_with_seed():
    """Same seed -> same buffer order (modulo internal RNG state)."""
    from research.runners.phrase_trainer import build_phrase_buffer
    buf1 = build_phrase_buffer(20, 10, 5, np.random.default_rng(42))
    buf2 = build_phrase_buffer(20, 10, 5, np.random.default_rng(42))
    # Buffers should be identical when same seed used
    assert len(buf1) == len(buf2)
    for a, b in zip(buf1, buf2):
        assert a == b


def test_buffer_zero_events():
    """Edge case: 0 events of each type returns empty buffer."""
    from research.runners.phrase_trainer import build_phrase_buffer
    rng = np.random.default_rng(42)
    buf = build_phrase_buffer(0, 0, 0, rng)
    assert buf == []


def test_buffer_uneven_phrase_count():
    """When phrase_events not divisible by 4, integer division floor.
    (200 / 4 = 50; 199 / 4 = 49 -> 196 phrase trials.)"""
    from research.runners.phrase_trainer import build_phrase_buffer
    rng = np.random.default_rng(42)
    buf = build_phrase_buffer(199, 0, 0, rng)
    # 199 // 4 = 49 per direction; total = 196
    n_phrase = sum(1 for t in buf if t["type"] == "phrase")
    assert n_phrase == 196


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
