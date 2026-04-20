"""Tests for G1 decoder + metric helpers."""
import numpy as np
import pytest

from research.runners.g1_decoder import decode_prediction, compute_margin, compute_metrics


def test_decode_prediction_argmax():
    counts = np.array([3, 10, 5, 2], dtype=np.int32)
    assert decode_prediction(counts) == 1


def test_decode_prediction_ties_deterministic():
    counts = np.array([5, 5, 5, 5], dtype=np.int32)
    assert decode_prediction(counts) == 0


def test_margin_positive_when_correct():
    counts = np.array([1, 10, 2, 3], dtype=np.int32)
    # 10 - mean([1, 2, 3]) = 10 - 2 = 8
    assert compute_margin(counts, correct_class=1) == pytest.approx(8.0)


def test_margin_negative_when_wrong():
    counts = np.array([10, 1, 2, 3], dtype=np.int32)
    # 1 - mean([10, 2, 3]) = 1 - 5 = -4
    assert compute_margin(counts, correct_class=1) == pytest.approx(-4.0)


def test_metrics_batch_accuracy_and_margin():
    counts = np.array([
        [3, 10, 5, 2],   # pred=1, true=1 ok
        [8, 2, 1, 0],    # pred=0, true=0 ok
        [1, 1, 9, 2],    # pred=2, true=3 no
        [0, 0, 0, 5],    # pred=3, true=3 ok
        [2, 5, 3, 4],    # pred=1, true=2 no
    ], dtype=np.int32)
    y = np.array([1, 0, 3, 3, 2], dtype=np.int32)
    m = compute_metrics(counts, y)
    assert m["accuracy"] == pytest.approx(3.0 / 5.0)
    assert m["n"] == 5
    assert "mean_margin" in m
    assert int(m["confusion"].sum()) == 5


def test_compute_margin_silent_network():
    counts = np.zeros(4, dtype=np.int32)
    assert decode_prediction(counts) == 0
    assert compute_margin(counts, correct_class=2) == 0.0
