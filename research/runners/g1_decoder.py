"""Decoder + loss helpers for G1: output spike counts -> class prediction, margin, metrics.

Pure NumPy; no sim or GPU dependencies.
"""

from __future__ import annotations

import numpy as np


def decode_prediction(spike_counts):
    """Argmax over per-output-neuron spike counts. Ties break to lowest index."""
    if spike_counts.size == 0:
        raise ValueError("spike_counts must be non-empty")
    return int(np.argmax(spike_counts))


def compute_margin(spike_counts, correct_class):
    """spikes[correct] - mean(spikes[others]).

    Positive margin = correct class outfires the average competitor.
    Graded signal that can improve before the argmax prediction flips.
    """
    if spike_counts.size < 2:
        return 0.0
    correct = float(spike_counts[correct_class])
    mask = np.ones_like(spike_counts, dtype=bool)
    mask[correct_class] = False
    others = spike_counts[mask]
    if others.size == 0:
        return 0.0
    return correct - float(others.mean())


def compute_metrics(spike_counts, labels):
    """Aggregate accuracy, mean margin, confusion matrix over a batch.

    Args:
        spike_counts: (n_examples, n_classes) int array
        labels:       (n_examples,) int array of true classes

    Returns:
        dict with accuracy, mean_margin, margins (per-example), predictions,
        confusion (n_classes x n_classes), and n.
    """
    assert spike_counts.ndim == 2
    n, K = spike_counts.shape
    preds = np.argmax(spike_counts, axis=1)
    acc = float((preds == labels).mean()) if n > 0 else 0.0
    margins = np.array([compute_margin(spike_counts[i], int(labels[i])) for i in range(n)])
    confusion = np.zeros((K, K), dtype=np.int32)
    for y_true, y_pred in zip(labels, preds):
        confusion[int(y_true), int(y_pred)] += 1
    return {
        "accuracy": acc,
        "mean_margin": float(margins.mean()) if n > 0 else 0.0,
        "margins": margins,
        "predictions": preds,
        "confusion": confusion,
        "n": n,
    }
