"""Smoke test for the G1 runner: 1 epoch, few examples.

Verifies the end-to-end pipeline runs and produces the expected JSON schema.
Does NOT assert convergence — that's for the findings doc.
"""
import json
from pathlib import Path

import pytest


def test_g1_runner_smoke(tmp_path):
    pytest.importorskip("cupy")
    from research.runners.g1_runner import run_g1
    from research.datasets.tiny_patterns import build_dataset

    ds_path = tmp_path / "ds.npz"
    build_dataset(ds_path, seed=0xD47A5E7, K=4, n_features=64,
                  n_train=20, n_test=8, noise_sigma=4.0)

    out = tmp_path / "result.json"
    result = run_g1(
        dataset_path=str(ds_path),
        out_path=str(out),
        seed=42,
        n_epochs=1,
        max_train_per_epoch=10,
        max_test_per_epoch=8,
        verbose=False,
    )

    assert out.exists()
    with open(out) as f:
        data = json.load(f)

    assert data["seed"] == 42
    assert data["n_epochs"] == 1
    assert "epochs" in data and len(data["epochs"]) == 1
    epoch = data["epochs"][0]
    for key in ("epoch", "train_accuracy", "test_accuracy", "mean_margin_test",
                "mean_weight", "weight_std", "time_seconds"):
        assert key in epoch, f"Missing key: {key}"
    assert 0.0 <= epoch["test_accuracy"] <= 1.0
    assert 0.0 <= epoch["train_accuracy"] <= 1.0
