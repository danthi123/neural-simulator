"""G2 runner smoke test: 1 epoch, few examples. Does not assert convergence."""
import json
from pathlib import Path

import pytest


def test_g2_runner_smoke(tmp_path):
    pytest.importorskip("cupy")
    pytest.importorskip("sklearn")
    from research.runners.g2_runner import run_g2
    from research.datasets.tiny_patterns import build_dataset

    ds_path = tmp_path / "ds.npz"
    build_dataset(ds_path, seed=0xD47A5E7, K=4, n_features=64,
                  n_train=20, n_test=8, noise_sigma=4.0)

    out = tmp_path / "result.json"
    result = run_g2(
        dataset_path=str(ds_path),
        out_path=str(out),
        seed=42,
        n_epochs=1,
        max_train_per_epoch=10,
        max_test_per_epoch=8,
        verbose=False,
    )

    assert out.exists()
    data = json.load(open(out))
    assert data["seed"] == 42
    assert data["n_plastic_synapses"] > 0
    assert data["n_frozen_synapses"] > 0
    assert len(data["epochs"]) == 1
    ep = data["epochs"][0]
    for k in ("epoch", "train_accuracy", "test_accuracy",
              "plastic_weight_min", "plastic_weight_max",
              "mean_hidden_rate_hz_train"):
        assert k in ep
    assert 0.0 <= ep["test_accuracy"] <= 1.0
