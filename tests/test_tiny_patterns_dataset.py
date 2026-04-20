"""Tests for TinyPatternDataset — G1 synthetic Poisson-rate dataset."""
import numpy as np
import pytest

from research.datasets.tiny_patterns import TinyPatternDataset, build_dataset


def test_build_dataset_stable_with_same_seed(tmp_path):
    out = tmp_path / "tp.npz"
    build_dataset(out, seed=0xD47A5E7, K=4, n_features=64,
                  n_train=200, n_test=50, noise_sigma=4.0,
                  rate_min=1.0, rate_max=40.0)

    ds1 = TinyPatternDataset.load(out)
    ds2 = TinyPatternDataset.load(out)

    assert ds1.X_train.shape == (200, 64)
    assert ds1.y_train.shape == (200,)
    assert ds1.X_test.shape == (50, 64)
    assert ds1.y_test.shape == (50,)
    assert ds1.X_train.dtype == np.float32
    assert ds1.y_train.dtype == np.int32
    assert np.array_equal(ds1.X_train, ds2.X_train)
    assert np.array_equal(ds1.y_train, ds2.y_train)


def test_rates_clipped_to_range(tmp_path):
    out = tmp_path / "tp.npz"
    build_dataset(out, seed=0xD47A5E7, K=4, n_features=64,
                  n_train=500, n_test=100, noise_sigma=20.0,
                  rate_min=1.0, rate_max=40.0)
    ds = TinyPatternDataset.load(out)
    assert ds.X_train.min() >= 1.0
    assert ds.X_train.max() <= 40.0
    assert ds.X_test.min() >= 1.0
    assert ds.X_test.max() <= 40.0


def test_classes_well_separated(tmp_path):
    """Class means should be distinguishable: min pairwise L2 distance > threshold."""
    out = tmp_path / "tp.npz"
    build_dataset(out, seed=0xD47A5E7, K=4, n_features=64,
                  n_train=400, n_test=100, noise_sigma=4.0,
                  rate_min=1.0, rate_max=40.0)
    ds = TinyPatternDataset.load(out)

    class_means = np.zeros((4, 64), dtype=np.float32)
    for k in range(4):
        class_means[k] = ds.X_train[ds.y_train == k].mean(axis=0)

    min_dist = float('inf')
    for i in range(4):
        for j in range(i + 1, 4):
            d = float(np.linalg.norm(class_means[i] - class_means[j]))
            min_dist = min(min_dist, d)
    assert min_dist > 30.0, (
        f"Classes not separated: min pairwise L2 distance = {min_dist:.2f}"
    )


def test_labels_balanced(tmp_path):
    out = tmp_path / "tp.npz"
    build_dataset(out, seed=0xD47A5E7, K=4, n_features=64,
                  n_train=200, n_test=50, noise_sigma=4.0,
                  rate_min=1.0, rate_max=40.0)
    ds = TinyPatternDataset.load(out)
    for k in range(4):
        assert (ds.y_train == k).sum() >= 40
        assert (ds.y_test == k).sum() >= 8


def test_metadata_roundtrip(tmp_path):
    out = tmp_path / "tp.npz"
    build_dataset(out, seed=0xD47A5E7, K=4, n_features=64,
                  n_train=200, n_test=50, noise_sigma=4.0,
                  rate_min=1.0, rate_max=40.0)
    ds = TinyPatternDataset.load(out)
    assert ds.metadata["seed"] == 0xD47A5E7
    assert ds.metadata["K"] == 4
    assert ds.metadata["n_features"] == 64
    assert ds.metadata["noise_sigma"] == pytest.approx(4.0)
