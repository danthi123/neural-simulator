"""Smoke tests for G3 runner: fresh run and resume run both produce
full-length result objects with consistent schema.

The deeper "resumed trajectory matches clean trajectory" test is a separate
full-run experiment in research/findings/, not part of the unit suite.
"""
import json
from pathlib import Path

import pytest


def test_g3_fresh_run_smoke(tmp_path):
    pytest.importorskip("cupy")
    pytest.importorskip("sklearn")
    from research.runners.g3_runner import run_g3
    from research.datasets.tiny_patterns import build_dataset

    ds = tmp_path / "ds.npz"
    build_dataset(ds, seed=0xD47A5E7, K=4, n_features=64,
                  n_train=20, n_test=8, noise_sigma=4.0)

    out = tmp_path / "fresh.json"
    r = run_g3(
        dataset_path=str(ds),
        out_path=str(out),
        seed=42,
        n_epochs=2,
        max_train_per_epoch=10,
        max_test_per_epoch=8,
        verbose=False,
    )
    assert len(r["epochs"]) == 2
    assert r["resumed_from"] is None


def test_g3_save_and_resume_smoke(tmp_path):
    pytest.importorskip("cupy")
    pytest.importorskip("sklearn")
    pytest.importorskip("h5py")
    from research.runners.g3_runner import run_g3
    from research.datasets.tiny_patterns import build_dataset

    ds = tmp_path / "ds.npz"
    build_dataset(ds, seed=0xD47A5E7, K=4, n_features=64,
                  n_train=20, n_test=8, noise_sigma=4.0)

    ckpt_prefix = str(tmp_path / "ckpt_ep0")
    out_phase1 = tmp_path / "phase1.json"
    r1 = run_g3(
        dataset_path=str(ds),
        out_path=str(out_phase1),
        seed=42,
        n_epochs=1,
        save_after=0,
        checkpoint_prefix=ckpt_prefix,
        max_train_per_epoch=10,
        max_test_per_epoch=8,
        verbose=False,
    )
    assert len(r1["epochs"]) == 1
    assert Path(f"{ckpt_prefix}.simstate.h5").exists()
    assert Path(f"{ckpt_prefix}.g3.json").exists()

    out_phase2 = tmp_path / "phase2.json"
    r2 = run_g3(
        dataset_path=str(ds),
        out_path=str(out_phase2),
        seed=42,
        n_epochs=3,
        start_from=ckpt_prefix,
        max_train_per_epoch=10,
        max_test_per_epoch=8,
        verbose=False,
    )
    # Resumed run appends epochs 1, 2 to the checkpoint's epoch 0.
    assert len(r2["epochs"]) == 3
    assert r2["epochs"][0]["epoch"] == 0
    assert r2["epochs"][-1]["epoch"] == 2
    assert r2["resumed_from"] == ckpt_prefix
