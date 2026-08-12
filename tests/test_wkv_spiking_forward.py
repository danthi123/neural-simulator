"""Regression guards for the WKV spiking-forward de-risk runner.

These are CPU-only tests for the runner wiring. The full RF-bridge verdict remains
the recorded GPU six-seed run; here we pin the cache discipline that prevented the
seed-43 id-reuse aliasing failure from returning.
"""
from __future__ import annotations

import gc
from types import SimpleNamespace

import numpy as np


def test_full_runner_reuses_rf_caches_across_all_seeds(monkeypatch):
    from research.runners import _wkv_spiking_forward_derisk as runner

    monkeypatch.setattr(
        runner,
        "build_numpy_model_from_ckpt",
        lambda _path: (
            {"vocab_size": 8},
            {
                "total_params_M": 0.01,
                "d_model": 4,
                "n_layers": 1,
                "vocab_size": 8,
            },
        ),
    )

    seen = []

    def fake_run_full_one_seed(_model, _info, seed, _args, bridges=None, w32=None):
        assert bridges is not None
        assert w32 is not None
        bridges.setdefault("bridge_marker", object())
        w32.setdefault("weight_marker", np.array([seed], dtype=np.float32))
        seen.append(
            (
                seed,
                id(bridges),
                id(w32),
                id(bridges["bridge_marker"]),
                id(w32["weight_marker"]),
            )
        )
        return {
            "seed": seed,
            "ann_ppl": 10.0,
            "spiking_ppl": 10.0,
            "ppl_ratio": 1.0,
            "logit_fid_spearman": 1.0,
            "logit_fid_cosine": 1.0,
            "n_windows": 1,
            "n_logit_pos": 1,
        }

    monkeypatch.setattr(runner, "run_full_one_seed", fake_run_full_one_seed)

    args = SimpleNamespace(
        ckpt="dummy.pt",
        backend="rf-bridge",
        nsteps=8,
        read_eps=5e-7,
        block_size=8,
        n_windows=1,
        n_logit_pos=1,
        seeds=[42, 43, 44],
        seed=42,
    )
    out = runner.run_full(args)

    assert [s[0] for s in seen] == [42, 43, 44]
    assert len({s[1] for s in seen}) == 1, "RF bridge cache must persist across seeds"
    assert len({s[2] for s in seen}) == 1, "float32 weight cache must persist across seeds"
    assert len({s[3] for s in seen}) == 1
    assert len({s[4] for s in seen}) == 1
    assert out["verdict_go_ppl_ratio<=1.05_and_fid>=0.99"] is True


def test_rf_graded_read_keeps_float32_weight_casts_alive(monkeypatch):
    from research.runners import _wkv_spiking_forward_derisk as runner
    from research.runners import _genseq_loopstep3_full_genf_generate_derisk as genf
    from research.runners import _genseq_loopstep3_rf_probe as rf_probe

    def fake_project(_bridge, W, h, *, period, nsteps, lam, measure_err=False):
        del period, nsteps, lam, measure_err
        return h.astype(np.float64) @ W.astype(np.float64), 0.0

    monkeypatch.setattr(genf, "_rf_project_seq", fake_project)
    monkeypatch.setattr(rf_probe, "_build_rf_bridge", lambda _n, seed=42: object())

    bridges = {}
    shared_w32 = {}
    read = runner.make_rf_graded_read(
        bridges,
        period=100000,
        nsteps=8,
        lam=0.0,
        err_accum=[],
        w32=shared_w32,
    )

    h = np.array([[1.0, -2.0]], dtype=np.float64)
    W1 = np.array([[1.0, 0.5], [-0.25, 2.0]], dtype=np.float64)
    np.testing.assert_allclose(read(W1, h), h @ W1)

    W1_key = id(W1)
    W1_cast_id = id(shared_w32[W1_key])
    np.testing.assert_allclose(read(W1, h), h @ W1)
    assert id(shared_w32[W1_key]) == W1_cast_id

    del W1
    gc.collect()

    W2 = np.array([[0.0, 3.0], [4.0, -1.0]], dtype=np.float64)
    np.testing.assert_allclose(read(W2, h), h @ W2)
    assert any(id(cast) == W1_cast_id for cast in shared_w32.values())
    assert len(shared_w32) == 2
