"""Local, pure-numpy unit test for `_sail_learn_s2_templates` (research/runners/
_vision_lindiscrim_readout_derisk.py's SAILnet explicit-decorrelation S2-template-learning lever,
`--s2-learn sail`, 2026-09-17). No SimulationBridge, no CoreSimConfig -- sub-second.

Drives the function on a small synthetic "Foldiak (1990) bars problem" patch pool (the classic testbed
for Hebbian + anti-Hebbian factorial-code learning: a few independent sparse causes, more template units
than causes so at least one is genuinely REDUNDANT at random init) and asserts the four properties named
in the build's own verification spec:
  (a) it runs and returns a finite (n_S2, D) W;
  (b) mean_pairwise_cosine_abs DROPS vs the frozen-random init (the explicit decorrelation works);
  (c) the lateral matrix L is non-negative and grows (nonzero) -- i.e. the anti-Hebbian term actually
      engaged, not just the feedforward competitive term BCM already had;
  (d) a no-learning control (alpha_w=alpha_l=alpha_theta=0) returns W unchanged.
"""
from __future__ import annotations

import numpy as np

from research.runners._vision_lindiscrim_readout_derisk import (
    _l2n,
    _mean_pairwise_cosine_abs,
    _sail_learn_s2_templates,
)


def _make_bars_patches(n_side, N, p_bar, noise, seed):
    """Foldiak (1990) bars problem: n_side x n_side grid, 2*n_side possible bars (rows+cols); each patch
    independently turns bars ON w.p. p_bar and ORs their pixel masks -- sparse independent causes."""
    rng = np.random.default_rng(seed)
    n_bars = 2 * n_side
    bar_masks = np.zeros((n_bars, n_side, n_side), dtype=np.float64)
    for i in range(n_side):
        bar_masks[i, i, :] = 1.0
        bar_masks[n_side + i, :, i] = 1.0
    bar_masks = bar_masks.reshape(n_bars, -1)
    active = rng.random((N, n_bars)) < p_bar
    X = active.astype(np.float64) @ bar_masks
    X = np.clip(X, 0.0, 1.0)
    X = X + noise * rng.standard_normal(X.shape)
    X = np.clip(X, 0.0, None)
    return _l2n(X, axis=1).astype(np.float64)


def _init_bank(n_S2, D, seed):
    """Same nonneg-random family as the runner's own `_init_templates` (abs(randn)+eps, unit-L2 rows) --
    this is also WHY the frozen-random baseline's own mean_pairwise_cosine_abs sits well above 0 (~0.65-0.7,
    not the near-0 a signed random bank would show): all-positive-orthant vectors are intrinsically
    correlated, which is exactly the redundancy this lever must overcome, not an artifact of this test."""
    rng = np.random.default_rng(seed)
    W0 = np.abs(rng.standard_normal((n_S2, D))).astype(np.float64) + 0.01
    return _l2n(W0, axis=1)


# n_S2=9 > n_bars=8 -- ONE genuinely redundant template beyond the 8 independent causes, the minimal
# case where BCM's competitive_frac alone (no explicit pairwise decorrelation) leaves a near-duplicate
# uncorrected (the residual this lever targets).
_N_SIDE = 4
_D = _N_SIDE * _N_SIDE
_N_BARS = 2 * _N_SIDE
_N_S2 = _N_BARS + 1
_TARGET_P = 2.0 / _N_S2
_N = 200
_EPOCHS = 60
_LCA_ITERS = 10


def _patches_and_bank():
    X = _make_bars_patches(_N_SIDE, _N, p_bar=1.0 / _N_BARS, noise=0.03, seed=1)
    W0 = _init_bank(_N_S2, _D, seed=11)
    return X, W0


def test_sail_runs_and_returns_finite_shapes():
    X, W0 = _patches_and_bank()
    W, L, theta, diag = _sail_learn_s2_templates(
        W0, X, alpha_w=0.02, alpha_l=0.02, alpha_theta=0.02,
        target_p=_TARGET_P, competitive_frac=_TARGET_P, lca_iters=_LCA_ITERS,
        epochs=_EPOCHS, renorm=1, seed=42)
    assert W.shape == (_N_S2, _D)
    assert L.shape == (_N_S2, _N_S2)
    assert theta.shape == (_N_S2,)
    assert np.all(np.isfinite(W))
    assert np.all(np.isfinite(L))
    assert np.all(np.isfinite(theta))
    assert "mean_pairwise_cosine_abs" in diag
    assert "mean_pairwise_cosine_abs_init" in diag


def test_sail_decorrelates_below_frozen_random_baseline():
    """(b): the explicit anti-Hebbian lateral term must leave the bank LESS mutually correlated than the
    frozen-random init it started from -- the diversity metric this whole lever is about."""
    X, W0 = _patches_and_bank()
    _W, _L, _theta, diag = _sail_learn_s2_templates(
        W0, X, alpha_w=0.02, alpha_l=0.02, alpha_theta=0.02,
        target_p=_TARGET_P, competitive_frac=_TARGET_P, lca_iters=_LCA_ITERS,
        epochs=_EPOCHS, renorm=1, seed=42)
    init_cos = diag["mean_pairwise_cosine_abs_init"]
    final_cos = diag["mean_pairwise_cosine_abs"]
    assert final_cos < init_cos - 0.02, (
        f"expected a real drop in mean_pairwise_cosine_abs (init={init_cos:.4f}, final={final_cos:.4f})")


def test_sail_lateral_matrix_nonnegative_and_grows():
    """(c): L must stay non-negative throughout (Dale's-law-rectified inhibitory lateral synapses) and
    must actually GROW (the anti-Hebbian term engaging, not staying at its all-zero init)."""
    X, W0 = _patches_and_bank()
    _W, L, _theta, diag = _sail_learn_s2_templates(
        W0, X, alpha_w=0.02, alpha_l=0.02, alpha_theta=0.02,
        target_p=_TARGET_P, competitive_frac=_TARGET_P, lca_iters=_LCA_ITERS,
        epochs=_EPOCHS, renorm=1, seed=42)
    assert np.all(L >= 0.0)
    assert np.all(np.diag(L) == 0.0)
    assert diag["L_max"] > 0.0
    assert diag["L_mean"] > 0.0


def test_sail_no_learning_control_leaves_templates_unchanged():
    """(d): alpha_w=alpha_l=alpha_theta=0 must be a true no-op on W (and L must stay at its all-zero
    init) -- the control that isolates "the function ran" from "the function actually learned"."""
    X, W0 = _patches_and_bank()
    W, L, theta, _diag = _sail_learn_s2_templates(
        W0, X, alpha_w=0.0, alpha_l=0.0, alpha_theta=0.0,
        target_p=_TARGET_P, competitive_frac=_TARGET_P, lca_iters=_LCA_ITERS,
        epochs=3, renorm=0, seed=42)
    assert np.allclose(W, W0.astype(np.float32), atol=1e-6)
    assert np.allclose(L, 0.0)
    assert np.allclose(theta, 0.0)
