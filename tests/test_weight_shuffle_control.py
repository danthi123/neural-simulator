"""Tests for the weight-shuffle DEPENDENCY control (tools.lab.shuffle_preserving_marginal / dependency_control).

The control is the distribution-preserving falsifier from Shiu & Sterne et al. 2024 (Nature 634:210-219): shuffle a
trained/structured weight matrix preserving its value distribution, re-run the function, and require it to COLLAPSE.
If the function survives, it rode on gross statistics, not learned structure.

Two layers:
  (A) FAST unit tests (always run) — the helper preserves the value multiset / per-row / per-col marginals exactly,
      is deterministic, refuses non-2-D input, and dependency_control both FIRES on a structure-dependent function
      AND correctly does NOT fire on one that reads only a preserved gross statistic (so the instrument can fail).
  (B) A SLOW circuit test (RUN_SLOW_TESTS=1) — the REAL gap#5 WHEN context->CA3 pathway `W_ctx`: the recency
      gradient works with the real weights and COLLAPSES under a distribution-preserving shuffle. CI-guarded and
      capable of failing (if the shuffled arm did not collapse, or the real arm did not work, it fails).
"""
import os

import numpy as np
import pytest

from tools.lab import shuffle_preserving_marginal, dependency_control


# ------------------------------------------------------------------------------------------------------------
# (A) FAST unit tests — the helper's invariants and the control's discriminating power.
# ------------------------------------------------------------------------------------------------------------
def _structured_matrix(seed=0, n=8, diag=4.0):
    """A strong DIAGONAL over small noise — the 'function' (diagonal alignment) rides on WHERE the mass sits."""
    W = np.random.default_rng(seed).random((n, n)) * 0.2
    np.fill_diagonal(W, W.diagonal() + diag)
    return W


def _diag_alignment(W):
    d = np.diag(W)
    off = W - np.diag(d)
    return float(d.mean() - off.sum() / (W.size - W.shape[0]))


def test_global_shuffle_preserves_value_multiset_exactly():
    W = _structured_matrix()
    out = shuffle_preserving_marginal(W, np.random.default_rng(1), mode="global")
    assert out.shape == W.shape
    assert np.array_equal(np.sort(out.reshape(-1)), np.sort(W.reshape(-1)))
    assert not np.array_equal(out, W)  # it actually permuted


def test_global_shuffle_returns_a_copy_not_a_view():
    W = _structured_matrix()
    out = shuffle_preserving_marginal(W, np.random.default_rng(1), mode="global")
    out[0, 0] = -999.0
    assert W[0, 0] != -999.0  # mutating the result must not touch the input


def test_per_row_preserves_each_row_sorted_multiset_and_row_sums():
    W = _structured_matrix(seed=2, n=10)
    out = shuffle_preserving_marginal(W, np.random.default_rng(3), mode="per_row")
    assert np.array_equal(np.sort(out, axis=1), np.sort(W, axis=1))  # per-row multiset preserved
    assert np.allclose(out.sum(axis=1), W.sum(axis=1))               # => row SUMS preserved (the marginal)
    # and the global multiset is preserved too (a per-row permutation is a global permutation)
    assert np.array_equal(np.sort(out.reshape(-1)), np.sort(W.reshape(-1)))


def test_per_col_preserves_each_col_sorted_multiset_and_col_sums():
    W = _structured_matrix(seed=4, n=9)
    out = shuffle_preserving_marginal(W, np.random.default_rng(5), mode="per_col")
    assert np.array_equal(np.sort(out, axis=0), np.sort(W, axis=0))
    assert np.allclose(out.sum(axis=0), W.sum(axis=0))


def test_shuffle_is_deterministic_at_a_fixed_rng_seed():
    W = _structured_matrix(seed=6)
    a = shuffle_preserving_marginal(W, np.random.default_rng(42), mode="global")
    b = shuffle_preserving_marginal(W, np.random.default_rng(42), mode="global")
    assert np.array_equal(a, b)
    # a DIFFERENT seed gives a different permutation (with overwhelming probability for a 64-element matrix)
    c = shuffle_preserving_marginal(W, np.random.default_rng(43), mode="global")
    assert not np.array_equal(a, c)


def test_shuffle_rejects_non_2d_input():
    with pytest.raises(ValueError):
        shuffle_preserving_marginal(np.arange(9), np.random.default_rng(0))
    with pytest.raises(ValueError):
        shuffle_preserving_marginal(np.zeros((2, 2, 2)), np.random.default_rng(0))


def test_shuffle_rejects_unknown_mode():
    with pytest.raises(ValueError):
        shuffle_preserving_marginal(_structured_matrix(), np.random.default_rng(0), mode="diagonal")


def test_dependency_control_collapses_a_structure_dependent_function():
    """The positive case: diagonal alignment rides on the ACTUAL weight positions, so a global shuffle collapses it."""
    W = _structured_matrix()
    dc = dependency_control(_diag_alignment, W, np.random.default_rng(1), n_shuffles=40, mode="global")
    assert dc["collapsed"] is True
    assert dc["real_score"] > dc["shuffled_p95"]          # clears the null's upper tail
    assert dc["real_score"] >= 3.0 * max(dc["shuffled_mean"], 1e-9) or dc["shuffled_mean"] <= 0
    assert dc["n_ge_real"] == 0                            # Shiu's "0/100": no shuffle matched the real score
    # schema
    for k in ("real_score", "shuffled_scores", "shuffled_mean", "shuffled_p95", "shuffled_max",
              "ratio_vs_mean", "margin_vs_p95", "n_shuffles", "mode", "collapsed"):
        assert k in dc
    assert len(dc["shuffled_scores"]) == 40


def test_dependency_control_does_NOT_collapse_a_gross_statistic_function():
    """The discriminating case (the instrument must be able to NOT fire): a function that reads ONLY the total sum —
    which a distribution-preserving shuffle holds EXACTLY fixed — must be reported as 'survived / rides on gross
    statistics', collapsed=False. A control that always fired would prove nothing."""
    W = _structured_matrix()
    dc = dependency_control(lambda M: float(M.sum()), W, np.random.default_rng(2), n_shuffles=40, mode="global")
    assert dc["collapsed"] is False
    assert dc["n_ge_real"] == 40                           # every shuffle equals the real score (sum is invariant)
    assert dc["real_score"] == pytest.approx(dc["shuffled_mean"])


def test_dependency_control_does_not_collapse_when_real_arm_is_null():
    """A both-arms-null comparison is a void, not a collapse: real=0 must give collapsed=False."""
    W = np.zeros((6, 6))
    dc = dependency_control(_diag_alignment, W, np.random.default_rng(3), n_shuffles=10, mode="global")
    assert dc["collapsed"] is False


# ------------------------------------------------------------------------------------------------------------
# (B) SLOW circuit test — the REAL gap#5 WHEN W_ctx pathway. CI-guarded via RUN_SLOW_TESTS.
#     Builds a full spiking substrate (~100s) + 20 distribution-preserving shuffle reads (~4s each).
# ------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(
    not os.environ.get("RUN_SLOW_TESTS"),
    reason="Slow (~3min: builds a spiking substrate + shuffle reads). Set RUN_SLOW_TESTS=1 to enable.",
)
def test_real_Wctx_recency_function_collapses_under_distribution_preserving_shuffle():
    """The demonstration with TEETH, CI-guarded. Everything is seeded (fixed substrate seed + fixed shuffle rng),
    so the numbers are deterministic run-to-run. Assertions use ROBUST margins (not the thin global p95), because
    the completion read is quantised and can shift by ~one held cell under a BLAS/numpy update.

    The PER_ROW shuffle (holds each cell's TOTAL drive / row sum FIXED and scrambles only which context dimension
    it aligns to) is the decisive control here and collapses hard; the GLOBAL shuffle (Shiu's exact control, holds
    only the weight histogram) collapses on this seed with a large ratio too."""
    os.environ.setdefault("SIM_BACKEND", "numpy")
    from research.runners._gap5_episodic_temporal_context_when_derisk import build_and_form, GO_DEFAULTS
    from research.runners._weight_shuffle_dependency_control_demo import make_recency_measure

    p = dict(GO_DEFAULTS)
    S = build_and_form(42, n_items=6, n_ctx=200, rho=0.72, beta=0.60, k_active=10, ctx_lr=1.0,
                       cue_frac=0.15, drive_pA=50.0, p=p, preassigned=True, n_ca3_pre=300, verbose=False)
    assert S.error is None, S.error
    measure = make_recency_measure(S, drive_pA=50.0, ctx_pA=700.0)

    # (1) the DEFAULT (real-weights) arm must genuinely WORK — a strong recency gradient — else the control is void.
    real = measure(S.W_ctx)
    lesion = measure(S.W_les)   # the runner's own anti-cheat: W_ctx := 0 => no gradient
    assert real > 0.15, f"real recency gradient absent (range={real:.4f}) — control is meaningless if the default fails"
    assert real > lesion + 0.1, f"real ({real:.4f}) must exceed the W_ctx=0 lesion ({lesion:.4f})"
    # the shuffle must not change the weight VALUES (only their positions)
    Wsh = shuffle_preserving_marginal(S.W_ctx, np.random.default_rng(1), mode="global")
    assert np.array_equal(np.sort(Wsh.reshape(-1)), np.sort(S.W_ctx.reshape(-1)))

    # (2a) PER_ROW shuffle — the decisive control: recency must collapse BELOW every shuffled draw.
    dc_r = dependency_control(measure, S.W_ctx, np.random.default_rng(4249), n_shuffles=20, mode="per_row")
    assert dc_r["collapsed"] is True, dc_r
    assert dc_r["real_score"] > dc_r["shuffled_max"], (
        f"real ({dc_r['real_score']:.4f}) must beat EVERY per-row shuffle (max={dc_r['shuffled_max']:.4f})")

    # (2b) GLOBAL shuffle (Shiu's exact control) — collapse by a robust margin (real >= 3x the shuffled mean,
    #      and real clears the bulk of the null). p95 itself is a thin margin on 6 serial positions, so assert the
    #      ratio + tail fraction, which are robust to a one-cell quantisation shift.
    dc_g = dependency_control(measure, S.W_ctx, np.random.default_rng(4249), n_shuffles=20, mode="global")
    assert dc_g["real_score"] >= 3.0 * max(dc_g["shuffled_mean"], 1e-9), (
        f"real ({dc_g['real_score']:.4f}) is not >= 3x the global-shuffle mean ({dc_g['shuffled_mean']:.4f}) — "
        "the recency function would be riding on the weight distribution, not learned structure")
    assert dc_g["frac_ge_real"] <= 0.2, (
        f"too many global shuffles ({dc_g['n_ge_real']}/{dc_g['n_shuffles']}) matched the real score")
