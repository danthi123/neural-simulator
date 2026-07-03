"""CI guard for EMERGE-47 — L2-input LOCAL NORMALIZATION (PPMI/divisive-norm) to surpass the EMERGE-46 stacked-pooler
boundary. These tests pin the MECHANISM FACTS (fast, CPU/numpy), NOT a GO (the honest verdict is a partial-lift
BOUNDARY): (1) the IDF/PPMI-marginal weights are DATA-DRIVEN (down-weight ubiquitous columns, up-weight rare ones; a
permuted-stats shuffle changes them); (2) in the on-substrate FAILING regime (strong winner-inactive depression), L2-input
normalization LIFTS held-out within-super overlap vs OFF WITHOUT raising cross-super (a real, directional, data-driven
lift); (3) the OFF arm (in_weights=None) is the identity of the EMERGE-44 pooler (a clean A/B). Skip if deps missing.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import pytest

pytest.importorskip("sim.bridge")
_mod = pytest.importorskip("research.runners._emerge47_l2_input_normalization_derisk")

compute_idf_weights = _mod.compute_idf_weights
_competitive_pool_normalized = _mod._competitive_pool_normalized
NormalizedStackedPoolerProbe = _mod.NormalizedStackedPoolerProbe
POOL_LD_STRONG = _mod.POOL_LD_STRONG


def test_idf_weights_are_data_driven():
    """The IDF/PPMI-marginal normalization weights DOWN-weight ubiquitous columns and UP-weight rare ones, and are
    computed FROM THE DATA (a permuted-stats shuffle changes them). This is the data-driven property the anti-cheat
    depends on -- the normalization is learned from the co-occurrence corpus, not hard-wired to the task."""
    n_in = 20
    # column 0 active in every sample (ubiquitous), column 19 active in only one (rare/informative)
    samples = [{0, 1, 2}, {0, 3, 4}, {0, 5, 6}, {0, 7, 19}]
    w = compute_idf_weights(samples, n_in)
    assert w.shape == (n_in,)
    assert w[0] < w[19], "ubiquitous column must be down-weighted vs a rare one"
    assert w[0] < w[1], "the always-on column is the least informative"
    # a never-active column gets the max weight (log((1+N)/1))
    assert w[10] >= w[19]
    # permuted-stats: shuffling column identities changes the weight vector (data-driven, not a constant)
    w_perm = compute_idf_weights(samples, n_in, shuffle_stats_seed=123)
    assert not np.allclose(w, w_perm), "permuted-stats must change the weights (they are data-dependent)"
    assert np.isclose(sorted(w), sorted(w_perm)).all(), "permuted-stats is a permutation of the same values"


def test_off_arm_is_identity_of_emerge44_pooler():
    """in_weights=None + ld=POOL_LD reproduces the EMERGE-44 `_competitive_pool` byte-for-byte (same RNG stream, same
    update), so the ON/OFF normalization toggle is a clean A/B on top of the validated pooler."""
    from research.runners._emerge44_stacked_pooler_derisk import _competitive_pool, POOL_LD
    rng = np.random.default_rng(7)
    samples = [set(rng.choice(30, 5, replace=False)) for _ in range(24)]
    ref = _competitive_pool(11, samples, 30, 40, 4, 20)
    test = _competitive_pool_normalized(11, samples, 30, 40, 4, 20, in_weights=None, ld=POOL_LD)
    for q in samples[:6]:
        assert ref(q) == test(q), "OFF arm must equal the EMERGE-44 pooler exactly (identity A/B)"


def test_normalization_lifts_heldout_in_failing_regime_without_raising_cross():
    """The core EMERGE-47 mechanism claim (numpy diagnostic, the on-substrate FAILING regime = strong winner-inactive
    depression): L2-input local normalization LIFTS the held-out within-super L2 overlap vs OFF, WITHOUT raising the
    cross-super overlap, and the lift is DATA-DRIVEN (permuted-stats collapses it). This is a directional/mechanism
    assertion, NOT a GO (the honest verdict is a partial-lift boundary -- super-acc does not reach 0.80 by normalization
    alone; soft/union pooling is the next rung)."""
    seeds = [42, 43, 44]
    off_w, on_w, on_c, off_c, ps_w = [], [], [], [], []
    for s in seeds:
        po = NormalizedStackedPoolerProbe(seed=s, epochs=40, normalize=False, ld=POOL_LD_STRONG)
        w, c = po.held_out_within_cross_overlap(); off_w.append(w); off_c.append(c)
        pn = NormalizedStackedPoolerProbe(seed=s, epochs=40, normalize=True, ld=POOL_LD_STRONG)
        w, c = pn.held_out_within_cross_overlap(); on_w.append(w); on_c.append(c)
        pp = NormalizedStackedPoolerProbe(seed=s, epochs=40, normalize=True, permute_stats=True, ld=POOL_LD_STRONG)
        w, _ = pp.held_out_within_cross_overlap(); ps_w.append(w)
    off_w, on_w, on_c, off_c, ps_w = map(lambda a: float(np.mean(a)), (off_w, on_w, on_c, off_c, ps_w))
    # the failing regime: OFF held-out within-super overlap is near-zero (reproduces the on-substrate boundary ~0.01)
    assert off_w < 0.02, f"OFF must be in the failing regime (got {off_w:.3f})"
    # normalization LIFTS held-out within-super overlap (a real, directional lift)
    assert on_w > off_w, f"normalization must lift held-out within-super overlap ({off_w:.3f} -> {on_w:.3f})"
    # WITHOUT raising cross-super overlap (does not break the anti-cheat)
    assert on_c <= off_c + 0.02, f"normalization must NOT raise cross-super overlap ({off_c:.3f} -> {on_c:.3f})"
    # DATA-DRIVEN: shuffling the column identities (permuted-stats) removes the lift
    assert on_w > ps_w, f"the lift must be data-driven (ON {on_w:.3f} > permuted-stats {ps_w:.3f})"
