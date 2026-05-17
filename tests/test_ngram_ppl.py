import math
import numpy as np
from sim.ngram_ppl import ngram_heldout_nll


class _T:
    def __init__(self, table):
        self.table = table
        self.V = len(next(iter(table.values())))
    def soft_dist(self, ctx):
        return np.asarray(self.table[tuple(ctx)], dtype=np.float64)


def test_nll_matches_hand_computed():
    t = _T({(0, 1): [0.25, 0.25, 0.5]})
    nll = ngram_heldout_nll(t, [0, 1, 2])
    assert len(nll) == 1
    assert abs(nll[0] - math.log(2.0)) < 1e-9


def test_short_input_is_empty():
    t = _T({(0, 1): [0.5, 0.5]})
    assert ngram_heldout_nll(t, []) == []
    assert ngram_heldout_nll(t, [0]) == []
    assert ngram_heldout_nll(t, [0, 1]) == []


def test_zero_prob_is_clamped_not_inf():
    t = _T({(0, 1): [1.0, 0.0, 0.0]})
    nll = ngram_heldout_nll(t, [0, 1, 2])
    assert math.isfinite(nll[0]) and nll[0] == -math.log(1e-12)


def test_perplexity_roundtrip():
    from research.runners.subword_lm_gate_core import perplexity
    t = _T({(0, 1): [0.0, 0.0, 1.0], (1, 2): [0.0, 0.0, 1.0]})
    assert abs(perplexity(ngram_heldout_nll(t, [0, 1, 2, 2]))
               - 1.0) < 1e-9
