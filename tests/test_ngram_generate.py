import numpy as np
from sim.ngram_generate import ngram_sample_next, ngram_generate


class _FakeTeacher:
    """Minimal stand-in with the NgramTeacher.soft_dist contract."""
    def __init__(self, V, peak=None):
        self.V = V
        self.peak = peak
    def soft_dist(self, ctx):
        q = np.full(self.V, 1.0 / self.V, dtype=np.float64)
        if self.peak is not None:
            q[:] = 0.01 / self.V
            q[self.peak] = 1.0 - 0.01 + 0.01 / self.V
            q = q / q.sum()
        return q


def test_sample_next_temp0_is_argmax_stable_first_max():
    t = _FakeTeacher(5, peak=3)
    assert ngram_sample_next(t, (1, 2), np.random.default_rng(0),
                             temperature=0.0) == 3
    tu = _FakeTeacher(5)
    assert ngram_sample_next(tu, (), np.random.default_rng(7),
                             temperature=0.0) == 0


def test_sample_next_temp_in_range_and_seed_reproducible():
    t = _FakeTeacher(8)
    a = ngram_sample_next(t, (1,), np.random.default_rng(42), 1.0)
    b = ngram_sample_next(t, (1,), np.random.default_rng(42), 1.0)
    assert a == b and 0 <= a < 8


def test_sample_next_degenerate_safe():
    t = _FakeTeacher(4)
    assert 0 <= ngram_sample_next(t, (1, 2),
                                  np.random.default_rng(1), 1.0) < 4
    assert 0 <= ngram_sample_next(t, (),
                                  np.random.default_rng(1), 0.0) < 4


def test_generate_length_and_range_and_reproducible():
    t = _FakeTeacher(6, peak=2)
    g1 = ngram_generate(t, [5, 1], 10, np.random.default_rng(3), 1.0)
    g2 = ngram_generate(t, [5, 1], 10, np.random.default_rng(3), 1.0)
    assert g1 == g2 and len(g1) == 10
    assert all(0 <= x < 6 for x in g1)
    assert g1 != [5, 1]
    g0 = ngram_generate(t, [], 5, np.random.default_rng(0), 0.0)
    assert g0 == [2, 2, 2, 2, 2]
