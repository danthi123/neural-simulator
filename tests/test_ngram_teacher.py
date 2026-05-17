import numpy as np
from sim.ngram_teacher import NgramTeacher

def test_soft_dist_is_a_valid_distribution():
    ids = [1,2,3,1,2,3,1,2,4,2,3,1] * 20
    t = NgramTeacher(); t.train(ids, vocab_size=8)
    q = t.soft_dist((1, 2))
    assert q.shape == (8,)
    assert abs(float(q.sum()) - 1.0) < 1e-9
    assert (q >= 0).all()

def test_beats_uniform_on_structured_corpus():
    import math
    ids = ([1,2,3] * 400)
    V = 6
    t = NgramTeacher(); t.train(ids, vocab_size=V)
    held = [1,2,3] * 50
    nll = []
    for i in range(2, len(held)):
        p = float(t.soft_dist((held[i-2], held[i-1]))[held[i]])
        nll.append(-math.log(max(p, 1e-12)))
    ppl = math.exp(sum(nll)/len(nll))
    assert ppl < V

def test_deterministic():
    ids = [3,1,4,1,5,9,2,6,5,3,5] * 30
    a = NgramTeacher(); a.train(ids, 12)
    b = NgramTeacher(); b.train(ids, 12)
    assert np.array_equal(a.soft_dist((1,5)), b.soft_dist((1,5)))

def test_backoff_and_short_context_safe():
    ids = [1,2,1,2,1,2] * 10
    t = NgramTeacher(); t.train(ids, vocab_size=5)
    q = t.soft_dist((4, 4))
    assert abs(float(q.sum()) - 1.0) < 1e-9
    assert t.soft_dist(()).shape == (5,)
    assert t.soft_dist((1,)).shape == (5,)
    assert abs(float(t.soft_dist(()).sum()) - 1.0) < 1e-9
