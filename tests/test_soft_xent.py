import numpy as np
from sim.soft_xent import soft_xent_loss, soft_xent_grad
from sim.bptt_snn import cross_entropy_loss_np, softmax_grad_np

def test_equals_hard_CE_when_q_is_one_hot():
    rng = np.random.default_rng(0)
    for _ in range(5):
        logits = rng.normal(0, 3, (1, 7)).astype(np.float64)
        tgt = int(rng.integers(0, 7))
        q = np.zeros(7); q[tgt] = 1.0
        assert abs(soft_xent_loss(logits, q)
                   - cross_entropy_loss_np(logits, tgt)) < 1e-6
        assert np.allclose(soft_xent_grad(logits, q),
                           softmax_grad_np(logits, tgt), atol=1e-6)

def test_grad_is_finite_difference_correct():
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 1, (1, 5)).astype(np.float64)
    q = rng.random(5); q = q / q.sum()
    g = soft_xent_grad(logits, q)
    eps = 1e-5
    for j in range(5):
        lp = logits.copy(); lp[0, j] += eps
        lm = logits.copy(); lm[0, j] -= eps
        fd = (soft_xent_loss(lp, q) - soft_xent_loss(lm, q)) / (2*eps)
        assert abs(fd - g[0, j]) < 1e-4

def test_loss_nonnegative_and_minimized_at_match():
    q = np.array([0.1, 0.7, 0.2])
    near = soft_xent_loss(np.log(q).reshape(1, 3), q)
    far = soft_xent_loss(np.array([[5.0, -5.0, 0.0]]), q)
    assert near <= far and near >= 0.0

def test_renormalizes_and_handles_nonfinite_without_crash():
    logits = np.array([[1.0, 2.0, 3.0]])
    q_bad = np.array([2.0, 2.0, 4.0])
    L = soft_xent_loss(logits, q_bad)
    assert np.isfinite(L)
    g = soft_xent_grad(logits, q_bad)
    assert np.isfinite(g).all() and g.shape == (1, 3)
    assert np.isfinite(soft_xent_loss(np.array([[1e9, -1e9, 0.0]]),
                                      np.array([0.3,0.3,0.4])))
