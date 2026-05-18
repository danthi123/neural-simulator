"""LOAD-BEARING: (A) per-layer FIXED-RANDOM feedback B never mutated
and never derived from forward W (no weight transport); (B) the
batched hidden update EQUALS the committed per-sample
sim.dendritic_plasticity sum (faithful reuse); (C) NO autograd in the
shipped path; (D) oracle mode is hand-derived numpy backprop that
genuinely descends loss on a tiny problem (positive control works);
(E) modes are clean + deterministic; (F) the V1 true-gradient
positive-control genuinely trains the REAL net depth on a real-MNIST
slice with the engineered (sigmoid + standardized + momentum)
optimizer -- the missing V1-at-real-config validation."""
import os
import numpy as np
import inspect
import pytest
import sim.dendritic_mlp as dm
import sim.dendritic_plasticity as dp
import sim.backend as _be
from sim.backend import get_backend

_MNIST_NPZ = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "mnist.npz")


def test_no_autograd_in_module():
    src = inspect.getsource(dm)
    assert "torch" not in src and "autograd" not in src


def test_fixed_feedback_never_mutated_no_weight_transport():
    net = dm.DendriticMLP([12, 16, 16, 4], seed=7)
    B0 = [b.copy() for b in net.B]
    W0 = [w.copy() for w in net.W]
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 12)); y = rng.integers(0, 4, 20)
    for _ in range(5):
        net.train_step(X, y, mode="local_correct", lr=0.1)
    for b, b0 in zip(net.B, B0):
        assert np.array_equal(b, b0)
    for b in net.B:
        for w in net.W:
            assert b.shape != w.shape or not np.array_equal(b, w)
            assert (b.shape != w.T.shape) or not np.array_equal(b, w.T)


@pytest.fixture
def numpy_backend_pinned(monkeypatch):
    """Numerics-correctness pins (NOT perf pins) run on the NumPy
    backend so the comparison is exact + deterministic against the
    BYTE-FROZEN numpy sim.dendritic_plasticity reference. The default
    process backend may be CuPy after the GPU port (that is the point
    -- training is GPU-accelerated); these two faithfulness pins stay
    numpy<->numpy so the 1e-9 equivalence and the oracle-descends
    numerics stay provably exact. No assertion/bar is weakened; only
    the array library under the formula is pinned for the comparison.
    """
    # Snapshot the shared backend cache so pinning numpy here does not
    # leak into other tests (which validate the GPU/CuPy path).
    saved = (_be._cached_backend, _be._cached_name, _be._cached_sparse)
    np_xp, _ = get_backend("numpy")
    monkeypatch.setattr(dm, "xp", np_xp)
    try:
        yield
    finally:
        (_be._cached_backend, _be._cached_name,
         _be._cached_sparse) = saved


def test_batched_update_equals_committed_per_sample_sum(
        numpy_backend_pinned):
    net = dm.DendriticMLP([5, 6, 3], seed=1)
    rng = np.random.default_rng(2)
    X = rng.normal(size=(8, 5)); y = rng.integers(0, 3, 8)
    dW0 = net._debug_hidden_dW0(X, y)
    acts, e = net._debug_fwd_err(X, y)
    pre, soma = acts[0], acts[1]
    ap = e @ net.B[0]
    ref = np.zeros_like(net.W[0])
    for i in range(8):
        ref += dp.urbanczik_senn_update(
            pre[i], soma[i], soma[i], np.ones(soma.shape[1]),
            apical_signal=ap[i])
    assert np.allclose(dW0, ref, atol=1e-9)


def test_oracle_mode_is_positive_control_descends_loss(
        numpy_backend_pinned):
    net = dm.DendriticMLP([8, 16, 16, 3], seed=3)
    rng = np.random.default_rng(4)
    X = rng.normal(size=(64, 8))
    y = (X[:, 0] + X[:, 1] > 0).astype(int) + (X[:, 2] > 0).astype(int)
    L0 = net.loss(X, y)
    for _ in range(300):
        net.train_step(X, y, mode="oracle", lr=0.2)
    assert net.loss(X, y) < 0.5 * L0


def test_modes_deterministic_given_seed():
    a = dm.DendriticMLP([6, 8, 3], seed=42)
    b = dm.DendriticMLP([6, 8, 3], seed=42)
    rng = np.random.default_rng(5)
    X = rng.normal(size=(10, 6)); y = rng.integers(0, 3, 10)
    a.train_step(X, y, mode="local_correct", lr=0.1)
    b.train_step(X, y, mode="local_correct", lr=0.1)
    assert all(np.array_equal(x, z) for x, z in zip(a.W, b.W))


def _load_mnist_slice(ntr, nte):
    """Load a small TRAIN/TEST slice from the cached MNIST .npz.

    Uses the same safe loader discipline as the runner
    (allow_pickle disabled -- no untrusted deserialization). Returns
    None when the cache is absent so the caller can skip cleanly.
    """
    if not os.path.exists(_MNIST_NPZ):
        return None
    safe = {"allow_pickle": False}  # never deserialize Python objects
    with np.load(_MNIST_NPZ, **safe) as z:
        xtr = z["x_train"].reshape(-1, 784).astype(np.float32) / 255.0
        ytr = z["y_train"].astype(np.int64).reshape(-1)
        xte = z["x_test"].reshape(-1, 784).astype(np.float32) / 255.0
        yte = z["y_test"].astype(np.int64).reshape(-1)
    return xtr[:ntr], ytr[:ntr], xte[:nte], yte[:nte]


def test_oracle_trains_real_config_mnist_subset():
    """V1-at-REAL-config positive control (the validation the toy
    test_oracle_mode_is_positive_control_descends_loss never did).

    On a real-MNIST slice at the FULL pre-registered net depth
    (784-512-256-128-10 sigmoid) -- the depth that exposed the
    under-engineered-optimizer VOID -- the engineered optimizer
    (sigmoid KEPT + per-pixel standardized inputs + heavy-ball
    momentum + mean-over-batch gradient) MUST drive the hand-derived
    true-gradient ``oracle`` mode to >= 0.90 heldout in a modest
    epoch budget. This FAILS against the pre-fix batch-summed plain
    SGD (which sits at MNIST chance ~0.10 even WITH the exact
    gradient at this depth) and PASSES after the instrument is
    engineered. The optimizer machinery is mode-agnostic, so this
    does NOT advantage the local rule over its controls.

    Skipped (not failed) when the cached dataset is absent so CI on
    a machine without the corpus stays green; the cheap CLI V1
    self-gate covers the full-corpus check.
    """
    data = _load_mnist_slice(6000, 2000)
    if data is None:
        pytest.skip("data/mnist.npz absent (cheap CLI V1 self-gate "
                    "covers the full-corpus oracle>=0.95 check)")
    xtr, ytr, xte, yte = data
    # Engineered preprocessing: per-pixel standardization on TRAIN
    # statistics (the runner does the same; sigmoid MLPs need it).
    mu = xtr.mean(0, keepdims=True)
    sd = xtr.std(0, keepdims=True)
    xtr = (xtr - mu) / (sd + 1e-6)
    xte = (xte - mu) / (sd + 1e-6)
    # FULL pre-registered depth = the configuration that VOIDed.
    net = dm.DendriticMLP([784, 512, 256, 128, 10], seed=42)
    rng = np.random.default_rng(1)
    n = len(ytr)
    for _ in range(12):
        order = rng.permutation(n)
        for bi in range(0, n, 128):
            idx = order[bi:bi + 128]
            net.train_step(xtr[idx], ytr[idx], mode="oracle", lr=0.5)
    acc = net.accuracy(xte, yte)
    assert acc >= 0.90, (
        "V1 true-gradient positive control must train the real-depth "
        "sigmoid MLP to >=0.90 heldout with the engineered optimizer; "
        "got %.4f (instrument still under-engineered)" % acc)
