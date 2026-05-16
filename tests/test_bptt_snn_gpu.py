"""Phase 2.2 GPU-aware BPTT tests -- numerical equivalence with numpy.

ONLY ON path-f-hybrid BRANCH.

Validates that bptt_snn_gpu produces same results as bptt_snn
(numpy reference) within fp32 tolerance.
"""
import numpy as np
import pytest


def test_atan_surrogate_xp_matches_np():
    """atan_surrogate (xp version) gives same numbers as numpy version."""
    from sim.bptt_snn import atan_surrogate_np
    from sim.bptt_snn_gpu import atan_surrogate

    rng = np.random.default_rng(42)
    v = rng.uniform(-2, 2, (5, 5)).astype(np.float32)
    g_np = atan_surrogate_np(v, alpha=2.0)
    g_xp = atan_surrogate(v, alpha=2.0, xp=np)
    np.testing.assert_allclose(g_np, g_xp, rtol=1e-5)


def test_forward_unroll_xp_matches_np():
    """forward_unroll_xp (numpy backend) matches sim.bptt_snn forward_unroll."""
    from sim.bptt_snn import LIFLayer, forward_unroll
    from sim.bptt_snn_gpu import LIFLayerXP, forward_unroll_xp

    rng = np.random.default_rng(42)
    W0 = rng.normal(0, 0.5, (3, 16)).astype(np.float32)
    W1 = rng.normal(0, 0.5, (16, 3)).astype(np.float32)
    inputs = rng.uniform(-1, 1, (10, 1, 3)).astype(np.float32)

    layers_np = [
        LIFLayer(W_in=W0.copy(), n_post=16, threshold=1.0, leak=0.95),
        LIFLayer(W_in=W1.copy(), n_post=3, threshold=1.0, leak=0.95),
    ]
    layers_xp = [
        LIFLayerXP(W_in=W0.copy(), n_post=16, threshold=1.0, leak=0.95),
        LIFLayerXP(W_in=W1.copy(), n_post=3, threshold=1.0, leak=0.95),
    ]

    out_np = forward_unroll(inputs, layers_np)
    out_xp = forward_unroll_xp(inputs, layers_xp, xp=np)

    for li in range(2):
        np.testing.assert_allclose(out_np["spikes"][li], out_xp["spikes"][li], rtol=1e-5)
        np.testing.assert_allclose(out_np["v"][li], out_xp["v"][li], rtol=1e-5)


def test_backward_unroll_xp_matches_np():
    """backward_unroll_xp (numpy backend) matches sim.bptt_snn backward_unroll."""
    from sim.bptt_snn import LIFLayer, forward_unroll, backward_unroll
    from sim.bptt_snn_gpu import LIFLayerXP, forward_unroll_xp, backward_unroll_xp

    rng = np.random.default_rng(42)
    W0 = rng.normal(0, 0.5, (3, 16)).astype(np.float32)
    W1 = rng.normal(0, 0.5, (16, 3)).astype(np.float32)
    inputs = rng.uniform(-1, 1, (10, 1, 3)).astype(np.float32)
    output_grad = rng.normal(0, 0.5, (10, 1, 3)).astype(np.float32)

    layers_np = [
        LIFLayer(W_in=W0.copy(), n_post=16, threshold=1.0, leak=0.95),
        LIFLayer(W_in=W1.copy(), n_post=3, threshold=1.0, leak=0.95),
    ]
    layers_xp = [
        LIFLayerXP(W_in=W0.copy(), n_post=16, threshold=1.0, leak=0.95),
        LIFLayerXP(W_in=W1.copy(), n_post=3, threshold=1.0, leak=0.95),
    ]

    state_np = forward_unroll(inputs, layers_np)
    state_xp = forward_unroll_xp(inputs, layers_xp, xp=np)
    grads_np, _ = backward_unroll(inputs, layers_np, state_np, output_grad)
    grads_xp, _ = backward_unroll_xp(inputs, layers_xp, state_xp, output_grad, xp=np)

    np.testing.assert_allclose(grads_np[0], grads_xp[0], rtol=1e-5)
    np.testing.assert_allclose(grads_np[1], grads_xp[1], rtol=1e-5)


def test_get_backend_returns_module():
    """_get_backend returns a (module, is_gpu_flag) tuple."""
    from sim.bptt_snn_gpu import _get_backend
    xp, is_gpu = _get_backend(prefer_gpu=False)
    # With prefer_gpu=False, should always be numpy
    assert xp.__name__ in ("numpy",)
    assert is_gpu is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
