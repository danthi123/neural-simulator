"""Phase 2.1 surrogate-gradient unit tests.

CPU-only tests where possible. Runs on path-f-hybrid branch.
"""
import pytest


def test_atan_at_threshold_matches_analytic():
    """At v == threshold, ATan surrogate returns 1/pi."""
    import numpy as np
    try:
        import cupy as cp
    except ImportError:
        pytest.skip("cupy required")
    from sim.surrogate_grad import atan_surrogate

    v = cp.zeros((1,), dtype=cp.float32)  # v - threshold = 0
    result = atan_surrogate(v, alpha=2.0)
    assert abs(float(result[0]) - 1.0 / np.pi) < 1e-5


def test_atan_decays_far_from_threshold():
    """Far from threshold, surrogate gradient should decay toward 0."""
    try:
        import cupy as cp
    except ImportError:
        pytest.skip("cupy required")
    from sim.surrogate_grad import atan_surrogate

    v_far = cp.array([10.0], dtype=cp.float32)
    result = atan_surrogate(v_far, alpha=2.0)
    assert float(result[0]) < 0.01


def test_softmax_grad_target_nonneg_others_pos():
    """For a softmax with cross-entropy, target class gradient is
    less than other classes (target position has -1 added)."""
    try:
        import cupy as cp
    except ImportError:
        pytest.skip("cupy required")
    from sim.surrogate_grad import softmax_grad

    logits = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
    grad = softmax_grad(logits, target_idx=2)
    grad_h = grad.get()
    # All other gradients should be positive (push away)
    assert grad_h[0] > 0
    assert grad_h[1] > 0
    # Target gradient should be negative (pull toward)
    assert grad_h[2] < 0


def test_softmax_grad_sums_to_zero():
    """Softmax + cross-entropy gradient sums to zero (probability)
    minus 1 at target (sums to 0)."""
    try:
        import cupy as cp
    except ImportError:
        pytest.skip("cupy required")
    from sim.surrogate_grad import softmax_grad

    logits = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
    grad = softmax_grad(logits, target_idx=1)
    total = float(grad.sum().get())
    assert abs(total) < 1e-5


def test_module_loads():
    """Module imports without errors."""
    from sim import surrogate_grad
    assert hasattr(surrogate_grad, 'atan_surrogate')
    assert hasattr(surrogate_grad, 'fast_sigmoid_surrogate')
    assert hasattr(surrogate_grad, 'cross_entropy_loss')
    assert hasattr(surrogate_grad, 'softmax_grad')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
