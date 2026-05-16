"""Phase 2.1 BPTT-SNN unit tests (numpy reference, CPU-only).

ONLY ON path-f-hybrid BRANCH.
"""
import numpy as np
import pytest


def test_lif_forward_step_resets_on_spike():
    """After a spike, voltage should reset (multiplied by 0)."""
    from sim.bptt_snn import LIFLayer, forward_step

    layer = LIFLayer(
        W_in=np.array([[10.0]], dtype=np.float32),  # strong input
        n_post=1,
        threshold=1.0,
        leak=0.95,
    )
    state = layer.init_state(batch_size=1)
    x = np.array([[1.0]], dtype=np.float32)  # input current
    state, s = forward_step(state, x, layer)
    # Should fire (10.0 input >> 1.0 threshold)
    assert s[0, 0] == 1.0
    # Next step with no input: voltage should be reset (s_prev was 1)
    x_zero = np.array([[0.0]], dtype=np.float32)
    state, s2 = forward_step(state, x_zero, layer)
    # leak * v(1) * (1 - 1) = 0; no input; v=0; no spike
    assert state["v"][0, 0] == 0.0
    assert s2[0, 0] == 0.0


def test_lif_no_spike_when_below_threshold():
    """Weak input shouldn't trigger a spike."""
    from sim.bptt_snn import LIFLayer, forward_step

    layer = LIFLayer(
        W_in=np.array([[0.1]], dtype=np.float32),  # weak input
        n_post=1,
        threshold=1.0,
        leak=0.95,
    )
    state = layer.init_state(batch_size=1)
    x = np.array([[1.0]], dtype=np.float32)
    state, s = forward_step(state, x, layer)
    assert s[0, 0] == 0.0
    # float32 representation of 0.1 isn't exact; use approx
    assert abs(state["v"][0, 0] - 0.1) < 1e-6


def test_forward_unroll_shapes_correct():
    """forward_unroll should return (T, B, n_post) per layer."""
    from sim.bptt_snn import LIFLayer, forward_unroll

    rng = np.random.default_rng(42)
    layers = [
        LIFLayer(W_in=rng.normal(0, 0.5, (3, 64)).astype(np.float32),
                 n_post=64),
        LIFLayer(W_in=rng.normal(0, 0.5, (64, 3)).astype(np.float32),
                 n_post=3),
    ]
    inputs = rng.uniform(-1, 1, (10, 2, 3)).astype(np.float32)  # T=10, B=2, V=3
    out = forward_unroll(inputs, layers)
    assert out["spikes"][0].shape == (10, 2, 64)
    assert out["spikes"][1].shape == (10, 2, 3)
    assert out["v"][0].shape == (10, 2, 64)


def test_make_abc_dataset_correct_format():
    """ABC dataset should have one-hot inputs and class indices for targets."""
    from sim.bptt_snn import make_abc_dataset

    rng = np.random.default_rng(42)
    inputs, targets = make_abc_dataset(n_samples=5, seq_len=10, rng=rng)
    assert inputs.shape == (5, 9, 3)  # n_samples, seq_len-1, V_in=3
    assert targets.shape == (5, 9)
    # Inputs are one-hot
    assert (inputs.sum(axis=-1) == 1.0).all()
    # Targets are 0, 1, or 2
    assert ((targets >= 0) & (targets <= 2)).all()


def test_make_abc_pattern_is_cyclic():
    """ABC sequence is (offset+0, offset+1, offset+2, offset+3, ...) % 3.
    Adjacent positions differ by 1 (mod 3)."""
    from sim.bptt_snn import make_abc_dataset

    rng = np.random.default_rng(42)
    inputs, targets = make_abc_dataset(n_samples=10, seq_len=20, rng=rng)
    # For each sample, target[t] = (input_class[t] + 1) % 3
    for i in range(10):
        for t in range(19):
            input_class = int(np.argmax(inputs[i, t]))
            expected_target = (input_class + 1) % 3
            assert targets[i, t] == expected_target, (
                f"Sample {i}, t={t}: input class {input_class}, "
                f"target {targets[i, t]}, expected {expected_target}"
            )


def test_atan_surrogate_np_shape_preserved():
    """atan_surrogate_np preserves array shape."""
    from sim.bptt_snn import atan_surrogate_np

    v = np.array([[0.0, 1.0], [-0.5, 2.0]], dtype=np.float32)
    grad = atan_surrogate_np(v)
    assert grad.shape == (2, 2)


def test_cross_entropy_loss_decreases_with_correct_logit():
    """Higher logit at target -> lower loss."""
    from sim.bptt_snn import cross_entropy_loss_np

    # Wrong logit
    logits_wrong = np.array([[5.0, 0.0, 0.0]], dtype=np.float32)
    loss_wrong = cross_entropy_loss_np(logits_wrong, target_idx=2)
    # Correct logit (target 2 highest)
    logits_correct = np.array([[0.0, 0.0, 5.0]], dtype=np.float32)
    loss_correct = cross_entropy_loss_np(logits_correct, target_idx=2)
    assert loss_correct < loss_wrong


def test_softmax_grad_np_sums_to_zero():
    """For softmax + cross-entropy, gradient sums to zero per sample."""
    from sim.bptt_snn import softmax_grad_np

    logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    grad = softmax_grad_np(logits, target_idx=1)
    # Sum across class dim should be 0 (probabilities sum to 1, target -=1)
    assert abs(grad.sum()) < 1e-5


def test_backward_unroll_correct_shapes():
    """backward_unroll returns weight gradients with same shape as W_in."""
    from sim.bptt_snn import (
        LIFLayer, forward_unroll, backward_unroll,
    )
    rng = np.random.default_rng(42)
    layers = [
        LIFLayer(W_in=rng.normal(0, 0.5, (3, 32)).astype(np.float32),
                 n_post=32),
        LIFLayer(W_in=rng.normal(0, 0.5, (32, 3)).astype(np.float32),
                 n_post=3),
    ]
    inputs = rng.uniform(-1, 1, (10, 1, 3)).astype(np.float32)
    state = forward_unroll(inputs, layers)
    output_grad = np.zeros((10, 1, 3), dtype=np.float32)
    output_grad[:, :, 1] = 1.0
    weight_grads, input_grad = backward_unroll(
        inputs, layers, state, output_grad
    )
    assert weight_grads[0].shape == (3, 32)
    assert weight_grads[1].shape == (32, 3)
    assert input_grad.shape == (10, 1, 3)


def test_backward_zero_grad_when_zero_output_grad():
    """Zero output_grad -> zero weight gradients."""
    from sim.bptt_snn import (
        LIFLayer, forward_unroll, backward_unroll,
    )
    rng = np.random.default_rng(42)
    layers = [
        LIFLayer(W_in=rng.normal(0, 0.5, (3, 16)).astype(np.float32),
                 n_post=16),
        LIFLayer(W_in=rng.normal(0, 0.5, (16, 3)).astype(np.float32),
                 n_post=3),
    ]
    inputs = rng.uniform(-1, 1, (10, 1, 3)).astype(np.float32)
    state = forward_unroll(inputs, layers)
    output_grad = np.zeros((10, 1, 3), dtype=np.float32)
    weight_grads, _ = backward_unroll(inputs, layers, state, output_grad)
    assert np.abs(weight_grads[0]).max() < 1e-6
    assert np.abs(weight_grads[1]).max() < 1e-6


def test_backward_nonzero_grad_with_nonzero_output_grad():
    """Non-zero output_grad through firing network -> non-zero weight grads."""
    from sim.bptt_snn import (
        LIFLayer, forward_unroll, backward_unroll,
    )
    rng = np.random.default_rng(42)
    layers = [
        LIFLayer(W_in=rng.normal(0, 1.0, (3, 32)).astype(np.float32),
                 n_post=32),
        LIFLayer(W_in=rng.normal(0, 1.0, (32, 3)).astype(np.float32),
                 n_post=3),
    ]
    inputs = np.ones((10, 1, 3), dtype=np.float32)
    state = forward_unroll(inputs, layers)
    output_grad = np.ones((10, 1, 3), dtype=np.float32)
    weight_grads, _ = backward_unroll(inputs, layers, state, output_grad)
    # At least the output layer should have non-trivial gradients
    assert np.abs(weight_grads[1]).max() > 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
