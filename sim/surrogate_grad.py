"""Phase 2.1 surrogate-gradient module.

ONLY USED ON path-f-hybrid BRANCH. Provides surrogate gradient
functions for backward-pass through spike events. The forward pass
uses Heaviside (real spikes); the backward pass replaces the
gradient with a smooth surrogate.

Surrogates:
- atan_surrogate (Zenke 2018, used by SuperSpike): more stable
  for deep networks
- fast_sigmoid_surrogate (Zenke 2021): slightly cheaper

Both gated by cfg.enable_surrogate_grad_pretraining (default False).
On main, this module is unused; on path-f-hybrid it powers the
cortex_pretraining runner.
"""
from __future__ import annotations

import cupy as cp


def atan_surrogate(v_minus_threshold, alpha: float = 2.0):
    """ATan surrogate gradient (Zenke 2018, SuperSpike).

    Forward: spike = (v >= threshold).astype(float32)
    Backward: dspike/dv = (1/pi) * (1 / (1 + (alpha * (v-thr))^2))

    This module computes the BACKWARD gradient only. Forward pass
    uses Heaviside (already in fused_*_dynamics_update kernels).

    Args:
        v_minus_threshold: cupy array of v - threshold values per
            neuron at a given timestep.
        alpha: slope hyperparameter. 2.0 is a good default for
            deep networks.

    Returns:
        cupy array of dspike/dv per neuron.
    """
    return (1.0 / cp.pi) * (
        1.0 / (1.0 + (alpha * v_minus_threshold) ** 2)
    )


def fast_sigmoid_surrogate(v_minus_threshold, alpha: float = 5.0):
    """Fast-sigmoid surrogate (Zenke 2021).

    Slightly cheaper than ATan; less stable for very deep networks.
    """
    return 1.0 / (1.0 + cp.abs(alpha * v_minus_threshold)) ** 2


def cross_entropy_loss(logits, target_idx: int) -> float:
    """Standard cross-entropy on raw logits.

    Args:
        logits: cupy array shape (vocab_size,)
        target_idx: int target class

    Returns:
        scalar loss value (host float)
    """
    log_probs = logits - cp.log(cp.sum(cp.exp(logits)))
    return float(-log_probs[target_idx].get())


def softmax_grad(logits, target_idx: int):
    """dL/d_logits for softmax + cross-entropy.

    Args:
        logits: cupy array shape (vocab_size,)
        target_idx: int target class

    Returns:
        cupy array shape (vocab_size,) of gradients
    """
    probs = cp.exp(logits) / cp.sum(cp.exp(logits))
    grad = probs.copy()
    grad[target_idx] -= 1.0
    return grad
