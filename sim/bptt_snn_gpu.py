"""Phase 2.2 GPU-aware BPTT (CuPy + numpy fallback).

ONLY ON path-f-hybrid BRANCH.

Mirrors sim.bptt_snn (numpy reference) but uses CuPy when available
for GPU acceleration. Falls back to numpy when CuPy unavailable
(e.g., on CPU-only systems for testing).

Numerical equivalence with numpy reference is validated in
tests/test_bptt_snn_gpu.py: same seed, same input -> same output
within fp32 tolerance (~1e-5).

Backend selection:
- xp = cupy if cupy installed and GPU available
- xp = numpy otherwise

The same forward_unroll / backward_unroll signatures work for both
backends. Caller passes input arrays in the chosen backend's
type; we don't auto-convert.

Per Phase 2.2 design (master plan).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import math


# Backend selection
def _get_backend(prefer_gpu: bool = True):
    """Return backend module (cupy or numpy) and a flag."""
    if prefer_gpu:
        try:
            import cupy as cp
            # Verify GPU is actually accessible
            _ = cp.array([1.0])
            return cp, True
        except (ImportError, Exception):
            pass
    import numpy as np
    return np, False


def atan_surrogate(v_minus_threshold, alpha: float = 2.0, xp=None):
    """ATan surrogate gradient. Backend-agnostic.

    Args:
        v_minus_threshold: array (any backend)
        alpha: slope hyperparameter
        xp: backend module (numpy or cupy). If None, auto-detects.
    """
    if xp is None:
        xp, _ = _get_backend()
    return (1.0 / math.pi) * (1.0 / (1.0 + (alpha * v_minus_threshold) ** 2))


@dataclass
class LIFLayerXP:
    """Backend-agnostic LIF layer."""
    W_in: object                      # numpy or cupy array
    n_post: int
    threshold: float = 1.0
    leak: float = 0.95

    def init_state(self, batch_size: int, xp=None):
        if xp is None:
            xp, _ = _get_backend()
        return {
            "v": xp.zeros((batch_size, self.n_post),
                           dtype=self.W_in.dtype),
            "s": xp.zeros((batch_size, self.n_post),
                           dtype=self.W_in.dtype),
        }


def forward_step_xp(state: dict, x, layer: LIFLayerXP, xp=None):
    """One LIF timestep, backend-agnostic."""
    if xp is None:
        xp, _ = _get_backend()
    v_pre = layer.leak * state["v"] * (1.0 - state["s"])
    v_new = v_pre + x @ layer.W_in
    s_new = (v_new >= layer.threshold).astype(layer.W_in.dtype)
    return {"v": v_new, "s": s_new}, s_new


def forward_unroll_xp(inputs, layers, xp=None):
    """Forward unroll T steps. backend-agnostic."""
    if xp is None:
        xp, _ = _get_backend()
    T, B, _ = inputs.shape
    L = len(layers)
    states = [layer.init_state(B, xp=xp) for layer in layers]

    # Allocate output buffers
    spikes = [xp.zeros((T, B, layer.n_post), dtype=layer.W_in.dtype)
              for layer in layers]
    v_per = [xp.zeros((T, B, layer.n_post), dtype=layer.W_in.dtype)
             for layer in layers]

    for t in range(T):
        x_in = inputs[t]
        for li, layer in enumerate(layers):
            states[li], s = forward_step_xp(states[li], x_in, layer, xp=xp)
            spikes[li][t] = s
            v_per[li][t] = states[li]["v"]
            x_in = s
    return {"spikes": spikes, "v": v_per}


def backward_unroll_xp(
    inputs,                       # (T, B, V_in)
    layers,
    forward_state,
    output_grad,                  # (T, B, V_out)
    alpha: float = 2.0,
    xp=None,
):
    """BPTT backward, backend-agnostic. Returns (weight_grads, input_grad)."""
    if xp is None:
        xp, _ = _get_backend()
    T, B, V_in = inputs.shape
    L = len(layers)
    spikes = forward_state["spikes"]
    v_per = forward_state["v"]

    weight_grads = [xp.zeros_like(layer.W_in) for layer in layers]
    dv_grads = [xp.zeros((T, B, layer.n_post), dtype=layer.W_in.dtype)
                for layer in layers]

    for li in range(L - 1, -1, -1):
        layer = layers[li]
        s_layer = spikes[li]
        v_layer = v_per[li]

        if li == L - 1:
            ds_grad = output_grad.copy()
        else:
            next_layer = layers[li + 1]
            ds_grad = xp.zeros((T, B, layer.n_post),
                               dtype=layer.W_in.dtype)
            for t in range(T):
                ds_grad[t] = dv_grads[li + 1][t] @ next_layer.W_in.T

        recurrent_dv = xp.zeros((B, layer.n_post),
                                 dtype=layer.W_in.dtype)
        recurrent_ds = xp.zeros((B, layer.n_post),
                                 dtype=layer.W_in.dtype)

        for t in range(T - 1, -1, -1):
            ds_total = ds_grad[t] + recurrent_ds
            surrogate_t = atan_surrogate(
                v_layer[t] - layer.threshold, alpha=alpha, xp=xp,
            )
            dv_t = ds_total * surrogate_t + recurrent_dv
            dv_grads[li][t] = dv_t
            if t > 0:
                s_prev = s_layer[t - 1]
                v_prev = v_layer[t - 1]
                recurrent_dv = dv_t * layer.leak * (1.0 - s_prev)
                recurrent_ds = -dv_t * layer.leak * v_prev

        if li == 0:
            x_pre = inputs
        else:
            x_pre = spikes[li - 1]
        for t in range(T):
            weight_grads[li] += x_pre[t].T @ dv_grads[li][t]

    input_grad = xp.zeros((T, B, V_in), dtype=layers[0].W_in.dtype)
    for t in range(T):
        input_grad[t] = dv_grads[0][t] @ layers[0].W_in.T
    return weight_grads, input_grad
