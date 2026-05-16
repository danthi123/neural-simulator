"""Phase 2.1 BPTT through SNN -- numpy reference implementation.

ONLY ON path-f-hybrid BRANCH.

CPU-runnable BPTT through a multi-layer LIF network with ATan
surrogate gradient. Used for:
1. Toy ABC task validation (Phase 2.1 smoke)
2. Gradient check vs finite differences
3. Reference for the future GPU port

Network architecture (per design):
  input (one-hot, V_in dim) ->
  cortex_l1 (H1 LIF neurons) ->
  cortex_l2 (H2 LIF neurons) ->
  output (V_out dim, log-softmax over class)

Forward pass: standard LIF with tau_v leak, threshold, hard reset.
Spikes computed as Heaviside (binary).

Backward pass: Manual BPTT through T unrolled timesteps. ATan
surrogate replaces the non-differentiable Heaviside in gradient
flow.

Loss: cross-entropy on summed-over-time output spikes (rate code).

This is a REFERENCE implementation. The GPU version (CuPy) will
follow once this validates on the ABC task.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import math

import numpy as np


def atan_surrogate_np(v_minus_threshold: np.ndarray, alpha: float = 2.0) -> np.ndarray:
    """ATan surrogate gradient (numpy version).

    dspike/dv = (1/pi) * (1 / (1 + (alpha * (v-thr))^2))
    """
    return (1.0 / math.pi) * (1.0 / (1.0 + (alpha * v_minus_threshold) ** 2))


@dataclass
class LIFLayer:
    """Single LIF layer with input weights W_in.

    State (per timestep):
      v: membrane potential (n_post,)
      s: spikes (n_post,) binary

    Dynamics:
      v(t) = leak * v(t-1) * (1 - s(t-1))  # hard reset on spike
            + W_in @ x(t)
      s(t) = Heaviside(v(t) - threshold)
    """
    W_in: np.ndarray              # (n_pre, n_post)
    n_post: int
    threshold: float = 1.0
    leak: float = 0.95            # exp(-dt/tau_v) with dt=1ms, tau=20ms

    def init_state(self, batch_size: int = 1) -> dict:
        return {
            "v": np.zeros((batch_size, self.n_post), dtype=np.float32),
            "s": np.zeros((batch_size, self.n_post), dtype=np.float32),
        }


def forward_step(
    state: dict,
    x: np.ndarray,                # (B, n_pre)
    layer: LIFLayer,
) -> Tuple[dict, np.ndarray]:
    """One timestep forward through a LIF layer.

    Returns (new_state, spikes).
    """
    # Reset on previous spike: v(t) = leak * v(t-1) * (1 - s(t-1))
    v_pre = layer.leak * state["v"] * (1.0 - state["s"])
    # Add input current
    v_new = v_pre + x @ layer.W_in
    # Spikes (Heaviside)
    s_new = (v_new >= layer.threshold).astype(np.float32)
    return {"v": v_new, "s": s_new}, s_new


def forward_unroll(
    inputs: np.ndarray,              # (T, B, V_in)
    layers: list[LIFLayer],
) -> dict:
    """Unroll T timesteps through stacked LIF layers.

    Returns dict with:
      "spikes": list of (T, B, n_post) per layer
      "v": list of (T, B, n_post) per layer (membrane potentials)
    """
    T, B, _ = inputs.shape
    L = len(layers)
    states = [layer.init_state(B) for layer in layers]

    spikes_per_layer = [np.zeros((T, B, layer.n_post), dtype=np.float32)
                         for layer in layers]
    v_per_layer = [np.zeros((T, B, layer.n_post), dtype=np.float32)
                    for layer in layers]

    for t in range(T):
        x_in = inputs[t]
        for li, layer in enumerate(layers):
            states[li], s = forward_step(states[li], x_in, layer)
            spikes_per_layer[li][t] = s
            v_per_layer[li][t] = states[li]["v"]
            x_in = s  # Output spikes feed next layer

    return {"spikes": spikes_per_layer, "v": v_per_layer}


def cross_entropy_loss_np(logits: np.ndarray, target_idx: int) -> float:
    """logits: (B, V_out). target_idx: int. Returns mean loss across batch.

    Numerically stabilized via the log-sum-exp trick (subtract per-row
    max before exp). softmax/CE are shift-invariant, so this is
    mathematically exact -- it only prevents exp() overflow when logits
    are large (e.g. rate-coded logits = sum of spikes over a long BPTT
    unroll: at T=96 raw logits reach ~96, exp(96)~5e41 overflows).
    """
    m = np.max(logits, axis=-1, keepdims=True)
    log_sum_exp = m + np.log(
        np.sum(np.exp(logits - m), axis=-1, keepdims=True))
    log_probs = logits - log_sum_exp
    return float(-log_probs[:, target_idx].mean())


def softmax_grad_np(logits: np.ndarray, target_idx: int) -> np.ndarray:
    """Gradient dL/d_logits for softmax + cross-entropy.
    logits: (B, V_out). Returns same shape.

    Same log-sum-exp stabilization as cross_entropy_loss_np (exact;
    softmax is shift-invariant).
    """
    m = np.max(logits, axis=-1, keepdims=True)
    e = np.exp(logits - m)
    probs = e / np.sum(e, axis=-1, keepdims=True)
    grad = probs.copy()
    grad[:, target_idx] -= 1.0
    return grad / logits.shape[0]  # batch-mean


# ABC dataset
def make_abc_dataset(
    n_samples: int = 1000,
    seq_len: int = 30,
    rng: Optional[np.random.Generator] = None,
):
    """Generate ABC sequence dataset.

    Each sample is a sequence ABCABCABC... with random offset.
    Predict next token given previous token (one-hot input).

    Returns:
        inputs: (n_samples, seq_len-1, 3) one-hot of token at position t
        targets: (n_samples, seq_len-1) integer class of token at position t+1
    """
    if rng is None:
        rng = np.random.default_rng()
    inputs = np.zeros((n_samples, seq_len - 1, 3), dtype=np.float32)
    targets = np.zeros((n_samples, seq_len - 1), dtype=np.int64)
    for i in range(n_samples):
        offset = rng.integers(0, 3)
        seq = np.array([(offset + j) % 3 for j in range(seq_len)],
                        dtype=np.int64)
        # input[t] = one-hot of seq[t]; target[t] = seq[t+1]
        for t in range(seq_len - 1):
            inputs[i, t, seq[t]] = 1.0
            targets[i, t] = seq[t + 1]
    return inputs, targets


def backward_unroll(
    inputs: np.ndarray,              # (T, B, V_in)
    layers: list,
    forward_state: dict,             # output of forward_unroll
    output_grad: np.ndarray,         # (T, B, V_out) dL/d_output_spikes
    alpha: float = 2.0,
):
    """BPTT backward pass through stacked LIF layers.

    Computes weight gradients dL/dW_in for each layer.

    Forward used HARD reset: v(t) = leak * v(t-1) * (1 - s(t-1)) + W @ x(t).
    Backward uses surrogate gradient through Heaviside spikes and
    chain-rule through the (1 - s(t-1)) reset factor.

    Returns:
        weight_grads: list of (n_pre, n_post) -- dL/dW_in per layer
        input_grad: (T, B, V_in) -- dL/d_inputs (for testing)
    """
    T, B, V_in = inputs.shape
    L = len(layers)
    spikes = forward_state["spikes"]
    v_per = forward_state["v"]

    weight_grads = [np.zeros_like(layer.W_in) for layer in layers]
    dv_grads = [np.zeros((T, B, layer.n_post), dtype=np.float32)
                for layer in layers]

    for li in range(L - 1, -1, -1):
        layer = layers[li]
        s_layer = spikes[li]
        v_layer = v_per[li]

        # ds_grad[t]: gradient on this layer's spikes at t
        if li == L - 1:
            ds_grad = output_grad.copy()
        else:
            next_layer = layers[li + 1]
            ds_grad = np.zeros((T, B, layer.n_post), dtype=np.float32)
            for t in range(T):
                ds_grad[t] = dv_grads[li + 1][t] @ next_layer.W_in.T

        # Recurrent contributions from t+1 -> t
        recurrent_dv = np.zeros((B, layer.n_post), dtype=np.float32)
        recurrent_ds = np.zeros((B, layer.n_post), dtype=np.float32)

        for t in range(T - 1, -1, -1):
            ds_total = ds_grad[t] + recurrent_ds
            surrogate_t = atan_surrogate_np(
                v_layer[t] - layer.threshold, alpha=alpha
            )
            dv_t = ds_total * surrogate_t + recurrent_dv
            dv_grads[li][t] = dv_t

            # Compute recurrent contributions for t-1
            if t > 0:
                s_prev = s_layer[t - 1]
                v_prev = v_layer[t - 1]
                # dv[t]/dv[t-1] direct = leak * (1 - s[t-1])
                recurrent_dv = dv_t * layer.leak * (1.0 - s_prev)
                # dv[t]/ds[t-1] = -leak * v[t-1]
                recurrent_ds = -dv_t * layer.leak * v_prev

        # Accumulate weight gradient
        if li == 0:
            x_pre = inputs
        else:
            x_pre = spikes[li - 1]
        for t in range(T):
            weight_grads[li] += x_pre[t].T @ dv_grads[li][t]

    # Input gradient
    input_grad = np.zeros((T, B, V_in), dtype=np.float32)
    for t in range(T):
        input_grad[t] = dv_grads[0][t] @ layers[0].W_in.T

    return weight_grads, input_grad


if __name__ == "__main__":
    # Quick smoke: build 2-layer SNN, forward-pass on ABC dataset
    rng = np.random.default_rng(42)
    inputs, targets = make_abc_dataset(n_samples=10, seq_len=30, rng=rng)
    print(f"Dataset: inputs {inputs.shape}, targets {targets.shape}")

    # Build 2-layer net: 3 -> 64 -> 3
    layers = [
        LIFLayer(W_in=rng.normal(0, 0.5, (3, 64)).astype(np.float32),
                 n_post=64),
        LIFLayer(W_in=rng.normal(0, 0.5, (64, 3)).astype(np.float32),
                 n_post=3),
    ]

    # Forward unroll on first sample. inputs[0] is (seq_len-1, V_in);
    # need to add batch axis at position 1 -> (T, B=1, V_in).
    sample_input = inputs[0][:, None, :]  # (T=29, B=1, V_in=3)
    out = forward_unroll(sample_input, layers)
    print(f"Layer 0 spikes shape: {out['spikes'][0].shape}")
    print(f"Layer 1 spikes shape: {out['spikes'][1].shape}")
    print(f"Final layer mean firing rate: "
          f"{out['spikes'][1].mean():.3f}")

    # Backward smoke: simulate output gradient (target=class 1)
    output_grad = np.zeros((sample_input.shape[0], 1, 3), dtype=np.float32)
    output_grad[:, :, 1] = 1.0  # push toward class 1 every step
    weight_grads, input_grad = backward_unroll(
        sample_input, layers, out, output_grad
    )
    print(f"Layer 0 weight grad shape: {weight_grads[0].shape}, "
          f"max abs: {np.abs(weight_grads[0]).max():.4f}")
    print(f"Layer 1 weight grad shape: {weight_grads[1].shape}, "
          f"max abs: {np.abs(weight_grads[1]).max():.4f}")
