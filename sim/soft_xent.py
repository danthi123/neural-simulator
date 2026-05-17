"""Pure soft cross-entropy loss + gradient -- the ONLY change vs
Generator-S's one-hot CE. soft_xent_loss = -sum_w q_w log softmax(z)_w;
soft_xent_grad = softmax(z) - q (exact d/dz of soft-xent). Numerically
stable (log-sum-exp), mirroring sim.bptt_snn.cross_entropy_loss_np /
softmax_grad_np, of which this is the faithful generalization (equal
when q is one-hot). Pure numpy; CPU-unit-testable."""
from __future__ import annotations
import numpy as np


def _log_softmax(z):
    z = np.asarray(z, dtype=np.float64)
    m = np.max(z, axis=-1, keepdims=True)
    zs = z - m
    return zs - np.log(np.exp(zs).sum(axis=-1, keepdims=True))


# NOTE (load-bearing): both _softmax and _log_softmax cast to float64
# and shift by the per-row max independently; at float64 this makes
# exp(_log_softmax(z)) == _softmax(z) to precision, which is why
# soft_xent_grad (uses _softmax) is the exact derivative of
# soft_xent_loss (uses _log_softmax). Do NOT drop the float64 cast.
def _softmax(z):
    z = np.asarray(z, dtype=np.float64)
    m = np.max(z, axis=-1, keepdims=True)
    e = np.exp(z - m)
    return e / e.sum(axis=-1, keepdims=True)


def _norm_q(q, V):
    """Renormalize a soft target to a valid distribution. DELIBERATE
    fail-safe (load-bearing): non-finite entries are ZEROED (NOT
    treated as dominant) and an all-zero/negative-sum q falls back to
    UNIFORM, so a malformed teacher distribution can never inject a
    NaN/inf into the BPTT gradient and silently corrupt the decisive
    verdict. The NgramTeacher always returns finite positive
    add-k-smoothed probabilities, so this path is defensive only."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    q = np.where(np.isfinite(q), q, 0.0)
    q = np.clip(q, 0.0, None)
    s = q.sum()
    if s <= 0:
        return np.full(V, 1.0 / V)
    return q / s


def soft_xent_loss(logits, q) -> float:
    """logits (1,V), q (V,). Returns the batch-mean soft cross-entropy
    -sum_w q_w log softmax(logits)_w (batch-mean to match
    cross_entropy_loss_np's contract)."""
    lg = np.asarray(logits, dtype=np.float64)
    V = lg.shape[-1]
    _q_arr = np.asarray(q).reshape(-1)
    if _q_arr.size != V:
        raise ValueError(
            "soft_xent: q length %d != logits V %d (teacher/student "
            "vocab mismatch would silently corrupt the gradient)"
            % (_q_arr.size, V))
    qn = _norm_q(q, V)
    ls = _log_softmax(lg)
    return float(-(qn * ls).sum(axis=-1).mean())


def soft_xent_grad(logits, q) -> np.ndarray:
    """d/dlogits of soft_xent_loss = (softmax(logits) - q) / batch.
    Shape (1,V), matching softmax_grad_np's batch-mean convention."""
    lg = np.asarray(logits, dtype=np.float64)
    B, V = lg.shape
    _q_arr = np.asarray(q).reshape(-1)
    if _q_arr.size != V:
        raise ValueError(
            "soft_xent: q length %d != logits V %d (teacher/student "
            "vocab mismatch would silently corrupt the gradient)"
            % (_q_arr.size, V))
    qn = _norm_q(q, V)
    return (_softmax(lg) - qn.reshape(1, V)) / B
