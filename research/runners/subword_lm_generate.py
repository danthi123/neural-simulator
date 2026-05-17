"""Generator-S autoregressive generation. `sample_next` is PURE
(temperature sampling, unit-tested). `generate` REUSES the validated
forward_unroll_xp (DRY) with the SAME logits semantics as training
(last-layer spike sum over the T-window -> next token). ASCII only."""
from __future__ import annotations
import numpy as np


def sample_next(logits, rng, temperature: float = 1.0) -> int:
    """Pure next-token choice from a (V,) logits vector.

    temperature == 0  -> deterministic argmax (stable first-max).
    temperature  > 0  -> categorical over softmax(logits / temperature).
    Degenerate inputs (all-equal, non-finite) fall back to argmax over
    the finite entries (never raises, never returns out-of-range)."""
    lg = np.asarray(logits, dtype=np.float64).reshape(-1)
    V = lg.shape[0]
    if V == 0:
        return 0
    finite = np.isfinite(lg)
    if not finite.any():
        return 0
    lg = np.where(finite, lg, -np.inf)
    if temperature is None or temperature <= 0.0:
        return int(np.argmax(lg))            # argmax; first-max on ties
    z = lg / float(temperature)
    z = z - np.max(z[np.isfinite(z)])
    ez = np.where(np.isfinite(z), np.exp(z), 0.0)
    s = ez.sum()
    if not np.isfinite(s) or s <= 0.0:
        return int(np.argmax(lg))
    p = ez / s
    return int(rng.choice(V, p=p))


def generate(layers, tok, prompt, n_tokens, T, xp=None, rng=None,
             temperature: float = 1.0):
    """Autoregressive generation. REUSES forward_unroll_xp (validated,
    DRY). Returns (token_ids, decoded_str). Same logits semantics as
    training: logits = last-layer spike sum over the T-step window."""
    from sim.bptt_snn_gpu import forward_unroll_xp, _get_backend
    if xp is None:
        xp, _ = _get_backend()
    if rng is None:
        rng = np.random.default_rng(0)
    V = tok.vocab_size
    ids = list(tok.encode(prompt))
    if not ids:
        ids = [0]
    out = []
    for _ in range(int(n_tokens)):
        ctx = ids[-T:]
        if len(ctx) < T:                      # left-pad with 0 (<UNK>)
            ctx = [0] * (T - len(ctx)) + ctx
        oh = np.zeros((T, 1, V), dtype=np.float32)
        for t, tid in enumerate(ctx):
            if 0 <= tid < V:
                oh[t, 0, tid] = 1.0
        x = xp.asarray(oh) if xp.__name__ == "cupy" else oh
        st = forward_unroll_xp(x, layers, xp=xp)
        logits = st["spikes"][-1].sum(axis=0)        # (1, V)
        logits = logits.get() if hasattr(logits, "get") else logits
        nxt = sample_next(logits[0], rng, temperature)
        ids.append(nxt)
        out.append(nxt)
    return out, tok.decode(out)
