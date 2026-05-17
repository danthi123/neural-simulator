"""Pure autoregressive sampler over a NgramTeacher's dense soft
distribution -- the Generator-E runtime generative model. The
NgramTeacher (sim.ngram_teacher) is reused UNMODIFIED; this only
samples from its `soft_dist`. Pure numpy/stdlib; CPU-unit-testable;
self-contained at runtime (count tables + BPE JSON only)."""
from __future__ import annotations
import numpy as np


def ngram_sample_next(teacher, ctx, rng, temperature: float = 1.0) -> int:
    """Sample the next token id from teacher.soft_dist(ctx).

    temperature == 0 -> deterministic argmax (stable FIRST max).
    temperature  > 0 -> sample from the temperature-reweighted
                        distribution p ~ q ** (1/T), renormalized.
    Degenerate / non-finite q is made safe (never raises; always an
    in-range int)."""
    q = np.asarray(teacher.soft_dist(ctx), dtype=np.float64).reshape(-1)
    V = q.shape[0]
    if V == 0:
        return 0
    q = np.where(np.isfinite(q), q, 0.0)
    q = np.clip(q, 0.0, None)
    if q.sum() <= 0:
        q = np.full(V, 1.0 / V)
    if temperature is None or temperature <= 0.0:
        return int(np.argmax(q))
    z = np.power(q, 1.0 / float(temperature))
    s = z.sum()
    if not np.isfinite(s) or s <= 0.0:
        return int(np.argmax(q))
    return int(rng.choice(V, p=z / s))


def ngram_generate(teacher, prompt_ids, n_tokens, rng,
                    temperature: float = 1.0):
    """Autoregressive generation. ctx = trailing up-to-2 ids of
    (prompt + generated-so-far) -- the trigram context the
    NgramTeacher backs off over. Returns ONLY the generated id list
    (prompt excluded)."""
    seq = list(prompt_ids)
    out = []
    for _ in range(int(n_tokens)):
        ctx = tuple(seq[-2:])
        nxt = ngram_sample_next(teacher, ctx, rng, temperature)
        seq.append(nxt)
        out.append(nxt)
    return out
