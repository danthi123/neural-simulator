"""Pure smoothed back-off trigram LM over BPE token ids -- the
Generator-D distillation teacher. Stdlib Counter + numpy ONLY (zero
new deps, zero external weights; a statistical model of the
user-authorized corpus -> in-constraints, training-time only).
Grounded competent: held-out ppl 14.3 vs uniform-random 513 on
TinyStories (probe ba1jyepwf). Deterministic. soft_dist returns the
dense soft target (length-V probability vector summing to 1)."""
from __future__ import annotations
from collections import Counter, defaultdict
import numpy as np


class NgramTeacher:
    def __init__(self):
        self._uni = Counter()
        self._bi = defaultdict(Counter)
        self._trg = defaultdict(Counter)
        self._V = 0
        self._k = 0.1

    def train(self, train_ids, vocab_size: int, k: float = 0.1) -> None:
        self._V = int(vocab_size)
        self._k = float(k)
        ti = list(train_ids)
        self._uni = Counter(ti)
        self._bi = defaultdict(Counter)
        self._trg = defaultdict(Counter)
        for i in range(len(ti) - 1):
            self._bi[ti[i]][ti[i + 1]] += 1
        for i in range(len(ti) - 2):
            self._trg[(ti[i], ti[i + 1])][ti[i + 2]] += 1

    def soft_dist(self, ctx) -> np.ndarray:
        """Dense length-V soft target. Back-off: trigram if its ctx
        count >= 5, else bigram if >= 2, else unigram; add-k smoothed
        over the FULL vocab so every entry is > 0 and the vector sums
        to 1. ctx may be (), (a,) or (a,b)."""
        V = self._V
        k = self._k
        ctx = tuple(ctx)
        counts = None
        if len(ctx) >= 2:
            c3 = self._trg.get((ctx[-2], ctx[-1]))
            if c3 is not None and sum(c3.values()) >= 5:
                counts = c3
        if counts is None and len(ctx) >= 1:
            c2 = self._bi.get(ctx[-1])
            if c2 is not None and sum(c2.values()) >= 2:
                counts = c2
        if counts is None:
            counts = self._uni
        tot = sum(counts.values()) + k * V
        q = np.full(V, k / tot, dtype=np.float64)
        for w, c in counts.items():
            if 0 <= w < V:
                q[w] = (c + k) / tot
        s = q.sum()
        if s > 0:
            q = q / s
        return q
