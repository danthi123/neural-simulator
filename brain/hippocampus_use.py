"""UseHippocampus: strength is earned by use inside the hippocampus (reconsolidation-like), not set by hand.

Option, not a default: Hippocampus is unchanged. Differences from the base class:
- a re-telling of an experience the hippocampus already holds does not make a duplicate episode; it re-activates the
  stored one and raises its strength (the trace is labile when it is re-experienced, then re-stored stronger);
- a successful cued retrieval already raises strength in Hippocampus.recall (USE_BOOST); here that is kept.
Sleep (brain/sleep.py) replays episodes in proportion to this learned strength.

Scaffold note (SCAFFOLDS.md): "same experience" is decided by the cosine of the whole stored matrix-memory change
(all layers) against RETELL_SIM; that threshold is a declared constant, not learned.
"""
from __future__ import annotations

import time

import torch

from brain.hippocampus import Episode, Hippocampus


def _flat(delta) -> torch.Tensor:
    return torch.cat([d.flatten().float() for d in delta if d is not None])


class UseHippocampus(Hippocampus):
    RETELL_SIM = 0.98    # cosine of the whole delta above which a telling is the same experience again
    RETELL_BOOST = 1.0   # strength added by each re-telling

    def store(self, cortex, base, text: str, speaker: str = "User", salience: float = 1.0, permanent: bool = False):
        _, st = cortex.read(f"{speaker}: {text}\n\n", base)
        delta = cortex.matrix_delta(st, base)
        flat = _flat(delta).cpu()   # stored episodes live on CPU (Hippocampus.store)
        best, best_sim = None, -1.0
        for e in self.episodes:
            s = float(torch.dot(flat, _flat(e.delta)) / (flat.norm() * _flat(e.delta).norm()).clamp_min(1e-8))
            if s > best_sim:
                best, best_sim = e, s
        self.last_match_sim = best_sim
        if best is not None and best_sim >= self.RETELL_SIM:
            best.strength += self.RETELL_BOOST
            best.uses += 1
            best.last_used = time.time()
            return best
        ep = Episode(text=f"{speaker}: {text}", delta=[d.half().cpu() if d is not None else None for d in delta],
                     key=cortex.key_of(delta).cpu(), strength=salience, permanent=permanent)
        self.episodes.append(ep)
        return ep
