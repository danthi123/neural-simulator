"""song_hvc: a songbird-HVC-style sparse SEQUENTIAL CONTROLLER.

Pure, deterministic, backend-agnostic (numpy). A synfire-like chain:
exactly one state active per step (Hahnloser et al. 2002). Each state
holds a learnable association to a concept index; the babble +
dopamine-reinforce loop (Fee & Goldberg 2011) shapes that association.
This module ONLY decides "ignite concept k at step t" -- it never
touches a bridge and never feeds activity back into concept pools
(the v12/v13/v15 dlpfc failure mode: "first, do no harm").
"""
from __future__ import annotations
import numpy as np


class SongHVC:
    def __init__(self, n_states: int, n_concepts: int, seed: int = 42):
        self.n_states = int(n_states)
        self.n_concepts = int(n_concepts)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        # state -> concept association weights (the learnable map).
        self.W = rng.normal(0.0, 0.01,
                            (n_states, n_concepts)).astype(np.float32)
        self._state = -1
        self._intention = 0

    def reset(self, intention: int = 0) -> None:
        self._state = 0
        self._intention = int(intention)

    def step(self) -> dict:
        s = self._state
        if s < 0 or s >= self.n_states:
            return {"state": -1, "concept": -1}
        concept = int(np.argmax(self.W[s]))
        self._state = s + 1
        return {"state": s, "concept": concept}
