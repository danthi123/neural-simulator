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
        self._bias: dict = {}

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

    def rollout(self, intention: int, length: int) -> list:
        self.reset(intention)
        out = []
        for _ in range(length):
            st = self.step()
            if st["state"] < 0:
                break
            # intention bias steers which concept this state emits
            bias = self._bias.get(
                (intention, st["state"]), None)
            out.append(bias if bias is not None else st["concept"])
        return out

    def set_intention_bias(self, intention: int,
                           concept_seq: list) -> None:
        if not hasattr(self, "_bias"):
            self._bias = {}
        for t, k in enumerate(concept_seq):
            self._bias[(int(intention), t)] = int(k)

    def babble(self, base_seq: list, rng, temperature: float) -> list:
        """LMAN-like exploratory variability: with prob ~temperature
        replace ONE slot's concept with a random one. Deterministic
        given `rng`. temperature=0 -> exact replay (no exploration)."""
        cand = list(base_seq)
        if temperature <= 0.0 or not cand:
            return cand
        if rng.random() < float(temperature):
            i = int(rng.integers(0, len(cand)))
            cand[i] = int(rng.integers(0, self.n_concepts))
        return cand

    def reinforce(self, intention: int, concept_seq: list,
                  reward: float, lr: float) -> None:
        """Three-factor (eligibility x dopamine) update: reward * lr
        added to W[state, emitted_concept] for each slot. reward<=0 ->
        no change (DA gate). Bounded by tanh squashing to keep the
        argmax map stable (no runaway)."""
        r = float(reward)
        if r <= 0.0:
            return
        for t, k in enumerate(concept_seq):
            if 0 <= t < self.n_states and 0 <= k < self.n_concepts:
                self.W[t, k] += float(lr) * r
        np.tanh(self.W, out=self.W)
