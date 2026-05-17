"""predictive_coding: net-new top-down predictive-coding core.

Pure, deterministic, backend-agnostic (numpy). A recurrent pc_state
encodes the sequence-so-far (prefix); a learned top-down predictor
maps pc_state -> next-concept logits; the Rao-Ballard prediction
error (realized - predicted) is the order-sensitive learning signal
the recognition-only G.20 substrate provably does NOT provide
(Rao & Ballard 1999; Friston active inference; Bastos 2012).

This module ONLY computes predictions/errors/updates on its own
small weights -- it never touches a bridge and never feeds
non-specific activity into concept pools (the v12/v13/v15/G1
"first, do no harm" lesson). The substrate stays UNCHANGED.
"""
from __future__ import annotations
import numpy as np


class PredictiveCoder:
    def __init__(self, n_concepts: int, state_dim: int = 64,
                 seed: int = 42, leak: float = 0.9):
        self.n_concepts = int(n_concepts)
        self.state_dim = int(state_dim)
        self.seed = int(seed)
        self.leak = float(leak)
        rng = np.random.default_rng(seed)
        # W_in: concept one-hot -> state increment; W_pred: state ->
        # next-concept logits. Small init (the predictor is the only
        # learned machinery; substrate untouched).
        self.W_in = rng.normal(
            0.0, 0.1, (n_concepts, state_dim)).astype(np.float32)
        self.W_pred = rng.normal(
            0.0, 0.1, (state_dim, n_concepts)).astype(np.float32)
        self.state = np.zeros(state_dim, dtype=np.float32)
        self._intention: list = []

    def reset(self, intention: list) -> None:
        self.state = np.zeros(self.state_dim, dtype=np.float32)
        self._intention = [int(c) for c in intention]

    def update_state(self, realized_concept_idx: int) -> None:
        c = int(realized_concept_idx)
        if not (0 <= c < self.n_concepts):
            raise IndexError(
                "concept idx %d out of [0,%d)" % (c, self.n_concepts))
        # leaky recurrent prefix accumulation (order-dependent)
        self.state = (self.leak * self.state
                      + self.W_in[c]).astype(np.float32)

    def predict_next(self) -> np.ndarray:
        """Top-down generative prediction: pc_state -> next-concept
        logits. Pure/deterministic."""
        return (self.state @ self.W_pred).astype(np.float32)

    def prediction_error(self, realized_next_idx: int) -> np.ndarray:
        """Rao-Ballard residual = softmax(predicted) - onehot(realized)
        = the stabilized CE gradient w.r.t. logits. Reuses
        sim.bptt_snn.softmax_grad_np (log-sum-exp stable since the
        Inc-3 fix). Order-sensitive: depends on pc_state (the prefix)."""
        from sim.bptt_snn import softmax_grad_np
        logits = self.predict_next().reshape(1, -1)
        return softmax_grad_np(logits, int(realized_next_idx))[0]
