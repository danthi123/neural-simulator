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
        # learnable output bias: makes the start-of-sequence ([] prefix,
        # zero pc_state) transition learnable (its CE gradient is `err`,
        # always nonzero -- textbook softmax/LM output-layer bias).
        self.b_pred = np.zeros(n_concepts, dtype=np.float32)
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
        logits, with a learnable output bias so the start-of-sequence
        ([] prefix, zero state) transition is learnable. Pure/det."""
        return (self.state @ self.W_pred + self.b_pred).astype(np.float32)

    def select_next(self, candidates: list) -> int:
        """Active inference: emit the candidate concept the top-down
        generative model most predicts given the current prefix
        (argmax predicted logit over candidates). Pure."""
        logits = self.predict_next()
        cand = [int(c) for c in candidates
                if 0 <= int(c) < self.n_concepts]
        if not cand:
            raise ValueError("no valid candidates")
        return max(cand, key=lambda c: float(logits[c]))

    def prediction_error(self, realized_next_idx: int) -> np.ndarray:
        """Rao-Ballard residual = softmax(predicted) - onehot(realized)
        = the stabilized CE gradient w.r.t. logits. Reuses
        sim.bptt_snn.softmax_grad_np (log-sum-exp stable since the
        Inc-3 fix). Order-sensitive: depends on pc_state (the prefix)."""
        from sim.bptt_snn import softmax_grad_np
        logits = self.predict_next().reshape(1, -1)
        return softmax_grad_np(logits, int(realized_next_idx))[0]

    def learn(self, prefix: list, target_next_idx: int,
              lr: float) -> None:
        self.reset(self._intention or (list(prefix) + [target_next_idx]))
        # recompute prefix state, tracking the concepts for W_in grad
        self.state = np.zeros(self.state_dim, dtype=np.float32)
        contribs = []
        for c in prefix:
            self.state = (self.leak * self.state
                          + self.W_in[int(c)]).astype(np.float32)
            contribs.append(int(c))
        err = self.prediction_error(int(target_next_idx))   # (n_concepts,)
        # dL/dW_pred = outer(state, err); dL/dstate = W_pred @ err
        gW_pred = np.outer(self.state, err).astype(np.float32)
        dstate = (self.W_pred @ err).astype(np.float32)
        self.W_pred -= lr * gW_pred
        # output-bias grad (softmax-CE): dL/db_pred == err. Always
        # nonzero, so the []->first-concept transition is learnable.
        self.b_pred -= lr * err
        # W_in grad: each prefix concept contributed leak**k * W_in[c]
        # to state; apply the same dstate to the concepts' rows (a
        # 1-step approximation -- sufficient for the cheap P probe).
        for c in set(contribs):
            self.W_in[c] -= lr * dstate
        np.clip(self.W_pred, -5.0, 5.0, out=self.W_pred)
        np.clip(self.b_pred, -5.0, 5.0, out=self.b_pred)
        np.clip(self.W_in, -5.0, 5.0, out=self.W_in)

    def rollout(self, intention: list, length: int,
                candidates: list) -> list:
        """Active-inference rollout: reset to the intention, then for
        each step emit the next concept the top-down generative model
        most predicts (select_next) and feed it back into pc_state.
        Returns the ordered produced concept list. Pure (no bridge)."""
        self.reset(intention)
        produced: list = []
        for _ in range(int(length)):
            c = self.select_next(candidates)
            produced.append(int(c))
            self.update_state(int(c))
        return produced
