"""Compositional A->B binding bridging a TEMPORAL GAP, learned by the
VALIDATED TD(lambda)+eligibility mechanism. The eligibility decay is
routed through the REUSED sim.kernels.fused_eligibility_trace_decay
(UNMODIFIED) so this runs in the sim's real eligibility substrate.
hebbian_no_trace = identical to td EXCEPT the eligibility trace is
zeroed every gap step (the faithful v16-cold-start-analog: no
temporal-credit mechanism to carry the A-time decision across the
gap). NO automatic differentiation; deterministic seeded; ASCII only.

Honest ceiling: a PASS validates the MECHANISM (temporal credit
bridges the bind-gap where the no-trace analog cannot) -- NOT
composition-solved, NOT compositional language, NOT scaled/integrated;
that is a SEPARATE later gated increment."""
from __future__ import annotations
import numpy as np
from sim.kernels import fused_eligibility_trace_decay  # REUSED UNMODIFIED

# Pre-registered schedule constants (NOT science bars -- the frozen
# bars live in compose_bind_core). N=12 is BY DESIGN: it makes the
# compositional task strictly harder (chance 1/12 ~ 0.083) and makes
# the control chance-distribution provably tight (the small-N
# absolute-bar artifact is structurally excluded).
_N = 12             # |A| = |B|; chance = 1/12 ~ 0.083
_GAP = 6            # temporal gap between the A-time decision & reward
_GAMMA = 0.95
_LAMBDA = 0.9
_ALPHA = 0.1
_EPS = 0.1          # epsilon-greedy exploration
_N_TRIALS = 8000


def run_bind(mode: str, seed: int, gap: int) -> float:
    """One run. A_i -> B_{pi(i)} bijection; reward arrives `gap` steps
    after the A-time eps-greedy decision; credit must bridge the gap
    via the eligibility trace. Returns greedy accuracy over all A.
    modes: 'td' | 'hebbian_no_trace' | 'permuted' | 'wrongsign'."""
    rng = np.random.default_rng(seed)
    pi = rng.permutation(_N)                  # the fixed compositional rule
    W = np.zeros((_N, _N))
    decay = _GAMMA * _LAMBDA
    for _t in range(_N_TRIALS):
        pi_eff = rng.permutation(_N) if mode == "permuted" else pi
        i = int(rng.integers(_N))
        if rng.random() < _EPS:
            b = int(rng.integers(_N))
        else:
            b = int(np.argmax(W[i]))
        e = np.zeros((_N, _N))
        e[i, b] = 1.0                         # eligibility set at decision
        q_dec = W[i, b]
        for _g in range(gap):
            if mode == "hebbian_no_trace":
                e[:] = 0.0                    # the faithful v16-analog
            else:
                # REUSED sim eligibility kernel (UNMODIFIED): the
                # decay term gamma*lambda*e. Numerically identical to
                # the validated probe's inline e*=gamma*lambda.
                e = np.asarray(fused_eligibility_trace_decay(e, decay))
        r = 1.0 if b == pi_eff[i] else 0.0
        delta = r + _GAMMA * 0.0 - q_dec      # terminal TD error
        step = -delta if mode == "wrongsign" else delta
        W = W + _ALPHA * step * e
    greedy = np.argmax(W, axis=1)
    return float(np.mean(greedy == pi))
