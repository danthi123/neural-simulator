"""Biologically-canonical TD(lambda) value-function critic on the
complete-serial-compound representation (Schultz98 / Sutton-Barto).
delta = r + gamma*V(s') - V(s) is the phasic-DA TD error -- the missing
"value-function critic of an actor-critic" (feature-catalog C.30). Pure
array math; reuses sim.kernels.fused_eligibility_trace_decay UNMODIFIED
for the trace decay; NO automatic differentiation (TD needs none);
deterministic seeded; ASCII only."""
from __future__ import annotations
import numpy as np
from sim.kernels import fused_eligibility_trace_decay  # REUSED UNMODIFIED

# Pre-registered Pavlovian-schedule constants (canonical Schultz trace
# conditioning; NOT science bars -- the frozen bars live in
# td_critic_core). Cue onset is JITTERED per trial (the cue's
# APPEARANCE must be temporally unpredicted, else delta->0 everywhere
# at convergence and the transfer is unmeasurable).
T = 20
CS_ONSETS = (3, 4, 5, 6, 7)
TRACE = 4
GAMMA = 0.95
ALPHA = 0.05
LAMBDA = 0.9
N_TRIALS = 1500
EARLY = 100
LATE = 100


def analytic_vstar(trace: int = TRACE, gamma: float = GAMMA):
    """Exact true expected discounted return GIVEN the cue, along the
    cue-anchored timeline: deterministic reward 1.0 at tap=trace."""
    return np.array([gamma ** (trace - k) for k in range(trace + 1)])


def csc_features(onset: int, T: int = T):
    """Bias feature (constant; the pre-cue baseline the critic CANNOT
    use to predict the uncertain cue onset) + one tap per
    time-since-cue. tap=t-onset; pre-cue (t<onset) is bias-only."""
    n_feat = T + 1
    X = np.zeros((T, n_feat))
    X[:, T] = 1.0
    for t in range(T):
        tap = t - onset
        if 0 <= tap < T:
            X[t, tap] = 1.0
    return X


def scale_free_transfer(dcs_abs: float, dus_abs: float) -> float:
    """Canonical scale-free Schultz transfer = fraction of asymptotic
    RPE now at the (unpredicted) CS vs the US. -> 1.0 for a perfect
    critic BY MATHEMATICAL IDENTITY (dUS->0 => fraction->1); ungameable
    (no fitted denominator)."""
    return dcs_abs / (dcs_abs + dus_abs + 1e-12)


def run_pavlovian(mode: str, seed: int, n_trials: int = N_TRIALS):
    """One critic run with PER-TRIAL JITTERED cue onset. Returns
    (vrmse_vs_analytic_vstar, scale_free_transfer, us_decay).
    modes: 'td' | 'no_bootstrap' | 'permuted' | 'wrongsign'."""
    rng = np.random.default_rng(seed)
    n_feat = T + 1
    w = np.zeros(n_feat)
    early_dUS, late_dUS, late_dCS = [], [], []
    decay = GAMMA * LAMBDA
    for trial in range(n_trials):
        if mode == "permuted":
            onset = int(rng.choice(CS_ONSETS))
            t_us = CS_ONSETS[len(CS_ONSETS) // 2] + TRACE  # cue uninformative
        else:
            onset = int(rng.choice(CS_ONSETS))
            t_us = onset + TRACE
        X = csc_features(onset, T)
        e = np.zeros(n_feat)
        for t in range(T):
            r = 1.0 if t == t_us else 0.0
            v_t = X[t] @ w
            v_tp1 = (X[t + 1] @ w) if t + 1 < T else 0.0
            if mode == "no_bootstrap":
                delta = r - v_t
            else:
                delta = r + GAMMA * v_tp1 - v_t
            # eligibility: e = gamma*lambda*e + phi(s). The decay term
            # reuses the project's eligibility kernel UNMODIFIED.
            e = np.asarray(fused_eligibility_trace_decay(e, decay)) + X[t]
            step = -delta if mode == "wrongsign" else delta
            w = w + ALPHA * step * e
            if trial < EARLY and t == t_us:
                early_dUS.append(abs(delta))
            if trial >= n_trials - LATE:
                if t == t_us:
                    late_dUS.append(abs(delta))
                if t == onset - 1:           # the UNPREDICTED cue arrival
                    late_dCS.append(delta)
    Vstar = analytic_vstar()
    wt = w[:TRACE + 1] + w[T]
    vrmse = float(np.sqrt(np.mean((wt - Vstar) ** 2)))
    e_us = float(np.mean(early_dUS)) if early_dUS else 0.0
    l_us = float(np.mean(late_dUS)) if late_dUS else 0.0
    l_cs = float(np.mean(np.abs(late_dCS))) if late_dCS else 0.0
    transfer = scale_free_transfer(l_cs, l_us)
    us_decay = l_us / (e_us + 1e-9)
    return vrmse, transfer, us_decay
