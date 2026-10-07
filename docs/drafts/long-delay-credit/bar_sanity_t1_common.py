"""Shared builder for the T1 (long-delay credit, Step 4b) engagement-bar-(a) bar-sanity specs.

The decision is the pilot's own code: `lhb_window_rates` (per-trial LHb rate from a spike-count key, split rewarded /
unrewarded) and `engagement_a` (the factor-2 comparison), exactly the two calls `run_condition` makes at the lines that
fill `lhb_rate_rewarded` / `lhb_engagement_a_rewardwin`. Only the glue (which key, which window) is chosen here.

A synthetic run is the list of TRAIN trials {"rewarded", "lhb_spikes", "lhb_spikes_rewardwin"} that `run_trial` records,
built from an LHb that fires at `BASE_HZ` per neuron (the ~63 Hz measured on unrewarded trials, A0.1 table), Poisson
counts per segment of the outcome window: [0, onset) | reward window [onset, onset + REWARD_MS) | the rest.
Brains, by what the reward does to the LHb on a REWARDED trial:
    silent_in_window  complete silencing for the 50 ms the reward is delivered (the A0.1 "perfect brain")
    silent_from_onset complete silencing from reward onset to the end of the outcome window (the strongest brain the
                      circuit allows: before onset both trial types get the same drive, so nothing earlier can differ)
    none              no reward-time change (the null brain)
    weak              17% suppression in the window (the round-1 release arm; reported, not thresholded)
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners import _ldc_omission_pilot as M  # noqa: E402
from research.runners._ldc_credit_omission import CreditParams  # noqa: E402

N_LHB = CreditParams().n_lhb          # 20
BASE_HZ = 63.0                        # LHb mean rate per neuron on unrewarded trials (A0.1, rounds 1-3)
N_TRAIN = 200                         # the n-train every Step 4b reading needs (A0.1)

# (outcome_ms, reward_on_ms): the Step 4a timing the tune uses, and the registered default
TIMING_4A = (250, 150)
TIMING_REGISTERED = (100, 30)


def trial_counts(rng, rewarded, profile, outcome_ms, onset, gain=1.0):
    """Poisson spike counts (whole window, reward window) for one trial."""
    reward_ms = M.REWARD_MS
    per_ms = BASE_HZ * gain * N_LHB / 1000.0
    pre, rw, post = onset, reward_ms, outcome_ms - onset - reward_ms
    f_rw = f_post = 1.0
    if rewarded:
        if profile == "silent_in_window":
            f_rw = 0.0
        elif profile == "silent_from_onset":
            f_rw = f_post = 0.0
        elif profile == "weak":
            f_rw = 0.83
    n_pre = rng.poisson(per_ms * pre)
    n_rw = rng.poisson(per_ms * rw * f_rw)
    n_post = rng.poisson(per_ms * post * f_post)
    return int(n_pre + n_rw + n_post), int(n_rw)


def make_run(rng, profile, timing, gain_sd=0.0):
    """TRAIN trials of one run. gain_sd > 0: a per-trial lognormal drive gain independent of reward (the random brain)."""
    outcome_ms, onset = timing
    p_rew = float(rng.uniform(0.3, 0.7))
    trials = []
    for _ in range(N_TRAIN):
        rewarded = bool(rng.random() < p_rew)
        gain = float(np.exp(rng.normal(0.0, gain_sd))) if gain_sd else 1.0
        tot, rw = trial_counts(rng, rewarded, profile, outcome_ms, onset, gain)
        trials.append({"rewarded": rewarded, "lhb_spikes": tot, "lhb_spikes_rewardwin": rw})
    return {"trials": trials, "outcome_ms": outcome_ms, "reward_on_ms": onset}


def _set_timing(data):
    M.OUTCOME_MS, M.REWARD_ON_MS = data["outcome_ms"], data["reward_on_ms"]


def decide_whole_window(data):
    """The bar as REGISTERED in STEP4B section 2: LHb rate on unrewarded trials > 2 x on rewarded, both over the whole
    outcome window (the `lhb_rate_rewarded` / `lhb_rate_unrewarded` fields)."""
    _set_timing(data)
    rew, unrew = M.lhb_window_rates(data["trials"], "lhb_spikes", N_LHB, M.OUTCOME_MS, True)
    return M.engagement_a(M._mean(rew), M._mean(unrew))["pass"] is True


def decide_reward_window(data):
    """The bar as AMENDED by A0.1: the same comparison over the reward-time window (`lhb_engagement_a_rewardwin`)."""
    _set_timing(data)
    rew, unrew = M.lhb_window_rates(data["trials"], "lhb_spikes_rewardwin", N_LHB, M.REWARD_MS, True)
    return M.engagement_a(M._mean(rew), M._mean(unrew))["pass"] is True
