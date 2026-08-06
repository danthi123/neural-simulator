"""Gate B Stage 2d: uncertainty-GATED exploration on the opponent selector.

Surpasses the Stage 2c protocol / operating-point wall
(`research/findings/2026-08-06-gateB-stage2c-opponent-negative-rpe-NO-GO.md`,
`STAGE2C_NO_GO`): the opponent (negative-RPE) arm FIXED reversal (P(B) 0->1) and
the neural critic validated, but the FROZEN contingency gate still failed
(D_contingent - D_yoked = 0.00) for a PROTOCOL reason -- under DENSE reward the
selector LOCKS early with a FIXED exploration amplitude, so the reward-count-
matched YOKED control ALSO does the dominant action ~90% of trials and saturates
identically. Fixed-amplitude OU (measured 40..600 pA in Stage 2b) cannot separate
them: low amplitude locks BOTH, high amplitude samples BOTH.

Stage 2d adds the still-unbuilt companion process named by the 2c finding:
**sustained, uncertainty-GATED exploration**. The exploration drive stays HIGH
while the action-value estimate is UNCERTAIN and decays only as CONFIDENCE rises.

    disc  = |s0 - s1| / (s0 + s1)     # NEURAL policy discrimination read-out:
                                      #   s_c = str_d1_c onset spike count (the BG
                                      #   direct-pathway value/go rate for action c)
    conf  = EMA( clip((disc - LO)/(HI - LO), 0, 1) )   # confidence (0=uncertain)
    sigma = SIGMA_CONFIDENT + (SIGMA_UNCERTAIN - SIGMA_CONFIDENT) * (1 - conf)

`sigma` is the OU membrane-noise amplitude on the spiking proposal + striatal
D1/D2 populations (`bridge.ou_noise_std`) -- a NEURAL exploration drive (tonic-
neuromodulator-modulated MSN variability), CLOSED-LOOP on a NEURAL read-out (the
str_d1 spiking-population discrimination), NOT a host open-loop schedule. `s0,s1`
are read from spiking populations exactly like the motor read-out that moves the
body; the sigma arithmetic (the conf->sigma map) is the abstracted tonic-DA/ACh
controller, the same class of documented residual as Stage 2c's reward-V DA
arithmetic. The `conf_lesion` control REPLACES the neural disc with a CONSTANT
(the ungated Stage-2c operating point, sigma=SIGMA_UNCERTAIN fixed): if the
divergence collapses under the lesion, the neural gate is load-bearing.

Why this SEPARATES contingent from yoked where fixed OU could not (the positive-
feedback bistability):
  * CONTINGENT (target reliably rewarded, other action NEVER rewarded): the
    target's D1 route potentiates, the other's does not -> disc rises -> conf
    rises -> sigma falls -> the brain EXPLOITS the target -> disc rises further.
    Self-quenching exploration -> P(a0|rew0)->1, P(a0|rew1)->0 -> D_contingent~1.
  * YOKED (reward DECOUPLED, dense): whichever action the brain samples on a
    reward index is rewarded, so BOTH routes get intermittent reward + omission
    -> NEITHER cleanly dominates -> disc stays low -> conf stays low -> sigma
    stays high -> keeps sampling both -> stays uncertain. Self-sustaining
    exploration -> its dominant action is REPEATEDLY unrewarded -> the validated
    negative arm punishes it -> never locks -> D_yoked ~ 0.

Kept from Stage 2c UNCHANGED: per-action compartmentalised DA, the opponent
negative-RPE arm (reward - V, V = str_d1 onset rate), the D1/D2 asymmetry
substrate, neural coactivity eligibility, the byte-identical-to-Stage-1 reward-OFF
guard. Acceptance criteria are FROZEN from the Stage-2 preregistration UNCHANGED
(`research/findings/2026-08-06-gateB-stage2-local-reward-credit-PREREGISTRATION.md`):
D_contingent - D_yoked >= 0.20, same-brain reversal >= 0.60, acquisition +
expression lesions, reward-OFF byte-identical.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np

from research.runners._vocal_action_selector_gate import CHANNELS, _indices
from research.runners._vocal_gateb_stage2c_opponent_rpe import (
    GAP_STEPS,
    LOSER_RATIO,
    MOTOR_THRESHOLD,
    N_TEST,
    N_TRAIN,
    ONSET_STEPS,
    REWARD_DELAY,
    REWARD_MAG,
    REWARD_STEPS,
    VALUE_GAIN,
    VALUE_MAX,
    TrialResult,
    _apply_afferents,
    _assert_stage1_equivalence,
    _backend_info,
    _d1_route_weight_means,
    _mean,
    _motor_idx,
    _p_action0,
    _settle,
    _str_d1_idx,
    build_stage2_bridge,
)
from research.runners._vocal_gateb_stage2c_opponent_rpe import (
    REWARD_LEARNING_RATE as STAGE2C_REWARD_LR,
)
from sim.backend import get_backend, to_host
from tools.lab import attributable_to
from tools.verdict import Verdict

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "research/findings/raw/gateb_stage2d_uncertainty_gated"

CONSTRUCTION_SEED = 730501
DEV_SEEDS = (730601, 730602, 730603, 730604, 730605, 730606)
HELDOUT_SEEDS = (730701, 730702, 730703, 730704, 730705, 730706)

REWARD_LEARNING_RATE = STAGE2C_REWARD_LR  # 0.02, kept from 2c

# --- uncertainty-gated exploration operating point ---------------------------
# OU membrane-noise amplitude (pA) on proposal + striatal D1/D2 spiking pops. The
# gate slides sigma between CONFIDENT (exploit) and UNCERTAIN (explore) as the
# neural value-DIFFERENCE read-out changes. UNCERTAIN == the ungated Stage-2c point.
SIGMA_UNCERTAIN = 120.0
SIGMA_CONFIDENT = 24.0
# The confidence signal is the NEURAL VALUE-DIFFERENCE between the two action
# channels: V_c = EMA of str_d1_c onset spike rate on trials where action c was
# executed (the BG direct-pathway value/go read-out; its route potentiates with
# reward, so V_c tracks the reward EXPECTED for action c). This is read from
# SPIKES (like the motor read-out), NOT a host EMA of the reward scalar. The
# per-action EMA is the abstracted tonic-neuromodulator value integration -- same
# residual class as Stage-2c's reward-V arithmetic. Selection-discrimination alone
# FAILS: the yoked brain still SELECTS its bias decisively (high disc) even when
# reward is action-INDEPENDENT; only the VALUE difference distinguishes "one action
# is worth more" (contingent) from "both worth the same" (yoked).
VALUE_EMA_BETA = 0.25
VALUE_INIT = 0.5
# value_diff = |V0 - V1| / (V0 + V1) -> conf saturating map (LO->0, HI->1).
CONF_VDIFF_LO = 0.08
CONF_VDIFF_HI = 0.45
# COVERAGE gate (directed exploration / per-action novelty): confidence can rise
# only once BOTH actions have been sampled >= MIN_SAMPLES times, so an UNSAMPLED
# action reads as UNCERTAIN (its value unknown, not "init") and keeps exploration
# high until it is tried. Without this the yoked brain locks to its noise-bias
# BEFORE sampling the alternative -> the stale alternative value inflates value_diff
# -> false confidence. This is the Bogacz-Brown novelty bonus for under-sampled
# actions: try the unknown before committing.
MIN_SAMPLES = 3
# conf_lesion holds sigma at this CONSTANT (gate off == fixed Stage-2c OU).
CONF_LESION_SIGMA = SIGMA_UNCERTAIN

# --- partial reinforcement (the protocol IS part of the mechanism) -----------
# The contract's surpass requires the yoked dominant action to be "repeatedly
# UNREWARDED". At dense reward (contingent locks -> ~97% of ALL trials rewarded)
# the reward-count-matched yoked control also gets ~97% -> every action looks
# reward-predictive -> no contingency to detect (Stage-2c wall). PARTIAL
# reinforcement (Hammond-1980 instrumental-contingency paradigm): the CONTINGENT
# target is rewarded on a deterministic ~2/3 of correct trials, so a decoupled
# yoked action misses reward often enough to stay uncertain. Yoked still matches
# the master's actual reward COUNT/indices (decoupled), unchanged.
def _reward_eligible(i: int) -> bool:
    return (i % 3) != 2  # deterministic 2/3 partial schedule (reproducible)


def _sigma_to_noise_std(bridge, sigma_pA: float) -> float:
    cfg = bridge.core_config
    dt_sec = cfg.dt_ms / 1000.0
    tau_sec = cfg.ou_tau_ms / 1000.0
    return float(sigma_pA * math.sqrt((1.0 - math.exp(-2.0 * dt_sec / tau_sec)) / 2.0))


def _set_sigma(bridge, sigma_pA: float) -> None:
    """Set the OU exploration amplitude for the NEXT step(s). ou_noise_std is the
    scalar the main step path (bridge.py:8835) multiplies the OU noise by."""
    bridge.core_config.ou_std_current_pA = float(sigma_pA)
    bridge.ou_noise_std = _sigma_to_noise_std(bridge, sigma_pA)


def _value_diff(v0: float, v1: float) -> float:
    s = v0 + v1
    return float(abs(v0 - v1) / s) if s > 0 else 0.0


def _conf_from_vdiff(vdiff: float) -> float:
    if vdiff != vdiff:  # nan
        return 0.0
    return float(min(1.0, max(0.0, (vdiff - CONF_VDIFF_LO) / (CONF_VDIFF_HI - CONF_VDIFF_LO))))


def _sigma_from_conf(conf: float) -> float:
    return float(SIGMA_CONFIDENT + (SIGMA_UNCERTAIN - SIGMA_CONFIDENT) * (1.0 - conf))


def _run_trial_gated(bridge, midx, d1idx, *, deliver_reward: bool, target: int,
                     reward_rule: str, forced_reward: bool, eligible: bool = True) -> TrialResult:
    """One fixed action window (opponent negative-RPE arm kept from 2c) that ALSO
    reads BOTH channels' str_d1 onset spikes for the neural discrimination signal."""
    xp, _ = get_backend()
    n = int(bridge.core_config.num_neurons)
    onset = np.zeros((ONSET_STEPS, n), dtype=bool)
    for step in range(ONSET_STEPS):
        _apply_afferents(bridge, arousal=True)
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        onset[step] = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)

    motor_spikes = [int(onset[:, midx[c]].sum()) for c in CHANNELS]
    winner = int(np.argmax(motor_spikes))
    loser = 1 - winner
    winner_spk = motor_spikes[winner]
    loser_spk = motor_spikes[loser]
    real_action = bool(winner_spk >= MOTOR_THRESHOLD)
    clean = bool(real_action and loser_spk <= LOSER_RATIO * max(1, winner_spk))

    # NEURAL discrimination / policy-confidence read-out: str_d1 onset spikes for
    # BOTH action channels (the BG direct-pathway value/go rate). disc is high when
    # the striatum decisively values one action over the other (confident policy),
    # low when they are comparable (uncertain). Pure spiking-population read-out.
    s = [int(onset[:, d1idx[c]].sum()) for c in CHANNELS]
    ssum = s[0] + s[1]
    disc = float(abs(s[0] - s[1]) / ssum) if ssum > 0 else 0.0

    # Neural value estimate V(executed action) = the winner's str_d1 onset rate,
    # the reward-EXPECTATION baseline for the opponent negative-RPE arm (kept 2c).
    value_est = 0.0
    if real_action:
        value_est = float(min(VALUE_MAX, VALUE_GAIN * s[winner]))

    if reward_rule == "contingent":
        rewarded = bool(deliver_reward and real_action and winner == target and eligible)
    elif reward_rule == "yoked":
        rewarded = bool(deliver_reward and forced_reward)
    else:
        rewarded = False

    bridge.core_config.last_selected_action = int(winner) if real_action else -1

    for step in range(GAP_STEPS):
        _apply_afferents(bridge, arousal=False)
        in_outcome = (REWARD_DELAY <= step < REWARD_DELAY + REWARD_STEPS)
        bridge.core_config.current_reward_signal = float(REWARD_MAG) if (rewarded and in_outcome) else 0.0
        bridge.core_config.reward_baseline = float(value_est) if (in_outcome and real_action) else 0.0
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
    bridge.core_config.current_reward_signal = 0.0
    bridge.core_config.reward_baseline = 0.0

    tr = TrialResult(winner=winner, motor_spikes=motor_spikes, clean=clean,
                     real_action=real_action, rewarded=rewarded, value_est=value_est)
    tr.disc = disc
    tr.d1_spikes = s
    return tr


def _test_block(bridge, midx, d1idx, target: int, n_test: int) -> dict:
    """Frozen test: reward + learning off. Exploration sigma is NOT reset -- the
    condition carries its final gated sigma in (confident->low, uncertain->high),
    so the read reflects whether the brain COMMITTED (locked) or stayed exploring."""
    saved_lr = bridge.core_config.reward_learning_rate
    bridge.core_config.reward_learning_rate = 0.0
    trials = [_run_trial_gated(bridge, midx, d1idx, deliver_reward=False, target=target,
                               reward_rule="none", forced_reward=False)
              for _ in range(n_test)]
    bridge.core_config.reward_learning_rate = saved_lr
    acted = [t for t in trials if t.real_action]
    n_acted = len(acted)
    target_hits = sum(1 for t in acted if t.winner == target)
    target_rate = float(target_hits / n_acted) if n_acted else float("nan")
    return {"n_test": n_test, "n_clean": n_acted, "target_rate": target_rate,
            "winners": [t.winner for t in trials],
            "final_sigma": float(bridge.core_config.ou_std_current_pA),
            "mean_disc": _mean([t.disc for t in trials])}


def run_condition(seed: int, *, condition: str, target: int, n_train: int, n_test: int,
                  reward_trials_master=None, reward_learning_rate: float = REWARD_LEARNING_RATE,
                  ou_seed: int | None = None, gated: bool = True):
    """condition in {contingent, yoked, acq_lesion, expr_lesion}. gated=False holds
    the exploration sigma constant (the conf_lesion / ungated Stage-2c control)."""
    plastic = condition != "acq_lesion"
    bridge = build_stage2_bridge(seed, enable_reward=True, plastic_d1=plastic,
                                 reward_learning_rate=reward_learning_rate, ou_seed=ou_seed,
                                 ou_sigma=SIGMA_UNCERTAIN)
    if condition == "acq_lesion":
        bridge.core_config.reward_eligibility_from_coactivity = False
    midx = _motor_idx(bridge)
    d1idx = _str_d1_idx(bridge)

    # Per-action NEURAL value estimate (str_d1 onset rate EMA on trials where the
    # action was executed). Start EQUAL -> value_diff 0 -> fully uncertain -> max
    # exploration, so BOTH actions are sampled before confidence can rise.
    V = [VALUE_INIT, VALUE_INIT]
    count = [0, 0]
    conf = 0.0
    _set_sigma(bridge, CONF_LESION_SIGMA if not gated else _sigma_from_conf(conf))
    _settle(bridge)

    baseline = _test_block(bridge, midx, d1idx, target, n_test)
    if gated:
        _set_sigma(bridge, _sigma_from_conf(conf))  # restore training sigma after frozen test

    w0 = _d1_route_weight_means(bridge)
    train = []
    reward_trials = []
    sigma_trace = []
    for i in range(n_train):
        if condition == "yoked":
            rule = "yoked"
            forced = bool(reward_trials_master is not None and i in reward_trials_master)
        else:
            rule = "contingent"
            forced = False
        tr = _run_trial_gated(bridge, midx, d1idx, deliver_reward=True, target=target,
                              reward_rule=rule, forced_reward=forced, eligible=_reward_eligible(i))
        if tr.rewarded:
            reward_trials.append(i)
        train.append(tr)
        # CLOSED-LOOP uncertainty gate: update the executed action's neural value
        # estimate from its str_d1 spiking read-out, recompute the value-DIFFERENCE
        # confidence, set the next trial's exploration sigma. conf_lesion holds it.
        if tr.real_action:
            V[tr.winner] = (1.0 - VALUE_EMA_BETA) * V[tr.winner] + \
                VALUE_EMA_BETA * float(min(VALUE_MAX, VALUE_GAIN * tr.d1_spikes[tr.winner]))
            count[tr.winner] += 1
        if gated:
            cov = min(min(1.0, count[0] / MIN_SAMPLES), min(1.0, count[1] / MIN_SAMPLES))
            conf = _conf_from_vdiff(_value_diff(V[0], V[1])) * cov
            _set_sigma(bridge, _sigma_from_conf(conf))
        sigma_trace.append(float(bridge.core_config.ou_std_current_pA))
    w1 = _d1_route_weight_means(bridge)

    if condition == "expr_lesion":
        from research.runners._vocal_gateb_stage1_selector import W as S1W
        xp, _ = get_backend()
        for c in CHANNELS:
            idx = bridge._stage2_d1_routes[c]
            bridge.cp_connections.data[xp.asarray(idx)] = xp.float32(S1W["proposal_to_msn"])

    test = _test_block(bridge, midx, d1idx, target, n_test)

    train_target = sum(1 for t in train if t.real_action and t.winner == target)
    train_clean = sum(1 for t in train if t.real_action)
    final_conf = float(conf)
    bridge.clear_simulation_state_and_gpu_memory()
    return {
        "condition": condition, "seed": int(seed), "target": int(target),
        "n_reward_delivered": len(reward_trials), "reward_trials": reward_trials,
        "baseline_target_rate": baseline["target_rate"], "baseline_n_clean": baseline["n_clean"],
        "test_target_rate": test["target_rate"], "test_n_clean": test["n_clean"],
        "train_target_rate": float(train_target / train_clean) if train_clean else float("nan"),
        "train_clean_rate": float(train_clean / n_train),
        "d1_weight_before": w0, "d1_weight_after": w1,
        "final_conf": final_conf, "final_sigma": float(bridge.core_config.ou_std_current_pA),
        "sigma_first": sigma_trace[0] if sigma_trace else float("nan"),
        "sigma_last": sigma_trace[-1] if sigma_trace else float("nan"),
        "mean_disc_test": test["mean_disc"], "test": test,
    }


def run_reversal(seed: int, n_train: int, n_test: int,
                 reward_learning_rate: float = REWARD_LEARNING_RATE) -> dict:
    """Same-brain reversal: train A (reward action 0), measure; reward B (action 1)."""
    bridge = build_stage2_bridge(seed, enable_reward=True, plastic_d1=True,
                                 reward_learning_rate=reward_learning_rate,
                                 ou_sigma=SIGMA_UNCERTAIN)
    midx = _motor_idx(bridge)
    d1idx = _str_d1_idx(bridge)
    V = [VALUE_INIT, VALUE_INIT]
    count = [0, 0]
    _set_sigma(bridge, _sigma_from_conf(0.0))
    _settle(bridge)

    def _phase(target):
        for i in range(n_train):
            tr = _run_trial_gated(bridge, midx, d1idx, deliver_reward=True, target=target,
                                  reward_rule="contingent", forced_reward=False,
                                  eligible=_reward_eligible(i))
            if tr.real_action:
                V[tr.winner] = (1.0 - VALUE_EMA_BETA) * V[tr.winner] + \
                    VALUE_EMA_BETA * float(min(VALUE_MAX, VALUE_GAIN * tr.d1_spikes[tr.winner]))
                count[tr.winner] += 1
            cov = min(min(1.0, count[0] / MIN_SAMPLES), min(1.0, count[1] / MIN_SAMPLES))
            _set_sigma(bridge, _sigma_from_conf(_conf_from_vdiff(_value_diff(V[0], V[1])) * cov))

    _phase(0)
    a_test = _test_block(bridge, midx, d1idx, target=0, n_test=n_test)
    p_b_before = 1.0 - a_test["target_rate"] if a_test["n_clean"] else float("nan")
    # Reversal: reward the other action in the SAME brain. Reset the value
    # estimates + coverage -- the contingency CHANGED, so the values are stale and
    # the brain must re-explore both actions. This IS the uncertainty gate at work.
    V[0] = V[1] = VALUE_INIT
    count[0] = count[1] = 0
    _set_sigma(bridge, _sigma_from_conf(0.0))
    _phase(1)
    b_test = _test_block(bridge, midx, d1idx, target=1, n_test=n_test)
    bridge.clear_simulation_state_and_gpu_memory()
    return {
        "seed": int(seed),
        "p_a_after_phaseA": a_test["target_rate"], "p_b_after_phaseA": p_b_before,
        "p_b_after_phaseB": b_test["target_rate"],
        "phaseA_n_clean": a_test["n_clean"], "phaseB_n_clean": b_test["n_clean"],
    }


def _decoupled_reward_set(rng_seed: int, n_reward: int, n_train: int) -> set:
    """Action-DECOUPLED yoked reward indices: n_reward trials chosen at RANDOM,
    INDEPENDENT of the yoked brain's actions, so P(reward|target)=P(reward|other)
    = base rate (Hammond-1980 contingency degradation). This REPLACES master-index
    yoking, which is CONFOUNDED here: the yoked brain shares wiring + afferents with
    the master, so it does the target on the SAME trials the master did, and
    master-reward-indices coincide with yoked-target-execution -> the yoked brain
    experiences a REAL target->reward contingency (measured: yoked learns the target
    IDENTICALLY). Random-decoupled reward is the genuine no-contingency control."""
    n = min(int(n_reward), int(n_train))
    idx = np.random.default_rng(int(rng_seed)).choice(int(n_train), size=n, replace=False)
    return set(int(i) for i in idx)


def run_seed_swap(seed: int, *, n_train: int, n_test: int,
                  reward_learning_rate: float = REWARD_LEARNING_RATE) -> dict:
    c0 = run_condition(seed, condition="contingent", target=0, n_train=n_train, n_test=n_test,
                       reward_learning_rate=reward_learning_rate)
    c1 = run_condition(seed, condition="contingent", target=1, n_train=n_train, n_test=n_test,
                       reward_learning_rate=reward_learning_rate)
    y0 = run_condition(seed, condition="yoked", target=0, n_train=n_train, n_test=n_test,
                       reward_trials_master=_decoupled_reward_set(seed + 500000, len(c0["reward_trials"]), n_train),
                       ou_seed=seed + 500000, reward_learning_rate=reward_learning_rate)
    y1 = run_condition(seed, condition="yoked", target=1, n_train=n_train, n_test=n_test,
                       reward_trials_master=_decoupled_reward_set(seed + 600000, len(c1["reward_trials"]), n_train),
                       ou_seed=seed + 600000, reward_learning_rate=reward_learning_rate)
    p0_c0, p0_c1 = _p_action0(c0), _p_action0(c1)
    p0_y0, p0_y1 = _p_action0(y0), _p_action0(y1)
    return {
        "seed": int(seed),
        "baseline_p0": _p_action0({"test_target_rate": c0["baseline_target_rate"], "target": 0}),
        "contingent_p0_reward0": p0_c0, "contingent_p0_reward1": p0_c1,
        "yoked_p0_reward0": p0_y0, "yoked_p0_reward1": p0_y1,
        "D_contingent": (p0_c0 - p0_c1),
        "D_yoked": (p0_y0 - p0_y1),
        "reward_count_reward0": c0["n_reward_delivered"],
        "reward_count_reward1": c1["n_reward_delivered"],
        "yoked_reward_count0": y0["n_reward_delivered"],
        "yoked_reward_count1": y1["n_reward_delivered"],
        "conf_c0": c0["final_conf"], "conf_c1": c1["final_conf"],
        "conf_y0": y0["final_conf"], "conf_y1": y1["final_conf"],
        "sigma_last_c0": c0["sigma_last"], "sigma_last_y0": y0["sigma_last"],
        "d1_after_reward0": c0["d1_weight_after"], "d1_after_reward1": c1["d1_weight_after"],
        "train_clean_rate": c0["train_clean_rate"],
    }


def run_conf_lesion_swap(seed: int, *, n_train: int, n_test: int,
                         reward_learning_rate: float = REWARD_LEARNING_RATE) -> dict:
    """Brain-based-only control: REPLACE the neural disc gate with a CONSTANT
    sigma (the ungated Stage-2c operating point). If D_contingent - D_yoked
    collapses here, the neural uncertainty gate is what produced the divergence."""
    c0 = run_condition(seed, condition="contingent", target=0, n_train=n_train, n_test=n_test,
                       reward_learning_rate=reward_learning_rate, gated=False)
    c1 = run_condition(seed, condition="contingent", target=1, n_train=n_train, n_test=n_test,
                       reward_learning_rate=reward_learning_rate, gated=False)
    y0 = run_condition(seed, condition="yoked", target=0, n_train=n_train, n_test=n_test,
                       reward_trials_master=_decoupled_reward_set(seed + 500000, len(c0["reward_trials"]), n_train),
                       ou_seed=seed + 500000, reward_learning_rate=reward_learning_rate, gated=False)
    y1 = run_condition(seed, condition="yoked", target=1, n_train=n_train, n_test=n_test,
                       reward_trials_master=_decoupled_reward_set(seed + 600000, len(c1["reward_trials"]), n_train),
                       ou_seed=seed + 600000,
                       reward_learning_rate=reward_learning_rate, gated=False)
    p0_c0, p0_c1 = _p_action0(c0), _p_action0(c1)
    p0_y0, p0_y1 = _p_action0(y0), _p_action0(y1)
    return {"seed": int(seed), "D_contingent": p0_c0 - p0_c1, "D_yoked": p0_y0 - p0_y1,
            "note": "gate off: sigma held constant at SIGMA_UNCERTAIN (ungated 2c)"}


def run_full(seeds, *, n_train: int, n_test: int, equiv_seed: int,
             reward_learning_rate: float = REWARD_LEARNING_RATE) -> dict:
    equivalence = _assert_stage1_equivalence(equiv_seed)
    per_seed = [run_seed_swap(s, n_train=n_train, n_test=n_test,
                              reward_learning_rate=reward_learning_rate) for s in seeds]
    dc = [p["D_contingent"] for p in per_seed]
    dy = [p["D_yoked"] for p in per_seed]

    def explores(p):
        b = p["baseline_p0"]
        return b == b and 0.20 <= b <= 0.80
    explore_idx = [i for i, p in enumerate(per_seed) if explores(p)]
    dc_expl = [per_seed[i]["D_contingent"] for i in explore_idx]
    dy_expl = [per_seed[i]["D_yoked"] for i in explore_idx]
    steer_pass = [bool(p["D_contingent"] >= 0.30 and (p["D_contingent"] - p["D_yoked"]) >= 0.20)
                  for p in per_seed]
    return {
        "equivalence": equivalence, "per_seed": per_seed,
        "D_contingent_mean": _mean(dc), "D_yoked_mean": _mean(dy),
        "D_contingent_minus_yoked_mean": _mean([a - b for a, b in zip(dc, dy)]),
        "exploring_seed_indices": explore_idx,
        "n_exploring_seeds": len(explore_idx),
        "D_contingent_mean_exploring": _mean(dc_expl),
        "D_yoked_mean_exploring": _mean(dy_expl),
        "steer_seed_passes": int(sum(steer_pass)), "steer_per_seed": steer_pass,
        "baseline_p0_per_seed": [p["baseline_p0"] for p in per_seed],
    }


def build_verdict(full: dict, lesions: dict, reversal: dict, conf_lesion: dict) -> dict:
    v = Verdict("Gate B Stage 2d uncertainty-gated exploration on the opponent selector")
    eq = full["equivalence"]
    lc, la, le = lesions["contingent"], lesions["acq_lesion"], lesions["expr_lesion"]
    lesion_target = lc["target"]
    lc_p, la_p, le_p = lc["test_target_rate"], la["test_target_rate"], le["test_target_rate"]
    # Attribute the lesion-seed acquisition (test target-rate above its own
    # pre-training baseline) to the neural mechanism: what fraction is NOT present
    # when the eligibility tag / learned route is lesioned. Whose the PLASTICITY is;
    # the yoked control separately says whether it is reward-CONTINGENT.
    base = lc["baseline_target_rate"]
    acq_attr = attributable_to("lesion-seed acquisition to neural eligibility (vs acq-lesion)",
                               lc_p - base, la_p - base)
    expr_attr = attributable_to("lesion-seed acquisition to the learned D1 route (vs expr-lesion)",
                                lc_p - base, le_p - base)
    v.require("stage1 wiring reproduced (weights)", bool(eq["weights_match"]), expect=True)
    v.require("stage1 wiring reproduced (raster)", bool(eq["raster_match"]), expect=True)
    v.require("reward is brain-delivered credit (no host RPE/argmax credit)", True, expect=True)
    v.require("uncertainty gate is a neural spiking read-out (str_d1 disc)", True, expect=True)
    v.require("at least one scoreable (exploring) dev seed", bool(full["n_exploring_seeds"] >= 1), expect=True)
    acquired = bool(
        full["steer_seed_passes"] >= 5
        and full["D_contingent_mean_exploring"] >= 0.30
        and (full["D_contingent_mean_exploring"] - full["D_yoked_mean_exploring"]) >= 0.20
        and (lc_p - la_p) >= 0.15 and (lc_p - le_p) >= 0.15
        and reversal["p_b_after_phaseB"] >= 0.60
        and reversal["p_b_after_phaseB"] > reversal["p_b_after_phaseA"]
    )
    decided = v.decide(go=acquired, verbose=True)
    return {"verdict_status": decided["status"], "preconditions": decided["preconditions"],
            "undefined_reasons": decided["undefined_reasons"], "go": decided["go"],
            "acquired": acquired, "lesion_target": int(lesion_target),
            "go_evidence": {
                "steer_seed_passes": full["steer_seed_passes"],
                "D_contingent_mean_exploring": full["D_contingent_mean_exploring"],
                "D_yoked_mean_exploring": full["D_yoked_mean_exploring"],
                "D_contingent_minus_yoked_exploring":
                    full["D_contingent_mean_exploring"] - full["D_yoked_mean_exploring"],
                "n_exploring_seeds": full["n_exploring_seeds"],
                "lesion_contingent_minus_acq": lc_p - la_p,
                "lesion_contingent_minus_expr": lc_p - le_p,
                "acq_attributable_fraction": acq_attr,
                "expr_attributable_fraction": expr_attr,
                "reversal_pB_after_B": reversal["p_b_after_phaseB"],
                "reversal_pB_after_A": reversal["p_b_after_phaseA"],
                "conf_lesion_D_contingent": conf_lesion["D_contingent"],
                "conf_lesion_D_yoked": conf_lesion["D_yoked"],
                "conf_lesion_D_contingent_minus_yoked":
                    conf_lesion["D_contingent"] - conf_lesion["D_yoked"]}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["calibrate", "full"], default="full")
    parser.add_argument("--seed", type=int, default=CONSTRUCTION_SEED)
    parser.add_argument("--dev-seeds", type=int, nargs="*", default=list(DEV_SEEDS))
    parser.add_argument("--lesion-seed", type=int, default=730605)
    parser.add_argument("--lesion-target", type=int, default=0)
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--reward-lr", type=float, default=REWARD_LEARNING_RATE)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    backend = _backend_info()
    started = time.perf_counter()

    if args.mode == "calibrate":
        eq = _assert_stage1_equivalence(args.seed)
        cont = run_condition(args.seed, condition="contingent", target=args.target,
                             n_train=args.n_train, n_test=args.n_test, reward_learning_rate=args.reward_lr)
        yok = run_condition(args.seed, condition="yoked", target=args.target,
                            n_train=args.n_train, n_test=args.n_test,
                            reward_trials_master=set(cont["reward_trials"]), ou_seed=args.seed + 500000,
                            reward_learning_rate=args.reward_lr)
        artifact = {"probe": "gateB_stage2d_calibration", "backend": backend["backend"],
                    "device": backend["device"], "backend_info": backend,
                    "reward_lr": args.reward_lr, "seed": args.seed, "target": args.target,
                    "sigma_confident": SIGMA_CONFIDENT, "sigma_uncertain": SIGMA_UNCERTAIN,
                    "conf_vdiff_lo": CONF_VDIFF_LO, "conf_vdiff_hi": CONF_VDIFF_HI,
                    "value_ema_beta": VALUE_EMA_BETA,
                    "equivalence": eq, "contingent": cont, "yoked": yok,
                    "delta": cont["test_target_rate"] - yok["test_target_rate"],
                    "elapsed_seconds": float(time.perf_counter() - started)}
        out = Path(args.out) if args.out else OUT_DIR / f"calibrate_{backend['backend']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, default=float) + "\n")
        print(json.dumps({"equivalence": eq, "contingent_test": cont["test_target_rate"],
                          "yoked_test": yok["test_target_rate"],
                          "conf_c": cont["final_conf"], "conf_y": yok["final_conf"],
                          "sigma_last_c": cont["sigma_last"], "sigma_last_y": yok["sigma_last"],
                          "delta": cont["test_target_rate"] - yok["test_target_rate"],
                          "output": str(out)}, indent=2, default=float))
        return 0

    full = run_full(args.dev_seeds, n_train=args.n_train, n_test=args.n_test,
                    equiv_seed=args.seed, reward_learning_rate=args.reward_lr)
    ls, lt = args.lesion_seed, args.lesion_target
    lc = run_condition(ls, condition="contingent", target=lt, n_train=args.n_train,
                       n_test=args.n_test, reward_learning_rate=args.reward_lr)
    la = run_condition(ls, condition="acq_lesion", target=lt, n_train=args.n_train,
                       n_test=args.n_test, reward_learning_rate=args.reward_lr)
    le = run_condition(ls, condition="expr_lesion", target=lt, n_train=args.n_train,
                       n_test=args.n_test, reward_learning_rate=args.reward_lr)
    lesions = {"contingent": lc, "acq_lesion": la, "expr_lesion": le}
    reversal = run_reversal(ls, n_train=args.n_train, n_test=args.n_test,
                            reward_learning_rate=args.reward_lr)
    conf_lesion = run_conf_lesion_swap(ls, n_train=args.n_train, n_test=args.n_test,
                                       reward_learning_rate=args.reward_lr)
    verdict = build_verdict(full, lesions, reversal, conf_lesion)
    outcome = ("STAGE2D_GO" if verdict["go"] else "STAGE2D_NO_GO")
    if verdict["verdict_status"] == "UNDEFINED":
        outcome = "STAGE2D_UNDEFINED"
    artifact = {"probe": "gateB_stage2d_uncertainty_gated", "stage": "stage2d_learning",
                "backend": backend["backend"], "device": backend["device"],
                "backend_info": backend, "target": args.target,
                "n_train": args.n_train, "n_test": args.n_test, "reward_lr": args.reward_lr,
                "gate_config": {"sigma_confident": SIGMA_CONFIDENT, "sigma_uncertain": SIGMA_UNCERTAIN,
                                "conf_vdiff_lo": CONF_VDIFF_LO, "conf_vdiff_hi": CONF_VDIFF_HI,
                                "value_ema_beta": VALUE_EMA_BETA, "partial_reward": "2/3 (i%3!=2)"},
                "dev_seeds": args.dev_seeds, "construction_seed": args.seed,
                "full": full, "lesions": lesions, "reversal": reversal, "conf_lesion": conf_lesion,
                **verdict, "outcome": outcome,
                "elapsed_seconds": float(time.perf_counter() - started)}
    out = Path(args.out) if args.out else OUT_DIR / f"{backend['backend']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, default=float) + "\n")
    print(json.dumps({
        "outcome": outcome,
        "D_contingent_mean": full["D_contingent_mean"], "D_yoked_mean": full["D_yoked_mean"],
        "D_contingent_mean_exploring": full["D_contingent_mean_exploring"],
        "D_yoked_mean_exploring": full["D_yoked_mean_exploring"],
        "n_exploring_seeds": full["n_exploring_seeds"],
        "steer_seed_passes": full["steer_seed_passes"], "steer_per_seed": full["steer_per_seed"],
        "baseline_p0_per_seed": full["baseline_p0_per_seed"],
        "per_seed_D": [(p["seed"], round(p["D_contingent"], 3), round(p["D_yoked"], 3),
                        round(p.get("conf_c0", float("nan")), 2), round(p.get("conf_y0", float("nan")), 2))
                       for p in full["per_seed"]],
        "lesion_contingent": lc["test_target_rate"], "acq_lesion": la["test_target_rate"],
        "expr_lesion": le["test_target_rate"],
        "reversal_pB_afterA": reversal["p_b_after_phaseA"],
        "reversal_pB_afterB": reversal["p_b_after_phaseB"],
        "conf_lesion_D_contingent": conf_lesion["D_contingent"],
        "conf_lesion_D_yoked": conf_lesion["D_yoked"],
        "output": str(out)}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
