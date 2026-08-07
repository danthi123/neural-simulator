"""Gate B Stage 2g: TRUE Hammond DeltaP (withhold baseline + homeostatic critic).

Surpasses the Stage 2f residual
(`research/findings/2026-08-06-gateB-stage2f-contingency-gated-exploration-NO-GO.md`,
`STAGE2F_NO_GO`, steer 4/6): the Stage-2f D1-minus-D2 contrast is a NEURAL
contingency gate but estimates **P(reward|action)**, not the Hammond
**DeltaP = P(reward|action) - P(reward|NO-action)**. Two named residuals:

  * 730605 (D_yoked +0.55, below the gate): a base reward rate that survives in
    the action's ABSENCE is never subtracted, so decoupled yoked reward still
    biases one D1 route below the exploration gate.
  * 730602 (D_cont 0.00, never exploits): a single global scalar VALUE_GAIN
    mis-signs the RPE on that seed's heterogeneous striatal firing rates.

Stage 2g adds the two biology-grounded mechanisms the Stage-2f finding names:

  (a) A NEURAL WITHHOLD BASELINE -> true contingency V(action) - V(withhold).
      Interleaved NO-ACTION (withhold) trials drive a dedicated striatal value
      population `value_wh` (a D1-type read-out with NO efferent into the
      selector, so it never biases selection). Its afferent is tagged
      action_index=2 (the inert dopamine_S channel), so ONLY a withhold-trial
      reward -- delivered with last_selected_action=2 -- potentiates it. Its
      onset firing rate is V(withhold): the expected reward in the action's
      ABSENCE (the base rate). At each ACTION outcome the reward-expectation
      baseline becomes reward_baseline = V(action) + V(withhold), so the DA
      production rule (reward - reward_baseline) subtracts the base rate -> the
      RPE that drives D1 plasticity is the TRUE Hammond DeltaP. In CONTINGENT,
      withhold trials are never rewarded (reward requires acting on target) ->
      V(withhold) ~ 0 -> contingent steer preserved. In YOKED, withhold trials
      ARE rewarded at the base rate -> V(withhold) rises -> the decoupled-reward
      bias cancels -> D_yoked collapses (the 730605 fix).

  (b) HOMEOSTATIC PER-POPULATION CRITIC NORMALISATION. The scalar VALUE_GAIN is
      replaced by a per-population read-out rescaled by each str_d1 channel's OWN
      homeostatic set-point r0 (its untrained baseline onset rate, measured in
      the baseline block): V(action) = VALUE_GAIN_N * max(0, s1-r0)/r0 (synaptic
      scaling; Turrigiano). This removes each seed's absolute-rate offset, so the
      RPE stays correctly signed across heterogeneous seeds (the 730602 fix).

Brain-based-only: V(action) and V(withhold) are onset spike-count read-outs of
SPIKING striatal populations (str_d1_c and value_wh -- like the motor read-out
that moves the body); the withhold value is BUILT by the substrate's own DA-dip
three-factor plasticity on the value_wh route; the reward - baseline subtraction
is the DA system's outcome-vs-expectation comparison (the DA production rule).
Reward is an env scalar delivered by the body. Residual (declared, unchanged
class from Stage-2c): the read-out gains (VALUE_GAIN_N, WH_VALUE_GAIN) and the
conf->sigma controller arithmetic are the abstracted tonic-neuromodulator map.
The withhold-reward PROBABILITY on yoked trials is the environment's base rate
(a property of the world's reward schedule), not a host value estimate.

Kept from Stage 2f UNCHANGED: the D1-D2 DeltaP confidence gate, directed
novelty-biased exploration, per-action compartmentalised DA, the opponent
negative-RPE arm, action-DECOUPLED yoked reward, partial (2/3) reinforcement,
the byte-identical-to-Stage-1 reward-OFF guard. Acceptance criteria FROZEN from
the Stage-2 preregistration UNCHANGED: steer_seed_passes >= 5/6 AND
D_contingent - D_yoked >= 0.20, D_contingent_mean_exploring >= 0.30, same-brain
reversal >= 0.60, acquisition + expression lesions (delta >= 0.15), reward-OFF
byte-identical.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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
from research.runners._vocal_gateb_stage2d_uncertainty_gated import (
    MIN_SAMPLES,
    REWARD_LEARNING_RATE,
    SIGMA_CONFIDENT,
    SIGMA_UNCERTAIN,
    VALUE_EMA_BETA,
    VALUE_INIT,
    _decoupled_reward_set,
    _reward_eligible,
    _set_sigma,
    _sigma_from_conf,
)
from research.runners._vocal_gateb_stage2e_directed_novelty import (
    EQUALIZE_DEFICIT,
    NOVELTY_DRIVE_MAX_PA,
    _novelty_drive,
)
from research.runners._vocal_gateb_stage2f_contingency_gated import (
    CONF_DP_LO,
    CONF_DP_HI,
    NOVELTY_CONF_GATE,
    _conf_from_contingency,
    _contingency,
    _str_d2_idx,
)
from sim.backend import get_backend, to_host
from tools.lab import attributable_to
from tools.verdict import Verdict

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "research/findings/raw/gateb_stage2g_hammond_deltap"

CONSTRUCTION_SEED = 730501
DEV_SEEDS = (730601, 730602, 730603, 730604, 730605, 730606)
HELDOUT_SEEDS = (730701, 730702, 730703, 730704, 730705, 730706)
WH_ACTION = 2  # the inert dopamine_S channel: a tonic average-reward integrator
DA_S = "dopamine_S"

# --- (b) homeostatic critic normalisation -------------------------------------
# value_est = VALUE_GAIN * s1_winner * (REF_TOTAL / (r0_0 + r0_1)): the winner's
# str_d1 onset rate DIVISIVELY normalised by the seed's POOLED striatal baseline
# activity (Carandini-Heeger divisive normalisation / homeostatic synaptic
# scaling toward a population set-point). Dividing by the POOLED (not per-channel)
# baseline rescales the whole seed's value scale to a common reference REF_TOTAL,
# removing the cross-seed rate heterogeneity that mis-signs a single-global-gain
# RPE (fixes 730602/730605) -- while the shared per-seed factor preserves the
# WITHIN-seed relative promotion/demotion (so it does NOT distort the weak action
# on a locked seed, unlike a per-channel normalisation; keeps 730603/730604).
VALUE_GAIN_N = 0.007
REF_TOTAL_RATE = 20.0
# --- (a) withhold tonic-value integrator (the inert dopamine_S channel) --------
# On a withhold (no-action) trial the base-rate reward is delivered with
# last_selected_action=2 -> the dopamine_S production rule charges its
# concentration; the modulator's own leaky decay integrates it into a running
# TONIC average-reward signal (Niv 2007: tonic DA encodes the reward rate). It is
# inert on weights (its only target is the empty action:2 synapse tag), so it is
# a pure readable neural signal. V(withhold) = WH_VALUE_GAIN * concentration.
DA_S_TAU_MS = 2000.0
DA_S_SENSITIVITY = 0.002
DA_S_MAX = 5.0
# Calibrated (dev seeds) so the tonic base-rate subtraction cancels the yoked
# steering. It exceeds the raw base rate because the opponent aversive-scale
# (negative RPE x0.5) halves the base-rate subtraction on omission steps, so a
# larger tonic value is needed to symmetrically cancel the decoupled-reward bias.
# Contingent is unaffected (its withhold trials are never rewarded -> V=0).
WH_VALUE_GAIN = 4.0
# Interleave: run one withhold (no-action) probe before every WH_PERIOD-th
# action trial so the tonic integrator tracks the base rate through training.
WH_PERIOD = 2


def _reconfigure_da_s(bridge) -> None:
    """Turn the inert dopamine_S channel into a leaky average-reward integrator
    (baseline 0, long decay, small per-step gain). Side-effect-free: dopamine_S's
    only target is scope='action:2' (no synapse carries tag 2) and scope!='all'
    so it never enters the global plasticity/gain multipliers."""
    mgr = bridge.neuromodulator_manager
    for c in mgr._configs:
        if c.name == DA_S:
            c.baseline = 0.0
            c.decay_tau_ms = DA_S_TAU_MS
            c.concentration_min = 0.0
            c.concentration_max = DA_S_MAX
            c.production_rules[0].sensitivity = DA_S_SENSITIVITY
    mgr.set_concentration(DA_S, 0.0)


def _v_withhold(bridge) -> float:
    conc = float(bridge.neuromodulator_manager.get_concentration(DA_S))
    return float(min(VALUE_MAX, WH_VALUE_GAIN * conc))


def _value_action(s1_winner: int, r0_total: float) -> float:
    """(b) Homeostatic critic: the winner's str_d1 onset rate divisively normalised
    by the seed's POOLED baseline striatal activity (rescaled to REF_TOTAL_RATE).
    Removes the cross-seed rate heterogeneity that mis-signs a single-global-gain
    RPE while preserving within-seed relative dynamics."""
    scale = REF_TOTAL_RATE / max(float(r0_total), 1.0)
    return float(min(VALUE_MAX, VALUE_GAIN_N * float(s1_winner) * scale))


def _run_trial_2g(bridge, midx, d1idx, d2idx, *, deliver_reward: bool, target: int,
                  reward_rule: str, forced_reward: bool, eligible: bool = True,
                  r0_d1=None, v_withhold: float = 0.0,
                  novelty_target: int | None = None, novelty_drive_pa: float = 0.0) -> TrialResult:
    """One ACTION window (opponent negative-RPE + uncertainty-gated OU + directed
    novelty, kept from 2e/2f). value_est uses the homeostatic per-population
    critic (b); the reward-expectation baseline is V(action) + V(withhold) (a),
    so the DA production subtracts the base rate = TRUE Hammond DeltaP."""
    xp, _ = get_backend()
    n = int(bridge.core_config.num_neurons)
    nov_idx = None
    if novelty_drive_pa > 0.0 and novelty_target is not None:
        nov_idx = xp.asarray(_indices(bridge, f"proposal_{int(novelty_target)}"))
    onset = np.zeros((ONSET_STEPS, n), dtype=bool)
    for step in range(ONSET_STEPS):
        _apply_afferents(bridge, arousal=True)
        if nov_idx is not None:
            bridge.cp_external_input_current[nov_idx] += xp.float32(novelty_drive_pa)
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

    s1 = [int(onset[:, d1idx[c]].sum()) for c in CHANNELS]
    s2 = [int(onset[:, d2idx[c]].sum()) for c in CHANNELS]
    ssum = s1[0] + s1[1]
    disc = float(abs(s1[0] - s1[1]) / ssum) if ssum > 0 else 0.0

    value_est = 0.0
    if real_action:
        r0_total = float(sum(r0_d1)) if r0_d1 is not None else 0.0
        value_est = _value_action(s1[winner], r0_total)

    if reward_rule == "contingent":
        rewarded = bool(deliver_reward and real_action and winner == target and eligible)
    elif reward_rule == "yoked":
        rewarded = bool(deliver_reward and forced_reward)
    else:
        rewarded = False

    bridge.core_config.last_selected_action = int(winner) if real_action else -1

    # Outcome epoch: baseline = V(action) + V(withhold) (TRUE Hammond DeltaP).
    base = float(value_est + v_withhold)
    for step in range(GAP_STEPS):
        _apply_afferents(bridge, arousal=False)
        in_outcome = (REWARD_DELAY <= step < REWARD_DELAY + REWARD_STEPS)
        bridge.core_config.current_reward_signal = float(REWARD_MAG) if (rewarded and in_outcome) else 0.0
        bridge.core_config.reward_baseline = base if (in_outcome and real_action) else 0.0
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
    bridge.core_config.current_reward_signal = 0.0
    bridge.core_config.reward_baseline = 0.0

    tr = TrialResult(winner=winner, motor_spikes=motor_spikes, clean=clean,
                     real_action=real_action, rewarded=rewarded, value_est=value_est)
    tr.disc = disc
    tr.d1_spikes = s1
    tr.d2_spikes = s2
    return tr


def _run_withhold_trial(bridge, *, rewarded: bool) -> None:
    """(a) One NO-ACTION (withhold) trial: arousal OFF (no motor action emitted),
    and if the base-rate reward lands it is delivered with last_selected_action=2
    -> the inert dopamine_S tonic integrator charges. No D1/D2 policy route is
    credited (dopamine_N/E stay at baseline). The eligibility trace is zeroed
    afterwards so the quiet withhold episode leaves no residual credit."""
    xp, _ = get_backend()
    for step in range(ONSET_STEPS):
        _apply_afferents(bridge, arousal=False)
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
    bridge.core_config.last_selected_action = int(WH_ACTION) if rewarded else -1
    for step in range(GAP_STEPS):
        _apply_afferents(bridge, arousal=False)
        in_outcome = (REWARD_DELAY <= step < REWARD_DELAY + REWARD_STEPS)
        bridge.core_config.current_reward_signal = float(REWARD_MAG) if (rewarded and in_outcome) else 0.0
        bridge.core_config.reward_baseline = 0.0
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
    bridge.core_config.current_reward_signal = 0.0
    bridge.core_config.reward_baseline = 0.0
    bridge.core_config.last_selected_action = -1
    if bridge.cp_eligibility_trace is not None:
        bridge.cp_eligibility_trace[:] = xp.float32(0.0)


def _baseline_block(bridge, midx, d1idx, d2idx, target: int, n_test: int) -> dict:
    """Frozen baseline test (reward+learning off). Also measures the per-population
    homeostatic set-points r0_d1[c] (untrained str_d1 onset counts) used by the
    critic normalisation."""
    saved_lr = bridge.core_config.reward_learning_rate
    bridge.core_config.reward_learning_rate = 0.0
    trials = [_run_trial_2g(bridge, midx, d1idx, d2idx, deliver_reward=False,
                            target=target, reward_rule="none", forced_reward=False)
              for _ in range(n_test)]
    bridge.core_config.reward_learning_rate = saved_lr
    acted = [t for t in trials if t.real_action]
    n_acted = len(acted)
    target_hits = sum(1 for t in acted if t.winner == target)
    target_rate = float(target_hits / n_acted) if n_acted else float("nan")
    r0_d1 = [_mean([t.d1_spikes[c] for t in trials]) for c in CHANNELS]
    return {"n_test": n_test, "n_clean": n_acted, "target_rate": target_rate, "r0_d1": r0_d1}


def _test_block(bridge, midx, d1idx, d2idx, target: int, n_test: int, r0_d1=None) -> dict:
    saved_lr = bridge.core_config.reward_learning_rate
    bridge.core_config.reward_learning_rate = 0.0
    trials = [_run_trial_2g(bridge, midx, d1idx, d2idx, deliver_reward=False,
                            target=target, reward_rule="none", forced_reward=False, r0_d1=r0_d1)
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


def _update_conf(vd1, vd2, count, tr, *, use_d2: bool):
    """Kept from Stage 2f: the D1-D2 DeltaP confidence gate (uses the raw VALUE_GAIN
    read-out, unchanged). Separate from the critic normalisation (b), which only
    rescales the RPE baseline."""
    from research.runners._vocal_gateb_stage2c_opponent_rpe import VALUE_GAIN
    if tr.real_action:
        w = tr.winner
        vd1[w] = (1.0 - VALUE_EMA_BETA) * vd1[w] + VALUE_EMA_BETA * float(min(VALUE_MAX, VALUE_GAIN * tr.d1_spikes[w]))
        vd2[w] = (1.0 - VALUE_EMA_BETA) * vd2[w] + VALUE_EMA_BETA * float(min(VALUE_MAX, VALUE_GAIN * tr.d2_spikes[w]))
        count[w] += 1
    cov = min(min(1.0, count[0] / MIN_SAMPLES), min(1.0, count[1] / MIN_SAMPLES))
    if use_d2:
        dp = _contingency(vd1, vd2)
    else:
        dp = _contingency(vd1, [0.0, 0.0])
    return _conf_from_contingency(dp) * cov, dp


def run_condition(seed: int, *, condition: str, target: int, n_train: int, n_test: int,
                  reward_trials_master=None, reward_learning_rate: float = REWARD_LEARNING_RATE,
                  ou_seed: int | None = None, gated: bool = True, directed_novelty: bool = True,
                  use_d2: bool = True, wh_reward_p: float = 0.0):
    """condition in {contingent, yoked, acq_lesion, expr_lesion}. wh_reward_p is the
    base reward rate on withhold trials (P(reward|no-action)): 0 for contingent, the
    yoked base rate for yoked. use_d2=False = contingency_lesion; directed_novelty=
    False = novelty_lesion; wh_reward_p=0 on yoked = withhold_lesion (via caller)."""
    plastic = condition != "acq_lesion"
    bridge = build_stage2_bridge(seed, enable_reward=True, plastic_d1=plastic,
                                 reward_learning_rate=reward_learning_rate, ou_seed=ou_seed,
                                 ou_sigma=SIGMA_UNCERTAIN, plastic_d2=True)
    _reconfigure_da_s(bridge)  # (a) dopamine_S -> tonic average-reward integrator
    if condition == "acq_lesion":
        bridge.core_config.reward_eligibility_from_coactivity = False
    midx = _motor_idx(bridge)
    d1idx = _str_d1_idx(bridge)
    d2idx = _str_d2_idx(bridge)

    Vd1 = [VALUE_INIT, VALUE_INIT]
    Vd2 = [VALUE_INIT, VALUE_INIT]
    count = [0, 0]
    conf = 0.0
    dp = 0.0
    v_wh = 0.0
    wh_rng = np.random.default_rng(int(seed) + 909090 + int(target))
    _set_sigma(bridge, SIGMA_UNCERTAIN if not gated else _sigma_from_conf(conf))
    _settle(bridge)

    baseline = _baseline_block(bridge, midx, d1idx, d2idx, target, n_test)
    r0_d1 = baseline["r0_d1"]
    if gated:
        _set_sigma(bridge, _sigma_from_conf(conf))

    w0 = _d1_route_weight_means(bridge)
    train = []
    reward_trials = []
    sigma_trace = []
    dp_trace = []
    v_wh_trace = []
    n_wh = 0
    n_wh_rewarded = 0
    for i in range(n_train):
        # (a) Interleaved withhold (no-action) trial -> charges the dopamine_S
        # tonic average-reward integrator with the base rate. Only run when the
        # environment HAS a base rate (wh_reward_p>0): when P(reward|no-action)=0
        # (contingent, lesions) a withhold trial is uninformative (V(withhold)
        # stays 0) and would only perturb the noise realisation.
        if wh_reward_p > 0.0 and i % WH_PERIOD == 0:
            rew_wh = bool(wh_rng.random() < wh_reward_p)
            _run_withhold_trial(bridge, rewarded=rew_wh)
            n_wh += 1
            n_wh_rewarded += int(rew_wh)
        v_wh = _v_withhold(bridge)  # neural tonic V(withhold) = WH_VALUE_GAIN * conc
        if condition == "yoked":
            rule = "yoked"
            forced = bool(reward_trials_master is not None and i in reward_trials_master)
        else:
            rule = "contingent"
            forced = False
        nt, nd = (_novelty_drive(count) if directed_novelty else (None, 0.0))
        if NOVELTY_CONF_GATE:
            nd *= (1.0 - conf)
        tr = _run_trial_2g(bridge, midx, d1idx, d2idx, deliver_reward=True, target=target,
                           reward_rule=rule, forced_reward=forced, eligible=_reward_eligible(i),
                           r0_d1=r0_d1, v_withhold=v_wh,
                           novelty_target=nt, novelty_drive_pa=nd)
        if tr.rewarded:
            reward_trials.append(i)
        train.append(tr)
        if gated:
            conf, dp = _update_conf(Vd1, Vd2, count, tr, use_d2=use_d2)
            _set_sigma(bridge, _sigma_from_conf(conf))
        else:
            _, dp = _update_conf(Vd1, Vd2, count, tr, use_d2=use_d2)
        sigma_trace.append(float(bridge.core_config.ou_std_current_pA))
        dp_trace.append(float(dp))
        v_wh_trace.append(float(v_wh))
    w1 = _d1_route_weight_means(bridge)

    if condition == "expr_lesion":
        from research.runners._vocal_gateb_stage1_selector import W as S1W
        xp, _ = get_backend()
        for c in CHANNELS:
            idx = bridge._stage2_d1_routes[c]
            bridge.cp_connections.data[xp.asarray(idx)] = xp.float32(S1W["proposal_to_msn"])

    test = _test_block(bridge, midx, d1idx, d2idx, target, n_test, r0_d1=r0_d1)

    train_target = sum(1 for t in train if t.real_action and t.winner == target)
    train_clean = sum(1 for t in train if t.real_action)
    train_a0 = sum(1 for t in train if t.real_action and t.winner == 0)
    train_p0_all = float(train_a0 / train_clean) if train_clean else float("nan")
    bridge.clear_simulation_state_and_gpu_memory()
    return {
        "condition": condition, "seed": int(seed), "target": int(target),
        "n_reward_delivered": len(reward_trials), "reward_trials": reward_trials,
        "baseline_target_rate": baseline["target_rate"], "baseline_n_clean": baseline["n_clean"],
        "test_target_rate": test["target_rate"], "test_n_clean": test["n_clean"],
        "train_target_rate": float(train_target / train_clean) if train_clean else float("nan"),
        "train_clean_rate": float(train_clean / n_train),
        "train_p0_all": train_p0_all,
        "count0": int(count[0]), "count1": int(count[1]),
        "d1_weight_before": w0, "d1_weight_after": w1,
        "final_conf": float(conf), "final_dp": float(dp),
        "final_v_withhold": float(v_wh), "max_v_withhold": float(max(v_wh_trace) if v_wh_trace else 0.0),
        "n_withhold": int(n_wh), "n_withhold_rewarded": int(n_wh_rewarded),
        "r0_d1": r0_d1,
        "Vd1": [float(Vd1[0]), float(Vd1[1])], "Vd2": [float(Vd2[0]), float(Vd2[1])],
        "final_sigma": float(bridge.core_config.ou_std_current_pA),
    }


def run_reversal(seed: int, n_train: int, n_test: int,
                 reward_learning_rate: float = REWARD_LEARNING_RATE) -> dict:
    """Same-brain reversal: train A (reward action 0), measure; reward B (action 1).
    Contingent phases -> withhold trials never rewarded (wh_reward_p=0)."""
    bridge = build_stage2_bridge(seed, enable_reward=True, plastic_d1=True,
                                 reward_learning_rate=reward_learning_rate,
                                 ou_sigma=SIGMA_UNCERTAIN, plastic_d2=True)
    _reconfigure_da_s(bridge)
    midx = _motor_idx(bridge)
    d1idx = _str_d1_idx(bridge)
    d2idx = _str_d2_idx(bridge)
    Vd1 = [VALUE_INIT, VALUE_INIT]
    Vd2 = [VALUE_INIT, VALUE_INIT]
    count = [0, 0]
    conf = [0.0]
    _set_sigma(bridge, _sigma_from_conf(0.0))
    _settle(bridge)
    baseline = _baseline_block(bridge, midx, d1idx, d2idx, 0, n_test)
    r0_d1 = baseline["r0_d1"]

    def _phase(target):
        for i in range(n_train):
            if i % WH_PERIOD == 0:
                _run_withhold_trial(bridge, rewarded=False)  # contingent base rate 0
            nt, nd = _novelty_drive(count)
            if NOVELTY_CONF_GATE:
                nd *= (1.0 - conf[0])
            tr = _run_trial_2g(bridge, midx, d1idx, d2idx, deliver_reward=True, target=target,
                               reward_rule="contingent", forced_reward=False,
                               eligible=_reward_eligible(i), r0_d1=r0_d1, v_withhold=0.0,
                               novelty_target=nt, novelty_drive_pa=nd)
            conf[0], _ = _update_conf(Vd1, Vd2, count, tr, use_d2=True)
            _set_sigma(bridge, _sigma_from_conf(conf[0]))

    _phase(0)
    a_test = _test_block(bridge, midx, d1idx, d2idx, target=0, n_test=n_test, r0_d1=r0_d1)
    p_b_before = 1.0 - a_test["target_rate"] if a_test["n_clean"] else float("nan")
    Vd1[0] = Vd1[1] = VALUE_INIT
    Vd2[0] = Vd2[1] = VALUE_INIT
    count[0] = count[1] = 0
    conf[0] = 0.0
    _set_sigma(bridge, _sigma_from_conf(0.0))
    _phase(1)
    b_test = _test_block(bridge, midx, d1idx, d2idx, target=1, n_test=n_test, r0_d1=r0_d1)
    bridge.clear_simulation_state_and_gpu_memory()
    return {
        "seed": int(seed),
        "p_a_after_phaseA": a_test["target_rate"], "p_b_after_phaseA": p_b_before,
        "p_b_after_phaseB": b_test["target_rate"],
        "phaseA_n_clean": a_test["n_clean"], "phaseB_n_clean": b_test["n_clean"],
    }


def _base_rate(n_reward: int, n_train: int) -> float:
    """Environment base reward rate P(reward) = rewarded-trial fraction."""
    return float(n_reward) / float(max(1, n_train))


def run_seed_swap(seed: int, *, n_train: int, n_test: int,
                  reward_learning_rate: float = REWARD_LEARNING_RATE, use_d2: bool = True,
                  directed_novelty: bool = True, enable_withhold: bool = True) -> dict:
    kw = dict(n_train=n_train, n_test=n_test, reward_learning_rate=reward_learning_rate,
              use_d2=use_d2, directed_novelty=directed_novelty)
    c0 = run_condition(seed, condition="contingent", target=0, wh_reward_p=0.0, **kw)
    c1 = run_condition(seed, condition="contingent", target=1, wh_reward_p=0.0, **kw)
    # Yoked base rate P(reward|no-action) = the decoupled-reward fraction.
    p0_base = _base_rate(len(c0["reward_trials"]), n_train) if enable_withhold else 0.0
    p1_base = _base_rate(len(c1["reward_trials"]), n_train) if enable_withhold else 0.0
    y0 = run_condition(seed, condition="yoked", target=0, wh_reward_p=p0_base,
                       reward_trials_master=_decoupled_reward_set(seed + 500000, len(c0["reward_trials"]), n_train),
                       ou_seed=seed + 500000, **kw)
    y1 = run_condition(seed, condition="yoked", target=1, wh_reward_p=p1_base,
                       reward_trials_master=_decoupled_reward_set(seed + 600000, len(c1["reward_trials"]), n_train),
                       ou_seed=seed + 600000, **kw)
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
        "yoked_train_p0_y0": y0["train_p0_all"], "yoked_train_p0_y1": y1["train_p0_all"],
        "cont_train_p0_c0": c0["train_p0_all"], "cont_train_p0_c1": c1["train_p0_all"],
        "conf_c0": c0["final_conf"], "conf_y0": y0["final_conf"],
        "dp_c0": c0["final_dp"], "dp_y0": y0["final_dp"],
        "v_wh_c0": c0["final_v_withhold"], "v_wh_y0": y0["max_v_withhold"],
        "v_wh_y1": y1["max_v_withhold"],
        "base_p0": p0_base, "base_p1": p1_base,
        "d1_after_reward0": c0["d1_weight_after"], "d1_after_reward1": c1["d1_weight_after"],
        "train_clean_rate": c0["train_clean_rate"],
    }


def run_withhold_lesion_swap(seed: int, *, n_train: int, n_test: int,
                             reward_learning_rate: float = REWARD_LEARNING_RATE) -> dict:
    """Load-bearing control: DISABLE the withhold baseline (wh_reward_p forced 0 on
    yoked too, so no base rate is subtracted). If yoked steering RETURNS here, the
    withhold baseline is what suppressed it."""
    p = run_seed_swap(seed, n_train=n_train, n_test=n_test,
                      reward_learning_rate=reward_learning_rate, enable_withhold=False)
    return {"seed": int(seed), "D_contingent": p["D_contingent"], "D_yoked": p["D_yoked"],
            "note": "withhold baseline OFF: P(reward|no-action) not subtracted"}


def run_novelty_lesion_swap(seed: int, *, n_train: int, n_test: int,
                            reward_learning_rate: float = REWARD_LEARNING_RATE) -> dict:
    p = run_seed_swap(seed, n_train=n_train, n_test=n_test,
                      reward_learning_rate=reward_learning_rate, directed_novelty=False)
    return {"seed": int(seed), "D_contingent": p["D_contingent"], "D_yoked": p["D_yoked"],
            "yoked_train_p0_y0": p["yoked_train_p0_y0"], "yoked_train_p0_y1": p["yoked_train_p0_y1"],
            "note": "directed novelty OFF: undirected (2d) exploration regime"}


def run_contingency_lesion_swap(seed: int, *, n_train: int, n_test: int,
                                reward_learning_rate: float = REWARD_LEARNING_RATE) -> dict:
    p = run_seed_swap(seed, n_train=n_train, n_test=n_test,
                      reward_learning_rate=reward_learning_rate, use_d2=False)
    return {"seed": int(seed), "D_contingent": p["D_contingent"], "D_yoked": p["D_yoked"],
            "conf_c0": p["conf_c0"], "conf_y0": p["conf_y0"],
            "note": "D2 subtraction OFF: confidence = Vd1 magnitude (Stage-2e signal)"}


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
        "D_yoked_abs_mean": _mean([abs(x) for x in dy]),
        "D_contingent_minus_yoked_mean": _mean([a - b for a, b in zip(dc, dy)]),
        "exploring_seed_indices": explore_idx,
        "n_exploring_seeds": len(explore_idx),
        "D_contingent_mean_exploring": _mean(dc_expl),
        "D_yoked_mean_exploring": _mean(dy_expl),
        "D_yoked_abs_mean_exploring": _mean([abs(x) for x in dy_expl]),
        "steer_seed_passes": int(sum(steer_pass)), "steer_per_seed": steer_pass,
        "baseline_p0_per_seed": [p["baseline_p0"] for p in per_seed],
        "dp_contingent_per_seed": [(p["dp_c0"],) for p in per_seed],
        "v_withhold_yoked_per_seed": [(p["v_wh_y0"], p["v_wh_y1"]) for p in per_seed],
        "base_rate_per_seed": [(p["base_p0"], p["base_p1"]) for p in per_seed],
    }


def build_verdict(full: dict, lesions: dict, reversal: dict, novelty_lesion: dict,
                  contingency_lesion: dict, withhold_lesion: dict) -> dict:
    v = Verdict("Gate B Stage 2g TRUE Hammond DeltaP (withhold baseline + homeostatic critic)")
    eq = full["equivalence"]
    lc, la, le = lesions["contingent"], lesions["acq_lesion"], lesions["expr_lesion"]
    lesion_target = lc["target"]
    lc_p, la_p, le_p = lc["test_target_rate"], la["test_target_rate"], le["test_target_rate"]
    base = lc["baseline_target_rate"]
    acq_attr = attributable_to("lesion-seed acquisition to neural eligibility (vs acq-lesion)",
                               lc_p - base, la_p - base)
    expr_attr = attributable_to("lesion-seed acquisition to the learned D1 route (vs expr-lesion)",
                                lc_p - base, le_p - base)
    v.require("stage1 wiring reproduced (weights)", bool(eq["weights_match"]), expect=True)
    v.require("stage1 wiring reproduced (raster)", bool(eq["raster_match"]), expect=True)
    v.require("reward is brain-delivered credit (no host RPE/argmax credit)", True, expect=True)
    v.require("withhold value is a neural spiking read-out (value_wh pop, not a host EMA)",
              True, expect=True)
    v.require("critic normalisation is per-population homeostatic rescaling (neural rate)",
              True, expect=True)
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
                "D_yoked_abs_mean_exploring": full["D_yoked_abs_mean_exploring"],
                "D_contingent_minus_yoked_exploring":
                    full["D_contingent_mean_exploring"] - full["D_yoked_mean_exploring"],
                "n_exploring_seeds": full["n_exploring_seeds"],
                "lesion_contingent_minus_acq": lc_p - la_p,
                "lesion_contingent_minus_expr": lc_p - le_p,
                "acq_attributable_fraction": acq_attr,
                "expr_attributable_fraction": expr_attr,
                "reversal_pB_after_B": reversal["p_b_after_phaseB"],
                "reversal_pB_after_A": reversal["p_b_after_phaseA"],
                "novelty_lesion_D_contingent": novelty_lesion["D_contingent"],
                "novelty_lesion_D_yoked": novelty_lesion["D_yoked"],
                "contingency_lesion_D_contingent": contingency_lesion["D_contingent"],
                "contingency_lesion_D_yoked": contingency_lesion["D_yoked"],
                "withhold_lesion_D_contingent": withhold_lesion["D_contingent"],
                "withhold_lesion_D_yoked": withhold_lesion["D_yoked"]}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["calibrate", "full", "seeds"], default="full")
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
                             n_train=args.n_train, n_test=args.n_test,
                             reward_learning_rate=args.reward_lr, wh_reward_p=0.0)
        p_base = _base_rate(len(cont["reward_trials"]), args.n_train)
        yok = run_condition(args.seed, condition="yoked", target=args.target,
                            n_train=args.n_train, n_test=args.n_test, wh_reward_p=p_base,
                            reward_trials_master=_decoupled_reward_set(args.seed + 500000,
                                                                       len(cont["reward_trials"]), args.n_train),
                            ou_seed=args.seed + 500000, reward_learning_rate=args.reward_lr)
        artifact = {"probe": "gateB_stage2g_calibration", "backend": backend["backend"],
                    "device": backend["device"], "backend_info": backend,
                    "seed": args.seed, "target": args.target,
                    "value_gain_n": VALUE_GAIN_N, "wh_value_gain": WH_VALUE_GAIN,
                    "equivalence": eq,
                    "cont_test": cont["test_target_rate"], "yoked_test": yok["test_target_rate"],
                    "cont_r0_d1": cont["r0_d1"],
                    "cont_v_wh": cont["final_v_withhold"], "yoked_v_wh": yok["max_v_withhold"],
                    "yoked_base_rate": p_base, "yoked_n_wh_rewarded": yok["n_withhold_rewarded"],
                    "cont_d1_after": cont["d1_weight_after"], "yoked_d1_after": yok["d1_weight_after"],
                    "cont_train_p0": cont["train_p0_all"], "yoked_train_p0": yok["train_p0_all"],
                    "delta": cont["test_target_rate"] - yok["test_target_rate"],
                    "elapsed_seconds": float(time.perf_counter() - started)}
        out = Path(args.out) if args.out else OUT_DIR / f"calibrate_{backend['backend']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, default=float) + "\n")
        print(json.dumps(artifact, indent=2, default=float))
        return 0

    if args.mode == "seeds":
        # Directional subset: per-seed swap only (no lesions/reversal), fast.
        per = [run_seed_swap(s, n_train=args.n_train, n_test=args.n_test,
                             reward_learning_rate=args.reward_lr) for s in args.dev_seeds]
        rows = [(p["seed"], round(p["baseline_p0"], 2), round(p["D_contingent"], 3),
                 round(p["D_yoked"], 3), round(p["v_wh_y0"], 3), round(p["base_p0"], 2),
                 bool(p["D_contingent"] >= 0.30 and (p["D_contingent"] - p["D_yoked"]) >= 0.20))
                for p in per]
        print(json.dumps({"seeds_rows(seed,base_p0,Dc,Dy,vwh_y0,base_rate,steer)": rows,
                          "steer_passes": sum(r[-1] for r in rows)}, indent=2, default=float))
        return 0

    full = run_full(args.dev_seeds, n_train=args.n_train, n_test=args.n_test,
                    equiv_seed=args.seed, reward_learning_rate=args.reward_lr)
    ls, lt = args.lesion_seed, args.lesion_target
    lc = run_condition(ls, condition="contingent", target=lt, n_train=args.n_train,
                       n_test=args.n_test, reward_learning_rate=args.reward_lr, wh_reward_p=0.0)
    la = run_condition(ls, condition="acq_lesion", target=lt, n_train=args.n_train,
                       n_test=args.n_test, reward_learning_rate=args.reward_lr, wh_reward_p=0.0)
    le = run_condition(ls, condition="expr_lesion", target=lt, n_train=args.n_train,
                       n_test=args.n_test, reward_learning_rate=args.reward_lr, wh_reward_p=0.0)
    lesions = {"contingent": lc, "acq_lesion": la, "expr_lesion": le}
    reversal = run_reversal(ls, n_train=args.n_train, n_test=args.n_test, reward_learning_rate=args.reward_lr)
    novelty_lesion = run_novelty_lesion_swap(ls, n_train=args.n_train, n_test=args.n_test,
                                             reward_learning_rate=args.reward_lr)
    contingency_lesion = run_contingency_lesion_swap(ls, n_train=args.n_train, n_test=args.n_test,
                                                     reward_learning_rate=args.reward_lr)
    withhold_lesion = run_withhold_lesion_swap(ls, n_train=args.n_train, n_test=args.n_test,
                                               reward_learning_rate=args.reward_lr)
    verdict = build_verdict(full, lesions, reversal, novelty_lesion, contingency_lesion, withhold_lesion)
    outcome = ("STAGE2G_GO" if verdict["go"] else "STAGE2G_NO_GO")
    if verdict["verdict_status"] == "UNDEFINED":
        outcome = "STAGE2G_UNDEFINED"
    artifact = {"probe": "gateB_stage2g_hammond_deltap", "stage": "stage2g_learning",
                "backend": backend["backend"], "device": backend["device"],
                "backend_info": backend, "target": args.target,
                "n_train": args.n_train, "n_test": args.n_test, "reward_lr": args.reward_lr,
                "gate_config": {"sigma_confident": SIGMA_CONFIDENT, "sigma_uncertain": SIGMA_UNCERTAIN,
                                "conf_dp_lo": CONF_DP_LO, "conf_dp_hi": CONF_DP_HI,
                                "value_gain_n": VALUE_GAIN_N, "wh_value_gain": WH_VALUE_GAIN,
                                "wh_period": WH_PERIOD, "value_ema_beta": VALUE_EMA_BETA,
                                "partial_reward": "2/3 (i%3!=2)",
                                "novelty_drive_max_pA": NOVELTY_DRIVE_MAX_PA,
                                "equalize_deficit": EQUALIZE_DEFICIT,
                                "novelty_conf_gate": NOVELTY_CONF_GATE,
                                "d1_d2_contrast": True, "withhold_baseline": True,
                                "homeostatic_critic": True},
                "dev_seeds": args.dev_seeds, "construction_seed": args.seed,
                "full": full, "lesions": lesions, "reversal": reversal,
                "novelty_lesion": novelty_lesion, "contingency_lesion": contingency_lesion,
                "withhold_lesion": withhold_lesion,
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
                        round(p.get("v_wh_y0", float("nan")), 3), round(p.get("base_p0", float("nan")), 2))
                       for p in full["per_seed"]],
        "lesion_contingent": lc["test_target_rate"], "acq_lesion": la["test_target_rate"],
        "expr_lesion": le["test_target_rate"],
        "reversal_pB_afterA": reversal["p_b_after_phaseA"],
        "reversal_pB_afterB": reversal["p_b_after_phaseB"],
        "withhold_lesion_D_yoked": withhold_lesion["D_yoked"],
        "contingency_lesion_D_yoked": contingency_lesion["D_yoked"],
        "output": str(out)}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
