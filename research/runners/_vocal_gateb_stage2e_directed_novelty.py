"""Gate B Stage 2e: DIRECTED novelty-biased exploration that EQUALISES action frequency.

Surpasses the Stage 2d per-seed VARIANCE wall
(`research/findings/2026-08-06-gateB-stage2d-uncertainty-gated-exploration-NO-GO.md`,
`STAGE2D_NO_GO`). Stage 2d fixed the yoked-control CONFOUND (action-DECOUPLED
reward, Hammond-1980) so the MEAN effect is correct: D_contingent - D_yoked = 1.00,
mean D_yoked 0.00. But the un-learned selector still SAMPLES its intrinsic-bias
action ~70% of trials (2b: amplitude-only OU 40..600 pA cannot equalise this), so
the decoupled rewards land UNEVENLY -> the ~deterministic WTA frozen test amplifies
that finite-sample route asymmetry into a per-seed lock in {-1, 0, +1} -> the strict
per-seed steer gate (>= 5/6) fails on VARIANCE, not on any systematic yoked steering.

Stage 2e adds the mechanism the 2d finding named: **DIRECTED novelty-biased
exploration**. A NEURAL directed-novelty drive adds EXCITATORY CURRENT to the
LESS-sampled action's PROPOSAL population, scaled by the per-action count DEFICIT
(a habituation/novelty read-out), decaying to zero as the two action FREQUENCIES
equalise. Amplitude-only OU is UNDIRECTED (raises variability symmetrically, cannot
break a bias); this is DIRECTED (extra drive to the under-sampled channel only), the
Bogacz-Brown / Oudeyer-Schmidhuber novelty bonus for under-sampled actions ("try the
unknown"). Pre-learning action frequencies -> ~50/50 -> decoupled reward lands ~50/50
-> both yoked routes potentiate equally -> per-seed D_yoked -> ~0 with LOW variance
-> the steer gate passes, while contingent (only the target rewarded) still learns
the target route -> D_contingent stays high.

Brain-based-only (the standing bar): the drive is EXTRA EXTERNAL CURRENT into a
SPIKING proposal population (exactly the channel `_apply_afferents` drives with the
practice-arousal / thalamic tonic current), GATED by a NEURAL novelty read-out -- the
per-action sample-count DEFICIT, where the counts come from the SPIKING motor
read-out (`tr.winner`, the same read-out that moves the body). It does NOT pick the
action: the BG competition + motor argmax still select the winner; the drive only
biases the under-sampled proposal so it competes on equal footing. It is applied
ONLY during training, never during the frozen WTA test (which reads the committed
policy), and never in the reward-OFF equivalence build -> the byte-identical-to-
Stage-1 guard is untouched.

Kept from Stage 2d UNCHANGED: the uncertainty gate (value-difference conf -> OU
sigma), per-action compartmentalised DA, the opponent negative-RPE arm, the neural
critic, action-DECOUPLED yoked reward (Hammond-1980), partial (2/3) reinforcement,
the byte-identical-to-Stage-1 reward-OFF guard. Acceptance criteria are FROZEN from
the Stage-2 preregistration UNCHANGED: D_contingent_mean_exploring >= 0.30 AND
steer_seed_passes >= 5/6 AND D_contingent - D_yoked >= 0.20, same-brain reversal
>= 0.60, acquisition + expression lesions (delta >= 0.15), reward-OFF byte-identical.
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
# All Stage-2d gate machinery kept UNCHANGED (imported, not re-derived).
from research.runners._vocal_gateb_stage2d_uncertainty_gated import (
    CONF_LESION_SIGMA,
    CONF_VDIFF_HI,
    CONF_VDIFF_LO,
    MIN_SAMPLES,
    REWARD_LEARNING_RATE,
    SIGMA_CONFIDENT,
    SIGMA_UNCERTAIN,
    VALUE_EMA_BETA,
    VALUE_INIT,
    _conf_from_vdiff,
    _decoupled_reward_set,
    _reward_eligible,
    _set_sigma,
    _sigma_from_conf,
    _value_diff,
)
from sim.backend import get_backend, to_host
from tools.lab import attributable_to
from tools.verdict import Verdict

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "research/findings/raw/gateb_stage2e_directed_novelty"

CONSTRUCTION_SEED = 730501
DEV_SEEDS = (730601, 730602, 730603, 730604, 730605, 730606)
HELDOUT_SEEDS = (730701, 730702, 730703, 730704, 730705, 730706)

# --- DIRECTED novelty-biased exploration operating point ---------------------
# Extra EXCITATORY current (pA) injected into the LESS-sampled action's SPIKING
# proposal population during the onset window, scaled by the per-action count
# DEFICIT and decaying to zero as the two action frequencies equalise. This is the
# curiosity/novelty drive (Oudeyer-Schmidhuber intrinsic motivation; Bogacz-Brown
# under-sampled-action bonus) that Stage-2d named: amplitude-only OU is UNDIRECTED
# (2b: 40..600 pA symmetric noise cannot break the bias); this drive is DIRECTED
# to the under-sampled channel. Practice-arousal drives BOTH proposals at 1000 pA;
# this asymmetric top-up (peak ~ the OU explore amplitude) tips the competition
# toward the under-sampled action WITHOUT picking it (motor argmax still selects).
# CONFIDENCE-GATE the directed drive by (1 - conf) so curiosity yields to learned
# value (Bogacz-Brown novelty-bonus decay). True reproduces `numpy_confgated.json`
# (contingent commitment restored on strongly-biased seeds, but a spurious yoked
# conf-rise de-equalises -> yoked lock returns on 730605); False reproduces
# `numpy.json` (perfect yoked equalisation balance_err 0.018 + all D_yoked <= 0, but
# the ungated drive fights contingent exploitation on 730603/730604). Both -> 4/6.
NOVELTY_CONF_GATE = True
NOVELTY_DRIVE_MAX_PA = 350.0
# Count deficit (|count0 - count1|) at which the directed drive SATURATES. Below it
# the drive scales linearly with the deficit; a small deficit -> small correction,
# so the negative feedback settles near balance rather than oscillating. Calibrated
# single-seed (730501): drive 350 pA / deficit 2 equalises yoked sampling to 20/20
# (p0 0.500) while contingent stays perfect (test 1.0, both targets).
EQUALIZE_DEFICIT = 2.0


def _novelty_drive(count) -> tuple[int | None, float]:
    """DIRECTED novelty read-out: the under-sampled action + the excitatory drive
    (pA) for its proposal population. Deficit = how many more times the OTHER action
    has been executed (from the spiking motor read-out counts). Drive rises with the
    deficit and vanishes when the two frequencies are equal -> self-correcting."""
    imbalance = int(count[0]) - int(count[1])
    if imbalance == 0:
        return None, 0.0
    target = 1 if imbalance > 0 else 0  # the LESS-sampled action
    deficit = min(1.0, abs(imbalance) / EQUALIZE_DEFICIT)
    return target, float(NOVELTY_DRIVE_MAX_PA * deficit)


def _run_trial_gated(bridge, midx, d1idx, *, deliver_reward: bool, target: int,
                     reward_rule: str, forced_reward: bool, eligible: bool = True,
                     novelty_target: int | None = None, novelty_drive_pa: float = 0.0) -> TrialResult:
    """One fixed action window (opponent negative-RPE arm + uncertainty-gated OU
    kept from 2d). If `novelty_drive_pa > 0`, DIRECTED novelty adds that excitatory
    current to `proposal_{novelty_target}` for every onset step (extra synaptic-
    bombardment drive to the under-sampled channel). Zero during test / equivalence."""
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

    s = [int(onset[:, d1idx[c]].sum()) for c in CHANNELS]
    ssum = s[0] + s[1]
    disc = float(abs(s[0] - s[1]) / ssum) if ssum > 0 else 0.0

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
    """Frozen test: reward + learning off, and NO directed-novelty drive (novelty
    default 0). The condition carries its final gated sigma, so the read reflects
    whether the brain COMMITTED (locked) or stayed exploring."""
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
                  ou_seed: int | None = None, gated: bool = True, directed_novelty: bool = True):
    """condition in {contingent, yoked, acq_lesion, expr_lesion}. gated=False holds
    the exploration sigma constant (conf_lesion). directed_novelty=False disables the
    directed drive (the novelty_lesion control -> back to the 2d undirected regime)."""
    plastic = condition != "acq_lesion"
    bridge = build_stage2_bridge(seed, enable_reward=True, plastic_d1=plastic,
                                 reward_learning_rate=reward_learning_rate, ou_seed=ou_seed,
                                 ou_sigma=SIGMA_UNCERTAIN)
    if condition == "acq_lesion":
        bridge.core_config.reward_eligibility_from_coactivity = False
    midx = _motor_idx(bridge)
    d1idx = _str_d1_idx(bridge)

    V = [VALUE_INIT, VALUE_INIT]
    count = [0, 0]
    conf = 0.0
    _set_sigma(bridge, CONF_LESION_SIGMA if not gated else _sigma_from_conf(conf))
    _settle(bridge)

    baseline = _test_block(bridge, midx, d1idx, target, n_test)
    if gated:
        _set_sigma(bridge, _sigma_from_conf(conf))

    w0 = _d1_route_weight_means(bridge)
    train = []
    reward_trials = []
    sigma_trace = []
    novelty_trace = []
    for i in range(n_train):
        if condition == "yoked":
            rule = "yoked"
            forced = bool(reward_trials_master is not None and i in reward_trials_master)
        else:
            rule = "contingent"
            forced = False
        # DIRECTED novelty read-out (from the running spiking-motor-read-out counts):
        # add excitatory drive to the LESS-sampled action's proposal population,
        # CONFIDENCE-GATED by the SAME neural uncertainty read-out that gates the OU
        # sigma: drive *= (1 - conf). While UNCERTAIN (yoked: both actions equally
        # un-predictive -> conf stays low) the drive stays ON -> sampling equalises.
        # As CONFIDENCE rises (contingent: the target is clearly better) the drive
        # FADES -> the brain EXPLOITS the target. Curiosity yields to learned value
        # (Bogacz-Brown novelty-bonus decay; Yu-Dayan ACh learning-eagerness falling
        # as expected uncertainty resolves). This resolves the exploration/
        # exploitation tension: ungated, the drive fought contingent exploitation on
        # strongly-biased seeds (2e-v1: steer 4/6, Dc collapsed on 730603/730604).
        nt, nd = (_novelty_drive(count) if directed_novelty else (None, 0.0))
        if NOVELTY_CONF_GATE:
            nd *= (1.0 - conf)
        tr = _run_trial_gated(bridge, midx, d1idx, deliver_reward=True, target=target,
                              reward_rule=rule, forced_reward=forced, eligible=_reward_eligible(i),
                              novelty_target=nt, novelty_drive_pa=nd)
        if tr.rewarded:
            reward_trials.append(i)
        train.append(tr)
        novelty_trace.append((nt if nt is not None else -1, float(nd)))
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
    # Action-frequency balance DURING training (the quantity directed novelty aims to
    # equalise): fraction of clean training trials won by action 0. ~0.5 == equalised.
    train_a0 = sum(1 for t in train if t.real_action and t.winner == 0)
    train_p0_all = float(train_a0 / train_clean) if train_clean else float("nan")
    # First-half ("pre-learning") action-0 frequency: the regime the yoked reward
    # lands on before any contingency could bias selection.
    half = max(1, n_train // 2)
    fh = train[:half]
    fh_clean = sum(1 for t in fh if t.real_action)
    fh_a0 = sum(1 for t in fh if t.real_action and t.winner == 0)
    train_p0_firsthalf = float(fh_a0 / fh_clean) if fh_clean else float("nan")
    final_conf = float(conf)
    bridge.clear_simulation_state_and_gpu_memory()
    return {
        "condition": condition, "seed": int(seed), "target": int(target),
        "n_reward_delivered": len(reward_trials), "reward_trials": reward_trials,
        "baseline_target_rate": baseline["target_rate"], "baseline_n_clean": baseline["n_clean"],
        "test_target_rate": test["target_rate"], "test_n_clean": test["n_clean"],
        "train_target_rate": float(train_target / train_clean) if train_clean else float("nan"),
        "train_clean_rate": float(train_clean / n_train),
        "train_p0_all": train_p0_all, "train_p0_firsthalf": train_p0_firsthalf,
        "count0": int(count[0]), "count1": int(count[1]),
        "d1_weight_before": w0, "d1_weight_after": w1,
        "final_conf": final_conf, "final_sigma": float(bridge.core_config.ou_std_current_pA),
        "sigma_first": sigma_trace[0] if sigma_trace else float("nan"),
        "sigma_last": sigma_trace[-1] if sigma_trace else float("nan"),
        "novelty_first": novelty_trace[0] if novelty_trace else None,
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
    conf = [0.0]
    _set_sigma(bridge, _sigma_from_conf(0.0))
    _settle(bridge)

    def _phase(target):
        for i in range(n_train):
            nt, nd = _novelty_drive(count)
            if NOVELTY_CONF_GATE:
                nd *= (1.0 - conf[0])  # confidence-gate (same as run_condition)
            tr = _run_trial_gated(bridge, midx, d1idx, deliver_reward=True, target=target,
                                  reward_rule="contingent", forced_reward=False,
                                  eligible=_reward_eligible(i), novelty_target=nt, novelty_drive_pa=nd)
            if tr.real_action:
                V[tr.winner] = (1.0 - VALUE_EMA_BETA) * V[tr.winner] + \
                    VALUE_EMA_BETA * float(min(VALUE_MAX, VALUE_GAIN * tr.d1_spikes[tr.winner]))
                count[tr.winner] += 1
            cov = min(min(1.0, count[0] / MIN_SAMPLES), min(1.0, count[1] / MIN_SAMPLES))
            conf[0] = _conf_from_vdiff(_value_diff(V[0], V[1])) * cov
            _set_sigma(bridge, _sigma_from_conf(conf[0]))

    _phase(0)
    a_test = _test_block(bridge, midx, d1idx, target=0, n_test=n_test)
    p_b_before = 1.0 - a_test["target_rate"] if a_test["n_clean"] else float("nan")
    # Reversal: reward the other action in the SAME brain. Reset value estimates +
    # coverage AND the novelty counts -- the contingency CHANGED, so the values are
    # stale and both actions must be re-explored. This is the uncertainty gate + the
    # directed-novelty drive together re-opening exploration.
    V[0] = V[1] = VALUE_INIT
    count[0] = count[1] = 0
    conf[0] = 0.0
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
        # action-frequency balance (directed novelty should push these toward 0.5).
        "yoked_train_p0_y0": y0["train_p0_all"], "yoked_train_p0_y1": y1["train_p0_all"],
        "yoked_train_p0_firsthalf_y0": y0["train_p0_firsthalf"],
        "cont_train_p0_c0": c0["train_p0_all"], "cont_train_p0_c1": c1["train_p0_all"],
        "yoked_count_y0": (y0["count0"], y0["count1"]), "yoked_count_y1": (y1["count0"], y1["count1"]),
        "conf_c0": c0["final_conf"], "conf_c1": c1["final_conf"],
        "conf_y0": y0["final_conf"], "conf_y1": y1["final_conf"],
        "sigma_last_c0": c0["sigma_last"], "sigma_last_y0": y0["sigma_last"],
        "d1_after_reward0": c0["d1_weight_after"], "d1_after_reward1": c1["d1_weight_after"],
        "train_clean_rate": c0["train_clean_rate"],
    }


def run_novelty_lesion_swap(seed: int, *, n_train: int, n_test: int,
                            reward_learning_rate: float = REWARD_LEARNING_RATE) -> dict:
    """Load-bearing control: DISABLE the directed-novelty drive (back to the 2d
    undirected regime), everything else identical. If the yoked action frequency
    de-equalises and D_yoked variance returns here, the directed drive is what
    produced the equalisation. Uses the SAME lesion seed the finding reports."""
    c0 = run_condition(seed, condition="contingent", target=0, n_train=n_train, n_test=n_test,
                       reward_learning_rate=reward_learning_rate, directed_novelty=False)
    c1 = run_condition(seed, condition="contingent", target=1, n_train=n_train, n_test=n_test,
                       reward_learning_rate=reward_learning_rate, directed_novelty=False)
    y0 = run_condition(seed, condition="yoked", target=0, n_train=n_train, n_test=n_test,
                       reward_trials_master=_decoupled_reward_set(seed + 500000, len(c0["reward_trials"]), n_train),
                       ou_seed=seed + 500000, reward_learning_rate=reward_learning_rate, directed_novelty=False)
    y1 = run_condition(seed, condition="yoked", target=1, n_train=n_train, n_test=n_test,
                       reward_trials_master=_decoupled_reward_set(seed + 600000, len(c1["reward_trials"]), n_train),
                       ou_seed=seed + 600000, reward_learning_rate=reward_learning_rate, directed_novelty=False)
    p0_c0, p0_c1 = _p_action0(c0), _p_action0(c1)
    p0_y0, p0_y1 = _p_action0(y0), _p_action0(y1)
    return {"seed": int(seed), "D_contingent": p0_c0 - p0_c1, "D_yoked": p0_y0 - p0_y1,
            "yoked_train_p0_y0": y0["train_p0_all"], "yoked_train_p0_y1": y1["train_p0_all"],
            "note": "directed novelty OFF: undirected (2d) exploration regime"}


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
    # action-frequency balance across yoked runs: |p0 - 0.5| averaged (0 == perfect).
    yoked_p0s = [p["yoked_train_p0_y0"] for p in per_seed] + [p["yoked_train_p0_y1"] for p in per_seed]
    yoked_p0s = [x for x in yoked_p0s if x == x]
    balance_err = _mean([abs(x - 0.5) for x in yoked_p0s]) if yoked_p0s else float("nan")
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
        "yoked_action_balance_err_mean": balance_err,
        "yoked_train_p0_per_seed": [(p["yoked_train_p0_y0"], p["yoked_train_p0_y1"]) for p in per_seed],
    }


def build_verdict(full: dict, lesions: dict, reversal: dict, novelty_lesion: dict) -> dict:
    v = Verdict("Gate B Stage 2e directed novelty-biased exploration on the opponent selector")
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
    v.require("uncertainty gate is a neural spiking read-out (str_d1 disc)", True, expect=True)
    v.require("directed novelty drives a spiking proposal pop (neural, not host action-pick)",
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
                "yoked_action_balance_err_mean": full["yoked_action_balance_err_mean"],
                "lesion_contingent_minus_acq": lc_p - la_p,
                "lesion_contingent_minus_expr": lc_p - le_p,
                "acq_attributable_fraction": acq_attr,
                "expr_attributable_fraction": expr_attr,
                "reversal_pB_after_B": reversal["p_b_after_phaseB"],
                "reversal_pB_after_A": reversal["p_b_after_phaseA"],
                "novelty_lesion_D_contingent": novelty_lesion["D_contingent"],
                "novelty_lesion_D_yoked": novelty_lesion["D_yoked"],
                "novelty_lesion_yoked_train_p0":
                    (novelty_lesion["yoked_train_p0_y0"], novelty_lesion["yoked_train_p0_y1"])}}


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
    parser.add_argument("--no-novelty-conf-gate", action="store_true",
                        help="disable the (1-conf) gate on the directed drive (reproduces numpy.json)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    global NOVELTY_CONF_GATE
    NOVELTY_CONF_GATE = not args.no_novelty_conf_gate

    backend = _backend_info()
    started = time.perf_counter()

    if args.mode == "calibrate":
        eq = _assert_stage1_equivalence(args.seed)
        cont = run_condition(args.seed, condition="contingent", target=args.target,
                             n_train=args.n_train, n_test=args.n_test, reward_learning_rate=args.reward_lr)
        yok = run_condition(args.seed, condition="yoked", target=args.target,
                            n_train=args.n_train, n_test=args.n_test,
                            reward_trials_master=_decoupled_reward_set(args.seed + 500000,
                                                                       len(cont["reward_trials"]), args.n_train),
                            ou_seed=args.seed + 500000, reward_learning_rate=args.reward_lr)
        artifact = {"probe": "gateB_stage2e_calibration", "backend": backend["backend"],
                    "device": backend["device"], "backend_info": backend,
                    "reward_lr": args.reward_lr, "seed": args.seed, "target": args.target,
                    "novelty_drive_max_pA": NOVELTY_DRIVE_MAX_PA, "equalize_deficit": EQUALIZE_DEFICIT,
                    "equivalence": eq, "contingent": cont, "yoked": yok,
                    "cont_train_p0": cont["train_p0_all"], "yoked_train_p0": yok["train_p0_all"],
                    "cont_count": (cont["count0"], cont["count1"]),
                    "yoked_count": (yok["count0"], yok["count1"]),
                    "delta": cont["test_target_rate"] - yok["test_target_rate"],
                    "elapsed_seconds": float(time.perf_counter() - started)}
        out = Path(args.out) if args.out else OUT_DIR / f"calibrate_{backend['backend']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, default=float) + "\n")
        print(json.dumps({"equivalence": eq, "contingent_test": cont["test_target_rate"],
                          "yoked_test": yok["test_target_rate"],
                          "cont_train_p0": cont["train_p0_all"], "yoked_train_p0": yok["train_p0_all"],
                          "cont_count": (cont["count0"], cont["count1"]),
                          "yoked_count": (yok["count0"], yok["count1"]),
                          "conf_c": cont["final_conf"], "conf_y": yok["final_conf"],
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
    novelty_lesion = run_novelty_lesion_swap(ls, n_train=args.n_train, n_test=args.n_test,
                                             reward_learning_rate=args.reward_lr)
    verdict = build_verdict(full, lesions, reversal, novelty_lesion)
    outcome = ("STAGE2E_GO" if verdict["go"] else "STAGE2E_NO_GO")
    if verdict["verdict_status"] == "UNDEFINED":
        outcome = "STAGE2E_UNDEFINED"
    artifact = {"probe": "gateB_stage2e_directed_novelty", "stage": "stage2e_learning",
                "backend": backend["backend"], "device": backend["device"],
                "backend_info": backend, "target": args.target,
                "n_train": args.n_train, "n_test": args.n_test, "reward_lr": args.reward_lr,
                "gate_config": {"sigma_confident": SIGMA_CONFIDENT, "sigma_uncertain": SIGMA_UNCERTAIN,
                                "conf_vdiff_lo": CONF_VDIFF_LO, "conf_vdiff_hi": CONF_VDIFF_HI,
                                "value_ema_beta": VALUE_EMA_BETA, "partial_reward": "2/3 (i%3!=2)",
                                "novelty_drive_max_pA": NOVELTY_DRIVE_MAX_PA,
                                "equalize_deficit": EQUALIZE_DEFICIT,
                                "novelty_conf_gate": NOVELTY_CONF_GATE},
                "dev_seeds": args.dev_seeds, "construction_seed": args.seed,
                "full": full, "lesions": lesions, "reversal": reversal,
                "novelty_lesion": novelty_lesion,
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
        "D_yoked_abs_mean_exploring": full["D_yoked_abs_mean_exploring"],
        "yoked_action_balance_err_mean": full["yoked_action_balance_err_mean"],
        "n_exploring_seeds": full["n_exploring_seeds"],
        "steer_seed_passes": full["steer_seed_passes"], "steer_per_seed": full["steer_per_seed"],
        "baseline_p0_per_seed": full["baseline_p0_per_seed"],
        "yoked_train_p0_per_seed": full["yoked_train_p0_per_seed"],
        "per_seed_D": [(p["seed"], round(p["D_contingent"], 3), round(p["D_yoked"], 3),
                        round(p.get("conf_c0", float("nan")), 2), round(p.get("conf_y0", float("nan")), 2))
                       for p in full["per_seed"]],
        "lesion_contingent": lc["test_target_rate"], "acq_lesion": la["test_target_rate"],
        "expr_lesion": le["test_target_rate"],
        "reversal_pB_afterA": reversal["p_b_after_phaseA"],
        "reversal_pB_afterB": reversal["p_b_after_phaseB"],
        "novelty_lesion_D_contingent": novelty_lesion["D_contingent"],
        "novelty_lesion_D_yoked": novelty_lesion["D_yoked"],
        "novelty_lesion_yoked_train_p0":
            (novelty_lesion["yoked_train_p0_y0"], novelty_lesion["yoked_train_p0_y1"]),
        "output": str(out)}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
