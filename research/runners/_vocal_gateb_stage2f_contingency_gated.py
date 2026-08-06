"""Gate B Stage 2f: contingency-based (DeltaP) confidence gate on the opponent selector.

Surpasses the Stage 2e residual
(`research/findings/2026-08-06-gateB-stage2e-directed-novelty-exploration-NO-GO.md`,
`STAGE2E_NO_GO`, steer 4/6, union-5/6): directed novelty EQUALISED action sampling
and removed the 2d yoked-lock killer (all D_yoked <= 0), but the confidence read-out
that gates the exploration/exploitation trade-off was the str_d1 value-DIFFERENCE
(magnitude), which CANNOT separate a genuine action->reward contingency from a
coincidental yoked reward STREAK -- a lucky streak transiently inflates one action's
D1 value, spuriously gating OFF the equalising drive exactly where it must stay ON.

Stage 2f replaces the value-magnitude confidence with a NEURAL **DeltaP / CONTINGENCY
estimate = the per-action D1-minus-D2 contrast** (Hammond-1980 instrumental
contingency). The substrate already carries the opponent evidence: the D2 (indirect /
NoGo) route, tagged per-action and made plastic here, has cp_d1_d2_sign=-1, so a DA
DIP (reward OMITTED, negative RPE) POTENTIATES str_d2_c (canonical A2A/D2 NoGo
learning; Shen 2008; Collins-Frank OpAL) while a DA BURST depresses it. Thus:

    Vd1[c] = EMA(str_d1_c onset rate | action c executed)   # Go / reward-DELIVERED
    Vd2[c] = EMA(str_d2_c onset rate | action c executed)   # NoGo / reward-OMITTED
    net[c] = Vd1[c] - Vd2[c]                                 # per-action DeltaP proxy
    conf   = clip((|net0-net1|/total - LO)/(HI-LO), 0, 1) * coverage

Why this separates yoked from contingent where value-magnitude could not:
  * CONTINGENT (target reliably rewarded, other NEVER): target Vd1 up, Vd2 low ->
    net_target >> 0; the other action, taken during exploration and never rewarded,
    accrues omissions -> Vd2 up, Vd1 down -> net_other < 0. |net0-net1| large -> conf
    rises -> the directed drive fades + OU sigma falls -> the brain EXPLOITS ->
    D_contingent high.
  * YOKED (reward DECOUPLED, partial 2/3): whichever action is taken is sometimes
    rewarded, often not, so EACH action accrues BOTH bursts (Vd1 up) AND omissions
    (Vd2 up) -> net[c] ~ 0 for both -> |net0-net1| ~ 0 -> conf stays low EVEN UNDER A
    LUCKY STREAK (the streak raises Vd1 but the decoupled omissions raise Vd2, and the
    subtraction cancels it -- the exact failure of the Stage-2e value-magnitude read-
    out). The equalising directed drive stays ON -> D_yoked ~ 0, low variance.

Brain-based-only: Vd1, Vd2 are onset spike-count read-outs of the str_d1_c / str_d2_c
SPIKING populations (like the motor read-out that moves the body); the D2 route learns
the omission evidence via the substrate's own per-action DA dip x cp_d1_d2_sign three-
factor rule (NOT a host P(reward|action) counter). The net-contrast->confidence->sigma
arithmetic is the abstracted tonic-neuromodulator controller, the same documented
residual class as Stage-2c's reward-V DA arithmetic. The `contingency_lesion` control
replaces net with Vd1 ALONE (= the Stage-2e value-magnitude signal): if the yoked/
contingent separation collapses there, the D2 (NoGo) subtraction is load-bearing.

Kept from Stage 2e UNCHANGED: DIRECTED novelty-biased exploration (equalises action
frequency), per-action compartmentalised DA, the opponent negative-RPE arm, the neural
critic, action-DECOUPLED yoked reward (Hammond-1980), partial (2/3) reinforcement, the
byte-identical-to-Stage-1 reward-OFF guard (D2 plasticity + tagging gate on
enable_reward, OFF in the equivalence build). Acceptance criteria FROZEN from the
Stage-2 preregistration UNCHANGED: D_contingent_mean_exploring >= 0.30 AND
steer_seed_passes >= 5/6 AND D_contingent - D_yoked >= 0.20, same-brain reversal >=
0.60, acquisition + expression lesions (delta >= 0.15), reward-OFF byte-identical.
"""
from __future__ import annotations

import argparse
import json
import math
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
from sim.backend import get_backend, to_host
from tools.lab import attributable_to
from tools.verdict import Verdict

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "research/findings/raw/gateb_stage2f_contingency_gated"

CONSTRUCTION_SEED = 730501
DEV_SEEDS = (730601, 730602, 730603, 730604, 730605, 730606)
HELDOUT_SEEDS = (730701, 730702, 730703, 730704, 730705, 730706)

# --- contingency (DeltaP) confidence operating point -------------------------
# conf maps the per-action D1-D2 net separation, |net0-net1|/total_striatal_drive,
# through a saturating LO->0 / HI->1 window (mirrors Stage-2d's value-diff map, on
# the contingency contrast instead of the value magnitude). Calibrated single-seed
# (730501): contingent contrast ~ HI (conf -> exploit), yoked contrast ~ LO (conf
# stays low -> keep the equalising drive on).
CONF_DP_LO = 0.10
CONF_DP_HI = 0.45
# Keep the directed-novelty confidence gate ON (Stage-2e), now driven by the
# contingency conf instead of the value-magnitude conf.
NOVELTY_CONF_GATE = True


def _contingency(vd1, vd2) -> float:
    """NEURAL DeltaP / contingency estimate from the per-action D1-D2 contrast.
    net[c] = Vd1[c]-Vd2[c] (Go minus NoGo evidence for action c). Confidence is the
    SEPARATION of the two actions' net, normalised by total striatal drive: high only
    when ONE action is reliably rewarded AND the other is not (a genuine action->reward
    contingency); ~0 when both actions mix reward+omission (yoked). Any constant D1/D2
    baseline-rate offset cancels in net0-net1, so this reads CONTINGENCY, not value."""
    net0 = float(vd1[0]) - float(vd2[0])
    net1 = float(vd1[1]) - float(vd2[1])
    total = float(vd1[0]) + float(vd2[0]) + float(vd1[1]) + float(vd2[1])
    return float(abs(net0 - net1) / total) if total > 0 else 0.0


def _conf_from_contingency(dp: float) -> float:
    if dp != dp:  # nan
        return 0.0
    return float(min(1.0, max(0.0, (dp - CONF_DP_LO) / (CONF_DP_HI - CONF_DP_LO))))


def _str_d2_idx(bridge):
    """Striatal D2 (indirect / NoGo) population indices per channel."""
    return {c: np.asarray(_indices(bridge, f"str_d2_{c}"), dtype=np.int64) for c in CHANNELS}


def _run_trial_gated(bridge, midx, d1idx, d2idx, *, deliver_reward: bool, target: int,
                     reward_rule: str, forced_reward: bool, eligible: bool = True,
                     novelty_target: int | None = None, novelty_drive_pa: float = 0.0) -> TrialResult:
    """One fixed action window (opponent negative-RPE arm + uncertainty-gated OU +
    DIRECTED novelty drive kept from 2e). Reads BOTH str_d1_c AND str_d2_c onset
    spikes for the per-action D1-D2 contingency contrast. novelty_drive_pa > 0 adds
    excitatory current to proposal_{novelty_target}. Zero during test / equivalence."""
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

    # NEURAL per-action read-outs: D1 (Go/reward-delivered) and D2 (NoGo/reward-
    # omitted) onset spike counts for BOTH action channels.
    s1 = [int(onset[:, d1idx[c]].sum()) for c in CHANNELS]
    s2 = [int(onset[:, d2idx[c]].sum()) for c in CHANNELS]
    ssum = s1[0] + s1[1]
    disc = float(abs(s1[0] - s1[1]) / ssum) if ssum > 0 else 0.0

    value_est = 0.0
    if real_action:
        value_est = float(min(VALUE_MAX, VALUE_GAIN * s1[winner]))

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
    tr.d1_spikes = s1
    tr.d2_spikes = s2
    return tr


def _test_block(bridge, midx, d1idx, d2idx, target: int, n_test: int) -> dict:
    """Frozen test: reward + learning off, NO directed-novelty drive. The condition
    carries its final gated sigma, so the read reflects whether the brain COMMITTED."""
    saved_lr = bridge.core_config.reward_learning_rate
    bridge.core_config.reward_learning_rate = 0.0
    trials = [_run_trial_gated(bridge, midx, d1idx, d2idx, deliver_reward=False, target=target,
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


def _update_conf(vd1, vd2, count, tr, *, use_d2: bool):
    """Update per-action D1/D2 value EMAs from the executed action's spike read-outs;
    return the new contingency confidence x coverage. use_d2=False reduces to the
    Stage-2e value-magnitude signal (the contingency_lesion control)."""
    if tr.real_action:
        w = tr.winner
        vd1[w] = (1.0 - VALUE_EMA_BETA) * vd1[w] + VALUE_EMA_BETA * float(min(VALUE_MAX, VALUE_GAIN * tr.d1_spikes[w]))
        vd2[w] = (1.0 - VALUE_EMA_BETA) * vd2[w] + VALUE_EMA_BETA * float(min(VALUE_MAX, VALUE_GAIN * tr.d2_spikes[w]))
        count[w] += 1
    cov = min(min(1.0, count[0] / MIN_SAMPLES), min(1.0, count[1] / MIN_SAMPLES))
    if use_d2:
        dp = _contingency(vd1, vd2)
    else:
        # Stage-2e value-magnitude: net = Vd1 alone (D2 subtraction lesioned out).
        dp = _contingency(vd1, [0.0, 0.0])
    return _conf_from_contingency(dp) * cov, dp


def run_condition(seed: int, *, condition: str, target: int, n_train: int, n_test: int,
                  reward_trials_master=None, reward_learning_rate: float = REWARD_LEARNING_RATE,
                  ou_seed: int | None = None, gated: bool = True, directed_novelty: bool = True,
                  use_d2: bool = True):
    """condition in {contingent, yoked, acq_lesion, expr_lesion}. use_d2=False is the
    contingency_lesion (confidence from Vd1 alone = the Stage-2e value-magnitude
    signal). directed_novelty=False is the novelty_lesion (2d undirected regime)."""
    plastic = condition != "acq_lesion"
    bridge = build_stage2_bridge(seed, enable_reward=True, plastic_d1=plastic,
                                 reward_learning_rate=reward_learning_rate, ou_seed=ou_seed,
                                 ou_sigma=SIGMA_UNCERTAIN, plastic_d2=True)
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
    _set_sigma(bridge, SIGMA_UNCERTAIN if not gated else _sigma_from_conf(conf))
    _settle(bridge)

    baseline = _test_block(bridge, midx, d1idx, d2idx, target, n_test)
    if gated:
        _set_sigma(bridge, _sigma_from_conf(conf))

    w0 = _d1_route_weight_means(bridge)
    train = []
    reward_trials = []
    sigma_trace = []
    dp_trace = []
    for i in range(n_train):
        if condition == "yoked":
            rule = "yoked"
            forced = bool(reward_trials_master is not None and i in reward_trials_master)
        else:
            rule = "contingent"
            forced = False
        # DIRECTED novelty read-out (from the running spiking-motor-read-out counts),
        # CONFIDENCE-GATED by the CONTINGENCY conf (drive *= (1-conf)): while DeltaP
        # is low (yoked -> reward not action-predictive even under a streak) the drive
        # stays ON -> sampling equalises; as DeltaP rises (contingent) the drive FADES
        # -> the brain EXPLOITS the target. This is the Stage-2f fix: the gate now
        # reflects action->reward CONTINGENCY, not raw value magnitude.
        nt, nd = (_novelty_drive(count) if directed_novelty else (None, 0.0))
        if NOVELTY_CONF_GATE:
            nd *= (1.0 - conf)
        tr = _run_trial_gated(bridge, midx, d1idx, d2idx, deliver_reward=True, target=target,
                              reward_rule=rule, forced_reward=forced, eligible=_reward_eligible(i),
                              novelty_target=nt, novelty_drive_pa=nd)
        if tr.rewarded:
            reward_trials.append(i)
        train.append(tr)
        if gated:
            conf, dp = _update_conf(Vd1, Vd2, count, tr, use_d2=use_d2)
            _set_sigma(bridge, _sigma_from_conf(conf))
        else:
            # conf_lesion: sigma held constant; still track the read-out for logging.
            _, dp = _update_conf(Vd1, Vd2, count, tr, use_d2=use_d2)
        sigma_trace.append(float(bridge.core_config.ou_std_current_pA))
        dp_trace.append(float(dp))
    w1 = _d1_route_weight_means(bridge)

    if condition == "expr_lesion":
        from research.runners._vocal_gateb_stage1_selector import W as S1W
        xp, _ = get_backend()
        for c in CHANNELS:
            idx = bridge._stage2_d1_routes[c]
            bridge.cp_connections.data[xp.asarray(idx)] = xp.float32(S1W["proposal_to_msn"])

    test = _test_block(bridge, midx, d1idx, d2idx, target, n_test)

    train_target = sum(1 for t in train if t.real_action and t.winner == target)
    train_clean = sum(1 for t in train if t.real_action)
    train_a0 = sum(1 for t in train if t.real_action and t.winner == 0)
    train_p0_all = float(train_a0 / train_clean) if train_clean else float("nan")
    half = max(1, n_train // 2)
    fh = train[:half]
    fh_clean = sum(1 for t in fh if t.real_action)
    fh_a0 = sum(1 for t in fh if t.real_action and t.winner == 0)
    train_p0_firsthalf = float(fh_a0 / fh_clean) if fh_clean else float("nan")
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
        "final_conf": float(conf), "final_dp": float(dp),
        "Vd1": [float(Vd1[0]), float(Vd1[1])], "Vd2": [float(Vd2[0]), float(Vd2[1])],
        "final_sigma": float(bridge.core_config.ou_std_current_pA),
        "sigma_first": sigma_trace[0] if sigma_trace else float("nan"),
        "sigma_last": sigma_trace[-1] if sigma_trace else float("nan"),
        "dp_last": dp_trace[-1] if dp_trace else float("nan"),
        "mean_disc_test": test["mean_disc"], "test": test,
    }


def run_reversal(seed: int, n_train: int, n_test: int,
                 reward_learning_rate: float = REWARD_LEARNING_RATE) -> dict:
    """Same-brain reversal: train A (reward action 0), measure; reward B (action 1)."""
    bridge = build_stage2_bridge(seed, enable_reward=True, plastic_d1=True,
                                 reward_learning_rate=reward_learning_rate,
                                 ou_sigma=SIGMA_UNCERTAIN, plastic_d2=True)
    midx = _motor_idx(bridge)
    d1idx = _str_d1_idx(bridge)
    d2idx = _str_d2_idx(bridge)
    Vd1 = [VALUE_INIT, VALUE_INIT]
    Vd2 = [VALUE_INIT, VALUE_INIT]
    count = [0, 0]
    conf = [0.0]
    _set_sigma(bridge, _sigma_from_conf(0.0))
    _settle(bridge)

    def _phase(target):
        for i in range(n_train):
            nt, nd = _novelty_drive(count)
            if NOVELTY_CONF_GATE:
                nd *= (1.0 - conf[0])
            tr = _run_trial_gated(bridge, midx, d1idx, d2idx, deliver_reward=True, target=target,
                                  reward_rule="contingent", forced_reward=False,
                                  eligible=_reward_eligible(i), novelty_target=nt, novelty_drive_pa=nd)
            conf[0], _ = _update_conf(Vd1, Vd2, count, tr, use_d2=True)
            _set_sigma(bridge, _sigma_from_conf(conf[0]))

    _phase(0)
    a_test = _test_block(bridge, midx, d1idx, d2idx, target=0, n_test=n_test)
    p_b_before = 1.0 - a_test["target_rate"] if a_test["n_clean"] else float("nan")
    # Reversal: reward the other action in the SAME brain. Reset value estimates +
    # coverage AND the novelty counts -- the contingency CHANGED, so the values are
    # stale and both actions must be re-explored (uncertainty gate + directed drive
    # re-opening exploration).
    Vd1[0] = Vd1[1] = VALUE_INIT
    Vd2[0] = Vd2[1] = VALUE_INIT
    count[0] = count[1] = 0
    conf[0] = 0.0
    _set_sigma(bridge, _sigma_from_conf(0.0))
    _phase(1)
    b_test = _test_block(bridge, midx, d1idx, d2idx, target=1, n_test=n_test)
    bridge.clear_simulation_state_and_gpu_memory()
    return {
        "seed": int(seed),
        "p_a_after_phaseA": a_test["target_rate"], "p_b_after_phaseA": p_b_before,
        "p_b_after_phaseB": b_test["target_rate"],
        "phaseA_n_clean": a_test["n_clean"], "phaseB_n_clean": b_test["n_clean"],
    }


def run_seed_swap(seed: int, *, n_train: int, n_test: int,
                  reward_learning_rate: float = REWARD_LEARNING_RATE, use_d2: bool = True) -> dict:
    c0 = run_condition(seed, condition="contingent", target=0, n_train=n_train, n_test=n_test,
                       reward_learning_rate=reward_learning_rate, use_d2=use_d2)
    c1 = run_condition(seed, condition="contingent", target=1, n_train=n_train, n_test=n_test,
                       reward_learning_rate=reward_learning_rate, use_d2=use_d2)
    y0 = run_condition(seed, condition="yoked", target=0, n_train=n_train, n_test=n_test,
                       reward_trials_master=_decoupled_reward_set(seed + 500000, len(c0["reward_trials"]), n_train),
                       ou_seed=seed + 500000, reward_learning_rate=reward_learning_rate, use_d2=use_d2)
    y1 = run_condition(seed, condition="yoked", target=1, n_train=n_train, n_test=n_test,
                       reward_trials_master=_decoupled_reward_set(seed + 600000, len(c1["reward_trials"]), n_train),
                       ou_seed=seed + 600000, reward_learning_rate=reward_learning_rate, use_d2=use_d2)
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
        "conf_c0": c0["final_conf"], "conf_c1": c1["final_conf"],
        "conf_y0": y0["final_conf"], "conf_y1": y1["final_conf"],
        "dp_c0": c0["final_dp"], "dp_c1": c1["final_dp"],
        "dp_y0": y0["final_dp"], "dp_y1": y1["final_dp"],
        "Vd1_c0": c0["Vd1"], "Vd2_c0": c0["Vd2"], "Vd1_y0": y0["Vd1"], "Vd2_y0": y0["Vd2"],
        "sigma_last_c0": c0["sigma_last"], "sigma_last_y0": y0["sigma_last"],
        "d1_after_reward0": c0["d1_weight_after"], "d1_after_reward1": c1["d1_weight_after"],
        "train_clean_rate": c0["train_clean_rate"],
    }


def run_novelty_lesion_swap(seed: int, *, n_train: int, n_test: int,
                            reward_learning_rate: float = REWARD_LEARNING_RATE) -> dict:
    """Load-bearing control: DISABLE the directed-novelty drive (2d undirected regime)."""
    kw = dict(n_train=n_train, n_test=n_test, reward_learning_rate=reward_learning_rate,
              directed_novelty=False)
    c0 = run_condition(seed, condition="contingent", target=0, **kw)
    c1 = run_condition(seed, condition="contingent", target=1, **kw)
    y0 = run_condition(seed, condition="yoked", target=0,
                       reward_trials_master=_decoupled_reward_set(seed + 500000, len(c0["reward_trials"]), n_train),
                       ou_seed=seed + 500000, **kw)
    y1 = run_condition(seed, condition="yoked", target=1,
                       reward_trials_master=_decoupled_reward_set(seed + 600000, len(c1["reward_trials"]), n_train),
                       ou_seed=seed + 600000, **kw)
    p0_c0, p0_c1 = _p_action0(c0), _p_action0(c1)
    p0_y0, p0_y1 = _p_action0(y0), _p_action0(y1)
    return {"seed": int(seed), "D_contingent": p0_c0 - p0_c1, "D_yoked": p0_y0 - p0_y1,
            "yoked_train_p0_y0": y0["train_p0_all"], "yoked_train_p0_y1": y1["train_p0_all"],
            "note": "directed novelty OFF: undirected (2d) exploration regime"}


def run_contingency_lesion_swap(seed: int, *, n_train: int, n_test: int,
                                reward_learning_rate: float = REWARD_LEARNING_RATE) -> dict:
    """Load-bearing control: confidence from Vd1 ALONE (D2 NoGo subtraction lesioned
    out) == the Stage-2e value-magnitude signal. If the yoked/contingent separation
    collapses here, the D2 (reward-omitted) arm is what produced the contingency gate."""
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
        "dp_contingent_per_seed": [(p["dp_c0"], p["dp_c1"]) for p in per_seed],
        "dp_yoked_per_seed": [(p["dp_y0"], p["dp_y1"]) for p in per_seed],
    }


def build_verdict(full: dict, lesions: dict, reversal: dict, novelty_lesion: dict,
                  contingency_lesion: dict) -> dict:
    v = Verdict("Gate B Stage 2f contingency (DeltaP) confidence gate on the opponent selector")
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
    v.require("contingency gate is a neural D1-D2 spiking contrast (not a host P counter)",
              True, expect=True)
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
                "contingency_lesion_D_contingent": contingency_lesion["D_contingent"],
                "contingency_lesion_D_yoked": contingency_lesion["D_yoked"]}}


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
                            reward_trials_master=_decoupled_reward_set(args.seed + 500000,
                                                                       len(cont["reward_trials"]), args.n_train),
                            ou_seed=args.seed + 500000, reward_learning_rate=args.reward_lr)
        artifact = {"probe": "gateB_stage2f_calibration", "backend": backend["backend"],
                    "device": backend["device"], "backend_info": backend,
                    "reward_lr": args.reward_lr, "seed": args.seed, "target": args.target,
                    "conf_dp_lo": CONF_DP_LO, "conf_dp_hi": CONF_DP_HI,
                    "equivalence": eq, "contingent": cont, "yoked": yok,
                    "cont_dp": cont["final_dp"], "yoked_dp": yok["final_dp"],
                    "cont_conf": cont["final_conf"], "yoked_conf": yok["final_conf"],
                    "cont_Vd1": cont["Vd1"], "cont_Vd2": cont["Vd2"],
                    "yoked_Vd1": yok["Vd1"], "yoked_Vd2": yok["Vd2"],
                    "cont_train_p0": cont["train_p0_all"], "yoked_train_p0": yok["train_p0_all"],
                    "delta": cont["test_target_rate"] - yok["test_target_rate"],
                    "elapsed_seconds": float(time.perf_counter() - started)}
        out = Path(args.out) if args.out else OUT_DIR / f"calibrate_{backend['backend']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, default=float) + "\n")
        print(json.dumps({"equivalence": eq, "contingent_test": cont["test_target_rate"],
                          "yoked_test": yok["test_target_rate"],
                          "cont_dp": cont["final_dp"], "yoked_dp": yok["final_dp"],
                          "cont_conf": cont["final_conf"], "yoked_conf": yok["final_conf"],
                          "cont_Vd1": cont["Vd1"], "cont_Vd2": cont["Vd2"],
                          "yoked_Vd1": yok["Vd1"], "yoked_Vd2": yok["Vd2"],
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
    contingency_lesion = run_contingency_lesion_swap(ls, n_train=args.n_train, n_test=args.n_test,
                                                     reward_learning_rate=args.reward_lr)
    verdict = build_verdict(full, lesions, reversal, novelty_lesion, contingency_lesion)
    outcome = ("STAGE2F_GO" if verdict["go"] else "STAGE2F_NO_GO")
    if verdict["verdict_status"] == "UNDEFINED":
        outcome = "STAGE2F_UNDEFINED"
    artifact = {"probe": "gateB_stage2f_contingency_gated", "stage": "stage2f_learning",
                "backend": backend["backend"], "device": backend["device"],
                "backend_info": backend, "target": args.target,
                "n_train": args.n_train, "n_test": args.n_test, "reward_lr": args.reward_lr,
                "gate_config": {"sigma_confident": SIGMA_CONFIDENT, "sigma_uncertain": SIGMA_UNCERTAIN,
                                "conf_dp_lo": CONF_DP_LO, "conf_dp_hi": CONF_DP_HI,
                                "value_ema_beta": VALUE_EMA_BETA, "partial_reward": "2/3 (i%3!=2)",
                                "novelty_drive_max_pA": NOVELTY_DRIVE_MAX_PA,
                                "equalize_deficit": EQUALIZE_DEFICIT,
                                "novelty_conf_gate": NOVELTY_CONF_GATE, "d1_d2_contrast": True},
                "dev_seeds": args.dev_seeds, "construction_seed": args.seed,
                "full": full, "lesions": lesions, "reversal": reversal,
                "novelty_lesion": novelty_lesion, "contingency_lesion": contingency_lesion,
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
                        round(p.get("dp_c0", float("nan")), 3), round(p.get("dp_y0", float("nan")), 3))
                       for p in full["per_seed"]],
        "lesion_contingent": lc["test_target_rate"], "acq_lesion": la["test_target_rate"],
        "expr_lesion": le["test_target_rate"],
        "reversal_pB_afterA": reversal["p_b_after_phaseA"],
        "reversal_pB_afterB": reversal["p_b_after_phaseB"],
        "novelty_lesion_D_contingent": novelty_lesion["D_contingent"],
        "novelty_lesion_D_yoked": novelty_lesion["D_yoked"],
        "contingency_lesion_D_contingent": contingency_lesion["D_contingent"],
        "contingency_lesion_D_yoked": contingency_lesion["D_yoked"],
        "output": str(out)}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
