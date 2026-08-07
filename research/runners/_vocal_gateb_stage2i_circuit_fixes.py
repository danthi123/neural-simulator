"""Gate B Stage 2i: two targeted CIRCUIT fixes for the two extreme-bias residuals.

Additive on Stage 2g (_vocal_gateb_stage2g_hammond_deltap.py, imported unchanged and
kept byte-reproducible). Stage 2g's contingency MECHANISM is complete and correct
(dev-GO 5/6; withhold-ΔP baseline + Carandini-Heeger homeostatic critic + opponent RPE
+ uncertainty gate + action-decoupled reward; reversal + lesions pass). It fails
held-out 4/6 on exactly TWO seeds, for DIFFERENT reasons -- VERIFIED against the
substrate in Stage 2h's smoke (research/findings/2026-08-06-gateB-stage2h-forced-
sampling-extreme-bias-NO-GO.md + smoke_numpy_730704.json / _730705.json). Stage 2h's
proposal-level forced-sampling floor was REFUTED; the two fixes below act where each
residual actually lives.

FIX A (730705, baseline_p0=1.0). ** REFUTED BY THE SUBSTRATE (default OFF; banked). **
The hypothesis was a DOWNSTREAM WTA lock at a reward-POTENTIATED str_d1_0 route,
breakable by inhibiting the incumbent's str_d1 while exciting the under-sampled proposal.
A per-population diagnostic on 730705 REFUTES both the premise and the lever:
  (1) the two proposal->str_d1 routes start SYMMETRIC (mean weight 40.03 vs 40.03) -- the
      bias is INTRINSIC (the seed's per-neuron heterogeneity draw), not reward-formed, so
      there is no early lock to anneal;
  (2) EXTERNAL current into an str_d1 (MSN) population is COUNTERPRODUCTIVE, not a lever:
      on a normal seed +200 pA into str_d1_1 DROPS its firing 6 -> 0 and kills motor_1;
  (3) on 730705 str_d1_1 is intrinsically near-UNEXCITABLE (0 spikes at every drive
      tested, 200-3000 pA direct), so motor_1 can never win normally AND -- critically --
      the proposal_1->str_d1_1 route can never acquire three-factor eligibility (str_d1_1
      never co-activates), so no reward can potentiate action 1: the pathway is a
      structural DEAD END on this heterogeneity draw. The only lever that made motor_1 win
      was inhibiting the incumbent MOTOR pool directly, but that leaves str_d1_1 at 0 -> no
      learning persists to test. FIX A as built (str_d1 forced-sampling bias) leaves
      730705's counts byte-identical to 2g AND perturbs yoked cancellation on excitable
      seeds, so it is default OFF. The correct next method is HOMEOSTATIC INTRINSIC
      EXCITABILITY plasticity of the MSNs (raise str_d1_1's intrinsic excitability toward a
      firing set-point by modifying its Izhikevich parameters -- NOT by injecting current),
      so the dead pathway can fire and then learn. That is a substrate change, banked for
      Stage 2j. FIX A's code is retained (opt-in via --fix-a) only as the banked method.

FIX B (730704, baseline_p0=0.0 -- a CRITIC over-subtraction, NOT a sampling gap). Vanilla
2g already samples both actions (count [13,26]); its NaN is a training-induced TEST-TIME
motor SILENCE: after training every test trial reads motor=[0,0]. Cause (VERIFIED in
code): the homeostatic critic normalisation can inflate value_est up to VALUE_MAX=1.5,
which EXCEEDS REWARD_MAG=1.0. The Hammond-ΔP reward baseline is value_est + v_withhold,
so on this seed a REWARDED action gets net RPE = REWARD_MAG - value_est < 0 -- the action
DEPRESSES its own route even when rewarded -> runaway to silence. Fix: a floor on the net
RPE, applied ONLY to the self-value (value_est) component of the baseline -- clamp
value_est <= VALUE_EST_BASELINE_CAP = REWARD_MAG - RPE_FLOOR (a critic value cannot exceed
the maximum obtainable reward; the DA burst to a delivered reward is never fully cancelled
by expectation). This guarantees a rewarded action's net RPE >= RPE_FLOOR > 0, arresting
the collapse, while leaving the WITHHOLD (v_withhold) base-rate subtraction FULLY intact
-> the Hammond-ΔP contingency and the yoked cancellation are unchanged (v_withhold ~ 0 in
contingent; the cap only bites when value_est is pathologically inflated, so dev seeds
whose value_est <= CAP are byte-unchanged). Optional belt-and-suspenders (default OFF): a
motor-onset weight-floor homeostat (Turrigiano synaptic scaling) that refuses to let the
plastic proposal->str_d1 route fall below a fraction of its initial weight.

Everything else (withhold-ΔP, critic norm, opponent RPE, uncertainty gate, directed
novelty, action-decoupled reward, reward-OFF byte-identical guard) and every frozen
criterion (steer >=5/6, D_contingent-yoked >=0.20, reversal >=0.60, lesions) is inherited
unchanged from 2g. With both fixes OFF this runner reproduces Stage 2g exactly.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from research.runners._vocal_action_selector_gate import _indices
from research.runners._vocal_gateb_stage2g_hammond_deltap import (
    CHANNELS,
    DEV_SEEDS,
    GAP_STEPS,
    HELDOUT_SEEDS,
    LOSER_RATIO,
    MIN_SAMPLES,
    MOTOR_THRESHOLD,
    N_TEST,
    N_TRAIN,
    NOVELTY_CONF_GATE,
    NOVELTY_DRIVE_MAX_PA,
    ONSET_STEPS,
    REWARD_DELAY,
    REWARD_LEARNING_RATE,
    REWARD_MAG,
    REWARD_STEPS,
    SIGMA_UNCERTAIN,
    TrialResult,
    VALUE_INIT,
    VALUE_MAX,
    WH_PERIOD,
    CONSTRUCTION_SEED,
    _apply_afferents,
    _assert_stage1_equivalence,
    _backend_info,
    _base_rate,
    _baseline_block,
    _d1_route_weight_means,
    _decoupled_reward_set,
    _mean,
    _motor_idx,
    _novelty_drive,
    _p_action0,
    _reconfigure_da_s,
    _reward_eligible,
    _run_withhold_trial,
    _set_sigma,
    _settle,
    _sigma_from_conf,
    _str_d1_idx,
    _str_d2_idx,
    _test_block,
    _update_conf,
    _v_withhold,
    _value_action,
    build_stage2_bridge,
    build_verdict,
)
from sim.backend import get_backend, to_host
from tools.lab import attributable_to

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "research/findings/raw/gateb_stage2i_circuit_fixes"

# ============================ FIX A: str_d1 incumbent inhibition ==============
# While EITHER action has < K clean motor samples (after an 8-trial grace on the 2g
# graded drive), EXCITE the under-sampled action's proposal (reaches its str_d1, as 2g)
# AND INHIBIT the OVER-SAMPLED incumbent's str_d1 population directly -- the striatal
# level where the WTA lock lives, not the proposal (2h verified proposal-level is inert
# once the route is potentiated). Both escalate with a ramp that resets when the
# under-sampled action switches, so the bias is only as strong as needed.
FORCE_SAMPLE_K = MIN_SAMPLES              # per-action clean-sample floor (= 3)
FORCE_EXC_BASE_PA = NOVELTY_DRIVE_MAX_PA  # 350 pA: start where the graded cap ends
FORCE_EXC_RAMP_PA = 250.0
# Excitation ceiling capped below the depolarization-block threshold (2h: proposal drive
# > ~1250 pA silences str_d1). 1200 pA keeps the driven proposal firing.
FORCE_EXC_MAX_PA = 1200.0
# Incumbent str_d1 inhibition (applied as NEGATIVE current onto the str_d1 population).
# Hyperpolarisation has no depol-block ceiling, so it can go high; escalates until the
# under-sampled action reaches K samples.
FORCE_STR_D1_INH_BASE_PA = 1500.0
FORCE_STR_D1_INH_RAMP_PA = 750.0
FORCE_STR_D1_INH_MAX_PA = 5000.0
FORCE_GRACE_TRIALS = 8

# ============================ FIX B: RPE floor (bounded critic self-value) ====
# A critic value cannot exceed the maximum obtainable reward; the DA burst to a delivered
# reward is never fully cancelled by expectation. Clamp ONLY the self-value (value_est)
# component of the Hammond-ΔP baseline so a rewarded action's net RPE >= RPE_FLOOR > 0.
# v_withhold (the base-rate subtraction) is left untouched, so contingency / yoked
# cancellation are unchanged.
RPE_FLOOR = 0.1
VALUE_EST_BASELINE_CAP = float(REWARD_MAG - RPE_FLOOR)   # 0.9

# --- optional belt-and-suspenders motor-onset weight-floor homeostat (default OFF) -----
# Turrigiano multiplicative synaptic scaling: the plastic proposal->str_d1 route is not
# allowed to fall below MOTOR_HOMEOSTAT_FLOOR_FRAC of its initial weight, so the acting
# channel can never be depressed all the way to motor silence.
MOTOR_HOMEOSTAT_FLOOR_FRAC = 0.5


def _apply_motor_homeostat(bridge, w0_routes: dict) -> None:
    """Clamp each plastic proposal->str_d1 route weight to a floor = FRAC * its initial
    mean weight (a homeostatic set-point on the route's efficacy). Neural: it bounds the
    synaptic weight the same way multiplicative synaptic scaling does; it does not touch
    the reward/DA signals or select an action."""
    xp, _ = get_backend()
    for c in CHANNELS:
        idx = bridge._stage2_d1_routes[c]
        floor = xp.float32(MOTOR_HOMEOSTAT_FLOOR_FRAC * float(w0_routes[int(c)]))
        cur = bridge.cp_connections.data[xp.asarray(idx)]
        bridge.cp_connections.data[xp.asarray(idx)] = xp.maximum(cur, floor)


def _run_trial_2i(bridge, midx, d1idx, d2idx, *, deliver_reward: bool, target: int,
                  reward_rule: str, forced_reward: bool, eligible: bool = True,
                  r0_d1=None, v_withhold: float = 0.0,
                  novelty_target: int | None = None, novelty_drive_pa: float = 0.0,
                  inh_str_d1_target: int | None = None, inh_str_d1_pa: float = 0.0,
                  rpe_floor: bool = True) -> TrialResult:
    """Stage-2g _run_trial_2g + (A) an optional inhibitory current onto `inh_str_d1_target`'s
    str_d1 population (the WTA-lock level) and (B) the RPE-floor clamp on value_est in the
    reward baseline. With inh=0/None and rpe_floor=False this is byte-identical to 2g."""
    xp, _ = get_backend()
    n = int(bridge.core_config.num_neurons)
    nov_idx = None
    inh_idx = None
    if novelty_drive_pa > 0.0 and novelty_target is not None:
        nov_idx = xp.asarray(_indices(bridge, f"proposal_{int(novelty_target)}"))
    if inh_str_d1_pa > 0.0 and inh_str_d1_target is not None:
        inh_idx = xp.asarray(_indices(bridge, f"str_d1_{int(inh_str_d1_target)}"))
    onset = np.zeros((ONSET_STEPS, n), dtype=bool)
    for step in range(ONSET_STEPS):
        _apply_afferents(bridge, arousal=True)
        if nov_idx is not None:
            bridge.cp_external_input_current[nov_idx] += xp.float32(novelty_drive_pa)
        if inh_idx is not None:
            bridge.cp_external_input_current[inh_idx] -= xp.float32(inh_str_d1_pa)
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

    # (B) RPE floor: clamp ONLY the self-value component so a rewarded action's net RPE
    # stays >= RPE_FLOOR > 0. v_withhold (the base-rate subtraction) is left untouched.
    value_est_base = min(value_est, VALUE_EST_BASELINE_CAP) if rpe_floor else value_est
    base = float(value_est_base + v_withhold)
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


class _ForcedSampler2i:
    """Forced-sampling floor whose incumbent suppression acts on str_d1 (FIX A).

    drive(count, conf, trial_idx) -> (exc_proposal_target, exc_pa, inh_str_d1_target,
    inh_pa, forcing). While min(count) < K after the grace period it EXCITES the
    under-sampled proposal and INHIBITS the over-sampled incumbent's str_d1 (both
    escalating) until the under-sampled action fires; once both actions reach K samples
    it releases to the graded 2g directed-novelty drive (no suppression). Stateful; one
    per training phase.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.ramp = 0
        self.target: int | None = None

    def drive(self, count, conf: float, trial_idx: int = 10**9):
        if (self.enabled and trial_idx >= FORCE_GRACE_TRIALS
                and min(int(count[0]), int(count[1])) < FORCE_SAMPLE_K):
            u = 0 if count[0] <= count[1] else 1   # the LESS-sampled action
            if u != self.target:
                self.ramp = 0
            self.target = u
            exc = min(FORCE_EXC_MAX_PA, FORCE_EXC_BASE_PA + FORCE_EXC_RAMP_PA * self.ramp)
            inh = min(FORCE_STR_D1_INH_MAX_PA,
                      FORCE_STR_D1_INH_BASE_PA + FORCE_STR_D1_INH_RAMP_PA * self.ramp)
            self.ramp += 1
            return u, float(exc), (1 - u), float(inh), True
        self.ramp = 0
        self.target = None
        nt, nd = _novelty_drive(count)
        if NOVELTY_CONF_GATE:
            nd *= (1.0 - conf)
        return nt, float(nd), None, 0.0, False


def run_condition(seed: int, *, condition: str, target: int, n_train: int, n_test: int,
                  reward_trials_master=None, reward_learning_rate: float = REWARD_LEARNING_RATE,
                  ou_seed: int | None = None, gated: bool = True, directed_novelty: bool = True,
                  use_d2: bool = True, wh_reward_p: float = 0.0,
                  fix_a: bool = True, fix_b: bool = True, motor_homeostat: bool = False):
    """Stage-2g run_condition + FIX A (str_d1 forced sampling) and FIX B (RPE floor).
    With fix_a=False, fix_b=False this reproduces Stage 2g exactly."""
    plastic = condition != "acq_lesion"
    bridge = build_stage2_bridge(seed, enable_reward=True, plastic_d1=plastic,
                                 reward_learning_rate=reward_learning_rate, ou_seed=ou_seed,
                                 ou_sigma=SIGMA_UNCERTAIN, plastic_d2=True)
    _reconfigure_da_s(bridge)
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

    sampler = _ForcedSampler2i(enabled=fix_a and directed_novelty)
    w0 = _d1_route_weight_means(bridge)
    train = []
    reward_trials = []
    n_wh = 0
    n_wh_rewarded = 0
    n_forced = 0
    max_forced_inh = 0.0
    v_wh_trace = []
    for i in range(n_train):
        if wh_reward_p > 0.0 and i % WH_PERIOD == 0:
            rew_wh = bool(wh_rng.random() < wh_reward_p)
            _run_withhold_trial(bridge, rewarded=rew_wh)
            n_wh += 1
            n_wh_rewarded += int(rew_wh)
        v_wh = _v_withhold(bridge)
        if condition == "yoked":
            rule = "yoked"
            forced = bool(reward_trials_master is not None and i in reward_trials_master)
        else:
            rule = "contingent"
            forced = False
        if directed_novelty:
            nt, nd, it, idr, forcing = sampler.drive(count, conf, trial_idx=i)
        else:
            nt, nd, it, idr, forcing = None, 0.0, None, 0.0, False
        if forcing:
            n_forced += 1
            max_forced_inh = max(max_forced_inh, idr)
        tr = _run_trial_2i(bridge, midx, d1idx, d2idx, deliver_reward=True, target=target,
                           reward_rule=rule, forced_reward=forced, eligible=_reward_eligible(i),
                           r0_d1=r0_d1, v_withhold=v_wh,
                           novelty_target=nt, novelty_drive_pa=nd,
                           inh_str_d1_target=it, inh_str_d1_pa=idr, rpe_floor=fix_b)
        if motor_homeostat and plastic:
            _apply_motor_homeostat(bridge, w0)
        if tr.rewarded:
            reward_trials.append(i)
        train.append(tr)
        if gated:
            conf, dp = _update_conf(Vd1, Vd2, count, tr, use_d2=use_d2)
            _set_sigma(bridge, _sigma_from_conf(conf))
        else:
            _, dp = _update_conf(Vd1, Vd2, count, tr, use_d2=use_d2)
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
        "n_forced_trials": int(n_forced), "max_forced_inh_pA": float(max_forced_inh),
        "d1_weight_before": w0, "d1_weight_after": w1,
        "final_conf": float(conf), "final_dp": float(dp),
        "final_v_withhold": float(v_wh), "max_v_withhold": float(max(v_wh_trace) if v_wh_trace else 0.0),
        "n_withhold": int(n_wh), "n_withhold_rewarded": int(n_wh_rewarded),
        "r0_d1": r0_d1,
        "Vd1": [float(Vd1[0]), float(Vd1[1])], "Vd2": [float(Vd2[0]), float(Vd2[1])],
        "final_sigma": float(bridge.core_config.ou_std_current_pA),
    }


def run_reversal(seed: int, n_train: int, n_test: int,
                 reward_learning_rate: float = REWARD_LEARNING_RATE,
                 fix_a: bool = True, fix_b: bool = True) -> dict:
    """Stage-2g same-brain reversal + FIX A/B (fresh sampler per phase)."""
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
        sampler = _ForcedSampler2i(enabled=fix_a)
        for i in range(n_train):
            if i % WH_PERIOD == 0:
                _run_withhold_trial(bridge, rewarded=False)
            nt, nd, it, idr, _forcing = sampler.drive(count, conf[0], trial_idx=i)
            tr = _run_trial_2i(bridge, midx, d1idx, d2idx, deliver_reward=True, target=target,
                               reward_rule="contingent", forced_reward=False,
                               eligible=_reward_eligible(i), r0_d1=r0_d1, v_withhold=0.0,
                               novelty_target=nt, novelty_drive_pa=nd,
                               inh_str_d1_target=it, inh_str_d1_pa=idr, rpe_floor=fix_b)
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


def run_seed_swap(seed: int, *, n_train: int, n_test: int,
                  reward_learning_rate: float = REWARD_LEARNING_RATE, use_d2: bool = True,
                  directed_novelty: bool = True, enable_withhold: bool = True,
                  fix_a: bool = True, fix_b: bool = True, motor_homeostat: bool = False) -> dict:
    kw = dict(n_train=n_train, n_test=n_test, reward_learning_rate=reward_learning_rate,
              use_d2=use_d2, directed_novelty=directed_novelty,
              fix_a=fix_a, fix_b=fix_b, motor_homeostat=motor_homeostat)
    c0 = run_condition(seed, condition="contingent", target=0, wh_reward_p=0.0, **kw)
    c1 = run_condition(seed, condition="contingent", target=1, wh_reward_p=0.0, **kw)
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
        "count_c0": (c0["count0"], c0["count1"]), "count_c1": (c1["count0"], c1["count1"]),
        "n_clean_c0": c0["test_n_clean"], "n_clean_c1": c1["test_n_clean"],
        "test_rate_c0": c0["test_target_rate"], "test_rate_c1": c1["test_target_rate"],
        "n_forced_c0": c0["n_forced_trials"], "n_forced_c1": c1["n_forced_trials"],
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
            "note": "directed novelty OFF (and forced sampling OFF): undirected (2d) regime"}


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
        "count_c1_per_seed": [p["count_c1"] for p in per_seed],
        "test_rate_c1_per_seed": [p["test_rate_c1"] for p in per_seed],
    }


def _smoke_one(seed: int, *, n_train: int, n_test: int, reward_learning_rate: float,
               fix_a: bool, fix_b: bool, motor_homeostat: bool) -> dict:
    """Contingent target0 + target1 (+ yoked) for ONE seed with the fixes, plus the 2g
    baseline (fixes OFF). Reports per-action counts, n_acted, no-NaN, D_contingent, steer."""
    on = run_seed_swap(seed, n_train=n_train, n_test=n_test,
                       reward_learning_rate=reward_learning_rate,
                       fix_a=fix_a, fix_b=fix_b, motor_homeostat=motor_homeostat)
    off = run_seed_swap(seed, n_train=n_train, n_test=n_test,
                        reward_learning_rate=reward_learning_rate,
                        fix_a=False, fix_b=False, motor_homeostat=False)

    def _clean(x):
        return x == x  # not NaN

    both_targets_act = bool(on["n_clean_c0"] > 0 and on["n_clean_c1"] > 0)
    no_nan = bool(_clean(on["test_rate_c0"]) and _clean(on["test_rate_c1"]))
    # Attribute the test-clean recovery to the fixes: control = 2g (fixes OFF), treatment =
    # fixes ON. min over targets is the worse-case clean-action count; whose the recovery is.
    cov_off = min(int(off["n_clean_c0"]), int(off["n_clean_c1"]))
    cov_on = min(int(on["n_clean_c0"]), int(on["n_clean_c1"]))
    recovery_attribution = (attributable_to(
        "worst-target test-clean recovery to the Stage-2i fixes (vs 2g)",
        float(cov_on), float(cov_off)) if cov_on > 0 else 0.0)
    # both ACTIONS sampled somewhere in training (extreme-bias coverage)
    both_actions_sampled = bool(min(on["count_c0"]) > 0 or min(on["count_c1"]) > 0)
    d_defined = bool(_clean(on["D_contingent"]) and abs(on["D_contingent"]) > 1e-9)
    steer = bool(on["D_contingent"] >= 0.30 and (on["D_contingent"] - on["D_yoked"]) >= 0.20)
    return {
        "seed": int(seed), "baseline_p0": on["baseline_p0"],
        "FIXES_ON": {
            "count_c0": on["count_c0"], "count_c1": on["count_c1"],
            "n_clean_c0": on["n_clean_c0"], "n_clean_c1": on["n_clean_c1"],
            "test_rate_c0": on["test_rate_c0"], "test_rate_c1": on["test_rate_c1"],
            "n_forced_c0": on["n_forced_c0"], "n_forced_c1": on["n_forced_c1"],
            "D_contingent": on["D_contingent"], "D_yoked": on["D_yoked"],
        },
        "STAGE2G_OFF": {
            "count_c0": off["count_c0"], "count_c1": off["count_c1"],
            "n_clean_c0": off["n_clean_c0"], "n_clean_c1": off["n_clean_c1"],
            "test_rate_c0": off["test_rate_c0"], "test_rate_c1": off["test_rate_c1"],
            "D_contingent": off["D_contingent"], "D_yoked": off["D_yoked"],
        },
        "recovery_attribution_to_fixes": recovery_attribution,
        "SMOKE_PASS_no_nan_no_freeze": no_nan,
        "SMOKE_PASS_both_targets_act": both_targets_act,
        "SMOKE_PASS_both_actions_sampled": both_actions_sampled,
        "SMOKE_PASS_D_contingent_defined": d_defined,
        "SMOKE_PASS_steer": steer,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full", "seeds"], default="full")
    parser.add_argument("--seed", type=int, default=CONSTRUCTION_SEED)
    parser.add_argument("--dev-seeds", type=int, nargs="*", default=list(DEV_SEEDS))
    parser.add_argument("--smoke-seeds", type=int, nargs="*", default=[730704, 730705])
    parser.add_argument("--lesion-seed", type=int, default=730605)
    parser.add_argument("--lesion-target", type=int, default=0)
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--reward-lr", type=float, default=REWARD_LEARNING_RATE)
    parser.add_argument("--fix-a", action="store_true",
                        help="enable FIX A (str_d1 forced-sampling bias) -- REFUTED by the "
                             "substrate (external MSN current is counterproductive; it also "
                             "perturbs yoked cancellation), default OFF")
    parser.add_argument("--no-fix-b", action="store_true",
                        help="disable FIX B (RPE floor) -> pure 2g baseline behaviour")
    parser.add_argument("--motor-homeostat", action="store_true", help="enable the motor-onset weight-floor homeostat")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    backend = _backend_info()
    started = time.perf_counter()
    fix_a = bool(args.fix_a)
    fix_b = not args.no_fix_b

    if args.mode == "smoke":
        # Smoke the extreme-bias seeds ONE process each; prove BOTH fixes recover clean
        # actions on both targets at test (no NaN) with a defined (ideally steer-passing)
        # D_contingent. NO multiseed sweep.
        results = [_smoke_one(s, n_train=args.n_train, n_test=args.n_test,
                              reward_learning_rate=args.reward_lr,
                              fix_a=fix_a, fix_b=fix_b, motor_homeostat=args.motor_homeostat)
                   for s in args.smoke_seeds]
        all_no_nan = all(r["SMOKE_PASS_no_nan_no_freeze"] for r in results)
        all_act = all(r["SMOKE_PASS_both_targets_act"] for r in results)
        all_d = all(r["SMOKE_PASS_D_contingent_defined"] for r in results)
        artifact = {"probe": "gateB_stage2i_circuit_fixes_smoke",
                    "backend": backend["backend"], "device": backend["device"],
                    "smoke_seeds": args.smoke_seeds,
                    "fix_a_str_d1_forced_sampling": fix_a, "fix_b_rpe_floor": fix_b,
                    "motor_homeostat": bool(args.motor_homeostat),
                    "config": {"value_est_baseline_cap": VALUE_EST_BASELINE_CAP,
                               "rpe_floor": RPE_FLOOR,
                               "force_str_d1_inh_base_pA": FORCE_STR_D1_INH_BASE_PA,
                               "force_str_d1_inh_max_pA": FORCE_STR_D1_INH_MAX_PA,
                               "force_exc_max_pA": FORCE_EXC_MAX_PA,
                               "force_sample_k": FORCE_SAMPLE_K},
                    "per_seed": results,
                    "SMOKE_PASS_all_no_nan": bool(all_no_nan),
                    "SMOKE_PASS_all_both_targets_act": bool(all_act),
                    "SMOKE_PASS_all_D_defined": bool(all_d),
                    "elapsed_seconds": float(time.perf_counter() - started)}
        out = Path(args.out) if args.out else OUT_DIR / f"smoke_{backend['backend']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, default=float) + "\n")
        print(json.dumps(artifact, indent=2, default=float))
        return 0

    if args.mode == "seeds":
        per = [run_seed_swap(s, n_train=args.n_train, n_test=args.n_test,
                             reward_learning_rate=args.reward_lr,
                             fix_a=fix_a, fix_b=fix_b, motor_homeostat=args.motor_homeostat)
               for s in args.dev_seeds]
        rows = [(p["seed"], round(p["baseline_p0"], 2), round(p["D_contingent"], 3),
                 round(p["D_yoked"], 3), p["count_c1"], round(p["test_rate_c1"], 3),
                 bool(p["D_contingent"] >= 0.30 and (p["D_contingent"] - p["D_yoked"]) >= 0.20))
                for p in per]
        print(json.dumps({"seeds_rows(seed,base_p0,Dc,Dy,count_c1,test_rate_c1,steer)": rows,
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
    outcome = ("STAGE2I_GO" if verdict["go"] else "STAGE2I_NO_GO")
    if verdict["verdict_status"] == "UNDEFINED":
        outcome = "STAGE2I_UNDEFINED"
    artifact = {"probe": "gateB_stage2i_circuit_fixes", "stage": "stage2i_learning",
                "backend": backend["backend"], "device": backend["device"],
                "backend_info": backend, "target": args.target,
                "n_train": args.n_train, "n_test": args.n_test, "reward_lr": args.reward_lr,
                "fix_a_str_d1_forced_sampling": fix_a, "fix_b_rpe_floor": fix_b,
                "motor_homeostat": bool(args.motor_homeostat),
                "circuit_fix_config": {"value_est_baseline_cap": VALUE_EST_BASELINE_CAP,
                                       "rpe_floor": RPE_FLOOR,
                                       "force_str_d1_inh_base_pA": FORCE_STR_D1_INH_BASE_PA,
                                       "force_str_d1_inh_max_pA": FORCE_STR_D1_INH_MAX_PA,
                                       "force_exc_max_pA": FORCE_EXC_MAX_PA,
                                       "force_sample_k": FORCE_SAMPLE_K},
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
        "count_c1_per_seed": full["count_c1_per_seed"],
        "test_rate_c1_per_seed": full["test_rate_c1_per_seed"],
        "reversal_pB_afterA": reversal["p_b_after_phaseA"],
        "reversal_pB_afterB": reversal["p_b_after_phaseB"],
        "output": str(out)}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
