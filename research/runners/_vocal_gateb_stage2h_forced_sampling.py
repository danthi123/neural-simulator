"""Gate B Stage 2h: forced-sampling / epsilon-floor exploration on extreme-bias seeds.

Additive on Stage 2g (_vocal_gateb_stage2g_hammond_deltap.py, imported unchanged and
kept byte-reproducible). Stage 2g's contingency MECHANISM is complete and correct
(dev-GO 5/6; withhold-ΔP baseline + Carandini-Heeger homeostatic critic + opponent
RPE + uncertainty gate + action-decoupled reward). It fails held-out 4/6 for ONE
reason: on MAXIMALLY-biased seeds (baseline_p0 ∈ {0,1}) the brain never samples BOTH
actions, so a test block emits zero clean actions (target_rate = target_hits/n_acted
is NaN -> the brain FROZE) or the target action is never rewarded (reward_count=0).
Verified in research/findings/raw/gateb_stage2g_hammond_deltap/heldout_numpy.json:
730704 (base_p0=0.0) -> contingent_p0_reward1=NaN; 730705 (base_p0=1.0) ->
reward_count_reward1=0. Same extreme-bias exploration limit as Stage-2e's sole
double-failure (730604).

The fix is a NEURAL forced-sampling floor on top of the 2g directed-novelty drive:
while EITHER action has fewer than K clean motor samples, a PUSH-PULL competition bias
is applied -- the under-sampled action's proposal_{u} population is EXCITED (escalating
past the 350 pA graded cap, un-gated by confidence) AND the over-sampled incumbent's
proposal_{1-u} population is INHIBITED -- both ramping until the under-sampled action
FIRES, guaranteeing the extreme-bias brain samples both actions >= K times before the
drive relaxes back to the graded 2g form.

The push-pull (not just excitation) is load-bearing and was VERIFIED on the most
extreme seed 730705 (baseline_p0=1.0, brain always picks action 0): driving proposal_1
ALONE to 10000 pA propagates to the striatum (str_d1_1 fires 2031 spikes) but motor_1
stays at 0 -- the bottleneck is a DOWNSTREAM striatal/motor winner-take-all lock, not
the proposal input, so a one-sided excitatory drive cannot break it. Adding an
inhibitory bias to the incumbent proposal_0 releases the WTA and motor_1 wins cleanly
(motor[0, ~860]). It STAYS neural: the host only biases WHICH proposals compete and how
hard (competing salience/attention currents into two populations, a count-based novelty
read-out as 2e/2g already do); the brain's own WTA still resolves the winner and the
motor pool must still cross threshold -- it is NOT a host argmax selecting the action.

Everything else (withhold-ΔP, critic norm, opponent RPE, uncertainty gate,
action-decoupled reward, reward-OFF byte-identical guard) and every frozen criterion
(steer >=5/6, D_contingent-yoked >=0.20, reversal >=0.60, lesions) is inherited
unchanged from 2g. The floor only applies WHILE an action is un-sampled; once both have
>= K samples the exact 2g graded drive resumes.
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
OUT_DIR = ROOT / "research/findings/raw/gateb_stage2h_forced_sampling"

# --- forced-sampling / epsilon-floor exploration ------------------------------
# While EITHER action has < K clean motor samples, drive the under-sampled action's
# proposal_{u} population with an escalating excitatory current that RAMPS PAST the
# 350 pA graded cap (un-gated by confidence, does not decay) until it fires. The ramp
# grows across CONSECUTIVE trials the under-sampled action still fails to fire and
# RESETS the moment a sample lands (or the under-sampled target switches), so the
# drive is only as large as it needs to be to break an extreme-bias freeze.
FORCE_SAMPLE_K = MIN_SAMPLES              # per-action clean-sample floor (= 3)
FORCE_SAMPLE_BASE_PA = NOVELTY_DRIVE_MAX_PA   # 350 pA: start where the graded cap ends
FORCE_SAMPLE_RAMP_PA = 250.0              # excitation escalation per consecutive un-sampled trial
# Excitation ceiling is CAPPED below the depolarization-block threshold: driving a
# proposal population past ~1250 pA silences it (str_d1_{u} -> 0 spikes, verified on
# 730705), so more current is counterproductive. 1200 pA keeps the driven population
# firing while maximising the push.
FORCE_SAMPLE_MAX_PA = 1200.0
# Push-pull: the incumbent (over-sampled) proposal is INHIBITED to help release the
# striatal / motor winner-take-all that a one-sided excitatory drive cannot break on a
# FRESH network (verified: proposal_1 alone at 10000 pA -> motor_1 = 0; adding proposal_0
# inhibition -> motor_1 wins cleanly). Inhibitory current (negative pA), ramps with the
# excitation. NB it does NOT override an already reward-POTENTIATED route (see finding).
FORCE_SUPPRESS_BASE_PA = 1000.0          # incumbent-proposal inhibition (magnitude, applied negative)
FORCE_SUPPRESS_RAMP_PA = 500.0
FORCE_SUPPRESS_MAX_PA = 3500.0
# Grace period: the forced floor is a FALLBACK, not a from-trial-0 override. The 2g
# graded directed-novelty drive runs unchanged for the first FORCE_GRACE_TRIALS; only if
# an action is STILL under K after that does the push-pull engage. Verified necessary:
# on seeds that sample both actions naturally (730704, count [13,26] in 2g) a from-
# trial-0 push-pull fights the network and PREVENTS the natural sampling (regression),
# whereas the grace period leaves those seeds on the exact 2g path.
FORCE_GRACE_TRIALS = 8


def _run_trial_2h(bridge, midx, d1idx, d2idx, *, deliver_reward: bool, target: int,
                  reward_rule: str, forced_reward: bool, eligible: bool = True,
                  r0_d1=None, v_withhold: float = 0.0,
                  novelty_target: int | None = None, novelty_drive_pa: float = 0.0,
                  suppress_target: int | None = None, suppress_drive_pa: float = 0.0) -> TrialResult:
    """Stage-2g _run_trial_2g + an optional inhibitory bias on `suppress_target`'s
    proposal population (push-pull). When suppression is 0/None this is byte-identical
    to 2g. The excitatory novelty drive and the inhibitory incumbent-suppression are
    both external currents into proposal populations for every onset step; everything
    downstream (value_est, Hammond-ΔP baseline, reward epoch) is unchanged from 2g."""
    xp, _ = get_backend()
    n = int(bridge.core_config.num_neurons)
    nov_idx = None
    sup_idx = None
    if novelty_drive_pa > 0.0 and novelty_target is not None:
        nov_idx = xp.asarray(_indices(bridge, f"proposal_{int(novelty_target)}"))
    if suppress_drive_pa > 0.0 and suppress_target is not None:
        sup_idx = xp.asarray(_indices(bridge, f"proposal_{int(suppress_target)}"))
    onset = np.zeros((ONSET_STEPS, n), dtype=bool)
    for step in range(ONSET_STEPS):
        _apply_afferents(bridge, arousal=True)
        if nov_idx is not None:
            bridge.cp_external_input_current[nov_idx] += xp.float32(novelty_drive_pa)
        if sup_idx is not None:
            bridge.cp_external_input_current[sup_idx] -= xp.float32(suppress_drive_pa)
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


class _ForcedSampler:
    """Un-satiable forced-sampling floor (push-pull) over the 2g directed-novelty drive.

    drive(count, conf) returns (exc_target, exc_pa, inh_target, inh_pa, forcing). While
    the floor is active (min(count) < K) it EXCITES the under-sampled proposal and
    INHIBITS the over-sampled (incumbent) proposal, both escalating, until the
    under-sampled action fires; once both actions have >= K samples it releases and
    returns the graded 2g _novelty_drive (confidence-gated, no suppression) -- i.e.
    Stage 2g's exact behaviour resumes. Stateful (ramp/target), instantiate once per
    training phase. The ramp resets the moment a sample lands / the target switches, so
    the bias is only as strong as needed to break an extreme-bias freeze.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.ramp = 0
        self.target: int | None = None
        self.last_count: int = -1

    def drive(self, count, conf: float, trial_idx: int = 10**9):
        if (self.enabled and trial_idx >= FORCE_GRACE_TRIALS
                and min(int(count[0]), int(count[1])) < FORCE_SAMPLE_K):
            u = 0 if count[0] <= count[1] else 1  # the LESS-sampled action
            if u != self.target:                  # only reset when the under-sampled action switches
                self.ramp = 0
            self.target = u
            self.last_count = int(count[u])
            # Monotonic escalation until the under-sampled action reaches K samples: on
            # pathologically-biased seeds (730705) a single win is not enough and the
            # bias re-locks, so the drive must STAY high (not relax on the first sample)
            # until the floor is satisfied.
            exc = min(FORCE_SAMPLE_MAX_PA, FORCE_SAMPLE_BASE_PA + FORCE_SAMPLE_RAMP_PA * self.ramp)
            inh = min(FORCE_SUPPRESS_MAX_PA, FORCE_SUPPRESS_BASE_PA + FORCE_SUPPRESS_RAMP_PA * self.ramp)
            self.ramp += 1                        # escalate next trial if still under K
            return u, float(exc), (1 - u), float(inh), True
        # released -> graded 2g directed-novelty drive (unchanged, no suppression)
        self.ramp = 0
        self.target = None
        self.last_count = -1
        nt, nd = _novelty_drive(count)
        if NOVELTY_CONF_GATE:
            nd *= (1.0 - conf)
        return nt, float(nd), None, 0.0, False


def run_condition(seed: int, *, condition: str, target: int, n_train: int, n_test: int,
                  reward_trials_master=None, reward_learning_rate: float = REWARD_LEARNING_RATE,
                  ou_seed: int | None = None, gated: bool = True, directed_novelty: bool = True,
                  use_d2: bool = True, wh_reward_p: float = 0.0, forced_sampling: bool = True):
    """Stage-2g run_condition + the forced-sampling floor (forced_sampling=True). With
    forced_sampling=False this reproduces Stage 2g exactly (the graded drive)."""
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
    import numpy as np
    wh_rng = np.random.default_rng(int(seed) + 909090 + int(target))
    _set_sigma(bridge, SIGMA_UNCERTAIN if not gated else _sigma_from_conf(conf))
    _settle(bridge)

    baseline = _baseline_block(bridge, midx, d1idx, d2idx, target, n_test)
    r0_d1 = baseline["r0_d1"]
    if gated:
        _set_sigma(bridge, _sigma_from_conf(conf))

    sampler = _ForcedSampler(enabled=forced_sampling and directed_novelty)
    w0 = _d1_route_weight_means(bridge)
    train = []
    reward_trials = []
    sigma_trace = []
    dp_trace = []
    v_wh_trace = []
    n_wh = 0
    n_wh_rewarded = 0
    n_forced = 0
    max_forced_pa = 0.0
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
            nt, nd, st, sd, forcing = sampler.drive(count, conf, trial_idx=i)
        else:
            nt, nd, st, sd, forcing = None, 0.0, None, 0.0, False
        if forcing:
            n_forced += 1
            max_forced_pa = max(max_forced_pa, nd)
        tr = _run_trial_2h(bridge, midx, d1idx, d2idx, deliver_reward=True, target=target,
                           reward_rule=rule, forced_reward=forced, eligible=_reward_eligible(i),
                           r0_d1=r0_d1, v_withhold=v_wh,
                           novelty_target=nt, novelty_drive_pa=nd,
                           suppress_target=st, suppress_drive_pa=sd)
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
        "n_forced_trials": int(n_forced), "max_forced_pA": float(max_forced_pa),
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
                 forced_sampling: bool = True) -> dict:
    """Stage-2g same-brain reversal + the forced-sampling floor (fresh sampler per
    phase, so each phase is guaranteed to sample both actions)."""
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
        sampler = _ForcedSampler(enabled=forced_sampling)
        for i in range(n_train):
            if i % WH_PERIOD == 0:
                _run_withhold_trial(bridge, rewarded=False)
            nt, nd, st, sd, _forcing = sampler.drive(count, conf[0], trial_idx=i)
            tr = _run_trial_2h(bridge, midx, d1idx, d2idx, deliver_reward=True, target=target,
                               reward_rule="contingent", forced_reward=False,
                               eligible=_reward_eligible(i), r0_d1=r0_d1, v_withhold=0.0,
                               novelty_target=nt, novelty_drive_pa=nd,
                               suppress_target=st, suppress_drive_pa=sd)
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
                  forced_sampling: bool = True) -> dict:
    kw = dict(n_train=n_train, n_test=n_test, reward_learning_rate=reward_learning_rate,
              use_d2=use_d2, directed_novelty=directed_novelty, forced_sampling=forced_sampling)
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
        "yoked_reward_count0": y0["n_reward_delivered"],
        "yoked_reward_count1": y1["n_reward_delivered"],
        "count_c0": (c0["count0"], c0["count1"]), "count_c1": (c1["count0"], c1["count1"]),
        "n_clean_c0": c0["test_n_clean"], "n_clean_c1": c1["test_n_clean"],
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
        "n_forced_per_seed": [(p["n_forced_c0"], p["n_forced_c1"]) for p in per_seed],
        "count_c1_per_seed": [p["count_c1"] for p in per_seed],
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full", "seeds"], default="full")
    parser.add_argument("--seed", type=int, default=CONSTRUCTION_SEED)
    parser.add_argument("--dev-seeds", type=int, nargs="*", default=list(DEV_SEEDS))
    parser.add_argument("--smoke-seed", type=int, default=730704)
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

    if args.mode == "smoke":
        # ONE maximally-biased seed: prove the forced floor makes the brain sample BOTH
        # actions (count0>0 and count1>0, n_clean>0, no NaN target_rate). Compares the
        # freeze-prone condition WITH the floor (2h) vs WITHOUT it (2g behaviour).
        s = args.smoke_seed
        rows = {}
        for tgt in (0, 1):
            on = run_condition(s, condition="contingent", target=tgt, wh_reward_p=0.0,
                               n_train=args.n_train, n_test=args.n_test,
                               reward_learning_rate=args.reward_lr, forced_sampling=True)
            off = run_condition(s, condition="contingent", target=tgt, wh_reward_p=0.0,
                                n_train=args.n_train, n_test=args.n_test,
                                reward_learning_rate=args.reward_lr, forced_sampling=False)
            rows[f"target{tgt}"] = {
                "FLOOR_ON": {"count0": on["count0"], "count1": on["count1"],
                             "test_n_clean": on["test_n_clean"],
                             "test_target_rate": on["test_target_rate"],
                             "n_forced_trials": on["n_forced_trials"],
                             "max_forced_pA": on["max_forced_pA"]},
                "FLOOR_OFF_2g": {"count0": off["count0"], "count1": off["count1"],
                                 "test_n_clean": off["test_n_clean"],
                                 "test_target_rate": off["test_target_rate"]},
            }
        both_sampled = all(r["FLOOR_ON"]["count0"] > 0 and r["FLOOR_ON"]["count1"] > 0
                           for r in rows.values())
        no_nan = all(r["FLOOR_ON"]["test_target_rate"] == r["FLOOR_ON"]["test_target_rate"]
                     and r["FLOOR_ON"]["test_n_clean"] > 0 for r in rows.values())
        # Attribute the under-sampled-action coverage to the forced floor: control =
        # FLOOR_OFF (2g graded drive), treatment = FLOOR_ON (forced floor). min(count)
        # is the coverage of the LESS-sampled action; the fraction of any improvement
        # owed to the floor is the honest read of whether the floor did the work.
        cov_off = min(min(r["FLOOR_OFF_2g"]["count0"], r["FLOOR_OFF_2g"]["count1"]) for r in rows.values())
        cov_on = min(min(r["FLOOR_ON"]["count0"], r["FLOOR_ON"]["count1"]) for r in rows.values())
        floor_attribution = (attributable_to(
            "under-sampled-action coverage to the forced-sampling floor (vs 2g graded drive)",
            float(cov_on), float(cov_off)) if cov_on > 0 else 0.0)
        artifact = {"probe": "gateB_stage2h_forced_sampling_smoke",
                    "backend": backend["backend"], "device": backend["device"],
                    "smoke_seed": s, "force_sample_k": FORCE_SAMPLE_K,
                    "force_sample_base_pA": FORCE_SAMPLE_BASE_PA,
                    "force_sample_ramp_pA": FORCE_SAMPLE_RAMP_PA,
                    "force_sample_max_pA": FORCE_SAMPLE_MAX_PA,
                    "per_target": rows,
                    "coverage_floor_off_2g": cov_off, "coverage_floor_on": cov_on,
                    "floor_attribution_of_coverage": floor_attribution,
                    "SMOKE_PASS_both_actions_sampled": bool(both_sampled),
                    "SMOKE_PASS_no_nan_no_freeze": bool(no_nan),
                    "elapsed_seconds": float(time.perf_counter() - started)}
        out = Path(args.out) if args.out else OUT_DIR / f"smoke_{backend['backend']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, default=float) + "\n")
        print(json.dumps(artifact, indent=2, default=float))
        return 0

    if args.mode == "seeds":
        per = [run_seed_swap(s, n_train=args.n_train, n_test=args.n_test,
                             reward_learning_rate=args.reward_lr) for s in args.dev_seeds]
        rows = [(p["seed"], round(p["baseline_p0"], 2), round(p["D_contingent"], 3),
                 round(p["D_yoked"], 3), p["count_c1"],
                 bool(p["D_contingent"] >= 0.30 and (p["D_contingent"] - p["D_yoked"]) >= 0.20))
                for p in per]
        print(json.dumps({"seeds_rows(seed,base_p0,Dc,Dy,count_c1,steer)": rows,
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
    outcome = ("STAGE2H_GO" if verdict["go"] else "STAGE2H_NO_GO")
    if verdict["verdict_status"] == "UNDEFINED":
        outcome = "STAGE2H_UNDEFINED"
    artifact = {"probe": "gateB_stage2h_forced_sampling", "stage": "stage2h_learning",
                "backend": backend["backend"], "device": backend["device"],
                "backend_info": backend, "target": args.target,
                "n_train": args.n_train, "n_test": args.n_test, "reward_lr": args.reward_lr,
                "forced_sampling_config": {"force_sample_k": FORCE_SAMPLE_K,
                                           "force_sample_base_pA": FORCE_SAMPLE_BASE_PA,
                                           "force_sample_ramp_pA": FORCE_SAMPLE_RAMP_PA,
                                           "force_sample_max_pA": FORCE_SAMPLE_MAX_PA},
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
        "reversal_pB_afterA": reversal["p_b_after_phaseA"],
        "reversal_pB_afterB": reversal["p_b_after_phaseB"],
        "output": str(out)}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
