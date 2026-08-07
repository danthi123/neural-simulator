"""Gate B Stage 2j: TWO additive fixes for the two Stage-2i residuals, each diagnosed
against the substrate BEFORE building (this lane has had FOUR wrong diagnoses; a fifth --
the "str_d1_1 intrinsically dead, 0 spikes at 200-3000 pA direct" claim -- is CORRECTED
below by direct measurement).

Imports Stage 2i (_vocal_gateb_stage2i_circuit_fixes) and Stage 2g unchanged, so 2i/2g
stay byte-reproducible. Both fixes are additive; FIX C is default-OFF (opt-in --fix-c),
FIX B' replaces 2i's unconditional clamp with an adaptive one (still default-on, but
byte-identical to 2g wherever the clamp never bites).

FIX B'  -- NON-REGRESSIVE (adaptive) RPE floor.
  DIAGNOSIS of why 2i's FIX B regressed dev 5/6->4/6 (730601, 730602): 2i clamped the
  self-value UNCONDITIONALLY -- value_est_base = min(value_est, REWARD_MAG-RPE_FLOOR) on
  EVERY real-action trial. But `base` (reward_baseline) is subtracted for BOTH rewarded
  and NON-rewarded real actions: a non-rewarded (wrong) action gets net RPE = -base, so
  LOWERING base (clamping value_est) makes -base LESS NEGATIVE -> the wrong action is
  DEPRESSED LESS -> contingency (D_contingent) WEAKENS. The 2i finding's premise ("dev
  value_est<=0.9 so byte-unchanged") was wrong: 730601/730602 DO have value_est>0.9 on
  non-rewarded trials, and clamping those weakened their contingency.
  FIX: gate the clamp on the SIGN of the net RPE -- clamp ONLY when the action is
  REWARDED (so a delivered-reward action's net RPE stays >= RPE_FLOOR > 0, arresting
  730704's runaway self-depression) and leave NON-rewarded real actions at the FULL 2g
  depression (value_est un-clamped) -> the wrong-action depression, hence D_contingent,
  is preserved -> no dev regression. Byte-identical to 2g on every trial where the
  rewarded-action net RPE already exceeds RPE_FLOOR (the saturated tail is the only place
  it bites). Neural grounding unchanged: the phasic DA burst to a DELIVERED reward is
  never fully cancelled by expectation (a floor on the reward-delivery RPE only).

FIX C  -- MSN intrinsic-excitability homeostasis (Desai/Turrigiano), extreme-asymmetry
  gated. CORRECTED DIAGNOSIS (direct substrate measurement, this session):
    * str_d1_1 on 730705 is NOT intrinsically unexcitable -- it fires 322 spikes at
      1500 pA and 682 at 3000 pA DIRECT injection (the 2i finding's "0 spikes at
      200-3000 pA" is refuted), its Izhikevich params are SYMMETRIC with str_d1_0
      (vt=-25, k=1, b~-2, C~100 both), and it sits at the SAME membrane potential
      (-64.8 mV vs -65.8 mV) as str_d1_0 -- it is NOT inhibition-clamped.
    * Under the REALISTIC held-out-GO drive (fix_a OFF, directed novelty <=350 pA into
      proposal_1) str_d1_1 fires 51 spikes -- it CAN co-activate when the novelty drive
      targets it. It is silent only under bare arousal+OU (push 0), which is ALSO true of
      the working seeds' channels (730601 d1=[1,6], 730706 d1=[15,1] at push 0), so
      near-silence at rest is NORMAL, not a dead pathway.
    * Lowering vt does NOTHING (str_d1_1 stays 0 at vt-15 mV); only raising k (gain, ~Na
      conductance) moves it (k*3: 0->121 spikes) -- so the intrinsic knob is k, not vt.
    * BUT the k-response is uniform across channels/seeds (k*3 pushes EVERY seed's str_d1
      from ~single-digits to ~250 at rest), and baseline r0_d1 for the FAILING 730705
      (ch1=1.1) is indistinguishable from the PASSING 730706 (ch1=1.0) -- so a
      firing-set-point homeostat cannot selectively revive 730705 without over-exciting
      the working seeds. The ONLY selective signature is the EXTREME rate ASYMMETRY
      (730705 sibling/dead ratio = 93x; every other measured seed <=25x).
  MECHANISM: a str_d1_c population that is near-silent (r0_d1[c] < HOMEO_DEAD_FLOOR) while
  its sibling is hyperactive (r0_d1[other]/r0_d1[c] > HOMEO_ASYM_RATIO) up-regulates its
  intrinsic excitability -- scale cp_izh_k up toward a firing set-point (bounded by
  HOMEO_K_MAX), the Izhikevich analogue of activity-dependent Na/K-channel homeostasis
  (Desai 1999; Turrigiano 2011). The k-scale is calibrated on a same-seed PROBE bridge so
  the training bridge's RNG stream is untouched; the gate fires (on the measured seeds)
  ONLY on 730705, so every non-engaging seed is byte-identical -> no regression by
  construction. Default OFF. HONEST-NEGATIVE CAVEAT: the smoke, not this note, is the
  verdict on whether reviving str_d1_1's firing actually recovers action-1 LEARNING
  (firing is necessary, not sufficient -- action 1 must also win at the motor level).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from research.runners._vocal_action_selector_gate import _indices
from research.runners._vocal_gateb_stage2i_circuit_fixes import (
    OUT_DIR as STAGE2I_OUT_DIR,
    _ForcedSampler2i,
    _apply_motor_homeostat,
    RPE_FLOOR,
    VALUE_EST_BASELINE_CAP,
)
from research.runners._vocal_gateb_stage2g_hammond_deltap import (
    CHANNELS,
    DEV_SEEDS,
    GAP_STEPS,
    HELDOUT_SEEDS,
    LOSER_RATIO,
    MOTOR_THRESHOLD,
    N_TEST,
    N_TRAIN,
    NOVELTY_CONF_GATE,
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
OUT_DIR = ROOT / "research/findings/raw/gateb_stage2j_intrinsic_homeostasis"

# ============================ FIX C: MSN intrinsic-excitability homeostasis ===========
# Engage only on a genuinely DEAD str_d1 channel: near-silent at baseline AND its sibling
# hyperactive (the extreme-asymmetry signature that -- alone among the measured knobs --
# selects 730705 [ratio 93x] and not the passing 730706 [17x] / 730603 [25x]).
HOMEO_DEAD_FLOOR = 2.0        # baseline r0_d1 onset spikes below this = near-silent
HOMEO_ASYM_RATIO = 30.0       # sibling/dead r0 ratio above this = extreme asymmetry
HOMEO_SETPOINT = 60.0         # target str_d1 onset spikes (normal drive) after homeostasis
HOMEO_K_GRID = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
HOMEO_K_MAX = float(HOMEO_K_GRID[-1])


def _probe_d1_onset(bridge, c: int, steps: int = ONSET_STEPS) -> int:
    """str_d1_c onset spikes under the NORMAL drive (arousal + OU, no external push) --
    the drive the homeostat wants the population to fire under."""
    d1 = np.asarray(_indices(bridge, f"str_d1_{c}"))
    tot = 0
    for _ in range(steps):
        _apply_afferents(bridge, arousal=True)
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        fs = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        tot += int(fs[d1].sum())
    return tot


def _homeostat_engage(r0_d1) -> tuple[int | None, float, float]:
    """Return (dead_channel_or_None, dead_r0, ratio). Dead = near-silent sibling of a
    hyperactive channel (extreme asymmetry)."""
    r = [max(float(x), 1e-6) for x in r0_d1]
    dead = int(np.argmin(r))
    other = 1 - dead
    ratio = r[other] / r[dead]
    if r[dead] < HOMEO_DEAD_FLOOR and ratio > HOMEO_ASYM_RATIO:
        return dead, float(r[dead]), float(ratio)
    return None, float(r[dead]), float(ratio)


def _calibrate_k_scale(seed: int, dead_c: int, build_kwargs: dict) -> tuple[float, int, int]:
    """Homeostatic search on a same-seed PROBE bridge (training bridge RNG untouched):
    the smallest k-scale in HOMEO_K_GRID that brings str_d1_dead_c's normal-drive onset
    firing to >= HOMEO_SETPOINT. Returns (k_scale, fired_at_1, fired_after)."""
    xp, _ = get_backend()
    # baseline firing at k=1
    p0 = build_stage2_bridge(seed, **build_kwargs)
    _reconfigure_da_s(p0)
    _set_sigma(p0, _sigma_from_conf(0.0))
    _settle(p0)
    fired1 = _probe_d1_onset(p0, dead_c)
    p0.clear_simulation_state_and_gpu_memory()
    if fired1 >= HOMEO_SETPOINT:
        return 1.0, int(fired1), int(fired1)
    chosen, fired_after = 1.0, fired1
    for ks in HOMEO_K_GRID:
        pb = build_stage2_bridge(seed, **build_kwargs)
        _reconfigure_da_s(pb)
        _set_sigma(pb, _sigma_from_conf(0.0))
        _settle(pb)
        idx = xp.asarray(_indices(pb, f"str_d1_{dead_c}"))
        pb.cp_izh_k[idx] = pb.cp_izh_k[idx] * xp.float32(ks)
        fired = _probe_d1_onset(pb, dead_c)
        pb.clear_simulation_state_and_gpu_memory()
        chosen, fired_after = ks, fired
        if fired >= HOMEO_SETPOINT:
            break
    return float(chosen), int(fired1), int(fired_after)


def _apply_k_homeostasis(bridge, dead_c: int, k_scale: float) -> None:
    """Scale the dead channel's str_d1 population intrinsic gain (cp_izh_k). Intrinsic
    plasticity only: touches no reward/DA signal and selects no action."""
    xp, _ = get_backend()
    idx = xp.asarray(_indices(bridge, f"str_d1_{dead_c}"))
    bridge.cp_izh_k[idx] = bridge.cp_izh_k[idx] * xp.float32(k_scale)


# ============================ FIX B': adaptive (sign-gated) RPE floor ==================
def _run_trial_2j(bridge, midx, d1idx, d2idx, *, deliver_reward: bool, target: int,
                  reward_rule: str, forced_reward: bool, eligible: bool = True,
                  r0_d1=None, v_withhold: float = 0.0,
                  novelty_target: int | None = None, novelty_drive_pa: float = 0.0,
                  inh_str_d1_target: int | None = None, inh_str_d1_pa: float = 0.0,
                  rpe_floor: bool = True) -> TrialResult:
    """Stage-2i _run_trial_2i with the RPE floor made ADAPTIVE (FIX B'): the value_est
    clamp bites ONLY on a REWARDED action whose unclamped net RPE would fall below
    RPE_FLOOR. Non-rewarded real actions keep the full 2g depression. With rpe_floor=False
    (and inh=0) this is byte-identical to 2g."""
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

    # (B') adaptive RPE floor: clamp the self-value ONLY on a REWARDED action, and only
    # enough to keep its net RPE >= RPE_FLOOR. Non-rewarded real actions -> value_est
    # un-clamped -> full 2g wrong-action depression -> contingency preserved.
    clamp_hit = False
    orig_clamp_would_hit_nonrewarded = False
    if rpe_floor and rewarded and real_action:
        cap = float(REWARD_MAG - RPE_FLOOR - v_withhold)
        value_est_base = min(value_est, max(0.0, cap))
        clamp_hit = bool(value_est_base < value_est)
    else:
        value_est_base = value_est
        # diagnostic: would 2i's UNCONDITIONAL clamp have bitten this non-rewarded trial?
        orig_clamp_would_hit_nonrewarded = bool(
            rpe_floor and real_action and (not rewarded)
            and value_est > VALUE_EST_BASELINE_CAP)
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
    tr.clamp_hit = clamp_hit
    tr.orig_clamp_would_hit_nonrewarded = orig_clamp_would_hit_nonrewarded
    return tr


def run_condition(seed: int, *, condition: str, target: int, n_train: int, n_test: int,
                  reward_trials_master=None, reward_learning_rate: float = REWARD_LEARNING_RATE,
                  ou_seed: int | None = None, gated: bool = True, directed_novelty: bool = True,
                  use_d2: bool = True, wh_reward_p: float = 0.0,
                  fix_a: bool = False, fix_b: bool = True, fix_c: bool = False,
                  motor_homeostat: bool = False):
    """Stage-2i run_condition with the adaptive RPE floor (FIX B') and the optional MSN
    k-homeostat (FIX C). fix_a=False, fix_b=False, fix_c=False -> reproduces Stage 2g."""
    plastic = condition != "acq_lesion"
    build_kwargs = dict(enable_reward=True, plastic_d1=plastic,
                        reward_learning_rate=reward_learning_rate, ou_seed=ou_seed,
                        ou_sigma=SIGMA_UNCERTAIN, plastic_d2=True)
    bridge = build_stage2_bridge(seed, **build_kwargs)
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

    # ---- FIX C: MSN intrinsic-excitability homeostasis (extreme-asymmetry gated) -------
    homeo = {"engaged": False, "dead_channel": None, "dead_r0": None, "ratio": None,
             "k_scale": 1.0, "fired_at_k1": None, "fired_after": None}
    if fix_c:
        dead_c, dead_r0, ratio = _homeostat_engage(r0_d1)
        homeo.update(dead_r0=dead_r0, ratio=ratio)
        if dead_c is not None:
            k_scale, f1, fa = _calibrate_k_scale(seed, dead_c, build_kwargs)
            if k_scale > 1.0:
                _apply_k_homeostasis(bridge, dead_c, k_scale)
            homeo.update(engaged=True, dead_channel=int(dead_c), k_scale=float(k_scale),
                         fired_at_k1=int(f1), fired_after=int(fa))

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
    n_clamp_hit = 0
    n_orig_clamp_nonrewarded = 0
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
        tr = _run_trial_2j(bridge, midx, d1idx, d2idx, deliver_reward=True, target=target,
                           reward_rule=rule, forced_reward=forced, eligible=_reward_eligible(i),
                           r0_d1=r0_d1, v_withhold=v_wh,
                           novelty_target=nt, novelty_drive_pa=nd,
                           inh_str_d1_target=it, inh_str_d1_pa=idr, rpe_floor=fix_b)
        n_clamp_hit += int(getattr(tr, "clamp_hit", False))
        n_orig_clamp_nonrewarded += int(getattr(tr, "orig_clamp_would_hit_nonrewarded", False))
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
        "n_clamp_hit": int(n_clamp_hit),
        "n_orig_clamp_nonrewarded": int(n_orig_clamp_nonrewarded),
        "homeostat": homeo,
        "d1_weight_before": w0, "d1_weight_after": w1,
        "final_conf": float(conf), "final_dp": float(dp),
        "final_v_withhold": float(v_wh), "max_v_withhold": float(max(v_wh_trace) if v_wh_trace else 0.0),
        "n_withhold": int(n_wh), "n_withhold_rewarded": int(n_wh_rewarded),
        "r0_d1": r0_d1,
        "Vd1": [float(Vd1[0]), float(Vd1[1])], "Vd2": [float(Vd2[0]), float(Vd2[1])],
        "final_sigma": float(bridge.core_config.ou_std_current_pA),
    }


def run_seed_swap(seed: int, *, n_train: int, n_test: int,
                  reward_learning_rate: float = REWARD_LEARNING_RATE, use_d2: bool = True,
                  directed_novelty: bool = True, enable_withhold: bool = True,
                  fix_a: bool = False, fix_b: bool = True, fix_c: bool = False,
                  motor_homeostat: bool = False) -> dict:
    kw = dict(n_train=n_train, n_test=n_test, reward_learning_rate=reward_learning_rate,
              use_d2=use_d2, directed_novelty=directed_novelty,
              fix_a=fix_a, fix_b=fix_b, fix_c=fix_c, motor_homeostat=motor_homeostat)
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
        "n_clamp_c0": c0["n_clamp_hit"], "n_clamp_c1": c1["n_clamp_hit"],
        "n_orig_clamp_nonrew_c0": c0["n_orig_clamp_nonrewarded"],
        "n_orig_clamp_nonrew_c1": c1["n_orig_clamp_nonrewarded"],
        "homeo_c0": c0["homeostat"], "homeo_c1": c1["homeostat"],
        "r0_d1_c0": c0["r0_d1"], "r0_d1_c1": c1["r0_d1"],
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


def _steer(p) -> bool:
    return bool(p["D_contingent"] >= 0.30 and (p["D_contingent"] - p["D_yoked"]) >= 0.20)


def _smoke_one(seed: int, *, n_train: int, n_test: int, reward_learning_rate: float,
               fix_a: bool, fix_b: bool, fix_c: bool) -> dict:
    """One seed: fixes ON vs 2g (all fixes OFF). Reports per-action counts, n_acted,
    no-NaN, D_contingent/steer, clamp counts, and the homeostat decision."""
    on = run_seed_swap(seed, n_train=n_train, n_test=n_test,
                       reward_learning_rate=reward_learning_rate,
                       fix_a=fix_a, fix_b=fix_b, fix_c=fix_c)
    off = run_seed_swap(seed, n_train=n_train, n_test=n_test,
                        reward_learning_rate=reward_learning_rate,
                        fix_a=False, fix_b=False, fix_c=False)

    def _clean(x):
        return x == x  # not NaN

    return {
        "seed": int(seed), "baseline_p0": on["baseline_p0"],
        "FIXES_ON": {
            "count_c0": on["count_c0"], "count_c1": on["count_c1"],
            "n_clean_c0": on["n_clean_c0"], "n_clean_c1": on["n_clean_c1"],
            "test_rate_c0": on["test_rate_c0"], "test_rate_c1": on["test_rate_c1"],
            "D_contingent": on["D_contingent"], "D_yoked": on["D_yoked"],
            "steer": _steer(on),
            "n_clamp_c0": on["n_clamp_c0"], "n_clamp_c1": on["n_clamp_c1"],
            "homeo_c0": on["homeo_c0"], "homeo_c1": on["homeo_c1"],
            "r0_d1_c0": on["r0_d1_c0"], "r0_d1_c1": on["r0_d1_c1"],
        },
        "STAGE2G_OFF": {
            "count_c0": off["count_c0"], "count_c1": off["count_c1"],
            "n_clean_c0": off["n_clean_c0"], "n_clean_c1": off["n_clean_c1"],
            "test_rate_c0": off["test_rate_c0"], "test_rate_c1": off["test_rate_c1"],
            "D_contingent": off["D_contingent"], "D_yoked": off["D_yoked"],
            "steer": _steer(off),
            "n_orig_clamp_nonrew_c0": off["n_orig_clamp_nonrew_c0"],
            "n_orig_clamp_nonrew_c1": off["n_orig_clamp_nonrew_c1"],
        },
        "SMOKE_no_nan": bool(_clean(on["test_rate_c0"]) and _clean(on["test_rate_c1"])),
        "SMOKE_both_targets_act": bool(on["n_clean_c0"] > 0 and on["n_clean_c1"] > 0),
        "SMOKE_D_defined": bool(_clean(on["D_contingent"]) and abs(on["D_contingent"]) > 1e-9),
        "SMOKE_D_improved_vs_2g": bool(_clean(on["D_contingent"]) and (
            not _clean(off["D_contingent"]) or on["D_contingent"] > off["D_contingent"] + 1e-9)),
        "SMOKE_steer": _steer(on),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full", "seeds"], default="smoke")
    parser.add_argument("--seed", type=int, default=CONSTRUCTION_SEED)
    parser.add_argument("--dev-seeds", type=int, nargs="*", default=list(DEV_SEEDS))
    parser.add_argument("--smoke-seeds", type=int, nargs="*", default=[730601, 730602, 730704, 730705, 730706])
    parser.add_argument("--lesion-seed", type=int, default=730605)
    parser.add_argument("--lesion-target", type=int, default=0)
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--reward-lr", type=float, default=REWARD_LEARNING_RATE)
    parser.add_argument("--fix-a", action="store_true", help="enable FIX A (REFUTED; default OFF)")
    parser.add_argument("--no-fix-b", action="store_true", help="disable FIX B' (adaptive RPE floor)")
    parser.add_argument("--fix-c", action="store_true",
                        help="enable FIX C (MSN k-homeostat, extreme-asymmetry gated; default OFF)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    backend = _backend_info()
    started = time.perf_counter()
    fix_a = bool(args.fix_a)
    fix_b = not args.no_fix_b
    fix_c = bool(args.fix_c)

    if args.mode == "smoke":
        results = [_smoke_one(s, n_train=args.n_train, n_test=args.n_test,
                              reward_learning_rate=args.reward_lr,
                              fix_a=fix_a, fix_b=fix_b, fix_c=fix_c)
                   for s in args.smoke_seeds]
        artifact = {"probe": "gateB_stage2j_intrinsic_homeostasis_smoke",
                    "backend": backend["backend"], "device": backend["device"],
                    "smoke_seeds": args.smoke_seeds,
                    "fix_a": fix_a, "fix_b_adaptive_rpe_floor": fix_b, "fix_c_k_homeostat": fix_c,
                    "config": {"rpe_floor": RPE_FLOOR, "value_est_baseline_cap": VALUE_EST_BASELINE_CAP,
                               "homeo_dead_floor": HOMEO_DEAD_FLOOR, "homeo_asym_ratio": HOMEO_ASYM_RATIO,
                               "homeo_setpoint": HOMEO_SETPOINT, "homeo_k_max": HOMEO_K_MAX},
                    "per_seed": results,
                    "elapsed_seconds": float(time.perf_counter() - started)}
        out = Path(args.out) if args.out else OUT_DIR / f"smoke_{backend['backend']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, default=float) + "\n")
        print(json.dumps(artifact, indent=2, default=float))
        return 0

    if args.mode == "seeds":
        per = [run_seed_swap(s, n_train=args.n_train, n_test=args.n_test,
                             reward_learning_rate=args.reward_lr,
                             fix_a=fix_a, fix_b=fix_b, fix_c=fix_c)
               for s in args.dev_seeds]
        rows = [(p["seed"], round(p["baseline_p0"], 2), round(p["D_contingent"], 3),
                 round(p["D_yoked"], 3), p["count_c1"], round(p["test_rate_c1"], 3),
                 p["homeo_c1"]["engaged"], _steer(p)) for p in per]
        print(json.dumps({
            "seeds_rows(seed,base_p0,Dc,Dy,count_c1,test_rate_c1,homeo_engaged,steer)": rows,
            "steer_passes": sum(r[-1] for r in rows)}, indent=2, default=float))
        return 0

    # full validation: dev-battery steer + acquisition-lesion ATTRIBUTION + reversal (frozen criteria).
    from research.runners._vocal_gateb_stage2i_circuit_fixes import run_reversal as _rev_2i
    equivalence = _assert_stage1_equivalence(args.seed)
    per_seed = [run_seed_swap(s, n_train=args.n_train, n_test=args.n_test,
                              reward_learning_rate=args.reward_lr,
                              fix_a=fix_a, fix_b=fix_b, fix_c=fix_c) for s in args.dev_seeds]
    steer_pass = [_steer(p) for p in per_seed]

    # LESION ATTRIBUTION (frozen criterion): whose is the contingency? Freeze acquisition-time D1
    # plasticity (condition="acq_lesion", plastic_d1=False during training) and ASK — measuring the
    # intact and lesioned D_contingent both is not the same as attributing the difference. If
    # acquisition plasticity owns the contingency, acq_share ~ 1.0 (the lesion collapses D_contingent).
    ls = args.lesion_seed
    lkw = dict(n_train=args.n_train, n_test=args.n_test, reward_learning_rate=args.reward_lr,
               fix_a=fix_a, fix_b=fix_b, fix_c=fix_c)
    intact = next((p for p in per_seed if p["seed"] == ls), None) or run_seed_swap(ls, **lkw)
    la0 = run_condition(ls, condition="acq_lesion", target=0, wh_reward_p=0.0, **lkw)
    la1 = run_condition(ls, condition="acq_lesion", target=1, wh_reward_p=0.0, **lkw)
    D_acq_lesion = _p_action0(la0) - _p_action0(la1)
    acq_share = attributable_to("acquisition D1 plasticity", intact["D_contingent"], D_acq_lesion)

    reversal = _rev_2i(ls, args.n_train, args.n_test, reward_learning_rate=args.reward_lr,
                       fix_a=fix_a, fix_b=fix_b)
    reversal_p_b = reversal["p_b_after_phaseB"]

    print(json.dumps({"equivalence": equivalence,
                      "steer_seed_passes": int(sum(steer_pass)), "steer_per_seed": steer_pass,
                      "baseline_p0_per_seed": [p["baseline_p0"] for p in per_seed],
                      "D_contingent_per_seed": [p["D_contingent"] for p in per_seed],
                      "lesion_seed": ls,
                      "D_contingent_intact": intact["D_contingent"],
                      "D_contingent_acq_lesion": D_acq_lesion,
                      "acquisition_plasticity_share": acq_share,
                      "reversal_p_b_after_phaseA": reversal["p_b_after_phaseA"],
                      "reversal_p_b_after_phaseB": reversal_p_b,
                      "reversal_pass": bool(reversal_p_b >= 0.60)},
                     indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
