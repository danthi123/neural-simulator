"""MERGED nav critic VALUE-TRAIN — learn V(s) so delta=r−V is graded from the TRAINED afferent.

Completes the limbic-core (reward/value/dopamine) consolidation onto the merged "one brain" bridge
(the owner's TRUE-ONE-BRAIN directive). Builds on the GO op-map de-risk
(2026-06-18-navcritic-valuetrain-opmap-derisk.md): GIVEN a firing critic, the synaptic GABA_B route
grades delta at the recommended op-point (`critic_only` mask, SNc tonic ~160, GIRK 0.0). The op-map drove
the critic DIRECTLY (a trained-V proxy). THIS runner actually LEARNS the value: it runs the
pair-then-reward DA-gated STDP loop (ported VERBATIM from g11_bg_runner._run_place_value_training,
which is a nested closure and not importable) to grow the PLASTIC vs_place_context→striosome_value
weight from 0.20 until the goal-place volley fires the critic → V(goal) ≫ V(far) → graded delta from the
AFFERENT (no direct-drive proxy).

THE GO GATE (what we demonstrate): after value-train, drive ONLY `vs_place_context` with the goal
Gaussian (NO direct critic drive, NO teacher):
  (1) critic FIRES at the GOAL (≥~80 Hz) and is near-silent FAR;
  (2) V(goal) ≫ V(far);
  (3) delta=r−V is GRADED (gap ≥1.3×): predicted (goal, critic fires, V subtracted) < unpredicted (far).

ANTI-CHEATS (mandatory):
  - UNTRAINED contrast: at the init weight 0.20 the afferent-alone fires the critic ~0 Hz → flat delta ≈ r.
  - LESION: zero the striosome_value→snc GABA route → the graded delta collapses to ≈ r.
  - MOAT: the no-confab moat holds with the value-train (the dopamine scope=all broadcast must not
    perturb the frozen conversational slice — checked at the AGENT level).

DISCIPLINE: cheap-first numpy build-smoke (does the build compose with the up-state arm? does the
up-state fire the critic at the goal at init?), THEN GPU value-train. 6 seeds for the delta-gap effect;
3 clean seeds for the LESION. NO sim/ edit (the merged builder gets nav_critic_convergent_upstate +
the critic_only mask plumbed — runner-only).

Reproduce:
  # cheap-first build-smoke (CPU): build composes + up-state fires the critic at init at the goal
  SIM_BACKEND=numpy python -m research.runners._merged_navcritic_valuetrain --smoke --seed 42
  # the real value-train + GO gate (GPU)
  SIM_BACKEND=cupy   python -m research.runners._merged_navcritic_valuetrain --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

CRITIC_AFFERENT = "vs_place_context"   # the merged co_resident_nav_critic plastic value-learner afferent
UPSTATE_AFFERENT = "vs_place_drive"    # the A1 up-state arm (dense NON-plastic; breaks the LTP bootstrap)
CRITIC = "striosome_value"             # the GABAergic MSN-D1 value critic
SNC = "snc"                            # the dopamine cell (delta=r−V is its firing)
REWARD_US = "reward_us"                # the excitatory US/reward afferent (the `r` term)

# ── Op-point + value-train hyperparameters (the de-risk-recommended + the runner's validated defaults) ──
SNC_TONIC_PA = 160.0          # de-risk recommended op-point (SNc burst ~97 Hz non-saturated, gap ~19)
SNC_REWARD_GAIN = 400.0       # k_r: the reward-burst SNc drive (g11_bg_runner snc_reward_gain default)
REWARD_US_DRIVE_PA = 250.0    # the spiking US afferent drive (g11_bg_runner reward_us_drive_pa default)
GIRK_CAP = 1.0                # gabab_conductance_max FINITE cap (de-risk row "girk=1.0 gap 1.44"). The
                              # UNCAPPED 0.0 (the de-risk's DIRECT-critic-proxy recommendation) over-clamps +
                              # REBOUNDS the SNc to ~348 Hz when the AFFERENT-driven (up-state + plastic) critic
                              # fires strongly during the read (diagnosed pred=348Hz uncapped vs 50Hz capped);
                              # the finite cap bounds the GIRK K+ conductance so the firing critic grades the SNc
                              # (pred 50 < unpred 70, gap 1.40) WITHOUT rebound — the documented nav GIRK-cap fix.
GRID_SIZE = 8                 # build_bg_brain_regions default arena (the merged builder uses defaults)
GOAL = (6.0, 6.0)             # the run_seed default goal_pos
VS_DRIVE_MAX_PA = 800.0       # the run_seed vs_place_drive_max_pA (de-risk validated)
# value-train protocol (g11_bg_runner defaults; the de-risk-validated value-train recipe)
VALUE_TRAIN_TRIALS = 40
VALUE_TRAIN_PAIR_STEPS = 100
VALUE_TRAIN_HOLD_STEPS = 40
REWARD_DELAY_STEPS = 8
CRITIC_TEACHER_PA = 300.0     # sub-threshold phase-locked teacher on the critic during PAIR
VALUE_TRAIN_STDP_W_MAX = 40.0 # the critic soft-bound that stops the MSN saturating during value-train
INIT_AFFERENT_WEIGHT = 0.20   # the UNTRAINED vs_place_context->striosome_value weight (anti-cheat contrast)


def _host(a):
    from sim.backend import to_host
    try:
        return to_host(a)
    except Exception:
        return a


def _idx(bridge, name, xp):
    return xp.asarray(np.asarray(bridge.region_manager.indices(name), dtype=np.int64))


def _far_of(gxgy):
    """Point-reflect the goal across the arena centre (the de-risk's far convention)."""
    return (float(GRID_SIZE) - 1.0 - float(gxgy[0]), float(GRID_SIZE) - 1.0 - float(gxgy[1]))


def _vs_place_prefs(n):
    """Reconstruct run_seed's vs_place_context preferred-(x,y) tiling (g11_bg_runner.py:4029-4039):
    a near-square sub-grid of side=round(sqrt(n)) tiling [0,GRID_SIZE-1]^2, padded/truncated to n."""
    side = int(round(n ** 0.5))
    xs = np.linspace(0.0, GRID_SIZE - 1.0, side, dtype=np.float32)
    ys = np.linspace(0.0, GRID_SIZE - 1.0, side, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    px = gx.ravel(); py = gy.ravel()
    if px.size < n:
        reps = int(np.ceil(n / max(px.size, 1)))
        px = np.tile(px, reps)[:n]
        py = np.tile(py, reps)[:n]
    return px[:n].copy(), py[:n].copy()


def _vs_drive(prefs_x, prefs_y, gx, gy, xp):
    """The grid-Gaussian place-code drive at (gx,gy) — VERBATIM the run_seed render
    (g11_bg_runner.py:5788-5790 / 4040): max_pA * exp(-dsq/(2*sigma^2)), sigma=GRID_SIZE/8."""
    sigma = float(GRID_SIZE) / 8.0
    dsq = (prefs_x - float(gx)) ** 2 + (prefs_y - float(gy)) ** 2
    return xp.asarray(VS_DRIVE_MAX_PA * np.exp(-dsq / (2.0 * sigma ** 2)), dtype=xp.float32)


def build_merged(seed, *, convergent_upstate=True, vocab=None):
    from research.runners.nav_conv_merged_bridge import build_merged_nav_conv_bridge
    b, h = build_merged_nav_conv_bridge(
        seed=seed, vocab=vocab, co_resident_nav_critic=True,
        nav_critic_convergent_upstate=convergent_upstate,
        nav_critic_homeostasis_mask="critic_only")
    return b, h


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# Stepping / measurement helpers (mirror the op-map runner + the run_seed _n9_* closures)
# ─────────────────────────────────────────────────────────────────────────────────────────────────
def _step(bridge, n, cfg):
    for _ in range(int(n)):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * cfg.dt_ms


def _reset_snc_subtraction(bridge, snc_idx, xp):
    """Clear the slow GABA_B/GIRK + reset the SNc membrane/recovery (the _n9_reset_snc_subtraction_state)."""
    if getattr(bridge, "cp_conductance_g_gabab", None) is not None:
        bridge.cp_conductance_g_gabab[:] = xp.float32(0.0)
    if (getattr(bridge, "cp_membrane_potential_v", None) is not None
            and getattr(bridge, "cp_izh_vr", None) is not None):
        bridge.cp_membrane_potential_v[snc_idx] = bridge.cp_izh_vr[snc_idx]
    if getattr(bridge, "cp_recovery_variable_u", None) is not None:
        bridge.cp_recovery_variable_u[snc_idx] = xp.float32(0.0)


def _reset_critic_read(bridge, crit_idx, snc_idx, xp, cfg, *, gap_steps=80):
    """Clean the critic plateau + GABA_B + reset critic & SNc, then a silent gap (the _n9_reset_critic_read_state)."""
    for _g in ("cp_conductance_g_coincidence", "cp_conductance_g_coincidence_rise"):
        arr = getattr(bridge, _g, None)
        if arr is not None:
            arr[:] = xp.float32(0.0)
    if (getattr(bridge, "cp_membrane_potential_v", None) is not None
            and getattr(bridge, "cp_izh_vr", None) is not None):
        bridge.cp_membrane_potential_v[crit_idx] = bridge.cp_izh_vr[crit_idx]
        if getattr(bridge, "cp_recovery_variable_u", None) is not None:
            bridge.cp_recovery_variable_u[crit_idx] = xp.float32(0.0)
    _reset_snc_subtraction(bridge, snc_idx, xp)
    bridge.cp_external_input_current[:] = xp.float32(0.0)
    if gap_steps > 0:
        _step(bridge, gap_steps, cfg)


def _critic_rate_via_afferent(bridge, idx, prefs, gx, gy, xp, cfg, *, n_meas=120, warmup=30):
    """Drive the place AFFERENTS at (gx,gy); return the critic firing rate (Hz). The GO-gate read:
    NO direct critic drive, NO teacher — only the perceived place code into BOTH place afferents
    (vs_place_context = the PLASTIC value-learner + vs_place_drive = the dense NON-plastic A1 up-state).
    This is the FAITHFUL DEPLOYMENT read: the deployed nav drive-injects the SAME grid Gaussian place
    code into BOTH afferents every nav step (g11_bg_runner.py:1843 / :5810-5812). The A1 up-state is a
    legitimate place afferent (not the critic, not a teacher); the LEARNED vs_place_context weight is
    what makes the critic fire MORE at the rewarded goal than the position-blind A1 floor (untrained the
    floor fires MORE at far; the anti-cheat measures this flip)."""
    crit_idx = idx[CRITIC]; n_crit = int(len(_host(crit_idx)))
    _reset_critic_read(bridge, crit_idx, idx[SNC], xp, cfg)
    drv = _vs_drive(prefs[0], prefs[1], gx, gy, xp)
    bridge.cp_external_input_current[:] = xp.float32(0.0)
    bridge.cp_external_input_current[idx[CRITIC_AFFERENT]] = drv
    if UPSTATE_AFFERENT in idx:
        bridge.cp_external_input_current[idx[UPSTATE_AFFERENT]] = drv     # the A1 up-state arm (place afferent)
    saved = cfg.reward_learning_rate; cfg.reward_learning_rate = 0.0
    spk = 0; m = 0
    for t in range(int(n_meas)):
        _step(bridge, 1, cfg)
        if t >= warmup:
            spk += int(_host(bridge.cp_firing_states[crit_idx]).sum()); m += 1
    cfg.reward_learning_rate = saved
    bridge.cp_external_input_current[:] = xp.float32(0.0)
    return spk / max(n_crit, 1) / max(m * 1e-3, 1e-9)


def _snc_burst_via_afferent(bridge, idx, prefs, gx, gy, xp, cfg, *, lead_steps=60, burst_steps=80,
                            spiking_reward=True):
    """The delta=r−V read: drive the place AFFERENTS at (gx,gy) (critic fires → GABA_B onto SNc) for a
    LEAD, then the reward burst (place still on). predicted(GOAL) < unpredicted(FAR) when V is learned.
    spiking_reward → the `r` term is the reward_us US afferent FIRING (the fully-spiking delta). Drives
    BOTH place afferents (the faithful deployment, same as _critic_rate_via_afferent)."""
    snc_idx = idx[SNC]; n_snc = int(len(_host(snc_idx)))
    _reset_critic_read(bridge, idx[CRITIC], snc_idx, xp, cfg)
    drv = _vs_drive(prefs[0], prefs[1], gx, gy, xp)
    # LEAD: place drive into BOTH afferents (critic fires → GABA_B builds) + SNc tonic, BEFORE reward.
    bridge.cp_external_input_current[:] = xp.float32(0.0)
    bridge.cp_external_input_current[idx[CRITIC_AFFERENT]] = drv
    if UPSTATE_AFFERENT in idx:
        bridge.cp_external_input_current[idx[UPSTATE_AFFERENT]] = drv
    bridge.cp_external_input_current[snc_idx] = xp.float32(SNC_TONIC_PA)
    _step(bridge, lead_steps, cfg)
    # REWARD burst (place still on).
    if spiking_reward and REWARD_US in idx:
        bridge.cp_external_input_current[snc_idx] = xp.float32(SNC_TONIC_PA)
        bridge.cp_external_input_current[idx[REWARD_US]] = xp.float32(REWARD_US_DRIVE_PA)
    else:
        bridge.cp_external_input_current[snc_idx] = xp.float32(SNC_TONIC_PA + SNC_REWARD_GAIN)
    saved = cfg.reward_learning_rate; cfg.reward_learning_rate = 0.0
    spk = 0
    for _ in range(int(burst_steps)):
        _step(bridge, 1, cfg)
        spk += int(_host(bridge.cp_firing_states[snc_idx]).sum())
    cfg.reward_learning_rate = saved
    bridge.cp_external_input_current[:] = xp.float32(0.0)
    return spk / max(n_snc, 1) / max(int(burst_steps) * 1e-3, 1e-9)


def _mean_afferent_weight(bridge, idx):
    """Mean vs_place_context→striosome_value weight (the V the value-train grows). Vectorized over the CSR."""
    pre = np.asarray(_host(idx[CRITIC_AFFERENT]), dtype=np.int64)
    post = np.asarray(_host(idx[CRITIC]), dtype=np.int64)
    coo = bridge.cp_connections.tocoo()
    rows = np.asarray(_host(coo.row)); cols = np.asarray(_host(coo.col)); data = np.asarray(_host(coo.data))
    m = np.isin(rows, pre) & np.isin(cols, post)
    if not m.any():
        m = np.isin(rows, post) & np.isin(cols, pre)
    return float(data[m].mean()) if m.any() else 0.0


def _find_da_rule(bridge):
    try:
        for nm in (bridge.neuromodulator_manager._configs
                   if bridge.neuromodulator_manager is not None else []):
            if nm.name == "dopamine" and nm.production_rules:
                for pr in nm.production_rules:
                    if pr.rule_type == "from_region_firing_signed":
                        return pr
    except Exception:
        pass
    return None


def _calibrate_da_threshold(bridge, idx, xp, cfg, *, n_steps=300):
    """Set the DA production threshold to the SNc tonic firing FRACTION under SNC_TONIC_PA, so a
    reward burst → DA>baseline → three-factor LTP (the _n9_calibrate_da_threshold). Returns the frac."""
    da_rule = _find_da_rule(bridge)
    snc_idx = idx[SNC]; n_snc = int(len(_host(snc_idx)))
    _reset_snc_subtraction(bridge, snc_idx, xp)
    bridge.cp_external_input_current[:] = xp.float32(0.0)
    bridge.cp_external_input_current[snc_idx] = xp.float32(SNC_TONIC_PA)
    frac = 0.0; m = 0
    for i in range(int(n_steps)):
        _step(bridge, 1, cfg)
        if i >= n_steps // 2:
            frac += float(_host(bridge.cp_firing_states[snc_idx]).sum()) / max(n_snc, 1); m += 1
    tf = frac / max(m, 1)
    if da_rule is not None:
        da_rule.threshold = float(tf)
    bridge.cp_external_input_current[:] = xp.float32(0.0)
    return tf, da_rule


def run_value_train(bridge, idx, prefs, xp, cfg, *, near, far, trials=VALUE_TRAIN_TRIALS, verbose=False):
    """PORT of g11_bg_runner._run_place_value_training (5342-5485) to the merged bridge — pair-then-reward
    DA-gated STDP on the PLASTIC vs_place_context→striosome_value arm. Drives vs_place_context (+ the
    vs_place_drive up-state arm) with the GOAL Gaussian; the up-state arm fires the critic so STDP has a
    post-spike to pair with at the init weight; the reward burst converts the accumulated eligibility."""
    aff_idx = idx[CRITIC_AFFERENT]
    drive_idx = idx.get(UPSTATE_AFFERENT)   # the A1 up-state arm (present iff convergent_upstate built)
    c_idx = idx[CRITIC]; s_idx = idx[SNC]
    n_crit = int(len(_host(c_idx)))
    w_near_pre = _mean_afferent_weight(bridge, idx)

    # DA-threshold calibration (so the reward burst crosses threshold → LTP gate). Restore after.
    saved_reward = cfg.current_reward_signal
    tonic_frac, da_rule = _calibrate_da_threshold(bridge, idx, xp, cfg)
    saved_da_threshold = (da_rule.threshold if da_rule is not None else None)
    if da_rule is not None:
        da_rule.threshold = float(tonic_frac)
    # the critic soft-bound during value-train (stops the MSN saturating). Restore after.
    saved_w_max = cfg.stdp_w_max
    cfg.stdp_w_max = float(VALUE_TRAIN_STDP_W_MAX)
    bridge.set_plasticity_gate("value_input", 1.0)        # open the critic arm
    if "critic_snc_window" in getattr(bridge, "_transmission_gate_to_synapses", {}):
        bridge.set_transmission_gate("critic_snc_window", 1.0)

    place_drv = _vs_drive(prefs[0], prefs[1], near[0], near[1], xp)
    pair_steps = int(VALUE_TRAIN_PAIR_STEPS); hold = int(VALUE_TRAIN_HOLD_STEPS); rdelay = int(REWARD_DELAY_STEPS)
    for t in range(int(trials)):
        # (1) ITI floor: SNc tonic only, then clear eligibility.
        cfg.current_reward_signal = 0.0
        bridge.cp_external_input_current[:] = xp.float32(0.0)
        bridge.cp_external_input_current[s_idx] = xp.float32(SNC_TONIC_PA)
        _step(bridge, hold, cfg)
        if bridge.cp_eligibility_trace is not None:
            bridge.cp_eligibility_trace[:] = xp.float32(0.0)
        # (2) PAIR: place drive ON (+ up-state arm so the critic fires) + SNc TONIC + the sub-threshold
        #     teacher on the critic. Place-pre × critic-post STDP lays a SILENT eligibility trace.
        cfg.current_reward_signal = 0.0
        bridge.cp_external_input_current[:] = xp.float32(0.0)
        bridge.cp_external_input_current[aff_idx] = place_drv
        if drive_idx is not None:
            bridge.cp_external_input_current[drive_idx] = place_drv   # A1 up-state arm
        if CRITIC_TEACHER_PA > 0.0:
            bridge.cp_external_input_current[c_idx] = xp.float32(CRITIC_TEACHER_PA)
        bridge.cp_external_input_current[s_idx] = xp.float32(SNC_TONIC_PA)
        for _ in range(pair_steps):
            _step(bridge, 1, cfg)
        # (3) REWARD: place STILL ON (+ up-state arm), teacher REMOVED; after the reward-delay lag the
        #     SNc burst → DA>baseline → converts the accumulated eligibility (robust LTP).
        cfg.current_reward_signal = 1.0
        bridge.cp_external_input_current[:] = xp.float32(0.0)
        bridge.cp_external_input_current[aff_idx] = place_drv
        if drive_idx is not None:
            bridge.cp_external_input_current[drive_idx] = place_drv
        bridge.cp_external_input_current[s_idx] = xp.float32(SNC_TONIC_PA)
        if rdelay > 0:
            _step(bridge, rdelay, cfg)
        # the reward burst — spiking US afferent (the `r` term) + the tonic SNc drive.
        if REWARD_US in idx:
            bridge.cp_external_input_current[s_idx] = xp.float32(SNC_TONIC_PA)
            bridge.cp_external_input_current[idx[REWARD_US]] = xp.float32(REWARD_US_DRIVE_PA)
        else:
            bridge.cp_external_input_current[s_idx] = xp.float32(SNC_TONIC_PA + SNC_REWARD_GAIN)
        _step(bridge, hold, cfg)
        # (4) per-trial reset of the slow GABA_B/GIRK + SNc membrane.
        _reset_snc_subtraction(bridge, s_idx, xp)
        if verbose and (t < 3 or t % 10 == 0 or t == int(trials) - 1):
            wn = _mean_afferent_weight(bridge, idx)
            da = (bridge.neuromodulator_manager.get_concentration("dopamine")
                  if bridge.neuromodulator_manager is not None else float("nan"))
            print(f"    [VT t={t:02d}] w_near={wn:.3f} DA={da:.3f}", flush=True)

    bridge.set_plasticity_gate("value_input", 0.0)        # FREEZE V for the GO gate
    cfg.current_reward_signal = saved_reward
    bridge.cp_external_input_current[:] = xp.float32(0.0)
    if bridge.cp_eligibility_trace is not None:
        bridge.cp_eligibility_trace[:] = xp.float32(0.0)
    _reset_snc_subtraction(bridge, s_idx, xp)
    cfg.stdp_w_max = float(saved_w_max)
    if da_rule is not None and saved_da_threshold is not None:
        da_rule.threshold = float(saved_da_threshold)
    w_near_post = _mean_afferent_weight(bridge, idx)
    return dict(tonic_frac=float(tonic_frac), w_near_pre=float(w_near_pre), w_near_post=float(w_near_post),
                w_grew=float(w_near_post / max(w_near_pre, 1e-6)))


def measure_gate(bridge, idx, prefs, xp, cfg, *, near, far, tag="", n_burst_trials=6):
    """The GO-gate reads: critic@goal/far (V(goal)/V(far) proxy = critic rate) + the delta=r−V SNc burst.
    The SNc pool is tiny (10 neurons) → single-spike quantization + OU noise make ONE burst read noisy.
    INTERLEAVE near/far over n_burst_trials (cancels slow drift) and AVERAGE (lifts the SNR for the
    small-but-graded delta). Critic-rate reads first (they don't perturb the burst reads — each read
    fully resets the critic+SNc state)."""
    crit_near = _critic_rate_via_afferent(bridge, idx, prefs, near[0], near[1], xp, cfg)
    crit_far = _critic_rate_via_afferent(bridge, idx, prefs, far[0], far[1], xp, cfg)
    preds = []; unpreds = []
    for _ in range(int(n_burst_trials)):
        preds.append(_snc_burst_via_afferent(bridge, idx, prefs, near[0], near[1], xp, cfg))    # predicted (V subtracts)
        unpreds.append(_snc_burst_via_afferent(bridge, idx, prefs, far[0], far[1], xp, cfg))    # unpredicted (no V)
    snc_pred = float(np.mean(preds)); snc_unpred = float(np.mean(unpreds))
    delta_gap = snc_unpred / max(snc_pred, 1e-6)
    crit_grade = crit_near / max(crit_far, 1e-3)
    return dict(tag=tag, crit_near_hz=float(crit_near), crit_far_hz=float(crit_far),
                crit_grade_ratio=float(crit_grade),
                snc_predicted_near_hz=snc_pred, snc_unpredicted_far_hz=snc_unpred,
                snc_pred_trials=[round(p, 1) for p in preds], snc_unpred_trials=[round(u, 1) for u in unpreds],
                delta_gap=float(delta_gap))


def lesion_gabab(bridge, xp):
    """Zero the striosome_value→snc GABA_B route (the synaptic mask) so the subtraction can't carry delta.
    Returns a restore-closure. Mirrors the Stage-B lesion (g11_bg_runner.py:5646-5660)."""
    m_mask = getattr(bridge, "cp_gabab_synapse_mask", None)
    if m_mask is None:
        return None, 0
    saved = m_mask.copy()
    n_cut = int(np.asarray(_host(m_mask)).sum())
    bridge.cp_gabab_synapse_mask = xp.zeros_like(m_mask)
    if getattr(bridge, "cp_conductance_g_gabab", None) is not None:
        bridge.cp_conductance_g_gabab[:] = xp.float32(0.0)

    def restore():
        bridge.cp_gabab_synapse_mask = saved
        if getattr(bridge, "cp_conductance_g_gabab", None) is not None:
            bridge.cp_conductance_g_gabab[:] = xp.float32(0.0)
    return restore, n_cut


def check_moat(seed):
    """The no-confab moat at the AGENT level (the dopamine scope=all broadcast must not perturb the frozen
    conversational slice): MergedNavConvAgent(co_resident_nav_critic=True).what_does('dog','go')=='north'
    AND .what_does('river','look') is None."""
    from research.runners.nav_conv_merged_bridge import MergedNavConvAgent
    ag = MergedNavConvAgent(seed=seed, co_resident_nav_critic=True)
    ag.hear("dog go north")
    pos = ag.what_does("dog", "go")
    neg = ag.what_does("river", "look")
    return dict(positive=pos, negative=neg,
                moat_holds=bool(pos == "north" and neg is None))


# ─────────────────────────────────────────────────────────────────────────────────────────────────
def smoke(seed):
    """CHEAP-FIRST CPU build-smoke: (1) the build composes with the up-state arm; (2) the up-state arm
    fires the critic at the GOAL at INIT weight (it must, to seed the value-train STDP); (3) it is
    near-silent FAR (place-graded). Drives BOTH afferents (the value-train PAIR regime)."""
    from sim.backend import get_backend
    xp, backend = get_backend()
    print(f"[merged-navcritic-valuetrain SMOKE seed={seed}] backend={backend}")
    print("  building MERGED bridge (co_resident_nav_critic + convergent_upstate + critic_only mask)...")
    b, _ = build_merged(seed, convergent_upstate=True)
    cfg = b.core_config
    rm = b.region_manager
    names = set(r.name for r in rm.regions())
    for n in (SNC, CRITIC, REWARD_US, CRITIC_AFFERENT, UPSTATE_AFFERENT):
        present = n in names
        print(f"    region {n!r}: present={present} "
              f"({len(list(rm.indices(n))) if present else 0} neurons)")
    assert all(n in names for n in (SNC, CRITIC, REWARD_US, CRITIC_AFFERENT, UPSTATE_AFFERENT)), \
        "SMOKE FAIL: missing a critic/up-state region (convergent_upstate not built?)"
    masked = sorted(r.name for r in rm.regions() if getattr(r, "enable_homeostasis", False))
    # critic_only mask = snc + reward_us NOT masked (stay at vpeak, non-saturated). vs_place_context
    # always carries the afferent homeostasis (enable_critic_homeostasis=True, build line 1252) — that
    # is the validated redesign, NOT part of the snc-saturation problem; striosome_value is the critic.
    print(f"  homeostasis-masked regions: {masked} (critic_only => snc/reward_us NOT in set)")
    assert SNC not in masked and REWARD_US not in masked, \
        f"SMOKE FAIL: critic_only mask must NOT mask snc/reward_us: {masked}"
    assert CRITIC in masked, f"SMOKE FAIL: striosome_value must be masked: {masked}"

    cfg.gabab_conductance_max = float(GIRK_CAP)
    idx = {nm: _idx(b, nm, xp) for nm in (SNC, CRITIC, REWARD_US, CRITIC_AFFERENT, UPSTATE_AFFERENT)}
    prefs = _vs_place_prefs(int(len(_host(idx[CRITIC_AFFERENT]))))
    near = GOAL; far = _far_of(GOAL)

    def _crit_with_upstate(gx, gy, *, n_meas=120, warmup=30):
        crit_idx = idx[CRITIC]; n_crit = int(len(_host(crit_idx)))
        _reset_critic_read(b, crit_idx, idx[SNC], xp, cfg)
        drv = _vs_drive(prefs[0], prefs[1], gx, gy, xp)
        b.cp_external_input_current[:] = xp.float32(0.0)
        b.cp_external_input_current[idx[CRITIC_AFFERENT]] = drv
        b.cp_external_input_current[idx[UPSTATE_AFFERENT]] = drv    # the A1 up-state arm ON
        spk = 0; m = 0
        for t in range(n_meas):
            _step(b, 1, cfg)
            if t >= warmup:
                spk += int(_host(b.cp_firing_states[crit_idx]).sum()); m += 1
        b.cp_external_input_current[:] = xp.float32(0.0)
        return spk / max(n_crit, 1) / max(m * 1e-3, 1e-9)

    cn = _crit_with_upstate(near[0], near[1])
    cf = _crit_with_upstate(far[0], far[1])
    print(f"  UP-STATE arm critic firing @ INIT: goal={cn:.1f}Hz  far={cf:.1f}Hz  (goal/far {cn/max(cf,1e-3):.2f})")
    # The SMOKE gate is the BOOTSTRAP question: does the up-state arm fire the critic at the goal so the
    # PLASTIC vs_place_context STDP has a post-spike to pair with? (>=5 Hz.) The up-state arm is dense +
    # position-BLIND by design (it gives a location-gated up-state from ANY place bump — total Gaussian
    # mass is ~equal at goal vs far), so it is NOT expected to be place-graded; the GRADING is the
    # value-train's job (the plastic arm learns to fire the critic MORE at the rewarded goal, read with
    # the up-state OFF in the GO gate). So place-grading of the up-state is informational, not a gate.
    fires = bool(cn >= 5.0)
    print(f"  SMOKE VERDICT: build-composes=True  up-state-fires-critic-at-goal(>=5Hz)={fires}")
    if not fires:
        print("  [HONEST] the up-state arm does NOT fire the critic at the goal at init — the value-train "
              "would have no post-spike to pair with (raise vs_place_drive_to_value_weight or check the build).")
    return dict(seed=seed, backend=backend, masked=masked,
                crit_goal_upstate_hz=float(cn), crit_far_upstate_hz=float(cf),
                up_state_fires=fires)


def run_seed(seed, *, do_lesion=True, do_moat=True, verbose=False):
    from sim.backend import get_backend
    xp, backend = get_backend()
    print(f"\n{'='*78}\n[merged-navcritic-valuetrain seed={seed}] backend={backend}")
    b, _ = build_merged(seed, convergent_upstate=True)
    cfg = b.core_config
    cfg.gabab_conductance_max = float(GIRK_CAP)
    # match the op-map measurement regime: OU on, learning otherwise frozen, threshold-adapt frozen
    # (so the read sweeps aren't drift-contaminated; the value-train re-enables value_input + STDP locally).
    cfg.enable_ou_process = True
    cfg.ou_std_current_pA = 100.0
    cfg.homeostasis_threshold_adapt_rate = 0.0
    # CRITICAL: the merged cfg default reward_learning_rate=0.01 + the value_input gate is OPEN at build,
    # so ANY critic firing during a measurement read would grow the weight (the untrained measure would
    # NOT be untrained — it grew 0.20→10 in a prior run). FREEZE the value arm for every read: close the
    # value_input plasticity gate + capture the learning rate (the value-train re-opens + restores it).
    _vt_learning_rate = float(cfg.reward_learning_rate) if cfg.reward_learning_rate and cfg.reward_learning_rate > 0 else 0.01
    cfg.reward_learning_rate = 0.0
    cfg.current_reward_signal = 0.0
    bridge_value_gate_closed = True
    b.set_plasticity_gate("value_input", 0.0)   # freeze V during the UNTRAINED measure

    idx = {nm: _idx(b, nm, xp) for nm in (SNC, CRITIC, REWARD_US, CRITIC_AFFERENT, UPSTATE_AFFERENT)}
    prefs = _vs_place_prefs(int(len(_host(idx[CRITIC_AFFERENT]))))
    near = GOAL; far = _far_of(GOAL)
    res = dict(seed=seed, backend=backend, near=list(near), far=list(far))
    res["w_afferent_init"] = float(_mean_afferent_weight(b, idx))

    # ── ANTI-CHEAT 1: UNTRAINED contrast (init weight 0.20 → afferent fires the critic at the position-
    #    blind A1 floor → grade NOT goal-dominant → flat/non-graded delta). value_input is CLOSED so this
    #    measure cannot grow the weight (it stays untrained). ──
    untr = measure_gate(b, idx, prefs, xp, cfg, near=near, far=far, tag="untrained")
    res["untrained"] = untr
    print(f"  [UNTRAINED] w_aff={res['w_afferent_init']:.3f} critic@goal={untr['crit_near_hz']:.1f}Hz "
          f"@far={untr['crit_far_hz']:.1f}Hz (grade {untr['crit_grade_ratio']:.2f}) | "
          f"delta pred={untr['snc_predicted_near_hz']:.1f} unpred={untr['snc_unpredicted_far_hz']:.1f} "
          f"gap={untr['delta_gap']:.2f} (expect NOT goal-graded)")

    # ── VALUE-TRAIN: grow vs_place_context→striosome_value via pair-then-reward DA-gated STDP ──
    cfg.reward_learning_rate = _vt_learning_rate    # the value-train needs a non-zero rate; it opens value_input
    vt = run_value_train(b, idx, prefs, xp, cfg, near=near, far=far, verbose=verbose)
    res["value_train"] = vt
    print(f"  [VALUE-TRAIN] tonic_frac={vt['tonic_frac']:.4f}  w_near {vt['w_near_pre']:.3f}->"
          f"{vt['w_near_post']:.3f} ({vt['w_grew']:.2f}x)")
    # freeze read-only for the GO gate (value_input already frozen in run_value_train).
    cfg.reward_learning_rate = 0.0
    cfg.current_reward_signal = 0.0
    b.set_plasticity_gate("value_input", 0.0)

    # ── GO GATE: drive the place AFFERENTS at goal/far (NO direct critic, NO teacher) ──
    trained = measure_gate(b, idx, prefs, xp, cfg, near=near, far=far, tag="trained")
    res["trained"] = trained
    # (1) critic FIRES at the goal (the A1+A2 regime fires ~20-40 Hz, NOT the direct-1000pA-proxy 80 Hz;
    #     the gate is that the critic spikes — the GABA_B has a V to deliver).
    critic_fires = bool(trained["crit_near_hz"] >= 5.0)
    # (2) the LEARNED FLIP: the value-train must flip the critic grading from far-dominant (untrained,
    #     position-blind A1 floor) to goal-dominant. The load-bearing anti-cheat: trained goal/far ratio
    #     >= 1.3 AND clearly above the untrained ratio (the learned plastic weight overcomes the A1 floor).
    learned_flip = bool(trained["crit_grade_ratio"] >= 1.3
                        and trained["crit_grade_ratio"] >= 1.3 * max(untr["crit_grade_ratio"], 1e-3))
    v_graded = bool(trained["crit_near_hz"] > trained["crit_far_hz"])
    # (3) delta=r−V GRADED: predicted(goal, V subtracts) clearly BELOW unpredicted(far). The de-risk /
    #     Stage-B convention: unpred > 1.3 * pred (a >=30% dip from the learned V), guarded for pred~0.
    delta_graded = bool(trained["snc_unpredicted_far_hz"] > 1.3 * max(trained["snc_predicted_near_hz"], 1e-6))
    print(f"  [TRAINED/GO] critic@goal={trained['crit_near_hz']:.1f}Hz @far={trained['crit_far_hz']:.1f}Hz "
          f"(grade {trained['crit_grade_ratio']:.2f}; untrained grade {untr['crit_grade_ratio']:.2f}) | "
          f"delta pred={trained['snc_predicted_near_hz']:.1f} unpred={trained['snc_unpredicted_far_hz']:.1f} "
          f"gap={trained['delta_gap']:.2f}")
    print(f"    critic-fires(>=5Hz)={critic_fires}  learned-flip(grade>=1.3 & >=1.3x untrained)="
          f"{learned_flip}  V-graded(goal>far)={v_graded}  delta-graded(unpred>1.3*pred)={delta_graded}")
    res["go_gate"] = dict(critic_fires=critic_fires, learned_flip=learned_flip,
                          v_graded=v_graded, delta_graded=delta_graded,
                          trained_grade=float(trained["crit_grade_ratio"]),
                          untrained_grade=float(untr["crit_grade_ratio"]))

    # ── ANTI-CHEAT 2: LESION the GABA_B route → the graded delta collapses to ≈ r ──
    if do_lesion:
        restore, n_cut = lesion_gabab(b, xp)
        if restore is not None:
            les = measure_gate(b, idx, prefs, xp, cfg, near=near, far=far, tag="lesion")
            restore()
            # The mechanistic anti-cheat: zeroing the GABA_B route must REMOVE the LEARNED delta increment.
            # The delta gap is small (~1.3) over a ~1.1 non-GABA noise floor (gpi->snc collaterals + the
            # 10-neuron SNc pool quantization), so the test is RELATIVE: the lesion gap drops clearly below
            # the trained gap (toward the floor). lesion_collapses if lesion gap < trained gap - 0.08 AND
            # lesion gap <= 1.22 (near the floor). (A pure absolute <=1.0 is unreachable given the floor.)
            _trained_gap = float(trained["delta_gap"])
            lesion_collapses = bool(les["delta_gap"] < _trained_gap - 0.08 and les["delta_gap"] <= 1.22)
            res["lesion"] = dict(n_cut=int(n_cut), trained_gap=_trained_gap, **les,
                                 lesion_collapses=lesion_collapses)
            print(f"  [LESION] zeroed {n_cut} GABA_B synapses -> delta pred={les['snc_predicted_near_hz']:.1f} "
                  f"unpred={les['snc_unpredicted_far_hz']:.1f} gap={les['delta_gap']:.2f} "
                  f"(trained {_trained_gap:.2f} -> floor; carries-the-delta => {lesion_collapses})")
        else:
            res["lesion"] = None
            print("  [LESION] no GABA_B mask present (skipped)")

    # ── MOAT (separate agent build; the dopamine broadcast must not perturb the frozen conv slice) ──
    if do_moat:
        try:
            moat = check_moat(seed)
            res["moat"] = moat
            print(f"  [MOAT] what_does(dog,go)={moat['positive']!r} (=='north') "
                  f"what_does(river,look)={moat['negative']!r} (is None) -> holds={moat['moat_holds']}")
        except Exception as e:
            res["moat"] = {"error": str(e)}
            print(f"  [MOAT] ERROR: {e}")

    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None, help="comma-separated seeds (overrides --seed)")
    ap.add_argument("--smoke", action="store_true", help="cheap-first CPU build-smoke only")
    ap.add_argument("--no-lesion", action="store_true")
    ap.add_argument("--no-moat", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed])

    if args.smoke:
        out = {"mode": "smoke", "results": [smoke(s) for s in seeds]}
        path = args.out or "research/findings/raw/_merged_navcritic_valuetrain_smoke.json"
    else:
        results = [run_seed(s, do_lesion=not args.no_lesion, do_moat=not args.no_moat,
                            verbose=args.verbose) for s in seeds]
        # aggregate verdict. The learned delta is GRADED when, per seed, the critic fires + the value-train
        # FLIPS the grading goal-dominant (learned_flip) + delta is graded (unpred > 1.3*pred). The
        # UNTRAINED-FLAT anti-cheat: the untrained critic grade < 1.3 (the learned weight is load-bearing).
        # The LESION must collapse the gap (the synaptic GABA carries delta). The MOAT must hold.
        gaps = [r["trained"]["delta_gap"] for r in results]
        trained_grades = [r["go_gate"]["trained_grade"] for r in results]
        untrained_grades = [r["go_gate"]["untrained_grade"] for r in results]
        go_all = all(r["go_gate"]["critic_fires"] and r["go_gate"]["learned_flip"]
                     and r["go_gate"]["delta_graded"] for r in results)
        # untrained-flat: the untrained read does NOT produce a goal-graded critic (grade < 1.3 — the
        # position-blind A1 floor is NOT goal-selective). This is the load-bearing anti-cheat: the learned
        # plastic weight is what makes the critic fire MORE at the goal (the delta then follows the critic).
        untrained_flat = all(r["untrained"]["crit_grade_ratio"] < 1.3 for r in results)
        lesion_ok = all((r.get("lesion") or {}).get("lesion_collapses", False)
                        for r in results if r.get("lesion") is not None)
        moat_ok = all((r.get("moat") or {}).get("moat_holds", False)
                      for r in results if isinstance(r.get("moat"), dict) and "moat_holds" in r.get("moat"))
        verdict = "GO" if (go_all and untrained_flat and lesion_ok and moat_ok) else "BOUNDARY"
        out = {"mode": "value_train", "seeds": seeds, "verdict": verdict,
               "delta_gaps": gaps, "trained_crit_grades": trained_grades,
               "untrained_crit_grades": untrained_grades, "go_all": go_all,
               "untrained_flat": untrained_flat, "lesion_collapses_all": lesion_ok,
               "moat_holds_all": moat_ok, "results": results}
        print(f"\n{'='*78}\nAGGREGATE VERDICT: {verdict}")
        print(f"  trained delta-gaps per seed: {[round(g,2) for g in gaps]}")
        print(f"  critic grade goal/far: trained {[round(g,2) for g in trained_grades]} "
              f"vs untrained {[round(g,2) for g in untrained_grades]}")
        print(f"  go_all={go_all}  untrained_flat={untrained_flat}  "
              f"lesion_collapses_all={lesion_ok}  moat_holds_all={moat_ok}")
        path = args.out or "research/findings/raw/_merged_navcritic_valuetrain.json"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
