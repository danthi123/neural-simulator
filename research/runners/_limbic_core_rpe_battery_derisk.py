"""Cheap-first de-risk for ROADMAP ITEM #1 — the shared reward/value/dopamine LIMBIC CORE
as a standalone, co-residable spiking ORGAN (the Schultz RPE battery, r/V/delta all neural).

Roadmap: research/findings/2026-06-18-full-spikeification-shared-substrate-roadmap.md (§4).
Directive: TRUE ONE BRAIN — move every cognitive computation onto the shared spiking
substrate. The merged "one brain" (nav_conv_merged_bridge) currently has NO limbic core
(build_bg_brain_regions is called with default kwargs). Before lifting the limbic slice onto
the merge, this de-risk re-validates it as a STANDALONE organ and PINS the frozen GO bar.

THE ORGAN UNDER TEST (the audit's minimal spiking actor-critic limbic core, §4)
------------------------------------------------------------------------------
    state_cue (CS)  --plastic-->  striosome_value (GABAergic MSN-D1 critic, learns V)
                                         |
                                   GABA_B/GIRK  (-V, E_K=-90mV; the value subtraction)
                                         v
    reward_us (US, PPN-like)  --exc-->  snc (DOPAMINE)  ==>  delta = r - V  (the SNc FIRING)
                                         ^
                                   tonic pacemaker
    dopamine modulator: from_region_firing_signed over [snc]  ->  plasticity_rate scope=all
    (so the critic LEARNS V via three-factor: STDP eligibility x the SNc-derived DA delta)

WHY reward_us (a SPIKING afferent) and NOT a direct SNc reward current: the minimal
cue->striosome->snc probe (snc_stageb_critic_probe.py) delivers the reward as a 400 pA
direct SNc current that SATURATES the SNc (~130 Hz), so the GABA_B -V cannot dent it (the
predicted/unpredicted gap collapses). A tunable SYNAPTIC reward (reward_us->snc) leaves the
SNc headroom for a graded value subtraction (the N5 probe gets corr=-0.99 this way). This is
the audit's exact topology + the brain-based-only bar (the reward enters as a SPIKE, not a
host current write).

THE SCHULTZ RPE BATTERY (the reward is the dependent variable; behavior is NOT the test)
----------------------------------------------------------------------------------------
  (1) BURST on an unpredicted US        : reward_us fires, V~0 -> snc >> tonic.
  (2) GRADED in reward magnitude        : bigger US drive -> bigger snc burst (corr >= +0.8).
  (3) OMISSION DIP                      : trained state (V>0), US withheld -> snc < tonic.
  (4) PREDICTED-US burst SHRINKS        : after the critic learns V at the cue, the US burst
                                          at the predicted state shrinks >= 50% vs unpredicted.
  (5) REWARD LESION (decisive)          : zero reward_us->snc -> the burst vanishes (the RPE
                                          IS the synaptic reward, not a re-hidden host scalar).
  (6) CRITIC (GABA_B) LESION (decisive) : zero the striosome_value->snc GABA_B routing mask ->
                                          the predicted/unpredicted gap collapses (V was the
                                          synaptic GABA_B, not host arithmetic).

FROZEN GO BAR (pre-registered, §4.3): (1) burst/tonic >= 3x; (2) corr >= +0.8; (3) omission <
tonic; (4) predicted burst shrinks >= 50%; (5) reward-lesion burst within +-15% of tonic;
(6) critic-lesion gap <= 1.2x. Multi-seed: >= 5/6 seeds (lesions mechanistic -> 3 clean ok).

HONEST-NEGATIVE framing (the deliverable): a delta=r-V (Rescorla-Wagner) that holds but no
TD cue-shift is the expected, documented R-W-vs-TD boundary -> feeds roadmap #3. A graded-delta
that needs the (already-shipped) GABA_B/coincidence edits re-confirms those are necessary.

CPU-friendly (tiny ~170-neuron bridge). Run under SIM_BACKEND=numpy.

Usage
-----
    SIM_BACKEND=numpy python -m research.runners._limbic_core_rpe_battery_derisk --seeds 42,43,44,100,101,102
    SIM_BACKEND=numpy python -m research.runners._limbic_core_rpe_battery_derisk --opsearch   # operating-point search
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as _st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def build_limbic_core(seed, *, n_cue=40, n_strio=60, n_reward_us=40, n_snc=30,
                      cue_to_strio_weight=10.0, reward_us_to_snc_weight=10.0,
                      strio_to_snc_weight=10.0, gabab_prop=0.22, gabab_tau_decay=150.0,
                      gabab_conductance_max=0.0, reward_learning_rate=0.08,
                      snc_da_sensitivity=8.0, enable_heterogeneity=True):
    """Build the minimal reward_us -> snc <- striosome_value(GABA_B) limbic organ.

    cue->striosome_value is PLASTIC (the value V learned by the SNc-derived DA delta via the
    three-factor pipeline). reward_us->snc is fixed excitatory (the synaptic reward r).
    striosome_value->snc is fixed inhibitory routed through the slow GABA_B/GIRK K+ conductance
    (E_K=-90mV) so V subtracts strongly + sign-correctly at the SNc membrane. The dopamine
    modulator reads snc firing via from_region_firing_signed so da_signal = the spiking delta.
    """
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType
    from sim.neuromodulators import NeuromodulatorConfig, ModulatorTarget, ProductionRule

    cfg = CoreSimConfig()
    # Pin the bridge RNG to `seed` (the harness-fix #5 lesson) so each --seed is reproducible.
    cfg.seed = int(seed); cfg.heterogeneity_seed = int(seed); cfg.ou_seed = int(seed)
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    # The critic LEARNS: STDP supplies eligibility (pre/post co-firing), reward modulation
    # converts eligibility -> weight change via the SNc-derived da_signal.
    cfg.enable_stdp = True
    cfg.enable_hebbian_learning = False
    cfg.enable_reward_modulation = True
    cfg.enable_short_term_plasticity = False   # depressing cortico-striatal STP starves the critic
    cfg.enable_structural_plasticity = False
    # Heterogeneity (default ON, biological). The merged "one brain" runs it OFF for nav/conv determinism, so a
    # co-resident validation can disable it here to MATCH the merged operating point (the het-off controlled test).
    cfg.enable_parameter_heterogeneity = bool(enable_heterogeneity)
    cfg.reward_learning_rate = float(reward_learning_rate)
    cfg.current_reward_signal = 0.0            # BRAIN-BASED: the SNc FIRING is the signal, not a host scalar
    cfg.reward_baseline = 0.0
    cfg.stdp_w_max = 40.0                       # above the critic working range (soft-bound gotcha)

    # GABA_B/GIRK slow K+ inhibitory conductance (the already-shipped + owner-approved edit).
    cfg.enable_gabab = True
    cfg.gabab_reversal_potential = -90.0
    cfg.gabab_tau_decay = float(gabab_tau_decay)
    cfg.gabab_propagation_strength = float(gabab_prop)
    cfg.gabab_conductance_max = float(gabab_conductance_max)   # 0 = no GIRK saturation cap

    RS = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name
    cfg.brain_regions = [
        BrainRegion(name="cue", n_neurons=n_cue, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False, izh_neuron_type=RS),
        BrainRegion(name="striosome_value", n_neurons=n_strio, exc_fraction=0.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                    weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D1.name,
                    syn_reversal_potential_i_override=-60.0),
        BrainRegion(name="reward_us", n_neurons=n_reward_us, exc_fraction=1.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                    weight_jitter=0.0, plastic_internal=False, izh_neuron_type=RS),
        BrainRegion(name="snc", n_neurons=n_snc, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False, izh_neuron_type=NeuronType.IZH2007_DOPAMINE.name,
                    syn_reversal_potential_i_override=-55.0),   # SNc lacks KCC2 -> depolarized E_GABA
    ]
    cfg.region_pathways = [
        # The learned value: cue (state) -> striosome (V). PLASTIC.
        RegionPathway(from_region="cue", to_region="striosome_value",
                      density=0.6, weight_mean=float(cue_to_strio_weight),
                      weight_jitter=0.5, plastic=True),
        # The synaptic reward r: reward_us -> snc (excitatory). FIXED.
        RegionPathway(from_region="reward_us", to_region="snc",
                      density=0.6, weight_mean=float(reward_us_to_snc_weight),
                      weight_jitter=0.2, plastic=False),
        # The value subtraction -V: striosome_value -> snc via the slow GABA_B/GIRK K+ conductance.
        RegionPathway(from_region="striosome_value", to_region="snc",
                      density=0.5, weight_mean=float(strio_to_snc_weight),
                      weight_jitter=0.2, plastic=False, receptor="gaba_b"),
    ]
    # The Stage-A/B dopamine modulator: production = from_region_firing_signed over [snc].
    snc_tonic_firing_fraction = 0.30
    cfg.enable_neuromodulator_subsystem = True
    cfg.neuromodulators = [NeuromodulatorConfig(
        name="dopamine", baseline=0.5, decay_tau_ms=200.0,
        concentration_min=0.0, concentration_max=2.0,
        targets=[ModulatorTarget(target_type="plasticity_rate", scope="all", sensitivity=+1.0)],
        production_rules=[ProductionRule(rule_type="from_region_firing_signed",
                                         sensitivity=float(snc_da_sensitivity),
                                         threshold=float(snc_tonic_firing_fraction),
                                         window_ms=200.0, source_regions=["snc"])])]
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = seed
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge, cfg


def _host(a):
    from sim.backend import to_host
    try:
        return to_host(a)
    except Exception:
        return a


def _idx(bridge, name):
    import numpy as np
    return np.asarray(bridge.region_manager.indices(name), dtype=np.int64)


def _drive(bridge, idx_map, drives, n_steps, xp, freeze_lr=None, cfg=None):
    """Set per-region external current (drives: {region: pA}), step n_steps, and return
    (snc_rate_hz, strio_rate_hz, da_mean). freeze_lr=0.0 measures without learning."""
    bridge.cp_external_input_current[:] = 0.0
    for region, pA in drives.items():
        bridge.cp_external_input_current[idx_map[region]] = xp.float32(pA)
    saved_lr = None
    if freeze_lr is not None and cfg is not None:
        saved_lr = cfg.reward_learning_rate
        cfg.reward_learning_rate = float(freeze_lr)
    snc_idx, strio_idx = idx_map["snc"], idx_map["striosome_value"]
    n_snc = len(_host(snc_idx)); n_strio = len(_host(strio_idx))
    snc_spk = strio_spk = 0
    da_sum = 0.0
    for _ in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        # STDP reads current_time_ms for the pre/post delta_t; without advancing it every
        # delta_t is 0 -> no eligibility ever forms (the critic can't learn).
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)
        snc_spk += int(bridge.cp_firing_states[snc_idx].sum())
        strio_spk += int(bridge.cp_firing_states[strio_idx].sum())
        da_sum += float(bridge.neuromodulator_manager.get_concentration("dopamine"))
    if saved_lr is not None:
        cfg.reward_learning_rate = saved_lr
    dur_s = n_steps * 1e-3
    return (snc_spk / max(n_snc, 1) / dur_s,
            strio_spk / max(n_strio, 1) / dur_s,
            da_sum / max(n_steps, 1))


def _settle(bridge, xp, n_steps=80):
    """Clean-reset read protocol (the nav reference's `_n9_reset_critic_read_state`,
    2026-06-10): zero all external current + the slow GABA_B/GIRK conductance, then run a
    silent inter-trial gap so the fast conductances + membranes decay to rest BEFORE the next
    frozen measurement. Without this the GABA_B (tau~150ms) from a prior window (e.g. the
    `predicted` read, where the critic fired) carries over into the next (`unpredicted`) and
    suppresses it — making predicted > unpredicted an ORDER ARTIFACT, not a value subtraction.
    A real inter-trial interval is biologically faithful (the slow GABA_B must decay between
    Pavlovian trials)."""
    bridge.cp_external_input_current[:] = 0.0
    if getattr(bridge, "cp_conductance_g_gabab", None) is not None:
        bridge.cp_conductance_g_gabab[:] = 0.0
    for _ in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)


def _calibrate_da_threshold(bridge, cfg, idx_map, tonic_drives, xp, n_steps=300):
    """Drive the tonic (snc-only) condition, measure the SNc firing FRACTION, set the dopamine
    rule's threshold to it (so a burst -> +da_signal=LTP, a dip -> -da_signal=LTD, tonic -> ~0)."""
    snc_idx = idx_map["snc"]; n_snc = len(_host(snc_idx))
    bridge.cp_external_input_current[:] = 0.0
    for region, pA in tonic_drives.items():
        bridge.cp_external_input_current[idx_map[region]] = xp.float32(pA)
    frac_sum = 0.0; m = 0
    for i in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)
        if i >= n_steps // 2:
            frac_sum += float(bridge.cp_firing_states[snc_idx].sum()) / max(n_snc, 1); m += 1
    tonic_frac = frac_sum / max(m, 1)
    cfg.neuromodulators[0].production_rules[0].threshold = float(tonic_frac)
    return tonic_frac


def _lesion_reward_pathway(bridge):
    """Zero every reward_us->snc edge (anti-cheat 5: the burst must vanish)."""
    import numpy as np
    pre_set = set(int(i) for i in _idx(bridge, "reward_us"))
    post_set = set(int(i) for i in _idx(bridge, "snc"))
    coo = bridge.cp_connections.tocoo()
    rows = np.asarray(_host(coo.row), dtype=np.int64); cols = np.asarray(_host(coo.col), dtype=np.int64)
    mask = np.array([(r in post_set and c in pre_set) for r, c in zip(rows, cols)])
    if not mask.any():
        mask = np.array([(r in pre_set and c in post_set) for r, c in zip(rows, cols)])
        pre = rows[mask]; post = cols[mask]
    else:
        pre = cols[mask]; post = rows[mask]
    if len(pre) == 0:
        return 0
    return bridge.set_pathway_weights("reward_us->snc(lesion)", pre, post,
                                      np.zeros(len(pre), dtype=np.float32))


def _lesion_gabab_mask(bridge):
    """Conductance lesion (anti-cheat 6): zero the GABA_B routing mask so the slow K+ conductance
    gets NO new increment -> the value subtraction must vanish (predicted == unpredicted)."""
    m = getattr(bridge, "cp_gabab_synapse_mask", None)
    if m is None:
        return 0
    n_was = int(_host(m).sum())
    from sim.backend import get_backend
    xp, _ = get_backend()
    bridge.cp_gabab_synapse_mask = xp.zeros_like(m)
    if getattr(bridge, "cp_conductance_g_gabab", None) is not None:
        bridge.cp_conductance_g_gabab[:] = 0.0
    return n_was


def run_battery(seed, *, snc_tonic_pa=220.0, cue_drive_pa=600.0, us_drive_pa=600.0,
                hold_steps=40, n_train=40, lesion_reward=False, lesion_critic=False,
                verbose=True, **build_kw):
    """The Schultz RPE battery on the minimal limbic organ. Acquisition trains V at the CS;
    then the 6 signatures are measured with learning frozen."""
    from sim.backend import get_backend
    import numpy as np
    xp, _ = get_backend()
    bridge, cfg = build_limbic_core(seed, **build_kw)
    idx_map = {n: xp.asarray(_idx(bridge, n)) for n in ("cue", "striosome_value", "reward_us", "snc")}

    tonic_frac = _calibrate_da_threshold(bridge, cfg, idx_map, {"snc": snc_tonic_pa}, xp)

    # Windows. US enters via reward_us (a spiking afferent), NOT a direct SNc current.
    W_floor = {"snc": snc_tonic_pa}                                          # inter-trial tonic floor
    W_cs_us = {"cue": cue_drive_pa, "reward_us": us_drive_pa, "snc": snc_tonic_pa}   # CS + US (LEARN)
    W_us_alone = {"reward_us": us_drive_pa, "snc": snc_tonic_pa}             # US, NO cue (unpredicted)
    W_omission = {"cue": cue_drive_pa, "snc": snc_tonic_pa}                  # CS, NO reward (omission)

    # --- Acquisition: CS->US trials; the critic learns V (US burst shrinks as V cancels r). ---
    us_burst, v_cs = [], []
    for t in range(n_train):
        _drive(bridge, idx_map, W_floor, hold_steps, xp)                    # inter-trial floor
        snc_r, strio_r, da = _drive(bridge, idx_map, W_cs_us, hold_steps, xp)   # LEARN
        us_burst.append(snc_r); v_cs.append(strio_r)
        if verbose and (t < 3 or t % 10 == 0 or t == n_train - 1):
            print(f"  [acq t={t:02d}] US-burst={snc_r:6.2f}Hz  V(strio)={strio_r:6.2f}Hz  DA={da:.3f}")

    early = slice(0, max(1, n_train // 5)); late = slice(-max(1, n_train // 5), None)
    us_early = _st.mean(us_burst[early]); us_late = _st.mean(us_burst[late])
    v_early = _st.mean(v_cs[early]); v_late = _st.mean(v_cs[late])

    if lesion_reward:
        n_cut = _lesion_reward_pathway(bridge)
        if verbose:
            print(f"  [lesion-reward] zeroed {n_cut} reward_us->snc edges")
    if lesion_critic:
        n_cut = _lesion_gabab_mask(bridge)
        if verbose:
            print(f"  [lesion-critic] zeroed {n_cut} GABA_B synapses")

    # --- (1)/(4)/(5)/(6) Test (frozen): baseline / predicted / unpredicted / omission ---
    # Each measurement is preceded by a CLEAN RESET (zero the slow GABA_B + a silent gap) so the
    # value subtraction of one window does not carry into the next (the order artifact fix).
    def measure(drives):
        _settle(bridge, xp)
        return _drive(bridge, idx_map, drives, hold_steps, xp, freeze_lr=0.0, cfg=cfg)[0]

    base_r = measure(W_floor)
    pred_r = measure(W_cs_us)
    unpred_r = measure(W_us_alone)
    omit_r = measure(W_omission)

    # --- (2) GRADED: vary the US drive magnitude (no cue) -> snc rate should rise monotone. ---
    mags = [0.25, 0.5, 0.75, 1.0]
    graded = [measure({"reward_us": us_drive_pa * m, "snc": snc_tonic_pa}) for m in mags]
    corr_mag = float(np.corrcoef(mags, graded)[0, 1]) if len(set(graded)) > 1 else 0.0

    # ---- Gates (the frozen GO bar) ----
    burst_ratio = unpred_r / max(base_r, 1e-6)
    burst_on_us = (burst_ratio >= 3.0)                       # (1)
    graded_ok = (corr_mag >= 0.8)                            # (2)
    omission_dip = (omit_r < base_r)                         # (3)
    pred_shrink = (pred_r <= 0.5 * max(unpred_r, 1e-6))      # (4) predicted burst shrinks >=50%
    gap_ratio = unpred_r / max(pred_r, 1e-6)                 # the state-specific gap

    return {
        "seed": seed, "tonic_frac": tonic_frac, "lesion_reward": lesion_reward,
        "lesion_critic": lesion_critic,
        "base_hz": base_r, "predicted_hz": pred_r, "unpredicted_hz": unpred_r, "omission_hz": omit_r,
        "us_burst_early_hz": us_early, "us_burst_late_hz": us_late,
        "v_cs_early_hz": v_early, "v_cs_late_hz": v_late,
        "burst_ratio": burst_ratio, "gap_ratio": gap_ratio, "corr_mag": corr_mag,
        "graded_rates_hz": graded, "graded_mags": mags,
        "burst_on_us": bool(burst_on_us), "graded": bool(graded_ok),
        "omission_dip": bool(omission_dip), "pred_shrink": bool(pred_shrink),
        "v_learned": bool(v_late > 1.2 * max(v_early, 1e-6)),
        "us_burst_curve": us_burst, "v_cs_curve": v_cs,
    }


def _print(r):
    print(f"  V(strio) on CS  : {r['v_cs_early_hz']:.2f} -> {r['v_cs_late_hz']:.2f} Hz "
          f"(learned: {r['v_learned']})")
    print(f"  tonic / unpred  : {r['base_hz']:.1f} / {r['unpredicted_hz']:.1f} Hz "
          f"(burst {r['burst_ratio']:.2f}x; >=3x: {r['burst_on_us']})")
    print(f"  predicted (CS+US): {r['predicted_hz']:.1f} Hz  (shrinks >=50% vs unpred: {r['pred_shrink']}; "
          f"gap unpred/pred {r['gap_ratio']:.2f})")
    print(f"  omission (CS,noUS): {r['omission_hz']:.1f} Hz  (dip < tonic {r['base_hz']:.1f}: {r['omission_dip']})")
    print(f"  graded (US mag)  : {[round(x,1) for x in r['graded_rates_hz']]}  corr={r['corr_mag']:+.2f} "
          f"(>=+0.8: {r['graded']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--snc-tonic-pa", type=float, default=220.0)
    ap.add_argument("--cue-drive-pa", type=float, default=600.0)
    ap.add_argument("--us-drive-pa", type=float, default=600.0)
    ap.add_argument("--hold-steps", type=int, default=40)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--cue-to-strio-weight", type=float, default=10.0)
    ap.add_argument("--reward-us-to-snc-weight", type=float, default=10.0)
    ap.add_argument("--strio-to-snc-weight", type=float, default=10.0)
    ap.add_argument("--gabab-prop", type=float, default=0.22)
    ap.add_argument("--gabab-conductance-max", type=float, default=0.0)
    ap.add_argument("--reward-learning-rate", type=float, default=0.08)
    ap.add_argument("--opsearch", action="store_true",
                    help="operating-point search on seed 42 (reward_us weight x gabab params)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    build_kw = dict(cue_to_strio_weight=args.cue_to_strio_weight,
                    reward_us_to_snc_weight=args.reward_us_to_snc_weight,
                    strio_to_snc_weight=args.strio_to_snc_weight,
                    gabab_prop=args.gabab_prop,
                    gabab_conductance_max=args.gabab_conductance_max,
                    reward_learning_rate=args.reward_learning_rate)
    run_kw = dict(snc_tonic_pa=args.snc_tonic_pa, cue_drive_pa=args.cue_drive_pa,
                  us_drive_pa=args.us_drive_pa, hold_steps=args.hold_steps, n_train=args.n_train)

    if args.opsearch:
        print("[limbic-core OPSEARCH seed=42] reward_us_w x strio_w x gabab_prop -> clean graded gap?")
        grid = []
        for rw in (5.0, 8.0, 12.0):
            for sw in (3.0, 6.0):
                for gp in (0.01, 0.02, 0.04):
                    bk = dict(build_kw); bk.update(reward_us_to_snc_weight=rw,
                                                   strio_to_snc_weight=sw, gabab_prop=gp)
                    r = run_battery(42, verbose=False, **run_kw, **bk)
                    n_pass = sum([r["burst_on_us"], r["graded"], r["omission_dip"], r["pred_shrink"]])
                    grid.append((rw, sw, gp, r, n_pass))
                    print(f"  rw={rw:4.1f} sw={sw:3.1f} gp={gp:.2f} | tonic={r['base_hz']:5.1f} "
                          f"unpred={r['unpredicted_hz']:5.1f} pred={r['predicted_hz']:5.1f} "
                          f"gap={r['gap_ratio']:.2f} burst={r['burst_ratio']:.1f}x corr={r['corr_mag']:+.2f} "
                          f"| {n_pass}/4 [{'B' if r['burst_on_us'] else '.'}"
                          f"{'G' if r['graded'] else '.'}{'O' if r['omission_dip'] else '.'}"
                          f"{'S' if r['pred_shrink'] else '.'}]")
        best = max(grid, key=lambda g: (g[4], -g[3]["gap_ratio"]))
        print(f"\n  BEST: rw={best[0]} sw={best[1]} gp={best[2]} -> {best[4]}/4 gates "
              f"(gap {best[3]['gap_ratio']:.2f})")
        return

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    results = []
    for s in seeds:
        print(f"[limbic-core seed={s}] reward_us -> snc <- striosome_value(GABA_B): Schultz RPE battery")
        r = run_battery(s, **run_kw, **build_kw)
        _print(r)
        gates = [r["burst_on_us"], r["graded"], r["omission_dip"], r["pred_shrink"]]
        print(f"  RPE battery (seed {s}): {sum(gates)}/4 core "
              f"[burst {r['burst_on_us']}, graded {r['graded']}, dip {r['omission_dip']}, "
              f"pred-shrink {r['pred_shrink']}]\n")
        results.append(r)

    # Lesion anti-cheats (mechanistic; on the first seed, 3 clean is conclusive).
    les_seeds = seeds[:3]
    print("=== LESION ANTI-CHEATS ===")
    lesion_results = []
    for s in les_seeds:
        rl = run_battery(s, lesion_reward=True, verbose=False, **run_kw, **build_kw)
        rc = run_battery(s, lesion_critic=True, verbose=False, **run_kw, **build_kw)
        reward_vanishes = abs(rl["unpredicted_hz"] - rl["base_hz"]) <= 0.15 * max(rl["base_hz"], 1e-6)
        gap_collapses = rc["gap_ratio"] <= 1.2
        print(f"  seed {s}: reward-lesion unpred={rl['unpredicted_hz']:.1f}Hz vs tonic={rl['base_hz']:.1f}Hz "
              f"(vanishes <=15%: {reward_vanishes}) | critic-lesion gap={rc['gap_ratio']:.2f} "
              f"(collapses <=1.2: {gap_collapses})")
        lesion_results.append({"seed": s, "reward_vanishes": bool(reward_vanishes),
                               "gap_collapses": bool(gap_collapses),
                               "reward_lesion": rl, "critic_lesion": rc})

    if len(results) > 1:
        n_core = sum(1 for r in results
                     if r["burst_on_us"] and r["graded"] and r["omission_dip"] and r["pred_shrink"])
        print(f"\n=== MULTI-SEED: {n_core}/{len(results)} PASS all 4 core gates ===")
        for g in ("burst_on_us", "graded", "omission_dip", "pred_shrink"):
            n = sum(1 for r in results if r[g])
            print(f"  {g:14s}: {n}/{len(results)}")
        n_rl = sum(1 for lr in lesion_results if lr["reward_vanishes"])
        n_cl = sum(1 for lr in lesion_results if lr["gap_collapses"])
        print(f"  reward-lesion vanishes: {n_rl}/{len(lesion_results)} | "
              f"critic-lesion gap-collapses: {n_cl}/{len(lesion_results)}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"mode": "limbic_core_rpe_battery", "results": results,
                       "lesions": lesion_results}, f, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
