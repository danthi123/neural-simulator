"""Rubicon delayed-credit de-risk: does a NEURAL maintained-goal bridge the delay where a DECAYED trace fails?

THE GENUINE UN-BUILT STEP (re-anchored 2026-08-07; see the report). Our record already has:
  * the N9 TD cue-shift (snc_stageb_critic_probe --td-csc): a spiking-SNc reward-timing GO, but its CS->US gap is
    bridged by a DECAYED eligibility trace + world-clocked CSC taps (short gaps only);
  * the PFC-WM recurrent region (2026-04-27-pfc-working-memory): a maintained-goal substrate (GO on nav; the
    delayed-response Stage-2 was PARTIAL/confounded);
  * R4 (2026-06-27-navcloseout-R4-...-NEGATIVE): a delayed-reward value task run NEGATIVE (2-seed) using the
    DECAYED eligibility-trace bridge (reward_eligibility_tau_ms=500), NO maintained-goal.
The Astera/Axon "Rubicon" mechanism (research/findings/raw/_landscape_adoption_plan_axon_rubicon.md #2): the delay
is bridged by a MAINTAINED GOAL (PT/PFC recurrent-NMDA sustained activity), so credit is assigned to the HELD goal,
not a decayed stimulus trace. NO ONE has wired the PFC-WM maintained goal AS the delay bridge into the neural
credit-assignment and run it HEAD-TO-HEAD against the decayed trace. That is this de-risk.

TASK (trace conditioning with a temporal GAP, on the numpy spiking limbic core):
  floor -> CS window (cue drives the critic + loads the PFC maintained-goal) -> GAP (ZERO external drive; the PFC
  must sustain by its OWN recurrence) -> US window (reward_us fires; the SNc bursts -> DA; if the PFC held the goal
  it is co-active with the reward, so DA-gated STDP grows the held-goal->value synapse). Across trials the held
  goal acquires anticipatory VALUE that bridges the gap.

HEAD-TO-HEAD (the only question):
  * treatment (maintained-goal): PFC recurrent slow-NMDA self-excitation intact (recur>0, Wang 2002 persistent
    activity) -> the goal is HELD across the gap -> credit assigned across the delay.
  * control (decayed-trace): the SAME network with the PFC recurrence LESIONED (recur=0, the exact
    _d3_persistent_slot no-recurrence control) -> the PFC dies during the gap -> only the decayed cue->critic
    trace remains -> at a LONG gap it has decayed -> credit is NOT assigned.
  * floor: no-learning (STDP frozen from t0) -> no acquired value (the value must be LEARNED, not structural).

ANTI-CHEATS:
  (a) the maintained goal is NEURAL: during the GAP the external input to cue+PFC is identically ZERO (asserted);
      the PFC firing that bridges the gap is its own recurrent slow-NMDA activity. It COLLAPSES when the recurrence
      is lesioned (the recur=0 control) -> the bridge is neural recurrence, not a host-held variable.
  (b) the RPE/timing is NEURAL: current_reward_signal==0 (no host scalar); the reward r enters synaptically via
      reward_us->snc; the value -V subtracts via the GABA_B/GIRK conductance at the SNc membrane; DA = the SNc
      firing delta. No host TD chain over host-stored states.
  (c) the decayed-trace control FAILS where the maintained-goal WINS: measured at a LONG gap (the informative
      window). If the control passed too, the task -- not the maintained goal -- would be doing the work.
  (d) 6-seed for a generalization claim (this smoke is 3-seed; the 6-seed command is printed for the parent).

numpy backend, tiny bridge, foreground. NO sim/ edit.

Run (smoke, 3-seed):
  SIM_BACKEND=numpy python -m research.runners._rubicon_delayed_credit_derisk --seeds 42,43,44 \
      --gap-short 20 --gap-long 200 --out research/findings/raw/rubicon_delayed_credit/smoke.json
"""
from __future__ import annotations
import argparse
import json
import os
import statistics as _st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def build_core(seed, *, recur=25.0, n_cue=40, n_pfc=60, n_fs=24, n_strio=60, n_reward_us=40, n_snc=30,
               cue_to_pfc_weight=9.0, pfc_recur_weight=None, pfc_to_fs=1.4, fs_to_pfc=10.0,
               pfc_to_strio_weight=3.0, cue_to_strio_weight=3.0, reward_us_to_strio_weight=0.0,
               reward_us_to_snc_weight=10.0, strio_to_snc_weight=10.0,
               gabab_prop=0.22, gabab_tau_decay=150.0, reward_learning_rate=0.10,
               snc_da_sensitivity=8.0,
               vspatch_gate=False, vspatch_gate_threshold=None, vspatch_gate_sensitivity=400.0,
               vspatch_gate_tau_ms=40.0, vspatch_gate_target_sensitivity=12.0,
               vspatch_gate_source="reward_us",
               reward_coactivity=False, coactivity_scale=0.05, coactivity_trace_tau_ms=25.0,
               stdp_off=False, da_from_reward_us=False, da_reward_threshold=0.06,
               da_reward_sensitivity=60.0):
    """Limbic core (cue->striosome_value->snc <- reward_us) EXTENDED with a PFC maintained-goal pool:
    a recurrent slow-NMDA self-excitation (Wang 2002 persistent activity, from _d3_persistent_slot) + a shared FS.
    `recur` is the persistence knob: recur>0 = the goal is HELD across the gap (treatment); recur=0 = the PFC
    cannot hold (the decayed-trace control). pfc->striosome_value is the PLASTIC held-goal->value synapse.

    HALF-2 (VSPatch reward-window-gated potentiation) is an ADDITIVE, DEFAULT-OFF extension: when
    ``vspatch_gate=True`` the plastic held-goal->value synapse (pfc->striosome_value) is tagged with a per-pathway
    ``plasticity_gate="reward_window"`` and a second neuromodulator ("vspatch_gate", from_region_firing on the SNc
    above tonic) DRIVES that gate. Effect: weight UPDATES on pfc->striosome_value are FROZEN outside the reward
    window (SNc at tonic -> gate=0) and PERMITTED only when the phasic DA burst opens the gate (SNc above tonic ->
    gate->1). This is the Rubicon/PVLV VSPatch mechanism: reward-TIME-gated corticostriatal LTP, so a rewarded held
    goal potentiates its value read-out instead of the whole-trial scope-all DA-STDP netting to LTD on the
    saturated synapse. It is NEURAL (the gate value = a spiking-driven NM concentration, no host if-reward flag) and
    CONTINGENT (no DA burst -> gate stays shut -> no potentiation). ``vspatch_gate=False`` reproduces the original
    scope-all DA-STDP HALF-2 control byte-for-byte (this branch is never entered)."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType
    from sim.neuromodulators import NeuromodulatorConfig, ModulatorTarget, ProductionRule

    pfc_recur_weight = float(recur if pfc_recur_weight is None else pfc_recur_weight)

    cfg = CoreSimConfig()
    cfg.seed = int(seed); cfg.heterogeneity_seed = int(seed); cfg.ou_seed = int(seed)
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.enable_stdp = (not stdp_off)
    cfg.enable_hebbian_learning = False
    cfg.enable_reward_modulation = True
    # VSPatch striatal value rule: DA-gated COACTIVITY eligibility (order-independent, always >=0), NOT pair-STDP.
    # On the saturated held-goal->value synapse pair-STDP nets to LTD; the DA-gated coactivity trace x phasic DA
    # gives LTP at reward time. Corticostriatal plasticity is dopamine-gated/eligibility-based, so disabling pair-
    # STDP on this value synapse (stdp_off) is the faithful choice. Both are default-OFF (scope-all baseline keeps
    # pair-STDP); set together for the VSPatch arm.
    cfg.reward_eligibility_from_coactivity = bool(reward_coactivity)
    if reward_coactivity:
        cfg.reward_coactivity_trace_tau_ms = float(coactivity_trace_tau_ms)
        cfg.reward_coactivity_scale = float(coactivity_scale)
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_parameter_heterogeneity = True
    cfg.reward_learning_rate = float(reward_learning_rate)
    cfg.current_reward_signal = 0.0          # BRAIN-BASED: the SNc FIRING is the signal, not a host scalar
    cfg.reward_baseline = 0.0
    cfg.stdp_w_max = 40.0

    # Slow-NMDA recurrent conductance = the persistent-activity substrate (Wang 2002); the PFC recurrent
    # self-pathway is routed exc_receptor="nmda_slow" (AMPA suppressed -> slow tau=100ms hold).
    cfg.enable_nmda = True
    cfg.enable_nmda_recurrent = True
    cfg.nmda_recurrent_tau_decay_ms = 100.0
    cfg.nmda_recurrent_propagation_strength = 0.05

    # GABA_B/GIRK slow K+ inhibitory conductance: the value -V subtracts at the SNc (E_K=-90mV).
    cfg.enable_gabab = True
    cfg.gabab_reversal_potential = -90.0
    cfg.gabab_tau_decay = float(gabab_tau_decay)
    cfg.gabab_propagation_strength = float(gabab_prop)
    cfg.gabab_conductance_max = 0.0

    # HALF-2: tag the held-goal->value synapse with a reward-window plasticity gate (None = original scope-all).
    _reward_gate = "reward_window" if vspatch_gate else None

    RS = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name
    cfg.brain_regions = [
        BrainRegion(name="cue", n_neurons=n_cue, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False, izh_neuron_type=RS),
        # PFC maintained-goal pool: recurrence is the nmda_slow SELF-pathway below (NOT internal_density).
        BrainRegion(name="pfc", n_neurons=n_pfc, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False, izh_neuron_type=RS),
        BrainRegion(name="pfc_fs", n_neurons=n_fs, exc_fraction=0.0, internal_density=0.0,
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
                    syn_reversal_potential_i_override=-55.0),
    ]
    cfg.region_pathways = [
        # CS loads the PFC maintained-goal (fixed excitatory).
        RegionPathway(from_region="cue", to_region="pfc", density=0.5,
                      weight_mean=float(cue_to_pfc_weight), weight_jitter=0.1, plastic=False),
        # THE MAINTAINED GOAL: PFC recurrent slow-NMDA self-excitation (the persistence knob `recur`).
        RegionPathway(from_region="pfc", to_region="pfc", density=0.9,
                      weight_mean=pfc_recur_weight, weight_jitter=0.05, plastic=False,
                      exc_receptor="nmda_slow"),
        RegionPathway(from_region="pfc", to_region="pfc_fs", density=0.6,
                      weight_mean=float(pfc_to_fs), weight_jitter=0.1, plastic=False),
        RegionPathway(from_region="pfc_fs", to_region="pfc", density=0.6,
                      weight_mean=float(fs_to_pfc), weight_jitter=0.1, plastic=False),
        # The learned held-goal->value synapse (PLASTIC): only the PFC survives the gap, so credit that BRIDGES
        # the delay must flow through here.
        RegionPathway(from_region="pfc", to_region="striosome_value", density=0.6,
                      weight_mean=float(pfc_to_strio_weight), weight_jitter=0.5, plastic=True,
                      plasticity_gate=_reward_gate),
        # The direct CS->value trace (PLASTIC): present in both arms, but the cue is OFF during the gap so it
        # cannot bridge a long delay on its own.
        RegionPathway(from_region="cue", to_region="striosome_value", density=0.6,
                      weight_mean=float(cue_to_strio_weight), weight_jitter=0.5, plastic=True),
        # US-time critic teacher (fixed): the reward drives the critic to fire so DA-gated eligibility can form
        # with whatever is co-active (the held goal in treatment) -- the innate-reflex-teaches-a-learned-circuit
        # cold-start (N9 A-CSC pattern). Excitatory reward_us->striosome.
        RegionPathway(from_region="reward_us", to_region="striosome_value", density=0.6,
                      weight_mean=float(reward_us_to_strio_weight), weight_jitter=0.2, plastic=False),
        # The synaptic reward r: reward_us -> snc (excitatory, fixed).
        RegionPathway(from_region="reward_us", to_region="snc", density=0.6,
                      weight_mean=float(reward_us_to_snc_weight), weight_jitter=0.2, plastic=False),
        # The value subtraction -V: striosome_value -> snc via slow GABA_B/GIRK.
        RegionPathway(from_region="striosome_value", to_region="snc", density=0.5,
                      weight_mean=float(strio_to_snc_weight), weight_jitter=0.2, plastic=False,
                      receptor="gaba_b"),
    ]
    snc_tonic_firing_fraction = 0.30
    cfg.enable_neuromodulator_subsystem = True
    if da_from_reward_us:
        # HALF-2 reward-time DA: a CLEAN phasic burst from the reward (US) population. baseline=0 with a one-sided
        # from_region_firing production means da_signal (= conc - baseline) is POSITIVE at the US and ~0 otherwise
        # -> the three-factor update (eligibility x da_signal) is LTP-signed at reward. On this substrate the
        # SNc-signed DA sits BELOW its 0.5 baseline at reward (the SNc does not clear its tonic firing threshold),
        # so the SNc-driven da_signal is NEGATIVE and every DA-gated update nets to LTD -- the operating-point trap
        # that produced the original HALF-2 depression. This is dopamine's reward burst (Schultz); the value
        # subtraction -V remains synaptic at the SNc (strio->snc GABA_B), so the critic (HALF-1) is unaffected.
        cfg.neuromodulators = [NeuromodulatorConfig(
            name="dopamine", baseline=0.0, decay_tau_ms=200.0,
            concentration_min=0.0, concentration_max=2.0,
            targets=[ModulatorTarget(target_type="plasticity_rate", scope="all", sensitivity=+1.0)],
            production_rules=[ProductionRule(rule_type="from_region_firing",
                                             sensitivity=float(da_reward_sensitivity),
                                             threshold=float(da_reward_threshold),
                                             window_ms=60.0, source_regions=["reward_us"])])]
    else:
        cfg.neuromodulators = [NeuromodulatorConfig(
            name="dopamine", baseline=0.5, decay_tau_ms=200.0,
            concentration_min=0.0, concentration_max=2.0,
            targets=[ModulatorTarget(target_type="plasticity_rate", scope="all", sensitivity=+1.0)],
            production_rules=[ProductionRule(rule_type="from_region_firing_signed",
                                             sensitivity=float(snc_da_sensitivity),
                                             threshold=float(snc_tonic_firing_fraction),
                                             window_ms=200.0, source_regions=["snc"])])]

    # HALF-2 VSPatch reward-window GATE (additive; only wired when vspatch_gate=True). A phasic modulator whose
    # production reads the firing fraction of the REWARD-window population (default reward_us, the US-encoding
    # population that fires ONLY when the reward is present -- ~0 during the CS/gap/floor, high at the US). Below
    # threshold it stays at baseline 0 (gate SHUT -> pfc->striosome_value updates FROZEN); at the reward burst it
    # accumulates concentration (fast tau) and DRIVES the "reward_window" plasticity gate open (->1). This gates
    # corticostriatal LTP to the reward window ONLY -- the Rubicon/PVLV VSPatch mechanism. The SNc itself is NOT
    # a clean reward-window signal on this substrate (it fires ~0.49 during the gap from the held-goal drive), so
    # the reward-time signal is taken from the US-encoding population. Neural (the gate = a spiking-driven NM
    # concentration) and contingent (no reward -> reward_us silent -> gate stays shut).
    if vspatch_gate:
        _default_thr = 0.10 if vspatch_gate_source == "reward_us" else float(snc_tonic_firing_fraction)
        _gate_thr = float(_default_thr if vspatch_gate_threshold is None else vspatch_gate_threshold)
        cfg.neuromodulators.append(NeuromodulatorConfig(
            name="vspatch_gate", baseline=0.0, decay_tau_ms=float(vspatch_gate_tau_ms),
            concentration_min=0.0, concentration_max=2.0,
            targets=[ModulatorTarget(target_type="plasticity_gate", scope="gate:reward_window",
                                     sensitivity=float(vspatch_gate_target_sensitivity))],
            production_rules=[ProductionRule(rule_type="from_region_firing",
                                             sensitivity=float(vspatch_gate_sensitivity),
                                             threshold=_gate_thr,
                                             window_ms=float(vspatch_gate_tau_ms),
                                             source_regions=[str(vspatch_gate_source)])]))

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


def _clear_conductances(bridge):
    """Between-trial reset: clear the slow conductances (GABA_B tau~150ms, NMDA-rec tau~100ms) that would
    otherwise carry a prior trial's held bump / value subtraction into the next trial (the _d3 re-ignition +
    the limbic order-artifact lessons)."""
    for attr in ("cp_conductance_g_gabab", "cp_conductance_g_nmda_recurrent",
                 "cp_conductance_g_nmda_recurrent_rise", "cp_conductance_g_e", "cp_conductance_g_i",
                 "cp_conductance_g_nmda", "cp_conductance_g_nmda_rise"):
        arr = getattr(bridge, attr, None)
        if arr is not None:
            arr[:] = 0.0


def _run_window(bridge, idx_map, drives, n_steps, xp, *, freeze_lr=None, cfg=None, record=()):
    """Set per-region external current, step n_steps, return {region: mean_hz} for `record` regions."""
    bridge.cp_external_input_current[:] = 0.0
    for region, pA in drives.items():
        bridge.cp_external_input_current[idx_map[region]] = xp.float32(pA)
    saved_lr = None
    if freeze_lr is not None and cfg is not None:
        saved_lr = cfg.reward_learning_rate; cfg.reward_learning_rate = float(freeze_lr)
    spk = {r: 0 for r in record}
    for _ in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)
        for r in record:
            spk[r] += int(bridge.cp_firing_states[idx_map[r]].sum())
    if saved_lr is not None:
        cfg.reward_learning_rate = saved_lr
    dur_s = n_steps * 1e-3
    return {r: spk[r] / max(len(_host(idx_map[r])), 1) / dur_s for r in record}


def _assert_zero_drive_during_gap(bridge, idx_map, xp):
    """ANTI-CHEAT (a): during the GAP the external input to cue AND pfc is identically ZERO. The bridge's
    _run_window zeroes cp_external_input_current then sets only the given drives; the GAP passes only
    {'snc': tonic}. Assert cue+pfc external current is exactly 0 so nothing host-side is holding the goal."""
    import numpy as np
    cue_i = _host(idx_map["cue"]); pfc_i = _host(idx_map["pfc"])
    cur = np.asarray(_host(bridge.cp_external_input_current), dtype=np.float64)
    return float(np.abs(cur[cue_i]).max()), float(np.abs(cur[pfc_i]).max())


def run_condition(seed, *, recur, gap_steps, n_train=45, cs_steps=35, us_steps=35, floor_steps=25,
                  probe_steps=35, snc_tonic_pa=220.0, cue_drive_pa=600.0, us_drive_pa=600.0,
                  no_learning=False, unpaired=False, yoke_reward=False, omit_reward=False,
                  verbose=False, **build_kw):
    """One arm: train the trace-conditioning task, then read the acquired value FROZEN.

    Returns the credit-assignment metrics. `no_learning=True` freezes STDP from t0 (the value cannot be learned
    -> the structural floor). `unpaired=True` fires the US at a random offset with no CS->US contingency."""
    from sim.backend import get_backend
    import numpy as np
    xp, _ = get_backend()
    rng = np.random.default_rng(seed)
    bridge, cfg = build_core(seed, recur=recur, **build_kw)
    regs = ("cue", "pfc", "pfc_fs", "striosome_value", "reward_us", "snc")
    idx_map = {n: xp.asarray(_idx(bridge, n)) for n in regs}

    W_floor = {"snc": snc_tonic_pa}
    W_cs = {"cue": cue_drive_pa, "snc": snc_tonic_pa}          # CS: drive cue (loads pfc) -- NO reward
    W_gap = {"snc": snc_tonic_pa}                              # GAP: ONLY tonic (pfc must self-sustain)
    W_us = {"reward_us": us_drive_pa, "snc": snc_tonic_pa}     # US: reward fires -- NO cue drive

    lr0 = 0.0 if no_learning else cfg.reward_learning_rate
    gap_ext_max = 0.0

    def _gate_val():
        # anti-cheat: the reward-window gate value, sampled from the bridge (NEURAL, NM-driven). NaN when the arm
        # has no gate (the scope-all control / floor) -- not used for those.
        try:
            return float(bridge.get_plasticity_gate_value("reward_window"))
        except Exception:
            return float("nan")

    pfc_hold_curve, us_burst_curve, v_probe_curve = [], [], []
    gate_gap_curve, gate_us_curve = [], []
    for t in range(n_train):
        if yoke_reward:
            # CONTINGENCY (yoked): deliver the reward with NO preceding CS/held-goal (floor -> US only). The gate
            # still OPENS at the US burst (SNc above tonic), but the PFC is at rest (never loaded), so the held-
            # goal->value synapse sees no pre/post coincidence -> it must NOT potentiate. The build is IDENTICAL
            # to the treatment, so the frozen readout stays comparable. This separates CREDIT (needs the held
            # goal co-active with reward) from a CLOCK (gate opens on reward regardless).
            _run_window(bridge, idx_map, W_floor, floor_steps, xp, freeze_lr=lr0, cfg=cfg)
            us_rec = _run_window(bridge, idx_map, W_us, us_steps, xp, freeze_lr=lr0, cfg=cfg, record=("snc",))
            gate_us_curve.append(_gate_val())
            us_burst_curve.append(us_rec["snc"]); pfc_hold_curve.append(0.0); v_probe_curve.append(0.0)
            _clear_conductances(bridge)
            continue
        _run_window(bridge, idx_map, W_floor, floor_steps, xp, freeze_lr=lr0, cfg=cfg)
        _run_window(bridge, idx_map, W_cs, cs_steps, xp, freeze_lr=lr0, cfg=cfg)
        # GAP: assert zero external drive to cue+pfc, then step. pfc sustains ONLY via recurrence.
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[idx_map["snc"]] = xp.float32(snc_tonic_pa)
        cmax, pmax = _assert_zero_drive_during_gap(bridge, idx_map, xp)
        gap_ext_max = max(gap_ext_max, cmax, pmax)
        gap_rec = _run_window(bridge, idx_map, W_gap, gap_steps, xp, freeze_lr=lr0, cfg=cfg,
                              record=("pfc", "striosome_value"))
        pfc_hold_curve.append(gap_rec["pfc"])
        gate_gap_curve.append(_gate_val())   # gate at the end of the GAP (no reward): should be SHUT (~0)
        if omit_reward:
            # CONTINGENCY (goal held, reward ABSENT): the held goal is loaded + sustained across the gap exactly
            # as in the paired arm, but NO US is delivered (floor replaces the US window). With the reward-window
            # gate this NEVER opens the gate -> the held-goal->value synapse must NOT potentiate. A Hebbian/clock
            # rule (co-firing alone) WOULD grow value here; a reward-CONTINGENT credit rule must not.
            _run_window(bridge, idx_map, W_floor, us_steps, xp, freeze_lr=lr0, cfg=cfg)
            gate_us_curve.append(_gate_val())
            us_burst_curve.append(0.0); v_probe_curve.append(gap_rec["striosome_value"])
            _clear_conductances(bridge); continue
        if unpaired:
            # break the CS->US contingency: skip the reward on ~half the trials at random (no reliable pairing)
            if rng.random() < 0.5:
                _run_window(bridge, idx_map, W_floor, us_steps, xp, freeze_lr=lr0, cfg=cfg)
                us_burst_curve.append(0.0); v_probe_curve.append(gap_rec["striosome_value"]); continue
        us_rec = _run_window(bridge, idx_map, W_us, us_steps, xp, freeze_lr=lr0, cfg=cfg, record=("snc",))
        gate_us_curve.append(_gate_val())    # gate at the end of the US (reward burst): should be OPEN (>0)
        us_burst_curve.append(us_rec["snc"])
        v_probe_curve.append(gap_rec["striosome_value"])
        _clear_conductances(bridge)

    # ---- FROZEN readout (learning off) ----
    def frozen(drives_seq):
        """Run a sequence of (drives, steps, record) windows with learning frozen; return the last record."""
        _clear_conductances(bridge)
        last = {}
        for drives, steps, rec in drives_seq:
            bridge.cp_external_input_current[:] = 0.0
            if drives is None:  # GAP -- only tonic, assert zero cue/pfc drive
                bridge.cp_external_input_current[idx_map["snc"]] = xp.float32(snc_tonic_pa)
                last = _run_window(bridge, idx_map, {"snc": snc_tonic_pa}, steps, xp,
                                   freeze_lr=0.0, cfg=cfg, record=rec)
            else:
                last = _run_window(bridge, idx_map, drives, steps, xp, freeze_lr=0.0, cfg=cfg, record=rec)
        return last

    # (1) anticipatory VALUE across the gap: CS -> GAP (NO US) -> measure strio in a post-gap expectation window.
    #     THE HEADLINE: does the held goal predict the reward ACROSS the delay?
    v_anticip = frozen([(W_cs, cs_steps, ()), (None, gap_steps, ()),
                        ({"snc": snc_tonic_pa}, probe_steps, ("striosome_value", "pfc"))])
    v_anticip_hz = v_anticip["striosome_value"]; pfc_expect_hz = v_anticip["pfc"]

    # (2) predicted vs unpredicted US burst (the reward becomes predicted iff credit was assigned).
    pred = frozen([(W_cs, cs_steps, ()), (None, gap_steps, ()), (W_us, us_steps, ("snc",))])["snc"]
    unpred = frozen([(W_floor, floor_steps, ()), (W_us, us_steps, ("snc",))])["snc"]
    base = frozen([(W_floor, floor_steps + probe_steps, ("striosome_value",))])["striosome_value"]

    e = slice(0, max(1, n_train // 5)); l = slice(-max(1, n_train // 5), None)
    return {
        "seed": seed, "recur": recur, "gap_steps": gap_steps, "no_learning": no_learning, "unpaired": unpaired,
        "yoke_reward": yoke_reward, "omit_reward": omit_reward,
        "pfc_hold_hz": _st.mean(pfc_hold_curve[l]), "pfc_hold_early_hz": _st.mean(pfc_hold_curve[e]),
        "pfc_expect_hz": pfc_expect_hz,
        "v_anticip_hz": v_anticip_hz, "v_base_hz": base,
        "v_probe_early_hz": _st.mean(v_probe_curve[e]), "v_probe_late_hz": _st.mean(v_probe_curve[l]),
        "us_burst_early_hz": _st.mean([x for x in us_burst_curve[e]]),
        "us_burst_late_hz": _st.mean([x for x in us_burst_curve[l]]),
        "predicted_hz": pred, "unpredicted_hz": unpred,
        "gap_ext_drive_max": gap_ext_max,               # anti-cheat (a): must be 0.0 (no host drive in the gap)
        "host_reward_signal": float(cfg.current_reward_signal),   # anti-cheat (b): must be 0.0
        # HALF-2 VSPatch gate telemetry (NaN when the arm has no gate). The gate must be SHUT in the gap and OPEN
        # at reward -> proof the reward-window gating is neural/temporal, not a host clock.
        "gate_open_gap": (_st.mean(gate_gap_curve[l]) if gate_gap_curve else float("nan")),
        "gate_open_us": (_st.mean(gate_us_curve[l]) if gate_us_curve else float("nan")),
    }


def _mean(rows, k):
    return _st.mean([r[k] for r in rows])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--gap-short", type=int, default=20)
    ap.add_argument("--gap-long", type=int, default=200)
    ap.add_argument("--recur", type=float, default=25.0)
    ap.add_argument("--n-train", type=int, default=45)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.replace(",", " ").split()]

    from tools.lab import attributable_to, assert_backend
    from tools.verdict import Verdict
    assert_backend("numpy", "rate/cheap spiking limbic core")

    print(f"[RUBICON delayed-credit] maintained-goal (recur={args.recur}, nmda_slow) vs decayed-trace (recur=0) "
          f"| gaps: short={args.gap_short} long={args.gap_long} | seeds={seeds}", flush=True)

    arms = {}  # (label) -> list of per-seed rows
    def sweep(label, **kw):
        rows = []
        for s in seeds:
            r = run_condition(s, n_train=args.n_train, **kw)
            rows.append(r)
            print(f"  [{label:22s} seed {s}] pfc_hold={r['pfc_hold_hz']:6.2f}Hz  "
                  f"v_anticip={r['v_anticip_hz']:6.2f}Hz  v_base={r['v_base_hz']:5.2f}Hz  "
                  f"pred/unpred={r['predicted_hz']:5.1f}/{r['unpredicted_hz']:5.1f}Hz  "
                  f"gap_ext={r['gap_ext_drive_max']:.1f}", flush=True)
        arms[label] = rows
        return rows

    # The 2x2: {maintained recur>0, decayed recur=0} x {short gap, long gap} + a no-learning floor + unpaired AC.
    m_long = sweep("maintained/long", recur=args.recur, gap_steps=args.gap_long)
    c_long = sweep("decayed/long",    recur=0.0,        gap_steps=args.gap_long)
    m_short = sweep("maintained/short", recur=args.recur, gap_steps=args.gap_short)
    c_short = sweep("decayed/short",    recur=0.0,        gap_steps=args.gap_short)
    m_long_nolearn = sweep("maintained/long/NL", recur=args.recur, gap_steps=args.gap_long, no_learning=True)
    m_long_unpaired = sweep("maintained/long/UNP", recur=args.recur, gap_steps=args.gap_long, unpaired=True)

    # ---- metrics (means over seeds) ----
    v_m_long = _mean(m_long, "v_anticip_hz"); v_c_long = _mean(c_long, "v_anticip_hz")
    v_m_nl = _mean(m_long_nolearn, "v_anticip_hz"); v_m_unp = _mean(m_long_unpaired, "v_anticip_hz")
    pfc_hold_m = _mean(m_long, "pfc_hold_hz"); pfc_hold_c = _mean(c_long, "pfc_hold_hz")
    base_m = _mean(m_long, "v_base_hz")

    print("\n  --- head-to-head (long gap = the informative window) ---", flush=True)
    frac = attributable_to("anticipatory value across the long delay (maintained vs decayed)", v_m_long, v_c_long)
    print(f"  maintained/long v_anticip = {v_m_long:.2f}Hz | decayed/long = {v_c_long:.2f}Hz "
          f"| no-learn floor = {v_m_nl:.2f}Hz | unpaired = {v_m_unp:.2f}Hz | v_base = {base_m:.2f}Hz", flush=True)
    print(f"  pfc hold across long gap: maintained = {pfc_hold_m:.2f}Hz | decayed(recur=0) = {pfc_hold_c:.2f}Hz",
          flush=True)

    # ---- TWO separable verdicts (the Rubicon mechanism has two halves) ----
    # HALF 1 -- the maintained-goal BRIDGE: does a neural held goal carry value across the delay where the
    #   decayed trace cannot? (structural = measured no-learning, so it is the goal's AVAILABILITY at reward
    #   time, the load-bearing prerequisite for delayed credit).
    # HALF 2 -- LEARNED credit: does DA-gated plasticity POTENTIATE the held-goal->value synapse across the
    #   delay (trained value > the no-learning structural floor)?
    v = Verdict("rubicon delayed-credit (maintained-goal bridge + learned credit)")
    # (a) the maintained goal is NEURAL and recurrence-dependent
    v.control("HALF1 PFC holds the goal across the gap via recurrence (neural, not host)",
              treatment=pfc_hold_m, control=pfc_hold_c)
    v.require("no host drive holds the goal in the gap (gap_ext==0)",
              max(_mean(m_long, "gap_ext_drive_max"), _mean(c_long, "gap_ext_drive_max")) == 0.0, expect=True)
    # (b) the RPE is neural (no host reward scalar)
    v.require("host reward signal is 0 (r is synaptic)", _mean(m_long, "host_reward_signal") == 0.0, expect=True)
    # anti-cheat (c): the maintained goal makes the value AVAILABLE across the long delay; decayed trace does not
    v.control("HALF1 maintained-goal expresses value across the long delay; decayed-trace does not",
              treatment=v_m_nl, control=v_c_long)
    # HALF 2 headline: does LEARNING potentiate the held-goal value above the structural (no-learning) floor?
    v.reaches("HALF2 LEARNED credit potentiates held-goal value above the no-learning floor",
              before=v_m_nl, after=v_m_long)

    bridge_go = ((pfc_hold_m > 3.0) and (pfc_hold_m > 3.0 * max(pfc_hold_c, 1e-6)) and
                 (v_m_nl > max(5.0, 5.0 * max(v_c_long, 1e-6))) and
                 (_mean(m_long, "gap_ext_drive_max") == 0.0) and (_mean(m_long, "host_reward_signal") == 0.0))
    learned_go = (v_m_long > 1.3 * max(v_m_nl, 1e-6)) and (v_m_long > 1.5 * max(v_m_unp, 1e-6))
    go = bool(bridge_go and learned_go)
    result = v.decide(go=go)
    result["bridge_go"] = bool(bridge_go)          # HALF 1: the maintained-goal bridge (Rubicon prerequisite)
    result["learned_credit_go"] = bool(learned_go)  # HALF 2: DA-gated potentiation of the held-goal value
    print(f"\n  HALF1 maintained-goal BRIDGE (held value available across delay, decayed fails): "
          f"{'GO-looking' if bridge_go else 'NO'}", flush=True)
    print(f"  HALF2 LEARNED credit (DA-STDP potentiates > structural floor): "
          f"{'GO-looking' if learned_go else 'NO'}  (trained {v_m_long:.1f}Hz vs no-learn {v_m_nl:.1f}Hz)", flush=True)
    print(f"  OVERALL VERDICT: {result.get('verdict', result.get('status'))}", flush=True)

    # secondary (informational): informative-window check -- decayed should also work at the SHORT gap
    v_m_short = _mean(m_short, "v_anticip_hz"); v_c_short = _mean(c_short, "v_anticip_hz")
    print(f"  [informative-window] SHORT gap: maintained={v_m_short:.2f}Hz decayed={v_c_short:.2f}Hz "
          f"(if decayed works here but not at long gap, the maintained goal -- not the task -- does the bridging)",
          flush=True)

    payload = {
        "arms": arms,
        "metrics": {
            "v_maintained_long": v_m_long, "v_decayed_long": v_c_long,
            "v_maintained_nolearn": v_m_nl, "v_maintained_unpaired": v_m_unp, "v_base": base_m,
            "v_maintained_short": v_m_short, "v_decayed_short": v_c_short,
            "pfc_hold_maintained": pfc_hold_m, "pfc_hold_decayed": pfc_hold_c,
            "attributable_maintained_vs_decayed": frac,
        },
        "seeds": seeds, "gap_short": args.gap_short, "gap_long": args.gap_long, "recur": args.recur,
        **result,
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(payload, open(args.out, "w"), indent=1)
        print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
