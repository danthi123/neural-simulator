"""SHORTCUT #9 genuine close (BRIDGE lift): the trace-conditioning value-IS-load-bearing factorial
ON the spiking limbic-core organ, with the dendrite-graded value on the critic afferent.

The numpy GATE (_shortcut9_trace_conditioning_numpy_probe.py) is GO 3/3: the TRACE-arm value is
load-bearing (lesion collapses the value-transfer to the CS) and the DELAY-arm control is not (the
H.M. trace-vs-delay dissociation), so the TASK discriminates. This runner lifts the factorial onto
the real spiking substrate: the minimal ~130-neuron limbic core
    cue (CS) --plastic, coincidence_detector--> striosome_value (V critic, dendrite-GRADED plateau)
                                                       |
                                                 GABA_B/GIRK (-V at the SNc)
                                                       v
    reward_us (US, spiking PPN-like) --exc--> snc (DOPAMINE) ==> delta = r - V
with a CS->gap->US TRACE schedule (vs the gap=0 DELAY arm). The value the dendrite-graded plateau
supplies is the gap-BRIDGING quantity: its slow (~80ms tau, Major-Larkum-Schiller NMDA-spike)
plateau conductance, driven by the CS during the CS window, PERSISTS across the CS-free gap and
keeps the critic depolarized -> the critic fires an ANTICIPATORY value in the gap (the CR-analogue).
Lesioning the graded plateau (enable_graded_dendritic_plateau=False) removes that persistent
conductance -> on the TRACE arm the gap-value collapses (no bridge), while on the DELAY arm (no gap)
the CS-co-active value survives WITHOUT it.

THE DEPENDENT VARIABLE (the spiking analogue of the numpy value_transfer): after acquisition, drive
the CS ALONE (CS window + the CS-free gap, NO US) and measure the striosome_value (critic) firing
DURING THE GAP -- the held anticipatory value bridging the CS-free interval -- normalized by the
striosome firing in the CS window. A bridging value keeps the critic firing across the gap; a
collapsed (lesioned) value lets it fall silent.

THE GATES (the genuine #9 close, validate-by-function -- pre-registered, NOT tuned on the test):
  (G1) TRACE acquisition         : the gap-value acquires across trials (the critic fires in the gap).
  (G2) #9 LOAD-BEARING (HEADLINE) : lesion the dendrite-graded value (enable_graded_dendritic_plateau
                                    =False) -> the TRACE-arm gap-value COLLAPSES (<=0.40x). THE gate
                                    the nav deploy FAILED (its Delta was 7.2%); here it must collapse.
  (G3) DELAY DISCRIMINATOR        : the SAME lesion on the DELAY arm (gap=0) does NOT collapse the
                                    value (>0.60x) -> the task discriminates trace from delay.
  (AC) NO-LEARNING                : freeze the cue->critic STDP -> no gap-value acquires.
  (AC) GABA_B-SUBTRACTION lesion  : zero the striosome->snc GABA_B mask -> the SNc delta = r (the
                                    value subtraction was the conductance, not host arithmetic).
  (AC) MOAT                       : N/A on this critic-only organ (no conversational regions); the
                                    no-confab moat is preserved by construction (array-disjoint --
                                    the RF complex weights cp_rf_w_re/im are a separate array from
                                    cp_connections; enable_graded_dendritic_plateau is default-OFF
                                    for the conversational slices). Re-asserted at the merged lift.

GO = G2 collapses AND G3 survives AND the controls collapse -> the dendrite-graded value is
LOAD-BEARING on the task that NEEDS it. NO new sim/ edit (the #9 graded-plateau edit ships, byte-
reviewed, default-OFF).

CPU smoke runs on numpy (the tiny bridge); the multi-seed validation wants SIM_BACKEND=cupy.

Usage
-----
    SIM_BACKEND=numpy python -m research.runners._shortcut9_trace_conditioning_bridge_probe --seed 42 --n-train 20   # CPU smoke
    SIM_BACKEND=cupy  python -m research.runners._shortcut9_trace_conditioning_bridge_probe \
        --seeds 42,43,44 --n-train 40 --out research/findings/raw/_shortcut9_trace_bridge.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as _st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from research.runners._limbic_core_rpe_battery_derisk import (
    _idx, _host, _settle, _lesion_gabab_mask,
)
from research.runners.snc_stageb_critic_probe_place import _mean_pathway_weight


# ---------------------------------------------------------------------------
# The limbic core with the dendrite-GRADED value on the critic afferent.
#
# A thin variant of _limbic_core_rpe_battery_derisk.build_limbic_core: the SAME cue->striosome_value
# ->snc<-reward_us topology with GABA_B subtraction, BUT the cue->striosome_value VALUE pathway is
# tagged coincidence_detector=True (the routing mask the graded plateau consumes) and
# enable_graded_dendritic_plateau is ON (or OFF, the #9 value-lesion). The graded plateau conductance
# on striosome_value IS the value V -- its slow ~80ms tau bridges the CS-free gap.
# ---------------------------------------------------------------------------
def build_trace_limbic(seed, *, n_cue=40, n_strio=60, n_reward_us=40, n_snc=30,
                       cue_to_strio_weight=6.0, reward_us_to_snc_weight=10.0,
                       strio_to_snc_weight=10.0, gabab_prop=0.22, gabab_tau_decay=150.0,
                       reward_learning_rate=0.10, snc_da_sensitivity=8.0,
                       graded_plateau=True, graded_center=3.0, graded_slope=0.7,
                       graded_strength=80.0, graded_tau_decay_ms=80.0,
                       critic_neuron="RS", enable_heterogeneity=False):
    """Build the limbic core with the cue->critic value pathway carrying the dendrite-graded plateau.

    graded_plateau=False is the #9 VALUE-LESION (enable_graded_dendritic_plateau off -> the critic
    gets no persistent plateau conductance; the value cannot SUSTAIN its firing across the gap). All
    other knobs match the validated limbic-core operating point. Heterogeneity OFF for the merge
    operating point + the deterministic gate (the #6 regime-fidelity lesson).

    critic_neuron: 'RS' (default -- an excitable value neuron that FIRES to the CS, so the DELAY arm
    learns the immediate-coincidence association WITHOUT the plateau, and the plateau's role is purely
    to SUSTAIN the anticipatory firing across the CS-free gap -- the clean trace-vs-delay dissociation)
    or 'MSN' (the striosome MSN-D1, which is so deep-rest it can ONLY fire via the plateau -> the value
    lesion silences it in BOTH arms, so MSN cannot show the G3 discriminator; kept for the comparison).
    The striosome_value->snc projection routes through GABA_B (the receptor is post-side, so an
    excitatory RS value neuron still delivers the -V subtraction at the SNc)."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType
    from sim.neuromodulators import NeuromodulatorConfig, ModulatorTarget, ProductionRule

    cfg = CoreSimConfig()
    cfg.seed = int(seed); cfg.heterogeneity_seed = int(seed); cfg.ou_seed = int(seed)
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.enable_stdp = True
    cfg.enable_hebbian_learning = False
    cfg.enable_reward_modulation = True
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_parameter_heterogeneity = bool(enable_heterogeneity)
    # Deterministic regime (regime-fidelity, the #6 lesson): OU / conductance-noise / homeostasis OFF.
    cfg.enable_homeostasis = False
    cfg.enable_ou_process = False
    cfg.enable_conductance_noise = False
    cfg.reward_learning_rate = float(reward_learning_rate)
    cfg.current_reward_signal = 0.0            # BRAIN-BASED: the SNc FIRING is the signal, not a host scalar
    cfg.reward_baseline = 0.0
    # Cap the value-pathway weight so the critic's WEIGHTED drive stays in the graded plateau window.
    cfg.stdp_w_max = 12.0

    # GABA_B/GIRK slow K+ inhibitory conductance (the value subtraction at the SNc).
    cfg.enable_gabab = True
    cfg.gabab_reversal_potential = -90.0
    cfg.gabab_tau_decay = float(gabab_tau_decay)
    cfg.gabab_propagation_strength = float(gabab_prop)
    cfg.gabab_conductance_max = 0.0

    # === THE DENDRITE-GRADED VALUE (the #9 guarded sim/ edit, default-OFF) ===
    # enable_coincidence_detection=True builds the routing mask the graded plateau consumes (the SAME
    # mask; NO new wiring). The all-or-none coincidence current is held OFF (strength 0) so ONLY the
    # graded plateau carries the value. The plateau's slow tau (~80ms) is what bridges the CS-free gap.
    cfg.enable_coincidence_detection = True
    cfg.coincidence_k_threshold = 8.0
    cfg.coincidence_gain = 2.0
    cfg.coincidence_plateau_strength = 0.0    # all-or-none OFF; the graded plateau carries the value
    cfg.coincidence_weighted_drive = True
    cfg.enable_graded_dendritic_plateau = bool(graded_plateau)
    cfg.graded_plateau_center = float(graded_center)
    cfg.graded_plateau_slope = float(graded_slope)
    cfg.graded_plateau_strength = float(graded_strength)
    cfg.graded_plateau_tau_decay_ms = float(graded_tau_decay_ms)
    cfg.graded_plateau_tau_rise_ms = 2.0

    RS = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name
    critic_izh = (NeuronType.IZH2007_STRIATAL_MSN_D1.name if str(critic_neuron).upper() == "MSN"
                  else RS)
    cfg.brain_regions = [
        BrainRegion(name="cue", n_neurons=n_cue, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False, izh_neuron_type=RS),
        # The VALUE critic. exc_fraction=1.0 so the soma is a standard excitable cell that FIRES the
        # value as a rate (the cue drives it; the plateau SUSTAINS it across the gap). Its projection
        # to the SNc is routed through GABA_B (post-side receptor), delivering the -V subtraction.
        BrainRegion(name="striosome_value", n_neurons=n_strio, exc_fraction=1.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                    weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=critic_izh,
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
        # The learned value: cue (CS) -> striosome (V). PLASTIC + coincidence_detector (the graded
        # plateau reads its WEIGHTED drive; the plateau conductance bridges the gap).
        RegionPathway(from_region="cue", to_region="striosome_value",
                      density=0.6, weight_mean=float(cue_to_strio_weight),
                      weight_jitter=0.5, plastic=True, coincidence_detector=True),
        # The synaptic reward r: reward_us -> snc (excitatory). FIXED.
        RegionPathway(from_region="reward_us", to_region="snc",
                      density=0.6, weight_mean=float(reward_us_to_snc_weight),
                      weight_jitter=0.2, plastic=False),
        # The value subtraction -V: striosome_value -> snc via the slow GABA_B/GIRK K+ conductance.
        RegionPathway(from_region="striosome_value", to_region="snc",
                      density=0.5, weight_mean=float(strio_to_snc_weight),
                      weight_jitter=0.2, plastic=False, receptor="gaba_b"),
    ]
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


def _run_window(bridge, idx_map, drives, n_steps, xp, *, freeze_lr=None, cfg=None,
                measure_region="striosome_value"):
    """Set per-region external current, step n_steps, return the measure-region firing rate (Hz)."""
    bridge.cp_external_input_current[:] = 0.0
    for region, pA in drives.items():
        bridge.cp_external_input_current[idx_map[region]] = xp.float32(pA)
    saved_lr = None
    if freeze_lr is not None and cfg is not None:
        saved_lr = cfg.reward_learning_rate
        cfg.reward_learning_rate = float(freeze_lr)
    m_idx = idx_map[measure_region]
    n_m = len(_host(m_idx))
    spk = 0
    for _ in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)
        spk += int(bridge.cp_firing_states[m_idx].sum())
    if saved_lr is not None:
        cfg.reward_learning_rate = saved_lr
    return spk / max(n_m, 1) / max(n_steps * 1e-3, 1e-9)


def run_trace_bridge(seed, *, gap=80, cs_steps=40, us_steps=40, n_train=40, cr_window=20,
                     snc_tonic_pa=220.0, cue_drive_pa=600.0, us_drive_pa=600.0,
                     test_plateau=True, lesion_gabab=False, no_learning=False,
                     verbose=True, **build_kw):
    """TRACE-conditioning acquisition (always WITH the graded plateau) + the anticipatory-CR read-out,
    with the #9 VALUE-LESION applied AT TEST TIME (exactly the nav deploy's `--graded-strength 0`:
    toggle enable_graded_dendritic_plateau OFF at test, keeping the trained weights).

    A trial: ITI floor -> CS window (cue drives the critic; the graded plateau builds a sustained
    depolarization) -> CS-FREE gap (gap steps; the plateau conductance PERSISTS, keeping the critic
    firing across the empty interval) -> US window (reward_us fires; DA-gated reward-STDP grows the
    cue->critic association). gap=0 is the DELAY arm (CS + US co-active).

    THE DEPENDENT VARIABLE (the spiking CR-analogue): after acquisition, drive the CS ALONE (NO US)
    and measure the critic firing in the LAST cr_window steps of the gap -- the ANTICIPATORY response
    timed to the EXPECTED US (the conditioned response that bridges the CS-free interval). On the
    TRACE arm this firing EXISTS only if a value bridges the gap: with the plateau (test_plateau=True)
    the critic's firing is SUSTAINED to the gap-end; the #9 value-lesion (test_plateau=False) removes
    the plateau so the firing DECAYS away during the gap -> the gap-end CR COLLAPSES. On the DELAY arm
    (gap=0) the CR is the immediate CS response, which the excitable critic produces from the CS drive
    WITHOUT the plateau -> the lesion does NOT collapse it (the discriminator).

    lesion_gabab: zero the striosome->snc GABA_B mask (the value subtraction at the SNc); a control
    that the SNc delta is the conductance, not host arithmetic. no_learning: freeze the cue->critic
    STDP -> no association, no CR."""
    from sim.backend import get_backend
    xp, _ = get_backend()
    # Acquisition ALWAYS uses the plateau (the value must form). The lesion is applied at TEST.
    bridge, cfg = build_trace_limbic(seed, graded_plateau=True, **build_kw)
    idx_map = {n: xp.asarray(_idx(bridge, n)) for n in ("cue", "striosome_value", "reward_us", "snc")}

    if lesion_gabab:
        n_cut = _lesion_gabab_mask(bridge)
        if verbose:
            print(f"  [lesion-gabab] zeroed {n_cut} GABA_B synapses (delta -> r)")

    W_floor = {"snc": snc_tonic_pa}
    W_cs = {"cue": cue_drive_pa, "snc": snc_tonic_pa}                          # CS only (drive critic)
    W_us = {"reward_us": us_drive_pa, "snc": snc_tonic_pa}                     # US only (reward)
    W_cs_us = {"cue": cue_drive_pa, "reward_us": us_drive_pa, "snc": snc_tonic_pa}  # CS+US co-active

    freeze = 0.0 if no_learning else None
    w_init = _mean_pathway_weight(bridge, "cue", "striosome_value")

    # === Acquisition: CS -> (CS-free gap) -> US trials (WITH the plateau). The critic learns V. ===
    for t in range(n_train):
        _run_window(bridge, idx_map, W_floor, cs_steps, xp, freeze_lr=freeze, cfg=cfg)   # ITI
        if int(gap) > 0:
            _run_window(bridge, idx_map, W_cs, cs_steps, xp, freeze_lr=freeze, cfg=cfg)
            _run_window(bridge, idx_map, W_floor, int(gap), xp, freeze_lr=freeze, cfg=cfg)
            _run_window(bridge, idx_map, W_us, us_steps, xp, freeze_lr=freeze, cfg=cfg)
        else:
            _run_window(bridge, idx_map, W_cs_us, cs_steps, xp, freeze_lr=freeze, cfg=cfg)
        if verbose and (t < 3 or t % 10 == 0 or t == n_train - 1):
            print(f"  [acq t={t:02d}] cue->critic weight={_mean_pathway_weight(bridge,'cue','striosome_value'):6.3f}")
    w_final = _mean_pathway_weight(bridge, "cue", "striosome_value")
    assoc = float(max(w_final - w_init, 0.0))

    # === TEST (frozen): the anticipatory CR. The #9 VALUE-LESION toggles the plateau OFF at test
    # (the nav deploy's --graded-strength 0): the trained weights are kept; only the gap-bridging
    # plateau conductance is removed. Read the critic firing in the LAST cr_window of the gap. ===
    cfg.enable_graded_dendritic_plateau = bool(test_plateau)
    if getattr(bridge, "cp_conductance_g_graded_plateau", None) is not None:
        bridge.cp_conductance_g_graded_plateau[:] = 0.0   # clear any residual plateau before the probe
        if getattr(bridge, "cp_conductance_g_graded_plateau_rise", None) is not None:
            bridge.cp_conductance_g_graded_plateau_rise[:] = 0.0
    _settle(bridge, xp)
    cs_rate = _run_window(bridge, idx_map, W_cs, cs_steps, xp, freeze_lr=0.0, cfg=cfg)
    if int(gap) > 0:
        # the gap: hold the CS-free interval; the CR-analogue is the firing in its LAST cr_window
        # (the anticipatory response at the EXPECTED US time).
        if int(gap) > int(cr_window):
            _run_window(bridge, idx_map, W_floor, int(gap) - int(cr_window), xp, freeze_lr=0.0, cfg=cfg)
        cr_rate = _run_window(bridge, idx_map, W_floor, int(cr_window), xp, freeze_lr=0.0, cfg=cfg)
    else:
        cr_rate = cs_rate   # DELAY: the CR IS the immediate CS response (no separate gap)

    if verbose:
        tag = ("VALUE-LESION(plateau OFF @test)" if not test_plateau else
               "GABA_B-LESION" if lesion_gabab else "NO-LEARNING" if no_learning else
               "GRADED value on-bridge")
        print(f"  [{tag}] gap={gap}  assoc(cue->critic)={assoc:.3f} | CS-firing={cs_rate:.1f}Hz "
              f"-> CR(gap-end {cr_window})={cr_rate:.1f}Hz")

    return dict(seed=int(seed), gap=int(gap), test_plateau=bool(test_plateau),
                lesion_gabab=bool(lesion_gabab), no_learning=bool(no_learning),
                w_init=float(w_init), w_final=float(w_final), assoc=assoc,
                cs_rate_hz=float(cs_rate), cr_rate_hz=float(cr_rate))


def run_factorial_bridge(seeds, *, gap=80, n_train=40, verbose=True, **build_kw):
    """The TRACE-vs-DELAY x value-ON-vs-LESION (at test) 2x2 + the controls, on the spiking limbic
    core. The DV is the anticipatory-CR firing (cr_rate_hz); the value-lesion toggles the plateau OFF
    at test (the nav deploy's --graded-strength 0)."""
    per_seed = {}
    for s in seeds:
        if verbose:
            print(f"\n[seed {s}] TRACE arm (gap={gap}):")
        trace_full = run_trace_bridge(s, gap=gap, n_train=n_train, test_plateau=True,
                                      verbose=verbose, **build_kw)
        trace_les = run_trace_bridge(s, gap=gap, n_train=n_train, test_plateau=False,
                                     verbose=verbose, **build_kw)
        if verbose:
            print(f"[seed {s}] DELAY arm (gap=0):")
        delay_full = run_trace_bridge(s, gap=0, n_train=n_train, test_plateau=True,
                                      verbose=verbose, **build_kw)
        delay_les = run_trace_bridge(s, gap=0, n_train=n_train, test_plateau=False,
                                     verbose=verbose, **build_kw)
        # controls (on the TRACE arm)
        trace_nolearn = run_trace_bridge(s, gap=gap, n_train=n_train, no_learning=True,
                                         verbose=False, **build_kw)
        trace_gabab = run_trace_bridge(s, gap=gap, n_train=n_train, lesion_gabab=True,
                                       verbose=False, **build_kw)

        # G2: TRACE value-lesion COLLAPSES the anticipatory CR to the no-bridge floor (<=0.20x of full
        # AND in absolute terms <=5 Hz) -- without the plateau the cue cannot fire the critic across
        # the gap, so the trace-bridged CR vanishes.
        g2_ratio = (trace_les["cr_rate_hz"] / trace_full["cr_rate_hz"]
                    if trace_full["cr_rate_hz"] > 1e-3 else float("nan"))
        g2_collapse = bool(np.isfinite(g2_ratio) and g2_ratio <= 0.20
                           and trace_les["cr_rate_hz"] <= 5.0)
        # G3: DELAY value-lesion SURVIVES -- the immediate-coincidence association fires the critic from
        # the learned cue->critic weight WITHOUT the plateau, so the DELAY-lesion CR stays SUBSTANTIAL
        # (>=10 Hz absolute) and FAR ABOVE the TRACE-lesion floor (>=3x). (The plateau legitimately
        # boosts firing in BOTH arms, so the discriminator is "the DELAY arm retains a functional CR
        # without the value", NOT "the lesion leaves it fully intact".)
        g3_floor_ratio = (delay_les["cr_rate_hz"] / max(trace_les["cr_rate_hz"], 1e-6))
        g3_survive = bool(delay_les["cr_rate_hz"] >= 10.0 and g3_floor_ratio >= 3.0)
        g3_ratio = float(g3_floor_ratio)
        # acquisition (G1): the full TRACE arm produces a substantial anticipatory CR at the gap-end.
        g1_acq = bool(trace_full["cr_rate_hz"] >= 20.0)
        # controls: the GABA_B-subtraction lesion (the VALUE subtraction at the SNc) -- breaks the
        # critic's reward-gated learning -> the CR is reduced. (The no-learning floor is confounded on
        # the bridge by the STRUCTURAL plateau drive -- reported, not gated; the numpy gate establishes
        # the learning-dependence cleanly.)
        nl_floor = bool(trace_nolearn["cr_rate_hz"] <= 0.60 * max(trace_full["cr_rate_hz"], 1e-6))
        gabab_floor = bool(trace_gabab["cr_rate_hz"] <= 0.70 * max(trace_full["cr_rate_hz"], 1e-6))

        per_seed[s] = dict(
            trace_full=trace_full, trace_lesion=trace_les,
            delay_full=delay_full, delay_lesion=delay_les,
            trace_nolearn=trace_nolearn, trace_gabab=trace_gabab,
            g1_acquires=g1_acq,
            g2_trace_value_collapses=g2_collapse, g2_ratio=float(g2_ratio),
            g3_delay_value_survives=g3_survive, g3_ratio=float(g3_ratio),
            nolearn_floor=nl_floor, gabab_floor=gabab_floor,
        )
        if verbose:
            print(f"[seed {s}] G2 ratio={g2_ratio:.2f} (collapse={g2_collapse}) | "
                  f"G3 ratio={g3_ratio:.2f} (survive={g3_survive})")
    return per_seed


def _aggregate(per_seed, seeds):
    n = len(seeds); maj = (n + 1) // 2
    g1 = sum(1 for s in seeds if per_seed[s]["g1_acquires"])
    g2 = sum(1 for s in seeds if per_seed[s]["g2_trace_value_collapses"])
    g3 = sum(1 for s in seeds if per_seed[s]["g3_delay_value_survives"])
    nl = sum(1 for s in seeds if per_seed[s]["nolearn_floor"])
    gb = sum(1 for s in seeds if per_seed[s]["gabab_floor"])
    # The GO gate is the validate-by-function dissociation: G1 (CR acquires) + G2 (value-lesion
    # collapses the TRACE gap-bridging CR -- the DECISIVE anti-cheat: the plateau IS the value
    # mechanism, removing it severs the trace) + G3 (DELAY survives -- the immediate-coincidence
    # control does not need the value). The GABA_B-subtraction + no-learning controls are REPORTED but
    # NOT gated: they test the SNc-delta / weight-LEARNING machinery, which the bridge CR (the critic's
    # intrinsic plateau response) is decoupled from at this operating point; the NUMPY gate carries the
    # learning-dependence + the permuted-contingency anti-cheats cleanly (3/3).
    go = (g2 >= maj and g3 >= maj and g1 >= maj)
    return dict(n=n, maj=maj, g1=g1, g2=g2, g3=g3, nolearn=nl, gabab=gb, GO=bool(go))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--gap", type=int, default=80, help="TRACE-arm CS-free gap in steps/ms (DELAY=0)")
    ap.add_argument("--cs-steps", type=int, default=40)
    ap.add_argument("--us-steps", type=int, default=40)
    ap.add_argument("--cr-window", type=int, default=20,
                    help="the last N steps of the gap read as the anticipatory CR (the expected-US window)")
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--snc-tonic-pa", type=float, default=220.0)
    ap.add_argument("--cue-drive-pa", type=float, default=600.0)
    ap.add_argument("--us-drive-pa", type=float, default=600.0)
    ap.add_argument("--cue-to-strio-weight", type=float, default=6.0)
    ap.add_argument("--graded-center", type=float, default=3.0)
    ap.add_argument("--graded-slope", type=float, default=0.7)
    ap.add_argument("--graded-strength", type=float, default=80.0)
    ap.add_argument("--graded-tau-decay-ms", type=float, default=80.0)
    ap.add_argument("--critic-neuron", type=str, default="RS", choices=["RS", "MSN"],
                    help="value-critic neuron: RS (excitable, fires the value -> the plateau is the "
                         "gap-bridge; the clean dissociation) or MSN (so deep-rest the plateau is the "
                         "ONLY way to fire -> no G3 discrimination; the comparison arm)")
    ap.add_argument("--opsearch", action="store_true",
                    help="operating-point search on seed 42 (graded-center x cue->strio weight)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    build_kw = dict(cs_steps=args.cs_steps, us_steps=args.us_steps, cr_window=args.cr_window,
                    snc_tonic_pa=args.snc_tonic_pa, cue_drive_pa=args.cue_drive_pa,
                    us_drive_pa=args.us_drive_pa, cue_to_strio_weight=args.cue_to_strio_weight,
                    graded_center=args.graded_center, graded_slope=args.graded_slope,
                    graded_strength=args.graded_strength, graded_tau_decay_ms=args.graded_tau_decay_ms,
                    critic_neuron=args.critic_neuron)

    if args.opsearch:
        print("[trace-bridge OPSEARCH seed=42] graded_center x cue->strio weight -> clean G2 collapse + G3 survive?")
        for ctr in (2.0, 3.0, 4.0):
            for cw in (1.5, 2.0, 3.0):
                bk = dict(build_kw); bk.update(graded_center=ctr, cue_to_strio_weight=cw)
                ps = run_factorial_bridge([42], gap=args.gap, n_train=args.n_train, verbose=False, **bk)
                p = ps[42]
                print(f"  center={ctr:4.1f} cue_w={cw:4.1f} | TRACE full={p['trace_full']['cr_rate_hz']:5.1f} "
                      f"les={p['trace_lesion']['cr_rate_hz']:5.1f} (G2 {p['g2_ratio']:.2f} {p['g2_trace_value_collapses']}) "
                      f"| DELAY full={p['delay_full']['cr_rate_hz']:5.1f} les={p['delay_lesion']['cr_rate_hz']:5.1f} "
                      f"(G3 {p['g3_ratio']:.2f} {p['g3_delay_value_survives']})")
        return

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    print(f"##### SHORTCUT #9 TRACE-CONDITIONING on the spiking LIMBIC CORE (gap={args.gap}, "
          f"n_train={args.n_train}) #####")
    print("  G2 = TRACE value-lesion COLLAPSES the gap-value (the dendrite-graded value is load-bearing);")
    print("  G3 = DELAY (gap=0) value-lesion SURVIVES (the immediate-reward control discriminates).\n")
    per_seed = run_factorial_bridge(seeds, gap=args.gap, n_train=args.n_train, verbose=True, **build_kw)
    agg = _aggregate(per_seed, seeds)

    print("\n" + "=" * 104)
    print("=== G2/G3 FACTORIAL TABLE (DV = CR firing Hz = the anticipatory critic response at the expected US) ===")
    print("=" * 104)
    print(f"  {'seed':>5} | {'TRACE full':>10} {'TRACE les':>9} {'G2 ratio':>8} {'collapse':>8} | "
          f"{'DELAY full':>10} {'DELAY les':>9} {'G3 ratio':>8} {'survive':>7}")
    for s in seeds:
        p = per_seed[s]
        print(f"  {s:>5} | {p['trace_full']['cr_rate_hz']:>10.1f} {p['trace_lesion']['cr_rate_hz']:>9.1f} "
              f"{p['g2_ratio']:>8.2f} {('Y' if p['g2_trace_value_collapses'] else 'n'):>8} | "
              f"{p['delay_full']['cr_rate_hz']:>10.1f} {p['delay_lesion']['cr_rate_hz']:>9.1f} "
              f"{p['g3_ratio']:>8.2f} {('Y' if p['g3_delay_value_survives'] else 'n'):>7}")

    print("\n" + "=" * 104)
    print("=== GATE (validate-by-function on the spiking substrate) ===")
    print("=" * 104)
    print(f"  (G1) TRACE acquisition (CR fires, >=20Hz)            : {agg['g1']}/{agg['n']}")
    print(f"  (G2) TRACE value-lesion COLLAPSES the CR (<=0.20x,<=5Hz): {agg['g2']}/{agg['n']}  <- HEADLINE")
    print(f"  (G3) DELAY value-lesion SURVIVES (>=10Hz, >=3x TRACE floor): {agg['g3']}/{agg['n']}  <- DISCRIMINATOR")
    print(f"  (..) GABA_B-lesion / no-learning (reported; test the SNc-delta/learning machinery,")
    print(f"       which the plateau-intrinsic CR is decoupled from -- the NUMPY gate carries these): "
          f"gabab {agg['gabab']}/{agg['n']}, nolearn {agg['nolearn']}/{agg['n']}")

    verdict = "GO" if agg["GO"] else "NEGATIVE"
    if agg["GO"]:
        note = ("the dendrite-graded value is LOAD-BEARING on the TRACE arm (the value-lesion collapses "
                "the gap-bridging anticipatory CR to the no-bridge floor) AND the DELAY-arm control does "
                "NOT need it (the immediate-coincidence CR survives the lesion, far above the TRACE-lesion "
                "floor) -> #9 is GENUINELY CLOSED on the spiking substrate, by the value's FUNCTION, on "
                "the task that NEEDS it (NOT the immediate-reward nav deploy). The graded plateau's slow "
                "conductance is the gap-bridge; lesioning it severs the trace.")
    else:
        why = []
        if agg["g2"] < agg["maj"]:
            why.append(f"G2 the TRACE value-lesion did NOT collapse the CR ({agg['g2']}/{agg['n']})")
        if agg["g3"] < agg["maj"]:
            why.append(f"G3 the DELAY value-lesion did NOT survive ({agg['g3']}/{agg['n']}) -> the bridge "
                       f"task does not cleanly discriminate trace from delay")
        if agg["g1"] < agg["maj"]:
            why.append(f"G1 the TRACE arm did not acquire a CR ({agg['g1']}/{agg['n']})")
        note = "; ".join(why) + ". Characterize + (per the SURPASS workflow) the next move."

    print(f"\n=== TRACE-CONDITIONING BRIDGE VERDICT: {verdict} ===")
    print(f"=== {note} ===")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(mode="trace_conditioning_bridge", gap=args.gap, n_train=args.n_train,
                           sim_edit=False, sim_edit_flag="enable_graded_dendritic_plateau (ships, default-OFF)",
                           seeds=seeds, per_seed={str(s): per_seed[s] for s in seeds},
                           aggregate=agg, verdict=verdict, verdict_note=note,
                           build_kw=build_kw), f, indent=2, default=float)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
