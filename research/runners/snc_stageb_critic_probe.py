"""Spiking-SNc Stage B — neural striosome value-critic de-risk (CS-gated reward prediction).

Stage A made the dopamine RPE the SNc's FIRING (delta = r - V), but V was still the
HOST reward-EMA scaffold. Stage B replaces it with a SPIKING striosome critic: a
GABAergic `striosome_value` population, driven by a cue (CS) through PLASTIC synapses
trained by the SNc's own dopamine delta, projecting inhibition to the SNc so the
subtraction r - V happens at the SNc MEMBRANE (no host reads V). Zero new protected
sim/ edits — rides the existing three-factor pipeline (eligibility from STDP co-firing,
the SNc-derived da_signal as the teaching factor, per-region inhibitory reversal).

HONEST SCOPING — Rescorla-Wagner vs Temporal-Difference
-------------------------------------------------------
The minimal membrane scheme I_snc = tonic + k_r*max(0,r) - inhibition(V) implements
Rescorla-Wagner (delta = r - V), NOT the temporal-difference delta = r + gamma*V(s')
- V(s). R-W produces the US-burst-SHRINK and the OMISSION-DIP (and a DIP, not a burst,
at the CS). The full Schultz burst-MIGRATION onto the CS needs the TD bootstrap (a
delayed value derivative) — a deeper, LATER increment. So this cheap-first de-risk
tests the R-W-achievable AND host-EMA-IMPOSSIBLE signature: CS-GATED reward prediction,
i.e. the value is NEURAL, STATE-SPECIFIC, and LEARNED. Four checks:

  (1) V-LEARNED        — the striosome firing on the CS RISES across training.
  (2) US-BURST-SHRINK  — the reward burst SHRINKS across training as V cancels r.
  (3) STATE-SPECIFIC   — (the host-EMA discriminator) a trained CS predicts the reward
                         (small burst), but the SAME reward WITHOUT the CS is
                         unpredicted (big burst). A host GLOBAL-EMA value gives the
                         same V regardless of the cue, so it CANNOT produce this gap.
  (4) OMISSION-DIP     — CS but no reward -> SNc dips below its tonic baseline.

  (+) LESION anti-cheat (--lesion) — after training, zero the striosome_value->snc
      weights: the prediction VANISHES (predicted == unpredicted, no dip). Proves the
      subtraction is the striosome FIRING -> GABA current, not a host formula in
      disguise. (The `unpredicted` condition above is already a functional per-trial
      lesion of the cue->striosome drive; --lesion cuts the conduit with the cue present.)

CPU-friendly (tiny bridge): run under SIM_BACKEND=numpy.

Usage
-----
    SIM_BACKEND=numpy python -m research.runners.snc_stageb_critic_probe --seed 42
    SIM_BACKEND=numpy python -m research.runners.snc_stageb_critic_probe --seed 42 --lesion
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as _st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _build_stageb_bridge(seed, *, snc_da_sensitivity=8.0, reward_learning_rate=0.08,
                         cue_to_strio_weight=3.0, strio_to_snc_weight=2.5,
                         n_cue=40, n_strio=60, n_snc=30,
                         bprime=False, snc_drive_to_snc_weight=6.0,
                         strio_to_drive_weight=15.0, n_drive=40,
                         bprime_snr=False, strio_to_disinhib_weight=20.0,
                         disinhib_to_gaba_weight=20.0, gaba_to_snc_weight=6.0, n_relay=40,
                         gabab=False, gabab_tau_decay=150.0, gabab_propagation_strength=0.105):
    """Minimal bridge: cue (CS) -> striosome_value (GABAergic critic) -> snc (DA).

    cue->striosome_value is PLASTIC (the value is learned by the SNc-derived delta via
    the three-factor pipeline). striosome_value->snc is fixed inhibitory (the value
    subtraction at the SNc membrane). The dopamine modulator reads the snc firing via
    `from_region_firing_signed` so da_signal = da_conc - baseline IS the spiking delta
    the reward-modulation block consumes (sim/bridge.py:5926-5953).

    gabab=True routes the striosome_value->snc inhibition through the NEW slow GABA_B/GIRK
    conductance (E_K=-90 mV, tau~150 ms) instead of weak GABA_A onto the depolarized SNc
    (E_GABA=-55 mV). The SNc KEEPS its GABA_A reversal (-55 mV, unchanged); the GABA_B
    current is a SEPARATE, parallel hyperpolarizing K+ term on the same SNc neurons. This
    is the protected-edit de-risk: the GABA_A direct path failed the state-specific gap 0/3
    because the depolarized reversal makes GABA weak/shunting; the K+ reversal (independent
    of the chloride gradient) should subtract value strongly + sign-correctly.
    """
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType
    from sim.neuromodulators import NeuromodulatorConfig, ModulatorTarget, ProductionRule

    cfg = CoreSimConfig()
    # Harness fix #5 (2026-06-08): PIN the bridge RNG to `seed`. Without this, cfg.seed
    # stays -1 -> _initialize_rng time-seeds the bridge, so connectivity/heterogeneity vary
    # run-to-run and the SAME --seed gives different dynamics each invocation (a multi-seed
    # verdict becomes noise). Setting cfg.seed (+ het/ou seeds) makes each --seed reproducible
    # so the GABA_B gap is a genuine per-seed result, not a per-process lottery.
    cfg.seed = int(seed)
    cfg.heterogeneity_seed = int(seed)
    cfg.ou_seed = int(seed)
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    # The critic LEARNS: STDP supplies eligibility (pre/post co-firing), reward
    # modulation converts eligibility -> weight change via the SNc-derived da_signal.
    cfg.enable_stdp = True
    cfg.enable_hebbian_learning = False
    cfg.enable_reward_modulation = True
    # Disable short-term plasticity for this minimal critic mechanism test: at the
    # cue rates needed to drive the MSN-typed striosome, the depressing cortico-
    # striatal E->I synapse (stp_U=0.15, tau_d=200ms) collapses transmission to
    # near-zero, starving the critic of the co-firing it needs to learn. STP is an
    # orthogonal biological feature; the value-critic claim is about the value being
    # neural + state-dependent, not about STP. (Documented confound removal.)
    cfg.enable_short_term_plasticity = False
    # Disable structural plasticity (synaptogenesis): the probe tests a FIXED circuit; activity-
    # dependent synapse growth would change the wiring mid-test, and it grows cp_connections
    # without growing some per-synapse arrays (cp_synapse_plastic_mask) -> IndexError on the
    # higher-activity B'-SNr circuit. Not needed for a mechanism probe.
    cfg.enable_structural_plasticity = False
    cfg.reward_learning_rate = float(reward_learning_rate)
    cfg.current_reward_signal = 0.0    # BRAIN-BASED: the SNc FIRING is the signal, not a host scalar
    cfg.reward_baseline = 0.0
    # STDP soft-bound gotcha (CLAUDE.md): Delta_w_LTP = A_plus*(w_max - w)*exp(..). With the
    # default w_max=2.0 and a cue->striosome design/grown weight >> 2, every LTP event goes
    # strongly NEGATIVE and the weight collapses to 2 — so V could never rise. Set w_max well
    # above the critic's working range so delta-LTP can actually grow V.
    cfg.stdp_w_max = 40.0

    # GABA_B -> GIRK slow K+ inhibitory conductance (protected edit, 2026-06-08). Off by
    # default (byte-identical); on, the striosome_value->snc pathway (tagged receptor="gaba_b"
    # below) subtracts value via the strong, sign-correct K+ reversal (-90 mV) instead of the
    # weak depolarized GABA_A reversal the direct baseline used.
    if gabab:
        cfg.enable_gabab = True
        cfg.gabab_reversal_potential = -90.0
        cfg.gabab_tau_decay = float(gabab_tau_decay)
        cfg.gabab_propagation_strength = float(gabab_propagation_strength)

    cfg.brain_regions = [
        BrainRegion(
            name="cue", n_neurons=n_cue, exc_fraction=1.0, internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
            plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ),
        BrainRegion(
            name="striosome_value", n_neurons=n_strio, exc_fraction=0.0,
            # FULLY GABAergic (MSNs are ~100% inhibitory). exc_fraction=0.05 left 3 excitatory
            # neurons whose output EXCITED the value's target, confounding the subtraction
            # (the value must be purely inhibitory to subtract).
            internal_density=0.0,   # no lateral self-inhibition: a graded VALUE readout,
                                    # not a winner-take-all gate (so V scales with the
                                    # learned cue->striosome weight instead of capping)
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D1.name,
            syn_reversal_potential_i_override=-60.0,   # MSN GABA_A reversal
        ),
        BrainRegion(
            name="snc", n_neurons=n_snc, exc_fraction=1.0, internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
            plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_DOPAMINE.name,
            syn_reversal_potential_i_override=-55.0,   # SNc lacks KCC2 -> depolarized E_GABA
        ),
    ]
    # The critic's learned value: cue (perceived state) -> striosome (value). PLASTIC. (both modes)
    pathways = [
        RegionPathway(from_region="cue", to_region="striosome_value",
                      density=0.6, weight_mean=float(cue_to_strio_weight),
                      weight_jitter=0.5, plastic=True),
    ]
    if bprime:
        # B'-DISINHIBIT-EXC (research 2026-06-08): the value subtraction is delivered via a
        # normal-reversal EXCITATORY relay, not weak GABA onto the depolarized SNc. The relay
        # `snc_drive` (exc, no reversal override -> default -75mV NORMAL reversal) is tonically
        # paced and supplies the SNc's excitatory drive; the GABAergic critic STRONGLY inhibits
        # the relay (full driving force, normal reversal); more V -> relay fires less -> less
        # excitation to the SNc -> SNc fires less. Sign-correct + strong; subtraction carried by
        # full-strength excitation, sidestepping the weak depolarized-GABA membrane.
        cfg.brain_regions.append(BrainRegion(
            name="snc_drive", n_neurons=n_drive, exc_fraction=1.0, internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
            plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
            # NO syn_reversal_potential_i_override -> default -75mV (normal) so GABA is strong on it
        ))
        pathways += [
            RegionPathway(from_region="striosome_value", to_region="snc_drive",
                          density=0.5, weight_mean=float(strio_to_drive_weight),
                          weight_jitter=0.2, plastic=False),   # value strongly inhibits the relay
            RegionPathway(from_region="snc_drive", to_region="snc",
                          density=0.6, weight_mean=float(snc_drive_to_snc_weight),
                          weight_jitter=0.2, plastic=False),   # relay excites the SNc (full strength)
        ]
    elif bprime_snr:
        # B'-DISINHIBIT-SNr (research #2, the biology-LITERAL disinhibition; owner: real effort here).
        # The textbook route by which the BG modulate DA: striatum/striosome -> (disinhibitor) -> SNr
        # tonic GABA -> SNc. Sign needs an ODD number of inhibitory links from value to SNc:
        #   value -(inh)-> disinhib -(inh)-> snr_tonic -(inh)-> snc  = net inhibitory = V up -> SNc down.
        # snr_tonic + disinhib are GABAergic, tonically paced; they have NORMAL reversals so the
        # inter-relay inhibition is strong. The SNc is held DEPOLARIZED (strong tonic+reward) so the
        # final SNr->SNc GABA hop is hyperpolarizing (the depolarized-reversal regime where GABA works).
        cfg.brain_regions.append(BrainRegion(
            name="snr_tonic", n_neurons=n_relay, exc_fraction=0.0, internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_GPI_OUTPUT.name,   # SNr-like GABAergic tonic output
        ))
        cfg.brain_regions.append(BrainRegion(
            name="disinhib", n_neurons=n_relay, exc_fraction=0.0, internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_GPE_PACEMAKER.name,  # tonically-active GABAergic
        ))
        pathways += [
            RegionPathway(from_region="striosome_value", to_region="disinhib",
                          density=0.5, weight_mean=float(strio_to_disinhib_weight),
                          weight_jitter=0.2, plastic=False),     # value inhibits the disinhibitor
            RegionPathway(from_region="disinhib", to_region="snr_tonic",
                          density=0.5, weight_mean=float(disinhib_to_gaba_weight),
                          weight_jitter=0.2, plastic=False),     # disinhibitor holds SNr-tonic partly off
            RegionPathway(from_region="snr_tonic", to_region="snc",
                          density=0.6, weight_mean=float(gaba_to_snc_weight),
                          weight_jitter=0.2, plastic=False),     # SNr tonic GABA onto the SNc
        ]
    else:
        # Direct: striosome GABA -> snc. With gabab=False this is the weak depolarized-reversal
        # GABA_A conduit (the de-risk baseline that FAILED the gap 0/3). With gabab=True the SAME
        # pathway is tagged receptor="gaba_b" so its inhibition routes through the slow GIRK K+
        # conductance (E_K=-90 mV) — the protected-edit fix. The A/B contrast (gaba_a fails,
        # gaba_b passes) localizes any win to the new conductance.
        pathways.append(
            RegionPathway(from_region="striosome_value", to_region="snc",
                          density=0.5, weight_mean=float(strio_to_snc_weight),
                          weight_jitter=0.2, plastic=False,
                          receptor=("gaba_b" if gabab else "gaba_a")))
    cfg.region_pathways = pathways

    # Stage-A dopamine modulator: production = from_region_firing_signed over ['snc'].
    snc_tonic_firing_fraction = 0.30
    cfg.enable_neuromodulator_subsystem = True
    cfg.neuromodulators = [
        NeuromodulatorConfig(
            name="dopamine", baseline=0.5, decay_tau_ms=200.0,
            concentration_min=0.0, concentration_max=2.0,
            targets=[ModulatorTarget(target_type="plasticity_rate", scope="all", sensitivity=+1.0)],
            production_rules=[ProductionRule(
                rule_type="from_region_firing_signed", sensitivity=float(snc_da_sensitivity),
                threshold=float(snc_tonic_firing_fraction), window_ms=200.0,
                source_regions=["snc"],
            )],
        )
    ]

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = seed
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge, cfg


def _idx(bridge, name):
    import numpy as np
    return np.asarray(bridge.region_manager.indices(name), dtype=np.int64)


def _drive(bridge, idx_map, drives, n_steps, xp, freeze_lr=None, cfg=None):
    """Set per-region external current (drives: {region: pA}), step n_steps, and
    return (snc_rate_hz, strio_rate_hz, mean_da). If freeze_lr is not None, the
    reward learning rate is temporarily set to it (0.0 = measure without learning)."""
    bridge.cp_external_input_current[:] = 0.0
    for region, pA in drives.items():
        bridge.cp_external_input_current[idx_map[region]] = xp.float32(pA)
    saved_lr = None
    if freeze_lr is not None and cfg is not None:
        saved_lr = cfg.reward_learning_rate
        cfg.reward_learning_rate = float(freeze_lr)
    snc_idx, strio_idx = idx_map["snc"], idx_map["striosome_value"]
    relay_idx = idx_map.get("snc_drive")   # B'-EXC relay
    if relay_idx is None:
        relay_idx = idx_map.get("snr_tonic")   # B'-SNr tonic-GABA relay (should go UP when V high)
    n_snc = len(_host(snc_idx)); n_strio = len(_host(strio_idx))
    n_relay = len(_host(relay_idx)) if relay_idx is not None else 0
    snc_spk = strio_spk = relay_spk = 0
    da_sum = 0.0
    for _ in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        # Advance sim time in MS — STDP reads current_time_ms for the pre/post delta_t.
        # Without this it stays 0, every delta_t is 0, STDP emits an exactly-zero update,
        # and no eligibility ever forms (the critic can't learn). The nav runner does
        # this manually each step too.
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)
        snc_spk += int(bridge.cp_firing_states[snc_idx].sum())
        strio_spk += int(bridge.cp_firing_states[strio_idx].sum())
        if relay_idx is not None:
            relay_spk += int(bridge.cp_firing_states[relay_idx].sum())
        da_sum += float(bridge.neuromodulator_manager.get_concentration("dopamine"))
    bridge._last_relay_rate = (relay_spk / max(n_relay, 1) / (n_steps * 1e-3)) if n_relay else 0.0
    if saved_lr is not None:
        cfg.reward_learning_rate = saved_lr
    dur_s = n_steps * 1e-3
    return (snc_spk / max(n_snc, 1) / dur_s,
            strio_spk / max(n_strio, 1) / dur_s,
            da_sum / max(n_steps, 1))


def _host(a):
    from sim.backend import to_host
    try:
        return to_host(a)
    except Exception:
        return a


def _calibrate_da_threshold(bridge, cfg, idx_map, tonic_drives, xp, n_steps=300):
    """Drive the tonic condition (snc directly, or the relay in B'), measure the SNc's mean
    firing FRACTION, and set the
    dopamine rule's threshold to it. The signed rule (neuromodulators.py:817) emits
    sensitivity*(rate_ema - threshold): with threshold = tonic, a burst (rate>tonic)
    -> da>baseline -> LTP, a dip (rate<tonic) -> da<baseline -> LTD, tonic -> ~0. The
    static 0.30 default is above even the reward-burst fraction, so it would make
    da_signal negative throughout (pure LTD). Auto-calibration removes that guesswork."""
    snc_idx = idx_map["snc"]; n_snc = len(_host(snc_idx))
    bridge.cp_external_input_current[:] = 0.0
    for region, pA in tonic_drives.items():
        bridge.cp_external_input_current[idx_map[region]] = xp.float32(pA)
    frac_sum = 0.0; m = 0
    for i in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        # Advance sim time in MS — STDP reads current_time_ms for the pre/post delta_t.
        # Without this it stays 0, every delta_t is 0, STDP emits an exactly-zero update,
        # and no eligibility ever forms (the critic can't learn). The nav runner does
        # this manually each step too.
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)
        if i >= n_steps // 2:
            frac_sum += float(bridge.cp_firing_states[snc_idx].sum()) / max(n_snc, 1); m += 1
    tonic_frac = frac_sum / max(m, 1)
    cfg.neuromodulators[0].production_rules[0].threshold = float(tonic_frac)
    return tonic_frac


def _mean_pathway_weight(bridge, pre_name, post_name):
    """Mean weight of the pre->post edges in the CSR (rows=post, cols=pre)."""
    import numpy as np
    pre = set(int(i) for i in _idx(bridge, pre_name))
    post = set(int(i) for i in _idx(bridge, post_name))
    coo = bridge.cp_connections.tocoo()
    rows = np.asarray(_host(coo.row)); cols = np.asarray(_host(coo.col)); data = np.asarray(_host(coo.data))
    # CSR orientation is rows=post, cols=pre — but fall back to the other orientation if
    # no edges match (so the reader is robust to the convention).
    m = np.fromiter(((r in post and c in pre) for r, c in zip(rows, cols)), dtype=bool, count=len(rows))
    if not m.any():
        m = np.fromiter(((r in pre and c in post) for r, c in zip(rows, cols)), dtype=bool, count=len(rows))
    return float(data[m].mean()) if m.any() else 0.0


def _lesion_pathway(bridge, pre_name, post_name):
    """Zero every pre_name->post_name edge in the CSR (the value conduit). Proves the
    subtraction is carried by neuron firing, not a host formula: after this, a trained CS
    can no longer subtract -> predicted == unpredicted. Direct mode cuts striosome_value->snc;
    B' mode cuts the relay snc_drive->snc."""
    import numpy as np
    pre_set = set(int(i) for i in _idx(bridge, pre_name))
    post_set = set(int(i) for i in _idx(bridge, post_name))
    coo = bridge.cp_connections.tocoo()
    rows = np.asarray(_host(coo.row), dtype=np.int64)   # post
    cols = np.asarray(_host(coo.col), dtype=np.int64)   # pre
    mask = np.array([(r in post_set and c in pre_set) for r, c in zip(rows, cols)])
    pre = cols[mask]; post = rows[mask]
    if len(pre) == 0:
        return 0
    return bridge.set_pathway_weights(f"{pre_name}->{post_name}(lesion)",
                                      pre, post, np.zeros(len(pre), dtype=np.float32))


def _lesion_gabab_mask(bridge):
    """Conductance lesion (the GABA_B anti-cheat): zero the per-synapse GABA_B routing mask
    so NO synapse feeds the slow K+ conductance any more. The slow conductance still decays
    each step but receives no new increment -> the GABA_B subtraction must VANISH (the SNc
    bursts to every reward regardless of prediction). Proves the state-specific gap was
    carried by the NEW GABA_B/GIRK conductance, not the residual weak GABA_A path or host
    arithmetic. Returns the number of GABA_B synapses zeroed."""
    m = getattr(bridge, "cp_gabab_synapse_mask", None)
    if m is None:
        return 0
    n_was = int(_host(m).sum())
    from sim.backend import get_backend
    xp, _ = get_backend()
    bridge.cp_gabab_synapse_mask = xp.zeros_like(m)
    # Also clear any residual charge already on the conductance so the lesion is clean.
    if getattr(bridge, "cp_conductance_g_gabab", None) is not None:
        bridge.cp_conductance_g_gabab[:] = 0.0
    return n_was


def run_diag(seed, *, cue_drive_pa=1000.0, cue_to_strio_weight=20.0,
             strio_to_snc_weight=3.5, hold_steps=60):
    """Diagnostic: is the cue firing? does cue->striosome transmit? can the striosome
    (MSN_D1) fire under DIRECT drive (its rheobase)? Pinpoints why V won't rise."""
    from sim.backend import get_backend
    xp, _ = get_backend()
    bridge, cfg = _build_stageb_bridge(seed, cue_to_strio_weight=cue_to_strio_weight,
                                       strio_to_snc_weight=strio_to_snc_weight)
    idx_map = {n: xp.asarray(_idx(bridge, n)) for n in ("cue", "striosome_value", "snc")}
    n = {k: len(_host(idx_map[k])) for k in idx_map}

    def rates(drives, steps=hold_steps):
        bridge.cp_external_input_current[:] = 0.0
        for r, pA in drives.items():
            bridge.cp_external_input_current[idx_map[r]] = xp.float32(pA)
        c = {k: 0 for k in idx_map}
        for _ in range(steps):
            bridge._run_one_simulation_step(); bridge.runtime_state.current_time_step += 1
            for k in idx_map:
                c[k] += int(bridge.cp_firing_states[idx_map[k]].sum())
        return {k: c[k] / max(n[k], 1) / (steps * 1e-3) for k in idx_map}

    print(f"  [diag seed={seed}] n_cue={n['cue']} n_strio={n['striosome_value']} n_snc={n['snc']}")
    print(f"  CS drive ({cue_drive_pa}pA -> cue), cue_to_strio_w={cue_to_strio_weight}:")
    r = rates({"cue": cue_drive_pa})
    print(f"    cue={r['cue']:.1f}Hz  striosome={r['striosome_value']:.1f}Hz  snc={r['snc']:.1f}Hz")
    print("  striosome DIRECT-drive rheobase sweep:")
    for pA in (200, 400, 600, 800, 1200, 1600, 2400):
        r = rates({"striosome_value": pA})
        print(f"    strio_drive={pA:5d}pA -> striosome={r['striosome_value']:6.1f}Hz  (snc={r['snc']:.1f}Hz)")


def run_stageb(seed, *, snc_tonic_pa=220.0, snc_reward_gain=400.0, cue_drive_pa=600.0,
               hold_steps=40, n_train=40, reward_learning_rate=0.08,
               cue_to_strio_weight=3.0, strio_to_snc_weight=2.5,
               snc_da_sensitivity=8.0, lesion=False, verbose=True,
               bprime=False, relay_tonic_pa=300.0, snc_drive_to_snc_weight=6.0,
               strio_to_drive_weight=15.0,
               bprime_snr=False, gaba_tonic_pa=300.0, disinhib_pa=250.0,
               strio_to_disinhib_weight=20.0, disinhib_to_gaba_weight=20.0,
               gaba_to_snc_weight=6.0,
               gabab=False, gabab_tau_decay=150.0, gabab_propagation_strength=0.105):
    from sim.backend import get_backend
    xp, _ = get_backend()
    bridge, cfg = _build_stageb_bridge(
        seed, snc_da_sensitivity=snc_da_sensitivity,
        reward_learning_rate=reward_learning_rate,
        gabab=gabab, gabab_tau_decay=gabab_tau_decay,
        gabab_propagation_strength=gabab_propagation_strength,
        cue_to_strio_weight=cue_to_strio_weight, strio_to_snc_weight=strio_to_snc_weight,
        bprime=bprime, snc_drive_to_snc_weight=snc_drive_to_snc_weight,
        strio_to_drive_weight=strio_to_drive_weight,
        bprime_snr=bprime_snr, strio_to_disinhib_weight=strio_to_disinhib_weight,
        disinhib_to_gaba_weight=disinhib_to_gaba_weight, gaba_to_snc_weight=gaba_to_snc_weight)
    region_names = (("cue", "striosome_value", "snc")
                    + (("snc_drive",) if bprime else ())
                    + (("snr_tonic", "disinhib") if bprime_snr else ()))
    idx_map = {n: xp.asarray(_idx(bridge, n)) for n in region_names}

    # Calibrate the dopamine threshold to the SNc's actual tonic firing fraction. B'-EXC: the
    # SNc tonic comes from the relay (pace it). B'-SNr: the SNc keeps its own tonic + the SNr/
    # disinhib relays are paced (pace all three for the true baseline).
    if bprime:
        tonic_drives = {"snc_drive": relay_tonic_pa}
    elif bprime_snr:
        tonic_drives = {"snc": snc_tonic_pa, "snr_tonic": gaba_tonic_pa, "disinhib": disinhib_pa}
    else:
        tonic_drives = {"snc": snc_tonic_pa}
    tonic_frac = _calibrate_da_threshold(bridge, cfg, idx_map, tonic_drives, xp)
    if verbose:
        print(f"  [calib] SNc tonic firing fraction = {tonic_frac:.4f} -> dopamine threshold"
              f"{' (B-prime: relay-paced)' if bprime else ''}")

    # Windows (drives in pA). US = reward current to the SNc; CS = drive to the cue.
    if bprime:
        # B'-DISINHIBIT-EXC: BOTH the tonic AND the reward drive the relay (snc_drive); the
        # value inhibits the relay, so the relay carries (tonic + reward - V) and the
        # snc_drive->snc excitation delivers delta = r - V to the SNc. (Reward must enter at
        # the relay, NOT directly at the SNc, or the value cannot subtract from it.)
        W_baseline = {"snc_drive": relay_tonic_pa}                                          # relay tonic -> SNc floor
        W_cs_us = {"cue": cue_drive_pa, "snc_drive": relay_tonic_pa + snc_reward_gain}      # relay: tonic+reward; CS=V inhibits it
        W_us_alone = {"snc_drive": relay_tonic_pa + snc_reward_gain}                        # relay: tonic+reward, NO cue (full)
        W_omission = {"cue": cue_drive_pa, "snc_drive": relay_tonic_pa}                     # relay: tonic; CS=V inhibits it, NO reward
    elif bprime_snr:
        # B'-SNr: the SNc has its OWN excitatory tonic + reward (held DEPOLARIZED so SNr GABA is
        # hyperpolarizing); the SNr-tonic + disinhib relays are paced; the value (via the disinhib
        # chain) modulates the SNr GABA onto the SNc. value up -> SNr GABA up -> SNc down.
        snr = {"snr_tonic": gaba_tonic_pa, "disinhib": disinhib_pa}
        W_baseline = {"snc": snc_tonic_pa, **snr}                                           # SNc tonic; relays paced; no cue/reward
        W_cs_us = {"cue": cue_drive_pa, "snc": snc_tonic_pa + snc_reward_gain, **snr}        # CS=V -> more SNr GABA; +reward
        W_us_alone = {"snc": snc_tonic_pa + snc_reward_gain, **snr}                          # +reward, NO cue (SNr GABA low)
        W_omission = {"cue": cue_drive_pa, "snc": snc_tonic_pa, **snr}                       # CS=V -> more SNr GABA; NO reward
    else:
        W_baseline = {"snc": snc_tonic_pa}                                  # tonic floor
        W_cs_us = {"cue": cue_drive_pa, "snc": snc_tonic_pa + snc_reward_gain}   # CS + reward
        W_us_alone = {"snc": snc_tonic_pa + snc_reward_gain}                # reward, NO cue
        W_omission = {"cue": cue_drive_pa, "snc": snc_tonic_pa}             # CS, NO reward

    # --- Acquisition: CS->US trials; the critic learns (V rises, US burst shrinks) ---
    us_burst, v_cs = [], []
    for t in range(n_train):
        _drive(bridge, idx_map, W_baseline, hold_steps, xp)            # inter-trial floor
        snc_r, strio_r, da = _drive(bridge, idx_map, W_cs_us, hold_steps, xp)  # LEARN
        us_burst.append(snc_r); v_cs.append(strio_r)
        if verbose and (t < 3 or t % 10 == 0 or t == n_train - 1):
            w = _mean_pathway_weight(bridge, "cue", "striosome_value")
            nnz = bridge.cp_connections.nnz
            elig = (float(abs(_host(bridge.cp_eligibility_trace[:nnz])).mean())
                    if bridge.cp_eligibility_trace is not None else -1)
            gain_arr = getattr(bridge, "cp_plasticity_rate_gain", None)
            gain = float(_host(gain_arr[:nnz]).mean()) if gain_arr is not None else -1
            print(f"  [acq t={t:02d}] US-burst={snc_r:6.2f}Hz  V(striosome)={strio_r:6.2f}Hz  "
                  f"w={w:.3f}  |elig|={elig:.2e}  gain={gain:.2f}  DA={da:.3f}")

    early = slice(0, max(1, n_train // 5)); late = slice(-max(1, n_train // 5), None)
    us_early = _st.mean(us_burst[early]); us_late = _st.mean(us_burst[late])
    v_early = _st.mean(v_cs[early]); v_late = _st.mean(v_cs[late])

    if lesion:
        if bprime:
            n_cut = _lesion_pathway(bridge, "snc_drive", "snc"); edge = "snc_drive->snc"
        elif bprime_snr:
            n_cut = _lesion_pathway(bridge, "snr_tonic", "snc"); edge = "snr_tonic->snc"
        elif gabab:
            # GABA_B conductance lesion (the decisive anti-cheat): cut the per-synapse GABA_B
            # routing mask so the slow K+ conductance gets NO increment. The subtraction must
            # vanish -> proves it was carried by the new GABA_B/GIRK conductance.
            n_cut = _lesion_gabab_mask(bridge); edge = "GABA_B mask (cp_gabab_synapse_mask)"
        else:
            n_cut = _lesion_pathway(bridge, "striosome_value", "snc"); edge = "striosome_value->snc"
        if verbose:
            print(f"  [lesion] zeroed {n_cut} {edge} edges")

    # --- Test (learning frozen): predicted vs unpredicted vs omission vs baseline ---
    base_r, base_v, _ = _drive(bridge, idx_map, W_baseline, hold_steps, xp, freeze_lr=0.0, cfg=cfg)
    base_relay = getattr(bridge, "_last_relay_rate", 0.0)
    pred_r, pred_v, _ = _drive(bridge, idx_map, W_cs_us, hold_steps, xp, freeze_lr=0.0, cfg=cfg)
    pred_relay = getattr(bridge, "_last_relay_rate", 0.0)
    unpred_r, unpred_v, _ = _drive(bridge, idx_map, W_us_alone, hold_steps, xp, freeze_lr=0.0, cfg=cfg)
    unpred_relay = getattr(bridge, "_last_relay_rate", 0.0)
    omit_r, omit_v, _ = _drive(bridge, idx_map, W_omission, hold_steps, xp, freeze_lr=0.0, cfg=cfg)
    if verbose:
        print(f"  [test V] predicted_strio={pred_v:.1f}Hz  unpredicted_strio={unpred_v:.1f}Hz  "
              f"omission_strio={omit_v:.1f}Hz  baseline_strio={base_v:.1f}Hz  "
              f"(V cue-gated if predicted/omission >> unpredicted/baseline)")
        if bprime or bprime_snr:
            want = ("snr_tonic should be HIGHER when V is high" if bprime_snr
                    else "snc_drive should be LOWER when V is high")
            print(f"  [test relay] predicted_relay={pred_relay:.1f}Hz  unpredicted_relay={unpred_relay:.1f}Hz  "
                  f"baseline_relay={base_relay:.1f}Hz  ({want})")

    v_learned = (v_late > 1.20 * v_early)               # (1) striosome value rose with training
    us_shrank = (us_late < 0.60 * us_early)             # (2) reward burst shrank
    state_specific = (unpred_r > 1.30 * max(pred_r, 1e-6))  # (3) unpredicted >> predicted (host-EMA can't)
    omission_dip = (omit_r < base_r)                    # (4) CS-no-reward dips below tonic

    gap_ratio = unpred_r / max(pred_r, 1e-6)             # unpredicted/predicted (>1.30 = state-specific)
    dip_depth = base_r - omit_r                          # tonic - omission (>0 = dip)
    return {
        "seed": seed, "lesion": lesion, "bprime": bprime, "bprime_snr": bprime_snr,
        "gabab": gabab,
        "us_burst_early_hz": us_early, "us_burst_late_hz": us_late,
        "v_cs_early_hz": v_early, "v_cs_late_hz": v_late,
        "test_baseline_hz": base_r, "test_predicted_hz": pred_r,
        "test_unpredicted_hz": unpred_r, "test_omission_hz": omit_r,
        "gap_ratio": gap_ratio, "dip_depth_hz": dip_depth,
        "v_learned": bool(v_learned), "us_burst_shrank": bool(us_shrank),
        "state_specific": bool(state_specific), "omission_dip": bool(omission_dip),
        "us_burst_curve": us_burst, "v_cs_curve": v_cs,
    }


def _print_result(r):
    print()
    print(f"  V(striosome) on CS : {r['v_cs_early_hz']:.2f} -> {r['v_cs_late_hz']:.2f} Hz   "
          f"(learned: {r['v_learned']})")
    print(f"  US burst           : {r['us_burst_early_hz']:.2f} -> {r['us_burst_late_hz']:.2f} Hz   "
          f"(shrank: {r['us_burst_shrank']})")
    print(f"  predicted (CS+US)  : {r['test_predicted_hz']:.2f} Hz")
    print(f"  unpredicted (US)   : {r['test_unpredicted_hz']:.2f} Hz   "
          f"(state-specific: {r['state_specific']})")
    print(f"  omission (CS,no US): {r['test_omission_hz']:.2f} Hz  vs baseline {r['test_baseline_hz']:.2f} Hz "
          f"(dip: {r['omission_dip']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None, help="comma seeds for multi-seed")
    ap.add_argument("--snc-tonic-pa", type=float, default=220.0)
    ap.add_argument("--snc-reward-gain", type=float, default=400.0)
    ap.add_argument("--cue-drive-pa", type=float, default=600.0)
    ap.add_argument("--hold-steps", type=int, default=40)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--reward-learning-rate", type=float, default=0.08)
    ap.add_argument("--cue-to-strio-weight", type=float, default=3.0)
    ap.add_argument("--strio-to-snc-weight", type=float, default=2.5)
    ap.add_argument("--snc-da-sensitivity", type=float, default=8.0)
    ap.add_argument("--lesion", action="store_true", help="anti-cheat: cut the value conduit after training")
    ap.add_argument("--diag", action="store_true", help="diagnostic: cue/striosome drive + MSN rheobase")
    ap.add_argument("--bprime", action="store_true",
                    help="B'-DISINHIBIT-EXC: value inhibits a normal-reversal excitatory relay that "
                         "drives the SNc (strong, sign-correct subtraction; sidesteps depolarized GABA)")
    ap.add_argument("--relay-tonic-pa", type=float, default=300.0)
    ap.add_argument("--snc-drive-to-snc-weight", type=float, default=6.0)
    ap.add_argument("--strio-to-drive-weight", type=float, default=15.0)
    ap.add_argument("--gabab", action="store_true",
                    help="GABA_B/GIRK: route striosome_value->snc through the slow K+ conductance "
                         "(E_K=-90mV, the protected edit) instead of weak GABA_A onto the depolarized SNc")
    ap.add_argument("--gabab-tau-decay", type=float, default=150.0,
                    help="GABA_B/GIRK decay time constant (ms); slow metabotropic, GIRK-IPSC range ~150-500")
    ap.add_argument("--gabab-propagation-strength", type=float, default=0.105,
                    help="per-spike GABA_B conductance increment scale")
    ap.add_argument("--bprime-snr", action="store_true",
                    help="B'-DISINHIBIT-SNr: striosome->disinhib->SNr-tonic-GABA->SNc (biology-literal disinhibition)")
    ap.add_argument("--gaba-tonic-pa", type=float, default=300.0)
    ap.add_argument("--disinhib-pa", type=float, default=250.0)
    ap.add_argument("--strio-to-disinhib-weight", type=float, default=20.0)
    ap.add_argument("--disinhib-to-gaba-weight", type=float, default=20.0)
    ap.add_argument("--gaba-to-snc-weight", type=float, default=6.0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.diag:
        run_diag(args.seed, cue_drive_pa=args.cue_drive_pa,
                 cue_to_strio_weight=args.cue_to_strio_weight,
                 strio_to_snc_weight=args.strio_to_snc_weight)
        return

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    kw = dict(snc_tonic_pa=args.snc_tonic_pa, snc_reward_gain=args.snc_reward_gain,
              cue_drive_pa=args.cue_drive_pa, hold_steps=args.hold_steps, n_train=args.n_train,
              reward_learning_rate=args.reward_learning_rate,
              cue_to_strio_weight=args.cue_to_strio_weight,
              strio_to_snc_weight=args.strio_to_snc_weight,
              snc_da_sensitivity=args.snc_da_sensitivity, lesion=args.lesion,
              bprime=args.bprime, relay_tonic_pa=args.relay_tonic_pa,
              snc_drive_to_snc_weight=args.snc_drive_to_snc_weight,
              strio_to_drive_weight=args.strio_to_drive_weight,
              bprime_snr=args.bprime_snr, gaba_tonic_pa=args.gaba_tonic_pa,
              disinhib_pa=args.disinhib_pa, strio_to_disinhib_weight=args.strio_to_disinhib_weight,
              disinhib_to_gaba_weight=args.disinhib_to_gaba_weight,
              gaba_to_snc_weight=args.gaba_to_snc_weight,
              gabab=args.gabab, gabab_tau_decay=args.gabab_tau_decay,
              gabab_propagation_strength=args.gabab_propagation_strength)
    results = []
    for s in seeds:
        tag = ("LESION" if args.lesion else "GABA_B/GIRK (E_K=-90mV)" if args.gabab
               else "B'-DISINHIBIT-SNr" if args.bprime_snr
               else "B'-DISINHIBIT-EXC" if args.bprime else "Stage-B critic (GABA_A direct)")
        print(f"[snc-stageB seed={s}] {tag} — CS-gated neural value (delta=r-V, R-W):")
        r = run_stageb(s, **kw)
        _print_result(r)
        if not args.lesion:
            gates = [r["v_learned"], r["us_burst_shrank"], r["state_specific"], r["omission_dip"]]
            verdict = "PASS" if all(gates) else f"PARTIAL ({sum(gates)}/4)"
            print(f"\n  Stage-B de-risk (seed {s}): {verdict}  "
                  f"[V-learned {r['v_learned']}, US-shrink {r['us_burst_shrank']}, "
                  f"state-specific {r['state_specific']}, omission-dip {r['omission_dip']}]")
            # The design's PRIMARY gate is (i) state-specific gap (the one GABA_A failed 0/3),
            # plus the regression guards (ii) v_learned and (iv) omission_dip. Report it explicitly.
            primary = r["state_specific"] and r["v_learned"] and r["omission_dip"]
            print(f"  [PRIMARY GATE — state-specific gap] gap_ratio(unpred/pred)={r['gap_ratio']:.2f} "
                  f"(>1.30 PASS) | V-learned {r['v_learned']} | dip {r['omission_dip']} "
                  f"=> {'PASS' if primary else 'FAIL'}")
        else:
            # Lesion EXPECTATION: prediction gone -> predicted ~= unpredicted, no dip.
            no_pred = (r["test_unpredicted_hz"] <= 1.30 * max(r["test_predicted_hz"], 1e-6))
            no_dip = not r["omission_dip"]
            print(f"\n  LESION anti-cheat (seed {s}): "
                  f"{'PASS' if (no_pred and no_dip) else 'UNEXPECTED'}  "
                  f"[prediction-gone {no_pred}, dip-gone {no_dip}] "
                  f"(cutting the neural conduit removed the subtraction)")
        results.append(r)
        print()

    if len(results) > 1 and not args.lesion:
        n_pass = sum(1 for r in results
                     if r["v_learned"] and r["us_burst_shrank"] and r["state_specific"] and r["omission_dip"])
        n_primary = sum(1 for r in results
                        if r["state_specific"] and r["v_learned"] and r["omission_dip"])
        print(f"=== MULTI-SEED: {n_pass}/{len(results)} PASS all 4 gates ===")
        print(f"=== MULTI-SEED PRIMARY GATE (state-specific gap + v-learned + dip): "
              f"{n_primary}/{len(results)} ===")
        gap_strs = ["{}={:.2f}".format(r["seed"], r["gap_ratio"]) for r in results]
        print("=== gap_ratio per seed: " + ", ".join(gap_strs) + " ===")

    if args.out:
        mode = ("stageb_lesion" if args.lesion
                else "stageb_gabab" if args.gabab else "stageb_critic")
        with open(args.out, "w") as f:
            json.dump({"mode": mode, "results": results}, f, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
