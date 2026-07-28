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
                         gabab=False, gabab_tau_decay=150.0, gabab_propagation_strength=0.105,
                         td_disinhibit=False, disinhib_tonic_weight=20.0,
                         csc=False, n_csc=8, n_csc_per=25,
                         csc_to_strio_weight=6.0, n_csc_strio=60, n_csc_drive=40, n_csc_disinhib=40,
                         csc_eligibility_tau_ms=None,
                         csc_gabab_level=True, csc_strio_to_snc_weight=2.5,
                         csc_gabab_tau_decay=60.0, csc_gabab_propagation_strength=0.105,
                         csc_conductance_deriv=True, csc_td_slow_tau_ms=400.0,
                         csc_td_derivative_gain=1.0, csc_gabab_conductance_max=0.0,
                         csc_stdp_w_max=None,
                         csc_fs_clamp=False, n_csc_fs=24, csc_to_fs_weight=20.0,
                         csc_fs_to_strio_weight=12.0,
                         csc_reward_relay=False, n_csc_reward_us=40,
                         csc_reward_us_to_snc_weight=6.0, csc_strio_to_reward_us_weight=8.0):
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

    if csc:
        # ============================================================================
        # A-CSC — COMPLETE SERIAL COMPOUND tapped-delay cue representation
        # (TD cue-shift design §2.1 A-CSC / §6.3 #2; Montague-Dayan-Sejnowski 1996;
        #  Sutton-Barto Ch 12). The cue is NOT one event but a CHAIN of K time-tagged
        # sub-state populations (csc_0 = cue@onset, csc_1 = cue@Delta, ... csc_{K-1}),
        # EACH driving the critic striosome_value through its OWN plastic synapse.
        # TD back-propagates value one tap per trial: the latest pre-reward sub-state's
        # value grows first (its eligibility overlaps the reward burst), then earlier
        # sub-states acquire value via the bootstrap (gamma*V(s_{t+1}) - V(s_t) > 0 at the
        # value's leading edge), until csc_0 (cue onset) carries value -> the SNc burst
        # MIGRATES from the reward onto the cue. The MULTIPLE sub-channels decouple the
        # single-channel B-2 conflict: each sub-state's value can be non-zero independently
        # (sparse, STDP-friendly), so the value LEVEL and the value DERIVATIVE no longer
        # fight on one channel.
        #
        # Delivery (ZERO sim/ edit, reuses the B-3 disinhibition relay): the value
        # DERIVATIVE drives the SNc via the disinhibition chain
        #   striosome_value --(inhib)--> disinhib --(inhib)--> snc_drive --(exc)--> snc
        # so a value RISE (the value's leading edge sweeping backward over trials)
        # DISINHIBITS/excites the SNc -> a burst AT that sub-state's time; a value FALL
        # (omission) adds inhibition -> the dip. The reward (US) enters at the relay.
        # The sub-state TIME-TAGGING (which tap is active in which bin) is the world's
        # cue-presentation timing (legitimate environment boundary, design §2.4 — same
        # status as the sustained cue in B-3); the VALUE LEARNING, the derivative, the
        # burst, the dip, and the credit assignment are all NEURAL.
        #
        # TEMPORAL CREDIT RESOLUTION (decisive for one-tap-per-trial back-propagation): the
        # eligibility trace tau (default 1000 ms) would smear credit across the whole ~160 ms
        # chain, crediting EVERY tap equally each trial => no migration gradient. A SHORT tau
        # (~40 ms, comparable to ~2 bins) makes each tap's eligibility decay before the next, so
        # the reward credits ONLY the last tap and the bootstrap-burst credits the tap just
        # before the value-carrying one — the tap-local credit CSC needs (this IS the TD(lambda)
        # stimulus-trace timescale; catalog C.29, Sutton-Barto Ch 12).
        if csc_eligibility_tau_ms is not None:
            cfg.reward_eligibility_tau_ms = float(csc_eligibility_tau_ms)
        # FULL TD delivers the value TWICE at the SNc membrane (design §2.2 B-1):
        #   (1) the value LEVEL as -V via the GABA_B/GIRK conductance (striosome_value -> snc,
        #       receptor="gaba_b"): more value -> the SNc fires LESS, so the REWARD burst SHRINKS
        #       as the reward-overlapping sub-state acquires value (the time-of-peak can then move
        #       OFF the reward) — the Stage-B / Eshel subtraction; AND
        #   (2) the value DERIVATIVE +dV/dt via the B-2 PROTECTED conductance-derivative edit
        #       (enable_td_value_derivative): I_td_deriv = gain*(g_gabab - g_gabab_slow)*(E_exc - V),
        #       the temporal derivative of the SAME GABA_B value channel computed at the membrane.
        #       On a value JUMP UP between consecutive taps it is positive -> a burst at that tap's
        #       onset; on a jump down -> the dip; flat -> ~0. This is the bootstrap gamma*V(s')-V(s).
        # On a SINGLE sustained cue (1) and (2) FIGHT (the B-2 edge-vs-level wall: one channel,
        # the derivative tracks the rising edge while the level grows). With CSC they DECOUPLE: the
        # value is a clean per-tap STEP function V(tap_0), V(tap_1), ... ; the LEVEL is the current
        # tap's value, the DERIVATIVE is the inter-tap difference V(tap_{k+1})-V(tap_k) — a clean
        # up-step whose positive burst rides the value's LEADING EDGE as it back-propagates over
        # trials (late taps acquire value first via the reward, then earlier) -> the burst MIGRATES
        # from the reward onto the cue. The GABA_B tau is SHORT (~60 ms) so g_gabab tracks the
        # per-tap value; td_slow_tau_ms (the EMA lag) sets the derivative window.
        K = int(n_csc)
        if csc_gabab_level:
            cfg.enable_gabab = True
            cfg.gabab_reversal_potential = -90.0
            cfg.gabab_tau_decay = float(csc_gabab_tau_decay)
            cfg.gabab_propagation_strength = float(csc_gabab_propagation_strength)
            # GIRK saturation cap (existing owner-approved guardrail, cfg.gabab_conductance_max):
            # bound g_gabab so a HOT critic (high value) cannot FULLY CLAMP the SNc to silence
            # (the B-2 tonic-death wall). With the cap, -V is a GRADED shift at any value rate, so
            # the reward burst SHRINKS without killing the live tonic the cue burst needs. 0 = no cap.
            cfg.gabab_conductance_max = float(csc_gabab_conductance_max)
        if csc_stdp_w_max is not None:
            # Cap the per-tap weight growth so the critic stays in the SPARSE MSN band (the B-2/B-3
            # lesson + the brief: drive the critic SPARSELY). Unbounded growth -> a HOT critic
            # (~180 Hz) whose GABA_B doesn't decay between bins -> a sustained -V that kills the
            # SNc tonic. A cap (~12-16) keeps the value in the graded band where -V is localized.
            cfg.stdp_w_max = float(csc_stdp_w_max)
        if csc_conductance_deriv:
            # B-2 PROTECTED edit (re-applied; byte-identical when OFF, proven COMBO e728d7f1...).
            # ON only for the CSC conductance-derivative delivery.
            cfg.enable_td_value_derivative = True
            cfg.td_slow_tau_ms = float(csc_td_slow_tau_ms)
            cfg.td_derivative_gain = float(csc_td_derivative_gain)
        regions = []
        for k in range(K):
            regions.append(BrainRegion(
                name=f"csc_{k}", n_neurons=int(n_csc_per), exc_fraction=1.0,
                internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                weight_jitter=0.0, plastic_internal=False,
                izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
            ))
        regions.append(BrainRegion(
            name="striosome_value", n_neurons=int(n_csc_strio), exc_fraction=0.0,
            internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D1.name,
            syn_reversal_potential_i_override=-60.0,
        ))
        if csc_fs_clamp:
            # Critic FS-clamp (the production N9 mechanism): a fast-spiking interneuron pool driven
            # FEEDFORWARD by the sub-states, inhibiting the critic. Its drive scales with the volley
            # (divisive-leaning), so it holds the critic in the physiological MSN band (~1-30 Hz)
            # EVEN AS the per-tap weights grow — decoupling "the weights/value grow" from "the critic
            # fires densely." This bounds g_gabab so the -V is a graded localized subtraction (the
            # reward burst shrinks) WITHOUT saturating to a tonic-killing clamp. catalog (Tepper PV-FSI).
            regions.append(BrainRegion(
                name="csc_fs", n_neurons=int(n_csc_fs), exc_fraction=0.0, internal_density=0.0,
                exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name,
                syn_reversal_potential_i_override=-60.0,
            ))
        if csc_reward_relay:
            # REWARD RELAY (the multi-channel critic, catalog C.33 PPN->DA + striosome->RMTg/PPN
            # inhibition): the reward r enters via an EXCITATORY relay reward_us -> snc, and the
            # critic INHIBITS reward_us. So the reward reaching the SNc is r - V(reward-state) — the
            # value cancels r AT the reward, the canonical δ=r-V. CRUCIALLY this localizes -V to the
            # reward window (reward_us is silent otherwise), so the chain's value does NOT suppress
            # the SNc tonic. With the reward handled here, the direct striosome->snc GABA_B can be
            # WEAK (its job is only to source the conductance-derivative for the cue-shift burst),
            # so the chain stays at a live physiological tonic and the derivative cue burst stands out.
            regions.append(BrainRegion(
                name="reward_us", n_neurons=int(n_csc_reward_us), exc_fraction=1.0,
                internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                plastic_internal=False,
                izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
            ))
        regions.append(BrainRegion(
            name="snc", n_neurons=int(n_snc), exc_fraction=1.0, internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
            plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_DOPAMINE.name,
            syn_reversal_potential_i_override=-55.0,
        ))
        cfg.brain_regions = regions
        pathways = []
        for k in range(K):
            # Each sub-state -> the critic through its OWN plastic synapse (the value w_k).
            pathways.append(RegionPathway(
                from_region=f"csc_{k}", to_region="striosome_value",
                density=0.6, weight_mean=float(csc_to_strio_weight),
                weight_jitter=0.5, plastic=True))
            if csc_fs_clamp:
                # The sub-state ALSO drives the FS pool (feedforward), which then clamps the critic.
                pathways.append(RegionPathway(
                    from_region=f"csc_{k}", to_region="csc_fs",
                    density=0.6, weight_mean=float(csc_to_fs_weight),
                    weight_jitter=0.2, plastic=False))
        if csc_fs_clamp:
            pathways.append(RegionPathway(
                from_region="csc_fs", to_region="striosome_value",
                density=0.7, weight_mean=float(csc_fs_to_strio_weight),
                weight_jitter=0.2, plastic=False))
        if csc_reward_relay:
            # reward_us -> snc (excitatory; carries r). The critic inhibits reward_us (so r - V
            # reaches the SNc). The critic's GABA onto reward_us uses the normal reversal (-60) so it
            # is a strong subtraction on the relay (the relay is held depolarized by the reward).
            pathways.append(RegionPathway(
                from_region="reward_us", to_region="snc",
                density=0.6, weight_mean=float(csc_reward_us_to_snc_weight),
                weight_jitter=0.2, plastic=False))
            pathways.append(RegionPathway(
                from_region="striosome_value", to_region="reward_us",
                density=0.6, weight_mean=float(csc_strio_to_reward_us_weight),
                weight_jitter=0.2, plastic=False))
        if csc_gabab_level:
            # The value channel: striosome_value -> snc via the slow K+ GABA_B/GIRK conductance
            # (E_K=-90mV). This is BOTH the -V LEVEL subtraction (the I_gabab current, which shrinks
            # the reward burst) AND the source of the conductance-derivative (g_gabab is what the
            # B-2 edit differentiates to deliver +dV/dt). The SNc is driven directly with the tonic
            # pacemaker + the reward (no disinhibition relay; that delivered +V level, not dV/dt).
            pathways.append(RegionPathway(
                from_region="striosome_value", to_region="snc",
                density=0.6, weight_mean=float(csc_strio_to_snc_weight),
                weight_jitter=0.2, plastic=False, receptor="gaba_b"))
        cfg.region_pathways = pathways
        return _finish_stageb_bridge(cfg, seed, snc_da_sensitivity)

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
    if td_disinhibit:
        # TD value-DERIVATIVE via DISINHIBITION (B-3, 2026-06-10 TD cue-shift design §2.2).
        # GOAL: a value RISE (cue onset) must DISINHIBIT/excite the SNc -> a burst AT THE CUE
        # (the bootstrap gamma*V(s') - V(s) > 0), and a value FALL (omission, expected-reward
        # time) must add inhibition -> a dip. The B' relay alone delivers the value LEVEL
        # (more V -> less SNc); the cue-burst needs the DERIVATIVE sign, achieved by routing the
        # value through ONE extra inhibitory stage so a value RISE *releases* the SNc drive:
        #
        #   striosome_value (phasic V) --(inhib)--> disinhib --(inhib)--> snc_drive --(exc)--> snc
        #
        # The critic's value is intrinsically PHASIC at cue onset (MSN spike-frequency adaptation
        # / FS-clamp: it bursts when the cue turns on, then adapts to a lower plateau) -> its
        # transient IS the value derivative. With the disinhibition chain:
        #   - cue onset: V transient UP -> disinhib DOWN -> snc_drive RELEASED (UP) -> SNc BURST (at the cue).
        #   - CS->US gap: V plateaus -> disinhib steady -> snc_drive steady -> SNc ~tonic (no gap burst).
        #   - omission (V falls at expected-reward time): V DOWN -> disinhib UP -> snc_drive DOWN -> SNc DIP.
        # `disinhib` is a tonically-paced GABAergic relay (normal reversal so the inter-relay
        # inhibition is strong); `snc_drive` is the tonically-paced EXC relay (default -75mV).
        # ZERO sim/ edit -- reuses the bprime EXC-relay + the bprime_snr disinhib recipe.
        cfg.brain_regions.append(BrainRegion(
            name="snc_drive", n_neurons=n_drive, exc_fraction=1.0, internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
            # NO reversal override -> default -75mV (normal) so disinhib's GABA is strong on it
        ))
        cfg.brain_regions.append(BrainRegion(
            name="disinhib", n_neurons=n_relay, exc_fraction=0.0, internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_GPE_PACEMAKER.name,  # tonically-active GABAergic
        ))
        pathways += [
            RegionPathway(from_region="striosome_value", to_region="disinhib",
                          density=0.5, weight_mean=float(strio_to_disinhib_weight),
                          weight_jitter=0.2, plastic=False),    # value inhibits the disinhibitor
            RegionPathway(from_region="disinhib", to_region="snc_drive",
                          density=0.5, weight_mean=float(disinhib_tonic_weight),
                          weight_jitter=0.2, plastic=False),     # disinhibitor holds snc_drive partly off
            RegionPathway(from_region="snc_drive", to_region="snc",
                          density=0.6, weight_mean=float(snc_drive_to_snc_weight),
                          weight_jitter=0.2, plastic=False),     # relay excites the SNc (full strength)
        ]
    elif bprime:
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
    return _finish_stageb_bridge(cfg, seed, snc_da_sensitivity)


def _finish_stageb_bridge(cfg, seed, snc_da_sensitivity):
    """Attach the Stage-A dopamine modulator (production = from_region_firing_signed over
    ['snc']) and build + initialize the bridge. Shared by every mode (direct / bprime /
    bprime_snr / td_disinhibit / csc) so the modulator + RNG-pinned construction are identical."""
    from sim.bridge import SimulationBridge
    from sim.config import RuntimeState, GPUConfig, VisualizationConfig
    from sim.neuromodulators import NeuromodulatorConfig, ModulatorTarget, ProductionRule
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


def _calibrate_da_baseline(bridge, cfg, idx_map, tonic_drives, xp, n_steps=400):
    """TD-mode calibration (2026-06-10): after the threshold is set, drive the TONIC condition
    and measure the SETTLED dopamine CONCENTRATION, then set the modulator `baseline` to it so
    `da_signal = da_conc - baseline` is centered at ZERO at tonic. Without this, the production
    rule (sensitivity*(rate_ema - threshold)) makes da_conc settle near 0 at tonic, far below the
    fixed baseline (0.5) -> da_signal is constantly NEGATIVE -> pure LTD -> the critic UNLEARNS V
    (the value cannot rise, so no migration). Centering the baseline makes a burst (rate>tonic)
    give +da_signal (LTP) and a dip give -da_signal (LTD) — the sign the three-factor critic
    needs to LEARN the cue value up across trials. (The threshold sets WHERE the firing-rate->
    production crossover is; the baseline sets WHERE da_conc->da_signal crossover is. Both must
    sit at tonic.)"""
    bridge.cp_external_input_current[:] = 0.0
    for region, pA in tonic_drives.items():
        bridge.cp_external_input_current[idx_map[region]] = xp.float32(pA)
    conc_sum = 0.0; m = 0
    for i in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)
        if i >= n_steps // 2:   # average over the settled second half
            conc_sum += float(bridge.neuromodulator_manager.get_concentration("dopamine")); m += 1
    tonic_conc = conc_sum / max(m, 1)
    cfg.neuromodulators[0].baseline = float(tonic_conc)
    return tonic_conc


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
    rows = np.asarray(_host(coo.row), dtype=np.int64)
    cols = np.asarray(_host(coo.col), dtype=np.int64)
    # CSR orientation is rows=post, cols=pre — but fall back to the other orientation if no
    # edges match (so the lesion is robust to the convention, same as _mean_pathway_weight).
    mask = np.array([(r in post_set and c in pre_set) for r, c in zip(rows, cols)])
    if not mask.any():
        mask = np.array([(r in pre_set and c in post_set) for r, c in zip(rows, cols)])
        pre = rows[mask]; post = cols[mask]
    else:
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


# ======================================================================================
# TD CUE-SHIFT (Pavlovian cue->reward burst MIGRATION) — B-3 zero-edit derivative probe
# ======================================================================================
#
# Design: docs/plans/2026-06-10-N9-TD-cue-shift-design.md (option B-3, §2.2/§4/§5/§6).
# Distinct from run_stageb (Rescorla-Wagner: delta = r - V). This tests the TD bootstrap
# signature: across cue->reward learning the SNc phasic-dopamine burst MIGRATES from the
# reward (US) onto the predictive cue (CS) — the iconic Schultz 1997 result, the one
# canonical dopamine signature the circuit does not yet show.
#
# Mechanism (zero sim/ edit, runner-side only): td_disinhibit=True wires the value through a
# disinhibition chain (striosome -> disinhib -> snc_drive -> snc) so a value RISE *releases*
# the SNc (burst at the cue) and a value FALL adds inhibition (dip at the expected-reward
# time). The critic's value is intrinsically phasic at cue onset (its transient IS the value
# derivative). The trial CLOCK (CS at t0, US at t0+ISI) is the world's event timing
# (legitimate environment/body boundary, design §2.4); every cognitive term — value, the
# derivative, the burst, the dip — is neural.


def _drive_timecourse(bridge, idx_map, drives, n_steps, xp, bin_steps,
                      events=None, freeze_lr=None, cfg=None):
    """Like _drive, but records a per-BIN SNc firing-rate TIME-COURSE so the time-of-peak is
    measurable. `drives` is the steady drive; `events` is an optional list of
    (start_step, end_step, {region: pA}) overrides applied within the window (so the protocol
    can turn the CS on at t0 and the US on at t0+ISI WITHIN one continuous stepping window —
    the cue trace and reward then co-exist in the same simulated trajectory, as in a real
    conditioning trial). Returns (snc_rate_per_bin[list], strio_rate_per_bin[list],
    snc_rate_hz_overall, strio_rate_hz_overall)."""
    import numpy as np
    bridge.cp_external_input_current[:] = 0.0
    for region, pA in drives.items():
        bridge.cp_external_input_current[idx_map[region]] = xp.float32(pA)
    saved_lr = None
    if freeze_lr is not None and cfg is not None:
        saved_lr = cfg.reward_learning_rate
        cfg.reward_learning_rate = float(freeze_lr)
    snc_idx, strio_idx = idx_map["snc"], idx_map["striosome_value"]
    n_snc = len(_host(snc_idx)); n_strio = len(_host(strio_idx))
    events = events or []
    snc_bins, strio_bins = [], []
    snc_bin_spk = strio_bin_spk = 0
    snc_total = strio_total = 0
    for step in range(n_steps):
        # Apply event overrides for this step (world's CS/US scheduling).
        for (s0, s1, ev_drives) in events:
            if s0 <= step < s1:
                for region, pA in ev_drives.items():
                    bridge.cp_external_input_current[idx_map[region]] = xp.float32(pA)
            elif step == s1:
                # Event ended this step: restore the steady drive for those regions (cue OFF / US OFF).
                for region in ev_drives:
                    base = drives.get(region, 0.0)
                    bridge.cp_external_input_current[idx_map[region]] = xp.float32(base)
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)
        s = int(bridge.cp_firing_states[snc_idx].sum())
        v = int(bridge.cp_firing_states[strio_idx].sum())
        snc_bin_spk += s; strio_bin_spk += v
        snc_total += s; strio_total += v
        if (step + 1) % bin_steps == 0:
            dur_s = bin_steps * bridge.core_config.dt_ms * 1e-3
            snc_bins.append(snc_bin_spk / max(n_snc, 1) / dur_s)
            strio_bins.append(strio_bin_spk / max(n_strio, 1) / dur_s)
            snc_bin_spk = strio_bin_spk = 0
    if saved_lr is not None:
        cfg.reward_learning_rate = saved_lr
    dur_all = n_steps * bridge.core_config.dt_ms * 1e-3
    return (snc_bins, strio_bins,
            snc_total / max(n_snc, 1) / dur_all,
            strio_total / max(n_strio, 1) / dur_all)


def _pearson_r(xs, ys):
    """Pearson correlation; returns 0.0 on degenerate input (no variance)."""
    import numpy as np
    xs = np.asarray(xs, dtype=np.float64); ys = np.asarray(ys, dtype=np.float64)
    if len(xs) < 2:
        return 0.0
    sx = xs.std(); sy = ys.std()
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return float(np.corrcoef(xs, ys)[0, 1])


def run_td(seed, *, snc_reward_gain=400.0, cue_drive_pa=600.0,
           relay_tonic_pa=300.0, disinhib_pa=100.0,
           cue_to_strio_weight=20.0, strio_to_disinhib_weight=20.0,
           disinhib_tonic_weight=8.0, snc_drive_to_snc_weight=6.0,
           snc_da_sensitivity=8.0, reward_learning_rate=0.08,
           n_train=60, n_cs_bins=6, n_isi_bins=4, n_post_bins=4,
           bin_steps=20, lesion_cue=False, unpaired=False, verbose=True):
    """TD cue-shift Pavlovian protocol on the spiking SNc (B-3 zero-edit derivative).

    Each TRIAL = cue ON at t0 (SUSTAINED across the CS->US interval, the A-trace) ->
    reward (US via the SNc-drive reward afferent) at t0+ISI -> inter-trial gap, ALL in one
    continuous stepping window so the cue trace and the reward co-exist in the trajectory.
    Learning is ON (the critic learns V from the SNc-derived dopamine). The SNc firing-rate
    TIME-COURSE is recorded per bin every trial so the time-of-peak is measurable.

    Window layout (bins), per trial:
        [ n_cs_bins (CS only) | n_isi_bins (CS+US fires partway) | n_post_bins (post) ]
    The CS spans the whole CS->ISI; the US fires in the FIRST `n_isi_bins` portion.

    HEADLINE metric: Pearson r between TRIAL NUMBER and the SNc burst TIME-OF-PEAK (bin index
    of the max SNc rate). Migration => the peak moves EARLIER (toward the CS, lower bin index)
    across learning => r < 0 (negative slope). Pass bar |r| > 0.7 with the correct (earlier)
    sign.

    Anti-cheats: lesion_cue=True zeros cue->striosome (migration must vanish, US reflex
    remains); unpaired=True decouples CS and US in time (no contingency -> no migration, no dip).
    """
    from sim.backend import get_backend
    xp, _ = get_backend()
    bridge, cfg = _build_stageb_bridge(
        seed, snc_da_sensitivity=snc_da_sensitivity,
        reward_learning_rate=reward_learning_rate,
        cue_to_strio_weight=cue_to_strio_weight,
        td_disinhibit=True, snc_drive_to_snc_weight=snc_drive_to_snc_weight,
        strio_to_disinhib_weight=strio_to_disinhib_weight,
        disinhib_tonic_weight=disinhib_tonic_weight, n_drive=40, n_relay=40)
    region_names = ("cue", "striosome_value", "snc", "snc_drive", "disinhib")
    idx_map = {n: xp.asarray(_idx(bridge, n)) for n in region_names}

    # PROVENANCE / anti-cheat (3): no host TD term reaches the SNc. The SNc current is
    # tonic(relay) + reward_us(at the relay) + synaptic disinhibition ONLY. Assert the
    # brain-based stance the build enforces: no host reward scalar, no host value/EMA.
    assert cfg.current_reward_signal == 0.0, "host reward scalar must be 0 (brain-based)"
    assert cfg.reward_baseline == 0.0, "host reward baseline must be 0 (brain-based)"
    # The SNc receives NO direct external current in the protocol (reward enters at the relay);
    # its drive is purely synaptic (snc_drive->snc) plus whatever the disinhibition chain sets.
    prov = {
        "snc_gets_direct_current": False,   # reward+tonic enter at snc_drive, not snc
        "host_reward_signal": float(cfg.current_reward_signal),
        "host_value_term": False,           # no host V / reward_ema in this probe
        "snc_drive_terms": "tonic(relay) + reward_us(relay) + synaptic disinhibition only",
    }

    # Calibrate the dopamine threshold to the SNc's tonic firing fraction, THEN the modulator
    # baseline to the settled tonic da concentration (so a burst -> +da_signal=LTP, a dip ->
    # -da_signal=LTD; without baseline-centering the critic UNLEARNS V — see _calibrate_da_baseline).
    tonic_drives = {"snc_drive": relay_tonic_pa, "disinhib": disinhib_pa}
    tonic_frac = _calibrate_da_threshold(bridge, cfg, idx_map, tonic_drives, xp)
    tonic_conc = _calibrate_da_baseline(bridge, cfg, idx_map, tonic_drives, xp)
    if verbose:
        print(f"  [calib] SNc tonic firing fraction = {tonic_frac:.4f} -> threshold; "
              f"tonic da conc = {tonic_conc:.4f} -> baseline (da_signal centered at tonic)")

    # Per-trial window: CS-only bins, then ISI bins (CS continues; US fires in the first half),
    # then post bins. Steps:
    n_win_bins = n_cs_bins + n_isi_bins + n_post_bins
    cs_start_bin = 0
    isi_start_bin = n_cs_bins                 # US-window onset bin (the expected-reward time)
    win_steps = n_win_bins * bin_steps
    cs_steps = (n_cs_bins + n_isi_bins) * bin_steps     # cue sustained across CS + ISI
    us_on_step = n_cs_bins * bin_steps                  # US turns on at ISI onset
    us_off_step = (n_cs_bins + max(1, n_isi_bins // 2)) * bin_steps  # US fires in first half of ISI

    # Steady (inter-trial floor): the relay tonic + the disinhibitor tone, NO cue, NO reward.
    floor = {"snc_drive": relay_tonic_pa, "disinhib": disinhib_pa}

    import random as _random
    rng = _random.Random(seed)   # for the unpaired control's random US offsets (per-seed pinned)

    peak_bins = []        # time-of-peak (bin index) per trial
    cs_rates = []         # SNc rate in the CS-only window per trial
    us_rates = []         # SNc rate in the US window per trial
    gap_rates = []        # SNc rate in the late-CS (pre-US) bins per trial
    v_cs_rates = []       # critic rate on the CS per trial
    snc_tc_first = snc_tc_last = None

    for t in range(n_train):
        # Inter-trial floor (let the system settle).
        _drive_timecourse(bridge, idx_map, floor, bin_steps * 2, xp, bin_steps)
        # Build the trial events. CS sustained for cs_steps; US fires us_on..us_off.
        if unpaired:
            # ANTI-CHEAT (b): decouple CS and US timing. The US fires at a RANDOM offset
            # unrelated to the CS (and sometimes after the cue is gone) so there is no
            # CS->US contingency for the critic to learn.
            jitter = rng.randint(0, max(1, n_post_bins)) * bin_steps
            us0 = us_on_step + jitter
            us1 = us0 + max(1, n_isi_bins // 2) * bin_steps
        else:
            us0, us1 = us_on_step, us_off_step
        events = [
            (0, cs_steps, {"cue": cue_drive_pa}),                                  # CS sustained (A-trace)
            (us0, us1, {"snc_drive": relay_tonic_pa + snc_reward_gain}),           # US = reward at the relay
        ]
        snc_bins, strio_bins, snc_all, strio_all = _drive_timecourse(
            bridge, idx_map, floor, win_steps, xp, bin_steps, events=events)
        # Time-of-peak: the bin index of the max SNc rate over the window.
        import numpy as np
        peak_bin = int(np.argmax(snc_bins)) if snc_bins else 0
        peak_bins.append(peak_bin)
        cs_rate = float(np.mean(snc_bins[cs_start_bin:isi_start_bin])) if isi_start_bin > 0 else 0.0
        us_rate = float(np.mean(snc_bins[isi_start_bin:isi_start_bin + max(1, n_isi_bins // 2)]))
        gap_rate = (float(np.mean(snc_bins[max(0, isi_start_bin - 2):isi_start_bin]))
                    if isi_start_bin >= 2 else cs_rate)
        cs_rates.append(cs_rate); us_rates.append(us_rate); gap_rates.append(gap_rate)
        v_cs_rates.append(float(np.mean(strio_bins[cs_start_bin:isi_start_bin])) if isi_start_bin > 0 else 0.0)
        if t == 0:
            snc_tc_first = list(snc_bins)
        if t == n_train - 1:
            snc_tc_last = list(snc_bins)
        if verbose and (t < 3 or t % 10 == 0 or t == n_train - 1):
            w = _mean_pathway_weight(bridge, "cue", "striosome_value")
            print(f"  [td t={t:02d}] peak_bin={peak_bin}  CS-rate={cs_rate:6.2f}  "
                  f"US-rate={us_rate:6.2f}Hz  V(strio)CS={v_cs_rates[-1]:6.1f}Hz  w(cue->strio)={w:.3f}")

    # --- Omission test (frozen learning): CS, no US. The dip should be at the EXPECTED-reward
    #     time (the ISI-onset bin), not the cue. ---
    omit_events = [(0, cs_steps, {"cue": cue_drive_pa})]   # CS sustained, NO US
    omit_bins, _, _, _ = _drive_timecourse(
        bridge, idx_map, floor, win_steps, xp, bin_steps, events=omit_events,
        freeze_lr=0.0, cfg=cfg)
    # --- Baseline (frozen): floor only, no CS, no US (the tonic time-course). ---
    base_bins, _, _, _ = _drive_timecourse(
        bridge, idx_map, floor, win_steps, xp, bin_steps, freeze_lr=0.0, cfg=cfg)

    import numpy as np
    # HEADLINE: r between trial number and time-of-peak. Migration => peak moves EARLIER
    # (toward the CS) => NEGATIVE slope. We report r (signed) AND the cue-ward boolean.
    trial_idx = list(range(n_train))
    r_migration = _pearson_r(trial_idx, peak_bins)
    early = slice(0, max(1, n_train // 5)); late = slice(-max(1, n_train // 5), None)
    peak_early = float(np.mean(peak_bins[early])); peak_late = float(np.mean(peak_bins[late]))
    cs_early = float(np.mean(cs_rates[early])); cs_late = float(np.mean(cs_rates[late]))
    us_early = float(np.mean(us_rates[early])); us_late = float(np.mean(us_rates[late]))
    v_early = float(np.mean(v_cs_rates[early])); v_late = float(np.mean(v_cs_rates[late]))
    gap_late = float(np.mean(gap_rates[late]))
    tonic_rate = float(np.mean(base_bins)) if base_bins else 0.0

    # Omission dip: the SNc dips below tonic AT the expected-reward bin (ISI onset), NOT at the cue.
    omit_at_reward = float(np.mean(omit_bins[isi_start_bin:isi_start_bin + max(1, n_isi_bins // 2)])) if omit_bins else 0.0
    omit_at_cue = float(np.mean(omit_bins[cs_start_bin:isi_start_bin])) if isi_start_bin > 0 else 0.0
    base_at_reward = float(np.mean(base_bins[isi_start_bin:isi_start_bin + max(1, n_isi_bins // 2)])) if base_bins else 0.0
    dip_at_reward_depth = base_at_reward - omit_at_reward

    # ---- Gates (design §4.2) ----
    # Headline: peak migrates earlier (cue-ward) with |r| > 0.7.
    migration_r_pass = (r_migration < -0.7)
    migration_dir_pass = (peak_late < peak_early - 0.5)   # peak moved earlier by >=~half a bin
    # Late: burst at CS, not US (the burst TRANSFERRED, not merely shrank).
    late_burst_at_cs = (cs_late > 1.10 * tonic_rate) and (us_late <= 1.20 * tonic_rate + 1e-6)
    # Early: burst at US.
    early_burst_at_us = (us_early > 1.10 * tonic_rate)
    # No burst between cue and reward (value flat -> derivative ~ 0).
    no_gap_burst = (gap_late <= 1.30 * tonic_rate + 1e-6)
    # Omission dip stays at REWARD time, not the cue.
    omission_dip_at_reward = (dip_at_reward_depth > 0) and (omit_at_reward < omit_at_cue + 1e-6)
    # Regression guards.
    v_learned = (v_late > 1.20 * v_early) if v_early > 1e-6 else (v_late > 1e-6)

    gates = {
        "migration_r_pass": bool(migration_r_pass),
        "migration_dir_pass": bool(migration_dir_pass),
        "early_burst_at_us": bool(early_burst_at_us),
        "late_burst_at_cs": bool(late_burst_at_cs),
        "no_gap_burst": bool(no_gap_burst),
        "omission_dip_at_reward": bool(omission_dip_at_reward),
        "v_learned": bool(v_learned),
    }

    return {
        "seed": seed, "lesion_cue": lesion_cue, "unpaired": unpaired,
        "n_train": n_train, "bin_steps": bin_steps,
        "n_cs_bins": n_cs_bins, "n_isi_bins": n_isi_bins, "n_post_bins": n_post_bins,
        "isi_start_bin": isi_start_bin,
        "r_migration": r_migration,
        "peak_bin_early": peak_early, "peak_bin_late": peak_late,
        "cs_rate_early": cs_early, "cs_rate_late": cs_late,
        "us_rate_early": us_early, "us_rate_late": us_late,
        "gap_rate_late": gap_late, "tonic_rate": tonic_rate,
        "v_cs_early_hz": v_early, "v_cs_late_hz": v_late,
        "omit_at_reward_hz": omit_at_reward, "omit_at_cue_hz": omit_at_cue,
        "base_at_reward_hz": base_at_reward, "dip_at_reward_depth_hz": dip_at_reward_depth,
        "peak_bins": peak_bins,
        "snc_tc_first": snc_tc_first, "snc_tc_last": snc_tc_last,
        "omit_tc": list(omit_bins), "base_tc": list(base_bins),
        "gates": gates, "provenance": prov,
    }


def run_td_lesion(seed, **kw):
    """ANTI-CHEAT (a): train, then zero cue->striosome and re-measure. The migration must
    VANISH (the SNc still bursts to the US reflex via the relay, but no cue burst, no dip).
    Re-uses run_td's machinery but cuts the cue conduit after acquisition by re-building +
    training, then lesioning, then a frozen test block over a few trials."""
    from sim.backend import get_backend
    import numpy as np
    xp, _ = get_backend()
    # Train normally first (short), then lesion the cue pathway and measure the time-course.
    n_train = kw.pop("n_train", 40)
    snc_reward_gain = kw.get("snc_reward_gain", 400.0)
    cue_drive_pa = kw.get("cue_drive_pa", 600.0)
    relay_tonic_pa = kw.get("relay_tonic_pa", 300.0)
    disinhib_pa = kw.get("disinhib_pa", 100.0)
    bin_steps = kw.get("bin_steps", 20)
    n_cs_bins = kw.get("n_cs_bins", 6); n_isi_bins = kw.get("n_isi_bins", 4); n_post_bins = kw.get("n_post_bins", 4)
    bridge, cfg = _build_stageb_bridge(
        seed, snc_da_sensitivity=kw.get("snc_da_sensitivity", 8.0),
        reward_learning_rate=kw.get("reward_learning_rate", 0.08),
        cue_to_strio_weight=kw.get("cue_to_strio_weight", 20.0),
        td_disinhibit=True, snc_drive_to_snc_weight=kw.get("snc_drive_to_snc_weight", 6.0),
        strio_to_disinhib_weight=kw.get("strio_to_disinhib_weight", 20.0),
        disinhib_tonic_weight=kw.get("disinhib_tonic_weight", 8.0), n_drive=40, n_relay=40)
    region_names = ("cue", "striosome_value", "snc", "snc_drive", "disinhib")
    idx_map = {n: xp.asarray(_idx(bridge, n)) for n in region_names}
    tonic_drives = {"snc_drive": relay_tonic_pa, "disinhib": disinhib_pa}
    _calibrate_da_threshold(bridge, cfg, idx_map, tonic_drives, xp)
    _calibrate_da_baseline(bridge, cfg, idx_map, tonic_drives, xp)
    n_win_bins = n_cs_bins + n_isi_bins + n_post_bins
    win_steps = n_win_bins * bin_steps
    cs_steps = (n_cs_bins + n_isi_bins) * bin_steps
    us_on_step = n_cs_bins * bin_steps; us_off_step = (n_cs_bins + max(1, n_isi_bins // 2)) * bin_steps
    isi_start_bin = n_cs_bins
    floor = {"snc_drive": relay_tonic_pa, "disinhib": disinhib_pa}
    # Train.
    for _ in range(n_train):
        _drive_timecourse(bridge, idx_map, floor, bin_steps * 2, xp, bin_steps)
        events = [(0, cs_steps, {"cue": cue_drive_pa}),
                  (us_on_step, us_off_step, {"snc_drive": relay_tonic_pa + snc_reward_gain})]
        _drive_timecourse(bridge, idx_map, floor, win_steps, xp, bin_steps, events=events)
    # LESION the cue->striosome conduit.
    n_cut = _lesion_pathway(bridge, "cue", "striosome_value")
    # Frozen test: predicted (CS+US) time-course + omission time-course.
    pred_events = [(0, cs_steps, {"cue": cue_drive_pa}),
                   (us_on_step, us_off_step, {"snc_drive": relay_tonic_pa + snc_reward_gain})]
    pred_bins, pred_strio, _, _ = _drive_timecourse(
        bridge, idx_map, floor, win_steps, xp, bin_steps, events=pred_events, freeze_lr=0.0, cfg=cfg)
    omit_events = [(0, cs_steps, {"cue": cue_drive_pa})]
    omit_bins, _, _, _ = _drive_timecourse(
        bridge, idx_map, floor, win_steps, xp, bin_steps, events=omit_events, freeze_lr=0.0, cfg=cfg)
    base_bins, _, _, _ = _drive_timecourse(bridge, idx_map, floor, win_steps, xp, bin_steps, freeze_lr=0.0, cfg=cfg)
    half = max(1, n_isi_bins // 2)
    us_rate = float(np.mean(pred_bins[isi_start_bin:isi_start_bin + half])) if pred_bins else 0.0
    cs_rate = float(np.mean(pred_bins[0:isi_start_bin])) if isi_start_bin > 0 else 0.0
    tonic = float(np.mean(base_bins)) if base_bins else 0.0
    v_cs = float(np.mean(pred_strio[0:isi_start_bin])) if isi_start_bin > 0 else 0.0
    omit_at_reward = float(np.mean(omit_bins[isi_start_bin:isi_start_bin + half])) if omit_bins else 0.0
    base_at_reward = float(np.mean(base_bins[isi_start_bin:isi_start_bin + half])) if base_bins else 0.0
    # EXPECTATION: cue is silenced -> V~0 -> no cue burst, no dip; the US reflex still bursts.
    cue_silenced = (v_cs <= 1e-3)
    no_cue_burst = (cs_rate <= 1.30 * tonic + 1e-6)
    no_dip = (base_at_reward - omit_at_reward) <= 0.5
    us_reflex_intact = (us_rate > 1.10 * tonic)
    return {
        "seed": seed, "n_cut": n_cut, "v_cs_hz": v_cs, "cs_rate": cs_rate, "us_rate": us_rate,
        "tonic_rate": tonic, "omit_at_reward_hz": omit_at_reward, "base_at_reward_hz": base_at_reward,
        "cue_silenced": bool(cue_silenced), "no_cue_burst": bool(no_cue_burst),
        "no_dip": bool(no_dip), "us_reflex_intact": bool(us_reflex_intact),
        "pred_tc": list(pred_bins), "omit_tc": list(omit_bins), "base_tc": list(base_bins),
    }


# ======================================================================================
# A-CSC — COMPLETE SERIAL COMPOUND tapped-delay TD cue-shift (escalation #2; design §2.1)
# ======================================================================================
#
# The cue is a CHAIN of K time-tagged sub-states (csc_0=cue@onset ... csc_{K-1}), each
# driving the critic through its OWN plastic synapse. Sub-state k is active during bin k of
# the trial window (the world's cue-presentation timing — legitimate, design §2.4). The
# reward (US) fires at `reward_bin`. The value DERIVATIVE is delivered to the SNc via the
# B-3 disinhibition relay (zero sim/ edit). TD back-propagates value one tap per trial -> the
# burst migrates from the reward onto the cue. The MULTIPLE sub-channels decouple the B-2
# single-channel conflict (each sub-state's value grows independently, sparse, STDP-friendly).


def _csc_substate_weights(bridge, K):
    """Per-sub-state value weight w_k = mean(csc_k -> striosome_value). The back-propagating
    value profile: late taps (near reward) grow first, then earlier taps, until csc_0."""
    return [_mean_pathway_weight(bridge, f"csc_{k}", "striosome_value") for k in range(K)]


def run_td_csc(seed, *, snc_reward_gain=400.0, csc_drive_pa=600.0,
               snc_tonic_pa=220.0,
               csc_to_strio_weight=14.0,
               snc_da_sensitivity=8.0, reward_learning_rate=0.08,
               n_csc=8, n_csc_per=25, reward_bin=None, n_post_bins=3,
               bin_steps=20, n_train=80, lesion_cue=False, unpaired=False,
               us_dur_bins=1, csc_eligibility_tau_ms=40.0,
               csc_gabab_level=True, csc_strio_to_snc_weight=2.5,
               csc_gabab_tau_decay=60.0,
               csc_conductance_deriv=True, csc_td_slow_tau_ms=400.0,
               csc_td_derivative_gain=1.0, csc_gabab_conductance_max=0.0,
               csc_critic_tonic_pa=0.0, csc_critic_teacher_pa=0.0,
               csc_stdp_w_max=None, csc_iti_bins=2,
               csc_fs_clamp=False, csc_to_fs_weight=20.0, csc_fs_to_strio_weight=12.0,
               csc_reward_relay=False, csc_reward_us_to_snc_weight=6.0,
               csc_strio_to_reward_us_weight=8.0, csc_reward_us_drive_pa=600.0,
               verbose=True):
    """A-CSC TD cue-shift Pavlovian protocol on the spiking SNc (escalation #2).

    Each TRIAL window = K sub-state bins + n_post_bins. Sub-state k (csc_k) is driven during
    bin k (the tapped delay = the world's cue-presentation timing). The SNc is driven DIRECTLY
    with the tonic pacemaker + the reward (US) during `reward_bin` (default = the last sub-state
    bin, K-1, so the reward overlaps csc_{K-1}). The value is delivered to the SNc as -V (the
    GABA_B/GIRK level subtraction, shrinks the reward burst) + dV/dt (the B-2 conductance-
    derivative, the bootstrap burst at the value's leading edge). Learning is ON; the critic
    learns each sub-state's value w_k from the SNc-derived dopamine. The SNc firing-rate
    TIME-COURSE is recorded per bin so the time-of-peak (and its migration toward the cue) is
    measurable.

    HEADLINE metric: Pearson r between TRIAL NUMBER and the SNc burst TIME-OF-PEAK. Migration
    => the peak moves EARLIER (toward csc_0, lower bin index) across learning => r < 0. Pass
    bar |r| > 0.7 with the correct (earlier) sign.

    Anti-cheats: lesion_cue zeros every csc_k->striosome after training (migration must vanish,
    US reflex remains); unpaired fires the US at a RANDOM bin unrelated to the chain (no
    contingency -> no migration).
    """
    from sim.backend import get_backend
    import numpy as np
    xp, _ = get_backend()
    K = int(n_csc)
    if reward_bin is None:
        reward_bin = K - 1                 # reward overlaps the last sub-state by default
    reward_bin = int(reward_bin)
    bridge, cfg = _build_stageb_bridge(
        seed, snc_da_sensitivity=snc_da_sensitivity,
        reward_learning_rate=reward_learning_rate,
        csc=True, n_csc=K, n_csc_per=int(n_csc_per),
        csc_to_strio_weight=csc_to_strio_weight,
        csc_eligibility_tau_ms=csc_eligibility_tau_ms,
        csc_gabab_level=csc_gabab_level, csc_strio_to_snc_weight=csc_strio_to_snc_weight,
        csc_gabab_tau_decay=csc_gabab_tau_decay,
        csc_conductance_deriv=csc_conductance_deriv,
        csc_td_slow_tau_ms=csc_td_slow_tau_ms, csc_td_derivative_gain=csc_td_derivative_gain,
        csc_gabab_conductance_max=csc_gabab_conductance_max, csc_stdp_w_max=csc_stdp_w_max,
        csc_fs_clamp=csc_fs_clamp, csc_to_fs_weight=csc_to_fs_weight,
        csc_fs_to_strio_weight=csc_fs_to_strio_weight,
        csc_reward_relay=csc_reward_relay,
        csc_reward_us_to_snc_weight=csc_reward_us_to_snc_weight,
        csc_strio_to_reward_us_weight=csc_strio_to_reward_us_weight)
    region_names = (tuple(f"csc_{k}" for k in range(K)) + ("striosome_value", "snc")
                    + (("reward_us",) if csc_reward_relay else ()))
    idx_map = {n: xp.asarray(_idx(bridge, n)) for n in region_names}

    # PROVENANCE / anti-cheat (3): no host TD term reaches the SNc. The SNc drive is
    # tonic + reward_us(direct) + synaptic GABA_B(-V) + the synaptic conductance-derivative(+dV/dt)
    # ONLY (the conductance-derivative is computed AT THE MEMBRANE from g_gabab, not a host term).
    assert cfg.current_reward_signal == 0.0, "host reward scalar must be 0 (brain-based)"
    assert cfg.reward_baseline == 0.0, "host reward baseline must be 0 (brain-based)"
    assert cfg.enable_td_value_derivative == bool(csc_conductance_deriv)
    _snc_terms = ("tonic(direct) + reward_us(synaptic relay; critic inhibits it = r-V) + "
                  "synaptic GABA_B(-V derivative source) + synaptic conductance-derivative(+dV/dt) only"
                  if csc_reward_relay else
                  "tonic + reward(direct at SNc) + synaptic GABA_B(-V) + synaptic conductance-derivative(+dV/dt) only")
    prov = {
        # With the reward relay, the reward r enters SYNAPTICALLY (reward_us->snc), not as a host
        # write at the SNc — the only direct SNc current is the tonic pacemaker. Without the relay,
        # the reward + tonic are the world's r/pacemaker written at the SNc.
        "snc_gets_direct_reward": (not bool(csc_reward_relay)),
        "reward_is_synaptic_relay": bool(csc_reward_relay),
        "host_reward_signal": float(cfg.current_reward_signal),
        "host_value_term": False,
        "snc_drive_terms": _snc_terms,
        "enable_td_value_derivative": bool(cfg.enable_td_value_derivative),
        "enable_gabab": bool(cfg.enable_gabab),
        "csc_substates": K, "reward_bin": reward_bin,
        "csc_value_synapses": "each csc_k -> striosome_value is an INDEPENDENT plastic synapse (the tap value w_k)",
    }

    # The critic gets a sub-threshold tonic (csc_critic_tonic_pa) that holds it near firing
    # threshold so a small per-tap weight produces GRADED firing (the MSN rheobase is otherwise
    # all-or-nothing -> the value can't grow smoothly from ~0 to back-propagate). Always on the
    # critic, including calibration (so the SNc baseline accounts for the small -V it produces).
    crit_tonic = ({"striosome_value": csc_critic_tonic_pa} if csc_critic_tonic_pa > 0 else {})

    # Calibrate the dopamine threshold + baseline at the tonic (floor) condition.
    tonic_drives = {"snc": snc_tonic_pa, **crit_tonic}
    tonic_frac = _calibrate_da_threshold(bridge, cfg, idx_map, tonic_drives, xp)
    tonic_conc = _calibrate_da_baseline(bridge, cfg, idx_map, tonic_drives, xp)
    if verbose:
        print(f"  [calib] K={K} sub-states, reward_bin={reward_bin}; SNc tonic frac={tonic_frac:.4f} "
              f"-> threshold; tonic da conc={tonic_conc:.4f} -> baseline")

    n_win_bins = K + int(n_post_bins)
    win_steps = n_win_bins * bin_steps
    floor = {"snc": snc_tonic_pa, **crit_tonic}

    import random as _random
    rng = _random.Random(seed)

    peak_bins = []         # time-of-peak (bin index) per trial
    snc_per_bin_hist = []  # full per-bin SNc rate per trial (for the heat-trace)
    v_substates_hist = []  # per-sub-state critic rate per trial
    w_substates_hist = []  # per-sub-state value weight per trial
    us_bin_rates = []      # SNc rate in the reward bin per trial
    cue_bin_rates = []     # SNc rate in the cue-onset (bin 0) per trial
    floor_rates = []       # SNc rate during the inter-trial floor (the IN-VIVO tonic the bursts ride on)
    snc_tc_first = snc_tc_last = None

    def _build_events(us_bin):
        ev = []
        for k in range(K):
            # The sub-state drive; csc_critic_tonic stays on the critic via the floor (steady).
            ev.append((k * bin_steps, (k + 1) * bin_steps, {f"csc_{k}": csc_drive_pa}))
        us0 = us_bin * bin_steps
        us1 = (us_bin + max(1, int(us_dur_bins))) * bin_steps
        if csc_critic_teacher_pa > 0:
            # CRITIC TEACHER (innate-reflex-teaches-learned-circuit): the reward (US) drives the
            # critic to FIRE during the reward window, so the reward-overlapping sub-state forms
            # CAUSAL eligibility -> the reward DA grows ITS value first -> the value GRADIENT
            # (steep near the reward) seeds the back-propagation. Without it the cold-start MSN
            # rheobase prevents any tap from firing, so no eligibility, no value, no migration.
            # (The teacher is added to the critic tonic, so the critic crosses threshold here.)
            ev.append((us0, us1, {"striosome_value": csc_critic_tonic_pa + csc_critic_teacher_pa}))
        if csc_reward_relay:
            # The reward enters at the EXCITATORY relay reward_us (which the critic inhibits ->
            # r - V reaches the SNc). The SNc keeps its tonic only (no direct reward), so the chain's
            # value does not suppress the SNc tonic — only the reward window is value-cancelled.
            ev.append((us0, us1, {"reward_us": csc_reward_us_drive_pa}))
        else:
            ev.append((us0, us1, {"snc": snc_tonic_pa + snc_reward_gain}))   # reward direct at the SNc
        return ev

    for t in range(n_train):
        # Inter-trial floor (settle). Long enough that the GABA_B (-V) conductance + its slow-EMA
        # DECAY back to ~0 before the next trial, so the floor's SNc sits at the live tonic (not a
        # residual -V suppression carried over from the chain). Record the floor SNc rate over the
        # LAST 2 floor bins = the in-vivo tonic the trial's bursts ride on (the correct gate ref).
        _, _, _f_snc, _ = _drive_timecourse(
            bridge, idx_map, floor, max(2, int(csc_iti_bins)) * bin_steps, xp, bin_steps)
        floor_rates.append(float(_f_snc))
        if unpaired:
            # ANTI-CHEAT (b): the US fires at a RANDOM bin unrelated to the chain.
            us_bin = rng.randint(0, max(0, n_win_bins - 1))
        else:
            us_bin = reward_bin
        events = _build_events(us_bin)
        snc_bins, strio_bins, _, _ = _drive_timecourse(
            bridge, idx_map, floor, win_steps, xp, bin_steps, events=events)
        peak_bin = int(np.argmax(snc_bins)) if snc_bins else 0
        peak_bins.append(peak_bin)
        snc_per_bin_hist.append(list(snc_bins))
        # Per-sub-state critic value = critic rate during that sub-state's bin.
        v_subs = [float(strio_bins[k]) if k < len(strio_bins) else 0.0 for k in range(K)]
        v_substates_hist.append(v_subs)
        w_substates_hist.append(_csc_substate_weights(bridge, K))
        us_bin_rates.append(float(snc_bins[reward_bin]) if reward_bin < len(snc_bins) else 0.0)
        cue_bin_rates.append(float(snc_bins[0]) if snc_bins else 0.0)
        if t == 0:
            snc_tc_first = list(snc_bins)
        if t == n_train - 1:
            snc_tc_last = list(snc_bins)
        if verbose and (t < 3 or t % 10 == 0 or t == n_train - 1):
            wprof = w_substates_hist[-1]
            wstr = " ".join(f"{w:.1f}" for w in wprof)
            print(f"  [csc t={t:02d}] peak_bin={peak_bin}  cue-bin={cue_bin_rates[-1]:5.1f}  "
                  f"US-bin={us_bin_rates[-1]:5.1f}Hz  w[k]=[{wstr}]")

    # A long settle so the slow GABA_B / slow-EMA conductances from the last training trial DECAY
    # to ~0 before the frozen test block (otherwise the residual -V ramps the measured baseline,
    # contaminating the tonic estimate + the gate thresholds).
    _settle_bins = max(8, int(csc_iti_bins) * 2)
    # --- Baseline (frozen, measured FIRST): floor only -> the clean tonic time-course. ---
    _drive_timecourse(bridge, idx_map, floor, _settle_bins * bin_steps, xp, bin_steps, freeze_lr=0.0, cfg=cfg)
    base_bins, _, _, _ = _drive_timecourse(
        bridge, idx_map, floor, win_steps, xp, bin_steps, freeze_lr=0.0, cfg=cfg)
    # --- Omission test (frozen): the full chain, NO US. The dip should be at the reward bin. ---
    _drive_timecourse(bridge, idx_map, floor, _settle_bins * bin_steps, xp, bin_steps, freeze_lr=0.0, cfg=cfg)
    omit_events = [(k * bin_steps, (k + 1) * bin_steps, {f"csc_{k}": csc_drive_pa}) for k in range(K)]
    omit_bins, _, _, _ = _drive_timecourse(
        bridge, idx_map, floor, win_steps, xp, bin_steps, events=omit_events, freeze_lr=0.0, cfg=cfg)

    trial_idx = list(range(n_train))
    r_migration = _pearson_r(trial_idx, peak_bins)
    early = slice(0, max(1, n_train // 5)); late = slice(-max(1, n_train // 5), None)
    peak_early = float(np.mean(peak_bins[early])); peak_late = float(np.mean(peak_bins[late]))
    cue_early = float(np.mean(cue_bin_rates[early])); cue_late = float(np.mean(cue_bin_rates[late]))
    us_early = float(np.mean(us_bin_rates[early])); us_late = float(np.mean(us_bin_rates[late]))
    base_rate = float(np.mean(base_bins)) if base_bins else 0.0   # bare baseline (no -V), for reporting
    # The IN-VIVO tonic the bursts ride on = the SNc rate during the inter-trial floor (the -V-
    # suppressed level the chain sits relative to), late in training. The bare baseline (no cue, no
    # residual -V) over-states the tonic because nothing suppresses it there; the floor is the
    # physiological reference for "burst above tonic" and "US transferred to tonic".
    tonic_rate = float(np.mean(floor_rates[late])) if floor_rates else base_rate
    # Per-sub-state value, early vs late (the back-propagation profile).
    v_arr = np.asarray(v_substates_hist, dtype=np.float64)   # (n_train, K)
    w_arr = np.asarray(w_substates_hist, dtype=np.float64)
    v_sub_early = v_arr[early].mean(axis=0).tolist()
    v_sub_late = v_arr[late].mean(axis=0).tolist()
    w_sub_early = w_arr[early].mean(axis=0).tolist()
    w_sub_late = w_arr[late].mean(axis=0).tolist()
    cue_v_early = float(v_arr[early, 0].mean()); cue_v_late = float(v_arr[late, 0].mean())

    half = max(1, int(us_dur_bins))
    omit_at_reward = float(np.mean(omit_bins[reward_bin:reward_bin + half])) if omit_bins else 0.0
    omit_at_cue = float(omit_bins[0]) if omit_bins else 0.0
    base_at_reward = float(np.mean(base_bins[reward_bin:reward_bin + half])) if base_bins else 0.0
    # Omission dip = the SNc at the expected-reward time falls BELOW the cue-time prediction burst
    # (and below the in-vivo tonic): the canonical "reward not delivered -> negative prediction
    # error AT the expected-reward time" signature.
    dip_at_reward_depth = omit_at_cue - omit_at_reward
    # The mid-chain GAP level (late trial) = the SNc between the cue burst and the reward (where the
    # value is flat -> derivative ~0 -> SNc ~tonic). The transfer is complete when the US burst has
    # shrunk to ~this gap level.
    gap_lo = max(1, reward_bin - 3); gap_hi = max(gap_lo + 1, reward_bin - 1)
    snc_last = snc_tc_last or []
    gap_late = (float(np.mean(snc_last[gap_lo:gap_hi])) if len(snc_last) > gap_hi else tonic_rate)

    # ---- Gates (design §4.2) ----
    migration_r_pass = (r_migration < -0.7)
    migration_dir_pass = (peak_late < peak_early - 0.5)
    # Late: burst at the cue, AND the US burst has TRANSFERRED (shrunk toward the in-chain gap/tonic
    # level), not merely shrunk a little. Reference = the in-vivo floor tonic + the mid-chain gap.
    cue_ref = max(tonic_rate, gap_late)
    late_burst_at_cue = (cue_late > 1.15 * cue_ref) and (us_late <= 1.40 * cue_ref + 1e-6)
    # Early: burst at US (above the in-vivo tonic).
    early_burst_at_us = (us_early > 1.15 * tonic_rate)
    # Omission dip at the reward bin (below the cue-time prediction burst).
    omission_dip_at_reward = (dip_at_reward_depth > 0) and (omit_at_reward < omit_at_cue + 1e-6)
    # Value grows on the cue sub-state (csc_0) across learning (the prerequisite).
    cue_value_grows = (cue_v_late > 1.20 * cue_v_early) if cue_v_early > 1e-6 else (cue_v_late > 1e-6)

    gates = {
        "migration_r_pass": bool(migration_r_pass),
        "migration_dir_pass": bool(migration_dir_pass),
        "early_burst_at_us": bool(early_burst_at_us),
        "late_burst_at_cue": bool(late_burst_at_cue),
        "omission_dip_at_reward": bool(omission_dip_at_reward),
        "cue_value_grows": bool(cue_value_grows),
    }

    return {
        "seed": seed, "lesion_cue": lesion_cue, "unpaired": unpaired,
        "mode": "td_csc", "n_train": n_train, "bin_steps": bin_steps,
        "n_csc": K, "reward_bin": reward_bin, "n_post_bins": int(n_post_bins),
        "r_migration": r_migration,
        "peak_bin_early": peak_early, "peak_bin_late": peak_late,
        "cue_rate_early": cue_early, "cue_rate_late": cue_late,
        "us_rate_early": us_early, "us_rate_late": us_late, "tonic_rate": tonic_rate,
        "base_rate_bare_hz": base_rate, "gap_late_hz": gap_late,
        "floor_rate_late_hz": tonic_rate,
        "cue_v_early_hz": cue_v_early, "cue_v_late_hz": cue_v_late,
        "v_sub_early": v_sub_early, "v_sub_late": v_sub_late,
        "w_sub_early": w_sub_early, "w_sub_late": w_sub_late,
        "omit_at_reward_hz": omit_at_reward, "omit_at_cue_hz": omit_at_cue,
        "base_at_reward_hz": base_at_reward, "dip_at_reward_depth_hz": dip_at_reward_depth,
        "peak_bins": peak_bins,
        "snc_tc_first": snc_tc_first, "snc_tc_last": snc_tc_last,
        "omit_tc": list(omit_bins), "base_tc": list(base_bins),
        "gates": gates, "provenance": prov,
    }


def run_td_csc_lesion(seed, **kw):
    """A-CSC ANTI-CHEAT (a): train, then zero EVERY csc_k->striosome and re-measure. The
    migration must VANISH (the SNc still bursts to the US reflex via the relay, but no cue
    burst, no dip). Proves the cue-time activity is the synaptic sub-state->critic conduit."""
    from sim.backend import get_backend
    import numpy as np
    xp, _ = get_backend()
    n_train = kw.pop("n_train", 60)
    K = int(kw.get("n_csc", 8))
    snc_reward_gain = kw.get("snc_reward_gain", 400.0)
    csc_drive_pa = kw.get("csc_drive_pa", 600.0)
    snc_tonic_pa = kw.get("snc_tonic_pa", 220.0)
    bin_steps = kw.get("bin_steps", 20)
    n_post_bins = kw.get("n_post_bins", 3)
    us_dur_bins = kw.get("us_dur_bins", 1)
    reward_bin = kw.get("reward_bin", None)
    if reward_bin is None:
        reward_bin = K - 1
    reward_bin = int(reward_bin)
    bridge, cfg = _build_stageb_bridge(
        seed, snc_da_sensitivity=kw.get("snc_da_sensitivity", 8.0),
        reward_learning_rate=kw.get("reward_learning_rate", 0.08),
        csc=True, n_csc=K, n_csc_per=int(kw.get("n_csc_per", 25)),
        csc_to_strio_weight=kw.get("csc_to_strio_weight", 14.0),
        csc_eligibility_tau_ms=kw.get("csc_eligibility_tau_ms", 40.0),
        csc_gabab_level=kw.get("csc_gabab_level", True),
        csc_strio_to_snc_weight=kw.get("csc_strio_to_snc_weight", 2.5),
        csc_gabab_tau_decay=kw.get("csc_gabab_tau_decay", 60.0),
        csc_conductance_deriv=kw.get("csc_conductance_deriv", True),
        csc_td_slow_tau_ms=kw.get("csc_td_slow_tau_ms", 400.0),
        csc_td_derivative_gain=kw.get("csc_td_derivative_gain", 1.0),
        csc_gabab_conductance_max=kw.get("csc_gabab_conductance_max", 0.0),
        csc_stdp_w_max=kw.get("csc_stdp_w_max", None),
        csc_fs_clamp=kw.get("csc_fs_clamp", False),
        csc_to_fs_weight=kw.get("csc_to_fs_weight", 20.0),
        csc_fs_to_strio_weight=kw.get("csc_fs_to_strio_weight", 12.0),
        csc_reward_relay=kw.get("csc_reward_relay", False),
        csc_reward_us_to_snc_weight=kw.get("csc_reward_us_to_snc_weight", 6.0),
        csc_strio_to_reward_us_weight=kw.get("csc_strio_to_reward_us_weight", 8.0))
    reward_relay = kw.get("csc_reward_relay", False)
    reward_us_drive_pa = kw.get("csc_reward_us_drive_pa", 600.0)
    region_names = (tuple(f"csc_{k}" for k in range(K)) + ("striosome_value", "snc")
                    + (("reward_us",) if reward_relay else ()))
    idx_map = {n: xp.asarray(_idx(bridge, n)) for n in region_names}
    crit_tonic_pa = kw.get("csc_critic_tonic_pa", 0.0)
    teacher_pa = kw.get("csc_critic_teacher_pa", 0.0)
    crit_tonic = ({"striosome_value": crit_tonic_pa} if crit_tonic_pa > 0 else {})
    tonic_drives = {"snc": snc_tonic_pa, **crit_tonic}
    _calibrate_da_threshold(bridge, cfg, idx_map, tonic_drives, xp)
    _calibrate_da_baseline(bridge, cfg, idx_map, tonic_drives, xp)
    n_win_bins = K + int(n_post_bins)
    win_steps = n_win_bins * bin_steps
    floor = {"snc": snc_tonic_pa, **crit_tonic}

    def _events(us_bin, teacher=True):
        ev = [(k * bin_steps, (k + 1) * bin_steps, {f"csc_{k}": csc_drive_pa}) for k in range(K)]
        us0 = us_bin * bin_steps; us1 = (us_bin + max(1, int(us_dur_bins))) * bin_steps
        if reward_relay:
            ev.append((us0, us1, {"reward_us": reward_us_drive_pa}))
        else:
            ev.append((us0, us1, {"snc": snc_tonic_pa + snc_reward_gain}))
        if teacher and teacher_pa > 0:
            ev.append((us0, us1, {"striosome_value": crit_tonic_pa + teacher_pa}))
        return ev

    iti_bins = max(2, int(kw.get("csc_iti_bins", 2)))
    for _ in range(n_train):
        _drive_timecourse(bridge, idx_map, floor, iti_bins * bin_steps, xp, bin_steps)
        _drive_timecourse(bridge, idx_map, floor, win_steps, xp, bin_steps, events=_events(reward_bin))
    # LESION every csc_k -> striosome.
    n_cut = sum(_lesion_pathway(bridge, f"csc_{k}", "striosome_value") for k in range(K))
    # Frozen test (NO teacher — the teacher is a training scaffold; the trained response must
    # stand on the learned csc->critic synapses, which the lesion has cut): predicted (chain + US)
    # + omission (chain, no US).
    pred_bins, pred_strio, _, _ = _drive_timecourse(
        bridge, idx_map, floor, win_steps, xp, bin_steps, events=_events(reward_bin, teacher=False),
        freeze_lr=0.0, cfg=cfg)
    omit_ev = [(k * bin_steps, (k + 1) * bin_steps, {f"csc_{k}": csc_drive_pa}) for k in range(K)]
    omit_bins, _, _, _ = _drive_timecourse(
        bridge, idx_map, floor, win_steps, xp, bin_steps, events=omit_ev, freeze_lr=0.0, cfg=cfg)
    base_bins, _, _, _ = _drive_timecourse(bridge, idx_map, floor, win_steps, xp, bin_steps, freeze_lr=0.0, cfg=cfg)
    half = max(1, int(us_dur_bins))
    us_rate = float(np.mean(pred_bins[reward_bin:reward_bin + half])) if pred_bins else 0.0
    cue_rate = float(pred_bins[0]) if pred_bins else 0.0
    tonic = float(np.mean(base_bins)) if base_bins else 0.0
    v_cue = float(pred_strio[0]) if pred_strio else 0.0
    omit_at_reward = float(np.mean(omit_bins[reward_bin:reward_bin + half])) if omit_bins else 0.0
    base_at_reward = float(np.mean(base_bins[reward_bin:reward_bin + half])) if base_bins else 0.0
    cue_silenced = (v_cue <= 1e-3)
    no_cue_burst = (cue_rate <= 1.30 * tonic + 1e-6)
    no_dip = (base_at_reward - omit_at_reward) <= 0.5
    us_reflex_intact = (us_rate > 1.10 * tonic)
    return {
        "seed": seed, "n_cut": n_cut, "v_cue_hz": v_cue, "cue_rate": cue_rate, "us_rate": us_rate,
        "tonic_rate": tonic, "omit_at_reward_hz": omit_at_reward, "base_at_reward_hz": base_at_reward,
        "cue_silenced": bool(cue_silenced), "no_cue_burst": bool(no_cue_burst),
        "no_dip": bool(no_dip), "us_reflex_intact": bool(us_reflex_intact),
        "pred_tc": list(pred_bins), "omit_tc": list(omit_bins), "base_tc": list(base_bins),
    }


def _print_td_csc_result(r):
    print()
    print(f"  SNc time-of-peak   : trial-early bin {r['peak_bin_early']:.2f} -> "
          f"trial-late bin {r['peak_bin_late']:.2f}   (reward bin = {r['reward_bin']}, cue = bin 0)")
    print(f"  migration r        : {r['r_migration']:+.3f}   "
          f"(MIGRATION = peak moves EARLIER/cue-ward => r < -0.7)")
    print(f"  cue-bin SNc rate   : {r['cue_rate_early']:.2f} -> {r['cue_rate_late']:.2f} Hz   "
          f"(tonic {r['tonic_rate']:.2f}Hz)")
    print(f"  US-bin SNc rate    : {r['us_rate_early']:.2f} -> {r['us_rate_late']:.2f} Hz   "
          f"(transferred if late US ~ tonic)")
    print(f"  V(strio) on cue    : {r['cue_v_early_hz']:.2f} -> {r['cue_v_late_hz']:.2f} Hz   "
          f"(cue value grows: {r['gates']['cue_value_grows']})")
    we = " ".join(f"{w:.1f}" for w in r["w_sub_early"])
    wl = " ".join(f"{w:.1f}" for w in r["w_sub_late"])
    print(f"  w[k] early -> late : [{we}] -> [{wl}]   (back-prop: late taps grow first, then earlier)")
    print(f"  omission @ reward  : {r['omit_at_reward_hz']:.2f} Hz vs @cue {r['omit_at_cue_hz']:.2f} Hz "
          f"vs base@reward {r['base_at_reward_hz']:.2f} Hz  (dip depth {r['dip_at_reward_depth_hz']:+.2f})")
    g = r["gates"]
    print(f"  gates: migration_r {g['migration_r_pass']} | dir {g['migration_dir_pass']} | "
          f"early@US {g['early_burst_at_us']} | late@cue {g['late_burst_at_cue']} | "
          f"omit-dip@reward {g['omission_dip_at_reward']} | cue-value-grows {g['cue_value_grows']}")


def _print_td_result(r):
    print()
    print(f"  SNc time-of-peak   : trial-early bin {r['peak_bin_early']:.2f} -> "
          f"trial-late bin {r['peak_bin_late']:.2f}   (ISI/reward onset = bin {r['isi_start_bin']})")
    print(f"  migration r        : {r['r_migration']:+.3f}   "
          f"(MIGRATION = peak moves EARLIER/cue-ward => r < -0.7)")
    print(f"  CS-window SNc rate : {r['cs_rate_early']:.2f} -> {r['cs_rate_late']:.2f} Hz   "
          f"(tonic {r['tonic_rate']:.2f}Hz)")
    print(f"  US-window SNc rate : {r['us_rate_early']:.2f} -> {r['us_rate_late']:.2f} Hz   "
          f"(transferred if late US ~ tonic)")
    print(f"  V(strio) on CS     : {r['v_cs_early_hz']:.2f} -> {r['v_cs_late_hz']:.2f} Hz   "
          f"(learned: {r['gates']['v_learned']})")
    print(f"  omission @ reward  : {r['omit_at_reward_hz']:.2f} Hz vs @cue {r['omit_at_cue_hz']:.2f} Hz "
          f"vs base@reward {r['base_at_reward_hz']:.2f} Hz  (dip depth {r['dip_at_reward_depth_hz']:+.2f})")
    g = r["gates"]
    print(f"  gates: migration_r {g['migration_r_pass']} | dir {g['migration_dir_pass']} | "
          f"early@US {g['early_burst_at_us']} | late@CS {g['late_burst_at_cs']} | "
          f"no-gap-burst {g['no_gap_burst']} | omit-dip@reward {g['omission_dip_at_reward']} | "
          f"v-learned {g['v_learned']}")


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


def _run_td_mode(args):
    """Orchestrate the TD cue-shift probe (single seed or multi-seed), with the two anti-cheats."""
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    td_kw = dict(
        snc_reward_gain=args.snc_reward_gain, cue_drive_pa=args.cue_drive_pa,
        relay_tonic_pa=args.relay_tonic_pa,
        disinhib_pa=(args.disinhib_pa if args.disinhib_pa != 250.0 else 100.0),
        cue_to_strio_weight=(args.cue_to_strio_weight if args.cue_to_strio_weight != 3.0 else 20.0),
        strio_to_disinhib_weight=args.strio_to_disinhib_weight,
        disinhib_tonic_weight=(args.disinhib_tonic_weight if args.disinhib_tonic_weight != 20.0 else 8.0),
        snc_drive_to_snc_weight=args.snc_drive_to_snc_weight,
        snc_da_sensitivity=args.snc_da_sensitivity,
        reward_learning_rate=args.reward_learning_rate,
        n_train=args.n_train,
        n_cs_bins=args.td_n_cs_bins, n_isi_bins=args.td_n_isi_bins,
        n_post_bins=args.td_n_post_bins, bin_steps=args.td_bin_steps,
    )
    results = []
    for s in seeds:
        if args.td_lesion_cue:
            print(f"[snc-TD seed={s}] CUE-LESION anti-cheat — train then zero cue->striosome:")
            lk = dict(td_kw); lk["unpaired"] = False
            r = run_td_lesion(s, **lk)
            print(f"  V(strio) on CS after lesion = {r['v_cs_hz']:.2f}Hz (cue silenced: {r['cue_silenced']})")
            print(f"  CS-rate={r['cs_rate']:.2f}Hz  US-rate={r['us_rate']:.2f}Hz  tonic={r['tonic_rate']:.2f}Hz")
            print(f"  omission@reward={r['omit_at_reward_hz']:.2f}Hz vs base@reward={r['base_at_reward_hz']:.2f}Hz")
            ok = r["cue_silenced"] and r["no_cue_burst"] and r["no_dip"] and r["us_reflex_intact"]
            print(f"  LESION anti-cheat (seed {s}): {'PASS' if ok else 'UNEXPECTED'}  "
                  f"[cue-silenced {r['cue_silenced']}, no-cue-burst {r['no_cue_burst']}, "
                  f"no-dip {r['no_dip']}, US-reflex-intact {r['us_reflex_intact']}]")
            r["_mode"] = "lesion"; results.append(r); print()
            continue
        tag = "UNPAIRED anti-cheat" if args.td_unpaired else "TD cue-shift (burst migration)"
        print(f"[snc-TD seed={s}] {tag} — does the SNc burst MIGRATE cue<-reward across learning?")
        r = run_td(s, unpaired=args.td_unpaired, **td_kw)
        _print_td_result(r)
        g = r["gates"]
        if args.td_unpaired:
            # Anti-cheat (b) EXPECTATION: no contingency -> no migration, no dip.
            no_mig = not (g["migration_r_pass"] and g["migration_dir_pass"])
            no_dip = not g["omission_dip_at_reward"]
            print(f"\n  UNPAIRED anti-cheat (seed {s}): {'PASS' if (no_mig and no_dip) else 'UNEXPECTED'}  "
                  f"[no-migration {no_mig}, no-dip {no_dip}]  (decoupled CS/US => no transfer)")
        else:
            headline = g["migration_r_pass"] and g["migration_dir_pass"]
            support = sum([g["early_burst_at_us"], g["late_burst_at_cs"], g["no_gap_burst"],
                           g["omission_dip_at_reward"], g["v_learned"]])
            verdict = ("GO" if (headline and support >= 4)
                       else "PARTIAL" if (g["migration_dir_pass"] or support >= 3)
                       else "NEGATIVE")
            print(f"\n  TD migration (seed {s}): {verdict}  "
                  f"[HEADLINE migration_r {g['migration_r_pass']} (r={r['r_migration']:+.3f}), "
                  f"dir {g['migration_dir_pass']}; support {support}/5]")
            r["_verdict"] = verdict
        r["_mode"] = "td"; results.append(r); print()

    if len(results) > 1 and not args.td_lesion_cue:
        if args.td_unpaired:
            n_ok = sum(1 for r in results
                       if not (r["gates"]["migration_r_pass"] and r["gates"]["migration_dir_pass"]))
            print(f"=== MULTI-SEED UNPAIRED: {n_ok}/{len(results)} show NO migration (anti-cheat holds) ===")
        else:
            n_go = sum(1 for r in results if r.get("_verdict") == "GO")
            n_partial = sum(1 for r in results if r.get("_verdict") == "PARTIAL")
            rs = ["{}={:+.3f}".format(r["seed"], r["r_migration"]) for r in results]
            print(f"=== MULTI-SEED TD: {n_go} GO + {n_partial} PARTIAL / {len(results)} ===")
            print("=== migration r per seed: " + ", ".join(rs) + " ===")
            # sign-consistency: all r same (negative/cue-ward) sign + omission-dip-at-reward
            signs = [(_r["r_migration"] < 0) for _r in results]
            dips = [_r["gates"]["omission_dip_at_reward"] for _r in results]
            print(f"=== sign-consistent (all cue-ward): {all(signs)} | "
                  f"omission-dip-at-reward all seeds: {all(dips)} ===")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"mode": "td_cue_shift", "results": results}, f, indent=2)
        print(f"  wrote {args.out}")


def _run_td_csc_mode(args):
    """Orchestrate the A-CSC TD cue-shift probe (escalation #2), with the two anti-cheats."""
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    kw = dict(
        snc_reward_gain=args.snc_reward_gain, csc_drive_pa=args.csc_drive_pa,
        snc_tonic_pa=args.snc_tonic_pa,
        csc_to_strio_weight=(args.csc_to_strio_weight if args.csc_to_strio_weight != 6.0 else 14.0),
        snc_da_sensitivity=args.snc_da_sensitivity,
        reward_learning_rate=args.reward_learning_rate,
        n_csc=args.csc_n, n_csc_per=args.csc_n_per,
        reward_bin=(args.csc_reward_bin if args.csc_reward_bin >= 0 else None),
        n_post_bins=args.csc_n_post_bins, bin_steps=args.td_bin_steps,
        us_dur_bins=args.csc_us_dur_bins, n_train=args.n_train,
        csc_eligibility_tau_ms=args.csc_eligibility_tau_ms,
        csc_gabab_level=(not args.csc_no_gabab_level),
        csc_strio_to_snc_weight=args.csc_strio_to_snc_weight,
        csc_gabab_tau_decay=args.csc_gabab_tau_decay,
        csc_conductance_deriv=(not args.csc_no_conductance_deriv),
        csc_td_slow_tau_ms=args.csc_td_slow_tau_ms,
        csc_td_derivative_gain=args.csc_td_derivative_gain,
        csc_critic_tonic_pa=args.csc_critic_tonic_pa,
        csc_critic_teacher_pa=args.csc_critic_teacher_pa,
        csc_gabab_conductance_max=args.csc_gabab_conductance_max,
        csc_stdp_w_max=(args.csc_stdp_w_max if args.csc_stdp_w_max > 0 else None),
        csc_iti_bins=args.csc_iti_bins,
        csc_fs_clamp=args.csc_fs_clamp,
        csc_to_fs_weight=args.csc_to_fs_weight,
        csc_fs_to_strio_weight=args.csc_fs_to_strio_weight,
        csc_reward_relay=args.csc_reward_relay,
        csc_reward_us_to_snc_weight=args.csc_reward_us_to_snc_weight,
        csc_strio_to_reward_us_weight=args.csc_strio_to_reward_us_weight,
        csc_reward_us_drive_pa=args.csc_reward_us_drive_pa,
    )
    results = []
    for s in seeds:
        if args.td_lesion_cue:
            print(f"[snc-CSC seed={s}] CUE-LESION anti-cheat — train then zero ALL csc_k->striosome:")
            r = run_td_csc_lesion(s, **kw)
            print(f"  V(strio) on cue after lesion = {r['v_cue_hz']:.2f}Hz (cue silenced: {r['cue_silenced']})")
            print(f"  cue-rate={r['cue_rate']:.2f}Hz  US-rate={r['us_rate']:.2f}Hz  tonic={r['tonic_rate']:.2f}Hz")
            print(f"  omission@reward={r['omit_at_reward_hz']:.2f}Hz vs base@reward={r['base_at_reward_hz']:.2f}Hz")
            ok = r["cue_silenced"] and r["no_cue_burst"] and r["us_reflex_intact"]
            print(f"  CSC LESION anti-cheat (seed {s}): {'PASS' if ok else 'UNEXPECTED'}  "
                  f"[cue-silenced {r['cue_silenced']}, no-cue-burst {r['no_cue_burst']}, "
                  f"no-dip {r['no_dip']}, US-reflex-intact {r['us_reflex_intact']}]")
            r["_mode"] = "csc_lesion"; results.append(r); print()
            continue
        tag = "UNPAIRED anti-cheat" if args.td_unpaired else "A-CSC TD cue-shift (burst migration)"
        print(f"[snc-CSC seed={s}] {tag} — does the SNc burst MIGRATE cue<-reward across learning?")
        r = run_td_csc(s, unpaired=args.td_unpaired, **kw)
        _print_td_csc_result(r)
        g = r["gates"]
        if args.td_unpaired:
            no_mig = not (g["migration_r_pass"] and g["migration_dir_pass"])
            print(f"\n  UNPAIRED anti-cheat (seed {s}): {'PASS' if no_mig else 'UNEXPECTED'}  "
                  f"[no-migration {no_mig}]  (US at random bin => no contingency => no transfer)")
        else:
            headline = g["migration_r_pass"] and g["migration_dir_pass"]
            support = sum([g["early_burst_at_us"], g["late_burst_at_cue"],
                           g["omission_dip_at_reward"], g["cue_value_grows"]])
            verdict = ("GO" if (headline and support >= 3)
                       else "PARTIAL" if (g["migration_dir_pass"] or support >= 2)
                       else "NEGATIVE")
            print(f"\n  A-CSC migration (seed {s}): {verdict}  "
                  f"[HEADLINE migration_r {g['migration_r_pass']} (r={r['r_migration']:+.3f}), "
                  f"dir {g['migration_dir_pass']}; support {support}/4]")
            r["_verdict"] = verdict
        r["_mode"] = "csc"; results.append(r); print()

    if len(results) > 1 and not args.td_lesion_cue:
        if args.td_unpaired:
            n_ok = sum(1 for r in results
                       if not (r["gates"]["migration_r_pass"] and r["gates"]["migration_dir_pass"]))
            print(f"=== MULTI-SEED CSC UNPAIRED: {n_ok}/{len(results)} show NO migration (anti-cheat holds) ===")
        else:
            n_go = sum(1 for r in results if r.get("_verdict") == "GO")
            n_partial = sum(1 for r in results if r.get("_verdict") == "PARTIAL")
            rs = ["{}={:+.3f}".format(r["seed"], r["r_migration"]) for r in results]
            print(f"=== MULTI-SEED A-CSC: {n_go} GO + {n_partial} PARTIAL / {len(results)} ===")
            print("=== migration r per seed: " + ", ".join(rs) + " ===")
            signs = [(_r["r_migration"] < 0) for _r in results]
            dips = [_r["gates"]["omission_dip_at_reward"] for _r in results]
            print(f"=== sign-consistent (all cue-ward): {all(signs)} | "
                  f"omission-dip-at-reward all seeds: {all(dips)} ===")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"mode": "td_csc_cue_shift", "results": results}, f, indent=2)
        print(f"  wrote {args.out}")


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
    # --- TD cue-shift (Pavlovian burst-migration) mode ---
    ap.add_argument("--td", action="store_true",
                    help="TD cue-shift: Pavlovian cue->reward protocol measuring whether the SNc "
                         "burst MIGRATES from the reward onto the cue across learning (B-3 "
                         "zero-edit value-derivative via disinhibition)")
    ap.add_argument("--disinhib-tonic-weight", type=float, default=20.0,
                    help="TD: disinhib->snc_drive inhibitory weight (the disinhibition stage)")
    ap.add_argument("--td-n-cs-bins", type=int, default=6, help="TD: CS-only bins per trial window")
    ap.add_argument("--td-n-isi-bins", type=int, default=4, help="TD: ISI bins (CS+US) per window")
    ap.add_argument("--td-n-post-bins", type=int, default=4, help="TD: post bins per window")
    ap.add_argument("--td-bin-steps", type=int, default=20, help="TD: sub-steps per time-course bin")
    ap.add_argument("--td-lesion-cue", action="store_true",
                    help="TD anti-cheat (a): train then zero cue->striosome; migration must vanish")
    ap.add_argument("--td-unpaired", action="store_true",
                    help="TD anti-cheat (b): decouple CS/US timing; no contingency -> no migration")
    # --- A-CSC TD cue-shift (complete-serial-compound tapped-delay) mode ---
    ap.add_argument("--td-csc", action="store_true",
                    help="A-CSC TD cue-shift (escalation #2): the cue is a CHAIN of K time-tagged "
                         "sub-states, each with its OWN plastic critic synapse, so TD back-propagates "
                         "value one tap per trial and the SNc burst MIGRATES onto the cue")
    ap.add_argument("--csc-n", type=int, default=8, help="A-CSC: number of cue sub-states (chain length K)")
    ap.add_argument("--csc-n-per", type=int, default=25, help="A-CSC: neurons per sub-state population")
    ap.add_argument("--csc-drive-pa", type=float, default=600.0, help="A-CSC: per-sub-state drive (pA)")
    ap.add_argument("--csc-to-strio-weight", type=float, default=6.0,
                    help="A-CSC: initial csc_k->striosome plastic weight (the tap value w_k seed)")
    ap.add_argument("--csc-reward-bin", type=int, default=-1,
                    help="A-CSC: bin index where the US fires (default -1 => last sub-state K-1)")
    ap.add_argument("--csc-us-dur-bins", type=int, default=1, help="A-CSC: US duration in bins")
    ap.add_argument("--csc-n-post-bins", type=int, default=3, help="A-CSC: post bins after the chain")
    ap.add_argument("--csc-eligibility-tau-ms", type=float, default=40.0,
                    help="A-CSC: eligibility-trace tau (ms); SHORT (~40) for tap-local credit so "
                         "TD back-propagates one tap per trial (default 1000ms smears credit)")
    ap.add_argument("--csc-no-gabab-level", action="store_true",
                    help="A-CSC: DISABLE the GABA_B -V level channel (derivative-only; ablation). "
                         "Default ON: -V shrinks the reward burst so the peak can migrate")
    ap.add_argument("--csc-strio-to-snc-weight", type=float, default=2.5,
                    help="A-CSC: striosome->snc GABA_B (-V level) weight")
    ap.add_argument("--csc-gabab-tau-decay", type=float, default=60.0,
                    help="A-CSC: GABA_B (-V) decay tau (ms); SHORT so -V tracks the per-tap value "
                         "(SNc tonic uses the existing --snc-tonic-pa, default 220)")
    ap.add_argument("--csc-no-conductance-deriv", action="store_true",
                    help="A-CSC: DISABLE the B-2 conductance-derivative +dV/dt channel (ablation: "
                         "the bootstrap source). Default ON (the protected edit, byte-identical when OFF)")
    ap.add_argument("--csc-td-slow-tau-ms", type=float, default=400.0,
                    help="A-CSC: the slow-EMA tau of g_gabab (ms); the derivative = g_gabab - g_gabab_slow")
    ap.add_argument("--csc-td-derivative-gain", type=float, default=1.0,
                    help="A-CSC: scales the conductance-derivative (+dV/dt) current onto the SNc")
    ap.add_argument("--csc-critic-tonic-pa", type=float, default=0.0,
                    help="A-CSC: sub-threshold tonic on the critic (graded firing so the value can "
                         "grow smoothly from ~0; the MSN rheobase is otherwise all-or-nothing)")
    ap.add_argument("--csc-critic-teacher-pa", type=float, default=0.0,
                    help="A-CSC: critic teacher current during the reward window (the US fires the "
                         "critic so the reward-adjacent tap forms eligibility -> seeds the value "
                         "gradient that back-propagates; innate-reflex-teaches-learned-circuit)")
    ap.add_argument("--csc-gabab-conductance-max", type=float, default=0.0,
                    help="A-CSC: GIRK saturation cap on g_gabab (owner-approved guardrail). Bounds "
                         "the -V so a HOT critic can't clamp the SNc dead (keeps the tonic alive). 0=off")
    ap.add_argument("--csc-stdp-w-max", type=float, default=0.0,
                    help="A-CSC: cap the per-tap weight growth (stdp soft-bound) so the critic stays "
                         "SPARSE (value in the graded MSN band, not the dense runaway). 0=use default(40)")
    ap.add_argument("--csc-iti-bins", type=int, default=2,
                    help="A-CSC: inter-trial floor duration in bins (long enough for the GABA_B -V to "
                         "decay so the floor sits at the live tonic)")
    ap.add_argument("--csc-fs-clamp", action="store_true",
                    help="A-CSC: add the production critic FS-clamp (csc->csc_fs->critic feedforward "
                         "inhibition) so the value stays SPARSE as weights grow (decouples value-growth "
                         "from dense firing -> -V doesn't saturate -> tonic survives)")
    ap.add_argument("--csc-to-fs-weight", type=float, default=20.0, help="A-CSC: csc_k->csc_fs weight")
    ap.add_argument("--csc-fs-to-strio-weight", type=float, default=12.0, help="A-CSC: csc_fs->critic weight")
    ap.add_argument("--csc-reward-relay", action="store_true",
                    help="A-CSC multi-channel: route the reward via an EXCITATORY relay reward_us->snc "
                         "that the critic INHIBITS (r-V reaches the SNc). Localizes -V to the reward "
                         "(catalog C.33), so the chain's value doesn't suppress the SNc tonic -> the "
                         "cue burst survives at a physiological tonic + the US fully vacates")
    ap.add_argument("--csc-reward-us-to-snc-weight", type=float, default=6.0, help="A-CSC: reward_us->snc weight")
    ap.add_argument("--csc-strio-to-reward-us-weight", type=float, default=8.0,
                    help="A-CSC: critic->reward_us inhibitory weight (the -V that cancels r at the reward)")
    ap.add_argument("--csc-reward-us-drive-pa", type=float, default=600.0,
                    help="A-CSC: the reward drive onto reward_us (the world's r afferent)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.diag:
        run_diag(args.seed, cue_drive_pa=args.cue_drive_pa,
                 cue_to_strio_weight=args.cue_to_strio_weight,
                 strio_to_snc_weight=args.strio_to_snc_weight)
        return

    if args.td_csc:
        _run_td_csc_mode(args)
        return

    if args.td:
        _run_td_mode(args)
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
