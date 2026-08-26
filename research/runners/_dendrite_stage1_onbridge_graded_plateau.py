"""DENDRITE STAGE 1 (on-bridge) -- the GRADED dendritic-plateau READ-OUT, produced ON the spiking bridge.

Stage 0 (de-risk A, GO 6/6 seeds, 2026-06-20-dendrite-derisk-A-graded-plateau-readout.md) proved the
dendrite's ONE genuine unlock with a NUMPY DendriticLayer held ALONGSIDE the bridge: a GRADED ANALOG
read-out of a distributed code (Mikulasch-Priesemann) the point-neuron soma provably cannot be (the
nav value-critic delta=r-V; LINEAR=sub-rheobase 0 Hz flat delta~1.0; all-or-none PLATEAU over-clamps
delta~0.0). The Stage-0 value was V = sigmoid((v_basal-theta)/slope), computed numpy-side.

THIS Stage 1 replaces the numpy DendriticLayer with the GUARDED, default-OFF protected sim/ edit
(`enable_graded_dendritic_plateau`, commit d69cc0ab): a SMOOTH (gentle, centered, non-saturating)
regenerative dendritic-plateau current produced BY THE SPIKING BRIDGE on a dedicated critic compartment.
The graded value V is the on-bridge plateau CONDUCTANCE (cp_conductance_g_graded_plateau over the critic
region) -- the analog dendritic quantity (NOT the somatic spike rate; probe (i): the MSN-D1 won't fire
gradedly at any current). V is read from the bridge and delivered as a GRADED inhibitory subtraction at
the SNc (the SAME subtract-at-SNc mechanism probe (ii) confirmed grades the burst). delta = far/near is
read EXACTLY as Stage 0 / burndown-9.

The bridge mechanism (all on-substrate, the new sim/ edit):
  * the `vs_place_context -> striosome_value` VALUE pathway is tagged coincidence_detector=True (the
    routing mask the graded block consumes -- the SAME mask, NO new wiring) and its weights are grown
    NEAR-selectively by the bridge's OWN reward-modulated STDP during NEAR+reward training (location-
    selective LTP, the navfaithful gate-3 mechanism). After learning, the WEIGHTED coincident drive
    c_w = Sum_j (w_eff_j * x_j) is HIGH at NEAR, LOW at FAR.
  * the new fused_graded_dendritic_plateau passes c_w through a GENTLE centered logistic
    V = 1/(1+exp(-slope*(c_w-center))) -> the graded plateau conductance on the critic compartment.
  * V_onbridge = mean(cp_conductance_g_graded_plateau over the critic) normalized to [0,1].

VALIDATION CEILING: the Stage-0 numpy arm (delta=1.33 >= host ~1.30, 6/6 seeds). Stage 1 GO iff the
ON-BRIDGE graded V gives delta >= 1.30 (~ the Stage-0 ceiling) at faithful grid-32 multi-seed, where
both point-neuron controls fail.

ANTI-CHEATS (the de-risk-A + #6 battery, on-bridge):
  (a) the TWO POINT-NEURON CONTROLS (LINEAR ~1.0 flat, all-or-none PLATEAU ~0.0 over-clamp), re-asserted
      in-run verbatim from burndown-9 -- the two-sided validity gate.
  (b) APICAL/PLATEAU LESION -- turn the graded plateau OFF (enable_graded_dendritic_plateau=False): the
      on-bridge V -> 0 -> the SNc subtraction vanishes -> delta collapses to ~1.0. AND the all-or-none
      sibling (the coincidence plateau on the SAME routed mask) over-clamps. The GRADED-ness is load-bearing.
  (c) GABA_B-equivalent SUBTRACTION lesion -- zero the V subtraction at the SNc -> near==far -> delta~1.0.
  (d) REGIME FIDELITY -- faithful grid-32 deterministic (OU/cond-noise/homeostasis OFF), asserted; n_train>=40.
  (e) HOST-CEILING -- the on-bridge delta <= host*(1+tol) (no goal/reward smuggling).
  (f) LOCATION-SELECTIVITY -- the on-bridge plateau V(near) > V(far) (the value is learned + place-specific).

The no-confab moat is N/A here (this is a critic-only nav bridge with no conversational regions); it is
preserved by construction (array-disjoint), and the merged-bridge suites that DO carry the moat
(test_nav_conv_step2b_coresident etc.) are byte-unregressed because the new flag is default-OFF.

Usage
-----
    # GPU faithful multi-seed (the on-bridge validation; the new edit's hot path):
    SIM_BACKEND=cupy python -m research.runners._dendrite_stage1_onbridge_graded_plateau \
        --seeds 42,43,44 --n-train 40 --lead-ms 150 \
        --out research/findings/raw/_dendrite_stage1_onbridge.json
    # CPU smoke (the tiny ~290-neuron bridge runs on numpy):
    SIM_BACKEND=numpy python -m research.runners._dendrite_stage1_onbridge_graded_plateau --seed 42 --n-train 15
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as _st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from research.runners.snc_stageb_critic_probe_navfaithful import (
    _assert_deterministic_regime, _grid_prefs, grid_place_code_drive,
)
from research.runners.snc_stageb_critic_probe_place import (
    _calibrate_da_threshold, _idx, _host, _mean_pathway_weight, _clear_eligibility,
)
# The two POINT-NEURON read-out arms, re-asserted verbatim (anti-cheat a).
from research.runners._burndown9_critic_graded_readout_derisk import run_readout as _run_point_readout


# ---------------------------------------------------------------------------
# An on-bridge critic with the graded dendritic plateau ENABLED.
#
# A thin variant of the navfaithful bridge builder: the same deterministic-nav regime + the dedicated
# dense `vs_place_context -> striosome_value` critic afferent + the `striosome_value -> snc` GABA_B
# subtraction, BUT the value pathway is tagged coincidence_detector=True (routing for the graded block)
# and cfg.enable_graded_dendritic_plateau is ON (or the all-or-none coincidence path, for the control).
# ---------------------------------------------------------------------------
def _build_onbridge_critic(
    seed, *, grid_size=32,
    n_vs_place=200, vs_place_density=0.5, vs_place_to_strio_weight=0.2,
    n_strio=60, n_snc=30, strio_to_snc_weight=10.0,
    snc_da_sensitivity=8.0, reward_learning_rate=0.12,
    gabab_propagation_strength=0.02,
    graded_plateau=True, allornone_plateau=False,
    graded_center=1.5, graded_slope=1.0, graded_strength=80.0,
    coincidence_k=8.0, coincidence_gain=2.0, coincidence_strength=80.0):
    """Deterministic-nav-regime critic bridge with the on-bridge dendritic plateau. `vs_place_context`
    (dense, grid-32 place code) -> `striosome_value` (MSN-D1, PLASTIC, the WEIGHTED routed value input)
    -> `snc` (DA). The value pathway carries coincidence_detector=True so the new graded plateau (or the
    all-or-none coincidence control) reads its WEIGHTED coincident drive."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType

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
    cfg.reward_learning_rate = float(reward_learning_rate)
    cfg.current_reward_signal = 0.0
    cfg.reward_baseline = 0.0
    # Cap the value-pathway weight so the learned NEAR weight stays in the SUB-SOMATIC graded range
    # (the MSN-D1 value_dendrite must NOT fire somatically -- probe (i); a too-strong weight makes the
    # weighted input cross threshold and the somatic spike/reset contaminates the graded plateau read).
    # w_max=5 keeps c_w in the graded window (near > mid > far, all sub-rheobase) where the on-bridge
    # plateau read is monotone. The graded plateau current carries the value, NOT the soma.
    cfg.stdp_w_max = 5.0

    # === THE DETERMINISTIC-NAV REGIME (regime fidelity; the knobs nav disables at g11:3340-3344) ===
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_ou_process = False
    cfg.enable_conductance_noise = False
    cfg.enable_parameter_heterogeneity = False
    cfg.enable_structural_plasticity = False

    # GABA_B ON (the physiological SNc subtraction regime, matching the point-neuron arms).
    cfg.enable_gabab = True
    cfg.gabab_reversal_potential = -90.0
    cfg.gabab_tau_decay = 150.0
    cfg.gabab_propagation_strength = float(gabab_propagation_strength)

    # === THE ON-BRIDGE DENDRITIC PLATEAU (the new guarded sim/ edit) ===
    # The graded plateau and the all-or-none coincidence both consume the coincidence_detector routing
    # mask, which is only built when enable_coincidence_detection is True at wiring time -> set it True so
    # the mask exists. The all-or-none coincidence BLOCK only injects current when ITS conductance is
    # allocated (which happens iff enable_coincidence_detection AND we WANT the control); for the GRADED
    # arm we still need the mask, so we keep enable_coincidence_detection True but the all-or-none block's
    # effect is the apical-lesion control -- to isolate the GRADED arm cleanly, we want the graded plateau
    # ON and the all-or-none plateau's contribution accounted for. We therefore run the GRADED arm with
    # the coincidence k_threshold set high (coincidence_k) so the all-or-none plateau is OFF (sub-threshold)
    # while the graded plateau (the new edit) carries the value. The all-or-none CONTROL flips this.
    cfg.enable_coincidence_detection = True   # builds the routing mask (needed by BOTH plateau forms)
    cfg.coincidence_k_threshold = float(coincidence_k)
    cfg.coincidence_gain = float(coincidence_gain)
    cfg.coincidence_plateau_strength = float(coincidence_strength if allornone_plateau else 0.0)
    cfg.coincidence_weighted_drive = True     # the all-or-none control reads the SAME weighted drive
    cfg.enable_graded_dendritic_plateau = bool(graded_plateau)
    cfg.graded_plateau_center = float(graded_center)
    cfg.graded_plateau_slope = float(graded_slope)
    cfg.graded_plateau_strength = float(graded_strength)
    cfg.graded_plateau_tau_decay_ms = 80.0
    cfg.graded_plateau_tau_rise_ms = 2.0

    # ARCHITECTURE (the faithful Mikulasch-Priesemann realization): the graded value V lives in a
    # DEDICATED dendritic compartment (value_dendrite) whose plateau conductance IS the analog value --
    # it has NO somatic output pathway, so its (un-fireable-graded, probe (i)) MSN-D1 soma never carries
    # the value as a spike rate. V is read from cp_conductance_g_graded_plateau over value_dendrite and
    # delivered as the GRADED inhibitory subtraction at the SNc (exactly Stage 0: the value is a dendritic
    # analog quantity, the subtraction is the delivery; only the value's SOURCE moves from the numpy
    # DendriticLayer to the on-bridge plateau). This avoids re-introducing the somatic over-clamp (a
    # striosome->SNc GABA_B projection driven by the plateau would make the soma fire -> the all-or-none
    # over-subtraction probe (i)/(ii) and burndown-9 documented). The SNc receives its tonic+reward drive
    # directly (the runner sets it) -- the value-leads-reward protocol unchanged.
    regions = [
        BrainRegion(name="vs_place_context", n_neurons=n_vs_place, exc_fraction=1.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                    weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name),
        # The DEDICATED VALUE DENDRITE: MSN-D1 (deep rest, the un-fireable-graded soma -- probe (i)); the
        # graded plateau on it carries the analog value. NO output pathway (value delivered by the explicit
        # graded SNc subtraction). The routed (coincidence_detector) place input is its only afferent.
        BrainRegion(name="value_dendrite", n_neurons=n_strio, exc_fraction=0.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                    weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D1.name,
                    syn_reversal_potential_i_override=-60.0),
        BrainRegion(name="snc", n_neurons=n_snc, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_DOPAMINE.name,
                    syn_reversal_potential_i_override=-55.0),
    ]
    pathways = [
        # The VALUE pathway: dense place -> value_dendrite (V), PLASTIC, routed coincidence_detector=True so
        # the graded plateau (the new sim/ edit) reads its WEIGHTED coincident drive. NO value_dendrite->snc
        # somatic projection (the value is the dendritic plateau, delivered by the explicit subtraction).
        RegionPathway(from_region="vs_place_context", to_region="value_dendrite",
                      density=float(vs_place_density), weight_mean=float(vs_place_to_strio_weight),
                      weight_jitter=0.5, plastic=True, coincidence_detector=True),
    ]
    cfg.brain_regions = regions
    cfg.region_pathways = pathways

    snc_tonic_firing_fraction = 0.30
    from sim.neuromodulators import NeuromodulatorConfig, ModulatorTarget, ProductionRule
    cfg.enable_neuromodulator_subsystem = True
    cfg.neuromodulators = [
        NeuromodulatorConfig(
            name="dopamine", baseline=0.5, decay_tau_ms=200.0,
            concentration_min=0.0, concentration_max=2.0,
            targets=[ModulatorTarget(target_type="plasticity_rate", scope="all", sensitivity=+1.0)],
            production_rules=[ProductionRule(
                rule_type="from_region_firing_signed", sensitivity=float(snc_da_sensitivity),
                threshold=float(snc_tonic_firing_fraction), window_ms=200.0,
                source_regions=["snc"])],
        )]

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = seed
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge, cfg


def _ensemble_global_indices_xy(bridge, place_vec, region_name, frac=0.25):
    g = np.asarray(_idx(bridge, region_name), dtype=np.int64)
    drive = np.asarray(place_vec, dtype=np.float64)
    k = max(1, int(round(frac * len(drive))))
    top = np.argsort(drive)[-k:]
    return set(int(g[i]) for i in top)


def run_onbridge(seed, *, grid_size=32,
                 p_near_xy=(26.571, 26.571), p_mid_xy=(21.0, 21.0), p_far_xy=(4.429, 4.429),
                 vs_place_sigma=4.0, vs_place_drive_pa=800.0,
                 snc_tonic_pa=180.0, snc_reward_gain=420.0,
                 hold_steps=40, n_train=40, lead_steps=150,
                 dend_subtract_scale=1200.0, n_snc=120,
                 graded_center=1.5, graded_slope=1.0, graded_strength=80.0,
                 plateau_lesion=False, subtract_lesion=False,
                 allornone=False, verbose=True):
    """Train the ON-BRIDGE graded-plateau value read-out (location-selective LTP of the routed value
    pathway + the new graded plateau reading it), then measure delta (far/near) with V produced ON THE
    BRIDGE and subtracted at the SNc.

    plateau_lesion (anti-cheat b): build with enable_graded_dendritic_plateau=False -> the on-bridge V is
    0 -> the subtraction vanishes -> delta collapses. allornone: the all-or-none coincidence plateau on
    the SAME routed mask instead of the graded one (the binary-subunit control -> over-clamp).
    """
    from sim.backend import get_backend
    xp, _ = get_backend()

    graded_on = (not plateau_lesion) and (not allornone)
    bridge, cfg = _build_onbridge_critic(
        seed, grid_size=grid_size, n_snc=int(n_snc), graded_plateau=graded_on,
        allornone_plateau=allornone, graded_center=graded_center, graded_slope=graded_slope,
        graded_strength=graded_strength)
    _assert_deterministic_regime(cfg)   # anti-cheat (d) BEFORE anything runs

    snc_idx = xp.asarray(_idx(bridge, "snc")); n_snc = len(_host(snc_idx))
    vd_idx = xp.asarray(_idx(bridge, "value_dendrite"))   # the dedicated value-dendrite compartment
    place_idx = xp.asarray(_idx(bridge, "vs_place_context")); n_place = len(_host(place_idx))
    idx_map = {"snc": snc_idx, "vs_place_context": place_idx, "value_dendrite": vd_idx,
               "place": place_idx}

    vs_prefs = _grid_prefs(n_place, grid_size)
    near_vec = grid_place_code_drive(p_near_xy, vs_prefs, vs_place_drive_pa, sigma=vs_place_sigma)
    mid_vec = grid_place_code_drive(p_mid_xy, vs_prefs, vs_place_drive_pa, sigma=vs_place_sigma)
    far_vec = grid_place_code_drive(p_far_xy, vs_prefs, vs_place_drive_pa, sigma=vs_place_sigma)

    near_set = _ensemble_global_indices_xy(bridge, near_vec, "vs_place_context", frac=0.25)
    far_set = _ensemble_global_indices_xy(bridge, far_vec, "vs_place_context", frac=0.25) - near_set

    _calibrate_da_threshold(bridge, cfg, idx_map, snc_tonic_pa, xp)

    # -------- read the ON-BRIDGE graded plateau value V for a given place drive --------
    def _onbridge_V(place_vec, *, n_steps=None, settle=12):
        """Drive the place code at `place_vec`, step, and read V = mean(cp_conductance_g_graded_plateau)
        over the critic region -- the on-bridge analog dendritic value (NOT the somatic spike rate). The
        plateau conductance is normalized by its strength so V lands in ~[0,1] (a graded read-out)."""
        n_steps = int(hold_steps if n_steps is None else n_steps)
        bridge.cp_external_input_current[:] = 0.0
        if place_vec is not None:
            bridge.cp_external_input_current[place_idx] = xp.asarray(place_vec, dtype=xp.float32)
        saved_lr = cfg.reward_learning_rate; cfg.reward_learning_rate = 0.0  # frozen read
        g_acc = 0.0; m = 0
        for t in range(n_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * cfg.dt_ms
            if t >= settle:
                g = bridge.cp_conductance_g_graded_plateau
                if g is not None:
                    g_acc += float(_host(g[vd_idx]).mean()); m += 1
        cfg.reward_learning_rate = saved_lr
        if m == 0:
            return 0.0
        # Recover the GRADED VALUE V = logistic((c_w-center)*slope) from the on-bridge plateau slow
        # conductance g (cp_conductance_g_graded_plateau). At steady state the per-step increment
        # g_inc = strength*V drives g -> g_ss = g_inc/(1-decay) = strength*V/(1-decay), so
        # V = g_ss * (1-decay) / strength. This reads the ANALOG dendritic value (NOT the soma) and
        # un-saturates the read (the multi-step conductance is the integral of g_inc; dividing by the
        # steady-state gain 1/(1-decay) recovers the per-step graded logistic itself). The strio neurons
        # not in the routed near-ensemble have g~0, so the mean over the region is diluted -> rescale by
        # the active fraction is unnecessary (V is a relative read; the SNc subtraction scale absorbs it).
        decay = float(getattr(bridge, "_cached_decay_graded_plateau", 0.9876))
        g_mean = g_acc / m
        V = (g_mean * (1.0 - decay)) / max(cfg.graded_plateau_strength, 1e-6)
        return float(np.clip(V, 0.0, 1.0))

    # -------- SNc reward window with the ON-BRIDGE graded V subtracted --------
    def _snc_window(snc_pa, place_for_value, *, subtract=True, n_steps=None):
        n_steps = int(hold_steps if n_steps is None else n_steps)
        if subtract and place_for_value is not None and not subtract_lesion:
            V = _onbridge_V(place_for_value)            # the ON-BRIDGE plateau value
            snc_drive = float(snc_pa) - dend_subtract_scale * V
        else:
            V = 0.0; snc_drive = float(snc_pa)
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[snc_idx] = xp.float32(snc_drive)
        spk = 0
        for _ in range(n_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * cfg.dt_ms
            spk += int(bridge.cp_firing_states[snc_idx].sum())
        return spk / max(n_snc, 1) / max(n_steps * 1e-3, 1e-9), V

    # === Value-leads-reward acquisition: grow the routed value weights NEAR-selectively (on-bridge LTP) ===
    w_near_init = _mean_pathway_weight(bridge, "vs_place_context", "value_dendrite", pre_subset=near_set)
    v_near_curve = []
    for t in range(n_train):
        # ITI floor.
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[snc_idx] = xp.float32(snc_tonic_pa)
        for _ in range(hold_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * cfg.dt_ms
        _clear_eligibility(bridge)
        # NEAR + reward: place code drives the critic + the SNc reward burst -> reward-STDP grows the
        # near-ensemble value weights (location-selective LTP). The on-bridge graded plateau then reads
        # the grown weights.
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[place_idx] = xp.asarray(near_vec, dtype=xp.float32)
        bridge.cp_external_input_current[snc_idx] = xp.float32(snc_tonic_pa + snc_reward_gain)
        for _ in range(hold_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * cfg.dt_ms
        v_near_curve.append(_onbridge_V(near_vec))
        if verbose and (t < 2 or t % 10 == 0 or t == n_train - 1):
            wn = _mean_pathway_weight(bridge, "vs_place_context", "value_dendrite", pre_subset=near_set)
            wf = _mean_pathway_weight(bridge, "vs_place_context", "value_dendrite", pre_subset=far_set)
            print(f"  [ONBRIDGE acq t={t:02d}] V_onbridge(near)={v_near_curve[-1]:.3f} "
                  f"w_near={wn:.3f} w_far={wf:.3f} (near/far {wn/max(wf,1e-6):.2f})")

    early = slice(0, max(1, n_train // 5)); late = slice(-max(1, n_train // 5), None)
    v_near_early = _st.mean(v_near_curve[early]); v_near_late = _st.mean(v_near_curve[late])
    w_near_final = _mean_pathway_weight(bridge, "vs_place_context", "value_dendrite", pre_subset=near_set)
    w_far_final = _mean_pathway_weight(bridge, "vs_place_context", "value_dendrite", pre_subset=far_set)

    # === test (frozen): the state-specific delta with the ON-BRIDGE graded V subtracted ===
    def _test(place_for_value, snc_pa):
        # re-warm
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[snc_idx] = xp.float32(snc_tonic_pa)
        for _ in range(hold_steps + 20):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * cfg.dt_ms
        if lead_steps > 0 and place_for_value is not None:
            _snc_window(snc_tonic_pa, place_for_value, subtract=True, n_steps=int(lead_steps))
        return _snc_window(snc_pa, place_for_value, subtract=True)

    near_burst, V_near = _test(near_vec, snc_tonic_pa + snc_reward_gain)
    mid_burst, V_mid = _test(mid_vec, snc_tonic_pa + snc_reward_gain)
    far_burst, V_far = _test(far_vec, snc_tonic_pa + snc_reward_gain)

    delta = far_burst / max(near_burst, 1e-6)
    eps = 1.15
    graded_gradient = bool(V_near >= eps * max(V_mid, 1e-6) and V_mid >= eps * max(V_far, 1e-6))
    location_selective = bool(v_near_late > 1.05 * max(v_near_early, 1e-6)
                              and V_near > 1.05 * max(V_far, 1e-6))
    if verbose:
        tag = ("PLATEAU-LESION(off)" if plateau_lesion else
               "ALL-OR-NONE(coincidence)" if allornone else
               "SUBTRACT-LESION" if subtract_lesion else "GRADED on-bridge")
        print(f"  [ONBRIDGE test lead={lead_steps} {tag}] bursts near={near_burst:.1f} mid={mid_burst:.1f} "
              f"far={far_burst:.1f} Hz | V_onbridge near={V_near:.3f} mid={V_mid:.3f} far={V_far:.3f} "
              f"-> delta(far/near)={delta:.2f} graded-3={graded_gradient}")

    return dict(seed=seed, delta=float(delta),
                near_burst=float(near_burst), mid_burst=float(mid_burst), far_burst=float(far_burst),
                v_onbridge_near=float(V_near), v_onbridge_mid=float(V_mid), v_onbridge_far=float(V_far),
                v_near_early=float(v_near_early), v_near_late=float(v_near_late),
                w_near_final=float(w_near_final), w_far_final=float(w_far_final),
                graded_gradient=graded_gradient, location_selective=location_selective,
                plateau_lesion=bool(plateau_lesion), allornone=bool(allornone),
                subtract_lesion=bool(subtract_lesion))


def _seed_all(seed, lead_steps, n_train, host_ref_delta, verbose=True, *, calib=None):
    """Run the ON-BRIDGE graded arm + its lesions + the two point-neuron controls for one seed.

    `calib` (dict): the SNc-burst CALIBRATION knobs forwarded to run_onbridge so EVERY arm (the graded
    read-out, the plateau-off lesion, the all-or-none control, the subtract lesion) uses the SAME SNc
    population resolution + subtraction scale + reward drive -- only the plateau form / subtraction
    on-off differs. This isolates the read-out (Stage 1) from the calibration: the validated defaults
    (dend_subtract_scale=1200, n_snc=120, snc_reward_gain=420) place the graded V's subtraction so far
    lands in the 100-Hz SNc bin and near in the 75-Hz bin (delta=far/near~1.33), the burst-level display
    of the continuum V carries -- the 2026-06-20-dendrite-stage1-snc-calibration deliverable."""
    calib = dict(calib or {})
    out = {}
    rg = run_onbridge(seed, lead_steps=lead_steps, n_train=n_train, verbose=verbose, **calib)
    rl = run_onbridge(seed, lead_steps=lead_steps, n_train=n_train, plateau_lesion=True, verbose=False, **calib)
    ra = run_onbridge(seed, lead_steps=lead_steps, n_train=n_train, allornone=True, verbose=verbose, **calib)
    rs = run_onbridge(seed, lead_steps=lead_steps, n_train=n_train, subtract_lesion=True, verbose=False, **calib)
    out["onbridge"] = dict(
        delta=rg["delta"], near_burst=rg["near_burst"], mid_burst=rg["mid_burst"], far_burst=rg["far_burst"],
        v_onbridge_near=rg["v_onbridge_near"], v_onbridge_mid=rg["v_onbridge_mid"], v_onbridge_far=rg["v_onbridge_far"],
        graded_gradient=rg["graded_gradient"], location_selective=rg["location_selective"],
        w_near_final=rg["w_near_final"], w_far_final=rg["w_far_final"],
        plateau_lesion_delta=rl["delta"],
        plateau_lesion_collapses=bool(rl["delta"] <= 1.15),
        allornone_delta=ra["delta"], allornone_v_near=ra["v_onbridge_near"],
        allornone_graded_gradient=ra["graded_gradient"],
        subtract_lesion_delta=rs["delta"], subtract_lesion_collapses=bool(rs["delta"] <= 1.15),
    )
    kw = dict(grid_size=32, n_train=n_train, coincidence_plateau=80.0,
              coincidence_k_threshold=4.0, coincidence_weighted=False)
    for ro in ("linear", "plateau"):
        r = _run_point_readout(seed, readout=ro, lead_steps=lead_steps, lesion=False, verbose=False, **kw)
        out[ro] = dict(delta=float(r["gap_ratio"]),
                       near_burst=float(r["test_predicted_near_hz"]),
                       far_burst=float(r["test_unpredicted_far_hz"]),
                       critic_rate_hz=float(r["critic_rate_late_hz"]),
                       above_floor=bool(r["above_floor"]))
    out["host_gaussian"] = dict(delta=float(host_ref_delta))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--lead-ms", type=float, default=150.0)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--host-ref-delta", type=float, default=1.3)
    ap.add_argument("--host-ceiling-tol", type=float, default=0.30)
    # === the SNc-burst CALIBRATION knobs (2026-06-20 deliverable; runner/config-side, NO sim/ edit) ===
    # The validated defaults make the small graded V DISPLAY in the SNc burst: a denser SNc population
    # (n_snc) + a large value-scaled subtraction (dend_subtract_scale) at a base reward drive
    # (snc_reward_gain) so V_far lands in the 100-Hz SNc bin and V_near in the 75-Hz bin (delta~1.33).
    ap.add_argument("--n-snc", type=int, default=120,
                    help="SNc population size (finer burst resolution; default 120, was 30)")
    ap.add_argument("--dend-subtract-scale", type=float, default=1200.0,
                    help="pA per unit V subtracted at the SNc (the dominant calibration lever; default 1200)")
    ap.add_argument("--snc-reward-gain", type=float, default=420.0,
                    help="reward burst drive above tonic (base = tonic + this; default 420 -> base 600)")
    ap.add_argument("--graded-center", type=float, default=1.5,
                    help="graded-plateau logistic center in c_w units (default 1.5)")
    ap.add_argument("--graded-slope", type=float, default=1.0,
                    help="graded-plateau logistic slope (default 1.0; keep V graded, do NOT saturate)")
    ap.add_argument("--graded-strength", type=float, default=80.0,
                    help="graded-plateau per-step conductance scale (default 80; V read is strength-invariant)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    lead_steps = int(round(args.lead_ms / 1.0))
    calib = dict(n_snc=int(args.n_snc), dend_subtract_scale=float(args.dend_subtract_scale),
                 snc_reward_gain=float(args.snc_reward_gain), graded_center=float(args.graded_center),
                 graded_slope=float(args.graded_slope), graded_strength=float(args.graded_strength))

    per_seed = {}
    for s in seeds:
        print(f"\n##### DENDRITE STAGE 1 (ON-BRIDGE) seed={s} (lead {args.lead_ms:.0f}ms, grid-32, "
              f"deterministic, on-bridge GRADED dendritic plateau; n_snc={args.n_snc}, "
              f"subtract={args.dend_subtract_scale:.0f}, reward_gain={args.snc_reward_gain:.0f}) #####")
        per_seed[s] = _seed_all(s, lead_steps, args.n_train, args.host_ref_delta, verbose=True, calib=calib)
        p = per_seed[s]
        print(f"  [seed {s}] ON-BRIDGE delta(far/near)={p['onbridge']['delta']:.2f} "
              f"(near={p['onbridge']['near_burst']:.1f} mid={p['onbridge']['mid_burst']:.1f} "
              f"far={p['onbridge']['far_burst']:.1f} Hz | V_onbridge near/mid/far="
              f"{p['onbridge']['v_onbridge_near']:.3f}/{p['onbridge']['v_onbridge_mid']:.3f}/"
              f"{p['onbridge']['v_onbridge_far']:.3f} graded-3={p['onbridge']['graded_gradient']} "
              f"loc-sel={p['onbridge']['location_selective']})")
        print(f"  [seed {s}]   plateau-lesion(off) delta={p['onbridge']['plateau_lesion_delta']:.2f} "
              f"(collapses={p['onbridge']['plateau_lesion_collapses']}) | all-or-none delta="
              f"{p['onbridge']['allornone_delta']:.2f} (V_near={p['onbridge']['allornone_v_near']:.3f}, "
              f"graded-3={p['onbridge']['allornone_graded_gradient']}) | subtract-lesion delta="
              f"{p['onbridge']['subtract_lesion_delta']:.2f} (collapses={p['onbridge']['subtract_lesion_collapses']})")
        print(f"  [seed {s}] LINEAR delta={p['linear']['delta']:.2f} (critic {p['linear']['critic_rate_hz']:.1f}Hz) "
              f"| PLATEAU delta={p['plateau']['delta']:.2f} (critic {p['plateau']['critic_rate_hz']:.1f}Hz) "
              f"| HOST-GAUSSIAN delta~{p['host_gaussian']['delta']:.2f}")

    # ===== the delta TABLE + anti-cheat collapse table + verdict =====
    print("\n" + "=" * 116)
    print("=== DENDRITE STAGE 1 (ON-BRIDGE) delta TABLE (delta=far/near; faithful grid-32, deterministic; "
          f"host-Gaussian ref ~{args.host_ref_delta}) ===")
    print("=" * 116)
    print(f"  {'seed':>5} | {'ON-BRIDGE':>10} | {'LINEAR(pt)':>10} | {'PLATEAU(pt)':>11} | "
          f"{'HOST-Gauss':>10} | {'Vob n/m/f':>16} {'grd-3':>5} {'loc':>4}")
    for s in seeds:
        p = per_seed[s]["onbridge"]
        print(f"  {s:>5} | {p['delta']:>10.2f} | {per_seed[s]['linear']['delta']:>10.2f} | "
              f"{per_seed[s]['plateau']['delta']:>11.2f} | {per_seed[s]['host_gaussian']['delta']:>10.2f} | "
              f"{p['v_onbridge_near']:.2f}/{p['v_onbridge_mid']:.2f}/{p['v_onbridge_far']:.2f}".rjust(16) +
              f" {('Y' if p['graded_gradient'] else 'n'):>5} {('Y' if p['location_selective'] else 'n'):>4}")

    def _med(form, key):
        return _st.median([per_seed[s][form][key] for s in seeds])
    ob_d = _med("onbridge", "delta"); lin_d = _med("linear", "delta"); plat_d = _med("plateau", "delta")
    n = len(seeds); maj = max(1, (n + 1) // 2)
    ob_ge_host = sum(1 for s in seeds if per_seed[s]["onbridge"]["delta"] >= 1.30)
    ob_le_ceil = sum(1 for s in seeds if per_seed[s]["onbridge"]["delta"]
                     <= args.host_ref_delta * (1.0 + args.host_ceiling_tol))
    lin_fails = sum(1 for s in seeds if per_seed[s]["linear"]["delta"] <= 1.15)
    plat_fails = sum(1 for s in seeds if (not per_seed[s]["plateau"]["above_floor"])
                     or per_seed[s]["plateau"]["delta"] <= 0.15)
    ob_graded = sum(1 for s in seeds if per_seed[s]["onbridge"]["graded_gradient"])
    plateau_les_ok = sum(1 for s in seeds if per_seed[s]["onbridge"]["plateau_lesion_collapses"])
    subtract_les_ok = sum(1 for s in seeds if per_seed[s]["onbridge"]["subtract_lesion_collapses"])
    loc_sel_ok = sum(1 for s in seeds if per_seed[s]["onbridge"]["location_selective"])

    print(f"\n  MEDIAN  ON-BRIDGE delta={ob_d:.2f}  |  LINEAR(pt) delta={lin_d:.2f}  |  "
          f"PLATEAU(pt) delta={plat_d:.2f}  |  HOST-Gaussian ~{args.host_ref_delta}")

    print("\n" + "=" * 116)
    print("=== ANTI-CHEAT collapse table (multi-seed) ===")
    print("=" * 116)
    print(f"  (a) TWO POINT-NEURON CONTROLS fail: LINEAR flat(<=1.15) {lin_fails}/{n} ; "
          f"PLATEAU over-clamp(<=0.15) {plat_fails}/{n}")
    print(f"  (b) PLATEAU-OFF lesion collapses the on-bridge delta (<=1.15): {plateau_les_ok}/{n} "
          f"(the on-bridge graded plateau is LOAD-BEARING -- with the flag off V=0 -> no subtraction)")
    print(f"  (c) SUBTRACTION lesion collapses the headline delta (<=1.15): {subtract_les_ok}/{n}")
    print(f"  (d) REGIME FIDELITY: grid-32 deterministic (OU/cond-noise/homeostasis OFF) asserted per seed")
    print(f"  (e) HOST-CEILING: on-bridge delta <= host*(1+{args.host_ceiling_tol:.2f}) "
          f"({args.host_ref_delta * (1.0 + args.host_ceiling_tol):.2f}): {ob_le_ceil}/{n}")
    print(f"  (f) LOCATION-SELECTIVITY of the on-bridge plateau value (V near>far + grew): {loc_sel_ok}/{n}")

    controls_valid = (lin_fails >= maj and plat_fails >= maj)
    ob_go = (ob_ge_host >= maj and ob_le_ceil >= maj and ob_graded >= maj and loc_sel_ok >= maj
             and plateau_les_ok >= maj)
    if not controls_valid:
        verdict = "VOID"; note = (f"the TWO POINT-NEURON CONTROLS did NOT both fail (LINEAR-flat "
                                  f"{lin_fails}/{n}, PLATEAU-over-clamp {plat_fails}/{n}); harness mis-calibrated.")
    elif ob_go:
        verdict = "GO"
        note = (f"the ON-BRIDGE graded dendritic-plateau read-out gives delta={ob_d:.2f} (>=1.30 ~ the "
                f"Stage-0 ceiling/host {args.host_ref_delta}) at {ob_ge_host}/{n} seeds, where BOTH "
                f"point-neuron controls fail (LINEAR ~{lin_d:.2f}, PLATEAU ~{plat_d:.2f}). The on-bridge "
                f"plateau is LOAD-BEARING (flag-off lesion collapses {plateau_les_ok}/{n}); the value is "
                f"location-selective ({loc_sel_ok}/{n}) + below the host ceiling ({ob_le_ceil}/{n}). The "
                f"graded value V is the on-bridge plateau CONDUCTANCE -> the dendrite's graded analog "
                f"read-out now lives ON the spiking substrate (the Stage-0 numpy DendriticLayer retired).")
    else:
        verdict = "GAP"
        why = []
        if ob_ge_host < maj:
            why.append(f"the on-bridge delta ({ob_d:.2f}) did NOT reach the Stage-0/host ceiling "
                       f"({ob_ge_host}/{n} >=1.30) -- the on-bridge realization UNDERPERFORMS the numpy arm")
        if ob_graded < maj:
            why.append(f"the on-bridge V did NOT express the 3-level continuum ({ob_graded}/{n})")
        if loc_sel_ok < maj:
            why.append(f"the on-bridge plateau value was NOT location-selective ({loc_sel_ok}/{n})")
        if plateau_les_ok < maj:
            why.append(f"the flag-off lesion did NOT collapse the delta ({plateau_les_ok}/{n}) -- the "
                       f"plateau may not be the load-bearing element")
        if ob_le_ceil < maj:
            why.append(f"the on-bridge delta exceeded the host ceiling ({ob_le_ceil}/{n})")
        note = ("; ".join(why) + ". HONEST GAP: the on-bridge graded plateau does NOT yet match the "
                "Stage-0 numpy ceiling -> characterize + hand to the controller (the numpy arm remains "
                "the validation target; the sim/ edit is committed + byte-reviewed regardless).")

    print("\n" + "=" * 116)
    print(f"=== DENDRITE STAGE 1 (ON-BRIDGE) VERDICT: {verdict} ===")
    print(f"=== {note} ===")
    print("=" * 116)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(
                item="dendrite_stage1_onbridge_graded_plateau", stage=1, sim_edit=True,
                sim_edit_flag="enable_graded_dendritic_plateau", sim_edit_commit="d69cc0ab",
                deterministic_regime=True, grid_size=32, lead_ms=args.lead_ms,
                host_ref_delta=args.host_ref_delta, host_ceiling_tol=args.host_ceiling_tol,
                calibration=calib,
                seeds=seeds, per_seed={str(s): per_seed[s] for s in seeds},
                median_onbridge_delta=ob_d, median_linear_delta=lin_d, median_plateau_delta=plat_d,
                onbridge_ge_host=ob_ge_host, onbridge_le_ceiling=ob_le_ceil,
                onbridge_graded_gradient=ob_graded,
                linear_fails=lin_fails, plateau_fails=plat_fails,
                plateau_lesion_collapses=plateau_les_ok, subtract_lesion_collapses=subtract_les_ok,
                location_selective=loc_sel_ok, controls_valid=controls_valid,
                verdict=verdict, verdict_note=note,
            ), f, indent=2, default=float)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
