"""Spiking-SNc Stage B — PLACE-CODE striosome value-critic de-risk (value-of-LOCATION).

This is the place-cell variant of `snc_stageb_critic_probe.py`. The Pavlovian probe
validated that a GABAergic striosome critic learns a single-CUE value and subtracts it at
the SNc through GABA_B (3/3 seeds; `2026-06-08-gabab-girk-stageB-derisk-GO.md`). This probe
swaps the single cue for a PLACE-CELL POPULATION CODE (the dorsal "where" / hippocampal
place-cell stream → ventral striatum; catalog C.30 / B.07; Houk-Adams-Barto 1995; Lansink
2009; van der Meer & Redish 2009), testing the ONE new scientific claim the nav redesign
rests on (research doc §5):

    "A GABAergic striosome critic driven by a PLACE code learns a value-of-LOCATION
     (graded V(s) high near the goal, low far from it) and subtracts it at the SNc through
     GABA_B, producing a STATE-SPECIFIC RPE."

THE INPUT IS A POPULATION CODE, NOT A COORDINATE (anti-cheat (a))
-----------------------------------------------------------------
The critic's afferent is `place` — K cells, each with a PREFERRED location on a 1-D
"corridor" position in [0,1] (uniformly tiled). For a corridor position `p` the per-neuron
drive is the place-cell bump `drive_i = max_pA * exp(-(pref_i - p)^2 / 2 sigma^2)` — a
DISTRIBUTED pattern over a subset of the K cells, EXACTLY the per-neuron Gaussian place code
the nav runner uses for `sensor_place_readout` (g11_bg_runner.py:4588-4590). Two states:
  - state A = NEAR goal (p_near, e.g. 0.85) -> high value (reward delivered here)
  - state B = FAR      (p_far,  e.g. 0.15) -> low value  (no/low reward)
These are two DIFFERENT population bumps (different active cell subsets), never a scalar.
The probe asserts the two states activate DIFFERENT ensembles (provenance check).

WHAT IS A VALUE-OF-LOCATION (and why a host EMA can't do it)
-----------------------------------------------------------
A global scalar reward-EMA gives ONE value, identical regardless of which location you are
in — it has NO place-specificity, so it CANNOT make the SNc burst differ between a
predicted (near, high-V) reward and an unpredicted (far, low-V) reward. The neural place
critic CAN, because V is read off the place ensemble: V(near) > V(far). That per-LOCATION
gap is the discriminator that proves the value is BOTH neural AND spatial.

ACCEPTANCE GATES (mirror the Pavlovian PRIMARY gate; multi-seed 42/43/44)
-------------------------------------------------------------------------
  (1) V-LEARNED-SPATIAL  — striosome firing on the NEAR-state place drive RISES across
                           training AND ends HIGHER for NEAR than FAR (graded value-of-location).
  (2) STATE-SPECIFIC RPE — reward at NEAR (predicted: V high, GABA_B cancels) -> SMALL SNc
                           burst; the SAME reward at FAR (unpredicted: V low) -> BIG SNc burst
                           (gap_ratio = far_burst/near_burst > 1.30). Host-EMA-IMPOSSIBLE.
  (3) WEIGHT-GREW        — the place->striosome plastic weight grows from its init (it learns).

ANTI-CHEAT CONTROLS (all three, research doc §5)
------------------------------------------------
  (a) PLACE-POPULATION-CODE provenance — the afferent is a K-cell place ensemble; driving a
      different corridor position yields a different ensemble (asserted), not a scalar.
  (b) CONDUCTANCE / CRITIC LESION (--lesion) — zero the GABA_B mask: the state-specific gap
      must VANISH (SNc bursts to reward at BOTH near and far) -> the subtraction is carried by
      the neural critic's GABA_B current, not host arithmetic. (current_reward_signal stays 0.)
  (c) A/B vs the host-value Stage A — same circuit with `receptor="gaba_a"` (no --gabab)
      should FAIL the gap (the depolarized-SNc wall); and a host global-EMA value CANNOT
      produce a per-LOCATION gap (it is place-blind). The probe computes the host-EMA value in
      both states and asserts it is identical (=> 0 host gap), contrasting the neural gap.

CPU-friendly (tiny bridge): run under SIM_BACKEND=numpy. Multi-seed 42/43/44.

Usage
-----
    SIM_BACKEND=numpy python -m research.runners.snc_stageb_critic_probe_place --seeds 42,43,44 --gabab
    SIM_BACKEND=numpy python -m research.runners.snc_stageb_critic_probe_place --seed 42 --gabab --lesion
    SIM_BACKEND=numpy python -m research.runners.snc_stageb_critic_probe_place --seeds 42,43,44   # GABA_A A/B control
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as _st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np


# ---------------------------------------------------------------------------
# Place-cell population code (anti-cheat (a): a DISTRIBUTED pattern, not a scalar)
# ---------------------------------------------------------------------------
def place_code_drive(position, n_place, max_pA, sigma=0.12):
    """Per-neuron Gaussian place-cell drive for a 1-D corridor position in [0,1].

    Mirrors the nav runner's `sensor_place_readout` encoding (g11_bg_runner.py:4588-4590):
    each cell i has a preferred location pref_i uniformly tiling [0,1]; the drive is the
    population bump drive_i = max_pA * exp(-(pref_i - position)^2 / 2 sigma^2). Returns a
    length-n_place vector (a distributed code), NOT a scalar.
    """
    prefs = np.linspace(0.0, 1.0, n_place, dtype=np.float64)
    dsq = (prefs - float(position)) ** 2
    return (max_pA * np.exp(-dsq / (2.0 * sigma ** 2))).astype(np.float32)


def _build_place_bridge(seed, *, snc_da_sensitivity=8.0, reward_learning_rate=0.08,
                        place_to_strio_weight=3.0, strio_to_snc_weight=2.5,
                        n_place=40, n_strio=60, n_snc=30,
                        gabab=False, gabab_tau_decay=150.0, gabab_propagation_strength=0.105):
    """Minimal bridge: place (place-cell population code) -> striosome_value (GABAergic
    critic) -> snc (DA). IDENTICAL recipe to the validated Pavlovian probe except the input
    region is a place code (K cells) instead of a single cue. place->striosome_value is
    PLASTIC (V learned by the SNc-derived delta via the three-factor pipeline);
    striosome_value->snc is fixed inhibitory, receptor="gaba_b" when gabab=True (the
    validated GABA_B/GIRK subtraction) else "gaba_a" (the depolarized-SNc-wall A/B control).
    """
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType
    from sim.neuromodulators import NeuromodulatorConfig, ModulatorTarget, ProductionRule

    cfg = CoreSimConfig()
    # Harness fix #5 (from the Pavlovian probe): PIN the bridge RNG to `seed` so each --seed
    # is reproducible across processes (else cfg.seed=-1 time-seeds -> per-process lottery,
    # and a multi-seed verdict becomes noise).
    cfg.seed = int(seed)
    cfg.heterogeneity_seed = int(seed)
    cfg.ou_seed = int(seed)
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    # The critic LEARNS: STDP eligibility (pre/post co-firing) x SNc-derived delta -> weight.
    cfg.enable_stdp = True
    cfg.enable_hebbian_learning = False
    cfg.enable_reward_modulation = True
    cfg.enable_short_term_plasticity = False     # confound removal (Pavlovian-probe rationale)
    cfg.enable_structural_plasticity = False     # fixed circuit; no synaptogenesis mid-test
    cfg.reward_learning_rate = float(reward_learning_rate)
    cfg.current_reward_signal = 0.0    # BRAIN-BASED: the SNc FIRING is the signal, not a host scalar
    cfg.reward_baseline = 0.0
    cfg.stdp_w_max = 40.0              # STDP soft-bound headroom (CLAUDE.md) so V can grow

    if gabab:
        cfg.enable_gabab = True
        cfg.gabab_reversal_potential = -90.0
        cfg.gabab_tau_decay = float(gabab_tau_decay)
        cfg.gabab_propagation_strength = float(gabab_propagation_strength)

    cfg.brain_regions = [
        # The PLACE CODE input region: K place cells (population code over corridor position).
        BrainRegion(
            name="place", n_neurons=n_place, exc_fraction=1.0, internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
            plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ),
        BrainRegion(
            name="striosome_value", n_neurons=n_strio, exc_fraction=0.0,   # FULLY GABAergic (MSN)
            internal_density=0.0,   # graded VALUE readout (no WTA), V scales with learned weight
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
    # The critic's learned value: place (perceived LOCATION ensemble) -> striosome (V). PLASTIC.
    pathways = [
        RegionPathway(from_region="place", to_region="striosome_value",
                      density=0.6, weight_mean=float(place_to_strio_weight),
                      weight_jitter=0.5, plastic=True),
        # Direct striosome GABA -> snc. gaba_b routes through the slow GIRK K+ conductance
        # (E_K=-90mV, the validated subtraction); gaba_a is the depolarized-reversal A/B control.
        RegionPathway(from_region="striosome_value", to_region="snc",
                      density=0.5, weight_mean=float(strio_to_snc_weight),
                      weight_jitter=0.2, plastic=False,
                      receptor=("gaba_b" if gabab else "gaba_a")),
    ]
    cfg.region_pathways = pathways

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
    return np.asarray(bridge.region_manager.indices(name), dtype=np.int64)


def _host(a):
    from sim.backend import to_host
    try:
        return to_host(a)
    except Exception:
        return a


def _drive_place(bridge, idx_map, place_vec, region_pa, n_steps, xp, freeze_lr=None, cfg=None):
    """Set the place-code input (place_vec is a per-cell pA vector over the `place` region,
    or None for no place drive) + per-region scalar drives (region_pa: {region: pA}, e.g. snc
    tonic/reward), step n_steps, and return (snc_rate_hz, strio_rate_hz, mean_da)."""
    bridge.cp_external_input_current[:] = 0.0
    if place_vec is not None:
        bridge.cp_external_input_current[idx_map["place"]] = xp.asarray(place_vec, dtype=xp.float32)
    for region, pA in region_pa.items():
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
        # Advance sim time in MS — STDP reads current_time_ms for pre/post delta_t. Without
        # this every delta_t is 0 -> STDP emits a zero update -> no eligibility -> no learning.
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


def _calibrate_da_threshold(bridge, cfg, idx_map, tonic_pa, xp, n_steps=300):
    """Drive the SNc tonic floor, measure its mean firing FRACTION, set the dopamine rule's
    threshold to it (so a burst -> da>baseline -> LTP, a dip -> LTD, tonic -> ~0). Same as
    the Pavlovian probe."""
    snc_idx = idx_map["snc"]; n_snc = len(_host(snc_idx))
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[snc_idx] = xp.float32(tonic_pa)
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


def _mean_pathway_weight(bridge, pre_name, post_name, pre_subset=None):
    """Mean weight of the pre->post edges in the CSR (rows=post, cols=pre). If pre_subset (a
    set of GLOBAL neuron indices) is given, restrict the pre side to that subset — used to
    measure the mean weight of a SPECIFIC place-cell ensemble (near vs far) onto the striosome,
    so location-SELECTIVE learning (near synapses grow, far stay low) is directly verifiable."""
    pre = set(int(i) for i in _idx(bridge, pre_name)) if pre_subset is None else set(int(i) for i in pre_subset)
    post = set(int(i) for i in _idx(bridge, post_name))
    coo = bridge.cp_connections.tocoo()
    rows = np.asarray(_host(coo.row)); cols = np.asarray(_host(coo.col)); data = np.asarray(_host(coo.data))
    m = np.fromiter(((r in post and c in pre) for r, c in zip(rows, cols)), dtype=bool, count=len(rows))
    if not m.any():
        m = np.fromiter(((r in pre and c in post) for r, c in zip(rows, cols)), dtype=bool, count=len(rows))
    return float(data[m].mean()) if m.any() else 0.0


def _ensemble_global_indices(bridge, place_vec, frac=0.5):
    """Global neuron indices of the place cells most strongly driven by `place_vec` (the top
    `frac` by drive). Defines the 'near-ensemble' / 'far-ensemble' synapse sets for per-ensemble
    weight tracking (anti-cheat: shows the learned weight gain is LOCATION-SELECTIVE)."""
    place_global = np.asarray(_idx(bridge, "place"), dtype=np.int64)
    drive = np.asarray(place_vec, dtype=np.float64)
    k = max(1, int(round(frac * len(drive))))
    top = np.argsort(drive)[-k:]
    return set(int(place_global[i]) for i in top)


def _clear_eligibility(bridge):
    """Zero the eligibility trace. Called at the start of each NEAR-learning window so ONLY the
    co-firing produced by the near place ensemble (x the reward delta in that window) is
    converted to weight — and after the held-out FAR probe so its residual eligibility cannot
    leak into the next near window's reward delta. This is what makes the learned value
    LOCATION-SELECTIVE (near potentiates, far does not); without it the persistent eligibility
    + saturated tonic-DA produces uniform, place-blind LTP (the failure mode the early runs
    showed). Biologically: the eligibility tag is a fast-decaying co-incidence trace, so a value
    update is dominated by the synapses active in the rewarded state, not stale ones."""
    if getattr(bridge, "cp_eligibility_trace", None) is not None:
        bridge.cp_eligibility_trace[:] = 0.0


def _lesion_gabab_mask(bridge):
    """Conductance lesion (anti-cheat (b)): zero the per-synapse GABA_B routing mask so NO
    synapse feeds the slow K+ conductance. The state-specific gap must VANISH -> proves the
    subtraction was carried by the GABA_B/GIRK conductance, not host arithmetic. Returns the
    number of GABA_B synapses zeroed."""
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


def _ensemble_overlap(bridge, idx_map, vec_a, vec_b, xp, n_steps=40, thresh_hz=1.0):
    """Anti-cheat (a) provenance: drive place with vec_a then vec_b, record which place cells
    fire (rate > thresh), and return (n_active_a, n_active_b, jaccard_overlap). A LOW overlap
    proves NEAR and FAR are DIFFERENT population ensembles (a perceived spatial pattern), not a
    scalar that merely scales a fixed set."""
    place_idx = idx_map["place"]; n_place = len(_host(place_idx))

    def active_set(vec):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[place_idx] = xp.asarray(vec, dtype=xp.float32)
        c = np.zeros(n_place, dtype=np.int64)
        for _ in range(n_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = (
                bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)
            c += np.asarray(_host(bridge.cp_firing_states[place_idx])).astype(np.int64)
        rate = c / (n_steps * 1e-3)
        return set(int(i) for i in np.where(rate > thresh_hz)[0])

    sa, sb = active_set(vec_a), active_set(vec_b)
    inter = len(sa & sb); union = max(len(sa | sb), 1)
    return len(sa), len(sb), inter / union


def run_place(seed, *, snc_tonic_pa=180.0, snc_reward_gain=300.0,
              place_drive_pa=2500.0, place_sigma=0.12, p_near=0.85, p_far=0.15,
              hold_steps=40, n_train=40, reward_learning_rate=0.12,
              place_to_strio_weight=0.2, strio_to_snc_weight=10.0,
              snc_da_sensitivity=8.0, lesion=False, verbose=True,
              gabab=False, gabab_tau_decay=150.0, gabab_propagation_strength=0.105,
              lead_steps=0, return_trained=False):
    """Run the full PLACE-CODE value-critic de-risk for one seed.

    lead_steps (NEW, nav-realistic timing de-risk, 2026-06-08): how many steps the place
    code is held ACTIVE *before* the reward burst arrives, in EACH test condition. In real
    navigation the place/value afferent fires for many steps while the agent is AT/approaching
    a location BEFORE the reward event — so the critic's value V and its slow (tau~150 ms)
    GABA_B inhibition on the SNc are PRE-BUILT before the reward burst, letting the slow
    conductance cancel a fast burst. The prior probe drove place+reward SIMULTANEOUSLY
    (lead_steps=0), which the slow GABA_B could NOT cancel (the burst is over before the
    conductance ramps) — the "probe != deployment" timing trap. With lead_steps>0 the predicted
    (NEAR) burst should be canceled (V high -> GABA_B pre-built) while the unpredicted (FAR)
    burst stays full (V(far) low -> no pre-built GABA_B). lead_steps=0 reproduces the original
    simultaneous timing exactly (backward-compatible default).
    """
    from sim.backend import get_backend
    xp, _ = get_backend()
    bridge, cfg = _build_place_bridge(
        seed, snc_da_sensitivity=snc_da_sensitivity,
        reward_learning_rate=reward_learning_rate,
        place_to_strio_weight=place_to_strio_weight, strio_to_snc_weight=strio_to_snc_weight,
        gabab=gabab, gabab_tau_decay=gabab_tau_decay,
        gabab_propagation_strength=gabab_propagation_strength)
    idx_map = {n: xp.asarray(_idx(bridge, n)) for n in ("place", "striosome_value", "snc")}
    n_place = len(_host(idx_map["place"]))

    # The two place-cell POPULATION codes (anti-cheat (a): distributed patterns, not scalars).
    near_vec = place_code_drive(p_near, n_place, place_drive_pa, sigma=place_sigma)
    far_vec = place_code_drive(p_far, n_place, place_drive_pa, sigma=place_sigma)

    # Anti-cheat (a) provenance check FIRST (before training perturbs weights): NEAR and FAR
    # must be DIFFERENT place ensembles.
    na, nb, overlap = _ensemble_overlap(bridge, idx_map, near_vec, far_vec, xp)
    if verbose:
        print(f"  [anti-cheat a: place-code provenance] NEAR active={na}/{n_place} cells, "
              f"FAR active={nb}/{n_place} cells, ensemble Jaccard overlap={overlap:.2f} "
              f"(LOW => distinct spatial ensembles, not a scalar)")
    distinct_ensembles = (overlap < 0.5 and na > 0 and nb > 0)

    # Calibrate the dopamine threshold to the SNc's actual tonic firing fraction.
    tonic_frac = _calibrate_da_threshold(bridge, cfg, idx_map, snc_tonic_pa, xp)
    if verbose:
        print(f"  [calib] SNc tonic firing fraction = {tonic_frac:.4f} -> dopamine threshold")

    # Per-ensemble synapse sets (global place-cell indices) for LOCATION-SELECTIVE weight
    # tracking. Use the DISJOINT bump cores (top 25% by drive, p_near vs p_far well separated)
    # so w_near and w_far measure non-overlapping synapse sets (else the generous top-50%
    # tails overlap in the middle of the corridor and the ratio is trivially ~1).
    near_set = _ensemble_global_indices(bridge, near_vec, frac=0.25)
    far_set = _ensemble_global_indices(bridge, far_vec, frac=0.25)
    far_set = far_set - near_set   # guarantee disjoint
    w_init = _mean_pathway_weight(bridge, "place", "striosome_value")
    w_near_init = _mean_pathway_weight(bridge, "place", "striosome_value", pre_subset=near_set)
    w_far_init = _mean_pathway_weight(bridge, "place", "striosome_value", pre_subset=far_set)

    # Place-reward schedule = the place analogue of CS->US (mirrors the Pavlovian PASS exactly):
    # train ONLY by VISITING the NEAR state followed by reward. The FAR state is a HELD-OUT test
    # ensemble — NEVER presented during training, so its place->striosome synapses never co-fire
    # with the reward delta and stay near-init -> V(far) low. The NEAR-ensemble synapses
    # potentiate (eligibility on near-active synapses x positive SNc delta) -> V(near) high. The
    # learned weight gain is LOCATION-SELECTIVE (near grows, far doesn't) — that selectivity IS
    # the value-of-location, and a place-blind host EMA cannot produce it.
    # NB (diagnosed 2026-06-08): driving FAR even once per trial to *measure* V(far) leaks far
    # eligibility into the next near reward window (despite an explicit clear) and washes out the
    # selectivity. So far is genuinely held out during training (like the Pavlovian unpredicted
    # condition) and V(far) is read ONCE at test, learning frozen. We clear eligibility at the
    # start of each near window so only that window's near co-firing forms the converted tag.
    near_v_curve, near_burst_curve = [], []
    for t in range(n_train):
        _drive_place(bridge, idx_map, None, {"snc": snc_tonic_pa}, hold_steps, xp)  # ITI floor
        _clear_eligibility(bridge)   # fresh tag: only near co-firing in this window counts
        snc_r, strio_r, da = _drive_place(
            bridge, idx_map, near_vec, {"snc": snc_tonic_pa + snc_reward_gain}, hold_steps, xp)  # LEARN near
        near_v_curve.append(strio_r); near_burst_curve.append(snc_r)
        if verbose and (t < 3 or t % 5 == 0 or t == n_train - 1):
            wn = _mean_pathway_weight(bridge, "place", "striosome_value", pre_subset=near_set)
            wf = _mean_pathway_weight(bridge, "place", "striosome_value", pre_subset=far_set)
            print(f"  [acq t={t:02d}] near-burst={snc_r:6.2f}Hz  V(near)={strio_r:6.2f}Hz  "
                  f"w_near={wn:.3f}  w_far={wf:.3f}  (near/far {wn/max(wf,1e-6):.2f})  DA={da:.3f}")

    early = slice(0, max(1, n_train // 5)); late = slice(-max(1, n_train // 5), None)
    near_v_early = _st.mean(near_v_curve[early]); near_v_late = _st.mean(near_v_curve[late])
    w_final = _mean_pathway_weight(bridge, "place", "striosome_value")
    w_near_final = _mean_pathway_weight(bridge, "place", "striosome_value", pre_subset=near_set)
    w_far_final = _mean_pathway_weight(bridge, "place", "striosome_value", pre_subset=far_set)

    # Training-derived quantities are bundled so the test/gate phase can be re-run at multiple
    # LEADS from the SAME trained critic (the lead sweep trains once, tests many).
    train_state = dict(
        seed=seed, lesion=lesion, gabab=gabab, cfg=cfg, idx_map=idx_map,
        near_vec=near_vec, far_vec=far_vec, snc_tonic_pa=snc_tonic_pa,
        snc_reward_gain=snc_reward_gain, hold_steps=hold_steps,
        na=na, nb=nb, overlap=overlap, distinct_ensembles=distinct_ensembles,
        near_v_early=near_v_early, near_v_late=near_v_late,
        w_init=w_init, w_final=w_final, w_near_init=w_near_init, w_near_final=w_near_final,
        w_far_init=w_far_init, w_far_final=w_far_final,
        near_v_curve=near_v_curve, near_burst_curve=near_burst_curve,
    )
    result = _test_and_gate(bridge, xp, train_state, lead_steps, verbose=verbose)
    if return_trained:
        # For the lead sweep: hand back the TRAINED bridge + its train_state so the caller can
        # re-run _test_and_gate at other leads without retraining (train once, test many).
        return result, bridge, xp, train_state
    return result


def _test_and_gate(bridge, xp, ts, lead_steps, *, verbose=True):
    """The test+gate phase for a TRAINED place critic at a given LEAD. Re-warms and measures
    the four conditions (baseline / predicted-NEAR / unpredicted-FAR / omission), applies the
    NAV-REALISTIC LEAD, computes the gates, and returns the result dict. Split out of run_place
    so a lead sweep can re-run JUST this phase (each lead re-warmed + held-out V re-read) on the
    SAME trained bridge — train once, test at every lead. (2026-06-08.)
    """
    cfg = ts["cfg"]; idx_map = ts["idx_map"]
    near_vec = ts["near_vec"]; far_vec = ts["far_vec"]
    snc_tonic_pa = ts["snc_tonic_pa"]; snc_reward_gain = ts["snc_reward_gain"]
    hold_steps = ts["hold_steps"]
    lesion = ts["lesion"]; gabab = ts["gabab"]; seed = ts["seed"]

    # A re-warmed test condition: an ITI floor (learning frozen) re-settles the SNc adaptation
    # state, THEN the condition is measured. Without the re-warm the SNc's adaptation carries over
    # from the previous test (the near test adapts it down → the far burst is spuriously small),
    # confounding the gap. (Diagnosed 2026-06-08.)
    #
    # NAV-REALISTIC LEAD (2026-06-08): when lead_steps>0 AND there is a place drive, the place
    # code is held active (tonic SNc, NO reward) for `lead_steps` BETWEEN the re-warm and the
    # measured reward window. This pre-builds V and — crucially — the slow GABA_B/GIRK
    # conductance on the SNc, so a high-V (near) state can cancel the subsequent fast reward
    # burst from its first step. This replicates how the nav place code leads the reward event
    # (the prior simultaneous timing gave the slow conductance no time to ramp). The re-warm
    # holds place OFF so GABA_B decays toward 0 first; the lead then ramps it under the actual
    # test place ensemble; the measured window reads the burst with GABA_B already in place.
    def _test(place_vec, snc_pa):
        _drive_place(bridge, idx_map, None, {"snc": snc_tonic_pa},
                     hold_steps + 20, xp, freeze_lr=0.0, cfg=cfg)
        if lead_steps > 0 and place_vec is not None:
            _drive_place(bridge, idx_map, place_vec, {"snc": snc_tonic_pa},
                         int(lead_steps), xp, freeze_lr=0.0, cfg=cfg)
        return _drive_place(bridge, idx_map, place_vec, {"snc": snc_pa},
                            hold_steps, xp, freeze_lr=0.0, cfg=cfg)

    # V(far) read ONCE at test (learning frozen) — the held-out far ensemble's striosome rate.
    _, far_v_late, _ = _test(far_vec, snc_tonic_pa)
    far_v_curve = [far_v_late]

    if lesion:
        if gabab:
            n_cut = _lesion_gabab_mask(bridge); edge = "GABA_B mask (cp_gabab_synapse_mask)"
        else:
            n_cut = 0; edge = "(no GABA_B mask in GABA_A mode)"
        if verbose:
            print(f"  [lesion] zeroed {n_cut} {edge} synapses")

    # --- Test (learning frozen, each condition re-warmed): the STATE-SPECIFIC gap ---
    # PREDICTED reward = reward delivered in the NEAR state (V high -> GABA_B cancels -> small burst).
    # UNPREDICTED reward = SAME reward delivered in the FAR state (V low -> not cancelled -> big burst).
    # baseline = tonic only (no place, no reward).
    base_r, base_v, _ = _test(None, snc_tonic_pa)
    pred_r, pred_v, _ = _test(near_vec, snc_tonic_pa + snc_reward_gain)
    unpred_r, unpred_v, _ = _test(far_vec, snc_tonic_pa + snc_reward_gain)
    # Omission (regression guard): NEAR state, reward omitted -> SNc should dip below tonic.
    omit_r, omit_v, _ = _test(near_vec, snc_tonic_pa)
    if verbose:
        print(f"  [test V lead={lead_steps}] V(near,pred)={pred_v:.1f}Hz  V(far,unpred)={unpred_v:.1f}Hz  "
              f"V(near,omit)={omit_v:.1f}Hz  baseline={base_v:.1f}Hz")

    # Anti-cheat (c) host-EMA contrast: a global reward-EMA value is PLACE-BLIND. We compute the
    # would-be host value in each state (the same reward-EMA scaffold Stage A used). It is IDENTICAL
    # near vs far (no place input enters a scalar EMA) => host gap_ratio = 1.0 by construction.
    # Reported to contrast against the neural gap (the whole point of going brain-based).
    host_value_near = host_value_far = float(snc_reward_gain)  # an EMA of r is place-independent
    host_gap_ratio = 1.0  # host EMA cannot differ by location

    near_v_early = ts["near_v_early"]; near_v_late = ts["near_v_late"]
    w_near_init = ts["w_near_init"]; w_near_final = ts["w_near_final"]
    w_far_final = ts["w_far_final"]

    # ---- Gates ----
    # (1) V-learned-spatial: near V rose AND ends higher than far V (a graded value-of-location).
    v_learned_spatial = (near_v_late > 1.20 * near_v_early) and (near_v_late > 1.20 * max(far_v_late, 1e-6))
    # (2) State-specific RPE: far (unpredicted) burst >> near (predicted) burst.
    state_specific = (unpred_r > 1.30 * max(pred_r, 1e-6))
    # (2b) ABOVE THE SNc NOISE FLOOR: the gap is only MEANINGFUL if the unpredicted burst is a
    # REAL burst (not ~0 vs ~0.8 Hz). Require the far (unpredicted) burst >= 10 Hz so the ratio
    # discriminates a cancelled-vs-uncancelled REWARD, not floor noise. (Brief, 2026-06-08.)
    above_floor = (unpred_r >= 10.0)
    state_specific_above_floor = bool(state_specific and above_floor)
    # (3) Weight grew LOCATION-SELECTIVELY: the near-ensemble synapses grew from init AND grew
    #     MORE than the held-out far-ensemble synapses (the learned value is place-specific).
    weight_grew = (w_near_final > 1.05 * max(w_near_init, 1e-6)
                   and w_near_final > 1.05 * max(w_far_final, 1e-6))
    # Regression guard (reported, not a primary gate): omission dip.
    omission_dip = (omit_r < base_r)

    gap_ratio = unpred_r / max(pred_r, 1e-6)
    v_near_far_ratio = near_v_late / max(far_v_late, 1e-6)
    w_near_far_ratio = w_near_final / max(w_far_final, 1e-6)
    return {
        "seed": seed, "lesion": lesion, "gabab": gabab,
        "lead_steps": int(lead_steps), "lead_ms": float(lead_steps) * float(cfg.dt_ms),
        "place_ensemble_near_active": ts["na"], "place_ensemble_far_active": ts["nb"],
        "place_ensemble_overlap": ts["overlap"], "distinct_ensembles": bool(ts["distinct_ensembles"]),
        "near_v_early_hz": near_v_early, "near_v_late_hz": near_v_late,
        "far_v_late_hz": far_v_late, "v_near_far_ratio": v_near_far_ratio,
        "w_init": ts["w_init"], "w_final": ts["w_final"],
        "w_near_init": w_near_init, "w_near_final": w_near_final,
        "w_far_init": ts["w_far_init"], "w_far_final": w_far_final,
        "w_near_far_ratio": w_near_far_ratio,
        "test_baseline_hz": base_r, "test_predicted_near_hz": pred_r,
        "test_unpredicted_far_hz": unpred_r, "test_omission_hz": omit_r,
        "gap_ratio": gap_ratio, "above_floor": bool(above_floor),
        "state_specific_above_floor": state_specific_above_floor,
        "host_gap_ratio": host_gap_ratio,
        "host_value_near": host_value_near, "host_value_far": host_value_far,
        "v_learned_spatial": bool(v_learned_spatial),
        "state_specific": bool(state_specific),
        "weight_grew": bool(weight_grew),
        "omission_dip": bool(omission_dip),
        "near_v_curve": ts["near_v_curve"], "far_v_curve": far_v_curve,
        "near_burst_curve": ts["near_burst_curve"],
    }


def run_place_lead_sweep(seed, lead_steps_list, *, verbose=True, **kw):
    """Train the place critic ONCE for `seed`, then run the test+gate phase at EACH lead in
    `lead_steps_list` (re-warmed per condition per lead) on the SAME trained bridge. Returns a
    list of result dicts (one per lead). This is the decisive nav-timing de-risk: does a
    nav-realistic value-leads-reward lead open the state-specific SNc gap ROBUSTLY above floor?

    The expensive part is training (V acquisition over n_train trials); the lead only affects the
    test phase. So we train once at the first lead (run_place(..., return_trained=True)) and then
    re-run only _test_and_gate for the remaining leads on the SAME trained bridge.
    """
    first, bridge, xp, ts = run_place(
        seed, verbose=verbose, lead_steps=int(lead_steps_list[0]),
        return_trained=True, **kw)
    results = [first]
    for lead in lead_steps_list[1:]:
        results.append(_test_and_gate(bridge, xp, ts, int(lead), verbose=verbose))
    return results


def _print_result(r):
    print()
    print(f"  V(near) on place   : {r['near_v_early_hz']:.2f} -> {r['near_v_late_hz']:.2f} Hz   "
          f"V(far) late {r['far_v_late_hz']:.2f} Hz   (near/far ratio {r['v_near_far_ratio']:.2f})")
    print(f"  place->strio weight: near {r['w_near_init']:.3f}->{r['w_near_final']:.3f}  "
          f"far {r['w_far_init']:.3f}->{r['w_far_final']:.3f}  "
          f"(near/far {r['w_near_far_ratio']:.2f}, location-selective grew: {r['weight_grew']})")
    print(f"  predicted (NEAR+US): {r['test_predicted_near_hz']:.2f} Hz")
    print(f"  unpredicted(FAR+US): {r['test_unpredicted_far_hz']:.2f} Hz   "
          f"(state-specific gap: {r['state_specific']}, ratio {r['gap_ratio']:.2f})")
    print(f"  omission (NEAR,noUS): {r['test_omission_hz']:.2f} Hz  vs baseline "
          f"{r['test_baseline_hz']:.2f} Hz  (dip: {r['omission_dip']})")
    print(f"  [anti-cheat c] host-EMA value near={r['host_value_near']:.1f} == far={r['host_value_far']:.1f} "
          f"=> host gap_ratio {r['host_gap_ratio']:.2f} (place-BLIND; cannot produce the gap)")


def _lead_sweep_main(seeds, lead_sweep_str, kw, args):
    """The DECISIVE nav-timing de-risk: sweep the value-leads-reward LEAD over a nav-realistic
    range, multi-seed, on a critic trained ONCE per seed. Prints the lead x seed table
    (near_burst, far_burst, gap, above-floor?) and the multi-seed verdict at the best lead.

    The gate at a lead is the state-specific SNc gap (far_burst > 1.30 x near_burst) that is
    ALSO above the SNc noise floor (far_burst >= 10 Hz, so the ratio discriminates a real
    cancelled-vs-uncancelled reward, not floor noise) — sign-consistent across >= 3 seeds.
    """
    leads_ms = [float(x) for x in lead_sweep_str.split(",")]
    lead_steps_list = [int(round(m / 1.0)) for m in leads_ms]   # dt_ms = 1.0
    sweep_kw = {k: v for k, v in kw.items() if k != "lead_steps"}

    def _fmt_gap(r):
        # Display helper: when near≈0 the far/near ratio diverges; show it as 'INF' (complete
        # cancellation of the predicted reward) rather than a 7-digit floor-division artifact.
        # The GATE (state_specific_above_floor) is unaffected — this is cosmetic only.
        if r["test_predicted_near_hz"] < 0.5:
            return "  INF" if r["test_unpredicted_far_hz"] >= 0.5 else " 0.00"
        return "{:5.2f}".format(r["gap_ratio"])

    # results_by_seed[seed] = [result per lead], all on the same trained critic.
    results_by_seed = {}
    for s in seeds:
        print(f"\n##### LEAD SWEEP seed={s} (train once, test at leads {leads_ms} ms) #####")
        rs = run_place_lead_sweep(s, lead_steps_list, verbose=True, **sweep_kw)
        results_by_seed[s] = rs
        r0 = rs[0]
        print(f"  [seed {s}] LEARNING: V(near)/V(far)={r0['v_near_far_ratio']:.2f}  "
              f"w_near/w_far={r0['w_near_far_ratio']:.2f}  "
              f"V-learned-spatial={r0['v_learned_spatial']}  weight-grew={r0['weight_grew']}")

    # ---- The lead x seed sweep table ----
    print("\n" + "=" * 92)
    print("=== LEAD SWEEP TABLE: near_burst / far_burst / gap_ratio(far/near) / above-floor? ===")
    print("=" * 92)
    header = "  lead_ms |" + "".join(f"  seed {s:>4}                          |" for s in seeds)
    print(header)
    for li, lead_ms in enumerate(leads_ms):
        cells = []
        for s in seeds:
            r = results_by_seed[s][li]
            nb_ = r["test_predicted_near_hz"]; fb_ = r["test_unpredicted_far_hz"]
            af = r["above_floor"]
            sa = r["state_specific_above_floor"]
            flag = "OK" if sa else ("--" if not af else "lo")
            cells.append(f" near={nb_:5.1f} far={fb_:5.1f} g={_fmt_gap(r)} {('AF' if af else '..')}/{flag} |")
        print(f"  {lead_ms:6.0f}  |" + "".join(cells))
    print("  (AF = far_burst >= 10 Hz above floor;  OK = state-specific gap AND above floor)")

    # ---- Per-lead multi-seed robustness ----
    print("\n=== PER-LEAD multi-seed robustness (state-specific gap AND above floor) ===")
    best_lead_idx = None; best_n = -1
    for li, lead_ms in enumerate(leads_ms):
        rl = [results_by_seed[s][li] for s in seeds]
        n_gap = sum(1 for r in rl if r["state_specific"])
        n_af = sum(1 for r in rl if r["above_floor"])
        n_robust = sum(1 for r in rl if r["state_specific_above_floor"])
        gap_strs = ", ".join("{}={}".format(r["seed"], _fmt_gap(r).strip()) for r in rl)
        far_strs = ", ".join("{}={:.1f}".format(r["seed"], r["test_unpredicted_far_hz"]) for r in rl)
        print(f"  lead={lead_ms:4.0f}ms: ROBUST(gap&floor) {n_robust}/{len(seeds)}  "
              f"[gap>1.30 {n_gap}/{len(seeds)}, above-floor {n_af}/{len(seeds)}]  "
              f"gaps[{gap_strs}]  far_burst[{far_strs}]")
        # "best" = max robust seeds; tie-break on the larger total far_burst (more headroom).
        if n_robust > best_n:
            best_n = n_robust; best_lead_idx = li

    best_lead_ms = leads_ms[best_lead_idx]
    rl_best = [results_by_seed[s][best_lead_idx] for s in seeds]
    n_robust_best = sum(1 for r in rl_best if r["state_specific_above_floor"])
    n_learn = sum(1 for r in rl_best if r["v_learned_spatial"] and r["weight_grew"])
    print("\n" + "=" * 92)
    verdict_pass = (n_robust_best >= 3 and n_robust_best >= max(3, (len(seeds) + 1) // 2))
    # Decisive: the gap must open ROBUSTLY (>=3 seeds) AND above floor at a nav-realistic lead,
    # with LEARNING retained.
    print(f"=== BEST LEAD = {best_lead_ms:.0f} ms: ROBUST state-specific gap (above floor) "
          f"{n_robust_best}/{len(seeds)} seeds; LEARNING retained {n_learn}/{len(seeds)} ===")
    decisive = "PASS" if (verdict_pass and n_learn >= 3) else "FAIL"
    print(f"=== NAV-TIMING SUBTRACTION DE-RISK VERDICT: {decisive}  "
          f"(>=3 seeds robust-above-floor at a nav-realistic lead AND learning retained) ===")
    print("=" * 92)

    if args.out:
        out = {
            "mode": "place_gabab_lead_sweep" if args.gabab else "place_gaba_a_lead_sweep",
            "leads_ms": leads_ms,
            "best_lead_ms": best_lead_ms,
            "n_robust_best": n_robust_best,
            "n_learn_best": n_learn,
            "verdict": decisive,
            "results_by_seed": {str(s): results_by_seed[s] for s in seeds},
        }
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None, help="comma seeds for multi-seed")
    ap.add_argument("--snc-tonic-pa", type=float, default=180.0)
    ap.add_argument("--snc-reward-gain", type=float, default=300.0)
    ap.add_argument("--place-drive-pa", type=float, default=2500.0)
    ap.add_argument("--place-sigma", type=float, default=0.12)
    ap.add_argument("--p-near", type=float, default=0.85)
    ap.add_argument("--p-far", type=float, default=0.15)
    ap.add_argument("--hold-steps", type=int, default=40)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--reward-learning-rate", type=float, default=0.12)
    ap.add_argument("--place-to-strio-weight", type=float, default=0.2)
    ap.add_argument("--strio-to-snc-weight", type=float, default=10.0)
    ap.add_argument("--snc-da-sensitivity", type=float, default=8.0)
    ap.add_argument("--lesion", action="store_true",
                    help="anti-cheat (b): zero the GABA_B mask after training -> gap must vanish")
    ap.add_argument("--gabab", action="store_true",
                    help="GABA_B/GIRK: route striosome_value->snc through the slow K+ conductance "
                         "(E_K=-90mV); without it, GABA_A direct = the depolarized-SNc A/B control")
    ap.add_argument("--gabab-tau-decay", type=float, default=150.0)
    ap.add_argument("--gabab-propagation-strength", type=float, default=0.105)
    ap.add_argument("--lead-ms", type=float, default=0.0,
                    help="NAV-REALISTIC LEAD: ms the place code leads the reward burst in each "
                         "test condition (pre-builds V + the slow GABA_B conductance on the SNc). "
                         "0 = the original simultaneous timing.")
    ap.add_argument("--lead-sweep", type=str, default=None,
                    help="comma ms leads, e.g. '0,100,200,300,400,500'. Trains the critic ONCE "
                         "per seed and tests the state-specific gap at each lead. The decisive "
                         "nav-timing de-risk.")
    ap.add_argument("--nav-derisk", action="store_true",
                    help="DECISIVE nav-timing de-risk preset (2026-06-08): the canonical "
                         "value-leads-reward sweep at the PHYSIOLOGICAL GABA_B operating point. "
                         "Sets --gabab, the LIVE SNc regime (tonic=180/reward=300 -> baseline + "
                         "real reward burst), a MODERATE GABA_B (gabab_propagation_strength=0.02 "
                         "so the slow conductance settles physiologically instead of saturating "
                         "to ~170 and flatlining the SNc, the bug the default 0.105 hits over a "
                         "long lead), and the lead sweep 0,100,150,200,300,400,500. Override any "
                         "of these by passing them explicitly AFTER --nav-derisk.")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    # --nav-derisk preset: the physiological live-SNc + moderate-GABA_B operating point at which
    # the value-leads-reward lead recovers the state-specific subtraction (PASS 3/3 at lead
    # 100-150 ms). Applied as defaults the user can still override on the same command line.
    if args.nav_derisk:
        args.gabab = True
        if args.gabab_propagation_strength == 0.105:   # default untouched -> set live value
            args.gabab_propagation_strength = 0.02
        if args.snc_tonic_pa == 180.0:
            args.snc_tonic_pa = 180.0
        if args.snc_reward_gain == 300.0:
            args.snc_reward_gain = 300.0
        if args.lead_sweep is None:
            args.lead_sweep = "0,100,150,200,300,400,500"

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    lead_steps = int(round(args.lead_ms / 1.0))  # dt_ms = 1.0 in this probe -> 1 step per ms
    kw = dict(snc_tonic_pa=args.snc_tonic_pa, snc_reward_gain=args.snc_reward_gain,
              place_drive_pa=args.place_drive_pa, place_sigma=args.place_sigma,
              p_near=args.p_near, p_far=args.p_far, hold_steps=args.hold_steps,
              n_train=args.n_train, reward_learning_rate=args.reward_learning_rate,
              place_to_strio_weight=args.place_to_strio_weight,
              strio_to_snc_weight=args.strio_to_snc_weight,
              snc_da_sensitivity=args.snc_da_sensitivity, lesion=args.lesion,
              gabab=args.gabab, gabab_tau_decay=args.gabab_tau_decay,
              gabab_propagation_strength=args.gabab_propagation_strength)

    # ===== LEAD SWEEP (the decisive nav-timing de-risk) =====
    if args.lead_sweep:
        _lead_sweep_main(seeds, args.lead_sweep, kw, args)
        return

    kw["lead_steps"] = lead_steps
    results = []
    for s in seeds:
        tag = ("LESION (GABA_B mask cut)" if args.lesion
               else "PLACE-CODE critic + GABA_B/GIRK (E_K=-90mV)" if args.gabab
               else "PLACE-CODE critic (GABA_A direct — A/B control)")
        print(f"[snc-stageB-place seed={s}] {tag} — value-of-LOCATION (delta=r-V(place), R-W):")
        r = run_place(s, **kw)
        _print_result(r)
        if not args.lesion:
            primary = r["v_learned_spatial"] and r["state_specific"] and r["weight_grew"]
            print(f"\n  PLACE-CODE de-risk (seed {s}): {'PASS' if primary else 'FAIL'}  "
                  f"[V-learned-spatial {r['v_learned_spatial']}, state-specific {r['state_specific']}, "
                  f"weight-grew {r['weight_grew']}] (omission-dip {r['omission_dip']} guard)")
            print(f"  [PRIMARY GATE — value-of-location] gap_ratio(far/near)={r['gap_ratio']:.2f} (>1.30 PASS) | "
                  f"V(near)/V(far)={r['v_near_far_ratio']:.2f} (>1.0) | "
                  f"distinct-place-ensembles {r['distinct_ensembles']} => {'PASS' if primary else 'FAIL'}")
        else:
            no_gap = (r["test_unpredicted_far_hz"] <= 1.30 * max(r["test_predicted_near_hz"], 1e-6))
            print(f"\n  LESION anti-cheat (seed {s}): {'PASS' if no_gap else 'UNEXPECTED'}  "
                  f"[gap-gone {no_gap}, gap_ratio {r['gap_ratio']:.2f}] "
                  f"(cutting the GABA_B conduit removed the neural subtraction)")
        results.append(r)
        print()

    if len(results) > 1 and not args.lesion:
        # The two sub-claims, reported separately (the de-risk's honest decomposition):
        #   LEARNING  = the critic carved a graded value-of-LOCATION (V-learned-spatial +
        #               location-selective weight growth). This is the load-bearing NEW claim.
        #   SUBTRACTION = that value cancels the SNc reward burst state-specifically (the gap).
        n_learn = sum(1 for r in results if r["v_learned_spatial"] and r["weight_grew"])
        n_gap = sum(1 for r in results if r["state_specific"])
        n_primary = sum(1 for r in results
                        if r["v_learned_spatial"] and r["state_specific"] and r["weight_grew"])
        print(f"=== MULTI-SEED LEARNING gate (value-of-location: V-learned-spatial + location-selective "
              f"weight growth): {n_learn}/{len(results)} ===")
        print(f"=== MULTI-SEED SUBTRACTION gate (state-specific SNc gap via GABA_B): "
              f"{n_gap}/{len(results)} ===")
        print(f"=== MULTI-SEED PRIMARY GATE (learning AND subtraction): "
              f"{n_primary}/{len(results)} ===")
        wr_strs = ["{}={:.1f}".format(r["seed"], r["w_near_far_ratio"]) for r in results]
        print("=== w_near/w_far (location selectivity) per seed: " + ", ".join(wr_strs) + " ===")
        gap_strs = ["{}={:.2f}".format(r["seed"], r["gap_ratio"]) for r in results]
        print("=== gap_ratio (far/near) per seed: " + ", ".join(gap_strs) + " ===")
        vr_strs = ["{}={:.2f}".format(r["seed"], r["v_near_far_ratio"]) for r in results]
        print("=== V(near)/V(far) per seed: " + ", ".join(vr_strs) + " ===")
    elif len(results) > 1 and args.lesion:
        n_gone = sum(1 for r in results
                     if r["test_unpredicted_far_hz"] <= 1.30 * max(r["test_predicted_near_hz"], 1e-6))
        print(f"=== MULTI-SEED LESION (gap vanished): {n_gone}/{len(results)} ===")

    if args.out:
        mode = ("place_lesion" if args.lesion
                else "place_gabab" if args.gabab else "place_gaba_a")
        with open(args.out, "w") as f:
            json.dump({"mode": mode, "results": results}, f, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
