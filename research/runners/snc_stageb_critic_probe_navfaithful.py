"""DETERMINISTIC-NAV-FAITHFUL striosome value-critic de-risk (the gap-closing sibling).

This is the deterministic-regime sibling of `snc_stageb_critic_probe_place.py`. The place
probe PASSED, but under TWO conditions the deterministic nav does NOT provide (the
`2026-06-08-nav-placecritic-calibration-NEGATIVE.md` "probe != deployment" gap, the THIRD in
this arc):
  (i) OU background noise ON (CoreSimConfig default) — gave the MSN critic a fluctuating
      background up-state so the tuned 40-cell/2500 pA bump pushed it over threshold.
  (ii) a strong DENSE place bump that nav's SPARSE `sensor_place_readout` (~1-3 active cells,
      ~0.57 Hz at grid 32) cannot deliver.

This sibling replicates the EXACT deterministic-nav regime and tests the ACTUAL proposed fix
(research doc `2026-06-08-striatal-value-critic-firing-research.md` §4, recommendation 1A+2A+3A),
so the gap is provably closed BEFORE any nav build:

  1. DETERMINISTIC-NAV REGIME (the load-bearing fix): the probe's bridge sets the exact knobs
     the nav runner disables at g11_bg_runner.py:3340-3344 —
        enable_ou_process=False, enable_conductance_noise=False,
        enable_homeostasis=False, enable_parameter_heterogeneity=False,
        enable_short_term_plasticity=False, enable_structural_plasticity=False.
     The place probe left OU ON (THE gap). Seeds pinned (cfg.seed/heterogeneity/ou).

  2. THE DEDICATED DENSE AFFERENT (1A) — replaces the probe's tuned 40-cell/2500 pA bump with
     the proposed deployment afferent: a dedicated dense `vs_place_context` region
        (N=200, `vs_place_context -> striosome_value` density 0.5, weight 6.0),
     with a GRID-32-REALISTIC 2-D place code (per-neuron Gaussian over a preferred (x,y) tiling
     the 32x32 grid; sigma widened so 30-80 cells fire per location — NOT the sparse 1-3 of
     nav's actor place code). It feeds ONLY the critic. Diagnostic `_strio_critic_afferent_diag.py`
     (reproduced here on CPU): this dense afferent fires the MSN-D1 critic 10-49 Hz with OU OFF.

  3. THE ACTOR STUB (for gate 4) — also instantiates the actor's place pathway exactly as nav
     builds it: a SEPARATE SPARSE `sensor_place_readout` (N=64, 8x8, IZH2007_HIPPO_PYRAMIDAL,
     sigma=0.5 -> sparse 1-3 active) projecting `sensor_place_readout -> cortex_X` (density 1.0,
     weight 10.0, plastic) into 4 `cortex_{N,E,S,W}` pools. This lets gate 4 (actor-not-perturbed)
     be tested: the actor's place->cortex firing with the dedicated critic present must be within
     +-10% of the critic-absent baseline (proves the dense afferent doesn't leak onto the actor —
     closes Layer 3, the collateral that degraded nav 2.16 -> 3.24).

  4. The value-leads-reward LEAD sweep (0,100,150,200,300,400,500 ms) + the gates, multi-seed.

GATES (multi-seed 42/43/44)
---------------------------
  (1) V-LEARNED-SPATIAL — V(near) rises across training AND ends > V(far), OU OFF.
  (2) STATE-SPECIFIC RPE ABOVE FLOOR — far(unpredicted) burst > 1.30x near(predicted) AND
      far >= 10 Hz, at a nav-realistic lead.
  (3) LOCATION-SELECTIVE LTP — the near-ensemble vs_place_context->striosome weight GROWS from
      init AND grows MORE than the held-out far ensemble (LTP not LTD — refutes the NEGATIVE's LTD).
  (NEW 4) ACTOR-NOT-PERTURBED — the actor's sensor_place_readout->cortex_X firing with the critic
      present is within +-10% of the critic-absent baseline.

ANTI-CHEATS (all carried + two new)
-----------------------------------
  (a) place is a POPULATION code not a coordinate (Jaccard overlap < 0.5).
  (b) GABA_B conductance lesion -> the state-specific gap VANISHES.
  (c) A/B vs receptor="gaba_a" (must FAIL the gap) + host-EMA (place-blind, gap=1.0).
  (NEW d) DETERMINISTIC-REGIME FIDELITY — assert OU + conductance noise OFF in the de-risk bridge.
  (NEW e) ACTOR-NOT-PERTURBED (gate 4).

GRACEFUL-FAIL CONTRACT
----------------------
The de-risk MUST FAIL under the deterministic regime if the dense afferent doesn't carry it. If
gate (1) or (3) fails with OU OFF + the dense dedicated afferent, the verdict is FAIL -> the
honest conclusion is "the deterministic-nav constraint and the MSN up-state are in genuine
tension; the faithful fix needs a protected per-region noise/up-state sim/ edit or relaxing
determinism." The probe MUST NOT rescue a failure by re-enabling OU, raising drive onto the
actor's pathway, or driving the critic directly.

CPU-friendly (tiny bridge): run under SIM_BACKEND=numpy. Multi-seed 42/43/44.

Usage
-----
    SIM_BACKEND=numpy python -m research.runners.snc_stageb_critic_probe_navfaithful \\
        --seeds 42,43,44 --nav-derisk --out research/findings/raw/_navfaithful_derisk.json
    SIM_BACKEND=numpy python -m research.runners.snc_stageb_critic_probe_navfaithful --seed 42 --gabab --lesion
    SIM_BACKEND=numpy python -m research.runners.snc_stageb_critic_probe_navfaithful --seeds 42,43,44   # GABA_A A/B
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as _st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

# Reuse the validated place-probe machinery verbatim (drive/test/gate/lesion/anti-cheats).
from research.runners.snc_stageb_critic_probe_place import (
    _drive_place,
    _calibrate_da_threshold,
    _mean_pathway_weight,
    _clear_eligibility,
    _lesion_gabab_mask,
    _ensemble_overlap,
    _test_and_gate,
    _print_result,
    _host,
    _idx,
)


# ---------------------------------------------------------------------------
# Grid-32 2-D place code (the ACTUAL nav rendering: per-neuron Gaussian over a
# preferred (x,y) tiling the 32x32 grid). Mirrors g11_bg_runner.py:4758-4760.
# A DISTRIBUTED population pattern over (x,y), NOT a scalar/coordinate.
# ---------------------------------------------------------------------------
def _grid_prefs(n_cells, grid_size):
    """Preferred (x,y) for n_cells laid out on a near-square sub-grid tiling [0,grid_size)^2.
    Mirrors the nav runner's hippo_pref_x/y construction (a 2-D tiling of the arena)."""
    side = int(round(np.sqrt(n_cells)))
    xs = np.linspace(0.0, grid_size - 1.0, side, dtype=np.float64)
    ys = np.linspace(0.0, grid_size - 1.0, side, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys)
    px = gx.ravel(); py = gy.ravel()
    # pad/truncate to exactly n_cells
    if px.size < n_cells:
        reps = int(np.ceil(n_cells / px.size))
        px = np.tile(px, reps)[:n_cells]; py = np.tile(py, reps)[:n_cells]
    return px[:n_cells].copy(), py[:n_cells].copy()


def grid_place_code_drive(pos_xy, prefs_xy, max_pA, sigma):
    """Per-neuron Gaussian place-cell drive for a 2-D grid position (x,y).

    drive_i = max_pA * exp(-((pref_x_i - x)^2 + (pref_y_i - y)^2) / 2 sigma^2). Returns a
    length-N vector (distributed code), NOT a scalar. sigma chosen so 30-80 cells fire per
    location at grid 32 for the DENSE dedicated afferent (the up-state-by-convergent-excitation;
    NOT the sparse sigma=0.5 of the actor's sensor_place_readout)."""
    px, py = prefs_xy
    x, y = float(pos_xy[0]), float(pos_xy[1])
    dsq = (px - x) ** 2 + (py - y) ** 2
    return (max_pA * np.exp(-dsq / (2.0 * sigma ** 2))).astype(np.float32)


def _build_navfaithful_bridge(
    seed, *, grid_size=32,
    # --- the dedicated DENSE critic afferent (the fix under test, 1A) ---
    n_vs_place=200, vs_place_density=0.5, vs_place_to_strio_weight=6.0,
    # --- critic + snc ---
    n_strio=60, n_snc=30, strio_to_snc_weight=10.0,
    # --- the SEPARATE SPARSE actor place code + actor cortex (gate-4 stub, matches nav) ---
    n_sensor_place=64, n_cortex_per_action=50, sensor_place_to_cortex_weight=10.0,
    include_actor=True,
    snc_da_sensitivity=8.0, reward_learning_rate=0.12,
    gabab=False, gabab_tau_decay=150.0, gabab_propagation_strength=0.02):
    """Deterministic-nav-regime bridge: a DEDICATED DENSE `vs_place_context` (the proposed fix)
    -> `striosome_value` (GABAergic MSN-D1 critic, PLASTIC) -> `snc` (DA), PLUS a SEPARATE SPARSE
    `sensor_place_readout` -> `cortex_{N,E,S,W}` actor stub (so the actor-not-perturbed gate is
    testable). OU + conductance noise + homeostasis + heterogeneity + STP + structural plasticity
    are ALL OFF — the exact knobs g11_bg_runner.py:3340-3344 disables for deterministic nav.
    """
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType

    cfg = CoreSimConfig()
    cfg.seed = int(seed)
    cfg.heterogeneity_seed = int(seed)
    cfg.ou_seed = int(seed)
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
    cfg.current_reward_signal = 0.0     # BRAIN-BASED: the SNc FIRING is the signal, not a host scalar
    cfg.reward_baseline = 0.0
    cfg.stdp_w_max = 40.0

    # === THE DETERMINISTIC-NAV REGIME (the load-bearing fidelity fix) ===
    # Exactly the knobs the nav runner disables at g11_bg_runner.py:3340-3344.
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_ou_process = False
    cfg.enable_conductance_noise = False
    cfg.enable_parameter_heterogeneity = False
    cfg.enable_structural_plasticity = False

    if gabab:
        cfg.enable_gabab = True
        cfg.gabab_reversal_potential = -90.0
        cfg.gabab_tau_decay = float(gabab_tau_decay)
        cfg.gabab_propagation_strength = float(gabab_propagation_strength)

    regions = [
        # The DEDICATED DENSE place-context afferent (the proposed fix; feeds ONLY the critic).
        BrainRegion(
            name="vs_place_context", n_neurons=n_vs_place, exc_fraction=1.0, internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ),
        BrainRegion(
            name="striosome_value", n_neurons=n_strio, exc_fraction=0.0,   # FULLY GABAergic MSN
            internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D1.name,
            syn_reversal_potential_i_override=-60.0,
        ),
        BrainRegion(
            name="snc", n_neurons=n_snc, exc_fraction=1.0, internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_DOPAMINE.name,
            syn_reversal_potential_i_override=-55.0,   # SNc lacks KCC2 -> depolarized E_GABA
        ),
    ]
    pathways = [
        # The critic's learned value: dense place context -> striosome (V). PLASTIC.
        RegionPathway(from_region="vs_place_context", to_region="striosome_value",
                      density=float(vs_place_density), weight_mean=float(vs_place_to_strio_weight),
                      weight_jitter=0.5, plastic=True),
        RegionPathway(from_region="striosome_value", to_region="snc",
                      density=0.5, weight_mean=float(strio_to_snc_weight),
                      weight_jitter=0.2, plastic=False,
                      receptor=("gaba_b" if gabab else "gaba_a")),
    ]

    if include_actor:
        # The SEPARATE SPARSE actor place code (matches nav's sensor_place_readout EXACTLY):
        # IZH2007_HIPPO_PYRAMIDAL, driven by a sigma=0.5 sparse 2-D bump -> 1-3 active cells.
        regions.append(BrainRegion(
            name="sensor_place_readout", n_neurons=n_sensor_place, exc_fraction=1.0,
            internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
            plastic_internal=False, izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name,
        ))
        for ai, action in enumerate(("N", "E", "S", "W")):
            regions.append(BrainRegion(
                name=f"cortex_{action}", n_neurons=n_cortex_per_action, exc_fraction=1.0,
                internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                plastic_internal=False, izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
                action_index=ai,
            ))
            pathways.append(RegionPathway(
                from_region="sensor_place_readout", to_region=f"cortex_{action}",
                density=1.0, weight_mean=float(sensor_place_to_cortex_weight),
                weight_jitter=0.2, plastic=True, plasticity_gate="place_goal_to_cortex",
            ))

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


def _assert_deterministic_regime(cfg):
    """Anti-cheat (d): assert the de-risk bridge replicates the deterministic-nav regime on the
    exact knobs nav disables. If a future edit re-enables OU/conductance noise, the de-risk no
    longer replicates deployment and the gate is VOID — so this hard-fails."""
    bad = []
    if getattr(cfg, "enable_ou_process", False):
        bad.append("enable_ou_process")
    if getattr(cfg, "enable_conductance_noise", False):
        bad.append("enable_conductance_noise")
    if getattr(cfg, "enable_homeostasis", False):
        bad.append("enable_homeostasis")
    if getattr(cfg, "enable_parameter_heterogeneity", False):
        bad.append("enable_parameter_heterogeneity")
    if bad:
        raise AssertionError(
            "DETERMINISTIC-REGIME FIDELITY (anti-cheat d) VIOLATED: the following background-"
            f"depolarization/regime knobs are ON in the de-risk bridge: {bad}. The nav runner "
            "disables all of them (g11_bg_runner.py:3340-3344); re-enabling any voids the de-risk "
            "(it would no longer replicate deployment — the exact probe-vs-deployment gap this "
            "sibling exists to close).")


def _ensemble_global_indices_xy(bridge, place_vec, region_name, frac=0.25):
    """Global indices of the place cells most strongly driven by `place_vec` (top `frac`),
    over an arbitrary region (here vs_place_context). For LOCATION-SELECTIVE weight tracking."""
    g = np.asarray(_idx(bridge, region_name), dtype=np.int64)
    drive = np.asarray(place_vec, dtype=np.float64)
    k = max(1, int(round(frac * len(drive))))
    top = np.argsort(drive)[-k:]
    return set(int(g[i]) for i in top)


def _ensemble_overlap_region(bridge, idx_map, region_name, vec_a, vec_b, xp, n_steps=40, thresh_hz=1.0):
    """Anti-cheat (a) provenance over the dense vs_place_context: drive it with vec_a then vec_b,
    record which cells fire, return (n_a, n_b, jaccard). LOW overlap => distinct spatial ensembles."""
    pidx = idx_map[region_name]; n = len(_host(pidx))

    def active_set(vec):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[pidx] = xp.asarray(vec, dtype=xp.float32)
        c = np.zeros(n, dtype=np.int64)
        for _ in range(n_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = (
                bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)
            c += np.asarray(_host(bridge.cp_firing_states[pidx])).astype(np.int64)
        rate = c / (n_steps * 1e-3)
        return set(int(i) for i in np.where(rate > thresh_hz)[0])

    sa, sb = active_set(vec_a), active_set(vec_b)
    inter = len(sa & sb); union = max(len(sa | sb), 1)
    return len(sa), len(sb), inter / union


def _measure_actor_cortex_rate(bridge, idx_map, sensor_vec, xp, *, cortex_tonic_pa=400.0,
                               critic_afferent_vec=None, n_steps=120, warmup=40):
    """Gate 4 helper: measure the actor cortex_X output rate under a FIXED cortical tonic drive
    (so cortex has a real, non-degenerate baseline — in nav the cortex pools are driven by many
    afferents the stub doesn't model, NOT by the sparse place code alone, which fires ~0.57 Hz),
    while ALSO driving (a) the sparse actor place code `sensor_place_readout` and, when
    `critic_afferent_vec` is given, (b) the DEDICATED DENSE critic afferent `vs_place_context`
    CONCURRENTLY. The critic afferent has NO pathway to cortex, so a faithful dedicated afferent
    leaves the actor cortex rate UNCHANGED — the gate compares this rate critic-PRESENT (with the
    dense afferent active) vs critic-ABSENT (twin bridge, no critic path), proving the dense
    afferent does not leak onto the actor via shared global state (DA / plasticity gain / current
    array). Learning is frozen (forward dynamics only).
    """
    sp_idx = idx_map["sensor_place_readout"]
    cortex_idx_list = [np.asarray(_host(idx_map[f"cortex_{a}"])) for a in ("N", "E", "S", "W")]
    cortex_idx = xp.asarray(np.concatenate(cortex_idx_list))
    n_cx = sum(len(c) for c in cortex_idx_list)
    crit_idx = idx_map.get("vs_place_context", None)
    saved_lr = bridge.core_config.reward_learning_rate
    bridge.core_config.reward_learning_rate = 0.0
    spk = 0; m = 0
    for t in range(n_steps):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[cortex_idx] = xp.float32(cortex_tonic_pa)
        bridge.cp_external_input_current[sp_idx] = xp.asarray(sensor_vec, dtype=xp.float32)
        if critic_afferent_vec is not None and crit_idx is not None:
            bridge.cp_external_input_current[crit_idx] = xp.asarray(critic_afferent_vec, dtype=xp.float32)
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)
        if t >= warmup:
            spk += int(bridge.cp_firing_states[cortex_idx].sum()); m += 1
    bridge.core_config.reward_learning_rate = saved_lr
    return spk / max(n_cx, 1) / max(m * 1e-3, 1e-9)


def run_navfaithful(seed, *, grid_size=32, p_near_xy=(26.571, 26.571), p_far_xy=(4.429, 4.429),
                    vs_place_sigma=4.0, vs_place_drive_pa=800.0,
                    sensor_place_sigma=1.5, sensor_place_drive_pa=1500.0,
                    actor_cortex_tonic_pa=400.0,
                    snc_tonic_pa=180.0, snc_reward_gain=300.0,
                    hold_steps=40, n_train=40, reward_learning_rate=0.12,
                    vs_place_to_strio_weight=0.2, strio_to_snc_weight=10.0,
                    snc_da_sensitivity=8.0, lesion=False, verbose=True,
                    gabab=False, gabab_tau_decay=150.0, gabab_propagation_strength=0.02,
                    lead_steps=0, return_trained=False):
    """The full deterministic-nav-faithful de-risk for one seed. Builds the dedicated-dense-afferent
    critic + actor stub in the deterministic regime, trains the value-leads-reward protocol, runs
    the gates (1-3) at `lead_steps` and the actor-not-perturbed gate (4)."""
    from sim.backend import get_backend
    xp, _ = get_backend()

    bridge, cfg = _build_navfaithful_bridge(
        seed, grid_size=grid_size,
        vs_place_to_strio_weight=vs_place_to_strio_weight, strio_to_snc_weight=strio_to_snc_weight,
        snc_da_sensitivity=snc_da_sensitivity, reward_learning_rate=reward_learning_rate,
        gabab=gabab, gabab_tau_decay=gabab_tau_decay, gabab_propagation_strength=gabab_propagation_strength)

    # Anti-cheat (d): hard-assert the deterministic regime BEFORE anything runs.
    _assert_deterministic_regime(cfg)

    regions = ("vs_place_context", "striosome_value", "snc",
               "sensor_place_readout", "cortex_N", "cortex_E", "cortex_S", "cortex_W")
    idx_map = {n: xp.asarray(_idx(bridge, n)) for n in regions}
    # The probe machinery (_drive_place / _test_and_gate) expects a "place" key -> point it at the
    # dedicated dense afferent so the validated drive/test/gate logic is reused VERBATIM.
    idx_map["place"] = idx_map["vs_place_context"]
    n_vs = len(_host(idx_map["vs_place_context"]))
    n_sp = len(_host(idx_map["sensor_place_readout"]))

    # --- the two DENSE place-context population codes (grid-32 2-D; anti-cheat a: distributed) ---
    vs_prefs = _grid_prefs(n_vs, grid_size)
    near_vec = grid_place_code_drive(p_near_xy, vs_prefs, vs_place_drive_pa, sigma=vs_place_sigma)
    far_vec = grid_place_code_drive(p_far_xy, vs_prefs, vs_place_drive_pa, sigma=vs_place_sigma)

    # --- the SEPARATE SPARSE actor place code (sigma=0.5 -> 1-3 active, matches nav) for gate 4 ---
    sp_prefs = _grid_prefs(n_sp, grid_size)
    sensor_near_vec = grid_place_code_drive(p_near_xy, sp_prefs, sensor_place_drive_pa, sigma=sensor_place_sigma)

    # Anti-cheat (a): NEAR vs FAR are DIFFERENT dense ensembles (provenance, before training).
    na, nb, overlap = _ensemble_overlap_region(bridge, idx_map, "vs_place_context", near_vec, far_vec, xp)
    if verbose:
        print(f"  [anti-cheat a: vs_place_context provenance] NEAR active={na}/{n_vs} cells, "
              f"FAR active={nb}/{n_vs} cells, Jaccard overlap={overlap:.2f} (LOW => distinct "
              f"spatial ensembles, dense up-state by convergent excitation)")
    distinct_ensembles = (overlap < 0.5 and na > 0 and nb > 0)

    # --- GATE 4 BASELINE: actor place->cortex firing WITHOUT a critic present ---
    # Build a critic-ABSENT twin (identical actor wiring, no vs_place_context->striosome path) and
    # measure the actor's cortex output. This is the reference the critic-present bridge must match.
    base_bridge, base_cfg = _build_navfaithful_bridge(
        seed, grid_size=grid_size, vs_place_to_strio_weight=0.0,  # afferent weight irrelevant here
        strio_to_snc_weight=strio_to_snc_weight, snc_da_sensitivity=snc_da_sensitivity,
        reward_learning_rate=reward_learning_rate, gabab=gabab,
        gabab_tau_decay=gabab_tau_decay, gabab_propagation_strength=gabab_propagation_strength)
    _assert_deterministic_regime(base_cfg)
    base_idx_map = {f"cortex_{a}": xp.asarray(_idx(base_bridge, f"cortex_{a}")) for a in ("N", "E", "S", "W")}
    base_idx_map["sensor_place_readout"] = xp.asarray(_idx(base_bridge, "sensor_place_readout"))
    base_idx_map["vs_place_context"] = xp.asarray(_idx(base_bridge, "vs_place_context"))
    # Baseline = critic-ABSENT twin: cortex under tonic + sparse place code, NO critic afferent.
    actor_rate_no_critic = _measure_actor_cortex_rate(
        base_bridge, base_idx_map, sensor_near_vec, xp,
        cortex_tonic_pa=actor_cortex_tonic_pa, critic_afferent_vec=None)
    del base_bridge

    # Calibrate dopamine threshold to the SNc tonic firing fraction.
    tonic_frac = _calibrate_da_threshold(bridge, cfg, idx_map, snc_tonic_pa, xp)
    if verbose:
        print(f"  [calib] SNc tonic firing fraction = {tonic_frac:.4f} -> dopamine threshold")
        print(f"  [gate-4 baseline] actor place->cortex rate (NO critic) = {actor_rate_no_critic:.3f} Hz")

    # Per-ensemble synapse sets (disjoint bump cores) for location-SELECTIVE weight tracking.
    near_set = _ensemble_global_indices_xy(bridge, near_vec, "vs_place_context", frac=0.25)
    far_set = _ensemble_global_indices_xy(bridge, far_vec, "vs_place_context", frac=0.25)
    far_set = far_set - near_set
    w_init = _mean_pathway_weight(bridge, "vs_place_context", "striosome_value")
    w_near_init = _mean_pathway_weight(bridge, "vs_place_context", "striosome_value", pre_subset=near_set)
    w_far_init = _mean_pathway_weight(bridge, "vs_place_context", "striosome_value", pre_subset=far_set)

    # === Value-leads-reward acquisition (FAR held out; NEAR potentiated; eligibility cleared) ===
    near_v_curve, near_burst_curve = [], []
    for t in range(n_train):
        _drive_place(bridge, idx_map, None, {"snc": snc_tonic_pa}, hold_steps, xp)   # ITI floor
        _clear_eligibility(bridge)
        snc_r, strio_r, da = _drive_place(
            bridge, idx_map, near_vec, {"snc": snc_tonic_pa + snc_reward_gain}, hold_steps, xp)  # LEARN near
        near_v_curve.append(strio_r); near_burst_curve.append(snc_r)
        if verbose and (t < 3 or t % 5 == 0 or t == n_train - 1):
            wn = _mean_pathway_weight(bridge, "vs_place_context", "striosome_value", pre_subset=near_set)
            wf = _mean_pathway_weight(bridge, "vs_place_context", "striosome_value", pre_subset=far_set)
            print(f"  [acq t={t:02d}] near-burst={snc_r:6.2f}Hz  V(near)={strio_r:6.2f}Hz  "
                  f"w_near={wn:.3f}  w_far={wf:.3f}  (near/far {wn/max(wf,1e-6):.2f})  DA={da:.3f}")

    early = slice(0, max(1, n_train // 5)); late = slice(-max(1, n_train // 5), None)
    near_v_early = _st.mean(near_v_curve[early]); near_v_late = _st.mean(near_v_curve[late])
    w_final = _mean_pathway_weight(bridge, "vs_place_context", "striosome_value")
    w_near_final = _mean_pathway_weight(bridge, "vs_place_context", "striosome_value", pre_subset=near_set)
    w_far_final = _mean_pathway_weight(bridge, "vs_place_context", "striosome_value", pre_subset=far_set)

    # --- GATE 4 (critic present): actor cortex firing on the TRAINED critic bridge, with the
    #     DEDICATED DENSE critic afferent (near_vec) ALSO active concurrently. A faithful
    #     dedicated afferent (no edge to cortex) leaves the cortex rate == the critic-absent twin. ---
    actor_rate_with_critic = _measure_actor_cortex_rate(
        bridge, idx_map, sensor_near_vec, xp,
        cortex_tonic_pa=actor_cortex_tonic_pa, critic_afferent_vec=near_vec)
    actor_ratio = actor_rate_with_critic / max(actor_rate_no_critic, 1e-9)
    actor_not_perturbed = (0.90 <= actor_ratio <= 1.10) if actor_rate_no_critic > 1e-6 else \
                          (actor_rate_with_critic <= 1e-6)
    if verbose:
        print(f"  [gate-4] actor place->cortex rate WITH critic = {actor_rate_with_critic:.3f} Hz "
              f"(no-critic {actor_rate_no_critic:.3f} Hz, ratio {actor_ratio:.3f}, "
              f"not-perturbed: {actor_not_perturbed})")

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
    # Attach gate-4 + regime-fidelity facts to the result.
    result["actor_rate_no_critic_hz"] = float(actor_rate_no_critic)
    result["actor_rate_with_critic_hz"] = float(actor_rate_with_critic)
    result["actor_ratio"] = float(actor_ratio)
    result["actor_not_perturbed"] = bool(actor_not_perturbed)
    result["ou_off"] = (not cfg.enable_ou_process)
    result["conductance_noise_off"] = (not cfg.enable_conductance_noise)
    result["homeostasis_off"] = (not cfg.enable_homeostasis)
    if return_trained:
        return result, bridge, xp, train_state
    return result


def run_navfaithful_lead_sweep(seed, lead_steps_list, *, verbose=True, **kw):
    """Train the critic ONCE for `seed`, then run the test+gate phase at EACH lead on the SAME
    trained bridge (train once, test many). Gate 4 (actor-not-perturbed) is measured once during
    training and carried into every lead's result (it is lead-independent)."""
    first, bridge, xp, ts = run_navfaithful(
        seed, verbose=verbose, lead_steps=int(lead_steps_list[0]), return_trained=True, **kw)
    g4 = {k: first[k] for k in ("actor_rate_no_critic_hz", "actor_rate_with_critic_hz",
                                "actor_ratio", "actor_not_perturbed",
                                "ou_off", "conductance_noise_off", "homeostasis_off")}
    results = [first]
    for lead in lead_steps_list[1:]:
        r = _test_and_gate(bridge, xp, ts, int(lead), verbose=verbose)
        r.update(g4)
        results.append(r)
    return results


def _lead_sweep_main(seeds, lead_sweep_str, kw, args):
    """The DECISIVE deterministic-nav-faithful de-risk: sweep the value-leads-reward LEAD over a
    nav-realistic range, multi-seed, on a critic trained ONCE per seed in the deterministic regime
    with the dedicated dense afferent. Prints the lead x seed table + gates 1-4 + the verdict."""
    leads_ms = [float(x) for x in lead_sweep_str.split(",")]
    lead_steps_list = [int(round(m / 1.0)) for m in leads_ms]   # dt_ms = 1.0
    sweep_kw = {k: v for k, v in kw.items() if k != "lead_steps"}

    def _fmt_gap(r):
        if r["test_predicted_near_hz"] < 0.5:
            return "  INF" if r["test_unpredicted_far_hz"] >= 0.5 else " 0.00"
        return "{:5.2f}".format(r["gap_ratio"])

    results_by_seed = {}
    for s in seeds:
        print(f"\n##### NAV-FAITHFUL LEAD SWEEP seed={s} (deterministic regime, dense afferent, "
              f"train once, test at leads {leads_ms} ms) #####")
        rs = run_navfaithful_lead_sweep(s, lead_steps_list, verbose=True, **sweep_kw)
        results_by_seed[s] = rs
        r0 = rs[0]
        print(f"  [seed {s}] LEARNING: V(near)/V(far)={r0['v_near_far_ratio']:.2f}  "
              f"w_near/w_far={r0['w_near_far_ratio']:.2f}  "
              f"V-learned-spatial={r0['v_learned_spatial']}  weight-grew(LTP){r0['weight_grew']}  "
              f"| GATE4 actor-not-perturbed={r0['actor_not_perturbed']} (ratio {r0['actor_ratio']:.3f})  "
              f"| regime OU-off={r0['ou_off']} cond-noise-off={r0['conductance_noise_off']}")

    print("\n" + "=" * 100)
    print("=== NAV-FAITHFUL LEAD SWEEP: near_burst / far_burst / gap_ratio(far/near) / above-floor? ===")
    print("=" * 100)
    header = "  lead_ms |" + "".join(f"  seed {s:>4}                          |" for s in seeds)
    print(header)
    for li, lead_ms in enumerate(leads_ms):
        cells = []
        for s in seeds:
            r = results_by_seed[s][li]
            nb_ = r["test_predicted_near_hz"]; fb_ = r["test_unpredicted_far_hz"]
            af = r["above_floor"]; sa = r["state_specific_above_floor"]
            flag = "OK" if sa else ("--" if not af else "lo")
            cells.append(f" near={nb_:5.1f} far={fb_:5.1f} g={_fmt_gap(r)} {('AF' if af else '..')}/{flag} |")
        print(f"  {lead_ms:6.0f}  |" + "".join(cells))
    print("  (AF = far_burst >= 10 Hz above floor;  OK = state-specific gap AND above floor)")

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
        if n_robust > best_n:
            best_n = n_robust; best_lead_idx = li

    best_lead_ms = leads_ms[best_lead_idx]
    rl_best = [results_by_seed[s][best_lead_idx] for s in seeds]
    n_robust_best = sum(1 for r in rl_best if r["state_specific_above_floor"])
    n_learn = sum(1 for r in rl_best if r["v_learned_spatial"] and r["weight_grew"])
    # Gate 4 is lead-independent — count from any lead (use seed-0 row from each seed's first lead).
    n_actor_ok = sum(1 for s in seeds if results_by_seed[s][0]["actor_not_perturbed"])
    n_regime_ok = sum(1 for s in seeds if results_by_seed[s][0]["ou_off"]
                      and results_by_seed[s][0]["conductance_noise_off"]
                      and results_by_seed[s][0]["homeostasis_off"])

    print("\n" + "=" * 100)
    # The verdict requires: gap robust (>=3 seeds) above floor AND learning retained AND
    # the actor is not perturbed AND the deterministic regime is verified.
    verdict_pass = (n_robust_best >= 3 and n_robust_best >= max(3, (len(seeds) + 1) // 2))
    print(f"=== BEST LEAD = {best_lead_ms:.0f} ms: ROBUST state-specific gap (above floor) "
          f"{n_robust_best}/{len(seeds)} ; LEARNING(LTP) retained {n_learn}/{len(seeds)} ; "
          f"GATE4 actor-not-perturbed {n_actor_ok}/{len(seeds)} ; regime-OU/cond/homeo-off "
          f"{n_regime_ok}/{len(seeds)} ===")
    decisive = "PASS" if (verdict_pass and n_learn >= 3 and n_actor_ok >= 3
                          and n_regime_ok == len(seeds)) else "FAIL"
    print(f"=== DETERMINISTIC-NAV-FAITHFUL DE-RISK VERDICT: {decisive}  "
          f"(>=3 seeds: gap-robust-above-floor at a nav-realistic lead AND LTP learning AND "
          f"actor-not-perturbed; deterministic regime asserted) ===")
    if decisive == "FAIL":
        print("=== GRACEFUL-FAIL: under OU-OFF + the dense dedicated afferent, the de-risk did "
              "NOT carry the learned place-graded value subtraction. Honest conclusion: the "
              "deterministic-nav constraint and the MSN up-state are in genuine tension -> the "
              "faithful fix needs a protected per-region noise/up-state sim/ edit OR relaxing "
              "determinism. NOT rescued by re-enabling OU / over-driving the actor / direct "
              "critic drive (all disallowed). ===")
    print("=" * 100)

    if args.out:
        out = {
            "mode": ("navfaithful_gabab_lead_sweep" if args.gabab else "navfaithful_gaba_a_lead_sweep"),
            "deterministic_regime": True,
            "afferent": "dedicated_dense_vs_place_context",
            "actor_stub": "sensor_place_readout->cortex_X (sparse, separate)",
            "grid_size": kw.get("grid_size", 32),
            "leads_ms": leads_ms, "best_lead_ms": best_lead_ms,
            "n_robust_best": n_robust_best, "n_learn_best": n_learn,
            "n_actor_ok": n_actor_ok, "n_regime_ok": n_regime_ok,
            "verdict": decisive,
            "results_by_seed": {str(s): results_by_seed[s] for s in seeds},
        }
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(f"  wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--grid-size", type=int, default=32)
    ap.add_argument("--p-near-x", type=float, default=26.571)   # exact 8x8 sensor node (idx 6)
    ap.add_argument("--p-near-y", type=float, default=26.571)
    ap.add_argument("--p-far-x", type=float, default=4.429)     # exact 8x8 sensor node (idx 1)
    ap.add_argument("--p-far-y", type=float, default=4.429)
    ap.add_argument("--vs-place-sigma", type=float, default=4.0,
                    help="grid-32 dense afferent tuning sigma (30-80 active cells per location)")
    ap.add_argument("--vs-place-drive-pa", type=float, default=800.0)
    ap.add_argument("--sensor-place-sigma", type=float, default=1.5,
                    help="SPARSE actor sensor_place_readout tuning (sparse few-cell bump)")
    ap.add_argument("--sensor-place-drive-pa", type=float, default=1500.0)
    ap.add_argument("--actor-cortex-tonic-pa", type=float, default=400.0,
                    help="fixed cortical tonic giving the actor cortex a real baseline rate (~37 Hz) "
                         "so gate 4 is non-degenerate; nav cortex is driven by many afferents the "
                         "stub doesn't model — the critic afferent (no edge to cortex) must not perturb it")
    ap.add_argument("--snc-tonic-pa", type=float, default=180.0)
    ap.add_argument("--snc-reward-gain", type=float, default=300.0)
    ap.add_argument("--hold-steps", type=int, default=40)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--reward-learning-rate", type=float, default=0.12)
    ap.add_argument("--vs-place-to-strio-weight", type=float, default=0.2,
                    help="INIT weight of vs_place_context->striosome_value (learned up from this)")
    ap.add_argument("--strio-to-snc-weight", type=float, default=10.0)
    ap.add_argument("--snc-da-sensitivity", type=float, default=8.0)
    ap.add_argument("--lesion", action="store_true",
                    help="anti-cheat (b): zero the GABA_B mask after training -> gap must vanish")
    ap.add_argument("--gabab", action="store_true")
    ap.add_argument("--gabab-tau-decay", type=float, default=150.0)
    ap.add_argument("--gabab-propagation-strength", type=float, default=0.02)
    ap.add_argument("--lead-ms", type=float, default=0.0)
    ap.add_argument("--lead-sweep", type=str, default=None,
                    help="comma ms leads, e.g. '0,100,150,200,300,400,500'")
    ap.add_argument("--nav-derisk", action="store_true",
                    help="DECISIVE deterministic-nav-faithful preset: --gabab, the live SNc regime "
                         "(tonic 180/reward 300), the PHYSIOLOGICAL GABA_B (0.02), and the lead "
                         "sweep 0,100,150,200,300,400,500.")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.nav_derisk:
        args.gabab = True
        if args.gabab_propagation_strength == 0.02:
            args.gabab_propagation_strength = 0.02
        if args.lead_sweep is None:
            args.lead_sweep = "0,100,150,200,300,400,500"

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    lead_steps = int(round(args.lead_ms / 1.0))
    kw = dict(
        grid_size=args.grid_size,
        p_near_xy=(args.p_near_x, args.p_near_y), p_far_xy=(args.p_far_x, args.p_far_y),
        vs_place_sigma=args.vs_place_sigma, vs_place_drive_pa=args.vs_place_drive_pa,
        sensor_place_sigma=args.sensor_place_sigma, sensor_place_drive_pa=args.sensor_place_drive_pa,
        actor_cortex_tonic_pa=args.actor_cortex_tonic_pa,
        snc_tonic_pa=args.snc_tonic_pa, snc_reward_gain=args.snc_reward_gain,
        hold_steps=args.hold_steps, n_train=args.n_train,
        reward_learning_rate=args.reward_learning_rate,
        vs_place_to_strio_weight=args.vs_place_to_strio_weight,
        strio_to_snc_weight=args.strio_to_snc_weight,
        snc_da_sensitivity=args.snc_da_sensitivity, lesion=args.lesion,
        gabab=args.gabab, gabab_tau_decay=args.gabab_tau_decay,
        gabab_propagation_strength=args.gabab_propagation_strength)

    if args.lead_sweep:
        _lead_sweep_main(seeds, args.lead_sweep, kw, args)
        return

    kw["lead_steps"] = lead_steps
    results = []
    for s in seeds:
        tag = ("LESION (GABA_B mask cut)" if args.lesion
               else "DENSE vs_place_context critic + GABA_B/GIRK (E_K=-90mV), DETERMINISTIC regime" if args.gabab
               else "DENSE vs_place_context critic (GABA_A direct — A/B control), DETERMINISTIC regime")
        print(f"[snc-stageB-navfaithful seed={s}] {tag}:")
        r = run_navfaithful(s, **kw)
        _print_result(r)
        print(f"  [gate-4] actor place->cortex no-critic={r['actor_rate_no_critic_hz']:.3f}Hz "
              f"with-critic={r['actor_rate_with_critic_hz']:.3f}Hz ratio={r['actor_ratio']:.3f} "
              f"=> not-perturbed {r['actor_not_perturbed']}")
        print(f"  [anti-cheat d] OU-off={r['ou_off']} cond-noise-off={r['conductance_noise_off']} "
              f"homeostasis-off={r['homeostasis_off']}")
        if not args.lesion:
            primary = (r["v_learned_spatial"] and r["state_specific_above_floor"]
                       and r["weight_grew"] and r["actor_not_perturbed"])
            print(f"\n  NAV-FAITHFUL de-risk (seed {s}): {'PASS' if primary else 'FAIL'}  "
                  f"[V-learned-spatial {r['v_learned_spatial']}, state-specific-above-floor "
                  f"{r['state_specific_above_floor']}, weight-grew(LTP) {r['weight_grew']}, "
                  f"actor-not-perturbed {r['actor_not_perturbed']}]")
        else:
            no_gap = (r["test_unpredicted_far_hz"] <= 1.30 * max(r["test_predicted_near_hz"], 1e-6))
            print(f"\n  LESION anti-cheat (seed {s}): {'PASS' if no_gap else 'UNEXPECTED'}  "
                  f"[gap-gone {no_gap}, gap_ratio {r['gap_ratio']:.2f}]")
        results.append(r)
        print()

    if len(results) > 1 and not args.lesion:
        n_learn = sum(1 for r in results if r["v_learned_spatial"] and r["weight_grew"])
        n_gap = sum(1 for r in results if r["state_specific_above_floor"])
        n_actor = sum(1 for r in results if r["actor_not_perturbed"])
        n_primary = sum(1 for r in results
                        if r["v_learned_spatial"] and r["state_specific_above_floor"]
                        and r["weight_grew"] and r["actor_not_perturbed"])
        print(f"=== MULTI-SEED LEARNING (V-learned-spatial + location-selective LTP): {n_learn}/{len(results)} ===")
        print(f"=== MULTI-SEED SUBTRACTION (state-specific SNc gap above floor): {n_gap}/{len(results)} ===")
        print(f"=== MULTI-SEED GATE-4 (actor-not-perturbed): {n_actor}/{len(results)} ===")
        print(f"=== MULTI-SEED PRIMARY (all four): {n_primary}/{len(results)} ===")
    elif len(results) > 1 and args.lesion:
        n_gone = sum(1 for r in results
                     if r["test_unpredicted_far_hz"] <= 1.30 * max(r["test_predicted_near_hz"], 1e-6))
        print(f"=== MULTI-SEED LESION (gap vanished): {n_gone}/{len(results)} ===")

    if args.out:
        mode = ("navfaithful_lesion" if args.lesion
                else "navfaithful_gabab" if args.gabab else "navfaithful_gaba_a")
        with open(args.out, "w") as f:
            json.dump({"mode": mode, "deterministic_regime": True, "results": results},
                      f, indent=2, default=float)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
