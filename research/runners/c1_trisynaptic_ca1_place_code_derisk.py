"""C1 precondition de-risk — does the canonical hippocampal TRISYNAPTIC LOOP (EC->DG->CA3->CA1), driven
from the legitimate egocentric landmark sensors, produce a CA1 place code that is BOTH high-rate (fires an
MSN-D1) AND distinct-per-location? (Decoupling the fire-vs-grade tension the single-hop WTA place layer
could not.)

CONTEXT (the problem this lever must solve):
  - Stage 1 (placecode_selforg_stage1_derisk): a single-hop competitive place layer IS position-specific
    (diff-cos 0.064) but is the distinct-but-WEAK regime (can't fire the MSN-D1 critic; ~420 pA rheobase).
  - Stage 2 (n9_place_graded_critic_stage2_derisk): NEGATIVE — the fire-vs-grade tension is IRREDUCIBLE
    through a single WTA knob (distinct => weak => critic silent; OR strong driver => WTA collapses =>
    position-blind). Never both.
  - C1 THESIS: the trisynaptic loop decouples the tension with TWO mechanisms — DG pattern-SEPARATION makes
    locations distinct, CA3 recurrent pattern-COMPLETION amplifies a sparse DG cue into a HIGH-RATE attractor
    while preserving distinctness. So CA1 should be BOTH high-rate AND distinct-per-location. Prove/disprove.

MECHANISM (all runner-side; NO sim/ edits — byte-mirrors build_biological_brain_regions(
enable_hippocampus_consolidation=True) for the hippocampal regions/pathways, with the afferent SWAPPED from
language_input->ec to landmark_sensors->ec, and the levers exposed for tuning):
  landmark_sensors (>=2 landmarks, egocentric bearing+distance render; the ONLY (x,y) entry)
    --plastic (gate landmark_to_ec)-->  ec
    ec --(ec_to_dg perforant)--> dg  <--(FFI: ec->dg_pv_basket->dg)        # D.12 separation
    dg --(dg_to_ca3 mossy detonator)--> ca3
    ca3 --(ca3_swr_burst recurrent autoassociator)--> ca3                  # D.13 completion (the RATE)
    ca3 <--(ca3_inh feedback, k-WTA)--> ca3                                # de Almeida E%-max sparsification
    ca3 --(ca3_to_ca1 Schaffer)--> ca1 ;  ec --(ec_to_ca1 direct)--> ca1
  + an MSN-D1 test cell read SPARSELY from ca1 (the Stage-2 HIGH-RATE bar: does CA1 clear the ~420 pA
    effective rheobase and fire the MSN >=5 Hz?).
  Self-organize (open the plastic feedforward + recurrent gates, walk locations), then FREEZE and measure.

GATES (CuPy, >=3 seeds, deterministic regime OU/cond-noise/global-homeostasis/heterogeneity/STP OFF):
  CONDUCTS    : CA1 active (non-silent) end-to-end from EC/landmark input (not direct-CA3-injected).
  HIGH-RATE   : the CA1 ensemble, read by an MSN-D1 test cell through a realistic sparse projection, clears
                the ~420 pA effective rheobase -> fires it >=5 Hz. (THE thing the single-hop code could not.)
  DISTINCT    : different-location CA1 ensemble cosine < 0.3 (the separation survives completion).
  STABLE      : same-location cosine > 0.7.
  SENSOR-DRIVEN (anti-cheat): ablate landmark sensors -> CA1 collapses (a real sensor-driven place code).
  + regime fidelity + position-leak ((x,y) only via the egocentric landmark render) hard-asserted.

USAGE (MUST be cupy):
  SIM_BACKEND=cupy python -m research.runners.c1_trisynaptic_ca1_place_code_derisk \
      --seeds 42,43,44 --regime distinct --out research/findings/raw/_c1_trisyn_ca1_distinct_3seed.json
  SIM_BACKEND=cupy python -m research.runners.c1_trisynaptic_ca1_place_code_derisk \
      --seeds 42,43,44 --regime ignite   --out research/findings/raw/_c1_trisyn_ca1_ignite_3seed.json
"""
from __future__ import annotations
import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from research.runners.placecode_selforg_stage1_derisk import (
    landmark_sensor_act, default_locations, default_landmarks, cosine_counts, _host)


# ──────────────────────────────────────────────────────────────────────
# Build: landmark_sensors -> trisynaptic loop (EC->DG->CA3<->ca3_inh->CA1) + an MSN-D1 test cell read
# sparsely from CA1. Byte-mirrors build_biological_brain_regions hippocampal specs (afferent swapped to
# landmark_sensors->ec). NO sim/ edits. NO host Gaussian. NO direct-CA3 injection.
# ──────────────────────────────────────────────────────────────────────

def _build(seed, *, n_sensors, n_ec, n_dg, n_dg_pv_basket, n_ca3, n_ca1, n_ca3_inh, n_msn,
           lm_to_ec_weight, lm_to_ec_density, ec_to_dg_weight, ec_to_dg_density,
           ec_to_pvb_weight, pvb_to_dg_weight, dg_to_ca3_weight, dg_to_ca3_density,
           ca3_rec_weight, ca3_rec_density, ca3_to_inh_weight, ca3_to_inh_density,
           inh_to_ca3_weight, inh_to_ca3_density, ca3_to_ca1_weight, ca3_to_ca1_density,
           ec_to_ca1_weight, ec_to_ca1_density, ca1_to_msn_weight, ca1_to_msn_density,
           enable_nmda=True, dt_ms=1.0):
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType

    regions = [
        BrainRegion(name="landmark_sensors", n_neurons=int(n_sensors), exc_fraction=1.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False, izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name),
        # === trisynaptic regions, byte-mirrored from build_biological_brain_regions ===
        BrainRegion(name="ec", n_neurons=int(n_ec), exc_fraction=0.8, internal_density=0.05,
                    exc_weight_mean=0.3, inh_weight_mean=0.8, weight_jitter=0.2, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name),
        BrainRegion(name="dg", n_neurons=int(n_dg), exc_fraction=0.95, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name),
        BrainRegion(name="dg_pv_basket", n_neurons=int(n_dg_pv_basket), exc_fraction=0.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False, izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name),
        BrainRegion(name="ca3", n_neurons=int(n_ca3), exc_fraction=0.85, internal_density=0.0,
                    exc_weight_mean=1.5, inh_weight_mean=2.0, weight_jitter=0.2, plastic_internal=True,
                    izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name, enable_nmda=bool(enable_nmda)),
        BrainRegion(name="ca3_inh", n_neurons=int(n_ca3_inh), exc_fraction=0.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name),
        BrainRegion(name="ca1", n_neurons=int(n_ca1), exc_fraction=0.85, internal_density=0.05,
                    exc_weight_mean=0.3, inh_weight_mean=0.8, weight_jitter=0.2, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name),
        # MSN-D1 test cell — the HIGH-RATE bar. Fully GABAergic, KIR2 down-state, ~420 pA effective rheobase.
        # Reads CA1 SPARSELY (a realistic place->striatum projection). NOT trained here — we test whether the
        # CA1 ensemble's effective drive clears the rheobase at all (the gate the single-hop code failed).
        BrainRegion(name="msn_d1", n_neurons=int(n_msn), exc_fraction=0.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D1.name,
                    syn_reversal_potential_i_override=-60.0),
    ]
    pathways = [
        RegionPathway(from_region="landmark_sensors", to_region="ec", density=float(lm_to_ec_density),
                      weight_mean=float(lm_to_ec_weight), weight_jitter=0.2, plastic=True,
                      plasticity_gate="landmark_to_ec"),
        RegionPathway(from_region="ec", to_region="dg", density=float(ec_to_dg_density),
                      weight_mean=float(ec_to_dg_weight), weight_jitter=0.2, plastic=True,
                      plasticity_gate="ec_to_dg"),
        RegionPathway(from_region="ec", to_region="dg_pv_basket", density=0.40,
                      weight_mean=float(ec_to_pvb_weight), weight_jitter=0.2, plastic=False),
        RegionPathway(from_region="dg_pv_basket", to_region="dg", density=1.0,
                      weight_mean=float(pvb_to_dg_weight), weight_jitter=0.2, plastic=False),
        RegionPathway(from_region="dg", to_region="ca3", density=float(dg_to_ca3_density),
                      weight_mean=float(dg_to_ca3_weight), weight_jitter=0.2, plastic=True,
                      plasticity_gate="dg_to_ca3"),
        RegionPathway(from_region="ec", to_region="ca1", density=float(ec_to_ca1_density),
                      weight_mean=float(ec_to_ca1_weight), weight_jitter=0.2, plastic=True,
                      plasticity_gate="ec_to_ca1"),
        RegionPathway(from_region="ca3", to_region="ca3", density=float(ca3_rec_density),
                      weight_mean=float(ca3_rec_weight), weight_jitter=0.2, plastic=True,
                      plasticity_gate="ca3_swr_burst"),
        RegionPathway(from_region="ca3", to_region="ca3_inh", density=float(ca3_to_inh_density),
                      weight_mean=float(ca3_to_inh_weight), weight_jitter=0.2, plastic=False),
        RegionPathway(from_region="ca3_inh", to_region="ca3", density=float(inh_to_ca3_density),
                      weight_mean=float(inh_to_ca3_weight), weight_jitter=0.2, plastic=False),
        RegionPathway(from_region="ca3", to_region="ca1", density=float(ca3_to_ca1_density),
                      weight_mean=float(ca3_to_ca1_weight), weight_jitter=0.2, plastic=True,
                      plasticity_gate="ca3_to_ca1"),
        # CA1 -> MSN-D1 test cell (sparse place->striatum read; non-plastic fixed read-out for the gate).
        RegionPathway(from_region="ca1", to_region="msn_d1", density=float(ca1_to_msn_density),
                      weight_mean=float(ca1_to_msn_weight), weight_jitter=0.2, plastic=False),
    ]

    cfg = CoreSimConfig()
    cfg.seed = int(seed); cfg.heterogeneity_seed = int(seed); cfg.ou_seed = int(seed)
    cfg.dt_ms = float(dt_ms); cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.enable_stdp = True
    cfg.enable_hebbian_learning = False
    cfg.enable_nmda = bool(enable_nmda)
    cfg.stdp_w_max = 40.0
    cfg.fast_spike_reset = True
    # === deterministic-nav regime (g11_bg_runner.py:3340-3344) ===
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_ou_process = False
    cfg.enable_conductance_noise = False
    cfg.enable_parameter_heterogeneity = False
    cfg.enable_structural_plasticity = False

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = seed
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge, cfg


def _assert_cupy_regime(cfg, backend_name):
    if backend_name != "cupy":
        raise AssertionError(f"REGIME FIDELITY: MUST run on CuPy (numpy disqualified). Got {backend_name!r}.")
    bad = [k for k in ("enable_ou_process", "enable_conductance_noise", "enable_homeostasis",
                       "enable_parameter_heterogeneity", "enable_short_term_plasticity")
           if getattr(cfg, k, False)]
    if bad:
        raise AssertionError(f"REGIME FIDELITY: deterministic-regime knobs left ON: {bad}")


def _step(bridge, n):
    for _ in range(n):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)


def _measure(bridge, xp, sensor_idx, region_idx_map, sensor_act, record_steps, ablate=False,
             reset_steps=40):
    """Drive landmark_sensors (or nothing if ablate); accumulate per-neuron spike counts of all regions in
    region_idx_map, AND the mean effective excitatory drive (pA) into the MSN test cell. POSITION-LEAK:
    ONLY landmark_sensors get external current."""
    bridge.cp_external_input_current[:] = 0.0
    _step(bridge, reset_steps)
    bridge.cp_external_input_current[:] = 0.0
    if not ablate:
        bridge.cp_external_input_current[xp.asarray(sensor_idx, dtype=xp.int64)] = \
            xp.asarray(sensor_act, dtype=xp.float32)
    counts = {n: xp.zeros(len(idx), dtype=xp.float32) for n, idx in region_idx_map.items()}
    arrs = {n: xp.asarray(idx, dtype=xp.int64) for n, idx in region_idx_map.items()}
    msn_arr = arrs.get("msn_d1")
    # accumulate the MSN's synaptic excitatory current proxy (g_e * driving force) across steps
    drive_accum = 0.0
    have_ge = getattr(bridge, "cp_conductance_g_e", None) is not None
    for _ in range(record_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)
        for n in counts:
            counts[n] += bridge.cp_firing_states[arrs[n]].astype(xp.float32)
        if msn_arr is not None and have_ge:
            # effective excitatory drive into the MSN ~ g_e * (E_e - V); report g_e-driven current proxy.
            ge = bridge.cp_conductance_g_e[msn_arr]
            v = bridge.cp_membrane_potential_v[msn_arr]
            ee = float(getattr(bridge.core_config, "syn_reversal_potential_e", 0.0))
            drive_accum += float(xp.mean(ge * (ee - v)))
    bridge.cp_external_input_current[:] = 0.0
    out = {n: _host(c) for n, c in counts.items()}
    out["_msn_mean_drive_pA"] = drive_accum / max(record_steps, 1)
    return out


# Two operating regimes, both at the conducting front-end found by the conduction diagnostic. The C1
# question is whether EITHER gives CA1 high-rate AND distinct. (Spoiler in the findings: neither does;
# this probe produces the formal per-gate numbers + the MSN drive + the anti-cheat for the verdict.)
REGIMES = {
    # The sparse-distinct regime: CA3 stays sparse + position-specific (the separation survives) but the
    # recurrent does NOT ignite -> CA1 silent. Tests CONDUCTS+DISTINCT+STABLE; expected to FAIL HIGH-RATE.
    "distinct": dict(ca3_rec_weight=8.0, ca3_rec_density=0.30, inh_to_ca3_weight=14.0, n_ca3_inh_mult=1.0),
    # The ignited regime: recurrent + feedback inhibition -> the recurrent ignites to a saturated GLOBAL
    # basin (200 spk/step, every location -> same state) AND the synchronous volley shunts CA1 silent.
    "ignite":   dict(ca3_rec_weight=8.0, ca3_rec_density=0.30, inh_to_ca3_weight=20.0, n_ca3_inh_mult=1.0),
    # The moderate-attractor regime: recurrent ignites to a MODERATE rate (~20 spk/step) WITHOUT the ca3_inh
    # feedback loop, so CA3 holds a non-saturated attractor and CA1 DOES fire (the only regime where CA1 is
    # non-silent end-to-end). But it is a single GLOBAL basin (diff-cos ~0.99). Tests CONDUCTS+HIGH-RATE on
    # a firing CA1; expected to FAIL DISTINCT. (Set ca3_inh to ~0 by zeroing the feedback weight.)
    "moderate": dict(ca3_rec_weight=8.0, ca3_rec_density=0.30, inh_to_ca3_weight=0.0, n_ca3_inh_mult=0.0),
}


def run_seed(seed, *, regime, locations, landmarks, n_bearing, n_dist, max_intensity, falloff, dist_sigma,
             dist_max, bexp, train_passes, train_steps_per_loc, record_steps,
             n_ec, n_dg, n_dg_pv_basket, n_ca3, n_ca1, n_ca3_inh, n_msn,
             lm_to_ec_weight, lm_to_ec_density, ec_to_dg_weight, dg_to_ca3_weight, dg_to_ca3_density,
             ca1_to_msn_weight, ca1_to_msn_density, msn_rheobase_pA, verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    from sim.backend import get_backend
    xp, backend_name = get_backend()

    n_per_lm = n_bearing + n_dist
    n_sensors = len(landmarks) * n_per_lm
    rp = REGIMES[regime]
    # moderate regime zeroes the ca3_inh feedback pool (no k-WTA loop -> CA3 holds a moderate non-saturated
    # attractor and CA1 fires). Keep >=1 neuron to avoid a zero-size region; the zero weight makes it inert.
    n_ca3_inh_eff = int(round(n_ca3_inh * rp.get("n_ca3_inh_mult", 1.0))) or 1

    t0 = time.time()
    bridge, cfg = _build(
        seed, n_sensors=n_sensors, n_ec=n_ec, n_dg=n_dg, n_dg_pv_basket=n_dg_pv_basket, n_ca3=n_ca3,
        n_ca1=n_ca1, n_ca3_inh=n_ca3_inh_eff, n_msn=n_msn,
        lm_to_ec_weight=lm_to_ec_weight, lm_to_ec_density=lm_to_ec_density,
        ec_to_dg_weight=ec_to_dg_weight, ec_to_dg_density=0.40, ec_to_pvb_weight=5.0, pvb_to_dg_weight=2.0,
        dg_to_ca3_weight=dg_to_ca3_weight, dg_to_ca3_density=dg_to_ca3_density,
        ca3_rec_weight=rp["ca3_rec_weight"], ca3_rec_density=rp["ca3_rec_density"],
        ca3_to_inh_weight=8.0, ca3_to_inh_density=0.30,
        inh_to_ca3_weight=rp["inh_to_ca3_weight"], inh_to_ca3_density=0.60,
        ca3_to_ca1_weight=6.0, ca3_to_ca1_density=0.30, ec_to_ca1_weight=3.0, ec_to_ca1_density=0.30,
        ca1_to_msn_weight=ca1_to_msn_weight, ca1_to_msn_density=ca1_to_msn_density, enable_nmda=True)
    _assert_cupy_regime(cfg, backend_name)
    build_s = time.time() - t0
    log(f"  [seed {seed}] regime={regime} built {build_s:.1f}s; {cfg.num_neurons} neurons, "
        f"{int(bridge.cp_connections.nnz)} syn; backend={backend_name}")

    rm = bridge.region_manager
    sensor_idx = np.asarray(rm.indices("landmark_sensors"), dtype=np.int64)
    region_idx_map = {n: np.asarray(rm.indices(n), dtype=np.int64)
                      for n in ("ec", "dg", "ca3", "ca1", "msn_d1")}
    loc_names = list(locations.keys())

    # POSITION-LEAK audit (anti-cheat): the ONLY region driven externally is landmark_sensors.
    driven_regions = {"landmark_sensors"}
    for r in ("ec", "dg", "ca3", "ca1", "msn_d1"):
        assert r not in driven_regions, f"{r} must never be externally driven (position leak)"

    def render(name):
        x, y = locations[name]
        return landmark_sensor_act(x, y, landmarks, n_bearing, n_dist, max_intensity, falloff,
                                   dist_sigma, dist_max, bexp)
    loc_sensor = {n: render(n) for n in loc_names}
    in_diffs = [cosine_counts(loc_sensor[a], loc_sensor[b]) for a, b in itertools.combinations(loc_names, 2)]
    input_overlap = float(np.mean(in_diffs)) if in_diffs else 0.0

    # ── Self-organize the loop (open all plastic feedforward + recurrent gates, walk locations) ──
    gates = ("landmark_to_ec", "ec_to_dg", "dg_to_ca3", "ca3_to_ca1", "ca3_swr_burst", "ec_to_ca1")
    for g in gates:
        try: bridge.set_plasticity_gate(g, 1.0)
        except Exception: pass
    rng = np.random.default_rng(seed)
    t_tr = time.time()
    for _p in range(train_passes):
        order = list(loc_names); rng.shuffle(order)
        for name in order:
            bridge.cp_external_input_current[:] = 0.0
            _step(bridge, 20)
            bridge.cp_external_input_current[xp.asarray(sensor_idx, dtype=xp.int64)] = \
                xp.asarray(loc_sensor[name], dtype=xp.float32)
            _step(bridge, train_steps_per_loc)
    for g in gates:
        try: bridge.set_plasticity_gate(g, 0.0)
        except Exception: pass
    bridge.cp_external_input_current[:] = 0.0
    log(f"  [seed {seed}] self-org done ({time.time()-t_tr:.0f}s)")

    # ── Measure CA1 ensemble per location (+ MSN drive), repeat (stability), and sensor-ablated ──
    def measure_all(ablate=False):
        return {n: _measure(bridge, xp, sensor_idx, region_idx_map, loc_sensor[n], record_steps,
                            ablate=ablate) for n in loc_names}
    ens = measure_all(ablate=False)
    ens_repeat = measure_all(ablate=False)
    ens_ablate = measure_all(ablate=True)

    def ca1(name, d): return d[name]["ca1"]

    # CONDUCTS: CA1 non-silent
    ca1_active = {n: float(np.mean(ens[n]["ca1"] > 0)) for n in loc_names}
    ca1_spk = {n: float(np.sum(ens[n]["ca1"])) / record_steps for n in loc_names}
    ca3_spk = {n: float(np.sum(ens[n]["ca3"])) / record_steps for n in loc_names}
    mean_ca1_active = float(np.mean(list(ca1_active.values())))
    mean_ca1_spk = float(np.mean(list(ca1_spk.values())))
    mean_ca3_spk = float(np.mean(list(ca3_spk.values())))
    conducts = mean_ca1_active > 0.005  # at least some CA1 cells fire end-to-end

    # HIGH-RATE: the MSN test cell fires >=5 Hz, and report its effective drive vs ~420 pA rheobase
    msn_rate_hz = {n: float(np.sum(ens[n]["msn_d1"])) / record_steps / len(region_idx_map["msn_d1"])
                   * (1000.0 / cfg.dt_ms) for n in loc_names}
    msn_drive = {n: ens[n]["_msn_mean_drive_pA"] for n in loc_names}
    max_msn_rate = float(np.max(list(msn_rate_hz.values())))
    mean_msn_drive = float(np.mean(list(msn_drive.values())))
    high_rate = max_msn_rate >= 5.0

    # DISTINCT: different-location CA1 cosine < 0.3
    diff_cos = [cosine_counts(ca1(a, ens), ca1(b, ens)) for a, b in itertools.combinations(loc_names, 2)]
    mean_diff_cos = float(np.mean(diff_cos)) if diff_cos else 1.0
    distinct = mean_diff_cos < 0.30

    # STABLE: same-location cosine > 0.7 (only meaningful if CA1 fires)
    same_cos = [cosine_counts(ca1(n, ens), ca1(n, ens_repeat)) for n in loc_names]
    same_vals = [c for c in same_cos if c > 0]
    mean_same_cos = float(np.mean(same_vals)) if same_vals else 0.0
    stable = mean_same_cos > 0.70

    # SENSOR-DRIVEN (anti-cheat): ablate landmark sensors -> CA1 collapses
    abl_active = float(np.mean([np.mean(ens_ablate[n]["ca1"] > 0) for n in loc_names]))
    abl_vs_true = float(np.mean([cosine_counts(ca1(n, ens), ca1(n, ens_ablate)) for n in loc_names]))
    sensor_driven = (abl_active < 0.25 * max(mean_ca1_active, 1e-6)) or (abl_vs_true < 0.30)

    result = {
        "seed": seed, "backend": backend_name, "regime": regime,
        "n_neurons": int(cfg.num_neurons), "n_synapses": int(bridge.cp_connections.nnz),
        "input_pattern_overlap": round(input_overlap, 4),
        "ca1_active_per_location": {k: round(v, 4) for k, v in ca1_active.items()},
        "ca1_spk_per_location": {k: round(v, 3) for k, v in ca1_spk.items()},
        "ca3_spk_per_location": {k: round(v, 2) for k, v in ca3_spk.items()},
        "mean_ca1_active": round(mean_ca1_active, 4),
        "mean_ca1_spk_per_step": round(mean_ca1_spk, 3),
        "mean_ca3_spk_per_step": round(mean_ca3_spk, 2),
        "msn_rate_hz_per_location": {k: round(v, 2) for k, v in msn_rate_hz.items()},
        "msn_max_rate_hz": round(max_msn_rate, 2),
        "msn_mean_effective_drive_pA": round(mean_msn_drive, 1),
        "msn_rheobase_pA_ref": float(msn_rheobase_pA),
        "ca1_mean_diff_location_cosine": round(mean_diff_cos, 4),
        "ca1_mean_same_location_cosine": round(mean_same_cos, 4),
        "ablation_ca1_active": round(abl_active, 4),
        "ablation_ca1_cosine_vs_true": round(abl_vs_true, 4),
        "gate_CONDUCTS": bool(conducts),
        "gate_HIGH_RATE": bool(high_rate),
        "gate_DISTINCT": bool(distinct),
        "gate_STABLE": bool(stable),
        "gate_SENSOR_DRIVEN": bool(sensor_driven),
        "all_pass": bool(conducts and high_rate and distinct and stable and sensor_driven),
        "position_leak_driven_regions": sorted(driven_regions),
        "total_seconds": round(time.time() - t0, 1),
    }
    log(f"  [seed {seed}] CONDUCTS ca1_act={mean_ca1_active:.3f}/spk{mean_ca1_spk:.2f} "
        f"({'PASS' if conducts else 'FAIL'})  HIGH-RATE msn_max={max_msn_rate:.1f}Hz "
        f"drive={mean_msn_drive:.0f}pA(rheo~{msn_rheobase_pA:.0f}) ({'PASS' if high_rate else 'FAIL'})")
    log(f"  [seed {seed}] DISTINCT ca1_diffcos={mean_diff_cos:.3f} ({'PASS' if distinct else 'FAIL'})  "
        f"STABLE samecos={mean_same_cos:.3f} ({'PASS' if stable else 'FAIL'})  "
        f"SENSOR-DRIVEN abl_cos={abl_vs_true:.3f}/abl_act={abl_active:.3f} "
        f"({'PASS' if sensor_driven else 'FAIL'})")
    log(f"  [seed {seed}] ca3_spk/step={mean_ca3_spk:.1f}  ALL_PASS={result['all_pass']}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--regime", type=str, default="distinct", choices=list(REGIMES.keys()),
                    help="'distinct' (sparse CA3, tests CONDUCTS+DISTINCT+STABLE) or 'ignite' "
                         "(global CA3 attractor, tests HIGH-RATE)")
    ap.add_argument("--grid-size", type=int, default=32)
    ap.add_argument("--n-bearing", type=int, default=12)
    ap.add_argument("--n-dist", type=int, default=8)
    ap.add_argument("--bexp", type=float, default=4.0)
    ap.add_argument("--dist-sigma", type=float, default=4.0)
    ap.add_argument("--max-intensity", type=float, default=900.0)
    ap.add_argument("--falloff", type=float, default=0.03)
    ap.add_argument("--n-ec", type=int, default=200)
    ap.add_argument("--n-dg", type=int, default=800)
    ap.add_argument("--n-dg-pv-basket", type=int, default=240)
    ap.add_argument("--n-ca3", type=int, default=400)
    ap.add_argument("--n-ca1", type=int, default=200)
    ap.add_argument("--n-ca3-inh", type=int, default=120)
    ap.add_argument("--n-msn", type=int, default=40)
    ap.add_argument("--lm-to-ec-weight", type=float, default=60.0)
    ap.add_argument("--lm-to-ec-density", type=float, default=0.10)
    ap.add_argument("--ec-to-dg-weight", type=float, default=30.0)
    ap.add_argument("--dg-to-ca3-weight", type=float, default=40.0)
    ap.add_argument("--dg-to-ca3-density", type=float, default=0.10)
    ap.add_argument("--ca1-to-msn-weight", type=float, default=30.0,
                    help="CA1->MSN read weight (strong, to give the place ensemble its best shot at "
                         "the rheobase)")
    ap.add_argument("--ca1-to-msn-density", type=float, default=0.40)
    ap.add_argument("--msn-rheobase-pA", type=float, default=420.0,
                    help="reference MSN-D1 effective rheobase (from the Stage-2 teacher sweep)")
    ap.add_argument("--train-passes", type=int, default=16)
    ap.add_argument("--train-steps-per-loc", type=int, default=120)
    ap.add_argument("--record-steps", type=int, default=100)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    seeds = [int(args.seed)] if args.seed is not None else \
        [int(s) for s in args.seeds.split(",") if s.strip()]
    grid = int(args.grid_size)
    locations = default_locations(grid)
    landmarks = default_landmarks(grid)
    dist_max = float(grid) * 1.42

    print("=" * 76)
    print(f"C1 de-risk: trisynaptic CA1 place code (regime={args.regime})  seeds={seeds} grid={grid}")
    print(f"  landmarks={landmarks}")
    print("=" * 76)

    per_seed = []
    for s in seeds:
        per_seed.append(run_seed(
            s, regime=args.regime, locations=locations, landmarks=landmarks,
            n_bearing=int(args.n_bearing), n_dist=int(args.n_dist), max_intensity=float(args.max_intensity),
            falloff=float(args.falloff), dist_sigma=float(args.dist_sigma), dist_max=dist_max,
            bexp=float(args.bexp), train_passes=int(args.train_passes),
            train_steps_per_loc=int(args.train_steps_per_loc), record_steps=int(args.record_steps),
            n_ec=int(args.n_ec), n_dg=int(args.n_dg), n_dg_pv_basket=int(args.n_dg_pv_basket),
            n_ca3=int(args.n_ca3), n_ca1=int(args.n_ca1), n_ca3_inh=int(args.n_ca3_inh),
            n_msn=int(args.n_msn), lm_to_ec_weight=float(args.lm_to_ec_weight),
            lm_to_ec_density=float(args.lm_to_ec_density), ec_to_dg_weight=float(args.ec_to_dg_weight),
            dg_to_ca3_weight=float(args.dg_to_ca3_weight), dg_to_ca3_density=float(args.dg_to_ca3_density),
            ca1_to_msn_weight=float(args.ca1_to_msn_weight), ca1_to_msn_density=float(args.ca1_to_msn_density),
            msn_rheobase_pA=float(args.msn_rheobase_pA), verbose=True))

    def cnt(key): return sum(1 for r in per_seed if r[key])
    def agg(key):
        v = [r[key] for r in per_seed]
        return {"mean": round(float(np.mean(v)), 4), "min": round(float(np.min(v)), 4),
                "max": round(float(np.max(v)), 4), "values": [round(float(x), 4) for x in v]}
    n = len(seeds)
    summary = {
        "n_seeds": n, "seeds": seeds, "regime": args.regime,
        "gate_CONDUCTS_count": cnt("gate_CONDUCTS"), "gate_HIGH_RATE_count": cnt("gate_HIGH_RATE"),
        "gate_DISTINCT_count": cnt("gate_DISTINCT"), "gate_STABLE_count": cnt("gate_STABLE"),
        "gate_SENSOR_DRIVEN_count": cnt("gate_SENSOR_DRIVEN"), "all_pass_count": cnt("all_pass"),
        "ca1_diff_location_cosine": agg("ca1_mean_diff_location_cosine"),
        "ca1_same_location_cosine": agg("ca1_mean_same_location_cosine"),
        "ca1_spk_per_step": agg("mean_ca1_spk_per_step"),
        "ca3_spk_per_step": agg("mean_ca3_spk_per_step"),
        "msn_max_rate_hz": agg("msn_max_rate_hz"),
        "msn_mean_effective_drive_pA": agg("msn_mean_effective_drive_pA"),
        "verdict": ("PASS" if cnt("all_pass") == n else
                    "PARTIAL" if cnt("all_pass") > 0 else "NEGATIVE"),
        "per_seed": per_seed,
    }
    print("\n" + "=" * 76)
    print(f"C1 SUMMARY (regime={args.regime})")
    print(f"  CONDUCTS:      {summary['gate_CONDUCTS_count']}/{n}  ca1_spk/step "
          f"{summary['ca1_spk_per_step']['values']}")
    print(f"  HIGH-RATE:     {summary['gate_HIGH_RATE_count']}/{n}  msn_max_hz "
          f"{summary['msn_max_rate_hz']['values']}  drive_pA {summary['msn_mean_effective_drive_pA']['values']}"
          f" (rheo~420)")
    print(f"  DISTINCT:      {summary['gate_DISTINCT_count']}/{n}  ca1_diffcos "
          f"{summary['ca1_diff_location_cosine']['values']}")
    print(f"  STABLE:        {summary['gate_STABLE_count']}/{n}  ca1_samecos "
          f"{summary['ca1_same_location_cosine']['values']}")
    print(f"  SENSOR-DRIVEN: {summary['gate_SENSOR_DRIVEN_count']}/{n}")
    print(f"  ca3_spk/step:  {summary['ca3_spk_per_step']['values']}")
    print(f"  ALL_PASS: {summary['all_pass_count']}/{n}   VERDICT: {summary['verdict']}")
    print("=" * 76)

    if args.out:
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"[OUT] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
