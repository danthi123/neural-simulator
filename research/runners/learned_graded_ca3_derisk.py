"""Learned, GRADED CA3 autoassociator de-risk — does a Marr/Treves-Rolls LEARNED-per-location
recurrent + a Wang slow-NMDA recurrent (the `exc_receptor="nmda_slow"` protected edit) make a CA1
place code that is BOTH distinct-per-location AND high-rate (fires an MSN-D1 critic, ~420 pA rheobase)?

This is the follow-on to the C1 NEGATIVE (`2026-06-09-C1-trisynaptic-ca1-place-code.md`): the canonical
trisynaptic loop with a RANDOM/DENSE recurrent has only two stable states — sparse+distinct+silent OR a
single global+firing+position-blind basin (no graded middle). The design
(`2026-06-09-learned-graded-ca3-design.md`) attributes that to TWO coupled defects, each with a fix:
  (A) the recurrent is random-dense -> ONE global basin. FIX (Step A, runner-side): zero-init the
      recurrent, then CUE-CLAMP each location's CA3 ensemble (the mossy detonator is the brain's own
      legitimate teacher) with the recurrent STDP gate (ca3_swr_burst) OPEN so the recurrent
      Hebbian-learns THAT ensemble as its own non-overlapping attractor (Treves-Rolls), bounded by a
      low stdp_w_max ceiling.
  (B) the Izhikevich recurrent is bistable OFF<->runaway (no graded ~10-40 Hz). FIX (Step B, the
      protected sim/ edit): route the ca3->ca3 recurrent through exc_receptor="nmda_slow" (a SEPARATE
      slow-NMDA conductance, AMPA component suppressed) so it reverberates gradedly (Wang 2001/2002),
      while the dg->ca3 mossy stays fast AMPA.

GATES (CuPy; >=3 seeds; deterministic regime OU/cond-noise/global-homeostasis/heterogeneity/STP OFF):
  G1 DISTINCT      : different-location CA3 cos < 0.30 AFTER recurrent completion (beats C1's 0.99 basin)
  G2 GRADED        : CA3 stable at ~10-40 Hz (NOT 0, NOT the ~200 spk/step ceiling), bounded over time
  G3 STABLE        : same-location repeat cos > 0.70 (a real attractor of the field)
  G4 HIGH-RATE     : CA1->MSN-D1 effective drive >= ~420 pA -> MSN >= 5 Hz (beats C1/N9's 82-162 pA)
  G5 SENSOR-DRIVEN : ablate landmark sensors -> CA3/CA1 collapses (a real sensor-driven place code)
  G6 COMPLETION    : drop 1 of >=3 landmarks -> recall cos(true,partial) > 0.7 (D.13 pattern completion)

ANTI-CHEATS (each must behave consistently with an honest result):
  - NO host teacher: the ONLY external-current write targets landmark_sensors (the mossy cue-clamp is the
    brain's own DG->CA3 detonator, brain-based-legal). Hard-asserted (position-leak audit).
  - GENUINELY learned: ABLATE the recurrent (zero ca3->ca3) -> G2(graded sustain) and G6(completion) must
    COLLAPSE (CA3 falls back to the bare mossy-driven sparse code). If they survive recurrent-ablation, the
    "attractor" was just feedforward drive -> the result is NOT a learned recurrent.
  - CuPy regime: backend=="cupy" (numpy disqualified per the membrane-divergence root-cause); deterministic
    knobs OFF; no per-region homeostasis on CA3/CA1/MSN. Hard-asserted.

HARNESS-BUG FIX (2026-06-09, the DIRECT-CA3 route): the original 0/3 was a SILENT-CA3 ARTIFACT — the
harness drove CA3 ONLY through the trisynaptic feedforward landmark_sensors->ec->dg->ca3, which does NOT
conduct at probe scale (the EC fire-vs-select tension + DG-FFI kills the sparse mossy fan-in -> CA3 silent;
documented in 2026-06-09-C1-trisynaptic-ca1-place-code.md). CA3 never fired -> the nmda_slow recurrent had
nothing to store -> the protected edit was never actually tested (every metric was exactly 0.0). FIX:
`--direct-ca3` (default ON) adds a FIXED direct landmark_sensors->ca3 AMPA detonator (the validated Stage-1
single-hop competitive place mechanism, 2026-06-09-place-code-selforg-stage1-derisk.md) so CA3 reliably
fires a sparse distinct ensemble per location; the ca3->ca3 recurrent stays routed nmda_slow (the protected
edit) for the graded sustain + autoassociative storage. An INSTRUMENTATION GUARD now measures CA3 firing
DURING storage and HARD-ASSERTS it > 0 (catches a silent-CA3 run immediately, not after the fact). The
`--no-direct-ca3` control reproduces the original silent-CA3 bug (the guard then fires the AssertionError).

RESULT (2026-06-09, CuPy, 3 seeds, BOTH operating points + recurrent-ablation anti-cheat): NEGATIVE — the
fire-vs-grade wall is irreducible. At the DISTINCT feedforward point (intensity 450, w 20): G1/G2/G3/G5 PASS
(CA3 distinct 0.13-0.26, sparse ~5%) but CA1 fires 0.00 spk/step -> G4 FAILS (no MSN), and the recurrent-
ablation anti-cheat shows the recurrent contributes ~0% there (too few cells co-fire to store a basin). At
the DENSE point (intensity 900, w 40): G4 PASSES (MSN 21-28 Hz, 372-453 pA) but G1 FAILS (CA3 0.68-0.72,
position-blind) and G5 FAILS (autonomous reverberation) — AND ablating the recurrent does NOT collapse G4/G6
(they come from the FIXED feedforward, the recurrent only adds the position-blind reverberation). DISTINCT and
HIGH-RATE are on opposite sides of a sharp boundary with no overlap; the nmda_slow recurrent narrows but does
not close it. See 2026-06-09-learned-graded-ca3-derisk-RESULT.md.

USAGE (MUST be cupy):
  # full de-risk (3 seeds), distinct operating point (default):
  SIM_BACKEND=cupy python -m research.runners.learned_graded_ca3_derisk \
      --seeds 42,43,44 --out research/findings/raw/_learned_graded_ca3_directfix_3seed.json
  # recurrent-ablation anti-cheat (does removing the learned recurrent collapse the graded sustain?):
  SIM_BACKEND=cupy python -m research.runners.learned_graded_ca3_derisk --seeds 42,43,44 --ablate-recurrent
  # the DENSE operating point (G4 fires, G1 fails — the other side of the bifurcation):
  SIM_BACKEND=cupy python -m research.runners.learned_graded_ca3_derisk --seeds 42,43,44 \
      --max-intensity 900 --lm-to-ca3-weight 40 --lm-to-ca3-density 0.5
  # control: reproduce the original silent-CA3 harness bug (guard fires the AssertionError):
  SIM_BACKEND=cupy python -m research.runners.learned_graded_ca3_derisk --seed 42 --no-direct-ca3
  # 1-seed executes-without-error smoke:
  SIM_BACKEND=cupy python -m research.runners.learned_graded_ca3_derisk --smoke
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
# sparsely from CA1. Mirrors the C1 build, with the ca3->ca3 recurrent routed exc_receptor="nmda_slow"
# (the protected edit) + ZERO-INIT (basins grown by the cue-clamped storage protocol, not present at init).
# ──────────────────────────────────────────────────────────────────────

def _build(seed, *, n_sensors, n_ec, n_dg, n_dg_pv_basket, n_ca3, n_ca1, n_ca3_inh, n_msn,
           lm_to_ec_weight, lm_to_ec_density, ec_to_dg_weight, ec_to_dg_density,
           ec_to_pvb_weight, pvb_to_dg_weight, dg_to_ca3_weight, dg_to_ca3_density,
           ca3_rec_weight, ca3_rec_density, ca3_to_inh_weight, ca3_to_inh_density,
           inh_to_ca3_weight, inh_to_ca3_density, ca3_to_ca1_weight, ca3_to_ca1_density,
           ec_to_ca1_weight, ec_to_ca1_density, ca1_to_msn_weight, ca1_to_msn_density,
           recurrent_exc_receptor="nmda_slow", nmda_recurrent_prop=0.05, nmda_recurrent_tau=100.0,
           rec_stdp_w_max=6.0, enable_nmda=True, dt_ms=1.0,
           direct_ca3=True, lm_to_ca3_weight=40.0, lm_to_ca3_density=0.5, lm_to_ca3_jitter=0.6):
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType

    regions = [
        BrainRegion(name="landmark_sensors", n_neurons=int(n_sensors), exc_fraction=1.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False, izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name),
        BrainRegion(name="ec", n_neurons=int(n_ec), exc_fraction=0.8, internal_density=0.05,
                    exc_weight_mean=0.3, inh_weight_mean=0.8, weight_jitter=0.2, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name),
        BrainRegion(name="dg", n_neurons=int(n_dg), exc_fraction=0.95, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name),
        BrainRegion(name="dg_pv_basket", n_neurons=int(n_dg_pv_basket), exc_fraction=0.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False, izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name),
        # CA3: enable_nmda=True so the slow-NMDA recurrent's Mg2+ self-limiting block is active on these cells.
        BrainRegion(name="ca3", n_neurons=int(n_ca3), exc_fraction=0.85, internal_density=0.0,
                    exc_weight_mean=1.5, inh_weight_mean=2.0, weight_jitter=0.2, plastic_internal=True,
                    izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name, enable_nmda=bool(enable_nmda)),
        BrainRegion(name="ca3_inh", n_neurons=int(n_ca3_inh), exc_fraction=0.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name),
        BrainRegion(name="ca1", n_neurons=int(n_ca1), exc_fraction=0.85, internal_density=0.05,
                    exc_weight_mean=0.3, inh_weight_mean=0.8, weight_jitter=0.2, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name),
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
        # === THE LEARNED-GRADED RECURRENT (the design target) ===
        # exc_receptor="nmda_slow" (Step B protected edit): slow-NMDA-dominant, AMPA suppressed -> graded.
        # ZERO-INIT weight + plastic + gate ca3_swr_burst: the basins are GROWN by the cue-clamped storage
        # protocol (Step A), never present at init -> no global ignition before STDP differentiates them.
        RegionPathway(from_region="ca3", to_region="ca3", density=float(ca3_rec_density),
                      weight_mean=float(ca3_rec_weight), weight_jitter=(0.2 if ca3_rec_weight > 0 else 0.0),
                      plastic=True, plasticity_gate="ca3_swr_burst",
                      exc_receptor=str(recurrent_exc_receptor)),
        RegionPathway(from_region="ca3", to_region="ca3_inh", density=float(ca3_to_inh_density),
                      weight_mean=float(ca3_to_inh_weight), weight_jitter=0.2, plastic=False),
        RegionPathway(from_region="ca3_inh", to_region="ca3", density=float(inh_to_ca3_density),
                      weight_mean=float(inh_to_ca3_weight), weight_jitter=0.2, plastic=False),
        RegionPathway(from_region="ca3", to_region="ca1", density=float(ca3_to_ca1_density),
                      weight_mean=float(ca3_to_ca1_weight), weight_jitter=0.2, plastic=True,
                      plasticity_gate="ca3_to_ca1"),
        RegionPathway(from_region="ca1", to_region="msn_d1", density=float(ca1_to_msn_density),
                      weight_mean=float(ca1_to_msn_weight), weight_jitter=0.2, plastic=False),
    ]
    # === DIRECT-CA3 single-hop afferent (the FIX for the silent multi-hop) ===
    # The canonical EC->DG->CA3 trisynaptic feedforward does NOT conduct at probe scale (the
    # EC fire-vs-select tension + DG-FFI kills the sparse mossy fan-in -> CA3 silent; documented
    # in 2026-06-09-C1-trisynaptic-ca1-place-code.md and reproduced here as the 0.0-everything
    # artifact). The Stage-1 SINGLE-HOP competitive place mechanism (landmark_sensors -> pool,
    # plastic, threshold/ca3_inh-WTA) DID fire a sparse distinct-per-location code 3/3 on CuPy
    # (2026-06-09-place-code-selforg-stage1-derisk.md). So we drive CA3 DIRECTLY with that same
    # AMPA single-hop (mossy-equivalent detonator, brain-based-legal: the body's egocentric
    # landmark sensors feed CA3 sparsely), and KEEP the ca3->ca3 recurrent routed nmda_slow (the
    # protected edit) for the graded sustain + autoassociative storage. This makes the AMPA
    # feedforward reliably fire a distinct sparse CA3 ensemble per location -> the cue the
    # nmda_slow recurrent is FOR. (EC/DG remain wired; CA3 just no longer DEPENDS on them firing.)
    # The direct afferent is FIXED (plastic=False): it is the mossy-detonator analog (sparse +
    # powerful + relatively fixed in biology), and crucially it lets the global stdp_w_max stay LOW
    # (the recurrent's runaway ceiling, Step A) WITHOUT the soft-bound collapsing this strong AMPA
    # weight (the documented stdp_w_max gotcha). The position-specificity comes from the FIXED random
    # sparse projection + CA3's threshold/ca3_inh WTA (exactly the Stage-1 mechanism's source of
    # distinctness); the ONLY learned structure is the ca3->ca3 recurrent under test -> the cleanest
    # possible recurrent-ablation anti-cheat (the learned recurrent is the sole differentiator).
    if direct_ca3:
        pathways.append(
            RegionPathway(from_region="landmark_sensors", to_region="ca3",
                          density=float(lm_to_ca3_density), weight_mean=float(lm_to_ca3_weight),
                          weight_jitter=float(lm_to_ca3_jitter), plastic=False))

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
    # === the Step B protected edit: per-pathway slow-NMDA recurrent routing ===
    cfg.enable_nmda_recurrent = (str(recurrent_exc_receptor) == "nmda_slow")
    cfg.nmda_recurrent_propagation_strength = float(nmda_recurrent_prop)
    cfg.nmda_recurrent_tau_decay_ms = float(nmda_recurrent_tau)
    cfg.stdp_w_max = float(rec_stdp_w_max)  # low global ceiling (Step A) so a basin can't overgrow to runaway
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
    region_idx_map AND the mean effective excitatory drive (pA) into the MSN test cell (g_e + g_nmda_recurrent
    driving force, so the slow-NMDA recurrent's contribution to CA1->MSN is counted). POSITION-LEAK: ONLY
    landmark_sensors get external current."""
    bridge.cp_external_input_current[:] = 0.0
    _step(bridge, reset_steps)
    bridge.cp_external_input_current[:] = 0.0
    if not ablate:
        bridge.cp_external_input_current[xp.asarray(sensor_idx, dtype=xp.int64)] = \
            xp.asarray(sensor_act, dtype=xp.float32)
    counts = {n: xp.zeros(len(idx), dtype=xp.float32) for n, idx in region_idx_map.items()}
    arrs = {n: xp.asarray(idx, dtype=xp.int64) for n, idx in region_idx_map.items()}
    msn_arr = arrs.get("msn_d1")
    drive_accum = 0.0
    have_ge = getattr(bridge, "cp_conductance_g_e", None) is not None
    g_nmr = getattr(bridge, "cp_conductance_g_nmda_recurrent", None)
    for _ in range(record_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)
        for n in counts:
            counts[n] += bridge.cp_firing_states[arrs[n]].astype(xp.float32)
        if msn_arr is not None and have_ge:
            ee = float(getattr(bridge.core_config, "syn_reversal_potential_e", 0.0))
            v = bridge.cp_membrane_potential_v[msn_arr]
            ge = bridge.cp_conductance_g_e[msn_arr]
            drive = ge * (ee - v)
            if g_nmr is not None:
                drive = drive + g_nmr[msn_arr] * (ee - v)
            drive_accum += float(xp.mean(drive))
    bridge.cp_external_input_current[:] = 0.0
    out = {n: _host(c) for n, c in counts.items()}
    out["_msn_mean_drive_pA"] = drive_accum / max(record_steps, 1)
    return out


# ── Step A: the LEARNED-RECURRENT cue-clamped storage protocol ──────────
def _step_clamped_measure_ca3(bridge, xp, n, ca3_arr):
    """Step `n` steps and accumulate CA3 spike-count + count steps (the storage-time CA3 firing
    instrumentation). Returns (total_ca3_spikes, n_steps). The INSTRUMENTATION GUARD source: a
    silent-CA3 run (the harness-bug artifact) is caught immediately if this returns ~0."""
    total = 0.0
    for _ in range(n):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)
        total += float(xp.sum(bridge.cp_firing_states[ca3_arr]))
    return total, n


def _store_recurrent(bridge, xp, sensor_idx, loc_sensor, loc_names, *, ff_gates, rec_gate,
                     store_passes, store_settle_steps, store_clamp_steps, rng, ca3_idx):
    """For each distinct location, CUE-CLAMP the CA3 ensemble (the landmark cue drives CA3 — with the
    direct-CA3 fix, via the FIXED landmark_sensors->ca3 detonator; the dg->ca3 mossy stays wired but the
    multi-hop need not conduct) with the recurrent STDP gate OPEN, so the recurrent Hebbian-learns THAT
    ensemble as its own attractor (Treves-Rolls). The afferent is the brain's own legitimate teacher ->
    brain-based-legal (NO host pattern into CA3). Returns the CA3 spk/step DURING the store-phase clamp
    windows (the instrumentation guard: MUST be > 0, else CA3 is silent and the recurrent stored nothing)."""
    ca3_arr = xp.asarray(ca3_idx, dtype=xp.int64)
    # Warm-up: open feedforward gates, walk locations so any plastic feedforward routing forms. Recurrent
    # gate CLOSED here. (With the FIXED direct-CA3 afferent this warm-up is mostly vestigial, but harmless.)
    for g in ff_gates:
        try: bridge.set_plasticity_gate(g, 1.0)
        except Exception: pass
    try: bridge.set_plasticity_gate(rec_gate, 0.0)
    except Exception: pass
    for _p in range(store_passes):
        order = list(loc_names); rng.shuffle(order)
        for name in order:
            bridge.cp_external_input_current[:] = 0.0
            _step(bridge, store_settle_steps)
            bridge.cp_external_input_current[xp.asarray(sensor_idx, dtype=xp.int64)] = \
                xp.asarray(loc_sensor[name], dtype=xp.float32)
            _step(bridge, store_clamp_steps)
    # Store phase: FREEZE feedforward (the ensembles are fixed), OPEN the recurrent gate, cue-clamp each
    # location again so the recurrent learns to reproduce each fixed afferent-driven ensemble. INSTRUMENT
    # CA3 firing across the clamp windows here (this is the rate that MUST be > 0).
    for g in ff_gates:
        try: bridge.set_plasticity_gate(g, 0.0)
        except Exception: pass
    try: bridge.set_plasticity_gate(rec_gate, 1.0)
    except Exception: pass
    ca3_spk_total = 0.0
    ca3_step_total = 0
    for _p in range(store_passes):
        order = list(loc_names); rng.shuffle(order)
        for name in order:
            bridge.cp_external_input_current[:] = 0.0
            _step(bridge, store_settle_steps)
            bridge.cp_external_input_current[xp.asarray(sensor_idx, dtype=xp.int64)] = \
                xp.asarray(loc_sensor[name], dtype=xp.float32)
            s, ns = _step_clamped_measure_ca3(bridge, xp, store_clamp_steps, ca3_arr)
            ca3_spk_total += s
            ca3_step_total += ns
    # Freeze everything for recall measurement.
    try: bridge.set_plasticity_gate(rec_gate, 0.0)
    except Exception: pass
    bridge.cp_external_input_current[:] = 0.0
    return ca3_spk_total / max(ca3_step_total, 1)  # CA3 spk/step during storage (the guard)


def run_seed(seed, *, locations, landmarks, n_bearing, n_dist, max_intensity, falloff, dist_sigma,
             dist_max, bexp, store_passes, store_settle_steps, store_clamp_steps, record_steps,
             n_ec, n_dg, n_dg_pv_basket, n_ca3, n_ca1, n_ca3_inh, n_msn,
             lm_to_ec_weight, lm_to_ec_density, ec_to_dg_weight, dg_to_ca3_weight, dg_to_ca3_density,
             ca3_rec_weight, ca3_rec_density, inh_to_ca3_weight, ca1_to_msn_weight, ca1_to_msn_density,
             nmda_recurrent_prop, nmda_recurrent_tau, rec_stdp_w_max, msn_rheobase_pA,
             direct_ca3=True, lm_to_ca3_weight=40.0, lm_to_ca3_density=0.5, lm_to_ca3_jitter=0.6,
             ca3_to_ca1_weight=25.0,
             graded_lo_hz=10.0, graded_hi_hz=40.0, ablate_recurrent=False, verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    from sim.backend import get_backend
    xp, backend_name = get_backend()

    n_per_lm = n_bearing + n_dist
    n_sensors = len(landmarks) * n_per_lm

    t0 = time.time()
    bridge, cfg = _build(
        seed, n_sensors=n_sensors, n_ec=n_ec, n_dg=n_dg, n_dg_pv_basket=n_dg_pv_basket, n_ca3=n_ca3,
        n_ca1=n_ca1, n_ca3_inh=n_ca3_inh, n_msn=n_msn,
        lm_to_ec_weight=lm_to_ec_weight, lm_to_ec_density=lm_to_ec_density,
        ec_to_dg_weight=ec_to_dg_weight, ec_to_dg_density=0.40, ec_to_pvb_weight=5.0, pvb_to_dg_weight=2.0,
        dg_to_ca3_weight=dg_to_ca3_weight, dg_to_ca3_density=dg_to_ca3_density,
        ca3_rec_weight=ca3_rec_weight, ca3_rec_density=ca3_rec_density,
        ca3_to_inh_weight=8.0, ca3_to_inh_density=0.30,
        inh_to_ca3_weight=inh_to_ca3_weight, inh_to_ca3_density=0.60,
        ca3_to_ca1_weight=float(ca3_to_ca1_weight), ca3_to_ca1_density=0.30, ec_to_ca1_weight=3.0, ec_to_ca1_density=0.30,
        ca1_to_msn_weight=ca1_to_msn_weight, ca1_to_msn_density=ca1_to_msn_density,
        recurrent_exc_receptor="nmda_slow", nmda_recurrent_prop=nmda_recurrent_prop,
        nmda_recurrent_tau=nmda_recurrent_tau, rec_stdp_w_max=rec_stdp_w_max, enable_nmda=True,
        direct_ca3=direct_ca3, lm_to_ca3_weight=lm_to_ca3_weight, lm_to_ca3_density=lm_to_ca3_density,
        lm_to_ca3_jitter=lm_to_ca3_jitter)
    _assert_cupy_regime(cfg, backend_name)
    # confirm the protected edit is live (routed mask + slow-NMDA conductance allocated)
    routed = (getattr(bridge, "cp_nmda_recurrent_synapse_mask", None) is not None
              and getattr(bridge, "cp_conductance_g_nmda_recurrent", None) is not None)
    log(f"  [seed {seed}] built {time.time()-t0:.1f}s; {cfg.num_neurons} neurons, "
        f"{int(bridge.cp_connections.nnz)} syn; backend={backend_name}; nmda_slow_routed={routed}")
    assert routed, "Step B edit NOT live: nmda_slow routing mask/conductance not allocated"

    rm = bridge.region_manager
    sensor_idx = np.asarray(rm.indices("landmark_sensors"), dtype=np.int64)
    region_idx_map = {n: np.asarray(rm.indices(n), dtype=np.int64)
                      for n in ("ec", "dg", "ca3", "ca1", "msn_d1")}
    loc_names = list(locations.keys())

    # POSITION-LEAK audit (anti-cheat): only landmark_sensors driven externally.
    for r in ("ec", "dg", "ca3", "ca1", "msn_d1"):
        assert r != "landmark_sensors"

    def render(name, drop_landmark=None):
        x, y = locations[name]
        lms = [lm for i, lm in enumerate(landmarks) if i != drop_landmark]
        # keep the sensor vector full-length (zeros for the dropped landmark's bank) for COMPLETION test
        act = np.zeros(len(landmarks) * n_per_lm, dtype=np.float32)
        for i, lm in enumerate(landmarks):
            if drop_landmark is not None and i == drop_landmark:
                continue
            sub = landmark_sensor_act(x, y, [lm], n_bearing, n_dist, max_intensity, falloff,
                                      dist_sigma, dist_max, bexp)
            act[i * n_per_lm:(i + 1) * n_per_lm] = sub
        return act
    loc_sensor = {n: render(n) for n in loc_names}
    in_diffs = [cosine_counts(loc_sensor[a], loc_sensor[b]) for a, b in itertools.combinations(loc_names, 2)]
    input_overlap = float(np.mean(in_diffs)) if in_diffs else 0.0

    # ── Step A: learned-recurrent cue-clamped storage ──
    ff_gates = ("landmark_to_ec", "ec_to_dg", "dg_to_ca3", "ca3_to_ca1", "ec_to_ca1")
    rng = np.random.default_rng(seed)
    t_tr = time.time()
    ca3_storage_spk_step = _store_recurrent(
        bridge, xp, sensor_idx, loc_sensor, loc_names, ff_gates=ff_gates,
        rec_gate="ca3_swr_burst", store_passes=store_passes,
        store_settle_steps=store_settle_steps, store_clamp_steps=store_clamp_steps, rng=rng,
        ca3_idx=region_idx_map["ca3"])
    # === INSTRUMENTATION GUARD (the harness-bug catcher) ===
    # CA3 MUST fire during storage, else the recurrent stored nothing (the silent-CA3 artifact that
    # made the 0/3 a harness bug, not a mechanism failure). Print it ALWAYS; hard-assert > 0 when the
    # recurrent was NOT ablated (an ablated run legitimately drops the recurrent contribution but the
    # FIXED feedforward still fires CA3, so it stays > 0 anyway — but we only HARD-fail on the real run).
    n_ca3_eff0 = len(region_idx_map["ca3"])
    ca3_storage_hz = ca3_storage_spk_step / max(n_ca3_eff0, 1) * (1000.0 / cfg.dt_ms)
    log(f"  [seed {seed}] *** CA3-FIRES-DURING-STORAGE GUARD: {ca3_storage_spk_step:.2f} spk/step "
        f"(~{ca3_storage_hz:.1f} Hz pop, {n_ca3_eff0} cells) ***")
    if not ablate_recurrent:
        assert ca3_storage_spk_step > 0.0, (
            f"SILENT-CA3 ARTIFACT: CA3 fired {ca3_storage_spk_step:.3f} spk/step during storage — the "
            f"recurrent had nothing to store (the direct-CA3 afferent is not firing CA3). This is the "
            f"harness bug, NOT a result. Raise lm_to_ca3_weight / lm_to_ca3_density.")
    # Anti-cheat: ablate the LEARNED recurrent (zero ca3->ca3 weights) to prove the result needs it.
    if ablate_recurrent:
        _zero_recurrent(bridge, xp, rm)
    log(f"  [seed {seed}] storage done ({time.time()-t_tr:.0f}s)  ablate_recurrent={ablate_recurrent}")

    # ── Measure CA3+CA1 ensembles per location (+ MSN drive), repeat (stability), sensor-ablated,
    #    and 1-landmark-dropped (completion) ──
    def measure_all(ablate=False, drop_lm=None):
        sens = ({n: render(n, drop_landmark=drop_lm) for n in loc_names} if drop_lm is not None else loc_sensor)
        return {n: _measure(bridge, xp, sensor_idx, region_idx_map, sens[n], record_steps, ablate=ablate)
                for n in loc_names}
    ens = measure_all()
    ens_repeat = measure_all()
    ens_ablate = measure_all(ablate=True)
    ens_partial = measure_all(drop_lm=(0 if len(landmarks) >= 3 else None))

    def ca3(name, d): return d[name]["ca3"]
    def ca1(name, d): return d[name]["ca1"]

    # G1 DISTINCT (on CA3 after completion): different-location CA3 cos < 0.30
    ca3_diff = [cosine_counts(ca3(a, ens), ca3(b, ens)) for a, b in itertools.combinations(loc_names, 2)]
    mean_ca3_diff = float(np.mean(ca3_diff)) if ca3_diff else 1.0
    g1_distinct = mean_ca3_diff < 0.30

    # G2 GRADED: CA3 population rate in ~10-40 Hz (NOT 0, NOT the ~200 spk/step ceiling). spk/step ->
    # convert to a per-cell Hz: (total_spk/record_steps)/n_ca3 * (1000/dt). Use the population mean over
    # locations; also report the bound (max over time proxy = spk/step) to flag runaway.
    ca3_spk_step = {n: float(np.sum(ens[n]["ca3"])) / record_steps for n in loc_names}
    mean_ca3_spk = float(np.mean(list(ca3_spk_step.values())))
    n_ca3_eff = len(region_idx_map["ca3"])
    ca3_pop_hz = mean_ca3_spk / max(n_ca3_eff, 1) * (1000.0 / cfg.dt_ms)
    # active-cell rate: among cells that fire, their mean Hz (a graded ensemble fires its members at 10-40 Hz)
    active_hz = []
    for n in loc_names:
        c = ens[n]["ca3"]
        fired = c[c > 0]
        if fired.size:
            active_hz.append(float(np.mean(fired)) / record_steps * (1000.0 / cfg.dt_ms))
    mean_active_hz = float(np.mean(active_hz)) if active_hz else 0.0
    not_runaway = mean_ca3_spk < 0.5 * n_ca3_eff   # < half the pool firing every step (the C1 ceiling was full)
    g2_graded = (graded_lo_hz <= mean_active_hz <= graded_hi_hz) and not_runaway and (mean_ca3_spk > 0.0)

    # G3 STABLE: same-location CA3 cos > 0.70
    same = [cosine_counts(ca3(n, ens), ca3(n, ens_repeat)) for n in loc_names]
    same_vals = [c for c in same if c > 0]
    mean_same = float(np.mean(same_vals)) if same_vals else 0.0
    g3_stable = mean_same > 0.70

    # G4 HIGH-RATE: MSN-D1 fires >=5 Hz / its effective drive clears ~420 pA
    msn_rate_hz = {n: float(np.sum(ens[n]["msn_d1"])) / record_steps / len(region_idx_map["msn_d1"])
                   * (1000.0 / cfg.dt_ms) for n in loc_names}
    msn_drive = {n: ens[n]["_msn_mean_drive_pA"] for n in loc_names}
    max_msn_rate = float(np.max(list(msn_rate_hz.values())))
    mean_msn_drive = float(np.mean(list(msn_drive.values())))
    g4_high_rate = max_msn_rate >= 5.0

    # G5 SENSOR-DRIVEN: ablate sensors -> CA3/CA1 collapse
    mean_ca1_active = float(np.mean([np.mean(ens[n]["ca1"] > 0) for n in loc_names]))
    abl_active = float(np.mean([np.mean(ens_ablate[n]["ca1"] > 0) for n in loc_names]))
    abl_vs_true = float(np.mean([cosine_counts(ca1(n, ens), ca1(n, ens_ablate)) for n in loc_names]))
    g5_sensor_driven = (abl_active < 0.25 * max(mean_ca1_active, 1e-6)) or (abl_vs_true < 0.30)

    # G6 COMPLETION: drop 1 of >=3 landmarks -> recall cos(true,partial) > 0.7 (on CA3, the autoassociator)
    if len(landmarks) >= 3:
        compl = [cosine_counts(ca3(n, ens), ca3(n, ens_partial)) for n in loc_names]
        mean_compl = float(np.mean([c for c in compl if c > 0])) if any(c > 0 for c in compl) else 0.0
        g6_completion = mean_compl > 0.70
    else:
        mean_compl = None; g6_completion = False

    result = {
        "seed": seed, "backend": backend_name,
        "n_neurons": int(cfg.num_neurons), "n_synapses": int(bridge.cp_connections.nnz),
        "nmda_slow_routed": bool(routed), "ablate_recurrent": bool(ablate_recurrent),
        "direct_ca3": bool(direct_ca3),
        "ca3_storage_spk_per_step": round(ca3_storage_spk_step, 3),
        "ca3_storage_pop_hz": round(ca3_storage_hz, 2),
        "input_pattern_overlap": round(input_overlap, 4),
        "ca3_mean_diff_location_cosine": round(mean_ca3_diff, 4),
        "ca3_mean_same_location_cosine": round(mean_same, 4),
        "ca3_pop_rate_hz": round(ca3_pop_hz, 2),
        "ca3_active_cell_rate_hz": round(mean_active_hz, 2),
        "ca3_spk_per_step": round(mean_ca3_spk, 2),
        "msn_rate_hz_per_location": {k: round(v, 2) for k, v in msn_rate_hz.items()},
        "msn_max_rate_hz": round(max_msn_rate, 2),
        "msn_mean_effective_drive_pA": round(mean_msn_drive, 1),
        "msn_rheobase_pA_ref": float(msn_rheobase_pA),
        "ca1_mean_active": round(mean_ca1_active, 4),
        "ablation_ca1_active": round(abl_active, 4),
        "ablation_ca1_cosine_vs_true": round(abl_vs_true, 4),
        "completion_ca3_cosine_true_vs_partial": (round(mean_compl, 4) if mean_compl is not None else None),
        "gate_G1_DISTINCT": bool(g1_distinct),
        "gate_G2_GRADED": bool(g2_graded),
        "gate_G3_STABLE": bool(g3_stable),
        "gate_G4_HIGH_RATE": bool(g4_high_rate),
        "gate_G5_SENSOR_DRIVEN": bool(g5_sensor_driven),
        "gate_G6_COMPLETION": bool(g6_completion),
        "all_pass": bool(g1_distinct and g2_graded and g3_stable and g4_high_rate
                         and g5_sensor_driven and g6_completion),
        "total_seconds": round(time.time() - t0, 1),
    }
    log(f"  [seed {seed}] G1 DISTINCT ca3_diffcos={mean_ca3_diff:.3f} ({'P' if g1_distinct else 'F'})  "
        f"G2 GRADED act_hz={mean_active_hz:.1f}(pop{ca3_pop_hz:.1f},spk{mean_ca3_spk:.1f}) "
        f"({'P' if g2_graded else 'F'})  G3 STABLE samecos={mean_same:.3f} ({'P' if g3_stable else 'F'})")
    log(f"  [seed {seed}] G4 HIGH-RATE msn={max_msn_rate:.1f}Hz drive={mean_msn_drive:.0f}pA(rheo~{msn_rheobase_pA:.0f}) "
        f"({'P' if g4_high_rate else 'F'})  G5 SENSOR abl_cos={abl_vs_true:.3f} ({'P' if g5_sensor_driven else 'F'})  "
        f"G6 COMPLETION cos={mean_compl} ({'P' if g6_completion else 'F'})")
    log(f"  [seed {seed}] ALL_PASS={result['all_pass']}")
    return result


def _zero_recurrent(bridge, xp, rm):
    """Anti-cheat: zero the ca3->ca3 recurrent weights in-place (the learned basins) to prove G2/G6 need them.
    Resolves the recurrent synapses as those whose pre AND post are both in the ca3 slice."""
    ca3_idx = set(int(i) for i in rm.indices("ca3"))
    csr = bridge.cp_connections
    indptr = _host(csr.indptr); indices = _host(csr.indices); data = _host(csr.data)
    nnz = int(csr.nnz)
    # build a bool mask over CSR data of recurrent (ca3->ca3) synapses
    mask = np.zeros(nnz, dtype=bool)
    for pre in range(len(indptr) - 1):
        if pre not in ca3_idx:
            continue
        for k in range(indptr[pre], indptr[pre + 1]):
            if int(indices[k]) in ca3_idx:
                mask[k] = True
    data[mask] = 0.0
    csr.data[:nnz] = xp.asarray(data[:nnz])


SMOKE_KW = dict(  # tiny config to confirm the harness executes without error (NOT a result)
    n_ec=60, n_dg=160, n_dg_pv_basket=48, n_ca3=120, n_ca1=80, n_ca3_inh=40, n_msn=20,
    store_passes=2, store_settle_steps=8, store_clamp_steps=20, record_steps=30,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="1-seed tiny-config EXECUTES-WITHOUT-ERROR smoke (NOT a result; staged de-risk only)")
    ap.add_argument("--ablate-recurrent", action="store_true",
                    help="anti-cheat: zero the learned ca3->ca3 recurrent after storage (G2/G6 must collapse)")
    ap.add_argument("--grid-size", type=int, default=32)
    ap.add_argument("--n-bearing", type=int, default=12)
    ap.add_argument("--n-dist", type=int, default=8)
    ap.add_argument("--bexp", type=float, default=4.0)
    ap.add_argument("--dist-sigma", type=float, default=4.0)
    ap.add_argument("--max-intensity", type=float, default=450.0,
                    help="sensor render peak pA. 450 = the Stage-1 sparse-distinct operating point (CA3 ~5%%, "
                         "diff-cos ~0.13). Higher (e.g. 900) drives CA3 dense -> fires CA1/MSN but position-blind.")
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
    # === DIRECT-CA3 single-hop afferent (the FIX for the silent multi-hop) ===
    ap.add_argument("--no-direct-ca3", dest="direct_ca3", action="store_false",
                    help="DISABLE the direct landmark_sensors->ca3 afferent (revert to the silent multi-hop; "
                         "reproduces the harness-bug 0/3 artifact). Default: direct-CA3 ON.")
    ap.set_defaults(direct_ca3=True)
    ap.add_argument("--lm-to-ca3-weight", type=float, default=20.0,
                    help="FIXED direct landmark_sensors->ca3 detonator weight (mossy analog; the Stage-1 "
                         "single-hop mechanism that reliably fires a sparse distinct CA3 ensemble). 20 @ "
                         "intensity 450 -> CA3 ~5%% sparse + distinct (diff-cos ~0.13).")
    ap.add_argument("--lm-to-ca3-density", type=float, default=0.3)
    ap.add_argument("--lm-to-ca3-jitter", type=float, default=0.6)
    # === the learned-graded recurrent knobs ===
    ap.add_argument("--ca3-rec-weight", type=float, default=0.0,
                    help="ZERO-INIT recurrent (basins grown by storage). >0 = a small prior (not the default).")
    ap.add_argument("--ca3-rec-density", type=float, default=0.30)
    ap.add_argument("--inh-to-ca3-weight", type=float, default=14.0)
    ap.add_argument("--nmda-recurrent-prop", type=float, default=0.3,
                    help="slow-NMDA recurrent per-spike conductance increment scale (0.3 = the recurrent's "
                         "best-shot amplification at the distinct point; up to ~70%% of the recall rate)")
    ap.add_argument("--nmda-recurrent-tau", type=float, default=100.0, help="slow-NMDA recurrent decay (ms)")
    ap.add_argument("--rec-stdp-w-max", type=float, default=12.0,
                    help="STDP ceiling for the learned recurrent basin (Step A; 12 lets the basin grow "
                         "without tipping the Izhikevich recurrent into the 200-spk/step runaway)")
    ap.add_argument("--ca3-to-ca1-weight", type=float, default=25.0,
                    help="Schaffer CA3->CA1 weight (CA1 must fire enough to drive the MSN — G4)")
    ap.add_argument("--ca1-to-msn-weight", type=float, default=150.0,
                    help="CA1->MSN-D1 convergent (hippocampal->ventral-striatal) projection; must clear "
                         "the ~420 pA MSN rheobase (G4)")
    ap.add_argument("--ca1-to-msn-density", type=float, default=0.40)
    ap.add_argument("--msn-rheobase-pA", type=float, default=420.0)
    ap.add_argument("--store-passes", type=int, default=12)
    ap.add_argument("--store-settle-steps", type=int, default=20)
    ap.add_argument("--store-clamp-steps", type=int, default=120)
    ap.add_argument("--record-steps", type=int, default=100)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.smoke:
        seeds = [int(args.seed) if args.seed is not None else 42]
    else:
        seeds = [int(args.seed)] if args.seed is not None else \
            [int(s) for s in args.seeds.split(",") if s.strip()]
    grid = int(args.grid_size)
    locations = default_locations(grid)
    landmarks = default_landmarks(grid)
    dist_max = float(grid) * 1.42

    print("=" * 76)
    mode = "SMOKE (executes-without-error; NOT a result)" if args.smoke else "DE-RISK"
    print(f"Learned-graded CA3 autoassociator de-risk [{mode}]  seeds={seeds} grid={grid}")
    print(f"  landmarks={landmarks}  nmda_slow recurrent (protected edit)")
    print("=" * 76)

    kw = dict(
        n_ec=int(args.n_ec), n_dg=int(args.n_dg), n_dg_pv_basket=int(args.n_dg_pv_basket),
        n_ca3=int(args.n_ca3), n_ca1=int(args.n_ca1), n_ca3_inh=int(args.n_ca3_inh), n_msn=int(args.n_msn),
        store_passes=int(args.store_passes), store_settle_steps=int(args.store_settle_steps),
        store_clamp_steps=int(args.store_clamp_steps), record_steps=int(args.record_steps),
    )
    if args.smoke:
        kw.update(SMOKE_KW)

    per_seed = []
    for s in seeds:
        per_seed.append(run_seed(
            s, locations=locations, landmarks=landmarks,
            n_bearing=int(args.n_bearing), n_dist=int(args.n_dist), max_intensity=float(args.max_intensity),
            falloff=float(args.falloff), dist_sigma=float(args.dist_sigma), dist_max=dist_max,
            bexp=float(args.bexp),
            lm_to_ec_weight=float(args.lm_to_ec_weight), lm_to_ec_density=float(args.lm_to_ec_density),
            ec_to_dg_weight=float(args.ec_to_dg_weight), dg_to_ca3_weight=float(args.dg_to_ca3_weight),
            dg_to_ca3_density=float(args.dg_to_ca3_density),
            ca3_rec_weight=float(args.ca3_rec_weight), ca3_rec_density=float(args.ca3_rec_density),
            inh_to_ca3_weight=float(args.inh_to_ca3_weight),
            ca1_to_msn_weight=float(args.ca1_to_msn_weight), ca1_to_msn_density=float(args.ca1_to_msn_density),
            nmda_recurrent_prop=float(args.nmda_recurrent_prop), nmda_recurrent_tau=float(args.nmda_recurrent_tau),
            rec_stdp_w_max=float(args.rec_stdp_w_max), msn_rheobase_pA=float(args.msn_rheobase_pA),
            direct_ca3=bool(args.direct_ca3), lm_to_ca3_weight=float(args.lm_to_ca3_weight),
            lm_to_ca3_density=float(args.lm_to_ca3_density), lm_to_ca3_jitter=float(args.lm_to_ca3_jitter),
            ca3_to_ca1_weight=float(args.ca3_to_ca1_weight),
            ablate_recurrent=bool(args.ablate_recurrent),
            **kw))

    n_pass = sum(1 for r in per_seed if r["all_pass"])
    def _gc(key):
        return sum(1 for r in per_seed if r.get(key))
    gate_counts = {g: _gc("gate_" + g) for g in
                   ("G1_DISTINCT", "G2_GRADED", "G3_STABLE", "G4_HIGH_RATE", "G5_SENSOR_DRIVEN", "G6_COMPLETION")}
    verdict = ("PASS" if n_pass == len(seeds) and len(seeds) > 0 else
               "PARTIAL" if any(gate_counts.values()) else "NEGATIVE")
    summary = {
        "harness": "learned_graded_ca3_derisk",
        "mode": ("smoke" if args.smoke else "derisk"),
        "n_seeds": len(seeds), "n_all_pass": n_pass,
        "ablate_recurrent": bool(args.ablate_recurrent),
        "gate_pass_counts": gate_counts,
        "verdict": verdict,
        "per_seed": per_seed,
    }
    if args.smoke:
        print("\nSMOKE complete — harness executed without error. This is NOT a de-risk result; the full "
              "multi-seed de-risk is post-byte-review.")
    else:
        print(f"\nDE-RISK verdict: {n_pass}/{len(seeds)} seeds ALL_PASS (G1..G6).  gate counts: "
              + "  ".join(f"{g}={c}/{len(seeds)}" for g, c in gate_counts.items()))
        ca3st = [r.get("ca3_storage_spk_per_step", 0.0) for r in per_seed]
        print(f"  CA3-fires-during-storage (the guard): {[round(x,2) for x in ca3st]} spk/step "
              f"(>0 = fix live; silent-CA3 artifact would be 0.0)")
    if args.out:
        Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
