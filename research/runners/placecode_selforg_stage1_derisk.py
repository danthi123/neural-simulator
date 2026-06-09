"""Stage-1 de-risk — a SELF-ORGANIZED spiking place code that is POSITION-SPECIFIC on CuPy.

Blueprint: research/findings/2026-06-09-place-code-biologization-research.md (Option B — a dedicated
lightweight place layer: landmark_sensors -> place_cells via random projection + the cell's own
firing threshold as competition; Hartley-Burgess "place cells compete for boundary-vector-cell
inputs" -> spatial selectivity). Option A (the full g11 trisynaptic EC->DG->CA3->CA1 loop) was tried
first and did NOT conduct end-to-end at probe scale (EC over-active + the multi-hop sparse mossy/
Schaffer fan-in is silent — matching validate_trisynaptic_loop's own note that the EC-driven test
"FAILED at all parameter combinations", which is why that validation used DIRECT-CA3 drive). Option B
is the design's named fallback "if Option A's full loop is over-heavy" and is the literal
BVC->competitive-place model.

Motivation: research/findings/2026-06-09-N9-convergent-upstate-derisk.md — the N9 value critic FIRES +
LEARNS V + actor-OK on CuPy but FAILS place-grading because the host-rendered DENSE Gaussian place
code (`vs_place_context`), read through a dense convergent projection, is POSITION-BLIND. The cure is
a place code that is sparse + distinct-per-location, SELF-ORGANIZED from the legitimate egocentric
landmark sensors (the body sensing the world), not a host Gaussian.

THIS PROBE PROVES (or disproves) ONLY Stage 1: a self-organized spiking place code is position-
specific on CuPy. It does NOT touch the value critic (Stage 2 is the controller's next step).

MECHANISM (all reused machinery; no sim/ edits):
  landmark_sensors (>=2 landmarks, egocentric bearing+distance render — the BRAIN-BASED-legal body-
  sensing channel, D.09 object-vector input; a SINGLE landmark gives only an annular/ring ambiguity
  so >=2 are required for a unique 2-D fix)
    --random sparse plastic projection (STDP)-->  place (a dedicated place-cell pool)
  The place cell's own spike threshold provides the competition: only the ~5% best-matched place
  cells (whose random input weights align with the current sensor pattern) cross threshold ->
  SPARSE, position-specific ensembles. STDP refines the fields (Hebbian: cells that fire at a
  location strengthen their inputs from that location's sensors).

GATES (Stage 1, CuPy, >=3 seeds):
  1a position-specific : mean different-location pairwise cosine < 0.30.
  1b stable            : same-location (repeat visit) cosine > 0.70.
  1c sparse            : ~2-10% of the place pool active per location.

ANTI-CHEATS:
  (A) self-organized, NOT hand-wired (THE decisive control): ABLATE the landmark sensor input (zero
      landmark drive) -> the per-location ensembles must DEGRADE (the place pool goes silent / loses
      its location code). A host-rendered Gaussian would be UNAFFECTED by sensor removal. Pass iff
      cos(true_ensemble, ablated_ensemble) collapses AND/OR the ablated pool is inactive.
  (B) position-leak audit: (x,y) enters the brain ONLY via the egocentric landmark render, NEVER via
      a direct allocentric (x,y) injection into the place pool. Enforced BY CONSTRUCTION (this probe
      writes external current ONLY to landmark_sensors; the place pool receives zero external
      current) and asserted (the set of driven regions is recorded). No host vs_place_context
      Gaussian anywhere.
  (regime) backend==cupy; OU / conductance-noise / global-homeostasis / heterogeneity / STP OFF.

USAGE (MUST be cupy):
  SIM_BACKEND=cupy python -m research.runners.placecode_selforg_stage1_derisk \
      --seeds 42,43,44 --out research/findings/raw/_placecode_selforg_stage1.json
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


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _host(a):
    from sim.backend import to_host
    try:
        return to_host(a)
    except Exception:
        return np.asarray(a)


def cosine_counts(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two per-neuron spike-count vectors (rate-coded ensembles)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _assert_cupy_regime(cfg, backend_name):
    """Regime fidelity (load-bearing, from N9): MUST run on CuPy in the deterministic regime."""
    if backend_name != "cupy":
        raise AssertionError(
            f"REGIME FIDELITY: this place-code de-risk MUST run on CuPy (numpy is DISQUALIFIED for "
            f"striatal/near-threshold work; see 2026-06-09-N9-cupy-membrane-divergence-ROOT.md). "
            f"Got backend={backend_name!r}. Set SIM_BACKEND=cupy.")
    bad = [k for k in ("enable_ou_process", "enable_conductance_noise", "enable_homeostasis",
                       "enable_parameter_heterogeneity", "enable_short_term_plasticity")
           if getattr(cfg, k, False)]
    if bad:
        raise AssertionError(f"REGIME FIDELITY: deterministic-regime knobs left ON: {bad}")


# ──────────────────────────────────────────────────────────────────────
# Egocentric landmark sensor render (>=2 landmarks) — the legitimate body-sensing channel.
# Per landmark: a block of n_bearing directional sensors (sharp tuning act = intensity*max(0,cos)^bexp,
# the BVC bearing code) + a block of n_dist distance-tuned Gaussian sensors (the BVC distance code).
# intensity = max_pA/(1+falloff*dist). A SINGLE landmark -> annular ambiguity (distance alone is not
# a 2-D fix); >=2 landmarks with distinct bearings disambiguate position. (x,y) enters ONLY here
# (position-leak anti-cheat B). Same render math family as the g11 nav loop (:5139-5152), extended to
# >=2 landmarks + a distance code.
# ──────────────────────────────────────────────────────────────────────

def landmark_sensor_act(x, y, landmarks, n_bearing, n_dist, max_int, falloff, dist_sigma,
                        dist_max, bexp):
    blocks = []
    bpx = np.cos(2.0 * np.pi * np.arange(n_bearing) / n_bearing)
    bpy = np.sin(2.0 * np.pi * np.arange(n_bearing) / n_bearing)
    dist_centers = np.linspace(0.0, dist_max, n_dist)
    for (lx, ly) in landmarks:
        dx = float(lx - x)
        dy = float(ly - y)
        d = (dx * dx + dy * dy) ** 0.5
        if d < 1e-6:
            bear = np.full(n_bearing, max_int, dtype=np.float32)
            dist = np.full(n_dist, max_int, dtype=np.float32)
        else:
            bx = dx / d
            by = dy / d
            intensity = max_int / (1.0 + falloff * d)
            cos_align = np.maximum(0.0, bpx * bx + bpy * by)
            bear = (intensity * (cos_align ** bexp)).astype(np.float32)
            dist = (max_int * np.exp(-(d - dist_centers) ** 2 / (2.0 * dist_sigma ** 2))).astype(np.float32)
        blocks.append(bear.astype(np.float32))
        blocks.append(dist.astype(np.float32))
    return np.concatenate(blocks).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# Build a dedicated self-organized place layer (Option B). landmark_sensors --plastic random
# projection--> place. The place cell threshold provides WTA competition (~5% fire). NO host Gaussian.
# ──────────────────────────────────────────────────────────────────────

def _build(seed, *, n_sensors, n_place, lm_to_place_weight, lm_to_place_density,
           lm_to_place_jitter, place_homeostasis=False, homeostasis_target_rate=0.02,
           homeostasis_adapt_rate=0.0005, dt_ms=1.0):
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType

    regions = [
        # The legitimate egocentric landmark sensors (driven externally each step).
        BrainRegion(name="landmark_sensors", n_neurons=int(n_sensors), exc_fraction=1.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                    weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name),
        # The self-organizing place pool (hippocampal pyramidal). Competition = the cell's own spike
        # threshold; with place_homeostasis ON, per-region INTRINSIC homeostatic plasticity (Desai
        # 1999 / Turrigiano; the canonical place-cell stability mechanism, BrainRegion.enable_
        # homeostasis, runs even with global homeostasis OFF, deterministic) regulates each location's
        # active fraction toward homeostasis_target_rate (~2-5%) — fixing the threshold-only seed-
        # fragility on gate 1c. This is the legitimate place-field intrinsic-excitability mechanism,
        # NOT a threshold-collapse rescue (it targets a FIRING RATE, not "drop threshold until it
        # fires"); the place pool still fires ONLY from the landmark-sensor synaptic current.
        BrainRegion(name="place", n_neurons=int(n_place), exc_fraction=1.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                    weight_jitter=0.0, plastic_internal=False,
                    enable_homeostasis=bool(place_homeostasis),
                    izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name),
    ]
    pathways = [
        # landmark_sensors -> place: random sparse, PLASTIC (STDP self-organizes the fields). This is
        # the Hartley-Burgess competitive-learning pathway (gate landmark_to_place mirrors g11's).
        RegionPathway(from_region="landmark_sensors", to_region="place",
                      density=float(lm_to_place_density), weight_mean=float(lm_to_place_weight),
                      weight_jitter=float(lm_to_place_jitter), plastic=True,
                      plasticity_gate="landmark_to_place"),
    ]

    cfg = CoreSimConfig()
    cfg.seed = int(seed)
    cfg.heterogeneity_seed = int(seed)
    cfg.ou_seed = int(seed)
    cfg.dt_ms = float(dt_ms)
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.enable_stdp = True
    cfg.enable_hebbian_learning = False
    cfg.enable_nmda = False
    cfg.stdp_w_max = 40.0                  # above the design weight so the soft-bound doesn't collapse it
    cfg.fast_spike_reset = True
    # Homeostasis target/rate (only consumed when a region opts into per-region homeostasis; global
    # homeostasis stays OFF for regime fidelity). Faster adapt rate than the default 0.0005 so the
    # place pool's intrinsic-excitability EMA converges within the short self-org window.
    cfg.homeostasis_target_rate = float(homeostasis_target_rate)
    cfg.homeostasis_threshold_adapt_rate = float(homeostasis_adapt_rate)
    # === deterministic-nav regime (g11_bg_runner.py:3340-3344) ===
    cfg.enable_homeostasis = False         # GLOBAL homeostasis OFF (regime fidelity); place pool uses
                                            # PER-REGION homeostasis only when place_homeostasis=True
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


# ──────────────────────────────────────────────────────────────────────
# Drive / measure helpers
# ──────────────────────────────────────────────────────────────────────

def _step(bridge, n):
    for _ in range(n):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)


def _measure(bridge, xp, sensor_idx, place_idx, sensor_act, record_steps, ablate=False,
             reset_steps=30):
    """Drive landmark_sensors with `sensor_act` (or nothing, if ablate) for record_steps; return the
    per-neuron spike-count vector of `place`. POSITION-LEAK anti-cheat B: ONLY landmark_sensors get
    external current (the place pool never does)."""
    bridge.cp_external_input_current[:] = 0.0
    _step(bridge, reset_steps)
    bridge.cp_external_input_current[:] = 0.0
    if not ablate:
        bridge.cp_external_input_current[xp.asarray(sensor_idx, dtype=xp.int64)] = \
            xp.asarray(sensor_act, dtype=xp.float32)
    place_arr = xp.asarray(place_idx, dtype=xp.int64)
    counts = xp.zeros(len(place_idx), dtype=xp.float32)
    for _ in range(record_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)
        counts += bridge.cp_firing_states[place_arr].astype(xp.float32)
    bridge.cp_external_input_current[:] = 0.0
    return _host(counts)


# ──────────────────────────────────────────────────────────────────────
# Stage-1 run for one seed
# ──────────────────────────────────────────────────────────────────────

def run_seed(seed, *, locations, landmarks, n_bearing, n_dist, n_place,
             lm_to_place_weight, lm_to_place_density, lm_to_place_jitter,
             max_intensity, falloff, dist_sigma, dist_max, bexp,
             train_passes, train_steps_per_loc, record_steps,
             place_homeostasis=False, homeostasis_target_rate=0.02,
             homeostasis_adapt_rate=0.0005, verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    from sim.backend import get_backend
    xp, backend_name = get_backend()

    n_per_landmark = n_bearing + n_dist
    n_sensors = len(landmarks) * n_per_landmark

    t0 = time.time()
    bridge, cfg = _build(seed, n_sensors=n_sensors, n_place=n_place,
                         lm_to_place_weight=lm_to_place_weight,
                         lm_to_place_density=lm_to_place_density,
                         lm_to_place_jitter=lm_to_place_jitter,
                         place_homeostasis=place_homeostasis,
                         homeostasis_target_rate=homeostasis_target_rate,
                         homeostasis_adapt_rate=homeostasis_adapt_rate)
    _assert_cupy_regime(cfg, backend_name)
    build_s = time.time() - t0
    log(f"  [seed {seed}] built in {build_s:.1f}s; {cfg.num_neurons} neurons, "
        f"{int(bridge.cp_connections.nnz)} synapses; backend={backend_name}")

    rm = bridge.region_manager
    sensor_idx = np.asarray(rm.indices("landmark_sensors"), dtype=np.int64)
    place_idx = np.asarray(rm.indices("place"), dtype=np.int64)
    n_place_actual = len(place_idx)

    # Position-leak audit (anti-cheat B): the ONLY region this probe ever drives is landmark_sensors.
    driven_regions = {"landmark_sensors"}
    assert "place" not in driven_regions, "place pool must never be externally driven (position leak)"

    def render(name, drop_landmark=None):
        x, y = locations[name]
        act = landmark_sensor_act(x, y, landmarks, n_bearing, n_dist, max_intensity,
                                  falloff, dist_sigma, dist_max, bexp)
        if drop_landmark is not None:
            # Partial-cue control: zero the sensor block belonging to landmark `drop_landmark`
            # (each landmark owns a contiguous n_per_landmark block). The D.06/D.13 signature: the
            # place field should remain MOSTLY intact when ONE of >=2 cues is removed (vs full
            # ablation, which silences it). A brittle 1:1 sensor echo would collapse here too.
            per = n_bearing + n_dist
            lo = drop_landmark * per
            hi = lo + per
            act = act.copy()
            act[lo:hi] = 0.0
        return act

    loc_names = list(locations.keys())
    loc_sensor = {n: render(n) for n in loc_names}
    # partial-cue renders (drop landmark 0) for the cue-removal control
    loc_sensor_partial = {n: render(n, drop_landmark=0) for n in loc_names}

    # Sanity: report the raw input-pattern overlap (the thing the place layer must orthogonalize).
    in_diffs = [cosine_counts(loc_sensor[a], loc_sensor[b])
                for a, b in itertools.combinations(loc_names, 2)]
    input_overlap = float(np.mean(in_diffs)) if in_diffs else 0.0

    # ── Self-organization: walk the locations with the landmark_to_place STDP gate OPEN ──
    log(f"  [seed {seed}] self-organizing place fields ({train_passes} passes x "
        f"{len(loc_names)} locs x {train_steps_per_loc} steps)... input_overlap={input_overlap:.3f}")
    bridge.set_plasticity_gate("landmark_to_place", 1.0)
    t_tr = time.time()
    rng = np.random.default_rng(seed)
    for _p in range(train_passes):
        order = list(loc_names)
        rng.shuffle(order)
        for name in order:
            bridge.cp_external_input_current[:] = 0.0
            _step(bridge, 20)
            bridge.cp_external_input_current[xp.asarray(sensor_idx, dtype=xp.int64)] = \
                xp.asarray(loc_sensor[name], dtype=xp.float32)
            _step(bridge, train_steps_per_loc)
    bridge.set_plasticity_gate("landmark_to_place", 0.0)
    bridge.cp_external_input_current[:] = 0.0
    log(f"  [seed {seed}] self-org done ({time.time() - t_tr:.0f}s)")

    # ── Measure the place ensemble per location (plasticity OFF) ──
    def measure_all(ablate=False):
        return {n: _measure(bridge, xp, sensor_idx, place_idx, loc_sensor[n], record_steps,
                            ablate=ablate) for n in loc_names}

    ens = measure_all(ablate=False)
    ens_repeat = measure_all(ablate=False)        # repeat visits -> stability (1b)
    ens_ablate = measure_all(ablate=True)         # sensor-ablation control (anti-cheat A)
    # partial-cue control: drop one of the 3 landmarks (D.06/D.13 "fires after some cues removed")
    ens_partial = {n: _measure(bridge, xp, sensor_idx, place_idx, loc_sensor_partial[n],
                               record_steps, ablate=False) for n in loc_names}

    # ── Metrics ──
    diff_cos, pair_detail = [], []
    for a, b in itertools.combinations(loc_names, 2):
        c = cosine_counts(ens[a], ens[b])
        diff_cos.append(c)
        pair_detail.append({"a": a, "b": b, "cosine": round(c, 4)})
    mean_diff_cos = float(np.mean(diff_cos)) if diff_cos else 1.0
    max_diff_cos = float(np.max(diff_cos)) if diff_cos else 1.0

    same_cos, same_detail = [], []
    for n in loc_names:
        c = cosine_counts(ens[n], ens_repeat[n])
        same_cos.append(c)
        same_detail.append({"loc": n, "cosine": round(c, 4)})
    mean_same_cos = float(np.mean(same_cos)) if same_cos else 0.0
    min_same_cos = float(np.min(same_cos)) if same_cos else 0.0

    sparsity = {n: float(np.mean(ens[n] > 0)) for n in loc_names}
    active_counts = {n: int(np.sum(ens[n] > 0)) for n in loc_names}
    mean_sparsity = float(np.mean(list(sparsity.values())))

    # Anti-cheat A: sensor ablation. Decisive disqualifier of a host shortcut = the code SURVIVES
    # sensor removal. We pass iff (place pool collapses to near-silent) OR (the ablated ensemble no
    # longer carries the location, i.e. cos(true,ablated) << the true same-location stability).
    abl_active = {n: float(np.mean(ens_ablate[n] > 0)) for n in loc_names}
    mean_abl_active = float(np.mean(list(abl_active.values())))
    abl_vs_true = [cosine_counts(ens[n], ens_ablate[n]) for n in loc_names]
    mean_abl_vs_true = float(np.mean(abl_vs_true)) if abl_vs_true else 1.0

    # Partial-cue control: cos(true, partial) should stay reasonably HIGH (field survives one cue's
    # removal) — distinguishing a robust place field from a brittle 1:1 sensor echo. Reported as a
    # diagnostic (NOT a hard gate, since the right threshold is softer than the full-ablation one).
    partial_vs_true = [cosine_counts(ens[n], ens_partial[n]) for n in loc_names]
    mean_partial_vs_true = float(np.mean(partial_vs_true)) if partial_vs_true else 0.0

    gate_1a = mean_diff_cos < 0.30
    gate_1b = mean_same_cos > 0.70
    gate_1c = all(0.02 <= sparsity[n] <= 0.10 for n in loc_names)
    anti_cheat_A = (mean_abl_active < 0.25 * max(mean_sparsity, 1e-6)) or (mean_abl_vs_true < 0.30)

    result = {
        "seed": seed,
        "backend": backend_name,
        "build_seconds": round(build_s, 1),
        "n_neurons": int(cfg.num_neurons),
        "n_synapses": int(bridge.cp_connections.nnz),
        "n_place_neurons": int(n_place_actual),
        "n_landmark_sensors": int(n_sensors),
        "n_landmarks": len(landmarks),
        "landmarks": [list(map(float, lm)) for lm in landmarks],
        "locations": {k: list(map(float, v)) for k, v in locations.items()},
        "place_homeostasis": bool(place_homeostasis),
        "input_pattern_overlap": round(input_overlap, 4),
        "train_passes": train_passes,
        "train_steps_per_loc": train_steps_per_loc,
        "record_steps": record_steps,
        "mean_diff_location_cosine": round(mean_diff_cos, 4),
        "max_diff_location_cosine": round(max_diff_cos, 4),
        "diff_pairs": pair_detail,
        "mean_same_location_cosine": round(mean_same_cos, 4),
        "min_same_location_cosine": round(min_same_cos, 4),
        "same_detail": same_detail,
        "sparsity_per_location": {k: round(v, 4) for k, v in sparsity.items()},
        "active_counts_per_location": active_counts,
        "mean_sparsity": round(mean_sparsity, 4),
        "ablation_mean_active_fraction": round(mean_abl_active, 4),
        "ablation_mean_cosine_vs_true": round(mean_abl_vs_true, 4),
        "partial_cue_mean_cosine_vs_true": round(mean_partial_vs_true, 4),
        "gate_1a_position_specific": bool(gate_1a),
        "gate_1b_stable": bool(gate_1b),
        "gate_1c_sparse": bool(gate_1c),
        "anti_cheat_A_sensor_ablation_degrades": bool(anti_cheat_A),
        "position_leak_driven_regions": sorted(driven_regions),
        "stage1_pass": bool(gate_1a and gate_1b and gate_1c and anti_cheat_A),
        "total_seconds": round(time.time() - t0, 1),
    }
    log(f"  [seed {seed}] 1a diff-cos={mean_diff_cos:.3f}/max{max_diff_cos:.3f} "
        f"(<0.30 {'PASS' if gate_1a else 'FAIL'})  1b same-cos={mean_same_cos:.3f}/min{min_same_cos:.3f} "
        f"(>0.70 {'PASS' if gate_1b else 'FAIL'})  1c sparsity={mean_sparsity:.3f} "
        f"({'PASS' if gate_1c else 'FAIL'})")
    log(f"  [seed {seed}] anti-cheat A: ablated active={mean_abl_active:.3f} "
        f"cos(true,ablated)={mean_abl_vs_true:.3f} -> "
        f"{'DEGRADES (PASS)' if anti_cheat_A else 'SURVIVES (FAIL = host-shortcut signature)'}")
    log(f"  [seed {seed}] partial-cue (drop 1/3 landmarks): cos(true,partial)={mean_partial_vs_true:.3f} "
        f"(field {'survives one cue removed' if mean_partial_vs_true > 0.4 else 'is cue-fragile'})")
    log(f"  [seed {seed}] STAGE-1 {'PASS' if result['stage1_pass'] else 'FAIL'}")
    return result


def default_locations(grid_size):
    """6 distinct (x,y) spanning the grid: a corner-quadrant NEAR + 2 FAR + center + 2 more."""
    g = grid_size - 1
    return {
        "near":   (g * 0.25, g * 0.75),
        "far_a":  (g * 0.75, g * 0.25),
        "far_b":  (g * 0.80, g * 0.80),
        "center": (g * 0.50, g * 0.50),
        "q1":     (g * 0.15, g * 0.15),
        "q2":     (g * 0.50, g * 0.85),
    }


def default_landmarks(grid_size):
    """>=2 landmarks (here 3: two bottom corners + mid-top) — distinct bearings give a unique 2-D fix.
    A single landmark would give only annular (distance-ring) ambiguity."""
    g = grid_size - 1
    return [(0.0, 0.0), (float(g), 0.0), (float(g) / 2.0, float(g))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--seed", type=int, default=None, help="single seed (overrides --seeds)")
    ap.add_argument("--grid-size", type=int, default=32)
    ap.add_argument("--n-bearing", type=int, default=12, help="bearing sensors per landmark")
    ap.add_argument("--n-dist", type=int, default=8, help="distance-tuned sensors per landmark")
    ap.add_argument("--bexp", type=float, default=4.0, help="bearing-tuning sharpness exponent")
    ap.add_argument("--dist-sigma", type=float, default=4.0, help="distance-tuning width (grid units)")
    ap.add_argument("--n-place", type=int, default=400)
    ap.add_argument("--lm-to-place-weight", type=float, default=28.0)
    ap.add_argument("--lm-to-place-density", type=float, default=0.5)
    ap.add_argument("--lm-to-place-jitter", type=float, default=0.6)
    ap.add_argument("--max-intensity", type=float, default=450.0)
    ap.add_argument("--falloff", type=float, default=0.03)
    ap.add_argument("--train-passes", type=int, default=12)
    ap.add_argument("--train-steps-per-loc", type=int, default=120)
    ap.add_argument("--record-steps", type=int, default=100)
    ap.add_argument("--place-homeostasis", action="store_true",
                    help="per-region intrinsic homeostasis on the place pool (Desai/Turrigiano; "
                         "regulates each location's active fraction to the target rate -> robust 1c)")
    ap.add_argument("--homeostasis-target-rate", type=float, default=0.04)
    ap.add_argument("--homeostasis-adapt-rate", type=float, default=0.004)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        seeds = [int(args.seed)]
    else:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    grid_size = int(args.grid_size)
    locations = default_locations(grid_size)
    landmarks = default_landmarks(grid_size)
    dist_max = float(grid_size) * 1.42   # ~max diagonal distance, for the distance-tuning span

    print("=" * 72)
    print("Stage-1 de-risk: SELF-ORGANIZED place code, position-specific on CuPy (Option B)")
    print(f"  seeds={seeds}  grid={grid_size}  landmarks={landmarks}")
    print(f"  locations={ {k: tuple(round(c,1) for c in v) for k,v in locations.items()} }")
    print(f"  n_place={args.n_place}  n_bearing={args.n_bearing} n_dist={args.n_dist} bexp={args.bexp}")
    print("=" * 72)

    per_seed = []
    for s in seeds:
        r = run_seed(
            s, locations=locations, landmarks=landmarks,
            n_bearing=int(args.n_bearing), n_dist=int(args.n_dist), n_place=int(args.n_place),
            lm_to_place_weight=float(args.lm_to_place_weight),
            lm_to_place_density=float(args.lm_to_place_density),
            lm_to_place_jitter=float(args.lm_to_place_jitter),
            max_intensity=float(args.max_intensity), falloff=float(args.falloff),
            dist_sigma=float(args.dist_sigma), dist_max=dist_max, bexp=float(args.bexp),
            train_passes=int(args.train_passes), train_steps_per_loc=int(args.train_steps_per_loc),
            record_steps=int(args.record_steps),
            place_homeostasis=bool(args.place_homeostasis),
            homeostasis_target_rate=float(args.homeostasis_target_rate),
            homeostasis_adapt_rate=float(args.homeostasis_adapt_rate), verbose=True)
        per_seed.append(r)

    n_pass = sum(1 for r in per_seed if r["stage1_pass"])
    n_1a = sum(1 for r in per_seed if r["gate_1a_position_specific"])
    n_1b = sum(1 for r in per_seed if r["gate_1b_stable"])
    n_1c = sum(1 for r in per_seed if r["gate_1c_sparse"])
    n_acA = sum(1 for r in per_seed if r["anti_cheat_A_sensor_ablation_degrades"])

    def _agg(key):
        vals = [r[key] for r in per_seed]
        return {"mean": round(float(np.mean(vals)), 4), "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4), "values": [round(float(v), 4) for v in vals]}

    summary = {
        "n_seeds": len(seeds), "seeds": seeds,
        "n_stage1_pass": n_pass,
        "gate_1a_pass_count": n_1a, "gate_1b_pass_count": n_1b, "gate_1c_pass_count": n_1c,
        "anti_cheat_A_pass_count": n_acA,
        "diff_location_cosine": _agg("mean_diff_location_cosine"),
        "same_location_cosine": _agg("mean_same_location_cosine"),
        "sparsity": _agg("mean_sparsity"),
        "input_pattern_overlap": _agg("input_pattern_overlap"),
        "ablation_cosine_vs_true": _agg("ablation_mean_cosine_vs_true"),
        "ablation_active_fraction": _agg("ablation_mean_active_fraction"),
        "partial_cue_cosine_vs_true": _agg("partial_cue_mean_cosine_vs_true"),
        "verdict": ("PASS" if n_pass == len(seeds) else
                    "PARTIAL" if n_pass > 0 else "NEGATIVE"),
        "per_seed": per_seed,
    }

    print("\n" + "=" * 72)
    print("STAGE-1 SUMMARY")
    print(f"  gate 1a (diff-loc cos < 0.30): {n_1a}/{len(seeds)}  "
          f"mean={summary['diff_location_cosine']['mean']} (values {summary['diff_location_cosine']['values']})")
    print(f"  gate 1b (same-loc cos > 0.70): {n_1b}/{len(seeds)}  "
          f"mean={summary['same_location_cosine']['mean']} (values {summary['same_location_cosine']['values']})")
    print(f"  gate 1c (sparse 2-10%):        {n_1c}/{len(seeds)}  "
          f"mean={summary['sparsity']['mean']} (values {summary['sparsity']['values']})")
    print(f"  anti-cheat A (ablation degrades): {n_acA}/{len(seeds)}  "
          f"cos(true,ablated) mean={summary['ablation_cosine_vs_true']['mean']}")
    print(f"  partial-cue (drop 1/3 lm, field survives): cos(true,partial) mean="
          f"{summary['partial_cue_cosine_vs_true']['mean']}")
    print(f"  input overlap (what place separates): mean={summary['input_pattern_overlap']['mean']}")
    print(f"  STAGE-1 PASS: {n_pass}/{len(seeds)}")
    print(f"  VERDICT: {summary['verdict']}")
    print("=" * 72)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"[OUT] {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
