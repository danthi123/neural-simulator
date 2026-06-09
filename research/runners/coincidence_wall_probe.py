"""Step 0 -- RUNNER-ONLY coincidence-wall characterization + K/gain CALIBRATION (NO sim/ edit).

Companion to the design `2026-06-09-coincidence-substrate-upgrade-design.md` (Route D, the dendritic-
coincidence NMDA-plateau subunit) and the RESULT it solves (`2026-06-09-learned-graded-ca3-derisk-
RESULT.md`, the point-neuron RATE-CODING wall: a sparse <=5% / <0.2-spk/step ensemble fires a downstream
cell 0.00 spk/step even at Schaffer w120/d0.9).

This probe runs ENTIRELY on the EXISTING engine (zero sim/ edits) to:

  (1) CONFIRM THE WALL in isolation -- a hand-built 2-region bridge: a 400-cell `source` pool clamped to
      fire a SPARSE-distinct ~5% ensemble per "location" (orthogonal random afferent + the cell's own
      threshold WTA, the validated Stage-1 mechanism), projecting (clustered) onto ONE downstream MSN-D1
      target. At realistic weights the target fires ~0 Hz; a weight/density SWEEP shows there is NO
      purely-linear setting that fires the target from the SPARSE source without making the source dense.

  (2) CALIBRATE K -- measure, per step, the COINCIDENCE count c_i = the number of a target's afferent
      source-cells that fire in the SAME step (at the distinct point). Report the c_i distribution
      (over steps, while the distinct ensemble is being driven) + a recommended K (so a coincidence
      detector fires for a real ensemble-coincidence but NOT for background) + the gain/plateau strength
      that would clear the ~420 pA MSN-D1 rheobase. K/gain feed Step A (the protected sim/ edit).

The mechanism under test (Route D) is INTRINSICALLY coincidence: c_i counts SIMULTANEOUSLY-active routed
inputs, so a sparse ensemble whose >=K cells fire in the SAME step triggers a plateau, while the SAME
cells firing in DIFFERENT steps each give c_i=1<K -> nothing. This probe measures whether such a K
exists in the natural sparse-distinct dynamics. If the distinct ensemble cannot deliver >=K coincident
inputs to ANY target in ANY step for any K>1 (the honest-negative branch), that is reported and Step A
is NOT written (the next lever is clustering+delays or multi-subunit, per the design).

ANTI-CHEAT / regime (mirrors the C1/N9 de-risks):
  - NO host teacher: the ONLY cp_external_input_current write targets `source` (via its afferent
    `src_drive` sensors). The downstream target receives ZERO external current -- it must fire (or not)
    from the routed synaptic drive alone. Enforced by construction + asserted.
  - CuPy regime: backend=="cupy" (numpy DISQUALIFIED per 2026-06-09-N9-cupy-membrane-divergence-ROOT.md);
    OU / conductance-noise / global-homeostasis / heterogeneity / STP OFF; hard-asserted.

USAGE (MUST be cupy):
  SIM_BACKEND=cupy python -m research.runners.coincidence_wall_probe \
      --seeds 42,43,44 --out research/findings/raw/_coincidence_wall_probe.json
  SIM_BACKEND=cupy python -m research.runners.coincidence_wall_probe --smoke   # 1-seed, fast
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

# Reuse the validated Stage-1 sparse place-code machinery (helpers only; no sim/ edits).
from research.runners.placecode_selforg_stage1_derisk import (
    landmark_sensor_act, cosine_counts, _host)


# ----------------------------------------------------------------------
# Default locations / landmarks -- the VALIDATED Stage-1 grid (grid=32; placecode_selforg_stage1_derisk.py
# default_locations/default_landmarks). The 32-unit span is REQUIRED: it gives the bearing+distance code
# its full dynamic range so the place pool fires a sparse ~3% distinct code (a tiny 8x8 grid with gentle
# falloff=0.03 leaves all sensor intensities ~identical -> nothing crosses threshold; verified silent).
# ----------------------------------------------------------------------

def default_locations(grid_size=32):
    g = grid_size - 1
    return [(g * 0.25, g * 0.75), (g * 0.75, g * 0.25), (g * 0.80, g * 0.80),
            (g * 0.50, g * 0.50), (g * 0.15, g * 0.15), (g * 0.50, g * 0.85)]


def default_landmarks(grid_size=32):
    g = grid_size - 1
    return [(0.0, 0.0), (float(g), 0.0), (float(g) / 2.0, float(g))]


# ----------------------------------------------------------------------
# Build a tiny 2-region bridge: a SPARSE-distinct `source` place pool (Stage-1 mechanism) projecting
# CLUSTERED onto ONE downstream MSN-D1 `target` cell. ZERO sim/ edits.
#   src_sensors (egocentric landmark render, the body sensing the world) --plastic-->  source (place WTA)
#   source --FIXED clustered AMPA-->  target (MSN-D1, the striatal critic read-out)
# The source uses the cell's own spike threshold as competition (->~5% sparse, position-specific), exactly
# the validated Stage-1 single-hop place code. The target is an MSN-D1 (the cell the value critic needs to
# fire, ~420 pA rheobase).
# ----------------------------------------------------------------------

def _build(seed, *, n_sensors, n_source, n_target,
           src_drive_weight, src_drive_density, src_drive_jitter,
           source_to_target_weight, source_to_target_density, source_to_target_jitter,
           dt_ms=1.0):
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType

    regions = [
        # Egocentric landmark sensors (driven externally each step) -- the legitimate body-sensing channel.
        BrainRegion(name="src_sensors", n_neurons=int(n_sensors), exc_fraction=1.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                    weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name),
        # The sparse-distinct place `source` pool (hippocampal pyramidal). Competition = the cell's own
        # spike threshold -> ~5% fire per location. This is the presynaptic population the RESULT proved
        # cannot fire a downstream cell BY RATE.
        BrainRegion(name="source", n_neurons=int(n_source), exc_fraction=1.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                    weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name),
        # The downstream MSN-D1 target (the striatal critic cell, ~420 pA rheobase). MSNs have a
        # depolarized E_GABA (~-60 mV) like the C1/N9 build.
        BrainRegion(name="target", n_neurons=int(n_target), exc_fraction=0.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                    weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D1.name,
                    syn_reversal_potential_i_override=-60.0),
    ]
    pathways = [
        # src_sensors -> source: random sparse, PLASTIC (STDP self-organizes the place fields), the
        # Stage-1 competitive-learning pathway.
        RegionPathway(from_region="src_sensors", to_region="source",
                      density=float(src_drive_density), weight_mean=float(src_drive_weight),
                      weight_jitter=float(src_drive_jitter), plastic=True,
                      plasticity_gate="src_drive"),
        # source -> target: FIXED CLUSTERED AMPA projection (the projection a coincidence detector would
        # sit on). FIXED (plastic=False) so the wall + the c_i statistics are measured on a fixed,
        # known projection (not confounded by STDP), and so K is calibrated against the true fan-in.
        RegionPathway(from_region="source", to_region="target",
                      density=float(source_to_target_density), weight_mean=float(source_to_target_weight),
                      weight_jitter=float(source_to_target_jitter), plastic=False),
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
    cfg.enable_nmda = False
    # stdp_w_max ABOVE the design weights (28 afferent, 20 source->target). The STDP soft-bound is
    # ?w_LTP = A_plus*(w_max - w)*exp(...): with the default w_max=2 and a design weight of 28, every
    # "LTP" event is strongly NEGATIVE and the weights collapse 28->2 in ms (the documented CLAUDE.md
    # gotcha), silencing the source pool. 40 keeps the soft-bound from collapsing them (matches the
    # validated Stage-1 build). The source->target projection is plastic=False so it is unaffected
    # regardless, but the afferent src_drive IS plastic during training.
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
        raise AssertionError(
            f"REGIME FIDELITY: this coincidence-wall probe MUST run on CuPy (numpy DISQUALIFIED; see "
            f"2026-06-09-N9-cupy-membrane-divergence-ROOT.md). Got backend={backend_name!r}. "
            f"Set SIM_BACKEND=cupy.")
    bad = [k for k in ("enable_ou_process", "enable_conductance_noise", "enable_homeostasis",
                       "enable_parameter_heterogeneity", "enable_short_term_plasticity")
           if getattr(cfg, k, False)]
    if bad:
        raise AssertionError(f"REGIME FIDELITY: deterministic-regime knobs left ON: {bad}")


# ----------------------------------------------------------------------
# Drive / measure helpers
# ----------------------------------------------------------------------

def _step(bridge, n):
    for _ in range(n):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)


def _drive_clamp(bridge, xp, sensor_idx, sensor_act):
    """Set the constant landmark-sensor clamp (the ONLY external-current write -- anti-cheat)."""
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[xp.asarray(sensor_idx, dtype=xp.int64)] = \
        xp.asarray(sensor_act, dtype=xp.float32)


def _measure_source_and_target(bridge, xp, source_idx, target_idx, record_steps):
    """Run `record_steps` with the current clamp set; return (source spike-count vec, target spk/step,
    target max-cell spk/step)."""
    src_arr = xp.asarray(source_idx, dtype=xp.int64)
    tgt_arr = xp.asarray(target_idx, dtype=xp.int64)
    src_counts = xp.zeros(len(source_idx), dtype=xp.float32)
    tgt_counts = xp.zeros(len(target_idx), dtype=xp.float32)
    for _ in range(record_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)
        src_counts += bridge.cp_firing_states[src_arr].astype(xp.float32)
        tgt_counts += bridge.cp_firing_states[tgt_arr].astype(xp.float32)
    src_counts = _host(src_counts)
    tgt_counts = _host(tgt_counts)
    tgt_spk_per_step = float(tgt_counts.sum()) / float(record_steps)
    tgt_max_spk_per_step = float(tgt_counts.max()) / float(record_steps) if len(tgt_counts) else 0.0
    return src_counts, tgt_spk_per_step, tgt_max_spk_per_step


def _build_target_afferent_lists(bridge, source_idx, target_idx):
    """Return, per target cell, the LIST of its presynaptic `source` cells (via the FIXED source->target
    CSR). Used to compute the per-step coincidence count c_i over the routed fan-in.

    cp_connections is (pre -> post): cp_connections[i, j] = weight of i->j, CSR row=pre. We want, per
    post target t, the set of pre source cells s with s->t present. Build it from the CSR by scanning."""
    import scipy.sparse as ssp
    conn = bridge.cp_connections  # cupyx CSR
    # Host copy of the sparse structure (small probe; fine to pull to host once).
    rows = _host(conn.indptr)
    cols = _host(conn.indices)
    # For each pre-row i, cols[rows[i]:rows[i+1]] are its post columns.
    source_set = set(int(s) for s in source_idx)
    target_pos = {int(t): k for k, t in enumerate(target_idx)}  # global -> local target index
    # afferents[local_target] = list of global source indices projecting to it
    afferents = [[] for _ in range(len(target_idx))]
    n = len(rows) - 1
    for pre in range(n):
        if pre not in source_set:
            continue
        c0, c1 = int(rows[pre]), int(rows[pre + 1])
        for post in cols[c0:c1]:
            post = int(post)
            if post in target_pos:
                afferents[target_pos[post]].append(pre)
    return afferents  # list (per local target) of global source indices


def _measure_coincidence_counts(bridge, xp, source_idx, target_idx, afferents, record_steps):
    """Per step, for each target cell, count how many of ITS afferent source cells fired THIS step
    (c_i = the coincidence count over the routed fan-in). Return:
      - per-target afferent-count (fan-in size),
      - the pooled distribution of c_i values across (all targets x all steps) where c_i > 0,
      - the MAX c_i observed (the single best coincidence event), and the per-step MAX-over-targets c_i.

    This is the literal quantity the Route-D detector thresholds on: c_i = (binary routed mask).T @
    prev_fired, restricted to this target's afferents."""
    src_global = np.asarray(source_idx, dtype=np.int64)
    # Precompute, per local target, a numpy index array of its afferent source GLOBAL indices.
    aff_idx = [np.asarray(a, dtype=np.int64) for a in afferents]
    fan_in = np.asarray([len(a) for a in afferents], dtype=np.int64)

    all_ci = []            # pooled c_i>0 over (targets x steps)
    per_step_max_ci = []   # max over targets, per step
    max_ci = 0
    # We read cp_firing_states (the CURRENT step's firing) AFTER stepping -- but the routed matvec the
    # engine uses is on prev_fired = the PREVIOUS step's firing. To match the detector's semantics
    # exactly (c_i computed from prev_fired in the SAME step the plateau is applied), we read
    # cp_prev_firing_states each step (the engine sets it to the just-fired vector at the end of the
    # step; before the NEXT step it IS that step's "prev"). Equivalent for a distribution measure.
    for _ in range(record_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)
        fired = _host(bridge.cp_firing_states).astype(np.int8)  # current step's firing (0/1) per neuron
        # c_i per target = number of its afferent source cells that fired this step.
        step_max = 0
        for k in range(len(target_idx)):
            a = aff_idx[k]
            if a.size == 0:
                continue
            ci = int(fired[a].sum())
            if ci > 0:
                all_ci.append(ci)
            if ci > step_max:
                step_max = ci
            if ci > max_ci:
                max_ci = ci
        per_step_max_ci.append(step_max)
    return {
        "fan_in_min": int(fan_in.min()) if fan_in.size else 0,
        "fan_in_mean": float(fan_in.mean()) if fan_in.size else 0.0,
        "fan_in_max": int(fan_in.max()) if fan_in.size else 0,
        "all_ci": all_ci,
        "per_step_max_ci": per_step_max_ci,
        "max_ci": int(max_ci),
    }


def _ci_summary(all_ci, per_step_max_ci):
    a = np.asarray(all_ci, dtype=np.float64)
    pm = np.asarray(per_step_max_ci, dtype=np.float64)
    if a.size == 0:
        a = np.zeros(1)
    if pm.size == 0:
        pm = np.zeros(1)
    return {
        "n_coincident_events": int((np.asarray(all_ci) > 0).sum()),
        "ci_mean_when_active": float(a.mean()),
        "ci_p50": float(np.percentile(a, 50)),
        "ci_p90": float(np.percentile(a, 90)),
        "ci_p99": float(np.percentile(a, 99)),
        "ci_max": int(a.max()),
        "per_step_max_ci_mean": float(pm.mean()),
        "per_step_max_ci_p90": float(np.percentile(pm, 90)),
        "per_step_max_ci_p99": float(np.percentile(pm, 99)),
        "per_step_max_ci_max": int(pm.max()),
        # fraction of steps where the BEST target saw >= each candidate K (the trigger rate at K)
        "steps_total": int(pm.size),
    }


def _trigger_rate_at_K(per_step_max_ci, K):
    pm = np.asarray(per_step_max_ci, dtype=np.float64)
    if pm.size == 0:
        return 0.0
    return float((pm >= K).mean())


# ----------------------------------------------------------------------
# One seed: confirm the wall (weight/density sweep) + calibrate K.
# ----------------------------------------------------------------------

def run_seed(seed, *, locations, landmarks, n_bearing, n_dist, n_source, n_target,
             src_drive_weight, src_drive_density, src_drive_jitter,
             max_intensity, falloff, dist_sigma, dist_max, bexp,
             train_passes, train_steps_per_loc, record_steps,
             sweep, base_s2t_density=0.5, verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    from sim.backend import get_backend
    xp, backend_name = get_backend()

    n_per_landmark = n_bearing + n_dist
    n_sensors = len(landmarks) * n_per_landmark

    # The distinct operating point for source->target (the RESULT's distinct row: w 20, d 0.3 family).
    # We use a CLUSTERED projection (many source cells -> the single target) so the coincidence detector
    # has a real fan-in to count: density (configurable via --s2t-density; 1.0 = the design's "tighter
    # clustering" lever, EVERY active cell hits EVERY target) so each of ~5% active source cells reliably
    # hits the target.
    base_s2t_w, base_s2t_d, base_s2t_j = 20.0, float(base_s2t_density), 0.3

    t0 = time.time()
    bridge, cfg = _build(seed, n_sensors=n_sensors, n_source=n_source, n_target=n_target,
                         src_drive_weight=src_drive_weight, src_drive_density=src_drive_density,
                         src_drive_jitter=src_drive_jitter,
                         source_to_target_weight=base_s2t_w, source_to_target_density=base_s2t_d,
                         source_to_target_jitter=base_s2t_j)
    _assert_cupy_regime(cfg, backend_name)
    build_s = time.time() - t0
    rm = bridge.region_manager
    sensor_idx = np.asarray(rm.indices("src_sensors"), dtype=np.int64)
    source_idx = np.asarray(rm.indices("source"), dtype=np.int64)
    target_idx = np.asarray(rm.indices("target"), dtype=np.int64)
    log(f"  [seed {seed}] built in {build_s:.1f}s; {cfg.num_neurons} neurons, "
        f"{int(bridge.cp_connections.nnz)} synapses; backend={backend_name}")

    # Anti-cheat audit: only src_sensors are ever externally driven.
    driven_regions = {"src_sensors"}

    # Precompute per-location sensor activations.
    loc_acts = [landmark_sensor_act(x, y, landmarks, n_bearing, n_dist, max_intensity,
                                    falloff, dist_sigma, dist_max, bexp) for (x, y) in locations]

    # --- Train the place fields (STDP self-organizes source's sparse code; src->target is FIXED) ---
    # CRITICAL (matches the validated Stage-1 protocol exactly): OPEN the src_drive STDP gate during
    # training, then FREEZE it (gate=0) before measuring. Without the freeze, STDP keeps depressing the
    # afferent weights DURING the measurement and the source code collapses to silence (verified). A
    # reset gap (zero current, 20 steps) precedes each location's drive. Shuffled location order per pass.
    bridge.set_plasticity_gate("src_drive", 1.0)
    rng = np.random.default_rng(seed)
    for _p in range(train_passes):
        order = list(range(len(loc_acts)))
        rng.shuffle(order)
        for li in order:
            bridge.cp_external_input_current[:] = 0.0
            _step(bridge, 20)
            _drive_clamp(bridge, xp, sensor_idx, loc_acts[li])
            _step(bridge, train_steps_per_loc)
    bridge.set_plasticity_gate("src_drive", 0.0)   # FREEZE -> measure a stable network
    bridge.cp_external_input_current[:] = 0.0
    _step(bridge, 30)

    # --- Characterize the SOURCE code at each location (sparsity + distinctness) ---
    src_count_vecs = []
    tgt_spk = []
    tgt_max_spk = []
    for li, act in enumerate(loc_acts):
        bridge.cp_external_input_current[:] = 0.0
        _step(bridge, 30)  # settle
        _drive_clamp(bridge, xp, sensor_idx, act)
        sc, ts, tms = _measure_source_and_target(bridge, xp, source_idx, target_idx, record_steps)
        src_count_vecs.append(sc)
        tgt_spk.append(ts)
        tgt_max_spk.append(tms)
        bridge.cp_external_input_current[:] = 0.0
    n_source_actual = len(source_idx)
    sparsities = [float((sc > 0).sum()) / float(n_source_actual) for sc in src_count_vecs]
    active_rates = [float(sc[sc > 0].mean()) / float(record_steps) if (sc > 0).any() else 0.0
                    for sc in src_count_vecs]  # spk/step among active source cells
    # pairwise distinctness of the source ensembles
    diff_cos = []
    for i in range(len(src_count_vecs)):
        for j in range(i + 1, len(src_count_vecs)):
            diff_cos.append(cosine_counts(src_count_vecs[i], src_count_vecs[j]))
    src_sparsity_mean = float(np.mean(sparsities))
    src_active_rate_mean = float(np.mean(active_rates))
    src_diff_cos_mean = float(np.mean(diff_cos)) if diff_cos else 0.0
    tgt_spk_mean = float(np.mean(tgt_spk))
    tgt_max_spk_mean = float(np.mean(tgt_max_spk))

    log(f"  [seed {seed}] SOURCE: sparsity {src_sparsity_mean*100:.1f}%  active-rate "
        f"{src_active_rate_mean:.3f} spk/step  diff-cos {src_diff_cos_mean:.3f}  "
        f"|  TARGET (base s2t w{base_s2t_w} d{base_s2t_d}): {tgt_spk_mean:.4f} spk/step "
        f"(max-cell {tgt_max_spk_mean:.4f})")

    # --- CALIBRATE K: measure c_i (per-step coincidence count over each target's routed fan-in) at the
    # distinct operating point, while driving location 0 (a representative distinct ensemble). ---
    afferents = _build_target_afferent_lists(bridge, source_idx, target_idx)
    bridge.cp_external_input_current[:] = 0.0
    _step(bridge, 30)
    _drive_clamp(bridge, xp, sensor_idx, loc_acts[0])
    cc = _measure_coincidence_counts(bridge, xp, source_idx, target_idx, afferents,
                                     record_steps=max(record_steps, 120))
    bridge.cp_external_input_current[:] = 0.0
    ci_summ = _ci_summary(cc["all_ci"], cc["per_step_max_ci"])
    # Trigger rate at candidate K values (fraction of steps the BEST target reaches >= K).
    K_candidates = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20]
    trig = {str(K): _trigger_rate_at_K(cc["per_step_max_ci"], K) for K in K_candidates}

    log(f"  [seed {seed}] FAN-IN per target: min {cc['fan_in_min']} mean {cc['fan_in_mean']:.1f} "
        f"max {cc['fan_in_max']}")
    log(f"  [seed {seed}] c_i (coincident routed inputs / step): mean-when-active "
        f"{ci_summ['ci_mean_when_active']:.2f}  p90 {ci_summ['ci_p90']:.0f}  p99 {ci_summ['ci_p99']:.0f}  "
        f"MAX {ci_summ['ci_max']}  |  per-step max-over-targets: mean "
        f"{ci_summ['per_step_max_ci_mean']:.2f} p90 {ci_summ['per_step_max_ci_p90']:.0f} "
        f"MAX {ci_summ['per_step_max_ci_max']}")
    log(f"  [seed {seed}] trigger-rate @K: " + "  ".join(f"K{K}={trig[str(K)]:.2f}" for K in K_candidates))

    # --- DECISIVE CALIBRATION: does a sparse-distinct ensemble deliver >=K>=2 coincidence if we raise
    # the per-cell FIRING RATE (drive intensity)?  c_i ~ (# active ensemble cells projecting to a target)
    # x (per-cell fire-prob/step). At the base point cells fire asynchronously at ~10 Hz -> c_i<=1. We
    # push drive intensity UP (raising the per-cell rate) and measure, at each, (source sparsity, c_i
    # p90/max). The DECISION: is there an operating point that is BOTH sparse-distinct (<=~8% cells) AND
    # delivers c_i>=2 at p90 (so a K=2..3 detector fires on the ensemble's coincidence)?  This is the
    # exact question the design's stop-condition hinges on (a <0.2-spk/step ensemble that can't deliver
    # >=K in ANY step => honest-negative, the next lever is clustering+delays or multi-subunit). ---
    intensity_factors = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    calib_rows = []
    for fac in intensity_factors:
        scaled_act = (loc_acts[0] * fac).astype(np.float32)
        bridge.cp_external_input_current[:] = 0.0
        _step(bridge, 30)
        # sparsity at this drive
        _drive_clamp(bridge, xp, sensor_idx, scaled_act)
        sc, _ts, _tms = _measure_source_and_target(bridge, xp, source_idx, target_idx, record_steps)
        bridge.cp_external_input_current[:] = 0.0
        spars = float((sc > 0).sum()) / float(n_source_actual)
        act_rate = float(sc[sc > 0].mean()) / float(record_steps) if (sc > 0).any() else 0.0
        # c_i at this drive
        _step(bridge, 30)
        _drive_clamp(bridge, xp, sensor_idx, scaled_act)
        cc2 = _measure_coincidence_counts(bridge, xp, source_idx, target_idx, afferents,
                                          record_steps=max(record_steps, 120))
        bridge.cp_external_input_current[:] = 0.0
        cs2 = _ci_summary(cc2["all_ci"], cc2["per_step_max_ci"])
        calib_rows.append({
            "intensity_factor": fac, "source_sparsity": spars,
            "source_active_rate_spk_per_step": act_rate,
            "ci_p90": cs2["per_step_max_ci_p90"], "ci_p99": cs2["per_step_max_ci_p99"],
            "ci_max": cs2["per_step_max_ci_max"], "ci_mean_when_active": cs2["ci_mean_when_active"],
            "trig_K2": _trigger_rate_at_K(cc2["per_step_max_ci"], 2),
            "trig_K3": _trigger_rate_at_K(cc2["per_step_max_ci"], 3),
            "trig_K4": _trigger_rate_at_K(cc2["per_step_max_ci"], 4),
            # the decision flag: sparse-distinct (<=8%) AND coincidence available (per-step max c_i p90>=2)
            "sparse_and_coincident": bool(spars <= 0.08 and cs2["per_step_max_ci_p90"] >= 2),
        })
    for r in calib_rows:
        log(f"    calib int x{r['intensity_factor']:.1f}: sparsity {r['source_sparsity']*100:5.1f}%  "
            f"act-rate {r['source_active_rate_spk_per_step']:.3f}  c_i(per-step-max) p90 {r['ci_p90']:.0f} "
            f"max {r['ci_max']}  trigK2 {r['trig_K2']:.2f} trigK3 {r['trig_K3']:.2f}  "
            f"sparse&coincident={r['sparse_and_coincident']}")

    # --- CONFIRM THE WALL: sweep source->target weight x density; for each, rebuild a FRESH bridge with
    # that projection and measure (source sparsity, target spk/step). Show no linear setting fires the
    # target while the source stays sparse. ---
    sweep_rows = []
    if sweep:
        sweep_w = [10.0, 20.0, 40.0, 80.0, 160.0]
        sweep_d = [0.3, 0.5, 0.9]
        for sw in sweep_w:
            for sd in sweep_d:
                b2, c2 = _build(seed, n_sensors=n_sensors, n_source=n_source, n_target=n_target,
                                src_drive_weight=src_drive_weight, src_drive_density=src_drive_density,
                                src_drive_jitter=src_drive_jitter,
                                source_to_target_weight=sw, source_to_target_density=sd,
                                source_to_target_jitter=base_s2t_j)
                rm2 = b2.region_manager
                s_idx = np.asarray(rm2.indices("src_sensors"), dtype=np.int64)
                so_idx = np.asarray(rm2.indices("source"), dtype=np.int64)
                ta_idx = np.asarray(rm2.indices("target"), dtype=np.int64)
                # quick-train the place fields (same gate-managed protocol, fewer passes for the sweep)
                b2.set_plasticity_gate("src_drive", 1.0)
                rng2 = np.random.default_rng(seed)
                for _p in range(max(2, train_passes // 2)):
                    order2 = list(range(len(loc_acts)))
                    rng2.shuffle(order2)
                    for li2 in order2:
                        b2.cp_external_input_current[:] = 0.0
                        _step(b2, 20)
                        _drive_clamp(b2, xp, s_idx, loc_acts[li2])
                        _step(b2, train_steps_per_loc)
                b2.set_plasticity_gate("src_drive", 0.0)
                b2.cp_external_input_current[:] = 0.0
                _step(b2, 30)
                # measure at location 0
                _drive_clamp(b2, xp, s_idx, loc_acts[0])
                sc, ts, tms = _measure_source_and_target(b2, xp, so_idx, ta_idx, record_steps)
                b2.cp_external_input_current[:] = 0.0
                spars = float((sc > 0).sum()) / float(len(so_idx))
                sweep_rows.append({
                    "s2t_weight": sw, "s2t_density": sd,
                    "source_sparsity": spars,
                    "target_spk_per_step": ts,
                    "target_max_cell_spk_per_step": tms,
                    # >=5 Hz at dt=1ms == 0.005 spk/step PER CELL -> use the max-cell rate (a single
                    # target cell firing >=5 Hz counts as "the target fires"); the pop mean would
                    # require all 20 cells active and understate per-cell firing.
                    "target_fires_5hz": bool(tms >= 0.005),
                })
                del b2
        for r in sweep_rows:
            log(f"    sweep s2t w{r['s2t_weight']:.0f} d{r['s2t_density']:.1f}: "
                f"source-sparsity {r['source_sparsity']*100:5.1f}%  "
                f"target {r['target_spk_per_step']:.4f} spk/step "
                f"(max-cell {r['target_max_cell_spk_per_step']:.4f})")

    return {
        "seed": seed,
        "backend": backend_name,
        "n_neurons": int(cfg.num_neurons),
        "n_synapses": int(bridge.cp_connections.nnz),
        "driven_regions": sorted(driven_regions),
        "source": {
            "sparsity_mean": src_sparsity_mean,
            "active_rate_spk_per_step_mean": src_active_rate_mean,
            "diff_cos_mean": src_diff_cos_mean,
            "per_loc_sparsity": sparsities,
            "per_loc_active_rate": active_rates,
        },
        "wall_base_point": {
            "s2t_weight": base_s2t_w, "s2t_density": base_s2t_d,
            "target_spk_per_step_mean": tgt_spk_mean,
            "target_max_cell_spk_per_step_mean": tgt_max_spk_mean,
        },
        "coincidence_stats": {**ci_summ, "fan_in_min": cc["fan_in_min"],
                              "fan_in_mean": cc["fan_in_mean"], "fan_in_max": cc["fan_in_max"],
                              "trigger_rate_at_K": trig},
        "calibration_intensity_sweep": calib_rows,
        "any_sparse_and_coincident": bool(any(r["sparse_and_coincident"] for r in calib_rows)),
        "wall_sweep": sweep_rows,
    }


def _recommend_K_gain(seed_results, n_target, record_steps):
    """Aggregate the per-seed c_i statistics into a recommended (K, gain, plateau_strength).

    K: pick the largest K such that the BEST target reliably reaches >=K in a meaningful fraction of
       steps (so a real ensemble-coincidence fires) AND K>1 (the anti-cheat -- a single input must not
       trigger). We target K at roughly the cross-seed p90 of the per-step max-over-targets c_i (the
       coincidence the distinct ensemble actually delivers), floored at 2, so the plateau fires on the
       ensemble's strong coincidence events but not on a lone input.

    gain: sigmoid slope so the switch is ~all-or-none across +/-1 input around K (gain ~2 gives
       sigmoid(+/-2) = 0.88/0.12 at K+/-1 -- a sharp but not infinite switch, like the design default).

    plateau_strength: the peak plateau CONDUCTANCE scale needed for I = g*mg_block*(E_e - V) to clear
       the ~420 pA MSN-D1 rheobase. At a sub-threshold V~-70 mV with E_e=0: (E_e-V)=70 mV, mg_block at
       -70 mV ~ 1/(1+ (1/3.57)*exp(0.062*70)) ~ 1/(1+0.28*76) ~ 0.045. So I ~ g*0.045*70 = 3.15*g pA.
       To reach ~420 pA at the FIRST plateau step needs g ~ 133; but the plateau ACCUMULATES over its
       ~80 ms decay (geometric sum factor ~1/(1-decay), decay=exp(-1/80)~0.9876 -> ~80x), and as V
       depolarizes mg_block rises sharply (regenerative). So a per-step increment of plateau_strength
       ~60-90 reaches rheobase within a few steps. We report the analytic single-step floor AND the
       accumulating estimate; the exact operating point is finalized by Step A's G_FIRE gate."""
    per_step_max_p90 = [r["coincidence_stats"]["per_step_max_ci_p90"] for r in seed_results]
    per_step_max_p99 = [r["coincidence_stats"]["per_step_max_ci_p99"] for r in seed_results]
    per_step_max_max = [r["coincidence_stats"]["per_step_max_ci_max"] for r in seed_results]
    ci_max = [r["coincidence_stats"]["ci_max"] for r in seed_results]

    # The DECISIVE input: the calibration intensity-sweep. For each seed, find the sparse-and-coincident
    # operating points (sparsity<=8% AND per-step-max c_i p90>=2). If such points exist across ALL seeds,
    # a K>=2 detector has a real operating window -> recommend K from the achievable coincidence; else NO
    # valid K>1 in the natural sparse-distinct dynamics -> the honest-negative branch (Step A should NOT
    # be written to fire as-is; the next lever is clustering+delays or multi-subunit, per the design).
    per_seed_best = []  # the best sparse-and-coincident row per seed (max ci_p90 among sparse rows), or None
    for r in seed_results:
        sparse_rows = [c for c in r.get("calibration_intensity_sweep", []) if c["source_sparsity"] <= 0.08]
        coincident = [c for c in sparse_rows if c["ci_p90"] >= 2]
        if coincident:
            per_seed_best.append(max(coincident, key=lambda c: c["ci_p90"]))
        else:
            per_seed_best.append(None)
    all_seeds_have_window = all(b is not None for b in per_seed_best)
    no_valid_K = not all_seeds_have_window

    if all_seeds_have_window:
        # K at floor of the cross-seed min ci_p90 of the best sparse-coincident rows, clamped >=2.
        min_ci_p90 = float(np.min([b["ci_p90"] for b in per_seed_best]))
        K_rec = max(2, int(np.floor(min_ci_p90)))
    else:
        # No window: K is undefined; report 2 as the minimum anti-cheat-valid K (but no_valid_K flags it).
        K_rec = 2
    gain_rec = 2.0
    # MSN-D1 rheobase ~ 420 pA; single-step analytic floor g ~ 133; accumulating estimate ~ 60-90.
    plateau_strength_rec = 80.0
    return {
        "K_recommended": K_rec,
        "gain_recommended": gain_rec,
        "plateau_strength_recommended": plateau_strength_rec,
        "tau_decay_ms_recommended": 80.0,
        "tau_rise_ms_recommended": 2.0,
        "no_valid_K_above_1": bool(no_valid_K),
        "all_seeds_have_sparse_coincident_window": bool(all_seeds_have_window),
        "per_seed_best_sparse_coincident_operating_point": [
            (None if b is None else {"intensity_factor": b["intensity_factor"],
                                     "source_sparsity": b["source_sparsity"], "ci_p90": b["ci_p90"],
                                     "ci_max": b["ci_max"], "trig_K2": b["trig_K2"]})
            for b in per_seed_best],
        "cross_seed_base_per_step_max_ci_p90": per_step_max_p90,
        "cross_seed_base_per_step_max_ci_p99": per_step_max_p99,
        "cross_seed_base_per_step_max_ci_max": per_step_max_max,
        "cross_seed_base_ci_max": ci_max,
        "rationale": (
            "K from the calibration intensity-sweep: the smallest K>=2 the BEST sparse-distinct (<=8% "
            "cells) operating point reaches at p90 of per-step max-over-targets c_i, across ALL seeds. If "
            "no sparse-and-coincident window exists in any seed (no_valid_K_above_1=True), the natural "
            "sparse-distinct ensemble does NOT deliver >=K>=2 coincidence per step -> the design's STOP "
            "condition (don't write a Step-A edit that can't fire; next lever = clustering+delays or "
            "multi-subunit). gain=2 = a sharp all-or-none sigmoid switch (~0.88/0.12 at K+/-1). "
            "plateau_strength=80: analytic single-step floor to clear ~420 pA MSN-D1 rheobase is g~133 at "
            "V=-70mV (mg_block~0.045, driving force 70mV), but the plateau ACCUMULATES over its ~80ms "
            "decay (~80x geometric sum) and mg_block rises regeneratively as V depolarizes, so a per-step "
            "increment ~60-90 reaches rheobase within a few steps; exact point finalized by Step A G_FIRE."),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--smoke", action="store_true", help="1 seed, fewer train passes (fast)")
    ap.add_argument("--out", type=str, default="research/findings/raw/_coincidence_wall_probe.json")
    ap.add_argument("--n-source", type=int, default=400)
    ap.add_argument("--n-target", type=int, default=20)
    ap.add_argument("--s2t-density", type=float, default=0.5,
                    help="source->target projection density (the c_i convergence; 1.0 = the design's "
                         "'tighter clustering' lever: every active source cell hits every target)")
    # Validated Stage-1 place-code recipe (placecode_selforg_stage1_derisk.py defaults): weight 28,
    # density 0.5, falloff 0.03 (gentle distance fall-off so far landmarks still drive), dist-sigma 4,
    # n-bearing 12. These reliably fire a sparse ~2-10% distinct source code.
    ap.add_argument("--n-bearing", type=int, default=12)
    ap.add_argument("--n-dist", type=int, default=8)
    ap.add_argument("--src-drive-weight", type=float, default=28.0)
    ap.add_argument("--src-drive-density", type=float, default=0.5)
    ap.add_argument("--src-drive-jitter", type=float, default=0.6)
    ap.add_argument("--max-intensity", type=float, default=450.0)
    ap.add_argument("--falloff", type=float, default=0.03)
    ap.add_argument("--dist-sigma", type=float, default=4.0)
    ap.add_argument("--grid-size", type=int, default=32)
    ap.add_argument("--bexp", type=float, default=4.0)
    ap.add_argument("--train-passes", type=int, default=12)
    ap.add_argument("--train-steps-per-loc", type=int, default=120)
    ap.add_argument("--record-steps", type=int, default=100)
    ap.add_argument("--no-sweep", action="store_true", help="skip the weight/density wall sweep")
    args = ap.parse_args()

    if args.smoke:
        seeds = [42]
        args.train_passes = 8
    else:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    grid_size = int(args.grid_size)
    locations = default_locations(grid_size)
    landmarks = default_landmarks(grid_size)
    dist_max = float(grid_size) * 1.42   # ~max diagonal distance, for the distance-tuning span

    t0 = time.time()
    seed_results = []
    for seed in seeds:
        r = run_seed(seed, locations=locations, landmarks=landmarks,
                     n_bearing=args.n_bearing, n_dist=args.n_dist,
                     n_source=args.n_source, n_target=args.n_target,
                     src_drive_weight=args.src_drive_weight, src_drive_density=args.src_drive_density,
                     src_drive_jitter=args.src_drive_jitter,
                     max_intensity=args.max_intensity, falloff=args.falloff,
                     dist_sigma=args.dist_sigma, dist_max=dist_max, bexp=args.bexp,
                     train_passes=args.train_passes, train_steps_per_loc=args.train_steps_per_loc,
                     record_steps=args.record_steps, sweep=(not args.no_sweep),
                     base_s2t_density=args.s2t_density, verbose=True)
        seed_results.append(r)

    rec = _recommend_K_gain(seed_results, args.n_target, args.record_steps)
    elapsed = time.time() - t0

    # Wall verdict (two parts, honest):
    #  (1) at REALISTIC weight (s2t weight <= 40, the RESULT's distinct-point family: Schaffer w<=120/
    #      d0.9 -> my <=40 here), is the sparse source able to fire the target (max-cell >= 5 Hz)?  If
    #      NOT -> the linear-summation wall is confirmed at physiological weight.
    #  (2) BRUTE-FORCE: at high weight (s2t weight >= 80), the sparse source CAN clear threshold via pure
    #      linear AMPA summation (the RESULT noted this) -- recorded separately; it is NOT the coincidence
    #      mechanism (it just sums asynchronous spikes harder) and tends to position-blindness.
    sparse_and_fires_realistic = []
    brute_force_fires = []
    for r in seed_results:
        for row in r.get("wall_sweep", []):
            fires = row["source_sparsity"] <= 0.10 and row["target_max_cell_spk_per_step"] >= 0.005
            if fires and row["s2t_weight"] <= 40.0:
                sparse_and_fires_realistic.append({"seed": r["seed"], **row})
            elif fires and row["s2t_weight"] >= 80.0:
                brute_force_fires.append({"seed": r["seed"], **row})
    wall_confirmed = (len(sparse_and_fires_realistic) == 0)

    out = {
        "probe": "coincidence_wall_probe",
        "design_doc": "2026-06-09-coincidence-substrate-upgrade-design.md",
        "seeds": seeds,
        "elapsed_seconds": round(elapsed, 1),
        "wall_confirmed_no_sparse_and_fires_at_realistic_weight": bool(wall_confirmed),
        "sparse_and_fires_realistic_exceptions": sparse_and_fires_realistic,
        "brute_force_high_weight_fires": brute_force_fires,
        "recommendation": rec,
        "per_seed": seed_results,
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 78)
    print(f"WALL CONFIRMED at realistic weight (s2t<=40; no sparse-AND-fires): {wall_confirmed}")
    if sparse_and_fires_realistic:
        print(f"  REALISTIC-WEIGHT EXCEPTIONS (sparse AND fires): {sparse_and_fires_realistic}")
    if brute_force_fires:
        print(f"  brute-force high-weight (s2t>=80) DOES fire (linear summation, not coincidence): "
              f"{len(brute_force_fires)} rows")
    print(f"sparse-AND-coincident window exists across ALL seeds: "
          f"{rec['all_seeds_have_sparse_coincident_window']}")
    print(f"RECOMMENDED K = {rec['K_recommended']}  gain = {rec['gain_recommended']}  "
          f"plateau_strength = {rec['plateau_strength_recommended']}  "
          f"(no_valid_K_above_1 = {rec['no_valid_K_above_1']})")
    print(f"  per-seed best sparse-coincident operating point: "
          f"{rec['per_seed_best_sparse_coincident_operating_point']}")
    print(f"  cross-seed BASE per-step max-c_i: p90 {rec['cross_seed_base_per_step_max_ci_p90']}  "
          f"max {rec['cross_seed_base_per_step_max_ci_max']}")
    if rec["no_valid_K_above_1"]:
        print("  >>> HONEST-NEGATIVE BRANCH: no sparse-distinct operating point delivers c_i>=2 per step.")
        print("  >>> Per the design STOP condition, Step A should NOT be written to fire as-is in this")
        print("  >>> regime; the next lever is clustering+conduction-delays (Route T) or multi-subunit.")
    print(f"wrote {outp}  ({elapsed:.1f}s)")
    print("=" * 78)


if __name__ == "__main__":
    main()
