"""Route T Step-0 -- RUNNER-ONLY (NO sim/ edit): does a GAMMA SYNCHRONIZING VOLLEY make the sparse-distinct
place ensemble fire the Route-D coincidence target, and does JITTER collapse it (proving it is the
synchronized volley, not rate)?

RESULT (2026-06-09, 3 CuPy seeds 42/43/44 -- see 2026-06-09-route-T-gamma-volley-RESULT.md): VERDICT
PARTIAL. Both synchronizers fire the MSN-D1 AND collapse under jitter (0.0 Hz, 3/3 -> coincidence, not
rate); Route D is load-bearing (ablate -> silent, 3/3); the volley packs into ONE dt step (G_VOLLEY 3/3
-> conduction delays NOT needed here). Host-pacing fires 3/3 (11-22 Hz) but densifies the place code
(~18-19%, G_SPARSE 0/3). The brain-based FS-PING is the BETTER synchronizer -- sparse-preserving
(5.9-7.2%) and fires 2/3 (11-15 Hz vs the ~3-6 Hz async chance floor) -- but marginally exceeds the
<=5% bar and is seed-variable. Not a clean GO; the named next step is tightening FS-PING (+ intrinsic
place-pool homeostasis), NOT a host teacher and NOT the conduction-delay sim/ edit.

Design: research/findings/2026-06-09-route-T-volley-synchronization-design.md (THE design).
Solves the ASYNCHRONY wall mapped by coincidence_wall_probe.py + _coincidence_wall_probe.json:
the sparse ~10 Hz place pool has NO synchronizer (internal_density=0, no FS) -> asynchronous ->
per-step coincidence c_i <= 1 -> Route D has nothing to detect (3 CuPy seeds: c_i p90 0-1, max 2,
no_valid_K_above_1=true). The fix is a GAMMA RHYTHM that re-times the CURRENTLY-ACTIVE place cells
into ONE gamma window (sets WHEN they fire, not WHICH -> distinctness preserved, NOT densified) ->
their per-step coincidence jumps >= K -> the landed Route-D plateau (b980070a) fires the downstream
MSN-D1.

TWO synchronizers tested (--sync):
  pacing : (a documented HOST scaffold, SH-5) a gamma-band pulse train added to cp_external_input_current
           on the PLACE pool (location-BLIND: a uniform depolarizing pulse on every place cell at the
           gamma peak -> the already-depolarized (active) cells cross threshold TOGETHER on the pulse
           step, sub-threshold (inactive) cells stay silent). Validates the MECHANISM. A host-imposed
           rhythm = a shortcut; labelled as such. The pacing pulse is location-blind, so it sets the
           TIMING window, the place code still selects WHICH cells (coincidence selection stays neural,
           no host teacher into the code).
  ping   : (the FAITHFUL, brain-based version) FS-PING gamma: an FS interneuron pool on the place pool
           (place->FS exc, FS->place GABA_A) so a gamma rhythm EMERGES from the pool's own recurrent
           inhibition (the engine already produces FS-PING gamma -- the gamma-oscillations benchmark /
           CORTEX_GAMMA_FS_NETWORK). The active place cells excite FS, FS GABA_A silences the pool for
           ~one GABA_A decay, then releases -> the active cells re-fire TOGETHER each gamma cycle. No
           host pacing at all -- synchrony from neurons + synapses. Still location-blind (FS sees only
           the active place cells; it sets the timing, the place code selects which).

The DECISIVE PAIR is run FIRST (before the full gate sweep), CuPy, >=3 seeds, deterministic regime:
  G_FIRE : with the synchronizer + Route D ON, the sparse-distinct ensemble fires the downstream MSN-D1
           >= 5 Hz (vs the c_i<=1 baseline's 0.0). THE HEADLINE.
  JITTER (--jitter-inputs, THE load-bearing anti-cheat, Branco-Hausser): de-synchronize/jitter the
           volley -> firing must COLLAPSE (proves coincidence, not rate). If firing survives jitter,
           it is rate not coincidence -> NOT a real volley win.

Then the full gates (CuPy; >=3 seeds; deterministic regime OU/cond-noise/global-homeostasis/
heterogeneity/STP OFF; backend=="cupy" -- numpy DISQUALIFIED):
  G_SPARSE   : the place ensemble stays sparse-distinct WITH the rhythm on (<= 5% active, diff-loc
               cos < 0.30) -- the volley did NOT densify the code (the property the densify-to-coincide
               approach destroyed). Asserted vs the rhythm-OFF code: sparsity + diff-cos preserved.
  G_VOLLEY   : the active ensemble's per-step max c_i reaches >= K in the volley step (vs Step-0's <=1)
               -- the rhythm actually synchronizes the sparse ensemble into a coincident packet. If
               the volley spans MULTIPLE dt steps (G_VOLLEY fails) -> the precise honest trigger for the
               deferred conduction-delay sim/ ring buffer (reported, NOT built).
  G_FIRE     : (above) MSN-D1 >= 5 Hz from the sparse-distinct ensemble.
  G_DISTINCT : downstream firing stays position-specific (near >> far >= 3x; downstream per-cell
               diff-cos < 0.30) -- it fired by WHICH cells coincided, not by going dense.
  G_MSN      : the place/CA1 effective drive >= ~420 pA -> MSN >= 5 Hz (the N9 striatal-critic read-out).

ANTI-CHEATS (each MUST behave consistently for an honest pass):
  - JITTER (--jitter-inputs, decisive): de-sync the volley -> G_FIRE COLLAPSES (else it's rate).
  - REMOVE-THE-RHYTHM (--no-rhythm: pacing/FS OFF, Route D still ON) -> G_VOLLEY + G_FIRE FAIL (back
    to c_i<=1, MSN silent -- reproducing Step-0). Confirms the rhythm is load-bearing (Route D alone
    is a no-op here).
  - ABLATE Route D (--ablate-subunit: enable_coincidence_detection=False) with the rhythm ON -> still
    NO firing at realistic weight (a synchronized volley of K SUB-threshold AMPA inputs without the
    supralinear plateau must not fire -- confirms BOTH halves are needed, the volley isn't just rate).
  - K > 1 (--k-sweep): coincidence_k_threshold must be > 1 (a single coincident input must not fire the
    plateau, else it's a per-synapse gain). G_DISTINCT must hold across the K that passes G_FIRE.
  - NO HOST TEACHER: the ONLY cp_external_input_current writes target the sensory afferent (src_sensors)
    AND, for pacing, the location-BLIND uniform gamma pulse on the place pool (it cannot encode WHICH
    cells fire -- it only sets WHEN the already-selected cells fire). Audited (driven-regions recorded;
    the pacing pulse is asserted location-independent: the SAME pulse vector regardless of location).
  - CuPy regime: backend=="cupy" (numpy DISQUALIFIED per 2026-06-09-N9-cupy-membrane-divergence-ROOT);
    deterministic knobs OFF; no per-region homeostasis on the MSN target (it must fire from the
    coincidence current, not threshold collapse). Hard-asserted.

USAGE (MUST be cupy):
  # decisive pair (default sync=pacing):
  SIM_BACKEND=cupy python -m research.runners.coincidence_volley_n9_derisk \
      --seeds 42,43,44 --k-threshold 4 --out research/findings/raw/_coincidence_volley_n9_derisk.json
  # jitter anti-cheat (must collapse):
  SIM_BACKEND=cupy python -m research.runners.coincidence_volley_n9_derisk --seeds 42,43,44 --jitter-inputs
  # remove-the-rhythm control (must reproduce the wall):
  SIM_BACKEND=cupy python -m research.runners.coincidence_volley_n9_derisk --seeds 42,43,44 --no-rhythm
  # ablate Route D with rhythm on (must NOT fire):
  SIM_BACKEND=cupy python -m research.runners.coincidence_volley_n9_derisk --seeds 42,43,44 --ablate-subunit
  # FS-PING (faithful, brain-based synchronizer):
  SIM_BACKEND=cupy python -m research.runners.coincidence_volley_n9_derisk --seeds 42,43,44 --sync ping
  # K-sweep (K>1):
  SIM_BACKEND=cupy python -m research.runners.coincidence_volley_n9_derisk --seeds 42 --k-sweep 2,4,6,8
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

# Reuse the Step-0 probe's grid + sensor render + helpers (the SAME isolated source->target topology;
# this file adds a SYNCHRONIZER to the source pool but makes NO sim/ edit -- it only toggles existing
# config flags / adds existing-field regions+pathways on the bridge the builder constructs).
from research.runners.coincidence_wall_probe import default_locations, default_landmarks
from research.runners.placecode_selforg_stage1_derisk import (
    landmark_sensor_act, cosine_counts, _host)


# ──────────────────────────────────────────────────────────────────────
# Build the isolated bridge: src_sensors -> source(place WTA) -> target(MSN-D1, Route-D coincidence
# afferent). Optionally add an FS-PING pool on `source` (--sync ping). Mirrors coincidence_wall_probe._build
# + coincidence_n9_derisk._build_with_coincidence (the Route-D flags), with the optional FS pool added.
# NO sim/ edits -- only existing config fields / existing-field regions+pathways.
# ──────────────────────────────────────────────────────────────────────

def _build(seed, *, n_sensors, n_source, n_target, src_drive_weight, src_drive_density, src_drive_jitter,
           s2t_weight, s2t_density, s2t_jitter, k_threshold, gain, plateau_strength,
           enable_coincidence=True, sync="pacing", n_fs=80,
           place_to_fs_weight=24.0, place_to_fs_density=0.6,
           fs_to_place_weight=14.0, fs_to_place_density=0.6, no_rhythm=False, dt_ms=1.0):
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType

    regions = [
        BrainRegion(name="src_sensors", n_neurons=int(n_sensors), exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name),
        BrainRegion(name="source", n_neurons=int(n_source), exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name),
        # MSN-D1 target: enable_nmda=True so the coincidence subunit's Mg2+ block is active; depolarized
        # E_GABA (-60) like the C1/N9 build.
        BrainRegion(name="target", n_neurons=int(n_target), exc_fraction=0.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D1.name,
                    syn_reversal_potential_i_override=-60.0, enable_nmda=True),
    ]
    pathways = [
        RegionPathway(from_region="src_sensors", to_region="source", density=float(src_drive_density),
                      weight_mean=float(src_drive_weight), weight_jitter=float(src_drive_jitter),
                      plastic=True, plasticity_gate="src_drive"),
        # === THE COINCIDENCE AFFERENT (Route D, landed b980070a) ===
        RegionPathway(from_region="source", to_region="target", density=float(s2t_density),
                      weight_mean=float(s2t_weight), weight_jitter=float(s2t_jitter),
                      plastic=False, coincidence_detector=bool(enable_coincidence)),
    ]

    if sync == "ping" and not no_rhythm:
        # FS-PING gamma generator: an FS interneuron pool reciprocally wired to the place pool. The
        # active place cells excite FS; FS GABA_A inhibition silences the pool for ~one GABA_A decay;
        # release -> the active cells re-fire TOGETHER each gamma cycle. Mirrors CORTEX_GAMMA_FS_NETWORK
        # (IZH2007_FS_CORTICAL_INTERNEURON, reciprocal exc/inh, high density). Location-blind: FS sees
        # only the currently-active place cells (it sets WHEN, the place code selects WHICH).
        # --no-rhythm OMITS this pool entirely (for ping) -> the source reverts to the asynchronous
        # wall (the load-bearing remove-the-rhythm control: firing must then collapse).
        regions.append(
            BrainRegion(name="place_fs", n_neurons=int(n_fs), exc_fraction=0.0, internal_density=0.0,
                        exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                        izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name))
        pathways.append(
            RegionPathway(from_region="source", to_region="place_fs", density=float(place_to_fs_density),
                          weight_mean=float(place_to_fs_weight), weight_jitter=0.2, plastic=False))
        pathways.append(
            RegionPathway(from_region="place_fs", to_region="source", density=float(fs_to_place_density),
                          weight_mean=float(fs_to_place_weight), weight_jitter=0.2, plastic=False))

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
    cfg.enable_nmda = True   # the per-region mask restricts NMDA to the target (enable_nmda=True there)
    cfg.stdp_w_max = 40.0
    cfg.fast_spike_reset = True
    # GABA_A decay sets the gamma frequency in the PING regime (gamma freq ~ 1/tau_g_i). The default
    # syn_tau_g_i is fine; keep the engine default so the rhythm is the engine's own.
    # === Route D (landed) ===
    cfg.enable_coincidence_detection = bool(enable_coincidence)
    cfg.coincidence_k_threshold = float(k_threshold)
    cfg.coincidence_gain = float(gain)
    cfg.coincidence_plateau_strength = float(plateau_strength)
    # === deterministic-nav regime ===
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
            f"REGIME FIDELITY: Route-T volley de-risk MUST run on CuPy (numpy DISQUALIFIED). "
            f"Got backend={backend_name!r}. Set SIM_BACKEND=cupy.")
    bad = [k for k in ("enable_ou_process", "enable_conductance_noise", "enable_homeostasis",
                       "enable_parameter_heterogeneity", "enable_short_term_plasticity")
           if getattr(cfg, k, False)]
    if bad:
        raise AssertionError(f"REGIME FIDELITY: deterministic-regime knobs left ON: {bad}")


# ──────────────────────────────────────────────────────────────────────
# Drive / step helpers
# ──────────────────────────────────────────────────────────────────────

def _step(bridge, n):
    for _ in range(n):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)


def _set_clamp(bridge, xp, idx_gpu, vec_gpu):
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[idx_gpu] = vec_gpu


def _afferent_lists(bridge, source_idx, target_idx):
    """Per target cell, the list of presynaptic `source` cells (via the source->target CSR). Used to
    compute the per-step coincidence count c_i over the routed fan-in (the literal quantity Route D
    thresholds on)."""
    conn = bridge.cp_connections
    rows = _host(conn.indptr)
    cols = _host(conn.indices)
    source_set = set(int(s) for s in source_idx)
    target_pos = {int(t): k for k, t in enumerate(target_idx)}
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
    return [np.asarray(a, dtype=np.int64) for a in afferents]


def _gamma_pulse_step(step_i, *, gamma_hz, dt_ms, amp_pA, duty):
    """Location-BLIND gamma pacing: a uniform depolarizing pulse on EVERY place cell, ON for `duty`
    fraction of each gamma period, else 0. The pulse vector is identical regardless of location (it
    cannot encode WHICH cells fire -- it only sets WHEN). Returns amp_pA on a pulse step, 0 otherwise."""
    period_steps = max(1.0, (1000.0 / gamma_hz) / dt_ms)   # e.g. 40 Hz, dt=1ms -> 25 steps
    phase = (step_i % period_steps) / period_steps          # 0..1 within the cycle
    return float(amp_pA) if phase < duty else 0.0


def _measure_volley(bridge, xp, sensor_idx, source_idx, target_idx, afferents, sensor_act, record_steps,
                    *, sync, gamma_hz, dt_ms, pace_amp_pA, pace_duty, jitter, no_rhythm,
                    pace_offset_steps=0):
    """Run `record_steps` with the sensor clamp held + the synchronizer active, recording:
      - source per-cell spike counts (for sparsity + distinctness),
      - target per-cell spike counts (firing + downstream distinctness),
      - per-step c_i = max-over-targets coincidence count over the routed fan-in (the volley signal),
      - the pacing-pulse vector (audited location-blind).

    sync='pacing' adds the location-blind gamma pulse to the place pool each pulse step (on top of the
    sensor clamp on the sensor indices -- disjoint index sets, so they compose; the place code reads the
    sensor-routed synaptic current, the pulse sets the timing). sync='ping' relies on the FS pool (no
    external place drive). --no-rhythm disables BOTH (reproduces the asynchronous wall). --jitter spreads
    the SENSOR drive across alternating steps (de-synchronizes the source) so the volley is destroyed."""
    src_arr = xp.asarray(source_idx, dtype=xp.int64)
    tgt_arr = xp.asarray(target_idx, dtype=xp.int64)
    sens_arr = xp.asarray(sensor_idx, dtype=xp.int64)
    place_arr = src_arr  # the place pool == source
    act_gpu = xp.asarray(sensor_act, dtype=xp.float32)
    pulse_gpu = xp.full(len(source_idx), float(pace_amp_pA), dtype=xp.float32)

    aff_idx = afferents
    src_counts = np.zeros(len(source_idx), dtype=np.float64)
    tgt_counts = np.zeros(len(target_idx), dtype=np.float64)
    per_step_max_ci = []
    pace_vectors_seen = set()   # audit: location-blindness (the pulse must not depend on location)

    for k in range(record_steps):
        bridge.cp_external_input_current[:] = 0.0
        # Sensor drive (the place code's selective input). Under --jitter, ON only every other step
        # (de-synchronizes the source ensemble; same active cells, same total drive, spikes spread).
        sensor_on = (not jitter) or (k % 2 == 0)
        if sensor_on:
            bridge.cp_external_input_current[sens_arr] = act_gpu
        # Gamma pacing (location-blind), unless --no-rhythm or sync=='ping'.
        pace_amp = 0.0
        if (not no_rhythm) and sync == "pacing":
            pace_amp = _gamma_pulse_step(k + pace_offset_steps, gamma_hz=gamma_hz, dt_ms=dt_ms,
                                         amp_pA=pace_amp_pA, duty=pace_duty)
            if pace_amp > 0.0:
                bridge.cp_external_input_current[place_arr] = (
                    bridge.cp_external_input_current[place_arr] + pulse_gpu)
                pace_vectors_seen.add(("pulse_on", round(pace_amp, 3)))
            else:
                pace_vectors_seen.add(("pulse_off", 0.0))
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)
        fired = _host(bridge.cp_firing_states)
        src_counts += fired[np.asarray(_host(src_arr))].astype(np.float64)
        tgt_counts += fired[np.asarray(_host(tgt_arr))].astype(np.float64)
        # c_i per target this step = number of its afferent source cells that fired this step.
        step_max = 0
        for a in aff_idx:
            if a.size == 0:
                continue
            ci = int(fired[a].sum())
            if ci > step_max:
                step_max = ci
        per_step_max_ci.append(step_max)

    bridge.cp_external_input_current[:] = 0.0
    return {
        "src_counts": src_counts, "tgt_counts": tgt_counts,
        "per_step_max_ci": per_step_max_ci,
        "pace_vectors_seen": sorted(pace_vectors_seen),
    }


# ──────────────────────────────────────────────────────────────────────
# One seed
# ──────────────────────────────────────────────────────────────────────

def run_seed(seed, *, locations, landmarks, n_bearing, n_dist, n_source, n_target,
             src_drive_weight, src_drive_density, src_drive_jitter,
             s2t_weight, s2t_density, s2t_jitter, k_threshold, gain, plateau_strength,
             max_intensity, falloff, dist_sigma, dist_max, bexp,
             train_passes, train_steps_per_loc, record_steps,
             sync, gamma_hz, pace_amp_pA, pace_duty, n_fs,
             place_to_fs_weight, place_to_fs_density, fs_to_place_weight, fs_to_place_density,
             enable_coincidence=True, jitter_inputs=False, no_rhythm=False, verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    from sim.backend import get_backend
    xp, backend_name = get_backend()

    n_per_landmark = n_bearing + n_dist
    n_sensors = len(landmarks) * n_per_landmark

    bridge, cfg = _build(
        seed, n_sensors=n_sensors, n_source=n_source, n_target=n_target,
        src_drive_weight=src_drive_weight, src_drive_density=src_drive_density,
        src_drive_jitter=src_drive_jitter, s2t_weight=s2t_weight, s2t_density=s2t_density,
        s2t_jitter=s2t_jitter, k_threshold=k_threshold, gain=gain, plateau_strength=plateau_strength,
        enable_coincidence=enable_coincidence, sync=sync, n_fs=n_fs,
        place_to_fs_weight=place_to_fs_weight, place_to_fs_density=place_to_fs_density,
        fs_to_place_weight=fs_to_place_weight, fs_to_place_density=fs_to_place_density,
        no_rhythm=no_rhythm)
    _assert_cupy_regime(cfg, backend_name)
    rm = bridge.region_manager
    sensor_idx = np.asarray(rm.indices("src_sensors"), dtype=np.int64)
    source_idx = np.asarray(rm.indices("source"), dtype=np.int64)
    target_idx = np.asarray(rm.indices("target"), dtype=np.int64)

    # anti-cheat: only the sensory afferent (and, for pacing, the LOCATION-BLIND place pulse) is driven.
    driven_regions = {"src_sensors"}
    if sync == "pacing" and not no_rhythm:
        driven_regions.add("source(gamma-pulse:location-blind)")

    loc_acts = [landmark_sensor_act(x, y, landmarks, n_bearing, n_dist, max_intensity,
                                    falloff, dist_sigma, dist_max, bexp) for (x, y) in locations]

    # Train the place fields (gate-managed; freeze before measure -- the Step-0 protocol).
    sens_arr = xp.asarray(sensor_idx, dtype=xp.int64)
    bridge.set_plasticity_gate("src_drive", 1.0)
    rng = np.random.default_rng(seed)
    for _p in range(train_passes):
        order = list(range(len(loc_acts)))
        rng.shuffle(order)
        for li in order:
            bridge.cp_external_input_current[:] = 0.0
            _step(bridge, 20)
            _set_clamp(bridge, xp, sens_arr, xp.asarray(loc_acts[li], dtype=xp.float32))
            _step(bridge, train_steps_per_loc)
    bridge.set_plasticity_gate("src_drive", 0.0)
    bridge.cp_external_input_current[:] = 0.0
    _step(bridge, 30)

    afferents = _afferent_lists(bridge, source_idx, target_idx)

    # --- Measure (with the synchronizer active) at each location ---
    src_count_vecs, tgt_count_vecs, tgt_max_spk, per_loc_volley_p90, per_loc_volley_max = [], [], [], [], []
    pace_audit = None
    for li, act in enumerate(loc_acts):
        bridge.cp_external_input_current[:] = 0.0
        _step(bridge, 30)
        m = _measure_volley(bridge, xp, sensor_idx, source_idx, target_idx, afferents, act, record_steps,
                            sync=sync, gamma_hz=gamma_hz, dt_ms=cfg.dt_ms, pace_amp_pA=pace_amp_pA,
                            pace_duty=pace_duty, jitter=jitter_inputs, no_rhythm=no_rhythm,
                            pace_offset_steps=li * 7)  # vary phase per loc so we don't accidentally lock
        src_count_vecs.append(m["src_counts"])
        tgt_count_vecs.append(m["tgt_counts"])
        tms = float(m["tgt_counts"].max()) / float(record_steps) if len(m["tgt_counts"]) else 0.0
        tgt_max_spk.append(tms)
        pm = np.asarray(m["per_step_max_ci"], dtype=np.float64)
        per_loc_volley_p90.append(float(np.percentile(pm, 90)) if pm.size else 0.0)
        per_loc_volley_max.append(int(pm.max()) if pm.size else 0)
        if pace_audit is None:
            pace_audit = m["pace_vectors_seen"]
        bridge.cp_external_input_current[:] = 0.0

    n_src = len(source_idx)
    sparsities = [float((sc > 0).sum()) / float(n_src) for sc in src_count_vecs]
    active_rates = [float(sc[sc > 0].mean()) / float(record_steps) if (sc > 0).any() else 0.0
                    for sc in src_count_vecs]
    diff_cos = [cosine_counts(src_count_vecs[i], src_count_vecs[j])
                for i in range(len(src_count_vecs)) for j in range(i + 1, len(src_count_vecs))]
    tgt_diff_cos = [cosine_counts(tgt_count_vecs[i], tgt_count_vecs[j])
                    for i in range(len(tgt_count_vecs)) for j in range(i + 1, len(tgt_count_vecs))]

    src_sparsity_mean = float(np.mean(sparsities))
    src_active_rate_mean = float(np.mean(active_rates))
    src_diff_cos_mean = float(np.mean(diff_cos)) if diff_cos else 0.0
    tgt_diff_cos_mean = float(np.mean(tgt_diff_cos)) if tgt_diff_cos else 0.0
    tgt_max_spk_mean = float(np.mean(tgt_max_spk))
    volley_p90_mean = float(np.mean(per_loc_volley_p90))
    volley_max = int(max(per_loc_volley_max)) if per_loc_volley_max else 0

    # Gates.
    g_sparse = bool(src_sparsity_mean <= 0.05 and src_active_rate_mean < 0.2 and src_diff_cos_mean < 0.30)
    g_volley = bool(volley_max >= k_threshold)   # the volley reaches >= K in at least one step
    g_fire = bool(tgt_max_spk_mean >= 0.005)      # MSN-D1 >= 5 Hz at dt=1ms
    near = tgt_max_spk[0] if tgt_max_spk else 0.0
    far = min(tgt_max_spk) if tgt_max_spk else 0.0
    if far > 0:
        g_distinct = bool(near >= 3.0 * far and tgt_diff_cos_mean < 0.30)
    else:
        g_distinct = bool(near > 0 and any(t == 0 for t in tgt_max_spk))

    mode = ("ablate-routeD" if not enable_coincidence else
            ("no-rhythm" if no_rhythm else ("jitter" if jitter_inputs else "volley")))
    log(f"  [seed {seed}] sync={sync} mode={mode} K={k_threshold}")
    log(f"    SOURCE: sparsity {src_sparsity_mean*100:.1f}%  act-rate {src_active_rate_mean:.3f}  "
        f"diff-cos {src_diff_cos_mean:.3f}  -> G_SPARSE={g_sparse}")
    log(f"    VOLLEY: per-step max c_i  p90 {volley_p90_mean:.1f}  MAX {volley_max}  (K={k_threshold})  "
        f"-> G_VOLLEY={g_volley}")
    log(f"    TARGET: max-cell {tgt_max_spk_mean:.4f} spk/step ({tgt_max_spk_mean*1000:.1f} Hz)  "
        f"tgt-diff-cos {tgt_diff_cos_mean:.3f}  near {near:.4f} far {far:.4f}  "
        f"-> G_FIRE={g_fire}  G_DISTINCT={g_distinct}")

    return {
        "seed": seed, "backend": backend_name, "sync": sync, "mode": mode,
        "coincidence": bool(enable_coincidence), "jitter_inputs": bool(jitter_inputs),
        "no_rhythm": bool(no_rhythm), "k_threshold": float(k_threshold),
        "gamma_hz": float(gamma_hz), "pace_amp_pA": float(pace_amp_pA), "pace_duty": float(pace_duty),
        "driven_regions": sorted(driven_regions),
        "pace_pulse_audit": pace_audit,  # location-blind: the SAME {on,off} pulse regardless of location
        "source_sparsity_mean": src_sparsity_mean, "source_active_rate_mean": src_active_rate_mean,
        "source_diff_cos_mean": src_diff_cos_mean,
        "volley_per_step_max_ci_p90_mean": volley_p90_mean, "volley_per_step_max_ci_max": volley_max,
        "target_max_cell_spk_per_step_mean": tgt_max_spk_mean,
        "target_diff_cos_mean": tgt_diff_cos_mean,
        "per_loc_target_max_spk": tgt_max_spk, "per_loc_volley_max_ci": per_loc_volley_max,
        "per_loc_source_sparsity": sparsities,
        "G_SPARSE": g_sparse, "G_VOLLEY": g_volley, "G_FIRE": g_fire, "G_DISTINCT": g_distinct,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--out", type=str, default="research/findings/raw/_coincidence_volley_n9_derisk.json")
    ap.add_argument("--sync", type=str, default="pacing", choices=["pacing", "ping"],
                    help="pacing = host gamma pulse (scaffold); ping = FS-PING emergent gamma (faithful)")
    ap.add_argument("--ablate-subunit", action="store_true",
                    help="enable_coincidence_detection=False (Route D OFF) -> with rhythm ON, must NOT fire")
    ap.add_argument("--jitter-inputs", action="store_true",
                    help="de-synchronize the source (THE coincidence control) -> G_FIRE must collapse")
    ap.add_argument("--no-rhythm", action="store_true",
                    help="disable the synchronizer (Route D still ON) -> reproduce the c_i<=1 wall")
    ap.add_argument("--k-threshold", type=float, default=4.0, help="coincidence_k_threshold (MUST be > 1)")
    ap.add_argument("--k-sweep", type=str, default=None, help="comma list of K to sweep (K>1 anti-cheat)")
    ap.add_argument("--gain", type=float, default=2.0)
    ap.add_argument("--plateau-strength", type=float, default=80.0)
    # gamma pacing
    ap.add_argument("--gamma-hz", type=float, default=40.0, help="gamma pacing frequency (pacing mode)")
    ap.add_argument("--pace-amp-pA", type=float, default=60.0,
                    help="location-blind gamma pulse amplitude on the place pool (pacing mode); 60 = the "
                         "firing operating point (amp below this stops firing; amp above densifies more)")
    ap.add_argument("--pace-duty", type=float, default=0.12, help="pulse-ON fraction of each gamma period")
    # FS-PING
    ap.add_argument("--n-fs", type=int, default=80)
    ap.add_argument("--place-to-fs-weight", type=float, default=24.0)
    ap.add_argument("--place-to-fs-density", type=float, default=0.6)
    ap.add_argument("--fs-to-place-weight", type=float, default=14.0)
    ap.add_argument("--fs-to-place-density", type=float, default=0.6)
    # topology (matches the wall probe)
    ap.add_argument("--n-source", type=int, default=400)
    ap.add_argument("--n-target", type=int, default=20)
    ap.add_argument("--s2t-weight", type=float, default=20.0)
    ap.add_argument("--s2t-density", type=float, default=0.5,
                    help="source->target density. 0.5 = CLUSTERED (each target samples a different ~50%% "
                         "subset -> different locations hit different targets -> downstream distinctness). "
                         "1.0 = every active cell hits every target (max fan-in for c_i, but position-blind "
                         "at the target).")
    ap.add_argument("--s2t-jitter", type=float, default=0.3)
    ap.add_argument("--src-drive-weight", type=float, default=28.0)
    ap.add_argument("--src-drive-density", type=float, default=0.5)
    ap.add_argument("--src-drive-jitter", type=float, default=0.6)
    ap.add_argument("--n-bearing", type=int, default=12)
    ap.add_argument("--n-dist", type=int, default=8)
    ap.add_argument("--max-intensity", type=float, default=450.0)
    ap.add_argument("--falloff", type=float, default=0.03)
    ap.add_argument("--dist-sigma", type=float, default=4.0)
    ap.add_argument("--grid-size", type=int, default=32)
    ap.add_argument("--bexp", type=float, default=4.0)
    ap.add_argument("--train-passes", type=int, default=12)
    ap.add_argument("--train-steps-per-loc", type=int, default=120)
    ap.add_argument("--record-steps", type=int, default=120)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    grid_size = int(args.grid_size)
    locations = default_locations(grid_size)
    landmarks = default_landmarks(grid_size)
    dist_max = float(grid_size) * 1.42
    Ks = [float(k) for k in args.k_sweep.split(",")] if args.k_sweep else [args.k_threshold]
    if any(k <= 1.0 for k in Ks):
        raise AssertionError("ANTI-CHEAT: coincidence_k_threshold must be > 1 (a single input must not trigger).")

    def _common(seed, K):
        return run_seed(seed, locations=locations, landmarks=landmarks, n_bearing=args.n_bearing,
                        n_dist=args.n_dist, n_source=args.n_source, n_target=args.n_target,
                        src_drive_weight=args.src_drive_weight, src_drive_density=args.src_drive_density,
                        src_drive_jitter=args.src_drive_jitter, s2t_weight=args.s2t_weight,
                        s2t_density=args.s2t_density, s2t_jitter=args.s2t_jitter, k_threshold=K,
                        gain=args.gain, plateau_strength=args.plateau_strength,
                        max_intensity=args.max_intensity, falloff=args.falloff, dist_sigma=args.dist_sigma,
                        dist_max=dist_max, bexp=args.bexp, train_passes=args.train_passes,
                        train_steps_per_loc=args.train_steps_per_loc, record_steps=args.record_steps,
                        sync=args.sync, gamma_hz=args.gamma_hz, pace_amp_pA=args.pace_amp_pA,
                        pace_duty=args.pace_duty, n_fs=args.n_fs,
                        place_to_fs_weight=args.place_to_fs_weight, place_to_fs_density=args.place_to_fs_density,
                        fs_to_place_weight=args.fs_to_place_weight, fs_to_place_density=args.fs_to_place_density,
                        enable_coincidence=(not args.ablate_subunit), jitter_inputs=args.jitter_inputs,
                        no_rhythm=args.no_rhythm, verbose=True)

    t0 = time.time()
    results = []
    for K in Ks:
        for seed in seeds:
            results.append(_common(seed, K))
    elapsed = time.time() - t0

    n_fire = sum(1 for r in results if r["G_FIRE"])
    n_volley = sum(1 for r in results if r["G_VOLLEY"])
    n_distinct = sum(1 for r in results if r["G_DISTINCT"])
    n_sparse = sum(1 for r in results if r["G_SPARSE"])
    mode = ("ablate-routeD" if args.ablate_subunit else
            ("no-rhythm" if args.no_rhythm else ("jitter" if args.jitter_inputs else "volley")))
    out = {
        "probe": "coincidence_volley_n9_derisk",
        "design_doc": "2026-06-09-route-T-volley-synchronization-design.md",
        "sync": args.sync, "mode": mode,
        "seeds": seeds, "K_values": Ks, "elapsed_seconds": round(elapsed, 1),
        "G_FIRE_pass_count": n_fire, "G_VOLLEY_pass_count": n_volley,
        "G_DISTINCT_pass_count": n_distinct, "G_SPARSE_pass_count": n_sparse, "n_runs": len(results),
        "per_run": results,
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print("\n" + "=" * 78)
    print(f"sync={args.sync}  mode={mode}  G_FIRE {n_fire}/{len(results)}  G_VOLLEY {n_volley}/{len(results)}"
          f"  G_DISTINCT {n_distinct}/{len(results)}  G_SPARSE {n_sparse}/{len(results)}")
    print(f"wrote {outp}  ({elapsed:.1f}s)")
    print("=" * 78)


if __name__ == "__main__":
    main()
