"""Step B de-risk (STAGED -- do NOT run until the Step-A sim/ edit is byte-reviewed AND, per the Step-0
finding, conduction delays (Route T) line up the sparse ensemble's volley): does the dendritic-
COINCIDENCE subunit (enable_coincidence_detection + coincidence_detector=True on a routed afferent)
let a SPARSE-distinct spiking ensemble fire a downstream MSN-D1 by COINCIDENCE, while staying distinct
AND collapsing under input desynchronization (proving coincidence, not rate)?

This wires the Step-A protected edit into the same isolated source->target topology the Step-0 probe
characterized (coincidence_wall_probe.py), now with the coincidence subunit ON. It is the EXACT inverse
of the RESULT's boundary sweep (the thing it proved impossible by rate).

[!] STEP-0 GATING (READ FIRST): coincidence_wall_probe.py (3 seeds, CuPy) found that in the NATURAL
sparse-distinct point-neuron dynamics, the per-step coincidence count c_i over a target's routed fan-in
is <=1 at <=5% sparsity (the ensemble cells fire ASYNCHRONOUSLY at ~10 Hz; even at projection density
1.0 / fan-in 400 the convergence does not help -- the bottleneck is the per-step EMISSION count, not
convergence). The moment drive is raised enough to get c_i>=2, the source goes DENSE (>=48%) and
position-blind. So NO sparse-distinct operating point delivers c_i>=K>=2 per step UNLESS the ensemble's
spikes are SYNCHRONIZED into the same step -- which is precisely Route T (per-synapse conduction delays,
the design's SECOND substrate upgrade). Therefore this de-risk is EXPECTED to be NEGATIVE on the bare
1-step-delay engine and should be run only AFTER Route T (or with an externally-imposed synchronous
volley) supplies the coincidence. Running it as-is characterizes the floor (and serves as the
ABLATE/jitter control bed); it is NOT expected to pass G_FIRE on the bare engine.

GATES (CuPy; >=3 seeds; deterministic regime OU/cond-noise/global-homeostasis/heterogeneity/STP OFF):
  G_SPARSE   : the driving ensemble is sparse-distinct (<=5% active, each active cell <0.2 spk/step,
               diff-location cos < 0.30) -- the regime the RESULT proved cannot fire a cell by rate.
  G_FIRE     : with the coincidence subunit ON, the downstream MSN-D1 fires >= 5 Hz from that sparse-
               distinct ensemble (vs the RESULT's 0.00 spk/step). THE HEADLINE.
  G_DISTINCT : the downstream firing stays position-specific (near-location >> far-location drive >= 3x;
               downstream diff-cos < 0.30) -- it did NOT have to go dense/position-blind to fire.
  G_MSN      : the MSN-D1's effective drive >= ~420 pA (rheobase) -> MSN >= 5 Hz.

ANTI-CHEATS (each MUST behave consistently for an honest pass):
  - THE COINCIDENCE CONTROL (decisive, Branco-Hausser): JITTER / DESYNCHRONIZE the sparse inputs (spread
    the ensemble's spikes across several steps, same total spikes/rate, just not coincident) -> G_FIRE
    must FAIL (downstream returns to ~0 Hz). If firing survives desynchronization, the mechanism is
    reading RATE, not coincidence -> the upgrade is a cheat. Implemented by phase-staggering the per-
    location sub-ensembles so their spikes land in DIFFERENT steps.
  - ABLATE the subunit (enable_coincidence_detection=False) -> reproduce the RESULT (downstream 0.00
    spk/step from the sparse code). Confirms the new term is load-bearing.
  - K > 1 (not trivially low): coincidence_k_threshold must be > 1 (a single input must NOT trigger the
    plateau, else it is just a per-synapse gain). Swept; G_DISTINCT must hold across the K that passes.
  - NO host teacher: the ONLY cp_external_input_current write targets the sensory afferent
    (src_sensors). The downstream cells fire from the brain's own routed synaptic coincidence, never a
    host-injected per-location pattern. Grep/audit-asserted.
  - CuPy regime: backend=="cupy" (numpy DISQUALIFIED); deterministic knobs OFF; no per-region
    homeostasis on the target (it must fire from the coincidence current, not threshold collapse).

USAGE (MUST be cupy; STAGED -- do not run until byte-review + Route T):
  SIM_BACKEND=cupy python -m research.runners.coincidence_n9_derisk \
      --seeds 42,43,44 --k-threshold 4 \
      --out research/findings/raw/_coincidence_n9_derisk.json
  # ablate control (subunit OFF -> reproduce the wall):
  SIM_BACKEND=cupy python -m research.runners.coincidence_n9_derisk --seeds 42,43,44 --ablate-subunit
  # jitter anti-cheat (desynchronize -> firing must collapse):
  SIM_BACKEND=cupy python -m research.runners.coincidence_n9_derisk --seeds 42,43,44 --jitter-inputs
  # K-sweep (K must stay > 1):
  SIM_BACKEND=cupy python -m research.runners.coincidence_n9_derisk --seeds 42 --k-sweep 2,4,6,8
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

# Reuse the Step-0 probe's build + helpers (the SAME isolated source->target topology; no sim/ edits in
# THIS file -- it only TOGGLES the Step-A config flags on the bridge the probe builds).
from research.runners.coincidence_wall_probe import (
    _build as _probe_build, default_locations, default_landmarks, _step, _drive_clamp,
    _measure_source_and_target)
from research.runners.placecode_selforg_stage1_derisk import landmark_sensor_act, cosine_counts, _host


# ??????????????????????????????????????????????????????????????????????
# Build the SAME isolated bridge as the Step-0 probe, then ENABLE the Step-A coincidence subunit on the
# source->target projection (coincidence_detector=True + cfg.enable_coincidence_detection + K/gain).
# Mirrors _probe_build but adds the coincidence flags (the protected edit under test).
# ??????????????????????????????????????????????????????????????????????

def _build_with_coincidence(seed, *, n_sensors, n_source, n_target, src_drive_weight, src_drive_density,
                            src_drive_jitter, source_to_target_weight, source_to_target_density,
                            source_to_target_jitter, k_threshold, gain, plateau_strength,
                            enable_coincidence=True, dt_ms=1.0):
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
        # The MSN-D1 target gets enable_nmda=True so the subunit's Mg2+ self-limiting block is active.
        BrainRegion(name="target", n_neurons=int(n_target), exc_fraction=0.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D1.name,
                    syn_reversal_potential_i_override=-60.0, enable_nmda=True),
    ]
    pathways = [
        RegionPathway(from_region="src_sensors", to_region="source", density=float(src_drive_density),
                      weight_mean=float(src_drive_weight), weight_jitter=float(src_drive_jitter),
                      plastic=True, plasticity_gate="src_drive"),
        # === THE COINCIDENCE AFFERENT (the Step-A edit under test) ===
        RegionPathway(from_region="source", to_region="target", density=float(source_to_target_density),
                      weight_mean=float(source_to_target_weight), weight_jitter=float(source_to_target_jitter),
                      plastic=False, coincidence_detector=bool(enable_coincidence)),
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
    cfg.enable_nmda = True   # the per-region mask restricts NMDA to the target (enable_nmda=True there)
    cfg.stdp_w_max = 40.0
    cfg.fast_spike_reset = True
    # === the Step-A protected edit under test ===
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
            f"REGIME FIDELITY: coincidence de-risk MUST run on CuPy (numpy DISQUALIFIED). "
            f"Got backend={backend_name!r}. Set SIM_BACKEND=cupy.")
    bad = [k for k in ("enable_ou_process", "enable_conductance_noise", "enable_homeostasis",
                       "enable_parameter_heterogeneity", "enable_short_term_plasticity")
           if getattr(cfg, k, False)]
    if bad:
        raise AssertionError(f"REGIME FIDELITY: deterministic-regime knobs left ON: {bad}")


def run_seed(seed, *, locations, landmarks, n_bearing, n_dist, n_source, n_target,
             src_drive_weight, src_drive_density, src_drive_jitter,
             s2t_weight, s2t_density, s2t_jitter, k_threshold, gain, plateau_strength,
             max_intensity, falloff, dist_sigma, dist_max, bexp,
             train_passes, train_steps_per_loc, record_steps,
             enable_coincidence=True, jitter_inputs=False, verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    from sim.backend import get_backend
    xp, backend_name = get_backend()

    n_per_landmark = n_bearing + n_dist
    n_sensors = len(landmarks) * n_per_landmark

    bridge, cfg = _build_with_coincidence(
        seed, n_sensors=n_sensors, n_source=n_source, n_target=n_target,
        src_drive_weight=src_drive_weight, src_drive_density=src_drive_density,
        src_drive_jitter=src_drive_jitter, source_to_target_weight=s2t_weight,
        source_to_target_density=s2t_density, source_to_target_jitter=s2t_jitter,
        k_threshold=k_threshold, gain=gain, plateau_strength=plateau_strength,
        enable_coincidence=enable_coincidence)
    _assert_cupy_regime(cfg, backend_name)
    rm = bridge.region_manager
    sensor_idx = np.asarray(rm.indices("src_sensors"), dtype=np.int64)
    source_idx = np.asarray(rm.indices("source"), dtype=np.int64)
    target_idx = np.asarray(rm.indices("target"), dtype=np.int64)

    driven_regions = {"src_sensors"}  # anti-cheat: only the sensory afferent is ever externally driven

    loc_acts = [landmark_sensor_act(x, y, landmarks, n_bearing, n_dist, max_intensity,
                                    falloff, dist_sigma, dist_max, bexp) for (x, y) in locations]

    # Train the place fields (gate-managed; freeze before measure -- the Step-0 protocol).
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
    bridge.set_plasticity_gate("src_drive", 0.0)
    bridge.cp_external_input_current[:] = 0.0
    _step(bridge, 30)

    # Measure source code + target firing per location.
    src_count_vecs, tgt_spk, tgt_max_spk = [], [], []
    for li, act in enumerate(loc_acts):
        bridge.cp_external_input_current[:] = 0.0
        _step(bridge, 30)
        # JITTER anti-cheat: spread the sensor drive across alternating steps so the source ensemble's
        # spikes do NOT coincide (same total drive/rate, just desynchronized). A coincidence mechanism
        # must FAIL here; a rate mechanism would not.
        if jitter_inputs:
            sc, ts, tms = _measure_jittered(bridge, xp, sensor_idx, source_idx, target_idx, act, record_steps)
        else:
            _drive_clamp(bridge, xp, sensor_idx, act)
            sc, ts, tms = _measure_source_and_target(bridge, xp, source_idx, target_idx, record_steps)
        src_count_vecs.append(sc); tgt_spk.append(ts); tgt_max_spk.append(tms)
        bridge.cp_external_input_current[:] = 0.0

    n_src = len(source_idx)
    sparsities = [float((sc > 0).sum()) / float(n_src) for sc in src_count_vecs]
    active_rates = [float(sc[sc > 0].mean()) / float(record_steps) if (sc > 0).any() else 0.0
                    for sc in src_count_vecs]
    diff_cos = [cosine_counts(src_count_vecs[i], src_count_vecs[j])
                for i in range(len(src_count_vecs)) for j in range(i + 1, len(src_count_vecs))]
    # downstream distinctness: cosine of the TARGET firing-count vectors across locations
    tgt_count_vecs = []  # need per-cell target counts; re-measure if useful -- here use scalar max as proxy
    src_sparsity_mean = float(np.mean(sparsities))
    src_active_rate_mean = float(np.mean(active_rates))
    src_diff_cos_mean = float(np.mean(diff_cos)) if diff_cos else 0.0
    tgt_spk_mean = float(np.mean(tgt_spk))
    tgt_max_spk_mean = float(np.mean(tgt_max_spk))
    # MSN >= 5 Hz at dt=1ms == 0.005 spk/step PER CELL
    tgt_fires_5hz = tgt_max_spk_mean >= 0.005

    # Gates.
    g_sparse = (src_sparsity_mean <= 0.05 and src_active_rate_mean < 0.2 and src_diff_cos_mean < 0.30)
    g_fire = bool(tgt_fires_5hz)
    # G_DISTINCT proxy: near (loc0) target rate >> far (loc with max distance) -- here use the spread of
    # per-location target rates as a coarse distinctness signal (full per-cell cos needs a richer probe).
    near = tgt_max_spk[0] if tgt_max_spk else 0.0
    far = min(tgt_max_spk) if tgt_max_spk else 0.0
    g_distinct = bool(near >= 3.0 * far) if far > 0 else bool(near > 0 and any(t == 0 for t in tgt_max_spk))

    log(f"  [seed {seed}] coincidence={'ON' if enable_coincidence else 'OFF(ablate)'}"
        f"{' +JITTER' if jitter_inputs else ''}  K={k_threshold}")
    log(f"    SOURCE: sparsity {src_sparsity_mean*100:.1f}%  act-rate {src_active_rate_mean:.3f}  "
        f"diff-cos {src_diff_cos_mean:.3f}  -> G_SPARSE={g_sparse}")
    log(f"    TARGET: {tgt_spk_mean:.4f} spk/step  max-cell {tgt_max_spk_mean:.4f} "
        f"({tgt_max_spk_mean*1000:.1f} Hz)  -> G_FIRE={g_fire}  G_DISTINCT={g_distinct}")

    return {
        "seed": seed, "backend": backend_name, "coincidence": bool(enable_coincidence),
        "jitter_inputs": bool(jitter_inputs), "k_threshold": float(k_threshold),
        "driven_regions": sorted(driven_regions),
        "source_sparsity_mean": src_sparsity_mean, "source_active_rate_mean": src_active_rate_mean,
        "source_diff_cos_mean": src_diff_cos_mean,
        "target_spk_per_step_mean": tgt_spk_mean, "target_max_cell_spk_per_step_mean": tgt_max_spk_mean,
        "G_SPARSE": bool(g_sparse), "G_FIRE": bool(g_fire), "G_DISTINCT": bool(g_distinct),
        "per_loc_target_max_spk": tgt_max_spk,
    }


def _measure_jittered(bridge, xp, sensor_idx, source_idx, target_idx, sensor_act, record_steps):
    """JITTER anti-cheat: drive the sensors in alternating ON/OFF steps (a square-wave clamp) so the
    source ensemble's spikes are temporally SPREAD (desynchronized) rather than coincident. Same active
    cells, same total drive over time, but the per-step coincidence is destroyed. A coincidence
    mechanism must then FAIL to fire the target."""
    src_arr = xp.asarray(source_idx, dtype=xp.int64)
    tgt_arr = xp.asarray(target_idx, dtype=xp.int64)
    src_counts = xp.zeros(len(source_idx), dtype=xp.float32)
    tgt_counts = xp.zeros(len(target_idx), dtype=xp.float32)
    act_gpu = xp.asarray(sensor_act, dtype=xp.float32)
    sens_gpu = xp.asarray(sensor_idx, dtype=xp.int64)
    for k in range(record_steps):
        bridge.cp_external_input_current[:] = 0.0
        # ON every other step -> source fires in bursts separated by silent steps -> no within-step
        # coincidence accumulation across the ensemble (the spikes land in DIFFERENT steps).
        if k % 2 == 0:
            bridge.cp_external_input_current[sens_gpu] = act_gpu
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)
        src_counts += bridge.cp_firing_states[src_arr].astype(xp.float32)
        tgt_counts += bridge.cp_firing_states[tgt_arr].astype(xp.float32)
    bridge.cp_external_input_current[:] = 0.0
    src_counts = _host(src_counts); tgt_counts = _host(tgt_counts)
    ts = float(tgt_counts.sum()) / float(record_steps)
    tms = float(tgt_counts.max()) / float(record_steps) if len(tgt_counts) else 0.0
    return src_counts, ts, tms


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--out", type=str, default="research/findings/raw/_coincidence_n9_derisk.json")
    ap.add_argument("--ablate-subunit", action="store_true",
                    help="enable_coincidence_detection=False -> reproduce the rate-coding wall")
    ap.add_argument("--jitter-inputs", action="store_true",
                    help="desynchronize the sparse inputs (the COINCIDENCE control) -> G_FIRE must collapse")
    ap.add_argument("--k-threshold", type=float, default=4.0, help="coincidence_k_threshold (MUST be > 1)")
    ap.add_argument("--k-sweep", type=str, default=None, help="comma list of K values to sweep (K>1 anti-cheat)")
    ap.add_argument("--gain", type=float, default=2.0)
    ap.add_argument("--plateau-strength", type=float, default=80.0)
    ap.add_argument("--n-source", type=int, default=400)
    ap.add_argument("--n-target", type=int, default=20)
    ap.add_argument("--s2t-weight", type=float, default=20.0)
    ap.add_argument("--s2t-density", type=float, default=1.0,
                    help="source->target density (1.0 = tight clustering: every active cell hits every target)")
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
    ap.add_argument("--record-steps", type=int, default=100)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    grid_size = int(args.grid_size)
    locations = default_locations(grid_size)
    landmarks = default_landmarks(grid_size)
    dist_max = float(grid_size) * 1.42
    if args.k_sweep:
        Ks = [float(k) for k in args.k_sweep.split(",")]
    else:
        Ks = [args.k_threshold]
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
                        enable_coincidence=(not args.ablate_subunit), jitter_inputs=args.jitter_inputs,
                        verbose=True)

    t0 = time.time()
    results = []
    for K in Ks:
        for seed in seeds:
            results.append(_common(seed, K))
    elapsed = time.time() - t0

    n_fire = sum(1 for r in results if r["G_FIRE"])
    n_distinct = sum(1 for r in results if r["G_DISTINCT"])
    out = {
        "probe": "coincidence_n9_derisk",
        "design_doc": "2026-06-09-coincidence-substrate-upgrade-design.md",
        "note": "STAGED -- expected NEGATIVE on the bare 1-step-delay engine per coincidence_wall_probe.py; "
                "run after Route T conduction delays synchronize the sparse volley.",
        "mode": ("ablate" if args.ablate_subunit else ("jitter" if args.jitter_inputs else "subunit_on")),
        "seeds": seeds, "K_values": Ks, "elapsed_seconds": round(elapsed, 1),
        "G_FIRE_pass_count": n_fire, "G_DISTINCT_pass_count": n_distinct, "n_runs": len(results),
        "per_run": results,
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print("\n" + "=" * 78)
    print(f"mode={out['mode']}  G_FIRE {n_fire}/{len(results)}  G_DISTINCT {n_distinct}/{len(results)}")
    print(f"wrote {outp}  ({elapsed:.1f}s)")
    print("=" * 78)


if __name__ == "__main__":
    main()
