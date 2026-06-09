"""C1 diagnostic — stage-by-stage conduction of the trisynaptic loop driven from landmark sensors.

Goal: find WHERE the EC->DG->CA3->CA1 loop goes silent (or floods non-selectively) when driven by the
legitimate egocentric landmark sensors (NOT language_input, NOT direct-CA3). Instruments per-hop firing
(ec, dg, dg_pv_basket, ca3, ca1) and per-hop diff-location cosine at >=2 locations, so I can tune the
levers (perforant weight/density, DG FFI strength, mossy dg->ca3 weight/density, CA3 recurrent gain)
runner-side until CA1 is BOTH non-silent (high-rate) AND distinct-per-location.

Reuses: the Stage-1 landmark render (placecode_selforg_stage1_derisk) + the trisynaptic regions/pathways
from build_biological_brain_regions(enable_hippocampus_consolidation=True), with the afferent SWAPPED from
language_input->ec to landmark_sensors->ec (runner-side; NO sim/ edit).

CuPy only. Not a gated probe — a forensic instrument.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
# repo root is 4 levels up: research/findings/raw/<file> -> research/findings/raw -> research/findings
# -> research -> <repo root>
_d = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_d))))

from research.runners.placecode_selforg_stage1_derisk import (
    landmark_sensor_act, default_locations, default_landmarks, cosine_counts, _host)


def build_trisyn_from_landmarks(seed, *, n_sensors, n_ec, n_dg, n_dg_pv_basket, n_ca3, n_ca1,
                                lm_to_ec_weight, lm_to_ec_density,
                                ec_to_dg_weight, ec_to_dg_density,
                                ec_to_pvb_weight, pvb_to_dg_weight,
                                dg_to_ca3_weight, dg_to_ca3_density,
                                ca3_rec_weight, ca3_rec_density,
                                ca3_to_ca1_weight, ca3_to_ca1_density,
                                ec_to_ca1_weight, ec_to_ca1_density,
                                ec_internal_density=0.05, ca3_inh_weight=2.0,
                                n_ca3_inh=0, ca3_to_inh_weight=8.0, ca3_to_inh_density=0.30,
                                inh_to_ca3_weight=6.0, inh_to_ca3_density=0.60,
                                enable_nmda=True, dt_ms=1.0):
    """Build the trisynaptic loop driven from a landmark_sensors region.

    Mirrors build_biological_brain_regions(enable_hippocampus_consolidation=True) EXACTLY for the
    hippocampal regions/pathways, but (a) adds a landmark_sensors region, (b) replaces language_input->ec
    with landmark_sensors->ec, (c) exposes every lever as a kwarg for tuning.
    """
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType

    regions = [
        BrainRegion(name="landmark_sensors", n_neurons=int(n_sensors), exc_fraction=1.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False, izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name),
        # === trisynaptic regions, byte-mirrored from build_biological_brain_regions ===
        BrainRegion(name="ec", n_neurons=int(n_ec), exc_fraction=0.8, internal_density=ec_internal_density,
                    exc_weight_mean=0.3, inh_weight_mean=0.8, weight_jitter=0.2, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name),
        BrainRegion(name="dg", n_neurons=int(n_dg), exc_fraction=0.95, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name),
        BrainRegion(name="dg_pv_basket", n_neurons=int(n_dg_pv_basket), exc_fraction=0.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False, izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name),
        BrainRegion(name="ca3", n_neurons=int(n_ca3), exc_fraction=0.85, internal_density=0.0,
                    exc_weight_mean=1.5, inh_weight_mean=float(ca3_inh_weight), weight_jitter=0.2,
                    plastic_internal=True, izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name,
                    enable_nmda=bool(enable_nmda)),
        BrainRegion(name="ca1", n_neurons=int(n_ca1), exc_fraction=0.85, internal_density=0.05,
                    exc_weight_mean=0.3, inh_weight_mean=0.8, weight_jitter=0.2, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name),
    ]
    # Optional CA3 feedback-inhibition interneuron pool (PV basket cells; de Almeida 2009 E%-max
    # sparsification). The missing piece for DISTINCT attractor basins: ca3 -> ca3_inh -> ca3 implements
    # k-WTA so the recurrent attractor stays SPARSE (only the best-matched ensemble wins) rather than
    # globally igniting all CA3 cells into one basin. Off when n_ca3_inh==0.
    if n_ca3_inh > 0:
        regions.append(BrainRegion(name="ca3_inh", n_neurons=int(n_ca3_inh), exc_fraction=0.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False, izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name))
    pathways = [
        # SWAPPED afferent: landmark_sensors -> ec (was language_input -> ec).
        RegionPathway(from_region="landmark_sensors", to_region="ec", density=float(lm_to_ec_density),
                      weight_mean=float(lm_to_ec_weight), weight_jitter=0.2, plastic=True,
                      plasticity_gate="landmark_to_ec"),
        # ec -> dg (perforant)
        RegionPathway(from_region="ec", to_region="dg", density=float(ec_to_dg_density),
                      weight_mean=float(ec_to_dg_weight), weight_jitter=0.2, plastic=True,
                      plasticity_gate="ec_to_dg"),
        # ec -> dg_pv_basket and dg_pv_basket -> dg (FFI for sparsity)
        RegionPathway(from_region="ec", to_region="dg_pv_basket", density=0.40,
                      weight_mean=float(ec_to_pvb_weight), weight_jitter=0.2, plastic=False),
        RegionPathway(from_region="dg_pv_basket", to_region="dg", density=1.0,
                      weight_mean=float(pvb_to_dg_weight), weight_jitter=0.2, plastic=False),
        # dg -> ca3 (mossy fibers)
        RegionPathway(from_region="dg", to_region="ca3", density=float(dg_to_ca3_density),
                      weight_mean=float(dg_to_ca3_weight), weight_jitter=0.2, plastic=True,
                      plasticity_gate="dg_to_ca3"),
        # ec -> ca1 (direct cortical bypass)
        RegionPathway(from_region="ec", to_region="ca1", density=float(ec_to_ca1_density),
                      weight_mean=float(ec_to_ca1_weight), weight_jitter=0.2, plastic=True,
                      plasticity_gate="ec_to_ca1"),
        # ca3 -> ca3 recurrent (the attractor; supplies the RATE)
        RegionPathway(from_region="ca3", to_region="ca3", density=float(ca3_rec_density),
                      weight_mean=float(ca3_rec_weight), weight_jitter=0.2, plastic=True,
                      plasticity_gate="ca3_swr_burst"),
        # ca3 -> ca1 (Schaffer collaterals)
        RegionPathway(from_region="ca3", to_region="ca1", density=float(ca3_to_ca1_density),
                      weight_mean=float(ca3_to_ca1_weight), weight_jitter=0.2, plastic=True,
                      plasticity_gate="ca3_to_ca1"),
    ]
    # CA3 feedback inhibition loop (sparsifies the attractor -> distinct basins).
    if n_ca3_inh > 0:
        pathways.append(RegionPathway(from_region="ca3", to_region="ca3_inh",
                      density=float(ca3_to_inh_density), weight_mean=float(ca3_to_inh_weight),
                      weight_jitter=0.2, plastic=False))
        pathways.append(RegionPathway(from_region="ca3_inh", to_region="ca3",
                      density=float(inh_to_ca3_density), weight_mean=float(inh_to_ca3_weight),
                      weight_jitter=0.2, plastic=False))
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


def _step(bridge, n):
    for _ in range(n):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1


def measure_hops(bridge, xp, sensor_idx, region_idx_map, sensor_act, record_steps, reset_steps=40):
    """Drive landmark_sensors with sensor_act; accumulate per-neuron spike counts of every named region."""
    bridge.cp_external_input_current[:] = 0.0
    _step(bridge, reset_steps)
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[xp.asarray(sensor_idx, dtype=xp.int64)] = \
        xp.asarray(sensor_act, dtype=xp.float32)
    counts = {n: xp.zeros(len(idx), dtype=xp.float32) for n, idx in region_idx_map.items()}
    arrs = {n: xp.asarray(idx, dtype=xp.int64) for n, idx in region_idx_map.items()}
    for _ in range(record_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        for n in counts:
            counts[n] += bridge.cp_firing_states[arrs[n]].astype(xp.float32)
    bridge.cp_external_input_current[:] = 0.0
    return {n: _host(c) for n, c in counts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid-size", type=int, default=32)
    ap.add_argument("--n-ec", type=int, default=200)
    ap.add_argument("--n-dg", type=int, default=800)
    ap.add_argument("--n-dg-pv-basket", type=int, default=240)
    ap.add_argument("--n-ca3", type=int, default=400)
    ap.add_argument("--n-ca1", type=int, default=200)
    ap.add_argument("--lm-to-ec-weight", type=float, default=4.0)
    ap.add_argument("--lm-to-ec-density", type=float, default=0.30)
    ap.add_argument("--ec-to-dg-weight", type=float, default=6.0)
    ap.add_argument("--ec-to-dg-density", type=float, default=0.40)
    ap.add_argument("--ec-to-pvb-weight", type=float, default=5.0)
    ap.add_argument("--pvb-to-dg-weight", type=float, default=6.0)
    ap.add_argument("--dg-to-ca3-weight", type=float, default=8.0)
    ap.add_argument("--dg-to-ca3-density", type=float, default=0.10)
    ap.add_argument("--ca3-rec-weight", type=float, default=1.5)
    ap.add_argument("--ca3-rec-density", type=float, default=0.30)
    ap.add_argument("--ca3-to-ca1-weight", type=float, default=4.0)
    ap.add_argument("--ca3-to-ca1-density", type=float, default=0.30)
    ap.add_argument("--n-ca3-inh", type=int, default=0, help="CA3 feedback-inhibition pool size (k-WTA)")
    ap.add_argument("--ca3-to-inh-weight", type=float, default=8.0)
    ap.add_argument("--inh-to-ca3-weight", type=float, default=6.0)
    ap.add_argument("--inh-to-ca3-density", type=float, default=0.60)
    ap.add_argument("--ec-to-ca1-weight", type=float, default=3.0)
    ap.add_argument("--ec-to-ca1-density", type=float, default=0.30)
    ap.add_argument("--max-intensity", type=float, default=450.0)
    ap.add_argument("--falloff", type=float, default=0.03)
    ap.add_argument("--bexp", type=float, default=4.0)
    ap.add_argument("--n-bearing", type=int, default=12)
    ap.add_argument("--n-dist", type=int, default=8)
    ap.add_argument("--dist-sigma", type=float, default=4.0)
    ap.add_argument("--record-steps", type=int, default=100)
    ap.add_argument("--no-nmda", action="store_true")
    # self-org: open the plastic feedforward gates and walk locations before measuring
    ap.add_argument("--selforg-passes", type=int, default=0,
                    help="if >0, run STDP self-org over the locations (open landmark_to_ec/ec_to_dg/"
                         "dg_to_ca3/ca3_to_ca1/ca3_swr_burst) for this many passes before measuring")
    ap.add_argument("--selforg-steps", type=int, default=120)
    args = ap.parse_args()

    from sim.backend import get_backend
    xp, backend = get_backend()
    assert backend == "cupy", f"need cupy, got {backend}"

    grid = int(args.grid_size)
    locations = default_locations(grid)
    landmarks = default_landmarks(grid)
    dist_max = float(grid) * 1.42
    n_per_lm = args.n_bearing + args.n_dist
    n_sensors = len(landmarks) * n_per_lm

    t0 = time.time()
    bridge, cfg = build_trisyn_from_landmarks(
        args.seed, n_sensors=n_sensors, n_ec=args.n_ec, n_dg=args.n_dg,
        n_dg_pv_basket=args.n_dg_pv_basket, n_ca3=args.n_ca3, n_ca1=args.n_ca1,
        lm_to_ec_weight=args.lm_to_ec_weight, lm_to_ec_density=args.lm_to_ec_density,
        ec_to_dg_weight=args.ec_to_dg_weight, ec_to_dg_density=args.ec_to_dg_density,
        ec_to_pvb_weight=args.ec_to_pvb_weight, pvb_to_dg_weight=args.pvb_to_dg_weight,
        dg_to_ca3_weight=args.dg_to_ca3_weight, dg_to_ca3_density=args.dg_to_ca3_density,
        ca3_rec_weight=args.ca3_rec_weight, ca3_rec_density=args.ca3_rec_density,
        ca3_to_ca1_weight=args.ca3_to_ca1_weight, ca3_to_ca1_density=args.ca3_to_ca1_density,
        ec_to_ca1_weight=args.ec_to_ca1_weight, ec_to_ca1_density=args.ec_to_ca1_density,
        n_ca3_inh=args.n_ca3_inh, ca3_to_inh_weight=args.ca3_to_inh_weight,
        inh_to_ca3_weight=args.inh_to_ca3_weight, inh_to_ca3_density=args.inh_to_ca3_density,
        enable_nmda=not args.no_nmda)
    rm = bridge.region_manager
    sensor_idx = np.asarray(rm.indices("landmark_sensors"), dtype=np.int64)
    region_idx_map = {n: np.asarray(rm.indices(n), dtype=np.int64)
                      for n in ("ec", "dg", "dg_pv_basket", "ca3", "ca1")}
    print(f"built in {time.time()-t0:.1f}s; {cfg.num_neurons} neurons, {int(bridge.cp_connections.nnz)} syn; "
          f"nmda={not args.no_nmda}")

    def render(name):
        x, y = locations[name]
        return landmark_sensor_act(x, y, landmarks, args.n_bearing, args.n_dist, args.max_intensity,
                                   args.falloff, args.dist_sigma, dist_max, args.bexp)
    loc_names = list(locations.keys())
    loc_sensor = {n: render(n) for n in loc_names}

    if args.selforg_passes > 0:
        for g in ("landmark_to_ec", "ec_to_dg", "dg_to_ca3", "ca3_to_ca1", "ca3_swr_burst", "ec_to_ca1"):
            try: bridge.set_plasticity_gate(g, 1.0)
            except Exception: pass
        rng = np.random.default_rng(args.seed)
        for _p in range(args.selforg_passes):
            order = list(loc_names); rng.shuffle(order)
            for name in order:
                bridge.cp_external_input_current[:] = 0.0
                _step(bridge, 20)
                bridge.cp_external_input_current[xp.asarray(sensor_idx, dtype=xp.int64)] = \
                    xp.asarray(loc_sensor[name], dtype=xp.float32)
                _step(bridge, args.selforg_steps)
        for g in ("landmark_to_ec", "ec_to_dg", "dg_to_ca3", "ca3_to_ca1", "ca3_swr_burst", "ec_to_ca1"):
            try: bridge.set_plasticity_gate(g, 0.0)
            except Exception: pass
        bridge.cp_external_input_current[:] = 0.0
        print(f"self-org done: {args.selforg_passes} passes x {len(loc_names)} locs")

    # Measure each location, all hops
    hop_counts = {n: measure_hops(bridge, xp, sensor_idx, region_idx_map, loc_sensor[n], args.record_steps)
                  for n in loc_names}

    hops = ("ec", "dg", "dg_pv_basket", "ca3", "ca1")
    print("\nPER-HOP active fraction (spikes>0) and total spikes/step, per location:")
    header = "  loc      " + "".join(f"{h:>22}" for h in hops)
    print(header)
    for name in loc_names:
        cells = []
        for h in hops:
            v = hop_counts[name][h]
            af = float(np.mean(v > 0))
            spk = float(np.sum(v)) / args.record_steps
            cells.append(f"af{af:5.2f}|spk{spk:6.1f}")
        print(f"  {name:8s} " + "".join(f"{c:>22}" for c in cells))

    # diff-location cosine per hop (NEAR=near vs FAR=far_a, far_b)
    print("\nDIFF-LOCATION cosine per hop (lower=more distinct):")
    pairs = [("near", "far_a"), ("near", "far_b"), ("far_a", "far_b")]
    for h in hops:
        cs = [cosine_counts(hop_counts[a][h], hop_counts[b][h]) for a, b in pairs]
        print(f"  {h:14s}  " + "  ".join(f"{a}-{b}={c:.3f}" for (a, b), c in zip(pairs, cs))
              + f"   mean={np.mean(cs):.3f}")
    # CA1 is the deliverable: report its summary line
    ca1_af = {n: float(np.mean(hop_counts[n]["ca1"] > 0)) for n in loc_names}
    ca1_spk = {n: float(np.sum(hop_counts[n]["ca1"])) / args.record_steps for n in loc_names}
    ca1_diff = float(np.mean([cosine_counts(hop_counts[a]["ca1"], hop_counts[b]["ca1"]) for a, b in pairs]))
    print(f"\n>>> CA1: active-frac {ca1_af}  spk/step {ca1_spk}  diff-cos {ca1_diff:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
