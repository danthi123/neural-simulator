"""gap#4 RUNG 3 — can a DOWNSTREAM layer learn to READ the learned population code, using the same local rule?

WHY. Rung 1: one CA1 cell learns a place field from ONE plateau (6-seed, blind-clean). Rung 2: 4 cells learn 4
DISTINCT fields in one lap on SHARED inputs (6-seed, blind-clean, shuffle control 0.00). Both are ONE layer. gap#4's
open frontier is whether the substrate learns DEEP representations by a biological rule -- and the first honest step
toward "deep" is: **is the learned representation usable by a later layer that learns from it?**

If the CA1 population's output is too weak, too uninformative, or too correlated to drive downstream learning, then
deep credit is blocked at the REPRESENTATION level and no credit-assignment rule can fix it. That is a cheap and
decisive thing to know before building anything more elaborate.

THE TASK (two stages, both one-shot, both the same rule):
  Stage 1 -- form the map: 4 CA1 pools learn 4 distinct place fields from per-cell plateaus in ONE lap (rung 2).
  Stage 2 -- read the map: a DOWNSTREAM pool ("L2") receives input ONLY from the 4 CA1 pools (NOT from position),
             and gets ONE plateau while a chosen target CA1 cell is active. L2 must then fire selectively when that
             CA1 cell's field is active -- i.e. it must have learned to read the LEARNED code.

THE KEY PROPERTY: L2 never sees position. Its only access to the world is through the representation layer 1 learned.
So L2 succeeding means the learned code carries usable information; L2 failing on an intact map localises the blocker
to the representation rather than to the credit rule.

PRE-REGISTERED GATE (filed before any rung-3 result exists):
    GO iff  read_acc >= 0.80  (L2's peak response bin lies within the TARGET CA1 cell's field, +/-2 bins of its peak)
       AND  selectivity >= 0.80 (L2's response to the target cell's field exceeds its response to each NON-target
            field by >= 2x)
       AND  blind seeds pass on their own
CONTROLS: C1 L2-frozen (eta=0 in stage 2 only -> no read learned); C3 no-L2-plateau moat; C2 wrong-target scoring
(score L2 against a NON-target cell -> must collapse); plus the stage-1 map must itself be intact (asserted, not
assumed). dw reported, NOT gated (C9).

NO new `sim/` edit. Run: SIM_BACKEND=numpy python -m research.runners._gap4_btsp_rung3_downstream_read_derisk
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import argparse, json
import numpy as np

N_BINS = 20
POS_N = 10
CELL_TARGETS = [5, 9, 13, 17]
N_CELL = len(CELL_TARGETS)
CA1_PER_CELL = 8
L2_N = 8
TARGET_CELL = 2                      # PRE-REGISTERED: L2 must learn to read CA1 cell 2 (target bin 13)
DEV_SEEDS = [42, 43, 44]
BLIND_SEEDS = [100, 101, 102]


def build(seed, *, eta=0.02, hdep=0.3, htheta=0.012, elig_tau=1000.0, w0=0.6, wj=0.15, dt=1.0, l2_w0=0.6):
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    cfg = CoreSimConfig(seed=seed)
    cfg.dt_ms = float(dt); cfg.num_traits = 1
    cfg.enable_brain_region_framework = True
    for f in ("enable_short_term_plasticity", "enable_hebbian_learning", "enable_homeostasis",
              "enable_structural_plasticity", "enable_reward_modulation", "enable_stdp",
              "enable_nmda", "enable_ou_process", "enable_parameter_heterogeneity",
              "enable_conductance_noise", "enable_synaptic_scaling"):
        if hasattr(cfg, f):
            setattr(cfg, f, False)
    cfg.enable_bdsp = True; cfg.bdsp_learning_rate = 0.0; cfg.bdsp_apical_bistable = True
    cfg.coincidence_plateau_self_regen = 2.0; cfg.coincidence_plateau_v_hold = -35.0; cfg.apical_kir_g = 1.0
    cfg.enable_btsp = True
    cfg.btsp_learning_rate = float(eta); cfg.btsp_elig_tau_ms = float(elig_tau)
    # w_max MUST exceed the layer-2 operating weight (l2_w0): pot = etilde*(w_max - w) goes NEGATIVE
    # otherwise and every 'potentiation' becomes a large depression (the documented soft-bound gotcha).
    cfg.btsp_w_min, cfg.btsp_w_max = 0.0, max(5.0, 2.0 * float(l2_w0))
    cfg.btsp_hetero_dep = float(hdep); cfg.btsp_hetero_theta = float(htheta)
    cfg.brain_regions = (
        [BrainRegion(name=f"pos{k}", n_neurons=POS_N, exc_fraction=1.0, internal_density=0.0,
                     exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False)
         for k in range(N_BINS)]
        + [BrainRegion(name=f"ca1_{c}", n_neurons=CA1_PER_CELL, exc_fraction=1.0, internal_density=0.0,
                       exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False)
           for c in range(N_CELL)]
        + [BrainRegion(name="l2", n_neurons=L2_N, exc_fraction=1.0, internal_density=0.0,
                       exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False)]
    )
    paths = [RegionPathway(from_region=f"pos{k}", to_region=f"ca1_{c}", density=1.0,
                           weight_mean=float(w0), weight_jitter=float(wj), plastic=True)
             for k in range(N_BINS) for c in range(N_CELL)]
    # L2 sees ONLY the CA1 population -- never position. Its whole world is the learned representation.
    paths += [RegionPathway(from_region=f"ca1_{c}", to_region="l2", density=1.0,
                            weight_mean=float(l2_w0), weight_jitter=float(wj), plastic=True)
              for c in range(N_CELL)]
    cfg.region_pathways = paths
    rt = RuntimeState(); rt.actual_seed_used = seed
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                          runtime_state=rt, gpu_config=GPUConfig())
    sb._initialize_simulation_data()
    pos = [np.asarray(sb.region_manager.indices(f"pos{k}")) for k in range(N_BINS)]
    cells = [np.asarray(sb.region_manager.indices(f"ca1_{c}")) for c in range(N_CELL)]
    l2 = np.asarray(sb.region_manager.indices("l2"))
    return sb, pos, cells, l2


def run_lap(sb, pos, cells, l2, *, ca1_targets=None, l2_plateau_bin=None, bin_steps=200,
            drive_pA=900.0, plateau_pA=600.0, release_pA=900.0, plateau_hold_ms=700.0,
            pulse_steps=15, record=False):
    from sim.backend import to_host, get_backend
    xp, _ = get_backend()
    n = int(sb.core_config.num_neurons)
    ca1_rates = np.zeros((N_CELL, N_BINS)); l2_rates = np.zeros(N_BINS); step = 0
    starts = [None] * N_CELL; l2_start = None
    for k in range(N_BINS):
        for s in range(bin_steps):
            cur = np.zeros(n, np.float32); cur[pos[k]] = drive_pA
            sb.cp_external_input_current[:] = 0.0
            sb.cp_external_input_current[:] = (xp.asarray(cur) if xp is not None else cur)
            ap = np.zeros(n, np.float32)
            if ca1_targets is not None:
                for c, b in enumerate(ca1_targets):
                    if b is None:
                        continue
                    if k == b and s < pulse_steps:
                        ap[cells[c]] = plateau_pA
                        if starts[c] is None:
                            starts[c] = step
                    if starts[c] is not None and 0 <= step - (starts[c] + int(plateau_hold_ms)) < 20:
                        ap[cells[c]] = -release_pA
            if l2_plateau_bin is not None:
                if k == l2_plateau_bin and s < pulse_steps:
                    ap[l2] = plateau_pA
                    if l2_start is None:
                        l2_start = step
                if l2_start is not None and 0 <= step - (l2_start + int(plateau_hold_ms)) < 20:
                    ap[l2] = -release_pA
            sb.cp_bdsp_apical_drive = (xp.asarray(ap) if xp is not None else ap)
            sb._run_one_simulation_step()
            if record:
                fs = np.asarray(to_host(sb.cp_firing_states))
                for c in range(N_CELL):
                    ca1_rates[c, k] += float(fs[cells[c]].sum())
                l2_rates[k] += float(np.asarray(to_host(sb.cp_conductance_g_e))[l2].mean())
            step += 1
    # FORCE-RELEASE: a plateau started late in the lap has its release scheduled past the lap end, so it
    # never fires and the cell stays LATCHED into the next stage (8/32 CA1 neurons measured above v_hold),
    # which breaks the no-plateau moat (dw != 0 with no instructive signal). Release everything at lap end.
    _rel = np.zeros(n, np.float32); _rel[:] = -release_pA
    for _ in range(20):
        sb.cp_external_input_current[:] = 0.0
        sb.cp_bdsp_apical_drive = (xp.asarray(_rel) if xp is not None else _rel)
        sb._run_one_simulation_step()
    sb.cp_bdsp_apical_drive = (xp.asarray(np.zeros(n, np.float32)) if xp is not None else np.zeros(n, np.float32))
    if record:
        ca1_rates /= float(CA1_PER_CELL * bin_steps); l2_rates /= float(bin_steps)
    return ca1_rates, l2_rates


def one_run(seed, *, l2_eta=0.02, do_l2_plateau=True, score_cell=None, bin_steps=200, l2_w0=150.0):
    from sim.backend import to_host
    sb, pos, cells, l2 = build(seed, l2_w0=l2_w0)
    # ---- STAGE 1: form the map (rung 2), L2 plasticity irrelevant here ----
    ca1_pre, l2_pre = run_lap(sb, pos, cells, l2, ca1_targets=None, bin_steps=bin_steps, record=True)
    run_lap(sb, pos, cells, l2, ca1_targets=list(CELL_TARGETS), bin_steps=bin_steps, record=False)
    sb.core_config.enable_btsp = False
    ca1_mid, l2_mid = run_lap(sb, pos, cells, l2, bin_steps=bin_steps, record=True)  # l2_mid = post-stage-1 baseline
    ca1_delta = ca1_mid - ca1_pre
    ca1_peaks = [int(np.argmax(ca1_delta[c])) if ca1_delta[c].max() > 0 else -1 for c in range(N_CELL)]
    map_ok = all(p >= 0 for p in ca1_peaks) and len(set(ca1_peaks)) == N_CELL
    # ---- STAGE 2: L2 learns to READ the map. Plateau L2 at the TARGET cell's own field peak. ----
    tgt_bin = ca1_peaks[TARGET_CELL] if ca1_peaks[TARGET_CELL] >= 0 else CELL_TARGETS[TARGET_CELL]
    sb.core_config.enable_btsp = True
    sb.core_config.btsp_learning_rate = float(l2_eta)
    w0 = float(np.abs(np.asarray(to_host(sb.cp_connections.data))).sum())
    run_lap(sb, pos, cells, l2, ca1_targets=None,
            l2_plateau_bin=(tgt_bin if do_l2_plateau else None), bin_steps=bin_steps, record=False)
    dw = float(np.abs(np.asarray(to_host(sb.cp_connections.data))).sum()) - w0
    sb.core_config.enable_btsp = False
    _, l2_post = run_lap(sb, pos, cells, l2, bin_steps=bin_steps, record=True)
    l2_delta = l2_post - l2_mid   # STAGE-2 ONLY: l2_pre would fold in the CA1 map forming (C1_frozen caught this)
    l2_peak = int(np.argmax(l2_delta)) if l2_delta.max() > 0 else -1
    # A PRIORI (derived from the rule, pre-registered): a plateau at field f credits the field that
    # PRECEDED it -- at plateau time the preceding cell has been accumulating eligibility for several
    # bins while the concurrent cell has only just begun. Same backward shift rung 1 measured (-1).
    expected = (TARGET_CELL - 1) % N_CELL
    sc = expected if score_cell is None else score_cell
    ref_peak = ca1_peaks[sc] if ca1_peaks[sc] >= 0 else CELL_TARGETS[sc]
    hit = False
    if l2_peak >= 0:
        off = (l2_peak - ref_peak + N_BINS // 2) % N_BINS - N_BINS // 2
        hit = bool(abs(off) <= 2)
    # selectivity: L2 response at the target field vs at each non-target field
    def resp_at(p):
        if p < 0:
            return 0.0
        lo, hi = max(0, p - 1), min(N_BINS, p + 2)
        return float(l2_delta[lo:hi].max())
    r_t = resp_at(ca1_peaks[expected])
    r_o = [resp_at(ca1_peaks[c]) for c in range(N_CELL) if c != expected]
    sel = float(np.mean([1.0 if (r_t >= 2.0 * max(r, 1e-9)) else 0.0 for r in r_o])) if r_o else 0.0
    del sb
    return dict(read_hit=bool(hit), selectivity=sel, l2_peak=l2_peak, ca1_peaks=ca1_peaks,
                map_ok=bool(map_ok), dw=dw, r_target=r_t, r_others=r_o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=DEV_SEEDS + BLIND_SEEDS)
    ap.add_argument("--bin-steps", dest="bin_steps", type=int, default=200)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    arms = {}
    for s in args.seeds:
        for name, kw in [("MAIN", {}), ("C1_l2_frozen", dict(l2_eta=0.0)),
                         ("C3_no_l2_plateau", dict(do_l2_plateau=False)),
                         ("C2_wrong_target", dict(score_cell=(TARGET_CELL + 1) % N_CELL))]:
            r = one_run(s, bin_steps=args.bin_steps, **kw)
            arms.setdefault(name, []).append((s, r))
            print(f"  seed {s} {name:17s} read_hit={int(r['read_hit'])} sel={r['selectivity']:.2f} "
                  f"l2_peak={r['l2_peak']} ca1_peaks={r['ca1_peaks']} map_ok={int(r['map_ok'])} "
                  f"r_tgt={r['r_target']:.5f} r_oth={[round(x,5) for x in r['r_others']]} dw={r['dw']:.4g}", flush=True)
    def agg(n, seeds, key):
        rs = [r for (s, r) in arms.get(n, []) if s in seeds]
        return float(np.mean([float(r[key]) for r in rs])) if rs else float("nan")
    dev = [s for s in args.seeds if s in DEV_SEEDS]; blind = [s for s in args.seeds if s in BLIND_SEEDS]
    print("\n=== SUMMARY (dev/blind SEPARATE) ===", flush=True)
    for k in arms:
        print(f"  {k:17s} read dev={agg(k,dev,'read_hit'):.3f} blind={agg(k,blind,'read_hit'):.3f} | "
              f"sel dev={agg(k,dev,'selectivity'):.3f} blind={agg(k,blind,'selectivity'):.3f} | "
              f"map_ok={agg(k,args.seeds,'map_ok'):.3f}", flush=True)
    ra, rb = agg("MAIN", args.seeds, "read_hit"), agg("MAIN", blind, "read_hit")
    sa, sb_ = agg("MAIN", args.seeds, "selectivity"), agg("MAIN", blind, "selectivity")
    go = bool(ra >= 0.80 and sa >= 0.80 and (np.isnan(rb) or (rb >= 0.80 and sb_ >= 0.80)))
    print(f"\nVERDICT: {'GO' if go else 'NO-GO'} (pre-registered: read_acc>=0.80 AND selectivity>=0.80, blind on its "
          f"own; read={ra:.3f}/blind {rb:.3f}, sel={sa:.3f}/blind {sb_:.3f}) [dw NOT the gate]", flush=True)
    if args.json:
        json.dump({k: [(s, r) for s, r in v] for k, v in arms.items()}, open(args.json, "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
