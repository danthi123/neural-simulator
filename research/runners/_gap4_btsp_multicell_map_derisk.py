"""gap#4 RUNG 2 — does the one-shot BTSP rule scale from ONE cell to a POPULATION forming DISTINCT fields?

WHY. 2026-07-20 established (6-seed, dev+blind 1.00, width 3/20, all controls collapsing, eligibility-tau ablation
load-bearing) that a single CA1 cell LEARNS a localized place field from ONE plateau via thresholded BTSP. That is one
cell and one plateau. Every downstream use -- a population code, a map, anything a later layer could read -- requires
N cells to acquire N DIFFERENT fields WITHOUT interfering. This is the prerequisite rung toward gap#4's still-open
deep-credit frontier, and it is the cheapest decisive test of whether the rule composes.

THE TASK. N_CELL CA1 cells share the same 20-bin position input. Each cell c receives its OWN plateau at its OWN
target bin b_c during ONE induction lap (plateaus delivered per-cell, all in the same lap -- so the inputs are shared
and the only thing distinguishing the cells is WHICH plateau each received). Afterwards, with plasticity OFF, each
cell should fire selectively near ITS OWN b_c.

WHAT COULD GO WRONG (the failure this test exists to catch): the position pools are SHARED, so every cell's
potentiation writes onto the same presynaptic inputs. If the rule cannot keep per-cell credit separate, all cells
converge to the same field (or to mush) -- a weight change that looked fine on one cell but does not compose.

PRE-REGISTERED GATE (filed before any multi-cell result exists):
    GO iff  per_cell_acc >= 0.80   (fraction of cells whose peak bin is in ITS OWN backward window -5..+1)
       AND  distinctness >= 0.80   (fraction of CELL PAIRS whose peak bins differ by > 2 bins)
       AND  blind seeds pass on their own
`distinctness` is the load-bearing new metric: per_cell_acc alone can be gamed by every cell learning the SAME field
if the targets happen to be close, so pair-separation is gated explicitly.

CONTROLS: C1 frozen (eta=0) -> no fields; C3 no-plateau moat -> no fields, dw==0; C2 SHUFFLED target assignment
(score cell c against another cell's b -> must drop to chance); C6 dev/blind reported SEPARATELY; C7 width guard;
C9 dw reported but NOT gated. Seeds must produce DIFFERENT networks (the n=1 trap that already bit this arc once).

NO new `sim/` edit -- reuses the committed thresholded-depression kernel (btsp_hetero_theta, default-off).
Run: SIM_BACKEND=numpy python -m research.runners._gap4_btsp_multicell_map_derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import argparse, json
import numpy as np

N_BINS = 20
POS_N = 10
CELL_TARGETS = [5, 9, 13, 17]          # PRE-REGISTERED: 4 cells, targets >=4 bins apart so distinctness is measurable
N_CELL = len(CELL_TARGETS)
CA1_PER_CELL = 8                        # each "cell" is a small pool (the single-cell result used 8)
DEV_SEEDS = [42, 43, 44]
BLIND_SEEDS = [100, 101, 102]


def build(seed, *, eta=0.02, hdep=0.3, htheta=0.012, elig_tau=1000.0, w0=0.6, wj=0.15, btsp=True, dt=1.0):
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    cfg = CoreSimConfig(seed=seed)
    cfg.dt_ms = float(dt)
    cfg.num_traits = 1                      # the cell-type lottery confound (2026-07-20 research gate)
    cfg.enable_brain_region_framework = True
    for f in ("enable_short_term_plasticity", "enable_hebbian_learning", "enable_homeostasis",
              "enable_structural_plasticity", "enable_reward_modulation", "enable_stdp",
              "enable_nmda", "enable_ou_process", "enable_parameter_heterogeneity",
              "enable_conductance_noise", "enable_synaptic_scaling"):
        if hasattr(cfg, f):
            setattr(cfg, f, False)
    cfg.enable_bdsp = True; cfg.bdsp_learning_rate = 0.0; cfg.bdsp_apical_bistable = True
    cfg.coincidence_plateau_self_regen = 2.0; cfg.coincidence_plateau_v_hold = -35.0; cfg.apical_kir_g = 1.0
    cfg.enable_btsp = bool(btsp)
    cfg.btsp_learning_rate = float(eta); cfg.btsp_elig_tau_ms = float(elig_tau)
    cfg.btsp_w_min, cfg.btsp_w_max = 0.0, 5.0
    cfg.btsp_hetero_dep = float(hdep); cfg.btsp_hetero_theta = float(htheta)
    cfg.brain_regions = [
        BrainRegion(name=f"pos{k}", n_neurons=POS_N, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False)
        for k in range(N_BINS)
    ] + [
        BrainRegion(name=f"ca1_{c}", n_neurons=CA1_PER_CELL, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False)
        for c in range(N_CELL)
    ]
    # SHARED position inputs -> every cell (this is what makes interference possible)
    cfg.region_pathways = [
        RegionPathway(from_region=f"pos{k}", to_region=f"ca1_{c}", density=1.0,
                      weight_mean=float(w0), weight_jitter=float(wj), plastic=True)
        for k in range(N_BINS) for c in range(N_CELL)
    ]
    rt = RuntimeState(); rt.actual_seed_used = seed
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                          runtime_state=rt, gpu_config=GPUConfig())
    sb._initialize_simulation_data()
    pos = [np.asarray(sb.region_manager.indices(f"pos{k}")) for k in range(N_BINS)]
    cells = [np.asarray(sb.region_manager.indices(f"ca1_{c}")) for c in range(N_CELL)]
    return sb, pos, cells


def run_lap(sb, pos, cells, *, targets=None, bin_steps=200, drive_pA=900.0,
            plateau_pA=600.0, release_pA=900.0, plateau_hold_ms=700.0, pulse_steps=15, record=False):
    """One lap. If `targets` given (list of per-cell plateau bins, None = no plateau for that cell), deliver each
    cell's plateau at ITS bin during this single lap -- shared inputs, per-cell instructive signals."""
    from sim.backend import to_host, get_backend
    xp, _ = get_backend()
    n = int(sb.core_config.num_neurons)
    rates = np.zeros((N_CELL, N_BINS)); step = 0
    starts = [None] * N_CELL
    for k in range(N_BINS):
        for s in range(bin_steps):
            cur = np.zeros(n, np.float32); cur[pos[k]] = drive_pA
            sb.cp_external_input_current[:] = 0.0
            sb.cp_external_input_current[:] = (xp.asarray(cur) if xp is not None else cur)
            ap = np.zeros(n, np.float32)
            if targets is not None:
                for c, b in enumerate(targets):
                    if b is None:
                        continue
                    if k == b and s < pulse_steps:
                        ap[cells[c]] = plateau_pA
                        if starts[c] is None:
                            starts[c] = step
                    if starts[c] is not None and 0 <= step - (starts[c] + int(plateau_hold_ms)) < 20:
                        ap[cells[c]] = -release_pA
            sb.cp_bdsp_apical_drive = (xp.asarray(ap) if xp is not None else ap)
            sb._run_one_simulation_step()
            if record:
                fs = np.asarray(to_host(sb.cp_firing_states))
                for c in range(N_CELL):
                    rates[c, k] += float(fs[cells[c]].sum())
            step += 1
    if record:
        rates /= float(CA1_PER_CELL * bin_steps)
    return rates


def peak_of(d):
    return int(np.argmax(d)) if d.max() > 0 else -1


def one_run(seed, *, do_plateau=True, eta=0.02, score_shuffled=False, deliver_shuffled=False, bin_steps=200):
    from sim.backend import to_host
    sb, pos, cells = build(seed, eta=eta)
    # deliver_shuffled is a GENUINE MANIPULATION: it permutes WHICH BIN each cell's plateau is delivered at,
    # so the whole run re-executes under a different cause. (score_shuffled only re-indexes the SCORING of an
    # unchanged run -- it is identical to MAIN by construction and carries ZERO evidential weight. That defect
    # was diagnosed in rung 3 and is BACK-PORTED here; the audit found rung 2 had committed it 26 min earlier.)
    _sh_d = N_CELL // 2
    _delivered = ([CELL_TARGETS[(c + _sh_d) % N_CELL] for c in range(N_CELL)]
                  if deliver_shuffled else list(CELL_TARGETS))
    targets = _delivered if do_plateau else [None] * N_CELL
    pre = run_lap(sb, pos, cells, targets=None, bin_steps=bin_steps, record=True)
    w0 = float(np.abs(np.asarray(to_host(sb.cp_connections.data))).sum())
    run_lap(sb, pos, cells, targets=targets, bin_steps=bin_steps, record=False)
    dw = float(np.abs(np.asarray(to_host(sb.cp_connections.data))).sum()) - w0
    sb.core_config.enable_btsp = False
    post = run_lap(sb, pos, cells, targets=None, bin_steps=bin_steps, record=True)
    delta = post - pre
    peaks = [peak_of(delta[c]) for c in range(N_CELL)]
    # C2: score cell c against ANOTHER cell's target (shuffled assignment) -> must fall to chance
    # C2 shuffle must shift by N_CELL//2, NOT 1: targets are 4 bins apart while the scoring window spans 7
    # (-5..+1), so a shift-by-1 puts each peak inside the NEIGHBOURING target's window and the control reads
    # 0.75 instead of chance -- a flaw in the CONTROL's geometry, not in the mechanism. Shift-by-2 separates
    # them maximally (peak-to-scored-target offset ~ +/-9, well outside the window).
    _sh = max(1, N_CELL // 2)
    # Scoring reference: when the delivery was shuffled we can score EITHER against what was actually
    # delivered (must PASS -- peaks follow delivery) or against the original targets (must COLLAPSE).
    score_t = ([CELL_TARGETS[(c + _sh) % N_CELL] for c in range(N_CELL)] if score_shuffled
               else (CELL_TARGETS if deliver_shuffled else _delivered))
    hits = []
    for c in range(N_CELL):
        if peaks[c] < 0:
            hits.append(False); continue
        off = (peaks[c] - score_t[c] + N_BINS // 2) % N_BINS - N_BINS // 2
        hits.append(bool(-5 <= off <= 1))
    # distinctness: fraction of cell PAIRS whose peaks differ by > 2 bins
    pairs = [(i, j) for i in range(N_CELL) for j in range(i + 1, N_CELL)]
    dist = []
    for i, j in pairs:
        if peaks[i] < 0 or peaks[j] < 0:
            dist.append(False); continue
        d = min(abs(peaks[i] - peaks[j]), N_BINS - abs(peaks[i] - peaks[j]))
        dist.append(bool(d > 2))
    widths = []
    for c in range(N_CELL):
        d = delta[c]
        widths.append(int((d >= 0.5 * d.max()).sum()) if d.max() > 0 else 0)
    del sb
    return dict(per_cell_acc=float(np.mean(hits)), distinctness=float(np.mean(dist)),
                peaks=peaks, targets=CELL_TARGETS, dw=dw, mean_width=float(np.mean(widths)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=DEV_SEEDS + BLIND_SEEDS)
    ap.add_argument("--bin-steps", dest="bin_steps", type=int, default=200)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    arms = {}
    for s in args.seeds:
        arms.setdefault("MAIN", []).append((s, one_run(s, bin_steps=args.bin_steps)))
        arms.setdefault("C1_frozen", []).append((s, one_run(s, eta=0.0, bin_steps=args.bin_steps)))
        arms.setdefault("C3_moat", []).append((s, one_run(s, do_plateau=False, bin_steps=args.bin_steps)))
        arms.setdefault("C2_shuffled", []).append((s, one_run(s, score_shuffled=True, bin_steps=args.bin_steps)))
        # BACK-PORTED genuine controls (the pair that can actually fail):
        #   C2r_deliver_shuffled_score_orig : plateaus MOVED, scored against the ORIGINAL targets -> must COLLAPSE
        #   C2p_deliver_shuffled_score_moved: plateaus MOVED, scored against WHERE THEY WERE MOVED TO -> must PASS
        arms.setdefault("C2r_deliver_shuffled_score_orig", []).append(
            (s, one_run(s, deliver_shuffled=True, bin_steps=args.bin_steps)))
        arms.setdefault("C2p_deliver_shuffled_score_moved", []).append(
            (s, one_run(s, deliver_shuffled=True, score_shuffled=True, bin_steps=args.bin_steps)))
        for k, v in arms.items():
            if v[-1][0] == s:
                r = v[-1][1]
                print(f"  seed {s} {k:12s} per_cell_acc={r['per_cell_acc']:.2f} distinct={r['distinctness']:.2f} "
                      f"peaks={r['peaks']} targets={r['targets']} width={r['mean_width']:.1f} dw={r['dw']:.4g}",
                      flush=True)

    def agg(name, seeds, key):
        rs = [r for (s, r) in arms.get(name, []) if s in seeds]
        return float(np.mean([r[key] for r in rs])) if rs else float("nan")

    dev = [s for s in args.seeds if s in DEV_SEEDS]; blind = [s for s in args.seeds if s in BLIND_SEEDS]
    print("\n=== SUMMARY (dev and blind SEPARATE) ===", flush=True)
    for k in arms:
        print(f"  {k:12s} acc dev={agg(k,dev,'per_cell_acc'):.3f} blind={agg(k,blind,'per_cell_acc'):.3f} | "
              f"distinct dev={agg(k,dev,'distinctness'):.3f} blind={agg(k,blind,'distinctness'):.3f}", flush=True)
    ma, mb = agg("MAIN", args.seeds, "per_cell_acc"), agg("MAIN", blind, "per_cell_acc")
    da, db = agg("MAIN", args.seeds, "distinctness"), agg("MAIN", blind, "distinctness")
    go = bool(ma >= 0.80 and da >= 0.80 and (np.isnan(mb) or (mb >= 0.80 and db >= 0.80)))
    print(f"\nVERDICT: {'GO' if go else 'NO-GO'} (pre-registered: per_cell_acc>=0.80 AND distinctness>=0.80, "
          f"blind on its own; acc={ma:.3f}/blind {mb:.3f}, distinct={da:.3f}/blind {db:.3f}) [dw NOT the gate]",
          flush=True)
    if args.json:
        json.dump({k: [(s, r) for s, r in v] for k, v in arms.items()}, open(args.json, "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
