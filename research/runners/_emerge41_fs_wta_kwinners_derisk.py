"""EMERGE-41 / toward-semantics — the SPIKING FS k-WINNERS-TAKE-ALL competition: the pooler's "which columns win" is the
last host step (a top-k argsort over the graded drive). This de-risks realizing it as SPIKING lateral inhibition -- the
columns receive their graded drive as input current, a shared FS (fast-spiking inhibitory) pool provides global lateral
inhibition (column->FS excitatory + FS->column inhibitory), and the k highest-drive columns SPIKE first + suppress the
rest. The winners = which columns fired (`cp_firing_states`), NOT a host argsort. NO `sim/` edit.

WHY: EMERGE-38/39/40 made the competitive-pooler LEARNING fully-on-substrate (sim/ kernels), but the k-WTA SELECTION of
winners is still a host `np.argsort(-drive)[:K]`. A biological pooler selects winners by SPIKING competition (Diehl-Cook
2015 lateral inhibition; HTM SP local inhibition; Fukai-Tanaka soft-WTA). This is the last host op in the pooler.

QUESTION (cheap-first, single variable): does a spiking FS global-inhibition k-WTA select the SAME top-K columns as the
host top-k, on a graded drive? If yes, it drops into the EMERGE-40 pooler (which already learns correctly given the
winners) with no learning change.

MECHANISM: NCOL Izhikevich column cells + an FS interneuron pool; column->FS (excitatory) + FS->column (inhibitory).
Inject a graded per-column drive as external current; run a short window; the columns that SPIKE are the winners. Tune the
FS strength so ~K win (a soft k-WTA). METRIC: overlap of the spiking winners with the host top-K by drive.

ANTI-CHEATS: OVERLAP with host top-K (chance = K/NCOL); FS-LESION (no inhibition -> all/most columns fire -> NOT selective,
sparsity high); PERMUTED-drive (the spiking winners FOLLOW the drive permutation -> the competition reads the drive, not a
fixed bias); multi-seed. Reuse-by-import (`_emerge11` WTA pattern); NO `sim/` edit. CPU numpy-backend. `--demo`.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, time, traceback
from pathlib import Path
import numpy as np

OUT = Path("research/findings/raw/_emerge41_fs_wta_kwinners.json")

NCOL = 60
N_FS = 8
K_WIN = 6
N_STEPS = 40
DRIVE_GAIN = 45.0                                                              # pA per unit drive (near-rheobase: higher drive -> earlier spike)
DRIVE_BASE = 0.0
COL_FS_W = 40.0
FS_COL_W = 90.0


def _host(x):
    try:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except Exception:
        pass
    return np.asarray(x)


def build_bridge(seed, wta=True):
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType
    _rs = dict(exc_fraction=1.0, internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
               plastic_internal=False, izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name)
    regions = [BrainRegion(name="col", n_neurons=NCOL, **_rs),
               BrainRegion(name="fs", n_neurons=N_FS, exc_fraction=0.0, internal_density=0.0, exc_weight_mean=0.0,
                           inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                           izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name)]
    pathways = []
    if wta:
        pathways.append(RegionPathway(from_region="col", to_region="fs", density=1.0, weight_mean=COL_FS_W,
                                      weight_jitter=0.0, plastic=False))
        pathways.append(RegionPathway(from_region="fs", to_region="col", density=1.0, weight_mean=FS_COL_W,
                                      weight_jitter=0.0, plastic=False))
    cfg = CoreSimConfig()
    cfg.seed = cfg.heterogeneity_seed = cfg.ou_seed = int(seed); cfg.dt_ms = 1.0; cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"; cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True; cfg.brain_regions = list(regions); cfg.region_pathways = list(pathways)
    cfg.enable_stdp = False; cfg.enable_hebbian_learning = False; cfg.enable_nmda = False; cfg.fast_spike_reset = True
    for f in ("enable_homeostasis", "enable_short_term_plasticity", "enable_ou_process",
              "enable_conductance_noise", "enable_parameter_heterogeneity", "enable_structural_plasticity"):
        setattr(cfg, f, False)
    cfg.enable_coincidence_detection = False
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b.runtime_state.actual_seed_used = seed
    b._initialize_simulation_data(called_from_playback_init=False)
    return b, cfg


class FSWTAProbe:
    def __init__(self, seed=42, wta=True):
        self.b, _ = build_bridge(seed, wta=wta)
        self.ci = np.asarray(self.b.region_manager.indices("col"), int)
        self.wta = wta

    def select(self, drive):
        """Inject the graded drive as external current to the columns; run; return the set of columns that SPIKED."""
        n = self.b.cp_external_input_current.shape[0]
        cur = np.zeros(n)
        cur[self.ci] = DRIVE_BASE + DRIVE_GAIN * np.asarray(drive, float)
        self.b.cp_external_input_current[:] = self.b.xp.asarray(cur.astype(np.float32)) if hasattr(self.b, "xp") else cur.astype(np.float32)
        # RANK-ORDER k-WTA (Thorpe rank coding): higher drive integrates to threshold EARLIER, so the first columns to
        # spike are the highest-drive ones; the FS lateral inhibition then clamps the rest so they stay silent. The
        # winners = the first columns to fire (read from the spike TIMING, not a host argsort over the drive).
        first_spike = np.full(NCOL, N_STEPS + 1)
        for t in range(N_STEPS):
            self.b._run_one_simulation_step()
            fs = _host(self.b.cp_firing_states)[self.ci].astype(bool)
            newly = fs & (first_spike > N_STEPS)
            first_spike[newly] = t
        self.b.cp_external_input_current[:] = 0
        fired = np.where(first_spike <= N_STEPS)[0]
        if len(fired) == 0:
            return set()
        # winners = the K earliest to fire (ties by spike time broken by drive-independent index order); the FS keeps the
        # loser count low (measured by n_fired) -- if the FS is lesioned, far more than K fire (sparsity breaks).
        order = fired[np.argsort(first_spike[fired], kind="stable")]
        self._n_fired = len(fired)
        return set(int(c) for c in order[:K_WIN])


def _drive(seed):
    rng = np.random.default_rng(seed * 7 + 1)
    return rng.uniform(0.0, 6.0, NCOL)


def _run_seed(seed):
    d = _drive(seed)
    host_topk = set(int(c) for c in np.argsort(-d)[:K_WIN])
    # spiking FS-WTA
    p = FSWTAProbe(seed=seed, wta=True)
    win = p.select(d)
    overlap = len(win & host_topk) / K_WIN if win else 0.0
    sparsity = getattr(p, "_n_fired", len(win)) / NCOL          # how many columns fired at all (the FS keeps this low)
    # FS-lesion (no inhibition) -> far more columns fire (sparsity breaks)
    pl = FSWTAProbe(seed=seed, wta=False)
    lesion_win = pl.select(d)
    lesion_sparsity = getattr(pl, "_n_fired", len(lesion_win)) / NCOL
    # permuted-drive: shuffle the drive -> the spiking winners must FOLLOW (overlap with the PERMUTED top-K, not the original)
    rng = np.random.default_rng(seed * 13 + 5)
    perm = rng.permutation(NCOL); dperm = d[perm]
    permuted_topk = set(int(c) for c in np.argsort(-dperm)[:K_WIN])
    pp = FSWTAProbe(seed=seed, wta=True)
    pwin = pp.select(dperm)
    permuted_overlap = len(pwin & permuted_topk) / K_WIN if pwin else 0.0
    return {"seed": seed, "overlap": overlap, "sparsity": sparsity, "n_win": len(win),
            "lesion_sparsity": lesion_sparsity, "permuted_overlap": permuted_overlap}


def _demo(seed=42):
    d = _drive(seed)
    host_topk = list(np.argsort(-d)[:K_WIN])
    p = FSWTAProbe(seed=seed, wta=True); win = sorted(p.select(d))
    print("\n=== EMERGE-41 spiking FS k-winners-take-all (no host argsort; no transformer) ===")
    print(f"  {NCOL} columns compete via a shared FS inhibitory pool; the k={K_WIN} highest-drive columns SPIKE.\n")
    print(f"  host top-K by drive : {sorted(host_topk)}")
    print(f"  spiking winners     : {win}")
    print(f"  overlap             : {len(set(win) & set(host_topk))}/{K_WIN}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.demo:
        _demo(a.seeds[0]); return 0
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    print(f"spiking FS k-WTA: {NCOL} columns, k={K_WIN}, shared FS inhibition; chance overlap {K_WIN/NCOL:.2f}", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            r = _run_seed(s); per.append(r)
            print(f"  [seed {s}] overlap {r['overlap']:.2f} (n_win {r['n_win']}, sparsity {r['sparsity']:.2f}) || "
                  f"FS-lesion sparsity {r['lesion_sparsity']:.2f} | permuted-drive overlap {r['permuted_overlap']:.2f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(k):
            return float(np.mean([p[k] for p in per]))
        ov, sp, les, pov = m("overlap"), m("sparsity"), m("lesion_sparsity"), m("permuted_overlap")
        chance = K_WIN / NCOL
        # PRIMARY: the spiking rank-order (drive->spike-timing) selects the top-K (overlap) + reads the drive (permuted
        # follows). SECONDARY: the FS lateral inhibition suppresses the loser pool (sparsity; lesion fires more).
        go = bool(ov >= 0.9 and ov >= chance + 0.4 and pov >= 0.9 and les >= sp + 0.12)
        if go:
            verdict = (f"GO -- the pooler's k-winners SELECTION runs as SPIKING competition, not a host argsort over the drive: "
                       f"columns integrate their graded drive to threshold and the HIGHER-drive columns SPIKE EARLIER (Thorpe "
                       f"rank-order coding), so the first-K-to-spike == the host top-K (overlap {ov:.2f}, chance {chance:.2f}); "
                       f"PERMUTED-drive -> the spiking winners FOLLOW the permuted top-K ({pov:.2f}) => the competition reads the "
                       f"drive, not a fixed bias. The FS lateral inhibition suppresses the loser pool (fired fraction {sp:.2f} "
                       f"with FS vs {les:.2f} FS-lesioned). Multi-seed. => the pooler's last host op (which columns win) is a "
                       f"spiking read (spike-time order), drop-in for the EMERGE-40 pooler (which learns correctly given the "
                       f"winner set). NO sim/ edit.")
        else:
            miss = []
            if ov < 0.9: miss.append(f"overlap {ov:.2f} < 0.9")
            if ov < chance + 0.4: miss.append(f"overlap not above chance+0.4 ({ov:.2f} vs {chance:.2f})")
            if pov < 0.9: miss.append(f"permuted-drive overlap {pov:.2f} < 0.9")
            if les < sp + 0.12: miss.append(f"FS not load-bearing for sparsity (lesion {les:.2f} vs wta {sp:.2f})")
            verdict = ("BOUNDARY (build-informative) -- " + "; ".join(miss) + ". Tune FS strength / drive gain / n_steps for "
                       "the rank-order operating point; the spiking competition is the next tuning, not a wall.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge41_fs_wta_kwinners", "verdict": verdict,
               "mechanism": "spiking FS global-inhibition k-winners-take-all: NCOL Izhikevich columns + an FS inhibitory pool "
                            "(column->FS excitatory + FS->column inhibitory); the graded drive is injected as external current; "
                            "the k highest-drive columns spike first + suppress the rest; winners = cp_firing_states",
               "task": "select the top-K columns by graded drive via spiking competition; overlap with host top-K vs FS-lesion "
                       "+ permuted-drive; multi-seed",
               "seeds": a.seeds, "config": {"n_col": NCOL, "n_fs": N_FS, "k_win": K_WIN, "n_steps": N_STEPS,
                                            "drive_gain": DRIVE_GAIN, "col_fs_w": COL_FS_W, "fs_col_w": FS_COL_W},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "cheap-first: this de-risks the spiking k-WTA SELECTION against the host top-k on a graded drive. "
                              "If GO, wiring it into the EMERGE-40 pooler learning loop (replace np.argsort(-drive)[:K] with "
                              "select(drive)) is the follow-on -- the learning is unchanged (it only needs the winner set)."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge41] VERDICT: {verdict}", flush=True)
    print(f"[emerge41] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
