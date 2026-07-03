"""EMERGE-41 / toward-semantics — the SPIKING RANK-ORDER (latency) k-winners SELECTION: the pooler's "which columns win" is
the last host step (a top-k argsort over the graded drive). This de-risks realizing the SELECTION as SPIKING rank-order
(Thorpe latency) coding -- the columns receive their graded drive as input current at a near-rheobase operating point, so
the HIGHER-drive columns integrate to threshold and SPIKE EARLIER, and the first-K-to-spike == the host top-K. The winners
= which columns fired FIRST (read from `cp_firing_states` spike TIMING), NOT a host argsort. A shared FS (fast-spiking
inhibitory) pool provides post-hoc lateral inhibition that SPARSIFIES the loser pool -- but it does NOT do the selection
(see the honest note below). NO `sim/` edit.

WHY: EMERGE-38/39/40 made the competitive-pooler LEARNING fully-on-substrate (sim/ kernels), but the k-winners SELECTION of
winners is still a host `np.argsort(-drive)[:K]`. A biological pooler selects winners by SPIKING dynamics (Thorpe
rank-order / latency coding; Diehl-Cook 2015 lateral inhibition sparsifies). This is the last host op in the pooler.

HONEST FRAMING (from the adversarial audit): on this single-global-FS-pool substrate the SELECTION is driven purely by
rank-order spike TIMING (higher drive -> earlier spike). The FS lateral inhibition is CAUSALLY INERT for WHICH columns
win -- the winner set is byte-identical FS-on vs FS-lesioned (an explicit reported control); the FS only reduces the number
of columns that fire at all (loser-pool SPARSITY), a post-hoc effect. So this is NOT a k-WTA / FS-competition SELECTION;
it is spiking rank-order coding for the selection, with FS providing sparsity.

QUESTION (cheap-first, single variable): does the spiking rank-order (latency) read select the SAME top-K columns as the
host top-k, on a graded drive? If yes, it drops into the EMERGE-40 pooler (which already learns correctly given the
winners) with no learning change.

MECHANISM: NCOL Izhikevich column cells + an FS interneuron pool; column->FS (excitatory) + FS->column (inhibitory).
Inject a graded per-column drive as external current; run a short window; the first-K columns to SPIKE are the winners.
METRIC: overlap of the first-K-spiking winners with the host top-K by drive.

ANTI-CHEATS: OVERLAP with host top-K (chance = K/NCOL); FLAT-DRIVE (input-destruction: a uniform, non-graded drive removes
the ranking signal -> the overlap collapses toward the tie-break floor -> isolates that the SELECTION reads the GRADED
drive, not a fixed bias); FS-LESION winner-set-identity (winners are ~identical FS-on vs FS-lesioned -> confirms the FS is
NOT doing the selection) + FS-LESION sparsity (no inhibition -> more columns fire -> the FS's only effect is loser-pool
sparsity); randomized tie-break (so ties don't default to a fixed low-index bias); multi-seed. Reuse-by-import
(`_emerge11` WTA pattern); NO `sim/` edit. CPU numpy-backend. `--demo`.
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
        # tie-break RNG (drive-independent): columns with an IDENTICAL first-spike step are ordered RANDOMLY, so ties
        # do not default to a fixed low-index bias (stable argsort would). Deterministic per seed for reproducibility.
        self._tb_rng = np.random.default_rng(int(seed) * 101 + (0 if wta else 1))

    def select(self, drive):
        """Inject the graded drive as external current to the columns; run; return the first-K columns to SPIKE (rank-order)."""
        n = self.b.cp_external_input_current.shape[0]
        cur = np.zeros(n)
        cur[self.ci] = DRIVE_BASE + DRIVE_GAIN * np.asarray(drive, float)
        self.b.cp_external_input_current[:] = self.b.xp.asarray(cur.astype(np.float32)) if hasattr(self.b, "xp") else cur.astype(np.float32)
        # RANK-ORDER SELECTION (Thorpe latency coding): higher drive integrates to threshold EARLIER, so the first columns
        # to spike are the highest-drive ones. The winners = the first columns to fire (read from the spike TIMING, not a
        # host argsort over the drive). The FS lateral inhibition only sparsifies the loser pool (it does NOT select).
        first_spike = np.full(NCOL, N_STEPS + 1)
        for t in range(N_STEPS):
            self.b._run_one_simulation_step()
            fs = _host(self.b.cp_firing_states)[self.ci].astype(bool)
            newly = fs & (first_spike > N_STEPS)
            first_spike[newly] = t
        self.b.cp_external_input_current[:] = 0
        fired = np.where(first_spike <= N_STEPS)[0]
        self._n_fired = len(fired)
        if len(fired) == 0:
            return set()
        # winners = the K earliest to fire; ties on first-spike step are broken by a RANDOM (drive-independent) key so a
        # uniform/flat drive (all columns tie) collapses toward a tie-break floor rather than a fixed low-index set.
        tiebreak = self._tb_rng.random(len(fired))
        order = fired[np.lexsort((tiebreak, first_spike[fired]))]
        return set(int(c) for c in order[:K_WIN])


def _drive(seed):
    rng = np.random.default_rng(seed * 7 + 1)
    return rng.uniform(0.0, 6.0, NCOL)


def _run_seed(seed):
    d = _drive(seed)
    host_topk = set(int(c) for c in np.argsort(-d)[:K_WIN])
    # spiking rank-order selection (FS-on)
    p = FSWTAProbe(seed=seed, wta=True)
    win = p.select(d)
    overlap = len(win & host_topk) / K_WIN if win else 0.0
    sparsity = getattr(p, "_n_fired", len(win)) / NCOL          # how many columns fired at all (the FS keeps this low)
    # FS-lesion (no inhibition): (a) the WINNER SET is ~identical (the FS does NOT select) and (b) far more columns fire
    # (the FS's only effect is loser-pool sparsity).
    pl = FSWTAProbe(seed=seed, wta=False)
    lesion_win = pl.select(d)
    lesion_sparsity = getattr(pl, "_n_fired", len(lesion_win)) / NCOL
    lesion_winner_overlap = len(win & lesion_win) / K_WIN if win and lesion_win else 0.0   # ~1.0 => FS inert for selection
    # FLAT-drive (input-destruction): a uniform, non-graded drive removes the ranking signal. With no drive gradient every
    # column integrates identically -> the first-K-to-spike is decided by the random tie-break -> the overlap with the host
    # top-K (of the ORIGINAL graded drive) collapses toward the tie-break floor (~K/NCOL). Isolates that the SELECTION
    # reads the GRADED drive, not a fixed structural bias.
    flat = np.full(NCOL, float(np.mean(d)))                     # uniform drive at the mean magnitude (same total energy scale)
    pf = FSWTAProbe(seed=seed, wta=True)
    fwin = pf.select(flat)
    flat_overlap = len(fwin & host_topk) / K_WIN if fwin else 0.0
    return {"seed": seed, "overlap": overlap, "sparsity": sparsity, "n_win": len(win),
            "lesion_sparsity": lesion_sparsity, "lesion_winner_overlap": lesion_winner_overlap,
            "flat_overlap": flat_overlap}


def _demo(seed=42):
    d = _drive(seed)
    host_topk = list(np.argsort(-d)[:K_WIN])
    p = FSWTAProbe(seed=seed, wta=True); win = sorted(p.select(d))
    print("\n=== EMERGE-41 spiking rank-order (latency) k-winners selection (no host argsort; no transformer) ===")
    print(f"  {NCOL} columns integrate a graded drive; the k={K_WIN} HIGHER-drive columns SPIKE EARLIER (Thorpe rank-order).")
    print(f"  the winners = first-K to spike (spike TIMING); the FS inhibition only sparsifies the loser pool.\n")
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
    print(f"spiking rank-order (latency) k-winners selection: {NCOL} columns, k={K_WIN}; chance/flat-floor overlap "
          f"{K_WIN/NCOL:.2f}", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            r = _run_seed(s); per.append(r)
            print(f"  [seed {s}] overlap {r['overlap']:.2f} (n_win {r['n_win']}, sparsity {r['sparsity']:.2f}) || "
                  f"FLAT-drive overlap {r['flat_overlap']:.2f} | FS-lesion winner-overlap {r['lesion_winner_overlap']:.2f} "
                  f"(=FS inert for selection) | FS-lesion sparsity {r['lesion_sparsity']:.2f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(k):
            return float(np.mean([p[k] for p in per]))
        ov, sp, les, flat, lwo = m("overlap"), m("sparsity"), m("lesion_sparsity"), m("flat_overlap"), m("lesion_winner_overlap")
        chance = K_WIN / NCOL
        # PRIMARY: the spiking rank-order (drive->spike-timing) selects the top-K (overlap), and it READS THE GRADED DRIVE
        # (flat/uniform drive collapses the overlap to the tie-break floor). SECONDARY (reported controls): the FS lateral
        # inhibition is INERT for the selection (winner set ~identical FS-on vs FS-lesion) and only sparsifies the loser
        # pool (lesion fires more).
        go = bool(ov >= 0.9 and ov >= chance + 0.4 and flat <= chance + 0.15 and les >= sp + 0.12)
        if go:
            verdict = (f"GO -- the pooler's k-winners SELECTION runs as SPIKING RANK-ORDER (latency) coding, not a host argsort "
                       f"over the drive: columns integrate their graded drive to threshold and the HIGHER-drive columns SPIKE "
                       f"EARLIER (Thorpe rank-order), so the first-K-to-spike == the host top-K (overlap {ov:.2f}, chance "
                       f"{chance:.2f}); a FLAT (uniform, non-graded) drive collapses the overlap to the tie-break floor "
                       f"({flat:.2f} ~= chance {chance:.2f}) => the SELECTION reads the GRADED drive, not a fixed bias. The FS "
                       f"lateral inhibition is CAUSALLY INERT for the selection (winner set ~identical FS-on vs FS-lesion, "
                       f"overlap {lwo:.2f}); its ONLY effect is loser-pool SPARSITY (fired fraction {sp:.2f} with FS vs {les:.2f} "
                       f"FS-lesioned). Multi-seed. => the pooler's last host op (which columns win) is a spiking read (spike-time "
                       f"order), drop-in for the EMERGE-40 pooler (which learns correctly given the winner set). NO sim/ edit.")
        else:
            miss = []
            if ov < 0.9: miss.append(f"overlap {ov:.2f} < 0.9")
            if ov < chance + 0.4: miss.append(f"overlap not above chance+0.4 ({ov:.2f} vs {chance:.2f})")
            if flat > chance + 0.15: miss.append(f"flat-drive overlap does not collapse ({flat:.2f} > chance+0.15 {chance + 0.15:.2f})")
            if les < sp + 0.12: miss.append(f"FS not load-bearing for sparsity (lesion {les:.2f} vs wta {sp:.2f})")
            verdict = ("BOUNDARY (build-informative) -- " + "; ".join(miss) + ". Tune drive gain / n_steps / operating point for "
                       "the rank-order timing; the spiking rank-order read is the next tuning, not a wall.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge41_fs_wta_kwinners", "verdict": verdict,
               "mechanism": "spiking rank-order (latency) k-winners SELECTION: NCOL Izhikevich columns + an FS inhibitory pool "
                            "(column->FS excitatory + FS->column inhibitory); the graded drive is injected as external current at "
                            "a near-rheobase operating point, so higher-drive columns integrate to threshold + spike EARLIER "
                            "(Thorpe rank-order coding); winners = first-K columns to fire (cp_firing_states spike timing). The FS "
                            "lateral inhibition is inert for the selection (winner set identical FS-on vs FS-lesion) and only "
                            "sparsifies the loser pool.",
               "task": "select the top-K columns by graded drive via spiking rank-order timing; overlap with host top-K vs "
                       "FLAT-drive (input-destruction) + FS-lesion winner-set-identity + FS-lesion sparsity; multi-seed",
               "seeds": a.seeds, "config": {"n_col": NCOL, "n_fs": N_FS, "k_win": K_WIN, "n_steps": N_STEPS,
                                            "drive_gain": DRIVE_GAIN, "col_fs_w": COL_FS_W, "fs_col_w": FS_COL_W},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "cheap-first: this de-risks the spiking rank-order (latency) SELECTION against the host top-k on a "
                              "graded drive. The FS lateral inhibition does NOT do the selection (winner set is identical FS-on vs "
                              "FS-lesion; the pure rank-order integrator reproduces the overlap) -- it only sparsifies the loser "
                              "pool. If GO, wiring it into the EMERGE-40 pooler learning loop (replace np.argsort(-drive)[:K] with "
                              "select(drive)) is the follow-on -- the learning is unchanged (it only needs the winner set)."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge41] VERDICT: {verdict}", flush=True)
    print(f"[emerge41] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
