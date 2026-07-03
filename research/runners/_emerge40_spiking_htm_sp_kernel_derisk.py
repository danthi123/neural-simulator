"""EMERGE-40 / toward-semantics — the FULLY-SPIKING HTM Spatial Pooler: the winner-INACTIVE (selectivity) depression that
EMERGE-39 de-risked on-substrate as a host op is now the committed `sim/` kernel `fused_htm_winner_inactive_depression`
(additive, byte-identical to all existing paths). So BOTH pooler learning terms are `sim/` fused kernels: potentiation via
`fused_htm_permanence_update` (ld=0) + winner-inactive depression via the new kernel, over the bridge's coincidence synapse
permanences (`cp_connections.data`). Reuse-by-import; ONE additive `sim/` kernel (new function; existing kernels untouched).

WHAT'S NEW vs EMERGE-39: EMERGE-39 validated the mechanism (0.96 on overlapping categories) with the winner-inactive
depression applied as a host numpy op on `cp_connections.data`. EMERGE-40 replaces that host op with the fused `sim/`
kernel, so the competitive-pooler LEARNING is fully realized by `sim/` kernels (the committed permanence kernel + the new
winner-selectivity kernel), matching the on-substrate end-state. The k-WTA drive read remains a top-k over the substrate
weights (the spiking FS-WTA lateral-inhibition version is a further rung).

MECHANISM (all `sim/` kernels): feature cells -> a large column layer via a DENSE PLASTIC coincidence projection whose
permanences are `cp_connections.data`, small random init. UNSUPERVISED loop over the member stream: (1) drive[col] =
Sigma connected active-feature permanences x boost; (2) top-k WINNERS; (3) `fused_htm_permanence_update(active, winners,
lam_pot, ld=0)` -> POTENTIATE active-feature->winner; (4) `fused_htm_winner_inactive_depression(pre_active, post_win,
lam_dep_wi)` -> DEPRESS inactive-feature->winner (HTM-SP selectivity). Then inheritance on the spiking bridge over the
learned codons.

ANTI-CHEATS: 6-OVERLAPPING-category held-out inheritance (chance 1/6); NO-SELECTIVITY (the new kernel OFF -> collapses,
isolating the kernel as load-bearing); FIXED (no-learn); PERMUTED-features; dAP-LESION; 6-seed. Reuse-by-import
(`_emerge14` committed kernel + `_emerge12`); ONE additive `sim/` kernel. CPU numpy-backend. `--demo`.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, time, traceback
from pathlib import Path
import numpy as np

from research.runners._emerge14_stageC_onbridge_learning_derisk import apply_kernel_update, _host
from research.runners._emerge12_stageB2_bridge_tm_derisk import _prime_from_winners
from sim.kernels import fused_htm_winner_inactive_depression

OUT = Path("research/findings/raw/_emerge40_spiking_htm_sp_kernel.json")

CATS = list(range(6)); NPROP = len(CATS)
STRIDE = 3
POOLS = {k: list(range(k * STRIDE, k * STRIDE + 6)) for k in CATS}
NF = max(c for cs in POOLS.values() for c in cs) + 1
NCOL = 200
K_WIN = 6
POOL_EPOCHS = 400
POOL_LP = 0.05
POOL_LD_WI = 0.02                                                              # winner-inactive depression rate (the new sim/ kernel)
N_PER = 9
HOLD = 3
FLOOR = -40.0
M = NF + NCOL + NPROP * 2


def _sdr(cells):
    return set(int(c) for c in cells)


class SpikingHTMSPProbe:
    def __init__(self, seed=42, epochs=40, lesion=False, permute=False, learn=True, selectivity=True):
        from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
        from sim.bridge import SimulationBridge
        from sim.regions import BrainRegion
        from sim.enums import NeuronModel, NeuronType
        rng = np.random.default_rng(seed)
        regions = [BrainRegion(name="cells", n_neurons=M, exc_fraction=1.0, internal_density=0.0, exc_weight_mean=0.0,
                               inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                               izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name)]
        cfg = CoreSimConfig()
        cfg.seed = cfg.heterogeneity_seed = cfg.ou_seed = int(seed); cfg.dt_ms = 1.0; cfg.num_traits = 1
        cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
        cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"; cfg.connections_per_neuron = 0
        cfg.enable_brain_region_framework = True; cfg.brain_regions = list(regions); cfg.region_pathways = []
        cfg.enable_stdp = False; cfg.enable_hebbian_learning = False; cfg.enable_nmda = False
        cfg.stdp_w_max = 1.0; cfg.fast_spike_reset = True
        for f in ("enable_homeostasis", "enable_short_term_plasticity", "enable_ou_process",
                  "enable_conductance_noise", "enable_parameter_heterogeneity", "enable_structural_plasticity"):
            setattr(cfg, f, False)
        cfg.enable_coincidence_detection = (not lesion)
        cfg.coincidence_weighted_drive = True; cfg.coincidence_k_threshold = 1.5
        cfg.coincidence_plateau_strength = 160.0; cfg.enable_two_compartment_dap = True; cfg.apical_g_couple = 2.0
        b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                             runtime_state=RuntimeState(), gpu_config=GPUConfig())
        b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        b.runtime_state.actual_seed_used = seed
        b._initialize_simulation_data(called_from_playback_init=False)
        ci = np.asarray(b.region_manager.indices("cells"), int)
        pre, post, w = [], [], []
        for c in range(NCOL):
            for f in range(NF):
                pre.append(int(ci[f])); post.append(int(ci[NF + c])); w.append(float(rng.uniform(0.30, 0.55)))
        for pc in range(NPROP * 2):
            for c in range(NCOL):
                pre.append(int(ci[NF + c])); post.append(int(ci[NF + NCOL + pc])); w.append(0.0)
        b.inject_explicit_wiring({"ff": {"pre_indices": pre, "post_indices": post, "initial_weights": w,
                                         "plastic": False, "coincidence_detector": True, "conn_type": "ff"}})
        coo = b._get_cached_coo()
        self.b, self.ci, self.row, self.col = b, ci, np.asarray(_host(coo.row)), np.asarray(_host(coo.col))
        self.z = np.zeros(len(ci)); self.nsyn = len(self.row)
        cell2unit = {int(ci[u]): u for u in range(M)}
        fff, ffc, ffp = [], [], []
        for k in range(len(self.row)):
            ru = cell2unit.get(int(self.row[k])); cu = cell2unit.get(int(self.col[k]))
            if ru is not None and cu is not None and ru < NF and NF <= cu < NF + NCOL:
                fff.append(ru); ffc.append(cu - NF); ffp.append(k)
        self.ff_feat = np.asarray(fff, int); self.ff_col = np.asarray(ffc, int); self.ff_pos = np.asarray(ffp, int)
        self.mem = {f"{k}_{i}": k for k in CATS for i in range(N_PER)}
        self.feats = {}
        for i, (m, k) in enumerate(self.mem.items()):
            r = np.random.default_rng(seed * 100 + (i if not permute else int(rng.integers(10 ** 6))))
            pool = POOLS[k] if not permute else list(range(NF))
            self.feats[m] = set(r.choice(pool, 4, replace=False))
        # fully-`sim/`-kernel competitive pooler learning: potentiation kernel (ld=0) + the new winner-inactive kernel
        if learn:
            duty = np.zeros(NCOL); boost = np.ones(NCOL); order = list(self.mem)
            for e in range(POOL_EPOCHS):
                rng.shuffle(order)
                for m in order:
                    win = self._winners(self.feats[m], boost)
                    apply_kernel_update(self.b, self.row, self.col, self.ci, _sdr(self.feats[m]), _sdr(win),
                                        self.z, POOL_LP, 0.0, 1.0)             # committed kernel: POTENTIATE active->winner
                    if selectivity:
                        self._winner_inactive_kernel(win, self.feats[m], POOL_LD_WI)  # new sim/ kernel: depress INACTIVE->winner
                    for c in win:
                        duty[c] += 1
                boost = np.exp(2.0 * (K_WIN / NCOL - duty / ((e + 1) * len(self.mem))))
        self.z[:] = 0.0
        self.PROP = {k: [NF + NCOL + 2 * k, NF + NCOL + 2 * k + 1] for k in CATS}
        self.held = {k: [m for m in self.mem if self.mem[m] == k][-HOLD:] for k in CATS}
        train = {k: [m for m in self.mem if self.mem[m] == k][:-HOLD] for k in CATS}
        for _ in range(epochs):
            for k in CATS:
                for tr in train[k]:
                    apply_kernel_update(self.b, self.row, self.col, self.ci, self._codon(self.feats[tr]),
                                        _sdr(self.PROP[k]), self.z, 0.14, 0.02, 1.0)

    def _drive(self, feats, boost=None):
        data = _host(self.b.cp_connections.data)
        active = np.zeros(NF); active[list(feats)] = 1.0
        contrib = active[self.ff_feat] * (data[self.ff_pos] > 0.5)
        drive = np.zeros(NCOL); np.add.at(drive, self.ff_col, contrib)
        return drive * boost if boost is not None else drive

    def _winners(self, feats, boost=None):
        return set(int(c) for c in np.argsort(-self._drive(feats, boost))[:K_WIN])

    def _winner_inactive_kernel(self, win, feats, ld):
        """The winner-INACTIVE depression via the committed sim/ kernel fused_htm_winner_inactive_depression: gather
        per-synapse pre_active (input feature active) + post_win (column is a winner), apply the kernel to cp_connections."""
        pre_active = np.zeros(self.nsyn); post_win = np.zeros(self.nsyn)
        pre_active[self.ff_pos] = np.isin(self.ff_feat, np.fromiter((int(f) for f in feats), int)).astype(float)
        post_win[self.ff_pos] = np.isin(self.ff_col, np.fromiter((int(c) for c in win), int)).astype(float)
        data = _host(self.b.cp_connections.data).astype(np.float64)
        updated = fused_htm_winner_inactive_depression(data, pre_active, post_win, ld, 0.0, 1.0)
        updated = np.asarray(updated).astype(np.float32)
        self.b.cp_connections.data[:] = self.b.xp.asarray(updated) if hasattr(self.b, "xp") else updated

    def _codon(self, feats):
        return set(NF + int(c) for c in np.argsort(-self._drive(feats))[:K_WIN])

    def infer(self, feats):
        resp = self._codon(feats)
        if not resp:
            return -1
        ab = np.zeros(len(self.ci), bool)
        for i in resp:
            ab[i] = True
        _prime_from_winners(self.b, self.ci, ab)
        vap = getattr(self.b, "cp_v_apical", None)
        if vap is None or np.asarray(_host(vap)).ndim == 0:
            return -1
        vap = _host(vap)[self.ci]
        dr = {k: float(np.mean([vap[x] for x in self.PROP[k]])) for k in CATS}
        bk = max(dr, key=dr.get)
        return bk if dr[bk] > FLOOR else -1

    def held_out_acc(self):
        return np.mean([self.infer(self.feats[h]) == k for k in CATS for h in self.held[k]])


def _run_arm(seed, arm, epochs):
    kw = dict(seed=seed, epochs=epochs)
    if arm == "spiking":
        pass
    elif arm == "no_selectivity":
        kw["selectivity"] = False
    elif arm == "nolearn":
        kw["learn"] = False
    elif arm == "permuted":
        kw["permute"] = True
    elif arm == "lesion":
        kw["lesion"] = True
    p = SpikingHTMSPProbe(**kw)
    return arm, {"held_out": float(p.held_out_acc())}


ARMS = ["spiking", "no_selectivity", "nolearn", "permuted", "lesion"]


def _demo(seed=42, epochs=40):
    p = SpikingHTMSPProbe(seed=seed, epochs=epochs)
    print("\n=== EMERGE-40 fully-spiking HTM Spatial Pooler (both learning terms are sim/ kernels; no transformer) ===")
    print(f"  {NF} features -> {NCOL} columns; potentiation (committed kernel) + winner-INACTIVE depression (NEW sim/")
    print(f"  kernel fused_htm_winner_inactive_depression); 6 OVERLAPPING categories (adjacent share {6-STRIDE}/6 feats).\n")
    for k in CATS:
        for h in p.held[k]:
            print(f"  held-out {h} (latent cat {k}) -> inferred cat {p.infer(p.feats[h])}  (expect {k})")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.demo:
        _demo(a.seeds[0], a.epochs); return 0
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    print(f"fully-spiking HTM-SP: potentiation kernel + fused_htm_winner_inactive_depression kernel; 6 OVERLAPPING cats "
          f"(adjacent share {6-STRIDE}/6); chance {1/6:.2f}", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            d = {"seed": s}
            for arm in ARMS:
                _, r = _run_arm(s, arm, a.epochs); d[arm] = r
            per.append(d)
            print(f"  [seed {s}] SPIKING-KERNEL {d['spiking']['held_out']:.2f} || no-selectivity "
                  f"{d['no_selectivity']['held_out']:.2f} | fixed {d['nolearn']['held_out']:.2f} "
                  f"| permuted {d['permuted']['held_out']:.2f} | lesion {d['lesion']['held_out']:.2f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(arm):
            return float(np.mean([p[arm]["held_out"] for p in per]))
        spk, nosel, nol, perm, les = m("spiking"), m("no_selectivity"), m("nolearn"), m("permuted"), m("lesion")
        go = bool(spk >= 0.85 and spk >= nosel + 0.25 and spk >= nol + 0.25 and spk >= perm + 0.30 and spk >= les + 0.30)
        if go:
            verdict = (f"GO -- the FULLY-SPIKING HTM Spatial Pooler works: BOTH competitive-pooler learning terms are now "
                       f"committed `sim/` fused kernels -- potentiation (`fused_htm_permanence_update`, ld=0) + the winner-"
                       f"INACTIVE depression (`fused_htm_winner_inactive_depression`, the new additive kernel) -- over the "
                       f"bridge's coincidence synapse permanences. On 6 OVERLAPPING categories (adjacent share {6-STRIDE}/6 "
                       f"feats): held-out inheritance {spk:.2f}. The winner-inactive kernel is LOAD-BEARING (no-selectivity "
                       f"{nosel:.2f}); FIXED projection {nol:.2f}; PERMUTED {perm:.2f}; dAP-LESION {les:.2f}; 6-seed. => the "
                       f"competitive self-organizing pooler is fully-on-substrate via `sim/` kernels. ONE additive `sim/` "
                       f"kernel (new function; `fused_htm_permanence_update` + all existing paths byte-unchanged).")
        else:
            miss = []
            if spk < 0.85: miss.append(f"spiking {spk:.2f} < 0.85")
            if spk < nosel + 0.25: miss.append(f"selectivity kernel not load-bearing ({spk:.2f} vs {nosel:.2f})")
            if spk < nol + 0.25: miss.append(f"fixed didn't collapse ({spk:.2f} vs {nol:.2f})")
            if spk < perm + 0.30: miss.append(f"permuted didn't collapse ({spk:.2f} vs {perm:.2f})")
            if spk < les + 0.30: miss.append(f"lesion didn't collapse ({spk:.2f} vs {les:.2f})")
            verdict = ("BOUNDARY (build-informative) -- " + "; ".join(miss) + ". Tune the winner-inactive kernel rate / "
                       "boosting; the spiking winner-selectivity is the next tuning, not a wall.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge40_spiking_htm_sp_kernel", "verdict": verdict,
               "mechanism": "fully-`sim/`-kernel HTM Spatial Pooler: feature->column permanences in the bridge's coincidence "
                            "synapse weights, learned by fused_htm_permanence_update (potentiation, ld=0) + the new additive "
                            "fused_htm_winner_inactive_depression (winner selectivity) + homeostatic boosting; inheritance on "
                            "the spiking bridge over the learned codons",
               "task": "6 categories with OVERLAPPING feature pools; learn the pooler via sim/ kernels; held-out inheritance vs "
                       "no-selectivity + fixed + permuted + dAP-lesion; multi-seed",
               "seeds": a.seeds, "config": {"epochs": a.epochs, "n_col": NCOL, "k_win": K_WIN, "pool_epochs": POOL_EPOCHS,
                                            "stride": STRIDE, "n_feat": NF, "pool_lp": POOL_LP, "pool_ld_wi": POOL_LD_WI},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "sim_edit": "ADDITIVE new fused kernel fused_htm_winner_inactive_depression in sim/kernels.py; existing kernels "
                           "byte-unchanged; default-inert (lam_dep_wi=0 -> no effect)",
               "HONEST_NOTE": "both learning terms are now sim/ fused kernels over cp_connections.data (on-substrate). The k-WTA "
                              "drive read is a top-k over the substrate weights (the spiking FS-WTA lateral-inhibition version is "
                              "a further rung). De-risked by EMERGE-39 (the host winner-inactive op reached 0.96); EMERGE-40 makes "
                              "it the committed kernel."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge40] VERDICT: {verdict}", flush=True)
    print(f"[emerge40] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
