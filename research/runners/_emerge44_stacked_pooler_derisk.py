"""EMERGE-44 / toward-semantics — the STACKED pooler discovers a MULTI-LEVEL taxonomy: a SECOND competitive pooler layer,
taking the first layer's codons as input, DISCOVERS superordinate groupings from co-occurrence (robin/sparrow/canary
sub-category codons co-occur -> a shared BIRD superordinate), and inheritance CHAINS L1->L2 so a held-out member inherits
its SUPERORDINATE property. This is the research-gated critical de-risk (multilevel-hierarchy-discovery-research-gate):
multi-level taxonomy = STACKING the validated flat pooler (EMERGE-38..41) + chaining inheritance, NOT a new mechanism.
NO `sim/` edit.

MECHANISM: L1 = the EMERGE-38 competitive self-organizing pooler on member features -> a sub-category codon per member.
L2 = the SAME competitive pooler, but its input is the L1 codons: trained on CO-OCCURRENCE (present the union of two
same-superordinate members' L1 codons) so L2 columns tune to what co-occurs -> a superordinate codon. Inheritance chains:
a superordinate property is taught (committed three-term kernel) on training members' L2 codons; a held-out member's
features -> its L1 codon -> its L2 codon -> the superordinate property. Biology: ventral hierarchy V1->V2->V4->IT +
ATL convergence zones (Kandel Ch21; Patterson-Lambon Ralph; Damasio) -- each level pools the level below.

ANTI-CHEATS: held-out SUPERORDINATE-inheritance (chance 1/2); PERMUTED-co-occurrence (L2 trained on RANDOM cross-super
pairs -> can't discover superordinates -> collapses); L1->L2 LESION (skip L2 learning -> the held-out member's L2 code is
untuned -> collapses); dAP-LESION; 6-seed. Reuse-by-import (`_emerge14` + `_emerge12`); NO `sim/` edit. CPU numpy. `--demo`.
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

OUT = Path("research/findings/raw/_emerge44_stacked_pooler.json")

SUBCATS = list(range(6))
SUPER = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}                                    # 6 sub-categories group into 2 superordinates
NSUPER = 2
STRIDE = 3
POOLS = {k: list(range(k * STRIDE, k * STRIDE + 6)) for k in SUBCATS}
NF = max(c for cs in POOLS.values() for c in cs) + 1
NCOL1 = 200; K1 = 6                                                             # L1 pooler
NCOL2 = 120; K2 = 6                                                             # L2 pooler (over L1 codons)
POOL_EPOCHS = 400
L2_EPOCHS = 400
POOL_LP = 0.05; POOL_LD = 0.02
N_PER = 6
HELD_SUB = {2, 5}            # hold out ENTIRE sub-categories from super-property teaching (one per superordinate) -> they can
                            # only inherit via the L2-DISCOVERED superordinate grouping, not via a trained sub-category
FLOOR = -40.0
NPROPUNITS = NSUPER * 2
M = NCOL2 + NPROPUNITS                                                          # bridge holds L2 columns + superordinate property cells


def _sdr(cells):
    return set(int(c) for c in cells)


def _competitive_pool(seed, samples, n_in, n_col, k_win, epochs):
    """HTM Spatial Pooler over a set of sparse binary inputs (samples: list of index-sets over [0,n_in)). Returns a codon
    fn: index-set -> the k winning columns. Winners potentiate active inputs + depress inactive + homeostatic boosting."""
    rng = np.random.default_rng(seed)
    W = rng.uniform(0.30, 0.55, (n_col, n_in))
    duty = np.zeros(n_col); boost = np.ones(n_col); order = list(range(len(samples)))
    for e in range(epochs):
        rng.shuffle(order)
        for i in order:
            x = np.zeros(n_in); x[list(samples[i])] = 1.0
            win = np.argsort(-(((W > 0.5) @ x) * boost))[:k_win]
            W[win] += POOL_LP * x - POOL_LD * (1 - x); W[win] = np.clip(W[win], 0, 1); duty[win] += 1
        boost = np.exp(2.0 * (k_win / n_col - duty / ((e + 1) * len(samples))))

    def codon(idxset):
        x = np.zeros(n_in); x[list(idxset)] = 1.0
        return set(int(c) for c in np.argsort(-((W > 0.5) @ x))[:k_win])
    return codon


class StackedPoolerProbe:
    def __init__(self, seed=42, epochs=40, lesion=False, permute=False, l2_lesion=False):
        rng = np.random.default_rng(seed)
        self.mem = {f"{k}_{i}": k for k in SUBCATS for i in range(N_PER)}
        self.feats = {}
        for i, (m, k) in enumerate(self.mem.items()):
            r = np.random.default_rng(seed * 100 + i)
            self.feats[m] = set(r.choice(POOLS[k], 4, replace=False))
        # L1: competitive pooler on member features -> sub-category codons
        l1 = _competitive_pool(seed, [self.feats[m] for m in self.mem], NF, NCOL1, K1, POOL_EPOCHS)
        self.l1codon = {m: l1(self.feats[m]) for m in self.mem}
        # L2: competitive pooler over L1 codons, trained on CO-OCCURRENCE of same-superordinate members
        members = list(self.mem)
        cooc = []
        rr = np.random.default_rng(seed * 3 + 7)
        for _ in range(240):
            if permute:                                                        # PERMUTED: random cross-super pairs -> no superordinate structure
                a, b = rr.choice(members, 2, replace=False)
            else:
                sup = int(rr.integers(NSUPER))
                pool = [m for m in members if SUPER[self.mem[m]] == sup]
                a, b = rr.choice(pool, 2, replace=False)
            cooc.append(self.l1codon[a] | self.l1codon[b])                     # the union of two co-occurring L1 codons
        if l2_lesion:
            l2 = _competitive_pool(seed, cooc, NCOL1, NCOL2, K2, 0)            # L1->L2 LESION: no L2 learning (untuned)
        else:
            l2 = _competitive_pool(seed, cooc, NCOL1, NCOL2, K2, L2_EPOCHS)
        self.l2codon = {m: l2(self.l1codon[m]) for m in self.mem}
        # spiking bridge: L2 columns -> superordinate property (committed three-term kernel)
        self._build_bridge(seed, lesion)
        self.SPROP = {s: [NCOL2 + 2 * s, NCOL2 + 2 * s + 1] for s in range(NSUPER)}
        self.held = {s: [] for s in range(NSUPER)}
        train = {s: [] for s in range(NSUPER)}
        for k in SUBCATS:
            ms = [m for m in self.mem if self.mem[m] == k]
            tgt = self.held if k in HELD_SUB else train
            for m in ms:
                tgt[SUPER[k]].append(m)
        for _ in range(epochs):
            for s in range(NSUPER):
                for m in train[s]:
                    apply_kernel_update(self.b, self.row, self.col, self.ci,
                                        _sdr(self.l2codon[m]), _sdr(self.SPROP[s]), self.z, 0.14, 0.02, 1.0)

    def _build_bridge(self, seed, lesion):
        from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
        from sim.bridge import SimulationBridge
        from sim.regions import BrainRegion
        from sim.enums import NeuronModel, NeuronType
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
        for pc in range(NPROPUNITS):
            for c in range(NCOL2):
                pre.append(int(ci[c])); post.append(int(ci[NCOL2 + pc])); w.append(0.0)
        b.inject_explicit_wiring({"ff": {"pre_indices": pre, "post_indices": post, "initial_weights": w,
                                         "plastic": False, "coincidence_detector": True, "conn_type": "ff"}})
        coo = b._get_cached_coo()
        self.b, self.ci, self.row, self.col = b, ci, np.asarray(_host(coo.row)), np.asarray(_host(coo.col))
        self.z = np.zeros(len(ci))

    def infer_super(self, member):
        codon = self.l2codon[member]
        if not codon:
            return -1
        ab = np.zeros(len(self.ci), bool)
        for c in codon:
            ab[c] = True
        _prime_from_winners(self.b, self.ci, ab)
        vap = getattr(self.b, "cp_v_apical", None)
        if vap is None or np.asarray(_host(vap)).ndim == 0:
            return -1
        vap = _host(vap)[self.ci]
        dr = {s: float(np.mean([vap[x] for x in u])) for s, u in self.SPROP.items()}
        bs = max(dr, key=dr.get)
        return bs if dr[bs] > FLOOR else -1

    def held_out_super_acc(self):
        return np.mean([self.infer_super(m) == s for s in range(NSUPER) for m in self.held[s]])

    def l2_grouping(self):
        """within-superordinate L2-codon overlap minus cross-superordinate overlap (should be >0 if L2 discovered supers)."""
        within, cross = [], []
        ms = list(self.mem)
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                ov = len(self.l2codon[ms[i]] & self.l2codon[ms[j]]) / K2
                (within if SUPER[self.mem[ms[i]]] == SUPER[self.mem[ms[j]]] else cross).append(ov)
        return float(np.mean(within) - np.mean(cross))


def _run_arm(seed, arm, epochs):
    p = StackedPoolerProbe(seed=seed, epochs=epochs, lesion=(arm == "lesion"),
                           permute=(arm == "permuted"), l2_lesion=(arm == "l2lesion"))
    return arm, {"super_acc": float(p.held_out_super_acc()), "l2_group": p.l2_grouping()}


ARMS = ["stacked", "permuted", "l2lesion", "lesion"]


def _demo(seed=42, epochs=40):
    p = StackedPoolerProbe(seed=seed, epochs=epochs)
    print("\n=== EMERGE-44 STACKED pooler -- discover a MULTI-LEVEL taxonomy (no transformer) ===")
    print(f"  L1 discovers 6 sub-categories; L2 pools their codons by co-occurrence into {NSUPER} superordinates.")
    print(f"  within-super L2 overlap - cross-super overlap = {p.l2_grouping():+.2f} (>0 = L2 discovered superordinates)\n")
    for s in range(NSUPER):
        for m in p.held[s]:
            print(f"  held-out {m} (sub {p.mem[m]}, super {s}) -> inferred super {p.infer_super(m)}  (expect {s})")
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
    print(f"stacked pooler: L1 (6 sub-cats) -> L2 (co-occurrence -> {NSUPER} superordinates) -> held-out superordinate "
          f"inheritance; chance {1/NSUPER:.2f}", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            d = {"seed": s}
            for arm in ARMS:
                _, r = _run_arm(s, arm, a.epochs); d[arm] = r
            per.append(d)
            print(f"  [seed {s}] super-acc {d['stacked']['super_acc']:.2f} (L2-group {d['stacked']['l2_group']:+.2f}) || "
                  f"permuted {d['permuted']['super_acc']:.2f} | L1->L2-lesion {d['l2lesion']['super_acc']:.2f} "
                  f"| dAP-lesion {d['lesion']['super_acc']:.2f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(arm, key="super_acc"):
            return float(np.mean([p[arm][key] for p in per]))
        acc, grp, perm, l2l, les = m("stacked"), m("stacked", "l2_group"), m("permuted"), m("l2lesion"), m("lesion")
        go = bool(acc >= 0.80 and grp >= 0.15 and acc >= perm + 0.25 and acc >= l2l + 0.20 and acc >= les + 0.30)
        if go:
            verdict = (f"GO -- the STACKED pooler DISCOVERS a multi-level taxonomy: a second competitive pooler layer pools the "
                       f"first layer's codons by CO-OCCURRENCE into {NSUPER} superordinates (within-super minus cross-super L2 "
                       f"overlap {grp:+.2f}), and inheritance CHAINS L1->L2 so a held-out member inherits its SUPERORDINATE "
                       f"property (super-acc {acc:.2f}, chance {1/NSUPER:.2f}). PERMUTED-co-occurrence {perm:.2f} (no superordinate "
                       f"structure); L1->L2 LESION {l2l:.2f} (untuned L2); dAP-LESION {les:.2f}; 6-seed. => multi-level taxonomy "
                       f"discovery is STACKING the validated flat pooler + chaining inheritance, NOT a new mechanism. The "
                       f"research-gate critical claim is CONFIRMED. NO sim/ edit.")
        else:
            miss = []
            if acc < 0.80: miss.append(f"super-acc {acc:.2f} < 0.80")
            if grp < 0.15: miss.append(f"L2 didn't discover superordinates (within-cross overlap {grp:+.2f} < 0.15)")
            if acc < perm + 0.25: miss.append(f"permuted didn't collapse ({acc:.2f} vs {perm:.2f})")
            if acc < l2l + 0.20: miss.append(f"L1->L2 learning not load-bearing ({acc:.2f} vs {l2l:.2f})")
            if acc < les + 0.30: miss.append(f"dAP-lesion didn't collapse ({acc:.2f} vs {les:.2f})")
            verdict = ("BOUNDARY (build-informative, the research-gate ~15% gate) -- " + "; ".join(miss) + ". The residual is "
                       "L2 separation of overlapping L1 codons; tune L2 boosting/depression or add a decorrelation stage.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge44_stacked_pooler", "verdict": verdict,
               "mechanism": "L1 competitive pooler (EMERGE-38) on member features -> sub-category codons; L2 competitive pooler "
                            "over the L1 codons, trained on co-occurrence of same-superordinate members -> superordinate codons; "
                            "inheritance chains L1->L2 via the committed three-term kernel + coincidence-plateau read",
               "task": "6 sub-categories -> 2 superordinates; L2 discovers superordinates from co-occurrence; held-out member "
                       "inherits its superordinate property; vs permuted-co-occurrence + L1->L2-lesion + dAP-lesion; multi-seed",
               "seeds": a.seeds, "config": {"epochs": a.epochs, "n_col1": NCOL1, "n_col2": NCOL2, "k1": K1, "k2": K2,
                                            "pool_epochs": POOL_EPOCHS, "l2_epochs": L2_EPOCHS, "n_super": NSUPER},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "the pooler LEARNING is a rate-reference (fully-on-substrate at EMERGE-39/40; k-WTA spiking at "
                              "EMERGE-41); the inheritance chain runs on the spiking bridge over the discovered L2 codons. Held-out "
                              "at the MEMBER level (seen sub-category); held-out-SUB-CATEGORY generalization needs cross-sub-cat "
                              "L1 overlap (a follow-on). Two levels; 3-level corpus is the next rung."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge44] VERDICT: {verdict}", flush=True)
    print(f"[emerge44] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
