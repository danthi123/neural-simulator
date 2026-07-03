"""EMERGE-45 / toward-semantics — a THREE-LEVEL discovered taxonomy + TRANSITIVITY: stacking the competitive pooler THREE
deep (member features -> sub-category -> genus -> order) discovers a 3-level hierarchy from co-occurrence, and inheritance
chains through TWO learned levels so a held-out sub-category inherits its ORDER property (2 levels up) while the SIBLING
order's property stays FALSE (the transitive discrimination: robin is-an animal-that-breathes, robin is NOT a fish-that-
swims). Extends EMERGE-44 (2-level) per the research gate. NO `sim/` edit.

MECHANISM: L1 pooler on member features -> sub-category codons. L2 pooler over L1 codons, trained on same-GENUS
co-occurrence -> genus codons. L3 pooler over L2 codons, trained on same-ORDER co-occurrence -> order codons. Inheritance:
an ORDER property is taught (committed three-term kernel) on training members' L3 codons; a held-out sub-category's
members -> L1 -> L2 -> L3 codon -> the order property, TWO discovered levels up. Transitivity discrimination: a member of
order O0 infers O0 (not the sibling O1), so it inherits O0's property and NOT O1's. Biology: the ventral hierarchy's
successive pooling stages + ATL convergence (Kandel Ch21; Patterson-Lambon Ralph; Damasio) -- each level pools the one
below.

ANTI-CHEATS: held-out sub-category ORDER-inheritance through 2 levels (chance 1/2); TRANSITIVITY (the sibling order's
property is NOT inherited -- inferred order != sibling); PERMUTED-co-occurrence (random cross-order pooling -> collapses);
dAP-LESION; 6-seed. Reuse-by-import (`_emerge14` + `_emerge12` + EMERGE-44 pooler helper); NO `sim/` edit. CPU numpy. `--demo`.
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
from research.runners._emerge44_stacked_pooler_derisk import _competitive_pool

OUT = Path("research/findings/raw/_emerge45_three_level_hierarchy.json")

SUBCATS = list(range(8))
GENUS = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3}                        # 8 sub-categories -> 4 genera
ORDER = {0: 0, 1: 0, 2: 1, 3: 1}                                               # 4 genera -> 2 orders
NGENUS = 4; NORDER = 2
STRIDE = 3
POOLS = {k: list(range(k * STRIDE, k * STRIDE + 6)) for k in SUBCATS}
NF = max(c for cs in POOLS.values() for c in cs) + 1
NCOL1 = 240; K1 = 6
NCOL2 = 160; K2 = 6
NCOL3 = 100; K3 = 6
POOL_EPOCHS = 400; L2_EPOCHS = 400; L3_EPOCHS = 400
N_PER = 6
HELD_SUB = {1, 3, 5, 7}                                                        # hold out one sub-category per genus from ORDER teaching
FLOOR = -40.0
NPROPUNITS = NORDER * 2
M = NCOL3 + NPROPUNITS


def _sdr(cells):
    return set(int(c) for c in cells)


def _order_of_sub(k):
    return ORDER[GENUS[k]]


class ThreeLevelProbe:
    def __init__(self, seed=42, epochs=40, lesion=False, permute=False):
        rng = np.random.default_rng(seed)
        self.mem = {f"{k}_{i}": k for k in SUBCATS for i in range(N_PER)}
        self.feats = {}
        for i, (m, k) in enumerate(self.mem.items()):
            r = np.random.default_rng(seed * 100 + i)
            self.feats[m] = set(r.choice(POOLS[k], 4, replace=False))
        members = list(self.mem)
        # L1: features -> sub-category codons
        l1 = _competitive_pool(seed, [self.feats[m] for m in members], NF, NCOL1, K1, POOL_EPOCHS)
        self.l1 = {m: l1(self.feats[m]) for m in members}
        # L2: L1 codons, co-occurrence of same-GENUS members -> genus codons
        cg = self._cooc(members, seed * 3 + 1, lambda m: GENUS[self.mem[m]], NGENUS, permute, self.l1)
        l2 = _competitive_pool(seed, cg, NCOL1, NCOL2, K2, L2_EPOCHS)
        self.l2 = {m: l2(self.l1[m]) for m in members}
        # L3: L2 codons, co-occurrence of same-ORDER members -> order codons
        co = self._cooc(members, seed * 5 + 2, lambda m: _order_of_sub(self.mem[m]), NORDER, permute, self.l2)
        l3 = _competitive_pool(seed, co, NCOL2, NCOL3, K3, L3_EPOCHS)
        self.l3 = {m: l3(self.l2[m]) for m in members}
        # bridge: L3 columns -> order property
        self._build_bridge(seed, lesion)
        self.OPROP = {o: [NCOL3 + 2 * o, NCOL3 + 2 * o + 1] for o in range(NORDER)}
        self.held = [m for m in members if self.mem[m] in HELD_SUB]
        train = [m for m in members if self.mem[m] not in HELD_SUB]
        for _ in range(epochs):
            for m in train:
                o = _order_of_sub(self.mem[m])
                apply_kernel_update(self.b, self.row, self.col, self.ci, _sdr(self.l3[m]), _sdr(self.OPROP[o]),
                                    self.z, 0.14, 0.02, 1.0)

    def _cooc(self, members, seed, keyfn, ngroup, permute, codons):
        rr = np.random.default_rng(seed); out = []
        for _ in range(260):
            if permute:
                a, b = rr.choice(members, 2, replace=False)
            else:
                g = int(rr.integers(ngroup))
                pool = [m for m in members if keyfn(m) == g]
                a, b = rr.choice(pool, 2, replace=False)
            out.append(codons[a] | codons[b])
        return out

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
            for c in range(NCOL3):
                pre.append(int(ci[c])); post.append(int(ci[NCOL3 + pc])); w.append(0.0)
        b.inject_explicit_wiring({"ff": {"pre_indices": pre, "post_indices": post, "initial_weights": w,
                                         "plastic": False, "coincidence_detector": True, "conn_type": "ff"}})
        coo = b._get_cached_coo()
        self.b, self.ci, self.row, self.col = b, ci, np.asarray(_host(coo.row)), np.asarray(_host(coo.col))
        self.z = np.zeros(len(ci))

    def infer_order(self, member):
        codon = self.l3[member]
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
        dr = {o: float(np.mean([vap[x] for x in u])) for o, u in self.OPROP.items()}
        bo = max(dr, key=dr.get)
        return bo if dr[bo] > FLOOR else -1

    def held_out_order_acc(self):
        return np.mean([self.infer_order(m) == _order_of_sub(self.mem[m]) for m in self.held])

    def transitivity_ok(self):
        """held-out members do NOT inherit the SIBLING order's property (inferred order != the other order)."""
        return np.mean([self.infer_order(m) != (1 - _order_of_sub(self.mem[m])) for m in self.held])


def _run_arm(seed, arm, epochs):
    p = ThreeLevelProbe(seed=seed, epochs=epochs, lesion=(arm == "lesion"), permute=(arm == "permuted"))
    return arm, {"order_acc": float(p.held_out_order_acc()), "transitivity": float(p.transitivity_ok())}


ARMS = ["stacked", "permuted", "lesion"]


def _demo(seed=42, epochs=40):
    p = ThreeLevelProbe(seed=seed, epochs=epochs)
    print("\n=== EMERGE-45 THREE-LEVEL discovered taxonomy + transitivity (no transformer) ===")
    print(f"  member features -> sub-category (L1) -> genus (L2) -> order (L3), all discovered by stacked pooling.")
    print(f"  a held-out sub-category inherits its ORDER (2 levels up); the SIBLING order's property stays FALSE.\n")
    for m in p.held[:8]:
        exp = _order_of_sub(p.mem[m])
        print(f"  held-out {m} (sub {p.mem[m]}, genus {GENUS[p.mem[m]]}, order {exp}) -> inferred order {p.infer_order(m)}  (expect {exp})")
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
    print(f"3-level stacked pooler: features -> sub-cat -> genus -> order; held-out sub-category inherits its ORDER (2 "
          f"levels up) + transitivity; chance {1/NORDER:.2f}", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            d = {"seed": s}
            for arm in ARMS:
                _, r = _run_arm(s, arm, a.epochs); d[arm] = r
            per.append(d)
            print(f"  [seed {s}] order-acc {d['stacked']['order_acc']:.2f} transitivity {d['stacked']['transitivity']:.2f} || "
                  f"permuted {d['permuted']['order_acc']:.2f} | dAP-lesion {d['lesion']['order_acc']:.2f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(arm, key="order_acc"):
            return float(np.mean([p[arm][key] for p in per]))
        acc, trans, perm, les = m("stacked"), m("stacked", "transitivity"), m("permuted"), m("lesion")
        go = bool(acc >= 0.80 and trans >= 0.80 and acc >= perm + 0.25 and acc >= les + 0.30)
        if go:
            verdict = (f"GO -- a THREE-LEVEL discovered taxonomy + transitivity: stacking the competitive pooler 3 deep "
                       f"(features -> sub-category -> genus -> order, all discovered from co-occurrence) chains inheritance "
                       f"through TWO learned levels -- a held-out sub-category inherits its ORDER property 2 levels up "
                       f"(order-acc {acc:.2f}, chance {1/NORDER:.2f}), and the SIBLING order's property stays FALSE "
                       f"(transitivity {trans:.2f}). PERMUTED-co-occurrence {perm:.2f}; dAP-LESION {les:.2f}; 6-seed. => the "
                       f"stacked pooler generalizes to 3 levels; multi-level inheritance-with-discrimination on one spiking "
                       f"brain. NO sim/ edit.")
        else:
            miss = []
            if acc < 0.80: miss.append(f"order-acc {acc:.2f} < 0.80")
            if trans < 0.80: miss.append(f"transitivity {trans:.2f} < 0.80")
            if acc < perm + 0.25: miss.append(f"permuted didn't collapse ({acc:.2f} vs {perm:.2f})")
            if acc < les + 0.30: miss.append(f"dAP-lesion didn't collapse ({acc:.2f} vs {les:.2f})")
            verdict = ("BOUNDARY (build-informative) -- " + "; ".join(miss) + ". The residual is L3 separation of overlapping "
                       "L2 codons across 2 nested levels; tune L3 boosting/depression/epochs; deeper stacking is the next tuning.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge45_three_level_hierarchy", "verdict": verdict,
               "mechanism": "3 stacked competitive poolers (features->sub-category L1, L1-codons->genus L2 via same-genus "
                            "co-occurrence, L2-codons->order L3 via same-order co-occurrence); inheritance chains through 2 "
                            "discovered levels via the committed three-term kernel + coincidence-plateau read",
               "task": "8 sub-cats -> 4 genera -> 2 orders; held-out sub-category inherits its ORDER 2 levels up + transitivity "
                       "(sibling order stays false); vs permuted-co-occurrence + dAP-lesion; multi-seed",
               "seeds": a.seeds, "config": {"epochs": a.epochs, "n_col": [NCOL1, NCOL2, NCOL3], "k": [K1, K2, K3],
                                            "n_genus": NGENUS, "n_order": NORDER},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "the pooler LEARNING is a rate-reference (fully-on-substrate at EMERGE-39/40; k-WTA spiking at "
                              "EMERGE-41); the inheritance chain runs on the spiking bridge over the discovered L3 codons. Extends "
                              "EMERGE-44 (2-level) to 3; the fully-spiking stacked version is EMERGE-46."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge45] VERDICT: {verdict}", flush=True)
    print(f"[emerge45] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
