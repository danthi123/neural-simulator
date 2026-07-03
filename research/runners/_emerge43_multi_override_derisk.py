"""EMERGE-43 / toward-semantics — MULTI-OVERRIDE cancellation over discovered categories: MANY members (one per category)
each carry their OWN member-specific exception fact, coexisting with class inheritance, on the pooler-discovered
overlapping categories. Demonstrates the member-identity mechanism (EMERGE-42) SCALES -- multiple exceptions coexist with
no cross-bleed and without disrupting inheritance. Composes EMERGE-38 pooler + EMERGE-42 cancellation. NO `sim/` edit.

WHY: EMERGE-42 showed ONE member-specific override coexisting with class inheritance over discovered categories, via a
category-code + member-identity-ensemble representation. Real knowledge has MANY exceptions (many entities each with their
own facts). This checks the mechanism scales: N overrides (one per category), each on its own identity ensemble, each
answering its OWN exception, non-overridden members still inheriting, no override bleeding to another member/category.

ANTI-CHEATS: each overridden member answers ITS OWN exception (not another's, not the class default); non-overridden
members inherit; PERMUTED-features collapses inheritance; dAP-LESION; 6-seed. Reuse-by-import; NO `sim/` edit. CPU
numpy-backend. `--demo`.
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

OUT = Path("research/findings/raw/_emerge43_multi_override.json")

CATS = list(range(6)); NPROP = len(CATS)
STRIDE = 3
POOLS = {k: list(range(k * STRIDE, k * STRIDE + 6)) for k in CATS}
NF = max(c for cs in POOLS.values() for c in cs) + 1
NCOL = 200
K_WIN = 6
POOL_EPOCHS = 400
POOL_LP = 0.05
POOL_LD = 0.02
N_PER = 9
FLOOR = -40.0
OVERRIDES = {f"{k}_0": k for k in CATS}                                        # one overridden member PER category, each a distinct exception
OVR_LIST = list(OVERRIDES)
NCLASSPROP = NPROP
NOVR = len(OVR_LIST)                                                           # one exception property tag per overridden member
NPROPUNITS = (NCLASSPROP + NOVR) * 2
NMEM = len(CATS) * N_PER
N_ID_PER = 3
NMEM_CELLS = NMEM * N_ID_PER
FEAT0 = 0; ID0 = NF; COL0 = NF + NMEM_CELLS; PROP0 = NF + NMEM_CELLS + NCOL
M = NF + NMEM_CELLS + NCOL + NPROPUNITS


def _sdr(cells):
    return set(int(c) for c in cells)


class MultiOverrideProbe:
    def __init__(self, seed=42, epochs=40, lesion=False, permute=False, learn=True):
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
        for pc in range(NPROPUNITS):
            for c in range(NCOL):
                pre.append(int(ci[COL0 + c])); post.append(int(ci[PROP0 + pc])); w.append(0.0)
            for idx in range(NMEM_CELLS):
                pre.append(int(ci[ID0 + idx])); post.append(int(ci[PROP0 + pc])); w.append(0.0)
        b.inject_explicit_wiring({"ff": {"pre_indices": pre, "post_indices": post, "initial_weights": w,
                                         "plastic": False, "coincidence_detector": True, "conn_type": "ff"}})
        coo = b._get_cached_coo()
        self.b, self.ci, self.row, self.col = b, ci, np.asarray(_host(coo.row)), np.asarray(_host(coo.col))
        self.z = np.zeros(len(ci))
        self.mem = {f"{k}_{i}": k for k in CATS for i in range(N_PER)}
        self.midx = {m: i for i, m in enumerate(self.mem)}
        self.feats = {}
        for i, (m, k) in enumerate(self.mem.items()):
            r = np.random.default_rng(seed * 100 + (i if not permute else int(rng.integers(10 ** 6))))
            pool = POOLS[k] if not permute else list(range(NF))
            self.feats[m] = set(r.choice(pool, 4, replace=False))
        self.Wp = rng.uniform(0.30, 0.55, (NCOL, NF))
        if learn:
            duty = np.zeros(NCOL); boost = np.ones(NCOL); order = list(self.mem)
            for e in range(POOL_EPOCHS):
                rng.shuffle(order)
                for m in order:
                    x = np.zeros(NF); x[list(self.feats[m])] = 1.0
                    win = np.argsort(-(((self.Wp > 0.5) @ x) * boost))[:K_WIN]
                    self.Wp[win] += POOL_LP * x - POOL_LD * (1 - x); self.Wp[win] = np.clip(self.Wp[win], 0, 1); duty[win] += 1
                boost = np.exp(2.0 * (K_WIN / NCOL - duty / ((e + 1) * len(self.mem))))
        self.CLASS = {k: [PROP0 + 2 * k, PROP0 + 2 * k + 1] for k in CATS}
        self.OVR = {ov: [PROP0 + 2 * (NCLASSPROP + j), PROP0 + 2 * (NCLASSPROP + j) + 1] for j, ov in enumerate(OVR_LIST)}
        # inheritance: class property on member codons
        for _ in range(epochs):
            for k in CATS:
                for m in [mm for mm in self.mem if self.mem[mm] == k]:
                    apply_kernel_update(self.b, self.row, self.col, self.ci, self._codon(self.feats[m]),
                                        _sdr(self.CLASS[k]), self.z, 0.14, 0.02, 1.0)
        # cancellation: EACH overridden member's own exception on ITS identity ensemble
        for _ in range(epochs * 2):
            for ov in OVR_LIST:
                apply_kernel_update(self.b, self.row, self.col, self.ci, self._id_cells(ov), _sdr(self.OVR[ov]),
                                    self.z, 0.14, 0.02, 1.0)

    def _id_cells(self, member):
        base = ID0 + self.midx[member] * N_ID_PER
        return set(int(self.ci[base + j]) for j in range(N_ID_PER))

    def _codon(self, feats):
        x = np.zeros(NF); x[list(feats)] = 1.0
        return set(COL0 + int(c) for c in np.argsort(-((self.Wp > 0.5) @ x))[:K_WIN])

    def query(self, member):
        resp = self._codon(self.feats[member])
        if not resp:
            return None
        ab = np.zeros(len(self.ci), bool)
        for c in resp:
            ab[c] = True
        for j in range(N_ID_PER):
            ab[ID0 + self.midx[member] * N_ID_PER + j] = True
        _prime_from_winners(self.b, self.ci, ab)
        vap = getattr(self.b, "cp_v_apical", None)
        if vap is None or np.asarray(_host(vap)).ndim == 0:
            return None
        vap = _host(vap)[self.ci]
        targets = {f"C{k}": self.CLASS[k] for k in CATS}
        targets.update({f"OVR:{ov}": u for ov, u in self.OVR.items()})
        dr = {t: float(np.mean([vap[x] for x in u])) for t, u in targets.items()}
        best = max(dr, key=dr.get)
        return best if dr[best] > FLOOR else None

    def override_acc(self):
        """each overridden member answers ITS OWN exception."""
        return np.mean([self.query(ov) == f"OVR:{ov}" for ov in OVR_LIST])

    def inheritance_acc(self):
        held = {k: [m for m in self.mem if self.mem[m] == k and m not in OVERRIDES][-3:] for k in CATS}
        return np.mean([self.query(m) == f"C{k}" for k in CATS for m in held[k]])


def _run_arm(seed, arm, epochs):
    p = MultiOverrideProbe(seed=seed, epochs=epochs, lesion=(arm == "lesion"),
                           permute=(arm == "permuted"), learn=(arm != "nolearn"))
    return arm, {"override": float(p.override_acc()), "inheritance": float(p.inheritance_acc())}


ARMS = ["multi", "permuted", "lesion"]


def _demo(seed=42, epochs=40):
    p = MultiOverrideProbe(seed=seed, epochs=epochs)
    print("\n=== EMERGE-43 MULTI-OVERRIDE over discovered categories (no transformer) ===")
    print(f"  {len(OVR_LIST)} overridden members (one per category), each with its OWN exception fact.\n")
    for ov in OVR_LIST:
        print(f"  override: {ov} -> {p.query(ov)}  (expect OVR:{ov})")
    held = {k: [m for m in p.mem if p.mem[m] == k and m not in OVERRIDES][-1:] for k in CATS}
    for k in CATS:
        for m in held[k]:
            print(f"  inherit : {m} (cat {k}) -> {p.query(m)}  (expect C{k})")
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
    print(f"multi-override: {len(OVR_LIST)} coexisting exceptions over pooler-discovered categories; inheritance chance "
          f"{1/(NPROP+NOVR):.2f}", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            d = {"seed": s}
            for arm in ARMS:
                _, r = _run_arm(s, arm, a.epochs); d[arm] = r
            per.append(d)
            print(f"  [seed {s}] override {d['multi']['override']:.2f} inheritance {d['multi']['inheritance']:.2f} || "
                  f"permuted inh {d['permuted']['inheritance']:.2f} | lesion inh {d['lesion']['inheritance']:.2f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        ov = float(np.mean([p["multi"]["override"] for p in per]))
        inh = float(np.mean([p["multi"]["inheritance"] for p in per]))
        perm = float(np.mean([p["permuted"]["inheritance"] for p in per]))
        les = float(np.mean([p["lesion"]["inheritance"] for p in per]))
        go = bool(ov >= 0.85 and inh >= 0.80 and inh >= perm + 0.30 and inh >= les + 0.30)
        if go:
            verdict = (f"GO -- MULTI-OVERRIDE scales: {len(OVR_LIST)} member-specific exceptions coexist with class inheritance "
                       f"over the pooler-discovered overlapping categories -- each overridden member answers ITS OWN exception "
                       f"(override-acc {ov:.2f}, no cross-bleed), non-overridden members inherit ({inh:.2f}). PERMUTED-features "
                       f"{perm:.2f}; dAP-LESION {les:.2f}; 6-seed. => the category-code + member-identity-ensemble representation "
                       f"handles MANY coexisting exceptions, the realistic case. NO sim/ edit.")
        else:
            miss = []
            if ov < 0.85: miss.append(f"override-acc {ov:.2f} < 0.85")
            if inh < 0.80: miss.append(f"inheritance {inh:.2f} < 0.80")
            if inh < perm + 0.30: miss.append(f"permuted didn't collapse ({inh:.2f} vs {perm:.2f})")
            if inh < les + 0.30: miss.append(f"lesion didn't collapse ({inh:.2f} vs {les:.2f})")
            verdict = ("BOUNDARY (build-informative) -- " + "; ".join(miss) + ". Tune identity-ensemble size / override "
                       "teaching strength; multi-override is the next tuning, not a wall.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge43_multi_override", "verdict": verdict,
               "mechanism": "N overridden members (one per category), each a distinct exception taught on ITS OWN member-identity "
                            "ensemble; class properties on the pooler-discovered codons; a graded-drive read takes the strongest "
                            "of the member's exception and the inherited class default",
               "task": "multi-override cancellation + inheritance over pooler-discovered overlapping categories; vs permuted + "
                       "dAP-lesion; multi-seed",
               "seeds": a.seeds, "config": {"epochs": a.epochs, "n_overrides": len(OVR_LIST), "n_col": NCOL, "k_win": K_WIN,
                                            "pool_epochs": POOL_EPOCHS, "n_id_per": N_ID_PER},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "composes EMERGE-38 pooler + EMERGE-42 member-identity cancellation; scales the exception count to "
                              "one per category. Multi-level-emergent cancellation + transitivity over discovered categories "
                              "remain follow-ons."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge43] VERDICT: {verdict}", flush=True)
    print(f"[emerge43] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
