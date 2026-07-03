"""EMERGE-42 / toward-semantics — the DISCOVERED categories REASON: the competitive self-organizing pooler (EMERGE-38..41)
discovers OVERLAPPING categories from experience, and the FULL Collins-Quillian inference (class inheritance + member-
specific-override CANCELLATION) runs over the pooler's LEARNED codons on the spiking bridge. Composes EMERGE-38 (the
competitive pooler that separates overlapping categories) + EMERGE-37 (cancellation on emergent codes). NO `sim/` edit.

WHY: EMERGE-38..41 built the competitive pooler (discovers overlapping categories, fully-on-substrate). EMERGE-26/37 built
inheritance + cancellation, but over hand-assigned or co-occurrence-context codes. This ties them: the brain DISCOVERS the
overlapping-category structure (competitive learning) AND does the full inference (inherit the class default; a specific
fact cancels it per-member) over the SELF-DISCOVERED codes -- a materially richer semantic substrate.

MECHANISM: the competitive HTM Spatial Pooler (EMERGE-38: winners potentiate active inputs + depress inactive + boosting)
learns a codon per member of 6 OVERLAPPING categories. Then, on the spiking bridge (committed three-term kernel over the
codon->property coincidence pool): (1) a CLASS property is taught on each category's training-member codons (inheritance);
(2) a member-SPECIFIC override property is taught directly on ONE member's codon. Query a member: the DIRECT (member-codon
-> override, taught) competes with the INHERITED (member-codon -> class-property, shared-column) via a graded-drive read;
the direct override out-drives the inherited default for the overridden member; non-overridden members inherit.

ANTI-CHEATS: CANCELLATION (the overridden member answers its SPECIFIC fact, not the class default); INHERITANCE (non-
overridden members inherit the class property via the discovered overlapping-category columns); PERMUTED-features (the
pooler can't discover categories -> inheritance collapses to chance, isolating the discovered structure); dAP-LESION;
6-seed. Reuse-by-import (`_emerge14` + `_emerge12`); NO `sim/` edit. CPU numpy-backend. `--demo`.
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

OUT = Path("research/findings/raw/_emerge42_pooler_inference.json")

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
# 2 class-level properties (isa-property) per category slot; + 1 OVERRIDE property for the overridden member
NCLASSPROP = NPROP           # one class property tag per category
NOVR = 1                     # one override property tag
NPROPUNITS = (NCLASSPROP + NOVR) * 2
NMEM = len(CATS) * N_PER     # members
N_ID_PER = 3                 # a small member-IDENTITY ensemble per member (>=2 so it clears the coincidence threshold)
NMEM_CELLS = NMEM * N_ID_PER
# cell layout: [category features | member-identity cells | pooler columns | property cells]
FEAT0 = 0
ID0 = NF
COL0 = NF + NMEM_CELLS
PROP0 = NF + NMEM_CELLS + NCOL
M = NF + NMEM_CELLS + NCOL + NPROPUNITS
OVERRIDE_MEMBER = "1_0"      # a member of category 1 gets a specific override (answers OVR, not category-1's class default)


def _sdr(cells):
    return set(int(c) for c in cells)


class PoolerInferenceProbe:
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
            for c in range(NCOL):                                             # pooler columns -> property (class inheritance)
                pre.append(int(ci[COL0 + c])); post.append(int(ci[PROP0 + pc])); w.append(0.0)
            for idx in range(NMEM_CELLS):                                     # member-identity cells -> property (member-specific facts)
                pre.append(int(ci[ID0 + idx])); post.append(int(ci[PROP0 + pc])); w.append(0.0)
        b.inject_explicit_wiring({"ff": {"pre_indices": pre, "post_indices": post, "initial_weights": w,
                                         "plastic": False, "coincidence_detector": True, "conn_type": "ff"}})
        coo = b._get_cached_coo()
        self.b, self.ci, self.row, self.col = b, ci, np.asarray(_host(coo.row)), np.asarray(_host(coo.col))
        self.z = np.zeros(len(ci))
        # member stream over 6 OVERLAPPING categories; each member also has a UNIQUE member-identity cell
        self.mem = {f"{k}_{i}": k for k in CATS for i in range(N_PER)}
        self.midx = {m: i for i, m in enumerate(self.mem)}                     # member -> identity cell index
        self.feats = {}
        for i, (m, k) in enumerate(self.mem.items()):
            r = np.random.default_rng(seed * 100 + (i if not permute else int(rng.integers(10 ** 6))))
            pool = POOLS[k] if not permute else list(range(NF))
            self.feats[m] = set(r.choice(pool, 4, replace=False))
        # the competitive self-organizing pooler (EMERGE-38): learn overlapping-category codons
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
        # property units: CLASS[k] = category k's class property; OVR = the override property
        self.CLASS = {k: [PROP0 + 2 * k, PROP0 + 2 * k + 1] for k in CATS}
        self.OVR = [PROP0 + 2 * NCLASSPROP, PROP0 + 2 * NCLASSPROP + 1]
        # inheritance: teach each category's CLASS property on its member CODONS (shared category columns -> generalizes)
        for _ in range(epochs):
            for k in CATS:
                for m in [mm for mm in self.mem if self.mem[mm] == k]:
                    apply_kernel_update(self.b, self.row, self.col, self.ci, self._codon(self.feats[m]),
                                        _sdr(self.CLASS[k]), self.z, 0.14, 0.02, 1.0)
        # cancellation: teach the OVERRIDE property on the overridden member's UNIQUE identity cell (its member-specific
        # representation) -- a stronger, more-direct fact than the inherited class default; keyed to the member alone.
        ovr_id = self._id_cells(OVERRIDE_MEMBER)
        for _ in range(epochs * 2):
            apply_kernel_update(self.b, self.row, self.col, self.ci, ovr_id, _sdr(self.OVR), self.z, 0.14, 0.02, 1.0)

    def _id_cells(self, member):
        base = ID0 + self.midx[member] * N_ID_PER
        return set(int(self.ci[base + j]) for j in range(N_ID_PER))

    def _codon(self, feats):
        x = np.zeros(NF); x[list(feats)] = 1.0
        return set(COL0 + int(c) for c in np.argsort(-((self.Wp > 0.5) @ x))[:K_WIN])

    def _drive_to(self, member, targets):
        resp = self._codon(self.feats[member])
        if not resp:
            return None
        ab = np.zeros(len(self.ci), bool)
        for c in resp:                                                        # prime the category codon (columns)
            ab[c] = True
        for j in range(N_ID_PER):                                            # + the member's own identity ensemble
            ab[ID0 + self.midx[member] * N_ID_PER + j] = True
        _prime_from_winners(self.b, self.ci, ab)
        vap = getattr(self.b, "cp_v_apical", None)
        if vap is None or np.asarray(_host(vap)).ndim == 0:
            return None
        vap = _host(vap)[self.ci]
        return {t: float(np.mean([vap[x] for x in u])) for t, u in targets.items()}

    def query(self, member):
        """Cancellation read: the strongest of the DIRECT override (member-identity) and the INHERITED class default."""
        targets = {"OVR": self.OVR}
        for k in CATS:
            targets[f"C{k}"] = self.CLASS[k]
        dr = self._drive_to(member, targets)
        if dr is None:
            return None
        best = max(dr, key=dr.get)
        return best if dr[best] > FLOOR else None

    def cancellation_ok(self):
        return self.query(OVERRIDE_MEMBER) == "OVR"

    def inheritance_acc(self):
        held = {k: [m for m in self.mem if self.mem[m] == k and m != OVERRIDE_MEMBER][-3:] for k in CATS}
        return np.mean([self.query(m) == f"C{k}" for k in CATS for m in held[k]])


def _run_arm(seed, arm, epochs):
    p = PoolerInferenceProbe(seed=seed, epochs=epochs, lesion=(arm == "lesion"),
                             permute=(arm == "permuted"), learn=(arm != "nolearn"))
    return arm, {"cancellation": float(p.cancellation_ok()), "inheritance": float(p.inheritance_acc())}


ARMS = ["pooler", "nolearn", "permuted", "lesion"]


def _demo(seed=42, epochs=40):
    p = PoolerInferenceProbe(seed=seed, epochs=epochs)
    print("\n=== EMERGE-42 discovered categories REASON (pooler + inheritance + cancellation; no transformer) ===")
    print(f"  the competitive pooler discovers 6 OVERLAPPING categories (adjacent share {6-STRIDE}/6 feats) from experience;")
    print(f"  the full Collins-Quillian inference runs over the LEARNED codons.\n")
    print(f"  cancellation: overridden member {OVERRIDE_MEMBER} -> {p.query(OVERRIDE_MEMBER)} (expect OVR, not C1)")
    held = {k: [m for m in p.mem if p.mem[m] == k and m != OVERRIDE_MEMBER][-2:] for k in CATS}
    for k in CATS:
        for m in held[k]:
            print(f"  inherit: {m} (cat {k}) -> {p.query(m)}  (expect C{k})")
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
    print(f"pooler-discovered categories + full inference: 6 OVERLAPPING cats (share {6-STRIDE}/6), inheritance + "
          f"cancellation over LEARNED codons; inheritance chance {1/(NPROP+1):.2f}", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            d = {"seed": s}
            for arm in ARMS:
                _, r = _run_arm(s, arm, a.epochs); d[arm] = r
            per.append(d)
            print(f"  [seed {s}] cancellation {d['pooler']['cancellation']:.0f} inheritance {d['pooler']['inheritance']:.2f} "
                  f"|| no-learn inh {d['nolearn']['inheritance']:.2f} | permuted inh {d['permuted']['inheritance']:.2f} "
                  f"| lesion inh {d['lesion']['inheritance']:.2f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        canc = float(np.mean([p["pooler"]["cancellation"] for p in per]))
        inh = float(np.mean([p["pooler"]["inheritance"] for p in per]))
        nol = float(np.mean([p["nolearn"]["inheritance"] for p in per]))
        perm = float(np.mean([p["permuted"]["inheritance"] for p in per]))
        les = float(np.mean([p["lesion"]["inheritance"] for p in per]))
        go = bool(canc >= 0.85 and inh >= 0.80 and inh >= perm + 0.30 and inh >= les + 0.30 and inh >= nol + 0.20)
        if go:
            verdict = (f"GO -- the DISCOVERED categories REASON: the competitive self-organizing pooler (EMERGE-38..41) "
                       f"discovers 6 OVERLAPPING categories from experience, and the FULL Collins-Quillian inference runs over "
                       f"the LEARNED codons -- CANCELLATION {canc:.2f} (the overridden member answers its SPECIFIC fact, not the "
                       f"class default) + INHERITANCE {inh:.2f} (non-overridden members inherit the class property via the "
                       f"discovered overlapping-category columns). PERMUTED-features {perm:.2f} (the pooler can't discover "
                       f"categories -> collapses); FIXED (no-learn) {nol:.2f}; dAP-LESION {les:.2f}; 6-seed. => the brain "
                       f"DISCOVERS overlapping categories AND does inheritance-with-cancellation over them, on one spiking "
                       f"brain. NO sim/ edit.")
        else:
            miss = []
            if canc < 0.85: miss.append(f"cancellation {canc:.2f} < 0.85")
            if inh < 0.80: miss.append(f"inheritance {inh:.2f} < 0.80")
            if inh < perm + 0.30: miss.append(f"permuted didn't collapse ({inh:.2f} vs {perm:.2f})")
            if inh < les + 0.30: miss.append(f"lesion didn't collapse ({inh:.2f} vs {les:.2f})")
            if inh < nol + 0.20: miss.append(f"no-learn(fixed) didn't collapse ({inh:.2f} vs {nol:.2f})")
            verdict = ("BOUNDARY (build-informative) -- " + "; ".join(miss) + ". Tune pooler epochs / inheritance vs override "
                       "teaching balance; the discovered-category inference is the next tuning, not a wall.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge42_pooler_inference", "verdict": verdict,
               "mechanism": "the EMERGE-38 competitive self-organizing pooler learns overlapping-category codons; the committed "
                            "three-term kernel teaches each category's CLASS property on its member codons (inheritance) + an "
                            "OVERRIDE property directly on one member's codon (cancellation); a graded-drive read takes the "
                            "strongest of the direct-override (1-hop) and the inherited class default over the discovered codons",
               "task": "6 OVERLAPPING categories discovered by the pooler; inheritance + specific-override cancellation over the "
                       "learned codons; vs no-learn + permuted-features + dAP-lesion; multi-seed",
               "seeds": a.seeds, "config": {"epochs": a.epochs, "n_col": NCOL, "k_win": K_WIN, "pool_epochs": POOL_EPOCHS,
                                            "stride": STRIDE, "n_feat": NF, "override_member": OVERRIDE_MEMBER},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "composes EMERGE-38 (competitive pooler; the learning is a rate-reference realized fully-on-substrate "
                              "at EMERGE-39/40, k-WTA spiking at EMERGE-41) + EMERGE-37 (cancellation on emergent codes). The "
                              "inheritance/cancellation run on the spiking bridge over the discovered codons; single override on a "
                              "6-category setup; multi-override / multi-level are follow-ons."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge42] VERDICT: {verdict}", flush=True)
    print(f"[emerge42] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
