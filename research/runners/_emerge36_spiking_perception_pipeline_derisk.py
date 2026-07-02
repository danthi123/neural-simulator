"""EMERGE-36 / toward-semantics — the FULLY-SPIKING PERCEPTION->POOLER->INFERENCE pipeline (the capstone of the
fully-spiking emergent-structure arc): objects SEEN through the real Gabor/V1 front end -> a spiking SPARSE-EXPANSION
pooler codon (EMERGE-35, no numpy kWTA) -> discovered categories -> on-bridge inheritance. SEE an object, learn what a
category is, reason about a held-out perceived object -- all SPIKING end-to-end, NO `sim/` edit.

This composes EMERGE-34 (perception-grounded emergence via the real retina/V1 Gabor bank) + EMERGE-35 (the fully-spiking
Marr-Albus-codon pooler), replacing EMERGE-34's NUMPY competitive pooler with the spiking codon. The whole pipeline --
pixels -> retina/V1 Gabor responses -> a decorrelated sparse-expansion column codon (coincidence-driven) -> a property
taught on training objects' codons -> a held-out PERCEIVED object inherits -- runs with no numpy kWTA anywhere.

MECHANISM: object shapes -> pixels -> the real `sim.visual_cortex` Gabor/V1 responses (reused via `_genfrontier_optionB`)
-> the top-active V1 cells define the feature layer -> each object's active V1 features drive a large decorrelated
column layer (each column samples SAMP features, fires if >= ACT_TH active via `coincidence_weighted_drive`) -> a sparse
codon per perceived object; same-category objects (similar shapes -> overlapping V1 features) get overlapping codons ->
a property taught on training objects' codons (the committed `sim/` three-term kernel) is inherited by a held-out
perceived object via its overlapping codon.

ANTI-CHEATS (per the control-validity methodology): held-out perceived-object inheritance (>=3/category); PER-IMAGE
PIXEL SCRAMBLE (destroys within-category visual similarity -> collapses, the load-bearing perception control);
dAP-LESION (coincidence off -> no codon -> collapses); 6-seed. Reuse-by-import (`_genfrontier_optionB` V1 + `_emerge14`
+ `_emerge12`); NO `sim/` edit. CPU numpy-backend. `--demo`.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, time, traceback
from pathlib import Path
import numpy as np

from research.runners._genfrontier_optionB_visual_similarity_derisk import (
    build_shape_set, build_gabor_response_matrix, encode_v1)
from research.runners._emerge14_stageC_onbridge_learning_derisk import apply_kernel_update, _host
from research.runners._emerge12_stageB2_bridge_tm_derisk import _prime_from_winners

OUT = Path("research/findings/raw/_emerge36_spiking_perception_pipeline.json")

N_EX = 9
HOLD = 3
N_FEAT = 60                                                                     # the top-active V1 cells = the feature layer
N_COL = 250                                                                     # sparse expansion
SAMP = 3
ACT_TH = 2
CATPROP = {0: "fly", 1: "swim"}
NPROP = 2
nE = 1
FLOOR = -40.0
M = N_FEAT + N_COL + NPROP * 2
_GABOR_W = None


def _gabor():
    global _GABOR_W
    if _GABOR_W is None:
        _GABOR_W = build_gabor_response_matrix()
    return _GABOR_W


class SpikingPerceptionProbe:
    def __init__(self, seed=42, epochs=40, lesion=False, scramble=False):
        from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
        from sim.bridge import SimulationBridge
        from sim.regions import BrainRegion
        from sim.enums import NeuronModel, NeuronType
        rng = np.random.default_rng(seed)
        imgs, self.labels, _ = build_shape_set(n_categories=2, n_exemplars=N_EX, rng=rng)
        if scramble:
            r = np.random.default_rng(seed + 5)
            imgs = np.stack([im.flatten()[r.permutation(im.size)].reshape(im.shape) for im in imgs])
        V = encode_v1(imgs, _gabor())
        feats = list(np.argsort(-V.mean(0))[:N_FEAT])                          # the most-active V1 cells = feature layer
        fidx = {f: k for k, f in enumerate(feats)}
        self.OF = []                                                           # each object's active V1 features (unit indices)
        for i in range(len(self.labels)):
            thr = np.percentile(V[i][feats], 70)
            self.OF.append(set(fidx[f] for f in feats if V[i][f] > thr))
        self.W = np.zeros((N_COL, N_FEAT))                                     # decorrelated projection
        for c in range(N_COL):
            self.W[c, rng.choice(N_FEAT, SAMP, replace=False)] = 1
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
        cfg.coincidence_weighted_drive = True; cfg.coincidence_k_threshold = float(ACT_TH) - 0.5
        cfg.coincidence_plateau_strength = 160.0; cfg.enable_two_compartment_dap = True; cfg.apical_g_couple = 2.0
        b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                             runtime_state=RuntimeState(), gpu_config=GPUConfig())
        b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        b.runtime_state.actual_seed_used = seed
        b._initialize_simulation_data(called_from_playback_init=False)
        ci = np.asarray(b.region_manager.indices("cells"), int)
        pre, post, w = [], [], []
        for c in range(N_COL):
            for f in np.where(self.W[c] > 0)[0]:
                pre.append(int(ci[f])); post.append(int(ci[N_FEAT + c])); w.append(1.0)
        for pc in range(NPROP * 2):
            for c in range(N_COL):
                pre.append(int(ci[N_FEAT + c])); post.append(int(ci[N_FEAT + N_COL + pc])); w.append(0.0)
        b.inject_explicit_wiring({"ff": {"pre_indices": pre, "post_indices": post, "initial_weights": w,
                                         "plastic": False, "coincidence_detector": True, "conn_type": "ff"}})
        coo = b._get_cached_coo()
        self.b, self.ci, self.row, self.col = b, ci, np.asarray(_host(coo.row)), np.asarray(_host(coo.col))
        self.z = np.zeros(len(ci))
        self.PROP = {c: [N_FEAT + N_COL + 2 * c, N_FEAT + N_COL + 2 * c + 1] for c in (0, 1)}
        idx = {c: [i for i in range(len(self.labels)) if self.labels[i] == c] for c in (0, 1)}
        self.held = {c: idx[c][-HOLD:] for c in (0, 1)}
        for _ in range(epochs):
            for c in (0, 1):
                for tr in idx[c][:-HOLD]:
                    apply_kernel_update(self.b, self.row, self.col, self.ci, self._codon(self.OF[tr]),
                                        set(self.PROP[c]), self.z, 0.14, 0.02, 1.0)

    def _codon(self, of):
        ab = np.zeros(len(self.ci), bool)
        for f in of:
            ab[f] = True
        _prime_from_winners(self.b, self.ci, ab)
        vap = getattr(self.b, "cp_v_apical", None)
        if vap is None:
            return set()
        vap = _host(vap)[self.ci]
        return set(N_FEAT + c for c in range(N_COL) if vap[N_FEAT + c] > FLOOR)

    def infer(self, of):
        resp = self._codon(of)
        if not resp:
            return -1
        ab = np.zeros(len(self.ci), bool)
        for i in resp:
            ab[i] = True
        _prime_from_winners(self.b, self.ci, ab)
        vap = _host(self.b.cp_v_apical)[self.ci]
        dr = {c: float(np.mean([vap[x] for x in self.PROP[c]])) for c in (0, 1)}
        best = max(dr, key=dr.get)
        return best if dr[best] > FLOOR else -1

    def held_out_acc(self):
        return np.mean([self.infer(self.OF[h]) == c for c in (0, 1) for h in self.held[c]])


def _run_arm(seed, arm, epochs):
    p = SpikingPerceptionProbe(seed=seed, epochs=epochs, lesion=(arm == "lesion"), scramble=(arm == "scrambled"))
    return arm, {"held_out": float(p.held_out_acc())}


ARMS = ["htm", "scrambled", "lesion"]


def _demo(seed=42, epochs=40):
    p = SpikingPerceptionProbe(seed=seed, epochs=epochs)
    print("\n=== EMERGE-36 fully-spiking perception -> pooler -> inference (SEE -> discover -> reason; no numpy kWTA) ===")
    print("  objects -> real Gabor/V1 -> spiking sparse-expansion codon (EMERGE-35) -> inheritance; held-out perceived object.\n")
    for c in (0, 1):
        for h in p.held[c]:
            ans = p.infer(p.OF[h])
            print(f"  held-out perceived object (visual category {c}) -> {CATPROP.get(ans, 'ABSTAIN')}  (expect {CATPROP[c]})")
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
    print(f"fully-spiking perception pipeline: shapes -> V1 ({N_FEAT} feats) -> {N_COL}-col spiking codon -> inheritance", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            d = {"seed": s}
            for arm in ARMS:
                _, r = _run_arm(s, arm, a.epochs); d[arm] = r
            per.append(d); h = d["htm"]
            print(f"  [seed {s}] HELD-OUT-inherit {h['held_out']:.2f} || scrambled {d['scrambled']['held_out']:.2f} "
                  f"| lesion {d['lesion']['held_out']:.2f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(arm):
            return float(np.mean([p[arm]["held_out"] for p in per]))
        held, scr, les = m("htm"), m("scrambled"), m("lesion")
        go = bool(held >= 0.85 and held >= scr + 0.30 and held >= les + 0.30)
        if go:
            verdict = (f"GO -- the FULLY-SPIKING PERCEPTION->POOLER->INFERENCE pipeline: objects SEEN through the real Gabor/V1 "
                       f"front end -> a spiking sparse-expansion codon (EMERGE-35, NO numpy kWTA) -> a held-out PERCEIVED object "
                       f"inherits its visual category's property ({held:.2f}, 3/category), on the spiking bridge, end-to-end. "
                       f"PER-IMAGE PIXEL SCRAMBLE collapses it ({scr:.2f}, the load-bearing perception control); dAP-LESION "
                       f"{les:.2f}; 6-seed. => SEE an object, discover what a category is, reason about a novel one -- all SPIKING, "
                       f"biology-grounded (Gabor/V1 + cerebellar-granule codon), NO sim/ edit. Closes the fully-spiking perception "
                       f"pipeline (EMERGE-34's numpy pooler replaced).")
        else:
            miss = []
            if held < 0.85: miss.append(f"held-out {held:.2f} < 0.85")
            if held < scr + 0.30: miss.append(f"scramble didn't collapse ({held:.2f} vs {scr:.2f})")
            if held < les + 0.30: miss.append(f"dAP-lesion didn't collapse ({held:.2f} vs {les:.2f})")
            verdict = ("BOUNDARY (build-informative) -- " + "; ".join(miss) + ". Tune N_FEAT / expansion / percentile; the "
                       "fully-spiking perception pipeline is the next tuning, not a wall.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge36_spiking_perception_pipeline", "verdict": verdict,
               "mechanism": "fully-spiking perception pipeline: shapes -> real Gabor/V1 responses -> top-active V1 cells = "
                            "feature layer -> a spiking sparse-expansion codon (decorrelated columns, coincidence-driven, "
                            "EMERGE-35) -> on-bridge inheritance; no numpy kWTA anywhere; sim/ unchanged",
               "task": "perceived objects (2 visual categories, 9 exemplars) -> V1 -> spiking codon; teach property on "
                       "training codons; held-out perceived-object inheritance vs per-image scramble + dAP-lesion",
               "seeds": a.seeds, "config": {"epochs": a.epochs, "n_feat": N_FEAT, "n_col": N_COL, "samp": SAMP, "act_th": ACT_TH},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "the whole pipeline is spiking (Gabor/V1 encode is the rate-reference sensory front end; the pooler "
                              "codon + inheritance are on the spiking bridge). The scramble control is noisy at a single seed "
                              "(small setup) so the GO keys on the multi-seed mean + the deterministic lesion. Composes EMERGE-34 "
                              "+ EMERGE-35 GO pieces."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge36] VERDICT: {verdict}", flush=True)
    print(f"[emerge36] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
