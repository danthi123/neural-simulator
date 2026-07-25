"""gap#4 ON-BRIDGE coincidence-PLATEAU reliable expander — the named escape from the input-drivenness<->reliability
tradeoff. Reuses the EMERGE-35 mechanism: a fixed decorrelated coincidence expansion (each of N_COL columns samples SAMP
input features), driven via `_prime_from_winners` which RESETS the soma+apical and holds the active features SYNCHRONOUSLY
so the dendritic plateau (cp_v_apical) rises DETERMINISTICALLY (reliable, no noisy rate settle, no state carryover — the
exact fix for the reproducibility-0.07 collapse). Read the codon = {columns with cp_v_apical > FLOOR}. This is input-driven
(coincidence) AND reliable (plateau threshold-crossing + full reset) at once.
GO iff held-out LINEAR of the plateau codon rises off 0.34 toward the numpy ceiling on >=5/6 seeds WITH reproducibility
>~0.8. Anti-cheats: non-expanding control (N_COL=n_in), label-shuffle -> chance, pool-silence lesion (no active feats) ->
chance, permuted-features (documented no-op for a random codon)."""
import os, sys, time
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
sys.path.insert(0, "/home/dant123/Projects/sim")
import numpy as np
from research.runners._semantic_inheritance_deep_credit_derisk import make_task_semantic_inheritance
from research.runners._emerge14_stageC_onbridge_learning_derisk import _host
from research.runners._emerge12_stageB2_bridge_tm_derisk import _prime_from_winners

FLOOR = -40.0
SAMP, ACT_TH, TOPK = 3, 2, 4


class PlateauExpander:
    def __init__(self, n_feat, n_col, seed):
        from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
        from sim.bridge import SimulationBridge
        from sim.regions import BrainRegion
        from sim.enums import NeuronModel, NeuronType
        rng = np.random.default_rng(seed)
        self.NF, self.NC = n_feat, n_col
        self.W = np.zeros((n_col, n_feat), bool)
        for c in range(n_col):
            self.W[c, rng.choice(n_feat, min(SAMP, n_feat), replace=False)] = True
        M = n_feat + n_col
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
        cfg.enable_coincidence_detection = True
        cfg.coincidence_weighted_drive = True; cfg.coincidence_k_threshold = float(ACT_TH) - 0.5
        cfg.coincidence_plateau_strength = 160.0; cfg.enable_two_compartment_dap = True; cfg.apical_g_couple = 2.0
        b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                             runtime_state=RuntimeState(), gpu_config=GPUConfig())
        b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        b.runtime_state.actual_seed_used = seed
        b._initialize_simulation_data(called_from_playback_init=False)
        ci = np.asarray(b.region_manager.indices("cells"), int)
        pre, post, w = [], [], []
        for c in range(n_col):
            for f in np.where(self.W[c])[0]:
                pre.append(int(ci[f])); post.append(int(ci[n_feat + c])); w.append(1.0)
        b.inject_explicit_wiring({"ff": {"pre_indices": pre, "post_indices": post, "initial_weights": w,
                                         "plastic": False, "coincidence_detector": True, "conn_type": "ff"}})
        self.b, self.ci = b, ci

    def codon(self, active_feats):
        ab = np.zeros(len(self.ci), bool)
        ab[np.asarray(list(active_feats), int)] = True
        _prime_from_winners(self.b, self.ci, ab)
        vap = getattr(self.b, "cp_v_apical", None)
        if vap is None:
            return np.zeros(self.NC)
        vap = np.asarray(_host(vap))[self.ci]
        return (vap[self.NF:self.NF + self.NC] > FLOOR).astype(np.float64)


def _sm(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def fit_lin(X, y, k, iters=600, lr=0.5, l2=3e-3):
    n, d = X.shape; W = np.zeros((d, k)); b = np.zeros(k); Y = np.eye(k)[y]
    for _ in range(iters):
        P = _sm(X @ W + b); g = (P - Y) / n; W -= lr * (X.T @ g + l2 * W); b -= lr * g.sum(0)
    return lambda Z: np.argmax(Z @ W + b, 1)


def topk_active(X, topk):
    th = np.sort(X, axis=1)[:, -topk][:, None]
    return [set(np.where(row >= t)[0]) for row, t in zip(X, th[:, 0])]


N_COL = 200
SEEDS = (42, 43, 44, 100, 101, 102)
res = {r: {"lin": [], "rep": []} for r in ["CODON expand plateau", "CODON non-expand(ctrl)", "label-shuffle(ctrl)", "pool-silence(ctrl)"]}
for SEED in SEEDS:
    t0 = time.time()
    (Xtr, ytr, _), (Xte, yte, _), meta, idx = make_task_semantic_inheritance(
        SEED, n_super=12, n_members=8, held_per_super=3, n_prop=2, n_obs=16, member_id_dim=3, noise=0.02)
    n_in = Xtr.shape[1]; k = meta["k_classes"]; inh = idx["inh_idx"]
    srng = np.random.default_rng(SEED * 13 + 1); keep = srng.permutation(len(Xtr))[:96]
    Xb, yb = Xtr[keep], ytr[keep]; Xh, yh = Xte[inh], yte[inh]
    afb = topk_active(Xb, TOPK); afh = topk_active(Xh, TOPK)
    for tag, ncol in [("CODON expand plateau", N_COL), ("CODON non-expand(ctrl)", n_in)]:
        exp = PlateauExpander(n_in, ncol, SEED)
        Cb = np.asarray([exp.codon(a) for a in afb]); Ch = np.asarray([exp.codon(a) for a in afh])
        rep2 = np.asarray([exp.codon(a) for a in afb[:8]])
        rep = float(np.mean([np.corrcoef(Cb[i], rep2[i])[0, 1] if Cb[i].std() > 0 and rep2[i].std() > 0 else 1.0 for i in range(8)]))
        clf = fit_lin(Cb, yb, k); res[tag]["lin"].append(float(np.mean(clf(Ch) == yh))); res[tag]["rep"].append(rep)
        if tag == "CODON expand plateau":
            ysh = np.random.default_rng(SEED).permutation(yb)
            clfs = fit_lin(Cb, ysh, k); res["label-shuffle(ctrl)"]["lin"].append(float(np.mean(clfs(Ch) == yh))); res["label-shuffle(ctrl)"]["rep"].append(rep)
            # pool-silence: codon with NO active feats -> should be constant -> chance
            Cs = np.asarray([exp.codon(set()) for _ in range(len(Xh))])
            clf0 = fit_lin(np.asarray([exp.codon(set()) for _ in afb]), yb, k)
            res["pool-silence(ctrl)"]["lin"].append(float(np.mean(clf0(Cs) == yh))); res["pool-silence(ctrl)"]["rep"].append(0.0)
            spars = float(Cb.mean())
    print(f"  seed {SEED} ({time.time()-t0:.0f}s) n_in={n_in} N_COL={N_COL} codon_spars={spars:.3f} chance={1.0/k:.2f} n_ho={len(inh)}", flush=True)

print("\n===== gap#4 PLATEAU-CODON held-out (mean over 3 seeds) — GO = ho-LIN off 0.34 + reproducibility high =====", flush=True)
for r in res:
    L = np.array(res[r]["lin"]); R = np.array(res[r]["rep"])
    print(f"  {r:24s} ho-lin {L.mean():.3f}+/-{L.std():.3f} {[round(x,3) for x in L]}   reproducibility {R.mean():.3f}", flush=True)
print("\nnumpy ceilings: random-ReLU 0.772, codon 0.617; boundary 0.34; input-lin 0.284", flush=True)
print("PLATEAU-EXPANDER DONE", flush=True)
