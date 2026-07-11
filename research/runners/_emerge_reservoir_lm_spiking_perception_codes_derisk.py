"""EMERGENCE-BAR follow-on (b): make the perception->category SURFACING itself SPIKING. The perception-grounded close
(`_emerge_reservoir_lm_perception_grounded_codes_derisk.py`) surfaced the V1 perception features into a category SDR with
a NUMPY fixed random codon. Here that codon is replaced by the FULLY-SPIKING Marr-Albus codon (EMERGE-35, F.12): a large
column layer on a real `SimulationBridge`, each column sampling `SAMP` V1 features via a fixed DECORRELATED coincidence
projection, firing via the validated `coincidence_weighted_drive` (a column fires when >= ACT_TH of its sampled features
are active) -- NO numpy kWTA. Same-category perceived objects (overlapping V1 features) converge on OVERLAPPING column
codons; a held-out perceived object inherits. That spiking codon feeds the (already-spiking) reservoir ladder; the
one-step-local-delta read-out is unchanged. ⇒ pixels -> Gabor/V1 -> SPIKING codon -> spiking reservoir -> Rung-3
generalization, with the category surfacing on spikes. NO `sim/` edit, NO BPTT, NO deep credit. CPU numpy-backend bridge.

ARMS: main (spiking codon over CATEGORY-structured V1 features) ; scramble (per-image PIXEL SCRAMBLE -> V1 has no category
structure) ; onehot (no block) ; untrained. METRIC: `heldagent_cat_acc`. GO: main >> scramble on all 6 seeds.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, time, traceback
from pathlib import Path
import numpy as np

import research.runners._emerge_reservoir_lm_rung3_systematic_generation_derisk as r3
from research.runners._emerge34_perception_grounded_emergence_derisk import (
    build_shape_set, build_gabor_response_matrix, encode_v1)
from research.runners._emerge14_stageC_onbridge_learning_derisk import _host
from research.runners._emerge12_stageB2_bridge_tm_derisk import _prime_from_winners

OUT = Path("research/findings/raw/_reslm_spiking_perception.json")

N_EX = 9
T_ACTIVE = 20        # top-T active V1 cells = each object's perception feature set
N_COL = 120          # spiking column layer (>> N_FEAT after V1 reduction)
SAMP = 3             # each column samples 3 decorrelated V1 features
ACT_TH = 2           # column fires if >= 2 of its 3 sampled features active (Marr-Albus codon)
K = 12               # (kept for the reservoir code width parity; the spiking codon size is data-driven)
V = r3.V
ACTION_POS = 3
FLOOR = -40.0

_GABOR = None
def _gabor():
    global _GABOR
    if _GABOR is None:
        _GABOR = build_gabor_response_matrix()
    return _GABOR


class SpikingCodon:
    """A fully-spiking Marr-Albus sparse-expansion codon (EMERGE-35 / F.12) over NF input features: a column layer on a
       real SimulationBridge, each column sampling SAMP features via a fixed decorrelated coincidence projection; the
       column fires when >= ACT_TH of its features are active. `codon(active_feature_set)` -> the fired column indices."""

    def __init__(self, nf, seed):
        from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
        from sim.bridge import SimulationBridge
        from sim.regions import BrainRegion
        from sim.enums import NeuronModel, NeuronType
        rng = np.random.default_rng(seed)
        self.nf = nf
        M = nf + N_COL
        self.W = np.zeros((N_COL, nf))
        for c in range(N_COL):
            self.W[c, rng.choice(nf, SAMP, replace=False)] = 1
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
        for c in range(N_COL):
            for f in np.where(self.W[c] > 0)[0]:
                pre.append(int(ci[f])); post.append(int(ci[nf + c])); w.append(1.0)
        b.inject_explicit_wiring({"ff": {"pre_indices": pre, "post_indices": post, "initial_weights": w,
                                         "plastic": False, "coincidence_detector": True, "conn_type": "ff"}})
        self.b, self.ci = b, ci

    def codon(self, active_features):
        ab = np.zeros(len(self.ci), bool)
        for f in active_features:
            ab[f] = True
        _prime_from_winners(self.b, self.ci, ab)
        vap = getattr(self.b, "cp_v_apical", None)
        if vap is None:
            return []
        vap = _host(vap)[self.ci]
        return sorted(c for c in range(N_COL) if vap[self.nf + c] > FLOOR)


def perception_active_features(seed, scramble):
    """Each animal's top-T active V1 cells (a set of feature indices) from its SEEN shape. Returns (feats_by_animal, NF)."""
    rng = np.random.default_rng(seed)
    imgs, labels, _ = build_shape_set(n_categories=2, n_exemplars=N_EX, rng=rng)
    imgs = np.asarray(imgs)
    if scramble:
        r = np.random.default_rng(seed * 7 + 1)
        imgs = np.stack([im.flatten()[r.permutation(im.size)].reshape(im.shape) for im in imgs])
    Vv = encode_v1(imgs, _gabor()); NF = Vv.shape[1]
    cat_imgs = {0: [i for i in range(len(labels)) if labels[i] == 0],
                1: [i for i in range(len(labels)) if labels[i] == 1]}
    feats = {}
    for ci_, cat in enumerate(("PRED", "PREY")):
        for j, a in enumerate(r3.CAT_ANIMALS[cat]["train"] + r3.CAT_ANIMALS[cat]["held"]):
            v1 = Vv[cat_imgs[ci_][j]]
            feats[a] = set(int(x) for x in np.argsort(-v1)[:T_ACTIVE])
    return feats, NF


def build_codes(seed, scramble):
    feats, NF = perception_active_features(seed, scramble)
    # Restrict the codon's feature space to the UNION of ever-active V1 cells -> the active-fraction is high enough that the
    # coincidence columns (sample SAMP, fire at >= ACT_TH active) actually fire (the raw V1 space is too sparse for the codon).
    active_union = sorted(set().union(*[feats[a] for a in feats]))
    remap = {f: i for i, f in enumerate(active_union)}
    nf_red = len(active_union)
    feats_red = {a: {remap[f] for f in feats[a]} for a in feats}
    codon = SpikingCodon(nf_red, seed)                                          # ONE fully-spiking codon bridge over the reduced space
    return {a: codon.codon(feats_red[a]) for a in feats}, N_COL


def word_code(w, codes, use_block, d_code):
    v = np.zeros(d_code); v[N_COL + r3.WORD_IDX[w]] = 1.0
    if use_block and w in r3.ANIMAL_CAT:
        for c in codes[w]:
            v[c] = 1.0
    return v


def cum_feat(res, prefix, codes, use_block, d_code):
    U = np.asarray([word_code(w, codes, use_block, d_code) for w in prefix])
    return res.per_token_states(U, feature="running_cumulative")[ACTION_POS - 1]


def _train(feats, tgts, ncls, epochs, lr, seed):
    X = np.array(feats); mean = X.mean(0); std = X.std(0) + 1e-6
    Xn = np.concatenate([(X - mean) / std, np.ones((len(X), 1))], 1)
    W = np.zeros((ncls, Xn.shape[1])); rng = np.random.default_rng(seed * 13 + 1); idx = list(range(len(Xn)))
    Ws = np.zeros_like(W); na = 0; burn = epochs // 2
    for ep in range(epochs):
        rng.shuffle(idx)
        for i in idx:
            z = W @ Xn[i]; z = z - z.max(); p = np.exp(z); p /= p.sum()
            t = np.zeros(ncls); t[tgts[i]] = 1.0; W += lr * np.outer(t - p, Xn[i])
        if ep >= burn:
            Ws += W; na += 1
    return (Ws / na if na else W), mean, std


def run_arm(seed, arm, epochs, lr, n_pool):
    scramble = (arm == "scramble"); use_block = (arm != "onehot")
    codes, ncol = build_codes(seed, scramble)
    d_code = ncol + V
    res = r3.ReservoirStates(d_code, seed=seed, n=n_pool)
    feats = [cum_feat(res, s[:3], codes, use_block, d_code) for s in r3.TRAIN_SENTS]
    tgts = [r3.WORD_IDX[s[3]] for s in r3.TRAIN_SENTS]
    if arm == "untrained":
        W = np.zeros((V, len(feats[0]) + 1)); m = np.zeros(len(feats[0])); sd = np.ones(len(feats[0]))
    else:
        W, m, sd = _train(feats, tgts, V, epochs, lr, seed)

    def pred(prefix):
        f = cum_feat(res, prefix, codes, use_block, d_code); x = np.concatenate([(f - m) / sd, [1.0]])
        return r3.WORDS[int(np.argmax(W @ x))]
    ok = tot = 0
    for prefix, true_cat in r3.HELD_PREFIXES:
        p = pred(prefix); ok += int(p in r3.ACTION_CAT and r3.ACTION_CAT[p] == true_cat); tot += 1
    return {"arm": arm, "heldagent_cat_acc": ok / tot}


ARMS = ["main", "scramble", "onehot", "untrained"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-pool", type=int, default=300)
    ap.add_argument("--json", type=str, default=str(OUT))
    args = ap.parse_args()

    t0 = time.time(); per_seed = {}
    for seed in args.seeds:
        rb = {}
        for arm in ARMS:
            try:
                rb[arm] = run_arm(seed, arm, args.epochs, args.lr, args.n_pool)
            except Exception as e:
                rb[arm] = {"arm": arm, "error": f"{e}", "trace": traceback.format_exc()}
            r = rb[arm]
            print(f"[seed {seed}] {arm:10s} " + (f"heldagent={r.get('heldagent_cat_acc'):.3f}" if "error" not in r else r["error"][:200]), flush=True)
        per_seed[seed] = rb

    def agg(arm):
        vals = [per_seed[s][arm]["heldagent_cat_acc"] for s in args.seeds if "error" not in per_seed[s][arm]]
        return float(np.mean(vals)) if vals else None
    aggregate = {arm: agg(arm) for arm in ARMS}
    per_seed_go = []
    for s in args.seeds:
        rb = per_seed[s]
        if any("error" in rb[a] for a in ARMS):
            per_seed_go.append(False); continue
        m = rb["main"]["heldagent_cat_acc"]; sc = rb["scramble"]["heldagent_cat_acc"]
        per_seed_go.append(bool(m >= 0.70 and (m - sc) >= 0.20))
    n_go = int(sum(per_seed_go))

    out = {"runner": "_emerge_reservoir_lm_spiking_perception_codes_derisk", "seeds": args.seeds,
           "n_col": N_COL, "samp": SAMP, "act_th": ACT_TH, "t_active": T_ACTIVE,
           "per_seed": {str(s): per_seed[s] for s in args.seeds}, "aggregate": aggregate,
           "per_seed_go": per_seed_go, "n_go": n_go, "n_seeds": len(args.seeds), "elapsed_s": round(time.time() - t0, 1)}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True); Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"\nAGG main={aggregate['main']} scramble={aggregate['scramble']} onehot={aggregate['onehot']} "
          f"GO {n_go}/{len(args.seeds)} ({out['elapsed_s']}s) -> {args.json}", flush=True)


if __name__ == "__main__":
    main()
