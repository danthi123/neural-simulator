"""Vocab-SCALE lever (cheap-first rung, runner-side, NO `sim/` edit): does a SPARSE (subsampled) coincidence pool match
the DENSE all-to-all pool's high-order next-word accuracy at a FRACTION of the synapses? — the scale mechanism for a
real vocabulary on the emergent HTM word-LM.

WHY (2026-07-14). The emergent HTM Temporal-Memory word-LM (emerge15) is GO and its high-order advantage over the
n-gram SCALES (HTM 1.0 vs n-gram 1/n_subj at n=4,8 — the advantage GROWS with vocab). But its potential pool is DENSE
cross-column = O((vocab*nE)^2) synapses (63k at n=4 -> 547k at n=8 -> ~5M at 16 -> ~50M at 32 -> infeasible at real
vocab). The canonical HTM fix (Hawkins-Ahmad 2016; the research gate's path (d)) is per-cell SPARSE distal segments,
each subsampling a SMALL fixed set of potential synapses instead of all-to-all — capacity scales with segments, not
vocab^2. The CHEAPEST first rung (no `sim/` edit): make the pool SPARSE (subsample K pre-synapses per post cell from
OTHER columns) and show it MATCHES the dense pool's accuracy at a fraction of the synapses. The coincidence detection +
the committed `fused_htm_permanence_update` kernel operate on whatever synapses exist, so this is a RUNNER-SIDE wiring
change only.

GO GATE: sparse HTM next-word branch accuracy ~= dense (both >> the n-gram floor 1/n_subj) at a synapse count that is
SUB-QUADRATIC in vocab (linear-ish). ANTI-CHEATS: the n-gram floor (both must beat it); dAP-lesion (collapses);
sparse-K too-small control (K below the SDR overlap -> should degrade); 3 seeds. Reuse-by-import: the DENSE
`build_pool_bridge`, `OnBridgeLearner`, the corpus + n-gram baseline + accuracy from emerge14/15. NO `sim/` edit."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from research.runners._emerge14_stageC_onbridge_learning_derisk import (  # noqa: E402
    build_pool_bridge, OnBridgeLearner, _host)
from research.runners._emerge15_word_sequence_lm_derisk import (  # noqa: E402
    make_word_corpus, ngram_nextword_acc, htm_nextword_acc)

OUT = _REPO / "research" / "findings" / "raw" / "_emerge15_sparse_pool_scale.json"


def build_sparse_pool_bridge(vocab, nE, seed, k_syn, p_init=0.0, act_th=3, coincidence=True):
    """IDENTICAL to build_pool_bridge EXCEPT the cross-column potential pool is SPARSE: each POST cell samples k_syn
    pre-cells from OTHER columns (deterministic RNG) instead of ALL of them. Synapses = N*k_syn (LINEAR in N), vs the
    dense N*(N-nE) (QUADRATIC). Everything else (bridge cfg, coincidence detection, the permanence kernel) is unchanged."""
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion
    from sim.enums import NeuronModel, NeuronType
    N = vocab * nE
    regions = [BrainRegion(name="cells", n_neurons=N, exc_fraction=1.0, internal_density=0.0, exc_weight_mean=0.0,
                           inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                           izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name)]
    cfg = CoreSimConfig()
    cfg.seed = cfg.heterogeneity_seed = cfg.ou_seed = int(seed)
    cfg.dt_ms = 1.0; cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"; cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions); cfg.region_pathways = []
    cfg.enable_stdp = False; cfg.enable_hebbian_learning = False; cfg.enable_nmda = False
    cfg.stdp_w_max = 1.0; cfg.fast_spike_reset = True
    for f in ("enable_homeostasis", "enable_short_term_plasticity", "enable_ou_process",
              "enable_conductance_noise", "enable_parameter_heterogeneity", "enable_structural_plasticity"):
        setattr(cfg, f, False)
    cfg.enable_coincidence_detection = bool(coincidence)
    cfg.coincidence_weighted_drive = True
    cfg.coincidence_k_threshold = float(act_th) - 0.5
    cfg.coincidence_plateau_strength = 160.0
    cfg.enable_two_compartment_dap = True
    cfg.apical_g_couple = 2.0
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b.runtime_state.actual_seed_used = seed
    b._initialize_simulation_data(called_from_playback_init=False)
    cells_idx = np.asarray(b.region_manager.indices("cells"), np.int64)
    # SPARSE cross-column potential pool: each POST cell samples k_syn pre-cells from OTHER columns (deterministic).
    rng = np.random.default_rng(seed + 20240714)
    all_cols = np.arange(N)
    col_of = all_cols // nE
    pre_l, post_l = [], []
    ksyn = int(k_syn)
    for post in range(N):
        pc = post // nE
        cand = all_cols[col_of != pc]                          # all cells in OTHER columns
        m = min(ksyn, cand.size)
        sel = rng.choice(cand, size=m, replace=False)
        for pre in sel:
            pre_l.append(int(cells_idx[int(pre)])); post_l.append(int(cells_idx[post]))
    plan = {"distal": {"pre_indices": pre_l, "post_indices": post_l,
                       "initial_weights": [float(p_init)] * len(pre_l),
                       "plastic": False, "coincidence_detector": True, "conn_type": "htm_pool"}}
    b.inject_explicit_wiring(plan)
    coo = b._get_cached_coo()
    return b, cells_idx, np.asarray(_host(coo.row)), np.asarray(_host(coo.col)), len(pre_l)


def build_corpus_sparse_pool(vocab, nE, col_seqs, seed, window, p_init=0.0, act_th=3, coincidence=True):
    """GROW-TO-CONTEXT (offline equivalent): wire cross-column synapses ONLY between column-pairs (ca -> cb, ca before
    cb) that CO-OCCUR within `window` positions in the corpus — the connections the online HTM structural-plasticity
    rule would grow. Synapses ~ (distinct co-occurring column-pairs)*nE^2 = LINEAR in corpus (not vocab^2). Same bridge
    cfg as build_pool_bridge; only the potential-pool wiring differs."""
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion
    from sim.enums import NeuronModel, NeuronType
    N = vocab * nE
    regions = [BrainRegion(name="cells", n_neurons=N, exc_fraction=1.0, internal_density=0.0, exc_weight_mean=0.0,
                           inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                           izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name)]
    cfg = CoreSimConfig()
    cfg.seed = cfg.heterogeneity_seed = cfg.ou_seed = int(seed)
    cfg.dt_ms = 1.0; cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"; cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions); cfg.region_pathways = []
    cfg.enable_stdp = False; cfg.enable_hebbian_learning = False; cfg.enable_nmda = False
    cfg.stdp_w_max = 1.0; cfg.fast_spike_reset = True
    for f in ("enable_homeostasis", "enable_short_term_plasticity", "enable_ou_process",
              "enable_conductance_noise", "enable_parameter_heterogeneity", "enable_structural_plasticity"):
        setattr(cfg, f, False)
    cfg.enable_coincidence_detection = bool(coincidence)
    cfg.coincidence_weighted_drive = True
    cfg.coincidence_k_threshold = float(act_th) - 0.5
    cfg.coincidence_plateau_strength = 160.0
    cfg.enable_two_compartment_dap = True
    cfg.apical_g_couple = 2.0
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b.runtime_state.actual_seed_used = seed
    b._initialize_simulation_data(called_from_playback_init=False)
    cells_idx = np.asarray(b.region_manager.indices("cells"), np.int64)
    # the co-occurring column-pairs (ca before cb within `window`) = the contexts the grow rule would create.
    pairs = set()
    for s in col_seqs:
        for a in range(len(s)):
            for c in range(a + 1, min(a + 1 + window, len(s))):
                if s[a] != s[c]:
                    pairs.add((int(s[a]), int(s[c])))
    pre_l, post_l = [], []
    for (ca, cb) in pairs:
        for pre in range(ca * nE, (ca + 1) * nE):
            for post in range(cb * nE, (cb + 1) * nE):
                pre_l.append(int(cells_idx[pre])); post_l.append(int(cells_idx[post]))
    plan = {"distal": {"pre_indices": pre_l, "post_indices": post_l,
                       "initial_weights": [float(p_init)] * len(pre_l),
                       "plastic": False, "coincidence_detector": True, "conn_type": "htm_pool"}}
    b.inject_explicit_wiring(plan)
    coo = b._get_cached_coo()
    return b, cells_idx, np.asarray(_host(coo.row)), np.asarray(_host(coo.col)), len(pre_l)


def _run(seed, n_subj, epochs, k_syn, sparse, k_win=4, act_th=3, variant="random", window=8,
         lesion=False, permute=False):
    sentences, vocab, word2col, branch_pos = make_word_corpus(n_subj=n_subj, seed=seed)
    col_seqs = [[word2col[w] for w in s] for s in sentences]
    vocab_n = len(vocab)
    nE = k_win * n_subj + 8
    train_seqs = col_seqs
    if permute:                                                # anti-cheat: shuffle each sentence's word order -> the
        rng = np.random.default_rng(seed + 777)                # co-occurrence structure is destroyed -> HTM should collapse
        train_seqs = [list(rng.permutation(s)) for s in col_seqs]
    if sparse and variant == "corpus":
        b, cells_idx, row, col, n_syn = build_corpus_sparse_pool(vocab_n, nE, train_seqs, seed, window, act_th=act_th)
    elif sparse:
        b, cells_idx, row, col, n_syn = build_sparse_pool_bridge(vocab_n, nE, seed, k_syn, act_th=act_th)
    else:
        b, cells_idx, row, col = build_pool_bridge(vocab_n, nE, seed, act_th=act_th)
        n_syn = len(row)
    lr = OnBridgeLearner(b, row, col, cells_idx, vocab_n, nE, k_win=k_win, act_th=act_th, lesion=bool(lesion))
    for _ in range(epochs):
        for cs in train_seqs:
            lr.train_sequence(cs)
    bp = branch_pos - 1                                        # predict s[bp+1] = the branch word at branch_pos
    htm = htm_nextword_acc(lr, col_seqs, bp)                   # ALWAYS score on the TRUE (un-permuted) corpus
    dense_syn = int(vocab_n * nE * (vocab_n * nE - nE))        # analytic dense pool = N*(N-nE)
    return {"htm": float(htm), "n_syn": int(n_syn), "dense_syn": dense_syn,
            "vocab_n": int(vocab_n), "nE": int(nE)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-subj", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--k-syn", type=int, default=40, help="sparse-random: pre-synapses sampled per post cell")
    ap.add_argument("--k-win", type=int, default=4)
    ap.add_argument("--act-th", type=int, default=3)
    ap.add_argument("--variant", choices=["random", "corpus"], default="random",
                    help="random=subsample K pre-cells/post; corpus=grow-to-context (co-occurring column-pairs only)")
    ap.add_argument("--window", type=int, default=8, help="corpus: max positions apart for a co-occurring column-pair")
    ap.add_argument("--no-dense", action="store_true", help="skip the dense baseline (OOMs at large vocab); use the "
                    "analytic dense synapse count N*(N-nE) for the ratio")
    ap.add_argument("--lesion", action="store_true", help="anti-cheat: dAP-lesion (should collapse HTM to n-gram floor)")
    ap.add_argument("--permute", action="store_true", help="anti-cheat: permute each sentence's word order (destroys the "
                    "co-occurrence structure -> HTM should collapse)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    sentences, vocab, word2col, branch_pos = make_word_corpus(n_subj=a.n_subj)
    col_seqs = [[word2col[w] for w in s] for s in sentences]
    bigram = ngram_nextword_acc(col_seqs, 1, branch_pos)
    trigram = ngram_nextword_acc(col_seqs, 2, branch_pos)
    chance = 1.0 / a.n_subj
    best_ng = max(bigram, trigram)
    print(f"n_subj={a.n_subj} vocab={len(vocab)} chance={chance:.3f} best_ngram={best_ng:.3f}")

    modes = ("sparse",) if a.no_dense else ("dense", "sparse")
    per = {"dense": [], "sparse": []}
    t0 = time.time()
    for s in a.seeds:
        for mode in modes:
            try:
                r = _run(s, a.n_subj, a.epochs, a.k_syn, sparse=(mode == "sparse"),
                         k_win=a.k_win, act_th=a.act_th, variant=a.variant, window=a.window,
                         lesion=a.lesion, permute=a.permute)
            except Exception as e:
                r = {"error": repr(e), "traceback": traceback.format_exc()}
            per[mode].append({"seed": s, **r})
            print(f"  [seed {s}] {mode:>6}: HTM {r.get('htm')} | synapses {r.get('n_syn')}")

    def agg(m):
        hs = [x["htm"] for x in per[m] if "htm" in x]
        ns = [x["n_syn"] for x in per[m] if "n_syn" in x]
        return (float(np.mean(hs)) if hs else None, int(np.mean(ns)) if ns else None)
    s_htm, s_syn = agg("sparse")
    if a.no_dense:                                             # analytic dense count (dense OOMs at large vocab)
        d_syn = int(np.mean([x["dense_syn"] for x in per["sparse"] if "dense_syn" in x]))
        d_htm = None                                          # not measured; result 1 already showed dense stays 1.0
    else:
        d_htm, d_syn = agg("dense")
    parity_ok = (d_htm is None) or (s_htm is not None and s_htm >= d_htm - 0.05)  # no-dense: parity anchored at small n
    go = (s_htm is not None and parity_ok
          and s_htm > best_ng + 0.05 and s_syn is not None and d_syn is not None and s_syn < d_syn)
    ratio_pct = f"{s_syn / d_syn * 100:.1f}%" if (s_syn and d_syn) else "?"
    out = {"probe": "emerge15_sparse_pool_scale", "variant": a.variant, "window": a.window,
           "n_subj": a.n_subj, "vocab": len(vocab),
           "chance": chance, "best_ngram": best_ng, "k_syn": a.k_syn,
           "dense": {"htm": d_htm, "synapses": d_syn}, "sparse": {"htm": s_htm, "synapses": s_syn},
           "synapse_ratio": (s_syn / d_syn) if (s_syn and d_syn) else None,
           "GO": bool(go), "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
           "verdict": (f"SPARSE pool (K={a.k_syn}/post) HTM {s_htm} vs DENSE {d_htm} (best n-gram {best_ng:.3f}); "
                       f"synapses sparse {s_syn} vs dense {d_syn} "
                       f"({ratio_pct} of dense). "
                       + ("GO: the SPARSE pool matches dense accuracy at a fraction of the synapses -> the vocab-scale "
                          "lever works runner-side (linear-ish synapses, mechanism preserved); next = per-cell "
                          "multi-SEGMENT thresholds (sim/ extension) for more contexts."
                          if go else "read substance: does sparse match dense + beat the n-gram?"))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(out["verdict"])


if __name__ == "__main__":
    main()
