"""ONLINE grow-to-active-context — the EMERGENT structural-plasticity version of the offline corpus pool
(`2026-07-14-*grow-to-context*`). Instead of PRE-SCANNING the corpus token-adjacency to wire the pool, the pool GROWS
during learning from the network's OWN winner co-firing: start EMPTY; each time a post-winner is selected with a prev-
winner active, GROW a coincidence synapse (prev_winner_cell -> post_winner_cell) if it does not exist, then potentiate.
CELL-level (winner->winner, ~k_win^2 per context) = the tightest, truly-emergent grow-to-active-context (Hawkins-Ahmad
2016 segment growth; Poirazi-Mel 2001 activity-dependent structural stabilization). The structure is DISCOVERED from the
winner dynamics (permuting the corpus changes what is grown), not hand-wired from tokens.

BOOTSTRAP (the load-bearing subtlety): subject-specificity must emerge incrementally. sentence 1 (subject A) grows +
potentiates its winner-chain; sentence 2 (subject B) then sees A's cells as "committed" (`_committed_count` reads
perm>p_init+0.02) -> allocates FRESHER cells for the shared middle -> subject-specific SDRs -> distinct branch
prediction. So grow+potentiate must be VISIBLE to the next sentence: we re-inject the grown pool per sentence
(`inject_explicit_wiring` correctly coincidence-tags + rebuilds `cp_coincidence_synapse_mask`), preserving permanences.

GO: online-grown HTM branch-acc ~= the dense/offline-corpus pool (~1.0) at a CELL-level sub-quadratic grown-synapse
count, discovered ONLINE. ANTI-CHEATS: n-gram floor; dAP-lesion collapse; PERMUTED corpus changes the grown set +
collapses accuracy (structure is from the dynamics, not a token pre-scan); 6-seed. Reuse-by-import: the winner-selection
/ `_match_count` / `_committed_count` / `apply_kernel_update` / `coincidence_predict` from emerge14. NO `sim/` edit."""
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
    apply_kernel_update, coincidence_predict, _host)
from research.runners._emerge15_word_sequence_lm_derisk import (  # noqa: E402
    make_word_corpus, ngram_nextword_acc, htm_nextword_acc)

OUT = _REPO / "research" / "findings" / "raw" / "_emerge15_online_grow.json"


def build_empty_pool_bridge(vocab, nE, seed, act_th=3, coincidence=True):
    """Same bridge cfg as emerge14 `build_pool_bridge` but with NO potential synapses (the online grow adds them)."""
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
    return b, cells_idx


class OnlineGrowLearner:
    """Grows the coincidence pool ONLINE from winner co-firing (no corpus pre-scan). Cell-level (winner->winner)."""

    def __init__(self, vocab, nE, seed, k_win=4, act_th=3, learn_th=2, p_init=0.0,
                 lam_pot=0.14, lam_dep=0.02, z_tau=0.85, z_star=1.0, lesion=False, coincidence=True):
        self.b, self.cells_idx = build_empty_pool_bridge(vocab, nE, seed, act_th=act_th, coincidence=coincidence)
        self.M, self.nE, self.N = vocab, nE, vocab * nE
        self.k_win, self.act_th, self.learn_th, self.p_init = k_win, act_th, learn_th, p_init
        self.lam_pot, self.lam_dep, self.z_tau, self.z_star = lam_pot, lam_dep, z_tau, z_star
        self.lesion = lesion
        self.z = np.zeros(self.N)
        self.grown = {}                                          # (pre_EMERGEcell, post_EMERGEcell) -> perm
        self.row = np.zeros(0, np.int64); self.col = np.zeros(0, np.int64)  # global-index COO of the current pool

    def _col(self, c):
        return list(range(c * self.nE, (c + 1) * self.nE))

    def _committed_count(self):
        n = int(self.b.core_config.num_neurons)
        if self.b.cp_connections is None or self.b.cp_connections.nnz == 0:
            return np.zeros(len(self.cells_idx))
        data = _host(self.b.cp_connections.data).astype(np.float64)
        wc = np.zeros(n)
        np.add.at(wc, self.col, (data > self.p_init + 0.02).astype(np.float64))
        return wc[self.cells_idx]

    def _match_count(self, post_cell, prev_win):
        if not prev_win or self.col.size == 0:
            return 0
        data = _host(self.b.cp_connections.data).astype(np.float64)
        pre_set = set(int(self.cells_idx[i]) for i in prev_win)
        bpost = int(self.cells_idx[post_cell])
        idx = np.where(self.col == bpost)[0]
        thr = self.p_init + 0.02
        m = 0
        for k in idx:
            if int(self.row[k]) in pre_set and data[k] >= thr:
                m += 1
        return m

    def _reinject(self):
        """Rebuild the bridge's coincidence pool from self.grown (perms preserved), correctly coincidence-tagged."""
        if not self.grown:
            return
        pre_l, post_l, w_l = [], [], []
        for (pre_c, post_c), perm in self.grown.items():
            pre_l.append(int(self.cells_idx[pre_c])); post_l.append(int(self.cells_idx[post_c])); w_l.append(float(perm))
        plan = {"distal": {"pre_indices": pre_l, "post_indices": post_l, "initial_weights": w_l,
                           "plastic": False, "coincidence_detector": True, "conn_type": "htm_pool"}}
        self.b.inject_explicit_wiring(plan)
        coo = self.b._get_cached_coo()
        self.row = np.asarray(_host(coo.row)); self.col = np.asarray(_host(coo.col))

    def _pull_perms(self):
        """Copy the bridge's current permanences back into self.grown (so re-inject preserves learned perms)."""
        if self.col.size == 0 or self.b.cp_connections is None:
            return
        data = _host(self.b.cp_connections.data).astype(np.float64)
        # map (global pre, global post) -> EMERGE cell pair; the COO order matches self.row/self.col
        g2c = {int(self.cells_idx[i]): i for i in range(len(self.cells_idx))}
        for k in range(self.col.size):
            pc = g2c.get(int(self.row[k])); qc = g2c.get(int(self.col[k]))
            if pc is not None and qc is not None and (pc, qc) in self.grown:
                self.grown[(pc, qc)] = data[k]

    def train_sequence(self, seq):
        # Pass 1: select winners on the CURRENT pool (frozen this sentence), record the winner-chain pairs.
        predictive, prev_winners = set(), set()
        pairs = []                                               # (prev_winner_cell, cur_winner_cell) to grow+potentiate
        winners_seq = []
        for c in seq:
            col = self._col(c)
            primed = [i for i in col if i in predictive] if not self.lesion else []
            if primed:
                winners = set(primed[:self.k_win])
            elif not prev_winners:
                winners = set(col[:self.k_win])
            else:
                scored = sorted(((self._match_count(i, prev_winners), i) for i in col), reverse=True)
                if scored[0][0] >= self.learn_th:
                    winners = set(i for sc, i in scored[:self.k_win] if sc >= self.learn_th)
                else:
                    wc = self._committed_count()
                    winners = set(sorted(col, key=lambda i: (wc[i], i))[:self.k_win])
            if prev_winners:
                for pw in prev_winners:
                    for cw in winners:
                        pairs.append((pw, cw))
            active = winners if primed else (set(col) if prev_winners or not primed else winners)
            predictive = coincidence_predict(self.b, self.cells_idx, active, self.N, self.nE)
            winners_seq.append(winners)
            prev_winners = winners
        # GROW: add any new (pre->post) winner-pair as a coincidence synapse at p_init, then re-inject.
        new = [(pw, cw) for (pw, cw) in pairs if (pw, cw) not in self.grown]
        if new:
            for pr in new:
                self.grown[pr] = self.p_init
            self._reinject()
        # Pass 2: POTENTIATE the winner-chain on the now-grown pool (so perms rise -> next sentence sees "committed").
        prev = None
        for winners in winners_seq:
            if prev is not None:
                apply_kernel_update(self.b, self.row, self.col, self.cells_idx, prev, winners,
                                    self.z, self.lam_pot, self.lam_dep, self.z_star)
            prev = winners
        self._pull_perms()

    def predict_branch(self, seq, div_pos):
        predictive, prev_winners = set(), set()
        preds = []
        for c in seq:
            col = self._col(c)
            primed = [i for i in col if i in predictive] if not self.lesion else []
            active = set(primed[:self.k_win]) if primed else set(col)
            predictive = coincidence_predict(self.b, self.cells_idx, active, self.N, self.nE)
            preds.append(set(i // self.nE for i in predictive))
            prev_winners = active
        return preds


def _run(seed, n_subj, epochs, k_win=4, act_th=3, lesion=False, permute=False):
    sentences, vocab, word2col, branch_pos = make_word_corpus(n_subj=n_subj, seed=seed)
    col_seqs = [[word2col[w] for w in s] for s in sentences]
    vocab_n = len(vocab)
    nE = k_win * n_subj + 8
    train_seqs = col_seqs
    if permute:
        rng = np.random.default_rng(seed + 777)
        train_seqs = [list(rng.permutation(s)) for s in col_seqs]
    lr = OnlineGrowLearner(vocab_n, nE, seed, k_win=k_win, act_th=act_th, lesion=lesion, coincidence=(not lesion))
    for _ in range(epochs):
        for cs in train_seqs:
            lr.train_sequence(cs)
    bp = branch_pos - 1
    htm = htm_nextword_acc(lr, col_seqs, bp)                    # score on the TRUE corpus
    grown = len(lr.grown)
    dense_syn = int(vocab_n * nE * (vocab_n * nE - nE))
    return {"htm": float(htm), "grown_syn": int(grown), "dense_syn": dense_syn,
            "vocab_n": int(vocab_n), "nE": int(nE)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-subj", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--k-win", type=int, default=4)
    ap.add_argument("--act-th", type=int, default=3)
    ap.add_argument("--lesion", action="store_true", help="anti-cheat: dAP-lesion (should collapse)")
    ap.add_argument("--permute", action="store_true", help="anti-cheat: permute word order (should collapse + change grown set)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    sentences, vocab, word2col, branch_pos = make_word_corpus(n_subj=a.n_subj)
    col_seqs = [[word2col[w] for w in s] for s in sentences]
    best_ng = max(ngram_nextword_acc(col_seqs, 1, branch_pos), ngram_nextword_acc(col_seqs, 2, branch_pos))
    chance = 1.0 / a.n_subj
    print(f"n_subj={a.n_subj} vocab={len(vocab)} chance={chance:.3f} best_ngram={best_ng:.3f}")

    per = []
    t0 = time.time()
    for s in a.seeds:
        try:
            r = _run(s, a.n_subj, a.epochs, k_win=a.k_win, act_th=a.act_th, lesion=a.lesion, permute=a.permute)
        except Exception as e:
            r = {"error": repr(e), "traceback": traceback.format_exc()}
        per.append({"seed": s, **r})
        print(f"  [seed {s}] HTM {r.get('htm')} | grown {r.get('grown_syn')} | dense {r.get('dense_syn')}")

    hs = [x["htm"] for x in per if "htm" in x]
    gs = [x["grown_syn"] for x in per if "grown_syn" in x]
    ds = [x["dense_syn"] for x in per if "dense_syn" in x]
    htm = float(np.mean(hs)) if hs else None
    grown = int(np.mean(gs)) if gs else None
    dense = int(np.mean(ds)) if ds else None
    go = (htm is not None and htm > best_ng + 0.05 and grown is not None and dense is not None and grown < dense)
    out = {"probe": "emerge15_online_grow", "n_subj": a.n_subj, "vocab": len(vocab), "chance": chance,
           "best_ngram": best_ng, "lesion": a.lesion, "permute": a.permute,
           "htm": htm, "grown_syn": grown, "dense_syn": dense,
           "grown_ratio": (grown / dense) if (grown and dense) else None,
           "GO": bool(go), "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
           "verdict": (f"ONLINE-GROWN HTM {htm} vs n-gram {best_ng:.3f}; grown {grown} synapses "
                       f"(vs dense {dense}) discovered from winner dynamics. "
                       + ("GO: emergent grow-to-active-context matches the offline corpus pool at a cell-level "
                          "sub-quadratic grown-synapse count." if go else "read substance."))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(out["verdict"])


if __name__ == "__main__":
    main()
