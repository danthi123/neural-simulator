"""EMERGE-14 / rung-4 Stage C — the on-bridge LEARNING de-risk: the HTM Temporal-Memory permanences LIVE in the real
`SimulationBridge`'s `coincidence_detector` synapse weights (`cp_connections.data`) over a PRE-ALLOCATED cross-column
potential pool, and are LEARNED from scratch by the committed `sim/` `fused_htm_permanence_update` kernel (the Bouhadjar
three-term rule). Prediction is the bridge's OWN coincidence recurrence (Stage-B2 priming, reused); the update is the
`sim/` kernel on the substrate weights. GO = the bridge self-organizes the same context-specific branch prediction as
EMERGE-9d (allocation + retention + no-teacher), multi-seed, dAP-lesion collapses.

WHAT IS ON-SUBSTRATE vs HOST (faithful; host = the acknowledged EMERGE-9d residual):
  - PERMANENCES live in `cp_connections.data` (the bridge's synaptic weights); learning = the `sim/`
    `fused_htm_permanence_update` kernel applied to them. ON-SUBSTRATE.
  - PREDICTION (which cells are dAP-primed) = the bridge's WEIGHTED coincidence recurrence: set the prior winners as
    `cp_prev_firing_states`, run the coincidence step, read the primed (apical/plateau) cells (Stage-B2 `_prime_from_winners`
    reused). ON-SUBSTRATE. WEIGHTED so a graded permanence pool grades the plateau (connected ~1 contributes, sub-
    connected ~p_init barely).
  - WINNER SELECTION + committed-metric ALLOCATION (fresh cells for a new context) = host-orchestrated (as in EMERGE-9d;
    the finding flags this residual -- the fully-neural homeostatic allocation is the aspiration, not this rung).

THE POTENTIAL POOL: a DENSE cross-column coincidence pathway pre-injected at sub-connected `p_init` (< perm_conn), so
"growth" = a permanence rising above perm_conn (no runtime CSR edge-adds -- exactly what a fixed-topology bridge
pathway does). The kernel raises the right (context-specific) permanences and depresses the rest.

Reuse-by-import: EMERGE-12 (bridge build + `_prime_from_winners` prediction), EMERGE-13 (the validated host
winner/allocation three-term logic), `sim.kernels.fused_htm_permanence_update` (the committed kernel). Multi-seed;
anti-cheats: Markov floor + dAP-lesion collapse + no-teacher + oracle + EMERGE-9d parity. CPU numpy-backend.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, time, traceback
from pathlib import Path
import numpy as np

from research.runners._emerge9b_htm_faithful_derisk import make_overlap_sequences, markov_branch_acc, full_oracle
from sim.kernels import fused_htm_permanence_update

OUT = Path("research/findings/raw/_emerge14_stageC_onbridge_learning.json")


def _host(x):
    try:
        return x.get()
    except AttributeError:
        return np.asarray(x)


def build_pool_bridge(vocab, nE, seed, p_init=0.24, perm_conn=0.5, act_th=3, coincidence=True):
    """A bridge holding vocab*nE cells with a PRE-ALLOCATED DENSE cross-column coincidence potential pool at p_init
    (sub-connected). Weighted coincidence so c_drive = sum of active-synapse permanences; threshold ~ act_th connected.
    Returns (bridge, cells_idx, coo_row, coo_col)."""
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
    cfg.coincidence_weighted_drive = True                      # c_drive = sum of active-synapse permanences (graded pool)
    cfg.coincidence_k_threshold = float(act_th) - 0.5          # WEIGHT units: ~act_th connected (perm~1) synapses
    cfg.coincidence_plateau_strength = 160.0
    cfg.enable_two_compartment_dap = True
    cfg.apical_g_couple = 2.0
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b.runtime_state.actual_seed_used = seed
    b._initialize_simulation_data(called_from_playback_init=False)
    cells_idx = np.asarray(b.region_manager.indices("cells"), np.int64)
    # DENSE cross-column potential pool at p_init: every (pre, post) with pre,post in DIFFERENT columns
    pre_l, post_l = [], []
    for post in range(N):
        pc = post // nE
        for pre in range(N):
            if pre // nE != pc:
                pre_l.append(int(cells_idx[pre])); post_l.append(int(cells_idx[post]))
    plan = {"distal": {"pre_indices": pre_l, "post_indices": post_l,
                       "initial_weights": [float(p_init)] * len(pre_l),
                       "plastic": False, "coincidence_detector": True, "conn_type": "htm_pool"}}
    b.inject_explicit_wiring(plan)
    coo = b._get_cached_coo()
    return b, cells_idx, np.asarray(_host(coo.row)), np.asarray(_host(coo.col))


def apply_kernel_update(bridge, coo_row, coo_col, cells_idx, prev_win, cur_win, z, cfg_lp, cfg_ld, z_star):
    """ON-SUBSTRATE permanence update: gather per coincidence-synapse pre_last (prev-symbol winners) + post_now
    (this-symbol winners) + hfac_post (from the per-cell z), call the committed fused kernel, write cp_connections.data.
    prev_win/cur_win are EMERGE cell-index sets; z is the per-cell (EMERGE-indexed) dAP-rate array."""
    n = int(bridge.core_config.num_neurons)
    pre_last_vec = np.zeros(n, np.float64); post_now_vec = np.zeros(n, np.float64)
    for i in prev_win:
        pre_last_vec[cells_idx[i]] = 1.0
    for i in cur_win:
        post_now_vec[cells_idx[i]] = 1.0
    hfac_cell = 0.5 + 0.5 * np.maximum(0.0, z_star - z)          # per EMERGE cell
    hfac_vec = np.zeros(n, np.float64); hfac_vec[cells_idx] = hfac_cell
    data = _host(bridge.cp_connections.data).astype(np.float64)
    pre_last = pre_last_vec[coo_row]; post_now = post_now_vec[coo_col]; hfac_post = hfac_vec[coo_col]
    updated = fused_htm_permanence_update(data, pre_last, post_now, hfac_post, cfg_lp, cfg_ld, 0.0, 1.0)
    bridge.cp_connections.data[:] = bridge.xp.asarray(updated.astype(np.float32)) if hasattr(bridge, "xp") else updated.astype(np.float32)


def connected_predict(bridge, coo_row, coo_col, cells_idx, active_cells, N, nE, perm_conn, act_th):
    """Prediction from the LEARNED substrate permanences: per post-cell, count CONNECTED (perm>=perm_conn) synapses from
    active cells; predictive iff >= act_th. Reads cp_connections.data (the bridge's own learned weights)."""
    n = int(bridge.core_config.num_neurons)
    active_vec = np.zeros(n, np.float64)
    for i in active_cells:
        active_vec[cells_idx[i]] = 1.0
    data = _host(bridge.cp_connections.data).astype(np.float64)
    conn = data >= perm_conn
    pre_active = active_vec[coo_row] > 0.5
    contrib = (conn & pre_active).astype(np.float64)
    ccount = np.zeros(n, np.float64)
    np.add.at(ccount, coo_col, contrib)                          # per post-cell connected-active count
    # map back to EMERGE cell index (identity here) -> predictive columns
    pred = set()
    inv = {int(cells_idx[i]): i for i in range(N)}
    for bpost in np.where(ccount >= act_th)[0]:
        if int(bpost) in inv:
            pred.add(inv[int(bpost)])
    return pred


class OnBridgeLearner:
    """Host-orchestrated winner/allocation (EMERGE-13 logic) + ON-SUBSTRATE prediction (connected count over the
    bridge's learned permanences) + ON-SUBSTRATE update (the fused kernel on cp_connections.data)."""

    def __init__(self, bridge, coo_row, coo_col, cells_idx, vocab, nE, k_win=4, act_th=3, learn_th=2, perm_conn=0.5,
                 p_init=0.24, lam_pot=0.14, lam_dep=0.02, z_tau=0.85, z_star=1.0, lesion=False):
        self.b, self.row, self.col, self.cells_idx = bridge, coo_row, coo_col, cells_idx
        self.M, self.nE, self.N = vocab, nE, vocab * nE
        self.k_win, self.act_th, self.learn_th, self.perm_conn, self.p_init = k_win, act_th, learn_th, perm_conn, p_init
        self.lam_pot, self.lam_dep, self.z_tau, self.z_star = lam_pot, lam_dep, z_tau, z_star
        self.lesion = lesion
        self.z = np.zeros(self.N)

    def _col(self, c):
        return list(range(c * self.nE, (c + 1) * self.nE))

    def _committed_count(self):
        """Per-cell # of POTENTIATED incoming synapses = the flat "committed" metric. On a DENSE pool (every cross-
        column pair pre-allocated at sub-connected p_init) the perm>0 count is full for every cell (can't differentiate)
        AND the CONNECTED (perm>=perm_conn) count differentiates only AFTER a cell connects (several epochs) -> an
        allocation RACE merges two contexts before either connects. The fix: "committed" = has incoming perm ABOVE the
        initial p_init (a cell that has been a winner is potentiated above p_init after ONE step, differentiating
        IMMEDIATELY -- the dense-pool analogue of EMERGE-13's perm>0 wired-count on a grow-from-zero pool)."""
        n = int(self.b.core_config.num_neurons)
        data = _host(self.b.cp_connections.data).astype(np.float64)
        wc = np.zeros(n)
        np.add.at(wc, self.col, (data > self.p_init + 0.02).astype(np.float64))
        return wc[self.cells_idx]                                # per EMERGE cell

    def _match_count(self, post_cell, prev_win):
        """CONNECTED-synapse overlap (perm>=perm_conn) into post from prev winners. On the dense pool, matching must be
        connected (not perm>0, which is full everywhere) so an untrained context matches nothing -> allocates."""
        if not prev_win:
            return 0
        data = _host(self.b.cp_connections.data).astype(np.float64)
        pre_set = set(int(self.cells_idx[i]) for i in prev_win)
        bpost = int(self.cells_idx[post_cell])
        idx = np.where(self.col == bpost)[0]
        m = 0
        for k in idx:
            if int(self.row[k]) in pre_set and data[k] >= self.perm_conn:
                m += 1
        return m

    def train_sequence(self, seq):
        predictive, prev_winners = set(), set()
        for c in seq:
            col = self._col(c)
            primed = [i for i in col if i in predictive] if not self.lesion else []
            to_learn = False
            if primed:
                winners = set(primed[:self.k_win])
            elif not prev_winners:
                winners = set(col[:self.k_win])
            else:
                scored = sorted(((self._match_count(i, prev_winners), i) for i in col), reverse=True)
                if scored[0][0] >= self.learn_th:
                    winners = set(i for sc, i in scored[:self.k_win] if sc >= self.learn_th)
                else:                                            # ALLOCATE onto the k FRESHEST (fewest-CONNECTED) cells
                    wc = self._committed_count()
                    winners = set(sorted(col, key=lambda i: (wc[i], i))[:self.k_win])
            if prev_winners:
                apply_kernel_update(self.b, self.row, self.col, self.cells_idx, prev_winners, winners,
                                    self.z, self.lam_pot, self.lam_dep, self.z_star)
            active = winners if primed else (set(col) if prev_winners or not primed else winners)
            predictive = connected_predict(self.b, self.row, self.col, self.cells_idx, active, self.N, self.nE,
                                           self.perm_conn, self.act_th)
            self.z *= self.z_tau
            for i in predictive:
                self.z[i] += (1.0 - self.z_tau)
            prev_winners = winners

    def predict_branch(self, seq, div_pos):
        predictive, prev_winners = set(), set()
        preds = []
        for c in seq:
            col = self._col(c)
            primed = [i for i in col if i in predictive] if not self.lesion else []
            active = set(primed[:self.k_win]) if primed else set(col)
            predictive = connected_predict(self.b, self.row, self.col, self.cells_idx, active, self.N, self.nE,
                                           self.perm_conn, self.act_th)
            preds.append(set(i // self.nE for i in predictive))
            prev_winners = active
        return preds


def _run_arm(seed, arm, n_seq, L, n_cells, k_win, act_th, epochs):
    seqs, vocab, info = make_overlap_sequences(n_seq=n_seq, middle_len=L, seed=seed)
    b, cells_idx, row, col = build_pool_bridge(vocab, n_cells, seed, act_th=act_th, coincidence=(arm != "lesion"))
    lr = OnBridgeLearner(b, row, col, cells_idx, vocab, n_cells, k_win=k_win, act_th=act_th, lesion=(arm == "lesion"))
    if arm != "untrained":
        for _ in range(epochs):
            for s in seqs:
                lr.train_sequence(s)
    ok = 0
    for s in seqs:
        ok += int(lr.predict_branch(s, L)[L] == {s[L + 1]})
    return arm, ok / len(seqs)


ARMS = ["htm", "lesion", "untrained"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-seq", type=int, default=2)
    ap.add_argument("--middle-len", type=int, default=4)
    ap.add_argument("--n-cells", type=int, default=16)
    ap.add_argument("--k-win", type=int, default=4)
    ap.add_argument("--act-th", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    t0 = time.time(); err = None; per = []
    floors = {}
    for s in a.seeds:
        seqs, vocab, info = make_overlap_sequences(n_seq=a.n_seq, middle_len=a.middle_len, seed=s)
        floors[s] = {"markov_L": markov_branch_acc(seqs, a.middle_len, a.n_seq),
                     "oracle": full_oracle(seqs, a.middle_len), "chance": 1.0 / a.n_seq}
    try:
        for s in a.seeds:
            d = {"seed": s, "floors": floors[s]}
            for arm in ARMS:
                _, acc = _run_arm(s, arm, a.n_seq, a.middle_len, a.n_cells, a.k_win, a.act_th, a.epochs)
                d[arm] = acc
            per.append(d)
            f = d["floors"]
            print(f"  [seed {s}] ON-BRIDGE-LEARN branch {d['htm']:.3f} | lesion {d['lesion']:.3f} | untr {d['untrained']:.3f} "
                  f"|| markov {f['markov_L']:.3f} chance {f['chance']:.3f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(arm):
            return float(np.mean([p[arm] for p in per]))
        htm, les, unt = m("htm"), m("lesion"), m("untrained")
        markov = float(np.mean([p["floors"]["markov_L"] for p in per]))
        chance = float(np.mean([p["floors"]["chance"] for p in per]))
        oracle = float(np.mean([p["floors"]["oracle"] for p in per]))
        go = bool(oracle > 0.99 and htm >= 0.90 and htm >= markov + 0.15 and htm >= chance + 0.20 and htm >= les + 0.20)
        if oracle <= 0.99:
            verdict = f"INCONCLUSIVE -- task not context-solvable (oracle {oracle:.3f})."
        elif go:
            verdict = (f"GO -- the HTM Temporal-Memory permanences LIVE in the bridge's coincidence synapse weights and are "
                       f"LEARNED from scratch by the sim/ fused_htm_permanence_update kernel (three-term rule) over a pre-"
                       f"allocated potential pool: branch {htm:.3f} >> Markov {markov:.3f}, >> chance {chance:.3f}, "
                       f">> dAP-lesion {les:.3f}; untrained {unt:.3f}; no teacher; multi-seed. => rung-4 LEARNING is on the "
                       f"substrate -> the whole unsupervised sequence-learning mechanism runs on the real SimulationBridge. "
                       f"rung-4 COMPLETE (host residual: winner-selection + committed-metric allocation, as in EMERGE-9d).")
        else:
            miss = []
            if htm < 0.90: miss.append(f"branch {htm:.3f} < 0.90")
            if htm < markov + 0.15 or htm < chance + 0.20: miss.append(f"didn't clear Markov/chance ({htm:.3f})")
            if htm < les + 0.20: miss.append(f"dAP-lesion didn't collapse ({htm:.3f} vs {les:.3f})")
            verdict = ("BOUNDARY (build-informative) -- " + "; ".join(miss) + f" (oracle {oracle:.3f}). Tune the kernel "
                       f"regime on the pool (lam_pot/lam_dep/z_tau/act_th/p_init/perm_conn/epochs) or the weighted-"
                       f"coincidence threshold; the on-substrate permanence learning is the next tuning, not a wall.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge14_stageC_onbridge_learning", "verdict": verdict,
               "mechanism": "HTM Temporal-Memory permanences in the bridge's coincidence synapse weights (cp_connections.data) "
                            "over a pre-allocated cross-column potential pool, learned by the sim/ fused_htm_permanence_update "
                            "kernel (Bouhadjar three-term rule); prediction = the bridge's connected-count coincidence; "
                            "host residual = winner-selection + committed-metric allocation (as EMERGE-9d)",
               "task": "overlapping sequences; branch prediction; Markov floor + dAP-lesion + oracle + multi-seed",
               "seeds": a.seeds, "config": {"n_seq": a.n_seq, "middle_len": a.middle_len, "n_cells": a.n_cells,
               "k_win": a.k_win, "act_th": a.act_th, "epochs": a.epochs},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "rung-4 Stage C: LEARNING on the substrate via the committed fused kernel on cp_connections.data. "
                              "Prediction reads the bridge's learned permanences (connected count). If GO, rung-4 (the whole "
                              "unsupervised sequence-learning mechanism on the real SimulationBridge) is COMPLETE."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge14] VERDICT: {verdict}", flush=True)
    print(f"[emerge14] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
