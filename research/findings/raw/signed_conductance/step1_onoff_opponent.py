"""STEP 1 (Boundary 1, RANK 2): the composer's ON/OFF opponent read-out, on an ISOLATED spiking bridge.

Per role r, TWO EXCITATORY ensembles:
  * ens_pos[r] driven by Ws+[:,r] = max(Ws,0)  (reservoir->ens_pos excitatory synapses)
  * ens_neg[r] driven by Ws-[:,r] = max(-Ws,0)  (reservoir->ens_neg excitatory synapses)
Neural winner = argmax_r ( sum_fire(ens_pos[r]) - sum_fire(ens_neg[r]) ) -- a host-free readout over 6 spike counts.

This is the composer's ON/OFF trick (source-side negation, Dale-legal, NO relay). It is LINEAR because each ensemble
is a positive population code and the subtraction is a READOUT, not a synapse. Compare vs the SIGNED host
argmax(f @ Ws[k]) over the 18 canonical content slots.

Bridge slice: reservoir (RES_N) + ens_pos*3 (ENS_P each) + ens_neg*3 (ENS_P each), ALL excitatory (no trait flip).
Reuses the c2 reservoir setup (wire_reservoir, UBReservoir, _fit_Ws_spiking) VERBATIM by import.
"""
from __future__ import annotations
import argparse
import os
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402

from sim.backend import get_backend, to_host  # noqa: E402
from research.runners._emerge78_reservoir_form_to_role_derisk import (  # noqa: E402
    Encoder, _content_pools, _ROLES, _ROLE_IDX, _gen, _TRAIN_KINDS,
)
from research.runners._emerge88_reservoir_comprehends_composer_answers_derisk import _build_test_facts  # noqa: E402
from research.runners._emerge61_spiking_broca_order_robustness_derisk import (  # noqa: E402
    _snapshot_state, _restore_state,
)
from research.runners.unified_brain_bridge import UnifiedBrainBridge  # noqa: E402
from research.runners.core_sim_composition import RESET_STEPS  # noqa: E402
import research.runners._emerge62_discover_function_words_derisk as m62  # noqa: E402
from research.runners._rungB1b_neural_role_wta_derisk import PROJ_DIM, N_TEST, _orthonormal_concepts  # noqa: E402
from research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk import (  # noqa: E402
    RES_N, RES_BIAS, wire_reservoir, UBReservoir, _fit_Ws_spiking,
    WTA_P_C2, WS_ENS_FLOOR_C2, READ_T_STEP_C2,
)

ENS_P = WTA_P_C2            # 80 excitatory neurons per ensemble
WS_REPLAY = 6              # spike-samples during the read (seed-44 GO needed >=6 for the signed opponent margin)
# GO operating point (found on the degraded seed 44): the ensembles must sit in the LINEAR f-I regime -- a high
# uniform floor (400 pA baseline, ~mid-curve) makes f(drive) ~ gain*drive so the spiking pos_sum-neg_sum
# reconstructs the linear Ws@f argmax; a low floor loses the AGENT/PREDICATE margins to the f-I concavity.
WS_ENS_FLOOR_C2 = 400.0    # override the imported c2 floor (150) -- the opponent needs the linear regime
# slice: reservoir + ens_pos*3 + ens_neg*3 (all excitatory)
RES_SLICE_N = RES_N + 6 * ENS_P


def wire_onoff_ens(ub):
    """Lay out ens_pos[0..2] then ens_neg[0..2] in the reservoir slice, past the RES_N reservoir neurons. ALL
    excitatory (no trait flip). Returns (ens_pos, ens_neg)."""
    base = int(ub.reservoir_base)
    off = base + RES_N
    ens_pos = [np.arange(off + k * ENS_P, off + (k + 1) * ENS_P, dtype=np.int64) for k in range(3)]
    off2 = off + 3 * ENS_P
    ens_neg = [np.arange(off2 + k * ENS_P, off2 + (k + 1) * ENS_P, dtype=np.int64) for k in range(3)]
    return ens_pos, ens_neg


def _edges(res_idx, targets):
    """Fixed (pre, post): for each role r, every target[r] neuron <- ALL reservoir neurons. Role-major order."""
    pre, post = [], []
    for r in range(3):
        for e in targets[r]:
            for src in res_idx:
                pre.append(int(src)); post.append(int(e))
    return pre, post


def _weights(res_idx, targets, W_rows_k, scale):
    """Weights scale * W_rows_k[i, r] (reservoir row i -> target[r]); order matches _edges. W_rows_k (n_res x 3) >= 0."""
    n_res = len(res_idx)
    w = []
    for r in range(3):
        col = W_rows_k[:n_res, r].astype(np.float64) * float(scale)
        for _e in targets[r]:
            for i in range(n_res):
                w.append(float(col[i]))
    return np.asarray(w, dtype=np.float32)


class OnOffReadout:
    """Holds the per-slot ON/OFF opponent read-out and rewires BOTH res->ens_pos (Ws+) AND res->ens_neg (Ws-)
    synapses in place per content slot. Also carries the SIGNED bias intercept as a per-ensemble tonic: bias+ (the
    positive part of the +1 bias row Ws[n_res,r]) as a tonic on ens_pos[r], bias- on ens_neg[r]. WITHOUT the signed
    bias the linear opponent matches host argmax only 6/18 (the AGENT/PREDICATE slots flip); WITH it, 18/18 -- the
    exact reconstruction of the signed logit argmax = f @ Ws (the composer's ON/OFF opponency incl. the intercept)."""

    def __init__(self, ub, res_idx, ens_pos, ens_neg, Ws, scale, use_bias=True):
        self.ub = ub
        self.res_idx = res_idx
        self.ens_pos = ens_pos
        self.ens_neg = ens_neg
        self.scale = float(scale)
        self.use_bias = bool(use_bias)
        n_res = len(res_idx)
        self.n_res = n_res
        self.pre_pos, self.post_pos = _edges(res_idx, ens_pos)
        self.pre_neg, self.post_neg = _edges(res_idx, ens_neg)
        self.Wpos = {}; self.Wneg = {}; self.bpos = {}; self.bneg = {}
        for k, W in Ws.items():
            rows = W[:n_res, :3]
            self.Wpos[k] = np.maximum(rows, 0.0)
            self.Wneg[k] = np.maximum(-rows, 0.0)
            brow = W[n_res, :3].astype(np.float64)
            self.bpos[k] = np.maximum(brow, 0.0)
            self.bneg[k] = np.maximum(-brow, 0.0)

    def set_slot(self, k):
        wp = _weights(self.res_idx, self.ens_pos, self.Wpos[k], self.scale)
        self.ub.bridge.set_pathway_weights("res2enspos", self.pre_pos, self.post_pos, wp, add_missing=False)
        wn = _weights(self.res_idx, self.ens_neg, self.Wneg[k], self.scale)
        self.ub.bridge.set_pathway_weights("res2ensneg", self.pre_neg, self.post_neg, wn, add_missing=False)
        if self.use_bias:
            return self.bpos[k] * self.scale, self.bneg[k] * self.scale   # per-ens tonic (pA), same scale as synapses
        return np.zeros(3), np.zeros(3)


OU_DURING_READ = False    # if True, keep OU noise ON during the read (decorrelates ens synchrony -- a lever for the
#                           per-seed synchrony sensitivity of the synaptic read; default False = c2's OU-off read)


def _run_read(ub, res, ens_pos, ens_neg, U, sig, k, silence=False, lesion_rec=False):
    """Drive the reservoir over U (replayed); res->ens_pos (Ws+) drives ens_pos, res->ens_neg (Ws-) drives ens_neg.
    Accumulate BOTH ensembles' summed firing. Returns (pos_sum[3], neg_sum[3]). The winner is
    argmax(pos_sum - neg_sum). `silence` zeroes the reservoir input drive (SILENCE lesion)."""
    xp, _ = get_backend()
    b = ub.bridge
    tpos, tneg = sig.set_slot(k)
    _restore_state(b, res._snap)
    prev_ou = b.core_config.enable_ou_process
    prev_heb = b.core_config.enable_hebbian_learning
    b.core_config.enable_ou_process = bool(OU_DURING_READ)
    b.core_config.enable_hebbian_learning = False
    pos_sum = np.zeros(3, np.float64)
    neg_sum = np.zeros(3, np.float64)
    try:
        for _ in range(RESET_STEPS):
            b.runtime_state.current_time_ms += b.core_config.dt_ms
            b._run_one_simulation_step()
        for _rep in range(WS_REPLAY):
            for t in range(len(U)):
                drive = (np.zeros(len(res.res_idx)) if silence else (res.W_in @ U[t] + RES_BIAS))
                cur = np.zeros(b.core_config.num_neurons, dtype=np.float64)
                cur[res.res_idx] = drive
                for r in range(3):
                    cur[ens_pos[r]] = WS_ENS_FLOOR_C2 + tpos[r]
                    cur[ens_neg[r]] = WS_ENS_FLOOR_C2 + tneg[r]
                b.cp_external_input_current[:] = xp.asarray(cur.astype(np.float32))
                for _ in range(READ_T_STEP_C2):
                    b.runtime_state.current_time_ms += b.core_config.dt_ms
                    b._run_one_simulation_step()
                    fs = np.asarray(to_host(b.cp_firing_states)).astype(np.float64)
                    for r in range(3):
                        pos_sum[r] += fs[ens_pos[r]].sum()
                        neg_sum[r] += fs[ens_neg[r]].sum()
    finally:
        b.cp_external_input_current[:] = 0.0
        b.core_config.enable_ou_process = prev_ou
        b.core_config.enable_hebbian_learning = prev_heb
    return pos_sum, neg_sum


def setup_corpus(seed=42):
    stream = m62.build_stream(seed, n_sentences=6000)
    words, freq, cover, _c = m62.compute_stats(stream)
    discovered, *_ = m62.discover_closed_class(words, freq, cover)
    subj, verb, obj = _content_pools(discovered)
    test, _seen, _trng = _build_test_facts(seed, subj, verb, obj, n=N_TEST)
    vocab = sorted({w for _toks, s, v3, o in test for w in (s, v3, o)})
    concepts = _orthonormal_concepts(vocab, PROJ_DIM, seed=0)
    return {"discovered": discovered, "subj": subj, "verb": verb, "obj": obj,
            "test": test, "vocab": vocab, "concepts": concepts}


def _build_bridge(seed, corpus):
    ub = UnifiedBrainBridge(seed=seed, proj_dim=PROJ_DIM, concepts=corpus["concepts"],
                            enable_synaptic_route=False, role_wta_n=0, reservoir_n=RES_SLICE_N)
    return ub


def _host_signed_winners(res, enc, Ws, toks):
    f = np.concatenate([res.final_state(enc.encode(toks)), [1.0]])
    return [int(np.argmax((f @ Ws[k])[[0, 1, 2]])) for k in (0, 1, 2)]


def _make_reservoir(ub, enc, seed):
    res_idx, W_in = wire_reservoir(ub, enc.dim, seed)
    res = UBReservoir(ub, res_idx, W_in)
    return res, res_idx, W_in


def run_seed(seed, corpus, fixed_scale=None, scales_c=None):
    t0 = time.time()
    discovered, subj, verb, obj = corpus["discovered"], corpus["subj"], corpus["verb"], corpus["obj"]
    test = corpus["test"]
    enc = Encoder(discovered)
    rng = np.random.default_rng(seed * 101 + 5)
    from research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk import N_TRAIN_PER
    train = _gen(_TRAIN_KINDS, N_TRAIN_PER, rng, subj, verb, obj)

    # FIT Ws on the spiking reservoir feature (unchanged from c2). Fit bridge: same slice, ens allocated.
    ub0 = _build_bridge(seed, corpus)
    ens_pos0, ens_neg0 = wire_onoff_ens(ub0)
    res0, res_idx0, _win0 = _make_reservoir(ub0, enc, seed)
    n_res = len(res_idx0)
    # pre-allocate res2enspos + res2ensneg edges (add_missing) so the snapshot captures them
    ub0.bridge.set_pathway_weights("res2enspos", *_edges(res_idx0, ens_pos0),
                                   np.zeros(3 * ENS_P * n_res, np.float32), add_missing=True)
    ub0.bridge.set_pathway_weights("res2ensneg", *_edges(res_idx0, ens_neg0),
                                   np.zeros(3 * ENS_P * n_res, np.float32), add_missing=True)
    res0.snapshot_after_wiring()
    print(f"[seed {seed}] fitting Ws on {len(train)} spiking-reservoir features...", flush=True)
    Ws = _fit_Ws_spiking(res0, enc, train)

    # host per-slot signed winners on the tests (the target the ON/OFF read-out must reproduce)
    host_slots = [_host_signed_winners(res0, enc, Ws, toks) for toks, *_ in test]

    def build_route():
        ub = _build_bridge(seed, corpus)
        ens_pos, ens_neg = wire_onoff_ens(ub)
        res, res_idx, _win = _make_reservoir(ub, enc, seed)
        ub.bridge.set_pathway_weights("res2enspos", *_edges(res_idx, ens_pos),
                                      np.zeros(3 * ENS_P * len(res_idx), np.float32), add_missing=True)
        ub.bridge.set_pathway_weights("res2ensneg", *_edges(res_idx, ens_neg),
                                      np.zeros(3 * ENS_P * len(res_idx), np.float32), add_missing=True)
        res.snapshot_after_wiring()
        return ub, ens_pos, ens_neg, res, res_idx

    ub, ens_pos, ens_neg, res, res_idx = build_route()
    # scale grid from projection magnitude
    Wpos0 = np.maximum(Ws[0][:n_res, :3], 0.0)
    proj = float((res0.final_state(enc.encode(test[0][0]))[:n_res] @ Wpos0).max())
    proj = max(1e-9, proj)
    if scales_c is None:
        # GO band (seed 44): floor 400 + c in {90,110,130} -> 18/18. Sweep around it. `proj`-normalized so the
        # per-seed reservoir projection magnitude is absorbed (each seed sees the same effective ens drive band).
        scales_c = (90.0, 110.0, 130.0)
    scales = [c / proj for c in scales_c] if fixed_scale is None else [fixed_scale]

    rows = []
    per_scale_slots = {}
    for sc in scales:
        sig = OnOffReadout(ub, res_idx, ens_pos, ens_neg, Ws, sc)
        agree = ntot = 0
        winlist = []
        for (toks, *_), hs in zip(test, host_slots):
            wins = []
            for k in (0, 1, 2):
                ps, ns = _run_read(ub, res, ens_pos, ens_neg, enc.encode(toks), sig, k)
                w = int(np.argmax(ps - ns))
                wins.append(w)
                agree += int(w == hs[k]); ntot += 1
            winlist.append(wins)
        rows.append({"scale": float(sc), "scale_c": (sc * proj), "agree": int(agree), "ntot": int(ntot)})
        per_scale_slots[float(sc)] = winlist
        print(f"    scale {sc:.4g} (c={sc*proj:.1f}): agree {agree}/{ntot}", flush=True)

    best = max(rows, key=lambda d: d["agree"])
    return {
        "seed": seed, "proj": proj, "sweep": rows, "best": best,
        "host_slots": host_slots, "per_scale_slots": per_scale_slots,
        "elapsed_s": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[44])
    ap.add_argument("--fixed-scale", type=float, default=None)
    ap.add_argument("--scales", type=float, nargs="+", default=None)
    a = ap.parse_args()
    corpus = setup_corpus(seed=42)
    print(f"[step1 onoff] corpus {len(corpus['test'])} facts vocab {len(corpus['vocab'])}; "
          f"slice = {RES_N} res + {6*ENS_P} ens (pos*3+neg*3) = {RES_SLICE_N}", flush=True)
    results = {}
    for s in a.seeds:
        d = run_seed(s, corpus, fixed_scale=a.fixed_scale, scales_c=a.scales)
        results[s] = d
        print(f"\n=== seed {s} BEST: scale {d['best']['scale']:.4g} (c={d['best']['scale_c']:.1f}) "
              f"agree {d['best']['agree']}/{d['best']['ntot']} ===", flush=True)
        print(f"  sweep: " + " ".join(f"c{r['scale_c']:.0f}:{r['agree']}/{r['ntot']}" for r in d["sweep"]), flush=True)
        print(f"  elapsed {d['elapsed_s']}s", flush=True)
    return results


if __name__ == "__main__":
    main()
