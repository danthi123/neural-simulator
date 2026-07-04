"""RUNG B-1c signed (+/-) synaptic read-out DE-RISK (ISOLATED bridge; NO WTA/composer/unified-bridge close-out).

De-risks the RESIDUAL BOUNDARY of the RUNG B-1c c2 close-out: does a SIGNED (+/-) synaptic read-out -- negative `Ws`
rows delivered through an INHIBITORY RELAY, instead of the argmax-preserving Dale OFFSET -- reproduce the SIGNED host
`argmax(f @ Ws[k])` well enough to (a) resolve the degraded seed-44 canonical margin AND (b) read a NON-CANONICAL
construction structurally (position != role)?

THE MECHANISM. Split `Ws[k]` (reservoir rows only) into `Ws+ = max(Ws,0)` and `Ws- = max(-Ws,0)`. Deliver:
  * `Ws+` as EXCITATORY synapses reservoir -> the 3 role ensembles ens[r]        (adds  Ws+ @ firing to ens[r])
  * `Ws-` as EXCITATORY synapses reservoir -> a per-role INHIBITORY RELAY pool relay[r] (trait=1); relay[r] -> ens[r]
    with a fixed inhibitory weight (the trait-1 firing routes through g_i).       (subtracts ~Ws- @ firing from ens[r])
So each ensemble's NET drive ~ (Ws+ - Ws-) @ firing = Ws @ firing = the SIGNED read-out, entirely on the substrate.

ISOLATED: a bridge with ONLY the spiking reservoir + the 3 role ensembles + the signed projection (+ the relay). NO
mutual-inhibition WTA competition, NO composer, NO gates. The "winner" is a NEURAL read: argmax over the 3 ensembles'
summed firing -- exactly what the c2 close-out reads, minus the downstream binding. That isolates the READ-OUT question.

Reuses the c2 runner's reservoir setup (UBReservoir statistics, wire_reservoir, _fit_Ws_spiking, the encoder/corpus,
Hebbian+OU-off read window) VERBATIM by import; only the read-out projection is new (signed vs positive).

Run: SIM_BACKEND=numpy PYTHONPATH=<repo> python b1c_signed_readout_derisk.py --seeds 42 43 44
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
# reuse the c2 reservoir constants/wiring VERBATIM (the reservoir + its Ws fit are unchanged; only the read-out differs)
from research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk import (  # noqa: E402
    RES_N, RES_BIAS, RES_T_STEP, wire_reservoir, UBReservoir, _fit_Ws_spiking,
    WTA_P_C2, WS_ENS_FLOOR_C2, READ_T_STEP_C2,
    _objrel_test_fact,
)

# ── the isolated read-out ensembles + inhibitory relay (an additive slice on `reservoir_n`) ──────────────────────
# We fold BOTH the 3 role ensembles AND the per-role inhibitory relay into the reservoir slice (past the RES_N
# reservoir neurons), so we need NO WTA slice at all -- the read-out is isolated (no mutual inhibition). The relay
# neurons are flipped to trait=1 (inhibitory), exactly like the WTA inh pool: their firing routes through g_i.
ENS_P = WTA_P_C2          # 80 excitatory neurons per role ensemble (the CRUX resolution; 3 ensembles)
RELAY_P = 40             # inhibitory relay neurons per role (trait=1); receives Ws- from the reservoir, inhibits ens[r]
WS_REPLAY = 3            # sentence replays during the read (more spike samples; the CRUX law-of-large-numbers lever)
RELAY_IE_W = 12.0        # relay[r] -> ens[r] inhibitory weight (per edge); tuned so the subtraction balances Ws+
RES_SLICE_N = RES_N + 3 * ENS_P + 3 * RELAY_P   # reservoir + ens + relay all in the `reservoir_n` additive slice


def wire_signed_readout(ub, res_idx, seed):
    """Lay out ens[0..2] (excitatory) + relay[0..2] (inhibitory, trait=1) in the reservoir slice, past the RES_N
    reservoir neurons. Returns (ens, relay). Flips the relay pools to trait=1 (their firing routes through g_i)."""
    base = int(ub.reservoir_base)
    off = base + RES_N
    ens = [np.arange(off + k * ENS_P, off + (k + 1) * ENS_P, dtype=np.int64) for k in range(3)]
    off2 = off + 3 * ENS_P
    relay = [np.arange(off2 + k * RELAY_P, off2 + (k + 1) * RELAY_P, dtype=np.int64) for k in range(3)]
    all_relay = np.concatenate(relay)
    ub.bridge.cp_traits[all_relay] = 1
    ub.bridge._cached_inhibitory_mask = None
    return ens, relay


def _wire_relay_ie(ub, relay, ens):
    """Wire relay[r] -> ens[r] inhibitory synapses (fixed weight RELAY_IE_W). Pre-allocates the edges. Returns
    (pre, post) for later lesion."""
    pre, post = [], []
    for r in range(3):
        for a in relay[r]:
            for b in ens[r]:
                pre.append(int(a)); post.append(int(b))
    ub.bridge.set_pathway_weights("relay_i2e", pre, post,
                                  np.full(len(pre), RELAY_IE_W, dtype=np.float32), add_missing=True)
    return pre, post


def _wpos_edges(res_idx, targets):
    """Fixed (pre, post) edges: for each role r, every target[r] neuron <- ALL reservoir neurons. Order role-major."""
    pre, post = [], []
    for r in range(3):
        for e in targets[r]:
            for src in res_idx:
                pre.append(int(src)); post.append(int(e))
    return pre, post


def _wpos_weights(res_idx, targets, W_rows_k, scale):
    """Weights for `scale * W_rows_k[i, r]` (reservoir row i -> target[r]); order matches `_wpos_edges`. `W_rows_k`
    is (n_res x 3) >= 0 (either Ws+ for ens or Ws- for relay)."""
    n_res = len(res_idx)
    w = []
    for r in range(3):
        col = W_rows_k[:n_res, r].astype(np.float64) * float(scale)
        for _e in targets[r]:
            for i in range(n_res):
                w.append(float(col[i]))
    return np.asarray(w, dtype=np.float32)


class SignedReadout:
    """Holds the per-slot SIGNED read-out and rewires BOTH the res->ens (Ws+) AND the res->relay (Ws-) synapses in
    place per content slot. The signed drive = ens gets Ws+ excitation, relay gets Ws- excitation and inhibits ens,
    so net ens drive ~ (Ws+ - Ws-) @ firing = the SIGNED Ws @ firing.

    `positive=True` reproduces the c2 POSITIVE baseline instead: ens gets the Dale-shifted Ws_shifted = Ws - Ws.min()
    (all >= 0) as excitation, relay is driven to ZERO (no inhibition) -- so the same substrate/ens/relay wiring runs
    both the signed and the positive read-out (a fair on-substrate comparison)."""

    def __init__(self, ub, res_idx, ens, relay, Ws, scale, positive=False, bias=False):
        self.ub = ub
        self.res_idx = res_idx
        self.ens = ens
        self.relay = relay
        self.Ws = Ws                                  # {slot: (feat_dim x 5) ridge read-out}; cols 0/1/2 = roles
        self.scale = float(scale)
        self.positive = bool(positive)
        self.bias = bool(bias)                        # carry the per-role bias intercept as a per-ens tonic?
        n_res = len(res_idx)
        self.n_res = n_res
        self.pre_ens, self.post_ens = _wpos_edges(res_idx, ens)
        self.pre_rel, self.post_rel = _wpos_edges(res_idx, relay)
        # precompute per-slot signed / positive row decompositions (reservoir rows only, cols 0/1/2)
        self.Wpos = {}; self.Wneg = {}; self.Wshift = {}; self.role_bias = {}
        for k, W in Ws.items():
            rows = W[:n_res, :3]
            self.Wpos[k] = np.maximum(rows, 0.0)
            self.Wneg[k] = np.maximum(-rows, 0.0)
            self.Wshift[k] = (W[:, :3] - W[:, :3].min())[:n_res, :]    # Dale offset (reservoir rows)
            self.role_bias[k] = W[n_res, :3].astype(np.float64)        # +1 bias row intercept (per role)

    def set_slot(self, k):
        """Overwrite the read-out synapses for slot k; return the per-role ens tonic bias (0 unless self.bias)."""
        if self.positive:
            wpos = _wpos_weights(self.res_idx, self.ens, self.Wshift[k], self.scale)
            self.ub.bridge.set_pathway_weights("res2ens", self.pre_ens, self.post_ens, wpos, add_missing=False)
            wneg = np.zeros(len(self.pre_rel), dtype=np.float32)      # positive read-out: relay OFF
            self.ub.bridge.set_pathway_weights("res2relay", self.pre_rel, self.post_rel, wneg, add_missing=False)
        else:
            wpos = _wpos_weights(self.res_idx, self.ens, self.Wpos[k], self.scale)
            self.ub.bridge.set_pathway_weights("res2ens", self.pre_ens, self.post_ens, wpos, add_missing=False)
            wneg = _wpos_weights(self.res_idx, self.relay, self.Wneg[k], self.scale)
            self.ub.bridge.set_pathway_weights("res2relay", self.pre_rel, self.post_rel, wneg, add_missing=False)
        if not self.bias:
            return np.zeros(3)
        # for the signed read-out the bias intercept is signed too; carry (bias - bias.min()) argmax-preservingly as
        # a per-ens tonic (only used in the "bias-on" probe; the load-bearing config drops it).
        rb = self.role_bias[k]
        return (rb - rb.min()) * self.scale


def _run_read(ub, res, ens, relay, U, sig, k, silence=False, lesion_rec=False):
    """Drive the reservoir over `U` (replayed) with slot-k's read-out wired; the res2ens (Ws+) synapses drive the
    ensembles, the res2relay (Ws-) synapses drive the inhibitory relay which inhibits the ensembles. Accumulate the
    3 ensembles' summed firing (the read-out feature). The winner is argmax over that. `silence` zeroes the reservoir
    input (the SILENCE lesion). `lesion_rec` zeroes the reservoir recurrence (the RECURRENCE lesion). Returns
    ens_sum[3]."""
    xp, _ = get_backend()
    b = ub.bridge
    role_bias = sig.set_slot(k)
    _restore_state(b, res._snap)
    prev_ou = b.core_config.enable_ou_process
    prev_heb = b.core_config.enable_hebbian_learning
    b.core_config.enable_ou_process = False
    b.core_config.enable_hebbian_learning = False
    rb = np.asarray(role_bias, dtype=np.float64)
    ens_sum = np.zeros(3, np.float64)
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
                    cur[ens[r]] = rb[r] + WS_ENS_FLOOR_C2
                b.cp_external_input_current[:] = xp.asarray(cur.astype(np.float32))
                for _ in range(READ_T_STEP_C2):
                    b.runtime_state.current_time_ms += b.core_config.dt_ms
                    b._run_one_simulation_step()
                    fs = np.asarray(to_host(b.cp_firing_states)).astype(np.float64)
                    for r in range(3):
                        ens_sum[r] += fs[ens[r]].sum()
    finally:
        b.cp_external_input_current[:] = 0.0
        b.core_config.enable_ou_process = prev_ou
        b.core_config.enable_hebbian_learning = prev_heb
    return ens_sum


def _lesion_recurrence(ub, res_idx, seed):
    """Zero the reservoir recurrence (reservoir->reservoir edges) read DIRECTLY from the CSR -> the LSM loses its
    recurrent integration (feedforward W_in only). Robust to how `wire_reservoir` enumerated the edges: the previous
    version re-derived them with its own RNG, which diverged from the actual CSR (8181/9068 pairs not found). Every
    reservoir->reservoir synapse IS the recurrence (res->ens/res->relay are separate pathways), so we zero exactly the
    res->res entries the CSR holds. Returns restore() (this runner uses a fresh bridge, so restore is unused)."""
    b = ub.bridge
    csr = b.cp_connections
    indptr = np.asarray(to_host(csr.indptr)).astype(np.int64)
    indices = np.asarray(to_host(csr.indices)).astype(np.int64)
    data = np.asarray(to_host(csr.data)).astype(np.float64)
    res_set = set(int(x) for x in res_idx)
    pre, post, wsave = [], [], []
    for a in res_idx:                                  # CSR is pre-major: row a = pre a's outgoing edges
        a = int(a)
        for off in range(int(indptr[a]), int(indptr[a + 1])):
            p = int(indices[off])
            if p in res_set:                           # reservoir -> reservoir = the recurrence
                pre.append(a); post.append(p); wsave.append(float(data[off]))
    if pre:
        b.set_pathway_weights("reservoir_rec", pre, post, np.zeros(len(pre), dtype=np.float32), add_missing=False)

    def restore():
        if pre:
            b.set_pathway_weights("reservoir_rec", pre, post, np.asarray(wsave, dtype=np.float32), add_missing=False)
    return restore


# ── corpus (canonical + one non-canonical objrel; reuse the c2 corpus builder logic) ─────────────────────────────
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
    """Fresh isolated bridge: reservoir slice holds reservoir + ens + relay. NO role_wta slice, NO composer bind used.
    (The composer is still constructed by UnifiedBrainBridge but we never drive its bind -- this is read-out only.)"""
    ub = UnifiedBrainBridge(seed=seed, proj_dim=PROJ_DIM, concepts=corpus["concepts"],
                            enable_synaptic_route=False, role_wta_n=0, reservoir_n=RES_SLICE_N)
    return ub


def _host_signed_winners(res, enc, Ws, toks):
    """Host SIGNED argmax(f @ Ws[k]) per content slot (the target the signed synaptic read-out must reproduce)."""
    f = np.concatenate([res.final_state(enc.encode(toks)), [1.0]])
    return [int(np.argmax((f @ Ws[k])[[0, 1, 2]])) for k in (0, 1, 2)]


def _make_reservoir(ub, enc, seed):
    res_idx, W_in = wire_reservoir(ub, enc.dim, seed)
    res = UBReservoir(ub, res_idx, W_in)
    return res, res_idx, W_in


def run_seed(seed, corpus, fixed_scale=None):
    t0 = time.time()
    discovered, subj, verb, obj = corpus["discovered"], corpus["subj"], corpus["verb"], corpus["obj"]
    test = corpus["test"]
    enc = Encoder(discovered)
    rng = np.random.default_rng(seed * 101 + 5)
    from research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk import N_TRAIN_PER
    train = _gen(_TRAIN_KINDS, N_TRAIN_PER, rng, subj, verb, obj)

    # FIT Ws on the spiking reservoir feature (unchanged from c2).
    ub0 = _build_bridge(seed, corpus)
    ens0, relay0 = wire_signed_readout(ub0, np.arange(int(ub0.reservoir_base), int(ub0.reservoir_base) + RES_N), seed)
    res0, res_idx0, _win0 = _make_reservoir(ub0, enc, seed)
    _wire_relay_ie(ub0, relay0, ens0)
    # pre-allocate the res2ens + res2relay edges (add_missing) with slot-0 so the snapshot captures them
    from research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk import _fit_Ws_spiking as _fit
    print(f"[seed {seed}] fitting Ws on {len(train)} spiking-reservoir features...", flush=True)
    res0.snapshot_after_wiring()
    Ws = _fit(res0, enc, train)
    n_res = len(res_idx0)
    Wpos0 = np.maximum(Ws[0][:n_res, :3], 0.0)
    Wneg0 = np.maximum(-Ws[0][:n_res, :3], 0.0)
    ub0.bridge.set_pathway_weights("res2ens", *_wpos_edges(res_idx0, ens0),
                                   _wpos_weights(res_idx0, ens0, Wpos0, 1.0), add_missing=True)
    ub0.bridge.set_pathway_weights("res2relay", *_wpos_edges(res_idx0, relay0),
                                   _wpos_weights(res_idx0, relay0, Wneg0, 1.0), add_missing=True)
    res0.snapshot_after_wiring()

    # host per-slot signed winners on the 6 tests (the target both read-outs must reproduce)
    host_slots = [_host_signed_winners(res0, enc, Ws, toks) for toks, *_ in test]

    def build_route(positive):
        ub = _build_bridge(seed, corpus)
        ens, relay = wire_signed_readout(ub, np.arange(int(ub.reservoir_base), int(ub.reservoir_base) + RES_N), seed)
        res, res_idx, _win = _make_reservoir(ub, enc, seed)
        _wire_relay_ie(ub, relay, ens)
        # allocate both read-out edge sets
        ub.bridge.set_pathway_weights("res2ens", *_wpos_edges(res_idx, ens),
                                      np.zeros(3 * ENS_P * len(res_idx), np.float32), add_missing=True)
        ub.bridge.set_pathway_weights("res2relay", *_wpos_edges(res_idx, relay),
                                      np.zeros(3 * RELAY_P * len(res_idx), np.float32), add_missing=True)
        res.snapshot_after_wiring()
        return ub, ens, relay, res, res_idx

    def sweep_scale(positive):
        ub, ens, relay, res, res_idx = build_route(positive)
        proj = float((res0.final_state(enc.encode(test[0][0]))[:n_res] @ Wpos0).max())
        proj = max(1e-9, proj)
        scales = [c / proj for c in (40.0, 60.0, 90.0, 130.0, 180.0, 240.0, 320.0)]
        rows = []
        for sc in scales:
            sig = SignedReadout(ub, res_idx, ens, relay, Ws, sc, positive=positive)
            agree = ntot = 0
            for (toks, *_), hs in zip(test, host_slots):
                for k in (0, 1, 2):
                    es = _run_read(ub, res, ens, relay, enc.encode(toks), sig, k)
                    agree += int(int(np.argmax(es)) == hs[k]); ntot += 1
            rows.append({"scale": float(sc), "agree": int(agree), "ntot": int(ntot)})
        return rows

    def route_recall(positive, scale, silence=False, lesion_rec=False, objrel=None):
        """Bind the 6 canonical facts (or a supplied objrel-augmented set) via the SIGNED/positive read-out; return
        (route_correct, per-fact winners). We do NOT run the composer -- recall here = the read-out winner matches
        the host SIGNED winner per content slot (the isolated read-out claim). Returns dict."""
        ub, ens, relay, res, res_idx = build_route(positive)
        if lesion_rec:
            _lesion_recurrence(ub, res_idx, seed)  # zero recurrence on THIS bridge (restore not needed; fresh bridge)
        sig = SignedReadout(ub, res_idx, ens, relay, Ws, scale, positive=positive)
        facts = objrel if objrel is not None else test
        # host signed winners for THIS fact set (objrel differs)
        hsets = [_host_signed_winners(res0, enc, Ws, toks) for toks, *_ in facts]
        slot_ok = 0; slot_tot = 0; per = []
        for (toks, *_), hs in zip(facts, hsets):
            wins = []
            for k in (0, 1, 2):
                es = _run_read(ub, res, ens, relay, enc.encode(toks), sig, k, silence=silence)
                w = int(np.argmax(es)); wins.append(w)
                slot_ok += int(w == hs[k]); slot_tot += 1
            per.append({"toks": toks, "host": hs, "syn": wins})
        return {"slot_ok": slot_ok, "slot_tot": slot_tot, "per": per}

    # ── (a) SEED-44 CANONICAL: signed vs positive, host-agree over the 18 content slots ──────────────────────────
    scale = fixed_scale
    sweep_signed = sweep_scale(positive=False)
    sweep_pos = sweep_scale(positive=True)
    if scale is None:
        max_ag = max(d["agree"] for d in sweep_signed)
        scale = [d for d in sweep_signed if d["agree"] == max_ag][0]["scale"]
    max_ag_pos = max(d["agree"] for d in sweep_pos)
    scale_pos = [d for d in sweep_pos if d["agree"] == max_ag_pos][0]["scale"]

    canon_signed = route_recall(positive=False, scale=scale)
    canon_pos = route_recall(positive=True, scale=scale_pos)

    # ── (b) NON-CANONICAL objrel: signed vs positive on slot-0 (THEME, position != role) + recurrence lesion ─────
    keys = {(s, v3) for _t, s, v3, _o in test}
    objrel_fact = _objrel_test_fact(seed, subj, verb, obj, keys)   # ["the", PAT, "that", "the", AGT, V3]
    objrel_set = [objrel_fact]
    oc_signed = route_recall(positive=False, scale=scale, objrel=objrel_set)
    oc_pos = route_recall(positive=True, scale=scale_pos, objrel=objrel_set)
    oc_signed_reclesion = route_recall(positive=False, scale=scale, objrel=objrel_set, lesion_rec=True)

    return {
        "seed": seed,
        "scale_signed": scale, "scale_pos": scale_pos,
        "sweep_signed": sweep_signed, "sweep_pos": sweep_pos,
        "canon_signed": canon_signed, "canon_pos": canon_pos,
        "objrel_fact": objrel_fact,
        "oc_signed": oc_signed, "oc_pos": oc_pos, "oc_signed_reclesion": oc_signed_reclesion,
        "elapsed_s": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[44])
    ap.add_argument("--fixed-scale", type=float, default=None)
    a = ap.parse_args()
    corpus = setup_corpus(seed=42)
    print(f"[b1c-signed] corpus {len(corpus['test'])} facts vocab {len(corpus['vocab'])}; "
          f"reservoir slice = {RES_N} res + {3*ENS_P} ens + {3*RELAY_P} relay = {RES_SLICE_N}", flush=True)
    for s in a.seeds:
        d = run_seed(s, corpus, fixed_scale=a.fixed_scale)
        cs, cp_ = d["canon_signed"], d["canon_pos"]
        print(f"\n=== seed {s} (scale signed {d['scale_signed']:.4g} / pos {d['scale_pos']:.4g}) ===", flush=True)
        print(f"  sweep signed: " + " ".join(f"{r['scale']:.3g}:{r['agree']}/{r['ntot']}" for r in d["sweep_signed"]),
              flush=True)
        print(f"  sweep pos   : " + " ".join(f"{r['scale']:.3g}:{r['agree']}/{r['ntot']}" for r in d["sweep_pos"]),
              flush=True)
        print(f"  (a) CANONICAL host-agree over 18 slots: SIGNED {cs['slot_ok']}/{cs['slot_tot']}  "
              f"POSITIVE {cp_['slot_ok']}/{cp_['slot_tot']}", flush=True)
        oc_s, oc_p, oc_l = d["oc_signed"], d["oc_pos"], d["oc_signed_reclesion"]
        print(f"  (b) OBJREL {d['objrel_fact'][0]}", flush=True)
        print(f"      host  slot-winners: {oc_s['per'][0]['host']}  (slot0 should be THEME=2)", flush=True)
        print(f"      SIGNED   syn winners: {oc_s['per'][0]['syn']}  (agree {oc_s['slot_ok']}/3)", flush=True)
        print(f"      POSITIVE syn winners: {oc_p['per'][0]['syn']}  (agree {oc_p['slot_ok']}/3)", flush=True)
        print(f"      SIGNED+RECURRENCE-LESION syn winners: {oc_l['per'][0]['syn']}  (agree {oc_l['slot_ok']}/3)",
              flush=True)
        print(f"  elapsed {d['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
