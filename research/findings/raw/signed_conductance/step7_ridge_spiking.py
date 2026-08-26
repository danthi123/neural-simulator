"""ISOLATE further: the RIDGE Ws solves objrel 100% via LINEAR argmax (signed AND positive-shifted). The c3 DELTA rule via
the SPIKING WTA gets objrel 0%. Two candidate walls: (a) the delta rule LEARNS position-weights (vs the ridge's structural
weights); (b) the SPIKING WTA deploy corrupts the structural read. This test deploys the RIDGE Ws_shift through the SAME
SPIKING WTA the c3 delta rule uses (run_with_ens -> argmax ens firing), scored per-slot vs TRUE roles on objrel.
  ridge+spiking solves objrel -> the spiking WTA is FINE with good weights -> wall (a): the DELTA RULE's learning.
  ridge+spiking fails objrel   -> the spiking WTA corrupts structure -> wall (b): the spiking DEPLOY."""
import os, sys, time
os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
import research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk as C
from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder, _gen, _TRAIN_KINDS, _ROLE_IDX

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
N_TRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 35
N_TEST = int(sys.argv[3]) if len(sys.argv) > 3 else 12
C.WS_BIAS_SCALE_C2 = 0.0; C.WS_ENS_FLOOR_C2 = 150.0; C.WS_REPLAY = 1; C.READ_T_STEP_C2 = 18
corpus = C.setup_corpus(seed=42)
subj, verb, obj = corpus["subj"], corpus["verb"], corpus["obj"]
enc = Encoder(corpus["discovered"])
rng = np.random.default_rng(seed * 101 + 5)
train = _gen(_TRAIN_KINDS, N_TRAIN, rng, subj, verb, obj)
ub, ens, inh = C._build_wired_bridge(seed, corpus, mode="c2")
res_idx, W_in = C.wire_reservoir(ub, enc.dim, seed)
res = C.UBReservoir(ub, res_idx, W_in); res.snapshot_after_wiring()
Ws = C._fit_Ws_spiking(res, enc, train)                       # SIGNED ridge (the c2 read-out that solves objrel linearly)
n_res = len(res_idx)
Ws_shift = {k: (Ws[k] - Ws[k].min()) for k in Ws}            # the positive-shift the spiking deploy uses (c2)
pre, post = C._ws_edges(res_idx, ens)
ub.bridge.set_pathway_weights("res2ens", pre, post, np.zeros(len(pre), np.float32), add_missing=True)  # wire res2ens first
res.snapshot_after_wiring()                                   # (ridge _fit_Ws_spiking doesn't wire it; _learn_Ws_spiking does)


def write(Wk):
    w = np.empty(len(pre), np.float32); p = 0
    for r in range(3):
        for _e in ens[r]:
            w[p:p + n_res] = Wk[r]; p += n_res
    ub.bridge.set_pathway_weights("res2ens", pre, post, w, add_missing=False)


def score_spiking(sentences):
    ok = tot = s0ok = s0t = 0
    for toks, roles in sentences:
        for k, pos in enumerate(sorted(roles)):
            if k >= 3:
                break
            tgt = _ROLE_IDX[roles[pos]]
            if tgt >= 3:
                continue
            Wk = np.array([Ws_shift[k][:n_res, r] for r in range(3)], np.float64)  # ridge Ws_shift as res2ens weights
            write(Wk)
            _rho, a = res.run_with_ens(enc.encode(toks), ens)   # SPIKING WTA deploy (ens competition, ignition order)
            pred = int(np.argmax(np.asarray(a, float)))
            hit = int(pred == tgt); ok += hit; tot += 1
            if k == 0:
                s0ok += hit; s0t += 1
    return ok, tot, s0ok, s0t


trng = np.random.default_rng(seed * 977 + 13)
canon = _gen(["transitive"], N_TEST, trng, subj, verb, obj)
objr = _gen(["objrel"], N_TEST, trng, subj, verb, obj)
co, ct, cs0, cs0t = score_spiking(canon)
oo, ot, os0, os0t = score_spiking(objr)
print(f"seed {seed} RIDGE-Ws -> SPIKING-WTA: CANON {co}/{ct}={co/max(ct,1):.2f} | OBJREL {oo}/{ot}={oo/max(ot,1):.2f} | "
      f"OBJREL slot0(THEME) {os0}/{os0t}={os0/max(os0t,1):.2f}", flush=True)
