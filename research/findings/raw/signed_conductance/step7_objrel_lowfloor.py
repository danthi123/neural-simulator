"""SURPASS de-risk (research-gate #1): the objrel structural read fails through the SPIKING WTA (0.03) because the
common-mode pedestal (WS_ENS_FLOOR + positive-shift) swamps the ignition-order, while the shift-invariant LINEAR argmax
solves it (1.00). Per the conductance finding: at LOW floor the shared WTA inhibition is SUBTRACTIVE (removes the common
mode) rather than divisive/shunting. So SWEEP the ens floor and deploy the RIDGE Ws_shift through the spiking WTA on objrel:
if objrel slot0(THEME) recovers toward 1.00 at low floor (while canonical stays ~1.00), the common-mode-pedestal hypothesis
is confirmed and the family is surpassable runner-side (NO sim/ edit). Scored per-slot vs TRUE roles (held-out). This is the
cheapest experiment that tells us the WHOLE common-mode/opponency family is surpassable on this residual."""
import os, sys, time
os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
import research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk as C
from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder, _gen, _TRAIN_KINDS, _ROLE_IDX

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
N_TRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 35
N_TEST = int(sys.argv[3]) if len(sys.argv) > 3 else 12
FLOORS = [float(x) for x in (sys.argv[4].split(",") if len(sys.argv) > 4 else ["150", "90", "60", "30", "15"])]
C.WS_BIAS_SCALE_C2 = 0.0; C.WS_REPLAY = 1; C.READ_T_STEP_C2 = 30
corpus = C.setup_corpus(seed=42)
subj, verb, obj = corpus["subj"], corpus["verb"], corpus["obj"]
enc = Encoder(corpus["discovered"])
rng = np.random.default_rng(seed * 101 + 5)
train = _gen(_TRAIN_KINDS, N_TRAIN, rng, subj, verb, obj)
ub, ens, inh = C._build_wired_bridge(seed, corpus, mode="c2")
res_idx, W_in = C.wire_reservoir(ub, enc.dim, seed)
res = C.UBReservoir(ub, res_idx, W_in); res.snapshot_after_wiring()
Ws = C._fit_Ws_spiking(res, enc, train)                       # signed ridge; deployed positive-SHIFTED (c2 style)
n_res = len(res_idx)
Ws_shift = {k: (Ws[k] - Ws[k].min()) for k in Ws}
pre, post = C._ws_edges(res_idx, ens)
ub.bridge.set_pathway_weights("res2ens", pre, post, np.zeros(len(pre), np.float32), add_missing=True)
res.snapshot_after_wiring()


def write(Wk):
    w = np.empty(len(pre), np.float32); p = 0
    for r in range(3):
        for _e in ens[r]:
            w[p:p + n_res] = Wk[r]; p += n_res
    ub.bridge.set_pathway_weights("res2ens", pre, post, w, add_missing=False)


def score(sentences):
    ok = tot = s0ok = s0t = 0
    for toks, roles in sentences:
        for k, pos in enumerate(sorted(roles)):
            if k >= 3:
                break
            tgt = _ROLE_IDX[roles[pos]]
            if tgt >= 3:
                continue
            write(np.array([Ws_shift[k][:n_res, r] for r in range(3)], np.float64))
            _rho, a = res.run_with_ens(enc.encode(toks), ens)
            pred = int(np.argmax(np.asarray(a, float)))
            hit = int(pred == tgt); ok += hit; tot += 1
            if k == 0:
                s0ok += hit; s0t += 1
    return ok, tot, s0ok, s0t


trng = np.random.default_rng(seed * 977 + 13)
canon = _gen(["transitive"], N_TEST, trng, subj, verb, obj)
objr = _gen(["objrel"], N_TEST, trng, subj, verb, obj)
for fl in FLOORS:
    t0 = time.time()
    C.WS_ENS_FLOOR_C2 = fl                                     # the common-mode pedestal (also gates the g_i subtractive regime)
    co, ct, cs0, cs0t = score(canon)
    oo, ot, os0, os0t = score(objr)
    print(f"seed {seed} FLOOR={fl:>5.0f}: CANON {co}/{ct}={co/max(ct,1):.2f} | OBJREL {oo}/{ot}={oo/max(ot,1):.2f} | "
          f"OBJREL slot0(THEME) {os0}/{os0t}={os0/max(os0t,1):.2f}  [{time.time()-t0:.0f}s]", flush=True)
