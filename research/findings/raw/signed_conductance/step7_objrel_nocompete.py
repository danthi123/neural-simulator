"""Locate the objrel wall: COMPETITION vs DRIVE-nonlinearity. Low floor didn't recover objrel (0/36 at all floors). Now
test whether REMOVING the WTA mutual inhibition (i2e lesion) -- so the ensembles fire ~independently proportional to their
drive, argmax(firing) ~ the LINEAR drive argmax -- recovers objrel. If yes: the WTA COMPETITION (biased-competition ignition
amplifying a pedestal-corrupted order) is the wall; a graded/non-competitive read resolves the structural margin (though the
earlier arc found removing i2e hurts canonical UNSEEN seeds -> a competition-vs-margin whack-a-mole). If objrel STILL fails
without competition: the drive->firing f-I nonlinearity itself loses the margin (deeper). Tested at floor 30 (subtractive
regime) and 150 (committed). Ridge Ws_shift, per-slot vs TRUE roles, held-out."""
import os, sys, time
os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
import research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk as C
from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder, _gen, _TRAIN_KINDS, _ROLE_IDX

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
N_TRAIN, N_TEST = 35, 12
C.WS_BIAS_SCALE_C2 = 0.0; C.WS_REPLAY = 1; C.READ_T_STEP_C2 = 30
corpus = C.setup_corpus(seed=42)
subj, verb, obj = corpus["subj"], corpus["verb"], corpus["obj"]
enc = Encoder(corpus["discovered"])
rng = np.random.default_rng(seed * 101 + 5)
train = _gen(_TRAIN_KINDS, N_TRAIN, rng, subj, verb, obj)
ub, ens, inh = C._build_wired_bridge(seed, corpus, mode="c2")
res_idx, W_in = C.wire_reservoir(ub, enc.dim, seed)
res = C.UBReservoir(ub, res_idx, W_in); res.snapshot_after_wiring()
Ws = C._fit_Ws_spiking(res, enc, train)
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
for fl in [150.0, 30.0]:
    for compete in (True, False):
        C.WS_ENS_FLOOR_C2 = fl
        restore = None if compete else C.lesion_wta_i2e_c2(ub, ens, inh)
        co, ct, cs0, cs0t = score(canon)
        oo, ot, os0, os0t = score(objr)
        if restore:
            restore()
        tag = "WTA-compete" if compete else "NO-compete (i2e lesion)"
        print(f"seed {seed} FLOOR={fl:>5.0f} {tag:>24}: CANON {co/max(ct,1):.2f} | OBJREL {oo/max(ot,1):.2f} | "
              f"objrel-slot0(THEME) {os0}/{os0t}={os0/max(os0t,1):.2f}", flush=True)
