"""ISOLATE the objrel residual (SURPASS move 1): the c3 POSITIVE spiking read-out reads objrel by POSITION (0/36). Is the
wall the POSITIVE-deploy constraint, or does the reservoir FEATURE itself fail to encode objrel on this substrate?

Test the SIGNED host read-out (the non-biological CEILING): fit Ws by ridge (_fit_Ws_spiking), then score argmax((f @ Ws))
UNSHIFTED (signed logits) on held-out canonical vs objrel, per-slot, vs TRUE roles. Three read-outs compared on the SAME
reservoir feature:
  (A) SIGNED host argmax  argmax((concat[f,1] @ Ws)[[0,1,2]])            -- signed ceiling (EMERGE-78-style)
  (B) POSITIVE-SHIFTED host argmax  argmax((concat[f,1] @ (Ws - Ws.min()))[[0,1,2]]) -- what the spiking deploy approximates
If (A) solves objrel but (B) doesn't -> the wall is the POSITIVE-SHIFT deploy (opponent/feedforward-inhibition channels
realize signed deploy -> the biological SURPASS). If (A) also fails -> the reservoir feature doesn't encode objrel here
(deeper). This is FEATURE-level (final_state), NO spiking WTA deploy -> fast + isolates the read-out constraint cleanly."""
import os, sys, time
os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
import research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk as C
from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder, _gen, _TRAIN_KINDS, _ROLE_IDX

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
N_TRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 35
N_TEST = int(sys.argv[3]) if len(sys.argv) > 3 else 12
C.WS_REPLAY = 1; C.READ_T_STEP_C2 = 18
corpus = C.setup_corpus(seed=42)
subj, verb, obj = corpus["subj"], corpus["verb"], corpus["obj"]
enc = Encoder(corpus["discovered"])
rng = np.random.default_rng(seed * 101 + 5)
train = _gen(_TRAIN_KINDS, N_TRAIN, rng, subj, verb, obj)
ub, ens, inh = C._build_wired_bridge(seed, corpus, mode="c2")
res_idx, W_in = C.wire_reservoir(ub, enc.dim, seed)
res = C.UBReservoir(ub, res_idx, W_in); res.snapshot_after_wiring()
Ws = C._fit_Ws_spiking(res, enc, train)                       # SIGNED ridge on the reservoir RATE feature
n_res = len(res_idx)
Ws_shift = {k: (Ws[k] - Ws[k].min()) for k in Ws}


def score(sentences, shifted):
    ok = tot = s0ok = s0tot = 0
    for toks, roles in sentences:
        f = np.concatenate([np.asarray(res.final_state(enc.encode(toks)), float), [1.0]])
        for k, pos in enumerate(sorted(roles)):
            if k >= 3:
                break
            tgt = _ROLE_IDX[roles[pos]]
            if tgt >= 3:
                continue
            W = Ws_shift[k] if shifted else Ws[k]
            pred = int(np.argmax((f @ W)[[0, 1, 2]]))
            hit = int(pred == tgt); ok += hit; tot += 1
            if k == 0:
                s0ok += hit; s0tot += 1
    return ok, tot, s0ok, s0tot


trng = np.random.default_rng(seed * 977 + 13)
canon = _gen(["transitive"], N_TEST, trng, subj, verb, obj)
objr = _gen(["objrel"], N_TEST, trng, subj, verb, obj)
for label, shifted in [("SIGNED (ceiling)", False), ("POSITIVE-SHIFTED (deploy approx)", True)]:
    co, ct, cs0, cs0t = score(canon, shifted)
    oo, ot, os0, os0t = score(objr, shifted)
    print(f"seed {seed} {label}: CANON {co}/{ct}={co/max(ct,1):.2f} | OBJREL {oo}/{ot}={oo/max(ot,1):.2f} | "
          f"OBJREL slot0(THEME) {os0}/{os0t}={os0/max(os0t,1):.2f}", flush=True)
