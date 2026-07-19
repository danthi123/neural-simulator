"""Design-D instrumentation: for seed 44's POSITIVE read (the degraded draw, 11/18), print per slot the LINEAR drive
(Ws_shifted@f) per role vs the spiking ens COUNT per role + the host winner. Discriminates the two seed-44 hypotheses:
  * DEPOL-BLOCK: winner-drive > runner-drive BUT winner-count < runner-count -> RANK 1/2 (floor/scale into monotone band
    + divnorm) is the fix.
  * FEATURE-DEGENERATE: winner-count > runner-count yet argmax wrong (or the linear drive itself doesn't separate) ->
    RANK 3/5 (multi-window feature / reservoir committee) is the fix.
"""
import os, sys
os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
import research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk as C
from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder, _gen, _TRAIN_KINDS
import step1_onoff_opponent as S

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 44
scale_c = float(sys.argv[2]) if len(sys.argv) > 2 else 130.0
corpus = S.setup_corpus(seed=42); test = corpus["test"]
enc = Encoder(corpus["discovered"])
rng = np.random.default_rng(seed * 101 + 5)
train = _gen(_TRAIN_KINDS, C.N_TRAIN_PER, rng, corpus["subj"], corpus["verb"], corpus["obj"])
ub, ens, inh = C._build_wired_bridge(seed, corpus, mode="c2")
res_idx, W_in = C.wire_reservoir(ub, enc.dim, seed)
res = C.UBReservoir(ub, res_idx, W_in)
res.snapshot_after_wiring()
Ws = C._fit_Ws_spiking(res, enc, train)
Ws_shift = {k: (W - W.min()) for k, W in Ws.items()}
n_res = len(res_idx)
ub_s, ens_s, _ = C._build_wired_bridge(seed, corpus, mode="c2")
res_idx_s, W_in_s = C.wire_reservoir(ub_s, enc.dim, seed)
res_s = C.UBReservoir(ub_s, res_idx_s, W_in_s)
C.wire_ws_synapses(ub_s, res_idx_s, ens_s, Ws_shift[0], 1.0, add_missing=True)
res_s.snapshot_after_wiring()
f_ref = np.concatenate([res.final_state(enc.encode(test[0][0])), [1.0]])
proj = max(1e-9, float((f_ref[:n_res] @ Ws_shift[0][:n_res, :3]).max()))
sc = scale_c / proj
C.WS_BIAS_SCALE_C2 = 0.0
sr = C.SlotReadout(ub_s, res_s, ens_s, Ws_shift, sc)
block = degen = 0
for si, (toks, *_) in enumerate(test):
    f = res.final_state(enc.encode(toks))
    for k in (0, 1, 2):
        hs = int(np.argmax((np.concatenate([f, [1.0]]) @ Ws[k])[[0, 1, 2]]))
        lin = f @ Ws_shift[k][:n_res, :3]                 # linear drive per role (what the ens SHOULD rank by)
        rb = sr.set_slot(k)
        _f, es = res_s.run_with_ens(enc.encode(toks), ens_s, role_bias=rb)
        amx = int(np.argmax(es))
        ru = int(np.argsort(lin)[-2])                     # linear runner-up role
        tag = ""
        if amx != hs:
            if lin[hs] > lin[ru] and es[hs] < es[ru]:
                tag = "BLOCK"; block += 1
            else:
                tag = "DEGEN"; degen += 1
        print(f" s{seed} fact{si} slot{k} host={hs} amx={amx} {tag} | lin={np.round(lin,2)} | es={np.round(es,0)}",
              flush=True)
print(f"=== seed {seed}: BLOCK-type errors {block}, DEGEN-type errors {degen} ===", flush=True)
