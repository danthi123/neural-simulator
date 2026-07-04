"""FAIR disambiguation + generalization test (on the VALIDATED c2 harness, per-read SCALE SWEEP -- the step2f confound
fixed). The positive read-out `Wr` as excitatory synapses reservoir->ens, winner = argmax ens firing, at floor 30, bias
OFF; sweep the scale and report the BEST host-agree. Two read-out matrices x SIX seeds (42/43/44 tuned + 100/101/102 unseen):
  * CLIP  = max(Ws,0)            (the anti-cheat's load-bearing read)
  * SHIFT = Ws - Ws.min()        (the committed c2 positive read = argmax(Ws_rows@f))
Tells: (a) does CLIP@floor30 give 18/18 on ALL 6 seeds (a genuine generalizing surpass) or only the 3 tuned ones (overfit)?
(b) is it the CLIPPING (CLIP >> SHIFT at floor 30) or just the low floor (CLIP ~ SHIFT)? For reference, the committed c2 read
is SHIFT at floor 150 = 42:18 43:18 44:11."""
import os, sys, time
os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
import research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk as C
from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder, _gen, _TRAIN_KINDS
import step1_onoff_opponent as S

seeds = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["42", "43", "44", "100", "101", "102"])]
FLOOR = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
SCALES_C = [40., 60., 90., 130., 180., 240., 320.]
corpus = S.setup_corpus(seed=42); test = corpus["test"]
C.WS_BIAS_SCALE_C2 = 0.0
C.WS_ENS_FLOOR_C2 = FLOOR


def best_for(seed, enc, Ws, host, kind):
    n_res = C.RES_N
    if kind == "clip":
        Wr = {k: np.maximum(Ws[k][:, :3], 0.0) for k in Ws}     # max(Ws,0) over ALL rows (reservoir + bias); bias off anyway
    else:
        Wr = {k: (Ws[k][:, :3] - Ws[k][:, :3].min()) for k in Ws}
    ub_s, ens_s, _ = C._build_wired_bridge(seed, corpus, mode="c2")
    res_idx_s, W_in_s = C.wire_reservoir(ub_s, enc.dim, seed)
    res_s = C.UBReservoir(ub_s, res_idx_s, W_in_s)
    C.wire_ws_synapses(ub_s, res_idx_s, ens_s, Wr[0], 1.0, add_missing=True)
    res_s.snapshot_after_wiring()
    f_ref = res_s.final_state(enc.encode(test[0][0]))
    proj = max(1e-9, float((f_ref[:len(res_idx_s)] @ np.maximum(Wr[0][:len(res_idx_s), :3], 0)).max()))
    best = 0
    for c in SCALES_C:
        sc = c / proj
        sr = C.SlotReadout(ub_s, res_s, ens_s, Wr, sc)
        ok = 0
        for (toks, *_), hs in zip(test, host):
            for k in (0, 1, 2):
                rb = sr.set_slot(k)
                _f, es = res_s.run_with_ens(enc.encode(toks), ens_s, role_bias=rb)
                ok += int(int(np.argmax(es)) == hs[k])
        best = max(best, ok)
    return best


for seed in seeds:
    t0 = time.time()
    enc = Encoder(corpus["discovered"])
    rng = np.random.default_rng(seed * 101 + 5)
    train = _gen(_TRAIN_KINDS, C.N_TRAIN_PER, rng, corpus["subj"], corpus["verb"], corpus["obj"])
    ub, ens, inh = C._build_wired_bridge(seed, corpus, mode="c2")
    res_idx, W_in = C.wire_reservoir(ub, enc.dim, seed)
    res = C.UBReservoir(ub, res_idx, W_in); res.snapshot_after_wiring()
    Ws = C._fit_Ws_spiking(res, enc, train)
    host = []
    for toks, *_ in test:
        f = np.concatenate([res.final_state(enc.encode(toks)), [1.0]])
        host.append([int(np.argmax((f @ Ws[k])[[0, 1, 2]])) for k in (0, 1, 2)])
    clip = best_for(seed, enc, Ws, host, "clip")
    shift = best_for(seed, enc, Ws, host, "shift")
    print(f"seed {seed} floor {FLOOR:.0f}: CLIP best {clip}/18  |  SHIFT best {shift}/18   [{time.time()-t0:.0f}s]", flush=True)
