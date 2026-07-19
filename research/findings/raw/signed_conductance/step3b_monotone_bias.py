"""The positive read is BLOCK-inverted on ALL slots (host-role fires most only when host=the low-drive role, which luck
matches the bias on slots 0/1 but NOT slot2). The clean fix = ensembles in the MONOTONE band (low floor + low scale ->
total drive < the ~200pA f-I peak, STEEP high-gain region that resolves the ~5% margin) AND the learned BIAS tonic, so
argmax(count) = argmax(rows*sc + bias*bscale) = host. Sweeps that combination (NO divnorm) on seed 44 (+ 42/43 no-regress).
Purely excitatory positive read; winner = neural argmax over ens firing."""
import os, sys, time
os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
import research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk as C
from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder, _gen, _TRAIN_KINDS
import step1_onoff_opponent as S

seeds = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["44"])]
floors = [float(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["20", "50", "90"])]
scales_c = [float(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["40", "70", "110"])]
bscales = [float(x) for x in (sys.argv[4].split(",") if len(sys.argv) > 4 else ["0", "2", "6"])]
corpus = S.setup_corpus(seed=42); test = corpus["test"]
for seed in seeds:
    t0 = time.time()
    enc = Encoder(corpus["discovered"])
    rng = np.random.default_rng(seed * 101 + 5)
    train = _gen(_TRAIN_KINDS, C.N_TRAIN_PER, rng, corpus["subj"], corpus["verb"], corpus["obj"])
    ub, ens, inh = C._build_wired_bridge(seed, corpus, mode="c2")
    res_idx, W_in = C.wire_reservoir(ub, enc.dim, seed)
    res = C.UBReservoir(ub, res_idx, W_in); res.snapshot_after_wiring()
    Ws = C._fit_Ws_spiking(res, enc, train)
    Ws_shift = {k: (W - W.min()) for k, W in Ws.items()}
    n_res = len(res_idx)
    host = []
    for toks, *_ in test:
        f = np.concatenate([res.final_state(enc.encode(toks)), [1.0]])
        host.append([int(np.argmax((f @ Ws[k])[[0, 1, 2]])) for k in (0, 1, 2)])
    ub_s, ens_s, _ = C._build_wired_bridge(seed, corpus, mode="c2")
    res_idx_s, W_in_s = C.wire_reservoir(ub_s, enc.dim, seed)
    res_s = C.UBReservoir(ub_s, res_idx_s, W_in_s)
    C.wire_ws_synapses(ub_s, res_idx_s, ens_s, Ws_shift[0], 1.0, add_missing=True)
    res_s.snapshot_after_wiring()
    f_ref = np.concatenate([res.final_state(enc.encode(test[0][0])), [1.0]])
    proj = max(1e-9, float((f_ref[:n_res] @ Ws_shift[0][:n_res, :3]).max()))
    best = 0; bestcfg = None
    for scale_c in scales_c:
        sc = scale_c / proj
        sr = C.SlotReadout(ub_s, res_s, ens_s, Ws_shift, sc)
        for floor in floors:
            C.WS_ENS_FLOOR_C2 = floor
            for bs in bscales:
                C.WS_BIAS_SCALE_C2 = bs
                ok = 0
                for (toks, *_), hs in zip(test, host):
                    for k in (0, 1, 2):
                        rb = sr.set_slot(k)
                        _f, es = res_s.run_with_ens(enc.encode(toks), ens_s, role_bias=rb)
                        ok += int(int(np.argmax(es)) == hs[k])
                if ok > best:
                    best = ok; bestcfg = (floor, scale_c, bs)
                print(f"seed {seed} floor {floor:4.0f} c{scale_c:4.0f} bias {bs:4.1f}: host-agree {ok}/18", flush=True)
    print(f"=== seed {seed}: BEST {best}/18 @ (floor,c,bias) {bestcfg}  [{time.time()-t0:.0f}s] ===", flush=True)
