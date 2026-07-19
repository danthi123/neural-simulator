"""RANK 1+2 (Design D): condition the POSITIVE read so seed 44's slot2 ensembles leave the DEPOL-BLOCK branch. Seed 44
fails because slot2's absolute drive (~17-19) is higher than slot0/1 (~15-16), so a global floor/scale pushes slot2 past
the Izhikevich f-I peak (winner fires FEWEST). DIVISIVE NORMALIZATION on the ens pool (cp_input_divisive_mask; divide each
ens input by sigma+gain*mean_ens_input) adaptively rescales each slot's drive into the MONOTONE band -> the count becomes
monotone -> the highest-drive role wins. Purely EXCITATORY positive read (Ws_shifted), NO shunting. If a single (floor,gain,
scale) gives ~18/18 on 44 AND keeps 42/43 at 18/18, the principled positive read reaches 3/3 -> the read-out shortcut closes."""
import os, sys, time
os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
from sim.backend import get_backend
import research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk as C
from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder, _gen, _TRAIN_KINDS
import step1_onoff_opponent as S

seeds = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["44"])]
floors = [float(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["80", "150"])]
gains = [float(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["0.02", "0.05", "0.10"])]
scales_c = [float(x) for x in (sys.argv[4].split(",") if len(sys.argv) > 4 else ["130"])]
xp, _ = get_backend()
corpus = S.setup_corpus(seed=42); test = corpus["test"]
C.WS_BIAS_SCALE_C2 = 0.0
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
    b = ub_s.bridge
    mask = np.zeros(b.core_config.num_neurons, dtype=bool)
    for r in range(3):
        mask[ens_s[r]] = True
    b.cp_input_divisive_mask = xp.asarray(mask)                 # divnorm on the ens pool (adaptive drive rescale)
    b.core_config.input_divisive_sigma = 1.0
    res_s.snapshot_after_wiring()                              # re-snapshot AFTER the mask so _restore_state keeps it
    f_ref = np.concatenate([res.final_state(enc.encode(test[0][0])), [1.0]])
    proj = max(1e-9, float((f_ref[:n_res] @ Ws_shift[0][:n_res, :3]).max()))
    best = 0; bestcfg = None
    for scale_c in scales_c:
        sc = scale_c / proj
        sr = C.SlotReadout(ub_s, res_s, ens_s, Ws_shift, sc)
        for floor in floors:
            C.WS_ENS_FLOOR_C2 = floor
            for gain in gains:
                b.core_config.input_divisive_gain = gain
                ok = 0
                for (toks, *_), hs in zip(test, host):
                    for k in (0, 1, 2):
                        rb = sr.set_slot(k)
                        _f, es = res_s.run_with_ens(enc.encode(toks), ens_s, role_bias=rb)
                        ok += int(int(np.argmax(es)) == hs[k])
                if ok > best:
                    best = ok; bestcfg = (floor, gain, scale_c)
                print(f"seed {seed} floor {floor:4.0f} gain {gain:.3f} c{scale_c:.0f}: host-agree {ok}/18", flush=True)
    print(f"=== seed {seed}: BEST {best}/18 @ (floor,gain,c) {bestcfg}  [{time.time()-t0:.0f}s] ===", flush=True)
