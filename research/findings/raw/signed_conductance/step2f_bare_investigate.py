"""HONEST re-attribution: the anti-cheat showed BARE (Wp=max(Ws,0) exc rows only, no follower/bias) = 18/18 on 42/43/44.
So the load-bearing mechanism is a CLIPPED-positive read at LOW floor, NOT the signed follower/bias. Disambiguate + test
generalization: the clipped-positive exc read (reservoir->ens with Wp=max(Ws,0), winner=argmax ens firing) vs the c2
Dale-shift Ws_shifted=Ws-Ws.min(), at floor {30, 150}, on SIX seeds (42/43/44 + unseen 100/101/102). Tells:
  * clipping vs floor: Wp@floor30 vs Wp@floor150 vs Ws_shift@floor30.
  * generalization: do 100/101/102 also give 18/18, or is it overfit to 42/43/44?"""
import os, sys, time
os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
from sim.backend import get_backend, to_host
import step1_onoff_opponent as S
import research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk as C
from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder, _gen, _TRAIN_KINDS
from research.runners._emerge61_spiking_broca_order_robustness_derisk import _restore_state
from research.runners.unified_brain_bridge import UnifiedBrainBridge
from research.runners._rungB1b_neural_role_wta_derisk import PROJ_DIM
from research.runners.core_sim_composition import RESET_STEPS

ENS_P = C.WTA_P_C2; RES_N = C.RES_N; WS_REPLAY = 6; RES_BIAS = C.RES_BIAS; READ_T = C.READ_T_STEP_C2
seeds = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["42", "43", "44", "100", "101", "102"])]
floors = [float(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["30", "150"])]
SCALE_C = 90.0
corpus = S.setup_corpus(seed=42); test = corpus["test"]
enc = Encoder(corpus["discovered"])
xp, _ = get_backend()


def run_seed(seed, kind):
    rng = np.random.default_rng(seed * 101 + 5)
    train = _gen(_TRAIN_KINDS, C.N_TRAIN_PER, rng, corpus["subj"], corpus["verb"], corpus["obj"])
    ub = UnifiedBrainBridge(seed=seed, proj_dim=PROJ_DIM, concepts=corpus["concepts"],
                            enable_synaptic_route=False, role_wta_n=3 * ENS_P, reservoir_n=RES_N)
    res_idx, W_in = C.wire_reservoir(ub, enc.dim, seed)
    res = C.UBReservoir(ub, res_idx, W_in)
    n_res = len(res_idx)
    rb = int(ub.role_wta_base)
    ens = [np.arange(rb + r * ENS_P, rb + (r + 1) * ENS_P, dtype=np.int64) for r in range(3)]
    pre, post = [], []
    for r in range(3):
        for e in ens[r]:
            for s in res_idx:
                pre.append(int(s)); post.append(int(e))
    ub.bridge.set_pathway_weights("r2e", pre, post, np.zeros(len(pre), np.float32), add_missing=True)
    res.snapshot_after_wiring()
    Ws = C._fit_Ws_spiking(res, enc, train)
    host = [S._host_signed_winners(res, enc, Ws, toks) for toks, *_ in test]
    # the read-out matrix: kind 'clip' = max(Ws,0); kind 'shift' = Ws - Ws.min() (the c2 Dale offset)
    if kind == "clip":
        Wr = {k: np.maximum(Ws[k][:n_res, :3], 0.0) for k in Ws}
    else:
        Wr = {k: (Ws[k][:, :3] - Ws[k][:, :3].min())[:n_res, :] for k in Ws}
    proj = float((res.final_state(enc.encode(test[0][0]))[:n_res] @ np.maximum(Ws[0][:n_res, :3], 0)).max())
    sc = SCALE_C / max(1e-9, proj)
    b = ub.bridge

    def read(U, k, floor):
        w = []
        for r in range(3):
            col = Wr[k][:n_res, r].astype(np.float64) * sc
            for _e in ens[r]:
                for i in range(n_res):
                    w.append(float(col[i]))
        b.set_pathway_weights("r2e", pre, post, np.asarray(w, np.float32), add_missing=False)
        _restore_state(b, res._snap)
        pou, ph = b.core_config.enable_ou_process, b.core_config.enable_hebbian_learning
        b.core_config.enable_ou_process = False; b.core_config.enable_hebbian_learning = False
        es = np.zeros(3)
        try:
            for _ in range(RESET_STEPS):
                b.runtime_state.current_time_ms += b.core_config.dt_ms; b._run_one_simulation_step()
            for _rep in range(WS_REPLAY):
                for t in range(len(U)):
                    cur = np.zeros(b.core_config.num_neurons)
                    cur[res_idx] = res.W_in @ U[t] + RES_BIAS
                    for r in range(3):
                        cur[ens[r]] = floor
                    b.cp_external_input_current[:] = xp.asarray(cur.astype(np.float32))
                    for _ in range(READ_T):
                        b.runtime_state.current_time_ms += b.core_config.dt_ms; b._run_one_simulation_step()
                        fs = np.asarray(to_host(b.cp_firing_states)).astype(np.float64)
                        for r in range(3):
                            es[r] += fs[ens[r]].sum()
        finally:
            b.cp_external_input_current[:] = 0.0
            b.core_config.enable_ou_process = pou; b.core_config.enable_hebbian_learning = ph
        return es

    outs = {}
    for floor in floors:
        ok = 0
        for (toks, *_), hs in zip(test, host):
            for k in (0, 1, 2):
                es = read(enc.encode(toks), k, floor)
                ok += int(int(np.argmax(es)) == hs[k])
        outs[floor] = ok
    return outs


for seed in seeds:
    t0 = time.time()
    clip = run_seed(seed, "clip")
    shift = run_seed(seed, "shift")
    cs = " ".join(f"fl{f:.0f}:{clip[f]}" for f in floors)
    ss = " ".join(f"fl{f:.0f}:{shift[f]}" for f in floors)
    print(f"seed {seed}: CLIP[{cs}]  SHIFT[{ss}]  /18  [{time.time()-t0:.0f}s]", flush=True)
