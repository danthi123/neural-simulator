"""DESIGN A, final: deliver the ENTIRE signed read-out (reservoir rows AND the +1 bias intercept) through the SAME
conductance pathway, so the bias is ADDITIVE/SUBTRACTIVE like the rows -- never a silencing tonic. Seed 43's slot2 failed
because its host role has a NEGATIVE bias (-0.54); the tonic floor+bias*bscale went negative -> silenced the host role. Fix:
a constant-rate BIAS UNIT (the '+1' feature element) -- bias_exc (exc, -> ens with Wp[n_res]) + bias_inh (inh follower,
-> ens with Wn[n_res]) -- so full = (Wp-Wn)@[f;1] is computed ENTIRELY in the linear conductance domain, one ensemble/role,
argmax(ens firing) = argmax(full) = host. NO tonic. Sweeps seed 43 (the laggard); then the shared fixed point on 42/44."""
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

ENS_P = C.WTA_P_C2; RES_N = C.RES_N; WS_REPLAY = 6; FOLLOW_W = 120.0; RES_BIAS = C.RES_BIAS; READ_T = C.READ_T_STEP_C2
BIAS_DRIVE = 150.0    # constant drive to the bias units (they fire at a ~constant rate = the '+1' feature element)


def build(seed, corpus, enc):
    # role_wta slice: ens[0..2] (3*ENS_P) + res_inh (RES_N) + bias_exc (1) + bias_inh (1)
    ub = UnifiedBrainBridge(seed=seed, proj_dim=PROJ_DIM, concepts=corpus["concepts"],
                            enable_synaptic_route=False, role_wta_n=3 * ENS_P + RES_N + 2, reservoir_n=RES_N)
    res_idx, W_in = C.wire_reservoir(ub, enc.dim, seed)
    res = C.UBReservoir(ub, res_idx, W_in)
    rb = int(ub.role_wta_base)
    ens = [np.arange(rb + r * ENS_P, rb + (r + 1) * ENS_P, dtype=np.int64) for r in range(3)]
    res_inh = np.arange(rb + 3 * ENS_P, rb + 3 * ENS_P + RES_N, dtype=np.int64)
    bias_exc = int(rb + 3 * ENS_P + RES_N)
    bias_inh = int(rb + 3 * ENS_P + RES_N + 1)
    ub.bridge.cp_traits[res_inh] = 1
    ub.bridge.cp_traits[bias_inh] = 1
    ub.bridge._cached_inhibitory_mask = None
    ub.bridge.set_pathway_weights("follow", [int(x) for x in res_idx], [int(x) for x in res_inh],
                                  np.full(RES_N, FOLLOW_W, np.float32), add_missing=True)
    exc_src = np.concatenate([res_idx, [bias_exc]])          # n_res + 1 exc sources (reservoir rows + bias row)
    inh_src = np.concatenate([res_inh, [bias_inh]])          # n_res + 1 inh sources
    return ub, res, res_idx, ens, res_inh, bias_exc, bias_inh, exc_src, inh_src


def _edges(src, ens):
    pre, post = [], []
    for r in range(3):
        for e in ens[r]:
            for s in src:
                pre.append(int(s)); post.append(int(e))
    return pre, post


def _weights(src, ens, W_rows, scale, bias_gain):
    n = len(src); w = []
    for r in range(3):
        col = W_rows[:n, r].astype(np.float64) * float(scale)
        col[n - 1] *= float(bias_gain)                       # the LAST source is the bias unit -> its own gain
        for _e in ens[r]:
            for i in range(n):
                w.append(float(col[i]))
    return np.asarray(w, np.float32)


def run(seed, corpus, floors, ratios, scale_c, bgains, enc):
    rng = np.random.default_rng(seed * 101 + 5)
    train = _gen(_TRAIN_KINDS, C.N_TRAIN_PER, rng, corpus["subj"], corpus["verb"], corpus["obj"])
    ub, res, res_idx, ens, res_inh, bias_exc, bias_inh, exc_src, inh_src = build(seed, corpus, enc)
    n_res = len(res_idx)
    pe, po = _edges(exc_src, ens); ie, io = _edges(inh_src, ens)
    ub.bridge.set_pathway_weights("res2ens_exc", pe, po, np.zeros(len(pe), np.float32), add_missing=True)
    ub.bridge.set_pathway_weights("res2ens_inh", ie, io, np.zeros(len(ie), np.float32), add_missing=True)
    res.snapshot_after_wiring()
    Ws = C._fit_Ws_spiking(res, enc, train)
    host = [S._host_signed_winners(res, enc, Ws, toks) for toks, *_ in corpus["test"]]
    proj = float((res.final_state(enc.encode(corpus["test"][0][0]))[:n_res] @ np.maximum(Ws[0][:n_res, :3], 0)).max())
    Wp = {k: np.maximum(Ws[k][:n_res + 1, :3], 0.0) for k in Ws}   # include the bias row (index n_res)
    Wn = {k: np.maximum(-Ws[k][:n_res + 1, :3], 0.0) for k in Ws}
    b = ub.bridge; xp, _ = get_backend()

    def read(U, k, floor, sc, sci, bgain):
        b.set_pathway_weights("res2ens_exc", pe, po, _weights(exc_src, ens, Wp[k], sc, bgain), add_missing=False)
        b.set_pathway_weights("res2ens_inh", ie, io, _weights(inh_src, ens, Wn[k], sci, bgain), add_missing=False)
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
                    cur[bias_exc] = BIAS_DRIVE; cur[bias_inh] = BIAS_DRIVE   # constant '+1' bias units
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

    sc = scale_c / max(1e-9, proj)
    best = 0; bestcfg = None
    for floor in floors:
        for ratio in ratios:
            for bgain in bgains:
                ok = 0
                for (toks, *_), hs in zip(corpus["test"], host):
                    for k in (0, 1, 2):
                        es = read(enc.encode(toks), k, floor, sc, ratio * sc, bgain)
                        ok += int(int(np.argmax(es)) == hs[k])
                if ok > best:
                    best = ok; bestcfg = (floor, ratio, bgain)
                print(f"seed {seed} floor {floor:4.0f} ratio {ratio:.2f} bgain {bgain:5.1f} (c{scale_c:.0f}): "
                      f"SYN-BIAS {ok}/18", flush=True)
    print(f"=== seed {seed}: BEST {best}/18 @ (floor,ratio,bgain) {bestcfg} (c{scale_c:.0f}) ===", flush=True)
    return best


if __name__ == "__main__":
    seeds = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["43"])]
    floors = [float(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["30"])]
    ratios = [float(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["1.0", "1.5"])]
    scale_c = float(sys.argv[4]) if len(sys.argv) > 4 else 90.0
    bgains = [float(x) for x in (sys.argv[5].split(",") if len(sys.argv) > 5 else ["2", "5", "10", "20"])]
    corpus = S.setup_corpus(seed=42)
    enc = Encoder(corpus["discovered"])
    for sd in seeds:
        t0 = time.time()
        run(sd, corpus, floors, ratios, scale_c, bgains, enc)
        print(f"[seed {sd} {time.time()-t0:.0f}s]", flush=True)
