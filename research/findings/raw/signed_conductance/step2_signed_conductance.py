"""DESIGN A -- the CONDUCTANCE-DOMAIN signed opponent (the design-workflow's #1). The signed subtraction happens in the
LINEAR pre-spike CURRENT domain (I_syn = g_e*(0-v) + g_i*(-75-v)), BEFORE the Izhikevich f-I nonlinearity, so ONE ensemble
per role fires monotonically in the true signed logit Ws@f and argmax(ens spikes) = argmax(Ws@f) -- preserved by the SAME
monotonicity that already makes the POSITIVE read 18/18 on 42/43. The spike-count OPPONENT plateaued at 9/18 because it
subtracts spike COUNTS (f(a)-f(b), nonlinear); this never subtracts spikes.

Why the prior RELAY failed and this fixes it: the relay was a SPIKING interneuron receiving Ws- (the weighted sum) and
THRESHOLDING it -> it delivered f_nonlin(Wn@f), re-inserting the nonlinearity. Here res_inh is a 1:1 FOLLOWER of the
reservoir (reservoir[i]->res_inh[i], unweighted -- a linear spike-relabel, NOT a weighted-sum threshold), trait-1 so its
synapses feed g_i; the Wn WEIGHTING is on the res_inh[i]->ens[r] synapses (AFTER the follower) -> g_i[r] ~ Wn[:,r]@f graded.

CLEAN layout (like the c2 runner, NO tangle): reservoir_n=RES_N (the LSM only); ens[0..2] (exc) + res_inh follower copy
(inh, RES_N) live on the role_wta slice. Key knob = the driving-force ratio scale_i/scale (~1.9 to compensate |E_e-v|/|E_i-v|
* prop/inh_prop). Sweeps seed 42 (the failing seed) at floor x ratio; then the winning FIXED point on 43/44 for 3/3.
"""
import os, sys, time
os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
from sim.backend import get_backend, to_host
import step1_onoff_opponent as S            # setup_corpus (compatible corpus)
import research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk as C
from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder, _gen, _TRAIN_KINDS
from research.runners._emerge61_spiking_broca_order_robustness_derisk import _restore_state
from research.runners.unified_brain_bridge import UnifiedBrainBridge
from research.runners._rungB1b_neural_role_wta_derisk import PROJ_DIM
from research.runners.core_sim_composition import RESET_STEPS

ENS_P = C.WTA_P_C2          # 80 exc neurons per role ensemble (ONE ensemble per role)
RES_N = C.RES_N            # 300 reservoir neurons
WS_REPLAY = 6
FOLLOW_W = 120.0           # reservoir[i] -> res_inh[i] follower weight (res_inh fires ~when reservoir[i] fires)
RES_BIAS = C.RES_BIAS
READ_T = C.READ_T_STEP_C2


def build(seed, corpus, enc):
    ub = UnifiedBrainBridge(seed=seed, proj_dim=PROJ_DIM, concepts=corpus["concepts"],
                            enable_synaptic_route=False, role_wta_n=3 * ENS_P + RES_N, reservoir_n=RES_N)
    res_idx, W_in = C.wire_reservoir(ub, enc.dim, seed)           # reservoir_n=RES_N -> res_idx is the clean LSM
    res = C.UBReservoir(ub, res_idx, W_in)
    rb = int(ub.role_wta_base)
    ens = [np.arange(rb + r * ENS_P, rb + (r + 1) * ENS_P, dtype=np.int64) for r in range(3)]
    res_inh = np.arange(rb + 3 * ENS_P, rb + 3 * ENS_P + RES_N, dtype=np.int64)
    ub.bridge.cp_traits[res_inh] = 1                              # inhibitory copy -> synapses feed g_i
    ub.bridge._cached_inhibitory_mask = None
    # 1:1 follower reservoir[i] -> res_inh[i] (excitatory, unweighted relabel)
    ub.bridge.set_pathway_weights("follow", [int(x) for x in res_idx], [int(x) for x in res_inh],
                                  np.full(RES_N, FOLLOW_W, np.float32), add_missing=True)
    return ub, res, res_idx, ens, res_inh


def _edges(src_idx, ens):
    pre, post = [], []
    for r in range(3):
        for e in ens[r]:
            for s in src_idx:
                pre.append(int(s)); post.append(int(e))
    return pre, post


def _weights(src_idx, ens, W_rows, scale):
    n = len(src_idx); w = []
    for r in range(3):
        col = W_rows[:n, r].astype(np.float64) * float(scale)
        for _e in ens[r]:
            for i in range(n):
                w.append(float(col[i]))
    return np.asarray(w, np.float32)


def run(seed, corpus, floors, ratios, scale_c, bscales, enc):
    rng = np.random.default_rng(seed * 101 + 5)
    train = _gen(_TRAIN_KINDS, C.N_TRAIN_PER, rng, corpus["subj"], corpus["verb"], corpus["obj"])
    ub, res, res_idx, ens, res_inh = build(seed, corpus, enc)
    n_res = len(res_idx)
    pe, po = _edges(res_idx, ens)          # reservoir -> ens (Wp, exc)
    ie, io = _edges(res_inh, ens)          # res_inh   -> ens (Wn, inh)
    ub.bridge.set_pathway_weights("res2ens_exc", pe, po, np.zeros(len(pe), np.float32), add_missing=True)
    ub.bridge.set_pathway_weights("res2ens_inh", ie, io, np.zeros(len(ie), np.float32), add_missing=True)
    res.snapshot_after_wiring()
    Ws = C._fit_Ws_spiking(res, enc, train)
    host = [S._host_signed_winners(res, enc, Ws, toks) for toks, *_ in corpus["test"]]
    proj = float((res.final_state(enc.encode(corpus["test"][0][0]))[:n_res] @ np.maximum(Ws[0][:n_res, :3], 0)).max())
    Wp = {k: np.maximum(Ws[k][:n_res, :3], 0.0) for k in Ws}
    Wn = {k: np.maximum(-Ws[k][:n_res, :3], 0.0) for k in Ws}
    bias = {k: Ws[k][n_res, :3].astype(np.float64) for k in Ws}     # the +1 signed bias intercept, per role
    b = ub.bridge; xp, _ = get_backend()

    def set_slot(k, sc, sci):
        b.set_pathway_weights("res2ens_exc", pe, po, _weights(res_idx, ens, Wp[k], sc), add_missing=False)
        b.set_pathway_weights("res2ens_inh", ie, io, _weights(res_inh, ens, Wn[k], sci), add_missing=False)

    def read(U, k, floor, bton):
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
                    cur[res_idx] = res.W_in @ U[t] + RES_BIAS      # res_inh gets NO external drive -> follows via synapse
                    for r in range(3):
                        cur[ens[r]] = floor + bton[r]              # + the signed bias intercept (per role, can be < 0)
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
            sci = ratio * sc
            for bscale in bscales:
                ok = 0
                for (toks, *_), hs in zip(corpus["test"], host):
                    for k in (0, 1, 2):
                        set_slot(k, sc, sci)
                        es = read(enc.encode(toks), k, floor, bias[k] * bscale)
                        ok += int(int(np.argmax(es)) == hs[k])
                if ok > best:
                    best = ok; bestcfg = (floor, ratio, bscale)
                print(f"seed {seed} floor {floor:4.0f} ratio {ratio:.2f} bscale {bscale:5.1f} (c{scale_c:.0f}): "
                      f"SIGNED-COND {ok}/18", flush=True)
    print(f"=== seed {seed}: BEST {best}/18 @ floor/ratio/bscale {bestcfg} (c{scale_c:.0f}) ===", flush=True)
    return best


if __name__ == "__main__":
    seeds = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["42"])]
    floors = [float(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["250"])]
    ratios = [float(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["1.9"])]
    scale_c = float(sys.argv[4]) if len(sys.argv) > 4 else 110.0
    bscales = [float(x) for x in (sys.argv[5].split(",") if len(sys.argv) > 5 else ["20", "40", "80", "160"])]
    corpus = S.setup_corpus(seed=42)
    enc = Encoder(corpus["discovered"])
    for sd in seeds:
        t0 = time.time()
        run(sd, corpus, floors, ratios, scale_c, bscales, enc)
        print(f"[seed {sd} {time.time()-t0:.0f}s]", flush=True)
