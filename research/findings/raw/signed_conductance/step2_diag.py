"""Diagnose the conductance-domain signed opponent 0/18: for fact 0, print per-slot host winner, ens firing WITH
inhibition (signed) vs WITHOUT (res_inh weights zeroed = positive-only control), and the res_inh follower's total
firing (is the follower even firing?). Tells apart: (a) follower dead -> should == positive; (b) over-inhibited ->
ens silenced; (c) mis-ordered."""
import os, sys
os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
from sim.backend import get_backend, to_host
import step1_onoff_opponent as S
import research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk as C
import step2_signed_conductance as M
from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder, _gen, _TRAIN_KINDS
from research.runners._emerge61_spiking_broca_order_robustness_derisk import _restore_state
from research.runners.core_sim_composition import RESET_STEPS

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
floor = float(sys.argv[2]) if len(sys.argv) > 2 else 250.0
ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 1.9
scale_c = float(sys.argv[4]) if len(sys.argv) > 4 else 110.0
BSCALE = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
corpus = S.setup_corpus(seed=42)
enc = Encoder(corpus["discovered"])
rng = np.random.default_rng(seed * 101 + 5)
train = _gen(_TRAIN_KINDS, C.N_TRAIN_PER, rng, corpus["subj"], corpus["verb"], corpus["obj"])
ub, res, res_idx, ens, res_inh = M.build(seed, corpus, enc)
n_res = len(res_idx)
pe, po = M._edges(res_idx, ens); ie, io = M._edges(res_inh, ens)
ub.bridge.set_pathway_weights("res2ens_exc", pe, po, np.zeros(len(pe), np.float32), add_missing=True)
ub.bridge.set_pathway_weights("res2ens_inh", ie, io, np.zeros(len(ie), np.float32), add_missing=True)
res.snapshot_after_wiring()
Ws = C._fit_Ws_spiking(res, enc, train)
host = [S._host_signed_winners(res, enc, Ws, toks) for toks, *_ in corpus["test"]]
proj = float((res.final_state(enc.encode(corpus["test"][0][0]))[:n_res] @ np.maximum(Ws[0][:n_res, :3], 0)).max())
Wp = {k: np.maximum(Ws[k][:n_res, :3], 0.0) for k in Ws}
Wn = {k: np.maximum(-Ws[k][:n_res, :3], 0.0) for k in Ws}
b = ub.bridge; xp, _ = get_backend()
sc = scale_c / max(1e-9, proj); sci = ratio * sc
print(f"seed {seed} floor {floor} ratio {ratio} sc {sc:.3f} sci {sci:.3f} proj {proj:.3f}", flush=True)

# ---- LINEAR analysis: is the bias load-bearing, or is the mechanism mis-balanced? ----
for si in (0, 1):
    toks = corpus["test"][si][0]; hs = host[si]
    f = res.final_state(enc.encode(toks))
    for k in (0, 1, 2):
        wp = f @ Wp[k]; wn = f @ Wn[k]; rows = f @ Ws[k][:n_res, :3]; bi = Ws[k][n_res, :3]
        full = rows + bi
        print(f" LIN fact{si} slot{k} host={hs[k]} | Wp@f={np.round(wp,2)} Wn@f={np.round(wn,2)} "
              f"(Wp-Wn)@f={np.round(rows,2)} amx_rows={int(np.argmax(rows))} | bias={np.round(bi,2)} "
              f"| full={np.round(full,2)} amx_full={int(np.argmax(full))}", flush=True)


def read(U, k, inhib):
    b.set_pathway_weights("res2ens_exc", pe, po, M._weights(res_idx, ens, Wp[k], sc), add_missing=False)
    b.set_pathway_weights("res2ens_inh", ie, io,
                          M._weights(res_inh, ens, Wn[k], sci if inhib else 0.0), add_missing=False)
    _restore_state(b, res._snap)
    pou, ph = b.core_config.enable_ou_process, b.core_config.enable_hebbian_learning
    b.core_config.enable_ou_process = False; b.core_config.enable_hebbian_learning = False
    es = np.zeros(3); rinh = 0.0
    for _ in range(RESET_STEPS):
        b.runtime_state.current_time_ms += b.core_config.dt_ms; b._run_one_simulation_step()
    for _rep in range(M.WS_REPLAY):
        for t in range(len(U)):
            cur = np.zeros(b.core_config.num_neurons)
            cur[res_idx] = res.W_in @ U[t] + M.RES_BIAS
            for r in range(3):
                cur[ens[r]] = floor + Ws[k][n_res, r] * BSCALE
            b.cp_external_input_current[:] = xp.asarray(cur.astype(np.float32))
            for _ in range(M.READ_T):
                b.runtime_state.current_time_ms += b.core_config.dt_ms; b._run_one_simulation_step()
                fs = np.asarray(to_host(b.cp_firing_states)).astype(np.float64)
                for r in range(3):
                    es[r] += fs[ens[r]].sum()
                rinh += fs[res_inh].sum()
    b.cp_external_input_current[:] = 0.0
    b.core_config.enable_ou_process = pou; b.core_config.enable_hebbian_learning = ph
    return es, rinh


ok = 0
for si in range(len(corpus["test"])):
    toks = corpus["test"][si][0]; hs = host[si]
    for k in (0, 1, 2):
        es_s, rinh = read(enc.encode(toks), k, True)
        amx = int(np.argmax(es_s)); ok += int(amx == hs[k])
        tag = "" if amx == hs[k] else "  <-- WRONG"
        print(f" fact{si} slot{k} host={hs[k]} | SIGNED es={np.round(es_s,0)} amx={amx}{tag}", flush=True)
print(f"=== seed {seed} bscale {BSCALE}: {ok}/18 ===", flush=True)
