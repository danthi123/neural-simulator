"""Anti-cheats for the conductance-domain signed read-out (N_BIAS=6 clean-3/3 config). For each seed 42/43/44 at the
shared config, report host-agree for:
  (INTACT)      the full read                        -> expect 18/18.
  (SYN-LESION)  zero res2ens_exc + res2ens_inh       -> the ens get NO read-out signal -> collapse to ~chance (proves the
                                                        read is SYNAPTIC, not a host computation).
  (FOLLOW-LES)  zero the reservoir->res_inh follower  -> res_inh silent -> g_i=0 -> the read degrades to POSITIVE-only
                                                        (no neg subtraction) -> seed 44 drops toward the positive read's
                                                        11/18 (proves the inhibitory FOLLOWER / neg-weight info is the
                                                        load-bearing mechanism that resolves the degraded seed 44).
  (BIAS-LES)    zero the bias-unit synapses           -> no +1 intercept -> drops (the learned bias is load-bearing).
Source-clean holds by construction: the winner is argmax over ens FIRING (a neural read), never a host f@Ws/argmax."""
import os, sys, time
os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
from sim.backend import get_backend, to_host
import step1_onoff_opponent as S
import research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk as C
import step2d_bias_pop as M
from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder, _gen, _TRAIN_KINDS
from research.runners._emerge61_spiking_broca_order_robustness_derisk import _restore_state
from research.runners.core_sim_composition import RESET_STEPS

seeds = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["42", "43", "44"])]
FLOOR = 30.0; RATIO = 1.4; SCALE_C = 90.0; BGAIN = 6.0
corpus = S.setup_corpus(seed=42); test = corpus["test"]
enc = Encoder(corpus["discovered"])
xp, _ = get_backend()

for seed in seeds:
    t0 = time.time()
    rng = np.random.default_rng(seed * 101 + 5)
    train = _gen(_TRAIN_KINDS, C.N_TRAIN_PER, rng, corpus["subj"], corpus["verb"], corpus["obj"])
    ub, res, res_idx, ens, res_inh, bias_exc, bias_inh, exc_src, inh_src = M.build(seed, corpus, enc)
    n_res = len(res_idx)
    pe, po = M._edges(exc_src, ens); ie, io = M._edges(inh_src, ens)
    ub.bridge.set_pathway_weights("res2ens_exc", pe, po, np.zeros(len(pe), np.float32), add_missing=True)
    ub.bridge.set_pathway_weights("res2ens_inh", ie, io, np.zeros(len(ie), np.float32), add_missing=True)
    res.snapshot_after_wiring()
    Ws = C._fit_Ws_spiking(res, enc, train)
    host = [S._host_signed_winners(res, enc, Ws, toks) for toks, *_ in corpus["test"]]
    proj = float((res.final_state(enc.encode(corpus["test"][0][0]))[:n_res] @ np.maximum(Ws[0][:n_res, :3], 0)).max())
    Wp = {k: np.maximum(Ws[k][:n_res + 1, :3], 0.0) for k in Ws}
    Wn = {k: np.maximum(-Ws[k][:n_res + 1, :3], 0.0) for k in Ws}
    b = ub.bridge; sc = SCALE_C / max(1e-9, proj); sci = RATIO * sc
    follow_pre = [int(x) for x in res_idx]; follow_post = [int(x) for x in res_inh]

    def read(U, k, mode):
        wp = M._weights(exc_src, ens, Wp[k], n_res, sc, BGAIN)
        wn = M._weights(inh_src, ens, Wn[k], n_res, sci, BGAIN)
        if mode == "syn":                       # zero BOTH read-out pathways
            wp = np.zeros_like(wp); wn = np.zeros_like(wn)
        if mode in ("bias", "bare"):            # zero only the bias-unit columns (last N_BIAS sources per role)
            wp = M._weights(exc_src, ens, Wp[k], n_res, sc, 0.0); wn = M._weights(inh_src, ens, Wn[k], n_res, sci, 0.0)
        b.set_pathway_weights("res2ens_exc", pe, po, wp, add_missing=False)
        b.set_pathway_weights("res2ens_inh", ie, io, wn, add_missing=False)
        if mode in ("follow", "bare"):          # zero the reservoir->res_inh follower (res_inh goes silent)
            b.set_pathway_weights("follow", follow_pre, follow_post, np.zeros(n_res, np.float32), add_missing=False)
        else:
            b.set_pathway_weights("follow", follow_pre, follow_post, np.full(n_res, M.FOLLOW_W, np.float32), add_missing=False)
        _restore_state(b, res._snap)
        pou, ph = b.core_config.enable_ou_process, b.core_config.enable_hebbian_learning
        b.core_config.enable_ou_process = False; b.core_config.enable_hebbian_learning = False
        es = np.zeros(3)
        try:
            for _ in range(RESET_STEPS):
                b.runtime_state.current_time_ms += b.core_config.dt_ms; b._run_one_simulation_step()
            for _rep in range(M.WS_REPLAY):
                for t in range(len(U)):
                    cur = np.zeros(b.core_config.num_neurons)
                    cur[res_idx] = res.W_in @ U[t] + M.RES_BIAS
                    cur[bias_exc] = M.BIAS_DRIVE; cur[bias_inh] = M.BIAS_DRIVE
                    for r in range(3):
                        cur[ens[r]] = FLOOR
                    b.cp_external_input_current[:] = xp.asarray(cur.astype(np.float32))
                    for _ in range(M.READ_T):
                        b.runtime_state.current_time_ms += b.core_config.dt_ms; b._run_one_simulation_step()
                        fs = np.asarray(to_host(b.cp_firing_states)).astype(np.float64)
                        for r in range(3):
                            es[r] += fs[ens[r]].sum()
        finally:
            b.cp_external_input_current[:] = 0.0
            b.core_config.enable_ou_process = pou; b.core_config.enable_hebbian_learning = ph
        return es

    out = {}
    for mode in ("intact", "syn", "follow", "bias", "bare"):
        ok = 0
        for (toks, *_), hs in zip(test, host):
            for k in (0, 1, 2):
                es = read(enc.encode(toks), k, mode)
                ok += int(int(np.argmax(es)) == hs[k])
        out[mode] = ok
    print(f"seed {seed}: INTACT {out['intact']}/18 | SYN-LES {out['syn']}/18 | FOLLOW-LES {out['follow']}/18 | "
          f"BIAS-LES {out['bias']}/18 | BARE(Wp-only) {out['bare']}/18  [{time.time()-t0:.0f}s]", flush=True)
