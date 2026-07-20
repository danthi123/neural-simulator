"""gap#1 M2 (research-gate ranked #2) — deliver the WKV input `v_t` via a GENUINE SPIKING POPULATION whose SYNAPSES
perform the decode, feeding the graded `cp_ssm_state` integrator that M1 proved beats the fair trigram (6-seed GO).

WHY: M1 closed "a graded multi-channel recurrent LM state runs on the bridge and beats the trigram", but its per-token
`cp_ssm_inject` was written by the HOST (standing in for the upstream cortical population's graded synaptic drive).
M2 closes that residual: an NEF (Eliasmith-Anderson) input population with HETEROGENEOUS encoders — distributed
INTERCEPTS (tile the range -> kills the dead-zone), MIXED-SIGN preferred directions, distributed gains — projects to the
state channel through synapses whose weights ARE the OPTIMAL least-squares DECODER. The postsynaptic conductance is then
literally `sum_i d_i * spikes_i` = the decoded value: the decode happens IN THE SYNAPSES, not in host code.
Off-bridge de-risk (2026-07-20): hetero+optimal-decode recovers v at corr 0.9993 vs the old homogeneous+uniform-sum
path's 0.8167 with 36/40 DEAD steps.

TIMING (the one subtlety): the bridge's SSM block advances EVERY step, so the encode window would decay the state
T_STEP times. We freeze it during the encode via the shipped shunt: `lam = clip(1 - k_leak*(1+shunt), 0, 1)`, so
`shunt = -1` -> `lam = 1` -> `s = s` (frozen, and (1-lam)=0 so inject is ignored). Then ONE step at `shunt = 0`
(`lam = decay`) with `inject = v_hat/(1-decay)` applies EXACTLY `a_t = decay*a_{t-1} + v_hat_t`.

GATE: on-bridge deep-NLL (d10-99) BEATS the fair interpolated trigram, with (a) verify-first corr(state, numpy) high,
(b) a MEMORYLESS control that collapses, (c) a HOMOGENEOUS-encoder control that reproduces the dead-zoned failure
(proving the heterogeneity+optimal decode is load-bearing). NO `sim/` edit (drives + reads public arrays).

Run: SIM_BACKEND=cupy python -m research.runners._emerge_wkv_m2_nef_onbridge_derisk --ssm <ssm>_seed42.npz --seed 42
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import argparse, json, math, time
from collections import defaultdict
import numpy as np

from research.runners._emerge_reservoir_lm_derisk import Vocab, fit_bigram
from research.runners._emerge_reservoir_lm_realcorpus_derisk import load_sentences
from research.runners._emerge_reservoir_lm_context_depth_derisk import _bucket
from research.runners._emerge_wkv_lm_derisk import fit_interp_trigram


def build_nef_ssm_bridge(D, seed, decay, n_enc, dec_p, dec_m, dt=1.0, pool_density=0.05):
    """pool (D*n_enc NEF encoders) --[decoder weights AS SYNAPSES]--> chan (2D graded SSM-state neurons)."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.enable_selective_ssm_state = True
    cfg.ssm_k_leak = float(max(0.0, min(1.0, 1.0 - decay)))
    cfg.dt = float(dt)
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False; cfg.enable_stdp = False; cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False; cfg.enable_short_term_plasticity = False
    cfg.enable_parameter_heterogeneity = False; cfg.enable_conductance_noise = False
    cfg.brain_regions = [
        BrainRegion(name="pool", n_neurons=D * n_enc, exc_fraction=1.0, internal_density=float(pool_density),
                    exc_weight_mean=1.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
        BrainRegion(name="chan", n_neurons=2 * D, exc_fraction=1.0, internal_density=0.05,
                    exc_weight_mean=1.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
    ]
    cfg.region_pathways = []
    rt = RuntimeState(); rt.actual_seed_used = seed
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt, gpu_config=GPUConfig())
    b._initialize_simulation_data()
    pool = np.asarray(b.region_manager.indices("pool")); chan = np.asarray(b.region_manager.indices("chan"))
    pool_groups = [pool[c * n_enc:(c + 1) * n_enc] for c in range(D)]
    # THE DECODE IS THE SYNAPSE: pool_c -> chan_ON[c] with weights dec_p, pool_c -> chan_OFF[c] with dec_m.
    pre, post, w = [], [], []
    for c in range(D):
        for j, pj in enumerate(pool_groups[c]):
            pre.append(int(pj)); post.append(int(chan[c]));     w.append(float(dec_p[j]))
            pre.append(int(pj)); post.append(int(chan[D + c])); w.append(float(dec_m[j]))
    b.inject_explicit_wiring({"nef_decode": {"pre_indices": pre, "post_indices": post,
                                             "initial_weights": w, "plastic": False, "conn_type": "MIXED"}})
    return b, pool_groups, chan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssm", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-eval", dest="n_eval", type=int, default=300)
    ap.add_argument("--n-sentences", dest="n_sentences", type=int, default=40000)
    ap.add_argument("--corpus", type=str, default="data/corpus/tinystories_train.txt")
    ap.add_argument("--n-enc", dest="n_enc", type=int, default=48, help="NEF encoders per channel")
    ap.add_argument("--pool-density", dest="pool_density", type=float, default=0.05, help="pool internal recurrent density; NEF encoders should be INDEPENDENT (~0)")
    ap.add_argument("--t-step", dest="t_step", type=int, default=6, help="encode-window bridge steps per token")
    ap.add_argument("--memoryless", action="store_true", help="anti-cheat: lam=0 (no integration) -> ~bigram")
    ap.add_argument("--homogeneous", action="store_true", help="anti-cheat: homogeneous encoders + uniform decode (must fail)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from sim.backend import to_host, get_backend
    xp, _bk = get_backend()

    W = np.load(args.ssm, allow_pickle=True)
    V = int(W["V"]); D = int(W["d_model"])
    emb = W["emb.weight"]; ln_w = W["ln.weight"]; ln_b = W["ln.bias"]
    Wv = W["Wv.weight"]; Wr = W["Wr.weight"]; Wo_sp = W["Wo_sp.weight"]
    head_w = W["head.weight"]; head_b = W["head.bias"]
    decay = float(np.exp(-np.log1p(np.exp(W["w"][0]))))
    dec_eff = 0.0 if args.memoryless else decay

    def _ln(v):
        m = v.mean(); s = v.std() + 1e-5
        return (v - m) / s * ln_w + ln_b

    sents = load_sentences(args.corpus, args.n_sentences)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(sents)); cut = int(0.85 * len(sents))
    tr = [sents[i] for i in idx[:cut]][:30000]; ev = [sents[i] for i in idx[cut:]][:args.n_eval]
    vocab = Vocab.build(tr, V=V); tr_ids = [vocab.ids(s) for s in tr]; ev_ids = [vocab.ids(s) for s in ev]
    P_bi = fit_bigram(tr_ids, V)
    tri, _l = fit_interp_trigram(tr_ids, V, [vocab.ids(s) for s in tr[-1500:]])

    # ---- NEF encoder params (heterogeneous vs the homogeneous control) ----
    er = np.random.default_rng(args.seed + 991)
    N = args.n_enc
    if args.homogeneous:
        x_int = np.zeros(N); sgn = np.ones(N); gain = np.full(N, 220.0)
    else:
        x_int = er.uniform(-1.0, 1.0, N); sgn = er.choice([-1.0, 1.0], N); gain = er.uniform(120.0, 320.0, N)

    # ---- calibrate: measure REAL tuning curves on a scratch pool, solve the optimal decoders ----
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion
    ccfg = CoreSimConfig(); ccfg.enable_brain_region_framework = True; ccfg.dt = 1.0
    ccfg.seed = ccfg.ou_seed = ccfg.heterogeneity_seed = args.seed
    ccfg.enable_ou_process = False; ccfg.enable_stdp = False; ccfg.enable_hebbian_learning = False
    ccfg.enable_homeostasis = False; ccfg.enable_short_term_plasticity = False
    ccfg.enable_parameter_heterogeneity = False; ccfg.enable_conductance_noise = False
    ccfg.brain_regions = [BrainRegion(name="pool", n_neurons=N, exc_fraction=1.0, internal_density=float(args.pool_density),
                                      exc_weight_mean=1.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False)]
    ccfg.region_pathways = []
    crt = RuntimeState(); crt.actual_seed_used = args.seed
    cb = SimulationBridge(core_config=ccfg, viz_config=VisualizationConfig(), runtime_state=crt, gpu_config=GPUConfig())
    cb._initialize_simulation_data()
    cidx = np.asarray(cb.region_manager.indices("pool")); cn = int(cb.core_config.num_neurons)

    def pool_rates(vv, bridge, ids_, nn):
        drive = gain * sgn * (vv - x_int) if not args.homogeneous else gain * vv
        cur = np.zeros(nn, np.float32); cur[ids_] = np.maximum(drive, 0.0).astype(np.float32)
        bridge.cp_membrane_potential_v[:] = -65.0; bridge.cp_recovery_variable_u[:] = 0.0
        bridge.cp_firing_states[:] = 0.0
        cnt = np.zeros(len(ids_))
        _dec_e = math.exp(-float(bridge.core_config.dt_ms) / float(bridge.core_config.syn_tau_g_e))
        for _ in range(args.t_step):
            bridge.cp_external_input_current[:] = 0.0
            bridge.cp_external_input_current[ids_] = (xp.asarray(cur[ids_]) if xp is not None else cur[ids_])
            bridge._run_one_simulation_step()
            # SAME recursion as cp_conductance_g_e (g = g*decay_e + input) => the decoder is fit on the deployed basis
            cnt = cnt * _dec_e + np.asarray(to_host(bridge.cp_firing_states))[ids_]
        return cnt

    vs = np.linspace(-2.5, 2.5, 51)
    R = np.stack([pool_rates(v, cb, cidx, cn) for v in vs], 0)            # [n_v, N]
    tgt_p = np.maximum(vs, 0.0); tgt_m = np.maximum(-vs, 0.0)
    if args.homogeneous:
        dec_p = np.ones(N) / N; dec_m = np.ones(N) / N                   # uniform-sum decode (the OLD path)
    else:
        # DALE'S LAW: the bridge routes exc/inh per PRESYNAPTIC NEURON, so a mixed-sign decoder on one pool is not
        # expressible on the substrate. Solve a NON-NEGATIVE (sign-constrained) least-squares decoder instead -- the
        # +v-preferring encoders carry relu(+v), the -v-preferring carry relu(-v); NNLS zeroes the wrong-sign ones.
        from scipy.optimize import nnls
        Rr = np.concatenate([R, 1e-2 * np.eye(N)], 0)                    # tiny ridge rows for conditioning
        dec_p = nnls(Rr, np.concatenate([tgt_p, np.zeros(N)]))[0]
        dec_m = nnls(Rr, np.concatenate([tgt_m, np.zeros(N)]))[0]
    vhat_p = R @ dec_p
    print(f"[calib] decoder fit: corr(relu(+v)_hat, relu(+v)) = {np.corrcoef(vhat_p, tgt_p)[0,1]:.4f} "
          f"({'HOMOGENEOUS control' if args.homogeneous else 'NEF heterogeneous'}, N={N})", flush=True)
    del cb

    # ---- the real bridge: pool --[decoder synapses]--> chan(graded SSM state) ----
    b, pool_groups, chan = build_nef_ssm_bridge(D, args.seed, dec_eff, N, dec_p, dec_m,
                                               pool_density=args.pool_density)
    nn = int(b.core_config.num_neurons)
    pool_all = np.concatenate(pool_groups).astype(np.int64)
    pool_chan_of = np.concatenate([[c] * len(pool_groups[c]) for c in range(D)]).astype(np.int64)
    enc_of = np.concatenate([np.arange(len(pool_groups[c])) for c in range(D)]).astype(np.int64)
    scale = 1.0 / max(1e-6, (1.0 - dec_eff))

    def onbridge_states(ids):
        for nm in ("cp_ssm_state", "cp_ssm_inject", "cp_ssm_shunt", "cp_conductance_g_e",
                   "cp_conductance_g_i", "cp_firing_states"):
            arr = getattr(b, nm, None)
            if arr is not None: arr[:] = 0.0
        b.cp_membrane_potential_v[:] = -65.0; b.cp_recovery_variable_u[:] = 0.0
        out = []
        for t in range(len(ids)):
            h = _ln(emb[ids[t]]); v = Wv @ h                              # [D] the per-channel value to deliver
            # (1) ENCODE window: freeze the state (shunt=-1 => lam=1) while the NEF pool spikes through the decoder synapses
            b.cp_ssm_shunt[:] = -1.0
            drv = (gain[enc_of] * sgn[enc_of] * (v[pool_chan_of] - x_int[enc_of]) if not args.homogeneous
                   else gain[enc_of] * v[pool_chan_of])
            cur = np.zeros(nn, np.float32); cur[pool_all] = np.maximum(drv, 0.0).astype(np.float32)
            # zero the FAST synaptic conductances so g_e reads THIS token's decoded value (the MEMORY lives in the
            # slow cp_ssm_state, not in g_e) -- otherwise the read mixes the new input with the decay of the old.
            b.cp_conductance_g_e[:] = 0.0; b.cp_conductance_g_i[:] = 0.0
            for _ in range(args.t_step):
                b.cp_external_input_current[:] = 0.0
                b.cp_external_input_current[pool_all] = (xp.asarray(cur[pool_all]) if xp is not None else cur[pool_all])
                b._run_one_simulation_step()
            # (2) READ the SYNAPTICALLY-DECODED value off the postsynaptic conductances (exc - inh = what the membrane sums)
            ge = np.asarray(to_host(b.cp_conductance_g_e)).astype(np.float64)[chan]
            gi = np.asarray(to_host(b.cp_conductance_g_i)).astype(np.float64)[chan]
            vhat = ge - gi
            # (3) ONE state step at shunt=0 (lam=decay) with inject=vhat/(1-decay) => exactly a_t = decay*a_{t-1} + vhat_t
            inj = np.zeros(nn, np.float32); inj[chan] = (vhat * scale).astype(np.float32)
            b.cp_ssm_inject[:] = (xp.asarray(inj) if xp is not None else inj)
            b.cp_ssm_shunt[:] = 0.0
            b.cp_external_input_current[:] = 0.0
            b._run_one_simulation_step()
            out.append(np.asarray(to_host(b.cp_ssm_state)).astype(np.float64)[chan])
        return np.asarray(out)                                             # [T, 2D]

    def ref_states(ids):
        ap = np.zeros(D); an = np.zeros(D); out = []
        for t in range(len(ids)):
            v = Wv @ _ln(emb[ids[t]])
            ap = dec_eff * ap + np.maximum(v, 0.0); an = dec_eff * an + np.maximum(-v, 0.0)
            out.append(np.concatenate([ap, an]))
        return np.asarray(out)

    # the state is a synaptically-decoded (hence SCALED) version of the reference -> rescale per channel before the
    # trained read-out (a fixed gain a downstream synapse would absorb; fit on a few sentences, NOT on the eval set).
    Xs, Ys = [], []
    for ids in tr_ids[:30]:
        if len(ids) < 3: continue
        Xs.append(onbridge_states(ids)); Ys.append(ref_states(ids))
    Xs = np.concatenate(Xs, 0); Ys = np.concatenate(Ys, 0)
    gains = (Xs * Ys).sum(0) / np.maximum((Xs * Xs).sum(0), 1e-12)
    gains = np.where(np.isfinite(gains) & (np.abs(gains) > 1e-9), gains, 1.0)
    _res = Xs * gains - Ys
    _sig = float(np.std(_res) / max(1e-9, np.std(Ys)))
    print(f"[m3-calib] spiking-delivery RESIDUAL noise after per-channel gain: sigma_rel = {_sig:.3f} "
          f"(this is the noise level an end-to-end co-adaptation must train against = M3 input)", flush=True)
    # VERIFY-FIRST (post-rescale = the state that ACTUALLY feeds the read-out), on HELD-OUT eval sentences.
    cs = []
    for ids in ev_ids[:5]:
        if len(ids) < 4: continue
        ob = onbridge_states(ids) * gains; rf = ref_states(ids)
        cs.append(np.corrcoef(ob.flatten(), rf.flatten())[0, 1])
    mapcorr = float(np.nanmean(cs)) if cs else float("nan")
    print(f"[verify] corr(on-bridge SSM state via SPIKING NEF input, numpy ref) POST-rescale = {mapcorr:.3f} "
          f"(gains fit on TRAIN sentences only)", flush=True)

    ce = defaultdict(float); bce = defaultdict(float); tce = defaultdict(float); cnt = defaultdict(int)
    for ids in ev_ids:
        if len(ids) < 2: continue
        st = onbridge_states(ids) * gains
        for t in range(len(ids) - 1):
            rh = 1.0 / (1.0 + np.exp(-(Wr @ _ln(emb[ids[t]]))))
            lg = head_w @ (rh * (Wo_sp @ st[t])) + head_b
            lg = lg - lg.max(); p = np.exp(lg); p = p / p.sum()
            d = t + 1; bk = _bucket(d)
            ce[bk] += -math.log(max(p[ids[t+1]], 1e-12))
            bce[bk] += -math.log(max(P_bi[ids[t], ids[t+1]], 1e-12))
            u = ids[t-1] if t >= 1 else -1
            tce[bk] += -math.log(max(tri(u, ids[t], ids[t+1]), 1e-12))
            cnt[bk] += 1
    res = {}
    for bk in ["1", "2", "3", "4-5", "6-9", "10-99"]:
        if cnt[bk] == 0: continue
        o, bg, tg = ce[bk]/cnt[bk], bce[bk]/cnt[bk], tce[bk]/cnt[bk]
        res[bk] = {"onbridge": o, "bigram": bg, "trigram": tg, "vs_trigram": tg - o, "n": cnt[bk]}
        print(f"    depth {bk:>5} (n={cnt[bk]:5d}): onbridge {o:.3f} | bigram {bg:.3f} | trigram {tg:.3f} "
              f"|| vs-trigram {tg-o:+.3f}", flush=True)
    deep = res.get("10-99", {}).get("vs_trigram", float("nan"))
    print(f"    VERDICT: {'GO' if deep > 0 else 'no-go'} (deep vs-trigram {deep:+.3f}, map-corr {mapcorr:.3f})", flush=True)
    if args.json:
        json.dump({"seed": args.seed, "deep_vs_trigram": deep, "map_corr": mapcorr, "per_depth": res,
                   "n_enc": N, "homogeneous": bool(args.homogeneous), "memoryless": bool(args.memoryless)},
                  open(args.json, "w"), indent=2)


if __name__ == "__main__":
    main()
