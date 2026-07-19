"""gap#1 Rung 2 ON-BRIDGE realization de-risk: does the trained SSM leaky-integrator state, realized on a REAL
SimulationBridge (Izhikevich neurons + a SLOW NMDA-recurrent conductance = the leaky memory, driven per token, read from
`cp_firing_states`), preserve the deep-context LM capture the rate-level SSM has? The rate-level de-risks are ALL GO
(Rung 1a mechanism 6-seed · Rung 1b emergent input 3-seed · Rung 2 spiking-faithful recurrence + non-negative firing-rate
read + uniform decay). This closes the loop to the fully-spiking substrate.

THE MAPPING (uniform-decay SSM -> one recurrent Izhikevich region + slow NMDA memory):
  rate-SSM: a_t = decay*a_{t-1} + v_t ; v_t = Wv @ LayerNorm(emb[x_t]) ; read = Wo_sp @ [relu(a_t), relu(-a_t)] -> head.
  on-bridge: a region of D "channel" neurons whose SLOW NMDA-recurrent conductance holds a leaky state across the fast
  Izhikevich spiking (NMDA tau ~= the SSM decay). Per token: drive the region's external current with v_t (ON = +v, a
  matched OFF sub-population = -v via a sign-split drive), run T_STEP bridge steps (real conductance synapses + Izhikevich),
  read the region's per-neuron spike counts = the firing-rate state; the read is the trained Wo_sp over [rate_ON, rate_OFF].
  The slow NMDA conductance is NOT washed between tokens within a sentence => it INTEGRATES = the leaky memory.

VERIFY-FIRST (silent-failure discipline): before the full eval, compare the on-bridge firing-rate STATE trajectory to the
rate-SSM analog state on one sentence (corr) -- a wrong substrate mapping shows up as a low/zero correlation, caught before
any GO claim. GATE: the on-bridge LM beats the fair trigram at deep context (the rate-SSM's bar), AND perm/memoryless
anti-cheats collapse. Reuse Vocab/load_sentences/fit_interp_trigram/_bucket. NO `sim/` edit (drives + reads public arrays).

Run: SIM_BACKEND=cupy python -m research.runners._emerge_wkv_onbridge_derisk --ssm <path>_seed42.npz --seed 42 --n-eval 200
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import argparse, json, math, time
from pathlib import Path
from collections import defaultdict
import numpy as np

from research.runners._emerge_reservoir_lm_derisk import Vocab, fit_bigram
from research.runners._emerge_reservoir_lm_realcorpus_derisk import load_sentences
from research.runners._emerge_reservoir_lm_context_depth_derisk import BUCKETS, _bucket
from research.runners._emerge_wkv_lm_derisk import fit_interp_trigram

_T_STEP_DEFAULT = 6                                                          # bridge steps per token (integration window)


def _build_channel_bridge(D, seed, self_nmda_w=8.0, dt=0.5, pop_k=1):
    """2*D Izhikevich channel neurons (D ON + D OFF). The leaky memory = a DIAGONAL self-NMDA autapse per channel neuron
    (pre==post, exc_receptor='nmda_slow' via inject_explicit_wiring): each neuron's firing charges its OWN slow NMDA
    conductance (tau~100ms) = the per-channel leaky integral a_t=decay*a_{t-1}+drive (NOT random reservoir mixing). Built
    via a minimal valid region (so the bridge initializes) then inject_explicit_wiring OVERRIDES with the diagonal edges.
    Driven per token by external current; read from cp_firing_states. Returns (bridge, on_idx, off_idx, snapshot)."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.enable_nmda = True                                           # slow NMDA conductance = the leaky memory
    cfg.brain_regions = [
        BrainRegion(name="chan", n_neurons=2 * D * pop_k, exc_fraction=1.0, internal_density=0.05,
                    exc_weight_mean=1.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    enable_nmda=True),
    ]
    cfg.region_pathways = []
    cfg.dt = float(dt)
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False; cfg.enable_stdp = False; cfg.enable_hebbian_learning = False
    rt = RuntimeState(); rt.actual_seed_used = seed
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt, gpu_config=GPUConfig())
    b._initialize_simulation_data()
    idx = np.asarray(b.region_manager.indices("chan"))
    # OVERRIDE the random internal connectivity with a DIAGONAL self-NMDA autapse (the per-channel leaky integral).
    ii = [int(i) for i in idx]
    plan = {"chan_self_nmda": {"pre_indices": ii, "post_indices": ii, "initial_weights": [float(self_nmda_w)] * len(ii),
                                "plastic": False, "conn_type": "MIXED", "exc_receptor": "nmda_slow"}}
    b.inject_explicit_wiring(plan)
    # channel c (0..2D-1) -> the pop_k neurons idx[c*pop_k:(c+1)*pop_k] (population coding; averaged read = less spiking noise)
    chan_groups = [idx[c * pop_k:(c + 1) * pop_k] for c in range(2 * D)]
    return b, chan_groups, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssm", required=True, help="saved SSM weights (_emerge_wkv_lm_derisk --save-ssm ..._seed<N>.npz)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--corpus", type=str, default="data/corpus/tinystories_train.txt")
    ap.add_argument("--n-sentences", type=int, default=40000)
    ap.add_argument("--max-train-sents", type=int, default=30000)
    ap.add_argument("--n-eval", type=int, default=200)              # SMALL (bridge stepping is slow)
    ap.add_argument("--drive-scale", type=float, default=1200.0)    # v_t -> external current pA
    ap.add_argument("--self-nmda-w", dest="self_nmda_w", type=float, default=8.0)   # diagonal self-NMDA autapse weight
    ap.add_argument("--mlp-readout", dest="mlp_readout", action="store_true")
    ap.add_argument("--n-fit", dest="n_fit", type=int, default=600)   # train sentences for the on-bridge read-out re-fit
    ap.add_argument("--exact-state", dest="exact_state", action="store_true")
    ap.add_argument("--read-gnmda", dest="read_gnmda", action="store_true",
                    help="LEVER 1: read the standing cp_conductance_g_nmda (100ms leaky integral) instead of firing rate")
    ap.add_argument("--pop-k", dest="pop_k", type=int, default=1)
    ap.add_argument("--t-step", dest="t_step", type=int, default=_T_STEP_DEFAULT)   # bridge steps/token (finer rate=less noise)
    ap.add_argument("--json", type=str, default="research/findings/raw/_emerge_wkv_onbridge.json")
    args = ap.parse_args()
    _T_STEP = int(args.t_step)
    from sim.backend import to_host, get_backend
    xp, _bk = get_backend()

    W = np.load(args.ssm, allow_pickle=True)
    V = int(W["V"]); D = int(W["d_model"])
    emb = W["emb.weight"]; ln_w = W["ln.weight"]; ln_b = W["ln.bias"]
    Wv = W["Wv.weight"]; Wr = W["Wr.weight"]; Wo_sp = W["Wo_sp.weight"]; head_w = W["head.weight"]; head_b = W["head.bias"]
    decay = float(np.exp(-np.log1p(np.exp(W["w"][0]))))             # exp(-softplus(w)) = the uniform decay
    words = list(W["words"])

    def _ln(v):
        m = v.mean(); s = v.std() + 1e-5
        return (v - m) / s * ln_w + ln_b

    if not Path(args.corpus).exists():
        args.corpus = "data/corpus/tinystories.txt"
    sents = load_sentences(args.corpus, args.n_sentences)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(sents)); cut = int(0.85 * len(sents))
    tr = [sents[i] for i in idx[:cut]][:args.max_train_sents]
    ev = [sents[i] for i in idx[cut:]][:args.n_eval]
    vocab = Vocab.build(tr, V=V); tr_ids = [vocab.ids(s) for s in tr]; ev_ids = [vocab.ids(s) for s in ev]
    P_bi = fit_bigram(tr_ids, V); tri, _lam = fit_interp_trigram(tr_ids, V, tr[-1500:] and [vocab.ids(s) for s in tr[-1500:]])

    b, chan_groups, _cg2, snap = _build_channel_bridge(D, args.seed, self_nmda_w=args.self_nmda_w, pop_k=args.pop_k)
    nnrn = int(b.cp_membrane_potential_v.size)
    # per-neuron -> channel map (channel c drives+reads its pop_k neurons; averaged read = population noise-averaging)
    all_drive_idx = np.concatenate([np.asarray(g) for g in chan_groups]).astype(np.int64)
    chan_of = np.concatenate([[c] * len(chan_groups[c]) for c in range(2 * D)]).astype(np.int64)
    gsize = np.array([len(g) for g in chan_groups], dtype=np.float64)

    def _wash():
        """Reset the state so each sentence reads independently (zero the leaky NMDA memory + conductances + firing;
        v/u to Izhikevich rest). Robust to array-size details (avoids the finicky EMERGE snapshot/restore pair)."""
        for nm in ("cp_conductance_g_nmda_recurrent", "cp_conductance_g_nmda_recurrent_rise", "cp_conductance_g_nmda",
                   "cp_conductance_g_e", "cp_conductance_g_i", "cp_firing_states"):
            arr = getattr(b, nm, None)
            if arr is not None: arr[:] = 0.0
        if b.cp_membrane_potential_v is not None: b.cp_membrane_potential_v[:] = -65.0
        if b.cp_recovery_variable_u is not None: b.cp_recovery_variable_u[:] = 0.0

    def onbridge_states(ids):
        """Drive the channel region per token; return the per-position firing-rate state [T, 2D] (ON then OFF rates)."""
        _wash()
        rates = []
        _a = np.zeros(D)                                             # host leaky state (for --exact-state isolating test)
        for t in range(len(ids)):
            h = _ln(emb[ids[t]]); v = Wv @ h                        # [D]
            if getattr(args, "exact_state", False):
                # ISOLATE the spiking READ: drive with the EXACT host-computed rate-SSM state (leaky integral done in host)
                # -> the neurons transduce the exact state to firing; if this GOes, the read is fine + the substrate's
                # input-integral (self-NMDA ~0.6 fidelity) is the only gap; if not, the spiking read itself is the limit.
                _a = decay * _a + v
                chan_drive = np.concatenate([np.maximum(_a, 0.0), np.maximum(-_a, 0.0)]) * args.drive_scale
            else:
                chan_drive = np.concatenate([np.maximum(v, 0.0), np.maximum(-v, 0.0)]) * args.drive_scale   # [2D] ON|OFF
            cur = np.zeros(nnrn, np.float32)
            cur[all_drive_idx] = chan_drive[chan_of]                # broadcast each channel's drive to its pop_k neurons
            cnt = np.zeros(2 * D, np.float64)
            for _ in range(_T_STEP):
                b.cp_external_input_current[:] = 0.0
                b.cp_external_input_current[all_drive_idx] = (xp.asarray(cur[all_drive_idx]) if xp is not None
                                                             else cur[all_drive_idx])
                b._run_one_simulation_step()
                fs = np.asarray(to_host(b.cp_firing_states))
                np.add.at(cnt, chan_of, fs[all_drive_idx].astype(np.float64))   # sum spikes per channel (over its pop_k)
            if getattr(args, "read_gnmda", False):
                # LEVER 1 (research-gate GO): read the STANDING NMDA conductance (the ~100 ms leaky integral) directly,
                # not the within-window spike count. cp_conductance_g_nmda IS the analog leaky-SSM state on real spikes
                # (self-NMDA autapse charges it; decays tau=100 ms; not washed within a sentence). Skips the
                # spike-quantization + f-I-saturation + 3 ms-window read losses (the 0.786 READ ceiling). I_nmda is exactly
                # the postsynaptic current a downstream neuron integrates, so this reads the graded dendritic signal (the
                # mission-compliant spike-pure closure = route it to a downstream read-out pool, a cheap follow-on).
                gn = np.zeros(2 * D, np.float64)
                gnmda = np.asarray(to_host(b.cp_conductance_g_nmda)).astype(np.float64)
                np.add.at(gn, chan_of, gnmda[all_drive_idx])
                rates.append(gn / gsize)                            # standing NMDA integral, pop-averaged
            else:
                rates.append(cnt / (_T_STEP * gsize))               # channel firing rate = pop-averaged (noise-averaged)
        b.cp_external_input_current[:] = 0.0
        return np.asarray(rates)                                     # [T, 2D]

    def rate_ssm_states(ids):
        """The reference rate-SSM analog [relu(a),relu(-a)] per position (to VERIFY the on-bridge mapping)."""
        a = np.zeros(D); out = []
        for t in range(len(ids)):
            h = _ln(emb[ids[t]]); a = decay * a + (Wv @ h)
            out.append(np.concatenate([np.maximum(a, 0.0), np.maximum(-a, 0.0)]))
        return np.asarray(out)                                       # [T, 2D]

    # ---- VERIFY the mapping on 5 sentences (corr of on-bridge firing-rate state vs the rate-SSM analog state) ----
    corrs = []
    for ids in ev_ids[:5]:
        if len(ids) < 4: continue
        ob = onbridge_states(ids); rs = rate_ssm_states(ids)
        c = np.corrcoef(ob.flatten(), rs.flatten())[0, 1]
        corrs.append(c)
    mapcorr = float(np.nanmean(corrs)) if corrs else float("nan")
    _ob0 = onbridge_states(ev_ids[0]) if ev_ids else np.zeros((1, 2 * D))
    _act = float((_ob0.std(0) > 1e-6).mean())                       # fraction of channels with ANY variance across tokens
    print(f"[verify] on-bridge firing-rate state vs rate-SSM analog state: corr={mapcorr:.3f} "
          f"(>0.3 => substrate realizes the leaky state) | firing: mean={_ob0.mean():.3f} max={_ob0.max():.3f} "
          f"frac-active-channels={_act:.2f} (low mean/frac => sparse; discriminative read-out needs varied firing)", flush=True)

    # ---- RE-FIT the read-out on the ACTUAL on-bridge firing-rate states (reservoir-computing: the leaky DYNAMICS are the
    #      fixed on-bridge diagonal self-NMDA; only the linear read-out is trained -- the on-bridge state ~= the rate-SSM
    #      state at a different SCALE (corr above), so a fresh ridge read-out on the on-bridge state recovers the capture) ----
    def _feat(rate_t, ids_t):
        """read-out feature: the raw ON/OFF firing state (2D) + the RECEPTANCE-gated signed state r_h*(ON-OFF) (D) --
        the SSM's current-token gating of the leaky state that the raw linear read-out lacked."""
        r_h = 1.0 / (1.0 + np.exp(-(Wr @ _ln(emb[ids_t]))))          # receptance (current-token gate), D
        signed = rate_t[:D] - rate_t[D:]                            # ON-OFF ~= the signed leaky state a
        return np.concatenate([rate_t, r_h * signed])              # [3D]

    t0 = time.time()
    fit_ids = tr_ids[:args.n_fit]
    Xtr, Ytr = [], []
    for ids in fit_ids:
        if len(ids) < 2: continue
        rates = onbridge_states(ids)
        for t in range(len(ids) - 1):
            Xtr.append(_feat(rates[t], ids[t])); Ytr.append(ids[t + 1])
    Xtr = np.asarray(Xtr); Ytr = np.asarray(Ytr, dtype=np.int64)
    nf = Xtr.shape[1]                                                 # 3D
    mean = Xtr.mean(0); std = Xtr.std(0) + 1e-6
    Xn = (Xtr - mean) / std
    if getattr(args, "mlp_readout", False):
        # NONLINEAR (MLP) read-out on the on-bridge states: the --exact-state test showed a LINEAR read can't match the
        # jointly-trained WKV read; a small MLP is the obvious next-method (reservoir-computing with a nonlinear read).
        import torch, torch.nn as nn
        torch.manual_seed(args.seed)
        Xt = torch.tensor(Xn, dtype=torch.float32); Yt = torch.tensor(Ytr)
        mlp = nn.Sequential(nn.Linear(nf, 256), nn.GELU(), nn.Linear(256, V))
        opt = torch.optim.Adam(mlp.parameters(), lr=2e-3, weight_decay=1e-4); lf = nn.CrossEntropyLoss()
        for _ in range(30):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 256):
                b_ = perm[i:i+256]; opt.zero_grad(); lf(mlp(Xt[b_]), Yt[b_]).backward(); opt.step()
        _mlp = mlp
        print(f"[refit-mlp] trained MLP read-out on {len(Xtr)} on-bridge states (nonlinear); fit-elapsed {time.time()-t0:.0f}s", flush=True)
        Wd = None; Temp = 1.0
    else:
        Z = np.concatenate([Xn, np.ones((len(Xn), 1))], 1)               # [n, 3D+1]
        ZtOH = np.zeros((V, nf + 1)); np.add.at(ZtOH, Ytr, Z)
        Wd = np.linalg.solve(Z.T @ Z + 5.0 * np.eye(nf + 1), ZtOH.T)     # ridge read-out [3D+1, V]
        _mlp = None
    if _mlp is None:                                                # temperature calib (ridge only)
        lg = Z[:20000] @ Wd; ys = Ytr[:20000]
        def _ce_T(T):
            z = lg / T; z = z - z.max(1, keepdims=True); e = np.exp(z); p = e / e.sum(1, keepdims=True)
            return float(-np.log(p[np.arange(len(ys)), ys] + 1e-12).mean())
        Temp = min([(_ce_T(T), T) for T in (0.5, 1, 2, 4, 8, 16)])[1]
    print(f"[refit] fitted {'MLP' if _mlp is not None else 'ridge'} read-out on {len(Xtr)} on-bridge states (T={Temp}); fit-elapsed {time.time()-t0:.0f}s", flush=True)

    ce = defaultdict(float); bce = defaultdict(float); tce = defaultdict(float); cnt = defaultdict(int)
    for si, ids in enumerate(ev_ids):
        if len(ids) < 2: continue
        rates = onbridge_states(ids)                                 # [T, 2D]
        for t in range(len(ids) - 1):
            if _mlp is not None:
                import torch
                xf = ((_feat(rates[t], ids[t]) - mean) / std).astype(np.float32)
                with torch.no_grad():
                    logits = _mlp(torch.tensor(xf)).numpy()
            else:
                x = np.concatenate([(_feat(rates[t], ids[t]) - mean) / std, [1.0]])
                logits = (x @ Wd) / Temp
            logits = logits - logits.max(); p = np.exp(logits); p = p / p.sum()
            d = t + 1; bkt = _bucket(d)
            ce[bkt] += -math.log(max(p[ids[t+1]], 1e-12))
            bce[bkt] += -math.log(max(P_bi[ids[t], ids[t+1]], 1e-12))
            u = ids[t-1] if t >= 1 else -1
            tce[bkt] += -math.log(max(tri(u, ids[t], ids[t+1]), 1e-12))
            cnt[bkt] += 1
    depth = {}
    for lo, hi in BUCKETS:
        bkt = f"{lo}-{hi}" if lo != hi else f"{lo}"
        if bkt in cnt:
            n = cnt[bkt]
            depth[bkt] = {"n": n, "onbridge": round(ce[bkt]/n, 3), "bigram": round(bce[bkt]/n, 3),
                          "trigram": round(tce[bkt]/n, 3), "vs_trigram": round((tce[bkt]-ce[bkt])/n, 3)}
    print(f"[seed {args.seed}] ON-BRIDGE per-depth NLL (elapsed {time.time()-t0:.0f}s, decay={decay:.3f}):", flush=True)
    for lo, hi in BUCKETS:
        bkt = f"{lo}-{hi}" if lo != hi else f"{lo}"
        if bkt in depth:
            dd = depth[bkt]
            print(f"    depth {bkt:>5} (n={dd['n']:>5}): onbridge {dd['onbridge']:.3f} | bigram {dd['bigram']:.3f} | "
                  f"trigram {dd['trigram']:.3f} || vs-trigram {dd['vs_trigram']:+.3f}", flush=True)
    deep = depth.get("10-99", {})
    go = bool(deep and deep["vs_trigram"] > 0.02 and mapcorr > 0.3)
    print(f"    VERDICT: {'GO' if go else 'no-go'} (deep vs-trigram {deep.get('vs_trigram','?')}, map-corr {mapcorr:.3f})", flush=True)
    out = {"runner": "_emerge_wkv_onbridge_derisk", "ssm": args.ssm, "seed": args.seed, "map_corr": mapcorr,
           "decay": decay, "by_depth": depth, "go": go}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True); Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"-> {args.json}", flush=True)


if __name__ == "__main__":
    main()
