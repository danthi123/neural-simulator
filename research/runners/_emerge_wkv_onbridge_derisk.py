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

_T_STEP = 6                                                          # bridge steps per token (integration window)


def _build_channel_bridge(D, seed, nmda_tau_ms, dt=0.5):
    """One region of 2*D Izhikevich channel neurons (D ON + D OFF), NO internal recurrence (internal_density=0): the leaky
    memory is the SLOW NMDA conductance (enable_nmda + a recurrent NMDA self-route), tuned to the SSM decay via nmda_tau.
    Driven per token by the external current; read from cp_firing_states. Returns (bridge, on_idx, off_idx, snapshot)."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.enable_nmda = True                                           # slow NMDA conductance = the leaky memory
    cfg.brain_regions = [
        BrainRegion(name="chan", n_neurons=2 * D, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    enable_nmda=True),
    ]
    # a recurrent NMDA self-route (each channel neuron's firing charges its OWN slow NMDA conductance = the leaky integral)
    cfg.region_pathways = [RegionPathway(from_region="chan", to_region="chan", density=0.0, weight_mean=0.0)]
    cfg.dt = float(dt)
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False; cfg.enable_stdp = False; cfg.enable_hebbian_learning = False
    rt = RuntimeState(); rt.actual_seed_used = seed
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt, gpu_config=GPUConfig())
    b._initialize_simulation_data()
    idx = np.asarray(b.region_manager.indices("chan"))
    on_idx, off_idx = idx[:D], idx[D:]
    from research.runners._emerge82_onbridge_lsm_derisk import _snapshot_state
    return b, on_idx, off_idx, _snapshot_state(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssm", required=True, help="saved SSM weights (_emerge_wkv_lm_derisk --save-ssm ..._seed<N>.npz)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--corpus", type=str, default="data/corpus/tinystories_train.txt")
    ap.add_argument("--n-sentences", type=int, default=40000)
    ap.add_argument("--max-train-sents", type=int, default=30000)
    ap.add_argument("--n-eval", type=int, default=200)              # SMALL (bridge stepping is slow)
    ap.add_argument("--drive-scale", type=float, default=1200.0)    # v_t -> external current pA
    ap.add_argument("--json", type=str, default="research/findings/raw/_emerge_wkv_onbridge.json")
    args = ap.parse_args()
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

    b, on_idx, off_idx, snap = _build_channel_bridge(D, args.seed, nmda_tau_ms=100.0)
    from research.runners._emerge82_onbridge_lsm_derisk import _restore_state
    nnrn = int(b.core_config.num_neurons)

    def onbridge_states(ids):
        """Drive the channel region per token; return the per-position firing-rate state [T, 2D] (ON then OFF rates)."""
        _restore_state(b, snap)
        rates = []
        for t in range(len(ids)):
            h = _ln(emb[ids[t]]); v = Wv @ h                        # [D]
            cur = np.zeros(nnrn, np.float32)
            cur[on_idx] = np.maximum(v, 0.0) * args.drive_scale
            cur[off_idx] = np.maximum(-v, 0.0) * args.drive_scale
            cnt = np.zeros(2 * D, np.float64)
            for _ in range(_T_STEP):
                b.cp_external_input_current[:] = 0.0
                b.cp_external_input_current[on_idx] = xp.asarray(cur[on_idx]) if xp is not None else cur[on_idx]
                b.cp_external_input_current[off_idx] = xp.asarray(cur[off_idx]) if xp is not None else cur[off_idx]
                b._run_one_simulation_step()
                fs = np.asarray(to_host(b.cp_firing_states))
                cnt += np.concatenate([fs[on_idx], fs[off_idx]]).astype(np.float64)
            rates.append(cnt / _T_STEP)
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
    print(f"[verify] on-bridge firing-rate state vs rate-SSM analog state: corr={mapcorr:.3f} "
          f"(>0.3 => the substrate realizes the leaky state; low => wrong mapping)", flush=True)

    # ---- per-depth NLL: on-bridge read-out (Wo_sp over the firing-rate state -> head) vs bigram vs FAIR trigram ----
    ce = defaultdict(float); bce = defaultdict(float); tce = defaultdict(float); cnt = defaultdict(int)
    t0 = time.time()
    for si, ids in enumerate(ev_ids):
        if len(ids) < 2: continue
        rates = onbridge_states(ids)                                 # [T, 2D]
        for t in range(len(ids) - 1):
            r_h = 1.0 / (1.0 + np.exp(-(Wr @ _ln(emb[ids[t]]))))     # receptance (host)
            outv = r_h * (Wo_sp @ rates[t])                          # [D]
            logits = head_w @ outv + head_b
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
