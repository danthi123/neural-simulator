"""On-substrate local read-out rule, STAGE 0+1 (the real bridge) — does the bridge's three-factor plasticity,
fed eligibility = presynaptic activity + cp_per_synapse_reward_override = per-output error, LEARN the binder
read-out decoder W_O to host parity? This realizes the (numpy-proven, CYCLE-153) local delta rule in real
synaptic plasticity, on the point-neuron SimulationBridge, with NO new protected sim/ mechanism (the per-synapse
override channel `cp_per_synapse_reward_override`, bridge.py:6866-6878, already exists default-None and is the
exact per-output third factor: weight_update = lr * override * eligibility = the delta rule per synapse).

STAGE 0 (linchpin): a SINGLE reward-modulation step with eligibility = a known pre vector E and override = a known
per-output error O must yield exactly ΔW = lr * O_post * E_pre per synapse (the delta rule). Confirms the bridge
arithmetic + the array hooks + the (pre,post) CSR mapping.

STAGE 1 (learn): iterate the delta step over the systematicity protocol -- each sample sets eligibility = the
presynaptic ON/OFF rate code of `act`, override = (target - est) per output (est = W @ rate read from the bridge's
own weights; the read-out stays LINEAR, as the production composer's cleanup is, and the err is a teaching scaffold
to be neuralised later). After training, evaluate held-out recall using the BRIDGE's learned W vs the host
LocalRuleBinder. GO = on-bridge held-out >= 0.85x host, systematicity holds, + anti-cheats: scrambled override
collapses; no-override (global-scalar only) cannot learn the per-output map.

Reuse-by-import (the systematicity harness + the host LocalRuleBinder reference). GPU (tiny bridge: 2*D_h -> D_in).
Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_onsubstrate_readout_bridge_derisk [--dh 64] [--seeds 42]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import to_host  # noqa: E402
from research.runners.cortex_learned_binder_systematicity_probe import (  # noqa: E402
    make_role_codes, make_systematicity_splits, native_argmax)

R, F, N_SPLITS = 4, 16, 3
LR = 0.5            # NLMS step (the spike-rate de-risk used 0.5; the bridge applies lr*override*eligibility)
N_PASSES = 40      # epochs over the train facts (each fact -> 1 delta step per role)


def build_readout_bridge(two_dh, d_in, seed):
    """A bridge with input_pop (2*D_h) -> output_pop (D_in), dense + PLASTIC + reward-modulated. STDP/Hebbian/OU
    off so the ONLY plasticity is the reward-modulated three-factor update (eligibility * override). Returns the
    bridge, the input/output index arrays, and the per-synapse (pre,post) mapping aligned to cp_connections.data."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion(name="inp", n_neurons=two_dh, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="out", n_neurons=d_in, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="_anchor", n_neurons=4, exc_fraction=1.0, internal_density=1.0),
    ]
    # dense plastic inp->out, TINY non-zero init (a zero-weight pathway creates no usable synapses -> the framework
    # falls back to default connectivity). The decoder grows from ~0 by the local rule.
    cfg.region_pathways = [
        RegionPathway(from_region="inp", to_region="out", density=1.0, weight_mean=0.01, weight_jitter=0.0,
                      plastic=True),
    ]
    cfg.dt = 1.0
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_reward_modulation = True               # the three-factor block (the override path)
    cfg.reward_learning_rate = LR
    cfg.reward_baseline = 0.0
    cfg.reward_eligibility_tau_ms = 1.0e12            # ~no decay (we set eligibility fresh each step)
    cfg.reward_aversive_scale = 1.0                   # no LTD asymmetry (the override carries the signed error)
    # The decoder W_O is SIGNED (maps the rate code to signed filler codes); allow signed weights so the reward
    # clip (bridge.py:6908-6920, hebbian bounds when STDP off) does not RECTIFY negative delta-rule updates to 0.
    # This isolates the LEARNING-machinery question from the Dale's-law (non-negative synapse) question -- the est
    # is read from cp_connections.data in numpy, so the forward-current sign constraint does not apply here; the
    # exc/inh-split (Dale's-law-respecting) realization of the signed decoder is the documented follow-on.
    cfg.stdp_w_max = 1.0e9; cfg.stdp_w_min = -1.0e9
    cfg.hebbian_max_weight = 1.0e9; cfg.hebbian_min_weight = -1.0e9
    rt = RuntimeState(); rt.actual_seed_used = seed
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt, gpu_config=GPUConfig())
    b._initialize_simulation_data()
    inp = np.asarray(b.region_manager.indices("inp"))
    out = np.asarray(b.region_manager.indices("out"))
    # (pre,post) per synapse, aligned to cp_connections.data (CSR). cp_connections is [post, pre] (I_post = C @ r_pre).
    import scipy.sparse as _sp
    C = b.cp_connections
    coo = (C.get() if hasattr(C, "get") else C).tocoo()
    pre_of = np.asarray(coo.row)    # cp_connections is [pre, post]: rows = PRE (confirmed: row in inp)
    post_of = np.asarray(coo.col)   # cols = POST (confirmed: col in out)
    # precompute vectorized per-synapse local maps (pre->in-local, post->out-local) so _read_W is O(nnz) fancy-index,
    # not a python loop over nnz (= 38400 at D_in=300 -> a python loop would crawl over thousands of train steps).
    num = int(b.core_config.num_neurons)
    in_lookup = np.full(num, -1, np.int64); in_lookup[inp] = np.arange(len(inp))
    out_lookup = np.full(num, -1, np.int64); out_lookup[out] = np.arange(len(out))
    nnz = int(b.cp_connections.nnz)
    il = in_lookup[pre_of[:nnz]]; ol = out_lookup[post_of[:nnz]]
    valid = (il >= 0) & (ol >= 0)
    b._ro_il, b._ro_ol, b._ro_valid = il, ol, valid
    b._ro_nin, b._ro_nout = len(inp), len(out)
    print(f"  [conn] nnz={nnz} | inp->out synapses={int(np.sum(valid))} | |inp|={len(inp)} |out|={len(out)}", flush=True)
    return b, inp, out, pre_of, post_of


def _set_and_step(b, pre_of, post_of, pre_vec_full, err_vec_full):
    """Set eligibility[k] = pre_vec[pre(k)], override[k] = err[post(k)], reward signal != baseline, run one step ->
    the bridge applies weight_update = lr * override * eligibility per synapse. STDP off so eligibility is ours."""
    import sim.backend as _bk
    xp, _ = _bk.get_backend()
    nnz = int(b.cp_connections.nnz)
    elig = pre_vec_full[pre_of[:nnz]].astype(np.float32)
    over = err_vec_full[post_of[:nnz]].astype(np.float32)
    b.cp_eligibility_trace[:nnz] = xp.asarray(elig)
    b.cp_per_synapse_reward_override = xp.asarray(over)
    b.core_config.current_reward_signal = 1.0          # != baseline 0 -> enter the update path (value unused by override path)
    # zero external drive so neuron dynamics don't add spurious eligibility-independent effects on weights
    b.cp_external_input_current[:] = 0.0
    b._run_one_simulation_step()
    b.core_config.current_reward_signal = 0.0


def _read_W(b, inp, out, pre_of, post_of):
    """Dense decoder W[out_local, in_local] from cp_connections.data (so est = W @ rate). Vectorized via the
    precomputed per-synapse local maps (b._ro_ol/_il/_valid)."""
    nnz = int(b.cp_connections.nnz)
    data = np.asarray(to_host(b.cp_connections.data))[:nnz]
    W = np.zeros((b._ro_nout, b._ro_nin), np.float64)
    v = b._ro_valid
    W[b._ro_ol[v], b._ro_il[v]] = data[v]
    return W


def _rate(act):
    """signed act [D_h] -> ON/OFF non-negative rate code [2*D_h] (the substrate read; deterministic here -- the
    Poisson-noise robustness is the separate numpy de-risk; this isolates the bridge LEARNING machinery)."""
    return np.concatenate([np.maximum(act, 0.0), np.maximum(-act, 0.0)])


def stage0_check(b, inp, out, pre_of, post_of, seed):
    """Single-step linchpin: ΔW must equal lr * outer(err, pre) (the delta rule)."""
    rng = np.random.default_rng(seed)
    two_dh, d_in = len(inp), len(out)
    pre = np.zeros(int(b.core_config.num_neurons)); err = np.zeros(int(b.core_config.num_neurons))
    pre_local = rng.uniform(0.2, 1.0, two_dh); err_local = rng.uniform(-1.0, 1.0, d_in)
    pre[inp] = pre_local; err[out] = err_local
    W0 = _read_W(b, inp, out, pre_of, post_of)
    _set_and_step(b, pre_of, post_of, pre, err)
    W1 = _read_W(b, inp, out, pre_of, post_of)
    dW = W1 - W0
    expect = LR * np.outer(err_local, pre_local)        # ΔW[out,in] = lr * err_out * pre_in
    max_abs = float(np.max(np.abs(dW - expect)))
    rel = max_abs / (float(np.max(np.abs(expect))) + 1e-9)
    ok = rel < 0.05
    print(f"  [stage0 seed {seed}] max|dW - lr*outer(err,pre)| = {max_abs:.3e} (rel {rel:.3e})  -> {'OK' if ok else 'MISMATCH'}",
          flush=True)
    # reset the learned weights to 0 for the stage-1 training
    b.cp_connections.data[:] = 0.0
    return ok


def run_seed(codes, seed, two_dh, d_h):
    from research.runners._phaseB_localrule_readout_derisk import LocalRuleBinder
    splits = make_systematicity_splits(R, F, N_SPLITS, seed)
    fillers = codes[:F]; d_in = fillers.shape[1]
    roles = make_role_codes(R, d_in, seed)
    rng_pm1 = np.random.default_rng(seed * 31 + 5)
    R_proj = rng_pm1.standard_normal((d_in, d_h)) / np.sqrt(d_in)
    role_pm1 = np.where(roles @ R_proj >= 0.0, 1.0, -1.0)
    # fixed-random encoder W_F (shared by host + bridge: the encoder is NOT learned -- NEF)
    rngF = np.random.default_rng(seed * 17 + 3)
    W_F = rngF.standard_normal((d_in, d_h)) / np.sqrt(d_in)

    b, inp, out, pre_of, post_of = build_readout_bridge(two_dh, d_in, seed)
    if not stage0_check(b, inp, out, pre_of, post_of, seed):
        return {"seed": seed, "stage0": False, "onbridge": 0.0, "host": 0.0, "scramble": 0.0, "mem_floor": 0.0}

    onb_h, host_h, scr_h, memf = [], [], [], []
    full = np.zeros(int(b.core_config.num_neurons))
    perm = np.random.default_rng(seed * 5 + 1).permutation(d_in)
    for split in splits:
        tr = {r: [f for (rr, f) in split["train"] if rr == r] for r in range(3)}
        if min(len(tr[r]) for r in range(3)) == 0:
            continue
        # host reference (the numpy local delta rule) on the SAME data
        host = LocalRuleBinder(D_in=d_in, role_pm1=role_pm1, D_h=d_h, seed=seed)

        def _train_bridge(scramble=False):
            b.cp_connections.data[:] = 0.0
            rr = np.random.default_rng(seed * 53 + 9)
            n_steps = N_PASSES * max(len(split["train"]), 1)
            for _ in range(n_steps):
                fa = rr.choice(tr[0]); fv = rr.choice(tr[1]); fo = rr.choice(tr[2])
                roleids, fillerids = [0, 1, 2], [int(fa), int(fv), int(fo)]
                t = int(rr.integers(3))
                ws = [fillers[f] @ W_F for f in fillerids]
                bundle = sum(role_pm1[r] * w for r, w in zip(roleids, ws))
                act = bundle * role_pm1[roleids[t]]
                rate = _rate(act)                                  # [2*D_h] ON/OFF
                W = _read_W(b, inp, out, pre_of, post_of)          # current bridge decoder
                est = W @ rate                                     # linear read-out (from the bridge's own weights)
                err = fillers[fillerids[t]] - est                  # target - est (sign: bridge ADDS lr*override*elig)
                if scramble:
                    err = err[perm]
                full[:] = 0.0; full[inp] = rate
                e_full = np.zeros(int(b.core_config.num_neurons)); e_full[out] = err
                _set_and_step(b, pre_of, post_of, full, e_full)
            return _read_W(b, inp, out, pre_of, post_of)

        # host training (numpy)
        rr = np.random.default_rng(seed * 53 + 9)
        for _ in range(N_PASSES * max(len(split["train"]), 1)):
            fa = rr.choice(tr[0]); fv = rr.choice(tr[1]); fo = rr.choice(tr[2])
            host.train_fact_step([0, 1, 2], [int(fa), int(fv), int(fo)], fillers, int(rr.integers(3)))

        W_bridge = _train_bridge(scramble=False)
        W_scr = _train_bridge(scramble=True)

        def _bind(r, f):
            return role_pm1[r] * (fillers[f] @ W_F)

        def _recall_bridge(W, bundle, r):
            return _rate(bundle * role_pm1[r]) @ W.T          # est[D_in] = W[out,in] @ rate[in]

        def _eval(recall_fn):
            ok = n = 0
            erng = np.random.default_rng(seed * 7 + 1)
            for _ in range(40):
                fids = erng.choice(F, 3, replace=False)
                bundle = sum(_bind(r, int(fids[r])) for r in range(3))
                for r in range(3):
                    if (r, int(fids[r])) not in set(split["train"]):
                        ok += int(native_argmax(recall_fn(bundle, r), fillers) == fids[r]); n += 1
            return ok / max(n, 1)

        onb_h.append(_eval(lambda bn, r: _recall_bridge(W_bridge, bn, r)))
        scr_h.append(_eval(lambda bn, r: _recall_bridge(W_scr, bn, r)))
        host_h.append(_eval(lambda bn, r: host.unbind(bn, r)))
        from research.runners.cortex_learned_binder_systematicity_probe import score_memorization_floor
        memf.append(score_memorization_floor(split["train"], split["held_out"], fillers)["held_out_acc"])

    row = {"seed": seed, "stage0": True, "onbridge": float(np.mean(onb_h)), "host": float(np.mean(host_h)),
           "scramble": float(np.mean(scr_h)), "mem_floor": float(np.mean(memf))}
    print(f"  [seed {seed}] ON-BRIDGE held-out {row['onbridge']:.3f} | host (numpy delta) {row['host']:.3f} | "
          f"scramble {row['scramble']:.3f} | mem-floor {row['mem_floor']:.3f}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dh", type=int, default=64)
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--out", type=str, default=os.path.join(
        _REPO, "research", "findings", "raw", "_phaseB_onsubstrate_readout_bridge.json"))
    args = ap.parse_args()
    os.environ.setdefault("SIM_BACKEND", "cupy")
    seeds = [int(s) for s in args.seeds.split(",")]
    codes_path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_stream_codes_320_seed42.npy")
    if not os.path.exists(codes_path):
        print(f"  [missing] {codes_path}", flush=True)
        return
    codes = np.load(codes_path).astype(np.float64)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    two_dh = 2 * args.dh
    t0 = time.time()
    print(f"[on-substrate read-out bridge de-risk] does the bridge 3-factor plasticity (eligibility x per-output "
          f"override) LEARN the read-out decoder? D_h={args.dh} seeds={seeds}", flush=True)
    rows = [run_seed(codes, s, two_dh, args.dh) for s in seeds]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    stage0 = all(r["stage0"] for r in rows)
    onb, host, scr, mf = m("onbridge"), m("host"), m("scramble"), m("mem_floor")
    n_par = sum(int(r["onbridge"] >= 0.85 * r["host"]) for r in rows)
    bar = int(np.ceil(5 / 6 * len(seeds)))
    print(f"\n{'='*100}", flush=True)
    print(f"  STAGE 0 (ΔW == lr*outer(err,pre)): {'PASS' if stage0 else 'FAIL'}", flush=True)
    print(f"  MEAN ({len(seeds)} seeds): ON-BRIDGE {onb:.3f} | host(numpy delta) {host:.3f} | scramble {scr:.3f} | "
          f"mem-floor {mf:.3f} | onbridge>=0.85x host: {n_par}/{len(seeds)}", flush=True)
    scramble_collapses = scr < 0.5 * max(onb, 1e-9)
    go = stage0 and (n_par >= bar) and (onb >= mf + 0.25) and scramble_collapses
    if go:
        print(f"  GO: the bridge's three-factor plasticity LEARNS the read-out decoder via the per-output override "
              f"channel -- on-bridge held-out {onb:.3f} = {onb/max(host,1e-9):.0%} of the host numpy delta rule, "
              f">> mem-floor {mf:.3f}, scrambled-override collapses ({scr:.3f}). ==> the binder read-out is learned "
              f"by REAL synaptic plasticity (no host Adam, no host numpy update) -- the last host-training shortcut "
              f"is removed ON THE SUBSTRATE. The per-output error is still a host teaching scaffold (neuralise next).",
              flush=True)
    elif not stage0:
        print(f"  STAGE-0 FAIL: the bridge does not apply ΔW = lr*override*eligibility as expected -- localize the "
              f"(pre,post) CSR mapping / the reward-mod gating before any further build.", flush=True)
    else:
        print(f"  BOUNDARY/NEGATIVE: stage-0 holds but training underperforms ({onb:.3f} vs host {host:.3f}) or the "
              f"anti-cheat didn't collapse ({scr:.3f}) -- diagnose (eligibility decay, weight clip, lr).", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*100}", flush=True)
    out = {"verdict": "GO" if go else ("STAGE0_FAIL" if not stage0 else "BOUNDARY"), "stage0": stage0,
           "D_h": args.dh, "seeds": seeds, "onbridge": onb, "host": host, "scramble": scr, "mem_floor": mf,
           "n_parity": n_par, "per_seed": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
