"""gap#1 — TOKEN-SDR write-fidelity de-risk (research-gate #1, 2026-07-20).

THE GATE'S REFRAME: M2's wall is the ENCODE — it charges a conductance from a few-spike RATE CODE of a continuous
`v_t` (a regression), capping write-fidelity at 0.786. The escape: `v_t = Wv·LN(emb[x_t])` is a deterministic
function of the DISCRETE token `x_t`, so the encode is a SELECTION + fixed-synapse lookup (point neurons do this
cleanly), NOT a regression. The token spikes carry IDENTITY only; a FIXED value-projection synapse set delivers the
magnitude → `g_e - g_i = Wv·s_token = v_t` exactly, with no few-spike quantization of a graded value to lose.

This runner runs ONLY the cheap checks the gate ranked BEFORE any deep-NLL:
  PRE-FLIGHT (the trap that refuted 3 mechanisms today — validate on DEPLOYED inputs, not synthetic one-hots):
    over the REAL deployed token sequence, does the token-SDR pool fire (a) DETERMINISTICALLY per token and
    (b) SEPARABLY across tokens, at the real window length / real vocab? If not, the "selection" is a rate code
    and #1 is dead pre-flight.
  WRITE-FIDELITY GATE: corr(v_t_true, conductance-derived v_t) over the deployed sequence vs M1's exact host inject.
    Pre-registered bar >= 0.95 (must clear M2's 0.786 by a wide margin; the -0.345 deep-NLL gap needs a near-exact
    input to recover M1's +0.126). Passing this essentially PREDICTS the 6-seed deep-NLL.

NO deep-NLL run here (that is the NEXT rung, only if write-fidelity clears). NO `sim/` edit (drives + reads public
arrays). Run: SIM_BACKEND=cupy python -m research.runners._emerge_wkv_tokensdr_writefidelity_derisk \
    --ssm bridges/wkv_ckpt/wkv_v1000_d128_seed42.npz --seed 42
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "cupy")
import argparse, json, math
import numpy as np
from collections import defaultdict


def build_tokensdr_bridge(D, V_pool, seed, k_active, dt=1.0, _no_ou=False):
    """A token pool (V_pool*K neurons: K SDR neurons per token) --[FIXED Wv value synapses]--> chan (2D graded
    SSM-state neurons: [0:D]=exc g_e carries relu(+v), [D:2D]=inh g_i carries relu(-v)). Signed value via exc/inh
    split (Dale's law: exc and inh are separate presynaptic neurons, so a signed value needs a push-pull pair)."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion
    cfg = CoreSimConfig(seed=seed)
    cfg.dt_ms = float(dt)
    cfg.enable_selective_ssm_state = True         # the slow SSM state (M1's validated integrator)
    if _no_ou:
        cfg.enable_ou_process = False             # DIAGNOSTIC ceiling: removes the orthogonal OU noise source
    cfg.ssm_k_leak = 0.06
    cfg.enable_brain_region_framework = True
    # token pool: V_pool groups of K exc neurons (one SDR per token). chan: 2D neurons (D exc-read + D inh-read).
    # We hand-wire synapses, so region internal density = 0.
    cfg.brain_regions = [
        # internal_density 0.05 (NOT 0.0 -- the documented profile_name_for_conn UnboundLocalError gotcha) with
        # weight_mean=0.0 -> connections exist structurally but carry ZERO weight = functionally inert recurrence.
        BrainRegion(name="tok", n_neurons=V_pool * k_active, exc_fraction=1.0, internal_density=0.05,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
        BrainRegion(name="chan", n_neurons=2 * D, exc_fraction=1.0, internal_density=0.05,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
    ]
    cfg.region_pathways = []
    rt = RuntimeState(); rt.actual_seed_used = seed
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=rt, gpu_config=GPUConfig())
    b._initialize_simulation_data()
    tok = np.asarray(b.region_manager.indices("tok"))
    chan = np.asarray(b.region_manager.indices("chan"))
    return b, tok, chan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssm", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--corpus", default="data/corpus/tinystories_train.txt")
    ap.add_argument("--n-sentences", dest="n_sentences", type=int, default=80000,
                    help="MUST match the value the SSM was trained with, or the vocab differs (documented silent failure).")
    ap.add_argument("--k-active", dest="k_active", type=int, default=8, help="SDR neurons per token")
    ap.add_argument("--t-step", dest="t_step", type=int, default=6, help="encode-window bridge steps per token")
    ap.add_argument("--drive-pa", dest="drive_pa", type=float, default=900.0)
    ap.add_argument("--n-eval-tokens", dest="n_eval", type=int, default=400)
    ap.add_argument("--no-ou", dest="no_ou", action="store_true",
                    help="DIAGNOSTIC: disable OU background noise to measure the noise-free CEILING (deployment is OU-on).")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from sim.backend import to_host, get_backend
    xp, _bk = get_backend()

    W = np.load(args.ssm, allow_pickle=True)
    V = int(W["V"]); D = int(W["d_model"])
    emb = W["emb.weight"]; ln_w = W["ln.weight"]; ln_b = W["ln.bias"]; Wv = W["Wv.weight"]

    def _ln(v):
        m = v.mean(); s = v.std() + 1e-5
        return (v - m) / s * ln_w + ln_b

    # per-token TRUE value v_x = Wv·LN(emb[x]) -- a FIXED vector per discrete token (the lookup table Wv synapses store)
    Vval = np.stack([Wv @ _ln(emb[x]) for x in range(V)])          # [V, D]

    # rebuild the SAME token stream the SSM used -- EXACTLY as M2 does (vocab MUST match; n_sentences matched)
    from research.runners._emerge_reservoir_lm_derisk import Vocab
    from research.runners._emerge_reservoir_lm_realcorpus_derisk import load_sentences
    sents = load_sentences(args.corpus, args.n_sentences)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(sents)); cut = int(0.85 * len(sents))
    tr = [sents[i] for i in idx[:cut]]; ev = [sents[i] for i in idx[cut:]]
    vocab = Vocab.build(tr, V=V)
    assert vocab.size == V, f"vocab mismatch: ckpt V={V} but stream rebuilt V={vocab.size} (n_sentences must match training)"
    ev_ids = [vocab.ids(s) for s in ev]
    flat = [t for s in ev_ids for t in s]                          # the DEPLOYED held-out token sequence
    ids = np.asarray(flat[:args.n_eval], dtype=np.int64)
    print(f"[setup] V={V} D={D} eval_tokens={len(ids)} k_active={args.k_active} t_step={args.t_step}", flush=True)

    # ---- build the bridge: token pool (V groups x K) + fixed Wv value synapses into chan ----
    b, tok, chan = build_tokensdr_bridge(D, V, args.seed, args.k_active, _no_ou=args.no_ou)
    nn = int(b.core_config.num_neurons)
    # token x -> its K SDR neurons (contiguous block)
    def sdr_of(x):
        return tok[x * args.k_active:(x + 1) * args.k_active]
    # FIXED value synapses: each token's K SDR neurons -> chan. The synapse from token x's SDR to channel c carries
    # v_x[c]/K split into exc (relu(+)) into chan[c] and inh (relu(-)) into chan[D+c]. Summed over K firing neurons
    # at ~unit rate over the window, g_e[c]-g_i[c] = v_x[c]. We install these as explicit wiring.
    pre, post, wts = [], [], []
    for x in range(V):
        s = sdr_of(x)
        vx = Vval[x]
        for c in range(D):
            vp = max(vx[c], 0.0) / args.k_active
            vm = max(-vx[c], 0.0) / args.k_active
            for j in s:
                if vp > 0: pre.append(int(j)); post.append(int(chan[c])); wts.append(float(vp))
                if vm > 0: pre.append(int(j)); post.append(int(chan[D + c])); wts.append(float(vm))
    # exc-only token neurons (Dale) charge g_e; relu(+v) -> chan[:D], relu(-v) -> chan[D:2D], both g_e; the readout
    # subtracts the two halves (vhat = ge_pos - ge_neg) -- the same two-branch difference M2 reads as ge - gi.
    b.inject_explicit_wiring({"wv_lookup": {"pre_indices": np.asarray(pre, np.int64),
                                            "post_indices": np.asarray(post, np.int64),
                                            "initial_weights": np.asarray(wts, np.float32),
                                            "plastic": False, "conn_type": "EXC"}})
    print(f"[wiring] installed {len(pre)} fixed Wv-lookup synapses", flush=True)

    def drive_token_window(x):
        """Fire token x's SDR for t_step steps; return the chan conductance-derived value vhat [D] and the SDR firing."""
        b.cp_conductance_g_e[:] = 0.0; b.cp_conductance_g_i[:] = 0.0
        b.cp_membrane_potential_v[:] = -65.0; b.cp_recovery_variable_u[:] = 0.0
        s = sdr_of(x); fired = np.zeros(len(s))
        cur = np.zeros(nn, np.float32); cur[s] = args.drive_pa
        for _ in range(args.t_step):
            b.cp_external_input_current[:] = 0.0
            b.cp_external_input_current[s] = (xp.asarray(cur[s]) if xp is not None else cur[s])
            b._run_one_simulation_step()
            fs = np.asarray(to_host(b.cp_firing_states))
            fired += fs[s]
        ge = np.asarray(to_host(b.cp_conductance_g_e)).astype(np.float64)
        ge_pos = ge[chan[:D]]; ge_neg = ge[chan[D:2 * D]]     # relu(+v) and relu(-v) halves, both via g_e (Dale)
        return ge_pos - ge_neg, fired

    # ===== PRE-FLIGHT: determinism + separability on the DEPLOYED token sequence =====
    uniq = sorted(set(int(x) for x in ids))[:60]
    fire_by_tok = {}
    det_ok = 0
    for x in uniq:
        _, f1 = drive_token_window(x); _, f2 = drive_token_window(x)
        fire_by_tok[x] = f1
        det_ok += int(np.array_equal(f1, f2) or np.corrcoef(f1, f2)[0, 1] > 0.99 if f1.std() > 0 else np.array_equal(f1, f2))
    # separability: mean firing-count per token should differ (each SDR block is disjoint, so cross-token overlap is 0
    # by construction; the real question is whether EACH token's SDR fires AT ALL and consistently)
    fired_counts = np.array([fire_by_tok[x].sum() for x in uniq])
    frac_silent = float(np.mean(fired_counts == 0))
    print(f"[preflight] determinism {det_ok}/{len(uniq)} tokens fire identically on repeat | "
          f"mean SDR spikes/window {fired_counts.mean():.1f} | silent-token frac {frac_silent:.2%}", flush=True)

    # ===== WRITE-FIDELITY GATE: corr(vhat, v_true) over the deployed sequence =====
    vhats, vtrues = [], []
    for x in ids:
        vh, _ = drive_token_window(int(x))
        vhats.append(vh); vtrues.append(Vval[int(x)])
    vhats = np.asarray(vhats); vtrues = np.asarray(vtrues)
    # a single global gain a downstream synapse would absorb (fit on first 30 tokens, applied to all -- no eval leak)
    fit = slice(0, 30)
    g = float((vhats[fit] * vtrues[fit]).sum() / max((vhats[fit] * vhats[fit]).sum(), 1e-12))
    corr = float(np.corrcoef((vhats * g).flatten(), vtrues.flatten())[0, 1])
    print(f"[write-fidelity] corr(conductance v_t, true v_t) = {corr:.4f}  (gain {g:.3g}; M2 wall 0.786; bar >= 0.95)", flush=True)
    verdict = "GO" if corr >= 0.95 and det_ok >= 0.9 * len(uniq) and frac_silent < 0.05 else "no-go"
    print(f"    VERDICT: {verdict} (write-fidelity {corr:.3f} vs bar 0.95; determinism {det_ok}/{len(uniq)}; silent {frac_silent:.1%})", flush=True)
    if args.json:
        json.dump({"seed": args.seed, "write_fidelity": corr, "gain": g, "determinism": det_ok,
                   "n_uniq": len(uniq), "frac_silent": frac_silent, "verdict": verdict}, open(args.json, "w"), indent=2)


if __name__ == "__main__":
    main()
