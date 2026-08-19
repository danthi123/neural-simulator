"""ON-BRIDGE realization of the Urbanczik-Senn neural teaching error (board #69, the #39 next rung).

WHAT THIS CLOSES. The read-out that learns the brain's word choices is trained ON the live spiking bridge by the
bridge's OWN three-factor plasticity: `weight_update = lr * cp_per_synapse_reward_override[synapse] *
cp_eligibility_trace[synapse]` = `lr * err_post * pre` (the cerebellar climbing-fiber delta rule, proven 6-seed GO
in `2026-06-17-onsubstrate-readout-rule-bridge-GO.md`). But the per-output error written into the override channel
was `err_j = target_j - est_j` -- a HOST subtraction. Under the BRAIN-BASED-ONLY standard that host formula is a
documented SHORTCUT: the *brain* is not computing the error, host bookkeeping is.

This runner delivers the read-out neuron's OWN somato-dendritic mismatch (Urbanczik & Senn, Neuron 2014) THROUGH
that same existing per-synapse channel, on the live bridge, so the production learning loop USES the neural error
and the host subtraction is retired. The value written into `cp_per_synapse_reward_override` for output j is the
neuron's intrinsic `(soma_rate_j - phi(v_basal_j))` mismatch -- computed by the SHIPPED
`sim.dendritic_plasticity.urbanczik_senn_update` (the neuron's biophysics), decoded to an error estimate by
dividing out the fixed small-signal transfer slope (rate decoding, exactly as the numpy de-risk does). No host
`target - est` anywhere in the NEURAL arm's loop.

DE-RISK BASE (reuse-by-import): this is the on-bridge realization of the 6-seed-GO numpy de-risk
`research/runners/_neural_error_population_derisk.py` (`2026-08-19-neural-error-population-GO.md`, NEURAL=0.964).
The U-S error math (gain/beta/spike-window/slope-decode) is byte-for-byte the same computation; the only change is
WHERE the resulting per-output error goes: into the live bridge's `cp_per_synapse_reward_override` instead of a
numpy `W_O +=`.

NO sim/ EDIT. The `cp_per_synapse_reward_override` array is already-present and default-None in the shipped bridge
(bridge.py:~714/4426/10445); the neural error is routed entirely runner-side. Therefore the production weight-update
path is byte-identical to main by construction -- proven by `git status` (no sim/ file) + a HOST/off md5 panel.

ARMS (6 seeds 42,43,44,100,101,102; systematicity protocol; bundled held-out generalization; the SAME real bridge):
  1. HOST-onbridge (reference / the current default) -- override = target - est (the shipped host subtraction).
  2. NEURAL-onbridge (the realization)               -- override = the U-S soma-vs-dendrite mismatch of the read-out
                                                        neurons, computed via the SHIPPED urbanczik_senn_update from
                                                        the bridge's OWN forward drive (est = W @ rate), decoded by
                                                        the fixed transfer slope. No host err.
  3. LESION-nodend (anti-cheat #2, dissociation)     -- pin v_basal=0 so the dendrite no longer predicts the soma ->
                                                        the mismatch stops tracking the estimate -> on-bridge
                                                        learning must fail. Proves the DENDRITIC self-prediction, not
                                                        a residual host term, drives the production path.
  4. LESION-noteach (anti-cheat #2)                  -- beta=0 -> soma == dendrite prediction -> mismatch ~ pure
                                                        Poisson noise -> the neural error emits ~nothing -> no learning.
  5. SCRAMBLE (anti-cheat)                            -- permute the neural error across outputs so err_j no longer
                                                        addresses output j -> must collapse (per-output error load-bearing).

GO = stage-0 delta-rule linchpin holds AND NEURAL-onbridge >= 0.85x HOST-onbridge held-out in >=5/6 seeds AND both
lesions AND scramble collapse (< 0.5x NEURAL). If NEURAL under-performs HOST, that gap IS the honest-negative
deliverable (it maps what the substrate's own on-bridge somato-dendritic error can do) -- reported precisely.

CPU/numpy (SIM_BACKEND=numpy; GPU shared). NO sim/ edit. Additive only.
Run:  OMP_NUM_THREADS=2 SIM_BACKEND=numpy python -u -m research.runners._neural_error_onbridge_derisk \
        --seeds 42,43,44,100,101,102

Biology: Urbanczik & Senn, "Learning by the Dendritic Prediction of Somatic Spiking," Neuron 81:521-528, 2014
(PubMed 24507189) -- shipped as sim/dendritic_plasticity.py. On-bridge learning channel: the cerebellar
climbing-fiber per-output third factor (Albus dw_i = -eta*pf_i*cf_burst) realized as
lr*cp_per_synapse_reward_override*cp_eligibility_trace (2026-06-17-onsubstrate-readout-rule-bridge-GO.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import to_host  # noqa: E402
from sim.dendritic_plasticity import urbanczik_senn_update, _sig  # noqa: E402  (THE shipped U-S rule)
from research.runners.cortex_learned_binder_systematicity_probe import (  # noqa: E402
    make_role_codes, make_systematicity_splits, native_argmax)
from tools.lab import attributable_to  # noqa: E402
from tools.verdict import Verdict      # noqa: E402

R, F, N_SPLITS = 4, 16, 3
LR = 0.5            # bridge reward_learning_rate (NLMS step; the shipped on-bridge GO used 0.5)
N_PASSES = 40      # epochs over the train facts (the shipped on-bridge GO budget)


def us_neural_error(est, target, g, beta, win, slope, rng, mode, n_pop=1):
    """The read-out neuron's OWN Urbanczik-Senn soma-vs-dendrite mismatch, decoded to a per-output error estimate.
    Byte-for-byte the numpy de-risk's error computation, using the SHIPPED urbanczik_senn_update. NO host err.

      dendrite   v_basal_j = g * est_j                         (forward drive through the plastic weights)
      soma       u_j       = (1-beta)*est_j + beta*target_j    (finite teacher nudging; beta=0 in lesion_noteach)
      soma rate  s_j       = <Poisson(sigma(g*u_j)*win)/win>_K (spiking somatic read, POOLED over K error-neurons)
      mismatch   m_j       = s_j - sigma(v_basal_j)            (the SHIPPED rule; v_basal pinned 0 in lesion_nodend)
      error      e_j       = m_j / slope                       (fixed transfer-slope rate decode)

    POPULATION-CODED READ (n_pop=K): K read-out error-neurons per output, each an independent Poisson somatic read of
    the SAME soma, pooled -> noise std / sqrt(K). This is the biological SNR lift the 2026-06-17 on-bridge boundary
    finding named ("several error neurons per output for SNR") -- required because the live on-bridge budget (40
    passes) is ~50x shorter than the numpy de-risk's 24000 steps, so per-step spike-count noise is not averaged out
    by the update stream. Faithfulness > speed: cortical words are redundantly coded by many neurons.
    """
    beta_eff = 0.0 if mode == "lesion_noteach" else beta
    u = (1.0 - beta_eff) * est + beta_eff * target
    s_clean = _sig(g * u)
    lam = np.clip(s_clean, 0.0, None) * win
    if n_pop <= 1:
        s_noisy = rng.poisson(lam) / win
    else:
        s_noisy = rng.poisson(lam[None, :].repeat(n_pop, axis=0)).mean(axis=0) / win  # pool K error-neurons
    v_basal = np.zeros_like(est) if mode == "lesion_nodend" else g * est
    # shipped rule with pre=ones(1) -> outer([1], (s_noisy - sig(v_basal))) -> row 0 is the per-output mismatch.
    mismatch = urbanczik_senn_update(np.ones(1), s_noisy, v_basal, np.ones(est.shape[0]), None, 1.0)[0]
    return mismatch / slope


def build_readout_bridge(two_dh, d_in, seed):
    """input_pop(2*D_h) -> output_pop(D_in), dense + PLASTIC + reward-modulated; STDP/Hebbian/OU OFF so the ONLY
    plasticity is the reward-modulated three-factor update (eligibility * override). (Verbatim plumbing from the
    on-substrate read-out bridge GO runner, which is a 6-seed GO.)"""
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
    cfg.region_pathways = [
        RegionPathway(from_region="inp", to_region="out", density=1.0, weight_mean=0.01, weight_jitter=0.0,
                      plastic=True),
    ]
    cfg.dt = 1.0
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed   # cfg.seed SEEDS the substrate (never actual_seed_used)
    cfg.enable_ou_process = False
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_reward_modulation = True
    cfg.reward_learning_rate = LR
    cfg.reward_baseline = 0.0
    cfg.reward_eligibility_tau_ms = 1.0e12
    cfg.reward_aversive_scale = 1.0
    cfg.stdp_w_max = 1.0e9; cfg.stdp_w_min = -1.0e9
    cfg.hebbian_max_weight = 1.0e9; cfg.hebbian_min_weight = -1.0e9
    rt = RuntimeState(); rt.actual_seed_used = seed
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt, gpu_config=GPUConfig())
    b._initialize_simulation_data()
    inp = np.asarray(b.region_manager.indices("inp"))
    out = np.asarray(b.region_manager.indices("out"))
    C = b.cp_connections
    coo = (C.get() if hasattr(C, "get") else C).tocoo()
    pre_of = np.asarray(coo.row)    # cp_connections is [pre, post]: rows = PRE
    post_of = np.asarray(coo.col)   # cols = POST
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
    """eligibility[k]=pre[pre(k)], override[k]=err[post(k)], reward!=baseline, one step -> the bridge applies
    weight_update = lr * override * eligibility per synapse."""
    import sim.backend as _bk
    xp, _ = _bk.get_backend()
    nnz = int(b.cp_connections.nnz)
    elig = pre_vec_full[pre_of[:nnz]].astype(np.float32)
    over = err_vec_full[post_of[:nnz]].astype(np.float32)
    b.cp_eligibility_trace[:nnz] = xp.asarray(elig)
    b.cp_per_synapse_reward_override = xp.asarray(over)
    b.core_config.current_reward_signal = 1.0
    b.cp_external_input_current[:] = 0.0
    b._run_one_simulation_step()
    b.core_config.current_reward_signal = 0.0


def _read_W(b):
    nnz = int(b.cp_connections.nnz)
    data = np.asarray(to_host(b.cp_connections.data))[:nnz]
    W = np.zeros((b._ro_nout, b._ro_nin), np.float64)
    v = b._ro_valid
    W[b._ro_ol[v], b._ro_il[v]] = data[v]
    return W


def _rate(act):
    return np.concatenate([np.maximum(act, 0.0), np.maximum(-act, 0.0)])


def stage0_check(b, inp, out, pre_of, post_of, seed):
    """ΔW must equal lr*outer(err,pre) (the delta rule) -- the validity linchpin."""
    rng = np.random.default_rng(seed)
    two_dh, d_in = len(inp), len(out)
    pre = np.zeros(int(b.core_config.num_neurons)); err = np.zeros(int(b.core_config.num_neurons))
    pre_local = rng.uniform(0.2, 1.0, two_dh); err_local = rng.uniform(-1.0, 1.0, d_in)
    pre[inp] = pre_local; err[out] = err_local
    W0 = _read_W(b)
    _set_and_step(b, pre_of, post_of, pre, err)
    W1 = _read_W(b)
    dW = W1 - W0
    expect = LR * np.outer(err_local, pre_local)
    max_abs = float(np.max(np.abs(dW - expect)))
    rel = max_abs / (float(np.max(np.abs(expect))) + 1e-9)
    ok = rel < 0.05
    print(f"  [stage0 seed {seed}] max|dW - lr*outer(err,pre)| = {max_abs:.3e} (rel {rel:.3e}) -> {'OK' if ok else 'MISMATCH'}",
          flush=True)
    b.cp_connections.data[:] = 0.0
    return ok


def run_seed(codes, seed, two_dh, d_h, args):
    splits = make_systematicity_splits(R, F, N_SPLITS, seed)
    fillers = codes[:F]; d_in = fillers.shape[1]
    roles = make_role_codes(R, d_in, seed)
    rng_pm1 = np.random.default_rng(seed * 31 + 5)
    R_proj = rng_pm1.standard_normal((d_in, d_h)) / np.sqrt(d_in)
    role_pm1 = np.where(roles @ R_proj >= 0.0, 1.0, -1.0)
    rngF = np.random.default_rng(seed * 17 + 3)
    W_F = rngF.standard_normal((d_in, d_h)) / np.sqrt(d_in)

    scale = float(np.std(fillers))
    g = float(args.gain) / max(scale, 1e-9)
    beta = float(args.beta); win = float(args.spike_gain)
    slope = max(beta, 1e-6) * g / 4.0

    b, inp, out, pre_of, post_of = build_readout_bridge(two_dh, d_in, seed)
    if not stage0_check(b, inp, out, pre_of, post_of, seed):
        return {"seed": seed, "stage0": False}

    n_pop = int(args.n_error_pop)
    perm = np.random.default_rng(seed * 5 + 1).permutation(d_in)

    def _train_arm(bb, inp_, out_, pre_, post_, split, tr, mode):
        """Train the on-bridge decoder on bridge `bb` with the per-output override set by `mode`. Returns final W +
        its md5. The NEURAL error is computed from bb's OWN forward drive and written into cp_per_synapse_reward_override."""
        num = int(bb.core_config.num_neurons)
        full = np.zeros(num); e_full = np.zeros(num)
        bb.cp_connections.data[:] = 0.0
        rr = np.random.default_rng(seed * 53 + 9)
        nerr_rng = np.random.default_rng(seed * 911 + 7)          # Poisson spike-count noise (seeded from cfg.seed)
        n_steps = N_PASSES * max(len(split["train"]), 1)
        for _ in range(n_steps):
            fa = rr.choice(tr[0]); fv = rr.choice(tr[1]); fo = rr.choice(tr[2])
            roleids, fillerids = [0, 1, 2], [int(fa), int(fv), int(fo)]
            t = int(rr.integers(3))
            ws = [fillers[f] @ W_F for f in fillerids]
            bundle = sum(role_pm1[r] * w for r, w in zip(roleids, ws))
            act = bundle * role_pm1[roleids[t]]
            rate = _rate(act)
            W = _read_W(bb)
            est = W @ rate                                         # the bridge's OWN forward drive (linear read-out)
            target = fillers[fillerids[t]]
            if mode == "host":
                err = target - est                                # HOST subtraction (the shortcut / current default)
            else:
                err = us_neural_error(est, target, g, beta, win, slope, nerr_rng, mode, n_pop)  # NEURAL soma-vs-dendrite
                if mode == "scramble":
                    err = err[perm]
            full[:] = 0.0; full[inp_] = rate
            e_full[:] = 0.0; e_full[out_] = err
            _set_and_step(bb, pre_, post_, full, e_full)
        W_final = _read_W(bb)
        md5 = hashlib.md5(np.ascontiguousarray(W_final).tobytes()).hexdigest()
        return W_final, md5

    def _bind(r, f):
        return role_pm1[r] * (fillers[f] @ W_F)

    def _recall_bridge(W, bundle, r):
        return _rate(bundle * role_pm1[r]) @ W.T

    def _eval(recall_fn, train_set):
        ok = n = 0
        erng = np.random.default_rng(seed * 7 + 1)
        for _ in range(40):
            fids = erng.choice(F, 3, replace=False)
            bundle = sum(_bind(r, int(fids[r])) for r in range(3))
            for r in range(3):
                if (r, int(fids[r])) not in train_set:
                    ok += int(native_argmax(recall_fn(bundle, r), fillers) == fids[r]); n += 1
        return ok / max(n, 1)

    acc = {k: [] for k in ("host", "neural", "lesion_nodend", "lesion_noteach", "scramble")}
    for split in splits:
        tr = {r: [f for (rr_, f) in split["train"] if rr_ == r] for r in range(3)}
        if min(len(tr[r]) for r in range(3)) == 0:
            continue
        train_set = set(split["train"])
        W_host, _ = _train_arm(b, inp, out, pre_of, post_of, split, tr, "host")
        W_neu, _ = _train_arm(b, inp, out, pre_of, post_of, split, tr, "neural")
        W_ld, _ = _train_arm(b, inp, out, pre_of, post_of, split, tr, "lesion_nodend")
        W_lt, _ = _train_arm(b, inp, out, pre_of, post_of, split, tr, "lesion_noteach")
        W_sc, _ = _train_arm(b, inp, out, pre_of, post_of, split, tr, "scramble")
        acc["host"].append(_eval(lambda bn, r: _recall_bridge(W_host, bn, r), train_set))
        acc["neural"].append(_eval(lambda bn, r: _recall_bridge(W_neu, bn, r), train_set))
        acc["lesion_nodend"].append(_eval(lambda bn, r: _recall_bridge(W_ld, bn, r), train_set))
        acc["lesion_noteach"].append(_eval(lambda bn, r: _recall_bridge(W_lt, bn, r), train_set))
        acc["scramble"].append(_eval(lambda bn, r: _recall_bridge(W_sc, bn, r), train_set))

    # BYTE-IDENTICAL-OFF panel (dedicated FRESH bridges, no cross-arm state): train the HOST/off path on two
    # independent fresh bridges (identical cfg.seed) -> md5 of the learned production weights must match bit-for-bit.
    # Proves the default on-bridge learning path is deterministic + unperturbed. (The load-bearing off guarantee is
    # no-sim-edit: the bridge weight-update code is byte-identical to main by construction.)
    split0 = next(s for s in splits if min(len(([f for (rr_, f) in s["train"] if rr_ == r] or [])) for r in range(3)) > 0)
    tr0 = {r: [f for (rr_, f) in split0["train"] if rr_ == r] for r in range(3)}
    bA, iA, oA, pA, qA = build_readout_bridge(two_dh, d_in, seed)
    _, md5_a = _train_arm(bA, iA, oA, pA, qA, split0, tr0, "host")
    bB, iB, oB, pB, qB = build_readout_bridge(two_dh, d_in, seed)
    _, md5_b = _train_arm(bB, iB, oB, pB, qB, split0, tr0, "host")
    byte_off_ok = (md5_a == md5_b)

    row = {"seed": seed, "stage0": True, **{k: float(np.mean(v)) for k, v in acc.items()},
           "byte_off_ok": bool(byte_off_ok), "host_md5": md5_a}
    print(f"  [seed {seed}] HOST-onbridge {row['host']:.3f} | NEURAL-onbridge {row['neural']:.3f} | "
          f"lesion-nodend {row['lesion_nodend']:.3f} | lesion-noteach {row['lesion_noteach']:.3f} | "
          f"scramble {row['scramble']:.3f} | byte-off {row['byte_off_ok']} (K={n_pop})", flush=True)
    return row


def _git_sim_clean():
    """Byte-identical-when-OFF, the load-bearing guarantee: no sim/ file changed -> the production weight-update
    path is byte-identical to main by construction."""
    try:
        out = subprocess.check_output(["git", "-C", _REPO, "status", "--short"], text=True)
    except Exception as e:
        return None, f"git status failed: {e}"
    sim_lines = [ln for ln in out.splitlines() if " sim/" in ln or ln[3:].startswith("sim/")]
    return (len(sim_lines) == 0), ("no sim/ file modified" if not sim_lines else "sim/ MODIFIED: " + "; ".join(sim_lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44,100,101,102")
    ap.add_argument("--dh", type=int, default=64)
    ap.add_argument("--gain", type=float, default=1.5)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--spike-gain", type=float, default=20.0)
    ap.add_argument("--n-error-pop", type=int, default=16,
                    help="K error-neurons pooled per output (SNR lift for the short on-bridge budget; K=1 = single soma)")
    ap.add_argument("--out", type=str, default=os.path.join(
        _REPO, "research", "findings", "raw", "_neural_error_onbridge.json"))
    args = ap.parse_args()
    os.environ.setdefault("SIM_BACKEND", "numpy")
    seeds = [int(s) for s in args.seeds.split(",")]
    codes_path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_stream_codes_320_seed42.npy")
    if not os.path.exists(codes_path):
        print(f"  [missing] {codes_path}", flush=True)
        return
    codes = np.load(codes_path).astype(np.float64)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    two_dh = 2 * args.dh
    t0 = time.time()
    print(f"[neural-error ON-BRIDGE de-risk] the read-out neurons' OWN Urbanczik-Senn soma-vs-dendrite mismatch "
          f"delivered through the live bridge's cp_per_synapse_reward_override channel. gain={args.gain} "
          f"beta={args.beta} spike_gain={args.spike_gain} D_h={args.dh} seeds={seeds}", flush=True)
    rows = [run_seed(codes, s, two_dh, args.dh, args) for s in seeds]
    rows = [r for r in rows if r.get("stage0")]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    stage0 = len(rows) == len(seeds)
    host, neural = m("host"), m("neural")
    les_d, les_t, scr = m("lesion_nodend"), m("lesion_noteach"), m("scramble")
    n_par = sum(int(r["neural"] >= 0.85 * r["host"]) for r in rows)
    bar = int(np.ceil(5 / 6 * len(seeds)))
    ref = max(neural, 1e-9)
    sep = 0.5 * ref
    lesion_d_collapses = les_d < sep
    lesion_t_collapses = les_t < sep
    scramble_collapses = scr < sep
    byte_off_ok = all(r["byte_off_ok"] for r in rows)
    sim_clean, sim_msg = _git_sim_clean()

    verdict = Verdict("neural-error ON-BRIDGE: U-S soma-vs-dendrite mismatch through cp_per_synapse_reward_override")
    verdict.control("lesion-nodend collapses (dendritic self-prediction load-bearing on-bridge)", ref, les_d, min_separation=sep)
    verdict.control("lesion-noteach collapses (somatic teaching load-bearing on-bridge)", ref, les_t, min_separation=sep)
    verdict.control("scramble collapses (per-output addressing load-bearing on-bridge)", ref, scr, min_separation=sep)
    go = stage0 and (n_par >= bar) and lesion_d_collapses and lesion_t_collapses and scramble_collapses
    decided = verdict.decide(go=go, verbose=True)
    status = decided["status"]

    attribution = {
        "dendritic_self_prediction_frac": attributable_to("dendritic self-prediction (NEURAL vs nodend)", neural, les_d),
        "somatic_teaching_frac": attributable_to("somatic teaching (NEURAL vs noteach)", neural, les_t),
        "per_output_addressing_frac": attributable_to("per-output addressing (NEURAL vs scramble)", neural, scr),
    }
    print(f"\n{'='*108}", flush=True)
    print(f"  STAGE 0 (delta-rule linchpin, all seeds): {'PASS' if stage0 else 'FAIL'}", flush=True)
    print(f"  MEAN ({len(rows)} seeds): HOST-onbridge {host:.3f} | NEURAL-onbridge {neural:.3f} | "
          f"lesion-nodend {les_d:.3f} | lesion-noteach {les_t:.3f} | scramble {scr:.3f} | "
          f"NEURAL>=0.85x HOST: {n_par}/{len(rows)} | (numpy de-risk ref NEURAL=0.964)", flush=True)
    print(f"  BYTE-IDENTICAL-OFF: no-sim-edit={sim_clean} ({sim_msg}) | HOST/off md5 reproducible on fresh bridges={byte_off_ok}", flush=True)
    print(f"  ATTRIBUTION: dendritic self-prediction {attribution['dendritic_self_prediction_frac']:.3f} | "
          f"somatic teaching {attribution['somatic_teaching_frac']:.3f} | "
          f"per-output addressing {attribution['per_output_addressing_frac']:.3f}", flush=True)
    if go:
        print(f"  GO: the read-out neurons' OWN U-S soma-vs-dendrite mismatch drives the LIVE bridge's three-factor "
              f"plasticity as well as the host error -- NEURAL-onbridge {neural:.3f} = {neural/max(host,1e-9):.0%} of "
              f"HOST-onbridge in {n_par}/{len(rows)} seeds; silencing the dendritic self-prediction ({les_d:.3f}) or "
              f"the somatic teaching ({les_t:.3f}) or mis-addressing it ({scr:.3f}) all collapse on-bridge learning. "
              f"==> the host subtraction is retired on the production path; the brain computes the error.", flush=True)
    elif not (lesion_d_collapses and lesion_t_collapses and scramble_collapses):
        print(f"  INVALID: a control did NOT collapse -- the neural error may not be load-bearing on-bridge.", flush=True)
    else:
        print(f"  BOUNDARY (honest negative): the on-bridge U-S error under-performs the host error ({neural:.3f} vs "
              f"{host:.3f} = {neural/max(host,1e-9):.0%}) while all controls collapse -- residual = finite-nudging + "
              f"spiking-soma noise carried through the live plasticity. This maps what the on-bridge substrate error can do.",
              flush=True)
    print(f"  EARNED STATUS (preconditions guard): {status}", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*108}", flush=True)
    out = {"verdict": status, "go_headline": bool(go), "preconditions": decided["preconditions"],
           "attribution": attribution, "chance": 1.0 / F, "byte_off_ok": bool(byte_off_ok),
           "no_sim_edit": bool(sim_clean) if sim_clean is not None else None, "no_sim_edit_msg": sim_msg,
           "seeds": seeds, "D_h": args.dh, "gain": args.gain, "beta": args.beta, "spike_gain": args.spike_gain,
           "n_error_pop": int(args.n_error_pop), "lr": LR, "n_passes": N_PASSES, "host": host, "neural": neural,
           "numpy_derisk_ref": 0.964,
           "lesion_nodend": les_d, "lesion_noteach": les_t, "scramble": scr, "n_parity": n_par, "per_seed": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
