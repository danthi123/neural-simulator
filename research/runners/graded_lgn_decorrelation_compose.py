"""GRADED LGN decorrelation stage — on-substrate, biology-faithful realization of the validated whitening rule,
gated on COMPOSITION (2026-06-06).

The rate model (research/findings/2026-06-06-option1-local-learning-whitening-VALIDATED-6seed.md) established
6/6 that a regularized LOCAL rule
    ΔM_ij ∝ <a_i a_j> - delta_ij - lambda·M_ij        (settled a inhibited by -(M @ a))
learns a GENTLE partial whitening that decorrelates the 320 grounded CIFAR codes WITHOUT over-amplifying noise
and COMPOSES the conversational benchmark at 100%. The prior ON-BRIDGE attempt realized this as a SHARED-FS
SPIKING lateral (it->fs->it) and hit a BOUNDARY (2026-06-06): a shared-FS spiking lateral does GLOBAL gain
control, not the pairwise M_ij the composing whitening needs (the Mikulasch-Priesemann wall) -> 66.7% = raw floor.

THIS runner uses the NEW additive sim/ mechanism (BrainRegion.graded_lateral + cfg.enable_graded_lateral): a
per-region GRADED pairwise lateral inhibition operating on the LGN region's SUB-THRESHOLD ANALOG activity
(a = clip((v-v_rest)/scale, 0, 1), NOT spikes), added pre-spike, with the plastic KxK M learning the exact rule
above. This is where the retina/LGN does variance equalization (the rate-code opponency wall: the common-mode
subtraction must be analog, pre-spike — a rate code physically can't). The graded lateral does the precise
PAIRWISE subtraction the spiking lateral could not.

Pipeline: CIFAR real-object V1 codes (Track A) -> fixed projection P -> drive the graded LGN region -> the lateral
settles -> READ the whitened LGN activity -> composer/agent -> composition.

TWO readouts (the graded->spiking re-correlation risk the task flagged is tested EXPLICITLY):
  - GRADED readout : the settled analog activity `a` (post-lateral) — the design's intent (graded LGN output).
  - SPIKING readout: the LGN spike counts over the read window — tests whether re-spiking to cortex
                     RE-CORRELATES the whitened gain away (the honest risk).

RIGOR (the arc caught FIVE convenient-but-wrong results — do not ship a sixth):
  1. GATE ON COMPOSITION (the agent benchmark %), NEVER coherence (it misled 3x; a noise-collapse has low
     coherence but does NOT compose).
  2. CONTROLS bracket every result: RAW grounded codes (~=66.7% floor) + CONCEPT-whiten (~=100% target). If those
     two are off, the harness is broken -> distrust everything.
  3. GUARDS every run: LGN graded activity (NOT silent / NOT blown up) + M norm (bounded) + a NO-LATERAL
     baseline (M forced 0) to attribute any lift to the LEARNED graded lateral specifically.
  4. A great composition with a SILENT/DEGENERATE LGN is the false positive to catch — check guards first.
  5. Multi-seed >=3 (toward 6) before any GO. Heavy GPU jobs SEQUENTIAL (2 concurrent OOM).

  REALOBJ_CIFAR=data/cifar10/cifar-10-batches-py/data_batch_1 \
      python -m research.runners.graded_lgn_decorrelation_compose --seeds 42 43 44 --baseline --out out.json
"""
from __future__ import annotations
import argparse
import json

import numpy as np

from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
from sim.regions import BrainRegion
from sim.bridge import SimulationBridge
from sim.backend import get_backend, to_host

from research.runners.unified_agent_realobject_grounded import build_realobject_features, run_seed
from research.runners.unified_agent_visual_grounded import _decorrelate
from research.runners.unified_agent_benchmark import build_vocab, aggregate
from research.runners._visual_grounding_probe import _v1_matrix


# ----------------------------------------------------------------------------------------------------
# Bridge build: ONE graded-LGN region (K neurons) flagged with graded_lateral=True. The fixed random
# projection P (the rate-model x_proj = feats @ P) is computed in numpy and applied as the EXTERNAL DRIVE
# (the rate-model's x is the recurrence input, applied as drive; M is the only learned weight). The plastic
# KxK M is the new cp_graded_lateral_M — a GRADED pairwise lateral, NOT a spiking it->fs->it loop.
# ----------------------------------------------------------------------------------------------------
def build_graded_lgn_bridge(seed, K, lam, lr, gain_pA, act_scale, coact_ema=0.0):
    """1-region spiking bridge whose ONLY learned weights are the GRADED lateral M (cp_graded_lateral_M).

    The region's neurons spike (Izhikevich), but the WHITENING happens in the GRADED domain: -(M @ a) is
    added to the membrane drive BEFORE the spike threshold, a = clip((v-v_rest)/scale, 0, 1) is the
    sub-threshold analog activity, and M learns ΔM ∝ <a aT> - I - lambdaM. No FS pool, no it->fs->it loop.
    """
    region = BrainRegion(name="lgn", n_neurons=K, exc_fraction=1.0, internal_density=0.0,
                         exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                         plastic_internal=False, graded_lateral=True)
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [region]
    cfg.region_pathways = []
    cfg.connections_per_neuron = 0          # region-framework signal: wiring injected (here: none)
    cfg.dt_ms = 0.5
    cfg.seed = int(seed)
    cfg.enable_nmda = False
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = False     # the GRADED lateral is the only plasticity (its own knobs below)
    cfg.enable_homeostasis = False
    cfg.enable_reward_modulation = False
    cfg.enable_structural_plasticity = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_synaptic_scaling = False
    cfg.enable_ou_process = False
    cfg.ou_std_current_pA = 0.0
    cfg.fast_spike_reset = True
    # ── the GRADED LGN decorrelation knobs (the new sim/ mechanism) ──
    cfg.enable_graded_lateral = True
    cfg.graded_lateral_lr = lr
    cfg.graded_lateral_lambda = lam         # the -lambdaM (settles the gentle, bounded fixed point)
    cfg.graded_lateral_gain_pA = gain_pA
    cfg.graded_lateral_act_scale = act_scale
    cfg.graded_lateral_coact_ema = coact_ema
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    assert b.is_initialized, "graded-LGN bridge init failed"
    assert b.cp_graded_lateral_M is not None, "graded lateral M not allocated (flag/opt-in?)"
    return b


def make_projection(n_feat, K, seed):
    """The FIXED random projection P (the rate-model's `feats @ P`)."""
    return np.random.default_rng(seed).standard_normal((n_feat, K)) / np.sqrt(n_feat)


def project_drive(feats, P, signed=False):
    """feats[N,n_feat] @ P[n_feat,K] -> peak-normalized per-batch DRIVE for LGN (>=0 pA scale).
    signed=False: rectified K-dim drive max(x,0) — drops the SIGN (the lossy realization).
    signed=True : ON/OFF 2K-dim drive [max(x,0) | max(-x,0)] — a 2K LGN pool PRESERVES the sign of the
                  projected code (the rate-model's x is signed). The readout recombines a[:K]-a[K:] to
                  recover the signed whitened code, so the graded lateral whitens in the SIGNED domain
                  (the rate-code opponency wall: a rate code can't subtract a signed common mode unless the
                  sign is carried by an ON/OFF pair)."""
    x = feats @ P                                   # signed real projection (the rate-model x)
    if signed:
        d = np.concatenate([np.maximum(x, 0.0), np.maximum(-x, 0.0)], axis=1)   # [N, 2K]
    else:
        d = np.maximum(x, 0.0)                       # [N, K] rectified (drop sign)
    return d / (float(d.max()) + 1e-9)              # peak 1.0 -> strongest dim ~= drive_scale pA


def read_codes(b, lgn_idx, drive_vec, scale, window, settle):
    """Drive LGN with the projected code (the rate-model x); the GRADED lateral settles (-(M@a) subtracts the
    common mode pre-spike); then read BOTH the settled GRADED activity `a` AND the spike counts over `window`.
    Returns (a_graded[n_it], spikes[n_it]) — the RAW per-neuron readouts (recombination for the signed ON/OFF
    pool happens in graded_lgn_codes). a_graded is the time-averaged analog activity (the whitened LGN output);
    spikes is the rate code (the graded->spiking readout, for the re-correlation risk)."""
    cp, _ = get_backend()
    start, end = b._graded_lateral_slice
    ext = cp.zeros(b.cp_external_input_current.shape[0], dtype=cp.float32)
    ext[cp.asarray(lgn_idx, dtype=cp.int64)] = cp.asarray(drive_vec * scale, dtype=cp.float32)
    b.cp_external_input_current[:] = ext
    # Let the lateral settle first (drive on, no read).
    for _ in range(settle):
        b._run_one_simulation_step()
    # Read window: accumulate the GRADED analog activity a AND the spikes.
    a_acc = np.zeros(end - start)
    sp_acc = np.zeros(end - start)
    for _ in range(window):
        b._run_one_simulation_step()
        a_acc += np.asarray(to_host(b._graded_lateral_activity())).astype(float)
        sp_acc += np.asarray(to_host(b.cp_firing_states)).astype(float)[lgn_idx]
    b.cp_external_input_current[:] = 0.0
    for _ in range(settle):
        b._run_one_simulation_step()
    return a_acc / window, sp_acc


def _recombine(vec, K, signed):
    """ON/OFF recombination: signed -> vec[:K]-vec[K:] (recover the signed code); else vec (rectified)."""
    return (vec[:K] - vec[K:]) if signed else vec


def coherence(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    G = np.abs(Xn @ Xn.T)
    off = G[~np.eye(len(X), dtype=bool)]
    return float(off.mean()), float(off.max())


def graded_lgn_codes(feats, seed, K, lam, lr, gain_pA, act_scale, epochs, drive_scale, window, settle,
                     proj_seed, coact_ema=0.0, return_baseline=False, signed=False):
    """FIXED numpy projection feats@P -> direct LGN drive; LEARN the GRADED lateral M from the settled analog
    activity over `epochs` passes; then READ the whitened LGN codes (graded + spiking). Returns
    (a_codes[N,K], sp_codes[N,K], guards[, a_nolat, sp_nolat]).

    signed=True drives a 2K LGN pool with ON/OFF [max(x,0)|max(-x,0)] so the rate code carries the SIGN; the
    graded lateral whitens over 2K dims; the readout recombines a[:K]-a[K:] back to a K-dim signed code.

    return_baseline=True also reads codes from a FRESH bridge with the lateral DISABLED (gain 0 + lr 0, M
    stays 0) so the caller attributes the decorrelation to the LEARNED graded lateral, not the spiking/drive
    nonlinearity."""
    P = make_projection(feats.shape[1], K, proj_seed)
    drives = project_drive(feats, P, signed=signed)  # [N, K] or [N, 2K] in [0,1]
    n_it = 2 * K if signed else K
    b = build_graded_lgn_bridge(seed, n_it, lam, lr, gain_pA, act_scale, coact_ema=coact_ema)
    lgn = np.asarray(b.region_manager.indices("lgn"))

    # cold-start probe.
    a0, sp0 = read_codes(b, lgn, drives[0], drive_scale, window, settle)
    m0 = float(np.abs(to_host(b.cp_graded_lateral_M)).max())
    print(f"  [seed={seed}] cold-start concept0: graded a>0 {int((a0 > 0).sum())}/{n_it}, "
          f"spikes>0 {int((sp0 > 0).sum())}/{n_it}, M_max0={m0:.4f}", flush=True)

    # LEARN the graded lateral (lr is on; -lambdaM bounds it). Each pass drives the per-concept projected code; the
    # lateral learns from the settled analog co-activity <a aT>.
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for i in rng.permutation(len(feats)):
            read_codes(b, lgn, drives[i], drive_scale, window, settle)

    # FREEZE the lateral (lr=0) and READ the settled whitened codes. raw_* are the per-neuron (n_it) readouts;
    # a_codes/sp_codes are recombined to K dims (signed: a[:K]-a[K:]).
    b.core_config.graded_lateral_lr = 0.0
    a_raw = np.zeros((len(feats), n_it))
    sp_raw = np.zeros((len(feats), n_it))
    for i in range(len(feats)):
        a_raw[i], sp_raw[i] = read_codes(b, lgn, drives[i], drive_scale, window, settle)
    a_codes = np.stack([_recombine(a_raw[i], K, signed) for i in range(len(feats))])
    sp_codes = np.stack([_recombine(sp_raw[i], K, signed) for i in range(len(feats))])

    M = to_host(b.cp_graded_lateral_M)
    a_active = (a_raw > 1e-6).sum(1)
    sp_active = (sp_raw > 0).sum(1)
    guards = {
        "mean_graded_active": float(a_active.mean()), "min_graded_active": int(a_active.min()),
        "n_silent_graded": int((a_active == 0).sum()),
        "mean_spike_active": float(sp_active.mean()), "min_spike_active": int(sp_active.min()),
        "n_silent_spiking": int((sp_active == 0).sum()), "total_spikes": int(sp_raw.sum()),
        "M_norm": float(np.linalg.norm(M)), "M_max": float(np.abs(M).max()),
        "M_offdiag_max": float(np.abs(M[~np.eye(n_it, dtype=bool)]).max()),
        "drive_coh_mean": coherence(drives)[0], "drive_coh_max": coherence(drives)[1],
        "graded_coh_mean": coherence(a_codes)[0], "graded_coh_max": coherence(a_codes)[1],
        "spike_coh_mean": coherence(sp_codes)[0], "spike_coh_max": coherence(sp_codes)[1],
    }
    if return_baseline:
        b2 = build_graded_lgn_bridge(seed, n_it, lam, 0.0, 0.0, act_scale, coact_ema=coact_ema)  # gain 0, lr 0
        lgn2 = np.asarray(b2.region_manager.indices("lgn"))
        a_nolat_raw = np.zeros((len(feats), n_it))
        sp_nolat_raw = np.zeros((len(feats), n_it))
        for i in range(len(feats)):
            a_nolat_raw[i], sp_nolat_raw[i] = read_codes(b2, lgn2, drives[i], drive_scale, window, settle)
        a_nolat = np.stack([_recombine(a_nolat_raw[i], K, signed) for i in range(len(feats))])
        sp_nolat = np.stack([_recombine(sp_nolat_raw[i], K, signed) for i in range(len(feats))])
        guards["nolat_graded_coh_mean"] = coherence(a_nolat)[0]
        guards["nolat_spike_coh_mean"] = coherence(sp_nolat)[0]
        return a_codes, sp_codes, guards, a_nolat, sp_nolat
    return a_codes, sp_codes, guards


def compose(codes, seeds, tokens, nouns, verbs, adjs):
    """Project codes -> phases -> NestedCompositionAgent -> the full capability benchmark (% correct)."""
    d = codes.shape[1]
    seed_res = [run_seed(s, codes, d, tokens, nouns, verbs, adjs, decorrelate=False) for s in seeds]
    _, gok, gtot = aggregate(seed_res)
    return gok, gtot


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--K", type=int, default=300, help="LGN-pool / subspace dimension (<= N=320)")
    ap.add_argument("--lam", type=float, default=0.01, help="graded_lateral_lambda = the -lambdaM (KEY: stable+bounded)")
    ap.add_argument("--lr", type=float, default=0.02, help="graded_lateral_lr = η for ΔM ∝ <a aT>-I")
    ap.add_argument("--gain", type=float, default=10.0, help="graded_lateral_gain_pA (inhibition strength; high silences)")
    ap.add_argument("--act-scale", type=float, default=15.0, help="graded_lateral_act_scale (mV->a normalizer)")
    ap.add_argument("--coact-ema", type=float, default=0.0, help="EMA decay for <a aT> (0=instantaneous)")
    ap.add_argument("--epochs", type=int, default=8, help="passes over the 320 codes to learn the lateral")
    ap.add_argument("--drive-scale", type=float, default=1400.0, help="pA scale for the projected LGN drive")
    ap.add_argument("--window", type=int, default=40, help="read-window steps (graded + spike accumulation)")
    ap.add_argument("--settle", type=int, default=15, help="settle steps before the read window")
    ap.add_argument("--baseline", action="store_true",
                    help="also read a NO-lateral baseline (M=0) to attribute decorrelation to the learned lateral")
    ap.add_argument("--signed", action="store_true",
                    help="ON/OFF 2K LGN pool preserving the projected code's SIGN (vs the lossy rectified drive)")
    ap.add_argument("--bench-seeds", type=int, nargs="+", default=None,
                    help="seeds for the composition agent (default: same as --seeds)")
    ap.add_argument("--out", default="research/findings/raw/_graded_lgn_decorrelation_compose.json")
    args = ap.parse_args()
    bench_seeds = args.bench_seeds if args.bench_seeds is not None else args.seeds

    nouns, verbs, adjs = build_vocab()
    W, _ = _v1_matrix()
    feats, dim, tokens, src = build_realobject_features(nouns, verbs, adjs, W, seed=args.seeds[0])
    rawm, rawx = coherence(feats)
    print(f"=== GRADED LGN decorrelation COMPOSITION gate | grounding={src} | {len(tokens)} concepts | K={args.K} "
          f"| lam(-lambdaM)={args.lam} lr={args.lr} gain={args.gain}pA ===", flush=True)
    print(f"  raw feature coherence: mean {rawm:.3f}, max {rawx:.3f}", flush=True)

    out = {"source": src, "K": args.K, "lam": args.lam, "lr": args.lr, "gain": args.gain,
           "act_scale": args.act_scale, "coact_ema": args.coact_ema, "epochs": args.epochs,
           "signed": bool(args.signed), "seeds": args.seeds, "bench_seeds": bench_seeds,
           "raw_coherence": [rawm, rawx],
           "params": {k: getattr(args, k) for k in ("drive_scale", "window", "settle")}}

    # ---- CONTROLS (bracket every result; if these are off, the harness is broken) ----
    raw_ok, raw_tot = compose(feats, bench_seeds, tokens, nouns, verbs, adjs)
    out["RAW"] = [raw_ok, raw_tot]
    print(f"  {'RAW grounded (floor control)':<44} {raw_ok}/{raw_tot} = {raw_ok/raw_tot*100:.1f}%   "
          f"(expect ~66.7%)", flush=True)
    cw_ok, cw_tot = compose(_decorrelate(feats), bench_seeds, tokens, nouns, verbs, adjs)
    out["CONCEPT-whiten"] = [cw_ok, cw_tot]
    print(f"  {'CONCEPT-whiten (100% target control)':<44} {cw_ok}/{cw_tot} = {cw_ok/cw_tot*100:.1f}%   "
          f"(expect ~100%)", flush=True)
    harness_ok = (raw_ok / raw_tot < 0.85) and (cw_ok / cw_tot > 0.9)
    print(f"  harness sanity: {'OK (controls bracket as expected)' if harness_ok else 'BROKEN - distrust below'}",
          flush=True)

    # ---- GRADED LGN learned whitening, per seed (heavy; SEQUENTIAL) ----
    out["GRADED_LGN"] = {}
    for s in args.seeds:
        res = graded_lgn_codes(
            feats, s, args.K, args.lam, args.lr, args.gain, args.act_scale, args.epochs,
            args.drive_scale, args.window, args.settle, proj_seed=s, coact_ema=args.coact_ema,
            return_baseline=args.baseline, signed=args.signed)
        a_codes, sp_codes, guards = res[0], res[1], res[2]
        # GUARD: silent / blown-up LGN is a FALSE POSITIVE regardless of composition.
        degenerate = (guards["n_silent_graded"] > 0 or guards["mean_graded_active"] < 2.0
                      or not np.isfinite(guards["M_norm"]) or guards["M_max"] > 1e3)
        g_ok, g_tot = compose(a_codes, bench_seeds, tokens, nouns, verbs, adjs)          # GRADED readout
        sp_ok, sp_tot = compose(sp_codes, bench_seeds, tokens, nouns, verbs, adjs)       # SPIKING readout
        entry = {"graded_compose": [g_ok, g_tot], "spiking_compose": [sp_ok, sp_tot],
                 "guards": guards, "degenerate": bool(degenerate)}
        nolat_str = ""
        if args.baseline:
            a_nolat, sp_nolat = res[3], res[4]
            ng_ok, ng_tot = compose(a_nolat, bench_seeds, tokens, nouns, verbs, adjs)
            nsp_ok, nsp_tot = compose(sp_nolat, bench_seeds, tokens, nouns, verbs, adjs)
            entry["nolat_graded_compose"] = [ng_ok, ng_tot]
            entry["nolat_spiking_compose"] = [nsp_ok, nsp_tot]
            nolat_str = (f"  [NO-lateral: graded {ng_ok}/{ng_tot}={ng_ok/ng_tot*100:.1f}% "
                         f"(coh {guards['nolat_graded_coh_mean']:.3f}), spiking {nsp_ok}/{nsp_tot}="
                         f"{nsp_ok/nsp_tot*100:.1f}%]")
        out["GRADED_LGN"][str(s)] = entry
        flag = "  ** DEGENERATE LGN (false-positive risk)" if degenerate else ""
        print(f"  [seed={s}] GRADED readout: {g_ok}/{g_tot} = {g_ok/g_tot*100:.1f}%  |  "
              f"SPIKING readout: {sp_ok}/{sp_tot} = {sp_ok/sp_tot*100:.1f}%", flush=True)
        print(f"            guards: graded_active={guards['mean_graded_active']:.1f}/{args.K} "
              f"(min {guards['min_graded_active']}, silent {guards['n_silent_graded']}/{len(feats)}) | "
              f"M_norm={guards['M_norm']:.2f} (max {guards['M_max']:.2f}, offdiag {guards['M_offdiag_max']:.2f}) | "
              f"drive_coh={guards['drive_coh_mean']:.3f}->graded_coh={guards['graded_coh_mean']:.3f} "
              f"(spike_coh={guards['spike_coh_mean']:.3f}){flag}{nolat_str}", flush=True)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {args.out}", flush=True)

    # ---- VERDICT (gate on the GRADED readout — the design's intent; report spiking as the re-correlation diag) ----
    g_rates = [v["graded_compose"][0] / v["graded_compose"][1] for v in out["GRADED_LGN"].values()]
    sp_rates = [v["spiking_compose"][0] / v["spiking_compose"][1] for v in out["GRADED_LGN"].values()]
    degen = any(v["degenerate"] for v in out["GRADED_LGN"].values())
    raw_rate = raw_ok / raw_tot
    cw_rate = cw_ok / cw_tot
    print("\n" + "=" * 80, flush=True)
    print(f"VERDICT (GRADED LGN learned whitening, {len(args.seeds)} seed(s)):", flush=True)
    print(f"  RAW floor {raw_rate*100:.1f}%  |  CONCEPT-whiten target {cw_rate*100:.1f}%  |  rate-model 100%",
          flush=True)
    print(f"  GRADED readout per-seed: {['%.1f%%' % (r*100) for r in g_rates]}  (mean {np.mean(g_rates)*100:.1f}%)",
          flush=True)
    print(f"  SPIKING readout per-seed: {['%.1f%%' % (r*100) for r in sp_rates]}  (mean {np.mean(sp_rates)*100:.1f}%)"
          f"  [the graded->spiking re-correlation diagnostic]", flush=True)
    if not harness_ok:
        verdict = "INVALID - controls did not bracket (harness broken)"
    elif degen:
        verdict = "FALSE-POSITIVE RISK - a seed had a degenerate/silent LGN; see guards"
    elif np.mean(g_rates) >= 0.95 and min(g_rates) >= 0.90:
        verdict = "GO - the GRADED LGN learned whitening COMPOSES (>=95% mean, >=90% min) == target"
    elif np.mean(g_rates) > raw_rate + 0.1:
        verdict = f"PARTIAL - composes ABOVE the raw floor but below target (graded mean {np.mean(g_rates)*100:.1f}%)"
    else:
        verdict = (f"BOUNDARY - the GRADED readout does NOT compose above the raw floor "
                   f"(graded mean {np.mean(g_rates)*100:.1f}% vs raw {raw_rate*100:.1f}%)")
    print(f"  => {verdict}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
