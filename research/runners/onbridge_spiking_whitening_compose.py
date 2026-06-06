"""ON-BRIDGE SPIKING realization of the VALIDATED rate-model regularized-local-whitening rule (2026-06-06).

The rate model (research/findings/2026-06-06-option1-local-learning-whitening-VALIDATED-6seed.md +
research/findings/raw/_A_whitening_compose_gate.py) established, 6/6 seeds: a REGULARIZED local rule
    ΔM_ij ∝ ⟨y_i y_j⟩ − δ_ij − λ·M_ij        (settled y = (I+M)^-1 x)
learns a GENTLE partial whitening (≈C^-1/3) that decorrelates the 320-concept grounded codes WITHOUT
over-amplifying noise, and COMPOSES the conversational benchmark end-to-end at 100%. The −λM synaptic
weight-decay is ESSENTIAL: without it the rule over-whitens (→ 66.7% = raw); the full whitening C^-1/2
also over-whitens (→ 66.7%). The decay settles the gentler fixed point that composes.

THIS RUNNER realizes that LEARNING on the spiking bridge and tests whether it still COMPOSES.

Mapping (rule term → spiking-bridge mechanism, NO sim/ edits):
  - K spiking IT neurons (region `it`)            = the K-subspace code dims y.
  - FIXED random projection inp→it (non-plastic)  = the rate-model projection P (x_proj = feats @ P).
                                                    Non-plastic so the GLOBAL Hebbian weight-decay does
                                                    NOT touch it — only the lateral M decays (=−λM exactly).
  - PLASTIC anti-Hebbian lateral it→fs→it         = M. it→fs is Hebbian-plastic (co-firing y_i y_j
    (gate "lat")                                    strengthens shared inhibition); fs→it fixed inhibitory
                                                    delivers the common-mode subtraction. Bridge Hebbian
                                                    Δw = η(w_max−w) on co-fire = the ⟨y_i y_j⟩ drive.
  - HOMEOSTASIS (cfg.enable_homeostasis)          = the identity-target DIAGONAL (y_i^2→1, unit variance:
                                                    threshold adaptation toward a target rate keeps every
                                                    IT neuron active, no dead/dominant units).
  - hebbian_weight_decay = λ  (THE KEY ADDITION)  = the −λM term. The prior on-bridge de-grounding attempt
                                                    (_A_spiking_decorrelation/_A_spiking_functional_gate)
                                                    had an UNSTABLE anti-Hebbian lateral (no fixed point);
                                                    this adds the decay so the lateral is STABLE + BOUNDED.

Grounded codes = CIFAR real-object V1 (Track A). Reuse build_realobject_features + run_seed.

RIGOR (the rate-model arc caught FIVE convenient-but-wrong results — do not ship a sixth):
  1. GATE ON COMPOSITION (the agent benchmark %), NEVER coherence (it misled 3x).
  2. CONTROLS bracket every result: RAW grounded codes (≈66.7% floor) + CONCEPT-whiten (≈100% target).
     If those two are off, the harness is broken → distrust everything.
  3. GUARDS every run: IT-pool firing stats (NOT silent, NOT blown up) + learned-lateral weight norm
     (bounded). A great composition with a silent/degenerate IT pool is a FALSE POSITIVE → flagged.
  4. Multi-seed ≥3 (toward 6) before any GO. Heavy GPU jobs SEQUENTIAL (2 concurrent OOM).

  REALOBJ_CIFAR=data/cifar10/cifar-10-batches-py/data_batch_1 \
      python -m research.runners.onbridge_spiking_whitening_compose --seeds 42 43 44 --out out.json
"""
from __future__ import annotations
import argparse
import json

import numpy as np

from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
from sim.regions import BrainRegion, RegionPathway
from sim.bridge import SimulationBridge
from sim.backend import get_backend, to_host

from research.runners.unified_agent_realobject_grounded import build_realobject_features, run_seed
from research.runners.unified_agent_visual_grounded import _decorrelate
from research.runners.unified_agent_benchmark import build_vocab, aggregate
from research.runners._visual_grounding_probe import _v1_matrix


# ----------------------------------------------------------------------------------------------------
# Bridge build: it (K IT neurons) <-> fs (FS interneurons) lateral, the ONLY learned weights.
#
# The fixed random projection P (rate-model x_proj = feats @ P) is computed in numpy and applied as the
# EXTERNAL DRIVE to IT — NOT as a synaptic pathway. This is both more faithful to the rate rule (x is the
# recurrence INPUT, applied as drive; M is the only learned weight) AND sidesteps a real bridge gotcha:
# the global Hebbian block unconditionally clips ALL cp_connections to [hebbian_min, hebbian_max] every
# step (the CLAUDE.md soft-bound w_max gotcha, in Hebbian form), which would collapse a large fixed
# projection weight regardless of the plasticity gate. With only the small lateral in cp_connections,
# the clip range bounds the lateral as intended and nothing else.
# ----------------------------------------------------------------------------------------------------
def build_whitening_bridge(seed, K, n_fs, lam, lat_density, lat_weight, fs_inh_weight,
                           hebbian_lr, hebbian_max):
    """2-region spiking bridge: the PLASTIC anti-Hebbian it<->fs lateral = the learned whitening M.

    it->fs : PLASTIC Hebbian (gate "lat") = the anti-Hebbian lateral M. Δw=η(w_max−w) on co-fire (y_i y_j)
             is the ⟨y_i y_j⟩ drive (co-active IT pairs strengthen shared FS drive); the GLOBAL Hebbian
             weight-decay cp_connections.data *= (1−λ) is the −λM regularizer (gain 1 here = full decay).
    fs->it : FIXED inhibitory return (gate "fixed", gain 0 = protected from decay) = the common-mode
             subtraction that realizes the (I+M)^-1 settling.
    enable_homeostasis = the identity-target DIAGONAL (each IT neuron held active ~target rate).
    """
    regions = [
        BrainRegion(name="it", n_neurons=K, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
        BrainRegion(name="fs", n_neurons=n_fs, exc_fraction=0.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
    ]
    pathways = [
        # PLASTIC anti-Hebbian lateral (gate "lat", gain 1 -> full −λM decay). Starts small (lat_weight) and
        # GROWS from co-firing toward hebbian_max, the −λM decay settling a bounded fixed point.
        RegionPathway(from_region="it", to_region="fs", density=lat_density, weight_mean=lat_weight,
                      weight_jitter=lat_weight * 0.3, plastic=True, plasticity_gate="lat"),
        # FIXED inhibitory return (gate "fixed" -> set to gain 0 so the global decay leaves it intact).
        RegionPathway(from_region="fs", to_region="it", density=lat_density, weight_mean=fs_inh_weight,
                      weight_jitter=fs_inh_weight * 0.2, plastic=False, plasticity_gate="fixed"),
    ]
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5
    cfg.seed = int(seed)
    cfg.enable_nmda = False
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = True          # the lateral learns Hebbianly (co-firing -> strengthen)
    cfg.hebbian_learning_rate = hebbian_lr
    cfg.hebbian_max_weight = hebbian_max        # bounds ONLY the lateral now (the fixed proj is a numpy drive)
    cfg.hebbian_min_weight = 0.0
    cfg.hebbian_weight_decay = lam              # ★ THE −λM TERM (the key addition the prior attempt lacked)
    cfg.enable_homeostasis = True               # ★ the identity-target diagonal (unit variance / no dead units)
    cfg.enable_reward_modulation = False
    cfg.enable_structural_plasticity = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_synaptic_scaling = False
    cfg.ou_std_current_pA = 0.0
    cfg.fast_spike_reset = True
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    return b


def make_projection(n_feat, K, seed):
    """The FIXED random projection P (the rate-model's `feats @ P`). Same form as _A_whitening_compose_gate._proj."""
    return np.random.default_rng(seed).standard_normal((n_feat, K)) / np.sqrt(n_feat)


def project_drive(feats, P, signed=False):
    """feats[N,n_feat] @ P[n_feat,K] -> peak-normalized per-batch DRIVE for IT (>=0 = excitatory drive).
    signed=False: rectified K-dim drive (spike rates are non-negative; the honest lossy realization).
    signed=True : ON/OFF 2K-dim drive [max(x,0) | max(-x,0)] -> a 2K IT pool preserves the SIGN of the
                  projected code (the rate-model's y is signed). The read recombines ON-OFF (see it_code)."""
    x = feats @ P                                   # signed real projection (the rate-model x)
    if signed:
        d = np.concatenate([np.maximum(x, 0.0), np.maximum(-x, 0.0)], axis=1)   # [N, 2K]
    else:
        d = np.maximum(x, 0.0)                       # [N, K] rectified (drop sign)
    return d / (float(d.max()) + 1e-9)               # peak 1.0 -> strongest dim ~= drive_scale pA


def it_code(b, it_idx, drive_vec, scale=2500.0, window=40, settle=15):
    """Drive IT directly with the projected K-dim code (the rate-model x); the plastic lateral settles
    (FS subtracts the common mode); accumulate IT firing over `window` steps = the settled code y."""
    cp, _ = get_backend()
    ext = cp.zeros(b.cp_external_input_current.shape[0], dtype=cp.float32)
    ext[cp.asarray(it_idx, dtype=cp.int64)] = cp.asarray(drive_vec * scale, dtype=cp.float32)
    b.cp_external_input_current[:] = ext
    acc = np.zeros(len(it_idx))
    for _ in range(window):
        b._run_one_simulation_step()
        acc += np.asarray(to_host(b.cp_firing_states)).astype(float)[it_idx]
    b.cp_external_input_current[:] = 0.0
    for _ in range(settle):
        b._run_one_simulation_step()
    return acc


def _lateral_weight_norm(b):
    """Frobenius-ish norm of the learned it->fs lateral weights (the M-bound guard)."""
    rm = b.region_manager
    it = set(int(i) for i in rm.indices("it"))
    fs = set(int(i) for i in rm.indices("fs"))
    coo = b.cp_connections.tocoo()
    rows = np.asarray(to_host(coo.row)); cols = np.asarray(to_host(coo.col))
    data = np.asarray(to_host(coo.data))
    mask = np.array([(int(r) in it and int(c) in fs) for r, c in zip(rows, cols)])
    w = data[mask] if mask.any() else np.zeros(1)
    return float(np.linalg.norm(w)), float(np.abs(w).max() if w.size else 0.0), float(w.mean() if w.size else 0.0)


def coherence(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    G = np.abs(Xn @ Xn.T)
    off = G[~np.eye(len(X), dtype=bool)]
    return float(off.mean()), float(off.max())


def _recombine(acc, K, signed):
    """ON/OFF read: signed -> acc[:K]-acc[K:] (recover sign); else acc (rectified rates)."""
    return (acc[:K] - acc[K:]) if signed else acc


def spiking_whiten_codes(feats, seed, K, n_fs, lam, epochs, drive_scale, window,
                         lat_density, lat_weight, fs_inh_weight, proj_seed,
                         hebbian_lr, hebbian_max, return_baseline=False, signed=False):
    """FIXED numpy projection feats@P -> direct IT drive; LEARN the it<->fs lateral M from the (noisy
    spiking) co-firing over `epochs` passes; then read the settled IT codes. Returns (codes[N,K], guards).
    signed=True uses an ON/OFF 2K IT pool so the rate code preserves the SIGN of the projected value.
    return_baseline=True also reads codes with the lateral DISABLED (gate lat=0 the whole time, no learning)
    so the caller can attribute the decorrelation to the LEARNED lateral, not the spiking nonlinearity."""
    P = make_projection(feats.shape[1], K, proj_seed)
    drives = project_drive(feats, P, signed=signed)  # [N, K] or [N, 2K] in [0,1]
    n_it = 2 * K if signed else K
    b = build_whitening_bridge(seed, n_it, n_fs, lam, lat_density, lat_weight, fs_inh_weight,
                               hebbian_lr, hebbian_max)
    rm = b.region_manager
    it = np.asarray(rm.indices("it"))

    # PROTECT the fixed fs->it inhibitory return from the global Hebbian weight-decay (gain 0 = frozen).
    try:
        b.set_plasticity_gate("fixed", 0.0)
    except KeyError:
        pass

    # cold-start probe: does the projected drive fire IT before any learning?
    pre = it_code(b, it, drives[0], scale=drive_scale, window=window)
    n0, _, _ = _lateral_weight_norm(b)
    print(f"  [seed={seed}] cold-start concept0: IT_active={int((pre > 0).sum())}/{n_it}  "
          f"lateral_norm0={n0:.3f}", flush=True)

    # LEARN the lateral (gate "lat" open). Each pass drives the per-concept projected code -> the lateral
    # learns from the NOISY spiking co-firing statistics; homeostasis holds the diagonal; −λM bounds it.
    try:
        b.set_plasticity_gate("lat", 1.0)
    except KeyError:
        pass
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for i in rng.permutation(len(feats)):
            it_code(b, it, drives[i], scale=drive_scale, window=window)
    try:
        b.set_plasticity_gate("lat", 0.0)
    except KeyError:
        pass

    # READ the settled codes (lateral frozen). raw_acc = IT firing [N, n_it]; codes = recombined [N, K].
    raw_acc = np.stack([it_code(b, it, drives[i], scale=drive_scale, window=window) for i in range(len(feats))])
    codes = np.stack([_recombine(raw_acc[i], K, signed) for i in range(len(feats))])
    active = (raw_acc > 0).sum(1)
    lnorm, lmax, lmean = _lateral_weight_norm(b)
    guards = {
        "mean_active_IT": float(active.mean()), "min_active_IT": int(active.min()),
        "n_silent_concepts": int((active == 0).sum()), "total_IT_spikes": int(raw_acc.sum()),
        "lateral_norm": lnorm, "lateral_max": lmax, "lateral_mean": lmean,
        "code_coh_mean": coherence(codes)[0], "code_coh_max": coherence(codes)[1],
        "drive_coh_mean": coherence(drives)[0], "drive_coh_max": coherence(drives)[1],
    }
    if return_baseline:
        # NO-lateral baseline: a SECOND fresh bridge, lateral disabled the whole time (gate lat=0, no
        # learning). Isolates how much decorrelation is the LEARNED lateral vs the spiking nonlinearity.
        b2 = build_whitening_bridge(seed, n_it, n_fs, lam, lat_density, lat_weight, fs_inh_weight,
                                    hebbian_lr, hebbian_max)
        it2 = np.asarray(b2.region_manager.indices("it"))
        for g in ("fixed", "lat"):
            try:
                b2.set_plasticity_gate(g, 0.0)
            except KeyError:
                pass
        raw2 = np.stack([it_code(b2, it2, drives[i], scale=drive_scale, window=window)
                         for i in range(len(feats))])
        codes_nolat = np.stack([_recombine(raw2[i], K, signed) for i in range(len(feats))])
        guards["nolat_code_coh_mean"] = coherence(codes_nolat)[0]
        return codes, guards, codes_nolat
    return codes, guards


def compose(label, codes, seeds, tokens, nouns, verbs, adjs):
    """Project codes -> phases -> NestedCompositionAgent -> the full capability benchmark (% correct)."""
    d = codes.shape[1]
    seed_res = [run_seed(s, codes, d, tokens, nouns, verbs, adjs, decorrelate=False) for s in seeds]
    _, gok, gtot = aggregate(seed_res)
    return gok, gtot


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--K", type=int, default=300, help="IT-pool / subspace dimension (<= N=320)")
    ap.add_argument("--n-fs", type=int, default=150, help="FS interneurons mediating the lateral")
    ap.add_argument("--lam", type=float, default=0.0002, help="hebbian_weight_decay = the −λM (KEY: tune for stable+bounded)")
    ap.add_argument("--epochs", type=int, default=12, help="passes over the 320 codes to learn the lateral")
    ap.add_argument("--drive-scale", type=float, default=2500.0, help="pA scale for the projected IT drive")
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--lat-density", type=float, default=0.6)
    ap.add_argument("--lat-weight", type=float, default=0.1, help="lateral it->fs START weight (grows via co-fire)")
    ap.add_argument("--fs-inh-weight", type=float, default=0.8, help="fixed fs->it inhibitory return weight")
    ap.add_argument("--hebbian-lr", type=float, default=0.05)
    ap.add_argument("--hebbian-max", type=float, default=4.0, help="lateral weight cap (bounds ONLY the lateral)")
    ap.add_argument("--baseline", action="store_true",
                    help="also read a NO-lateral baseline to attribute decorrelation to the learned lateral")
    ap.add_argument("--signed", action="store_true",
                    help="ON/OFF 2K IT pool preserving the projected code's SIGN (vs the lossy rectified drive)")
    ap.add_argument("--bench-seeds", type=int, nargs="+", default=None,
                    help="seeds for the composition agent (default: same as --seeds)")
    ap.add_argument("--out", default="research/findings/raw/_onbridge_spiking_whitening_compose.json")
    args = ap.parse_args()
    bench_seeds = args.bench_seeds if args.bench_seeds is not None else args.seeds

    nouns, verbs, adjs = build_vocab()
    W, _ = _v1_matrix()
    feats, dim, tokens, src = build_realobject_features(nouns, verbs, adjs, W, seed=args.seeds[0])
    rawm, rawx = coherence(feats)
    print(f"=== ON-BRIDGE SPIKING whitening COMPOSITION gate | grounding={src} | {len(tokens)} concepts "
          f"| K={args.K} | lam(−λM)={args.lam} | drive={'ON/OFF signed' if args.signed else 'rectified'} ===",
          flush=True)
    print(f"  raw feature coherence: mean {rawm:.3f}, max {rawx:.3f}", flush=True)

    out = {"source": src, "K": args.K, "lam": args.lam, "epochs": args.epochs, "seeds": args.seeds,
           "signed": bool(args.signed), "bench_seeds": bench_seeds, "raw_coherence": [rawm, rawx],
           "params": {k: getattr(args, k) for k in ("n_fs", "drive_scale", "window", "lat_density",
                      "lat_weight", "fs_inh_weight", "hebbian_lr", "hebbian_max")}}

    # ---- CONTROLS (bracket every result; if these are off, the harness is broken) ----
    raw_ok, raw_tot = compose("RAW", feats, bench_seeds, tokens, nouns, verbs, adjs)
    out["RAW"] = [raw_ok, raw_tot]
    print(f"  {'RAW grounded (floor control)':<42} {raw_ok}/{raw_tot} = {raw_ok/raw_tot*100:.1f}%   "
          f"(expect ~66.7%)", flush=True)
    cw_ok, cw_tot = compose("CONCEPT", _decorrelate(feats), bench_seeds, tokens, nouns, verbs, adjs)
    out["CONCEPT-whiten"] = [cw_ok, cw_tot]
    print(f"  {'CONCEPT-whiten (100% target control)':<42} {cw_ok}/{cw_tot} = {cw_ok/cw_tot*100:.1f}%   "
          f"(expect ~100%)", flush=True)
    harness_ok = (raw_ok / raw_tot < 0.85) and (cw_ok / cw_tot > 0.9)
    print(f"  harness sanity: {'OK (controls bracket as expected)' if harness_ok else '⚠ BROKEN — distrust below'}",
          flush=True)

    # ---- ON-BRIDGE SPIKING learned whitening, per seed (heavy; run sequentially) ----
    out["SPIKING"] = {}
    for s in args.seeds:
        res = spiking_whiten_codes(
            feats, s, args.K, args.n_fs, args.lam, args.epochs, args.drive_scale, args.window,
            args.lat_density, args.lat_weight, args.fs_inh_weight, proj_seed=s,
            hebbian_lr=args.hebbian_lr, hebbian_max=args.hebbian_max, return_baseline=args.baseline,
            signed=args.signed)
        codes, guards = res[0], res[1]
        # GUARD: silent / blown-up IT pool is a FALSE POSITIVE regardless of composition.
        degenerate = (guards["n_silent_concepts"] > 0 or guards["total_IT_spikes"] == 0
                      or guards["mean_active_IT"] < 2.0 or guards["lateral_max"] >= args.hebbian_max * 0.999)
        sp_ok, sp_tot = compose("SPIKING", codes, bench_seeds, tokens, nouns, verbs, adjs)
        entry = {"compose": [sp_ok, sp_tot], "guards": guards, "degenerate": bool(degenerate)}
        nolat_str = ""
        if args.baseline:
            codes_nolat = res[2]
            nolat_ok, nolat_tot = compose("NOLAT", codes_nolat, bench_seeds, tokens, nouns, verbs, adjs)
            entry["nolat_compose"] = [nolat_ok, nolat_tot]
            nolat_str = (f"  [no-lateral baseline: {nolat_ok}/{nolat_tot}={nolat_ok/nolat_tot*100:.1f}%, "
                         f"coh {guards['nolat_code_coh_mean']:.3f}]")
        out["SPIKING"][str(s)] = entry
        flag = "  ⚠ DEGENERATE IT POOL (false-positive risk)" if degenerate else ""
        print(f"  [seed={s}] SPIKING learned-whiten: {sp_ok}/{sp_tot} = {sp_ok/sp_tot*100:.1f}%   "
              f"| IT active={guards['mean_active_IT']:.1f}/{args.K} (min {guards['min_active_IT']}, "
              f"silent {guards['n_silent_concepts']}/{len(feats)}) | lateral_norm={guards['lateral_norm']:.2f} "
              f"(max {guards['lateral_max']:.2f}) | drive_coh={guards['drive_coh_mean']:.3f}->"
              f"code_coh={guards['code_coh_mean']:.3f}{flag}{nolat_str}", flush=True)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {args.out}", flush=True)

    # ---- VERDICT ----
    sp_rates = [v["compose"][0] / v["compose"][1] for v in out["SPIKING"].values()]
    sp_degen = any(v["degenerate"] for v in out["SPIKING"].values())
    raw_rate = raw_ok / raw_tot
    cw_rate = cw_ok / cw_tot
    print("\n" + "=" * 78, flush=True)
    print(f"VERDICT (on-bridge SPIKING learned whitening, {len(args.seeds)} seed(s)):", flush=True)
    print(f"  RAW floor {raw_rate*100:.1f}%  |  CONCEPT-whiten target {cw_rate*100:.1f}%  |  "
          f"rate-model 100%", flush=True)
    print(f"  SPIKING per-seed: {['%.1f%%' % (r*100) for r in sp_rates]}  "
          f"(mean {np.mean(sp_rates)*100:.1f}%)", flush=True)
    if not harness_ok:
        verdict = "INVALID — controls did not bracket (harness broken)"
    elif sp_degen:
        verdict = "FALSE-POSITIVE RISK — a seed had a degenerate IT pool (silent/collapsed); see guards"
    elif np.mean(sp_rates) >= 0.95 and min(sp_rates) >= 0.90:
        verdict = "GO — on-bridge spiking learned whitening COMPOSES (>=95% mean, >=90% min), == target"
    elif np.mean(sp_rates) > raw_rate + 0.1:
        verdict = f"PARTIAL — composes ABOVE raw floor but below target (mean {np.mean(sp_rates)*100:.1f}%)"
    else:
        verdict = f"BOUNDARY — does NOT compose above raw floor (mean {np.mean(sp_rates)*100:.1f}% vs raw {raw_rate*100:.1f}%)"
    print(f"  => {verdict}", flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
