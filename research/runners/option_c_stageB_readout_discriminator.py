"""OPTION C STAGE-B follow-up -- the READ-OUT vs MECHANISM DISCRIMINATOR (the decisive localizer).

WHY THIS RUNNER EXISTS
======================
Stage B (option_c_stageB_fair_test.py) asked: does the project's brain-based spiking-Hebbian learn +
the validated divnorm read-out LEARN graded paradigmatic semantics (animals cluster, colors cluster,
...) from REAL TinyStories co-occurrence, measured against an INDEPENDENT a-priori taxonomy? Seed 42
came back a COMPLETE NULL: Pearson(S_learned, S_true) = -0.008 (generalization at chance) WHILE the
host distributional-semantics pre-gate proved the structure IS in the data (Pearson +0.532).

A complete null at the FINAL read-out has TWO possible causes, and they have OPPOSITE implications:
  (1) the spiking-Hebbian LEARN never captured the structure (the learned recurrent W has no
      category-structured target->hub connectivity)  -> a MECHANISM WALL: the point-neuron substrate
      cannot learn the paradigmatic structure from experience (the documented Mikulasch-Priesemann
      point-neuron decorrelation limit). Option C needs the dendritic substrate.
  (2) the LEARN captured it (W's target->hub profiles ARE category-structured) but the brain-based
      divnorm READ-OUT fails to surface it into the final codes on real-text W  -> a FIXABLE READ-OUT:
      Option C is recoverable with a better (still brain-based) read-out.

This runner LOCALIZES the failure with a THREE-LEVEL decomposition of the paradigmatic signal, then a
read-out-variant sweep. It is a DIAGNOSTIC (a labelled disambiguator, like the host pre-gate) -- NOT a
brain-based deliverable. The host PPMI+SVD lens is used here ONLY as a measurement instrument applied
identically to the raw counts AND the learned W, so the comparison is apples-to-apples.

THE THREE LEVELS (each = cosine of every target's HUB-connectivity profile -> Pearson vs INDEPENDENT
S_true). The hub-connectivity profile is the genuine SECOND-ORDER measure: two targets are paradigmatic
neighbours (cat~dog) iff they connect to the SAME context hubs.
  L1  raw co-occurrence counts C[targets, hubs]    -- the CEILING ("is the structure in what the learn
                                                      saw?"). Should be clearly positive (the data
                                                      carries it; ~ the host's +0.53).
  L2  the LEARNED recurrent      W[targets, hubs]    -- THE DECISIVE NUMBER ("did the spiking-Hebbian
                                                      learn PRESERVE it?"). high -> read-out problem;
                                                      ~0 -> the learn destroyed it (mechanism wall).
  L3  the divnorm READ-OUT codes                     -- the known null (~0). Plus a read-out-variant
                                                      sweep on the SAME W (steps / alpha / sigma / exp):
                                                      if any brain-based variant recovers the signal
                                                      that L2 holds -> FIXABLE.
Each level is reported under TWO lenses: a plain cosine of the hub-profiles (most interpretable) AND
the host PPMI+SVD lens (the same instrument that gave the +0.53 data ceiling; L2-under-PPMI is exactly
the documented "host-method-on-W stand-in", which on the SYNTHETIC decorrelated toy was +0.84).

VERDICTS (single seed is sufficient for a localizing diagnostic; default seed 42 = the Stage-B null):
  MECHANISM_WALL    -- data carries it (L1 high) but the LEARN loses it (L2 ~0). The spiking-Hebbian
                       point-neuron learn cannot preserve the paradigmatic (common-mode-bearing)
                       structure. Confirms the Stage-B null is a CLEAN mechanism negative; Option C
                       needs the dendritic substrate. (The expected outcome per the point-neuron limit.)
  FIXABLE_READOUT   -- data carries it (L1 high), the LEARN keeps it (L2 high), the default read-out
                       loses it (L3 ~0) BUT a brain-based read-out VARIANT recovers it (sweep high).
                       Option C is recoverable with a better read-out -> re-run Stage B with the variant.
  READOUT_WALL      -- L1 high, L2 high, L3 ~0, and NO brain-based read-out variant recovers it (only
                       the host-on-W stand-in does). A subtler boundary: the structure is in W but no
                       spreading read-out extracts it on real-text W.
  CORPUS_LOST_IT    -- even L1 (raw counts, the ceiling) is ~0 under both lenses -> the 500-hub-restricted
                       count matrix doesn't carry the signal the full-context host saw (a corpus/extraction
                       issue, not a mechanism statement). Informative: widen the hub set / revisit.

NO sim/ edits; reuse-by-import only. The learn is the SAME spiking-Hebbian learn_W_homeostatic on a real
Izhikevich bridge (GPU). The interrogation + sweep are cheap numpy on the learned W.

Run (GPU; ONE learn ~20-40 min on the full 564-concept context-inclusive TinyStories corpus, then the
interrogation + sweep are seconds):
  SIM_BACKEND=cupy python -u -m research.runners.option_c_stageB_readout_discriminator \
      --seed 42 --out research/findings/raw/_option_c_stageB_discriminator_seed42.json

Run (CPU-numpy PLUMBING smoke -- tiny taxonomy + first ~300KB, validates the decomposition + verdict
plumbing end-to-end in ~1-2 min):
  SIM_BACKEND=numpy python -u -m research.runners.option_c_stageB_readout_discriminator \
      --smoke --seed 42 --out research/findings/raw/_option_c_stageB_discriminator_smoke.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import itertools

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# ----- reuse VERBATIM by import (NO reimplementation; NO sim/ edits) -----
# the context-inclusive corpus builder (Stage B's one new corpus function):
from research.runners.option_c_stageB_fair_test import build_context_inclusive_cooccurrence  # noqa: E402
# the brain-based spiking-Hebbian learn (Oja homeostatic ceiling) -> W [Nc, Nc]:
from research.runners.learned_graded_embedding_homeostasis_probe import learn_W_homeostatic  # noqa: E402
# the validated brain-based divnorm spreading read-out:
from research.runners.learned_graded_embedding_divnorm_readout_probe import (  # noqa: E402
    divnorm_spreading_readout,
)
# structure recovery (Pearson vs the independent taxonomy) for the L3 read-out codes:
from research.runners.learned_graded_embedding_derisk_probe import structure_recovery  # noqa: E402
# the host PPMI+SVD lens + Pearson scoring (the measurement instrument applied to BOTH C and W):
from research.runners.option_c_paradigmatic_host_precheck import ppmi_svd_sim, score  # noqa: E402
# the raw concept-concept count matrix + learning-not-passthrough check:
from research.runners.learned_graded_embedding_diagnose import raw_count_matrix, offdiag_pearson  # noqa: E402
# the taxonomy + flattener:
from research.runners.option_c_real_cooccurrence_derisk import (  # noqa: E402
    TAXONOMY_8x8,
    TAXONOMY_SMOKE,
    taxonomy_to_vocab_categories,
)


# ===========================================================================
# The two measurement lenses applied to a [Nt x Ncols] hub-connectivity profile matrix P.
# Both return (pearson_vs_Strue, within_minus_between_margin, nn_same_category_rate).
# ===========================================================================
def _cosine_lens(P: np.ndarray, labels: np.ndarray) -> tuple:
    """Plain cosine of the hub-profile rows (the most interpretable second-order measure)."""
    norm = np.linalg.norm(P, axis=1, keepdims=True)
    Pn = P / (norm + 1e-12)
    sim = Pn @ Pn.T
    pearson, margin, nn_same, _ = score(sim, labels)
    return pearson, margin, nn_same


def _ppmi_svd_lens(P: np.ndarray, labels: np.ndarray, svd_dim: int, alpha: float) -> tuple:
    """The host PPMI+SVD lens (the SAME instrument that gave the +0.53 data ceiling), applied to a
    non-negative hub-profile matrix. On the LEARNED W's hub block this IS the documented host-method-
    on-W stand-in. PPMI needs non-negative input; the learned recurrent is excitatory (>=0)."""
    P = np.maximum(P, 0.0)
    sim = ppmi_svd_sim(P, svd_dim=min(svd_dim, max(2, min(P.shape) - 1)), alpha=alpha)
    pearson, margin, nn_same, _ = score(sim, labels)
    return pearson, margin, nn_same


def _level(name: str, P: np.ndarray, labels: np.ndarray, svd_dim: int, alpha: float) -> dict:
    cos_p, cos_m, cos_nn = _cosine_lens(P, labels)
    ppmi_p, ppmi_m, ppmi_nn = _ppmi_svd_lens(P, labels, svd_dim, alpha)
    print(f"    [{name}] cosine-lens   Pearson={cos_p:+.3f} margin={cos_m:+.3f} nn-same={cos_nn:.3f}",
          flush=True)
    print(f"    [{name}] PPMI+SVD-lens Pearson={ppmi_p:+.3f} margin={ppmi_m:+.3f} nn-same={ppmi_nn:.3f}",
          flush=True)
    return {"cosine_pearson": cos_p, "cosine_margin": cos_m, "cosine_nn_same": cos_nn,
            "ppmisvd_pearson": ppmi_p, "ppmisvd_margin": ppmi_m, "ppmisvd_nn_same": ppmi_nn}


# ===========================================================================
# The read-out-variant sweep on the SAME learned W (all brain-based divnorm variants; cheap numpy).
# If any variant recovers the signal that L2 (W) holds but the default read-out lost -> FIXABLE.
# ===========================================================================
def _readout_sweep(W, member_rows, S_true, second_order_pairs, labels, seed, args) -> dict:
    best = {"pearson": -2.0, "variant": None, "margin": None}
    rows = []
    # the DEFAULT validated operating point is in the grid (steps=2, alpha=0.5, sigma=0.001, exp=2.0,
    # ch/interleave); the sweep widens diffusion depth (to capture deeper 2-hop target->hub->target
    # propagation) and the divnorm sharpening knobs.
    for steps, alpha, sigma, exponent, order in itertools.product(
            args.sweep_steps, args.sweep_alpha, args.sweep_sigma, args.sweep_exponent, args.sweep_order):
        codes = divnorm_spreading_readout(
            W, member_rows, divnorm=args.readout_divnorm, order=order,
            sigma=sigma, exponent=exponent, alpha=alpha, steps=steps,
            log_clip=args.readout_log_clip)
        rec = structure_recovery(codes, S_true, second_order_pairs, seed)
        p = rec["pearson_learned_vs_Strue"]
        rows.append({"steps": steps, "alpha": alpha, "sigma": sigma, "exponent": exponent,
                     "order": order, "pearson": p, "second_order_margin": rec["second_order_margin"]})
        if p > best["pearson"]:
            best = {"pearson": p, "variant": {"steps": steps, "alpha": alpha, "sigma": sigma,
                                              "exponent": exponent, "order": order},
                    "margin": rec["second_order_margin"]}
    rows.sort(key=lambda r: r["pearson"], reverse=True)
    print(f"    read-out sweep ({len(rows)} variants): BEST Pearson={best['pearson']:+.3f} "
          f"(margin {best['margin']:+.3f}) at {best['variant']}", flush=True)
    for r in rows[:5]:
        print(f"      steps={r['steps']} alpha={r['alpha']} sigma={r['sigma']} exp={r['exponent']} "
              f"{r['order']:10s} -> Pearson={r['pearson']:+.3f} 2nd-margin={r['second_order_margin']:+.3f}",
              flush=True)
    return {"best": best, "top5": rows[:5], "n_variants": len(rows)}


def decide_verdict(L1, L2, L3_default, sweep_best, args) -> tuple:
    """Localize the Stage-B null. Uses the PPMI+SVD lens as the primary instrument (the same one that
    proved the +0.53 data ceiling), with the plain-cosine lens reported alongside for robustness."""
    data_carries = (L1["ppmisvd_pearson"] >= args.data_bar) or (L1["cosine_pearson"] >= args.data_bar)
    learn_keeps = (L2["ppmisvd_pearson"] >= args.keep_bar) or (L2["cosine_pearson"] >= args.keep_bar)
    learn_loses = (L2["ppmisvd_pearson"] < args.lose_bar) and (L2["cosine_pearson"] < args.lose_bar)
    readout_recovers = sweep_best["pearson"] >= args.recover_bar

    if not data_carries:
        verdict = "CORPUS_LOST_IT"
        why = (f"even L1 (raw counts ceiling) is below the data bar {args.data_bar} under both lenses "
               f"(PPMI {L1['ppmisvd_pearson']:+.3f}, cosine {L1['cosine_pearson']:+.3f}); the 500-hub "
               f"restriction lost the signal the full-context host saw -- a corpus/extraction issue, "
               f"NOT a mechanism statement.")
    elif learn_loses:
        verdict = "MECHANISM_WALL"
        why = (f"data carries it (L1 PPMI {L1['ppmisvd_pearson']:+.3f}) but the spiking-Hebbian LEARN "
               f"loses it (L2 PPMI {L2['ppmisvd_pearson']:+.3f}, cosine {L2['cosine_pearson']:+.3f} "
               f"< lose-bar {args.lose_bar}). The point-neuron learn cannot preserve the paradigmatic "
               f"(common-mode-bearing) structure -> the Stage-B null is a CLEAN mechanism negative; "
               f"Option C needs the dendritic substrate (the Mikulasch-Priesemann point-neuron limit).")
    elif learn_keeps and readout_recovers:
        verdict = "FIXABLE_READOUT"
        why = (f"data carries it (L1 {L1['ppmisvd_pearson']:+.3f}), the LEARN keeps it "
               f"(L2 {L2['ppmisvd_pearson']:+.3f}), the default read-out loses it "
               f"(L3 {L3_default:+.3f}) BUT a brain-based read-out variant recovers it "
               f"(sweep best {sweep_best['pearson']:+.3f} at {sweep_best['variant']}) -> Option C is "
               f"recoverable; re-run Stage B with the variant.")
    elif learn_keeps and not readout_recovers:
        verdict = "READOUT_WALL"
        why = (f"data carries it (L1 {L1['ppmisvd_pearson']:+.3f}), the LEARN keeps it under the host "
               f"lens (L2 PPMI {L2['ppmisvd_pearson']:+.3f}) but NO brain-based read-out variant "
               f"surfaces it (sweep best {sweep_best['pearson']:+.3f} < recover-bar {args.recover_bar}; "
               f"L3 default {L3_default:+.3f}). Only the host-on-W stand-in extracts it -> a subtler "
               f"boundary (structure in W, no spreading read-out gets it out on real-text W).")
    else:
        verdict = "AMBIGUOUS_partial_learn"
        why = (f"L2 sits between the keep/lose bars (PPMI {L2['ppmisvd_pearson']:+.3f}, "
               f"cosine {L2['cosine_pearson']:+.3f}); the learn PARTIALLY preserved the structure. "
               f"Characterize: closer to wall (raise hubs/cycles) vs closer to fixable (sweep recover "
               f"{sweep_best['pearson']:+.3f}).")
    return verdict, why


def main():
    p = argparse.ArgumentParser(description="Option C Stage-B read-out vs mechanism discriminator "
                                            "(three-level localizer + read-out sweep)")
    p.add_argument("--seed", type=int, default=42, help="the Stage-B null seed to localize (default 42)")
    p.add_argument("--smoke", action="store_true",
                   help="CPU-numpy plumbing smoke: tiny 4-category taxonomy, first ~300KB, small pool/hubs")
    p.add_argument("--corpus", default=None)
    # context-inclusive corpus (mirror the Stage-B decisive operating point)
    p.add_argument("--window", type=int, default=2)
    p.add_argument("--n-context-hubs", type=int, default=500)
    p.add_argument("--repeat-cap", type=int, default=40)
    p.add_argument("--max-bytes", type=int, default=None)
    p.add_argument("--freq-floor", type=int, default=30)
    p.add_argument("--min-facts-per-category", type=int, default=20)
    # brain-based learn (match the validated homeostasis defaults = Stage B)
    p.add_argument("--n-pool", type=int, default=2000)
    p.add_argument("--pattern-size", type=int, default=100)
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--homeo", default="oja")
    p.add_argument("--homeo-target", type=float, default=2.0)
    # the host lens operating point (the SAME instrument as the Stage-A pre-gate / +0.53 ceiling)
    p.add_argument("--lens-svd-dim", type=int, default=100)
    p.add_argument("--lens-alpha", type=float, default=0.75)
    # the DEFAULT validated divnorm read-out (for the L3 baseline + the sweep grid centre)
    p.add_argument("--readout-divnorm", default="ch")
    p.add_argument("--readout-log-clip", action="store_true")
    # the read-out-variant sweep grid (all brain-based; widen diffusion depth + the sharpening knobs)
    p.add_argument("--sweep-steps", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    p.add_argument("--sweep-alpha", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    p.add_argument("--sweep-sigma", type=float, nargs="+", default=[0.001, 0.01, 0.1])
    p.add_argument("--sweep-exponent", type=float, nargs="+", default=[1.0, 2.0])
    p.add_argument("--sweep-order", nargs="+", default=["interleave", "post"])
    # verdict bars
    p.add_argument("--data-bar", type=float, default=0.25,
                   help="L1 ceiling must clear this (under either lens) for 'data carries it'")
    p.add_argument("--keep-bar", type=float, default=0.35,
                   help="L2 must clear this (under either lens) for 'the learn kept it'")
    p.add_argument("--lose-bar", type=float, default=0.15,
                   help="L2 below this (under BOTH lenses) = 'the learn lost it' (mechanism wall)")
    p.add_argument("--recover-bar", type=float, default=0.30,
                   help="a read-out sweep variant clearing this = 'fixable read-out'")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.smoke:
        os.environ.setdefault("SIM_BACKEND", "numpy")
        taxonomy = TAXONOMY_SMOKE
        if args.max_bytes is None:
            args.max_bytes = 300_000
        if args.n_pool == 2000:
            args.n_pool = 400
        if args.pattern_size == 100:
            args.pattern_size = 20
        if args.n_context_hubs == 500:
            args.n_context_hubs = 60
        if args.freq_floor == 30:
            args.freq_floor = 5
        if args.min_facts_per_category == 20:
            args.min_facts_per_category = 5
        # keep the sweep small for the smoke
        args.sweep_steps = [2, 4]
        args.sweep_alpha = [0.5]
        args.sweep_sigma = [0.001, 0.1]
        args.sweep_exponent = [2.0]
        args.sweep_order = ["interleave"]
    else:
        taxonomy = TAXONOMY_8x8

    corpus_path = args.corpus or os.path.join(_REPO, "data", "corpus", "tinystories.txt")
    if not os.path.exists(corpus_path):
        print(f"[ERROR] corpus not found: {corpus_path}", flush=True)
        sys.exit(2)

    backend = os.environ.get("SIM_BACKEND", "auto")
    vocab, cat_ids, cat_names = taxonomy_to_vocab_categories(taxonomy)
    labels = np.asarray(cat_ids, dtype=int)
    Nt = len(vocab)
    t0 = time.time()

    print(f"[option-c STAGE-B read-out/mechanism DISCRIMINATOR] seed={args.seed} backend={backend} "
          f"smoke={args.smoke}", flush=True)
    print(f"  taxonomy: {len(cat_names)}x{len(taxonomy[cat_names[0]])} = {Nt} targets ({cat_names})",
          flush=True)
    print(f"  localizing the Stage-B null: does the LEARN lose the paradigmatic structure (mechanism "
          f"wall) or does the READ-OUT (fixable)?", flush=True)

    # ----- STEP 1: the SAME context-inclusive corpus Stage B used -----
    corpus = build_context_inclusive_cooccurrence(
        corpus_path, vocab, cat_ids, window=args.window, n_context_hubs=args.n_context_hubs,
        repeat_cap=args.repeat_cap, seed=args.seed, max_bytes=args.max_bytes, freq_floor=args.freq_floor,
        min_facts_per_category=args.min_facts_per_category, verbose=True)
    concepts = corpus["concepts"]          # targets FIRST [0..Nt), then hubs [Nt..Nc)
    members = corpus["members"]            # the Nt targets
    S_true = corpus["S_true"]
    second_order_pairs = corpus["second_order_pairs"]
    member_rows = np.arange(Nt, dtype=int)  # targets are the first Nt concepts
    hub_cols = np.arange(Nt, len(concepts), dtype=int)
    assert len(members) == Nt and list(members) == list(vocab)
    assert len(hub_cols) == corpus["_n_context_hubs"]

    # ----- STEP 2: the SAME brain-based spiking-Hebbian learn -> W [Nc, Nc] -----
    print(f"  [STEP 2 -- brain-based spiking-Hebbian learn (learn_W_homeostatic), {len(concepts)} concepts]",
          flush=True)
    tL = time.time()
    W, learn_info = learn_W_homeostatic(
        concepts, corpus["facts"], args.seed, args.n_pool, args.pattern_size, args.cycles,
        gamma=1.0, cap=None, homeo=args.homeo, homeo_target=args.homeo_target, homeo_clip_only=True)
    learn_s = time.time() - tL
    C_full = raw_count_matrix(concepts, corpus["facts"])
    pearson_W_counts = offdiag_pearson(W, C_full)
    print(f"    learned recurrent: mean={learn_info['recurrent_mean']:.3f} max={learn_info['recurrent_max']:.3f} "
          f"({learn_info['n_neurons']} neurons, {learn_s:.1f}s); Pearson(W, raw_counts)={pearson_W_counts:+.3f}",
          flush=True)

    # ----- STEP 3: the THREE-LEVEL decomposition of the paradigmatic signal -----
    # the hub-connectivity PROFILE of each target = its row restricted to the hub columns.
    C_hub = C_full[np.ix_(member_rows, hub_cols)]   # [Nt x n_hubs] raw counts target->hub
    W_hub = W[np.ix_(member_rows, hub_cols)]         # [Nt x n_hubs] learned weight target->hub
    print("  [STEP 3 -- three-level decomposition: cosine of each target's HUB-connectivity profile]",
          flush=True)
    print("    L1 = raw counts ceiling | L2 = learned W (DECISIVE) | L3 = divnorm read-out codes", flush=True)
    L1 = _level("L1 raw-counts  C[targets,hubs]", C_hub, labels, args.lens_svd_dim, args.lens_alpha)
    L2 = _level("L2 LEARNED-W   W[targets,hubs]", W_hub, labels, args.lens_svd_dim, args.lens_alpha)
    # L2b: the FULL-column W profile (incl. target-target), for completeness
    L2_full = _level("L2b full-col  W[targets, :]", W[np.ix_(member_rows, np.arange(len(concepts)))],
                     labels, args.lens_svd_dim, args.lens_alpha)

    # ----- L3: the default validated divnorm read-out (the known Stage-B null) -----
    print("  [STEP 4 -- L3: the validated divnorm read-out codes (the Stage-B null) + a read-out sweep]",
          flush=True)
    default_codes = divnorm_spreading_readout(
        W, member_rows, divnorm=args.readout_divnorm, order="interleave",
        sigma=0.001, exponent=2.0, alpha=0.5, steps=2, log_clip=args.readout_log_clip)
    rec_default = structure_recovery(default_codes, S_true, second_order_pairs, args.seed)
    L3_default = rec_default["pearson_learned_vs_Strue"]
    print(f"    L3 default divnorm read-out: Pearson(S_learned, S_true)={L3_default:+.3f} "
          f"(2nd-order margin {rec_default['second_order_margin']:+.3f}) -- the Stage-B null", flush=True)

    # ----- the read-out-variant sweep on the SAME W -----
    sweep = _readout_sweep(W, member_rows, S_true, second_order_pairs, labels, args.seed, args)

    # ----- VERDICT -----
    verdict, why = decide_verdict(L1, L2, L3_default, sweep["best"], args)
    print(f"\n{'='*84}", flush=True)
    print(f"  DISCRIMINATOR VERDICT: {verdict}", flush=True)
    print(f"  {why}", flush=True)
    print(f"  ladder (PPMI+SVD lens): L1 raw-counts {L1['ppmisvd_pearson']:+.3f}  ->  "
          f"L2 LEARNED-W {L2['ppmisvd_pearson']:+.3f}  ->  L3 read-out {L3_default:+.3f}  "
          f"(sweep best {sweep['best']['pearson']:+.3f})", flush=True)
    print(f"  ladder (cosine lens):   L1 raw-counts {L1['cosine_pearson']:+.3f}  ->  "
          f"L2 LEARNED-W {L2['cosine_pearson']:+.3f}", flush=True)
    print(f"  Total elapsed: {time.time() - t0:.1f}s", flush=True)
    print(f"{'='*84}\n", flush=True)

    out = {
        "verdict": verdict,
        "why": why,
        "seed": args.seed,
        "backend": backend,
        "smoke": bool(args.smoke),
        "taxonomy_size": f"{len(cat_names)}x{len(taxonomy[cat_names[0]])}",
        "taxonomy_categories": cat_names,
        "corpus": os.path.basename(corpus_path),
        "n_targets": Nt,
        "n_context_hubs": corpus["_n_context_hubs"],
        "n_concepts": len(concepts),
        "n_facts": corpus["n_facts"],
        "n_second_order_pairs": len(second_order_pairs),
        "learn_info": learn_info,
        "pearson_W_vs_rawcounts": pearson_W_counts,
        "lens_operating_point": {"svd_dim": args.lens_svd_dim, "alpha": args.lens_alpha},
        "levels": {
            "L1_raw_counts_hub_profile": L1,
            "L2_learned_W_hub_profile": L2,
            "L2b_learned_W_full_column": L2_full,
            "L3_divnorm_readout_default_pearson": L3_default,
            "L3_divnorm_readout_default_second_order_margin": rec_default["second_order_margin"],
        },
        "readout_sweep": sweep,
        "bars": {"data_bar": args.data_bar, "keep_bar": args.keep_bar, "lose_bar": args.lose_bar,
                 "recover_bar": args.recover_bar},
        "config": {"window": args.window, "n_context_hubs": args.n_context_hubs,
                   "repeat_cap": args.repeat_cap, "max_bytes": args.max_bytes,
                   "n_pool": args.n_pool, "pattern_size": args.pattern_size, "cycles": args.cycles,
                   "homeo": args.homeo, "homeo_target": args.homeo_target},
        "note": ("DIAGNOSTIC, not a brain-based deliverable. The host PPMI+SVD lens is a measurement "
                 "instrument applied identically to the raw counts (L1) and the learned W (L2) so the "
                 "comparison is apples-to-apples; L2-under-PPMI is the documented host-method-on-W "
                 "stand-in (+0.84 on the SYNTHETIC decorrelated toy). The decisive number is L2: high "
                 "-> read-out problem (fixable); ~0 -> the learn lost it (the point-neuron mechanism "
                 "wall, consistent with the Stage-B null)."),
        "elapsed_total_s": time.time() - t0,
    }
    if args.out is None:
        raw_dir = os.path.join(_REPO, "research", "findings", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.out = os.path.join(raw_dir, f"_option_c_stageB_discriminator_seed{args.seed}_{ts}.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)
    return out


if __name__ == "__main__":
    main()
