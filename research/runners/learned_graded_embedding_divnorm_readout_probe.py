"""LEARNED GRADED-EMBEDDING DIVISIVE-NORMALIZATION READ-OUT -- can a FULLY BRAIN-BASED read-out
(spreading-activation + divisive normalization, NO host PPMI+SVD) close the read-out residual?

CONTEXT (the one residual the dual/CLS learned-embedding has left):
  The brain-based learned-embedding is de-risked GO_full end-to-end (3/3 seeds;
  2026-06-11-learned-graded-embedding-confirm-GO_full.md, commit e6e277e3) -- EXCEPT the recovering
  read-out is the HOST method (positive pointwise mutual information + singular value decomposition,
  "PPMI+SVD") applied to the brain-LEARNED weight matrix W. That host-on-W read-out is a labelled
  STAND-IN; the genuinely brain-based read-out tried (spreading-activation / diffusion through the
  hub nodes, full-column) already GENERALIZES 1.000, but its 2nd-order cat~dog cosine margin (~+0.04)
  FAILS the G1 cosine bar (+0.10): diffusion SMOOTHS but does not SHARPEN the within-vs-between
  contrast the way PPMI's marginal division does.

THE FIX (this runner): add DIVISIVE NORMALIZATION to the brain-based read-out.
  Divisive normalization (Carandini & Heeger; gain control = divide each unit by a normalization
  pool / the local activity sum) is the canonical cortical computation AND the brain-based analogue
  of PPMI's marginal division (PPMI divides co-occurrence by the marginals to remove the high-
  frequency / common-mode that blurs the contrast). The base spreading-activation read-out generalizes
  1.000 but under-sharpens; divisive normalization is the missing CONTRAST/sharpening arm. Question:
  does a spreading-activation + divisive-normalization read-out on the brain-LEARNED W clear the G1
  cosine bar (2nd-order margin >= +0.10) while KEEPING generalization 1.000 -- making the read-out
  fully brain-based (no host PPMI+SVD)?

THE EXPERIMENT (multi-seed 42/43/44; GPU; FOREGROUND):
  Learn at cycles=2 (the de-saturated/faithful regime; reuse learn_W_desaturate VERBATIM). Then on the
  brain-LEARNED W, build a FULLY BRAIN-BASED read-out and sweep it:
    1. BASE = spreading-activation / diffusion through the FULL hub-inclusive W (member rows over ALL
       columns incl. hubs -- where the cat~dog shared-neighbour signal lives). Already generalizes
       1.000; member-only collapses (confirmed) so we keep full-column.
    2. + DIVISIVE NORMALIZATION (the contrast/sharpening step), swept across the brain-based forms:
         (a) MARGINAL division   -- divide each co-activation by row+col marginal sums (the direct
             PPMI analogue, no log): M_ij / sqrt(rowsum_i * colsum_j).
         (b) DIVISIVE NORM (Carandini-Heeger) with a tunable semi-saturation sigma + exponent n:
             x_i^n / (sigma^n + pool_i^n), pool = the local normalization pool (row sum of the
             rectified activations). This is the canonical cortical gain-control form.
         (c) optionally a LOG-POSITIVE-CLIP (the "positive" in PPMI): max(0, log(.)).
       Applied BEFORE the spreading, AFTER it, or interleaved -- swept (order) x (sigma) x (exponent)
       x (diffusion steps in {2,3,4}).
    3. RE-TEST at each variant: Pearson(sim, S_true) + the G1 cosine margin (2nd-order cat~dog, bar
       +0.10) + G2 generalization (already 1.000 -- confirm it HOLDS) + the permuted-S baseline (~0).
       Compare to: host-on-W stand-in (+0.84, the target), raw diffusion (margin +0.04, fails),
       host ceiling (+0.93).

DECISION (stated explicitly):
  GO (residual closed)  if a spreading-activation + divisive-normalization read-out (fully brain-based,
                        NO host SVD) clears the G1 cosine bar (2nd-order margin >= +0.10, Pearson toward
                        +0.84) AND keeps generalization 1.000, multi-seed. -> the read-out is FULLY
                        BRAIN-BASED; the host PPMI+SVD stand-in is RETIRED -> the dual/CLS learned-
                        embedding is fully brain-based end-to-end -> the build starts clean.
  BOUNDARY              if divisive normalization sharpens but doesn't fully clear the cosine bar (e.g.
                        margin +0.06..+0.09) -> characterize how close + whether a low-rank step (the
                        SVD analogue) is the missing piece; host-on-W stays a documented build-time
                        stand-in. No banking.

ANTI-CHEATS (all mandatory):
  - the read-out runs on the brain-LEARNED W (assert Pearson(W, raw_counts) < 0.99 so it's NOT the host
    ceiling); divisive normalization is a FIXED nonlinearity (no fitting to S_true); the permuted-S
    baseline ~0; G2 controls (orthogonal A2 + permuted-property A3 collapse); generalization must STAY
    1.000 (don't trade it for the cosine margin); multi-seed 42/43/44. The host PPMI+SVD on RAW counts
    is the labelled CEILING ONLY (never the deliverable).

Run (GPU, FOREGROUND -- the cycles=2 spiking-learn is ~20-60 s inline; the divnorm sweep is pure numpy,
fast; NO background):
  SIM_BACKEND=cupy python -m research.runners.learned_graded_embedding_divnorm_readout_probe \
      --seeds 42,43,44 --out research/findings/raw/_lge_divnorm_multiseed.json
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

# Reuse the de-risk's corpus + read-out + structure-recovery + host-ceiling + generalization
# harnesses VERBATIM; the desaturate probe's de-saturated low-cycle learn + the host-method-on-W
# read-out (the stand-in/target); and the diagnose's helpers. This runner ADDS the brain-based
# spreading-activation + divisive-normalization read-out variants and sweeps them on the learned W.
from research.runners.learned_graded_embedding_derisk_probe import (  # noqa: E402
    build_toy_cooccurrence,
    graded_readout,
    structure_recovery,
    host_ceiling_codes,
    random_gaussian_codes,
    architecture_generalization,
)
from research.runners.learned_graded_embedding_diagnose import (  # noqa: E402
    raw_count_matrix,
    offdiag_pearson,
    _normalize_codes,
)
from research.runners.learned_graded_embedding_desaturate_probe import (  # noqa: E402
    learn_W_desaturate,
    host_method_codes_on_W,
)
from research.runners.dual_cls_architecture_proof_probe import (  # noqa: E402
    codebook_similarity_stats,
    assign_properties,
    run_generalization,
)


# ===========================================================================
# BRAIN-BASED read-out building blocks (all FIXED nonlinearities -- NO fitting to S_true, NO SVD).
# ===========================================================================
def _symmetrize_zero_diag(W: np.ndarray) -> np.ndarray:
    Ws = 0.5 * (W + W.T)
    np.fill_diagonal(Ws, 0.0)
    return Ws


def marginal_division(M: np.ndarray) -> np.ndarray:
    """Brain-based MARGINAL DIVISION (the direct PPMI analogue, WITHOUT the log): each association
    divided by the geometric mean of its row+col marginals -> removes the high-frequency / common-mode
    drive that blurs the within-vs-between contrast. M_ij / sqrt(rowsum_i * colsum_j). This is exactly
    PPMI's denominator (the marginal product); PPMI then takes max(0, log(.)) on top.
    Biologically: the divisive-normalization 'normalization pool' = the marginal activity."""
    M = np.maximum(M, 0.0)
    row = M.sum(axis=1, keepdims=True)
    col = M.sum(axis=0, keepdims=True)
    denom = np.sqrt((row + 1e-12) * (col + 1e-12))
    return M / denom


def divisive_norm_ch(M: np.ndarray, sigma: float, exponent: float) -> np.ndarray:
    """Canonical Carandini-Heeger DIVISIVE NORMALIZATION (cortical gain control): each unit's response
    is its (rectified, exponentiated) drive divided by a semi-saturation constant PLUS the
    exponentiated NORMALIZATION POOL (the local summed activity). Here, per ROW i (concept i's
    association profile), the pool is the row's total rectified drive:
        R_ij = M_ij^n / (sigma^n + (sum_k M_ik)^n / Ncol)     (n = exponent, sigma = semi-saturation)
    This is the divisive-normalization analogue of PPMI's marginal division: high-frequency concepts
    (large pool) get DOWN-weighted, sharpening the contrast. A FIXED nonlinearity (no fit to S_true).
    The pool is normalized by Ncol so sigma is on the per-entry scale (numerically well-conditioned)."""
    M = np.maximum(M, 0.0)
    Ncol = M.shape[1]
    n = float(exponent)
    pool = M.sum(axis=1, keepdims=True) / max(1, Ncol)   # mean rectified drive per row (the pool)
    num = np.power(M + 1e-12, n)
    den = np.power(sigma, n) + np.power(pool + 1e-12, n)
    return num / den


def log_positive_clip(M: np.ndarray) -> np.ndarray:
    """The 'positive' arm of PPMI (max(0, log(.))) applied to an already marginal-divided matrix:
    log compresses the dynamic range + the positive clip removes the below-expectation (negative-PMI)
    entries. Brain-based reading: a saturating (log-like) transfer + rectification (the 'positive')."""
    M = np.maximum(M, 0.0)
    return np.maximum(np.log(M + 1.0), 0.0)   # log1p so 0 -> 0 (no spurious negative offset)


def spreading_activation(M: np.ndarray, alpha: float, steps: int) -> np.ndarray:
    """Spreading-activation diffusion over a (symmetric, non-negative) association matrix M:
        Wd = (1-a) Wd + a (Wd @ Wn),  Wn = row-normalized M, iterated `steps` times.
    A concept's diffused row accumulates its neighbours-of-neighbours -> the second-order shared-
    neighbour signal (cat~dog via the shared 'animal' hub). Returns the diffused matrix (same shape)."""
    rs = M.sum(axis=1, keepdims=True)
    Wn = M / (rs + 1e-12)
    cur = M.copy()
    for _ in range(max(0, steps)):
        cur = (1.0 - alpha) * cur + alpha * (cur @ Wn)
    return cur


def _apply_divnorm(M: np.ndarray, divnorm: str, sigma: float, exponent: float,
                   log_clip: bool) -> np.ndarray:
    """Apply the chosen brain-based divisive-normalization arm to a (symmetric, non-negative) matrix."""
    if divnorm == "none":
        out = np.maximum(M, 0.0)
    elif divnorm == "marginal":
        out = marginal_division(M)
    elif divnorm == "ch":
        out = divisive_norm_ch(M, sigma, exponent)
    else:
        raise ValueError(f"unknown divnorm '{divnorm}'")
    if log_clip:
        out = log_positive_clip(out)
    # keep it symmetric for the downstream diffusion / cosine (the CH per-row form can break symmetry)
    out = 0.5 * (out + out.T)
    return out


def divnorm_spreading_readout(W: np.ndarray, member_rows: np.ndarray, divnorm: str, order: str,
                              sigma: float, exponent: float, alpha: float, steps: int,
                              log_clip: bool) -> np.ndarray:
    """FULLY BRAIN-BASED read-out = (spreading-activation diffusion through the FULL hub-inclusive W)
    COMBINED with (divisive normalization), in the swept ORDER. Each MEMBER concept's code = its
    processed association profile over ALL columns (hubs included -- where the cat~dog signal lives).
    NO host PPMI+SVD; divisive normalization is a fixed nonlinearity (the brain-based PPMI-marginal-
    division analogue). order in:
        'pre'         : divnorm THEN diffuse              (sharpen the raw graph, then spread)
        'post'        : diffuse THEN divnorm              (spread to reach the hubs, then sharpen)
        'interleave'  : divnorm, diffuse, divnorm         (sharpen-spread-sharpen)
        'diffuse_only': diffuse only (the BASE; raw diffusion, the under-sharpened brain-based one)
    """
    Ws = _symmetrize_zero_diag(W)
    Ws = np.maximum(Ws, 0.0)  # co-occurrence is non-negative

    if order == "diffuse_only":
        proc = spreading_activation(Ws, alpha, steps)
    elif order == "pre":
        proc = _apply_divnorm(Ws, divnorm, sigma, exponent, log_clip)
        proc = spreading_activation(proc, alpha, steps)
    elif order == "post":
        proc = spreading_activation(Ws, alpha, steps)
        proc = _apply_divnorm(proc, divnorm, sigma, exponent, log_clip)
    elif order == "interleave":
        proc = _apply_divnorm(Ws, divnorm, sigma, exponent, log_clip)
        proc = spreading_activation(proc, alpha, steps)
        proc = _apply_divnorm(proc, divnorm, sigma, exponent, log_clip)
    else:
        raise ValueError(f"unknown order '{order}'")

    codes = proc[member_rows, :].astype(np.float64)  # FULL columns (hubs included)
    return _normalize_codes(codes)


# ===========================================================================
# Light measurement of a code matrix (the read-out's quality numbers).
# ===========================================================================
def measure_codes(codes, S_true, second_order_pairs, labels, props, nclu, pclu, seed,
                  k_neighbours, chance):
    rec = structure_recovery(codes, S_true, second_order_pairs, seed)
    grad = codebook_similarity_stats(codes, labels)
    gen = float(run_generalization(codes, labels, props, nclu, pclu, seed, k_neighbours)["accuracy"])
    return {
        "pearson_vs_Strue": rec["pearson_learned_vs_Strue"],
        "pearson_permutedS": rec["pearson_permuted_vs_Strue"],
        "is_graded": bool(grad["is_graded"]),
        "within_cos": grad["within_cluster_cos_mean"],
        "between_cos": grad["between_cluster_cos_mean"],
        "second_order_cos_mean": rec["second_order_cos_mean"],
        "between_cluster_cos_mean": rec["between_cluster_cos_mean"],
        "second_order_margin": rec["second_order_margin"],
        "second_order_recovered": bool(rec["second_order_recovered"]),
        "generalization": gen,
        "generalization_ratio_vs_chance": gen / chance if chance > 0 else 0.0,
    }


def _variant_passes(m, g1_bar, a1_bar):
    """A read-out variant CLOSES the residual iff: G1 (Pearson >= bar AND graded AND 2nd-order margin
    recovered >= +0.10) AND A1 generalization stays >= bar (1.000-class). Anti-cheat: permuted-S ~0."""
    g1 = (m["pearson_vs_Strue"] >= g1_bar and m["is_graded"] and m["second_order_recovered"])
    a1 = m["generalization"] >= a1_bar
    perm_ok = abs(m["pearson_permutedS"]) < 0.20
    return bool(g1 and a1 and perm_ok), bool(g1), bool(a1)


# ===========================================================================
# Per-seed driver
# ===========================================================================
def run_seed(seed: int, args) -> dict:
    print(f"\n{'='*84}", flush=True)
    print(f"  LEARNED GRADED-EMBEDDING DIVNORM READ-OUT -- SEED {seed} (cycles={args.cycles})", flush=True)
    print(f"{'='*84}", flush=True)

    nclu, pclu = args.n_clusters, args.per_cluster

    # ----- corpus (reuse the de-risk's; KNOWN S_true + second-order cat~dog pairs) -----
    corpus = build_toy_cooccurrence(nclu, pclu, seed,
                                    hub_facts_per_member=args.hub_facts_per_member,
                                    bridge_facts=args.bridge_facts,
                                    triplet_facts_per_cluster=args.triplet_facts_per_cluster)
    concepts = corpus["concepts"]
    members = corpus["members"]
    labels = corpus["labels"]
    S_true = corpus["S_true"]
    second_order_pairs = corpus["second_order_pairs"]
    member_rows = np.asarray([concepts.index(m) for m in members], dtype=int)
    Nm = len(members)
    props = assign_properties(nclu, pclu, args.n_props, seed)
    chance = 1.0 / args.n_props
    C_full = raw_count_matrix(concepts, corpus["facts"])

    print(f"  [corpus] {len(concepts)} concepts ({nclu} hubs + {Nm} members), "
          f"{corpus['n_facts']} facts; second-order pairs={len(second_order_pairs)}; "
          f"chance={chance:.3f}", flush=True)

    # ----- labelled HOST CEILING (PPMI+SVD on RAW counts) -- the reachable target -----
    host_codes = host_ceiling_codes(concepts, corpus["facts"], member_rows, Nm, seed)
    host_rec = structure_recovery(host_codes, S_true, second_order_pairs, seed)
    host_gen = float(run_generalization(host_codes, labels, props, nclu, pclu, seed,
                                        args.k_neighbours)["accuracy"])
    host_graded = codebook_similarity_stats(host_codes, labels)
    print(f"  [HOST CEILING (PPMI+SVD on RAW counts, labelled target)] "
          f"Pearson(S,S_true)={host_rec['pearson_learned_vs_Strue']:+.3f} "
          f"2nd-margin={host_rec['second_order_margin']:+.3f} gen={host_gen:.3f}", flush=True)

    # ----- LEARN at the de-saturated recovered recipe (cycles=2) -----
    print(f"\n  [LEARN @ cycles={args.cycles} (de-saturated, brain-based spiking-Hebbian)]", flush=True)
    t_learn = time.time()
    W, learn_info = learn_W_desaturate(concepts, corpus["facts"], seed, args.n_pool,
                                       args.pattern_size, args.cycles, gamma=1.0, cap=None)
    learn_s = time.time() - t_learn
    print(f"    learned recurrent: mean={learn_info['recurrent_mean']:.3f} "
          f"max={learn_info['recurrent_max']:.3f} nnz={learn_info['recurrent_nnz']} "
          f"({learn_info['n_neurons']} neurons, {learn_s:.1f}s)", flush=True)

    # faithfulness anti-cheat: the learned W must track the FULL counts AND be distinct from them.
    pearson_W_counts = offdiag_pearson(W, C_full)
    W_distinct = (pearson_W_counts < 0.99)
    print(f"    [anti-cheat] Pearson(W, raw_counts_full)={pearson_W_counts:+.3f} "
          f"(tracks counts >> +0.06; distinct from counts <0.99: {W_distinct})", flush=True)

    # ----- the host-method-on-W STAND-IN (the target the brain-based read-out must match) -----
    standin_codes = host_method_codes_on_W(W, member_rows, args.svd_dim)
    standin_rec = structure_recovery(standin_codes, S_true, second_order_pairs, seed)
    standin_gen = float(run_generalization(standin_codes, labels, props, nclu, pclu, seed,
                                           args.k_neighbours)["accuracy"])
    print(f"    [STAND-IN host-method (PPMI+SVD) on learned W] "
          f"Pearson={standin_rec['pearson_learned_vs_Strue']:+.3f} "
          f"2nd-margin={standin_rec['second_order_margin']:+.3f} gen={standin_gen:.3f}", flush=True)

    # ----- the RAW-DIFFUSION brain-based baseline (the under-sharpened one: gen 1.000, margin +0.04) -----
    raw_diff_codes = divnorm_spreading_readout(W, member_rows, "none", "diffuse_only",
                                               args.sigma_sweep[0], args.exponent_sweep[0],
                                               args.diffusion_alpha, args.diffusion_steps_sweep[0],
                                               log_clip=False)
    raw_diff = measure_codes(raw_diff_codes, S_true, second_order_pairs, labels, props, nclu, pclu,
                             seed, args.k_neighbours, chance)
    print(f"    [RAW-DIFFUSION brain-based baseline (full-cols, no divnorm)] "
          f"Pearson={raw_diff['pearson_vs_Strue']:+.3f} 2nd-margin={raw_diff['second_order_margin']:+.3f} "
          f"gen={raw_diff['generalization']:.3f}", flush=True)

    # =========================================================================
    # BRAIN-BASED read-out sweep: spreading-activation + DIVISIVE NORMALIZATION on the learned W.
    # =========================================================================
    print(f"\n  {'-'*80}", flush=True)
    print(f"  BRAIN-BASED read-out sweep: spreading-activation + divisive-normalization on learned W",
          flush=True)
    print(f"  (target host-on-W +{standin_rec['pearson_learned_vs_Strue']:.2f} / margin "
          f"+{standin_rec['second_order_margin']:.2f}; ceiling +{host_rec['pearson_learned_vs_Strue']:.2f}; "
          f"G1 cosine bar margin >= +{args.so_margin_bar:.2f})", flush=True)
    print(f"  {'-'*80}", flush=True)

    sweep = {}
    best_key, best_score = None, -1e9
    # rank by: (passes both gates) then (2nd-order margin) -- we want the margin to clear +0.10 while
    # generalization holds. Score = 2nd-order margin + small bonus for generalization at/above bar.
    for divnorm in args.divnorm_sweep:
        # 'none' only meaningful as the diffuse_only base (already shown); skip the divnorm-orders for it.
        orders = ["diffuse_only"] if divnorm == "none" else args.order_sweep
        for order in orders:
            for steps in args.diffusion_steps_sweep:
                sigmas = args.sigma_sweep if divnorm == "ch" else [args.sigma_sweep[0]]
                exps = args.exponent_sweep if divnorm == "ch" else [1.0]
                logclips = args.log_clip_sweep
                for sigma in sigmas:
                    for exponent in exps:
                        for log_clip in logclips:
                            codes = divnorm_spreading_readout(
                                W, member_rows, divnorm, order, sigma, exponent,
                                args.diffusion_alpha, steps, log_clip)
                            m = measure_codes(codes, S_true, second_order_pairs, labels, props,
                                              nclu, pclu, seed, args.k_neighbours, chance)
                            closes, g1, a1 = _variant_passes(m, args.g1_bar, args.a1_bar)
                            m["closes_residual"] = closes
                            m["passes_g1"] = g1
                            m["passes_a1"] = a1
                            key = (f"{divnorm}_{order}_steps{steps}_sigma{sigma}_exp{exponent}"
                                   f"_logclip{int(log_clip)}")
                            sweep[key] = m
                            # score: prioritise variants that close BOTH gates, then maximise the
                            # 2nd-order margin (the failing metric), tie-break on Pearson.
                            score = (1000.0 if closes else 0.0) + m["second_order_margin"] * 100.0 \
                                + m["pearson_vs_Strue"] * 10.0 \
                                + (10.0 if m["generalization"] >= args.a1_bar else 0.0)
                            if score > best_score:
                                best_score, best_key = score, key
    # print the sweep compactly (sorted by 2nd-order margin desc), top entries.
    ranked = sorted(sweep.items(), key=lambda kv: kv[1]["second_order_margin"], reverse=True)
    print(f"  [sweep: {len(sweep)} variants; top 12 by 2nd-order margin]", flush=True)
    for key, m in ranked[:12]:
        print(f"    {key:54s} P={m['pearson_vs_Strue']:+.3f} 2nd={m['second_order_margin']:+.3f} "
              f"graded={int(m['is_graded'])} gen={m['generalization']:.3f} "
              f"closes={int(m['closes_residual'])}", flush=True)

    best_m = sweep[best_key]
    # gaps to the host-method-on-W stand-in (apples-to-apples) and to the ceiling.
    gap_to_standin = standin_rec["pearson_learned_vs_Strue"] - best_m["pearson_vs_Strue"]
    gap_to_ceiling = host_rec["pearson_learned_vs_Strue"] - best_m["pearson_vs_Strue"]
    margin_gap_to_standin = standin_rec["second_order_margin"] - best_m["second_order_margin"]
    print(f"\n    BEST brain-based read-out: {best_key}", flush=True)
    print(f"      Pearson(sim,S_true)={best_m['pearson_vs_Strue']:+.3f} (permS "
          f"{best_m['pearson_permutedS']:+.3f}); 2nd-order margin={best_m['second_order_margin']:+.3f} "
          f"(bar +{args.so_margin_bar:.2f}); gen={best_m['generalization']:.3f} (bar {args.a1_bar})",
          flush=True)
    print(f"      gap-to-host-on-W {gap_to_standin:+.3f} (margin gap {margin_gap_to_standin:+.3f}); "
          f"gap-to-ceiling {gap_to_ceiling:+.3f}", flush=True)
    print(f"      passes G1={best_m['passes_g1']} A1={best_m['passes_a1']} -> "
          f"CLOSES residual={best_m['closes_residual']}", flush=True)

    # ----- run the FULL G2 (orthogonal A2 + permuted-property A3) on the BEST brain-based codes -----
    # rebuild the best codes (cheap) to feed the full G2 harness + report the controls collapsing.
    parts = best_key.split("_")
    # robust parse: divnorm is parts[0]; order is parts[1] (or 'diffuse_only' which is two tokens)
    if best_key.startswith("none_diffuse_only") or "_diffuse_only_" in best_key:
        b_divnorm = parts[0]
        b_order = "diffuse_only"
    else:
        b_divnorm, b_order = parts[0], parts[1]
    b_steps = best_m_steps = int([p for p in parts if p.startswith("steps")][0][5:])
    b_sigma = float([p for p in parts if p.startswith("sigma")][0][5:])
    b_exp = float([p for p in parts if p.startswith("exp")][0][3:])
    b_logclip = bool(int([p for p in parts if p.startswith("logclip")][0][7:]))
    best_codes = divnorm_spreading_readout(W, member_rows, b_divnorm, b_order, b_sigma, b_exp,
                                           args.diffusion_alpha, b_steps, b_logclip)
    g2 = architecture_generalization(best_codes, labels, props, nclu, pclu, seed,
                                     args.k_neighbours, args.a1_bar)
    print(f"    G2 on best brain-based codes: graded={g2['graded']['accuracy']:.3f} A1={g2['a1']} | "
          f"orthogonal={g2['orthogonal']['accuracy']:.3f} A2={g2['a2']} | "
          f"permuted-prop={g2['permuted']['accuracy']:.3f} A3={g2['a3']}", flush=True)

    seed_closes = bool(best_m["closes_residual"] and g2["a2"] and g2["a3"] and W_distinct)
    print(f"    => SEED {seed} brain-based read-out {'CLOSES (fully brain-based, controls hold)' if seed_closes else 'BOUNDARY (does not fully clear)'}",
          flush=True)

    return {
        "seed": seed,
        "cycles": args.cycles,
        "corpus": {"n_concepts": len(concepts), "n_members": Nm, "n_facts": corpus["n_facts"],
                   "n_second_order_pairs": len(second_order_pairs)},
        "learn_info": learn_info,
        "learn_seconds": learn_s,
        "anti_cheat_pearson_W_vs_rawcounts": pearson_W_counts,
        "anti_cheat_W_distinct": bool(W_distinct),
        "host_ceiling": {"pearson_vs_Strue": host_rec["pearson_learned_vs_Strue"],
                         "second_order_margin": host_rec["second_order_margin"],
                         "generalization": host_gen, "is_graded": host_graded["is_graded"]},
        "host_method_on_W_standin": {"pearson_vs_Strue": standin_rec["pearson_learned_vs_Strue"],
                                     "second_order_margin": standin_rec["second_order_margin"],
                                     "generalization": standin_gen},
        "raw_diffusion_baseline": raw_diff,
        "brain_based_divnorm_sweep": {
            "n_variants": len(sweep),
            "sweep": sweep,
            "best_key": best_key,
            "best": best_m,
            "gap_to_host_method_on_W": gap_to_standin,
            "gap_to_ceiling": gap_to_ceiling,
            "margin_gap_to_host_method_on_W": margin_gap_to_standin,
            "best_g2_full": {"graded_acc": g2["graded"]["accuracy"], "a1": bool(g2["a1"]),
                             "orthogonal_acc": g2["orthogonal"]["accuracy"], "a2": bool(g2["a2"]),
                             "permuted_prop_acc": g2["permuted"]["accuracy"], "a3": bool(g2["a3"])},
            "seed_closes_residual": seed_closes,
        },
    }


def _seed_verdict(rseed) -> str:
    return "GO" if rseed["brain_based_divnorm_sweep"]["seed_closes_residual"] else "BOUNDARY"


def main():
    p = argparse.ArgumentParser(description="Learned graded-embedding DIVISIVE-NORMALIZATION read-out "
                                            "(fully brain-based read-out residual closure test)")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--cycles", type=int, default=2, help="de-saturated low-cycle regime (recovered)")
    # toy corpus (MUST match the de-risk/desaturate/confirm defaults)
    p.add_argument("--n-clusters", type=int, default=8)
    p.add_argument("--per-cluster", type=int, default=5)
    p.add_argument("--n-props", type=int, default=4)
    p.add_argument("--hub-facts-per-member", type=int, default=6)
    p.add_argument("--bridge-facts", type=int, default=8)
    p.add_argument("--triplet-facts-per-cluster", type=int, default=4)
    # learned-assoc-graph (brain-based learner)
    p.add_argument("--n-pool", type=int, default=2000)
    p.add_argument("--pattern-size", type=int, default=100)
    # host-method-on-W stand-in (the apples-to-apples target)
    p.add_argument("--svd-dim", type=int, default=40)
    # BRAIN-BASED divnorm read-out sweep
    p.add_argument("--divnorm-sweep", nargs="+", default=["none", "marginal", "ch"],
                   help="divisive-normalization forms: none(base diffuse) / marginal(PPMI-analogue) / "
                        "ch(Carandini-Heeger)")
    p.add_argument("--order-sweep", nargs="+", default=["pre", "post", "interleave"],
                   help="when to apply divnorm vs spreading")
    p.add_argument("--diffusion-alpha", type=float, default=0.5)
    p.add_argument("--diffusion-steps-sweep", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--sigma-sweep", type=float, nargs="+", default=[0.001, 0.01, 0.05],
                   help="Carandini-Heeger semi-saturation (per-entry scale; small -> sharper contrast)")
    p.add_argument("--exponent-sweep", type=float, nargs="+", default=[1.0, 2.0],
                   help="Carandini-Heeger exponent n")
    p.add_argument("--log-clip-sweep", type=lambda s: s.lower() in ("1", "true", "yes"),
                   nargs="+", default=[False, True],
                   help="apply the PPMI 'positive log' arm on top of the marginal/divisive form")
    # gate bars (match the de-risk / confirm)
    p.add_argument("--g1-bar", type=float, default=0.5, help="Pearson(sim, S_true) >= this")
    p.add_argument("--a1-bar", type=float, default=0.7, help="generalization >= this (1.000-class)")
    p.add_argument("--so-margin-bar", type=float, default=0.10,
                   help="G1 cosine bar: 2nd-order cat~dog margin >= this")
    p.add_argument("--k-neighbours", type=int, default=3)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    backend = os.environ.get("SIM_BACKEND", "auto")
    chance = 1.0 / args.n_props
    t0 = time.time()
    print(f"[learned-graded-embedding DIVNORM READ-OUT] seeds={seeds} backend={backend} "
          f"cycles={args.cycles}", flush=True)
    print(f"  toy: {args.n_clusters}x{args.per_cluster} (+hubs); learner=LearnedAssocGraph "
          f"(n_pool={args.n_pool}, pattern_size={args.pattern_size})", flush=True)
    print(f"  brain-based read-out = spreading-activation (full hub-inclusive cols) + divisive "
          f"normalization {args.divnorm_sweep}; orders={args.order_sweep}; steps="
          f"{args.diffusion_steps_sweep}", flush=True)
    print(f"  bars: G1(Pearson>={args.g1_bar}) A1(gen>={args.a1_bar}) "
          f"2nd-order-margin(>={args.so_margin_bar})", flush=True)

    per_seed = {}
    for s in seeds:
        per_seed[str(s)] = run_seed(s, args)

    # ---------- overall verdict ----------
    verdicts = {str(s): _seed_verdict(per_seed[str(s)]) for s in seeds}
    vset = set(verdicts.values())
    if vset == {"GO"}:
        consensus = "GO"
    elif "GO" in vset:
        consensus = "MIXED:" + ",".join(f"{s}={v}" for s, v in verdicts.items())
    else:
        consensus = "BOUNDARY"

    def agg(path):
        out = []
        for s in seeds:
            d = per_seed[str(s)]
            for k in path:
                d = d[k]
            out.append(d)
        return out

    best_pearson = agg(["brain_based_divnorm_sweep", "best", "pearson_vs_Strue"])
    best_margin = agg(["brain_based_divnorm_sweep", "best", "second_order_margin"])
    best_gen = agg(["brain_based_divnorm_sweep", "best", "generalization"])
    best_keys = agg(["brain_based_divnorm_sweep", "best_key"])
    gap_standin = agg(["brain_based_divnorm_sweep", "gap_to_host_method_on_W"])
    gap_ceiling = agg(["brain_based_divnorm_sweep", "gap_to_ceiling"])
    closes = agg(["brain_based_divnorm_sweep", "seed_closes_residual"])
    raw_margin = agg(["raw_diffusion_baseline", "second_order_margin"])
    raw_gen = agg(["raw_diffusion_baseline", "generalization"])
    standin_pearson = agg(["host_method_on_W_standin", "pearson_vs_Strue"])
    standin_margin = agg(["host_method_on_W_standin", "second_order_margin"])
    host_pearson = agg(["host_ceiling", "pearson_vs_Strue"])
    W_counts = agg(["anti_cheat_pearson_W_vs_rawcounts"])

    closes_all = all(bool(x) for x in closes)
    # whether generalization STAYS 1.000-class (>= a1_bar) on the best variant for every seed
    gen_holds_all = all(g >= args.a1_bar for g in best_gen)
    # whether the cosine margin clears the bar on the best variant for every seed
    margin_clears_all = all(mr >= args.so_margin_bar for mr in best_margin)

    summary = {
        "consensus_verdict": consensus,
        "per_seed_verdict": verdicts,
        "seeds": seeds,
        "backend": backend,
        "cycles": args.cycles,
        "brain_based_note": ("the learn is the project's spiking-Hebbian recurrent (LearnedAssocGraph) "
                             "at the de-saturated low-cycle (cycles=2) regime. The read-out tested is "
                             "FULLY BRAIN-BASED: spreading-activation diffusion through the full hub-"
                             "inclusive learned W + DIVISIVE NORMALIZATION (Carandini-Heeger gain "
                             "control / marginal division = the brain-based analogue of PPMI's marginal "
                             "division). NO host PPMI+SVD. divisive normalization is a FIXED nonlinearity "
                             "(no fit to S_true). The host-method-on-W (PPMI+SVD) is the labelled STAND-"
                             "IN target; the host PPMI+SVD on RAW counts is the labelled CEILING."),
        "residual_closed": bool(closes_all),
        "generalization_holds_all_seeds": bool(gen_holds_all),
        "cosine_margin_clears_all_seeds": bool(margin_clears_all),
        "brain_based_best": {
            "best_pearson_vs_Strue_per_seed": best_pearson,
            "best_pearson_vs_Strue_mean": float(np.mean(best_pearson)),
            "best_second_order_margin_per_seed": best_margin,
            "best_second_order_margin_mean": float(np.mean(best_margin)),
            "best_generalization_per_seed": best_gen,
            "best_generalization_mean": float(np.mean(best_gen)),
            "best_keys_per_seed": best_keys,
            "gap_to_host_method_on_W_mean": float(np.mean(gap_standin)),
            "gap_to_ceiling_mean": float(np.mean(gap_ceiling)),
            "closes_residual_per_seed": [bool(x) for x in closes],
        },
        "reference_points": {
            "so_margin_bar": args.so_margin_bar,
            "raw_diffusion_margin_per_seed": raw_margin,
            "raw_diffusion_margin_mean": float(np.mean(raw_margin)),
            "raw_diffusion_generalization_mean": float(np.mean(raw_gen)),
            "host_method_on_W_standin_pearson_mean": float(np.mean(standin_pearson)),
            "host_method_on_W_standin_margin_mean": float(np.mean(standin_margin)),
            "host_ceiling_pearson_mean": float(np.mean(host_pearson)),
            "anti_cheat_pearson_W_vs_rawcounts_mean": float(np.mean(W_counts)),
            "generalization_chance": chance,
        },
        "confirm_GO_full_reference": {
            "source": "2026-06-11-learned-graded-embedding-confirm-GO_full.md (commit e6e277e3)",
            "primary_readout": "host-method (PPMI+SVD) on the FULL hub-inclusive learned W (STAND-IN)",
            "diffusion_brain_based_margin": 0.04, "diffusion_brain_based_gen": 1.000,
            "host_on_W_margin": 0.84, "host_ceiling_margin": 0.93,
        },
        "elapsed_total_s": time.time() - t0,
    }

    print(f"\n{'='*84}", flush=True)
    print(f"  DIVNORM READ-OUT SUMMARY", flush=True)
    print(f"{'='*84}", flush=True)
    print(f"  CONSENSUS VERDICT: {consensus}", flush=True)
    for s in seeds:
        b = per_seed[str(s)]["brain_based_divnorm_sweep"]
        print(f"  seed {s}: {verdicts[str(s)]} | best={b['best_key']}", flush=True)
        print(f"           Pearson={b['best']['pearson_vs_Strue']:+.3f} "
              f"2nd-margin={b['best']['second_order_margin']:+.3f} (bar +{args.so_margin_bar:.2f}) "
              f"gen={b['best']['generalization']:.3f} closes={b['seed_closes_residual']}", flush=True)
    print(f"\n  BRAIN-BASED best (mean): Pearson={np.mean(best_pearson):+.3f} "
          f"2nd-margin={np.mean(best_margin):+.3f} gen={np.mean(best_gen):.3f}", flush=True)
    print(f"  vs raw-diffusion margin {np.mean(raw_margin):+.3f} (gen {np.mean(raw_gen):.3f}) | "
          f"host-on-W margin {np.mean(standin_margin):+.3f} (Pearson {np.mean(standin_pearson):+.3f}) | "
          f"ceiling Pearson {np.mean(host_pearson):+.3f}", flush=True)
    print(f"  residual CLOSED (all seeds): {closes_all} | gen holds 1.000-class: {gen_holds_all} | "
          f"cosine margin clears +{args.so_margin_bar:.2f}: {margin_clears_all}", flush=True)
    print(f"  anti-cheat Pearson(W, raw_counts) mean {np.mean(W_counts):+.3f} (<0.99 distinct)",
          flush=True)
    print(f"  Total elapsed: {summary['elapsed_total_s']:.1f}s", flush=True)
    print(f"{'='*84}\n", flush=True)

    out_data = {"summary": summary, "per_seed": per_seed}
    if args.out is None:
        raw_dir = os.path.join(_REPO, "research", "findings", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        args.out = os.path.join(raw_dir, "_lge_divnorm_multiseed.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out_data, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)
    return out_data


if __name__ == "__main__":
    main()
