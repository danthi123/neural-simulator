"""LEARNED GRADED-EMBEDDING DIAGNOSIS -- localize WHERE the brain-based learned-embedding collapsed.

CONTEXT (the NEGATIVE this dissects):
  The dual/CLS architecture's one unbuilt piece is a LEARNED graded-similarity cortex embedding.
  The recommended brain-based mechanism -- a spiking-Hebbian co-occurrence learner
  (`LearnedAssocGraph`) + a diffusion graded read-out -- FAILED
  (research/findings/2026-06-11-learned-graded-embedding-derisk-NEGATIVE.md, commit 4272f7dc):
    - learned codes COLLAPSED to near-uniform (within-cos 0.955 ~= between-cos 0.956),
    - generalization at chance (0.237 < random 0.312),
    - Pearson(S_learned, S_true) = -0.024.
  BUT the host ceiling (PPMI + truncated SVD) on the EXACT SAME co-occurrence data recovers the
  structure PERFECTLY (gen 1.000, Pearson +0.932). So the corpus + architecture are fine; the
  brain-based LEARNING + READ-OUT is the failure.

THE DIAGNOSIS (this runner):
  Learn ONCE on the real corpus (GPU, ~2.3 min foreground via LearnedAssocGraph), then dissect the
  pipeline at each stage and compute Pearson(stage_similarity, S_true) + the generalization gate at
  each stage:

  STAGE W   (the learned weights, BEFORE any read-out):
    extract the Hebbian-learned recurrent W [Nc, Nc] (mean a->b recurrent weight between each
    concept-pair's sparse patterns -- the SAME extraction the de-risk's learn_assoc_matrix does);
    form member concept vectors from W's MEMBER<->MEMBER submatrix rows (native mean-removed +
    unit-norm); sim_W = cosine.
      -> Pearson(sim_W, S_true)            : does the Hebbian LEARN capture the graded structure?
      -> Pearson(sim_W, C_counts)          : does W track the raw co-occurrence COUNT matrix the
                                             corpus defines (i.e. is W a faithful learned count, or
                                             degenerate/saturated/uniform)?
      -> generalization on the W-row codes : property inheritance directly off the learned weights.

  STAGE diffusion (the CURRENT read-out): apply the de-risk's graded_readout (diffusion alpha 0.5,
    2 steps) -> reproduce the collapse (Pearson ~= -0.02). Confirm WHETHER/WHERE the read-out
    collapses what STAGE W carried.

  STAGE PPMI/divisive-norm (the PRIME-SUSPECT FIX): apply a PPMI / divisive-normalization to the
    LEARNED W (divide W by its row+col marginals, log-positive-clip -- the brain-based analogue =
    DIVISIVE NORMALIZATION, Carandini-Heeger, a canonical cortical computation), THEN a low-dim
    read-out (truncated SVD over the normalized W). Two variants:
      (a) divnorm-only  : sym divisive normalization of W, code = normalized rows (no SVD).
      (b) PPMI+SVD on W : PPMI on W + truncated SVD -> dense graded codes (the host *method* applied
                          to the BRAIN-LEARNED W, NOT a fresh host SVD of the raw counts).
      -> Pearson(sim_PPMI_W, S_true) + generalization gate.
    Does PPMI/divisive-normalization on the LEARNED weights recover the structure toward the host
    ceiling?

  LOCALIZE + VERDICT:
    - If STAGE W already carries the structure (Pearson high) but diffusion collapses it AND
      PPMI/divnorm recovers it -> the READ-OUT is the bug; the fix is a brain-based divisive-
      normalization read-out (cheap; re-run the full de-risk with it).
    - If STAGE W is degenerate (Pearson ~0) even before any read-out -> the Hebbian LEARN failed to
      capture graded W; the fix is a different brain-based rule (predictive / contrastive-Hebbian).

ANTI-CHEATS:
  - report the per-stage Pearson AND the generalization at EACH stage (don't conflate);
  - a PERMUTED-S baseline (~0) for ANY "recovered" stage;
  - the host PPMI+SVD ceiling (on the RAW counts) as the labelled TARGET (NOT the deliverable);
  - the deliverable PPMI/divnorm read-out runs on the BRAIN-LEARNED W -- assert W != raw counts
    (W_vs_counts_pearson < 0.999 and a max-abs-diff guard) so a "recovery" is NOT silently
    re-deriving the host ceiling from the counts.

Run (GPU, FOREGROUND -- the ONE spiking-learn step is ~2.3 min inline; NO background):
  SIM_BACKEND=cupy python -m research.runners.learned_graded_embedding_diagnose \
      --seed 42 --out research/findings/raw/_lge_diagnose_seed42.json
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

# Reuse the de-risk's corpus builder + learn + graded read-out + structure-recovery machinery.
from research.runners.learned_graded_embedding_derisk_probe import (  # noqa: E402
    build_toy_cooccurrence,
    learn_assoc_matrix,
    graded_readout,
    structure_recovery,
    host_ceiling_codes,
    random_gaussian_codes,
)
from research.runners.dual_cls_architecture_proof_probe import (  # noqa: E402
    codebook_similarity_stats,
    assign_properties,
    run_generalization,
)


# ===========================================================================
# helpers
# ===========================================================================
def raw_count_matrix(concepts: list, facts: list) -> np.ndarray:
    """The raw co-occurrence COUNT matrix C [Nc, Nc] the corpus defines (the thing W is *supposed*
    to be a learned analogue of). Used to ask: does the learned W track the counts, and is the
    PPMI/divnorm fix running on W rather than silently on C?"""
    Nc = len(concepts)
    idx = {c: i for i, c in enumerate(concepts)}
    C = np.zeros((Nc, Nc), dtype=np.float64)
    for f in facts:
        ids = [idx[c] for c in f if c in idx]
        for a in ids:
            for b in ids:
                if a != b:
                    C[a, b] += 1.0
    return C


def _normalize_codes(codes: np.ndarray) -> np.ndarray:
    """Native code convention: mean-removed per row + unit-norm."""
    codes = codes - codes.mean(axis=1, keepdims=True)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    return codes


def offdiag_pearson(A: np.ndarray, B: np.ndarray) -> float:
    """Pearson over the strict-upper-triangle (off-diagonal) of two same-shape symmetric matrices."""
    n = A.shape[0]
    iu = np.triu_indices(n, k=1)
    a, b = A[iu], B[iu]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def member_submatrix(M: np.ndarray, member_rows: np.ndarray) -> np.ndarray:
    """The MEMBER<->MEMBER submatrix (the cortex code space is member-to-member; hubs are mediators)."""
    return M[np.ix_(member_rows, member_rows)].astype(np.float64)


def rows_to_codes(M: np.ndarray) -> np.ndarray:
    """Each concept's code = its (symmetrized, zero-diagonal) row in M; native-normalized."""
    Ms = 0.5 * (M + M.T)
    np.fill_diagonal(Ms, 0.0)
    return _normalize_codes(Ms.copy())


def ppmi_transform(M: np.ndarray) -> np.ndarray:
    """PPMI on a non-negative association matrix M (the host *method*, applied here to the LEARNED W).
    PPMI = max(0, log( (M_ij * total) / (row_i * col_j) )). This is the marginal division that
    removes high-frequency-concept dominance -- the prime suspect the brain-based read-out lacks."""
    M = np.maximum(M, 0.0)
    total = M.sum()
    if total <= 0:
        return np.zeros_like(M)
    row = M.sum(axis=1, keepdims=True)
    col = M.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((M * total) / (row * col + 1e-12) + 1e-12)
    return np.maximum(pmi, 0.0)


def divisive_norm(M: np.ndarray) -> np.ndarray:
    """Brain-based DIVISIVE NORMALIZATION (Carandini-Heeger) of an association matrix:
    each entry divided by a pooled (row+col marginal) normalizer. The canonical cortical
    'remove the common/high-frequency drive' computation -- the analogue of PPMI's marginal division,
    WITHOUT the log. Symmetric: M_ij / sqrt(rowsum_i * colsum_j)."""
    M = np.maximum(M, 0.0)
    row = M.sum(axis=1, keepdims=True)
    col = M.sum(axis=0, keepdims=True)
    denom = np.sqrt((row + 1e-12) * (col + 1e-12))
    return M / denom


def svd_lowdim(M: np.ndarray, dim: int) -> np.ndarray:
    """Truncated SVD low-dim embedding of a (normalized) association matrix -> dense codes."""
    U, Sv, _ = np.linalg.svd(M, full_matrices=False)
    d = min(dim, U.shape[1])
    return (U[:, :d] * Sv[:d]).astype(np.float64)


def gen_acc(codes: np.ndarray, labels: np.ndarray, props: np.ndarray,
            nclu: int, pclu: int, seed: int, k_neighbours: int) -> float:
    return float(run_generalization(codes, labels, props, nclu, pclu, seed, k_neighbours)["accuracy"])


def stage_report(name: str, codes: np.ndarray, S_true: np.ndarray, second_order_pairs: list,
                 labels: np.ndarray, props: np.ndarray, nclu: int, pclu: int, seed: int,
                 k_neighbours: int, chance: float) -> dict:
    """Full per-stage report: Pearson(sim, S_true), permuted-S baseline, graded stats, generalization
    (+ the permuted-S generalization baseline = shuffle code rows -> property inference must collapse)."""
    rec = structure_recovery(codes, S_true, second_order_pairs, seed)
    grad = codebook_similarity_stats(codes, labels)
    gen = gen_acc(codes, labels, props, nclu, pclu, seed, k_neighbours)
    # permuted-code generalization baseline (shuffle rows -> structure-free; gen must -> chance).
    rng = np.random.RandomState(seed * 99 + 17)
    perm = rng.permutation(codes.shape[0])
    gen_permcode = gen_acc(codes[perm], labels, props, nclu, pclu, seed, k_neighbours)
    rep = {
        "pearson_vs_Strue": rec["pearson_learned_vs_Strue"],
        "pearson_permutedS_baseline": rec["pearson_permuted_vs_Strue"],
        "within_cos": grad["within_cluster_cos_mean"],
        "between_cos": grad["between_cluster_cos_mean"],
        "graded_margin": grad["graded_margin"],
        "is_graded": grad["is_graded"],
        "second_order_margin": rec["second_order_margin"],
        "second_order_recovered": rec["second_order_recovered"],
        "generalization": gen,
        "generalization_ratio_vs_chance": gen / chance if chance > 0 else 0.0,
        "generalization_permutedcode_baseline": gen_permcode,
    }
    print(f"    [{name:22s}] Pearson(S,S_true)={rep['pearson_vs_Strue']:+.3f} "
          f"(permS {rep['pearson_permutedS_baseline']:+.3f}) | within {rep['within_cos']:+.3f} "
          f"between {rep['between_cos']:+.3f} margin {rep['graded_margin']:+.3f} "
          f"graded={rep['is_graded']} | gen {rep['generalization']:.3f} "
          f"({rep['generalization_ratio_vs_chance']:.2f}x; permcode {gen_permcode:.3f}) | "
          f"2nd-order margin {rep['second_order_margin']:+.3f}", flush=True)
    return rep


# ===========================================================================
# per-seed driver
# ===========================================================================
def run_seed(seed: int, args) -> dict:
    print(f"\n{'='*78}", flush=True)
    print(f"  LEARNED GRADED-EMBEDDING DIAGNOSIS -- SEED {seed}", flush=True)
    print(f"{'='*78}", flush=True)

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
    print(f"  [corpus] {len(concepts)} concepts ({nclu} hubs + {Nm} members), "
          f"{corpus['n_facts']} facts; second-order pairs={len(second_order_pairs)}; "
          f"chance={chance:.3f}", flush=True)

    # ----- LEARN ONCE on the real corpus (the GPU spiking-Hebbian, ~2.3 min foreground) -----
    print("  [LEARN -- brain-based spiking-Hebbian co-occurrence (LearnedAssocGraph), ONCE]",
          flush=True)
    t_learn = time.time()
    W, learn_info = learn_assoc_matrix(concepts, corpus["facts"], seed,
                                       args.n_pool, args.pattern_size, args.store_cycles)
    learn_s = time.time() - t_learn
    print(f"    learned recurrent: mean={learn_info['recurrent_mean']:.3f} "
          f"max={learn_info['recurrent_max']:.3f} nnz={learn_info['recurrent_nnz']} "
          f"({learn_info['n_neurons']} neurons, {learn_s:.1f}s)", flush=True)

    # raw co-occurrence COUNT matrix (member submatrix) for the W-vs-counts faithfulness check.
    C_full = raw_count_matrix(concepts, corpus["facts"])
    C_members = member_submatrix(C_full, member_rows)
    W_members = member_submatrix(W, member_rows)

    # ===================== STAGE W (learned weights, BEFORE any read-out) =====================
    print("\n  [STAGE W -- the learned recurrent weights, BEFORE any read-out]", flush=True)
    sim_W_codes = rows_to_codes(W_members)
    sim_W = sim_W_codes @ sim_W_codes.T
    pearson_W_vs_Strue = offdiag_pearson(sim_W, S_true)
    # does W track the raw counts? (faithfulness of the learned weights to the co-occurrence stats)
    pearson_W_vs_counts = offdiag_pearson(W_members, C_members)
    # also: how degenerate is W? coefficient of variation of off-diagonal entries.
    iu = np.triu_indices(Nm, k=1)
    w_off = W_members[iu]
    w_cv = float(np.std(w_off) / (np.abs(np.mean(w_off)) + 1e-12))
    stage_W = stage_report("STAGE W (rows)", sim_W_codes, S_true, second_order_pairs,
                           labels, props, nclu, pclu, seed, args.k_neighbours, chance)
    stage_W["pearson_simW_vs_Strue"] = pearson_W_vs_Strue
    stage_W["pearson_W_vs_rawcounts"] = pearson_W_vs_counts
    stage_W["W_offdiag_mean"] = float(np.mean(w_off))
    stage_W["W_offdiag_std"] = float(np.std(w_off))
    stage_W["W_offdiag_cv"] = w_cv
    print(f"    >>> Pearson(sim_W, S_true) = {pearson_W_vs_Strue:+.3f}  "
          f"(W-row codes, before any read-out)", flush=True)
    print(f"    >>> Pearson(W, raw_counts) = {pearson_W_vs_counts:+.3f}  "
          f"(does the learned W track the co-occurrence counts?)  "
          f"W off-diag mean={np.mean(w_off):.4f} std={np.std(w_off):.4f} CV={w_cv:.3f}", flush=True)

    # ===================== STAGE diffusion (the CURRENT read-out) =====================
    print("\n  [STAGE diffusion -- the CURRENT graded read-out (reproduce the collapse)]", flush=True)
    diff_codes = graded_readout(W, member_rows, args.diffusion_alpha, args.diffusion_steps)
    stage_diff = stage_report("STAGE diffusion", diff_codes, S_true, second_order_pairs,
                              labels, props, nclu, pclu, seed, args.k_neighbours, chance)

    # ===================== STAGE PPMI/divisive-norm (the PRIME-SUSPECT FIX), on the LEARNED W ====
    print("\n  [STAGE PPMI/divnorm -- the prime-suspect fix, on the SPIKING-LEARNED W]", flush=True)
    # symmetrize + zero diagonal first (co-occurrence is symmetric).
    Ws = 0.5 * (W_members + W_members.T)
    np.fill_diagonal(Ws, 0.0)

    # (a) divisive-normalization-ONLY read-out (no SVD): brain-based divisive norm, code = rows.
    Wd = divisive_norm(Ws)
    divnorm_codes = _normalize_codes(Wd.copy())
    stage_divnorm = stage_report("STAGE divnorm-only", divnorm_codes, S_true, second_order_pairs,
                                 labels, props, nclu, pclu, seed, args.k_neighbours, chance)

    # (b) PPMI on the LEARNED W + truncated SVD low-dim read-out (host method on brain-learned W).
    ppmi_W = ppmi_transform(Ws)
    ppmi_emb = svd_lowdim(ppmi_W, args.svd_dim)
    ppmi_codes = _normalize_codes(ppmi_emb)
    stage_ppmi = stage_report("STAGE PPMI+SVD(W)", ppmi_codes, S_true, second_order_pairs,
                              labels, props, nclu, pclu, seed, args.k_neighbours, chance)

    # (c) divisive-norm + SVD (the divnorm analogue of PPMI+SVD).
    divnorm_emb = svd_lowdim(Wd, args.svd_dim)
    divnorm_svd_codes = _normalize_codes(divnorm_emb)
    stage_divnorm_svd = stage_report("STAGE divnorm+SVD(W)", divnorm_svd_codes, S_true,
                                     second_order_pairs, labels, props, nclu, pclu, seed,
                                     args.k_neighbours, chance)

    # ANTI-CHEAT: confirm PPMI/divnorm is running on the BRAIN-LEARNED W, not silently on raw counts.
    # PPMI+SVD applied to the RAW COUNTS (member submatrix) -- if STAGE PPMI(W) ~= this, the "fix"
    # is just re-deriving the host ceiling. We WANT them to differ (W != counts).
    Cs = 0.5 * (C_members + C_members.T)
    np.fill_diagonal(Cs, 0.0)
    ppmi_counts_codes = _normalize_codes(svd_lowdim(ppmi_transform(Cs), args.svd_dim))
    sim_ppmi_W = ppmi_codes @ ppmi_codes.T
    sim_ppmi_counts = ppmi_counts_codes @ ppmi_counts_codes.T
    ppmiW_vs_ppmicounts = offdiag_pearson(sim_ppmi_W, sim_ppmi_counts)
    W_is_distinct_from_counts = (pearson_W_vs_counts < 0.999)
    print(f"    [ANTI-CHEAT] Pearson(W, raw_counts)={pearson_W_vs_counts:+.3f} "
          f"(<0.999 => the fix runs on the LEARNED W, distinct from counts: "
          f"{W_is_distinct_from_counts})", flush=True)
    print(f"                 Pearson(sim_PPMI(W), sim_PPMI(counts))={ppmiW_vs_ppmicounts:+.3f} "
          f"(how close the W-fix is to the host-on-counts ceiling)", flush=True)

    # ===================== labelled host CEILING (PPMI+SVD on RAW counts) =====================
    host_codes = host_ceiling_codes(concepts, corpus["facts"], member_rows, Nm, seed)
    host_rec = structure_recovery(host_codes, S_true, second_order_pairs, seed)
    host_gen = gen_acc(host_codes, labels, props, nclu, pclu, seed, args.k_neighbours)
    host_graded = codebook_similarity_stats(host_codes, labels)
    print(f"\n  [HOST CEILING (PPMI+SVD on RAW counts, labelled, NOT deliverable)] "
          f"Pearson(S,S_true)={host_rec['pearson_learned_vs_Strue']:+.3f} gen={host_gen:.3f} "
          f"graded={host_graded['is_graded']}", flush=True)

    # random-Gaussian baseline (the de-risk's beats-baseline reference).
    rand_codes = random_gaussian_codes(Nm, Nm, seed)
    rand_gen = gen_acc(rand_codes, labels, props, nclu, pclu, seed, args.k_neighbours)

    # ===================== LOCALIZATION =====================
    g1_bar = args.g1_bar
    a1_bar = args.a1_bar
    W_carries_structure = (pearson_W_vs_Strue >= g1_bar) and stage_W["is_graded"]
    diffusion_collapses = (stage_diff["pearson_vs_Strue"] < g1_bar * 0.6) and (not stage_diff["is_graded"])
    # the BEST of the PPMI/divnorm fixes (on the LEARNED W).
    fix_stages = {
        "divnorm_only": stage_divnorm,
        "ppmi_svd_W": stage_ppmi,
        "divnorm_svd_W": stage_divnorm_svd,
    }
    best_fix_name = max(fix_stages, key=lambda k: fix_stages[k]["pearson_vs_Strue"])
    best_fix = fix_stages[best_fix_name]
    fix_recovers = (best_fix["pearson_vs_Strue"] >= g1_bar) and best_fix["is_graded"]
    fix_generalizes = best_fix["generalization"] >= a1_bar

    if W_carries_structure and fix_recovers:
        localization = "READ_OUT_BUG"
        verdict_detail = ("STAGE W carries the graded structure; the diffusion read-out "
                          "destroys it; PPMI/divisive-normalization on the LEARNED W recovers it. "
                          "FIX = brain-based divisive-normalization read-out (cheap; re-run the "
                          "de-risk with it).")
    elif W_carries_structure and not fix_recovers:
        localization = "READ_OUT_BUG_PARTIAL"
        verdict_detail = ("STAGE W carries graded structure but none of the tested read-outs "
                          "(diffusion / PPMI / divnorm on W) recover it to the G1 bar -- the "
                          "structure is in W but the read-out family tested is insufficient.")
    elif (not W_carries_structure) and fix_recovers:
        localization = "READ_OUT_BUG_NORMALIZATION"
        verdict_detail = ("STAGE W rows look ungraded by raw cosine, but PPMI/divisive-normalization "
                          "of W RECOVERS the structure -- the LEARN captured it (the marginal-"
                          "normalized W is graded); the missing NORMALIZATION in the read-out was "
                          "the bug. FIX = brain-based divisive-normalization read-out.")
    else:
        localization = "LEARN_FAILURE"
        verdict_detail = ("STAGE W is degenerate (Pearson ~0, ungraded) AND PPMI/divisive-"
                          "normalization of W does NOT recover the structure -- the Hebbian LEARN "
                          "failed to capture a graded W. FIX = a different brain-based rule "
                          "(predictive / contrastive-Hebbian).")

    print(f"\n  {'-'*74}", flush=True)
    print(f"  LOCALIZATION (seed {seed}): {localization}", flush=True)
    print(f"    STAGE W carries structure (Pearson {pearson_W_vs_Strue:+.3f} >= {g1_bar}, "
          f"graded {stage_W['is_graded']}): {W_carries_structure}", flush=True)
    print(f"    diffusion read-out collapses it: {diffusion_collapses} "
          f"(Pearson {stage_diff['pearson_vs_Strue']:+.3f})", flush=True)
    print(f"    BEST fix = {best_fix_name}: Pearson {best_fix['pearson_vs_Strue']:+.3f} "
          f"gen {best_fix['generalization']:.3f} -> recovers={fix_recovers} "
          f"generalizes={fix_generalizes}", flush=True)
    print(f"    => {verdict_detail}", flush=True)
    print(f"  {'-'*74}", flush=True)

    return {
        "seed": seed,
        "corpus": {"n_concepts": len(concepts), "n_members": Nm, "n_facts": corpus["n_facts"],
                   "n_second_order_pairs": len(second_order_pairs)},
        "learn_info": learn_info,
        "learn_seconds": learn_s,
        "stage_W": stage_W,
        "stage_diffusion": stage_diff,
        "stage_divnorm_only": stage_divnorm,
        "stage_ppmi_svd_W": stage_ppmi,
        "stage_divnorm_svd_W": stage_divnorm_svd,
        "host_ceiling": {"pearson_vs_Strue": host_rec["pearson_learned_vs_Strue"],
                         "generalization": host_gen, "is_graded": host_graded["is_graded"]},
        "random_baseline_generalization": rand_gen,
        "anti_cheat": {
            "pearson_W_vs_rawcounts": pearson_W_vs_counts,
            "W_is_distinct_from_counts": bool(W_is_distinct_from_counts),
            "pearson_simPPMI_W_vs_simPPMI_counts": ppmiW_vs_ppmicounts,
        },
        "localization": {
            "verdict": localization,
            "detail": verdict_detail,
            "W_carries_structure": bool(W_carries_structure),
            "diffusion_collapses": bool(diffusion_collapses),
            "best_fix_stage": best_fix_name,
            "best_fix_pearson_vs_Strue": best_fix["pearson_vs_Strue"],
            "best_fix_generalization": best_fix["generalization"],
            "fix_recovers_structure": bool(fix_recovers),
            "fix_generalizes": bool(fix_generalizes),
        },
    }


def main():
    p = argparse.ArgumentParser(description="Learned graded-embedding collapse diagnosis")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", default=None, help="comma list (overrides --seed); usually just 42")
    # toy corpus (MUST match the de-risk's defaults so we reproduce the exact NEGATIVE)
    p.add_argument("--n-clusters", type=int, default=8)
    p.add_argument("--per-cluster", type=int, default=5)
    p.add_argument("--n-props", type=int, default=4)
    p.add_argument("--hub-facts-per-member", type=int, default=6)
    p.add_argument("--bridge-facts", type=int, default=8)
    p.add_argument("--triplet-facts-per-cluster", type=int, default=4)
    # learned-assoc-graph (brain-based learner)
    p.add_argument("--n-pool", type=int, default=2000)
    p.add_argument("--pattern-size", type=int, default=100)
    p.add_argument("--store-cycles", type=int, default=20)
    # current diffusion read-out (the de-risk's defaults -> reproduce the collapse)
    p.add_argument("--diffusion-alpha", type=float, default=0.5)
    p.add_argument("--diffusion-steps", type=int, default=2)
    # the fix read-outs
    p.add_argument("--svd-dim", type=int, default=40,
                   help="truncated-SVD dim for the PPMI/divnorm low-dim read-out on W")
    # gate bars
    p.add_argument("--g1-bar", type=float, default=0.5, help="Pearson(sim, S_true) >= this = recovered")
    p.add_argument("--a1-bar", type=float, default=0.7, help="generalization >= this = generalizes")
    p.add_argument("--k-neighbours", type=int, default=3)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = [args.seed]
    backend = os.environ.get("SIM_BACKEND", "auto")
    t0 = time.time()
    print(f"[learned-graded-embedding DIAGNOSIS] seeds={seeds} backend={backend}", flush=True)
    print(f"  toy: {args.n_clusters}x{args.per_cluster} (+hubs); learner=LearnedAssocGraph "
          f"(n_pool={args.n_pool}, pattern_size={args.pattern_size}, cycles={args.store_cycles})",
          flush=True)
    print(f"  current read-out: diffusion alpha={args.diffusion_alpha} steps={args.diffusion_steps}; "
          f"fix: PPMI/divnorm on W + SVD(dim={args.svd_dim})", flush=True)
    print(f"  bars: G1(Pearson>={args.g1_bar}) A1(gen>={args.a1_bar})", flush=True)

    per_seed = {}
    for s in seeds:
        per_seed[str(s)] = run_seed(s, args)

    # ---------- aggregate ----------
    def agg(path):
        out = []
        for s in seeds:
            d = per_seed[str(s)]
            for k in path:
                d = d[k]
            out.append(d)
        return out

    pW = agg(["stage_W", "pearson_vs_Strue"])
    pDiff = agg(["stage_diffusion", "pearson_vs_Strue"])
    pPPMI = agg(["stage_ppmi_svd_W", "pearson_vs_Strue"])
    pDivnorm = agg(["stage_divnorm_only", "pearson_vs_Strue"])
    pDivnormSvd = agg(["stage_divnorm_svd_W", "pearson_vs_Strue"])
    genW = agg(["stage_W", "generalization"])
    genDiff = agg(["stage_diffusion", "generalization"])
    genPPMI = agg(["stage_ppmi_svd_W", "generalization"])
    genDivnorm = agg(["stage_divnorm_only", "generalization"])
    genDivnormSvd = agg(["stage_divnorm_svd_W", "generalization"])
    pHost = agg(["host_ceiling", "pearson_vs_Strue"])
    genHost = agg(["host_ceiling", "generalization"])
    pWcounts = agg(["anti_cheat", "pearson_W_vs_rawcounts"])
    localizations = agg(["localization", "verdict"])

    # consensus localization (all seeds same -> that; else MIXED)
    consensus = localizations[0] if len(set(localizations)) == 1 else "MIXED:" + ",".join(localizations)

    summary = {
        "seeds": seeds,
        "backend": backend,
        "localization_consensus": consensus,
        "localization_per_seed": localizations,
        "brain_based_note": ("the learned W is the project's spiking-Hebbian recurrent (LearnedAssocGraph). "
                             "STAGE W reads W directly; the fix (PPMI/divisive-normalization + low-dim SVD) "
                             "runs on the BRAIN-LEARNED W (anti-cheat: W distinct from raw counts). The host "
                             "PPMI+SVD on RAW counts is the labelled CEILING ONLY."),
        "per_stage_pearson_vs_Strue": {
            "STAGE_W_mean": float(np.mean(pW)),
            "STAGE_diffusion_mean": float(np.mean(pDiff)),
            "STAGE_divnorm_only_mean": float(np.mean(pDivnorm)),
            "STAGE_ppmi_svd_W_mean": float(np.mean(pPPMI)),
            "STAGE_divnorm_svd_W_mean": float(np.mean(pDivnormSvd)),
            "HOST_CEILING_mean": float(np.mean(pHost)),
        },
        "per_stage_generalization": {
            "STAGE_W_mean": float(np.mean(genW)),
            "STAGE_diffusion_mean": float(np.mean(genDiff)),
            "STAGE_divnorm_only_mean": float(np.mean(genDivnorm)),
            "STAGE_ppmi_svd_W_mean": float(np.mean(genPPMI)),
            "STAGE_divnorm_svd_W_mean": float(np.mean(genDivnormSvd)),
            "HOST_CEILING_mean": float(np.mean(genHost)),
            "random_baseline_mean": float(np.mean(agg(["random_baseline_generalization"]))),
            "chance": 1.0 / args.n_props,
        },
        "per_stage_pearson_vs_Strue_per_seed": {
            "STAGE_W": pW, "STAGE_diffusion": pDiff, "STAGE_divnorm_only": pDivnorm,
            "STAGE_ppmi_svd_W": pPPMI, "STAGE_divnorm_svd_W": pDivnormSvd, "HOST_CEILING": pHost,
        },
        "anti_cheat": {
            "pearson_W_vs_rawcounts_mean": float(np.mean(pWcounts)),
            "W_distinct_from_counts_all_seeds": all(
                per_seed[str(s)]["anti_cheat"]["W_is_distinct_from_counts"] for s in seeds),
        },
        "elapsed_total_s": time.time() - t0,
    }

    print(f"\n{'='*78}", flush=True)
    print(f"  DIAGNOSIS SUMMARY", flush=True)
    print(f"{'='*78}", flush=True)
    print(f"  Pearson(sim, S_true) per stage (mean over seeds {seeds}):", flush=True)
    print(f"    STAGE W (learned weights, no read-out) : {np.mean(pW):+.3f}", flush=True)
    print(f"    STAGE diffusion (current read-out)     : {np.mean(pDiff):+.3f}   <- the collapse",
          flush=True)
    print(f"    STAGE divnorm-only (W, brain-based)    : {np.mean(pDivnorm):+.3f}", flush=True)
    print(f"    STAGE PPMI+SVD (on LEARNED W)          : {np.mean(pPPMI):+.3f}", flush=True)
    print(f"    STAGE divnorm+SVD (on LEARNED W)       : {np.mean(pDivnormSvd):+.3f}", flush=True)
    print(f"    HOST CEILING (PPMI+SVD on RAW counts)  : {np.mean(pHost):+.3f}   (labelled target)",
          flush=True)
    print(f"  Generalization per stage (chance {1.0/args.n_props:.3f}):", flush=True)
    print(f"    STAGE W {np.mean(genW):.3f} | diffusion {np.mean(genDiff):.3f} | "
          f"divnorm {np.mean(genDivnorm):.3f} | PPMI+SVD(W) {np.mean(genPPMI):.3f} | "
          f"divnorm+SVD(W) {np.mean(genDivnormSvd):.3f} || host {np.mean(genHost):.3f}", flush=True)
    print(f"  ANTI-CHEAT: Pearson(W, raw_counts) mean {np.mean(pWcounts):+.3f} "
          f"(W distinct from counts: {summary['anti_cheat']['W_distinct_from_counts_all_seeds']})",
          flush=True)
    print(f"\n  LOCALIZATION CONSENSUS: {consensus}", flush=True)
    print(f"  Total elapsed: {summary['elapsed_total_s']:.1f}s", flush=True)
    print(f"{'='*78}\n", flush=True)

    out_data = {"summary": summary, "per_seed": per_seed}
    if args.out is None:
        raw_dir = os.path.join(_REPO, "research", "findings", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        args.out = os.path.join(raw_dir, f"_lge_diagnose_seed{seeds[0]}.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out_data, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)
    return out_data


if __name__ == "__main__":
    main()
