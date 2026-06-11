"""Dual / complementary-learning-systems (CLS) ARCHITECTURE PROOF — the cheap-first
load-bearing falsification that gates the dual-CLS build.

CONTEXT (the Option-B pivot):
  The "whiten the brain's similar codes in place" path (Option B direct) was FALSIFIED
  (2026-06-11-option-B-whitening-derisk-NEGATIVE.md): the ideal whitening cannot
  co-satisfy decorrelation + reproducibility + composition on the brain's denoise64 codes.
  The pivot is the DUAL / CLS architecture (docs/plans/2026-06-11-dual-CLS-architecture-design.md):
    - a "CORTEX" representation with GRADED-similar codes (similar concepts -> similar codes
      -> generalization), AND
    - a linked "HIPPOCAMPAL" DECORRELATED sparse expansion (between-cos ~ 0.05) the FHRR
      binder reads (binding validated),
    - coupled by an ENCODE path (cortex -> decorrelated, DG-style pattern separation) and a
      RETRIEVE path (decorrelated -> cortex, CA1 link).
  ~80% of the plumbing is built/validated; the genuinely-new piece is a learned graded-
  similarity embedding. BEFORE committing to that (months-scale) build, this probe must
  prove the ARCHITECTURE works on a SYNTHETIC graded codebook.

THE LOAD-BEARING RISK (the inverse of the binding problem):
  Decorrelation is similarity-REMOVING by design (pattern separation makes similar things
  distinct). So the encode->decorrelate->retrieve ROUND-TRIP may DESTROY the very graded
  similarity that generalization needs. Probe C's Pearson(S, S') is THE load-bearing number.

THREE PROBES (multi-seed 42/43/44; synthetic graded codebook FIRST):

  PROBE A -- GENERALIZATION (does graded similarity enable inference?)
    Synthetic graded codebook: K clusters of M concepts; within-cluster cosine HIGH,
    between-cluster LOW (a "category factor + concept residual" generator). Train a
    relation/property read-out on a SUBSET of each cluster; test inference on a HELD-OUT
    cluster-neighbour never trained in that relation (the cat~dog analogue) via a
    similarity-weighted nearest-trained-neighbour vote.
      GATE A1: held-out inference >> chance on graded codes (>= 0.7, chance = 1/n_props).
      GATE A2 (DECISIVE CONTRAST): the IDENTICAL test on the project's ORTHOGONAL sparse
               codes (generate_sparse_patterns, between-cos ~ 0.05) collapses to chance.
      GATE A3 (HEADLINE ANTI-CHEAT): permuted-similarity control -- shuffle which concepts
               are "similar" (decouple cluster label from code) -> generalization collapses.

  PROBE B -- BINDING preserved (reuse the positive control verbatim)
    Import cortex_sparse_attractor_poscontrol_probe.run_seed; confirm argmax/Hopfield parity
    ~ 1.000 on the decorrelated sparse codes (this half is validated; re-confirm it composes).
      GATE B: PASS = the existing 1.000 (Gate A + Gate B + noise-cue anti-cheat of that probe).

  PROBE C -- the ROUND-TRIP (the deepest, most-novel gate)
    graded cortex code -> ENCODE to decorrelated sparse expansion (DG-style random-projection
    + top-k sparsifier) -> bind (store a fact) -> retrieve -> DECODE back toward cortex (a
    learned linear read-out, the CA1->cortex link analogue). MEASURE whether GRADED SIMILARITY
    SURVIVES: Pearson correlation between original graded cosine matrix S and round-tripped S'.
      GATE C1: round-trip concept identity correct (binding round-trips).
      GATE C2 (LOAD-BEARING): Pearson(S, S') high (>= 0.7) at an operating point where binding
               also works. Sweep expansion sparsity / codec capacity to find where (if anywhere)
               similarity survives AND binding works. Permuted-S baseline (~0 Pearson) makes a
               high true Pearson meaningful, not an artifact.
      C2 SHARPENS rather than gates binary: PASS -> fast bidirectional codec; FAIL at every
      binding-viable point -> encode-fast/consolidate-slow link (the biological default).

DECISION (stated explicitly at end):
  GO if A1 (graded generalizes) AND A2 (orthogonal collapses) AND A3 (permuted collapses)
     AND B (binding ~ 1.000) AND C1 (round-trip identity) AND C2 (Pearson high at a
     binding-viable operating point).
  NEGATIVE/BOUNDARY/PARTIAL otherwise -- name the failing probe. If C2 destroys similarity
  at every binding-viable point -> recommend the slow-consolidation-link variant.

ANTI-CHEATS: orthogonal contrast (A2); permuted-similarity (A3, headline); round-trip Pearson
vs permuted-S baseline (C2); binding parity (B). Native-code conventions; multi-seed.

CPU ONLY; SIM_BACKEND=numpy; no sim/ edits; reuse-by-import only.
Run: python -m research.runners.dual_cls_architecture_proof_probe --seeds 42,43,44
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

os.environ.setdefault("SIM_BACKEND", "numpy")


# ===========================================================================
# Synthetic graded codebook (category factor + concept residual)
# ===========================================================================

def build_graded_codebook(n_clusters: int, per_cluster: int, dim: int,
                          seed: int, residual_frac: float = 0.55) -> tuple:
    """Build a synthetic GRADED-SIMILAR codebook.

    Each concept code = (shared CATEGORY factor) + (per-concept RESIDUAL).
    Concepts in the same cluster share a category factor -> they are systematically
    CLOSER (high within-cluster cosine); concepts in different clusters have
    independent category factors -> LOW between-cluster cosine. This is graded and
    SEMANTIC-by-construction (cluster = "category"). `residual_frac` controls the
    within-cluster spread: smaller residual -> tighter cluster (higher within-cosine).

    Returns:
      codes [N, dim] (unit-normalized, mean-removed per row -- native convention)
      labels [N] cluster id of each concept
      S [N, N] the graded cosine matrix
    """
    rng = np.random.RandomState(seed * 31 + 7)
    N = n_clusters * per_cluster
    # Category factors: one shared direction per cluster.
    cat = rng.randn(n_clusters, dim)
    cat = cat / (np.linalg.norm(cat, axis=1, keepdims=True) + 1e-12)
    codes = np.zeros((N, dim), dtype=np.float64)
    labels = np.zeros(N, dtype=int)
    idx = 0
    cat_amp = 1.0 - residual_frac
    for c in range(n_clusters):
        for _ in range(per_cluster):
            resid = rng.randn(dim)
            resid = resid / (np.linalg.norm(resid) + 1e-12)
            v = cat_amp * cat[c] + residual_frac * resid
            codes[idx] = v
            labels[idx] = c
            idx += 1
    # Native convention: mean-remove each row, then unit-normalize.
    codes = codes - codes.mean(axis=1, keepdims=True)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    S = codes @ codes.T
    return codes, labels, S


def codebook_similarity_stats(codes: np.ndarray, labels: np.ndarray) -> dict:
    """Within-cluster vs between-cluster cosine summary (the unit check for graded-ness)."""
    N = codes.shape[0]
    S = codes @ codes.T
    within, between = [], []
    for i in range(N):
        for j in range(i + 1, N):
            (within if labels[i] == labels[j] else between).append(float(S[i, j]))
    wm = float(np.mean(within)) if within else 0.0
    bm = float(np.mean(between)) if between else 0.0
    return {
        "within_cluster_cos_mean": wm,
        "within_cluster_cos_min": float(np.min(within)) if within else 0.0,
        "between_cluster_cos_mean": bm,
        "between_cluster_cos_max": float(np.max(between)) if between else 0.0,
        "graded_margin": wm - bm,
        "is_graded": (wm - bm) > 0.25,   # within systematically > between
    }


def load_orthogonal_codes(seed: int, N: int, n_pool: int = 2000,
                          pattern_size: int = 100) -> np.ndarray:
    """The project's ORTHOGONAL sparse codes (generate_sparse_patterns), native mean-removed.

    These are the DECISIVE CONTRAST for GATE A2: between-cos ~ 0.05 (equidistant by
    construction), so similarity-based inference MUST collapse to chance on them.
    """
    from research.runners.concept_pool_sparse_distributed import generate_sparse_patterns
    patterns = generate_sparse_patterns(N, n_pool, pattern_size, seed)
    codes = np.zeros((N, n_pool), dtype=np.float64)
    for i, pat in enumerate(patterns):
        codes[i, pat] = 1.0
    codes = codes - codes.mean(axis=1, keepdims=True)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    return codes


# ===========================================================================
# PROBE A -- GENERALIZATION (held-out-neighbour property inference)
# ===========================================================================

def assign_properties(n_clusters: int, per_cluster: int, n_props: int,
                      seed: int) -> np.ndarray:
    """Assign a property label to each concept, structured so that CLUSTER predicts
    property (semantic property inheritance: canids share property P).

    Property is a deterministic function of cluster (cluster c -> property c % n_props),
    so similarity-based inference SHOULD recover it: a held-out concept's nearest
    *trained* neighbours are its cluster-mates, which carry the cluster's property.
    Returns props [N] in [0, n_props).
    """
    N = n_clusters * per_cluster
    props = np.zeros(N, dtype=int)
    for c in range(n_clusters):
        p = c % n_props
        for m in range(per_cluster):
            props[c * per_cluster + m] = p
    return props


def similarity_vote_infer(codes: np.ndarray, train_mask: np.ndarray,
                          props: np.ndarray, query_idx: int,
                          k_neighbours: int = 3) -> int:
    """Infer the property of `query_idx` by a similarity-weighted vote over its
    k nearest TRAINED neighbours (the cortex 'read whatever code arrives' stand-in --
    NOT the exact-inverse algebra). Returns predicted property label.

    This is the simplest mechanism that CAN generalize: it only succeeds if similar
    concepts have similar codes (so the query's neighbours are its cluster-mates).
    """
    n_props = int(props.max()) + 1
    q = codes[query_idx]
    sims = codes @ q
    train_idx = np.where(train_mask)[0]
    train_idx = train_idx[train_idx != query_idx]
    if len(train_idx) == 0:
        return -1
    order = train_idx[np.argsort(-sims[train_idx])]
    kk = order[:k_neighbours]
    weights = np.maximum(sims[kk], 0.0) + 1e-9
    score = np.zeros(n_props, dtype=np.float64)
    for nb, w in zip(kk, weights):
        score[props[nb]] += w
    return int(np.argmax(score))


def run_generalization(codes: np.ndarray, labels: np.ndarray, props: np.ndarray,
                       n_clusters: int, per_cluster: int, seed: int,
                       k_neighbours: int = 3, n_heldout_per_cluster: int = 1,
                       n_splits: int = 20) -> dict:
    """Held-out-neighbour inference: in each cluster, HOLD OUT some concepts (never
    'trained' = excluded from the property table), train on the rest, then infer the
    held-out concepts' property from their nearest TRAINED neighbours.

    Accuracy >> chance proves graded similarity enables property inheritance. Averaged
    over `n_splits` random held-out partitions so the estimate is tight (the contrast
    against the orthogonal/permuted controls is then statistically decisive, not coarse).
    """
    rng = np.random.RandomState(seed * 13 + 5)
    N = codes.shape[0]
    n_props = int(props.max()) + 1
    n_correct = 0
    n_total = 0
    preds = []
    for _ in range(n_splits):
        train_mask = np.ones(N, dtype=bool)
        heldout = []
        for c in range(n_clusters):
            members = np.where(labels == c)[0]
            # Hold out concepts (require >=1 trained neighbour remains in cluster).
            nh = min(n_heldout_per_cluster, per_cluster - 1)
            ho = rng.choice(members, size=nh, replace=False)
            for h in ho:
                train_mask[h] = False
                heldout.append(int(h))
        for h in heldout:
            pred = similarity_vote_infer(codes, train_mask, props, h, k_neighbours)
            correct = int(pred == props[h])
            n_correct += correct
            n_total += 1
            if len(preds) < 16:   # keep a small sample for the JSON, not all splits
                preds.append({"query": h, "true_prop": int(props[h]),
                              "pred_prop": int(pred), "correct": bool(correct)})
    acc = n_correct / max(1, n_total)
    return {
        "n_heldout": n_total,
        "n_splits": n_splits,
        "n_correct": n_correct,
        "accuracy": acc,
        "chance": 1.0 / n_props,
        "ratio_vs_chance": acc / (1.0 / n_props) if n_props > 0 else 0.0,
        "preds": preds,
    }


def run_generalization_permuted(codes: np.ndarray, labels: np.ndarray, props: np.ndarray,
                                n_clusters: int, per_cluster: int, seed: int,
                                k_neighbours: int = 3,
                                n_heldout_per_cluster: int = 1) -> dict:
    """PERMUTED-SIMILARITY anti-cheat (A3, headline): break the code-similarity <->
    semantic-similarity correspondence by SHUFFLING the property labels across all
    concepts (decoupling property from cluster/code structure). Generalization MUST
    collapse to chance -- otherwise the 'generalization' is code overlap unrelated to
    meaning.
    """
    rng = np.random.RandomState(seed * 911 + 3)
    perm = rng.permutation(len(props))
    props_shuf = props[perm]
    return run_generalization(codes, labels, props_shuf, n_clusters, per_cluster,
                              seed, k_neighbours, n_heldout_per_cluster)


# ===========================================================================
# PROBE C -- the ENCODE -> DECORRELATE -> BIND -> RETRIEVE -> DECODE round-trip
# ===========================================================================

def make_dg_encoder(dim_in: int, n_pool: int, pattern_size: int,
                    seed: int) -> tuple:
    """DG-style pattern-separation encoder (the cheap numpy stand-in for the
    trisynaptic-loop DG): fixed random projection then top-k sparsification.

    A random projection scatters the (correlated) cortex code; the top-k WTA
    (k = pattern_size active of n_pool) is the DG PV-basket feedforward-inhibition
    analogue that makes the sparse expansion DECORRELATED. Returns (proj_matrix, encode_fn).

    encode_fn(code) -> binary {0,1} sparse expansion (native, mean-removed + unit-norm
    is applied by the caller where cosines are measured).
    """
    rng = np.random.RandomState(seed * 53 + 11)
    P = rng.randn(dim_in, n_pool) / np.sqrt(dim_in)

    def encode(code_row: np.ndarray) -> np.ndarray:
        proj = code_row @ P            # [n_pool]
        # top-k WTA (DG sparsification): keep the pattern_size strongest units.
        thresh_idx = np.argpartition(-proj, pattern_size)[:pattern_size]
        out = np.zeros(n_pool, dtype=np.float64)
        out[thresh_idx] = 1.0
        return out

    return P, encode


def encode_codebook(codes: np.ndarray, encode_fn, n_pool: int) -> np.ndarray:
    """Encode every cortex code -> its decorrelated sparse expansion (binary {0,1})."""
    N = codes.shape[0]
    expansion = np.zeros((N, n_pool), dtype=np.float64)
    for i in range(N):
        expansion[i] = encode_fn(codes[i])
    return expansion


def native_cos_matrix(rows: np.ndarray) -> np.ndarray:
    """Mean-remove + unit-normalize each row, then full cosine matrix (native convention)."""
    r = rows - rows.mean(axis=1, keepdims=True)
    r = r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-12)
    return r @ r.T


def expansion_between_cos(expansion: np.ndarray) -> dict:
    """Between-code cosine of the sparse expansion (must be ~0.05 for binding to work)."""
    M = native_cos_matrix(expansion)
    N = M.shape[0]
    off = [float(M[i, j]) for i in range(N) for j in range(i + 1, N)]
    return {"expansion_cos_mean": float(np.mean(off)) if off else 0.0,
            "expansion_cos_max": float(np.max(np.abs(off))) if off else 0.0}


def fit_decoder(expansion: np.ndarray, codes: np.ndarray, ridge: float = 1e-2) -> np.ndarray:
    """Learn the CA1->cortex DECODE link: a ridge-regularized linear map from the
    sparse expansion back to the graded cortex codes (W: n_pool -> dim).

    This is the project's consolidation-pathway analogue (a learned read-out from the
    decorrelated expansion to the cortex codebook). Trained on the codebook itself
    (the concepts the system knows).
    """
    # expansion: [N, n_pool], codes: [N, dim]. Solve W minimizing ||X W - Y||^2 + ridge||W||^2.
    X = expansion - expansion.mean(axis=1, keepdims=True)  # match native readout
    Y = codes
    n_pool = X.shape[1]
    A = X.T @ X + ridge * np.eye(n_pool)
    B = X.T @ Y
    W = np.linalg.solve(A, B)   # [n_pool, dim]
    return W


def hopfield_retrieve_all(expansion: np.ndarray, n_pool: int,
                          flip_frac: float, seed: int) -> tuple:
    """Bind/retrieve on the decorrelated expansion (Probe B's validated path): build a
    Hopfield attractor over the expansion, present a NOISED cue per concept, settle, and
    return both the recovered identities AND the settled (real-valued) states.

    The settled state is what the DECODE step reads (so the round-trip carries whatever
    the attractor reconstructs, not the clean stored code -- a faithful round-trip).
    Returns (recovered_idx [N], settled_states [N, n_pool]).
    """
    from research.runners.cortex_sparse_attractor_poscontrol_probe import (
        build_hopfield_weights, noisy_cue_sparse,
    )
    # Native mean-removed codes for the attractor basis.
    codes_native = expansion - expansion.mean(axis=1, keepdims=True)
    codes_native = codes_native / (np.linalg.norm(codes_native, axis=1, keepdims=True) + 1e-12)
    W = build_hopfield_weights(codes_native)
    rng = np.random.default_rng(seed * 7 + int(flip_frac * 1000) + 17)
    N = expansion.shape[0]
    recovered = np.zeros(N, dtype=int)
    settled = np.zeros((N, n_pool), dtype=np.float64)
    for i in range(N):
        cue = noisy_cue_sparse(codes_native[i], rng, flip_frac, n_pool)
        s = cue.copy().astype(np.float64)
        nn = np.linalg.norm(s)
        if nn > 1e-12:
            s = s / nn
        for _ in range(5):
            s_new = W @ s
            n2 = np.linalg.norm(s_new)
            if n2 < 1e-12:
                break
            s_new = s_new / n2
            if np.max(np.abs(s_new - s)) < 1e-8:
                break
            s = s_new
        recovered[i] = int(np.argmax(codes_native @ s))
        settled[i] = s
    return recovered, settled


def run_roundtrip(codes: np.ndarray, S_orig: np.ndarray, seed: int,
                  n_pool: int, pattern_size: int, flip_frac: float = 0.1,
                  ridge: float = 1e-2) -> dict:
    """The full round-trip for one operating point (n_pool, pattern_size).

    cortex codes -> DG encode (decorrelate) -> Hopfield bind/retrieve (noised cue) ->
    linear decode back toward cortex -> measure (C1) identity + (C2) Pearson(S, S').
    """
    dim = codes.shape[1]
    N = codes.shape[0]
    # --- ENCODE (decorrelating DG) ---
    P, encode_fn = make_dg_encoder(dim, n_pool, pattern_size, seed)
    expansion = encode_codebook(codes, encode_fn, n_pool)
    exp_stats = expansion_between_cos(expansion)

    # --- DECODE link (learn CA1->cortex from the codebook) ---
    W_dec = fit_decoder(expansion, codes, ridge=ridge)

    # --- BIND / RETRIEVE (noised cue, settle attractor on the expansion) ---
    recovered, settled = hopfield_retrieve_all(expansion, n_pool, flip_frac, seed)
    n_identity = int(np.sum(recovered == np.arange(N)))
    identity_acc = n_identity / N

    # --- DECODE the settled states back toward cortex ---
    settled_centered = settled - settled.mean(axis=1, keepdims=True)
    decoded = settled_centered @ W_dec     # [N, dim]
    S_round = native_cos_matrix(decoded)

    # --- C2: similarity survival (Pearson between off-diagonal of S_orig and S_round) ---
    iu = np.triu_indices(N, k=1)
    s_orig_off = S_orig[iu]
    s_round_off = S_round[iu]
    pearson = float(np.corrcoef(s_orig_off, s_round_off)[0, 1])

    return {
        "n_pool": n_pool,
        "pattern_size": pattern_size,
        "flip_frac": flip_frac,
        "expansion_cos_mean": exp_stats["expansion_cos_mean"],
        "expansion_cos_max": exp_stats["expansion_cos_max"],
        "binding_viable": exp_stats["expansion_cos_mean"] < 0.15,
        "identity_acc": identity_acc,
        "n_identity": n_identity,
        "N": N,
        "pearson_S_Sround": pearson,
    }


def run_roundtrip_permuted_baseline(codes: np.ndarray, S_orig: np.ndarray, seed: int,
                                    n_pool: int, pattern_size: int,
                                    flip_frac: float = 0.1, ridge: float = 1e-2) -> dict:
    """PERMUTED-S baseline for C2: permute the concept ROWS of the cortex codebook before
    the round-trip (so the decode target is a random concept). The Pearson(S_orig, S'_perm)
    should be ~0 -- proving a high TRUE Pearson is meaningful, not an artifact of the
    pipeline always producing similar-looking matrices.
    """
    rng = np.random.RandomState(seed * 617 + 29)
    perm = rng.permutation(codes.shape[0])
    codes_perm = codes[perm]
    # S_orig stays in the original order; the decoder is fit on the permuted codes, so
    # the round-tripped matrix has no reason to correlate with S_orig.
    r = run_roundtrip(codes_perm, S_orig, seed, n_pool, pattern_size, flip_frac, ridge)
    r["permuted_baseline"] = True
    return r


# ===========================================================================
# PROBE B -- binding (reuse the positive control verbatim)
# ===========================================================================

def run_binding_poscontrol(seed: int, V: int = 16, n_pool: int = 2000,
                           pattern_size: int = 100, n_trials: int = 120,
                           run_bridge: bool = False) -> dict:
    """Reuse cortex_sparse_attractor_poscontrol_probe.run_seed VERBATIM (Probe B).

    Returns the gate_a (attractor ~ argmax on decorrelated), gate_b (collapse on
    correlated), and noise-cue anti-cheat outcomes, plus the headline parity at p<=0.2.
    """
    from research.runners.cortex_sparse_attractor_poscontrol_probe import run_seed
    r = run_seed(seed=seed, V=V, n_pool=n_pool, pattern_size=pattern_size,
                 n_trials=n_trials, run_bridge=run_bridge)
    parity = r.get("parity_sparse_decorrelated", {})
    hop_parity = {str(fp): parity.get(fp, {}).get("hopfield_mf") for fp in (0.0, 0.1, 0.2)}
    return {
        "gate_a": r.get("gate_a"),
        "gate_b": r.get("gate_b"),
        "noise_cheat_ok": r.get("noise_cheat_ok"),
        "hopfield_parity_p_le_0p2": hop_parity,
        "binding_pass": bool(r.get("gate_a") and r.get("gate_b") and r.get("noise_cheat_ok")),
    }


# ===========================================================================
# Per-seed driver
# ===========================================================================

def run_seed_full(seed: int, n_clusters: int, per_cluster: int, dim: int,
                  n_props: int, k_neighbours: int, residual_frac: float,
                  roundtrip_sweep: list, flip_frac: float, ridge: float,
                  binding_V: int, binding_n_pool: int, binding_pattern_size: int,
                  run_bridge: bool, a1_bar: float, c2_bar: float) -> dict:
    print(f"\n{'='*64}", flush=True)
    print(f"  SEED {seed}", flush=True)
    print(f"{'='*64}", flush=True)
    N = n_clusters * per_cluster

    # ---------- Synthetic graded codebook ----------
    codes, labels, S = build_graded_codebook(n_clusters, per_cluster, dim, seed,
                                              residual_frac)
    grad_stats = codebook_similarity_stats(codes, labels)
    print(f"  [graded codebook] N={N} ({n_clusters}x{per_cluster}) dim={dim}", flush=True)
    print(f"    within-cluster cos={grad_stats['within_cluster_cos_mean']:.3f} "
          f"between-cluster cos={grad_stats['between_cluster_cos_mean']:.3f} "
          f"margin={grad_stats['graded_margin']:.3f} graded={grad_stats['is_graded']}",
          flush=True)

    # ---------- Orthogonal control codebook (the decisive contrast) ----------
    ortho = load_orthogonal_codes(seed, N, n_pool=2000, pattern_size=100)
    ortho_stats = codebook_similarity_stats(ortho, labels)
    print(f"    [orthogonal control] between-cos="
          f"{(ortho_stats['within_cluster_cos_mean']+ortho_stats['between_cluster_cos_mean'])/2:.4f} "
          f"(decorrelated by construction)", flush=True)

    # ---------- Property assignment (cluster-predicts-property) ----------
    props = assign_properties(n_clusters, per_cluster, n_props, seed)

    # ============ PROBE A ============
    print("  [PROBE A -- generalization]", flush=True)
    gen_graded = run_generalization(codes, labels, props, n_clusters, per_cluster,
                                    seed, k_neighbours)
    gen_ortho = run_generalization(ortho, labels, props, n_clusters, per_cluster,
                                   seed, k_neighbours)
    gen_perm = run_generalization_permuted(codes, labels, props, n_clusters,
                                           per_cluster, seed, k_neighbours)
    print(f"    graded    acc={gen_graded['accuracy']:.3f} "
          f"(chance={gen_graded['chance']:.3f}, {gen_graded['ratio_vs_chance']:.1f}x)",
          flush=True)
    print(f"    orthogonal acc={gen_ortho['accuracy']:.3f}  (MUST collapse to chance)",
          flush=True)
    print(f"    permuted-S acc={gen_perm['accuracy']:.3f}  (MUST collapse to chance)",
          flush=True)

    chance = gen_graded["chance"]
    a1 = gen_graded["accuracy"] >= a1_bar
    # A2: orthogonal collapses -- at or below ~1.5x chance.
    a2 = gen_ortho["accuracy"] <= 1.5 * chance
    # A3: permuted collapses -- at or below ~1.5x chance.
    a3 = gen_perm["accuracy"] <= 1.5 * chance
    print(f"    GATE A1 (graded >= {a1_bar}): {a1}", flush=True)
    print(f"    GATE A2 (orthogonal collapses): {a2}", flush=True)
    print(f"    GATE A3 (permuted collapses, HEADLINE): {a3}", flush=True)

    # ============ PROBE B ============
    print("  [PROBE B -- binding (reuse positive control)]", flush=True)
    binding = run_binding_poscontrol(seed, V=binding_V, n_pool=binding_n_pool,
                                     pattern_size=binding_pattern_size,
                                     run_bridge=run_bridge)
    print(f"    binding gate_a={binding['gate_a']} gate_b={binding['gate_b']} "
          f"noise_cheat_ok={binding['noise_cheat_ok']} -> PASS={binding['binding_pass']}",
          flush=True)

    # ============ PROBE C ============
    print("  [PROBE C -- encode->decorrelate->bind->retrieve->decode round-trip]", flush=True)
    sweep = []
    for (np_pool, pk) in roundtrip_sweep:
        rt = run_roundtrip(codes, S, seed, np_pool, pk, flip_frac, ridge)
        base = run_roundtrip_permuted_baseline(codes, S, seed, np_pool, pk, flip_frac, ridge)
        rt["pearson_permuted_baseline"] = base["pearson_S_Sround"]
        sweep.append(rt)
        print(f"    pool={np_pool:5d} K={pk:4d}  exp_cos={rt['expansion_cos_mean']:.3f} "
              f"bind_viable={rt['binding_viable']}  identity={rt['identity_acc']:.3f}  "
              f"Pearson(S,S')={rt['pearson_S_Sround']:+.3f}  "
              f"(permuted_baseline={base['pearson_S_Sround']:+.3f})", flush=True)

    # C1: round-trip identity at the best binding-viable operating point.
    viable = [r for r in sweep if r["binding_viable"]]
    # Best-case for C2: among binding-viable points, the highest Pearson where identity also works.
    c1_points = [r for r in viable if r["identity_acc"] >= 0.9]
    c1 = len(c1_points) > 0
    # C2: is there a binding-viable + identity-working operating point with high Pearson?
    best_pearson = max((r["pearson_S_Sround"] for r in c1_points), default=float("-inf"))
    best_point = max(c1_points, key=lambda r: r["pearson_S_Sround"], default=None)
    c2 = c1 and best_pearson >= c2_bar
    # Sharpening characterization: best Pearson among ALL binding-viable points (even if
    # identity < 0.9) -- informs the fallback recommendation.
    best_pearson_viable = max((r["pearson_S_Sround"] for r in viable), default=float("-inf"))
    print(f"    GATE C1 (binding-viable round-trip identity >= 0.9 exists): {c1}", flush=True)
    print(f"    GATE C2 (Pearson >= {c2_bar} at a binding-viable+identity point, LOAD-BEARING): "
          f"{c2}  best_pearson={best_pearson:+.3f}", flush=True)

    # ---------- per-seed verdict pieces ----------
    return {
        "seed": seed,
        "graded_stats": grad_stats,
        "orthogonal_stats": ortho_stats,
        "probe_a": {
            "graded": gen_graded,
            "orthogonal": gen_ortho,
            "permuted": gen_perm,
            "a1": bool(a1), "a2": bool(a2), "a3": bool(a3),
            "chance": chance,
        },
        "probe_b": binding,
        "probe_c": {
            "sweep": sweep,
            "c1": bool(c1),
            "c2": bool(c2),
            "best_pearson_at_binding_identity": (None if best_pearson == float("-inf")
                                                 else best_pearson),
            "best_point": (None if best_point is None
                           else {"n_pool": best_point["n_pool"],
                                 "pattern_size": best_point["pattern_size"],
                                 "identity_acc": best_point["identity_acc"],
                                 "pearson": best_point["pearson_S_Sround"],
                                 "expansion_cos_mean": best_point["expansion_cos_mean"]}),
            "best_pearson_any_binding_viable": (None if best_pearson_viable == float("-inf")
                                                else best_pearson_viable),
        },
        "gates": {"a1": bool(a1), "a2": bool(a2), "a3": bool(a3),
                  "b": bool(binding["binding_pass"]),
                  "c1": bool(c1), "c2": bool(c2)},
    }


def main():
    p = argparse.ArgumentParser(description="Dual-CLS architecture proof probe")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--n-clusters", type=int, default=8,
                   help="Number of semantic clusters (categories)")
    p.add_argument("--per-cluster", type=int, default=5,
                   help="Concepts per cluster")
    p.add_argument("--dim", type=int, default=256, help="Cortex code dimensionality")
    p.add_argument("--n-props", type=int, default=4,
                   help="Number of distinct properties (chance = 1/n_props)")
    p.add_argument("--k-neighbours", type=int, default=3)
    p.add_argument("--residual-frac", type=float, default=0.55,
                   help="Within-cluster spread (smaller = tighter clusters)")
    p.add_argument("--flip-frac", type=float, default=0.1,
                   help="Round-trip retrieval cue noise (active-bit flip fraction)")
    p.add_argument("--ridge", type=float, default=1e-2,
                   help="Ridge for the CA1->cortex linear decode")
    p.add_argument("--a1-bar", type=float, default=0.7)
    p.add_argument("--c2-bar", type=float, default=0.7)
    p.add_argument("--binding-V", type=int, default=16)
    p.add_argument("--binding-n-pool", type=int, default=2000)
    p.add_argument("--binding-pattern-size", type=int, default=100)
    p.add_argument("--run-bridge", action="store_true",
                   help="Run the on-bridge spiking attractor inside Probe B (slow; default off)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    t_start = time.time()

    # Round-trip sweep: vary expansion sparsity / capacity. K/pool ratio controls how
    # aggressively DG decorrelates; we sweep from a denser expansion (less decorrelation,
    # similarity may survive but binding may not) to the validated 100/2000 (binding works).
    roundtrip_sweep = [
        (2000, 400),   # K/N=0.20 -- denser, weaker decorrelation
        (2000, 200),   # K/N=0.10
        (2000, 100),   # K/N=0.05 -- the validated binding operating point
        (4000, 100),   # K/N=0.025 -- sparser
        (8000, 100),   # K/N=0.0125 -- very sparse, strongest decorrelation
    ]

    per_seed = {}
    for seed in seeds:
        per_seed[str(seed)] = run_seed_full(
            seed=seed, n_clusters=args.n_clusters, per_cluster=args.per_cluster,
            dim=args.dim, n_props=args.n_props, k_neighbours=args.k_neighbours,
            residual_frac=args.residual_frac, roundtrip_sweep=roundtrip_sweep,
            flip_frac=args.flip_frac, ridge=args.ridge,
            binding_V=args.binding_V, binding_n_pool=args.binding_n_pool,
            binding_pattern_size=args.binding_pattern_size,
            run_bridge=args.run_bridge, a1_bar=args.a1_bar, c2_bar=args.c2_bar,
        )

    # ---------- Overall verdict ----------
    def all_gate(g):
        return all(per_seed[str(s)]["gates"][g] for s in seeds)
    g_a1, g_a2, g_a3 = all_gate("a1"), all_gate("a2"), all_gate("a3")
    g_b = all_gate("b")
    g_c1, g_c2 = all_gate("c1"), all_gate("c2")

    if g_a1 and g_a2 and g_a3 and g_b and g_c1 and g_c2:
        verdict = "GO"
    elif g_a1 and g_a2 and g_a3 and g_b and g_c1 and not g_c2:
        # The designed sharpening: generalization + binding work, but the round-trip
        # destroys similarity at every binding-viable point -> encode-fast/consolidate-slow.
        verdict = "BOUNDARY_roundtrip_destroys_similarity"
    elif g_a1 and g_a2 and g_a3 and g_b:
        verdict = "PARTIAL_roundtrip_identity_fail"
    elif g_a1 and g_a2 and g_a3:
        verdict = "PARTIAL_binding_or_roundtrip"
    elif g_a1 and not (g_a2 and g_a3):
        verdict = "NEGATIVE_generalization_not_similarity_driven"
    else:
        verdict = "NEGATIVE_no_generalization"

    # Aggregate the load-bearing numbers across seeds.
    pearsons_best = [per_seed[str(s)]["probe_c"]["best_pearson_at_binding_identity"]
                     for s in seeds]
    pearsons_best = [x for x in pearsons_best if x is not None]
    pearsons_any = [per_seed[str(s)]["probe_c"]["best_pearson_any_binding_viable"]
                    for s in seeds]
    pearsons_any = [x for x in pearsons_any if x is not None]
    graded_acc = [per_seed[str(s)]["probe_a"]["graded"]["accuracy"] for s in seeds]
    ortho_acc = [per_seed[str(s)]["probe_a"]["orthogonal"]["accuracy"] for s in seeds]
    perm_acc = [per_seed[str(s)]["probe_a"]["permuted"]["accuracy"] for s in seeds]

    summary = {
        "verdict": verdict,
        "seeds": seeds,
        "gates_all_seeds": {"a1": g_a1, "a2": g_a2, "a3": g_a3,
                            "b": g_b, "c1": g_c1, "c2": g_c2},
        "probe_a_mean": {
            "graded_acc": float(np.mean(graded_acc)),
            "orthogonal_acc": float(np.mean(ortho_acc)),
            "permuted_acc": float(np.mean(perm_acc)),
            "chance": per_seed[str(seeds[0])]["probe_a"]["chance"],
        },
        "probe_c_load_bearing_pearson": {
            "best_at_binding_and_identity_per_seed": pearsons_best,
            "best_at_binding_and_identity_mean": (float(np.mean(pearsons_best))
                                                  if pearsons_best else None),
            "best_any_binding_viable_per_seed": pearsons_any,
            "best_any_binding_viable_mean": (float(np.mean(pearsons_any))
                                             if pearsons_any else None),
        },
        "elapsed_total_s": time.time() - t_start,
    }

    print(f"\n{'='*64}", flush=True)
    print(f"  OVERALL VERDICT: {verdict}", flush=True)
    print(f"  GATE A1 (graded generalizes, all seeds):  {g_a1}", flush=True)
    print(f"  GATE A2 (orthogonal collapses, all seeds): {g_a2}", flush=True)
    print(f"  GATE A3 (permuted collapses, HEADLINE):    {g_a3}", flush=True)
    print(f"  GATE B  (binding ~ 1.000, all seeds):      {g_b}", flush=True)
    print(f"  GATE C1 (round-trip identity, all seeds):  {g_c1}", flush=True)
    print(f"  GATE C2 (Pearson survives, LOAD-BEARING):  {g_c2}", flush=True)
    if summary["probe_c_load_bearing_pearson"]["best_at_binding_and_identity_mean"] is not None:
        print(f"  >>> Round-trip Pearson(S,S') at binding+identity point (mean): "
              f"{summary['probe_c_load_bearing_pearson']['best_at_binding_and_identity_mean']:+.3f}",
              flush=True)
    print(f"  Total elapsed: {summary['elapsed_total_s']:.1f}s", flush=True)
    print(f"{'='*64}\n", flush=True)

    out_data = {"summary": summary, "per_seed": per_seed}

    if args.out is None:
        raw_dir = os.path.join(_REPO, "research", "findings", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.out = os.path.join(raw_dir, f"_dual_cls_proof_{ts}.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out_data, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)
    return out_data


if __name__ == "__main__":
    main()
