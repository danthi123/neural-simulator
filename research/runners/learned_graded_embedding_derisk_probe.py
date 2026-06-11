"""LEARNED GRADED-SIMILARITY EMBEDDING DE-RISK — the load-bearing cheap-first falsification that
gates the months-scale dual/CLS learned-embedding build.

CONTEXT (the one unbuilt piece):
  The dual / complementary-learning-systems (CLS) architecture for a cortex that GENERALIZES
  ("a cat is like a dog" because related concepts get similar codes) is FULLY de-risked ON the
  real substrate -- but only with SYNTHETIC graded codes
  (2026-06-11-dual-CLS-architecture-proof-GO.md / -strong-encode-derisk-BOUNDARY.md /
  -cortex-channel-derisk-GO.md, commit 343c721d). The ONE unbuilt piece is a LEARNED graded-
  similarity embedding: codes where RELATED CONCEPTS GET SIMILAR CODES (cat near dog -- INCLUDING
  the second-order case where cat & dog never directly co-occur but share neighbours like
  "animal"/"pet"), LEARNED by a BRAIN-BASED rule from co-occurrence experience.

THE FALSIFICATION (the exact question, design doc §4):
  Can a BRAIN-BASED Hebbian co-occurrence rule produce graded codes that
    (1) RECOVER the intended semantic structure (incl. the second-order cat~dog closeness), AND
    (2) PASS the de-risked architecture (generalization + cortex-channel round-trip + strong-
        encode compatibility)?

  Two honest risks targeted head-on:
   (i)  biological Hebbian learning is classically WEAKER than backprop-on-big-data (the project
        hit this exact "~4 orders too small" gap in Phase 2.3a) -> the graded structure may be too
        COARSE for real generalization (realistic partial: BOUNDARY_weak_graded).
   (ii) the STRONG-REPRODUCIBLE-vs-GRADED tension: the project's strong/reproducible codes
        (generate_sparse_patterns) are ORTHOGONAL by construction; a graded code is by definition
        NOT orthogonal; these two properties have NEVER been co-satisfied. The dual design's intent
        is that they are DIFFERENT linked populations (graded cortex -> encode -> decorrelated DG);
        G4 tests this directly (does a graded cortex code still drive a clean reproducible+
        decorrelated DG encode -- it should, since the DG decorrelates ANY input).

THE MECHANISM (Option A, the recommended brain-based tier):
  Reuse research/runners/learned_assoc_graph.LearnedAssocGraph -- the ALREADY-BUILT + validated
  spiking-Hebbian concept co-occurrence learner (store_fact co-fires a fact's concept patterns;
  the PLASTIC pool->pool recurrent LEARNS their pairwise co-occurrence by Hebbian growth, NOT set;
  multi-seed-matched to a Python co-occurrence oracle; NO sim/ edits). We ADD a GRADED READ-OUT
  that turns the learned co-occurrence GRAPH into graded CODES: each concept's code = its row in
  the learned association matrix, optionally spread by graph diffusion (spreading activation, the
  recurrent's own dynamics -- brain-based). Two concepts that co-occur with OVERLAPPING neighbour
  sets end up with SIMILAR rows -> graded similarity, INCLUDING the second-order shared-neighbour
  case (cat~dog close via shared "animal", never directly co-occurring). This is the distributional
  principle (Harris 1954) realized as Hebbian spreading.

  BRAIN-BASED BAR (design doc §4.2): the learning rule is the project's Hebbian (via
  LearnedAssocGraph, which is). A host word2vec/SVD/PPMI embedding is run ONLY as a labelled
  CEILING (the best a tuned objective achieves on this toy co-occurrence), never as the deliverable.

THE EXPERIMENT (STEP 1-4; multi-seed 42/43/44; numpy first):
  STEP 1  A TOY but REAL co-occurrence corpus with KNOWN ground-truth graded structure S_true,
          INCLUDING the second-order shared-neighbour case. K clusters of M concepts; each cluster
          has a HUB ("animal"/"pet"); members co-occur with the hub (and a few cross-cluster bridge
          facts for realism) but NOT directly with their cluster-mates -> cat~dog are close ONLY
          via the shared hub (genuine second-order distributional structure). S_true = the cluster
          block matrix (within-cluster high, between low).
  STEP 2  LEARN the codes with the brain-based Hebbian rule (LearnedAssocGraph.store_fact over the
          corpus) + the graded read-out (learned association-row + diffusion).
  STEP 3  GATE 1 -- structure recovery: Pearson(S_learned, S_true) (incl. the cat~dog second-order
          pairs) vs a permuted baseline (~0); is_graded over the recovered cluster labels.
  STEP 4  GATE 2/3/4 -- the de-risked ARCHITECTURE gates on the LEARNED codes (swap learned ->
          synthetic): run_generalization >= 0.7 with A2 (orthogonal) + A3 (permuted-property)
          collapsing; the cortex-channel round-trip closing (Pearson high >> permuted, binding
          identity ~1.000); strong-encode compatibility (G4: drive the GRADED code through the
          spiking StrongDGEncoder -> repro 1.000 AND decorr ~0).

ANTI-CHEATS (all mandatory):
  - PERMUTED-CO-OCCURRENCE control (HEADLINE): scramble the training corpus's context structure
    (random facts, same concepts) -> re-learn -> the learned codes must NOT be graded (Pearson vs
    S_true ~0) AND generalization must COLLAPSE -> proving the graded structure came from the REAL
    co-occurrence statistics, not the architecture / read-out.
  - BEATS the random-Gaussian (text_embeddings.embed) AND the orthogonal (generate_sparse_patterns)
    baselines on the architecture gates.
  - held-out-neighbour generalization on the LEARNED structure (A1); orthogonal contrast (A2);
    permuted-property (A3); permuted-S round-trip baseline; native code conventions (mean-removed,
    unit-norm); multi-seed 42/43/44.

DECISION (stated explicitly at end):
  GO                    if the brain-based rule produces graded codes that recover S_true (Pearson
                        high, permuted ~0) AND pass the architecture gates (gen >= 0.7, controls
                        collapse, round-trip closes) AND G4 resolves (graded-cortex + decorrelated-
                        DG coexist), multi-seed. -> months-scale build JUSTIFIED end-to-end.
  BOUNDARY_weak_graded  if it learns the RIGHT structure (permuted control collapses, Pearson vs
                        S_true positive) but generalization is only MARGINAL (the biological-
                        learning-strength gap) -> characterize the gap + what closes it. (The honest
                        realistic outcome.)
  BOUNDARY_strong_vs_graded_conflict  if graded (G1) OR strong-reproducible-encode (G4) but NOT
                        both -> the cortex and DG codes must be different linked populations.
  NEGATIVE_not_cooccurrence_driven  if generalization passes on graded BUT the permuted-co-occurrence
                        control ALSO passes (graded structure is a read-out/architecture artifact).
  NEGATIVE_no_structure if it cannot recover graded structure at all. No banking.

CPU/numpy first (the Hebbian learn + harness gates run in seconds); tiny GPU only for the G4
spiking strong-DG encode if needed. NO sim/ edits; reuse-by-import only.

Run:
  # full numpy multi-seed (the Hebbian learn + recovery + generalization + cortex-channel numpy
  # round-trip; G4 spiking encode optional via --run-g4)
  SIM_BACKEND=numpy python -m research.runners.learned_graded_embedding_derisk_probe \
      --seeds 42,43,44 --run-g4 \
      --out research/findings/raw/_learned_graded_embedding_multiseed.json
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

# Reuse the de-risked architecture harnesses VERBATIM (codes swapped synthetic -> learned).
from research.runners.dual_cls_architecture_proof_probe import (  # noqa: E402
    codebook_similarity_stats,
    assign_properties,
    run_generalization,
    run_generalization_permuted,
    native_cos_matrix,
    load_orthogonal_codes,
)


# ===========================================================================
# STEP 1 -- a TOY but REAL co-occurrence corpus with KNOWN ground-truth graded structure
#           (including the second-order shared-neighbour cat~dog case).
# ===========================================================================
def build_toy_cooccurrence(n_clusters: int, per_cluster: int, seed: int,
                           hub_facts_per_member: int = 6,
                           bridge_facts: int = 8,
                           triplet_facts_per_cluster: int = 4) -> dict:
    """Build concepts with a KNOWN cluster structure + a co-occurrence corpus that reflects it,
    where cluster-mates are close ONLY via SHARED NEIGHBOURS (second-order), not direct co-occurrence.

    Design (controlled, measurable):
      - K clusters of M *member* concepts (cat, dog, wolf, ...).
      - Each cluster has ONE HUB concept (the shared neighbour: "animal" / "pet" / "vehicle" ...).
      - The corpus's facts:
          (a) HUB facts: (hub, member) pairs -- every member co-occurs with its cluster's hub
              MANY times. This is the ONLY thing tying cluster-mates together: cat-animal,
              dog-animal, ... -> cat & dog share the neighbour "animal" but NEVER appear in a fact
              together. THE SECOND-ORDER cat~dog case.
          (b) TRIPLET facts: (hub, member_i, member_j) -- a few facts where the hub appears with a
              PAIR of members (still mediated by the hub; members co-occur in a fact only via the
              hub's presence, mirroring "the animal ran" naming two animals in one scene). Kept
              small so the dominant signal is the shared-hub second-order structure.
          (c) BRIDGE facts: a few cross-cluster (member_a, member_b) facts (realism: the world is
              not perfectly blocked). Kept small so S_true (the cluster block) stays the dominant
              structure.

    Ground truth S_true: the cluster block matrix over the MEMBER concepts -- within-cluster
    relatedness 1.0, between-cluster 0.0 (the intended graded target the learned codes must recover).

    Returns dict with: concepts (all, hubs+members), members, hubs, labels (member cluster ids),
    member_index (member-name -> row in S_true), S_true [Nm, Nm], facts (list of concept tuples),
    second_order_pairs (the (cat,dog)-type within-cluster member pairs that NEVER co-occur directly).
    """
    rng = np.random.RandomState(seed * 101 + 3)
    hubs = [f"hub{c}" for c in range(n_clusters)]
    members = []
    labels = []
    for c in range(n_clusters):
        for m in range(per_cluster):
            members.append(f"c{c}_m{m}")
            labels.append(c)
    labels = np.asarray(labels, dtype=int)
    concepts = hubs + members
    member_index = {name: i for i, name in enumerate(members)}
    Nm = len(members)

    # Ground-truth relatedness: cluster block.
    S_true = np.zeros((Nm, Nm), dtype=np.float64)
    for i in range(Nm):
        for j in range(Nm):
            if labels[i] == labels[j]:
                S_true[i, j] = 1.0

    # Which (member_i, member_j) within-cluster pairs are SECOND-ORDER (no direct co-occurrence)?
    # By construction ALL within-cluster member pairs are second-order EXCEPT those that share a
    # bridge or triplet fact -- we track the pure ones explicitly after building the corpus.
    facts = []
    direct_cooccur = set()  # (min,max) member-name pairs that DO co-occur directly in some fact

    def _add_fact(tup):
        facts.append(tuple(tup))
        names = [t for t in tup]
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                x, y = names[a], names[b]
                if x in member_index and y in member_index:
                    direct_cooccur.add(tuple(sorted((x, y))))

    # (a) HUB facts -- the dominant second-order tie.
    for c in range(n_clusters):
        for m in range(per_cluster):
            for _ in range(hub_facts_per_member):
                _add_fact((hubs[c], f"c{c}_m{m}"))
    # (b) TRIPLET facts -- hub + a pair of members (members co-occur ONLY with the hub present).
    for c in range(n_clusters):
        cluster_members = [f"c{c}_m{m}" for m in range(per_cluster)]
        for _ in range(triplet_facts_per_cluster):
            if per_cluster >= 2:
                pair = rng.choice(per_cluster, size=2, replace=False)
                _add_fact((hubs[c], cluster_members[pair[0]], cluster_members[pair[1]]))
    # (c) BRIDGE facts -- cross-cluster realism (kept small).
    for _ in range(bridge_facts):
        ca, cb = rng.choice(n_clusters, size=2, replace=False)
        ma = rng.randint(per_cluster)
        mb = rng.randint(per_cluster)
        _add_fact((f"c{ca}_m{ma}", f"c{cb}_m{mb}"))

    # Identify the PURE second-order within-cluster pairs (cat~dog with NO direct co-occurrence).
    second_order_pairs = []
    for c in range(n_clusters):
        cm = [f"c{c}_m{m}" for m in range(per_cluster)]
        for a in range(per_cluster):
            for b in range(a + 1, per_cluster):
                key = tuple(sorted((cm[a], cm[b])))
                if key not in direct_cooccur:
                    second_order_pairs.append((member_index[cm[a]], member_index[cm[b]]))

    rng.shuffle(facts)
    return {
        "concepts": concepts, "members": members, "hubs": hubs,
        "labels": labels, "member_index": member_index, "S_true": S_true,
        "facts": facts, "second_order_pairs": second_order_pairs,
        "n_facts": len(facts),
    }


def permute_corpus(facts: list, concepts: list, seed: int) -> list:
    """PERMUTED-CO-OCCURRENCE control: keep the SAME concepts and the SAME fact SIZES, but
    re-draw each fact's members at RANDOM (scrambling the context structure). The learned codes
    must NOT be graded under this -> proving the graded structure came from the real statistics."""
    rng = np.random.RandomState(seed * 777 + 13)
    concepts = list(concepts)
    out = []
    for f in facts:
        size = len(f)
        chosen = rng.choice(len(concepts), size=size, replace=False)
        out.append(tuple(concepts[i] for i in chosen))
    return out


# ===========================================================================
# STEP 2 -- LEARN the codes with the brain-based Hebbian rule + the GRADED READ-OUT.
# ===========================================================================
def learn_assoc_matrix(concepts: list, facts: list, seed: int,
                       n_pool: int, pattern_size: int, cycles: int) -> "tuple[np.ndarray, dict]":
    """Reuse LearnedAssocGraph: Hebbian growth on the spiking pool->pool recurrent LEARNS the
    pairwise co-occurrence from the facts. Read the learned recurrent as the concept->concept
    association matrix W [Nc, Nc] (mean a->b recurrent weight between the concepts' sparse patterns).

    This is the BRAIN-BASED learned signal: W is the learned recurrent weights, NOT a Python
    co-occurrence count. Returns (W, info)."""
    from research.runners.learned_assoc_graph import LearnedAssocGraph
    from sim.backend import to_host
    lag = LearnedAssocGraph(concepts, seed=seed, n_pool=n_pool, pattern_size=pattern_size)
    for f in facts:
        lag.store_fact(list(f), cycles=cycles)
    # Dense learned recurrent over the pool, then average over each concept-pair's sparse patterns.
    M = to_host(lag.bridge.cp_connections)
    pb = lag.pool_base
    dense = np.asarray(M[pb][:, pb].todense())
    Nc = len(concepts)
    W = np.zeros((Nc, Nc), dtype=np.float64)
    pats = [np.asarray(p) for p in lag.patterns]
    for a in range(Nc):
        for b in range(Nc):
            if a == b:
                continue
            W[a, b] = float(dense[np.ix_(pats[a], pats[b])].mean())
    info = {
        "recurrent_mean": float(np.abs(np.asarray(M[pb][:, pb].data)).mean()),
        "recurrent_max": float(np.abs(np.asarray(M[pb][:, pb].data)).max()),
        "recurrent_nnz": int(len(np.asarray(M[pb][:, pb].data))),
        "n_neurons": int(lag.bridge.cp_membrane_potential_v.shape[0]),
    }
    return W, info


def graded_readout(W: np.ndarray, member_rows: np.ndarray,
                   diffusion_alpha: float = 0.5, diffusion_steps: int = 2) -> np.ndarray:
    """Turn the learned concept-concept association matrix W into GRADED codes.

    Brain-based graded read-out = SPREADING ACTIVATION on the learned recurrent: each concept's
    code is its association-PROFILE (its row in W), optionally SPREAD by the recurrent's own
    diffusion so that two concepts with OVERLAPPING neighbour sets get SIMILAR codes -- the
    second-order / shared-neighbour similarity (cat~dog via shared "animal"). Diffusion:
        Wd = (1-a) W + a (W @ Wn)   (Wn = row-normalized W), iterated diffusion_steps times,
    i.e. a concept's code accumulates not just its direct neighbours but its neighbours-of-
    neighbours (the distributional second-order signal). The CODE for the MEMBER concepts is then
    their rows of the diffused matrix, restricted to MEMBER columns (the cortex code space).

    Returns codes [Nm, dim] (native mean-removed + unit-norm), dim = number of member concepts.
    """
    Nc = W.shape[0]
    # Symmetrize (co-occurrence is symmetric) + zero the diagonal.
    Ws = 0.5 * (W + W.T)
    np.fill_diagonal(Ws, 0.0)
    # Row-normalized transition matrix for diffusion (spreading activation).
    rs = Ws.sum(axis=1, keepdims=True)
    Wn = Ws / (rs + 1e-12)
    Wd = Ws.copy()
    cur = Ws.copy()
    for _ in range(max(0, diffusion_steps)):
        cur = (1.0 - diffusion_alpha) * cur + diffusion_alpha * (cur @ Wn)
        Wd = cur
    # The code for each MEMBER concept = its diffused association profile over the MEMBER columns
    # (the shared hubs are the *mediators*; the cortex code is the member-to-member graded structure).
    codes = Wd[np.ix_(member_rows, member_rows)].astype(np.float64)
    codes = codes - codes.mean(axis=1, keepdims=True)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    return codes


# ===========================================================================
# STEP 3 -- structure recovery (G1).
# ===========================================================================
def structure_recovery(codes: np.ndarray, S_true: np.ndarray,
                       second_order_pairs: list, seed: int) -> dict:
    """Pearson(off-diag of learned cosine matrix, off-diag of S_true) -- does the learned code
    RECOVER the intended graded structure? Plus a PERMUTED baseline (shuffle code rows -> ~0) and
    a SECOND-ORDER-only Pearson (restricted to the cat~dog pairs that never directly co-occur)."""
    Nm = codes.shape[0]
    S_learned = codes @ codes.T
    iu = np.triu_indices(Nm, k=1)
    s_learn = S_learned[iu]
    s_true = S_true[iu]
    pearson = float(np.corrcoef(s_learn, s_true)[0, 1]) if np.std(s_learn) > 1e-12 else 0.0

    # permuted baseline: shuffle code rows, re-measure.
    rng = np.random.RandomState(seed * 313 + 7)
    perm = rng.permutation(Nm)
    S_perm = (codes[perm]) @ (codes[perm]).T
    s_perm = S_perm[iu]
    # Pearson of permuted-codes' similarity vs the (unpermuted) S_true.
    pearson_perm = float(np.corrcoef(s_perm, s_true)[0, 1]) if np.std(s_perm) > 1e-12 else 0.0

    # Second-order-only: the within-cluster cat~dog pairs that have NO direct co-occurrence.
    # Their S_true is 1.0 (same cluster); we report the MEAN learned cosine on those pairs vs the
    # mean learned cosine on between-cluster pairs (must be systematically higher).
    so_cos = [float(S_learned[i, j]) for (i, j) in second_order_pairs]
    between = [float(S_learned[i, j]) for i in range(Nm) for j in range(i + 1, Nm)
               if S_true[i, j] < 0.5]
    so_mean = float(np.mean(so_cos)) if so_cos else 0.0
    between_mean = float(np.mean(between)) if between else 0.0

    return {
        "pearson_learned_vs_Strue": pearson,
        "pearson_permuted_vs_Strue": pearson_perm,
        "second_order_pairs_n": len(second_order_pairs),
        "second_order_cos_mean": so_mean,
        "between_cluster_cos_mean": between_mean,
        "second_order_margin": so_mean - between_mean,
        "second_order_recovered": (so_mean - between_mean) > 0.10,
    }


# ===========================================================================
# Baselines for the BEATS-baseline anti-cheat (random-Gaussian + orthogonal).
# ===========================================================================
def random_gaussian_codes(Nm: int, dim: int, seed: int) -> np.ndarray:
    """The project's text_embeddings.embed placeholder = random near-orthogonal Gaussian codes
    (the current non-graded default). The learned codes must BEAT this on generalization."""
    rng = np.random.RandomState(seed * 271 + 5)
    codes = rng.randn(Nm, dim)
    codes = codes - codes.mean(axis=1, keepdims=True)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    return codes


# ===========================================================================
# Host CEILING (labelled, NOT the deliverable) -- PPMI + truncated SVD over the SAME corpus.
# ===========================================================================
def host_ceiling_codes(concepts: list, facts: list, member_rows: np.ndarray,
                       dim: int, seed: int) -> np.ndarray:
    """A host distributional embedding (PPMI co-occurrence + truncated SVD) over the SAME toy
    corpus -- the CEILING the brain-based Hebbian rule is compared against (design doc §4.2: a host
    embedding is allowed ONLY as a labelled ceiling). NOT a deliverable; reported as 'host_ceiling'.
    """
    Nc = len(concepts)
    idx = {c: i for i, c in enumerate(concepts)}
    C = np.zeros((Nc, Nc), dtype=np.float64)
    for f in facts:
        ids = [idx[c] for c in f if c in idx]
        for a in ids:
            for b in ids:
                if a != b:
                    C[a, b] += 1.0
    # PPMI.
    total = C.sum()
    if total <= 0:
        return random_gaussian_codes(len(member_rows), dim, seed)
    row = C.sum(axis=1, keepdims=True)
    col = C.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((C * total) / (row * col + 1e-12) + 1e-12)
    ppmi = np.maximum(pmi, 0.0)
    # Truncated SVD -> dense embedding.
    U, Sv, Vt = np.linalg.svd(ppmi, full_matrices=False)
    d = min(dim, U.shape[1])
    emb = U[:, :d] * Sv[:d]
    codes = emb[member_rows]
    codes = codes - codes.mean(axis=1, keepdims=True)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    # pad to dim if needed (native conventions preserved).
    if codes.shape[1] < dim:
        pad = np.zeros((codes.shape[0], dim - codes.shape[1]))
        codes = np.concatenate([codes, pad], axis=1)
    return codes


# ===========================================================================
# STEP 4 -- the de-risked ARCHITECTURE gates on the LEARNED codes.
# ===========================================================================
def architecture_generalization(codes: np.ndarray, labels: np.ndarray, props: np.ndarray,
                                 n_clusters: int, per_cluster: int, seed: int,
                                 k_neighbours: int, a1_bar: float) -> dict:
    """GATE 2 (generalization) on the LEARNED codes: held-out-neighbour property inference (A1) +
    the orthogonal contrast (A2) + the permuted-property control (A3). Reuses run_generalization
    VERBATIM from the architecture-proof harness."""
    Nm = codes.shape[0]
    gen_graded = run_generalization(codes, labels, props, n_clusters, per_cluster, seed, k_neighbours)
    ortho = load_orthogonal_codes(seed, Nm)
    gen_ortho = run_generalization(ortho, labels, props, n_clusters, per_cluster, seed, k_neighbours)
    gen_perm = run_generalization_permuted(codes, labels, props, n_clusters, per_cluster, seed,
                                           k_neighbours)
    chance = gen_graded["chance"]
    return {
        "graded": gen_graded, "orthogonal": gen_ortho, "permuted": gen_perm, "chance": chance,
        "a1": bool(gen_graded["accuracy"] >= a1_bar),
        "a2": bool(gen_ortho["accuracy"] <= 1.5 * chance),
        "a3": bool(gen_perm["accuracy"] <= 1.5 * chance),
    }


def cortex_channel_gate(codes: np.ndarray, seed: int, n_clusters: int, per_cluster: int,
                        c2_bar: float, flip_frac: float = 0.1,
                        n_dg: int = 2000, ensemble_size: int = 100) -> dict:
    """GATE 3 (architecture pass / cortex channel) on the LEARNED codes: assign per-concept sparse
    DG ensembles, run the (numpy) spiking-Hopfield recall to recover identity, reinstate the
    recall-identified concept's LEARNED graded cortex code, measure Pearson(S_orig, S'). Reuses
    cortex_channel_roundtrip + recall_identity_and_settle VERBATIM (the DG codes here are the
    deterministic generate_sparse_patterns ensembles -- the encode-side is already de-risked; the
    point is whether the GRADED cortex codes reinstate correctly and the round-trip closes).

    This is the numpy cortex-channel test (the encode-side spiking strong-DG is G4 below). The DG
    code is the binary deterministic ensemble matrix (reproducible by construction)."""
    from research.runners.dual_cls_cortex_channel_derisk_probe import (
        cortex_channel_roundtrip, cortex_channel_permuted_baseline, recall_identity_and_settle,
    )
    from research.runners.dual_cls_strong_encode_derisk_probe import assign_sparse_dg_ensembles
    Nm = codes.shape[0]
    S_orig = codes @ codes.T
    _, binary_dg = assign_sparse_dg_ensembles(Nm, n_dg, ensemble_size, seed)
    dg_codes = binary_dg.astype(np.float64)
    recovered, _, identity_acc = recall_identity_and_settle(dg_codes, flip_frac, seed, n_dg)
    pearson, _, _ = cortex_channel_roundtrip(codes, S_orig, recovered)
    perm = cortex_channel_permuted_baseline(codes, S_orig, recovered, seed)
    c2_ok = (pearson >= c2_bar) and (pearson > perm + 0.3)
    return {
        "binding_identity_acc": identity_acc,
        "cortex_channel_pearson": pearson,
        "cortex_channel_permuted": perm,
        "cortex_roundtrip_closes": bool(c2_ok),
    }


def strong_encode_g4(codes: np.ndarray, seed: int, args) -> dict:
    """GATE 4 (strong-encode compatibility / the strong-vs-graded tension, design doc §5.2):
    drive the GRADED cortex codes' sparse DG ensembles through the REAL spiking StrongDGEncoder at
    the validated operating point (drive>=800 pA, k=40) and check the spiking DG read is repro
    1.000 AND decorr ~0 -- confirming a graded cortex code STILL drives a clean reproducible+
    decorrelated DG encode (i.e. graded-cortex + decorrelated-DG COEXIST as linked populations,
    resolving the tension: the DG decorrelates ANY input, independent of the cortex code's graded-
    ness).

    Reuses StrongDGEncoder VERBATIM. The per-concept DG ensembles are the standard deterministic
    assignment (concept cells); the cortex codes are the LEARNED graded codes. The encode does not
    READ the cortex code -- it drives the assigned ensemble -- so this tests exactly the claim that
    the strong stable DG encode is independent of (and thus compatible with) a graded cortex code."""
    from research.runners.dual_cls_strong_encode_derisk_probe import (
        StrongDGEncoder, assign_sparse_dg_ensembles, _cos, _mean_offdiag_cos,
    )
    Nm = codes.shape[0]
    ensembles, _ = assign_sparse_dg_ensembles(Nm, args.g4_n_dg, args.g4_ensemble_size, seed)
    enc = StrongDGEncoder(
        seed=seed, n_lang_input=args.g4_n_lang_input, n_dg=args.g4_n_dg,
        n_dg_pv_basket=args.g4_n_dg_pv_basket, n_ca3=args.g4_n_ca3, n_ca1=args.g4_n_ca1,
        n_ec=args.g4_n_ec, ca3_recurrent_density=0.30, ca3_recurrent_weight=2.0, verbose=True)
    dg_codes, spikes = enc.encode_codebook_dg(ensembles, args.g4_drive_pA, args.g4_window,
                                              args.g4_k, args.g4_reset_steps)
    between = _mean_offdiag_cos(dg_codes)
    sparsity = float(np.mean(dg_codes > 0))
    repro_rng = np.random.default_rng(seed + 777)
    repro_vals = []
    for _ in range(args.g4_n_repro_pairs):
        ci = int(repro_rng.integers(Nm))
        c1, _, _ = enc.rate_kwta_dg_read(ensembles[ci], args.g4_drive_pA, args.g4_window,
                                         args.g4_k, args.g4_reset_steps)
        c2, _, _ = enc.rate_kwta_dg_read(ensembles[ci], args.g4_drive_pA, args.g4_window,
                                         args.g4_k, args.g4_reset_steps)
        repro_vals.append(_cos(c1, c2))
    repro_mean = float(np.mean(repro_vals))
    decorr_ok = between <= args.g4_decorr_bar
    repro_ok = repro_mean >= args.g4_repro_bar
    return {
        "dg_between_cos_mean": between,
        "dg_repro_mean": repro_mean,
        "dg_sparsity": sparsity,
        "dg_total_spikes_mean": float(np.mean(spikes)),
        "decorrelated": bool(decorr_ok),
        "reproducible": bool(repro_ok),
        "g4_graded_cortex_decorr_dg_coexist": bool(decorr_ok and repro_ok),
        "n_neurons": int(enc.n_neurons),
        "n_synapses": int(enc.n_synapses),
        "build_seconds": float(enc.build_seconds),
    }


# ===========================================================================
# Per-seed driver
# ===========================================================================
def run_seed(seed: int, args) -> dict:
    print(f"\n{'='*72}", flush=True)
    print(f"  LEARNED GRADED EMBEDDING DE-RISK -- SEED {seed}", flush=True)
    print(f"{'='*72}", flush=True)

    nclu, pclu = args.n_clusters, args.per_cluster

    # ----- STEP 1: toy corpus with KNOWN ground-truth (second-order cat~dog) -----
    corpus = build_toy_cooccurrence(nclu, pclu, seed,
                                    hub_facts_per_member=args.hub_facts_per_member,
                                    bridge_facts=args.bridge_facts,
                                    triplet_facts_per_cluster=args.triplet_facts_per_cluster)
    concepts = corpus["concepts"]
    members = corpus["members"]
    labels = corpus["labels"]
    S_true = corpus["S_true"]
    member_rows = np.asarray([concepts.index(m) for m in members], dtype=int)
    Nm = len(members)
    print(f"  [corpus] {len(concepts)} concepts ({nclu} hubs + {Nm} members), "
          f"{corpus['n_facts']} facts; second-order pairs (no direct co-occur)="
          f"{len(corpus['second_order_pairs'])}/{Nm*(pclu-1)//2 if pclu>1 else 0}", flush=True)

    # ----- STEP 2: LEARN the codes (brain-based Hebbian) + graded read-out -----
    print("  [STEP 2 -- brain-based Hebbian co-occurrence learning (LearnedAssocGraph)]", flush=True)
    t_learn = time.time()
    W, learn_info = learn_assoc_matrix(concepts, corpus["facts"], seed,
                                       args.n_pool, args.pattern_size, args.store_cycles)
    learn_s = time.time() - t_learn
    print(f"    learned recurrent: mean={learn_info['recurrent_mean']:.3f} "
          f"max={learn_info['recurrent_max']:.3f} nnz={learn_info['recurrent_nnz']} "
          f"({learn_info['n_neurons']} neurons, {learn_s:.1f}s)", flush=True)
    learned_codes = graded_readout(W, member_rows, args.diffusion_alpha, args.diffusion_steps)

    # PERMUTED-CO-OCCURRENCE control (the HEADLINE anti-cheat): re-learn on a scrambled corpus.
    print("  [STEP 2b -- PERMUTED-CO-OCCURRENCE control (re-learn on scrambled corpus)]", flush=True)
    perm_facts = permute_corpus(corpus["facts"], concepts, seed)
    W_perm, _ = learn_assoc_matrix(concepts, perm_facts, seed,
                                   args.n_pool, args.pattern_size, args.store_cycles)
    permuted_codes = graded_readout(W_perm, member_rows, args.diffusion_alpha, args.diffusion_steps)

    # ----- STEP 3: structure recovery (G1) -----
    print("  [STEP 3 -- structure recovery (G1)]", flush=True)
    grad_stats = codebook_similarity_stats(learned_codes, labels)
    rec = structure_recovery(learned_codes, S_true, corpus["second_order_pairs"], seed)
    rec_perm = structure_recovery(permuted_codes, S_true, corpus["second_order_pairs"], seed)
    print(f"    learned: within-cos={grad_stats['within_cluster_cos_mean']:.3f} "
          f"between-cos={grad_stats['between_cluster_cos_mean']:.3f} "
          f"margin={grad_stats['graded_margin']:.3f} graded={grad_stats['is_graded']}", flush=True)
    print(f"    >>> Pearson(S_learned, S_true) = {rec['pearson_learned_vs_Strue']:+.3f}  "
          f"(permuted-codes baseline {rec['pearson_permuted_vs_Strue']:+.3f})", flush=True)
    print(f"    >>> SECOND-ORDER cat~dog: shared-neighbour cos={rec['second_order_cos_mean']:+.3f} "
          f"vs between-cluster cos={rec['between_cluster_cos_mean']:+.3f} "
          f"(margin {rec['second_order_margin']:+.3f}, recovered={rec['second_order_recovered']})",
          flush=True)
    print(f"    [PERMUTED-CO-OCCURRENCE control] Pearson(S_learned_perm, S_true) = "
          f"{rec_perm['pearson_learned_vs_Strue']:+.3f}  graded="
          f"{codebook_similarity_stats(permuted_codes, labels)['is_graded']} (must be ~0 / False)",
          flush=True)
    g1 = (rec["pearson_learned_vs_Strue"] >= args.g1_bar
          and grad_stats["is_graded"]
          and rec["second_order_recovered"])
    g5_permco = (abs(rec_perm["pearson_learned_vs_Strue"]) < args.g1_bar * 0.6
                 and not codebook_similarity_stats(permuted_codes, labels)["is_graded"])

    # ----- STEP 4: architecture gates on the LEARNED codes -----
    props = assign_properties(nclu, pclu, args.n_props, seed)
    print("  [STEP 4 -- architecture gates on the LEARNED codes]", flush=True)
    gen = architecture_generalization(learned_codes, labels, props, nclu, pclu, seed,
                                      args.k_neighbours, args.a1_bar)
    print(f"    GATE 2 generalization: graded acc={gen['graded']['accuracy']:.3f} "
          f"(chance={gen['chance']:.3f}, {gen['graded']['ratio_vs_chance']:.1f}x) "
          f"A1={gen['a1']}  orthogonal={gen['orthogonal']['accuracy']:.3f} A2={gen['a2']}  "
          f"permuted={gen['permuted']['accuracy']:.3f} A3={gen['a3']}", flush=True)

    # PERMUTED-CO-OCCURRENCE generalization (must also collapse).
    gen_permco = run_generalization(permuted_codes, labels, props, nclu, pclu, seed,
                                    args.k_neighbours)
    print(f"    [PERMUTED-CO-OCCURRENCE generalization] acc={gen_permco['accuracy']:.3f} "
          f"(must collapse to ~chance {gen['chance']:.3f})", flush=True)
    g5_permco = g5_permco and (gen_permco["accuracy"] <= 1.5 * gen["chance"])

    # BEATS-baseline anti-cheat: random-Gaussian + the host ceiling.
    rand_codes = random_gaussian_codes(Nm, learned_codes.shape[1], seed)
    gen_rand = run_generalization(rand_codes, labels, props, nclu, pclu, seed, args.k_neighbours)
    host_codes = host_ceiling_codes(concepts, corpus["facts"], member_rows,
                                    learned_codes.shape[1], seed)
    host_stats = codebook_similarity_stats(host_codes, labels)
    host_rec = structure_recovery(host_codes, S_true, corpus["second_order_pairs"], seed)
    gen_host = run_generalization(host_codes, labels, props, nclu, pclu, seed, args.k_neighbours)
    beats_random = gen["graded"]["accuracy"] > gen_rand["accuracy"] + 1e-9
    print(f"    [BEATS-baseline] learned={gen['graded']['accuracy']:.3f} > "
          f"random-Gaussian={gen_rand['accuracy']:.3f} : {beats_random}", flush=True)
    print(f"    [HOST CEILING (PPMI+SVD, labelled, NOT deliverable)] Pearson(S,S_true)="
          f"{host_rec['pearson_learned_vs_Strue']:+.3f} gen={gen_host['accuracy']:.3f} "
          f"graded={host_stats['is_graded']}", flush=True)

    # GATE 3 cortex-channel round-trip (numpy).
    cc = cortex_channel_gate(learned_codes, seed, nclu, pclu, args.c2_bar, args.flip_frac,
                             n_dg=args.cc_n_dg, ensemble_size=args.cc_ensemble_size)
    print(f"    GATE 3 cortex-channel: identity={cc['binding_identity_acc']:.3f} "
          f"Pearson={cc['cortex_channel_pearson']:+.3f} (permuted {cc['cortex_channel_permuted']:+.3f}) "
          f"closes={cc['cortex_roundtrip_closes']}", flush=True)

    # GATE 4 strong-encode compatibility (the strong-vs-graded tension) -- spiking, optional.
    g4 = None
    if args.run_g4:
        print("  [STEP 4b -- GATE 4 strong-encode compatibility (spiking StrongDGEncoder)]",
              flush=True)
        g4 = strong_encode_g4(learned_codes, seed, args)
        print(f"    GATE 4: DG between-cos={g4['dg_between_cos_mean']:+.3f} "
              f"(decorr={g4['decorrelated']}) repro={g4['dg_repro_mean']:.3f} "
              f"(repro_ok={g4['reproducible']}) -> graded-cortex+decorrelated-DG COEXIST="
              f"{g4['g4_graded_cortex_decorr_dg_coexist']}", flush=True)

    gates = {
        "g1_structure_recovered": bool(g1),
        "g2_a1_generalizes": bool(gen["a1"]),
        "g2_a2_orthogonal_collapses": bool(gen["a2"]),
        "g2_a3_permuted_property_collapses": bool(gen["a3"]),
        "g3_cortex_roundtrip_closes": bool(cc["cortex_roundtrip_closes"]),
        "g3_binding_identity_clean": bool(cc["binding_identity_acc"] >= args.binding_bar),
        "g5_permuted_cooccurrence_collapses": bool(g5_permco),
        "g5_beats_random_baseline": bool(beats_random),
    }
    if g4 is not None:
        gates["g4_strong_encode_compatible"] = bool(g4["g4_graded_cortex_decorr_dg_coexist"])
    print(f"\n  [SEED {seed} gates] {gates}", flush=True)

    return {
        "seed": seed,
        "corpus": {"n_concepts": len(concepts), "n_members": Nm, "n_facts": corpus["n_facts"],
                   "n_second_order_pairs": len(corpus["second_order_pairs"])},
        "learn_info": learn_info,
        "graded_stats": grad_stats,
        "structure_recovery": rec,
        "structure_recovery_permuted_cooccurrence": rec_perm,
        "generalization": gen,
        "generalization_permuted_cooccurrence": gen_permco,
        "generalization_random_baseline": gen_rand,
        "host_ceiling": {"structure_recovery": host_rec, "graded_stats": host_stats,
                         "generalization": gen_host},
        "cortex_channel": cc,
        "strong_encode_g4": g4,
        "gates": gates,
    }


def main():
    p = argparse.ArgumentParser(description="Learned graded-similarity embedding de-risk probe")
    p.add_argument("--seeds", default="42,43,44")
    # toy corpus structure
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
    # graded read-out (spreading activation / diffusion)
    p.add_argument("--diffusion-alpha", type=float, default=0.5)
    p.add_argument("--diffusion-steps", type=int, default=2)
    # gate bars
    p.add_argument("--g1-bar", type=float, default=0.5,
                   help="Pearson(S_learned, S_true) >= this for G1")
    p.add_argument("--a1-bar", type=float, default=0.7)
    p.add_argument("--c2-bar", type=float, default=0.7)
    p.add_argument("--binding-bar", type=float, default=0.9)
    p.add_argument("--k-neighbours", type=int, default=3)
    p.add_argument("--flip-frac", type=float, default=0.1)
    # cortex-channel DG (numpy)
    p.add_argument("--cc-n-dg", type=int, default=2000)
    p.add_argument("--cc-ensemble-size", type=int, default=100)
    # GATE 4 strong-encode (spiking) -- optional
    p.add_argument("--run-g4", action="store_true",
                   help="run the spiking StrongDGEncoder G4 strong-encode-compatibility gate")
    p.add_argument("--g4-n-lang-input", type=int, default=256)
    p.add_argument("--g4-n-ec", type=int, default=120)
    p.add_argument("--g4-n-dg", type=int, default=600)
    p.add_argument("--g4-n-dg-pv-basket", type=int, default=180)
    p.add_argument("--g4-n-ca3", type=int, default=300)
    p.add_argument("--g4-n-ca1", type=int, default=120)
    p.add_argument("--g4-ensemble-size", type=int, default=40)
    p.add_argument("--g4-drive-pA", type=float, default=800.0)
    p.add_argument("--g4-k", type=int, default=40)
    p.add_argument("--g4-window", type=int, default=150)
    p.add_argument("--g4-reset-steps", type=int, default=40)
    p.add_argument("--g4-n-repro-pairs", type=int, default=6)
    p.add_argument("--g4-decorr-bar", type=float, default=0.10)
    p.add_argument("--g4-repro-bar", type=float, default=0.90)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    backend = os.environ.get("SIM_BACKEND", "auto")
    t0 = time.time()
    print(f"[learned-graded-embedding de-risk] seeds={seeds} backend={backend} run_g4={args.run_g4}",
          flush=True)
    print(f"  toy: {args.n_clusters}x{args.per_cluster} concepts (+hubs); learner=LearnedAssocGraph "
          f"(n_pool={args.n_pool}, pattern_size={args.pattern_size}, cycles={args.store_cycles})",
          flush=True)
    print(f"  graded read-out: diffusion alpha={args.diffusion_alpha} steps={args.diffusion_steps}",
          flush=True)
    print(f"  bars: G1(Pearson>={args.g1_bar}) A1(gen>={args.a1_bar}) C2(round-trip>={args.c2_bar})",
          flush=True)

    per_seed = {}
    for s in seeds:
        per_seed[str(s)] = run_seed(s, args)

    # ---------- overall verdict ----------
    def all_gate(g):
        return all(per_seed[str(s)]["gates"].get(g, False) for s in seeds)

    g1 = all_gate("g1_structure_recovered")
    g2_a1 = all_gate("g2_a1_generalizes")
    g2_a2 = all_gate("g2_a2_orthogonal_collapses")
    g2_a3 = all_gate("g2_a3_permuted_property_collapses")
    g3 = all_gate("g3_cortex_roundtrip_closes")
    g3_id = all_gate("g3_binding_identity_clean")
    g5_permco = all_gate("g5_permuted_cooccurrence_collapses")
    g5_beats = all_gate("g5_beats_random_baseline")
    g4 = all_gate("g4_strong_encode_compatible") if args.run_g4 else None

    # aggregate load-bearing numbers
    def agg(path):
        out = []
        for s in seeds:
            d = per_seed[str(s)]
            for k in path:
                d = d[k]
            out.append(d)
        return out

    pearson_struct = agg(["structure_recovery", "pearson_learned_vs_Strue"])
    pearson_struct_perm = agg(["structure_recovery_permuted_cooccurrence", "pearson_learned_vs_Strue"])
    so_margin = agg(["structure_recovery", "second_order_margin"])
    gen_graded = [per_seed[str(s)]["generalization"]["graded"]["accuracy"] for s in seeds]
    gen_ortho = [per_seed[str(s)]["generalization"]["orthogonal"]["accuracy"] for s in seeds]
    gen_perm_prop = [per_seed[str(s)]["generalization"]["permuted"]["accuracy"] for s in seeds]
    gen_permco = [per_seed[str(s)]["generalization_permuted_cooccurrence"]["accuracy"] for s in seeds]
    gen_rand = [per_seed[str(s)]["generalization_random_baseline"]["accuracy"] for s in seeds]
    gen_host = [per_seed[str(s)]["host_ceiling"]["generalization"]["accuracy"] for s in seeds]
    pearson_host = agg(["host_ceiling", "structure_recovery", "pearson_learned_vs_Strue"])
    cortex_pearson = agg(["cortex_channel", "cortex_channel_pearson"])
    bind_id = agg(["cortex_channel", "binding_identity_acc"])
    chance = per_seed[str(seeds[0])]["generalization"]["chance"]

    # Decision logic (design doc §4.4).
    structure_real = g5_permco and (np.mean(pearson_struct) > 0.0) and g1
    if not g1 and not structure_real:
        verdict = "NEGATIVE_no_structure"
    elif structure_real and not g5_permco:
        # cannot happen given structure_real includes g5_permco, kept for clarity
        verdict = "NEGATIVE_not_cooccurrence_driven"
    elif g1 and g2_a1 and not g5_permco:
        # generalization passes on graded but permuted-co-occurrence ALSO passes -> artifact
        verdict = "NEGATIVE_not_cooccurrence_driven"
    elif (g1 and g2_a1 and g2_a2 and g2_a3 and g3 and g3_id and g5_permco and g5_beats
          and (g4 is True or g4 is None)):
        verdict = "GO"
    elif (g1 and g5_permco and g5_beats and not g2_a1
          and np.mean(gen_graded) > 1.2 * chance):
        # learns the RIGHT structure (permuted collapses, Pearson positive) but generalization
        # only MARGINAL (above chance, below the 0.7 bar) -> the biological-strength gap.
        verdict = "BOUNDARY_weak_graded"
    elif args.run_g4 and g1 and g2_a1 and not g4:
        verdict = "BOUNDARY_strong_vs_graded_conflict"
    elif g1 and g5_permco and not g2_a1:
        verdict = "BOUNDARY_weak_graded"
    else:
        verdict = "BOUNDARY_unspecified"

    summary = {
        "verdict": verdict,
        "seeds": seeds,
        "backend": backend,
        "run_g4": bool(args.run_g4),
        "brain_based_note": ("the learning rule is the project's spiking Hebbian (LearnedAssocGraph: "
                             "pool->pool recurrent grows by co-fire Hebbian growth); the graded read-out "
                             "is spreading activation on the learned recurrent (brain-based). The host "
                             "PPMI+SVD is a labelled CEILING ONLY, NOT the deliverable."),
        "gates_all_seeds": {
            "g1_structure_recovered": g1,
            "g2_a1_generalizes": g2_a1,
            "g2_a2_orthogonal_collapses": g2_a2,
            "g2_a3_permuted_property_collapses": g2_a3,
            "g3_cortex_roundtrip_closes": g3,
            "g3_binding_identity_clean": g3_id,
            "g5_permuted_cooccurrence_collapses": g5_permco,
            "g5_beats_random_baseline": g5_beats,
            "g4_strong_encode_compatible": g4,
        },
        "load_bearing": {
            "pearson_struct_recovery_per_seed": pearson_struct,
            "pearson_struct_recovery_mean": float(np.mean(pearson_struct)),
            "pearson_struct_permuted_cooccurrence_per_seed": pearson_struct_perm,
            "pearson_struct_permuted_cooccurrence_mean": float(np.mean(pearson_struct_perm)),
            "second_order_margin_per_seed": so_margin,
            "second_order_margin_mean": float(np.mean(so_margin)),
            "generalization_graded_per_seed": gen_graded,
            "generalization_graded_mean": float(np.mean(gen_graded)),
            "generalization_orthogonal_mean": float(np.mean(gen_ortho)),
            "generalization_permuted_property_mean": float(np.mean(gen_perm_prop)),
            "generalization_permuted_cooccurrence_per_seed": gen_permco,
            "generalization_permuted_cooccurrence_mean": float(np.mean(gen_permco)),
            "generalization_random_baseline_mean": float(np.mean(gen_rand)),
            "generalization_host_ceiling_per_seed": gen_host,
            "generalization_host_ceiling_mean": float(np.mean(gen_host)),
            "pearson_host_ceiling_mean": float(np.mean(pearson_host)),
            "generalization_chance": chance,
            "cortex_channel_pearson_per_seed": cortex_pearson,
            "cortex_channel_pearson_mean": float(np.mean(cortex_pearson)),
            "binding_identity_mean": float(np.mean(bind_id)),
            "brain_vs_host_generalization_gap": float(np.mean(gen_host) - np.mean(gen_graded)),
        },
        "elapsed_total_s": time.time() - t0,
    }
    if args.run_g4:
        g4_decorr = agg(["strong_encode_g4", "dg_between_cos_mean"])
        g4_repro = agg(["strong_encode_g4", "dg_repro_mean"])
        summary["load_bearing"]["g4_dg_between_cos_mean_per_seed"] = g4_decorr
        summary["load_bearing"]["g4_dg_repro_mean_per_seed"] = g4_repro

    print(f"\n{'='*72}", flush=True)
    print(f"  OVERALL VERDICT: {verdict}", flush=True)
    print(f"  G1 structure recovered (Pearson vs S_true), all seeds: {g1}  "
          f"(mean {np.mean(pearson_struct):+.3f}; permuted-co-occurrence "
          f"{np.mean(pearson_struct_perm):+.3f}; second-order margin {np.mean(so_margin):+.3f})",
          flush=True)
    print(f"  G2 A1 generalizes >= {args.a1_bar}, all seeds: {g2_a1}  "
          f"(graded {np.mean(gen_graded):.3f} vs chance {chance:.3f}; "
          f"orthogonal {np.mean(gen_ortho):.3f}; permuted-prop {np.mean(gen_perm_prop):.3f})",
          flush=True)
    print(f"  G2 A2 orthogonal collapses: {g2_a2}   G2 A3 permuted-property collapses: {g2_a3}",
          flush=True)
    print(f"  G3 cortex-channel round-trip closes: {g3}  (Pearson {np.mean(cortex_pearson):+.3f}, "
          f"binding identity {np.mean(bind_id):.3f})", flush=True)
    print(f"  G5 permuted-CO-OCCURRENCE collapses (HEADLINE): {g5_permco}  "
          f"(gen {np.mean(gen_permco):.3f} vs chance {chance:.3f})", flush=True)
    print(f"  G5 beats random baseline: {g5_beats}  "
          f"(learned {np.mean(gen_graded):.3f} > random {np.mean(gen_rand):.3f})", flush=True)
    if args.run_g4:
        print(f"  G4 strong-encode compatible (graded-cortex + decorrelated-DG coexist): {g4}",
              flush=True)
    print(f"  HOST CEILING (labelled, NOT deliverable): gen {np.mean(gen_host):.3f} "
          f"(Pearson {np.mean(pearson_host):+.3f}); brain-vs-host gap "
          f"{summary['load_bearing']['brain_vs_host_generalization_gap']:+.3f}", flush=True)
    print(f"  Total elapsed: {summary['elapsed_total_s']:.1f}s", flush=True)
    print(f"{'='*72}\n", flush=True)

    out_data = {"summary": summary, "per_seed": per_seed}
    if args.out is None:
        raw_dir = os.path.join(_REPO, "research", "findings", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.out = os.path.join(raw_dir, f"_learned_graded_embedding_{ts}.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out_data, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)
    return out_data


if __name__ == "__main__":
    main()
