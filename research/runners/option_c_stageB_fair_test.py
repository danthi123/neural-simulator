"""OPTION C fairer de-risk -- STAGE B (the brain-based fair test, GPU-gated behind a host PRE-GATE).

Design: docs/plans/2026-06-14-option-c-fairer-derisk-design.md.
Stage-A finding: research/findings/2026-06-14-option-c-paradigmatic-host-precheck-VIABLE.md (host PASS,
  Pearson +0.539 on the full 8x8 paradigmatic taxonomy at window=2 / ctx=5000 / svd=100 / alpha=0.75).
Predecessor (the prior brain-based attempt): research/runners/option_c_real_cooccurrence_derisk.py.

THE QUESTION: does the project's brain-based spiking-Hebbian learn (learn_W_homeostatic, Oja-bounded
pool<->pool Hebbian growth on a real Izhikevich bridge) + the validated brain-based divnorm read-out
LEARN graded PARADIGMATIC semantic structure (animals cluster, colors cluster, ...) from REAL
TinyStories co-occurrence -- measured against an INDEPENDENT a-priori category taxonomy?

THE TWO LOAD-BEARING DESIGN POINTS
==================================
(A) THE HOST PRE-GATE (Stage A) RUNS FIRST AND SHORT-CIRCUITS THE GPU.
    Before any bridge is built, compute the validated SECOND-ORDER host ceiling (a target x FULL-context
    PPMI+SVD = cosine of the context-profile rows -- the distributional-semantics standard for
    paradigmatic similarity) on the chosen (sub-)taxonomy, reusing build_target_context_counts +
    ppmi_svd_sim + score from option_c_paradigmatic_host_precheck VERBATIM at the validated operating
    point (window=2, context_vocab=5000, svd_dim=100, alpha=0.75). HOST GATE: Pearson(sim, S_true) >= 0.50.
    If the host FAILS -> emit NEGATIVE_data_too_syntagmatic and EXIT (zero GPU). Only if it PASSES does
    the brain-based run proceed.

(B) THE CRITICAL REFINEMENT -- the brain-based learn uses a CONTEXT-INCLUSIVE corpus (NOT the prior
    target-restricted one).
    The prior Option-C `build_real_cooccurrence` restricted co-occurrence to the 64 target words -> the
    learn only saw target<->target (FIRST-ORDER / syntagmatic) co-occurrence, so it could only learn
    syntagmatic structure. PARADIGMATIC similarity (cat~dog) is SECOND-ORDER: two targets are similar
    because they share CONTEXT neighbours (both near "little","run","pet"), NOT because they co-occur
    directly. So the brain-based learn MUST see the context words too -- exactly the hub-mediated
    shared-neighbour mechanism the validated synthetic `build_toy_cooccurrence` models (members tied to
    each other ONLY via a shared hub). `build_context_inclusive_cooccurrence` (the ONE new corpus
    function) streams TinyStories and builds facts that include BOTH the in-window TARGET words AND the
    in-window top-`n_context_hubs` high-frequency CONTEXT words. The context words play the hub role; the
    64 TARGETS are the `members` whose codes are read out and scored against the taxonomy `S_true`.

    *** This is a deliberate DEPARTURE from the design's "byte-identical to the prior probe" -- it is the
    load-bearing correction (the design's §(B) refinement). Flagged explicitly. ***

EVERYTHING ELSE is reused by import (the brain-based learn + read-out + the full battery + the verdict
shape). The S_true reference is the INDEPENDENT a-priori taxonomy (NEVER corpus-derived; the
`s_true_independent` assertion is the single most important correctness property).

BATTERY (on the context-inclusive corpus): G1 structure_recovery (Pearson vs the independent taxonomy +
2nd-order margin); G2 architecture_generalization (held-out-neighbour A1 + A2 orthogonal + A3
permuted-property collapse); HEADLINE permuted-co-occurrence (permute_corpus -> re-learn -> must collapse
via _g5_robust); beats random-Gaussian; W-distinct-from-counts. The FIRST-ORDER host is reported
ALONGSIDE the SECOND-ORDER host (the expected signature: first-order low < second-order >=0.5 -- the
anti-cheat that confirms we fixed the right thing).

VERDICTS (the host-pass precondition gates GO / BOUNDARY_weak_graded / NEGATIVE_no_structure):
  NEGATIVE_data_too_syntagmatic -- the host pre-gate FAILS (even the gold-standard 2nd-order measure
                                   can't recover the paradigmatic taxonomy). EXIT before GPU.
  GO                            -- host passes AND brain-based: G1 + graded + 2nd-margin; G2 A1 + A2/A3
                                   collapse; HEADLINE permuted collapses; beats random.
  BOUNDARY_weak_graded          -- host passes; brain learns the RIGHT (co-occurrence-driven) structure
                                   coarsely (permuted collapses + Pearson>0) but gen/2nd-margin marginal
                                   (the Mikulasch-Priesemann point-neuron strength gap).
  NEGATIVE_no_structure         -- host passes but the brain-based substrate fails G1 (a clean,
                                   biology-translatable mechanism negative).

NO sim/ edits; reuse-by-import only. CPU-numpy smoke first (12-word/4-category, small n_context_hubs,
first ~300KB of TinyStories) validates the host pre-gate + the context-inclusive corpus shape + the full
battery + the verdict plumbing. The decisive 64-word 3-seed run is GPU (controller-driven).

Run (the DECISIVE GPU run -- controller-driven, after the GPU frees):
  SIM_BACKEND=cupy python -u -m research.runners.option_c_stageB_fair_test \
      --seeds 42,43,44 --out research/findings/raw/_option_c_stageB_fair_multiseed.json

Run (the CPU-numpy plumbing SMOKE -- seconds-to-~2min):
  SIM_BACKEND=numpy python -u -m research.runners.option_c_stageB_fair_test \
      --smoke --seeds 42 --out research/findings/raw/_option_c_stageB_smoke.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# ----- Reuse VERBATIM by import (NO reimplementation) -----
# The brain-based spiking-Hebbian learn (Oja homeostatic ceiling) + the brain-based divnorm read-out:
from research.runners.learned_graded_embedding_homeostasis_probe import (  # noqa: E402
    learn_W_homeostatic,
)
from research.runners.learned_graded_embedding_divnorm_readout_probe import (  # noqa: E402
    divnorm_spreading_readout,
)
# The de-risk's structure-recovery + generalization battery + random baseline + permuted control:
from research.runners.learned_graded_embedding_derisk_probe import (  # noqa: E402
    permute_corpus,
    structure_recovery,
    architecture_generalization,
    random_gaussian_codes,
)
# The margin-based anti-cheat gate (NEVER the bare is_graded boolean):
from research.runners.multibridge_graded_derisk import _g5_robust  # noqa: E402
# learning-not-pass-through check helpers:
from research.runners.learned_graded_embedding_diagnose import (  # noqa: E402
    raw_count_matrix,
    offdiag_pearson,
)
# graded-ness unit check + property assignment + generalization:
from research.runners.dual_cls_architecture_proof_probe import (  # noqa: E402
    codebook_similarity_stats,
    assign_properties,
    run_generalization,
)
# The 8x8 taxonomy + flattener + the validated SECOND-ORDER host pre-check primitives.
from research.runners.option_c_real_cooccurrence_derisk import (  # noqa: E402
    TAXONOMY_8x8,
    TAXONOMY_SMOKE,
    taxonomy_to_vocab_categories,
)
from research.runners.option_c_paradigmatic_host_precheck import (  # noqa: E402
    build_target_context_counts,
    ppmi_svd_sim,
    score,
    STOPLIST,
)


# ===========================================================================
# STAGE A -- the SECOND-ORDER paradigmatic host PRE-GATE (CPU-numpy; GATES the GPU).
# Reuses build_target_context_counts + ppmi_svd_sim + score VERBATIM at the validated operating point.
# Also computes the OLD FIRST-ORDER (target x target / syntagmatic) host alongside, so the JSON carries
# the validating signature: first-order LOW (syntagmatic) < second-order HIGH (paradigmatic, >=0.50).
# ===========================================================================
def _first_order_host_pearson(tokens, vocab, labels, window):
    """The prior FIRST-ORDER (syntagmatic) host: a target x TARGET windowed co-occurrence -> PPMI+SVD ->
    cosine. Reported ONLY as the anti-cheat baseline (expected ~+0.13, well below the second-order host).
    Reuses build_target_context_counts with the CONTEXT VOCAB = the targets themselves (the defect we
    fixed), and ppmi_svd_sim/score VERBATIM."""
    M = build_target_context_counts(tokens, vocab, vocab, window)  # context vocab == targets (1st-order)
    sim = ppmi_svd_sim(M, svd_dim=100, alpha=0.75)
    pearson, margin, nn_same, _ = score(sim, labels)
    return {"pearson": pearson, "margin": margin, "nn_same": nn_same}


def host_pre_gate(tokens, vocab, labels, cat_names, args) -> dict:
    """STAGE A: compute the validated SECOND-ORDER paradigmatic host on the chosen taxonomy and decide
    whether the GPU brain-based run may proceed. Returns a dict including `host_passes`.

    Operating point (validated in the Stage-A finding, Pearson +0.539 on the full 8x8):
        window=2, context_vocab=5000, svd_dim=100, alpha=0.75.  HOST GATE: Pearson >= 0.50.
    """
    freq = Counter(tokens)
    ctx_words = [w for w, _ in freq.most_common(args.host_ctx_size + len(STOPLIST))
                 if w not in STOPLIST][:args.host_ctx_size]
    # also exclude the targets themselves from the context columns (a target's OWN occurrences are
    # first-order; the paradigmatic signal is the SHARED *other* context) -- matches the precheck intent.
    vocab_set = set(vocab)
    ctx_words = [w for w in ctx_words if w not in vocab_set]
    M = build_target_context_counts(tokens, vocab, ctx_words, args.host_window)
    sim = ppmi_svd_sim(M, svd_dim=args.host_svd_dim, alpha=args.host_alpha)
    pearson, margin, nn_same, nn = score(sim, labels)

    # per-category nearest-neighbour-same rate (the Option-2 subset selector, reported for the trail).
    per_cat = {}
    for ci, cn in enumerate(cat_names):
        members = [k for k in range(len(vocab)) if labels[k] == ci]
        per_cat[cn] = float(np.mean([labels[nn[k]] == ci for k in members])) if members else 0.0

    first_order = _first_order_host_pearson(tokens, vocab, labels, args.host_window)
    host_passes = bool(pearson >= args.host_gate_pearson)

    print(f"  [STAGE A -- SECOND-ORDER paradigmatic host PRE-GATE] "
          f"w={args.host_window} ctx={len(ctx_words)} svd={args.host_svd_dim} alpha={args.host_alpha}",
          flush=True)
    print(f"    >>> second-order host Pearson(sim,S_true) = {pearson:+.3f}  margin={margin:+.3f}  "
          f"nn-same={nn_same:.3f}", flush=True)
    print(f"    first-order (syntagmatic, target x target) host Pearson = {first_order['pearson']:+.3f}  "
          f"(expected LOW < second-order -- confirms the measure fix)", flush=True)
    fo, so = first_order["pearson"], pearson
    print(f"    validating signature first-order<{so_repr(so)} : "
          f"{'OK (fo %.3f < so %.3f)' % (fo, so) if fo < so else 'WARN (fo %.3f >= so %.3f)' % (fo, so)}",
          flush=True)
    pc = ", ".join(f"{cn}:{per_cat[cn]:.2f}" for cn in cat_names)
    print(f"    per-category nn-same: {pc}", flush=True)
    print(f"    >>> HOST GATE Pearson>={args.host_gate_pearson}: "
          f"{'PASS -> brain-based Stage B PROCEEDS' if host_passes else 'FAIL -> NEGATIVE_data_too_syntagmatic (EXIT, zero GPU)'}",
          flush=True)

    return {
        "host_passes": host_passes,
        "second_order_pearson": pearson,
        "second_order_margin": margin,
        "second_order_nn_same": nn_same,
        "first_order_pearson": first_order["pearson"],
        "first_order_margin": first_order["margin"],
        "first_order_nn_same": first_order["nn_same"],
        "first_order_below_second_order": bool(first_order["pearson"] < pearson),
        "per_category_nn_same": per_cat,
        "n_context_words": len(ctx_words),
        "operating_point": {"window": args.host_window, "context_vocab": args.host_ctx_size,
                            "svd_dim": args.host_svd_dim, "alpha": args.host_alpha,
                            "gate_pearson": args.host_gate_pearson},
    }


def so_repr(x):
    return f"{x:+.3f}"


# ===========================================================================
# (B) THE ONE NEW CORPUS FUNCTION: the CONTEXT-INCLUSIVE co-occurrence corpus + the INDEPENDENT S_true.
# Returns EXACTLY the keys `build_toy_cooccurrence` / `build_real_cooccurrence` return so the
# learn/read-out/gates are reused unchanged -- BUT the facts now include the CONTEXT-WORD HUBS so the
# learn can pick up the SECOND-ORDER (paradigmatic) structure (two targets near the same context hubs).
#   concepts = targets + context-hubs (targets FIRST, mirroring build_toy_cooccurrence's hubs+members
#              layout but inverted -- members(targets) first so member_rows == identity over [0..Nt)).
#   members  = the 64 TARGETS (whose codes are read out + scored against S_true).
#   hubs     = the context words (the shared neighbours that mediate cat~dog).
#   labels   = each TARGET's category id; S_true = the a-priori category block over the 64 targets ONLY.
#   member_index = target-name -> row in S_true (0..Nt-1).
# ===========================================================================
def build_context_inclusive_cooccurrence(corpus_path: str, vocab: list, categories: np.ndarray,
                                         window: int = 2, n_context_hubs: int = 500,
                                         repeat_cap: int = 40, seed: int = 42,
                                         max_bytes: int | None = None, freq_floor: int = 30,
                                         min_facts_per_category: int = 20,
                                         verbose: bool = True) -> dict:
    """Extract a CONTEXT-INCLUSIVE windowed co-occurrence corpus over REAL TinyStories text, and pair it
    with an INDEPENDENT a-priori category taxonomy (`categories`).

    THE REFINEMENT (design §(B)): unlike the prior `build_real_cooccurrence` (which restricted scenes to
    the targets, giving ONLY first-order target<->target co-occurrence), each window's scene here includes
    BOTH the in-window TARGET words AND the in-window top-`n_context_hubs` high-frequency CONTEXT words.
    The context words are the SHARED NEIGHBOUR HUBS -- two targets that share context hubs become
    SECOND-ORDER similar through the learn's recurrent (cat~dog via "little"/"run"/"pet"), exactly the
    `build_toy_cooccurrence` mechanism, now realized on real text.

    Algorithm:
      0. Pre-pass: stream the corpus, count token frequencies, pick the `n_context_hubs` most-frequent
         CONTEXT words (excluding a small stoplist AND the targets themselves) -> the hub set.
      1. Re-stream; per story tokenize with re.findall(r"[a-z]+", ...). A token is "kept" if it is a
         target OR a context-hub.
      2. Slide a symmetric +-`window`; each window's SET of distinct kept words = one scene -> one fact.
         A scene must contain >=2 kept words AND >=1 TARGET (a pure-context scene carries no target
         signal). Drop the rest.
      3. Per-pair repeat cap: emit a scene only while EVERY unordered pair it contains is still under
         `repeat_cap` (seeded; the same conservative cap as build_real_cooccurrence) so a few
         high-frequency hubs don't saturate the recurrent.
      4. labels = each TARGET's category id; S_true = the category-block matrix over the TARGETS ONLY
         (within-category 1.0, between 0.0). *** S_true comes ONLY from the a-priori taxonomy. ***
      5. second_order_pairs = within-category TARGET pairs whose DIRECT windowed co-occurrence count is 0
         (the genuine cat~dog case), from the REAL direct-target-cooccurrence set.
      6. Print a frequency report; warn (don't crash) on words below freq_floor / categories below
         min_facts_per_category.

    `concepts == targets + hubs`; `members == targets`; `member_index[target] = row in S_true`.
    `_local` carries the same fields per_bridge_gates/_g5_robust read (labels/S_true/second_order_pairs/
    concepts/members/facts), for parity with the multibridge harness.
    """
    rng = np.random.RandomState(seed * 911 + 7)
    vocab = list(vocab)
    labels = np.asarray(categories, dtype=int)
    Nt = len(vocab)
    assert labels.shape[0] == Nt, "categories must be one id per target word"
    target_set = set(vocab)
    target_to_row = {w: i for i, w in enumerate(vocab)}   # row in S_true (targets only)

    # ----- read corpus (optionally only the first max_bytes for the CPU smoke) -----
    with open(corpus_path, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read(max_bytes) if max_bytes is not None else fh.read()
    text = text.lower()
    stories = text.split("<|endoftext|>")

    # ----- STEP 0: pick the top-n_context_hubs CONTEXT words (excl. stoplist + targets) -----
    global_freq = Counter()
    tokenized_stories = []
    for story in stories:
        toks = re.findall(r"[a-z]+", story)
        tokenized_stories.append(toks)
        global_freq.update(toks)
    hub_words = [w for w, _ in global_freq.most_common()
                 if w not in STOPLIST and w not in target_set][:n_context_hubs]
    hub_set = set(hub_words)

    # concept layout: TARGETS first (so member_rows == [0..Nt)), then the context HUBS.
    concepts = list(vocab) + list(hub_words)
    concept_to_idx = {c: i for i, c in enumerate(concepts)}

    # ----- STEP 1-3: slide a +-window; each window's distinct KEPT set (targets + hubs) = a scene -----
    keep_set = target_set | hub_set
    facts = []
    pair_emit_count = defaultdict(int)        # unordered (concept_idx_a, concept_idx_b) -> #scenes emitted
    direct_target_cooccur = set()             # unordered TARGET-row pairs that DO co-occur directly
    target_count = Counter()                  # per-target in-corpus (windowed-token) frequency
    hub_count = Counter()                     # per-hub in-corpus frequency (diagnostics)

    for toks in tokenized_stories:
        kept = [t for t in toks if t in keep_set]
        for t in kept:
            if t in target_set:
                target_count[t] += 1
            else:
                hub_count[t] += 1
        n = len(kept)
        if n < 2:
            continue
        for c in range(n):
            lo = max(0, c - window)
            hi = min(n, c + window + 1)
            scene = sorted(set(kept[lo:hi]))           # distinct kept words in the window
            if len(scene) < 2:
                continue
            # require >=1 TARGET in the scene (a pure-context scene carries no target signal)
            if not any(w in target_set for w in scene):
                continue
            cidx = sorted(concept_to_idx[w] for w in scene)
            # record DIRECT target-target co-occurrence (for second_order_pairs), BEFORE the cap.
            tgt_rows = sorted(target_to_row[w] for w in scene if w in target_set)
            for a in range(len(tgt_rows)):
                for b in range(a + 1, len(tgt_rows)):
                    direct_target_cooccur.add((tgt_rows[a], tgt_rows[b]))
            # per-pair repeat cap over ALL concept pairs in the scene (targets+hubs).
            pairs = [(cidx[a], cidx[b]) for a in range(len(cidx)) for b in range(a + 1, len(cidx))]
            if all(pair_emit_count[p] < repeat_cap for p in pairs):
                facts.append(tuple(scene))
                for p in pairs:
                    pair_emit_count[p] += 1

    rng.shuffle(facts)

    # ----- INDEPENDENT S_true = category-block matrix over the TARGETS (NEVER corpus-derived) -----
    S_true = (labels[:, None] == labels[None, :]).astype(np.float64)
    np.fill_diagonal(S_true, 1.0)  # cosmetic; off-diag is what the gates read

    # ----- second_order_pairs: within-category TARGET pairs with ZERO direct windowed co-occurrence ----
    second_order_pairs = []
    for i in range(Nt):
        for j in range(i + 1, Nt):
            if labels[i] == labels[j] and (i, j) not in direct_target_cooccur:
                second_order_pairs.append((i, j))

    # ----- frequency report (warn, don't crash) -----
    per_cat_facts = Counter()
    for f in facts:
        cats_in_fact = set(int(labels[target_to_row[w]]) for w in f if w in target_set)
        for cid in cats_in_fact:
            per_cat_facts[cid] += 1
    low_freq_words = [(w, target_count.get(w, 0)) for w in vocab if target_count.get(w, 0) < freq_floor]
    if verbose:
        print(f"  [build_context_inclusive_cooccurrence] corpus={os.path.basename(corpus_path)} "
              f"({'first %d bytes' % max_bytes if max_bytes else 'full'}), {len(stories)} stories, "
              f"window=+-{window}, repeat_cap={repeat_cap}, n_context_hubs={len(hub_words)}", flush=True)
        print(f"    {Nt} TARGET words + {len(hub_words)} CONTEXT-HUB words = {len(concepts)} concepts; "
              f"{len(facts)} context-inclusive co-occurrence facts (singletons / pure-context dropped); "
              f"second-order TARGET pairs (within-cat, NO direct co-occur)={len(second_order_pairs)}",
              flush=True)
        freq_line = ", ".join(f"{w}:{target_count.get(w,0)}" for w in vocab)
        print(f"    per-target in-corpus count: {freq_line}", flush=True)
        top_hubs = ", ".join(f"{w}:{hub_count.get(w,0)}" for w in hub_words[:12])
        print(f"    top context hubs (<=12 shown): {top_hubs}{' ...' if len(hub_words) > 12 else ''}",
              flush=True)
        cov = ", ".join(f"cat{cid}:{per_cat_facts.get(cid,0)}" for cid in sorted(set(labels.tolist())))
        print(f"    per-category fact coverage (#facts touching the category): {cov}", flush=True)
        if low_freq_words:
            print(f"    [WARN] {len(low_freq_words)} target(s) below freq-floor {freq_floor}: "
                  f"{low_freq_words}", flush=True)
        thin_cats = [cid for cid in sorted(set(labels.tolist()))
                     if per_cat_facts.get(cid, 0) < min_facts_per_category]
        if thin_cats:
            print(f"    [WARN] {len(thin_cats)} category(ies) below ~{min_facts_per_category} facts: "
                  f"{thin_cats}", flush=True)

    _local = {
        "concepts": list(concepts), "members": list(vocab),
        "labels": labels, "S_true": S_true, "facts": facts,
        "second_order_pairs": second_order_pairs,
    }
    return {
        "concepts": list(concepts),    # targets FIRST, then context hubs
        "members": list(vocab),        # the 64 TARGETS (scored against S_true)
        "hubs": list(hub_words),       # the context words (shared-neighbour hubs)
        "labels": labels,
        "member_index": target_to_row,  # target-name -> row in S_true (0..Nt-1)
        "S_true": S_true,
        "facts": facts,
        "second_order_pairs": second_order_pairs,
        "n_facts": len(facts),
        "_local": _local,
        # diagnostics:
        "_target_count": dict(target_count),
        "_hub_words": list(hub_words),
        "_hub_count": dict(hub_count),
        "_per_category_fact_coverage": {int(k): int(v) for k, v in per_cat_facts.items()},
        "_low_freq_words": low_freq_words,
        "_n_direct_target_cooccur_pairs": len(direct_target_cooccur),
        "_n_stories": len(stories),
        "_n_context_hubs": len(hub_words),
    }


# ===========================================================================
# The brain-based codes for a learned W (FIXED validated divnorm recipe).
# ===========================================================================
def _brain_based_codes(W: np.ndarray, member_rows: np.ndarray, args) -> np.ndarray:
    return divnorm_spreading_readout(
        W, member_rows,
        divnorm=args.readout_divnorm, order=args.readout_order,
        sigma=args.readout_sigma, exponent=args.readout_exponent,
        alpha=args.diffusion_alpha, steps=args.diffusion_steps,
        log_clip=args.readout_log_clip)


def _learn_codes(concepts, facts, member_rows, seed, args):
    """Learn the brain-based co-occurrence recurrent (Oja homeostatic ceiling) + the validated divnorm
    read-out. Returns (W, codes, info)."""
    W, info = learn_W_homeostatic(
        concepts, facts, seed, args.n_pool, args.pattern_size, args.cycles,
        gamma=1.0, cap=None, homeo=args.homeo, homeo_target=args.homeo_target,
        homeo_clip_only=True)
    codes = _brain_based_codes(W, member_rows, args)
    return W, codes, info


# ===========================================================================
# Per-seed driver (Stage B brain-based fair test on the context-inclusive corpus).
# ===========================================================================
def run_seed(seed: int, args, corpus_path: str, taxonomy: dict) -> dict:
    print(f"\n{'='*84}", flush=True)
    print(f"  OPTION C STAGE-B FAIR TEST (brain-based, context-inclusive corpus) -- SEED {seed}",
          flush=True)
    print(f"{'='*84}", flush=True)

    vocab, cat_ids, cat_names = taxonomy_to_vocab_categories(taxonomy)
    nclu = len(cat_names)
    pclu = len(taxonomy[cat_names[0]])
    assert all(len(taxonomy[c]) == pclu for c in cat_names), \
        "taxonomy categories must all be the same size (assign_properties assumes a regular grid)"
    n_props = args.n_props
    chance = 1.0 / n_props

    # ----- STEP 1: CONTEXT-INCLUSIVE co-occurrence corpus + INDEPENDENT S_true -----
    corpus = build_context_inclusive_cooccurrence(
        corpus_path, vocab, cat_ids, window=args.window, n_context_hubs=args.n_context_hubs,
        repeat_cap=args.repeat_cap, seed=seed, max_bytes=args.max_bytes, freq_floor=args.freq_floor,
        min_facts_per_category=args.min_facts_per_category, verbose=True)
    concepts = corpus["concepts"]
    members = corpus["members"]
    labels = corpus["labels"]
    S_true = corpus["S_true"]
    second_order_pairs = corpus["second_order_pairs"]
    member_rows = np.asarray([concepts.index(m) for m in members], dtype=int)
    Nm = len(members)
    props = assign_properties(nclu, pclu, n_props, seed)
    C_full = raw_count_matrix(concepts, corpus["facts"])

    # ----- INDEPENDENCE SELF-CHECK (the load-bearing correctness property) -----
    S_recon = (labels[:, None] == labels[None, :]).astype(np.float64)
    np.fill_diagonal(S_recon, 1.0)
    s_true_independent = bool(np.array_equal(S_true, S_recon)) and bool(np.array_equal(S_true, S_true.T))
    block_ok = True
    start = 0
    for cid, cname in enumerate(cat_names):
        sz = len(taxonomy[cname])
        block = S_true[start:start + sz, start:start + sz]
        if not np.allclose(block, 1.0):
            block_ok = False
        start += sz
    s_true_independent = s_true_independent and block_ok
    # the members must be the TARGETS only (context hubs are NOT scored): Nm == sum of category sizes.
    members_are_targets = bool(Nm == sum(len(taxonomy[c]) for c in cat_names))
    # the hubs must be DISJOINT from the members (context words, not targets).
    hubs_disjoint = bool(len(set(corpus["hubs"]) & set(members)) == 0)
    assert s_true_independent, "S_true MUST be the a-priori category block matrix (NEVER corpus-derived)"
    print(f"  [S_true INDEPENDENCE CHECK] a-priori block-diagonal: {s_true_independent}; "
          f"members==targets ({Nm}): {members_are_targets}; hubs disjoint from members: {hubs_disjoint}",
          flush=True)

    # ----- STEP 2: LEARN brain-based codes (Oja homeostatic) + divnorm read-out -----
    print("  [STEP 2 -- brain-based spiking-Hebbian learn (learn_W_homeostatic) + divnorm read-out]",
          flush=True)
    t0 = time.time()
    W, learned_codes, learn_info = _learn_codes(concepts, corpus["facts"], member_rows, seed, args)
    learn_s = time.time() - t0
    pearson_W_counts = offdiag_pearson(W, C_full)
    W_distinct = bool(pearson_W_counts < 0.99)
    print(f"    learned recurrent: mean={learn_info['recurrent_mean']:.3f} "
          f"max={learn_info['recurrent_max']:.3f} nnz={learn_info['recurrent_nnz']} "
          f"({learn_info['n_neurons']} neurons, {learn_s:.1f}s); "
          f"Pearson(W, raw_counts)={pearson_W_counts:+.3f} (distinct<0.99={W_distinct})", flush=True)

    # ----- STEP 3: structure recovery (G1) -----
    print("  [STEP 3 -- structure recovery (G1) on the brain-based codes]", flush=True)
    grad_stats = codebook_similarity_stats(learned_codes, labels)
    rec = structure_recovery(learned_codes, S_true, second_order_pairs, seed)
    print(f"    learned: within-cos={grad_stats['within_cluster_cos_mean']:.3f} "
          f"between-cos={grad_stats['between_cluster_cos_mean']:.3f} "
          f"margin={grad_stats['graded_margin']:.3f} graded={grad_stats['is_graded']}", flush=True)
    print(f"    >>> Pearson(S_learned, S_true) = {rec['pearson_learned_vs_Strue']:+.3f}  "
          f"(permuted-codes baseline {rec['pearson_permuted_vs_Strue']:+.3f})", flush=True)
    print(f"    >>> SECOND-ORDER (within-cat, NO direct co-occur): shared-neighbour cos="
          f"{rec['second_order_cos_mean']:+.3f} vs between-cat cos={rec['between_cluster_cos_mean']:+.3f} "
          f"(margin {rec['second_order_margin']:+.3f}, recovered={rec['second_order_recovered']})",
          flush=True)
    g1 = bool(rec["pearson_learned_vs_Strue"] >= args.g1_bar
              and grad_stats["is_graded"]
              and rec["second_order_margin"] >= args.so_margin_bar)

    # ----- STEP 4: architecture generalization gates (G2) -----
    print("  [STEP 4 -- architecture generalization gates (G2)]", flush=True)
    gen = architecture_generalization(learned_codes, labels, props, nclu, pclu, seed,
                                      args.k_neighbours, args.a1_bar)
    print(f"    GATE 2 generalization: graded acc={gen['graded']['accuracy']:.3f} "
          f"(chance={gen['chance']:.3f}, {gen['graded']['ratio_vs_chance']:.1f}x) "
          f"A1={gen['a1']}  orthogonal={gen['orthogonal']['accuracy']:.3f} A2={gen['a2']}  "
          f"permuted-prop={gen['permuted']['accuracy']:.3f} A3={gen['a3']}", flush=True)

    # ----- HEADLINE anti-cheat: permuted co-occurrence -> re-learn -> MUST collapse (margin gate) -----
    print("  [HEADLINE anti-cheat -- PERMUTED-CO-OCCURRENCE (re-learn on scrambled corpus)]",
          flush=True)
    perm_facts = permute_corpus(corpus["facts"], concepts, seed)
    W_perm, perm_codes, _ = _learn_codes(concepts, perm_facts, member_rows, seed, args)
    local = {"labels": labels, "S_true": S_true, "second_order_pairs": second_order_pairs,
             "_props": props, "_nclu": nclu, "_pclu": pclu}
    args._seed_for_g5 = seed
    g5 = _g5_robust(perm_codes, W_perm, member_rows, local, args, chance)
    print(f"    permuted: 2nd-margin={g5['permuted_second_order_margin']:+.3f} "
          f"Pearson(S,S_true)={g5['permuted_pearson_vs_Strue']:+.3f} "
          f"gen={g5['permuted_generalization']:.3f} (chance {chance:.3f}) "
          f"-> COLLAPSES(robust)={g5['g5_collapses_robust']}", flush=True)

    # ----- BEATS-baseline: random Gaussian -----
    rand_codes = random_gaussian_codes(Nm, learned_codes.shape[1], seed)
    gen_rand = run_generalization(rand_codes, labels, props, nclu, pclu, seed, args.k_neighbours)
    beats_random = bool(gen["graded"]["accuracy"] > gen_rand["accuracy"] + 1e-9)
    print(f"    [BEATS-baseline] learned={gen['graded']['accuracy']:.3f} > "
          f"random-Gaussian={gen_rand['accuracy']:.3f} : {beats_random}", flush=True)

    gates = {
        "g1_structure_recovered": g1,
        "g2_a1_generalizes": bool(gen["a1"]),
        "g2_a2_orthogonal_collapses": bool(gen["a2"]),
        "g2_a3_permuted_property_collapses": bool(gen["a3"]),
        "g5_permuted_cooccurrence_collapses": bool(g5["g5_collapses_robust"]),
        "g5_beats_random_baseline": beats_random,
        "W_distinct_from_counts": W_distinct,
        "s_true_independent": s_true_independent,
        "members_are_targets": members_are_targets,
        "hubs_disjoint_from_members": hubs_disjoint,
    }
    print(f"\n  [SEED {seed} gates] {gates}", flush=True)

    return {
        "seed": seed,
        "corpus": {"n_targets": Nm, "n_context_hubs": corpus["_n_context_hubs"],
                   "n_concepts": len(concepts), "n_facts": corpus["n_facts"],
                   "n_second_order_pairs": len(second_order_pairs),
                   "n_direct_target_cooccur_pairs": corpus["_n_direct_target_cooccur_pairs"],
                   "n_stories": corpus["_n_stories"],
                   "per_category_fact_coverage": corpus["_per_category_fact_coverage"],
                   "low_freq_words": corpus["_low_freq_words"],
                   "target_count": corpus["_target_count"],
                   "hub_words": corpus["_hub_words"][:50]},
        "learn_info": learn_info,
        "pearson_W_vs_rawcounts": pearson_W_counts,
        "W_distinct_from_counts": W_distinct,
        "s_true_independent": s_true_independent,
        "members_are_targets": members_are_targets,
        "hubs_disjoint_from_members": hubs_disjoint,
        "graded_stats": grad_stats,
        "structure_recovery": rec,
        "generalization": {"graded": gen["graded"]["accuracy"],
                           "orthogonal": gen["orthogonal"]["accuracy"],
                           "permuted_property": gen["permuted"]["accuracy"],
                           "chance": gen["chance"],
                           "a1": bool(gen["a1"]), "a2": bool(gen["a2"]), "a3": bool(gen["a3"])},
        "generalization_random_baseline": gen_rand["accuracy"],
        "headline_permuted_cooccurrence": g5,
        "gates": gates,
    }


# ===========================================================================
# Verdict logic -- the host-pass precondition gates GO / BOUNDARY_weak_graded / NEGATIVE_no_structure.
# (NEGATIVE_data_too_syntagmatic is decided in main(), BEFORE any seed runs, by the host pre-gate.)
# ===========================================================================
def decide_verdict(per_seed: dict, seeds: list, args, host_passes: bool) -> tuple:
    """Multi-seed verdict per the design §3 table, GATED by the host pre-gate (host_passes):
      GO                     -- host passes AND G1 + graded + 2nd-margin; G2 A1 + A2/A3 collapse;
                                HEADLINE permuted collapses; beats random. => revisit Option C vs B.
      BOUNDARY_weak_graded   -- host passes; permuted collapses + Pearson>0 (real, co-occurrence-driven)
                                BUT gen/2nd-margin marginal (the point-neuron strength gap).
      NEGATIVE_no_structure  -- host passes but brain-based fails G1 (collapse, gen <= chance): a clean,
                                biology-translatable mechanism negative.
      NEGATIVE_not_cooccurrence_driven -- gen passes on graded BUT the permuted control ALSO passes.
    (host_passes is a precondition for reaching Stage B at all; it is True here by construction unless
    the caller forced --skip-host-gate for a plumbing run, in which case it is reported as such.)
    """
    def allseed(field_path):
        out = []
        for s in seeds:
            d = per_seed[str(s)]
            for k in field_path:
                d = d[k]
            out.append(d)
        return out

    g1 = all(per_seed[str(s)]["gates"]["g1_structure_recovered"] for s in seeds)
    g2_a1 = all(per_seed[str(s)]["gates"]["g2_a1_generalizes"] for s in seeds)
    g2_a2 = all(per_seed[str(s)]["gates"]["g2_a2_orthogonal_collapses"] for s in seeds)
    g2_a3 = all(per_seed[str(s)]["gates"]["g2_a3_permuted_property_collapses"] for s in seeds)
    g5_permco = all(per_seed[str(s)]["gates"]["g5_permuted_cooccurrence_collapses"] for s in seeds)
    g5_beats = all(per_seed[str(s)]["gates"]["g5_beats_random_baseline"] for s in seeds)
    w_distinct = all(per_seed[str(s)]["gates"]["W_distinct_from_counts"] for s in seeds)

    pearson_struct = allseed(["structure_recovery", "pearson_learned_vs_Strue"])
    so_margin = allseed(["structure_recovery", "second_order_margin"])
    gen_graded = allseed(["generalization", "graded"])
    chance = per_seed[str(seeds[0])]["generalization"]["chance"]
    pearson_mean = float(np.mean(pearson_struct))
    gen_graded_mean = float(np.mean(gen_graded))
    so_margin_mean = float(np.mean(so_margin))

    co_occurrence_driven = g5_permco
    structure_above_chance = (pearson_mean > 0.0) and (gen_graded_mean > 1.2 * chance)

    if not co_occurrence_driven and g2_a1:
        verdict = "NEGATIVE_not_cooccurrence_driven"
    elif g1 and g2_a1 and g2_a2 and g2_a3 and g5_permco and g5_beats and w_distinct:
        verdict = "GO"
    elif co_occurrence_driven and structure_above_chance and not (g1 and g2_a1):
        verdict = "BOUNDARY_weak_graded"
    elif co_occurrence_driven and (g1 or g2_a1):
        verdict = "BOUNDARY_weak_graded"
    else:
        # brain-based recovered ~no structure. The HOST PRE-GATE is the disambiguator: it has already
        # PASSED (we only reach Stage B if it did), so the data demonstrably carries the structure ->
        # this is the clean mechanism negative.
        verdict = "NEGATIVE_no_structure" if host_passes else "NEGATIVE_data_too_syntagmatic"

    detail = {
        "host_pre_gate_passes": host_passes,
        "g1_structure_recovered_all": g1,
        "g2_a1_generalizes_all": g2_a1,
        "g2_a2_orthogonal_collapses_all": g2_a2,
        "g2_a3_permuted_property_collapses_all": g2_a3,
        "g5_permuted_cooccurrence_collapses_all": g5_permco,
        "g5_beats_random_baseline_all": g5_beats,
        "W_distinct_from_counts_all": w_distinct,
        "co_occurrence_driven": co_occurrence_driven,
        "structure_above_chance": structure_above_chance,
        "pearson_struct_per_seed": pearson_struct,
        "pearson_struct_mean": pearson_mean,
        "second_order_margin_per_seed": so_margin,
        "second_order_margin_mean": so_margin_mean,
        "generalization_graded_per_seed": gen_graded,
        "generalization_graded_mean": gen_graded_mean,
        "generalization_chance": chance,
        "generalization_random_baseline_mean": float(np.mean(
            allseed(["generalization_random_baseline"]))),
        "permuted_cooccurrence_pearson_mean": float(np.mean(
            allseed(["headline_permuted_cooccurrence", "permuted_pearson_vs_Strue"]))),
        "permuted_cooccurrence_generalization_mean": float(np.mean(
            allseed(["headline_permuted_cooccurrence", "permuted_generalization"]))),
    }
    return verdict, detail


def main():
    p = argparse.ArgumentParser(description="Option C STAGE-B fair test "
                                            "(host pre-gate -> brain-based spiking-Hebbian on a "
                                            "CONTEXT-INCLUSIVE TinyStories corpus)")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--smoke", action="store_true",
                   help="CPU-numpy PLUMBING smoke: 12-word/4-category vocab, small n_context_hubs, first "
                        "~300KB of TinyStories, small pool, low cycles (seconds-to-~2min, validates the "
                        "host pre-gate + context-inclusive corpus shape + the full battery end-to-end). "
                        "NOT the decisive run.")
    p.add_argument("--corpus", default=None,
                   help="path to the corpus (default data/corpus/tinystories.txt)")
    # ----- STAGE A host pre-gate (the validated SECOND-ORDER operating point) -----
    p.add_argument("--host-window", type=int, default=2,
                   help="host pre-gate co-occurrence window (validated: 2 -- category structure)")
    p.add_argument("--host-ctx-size", type=int, default=5000,
                   help="host pre-gate context-vocab size (validated: 5000)")
    p.add_argument("--host-svd-dim", type=int, default=100, help="host pre-gate SVD dim (validated: 100)")
    p.add_argument("--host-alpha", type=float, default=0.75,
                   help="host pre-gate PPMI context-smoothing alpha (validated: 0.75)")
    p.add_argument("--host-gate-pearson", type=float, default=0.50,
                   help="HOST GATE: brain-based Stage B runs only if host Pearson(sim,S_true) >= this")
    p.add_argument("--skip-host-gate", action="store_true",
                   help="(PLUMBING ONLY) run Stage B even if the host pre-gate fails -- so the smoke can "
                        "exercise the full battery on the tiny taxonomy. NEVER set on a decisive run.")
    # ----- context-inclusive corpus extraction (the REFINEMENT) -----
    p.add_argument("--window", type=int, default=2,
                   help="symmetric co-occurrence window for the brain-based corpus "
                        "(mirror the host paradigmatic operating point: 2)")
    p.add_argument("--n-context-hubs", type=int, default=500,
                   help="number of top-frequency CONTEXT words included as shared-neighbour HUBS "
                        "(the SECOND-ORDER mechanism; sweep 200-1000)")
    p.add_argument("--repeat-cap", type=int, default=40,
                   help="max scene-tuples emitted per unordered concept-pair (saturation guard)")
    p.add_argument("--max-bytes", type=int, default=None,
                   help="read only the first N bytes of the corpus (smoke sets this)")
    p.add_argument("--freq-floor", type=int, default=30,
                   help="warn if any TARGET word's in-corpus count is below this")
    p.add_argument("--min-facts-per-category", type=int, default=20,
                   help="warn if any category has fewer facts than this")
    # taxonomy size
    p.add_argument("--n-props", type=int, default=4)
    # learned-assoc-graph (brain-based learner) -- match the validated homeostasis defaults
    p.add_argument("--n-pool", type=int, default=2000)
    p.add_argument("--pattern-size", type=int, default=100)
    p.add_argument("--cycles", type=int, default=2,
                   help="store cycles per fact (the validated low-cycle operating point)")
    p.add_argument("--homeo", default="oja", help="biological homeostatic rule: oja / scaling / none")
    p.add_argument("--homeo-target", type=float, default=2.0,
                   help="Oja per-post-neuron incoming-L2 set-point (FIXED; not fit to S_true)")
    # FIXED brain-based divnorm read-out -- the VALIDATED divnorm-GO recipe (ch / interleave / steps2 /
    # sigma0.001 / exp2.0 / logclip off).
    p.add_argument("--readout-divnorm", default="ch")
    p.add_argument("--readout-order", default="interleave")
    p.add_argument("--readout-sigma", type=float, default=0.001)
    p.add_argument("--readout-exponent", type=float, default=2.0)
    p.add_argument("--readout-log-clip", action="store_true")
    p.add_argument("--diffusion-alpha", type=float, default=0.5)
    p.add_argument("--diffusion-steps", type=int, default=2)
    # gate bars (match the de-risk / divnorm / multibridge)
    p.add_argument("--g1-bar", type=float, default=0.5, help="Pearson(S_learned, S_true) >= this")
    p.add_argument("--a1-bar", type=float, default=0.7, help="generalization >= this (1.000-class)")
    p.add_argument("--so-margin-bar", type=float, default=0.10, help="2nd-order cat~dog margin bar")
    p.add_argument("--k-neighbours", type=int, default=3)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    # ----- smoke configuration (CPU-numpy plumbing, seconds-to-~2min) -----
    if args.smoke:
        os.environ.setdefault("SIM_BACKEND", "numpy")
        taxonomy = TAXONOMY_SMOKE
        if args.max_bytes is None:
            args.max_bytes = 300_000          # first ~300 KB of TinyStories
        if args.n_pool == 2000:
            args.n_pool = 400                 # small pool
        if args.pattern_size == 100:
            args.pattern_size = 20
        if args.n_context_hubs == 500:
            args.n_context_hubs = 60          # small hub set for the tiny slice
        if args.host_ctx_size == 5000:
            args.host_ctx_size = 800          # small context vocab for the tiny slice
        args.n_props = 4
        # relax the freq-floor warning for the tiny slice (informational only; never crashes)
        if args.freq_floor == 30:
            args.freq_floor = 5
        if args.min_facts_per_category == 20:
            args.min_facts_per_category = 5
        # the tiny 4-category slice will NOT clear the 0.50 host gate -> allow Stage B to run anyway so
        # the smoke exercises the full battery (PLUMBING ONLY; the decisive run never sets this).
        args.skip_host_gate = True
    else:
        taxonomy = TAXONOMY_8x8

    corpus_path = args.corpus or os.path.join(_REPO, "data", "corpus", "tinystories.txt")
    if not os.path.exists(corpus_path):
        print(f"[ERROR] corpus not found: {corpus_path}", flush=True)
        sys.exit(2)

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    backend = os.environ.get("SIM_BACKEND", "auto")
    cat_names = list(taxonomy.keys())
    t0 = time.time()
    print(f"[option-c STAGE-B fair test] seeds={seeds} backend={backend} smoke={args.smoke}", flush=True)
    print(f"  taxonomy: {len(cat_names)}x{len(taxonomy[cat_names[0]])} = "
          f"{sum(len(v) for v in taxonomy.values())} words ({cat_names})", flush=True)
    print(f"  corpus={corpus_path}  window=+-{args.window} n_context_hubs={args.n_context_hubs} "
          f"repeat_cap={args.repeat_cap} max_bytes={args.max_bytes}", flush=True)
    print(f"  brain-based learn: homeo={args.homeo} target={args.homeo_target} cycles={args.cycles} "
          f"(n_pool={args.n_pool}, pattern_size={args.pattern_size})", flush=True)
    print(f"  read-out FIXED = brain-based divnorm '{args.readout_divnorm}'/{args.readout_order} "
          f"(sigma={args.readout_sigma} exp={args.readout_exponent} steps={args.diffusion_steps})",
          flush=True)
    print(f"  bars: G1(Pearson>={args.g1_bar}) A1(gen>={args.a1_bar}) 2nd(>={args.so_margin_bar})",
          flush=True)

    # =======================================================================
    # STAGE A -- the host PRE-GATE (CPU-numpy). Runs ONCE, BEFORE any GPU bridge. Short-circuits.
    # =======================================================================
    print(f"\n{'#'*84}", flush=True)
    print("  STAGE A -- SECOND-ORDER PARADIGMATIC HOST PRE-GATE (decides whether Stage B may proceed)",
          flush=True)
    print(f"{'#'*84}", flush=True)
    vocab, cat_ids, _ = taxonomy_to_vocab_categories(taxonomy)
    labels = np.asarray(cat_ids, dtype=int)
    with open(corpus_path, "r", encoding="utf-8", errors="ignore") as fh:
        host_text = (fh.read(args.max_bytes) if args.max_bytes is not None else fh.read()).lower()
    host_tokens = re.findall(r"[a-z]+", host_text)
    host = host_pre_gate(host_tokens, vocab, labels, cat_names, args)
    del host_text, host_tokens  # free the corpus text before the GPU work

    if not host["host_passes"] and not args.skip_host_gate:
        # short-circuit: the host fails -> the data is too syntagmatic for even the gold-standard 2nd-order
        # measure -> the cheap-local Option-C question is closed; report + EXIT before any GPU bridge.
        verdict = "NEGATIVE_data_too_syntagmatic"
        summary = {
            "verdict": verdict,
            "stage": "A_only (host pre-gate FAILED -> EXIT before GPU)",
            "seeds": seeds, "backend": backend, "smoke": bool(args.smoke),
            "taxonomy_size": f"{len(cat_names)}x{len(taxonomy[cat_names[0]])}",
            "taxonomy_categories": cat_names,
            "corpus": os.path.basename(corpus_path),
            "host_pre_gate": host,
            "note": ("The SECOND-ORDER paradigmatic host pre-gate did not clear Pearson>="
                     f"{args.host_gate_pearson}; even the gold-standard distributional measure cannot "
                     "recover the (sub-)taxonomy -> the cheap-local Option-C question is closed for this "
                     "scope. ZERO GPU spent. (Per the design, a host-recovered SUBSET may still be a fair "
                     "narrower scope.)"),
            "elapsed_total_s": time.time() - t0,
        }
        print(f"\n{'='*84}", flush=True)
        print(f"  OVERALL VERDICT: {verdict}  (host pre-gate FAILED, Stage B NOT run -- zero GPU)",
              flush=True)
        print(f"  second-order host Pearson {host['second_order_pearson']:+.3f} "
              f"(< gate {args.host_gate_pearson}); first-order host "
              f"{host['first_order_pearson']:+.3f}", flush=True)
        print(f"  Total elapsed: {summary['elapsed_total_s']:.1f}s", flush=True)
        print(f"{'='*84}\n", flush=True)
        _write_out(args, {"summary": summary, "per_seed": {}}, smoke=args.smoke)
        return {"summary": summary, "per_seed": {}}

    if not host["host_passes"] and args.skip_host_gate:
        print("  [--skip-host-gate] host pre-gate did NOT pass but PLUMBING override is set -> running "
              "Stage B anyway (NOT a decisive verdict).", flush=True)

    # =======================================================================
    # STAGE B -- the brain-based fair test on the CONTEXT-INCLUSIVE corpus (GPU for the decisive run).
    # =======================================================================
    print(f"\n{'#'*84}", flush=True)
    print("  STAGE B -- BRAIN-BASED FAIR TEST (context-inclusive corpus; host pre-gate cleared)",
          flush=True)
    print(f"{'#'*84}", flush=True)
    per_seed = {}
    for s in seeds:
        per_seed[str(s)] = run_seed(s, args, corpus_path, taxonomy)

    verdict, detail = decide_verdict(per_seed, seeds, args, host["host_passes"])

    _stage_label = ("A+B (host pre-gate cleared -> brain-based Stage B run)" if host["host_passes"]
                    else "A+B (host pre-gate FAILED but --skip-host-gate PLUMBING override -> Stage B run; "
                         "NOT a decisive verdict)")
    summary = {
        "verdict": verdict,
        "stage": _stage_label,
        "host_pre_gate_passed": bool(host["host_passes"]),
        "host_pre_gate_skipped_for_plumbing": bool(args.skip_host_gate and not host["host_passes"]),
        "seeds": seeds,
        "backend": backend,
        "smoke": bool(args.smoke),
        "taxonomy_size": f"{len(cat_names)}x{len(taxonomy[cat_names[0]])}",
        "taxonomy_categories": cat_names,
        "corpus": os.path.basename(corpus_path),
        "config": {"window": args.window, "n_context_hubs": args.n_context_hubs,
                   "repeat_cap": args.repeat_cap, "max_bytes": args.max_bytes,
                   "homeo": args.homeo, "homeo_target": args.homeo_target, "cycles": args.cycles,
                   "n_pool": args.n_pool, "pattern_size": args.pattern_size,
                   "readout_divnorm": args.readout_divnorm, "readout_order": args.readout_order,
                   "g1_bar": args.g1_bar, "a1_bar": args.a1_bar, "so_margin_bar": args.so_margin_bar,
                   "host_operating_point": host["operating_point"]},
        "host_pre_gate": host,
        "brain_based_note": (
            "the learn is the project's spiking-Hebbian recurrent (learn_W_homeostatic: pool<->pool "
            "Oja-bounded Hebbian growth on a real Izhikevich bridge); the read-out is the validated "
            "brain-based divnorm spreading-activation (ch/interleave). REFINEMENT (design SS B): the "
            "corpus is CONTEXT-INCLUSIVE -- each scene includes the in-window TARGET words AND the "
            "in-window top-N CONTEXT-WORD HUBS, so the learn can pick up the SECOND-ORDER (paradigmatic) "
            "structure (two targets near the same context hubs), exactly the build_toy_cooccurrence "
            "shared-hub mechanism on real text. The S_true reference is the INDEPENDENT a-priori "
            "category taxonomy over the TARGETS ONLY (NEVER corpus-derived). The host PPMI+SVD is the "
            "CPU pre-GATE that runs FIRST and short-circuits the GPU -- NOT a deliverable."),
        "detail": detail,
        "elapsed_total_s": time.time() - t0,
    }

    print(f"\n{'='*84}", flush=True)
    print(f"  OVERALL VERDICT: {verdict}", flush=True)
    print(f"  HOST PRE-GATE (Stage A) passed: {host['host_passes']}  "
          f"(second-order Pearson {host['second_order_pearson']:+.3f} vs gate {args.host_gate_pearson}; "
          f"first-order {host['first_order_pearson']:+.3f})", flush=True)
    print(f"  G1 structure recovered (Pearson vs INDEPENDENT S_true), all seeds: "
          f"{detail['g1_structure_recovered_all']}  (mean {detail['pearson_struct_mean']:+.3f}; "
          f"2nd-order margin {detail['second_order_margin_mean']:+.3f})", flush=True)
    print(f"  G2 A1 generalizes >= {args.a1_bar}, all seeds: {detail['g2_a1_generalizes_all']}  "
          f"(graded {detail['generalization_graded_mean']:.3f} vs chance "
          f"{detail['generalization_chance']:.3f}; random {detail['generalization_random_baseline_mean']:.3f})",
          flush=True)
    print(f"  G2 A2 orthogonal collapses: {detail['g2_a2_orthogonal_collapses_all']}   "
          f"G2 A3 permuted-property collapses: {detail['g2_a3_permuted_property_collapses_all']}",
          flush=True)
    print(f"  HEADLINE permuted-CO-OCCURRENCE collapses (robust margin gate), all seeds: "
          f"{detail['g5_permuted_cooccurrence_collapses_all']}  (permuted Pearson "
          f"{detail['permuted_cooccurrence_pearson_mean']:+.3f}, gen "
          f"{detail['permuted_cooccurrence_generalization_mean']:.3f})", flush=True)
    print(f"  W distinct from raw counts (learning, not pass-through), all seeds: "
          f"{detail['W_distinct_from_counts_all']}", flush=True)
    print(f"  Total elapsed: {summary['elapsed_total_s']:.1f}s", flush=True)
    print(f"{'='*84}\n", flush=True)

    out_data = {"summary": summary, "per_seed": per_seed}
    _write_out(args, out_data, smoke=args.smoke)
    return out_data


def _write_out(args, out_data, smoke):
    if args.out is None:
        raw_dir = os.path.join(_REPO, "research", "findings", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        tag = "smoke" if smoke else "multiseed"
        args.out = os.path.join(raw_dir, f"_option_c_stageB_fair_{tag}_{ts}.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out_data, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
