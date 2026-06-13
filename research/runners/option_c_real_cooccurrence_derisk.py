"""OPTION C DE-RISK -- can the project's validated BRAIN-BASED spiking-Hebbian learn extract GRADED
SEMANTIC structure (animals cluster, colors cluster, ...) from REAL TinyStories text co-occurrence,
measured against an INDEPENDENT a-priori category taxonomy?

CONTEXT (design doc 2026-06-13-option-c-real-cooccurrence-derisk-design.md -- the authoritative spec):
  The Phase-1 corpus-source decision is Option B (hand-curate the within-cluster semantic
  sub-taxonomy) vs Option C (LEARN the similarity from REAL distributional co-occurrence -- the
  brain-faithful path). Per the owner's durable goal (a proper brain analogue; honest negatives under
  strict biology ARE the deliverable), de-risk C CHEAPLY before committing the multi-day build.

THE DECISIVE INSIGHT (design SS0):
  The ENTIRE validated learned-graded-embedding GO chain (NEGATIVE -> diagnosis -> desaturate ->
  divnorm -> confirm -> homeostasis) ran on the SYNTHETIC `build_toy_cooccurrence`, where `S_true` is
  the cluster-block matrix INJECTED BY CONSTRUCTION and the corpus is generated FROM that block. So
  every prior GO tested the MECHANISM on a structure we PLANTED -- never whether REAL text
  co-occurrence carries INDEPENDENT semantic category structure the learn can EXTRACT. That is exactly
  the Option-C question, and it is a clean ONE-FUNCTION SWAP: replace the synthetic corpus + injected
  `S_true` with (a) real windowed co-occurrence over TinyStories and (b) an INDEPENDENT a-priori
  semantic reference. EVERYTHING downstream (learn, read-out, gate harness, anti-cheats, host ceiling)
  is reused VERBATIM by import.

THE FALSIFICATION (design SS2 -- the validated stack, codes-source swapped synthetic -> REAL):
  TinyStories --build_real_cooccurrence--> {facts, S_true(INDEPENDENT), labels, second_order_pairs}
    +-> learn_W_homeostatic(oja) -> divnorm_spreading_readout(ch/interleave) -> learned graded CODES
    +- G1 structure_recovery(codes, S_true, second_order_pairs): Pearson(sim,S_true) + 2nd-order margin
    +- G2 architecture_generalization(codes, labels, props): held-out-neighbour A1 + A2(orth) + A3(perm)
    +- HEADLINE anti-cheat: permute_corpus(facts) -> re-learn -> structure + generalization MUST
       collapse, gated via the _g5_robust MARGIN conjunction (NOT the brittle is_graded boolean)
    +- CEILING (labelled, NOT a deliverable): host_ceiling_codes (PPMI+SVD over the SAME real facts)

  CRITICAL CORRECTNESS PROPERTY (design SS1): `S_true` MUST come ONLY from the a-priori `categories`
  taxonomy -- NEVER from the corpus co-occurrence. A corpus-derived reference would be circular and
  silently invalidate the whole experiment.

  The host PPMI+SVD CEILING is the LOAD-BEARING disambiguator between NEGATIVE_no_structure (host
  passes, mechanism fails) and NEGATIVE_data_too_sparse (host ALSO fails). DO NOT run this de-risk
  without it.

VERDICTS (design SS3, multi-seed): GO / BOUNDARY_weak_graded / NEGATIVE_no_structure /
  NEGATIVE_data_too_sparse / NEGATIVE_not_cooccurrence_driven.

NO sim/ edits; reuse-by-import only. CPU-numpy smoke first (12-word/4-category, first ~200 KB of
TinyStories) validates the adapter + the full battery plumbing in seconds. The decisive 64-word
3-seed run is GPU (controller-driven).

Run (the DECISIVE GPU run -- controller-driven, after the GPU frees):
  SIM_BACKEND=cupy python -u -m research.runners.option_c_real_cooccurrence_derisk \
      --seeds 42,43,44 --out research/findings/raw/_option_c_real_cooccurrence_multiseed.json

Run (the CPU-numpy plumbing SMOKE -- seconds-to-~2min):
  SIM_BACKEND=numpy python -u -m research.runners.option_c_real_cooccurrence_derisk \
      --smoke --seeds 42 --out research/findings/raw/_option_c_smoke.json
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
# The de-risk's structure-recovery + generalization battery + host CEILING + random baseline +
# permuted-co-occurrence control:
from research.runners.learned_graded_embedding_derisk_probe import (  # noqa: E402
    permute_corpus,
    structure_recovery,
    architecture_generalization,
    host_ceiling_codes,
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


# ===========================================================================
# The 8x8 a-priori taxonomy (design SS1). 8 categories x 8 common words. The INDEPENDENT semantic
# reference: S_true is the category-block matrix over THIS taxonomy -- NEVER corpus-derived.
# All 34 sampled words verified present in TinyStories; the freq report confirms the full 64.
# ===========================================================================
TAXONOMY_8x8 = {
    "animals": ["dog", "cat", "bird", "fish", "frog", "bear", "mouse", "duck"],
    "food":    ["apple", "cake", "bread", "milk", "egg", "soup", "candy", "cookie"],
    "body":    ["hand", "eye", "foot", "head", "hair", "arm", "leg", "face"],
    "family":  ["mom", "dad", "girl", "boy", "baby", "friend", "sister", "brother"],
    "actions": ["run", "jump", "walk", "play", "look", "eat", "sleep", "sing"],
    "colors":  ["red", "blue", "green", "yellow", "black", "white", "pink", "brown"],
    "places":  ["house", "park", "room", "garden", "tree", "road", "school", "beach"],
    "toys":    ["ball", "toy", "book", "doll", "box", "blocks", "kite", "bell"],
}

# A small 12-word / 4-category subset for the CPU-numpy smoke (the plumbing test). Words chosen for
# high TinyStories frequency so the tiny-corpus slice still yields facts.
TAXONOMY_SMOKE = {
    "animals": ["dog", "cat", "bird"],
    "family":  ["mom", "dad", "girl"],
    "actions": ["run", "play", "look"],
    "places":  ["house", "park", "tree"],
}


def taxonomy_to_vocab_categories(taxonomy: dict):
    """Flatten a category->words dict into (vocab list grouped category-by-category in order,
    per-word category-id array, category-name list). The category-by-category ORDER is required so
    `assign_properties` (which orders props by cluster*per_cluster+m) and `labels` agree."""
    cat_names = list(taxonomy.keys())
    vocab = []
    cat_ids = []
    for cid, cname in enumerate(cat_names):
        for w in taxonomy[cname]:
            vocab.append(w)
            cat_ids.append(cid)
    return vocab, np.asarray(cat_ids, dtype=int), cat_names


# ===========================================================================
# THE ONE NEW FUNCTION: the corpus->facts adapter + the INDEPENDENT S_true.
# Returns EXACTLY the keys `build_toy_cooccurrence` returns so the learn/read-out/gates are reused
# unchanged: concepts, members, hubs(=[] for real text), labels, member_index, S_true, facts,
# second_order_pairs (+ n_facts for the de-risk's run_seed convenience).
# ===========================================================================
def build_real_cooccurrence(corpus_path: str, vocab: list, categories: np.ndarray,
                            window: int = 5, repeat_cap: int = 40, seed: int = 42,
                            max_bytes: int | None = None, freq_floor: int = 30,
                            min_facts_per_category: int = 20, verbose: bool = True) -> dict:
    """Extract a windowed co-occurrence corpus over REAL TinyStories text, restricted to `vocab`, and
    pair it with an INDEPENDENT a-priori category taxonomy (`categories`).

    Algorithm (design SS1.3):
      1. Read TinyStories, lowercase, split on '<|endoftext|>' into stories; within each story
         tokenize with re.findall(r"[a-z]+", ...).
      2. Restrict tokens to `vocab` (drop everything else).
      3. Slide a symmetric window of +-`window` tokens; each window's SET of distinct in-vocab words
         = one co-occurrence scene -> one fact tuple. Drop singletons (need >=2 members).
      4. Per-pair repeat cap: emit at most `repeat_cap` scene-tuples containing any given unordered
         word-pair (seeded RNG) so a few high-frequency words don't saturate the recurrent.
      5. labels = each vocab word's category id; S_true = the category-block matrix from labels
         (within-category 1.0, between 0.0). *** S_true comes ONLY from the a-priori taxonomy. ***
      6. second_order_pairs = within-category word pairs whose DIRECT windowed co-occurrence count
         is 0 (the genuine cat~dog case), from the REAL direct_cooccur set.
      7. Print a frequency report; warn (don't crash) on words below freq_floor / categories below
         min_facts_per_category.

    `concepts == members == vocab` (real text has no hub concepts; shared within-category context
    words play the hub role naturally). `member_index` maps word -> row in S_true.
    """
    rng = np.random.RandomState(seed * 911 + 7)
    vocab = list(vocab)
    labels = np.asarray(categories, dtype=int)
    Nm = len(vocab)
    assert labels.shape[0] == Nm, "categories must be one id per vocab word"
    vocab_set = set(vocab)
    word_to_row = {w: i for i, w in enumerate(vocab)}

    # ----- read corpus (optionally only the first max_bytes for the CPU smoke) -----
    with open(corpus_path, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read(max_bytes) if max_bytes is not None else fh.read()
    text = text.lower()
    stories = text.split("<|endoftext|>")

    # ----- slide a +-window over each story's in-vocab tokens; each window's distinct set = a fact --
    facts = []
    pair_emit_count = defaultdict(int)          # unordered (rowA,rowB) -> #scene-tuples emitted with it
    direct_cooccur = set()                       # unordered (rowA,rowB) word-pairs that DO co-occur
    word_count = Counter()                       # per-word in-corpus (windowed-token) frequency

    for story in stories:
        toks = re.findall(r"[a-z]+", story)
        in_vocab = [t for t in toks if t in vocab_set]
        for t in in_vocab:
            word_count[t] += 1
        n = len(in_vocab)
        if n < 2:
            continue
        for c in range(n):
            lo = max(0, c - window)
            hi = min(n, c + window + 1)
            scene = sorted(set(in_vocab[lo:hi]))   # distinct in-vocab words in the window
            if len(scene) < 2:
                continue
            rows = sorted(word_to_row[w] for w in scene)
            # record ALL direct co-occurrences in this window (for second_order_pairs, BEFORE the cap)
            for a in range(len(rows)):
                for b in range(a + 1, len(rows)):
                    direct_cooccur.add((rows[a], rows[b]))
            # per-pair repeat cap: emit this scene only if EVERY pair it contains is still under-cap.
            # (a conservative cap: a scene is admitted only while none of its pairs has saturated, so
            #  high-frequency words stop contributing once their pairs hit the cap.)
            pairs = [(rows[a], rows[b]) for a in range(len(rows)) for b in range(a + 1, len(rows))]
            if all(pair_emit_count[p] < repeat_cap for p in pairs):
                facts.append(tuple(scene))
                for p in pairs:
                    pair_emit_count[p] += 1

    rng.shuffle(facts)

    # ----- INDEPENDENT S_true = category-block matrix (NEVER corpus-derived) -----
    S_true = (labels[:, None] == labels[None, :]).astype(np.float64)
    np.fill_diagonal(S_true, 1.0)  # cosmetic; off-diag is what the gates read

    # ----- second_order_pairs: within-category pairs with ZERO direct windowed co-occurrence -----
    second_order_pairs = []
    for i in range(Nm):
        for j in range(i + 1, Nm):
            if labels[i] == labels[j] and (i, j) not in direct_cooccur:
                second_order_pairs.append((i, j))

    # ----- frequency report (warn, don't crash) -----
    per_cat_facts = Counter()
    for f in facts:
        cats_in_fact = set(int(labels[word_to_row[w]]) for w in f)
        for cid in cats_in_fact:
            per_cat_facts[cid] += 1
    low_freq_words = [(w, word_count.get(w, 0)) for w in vocab if word_count.get(w, 0) < freq_floor]
    cat_names_present = {}
    # derive category names from contiguous label blocks if not provided externally
    if verbose:
        print(f"  [build_real_cooccurrence] corpus={os.path.basename(corpus_path)} "
              f"({'first %d bytes' % max_bytes if max_bytes else 'full'}), {len(stories)} stories, "
              f"window=+-{window}, repeat_cap={repeat_cap}", flush=True)
        print(f"    {Nm} vocab words; {len(facts)} co-occurrence facts (singletons dropped); "
              f"second-order pairs (within-cat, NO direct co-occur)={len(second_order_pairs)}",
              flush=True)
        # per-word frequency line
        freq_line = ", ".join(f"{w}:{word_count.get(w,0)}" for w in vocab)
        print(f"    per-word in-corpus count: {freq_line}", flush=True)
        # per-category fact coverage
        cov = ", ".join(f"cat{cid}:{per_cat_facts.get(cid,0)}" for cid in sorted(set(labels.tolist())))
        print(f"    per-category fact coverage (#facts touching the category): {cov}", flush=True)
        if low_freq_words:
            print(f"    [WARN] {len(low_freq_words)} word(s) below freq-floor {freq_floor}: "
                  f"{low_freq_words}", flush=True)
        thin_cats = [cid for cid in sorted(set(labels.tolist()))
                     if per_cat_facts.get(cid, 0) < min_facts_per_category]
        if thin_cats:
            print(f"    [WARN] {len(thin_cats)} category(ies) below ~{min_facts_per_category} facts: "
                  f"{thin_cats}", flush=True)

    return {
        "concepts": list(vocab),       # real text: concepts == members == vocab (no hubs)
        "members": list(vocab),
        "hubs": [],                    # real text has no hub concepts
        "labels": labels,
        "member_index": word_to_row,
        "S_true": S_true,
        "facts": facts,
        "second_order_pairs": second_order_pairs,
        "n_facts": len(facts),
        # diagnostics (not part of the toy contract, but useful in the JSON):
        "_word_count": dict(word_count),
        "_per_category_fact_coverage": {int(k): int(v) for k, v in per_cat_facts.items()},
        "_low_freq_words": low_freq_words,
        "_n_direct_cooccur_pairs": len(direct_cooccur),
        "_n_stories": len(stories),
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
    """Learn the brain-based co-occurrence recurrent (Oja homeostatic ceiling) + the validated
    divnorm read-out. Returns (W, codes, info)."""
    W, info = learn_W_homeostatic(
        concepts, facts, seed, args.n_pool, args.pattern_size, args.cycles,
        gamma=1.0, cap=None, homeo=args.homeo, homeo_target=args.homeo_target,
        homeo_clip_only=True)
    codes = _brain_based_codes(W, member_rows, args)
    return W, codes, info


# ===========================================================================
# Per-seed driver
# ===========================================================================
def run_seed(seed: int, args, corpus_path: str, taxonomy: dict) -> dict:
    print(f"\n{'='*84}", flush=True)
    print(f"  OPTION C REAL-CO-OCCURRENCE DE-RISK -- SEED {seed}", flush=True)
    print(f"{'='*84}", flush=True)

    vocab, cat_ids, cat_names = taxonomy_to_vocab_categories(taxonomy)
    nclu = len(cat_names)
    pclu = len(taxonomy[cat_names[0]])
    # assert balanced taxonomy (all categories same size) -- required by assign_properties' layout.
    assert all(len(taxonomy[c]) == pclu for c in cat_names), \
        "taxonomy categories must all be the same size (assign_properties assumes a regular grid)"
    n_props = args.n_props
    chance = 1.0 / n_props

    # ----- STEP 1: REAL co-occurrence corpus + INDEPENDENT S_true -----
    corpus = build_real_cooccurrence(
        corpus_path, vocab, cat_ids, window=args.window, repeat_cap=args.repeat_cap,
        seed=seed, max_bytes=args.max_bytes, freq_floor=args.freq_floor,
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
    # S_true must be exactly the a-priori category-block matrix (symmetric, block-diagonal under the
    # category-grouped vocab order), and must NOT equal a corpus-derived block.
    S_recon = (labels[:, None] == labels[None, :]).astype(np.float64)
    np.fill_diagonal(S_recon, 1.0)
    s_true_independent = bool(np.array_equal(S_true, S_recon)) and bool(np.array_equal(S_true, S_true.T))
    # also confirm it is block-diagonal under the grouped order (each contiguous category block == 1)
    block_ok = True
    start = 0
    for cid, cname in enumerate(cat_names):
        sz = len(taxonomy[cname])
        block = S_true[start:start + sz, start:start + sz]
        if not np.allclose(block, 1.0):
            block_ok = False
        start += sz
    s_true_independent = s_true_independent and block_ok
    print(f"  [S_true INDEPENDENCE CHECK] from a-priori taxonomy only, symmetric block-diagonal: "
          f"{s_true_independent}", flush=True)

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
    # _g5_robust expects a `local` dict + args._seed_for_g5 set.
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

    # ----- HOST CEILING (PPMI+SVD over the SAME real facts; labelled, NOT a deliverable) -----
    print("  [HOST CEILING -- PPMI+SVD over the SAME real facts (labelled disambiguator)]", flush=True)
    host_codes = host_ceiling_codes(concepts, corpus["facts"], member_rows, learned_codes.shape[1], seed)
    host_grad = codebook_similarity_stats(host_codes, labels)
    host_rec = structure_recovery(host_codes, S_true, second_order_pairs, seed)
    gen_host = run_generalization(host_codes, labels, props, nclu, pclu, seed, args.k_neighbours)
    host_g1 = bool(host_rec["pearson_learned_vs_Strue"] >= args.g1_bar and host_grad["is_graded"]
                   and host_rec["second_order_margin"] >= args.so_margin_bar)
    host_g2 = bool(gen_host["accuracy"] >= args.a1_bar)
    host_passes = bool(host_g1 and host_g2)   # the disambiguator: does the data HAVE the structure?
    print(f"    host: Pearson(S,S_true)={host_rec['pearson_learned_vs_Strue']:+.3f} "
          f"2nd-margin={host_rec['second_order_margin']:+.3f} graded={host_grad['is_graded']} "
          f"gen={gen_host['accuracy']:.3f} -> host_G1={host_g1} host_G2={host_g2} "
          f"HOST_PASSES={host_passes}", flush=True)

    gates = {
        "g1_structure_recovered": g1,
        "g2_a1_generalizes": bool(gen["a1"]),
        "g2_a2_orthogonal_collapses": bool(gen["a2"]),
        "g2_a3_permuted_property_collapses": bool(gen["a3"]),
        "g5_permuted_cooccurrence_collapses": bool(g5["g5_collapses_robust"]),
        "g5_beats_random_baseline": beats_random,
        "W_distinct_from_counts": W_distinct,
        "s_true_independent": s_true_independent,
        "host_ceiling_passes": host_passes,
    }
    print(f"\n  [SEED {seed} gates] {gates}", flush=True)

    return {
        "seed": seed,
        "corpus": {"n_vocab": Nm, "n_facts": corpus["n_facts"],
                   "n_second_order_pairs": len(second_order_pairs),
                   "n_direct_cooccur_pairs": corpus["_n_direct_cooccur_pairs"],
                   "n_stories": corpus["_n_stories"],
                   "per_category_fact_coverage": corpus["_per_category_fact_coverage"],
                   "low_freq_words": corpus["_low_freq_words"],
                   "word_count": corpus["_word_count"]},
        "learn_info": learn_info,
        "pearson_W_vs_rawcounts": pearson_W_counts,
        "W_distinct_from_counts": W_distinct,
        "s_true_independent": s_true_independent,
        "graded_stats": grad_stats,
        "structure_recovery": rec,
        "generalization": {"graded": gen["graded"]["accuracy"],
                           "orthogonal": gen["orthogonal"]["accuracy"],
                           "permuted_property": gen["permuted"]["accuracy"],
                           "chance": gen["chance"],
                           "a1": bool(gen["a1"]), "a2": bool(gen["a2"]), "a3": bool(gen["a3"])},
        "generalization_random_baseline": gen_rand["accuracy"],
        "headline_permuted_cooccurrence": g5,
        "host_ceiling": {"structure_recovery": host_rec, "graded_stats": host_grad,
                         "generalization": gen_host["accuracy"],
                         "host_g1": host_g1, "host_g2": host_g2, "host_passes": host_passes},
        "gates": gates,
    }


# ===========================================================================
# Verdict logic (design SS3) -- the only genuinely new logic.
# ===========================================================================
def decide_verdict(per_seed: dict, seeds: list, args) -> tuple:
    """Multi-seed verdict per the design SS3 table:
      GO                         -- G1 + graded + 2nd-margin; G2 A1 + A2/A3 collapse; HEADLINE
                                    permuted collapses; beats random. => Build on C.
      BOUNDARY_weak_graded       -- permuted collapses + Pearson > 0 (real, co-occurrence-driven) BUT
                                    gen marginal (chance<gen<bar) or 2nd-margin in [0, +bar);
                                    especially if the host ceiling PASSES while brain-based lags.
      NEGATIVE_no_structure      -- brain-based fails G1 (collapse, gen <= chance) AND host PASSES.
      NEGATIVE_data_too_sparse   -- host ceiling ALSO fails (inconclusive for the mechanism).
      NEGATIVE_not_cooccurrence_driven -- gen passes on graded BUT the permuted control ALSO passes.
    """
    def allseed(field_path):
        out = []
        for s in seeds:
            d = per_seed[str(s)]
            for k in field_path:
                d = d[k]
            out.append(d)
        return out

    # all-seed gate conjunctions
    g1 = all(per_seed[str(s)]["gates"]["g1_structure_recovered"] for s in seeds)
    g2_a1 = all(per_seed[str(s)]["gates"]["g2_a1_generalizes"] for s in seeds)
    g2_a2 = all(per_seed[str(s)]["gates"]["g2_a2_orthogonal_collapses"] for s in seeds)
    g2_a3 = all(per_seed[str(s)]["gates"]["g2_a3_permuted_property_collapses"] for s in seeds)
    g5_permco = all(per_seed[str(s)]["gates"]["g5_permuted_cooccurrence_collapses"] for s in seeds)
    g5_beats = all(per_seed[str(s)]["gates"]["g5_beats_random_baseline"] for s in seeds)
    w_distinct = all(per_seed[str(s)]["gates"]["W_distinct_from_counts"] for s in seeds)
    host_passes = all(per_seed[str(s)]["gates"]["host_ceiling_passes"] for s in seeds)

    pearson_struct = allseed(["structure_recovery", "pearson_learned_vs_Strue"])
    so_margin = allseed(["structure_recovery", "second_order_margin"])
    gen_graded = allseed(["generalization", "graded"])
    chance = per_seed[str(seeds[0])]["generalization"]["chance"]
    pearson_mean = float(np.mean(pearson_struct))
    gen_graded_mean = float(np.mean(gen_graded))
    so_margin_mean = float(np.mean(so_margin))

    # The structure is REAL + co-occurrence-driven iff the permuted control collapses AND the learned
    # Pearson is positive (above the permuted/random level).
    co_occurrence_driven = g5_permco
    structure_above_chance = (pearson_mean > 0.0) and (gen_graded_mean > 1.2 * chance)

    # ---- branch the verdict ----
    if not co_occurrence_driven and g2_a1:
        # generalization passes on graded BUT the permuted control did NOT collapse -> read-out
        # artifact, not data (guarded by the headline control).
        verdict = "NEGATIVE_not_cooccurrence_driven"
    elif g1 and g2_a1 and g2_a2 and g2_a3 and g5_permco and g5_beats and w_distinct:
        verdict = "GO"
    elif co_occurrence_driven and structure_above_chance and not (g1 and g2_a1):
        # learns the RIGHT (co-occurrence-driven) structure -- permuted collapses, Pearson positive,
        # above chance -- but gen marginal / 2nd-margin weak: the biological-strength gap.
        verdict = "BOUNDARY_weak_graded"
    elif co_occurrence_driven and (g1 or g2_a1):
        # passes part of the battery but not all (and not the clean weak-graded shape) -> still the
        # weak-graded characterization (real but partial).
        verdict = "BOUNDARY_weak_graded"
    else:
        # brain-based recovered ~no structure. The host ceiling is the disambiguator.
        if host_passes:
            verdict = "NEGATIVE_no_structure"        # data HAS the structure; the mechanism can't
        else:
            verdict = "NEGATIVE_data_too_sparse"      # host ALSO fails -> inconclusive for the mechanism

    detail = {
        "g1_structure_recovered_all": g1,
        "g2_a1_generalizes_all": g2_a1,
        "g2_a2_orthogonal_collapses_all": g2_a2,
        "g2_a3_permuted_property_collapses_all": g2_a3,
        "g5_permuted_cooccurrence_collapses_all": g5_permco,
        "g5_beats_random_baseline_all": g5_beats,
        "W_distinct_from_counts_all": w_distinct,
        "host_ceiling_passes_all": host_passes,
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
        "host_ceiling_pearson_mean": float(np.mean(
            allseed(["host_ceiling", "structure_recovery", "pearson_learned_vs_Strue"]))),
        "host_ceiling_generalization_mean": float(np.mean(allseed(["host_ceiling", "generalization"]))),
        "brain_vs_host_generalization_gap": float(
            np.mean(allseed(["host_ceiling", "generalization"])) - gen_graded_mean),
        "permuted_cooccurrence_pearson_mean": float(np.mean(
            allseed(["headline_permuted_cooccurrence", "permuted_pearson_vs_Strue"]))),
        "permuted_cooccurrence_generalization_mean": float(np.mean(
            allseed(["headline_permuted_cooccurrence", "permuted_generalization"]))),
    }
    return verdict, detail


def main():
    p = argparse.ArgumentParser(description="Option C real-co-occurrence de-risk "
                                            "(brain-based spiking-Hebbian on REAL TinyStories text)")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--smoke", action="store_true",
                   help="CPU-numpy PLUMBING smoke: 12-word/4-category vocab, first ~200KB of "
                        "TinyStories, small pool, low cycles (seconds-to-~2min, validates the "
                        "adapter + full battery end-to-end). NOT the decisive run.")
    p.add_argument("--corpus", default=None,
                   help="path to the corpus (default data/corpus/tinystories.txt)")
    # corpus extraction
    p.add_argument("--window", type=int, default=5, help="symmetric co-occurrence window (+- tokens)")
    p.add_argument("--repeat-cap", type=int, default=40,
                   help="max scene-tuples emitted per unordered word-pair (saturation guard)")
    p.add_argument("--max-bytes", type=int, default=None,
                   help="read only the first N bytes of the corpus (smoke sets this)")
    p.add_argument("--freq-floor", type=int, default=30,
                   help="warn if any vocab word's in-corpus count is below this")
    p.add_argument("--min-facts-per-category", type=int, default=20,
                   help="warn if any category has fewer facts than this")
    # taxonomy size (smoke overrides n-props default via the smoke block)
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
            args.max_bytes = 200_000          # first ~200 KB of TinyStories
        if args.n_pool == 2000:
            args.n_pool = 400                 # small pool
        if args.pattern_size == 100:
            args.pattern_size = 20
        if args.cycles == 2:
            args.cycles = 2                   # already low
        args.n_props = 4
        # relax the freq-floor warning for the tiny slice (informational only; never crashes)
        if args.freq_floor == 30:
            args.freq_floor = 5
        if args.min_facts_per_category == 20:
            args.min_facts_per_category = 5
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
    print(f"[option-c real-co-occurrence de-risk] seeds={seeds} backend={backend} "
          f"smoke={args.smoke}", flush=True)
    print(f"  taxonomy: {len(cat_names)}x{len(taxonomy[cat_names[0]])} = "
          f"{sum(len(v) for v in taxonomy.values())} words ({cat_names})", flush=True)
    print(f"  corpus={corpus_path}  window=+-{args.window} repeat_cap={args.repeat_cap} "
          f"max_bytes={args.max_bytes}", flush=True)
    print(f"  brain-based learn: homeo={args.homeo} target={args.homeo_target} cycles={args.cycles} "
          f"(n_pool={args.n_pool}, pattern_size={args.pattern_size})", flush=True)
    print(f"  read-out FIXED = brain-based divnorm '{args.readout_divnorm}'/{args.readout_order} "
          f"(sigma={args.readout_sigma} exp={args.readout_exponent} steps={args.diffusion_steps})",
          flush=True)
    print(f"  bars: G1(Pearson>={args.g1_bar}) A1(gen>={args.a1_bar}) 2nd(>={args.so_margin_bar})",
          flush=True)

    per_seed = {}
    for s in seeds:
        per_seed[str(s)] = run_seed(s, args, corpus_path, taxonomy)

    verdict, detail = decide_verdict(per_seed, seeds, args)

    summary = {
        "verdict": verdict,
        "seeds": seeds,
        "backend": backend,
        "smoke": bool(args.smoke),
        "taxonomy_size": f"{len(cat_names)}x{len(taxonomy[cat_names[0]])}",
        "taxonomy_categories": cat_names,
        "corpus": os.path.basename(corpus_path),
        "config": {"window": args.window, "repeat_cap": args.repeat_cap, "max_bytes": args.max_bytes,
                   "homeo": args.homeo, "homeo_target": args.homeo_target, "cycles": args.cycles,
                   "n_pool": args.n_pool, "pattern_size": args.pattern_size,
                   "readout_divnorm": args.readout_divnorm, "readout_order": args.readout_order,
                   "g1_bar": args.g1_bar, "a1_bar": args.a1_bar, "so_margin_bar": args.so_margin_bar},
        "brain_based_note": (
            "the learn is the project's spiking-Hebbian recurrent (learn_W_homeostatic: pool<->pool "
            "Oja-bounded Hebbian growth on a real Izhikevich bridge); the read-out is the validated "
            "brain-based divnorm spreading-activation (ch/interleave). The S_true reference is the "
            "INDEPENDENT a-priori category taxonomy (NEVER corpus-derived). The host PPMI+SVD over the "
            "SAME real facts is a labelled CEILING ONLY, the NEGATIVE_no_structure vs "
            "NEGATIVE_data_too_sparse disambiguator -- NOT a deliverable."),
        "detail": detail,
        "elapsed_total_s": time.time() - t0,
    }

    print(f"\n{'='*84}", flush=True)
    print(f"  OVERALL VERDICT: {verdict}", flush=True)
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
    print(f"  HOST CEILING (PPMI+SVD on SAME facts; labelled disambiguator): PASSES all seeds="
          f"{detail['host_ceiling_passes_all']}  (Pearson {detail['host_ceiling_pearson_mean']:+.3f}, "
          f"gen {detail['host_ceiling_generalization_mean']:.3f}); brain-vs-host gen gap "
          f"{detail['brain_vs_host_generalization_gap']:+.3f}", flush=True)
    print(f"  Total elapsed: {summary['elapsed_total_s']:.1f}s", flush=True)
    print(f"{'='*84}\n", flush=True)

    out_data = {"summary": summary, "per_seed": per_seed}
    if args.out is None:
        raw_dir = os.path.join(_REPO, "research", "findings", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        tag = "smoke" if args.smoke else "multiseed"
        args.out = os.path.join(raw_dir, f"_option_c_real_cooccurrence_{tag}_{ts}.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out_data, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)
    return out_data


if __name__ == "__main__":
    main()
