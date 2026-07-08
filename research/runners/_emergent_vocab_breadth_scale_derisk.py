"""EMERGENT-VOCAB BREADTH SCALE de-risk -- is open-domain conversational BREADTH a cheap DATA/SCALE lever
on the emergent stream cortex, or a MECHANISM gap?

THE FRONTIER (the goal-relevant communication wall): the talkable brain reasons + speaks on spikes, grounded,
no-confab -- but the MOST BLOCKING wall (fluid-conversation gap assessment) is open-domain BREADTH: the
vocab/knowledge is a fixed few-thousand(-or-fewer)-concept closed set. The emergent stream cortex
(`_phaseB_online_stream_cortex_derisk.py`) LEARNS word co-occurrence structure from a corpus, BUT its vocab is
a HAND-FIXED 64-word 8x8 taxonomy. The mission (emergent-from-experience) wants the vocab itself DISCOVERED
from a real corpus, at scale.

THE QUESTION: does the emergent stream cortex SCALE to a BIGGER EMERGENT vocab -- the vocab DISCOVERED from a
real corpus (top-frequency content words), and their co-occurrence/semantic structure LEARNED (not
hand-assigned)? Measure the scaling: vocab K in {64, 256, 1024}. Does the emergent semantic structure hold as
the vocab grows, or degrade? If it HOLDS -> breadth is a cheap DATA/SCALE lever (controller runs bigger +
multi-seed). If it DEGRADES -> a MECHANISM gap (new mechanism / research-gate needed).

THE BUILD (three pieces, all emergent-from-experience):
  (1) DISCOVER the vocab from the real corpus = the top-K most-frequent CONTENT words (K in {64,256,1024}),
      emergent -- NOT a hand-assigned taxonomy. (stoplist + short-word filter, else pure corpus frequency.)
  (2) LEARN their co-occurrence/semantic structure via the EXISTING emergent stream mechanism -- ONLINE
      Hebbian co-occurrence in a working-memory window + running-frequency normalization + log-domain
      double-centering (the CYCLE-93 owner reframe: NO global PPMI-matrix shortcut; the cortex HEARS the
      stream word-by-word and accumulates the association weights incrementally). Reused VERBATIM from
      `_phaseB_online_stream_cortex_derisk` (imported `double_center`; the stream loop re-expressed here so it
      can run over the DISCOVERED vocab of any size, byte-faithful to the mechanism).
  (3) MEASURE whether the emergent structure HOLDS at scale. Because the vocab is DISCOVERED (unlabeled), the
      semantic metric uses a PROBE TAXONOMY: the known-category words (a curated a-priori taxonomy) that HAPPEN
      to appear in the discovered vocab. The probe words are embedded WITHIN the full K-vocab co-occurrence; we
      measure their nearest-neighbour category COHERENCE (`heldout_generalization`, reused) + the
      within-vs-between-category cosine MARGIN + Pearson(cos, S_true_probe). As K grows, MORE (unlabeled) vocab
      surrounds the probe anchors -- does that PRESERVE or DEGRADE the anchors' semantic neighbourhoods? THAT
      is the scaling answer. (The probe taxonomy is used ONLY as an evaluation yardstick; it never labels the
      vocab -- the vocab + co-occurrence are 100% corpus-discovered.)

ANTI-CHEATS (the structure must be GENUINELY learned from REAL corpus co-occurrence, at every K):
  * SCRAMBLED-CORPUS control: shuffle the token order within each story before streaming, destroying the real
    windowed co-occurrence while preserving the exact unigram frequency (so vocab discovery is identical). If
    the probe-structure is real, it COLLAPSES under scramble (-> chance).
  * FREQUENCY-ONLY baseline: codes built from the marginal running frequency ONLY (no co-occurrence) -- must
    be at chance (structure is not a frequency artifact).

VERDICTS (1-seed smoke here; controller runs multi-seed):
  SCALES        probe-coherence HOLDS (>= hold_frac of the K=64 value) up to the largest feasible K, scramble
                collapses at every K, beats frequency-only -> BREADTH IS A CHEAP DATA/SCALE LEVER.
  DEGRADES      probe-coherence falls with K (structure dilutes as unlabeled vocab grows) -> a MECHANISM gap;
                dispatch a research gate (e.g. hierarchical / sparse-distributed capacity).
  DATA_BOUND    the corpus is too small/narrow to populate a K-vocab with reliable co-occurrence (probe words
                too sparse; frequency-only ~ learned) -> INCONCLUSIVE for the mechanism; a bigger corpus is the
                prerequisite (a DATA lever, but not yet demonstrated).

Reuse-by-import (the stream double-center + the metric helpers + the a-priori probe taxonomy); NO sim/ edits;
numpy. The corpus is `--corpus-path` (default the repo's `research/datasets/distill_corpus.txt`); the
controller points it at a bigger corpus (TinyStories / WikiText / BabyLM) for the decisive multi-seed run.

Run (1-seed smoke, CPU):
  SIM_BACKEND=numpy python -u -m research.runners._emergent_vocab_breadth_scale_derisk --seeds 42 \
      --vocab-sizes 64,256,1024
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# ----- Reuse VERBATIM by import -----
# The stream cortex's log-domain double-centering (the validated normalization) + the WM/EMA constants:
from research.runners._phaseB_online_stream_cortex_derisk import double_center, WINDOW, EMA_ALPHA  # noqa: E402
# The metric helpers (nearest-category coherence, cosine-similarity, Pearson-vs-Strue, effective rank):
from research.runners.dendritic_d1_learn_graded_structure_derisk import (  # noqa: E402
    _cos_sim, _pearson_vs_Strue, heldout_generalization, effective_rank,
)
# The a-priori PROBE taxonomy (evaluation yardstick ONLY -- never labels the discovered vocab) + the stoplist:
from research.runners.option_c_real_cooccurrence_derisk import TAXONOMY_8x8  # noqa: E402
from research.runners.option_c_stageB_fair_test import STOPLIST  # noqa: E402
# The streaming corpus loader (bounded-memory, multi-file capable):
from research.runners.corpus_stream import load_token_stream_multi  # noqa: E402
# The host PPMI ceiling (the load-bearing DATA_BOUND-vs-DEGRADES disambiguator -- labelled, NOT a deliverable):
from research.runners.learned_graded_cortex_fair_test import ppmi_matrix  # noqa: E402


DEFAULT_CORPUS = os.path.join(_REPO, "research", "datasets", "distill_corpus.txt")
N_HUB = 500            # number of context hubs (frequent context words), mirrors the stream cortex
MIN_WORD_LEN = 3       # drop 1-2 char tokens (a/an/is-style residue not already in the stoplist)


# ============================================================================================================
# SYNTHETIC-BROAD corpus: a POSITIVE CONTROL that isolates the SCALING question from the data-sparsity limit.
# Builds a corpus where the top-K discovered vocab is `K` concept words, organized into semantic categories
# with genuine SHARED-CONTEXT co-occurrence, ALL well-attested (each concept appears >= min_count times). This
# lets the de-risk answer the pure mechanism question: does the emergent stream SCALE when the DATA is
# adequate? (The concept words are named c_<cat>_<i>; the probe taxonomy is BUILT from the same grid so the
# metric applies without the a-priori TAXONOMY_8x8. NOT a real-corpus claim -- a mechanism scaling control.)
# ============================================================================================================
def build_synthetic_broad_corpus(K, seed, n_categories=8, occ_per_word=60, hubs_per_cat=None,
                                  n_noise_hubs=None):
    """Return (stories: list[list[str]], synthetic_taxonomy: dict). `K` concept words split into
    `n_categories` categories; each category has DISTINCTIVE shared-context hubs that carry the co-occurrence
    signal (the within-category words all co-occur with their category's hubs). Each concept word appears
    ~`occ_per_word` times (well-attested at every K). A pool of shared NOISE hubs adds non-category context.

    Discovery must pick the CONCEPT words for the top-K vocab (not the hubs), so `hubs_per_cat` scales with the
    category size: with `per_cat` concept words each appearing ~occ_per_word times, and a scene emitting ~3
    hubs, per-hub frequency ~= per_cat*occ_per_word*3/hubs_per_cat. Setting hubs_per_cat >= per_cat keeps each
    hub RARER than each concept word, so the concept words rank above the hubs in frequency discovery."""
    rng = np.random.RandomState(seed * 131 + 5)
    per_cat = max(2, K // n_categories)
    if hubs_per_cat is None:
        hubs_per_cat = max(6, 2 * per_cat)     # enough distinct hubs that each hub stays rarer than a concept
    if n_noise_hubs is None:
        # a LARGE distractor pool so each noise token stays RARE (frequent noise would out-rank concepts and
        # capture the discovered top-K). Scales with K so discovery always resolves to the concept words.
        n_noise_hubs = max(2000, 20 * K)
    K_eff = per_cat * n_categories
    taxonomy = {}
    cat_words, cat_hubs = [], []
    for ci in range(n_categories):
        words = [f"c{ci:02d}w{wi:03d}" for wi in range(per_cat)]
        hubs = [f"h{ci:02d}k{hi:03d}" for hi in range(hubs_per_cat)]
        taxonomy[f"cat{ci}"] = words
        cat_words.append(words)
        cat_hubs.append(hubs)
    noise = [f"z{i:04d}" for i in range(n_noise_hubs)]
    # Build per-word SCENES (concept + same-category mates + that category's hubs + a noise token). A scene is
    # a LOCAL co-occurrence bundle -- the mechanism's signal is that a concept sits in a window with its
    # category's hubs. Then CONCATENATE several scenes (from DIFFERENT categories) into each long "story", so
    # within-story token-shuffling genuinely DESTROYS the local windows (the scramble anti-cheat bites -- as it
    # does on real multi-topic prose). A single-scene story would be scramble-invariant (the whole story is one
    # category), which is why the earlier flat version failed its own control.
    scenes = []
    for ci in range(n_categories):
        for w in cat_words[ci]:
            for _ in range(occ_per_word):
                mates = list(rng.choice(cat_words[ci], size=min(2, per_cat), replace=False))
                khs = list(rng.choice(cat_hubs[ci], size=min(hubs_per_cat, 3), replace=False))
                nz = list(rng.choice(noise, size=1))
                scene = [w] + mates + khs + nz
                rng.shuffle(scene)
                scenes.append(scene)
    rng.shuffle(scenes)
    stories = []
    scenes_per_story = 6            # >1 category per story -> within-story shuffle destroys local co-occurrence
    for i in range(0, len(scenes), scenes_per_story):
        story = []
        for sc in scenes[i:i + scenes_per_story]:
            story.extend(sc)
            story.append(rng.choice(noise))   # ONE inter-scene noise separator (kept rare via a big pool)
        stories.append(story)
    rng.shuffle(stories)
    return stories, taxonomy


# ============================================================================================================
# (1) DISCOVER the vocab from the corpus = top-K most-frequent CONTENT words. Emergent, not hand-assigned.
# ============================================================================================================
def discover_vocab(stories, K, extra_exclude=frozenset()):
    """Return the top-K most-frequent CONTENT words (minus the stoplist + very short tokens + any
    extra_exclude) -- the vocab DISCOVERED from the corpus by pure frequency. This is emergent: no taxonomy,
    no hand list; the corpus's own word-frequency ranking picks the vocab."""
    gfreq = Counter()
    for toks in stories:
        gfreq.update(toks)
    vocab = []
    for w, _ in gfreq.most_common():
        if w in STOPLIST or w in extra_exclude or len(w) < MIN_WORD_LEN:
            continue
        vocab.append(w)
        if len(vocab) >= K:
            break
    return vocab, gfreq


# ============================================================================================================
# (2) LEARN the co-occurrence/semantic structure via the EXISTING emergent stream mechanism.
#     ONLINE Hebbian co-occurrence in a WM window + running per-hub frequency + log-double-center.
#     NO global PPMI matrix; the cortex HEARS the stream word-by-word (CYCLE-93 owner reframe).
#     Byte-faithful to `_phaseB_online_stream_cortex_derisk.run_seed`, generalized to any discovered vocab.
# ============================================================================================================
def learn_stream_codes(seed, stories, vocab, hubs, scramble=False, freq_only=False, window=WINDOW):
    """Learn concept codes for `vocab` from the token `stories` via the online stream mechanism.

    - hubs: the context words the association weights are learned against (the cortex's context basis).
    - scramble: if True, shuffle the token order WITHIN each story before streaming (destroys the real
      windowed co-occurrence while keeping the exact unigram frequency -> the anti-cheat).
    - freq_only: if True, the code = the (double-centered log) OUTER PRODUCT of the target's running frequency
      with the hub running frequency (no genuine co-occurrence) -> the frequency-only baseline.
    Returns (codes [Nt x N_HUB], n_updates).
    """
    rng = np.random.RandomState(seed)
    targets = list(vocab)
    target_set = set(targets)
    Nt = len(targets)
    hub_idx = {w: i for i, w in enumerate(hubs)}
    n_hub = len(hubs)
    keep = target_set | set(hubs)
    tgt_row = {w: i for i, w in enumerate(targets)}

    M = np.zeros((Nt, n_hub), dtype=np.float64)     # the LEARNED cortex (synaptic weights), built ONLINE
    freq = np.zeros(n_hub, dtype=np.float64)        # running per-hub frequency EMA (online normalization)
    n_updates = 0
    story_order = rng.permutation(len(stories))
    for si in story_order:
        kept = [t for t in stories[si] if t in keep]
        if scramble:
            kept = list(kept)
            rng.shuffle(kept)                        # destroy co-occurrence; unigram frequency preserved
        for c in range(len(kept)):
            w = kept[c]
            lo, hi = max(0, c - window), min(len(kept), c + window + 1)
            ctx = set(kept[lo:hi]) - {w}
            for u in kept[lo:hi]:
                if u in hub_idx:
                    freq[hub_idx[u]] += EMA_ALPHA * (1.0 - freq[hub_idx[u]])
            if w in target_set and not freq_only:
                for u in ctx:
                    if u in hub_idx:
                        M[tgt_row[w], hub_idx[u]] += 1.0
                        n_updates += 1

    if freq_only:
        # frequency-only: each target's per-hub "association" = its running frequency x the hub frequency
        # (a rank-1 marginal outer product -- NO real co-occurrence). Must be ~chance.
        tfreq = np.zeros(Nt, dtype=np.float64)
        for si in story_order:
            for t in stories[si]:
                if t in target_set:
                    tfreq[tgt_row[t]] += 1.0
        tfreq /= (tfreq.sum() + 1e-12)
        M = np.outer(tfreq, freq + 1e-9)
        n_updates = Nt

    code = double_center(np.log1p(M * 100.0))       # the validated log-domain normalization
    return code, n_updates


def batch_count_matrix(stories, vocab, hubs, window=WINDOW):
    """The BATCH target x hub co-occurrence COUNT matrix over the SAME window (for the host PPMI ceiling).
    NOT the emergent mechanism -- a labelled ceiling ONLY, the DATA_BOUND-vs-DEGRADES disambiguator."""
    targets = list(vocab)
    target_set = set(targets)
    hub_idx = {w: i for i, w in enumerate(hubs)}
    tgt_row = {w: i for i, w in enumerate(targets)}
    keep = target_set | set(hubs)
    C = np.zeros((len(targets), len(hubs)), dtype=np.float64)
    for toks in stories:
        kept = [t for t in toks if t in keep]
        for c in range(len(kept)):
            w = kept[c]
            if w not in target_set:
                continue
            lo, hi = max(0, c - window), min(len(kept), c + window + 1)
            for u in set(kept[lo:hi]) - {w}:
                if u in hub_idx:
                    C[tgt_row[w], hub_idx[u]] += 1.0
    return C


# ============================================================================================================
# (3) MEASURE the emergent structure via the PROBE taxonomy (evaluation yardstick only).
# ============================================================================================================
def build_probe(vocab, taxonomy):
    """The PROBE = the a-priori-category words that HAPPEN to appear in the DISCOVERED vocab, with categories
    that retain >= 2 such words (a nearest-neighbour coherence needs >=2 same-category members). Returns
    (probe_rows [into vocab], probe_labels [category ids], probe_words, n_categories, per_cat_counts).
    The vocab is NEVER filtered/reordered by this -- the probe just marks which vocab rows are evaluable."""
    row_of = {w: i for i, w in enumerate(vocab)}
    cat_names = list(taxonomy.keys())
    rows, labels, words, per_cat = [], [], [], {}
    for cid, cname in enumerate(cat_names):
        present = [w for w in taxonomy[cname] if w in row_of]
        if len(present) >= 2:
            for w in present:
                rows.append(row_of[w])
                labels.append(cid)
                words.append(w)
            per_cat[cname] = len(present)
    return (np.asarray(rows, dtype=int), np.asarray(labels, dtype=int), words,
            len(set(labels)), per_cat)


def _probe_min_count(stories, probe_words):
    """Min unigram count among the probe words (a data-sparsity read: if the rarest probe word appears only a
    handful of times, its co-occurrence profile is noise, not structure)."""
    if not probe_words:
        return 0
    cnt = Counter()
    pw = set(probe_words)
    for toks in stories:
        for t in toks:
            if t in pw:
                cnt[t] += 1
    return int(min(cnt.get(w, 0) for w in probe_words))


def probe_structure(codes, probe_rows, probe_labels):
    """Nearest-category coherence + within/between margin + Pearson(cos, S_true_probe) over the PROBE words
    embedded within the full-K co-occurrence codes. `codes` is the FULL K-vocab embedding; we index the probe
    rows out of it (so the probe words see the WHOLE K-vocab context -- the scaling question)."""
    if probe_rows.size < 4 or len(set(probe_labels.tolist())) < 2:
        return None
    sub = codes[probe_rows]                                  # probe words' codes (embedded in the full K-vocab)
    coherence, chance = heldout_generalization(sub, probe_labels)
    sim = _cos_sim(sub)
    Np = sub.shape[0]
    same = probe_labels[:, None] == probe_labels[None, :]
    iu = np.triu_indices(Np, k=1)
    within = sim[iu][same[iu]]
    between = sim[iu][~same[iu]]
    within_m = float(within.mean()) if within.size else 0.0
    between_m = float(between.mean()) if between.size else 0.0
    S_true = same.astype(np.float64)
    pearson = _pearson_vs_Strue(sim, S_true)
    return {"coherence": float(coherence), "chance": float(chance),
            "within_cos": within_m, "between_cos": between_m, "margin": within_m - between_m,
            "pearson_vs_Strue": float(pearson), "n_probe": int(Np),
            "n_probe_categories": int(len(set(probe_labels.tolist()))),
            "eff_rank": float(effective_rank(sub))}


# ============================================================================================================
# Per-(seed, K) driver
# ============================================================================================================
def run_seed_K(seed, K, stories, taxonomy, args):
    # STEP 1: discover the vocab (top-K content words) -- emergent, corpus-frequency only.
    vocab, gfreq = discover_vocab(stories, K)
    if len(vocab) < K:
        print(f"    [WARN K={K}] corpus yields only {len(vocab)} content words < K={K} "
              f"(the corpus is too small for this K)", flush=True)
    # context hubs = the top-N_HUB frequent CONTENT words NOT in the target vocab (the context basis).
    target_set = set(vocab)
    hubs = []
    for w, _ in gfreq.most_common():
        if w in STOPLIST or w in target_set or len(w) < MIN_WORD_LEN:
            continue
        hubs.append(w)
        if len(hubs) >= N_HUB:
            break

    # STEP 3-prep: the PROBE (a-priori-category words present in the discovered vocab).
    probe_rows, probe_labels, probe_words, n_probe_cat, per_cat = build_probe(vocab, taxonomy)

    # STEP 2: LEARN the co-occurrence codes (the emergent stream mechanism).
    codes, n_upd = learn_stream_codes(seed, stories, vocab, hubs, window=args.window)
    learned = probe_structure(codes, probe_rows, probe_labels)

    # ANTI-CHEAT A: scrambled-corpus (co-occurrence destroyed, unigram frequency preserved).
    codes_scr, _ = learn_stream_codes(seed, stories, vocab, hubs, scramble=True, window=args.window)
    scrambled = probe_structure(codes_scr, probe_rows, probe_labels)

    # ANTI-CHEAT B: frequency-only baseline (no genuine co-occurrence).
    codes_fo, _ = learn_stream_codes(seed, stories, vocab, hubs, freq_only=True, window=args.window)
    freq_only = probe_structure(codes_fo, probe_rows, probe_labels)

    # HOST CEILING: batch PPMI over the SAME window/vocab/hubs. The DATA_BOUND-vs-DEGRADES disambiguator --
    # if the host ALSO can't recover probe-structure at this K, the corpus is too sparse (DATA_BOUND); if the
    # host CAN but the emergent stream can't, that's the mechanism gap (DEGRADES). Labelled, NOT a deliverable.
    C = batch_count_matrix(stories, vocab, hubs, window=args.window)
    host_codes = ppmi_matrix(C, 0.75)               # PPMI rows = per-target profile over the hubs
    host = probe_structure(host_codes, probe_rows, probe_labels)

    row = {"K": K, "n_vocab": len(vocab), "n_hubs": len(hubs), "n_updates": int(n_upd),
           "n_probe_words": int(probe_rows.size), "n_probe_categories": int(n_probe_cat),
           "per_cat_probe_counts": per_cat, "vocab_head": vocab[:20],
           "probe_min_count": _probe_min_count(stories, probe_words),
           "learned": learned, "scrambled": scrambled, "freq_only": freq_only, "host": host}

    if learned is None:
        print(f"  [K={K}] vocab={len(vocab)} hubs={len(hubs)} updates={n_upd} | PROBE too thin "
              f"(only {probe_rows.size} probe words in {n_probe_cat} cats) -> not evaluable at this K",
              flush=True)
        return row

    scr_c = scrambled["coherence"] if scrambled else float("nan")
    fo_c = freq_only["coherence"] if freq_only else float("nan")
    host_c = host["coherence"] if host else float("nan")
    print(f"  [K={K}] vocab={len(vocab)} hubs={len(hubs)} updates={n_upd} | "
          f"probe={probe_rows.size}w/{n_probe_cat}cat (chance {learned['chance']:.3f}, "
          f"rarest-probe-count {row['probe_min_count']})", flush=True)
    print(f"      LEARNED   coherence={learned['coherence']:.3f}  margin={learned['margin']:+.3f}  "
          f"Pearson(cos,S)={learned['pearson_vs_Strue']:+.3f}  (within {learned['within_cos']:+.3f} / "
          f"between {learned['between_cos']:+.3f}, eff-rank {learned['eff_rank']:.1f})", flush=True)
    print(f"      SCRAMBLED coherence={scr_c:.3f}  (co-occurrence destroyed -> should collapse to chance)",
          flush=True)
    print(f"      FREQ-ONLY coherence={fo_c:.3f}  (no co-occurrence -> should be chance)", flush=True)
    print(f"      HOST-PPMI coherence={host_c:.3f}  (batch ceiling: does the DATA carry structure at this K?)",
          flush=True)
    return row


# ============================================================================================================
# Scaling verdict
# ============================================================================================================
def decide_verdict(per_seed, seeds, vocab_sizes, args):
    """1-seed smoke: characterize the scaling shape + the anti-cheat collapse. (Multi-seed uses the same
    logic; the controller aggregates.)"""
    # gather per-K learned coherence + anti-cheat, averaged over seeds where the K was evaluable.
    def mean_over_seeds(K, path):
        vals = []
        for s in seeds:
            for r in per_seed[str(s)]:
                if r["K"] == K and r.get(path[0]) is not None:
                    d = r
                    ok = True
                    for k in path:
                        if d is None:
                            ok = False
                            break
                        d = d[k]
                    if ok and d is not None:
                        vals.append(d)
        return float(np.mean(vals)) if vals else None

    k_learned = {K: mean_over_seeds(K, ["learned", "coherence"]) for K in vocab_sizes}
    k_scr = {K: mean_over_seeds(K, ["scrambled", "coherence"]) for K in vocab_sizes}
    k_fo = {K: mean_over_seeds(K, ["freq_only", "coherence"]) for K in vocab_sizes}
    k_host = {K: mean_over_seeds(K, ["host", "coherence"]) for K in vocab_sizes}
    # MARGIN (within - between cosine) is the ROBUST discriminator -- it does NOT saturate the way the
    # nearest-of-C-category coherence does when the probe has few categories but many words. The anti-cheat
    # collapse + the scaling "holds" checks gate on the margin; coherence is a reported diagnostic.
    k_margin = {K: mean_over_seeds(K, ["learned", "margin"]) for K in vocab_sizes}
    k_scr_margin = {K: mean_over_seeds(K, ["scrambled", "margin"]) for K in vocab_sizes}
    k_fo_margin = {K: mean_over_seeds(K, ["freq_only", "margin"]) for K in vocab_sizes}
    k_host_margin = {K: mean_over_seeds(K, ["host", "margin"]) for K in vocab_sizes}
    k_chance = {K: mean_over_seeds(K, ["learned", "chance"]) for K in vocab_sizes}
    evaluable = [K for K in vocab_sizes if k_learned[K] is not None]

    detail = {"per_K_learned_coherence": k_learned, "per_K_scrambled_coherence": k_scr,
              "per_K_freq_only_coherence": k_fo, "per_K_host_ppmi_coherence": k_host,
              "per_K_learned_margin": k_margin, "per_K_scrambled_margin": k_scr_margin,
              "per_K_freq_only_margin": k_fo_margin, "per_K_host_margin": k_host_margin,
              "per_K_chance": k_chance, "evaluable_K": evaluable}

    if not evaluable:
        return "DATA_BOUND", ("no K was evaluable -- the corpus is too small/narrow to populate even the "
                              "smallest vocab with enough probe words. A bigger corpus is the prerequisite."), detail

    K0 = evaluable[0]
    base = k_margin[K0]                                  # the ROBUST base structure signal = the margin
    margin_bar = args.margin_bar                         # a margin above this = real category structure
    # scramble collapses at every evaluable K? (learned margin clearly above the scrambled margin + above bar)
    scramble_collapses = all(
        (k_margin[K] is not None and k_scr_margin.get(K) is not None
         and k_margin[K] - k_scr_margin[K] >= args.margin_collapse_gap
         and k_margin[K] >= margin_bar)
        for K in evaluable)
    beats_freq_only = all(
        (k_fo_margin.get(K) is not None and k_margin[K] - k_fo_margin[K] >= args.margin_collapse_gap)
        for K in evaluable)
    # the largest evaluable K holds >= hold_frac of the smallest-K MARGIN (structure did not dilute)?
    K_big = evaluable[-1]
    holds = (base is not None and base > margin_bar and
             k_margin[K_big] is not None and
             k_margin[K_big] >= args.hold_frac * base)

    # is the learned structure even real at the base K (margin above bar + scramble/freq-only collapse there)?
    real_at_base = (base is not None and base >= margin_bar
                    and (k_scr_margin.get(K0) is not None and base - k_scr_margin[K0] >= args.margin_collapse_gap)
                    and (k_fo_margin.get(K0) is not None and base - k_fo_margin[K0] >= args.margin_collapse_gap))

    # the HOST-ceiling disambiguator at the largest K: does the DATA carry structure there at all? (margin)
    host_base = k_host_margin.get(K0)
    host_big = k_host_margin.get(K_big)
    host_holds_at_big = (host_big is not None and host_big >= margin_bar)
    # rarest-probe-count at the largest evaluable K (the direct data-sparsity read)
    rarest_big = None
    for s in seeds:
        for r in per_seed[str(s)]:
            if r["K"] == K_big:
                rarest_big = r.get("probe_min_count")
    detail["host_holds_at_largest_K"] = bool(host_holds_at_big)
    detail["rarest_probe_count_at_largest_K"] = rarest_big

    big_margin = k_margin[K_big]
    if not real_at_base:
        verdict = "DATA_BOUND"
        why = (f"at the smallest evaluable K={K0} the learned within-vs-between MARGIN ({base:+.3f}) is not "
               f"clearly above the structure bar ({margin_bar}) with scramble/freq-only collapse -- the corpus "
               f"co-occurrence is too sparse to establish the structure even at the base vocab (host-PPMI margin "
               f"{host_base}). A bigger/broader corpus is the prerequisite before the scaling question can be "
               f"answered (a DATA lever, not yet demonstrated).")
    elif holds and scramble_collapses and beats_freq_only:
        verdict = "SCALES"
        why = (f"the emergent structure MARGIN HOLDS from K={K0} ({base:+.3f}) to K={K_big} ({big_margin:+.3f}) "
               f"(>= {args.hold_frac:.0%} of the base retained), the scramble control collapses the margin at "
               f"every K, and it beats frequency-only -> the structure is genuinely learned from real "
               f"co-occurrence AND does not dilute as the vocab grows ==> open-domain BREADTH is a cheap "
               f"DATA/SCALE lever (the controller runs a bigger corpus + multi-seed).")
    elif not host_holds_at_big:
        # the STREAM degraded by K_big -- BUT so did the host PPMI ceiling. The DATA is too sparse at K_big
        # (the newly-included probe words are ultra-rare), NOT a mechanism dilution. => DATA_BOUND.
        verdict = "DATA_BOUND"
        why = (f"the learned structure margin is real at K={K0} ({base:+.3f}) but degrades by K={K_big} "
               f"({big_margin:+.3f}) -- HOWEVER the host-PPMI ceiling ALSO fails to carry margin at K={K_big} "
               f"(host margin {host_big}; rarest-probe-count {rarest_big}). The newly-included larger-K probe "
               f"words are ULTRA-RARE in this corpus, so the co-occurrence is genuinely too sparse there -- this "
               f"is a DATA limit, NOT a mechanism dilution. The scaling question needs a bigger/broader corpus "
               f"(where the top-K words are all well-attested) to be answered. On THIS corpus the emergent "
               f"structure is demonstrated only up to the well-attested vocab.")
    else:
        # the host PPMI ceiling HOLDS at K_big but the emergent STREAM does not -> a genuine MECHANISM gap.
        verdict = "DEGRADES"
        why = (f"the learned structure margin is real at K={K0} ({base:+.3f}, scramble/freq-only collapse) BUT "
               f"the emergent stream's margin DEGRADES by K={K_big} ({big_margin:+.3f}; holds={holds}, "
               f"scramble_collapses={scramble_collapses}, beats_freq_only={beats_freq_only}) WHILE the host-PPMI "
               f"ceiling STILL carries structure there (host margin {host_big} >= bar {margin_bar}) -> the DATA "
               f"has the structure but the emergent mechanism dilutes the semantic neighbourhoods as the vocab "
               f"grows ==> a MECHANISM gap; dispatch a research gate (hierarchical / sparse-distributed capacity "
               f"for large emergent vocab).")
    return verdict, why, detail


def main():
    p = argparse.ArgumentParser(description="Emergent-vocab breadth SCALE de-risk (discover vocab from a real "
                                            "corpus, learn co-occurrence via the emergent stream cortex, "
                                            "measure whether the structure holds as the vocab grows).")
    p.add_argument("--seeds", default="42", help="comma-separated seeds (smoke: 42)")
    p.add_argument("--vocab-sizes", default="64,256,1024", help="comma-separated K values to scale over")
    p.add_argument("--corpus-path", default=None,
                   help="corpus file(s), comma/pathsep-separated (default: research/datasets/distill_corpus.txt; "
                        "the controller points this at a bigger corpus for the decisive run)")
    p.add_argument("--max-stories", type=int, default=None,
                   help="cap the number of stories materialized (bounds RAM on a huge corpus)")
    p.add_argument("--synthetic-broad", action="store_true",
                   help="POSITIVE CONTROL: instead of a real corpus, build a synthetic broad corpus per K where "
                        "ALL top-K discovered words are well-attested concept words with genuine shared-context "
                        "co-occurrence. Isolates the pure MECHANISM SCALING question from the data-sparsity "
                        "limit -- answers 'does the emergent stream scale when the DATA is adequate?'")
    p.add_argument("--syn-occ-per-word", type=int, default=60,
                   help="(synthetic-broad) occurrences per concept word (attestation level)")
    p.add_argument("--window", type=int, default=WINDOW, help="co-occurrence WM window (+- tokens)")
    # verdict bars (the MARGIN is the robust discriminator; coherence is a reported diagnostic)
    p.add_argument("--hold-frac", type=float, default=0.60,
                   help="the largest-K MARGIN must be >= this fraction of the smallest-K's (structure holds)")
    p.add_argument("--margin-bar", type=float, default=0.03,
                   help="within-vs-between cosine margin above this = real category structure")
    p.add_argument("--margin-collapse-gap", type=float, default=0.02,
                   help="min (learned - control) MARGIN gap for a control to count as collapsed")
    p.add_argument("--collapse-gap", type=float, default=0.10,
                   help="(legacy; unused by the margin-based verdict) coherence gap")
    p.add_argument("--above-chance", type=float, default=0.10,
                   help="learned coherence must exceed chance by at least this (diagnostic reporting only)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    os.environ.setdefault("SIM_BACKEND", "numpy")
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    vocab_sizes = [int(k.strip()) for k in args.vocab_sizes.split(",")]
    corpus_arg = args.corpus_path if args.corpus_path is not None else DEFAULT_CORPUS

    t0 = time.time()
    mode = "SYNTHETIC-BROAD (mechanism scaling control)" if args.synthetic_broad else "REAL corpus"
    print(f"[emergent-vocab breadth SCALE de-risk] seeds={seeds} vocab_sizes={vocab_sizes} "
          f"window=+-{args.window} mode={mode}", flush=True)
    print(f"  mechanism: DISCOVER vocab (top-K content words) -> LEARN online Hebbian co-occurrence (stream "
          f"cortex, NO global PPMI) -> MEASURE probe-taxonomy coherence as K grows.", flush=True)

    stories_full = None
    if not args.synthetic_broad:
        print(f"  corpus={corpus_arg}", flush=True)
        # load the real corpus once (the stream mechanism permutes stories per-seed internally).
        stories_full = load_token_stream_multi(corpus_arg, max_stories=args.max_stories)
        total_toks = sum(len(s) for s in stories_full)
        uniq = len(set(w for s in stories_full for w in s))
        print(f"  loaded {len(stories_full)} stories, {total_toks} tokens, {uniq} unique words "
              f"({'small -- risk noted' if total_toks < 500_000 else 'ok'})", flush=True)
    else:
        total_toks, uniq = None, None
        print(f"  synthetic-broad: {args.syn_occ_per_word} occ/concept-word, all top-K words well-attested "
              f"(the mechanism-scaling positive control; NOT a real-corpus claim).", flush=True)

    per_seed = {}
    for s in seeds:
        print(f"\n{'='*96}\n  SEED {s}\n{'='*96}", flush=True)
        rows = []
        for K in vocab_sizes:
            if args.synthetic_broad:
                # a fresh well-attested corpus + its OWN grid taxonomy per K (the probe = the same grid).
                stories, taxonomy = build_synthetic_broad_corpus(
                    K, s, occ_per_word=args.syn_occ_per_word)
            else:
                stories, taxonomy = stories_full, TAXONOMY_8x8
            rows.append(run_seed_K(s, K, stories, taxonomy, args))
        per_seed[str(s)] = rows

    verdict, why, detail = decide_verdict(per_seed, seeds, vocab_sizes, args)

    def _f(v):
        return f"{v:+.3f}" if v is not None else "   n/a"
    print(f"\n{'='*96}", flush=True)
    print(f"  SCALING -- within-vs-between MARGIN vs K (the robust structure signal; the verdict gates on this):",
          flush=True)
    for K in vocab_sizes:
        print(f"    K={K:5d}:  learned={_f(detail['per_K_learned_margin'].get(K))}  "
              f"scrambled={_f(detail['per_K_scrambled_margin'].get(K))}  "
              f"freq-only={_f(detail['per_K_freq_only_margin'].get(K))}  "
              f"host-PPMI={_f(detail['per_K_host_margin'].get(K))}", flush=True)
    print(f"  SCALING -- nearest-category COHERENCE vs K (diagnostic; saturates at large probe sizes):",
          flush=True)
    for K in vocab_sizes:
        lc = detail["per_K_learned_coherence"].get(K)
        sc = detail["per_K_scrambled_coherence"].get(K)
        fo = detail["per_K_freq_only_coherence"].get(K)
        ho = detail["per_K_host_ppmi_coherence"].get(K)
        ch = detail["per_K_chance"].get(K)
        s_lc = f"{lc:.3f}" if lc is not None else "  n/a"
        s_sc = f"{sc:.3f}" if sc is not None else "  n/a"
        s_fo = f"{fo:.3f}" if fo is not None else "  n/a"
        s_ho = f"{ho:.3f}" if ho is not None else "  n/a"
        s_ch = f"{ch:.3f}" if ch is not None else "  n/a"
        print(f"    K={K:5d}:  learned={s_lc}  scrambled={s_sc}  freq-only={s_fo}  host-PPMI={s_ho}  "
              f"(chance {s_ch})", flush=True)
    print(f"\n  OVERALL VERDICT: {verdict}\n  {why}", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*96}\n", flush=True)

    out = {"verdict": verdict, "why": why, "seeds": seeds, "vocab_sizes": vocab_sizes,
           "mode": "synthetic_broad" if args.synthetic_broad else "real_corpus",
           "corpus": None if args.synthetic_broad else corpus_arg,
           "n_stories": None if args.synthetic_broad else len(stories_full),
           "n_tokens": total_toks, "n_unique": uniq,
           "config": {"window": args.window, "n_hub": N_HUB, "hold_frac": args.hold_frac,
                      "collapse_gap": args.collapse_gap, "above_chance": args.above_chance,
                      "syn_occ_per_word": args.syn_occ_per_word if args.synthetic_broad else None},
           "detail": detail, "per_seed": per_seed,
           "note": ("Emergent-vocab breadth scale de-risk: the vocab is DISCOVERED from the corpus (top-K "
                    "content words by frequency), the co-occurrence structure LEARNED via the emergent online "
                    "stream cortex (Hebbian WM-window co-occurrence + running-freq + log-double-center, NO "
                    "global PPMI matrix). The probe taxonomy is an EVALUATION YARDSTICK ONLY (it never labels "
                    "the discovered vocab). Anti-cheats: scrambled-corpus (co-occurrence destroyed, unigram "
                    "frequency preserved) + frequency-only baseline. NO sim/ edits; reuse-by-import.")}
    if args.out is None:
        raw_dir = os.path.join(_REPO, "research", "findings", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.out = os.path.join(raw_dir, f"_emergent_vocab_breadth_scale_{ts}.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)
    return out


if __name__ == "__main__":
    main()
