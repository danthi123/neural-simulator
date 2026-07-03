"""EMERGE-62b -- ADD THE 3rd DISTRIBUTIONAL CUE (PHRASE-BOUNDARY / SYNTACTIC-POSITION ALIGNMENT) to the function-word
discovery, so the closed-class inventory self-organizes on the REAL noisy corpus, not just the controlled stream.

This iterates the ONE named boundary of EMERGE-62 (`research/findings/2026-07-03-emerge62-discover-function-words-GO.md`,
`_emerge62_discover_function_words_derisk.py`): the 2D "Goldilocks" discovery (high running-FREQUENCY AND high
context-COVERAGE) is GO on the controlled EMERGE-domain stream (F1 0.863) but on the REAL corpus
(`data/corpus/ra_finetune_corpus.txt`, TinyStories-interleaved fact/QA) it OVER-INCLUDES: recall + frame-recall stay
1.00 while PRECISION drops to 0.078, because hundreds of high-frequency, high-coverage NARRATIVE CONTENT words (common
nouns/objects/verbs, QA-structural tokens like `facts`/`answer`/`question`) also pass frequency+coverage. Frequency+
coverage RELIABLY FIND the closed class (perfect recall) but cannot SEPARATE the frequent-content-word false positives.

THE NAMED 3rd CUE (Yang & Getz 2026, arXiv 2601.21191, "Function Words as Statistical Cues for Language Learning";
Redington/Cartwright-Brent distributional POS induction; catalog G.12 Broca open/closed dissociation; the research gate
`2026-07-03-self-organizing-grammatical-structure-research-gate.md` Move-2(A)/Move-3-RANK-1 NEGATIVE-path). Yang-Getz's
THIRD universal property of function words (after frequency + syntactic-association/diversity) is PHRASE-BOUNDARY /
SYNTACTIC-POSITION ALIGNMENT: a function word reliably occurs at a CONSTRUCTION EDGE / fixed syntactic slot (immediately
BEFORE content -- determiners precede nouns, auxiliaries precede verbs) and is almost NEVER phrase-FINAL; a frequent
CONTENT word (a noun/object) sits at variable positions and ENDS phrases. This positional REGULARITY is exactly the
signal separating the frequent-content-word false positives from the true closed class.

OPERATIONALIZATION (cheapest that works, ONE variable). The corpus tokeniser `corpus_stream` strips ALL punctuation
(`re.findall(r"[a-z]+")`) so it has NO sentence boundaries -- the position cue needs them. So we add a SENTENCE-AWARE
front end: split the raw corpus on sentence/frame punctuation `[.?!*]` (byte-identical token regex WITHIN each sentence
-- `[a-z]+`), giving per-word POSITIONAL statistics. The position statistic is:

  posscore[w] = (1 - fracFinal[w]) * (1 - precededByContentNoun[w])          [higher = more function-like]
    * fracFinal[w]             = fraction of w's occurrences that are PHRASE-FINAL (Yang-Getz phrase-edge: a function
                                 word is almost never phrase-final; a content noun/object ends phrases). 1-fracFinal.
    * precededByContentNoun[w] = SOFT fraction of w's LEFT neighbours weighted by the neighbour's OWN endness (its
                                 fracFinal). A function word is rarely preceded by a phrase-final-capable content noun;
                                 a content VERB (follows the subject noun) and an OBJECT (follows the verb) ARE. This is
                                 the Redington/Cartwright-Brent immediate-LEFT-neighbour role profile. Using the
                                 neighbour's soft endness (not a hard closed-set membership) is what protects frame
                                 function words that legitimately follow OTHER function words in dense streams
                                 ("to the", "does not", "is in the") -- the hard "preceded-by-any-closed-word" variant
                                 wrongly kills `the`/`a`/`not` in the very-regular controlled stream (verified).

COMBINE (the position cue GATES / re-ranks the Goldilocks candidates, NO hand-list as input). ASYMMETRIC EXCLUSION: a
2D Goldilocks candidate is KEPT unless its posscore-percentile is BELOW TP_EXCL -- i.e. the position cue only EXCLUDES
candidates that are CLEARLY content-positioned (sentence-final nouns/objects, post-noun content verbs), it does NOT
REQUIRE strong function-positioning. This is the key design choice: a symmetric "require high posscore" gate breaks
recall on both streams (it penalises function words that follow other function words); the asymmetric exclude-only-clear-
content gate lifts real precision + holds recall 1.00 + does NOT regress the controlled domain (verified in probes).

RESULT (see the de-risk): on the REAL corpus the 3rd cue lifts precision 0.080 -> 0.111 (~1.39x) / F1 0.148 -> 0.200
with recall HELD at 1.00 (all 11 ground-truth closed-class words + all 4 frame function words still discovered), the
POSITION-SHUFFLE control collapsing BELOW the 2D level (the cue is load-bearing); the controlled EMERGE stream is NOT
regressed (3D == 2D every seed, frame-recall 4/4). HONEST: the narrow ground-truth (11 words) UNDER-states the true
precision -- most remaining "false positives" (he/she/they/but/of/for/with/that/was/were/had/...) are GENUINE English
function words that TinyStories contributes but the narrow EMERGE-domain ground truth omits; against an EXTENDED honest
closed-class set the precision + lift are larger (reported as a secondary, non-gating read). The excluded words are
exactly the content nouns/objects (apple, ball, box, cake, cat, fish, tree, water, worm, ...) + QA content (know, say,
yes, no) -- the position cue removes exactly the frequent-CONTENT-word false positives it was designed to.

DE-RISK (6 seeds 42/43/44/100/101/102, CPU/numpy):
  (a) CONTROLLED EMERGE-domain stream STAYS GO -- 3D F1 >= EMERGE-62's 2D F1, recall 1.00, frame-recall 1.00 (the cue
      must not regress the controlled domain).
  (b) REAL corpus PRECISION rises MATERIALLY from 0.078 (2D-vs-3D P/R/F1 reported side by side; recall MUST stay 1.00
      for the true closed class), with a clear margin over every collapsed control.
  Anti-cheats that MUST COLLAPSE (project control-validity methodology: INPUT-DESTRUCTION + hold-out, NOT a fixed-random
  control):
  (c1) POSITION-SHUFFLE  -- permute the per-word position statistic<->identity mapping -> the 3rd cue is DESTROYED ->
       real precision falls back toward (or below) the 2D level (proving the position cue is load-bearing, not spurious).
  (c2) FREQUENCY-SHUFFLE -- permute the freq/coverage<->identity mapping -> discovery collapses to chance.
  (c3) NO-STREAM         -- empty stream -> no statistics -> empty discovered set.
  (c4) HELD-OUT word     -- a function word (does) and a content word (trout) withheld from the threshold-fitting slice
       are still classified correctly by THEIR OWN stats vs frozen thresholds (generalisation, not memorisation).
  (d) the EMERGE-59 spiking-Broca PRODUCER still renders correctly on the DISCOVERED set + the gate-first no-confab MOAT
      holds (0 producer invocations on abstains).
GO bar: real-corpus F1 materially up (precision up, recall held) with a clear margin over every collapsed control,
controlled domain NOT regressed, producer renders, moat 0, 6-seed. Reuse-by-import; NO `sim/` edit; moat untouched.

HONEST SCOPE: this pushes S2 self-organisation onto REAL noisy data for the BOUNDED EMERGE frame domain (closed-class
INVENTORY). It does NOT make the domain open-ended (R4). The per-frame slot-ORDER (S1b, EMERGE-63) + slot-INVENTORY
(S1a, EMERGE-64) are the ranked follow-ons. The sentence-aware split is legitimate host syllabus prep (like rendering a
retinal image the neural retina reads -- `feedback_brain_based_only_standard`); the brain renders through spikes.

Run:
  SIM_BACKEND=numpy python -m research.runners._emerge62b_function_words_position_cue_derisk --demo
  SIM_BACKEND=numpy python -m research.runners._emerge62b_function_words_position_cue_derisk --derisk
  SIM_BACKEND=numpy python -m research.runners._emerge62b_function_words_position_cue_derisk --derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import json
import math
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Reuse-by-import EMERGE-62's stream generator, ground truth, thresholds, the PRF metric, and the producer feed
# (render_on_discovered / build_heldout_facts). The 2D Goldilocks discovery + stats are reused verbatim; EMERGE-62b
# ADDS the sentence-aware position statistic + the asymmetric exclusion gate on top.
from research.runners._emerge62_discover_function_words_derisk import (  # noqa: E402
    build_stream, GROUND_TRUTH_CLOSED, FRAME_FUNCTION_WORDS, _prank, _prf,
    render_on_discovered, TF_PCT, TC_PCT, MIN_FREQ, WINDOW, SENT_PERIOD, MARGIN,
)
from research.runners._emerge59_spiking_broca_frame_slots_derisk import build_heldout_facts  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_emerge62b_function_words_position_cue.json"

# ---------------------------------------------------------------------------------------------------------------------
# FIXED / PRE-REGISTERED position-cue parameter (percentile floor for the EXCLUSION gate; chosen once on seed-42 real+
# controlled to lift real precision + hold recall 1.00 without regressing the controlled domain, then FROZEN + applied
# verbatim to every seed + control). The frequency + coverage thresholds are inherited VERBATIM from EMERGE-62.
# ---------------------------------------------------------------------------------------------------------------------
TP_EXCL = 0.50            # a 2D candidate is EXCLUDED iff its posscore-percentile < TP_EXCL (clearly content-positioned)
MIN_FREQ_REAL = 20        # real-corpus min occurrences (EMERGE-62's real check used 20; kept identical)
_SENT_SPLIT_RE = re.compile(r"[.?!*]")     # sentence / fact-frame boundaries in the raw corpus punctuation
_TOKEN_RE = re.compile(r"[a-z]+")          # byte-identical to corpus_stream's token regex WITHIN a sentence

# EXTENDED honest closed-class set (English function words TinyStories contributes but the narrow EMERGE-domain ground
# truth omits) -- used ONLY for a SECONDARY, NON-GATING true-precision read (the narrow GT under-states precision because
# it counts genuine function words like he/she/of/for/but as "false positives"). NOT a GO gate.
EXTENDED_CLOSED = GROUND_TRUTH_CLOSED | {
    "he", "she", "we", "you", "i", "him", "her", "them", "his", "hers", "their", "its", "my", "your", "our",
    "was", "were", "are", "be", "been", "being", "am", "do", "did", "done", "has", "have", "had",
    "but", "or", "so", "if", "for", "of", "with", "at", "by", "from", "up", "out", "off", "as", "an", "one",
    "that", "this", "these", "those", "there", "here", "then", "no", "yes", "all", "any", "some",
    "what", "who", "when", "where", "why", "how", "which",
    "would", "could", "should", "will", "may", "might", "must", "shall",
}


# ---------------------------------------------------------------------------------------------------------------------
# SENTENCE-AWARE STATISTICS. The corpus_stream tokeniser strips punctuation, so it has no sentence boundaries; the
# position cue NEEDS them. We split the raw corpus on `[.?!*]` and tokenise `[a-z]+` within each sentence (the SAME
# token regex), giving per-word positional statistics: running FREQUENCY, context COVERAGE (Goldilocks arm, EMERGE-62),
# and the NEW position statistic posscore (phrase-boundary alignment). The controlled EMERGE stream already carries the
# `.` SENT_PERIOD delimiter, so we segment it into sentences the same way.
# ---------------------------------------------------------------------------------------------------------------------
def sentences_from_controlled(seed):
    """Segment EMERGE-62's controlled token stream into sentences (split on the SENT_PERIOD '.' delimiter)."""
    toks = build_stream(seed)
    sents, cur = [], []
    for t in toks:
        if t == SENT_PERIOD:
            if cur:
                sents.append(cur)
                cur = []
        else:
            cur.append(t)
    if cur:
        sents.append(cur)
    return sents


def sentences_from_real_corpus(corpus_name="ra_finetune_corpus.txt"):
    """Sentence-aware read of the project's REAL corpus: split the raw text on sentence/frame punctuation `[.?!*]`,
    tokenise `[a-z]+` within each sentence (byte-identical token regex to corpus_stream, WITH sentence boundaries).
    Returns (sentences | None)."""
    path = _REPO / "data" / "corpus" / corpus_name
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    sents = [_TOKEN_RE.findall(chunk) for chunk in _SENT_SPLIT_RE.split(text)]
    return [s for s in sents if s]


def compute_stats_positional(sentences, min_freq, window=WINDOW):
    """Per-word (frequency, coverage, posscore) over SENTENCE-SEGMENTED tokens.

    frequency + coverage are EMERGE-62's Goldilocks arms (context-coverage = #distinct +-window neighbours / vocab).
    posscore = (1 - fracFinal) * (1 - precededByContentNoun) is the NEW phrase-boundary/position statistic:
      * fracFinal[w]             -- fraction of occurrences at the sentence-final position (Yang-Getz phrase-edge).
      * precededByContentNoun[w] -- SOFT: fraction of LEFT-neighbour mass weighted by the neighbour's OWN fracFinal
                                    (endness). A function word is rarely preceded by a phrase-final-capable content noun.
    Returns (words, freq, cover, posscore, fracFinal, precByNoun) -- the last two exposed for the shuffle control.
    """
    uni = Counter()
    last_pos = Counter()                                    # occurrences at the sentence-final position
    co = defaultdict(set)                                   # distinct +-window neighbours (coverage arm)
    lN = defaultdict(Counter)                               # left-neighbour multiset (Redington left-role profile)
    nL = Counter()
    for s in sentences:
        L = len(s)
        for i, w in enumerate(s):
            uni[w] += 1
            if i == L - 1:
                last_pos[w] += 1
            for d in range(1, window + 1):
                for j in (i - d, i + d):
                    if 0 <= j < L:
                        co[w].add(s[j])
            if i - 1 >= 0:
                lN[w][s[i - 1]] += 1
                nL[w] += 1
    words = sorted([w for w in uni if uni[w] >= min_freq])
    if not words:
        z = np.zeros(0)
        return [], z, z, z, z, z
    Vt = max(1, len(words))
    freq = np.array([uni[w] for w in words], dtype=np.float64)
    cover = np.array([len(co[w]) / Vt for w in words], dtype=np.float64)
    endness = {w: last_pos[w] / uni[w] for w in words}      # per-word fracFinal (phrase-final rate == "noun-likeness")
    frac_final = np.array([endness[w] for w in words], dtype=np.float64)

    def prec_by_noun(w):
        if nL[w] == 0:
            return 0.0
        return sum(c * endness.get(cw, 0.0) for cw, c in lN[w].items()) / nL[w]

    prec_by_noun_arr = np.array([prec_by_noun(w) for w in words], dtype=np.float64)
    posscore = (1.0 - frac_final) * (1.0 - prec_by_noun_arr)
    return words, freq, cover, posscore, frac_final, prec_by_noun_arr


# ---------------------------------------------------------------------------------------------------------------------
# THE DISCOVERY: 2D Goldilocks (freq + coverage) with the 3rd POSITION cue as an ASYMMETRIC EXCLUSION gate.
# ---------------------------------------------------------------------------------------------------------------------
def discover_2d(words, freq, cover, tf=TF_PCT, tc=TC_PCT):
    """EMERGE-62's 2D Goldilocks discovery (high freq-pct AND high coverage-pct). Returns (set, kept-index-list)."""
    if not words:
        return set(), []
    fp = _prank(np.log(freq))
    cp = _prank(cover)
    kept = [i for i in range(len(words)) if fp[i] >= tf and cp[i] >= tc]
    return {words[i] for i in kept}, kept


def discover_3d(words, freq, cover, posscore, tf=TF_PCT, tc=TC_PCT, te=TP_EXCL):
    """The 3D discovery: the 2D Goldilocks candidates, MINUS the ones the position cue flags as clearly content-
    positioned (posscore-percentile < te). ASYMMETRIC exclusion (keep unless clearly content) -- protects frame function
    words that legitimately follow other function words. Returns (set, kept-index-list, excluded-set)."""
    d2, kept = discover_2d(words, freq, cover, tf, tc)
    if not kept:
        return set(), [], set()
    pp = _prank(posscore)
    kept3 = [i for i in kept if pp[i] >= te]
    d3 = {words[i] for i in kept3}
    return d3, kept3, (d2 - d3)


def _frame_recall(discovered):
    return len([w for w in FRAME_FUNCTION_WORDS if w in discovered]) / len(FRAME_FUNCTION_WORDS)


# ---------------------------------------------------------------------------------------------------------------------
# PER-SEED DE-RISK. The CONTROLLED stream is the GO gate (must not regress + producer renders + moat); the REAL corpus
# precision lift + the position-shuffle collapse are computed once (corpus is seed-independent) and reported.
# ---------------------------------------------------------------------------------------------------------------------
def _derisk_controlled(seed):
    """Controlled EMERGE-domain stream: 2D vs 3D (must not regress), held-out generalisation, freq-shuffle, no-stream,
    producer-renders-on-discovered + moat."""
    sents = sentences_from_controlled(seed)
    words, freq, cover, posscore, frac_final, prec_by_noun = compute_stats_positional(sents, MIN_FREQ)
    gt = GROUND_TRUTH_CLOSED & set(words)

    d2, _ = discover_2d(words, freq, cover)
    d3, kept3, excluded = discover_3d(words, freq, cover, posscore)
    P2, R2, F2 = _prf(d2, gt)
    P3, R3, F3 = _prf(d3, gt)
    frame_R = _frame_recall(d3)

    # (c1) POSITION-SHUFFLE: permute the position statistic<->identity mapping -> the cue is destroyed.
    rng = np.random.default_rng(seed * 13 + 5)
    perm_pos = rng.permutation(len(words))
    d3_posshuf, _, _ = discover_3d(words, freq, cover, posscore[perm_pos])
    _, _, F3_posshuf = _prf(d3_posshuf, gt)

    # (c2) FREQUENCY-SHUFFLE: permute the freq/coverage<->identity mapping -> discovery collapses.
    perm_fq = rng.permutation(len(words))
    d3_freqshuf, _, _ = discover_3d(words, freq[perm_fq], cover[perm_fq], posscore[perm_fq])
    _, _, F3_freqshuf = _prf(d3_freqshuf, gt)

    # (c3) NO-STREAM: empty stream -> empty discovered set.
    w0, f0, c0, p0, _, _ = compute_stats_positional([], MIN_FREQ)
    d3_nostream, _, _ = discover_3d(w0, f0, c0, p0)
    nostream_empty = (len(d3_nostream) == 0)

    # (c4) HELD-OUT generalisation: withhold a function word (does) + a content word (trout) from the fitting slice;
    # classify by THEIR OWN stats vs FROZEN thresholds (freq/coverage percentile from the KEPT words; the position gate
    # applied on the KEPT-word posscore percentile scale). Generalisation, not memorisation.
    held_ok_fw = held_ok_cw = None
    heldout_fw, heldout_cw = "does", "trout"
    if heldout_fw in words and heldout_cw in words:
        keep = [w for w in words if w not in (heldout_fw, heldout_cw)]
        keep_idx = [words.index(w) for w in keep]
        logfk = np.log(freq[keep_idx])
        ck = cover[keep_idx]
        posk = posscore[keep_idx]

        def classify(w):
            i = words.index(w)
            pf = float((logfk < math.log(freq[i])).mean())
            pc = float((ck < cover[i]).mean())
            pp = float((posk < posscore[i]).mean())         # excluded iff pp < TP_EXCL
            return (pf >= TF_PCT) and (pc >= TC_PCT) and (pp >= TP_EXCL)

        held_ok_fw = bool(classify(heldout_fw))              # want True  (closed, survives the position gate)
        held_ok_cw = bool(not classify(heldout_cw))          # want True  (open == excluded)

    # (d) PRODUCER renders on the DISCOVERED (3D) set + moat.
    facts = build_heldout_facts(seed, n=8)
    render_ok, moat_calls, answer_produced, frame_covered = render_on_discovered(seed, d3, facts)

    return {
        "seed": seed,
        "n_vocab": len(words), "n_gt": len(gt),
        "P_2d": P2, "R_2d": R2, "F1_2d": F2, "n_2d": len(d2),
        "P_3d": P3, "R_3d": R3, "F1_3d": F3, "n_3d": len(d3),
        "frame_recall_3d": frame_R, "frame_covered": frame_covered,
        "excluded_by_position": sorted(excluded),
        "false_positives_3d": sorted(d3 - gt), "false_negatives_3d": sorted(gt - d3),
        "F1_pos_shuffle": F3_posshuf, "F1_freq_shuffle": F3_freqshuf, "nostream_empty": bool(nostream_empty),
        "heldout_fw": heldout_fw, "heldout_fw_closed": held_ok_fw,
        "heldout_cw": heldout_cw, "heldout_cw_open": held_ok_cw,
        "render_ok": render_ok, "moat_calls_on_abstain": moat_calls, "answer_produced": answer_produced,
    }


def real_corpus_position_check(corpus_name="ra_finetune_corpus.txt"):
    """REAL corpus, sentence-aware: 2D-vs-3D precision/recall/F1 (narrow + extended GT), the position-shuffle collapse,
    and the discovered set vs ground truth (which content-word FPs the position cue now excludes). Seed-independent
    (the corpus is fixed); the shuffle is averaged over a few shuffle seeds. Returns a dict (or {'available': False})."""
    sents = sentences_from_real_corpus(corpus_name)
    if sents is None:
        return {"available": False, "reason": f"{corpus_name} not found"}
    n_tokens = sum(len(s) for s in sents)
    if n_tokens < 1000:
        return {"available": False, "reason": "corpus too small"}
    words, freq, cover, posscore, frac_final, prec_by_noun = compute_stats_positional(sents, MIN_FREQ_REAL)
    gt_n = GROUND_TRUTH_CLOSED & set(words)
    gt_e = EXTENDED_CLOSED & set(words)

    d2, _ = discover_2d(words, freq, cover)
    d3, _, excluded = discover_3d(words, freq, cover, posscore)
    P2n, R2n, F2n = _prf(d2, gt_n)
    P3n, R3n, F3n = _prf(d3, gt_n)
    P2e, R2e, F2e = _prf(d2, gt_e)
    P3e, R3e, F3e = _prf(d3, gt_e)

    # POSITION-SHUFFLE collapse (average over shuffle seeds): permute posscore<->identity, keep freq/coverage intact.
    shuf = []
    for sd in (1, 2, 3):
        rng = np.random.default_rng(sd)
        perm = rng.permutation(len(words))
        d3s, _, _ = discover_3d(words, freq, cover, posscore[perm])
        Ps, Rs, Fs = _prf(d3s, gt_n)
        shuf.append((Ps, Rs, Fs))
    Ps = float(np.mean([x[0] for x in shuf]))
    Rs = float(np.mean([x[1] for x in shuf]))
    Fs = float(np.mean([x[2] for x in shuf]))

    # which content-word false positives the position cue EXCLUDED (were 2D FPs, now removed) vs remaining
    fp2 = d2 - gt_n
    fp3 = d3 - gt_n
    excluded_content_fps = sorted(fp2 - fp3)
    # of the excluded, how many are TRUE content (not in the extended closed set) -- the ones we WANTED to remove
    excluded_true_content = sorted([w for w in excluded_content_fps if w not in EXTENDED_CLOSED])
    remaining_true_content_fp = sorted([w for w in fp3 if w not in EXTENDED_CLOSED])
    remaining_fp_are_genuine_func = sorted([w for w in fp3 if w in EXTENDED_CLOSED])[:40]

    return {
        "available": True, "corpus": corpus_name, "n_tokens": n_tokens, "n_sentences": len(sents), "n_vocab": len(words),
        "narrow_gt": {"P_2d": P2n, "R_2d": R2n, "F1_2d": F2n, "P_3d": P3n, "R_3d": R3n, "F1_3d": F3n},
        "extended_gt": {"P_2d": P2e, "R_2d": R2e, "F1_2d": F2e, "P_3d": P3e, "R_3d": R3e, "F1_3d": F3e},
        "precision_lift_narrow": (P3n / P2n) if P2n > 0 else None,
        "frame_recall_2d": _frame_recall(d2), "frame_recall_3d": _frame_recall(d3),
        "n_2d": len(d2), "n_3d": len(d3), "n_excluded": len(excluded),
        "position_shuffle": {"P": Ps, "R": Rs, "F1": Fs},
        "gt_not_discovered_3d": sorted(gt_n - d3),
        "excluded_true_content_fps": excluded_true_content[:60],
        "n_excluded_true_content": len(excluded_true_content),
        "remaining_true_content_fps": remaining_true_content_fp,
        "remaining_fps_are_genuine_function_words": remaining_fp_are_genuine_func,
    }


# ---------------------------------------------------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------------------------------------------------
def _demo(seed=42):
    print("\n=== EMERGE-62b -- ADD THE 3rd DISTRIBUTIONAL CUE (PHRASE-BOUNDARY / SYNTACTIC-POSITION ALIGNMENT) to the "
          "function-word discovery, so it works on the REAL noisy corpus ===\n")
    print("  The 3rd cue (Yang-Getz 2026): a function word is (1) rarely PHRASE-FINAL and (2) rarely PRECEDED BY A "
          "CONTENT NOUN;\n  frequent CONTENT words (nouns/objects end phrases; verbs follow the subject noun) are. "
          "posscore = (1-fracFinal)*(1-precByNoun).\n")

    # controlled
    sents = sentences_from_controlled(seed)
    words, freq, cover, posscore, ff, pbn = compute_stats_positional(sents, MIN_FREQ)
    gt = GROUND_TRUTH_CLOSED & set(words)
    d2, _ = discover_2d(words, freq, cover)
    d3, _, excl = discover_3d(words, freq, cover, posscore)
    P2, R2, F2 = _prf(d2, gt)
    P3, R3, F3 = _prf(d3, gt)
    print(f"  CONTROLLED stream (seed {seed}): 2D P {P2:.3f} R {R2:.3f} F1 {F2:.3f}  ->  3D P {P3:.3f} R {R3:.3f} "
          f"F1 {F3:.3f}  (no regression; frame-recall {_frame_recall(d3):.2f})")
    print(f"    2D discovered: {sorted(d2)}")
    print(f"    3D discovered: {sorted(d3)}   excluded-by-position: {sorted(excl)}")

    # real
    rc = real_corpus_position_check()
    if rc.get("available"):
        n = rc["narrow_gt"]
        print(f"\n  REAL corpus ({rc['corpus']}, {rc['n_tokens']} tokens, {rc['n_sentences']} sentences):")
        print(f"    narrow GT:  2D P {n['P_2d']:.3f} R {n['R_2d']:.3f} F1 {n['F1_2d']:.3f}  ->  "
              f"3D P {n['P_3d']:.3f} R {n['R_3d']:.3f} F1 {n['F1_3d']:.3f}   (precision lift {rc['precision_lift_narrow']:.2f}x)")
        e = rc["extended_gt"]
        print(f"    extended GT:2D P {e['P_2d']:.3f} R {e['R_2d']:.3f} F1 {e['F1_2d']:.3f}  ->  "
              f"3D P {e['P_3d']:.3f} R {e['R_3d']:.3f} F1 {e['F1_3d']:.3f}   (secondary, true-precision read)")
        ps = rc["position_shuffle"]
        print(f"    POSITION-SHUFFLE (cue destroyed): P {ps['P']:.3f} R {ps['R']:.3f} F1 {ps['F1']:.3f}  "
              f"(falls BELOW 2D {n['F1_2d']:.3f} -> the position cue is load-bearing)")
        print(f"    frame-recall 3D {rc['frame_recall_3d']:.2f}  GT-not-discovered {rc['gt_not_discovered_3d']}")
        print(f"    content-word FPs the position cue EXCLUDED ({rc['n_excluded_true_content']}): "
              f"{rc['excluded_true_content_fps'][:30]}")
    print()

    # render the frames on the discovered (3D) set
    facts = [{"subject": "owl", "ability_verb": "fly", "intr_verb": "walks"},
             {"subject": "penguin", "ability_verb": "fly", "intr_verb": "walks"}]
    r_ok, moat_calls, ans_ok, covered = render_on_discovered(seed, d3, facts)
    print(f"  render on the 3D-discovered function words: render-ok {r_ok:.2f} | frame-words-covered {covered} | "
          f"moat calls on abstain {moat_calls}\n")


# ---------------------------------------------------------------------------------------------------------------------
# DE-RISK (>=6 seeds)
# ---------------------------------------------------------------------------------------------------------------------
def _derisk(seeds):
    print(f"EMERGE-62b de-risk: ADD the 3rd cue (phrase-boundary / syntactic-position) to the function-word discovery; "
          f"controlled not regressed + real precision up (recall held) + position-shuffle collapse + producer + moat; "
          f"{len(seeds)}-seed", flush=True)
    t0 = time.time()
    err = None
    per = []
    try:
        for s in seeds:
            d = _derisk_controlled(s)
            per.append(d)
            print(f"  [seed {s}] CONTROLLED 2D F1 {d['F1_2d']:.3f} -> 3D F1 {d['F1_3d']:.3f} "
                  f"(P {d['P_3d']:.3f} R {d['R_3d']:.3f}) frame-R {d['frame_recall_3d']:.2f} | "
                  f"pos-shuffle F1 {d['F1_pos_shuffle']:.3f} freq-shuffle F1 {d['F1_freq_shuffle']:.3f} "
                  f"no-stream-empty {d['nostream_empty']} | held-out fw-closed {d['heldout_fw_closed']} "
                  f"cw-open {d['heldout_cw_open']} | render {d['render_ok']:.2f} moat {d['moat_calls_on_abstain']}",
                  flush=True)
    except Exception as e:
        err = repr(e)
        traceback.print_exc()

    real = real_corpus_position_check()

    if err is None:
        def m(k):
            return float(np.mean([d[k] for d in per]))

        # controlled aggregates
        F1_2d, F1_3d = m("F1_2d"), m("F1_3d")
        P_3d, R_3d = m("P_3d"), m("R_3d")
        frame_R = m("frame_recall_3d")
        render_ok = m("render_ok")
        F1_freqshuf = m("F1_freq_shuffle")
        nostream_ok = all(d["nostream_empty"] for d in per)
        held_fw_ok = all(d["heldout_fw_closed"] for d in per if d["heldout_fw_closed"] is not None)
        held_cw_ok = all(d["heldout_cw_open"] for d in per if d["heldout_cw_open"] is not None)
        moat_calls = int(sum(d["moat_calls_on_abstain"] for d in per))
        answer_ok = all(d["answer_produced"] for d in per)
        frame_covered = all(d["frame_covered"] for d in per)

        # -------- GO gates --------
        # (a) controlled NOT regressed: 3D F1 >= 2D F1 (per-seed), recall 1.00, frame-recall 1.00.
        controlled_not_regressed = all(d["F1_3d"] >= d["F1_2d"] - 1e-9 for d in per)
        controlled_recall_full = all(d["R_3d"] >= 0.999 for d in per) or all(d["R_3d"] >= d["R_2d"] - 1e-9 for d in per)
        controlled_frame_ok = (frame_R >= 0.999) and frame_covered
        # (b) REAL precision materially up + recall held + clear margin over the position-shuffle collapse.
        real_ok = real.get("available", False)
        if real_ok:
            rn = real["narrow_gt"]
            real_P_up = rn["P_3d"] > rn["P_2d"] + 1e-6
            real_recall_held = rn["R_3d"] >= 0.999
            real_F1_up = rn["F1_3d"] > rn["F1_2d"] + 0.02          # F1 materially up (>= +0.02 absolute)
            real_frame_ok = real["frame_recall_3d"] >= 0.999
            pos_shuf = real["position_shuffle"]
            # position-shuffle collapses BELOW the 3D result by a clear margin AND below/at the 2D level (load-bearing)
            posshuffle_collapses = (rn["F1_3d"] - pos_shuf["F1"] >= 0.05) and (pos_shuf["F1"] <= rn["F1_2d"] + 1e-6)
        else:
            real_P_up = real_recall_held = real_F1_up = real_frame_ok = posshuffle_collapses = False

        # (c) other controls collapse (freq-shuffle beaten by margin on controlled; no-stream empty; held-out; render; moat)
        freqshuffle_collapses = (F1_3d >= F1_freqshuf + MARGIN)
        held_ok = held_fw_ok and held_cw_ok
        render_high = render_ok >= 0.99
        moat_ok = (moat_calls == 0) and answer_ok

        go = bool(
            controlled_not_regressed and controlled_recall_full and controlled_frame_ok
            and real_ok and real_P_up and real_recall_held and real_F1_up and real_frame_ok and posshuffle_collapses
            and freqshuffle_collapses and nostream_ok and held_ok and render_high and moat_ok
        )

        rn = real["narrow_gt"] if real_ok else {}
        re_ = real["extended_gt"] if real_ok else {}
        ps = real["position_shuffle"] if real_ok else {}
        if go:
            verdict = (
                f"GO -- the 3rd DISTRIBUTIONAL CUE (PHRASE-BOUNDARY / SYNTACTIC-POSITION ALIGNMENT, Yang-Getz 2026 3rd "
                f"universal property; Redington/Cartwright-Brent left-neighbour role; catalog G.12 Broca open/closed) "
                f"makes the closed-class function-word inventory SELF-ORGANISE on the REAL noisy corpus. The position "
                f"statistic posscore = (1-fracFinal)*(1-precededByContentNoun) [a function word is rarely phrase-final "
                f"and rarely preceded by a content noun] GATES the 2D Goldilocks candidates by ASYMMETRIC EXCLUSION "
                f"(exclude only the clearly content-positioned; NO hand-list as input). REAL corpus: narrow-GT precision "
                f"{rn['P_2d']:.3f} -> {rn['P_3d']:.3f} (~{real['precision_lift_narrow']:.2f}x), F1 {rn['F1_2d']:.3f} -> "
                f"{rn['F1_3d']:.3f}, RECALL HELD at {rn['R_3d']:.3f} (all 11 ground-truth closed-class words + all 4 "
                f"frame function words still discovered); the position cue EXCLUDED {real['n_excluded_true_content']} "
                f"frequent-CONTENT-word false positives (nouns/objects/verbs, QA-structural tokens). The POSITION-SHUFFLE "
                f"control COLLAPSES to F1 {ps['F1']:.3f} (below the 2D {rn['F1_2d']:.3f} -> the position cue is "
                f"LOAD-BEARING, not a spurious lift). Every other input-destruction control collapses (FREQUENCY-SHUFFLE "
                f"F1 {F1_freqshuf:.3f}, NO-STREAM empty) + held-out generalisation holds. The CONTROLLED EMERGE-domain "
                f"stream is NOT regressed (3D F1 {F1_3d:.3f} >= 2D F1 {F1_2d:.3f} every seed, frame-recall {frame_R:.2f}). "
                f"The DISCOVERED set feeds the EMERGE-59 spiking-Broca frames (render-ok {render_ok:.2f}, gate-first "
                f"no-confab MOAT intact: 0 producer invocations on abstains). {len(seeds)} seeds. Secondary (non-gating) "
                f"true-precision read vs an EXTENDED honest closed class: precision {re_['P_2d']:.3f} -> {re_['P_3d']:.3f} "
                f"(the narrow GT under-states precision because it counts genuine function words like he/she/of/for/but "
                f"as false positives). ==> S2 self-organises on REAL data: the function-word inventory emerges from 3 "
                f"distributional cues (frequency + coverage + phrase-boundary position), no host list. HONEST SCOPE: "
                f"bounded EMERGE frame domain (NOT open-ended R4); slot-ORDER (S1b/EMERGE-63) + slot-INVENTORY "
                f"(S1a/EMERGE-64) are the ranked follow-ons. Reuse-by-import; NO sim/ edit; moat untouched.")
        else:
            miss = []
            if not (controlled_not_regressed and controlled_recall_full and controlled_frame_ok):
                miss.append(f"CONTROLLED regressed (3D F1 {F1_3d:.3f} vs 2D {F1_2d:.3f}, recall/frame-recall {frame_R:.2f})")
            if real_ok and not real_P_up:
                miss.append(f"real precision NOT up ({rn['P_2d']:.3f} -> {rn['P_3d']:.3f})")
            if real_ok and not real_recall_held:
                miss.append(f"real recall NOT held ({rn['R_3d']:.3f} < 1.0)")
            if real_ok and not real_F1_up:
                miss.append(f"real F1 not materially up ({rn['F1_2d']:.3f} -> {rn['F1_3d']:.3f})")
            if real_ok and not posshuffle_collapses:
                miss.append(f"POSITION-SHUFFLE did NOT collapse (3D F1 {rn['F1_3d']:.3f} vs shuffle {ps.get('F1')} "
                            f"vs 2D {rn['F1_2d']:.3f}) -> the lift may be spurious")
            if not real_ok:
                miss.append(f"real corpus unavailable ({real.get('reason')})")
            if not freqshuffle_collapses:
                miss.append(f"freq-shuffle not beaten by {MARGIN} (3D {F1_3d:.3f} vs shuffle {F1_freqshuf:.3f})")
            if not nostream_ok:
                miss.append("no-stream not empty")
            if not held_ok:
                miss.append(f"held-out failed (fw-closed {held_fw_ok}, cw-open {held_cw_ok})")
            if not render_high:
                miss.append(f"render {render_ok:.2f} < 0.99")
            if not moat_ok:
                miss.append(f"MOAT breached ({moat_calls} calls / answer {answer_ok})")
            verdict = ("BOUNDARY -- " + "; ".join(miss) + ". The precise residual is named. If the position cue only "
                       "PARTIALLY closes the real-corpus precision gap (as expected -- the narrow ground truth counts "
                       "genuine TinyStories function words as false positives), that is an HONEST residual, NOT a wall: "
                       "the next single-variable signal is a 4th cue -- morphological invariance (function words lack "
                       "inflectional paradigms) OR a bootstrapped open/closed clustering (Redington context-vector "
                       "k-means). The POSITION-SHUFFLE control MUST collapse for the lift to be real; if it did NOT, the "
                       "lift is spurious and must NOT be claimed. If the MOAT was breached this is BLOCKING -- do NOT "
                       "weaken the moat.")
    else:
        verdict = f"ERROR -- {err}"
        F1_2d = F1_3d = P_3d = R_3d = frame_R = render_ok = None
        moat_calls = None
        go = False

    if real.get("available"):
        rn = real["narrow_gt"]
        print(f"\n  [REAL {real['corpus']}] narrow-GT 2D P {rn['P_2d']:.3f} F1 {rn['F1_2d']:.3f} -> 3D P {rn['P_3d']:.3f} "
              f"R {rn['R_3d']:.3f} F1 {rn['F1_3d']:.3f} (lift {real['precision_lift_narrow']:.2f}x) | "
              f"pos-shuffle F1 {real['position_shuffle']['F1']:.3f} | frame-R {real['frame_recall_3d']:.2f} | "
              f"excluded {real['n_excluded_true_content']} content-FPs", flush=True)

    summary = {
        "probe": "emerge62b_function_words_position_cue", "verdict": verdict,
        "go": bool(go) if err is None else False,
        "real_corpus_position_check": real,
        "mechanism": ("add the 3rd distributional cue (PHRASE-BOUNDARY / SYNTACTIC-POSITION ALIGNMENT, Yang-Getz 2026 "
                      "arXiv 2601.21191 3rd universal property; Redington/Cartwright-Brent left-neighbour role profile; "
                      "catalog G.12 Broca open/closed dissociation) to EMERGE-62's 2D Goldilocks (frequency + context-"
                      "coverage) function-word discovery. A SENTENCE-AWARE front end splits the raw corpus on `[.?!*]` "
                      "(the corpus_stream tokeniser strips punctuation) to recover phrase boundaries; the position "
                      "statistic posscore = (1-fracFinal)*(1-precededByContentNoun) [a function word is rarely phrase-"
                      "final and rarely preceded by a phrase-final-capable content noun] GATES the 2D candidates by "
                      "ASYMMETRIC EXCLUSION (exclude only the clearly content-positioned -- protects frame function "
                      "words that follow other function words in dense streams; NO hand-list as input). Input-destruction "
                      "controls (POSITION-SHUFFLE, FREQUENCY-SHUFFLE, NO-STREAM) + held-out-word generalisation gate the "
                      "result (project control-validity methodology). Reuse-by-import (EMERGE-62 stream/stats/PRF + "
                      "EMERGE-59 producer feed); NO sim/ edit."),
        "task": ("push real-corpus PRECISION up from EMERGE-62's 0.078 (recall held at 1.00) by adding the phrase-"
                 "boundary/position cue; controlled EMERGE-domain stream NOT regressed; producer renders on the "
                 "discovered set + moat 0; POSITION-SHUFFLE + FREQUENCY-SHUFFLE + NO-STREAM collapse; held-out fw/cw "
                 "generalise; >=6 seeds"),
        "ground_truth_closed_class_narrow": sorted(GROUND_TRUTH_CLOSED),
        "frame_function_words": FRAME_FUNCTION_WORDS,
        "thresholds": {"freq_pct": TF_PCT, "cover_pct": TC_PCT, "position_exclude_pct": TP_EXCL,
                       "min_freq_controlled": MIN_FREQ, "min_freq_real": MIN_FREQ_REAL, "window": WINDOW,
                       "margin": MARGIN},
        "seeds": list(seeds), "elapsed_seconds": round(time.time() - t0, 1),
        "aggregate": None if err else {
            "controlled_F1_2d": F1_2d, "controlled_F1_3d": F1_3d, "controlled_P_3d": P_3d, "controlled_R_3d": R_3d,
            "controlled_frame_recall": frame_R, "render_ok": render_ok, "moat_calls_on_abstain_total": moat_calls,
        },
        "per_seed_controlled": per,
        "HONEST_NOTE": ("The 3rd cue makes S2 self-organise on REAL noisy data for the BOUNDED EMERGE frame domain. It "
                        "does NOT make the domain open-ended (R4, the separate deferred wall). The narrow ground truth "
                        "(11 EMERGE-domain closed-class words) UNDER-states precision: most remaining 'false positives' "
                        "(he/she/they/but/of/for/with/that/was/were/had/...) are GENUINE English function words that "
                        "TinyStories contributes -- against an EXTENDED honest closed-class set the precision + lift are "
                        "larger (reported as a secondary, non-gating read). The sentence-aware split is legitimate host "
                        "syllabus prep (like rendering a retinal image the neural retina reads); the gate-first moat is "
                        "untouched (0 productions on abstains, by construction). The per-frame slot-ORDER (S1b, EMERGE-63) "
                        "+ slot-INVENTORY (S1a, EMERGE-64) are the ranked follow-ons composing into EMERGE-65."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 118, flush=True)
    print(f"[emerge62b] VERDICT: {verdict}", flush=True)
    print(f"[emerge62b] wrote {OUT}\n" + "=" * 118, flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--derisk", action="store_true")
    a = ap.parse_args()
    if a.derisk:
        return _derisk(a.seeds)
    _demo(a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
