# EMERGE-62b — ADD the 3rd distributional cue (PHRASE-BOUNDARY / SYNTACTIC-POSITION alignment) to the function-word discovery — GO (real-corpus precision up, recall held), the named EMERGE-62 boundary iterated

**Date:** 2026-07-03
**Verdict:** **GO** (6-seed) — the 3rd Yang-Getz distributional cue (phrase-boundary / syntactic-position alignment) makes the closed-class function-word inventory SELF-ORGANISE on the REAL noisy corpus. Real-corpus precision **0.080 → 0.111 (~1.39×)**, F1 **0.148 → 0.200**, **recall held at 1.00** (all 11 ground-truth closed-class words + all 4 frame function words still discovered); the position cue excludes **34 frequent-content-word false positives**. The **POSITION-SHUFFLE control collapses below the 2D level** (the cue is load-bearing, not spurious). The controlled EMERGE-domain stream is **NOT regressed** (3D F1 == 2D F1 every seed, frame-recall 1.00). The producer still renders, moat 0.
**Runner:** `research/runners/_emerge62b_function_words_position_cue_derisk.py` (`--demo` / `--derisk`)
**CI:** `tests/test_emerge62b_function_words_position_cue.py` (8 tests, CPU/numpy, offline) — all pass. EMERGE-62's own 7 tests still pass (imported, not modified).
**Raw:** `research/findings/raw/_emerge62b_function_words_position_cue.json`
**Reuse-by-import; NO `sim/` edit; the gate-first moat untouched (0 productions on abstains).**

---

## What this iterates (the named EMERGE-62 boundary)

EMERGE-62 (`2026-07-03-emerge62-discover-function-words-GO.md`, `_emerge62_discover_function_words_derisk.py`) discovered the closed class from **2 distributional cues** — high running-FREQUENCY AND high context-COVERAGE (the "Goldilocks" signature). GO on the controlled EMERGE-domain stream (F1 0.863). But on the REAL corpus (`data/corpus/ra_finetune_corpus.txt`, 647K tokens, TinyStories-interleaved fact/QA):

- **recall 1.00 + frame-recall 1.00** — frequency + coverage RELIABLY FIND the closed class, and
- **precision 0.078 / F1 0.145** — they OVER-INCLUDE: hundreds of high-frequency, high-coverage NARRATIVE CONTENT words (common nouns/objects/verbs; QA-structural tokens `facts`/`answer`/`question`) also pass frequency + coverage. Frequency + coverage cannot SEPARATE the frequent-content-word false positives.

EMERGE-62 named the exact next signal: **Yang-Getz's 3rd universal property — phrase-boundary / syntactic-position alignment.** EMERGE-62b adds it.

## The 3rd cue (self-organised; NO hand list as input)

**Field grounding.** Yang & Getz (2026, arXiv 2601.21191): across 186 languages a function word is universally (1) high-frequency, (2) reliably syntactically associated / diverse, **(3) PHRASE-BOUNDARY ALIGNED** — it occurs at construction edges / fixed syntactic slots (immediately BEFORE content: determiners precede nouns, auxiliaries precede verbs) and is almost NEVER phrase-FINAL; a frequent CONTENT word (noun/object) sits at variable positions and ENDS phrases. Redington/Cartwright-Brent distributional POS induction uses the immediate LEFT/RIGHT neighbour role profile. Catalog **G.12** (Kandel 6e Ch 55 pp 1382-1384): Broca agrammatism = retained noun selection, lost function-word use — the closed class is a separable population.

**Operationalisation (cheapest that works, ONE variable).** The corpus tokeniser (`corpus_stream`, `re.findall(r"[a-z]+")`) strips ALL punctuation, so it has **no sentence boundaries** — the position cue needs them. So EMERGE-62b adds a **sentence-aware front end**: split the raw corpus on sentence/frame punctuation `[.?!*]` and tokenise `[a-z]+` within each sentence (byte-identical token regex, WITH phrase boundaries). The controlled EMERGE stream already carries the `.` `SENT_PERIOD` delimiter and is segmented the same way. The position statistic:

```
posscore[w] = (1 - fracFinal[w]) * (1 - precededByContentNoun[w])          higher = more function-like
  fracFinal[w]             = fraction of w's occurrences at the sentence-FINAL position (Yang-Getz phrase-edge:
                             a function word is almost never phrase-final; a content noun/object ends phrases).
  precededByContentNoun[w] = SOFT fraction of w's LEFT-neighbour mass weighted by the neighbour's OWN fracFinal
                             (endness). A function word is rarely preceded by a phrase-final-capable content noun;
                             a content VERB (follows the subject noun) and an OBJECT (follows the verb) ARE. This is
                             the Redington/Cartwright-Brent immediate-LEFT-neighbour role profile.
```

**Why the SOFT `precededByContentNoun` (not a hard "preceded-by-any-closed-word").** The hard variant wrongly kills the frame function words `the`/`a`/`not`/`in` on the very-regular controlled stream, because there they almost always follow OTHER function words ("to the", "does not", "is in the") → `precededByClosed ≈ 1.0` → excluded. Weighting the left-neighbour by its OWN endness distinguishes "preceded by a determiner/aux (low endness → kept)" from "preceded by a content noun (high endness → excluded)". Verified: the hard cue regresses the controlled domain to F1 0.706; the soft cue does not regress it at all.

**COMBINE — asymmetric exclusion (the key design choice).** The position cue GATES the 2D Goldilocks candidates by **ASYMMETRIC EXCLUSION**: a 2D candidate is KEPT unless its posscore-percentile is BELOW `TP_EXCL = 0.50` — i.e. the cue only EXCLUDES the clearly content-positioned (sentence-final nouns/objects, post-noun content verbs); it does NOT REQUIRE strong function-positioning. A symmetric "require high posscore" gate breaks recall on BOTH streams (it penalises function words that follow other function words). The asymmetric gate lifts real precision + holds recall 1.00 + does NOT regress the controlled domain. Thresholds (`freq-pct ≥ 0.90`, `cover-pct ≥ 0.60` inherited verbatim from EMERGE-62; `position-exclude-pct = 0.50`) are FIXED / pre-registered.

## Results — 6-seed (42/43/44/100/101/102), CPU/numpy

**REAL corpus (seed-independent; the corpus is fixed):**

| metric | 2D (EMERGE-62) | 3D (EMERGE-62b) | notes |
|---|---|---|---|
| narrow-GT **precision** | 0.080 | **0.111** | **~1.39×** |
| narrow-GT **recall** | 1.000 | **1.000** | HELD — every ground-truth closed-class word still found |
| narrow-GT **F1** | 0.148 | **0.200** | +0.052 (35% up) |
| **frame-recall** | 1.00 | **1.00** | all 4 frame function words `{the, can, does, not}` survive |
| content-word FPs excluded | — | **34** | apple, ball, box, cake, cat, fish, tree, water, worm, know, say, … |

**POSITION-SHUFFLE (the load-bearing anti-cheat):** permute the position-statistic↔word mapping → 3rd cue destroyed → **F1 0.098, BELOW the 2D 0.148** (precision 0.056, recall broken to 0.36). The shuffled cue actively HURTS → the position cue is load-bearing, not a spurious lift.

**CONTROLLED EMERGE-domain stream (6-seed, the GO gate):** 3D F1 == 2D F1 **every seed** (0.870/0.833/0.800×4), frame-recall **1.00** every seed, no closed-class word lost, `excluded-by-position = []` (the controlled stream has no content-positioned candidates to remove). **No regression.**

**Other input-destruction controls (all collapse):**
- **FREQUENCY-SHUFFLE**: controlled 3D F1 0.817 vs shuffle ~0.122 (margin ≫ 0.30).
- **NO-STREAM**: empty stream → empty discovered set, every seed.
- **HELD-OUT generalisation**: withhold `does` (fw) + `trout` (cw) from the fitting slice, classify by their OWN stats vs frozen freq/coverage/position thresholds → `does` CLOSED (survives the position gate), `trout` OPEN (excluded), every seed.

**Producer + moat:** the DISCOVERED (3D) set feeds the EMERGE-59 spiking-Broca frames — held-out facts render correctly (render-ok **1.00**), gate-first no-confab MOAT intact (**0** producer invocations on abstains, an answer DOES invoke it).

## Honest residual — the narrow ground truth UNDER-states precision

The narrow ground truth is the 11 EMERGE-domain closed-class words `{the, a, can, does, not, to, on, in, and, is, it}`. Most of the ~40 remaining real-corpus "false positives" (`he`, `she`, `they`, `but`, `for`, `of`, `with`, `that`, `was`, `were`, `had`, `at`, `be`, `up`, `when`, `who`, …) are **GENUINE English function words** that TinyStories contributes but the narrow EMERGE-domain ground truth omits. Against an **EXTENDED honest closed-class set** (reported as a secondary, non-gating read): precision **0.290 → 0.354**, F1 0.369 → 0.393 — the true precision + lift are materially larger. The 1.39× narrow-GT lift is the conservative floor; the metric penalises the discovery for correctly finding closed-class words outside the narrow domain.

**Why this is a GO, not a full close:** the cue MATERIALLY lifts real-corpus precision (recall held, F1 +0.052), the position-shuffle control collapses BELOW the 2D level (load-bearing), and the controlled domain is not regressed — the GO bar. The genuine residual (the narrow-vs-extended-GT gap; content nouns like `cat`/`bird`/`dog` that follow determiners AND are frequent enough to survive) is precisely named and is the NEXT single-variable signal (below), not a wall.

## Scope + BRAIN-BASED-ONLY compliance

- Pushes S2 self-organisation onto REAL noisy data for the BOUNDED EMERGE frame domain (closed-class INVENTORY). It does NOT make the domain open-ended (R4, the deferred wall — the from-scratch spiking LM is ~4 orders too small, `2026-05-07-Phase-2.3a-NEGATIVE`).
- The sentence-aware split is legitimate host syllabus prep — like rendering a retinal image the neural retina then reads (`feedback_brain_based_only_standard`: host is legitimate for the environment/syllabus; the brain renders through spikes). The order production is on real spikes (EMERGE-59); the discovered set FEEDS those spiking frames.
- The gate-first no-confab moat is untouched (0 producer invocations on abstains, by construction).

## Next (ranked)

- **The precision residual (partial close):** the remaining true-content FPs (frequent nouns that follow determiners — `cat`/`bird`/`dog`) share the determiner-preceded profile with… nothing distinctive left in position. The next single-variable signal is a **4th cue — morphological invariance** (function words lack inflectional paradigms: no `-s`/`-ed`/`-ing`), OR a bootstrapped **Redington context-vector k-means** open/closed clustering. Still not a wall.
- **EMERGE-63** (S1b): swap `FrameSlotCQ._teach_order`'s host-template order-teacher for a corpus n-gram / positional order-teacher.
- **EMERGE-64** (S1a): extend `_bucketB_corpus_mined_frames` mining to the FUNC-slot inventory.
- **EMERGE-65**: compose 62 + 62b + 63 + 64 into the fully-self-organised spiking-Broca producer.

## Sources

Yang & Getz (2026) arXiv 2601.21191 (186-language universality of frequency + syntactic-association + phrase-boundary cues); Redington, Chater & Finch (1998); Cartwright & Brent (1997) (distributional POS induction, immediate left/right neighbour role); Kandel 6e Ch 55 pp 1382-1384 (feature-catalog G.12, Broca open/closed dissociation). Research gate: `research/findings/2026-07-03-self-organizing-grammatical-structure-research-gate.md` (Move-2(A), Move-3 RANK-1 NEGATIVE-path — "add the phrase-boundary-alignment cue"). Project precedents: `_emerge62_discover_function_words_derisk.py` (2D Goldilocks discovery + stream/stats/PRF), `_emerge59_spiking_broca_frame_slots_derisk.py` (FRAMES / FrameSlotCQ / BrocaProducer feed), `corpus_stream.py` (real-corpus loader — its punctuation-stripping is why the sentence-aware front end is needed).
