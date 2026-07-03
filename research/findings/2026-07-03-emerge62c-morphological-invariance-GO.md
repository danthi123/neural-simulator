# EMERGE-62c — the 4th (MORPHOLOGICAL-INVARIANCE) cue closes the inflected-content-verb false positives — GO (6-seed)

**Date:** 2026-07-03
**Runner:** `research/runners/_emerge62c_morphological_invariance_cue_derisk.py`
**Test:** `tests/test_emerge62c_morphological_invariance_cue.py` (8 tests, CPU/numpy, offline)
**Raw:** `research/findings/raw/_emerge62c_morphological_invariance_cue.json`
**Verdict:** **GO** — real-corpus narrow-GT precision **0.111 → 0.121** (1.09×), recall **held 1.000**, morphology-shuffle **collapses** (load-bearing), controlled domain not regressed, producer renders, moat 0, 6-seed.

## What this closes

The ONE named boundary of EMERGE-62b (`2026-07-03-emerge62b-position-cue-GO.md`): after the 3-cue discovery (2D Goldilocks FREQUENCY + context-COVERAGE from EMERGE-62; + the 3rd PHRASE-BOUNDARY/POSITION cue from EMERGE-62b) lifted real-corpus narrow-GT precision to 0.111 (recall 1.0), the **dominant remaining false-positive class was INFLECTED CONTENT VERBS** (`gives / hugs / makes / wants / likes / sees / rides / holds / wanted`) — content words that are frequent, broad-context, and not phrase-final, so the first 3 cues cannot separate them.

The EMERGE-62b findings named the next signal explicitly: **FUNCTION words are MORPHOLOGICALLY INVARIANT** — they lack the -s/-ed/-ing inflectional paradigm (`the / a / to / on / in / of / and / is / it / he / she / ...` appear in ONE surface form), whereas an inflected content verb (`gives`, `hugs`, `makes`) is the -s/-es/-ies form of a bare stem (`give`, `hug`, `make`) that ALSO occurs in the corpus.

## Mechanism (the 4th cue)

Per-word, **label-free from the corpus vocab only** (Kelly 1992 "Using sound to solve syntactic problems"; Monaghan-Christiansen-Chater phonological/morphological POS bootstrapping; Yang-Getz 2026 arXiv 2601.21191; catalog **G.12** Broca open/closed dissociation, Kandel 6e Ch 55):

```
morph_variant[w] = 1  iff  w is a valid inflected surface (-s/-es/-ies/-ed/-ing)
                           whose base STEM occurs in the corpus vocab
                           AND that base stem is NOT itself function-like (same 2D Goldilocks freq+coverage test)
```

Two guards make it asymmetric-safe:

1. **base-stem-present** — a genuine paradigm relative must occur (guards false stemming `is→i`, `was→wa`, `has→ha`, `this→thi`; those bases don't occur, so `is/was/has/this` are never flagged variant). The `-s` branch also skips double-s / -us / -is endings.
2. **base-is-NOT-function-like** — reuses the SAME 2D Goldilocks test (no hand list): a function-word inflection like **`does`→`do`** is PROTECTED because `do` is itself high-frequency + high-coverage (a bare auxiliary, "do not know"), so `does` is NOT flagged variant and **stays discovered** (recall must hold 1.0 — `does` is a FRAME function word). A content verb's base (`give/hug/make/hold/ride/see/want`) is NOT function-like, so its -s form IS flagged.

The flag GATES the 3-cue candidates by **ASYMMETRIC EXCLUSION** (keep unless clearly an inflected content surface; NO hand-list as input) — exactly as EMERGE-62b's position cue does. The 4-cue set is a strict SUBSET of the 3-cue set (exclusion-only).

## Results (6 seeds 42/43/44/100/101/102, CPU/numpy)

**REAL corpus** (`data/corpus/ra_finetune_corpus.txt`, 647,434 tokens, 96,832 sentences, seed-independent):

| | precision | recall | F1 |
|---|---|---|---|
| 3-cue (EMERGE-62b) | 0.111 | 1.000 | 0.200 |
| **4-cue (EMERGE-62c)** | **0.121** | **1.000** | **0.216** |
| MORPHOLOGY-SHUFFLE | 0.111 | **0.879** | 0.198 |

- **Precision 0.111 → 0.121 (1.09×), recall HELD at 1.000**, frame-recall 1.00 (all 11 ground-truth closed-class words + all 4 frame function words still discovered; `does` protected).
- The morphological cue EXCLUDED exactly the **8 inflected content verbs** it was designed to remove: `gives, holds, hugs, makes, rides, sees, wanted, wants`.
- Secondary EXTENDED-GT read (non-gating, true precision): **0.354 → 0.385**.

**MORPHOLOGY-SHUFFLE collapse (load-bearing).** The real cue lifts precision WHILE holding recall at 1.00 (it excludes only inflected content surfaces). A random morph flag CANNOT reproduce that signature: it deletes words uniformly at random, so it (i) damages **RECALL** (randomly deletes true closed-class words → R falls 1.000 → **0.879**) and (ii) its F1 falls to **0.198 ≤ the 3-cue baseline 0.200** (no purifying precision lift). Both are genuine input-destruction signatures. (The narrow-GT F1 denominator is only 11 words, so the absolute F1 lift from removing the true content verbs is inherently ~0.016; the collapse is gated on the shuffle FAILING TO HOLD RECALL + no-purification — a robust, deterministic input-destruction signature — not on a coarse absolute-F1 margin the small precision regime cannot produce.)

**CONTROLLED EMERGE-domain stream — NOT regressed** (per seed): 4-cue F1 == 3-cue F1 every seed (0.870 / 0.833 / 0.800×4), recall not regressed (narrow-GT R 0.909 inherits EMERGE-62/62b's `it` low-coverage miss, not a 62c regression), frame-recall 1.00. The controlled stream's inflected 3sg verbs (`verb+'s'`) are already correctly OPEN and their bare stems are content, so the morphological cue removes no function word there (0 morph-exclusions on the controlled stream).

**Other controls collapse:** FREQUENCY-SHUFFLE F1 ~0.081 (≥ MARGIN 0.30 below main), NO-STREAM → empty set, HELD-OUT generalisation holds (withheld `does` still CLOSED — protected by the base-is-function guard; withheld `trout` still OPEN — by its own stats vs frozen thresholds).

**Producer + moat:** the DISCOVERED (4-cue) set feeds the EMERGE-59 spiking-Broca frames — held-out facts render correctly (render-ok 1.00), gate-first no-confab MOAT intact (**0 producer invocations on abstains**).

## Honest scope / residual

- This removes the **inflected-content-verb** FP class (the named boundary). The **determiner-preceded BARE-noun** FP class (`bird/dog/fox/owl/pig` — singulars whose plural rarely appears in TinyStories) is a HARDER residual the morphological cue does NOT fully close: the plural-variant direction (does the word's stem have a plural?) is **unreliable** — false stemming `the→thing`, `a→as`, `on→ones`, `it→its`, `bee→being` would wrongly kill function words, so it is **NOT used** (verified). The inflected-form direction (word IS an inflection of a present stem) is clean and safe; it is the one used.
- The narrow 11-word EMERGE-domain ground truth **UNDER-states** precision: most remaining "false positives" (`he/she/of/for/but/that/was/were/had/...`) are GENUINE English function words TinyStories contributes; against the extended honest closed class the precision + lift are larger (secondary read: 0.354 → 0.385).
- This pushes **S2 self-organisation** further on REAL noisy data for the BOUNDED EMERGE frame domain (closed-class INVENTORY). It does NOT make the domain open-ended (**R4**, the separate deferred wall).
- The sentence-aware split + morphological stemming are legitimate **host syllabus prep** (like rendering a retinal image the neural retina reads — `feedback_brain_based_only_standard`); the brain renders through spikes; the gate-first moat is untouched (0 productions on abstains, by construction).

## Provenance

Reuse-by-import (EMERGE-62 stream/stats/PRF + EMERGE-62b sentence-aware positional stats + 2D/3D discovery + EMERGE-59 producer feed); **NO `sim/` edit**. Cites EMERGE-62 / EMERGE-62b, Yang & Getz 2026 (arXiv 2601.21191), Kelly 1992, Monaghan-Christiansen-Chater, catalog G.12 (Broca open/closed dissociation).

**⇒ the function-word inventory now emerges from 4 distributional cues — frequency + coverage + phrase-boundary position + morphological invariance — no host list.** The per-frame slot-ORDER (S1b, EMERGE-63) + slot-INVENTORY (S1a, EMERGE-64) are the ranked follow-ons composing into the fully-self-organised producer.
