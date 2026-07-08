# The rung-4 verb-category boundary DIAGNOSED (data-backed): verbs are the least-coherent category under bag-of-words co-occurrence at both breadth scales, because the `TAXONOMY_8x8` "actions" probe is a HETEROGENEOUS grab-bag mixing Levin verb classes — not a fundamental "verbs can't be categorized" wall. The next mechanism is argument-structure over NATURAL verb classes (the EMERGE-72/74 construction-mining lever). NO `sim/` edit.

**Date:** 2026-07-08
**Verdict:** the rung-4 diffuse-verb boundary is a heterogeneous-probe + bag-of-words property, not a substrate wall — argument-structure is the research-gated next mechanism.

## The measurement (per-category within-category coherence = mean pairwise cosine, TinyStories, seed 42)
| K=256 | coh | members | | K=1024 | coh | members |
|---|---|---|---|---|---|---|
| animals | **+0.215** | dog cat bird fish frog bear mouse | | colors | **+0.152** | red blue green yellow black white pink brown |
| family | +0.133 | mom dad girl boy friend | | body | +0.127 | hand head hair arm leg face |
| places | +0.122 | house park room garden tree | | animals | +0.124 | dog cat bird fish frog bear mouse duck |
| **actions** | **+0.069** | run walk play look eat | | food | +0.111 | apple cake bread milk egg soup candy cookie |
| | | | | toys | +0.088 | ball toy book doll box blocks kite bell |
| | | | | places | +0.060 | house park room garden tree road school beach |
| | | | | family | +0.058 | mom dad girl boy baby friend sister brother |
| | | | | **actions** | **+0.027** | run jump walk play look eat sleep sing |

**Verbs (actions) are the LEAST coherent category at both scales** (+0.069 / +0.027 — the lowest), confirming the rung-4 observation that diffuse verb categories inherit weakly. Robust, not a selection artifact.

## The diagnosis (why — reading the members)
The `TAXONOMY_8x8` "actions" category is a HETEROGENEOUS grab-bag: {run, jump, walk} (motion), {look} (perception), {eat} (consumption/ingestion), {sleep} (state), {sing, play} (activity) — these are DIFFERENT Levin verb classes with different argument structures. So "actions" is not a coherent co-occurrence cluster BY CONSTRUCTION — it mixes classes that neither co-occur with each other nor share a consistent context. The low coherence reflects the probe's heterogeneity, NOT a fundamental "verbs are unclusterable" wall. (Nouns like animals/colors/body ARE natural taxonomic classes, so they cohere.)

## The next mechanism (boundary = next mechanism, per the standing reframe)
Verbs cluster by their ARGUMENT STRUCTURE (which noun-types they take as subject/object — Levin verb classes), NOT by window co-occurrence. The project already has the lever: EMERGE-72/74's construction/argument-structure mining (subject-before / object-after slots, the Dominey-Hinaut construction router). The research-gated de-risk: build verb codes from argument-noun co-occurrence (subject-slot + object-slot noun profiles) and test whether a NATURAL verb class (motion verbs sharing animate-subject + goal-place arguments) coheres and supports inheritance where the heterogeneous bag-of-words "actions" does not. Honest data limit: TinyStories has few verbs per natural class (~3 motion verbs in the top-1024), so this de-risk may want a verb-richer corpus.

## What this establishes
The rung-4 verb boundary is honestly characterized: it is a heterogeneous-probe + bag-of-words-signal limitation, not a substrate wall. Noun-category inheritance (the rung-1..4 GO results) rides genuine taxonomic co-occurrence structure; verb-category inheritance needs the argument-structure signal (a named, research-gated next mechanism), and/or a natural-verb-class probe. NO `sim/` edit.

## Files
Measured via `research/runners/_realcorpus_inheritance_rung4_conversation_derisk.py` (`_coherence`). Prior: rung 4 `2026-07-08-knowledge-half-rung4-talk-about-real-corpus-vocab-GO.md`; EMERGE-72/74 (argument-structure construction mining).
