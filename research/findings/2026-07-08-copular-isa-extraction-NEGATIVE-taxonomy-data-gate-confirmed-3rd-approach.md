# Copular is-a extraction (the explicit-definitional angle) is ALSO NEGATIVE on WikiText — confirming the multi-level taxonomy DATA-gate via a THIRD, mechanistically-distinct approach. Even NP-head "X is a [adj]* Y" mining yields NO usable is-a hierarchy (1 spurious "hub" = song titles). The stacked-pooler taxonomy mechanism is ready; the gate is a genuinely is-a-structured corpus, which the available corpora do not provide. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_copular_isa_miner_derisk.py`. numpy. NO `sim/` edit.
**Verdict:** NEGATIVE (data-gate confirmed) — explicit copular is-a extraction does not unlock the taxonomy on WikiText.

## Why this ran (the untested taxonomy angle)
The multi-level taxonomy was characterized as a DATA gate via TWO distributional approaches (centroid clustering of fine-cluster codes; stacked-pooler co-occurrence supers) × two corpora (TinyStories NEGATIVE, WikiText degenerate) + a dictionary-genus attempt (noisy). The remaining UNTESTED angle is EXPLICIT extraction: mine the is-a signal directly from definitional copular sentences ("a cat is an animal", "a robin is a bird") — which encyclopedic text (WikiText) should contain, unlike children's stories. This is mechanistically distinct from the distributional clustering (explicit relation extraction, not co-occurrence).

## What was built + the result
A copular miner: "<child> is a/an [adj]* <parentNOUN>" taking the NP-HEAD noun (skipping attributive adjectives via common adjectival/participial suffixes), filtering `unk`, keeping noun→noun, requiring HUB superordinates (≥3 distinct children = a real category).
```
WikiText: 991 raw copular hits -> 945 distinct pairs -> 41 pairs at >=2x -> 1 HUB superordinate
  the 1 "hub" = "song" (children: love, joyful, blue, imagine)  <- SONG TITLES ("Love is a song"), NOT a taxonomy
VERDICT: DATA-GATE HOLDS
```
The naive first pass (no NP-head, no hub filter) *falsely* passed a loose gate on pure noise (parents "often/known/used/generally" = adjectives/adverbs; children like "and"); the refined NP-head + hub-superordinate filter reveals the truth: **no usable multi-level is-a graph** — the sole "hub" is a coincidental song-title cluster.

## The decisive conclusion (the taxonomy data-gate, triply-confirmed)
Three mechanistically-distinct approaches now all fail to extract a multi-level is-a hierarchy from the available corpora:
1. **Distributional centroid** (cluster fine-cluster code centroids into supers) — NEGATIVE (real ≈ deranged).
2. **Distributional co-occurrence** (stacked-pooler second-order co-occurrence supers) — NEGATIVE (TinyStories) / degenerate (WikiText).
3. **Explicit copular extraction** (NP-head "X is a Y" mining) — NEGATIVE (this: no usable hubs; + the dictionary-genus was noisy).

⇒ The multi-level taxonomy is a genuine DATA gate: the EMERGE-44/45 stacked-pooler mechanism is validated + ready on is-a-structured synthetic data, but NO available natural corpus (children's stories, encyclopedic WikiText) provides a cleanly-extractable unsupervised is-a signal — neither distributionally nor via explicit copular definitions (children's stories lack definitions; WikiText's copulas are dominated by predicate-nominal descriptions/titles, not clean taxonomic is-a). The lever is a genuinely taxonomically-structured corpus (a curated encyclopedic/definitional is-a resource, or WordNet-style is-a text) — a data-acquisition step, not a mechanism gap. Single-level reasoning (inherit + cancel, comprehensively GO) stands.

## Files
`research/runners/_realcorpus_copular_isa_miner_derisk.py`. Prior taxonomy data-gate findings: `2026-07-08-multilevel-taxonomy-stacked-pooler-cooccurrence-ALSO-NEGATIVE-data-gate-confirmed.md`, `2026-07-08-emergent-is-a-from-dictionary-genus-NOISY-taxonomy-data-gate-confirmed.md`. The ready mechanism: EMERGE-44/45 (stacked-pooler multi-level taxonomy, GO on synthetic is-a data).
