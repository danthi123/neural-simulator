---
type: plan
status: live
date: 2026-05-15
---

# Vocab scaling: paths 1-3 for near-future work (post-60-word milestone)

## Context

Current state (2026-05-15 ~05:00 EDT):
- 60-word multi-bridge conversational system shipped + 5-seed validated
- 11 conversational features (sentence, negation, conjunction, possessive,
  pronoun, tense, comparison, yes/no, role/relational queries, CRUD)
- 91 unit tests, all CPU-only

Question: what's needed to close the gap to modern tiny LLMs
(~30K-128K tokens)?

## What has already been tested (verified before starting new work)

| Architecture | Vocab ceiling | Status | Notes |
|---|---|---|---|
| v16 concept-pool (16 dedicated output pools) | 16 words | 5-seed GO | Tonight's bridge architecture |
| v17 28-pool concept-pool | 28 words | NEGATIVE | 8 hypotheses failed at single-bridge (motor sat, lang_input cap) |
| Encoding-axis Tier 2.1 (sub-pops in 4 motor pools) | 8 words | 6-seed GO | bio_three_factor pipeline |
| Encoding-axis 16-word | 16 words | 5-seed GO | n_motor=2000 |
| Encoding-axis 32-word | 32 words | seed42 GO | n_motor=3000 |
| Encoding-axis 64-word | 64 words | 3-seed GO (primary 62.5%, synonym 17.5%) | n_motor=2000, n_lang=8192, encoding-scale validated |
| Encoding-axis 96-word XL | 96 words | NEGATIVE | Converges to 25% floor (1-of-4 dir); synonyms don't bind even at n_lang=16384 |
| Multi-bridge 60-word (tonight) | 60 unique | 5-seed validated (74.4%) | 5 × v16 bridges, dispatch layer |
| v18/v19 cross-pool concept pathways | various | NEGATIVE | STDP pathway weights stay weak |
| Synonym32 multi-language | 32 words | 100% A->W seed 42 | chat_speak pipeline |

**Key catalog reference: G.20 Pulvermüller distributed-coding**
(language-mechanisms-additions.md). Explicitly marked PARTIALLY
MISSING with the open finding: "The W->A ceiling is a property of
the 4-separate-pool architecture. With distributed preferred-direction
tuning... across shared pool, STDP can sculpt smoother direction
selectivity."

## Path 1 — Distributed encoding in shared pools (catalog G.20)

**Status: NOT YET BUILT (this is the path I'll prototype tonight)**

### What it is

Replace "16 pools × 200 neurons" (v16) with **"1 shared 2000-neuron
pool"** where each concept is an engram tag (top-K=100 sparse pattern)
captured from the shared pool during encoding.

- No pool-per-concept allocation
- Population coding: concept identity = which subset of shared neurons
  cofired
- Capacity in principle: C(2000, 100) ≈ 10²¹⁵ patterns; practical
  discrimination depends on Hamming separation
- Catalog: G.20 Pulvermüller cortical word ensembles, D.14 Tonegawa
  engram cells

### Why this differs from prior work

- v17 28-pool: still had pool-per-concept (just more pools). Failed
  via motor saturation + lang_input collision.
- Encoding-axis 64-word: has 4 master motor pools w/ sub-pops, NOT
  truly distributed in a single shared pool. Synonyms within a pool
  collide at 16 sub-pops/pool (validated 96-word NEGATIVE).
- v18/v19 cross-pool pathways: tried to grow STDP weights between
  pool-per-concept regions. STDP didn't grow strongly enough.
- The G.20-style "one large pool with sparse population-coded engram
  tags as concept storage" has not been built.

### Prototype plan

```python
# concept_pool_demo_shared.py (new):
# - 1 shared_concept_pool (2000 neurons, 200 FS)
# - lang_input -> shared pool (plastic, gate-tagged)
# - shared pool -> language_output (reciprocal, plastic)
# - Per-concept training: orthogonal lang_input drive +
#   per-concept topographic prior on a SUBSET of the shared pool
#   (e.g. word N is "preferred" for shared_pool[N*40 : (N+1)*40])
# - After Phase 1 training, commit engram tag for each word:
#   start_engram_recording(word) -> drive word -> commit_engram_tag(top_k=100)
# - Validation: stim each tag -> read lang_output -> cosine match
#   to all N words. PASS if target word is rank 1.
```

Capacity sweep: test at N = 16, 32, 64, 128, 256 concepts in one pool.
Each test takes ~15 min (single-pool training is cheaper than 16-pool).

### Predicted outcomes

Hypothesis A: Capacity scales to 100+ concepts per shared pool.
- Engram tags are sparse (100/2000 = 5% active), so 2 random tags
  have ~5 neurons overlap by chance.
- Topographic prior should give per-concept selectivity.
- Stim-recall of correct tag should produce concentrated firing in
  the trained "preferred" subset.

Hypothesis B: Discrimination breaks down past 50-100 concepts due to
overlap.
- If capacity caps at ~50-100, multi-bridge × 100 concepts gives
  500-1000 word vocab — already a huge improvement.

Hypothesis C: Topographic prior alone isn't enough; FS lateral
inhibition across the shared pool is required.
- Would need per-concept FS structure (selective WTA across
  shared pool subsets).

### Decision tree

If 64+ concepts validate in single shared pool: ship as production
architecture. Multi-bridge gives 5 × N words trivially.

If 16-32 concepts ceiling: matches current v16 ceiling. Means
distributed-coding-only doesn't help; revisit path 2/3.

If failure: characterize the failure mode (overlap, weight collapse,
etc.) and feed back to catalog G.20.

## Path 2 — Subword tokenization (BPE-style morpheme bridges)

**Status: NOT YET BUILT**

### What it is

Tokenize input as morphemes/BPE units BEFORE encoding:
- "running" → ["run", "ing"]
- "dogs" → ["dog", "s"]
- "unhappy" → ["un", "happy"]

Each morpheme has its own engram tag. Combinations build via tag
co-firing (catalog D.14 composition).

### Why it's powerful

- ~6,000 morphemes in English yield ~60,000+ words via composition
- Combinatorial reach: 10× per morpheme bridge
- Combined with path 1 (assume 100 morphemes per shared bridge):
  60 bridges × 100 morphemes = 6,000 morphemes → 60K-word vocab

### Biology

Bozic 2010, Marslen-Wilson 2007: left inferior frontal gyrus
performs morphological decomposition. Real cortex doesn't store
"unhappy" as one engram — it stores "un" + "happy" and composes.

### Cost

~1 week to build:
- BPE tokenizer (SentencePiece or custom)
- Morpheme dictionary from common 6K-token list
- Composition layer in chat REPL (co-fire morpheme engrams)
- Eval suite for compositional accuracy

### Status

Should follow path 1 validation. If shared-pool encoding handles
100+ concepts per bridge, this is the clean multiplier.

## Path 3 — Hierarchical concept trees (hub-and-spoke)

**Status: NOT YET BUILT**

### What it is

Patterson 2007 hub-and-spoke semantic memory:
- Top: abstract categories (animate, action, property)
- Middle: subcategories (mammal, motion, color)
- Leaves: specific concepts (dog, run, red)

Each level uses smaller pool-per-concept at coarser granularity:
- Top: 4 categories × 200 neurons = 800 neurons
- Mid: 4×4 = 16 subcategories × 100 neurons = 1600 neurons
- Leaves: 4×4×4 = 64 concrete × 50 neurons = 3200 neurons

Total: 5600 neurons for 64 leaf concepts (vs v16's 3200 for 16
concepts). 4x capacity with hierarchical sharing.

### Biology

Patterson, Nestor, Rogers 2007: anterior temporal lobe as semantic
hub; modality-specific cortices as spokes. Lesion → semantic
dementia with characteristic graded loss (abstract → concrete).

### Cost

~2-3 weeks:
- Bigger architecture rework (3 tiered region groups)
- Curriculum learning: train top first, freeze, train mid, freeze,
  train leaves
- New evaluation: hierarchical similarity (dog should activate
  both mammal AND animate)

### Status

Lower priority than path 1+2. Useful if path 1 hits a ceiling at
~100 concepts. Hierarchical sharing keeps capacity scaling
sub-linearly with vocab.

## Path 4 — DEFERRED

Hybrid vector-DB + bridges. Engineering scale-out using FAISS or
sentence-transformers. Effective but "cheating" — doesn't demonstrate
biology-grounded conversational capability. User explicitly excluded
from near-future plan.

## Execution priority (autonomous)

1. **Tonight: path 1 prototype.** Single-pool distributed encoding
   with engram tags. Capacity sweep 16/32/64/128 concepts. ~3-5 hours
   wall clock if proceeds smoothly.
2. **If path 1 validates ≥ 64 concepts:** ship as new bridge
   architecture. Multi-bridge composition gives 5×64 = 320 words
   for free.
3. **Path 2 (subword tokenization):** queue for 1-week build after
   path 1 verdict.
4. **Path 3 (hierarchical trees):** queue if path 1+2 hit a wall
   under 1000 words.

## Open questions

- Does shared-pool encoding require Hippocampus + sleep consolidation
  for stable binding? (encoding-axis required this; engram-tags
  don't currently)
- Is the topographic-prior-on-shared-pool sufficient for selectivity,
  or do we need per-concept FS interneurons (which would re-introduce
  pool-per-concept structure)?
- How does cross-set retrieval work when concepts share a substrate?
  (multi-bridge cross-set was solved tonight via partial encoding;
  shared-pool may need different routing)

## References

- Catalog G.20: Pulvermüller distributed action-word ensembles
  (`references/language-mechanisms-additions.md` line 18)
- Catalog D.14: Tonegawa engram cells
  (`references/glossary.md` line 490)
- Existing v16 encoding-axis 64-word 3-seed GO:
  `research/findings/2026-05-10-encoding-axis-64word-3SEED-GO.md`
- Existing 96-word encoding-axis NEGATIVE:
  `research/findings/2026-05-11-96word-XL-encoding-NEGATIVE.md`
- Engram-tag stim-recall 87.5% multi-seed:
  `research/findings/2026-05-14-engram-stim-recall-multi-seed-VALIDATED.md`
- 60-word multi-bridge milestone:
  `research/findings/2026-05-15-multibridge-5seed-FINAL.md`
