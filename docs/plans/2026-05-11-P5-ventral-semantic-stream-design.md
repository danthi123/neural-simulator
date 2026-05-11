# P5 — Ventral semantic stream + Wernicke's area design

**Date:** 2026-05-11
**Phase:** P5 of realigned plan v3
**Catalog entries:** G.11 (dual-stream model of language), G.13
(Wernicke's area — auditory-to-semantic mapping)
**Citations:** Kandel 6e Ch 55 pp 1380–1387 (Hickok & Poeppel dual-
stream model); Kandel pp 1384–1385 (Wernicke's)
**Prerequisites:** P4 episodic encoder (which itself gates on P1+P2)

## Why this comes after P1-P4

The catalog G.13 entry explicitly lists "**semantic memory store**"
as a prerequisite for Wernicke's area. Semantic memory store = the
substrate that maps words to abstract concept representations. That
substrate is built by P1-P4:

- P1: trisynaptic loop with pattern separation + completion
- P2: engram tagging (concepts as named ensembles)
- P3: SWR replay (concepts consolidate to cortex)
- P4: item-in-context binding (word + position → episode)

With those in place, P5 adds the cortical layer that turns "an
engram ensemble" into "the meaning of a word."

## What G.11 + G.13 are

From catalog G.11 (Kandel pp 1380–1387, Hickok & Poeppel):

> Dorsal stream: posterior superior temporal → arcuate fasciculus →
> Broca's area (**sensorimotor mapping for speech production**).
> Ventral stream: superior + middle temporal → semantic interface
> (**sound→meaning**).

The current sim has a *dorsal-stream-like* path (`language_input → cortex
→ motor`) for direction-word binding. **What's missing is the
ventral semantic interface.**

From G.13 (Kandel pp 1384–1385):

> Left posterior superior temporal gyrus + middle temporal gyrus;
> ventral-stream comprehension. Selects words matching intended
> meaning.

Wernicke's area is the convergence point where words → concepts.
Lesion causes fluent paraphasic speech with comprehension deficit
(patient can talk but neither understands nor produces meaningful
content).

## What P5 builds

### Architecture additions

Two new regions:
- **`semantic_cortex`** (~1000 neurons, sparse code per concept,
  recurrent for stability) — the ventral-stream semantic store. Each
  concept activates a sparse pattern in this region. Functions like
  Patterson 2007's anterior-temporal-lobe hub (catalog references
  this as G.11 ventral stream).
- **`wernicke`** (~200 neurons, integrator) — bidirectional bridge
  between phonological (lang_input/lang_output) and semantic
  (semantic_cortex) representations. The "word↔meaning" interface.

Three new pathways:
- **`lang_input → wernicke`** (plastic) — phonological input → bridge
- **`wernicke → semantic_cortex`** (plastic) — meaning lookup
  (comprehension)
- **`semantic_cortex → wernicke`** (plastic, weaker) — meaning →
  word form selection (production / naming)

One bridging pathway (the key new piece):
- **`ca1 → semantic_cortex`** (plastic) — hippocampal episode
  encoding consolidates to semantic cortex during P3 sleep replay.
  This is how engram-tagged concepts become cortical meanings.

### Mechanism

**Comprehension** (input → meaning):
```
"apple" (sound) → lang_input → wernicke → semantic_cortex
                                              ↓
                                          (sparse pattern that IS "apple meaning")
```

**Recall / naming** (meaning → word):
```
semantic_cortex (concept active) → wernicke → lang_output
                                                  ↓
                                              "apple" (word emitted)
```

**Hippo → cortex consolidation** (P3 replay):
```
ca1 (episode) → semantic_cortex (concept) — via STDP during NREM SWR replay
```

After enough replay cycles, semantic_cortex activates the right
pattern in response to lang_input WITHOUT needing the hippocampus.
This is the catalog's "consolidation transforms labile traces into
durable, distributed cortical representations" (D.01).

### Why no separate "concept hub" beyond semantic_cortex

The original "Patterson hub-and-spoke" framing I invented earlier
(plan v2) would have added a `semantic_hub` region. The catalog-
grounded P5 instead uses `semantic_cortex` for the same role —
distributed sparse code per concept, no centralized "hub" beyond
the cortical population. This is more biology-faithful:
anterior-temporal-lobe semantic hub IS a cortical region, not a
separate computational class.

### Validation

**Test 1 — Comprehension (lang_input → semantic_cortex)**:
1. Encode 5 concepts using P1+P2+P4 (each gets an engram tag).
2. Run P3 concept replay to consolidate.
3. Drive `lang_input` with each concept's word pattern.
4. Measure `semantic_cortex` response.
5. PASS: each concept produces a distinct, stable sparse pattern in
   semantic_cortex (cosine of same-concept activations across
   trials > 0.6; cosine across different concepts < 0.3).

**Test 2 — Naming (semantic_cortex → lang_output)**:
1. Drive `semantic_cortex` with the stored pattern for concept X
   (via reverse-tag stimulation).
2. Measure `lang_output` response.
3. PASS: lang_output's top-activated pattern matches the word for X
   (closest cosine to the X word embedding).

**Test 3 — Hippo-independent recall**:
1. After consolidation, silence ca3+ca1 (set excitability_drive to
   strongly negative).
2. Drive `lang_input("apple")`.
3. Measure `semantic_cortex` response.
4. PASS: semantic_cortex still produces the "apple" pattern even
   without hippocampus (Phase 1.3-style anti-cheat).

**Test 4 — Aphasia analog**:
1. After successful comprehension, silence `wernicke`.
2. Drive `lang_input("apple")`.
3. Measure `lang_output`.
4. PASS: lang_output produces fluent but unrelated output (paraphasic;
   the dorsal motor path still functions but loses semantic targeting).
   This is the Wernicke's-lesion phenotype from catalog G.13.

## Open questions

1. **Sparsity of semantic_cortex codes?** Real anterior temporal
   lobe doesn't have a uniformly sparse code; some concepts cluster
   by semantic category. Start with ~5% sparse (similar to DG) and
   adjust.

2. **Should wernicke be unidirectional or recurrent?** Bidirectional
   word↔meaning suggests recurrent. Start with separate input/output
   pathways; add recurrent if needed.

3. **How do new concepts integrate?** Tse 2007 schema-supported
   memory says new concepts attach faster if they overlap an existing
   schema. Our P4 item-in-context tagging + P3 replay should handle
   this automatically — replay strengthens the new concept's
   semantic-cortex pattern alongside related existing patterns.

4. **Composition?** This is G.12 (Broca's, P6). P5 only handles
   single-word concepts. Sentence-level composition is P6 territory.

## Sequencing

```
P1 (trisynaptic) — partial pass (multi-seed in flight)
P2 (engram tagging) — SHIPPED
P3 (concept replay during NREM) — design done
P4 (item-in-context binding) — design done
    ↓
P5 (this design)
    ↓
P6 (Broca's + grammar)
    ↓
P7 conversational sim
```

## Effort estimate

- semantic_cortex + wernicke regions: ~50 LOC
- 4 new pathways: ~30 LOC
- Test runner with 4 tests: ~300 LOC
- Multi-seed validation: 2-3 days wall clock
- **Total: ~1 week implementation + 1-2 weeks validation**

## Why this matters

P5 is the leap from "concepts as ensembles" to "concepts as
meanings." After P5:
- Words have abstract semantic representations independent of motor
  pools.
- Comprehension is mechanistically distinct from generation.
- The semantic store survives hippo silencing (the durability test
  from catalog D.01).
- The architecture supports Wernicke's-style lesion experiments
  (G.13 behavioral validation).

This is the substrate the user's "concepts ≠ motor pools" insight
needs. After P5, the sim has BOTH the motor-pool grounding (for
action words) AND the semantic-cortex grounding (for abstract
concepts), unified through the engram-tag → semantic-cortex
consolidation.
