---
type: plan
status: live
date: 2026-05-11
---

# P6 — Broca's area + compositional syntax design

**Date:** 2026-05-11
**Phase:** P6 of realigned plan v3
**Catalog entries:** G.12 (Broca's area)
**Citations:** Kandel 6e Ch 55 pp 1382–1384, Fig 55-6
**Prerequisites:** P5 ventral semantic stream

## Why this comes after P5

The catalog G.12 entry lists "**language model substrate**" as a
prerequisite. The language model substrate = the semantic
representation (P5 ventral stream + Wernicke's) + the engram-tagged
episode encoding (P1-P4). With those in place, P6 adds Broca's-style
compositional generation.

Replaces the failed Tier 2.3 PFC verb pool design (2026-05-07
finding: 39.8% phrase accuracy, phrase < direction-only across all
6 seeds — action_gate's indiscriminate motor boost was the bug).
P6 uses the catalog-grounded Broca's framing instead of inventing
action_gate.

## What Broca's actually does

From G.12 (Kandel pp 1382–1384):

> Maps stored auditory word-forms to motor articulation; supports
> comprehension of grammatically complex (non-canonical) sentences.
> Damage → labored, agrammatic speech, retained noun selection,
> **lost function-word/verb use**, repetition deficit.

So Broca's:
1. **Builds sentence structure** from words (grammar)
2. **Maps to motor articulation** (the sensorimotor half of Hickok
   & Poeppel dorsal stream)
3. **Distinguishes function-words from content-words** (the
   agrammatism pattern in lesions)

Behavioral validation:
> "The girl that the boy is chasing is tall" — comprehension fails
> (grammar-dependent). "The apple the girl ate was green" —
> succeeds (semantically constrained).

The first sentence requires parsing the embedded relative clause;
the second can be guessed from semantics alone. Lesions to Broca's
preserve semantics-based comprehension but lose grammar-based.

## What P6 builds

### Architecture additions

One new region:
- **`broca`** (~500 neurons, plastic recurrent, supports working-
  memory for the in-progress sentence) — the syntactic
  composition module.

Five new pathways:
- **`wernicke → broca`** (plastic) — word-level semantic content
  feeds into syntax composition.
- **`broca → broca`** (plastic recurrent) — working memory for the
  in-progress sentence. Supports holding "the apple" while
  receiving "the girl ate".
- **`semantic_cortex → broca`** (plastic) — semantic constraints
  on composition. "The apple ate alice" fails this constraint
  (apples don't eat); "alice ate the apple" passes.
- **`broca → motor_speech`** (plastic, NEW motor region) — speech
  motor articulation. Different from the existing motor_{N,E,S,W}
  navigation pool.
- **`broca → ec_context`** (plastic) — Broca's drives positional
  context during composition (sentence-level position is generated
  from the syntax, not from external input).

One new motor region:
- **`motor_speech`** (4 sub-pools for now, expand later — start
  small for validation) — speech articulation output. Each sub-pool
  could code for a phonological class (vowel, consonant cluster,
  etc.). For initial validation: 4 "slots" representing
  subject/verb/object/punctuation.

### Mechanism

**Comprehension** (P5 already does single-word):
```
"alice ate the apple"
  ↓ (each word in turn)
lang_input → wernicke → semantic_cortex
  ↓
broca builds up the sentence structure across positions
```

**Production**:
```
semantic_cortex (concept active: APPLE-PATIENT-EATEN-BY-ALICE)
  ↓
broca decomposes into sequence: ALICE → ATE → THE → APPLE
  ↓
semantic_cortex → wernicke → lang_output (word at each position)
  ↓
broca → motor_speech (articulation drive)
```

### Validation

**Test 1 — Two-word phrase composition**:
1. Train concepts: ALICE (engram tag in CA3/semantic_cortex),
   APPLE, ATE.
2. Train sentence-structure binding via P4 item-in-context:
   ("alice", pos_1) + ("ate", pos_2) + ("apple", pos_3) → episode
   tag "alice_ate_apple".
3. Drive partial cue: "alice" + position context.
4. Measure broca activation: does it predict the rest of the
   sentence?
5. PASS: broca's output drives the right (semantic_cortex →
   lang_output) sequence to produce "ate apple" continuation.

**Test 2 — Grammar vs semantics distinction**:
1. Train both "alice ate apple" and "apple ate alice" as episodes.
2. Drive partial: "alice" + position context.
3. PASS: broca + semantic_cortex jointly produce "alice ate apple"
   (semantically plausible), NOT "alice ... apple ate alice" (the
   constraint blocks reverse-direction reading).

**Test 3 — Broca's lesion (agrammatism)**:
1. After successful production, silence `broca`.
2. Drive lang_input with sentence.
3. PASS: comprehension still works for semantically constrained
   sentences ("the apple the girl ate was green" via wernicke +
   semantic_cortex). FAILS for grammar-dependent sentences ("the
   girl that the boy is chasing is tall" → ambiguous parse).

This is the Kandel-Ch-55-cited behavioral validation.

**Test 4 — Function-word vs content-word distinction**:
1. Train "alice ate the apple" and "the alice ate apple"
   (deliberately ungrammatical).
2. Measure broca's response to function-word ("the") vs
   content-word ("alice"):
3. PASS: broca develops distinct sub-population activation for
   function-words (Broca's specifically uses these to organize
   grammar).

## Replaces failed Tier 2.3

The 2026-05-07 Tier 2.3 PARTIAL result (39.8% phrase, phrase <
direction-only across all seeds) was due to action_gate's
indiscriminate motor boost. P6 doesn't use action_gate at all:
- Composition happens in `broca` (proper PFC/Broca's analog), not
  via gain modulation on motor pools.
- `semantic_cortex` provides the meaning-level constraint, not a
  blanket excitability boost.
- Per-word selection happens through `semantic_cortex → wernicke →
  lang_output`, not by amplifying all 4 motor_X pools.

The action_gate redesign options I drafted in plan v2 (per-direction
PFC subpools / inhibitory action_gate) become unnecessary — the
catalog-grounded Broca's design solves the problem at the level it
should be solved (syntactic composition, not motor gating).

## Open questions

1. **What does `motor_speech` actually output?** Start with 4 slots
   = (subject, verb, object, end). Later: phoneme-level articulation
   if/when we wire to muscle (T3.C territory).

2. **How does `broca → broca` working memory bootstrap?** Real
   biology: Broca's exhibits sustained delay-period activity
   (G.08 in catalog). Need to validate that our recurrent
   connectivity supports this.

3. **Grammar via rules or via statistics?** Two options:
   - Rule-based: hard-code position-role bindings (subj@pos_1,
     verb@pos_2, obj@pos_3).
   - Emergent: train on many examples, let broca's recurrent
     connectivity learn the regularities.
   Biology suggests emergent (G.14: language is learned, not innate
   in detail). Start emergent.

4. **Multi-word vocabulary?** Need ≥ 10-20 words for grammar tests
   to be meaningful. Builds on P4+P5 vocab capacity.

## Sequencing

```
P1 (trisynaptic) — partial pass
P2 (engram tagging) — SHIPPED
P3 (concept replay) — design done
P4 (item-in-context) — design done
P5 (ventral semantic stream) — design done
    ↓
P6 (this design)
    ↓
P7 — sentence-level comprehension + production
P8+ — reasoning, conversation
```

## Effort estimate

- broca + motor_speech regions: ~50 LOC
- 5 new pathways: ~30 LOC
- Test runner with 4 tests: ~500 LOC
- Multi-seed validation: 3-5 days wall clock
- **Total: ~2 weeks implementation + 2-3 weeks validation**

## Why this matters

P6 is what makes "compositional language" possible. After P6:
- The sim can produce multi-word sentences.
- Word order conveys meaning (alice ate apple ≠ apple ate alice).
- Function words behave differently from content words.
- Broca's lesion reproduces the agrammatism phenotype.

This is the second-to-last architectural piece before "conversational
sim" becomes plausible. P7+ adds reasoning and dialogue management
on top.
