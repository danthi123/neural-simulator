# Engram-tag stim-recall: 87.5% multi-seed (concept-concept semantic memory)

## Context

After the architecture-mismatch bug retraction (see
[`2026-05-14-CRITICAL-bug-compose-concept-architecture-mismatch.md`](2026-05-14-CRITICAL-bug-compose-concept-architecture-mismatch.md)),
the previous claims of "65% direct + 90% chained" were retracted to 0/8
strict / 1/4 chain on seed 42 with corrected bridge architecture.

This finding documents a **clean, multi-seed validated capability** that
survives the bug fix: engram-tag stim-recall at strong-encoding settings.

## Configuration

Bridge: v16 concept-pool architecture (16 pools = 4 motor + 4 noun + 4 verb
+ 4 adjective), trained via standard Phase 1 (200 events × 16 words,
interleaved, weak dynamics, topographic prior 3.0/0.3, orthogonal codes).

```bash
python -m research.runners.concept_pool_demo --seed N \
    --n-train-events 200 --n-lang-input 2048 --n-per-pool 200 \
    --n-fs-per-pool 24 --weak-concept-dynamics --interleaved \
    --topographic-factor 3.0 --off-target-factor 0.3 \
    --enable-adjective --orthogonal-codes --sparsity 0.05 \
    --save-bridge bridges/v16/seed${N}.simstate.h5
```

Eval: 8 concept pairs (apple:big, dog:small, cat:hot, river:cold,
go:look, come:stop, big:hot, small:cold) encoded with stronger settings:

```bash
python -m research.runners.compose_concept_engram \
    --load-bridge bridges/v16/seed${N}.simstate.h5 --seed N \
    --n-lang-input 2048 --n-per-pool 200 --n-fs-per-pool 24 \
    --n-words-for-orthogonal 16 --encoding-steps 500 --sparsity 0.05 \
    --pairs "apple:big,dog:small,cat:hot,river:cold,go:look,come:stop,big:hot,small:cold" \
    --balanced-teacher-pA 500.0 \
    --out engram_results/seed${N}.json
```

Two changes vs prior bug-affected runs:
1. `--balanced-teacher-pA 500.0` — drives both concept pools directly
   during encoding so they fire reliably (analog of motor_teacher but
   symmetric across both concepts).
2. `--encoding-steps 500` — 2.5× the default 200; gives engram tag time
   to capture stable co-firing across both concept pools.

## Results

Multi-seed (5 seeds × 8 pairs = 40 trials):

| Seed | Stim-recall | Assoc-recall |
|---|---|---|
| 42 | 7/8 | 2/8 |
| 43 | 6/8 | 3/8 |
| 44 | 8/8 | 2/8 |
| 45 | 8/8 | 3/8 |
| 46 | 6/8 | 1/8 |
| **Total** | **35/40 = 87.5%** | 11/40 = 27.5% |

### Chance baseline

**Stim-recall** (both A and B in lang_output top-5 of 16 words):
P = 5/16 × 4/15 = 8.3%

87.5% observed vs 8.3% chance: **massively above chance**.

**Assoc-recall** (B in non-A top-3 of 15 remaining words): P = 3/15 = 20%

27.5% observed vs 20% chance: barely above chance, not significant.

## What the test measures

### Stim-recall (the real signal)

1. Encode (A, B) by driving both `lang_input(A)` and `lang_input(B)`
   simultaneously, with teacher current on both concept pools.
2. Engram tag captures top-K co-fired neurons in concept-pool regions.
3. At test time: `stimulate_tag(tag_name)` drives the tagged neurons.
4. Read `lang_output` firing pattern, cosine-match to spelling patterns
   of all 16 vocab words.
5. PASS if BOTH A and B are in the top-5 by cosine score.

This works at 87.5% because:
- Engram tag captures concept-pool neurons for BOTH A and B
- Stimulating the tag fires both pools at once
- Both concept pools have reciprocal STDP-trained pathways to
  `language_output` (Phase 1 reciprocal binding)
- So lang_output produces a mixed spelling pattern with both words
  represented

This is **biologically faithful Tonegawa-style engram recall** (catalog
D.14). The tag is a learned ensemble. Stimulating it pattern-completes
to the full concept set.

### Assoc-recall (marginal)

1. Drive only `lang_input(A)` (cue alone, no engram stim).
2. Phase 1 weights fire `concept_pool_A` strongly.
3. Cross-pool transmission: does `concept_pool_B` fire?
4. lang_output reflects both pools' activity.
5. PASS if B is in top-3 of non-A scores.

This is the OUTPUT of cross-pool weights grown during encoding STDP.
With 500 encoding events + teacher current, cross-pool weights grow
SOME but not enough to reliably propagate. Result: 27.5% (barely above
chance 20%).

## Comparison with earlier (bug-affected) measurements

| Test | Old (bug) | Re-test (corrected) |
|---|---|---|
| Stim-recall (200 enc, no teacher) | 23/40 = 57.5% (claim) | 4/8 = 50% on seed 42 |
| Stim-recall (500 enc + teacher 500 pA) | not tested | **35/40 = 87.5% multi-seed** |
| Assoc-recall (pool-firing readout) | 26/40 = 65% (claim) | 11/40 = 27.5% multi-seed |
| Assoc-recall (lang-out cosine) | 12/40 = 30% (claim) | 11/40 = 27.5% multi-seed |

The bug was inflating BOTH stim and assoc results. With corrected
measurement, stim-recall is even HIGHER (87.5%) than the bug claimed
(57.5%) — because we added stronger encoding. But assoc-recall is
LOWER (27.5%) than claimed (65%) — the prior signal was the bug, not
real cross-pool propagation.

## Implications for conversational capability

**Capability achieved:**
- "Remember (apple, big)" — system encodes engram, names it "apple_big".
- "Recall (apple_big)" — stimulating the engram tag produces "apple"
  AND "big" reliably (87.5% multi-seed).

This is a real, biologically-grounded semantic memory primitive. It
matches the Liu 2012 inception-of-fear paradigm in spirit: tag an
ensemble, stim later, recover the bound state.

**Capability NOT yet achieved:**
- "User types apple alone, system associates with big" — only 27.5%
  multi-seed, barely above chance.

For the chat interface, this means:
- Tag-based recall is reliable (chat command: `/remember alice 42`
  → encode, `/recall alice` → stim, retrieve binding).
- Cue-based association is unreliable (chat command: `user types apple`
  → system says "big" — only sometimes correct).

The compose_concept_chat REPL operating in tag-stim mode is usable. In
cue-only mode it's noisy.

## Comparison to other validated capabilities

| Capability | Multi-seed result |
|---|---|
| Tier 1 4-word direction (bio_three_factor) | 6/6 BIDIR multi-seed |
| Tier 2.1 8-word synonym (bio_three_factor) | 6/6 BIDIR multi-seed |
| Synonym32 32-word multi-language (chat_speak) | 100% A→W seed 42 |
| Phase 1.3 hippocampus consolidation | 3/3 strict anti-cheat |
| P5 ventral semantic comprehension | 6/6 multi-seed |
| Encoding-axis 64-word | 3/3 GO unanimous |
| **Engram-tag stim-recall (THIS)** | **35/40 = 87.5%, 5/5 seeds ≥ 75%** |

This sits alongside the other validated capabilities as a legitimate
multi-seed result.

## v19 (cross-pool pathways) verdict

v19 added 240 plastic pathways between all 16 concept pool pairs, with
the cross_pool_concept gate FROZEN during Phase 1 and OPENED only
during encoding. Compared to v16 baseline at same encoding (seed 42):

| Metric | v16 + teacher 500 + enc 500 | v19 + teacher 500 + enc 500 |
|---|---|---|
| Stim-recall | 7/8 | 6/8 |
| Assoc-recall | 2/8 | 3/8 |

v19's cross-pool pathways HURT stim-recall slightly (6 vs 7) and
marginally help assoc-recall (3 vs 2). Net effect: ~neutral / slightly
negative. Not worth the architectural complexity.

**v19 conclusion: NEGATIVE.** The cross-pool plastic pathway approach
doesn't add measurable concept-concept binding capability beyond what
engram tagging already provides.

The v19 runtime-gate-management infrastructure (close gate during
Phase 1, open during encoding) remains in code as reusable
infrastructure for future experiments.

## Production recipe

For semantic memory at 87.5% stim-recall:

1. **Train v16 bridge** (standard 16-pool architecture, no cross-pool
   pathways):
   ```bash
   python -m research.runners.concept_pool_demo --seed N \
       --n-train-events 200 --n-lang-input 2048 --n-per-pool 200 \
       --n-fs-per-pool 24 --weak-concept-dynamics --interleaved \
       --topographic-factor 3.0 --off-target-factor 0.3 \
       --enable-adjective --orthogonal-codes --sparsity 0.05 \
       --save-bridge bridges/v16/seed${N}.simstate.h5
   ```

2. **Encode concept-concept engrams** with strong settings:
   ```bash
   python -m research.runners.compose_concept_engram \
       --load-bridge bridges/v16/seed${N}.simstate.h5 --seed N \
       --pairs "..." --encoding-steps 500 \
       --balanced-teacher-pA 500.0
   ```

3. **Recall via tag stim** (chat REPL): `compose_concept_chat.py` can
   operate in stim-recall mode for reliable retrieval.

## Files

- `research/findings/raw/g11_bg/compose_concept_strict/seed{42-46}_v16_teacher500_enc500.json`
- `research/findings/raw/g11_bg/compose_concept_strict/seed{42-46}_v16_teacher500_enc500.log`

## Open questions

1. **Does longer encoding (1000+ events) push stim-recall to 95%+?**
   Each additional teacher-driven encoding event should reinforce the
   engram. Test would take ~5 min per seed extra.

2. **Can assoc-recall be pushed above chance via a different mechanism?**
   - Hippocampus CA3 autoassociator (catalog D.13 pattern completion)
   - Replay-mediated consolidation (cross-pool weights grow during sleep)
   - Lateral concept-pool inhibition (forces winner-take-all to surface
     the strongest cross-pool target)

3. **How does stim-recall scale with vocab size?** Current test is 8
   pairs in 16-word vocab. Would 28-word v17 or 64-word vocab maintain
   similar accuracy? Critical for true semantic capacity.

4. **Multi-tag retrieval interference?** If the bridge has 50 encoded
   engrams, do later engrams overwrite earlier ones (catastrophic
   forgetting at the engram level)? Need Phase 1.4-style retention
   tests at the engram-tag granularity.

## Lessons

- **Architecture-matching matters.** The previous "65%/90%" claims were
  measurement bugs. With proper architecture matching, the genuine
  result is different but still real.
- **Engram-tag stim-recall is the workhorse.** Cross-pool plastic
  pathways add little for cue-only associative recall. The Tonegawa
  ensemble-based mechanism is the strong substrate.
- **Strong encoding matters.** 200 events + no teacher gives 50% stim.
  500 events + teacher current gives 87.5%. The same architecture can
  produce vastly different signal depending on encoding strength.
- **Concept-concept ≠ cue-driven retrieval.** The engram-tag is
  retrievable by stimulating the tag explicitly (87.5%), but not by
  driving one of its constituent cues alone (27.5%). These are
  different cognitive operations.
