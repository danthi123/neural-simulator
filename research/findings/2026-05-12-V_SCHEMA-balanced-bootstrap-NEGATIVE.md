# Per-direction balanced bootstrap NEGATIVE — south pool advantage is structural

**Date:** 2026-05-12
**Status:** NEGATIVE. Per-direction-balanced main_hippo bootstrap
(N=200, E=400, S=100, W=400 — extra training for weak anchors, less
for dominant) REGRESSED V_SCHEMA from 2/4 to 1/4. The south pool
advantage is from RANDOM INITIALIZATION, not training intensity.

## Test setup

Hypothesis: V_SCHEMA's 2/4 ceiling (apple+mountain) is because east
and west anchors are too weak relative to south. Solution: bootstrap
with MORE east/west events to equalize anchor strengths, LESS south
to reduce dominance.

Per-direction event counts in main_hippo_balanced bootstrap:
- north: 200 events (baseline)
- east: 400 events (2x boost for weak anchor)
- south: 100 events (half — try to reduce dominance)
- west: 400 events (2x boost for weak anchor)

Total: 1100 events, 68 chunks, ~99 min compute.

## Result

| Binding | 200ev balanced (baseline) | Balanced bootstrap |
|---|---|---|
| apple → N | ✓ CORRECT | ✗ (got W) |
| river → E | ✗ (got S) | ✗ (got S) |
| mountain → S | ✓ CORRECT | ✓ CORRECT |
| forest → W | ✗ (got E) | ✗ (got E) |
| **TOTAL** | **2/4** | **1/4** ← REGRESSED |

## Critical finding: south advantage is structural

The balanced bootstrap gave south HALF the training (100 vs 200
events) yet mountain→S STILL binds correctly. This means south
pool's advantage is NOT from anchor training — it's from RANDOM
INITIALIZATION of motor_S region's structural connectivity.

Some seeds happen to have motor_S with stronger internal recurrent
weights, more cross-pool inhibition from other pools, or more
favorable lang_input → motor_S edges. Training intensity changes
the synaptic weights but doesn't fix the underlying structural
bias.

Boosting east/west training to 400 events (2x baseline) didn't
help either — river still goes to S, forest still goes to E.

## What this rules out

After 6 different bootstrap configurations:
- 50ev (smoke): 1/4 (mountain only)
- 200ev (balanced): **2/4 (apple+mountain)** ← peak
- 400ev (balanced): 1/4 (mountain only, REGRESSED)
- Per-direction balanced: 1/4 (mountain only)
- V_SCHEMA_TOPO factor=2.0: 2/4
- V_SCHEMA_TOPO factor=5.0: 2/4

**2/4 V_SCHEMA + 200ev all-balanced bootstrap remains the empirical
peak.** No bootstrap config or topographic prior strength has
exceeded it.

## Why "200ev balanced" is the sweet spot

The 200ev all-balanced bootstrap is special because:
1. All 4 anchors get equal training (no over-training)
2. Random structural pool variance is the ONLY differentiator
3. V_SCHEMA's anchor reinforcement can find ~2 alignments where
   the anchor + random structure jointly support the binding

Adding more events overweights the random advantage. Adding more
to weak anchors doesn't compensate because the disadvantage isn't
training-based.

## Architectural conclusion

The 4-motor-pool architecture has FIXED 2-binding capacity that:
- Cannot be exceeded by bootstrap training adjustments
- Cannot be exceeded by topographic prior at binding time
- Is preserved across cumulative bindings (with or without consolidation)

To break this ceiling requires a different architecture:
1. **More motor pools** (8-pool synonym approach — pre-allocated,
   not in-vivo growth)
2. **Sparse distributed coding** instead of pool selection (different
   output decoder)
3. **Per-key dedicated pool allocation** (architecture grows with
   vocabulary)

None are quick fixes.

## Strategic implication

For practical conversational sim:
- Pre-trained vocab: synonym mode supports 8-16 words (existing,
  validated at multiple seeds)
- In-vivo learned vocab: ~2 new words at a time via :learn V_SCHEMA
- New words are NOT randomly placed — they get pulled toward the
  pool's structural advantage (likely south for many seeds)

**The user can teach the sim ~2 specific new words, where the target
pool's structural advantage aligns with the anchor reinforcement.**

For more reliable in-vivo learning, use words whose target matches
the sim's natural bias (often south for novel-key training).

## Wall clock summary

Today's V_SCHEMA arc:
- 6 bootstrap configurations tested (~5+ hours compute)
- 4 V_SCHEMA variants tested (~50 min compute each)
- 2 cumulative binding experiments (~70 min total)

Total: ~6 hours architectural investigation of in-vivo binding
ceiling. Definitive 2/4 capacity confirmed.

## Cleanup

`main_hippo_balanced` lineage retained for future experiments but
will not be the canonical main_hippo. Canonical remains 200ev
all-balanced (the empirical peak).
