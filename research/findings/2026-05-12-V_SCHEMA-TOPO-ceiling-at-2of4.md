# V_SCHEMA_TOPO ceiling at 2/4 — topographic prior at binding time insufficient

**Date:** 2026-05-12
**Status:** NEGATIVE for >2/4. V_SCHEMA + topographic prior at binding
time (factor=2.0 and 5.0 tested) does NOT exceed plain V_SCHEMA's 2/4
on 200ev main_hippo. Pool_S structural dominance from bootstrap is
strong enough to overcome 5x topographic prior on novel-key edges.

## Results

| Variant | apple | river | mountain | forest | Score |
|---|---|---|---|---|---|
| V_SCHEMA (baseline, 200ev) | ✓N | ✗S | ✓S | ✗E | 2/4 |
| V_SCHEMA_TOPO factor=2.0 | ✓N | ✗S | ✓S | ✗S | 2/4 |
| V_SCHEMA_TOPO factor=5.0 | ✓N | ✗S | ✓S | ✗S | 2/4 |

Per-binding outcomes are IDENTICAL across factor values for apple,
river, mountain. Forest shifted from "→E" (plain V_SCHEMA) to "→S"
(both topo factors).

## Why topographic prior at binding time fails

The 200ev main_hippo bootstrap creates a "south" pool with strong
lang_input→motor_S edges across MANY language_input neurons (not
just the "south" anchor's active set). Active language_input neurons
for novel keys (apple's ~205 active, river's ~205, etc.) connect to
motor_S via these strong existing weights.

The topographic prior at binding time boosts the SPECIFIC novel-key →
target_motor edges by 2-5x. But:
- Apple's active edges: ~30K to motor_target boosted 5x
- The bootstrap's accumulated apple→motor_S edges: similar magnitude
  because of overlap between apple's active set and bootstrapped
  south-pool connections

The factor 5x boost on novel key's specific connections isn't
enough to overcome the SHARED edges where apple's active neurons
already have stronger weights to motor_S via prior training.

## The real bottleneck: pool balance in bootstrap

V_SCHEMA succeeds when:
1. Target pool's anchor word is well-trained AND
2. Target pool is NOT over-dominant

At 200ev main_hippo:
- "south" anchor + south pool: strong AND not over-dominant → mountain works
- "north" anchor: strong → apple works
- "east"/"west" anchors: too weak relative to south's structural pull
- topographic prior can't fix this — the south pool DRAINS attention

At 400ev main_hippo (tested separately):
- south pool becomes OVER-dominant → apple regresses to wrong pool
- All bindings except mountain pulled to south

So the sweet spot is narrow: 200ev provides 2/4 max with V_SCHEMA.

## What WOULD work (untested options)

1. **Per-direction-balanced bootstrap**: in consolidation_trainer,
   accept per-direction event counts. Give weaker directions
   (east/west) extra training to equalize pool strengths. Likely
   requires bootstrap-side code change.

2. **Cross-direction lateral inhibition during bootstrap**: stronger
   motor_FS cross-inhibition prevents one pool from dominating.

3. **Homeostatic firing rate target during bootstrap**: equalize
   pool activity targets via cfg.enable_homeostasis with appropriate
   target firing rates.

4. **Drop bootstrap pre-training entirely**: use a much weaker
   main_hippo or none at all, and rely entirely on V_SCHEMA's
   anchor reinforcement for new bindings. May give equal
   weights for all 4 anchors but compounds the difficulty of
   each binding.

## Strategic conclusion

The investigation is complete. **2/4 novel-key binding is the
demonstrated ceiling for the V_SCHEMA + main_hippo approach** with
any of the 6 anchor strength configurations tested (50/200/400 events
+ smoke vs hybrid SWR + 2.0 vs 5.0 topographic prior).

The architectural ceiling at biological scale for non-motor binding
is genuinely real and consistent across:
- P5 abstract concept binding (iter PP: 1/4 BIDIR)
- In-vivo novel-key binding (V_SCHEMA family: max 2/4)

Both reflect the same underlying issue: per-pool structural variance
in the 4-direction motor pool architecture creates unequal
"basins of attraction" that pull novel inputs unequally regardless
of training methodology.

## What we KNOW works (demonstrated capability)

- Tier 1 direction word binding: 6/6 BIDIR PASS multi-seed (74% W→A
  / 98% A→W at seed 42)
- Tier 2.1 synonym binding (8-word): 6/6 BIDIR PASS multi-seed
- Phase 1.3 hippocampus consolidation: 3/3 PASS
- Phase 1.4 catastrophic forgetting eval: 5/6 PASS

These provide the foundation for a conversational sim with ~8
direction-related words and continual learning capability.

## What requires architectural rework

- Robust >2/4 novel-key in-vivo binding
- P5 abstract concept binding at biological scale
- Compositional 2-word phrases (Tier 2.3 stuck at 39.8%)

These require either:
- Per-pool balanced architecture (homeostasis + lateral inhibition
  during bootstrap)
- Different output architecture (not 4 fixed motor pools)
- Pre-allocated novel-key pools (gives up on arbitrary new vocab)

## Wall clock summary (today's invivo investigation)

- BridgeMemory arch bug investigation + fix: ~30 min
- V_SCHEMA smoke seed 42: ~7 min (50ev hippo)
- V_SCHEMA seed 43 + 44 cleanup + redo: ~25 min
- Hybrid 200ev bootstrap: 53 min
- V_SCHEMA on 200ev: 8 min → **2/4 breakthrough**
- 400ev bootstrap: 112 min
- V_SCHEMA on 400ev: 8 min → 1/4 regression (sweet spot identified)
- V_SCHEMA_TOPO impl + factor 2.0 test: ~12 min
- V_SCHEMA_TOPO factor 5.0 test: ~8 min

**Total this morning's invivo work: ~4.5 hr compute + ~1 hr implementation.**

## Recommendation

V_SCHEMA + 200ev main_hippo achieves 2/4 in-vivo novel-key binding —
the **best biology-grounded mechanism demonstrated this session**.
Pushing further requires per-pool balanced bootstrap (next session
scope) or accepting the 2/4 as the demonstrated capability.

For the user's conversational goal:
- 8 direction-related words: 6/6 BIDIR multi-seed validated
- New words can be added in-vivo at 2/4 rate (apple→N + mountain→S
  pattern)
- Future work: balanced bootstrap to push toward 3-4/4
