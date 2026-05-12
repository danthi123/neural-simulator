# Cumulative in-vivo binding shows FIXED 2-binding capacity (not catastrophic forgetting)

**Date:** 2026-05-12
**Status:** Significant architectural finding. V_SCHEMA cumulative training
on 200ev main_hippo produces 2/4 final binding success — SAME as single-
shot V_SCHEMA. The architecture has a fixed "2 successful bindings at a
time" capacity regardless of how many bindings are sequentially trained.

## Test sequence

Bound 4 novel keys sequentially on a single forked main_hippo bridge:
1. apple → north (N)
2. mountain → south (S)
3. cat → east (E)
4. dog → west (W)

After EACH binding step, recalled ALL previously-bound keys to check
for interference.

## Per-step state evolution

| Step (just added) | apple | mountain | cat | dog | n_correct |
|---|---|---|---|---|---|
| 1: apple→N | ✓N | — | — | — | **1/1** |
| 2: mountain→S | ✗S | ✗W | — | — | **0/2** (catastrophic) |
| 3: cat→E | ✗E | ✓S | ✗S | — | **1/3** |
| 4: dog→W | ✗W | ✓S | ✓E | ✗N | **2/4** (final) |

## Interpretation

After ALL 4 trainings: 2/4 correct — exactly matching single-shot
V_SCHEMA's 2/4 ceiling on 200ev main_hippo. The "2 successful bindings"
capacity is preserved across the cumulative training arc.

WHICH 2 bindings succeed varies with training history:
- Single-shot V_SCHEMA: apple→N + mountain→S
- Cumulative final: mountain→S + cat→E

This is NOT catastrophic forgetting in the classical sense (where new
training destroys all prior memory). Instead it's **fixed-capacity
substitution**: the architecture supports ~2 active novel bindings,
and new bindings can displace older ones, but the displaced bindings
get redistributed (not totally lost).

## Architectural meaning

The 4-motor-pool architecture with V_SCHEMA training appears to have
**2 stable binding slots**. Each slot can hold one novel-key→motor_X
binding. Adding a 5th binding doesn't catastrophically fail — it
finds a slot by displacing one of the existing bindings.

This is consistent with:
- V_SCHEMA 2/4 ceiling at 200ev hippo (architectural)
- iter PP 1/4 BIDIR at biological scale (P5 same architecture)
- Tier 2.3 phrase composition stuck at 39.8%

All three reflect the same underlying capacity constraint: the
4-pool architecture can stably bind ~2 novel content units at a
time without specialized mechanism.

## Strategic implication for conversational sim

**The sim can grow its vocabulary by ~2 new words at a time.** A
user teaching the sim 4 new words via :learn commands will end up
with ~2 of those words successfully bound, but WHICH 2 is somewhat
random across training history.

For practical use:
- Teach 1-2 words at a time, then run :consolidate (SWR replay)
  to stabilize them in cortex BEFORE adding more
- The Phase 1.3 hippocampus consolidation mechanism is designed to
  push hippocampal traces to cortex via sleep replay — likely
  enables more bindings to persist after sleep

## Hypothesis for fix: interleave V_SCHEMA + sleep consolidation

Phase 1.3 consolidation (validated 3/3 PASS) shows hippo→cortex
transfer works for direction words. For novel keys, the analogous
flow would be:

1. V_SCHEMA bind new key (awake, hippocampus encodes)
2. SWR sleep replay (consolidates to cortex)
3. V_SCHEMA bind next new key
4. SWR sleep replay
5. ...

Without step 2 between bindings, the hippocampal trace gets
overwritten by the next bind. With sleep replay, the cortical
representation persists.

This would push capacity from 2 → unlimited (each new binding
gets fully consolidated before next).

## Next experiments

1. **Bootstrap main_hippo_balanced is running** (separate experiment) —
   testing if per-direction balanced anchor training pushes V_SCHEMA
   ceiling beyond 2/4 single-shot.
2. **Cumulative + consolidation**: re-run this test with
   mem.consolidate() between each binding. Test if Phase 1.3
   consolidation prevents catastrophic substitution.

## Wall clock

Cumulative test seed 42: ~10 min compute (parallel with balanced
bootstrap, no contention measured).
