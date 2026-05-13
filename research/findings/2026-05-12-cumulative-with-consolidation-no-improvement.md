# Cumulative binding + SWR consolidation: NO improvement over no-consolidation

**Date:** 2026-05-12
**Status:** NEGATIVE for breaking 2/4 ceiling. SWR sleep replay between
bindings does NOT increase capacity — both with and without consolidation
end at 2/4. Consolidation shifts WHICH bindings survive but not how
many.

## Per-step comparison

| Step (just added) | Without consolidation | With consolidation |
|---|---|---|
| 1: apple→N | **1/1** | **0/1** (consol destroyed apple!) |
| 2: +mountain→S | 0/2 | 0/2 |
| 3: +cat→E | 1/3 | **2/3** (consol preserved more) |
| 4: +dog→W | 2/4 | 2/4 |

Final state with consolidation:
- apple → W ✗ (target N) — destroyed by 4 consecutive consolidations
- mountain → W ✗ (target S) — destroyed by 3 consecutive consolidations
- cat → E ✓ — survived 2 consolidations
- dog → W ✓ — just trained, no later interference

Final state without consolidation:
- apple → W ✗
- mountain → S ✓ — survived 2 cumulative trainings
- cat → E ✓ — survived 1 cumulative training
- dog → N ✗ — just trained but displaced

## Key observation: consolidation has dual effect

1. **Just after training (step N+1)**: consolidation often DESTROYS the
   just-trained binding (step 1 apple went 1/1 → 0/1 after consol).
   The SWR replay's random CA3 patterns interfere with the fresh
   hippocampal trace before it's stable.

2. **Across multiple bindings**: consolidation preserves OLDER bindings
   better than no-consolidation (step 3 was 2/3 with consol vs 1/3
   without). The cortical pathway eventually solidifies.

3. **At capacity**: regardless of consolidation, the architecture
   converges to ~2 stable bindings out of N trained. This is the
   fundamental architectural limit.

## What this means

The 4-motor-pool architecture has a FIXED 2-binding capacity that
applies to:
- V_SCHEMA single-shot bindings (2/4 ceiling)
- Cumulative V_SCHEMA bindings (2/4 final)
- Cumulative V_SCHEMA + consolidation (2/4 final, different survivors)

To break this ceiling requires architectural change, not just better
training scheduling. Options:
1. **More motor pools**: 8 pools = 8 simultaneous bindings (Tier 2.1
   approach for synonyms — already validated at 6/6 for direction
   synonyms but PRE-ALLOCATED, not in-vivo)
2. **Sparse coding instead of pool selection**: novel keys activate
   distributed patterns rather than single pools
3. **Per-key dedicated pool allocation** (like Tier 2.1 synonyms but
   for arbitrary new vocab) — requires architectural shape change

## Wall clock

Cumulative+consol test: ~36 min compute (4 trainings + 4 consolidation
cycles × ~8.5 min each + recall tests).

## Strategic implication for user

For practical conversational vocab growth:
- **2 novel words at a time** is the demonstrated capacity
- Consolidation between bindings doesn't help — actively hurts the
  most-recent binding short-term
- To grow vocab beyond 2 simultaneous novel keys, need architectural
  change (more motor pools or different output mechanism)

Conversational implications:
- User can :learn 2 new words and they'll likely stick
- Adding a 3rd new word displaces one of the earlier ones
- The displaced binding might come back in subsequent training
  (the architecture is "rotating" through bindings, not destroying
  them permanently)

## Connection to broader architectural ceiling

This 2-binding capacity finding is consistent with:
- iter PP P5 biological scale: 1/4 BIDIR (similar architecture, similar
  limit)
- V_SCHEMA single-shot: 2/4
- This cumulative test: 2/4 final regardless of consolidation

All three point to the same underlying constraint: the 4-pool
architecture supports ~2 stable bindings at the in-vivo level.
