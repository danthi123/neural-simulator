# Concept pool FS topology — design lessons from v1 FAIL

**Date:** 2026-05-13
**Status:** Lesson learned from v1 0/10 PASS result. v2 fix in flight.

## The lesson

When using "FS within-kind only" topology (deliberate, to enable
composition across kinds), the absolute count of pools per kind matters.

With FS within-kind only:
- Pool with N partners gets (N-1) cross-FS inhibition edges
- Each edge provides ~constant suppression (fs_to_motor_weight)
- Total within-kind FS suppression per pool = (N-1) × weight

If kind A has 4 pools and kind B has 2 pools:
- Pool in A receives 3 × W cross-FS inhibition
- Pool in B receives 1 × W cross-FS inhibition
- B's pools are 3× less suppressed than A's

Combined with internal recurrent gain (motor_exc_weight × density)
and NMDA bistability, the under-suppressed pool can lock into
sustained high firing that dominates EVERY stimulus, not just its
target word.

## Empirical observation (v1 seed 42)

| Kind | Pools | Cross-FS edges per pool | Result |
|---|---|---|---|
| Motor | 4 | 3 | Trained pools moderate, fired on target +/- 0.5 |
| Noun | 4 | 3 | Same — trained moderately |
| Verb | 2 | 1 | verb_pool_COME dominated, fired 2.8-3.2 on ALL 10 words |

## Two fix strategies

### Strategy A (simplest): equal pool count per kind

Make all kinds have the same pool count, even if some words are
artificial padding. With 4 pools per kind, FS topology is symmetric.

**Pro:** trivial to implement
**Con:** forces vocabulary structure (4 nouns minimum, 4 verbs minimum, etc.)
**Used in:** v2 (added STOP, LOOK to bring verb count to 4)

### Strategy B (flexible): FS weight scaling

Scale FS-to-pool weight by 1 / (n_pools_in_kind - 1) so total within-
kind inhibition per pool is constant.

```python
fs_strength_per_kind = base_fs_strength × 3 / (n_pools_in_kind - 1)
# 4 pools: 3/3 = 1× (unchanged)
# 3 pools: 3/2 = 1.5×
# 2 pools: 3/1 = 3×
```

**Pro:** arbitrary pool count per kind, automatic balance
**Con:** harder to implement (need per-kind weight parameter)
**Considered for:** future iteration if more flexibility needed

## Implementation note for Strategy B

If we later need flexible per-kind pool counts, add to
`build_biological_brain_regions._add_concept_kind`:

```python
def _add_concept_kind(kind, names, n_per, n_fs_per, enable_fs_for_kind):
    n_kinds = len(names)
    n_cross_partners = n_kinds - 1
    if n_cross_partners <= 0:
        return  # single pool, no within-kind FS makes sense

    # Scale FS-to-other-pool weight to keep total suppression constant
    base_fs_partners = 3  # reference: 4-pool kind (motor)
    fs_scale = base_fs_partners / max(n_cross_partners, 1)
    effective_fs_weight = fs_to_motor_weight * fs_scale

    # ...
    pathways.append(RegionPathway(
        from_region=fs_region,
        to_region=f"{kind}_pool_{other}",
        density=0.5,
        weight_mean=effective_fs_weight,  # <-- scaled
        ...
    ))
```

## NMDA bistability + reset window interaction

A second-order concern observed in v1: with 50-step reset (25ms),
NMDA decay (~150ms) doesn't complete between training events. This
means previous-event activity can persist into the next event,
biasing STDP outcomes.

Possible mitigations (if v2 still fails):
- Longer reset between events (300+ steps = 150+ ms, matches NMDA tau)
- Explicit NMDA voltage reset between events (bridge API change)
- Stronger FS to break recurrent self-sustaining loops

Cost: longer reset = 3× wall clock. Trade-off vs reliability.

## Recommended FS topology summary

For symmetric concept pool architecture:

1. **Equal pool count per kind** (current v2: 4 each for motor, noun, verb)
2. **Within-kind FS only** (cross-kind FS off; enables composition)
3. **FS strength constant per edge** (don't scale by count if Strategy A)
4. **Reset between events ≥ NMDA tau** (currently 50 steps = 25ms; may
   need 300+ steps to fully decay NMDA between events)
5. **Topographic prior at edge level** (Pulvermüller 2003: 2-4x target/
   off-target weight ratio; v2 uses 4x)

This combination should give the best chance of cross-category
isolation while preserving cross-category composition.
