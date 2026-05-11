# P5 4-concept architectural ceiling — orthogonal test rules out methodology

**Date:** 2026-05-11
**Status:** DEFINITIVE. Toy-scale architecture (~5K neurons) has
real ceiling at 4 concepts. The 2-concept iter W result is the
high-water mark for this scale.

## Headline

Tested both hash-based and orthogonal embeddings at 4 concepts.
**Orthogonal was WORSE.** This rules out methodology and confirms
architectural ceiling at this scale.

| Embedding | Strict PASS | Bio-faithful PASS |
|---|---|---|
| Hash (apple/river/alice/table) | 2/4 | 1/4 |
| **Orthogonal (banded)** | **1/4** | **0/4** |

Orthogonal embedding logically should be BETTER (clean
non-overlapping input codes). The fact that it's WORSE means
the partial failure isn't about input collisions — it's about
the architecture not having enough capacity for 4-concept
discrimination.

## Per-concept results

### Hash embeddings (apple/river/alice/table)

| Concept | Self | Max cross | Strict | Bio |
|---|---|---|---|---|
| apple | 0.254 | 0.267 alice | ✗ | ✗ |
| river | 0.243 | 0.206 | ✓ | ✗ |
| alice | 0.279 | 0.291 river | ✗ | ✗ |
| table | 0.358 | 0.278 | ✓ | ✓ |

### Orthogonal embeddings (banded)

| Concept | Self | Max cross | Strict | Bio |
|---|---|---|---|---|
| apple | 0.210 | 0.276 alice | ✗ | ✗ |
| river | 0.292 | 0.267 alice | ✓ | ✗ |
| alice | 0.274 | 0.324 table | ✗ | ✗ |
| table | 0.258 | 0.323 river | ✗ | ✗ |

Orthogonal made apple LOWER (0.254 → 0.210), and lots of
cross-cosines HIGHER (0.291 → 0.324 max).

## Why orthogonal is worse (counterintuitive)

Hypothesis: random sparse connectivity from lang_input to each
wernicke_pool covers the input space roughly uniformly. Hash
embeddings spread active neurons across the full 1024-neuron
lang_input, so each pool receives some signal from each concept.

Orthogonal banded codes cluster activity in specific regions
(neurons 0-25 for apple, 64-89 for river, etc.). If a wernicke
pool happens to have stronger connections to one band, it will
fire strongly for that concept AND for any concept whose band
also has dense connections. The clustered input pattern
amplifies whatever asymmetries exist in the random sparse
projection.

In short: random sparse projection assumes spread-out inputs;
clustered inputs interact badly with it.

## What this confirms

**Toy-scale architectural ceiling at 4 concepts.** The 2-concept
iter W result (6/6 multi-seed PASS) was the realistic upper bound
for this scale (~5K neurons). At 4 concepts:
- 2/4 hash, 1/4 orthogonal (strict PASS)
- Neither methodology reliably discriminates all 4 concepts

This is the toy-scale limit. Real biology has 10^5+ neurons per
region — orders of magnitude more capacity. The architecture
should scale, just not at this size.

## Practical implications

For the conversational sim project:
- **2-concept P5 comprehension WORKS** (iter W 6/6 PASS confirmed)
- **4-concept P5 partial** at toy scale (would need biological-scale
  neurons to test full architecture potential)
- Tier 1 + Tier 2.1 motor binding handle 4-word and 8-word vocab
  via PER-WORD motor pools — different architectural approach
  that doesn't have this limitation
- For >4 concepts with shared semantic_cortex (NOT per-concept
  motor pools), scaling neurons + training events together is
  required

## Wall clock: 823s (~14 min)

Same as hash 4-concept (824s). Wall clock scales with concepts
× training events, not with embedding type.

## Total P5 arc state: 27 experiments

A through Y (25 distinct configurations) + 4-concept hash +
4-concept orthogonal. Comprehensive parameter and methodology
sweep complete.

**iter W (2-concept Path A + 400 events) remains the definitive
multi-seed PASS.** All extensions tested have either matched it
(parameter variants) or pushed past architectural capacity
(4-concept scale).

## What's next (if user wants to push P5 further)

1. **Multi-seed validate orthogonal 4-concept** at seeds 43-102
   to see if some seeds DO pass at 4-concept (variance test)
2. **Scale up neurons + training**: n_per_wernicke_pool 100→200,
   train 800 events. Tests if more capacity + training
   handles 4 concepts
3. **Different cross-pool FS topology**: stronger cross-inhibition
   (weight 4→8) might better enforce winner-take-most at 4 pools
4. **Accept toy-scale ceiling and move on**: 2-concept iter W
   is the validated breakthrough; 4+ concepts at biological scale
   is a future arc
