# P5 iter Z (interleaved training) reveals toy-scale limit

**Date:** 2026-05-11
**Status:** Critical scientific finding. Interleaved training
DESTROYS iter W's "6/6 PASS" — revealing it was partly an
artifact of asymmetric (apple-first) training.

## Headline

| Iter | Training | COMP margin (apple-dir) | NAMING margin |
|---|---|---|---|
| W | sequential (apple block, river block) | 6/6 positive | 3/6 partial |
| **Z** | **interleaved per-event** | **1/6 positive** | **4/6 positive** |

Per-seed:

| Seed | iter W COMP | iter Z COMP | iter W NAMING | iter Z NAMING |
|---|---|---|---|---|
| 42 | +0.050 | -0.039 | +0.009 | +0.065 |
| 43 | +0.103 | -0.012 | +0.028 | -0.027 |
| 44 | +0.118 | +0.054 | -0.093 | +0.018 |
| 100 | +0.066 | +0.000 | -0.009 | +0.053 |
| 101 | +0.074 | -0.012 | -0.017 | -0.018 |
| 102 | +0.100 | +0.000 | +0.009 | +0.009 |

COMP dropped from 6/6 → 1/6. NAMING slightly improved 3/6 → 4/6.

## What this reveals

**iter W's "6/6 PASS" was partly an artifact of training order.**

The iter W comprehension test:
```python
pass_comprehension = (cos_apple_self > 0.5) and (cos_apple_river < 0.4)
```

Tests: drive apple → response similar to apple-tag, NOT similar to
river-tag. Both conditions are about apple-direction.

Sequential training (apple block first, river block second) caused:
- Apple weights to consolidate STRONGLY (entire training pass for apple)
- River weights to grow SECONDARILY (with apple already strongly
  encoded, river training pushes against existing apple structure)
- Net: apple is over-represented in semantic_cortex; river is
  under-represented
- Result: apple_self is HIGH (apple is well-encoded), apple_river
  is LOW (river-tag points to underdeveloped river pattern)
- Margin apple_self - apple_river is LARGE → "PASS"

When training is interleaved per-event:
- Apple and river weights grow in alternation
- Both concepts get equal encoding pressure
- Neither concept dominates semantic_cortex
- apple_self drops (apple no longer over-learned)
- apple_river RISES (river-tag now matches semantic patterns
  similar to apple's because both are partially trained)
- Margin shrinks → "FAIL"

## The true state of P5 at toy scale

When training is FAIR (interleaved):
- COMP: 1/6 multi-seed (seed 44 only)
- NAMING: 4/6 positive (margins ~0.02-0.07)

This is the **honest underlying discrimination capability** of the
Path A multi-pool architecture at toy scale. iter W's apparent
strong PASS came from giving apple preferential training.

## What this means strategically

The conversational sim's bidirectional concept discrimination at
toy scale (~5K neurons) is genuinely PARTIAL. The architectural
recipe (multi-pool wernicke + cross-pool FS + 400 events) helps,
but:
- Sequential training artificially boosts one direction
- Interleaved training reveals true underlying signal (margins ~0.02-0.07)
- Both training orders give ~50% single-trial accuracy on bidirectional demo

iter W's COMP margin +0.05 wasn't entirely meaningful — most of it
was apple-overrepresentation, not symmetric bidirectional learning.

## Three options forward (honest priorities)

### A. Per-concept lang_output pools (architectural rework, ~3-4 hr)
Mirror Tier 1 motor architecture at the output: each concept gets
its own lang_output_pool. Forces symmetric per-concept output
paths regardless of upstream training asymmetry.

Highest likelihood of giving true bidirectional PASS.

### B. Stronger cross-pool FS inhibition (parametric, ~30 min)
Current cross_inhibition weight is 4.0. Bumping to 8.0 might
force sharper winner-take-most. Tests if FS strength is the
limiting factor in symmetric encoding.

### C. Accept toy-scale limit, document, move on
P5 ventral semantic at toy scale has been comprehensively
characterized. Tier 1/2.1 motor binding is genuinely usable.
Move forward with that for conversational milestones at
direction/synonym vocab.

## Tier 1 contrast: why motor pools work bidirectionally

Tier 1 architecture has:
- Per-action motor pools (separate regions)
- FS lateral inhibition within and between pools
- Per-action topographic bias on lang→motor weights
- Bidirectional training (paired lang_input(word) + motor_X
  teacher current = embodied co-firing)

The KEY difference: each ACTION has its own dedicated motor pool.
The architecture pre-allocates per-concept resources at the
output side.

P5 doesn't have per-concept lang_output pools — they all share
the same lang_output region with random sparse connectivity.
Even Path A's per-concept wernicke pools converge back through
shared semantic_cortex and shared lang_output.

This is why Tier 1 gives bidirectional 6/6 PASS and P5 doesn't.

## Honest path forward

For TRUE bidirectional concept recognition for non-motor
concepts, the architecture needs the same pattern that Tier 1
proves works: **per-concept dedicated output paths** (option A
above). The Path A multi-pool wernicke was a partial fix; it
needs per-concept lang_output pools too.

Estimated effort: 3-4 hours of code (add lang_output_pool_<i>
regions, route each wernicke_pool to its corresponding
lang_output_pool, update topographic bias function).

## Total P5 arc

28 distinct P5 experiments now. Cumulative findings:
- Single-region wernicke (iter A-Q): hard floor at margin ~0.05
- Path D scale-up: hurt comprehension
- Path A multi-pool wernicke (iter T): mixed multi-seed
- Path A + 400 events (iter W): 6/6 apple-direction PASS
  (asymmetric, training-order artifact)
- 4-concept scalability: architectural ceiling at toy scale
- Demo V1/V2/V3: ~50% single-trial accuracy
- iter Z interleaved training: reveals iter W was partly artifact

The architecture's TRUE bidirectional discrimination at toy
scale is partial. Honest, scientifically valuable finding.

The conversational sim's motor-binding side remains rock-solid;
the non-motor concept side needs option A (per-concept
lang_output pools) to achieve genuine bidirectional performance.
