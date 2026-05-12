# P5 iter W — CRITICAL CORRECTION: 6/6 PASS is asymmetric (apple-only)

**Date:** 2026-05-11
**Status:** IMPORTANT CORRECTION to the autonomous arc narrative.
The iter W "6/6 multi-seed COMP PASS" was tested only for apple-
direction discrimination. River-direction was never explicitly
tested. Demo data shows the architecture is **asymmetric**.

## The asymmetry

The iter W test code:
```python
pass_comprehension = (cos_apple_self > 0.5) and (cos_apple_river < 0.4)
```

This tests:
1. Drive lang_input(apple) → resulting sem_cortex pattern similar
   to apple-tagged ensemble (self > threshold)
2. Drive lang_input(apple) → resulting sem_cortex pattern NOT
   similar to river-tagged ensemble (cross < threshold)

**Both conditions test APPLE-DIRECTION ONLY.** River-direction
(does river-drive activate river-tag more than apple-tag?) was
never tested.

## Demo data exposes the asymmetry

V3 demo seed 42 (5-trial averaging):
| Trial | Input | Got | Verdict |
|---|---|---|---|
| 1 | apple | apple | ✓ |
| 2 | river | river | ✓ |
| 3 | apple | apple | ✓ |
| 4 | river | **apple** | ✗ |
| 5 | apple | **river** | ✗ |
| 6 | river | **apple** | ✗ |
| 7 | apple | apple | ✓ |
| 8 | river | **apple** | ✗ |

Apple direction: 3/4 correct (75%)
River direction: 1/4 correct (25%)

This pattern holds across most seeds. V3 multi-seed mean: 56%
(barely above chance 50%). The 50% baseline = apple-direction
mostly correct, river-direction mostly wrong, balancing out.

## Why this happened

The training paradigm:
- Encode apple: drive lang_input(apple), let STDP grow weights
- Encode river: drive lang_input(river), let STDP grow weights

Each concept gets its OWN training pass. STDP grows weights for
co-firing pairs during that concept's drive.

But CONCEPT REPLAY during sleep drives BOTH CA3 ensembles in
sequence. The semantic_cortex pattern that forms from concept
replay isn't necessarily symmetric:
- Apple consolidates strongly (some random asymmetry from seed)
- River may consolidate to a pattern that OVERLAPS with apple

Result: apple-tag is well-separated; river-tag overlaps with
apple-tag. Apple-direction discrimination works; river-direction
doesn't (when driven by river, both tags fire, apple-tag often
fires more strongly).

## What iter W actually validated

The iter W 6/6 multi-seed PASS is technically correct for what
it tested: apple-direction comprehension. Specifically:
- "Drive apple → sem_cortex pattern reactivates apple-tag" PASS
- "Drive apple → sem_cortex pattern doesn't activate river-tag" PASS

This is the asymmetric form of comprehension. The bidirectional
test (also testing river-direction) would yield ~3/6 multi-seed
PASS based on the demo data.

## Honest reframing of session

Original claim: "P5 comprehension 6/6 multi-seed PASS (iter W
breakthrough)"

Corrected claim: "P5 apple-direction comprehension 6/6 multi-seed
PASS at iter W; river-direction discrimination unreliable
(~25-40% per-trial accuracy)."

Bidirectional concept recognition (which is what a user would
need) is not yet validated at multi-seed.

## What's actually robust this session

| Capability | Status | Caveat |
|---|---|---|
| Tier 1 motor binding (4-word) | 6/6 PASS | Bidirectional W↔A 86%/98% |
| Tier 2.1 synonym (8-word) | 6/6 PASS | Bidirectional W↔A 60%/95% |
| P5 apple-direction comp | 6/6 PASS | Only ONE direction tested |
| **P5 bidirectional concept demo** | **~50% single-trial** | **At chance for usable behavior** |

The motor-binding architectures (Tier 1, Tier 2.1) genuinely
support bidirectional language↔action — validated with both
W→A and A→W metrics multi-seed.

The P5 ventral semantic architecture has a one-directional pass
only — apple recognition works, river recognition doesn't
symmetrically.

## What would fix the asymmetry

1. **Symmetric training paradigm**: interleave apple and river
   training (not sequential blocks) so STDP can't grow apple-
   biased weights without immediate river contrast.
2. **Contrastive training**: train apple while ACTIVELY
   suppressing river's tag (LTD on cross-concept connections).
3. **Per-concept lang_output pools**: force lang_output to have
   separable per-concept activation patterns, like Tier 1's
   per-action motor pools.
4. **Architectural pre-allocation** (Path G++): semantic_pool_0
   for apple, semantic_pool_1 for river, with cross-pool FS.

## Total session arc

After this correction, the autonomous arc's accomplishments:

**Robustly validated multi-seed (today):**
- Tier 1 motor binding 6/6 (bidirectional, 86%/98%)
- Tier 2.1 synonym binding 6/6 (bidirectional, 60%/95%)

**Partially validated:**
- P5 apple-direction comprehension 6/6 at iter W (one-directional)

**Demonstrated NOT yet usable as user-facing oracle:**
- P5 bidirectional concept recognition (~50% single-trial)
- Path G+ (multi-pool wernicke) introduces NAMING positive
  direction but unreliably

**Architectural mapping established:**
- 27 P5 experiments characterized the architecture's capabilities
- Toy-scale ceiling at 4 concepts (architectural, not methodology)
- 400 training events is the sweet spot
- Multi-pool wernicke + cross-pool FS is the proven pattern

## Bottom line (corrected)

The conversational sim is robust for **motor-bindable concepts**
(direction words at 4-word and 8-word vocabs). P5 ventral semantic
stream is **architecturally one-directional** — drives apple
recognition correctly but doesn't symmetrically learn river.
Bidirectional non-motor concept recognition needs further
architectural work (interleaved training, contrastive paradigm,
per-concept lang_output pools, OR biological scale).

This is the honest scientific state after ~17 hours of autonomous
investigation.
