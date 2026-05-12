# P5 iter NN NEGATIVE — orthogonal codes FLIP winner but don't fix BIDIR

**Date:** 2026-05-12
**Status:** NEGATIVE. Orthogonal concept codes (zero apple/river overlap)
swap which direction wins but BIDIR still fails. Confirms biological-
scale failure is random structural pool variance, NOT input encoding
overlap.

## Hypothesis

vocab_to_drive_pattern gives apple/river ~9pp overlap (18/205 shared
lang_input neurons). Shared neurons get ambiguous topographic bias —
last-write-wins between apple's pool_0 bias and river's pool_1 bias.

iter NN test: replace with orthogonal_drive_pattern (zero overlap,
contiguous bands). Same biological scale (500-neuron pools), same
strong topographic (3.0/0.33), same seed 42.

## Result

| | iter LL (vocab + topo 1.5) | iter MM (vocab + topo 3.0) | iter NN (orthogonal + topo 3.0) |
|---|---|---|---|
| apple p0 | 218 | 211 | **217** |
| apple p1 | 223 | 217 | **212** |
| apple margin | -5 (WRONG) | -6 (WRONG) | **+5 (CORRECT)** |
| river p0 | 208 | 210 | **223** |
| river p1 | 216 | 227 | **220** |
| river margin | +8 (correct) | +17 (correct) | **-3 (WRONG)** |
| **BIDIR** | NO | NO | **NO (FLIPPED)** |
| selectivity | 0.001 | 0.006 | 0.001 |

**Apple-direction FLIPPED from wrong to correct.**
**River-direction FLIPPED from correct to wrong.**

The "structural pool advantage" SWAPPED with orthogonal codes:
- vocab codes (iter LL/MM): pool_1 had the natural advantage
- orthogonal codes (iter NN): pool_0 has the natural advantage

Neither configuration gives BIDIR. The architecture has a 1/2 BIDIR
ceiling at biological scale where ALWAYS ONE direction wins by
structural luck, and the other loses.

## Diagnosis: random structural bias is THE limit

iter LL/MM/NN demonstrate the fundamental issue:
- Wernicke pool internal recurrence (500 neurons × 0.05 density = 25
  internal connections/neuron) creates self-sustaining firing
- Random initialization makes ONE pool fire more vigorously than the
  other (per-seed structural advantage)
- Topographic bias prior (lang_input → wernicke_pool) can be 9x
  stronger but can't fully overcome internal recurrence dominance
- Changing input codes (orthogonal vs hashed) only changes WHICH
  pool has the structural advantage — it doesn't eliminate the
  asymmetry

This is the SAME failure mode at scale that iter AA seeds 44/101 had at
toy scale: per-seed structural pool dominance. iter AA's 4/6 BIDIR
ceiling at toy scale corresponds to a roughly 1/2 BIDIR floor at
biological scale (worse, not better).

## What this rules out

After 4 biological-scale iterations:
- ❌ Tier 1 cortical canon (iter KK): amplifies bias 2.5x
- ❌ Biological scale with weak dynamics (iter LL): collapses
  discrimination ratio from 1.4x → 1.04x
- ❌ Stronger topographic bias (iter MM): helps river +2x but apple
  unchanged (structural floor)
- ❌ Orthogonal codes (iter NN): swaps winner, doesn't fix BIDIR
- ❌ Multi-trial averaging (iter DD, separate finding): catastrophic
  regression at toy scale due to neural adaptation

## What's been confirmed

The architecture's discrimination at biological scale is **per-seed
structural-luck-driven**. The topographic bias PRIOR can win when
aligned with the random structural advantage (river-direction in
iter LL/MM, apple-direction in iter NN) but loses when conflicting.

## Next: architectural pivot required

After 4 systematic biological-scale tests this morning (+ ~38
iterations of P5 over 24+ hours), the per-concept pool architecture
has been thoroughly characterized as having a fundamental ceiling at
biological scale. No parameter combination has produced 2/2 BIDIR
on seed 42 at biological scale.

The path forward (per Pulvermüller embodied semantics + catalog G.11/
G.13 + K.01): **sensory grounding via Cluster K v2 visual cortex.**

Design document at `docs/plans/2026-05-12-P5-sensory-grounding-design.md`.

Mechanism: add a SECOND strong training signal (visual stream:
retina → V1 → V2 → IT → multimodal_hub) that's INDEPENDENT of random
connectivity. This mirrors Tier 1's success — motor teacher current
during training overrides random structure for direction words.
Visual teacher would do the same for abstract concepts.

Estimated effort: 1-2 days focused work (10 hr engineering + 4 hr
compute) for smoke validation, then multi-seed.

## Recommendation

Stop parameter exploration. Pivot to sensory grounding (Option B in
strategic options summary).

Alternative quick test before committing: iter OO (smaller pools at
biological scale, ~2 hr) tests if pool size specifically vs overall
scale was the issue. If iter OO also fails 6/6 BIDIR, architectural
pivot is definitive.
