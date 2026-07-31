---
type: plan
status: live
date: 2026-05-25
---

# Direction 4 design: cross-bridge composition on bio_brain_regions (mirror G.20 sparse 5-bridge pattern)

**Date:** 2026-05-25
**Status:** Brainstorm/design pass (pre-staged during Direction 3 V=32 smoke wait; per user ordered direction Q -> 3 -> 4 -> R, this is next after Direction 3)

## Goal

Test whether the validated bio_brain_regions substrate supports
cross-bridge composition like the G.20 sparse 5-bridge pattern
already validates at the 160-concept ensemble level (pillar n=95).
This extends the parallel-matching mode-unification mechanism from
single-substrate to multi-substrate composition on the more
biology-faithful bio_brain_regions architecture.

Pre-conditions (per dependency on Direction 3 outcome):
- If Direction 3 V=32 PASSes: single-bridge supports doubled vocab,
  cross-bridge probe extends to 5x32=160 cross-bridge cleanly
- If Direction 3 V=32 PARTIAL/NEGATIVE: cross-bridge has narrower
  per-bridge headroom; may need V=16 cross-bridge as first probe

## Biology reference

G.20 sparse 5-bridge pillar n=95: cross-bridge OB perfect (1.000)
at every load; cross-bridge OI L=5 at the boundary (0.77 multi-seed;
below 0.80 bar). The 160-concept-union load-ceiling map (2026-05-24)
characterized OI L=5 0.77 / L=6 0.45 / L=7 0.16 (chance) for the
cross-bridge case.

Bio_brain_regions analog has NOT been built. The 2026-05-24 post-c
roadmap explicitly identified this as the open frontier:

> "Direction 4: Cross-bridge composition on bio_brain_regions
> (the 160-ensemble pattern). If multiple bio_brain_regions
> substrates are trained on different vocabulary categories,
> cross-bridge composition extends the conversational vocabulary
> substantially.
> Test: train 5 bio_brain_regions bridges on noun/verb/adj/spatial/
> functional vocabularies; ensemble them; run parallel-matching
> mode-unification cross-bridge.
> Cost: per-bridge ~30 min train; full ensemble ~3 hours; cross-
> bridge probe ~10 min CPU."

The cost estimate may be optimistic (Direction 3 V=32 smoke
preliminary timing suggests ~25-40 min/seed at V=32 reduced scale).

## Approach selection

**Approach A: 5 bio_brain_regions bridges, each at V=16 with
DIFFERENT vocab category** (cheapest first):
- Bridge A: 16 nouns (apple, river, dog, cat, tree, bird, sun,
  moon, book, chair, house, wheel, ball, cup, lamp, road)
- Bridge B: 16 verbs (go, come, stop, look, walk, run, eat, sleep,
  sit, stand, jump, climb, throw, catch, lift, pull)
- Bridge C: 16 adjectives (big, small, hot, cold, fast, slow,
  bright, dark, loud, quiet, sweet, sour, heavy, light, sharp, soft)
- Bridge D: 16 spatial (north, east, south, west, up, down, left,
  right, in, out, near, far, top, bottom, center, side)
- Bridge E: 16 functional (i, you, he, she, the, a, and, or, with,
  for, this, that, these, those, what, when)
- TOTAL: 80 cross-bridge unique concepts
- Per-bridge ~17 min train (matching v14/v16 production)
- 5 bridges × 3 seeds = 15 bridge trains = ~4-5 hr GPU

**Approach B: Use Direction 3's V=32 bridge × 5 with different categories**
- More vocab but requires Direction 3 V=32 PASS first
- 5 bridges × 32 = 160 cross-bridge concepts
- Per-bridge ~30-40 min train at V=32; 5 × 3 seeds = ~15 hr GPU

**Recommended**: Approach A first (cheapest; doesn't depend on
Direction 3 outcome; directly addresses the post-c roadmap's
Direction 4 spec).

## Pre-registered test + bar

**Test**: cross-bridge parallel-matching mode-unification at load
ladder {L=2, L=3, L=5} on the union of 5 bio_brain_regions bridges.
Each composite samples K items uniformly from the 80-concept union;
parallel-matching decodes per-slot identification.

**Bar UNCHANGED** at 0.80 multi-seed (same as pillars n=93+ and
Directions Q, 3):
- `DIRECTION_4_PASS`: multi-seed-mean >= 0.80 at every L in {2, 3, 5}
  on BOTH OB AND OI readouts; matches G.20 sparse n=95 OB but
  improves the OI L=5 boundary
- `DIRECTION_4_BOUNDARY`: either readout misses; precise per-load
  breakdown; biology-translatable comparison to G.20 sparse n=95
  (does bio_brain_regions geometry match or exceed sparse-coding
  geometry?)
- `DIRECTION_4_NEGATIVE`: most cells miss; cross-bridge geometry
  on bio_brain_regions doesn't extend cleanly

## Cost estimate

- Approach A: ~4-5 hr GPU for 5 bridges × 3 seeds + cross-bridge
  probe ~30 min CPU = ~5-6 hr total
- Approach B: ~15+ hr GPU (depends on Direction 3 outcome)

Approach A is the cheapest first probe per autonomous-runs principle.

## Files to create (writing-plans output expected when Direction 3 lands)

- `research/findings/raw/direction_4_vocab_spec.py` - 5x16=80
  cross-bridge vocab definition
- `research/findings/raw/direction_4_bridge_builder.py` - reuses
  Direction 3 builder pattern but with per-bridge category vocab
- `research/findings/raw/direction_4_cross_bridge_probe.py` -
  reuses pillar n=95 cross-bridge probe primitives byte-unchanged
- `research/findings/raw/direction_4_verdict.py` - frozen verdict
  module mirroring Direction 3/Q pattern
- `tests/test_direction_4_grounding.py` - grounding pin

## Pre-staged post-Direction-4 chain

- PASS: pillar n=105 candidate (or n=106 if Direction 3 also PASSes
  and gets a pillar first); the bio_brain_regions cross-bridge
  composition is validated; conversational-capability foundation
  extended to 5x16 = 80 biology-faithful concepts
- BOUNDARY: precise comparison to G.20 sparse n=95 (which mechanism
  is the bottleneck: bio substrate geometry vs sparse coding?)
- NEGATIVE: cross-bridge requires sparse coding (Approach B not
  available without Direction 3 PASS); or fundamental architecture
  redesign

## Discipline (binding)

- Bar UNCHANGED throughout (0.80 multi-seed; same frozen value
  used in pillars n=93+ and Directions Q, 3)
- No protected/frozen/moat modification
- No autograd
- GPU/CuPy for training; numpy for cross-bridge probe (CPU-only
  per pillar n=95 pattern)
- Honest propagation EVERY outcome both remotes
- Pre-launch grep BEFORE starting Direction 4 implementation:
  confirm no prior bio_brain_regions cross-bridge work exists
  (only G.20 sparse cross-bridge exists per 2026-05-24 audit)
- Mandatory NMDA-on-equivalent (parallel-matching uses the same
  validated substrate state; no explicit NMDA control needed since
  the mode-unification primitive is established at pillar n=93)

## Continuation pointer

When Direction 3 V=32 smoke + decisive complete:
1. If Direction 3 PASS: this Direction 4 plan ready for
   writing-plans -> subagent-driven-development
2. If Direction 3 PARTIAL/NEGATIVE: re-evaluate; Approach A (V=16
   cross-bridge) still proceeds since it doesn't depend on V=32
   per-bridge result

The Direction Q infrastructure (frozen verdict pattern, grounding
pin pattern, multi-seed runner template) plus Direction 3
infrastructure (bio_brain_regions builder pattern) are reusable
templates that Direction 4 follows.
