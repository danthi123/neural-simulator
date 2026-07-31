---
type: plan
status: live
date: 2026-05-25
---

# Direction 3 design: vocab scaling on bio_brain_regions (V=32 then V=64)

**Date:** 2026-05-25
**Status:** Brainstorm/design pass (Direction Q complete; per user ordered direction Q -> 3 -> 4 -> R, now executing Direction 3)

## Goal

Test whether the validated bio_brain_regions substrate (n=96/n=97/
n=98 pillars at V=16; load-ceiling map at L=7 OI 0.90+) supports
larger vocabularies. The 2026-05-24 load-ceiling map showed V=16
has "DRAMATICALLY more load headroom" than G.20 sparse V=160 (the
FHRR capacity envelope predicts capacity proportional to N_dim/V).

Direction 3 tests:
- V=32: 2x vocabulary; expect OI capacity roughly halved per FHRR
  algebra (L=3-4 might still pass; L=5+ might not)
- V=64: 4x vocabulary; expect OI L=2-3 still passing; higher loads
  fail

Each tier produces biology-translatable scaling data: the
parallel-matching mode-unification mechanism's V-vs-L envelope on
the validated substrate.

## Biology reference

The FHRR (Fourier Holographic Reduced Representations) algebra
capacity envelope (pillar n=87 + characterizations): for a vector
dimension N_dim and vocabulary size V, the load capacity L scales
roughly as L ~ N_dim/V. For our substrate's concept-pool union
N_dim ~ 3200 (16 pools x 200 neurons), V=16 yields L_capacity ~ 200
(far above the 7-slot gamma ceiling). V=32 would predict L_capacity
~ 100; V=64 ~ 50. Both still well above the L=7 ceiling so OI
should pass at every gamma slot - UNLESS the substrate's concept-
pool architecture itself doesn't extend cleanly to V=32+.

The latter is what Direction 3 actually tests. Per the 2026-05-22
ceiling map: at V=16 the bio_brain_regions architecture has 12
concept pools (4 noun + 4 verb + 4 adjective) + 4 motor pools.
Each word fires ITS OWN 200-neuron pool exclusively. Going to V=32
requires either:

**Option A: more concept pools** (extend the architecture from 12
to 24 or 32 pools): biologically appropriate (each concept has its
own cortical column-like representation); requires architectural
changes to the bridge builder

**Option B: sparse coding within existing pools** (multiple
concepts share each pool with different K-of-N sparse codes):
matches the G.20 sparse architecture; more capacity-efficient;
but requires bridge-builder changes to support sparse mode on
bio_brain_regions

**Option C: hybrid** (split each pool into N sub-pools holding
different concepts): biologically reasonable (cortical mini-columns
within columns); intermediate change to builder

## Architectural approach selection

**Option A (more pools) RECOMMENDED** for the cheapest first probe
because:

1. The existing v16 bridge architecture already supports arbitrary
   pool counts via the noun_pool_names / verb_pool_names /
   adjective_pool_names parameters of `build_biological_brain_regions`
2. No fundamental architecture change required; just additional
   vocab entries + same training schedule
3. Direct biology-translatable mapping: V=32 = 32 cortical columns,
   each its own dedicated 200-neuron pool with FS interneurons
4. Compounds on the validated v14/v16 production recipe

The trade-off: at V=32 the substrate is ~2x larger (16K neurons
total at V=32 vs 8K at V=16). Training wall-clock scales accordingly
(~2-3 hr per seed vs ~17 min per seed at V=16).

Option B (sparse coding) is the more capacity-efficient long-term
direction but requires substantial builder changes. Reserve for
Direction 3-prime.

## Pre-registered test + bar

**Test**: parallel-matching mode-unification at V=32 across the load
ladder {L=2, L=3, L=5}. The same primitives validated in pillars
n=93/n=94 + the 2026-05-24 load-ceiling map; same OB and OI readout
decoders.

**Bar UNCHANGED** (frozen 0.80 multi-seed strict; same as pillars
n=93 onwards):
- `DIRECTION_3_V32_PASS`: multi-seed-mean >= 0.80 at every load
  L in {2, 3, 5} on BOTH order-bearing AND order-invariant readouts
- `DIRECTION_3_V32_BOUNDARY`: either readout misses at some load;
  precise per-load breakdown
- `DIRECTION_3_V32_NEGATIVE`: most cells miss; substrate doesn't
  scale to V=32 (more substantial finding)

If V=32 PASSes, follow with V=64 (additional ~3-4 hr training).

## Cost estimate

Per the 2026-05-24 post-c roadmap: "each tier ~1.5-2 hr GPU;
iterative scaling". Updated based on Direction Q wall-clock
observations (smoke pattern scale-ups were 6x faster than the design
doc estimate):

- V=32 substrate train: ~2-3 hr GPU per seed; 3 seeds = ~6-9 hr
- V=32 parallel-matching probe: ~30 min CPU
- V=64 substrate train: ~4-6 hr GPU per seed; 3 seeds = ~12-18 hr
- V=64 probe: ~30 min CPU

Total Direction 3 (V=32 + V=64 if V=32 PASSes): ~20-30 hr GPU.
Substantial but well within autonomous-runs budget per user's
standing 24/7 autonomy.

## Files to create (writing-plans output expected)

- `research/findings/raw/direction_3_vocab_scaling_bridge_builder.py`
  - Wraps `build_biological_brain_regions` with extended noun /
    verb / adjective name lists (8 of each = 24 + 4 motor = 28 or
    32 by adding a 4th kind like prepositions)
- `research/findings/raw/direction_3_v32_probe.py`
  - Parallel-matching mode-unification probe at V=32; reuses pillars
    n=93/n=94 + load-ceiling map primitives byte-unchanged
- `research/findings/raw/direction_3_verdict.py`
  - Frozen-threshold verdict (similar to Direction Q's pattern)
- `tests/test_direction_3_grounding.py` (grounding pin)

## Pre-staged post-V32 chain

- V32 PASS: proceed to V=64 (same architecture, larger vocab)
- V32 PARTIAL: characterize which loads pass / fail; informs
  capacity-envelope scaling exponent
- V32 NEGATIVE: substrate's concept-pool architecture doesn't scale;
  pivot to Option B (sparse coding) or Direction 4 (cross-bridge)

## Discipline (binding)

- Bar UNCHANGED throughout (0.80 multi-seed; same as pillars n=93+)
- No protected/frozen/moat modification (build_biological_brain_regions
  byte-unchanged; Direction 3 uses its OWN vocab spec wrapper)
- No autograd
- GPU/CuPy for real runs; numpy only for cheap probes
- Honest propagation EVERY outcome both remotes
- Pre-launch grep confirmed before this design: no prior V=32 work
  on bio_brain_regions (G.20 sparse 5-bridge 320-concept is a
  DIFFERENT substrate; the bio_brain_regions vocab scaling is
  genuinely net-new)

## Continuation pointer for next session/watchdog

When the next watchdog cycle (or next session) reads this file +
AUTONOMOUS_STATE.md, the immediate next concrete action is:

1. Read this design doc
2. Invoke superpowers:writing-plans to produce
   `docs/plans/2026-05-25-direction-3-vocab-scaling-implementation.md`
   (TDD plan with bite-sized tasks; same pattern as Direction Q's
   implementation plan)
3. Invoke superpowers:subagent-driven-development to execute Tasks
   0 -> N
4. Controller-only decisive multi-seed GPU run when Task N is built
5. Smell-test + findings doc + pillar candidate if PASS

Direction Q infrastructure (verdict module pattern, grounding pin
pattern, multi-seed runner pattern, mandatory control pattern) is
proven; Direction 3 follows the same template.
