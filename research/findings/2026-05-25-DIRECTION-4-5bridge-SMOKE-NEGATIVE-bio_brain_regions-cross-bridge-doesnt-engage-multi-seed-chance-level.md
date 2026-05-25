# Direction 4 5-bridge SMOKE = DIRECTION_4_NEGATIVE multi-seed (essentially chance at all 18 cells; OB mean 0.050/0.008/0.005 OI mean 0.007/0.000/0.000); bio_brain_regions cross-bridge composition does NOT engage at smoke scale on this substrate, in contrast to G.20 sparse n=95 which got OB 1.000 cross-bridge at V=160

**Date:** 2026-05-25 ~09:20 EDT
**Status:** DIRECTION_4_NEGATIVE at smoke scale; pivot per pre-registered chain to Approach B (Direction 3 V=32 × 5 = 160 cross-bridge concepts using already-promoted pillar n=105 substrates)

## What was tested

Direction 4 from the 2026-05-25 mechanism-class audit + design:
5 bio_brain_regions bridges (V=16 each, different vocab category)
= 80 cross-bridge concepts. Cross-bridge parallel-matching mode-
unification at L=[2,3,5] mirroring the G.20 sparse pillar n=95
pattern byte-unchanged.

Tasks 0-5 SCAFFOLDED (commits aeb9314 + d162dc3) via 2 subagent
dispatches. Task 5 controller-only smoke launched 07:11 EDT;
completed 09:13 EDT. Total wall 77.6 min training + 124.6s probe
= ~80 min.

Substrate (5 bridges, smoke params):
- A_nouns: 16 nouns (apple, river, dog, cat, tree, bird, sun, moon, book, chair, house, wheel, ball, cup, lamp, road)
- B_verbs: 16 verbs (go, come, stop, look, walk, run, eat, sleep, sit, stand, jump, climb, throw, catch, lift, pull)
- C_adj: 16 adjectives (big, small, hot, cold, fast, slow, bright, dark, loud, quiet, sweet, sour, heavy, light, sharp, soft)
- D_spatial: 16 spatial (north, east, south, west, up, down, left, right, in, out, near, far, top, bottom, center, side)
- E_functional: 16 functional (i, you, he, she, the, a, and, or, with, for, this, that, these, those, what, when)
- TOTAL = 80 cross-bridge concepts; 4288 neurons/bridge

Per-bridge per-seed training: 5-7 min smoke. 15 trainings = 77.6 min total.

Cross-bridge probe (CPU, reuses pillar n=95 byte-unchanged):
parallel_population_matching_batched on V=80 union; per_bridge_local
mean-centring (n=95 pattern); 200 trials/load.

## Result: DIRECTION_4_NEGATIVE

Multi-seed mean across 3 seeds:

| Load | OB (order-bearing) | OI (order-invariant) |
|---|---|---|
| L=2 | 0.050 | 0.007 |
| L=3 | 0.008 | 0.000 |
| L=5 | 0.005 | 0.000 |

Chance baseline 1/80 = 0.0125; observed values ~ chance or BELOW.

Per-seed L=2: seed 42 OB 0.045 / OI 0.015; seed 43 OB 0.045 / OI
0.000; seed 44 OB 0.060 / OI 0.005. Consistent multi-seed
chance-level performance.

Verdict (computed by frozen `direction_4_verdict.compute_verdict`
from recorded JSON): **DIRECTION_4_NEGATIVE**.

## Honest comparison to G.20 sparse cross-bridge (pillar n=95)

| Substrate | V | OB L=2 | OB L=3 | OB L=5 | OI L=5 |
|---|---|---|---|---|---|
| G.20 sparse cross-bridge (n=95) | 160 | 1.000 | 1.000 | 1.000 | 0.790 |
| bio_brain_regions cross-bridge (D4 smoke) | 80 | 0.050 | 0.008 | 0.005 | 0.000 |

The G.20 sparse substrate (Kanerva SDM, K-of-N scattered patterns,
2000-neuron shared pool) supports cross-bridge composition at V=160
with PERFECT OB and 0.79 OI L=5. The bio_brain_regions substrate
(concept-pool architecture, each concept = its own 100-neuron pool)
DOES NOT support cross-bridge composition at V=80 even at much
lower load.

This is a 20x-200x degradation. Not noise. Not a smoke artifact.

## Biology-translatable insight

The bio_brain_regions concept-pool architecture is fundamentally
different from G.20 sparse:
- **G.20 sparse**: each concept = scattered K-of-N pattern in a
  shared 2000-neuron pool; cross-bridge naturally composes because
  the union of bridges shares a common substrate geometry
- **bio_brain_regions**: each concept = its own dedicated 100-200
  neuron pool with FS interneurons; cross-bridge has DISJOINT
  per-bridge substrates that must be ground-symbol-derived
  INDEPENDENTLY before composition

The per_bridge_local mean-centring (which works for G.20 sparse)
may not be the right transform for bio_brain_regions cross-bridge:
each bridge's "local mean" is across that bridge's 16 concepts ONLY,
but the cross-bridge decoder operates on the union of 80 concepts.
The local mean centring removes a different bias per bridge, so the
80-concept ground-symbol space has 5 distinct local-mean biases mixed in.

A GLOBAL mean centring (across all 80 concepts after concatenation)
might produce a more uniform phasor space. This is exactly what
`cross_bridge_mode_unification_probe.py` does for G.20 sparse with
the `global_mean` option (vs `per_bridge_mean`).

But even with global-mean centring, the bio_brain_regions cross-bridge
may not work because the per-bridge substrates have DISJOINT
neuron-space identities (each pool is 100 neurons of a specific
concept; the union of 5 bridges = 80 disjoint 100-neuron pools, vs
G.20 sparse's overlapping K-of-N codes in a shared substrate).

## What this rules in vs out

**Rules out (at smoke scale)**: bio_brain_regions cross-bridge with
the per_bridge_local mean-centring pattern (byte-unchanged from
pillar n=95) does NOT engage at V=80. Different substrate geometry
than G.20 sparse.

**Does NOT rule out**:
- Production scale (n_per_pool=200, events=200, lang=2048): unlikely
  to fix this since smoke result is ~chance (not borderline)
- Global mean centring across 80 concepts: cheap probe; worth testing
- Approach B (D3 V=32 × 5 bridges = 160 cross-bridge): single-substrate
  + cross-bridge query; uses pillar n=105 validated substrate
- Sparse coding within bio_brain_regions pools: substantial
  architectural change

## Pre-registered post-NEGATIVE chain

Per the 2026-05-25 mechanism-class audit + Direction 4 design doc:

> "DIRECTION_4_NEGATIVE: cross-bridge requires sparse coding;
> pivot to Approach B (use D3 V=32 x 5 = 160 cross-bridge
> concepts) if D3 PASSed, otherwise re-think architecture"

D3 V=32 PASSED + PILLAR N=105 PROMOTED. So Approach B is available
as the next concrete action. Cost estimate ~7-15 hr GPU for 5
V=32 bridges × 3 seeds.

ALTERNATIVE cheaper next probe: test the global_mean centring on
the EXISTING D4 smoke cache (no new training needed; ~5 min CPU).
This characterizes whether the per_bridge_local centring choice
is the binding constraint.

## What is preserved unconditionally

- Pillar n=105 (D3 V=32 production PASS) stands UNAFFECTED
- Pillars n=93/n=94/n=95/n=96/n=97/n=98 stand UNAFFECTED
- bio_brain_regions substrate (build_biological_brain_regions
  byte-unchanged) stands
- G.20 sparse cross-bridge pillar n=95 stands UNAFFECTED
- Direction M deliverable (320-concept multi-bridge chat) stands
- No-confab moat 7/7 byte-identical
- Bar UNCHANGED at 0.80 throughout
- Direction 4 Tasks 0-5 infrastructure (verdict module + grounding
  pin + 5 builder wrappers + vocab spec + cross-bridge probe +
  5-bridge runner) reusable for any future cross-bridge investigation

## Discipline preserved

- Multi-seed [42, 43, 44] decisive (not one-seed artifact)
- Frozen verdict computed from recorded JSON; not tuned
- Honest propagation: NEGATIVE recorded as NEGATIVE; not spun
- Pre-registered chain identified Approach B as next concrete action
- Both remotes pushed
- ~80 min wall (much faster than 7-10 hr estimate)

## Files

- Runner: `research/findings/raw/direction_4_5bridge_runner.py`
- Vocab spec: `research/findings/raw/direction_4_vocab_spec.py`
- Bridge builders (5): `research/findings/raw/direction_4_bridge_builder.py`
- Cross-bridge probe: `research/findings/raw/direction_4_cross_bridge_probe.py`
- Verdict module (frozen): `research/findings/raw/direction_4_verdict.py`
- Result JSON: `research/findings/raw/direction_4_5bridge_smoke.json`
- Log: `research/findings/raw/direction_4_5bridge_smoke.log`
- Design doc: `docs/plans/2026-05-25-direction-4-cross-bridge-bio_brain_regions-design.md`
- Implementation plan: `docs/plans/2026-05-25-direction-4-cross-bridge-bio_brain_regions-implementation.md`
- Mechanism-class audit guide: `docs/plans/2026-05-25-prior-mechanism-class-audit-direction-selection-guide.md`
- G.20 sparse cross-bridge n=95 reference: `research/findings/2026-05-24-cross-bridge-OI-load-ceiling-map-extension-of-n95-ceiling-between-L4-and-L5.md`
