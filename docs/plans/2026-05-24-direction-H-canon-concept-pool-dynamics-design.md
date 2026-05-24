# Direction H: stronger (canon) concept-pool dynamics — design

**Date:** 2026-05-24
**Status:** DESIGN (queued; the cheapest biology iteration after pillar n=104 BOUNDARY)
**Predecessor:** pillar n=104 extended (6 substrate sequence-storage attempts all BOUNDARY); reviewer-driven Direction K characterization (substrate not load-bearing at dim-overkill; biologized pipeline too strict)

## Goal

Test whether changing v16 concept-pool dynamics from WEAK
(deliberate v14/v16 design) to CANON (motor-pool style; stronger
internal density + exc/inh weights) enables substrate-level
sequence-position retrieval that the engram-tag mechanism couldn't
achieve with weak dynamics.

## Hypothesis

Per pillar n=104 diagnosis: v16's WEAK concept-pool dynamics
(density 0.05, exc_weight 0.3, inh_weight 0.8) make all pool neurons
fire ~equally during engram capture; top-K cannot distinguish slot-i
from slot-j. CANON dynamics (density 0.10, exc_weight 2.0,
inh_weight 4.0; mirrors motor pools) might:
- Make pool firing more SELECTIVE (winner-take-most across slots)
- Or break the v14 "canon amplifies bias collapse" trainability
  (multi-concept Phase 1 binding would fail)

## Critical pre-registered control

Phase 1 multi-concept trainability MUST still pass after dynamics
change. If Phase 1 binding (validated 88.75% multi-seed; pillar
n=82-ish) drops below ~70%, the dynamics change is breaking the
foundation and Direction H is closed as NEGATIVE regardless of
sequence-storage outcome.

## Mechanism

1. Modify `build_concept_bridge` (or equivalent) to use canon
   dynamics on concept pools (motor-pool style: density=0.10,
   exc_weight=2.0, inh_weight=4.0)
2. Train via v16 recipe (200 events/word × 16 words)
3. CONTROL: run validated Phase 1 W→A test (concept-pool word →
   target pool); strict pre-registered ≥ 0.70 multi-seed
4. If Phase 1 OK: run Direction A v1's engram-tag sequence storage
   test on canon-dynamics substrate
5. Multi-seed strict top-1; same frozen 0.80 bar

## Outcomes pre-registered

**(a) Phase 1 CANON ≥ 0.70 + Sequence Storage ≥ 0.80:**
PILLAR n=105 candidate (canon concept-pool dynamics enables both
multi-concept trainability AND sequence storage). The v14
"canon amplifies bias collapse" finding was substrate-specific to
the prior recipe; canon dynamics work in the current substrate
context.

**(b) Phase 1 CANON ≥ 0.70 + Sequence Storage < 0.80:**
HONEST BOUNDARY. Canon dynamics preserve trainability but don't
help sequence-position retrieval. The bound is elsewhere (likely
the engram-tag top-K aggregation mechanism, regardless of dynamics).

**(c) Phase 1 CANON < 0.70:**
HONEST NEGATIVE. Canon dynamics break v14/v16 trainability per the
original finding; Direction H closed. Pivot to Direction I (PFC
sequence buffer) or Direction L (chat REPL on multitag).

## Cost

- Modify builder: ~30 min
- Train substrate (3 seeds × ~60 min): ~3 hr GPU
- Phase 1 control test: ~30 min
- Sequence storage test: ~30 min
- Smell test: ~15 min
- Adversarial review: ~30 min
- Total: ~5 hr

## Implementation order

1. Read `research/runners/concept_pool_demo.py` build_concept_bridge
   signature; add `canon_concept_dynamics` flag
2. Write `direction_H_canon_dynamics_smoke.py` — single-seed test of
   builder + Phase 1 control + sequence storage on small scale
3. If smoke OK: write `direction_H_canon_dynamics_full.py` —
   multi-seed full scale
4. Smell test mirror (anti-cheat controls reusing Direction A pattern)
5. Adversarial review

## Status

QUEUED. Per discipline: this is the next concrete autonomous chain
step. Implementation starts in next session OR via watchdog continuation.

## Honest scope (pre-stated)

Direction H is a substrate-architecture iteration test. The outcome
extends or contradicts the v14/v16 "canon amplifies bias collapse"
finding. The result is honest regardless:
- (a) is a substantive validated capability extension
- (b) is honest characterization that sequence storage bound is
  mechanism-level not dynamics-level
- (c) is honest reconfirmation of the v14/v16 trainability tradeoff

The 0.80 bar stays frozen. No protected/frozen/moat modification.
No autograd. Reuse-by-import only (modify only the builder, not
the validated training/encoding/retrieval primitives).
