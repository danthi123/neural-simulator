# GATE 2 OVERTURNS GATE 1: the 28-word "wall" is substantially UNDERTRAINING + single-shot noise, NOT a
# fundamental representation limit -- 2026-06-01

## What happened
Gate 1 (2026-05-31) concluded the 28-word front-end wall is a genuine representation limit (clean codes
plateau ~0.64, vs 16-word 100%). Gate 2 -- a controlled training pair to test a stronger topographic prior --
inadvertently REVEALED that Gate 1's conclusion was confounded: the `_v17_28word` bridge Gate 1 measured was
trained at only ~50 events/word, while the 16-word control bridge was trained at 200 events. Unfair
comparison.

## The decisive numbers (same validated capture+decode pipeline, clean 16-avg codes)
| bridge | training | clean pool-argmax | single-shot (k=1) | between-cos | best full-code NN |
|--------|---------:|------------------:|------------------:|------------:|------------------:|
| _v17 28-word (Gate 1) | ~50 ev | 0.643 | ~0.40 | 0.606 | 0.625 |
| Gate-2 baseline (topo 3.0) | 150 ev | **0.893** | 0.569 | 0.564 | 0.625 |
| Gate-2 strong (topo 10.0) | 150 ev | 0.857 | 0.647 | 0.532 | 0.688 |
| 16-word control | 200 ev | ~1.000 | 0.801 | -- | 1.000 |

## Two corrections, both honest
1. **The 28-word wall is NOT a fundamental representation limit.** With matched training (150 events) the
   clean 28-word recognition is 0.893 -- close to the 16-word 1.000, not the 0.64 "limit". The 0.64 was an
   UNDERTRAINED bridge (50 events). Gate 1's representation-limit verdict is RETRACTED.
2. **The single-shot "~50% wall" is largely NOISE / readout, not overlap.** Single-shot k=1 = 0.569
   (matches the documented v17 "~50% wall"); clean 16-avg = 0.893. So temporal integration / averaging (a
   biological readout = a longer integration window) recovers ~32 points. The codes carry the identity; the
   single-shot readout is noise-limited. (This mirrors the 2026-05-31 real-substrate boundary, also lifted
   by temporal integration.)

## The stronger topographic prior is NEUTRAL
topo 3.0 -> 10.0: clean pool-argmax 0.893 -> 0.857 (slightly worse), between-cos 0.564 -> 0.532 (slightly
better), best-NN 0.625 -> 0.688 (slightly better). No clear win. The cheap "stronger prior" lever does not
materially help beyond adequate training.

## Implication for Direction A (the compute decision) -- MAJOR
The premise that motivated the ~100hr "richer representation learning" -- that 28-word recognition is a hard
representation wall -- is substantially WRONG. With adequate training + a temporal-integration (averaged)
readout, 28-word recognition is ~0.89. So the cheap levers (more training events + temporal-integration
readout) likely carry the front-end much further than the documented single-shot "wall" suggested -- WITHOUT
the 100hr, WITHOUT BPTT, WITHOUT new representation learning.

## RESOLVED by the training-events trajectory (50/150/300/500 events, matched recipe, validated pipeline)
| events | clean 16-avg pool-argmax | single-shot k=1 | full-code best-NN | between-concept cos (OVERLAP) |
|-------:|-------------------------:|----------------:|------------------:|------------------------------:|
| 50  | 0.643 | 0.395 | 0.402 | 0.606 |
| 150 | 0.893 | 0.569 | 0.625 | 0.564 |
| 300 | 0.929 | 0.654 | 0.804 | 0.495 |
| 500 | 0.929 | 0.714 | 0.893 | 0.389 |

MULTI-SEED CONFIRMED (300 events, seeds 42/43/44): clean 28-word recognition 0.929/0.964/0.964 (mean ~0.95); between-cos ~0.50; single-shot 0.654/0.795/0.725; NN 0.804/0.884/0.893. The refutation is NOT seed-luck.

THREE decisive results:
1. **Clean 28-word recognition RISES to ~0.93 and plateaus** (300 = 500 = 0.929). NOT a representation wall;
   the documented ~0.57 "wall" and the retracted 0.64 "limit" were both UNDERTRAINING. ~0.93 is the practical
   28-word recognition ceiling with adequate training (>= 300 events).
2. **Concept OVERLAP DECREASES MONOTONICALLY with training** (between-cos 0.606 -> 0.564 -> 0.495 -> 0.389).
   This is the key result: more training makes the learned concept codes genuinely LESS overlapping. The
   internal map's lesson ("less-overlapping codes require it BY CONSTRUCTION during acquisition") is
   CONFIRMED -- and the simplest acquisition lever (just more training of the existing v16 architecture)
   achieves it. NO BPTT, NO new mechanism, NO 100hr needed -- the cheap lever IS more training.
3. **Single-shot and NN keep rising** (k=1 0.395 -> 0.714; NN 0.402 -> 0.893). At 500 events the full-code NN
   beats single-shot pool-argmax (0.893 vs 0.714) -- the lossy-readout escape RETURNS once the codes are
   well-trained. So a temporal-integration / NN readout adds on top of training.

## DECISIVE conclusion for Direction A (the compute decision)
The ~100hr "richer representation learning at scale" is NOT warranted for the 28-word front-end. The wall was
undertraining + single-shot noise. The CHEAP fix -- more training events (300-500, ~1-2 GPU-hr) of the
existing architecture + a temporal-integration / NN readout -- reaches ~0.93 clean recognition AND
genuinely reduces concept overlap (0.606 -> 0.389). The preparation has now REFUTED the premise that
motivated the big run.

The remaining real question is SCALE: does "more training reduces overlap" extend to 64/160/320-word vocab,
or does the overlap floor reappear at larger N? The G.20 sparse-distributed architecture already handles
64/bridge at 100% (engineered codes); the open piece is whether the LEARNED (v16) reps stay separable at
those sizes with adequate training. THAT is the worthwhile next characterization -- a training x vocab-size
sweep -- and it is far cheaper than the 100hr (and would tell us whether ANY big run is warranted, and at
what vocab size the cheap lever finally breaks).

## Discipline note
Gate 1's verdict was propagated as "validated" (it had a 16-word positive control). But the control validated
the PIPELINE, not the FAIRNESS of the cross-vocab comparison -- the training-amount confound slipped through.
Gate 2 caught it. The lesson: a positive control for the instrument is necessary but not sufficient; the
COMPARISON itself (matched training) must also be controlled. Both findings are kept; Gate 1's representation-
limit verdict is explicitly retracted here.
