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

## Open question (the confirming sweep, launched)
Does 28-word separability keep rising toward the 16-word 1.000 with MORE training (300, 500 events), or
plateau at ~0.89? And does it hold at the larger G.20 vocab tiers? A training-events trajectory (300, 500
events, matched recipe) is running. If it rises toward ~0.95+, the front-end is essentially a training+readout
problem (cheap); if it plateaus at ~0.89, that is the modest 28-word ceiling (still far better than the
retracted 0.64). Either way the 100hr representation-learning premise is not supported at 28 words.

## Discipline note
Gate 1's verdict was propagated as "validated" (it had a 16-word positive control). But the control validated
the PIPELINE, not the FAIRNESS of the cross-vocab comparison -- the training-amount confound slipped through.
Gate 2 caught it. The lesson: a positive control for the instrument is necessary but not sufficient; the
COMPARISON itself (matched training) must also be controlled. Both findings are kept; Gate 1's representation-
limit verdict is explicitly retracted here.
