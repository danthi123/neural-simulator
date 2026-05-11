# P5 iter E — CRITICAL FINDING: training itself didn't learn selective binding

**Date:** 2026-05-11
**Phase:** P5 of realigned plan v3
**Status:** CRITICAL diagnostic finding. Weight inspection shows
STDP did NOT learn a selective wernicke→semantic_cortex binding.
The training paradigm itself is the bottleneck, not the dynamics.

## Weight selectivity index: 0.004

After iter D's training (same params + weight inspection):

```
w_apple_wernicke -> apple_semantic  = 3.989
w_apple_wernicke -> river_semantic  = 4.054   <-- HIGHER than same!
w_river_wernicke -> river_semantic  = 4.083
w_river_wernicke -> apple_semantic  = 3.952

selectivity_index = (same_mean - cross_mean) / (same_mean + cross_mean)
                   = (4.036 - 4.003) / (4.036 + 4.003)
                   = 0.004  (~0% — essentially uniform)
```

All weights cluster around 4.0 (the recurrent_weight init for
iter D). STDP did NOT differentiate the bindings. Apple and river
both grew the SAME weight pattern.

This is **conclusive evidence** that the issue isn't:
- attractor dynamics (dynamics rework would still need a
  binding to work with — there's no binding to amplify)
- semantic_cortex size (no amount of capacity will help if
  training doesn't learn)
- FS lateral inhibition (won't add selectivity that isn't
  there in the weights)

## Why iter E (and previous) failed

Hypothesis: wernicke fires SIMILAR neurons for apple and river.

Evidence:
- wernicke is only 200 neurons
- lang_input → wernicke pathway density 0.30 means each wernicke
  neuron receives ~308 connections from 1024-neuron lang_input
- 100 active lang_input neurons (10% sparsity for apple OR river)
  drive essentially ALL wernicke neurons via the dense projection
- wernicke firing isn't selectively encoded per concept
- STDP grows wernicke→semantic_cortex weights uniformly because
  the SAME wernicke fires for both concepts

Real Wernicke's area has ~10⁵+ neurons with topographic sub-
specialization. Our 200-neuron toy is fundamentally undersized
for selective concept ensembles.

## What this means

The architecture needs **distinct wernicke (or upstream) ensembles
per concept** before STDP can learn anything useful downstream.

Three biology-grounded paths:

**Path G (next): enforce wernicke sparsity via FS inhibition WITHIN
wernicke**
- Add wernicke_fs region (PV-FS interneurons within wernicke)
- wernicke_E → wernicke_FS (excite all FS)
- wernicke_FS → wernicke_E (broad inhibit, NOT recurrent self)
- Limits total wernicke firing to ~5-10% of pool
- Different lang_input patterns will reach different sparse
  wernicke ensembles by chance + STDP refinement

**Path G+: multi-pool wernicke (explicit concept ensembles)**
- Mirror Tier 1 motor pool architecture for wernicke
- wernicke_pool_N (per concept), wernicke_fs_N
- Topographic bias from lang_input matches Tier 1 pattern
- Most reliable but requires per-concept allocation (limited
  scalability to large vocabs)

**Path H: scale wernicke AND enforce sparsity**
- wernicke 200 → 1000 (5x)
- Add wernicke_FS for sparsity enforcement
- Combination of structural scale + sparsity mechanism

## What iter F (currently running) will show

Iter F has --enable-semantic-fs (FS in semantic_cortex). It will
test whether FS inhibition between semantic_cortex neurons helps
discrimination. Expected: still FAIL, because wernicke is still
the bottleneck (sends same input to semantic_cortex regardless
of which concept).

But iter F is still informative — confirms that semantic_cortex
FS isn't sufficient, focusing the next experiment on wernicke.

## Updated diagnostic

| Iter | Hypothesis | Result | New Info |
|---|---|---|---|
| Original | default | FAIL | baseline |
| A | engram-tag methodology | FAIL | methodology improved |
| B | strict two-stage gating | FAIL | gating not the issue |
| C | scale wernicke+sem 2x | WORSE | size not the issue |
| D | attractor tuning | FAIL (monolithic) | attractor forms but uniform |
| **E** | **weight inspection** | **selectivity 0.004** | **TRAINING DIDN'T LEARN** |
| F (running) | + FS in semantic_cortex | (expected FAIL) | confirms wernicke is upstream bottleneck |
| **G (next)** | **FS in wernicke** | TBD | this is where the fix is |

## Iron law update

Per superpowers:systematic-debugging Phase 4.5 (3+ fails =
architecture). We're at 5 fails, but iter E gave us NEW
information — the upstream bottleneck is wernicke, not dynamics
or size. The architectural redesign target is clear.

This is the kind of breakthrough finding that justifies
continuing to iterate. Iter E weight diagnostic was the right
next move — it told us where to look.

## Wall clock so far

~2.5 hours of autonomous P5 work:
- 5 iterations × ~5 min = 25 min compute
- Liu 2012 × 3 seeds × 2 min = 6 min compute
- iter F currently running (~10 min)
- Total compute so far: ~31 min
- Documentation + diagnostic code: rest

Per autonomous-runs: hardware-bound estimates are reliable.
Total wall clock fine.

## Path forward (decision)

1. Let iter F (Path B+ FS in semantic_cortex) complete.
2. Implement Path G (FS in wernicke) — adds 1 region + 2
   pathways to the builder. ~30 min of code.
3. Launch iter G with same iter D-style params + wernicke FS.
4. If iter G shows weight selectivity > 0.1: dynamics tuning
   then makes sense (iter D + iter G combined).
5. If iter G still ~0: multi-pool wernicke (Path G+) needed.
