# P5 iter F seed 42 FAIL — semantic_FS doesn't help (confirms iter E)

**Date:** 2026-05-11
**Phase:** P5 of realigned plan v3
**Status:** Confirms iter E diagnosis. semantic_cortex FS lateral
inhibition (Path B+) does NOT add selectivity. The bottleneck is
upstream in wernicke, not downstream in semantic_cortex.

## Result (seed 42)

| Metric | Iter D (no FS) | Iter F (semantic_FS) | Target |
|---|---|---|---|
| apple_self cosine | 0.351 | **0.333** ↓ | > 0.5 |
| apple_river cosine | 0.342 | **0.325** ↓ | < 0.4 |
| Margin (self - cross) | 0.009 | **0.008** | high |
| Naming ratio | 1.11x | **1.06x** | > 1.3x |
| **Weight selectivity** | 0.004 | **0.0007** | > 0.1 |
| Verdict | FAIL | **FAIL** | — |

semantic_FS slightly dampened the monolithic attractor (cosines
down from 0.35 to 0.33) but did NOT add selectivity. Weight
selectivity dropped further (0.004 → 0.0007).

This is exactly the predicted outcome per iter E: with wernicke
firing uniformly for both concepts, downstream FS can't
discriminate signals that aren't differentiated upstream.

## Why iter F failed as predicted

Per iter E: weight inspection showed all wernicke→semantic
weights ~4.0 regardless of concept (selectivity_index=0.004).
This means wernicke fires the SAME neurons for both apple AND
river inputs. STDP grows weights to whichever semantic_cortex
neurons happen to fire, but those firing neurons are also the
same for both concepts.

Adding FS lateral inhibition in semantic_cortex:
- Could enforce sparsity in semantic_cortex firing (top-k WTA)
- But ALL inputs reach the SAME monolithic attractor basin
- FS just dampens that one basin — no new basins emerge

For selectivity to emerge, wernicke needs DISTINCT ensembles
per concept. That's what iter G (wernicke_FS) tests.

## Iter G launched in parallel

Iter G launched at ~8:04 (just after iter F at ~7:59). Same
parameters as iter F but with `--enable-wernicke-fs` instead.
Expected completion ~8:14-8:18.

If iter G shows selectivity > 0.1, that's a real win — the
training paradigm CAN learn selective bindings given proper
substrate. Downstream dynamics work would then complete the
picture.

If iter G also shows selectivity ~0, the training paradigm
itself (not just substrate) needs work — likely contrastive
training or pre-allocated multi-pool wernicke (Path G+).

## What we've now shown

| Hypothesis | Test | Result |
|---|---|---|
| Methodology too noisy | iter A (engram-tag) | improved direction but FAIL |
| Mixed gate timing | iter B (strict two-stage) | no movement |
| Wernicke too small | iter C (scale 2x) | WORSE |
| No attractor dynamics | iter D (recurrent tuning) | monolithic attractor forms |
| Training didn't learn | iter E (weight inspection) | confirmed (selectivity=0.004) |
| Downstream FS fixes it | **iter F (semantic_FS)** | **NO (selectivity=0.0007)** |
| Upstream FS in wernicke | iter G (running) | TBD |

The diagnostic narrative is clean. Each iteration tested a
specific hypothesis. We've ruled out: methodology, gating, size,
downstream attractor tuning, downstream FS. The remaining
hypothesis is upstream sparsity (iter G).

## Wall clock so far

~3 hours of autonomous P5 work:
- 7 iterations × ~5-10 min = ~45 min compute
- iter G still running (~10 min)
- Liu 2012 × 3 seeds = 6 min compute
- Total compute: ~55 min
- Rest: code + docs + diagnostics
