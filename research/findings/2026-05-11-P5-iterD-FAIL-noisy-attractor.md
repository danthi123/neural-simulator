# P5 iter D seed 42 FAIL — attractor IS forming but no discrimination

**Date:** 2026-05-11
**Phase:** P5 of realigned plan v3
**Status:** QUALITATIVELY DIFFERENT failure mode. Iter D made
semantic_cortex form an attractor — but a MONOLITHIC one
(same cortex pattern fires for any input). Discrimination
margin SHRUNK.

## Comparison (seed 42)

| Metric | Iter A | Iter B | Iter C | **Iter D (attractor)** | Target |
|---|---|---|---|---|---|
| apple_self cosine | 0.227 | 0.226 | 0.207 | **0.351** ↑ | > 0.5 |
| apple_river cosine | 0.174 | 0.186 | 0.198 | **0.342** ↑↑ | < 0.4 |
| Margin (self - cross) | 0.053 | 0.040 | 0.009 | **0.009** | high |
| Naming ratio | 1.08x | 1.08x | 0.99x | **1.11x** | > 1.3x |
| Wall clock | 295s | 306s | 339s | **303s** | — |
| Verdict | FAIL | FAIL | FAIL | **FAIL** | — |

Apple_self DID grow significantly (0.227 → 0.351). But
apple_river grew EVEN MORE (0.174 → 0.342). The attractor IS
forming — but it's MONOLITHIC. Same cortex sub-population fires
in response to ANY input.

## Diagnosis: monolithic attractor, not competitive

Iter D parameters:
- recurrent_density 0.10 → 0.25
- recurrent_weight 1.0 → 4.0 (matches feedforward)
- drive_steps 100 → 300 (settle time)

These create strong attractor dynamics — but the architecture
has UNIFORM random recurrent connectivity with no
scaffolding for separate attractor basins per concept.

Classical attractor networks need either:
1. **Pre-stored orthogonal patterns** (Hopfield-style)
2. **Lateral inhibition between sub-populations** (real cortex
   PV-FS interneurons; the Pulvermüller 2001-2003 distributed
   grounding theory)
3. **Hebbian plasticity that carves distinct basins from
   training** (Marr-style)

Our semantic_cortex has NONE of these. STDP on uniformly random
recurrent connections at high density just creates one giant
basin where all neurons co-fire.

This is biology's **competitive attractor** principle: you need
mechanisms that ENFORCE separation between concept ensembles,
not just hope they emerge.

## What iter E will tell us

Iter E launched: same params as iter D but with the weight
inspection diagnostic shipped on bac3f26. It will compute:

```
selectivity = (mean_weight[apple_wernicke->apple_sem] -
                mean_weight[apple_wernicke->river_sem]) /
               (same + cross)
```

- selectivity > 0.1: STDP learned the binding but dynamics
  are too noisy. → fix is dynamics (lateral inhibition).
- selectivity ~0: training itself failed to learn the binding.
  → fix is training paradigm (contrastive multi-concept).

## Two next-step paths

**Path B: Multi-pool semantic with FS lateral inhibition.**
Mirror Tier 1 architecture (which passed multi-seed 5/6) at
the SEMANTIC level:
- semantic_pool_apple (500 neurons)
- semantic_pool_river (500 neurons)
- semantic_FS_apple (60 PV-FS), semantic_FS_river (60)
- semantic_pool_X → semantic_FS_Y for X != Y (cross-inhibition)
- Topographic prior matching lang(concept) → semantic_pool[concept]
- Estimated 1-2 hours implementation

**Path B+: Stay with single semantic_cortex but add internal
FS lateral inhibition.** Keep ATL hub theory geometry (one
cortex pool) but enforce sub-population separation via:
- Add semantic_FS_inhibition region (100 PV-FS neurons)
- semantic_cortex (excit) → semantic_FS (excite)
- semantic_FS → semantic_cortex (inhibitory, broad)
- This implements winner-take-most among co-active sub-populations
- Smaller change than Path B but tests same hypothesis

## What iter D tells us about the architecture

Positive: attractor formation IS achievable with the right
parameters. Recurrent_weight 4.0 + density 0.25 produces
stable point attractors.

Negative: without lateral inhibition or other separation
mechanisms, all input converges to the same attractor. The
architecture needs scaffolding for SELECTIVE attractors.

This is exactly what Pulvermüller distributed grounding
(2001-2003) and Patterson hub-and-spoke (2007) require but
don't fully specify. The catalog's G.13 prerequisite of
"semantic memory store" is abstract — competitive attractor
substrate is implicit.

## Iron law: 5 fails total now

Per superpowers:systematic-debugging Phase 4.5: 5 attempts
without PASS. The architecture needs lateral inhibition or
multi-pool scaffolding — both of which are SUBSTANTIAL code
changes, not parameter tweaks.

Iter E (running) will resolve "training learned" vs "training
didn't learn" — that determines which architectural change
is needed.
