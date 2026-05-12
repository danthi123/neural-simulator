# P5 strategic options summary (post iter LL/KK/MM)

**Date:** 2026-05-12
**Audience:** Human researcher returning to autonomous arc decision point

## What the sim ALREADY has (multi-seed validated)

| Capability | Multi-seed result | Notes |
|---|---|---|
| Tier 1 motor binding (4-word: N/E/S/W) | 6/6 BIDIR, 86%/98% acc | Production-ready |
| Tier 2.1 synonym binding (8-word: + up/right/down/left) | 6/6 BIDIR, 60%/95% acc | Scaled-arch breakthrough |
| Phase 1.3 hippocampus consolidation | 3/3 GO, 91.2% retention | CLS theory validated |
| Phase 1.4 catastrophic forgetting eval (Branch A) | 5/6 GO at 103% retention | Continual learning works |
| Phase 2.1 ABC surrogate-grad BPTT | 100% loss reduction | Pretraining stack works |
| Phase 2.2 Tiny Shakespeare BPTT | 84% loss reduction | GPU pretraining works |

**The sim has working bidirectional conversational capability for ~8
direction-related words across 6 seeds.** This is real and usable.

## What P5 (this arc) was trying to achieve

Extend bidirectional concept binding from MOTOR-grounded words
(direction words have explicit somatotopy) to NON-MOTOR abstract
concepts (apple, river — no motor referent).

Catalog target: G.11 (Hickok & Poeppel dual-stream language) + G.13
(Wernicke's area, semantic comprehension).

## What we've learned about P5 (~38 iterations, 24+ hours autonomous)

### iter AA breakthrough (2026-05-11)
- Per-concept dedicated wernicke pools + per-concept lang_output pools
- 4/6 BIDIRECTIONAL multi-seed PASS at toy scale (5K neurons)
- Apple 6/6, river 4/6 (seeds 44/101 fail river-direction)

### iter KK NEGATIVE (2026-05-12 — this autonomous arc)
- Tier 1 cortical canon (internal_density=0.10, exc=2.0, inh=4.0)
  at biological scale (8.6K neurons, 500-neuron pools)
- FAILED: pool firing exploded (200+ vs iter AA 80-100) but
  discrimination collapsed (both pools fire similarly)
- Cortical canon AMPLIFIES seed structural bias instead of averaging

### iter LL NEGATIVE (2026-05-12)
- iter AA recipe (weak dynamics) at biological scale
- FAILED: apple discrimination ratio drops from 1.08 to 0.978
- Biological scale makes pools fire MORE symmetric, not LESS

### iter MM NEGATIVE / pending (2026-05-12)
- Stronger topographic bias (3.0/0.33 vs 1.5/0.7)
- Status: TBD (running)
- Hypothesis: stronger lang_input → wernicke_pool prior
  restores discrimination at biological scale

### Critical diagnostic finding
- iter AA weight selectivity_index across 6 seeds: -0.005 to +0.007
- Target for "clear binding" is >0.1
- iter AA passes BIDIR despite essentially ZERO learned weight asymmetry
- **Discrimination comes from TOPOGRAPHIC BIAS PRIOR, not STDP learning**

This is profound: the architecture's discrimination is FIXED by the
initial prior. Training doesn't add concept-specific selectivity.
Scaling without strengthening the prior LOSES the discrimination.

## Path forward — strategic options

### Option 1: Stop here, ship iter AA (4/6 BIDIR)
- iter AA is genuinely useful (4/6 multi-seed for 2 concepts)
- Document architectural ceiling honestly
- Move to scaling motor-binding (Tier 2.2 with more synonyms)
  or to P6 (Broca's compositional) building on current results
- **Effort:** Hours

### Option 2: Sensory grounding via Cluster K v2 visual cortex
- Real concepts are MULTIMODAL (apple has visual features)
- Integrate visual_cortex regions into text_minimal_isolation builder
- Train: lang_input("apple") + retina(red-round-image) paired
- Test: cortex_IT (300-neuron IT region with canon dynamics)
  acts as concept hub binding lang_input + visual
- **Biology-faithful:** Pulvermüller embodied semantics, Lambon
  Ralph hub-and-spoke model
- **Effort:** 1-2 weeks of focused work

### Option 3: Unified Wernicke + sparse coding
- Drop per-concept pools (the architectural cheat acknowledged in
  P5 Path G+ design doc)
- Single Wernicke region with strong FS lateral inhibition
- Different concepts naturally select different sparse subsets
- Test if sparse-coding discrimination beats per-pool ceiling
- **Biology-faithful:** Matches catalog G.13 (single Wernicke area)
- **Effort:** 1 week

### Option 4: Drop semantic_cortex from naming path
- Shorter chain: lang_input → wernicke_pool_i → lang_output_pool_i
- Skip semantic_cortex bottleneck entirely for recognition
- Test if simpler chain preserves discrimination at biological scale
- **Effort:** 1-2 days

### Option 5: Different concepts with truly orthogonal codes
- iter AA codes have 8.8% overlap between apple/river
- Use orthogonal_drive_pattern (zero overlap)
- Test if clean input separation rescues iter LL-style scaling
- **Effort:** Hours (code change + one smoke)

## Recommendation

The user's stated goal is "conversational sim for non-motor concepts."
P5's per-concept pool architecture has an architectural ceiling at
toy scale (iter AA) that doesn't yield to biological scale (iter LL/KK).

**Recommended path:** 
1. (If iter MM PASS) — proceed with iter MM 6-seed validation
2. (If iter MM FAIL) — Option 2 (sensory grounding via Cluster K v2)
   is most biology-faithful and matches the user's "no cheats"
   directive. Provides multimodal semantic content rather than
   arbitrary sparse codes.

Option 5 (orthogonal codes) is a cheap parallel test worth running
regardless — if it works, it eliminates the per-seed bias issue
without architectural change.

## Honest assessment of progress

The architectural ceiling at iter AA (4/6 BIDIR) is REAL and represents
a genuine validated capability. The remaining 2/6 failures are
structural per-seed asymmetries that no parameter tuning has solved
in 38 iterations. The next move requires changing the architecture
(sensory grounding, unified Wernicke, or simplified chain) rather
than more parameter exploration.

The sim's conversational capability remains rock-solid at the
**8-word direction vocabulary** (Tier 1/2.1 validated). Extending
to **non-motor abstract concepts** requires architectural rework.
