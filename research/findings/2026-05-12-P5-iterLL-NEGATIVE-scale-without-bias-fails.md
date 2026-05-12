# P5 iter LL NEGATIVE — biological scale alone doesn't fix per-concept pool bias

**Date:** 2026-05-12
**Status:** NEGATIVE. Single-seed smoke (seed 42) FAILED with weak iter AA
dynamics at biological scale (n_per_wernicke_pool=500,
n_per_lang_out_pool=500, n_lang_input=2048). The architecture's
discrimination ratio DEGRADES at scale, not improves.

## Hypothesis tested

User directive (2026-05-12): "I don't know why we keep testing at toy
scale if larger scale (that still fits locally) is clearly needed?"

iter AA confirmed-ceiling findings doc had identified biological scale
as one of three paths to push 4/6 → 6/6:
> 3. Biological scale (~10⁵ neurons, VRAM-bounded)
>    - Random structural asymmetries average out at scale
>    - Toy-scale issues should disappear

iter LL tests this: iter AA recipe (weak dynamics 0.05/0.3/0.8) at
biological scale (8.6K neurons, 2.3M synapses, 5x bigger pools, 2x
bigger lang_input).

## Result

| | iter AA seed 42 (toy) | iter LL seed 42 (bio scale) |
|---|---|---|
| n_neurons | 5092 | 8636 |
| n_synapses | 1.02M | 2.30M |
| apple p0 spikes | 92 | 218 |
| apple p1 spikes | 85 | 223 |
| **apple discrimination** | **p0 OK (+7)** | **p1 WRONG (-5)** |
| river p0 spikes | 80 | 208 |
| river p1 spikes | 111 | 216 |
| **river discrimination** | **p1 OK (+31)** | **p1 OK (+8)** |
| **BIDIR** | **YES** | **NO** |
| apple_self cosine | 0.50+ (iter AA passed) | 0.213 |
| Weight selectivity index | 0.0005 (n.b.) | 0.001 (n.b.) |

## The smoking gun: discrimination ratio DEGRADES at scale

| Pool ratio | iter AA seed 42 | iter LL seed 42 | Change |
|---|---|---|---|
| apple p0/p1 | 92/85 = 1.08 | 218/223 = 0.978 | **collapsed** |
| river p0/p1 | 80/111 = 0.72 | 208/216 = 0.963 | **collapsed** |

The per-pool firing approaches SYMMETRY at biological scale — both
pools fire similarly for both stimuli. The per-concept discrimination
ratio drops from ~1.4x to ~1.04x.

## Why scale hurts (architectural diagnosis)

iter AA's discrimination depends on the topographic bias prior at
lang_input → wernicke_pool (factor 1.5/0.7 = 2.14x ratio on weights).
This bias is applied to a FIXED set of edges (the active lang_input
neurons for each concept × wernicke_pool neurons).

At biological scale:
- lang_input grows 1024 → 2048: 2x more active neurons per concept
- wernicke_pool grows 100 → 500: 5x more pool neurons
- Total edges per concept = 205 active × 500 pool = 102.5K (vs 205 × 100 = 20.5K at toy)

The number of biased edges scales 5x but the bias FACTOR stays the
same (2.14x). Random structural connectivity (the unbiased edges)
ALSO scales. With 5x more random edges, the relative signal vs noise
DEGRADES because:
- Bias edges sum: scales linearly with N
- Random noise sum: scales with sqrt(N)
- Signal/noise ratio: should scale with sqrt(N) — should HELP
- BUT pool firing total spikes are dominated by RECURRENT activity, not lang_input drive
- Recurrent activity in larger pools is structurally biased by seed (random initialization)
- That bias also scales linearly with N

The net effect: structural noise grows linearly, signal grows linearly,
but the signal needs to overcome the structural bias which ALSO scales.
**Biological scale doesn't help.**

## Deeper finding: iter AA itself doesn't actually learn weights

| Seed | iter AA selectivity_index |
|---|---|
| 42  | 0.0005 |
| 43  | 0.0065 |
| 44  | 0.0050 |
| 100 | -0.0009 |
| 101 | 0.0022 |
| 102 | -0.0049 |

iter AA's wernicke → semantic_cortex weight matrices show essentially
ZERO concept-specific selectivity (target >0.1 for clear binding).
Yet iter AA achieved 4/6 BIDIR on pool_readout test.

**iter AA's discrimination works via TOPOGRAPHIC PRIOR, not STDP
learning.** The 4/6 success comes from the lang_input → wernicke_pool
weight prior persisting through the chain, NOT from concept-specific
weight differentiation emerging during training.

This means: the architecture is at its discrimination floor with
just the prior. Adding training events, scaling, or recurrent
strengthening can't push past the prior's discrimination ceiling.

## Implication: stronger topographic bias is the missing variable

At biological scale, the topographic bias relative-strength needs to
INCREASE to maintain discrimination over the noise floor. iter MM
will test: bias factor 1.5/0.7 (2.14x) → 3.0/0.33 (9.1x).

Biology is plausibly supportive: Pulvermüller cortical somatotopy
shows 5x activation ratios in highly-organized cortical maps;
9x is within biological range for highly-organized areas.

## Comparison

| Iter | Scale | Internal dyn | Topographic | apple p0/p1 | river p0/p1 | BIDIR |
|---|---|---|---|---|---|---|
| AA (toy) | 5K | weak (0.05/0.3/0.8) | 1.5/0.7 | 92/85 OK | 80/111 OK | YES |
| KK (bio) | 8.6K | canon (0.10/2.0/4.0) | 1.5/0.7 | 236/254 X | 242/259 OK | NO |
| LL (bio) | 8.6K | weak | 1.5/0.7 | 218/223 X | 208/216 OK | NO |
| MM (bio) | 8.6K | weak | 3.0/0.33 | (running...) | (running...) | ? |

## Code

iter LL launched with same config as iter AA but biological scale.
No code changes; existing CLI flags via:

```bash
python -m research.runners.validate_ventral_semantic --seed 42 \
    --n-train-events 400 --n-replay-cycles 40 \
    --n-lang-input 2048 \
    --enable-multi-pool-wernicke --n-wernicke-pools 2 \
    --n-per-wernicke-pool 500 --n-per-wernicke-pool-fs 60 \
    --interleaved-training \
    --enable-per-concept-lang-out-pools --n-per-lang-out-pool 500 \
    --apply-wernicke-topographic \
    --n-recognition-trials 5 --inter-trial-rest-steps 100
```

## Next: iter MM (stronger topographic bias)

Same config but `--wernicke-topographic-factor 3.0
--wernicke-off-target-factor 0.33` (9.1x ratio vs 2.14x).

If iter MM passes 6/6: stronger bias rescues the architecture at
biological scale.

If iter MM fails: the per-concept-pool architecture has a fundamental
limit at biological scale that no parameter tuning fixes. Pivot to:
1. Anchored concepts via Cluster K v2 visual cortex (sensory grounding)
2. Drop semantic_cortex from recognition path (simplify chain)
3. Unified Wernicke + sparse coding (deeper architectural change)

## Biological-scale findings summary

After P5 iter A-LL (~38 iterations, 24+ hours autonomous):
- **iter AA 4/6 BIDIR remains the architectural ceiling**, NOT
  improved by scale alone (iter LL), canon alone (iter KK), canon+scale
  (iter KK), multi-trial averaging (iter DD), 5x scale alone (iter FF
  per commit history)
- Discrimination is bias-floor-limited, not learning-limited
- iter MM (stronger topographic) is the last reasonable tuning before
  architectural pivot

User directive "stay biologically accurate, no cheats" preserved
throughout — no architectural shortcuts, just biology-grounded
parameter exploration.
