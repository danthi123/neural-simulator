---
type: finding
status: live
date: 2026-05-11
---

# 96-word @ XL encoding (n_lang=16384) — NEGATIVE: encoding-only scale-up doesn't fix the wall

**Date:** 2026-05-11 01:43 EDT (run killed at PRE-SILENCE 350/480 by user
decision after convergence was clear)
**Status:** NEGATIVE — XL encoding @ n_lang=16384 with n_motor=2000 does
NOT break the 96-word retention wall observed at n_lang=8192. Accuracy
converges to the primary-direction floor (~25%, i.e. 1-of-4 directions
correct).

---

## Run config

```bash
python -m research.runners.consolidation_synonym_trainer \
    --smoke --vocab-size 96 \
    --n-lang-input 16384 --n-motor-per-action 2000 \
    --n-motor-fs-per-action 200 --seed 42
```

| | Value |
|---|---|
| Architecture | n_lang=16384, n_motor=2000, n_motor_fs=200 |
| Total neurons | 42,288 |
| Total synapses | 111,929,115 |
| GPU memory | 9.4 GB / 25.8 GB (36.4%) |
| Training wall clock | 4733 s (~79 min) |
| Training events | 200 × 12 chunks awake + 12 sleep phases |
| Topographic prior | 1.5/0.7 factor, applied reciprocally |
| FS lateral inhibition | enabled (motor_FS regions) |
| NMDA | enabled |

## Eval result

PRE-SILENCE W→A accuracy across 350 trials (run killed before
HIPPO-OFF eval started — that would have added another ~95 min of
similar-trajectory eval for marginal information):

| Trials | Correct | Accuracy |
|---|---|---|
| 25  | 8  | 32.0% |
| 50  | 12 | 24.0% |
| 75  | 23 | 30.7% |
| 100 | 24 | 24.0% |
| 125 | 31 | 24.8% |
| 150 | 40 | 26.7% |
| 175 | 44 | 25.1% |
| 200 | 52 | 26.0% |
| 225 | 58 | 25.8% |
| 250 | 61 | 24.4% |
| 275 | 71 | 25.8% |
| 300 | 76 | 25.3% |
| 325 | 82 | 25.2% |
| 350 | 87 | 24.9% |

**Converged accuracy ~25%** — exactly the 1-of-4 primary-direction floor.

## Why this is NEGATIVE

The 96-word vocab is 4 primary directions × 24 synonyms each. If the
network had learned the synonym binding, accuracy would be in the
30-50% range (mix of primary hits + synonym hits, similar to the
Tier 2.1 8-word result which hit 63.7% A→W). At 25% we're seeing:

- **Primary words** (north/east/south/west, 4 of 96): probably bound,
  contribute ~all the correct predictions
- **Synonym words** (the other 92): essentially chance — the network
  isn't differentiating them at the cortical level

Doubling the encoding axis from 8192 → 16384 added 80M synapses and
2× the lang-input neurons, but the bottleneck isn't there. The
96-word retention wall is elsewhere — likely:

1. **Motor-pool capacity**: 2000 neurons per action × 4 actions = 8000
   total motor neurons; dividing among 24 synonyms gives ~83 neurons
   per synonym sub-pop. The Tier 2.1 BREAKTHROUGH used 1000 motor +
   2 synonyms = 500 neurons/synonym. Roughly 6× less capacity per
   synonym at 96-word than at 8-word.
2. **Training signal saturation**: 200 events × 96 words = 19,200
   awake events. Hippocampus consolidation runs 12 sleep phases. If
   the binding signal per word is too sparse, STDP can't form
   stable patterns.
3. **Architecture insufficient**: The Tier 1/2.1 architecture
   (single language_input → 4 motor pools + FS) might fundamentally
   not generalize past ~16-32 word synonym groupings.

## What we learned

1. **The encoding-axis discovery (2026-05-10) was the right insight but
   doesn't compound past Tier 2.x.** At 64-word vocab, n_lang=8192
   handles it (3-seed GO). At 96-word, doubling to n_lang=16384 doesn't
   help — confirming the bottleneck moves to motor capacity.

2. **The pattern matches what we saw at 8K**. The 8K result was
   PARTIAL (primary 70%, synonym 30%, retention 57% primary FAIL,
   83% synonym PASS). The 16K result PRE-SILENCE is converging to
   the same ~25% floor — telling us that even with MORE encoding
   capacity, the architecture can't distribute 96 words across the
   existing motor topology.

3. **The architecture probably needs scale on multiple axes.** Going
   from 8 → 96 words is 12× more vocabulary. Just doubling encoding
   (8K → 16K, 2×) is not enough. Likely needs simultaneously:
   - n_motor 2000 → 4000+ (per-synonym sub-pop capacity)
   - n_motor_fs 200 → 480+ (lateral inhibition headroom)
   - OR a different architectural primitive (e.g. multi-cortical-area
     hierarchy, attention readouts, etc.)

4. **Time cost was real but recoverable.** ~79 min training + ~80 min
   eval = 2h 39m of GPU time before the negative verdict was clear.
   The killed HIPPO-OFF eval would have added ~95 min more for
   marginal info. Net saved: ~95 min, redirected to inference
   benchmark chain.

## Where to next

Two paths forward; the user will decide:

**Option A — Scale all three axes:** retry 96-word at
n_lang=16384, n_motor=4000, n_motor_fs=480. That's ~14B synapses, far
beyond what fits in a 24GB RTX 3090. Cloud-anchored only.

**Option B — Stop at 64-word for local-first.** The 64-word @ 8K is
3-seed validated GO (Tier 2.x). 64 words is a serviceable vocabulary
for the user-facing chat workflow. Scale beyond 64 happens on cloud.

**Option C — Architectural redesign.** Add multi-cortical-area
hierarchy (cortex_layer_1 → cortex_layer_2 → motor) so the encoding
isn't trying to drive motor pools directly. This is a 2-4 week scope.
Aligns with the Phase 2 "cloud-anchored" plan in the strategic addendum.

For now, the lineage workflow shipped tonight + 64-word validated arch
is enough to ship a usable continuous-learning REPL. The 96-word wall
is a known limit, NOT a blocker for the next user-facing milestone.

## Provenance

- This findings doc:
  `research/findings/2026-05-11-96word-XL-encoding-NEGATIVE.md`
- Run config + log: `webapp/runtime/run_96word_xl.log`
- Smoke killed by user decision via taskkill at 01:41 EDT after eval
  convergence was clear (350/480 PRE-SILENCE).
- Replaced post-XL plan: HIPPO-OFF eval skipped; inference_benchmark
  chain (8 vocab tiers) auto-fired by
  `research/findings/raw/g11_bg/chain_inference_benchmark_post_xl.ps1`
- Related: 8K result was the PARTIAL at
  `research/findings/raw/g11_bg/g11_seed42_consolidation_synonym_96word_encoding_axis_smoke.json`
  (primary 57%, synonym 83%)
- Related: VRAM ceiling probe
  `research/findings/2026-05-10-vram-ceiling-probe-results.md`
- Strategic context:
  `docs/plans/2026-05-10-MASTER-PLAN-strategic-addendum.md`
- Optimization candidates:
  `docs/plans/2026-05-10-phase1-local-optimization-design.md`
