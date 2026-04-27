# Cortex WTA + Adaptive DA + Hippocampus — PARTIAL: combo helps but still misses baseline

**Date:** 2026-04-26 (combo follow-up to cortex-wta PARTIAL result)
**Status:** **PARTIAL** — adaDA delivers ~14% improvement over WTA+hippo, but combo still 1.36× worse than baseline. Confirms plastic-input-layer ceiling isn't crackable by parameter combinations alone.
**Companion:** [Cortex WTA partial](2026-04-26-cortex-wta.md), [Asymmetric adaptive DA](2026-04-26-asymmetric-adaptive-da.md), [6-seed correction](2026-04-26-six-seed-correction.md)

## TL;DR

After cortex-WTA showed a partial fix (16% improvement over hippo-alone
but still 58% worse than baseline), the next logical test was: combine
WTA (which fixes selectivity) with adaptive DA (which earlier improved
readaptation by gating eligibility on reward EMA). Hypothesis: WTA
handles the commitment problem within a phase, adaDA releases commitment
when reward drops at goal flip.

**Result: partial confirmation.** The combo does improve over WTA+hippo
alone, but only by ~14% and still doesn't reach baseline.

| Variant | Sum (3-seed avg) | vs baseline |
|---|---:|---|
| Baseline | **5.88** | reference |
| WTA only (1 seed) | 6.44 | -10% |
| WTA + adaDA only (1 seed) | 6.88 | -17% |
| WTA + hippo (3-seed) | 9.26 | -58% |
| **WTA + hippo + adaDA** | **8.01** | **-36%** |
| Hippo additive | 10.98 | -87% |

## Per-seed details

```
Combo A (WTA + hippo + adaDA, asym ema_neg=0.7):
seed 42: P0 finalQ=3.31  P1 finalQ=3.47  sum=6.78  n_at_goal P1=46
seed 43: P0 finalQ=2.21  P1 finalQ=6.67  sum=8.88  n_at_goal P1=29  (worst)
seed 44: P0 finalQ=3.67  P1 finalQ=4.69  sum=8.35  n_at_goal P1=34
avg: 8.01

Control (WTA + adaDA, no hippo, seed 42):
seed 42: P0 finalQ=3.39  P1 finalQ=3.49  sum=6.88  n_at_goal P1=43
```

Per-seed variance is very high: seed 42 is close to baseline (6.78), but
seeds 43-44 are 8.4-8.9. The combo doesn't reliably converge.

## Key insights

### 1. AdaDA provides ~14% improvement when stacked with WTA+hippo

The combo (8.01) beats WTA+hippo alone (9.26) by ~14%. AdaDA's
reward-EMA gating does relax commitment when reward drops, helping the
agent break out of phase-0 lock-in. But not enough.

### 2. AdaDA doesn't help WTA's intrinsic readaptation penalty

Looking at WTA-only (6.44) vs WTA+adaDA (6.88): adaDA actually slightly
*hurts* on the seed-42 control without hippo. The combo's value comes
entirely from compensating for hippo's cold-start, not from fixing WTA's
commitment-vs-readaptation pattern.

This is consistent with the 6-seed correction earlier today: adaDA's
benefit was overstated and seed-dependent in the first place.

### 3. P1 action distributions still ~uniform

All 3 seeds: P1 actions clustered in ~360-390 across all 4 directions
(seed 42: [384,380,349,387]). The cascade is firing more than under
hippo-alone (where it was silenced) but not consistently selecting the
right action.

### 4. Plastic-input-layer ceiling is structural, not parametric

This now closes off a substantial slice of the parameter space:

  Baseline (1-input)               5.88   (clean)
  Hippo additive (cold-start)      10.98  (cascade silenced)
  + WTA                            9.26   (selectivity fixed; readaptation broken)
  + WTA + adaDA                    8.01   (readaptation partially helped)
  +-+ adaDA only? not tested with hippo, but expected worse than WTA+adaDA+hippo

Each addition to the parameter stack closes a sub-problem (selectivity,
then commitment) but exposes the next bottleneck. The progression is
asymptoting toward "approximately baseline" without crossing it.

The architecture genuinely prefers a single clean cortex input source.

## Decision

- Keep combo flags opt-in. Don't recommend as default.
- **Combo A is the last reasonable runner-side flag combination test.**
  Two more architectural levers (curriculum, pivot) remain.
- The structural hypothesis to test next: **curriculum learning** (Option
  B). Lock cortex→D1 plasticity during a fixed-goal phase. Once cortex
  selectivity is established, thaw the input layer. Hippocampus then
  learns place→action *given that the cascade is already working*. This
  is structurally different from the cold-start problem.

## Files

- `research/findings/raw/g11_bg/g11_seed{42,43,44}_cortexwta_hippo_adaDA.json`: 3-seed acid test
- `research/findings/raw/g11_bg/g11_seed42_cortexwta_adaDA_only.json`: WTA+adaDA control

## Lesson

Five plastic-input-layer attempts have now converged on the same finding:
**you cannot solve cold-start by adding more flags.** The cleanest fix
identified so far (cortex WTA + adaptive DA) gets within 36% of baseline,
which is a genuine improvement over the worst variant (hippo additive
at 87% worse), but a substantial regression from baseline (which has the
heuristic doing the perception heavy-lifting).

Curriculum learning is structurally different: it sequences plasticity
in time rather than gating it spatially. If it works, it would unblock
the entire plastic-input-layer arc cleanly. If it also fails, that's
strong evidence the architecture's "single clean input" preference is
structural and we should pivot rather than continue.
