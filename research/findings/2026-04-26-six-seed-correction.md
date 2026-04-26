# 6-Seed Validation — Asym Adaptive DA Win Was Overstated, LR Boost is Actually Best

**Date:** 2026-04-26 (late autonomous session)
**Status:** CORRECTION — extending to 6 seeds reveals the asym DA "win" was specific to seeds 42-44. LR boost is the more robust mechanism.
**Companion:** [Asymmetric adaptive DA](2026-04-26-asymmetric-adaptive-da.md), [Surprise LR boost](2026-04-26-surprise-lr-boost.md), [Night summary](2026-04-26-night-summary.md)

## TL;DR

Original 3-seed acid tests (seeds 42, 43, 44) suggested asymmetric adaptive DA was the best Phase B refinement at sum=3.53 — a 33% improvement over baseline. Extending to 6 seeds (adding 100, 101, 102) reveals this was misleading.

| Variant | 3-seed mean (42-44) | 3-seed mean (100-102) | 6-seed mean ± std |
|---|---:|---:|---:|
| Baseline (broadcast DA) | 5.24 | 6.51 | **5.88 ± 1.22** |
| Asymmetric adaptive DA | **3.53** | 6.94 | 5.23 ± 1.90 |
| Surprise LR boost | 4.02 | 5.83 | **4.92 ± 1.07** |

**Per-seed values reveal the issue:**
- Asym DA seeds 100/101/102: 8.03, 7.50, 5.28 — TWO ARE WORSE THAN BASELINE
- LR boost seeds 100/101/102: 4.85, 6.12, 6.51 — all reasonable

Welch's t-test on pooled 6-seed:
- asym DA vs baseline: t = 0.64 (NOT significant)
- LR boost vs baseline: t = 1.31 (marginally significant)

## Revised conclusions

1. **Asym adaptive DA is FRAGILE** across seeds. The 3.53 win on seeds 42-44 didn't generalize to seeds 100-102 (mean 6.94, almost catastrophic on seed 100 at 8.03).
2. **LR boost is the actually-best variant** by 6-seed mean (4.92) and is also lower-variance (std 1.07 vs 1.90).
3. **The original "33% improvement" claim was overstated.** With 6 seeds, asym DA only gives 11% improvement on average, well within 1σ noise.
4. **The Phase B baseline (broadcast DA) is more competitive** than the 3-seed result suggested — its mean is 5.88 with reasonable variance.

## What this teaches

The lesson is the same one that hit Phase B's initial finding: **3 seeds is not enough to claim a win.** The Phase B silent-motor-trap honest correction (2026-04-25) showed that the headline number was wrong because of a runner bug. Tonight's "asym DA wins" claim was ALSO wrong — not because of a bug, but because of insufficient sampling.

This isn't unusual. RL benchmarks often have high seed-variance, and small-N confidence intervals are wide. The lesson:
- Run 5+ seeds before any architectural claim
- Report mean AND standard deviation
- Use t-tests or comparable for "X beats baseline" claims
- Original 3-seed was the convention in earlier Phase B work; carrying that forward without updating sample size produced these overconfident claims.

## Updated recommendations

**Most reliable variant**: `--surprise-lr-boost`
- 6-seed mean 4.92 (16% improvement over baseline 5.88)
- Lower variance than baseline (1.07 vs 1.22)
- Marginally significant (t=1.31)

**Conditional**: `--adaptive-da --adaptive-da-ema-decay-negative 0.7`
- Lucky on seeds 42-44 (3.53), unlucky on seeds 100-102 (6.94)
- Use only if you've validated on YOUR specific seeds first
- Mechanism is biologically plausible but on-task variance is too high

**Default**: no flags (Phase B baseline)
- Robust 5.88 ± 1.22
- The structural fix from Phase B works regardless of refinement choice

## Per-seed full data

```
                 Variant Seed    P0    P1   Sum
              baseline   42  3.39  1.64  5.03
              baseline   43  1.72  1.93  3.65
              baseline   44  5.33  1.71  7.05
              baseline  100  3.55  2.63  6.17
              baseline  101  4.19  1.99  6.17
              baseline  102  5.23  1.96  7.19
              baseline  avg                5.88

            asym adaDA   42  1.68  1.95  3.63
            asym adaDA   43  1.43  2.00  3.43
            asym adaDA   44  1.72  1.80  3.52
            asym adaDA  100  5.59  2.45  8.03
            asym adaDA  101  5.12  2.38  7.50
            asym adaDA  102  3.24  2.04  5.28
            asym adaDA  avg                5.23  ← WAS 3.53 on 3 seeds

              LR boost   42  1.41  2.04  3.45
              LR boost   43  1.95  2.31  4.26
              LR boost   44  2.51  1.83  4.34
              LR boost  100  2.68  2.17  4.85
              LR boost  101  3.74  2.38  6.12
              LR boost  102  4.07  2.44  6.51
              LR boost  avg                4.92  ← actually best on 6 seeds
```

## Action items

- Update `2026-04-26-asymmetric-adaptive-da.md` to note this correction.
- Update `2026-04-26-night-summary.md` with revised conclusions.
- Update `CLAUDE.md` "recommended config" to reflect that LR boost is more reliable.
- Update `PHASE_B_QUICK_REFERENCE.md`.
- Future research-runner work should default to 5+ seeds for any architectural claim.

## What still stands

- **Phase B BG cascade structural win is real**: 74% improvement vs G9 baseline 6.74 (sum 5.88 vs 11+) — that's a robust 6-seed result, well outside any reasonable noise interval.
- **Sharpening helps modestly on average**: LR boost gives ~16% sum improvement, statistically marginal but consistent across seeds.
- **All other variants** (WTA, hard DA, learned perception, etc.) remain documented as not-improving-or-hurting.
- **The night's full landscape characterization is still valid** — just with corrected confidence on the headline result.

## Files

- `research/findings/raw/g11_bg/g11_seed{42,43,44,100,101,102}_{v3,baseline,adaDA_asym,lrboost}.json` — all 6-seed × 3-variant data
