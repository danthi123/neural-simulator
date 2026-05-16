# Cross-benchmark failure analysis — the bottleneck is function-word category encodability, NOT pattern/seed defects

## TL;DR

A recurring failure pattern appeared across every 320 benchmark.
Cross-correlating the row-level failures of three independent
benchmarks (pair / sentence / interference) against structural
features yields a non-obvious, actionable conclusion:

- **Sparse-pattern overlap does NOT predict failure** (failed-concept
  mean max-overlap 10.36 vs never-failed 10.58 — identical).
- **Orthogonal-drive overlap is 0 for all** (never the cause).
- **Only 2 of 59 concepts fail in ≥2 independent benchmarks** →
  failures are overwhelmingly **benchmark-specific (stochastic),
  NOT concept-intrinsic**. There are no consistently "cursed" concepts.
- Failure IS bridge-correlated: **adj 3.1% ≪ nouns 11% < verbs 15% <
  functional 25% < spatial 30%**.

**The bottleneck is the function-word categories' intrinsic
encodability, not a fixable per-pattern or per-seed defect.**

## Evidence

Target-failure rate by bridge (pooled across all 3 benchmarks):

| Bridge | target-fail | rate |
|---|---|---|
| bridgeC_adj | 1/32 | **3.1%** (cleanest) |
| bridgeA_nouns | 1/9 | 11.1% |
| bridgeB_verbs | 5/33 | 15.2% |
| bridgeE_functional | 4/16 | 25.0% |
| bridgeD_spatial | 3/10 | **30.0%** (worst) |

Repeat offenders (fail in ≥2 independent benchmarks): only `every`
(functional) and `touch` (verb). 57 of 59 concepts-with-data do NOT
fail consistently → the recurring "functional/verb weakness" is a
**bridge-level statistical tendency**, not specific bad concepts.

Structural predictors (failed-any vs never-failed concepts):
- mean pattern max-overlap: 10.36 vs 10.58 → **not predictive**
- mean drive max-overlap: 0.00 vs 0.00 → never a factor
- mean vocab idx: 26.6 vs 31.8 → weak, non-mechanistic

## Why this matters (reframes prior conclusions)

1. **Disconfirms the pattern-overlap hypothesis at the
   cross-benchmark level.** The idx-12 / "unlucky high-overlap
   pattern" framing (320 SHIPPED doc) does not generalize: across
   independent benchmarks, overlap simply doesn't separate failures
   from successes. The flagged recovery task's overlap-rejection
   lever will help LESS than its premise assumed — worth noting in
   that task.
2. **Reframes the bottleneck as linguistic, not numerical.** All
   bridges share architecture + the seed-42 pattern set, yet adj is
   3% and spatial/functional are 25–30%. The difference is the
   VOCABULARY: concrete adjectives (red/big/hot — semantically
   distinct) encode into separable engrams; short high-frequency
   function/spatial words (in/on/every/one/maybe) produce
   less-separable lang_input drive + overlapping ensembles. This
   mirrors real language acquisition, where function words are a
   well-known hard, late-acquired class — an expected, not
   anomalous, result.
3. **Actionable for future design (informs, does not implement).**
   Improving function-word conversational reliability needs better
   *category encoding* (e.g., dedicated higher-capacity coding for
   function words, or accepting them as inherently lower-confidence
   and routing around them) — NOT more pattern/seed tuning. This is
   input for a future brainstorming-class design decision; no
   implementation is taken here.

## Honest scope

- Single-seed (42) row data; n per bridge is modest (spatial 10,
  functional 16) so the 25–30% rates have wide CIs. The QUALITATIVE
  ordering (adj cleanest, function-word bridges weakest) is robust
  across 3 independent benchmarks; the exact percentages are not.
- Pure analysis of already-committed JSON + deterministic
  pattern/drive functions. No GPU, no implementation.

## Files

- `research/runners/g20_failure_analysis.py`
- Inputs: `g20_{xbridge,sentence,interference}_bench_320.json`
- Reframes: `2026-05-16-G20-sparse-ensemble-320concept-SHIPPED.md`
  (idx-12), and the flagged 320-recovery task's overlap-rejection
  premise.
