# Cross-benchmark failure analysis — INDEX-intrinsic seed-42 pattern weakness (corrected)

## ⚠️ CORRECTION (supersedes the original conclusion below)

The original conclusion of this doc ("function-word category
encodability is the bottleneck") was **WRONG** and is retracted.
A deeper mechanistic check (committed same session) shows why:

- `orthogonal_drive_pattern(cue_idx=i,…)` and
  `generate_sparse_patterns(…,seed=42)` are **purely
  index-determined**. All 5 bridges train with `--seed 42`. So a
  concept at vocab index *k* has a **byte-identical** input code AND
  sparse pattern in *every* bridge. The bridges are neurally
  near-identical, differing only in word *labels* (tag names /
  readout), which are never used in the neural encoding.
- Therefore a real "function-word category" effect is
  **mechanistically impossible** — there is no neural channel by
  which "spatial" vs "adjective" words could differ.
- Attributing failures by **index** instead of bridge: only **2
  indices fail ≥2×** — idx 10 (`every`) 2/2, idx 42 (`touch`) 3/5.
  The failing unit is the **INDEX**, not the word or category.
  (Cf. idx-12 at the 64-concept tier — same phenomenon, different
  weak position.)
- The apparent "adj 3% ≪ spatial 30%" bridge spread is a **small-n
  sampling artifact**: only 4–8 targets were drawn per bridge, so
  which of the 64 indices happened to be probed per bridge dominated
  the per-bridge rate. Not a category property.

**Corrected conclusion:** failure is **index-intrinsic** — specific
positions in the seed-42 sparse pattern set (10, 42, and 12 at the
64-tier) are weak in a **bridge-agnostic** way. This is the *same*
root cause as the flagged idx-12 recovery. It therefore **STRENGTHENS
(does NOT temper) the flagged recovery's premise**: per-bridge
distinct seeds / overlap-rejection / a different seed *is* the right
lever, because the defect is a property of the index-determined
pattern set, exactly what those interventions change. The original
doc said the opposite; the original was wrong.

This correction is propagated forthrightly (same intellectual-honesty
discipline as the earlier seed-42 100%→92.7% correction). The
original analysis is preserved below for the audit trail.

---

## (ORIGINAL — RETRACTED) TL;DR

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
