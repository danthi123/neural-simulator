# Seed-quality structural predictor — NEGATIVE: static pattern-overlap does not explain failure

## TL;DR

Built a **falsifiable** structural predictor (per-seed
`generate_sparse_patterns` max-overlap distribution + outlier count)
and validated it against the 5 KNOWN 160 multi-seed per-bridge
results. **It failed.** Static sparse-pattern overlap does not
predict which seeds/indices fail — not at concept level (prior
cross-benchmark analysis) and not at seed level (this test, 60-seed
sample + 5-point validation).

This rigorously rules out the static-geometry explanation and means
the flagged recovery's **overlap-rejection lever is unlikely to
help**. The open question is **dynamical**, not static.

## The falsification

Predictor: a seed's pattern set is "bad" if it has more high-overlap
outlier patterns (overlap > mean+2σ). Validate vs known 160
(32-concept) per-bridge top-1:

| Seed | n_outliers | max_overlap | KNOWN per-bridge |
|---|---|---|---|
| 42 | 2 | 15 | **100.0%** |
| 43 | 2 | 12 | 96.9% |
| 44 | 2 | 13 | **100.0%** |
| 45 | 2 | 13 | **100.0%** |
| 46 | 2 | 12 | 93.8% |

- **All five seeds have n_outliers = 2** → zero discriminative power.
  The "clean-vs-weak separation = True" the script prints is a
  degenerate artifact (max-of-clean 2 ≤ min-of-weak 2 is trivially
  true), NOT real separation. exact-rank-match = False.
- **Overlap weakly ANTI-correlates with accuracy**: the worst seeds
  (43 = 96.9%, 46 = 93.8%) have the *lowest* max-overlap (12); the
  100% seeds include the *highest* (42 = 15). Higher overlap did not
  mean worse accuracy — if anything the reverse.

## Honest implications (3rd refinement of this root-cause)

This question has now been refined three times, each by careful
falsification:
1. "idx-12 = unlucky high-overlap pattern" (320 SHIPPED) — disconfirmed
   (idx 8/17 overlap more, pass).
2. "function-word category encodability" (cross-benchmark v1) —
   **retracted** (bridges neurally identical; impossible).
3. "index-intrinsic seed-pattern-overlap weakness" (cross-benchmark
   corrected) — **now also disconfirmed as a STATIC-overlap effect**:
   a falsifiable overlap predictor fails to rank the known seeds.

The failure indices are real and reproducible (idx-12 @ 64-tier; idx
10/42 in 320 benchmarks; the 1-fact dips in 160 seeds 43/46), but
they are **NOT explained by static sparse-pattern overlap geometry**.
The cause must be **dynamical** — how the engram-commit + STDP +
recall stimulation interact for particular patterns — which static
pairwise-overlap cannot capture.

## What this means for the flagged recovery task

- **Overlap-rejection in `generate_sparse_patterns` is unlikely to
  work** — you cannot reject "bad" patterns by an overlap criterion
  that does not predict badness. The flagged task should NOT pursue
  overlap-rejection as the primary lever.
- **Per-bridge distinct seeds may still help** (a different seed =
  a different pattern set that empirically may have fewer weak
  indices) — but there is **no validated structural metric to pick
  the seed a priori**. The earlier "try seed 63" recommendation is
  **withdrawn** — it came from the disconfirmed outlier metric.
  Seed selection would be empirical (run, measure), which is
  precisely the multi-hour search the analysis hoped to avoid.
- The genuinely open, correctly-localized question for that task:
  characterize the *dynamical* recall behavior of a known-failing
  index (e.g. idx-12) — instrument the engram-commit/recall for that
  pattern vs a passing one. That is GPU work for the dedicated
  session, now correctly targeted.

## Not overclaiming

This is a NEGATIVE result reported as the finding. No seed
recommendation is made (the metric that would produce one is
disconfirmed). The value is in *eliminating* a plausible-but-wrong
lever (overlap-rejection) before the flagged task spends GPU hours
on it, and in correctly relocating the open question to the
dynamical regime.

## Files

- `research/runners/g20_seed_quality_analysis.py` (the falsification)
- `research/findings/raw/g11_bg/g20_seed_quality.json`
- Supersedes the "STRENGTHENS overlap-rejection premise" line in
  `2026-05-16-G20-cross-benchmark-failure-analysis.md`'s correction
  block — that is now itself refined: the premise is NOT
  overlap-based.
