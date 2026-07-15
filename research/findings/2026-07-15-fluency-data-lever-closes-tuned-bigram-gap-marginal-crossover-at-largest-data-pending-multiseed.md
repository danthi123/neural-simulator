# Fluency crossover: the DATA lever CLOSES the tuned-bigram gap monotonically, with a MARGINAL crossover at the largest data (np=200/nt=96000, sel_over_TUNED +0.025, single-seed) — decelerating; multi-seed confirmation + adversarial-verify in flight

**Date:** 2026-07-15 (compute unlocked) · **Status:** 25-point (np × nt) surface aggregated vs the TUNED bigram (per the standing anti-confound rule — the add-1 strawman was refused). Honest read + the crossover-confirmation batch launched. NOT a GO yet (single-seed crossover, tiny margin, 3×-caught confound → adversarial-verify required).

## The surface (`sel_over_TUNED = tuned_bigram_ce − sel_ce`; >0 = the selective-SSM generator BEATS the tuned bigram)
Selective-SSM generator on the real EMERGE stream, vectorized scale runner, vs the TUNED add-k bigram baked into the runner:
```
np=200:  nt=3k -0.375 | 6k -0.295 | 12k -0.172 | 24k -0.069 | 48k -0.051 | 96k +0.025   <- CROSSOVER (seed 42)
np=500:  nt=3k -0.347 | 6k -0.234 | 12k -0.106 | 24k -0.015/-0.018 (s43/44)
np=1000: nt=6k -0.212 | 12k -0.118
```
- **The DATA lever CLOSES the gap monotonically** at fixed reservoir size, and CROSSES 0 at the largest data point (np=200/nt=96000: +0.025). np=500 tracks it (−0.015 by nt=24000, near-crossing sooner with a bigger reservoir).
- **Reservoir size helps modestly** (np=500 > np=200 at fixed nt; np=1000 ≈ np=500 — saturating).
- **The deceleration is real:** per-doubling gain fades (np=200: +0.08 → +0.12 → +0.10 → +0.02 → +0.08). A marginal crossover, not a runaway win.
- **The deep-tail mechanism holds ROBUSTLY throughout:** sel_over_bag (vs the memoryless bag-of-prefix control) ≈ +0.85 to +1.18 at every point — the selective SSM's dynamics are load-bearing regardless of the bigram comparison.

## Honest read
The reservoir-LM generator's OVERALL fluency **asymptotically reaches / marginally crosses the tuned bigram as data grows** — the crossover appears at ~nt=96k (np=200, seed 42, +0.025). This is a PROMISING but MARGINAL, single-seed, decelerating signal. Given the ordered-bigram-starvation confound was caught THREE times this session (input-repr gate, vec-scale trajectory, and the standing ROADMAP §9 flag), the bar for a "fluency crossover GO" is high:
- **Confirmation in flight (compute unlocked):** np=200/nt=96000 seeds 43+44 (is the crossover seed-robust?), np=500/nt=96000 seed 42 (bigger reservoir at crossover data), np=200/nt=192000 seed 42 (does the trend keep rising past 0 or asymptote at ~0?). Batch `byirprbgy`.
- **Before any GO:** adversarial-verify (a skeptic pass) — confirm sel_ce is measured on the same held-out split as tuned_bi_ce, the tuned-k was swept on THIS data, and the crossover isn't an eval-set artifact.

## The decision this resolves
- If the crossover is **seed-robust AND the trend keeps rising** with more data/reservoir → the selective-SSM generator genuinely surpasses the tuned bigram at scale (a real fluency win) → the full ~23.7M-word scale run is warranted.
- If it **asymptotes at ~0** (crosses marginally then flattens) → the reservoir-LM generator is tuned-bigram-EQUIVALENT at tractable scale (matches, doesn't decisively beat) → the deep-tail mechanism is the real deliverable and the next fluency lever is a richer substrate (the emergence-engine recurrent cortex), not more reservoir-LM data.

Either outcome is decisive for the mission's fluency question. The deep-tail mechanism (sel_over_bag ~+1.0) is a committed, robust, real result independent of the bigram-crossover verdict.
