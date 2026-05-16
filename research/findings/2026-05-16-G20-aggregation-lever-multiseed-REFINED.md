# Aggregation lever multi-seed — REFINED: modest, seed-variant, NOT robust (honest down-grade)

## TL;DR

Multi-seed confirmation of the `samebridge_downweight` aggregation
lever (seeds 43–46, existing 160 bridges, zero retrain — pure
query-time lever) **down-grades** the earlier clean seed-42 +3.3pp:

| Seed | max | samebridge_downweight | Δ |
|---|---|---|---|
| 42 (prior, remediated 320) | — | — | +3.3pp |
| 43 | 90.0% | 86.7% | **−3.3pp (REGRESSED)** |
| 44 | 90.0% | 93.3% | +3.3pp |
| 45 | 90.0% | 93.3% | +3.3pp |
| 46 | 90.0% | 96.7% | +6.7pp |

**Multi-seed (43–46): mean +2.5pp, range [−3.3, +6.7], 1/4
regressed.** Including seed 42: 4/5 positive, 1/5 negative. The
lever is a **modest net-positive heuristic with real seed variance
and occasional regression — NOT a robust, blindly-applicable win.**

## Honest interpretation

- Multi-seed rigor did exactly its job: the seed-42 +3.3pp snapshot
  was favorable-ish. The honest expectation is **~+2.5pp mean with
  σ comparable to the effect, and a real chance of regression on a
  given seed** (seed 43: −3.3pp).
- This is mechanistically consistent with the `perbridge_norm`
  NEGATIVE: the query-word's home-bridge firing magnitude is
  **sometimes legitimately informative** (the home bridge often DOES
  hold the right strong answer). Blanket ×0.4 down-weighting helps
  when the home bridge is a distractor (the 50% same-bridge-crosstalk
  misses) but HURTS when the home bridge held the correct strong
  signal — net mildly positive, but seed-dependent which way it
  tips.
- Substrate caveat: seed-42's +3.3pp was on the *remediated 320*
  ensemble; 43–46 here are *un-remediated 160* (the only existing
  multi-seed artifacts; zero-retrain constraint). Same lever,
  slightly different substrate — the qualitative conclusion
  (modest, variable, occasionally-regressing) is the robust takeaway,
  not a precise cross-substrate point estimate.

## Refined stacked-levers picture (honest)

| Lever | Robustness | Effect |
|---|---|---|
| Under-recall remediation | **Robust** (5/5 bridges, mechanism-deterministic) | +3.3pp (modest) |
| `samebridge_downweight` | **Seed-variant** (4/5 positive, 1/5 −3.3pp) | ~+2.5pp mean (not guaranteed) |
| `perbridge_norm` | — | **FALSIFIED** (−16.7pp) |

Honest combined expectation: remediation is the dependable piece
(+3.3pp, robust); samebridge_downweight adds ~+2.5pp on average but
should be applied **conditionally / tuned**, not as a blanket
default, given the real regression risk on some seeds. Earlier
"+6.7pp stacked" was a seed-42 best-case; the honest typical is
lower and variable.

## Recommendation (not overclaimed)

- Ship **under-recall remediation** as a default recipe add-on
  (robust, mechanism-deterministic, +3.3pp).
- Treat **samebridge_downweight** as an *optional, tunable*
  heuristic (mean ~+2.5pp, seed-variant, can regress) — NOT a
  default. The ×factor and a per-query confidence gate (only
  down-weight when the home candidate is ambiguous) are the
  dedicated session's tuning targets.
- `perbridge_norm`: do not pursue (cleanly falsified).

## Files

- `g20_aggms_s{43,44,45,46}_{max,samebridge_downweight}.json`
- Refines (honestly down-grades): `2026-05-16-G20-stacked-artifact-safe-levers-CLOSURE.md`'s
  seed-42 +6.7pp to a multi-seed ~ +5pp-typical-but-variable.
