# 2026-05-02 — Multi-seed validation in progress (live document)

This doc tracks the multi-seed validation of the v2 breakthrough config
(Hebbian off + stdp_w_max=5 + readout init=0.5). Updated as each seed
completes. Final summary will be a separate finalized doc.

## Headline status

| Seed | Status | I→W | W→A | Tokens learned | Training-time corr |
|---|---|---|---|---|---|
| 42 | DONE | **33.0%** (p=0.042) | 27.0% | 3/4 | 29.6% |
| 43 | DONE | 25.0% (p=0.55) | 29.0% | 2/4 | 38.2% |
| 44 | RUNNING | TBD | TBD | TBD | TBD |

**Variance across seeds is significant.** Seed=42's 33% may be favorable
variance; seed=43 returned to chance. With n=2 averaging 29% I→W (still
above chance trend but not significant). Need more seeds to determine
true accuracy.

## Per-direction patterns (consistent across seeds)

```
token-targeted weight differential:
                  seed42        seed43
north             -0.079 REV    -0.138 REV     <-- CONSISTENT REV
east              +0.210 LEARN  +0.116 LEARN   <-- CONSISTENT LEARN
south             +0.304 LEARN  -0.060 ~       <-- VARIES
west              +0.073 LEARN  +0.199 LEARN   <-- CONSISTENT LEARN
```

**North is REVERSED in BOTH seeds.** This is consistent — likely cascade
structural N-bias documented in `g11_bg_runner.py` line 1578: "cortex_N
fires 2x more at init". Without compensation, language→motor_N signal
is drowned out by N-baseline activity, and reward-modulated STDP can
even create REVERSED preference (motor_N fires anyway, so its firing
isn't language-token-specific).

**East and West LEARN in both seeds.** These directions are stably
distinguishable from baseline.

**South varies.** Sometimes learns (seed=42), sometimes fails (seed=43).

## Per-direction eval (W→A, the one that varies most)

```
seed=42:  N 28%  E 20%  S 40%  W 20%   (dominant: south)
seed=43:  N 20%  E 32%  S 44%  W 20%   (dominant: south)
```

South is the BEST W→A direction in BOTH seeds! Despite being
inconsistent in weight diagnostic. Combined with the strong reading on
W→A=32% at drive=500 in the sweep (where east+south reached 40%), this
suggests:

- South can be read out cleanly when drive is right
- North suffers from cascade structural bias
- East/West moderately learn

## Key findings so far

1. **The Hebbian fix is REAL and CONSISTENT.** Across all seeds tested,
   weights preserve at design values (mean 2.0/3.0/0.5) instead of
   collapsing to 0.05 floor.

2. **Differential learning happens but variance is significant.** Mean
   of 2 seeds: I→W ~29%, W→A ~28%. Both above chance but not at
   significance with n=2.

3. **North is structurally challenged.** Both seeds REVERSE on north.
   This is not noise — it's a consistent architectural pattern.

4. **East and West are reliable learners.** Stable across seeds.

## Recommendation

Run 6-seed validation total. Currently 2 done; seed=44 running.
After seed=44, if budget allows, queue seeds 100, 101, 102.

For NORTH-direction fix, two paths:
- (A) Reduce cascade structural N-bias in build_bg_brain_regions
- (B) Asymmetric reward / stronger drive specifically for north training trials
- Likely need (A); architectural change deferred.

## In-flight

- PID 2724 seed=44, launched 06:35:56, ETA ~07:25.
