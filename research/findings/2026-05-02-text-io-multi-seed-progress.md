# 2026-05-02 — Multi-seed validation in progress (live document)

This doc tracks the multi-seed validation of the v2 breakthrough config
(Hebbian off + stdp_w_max=5 + readout init=0.5). Updated as each seed
completes. Final summary will be a separate finalized doc.

## Headline status

| Seed | Status | I→W | W→A | Tokens learned | Training-time corr |
|---|---|---|---|---|---|
| 42 | DONE | **33.0%** (p=0.042) | 27.0% | 3/4 | 29.6% |
| 43 | DONE | 25.0% (p=0.55) | 29.0% | 2/4 | 38.2% |
| 44 | DONE | 27.0% (p=0.36) | 26.0% | 3/4 | 43.5% |
| 100 | RUNNING | TBD | TBD | TBD | TBD |

**3-seed mean:** I→W = **28.3%**, W→A = **27.3%**.

Both above 25% chance but variance is significant. With only 3 seeds, no
formal significance test is meaningful (need ≥5 seeds for reliable
mean+std). Trend is positive but not yet conclusive.

**Variance across seeds is significant.** Seed=42's 33% may be favorable
variance; seed=43 returned to chance. With n=2 averaging 29% I→W (still
above chance trend but not significant). Need more seeds to determine
true accuracy.

## Per-direction patterns (consistent across 3 seeds)

```
token-targeted weight differential:
                  seed42        seed43        seed44
north             -0.079 REV    -0.138 REV    -0.094 REV    <-- CONSISTENTLY REVERSED
east              +0.210 LEARN  +0.116 LEARN  +0.188 LEARN  <-- CONSISTENTLY LEARNS
south             +0.304 LEARN  -0.060 REV    +0.075 LEARN  <-- VARIES
west              +0.073 LEARN  +0.199 LEARN  +0.021 weak   <-- CONSISTENTLY LEARNS
```

**North is REVERSED in ALL 3 seeds.** This is structural, not noise.
Likely cascade structural N-bias documented in `g11_bg_runner.py`:
"cortex_N fires 2x more at init". Without compensation, language→motor_N
signal is drowned out by N-baseline activity. Reward-modulated STDP
sees motor_N firing for non-north targets too, so the differential
"north_active → motor_N" preference fails to grow above the
"north_active → motor_other" levels.

**East and West LEARN in all 3 seeds.** These directions are stably
distinguishable from baseline. The lang_input → motor_E and
lang_input → motor_W pathways consistently grow target-preference.

**South varies across seeds.** Sometimes learns strongly (seed=42 +0.30),
sometimes essentially fails (seed=43 -0.06), sometimes weakly learns
(seed=44 +0.08). The variance is more than noise — depends on
seed-specific cascade dynamics during training.

## I→W vs W→A dissociation (interesting finding)

Seed=44 illustrates this: weight diagnostic shows north REVERSED for
PFC-bypass (lang_input → motor_N), but I→W eval shows north got
12/22 = 54.5% (best of all directions in that seed)!

This is because I→W and W→A use DIFFERENT pathways:
- I→W: image → retina → V1 → V2 → IT → language_output
       (and cortex_N → language_output)
- W→A: language_input → cortex_X → BG cascade → motor_X
       (and lang_input → motor_X PFC-bypass)

The cortex_N → language_output and IT → language_output pathways can
learn correctly (north image → north output) even when the
lang_input → motor_N PFC-bypass is structurally biased.

This dissociation is BIOLOGY-CONSISTENT: real dorsal/ventral language
streams (Wernicke for comprehension via temporal cortex; Broca for
production via frontal cortex) are anatomically separable. Damage to
one doesn't necessarily damage the other (Geschwind 1965).

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
