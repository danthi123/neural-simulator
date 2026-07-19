# Minimal-isolation diagnostic — why is it below chance?

**Date:** 2026-05-04
**Source:** research/findings/raw/g11_bg/text_eval_minimal_iso_seed{42,43,44}.json
**Question:** Minimal-iso (cascade-stripped) gave 16.7% mean — below 25% chance. Why?

---

## TL;DR

- TRUE accuracy: [0.16, 0.13, 0.21, 0.26, 0.32, 0.21] (mean 21.5%)
- Best permutation accuracy: [0.39, 0.37, 0.38, 0.32, 0.32, 0.32] (mean 35.0%)
- North-South inversion present in 1/6 seeds
  (north->S rate > 30% AND south->N rate > 30%)
- E-as-sink (>= 2 words mapping to E): 4/6 seeds

**Best permutations vary by seed** — random architecture noise:
  - seed 42: `N:S->E:W->S:N->W:E` (acc 39.0%)
  - seed 43: `N:S->E:W->S:E->W:N` (acc 37.0%)
  - seed 44: `N:W->E:S->S:N->W:E` (acc 38.0%)
  - seed 100: `N:N->E:E->S:W->W:S` (acc 32.0%)
  - seed 101: `N:N->E:E->S:S->W:W` (acc 32.0%)
  - seed 102: `N:W->E:N->S:E->W:S` (acc 32.0%)

## Action distribution (motor pool firing rate, all words combined)

If the architecture has structural bias toward one motor pool, this will be skewed away from 25% per pool.

| seed | N | E | S | W |
|---|---|---|---|---|
| 42 | 26.0% | 24.0% | 25.0% | 25.0% |
| 43 | 25.0% | 26.0% | 28.0% | 21.0% |
| 44 | 26.0% | 33.0% | 22.0% | 19.0% |
| 100 | 31.0% | 32.0% | 20.0% | 17.0% |
| 101 | 25.0% | 36.0% | 22.0% | 17.0% |
| 102 | 30.0% | 27.0% | 20.0% | 23.0% |

## Per-seed details

### Seed 42

**TRUE-mapping accuracy:** 16.0%

Per-word TRUE accuracy:
- north -> N: 12.0%
- east -> E: 16.0%
- south -> S: 12.0%
- west -> W: 24.0%

**Best permutation:** `N:S->E:W->S:N->W:E` (accuracy 39.0%)

Top 5 permutations:
- `N:S->E:W->S:N->W:E`: 39.0%
- `N:W->E:S->S:N->W:E`: 35.0%
- `N:S->E:N->S:W->W:E`: 33.0%
- `N:S->E:E->S:N->W:W`: 32.0%
- `N:E->E:S->S:N->W:W`: 30.0%

Confusion matrix:

| word \ action | N | E | S | W | total |
|---|---|---|---|---|---|
| north | 3 | 5 | 11 | 6 | 25 |
| east | 6 | 4 | 8 | 7 | 25 |
| south | 11 | 5 | 3 | 6 | 25 |
| west | 6 | 10 | 3 | 6 | 25 |

NS-inversion: north->S 44.0%, south->N 44.0%

### Seed 43

**TRUE-mapping accuracy:** 13.0%

Per-word TRUE accuracy:
- north -> N: 8.0%
- east -> E: 16.0%
- south -> S: 16.0%
- west -> W: 12.0%

**Best permutation:** `N:S->E:W->S:E->W:N` (accuracy 37.0%)

Top 5 permutations:
- `N:S->E:W->S:E->W:N`: 37.0%
- `N:S->E:W->S:N->W:E`: 36.0%
- `N:S->E:N->S:W->W:E`: 35.0%
- `N:S->E:N->S:E->W:W`: 33.0%
- `N:S->E:E->S:W->W:N`: 30.0%

Confusion matrix:

| word \ action | N | E | S | W | total |
|---|---|---|---|---|---|
| north | 2 | 4 | 13 | 6 | 25 |
| east | 8 | 4 | 6 | 7 | 25 |
| south | 7 | 9 | 4 | 5 | 25 |
| west | 8 | 9 | 5 | 3 | 25 |

NS-inversion: north->S 52.0%, south->N 28.0%

### Seed 44

**TRUE-mapping accuracy:** 21.0%

Per-word TRUE accuracy:
- north -> N: 28.0%
- east -> E: 20.0%
- south -> S: 20.0%
- west -> W: 16.0%

**Best permutation:** `N:W->E:S->S:N->W:E` (accuracy 38.0%)

Top 5 permutations:
- `N:W->E:S->S:N->W:E`: 38.0%
- `N:S->E:W->S:N->W:E`: 37.0%
- `N:N->E:W->S:S->W:E`: 33.0%
- `N:W->E:N->S:S->W:E`: 33.0%
- `N:N->E:S->S:W->W:E`: 32.0%

Confusion matrix:

| word \ action | N | E | S | W | total |
|---|---|---|---|---|---|
| north | 7 | 6 | 6 | 6 | 25 |
| east | 7 | 5 | 7 | 6 | 25 |
| south | 10 | 7 | 5 | 3 | 25 |
| west | 2 | 15 | 4 | 4 | 25 |

NS-inversion: north->S 24.0%, south->N 40.0%

### Seed 100

**TRUE-mapping accuracy:** 26.0%

Per-word TRUE accuracy:
- north -> N: 40.0%
- east -> E: 32.0%
- south -> S: 8.0%
- west -> W: 24.0%

**Best permutation:** `N:N->E:E->S:W->W:S` (accuracy 32.0%)

Top 5 permutations:
- `N:N->E:E->S:W->W:S`: 32.0%
- `N:E->E:N->S:W->W:S`: 31.0%
- `N:N->E:S->S:E->W:W`: 30.0%
- `N:N->E:W->S:E->W:S`: 30.0%
- `N:S->E:N->S:E->W:W`: 29.0%

Confusion matrix:

| word \ action | N | E | S | W | total |
|---|---|---|---|---|---|
| north | 10 | 9 | 6 | 0 | 25 |
| east | 8 | 8 | 5 | 4 | 25 |
| south | 7 | 9 | 2 | 7 | 25 |
| west | 6 | 6 | 7 | 6 | 25 |

NS-inversion: north->S 24.0%, south->N 28.0%

### Seed 101

**TRUE-mapping accuracy:** 32.0%

Per-word TRUE accuracy:
- north -> N: 32.0%
- east -> E: 44.0%
- south -> S: 24.0%
- west -> W: 28.0%

**Best permutation:** `N:N->E:E->S:S->W:W` (accuracy 32.0%)

Top 5 permutations:
- `N:N->E:E->S:S->W:W`: 32.0%
- `N:N->E:E->S:W->W:S`: 31.0%
- `N:S->E:E->S:N->W:W`: 31.0%
- `N:E->E:N->S:S->W:W`: 30.0%
- `N:E->E:N->S:W->W:S`: 29.0%

Confusion matrix:

| word \ action | N | E | S | W | total |
|---|---|---|---|---|---|
| north | 8 | 11 | 6 | 0 | 25 |
| east | 6 | 11 | 4 | 4 | 25 |
| south | 7 | 6 | 6 | 6 | 25 |
| west | 4 | 8 | 6 | 7 | 25 |

NS-inversion: north->S 24.0%, south->N 28.0%

### Seed 102

**TRUE-mapping accuracy:** 21.0%

Per-word TRUE accuracy:
- north -> N: 28.0%
- east -> E: 16.0%
- south -> S: 20.0%
- west -> W: 20.0%

**Best permutation:** `N:W->E:N->S:E->W:S` (accuracy 32.0%)

Top 5 permutations:
- `N:W->E:N->S:E->W:S`: 32.0%
- `N:W->E:N->S:S->W:E`: 31.0%
- `N:W->E:S->S:N->W:E`: 29.0%
- `N:E->E:N->S:W->W:S`: 28.0%
- `N:W->E:S->S:E->W:N`: 28.0%

Confusion matrix:

| word \ action | N | E | S | W | total |
|---|---|---|---|---|---|
| north | 7 | 7 | 3 | 8 | 25 |
| east | 10 | 4 | 6 | 5 | 25 |
| south | 7 | 8 | 5 | 5 | 25 |
| west | 6 | 8 | 6 | 5 | 25 |

NS-inversion: north->S 12.0%, south->N 28.0%

## Interpretation

### Why below chance?

Pure random would give 25% TRUE accuracy in expectation. Below-chance means the network is making CORRELATED wrong answers — it's actively picking the wrong motor pool for at least some words. The mechanism could be:

1. **Reward eligibility window mismatch**: paired-stim training reinforces (lang_active, motor_target) pairs, but if the eval drives lang_active and the WINNER motor pool fires AFTER the intended one, eligibility might consolidate the wrong pair.

2. **Lateral inhibition asymmetry** (architecture has none, but reset windows might differ across motor pools).

3. **Sparse code overlap**: if 'north' and 'south' patterns share many active neurons, training on 'north' partially decreases weights from those overlapping neurons to motor_S (via STDP LTD on uncorrelated firing).

### Cross-seed pattern stability

Best permutations differ by seed = SEED-DEPENDENT noise with weak structural alignment. Each random init creates its own private bias. This pattern is consistent with the permuted-label control test result (0/45 prior runs had TRUE labels as the best of 24 permutations).

## Implication for biology sweep

The biology sweep (in flight) tests three fixes on this same minimal architecture:

- **+FS only** (motor PV-FS lateral inhibition): if the issue is pure WTA selection, this should help. Predicts: aligned ratio moves from 0/3 toward 4/6+.

- **+Topo only** (Wernicke->motor topographic prior): if the issue is sparse-code-overlap-induced LTD on the right pairs, the topographic prior gives STDP a head start with correct weights. Predicts: aligned ratio moves toward 4/6+.

- **+Topo +FS** (combined): if BOTH are needed.

If the NS-inversion pattern is structural (same best perm across seeds), the topographic prior should specifically fix it because the prior tells the network 'north -> motor_N' as starting weights.

If the pattern is seed-dependent random noise, then biology fixes might still help by providing systematic structure that STDP can refine, even if the underlying issue is overlap-driven.

