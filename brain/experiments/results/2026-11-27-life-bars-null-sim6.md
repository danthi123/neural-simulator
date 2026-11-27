# Life bars permutation null (6 seeds: [9000, 9001, 9002, 9003, 9004, 9005]) - SIMULATED, bootstrap of the given lives' item pools, not data

K=200 null replicates, inner permutations 300 (registered observed run: 10000), master seed 20261127. Labels shuffled within seed; life_bars scoring unmodified.

## Per bar

| bar | observed verdict | obs stat | null mean | null p95 | null p99 | frac_ge | rank | beyond p95 | beyond p99 | false-pass | rule (<=0.05, +-2 SE MC) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P0 | FAIL | 0.101 | 0.068 | 0.225 | 0.294 | 0.373 | 75 | False | False | 0.000 | n/a (not a p-value bar) |
| SAV-G | FAIL | -0.127 | -0.000 | 0.374 | 0.474 | 0.682 | 137 | False | False | 0.035 | inconclusive |
| SAV-dR | FAIL | -0.333 | -0.030 | 1.333 | 1.667 | 0.746 | 115 | False | False | 0.030 | n/a (not a p-value bar) |
| TRF-T4 | PASS | 0.497 | 0.004 | 0.396 | 0.543 | 0.035 | 7 | True | False | 0.030 | inconclusive |
| TRF-K | FAIL | -0.295 | -0.011 | 0.540 | 0.904 | 0.821 | 165 | False | False | 0.430 | n/a (not a p-value bar) |
| TRF-C | PASS | 0.594 | -0.009 | 0.561 | 0.684 | 0.040 | 8 | True | False | 0.105 | n/a (not a p-value bar) |
| CUE-R | FAIL | 0.021 | -0.082 | 0.042 | 0.083 | 0.119 | 18 | False | False | 0.005 | yes |
| CUE-D | PASS | 0.506 | 0.002 | 0.106 | 0.165 | 0.005 | 1 | True | True | 0.000 | n/a (not a p-value bar) |

## Composite verdicts (all label-dependent bars of the section PASS) - UPPER BOUND ONLY; the rule is applied per bar above

| section | observed | null false-pass rate | SE | rule (<=0.05, +-2 SE MC) |
|---|---|---|---|---|
| p0 | not PASS | 0.000 | 0.000 | yes |
| savings | not PASS | 0.000 | 0.000 | yes |
| transfer | not PASS | 0.005 | 0.005 | yes |
| cue | not PASS | 0.000 | 0.000 | yes |
| all | not PASS | 0.000 | 0.000 | yes |

## Null distribution of the seed count (count: replicates)

- P0 (observed 0): {'0': 177, '1': 21, '2': 1, '3': 1}
- SAV-G (observed 2): {'0': 3, '1': 18, '2': 48, '3': 61, '4': 51, '5': 16, '6': 3}
- SAV-dR (observed 1): {'0': 71, '1': 84, '2': 36, '3': 8, '4': 1}
- TRF-T4 (observed 5): {'1': 20, '2': 46, '3': 65, '4': 45, '5': 22, '6': 2}
- TRF-K (observed -): {'None': 200}
- TRF-C (observed 5): {'0': 3, '1': 19, '2': 52, '3': 59, '4': 46, '5': 20, '6': 1}
- CUE-R (observed 1): {'0': 62, '1': 99, '2': 34, '3': 4, '4': 1}
- CUE-D (observed -): {'None': 200}
