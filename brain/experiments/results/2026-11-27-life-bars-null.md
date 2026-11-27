# Life bars permutation null (3 seeds: [319, 320, 321])

K=200 null replicates, inner permutations 300 (registered observed run: 10000), master seed 20261127. Labels shuffled within seed; life_bars scoring unmodified.

## Per bar

| bar | observed verdict | obs stat | null mean | null p95 | null p99 | frac_ge | rank | beyond p95 | beyond p99 | false-pass | rule (<=0.05, +-2 SE MC) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P0 | FAIL | 0.101 | 0.113 | 0.356 | 0.441 | 0.542 | 109 | False | False | 0.000 | n/a (not a p-value bar) |
| SAV-G | FAIL | -0.132 | -0.020 | 0.604 | 0.860 | 0.617 | 124 | False | False | 0.035 | inconclusive |
| SAV-dR | FAIL | -2.000 | -0.057 | 1.367 | 2.667 | 0.960 | 183 | False | False | 0.050 | n/a (not a p-value bar) |
| TRF-T4 | PASS | 0.812 | 0.000 | 0.586 | 0.879 | 0.025 | 5 | True | False | 0.025 | inconclusive |
| TRF-K | PASS | 0.222 | 0.041 | 1.019 | 1.373 | 0.363 | 73 | False | False | 0.505 | n/a (not a p-value bar) |
| TRF-C | PASS | 0.830 | 0.042 | 0.922 | 1.169 | 0.075 | 15 | False | False | 0.150 | n/a (not a p-value bar) |
| CUE-R | FAIL | 0.095 | -0.120 | 0.048 | 0.095 | 0.025 | 2 | True | True | 0.000 | yes |
| CUE-D | PASS | 0.523 | 0.002 | 0.159 | 0.204 | 0.005 | 1 | True | True | 0.000 | n/a (not a p-value bar) |

## Composite verdicts (all label-dependent bars of the section PASS) - UPPER BOUND ONLY; the rule is applied per bar above

| section | observed | null false-pass rate | SE | rule (<=0.05, +-2 SE MC) |
|---|---|---|---|---|
| p0 | not PASS | 0.000 | 0.000 | yes |
| savings | not PASS | 0.005 | 0.005 | yes |
| transfer | PASS | 0.005 | 0.005 | yes |
| cue | not PASS | 0.000 | 0.000 | yes |
| all | not PASS | 0.000 | 0.000 | yes |

## Null distribution of the seed count (count: replicates)

- P0 (observed 0): {'0': 180, '1': 20}
- SAV-G (observed 1): {'0': 36, '1': 62, '2': 74, '3': 28}
- SAV-dR (observed 0): {'0': 98, '1': 87, '2': 15}
- TRF-T4 (observed 3): {'0': 20, '1': 75, '2': 78, '3': 27}
- TRF-K (observed -): {'None': 200}
- TRF-C (observed 3): {'0': 21, '1': 75, '2': 74, '3': 30}
- CUE-R (observed 1): {'0': 141, '1': 56, '2': 3}
- CUE-D (observed -): {'None': 200}
