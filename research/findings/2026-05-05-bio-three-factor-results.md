# Result aggregation

**Headline:** **No real learning.** All conditions show 0-1 aligned of N. Architecture noise, not word-action learning.

## Summary

| Condition | n | true mean | best mean | excess | **aligned/n** | I->W mean |
|---|---|---|---|---|---|---|
| 3-factor vanilla | 6 | 25.5% | 33.3% | +7.8pp | **0/6** (noise) | - |
| 3-factor with topo + FS | 6 | 26.5% | 32.3% | +5.8pp | **1/6** (noise) | - |
| 3-factor with topo only | 6 | 26.5% | 33.5% | +7.0pp | **0/6** (noise) | - |

## Per-seed cross-condition

| seed | 3-factor vanilla | 3-factor with topo + FS | 3-factor with topo only |
|---|---|---|---|
| 42 | 21% | 26% | 24% |
| 43 | 31% | 20% | 30% |
| 44 | 25% | 27% | 25% |
| 100 | 23% | 27% | 23% |
| 101 | 23% | 30% * | 26% |
| 102 | 30% | 29% | 31% |

* marks seeds where TRUE = best permutation (aligned).

## Aligned details

| condition | seed | true | best | best_perm | aligned |
|---|---|---|---|---|---|
| 3-factor vanilla | 42 | 21.0% | 36.0% | WNSE | no |
| 3-factor vanilla | 43 | 31.0% | 33.0% | NEWS | no |
| 3-factor vanilla | 44 | 25.0% | 32.0% | NSWE | no |
| 3-factor vanilla | 100 | 23.0% | 36.0% | WESN | no |
| 3-factor vanilla | 101 | 23.0% | 32.0% | SWEN | no |
| 3-factor vanilla | 102 | 30.0% | 31.0% | WSNE | no |
| 3-factor with topo + FS | 42 | 26.0% | 35.0% | EWNS | no |
| 3-factor with topo + FS | 43 | 20.0% | 33.0% | WSEN | no |
| 3-factor with topo + FS | 44 | 27.0% | 32.0% | SWNE | no |
| 3-factor with topo + FS | 100 | 27.0% | 32.0% | WNSE | no |
| 3-factor with topo + FS | 101 | 30.0% | 30.0% | NESW | **YES** |
| 3-factor with topo + FS | 102 | 29.0% | 32.0% | SWEN | no |
| 3-factor with topo only | 42 | 24.0% | 34.0% | WNSE | no |
| 3-factor with topo only | 43 | 30.0% | 32.0% | NEWS | no |
| 3-factor with topo only | 44 | 25.0% | 32.0% | ESWN | no |
| 3-factor with topo only | 100 | 23.0% | 38.0% | WESN | no |
| 3-factor with topo only | 101 | 26.0% | 32.0% | NWES | no |
| 3-factor with topo only | 102 | 31.0% | 33.0% | NSWE | no |
