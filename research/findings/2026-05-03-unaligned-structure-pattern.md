# Unaligned-structure pattern analysis

Analyzed 39 W->A confusion matrices across 19 conditions.

## Most common best-permutation across all runs

If learning was real, the TRUE permutation `(N, E, S, W)` would dominate. Instead:

| best perm | count | % | mapping (north->_, east->_, south->_, west->_) |
|---|---|---|---|
| `ENSW` | 4 | 10.3% | north->E, east->N, south->S, west->W |
| `SNEW` | 4 | 10.3% | north->S, east->N, south->E, west->W |
| `ESNW` | 3 | 7.7% | north->E, east->S, south->N, west->W |
| `ESWN` | 3 | 7.7% | north->E, east->S, south->W, west->N |
| `SEWN` | 3 | 7.7% | north->S, east->E, south->W, west->N |
| `SWNE` | 3 | 7.7% | north->S, east->W, south->N, west->E |
| `NWES` | 3 | 7.7% | north->N, east->W, south->E, west->S |
| `WENS` | 3 | 7.7% | north->W, east->E, south->N, west->S |
| `WSNE` | 2 | 5.1% | north->W, east->S, south->N, west->E |
| `NSEW` | 2 | 5.1% | north->N, east->S, south->E, west->W |
| `WNSE` | 2 | 5.1% | north->W, east->N, south->S, west->E |
| `ENWS` | 2 | 5.1% | north->E, east->N, south->W, west->S |
| `WNES` | 2 | 5.1% | north->W, east->N, south->E, west->S |
| `EWSN` | 1 | 2.6% | north->E, east->W, south->S, west->N |
| `EWNS` | 1 | 2.6% | north->E, east->W, south->N, west->S |
| **TRUE (NESW)** | **0** | **0.0%** | (never the best!) |

## Per-word: where does each word's signal go?

Across all runs, count what action the BEST permutation
assigned to each word.

| word | true action | most common best-perm action | count | % |
|---|---|---|---|---|
| north | N | E (DIFFERENT from true) | 14 | 35.9% |
|  |  | (all 4 actions: N=5, E=14, S=10, W=10) |
| east | E | N (DIFFERENT from true) | 14 | 35.9% |
|  |  | (all 4 actions: N=14, E=7, S=10, W=8) |
| south | S | N (DIFFERENT from true) | 12 | 30.8% |
|  |  | (all 4 actions: N=12, E=11, S=8, W=8) |
| west | W | W (matches true) | 13 | 33.3% |
|  |  | (all 4 actions: N=8, E=7, S=11, W=13) |

## Per-cell average count across all runs

Total counts in each (word, action) cell, summed across all
runs. If cascade has structural biases, certain cells will
dominate regardless of which word is driving.

| | -> N | -> E | -> S | -> W | row total |
|---|---|---|---|---|---|
| north | **224 (23.0%)** | 275 (28.2%) | 233 (23.9%) | 243 (24.9%) | 975 |
| east | 251 (25.7%) | **275 (28.2%)** | 234 (24.0%) | 215 (22.1%) | 975 |
| south | 253 (25.9%) | 271 (27.8%) | **226 (23.2%)** | 225 (23.1%) | 975 |
| west | 231 (23.7%) | 247 (25.3%) | 251 (25.7%) | **246 (25.2%)** | 975 |

## Action prediction frequency (overall cascade bias)

How often is each action predicted, summed across all words?
If the architecture were unbiased, each action would be ~25%.

| action | total predictions | % |
|---|---|---|
| N | 959 | **24.6%** |
| E | 1068 | **27.4%** |
| S | 944 | **24.2%** |
| W | 929 | **23.8%** |

## Implication

The most common best permutation is `ENSW` (4/39 runs = 10.3%).

If a single permutation appeared as best across many seeds, the
architecture has a CONSISTENT structural bias that overrides
training. If best perms are scattered (each ~5-10% frequency),
the bias is seed-dependent (each random init creates its own
private misalignment).

**Result: scattered (seed-dependent) bias** — no single
permutation dominates. Each seed builds its own private
misalignment from random init dynamics.
