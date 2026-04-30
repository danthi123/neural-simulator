# 2026-04-30 — Reverse-order trajectory replay: NULL

## Setup

`g11_bg_runner.py --enable-reverse-replay`, multi-goal det, sleep replay
window at step 1350-1500 (start of phase 3, leaves 300 steps post-sleep
for phase-3 final-quarter measurement). 6 seeds × 2 conditions.

## Hypothesis

Real CA1/CA3 ripples replay trajectories in **reverse time order** during
NREM (Foster & Wilson 2006, Diba & Buzsaki 2007). Last-position-before-
goal replayed first, working backward to start. Biologically grounded as
TD-style backward credit assignment: high-reward states "send signal back"
through the trajectory.

The SCIENCE_ROADMAP §4.7 explicitly flagged "content quality is the
bottleneck" for sleep replay — implying that *what* is replayed matters
more than *when* plasticity is gated. This was the last of the three
pivots (A=scaling, B=harder benchmark, C=replay content) recommended in
the cluster-stacking synthesis.

## Result: NULL

| Cond (n=6) | Mean | Std |
|---|---|---|
| A+E+D (forward sampling, baseline) | 26.37 | 6.41 |
| A+E+D + reverse-replay | 26.07 | 7.09 |

Δmean = **-0.29** (essentially zero noise difference)
Welch t = **-0.07** (no effect)

Both conditions still far above A+E baseline (7.18 ± 1.58). Reverse-replay
content does NOT change the fundamental issue: **sleep replay during
multi-goal phases hurts learning regardless of replay order**.

## Cross-pivot summary (all three null)

| Pivot | Hypothesis | Result |
|---|---|---|
| **A: Scaling** | F v2's PF/PC scale-mismatch breaks at our reduced model | NULL — n_granule=1000 (4×) gave 6.34 vs 6.12 default |
| **B: Harder benchmark** | A+E sufficient for cheat-5; biology shines on harder tasks | NULL — multi (corner) is already the hardest per-phase; multi-fast / random / random-far all easier |
| **C: Replay content** | Reverse-order trajectory replay (Foster & Wilson 2006) | NULL — Δmean=-0.29 vs forward |

The cluster-stacking synthesis predicted these three pivots as the
remaining options after 9 cluster-stacking attempts past A+E all came
back NEUTRAL. With A, B, C all null, **the operational ceiling at A+E
6.97 ± 0.83 is robust to all explored modifications**.

## What this means

The bottleneck for cheat-5 closure is NOT:
- Adding more biology clusters (10 attempts NEUTRAL/PARTIAL)
- Increasing model scale at the cluster level (granule 4×)
- Making the benchmark harder (random/multi-fast/random-far)
- Changing replay content (reverse vs forward)

Possible remaining levers (none currently tested):
1. **Migrate to the FIXED replicated runner as canonical** — within the
   replicated runner, A+F v2 = 2.64 ± 0.17 vs A+E = 3.40 ± 0.57 (Welch
   t=-3.12). The replicated runner produces ~2× lower means than single,
   but cross-runner comparison is invalid. Need to identify which runner
   is "right" before claiming F v2 is a real ceiling break.
2. **Recency-weighted replay** — only replay trajectories from the
   *current* goal phase (drop stale trajectories from earlier phases).
   The "stale content" issue is more structural than the order issue.
3. **Hindsight Experience Replay (HER)** — relabel each step's reward
   retrospectively as if the reached position was the intended goal.
   Standard in robotics RL; biological correlate in mental-simulation.
4. **TD-style value critic** — add a learned value head, use TD error
   instead of step-level distance change for the plasticity signal.

## Decision

**Recommend pivoting to (1) — investigate the replicated-runner-vs-single
discrepancy.** That's the highest-value remaining lever because:
- The replicated runner already shows F v2 winning over A+E within itself
- If the replicated runner is "more correct" (e.g. its weight-jitter
  topology is closer to the intended distribution), then F v2 is a real
  ceiling break we just couldn't see before due to the runner bugs.
- If the single runner is canonical, the replicated discrepancy is a
  fixable (additional) bug.

Either way, identifying the runner-level discrepancy gives us either:
- A confirmed ceiling break (if replicated is canonical), OR
- A fully-aligned replicated runner that can do 6× speedup on parallel
  evals without confounding (if single is canonical).

Both outcomes are valuable.

## Files

- Results: `research/findings/raw/g11_bg/rev_AED*.json` (12 files)
- Implementation: `research/runners/g11_bg_runner.py:enable_reverse_replay`
  (~15 LOC in the existing sleep replay block)
- CLI flag: `--enable-reverse-replay`
- Cluster-stacking synthesis (now updated): `research/findings/2026-04-30-cluster-stacking-synthesis.md`
- F v2 correction-of-correction: `research/findings/2026-04-30-fv2-correction-replicated-runner-bug.md`
