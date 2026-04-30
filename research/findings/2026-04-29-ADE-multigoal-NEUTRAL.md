# 2026-04-29 — A+D+E under multi-goal det: NEUTRAL/worse

**Run:** `g11_bg_runner.py --moving-goal --enable-msn-lateral-inhibition --enable-d1-d2-asymmetry --enable-striatal-pv-fsi --enable-cluster-a-closed-loop --enable-cluster-d-hippocampus --enable-cluster-e-topography --goal-schedule multi --deterministic`. 6 seeds × 2 conditions = 12 runs. `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

**Hypothesis (going in):** A+D was NEUTRAL with high variance driven by seed-100 catastrophic failure (Cluster D's CA3 recurrent autoassociator converging on the bad pattern-fragmentation attractor without SWR cleanup). Question: does Cluster E's topographic-cortex structural stability rescue D's variance issue when stacked together?

## Headline

**NO, E does not rescue D. A+D+E is slightly WORSE than baseline.**

| Condition | Mean | Std | n | Welch's t | Verdict |
|---|---|---|---|---|---|
| baseline (no clusters) | 6.76 | 2.42 | 6 | reference | acid-test |
| **A+D+E** | **9.03** | 4.12 | 6 | t=1.17 | NEUTRAL/worse |
| A+E (Step 4 reference) | 6.97 | 0.83 | 6 | — | **best operational** |
| A+D (this morning) | 8.67 | 5.13 | 6 | — | NEUTRAL with seed-100 outlier |

**A+D+E delta vs baseline: +2.27 (+33.6% mean, +70% std). NOT statistically significant.**

## Per-seed breakdown — same seed-100 pattern as A+D

| Seed | Baseline | A+D+E | Delta | Baseline phases (P0,P1,P2,P3) | A+D+E phases |
|---|---|---|---|---|---|
| 42 | 8.88 | **6.04** | −2.84 | [1.68, 0.91, 3.65, 2.65] | [2.89, 1.11, 1.02, 1.03] |
| 43 | 4.56 | 8.37 | +3.81 | [1.15, 1.02, 1.40, 0.99] | [1.01, 2.46, 3.63, 1.27] |
| 44 | 4.71 | 7.39 | +2.68 | [1.24, 0.88, 0.99, 1.59] | [2.11, 3.13, 1.13, 1.02] |
| **100** | **10.09** | **17.30** | **+7.21** | [1.11, 3.56, 4.48, 0.95] | [3.20, 2.98, 3.87, **7.25**] |
| 101 | 7.52 | 7.50 | −0.02 | [1.55, 1.77, 2.23, 1.97] | [1.13, 1.30, 1.27, 3.80] |
| 102 | 4.77 | 7.56 | +2.79 | [1.09, 1.20, 1.16, 1.32] | [1.58, 3.56, 1.11, 1.31] |

**Seed 100 pattern same as A+D**: P3=7.25 (catastrophic — agent never recovers from third transition). Without seed 100, A+D+E mean across other 5 seeds = **7.37** (slightly worse than the 5-seed baseline mean **6.09**).

A+D+E without seed-100 is comparable to baseline; with seed-100 it's much worse. Same vulnerability profile as A+D alone — Cluster E's topographic stability doesn't compensate for Cluster D's CA3 bistability.

## Per-phase mean

| Condition | P0 | P1 | P2 | P3 |
|---|---|---|---|---|
| baseline | **1.30** | **1.56** | 2.32 | 1.58 |
| A+D+E | 1.99 | 2.42 | **2.00** | 2.61 |

A+D+E is **worse on P0 and P1** (initial learning + first transition) and slightly better on P2 (mid-task), worse on P3 (cumulative drift). The pattern is clear: **adding Cluster D upstream of the BG cascade introduces noise that the BG has to learn to ignore**, slowing initial acquisition. Cluster E's topographic stability helps mid-task but doesn't compensate for the upstream noise.

## Implication for cheat-5 strategy

**A+E remains the operational best (6.97 ± 0.83 multi-goal det, n=6).** All cluster-stacking attempts so far have FAILED to improve on A+E:

| Combo | Sum | Std | Verdict |
|---|---|---|---|
| baseline | 7.36-7.64 | 2.42-3.30 | reference |
| **A+E** | **6.97** | **0.83** | **operational BEST** |
| A+D | 8.67 | 5.13 | NEUTRAL (seed-100 vulnerability) |
| A+D+E | 9.03 | 4.12 | NEUTRAL/worse (same seed-100 pattern) |
| B.3+C v1 | 16.50 | 3.97 | NEGATIVE — phase-0 broken |

**Three remaining options:**

1. **Cluster D v2 (SWR replay)** — implement the deferred v2 from cluster D design doc. The hypothesis is that without SWR-driven offline cleanup, the CA3 recurrent autoassociator drifts toward bad attractors. SWR would refresh CA3 patterns during quiet rest, preventing the seed-100 catastrophe. Multi-day code work.

2. **Cluster F (cerebellum)** — biggest unbuilt cluster, 73× citations to Marr/Albus/Hesslow. Independent of BG/hippocampus; adds closed-loop motor learning machinery. Multi-day buildout.

3. **Accept A+E as the operational ceiling** for cheat-5 and pivot to other research directions:
   - Tier-2 cleanup of A+E (refine cortex_to_msn_density, lateral inhibition strength, etc.)
   - Cross-task generalization (does A+E work on tasks beyond multi-goal navigation?)
   - 3D environment / continuous-action extensions

## Key takeaway

The "more clusters = better cheat-5" hypothesis is now empirically falsified. Three cluster-stacking attempts (A+E, A+D, A+D+E, plus B.3+C v1) show that beyond A+E, additional clusters add variance without benefit. The seed-100 vulnerability is robust across A+D and A+D+E — confirming it's a real Cluster D issue, not random.

**Cheat-5 closure requires either (a) Cluster D v2 (SWR) or (b) a different cluster combination than the BG-loop + hippocampus stack.** A+E shipped as the new flagship operational point until either approach validates.

## Provenance

- Code SHA at run time: `9d33349` (after RUN HUD v2 commit; before v3).
- Wall-clock: ~58 min for 12 parallel runs.
- All 12 result JSONs at `research/findings/raw/g11_bg/g11_seed{42,43,44,100,101,102}_ADE_test_{baseline,ADE}.json`.
- Acid test passes again: this baseline 6.76 ± 2.42 is consistent with prior baselines (Step 4: 7.64 ± 3.30; A+D test: 7.36 ± 2.47). All 20 Wave-1+2/3 renames remain behaviorally neutral.
