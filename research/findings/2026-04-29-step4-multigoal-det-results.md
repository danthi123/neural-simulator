# 2026-04-29 — Step 4 results: A+E multi-goal deterministic (n=12)

**Run:** `g11_bg_runner.py --moving-goal --bg-lateral-inhibition --enable-d1-d2-asymmetry --enable-striatal-fsis [+ --enable-cluster-a-closed-loop --enable-cluster-e-topography for AE] --goal-schedule multi --deterministic`. 6 seeds (42, 43, 44, 100, 101, 102) × 2 conditions (baseline vs A+E) = 12 runs. `CUBLAS_WORKSPACE_CONFIG=:4096:8` set via `--deterministic` runner flag (env propagation before cupy import).

## Headline

**Variance reduction confirmed; mean improvement is small and not statistically significant on multi-goal.**

| Condition | Mean | Std | n | Welch's t vs baseline |
|---|---|---|---|---|
| **baseline** (no A+E) | **7.64** | 3.30 | 6 | reference |
| **A+E** | **6.97** | 0.83 | 6 | t=−0.49 (NOT sig) |
| Delta | −0.68 (−8.9%) | **−75% std** ★ | | |

## What this changes vs prior understanding

| Prior finding | Source | Status |
|---|---|---|
| A+E variance reduction is real | overnight FINAL (multi, non-det, n=6) | ✓ confirmed under determinism |
| A+E mean improvement on multi-goal is within noise | overnight FINAL | ✓ confirmed |
| A+E single-goal det mean improves 10% (n=12) | tier-4 expanded | ✓ same effect-size in multi-goal (8.9%) |
| Determinism flag tightens noise floor | new infra | ✓ A+E std halved further (1.76 → 0.83) |

## Per-seed breakdown

| Seed | Baseline sum | A+E sum | Delta | Baseline phases (P0,P1,P2,P3) | A+E phases |
|---|---|---|---|---|---|
| 42 | 13.80 | **6.34** | **−7.46** | [2.71, 0.99, 4.81, 5.29] | [1.39, 0.99, 1.38, 2.58] |
| 43 | 5.20 | 7.66 | +2.46 | [1.54, 1.65, 1.13, 0.88] | [2.19, 2.96, 1.15, 1.37] |
| 44 | 5.58 | 6.24 | +0.66 | [1.95, 1.73, 0.92, 0.98] | [1.24, 2.22, 1.39, 1.39] |
| 100 | 6.95 | 8.15 | +1.20 | [1.32, 3.61, 1.03, 0.99] | [1.15, 4.39, 1.52, 1.09] |
| 101 | 8.83 | 6.23 | **−2.60** | [1.19, 1.66, 1.81, 4.17] | [1.09, 2.34, 1.17, 1.64] |
| 102 | 5.51 | 7.19 | +1.67 | [0.97, 2.08, 1.33, 1.13] | [2.64, 1.66, 0.98, 1.9] |

**Range comparison:**
- Baseline: 5.20 to 13.80 (8.60 range, 165% of mean)
- A+E: 6.23 to 8.15 (1.92 range, 28% of mean)

## Per-phase mean

| Condition | P0 | P1 | P2 | P3 | Notes |
|---|---|---|---|---|---|
| baseline | 1.61 | 1.95 | 1.84 | 2.24 | Phase 3 finalQ = 2.24 (drift after 3 transitions) |
| A+E | 1.62 | 2.43 | 1.27 | 1.66 | Phase 3 = 1.66 (better recovery), but P1 = 2.43 (worse than baseline 1.95) |

A+E is *worse* on phase 1 (the first transition recovery) but *better* on phase 3 (the cumulative-drift outcome). This matches the intuition that A+E (closed BG loop + topographic cortex) provides structural stability — it doesn't help the agent learn fast, but it helps the agent NOT degrade across multiple transitions.

## What this means for cheat-5

A+E **does not close cheat-5 on its own** in the multi-goal benchmark. Mean improvement is within noise (8.9%, t=−0.49). However:

1. **A+E provides a robust variance reduction** (75% std drop, 1.92 range vs 8.60 baseline range). For practical use, A+E gives **predictable** behavior across seeds.
2. **The "rogue seed" problem is solved.** Baseline at seed 42 hit 13.80 (catastrophic phase-2 + phase-3 drift); A+E at the same seed got 6.34 (recovered).
3. **Cheat-5 closure remains an open problem.** Per the post-v4 reframe, cheat-5 needs the FULL biology buildout (more clusters), not just A+E.

## Comparison to prior multi-goal results

| Date | Recipe | n | Mean | Std | Notes |
|---|---|---|---|---|---|
| 2026-04-28 | v3 (lateral inh + B.1+B.2, no A/E) | 6 | 7.41 | 3.67 | non-det |
| 2026-04-29 (overnight FINAL) | A+E | 6 | 7.28 | 1.76 | non-det |
| **2026-04-29 (THIS Step 4)** | **A+E det** | **6** | **6.97** | **0.83** | det, this run |
| 2026-04-29 (THIS Step 4) | det baseline (no A+E) | 6 | 7.64 | 3.30 | reference |

Determinism + A+E together give the lowest std (0.83) yet observed on multi-goal. This is a useful operational point: for any future research run that needs *reliable, comparable* multi-goal results, the recipe should be `--bg-lateral-inhibition --enable-d1-d2-asymmetry --enable-striatal-fsis --enable-cluster-a-closed-loop --enable-cluster-e-topography --deterministic`.

## Open questions

1. **Why is A+E P1 (after first transition) worse than baseline P1 (2.43 vs 1.95)?** Possible: the topographic cortex (E) takes longer to remap when the goal moves; baseline cortex stays plastic and re-adapts faster on the first hit.
2. **Does adding Cluster D (hippocampus trisynaptic pathway) on top close cheat-5?** Cluster D scaffolding shipped 2026-04-29 but not yet evaluated.
3. **Does adding Cluster C v1 (tonic DA) on top help?** C v1 scaffolding shipped 2026-04-29 but not yet evaluated.
4. **Combined A+C+D under multi-goal det:** not yet run — this is the natural next combination test.

## Provenance

- Code SHA at run time: `4ec7486` (post structural-naming-audit, pre Tier-1 prose pass `8aa2fcb`).
- Wall-clock: ~60 min per run, 12 runs in parallel batches → ~60 min total.
- All 12 result JSONs at `research/findings/raw/g11_bg/g11_seed{42,43,44,100,101,102}_step4_multidet_{baseline,AE}.json`.
- Step 3 + Step 4 chain script `step4_chain.sh` exited cleanly at 18:45:30.
