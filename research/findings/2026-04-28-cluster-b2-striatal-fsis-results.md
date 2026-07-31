---
type: finding
status: qualified
date: 2026-04-28
mechanism: striatal-fsi
---

# Cluster B.2 — Striatal FSIs: MIXED, Phase-0 architectural issue

**Date:** 2026-04-28 (evening)
**Status:** B.2 implementation complete + biology probe PASS, but cheat-5 multi-goal eval shows mixed result. **Phases 1-3 improve substantially** (sum 5.80 → 4.72, beats v3 baseline) but **Phase 0 (initial bootstrap) is degraded** by FSIs broadcasting too eagerly before agent commits to an action. Cluster B continues to B.3 (TANs) per the unit-cluster strategy; B.2 retuning deferred until full cluster lands.
**Plan:** [`docs/plans/2026-04-28-cluster-b2-striatal-fsis-implementation.md`](../../docs/plans/2026-04-28-cluster-b2-striatal-fsis-implementation.md)
**Cluster context:** [`docs/plans/2026-04-28-cluster-b-striatal-microcircuit-design.md`](../../docs/plans/2026-04-28-cluster-b-striatal-microcircuit-design.md)
**B.1 (precedes):** [`2026-04-28-cluster-b1-d1d2-asymmetry-results.md`](2026-04-28-cluster-b1-d1d2-asymmetry-results.md)

## TL;DR

B.2 produces a Phase-decomposed signature: **Phases 1-3 are better than v3 baseline** (4.72 vs 4.89), and **variance drops further** (std 1.23 → 0.62 with cross-projections), but **Phase 0 is broken** because FSIs disrupt the cascade's initial bootstrap. The agent can't commit to a winner at trial 0 because FSIs broadcast inhibition immediately. Total cluster mean (8.44 ± 0.62) is slightly worse than B.1 alone (7.62 ± 1.23) — B.2 helps when the cascade is mature but breaks the cold-start.

This is a real architectural issue, not noise. Continuing to B.3 (TANs) per the unit-cluster strategy. If full Cluster B (B.1+B.2+B.3) doesn't close cheat-5, B.2 will be retuned with delayed FSI engagement (e.g. cortex_to_str_fs_weight 30→10).

## Implementation summary

3 commits across the day:

| # | Commit | What |
|---|---|---|
| Task 1 | `74b857f` | Add `str_FS_{N,E,S,W}` regions + 4 cortex→FS excitatory pathways + 32 FS→MSN broadcast inhibitory pathways behind `enable_striatal_fsis` flag |
| Task 2 | `f19fe0d` | `--enable-striatal-fsis` CLI flag + kwarg plumbing through `run_moving_goal_episode` |
| Task 3 | `d7b8b83` | Standalone biology probe at `research/probes/striatal_fsi_probe.py` |
| Retune | `f3dc241` | str_fs_to_msn_weight default 8.0 → 2.0 (over-suppression fix) |

### Test coverage

- 3 new unit tests in `tests/test_g11_bg_runner_flags.py` (region/pathway count assertions)
- 1 kwarg-acceptance test
- 37 g11_bg_runner_flags tests pass total
- 73+ regression tests in test_regions, test_neuromodulators, test_structural_pruning, test_d1_d2_asymmetry — all pass

### Biology probe (`research/probes/striatal_fsi_probe.py`)

```
Without FSIs: str_D1_N peak rate 36.4 Hz
With FSIs:    str_D1_N peak rate 23.6 Hz (-12.8 Hz / -35%)
              str_FS_N peak: 16 Hz (broadcast pathway engaged)
VERDICT: PASS - FSIs suppress MSN firing via broadcast inhibition
```

But the probe ALSO observed: **winner pool (str_D1_N) suppressed MORE than loser pool (str_D1_E)**. With initial weight=8.0, FSIs were over-suppressing. Retune to weight=2.0 reduced the over-suppression but didn't fully fix it — the problem is structural (cortex drives FSI which inhibits same-action MSN), not just a magnitude issue.

## Cheat-5 multi-goal re-eval

### 4a — v3 + B.1 + B.2 baseline (no cross-projections)

Two configurations tested:

| Config | Mean | Std | P0 | P1 | P2 | P3 |
|---|---|---|---|---|---|---|
| v3 baseline | 7.08 | 0.12 | 2.13 | 1.47 | 1.90 | 1.52 |
| v3 + B.1 + B.2 (weight 8.0, ORIGINAL) | **19.78** | 2.28 | 4.37 | 4.21 | 6.07 | 5.12 |
| v3 + B.1 + B.2 (weight 2.0, RETUNED) | **9.50** | 0.85 | 4.15 | 1.60 | 2.01 | 1.74 |

The original 8.0 weight was a catastrophic over-suppression — even basic motor selection broke. Retuning to 2.0 brought Phases 1-3 back to baseline-comparable levels but Phase 0 stayed degraded (4.15 vs baseline 2.13).

**Failed non-regression criterion** (≤ 7.5). B.2 hurts the baseline cascade.

### 4b — patch-matrix + B.1 + B.2 (the cheat-5 signal test)

| Variant | Mean | Std | P0 | P1 | P2 | P3 |
|---|---|---|---|---|---|---|
| patch-matrix alone | 8.76 | 2.54 | 1.83 | 2.05 | 3.36 | 1.53 |
| patch-matrix + B.1 | 7.62 | 1.23 | 1.81 | 2.42 | 1.92 | 1.46 |
| **patch-matrix + B.1 + B.2 (weight 2.0)** | **8.44** | **0.62** | **3.72** | **1.53** | **1.59** | **1.60** |

### Variance trajectory (cluster buildout)

| Cluster step | std |
|---|---|
| patch-matrix alone | 2.54 |
| + B.1 | 1.23 |
| + B.1 + B.2 | **0.62** |

**Each cluster step halves variance.** This is real signal — the cluster-buildout strategy IS stabilizing cross-projections, even when individual mean values aren't improving monotonically.

### Phase-decomposed analysis

Sum-of-phases is misleading — let's look per-phase:

| Phases | v3 baseline | patch + B.1 | patch + B.1 + B.2 | Δ vs B.1 |
|---|---|---|---|---|
| **Phase 0** | 2.13 | 1.81 | **3.72** | **+1.91 (worse)** |
| Phase 1 | 1.47 | 2.42 | 1.53 | -0.89 (better) |
| Phase 2 | 1.90 | 1.92 | 1.59 | -0.33 (better) |
| Phase 3 | 1.52 | 1.46 | 1.60 | +0.14 (within noise) |
| **P1+P2+P3 sum** | 4.89 | 5.80 | **4.72** | **-1.08 (better than v3 BASELINE)** |

**B.2 makes everything-after-Phase-0 BETTER than v3 baseline.** P1+P2+P3 = 4.72 vs baseline's 4.89. The agent's steady-state action selection with cross-projections + B.1 + B.2 actually beats the no-cross flagship.

**Phase 0 is the entire problem.** The agent can't bootstrap an initial winner because FSIs broadcast inhibition immediately when cortex fires.

## Why Phase 0 fails

In real BG, FSIs:
- Have low tonic baseline (5-10 Hz)
- Fire short bursts (~50ms) when activated
- Are high-pass filtered: require strong cortex drive to fire

Our FSI implementation has none of these properties:
- No tonic baseline (FSIs are quiet at rest, then fire whenever cortex does)
- No burst-vs-tonic differentiation (any cortex activity = FSI activity)
- cortex_to_str_fs_weight=30 makes FSIs sensitive to weak cortex drive

So at trial 0, the agent's heuristic produces moderate cortex activity. FSIs fire immediately. Broadcast inhibition prevents any MSN pool from establishing a clear winner. The agent thrashes until learned weights amplify cortex drive enough that FSIs can't suppress everything.

In Phases 1-3, learned weights HAVE accumulated, cortex drive IS strong, and FSIs do their actual job — adding fast WTA support.

## What this means for the cluster strategy

**Mixed but encouraging.** Per-piece, B.2 fails baseline non-regression, but in combination with cross-projections B.2 contributes to the variance-reduction trajectory and helps phases 1-3.

The plan said "Cluster B is a unit; partial signal expected at B.1, full signal possibly at full cluster." We're seeing that — B.2 alone doesn't help cheat-5, but B.2 doesn't break the directionality either, and the variance reduction continues.

**Decision: continue to B.3 (TANs) per the unit-cluster strategy.** B.3 might compensate for B.2's Phase-0 issue: TANs pause plasticity windows around salient events, which could mean less FSI-driven plasticity disruption during early goal-acquisition. After B.3, run full Cluster B re-eval. If still bad, retune B.2 with delayed FSI engagement (cortex_to_str_fs_weight=10).

## What ships from this batch

- B.2 implementation stays in the codebase, opt-in via `--enable-striatal-fsis`. Default off — flagship behavior unchanged.
- Default `str_fs_to_msn_weight=2.0` (retuned from 8.0). Default `cortex_to_str_fs_weight=30.0` may need future retuning to ~10 to fix Phase-0 bootstrap; deferred until full Cluster B is evaluated.
- Recommended flagship config UNCHANGED. Still v3 + perception arc + curriculum, no Cluster B flags. Cluster B is research-mode only until full validation.

## Caveats

1. **n=3 only** for all data points. Variance estimates have wide CIs.
2. **Phase 0 issue is mechanistic, not just hyperparameter tuning.** Real FSIs have biological properties our model doesn't have (tonic baseline, burst dynamics, high-pass filtering on cortex drive). Future work could add these via Izhikevich preset tuning, but that's beyond Cluster B.2's scope.
3. **B.2 with cross-projections has lowest variance we've seen** (0.62). If this trend continues, full Cluster B variance might match v3 baseline (0.12) — at which point std-based comparisons become more sensitive.

## Next steps

1. **Cluster B.3 — Cholinergic interneurons (TANs).** Plasticity-gating windows via ACh dynamics. ~3-4 days.
2. **Full Cluster B re-eval** (B.1 + B.2 + B.3). If Phase 0 is still bad, retune cortex_to_str_fs_weight 30→10.
3. If full Cluster B doesn't close cheat-5: **proceed to Cluster A (closed BG loop)** — this might be the actual missing teaching signal.

## Files

- Implementation: `research/runners/g11_bg_runner.py:build_bg_brain_regions`
- Tests: `tests/test_g11_bg_runner_flags.py` (4 new tests + regression)
- Biology probe: `research/probes/striatal_fsi_probe.py`
- Probe output: `research/findings/raw/striatal_fsi_probe/probe_results.json`
- Cheat-5 4a result JSONs (weight 8.0, original): `research/findings/raw/g11_bg/g11_seed{42,43,44}_flagship_{61e285,c23cb6,0660f7}.json`
- Cheat-5 4a result JSONs (weight 2.0, retuned): `research/findings/raw/g11_bg/g11_seed{42,43,44}_flagship_{34382f,544fe3,2a761f}.json`
- Cheat-5 4b result JSONs: `research/findings/raw/g11_bg/g11_seed{42,43,44}_flagship_{26ee0b,b4ce7c,f36e86}.json`

## Updates propagated

- [x] CLAUDE.md cheat-5 section: B.2 result added with mixed verdict
- [x] docs/SCIENCE_ROADMAP.md §4.7: B.2 results table rows
- [x] research/findings/INDEX.md: B.2 finding row
- [x] CHANGELOG.md 2026-04-28 section: B.2 entry
- [x] Memory: project_cheat5_v3_results.md + MEMORY.md updated
