# Cluster B.1 — D1/D2 Plasticity Asymmetry: PARTIAL SIGNAL

**Date:** 2026-04-28 (evening)
**Status:** B.1 implementation complete, biology probe PASSED, cheat-5 multi-goal re-eval shows **partial signal** — variance halved + Phase 2 catastrophe eliminated, but mean still 7% above v3 baseline. Cluster B continues to B.2 (FSIs).
**Plan:** [`docs/plans/2026-04-28-cluster-b1-d1d2-asymmetry-implementation.md`](../../docs/plans/2026-04-28-cluster-b1-d1d2-asymmetry-implementation.md)
**Cluster context:** [`docs/plans/2026-04-28-cluster-b-striatal-microcircuit-design.md`](../../docs/plans/2026-04-28-cluster-b-striatal-microcircuit-design.md)
**Reframe context:** [`2026-04-28-cheat5-post-v4-reframe.md`](2026-04-28-cheat5-post-v4-reframe.md)

## TL;DR

D1/D2 plasticity asymmetry (Cluster B.1, the smallest piece of the striatal microcircuit cluster) **partially stabilizes** patch-matrix cross-projections under multi-goal evaluation. Variance drops 52% (std 2.54 → 1.23), the Phase 2 catastrophic-failure mode is eliminated (P2 mean 3.36 → 1.92), but mean cheat-5 sum is still 7% worse than v3 baseline (7.62 vs 7.08).

This is the **first cluster-buildout signal we've measured.** The cluster strategy — adding biology systematically to scaffold cross-projection refinement — is empirically supported. Continuing to Cluster B.2 (striatal FSIs) and B.3 (cholinergic interneurons / TANs) as planned.

## Implementation summary

8 commits across 4 days of focused work:

| # | Commit | What |
|---|---|---|
| Task 1 | `69b0480` | `enable_d1_d2_asymmetry` config field + `cp_d1_d2_sign` GPU array allocation + D2-targeting synapse tagging |
| Task 2 | `14c82ec` | Single-line plasticity rule integration at sim/bridge.py:4309 — multiplicative sign factor on the reward-modulated weight update |
| Task 3 | `1155fc1` | `--enable-d1-d2-asymmetry` CLI flag + kwarg plumbing through run_moving_goal_episode |
| Task 4 | `2fc8975` | Standalone biology probe at `research/probes/d1_d2_asymmetry_probe.py` |

### Test coverage

- 6 new unit tests in `tests/test_d1_d2_asymmetry.py` — all pass
- 1 new test in `tests/test_g11_bg_runner_flags.py` — kwarg acceptance
- 73 regression tests in `test_regions.py + test_neuromodulators.py + test_structural_pruning.py` — no regressions
- 33 g11_bg_runner_flags tests — no regressions

### Biology validation (`research/probes/d1_d2_asymmetry_probe.py`)

```
=== D1/D2 Plasticity Asymmetry Biology Probe ===
Phase 1: 50 steps with reward = +1.0
  D1 synapses (N=1000): mean dw=+0.49953  ← expected +
  D2 synapses (N=1000): mean dw=-0.49953  ← expected -
  Other synapses (N=4572): mean dw=+0.49950 ← expected +
Phase 2: 50 steps with reward = -1.0
  D1 synapses: mean dw=-0.49953           ← expected -
  D2 synapses: mean dw=+0.49953           ← expected +
  Other synapses: mean dw=-0.49950        ← expected -
VERDICT: PASS - asymmetry verified
```

Math: `lr × reward × eligibility × sign × n_steps = 0.01 × 1 × 1 × (±1) × 50 = ±0.5`. Observed ±0.49953 (small eligibility-decay correction). Asymmetry signal is crisp; std ~1e-5 across 1000 synapses per class.

## Cheat-5 multi-goal re-eval (Task 5)

### 5a — v3 + B.1 baseline (no cross-projections)

Verifying B.1 doesn't regress flagship behavior.

| Seed | Sum | per-phase |
|---|---|---|
| 42 | 7.59 | 2.63, 1.70, 1.58, 1.68 |
| 43 | 6.77 | 1.90, 1.68, 1.82, 1.36 |
| 44 | 6.64 | 2.09, 1.67, 1.48, 1.40 |
| **mean** | **7.00 ± 0.52** | 2.21, 1.68, 1.63, 1.48 |

vs v3 baseline 7.08 ± 0.12. Difference: **-0.08 (within noise)**. **PASSED non-regression** criterion.

Note: variance increased slightly (0.12 → 0.52). B.1 introduces seed-dependent variance even without cross-projections. Watch this in future evaluations.

### 5b — patch-matrix + B.1 (the cheat-5 signal test)

Same flag set as the patch-matrix-alone n=3 result (cross-projection density 0.25, topology seed 0, multi-goal), with `--enable-d1-d2-asymmetry` added.

| Seed | Sum | P0 | P1 | P2 | P3 |
|---|---|---|---|---|---|
| 42 | 6.82 | 1.55 | 1.88 | 1.75 | 1.65 |
| 43 | 9.04 | 1.96 | 2.96 | 2.76 | 1.35 |
| 44 | 6.99 | 1.93 | 2.43 | 1.24 | 1.39 |
| **mean** | **7.62 ± 1.23** | 1.81 | 2.42 | 1.92 | 1.46 |

### Multi-goal comparison table

| Variant | Mean | Std | P2 mean | P2 std |
|---|---|---|---|---|
| v3 baseline (no cross, no B.1) | 7.08 | 0.12 | 1.90 | (low) |
| v3 + B.1 (no cross) | 7.00 | 0.52 | 1.63 | 0.18 |
| **patch-matrix alone (no B.1)** | **8.76** | **2.54** | **3.36** | **2.09** |
| **patch-matrix + B.1** | **7.62** | **1.23** | **1.92** | **0.77** |

### What B.1 does to patch-matrix

1. **Variance reduction (52%):** std 2.54 → 1.23. The "topology luck" signature is being damped.
2. **Phase 2 catastrophe eliminated:** P2 mean 3.36 → 1.92, P2 std 2.09 → 0.77. The (1,6)→(1,1) failure that hit seed 42 at 10.65 in patch-matrix-alone is gone.
3. **Per-seed clustering tighter:** patch-matrix-alone (10.65, 9.77, 5.88) vs +B.1 (6.82, 9.04, 6.99).
4. **Mean still above baseline:** 7.62 vs 7.08 = +0.54. Cheat-5 is NOT fully closed by B.1 alone.

### Decision-matrix outcome

Per the design doc:

| Result | Verdict |
|---|---|
| Mean ≤ 7.0 + std < 1.0 | **first real cheat-5 partial closure signal** |
| Mean ~8 (close to patch-matrix-alone 8.76) | **no improvement** |
| Mean > 10 | **B.1 hurt with cross-projections** |

We're at **mean 7.62, std 1.23**. Between the GO and "no improvement" bands — cleaner than expected at the lower bound but std exceeds 1.0. **Marginal partial signal**, not full closure.

**Action per the design doc:** proceed to B.2 (FSIs) and B.3 (TANs) — Cluster B is a unit and we expected partial signal at B.1 alone.

## Why this is significant

This is the **first piece of empirical evidence supporting the cluster-buildout strategy.** Before today, all cheat-5 attempts had either:
- Failed catastrophically (option 1 / structural pruning at 22.46)
- Shown high-variance partial signal but no consistent improvement (patch-matrix at 8.76 ± 2.54)

B.1 alone closes the gap noticeably:
- Variance halved
- Worst-case phase eliminated
- Mean within 7% of baseline (vs 24% off for patch-matrix-alone)

The biology in B.1 (D1 LTPs under +DA, D2 inverts) is well-understood real biology with no controversy. It's not a hack to make cheat-5 numbers look better — it's a known missing piece. And it shifted the data the way the cluster hypothesis predicted.

If B.2 (FSIs) and B.3 (TANs) compound similarly, Cluster B as a whole could close cheat-5. We won't know until they're implemented, but the trajectory is encouraging.

## Caveats

1. **n=3 only.** Single topology seed, single eval-seed range (42/43/44). Need 6-seed validation before any "GO" conclusions. Holding off on full validation until full Cluster B is implemented.
2. **Variance increased on the B.1-no-cross baseline** (0.12 → 0.52). Worth understanding — B.1 changes plasticity dynamics on same-action paths too, since same-action D2-targeting synapses also get the inverted sign. May be a wash, may indicate a subtle issue. Will track across the cluster.
3. **Pre-textbook implementation.** B.1 was implemented before the textbook catalog session shipped Section IV citations. Implementation is grounded in training-data knowledge of canonical Shen et al. 2008 / Kreitzer & Malenka 2008 framework. Citations + any refinements will be backfilled when the catalog lands.
4. **Cluster B is a unit.** B.1 alone giving partial signal doesn't mean "ship B.1 to flagship." We continue building the cluster; flagship updates wait for full Cluster B validation.

## Next steps

1. **Cluster B.2 — striatal FSIs.** Design doc + implementation plan + TDD code. The FSI cluster adds a fast-spiking interneuron population with broadcast inhibition. ~3-4 days estimated.
2. **Cluster B.3 — cholinergic interneurons / TANs.** Plasticity-gating windows via ACh dynamics. ~3-4 days.
3. **Full Cluster B re-eval.** v3 + B.1 + B.2 + B.3 (no cross) baseline + patch-matrix + full Cluster B at n=3 then n=6.
4. **Findings + propagation per cluster** — same template as this doc.

## Files

- Implementation: `sim/config.py`, `sim/bridge.py`, `research/runners/g11_bg_runner.py`
- Tests: `tests/test_d1_d2_asymmetry.py`, `tests/test_g11_bg_runner_flags.py::test_d1_d2_asymmetry_kwarg_accepted`
- Biology probe: `research/probes/d1_d2_asymmetry_probe.py`
- Probe output: `research/findings/raw/d1_d2_probe/probe_results.json`
- Cheat-5 5a result JSONs: `research/findings/raw/g11_bg/g11_seed{42,43,44}_flagship_{be48d5,7484d9,031201}.json`
- Cheat-5 5b result JSONs: `research/findings/raw/g11_bg/g11_seed{42,43,44}_flagship_{e488bb,9ccee2,efbb36}.json`

## Updates propagated

- [x] CLAUDE.md "Cheat #5 progress" section updated with B.1 partial signal
- [x] docs/SCIENCE_ROADMAP.md §4.7: B.1 row added, cluster trajectory note
- [x] research/findings/INDEX.md: B.1 finding row
- [x] CHANGELOG.md 2026-04-28 section: B.1 result entry
- [x] Memory: project_cheat5_v3_results.md + MEMORY.md index updated
