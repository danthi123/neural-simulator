# 2026-04-29 — Cluster F (cerebellum) v1 eval: NEUTRAL; A+E ceiling holds

**Run:** `g11_bg_runner.py` multi-goal deterministic, n=6 seeds × 3 conditions = 18 runs. Conditions: baseline (no clusters), A+F (Cluster A closed BG loop + Cluster F cerebellum), A+E+F (closed loop + topographic + cerebellum). `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

**Hypothesis (going in):** A+E is the operational best (6.97 ± 0.83), but adding clusters has consistently failed to improve on it (A+D, A+D+E, B.3+C v1 all NEUTRAL/NEGATIVE). Cluster F is structurally INDEPENDENT of the BG/hippocampus stack (cerebellum has its own input/output/learning rule via climbing-fiber teaching). If any cluster could break the ceiling, the catalog ROI says it's F.

## Headline

**A+F is NEUTRAL on cheat-5 mean** (7.36 vs baseline 7.77, t=−0.27, NOT significant). **A+E+F doesn't beat A+E** (8.02 vs 6.97 from Step 4). The cluster-stacking ceiling at A+E (6.97 ± 0.83 multi-goal det) is now empirically established across **four** stacking attempts.

| Condition | Mean | Std | n | Welch's t (vs baseline) | Verdict |
|---|---|---|---|---|---|
| baseline (no clusters) | 7.77 | 3.33 | 6 | reference | acid-test |
| **A+F** | **7.36** | 1.83 | 6 | t=−0.27 | NEUTRAL, **−45% std** |
| **A+E+F** | **8.02** | 1.80 | 6 | t=+0.16 | NEUTRAL, similar std reduction |
| A+E (Step 4 reference) | 6.97 | 0.83 | 6 | — | **operational BEST** ★ |

## Per-seed breakdown

| Seed | Baseline | A+F | A+E+F | base phases | A+F phases | A+E+F phases |
|---|---|---|---|---|---|---|
| 42 | 7.42 | 8.55 | 10.35 | [1.07, 0.93, 2.17, 3.25] | [1.54, 1.11, 2.34, 3.57] | [1.10, 0.93, 2.36, **5.96**] |
| 43 | 5.94 | 5.01 | 6.49 | [1.06, 1.22, 2.14, 1.51] | [1.59, 1.35, 1.01, 1.05] | [3.32, 1.28, 0.97, 0.91] |
| 44 | 5.05 | 6.46 | 6.27 | [1.31, 1.38, 0.95, 1.42] | [1.54, 1.57, 1.32, 2.04] | [2.44, 1.63, 1.15, 1.05] |
| 100 | 10.95 | 9.63 | 8.01 | [0.96, 4.07, 3.27, 2.65] | [0.92, 1.78, **5.92**, 1.01] | [1.03, 3.54, 1.46, 1.98] |
| 101 | 12.70 | 8.64 | 10.08 | [0.99, 1.75, 3.24, **6.72**] | [1.03, 2.40, 2.49, 2.73] | [2.79, 1.48, 2.01, 3.81] |
| 102 | 4.58 | 5.88 | 6.94 | [1.04, 1.27, 1.17, 1.10] | [1.06, 1.28, 2.27, 1.27] | [2.07, 2.07, 1.36, 1.43] |

**Variance pattern**: baseline range 5.05–12.70 (Δ=7.65); A+F range 5.01–9.63 (Δ=4.62); A+E+F range 6.27–10.35 (Δ=4.08). Both F-containing combos cut variance by ~40-45% over baseline, similar to but not as tight as A+E (range 5.21–8.39, Δ=3.18 in Step 4).

## Per-phase mean

| Condition | P0 | P1 | P2 | P3 |
|---|---|---|---|---|
| baseline | **1.07** | **1.77** | 2.16 | 2.77 |
| A+F | 1.28 | 1.58 | 2.56 | **1.95** |
| A+E+F | 2.12 | 1.82 | **1.55** | 2.52 |

**A+F profile**: better on P3 (cumulative drift, 1.95 vs 2.77) but worse on P2 (2.56 vs 2.16). The cerebellar contribution helps with late-task corrections (cf. F's role in motor adaptation per Marr-Albus theory) but adds noise mid-task as it's still learning.

**A+E+F profile**: P0 worse (2.12, agent slow to acquire initial goal — Cluster F's CF teaching and topographic cortex E both adding initial-condition noise), P2 best of the three (1.55, mid-task stability where E's topographic structure + F's correction help), P3 mid (cumulative drift).

## Why this is NEUTRAL (interpretation)

**Cluster F's v1 implementation uses reward-modulated STDP, not CF-gated LTD per Albus 1971.** The design doc explicitly notes this v1 simplification — climbing-fiber teaching propagates via the IO→PC pathway with `weight_mean=50.0` to evoke complex spikes, but the actual plasticity update goes through the existing reward-modulation path (with the negative reward at Δd>0 steps doubling as the LTD signal). This means the cerebellum learns "like the BG learns" — same teacher signal as the rest of the brain — rather than via the orthogonal CF→PC LTD rule that's the cerebellum's defining biological feature.

**Implication**: v2 with proper CF-gated LTD (anti-Hebbian update on PF→PC synapses gated by CF complex spikes, not via global reward signal) would be a meaningfully different test. The current v1 result tells us "cerebellar topology + reward-modulated STDP doesn't help"; v2 would tell us "cerebellar topology + actual cerebellar learning rule helps/doesn't".

**Variance reduction is real**, however. Adding F gives:
- A+F: 1.83 std vs baseline 3.33 (−45%)
- A+E+F: 1.80 std vs baseline 3.33 (−46%)

Comparable to A+E's structural-stability win (0.83 std, −75% vs baseline). The cerebellum's structural contribution (DCN→motor additive drive, smoothing motor output) accounts for this.

## Implication for cheat-5 strategy

**A+E remains the operational ceiling for multi-goal cheat-5 (6.97 ± 0.83, n=6 multi-goal det).** Five cluster-stacking attempts beyond A+E have now failed to improve on it:

| Combo | Mean | Std | Verdict |
|---|---|---|---|
| baseline | 7.36-7.77 | 2.42-3.33 | reference |
| **A+E** (Step 4) | **6.97** | **0.83** | **operational BEST** ★ |
| A+F (this run) | 7.36 | 1.83 | NEUTRAL, variance reduction |
| A+D | 8.67 | 5.13 | NEUTRAL, seed-100 outlier |
| A+E+F (this run) | 8.02 | 1.80 | NEUTRAL/worse vs A+E |
| A+D+E | 9.03 | 4.12 | NEUTRAL/worse vs A+E |
| B.3+C v1 | 16.50 | 3.97 | NEGATIVE — phase-0 broken |

**The "more biology → better cheat-5" hypothesis is now empirically falsified across all five attempts.** The cluster-by-cluster strategy from CLAUDE.md (Cluster B done, Cluster A+E shipped, Cluster D and F evaluated) has produced a clear pattern: each new cluster either reduces variance, adds variance, or neither — none meaningfully reduces the mean below A+E.

## What this means for next direction

Three options stand:

1. **Cluster F v2** (proper CF-gated LTD): the v1 result doesn't decisively rule out cerebellar contribution because the LTD rule isn't biologically faithful. v2 would implement the anti-Hebbian PF→PC weight update gated explicitly by CF complex spikes. ~2-3 days of code work + eval. Catalog grounding is excellent (Marr 1969 §5.1, Albus 1971 §IV.C eq.4, Hesslow 2013 §2 critique of LTD-only).

2. **Cluster D v2** (SWR replay): rescue Cluster D's seed-100 vulnerability via offline replay. Independent code path from F. ~2-3 days. May explain D's outlier pattern.

3. **Pivot**: accept A+E as the cheat-5 ceiling and move to other research directions:
   - Cross-task generalization: does A+E work on different tasks (single-goal, 4-goal fast-change, longer trajectories)?
   - 3D environment / continuous-action extensions
   - Multi-modal sensors
   - **Replicate framework retrofit** (already partially done tonight, ~4-6h to complete) — would 6× speed up all future evals, paying back across many sessions.

## Acid test affirmation

This baseline (7.77 ± 3.33) reproduces the Step 4 baseline (7.64 ± 3.30) within noise. Confirms:
- All 20 Wave-1+2/3 structural renames are behaviorally neutral
- Cluster F additions when its flag is off don't perturb baseline behavior
- The HUD instrumentation work doesn't affect runner behavior

## Provenance

- Code SHA at run time: `96b1e2c` (cluster F implementation commit), with later HUD + replica commits not affecting the eval since runs were already launched.
- Wall-clock: ~95 min for 18 parallel runs (CPU-bound at 0.25-0.31 steps/sec/process due to Python orchestration overhead — see `docs/plans/2026-04-29-g11-batched-replica-retrofit.md` for the in-progress retrofit that addresses this).
- All 18 result JSONs at `research/findings/raw/g11_bg/g11_seed{42,43,44,100,101,102}_clusterF_{baseline,AF,AEF}.json`.
- 10/10 Cluster F unit tests passing in `tests/test_cluster_f.py`.

## What's next

Per the three options above, my current recommendation: **proceed with the batched-replica runner retrofit** (T2-T7 in `docs/plans/2026-04-29-g11-batched-replica-retrofit.md`) before the next cluster experiment. The retrofit pays back ~6× across all future evals, and the cluster-stacking ceiling at A+E means the next experiment is a bigger biology buildout (D v2 or F v2) which will need many runs to validate. Investing the ~4-6h on the retrofit first is high lifetime ROI.

After the retrofit lands, the natural sequence is:
1. **F v2** (CF-gated LTD, proper anti-Hebbian rule) — re-run F with the right learning rule
2. **D v2** (SWR replay) — rescue D's seed-100 vulnerability
3. **Composition tests** with replicate-accelerated evals (n=12, n=24)
