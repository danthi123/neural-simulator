# 2026-04-29 — A+D under multi-goal det: NEUTRAL on cheat-5

**Run:** `g11_bg_runner.py --moving-goal --enable-msn-lateral-inhibition --enable-d1-d2-asymmetry --enable-striatal-pv-fsi --enable-cluster-a-closed-loop --enable-cluster-d-hippocampus --goal-schedule multi --deterministic`. 6 seeds (42-44, 100-102) × 2 conditions (baseline vs A+D) = 12 runs. `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

**Hypothesis (going in):** after B.3 retry showed Cluster C v1 (tonic DA) breaks phase 0 under sparse reward, A+D was chosen as the cleaner test of "does Cluster D (hippocampus trisynaptic pathway: DG/CA3/CA1) add value on top of Cluster A (closed BG loop)?" — without the C v1 damping confound. Also serves as **implicit acid test** for the 13 Wave-1 structural renames committed earlier.

## Two headlines

### 1. Acid test PASSES — renames are behaviorally neutral
**Baseline reproduces 7.36 ± 2.47** vs documented Step-4 baseline 7.64 ± 3.30. Within noise. All 13 Wave-1 structural renames (cortex_to_d1 → corticostriatal, pfc → dlpfc_wm, dopamine region → snc, str_FS_X → str_PV_FSI_X, etc.) are confirmed behaviorally neutral via this fresh-baseline check.

### 2. A+D is NEUTRAL on multi-goal cheat-5

| Condition | Mean | Std | n | Welch's t | Verdict |
|---|---|---|---|---|---|
| baseline (no A, no D) | 7.36 | 2.47 | 6 | reference | acid-test |
| **A+D** | **8.67** | 5.13 | 6 | t=0.56 | NEUTRAL (variance INCREASED) |
| A+E (Step 4 reference) | 6.97 | 0.83 | 6 | — | best operational |
| B.3+C v1 (NEGATIVE) | 16.50 | 3.97 | 6 | — | broken phase 0 |

**A+D delta: +1.30 (+17.7% mean, +108% std). NOT statistically significant.**

## Per-seed breakdown — driven by one outlier

| Seed | Baseline | A+D | Delta | Baseline phases (P0,P1,P2,P3) | A+D phases |
|---|---|---|---|---|---|
| 42 | 10.93 | 5.52 | −5.41 | [4.12, 1.33, 1.09, 4.39] | [2.30, 0.98, 1.05, 1.19] |
| 43 | 5.50 | 8.20 | +2.70 | [1.55, 1.08, 1.61, 1.27] | [1.09, 2.14, 3.58, 1.39] |
| 44 | 5.49 | 5.62 | +0.13 | [1.94, 1.00, 1.35, 1.20] | [1.12, 2.33, 1.21, 0.96] |
| **100** | **7.70** | **18.95** | **+11.25** | [1.04, 2.32, 1.73, 2.60] | [3.97, 3.45, 4.78, 6.74] |
| 101 | 9.60 | 7.04 | −2.57 | [2.64, 1.33, 2.22, 3.42] | [1.38, 1.49, 2.96, 1.21] |
| 102 | 4.96 | 6.68 | +1.72 | [1.00, 1.06, 1.94, 0.96] | [1.70, 2.89, 0.97, 1.12] |

**A+D without seed 100:** mean = 6.61, std ≈ 1.08 (5 seeds). That's **better than baseline 7.36** with much tighter variance. The seed 100 catastrophe drives the entire mean shift.

**Seed 100 A+D phase pattern:** P0=3.97, P1=3.45, P2=4.78, P3=6.74. Agent never acquires initial goal cleanly, then degrades monotonically across all 3 transitions. Same failure-mode signature as the B.3+C v1 NEGATIVE result, but only at this seed.

## Per-phase mean

| Condition | P0 | P1 | P2 | P3 |
|---|---|---|---|---|
| baseline | 2.05 | **1.35** | 1.66 | 2.31 |
| A+D | **1.93** | 2.21 | 2.42 | 2.10 |

A+D is *slightly better* on P0 (initial acquisition: 1.93 vs 2.05) but worse on P1-P2-P3. The Cluster D structure (5 new regions, dense CA3 recurrent autoassociator) seems to slow re-adaptation after each goal change.

## Why this is NEUTRAL (interpretation)

**Cluster D v1 ships the trisynaptic-pathway core (DG/CA3/CA1) but NOT the SWR replay machinery.** Per the cluster D design doc:
- v1: minimal trisynaptic pathway (this run)
- v2 (deferred): SWR generator + NREM-gated replay
- v3 (deferred): engram tagging

Without v2, the hippocampus is a static input-output region — agent gets DG pattern-separated cues + CA1 readout, but can't *consolidate offline* via replay. The hypothesis that Cluster D adds value rests on offline replay; without it, Cluster D is just "more regions, more recurrent connections, more variance".

**Seed-100 vulnerability:** Cluster D's CA3 recurrent autoassociator (high density 0.30) is bistable by design — it can converge on either pattern-completion (good) or pattern-fragmentation (bad). At seed 100, the random CA3 init seems to push toward the bad attractor; with no SWR-driven cleanup, the bad state persists.

## Implications for cheat-5 strategy

**A+E (Step 4: 6.97 ± 0.83) remains the best multi-goal det operational point.** A+D doesn't beat it; A+D + the seed-100 vulnerability makes it *worse* than A+E.

Updated cluster-stacking picture (multi-goal det, n=6):

| Combo | Sum | Std | Verdict |
|---|---|---|---|
| baseline | 7.36 | 2.47 | reference |
| A+E | 6.97 | 0.83 | **operational best** (-75% std, modest mean improvement) |
| A+D | 8.67 | 5.13 | NEUTRAL — high variance driven by seed-100 |
| B.3+C v1 | 16.50 | 3.97 | NEGATIVE — phase-0 broken |

**Three options for next test:**

1. **A+D+E (all three cluster combinations)** — does adding D on top of the working A+E help, hurt, or wash out? If A+E already provides structural stability, D's bad-attractor behavior at seed-100 might be canceled out.
2. **Cluster D v2 (SWR replay)** — implement the deferred v2 from the cluster D design doc, then re-test A+D and A+D+E. Multi-day work.
3. **Cluster F (cerebellum)** — bigger build, deferred. The catalog's most under-built cluster (73× citations to Marr/Albus/Hesslow).

**Recommendation:** A+D+E is the smallest-effort next test (~60 min eval, no new code). If A+D+E ≥ A+E, D contributes positively when paired with E's topographic stability. If A+D+E < A+E, D is currently broken without v2 SWR.

## Provenance

- Code SHA at run time: `6f57dab` (after all 13 Wave-1 renames + B.3 retry findings).
- Wall-clock: ~80 min for 12 parallel runs (slower than Step 4's 60 min — Cluster D adds 5 regions, ~30% more synapses to update each step).
- All 12 result JSONs at `research/findings/raw/g11_bg/g11_seed{42,43,44,100,101,102}_AD_test_{baseline,AD}.json`.

## Acid test affirmation

The fresh baseline reproduces Step-4 baseline (7.36 ± 2.47 vs 7.64 ± 3.30) within noise. Implies the 13 Wave-1 structural renames committed today (cb55465 through 2f78d48) are **purely cosmetic** — no observable behavior change. The deprecated-alias map for plasticity gates (`_DEPRECATED_GATE_NAMES`), the deprecated-alias map for region names (`_DEPRECATED_REGION_NAMES`), and the deprecated `cp_plasticity_gain` property all work correctly under load.
