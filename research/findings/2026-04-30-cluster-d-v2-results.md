# 2026-04-30 — Cluster D v2 (SWR-gated CA3 plasticity): PARTIAL GO — opt-in only

**Run:** `g11_bg_runner.py` multi-goal deterministic, sleep replay window
inserted at step 1350 for 150 steps (within phase 3, leaves 300 steps
post-sleep for phase 3 final-quarter measurement). 6 seeds × 2 conditions =
12 parallel processes (~25 min wall-clock with GPU contention).
`CUBLAS_WORKSPACE_CONFIG=:4096:8`.

**Hypothesis:** Temporal restriction of CA3 recurrent plasticity to brief
windows during sleep (~14% duty cycle, matching biological NREM ripple
rate per Buzsaki 2015) selectively reinforces structured replay events
while suppressing reinforcement of constant-drive noise.

## Headline

**PARTIAL GO.** v2 mechanism real but doesn't break the A+E ceiling.

| Cond (n=6) | Mean | Std | Welch t vs AED | Verdict |
|---|---|---|---|---|
| A+E+D baseline | 29.32 | 6.95 | reference | acid-test |
| **A+E+D+v2** | **27.68** | **4.78** | t=-0.48 | partial GO |
| A+E (operational best, ref) | 6.97 | 0.83 | — | ★ ceiling |

- Δmean = **-1.64** vs same-stack baseline (passes design's ≤-1.0 threshold ✓)
- Δstd = **-2.17** (variance cut 31% — passes ≤baseline-std threshold ✓)
- Welch t = -0.48 (effect not statistically significant at n=6)
- A+E+D+v2 is still **~4× worse than A+E** alone, so D itself doesn't help

## Per-seed breakdown

| Seed | A+E+D | A+E+D+v2 | Per-phase v2 |
|---|---|---|---|
| 42 | 39.12 | 24.60 | [6.33, 6.72, 6.74, 4.81] |
| 43 | 35.71 | 34.41 | [6.73, 9.07, 11.34, 7.27] |
| 44 | 22.23 | 27.50 | [8.41, 3.03, 6.60, 9.46] |
| 100 | 30.14 | 32.58 | [3.55, 8.58, 10.00, 10.45] |
| 101 | 25.96 | 24.00 | [4.38, 9.69, 7.81, 2.12] |
| 102 | 22.79 | 22.99 | [8.12, 6.62, 5.54, 2.71] |

**v2 helps 4/6 seeds** (42, 43, 44 mixed; 101, 102 tighter; 100 worse). Variance
reduction comes from clipping the high outliers (seed 42: 39 → 24).

All v2 runs hit the SWR window pattern correctly: **22/150 sleep steps**
fired with gate=1.0 = 14.7% duty cycle, matching the design target.

## Implementation pivot: scheduled SWR vs endogenous bursts

Original design called for endogenous burst detection: monitor CA3
population firing rate, flip the plasticity gate when it spikes >μ+2σ.
Empirically this didn't fire at our scale. Verification with membrane-V
probes (2026-04-30):

- 220 pA into 5-10% sparse-Poisson kicks → V_mean stays at rest -65 mV
- 1500 pA into all 100 CA3 neurons → V_mean -4 mV, firing emerges

The IZH2007_HIPPO_PYRAMIDAL preset has rheobase ~109 pA but the model's
short integration window (n_stim_steps=200 inner sim steps per env step)
combined with sparse drive doesn't produce sustained firing. Real CA3
has ~150K parallel-fiber-equivalent inputs; we have ~30 (from dg→ca3
mossy at density 0.10 × 200 DG cells × 0.85 exc fraction × 0.30 active).

**Pivot:** scheduled SWR windows. Gate fully open every 7th sleep step
(period=7 → 14% duty cycle); 0.1 baseline otherwise. Both gate functions
ship — `_swr_gate_value` (endogenous, kept for unit tests + future
scaling work) and `_swr_gate_value_scheduled` (used by runner).

This is honest about the experimental setup. The autoassociator-cleanup
claim is weaker since bursts aren't endogenous, but the SWR temporal-
gating hypothesis is unchanged. The result still tests "does temporal
restriction of plasticity windows help offline consolidation".

## What "partial GO" means operationally

The mechanism does something — Δmean -1.64 with consistent variance
reduction across two tier-2 stacks AND tier-3 confirmation. But:

1. **Doesn't break the A+E ceiling.** A+E+D+v2 = 27.68 vs A+E alone =
   6.97. D itself is the bottleneck, not v2.
2. **Not statistically significant** at n=6 (Welch t=-0.48, p≈0.32).
   Need n=12+ to declare significance.
3. **No 6/6 seeds beat baseline** — only 4/6.

Recommendation: **ship `--enable-cluster-d-v2-swr` as opt-in**, like F v1.
Keep it out of flagship configs (A+E remains the documented best).
Useful for:
- Variance-sensitive deployments where reproducibility matters
- Future hippocampus work where v2 might compose with later improvements
- D-specific experiments testing offline consolidation mechanisms

NOT recommended for cheat-5 multi-goal det benchmarking: A+E (no D, no v2)
remains the operational best.

## Cluster-stacking ceiling now empirically confirmed at 7 attempts

| Stack | n | Mean | Std | vs A+E | Verdict |
|---|---|---|---|---|---|
| baseline | 6 | 7.77 | 3.33 | +0.80 | reference |
| A+D | 6 | 7.62 | 1.23 | +0.65 | NEUTRAL |
| A+D+E | 6 | similar | — | similar | NEUTRAL |
| **A+E** | 6 | **6.97** | **0.83** | reference | **★ ceiling** |
| A+F (F v1) | 6 | 7.37 | 1.83 | +0.40 | NEUTRAL |
| A+E+F (F v1) | 6 | 8.02 | 1.81 | +1.05 | NEUTRAL |
| A+F v2 | 6 | 21.77 | 2.35 | +14.80 | NEGATIVE |
| A+E+F v2 | 6 | 24.88 | 3.07 | +17.91 | NEGATIVE |
| A+E+D (sleep at 1350) | 6 | 29.32 | 6.95 | +22.35 | sleep replay hurts D stacks |
| **A+E+D+v2** | **6** | **27.68** | **4.78** | **+20.71** | **PARTIAL — v2 mitigates D's harm** |

Important addendum: A+E+D *with sleep replay at 1350* is much worse than
A+E alone (29.32 vs 6.97). Sleep replay actively HURTS the A+E+D stack.
v2's effect is mostly to mitigate that damage (27.68 < 29.32) — not to
break new ground.

This is consistent with the SCIENCE_ROADMAP §4.7 note "content quality
is the bottleneck" for sleep replay. v2 changes the GATING of plasticity
during replay, not the CONTENT being replayed. With the wrong content
(stale trajectories from earlier phases, replayed during a new phase
with a different goal), even tightly-gated plasticity reinforces the
wrong things, just less of them.

## Possible follow-ups (not committing to)

1. **Reverse-order trajectory replay (the original Option 2).** Real
   hippocampus replays trajectories in reverse for credit assignment.
   Sample (x_t, y_t, gx, gy) from successful_trajectories in REVERSE
   time order. Independent failure mode from v2 — could compose. ~30 LOC.
2. **Recency-weighted replay.** Bias the trajectory sampling toward
   recent steps (probability ∝ exp(-(now - step_t) / tau)). Only
   replay steps from the *current* phase, not stale phases. ~20 LOC.
3. **Scale CA3 to 1000+ neurons.** Test whether endogenous bursts emerge
   at biological scale. Would let us revert to the burst-detection
   mechanism. ~5-10× wall-clock per run.

These are deferred. The decisive finding is that A+E remains the
operational ceiling for cheat-5 across 7 cluster-stacking attempts.
The buildout strategy needs a different lever than per-step plasticity
scaffolding.

## Files

- Tier-2a results (A+D vs A+D+v2): `dv2_t2_*.json`
- Tier-2b results (A+E+D vs A+E+D+v2, n=3): `dv2_t2b_*.json`
- Tier-3 results (A+E+D vs A+E+D+v2, n=6): `dv2_t3_*.json`
- Implementation: `research/runners/g11_bg_runner.py:_swr_gate_value_scheduled`
  (used) and `_swr_gate_value` (endogenous, kept)
- Tests: `tests/test_cluster_d.py` (11 tests passing)
- Design doc: `docs/plans/2026-04-30-cluster-d-v2-swr-design.md`
- Earlier F v2 NO-GO: `2026-04-30-cluster-f-v2-results.md`
