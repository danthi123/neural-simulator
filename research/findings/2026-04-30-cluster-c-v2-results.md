# 2026-04-30 — Cluster C v2 (compartmentalized DA): NEGATIVE on cheat-5

**Run:** `g11_bg_runner.py` multi-goal deterministic, n=6 seeds × 2 conditions =
12 parallel processes (~30 min wall-clock with GPU contention).
`CUBLAS_WORKSPACE_CONFIG=:4096:8`.

**Hypothesis:** Per-action DA channels (4 modulators: dopamine_N, dopamine_E,
dopamine_S, dopamine_W) decouple credit assignment by action specificity.
Real BG has compartmentalized DA innervation matched to striatal action
selectivity (Mohebi 2019; Howe & Dombeck 2016). Each synapse carries an
`action_tag ∈ {0..3}` and only responds to its target's DA channel.
Should fix what broadcast DA can't: cross-action contamination during
multi-goal phases.

## Headline

**NEGATIVE.** A+E+C v2 mean is 2.08 *higher* than A+E alone (worse),
with 2.5× the variance. Welch t=+1.21 (not statistically significant
at n=6, but trends in the wrong direction).

| Cond (n=6) | Mean | Std | Welch t vs A+E | Verdict |
|---|---|---|---|---|
| A+E baseline | 7.18 | 1.58 | reference | acid-test |
| **A+E+C v2** | **9.26** | **3.91** | **t=+1.21** | **NEGATIVE** |
| A+E (doc 2026-04-29) | 6.97 | 0.83 | — | ★ ceiling |

## Per-seed breakdown

| Seed | A+E | A+E+C v2 | v2 helps? | per-phase v2 |
|---|---|---|---|---|
| 42 | 9.86 | 6.73 | ✓ helps | [1.45, 1.35, 1.32, 2.61] |
| 43 | 7.00 | 5.95 | ✓ helps | [1.56, 1.45, 1.81, 1.13] |
| 44 | 5.39 | 6.45 | hurts | [1.34, 2.76, 1.07, 1.28] |
| 100 | 6.95 | 10.40 | hurts | [1.35, 5.50, 1.59, 1.96] |
| 101 | 7.91 | 16.31 | **hurts badly** | [5.73, 2.00, 2.47, 6.11] |
| 102 | 5.95 | 9.73 | hurts | [2.11, 2.46, 1.76, 3.40] |

**C v2 helps 2/6 seeds, hurts 4/6.** Worst case (seed 101) is more than
2× baseline. Strong seed-dependence — same pattern as the asymmetric
adaptive DA findings from 2026-04-26 (seed 42-44 win, seed 100-102 lose).

## Why per-action DA may have hurt

Three plausible mechanisms (would need controlled experiments to disambiguate):

1. **Action selection is noisy.** During exploration, the agent picks
   action X but might benefit from learning at action-Y synapses too
   (off-policy credit assignment). Compartmentalizing DA strictly to
   the chosen action prevents that off-policy learning. Broadcast DA
   has built-in robustness against noisy action selection because every
   synapse gets the signal.

2. **Phase transitions reset action-X DA channels asymmetrically.**
   When goal changes, some actions become newly correct (high reward)
   and others newly incorrect (negative reward). Compartmentalized DA
   means each action's DA channel sees a different signal — but our
   reduced model doesn't have separate striatal patches per action,
   so the channels' decay time constants and baselines drift apart
   during phase transitions, sometimes destructively.

3. **The DA tagging is correct but the rest of the architecture isn't
   action-channelized.** Our cortex_X / motor_X regions are tagged
   per-action, but the global cortical drive (heuristic, learned
   perception) routes the same signal to all 4 cortex pools. So the
   "credit assignment" handle that C v2 adds doesn't compose with our
   non-channelized cortex inputs.

## Cluster-stacking ceiling: 8 attempts now NEUTRAL or NEGATIVE past A+E

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
| A+E+D (sleep) | 6 | 29.32 | 6.95 | +22.35 | NEGATIVE |
| A+E+D+v2 | 6 | 27.68 | 4.78 | +20.71 | PARTIAL |
| **A+E+C v2** | **6** | **9.26** | **3.91** | **+2.29** | **NEGATIVE** |

**Eight cluster-stacking attempts past A+E. None help.** Five hurt.

The cluster strategy was: build out missing biology (A, B, C, D, E, F)
incrementally, expecting each cluster to add complementary capacity.
Empirically: A+E sits at a robust ceiling and additional clusters either
do nothing (A+D, A+F, A+E+F) or actively disrupt (F v2, D with sleep,
C v2).

## Implications

The bottleneck is not "missing biology cluster X." Adding scaffolding
doesn't help. Possible alternative explanations:

1. **The reduced model is fundamentally too small.** ~1500 neurons can't
   support the dynamics that the cluster mechanisms assume (large PF
   pools for cerebellum, large CA3 for hippocampus, large striatal
   patches for compartmentalized DA). Each cluster's biology needs more
   neurons than we provide.

2. **The cheat-5 multi-goal task is too easy for biology to matter.**
   At 1800 steps × 4 phases, the agent can solve the task with simple
   perception + closed BG loop. Biology buildouts that would matter for
   *harder* tasks (e.g. delayed credit, partial observability, planning)
   don't show up on this benchmark.

3. **Cluster A is doing all the real work.** The closed BG loop
   (cortex→stn hyperdirect + thal→cortex feedback) already provides the
   credit-assignment topology that other clusters were trying to add.
   E (topographic maps) gives a ~5-10% additional boost. Other clusters
   are redundant or interfering.

## Recommendation

**Stop the cluster-stacking strategy.** Empirically falsified across 8
attempts. Future work should pivot to one of:

- **(a) Scaling.** Increase model size 5-10×. Several clusters were
  flagged as "scale-bound" (F v2 specifically, D v2 implicitly). At
  larger N the autoassociator dynamics, parallel-fiber populations,
  and patch sizes might become functionally meaningful.
- **(b) Harder benchmarks.** Cheat-5 multi-goal det is solved by A+E.
  Biology buildouts would need to be tested on tasks where A+E is
  *not* sufficient: delayed credit assignment (>30s), partial
  observability, or compositional structure.
- **(c) Interactive/adaptive evaluation.** The current eval lets the
  agent be greedy with full state. Real biology shines in exploration
  and recovery from perturbation. Eval framework redesign.

For immediate code shipping: **`--enable-compartmentalized-da` ships as
opt-in like F v1 / D v2**, kept out of flagship. Useful for future
experiments where per-action credit assignment might compose differently
(e.g. scaled model, harder task).

## Files

- Results: `research/findings/raw/g11_bg/cv2_eval_AE_seed*.json`,
  `cv2_eval_AECv2_seed*.json` (12 files)
- Implementation: pre-existing in `sim/neuromodulators.py`,
  `sim/bridge.py`, `research/runners/g11_bg_runner.py`
- Tests: `tests/test_g11_bg_runner_flags.py::test_compartmentalized_da_*`
- Design doc: `docs/plans/2026-04-29-cluster-c-v2-compartmentalized-da-design.md`
