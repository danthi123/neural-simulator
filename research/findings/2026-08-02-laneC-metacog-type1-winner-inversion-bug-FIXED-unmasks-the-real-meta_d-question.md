---
type: finding
status: contributing
date: 2026-08-02
mechanism: second-order-metacognition-monitor
artifacts:
  - research/findings/raw/lanes/metacog/metacog_beforeafter_full_s42.json
  - research/findings/raw/lanes/metacog/metacog_beforeafter_late_s42.json
---

# lane C (metacognition / self-awareness): the first-attempt NEGATIVE was a TYPE-1 SETUP BUG, not a monitor failure — the 2AFC winner was read in an adaptation-INVERTED window (d'=-0.92, acc 0.32 << chance); reading the decision over the full evidence window FIXES it (d'=+1.61, acc 0.80), which UNMASKS the real question the bug was hiding: is the slow-NMDA monitor's confidence genuinely metacognitive (meta_d) once the first-order task is correct?

<!--derived-->
**One-line verdict.** lane C's first-attempt 6-seed result (`metacog_6seed.json`, mean type1_accuracy 0.239, mean d1
-1.42) was scored as a metacognition NEGATIVE, but the type-1 discrimination itself was BELOW CHANCE with a NEGATIVE d' —
a systematic winner INVERSION, not a weak monitor. Root cause: the first-order winner (argmax over the two workspace
assembly rates) was read only in the LAST THIRD of the free-run, where spike-frequency ADAPTATION has let the
strongly-driven correct assembly fall below the weakly-driven one, inverting the decision. Reading the winner over the
FULL evidence window (the balance of evidence the competition actually accumulates) fixes the type-1 task; the confidence
monitor still reads its sustained late-window state. Runner-side fix only (`--decision-window full`, default; `late`
reproduces the original exactly as the before/after control). No `sim/` edit.

## Before/after (1-seed smoke, n=120 trials, seed 42) — the fix is a large unambiguous flip

<!--derived-->
Representative artifacts: `research/findings/raw/lanes/metacog/metacog_beforeafter_full_s42.json` and
`research/findings/raw/lanes/metacog/metacog_beforeafter_late_s42.json`.

<!--derived-->
| decision window | type1_accuracy | d1 | type2_auc | meta_d |
|---|---|---|---|---|
| `late` (original = the bug) | 0.317 | **-0.923** (inverted) | 0.650 | 0.036 |
| `full` (the fix) | **0.800** (in the [0.60,0.90] operating window) | **+1.614** | 0.650 | 0.005 |

<!--derived-->
The `late` arm reproduces the committed first-attempt failure (negative d', sub-chance accuracy); the `full` arm lifts
type-1 accuracy to 0.800 — squarely inside the GO gate's operating window [0.60, 0.90], where the first-order task has
genuine errors to be metacognitive about (not a ceiling, not chance). So the type-1 setup bug is fixed and confirmed by
a clean before/after control.

## What the fix UNMASKS — the real lane-C question (no-defer)

<!--derived-->
With the type-1 inversion removed, the metacognition read is now interpretable — and the 1-seed smoke shows **meta_d ~=
0.005** (essentially zero) despite type2_auc 0.650. That is the pattern of a type-2 AUC riding on the between-class
signal rather than genuine metacognitive sensitivity: the monitor's confidence may not separate CORRECT from ERROR
trials WITHIN a stimulus class once the first-order decision is right (`min_per_class_type2_auc` 0.450 on this seed). The
bug was MASKING this question (with an inverted type-1, "correct" was mislabeled, so no metacognition read was
meaningful). A 6-seed run (`--seeds 42 43 44 100 101 102 --n-trials 160`, full window) with the runner's built-in
adversarial controls — the meta-lesion dissociation (does zeroing the monitor's read collapse meta_d while leaving d'
unchanged?), permuted-confidence (does a decorrelated confidence drop type-2 to chance?), and within-class type-2 AUC —
is running to decide: (a) meta_d is genuinely ~0 => the slow-NMDA monitor as wired does not yet carry metacognitive
sensitivity (an honest negative that LAUNCHES the monitor-mechanism search: read-weight tuning, an evidence-margin
read rather than a winner-rate read, or a learned monitor), or (b) meta_d is real at 6 seeds and the 1-seed smoke was
noise. Either way the type-1 fix is a prerequisite that was blocking any valid metacognition read, now removed.

## Honest scope

<!--derived-->
This finding banks a SETUP-BUG FIX (validated before/after) and RE-OPENS the lane-C metacognition question that the bug
had prematurely closed as a negative — it does NOT itself claim a working metacognition monitor. The functional-correlate
framing stands: any positive meta_d is reported as an honest functional read-out ("the monitor's confidence tracks
first-order correctness"), never as subjective awareness. The prior `metacog_6seed.json` NEGATIVE is superseded by this
fix (its type-1 was inverted); the real verdict awaits the 6-seed full-window run.
