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

## Update 1 (2026-08-02, same-cycle) — the 6-seed full-window verdict: type-1 fix ROBUST (GO), metacognition a GENUINE honest-negative

<!--derived-->
The 6-seed full-window run landed (n=160/seed; artifacts `research/findings/raw/lanes/metacog/metacog_full_s42.json`
etc.). **Type-1 fix is robust:** type1_accuracy mean 0.718, ALL 6 seeds inside the [0.60, 0.90] operating window
(0.625-0.769), d1 mean +1.23 (all positive) — the winner-inversion fix holds across seeds, not a 1-seed fluke.
**Metacognition is a genuine honest-negative:** meta_d = 0.000 on 4/6 seeds (mean 0.188; only s101 +0.727, s43 +0.401),
type2_auc mean 0.490 (< the 0.65 GO threshold), within-class type-2 AUC mean 0.450 (< the 0.55 threshold, ~chance),
permuted-confidence type-2 0.526 (~chance, i.e. no real confidence-correctness coupling to destroy), and the meta-lesion
type-2 sits at exactly 0.500. So the slow-NMDA monitor AS WIRED does not carry reliable metacognitive sensitivity once
the first-order task is correct — this is a real negative, no longer masked by the type-1 bug. It is NOT structurally
impossible: s101 reaches meta_d +0.727 with within-class 0.531, so the monitor CAN separate correct-from-error on some
seeds — the read is on the edge, not dead.

<!--derived-->
**The mechanism search this launches (no-defer).** Per THE LAW this negative is a verdict on the CURRENT monitor WIRING,
not the capability. Two candidate surpasses, cheapest first: (1) TUNE the monitor read balance (a `--meta-exc-w x
--meta-inh-w` sweep — running — asks whether the winner-exc vs total-inh balance can be set so confidence tracks the
winning MARGIN reliably); (2) if magnitude-tuning does not lift meta_d, the read is STRUCTURALLY margin-blind (winner
rate alone cannot encode confidence when a wrong winner also fires strongly) => change the monitor to read the
evidence-MARGIN (winner - runner-up) directly, biologically a difference-of-assemblies / normalization read, or a LEARNED
monitor. The sweep decides which branch; the 1-seed smoke's meta_d~=0 is now confirmed as the 6-seed pattern.

## Honest scope

<!--derived-->
This finding banks a SETUP-BUG FIX (validated before/after AND 6-seed robust for type-1) and a GENUINE 6-seed
honest-negative for the metacognition monitor as wired — it does NOT claim a working metacognition monitor. The
functional-correlate framing stands: any positive meta_d is reported as an honest functional read-out ("the monitor's
confidence tracks first-order correctness"), never as subjective awareness. The prior `metacog_6seed.json` NEGATIVE is
superseded (its type-1 was inverted, making its metacognition read meaningless); this is the corrected verdict — type-1
GO, monitor sensitivity an open honest-negative with a named mechanism search in progress.
