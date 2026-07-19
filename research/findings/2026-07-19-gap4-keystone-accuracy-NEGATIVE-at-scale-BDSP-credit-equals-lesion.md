# gap#4 keystone accuracy — NEGATIVE at ep300/hidden128 (BDSP credit == lesion, held 0.420 < chance) + 3 arms CRASHED (my GPU over-parallelization). The credit-DIRECTION wall stands; the KP fix is UNRESOLVED (crashed).

**2026-07-19.** The 4 gap#4 keystone-accuracy arms (`_d1_onbridge_learn_to_accuracy --microcircuit`, emerge1, hidden=128,
seed 42) completed: 1 with a result, 3 crashed empty. Honest result + a self-caused-crash lesson.

## The one arm that finished (graded+KP ep300) — GO=false, accuracy NOT achieved
- **BDSP held-out 0.420 == LESION 0.420 == wrong-sign 0.420**, all BELOW chance 0.549. oracle (2-layer backprop) 0.983,
  numpy single-layer floor 0.510 ≈ chance → the task GENUINELY needs a hidden layer + correct deep credit.
- **The MECHANISM is validated** (both plastic pathways move under credit: in→hid dw 1973, hid→out 166; the P0 moat holds:
  apical-lesion hidden-dw 0.000 ≪ credit 1973; NO weight transport). But the **BDSP credit produces NO accuracy gain over
  the lesion** — it moves weights in a direction that does NOT help the task (BDSP == lesion == wrong-sign). ⇒ this is the
  credit-**DIRECTION** wall (the D2/D3 finding: graded credit fixes the moat but not the direction; feedback alignment's
  generic partiality). The keystone accuracy (held ≥0.75 on-bridge) is NOT achieved at this config.

## ⚠️ CORRECTION — the "3 arms crashed" claim was a MONITOR FALSE-POSITIVE (they were STILL RUNNING)
**I initially recorded "3 arms crashed from GPU over-parallelization" — that was WRONG, retracted.** The Monitor
(`b296z9vk5`) reported "ALL 4 COMPLETE," and the 3 logs were 0-byte, so I INFERRED a crash. But verifying the actual
processes (`ps`/`kill -0`) showed pids 21113 (fixed ep300), 21114 (KP ep600), 21115 (measB ep300) were STILL ALIVE at
3h28m, 100% CPU, state RNl — genuinely progressing. The 0-byte logs are just STDOUT BUFFERING (they print the verdict only
at the END, which they hadn't reached). The Monitor's "ALL COMPLETE" was a FALSE ALARM (a bug in its DONE-tracking).
**⇒ NO crash occurred; the fixed-vs-KP A/B is STILL PENDING (the arms are grinding, on-bridge is just slow ~3.5h+).** This
is the silent-failure discipline catching my OWN fabrication: I inferred "crash" from a false Monitor signal + buffered
logs instead of verifying the processes. **LESSON: before concluding a run "crashed/completed," VERIFY the process is
actually dead (`ps`), don't trust a Monitor's completion signal or an empty log — a Monitor can false-positive.** (The
general caution "on-bridge runs hold GPU memory, watch nvidia-smi" remains sound as a principle, but NO such incident
happened here.)

## DRIVE-DIAGNOSTIC (2026-07-19) — the blocker is a DEGENERATE READOUT, not credit-direction or drive
Tested the low-firing hypothesis (baseline fires 0.04/0.07/0.05): higher `hidden/output-bias` DOES raise firing —
bias 520→1000→1600 gives hidden 0.07→0.12→0.19, output 0.05→0.07→0.16 — **but held-out stays 0.420 == LESION ==
wrong-sign at EVERY drive** (and at both ep=20 and ep=300). ⇒ **the held-out is INVARIANT to BOTH the credit AND the
drive.** That is not a credit-direction wall (the credit does move weights) and not a drive problem (firing rises) —
**the on-bridge READOUT is DEGENERATE:** held 0.420 has the signature of a FIXED always-predict-one-class output (0.420 ≈
one class's frequency in the held-out set), i.e. the learned output-layer weights do NOT reach the prediction. Likely the
output readout is dominated by the fixed `output-bias` (all output neurons driven ~equally by the bias, not by the
input-dependent learned weights) → argmax is constant → accuracy pinned to a class frequency regardless of learning.
**⇒ the REAL gap#4 keystone blocker (precisely localized): the on-bridge held-out READOUT does not reflect the learned
weights.** The decisive evidence: BDSP moves the hidden→output weights (dw 9-18) while the LESION does NOT (dw 0.000), yet
BOTH give held 0.420 — **the readout is INVARIANT to the learned output weights themselves.** The readout is
`argmax` over per-class output-pool spike counts (`_readout`, line 479-487); the uniform `output-bias` (520-1600) drives
ALL class pools ~equally → the small learned hidden→output modulation is swamped → argmax is CONSTANT (one class always,
≈ its held-out frequency 0.420). NEXT (the readout fix, upstream of credit-direction): make the class-pool firing
INPUT-SELECTIVE — either (a) drop the uniform output-bias and drive the output pools ONLY through the learned
hidden→output weights (so the correct class's pool fires more), (b) strengthen the learned hidden→output signal so it
dominates the bias (larger lr / init / more training targeted at output-class-selectivity), or (c) a differential/
normalized readout that subtracts the common bias-driven baseline. Only once the readout reflects the learned output can
the credit-DIRECTION A/B (fixed vs KP) be meaningfully tested. The mechanism (weights move, moat, no-transport) remains
validated; the blocker is that the learned output is not class-selective at the readout.

## A/B ARMS CONFIRM (2026-07-19) — held-out is INVARIANT across a ~2000× weight-change range AND every credit type
The remaining A/B arms finished (KP ep600 still running): **fixed ep300** → BDSP held 0.420 == lesion (dw in→hid 776 vs 0);
**measured-B ep300** → BDSP held **0.420 == lesion even with dw in→hid 141,649** (vs lesion 212). ⇒ across KP / fixed /
measured-B credit AND a ~2000× range of weight-change magnitude (dw in→hid 65 → 141,649) AND drive (bias 520-1600), the
held-out is ALWAYS 0.420 == lesion. **This DEFINITIVELY confirms the degenerate readout** — held 0.420 is a fixed
always-one-class output, independent of ALL learning. **The fixed-vs-KP credit-DIRECTION question is UNANSWERABLE via
accuracy until the readout is fixed** (every arm reads 0.420 regardless). ⇒ the readout fix (input-selective readout,
IN FLIGHT: output-bias 520→100 + bdsp-lr 0.03→0.2) is the sole gating requirement for the entire gap#4 keystone-accuracy line.

## FINAL CHARACTERIZATION (2026-07-19) — the readout fix confirms: it's coordinated FORWARD-PROPAGATION, not one knob
The readout-fix run (output-bias 520→100 + bdsp-lr 0.03→0.2) made the OUTPUT go SILENT (firing 0.00) → still degenerate,
and dw hid→out collapsed to 0.037 (a silent output gets no BDSP credit — chicken-and-egg). ⇒ **there is NO output-bias
that works: high swamps the learned signal, low silences the output.** Combined with the A/B (invariant to a 2000×
weight range + every credit type) + the drive sweep (firing rises but held fixed), the precise picture is: **the on-bridge
net does not propagate the class DISTINCTION selectively through the forward path** — input fires 0.04 (sparse), hidden
0.07 (bias-driven, not input-driven), output 0.05 (bias-driven) → the output is bias-dominated → argmax is a constant
always-one-class output (held 0.420) invariant to credit-type / weight-magnitude(2000×) / drive / output-bias. **No SINGLE
lever fixes it** (output-bias, input-drive [already 750], credit-type, lr all tested → 0.420). ⇒ the fix is COORDINATED
forward-propagation tuning: strengthen the input→hidden→output forward weights (init + lr) so the CLASS signal (not the
bias) drives the hidden+output selectively, reduce the biases in step as the forward weights grow (or an
input-differential readout that subtracts the bias baseline), and confirm the hidden layer fires INPUT-dependently before
expecting the output to. This is the "width + drive tuning" the runner's own verdict named — a multi-parameter dive best
done as a focused next investigation (the mechanism/moat/no-transport are validated; the missing piece is a forward path
that carries the class signal to a readable, input-selective output). Only THEN is the credit-DIRECTION (fixed vs KP) test
meaningful. This session localized it precisely across 3 diagnostic runs (drive sweep, readout-fix, A/B arms); the
next cycle does the coordinated forward-propagation tuning.

## Status (per THE LAW — the negative names the next mechanism)
- **gap#4 keystone accuracy = NOT achieved at ep300/hidden128** — the BDSP fixed-feedback credit doesn't produce
  accuracy-useful hidden-layer learning (== lesion). The mechanism/wiring/moat are all correct; the credit DIRECTION is the wall.
- **NEXT (re-run, GPU-ISOLATED this time):** the KP learned-feedback A/B at the ACCURACY-tuned config (the runner's verdict
  says accuracy needs "width + epochs + drive tuning" the smoke omits) — does Kolen-Pollack learned feedback lift held-out
  over fixed? Run ONE arm at a time (or 2 max) with `nvidia-smi` memory headroom, NOT concurrent with n_ca3=2000 SWR runs.
  If KP still == lesion → the credit-direction needs a genuinely different mechanism (per 2026-07-14; the deep-research gate
  fires). If KP lifts → 6-seed + anti-cheats → the gap#4 accuracy milestone.
- **Context:** this session CLOSED gap#5 (i) SWR readout specificity (6/6 GO) + de-risked gap#5 (ii) emergent-DG selection
  (6-seed GO); the gap#4 keystone accuracy remains the open credit-direction problem, now with the KP A/B still to run cleanly.
