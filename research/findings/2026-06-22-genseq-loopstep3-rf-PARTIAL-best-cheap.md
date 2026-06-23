# Loop-step 3 RF complex-accumulator probe = PARTIAL: the RF path escapes the `g·(V−E)` wall + the accumulator is rank-FAITHFUL, but the stacked per-layer clip still compresses (0.556 < 0.85). The best cheap option; the cheap ladder is now exhausted (2026-06-22)

**Scope:** the last cheap probe (P1, the RF resonate-and-fire complex-synapse path) before the multi-week
differentiable-bridge `sim/` edit. `research/runners/_genseq_loopstep3_rf_probe.py`, GPU. **NO `sim/` edit.** On
`main`.

## Result — PARTIAL (0.556; the best cheap option; below the 0.85 both-walls-escape bar)
The RF complex accumulator computes `Re(Z) = nsteps·(a@W)` EXACTLY (verified rank 1.000, im→0) — **NO clip, NO
`g·(V−E)`, NO refractory ceiling**. So the RF path ESCAPES both walls *for the linear matvec*.
| | per-layer | cumulative |
|---|---|---|
| RF (this probe) | [0.934, 0.675, 0.556] | **0.556** |
| rate/graded (the 4 NEGATIVEs) | [0.846, 0.620, 0.288] | 0.288 |

~2× the rate/graded. Specificity margin **0.190** (re-opened vs the distill's tiny margins); shuffled-control 0.373
(real > shuffled by 0.183 — the same metric confound, teacher final-reps char-correlated, reported honestly). The
RF-native PHASE channel = near-chance (~0.06) — encoding rank in phase does NOT carry a dense layer's rank.

## Why PARTIAL not GO (the residual)
The RF accumulator is rank-faithful for the LINEAR op, but each layer's READOUT re-imposes
`a_hat = clip(Re(Z)/scale, 0, 1)` (to match the teacher's clip nonlinearity + feed the next layer's magnitude) → the
per-layer clip STILL compresses, accumulating (0.934→0.675→0.556). RF solves **W2 (`g·(V−E)`) but not W1 (the
per-layer clip)**.

## The KEY INSIGHT — a possible last cheap shot
The RF accumulator has **NO `g·(V−E)`** — which was the EXACT killer of the distillation NEGATIVE (the graded
distillation's weights didn't survive the live `g·(V−E)`). So **DISTILLING on the RF path** (the accumulator is
trivially differentiable, `nsteps·(a@W)`, no conductance term) would NOT face that killer — the trained weights would
install faithfully. RF + distillation combines the two best partial results (RF's no-conductance escape +
distillation's train-through-the-clip). A genuine last cheap shot before the multi-week edit.

## Verdict — cheap ladder exhausted (5 attempts, best 0.556); decision point
spike-rate NEG(0.009) → graded PARTIAL(0.327) → pop-code NO-OP → distill-hybrid NEG-on-live(0.444) → RF
PARTIAL(0.556). None reach 0.85. NEXT options: **(a) RF + distillation** (the last cheap shot — the insight above;
the RF path removes the distillation's killer); **(b) the multi-week differentiable-bridge `sim/` edit**; **(c) pivot
to C2 on the knowledge store** (the deepest science; needs no consolidation). The spiking-CONVERT GO + P2-KNOWLEDGE
GO stand regardless.
