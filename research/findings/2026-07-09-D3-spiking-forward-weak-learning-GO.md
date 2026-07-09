# D3 — the WEAK-SUPERVISION LEARNING with a SPIKING-FORWARD transition (6-seed GO): δ learned from end-state-only supervision THROUGH a spiking LIF hidden

**Date:** 2026-07-09
**Runner:** `research/runners/_d3_spiking_weak_learning_derisk.py` (reuse-by-import: `make_group_task` + `sim.surrogate_grad`; numpy; NO `sim/` edit).
**Verdict:** GO (S3, 6-seed) — the transition FORWARD is spiking throughout the weak-supervision learning.

## What this composes
Two D3 pieces: **RANK-1** learned δ from END-STATE-only supervision + curriculum through a RATE (tanh) hidden; **rung-2** showed a spiking LIF hidden REPRESENTS δ (surrogate grad, but on teacher-forced triples). This composes them — the transition's hidden is a **spiking LIF pool** (rate-coded over T steps, trained by surrogate gradient), and it is trained from **weak (end-state-only) supervision** via the Dyna-style detached-rollout curriculum (roll with the LIF-argmax, supervise ONLY the final K-way state). So the transition **forward is spiking THROUGHOUT the weak-supervision learning**, not just at execution.

## The result (S3, 6-seed; NO `sim/` edit)
| arm (DEEPER = held-out lengths 6/7/8, chance 0.5), 6-seed | mean |
|---|---|
| **SPIKING-FORWARD weak-learn STATE (LIF hidden, surrogate grad, end-state-only)** | **1.000** (state-track **1.000**, every seed) |
| PROPERTY endpoint (= rung 3, 1 bit) | 0.639 (chance-ish; 2 blind seeds' 2-coloring partly learnable, but STATE is 1.0 every seed) |
| SHUFFLE (memorization floor) | 0.589 (collapses) |

GO all 6 seeds (42/43/44/100/101/102): STATE 1.000 ≫ PROPERTY 0.639 ≫ SHUFFLE 0.589.

**GO:** the group-multiplication δ is LEARNED from end-state-only supervision THROUGH a spiking LIF hidden and length-generalizes to held-out-deeper ≫ the 1-bit property endpoint (the reframe) and the shuffle floor (genuine learning).

## ⇒ the weak-supervision transition learning is spiking-forward
Combining the arc: the discrete-attractor recurrent composition is **found** (CA3), **learned from weak (end-state-only) supervision** with a **spiking LIF forward** (this), and **executed on spikes** (FS-WTA re-discretization + the weak-learned δ on spikes). The simulated recurrent sequence/language cortex step is learned-from-weak-supervision with a spiking forward AND executed on spikes.

## Honest scope
- The surrogate-gradient BACKWARD is still host BPTT — a **biologically-plausible LOCAL learning rule** (three-factor / eligibility / e-prop) is the separate DEEP wall (EMERGE-6..8's 5×-confirmed dead-end: local rules don't beat a fixed reservoir + trained read-out; the research gate explicitly said do NOT re-attack it). This rung makes the transition FORWARD spiking during learning, matching rung-2's "transition on spikes" sense (spiking forward, host backward).
- S3 (K=6). A5 weak-supervision is the honest structureless-worst-case boundary (see `2026-07-09-D3-weak-supervision-RANK1-GO.md`).

## Files
`research/runners/_d3_spiking_weak_learning_derisk.py`; the D3 arc `2026-07-09-D3-*.md`.
