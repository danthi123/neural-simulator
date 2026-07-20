# gap#4 RUNG 3 — NO-GO: stacking is blocked by a PER-LAYER SCALE dependence in the depression threshold

**2026-07-20.** Rung 1: one cell learns a place field from ONE plateau (6-seed, blind-clean). Rung 2: 4 cells learn 4
distinct fields in one lap on shared inputs (6-seed, blind-clean, shuffle control 0.00). Rung 3 asks the first
question that earns the word *deep*: **can a downstream layer learn to READ the learned code, using the same rule?**

L2 receives input ONLY from the 4 CA1 pools — never from position. Its entire access to the world is the
representation layer 1 learned.

## Verdict: NO-GO — and the blocker is precisely located

| stage | result |
|---|---|
| Stage 1 (form the map) | **intact on every seed** — `ca1_peaks = [4, 8, 12, 16]`, `map_ok = 1` |
| Stage 2 (L2 learns to read) | **`l2_delta_max = 0.00000`, `l2_peak = -1` — L2's firing never changes** |

## Two distinct failures, separated by measurement

**(1) REPRESENTATION ATTENUATION — quantified.** CA1's field-peak rate is **0.005 spikes/neuron/step**. Position pools
are driven with **900 pA of direct current**; CA1 drives L2 only with those sparse spikes. Measured gain needed before
the learned code can drive a downstream cell **at all**:

| `ca1→l2` weight | L2 response |
|---|---|
| 0.6 (same as position→CA1) | silent |
| 5.0 | silent |
| 20.0 | silent |
| **60.0** | responds |
| 150.0 | responds |

⇒ reading a learned layer needs **~100-250× the synaptic weight** of reading the input layer. Each learned layer
attenuates enormously in drive terms. This alone makes naive stacking fail.

**(2) THE DEPRESSION THRESHOLD IS PER-LAYER SCALE-DEPENDENT — the actual blocker.** `btsp_hetero_theta = 0.012` was
calibrated to **layer 1's** eligibility range (0.0068–0.0227, produced by 900 pA drive). CA1 fires ~100× more sparsely,
so **every `ca1→l2` synapse sits far below θ and takes FULL depression** — stage 2 recorded `dw = −1289`, i.e. it
*crushed* the very weights it was supposed to shape.

Turning stage-2 depression off isolates the rest: potentiation then works (`dw = +6431`) but **`l2_delta` is still
exactly 0** — because L2 already responds to *all four* CA1 fields, and non-selective potentiation cannot change
*which* bins it fires at. **Selectivity requires depressing the non-target inputs — the very thing θ mis-scaling
prevents.** So the two failures are not independent: with θ correct, depression is what would create selectivity.

## Why this matters more than the NO-GO

`btsp_hetero_theta` is a **single global scalar** in `CoreSimConfig`. Rungs 1–2 work because every synapse in them
shares one input statistic (position pools at 900 pA). The moment a *second* layer reads a *learned* layer, the two
pathways have eligibility distributions ~100× apart and **no single θ can serve both**.

⇒ **The thresholded-depression mechanism that made rungs 1–2 work does not, as implemented, stack.**

## Named next levers (in order; none is "tune θ")

1. **Per-pathway θ** — the principled fix. `btsp_hetero_theta` must become per-synapse/per-pathway (matching how
   `plasticity_gate` / `transmission_gate` are already per-pathway) rather than one global scalar. Additive,
   default-off, byte-identical when unset — the same discipline as the θ kernel edit itself.
2. **Or normalize eligibility per postsynaptic cell** — make the gate relative (e.g. θ as a *quantile* of each cell's
   own presynaptic eligibility distribution) so it is scale-free by construction. This is arguably the more
   biological answer and would remove the calibration burden entirely.
3. Independently: the ~100-250× read-out gain requirement is worth understanding on its own — it is a property of how
   sparsely the learned code fires, and it will recur in any layered use of these representations.

## Honest scope

Rungs 1–2 stand unchanged (6-seed, blind-clean, controls collapsing). What rung 3 establishes is that **the local rule
composes WITHIN a layer but does not yet stack ACROSS layers**, for a specific and fixable reason. gap#4's deep-credit
frontier remains **open** — but it is now open at a named, mechanical obstacle rather than a diffuse one.
