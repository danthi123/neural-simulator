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

---

## UPDATE — per-pathway θ implemented and tested: NECESSARY but NOT SUFFICIENT

Lever 1 was implemented (`sim/bridge.py`: per-synapse `cp_btsp_theta`, `None` ⇒ the scalar cfg value ⇒
byte-identical; the kernel already takes θ elementwise). Layer 1 keeps θ=0.012; layer 2 gets its own θ calibrated to
the **measured** CA1 eligibility scale:

| pathway | presynaptic eligibility (measured) | θ |
|---|---|---|
| layer 1 `pos→ca1` | 0.00052 – 0.02310 | 0.012 |
| layer 2 `ca1→l2` | 0.0000000 – 0.00084 | 0.00045 |

**Measured scale ratio: 27.4×** — confirming a single global θ cannot serve both layers.

**Result: still NO-GO.** Depression softened (`dw` −1289 → −685.8, so θ *is* doing its job) but
**`l2_delta_max = 0.00000` and `l2_peak = −1` on every seed** — the response to each of the four fields stays
`[0.0, 0.0, 0.0, 0.0]`.

## The deeper blocker: the learned code has too little DYNAMIC RANGE to express graded learning

This is the C9 signature again, one layer up: **substantial weight change (`dw` = −686) with zero behavioural
change.** Combined with the gain measurement (L2 silent below `w0≈60`, responding at 60–150 with 4/20 bins active),
the picture is:

- CA1's learned code fires at **0.005 spikes/neuron/step** — extremely sparse.
- At that sparsity, L2's response to a field is effectively **all-or-none**: it fires when some CA1 field is active
  and not otherwise.
- Graded weight changes therefore produce **no graded firing change** — there is no dynamic range for learning to
  express itself in.

⇒ **Stacking is blocked by the SPARSITY of the learned representation, not (only) by the credit rule or by θ.** The
same property that makes the layer-1 field crisp and localized — very sparse, near-binary output — leaves the next
layer nothing to modulate. Per-pathway θ remains a correct and necessary fix (kept, default-off, byte-identical), but
it does not address this.

## Revised next levers

1. **Increase the learned code's dynamic range** — more CA1 neurons per field and/or a baseline that lets CA1 fire
   *gradedly* rather than near-binary. Note rung 1 measured the analogous bind at layer 1 (silent below W0≈2,
   saturated above 5); layer 2 inherits it and compounds it.
2. **Read the graded conductance rather than spikes.** This project has repeatedly found that graded/analog reads
   succeed where spike-rate reads hit the point-neuron wall — and gap#1's M1 result this same day is exactly that
   (the on-bridge WKV state works *because* it is held in a graded conductance, not a firing rate). The same move may
   apply here: let L2 read CA1's graded plateau/conductance instead of its sparse spikes.
3. Only after dynamic range is addressed does re-testing the credit rule across layers become informative.

**Honest status:** rungs 1–2 stand (6-seed, blind-clean). Rung 3 is a NO-GO with the blocker now localized to
**representation sparsity / dynamic range**, with per-pathway θ implemented and eliminated as the (sole) cause.
