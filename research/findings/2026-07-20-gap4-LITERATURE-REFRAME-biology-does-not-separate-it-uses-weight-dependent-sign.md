# gap#4 — LITERATURE REFRAME: biology does NOT separate the signals. It makes the SIGN depend on current weight.

A read-in-depth literature check (Bittner 2017 Science; Milstein 2021 eLife; Rich-Liaw-Lee 2014 Science) was
dispatched **before** rung 6's result was interpreted, precisely so the interpretation could not be motivated by it.
It returns a reframe that overturns the arc's premise — and corrects one of my own refutations.

## 1. The geometric hypothesis was not just falsified — it was BACKWARD

| quantity | value | status |
|---|---|---|
| BTSP field-peak offset from plateau | **~19.5 +/- 4.7 cm** (~0.8 s @25 cm/s) | measured (Milstein 2021, n=26) |
| BTSP potentiation backward extent | **~75-150 cm** (~3-6 s) | measured (Bittner 2017: ramp starts 3.8 +/- 0.2 s before) |
| CA1 place-field spacing between cells | **Poisson / uniform / uncorrelated** — mean gap = 1/density; ~4.7 cm even in a 253-cell *sample*, sub-mm at true density | measured (Rich 2014: 0/61 cells deviate from Poisson) |

**Real spacing is SMALLER than the backward shift — by an order of magnitude under the most generous sampling
assumption, and by three-plus orders at population density.** Field locations are a spatial Poisson process, so
**the modal gap between neighbouring fields is ZERO**. My evenly-spaced-4-bins layout has no empirical basis; the
paper that uses it (Front. Comput. Neurosci. 2021) adopts equal spacing explicitly for tractability.

⇒ Rung 6's falsification is **confirmed from the other direction and strengthened**: widening spacing could never
have been the answer, because biology's spacing is far *tighter* than mine. **The six mechanisms did not fail
because my geometry was unrealistic. They failed because they were all pursuing a separation biology never achieves.**

## 2. What biology actually does — and it is NOT feedback inhibition

Milstein 2021, measured:
- **"weak inputs potentiate, and strong inputs depress"** — the *sign* of the weight change is set by the synapse's
  **current weight**, not by its timing or its lag.
- The evidence is stark: **dVm vs initial Vm correlates at r = -0.91**, while **final Vm vs initial Vm correlates at
  r = 0.04**. The final field shape is essentially independent of the starting state — a genuine fixed point.
- BTSP is therefore **"inherently stable, converting synaptic potentiation into depression when input strengths
  exceed a particular range, whereas most models of Hebbian learning require additional homeostatic mechanisms."**
- A plateau near an existing field **translocates** it rather than creating a competing one.
- Changes in *inhibitory* weights are **explicitly ruled out** (CA1 interneurons show low spatial selectivity).
- The feedback-inhibition circuit model governs **which cells plateau** (allocation across the population) — **not
  which synapses within a cell potentiate**. ⇒ **The one route I had left as "remaining" was mis-scoped**: it does
  not address the within-cell problem at all.

## 3. ⛔ MY PF-1 REFUTATION TARGETED THE WRONG CLAIM

PF-1 measured the weight axis at **1.093x** between adjacent-lag and field-forming synapses and I recorded it as
too weak — "a 1.093x axis cannot deliver the >2x weight contrast the transfer loss demands."

**That tested weight as a SEPARATOR. The mechanism does not use it as one.** Weight-dependent plasticity does not
try to tell adjacent from forming; it makes *every* strong synapse depress and *every* weak synapse potentiate, so
the field converges to a target shape **regardless of which inputs were eligible**. A separation ratio of 1.093x is
simply not the quantity that governs whether that works. **My measurement was correct and my inference from it was
wrong**, and it may have caused me to under-rate the one candidate whose second axis was the right one.

## 4. What this changes, concretely

1. **Stop pursuing separation.** Seven routes closed; the literature says the target was never achievable. That
   reframes the whole arc from "seven failures" to "a systematically wrong objective, now identified."
2. **Implement weight-dependent bidirectional plasticity with a fixed point** — `W_target = k+Q+/(k+Q+ + k-Q-)`,
   weak potentiate / strong depress. This is what none of the seven provided (the closest, Milstein two-sigmoid,
   had it as a secondary axis I then dismissed for the wrong reason).
3. **Randomize field centres** (Poisson/uniform), not even spacing — my layout may be generating its own artifacts.
4. **Drop the feedback-inhibition route** as a within-cell fix; it is a population-allocation mechanism, modelled
   not measured, and does not address this problem.

## 5. Honest caveats carried from the source

- The **sign** of the 19.5 cm peak offset is *inferred*, not directly quoted: Milstein reports an unsigned "average
  distance", and backwardness comes from the 2.2:1 before/after asymmetry plus Bittner's qualitative statement.
- Population field-density figures are **inferred arithmetic** from measured recruitment fractions — order of
  magnitude, not digits.
- The agent flags a genuine transfer risk: at my ~5-cell density the depression term will bite differently than at
  CA1 density, so the weight-dependent rule must be validated on its own terms rather than assumed to port.

## 6. Process note

This is the single most valuable input of the arc, and it arrived because the question — *is real place-field
spacing greater than the BTSP shift?* — was dispatched **before** the result it would interpret. Had I run it after,
the answer would have been available to rationalize whichever way rung 6 came out.
