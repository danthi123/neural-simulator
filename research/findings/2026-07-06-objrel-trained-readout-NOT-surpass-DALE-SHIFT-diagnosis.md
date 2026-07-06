# objrel trained-graded-spiking-readout — NOT a surpass (adversarial-verify #2 caught it) — BUT it CONVERGES the arc on the real root cause: the DALE-SHIFT, not a "sub-1% margin"

**Date:** 2026-07-06
**Runner:** `research/runners/_rungB1c_objrel_trained_spiking_readout_derisk.py`
**Raw:** `research/findings/raw/_rungB1c_objrel_trained_spiking_readout.json`
**Verdict:** the 6-seed-blind "GO" is **NOT a valid surpass** — a mandated adversarial-verify workflow (5 skeptics + a synthesizer running its own controls) INVALIDATED the mechanistic framing. Kept as an honest correction + the decisive diagnostic convergence of the whole objrel arc.

## What was claimed vs what the adversarial-verify found

The de-risk trained a spiking read-out (warm-started from the closed-form ridge, deployed as graded/low-gain output LIF, BPTT fine-tuned) and read the objrel role from the output LIF **summed spike count** — reporting objrel 0.5→1.0 + canonical 0.44→1.0, 6-seed-blind, all anti-cheats True, framed as "a GRADED spike-count read resolves the sub-1% margin the saturating WTA quantizes away." The adversarial-verify confirmed the read is genuinely spike-count-based (the LITERAL prior confound — host argmax vs spiking WTA — is NOT repeated), but REFUTED the mechanism with controls it actually ran:

1. **BPTT is inert.** Ridge warm-start with **ZERO BPTT epochs** already gives objrel 1.0 / canon 1.0 with the same spike counts. The "trained through the spike nonlinearity" is a veneer; the nonlinear hidden is also inert (H=0 == H>0 == the ridge's held-out 1.0). The read is a monotone re-expression of `argmax(feature · W_ridge)`.
2. **The "sub-1% margin" premise is FALSE.** The actual ridge scores on objrel-slot0 are `[0.25, 0.0, 0.75]` — the THEME winner beats by a **66% relative margin**, trivially separable. There is no intrinsic sub-1% margin. (A genuinely sub-1% margin would TIE the coarse LIF f-I step and FAIL the graded read — so "graded spike-count preserves a sub-1% margin" is not even the operative mechanism.)
3. **The real lever is SIGNED (Dale-ILLEGAL) weights, not the graded regime.** The warm-start weights are signed (28% negative). Taking those EXACT weights and **Dale-shifting** them (`W − W.min()`, which is what the fixed spiking WTA is FORCED to do to make its read-out synapses excitatory) collapses objrel to 0.0. So the beat over the fixed spiking WTA is because the read **avoids the Dale offset** (which dilutes the 66% margin down to the WTA's ~1-3% residual) + drops mutual inhibition — NOT because a graded spike count preserves anything the WTA saturates.

⇒ genuinely spiking in deployment, but the spiking does **no computational work beyond re-expressing the host ridge argmax**, and it only beats the WTA by using **Dale-illegal signed weights** (a single read-out neuron with both + and − output synapses — biologically illegal). NOT a valid surpass.

## The decisive CONVERGENCE (the real value — this reframes the entire arc)

The whole objrel arc was framed around a **"sub-1% margin the spiking read can't resolve."** The adversarial-verify shows that framing was an **artifact of the DALE-SHIFT**, and BOTH this de-risk AND the RANK-2 opponent accumulator converge on the true root cause:

- The reservoir feature holds a **BIG (66%) objrel margin** — the info is richly present (host-decodable).
- **Dale's law forces the spiking read-out to be EXCITATORY** (the read-out synapses must be positive), so the c2 read Dale-shifts the signed ridge weights (`W − W.min()`). **The Dale-shift DESTROYS THE SIGN** — and objrel's THEME evidence lives in the NEGATIVE ridge rows — diluting the 66% margin to the ~1-3% residual the WTA then can't resolve. The "sub-1% margin" was never intrinsic; it is manufactured by the Dale-shift.
- **A SIGNED read recovers objrel** (both this de-risk's signed weights AND RANK-2's signed opponent accumulator get objrel → 1.0). But:
  - **Signed weights DIRECTLY** (this de-risk) → Dale-ILLEGAL (a neuron can't have both excitatory + inhibitory outputs).
  - **Signed via an inhibitory RELAY** (RANK-2, Dale-legal) → SEE-SAWS: recovers objrel but collapses canonical, because the pooled inhibitory relay computes `g(ON) − g(OFF) ≠ g(ON − OFF)` — it can't reproduce the per-neuron signed subtraction, and the canonical per-role bias INTERCEPT can't be delivered cleanly through it.

**⇒ THE GENUINE RESIDUAL (precisely located): a DALE-LEGAL signed read — proper GABAergic inhibitory-interneuron circuitry (the striatal read-out biology, Kandel Ch 38; the deep-source read this session) — that delivers the per-neuron signed subtraction AND the canonical intercept without the see-saw.** Not "resolve a sub-1% margin"; **carry the negative rows through Dale-legal inhibition.** This is a much sharper, correct target than the arc's prior framing, and it points straight at the striatal interneuron circuitry (Tepper PV-FSI/NPY/cholinergic) + the dendritic plateau (a nonlinear read that could hold both) — the mechanisms the in-depth Kandel read surfaced.

## Process note (the discipline is load-bearing)

This is the SECOND false "surpass" the mandated adversarial-verify caught this session (after the per-role host-argmax confound). Both would have entered the record as surpasses without it. It also did what a rubber-stamp never would: it ran its own controls (0-epoch warm-start, the actual ridge scores, the Dale-shift ablation) and thereby CONVERTED a false GO into the correct diagnosis. This is exactly why the `neural-simulator` skill now mandates it for any surpass claim.

## Files
- `research/runners/_rungB1c_objrel_trained_spiking_readout_derisk.py` — the de-risk (genuinely-spiking read, but ridge-argmax re-expression via Dale-illegal signed weights; NO sim/ edit).
- `research/findings/raw/_rungB1c_objrel_trained_spiking_readout.json` — the 6-seed record + the honest controls (0-epoch, linear-H0, Dale-shift ablation).
