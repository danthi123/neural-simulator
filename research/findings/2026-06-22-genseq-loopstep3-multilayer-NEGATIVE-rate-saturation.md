# Loop-step 3 de-risk #1 — multi-layer DENSE consolidation hits the RATE-SATURATION WALL (honest NEGATIVE, the load-bearing Q3 finding) (2026-06-22)

**Scope:** the first de-risk of loop-step 3 (consolidate the converted spiking generator onto the ONE bridge) —
extend step-0's single-layer load to MULTI-LAYER + SIGNED weights on an MLP-only slice (no attention).
`research/runners/_genseq_loopstep3_multilayer_signed_derisk.py`, GPU. **NO `sim/` edit.** On `main`.

## Result — NEGATIVE (mechanistically pinned, cheap-fixes-exhausted)
3 stacked signed MLP blocks (66→2048→2048→2048, ~50% negative weights, E/I split-channel) as co-resident bridge
slices (10,372 neurons, 12.85M synapses), driven with REAL (non-one-hot) activations:

| block | input | Spearman vs off-bridge forward | on_max_rate |
|---|---|---|---|
| 0 (66→2048, one-hot, SIGNED) | sparse | **0.321** (above chance — signed routing partially faithful) | 0.75 |
| 1 (2048→2048, dense) | dense | **−0.019** (CHANCE) | **0.500 ← SATURATED** |
| 2 (2048→2048, dense) | dense | **0.009** (CHANCE) | **0.500 ← SATURATED** |

Cumulative Spearman **0.009**; specificity margin **0.000**. (Step-0's positive-only single layer was 0.92.)

## The mechanism (pinned, not fudged)
The signed E/I split DOES carry rank at layer 0 (0.32 — degraded from 0.92 by the signed split, but well above
chance, so signed routing is partially faithful). But every DENSE hidden→hidden stage SATURATES to the refractory
ceiling (`on_max_rate` pinned at 0.5 = the 1/dt refractory cap; off-bridge max is 1.0): with ~1000+ active sources
fanning into each target at supra-threshold drive, the per-neuron net-input variation that should encode the signed
sum is SWAMPED → every target pins at the ceiling → rank destroyed. **This is the documented RATE-SATURATION WALL
(the rate-code boundary family.)** Cheap fixes EXHAUSTED, all NEGATIVE for the dense stage: global-gain sweep
(0.25→256×), per-block geometric-bisection threshold-balance (block-1 driven to the 0.2 lower bound — still
2048/2048 active), and an I/E gain-ratio sweep (≤16× extra inhibition — block-1 stays pinned at 0.5). No
NO-`sim/`-edit calibration recovers the hidden-layer rank.

## What this means — the research gate fires
⇒ The CHEAPEST consolidation path (rate-coded, no-edit) does NOT work for the multi-layer DENSE transformer — it hits
the rate-saturation wall (a CONFIRMED boundary of the rate-code family). The scoping ladder's step-#1 NO-GO gate
("fix calibration/per-layer balance BEFORE any attention work") has fired and the cheap levers are spent; the
attention `sim/` edit is correctly NOT worth committing until this is resolved. **Per the standing research gate
(confirmed boundary + known family), the next move is deep-research-first.** The candidate resolutions (to rank):
1. **Surrogate-grad-on-bridge finetune** — re-learn the on-bridge weights to compensate for the spiking dynamics (the
   parent scoping's §4.2 fallback; a learning step / guarded `sim/` use).
2. **Graded / sub-threshold drive regime** — keep neurons OUT of saturation (operate in the graded/analog regime the
   bridge supports), so the signed sum survives.
3. **Population coding** — the documented prior rate-code-wall lift (more neurons per feature → the population mean
   does not saturate).
4. **Lower fan-in / a non-saturating code** (temporal/phase), or a reframe of the on-bridge representation.

## Honest framing
This is the FIRST genuine roadblock in the generative-frontier arc — everything prior (the spiking CONVERT GO, the
P2 Claude-teacher KNOWLEDGE GO, the step-0 single-layer consolidation ENTRY GO) stands. Only the **multi-layer
bridge-consolidation of a dense transformer** hits this wall. It is real, documented, well-characterized, NO `sim/`
edit, cheap-fixes-exhausted → a robust NEGATIVE that redirects loop-step 3 to its deeper fallbacks (NOT a failure of
the convert or the knowledge half).
