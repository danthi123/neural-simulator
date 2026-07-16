# MDGL population-coded spiking port — the boost is a MAGNITUDE CONFOUND, not recovered directional off-diagonal credit (honest NEGATIVE; the sign-flip anti-cheat caught it); the toy testbed is exhausted for the spiking directional question

**Date:** 2026-07-15
**Status:** NEGATIVE (honest) — the named "population-coding surpass" does NOT cleanly recover the directional off-diagonal credit on this toy. The rate mechanism stays validated; the real test moves to the on-bridge substrate.
**Runners:** `research/runners/_mdgl_popcoded_spiking_derisk.py` (created), `_mdgl_spiking_port_derisk.py`, `_mdgl_offdiagonal_credit_derisk.py`

## What was tested

The single-neuron LIF port of MDGL ports the off-diagonal mechanism but DEGRADED (+11% of the diagonal-vs-BPTT gap vs the rate net's +48–64%; root cause = eligibility from sparse binary spikes). The **named surpass** was POPULATION CODING — the exact lever that closed FEEDFORWARD spiking credit this session (e-prop K=1 0.47 → K=8 0.877 ≈ LIF ceiling). Built it: each logical unit = POP_K spiking neurons (hard 0/1 spikes, per-neuron bias tiling the threshold), the unit output = the POOLED spike-rate r_u = mean_k s_{u,k} (a graded, low-variance signal), and the recurrence + eligibility + MDGL Γ all operate at the unit level on r.

## The result — pop-coded MDGL boosts, but the sign-flip anti-cheat REFUTES directionality

**pop_K sweep (seed 42, XOR T=8), gamma_gain 0.4:** MDGL rises with K (K=1 0.42 → K=8 **0.700** → K=16 0.66) — a large boost vs e-prop 0.370. So pooling DOES lift the number. BUT the pop-coded **BPTT ceiling is broken** (0.28–0.40 ≈ chance — my pop-coded BPTT backward drops the membrane-through-time gradient, a real bug in that arm) → no valid ceiling, and the whole net trains from a near-chance e-prop baseline.

**The decisive gain sweep (K=8, MDGL vs sign-flipped Γ vs e-prop 0.370):**

| gain | MDGL (correct sign) | sign-flip Γ | verdict |
|---|---|---|---|
| 0.05 | 0.375 | **0.425** | flip BETTER |
| 0.1  | 0.455 | **0.535** | flip BETTER |
| 0.2  | 0.550 | **0.670** | flip BETTER |
| 0.4  | 0.700 | 0.630 | correct edges ahead (marginal) |

**If the off-diagonal DIRECTION were load-bearing, correct-sign must beat sign-flip at every gain.** It does not — the FLIPPED Γ wins at 3 of 4 gains. ⇒ the pop-coded MDGL boost is a **magnitude / capacity effect** (a large extra gradient signal that helps training regardless of sign), NOT the recovered directional cross-neuron off-diagonal credit. The anti-cheat did its job.

## Root cause (systematic-debugging Phase 1) — a degenerate training-regime artifact, not a spiking-fundamental

The pop-coded net sits in a near-critical, hard-to-train regime: e-prop is at 0.370 (≈ chance 0.25), the pop-coded BPTT ceiling is broken (0.325). In that regime the net is so under-trained that ANY gradient magnitude lifts it off chance → the Γ's MAGNITUDE swamps its DIRECTION. On the clean RATE net the e-prop baseline was proper and the Γ direction mattered (sign-flip HURT, +48–64% clean). So this is a **testbed artifact of the fragile toy**, not evidence the mechanism fails on spikes — but it means the toy CANNOT cleanly answer the spiking directional question (3 issues stacked: broken pop-coded BPTT + near-critical dynamics + magnitude-confounded Γ = the "3+ fixes → question the architecture" signal; the toy is the wrong architecture for the spiking off-diagonal test).

## What stands / what this launches (a NEGATIVE is a next-mechanism trigger, not an endpoint)

- **The core science is UNCHANGED + validated:** the learned-cortex blocker = recurrent OFF-DIAGONAL cross-neuron temporal credit; the biological fix = MDGL cell-type one-hop neuromodulation; **clean directional +48–64% on the rate trainable-RNN** (sign-flip hurts, zero-Γ collapses, permuted chance). The owner's "what are we missing?" is answered at the mechanism level.
- **The spiking port** ports degraded on the single-neuron LIF (+11%); population coding does NOT cleanly amplify it on this toy (magnitude confound). The real substrate test is the **on-bridge Izhikevich realization** where the population coding is REAL (a `BrainRegion` of many neurons per logical unit) and the training regime is the actual substrate — via the existing per-synapse-DA path (`cp_synapse_action_tag` + `compute_per_synapse_da_signal`, `bridge.py:7393`): 6 modulators = per-cell-type error broadcast, tag = presynaptic cell type, × the on-bridge eligibility. That is the next build.
- **NO `sim/` edit.** The pop-coded runner + this negative are committed to both remotes.
