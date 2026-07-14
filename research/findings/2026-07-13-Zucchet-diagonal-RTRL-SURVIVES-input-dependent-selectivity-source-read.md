# Source-read (Zucchet 2305.15947, in depth): the O(n) forward-mode transport-free diagonal-RTRL SURVIVES per-neuron input-dependent (SELECTIVE) gating — the load-bearing green-light for Rung 2, correcting a summary error

**Date:** 2026-07-13
**Source:** Zucchet, Meier, Schug, Mujika, Sacramento, "Online learning of long-range dependencies," NeurIPS 2023, arXiv:2305.15947 (read via ar5iv, equations extracted + the crucial claim re-derived by hand).
**Why:** Rung 2 of the past-reservoir arc (recurrent selective diagonal gating trained by exact diagonal-RTRL) rests on the claim that the O(n) diagonal-RTRL cheapness SURVIVES when the transition λ becomes INPUT-DEPENDENT (Mamba-style selectivity). Per the read-the-source-in-depth discipline (`feedback_read_sources_in_depth_not_skim`) I read the equations myself — and a small-model summary got the load-bearing point WRONG, so this is exactly a case the discipline is for.

## What the paper establishes (fixed-λ case)

Diagonal linear recurrent unit (LRU): `h_{t+1} = λ ⊙ h_t + γ ⊙ B x_t` (complex diagonal `λ ∈ ℂ^N`). The parameters are trained ONLINE by forward-propagated eligibility traces:
```
e^λ_{t+1} = λ ⊙ e^λ_t + h_t
Δλ ∝ Σ_t δ_t ⊙ e^λ_t          (δ_t = ∇_{h_t} L_t, the SPATIAL/local error)
```
O(n) memory per step (not the O(n²)/O(n³) of dense RTRL), FORWARD-mode (no BPTT), and transport-free (local eligibility trace × local error — no backward weight matrices, unlike feedback alignment). The diagonal structure is the key: "recurrent neurons within a layer are independent — the recurrent parameters of a given neuron do not impact other neurons," so the sensitivity tensor is diagonal (element-wise `λ ⊙ h` has zero cross-derivatives). Validated to 4000-step LRA / copy; online closes ~70–90% of the BPTT gap; degrades with depth (the inner-layer spatial-backprop approximation).

## The load-bearing question + the CORRECTION (derived by hand)

The paper does NOT discuss input-dependent/selective `λ(u_t)`. A small-model summary claimed this "breaks the analysis — the diagonal independence is lost, sensitivities revert toward O(n²)." **That is WRONG for the case Rung 2 needs**, and reading the recurrence + deriving the sensitivity shows why.

Take a **per-neuron selective** diagonal SSM (the standard selective-SSM / Mamba per-channel structure): neuron i's transition depends on the input and neuron i's OWN parameters θ_i only:
```
h_{t+1,i} = λ_{t,i} · h_{t,i} + b_i · u_t ,     λ_{t,i} = f(u_t ; θ_i)
```
The eligibility trace of neuron i w.r.t. its own gate params θ_i:
```
e_{i,t+1} = ∂h_{t+1,i}/∂θ_i + (∂h_{t+1,i}/∂h_{t,i}) · e_{i,t}
          = (∂λ_{t,i}/∂θ_i) · h_{t,i}  +  λ_{t,i} · e_{i,t}
```
Every term is LOCAL to neuron i: `λ_{t,i}`, `h_{t,i}`, `θ_i`, and the SHARED input `u_t` (inside λ). **No neuron j appears.** The cross-neuron sensitivity `∂h_{t,i}/∂θ_j = 0` for i≠j is UNCHANGED — because neuron i's state depends on the input via its OWN gate, not on any other neuron's state or params. So:

- **Diagonal independence is PRESERVED.** The Jacobian stays diagonal (the input-dependence enters through the shared `u_t`, which does not couple neurons).
- **The eligibility trace stays O(n) in the hidden size** — it becomes a per-neuron VECTOR of size `d_in` (the gate's input-projection dimension) instead of a scalar, so total O(n · d_in), linear in n (d_in is a constant factor, not n).
- **Still forward-mode + transport-free** (the extra term `(∂λ_{t,i}/∂θ_i)·h_{t,i}` is a local Jacobian-vector product of neuron i's own gate, no backward pass).

The summary conflated "input-dependent" with "cross-neuron coupling." Input-dependence via a shared `u_t` + per-neuron gate parameters does NOT couple neurons; only a DENSE (non-diagonal) or cross-neuron-parameterized transition would. (Caveat/scope: this holds for per-state-dim selectivity — λ_{t,i} a function of u_t and θ_i alone. A selectivity that mixes state dims, or a shared-across-dims Δ that then multiplies a dense A, would reintroduce coupling; Rung 2 uses the per-neuron form, which is also the Rung-1-validated conjunction structure carried recurrently.)

## ⇒ Rung 2 is mechanistically green-lit

A **per-neuron selective diagonal SSM** — `h_{t,i} = λ_{t,i}·h_{t-1,i} + b_i·u_t`, `λ_{t,i} = f(u_t;θ_i)` — computes input×state MULTIPLICATIVE conjunctions (the Rung-1 ingredient) ACROSS the sequence, and is trainable by an EXACT per-neuron forward-mode eligibility trace `e_{i,t+1} = λ_{t,i} e_{i,t} + (∂λ_{t,i}/∂θ_i) h_{t,i}` — O(n·d_in), NO BPTT, NO weight transport. This is the honest path past the reservoir bound that avoids the exhausted deep-credit wall. Spiking-realizable (a diagonal SSM = per-neuron leaky integrators with input-modulated leak).

**Rung-2 de-risk (next build, pending the Rung-1 adversarial-verify):** on a conjunction-requiring next-token task, compare (a) the fixed-λ diagonal reservoir + linear read-out (Rung-1 baseline) vs (b) a per-neuron SELECTIVE-λ diagonal SSM trained by the eligibility trace above (learn the gate params θ_i + the read-out locally). Single variable = λ fixed vs input-dependent. Anti-cheat: an eligibility-detached control (stop-grad the trace) + permuted gate-input. GO iff the selective-λ eligibility-trained SSM beats the fixed-λ reservoir at deep context, confirming the recurrent conjunction is captured by the LOCAL rule.

## Files
- Read of arXiv:2305.15947 (equations 8–17). Follows `2026-07-13-PAST-RESERVOIR-conjunction-readout-...` (Rung 1) + the 2026-07-13 fresh-mechanism-class research gate.
