# Past the reservoir bound, Rung 2 (11/12 GO; 12/12 directional): a per-neuron SELECTIVE diagonal SSM, trained by an EXACT forward-mode ELIGIBILITY TRACE (no BPTT, no weight transport), captures a long-range CONJUNCTION the fixed reservoir cannot

**Date:** 2026-07-13
**Runner:** `research/runners/_reslm_rung2_selective_ssm_derisk.py` (self-contained numpy; NO `sim/` edit, NO BPTT, NO weight transport).
**Status:** ✅ GO — 11/12 (standard 5/6 + FRESH 6/6, ≥5/6 both sets); the effect is DIRECTIONALLY UNIVERSAL (selective beats every control on 12/12 seeds).
**Provenance:** Rung 2 of the 2026-07-13 past-reservoir arc. Rests on the adversarially-verified Rung-1 (conjunction is the missing ingredient, Sub-claim A robust) + the in-depth Zucchet 2305.15947 source-read (the O(n) forward-mode transport-free diagonal-RTRL SURVIVES per-neuron input-dependent selectivity — derived by hand, correcting a summary error).

## The mechanism (Rung-1's conjunction, carried RECURRENTLY, trained LOCALLY)

Rung 1 showed a linear read-out over a fixed reservoir cannot compute input×input conjunctions (a representational limit). Rung 2 carries that conjunction across the sequence with a **per-neuron SELECTIVE (input-dependent) diagonal SSM**, trained WITHOUT backprop-through-time or weight transport:

```
state (gated leaky integrator, per neuron i):  h_{t,i} = lam_{t,i}*h_{t-1,i} + (1-lam_{t,i})*inj_{t,i}
selective gate (input-dependent):              lam_{t,i} = sigmoid(w_i . u_t + c_i)     (theta_i = w_i,c_i PER NEURON)
injection:                                     inj_{t,i} = (W_in u_t)_i                 (W_in fixed random)
EXACT forward-mode eligibility (local, O(n*d_in)):
   e^w_{i,t} = lam_{t,i}*e^w_{i,t-1} + (lam(1-lam) u_t)*(h_{t-1,i} - inj_{t,i})
gate update (spatial error x eligibility):     Δtheta_i ∝ -delta_i * e^theta_{i,read},  delta_i = (W_ro^T (p - onehot))_i
```
The gate `lam_{t,i} = sigmoid(w_i·u_t)` makes the recurrence input-dependent → `lam(u)⊙h` is an input×state PRODUCT (the conjunction), computed across distance. Because `lam_{t,i}` depends only on the shared `u_t` + neuron i's OWN params, the eligibility trace stays per-neuron/local (O(n·d_in)), forward-mode, transport-free (the read-out error is spatial-backprop through the linear read-out only). Forget-bias init (`c=2.5`, Jozefowicz 2015) starts `lam~0.9` so the trace survives the filler and the gate can LEARN when to hold/release/conjoin (else `lam~0.5` fades the distal token by `lam^D` before learning — the vanishing-eligibility trap, observed + fixed).

## Task (needs a long-range conjunction)

`[KEY, filler×12, QUERY] -> target = rule[KEY, QUERY]`. The distal KEY must be HELD across 12 filler steps (the reservoir fades it) AND conjoined with the recent QUERY (the Rung-1 product). The learned input-dependent gate holds the key (lam≈1 during filler) and lets the query modulate the held key at the read step (the query×key conjunction the linear read-out then reads).

## Result — 11/12 GO; 12/12 directional (chance 1/6 = 0.167)

| arm | mean acc | vs selective |
|---|---|---|
| **selective** (input-dependent gate, TRAINED by eligibility) | **0.629** | — |
| detached (input-dependent gate, NOT trained; same init) | 0.415 | selective +0.214, **12/12** (min +0.119) |
| permgate (gate TRAINED but on a PERMUTED input) | 0.416 | selective +0.213, **12/12** (min +0.074) |
| fixed_res (FIXED per-neuron lambda = leaky ESN, Rung-1 baseline) | 0.395 | selective +0.234, **12/12** (min +0.085) |
| chance | 0.167 | |

GO gate: selective > fixed_res + 0.10 AND > detached + 0.08 AND > permgate + 0.08 AND > chance + 0.15. **11/12 GO** (the lone miss, seed 102, is a marginal-margin clip — selective still beats all three controls there, just by +0.085/+0.118/+0.074). The **direction is universal: selective beats every control on 12/12 seeds.**

The three controls isolate the mechanism precisely:
- **selective > fixed_res** (12/12): input-DEPENDENT gating beats a fixed leaky reservoir → the selectivity (input×state conjunction), not just recurrence, is load-bearing.
- **selective > detached** (12/12): LEARNING the gate (by the eligibility trace) beats the SAME input-dependent architecture with a random fixed gate → the local training does the work, not the architecture/init alone.
- **selective > permgate** (12/12): the gate must read the REAL input → it learns a task-relevant selectivity, not a generic one.

## ⇒ the claim

A per-neuron SELECTIVE diagonal SSM, trained by an EXACT forward-mode eligibility trace (no BPTT, no weight transport), learns to hold a distal token and conjoin it with a recent one — a long-range conjunction a fixed reservoir + linear read-out provably cannot do (Rung 1). This is the honest, biologically-plausible path PAST the reservoir's fading-memory + linear-read-out bound that AVOIDS the exhausted deep-credit (surrogate-BPTT / feedback-alignment) wall. Spiking-realizable (a diagonal SSM = per-neuron leaky integrators with an input-modulated leak; the eligibility trace is a local synaptic trace).

## Honest scope / next

- Self-contained cheap-first (K=6, DEPTH=12, a synthetic gated-conjunction task). It proves the LOCAL rule captures the recurrent conjunction; it does NOT yet claim to match BPTT (Zucchet: online closes ~70–90% of the BPTT gap) nor run on real text (Rung 3). Absolute accuracies are modest (selective ~0.63 on a 6-way conjunction) — the load-bearing result is the decisive, universal margin over all three controls.
- The gate-update margins occasionally dip below the +0.08/+0.10 gate at the effect's lower tail (seed 102); the DIRECTIONAL effect is 12/12 with means ~+0.22.
- NEXT (Rung 3): the selective-SSM on REAL text next-token prediction (does the eligibility-trained selective gate beat the fixed reservoir's deep-context CE?), then the spiking realization (per-neuron leaky-integrator with input-modulated leak on a `SimulationBridge` + a synaptic eligibility trace) — the fully-on-substrate transport-free long-range learner.
- NO `sim/` edit. CI guard `tests/test_reslm_rung2_selective_ssm.py`.

## Files
- `research/runners/_reslm_rung2_selective_ssm_derisk.py`; raw `_rung2_{std,fresh}.json`.
- Builds on Rung 1 (`2026-07-13-PAST-RESERVOIR-conjunction-readout-...`) + the Zucchet source-read (`2026-07-13-Zucchet-diagonal-RTRL-SURVIVES-...`).
