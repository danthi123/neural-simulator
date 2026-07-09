# D3 rung 3 — the honest GENUINENESS boundary: end-label supervision alone does NOT length-generalize; the discrete-attractor's length-generalization is CONTINGENT on the per-step transition supervision (the credit-assignment wall persists)

**Date:** 2026-07-09
**Runner:** `research/runners/_d3_endlabel_supervision_derisk.py` (straight-through re-discretization rollout, end-label CE; numpy; NO `sim/` edit).
**Verdict:** honest NEGATIVE (a precise scope delineation of the D3 discrete-attractor result).

## The question
The rate + spiking D3 GOs trained the transition on teacher-forced (state,input)→next-state triples — a strong per-step signal. The adversarial genuineness question: is the composition LEARNED, or a taught DFA rolled out? Does the discrete-attractor's clean-state re-discretization make recurrent END-LABEL credit tractable where the continuous RNN failed?

## The result — end-label alone does NOT length-generalize (either architecture)
Full rollout with a STRAIGHT-THROUGH re-discretization (forward = hard argmax attractor `emb[argmax]`; backward = softmax-weighted, so gradient flows through the discrete state), the final property read from the final state, trained with ONLY the end-of-sequence property label (CE, BPTT through the rollout). S3, seed 42, train len 1/2/3, held-out-DEEPER 4/5/6 (chance 0.5):

| arm (END-LABEL only) | same | DEEPER |
|---|---|---|
| **DISCRETE-attractor** (straight-through) | 0.779 | **0.552** |
| CONTINUOUS control (soft state carried) | 0.931 | 0.627 |

**Neither length-generalizes** (DEEPER 0.55–0.63 ≈ chance). And the discrete-attractor is actually WORSE at same-length (0.779 vs the soft control's 0.931) — the straight-through hard-argmax is a rougher gradient path, HARDER to train from end-label. ⇒ **the re-discretization does NOT make end-label credit tractable; it HELPS length-generalization only WHEN the transition is already learned (via per-step supervision), and it HURTS end-label training of the transition itself.**

## What this precisely establishes (the honest scope of the D3 result)
- **The discrete-attractor mechanism length-generalizes recurrent multi-hop composition GIVEN a learned per-step transition** (the rate + spiking GOs: S3 0.999, theorem-backed A5 0.996, both halves on spikes). The re-discretization prevents drift → arbitrary depth.
- **But LEARNING that transition end-to-end from END-LABEL alone remains the credit-assignment wall** — the per-step transition supervision (teaching the group-mult DFA table) is LOAD-BEARING; end-label credit through the rollout does not discover the length-generalizing transition (for either the discrete or the continuous architecture).
- So the honest, precise claim: the discrete-attractor is the mechanism for length-generalizing composition **execution / representation** (given the DFA), NOT (by itself) a solution to **learning the DFA from weak (end-label) supervision**. The two are separable; D3 solved the first.

## The next mechanism (per the workflow — a boundary is the next undiscovered mechanism)
End-label / weak-supervision learning of the transition is the residual. Candidate mechanisms (research-gate the choice): (a) a CURRICULUM that grows supervision sparsity (per-step → every-k → end-label); (b) SELF-SUPERVISED transition learning (predict-the-next-input-consistency, or the group's algebraic closure as a constraint); (c) REWARD-modulated / RL credit for the discrete rollout (the discrete states suit tabular-RL-style credit); (d) the biological answer — the hippocampal/striatal system LEARNS sequences with sparse reward via replay + eligibility traces (the project's own SWR-replay + eligibility machinery). This is the genuine open problem (sparse-supervision sequence learning), distinct from the SOLVED execution mechanism.

## Landing
The D3 arc is comprehensively + honestly mapped: the discrete-attractor mechanism (= the project's CA3 substrate) length-generalizes recurrent multi-hop composition and is FULLY realized on spikes (transition via LIF hidden + re-discretization via FS-WTA) — GIVEN a learned transition; learning that transition from end-label alone is the honest residual (the sparse-supervision credit wall), the next mechanism to find. NO `sim/` edit anywhere in the D3 arc.

## Files
`research/runners/_d3_endlabel_supervision_derisk.py`; the D3 arc `2026-07-09-D3-*.md`.
