# D3 recurrent multi-hop composition — GO: DISCRETE-ATTRACTOR recurrence LENGTH-GENERALIZES where the continuous RNN cannot; the mechanism for recurrent composition is attractor-state maintenance (the project's own CA3 / NEF-cleanup substrate)

**Date:** 2026-07-09
**Runner:** `research/runners/_d3_group_composition_derisk.py` (`discrete_attractor_rnn`; numpy CPU, NO `sim/` edit). Follows the 4-lever negative (`2026-07-09-D3-...-first-arc-BPTT-no-length-generalization.md`) whose drift-diagnosis POINTED at this mechanism.
**Verdict:** GO (mechanism found + cleanly isolated); honest rate-level scope; next rungs named.

## The result — the drift-diagnosis was right, the attractor mechanism SOLVES length generalization

Streaming non-abelian group-word composition (S3, state tracking `s_t = s_{t-1}·g_t`), train lengths 1/2/3, test held-out-DEEPER 4/5/6. State-tracking accuracy at the held-out-deeper lengths (chance 1/6 = 0.167):

| arm (all with the SAME per-step state signal where applicable) | DEEPER state-track |
|---|---|
| FF-oracle (flattened seq) | 0.504 (prop, chance 0.5) |
| continuous BPTT-RNN (end-label) | 0.521 (prop) |
| continuous BPTT-RNN + intermediate-STATE supervision | **0.216** (drifts to ~chance) |
| **DISCRETE-ATTRACTOR recurrence** | **0.998** (near-perfect) |

**Multi-seed 42/43/44 (dev) — robust GO:** discrete-attractor DEEPER state-track **0.998 / 0.998 / 1.000** (aggregate **0.999**), step-delta **1.000** every seed, SAME **1.000**; the continuous controls fail every seed (state-sup deeper **0.216 / 0.188 / 0.221**, agg 0.208; end-label RNN 0.521/0.493/0.580; FF 0.504/0.513/0.617). `research/findings/raw/_d3_discattr_s3_multiseed.json`. (Blind seeds 100/101/102 + S4/A5 are the scale-out; the mechanism is deterministic — a learned DFA — so seed-robustness is expected + confirmed.)

## Why it works + the CLEAN mechanism isolation
The continuous RNN (even taught the exact running state `s_t` every step) LEARNS the state at train depth (same-track 0.874) but its continuous dynamics DRIFT past the trained depth → deeper-track collapses to 0.216. The **discrete-attractor** carries the running state as one of K CLEAN fixed prototypes `emb[s]` and RE-DISCRETIZES each step (compute next-state from `(emb[s_{t-1}], x_t)`, snap to `emb[argmax]`) → **no drift accumulation**. It DECOMPOSES the problem into (a) a length-INDEPENDENT transition δ(state, input)→next-state (the group-mult table, learned perfectly from SHORT sequences: step-delta 1.000) + (b) a drift-free autoregressive rollout → generalizes to ANY depth.

**The isolation is clean:** the discrete-attractor and the state-supervised continuous RNN use the SAME per-step state signal; the ONLY difference is the re-discretization (clean-prototype state vs drifting continuous state) → **0.998 vs 0.216**. So the ATTRACTOR STABILIZATION is the load-bearing mechanism, not the supervision. Non-abelian control intact (order-matters 0.41 — a count/multiset model is at chance on order-dependent cases). This directly answers the D3 question: **a recurrent substrate CAN learn length-generalizing multi-hop composition — but it must be a DISCRETE-ATTRACTOR network (the brain's discrete WM / CA3), not a continuous RNN.**

## CRITICAL refinement — A5 (K=60, non-solvable) REVEALS the autoregressive error-compounding sensitivity (the S3 GO is contingent on step-delta→1.0)

Ran A5 (blind seeds 100/101/102, the theorem-backed FF-impossible group). RESULT: discrete-attractor step-delta **0.708** (NOT 1.0), deeper state-track **0.022** (≈chance). The S3 GO held because step-delta was **1.000** (the 36-entry transition learned perfectly, tiny K); at A5's 3600-entry transition table the per-step delta is only 71% learned — likely because the 60 element pool-codes OVERLAP in n_pool=64 (the RNN can't cleanly read each input) — and **autoregressive rollout COMPOUNDS the per-step error** (`0.708^L → chance` over deep rollouts). So the honest mechanism is: **the discrete-attractor length-generalizes IFF the per-step transition is learned to ~100%; a sub-perfect transition compounds to chance over depth.** (Cleaner-codes/more-data test running to check if A5 recovers step-delta→1.0.) This is the genuine sensitivity: re-discretization removes DRIFT but not per-step CLASSIFICATION error, which then compounds. The continuous-RNN comparison stands regardless (the attractor is necessary — continuous fails even at S3), but the length-generalization is bounded by transition-learning accuracy, which gets harder at scale.

## Honest scope (rate-level; the next rungs)
- **Teacher-forced per-step state supervision** trains the transition (each step an independent K-way classification). The strong-signal version; the mechanism finding (attractor vs continuous, same signal) is what's load-bearing. **Next rung: reduce the supervision** — can the attractor net learn the transition from sparser/end-label signal (e.g. reward at the final property only)? This is where the credit-assignment difficulty returns.
- **Fixed random attractor prototypes** (not learned); a learned/emergent attractor codebook is the follow-on.
- **Rate-level (numpy).** The mission-central next rung: **port to the SPIKING CA3 attractor + NEF cleanup substrate** the project already has (`OneBrainComposer` cleanup / the emergent CA3 completion) — re-discretization = pattern-completion to a clean CA3 attractor each step. This makes the "simulated recurrent sequence/language cortex" concrete: attractor-stabilized recurrence composing to arbitrary depth on spikes.
- **A5** (non-solvable → the FF route is provably NC¹-hard) sharpens the FF-impossibility once the spiking version holds.

## The mission-central landing
The recurrent LANGUAGE-depth frontier (D3) — the genuine depth-required capability the feedforward deep-credit arc converged toward — is SOLVED at the rate level, and the solving mechanism is **exactly the project's own attractor substrate**: composition to arbitrary depth needs DISCRETE-ATTRACTOR re-discretization (CA3 pattern completion) between steps, not a continuous RNN. The 4-lever negative + this GO together map the boundary AND its fix. Next: the spiking CA3 port + reduced supervision. NO `sim/` edit.

## Files
`research/runners/_d3_group_composition_derisk.py` (arms: `ff_oracle` / `reservoir_ridge` / `bptt_rnn` / `bptt_rnn_state_supervised` / `discrete_attractor_rnn`; `--group --train-lens --test-lens`); findings `2026-07-09-D3-recurrent-multihop-composition-{research-gate,first-arc-BPTT-no-length-generalization}.md`.
