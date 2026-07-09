# D3 residual SURPASSED (RANK 1, 6-seed GO) — the transition δ is GENUINELY LEARNED from K-way end-state-only supervision + a short-length curriculum; no per-step intermediate teaching, and it length-generalizes perfectly

**Date:** 2026-07-09
**Runner:** `research/runners/_d3_weak_supervision_derisk.py` (reuse-by-import of the group task; numpy; NO `sim/` edit).
**Verdict:** GO (S3, 6-seed 42/43/44/100/101/102) — the residual credit-assignment wall of the whole D3 arc is surpassed.

## The wall this closes
The D3 arc realized the discrete-attractor recurrent composition on spikes (transition LIF + re-discretization FS-WTA, one loop, length-generalizing), but the transition δ was trained on teacher-forced PER-STEP (state,input)→next-state triples (every intermediate state handed to the learner). Rung 3 showed END-LABEL-ONLY supervision does NOT length-generalize (deeper ≈ chance). Learning δ from weaker-than-per-step supervision was the residual.

## The reframe (research-gated, controller-verified) + the mechanism
Rung 3 supervised the **2-way property** at the endpoint = **1 bit**; the running K-way state needs **≈log₂K bits** (S3: 2.58). One endpoint bit cannot pin the K-way state → BPTT through the scrambling latent walk has no learnable target. But δ is **length-INDEPENDENT** (learned from short sequences; depth-gen is the drift-free rollout). So the fix: supply ≈log₂K bits at SHORT lengths WITHOUT per-step states — **supervise the K-way FINAL STATE** on short sequences.

The correct realization (the straight-through BPTT taught garbage early — the final-step target is conditioned on the model's OWN wrong intermediate prediction): a **Dyna-style DETACHED-rollout CURRICULUM.** Each sequence is rolled forward autoregressively with argmax (the intermediate states are the model's OWN predictions, DETACHED — never targets); ONE supervised gradient step on the FINAL step only: `CE(f(emb[roll_state_{L-1}], x_L), true_final_state_L)`. The **curriculum** (train length-1 → include length-2 → length-3) makes the rolled prev-state correct depth-by-depth: length-1 teaches δ(ident,·)=g₁ (a rollout-free clean target); once learned, the model rolls step-1 correctly → length-2's end-state cleanly teaches δ(s₁,·) over all s₁; etc. **No intermediate state is ever a target — end-states only** (the observable outcome of a k-step behavior, the research gate's Kandel place-cell anchor).

## The result (S3, 6-seed; NO `sim/` edit)
| arm (DEEPER = held-out lengths 6/7/8, chance 0.5) | mean | per-seed |
|---|---|---|
| **STATE endpoint (RANK 1, log₂K bits)** | **1.000** | 1.0 / 0.999 / 1.0 / 1.0 / 1.0 / 0.999 (state-track ~1.0) |
| PROPERTY endpoint (= rung 3, 1 bit) | 0.610 | 0.499 / 0.510 / 0.553 / 0.721 / 0.786 / 0.593 |
| SHUFFLE (memorization-floor) | 0.584 | 0.508 / 0.500 / 0.550 / 0.690 / 0.709 / 0.549 |

**GO all 6 seeds:** STATE 1.000 ≫ PROPERTY 0.610 (Δ 0.39) ≫ SHUFFLE 0.584 (Δ 0.42).

**Train-lens≤2 ablation (train lengths 1,2 ONLY → test 6,7,8), 3 seeds: STATE deeper = 1.000** — δ learned from the two SHORTEST lengths alone generalizes perfectly to ~3–4× deeper. This is the decisive length-independence proof: the transition is learned from short sequences and the drift-free re-discretized rollout carries it to arbitrary depth (not depth-memorization).

## What the anti-cheats establish
- **STATE ≫ PROPERTY** (1.000 vs 0.610): the reframe is confirmed — the K-way endpoint (log₂K bits) is FULLY learnable where the 1-bit property endpoint (= rung 3) is at best partially learnable. (Honest: on 2 blind seeds the random 2-coloring is ~0.72–0.79 learnable by an endpoint shortcut, but STATE is a perfect 1.0 every seed.)
- **STATE ≫ SHUFFLE** (1.000 vs 0.584): it learns the REAL transition, not a memorized endpoint mapping (shuffling the endpoint labels collapses it).
- **train-lens≤2 → 1.000**: length-independence (learned from lengths 1,2, generalizes to 6,7,8).
- permuted-ORDER (order-changes 0.43): the task is genuinely order-dependent (non-abelian).

## ⇒ the D3 arc is now closed on the LEARNING side too
The transition δ — the DFA table the discrete-attractor rolls out — is GENUINELY LEARNED from **end-state-only** supervision (a weak, observable signal), NOT per-step taught. Combined with: the mechanism (discrete-attractor = CA3), the rate GOs (S3 + theorem-backed A5), and the full spiking realization (transition LIF + re-discretization FS-WTA in one loop), the discrete-attractor recurrent multi-hop composition is now **found, learned-from-weak-supervision, and realized on spikes.**

## Honest scope + next
- Validated on S3 (K=6). **A5-scale the weak-supervision learning** (does the curriculum learn the 60-way non-solvable DFA from end-states? — the coverage lever applies) is the immediate next test.
- RANK 3 (the more emergent self-supervised state-tied-OBSERVATION version — HAE/TEM, δ from next-observation prediction, homomorphism by construction) is the deeper follow-on; RANK 2 (the biological dopamine-TD + reverse-replay realization on `bridge.py`) is a documented-boundary on the pure task (temporal not structural credit) → build after RANK 3 with an observable state.
- The end-state-only + curriculum is still a HOST training loop; the on-spikes / on-bridge realization of the curriculum learning (via the project's SWR-replay + eligibility machinery) is the spiking port.
- Apply to real LANGUAGE sequences (incremental composition of a running linguistic state) — the mission payoff.

## Files
`research/runners/_d3_weak_supervision_derisk.py`; research gate `2026-07-09-D3-sparse-supervision-research-gate.md`; the D3 arc `2026-07-09-D3-*.md`.
