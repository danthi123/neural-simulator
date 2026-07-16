# Off-diagonal recurrent-credit ARC — synthesis: the missing mechanism is FOUND + rate-validated (MDGL off-diagonal); the spiking realization is fragile on the abstract toy (rate→spike degradation, toy exhausted); the honest strategic read + the next substrate

**Date:** 2026-07-15 · **Status:** MILESTONE synthesis (consolidation, not closure — a negative/partial launches the next substrate). Answers the owner's directive "we've got to be missing something if biology can do it and our code can't."

## What the owner asked + what this arc found

**Owner directive (this session):** "put our full focus on the learned graded cortex — that's our main blocker … rely on our academic resources to properly implement what's needed. We've got to just be missing something if real biology can manage it while our code can't."

**What we found (the missing thing, named + located):** feedforward spiking deep credit is ALREADY SOLVED on our substrate (e-prop forward-eligibility × membrane surrogate × fixed-random DFA + population coding → on the production Izhikevich bridge to the LIF ceiling, K=1 0.47 → K=8 0.877, 6-seed GO). The genuine open piece is **RECURRENT OFF-DIAGONAL cross-neuron temporal credit** — e-prop's diagonal RTRL zeroes ∂hₖ/∂hⱼ, so on a delayed-cue task the diagonal sits at chance (0.38) while BPTT solves it (0.99–1.0). That off-diagonal term is the one the WHOLE transport-free family (FA / burstprop / microcircuit / graded / DECOLLE / node-perturbation) omits. The biological realization = **MDGL** (Liu et al. PNAS 2021) — cell-type-specific one-hop neuromodulation: each neuron emits a neuropeptide-like signal to its direct synaptic partners, so a synapse's update sees its postsynaptic partners' loss-contribution. Single-phase, transport-free, reuses `sim/neuromodulators.py`.

## The evidence ladder (this session, all anti-cheated, NO `sim/` edit)

| rung | testbed | result |
|---|---|---|
| **Stage 0 — calibration gate** (Merin arXiv:2603.28750 "immediate derivatives suffice" — is the diagonal just mis-calibrated?) | recurrent e-prop + Adam + swept trace-decay, delayed-XOR | **REFUTED** — no LR/decay makes the diagonal solve delayed-XOR; the off-diagonal is a real missing term, not a calibration artifact |
| **MDGL on rate** | trainable leaky-tanh RNN, delayed-XOR/DMS | **VALIDATED, clean directional** — closes **+48–64%** of the diagonal-vs-BPTT gap; sign-flip HURTS, zero-Γ collapses to diagonal, permuted→chance. The science. |
| **MDGL spike port (single-neuron LIF)** | surrogate-gradient LIF, delayed-XOR | **FRAGILE** — BPTT 0.99 (valid ceiling), diagonal 0.38 (boundary reproduces), but MDGL vs sign-flip is gain-noisy: clean only at gain 0.4 (0.43 vs 0.27), below e-prop or flip-wins at 0.2/0.7/1.0 |
| **MDGL spike port (population-coded, K neurons/unit)** | pooled-rate LIF | **NEGATIVE — magnitude confound** — MDGL rises to 0.70 but the sign-FLIPPED Γ wins at 3/4 gains → the boost is capacity, not directional credit (degenerate near-critical regime; the sign-flip anti-cheat caught it) |

⇒ the off-diagonal mechanism is **rate-validated + clean**, but **does not port cleanly to spikes on the abstract XOR toy** — the rate→spike degradation (the project's known recurring theme; the known surpass is genuine population coding, which the toy can only fake, degrading its own dynamics). The abstract toy is **exhausted** (broken pop-coded BPTT + near-critical dynamics + magnitude-confounded Γ = the systematic-debugging "3 issues → question the architecture / testbed" signal).

## The strategic read (our own record makes this decisive — a-1)

Two prior findings + the ceiling run fix the priority, and they must be weighed HONESTLY (not skimmed past):
- `2026-07-14-eprop-recurrent-synthesis-CONTROLS-REFUTED`: diagonal e-prop's reservoir-LM "beats bigram at deep context" was a **credit-direction-INDEPENDENT memory-timescale artifact** (sign-flip==plastic, true-gradient hurts, loses to a trigram). ⇒ a FIXED reservoir + a diagonal rule does NOT learn real long-range structure — so the off-diagonal genuinely IS the *recurrent-learning enabler* (the thing that would make a recurrent cortex learn real structure).
- `2026-07-15` deep-credit plan + the CEILING run (memory `feedback_run_ceiling_early`): the off-diagonal is **explicitly 3–4 orders below language scale and OFF the open-generation critical path** — fluency is a **DATA/SCALE wall** (all model classes, incl. a full transformer, lose to a tuned n-gram at ~5M tokens / V=300). The off-diagonal is the emergence-ENABLER (necessary-not-sufficient), not the fluency bottleneck.

**⇒ Honest synthesis:** we FOUND what we were missing (off-diagonal cross-neuron credit) and VALIDATED it at rate. It is the emergence-engine enabler — but it is (a) fragile to realize on spikes on an abstract toy, and (b) off the fluency critical path (scale is the binding constraint for conversation). Sinking a large on-bridge `sim/`-adjacent build into an off-critical-path mechanism that is fragile-on-toy is the wrong ROI **without first testing it on the one substrate where population coding is GENUINE** (a `BrainRegion` of many real Izhikevich neurons per logical unit — the real form of the surpass the toy could only fake).

## NEXT (the decisive, cheapest-first substrate test — NOT more toy-engineering)

The single test that resolves whether the off-diagonal is worth the full on-bridge MDGL build: **does the diagonal boundary even reproduce on the REAL on-bridge recurrent substrate?** On the toy the diagonal fails because a single binary-spike neuron's eligibility is sparse; on the real substrate the natural population coding (many neurons + independent OU noise per unit) may already recover the graded eligibility the toy lacked. So the cheapest-first on-bridge rung is:
1. Build a **recurrent on-bridge Izhikevich net** on a delayed-cue task (reuse `_onbridge_eprop_port_derisk` machinery, add a recurrent slice), diagonal e-prop only.
2. If diagonal e-prop + real population coding already solves it on-bridge → the off-diagonal frontier **dissolves on the real substrate** (the toy's failure was a single-neuron-sparsity artifact); STOP, no MDGL build needed.
3. If the diagonal boundary reproduces on-bridge → add MDGL's Γ via the per-synapse-DA path (6 per-cell-type modulators, tag = presynaptic cell type, × the on-bridge eligibility) — the full realization is now warranted.

This is the boundary-surpassing workflow's cheapest-first single-variable rung on the RIGHT substrate, and it is directive-aligned (the learned recurrent cortex on our one spiking brain). NO `sim/` edit for the rung (reuse-by-import + the existing per-synapse-DA path).
