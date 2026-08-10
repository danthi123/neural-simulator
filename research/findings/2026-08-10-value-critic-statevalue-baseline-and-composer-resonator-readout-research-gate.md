---
type: research-gate
status: active
date: 2026-08-10
mechanism: songbird-statevalue-actor-critic
lane: D-pragmatics / composer
---

# Research gate — two aimed new-mechanism designs from the deep-research batch: (1) the value-critic LEARN-TO-SPEAK fix = a per-context STATE-VALUE baseline (songbird Area-X/VTA actor-critic proper), and (2) the composer offset-invariant RESONATOR readout. Both runner-side, NO sim edit. Design 1 is BUILDING (agent).

Produced by a 2-frontier deep-research workflow that READ the load-bearing NO-GOs line-by-line. Both are DISTINCT IN
KIND from the refuted attempts (the #1 requirement at this mature-project edge), with cited biology grounding.

## Design 1 — "learn to speak from communicative success": a STATE-VALUE baseline (BUILDING now)

<!--derived-->

**The load-bearing BUG (isolated by reading `_pragmatic_success_readback_leg2_v2_derisk.py`):** the critic `V` is a
K-vector of per-UTTERANCE rates (crit[u], ~L302-304); the update `rpe = REWARD_GAIN*(success - V[winner])` (~L464)
is a PER-ACTION advantage. At convergence each `V[u] -> success(intent,u)`, so the advantage COLLAPSES to ~0 for
EVERY utterance — the actor loses its differential teacher and decays to heterogeneity-noise chance (critic-argmax
0.556, actor 0.500 vs chance 0.333; seed-100 critic INVERTED 0.000). This is why BOTH readout-SNR duals (homeostat,
amp-attractor) failed: the learned VALUE is wrong, not un-read-out.

**The fix (distinct in kind):** replace the per-action Q with a per-CONTEXT STATE-VALUE baseline `V(intent)` — one
scalar per intent predicting EXPECTED success over the current policy. The advantage `A = success(chosen) -
V(intent)` is then SIGNED: positive for aligned (above-context-average) utterances -> potentiate; negative for
misaligned -> actively DEPRESS. The signed increment COMPOUNDS over trials (aligned weight climbs, misaligned
falls, soft-bounded), so the final separation is set by TRIAL COUNT, not the tiny single-trial success gap — exactly
why it can succeed where readout amplification (a single-trial operation) could not. Delivered as a discrete
decision-locked DA pulse (sign = A) converting the already-action-localized silent eligibility.

**Grounding (READ, not abstract-skimmed):** Kasdin et al. 2025 *Nature* — Area-X dopamine reflects the CONTRAST
between the current rendition and the RECENT-RENDITION HISTORY (a STATE baseline, explicitly NOT per-action),
"consistent with an actor-critic model". Gadagkar et al. 2016 *Science* 354:1278 (PMID 27940871) — Area-X-projecting
VTA DA encodes a bidirectional, prediction-relative PERFORMANCE error (suppressed worse-than-predicted, activated
better-than-predicted). Chen et al. 2018 — a ventral state-value critic learns the prediction; VTA relays the signed
error to the Area-X actor.

**Build:** fork -> `_pragmatic_readback_leg2_v3_statevalue_derisk.py` (reuse v2 by import; NO sim edit). Two changes:
(1) neural per-context critic `Vctx[intent]` (K neurons, one per intent, reading `cp_firing_states`; a host EMA is a
flagged shortcut); (2) `A = success - rate(Vctx[intent])`. **THE DECISIVE TEST = the CONTINGENCY gate** the prior
gateB attempts failed: a YOKED / shuffled-reward arm (same DA magnitude distribution, DECOUPLED from the action)
MUST NOT converge. Reads: advantage-sign-accuracy (leading indicator) · actor-WTA accuracy · the intent->utter
WEIGHT separation `w_aligned - w_misaligned` (isolates the learning fix from any readout confound) · critic-argmax.
**Honest risk:** if the single-trial success sign is too noisy, `sign(A)` disagrees with alignment often enough that
the compounding averages out — an honest negative that would redirect to a smoother multi-trial success estimate.

## Design 2 — composer offset-invariant RESONATOR readout (for the next build)

<!--derived-->

Tonight established the composer is REPRESENTATION-robust to correlated codes; the "capacity break" is a
readout-RULER offset artifact (Euclidean nearest-proto is offset-sensitive; cosine saturates under collinearity).
The open question is a READOUT: what NEURAL, offset-invariant, collinearity-robust readout recalls the bundle, and
does a REAL bundling-capacity limit exist once the ruler is fixed? **Design:** a RESONATOR / explain-away cleanup
(Frady-Kanerva resonator networks; Plate HRR cleanup) over the generator's per-family readout codebooks
`{readout_m(v)}` (already computed as the running-mean cleanups) — offset-invariant by construction (it factorizes
the bundle by iterated codebook projection + cleanup, not by absolute magnitude). Runner-side (extend
`_teacher_loop_arity_capacity_correlated_derisk.py` with a `pred_resonator` arm). Decisive smoke: does the resonator
recall correctly where Euclidean-nearest-proto craters, and where does a genuine limit (cos-to-true finally
DROPPING) appear? Skeptical: rho=0 must reproduce 1.00; report resonator CONVERGENCE RATE (Kent 2020: resonators
can fall into limit cycles near capacity that masquerade as a "limit"). **Lower mission-priority** than Design 1
(the composer already works for realistic arities; this resolves the academic capacity tail).

NEXT: Design 1 is building (agent) — bank its result (GO only if the yoked control FAILS). Design 2 is the queued
follow-up. NO-EXTERNAL-NEEDED beyond the cited grounding.
