# D3 — **the emission carries the agent**: a floor that reframes which of this arc's metrics were honest

**Date:** 2026-07-10
**Measurement only** (numpy; NO `sim/` edit). Raw: `research/findings/raw/_d3_popgate_rand_seed*.json` harness + the
scratch probes recorded below.

## What was about to be built, and why it would have been vacuous

The last host computation in the event register is the transition `δ`: the learned map
`(a_curr, a_prev, patient, clause-code) → next agent`. Both gates are now structural and both memories are spiking, so δ
is the one genuinely *learned* piece, and the natural next step is to put it on the substrate using this repo's committed
spiking learning rules (`fused_htm_permanence_update`, the Bouhadjar–Diesmann three-term temporal-memory rule; or
`fused_bdsp_update`, Burstprop).

The obvious probe: train the temporal memory teacher-free on the emission sequence and ask whether **the latent agent
falls out as its context-specific cells**, measured by decoding the agent from the winner cells.

**Measured first, and it kills that probe:**

| | seed 42 | seed 43 |
|---|---|---|
| **P(agent \| the CURRENT emission alone)** | **0.794** | **0.764** |
| P(top emission \| agent) — the intrinsic emission noise | 0.776 | 0.749 |
| majority-agent floor | 0.174 | 0.171 |
| first-order Markov next-emission | 0.390 | 0.369 |

The agent is **~0.78 recoverable from the current emission alone** — *higher* than the self-supervised transition
network's own `a_curr` accuracy (0.68–0.73). Since a temporal memory's cells are driven by emissions, "the agent is
decodable from the cells" would have measured the emission, not the mechanism. A vacuous claim, caught by a
thirty-second measurement instead of by a reviewer.

## The consequence that matters: which metrics in this arc were load-bearing

`a_curr` **can be half-read off the current utterance.** `a_prev` **cannot be read off anything currently observable** —
nothing in the present clause mentions the protagonist of the event that ended.

That is exactly why the honest tests of this whole arc were the ones about the **held** slot:

- **resumption** ("who is doing it now?" right after a discourse pop) — the answer is an agent no present utterance names;
- **BEFORE**, restricted to discourses ≥2 clauses past the push;
- the **stateless** control, which collapses precisely because a slot that cannot hold has nothing to offer that the
  current clause does not already contain.

Conversely it explains a result that had looked merely lucky: a *rate-matched random* pop gate scores **below** never
popping (0.286 vs 0.333) and collapses the held slot (0.532 → 0.441). Opening the read gate at the wrong time replaces a
present agent — which the emission would have half-revealed anyway — with a **stale** one, destroying the only
information the register holds that the utterance does not.

## The correct bracket for a spiking δ

A spiking, locally-learned replacement for δ must be gated on **next-emission prediction** — the quantity that genuinely
requires the agent state, through coreference, promotion and return — not on agent decodability:

| first-order Markov floor | **0.390** |
|---|---|
| **host δ (tanh + softmax, BPTT through the gates)** | **0.619** (0.598 / 0.635 / 0.623 on seeds 43 / 42 / 44) |
| intrinsic ceiling (emission noise) | **0.780** |

The host δ's next-emission accuracy is close to **uniform across operations** (INTRO 0.60–0.64, COREF 0.59–0.63,
PROMOTE 0.60–0.64, BOUND 0.62–0.65, RETURN 0.59–0.64) once the pop gate is installed — consistent with the pop gate
having repaired the one operation that was failing.

## ⇒ what this changes
1. **The δ-replacement gate is next-emission accuracy in [0.39, 0.78], targeting 0.62.** Any claim about "the agent
   emerging in the cells" must beat the emission-alone floor of 0.78, which is very likely unreachable and is the wrong
   question.
2. **`a_prev` is the substrate's real memory claim.** Every result in this arc that rides on the held slot is measuring
   something the present utterance cannot supply. Every result that rides on `a_curr` is partially measuring the
   utterance.
3. Reported rather than tuned away: this makes several `a_curr` numbers in earlier rungs *less* impressive than they look,
   and the held-slot numbers *more* so.

## Next
Build the spiking δ against the bracket above: `(a_curr assembly, a_prev assembly, patient, clause-code) → next agent`,
learned by a local rule on the substrate (the committed three-term temporal-memory kernel, teacher-free; or Burstprop
with the emission cross-entropy as the top-down factor). Anti-cheats: the Markov floor (0.39), a context-lesion, shuffled
emissions, an untrained pool, and the host δ (0.62) as reference — with next-emission accuracy, not agent decodability,
as the gate.

## Files
The registers this measures: `_d3_event_pop_gate_derisk.py`, `_d3_event_pair_spiking_derisk.py`.
The committed rules a replacement would use: `sim/kernels.py` `fused_htm_permanence_update`, `fused_bdsp_update`.
