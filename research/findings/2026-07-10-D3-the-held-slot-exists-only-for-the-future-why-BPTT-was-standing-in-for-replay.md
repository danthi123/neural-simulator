# D3 — **the held slot exists only for the future**: why backprop-through-time was standing in for replay

**Date:** 2026-07-10
**Runner:** `research/runners/_d3_event_pop_gate_derisk.py` (`truncate=True`; numpy; NO `sim/` edit).
**Verdict:** a clean, decisive NEGATIVE that names the mechanism the next build needs.

## The question

Both gates in the event register are now **structural** (a gated copy between two spiking attractors) rather than
learned recurrences. If the recurrence is structural, credit might no longer need to flow *through time* — and a local,
biologically-plausible rule (Burstprop, `sim/kernels.py: fused_bdsp_update`) is a **one-step** rule. So, before building
a spiking δ:

> Does δ still learn if the gradient is **cut between clauses**?

Single variable: zero `d_c_next` and `d_p_next` each step. Everything else identical.

## Result (6-seed, held-out-deeper)

| | next-emission | a_curr | RETURN | **a_prev (the held slot)** | pop-gate separation |
|---|---|---|---|---|---|
| with BPTT | 0.584 | 0.731 | 0.713 | **0.610** | **+0.751** |
| **no BPTT (truncated)** | **0.609** | **0.767** | **0.175** | **0.195** | **+0.067** |

*(Markov next-emission floor 0.390; intrinsic emission ceiling 0.780.)*

Cutting time-credit makes next-emission and `a_curr` **slightly better**, and **destroys the held slot**.

## Why — and it is exact, not statistical

`a_prev` influences **nothing at the current step.** It enters the transition's input, but its *own* gradient arrives
solely through `d_p_next`. Truncate time and the push gate's only surviving gradient is its own **opening cost** — so it
closes, `a_prev` holds the initial token forever (0.195 ≈ the majority-token floor), and the pop gate has nothing worth
reading (separation +0.751 → +0.067).

**The held slot exists only for the future.** Nothing in the present rewards holding an event. Its credit is
*intrinsically* non-local in time — not as an artifact of the architecture, but as a property of what a held event *is*.

## A confound I walked into, one hour after documenting it

The runner's own automatic verdict line read *"structural gating makes credit local in time — a one-step rule applies,"*
because it gated on **next-emission accuracy**, which went **up**.

But this arc had just established that `P(agent | the current emission) ≈ 0.78` — the emission half-reveals `a_curr`. So
**next-emission accuracy is blind to the collapse of the held slot.** The metric I had warned about in
`2026-07-10-D3-the-emission-carries-the-agent-...md` mis-certified the very next experiment I ran. Recorded, not quietly
fixed: a floor you have measured is only useful if you actually gate on it.

**The δ-replacement gate must therefore be `a_prev` / RETURN, not next-emission.**

## What this means for a local, biological learning rule

A one-step local rule (Burstprop, or the three-term temporal-memory rule) can learn:
- the **current-event map** — confirmed: truncated training reaches next-emission 0.609 and `a_curr` 0.767, *better* than
  BPTT;

and cannot, by construction, learn:
- the **push gate**, because the benefit of writing the held slot is realized only on a later clause.

That is precisely the gap biology fills with a **non-local credit signal**, and this arc has already measured which one
works. An earlier rung established, with its own controls, that **forward prediction does not teach a held prior event
(0.226 ≈ chance) while replay does (0.597)** — reconstructing the just-ended event from the held slot supplies the only
gradient it ever receives.

⇒ **Backprop-through-time was standing in for replay all along.** BPTT delivers, by machinery, the retrodictive credit
that a brain delivers by replaying the episode that just ended (hippocampal sharp-wave ripples). Remove BPTT and the held
slot dies; supply replay and it lives. The two are the same signal in different clothes.

## ⇒ the claim
Structural gating makes the **transition** local in time — a one-step local rule is sufficient for it, and slightly
better than BPTT. It does **not** make the **gate** local: the push gate's entire gradient lives in the future. A spiking
δ built on a one-step local rule must therefore pair it with a **replay/retrodiction signal** to teach the write gate.
This is not a limitation discovered by failure; it is the same conclusion the replay rung reached from the opposite
direction, and the two now agree.

## Next
Build `_d3_delta_spiking_derisk.py` as **two learning problems, not one**:
1. **the transition** — a one-step local rule (Burstprop `fused_bdsp_update`, the emission cross-entropy as the top-down
   factor; or the teacher-free three-term temporal-memory kernel). Gate: next-emission in [0.39, 0.78], target 0.62.
2. **the push gate** — taught by **replay** (retrodict the just-ended event's last observed emission from the held slot),
   the mechanism this arc already validated at 0.597. Gate: `a_prev` and RETURN, **never** next-emission.

Anti-cheats: the Markov floor (0.39); a context lesion; shuffled emissions; an untrained pool; the host δ (0.619) and the
BPTT-taught gate (`a_prev` 0.610, pop-sep +0.751) as references.

## Files
`research/runners/_d3_event_pop_gate_derisk.py` (`truncate`); raw `research/findings/raw/_d3_trunc_seed*.json`.
The emission floor: `2026-07-10-D3-the-emission-carries-the-agent-a-floor-that-reframes-the-metrics.md`.
The replay rung it converges with: `2026-07-10-D3-event-pair-selfsup-NEGATIVE-then-replay-mechanism.md`.
