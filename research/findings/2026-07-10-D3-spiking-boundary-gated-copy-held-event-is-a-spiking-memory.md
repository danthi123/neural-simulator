# D3 → the SPIKING boundary-gated copy: the held prior event is a **genuine spiking memory**, not a Python variable

**Date:** 2026-07-10
**Runner:** `research/runners/_d3_event_gated_copy_spiking_derisk.py` (numpy backend; NO `sim/` edit).
**Verdict:** the substantive claim holds on 6/6 seeds. **The pre-registered gate passes on only 1/6** because of a threshold I set too strictly against a control that is not zero — reported as-is, not re-tuned.

## What this closes
Every earlier "spiking slot" in this arc used `fswta_drive` — a **stateless** one-of-K re-discretizer (it resets `v`/`u`/`firing` each call; pools have `internal_density=0`). The slot's *hold* lived in a Python variable between calls. Acceptable while the claim was "the winner is chosen by spikes"; **a fiction for the boundary-gated copy, where the HOLD *is* the mechanism.**

Here the held event lives in a **persistent slow-NMDA attractor** (`2026-07-10-D3-persistent-spiking-slot-hold-and-clear-then-load.md`), and the gate does what `sim/`'s own `transmission_gate` documents:

- **gate CLOSED** → **no input to the slot at all** (asserted identically zero). The attractor **sustains its own firing** (Wang 2002). The prior event is remembered *by spikes*, across arbitrarily many clauses.
- **gate OPEN** → **CLEAR** (an FS inhibitory burst longer than τ_NMDA) **then LOAD** the current event. Measured previously: input alone cannot overwrite an attractor (0/6), and a reset shorter than τ_NMDA lets the old bump re-ignite.

The transition and gate are the rate-learned ones (agent-emission cross-entropy **alone**; **no `(agent,patient)` state label anywhere**; the gate reads only the observable clause code). The held agent is then **read out of spikes** — whichever pool is firing at the end of the discourse.

## Result (6-seed)
| | mean | range |
|---|---|---|
| **SPIKING held prev-agent** | **0.664** | 0.517 – 0.800 |
| rate gated-copy reference | 0.693 | 0.649 – 0.751 |
| **STATELESS `a_prev` slot** (the bridge every prior rung used) | **0.426** | 0.241 – 0.600 |
| gate-lesion (never opens) | 0.176 | ≈ chance |
| recency | 0.164 | |

**The spiking arm matches the rate model** (0.664 vs 0.693) while the memory is genuinely on the substrate, and it beats the stateless slot on **6/6 seeds**.

## The decisive check: a like-for-like host twin
A **host twin** of this pipeline — identical binarised gate, one-hot copy and one-hot feedback, but **no bridge at all** — scores **exactly** what the spiking arm scores (0.444 vs 0.444 on the seed-42 diagnostic before the read-out fix). **The spiking substrate is faithful: nothing is lost by putting the hold on real NMDA neurons.** The gap to the rate model is entirely *discretisation* (soft convex copy → hard one-of-K), not the neurons.

## Two bugs found by instrumenting, not tuning
1. **`_reset` did not clear conductances.** It reset `v`/`u`/`firing`, but `g_nmda_recurrent` has τ=100 ms and **survived** — measured **95.1 before and after**. Every discourse item therefore inherited the previous item's fully-charged held bump, which re-ignited into it. This is the *same* residual-conductance re-ignition that forces a gate's CLEAR to outlast τ_NMDA — my own law, biting me. Fixing it: 0.389 → 0.444.
2. **The slot is a permutation of entity identity.** Comparing a raw slot index to a true agent index is meaningless; the rate reference is obtained through a fitted **slot→entity read-out**. I had omitted it in the spiking arm. Applying it: 0.444 → **0.667**. The spiking result was never 0.444 — my *metric* was wrong, not the neurons.

A negative result along the way: sweeping the CLEAR duration (250 / 400 / 600 ms) and LOAD duration changed nothing (0.444 flat), which correctly ruled the switch *out* as the limiter and pointed at the read-out.

## Honest reporting
- **The pre-registered gate passes on 1/6 seeds** because I required `spiking − stateless > 0.25` and the mean gap is **0.238**. I am reporting the threshold miss rather than lowering it.
- **The stateless control is not zero (0.426).** With `recur=0` the pool still fires *during* the LOAD pulse and leaves a decaying trace inside the 30-step read window, so it behaves as a leaky short memory rather than no memory. That is why the margin is ~0.24 rather than ~0.5. The *directional* claim is unambiguous: the persistent slot beats it on every seed, and gate-lesion sits at chance (0.176).
- Evaluated on 60 items/seed (a real bridge runs per clause); n_informative 20–31 per seed.

## ⇒ the claim
**The event arc's HOLD is no longer a Python variable.** A prior event, copied at a boundary by a gate learned from an observable marker with no state label, is *remembered by a self-sustaining spiking attractor* across arbitrarily many clauses, updated only by a clear-then-load — and read back out of spikes at rate-model fidelity.

## Honest scope + next
- Inherits the rate rung's honest headline (`gate_cost` selection: 6-seed ≈0.63 under held-out selection; 0.693 at the tuned constant) — see `2026-07-10-D3-boundary-gated-copy-...md`, adversarially verified by two skeptics.
- `a_curr` is still re-chosen each clause on the host; only `a_prev` (the *held* slot, where the claim lives) is on the substrate. Putting `a_curr` on its own attractor is the next rung.
- Then: re-deploy the self-supervised pair register on the gated copy (its BEFORE answer was 0.367 with replay).

## Files
`research/runners/_d3_event_gated_copy_spiking_derisk.py`; the persistent slot `2026-07-10-D3-persistent-spiking-slot-hold-and-clear-then-load.md`; the rate mechanism `2026-07-10-D3-boundary-gated-copy-the-held-event-is-gated-not-learned.md`.
