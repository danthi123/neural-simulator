# D3 CAPSTONE — the deployed brain answers *"who was doing it before?"* from a prior event **remembered by spikes**, with no state label anywhere

**Date:** 2026-07-10
**Runner:** `research/runners/_d3_event_gatedcopy_spiking_agent_derisk.py` (numpy; NO `sim/` edit).
**Verdict:** 4/6 seeds GO. The deployed answer is **0.708** — matching the rate deployment (0.711) while the memory is genuinely on the substrate.

## What this fuses
| property | how |
|---|---|
| **EMERGENT** | the transition δ is learned from an agent-emission cross-entropy **alone**; the gate reads only the **observable** clause code; the single slot→name read-out is fitted from clauses whose subject is **spoken**. **No `(agent, patient)` state label anywhere.** |
| **SPIKING HOLD** | the held prior event lives in a **persistent slow-NMDA attractor**. Gate CLOSED ⇒ **zero input** (asserted) and the attractor sustains its own firing across arbitrarily many clauses. Gate OPEN ⇒ **CLEAR** (an inhibitory reset longer than τ_NMDA) **then LOAD** — exactly `sim/`'s `transmission_gate` semantics. |
| **DEPLOYED** | a drop-in `event_register` on the real `MultiTurnAgent`, answering `who_agent_before()`. |

The held agent is **read out of spikes** (whichever pool is firing), then named by the label-free read-out.

## Result (6-seed)
| | mean | range |
|---|---|---|
| **BEFORE — prior event held by SPIKES** | **0.708** | 0.350 – 0.900 |
| rate gated-copy deployment (reference) | 0.711 | — |
| **STATELESS held slot** (the re-discretizer every earlier rung used) | **0.492** | beaten on 6/6 |
| gate-lesion (gate never opens) | 0.167 | ≈ chance |
| SINGLE-EVENT register | 0.000 | structurally cannot answer |
| recency | 0.167 | |
| naive "answer the current agent" | 0.067 | |
| NOW (current event) | 0.775 | |
| *replay deployment* | *0.367* | |
| *fully-labelled register* | *0.928* | |

**Putting the hold on real neurons costs nothing** (0.708 vs 0.711) — consistent with the host-twin result that the spiking substrate is faithful.

## A contamination I caught in my own control
The first run reported **BEFORE 0.833 and stateless 0.833 — identical.** A stateless slot cannot hold anything, so that was impossible. Reading my own code: `who_agent_prev()` fell back to `self.prev_winner` when the spiking read was silent, and `prev_winner` is a **Python variable** that retained the last value. The stateless arm was being silently rescued by **host memory** — precisely the fiction this rung exists to remove.

Fixed: `a_prev` now comes **only** from spikes; a silent slot means *nothing is held* (no host fallback, in the read *or* in the transition feedback). The stateless control immediately collapsed to 0.333, as it must. **A control that cannot fail is not a control.**

## Honest reporting
- **4/6 seeds GO.** Seed 43 misses the stateless margin (gap 0.10 < my pre-registered 0.15). **CORRECTION (same day, by measurement):** seed 102 (0.350) is NOT a gate-learnability failure as I first wrote — its gate separation is **+0.890** and its rate held-slot decode **0.649**. Nor is it a naming-collision failure: a bijective label-free read-out changes no deployed answer. The residual is **slot-tracking under the deployed discourse distribution**.
- **The stateless control is not zero (0.492).** With `recur=0` the pool still fires during the LOAD pulse and leaves a decaying trace inside the read window, so it behaves as a leaky short memory rather than no memory. The directional claim is unambiguous (beaten on 6/6; gate-lesion at chance) but the margin is ~0.22, not ~0.5.
- 20 informative discourses/seed (a real bridge runs per clause).
- Only the **held** slot is on the substrate; `a_curr` is still re-chosen on the host each clause. That is the next rung.
- The upstream held-slot decode headline remains the adversarially-corrected one (≈0.63 under held-out `gate_cost` selection; comparable to replay's 0.597, not far past it).

## ⇒ the arc, in one line
A brain that was **never told who the agent is** learns a discourse transition from prediction alone; a **boundary opens a gate** and the just-ended event transfers into a **self-sustaining spiking attractor**; that attractor **remembers it with no input at all** across arbitrarily many clauses; and when asked *"who was doing it before?"*, the deployed agent **reads the answer out of spikes** — at **0.708**, against 0.367 for replay and 0.928 for a fully-labelled register.

## Files
`research/runners/_d3_event_gatedcopy_spiking_agent_derisk.py`; the mechanism `2026-07-10-D3-boundary-gated-copy-the-held-event-is-gated-not-learned.md`; the persistent slot `2026-07-10-D3-persistent-spiking-slot-hold-and-clear-then-load.md`; the rate deployment `2026-07-10-D3-gatedcopy-deployed-price-of-emergence-halved.md`.
