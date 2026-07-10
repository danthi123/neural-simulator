# D3 EVENT — THE CAPSTONE: the deployed brain answers a question about a running discourse from a composed meaning it was **never taught to represent**, on spikes

**Date:** 2026-07-10
**Runner:** `research/runners/_d3_event_selfsup_capstone_derisk.py` (reuse-by-import: `_d3_event_selfsup_derisk` + `build_fswta_score_bridge`/`fswta_drive` + `MultiTurnAgent`; numpy; NO `sim/` edit).
**Verdict:** GO on the 6-seed aggregate — **5/6 seeds GO, seed 100 exactly at the 0.75 gate** (see honest reporting below).

## What this fuses
Three separately-validated rungs, now one deployed system:

| piece | property |
|---|---|
| `_d3_event_selfsup_derisk` | δ learned from an agent-emission cross-entropy ALONE — **no `(agent,patient)` state label anywhere** (adversarially verified by 3 skeptics) |
| `_d3_event_selfsup_spiking_derisk` | that δ executed on a **spiking one-of-K FS-WTA Izhikevich attractor** |
| `_d3_event_qa_wire_derisk` | the deployed QA: resolve the coref-DEEP pronoun via the event register, then query the agent's own KB |

`SelfSupEventRegister` (a drop-in for `D3EventRegister`) plugs into the real `MultiTurnAgent`'s additive `event_register` hook. The agent hears a deep-coref discourse, maintains the running event **on spikes** via a δ **it was never given labels for**, and answers *"what does HE eat?"* by keying its separately-taught eat-KB with the composed referent.

## The label-free naming problem — and its honest solution
The emergent slot is a **permutation** of entity identity, so the register must map slot → referent *name* to answer at all. Fitting that map with true-agent labels would smuggle the supervision straight back in.

**Solution:** INTRODUCE clauses *name the agent in the observable utterance* ("dog chase cat"). So the slot→name read-out is fitted from `(slot-state-after-an-INTRODUCE, the named subject)` pairs only — all observable, **zero hidden labels**. The whole register is therefore label-free: an emergent δ plus a read-out learned from what the brain hears. (Biologically: a downstream region learning to read the slot.)

## Result (6-seed, `--spiking`, 40 random deep discourses per seed; NO `sim/` edit)
| | mean | min | max |
|---|---|---|---|
| **CAPSTONE-QA** (live agent, emergent δ, spiking slot) | **0.908** | 0.750 | 1.000 |
| EMISSION-SEVERED register (δ trained with the agent→emission link cut) | 0.250 | 0.175 | 0.325 |
| FLAT-FACT (unresolved 'he') | 0.271 | — | — |
| RECENCY (last-mentioned entity's eat-fact) | 0.146 | — | — |

**The load-bearing control:** an **emission-severed register collapses to 0.250** through the *identical* deployment — so the deployed answer rides a **learned** δ, not a generic attractor that would sort any scores. Flat-fact and recency both fail.

## Honest reporting
- **5/6 seeds GO; seed 100 lands exactly on the 0.75 gate** (0.750 vs its own severed control at 0.275 — a 0.475 margin, still decisive against every control). Reported, not rounded up.
- **The capstone inherits the emergent δ's error rate (~0.91), not the supervised register's 1.00.** That gap **is the honest price of removing the state label**, and it is the scientifically interesting number: a brain given no labels for who the agent is answers correctly ~9 times in 10 instead of 10 in 10.
- A first pass used **4 hand-picked scenarios** and was replaced: at 0.25 resolution neither the estimate nor the severed control was readable (the severed arm scored 0.5 on two seeds by coincidence). 40 randomly-generated deep discourses per seed fixed both.

## ⇒ the claim
**EMERGENT + SPIKING + DEPLOYED + QA, end-to-end.** The live conversational agent answers a question about a running discourse from a composed *who-did-what-to-whom* meaning whose transition rule it was **never taught** — it fell out of predicting what it heard — and which is maintained by spiking one-of-K attractor dynamics rather than any host arithmetic in the state path.

## Honest scope + next
- Inherits the rate rung's scope: K=6 at the shipped capacity; `M ≥ K` enforced; robust to emission noise and to `p_coref=0.8`.
- The eat-KB query still routes through the composer's `query_patient`; the KB itself is the existing (validated) fact store.
- **Next:** discourse connectives (relate two composed events, not just carry one); the self-supervised δ under multi-turn coherence; scaling K with capacity.

## Files
`research/runners/_d3_event_selfsup_capstone_derisk.py`; rate `2026-07-10-D3-event-selfsupervised-delta-GO.md`; spiking `2026-07-10-D3-event-selfsupervised-delta-ON-SPIKES-GO.md`; QA wire `2026-07-09-D3-event-QA-live-agent-wire-GO.md`; multi-turn `2026-07-10-D3-event-multiturn-coherence-GO.md`.
