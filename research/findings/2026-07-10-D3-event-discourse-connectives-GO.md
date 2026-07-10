# D3 EVENT → DISCOURSE CONNECTIVES (6-seed GO): a connective marks an event boundary, so the brain holds a PAIR of composed events and can relate them

**Date:** 2026-07-10
**Runner:** `research/runners/_d3_event_connective_derisk.py` (numpy; NO `sim/` edit).
**Verdict:** GO (6-seed dev 42/43/44 + blind 100/101/102).

## What this closes
Every event rung so far held **one** running event and **overwrote** it. So the register structurally could not answer *"who was doing it **before**?"* — it had no prior event to relate the current one to. That is the research gate's RANK-3 "connectives" residual.

**The mechanism (Zacks & Radvansky event segmentation):** a connective ("then", "but") marks an **EVENT BOUNDARY**. Instead of overwriting, the current event model is **SHIFTED** into a previous slot, and a new event opens:

```
state = ( a_curr, p_curr | a_prev, p_prev )        each a K-way discrete-attractor slot

non-boundary clause  -> update the CURRENT event in place   (INTRODUCE / AGENT-COREF / PROMOTE)
boundary clause      -> (a_prev, p_prev) <- (a_curr, p_curr), then the new clause opens the new current event
```

The prior event must then be **held across however many non-boundary clauses follow** — so the shift is *not* a delayed copy of the last clause; it must survive an arbitrary run. That is precisely what a single-event register cannot do.

## Result (6-seed, held-out-DEEPER lengths 7/8/9 vs train 3/4/5; NO `sim/` edit)
| | mean | range |
|---|---|---|
| **EVENT-PAIR: previous-event agent** | **0.881** | 0.839 – 0.928 |
| **EVENT-PAIR: same-agent RELATION across the pair** | **0.929** | 0.907 – 0.960 |
| EVENT-PAIR: current-event agent | 0.920 | 0.895 – 0.949 |
| **SINGLE-EVENT control: previous agent** | **0.467** | 0.396 – 0.493 |
| SINGLE-EVENT control: *current* agent | 0.990 | 0.985 – 0.995 |
| recurrence-lesion (previous agent) | 0.276 | 0.258 – 0.289 |
| RECENCY (previous agent) | 0.167 | 0.154 – 0.176 |

Chance is 0.167.

## The load-bearing contrast
The **single-event control is not a weaker model — it is a structurally incapable one.** It tracks the *current* event essentially perfectly (**0.990**, better than the pair model) while failing on the *previous* event (**0.467**). It is asked the same question through the same prev-agent head; it simply has no prior-event slot to carry the answer in. The event-pair recovers the prior agent at **0.881** and reads the same-agent relation across the pair at **0.929**.

## Anti-cheats (all pass)
- **(a)** PAIR prev-agent (0.881) ≫ SINGLE-EVENT prev-agent (0.467) — the load-bearing structural contrast.
- **(b)** ≫ RECENCY (0.167 = chance): the prior agent is not the last-mentioned entity.
- **(c)** recurrence-lesion collapses to 0.276 — the prior event is *held* by the recurrent state, not recomputed from the current clause.
- **(d)** the same-agent RELATION (0.929) is read across the pair, not from either slot alone.
- **(e)** held-out-DEEPER lengths (7/8/9 vs trained 3/4/5).

## Honest reporting
- **The single-event control sits at 0.467, not chance.** It is not guessing blindly: it learns the useful prior that when *no boundary has occurred yet*, the previous slot is still the initial (identity) state. That is a real, partially-correct strategy — and it is exactly why the gap (0.881 vs 0.467) rather than "0.881 vs chance" is the honest measure of the capability.
- **The pair model's current-event accuracy (0.920) is *below* the single-event model's (0.990).** Holding two events costs capacity: four K-way heads share one hidden layer. This is a real trade-off, reported, not hidden.

## ⇒ the claim
A connective-marked event boundary shifts the running event into a previous slot, so the discrete-attractor holds a **pair** of composed events and can **relate** them. **The brain relates two composed meanings, not just carries one.**

## Honest scope + next
- Per-step supervised (like the RANK-1 rung) — the self-supervised δ for the *pair* (no state label) is the natural follow-on, as is the spiking port (four FS-WTA slots) and the deployed register.
- Two events (depth-2). A deeper event stack, and true Contrast/Cause semantics (not just Sequence + a same-agent relation), remain open.

## Files
`research/runners/_d3_event_connective_derisk.py`; the capstone `2026-07-10-D3-event-CAPSTONE-emergent-spiking-deployed-QA.md`; multi-turn `2026-07-10-D3-event-multiturn-coherence-GO.md`.
