# D3 EVENT COMPOSITION — the DEPLOYMENT (the anti-RAG payoff): a conversational agent ANSWERS who/what from the running EVENT it maintains, on spikes

**Date:** 2026-07-09
**Runner:** `research/runners/_d3_event_agent_derisk.py` (reuse-by-import: `make_event_task` + `factored_event_rnn` + `build_fswta_score_bridge`/`fswta_drive`; numpy; NO `sim/` edit).
**Verdict:** GO (6-seed: dev 42/43/44 + blind 100/101/102), host AND fully-spiking.

## What this closes
The event-composition rungs proved the discrete-attractor maintains a running FACTORED (agent, patient) MEANING (deep, adversarially verified; weak-supervisable; spiking). This **deploys it as a conversational interface** — the anti-RAG payoff. An `EventDiscourseAgent` HEARS a multi-clause discourse (relational events incl. AGENT-COREF "he …"), maintains the running event ON SPIKES (two FS-WTA slots), and ANSWERS "who is the agent now?" / "who is the patient now?" from the **COMPOSED event** — not a retrieved or last-mentioned fact. This is the concrete demonstration that the conversational loop is no longer *retrieve-a-set → render* (RAG-like); the agent answers about a running MEANING it composes across the discourse.

## The result (6-seed; host AND fully-spiking; NO `sim/` edit)
| who-is-the-agent-now?, deep-coref discourses (held-out-DEEPER), K=6 | host mean | spiking mean |
|---|---|---|
| **EVENT — answered from the running composed event** | **0.972** | **0.972** |
| FLAT-FACT baseline (the last clause's literal subject) | 0.275 | 0.275 |
| RECENCY baseline (the last-mentioned entity) | 0.164 | 0.164 |

(who-patient ≈ 0.99 both.) **GO (all 6 seeds, dev + blind, host == spiking):** the agent answers who-is-the-agent-now from the running EVENT it maintains across the discourse (**0.972**) where the two mechanisms the current conversational stack uses both FAIL: a **FLAT-FACT retriever** (answer the last clause's literal subject — on a coref/promote that is the UNRESOLVED "he"/"it" marker, cannot name an entity) scores **0.275**, and a **RECENCY resolver** (answer the last-mentioned entity) scores **0.164**. ⇒ the agent answers about the **COMPOSED MEANING it tracks**, not a retrieved/last-mentioned fact = the anti-RAG payoff DEPLOYED, and (fully-spiking) on the project's substrate.

## The mechanism (the deployment)
`EventDiscourseAgent.hear(clause)` folds one utterance into the running (agent, patient) via the rate-learned `factored_event_rnn` transition δ + the re-discretization (host argmax, or — `spiking=True` — two FS-WTA Izhikevich slots). `who_agent()`/`who_patient()` read the composed slots. On "the dog chased the cat. he chased the fish. he chased the bird." the agent resolves "who is chasing now?" → **dog** (persisted through the coref run), where flat-fact answers "he" (unresolved) and recency answers "bird" (the last object).

## The anti-cheats
- **EVENT ≫ FLAT-FACT (0.972 vs 0.275):** the flat-fact retriever cannot resolve a coref/promote subject → the agent must COMPOSE the running event.
- **EVENT ≫ RECENCY (0.972 vs 0.164):** the deep-tracked agent ≠ the last-mentioned entity.
- **held-out-DEEPER** discourses (longer than the ≤3 training); **fully-spiking** (two FS-WTA slots) == host; **6-seed** dev+blind.

## Honest scope + next
- The `EventDiscourseAgent` is a standalone running-event register + query interface (like the anaphora ADAPTER before the live `MultiTurnAgent` wire). **Folding the running-event register into the production `MultiTurnAgent` alongside its flat-fact store** — so the deployed agent answers who/what from BOTH the stored facts AND the composed running event (and a query routes to the event when the answer is a composed/coref referent) — is the engineering follow-on (mirrors the anaphora `focus_bias_source` production hook).
- The transition δ is rate-learned (ported to on-spikes re-discretization); on-substrate transition learning + the fully-self-supervised TEM signal are the deeper follow-ons. The fixed FHRR bind stays load-bearing (the running event supplies the composed referent; the composer binds it into a fact).

## Files
`research/runners/_d3_event_agent_derisk.py`; the event arc `2026-07-09-D3-event-composition-running-meaning-GO.md` + `-event-weak-supervision-lookup-op-GO.md` + `-event-composition-spiking-port-GO.md`.
