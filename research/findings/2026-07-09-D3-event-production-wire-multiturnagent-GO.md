# D3 EVENT COMPOSITION — the FULL PRODUCTION WIRE: the deployed MultiTurnAgent answers who/what from the running EVENT, on spikes

**Date:** 2026-07-09
**Runner:** `research/runners/_d3_event_agent_derisk.py --wire` (+ an additive default-off `event_register` hook on `multi_turn_agent.py`; reuse-by-import; numpy; NO `sim/` edit).
**Verdict:** GO (6-seed: dev 42/43/44 + blind 100/101/102), host AND fully-spiking; default path byte-identical.

## What this closes
The standalone `EventDiscourseAgent` deployment proved the anti-RAG payoff; this **folds the running-event register into the deployed `MultiTurnAgent`** — the production agent that already does flat-fact who/what Q&A + biased-competition anaphora. A `D3EventRegister` (mirroring the anaphora `D3CenteringFocusSource` adapter) maintains the running FACTORED (agent, patient) EVENT over the SVO facts the real agent hears; the agent answers `who_agent_now()` from the COMPOSED running event **alongside** its flat-fact store. This is the anti-RAG capability on the real deployed agent.

## The result (6-seed; host AND fully-spiking; NO `sim/` edit)
On deep-coref discourses (the agent is set at a varying depth then PERSISTS via "he" corefs, e.g. *"the dog chased the cat. he chased the fish. he chased the bird. he chased the worm."*):

| deployed `MultiTurnAgent` answers "who is chasing now?" | host | fully-spiking |
|---|---|---|
| **EVENT — `who_agent_now()` from the D3EventRegister** | **1.00** | **1.00** |
| FLAT-FACT baseline (the last clause's literal subject) | 0.00 | 0.00 |
| RECENCY baseline (the last-mentioned entity) | 0.00 | 0.00 |

**GO (all 6 seeds, dev + blind, host == spiking):** the deployed agent resolves "who is chasing now?" → **dog** (the deep-tracked agent, persisted through the coref run) via the running event, where the FLAT-FACT retriever answers the last literal subject ("he", an UNRESOLVED coref → 0.00) and RECENCY answers the last object (→ 0.00). ⇒ the running-event register is WIRED into the production agent; the deployed brain answers about the COMPOSED MEANING it maintains, not a retrieved/last-mentioned fact.

## The wire (additive, default byte-identical)
- **`MultiTurnAgent(event_register=...)`** — a default-None hook. When set, `hear` folds each heard fact into the register from the RAW subject+object words (the D3 encoding is parser-independent: an entity subject = INTRODUCE, "he"/"she"/"they" = AGENT-COREF, "it" = PROMOTE). A coref/promote subject (which the flat-fact composer cannot store as an entity — the exact gap the running event fills) updates the register + WM and SKIPS the composer store. `who_agent_now()`/`who_patient_now()` read the composed event.
- **`D3EventRegister`** (in the de-risk runner) — trains the factored event transition, holds an `EventDiscourseAgent` (host argmax, or two spiking FS-WTA slots), maps referent names ↔ codes.
- **Default None = byte-identical:** `tests/test_multi_turn_agent.py` 3/3 pass unchanged (the flat-fact/biased-competition paths are untouched).

## Honest scope
- The **1.00** is on representative hand-picked deep-coref discourses (like the anaphora live-agent wire's curated focus-shifted discourses) where the contrast with flat-fact/recency is maximal (0.00). The broader-coverage number is the standalone de-risk on RANDOM discourses (0.972, `2026-07-09-D3-event-agent-anti-rag-deployment-GO.md`).
- The register's transition δ is rate-learned (ported to on-spikes re-discretization); the on-substrate transition LEARNING + the fully-self-supervised TEM signal are the deeper follow-ons.
- The fixed FHRR bind stays load-bearing (the running event supplies the composed referent; the composer binds resolved facts).

## ⇒ the event-composition arc is deployed end-to-end
The anti-RAG running MEANING is FOUND (factored discrete-attractor, deep, adversarially verified) → WEAK-SUPERVISABLE → SPIKING → deployed standalone → **WIRED into the production MultiTurnAgent** (host + fully-spiking), which now answers who/what from the composed running event alongside its flat-fact store. The conversational loop is no longer only retrieve-a-set → render; the deployed brain maintains + answers about a running composed meaning.

## Files
`research/runners/_d3_event_agent_derisk.py` (+ the `event_register` hook on `multi_turn_agent.py`); the event arc `2026-07-09-D3-event-*.md` (5 findings).
