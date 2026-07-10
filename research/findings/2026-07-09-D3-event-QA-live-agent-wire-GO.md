# D3 EVENT QA → the LIVE MultiTurnAgent (6-seed GO, host + fully-spiking): the RANK-3 QA DEPLOYED — the brain answers "what does HE eat?" over a running discourse

**Date:** 2026-07-09
**Runner:** `research/runners/_d3_event_qa_wire_derisk.py` (reuse-by-import: `D3EventRegister` + the additive `MultiTurnAgent.what_does_agent_now`; numpy; NO `sim/` edit).
**Verdict:** GO (6-seed dev 42/43/44 + blind 100/101/102, BOTH host and `--spiking`).

## What this closes
The rate + fully-spiking QA rungs proved the situation-model × fact-store QA de-risk. This **deploys it on the real agent**: a `MultiTurnAgent` ANSWERS a property question over a running discourse, unifying its D3 event register (the running meaning) with its own fact-store KB.

```
1. TEACH separate knowledge:  "dog eat worm. cat eat ball. bird eat fish. ..."   -> the composer KB (fact store)
2. RESET the event register (clear the running event; the KB persists).
3. HEAR a deep-coref discourse: "dog chase cat. he chase fish. he chase bird. he chase worm."
                                 -> the D3 register composes: agent = DOG (set@0, persisted through the 'he' run = coref-DEEP)
4. ASK "what does HE eat?"  -> what_does_agent_now("eat"): resolve 'he' -> DOG (register) -> query KB (dog eat ?) -> "worm"
```
The answer needs BOTH the running event (to resolve 'he' to the deep agent — the chase-discourse never names "he" as an entity) AND the fact store (the eat-KB, taught separately, never in the chase-discourse). A retrieve-a-set→render loop has neither a running referent nor a way to key the store with it.

## Result (6-seed; NO `sim/` edit)
| the DEPLOYED agent answers "what does HE eat?" | host | fully-spiking (`--spiking`) |
|---|---|---|
| **LIVE-agent EVENT-QA** | **1.00** all 6 seeds | **1.00** all 6 seeds |
| FLAT-FACT ('he' unresolved → no KB entry) | 0.00 | 0.00 |
| RECENCY (last-mentioned's eat-fact = wrong agent) | 0.00 | 0.00 |

Unanimous, decisive, on the real agent object, both backends (the `--spiking` variant maintains the running event on two FS-WTA Izhikevich attractor slots — the whole resolve on spikes).

## The wire (additive, `research/runners/`, NOT `sim/`)
`MultiTurnAgent.what_does_agent_now(action)`: resolve the current agent from the running-event register (`who_agent_now`, the DEEP coref track — NOT the WM/recency), THEN query the KB (`agent.what_does`). Default-safe (returns None with no register). Mirrors the anaphora `focus_bias_source` + the event-agent `who_agent_now` hooks.

## Anti-cheats (all pass)
- **(a) EVENT-QA ≫ FLAT-FACT + RECENCY** (1.00 vs 0.00 / 0.00): the answer comes from the composed running referent, not the last literal subject ("he", unresolved) or the last-mentioned entity.
- **(b) the eat-KB is SEPARATE** (taught before, `register.reset()` between, never in the chase-discourse) → the answer is not readable from the discourse; it requires keying the store with the composed referent.
- **(c) fully-spiking** (`--spiking`): the register maintains the running event on the FS-WTA substrate; the QA still 1.00.

## Honest scope + next
- **Deployment demonstration on the 4 canonical deep scenarios** (matching the event-agent wire), incl. a PROMOTE ("it") case. The held-out-DEEPER *generalization* (hundreds of length-6/7/8 discourses at ~0.98) is the earlier rate + spiking QA rungs (`2026-07-09-D3-event-QA-{unification,fully-spiking}-GO.md`); this rung shows it runs end-to-end on the real agent + KB + spikes.
- The eat-KB query uses the composer's `query_patient`; the resolution uses the D3 register — two existing spiking pieces, now unified in one deployed method.
- **Next:** multi-turn QA / connectives (chain two composed events / relate them); the self-supervised TEM δ (removes the endpoint label — the genuinely-open crux); more query types (what-does-the-patient-do, is-it-true over the running event).

## Files
`research/runners/_d3_event_qa_wire_derisk.py`; `research/runners/multi_turn_agent.py` (`what_does_agent_now`/`what_does_patient_now`); the QA rungs `2026-07-09-D3-event-QA-{unification,fully-spiking}-GO.md`.
