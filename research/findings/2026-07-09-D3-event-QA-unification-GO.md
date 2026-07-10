# D3 EVENT → RANK-3 reasoning/QA over the composed running event, UNIFIED with the fact store (6-seed GO)

**Date:** 2026-07-09
**Runner:** `research/runners/_d3_event_qa_derisk.py` (reuse-by-import: `make_event_task` + `factored_event_rnn` [the RANK-1 composer]; numpy; NO `sim/` edit).
**Verdict:** GO (6-seed: dev 42/43/44 + blind 100/101/102).

## What this is — the anti-RAG payoff
The event-composition arc built a running FACTORED (agent, patient) MEANING on the discrete-attractor. This rung is the research-gate's **RANK-3 payoff** ("cheap once RANK-1 exists"): the QA **reads the composed running event** to resolve a pronoun, then **keys a SEPARATE fact store** for the answer — joining the situation model to stored knowledge. The answer is derivable from *neither alone*:

```
discourse:  "dog chases cat. he chases fish. it flees bird."   -> D3 composes: agent=dog (coref-DEEP via 'he'), patient=bird
fact store: dog->meat, cat->fish, bird->seed ...               -> SEPARATE knowledge, NEVER uttered in the discourse
question:   "what does HE eat?"   -> resolve 'he' to the running AGENT (dog, traced through the coref run by D3, NOT the
                                      last-mentioned) -> key the store -> "meat"
```
A retrieve-a-set→render-each→concatenate loop **cannot** do this: it has no running referent to key the store with. (Frankland-Greene 2015 lmSTC agent/patient registers + a Collins-Quillian property store; the QA = read-the-register → associative-recall.)

## Result (6-seed; NO `sim/` edit)
The **AGENT** slot is the coref-DEEP referent (persists through the `he`-run back to a random-depth setting) = the load-bearing test. The PATIENT slot is shallow-by-design ("it" = the most-recent object, which is realistic) = the secondary "both slots are QA-able" check.

| QA over the composed running event | dev 42/43/44 | blind 100/101/102 |
|---|---|---|
| **AGENT(deep)-QA** (D3-composed) | 0.979 / 0.983 / 0.983 | 0.985 / 0.976 / 0.983 |
| RECENCY-QA (resolve→last-mentioned) | 0.390 / 0.381 / 0.377 | 0.352 / 0.363 / 0.388 |
| NO-EVENT lesion (recurrence-off) | 0.398 / 0.350 / 0.373 | 0.367 / 0.357 / 0.389 |
| PATIENT-slot-QA (2nd slot) | 0.995 / 0.998 / 0.998 | 0.999 / 0.997 / 0.999 |

Mean deep-agent QA **≈0.982** vs recency **≈0.375** vs no-event **≈0.372** (chance 1/K=0.167). Clean, decisive, consistent across dev + blind.

## Anti-cheats (all pass)
- **(a) COMPOSED ≫ chance:** 0.98 vs 0.167.
- **(b) RECENCY FAILS on the deep agent (0.375):** the running agent ≠ the last-mentioned (it persists through the coref run) → a recency resolver keys the store with the WRONG referent. (Recency is a *fair* floor: it correctly nails the shallow PATIENT slot at 0.99, so it is not artificially crippled — it fails specifically on the deep referent.)
- **(c) NO-EVENT lesion collapses (0.372):** zero the recurrent state (current-token-only) → no running referent → ~recency.
- **(d) PERMUTED-fact-store:** the answer TRACKS the permuted store (`AGENT_qa_permstore == AGENT_qa`) → the answer routes THROUGH the store (`prop[resolved]`), not a discourse shortcut; the property values never appear in the discourse, so the model cannot emit them without keying the store.
- **(e) BOTH slots QA-able:** agent AND patient (0.98 / 0.99) → a 2-slot situation model, not 1.

## Honest scope + next
- **This is resolve-referent + recall (the first RANK-3 rung), not multi-hop inference.** The QA reads the composed running referent and applies a fixed associative store. The D3 event is the load-bearing new piece (it lets QA key the store with a *running, deep* referent where recency fails); the store is the existing composer's role. Full multi-hop reasoning / discourse connectives are the next rung.
- Reuses the **already-adversarially-verified deep (coref) task** (the 2-token-shallow confound the skeptic caught on the RANK-1 task is pre-handled here).
- **Next:** multi-turn QA / connectives (chain two composed events); the **spiking QA read-out** (the resolve+recall on the substrate — the composed slot already ports to the spiking FS-WTA; add the associative-recall read-out); the self-supervised TEM δ (removes the endpoint label).

## Files
`research/runners/_d3_event_qa_derisk.py`; the event arc `2026-07-09-D3-event-*.md`; the D3 arc `2026-07-09-D3-*.md`.
