# Attributed entities, fully brain-based END-TO-END — GO 6/6 (2026-06-18, CYCLE 201)

## Headline

Richer-syntax #1 — **attributed entities** ("dog eat **big red apple**") — is now realized
**fully brain-based, end-to-end, in spikes from comprehension through retrieval**, GO 6/6 seeds.
The on-bridge neural attributed parser (CYCLE 200) reads each word's role off the bridge in SPIKES,
the reconstructed `(adjs, noun)` patient is stored + retrieved through the spiking RF
bind/bundle/unbind composer (the ready back end), and `query_patient` returns "big red apple":
attributed end-to-end **0.993**, flat-SVO un-regressed **1.000**, the no-confab moat holds.

The only host steps are the environment (the token string) and zipping words to the
spike-read-out roles — the cognition (parse + bind/store + unbind/retrieve) is neural.

## The three-layer validation of richer-syntax #1

| layer | what | result |
|-------|------|--------|
| CYCLE 199 (host cheap-first) | adjacency-extended conjunctive parser (host readout) + composer round-trip | 0.993, 6/6, flat-only control 0.000 |
| CYCLE 200 (brain-based parse) | the parse in SPIKES (from-start × from-END × voice conjunction on the bridge) | role read-out 1.000, 6/6, NO-END control 0.889 |
| **CYCLE 201 (end-to-end)** | **neural parse (spikes) → RF store/query (spikes)** | **0.993, 6/6, flat un-regressed 1.000, moat holds** |

## Results (6 seeds: 42–47, GPU)

| metric | mean | per-seed |
|--------|------|----------|
| attributed end-to-end round-trip | **0.993** | 1.0 / 1.0 / 0.958 / 1.0 / 1.0 / 1.0 |
| flat-SVO end-to-end non-regression | **1.000** | all 1.0 |
| no-confab moat (unstored cue → abstain) | True | all True |
| seeds GO | **6/6** | — |

## Significance

This is the **first richer-than-flat-SVO conversational capability** realized end-to-end in spikes
on the project's validated substrate — the first concrete capability advance on the owner's primary
direction (basic-LLM-competitive conversation) beyond the flat-SVO + settled-bind resting point.
The scoping's key insight held: the composer back end was already richer than the parser front end,
so richer syntax is a parser-front-end build — and the brain-based parser (from-END conjunction =
adjacency-to-the-head) + the ready composer compose end-to-end with no regression.

## Honest scope + next

- Scope: active "S V adj* N" (1–2 adjectives) + flat SVO. Passive-attributed and ≥3 adjectives are
  bounded follow-ons (more conjunction buckets + teacher frames).
- The de-risk uses the standalone `AttributedBridgeParser` (CYCLE 200) + the `RFPhasorComposer`.
  Threading the attributed parser into the PRODUCTION `BrainConversationalAgent.hear` (so the
  production agent comprehends attributed sentences without a separate runner) is the integration
  step — well-specified, mirroring the CYCLE-188 agent-level pattern.
- Then richer-syntax #2 (multi-frame comprehension, the dual of the already-GO generation
  frame-conditioning).
- NO `sim/` edit anywhere in the richer-syntax #1 arc (reuse-by-import throughout).

## Reproduce

```bash
SIM_BACKEND=cupy python -u -m research.runners._phaseB_neural_attributed_endtoend_derisk
```
Runner: `research/runners/_phaseB_neural_attributed_endtoend_derisk.py`. Prior layers:
`2026-06-18-attributed-entity-parser-GO.md` (host), `2026-06-18-neural-attributed-parser-GO.md`
(neural parse). Scoping: `2026-06-18-richer-syntax-conversational-frontier-scoping.md`.
