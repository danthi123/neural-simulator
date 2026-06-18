# Attributed-entity parser (adj+noun comprehension) — GO, the first richer-syntax capability (2026-06-18, CYCLE 199)

## Headline

The cheapest richer-syntax capability toward basic-LLM-competitive conversation — **attributed
entities** ("dog eat **big red apple**") — is realized **GO, 6/6 seeds**, as a **parser-front-end
build on the composer back end that was already ready**. An adjacency-extended conjunctive parser
parses "S V adj* N" into {agent, action, attribute, attribute2, patient} and round-trips through
the UNCHANGED `RFPhasorComposer` (store the `(adjs, noun)` patient → `query_patient` renders "big
red apple"): attributed round-trip **0.993**, flat-SVO **un-regressed 1.000**, the **flat-only
control collapses to 0.000** (load-bearing adjacency factor), the no-confab moat holds.

## What this confirms

The richer-syntax scoping (`2026-06-18-richer-syntax-conversational-frontier-scoping.md`) found the
composer BACK END is already richer than the parser FRONT END: `RFPhasorComposer.store` already
accepts a `(adjs, noun)` patient → attribute/attribute2 roles, and `query_patient` renders the
attributed entity. So attributed entities were a PARSER-ONLY gap. This de-risk closes it.

The conjunctive insight generalizes from voice to **adjacency**: the flat-SVO parser learns
(position × voice) → role; the attributed parser adds **position-from-END** (the head noun is end-0
= patient; the preceding modifiers = attribute, attribute2) — adjacency-to-the-head is the new
conjunctive factor, exactly like voice was. The closed-form readout, fit on the canonical frames
(flat SVO active/passive + attributed 1-adj + attributed 2-adj), assigns roles structurally
(word-independent), so it generalizes to any adj+noun combination.

## Results (6 seeds: 42–47)

| metric | mean | per-seed |
|--------|------|----------|
| attributed round-trip (held-out combos) | **0.993** | 1.0 / 1.0 / 0.958 / 1.0 / 1.0 / 1.0 |
| flat-SVO non-regression | **1.000** | all 1.0 |
| FLAT-ONLY control (anti-cheat — must fail) | **0.000** | all 0.0 |
| no-confab moat (empty store → abstain) | True | all True |
| seeds GO (attr≥0.90 ∧ flat≥0.90 ∧ moat) | **6/6** | — |

GO bar (frozen): attributed round-trip ≥ 0.90 AND ≥ 5/6 seeds AND flat-SVO un-regressed AND the
flat-only control < 0.30. Met decisively (6/6, control 0.000).

## Anti-cheat controls (all passed)

- **FLAT-ONLY control collapses (0.000):** the original 3-word `ConjunctiveParser` returns None for
  a >3-word attributed sentence — it cannot parse adjectives — so the adjacency-extended factor is
  genuinely load-bearing, not an artifact.
- **Held-out combos:** the readout is word-independent (structural), so role assignment is
  leakage-free by construction; the test draws fresh adj+noun combos as the generalization probe.
- **Flat-SVO non-regression (1.000):** the extended parser still parses 3-word sentences correctly.
- **The round-trip is real:** the attribute recovery goes THROUGH the composer's spiking RF
  bind/bundle/unbind + cleanup (the back-end mechanism), not a host echo.
- **No-confab moat:** an empty-store `query_patient` abstains (the moat is parser-independent).

## Honest scope + next

- The parser here is the LEARNED closed-form conjunctive readout (the `ConjunctiveParser` family),
  not yet the fully-neural `BridgeParser`. The de-risk's purpose was the cheapest falsification of
  "can an adjacency-extended conjunctive parser + the ready composer do attributed entities" — GO.
- NEXT (the wire-in): realize the adjacency factor in the neural `BridgeParser` (the parser's firing
  selects the attribute role) + thread attributed patients through `BrainConversationalAgent.hear`
  so the production agent comprehends "big red apple" end-to-end. Then richer-syntax #2 (multi-frame
  comprehension, the dual of the already-GO generation frame-conditioning).
- This is the first concrete advance on the owner's PRIMARY (basic-LLM-competitive conversation)
  since the conversational stack reached its flat-SVO + settled-bind resting point.

## Reproduce

```bash
SIM_BACKEND=numpy python -u -m research.runners._phaseB_attributed_entity_parser_derisk
```
Runner: `research/runners/_phaseB_attributed_entity_parser_derisk.py`. Scoping:
`2026-06-18-richer-syntax-conversational-frontier-scoping.md`.
