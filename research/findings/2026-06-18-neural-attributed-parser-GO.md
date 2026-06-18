# Neural attributed-entity parse — brain-based, GO 6/6 (2026-06-18, CYCLE 200)

## Headline

The attributed-entity parse ("dog eat **big red apple**") is now **brain-based** — realized as
neurons/synapses on the bridge, not a host readout. An on-bridge Hebbian parser whose conjunction
space is extended from `(position × voice)` to `(position-from-START × position-from-END × voice)`,
with the role set grown 3→5 ({agent, action, patient, attribute, attribute2}), reads out the
per-position role **in spikes at 1.000, all 6 seeds**, with flat-SVO un-regressed (1.000), and the
spiking from-END conjunction unit is load-bearing.

This is the BRAIN-BASED-ONLY realization of the CYCLE-199 cheap-first GO (which used a closed-form
host readout): the parse itself is now the firing of conjunction units selecting role ensembles.

## Mechanism

The flat-SVO `BridgeParser` learns `(position × voice) → role` by Hebbian co-firing (6 conjunction
units → 3 role ensembles). The attributed extension adds **position-from-END** as a second
conjunctive factor — the biological "adjacency-to-the-head" cue: the head noun is end-position 0
(= patient), the preceding modifiers are attribute/attribute2. A conjunction unit per
`(s_bucket ∈ 0..3, e_bucket ∈ 0..2, voice)` Hebbian-learns → its role ensemble (the validated v16
embodied-co-firing teacher). At parse time, driving a word's `(s, e, voice)` conjunction ALONE
reads its role off the bridge in spikes — no host role rule. The from-END factor disambiguates the
one position pure-from-start cannot: at from-start 2, a flat patient (`e=0`) vs an attribute
(`e≥1`) share the same from-start bucket; the from-END conjunction tells them apart.

## Results (6 seeds: 42–47, GPU)

| metric | mean | per-seed |
|--------|------|----------|
| attributed role read-out (spiking, on the bridge) | **1.000** | all 1.0 |
| flat-SVO non-regression | **1.000** | all 1.0 |
| NO-END control (drop the from-END conjunction) | **0.889** | all 0.889 |
| seeds GO (attr≥0.90 ∧ flat≥0.90) | **6/6** | — |

GO bar (frozen): attributed ≥ 0.90 AND ≥ 5/6 seeds AND flat-SVO un-regressed AND the NO-END control
< attr − 0.10 (i.e. the from-END factor is load-bearing). Met (1.000, 6/6, control 0.889 < 0.90).

## Honest scope

- **The from-END control is a SOFT 0.889, not a hard collapse** — and that is the faithful result,
  not a weak anti-cheat. Most of the attributed structure (agent at from-start 0, action at 1, the
  head noun via from-start position) is recoverable from from-start alone; the from-END factor's
  specific, necessary job is the single ambiguous position (head-noun=patient vs modifier=attribute
  at from-start 2). It resolves exactly that, lifting 0.889 → 1.000. So the spiking from-END
  conjunction is load-bearing for the disambiguation it is designed for, not a global crutch.
- Scope: active "S V adj* N" (1–2 adjectives) + flat SVO active/passive. Passive attributed and
  ≥3 adjectives are bounded follow-ons (more conjunction buckets + teacher frames).
- This validates the PARSE in spikes; the end-to-end agent wire-in (thread `AttributedBridgeParser`
  into `BrainConversationalAgent.hear` so the production agent comprehends "big red apple" and the
  composer stores the `(adjs, noun)` patient) is the next step, mirroring the CYCLE-188 agent-level
  pattern.

## Anti-cheat controls (passed)

- **NO-END control < bar:** dropping the from-END conjunction (so the parser sees only from-start ×
  voice) drops attributed accuracy to 0.889 < 0.90 — the from-END factor is necessary.
- **Flat-SVO non-regression (1.000):** the extended 5-role parser still reads flat 3-word SVO
  (active + passive) correctly.
- **The read-out is SPIKING:** the role is the max-firing role ensemble on the bridge
  (`cp_firing_states`), driven by the conjunction unit — neurons/synapses, not a host rule.

## Reproduce

```bash
SIM_BACKEND=cupy python -u -m research.runners._phaseB_neural_attributed_parser_derisk
```
Runner: `research/runners/_phaseB_neural_attributed_parser_derisk.py`. Prior (host cheap-first):
`2026-06-18-attributed-entity-parser-GO.md`. Scoping:
`2026-06-18-richer-syntax-conversational-frontier-scoping.md`.
