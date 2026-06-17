# Multi-turn dialogue (anaphora across a turn boundary) — cheap-first de-risk GO (3 seeds)

**Date:** 2026-06-17
**Status:** **GO, 3 seeds.** A persistent spiking working-memory loop carries the discourse referent across a turn
boundary, so a turn-2 anaphor ("it") resolves to the held concept and the production composer answers correctly —
with the reset / attractor-lesion / empty-WM controls all collapsing. This is the natural Phase-2 of the
just-shipped multi-hop reasoning (`2026-06-17-multihop-query-chain-GO.md`).

## What was de-risked

Per the scoping (`2026-06-17-multihop-reasoning-multiturn-dialogue-scoping.md`, Option 3): multi-turn dialogue
needs discourse state held *across* turns. The spiking cortico-PFC loop `SpikingLoopContextBuffer`
(`content_selection_spiking.py`) is a validated working memory (holds a driven concept via a pattern-specific
attractor). The new thing tested here is the **integration**: hold turn-1's salient referent in the loop, do NOT
reset between turns, and on turn 2 resolve "it" by *reading the held attractor*, then answer with the composer.

```
Turn 1 (user): "dog chase cat"      -> store the fact + write the object referent (cat) into the WM loop
Turn 2 (user): "what does it eat?"  -> "it" = read the WM loop -> cat -> composer.query_patient(cat, eat) -> fish
```

## Result (`_phaseB_multiturn_anaphora_derisk.py`, CPU, ~600-neuron WM bridge)

| seed | persistent: "it" → / answer / specificity | reset control | lesion control | empty-WM moat |
|---|---|---|---|---|
| 42 | **cat / fish** / 574× | None / None | worm / None | **None (abstain)** |
| 43 | **cat / fish** / very high | None / None | None / None | **None (abstain)** |
| 44 | **cat / fish** / 1750× | None / None | None / None | **None (abstain)** |

**GO, 3/3.** The persistent loop resolves the anaphor to the **right** referent (cat dominates the read by
≫1.5×, often hundreds-fold) and the composer answers **fish** — across the turn boundary, every seed. Every
control collapses:

- **Reset between turns** → the referent is wiped → "it" unresolved (None), no answer. The *persistence* is
  load-bearing.
- **Attractor-lesion** (zero the c2d/d2c loop weights) → no reverberation → the referent decays → resolution
  fails (None or a wrong concept). The *attractor loop* is the mechanism.
- **Empty WM** (no turn-1 referent) → no dominant concept → **abstain** (None). No confabulated antecedent — the
  no-confab discipline extends to anaphora.

## Reading it honestly

- This de-risks the **mechanism** (persistent WM carries a referent across a turn boundary, moat-safe), reusing
  the already-validated WM loop + the production composer. It is not yet a production multi-turn agent: the
  "salient referent" is written explicitly (the object of the last sentence), and the anaphor is the single
  pronoun "it". A production `MultiTurnAgent` would (a) choose the salient referent from the parse, (b) handle
  multiple held referents (the ~7±2 WM span the loop already supports) and several anaphor types, (c) carry the
  intermediate filler of a multi-hop chain in the same loop (uniting Options 1 + 3). Those are the follow-on
  build, not this de-risk.
- The specificity is enormous (hundreds- to thousands-fold) because the loop holds a single clean attractor; the
  realistic stress is holding *several* referents at once and resolving *which* one a pronoun binds — the
  bounded next test.

## Next (the production build, if pursued)

A `MultiTurnAgent` wrapping `BrainConversationalAgent` + a persistent `SpikingLoopContextBuffer`: `hear()` writes
the salient referent; an anaphor in the next turn reads the loop; the multi-hop `query_chain` intermediate is
carried in the same loop (so the chain's working state is genuinely neural, not a Python variable). Reuse-by-import;
no `sim/` edit anticipated.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._phaseB_multiturn_anaphora_derisk --seeds 42 43 44
```

No `sim/` edit. Reuse-by-import: `SpikingLoopContextBuffer` (validated spiking WM) + `RFPhasorComposer.query_patient`
(validated single-hop retrieval + its moat).
