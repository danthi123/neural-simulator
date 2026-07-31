---
type: finding
status: live
date: 2026-06-10
---

# Conversational vocab / capability ceiling — multi-seed: GO (the consolidated agent scales to 320 concepts, no-confab moat 100%)

> **Result: GO multi-seed.** The full `BrainConversationalAgent` capability matrix — comprehension + fact memory + who/what Q&A, **abstention (the no-confab moat)**, negation/yes-no, embedded clauses, one-attribute, two-attribute, generation, dialogue — holds at **V=320 concepts across 6 seeds (42–47)**. The load-bearing no-confab moat is **100% (20/20) in every cell**, the two-attribute binding (the old K=5 boundary) **resolves everywhere**, and the shuffled-fact control has **zero false hits everywhere**. The only degradation is the embedded clause, and it is a pure **code-dimension floor**: 5/6 seeds pass at D=128; all 6 pass at D=256.

This closes the owner-raised question — "test the more-biological sim's conversational performance at larger / more-complex vocab before the cortex work." The agent's conversational ceiling is far above the V=16 probe scale it was being exercised at; the production FHRR-on-bridge composer carries the full agent loop to 320 concepts.

## Method

`research/runners/vocab_ceiling_probe.py` (built in the cheap-first pass, commit `7cc90f7f`) builds the `BrainConversationalAgent` (Hebbian parser bridge + the resonate-and-fire `RFPhasorComposer` + the dlPFC dialogue planner) on a V-word vocabulary and scores the full capability matrix as pass/fail, with two anti-cheats per cell: an **abstention floor** (20 unstored cues that must all return "I don't know") and a **shuffled-fact permuted control** (wrong-queries must abstain). The RF composer self-generates its phasor codes from the seed, so only a V-word list is needed (the first V words of the curated 320-word list). GPU (`SIM_BACKEND=cupy`); ~3 min/cell. Sweep driver: `research/findings/raw/_run_vocab_ceiling_multiseed.sh`.

## Results — V=320, 6 seeds (42–47)

| seed | D=128 | D=256 |
|---|---|---|
| 42 | GO (8/8) | GO (8/8) |
| 43 | PARTIAL — embedded clause 0/1 | **GO (8/8)** |
| 44 | GO (8/8) | GO (8/8) |
| 45 | GO (8/8) | GO (8/8) |
| 46 | GO (8/8) | GO (8/8) |
| 47 | GO (8/8) | GO (8/8) |

**Every cell, every seed, both D:** abstention moat **20/20**, two-attribute **1/1**, shuffled-control false hits **0**. (The cheap-first pass also confirmed GO at V=64 and V=128, seed 42.)

## The per-capability degradation map (the spec step-3 cortex inherits)

- **Holds at D=128 through V=320, all seeds:** comprehension + Q&A, the no-confab moat, negation/yes-no, one-attribute, **two-attribute** (the old K=5 boundary, now lifted on the production agent), generation, dialogue.
- **Embedded clause:** needs **D≥256** at V=320 (a code-dimension floor — at D=128 the recursive clause's bound code is below the algebra's SNR at 320-concept density on ~1/6 seeds; D=256 restores it multi-seed). This is the single, characterized, dimension-tunable degradation.

## Honest caveat (the genuinely-new signal)

The composer is a principled **idealization** — an exact-inverse vector-symbolic algebra that demands clean codes — so a clean pass at 320 is "the algebra works at 320," not "more brain-like." The genuinely-new information is narrow and real: (i) the algebra holds on the **consolidated nav+conv one-bridge substrate + the full agent loop** (parser + composer + dlPFC together), not just the kernel in isolation; and (ii) the **degradation map** above — abstention-at-scale (the hardest acceptance bar) never breaks, and the only code-dimension dependence is the recursive clause. That degradation map is exactly the spec a learned spiking-cortical binder (step 3) inherits: it must match this matrix while learning lossy, robust codes that don't demand the algebra's clean-code precision.

## Trail

- Cheap-first (V=64/128/320 single-seed): `2026-06-10-vocab-ceiling-V64-cheap-first-GO.md` (commit `7cc90f7f`).
- Design: `docs/plans/2026-06-10-conversational-vocab-ceiling-characterization-design.md`.
- Raw per-cell JSON: `research/findings/raw/_vocab_ceiling_V320_s4{2..7}_D{128,256}.json`.
