# Cross-sentence pronoun COHERENCE on the spiking substrate — cheap-first de-risk (GO, 6/6)

**Date:** 2026-06-17
**Status:** **GO, 6 seeds (42 43 44 100 101 102), CPU/numpy.** The conversational agent now produces COHERENT
multi-sentence output: when a referent introduced in an earlier sentence RECURS as the subject of a later
sentence, it is rendered as a PRONOUN ("it") that RESOLVES — on the project's spiking resonate-and-fire phasor
substrate, via the validated slot-anaphora — back to the correct ANTECEDENT referent. All four pre-registered
load-bearing tests pass 6/6. **No `sim/` edit; reuse-by-import only.**

## The capability (the fluency increment)

The CYCLE-136 multi-sentence emission (`2026-06-17-multisentence-ordered-emission-derisk.md`) emits one sentence
per ordered-WM slot, but the sentences are INDEPENDENT — "dog ran north. dog saw cat." re-names the dog in full
every sentence (a list, not a discourse). COHERENCE is the next fluency increment: a recurring subject becomes a
pronoun that resolves to its antecedent — **"dog ran north. bird ate worm. then IT ran north."** (the "it" = dog).

## Mechanism (reuse-by-composition, NO new machinery)

Three SEPARATELY-VALIDATED pieces, composed:

| piece | role here | provenance |
|---|---|---|
| `OrderedPositionWM` | hold the referent stream in gamma-slot POSITION phasors on the RF substrate; `read_slot(C, pos_k)` = spiking unbind, familiarity-gated | CYCLE-135 GO (`2026-06-17-ordered-wm-position-binding-derisk.md`) |
| `MultiTurnAgentV2.referent_at(slot)` | the by-slot pronoun RESOLUTION (spiking unbind of an arbitrary held slot) | production GO (`2026-06-17-multiturn-ordered-wm-integration.md`) |
| multi-sentence ordered emission + `describe()`/`render_fact` (neural word order) | render each non-pronominalized sentence | CYCLE-136 GO + serial-order renderer |

**The loop** (`research/runners/_phaseB_cross_sentence_coherence_derisk.py`, class `CoherentDiscourse`): a fresh
order-encoded discourse buffer accumulates referents in surface order as each sentence is processed, tracking
`_slot_of[referent]` = the EARLIEST gamma-slot each referent was introduced at (its **antecedent slot**). When a
sentence's SUBJECT was already introduced at an earlier slot (a recurring subject), the body emits a PRONOUN for
it and RESOLVES the pronoun by reading the antecedent's slot on the spiking substrate
(`agent.referent_at(antecedent_slot)` — a familiarity-gated spiking unbind). The first mention of any subject is a
full-noun sentence (the validated `describe` path). The order comes from the WM slots; the resolution is the
validated slot-anaphora; only the surface pronoun token + the sentence join are the body's emission.

**Why antecedent-slot, not most-recent-slot (the load-bearing design point).** After "dog ran north" the
surface-order discourse window is `[dog(slot0), north(slot1)]` — "north" is the *most-recent* slot. A recurring
"dog" must resolve to the ANTECEDENT dog at **slot 0**, NOT the most-recent slot. So coherence reads the
antecedent's slot (`referent_at(antecedent_slot)`) — exactly the by-slot ADDRESSING the rate-attractor buffer
structurally lacked (its only read was the intrinsic-basin winner). This is the capability the order-encoded WM
uniquely enables, and it is why the order-control (below) flips.

## Results — the four pre-registered tests, 6 seeds

| seed | calib. thr. | (1) COHERENCE | (2) ORDER-CONTROL flip | (3) DISTINCT-REFERENT | (4) NO-CONFAB moat | full |
|---|---|---|---|---|---|---|
| 42  | 0.333 | 1.000 | 1.000 (40/40 flips) | 1.000 | 1.000 | PASS |
| 43  | 0.304 | 1.000 | 1.000 (40/40 flips) | 1.000 | 1.000 | PASS |
| 44  | 0.289 | 1.000 | 1.000 (40/40 flips) | 1.000 | 1.000 | PASS |
| 100 | 0.334 | 0.975 | 0.825 (40/40 flips) | 1.000 | 1.000 | PASS |
| 101 | 0.327 | 1.000 | 1.000 (40/40 flips) | 1.000 | 1.000 | PASS |
| 102 | 0.321 | 1.000 | 1.000 (40/40 flips) | 1.000 | 1.000 | PASS |
| **mean** | — | **0.996** | **0.971** | **1.000** | **1.000** | — |
| **count** | — | **6/6** | **6/6** | **6/6** | **6/6** | **6/6** |

(GO bar = ≥ 5/6 of seeds per control. Trials/seed: coherence 40, order 40, distinct 40, no-confab 30.)

- **(1) COHERENCE (the capability).** Discourse `[sA, sB, sA]` — sA recurs as the 3rd sentence's subject. That
  sentence must be pronominalized AND the pronoun must RESOLVE (spiking slot-anaphora) to sA. **0.996** mean.
- **(2) ORDER-CONTROL (the wall).** `discourse_A=[sA,sB,sA]` (sA recurs) vs `discourse_B=[sB,sA,sB]` (sB recurs):
  the recurring sentence's resolved antecedent must be sA in A and sB in B — it **FLIPS** with WHICH referent
  recurs (the flip is observed 40/40 every seed; scored CORRECT only when both resolve correctly AND differ).
  **0.971** mean. A fixed-entity resolver could not flip; this proves resolution is by the recurring referent's
  own antecedent slot.
- **(3) DISTINCT-REFERENT (load-bearing).** `[sA, sB, sC]`, all distinct → NO sentence is pronominalized; every
  subject stays a full noun rendering its correct stored fact. **1.000** — no spurious pronoun is ever introduced
  when there is no antecedent.
- **(4) NO-CONFAB moat (load-bearing, free).** A pronoun whose antecedent slot was never bound (the dedicated
  never-bound `emptyslot` probe read against the real composite, gated by the familiarity moat) → ABSTAIN (None).
  **1.000** — no confabulated antecedent. (Per the owner's 2026-06-17 moat-relaxation the moat is not a hard gate;
  it is free here from the WM's familiarity gate and reported clean — zero breaches.)

## Example coherent transcript (seed 42)

```
discourse order (subjects): ['dog', 'bird', 'dog']
emitted:  "dog ran north. bird ate worm. then it ran north."
          the pronoun 'it' (recurring subject 'dog') RESOLVED on the substrate to
          antecedent: 'dog' (read from gamma-slot 0)
```

The third sentence pronominalizes the recurring "dog", and the "it" resolves (spiking unbind of gamma-slot 0,
familiarity-gated) back to "dog" — the correct antecedent, not the most-recent slot ("worm"/"north").

## Honest scope

- **Capability + all controls: robustly GO.** Coherence (pronominalize + correct antecedent) and the
  order-control FLIP are 6/6; the distinct-referent control and the familiarity moat are 6/6 clean.
- **The one sub-1.0 seed (100): honest variance, not a control failure.** Seed 100 reads coherence 0.975 / order
  0.825 — both still ≥ 0.80, so it passes — driven by the bundle cross-talk at the lower-fidelity slot reads on
  that seed (the same bundle-noise tail that motivated the calibrated familiarity threshold). It is seed variance
  in the underlying ordered-WM read fidelity, not a coherence-mechanism breakage; the resolution still flips
  (40/40) and abstains (1.000) on seed 100.
- **What this is and isn't.** This de-risks the COHERENCE MECHANISM by composing already-validated parts: a
  recurring SUBJECT is pronominalized and resolved to its antecedent by slot. "Recurring subject → its first-
  introduction slot" is the antecedent heuristic; richer anaphora (a pronoun for a recurring OBJECT, gender/number
  agreement beyond singular "it"/plural "they", or binding by syntactic role rather than antecedent slot) is a
  bounded follow-on — any held slot is readable (`referent_at`), so the substrate supports it; the selection
  policy is the open part. Validated at vocab 10 (6 subjects + objects + an absent probe), D=128, CPU/numpy (the
  spiking RF composer runs each op as a small `SimulationBridge`).
- **Substrate is the deployed one.** Binding/unbinding/bundling/cleanup are the production composer's spiking RF
  operations; the discourse buffer reuses them by import. No new mechanism, no `sim/` edit.

## Contrast (the headline)

| | independent multi-sentence emission (CYCLE-136) | cross-sentence COHERENCE (this de-risk) |
|---|---|---|
| recurring subject | re-named in full every sentence ("dog … dog …") | pronominalized ("dog … then **it** …") |
| pronoun resolution | n/a | **6/6** to the correct antecedent (spiking slot-anaphora) |
| order-control | the emission order permutes with slots (validated) | the resolved antecedent **FLIPS** with which referent recurs, **6/6** |
| no spurious pronoun | n/a | distinct subjects stay full nouns, **6/6** |
| no-confab | unknown topic abstains | absent antecedent abstains, **6/6** |

## Reproduce

```bash
SIM_BACKEND=numpy python -u -m research.runners._phaseB_cross_sentence_coherence_derisk \
    --seeds 42 43 44 100 101 102
```

CPU (numpy backend). Deliverables: `research/runners/_phaseB_cross_sentence_coherence_derisk.py`,
`research/findings/raw/_phaseB_cross_sentence_coherence.json`. No `sim/` edit; no git commit (controller commits
after verifying).
