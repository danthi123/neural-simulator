# Roadmap phase 2, STEP A3 — the `OneBrainComposer`: the whole who/what pipeline on ONE persistent brain: GO

**Date:** 2026-06-18 (the real "one brain" headline arc). **Status:** **GO** — uniform across 3 seeds × 2 D (6/6).
This is the **capstone** of Phase 2: the validated GO pieces of the arc assembled into one production composer that
runs the whole who/what conversational turn — comprehend, store, query, abstain — on ONE persistent spiking bridge,
with no host round-trips between operations.

**Runner:** `research/runners/_phaseB_onebrain_composer_derisk.py` (the `OneBrainComposer` class + the de-risk) |
**Raw:** `research/findings/raw/_phaseB_onebrain_composer.json`

## What it assembles (each piece separately GO this arc)

| piece | finding | role in `OneBrainComposer` |
|---|---|---|
| parser front-end (GAP B) | `2026-06-18-one-brain-parser-frontend-GO.md` | the parser slice `[0:P]` comprehends a sentence; its neural role firing selects each word's bind (voice-invariant) |
| multi-fact store (GAP A) | `2026-06-18-one-brain-multifact-store-GAP-A-GO.md` | each fact = a 3-role composite written into a `(1+D)` trigger→readout block in the bridge's complex weights (to K=32) |
| cue-matching scan | (this finding, validated within A3) | a who/what question finds the matching stored fact (reconstruct + unbind cue roles + cleanup + first-match) |
| cleanup + moat (3a/3b) | `2026-06-18-one-brain-{cleanup,moat}-onbridge-GO.md` | the matched-filter answer read + the abstain-when-no-fact |
| 4-role coherence (3c) | `2026-06-18-one-brain-multirole-coherence-GO.md` | phase coherence across the bundled binds |

The parser (Izhikevich, voltage in `v`/`u`) and the resonate-and-fire composer registers (a complex phasor in `v`/`u`)
co-reside as disjoint slices on ONE bridge — the merged-bridge regime, with the resonate-and-fire operations masked to
their slice.

## The API (mirrors `RFPhasorComposer`, so the agent can use it)

- `hear(sentence, voice)` — the parser comprehends the sentence (the role it fires for each word selects that word's
  bind); the three binds bundle into a composite; the composite is appended to the persistent store as a new fact's
  trigger→readout complex weights. Comprehension and storage are on the bridge; the host passes only the text in.
- `query_patient(agent, action)` / `query_agent(action, patient)` — the cue-matching scan: each stored block is
  reconstructed (fire its trigger) and all three roles are unbound **in parallel** into separate registers and cleaned
  up (one reconstruction + two resonate windows per block — no reconstruct-per-read, so no phase drift); the first block
  whose cue roles match returns its answer role.
- `ask_yes_no(agent, action, patient)` — "yes" if the full subject-verb-object matches a stored fact, else "unknown"
  (the no-confab moat — abstain rather than assert a falsehood).

## Result — 3 seeds × {D=64, D=128}

| metric | result (mean, 6/6) |
|---|---|
| `query_patient` == ground truth | **1.000** |
| `query_agent` == ground truth | **1.000** |
| `ask_yes_no` == ground truth ("yes" for stored) | **1.000** |
| host-parity (patient / agent / yes-no) == numpy `RFPhasorComposer` | **1.000 / 1.000 / 1.000** |
| moat: absent cue → None (abstain) | **1.00** |
| moat: unstored fact → "unknown" (abstain) == oracle | **1.00** |
| voice-invariance: a fact stored via its passive frame queries back | **1.00** (the 6th fact, every seed) |

Every one of the 6 configs is full GO on all metrics.

## Reading

- **The whole who/what conversational turn runs on one persistent brain.** Comprehend a sentence (the parser's
  firing), store the fact in synapses, query a role by cue, abstain when there is no fact — all on one
  `SimulationBridge`, the value flowing operation-to-operation through complex synapses, the host doing only text I/O.
  Every answer matches the validated numpy composer and the ground truth.
- **Comprehension is neural and voice-invariant.** One of the six facts is stored via its passive frame; it still
  queries back correctly, because the parser genuinely assigns roles from sentence structure (not word position).
- **The no-confab moat is preserved end-to-end.** An absent cue returns None; an unstored fact returns "unknown" (==
  the oracle's abstention). The integration does not weaken the moat.

## Honest scope + next

- This first `OneBrainComposer` cut handles **affirmative** facts (who / what / affirmative yes-no). **Negation** (a
  bound polarity tag = a 4th role) is a documented follow-on — the production `RFPhasorComposer` binds it; the
  `OneBrainComposer` can add the same 4th-role bind (the 4-role coherence is already GO).
- The store-write reads the composite to the host once to install the block weights (a store-time consolidation hop);
  the all-synaptic store-write is a later refinement (the query path already has no host round-trips between ops).
- **Next (the production wiring):** swap the `OneBrainComposer` into `BrainConversationalAgent` — the agent's
  `hear(sentence)` delegates comprehension+storage to the composer (one parser on the bridge, not the agent's separate
  parser), with a CI guard test running the agent's existing capability suite against this composer == the numpy
  oracle. Then A4 (the optional spiking winner-take-all selection) and A5 (make it the default + megakernel the
  persistent loop + retire the legacy numpy runtime, keeping numpy as the test oracle).

## Reproduce
```bash
SIM_BACKEND=cupy python -u -m research.runners._phaseB_onebrain_composer_derisk --seeds 42,43,44 --dims 64,128
```
