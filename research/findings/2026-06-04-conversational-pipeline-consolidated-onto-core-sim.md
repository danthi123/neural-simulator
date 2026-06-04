# Conversational pipeline CONSOLIDATED onto the core sim (the brain) — 2026-06-04

**One line:** Per the owner's directive ("the core sim IS the simulated brain; capabilities realized through it, no
bolted-on modules"), the conversational pipeline now runs as a genuine spiking computation **on the core
`SimulationBridge`** — comprehend (Hebbian parser bridge) + SVO fact memory / who-what Q&A / abstention / negation
/ clauses / one-attribute (coincidence-neuron composer bridge) + dialogue planning (dlPFC content-selection bridge)
— with the two bolted-on numpy "spiking" simulators (`spiking_phasor_fhrr`, `resonate_fire_fhrr`) **relabelled as
numpy reference, not the production substrate**. Honest residual: two-attribute is a documented boundary; the FHRR
F=3 resonator stays a numpy reference.

## Why (the directive + the audit)

The owner asked, before scaling, to consolidate so the sim is clean + self-contained. The substrate audit
(`2026-06-04-conversational-pipeline-substrate-audit.md`) found the uncomfortable truth: the "capstone" + all the
unified composition agents ran on **standalone numpy spiking simulators**, NOT the core `SimulationBridge` — while
the core sim already had **validated** spiking realizations of 11/13 capabilities, sitting as archived
`_insubstrate_*` probes used only by demos. So this was assembly + promotion, not new research (except the
attribute gap).

## What was built (4 phases, plan: `docs/plans/2026-06-04-consolidate-conversational-pipeline-onto-core-sim-design.md`)

**Phase 1 — `research/runners/core_sim_composition.py` (`CoreSimComposer`).** The role-filler VSA composition
promoted from the archived probes into ONE clean, self-contained, tested module. A ±1 Hadamard computed by
**coincidence neurons on a real `SimulationBridge`** (6400 Izhikevich neurons): bind/unbind, SVO fact memory,
who/what Q&A, abstention (the no-confab moat = None when no fact's agent matches the cue), negation/yes-no (a bound
polarity tag). Concept codes are the substrate's own (the `denoise64` concept-pool-activity cache) — grounded in
the brain. 3 → 5 regression tests pin the frozen bars (recovery ≥ 0.80, faithful control).

**Phase 2 — `research/runners/brain_conversational_agent.py` (`BrainConversationalAgent`).** The full conversational
loop on the brain, assembling validated core-sim bridges: a **Hebbian-learned parser bridge** (comprehension: the
(word-position × voice) → role mapping, voice-invariant — active "dog go north" and its passive frame assign the
same agent) + the composer + **clauses** (recursive role-filler: "dog look (cat go south)" → "cat go south") +
**dialogue planning** (`elaborate(topic)` via the dlPFC spiking content-selection Control over an association graph
built from the agent's own facts). 5 on-brain tests pass: comprehend+Q&A+abstention, voice-invariant
comprehension, negation, clause, dialogue planning. **No bolted-on numpy simulator anywhere in the path.**

**Phase 3 — attributes (the one gap), honest 3-state outcome.** The ±1 coincidence scheme cannot invertibly bind
two concept codes (adj⊗noun) — that's exactly why the resonator was the gap. Resolved on the brain via a
**feature-binding ATTRIBUTE role-tag** ("big apple" = patient⊗apple + attribute⊗big, biologically a binding pool):
- **1-attribute RESOLVES** (perfect: "cat go (big apple)" → "big apple", test passes).
- **2-attribute BOUNDARY** (K=5 load — the adjectives recover but the noun degrades at the bind-capacity edge
  ~0.93; liftable with higher D).
- The FHRR resonator's general multi-attribute *factoring* stays a numpy reference.

**Phase 4 — retire the bolted-on simulators.** `spiking_phasor_fhrr.py` + `resonate_fire_fhrr.py` carry a clear
header: NUMPY REFERENCE, not the production substrate; the production conversational agent runs on the core sim via
`brain_conversational_agent.py`. They are retained (not deleted) as the FHRR validation ceiling.

## Honest scope (what this is and isn't)

- **Is:** the conversational pipeline (comprehend / store / recall / who-what Q&A / abstention / negation / clauses
  / one-attribute / dialogue planning) running as genuine spiking dynamics on `SimulationBridge` neurons — three
  interacting bridges (parser, composer, dlPFC), the owner's "interacting brain regions" picture.
- **Isn't (yet):** (1) two-attribute is a boundary, the general FHRR resonator a numpy reference; (2) vocabulary is
  the validated probe scale (V=16 concept pools) per the owner's "probe-scale first" steer — production 320-concept
  scale on the brain agent is a follow-on; (3) the three bridges are orchestrated, not yet ONE bridge with all
  regions (a deeper unification); (4) "emergent" composition (arising from learning rather than the hand-wired
  coincidence/role circuits) remains the longer-term north star.

## Files

- `research/runners/core_sim_composition.py` (`CoreSimComposer`) + `tests/test_core_sim_composition.py` (5)
- `research/runners/brain_conversational_agent.py` (`BrainConversationalAgent`, `BridgeParser`) +
  `tests/test_brain_conversational_agent.py` (5)
- `research/runners/spiking_phasor_fhrr.py` + `resonate_fire_fhrr.py` — relabelled NUMPY REFERENCE.
- Plan: `docs/plans/2026-06-04-consolidate-conversational-pipeline-onto-core-sim-design.md`
- Audit: `research/findings/2026-06-04-conversational-pipeline-substrate-audit.md`
