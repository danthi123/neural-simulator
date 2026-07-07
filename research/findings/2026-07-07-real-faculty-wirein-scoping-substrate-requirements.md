# Real-faculty wire-in — scoping: the workspace SPEAKS its reasoned answer via the REAL faculties, cheap + NO training + substrate-independent; and the exact REQUIREMENTS it places on the future deep substrate

**Date:** 2026-07-07
**Type:** read-only scoping (the thin "stable-architecture" thread, run in parallel with the deep-lever research gate). No `sim/` edit, no build, no training.
**Why now (owner strategy):** the project is pivoting its main effort to a new learning substrate (biological deep credit assignment). This scoping separates the SUBSTRATE-INDEPENDENT architecture (never wasted on a substrate swap) from the substrate-specific trained weights — so the cheap wire-in can be built now, and the deep-substrate work has concrete requirements to hit.

## The milestone
Replace the GNW workspace's two ABSTRACT downstream consumers (`report`, `reason`) with the project's REAL faculties: the validated **A→W spiking read-out** (`UnifiedNeuralSpell`, EMERGE-67/68) as `report` (the brain literally SPEAKS the ignited concept), and the **EMERGE inheritance reasoner** (`PerDimensionConsole`, EMERGE-52/54) as `reason`. ⇒ "the concepts the brain can SAY are the ones it REASONS with" made literally true — and it BREAKS Rung-3's "by-construction identity" critique for free (report and reason are now completely different substrates/populations, sharing ONLY the ignited concept's string identity).

## The key architectural finding: a STRING-KEYED dispatch (not a code hand-off)
The three faculties use three different concept representations (workspace = a neuron-slice assembly; A→W = a driven concept pool decoded from `language_output` spikes; reasoner = an HTM-pooler codon), and **none shares a code vector — they share only the concept STRING.** So the wire-in is a thin string-keyed orchestration:
```
ignite(concept) on the GNW workspace  → read which report-pop sustains → concept STRING  ("none" ⇒ gate CLOSED = moat)
reason  = reasoner.ask_can(concept, property)     # affirm / override / abstain (the moat)
   ↓ gate-first: abstain ⇒ report is NEVER invoked
report  = UnifiedNeuralSpell.spell(subject, verb) # drive pools → decode "the owl can fly" from language_output spikes
```
**No new training:** the A→W engines are GPU-trained ONCE + cached; the reasoner self-trains from the teach-script in seconds of CPU; the workspace settles at build. The whole minimal wire-in is an ORCHESTRATION file, reuse-by-import, NO `sim/` edit — it is essentially EMERGE-70's gate-first `_emerge_turn` with the GNW ignition as the front-end.

## One-backend co-execution — already solved
`sim.bridge` binds ONE backend per process; the reasoner is numpy-native, the A→W read-out is cupy-native. EMERGE-70 found the residual was 3 host→device write lines; EMERGE-71 fixed it production-clean with the single additive `SimulationBridge.xp` property (`sim/bridge.py:213`) + `sim.backend.from_host`. The GNW workspace builds a real `SimulationBridge`, so it **inherits `bridge.xp` cleanly**; its own writes are already backend-safe (scalar fills / active-`xp` index arrays). The three bridges co-execute in ONE cupy process (EMERGE-70's proven "3 bridges, one process" pattern — full single-bridge masked co-residence is a later consolidation, not required).

## Substrate-independence (the strategically load-bearing split)
**Pure architecture / interfaces — NEVER wasted on a learning-substrate swap:** the ignite→identity→dispatch pipeline; the gate-first moat control-flow invariant (reasoner decides answer-vs-abstain FIRST; on abstain the speech faculty is never invoked); the two faculty INTERFACES (`report(concept)→surface`, `reason(concept,prop)→{affirm|override|abstain}`); the `bridge.xp`/`from_host` co-execution contract; the concept-string as the universal key. **Substrate-specific (behind the interfaces, re-derived on a swap):** the A→W concept-pool weights, the reasoner's learned codons, the GNW assembly recurrence + Rung-3c learned category. A deep-credit-assignment substrate that learns BETTER report/reason weights slots in **without touching the wire-in** — it just implements the same two interface functions with better internals.

## REQUIREMENTS the wire-in places on any future deep substrate (this is the de-risk for the B work)
To slot into the "one brain you talk to", the future deep substrate MUST:
1. **Expose a concept→word read-out** (`report`): given an ignited concept, produce its spoken surface.
2. **Expose a concept→property inference with a MOAT** (`reason`): affirm/override/abstain, abstaining on unknown — the "I don't know" is a hard requirement.
3. **Hold a SUSTAINED ignited assembly** long enough for a downstream read (the reads happen in a late post-drive window — the substrate must sustain, not just transiently represent).
4. **Enforce mutual exclusion** (one concept ignited at a time) so the identity read-out is unambiguous.
5. **Support the one-process backend abstraction** (`bridge.xp`/`from_host`) — any substrate keeping the `cp_*`-array convention inherits it free.
6. **Key concepts by a recoverable identity** the orchestration can map. A substrate producing ONLY a distributed code with no recoverable identity would force a code-vector hand-off (a more expensive redesign) — a design constraint to keep in mind for the new substrate.

## Cheapest-first first step (NO expensive training)
`_gnw_rung4_real_faculty_wirein_derisk.py` — one GPU turn: build the GNW workspace with a HELD-OUT member (owl→BIRD, never taught fly); build the real reasoner (teach the taxonomy in-script) + real A→W (`UnifiedNeuralSpell(load=True)` from cache) in ONE cupy process; ignite `owl` → recover the string → `reason=ask_can("owl","fly")`→"Yes" (inherited) → gate-first → `report`: drive owl+fly pools → decode "the owl can fly" from `language_output` spikes; MOAT: ignite a never-taught concept → reasoner abstains → assert A→W is NEVER invoked (0 spell calls). GO bar: the ignited-concept string round-trips (workspace→reason→report), the answer is INHERITED (reasoner, not looked up), the surface is spike-decoded (lesion `pool→language_output` → decode collapses), and the moat holds. Honest scope: use the vocab INTERSECTION of the A→W-trained words and the reasoner taxonomy (owl/penguin/... × fly/walk); full vocab = a cached-A→W rebind (a `--train` re-run, a data lever, not a mechanism). The correctness gate to ASSERT: the string that ignites is the string the reasoner reasons about AND the A→W speaks (else the report==reasoning identity isn't actually demonstrated).

## Files (scoped, not edited)
A→W: `concept_speak_demo.py`, `_emerge6{7,8,9}_*`, `_emerge70_*` (`_emerge_turn` = the wire-in template). Reasoner: `_emerge5{2,4}_*`, `_emerge30_*`, `_emerge29_*`. GNW: `_gnw_rung{1,3,3b,3c}_*`. Substrate: `sim/bridge.py:213` (`SimulationBridge.xp`), `sim/backend.py` (`from_host`).
