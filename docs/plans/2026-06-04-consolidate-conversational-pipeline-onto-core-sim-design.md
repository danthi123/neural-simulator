---
type: plan
status: live
date: 2026-06-04
---

# Consolidate the conversational pipeline onto the core sim (the brain) — design + phased plan

> **Status:** DRAFT for owner sign-off (owner asked: "plan doc first, sign-off before building").
> **For Claude:** after sign-off, execute phase-by-phase; honest negatives are deliverables; propagate every
> outcome to both remotes; never weaken a frozen bar or the no-confab moat; reuse-by-import; do not edit protected
> sim-core internals beyond what a phase explicitly authorizes.

## The principle (owner's vision, 2026-06-04)

The core sim **is** the simulated brain: a `SimulationBridge` of spiking neurons organized into interacting brain
regions. All capabilities should be **realized through it**, not by external/bolted-on modules. "Emergent" is the
north star — capabilities arising from region dynamics + learning rather than imposed algebra — but the immediate,
achievable goal of THIS arc is narrower and in the same spirit: move the existing conversational capabilities
**off the bolted-on numpy simulators and onto the brain substrate**, as a region-based spiking computation, so the
brain is self-contained. Emergent/learned composition then grows on that foundation (a later arc).

## What the substrate audit found (`2026-06-04-conversational-pipeline-substrate-audit.md`)

- The core `SimulationBridge` already has **validated spiking realizations of 11/13** conversational capabilities
  (bind/unbind, relational KB, who/what Q&A, abstention, negation, learned parser, dlPFC content-selection,
  grounding, generation) — but they live as archived `_insubstrate_*` probes in `research/findings/raw/`, used only
  by owner-facing demos.
- The "unified" agents (`nested_composition_agent`, `spiking_unified_agent`, `unified_agent_benchmark`, and the
  `unified_agent_*` runners shipped earlier 2026-06-04) **bypass** the brain and compute on two **bolted-on
  standalone numpy-spiking simulators**: `spiking_phasor_fhrr.py` + `resonate_fire_fhrr.py`.
- The **lone capability with NO core-sim realization** is the F=3 two-attribute resonator (adj⊗noun, both unknown
  concept codes — not a known-role unbind).

## Two VSA schemes (the substrate decision)

| | core-sim coincidence scheme (`_insubstrate_*`) | phasor FHRR (`spiking_phasor_fhrr`) |
|---|---|---|
| substrate | **the real `SimulationBridge`** (AND coincidence neurons, `cp_firing_states`) | bolted-on numpy time-stepped sim |
| binding | ±1 role ⊗ graded-ON/OFF filler (invertible) | unit-complex phase add |
| validated for | bind/unbind/KB/Q&A/abstain/negation/clauses (role-filler) | + attributes (F=3 resonator), deep nesting |
| gap | **attribute composition** (adj⊗noun) | runs off the brain |

**Decision: adopt the core-sim coincidence scheme** (it is already ON the brain and validated), consolidate the
role-filler capabilities first, and treat attribute composition as a scoped bridge-native sub-problem. Rationale:
lowest-risk and fastest real progress (uses what's proven on the brain); biologically grounded (coincidence
detection); avoids a giant upfront port of the entire phasor stack (including the iterative resonator) as one bet;
treats the one hard gap honestly. The phasor numpy simulators become labelled *reference* implementations, not the
production substrate. (If stage 2 finds attribute composition genuinely needs phase coding, a TARGETED bridge-native
phasor sub-module is reconsidered then — informed, not upfront.)

## Phased plan

### Phase 0 — Grounding pin + frozen bars (no behavior change)
- Read the four core-sim primitive probes (`_insubstrate_bind_unbind_probe`, `_insubstrate_relational_memory_probe`,
  `_insubstrate_negation_probe`, `_insubstrate_parser_stdp_probe`) + `content_selection_spiking` + `abstention_gate`;
  record each one's exact API, the SimulationBridge wiring it builds, its FROZEN bar, and its validated multi-seed
  result. Output: a short "primitives ledger" in the plan's findings.
- **Frozen bars carried forward unchanged:** spiking bind/unbind recovery ≥ 0.80 multi-seed with the wrong-query
  control at chance; relational Q&A ≥ 0.80; negation ≥ 0.80; abstention 100% (no-confab moat); parser conjunctions
  multi-seed. The consolidated agent must clear the SAME bars the probes cleared — no bar weakening.

### Phase 1 — Promote the validated core-sim primitives into a clean, tested module
- **Create `research/runners/core_sim_composition.py`** (or `sim/`-adjacent if the owner prefers core-package
  placement — flagged as an open question below). Refactor (NOT rewrite) the validated primitives out of
  `findings/raw/` into importable functions/classes: `build_bind_bridge`, `bind`, `unbind`, `bundle`,
  `cleanup`/`cleanup_with_abstention`, `store_fact`, `query_patient`, `query_agent`, `bind_polarity`/`ask_yes_no`.
  Keep the bridge-building + coincidence operating point byte-faithful to the validated probes.
- **Tests** (`tests/test_core_sim_composition.py`): each primitive reproduces its probe's frozen-bar result
  multi-seed (bind/unbind recovery, Q&A, negation, abstention). These tests ARE the regression guard for the rest
  of the arc.
- Anti-cheat: the new module must produce numerically the same recovery as the probes (port, not reimplement);
  add a regression test asserting parity with a probe run at a fixed seed.

### Phase 2 — Build ONE core-sim unified conversational agent on the brain
- **Create `research/runners/brain_conversational_agent.py`**: a `BrainConversationalAgent` that owns ONE
  `SimulationBridge` and realizes the conversational loop on it using the Phase-1 primitives:
  comprehend (parser → roles) → store SVO fact (bind+bundle on the bridge) → answer who/what (unbind+cleanup) →
  abstain on the unknown → negation/yes-no → **clauses** (recursive role-filler: bind a clause-bundle as a filler).
  Dialogue planning uses `content_selection_spiking` (the dlPFC working-memory Control, already on the bridge).
- **A frozen conversational test set on the BRAIN** (mirror the numpy `unified_agent_benchmark` categories MINUS
  attributes for now): flat SVO / clause-depth1 / who / abstain / negation — multi-seed, with the no-confab moat.
  Honest reporting: a per-category pass-rate on the real bridge vs the numpy ceiling, with any below-bar category
  surfaced (not hidden).
- This is the milestone: **the conversational pipeline runs on the brain**, self-contained, no phasor numpy sim in
  the path. Owner check-in here.

### Phase 3 — Attribute composition on the brain (the one gap) — research, honest outcome
- Cheap-first: can the coincidence scheme factor adj⊗noun? Options to probe (pre-registered, frozen bar = match the
  numpy resonator's clean-decode accuracy multi-seed): (a) enumeration over the adjective codebook (unbind each adj,
  cleanup the residual to a noun — this is just bridge bind/unbind+cleanup, likely tractable for ONE attribute);
  (b) a bridge-native two-factor relaxation (iterative unbind↔cleanup stepping the bridge); (c) a targeted
  bridge-native phasor sub-module ONLY if (a)/(b) fail.
- **Three-state outcome, honestly propagated:** RESOLVES (attribute composition on the brain, multi-seed) /
  BOUNDARY (one-attribute works, two-attribute degrades) / DOES-NOT-RESOLVE (documented gap — the brain agent ships
  without attribute composition, which is itself the finding; the numpy resonator stays a labelled reference, NOT in
  the production path).

### Phase 4 — Retire / relabel the bolted-on simulators
- Add a clear header to `spiking_phasor_fhrr.py` + `resonate_fire_fhrr.py`: "NUMPY REFERENCE implementation of FHRR
  composition — NOT the production substrate; the production conversational agent runs on the core `SimulationBridge`
  via `brain_conversational_agent.py`. Retained for validation/comparison only."
- Update the unified-agent runners + demos + `capability_status.json` + CLAUDE.md to point the production path at
  the brain agent; mark the numpy/phasor agents as reference. Do NOT delete (they're the validation ceiling), but
  remove them from any "this is the brain analogue" claim.
- `keep-webapp-current` + `sync-documentation` passes.

## Honest risks / open questions for the owner

1. **Capability scope.** Stage 1+2 gives the brain: SVO facts, KB, who/what, abstention, negation, clauses, and
   (pending Phase 3) attributes. Deep multi-attribute + the exact phasor nesting depth this session validated are
   NOT automatically carried over to the coincidence scheme — they'd be re-validated or scoped. Acceptable?
2. **Module home.** Should the consolidated core-sim composition live in `research/runners/` (application layer that
   uses the bridge) or be promoted into the `sim/` core package (most "self-contained")? The `_insubstrate` probes
   build bridges via `inject_explicit_wiring` (application-level wiring), so `research/runners/` is the natural home,
   but `sim/` is defensible if you want it in the engine package. (My lean: `research/runners/` — the engine is the
   neuron simulator; the composition wiring is an application of it.)
3. **Scale.** This consolidation is at the validated probe scale (V≈16–64 concept pools). Production-scale
   (320-concept G.20) on the brain agent is a follow-on, not this arc.

## Wall-clock honesty

Phases 1–2 are mostly refactor + assembly + tests (the primitives are validated) — bounded. Phase 3 is genuine
research with a real chance of an honest negative. Phase 4 is mechanical. Multi-seed bridge runs are GPU/CuPy
(minutes each); tests are CPU.

## Success criterion

The conversational agent runs end-to-end on the core `SimulationBridge` (the brain) — comprehend, store, recall,
compose role-filler structure incl. clauses, abstain, negate — clearing the same frozen bars the probes cleared,
with the bolted-on numpy simulators removed from the production path (retained only as labelled reference).
Attribute composition either joins it (Phase 3 RESOLVES) or is a documented, honestly-surfaced gap.
