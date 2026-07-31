---
type: plan
status: live
date: 2026-05-30
---

# Phase-factored integrated closed-loop: resolving the encode-order conflict via online theta-ordered episodic encode + offline shuffled-replay consolidation

**Date:** 2026-05-30
**Status:** Design (autonomous brainstorm, owner pre-approved the pivot). Supersedes the parked Q5 single-pass integrated loop at its pre-registered next step (the phase-factored architecture the 2026-05-19 iteration-4 finding prescribed but never wrote). Transition: writing-plans -> cheap-first falsification -> (conditional) full spiking build.

## 1. Problem and background

The Q5 integrated closed-loop (design 2026-05-18) hypothesized that compositional memory emerges from many validated brain subsystems composed into ONE loop under ONE shared theta-gamma rhythm. Its cheap-tier numpy probe confirmed the load-bearing core (removing the shared rhythm collapses both readouts together -- the Lisman-Idiart non-separability prediction). But the full spiking build STALLED at instrument soundness (research/findings/2026-05-19-integrated-loop-PROGRAM-LEVEL-encode-order-conflict-between-validated-concept-binding-and-episodic-store.md): a single online encode pass cannot simultaneously satisfy two contradictory requirements.

- **Concept selectivity** (which neurons mean "apple" vs "river") is built by the project's validated mechanism (embodied co-firing + topographic prior; the v16 16-pool recipe). It needs SHUFFLED / interleaved presentation across examples to break winner-take-most and form selective sub-populations.
- **Episodic order** (apple THEN river THEN dog) is held by the theta-gamma store. It needs presentation-order == gamma-slot-index, i.e. ORDERED presentation, so the sequence reads back from peak/phase order.

Shuffle vs order: contradictory in one pass. Iteration 4 confirmed this is structural, not a tuning gap -- applying the validated concept-binding mechanism recovered working-memory selectivity but destroyed the (previously perfect) episodic binding (1.0 -> 0.0).

This is the SAME conclusion the Direction cross-bridge arc reached from the geometry side (research/findings/2026-05-30-D-arc-SYNTHESIS-...md): concept orthogonality erodes when many concepts are built in one pass against a shared common-mode; pattern-separation + offline re-separation is the resolution. Two independent arcs, one prescription: **composition at scale needs a two-phase process -- fast online encoding plus slow offline consolidation that re-separates representations.**

## 2. The biological resolution (complementary learning systems)

Real brains do not build episodic order and concept selectivity in one pass. They factorize across time (McClelland-McNaughton-O'Reilly 1995; the project has validated each half in isolation -- Phase 1.3 CLS consolidation 3/3 strict anti-cheat; the hippocampal trisynaptic loop pattern-separation/completion):

- **ONLINE (awake, theta-ordered):** the hippocampus rapidly binds the CURRENT episode in presentation order -- a one-shot, order-preserving, sparse, pattern-separated INDEX (DG sparsification + CA3 autoassociation). It does NOT build durable concept selectivity; it points to whatever cortex currently represents.
- **OFFLINE (quiet/sleep, replay):** the hippocampus replays episodes to cortex in INTERLEAVED order (documented replay statistics sample across episodes; replay is not strictly ordered). This shuffled replay is exactly what cortex needs to build concept selectivity WITHOUT catastrophic interference (McClelland 1995's core argument), and it re-separates overlapping concepts (the D-arc's missing ingredient).

The factorization: **online hippocampus binds ORDER; offline replay builds concept SELECTIVITY in cortex.** Order lives in the hippocampal index (decoupled from concept-weight learning); selectivity is built offline by shuffled replay (decoupled from order). The contradiction dissolves because the two operations are now in different phases, each fed the presentation statistics it needs.

## 3. Architecture

Two explicit phases plus two readouts, all on the validated spiking substrate, joined by the shared theta-gamma rhythm.

**Phase 1 -- ONLINE episodic encode (theta-ordered).** Present a length-N concept sequence (N=2 minimal: apple, river) in order. The hippocampal index binds them order-preservingly: gamma sub-cycle k within the theta period holds item k (theta-gamma slot code). Reuse the validated engram-tagging API (start/commit/stimulate, D.14) for the one-shot order-preserving bind. Concept symbols are whatever cortex currently represents -- the index points, it does not build selectivity.

**Phase 2 -- OFFLINE consolidation (shuffled replay).** Replay the bound episode (plus interleaved items from other episodes) to cortex in SHUFFLED order via the validated SWR replay / Phase 1.3 consolidation machinery. Cortex builds concept selectivity via the validated co-firing/topographic mechanism. Crucially, the episodic ORDER is NOT touched -- it lives in the hippocampal index from Phase 1, not in cortex's concept weights -- so shuffled replay builds selectivity without destroying order.

**Readout 1 -- working-memory / concept query** ("is apple in the current buffer?"): tests concept selectivity (built offline in Phase 2). Decoded from cortical concept-pool activity via the validated readout.

**Readout 2 -- episodic order** ("what came after apple?"): tests order (held online in the Phase-1 hippocampal index). Decoded from the gamma-slot/peak order of the index replay.

**Shared theta-gamma rhythm.** One theta period spans the compositional buffer; gamma sub-cycles index the role-filler slots; the SAME rhythm drives Phase-1 online binding (slot = presentation position) and the Phase-2 replay read-out (slot order recovered from peak phase). The shared clock is the load-bearing integration piece -- the non-separability prediction (lesioning it collapses BOTH readouts) is the science signature.

**Net-new vs reused.** Reused BYTE-UNCHANGED: engram-tagging API (online bind), SWR/Phase-1.3 consolidation (offline replay), co-firing/topographic concept-binding (selectivity), abstention moat (output gate), the theta-gamma timing controller (from the parked loop's Task-2, which was built + adversarially reviewed). Net-new: the two-PHASE CONTROLLER that sequences online-encode-then-offline-consolidate (the parked loop did one pass; this is the only structural change), plus the order-preserving index readout (Readout 2 from the gamma-slot order).

## 4. Success criteria and verdict gate (frozen, pre-registered)

Inherit the parked loop's frozen gate discipline (its verdict module integrated_loop_core.py was built + adversarially reviewed; bars never moved). The phase-factored design's NEW contribution is passing the INSTRUMENT-SOUNDNESS gate the single-pass version blocked on; the SCIENCE gate is inherited unchanged.

- **Instrument soundness (the parked-loop blocker):** at minimal load N=2, BOTH readouts (working-memory query AND episodic order) >= 0.90. The single online pass had these mutually exclusive (one at 1.0 forced the other to 0.0). The phase-factored claim: with phases separated, BOTH clear 0.90 simultaneously. THIS is the decisive instrument fix.
- **Science gate (inherited, frozen):** the full loop composes (both readouts pass) AND each single-system lesion abolishes the capability it is responsible for (lesion online-bind -> order readout collapses; lesion offline-consolidate -> concept readout collapses; lesion shared rhythm -> BOTH collapse together = the non-separability signature). Scale ladder N=(2,4,8); three-state verdict (PASS / BOUNDARY / NEGATIVE); fixed bars never tuned by results; malformed -> VOID-not-crash; instrument-validity checked first.
- **Anti-cheat:** a broken/unsound run cannot score a PASS (instrument soundness gates the science gate); the lesion controls are identical to the full loop minus exactly one system, same random draws; no new autograd/training (only the reused validated learning rules); protected set byte-empty; abstention moat 7/7.

## 5. Cheap-first falsification (mandatory, before any spiking)

Per the project's proven discipline (the parked loop's cheap tier correctly de-risked its core; "falsify cheaply first" is a standing rule). A numpy/logic model of the two phases + two readouts that isolates the ONE structural question: does separating online-ordered-encode from offline-shuffled-consolidate resolve the encode-order conflict at N=2?

- Model Phase 1 as an order-preserving index over fixed pre-given concept symbols (slot k = item k).
- Model Phase 2 as shuffled replay building a concept-selectivity classifier from interleaved presentation.
- Test BOTH readouts. Pre-registered cheap bar (e.g. both >= 0.90 at N=2; the same instrument-soundness bar).
- **Cheap PASS** (both readouts clear where a modeled single-pass has them mutually exclusive) -> the factorization resolves the conflict in principle -> commit to the spiking build.
- **Cheap FAIL** -> the factorization is necessary but NOT sufficient; the conflict survives even with phases separated -> a deeper structural finding, surfaced honestly, before spending GPU. This is the high-value cheap save.

The cheap tier honestly CANNOT settle everything (per the parked loop's own honest note -- a 2-item compose is near-trivial; the cheap tier screens for fatal flaws + de-risks the core, the spiking build is the decisive test). A cheap PASS de-risks; it does not certify.

## 6. Staged plan

1. Cheap-first falsification probe (numpy; hours). Gate: resolves the conflict in principle?
2. IF cheap PASS: full spiking phase-factored loop on the validated bridge -- two-phase controller wiring + order-preserving index readout, reusing the four validated subsystems byte-unchanged. Grounding pin + frozen verdict (mostly inherited) + adversarial review of the load-bearing two-phase controller BEFORE the decisive run.
3. Controller-only decisive multi-seed run: instrument soundness N=2, then science gate + lesion study across N=(2,4,8). Mandatory smell-test (scrutinize a PASS harder than a FAIL). Honest propagation (findings + capability pillar + both remotes).
4. Continue per outcome: clean PASS -> the next catalog integration increment (multi-step sequential composition, then fluent-prior); honest NEGATIVE -> the next biology-identified factorization fix, iterate following the catalog, no hand-back.

## 7. Honest ceiling (never spun)

A clean PASS = the validated two-phase (online-encode / offline-consolidate) factorization carries MINIMAL compositional memory (both readouts at N=2-8) where a single online pass provably cannot, and lesioning either phase or the shared rhythm abolishes the capability it is responsible for (emergence-from-integration, the non-separability signature). Explicitly NOT fluent open-ended language, NOT an LLM, NOT scaled conversation -- those are separate later pre-registered increments. A FAIL is a real finding: the two-phase factorization is necessary but not sufficient, with its precise residual blocker localized. Prior validated results (pillars, the no-confab moat, the D-arc geometry insight) stand unaffected.

## 8. Reuse inventory (validated subsystems, byte-unchanged)

- Engram-tagging API (start/commit/stimulate, D.14) -- online order-preserving bind. Validated 87.5%/90% multi-seed.
- SWR replay / Phase-1.3 CLS consolidation -- offline shuffled replay. Validated 3/3 strict anti-cheat, no catastrophic forgetting.
- Co-firing + topographic concept-binding (v16) -- concept selectivity. Validated 88.75% multi-seed.
- Hippocampal trisynaptic loop (DG pattern-separation / CA3 completion) -- the re-separation the D-arc identified as the scale ingredient.
- Theta-gamma timing controller -- the shared rhythm (built + adversarially reviewed in the parked loop Task 2).
- Abstention / no-confab moat -- trustworthy output gate. 7/7 byte-identical.
- build_biological_brain_regions + the spiking bridge + checkpoint module -- substrate, byte-unchanged.

Net-new = the two-phase controller + the order-preserving index readout. No new learning mechanism; no autograd.

## Discipline

Bars frozen + pre-registered (inherited from the parked loop's reviewed verdict module; never tuned by results). Reuse-by-import only; protected/frozen/moat modules byte-unchanged. Cheap-first falsification before spiking. Adversarial review of the load-bearing two-phase controller before the decisive run. Honest propagation of every outcome to both remotes. Plain ASCII. No hand-back: a NEGATIVE triggers the next biology-identified factorization fix, not a stop.
