# Shared theta-gamma rhythm (Separate Phases of Encoding And Retrieval) + generative replay: a conversational-stage design

**Status:** Design (autonomous; no hand-back). This is the pre-registered
next stage the Stage-1 honest negative triangulates onto. Stage 1
(regime-correct static two-path composition) decisively FAILed at full
biological scale (full accuracy 0.00 at every load and seed; the
no-confabulation moat held — zero confabulation under composition), which
empirically confirmed that static two-store composition is not how biology
produces this capability.

**Date:** 2026-05-19

**Plain-language commitment:** ordinary scientific terms, each defined once;
no internal codenames or letter-number labels are load-bearing; catalog
identifiers appear only in parentheses.

---

## 1. Why this stage, grounded in what the Stage-1 negative taught

The five-convergent terminal finding closed the necessity-instrument line:
no faithful architecture satisfies a *single simultaneous* readout because
that is the complementary-learning-systems division of labor. The Stage-1
decisive negative then showed empirically that *statically* reading the two
systems and combining them also does not yield the capability. Both results
point at the same resolution, which biology actually uses and which the
project has never built: **the conflict is dissolved by temporal
multiplexing under one shared rhythm, not by spatial separation into two
readouts.** The full biological grounding (with citations) is in
`docs/plans/2026-05-19-regime-correct-compositional-retrieval-design.md`
section 2b and references [9]-[17]; this document does not re-litigate it.
The load-bearing mechanisms:

- **Separate Phases of Encoding And Retrieval (Hasselmo SPEAR).** One
  shared ~125 ms theta rhythm time-multiplexes a write phase
  (entorhinal-afferent drive, high acetylcholine, plasticity on, retrieval
  suppressed) and a read / pattern-complete phase (CA3-recurrent drive,
  low acetylcholine, plasticity off); the same framework gates the slower
  acetylcholine-regulated encoding↔consolidation transition. Write and
  read never compete for one readout because they are time-shared.
- **Order-bearing vs order-invariant are operating modes of one
  theta-gamma code** (set by the inhibitory regime), not two stores.
- **Conversation = a generative hippocampal-prefrontal replay loop:**
  prefrontal cortex holds the ordered compositional / sequence frame;
  replay proposes-and-pattern-completes configurations against the
  consolidated schema.

## 2. The reframe (explicit)

Unit of analysis: a **single shared theta-gamma rhythm** that, every cycle,
*time-multiplexes* an encode phase and a retrieve/pattern-complete phase
across the already-validated subsystems, with acetylcholine (the validated
neuromodulator subsystem) as the phase gate, a prefrontal working-memory
region holding the compositional sequence frame, and a generative replay
loop producing novel schema-constrained ordered output. The hypothesis
under test: this rhythm-multiplexed composition produces grounded
compositional/sequential capability that the static two-store composition
(Stage-1 negative) does not — at biological scale, holding or improving as
load scales, abstaining rather than confabulating. There is no necessity
partition anywhere (that line is closed); the single frozen artifact is a
new capability-verdict module with fixed bars.

## 3. Inventory of validated subsystems to reuse byte-unchanged (exact, code-grounded)

Reuse-by-import only; no edits to any protected/frozen/validated module or
the no-confabulation moat. Interfaces confirmed at these file:line:

- **Acetylcholine phase gate = the validated neuromodulator subsystem.**
  `sim/neuromodulators.py`: `NeuromodulatorConfig` / `ModulatorTarget`
  (target_type `plasticity_window_gate`, scope `gate:<name>`; high
  concentration blocks plasticity, low permits — exactly the SPEAR
  encode/retrieve plasticity gate) / `ProductionRule` (rule_type
  `manual`); `NeuromodulatorManager.set_concentration(name, value)`,
  `.initialize(...)`, `.step(bridge)` (decay+production per timestep),
  `.compute_plasticity_rate_multiplier()`. The controller drives ACh
  high on the encode phase and low on the retrieve phase via
  `set_concentration`, with a `plasticity_window_gate` target wired to
  the plastic pathways.
- **Hippocampal theta-gamma episodic store + trisynaptic
  pattern-completion.** `research/runners/text_minimal_isolation.py:173`
  `build_biological_brain_regions(enable_hippocampus_consolidation=True,
  ca3_recurrent_density, ca3_recurrent_weight, ...)` builds
  ec/dg/dg_pv_basket/ca3/ca1. Validation entries
  `research/runners/validate_trisynaptic_loop.py`
  `test_pattern_separation` (:170, D.12) / `test_pattern_completion`
  (:235, D.13).
- **Replay-consolidation + awake/sleep phase gates.**
  `research/runners/consolidation_trainer.py`
  `run_concept_replay_phase` (:43), `run_swr_replay_phase` (:154),
  `run_consolidation_training` (:206);
  `research/runners/text_minimal_isolation.py`
  `set_awake_gates` (:1664) / `set_sleep_gates` (:1704) /
  `freeze_all_gates` (:1744). These are the slower
  encode↔consolidate transition the same SPEAR framework governs.
- **Prefrontal working-memory compositional frame.**
  `build_biological_brain_regions(enable_dlpfc_verb=True,
  n_dlpfc_verb=200, dlpfc_verb_internal_density=0.15)`
  (text_minimal_isolation.py:199); NMDA bistable maintenance via global
  `CoreSimConfig.enable_nmda=True` (validated PREFRONTAL_CORTEX_WM
  profile, sim/profiles.py:206).
- **Concept substrate + engram API + no-confabulation moat + stepping.**
  the validated 16-pool concept recipe (the same construction the Stage-1
  re-review cleared), the engram-tagging API on the bridge, the
  no-confabulation moat `research/runners/abstention_gate.py`
  `gate(ranked, 650.0)` (:12; `DEFAULT_THRESHOLD=650.0` :7, byte-
  unchanged, 7/7), and `SimulationBridge.step_simulation(num_steps)`
  (sim/bridge.py:3052) / `_run_one_simulation_step` (:4776).

## 4. What is genuinely net-new (bounded precisely)

A single small **shared theta-gamma rhythm/phase controller** plus the
wiring that composes the reused subsystems under it. It is a *timing
controller, not a new learning mechanism* — every learning rule is a reused
validated rule; no automatic differentiation anywhere. Concretely the
net-new code:

- a theta-phase clock (period ~125 ms in simulation time) with a
  nested gamma sub-cycle index;
- per-phase gating that, on the encode phase, sets acetylcholine high
  (via the reused neuromodulator `set_concentration` + a
  `plasticity_window_gate` target) and routes afferent drive +
  plasticity-on; on the retrieve phase, sets acetylcholine low and
  routes recurrent/pattern-completion + plasticity-off — driving the
  reused subsystems by calling the reused `step_simulation` in the
  controller's own loop (no edit to the step itself);
- a prefrontal-frame hold/advance of the compositional sequence slot
  across gamma sub-cycles (reused dlpfc region + reused NMDA
  bistability);
- a generative replay loop that, using the reused replay-consolidation
  phase functions + the prefrontal frame, emits an ordered sequence and
  gates each emitted item through the reused no-confabulation moat.

## 5. Three concrete architectures, honest ceilings, falsify-cheaply-first

- **A — Shared-rhythm SPEAR retrieval (RECOMMENDED first; cheaply
  de-riskable).** Add only the theta-phase clock + acetylcholine
  encode/retrieve phase gate over the existing validated
  hippocampal+concept substrate; test that phase-multiplexed
  encode-then-retrieve recovers a bound item the Stage-1 *static*
  composition could not (same task family, so the Stage-1 negative is
  the built-in control). Smallest net-new (clock + ACh gate wiring). A
  fast pure-NumPy precursor de-risks the phase-multiplex logic before any
  GPU spend; its toy numbers are not a result.
- **B — A + prefrontal compositional frame.** Add the reused dlpfc
  NMDA-bistable frame holding an ordered multi-item slot across gamma
  sub-cycles; test ordered multi-item composition. Staged only if A
  passes.
- **C — B + generative replay loop.** Add the generative
  replay-consolidation loop emitting novel schema-constrained ordered
  sequences gated by the moat; the conversational stage proper. Staged
  only if B passes.

**Recommendation:** build A first under a pre-registered fixed-bar gate
whose decisive control is the Stage-1 static-composition negative (A must
beat it on the same task family or A is itself an honest negative). B and C
are pre-registered staged follow-ons, each its own fixed-bar test,
autonomous, iterate-following-biology.

## 6. Pre-registered gate, falsify-cheaply-first, anti-cheat

- **New frozen capability-verdict module** (its own file; standard library
  + typing only; does NOT import or modify any existing verdict module or
  the moat). Fixed constants set now and never tuned: full-capability
  accuracy bar, the Stage-1-static-composition control ceiling the
  rhythm-multiplexed system must exceed, ablation-collapse bar (remove the
  shared rhythm → collapse), abstention-correctness bar, fixed load
  ladder, min seeds ≥ 3, scale tolerance. Three states plus VOID strictly
  distinct from FAIL; instrument-validity first; malformed → cannot
  conclude, never a crash; recompute from raw numbers (a caller verdict is
  ignored).
- **The decisive built-in control:** the ablation "remove the shared
  rhythm" must reduce the system to the Stage-1 static composition and
  collapse — i.e., the capability must be *attributable to the rhythm
  multiplexing*, not to any single reused subsystem. This makes a
  rhythm-artifact false-PASS structurally detectable.
- **Falsify-cheaply-first** NumPy precursor of the phase-multiplex logic
  before any decisive GPU run (toy numbers explicitly not a result).
- **Anti-cheat (non-negotiable):** mandatory smell-test scrutinising a
  nominal PASS harder than a FAIL; a dedicated adversarial review of the
  net-new controller + the verdict module before the no-harm phase
  (primary mandate: is the capability genuinely emergent from the shared
  rhythm vs a wiring/timing artifact; is the rhythm-removed ablation
  faithful; can a degenerate run score PASS; are the bars movable; any
  automatic differentiation; are subsystems genuinely byte-reused);
  controller trust-but-verify diffs with the full protected set
  byte-empty; the no-confabulation moat + its 7/7 test byte-identical;
  GPU/CuPy for every decisive run (NumPy only for the smoke); honest
  propagation of every outcome both remotes; no configuration-cranking
  past the pre-registered terminus; an honest negative is a real finding,
  propagated without spin, followed by the next biology-identified
  refinement — autonomous, no hand-back.

## 7. Honest ceiling (stated up front, never spun)

A clean scrutinised success = a biology-grounded shared-theta-gamma-rhythm
composition shows grounded compositional/sequential capability that the
static two-store composition provably does not (the Stage-1 negative is the
built-in control), holding or improving as load scales, and abstaining
rather than confabulating. This is explicitly **NOT** fluent open-ended
language, **NOT** an LLM, and **NOT** the retracted transitive-inference
claim, unless a later pre-registered stage genuinely shows it. All prior
validated results and honest boundaries are unaffected; the original frozen
verdict, the corrected module, the Stage-1 capability-verdict module, and
the no-confabulation gate remain byte-unchanged. The genuine durable
contribution of this stage, regardless of outcome, is a faithful,
adversarially-hardened, fixed-bar test of whether the biological
conflict-resolution mechanism (shared-rhythm temporal multiplexing) yields
the capability the static composition could not.

## 8. Components / data flow / error handling / testing (for the plan)

- **Components:** reused substrate+hippocampus+dlpfc builder; reused
  neuromodulator subsystem (acetylcholine phase gate); net-new
  theta-gamma rhythm/phase controller; reused replay-consolidation phase
  functions + awake/sleep gates; reused engram API; reused
  no-confabulation moat at output; new frozen capability-verdict module;
  kill-safe multi-seed runner mirroring the proven scaffold.
- **Data flow:** each theta cycle → encode phase (ACh high, afferent +
  plasticity on) then retrieve/pattern-complete phase (ACh low,
  recurrent + plasticity off); gamma sub-cycles index sequence slots in
  the dlpfc frame; the generative replay loop emits ordered items each
  gated by the moat; the verdict module scores full vs rhythm-removed
  ablation across the load ladder and seeds.
- **Error handling:** instrument-validity first; any malformed/instrument
  failure → VOID (not FAIL, not a fabricated PASS); kill-safe/resumable
  via the reused checkpoint module.
- **Testing:** ≥ 12-case adversarial matrix on the verdict module
  (full beats the Stage-1 control; rhythm-removed ablation collapses;
  abstention-correct; non-decreasing across load; threshold-tamper →
  cannot conclude; malformed → cannot conclude not crash; fixed-threshold
  pins); no-harm proving the full protected set byte-unchanged and the
  moat still 7/7.

---

**Next:** writing-plans for this design (Task 0 pin; Task 1 the new frozen
capability-verdict module fully specified + adversarial matrix; Task 2 the
net-new shared-rhythm controller + runner reusing the validated subsystems
byte-unchanged; Task 3 dedicated adversarial review; Task 4 no-harm;
Task 5 controller-only decisive run + smell-test + honest propagation),
then subagent-driven-development under the pre-registered fixed-bar gate,
honest propagation of every outcome to both remotes, iterating following
the biology — autonomous, no hand-back.
