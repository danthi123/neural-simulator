# Pirazzini-reference three-layer theta-gamma conversational architecture (adapted to the project's validated subsystems)

**Status:** Design (autonomous; no hand-back). This is the pre-registered
next stage after the SPEAR full arc closed with an honest negative + the
convergent-ceiling biology-translatable insight (Stage-1 static and SPEAR
rhythm-gating composition both fail to lift composed readout above the
calibrated no-confabulation threshold at biological scale). The broader-
search-first investigation surfaced a *directly-implementable existing
reference architecture* — Pirazzini 2024 *Frontiers in Neural Circuits* —
that already realises the integrated theta-gamma + ACh-phase-gated +
PFC/CA3/CA1 sequential-memory + imagination/dreaming capability the
project wants, with 99 % recovery on early sequence positions.

**Date:** 2026-05-19

**Plain-language commitment:** ordinary scientific terms, each defined
once; no internal codenames or letter-number labels are load-bearing;
catalog identifiers appear only in parentheses.

**Top-level goal context (owner-stated, durable):** the project's goal is
artificial life with a proper brain analogue; insights from the sim
should translate back to real-biology insights; capabilities like
conversation / composition are instrumental. This stage is selected
because it (a) directly addresses the convergent-ceiling insight produced
by the prior arc, AND (b) maps onto a published biology-grounded
implementation, AND (c) yields biology-translatable lessons regardless of
outcome.

---

## 1. Why this stage (what the prior arc taught + what the broader-search produced)

The five-convergent terminal finding closed the necessity-instrument line
(biologically unsatisfiable in every memory regime — CLS division of
labor). The Stage-1 static composition decisively FAILed at biological
scale (full accuracy 0.00). The SPEAR rhythm-multiplexed composition
ALSO decisively FAILed at biological scale (full accuracy 0.00), with
the rhythm controller mechanistically active (14.15 mV bridge-state
divergence proven) but not lifting composed readout above the calibrated
no-confabulation threshold (650). The trustworthy abstention property
held in BOTH architectures (`abstain_correct = 1.0` everywhere). The
convergent ceiling localised the gap precisely: rhythm-gating alone (in
either static or rhythm-multiplexed form) does not yield a composed
readout that exceeds the trustworthy threshold for compositional queries.

The broader-search-first investigation (consensus + WebSearch + direct
WebFetch of full text) identified **Pirazzini 2024
*Frontiers in Neural Circuits* "Modeling the contribution of theta-gamma
coupling to sequential memory, imagination, and dreaming"** as the
directly-implementable existing reference architecture. Its mechanism is
*disinhibition-based* (theta rhythm rhythmically disinhibits an
ordinarily-inhibited CA3 layer via glutamatergic excitatory synapses
from an external theta-generator unit), NOT synaptic-gain-modulation
(which is what our SPEAR runner ended up with after the
faithfulness fix). And it uses the standard Hasselmo ACh polarity:
**HIGH ACh during encoding** suppresses CA3→CA1 transmission +
strengthens cortical input + facilitates LTP; **LOW ACh during retrieval
/ imagination** promotes pattern completion and replay. These polarity
and mechanism choices are different from the SPEAR runner, which is
why this is a substantively different next stage rather than a
config-crank.

## 2. The Pirazzini-2024 reference architecture (the concrete model, in
plain language)

A three-layer biology-grounded model with an external theta generator:

- **Working-Memory Layer (PFC)**: maintains an external cue (an
  auto-excitatory pyramidal loop) until reset by new stimuli.
  Pirazzini's reference uses 75 units.
- **L1 (CA3)**: auto-associative memory implementing within-episode
  feature recovery via plastic recurrent connections. Ordinarily
  inhibited; rhythmically disinhibited by the theta generator via
  glutamatergic excitatory synapses (the theta-rhythm mechanism).
  Pirazzini uses 75 units.
- **L2 (CA1)**: hetero-associative network linking consecutive episodes
  in a sequence. Pirazzini uses 75 units.
- **External theta generator**: oscillates at ~4 Hz (theta period
  ~250 ms; range 2.67-6 Hz); gamma at ~40 Hz (~25 ms period; range
  35-45 Hz). One theta cycle accommodates ~5 episodes (theta / gamma
  with adjustment).

Synaptic rules during encoding (high ACh):

- **Wp (Hebbian excitatory)**: synchronise features within an episode.
- **Wf (Hebbian inhibitory targeting fast GABAergic interneurons)**:
  contribute to within-episode synchronisation.
- **Af (anti-Hebbian desynchronising)**: rapidly inhibit features
  across DIFFERENT episodes, preventing superimposition.

Training: episode pairs presented sequentially to L1 and L2
SIMULTANEOUSLY for 250 ms per pair, **ONE-SHOT** (only once as input).

ACh phase mechanism (the standard Hasselmo polarity, biology-correct):

- HIGH ACh (encoding): suppress CA3→CA1, strengthen cortical input,
  facilitate LTP.
- LOW ACh (retrieval / imagination): promote pattern completion and
  memory replay.
- INTERMEDIATE ACh (dreaming): 2/3 reduction in all synapses' strength.

Decode (within-theta-cycle): an episode is wholly recovered iff all
features of that episode are simultaneously above 70 % max-activity
threshold for at least one integration step within a theta cycle.

Demonstrated scales: 4-6 features per episode tested (50 reachable
per unpublished results); ~5 episodes per theta cycle; 3 sequences
trained; >99 % recovery on episodes 2-3, 87-90 % on episode 5.

Imagination: isolated WM + high random noise → casual recall of stored
sequences in random order.

Dreaming: 2/3 synapse reduction + high noise → novel sequence
recombinations exploiting shared features (creative combination).

## 3. Inventory of validated subsystems to reuse byte-unchanged (exact)

Reuse-by-import only; no edits to any protected/frozen/validated module
or the no-confabulation moat.

- **PFC working-memory frame** (Pirazzini's WM layer):
  `build_biological_brain_regions(enable_dlpfc_verb=True,
  n_dlpfc_verb=200, dlpfc_verb_internal_density=0.15, ...)` in
  research/runners/text_minimal_isolation.py:199; global NMDA
  bistability via `CoreSimConfig.enable_nmda=True`. (We have 200 units
  vs Pirazzini's 75 — more capacity, biologically faithful pyramidal
  range.)
- **L1 (CA3 auto-associative)**: `enable_hippocampus_consolidation=True`
  in the same builder creates `ec / dg / dg_pv_basket / ca3 / ca1`
  regions; `ca3_recurrent_density=0.30, ca3_recurrent_weight=1.5`
  (text_minimal_isolation.py:199). Engram-tagging API on
  SimulationBridge for one-shot bindings (sim/bridge.py:2485-2659).
  Trisynaptic pattern-completion validated (D.12/D.13;
  research/runners/validate_trisynaptic_loop.py).
- **L2 (CA1 hetero-associative)**: same builder; the `ca1` region +
  `ca3->ca1` pathway are already wired. The hetero-associative
  cross-episode link is the validated Phase-1.3 consolidation /
  replay pathway (research/runners/consolidation_trainer.py
  `run_concept_replay_phase` :43; `run_swr_replay_phase` :154;
  awake/sleep gates `set_awake_gates`/`set_sleep_gates`
  text_minimal_isolation.py:1664/1704).
- **Acetylcholine multi-target modulator** (NEW NeuromodulatorConfig
  composed of REUSED `sim/neuromodulators.py` primitives):
  `NeuromodulatorConfig` + `ModulatorTarget` + `ProductionRule` +
  `NeuromodulatorManager.set_concentration / step` are reused
  byte-unchanged. The new config combines targets so HIGH ACh
  simultaneously: (a) suppresses CA3→CA1 transmission via a
  pathway-scoped `synaptic_gain` target (scope=`gate:ca3_to_ca1`,
  negative sensitivity); (b) strengthens cortical input via
  pathway-scoped `synaptic_gain` (scope=`gate:lang_input_to_ca3` or
  `gate:lang_input_to_dlpfc_verb`, positive sensitivity);
  (c) facilitates LTP via a `plasticity_rate` (scope=`all`) target
  with positive sensitivity (encoding-favourable rate boost). This is
  net-new wiring of REUSED primitives, not new mechanism.
- **External theta-generator unit**: net-new (the project lacks a
  rhythm primitive; confirmed by the rhythm/oscillation grep done
  earlier in the SPEAR design pass). Implemented as a tiny
  controller class that, each simulation step, computes the current
  theta phase and gamma sub-phase, sets ACh concentration
  appropriately, and drives a rhythmic disinhibitory current onto
  CA3 inhibitory interneurons (`dg_pv_basket` or a CA3-targeted
  interneuron group; the precise wiring uses existing
  `bridge.cp_external_input_current` on the inhibitory population at
  trough-of-theta phase to release CA3 pyramidals).
- **No-confabulation moat at output**: `research/runners/abstention_gate.py`
  byte-unchanged; threshold 650; tests 7/7.
- **Decode**: within-theta-cycle, count features (concept-pool units)
  simultaneously above the Pirazzini-style threshold. We adapt this
  by computing the raw `lang_output` firing-rate confidence (the
  validated quantity the moat is calibrated on, per the SPEAR
  faithfulness-fix lesson) across a full theta cycle (~250 ms), then
  passing the top-confidence answer through the moat. Avoids both
  the SPEAR ceiling failure mode (moat fed an uncalibrated quantity)
  and the Pirazzini-only retrieval criterion (which assumes the
  network internally tags features but does not gate against
  confabulation).

## 4. What is genuinely net-new (bounded precisely)

- **External theta-generator controller** (~50 lines): a small
  per-step class that computes theta phase (4 Hz; 250 ms period at
  the bridge dt) and gamma sub-cycle index, sets ACh via reused
  `NeuromodulatorManager.set_concentration`, and writes a
  disinhibitory current onto the validated CA3-targeting inhibitory
  population at the theta-trough phase (the disinhibition mechanism
  Pirazzini uses).
- **Multi-target ACh modulator config** (a NeuromodulatorConfig
  combining four REUSED target types: pathway-scoped synaptic_gain
  with positive and negative sensitivities; global plasticity_rate;
  global excitability_drive optionally) — net-new configuration of
  reused primitives, not a new modulator type.
- **One-shot encoding routine** (~30 lines): present each
  episode-pair (Ep_i to L2 / Ep_i+1 to L1 simultaneously) for 250 ms
  ONCE; the validated engram-tagging API records the bound
  ensemble; the multi-target ACh sits at the HIGH-encoding level
  throughout.
- **Within-theta-cycle decode** (~30 lines): over one theta cycle,
  read the raw `lang_output` firing-rate confidence from the
  pattern-completion phase of CA3 via CA1; pass through the reused
  moat. The validated `compose_concept_engram.lang_output_pattern_during_*`
  + `cosine_to_word` + `abstention_gate.gate(ranked, 650.0)` are
  reused byte-unchanged.
- **New frozen capability-verdict module** (stdlib + typing only;
  fixed bars; does NOT import any existing verdict module or the
  moat). Bars set now and NEVER tuned. Same three-state +
  VOID-distinct-from-FAIL discipline as Stage-1 and SPEAR.
- **Decisive built-in control**: the convergent Stage-1+SPEAR
  ceiling. The Pirazzini-reference runner's `full_acc` must
  significantly exceed the convergent ceiling (≈ 0.00 in both
  prior architectures), AND a `theta_disabled` ablation must
  collapse to that ceiling. This makes any rhythm-artifact
  false-PASS structurally detectable.

NO automatic differentiation anywhere; every learning rule is a
reused validated rule (Hebbian / anti-Hebbian / STDP). GPU/CuPy for
the decisive path; NumPy only for `--tiny-synth`.

## 5. Two pre-registered architectures, falsify-cheaply-first

- **A — Pirazzini-reference single-sequence retrieval (RECOMMENDED
  first, falsify-cheaply-first):** the three-layer architecture, one
  theta generator, encode a single short sequence (e.g. 3 episodes
  × 4 features) one-shot, attempt retrieval-from-cue; capability =
  retrieved-episode features clear the moat threshold;
  `theta_disabled` collapses; abstention holds. Smallest net-new
  scope. NumPy precursor de-risks the controller logic.
- **B — Pirazzini-reference + imagination/dreaming modes (staged
  follow-on):** add isolated-WM + high-noise mode for mind-wandering;
  intermediate-ACh + noise for "dreaming" creative recombination.
  Biology-translatable goal richer; staged only if A passes.

(The Orchard 2023/2024 spiking-phasor FHRR + a phase-coded vector-
symbolic-architecture extension is a possible Stage-C if A and B
land but compositional capacity needs the explicit VSA substrate;
remains pre-registered for that case but not built first.)

## 6. Pre-registered gate, anti-cheat, honest ceiling

- **New frozen capability-verdict module** with its own constants
  (proposed; finalised in writing-plans). Fixed bars set now and
  NEVER tuned: full-retrieval accuracy bar (e.g. ≥ 0.80 for
  early-episode features above 70% max-activity, per Pirazzini's
  >99% on episodes 2-3); theta-disabled ablation must collapse
  ≤ 0.40; abstention-correct ≥ 0.90; min seeds 3; fixed load
  ladder over episodes-per-sequence (e.g. 2, 3, 5).
- **Falsify-cheaply-first**: NumPy precursor of the theta-controller
  + ACh-multi-target wiring + within-theta-cycle decode before any
  decisive GPU run; toy numbers explicitly not a result.
- **Anti-cheat (non-negotiable)**: mandatory smell-test scrutinising
  PASS HARDER than FAIL; dedicated adversarial reviewer BEFORE
  no-harm whose mandate includes specifically (a) is the Pirazzini-
  faithful disinhibition mechanism genuinely wired (NOT
  synaptic_gain-modulation like the SPEAR runner's choice — VERIFY
  the disinhibitory current actually releases CA3 pyramidals at
  theta-trough by measuring bridge-state divergence between
  theta-on and theta-off, mirroring the SPEAR re-review's 50-step
  probe); (b) is the multi-target ACh modulator's three effects
  (suppress CA3→CA1, strengthen cortical input, facilitate LTP)
  ALL active (consumed by the right paths); (c) is the ACh polarity
  Hasselmo-correct (HIGH ACh = encoding-permissive at the right
  paths); (d) decisive built-in control (theta-disabled reduces to
  convergent ceiling) genuinely active; (e) can a degenerate run
  score PASS — re-run Stage-1/SPEAR exploit classes adapted; (f)
  the moat is fed its calibrated raw firing-rate quantity (the
  Stage-1 fix lesson). STRENGTHEN-only fixes; frozen bars
  byte-unchanged. Controller trust-but-verify diffs with the full
  protected set byte-empty.

**Honest ceiling (stated up front, never spun):** a clean scrutinised
success = the biology-grounded Pirazzini-reference three-layer
architecture, adapted to our validated substrate with the correct
Hasselmo ACh polarity and disinhibition-based theta, shows grounded
sequence-retrieval above the trustworthy moat threshold that the
convergent Stage-1 + SPEAR ceiling could not — at biological scale,
multi-seed. This is biology-faithful brain-analogue retrieval, not
fluent open-ended language, not an LLM. All prior validated results +
honest boundaries unaffected. The trustworthy abstention moat stays
the gate at output. Under the project's reframed goal (artificial
life with proper brain analogue, biology-translatable insights),
EITHER outcome is the deliverable: a PASS confirms the
Pirazzini-reference is the missing mechanism (and yields concrete
biological recommendations about which subsystems must compose to
produce above-threshold composed recall); a FAIL identifies the next
ceiling and the next biology-faithful refinement (with the
Orchard spiking-phasor FHRR phase-coded-VSA extension as the
already-pre-registered Stage-C candidate).

## 7. References

(Stage-1 references [1]-[8] in the prior design; SPEAR references
[9]-[30] in the prior design; the additional references this
stage introduces:)

[31] [Modeling the contribution of theta-gamma coupling to sequential memory, imagination, and dreaming](https://consensus.app/papers/details/6ec3ed94cd505ae7b2edadc0be788ccc/?utm_source=claude_code) (Pirazzini et al., 2024, Frontiers in Neural Circuits) — the reference architecture this stage adapts.
[32] [Control of working memory by phase-amplitude coupling of human hippocampal neurons](https://consensus.app/papers/details/fa3aa22030e857139a2be002cc23b5f4/?utm_source=claude_code) (Daume et al., 2024, Nature) — human single-neuron experimental confirmation of theta-gamma PAC coordinating frontal control + hippocampal persistent activity during multi-item working memory; biology ground truth the architecture should map to.
[33] [Implementing Holographic Reduced Representations for Spiking Neural Networks](https://consensus.app/papers/details/81a7d0e68d995d07b52ad6d7d07be1ca/?utm_source=claude_code) (Sumanasena et al., 2025, IEEE Access) — recent (2025) VSA-in-spikes implementation on Intel Lava; alternate implementation track for the staged Orchard-VSA Stage-C if Pirazzini-reference Stage-A/B is insufficient.

---

**Next:** writing-plans for this design — Task 0 grounding pin;
Task 1 the new frozen fixed-bar three-state capability-verdict
module fully specified for exact transcription with an
adversarial-matrix test; Task 2 the net-new theta-generator
controller + multi-target ACh modulator + one-shot encoding +
within-theta-cycle decode runner, reusing the validated
subsystems byte-unchanged (no autograd); Task 3 dedicated
adversarial review with the Pirazzini-specific concerns above
flagged explicitly; Task 4 no-harm; Task 5 CONTROLLER-ONLY
decisive run + smell-test + honest propagation, then autonomous
continuation per outcome.
