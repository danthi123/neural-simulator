---
type: plan
status: live
date: 2026-05-19
---

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

## 2c. Existing biology-grounded sims that already implement related capability (owner-flagged gap; internalised, durable)

The 2b biological grounding cites primary papers; this section cites
existing biology-grounded *sim implementations* that have already
realised related capability. These reshape the staged plan: future
design passes after this stage MUST start from "what existing sims
already do" before designing from biology alone.

- **Spaun / Nengo / Semantic Pointer Architecture (Eliasmith et al.)
  [10][11][12]** — the canonical large-spiking cognitive model
  (2.5 M neurons; 8 cognitive tasks in one network including serial
  working memory, question answering, symbolic pattern completion,
  multi-item compositional binding into symbol-tree structures).
  Achieves compositional binding via SPA: high-dimensional vectors
  encode symbol-tree structures distributed across spiking-neuron
  groups via the Neural Engineering Framework; basal-ganglia
  cognitive control routes information between cortical components.
  Open source (Nengo). SPA's vector-symbolic binding operator is
  exactly the mechanism the project's previously-retracted
  concept-concept compose arc, the integrated-loop necessity
  terminus, and the Stage-1 static-composition negative all failed
  to produce. The project has never tried SPA.
- **Lisman-Idiart theta-gamma multi-item working memory --
  implemented and characterised in spiking models [13][14][15].**
  Ursino 2022 [13] is a worked-out spiking implementation: theta-
  gamma cross-frequency coupling, multi-item maintenance up to
  nine items desynchronised on gamma, ordered-sequence replay via
  gamma-nested-in-theta, AND an "imagination phase" where the
  isolated network under uniform noise spontaneously recovers
  learned sequences and links them by similarity (exactly the
  generative-replay loop the SPEAR Stage-C is supposed to build).
  Soroka 2024/2021 [14][15] extend with alpha-vs-theta interference
  and a modular traveling-wave variant. The SPEAR rhythm controller
  should parameter-regime-ground in these (GABAergic strength,
  theta/gamma frequencies, Hebbian/anti-Hebbian training) rather
  than reinvent.
- **Spiking Tolman-Eichenbaum Machine [16][17] -- hippocampal-
  entorhinal compositional generalisation in spikes.** Whittington
  2020 [16] (568 citations, Cell) formally links the
  hippocampal-entorhinal circuit to structural generalisation;
  medial entorhinal cells form a basis for structural knowledge,
  hippocampal cells link it to sensory representations; learns
  grid/place/border/object-vector cells; predicts and confirms
  structural-knowledge transfer across remapping. Spiking TEM
  (Kawahara 2025, bioRxiv) [17] extends this to spike-based
  computation with theta phase locking, phase precession, and
  predictive grid cells. Manns 2006 [18] is direct experimental
  confirmation in CA1 of the Hasselmo encode/retrieve theta-phase
  separation. Spiking TEM is the most directly relevant prior art
  for the hippocampal-entorhinal compositional path; future stages
  should learn from / borrow its implementation.

## 2d. Broader prior-art search (owner-flagged: "search, don't just recall")

A broader literature + web + code-repo search (not just memory)
returned an active 2023-2026 spike-driven language-model space the
project's "no local generative scale-confident" terminus did not
test against, plus the live open-source libraries and curated
paper-lists the discipline should pull from:

- **Spike-driven generative language models (open source, active
  development).** SpikeGPT [19] (Zhu 2023, 130 cites; 45M-216M
  parameter RWKV-inspired binary spiking LM trained on 5B tokens of
  OpenWebText, ~22x less energy, 32.2x fewer ops on neuromorphic
  hardware, competitive with non-spiking on benchmarks, Hugging Face
  hosted, Discord community building chatbots, github
  ridgerchu/SpikeGPT). SpikeLM [20] (Xing 2024, 34 cites; first
  fully spike-driven model for BOTH discriminative AND generative
  language tasks via elastic bi-spiking, github Xingrun-Xing/SpikeLM).
  SpikingBERT [21] (Bal 2023, 78 cites; BERT distilled into a
  spiking LM with implicit differentiation, multiple GLUE tasks,
  github NeuroCompLab-psu/SpikingBERT). Spikingformer (AAAI 2026
  foundation model for SNNs; github TheBrainLab/Spikingformer).
  NeuronSpark (2026 arxiv: state-space SNN language model with
  generative behaviour). These trade some biological strictness for
  surrogate-gradient BPTT to get generative capability -- the
  project already uses surrogate-grad BPTT in Phase 2.1/2.2 but
  never scaled it to LLM-class architectures.
- **Phase-coded VSA in spikes -- the natural unification with
  SPEAR.** Orchard 2023/2024 [22][23] implement Fourier Holographic
  Reduced Representation (FHRR, a VSA variant) in spiking neurons
  by encoding each complex-vector phase as a SPIKE TIME WITHIN A
  CYCLE. Bind, unbind, spatial reasoning, function representation,
  memory delay -- all on spike-phase. The strategic implication: a
  theta-gamma SPEAR rhythm and SPA-style vector-symbolic binding are
  NOT competing architectures -- they are the SAME phase code with
  two purposes (write/read multiplexing AND compositional binding).
  Recent VSA advances continue (Furlong 2024 PMC11655797 on neural
  probabilistic computation via VSA; arxiv 2511.01838 Nov 2025 on
  efficient VSA from histogram recovery).
- **Open-source libraries + curated lists to pull from
  periodically.** NengoSPA (v2.0.1.dev0; actively maintained;
  documented; recent models with 2048-d spatial semantic pointers
  in spiking ReLU nets). TheBrainLab/Awesome-Spiking-Neural-Networks
  is a curated paper+code list to scan at every design pass.
- **Large-scale biology-plausible platforms.** BiCoSS [24] (Yang
  2021, 164 cites; >4M biology-plausible spiking neurons for
  cognitive activities). BrainScaleS-2 [25] (Pehle 2022, 221 cites;
  neuromorphic hardware with hybrid plasticity). 2025 SNN review
  [26] (Huo 2025).

**Strategic pre-registered consequence (new, durable, propagated):**

- The project's "no local generative scale-confident" terminus is
  *NOT* a property of biology-grounded spiking sims in general; it
  is a property of the project's specific anti-cheat constraints
  (no autograd; only validated local learning rules). The
  spike-driven LLM family achieves generative language by trading
  some biological strictness for surrogate-gradient BPTT. There is
  a legitimate, previously-undiscussed strategic question: should
  the discipline allow a clearly-marked, non-load-bearing
  **comparison baseline** that scales surrogate-grad BPTT to a
  SpikeGPT-class architecture purely to answer "are our anti-cheat
  constraints the actual ceiling, or is the substrate the ceiling?"
  This is not pivoting away from biological discipline -- it is
  testing whether the discipline is what is blocking us. Surfaced
  for owner consideration; not adopted unilaterally.
- Phase-coded VSA in spikes (Orchard 2023/2024) directly unifies
  our two recent strategic threads (SPEAR theta-gamma + SPA
  binding). Post-Stage-A design passes should evaluate this
  unification rather than treat them as separate architectures.
- The discipline for every future design pass: broad search
  first -- consensus + web + open-source code + curated lists
  (Awesome-Spiking-Neural-Networks, NengoSPA docs) -- not
  narrow-from-memory.

**Strategic consequence (pre-registered, durable).** The in-flight
SPEAR Stage decisive run remains a valid experiment (testing whether
the project's specific reuse of the validated subsystems under a
SPEAR rhythm works at biological scale). Regardless of its outcome:
- Stage-A (this stage) parameter regimes for the rhythm controller
  should be cross-checked against Ursino 2022 before any future
  refinement iteration.
- Stage-B / Stage-C (already pre-registered as follow-ons) must
  adopt mechanisms from the above sims rather than reinvent --
  specifically: a Semantic-Pointer-Architecture-style vector-
  symbolic binding layer (Spaun open-source prior art) is the
  strongest candidate for the compositional binding the project has
  repeatedly failed to produce; the hippocampal-entorhinal pathway
  should learn from Spiking TEM; the generative replay loop should
  learn from Ursino 2022 imagination-phase.
- Honest meta-finding (owner-flagged): the project's discipline
  for future stages must be "check existing biology-grounded sim
  implementations FIRST, adopt proven mechanisms, then iterate"
  rather than "design from biology papers alone".

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

## 7b. References (for §2c existing-sim prior art)

[10] [Spaun: A Perception-Cognition-Action Model Using Spiking Neurons](https://consensus.app/papers/details/7f744298b6a15dfb820d6d3e5030aa9e/?utm_source=claude_code) (Stewart, Choo, Eliasmith, 2012, Cognitive Science)
[11] [Large-Scale Synthesis of Functional Spiking Neural Circuits](https://consensus.app/papers/details/2a4b40fe07365d8fbe7cedc18bfdfcd1/?utm_source=claude_code) (Stewart et al., 2014, Proceedings of the IEEE)
[12] [Large-scale cognitive model design using the Nengo neural simulator](https://consensus.app/papers/details/055032017c895ed19c396ca262e8104b/?utm_source=claude_code) (Sharma et al., 2016)
[13] [A model of working memory for encoding multiple items and ordered sequences exploiting the theta-gamma code](https://consensus.app/papers/details/e17f7fc053485161afe1538750fd0c28/?utm_source=claude_code) (Ursino et al., 2022, Cognitive Neurodynamics)
[14] [Mechanistic role of alpha oscillations in a computational model of working memory](https://consensus.app/papers/details/f3030fdfec135115ab01ded5aa34c45c/?utm_source=claude_code) (Soroka et al., 2024, PLOS ONE)
[15] [Theta, alpha and gamma traveling waves in a multi-item working memory model](https://consensus.app/papers/details/747e59c429fa5cd19e41b0d052078464/?utm_source=claude_code) (Soroka et al., 2021)
[16] [The Tolman-Eichenbaum Machine: Unifying Space and Relational Memory through Generalization in the Hippocampal Formation](https://consensus.app/papers/details/12de2e1b677d533796b8d7cfcc3f03dc/?utm_source=claude_code) (Whittington et al., 2019, Cell)
[17] [The Spiking Tolman-Eichenbaum Machine: Emergent Spatial and Temporal Coding through Spiking Network Dynamics](https://consensus.app/papers/details/c7095e94ac295cff8446de5ad50678cf/?utm_source=claude_code) (Kawahara et al., 2025, bioRxiv)
[18] [Hippocampal CA1 spiking during encoding and retrieval: relation to theta phase](https://consensus.app/papers/details/15ce17ce14a8549fb8e39e951c4fe20c/?utm_source=claude_code) (Manns et al., 2006, Neurobiology of Learning and Memory)
[19] [SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks](https://consensus.app/papers/details/d2b9c1c34dac56b4b12c0147f5a3c75b/?utm_source=claude_code) (Zhu et al., 2023, TMLR; [GitHub](https://github.com/ridgerchu/SpikeGPT))
[20] [SpikeLM: Towards General Spike-Driven Language Modeling via Elastic Bi-Spiking Mechanisms](https://consensus.app/papers/details/dc5fe60a771d5d839c935dd72b0655bb/?utm_source=claude_code) (Xing et al., 2024; [GitHub](https://github.com/Xingrun-Xing/SpikeLM))
[21] [SpikingBERT: Distilling BERT to Train Spiking Language Models Using Implicit Differentiation](https://consensus.app/papers/details/d94a2b4a097854a9b6b3a2ce4e23d8d9/?utm_source=claude_code) (Bal et al., 2023; [GitHub](https://github.com/NeuroCompLab-psu/SpikingBERT))
[22] [Hyperdimensional Computing with Spiking-Phasor Neurons](https://consensus.app/papers/details/be1d5a9091145d4d94774194b95895cf/?utm_source=claude_code) (Orchard et al., 2023, ICONS)
[23] [Efficient Hyperdimensional Computing With Spiking Phasors](https://consensus.app/papers/details/d689c878eb7d55888a796a9db9fdc6af/?utm_source=claude_code) (Orchard et al., 2024, Neural Computation)
[24] [BiCoSS: Toward Large-Scale Cognition Brain With Multigranular Neuromorphic Architecture](https://consensus.app/papers/details/50112a70a15c53bdb423b07b4c137fb9/?utm_source=claude_code) (Yang et al., 2021, IEEE TNNLS)
[25] [The BrainScaleS-2 Accelerated Neuromorphic System With Hybrid Plasticity](https://consensus.app/papers/details/cec2b4b9129f561f8f3401c42f302b81/?utm_source=claude_code) (Pehle et al., 2022, Frontiers Neurosci)
[26] [Research on SNN Learning Algorithms and Networks Based on Biological Plausibility](https://consensus.app/papers/details/d646136c2c205e998422a53ab50da9e4/?utm_source=claude_code) (Huo et al., 2025, IEEE Access)
[27] [Vector Symbolic Architectures as a Computing Framework for Emerging Hardware](https://consensus.app/papers/details/bdb7c7653fae58c4bfa9274d16e9b034/?utm_source=claude_code) (Kleyko et al., 2021, Proc IEEE)
[28] [NengoSPA documentation (open source library)](https://www.nengo.ai/nengo-spa/user-guide/spa-intro.html)
[29] [Spikingformer GitHub (AAAI 2026 foundation model)](https://github.com/TheBrainLab/Spikingformer)
[30] [Awesome-Spiking-Neural-Networks (curated paper+code list)](https://github.com/TheBrainLab/Awesome-Spiking-Neural-Networks)

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
