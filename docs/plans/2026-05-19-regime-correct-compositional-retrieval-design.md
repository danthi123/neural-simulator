---
type: plan
status: live
date: 2026-05-19
---

# Regime-correct compositional retrieval: a constructive design grounded in the brain's complementary-learning-systems division of labor

**Status:** Design (autonomous; no hand-back). Supersedes the integrated-loop
necessity-instrument line, which is scientifically terminal (five convergent
faithful routes; the fifth validly GPU-measured). This is a *constructive
capability* design, not a necessity test.

**Date:** 2026-05-19

**Plain-language commitment:** This document uses ordinary scientific terms,
each defined once. No internal codenames or letter-number labels are
load-bearing. Catalog identifiers appear only in parentheses for traceability.

---

## 1. Why this direction, and what the five terminal findings taught

The necessity-instrument line asked: *is there a single experimental setup in
which removing the memory-consolidation system is strictly necessary for
recalling a time-ordered sequence?* Five independent biology-faithful routes —
the last validly measured on the GPU with a corrected, sound instrument —
established that the answer is *no, in every memory regime*, because that
requirement contradicts how the brain's two memory systems divide labor:
recent memory (hippocampus) keeps serial order but does not need
consolidation; remote memory (consolidated neocortex) needs consolidation but
is order-invariant by construction. A pre-committed bound forbids any further
edits to that test's structure. The line is honestly exhausted.

The findings do not just close a line — they *prescribe the next one*: stop
forcing one shared readout to satisfy contradictory regime requirements;
instead **read each memory system in the regime it is biologically built for,
and compose the two**. That is exactly what the biology of compositional
retrieval says the brain does.

## 2. Genuine biological investigation (done first, with citations)

The brain achieves compositional retrieval as **retrieval-augmented
generation across two systems, each read in its own regime**:

- The hippocampal and neocortical systems interact during recall: the
  hippocampus retrieves relevant *recent, specific* episodic content into
  working memory, and the neocortical network supplies *general knowledge*;
  composition is literally modelled as retrieval-augmented generation across
  the two [1].
- Hippocampal replay trains the neocortical generative/semantic network;
  unique specific elements and predictable conceptual elements are stored and
  reconstructed by *efficiently combining both systems*, reserving limited
  hippocampal storage for new/unusual information [2].
- Detail-rich, gist, and schematic representations **co-exist**; which one is
  used is set by *availability and task demands* (posterior hippocampus =
  specific detail, anterior hippocampus = gist, ventromedial prefrontal cortex
  = schema) [3].
- A pre-existing associative **schema causally accelerates** assimilation of
  new related facts, which can become hippocampus-independent rapidly
  [4][5][6]; schema acts as a scaffold for neocortical integration [5].
- Crucially for trustworthiness: the brain runs **separate, parallel
  metacognitive monitors** for recent vs remote memory — doubly dissociable —
  that gate retrieval confidence *independently per system* without changing
  recognition itself [7]; hippocampal ripples carry a "semantization"
  dimension as memories age [8].

The unifying biological picture: **two retrieval paths read in their own
regimes, composed, with a per-regime confidence monitor deciding when to
answer and when to abstain.** This maps almost one-to-one onto subsystems
this project has *already validated*.

## 2b. How biology RESOLVES the recent/remote conflict (the load-bearing core of the conversational path)

The static "read two stores, combine, abstain" picture above is a
*substrate*, not the resolution. The recent/remote distinction is exactly
what made the necessity-instrument line fail five times: every faithful
architecture refused a *single simultaneous* readout. The literature is
unambiguous that biology never does a single simultaneous readout — it
resolves the conflict by three mechanisms the conversational path MUST be
built around (not retrieval-augmented ranking):

- **Temporal multiplexing under one shared rhythm (Separate Phases of
  Encoding And Retrieval).** The hippocampus interleaves *encoding* and
  *retrieval* into opposite phases of the same ~125 ms theta cycle: one
  phase has strong entorhinal afferent drive + high acetylcholine +
  plasticity on + retrieval suppressed (write); the opposite phase has
  strong CA3 recurrent drive + low acetylcholine + plasticity off
  (read / pattern-complete); the same framework governs the slower
  acetylcholine-gated encoding↔consolidation transition [9][10][11]. Write
  and read never compete for one readout because they are time-shared on
  one rhythm. This is precisely the structure the five-convergent
  necessity finding kept re-deriving as load-bearing, and exactly the
  shared theta-gamma rhythm the project's own catalog (Lisman-Idiart,
  N.16) flagged as never built.
- **Order-bearing vs order-invariant are operating modes of one
  theta-gamma code, not two stores.** Items are gamma cell-assemblies;
  their order is the theta phase at which they fire (phase-amplitude
  coupling) [12][13]. One trained theta-gamma network can, purely by
  changing inhibitory (GABAergic) strength, operate as pattern-completion
  from a partial cue, OR hold multiple items WITHOUT order, OR replay an
  ORDERED sequence [14]. Recent-ordered and remote-order-invariant are the
  same substrate read under different rhythm/inhibition regimes — the
  conflict is dissolved, not relocated.
- **Conversation = a generative hippocampal-prefrontal replay loop.**
  Replay sequences are compositional hypotheses that evolve from
  predictable→uncertain and converge on a configuration [15]; prefrontal
  neurons encode the specific order and structure of planned words before
  utterance in a temporally-ordered dynamic [16]; combinatory meaning is
  shared between comprehension and production [17]. Prefrontal cortex
  holds the ordered compositional frame; replay proposes-and-pattern-
  completes configurations against the consolidated schema.

**Consequence for the staged plan.** Stage 1 (regime-correct compositional
retrieval + trustworthy abstention, the rest of this document) remains a
valid, necessary substrate — the two systems must be readable in their
correct regimes before they can be phase-multiplexed. But the path to
conversational capability is NOT more retrieval-augmented ranking; it is a
later, pre-registered stage whose load-bearing core is: a single shared
theta-gamma rhythm time-multiplexing an encode phase and a
retrieve/pattern-complete phase (the project already has a validated
theta-gamma episodic store, a validated trisynaptic pattern-completion
pathway, a validated replay-consolidation subsystem, and a validated
neuromodulator subsystem for the acetylcholine gate), with a prefrontal
working-memory frame holding compositional/sequence structure and a
generative replay loop producing novel schema-constrained ordered
sequences. Each such stage is its own pre-registered fixed-bar test,
pursued autonomously following the biology — honest ceiling unchanged
(grounded, trustworthy; NOT fluent open-ended language / an LLM until a
pre-registered stage genuinely shows it).

## 3. The reframe (explicit)

- Old (now terminal) unit of analysis: a single shared-readout loop in which
  one lesion must be *necessary* for one combined readout. Biologically
  ill-posed; closed.
- New unit of analysis: a **constructive two-path composition** —
  (a) a recent-specific hippocampal retrieval path, (b) an order-invariant
  consolidated semantic path, read separately in their correct regimes and
  combined by retrieval-augmented composition, (c) a per-regime confidence /
  abstention monitor at output. The capability under test is *grounded
  compositional retrieval that holds or improves as load scales and that
  abstains rather than confabulates under ablation*.

There is **no frozen necessity partition** anywhere in this design (the
necessity line is closed), so there is nothing that could be goalpost-moved.
The single frozen artifact is a capability-verdict module with fixed accuracy
and abstention bars, mirroring the project's existing frozen-verdict
discipline exactly.

## 4. Inventory of already-validated subsystems to reuse byte-unchanged

Reuse-by-import only; no edits to any protected/frozen/validated module or the
no-confabulation moat. Exact interfaces (file:line) confirmed in code:

- **Concept substrate (remote-semantic carrier).** 16-pool concept binding
  (≈89% multi-seed, bidirectional) via
  `build_biological_brain_regions(...)` (research/runners/text_minimal_isolation.py:173):
  kwargs `enable_noun_pools / enable_verb_pools / enable_adjective_pools`,
  orthogonal codes, FS pools. Bridge built via `run_minimal_isolation(...)`
  (text_minimal_isolation.py:2423): `CoreSimConfig` →
  `enable_brain_region_framework=True` → `SimulationBridge`.
- **Hippocampal recent-specific path.** `enable_hippocampus_consolidation=True`
  creates regions `ec, dg, dg_pv_basket, ca3, ca1` (validated trisynaptic
  pattern separation/completion, D.12/D.13). Engram tagging
  (Tonegawa, D.14; ≈88% stim-recall) on `SimulationBridge`:
  `start_engram_recording` (sim/bridge.py:2485),
  `commit_engram_tag(name, threshold_hz=5.0, top_k=None, region_filter=None)`
  (:2514), `stimulate_tag(name, drive_pA, additive=False)` (:2599),
  `clear_tag_drive` (:2629), `list_engram_tags` (:2643),
  `get_engram_tag_indices` (:2651), `delete_engram_tag` (:2659).
- **Consolidation (remote-semantic builder + regime switch).**
  `run_concept_replay_phase(bridge, tag_names, n_replays_per_tag=20, ...)`
  (research/runners/consolidation_trainer.py:43);
  `run_swr_replay_phase(bridge, n_swr_events=200, ...)` (:154);
  `run_consolidation_training(...)` (:206). Regime switch / hippo-OFF
  evaluation: `HIPPO_REGIONS = ["ec","dg","dg_pv_basket","ca3","ca1"]`
  (research/runners/consolidation_eval.py:31),
  `evaluate_with_hippo_off(bridge, n_trials_per_word=25, silence_current_pA=-200.0, ...)`
  (:34) and `evaluate_consolidation_proof(...)` (:120). The validated strict
  anti-cheat protocol uses a stronger silence (−2000 pA) and edge-zeroing;
  this design reuses that protocol byte-unchanged for the remote-regime read.
- **Direct-associate retrieval.** Multi-tag cue retrieval (90% FULL / 100%
  PARTIAL multi-seed): `handle_multitag(cue_word)`
  (research/runners/compose_concept_chat.py:210) — stimulate every engram tag
  containing the cue, aggregate `lang_output` cosines, rank associates.
- **Trustworthy output (the project's distinctive contribution).**
  No-confabulation abstention moat: `DEFAULT_THRESHOLD = 650.0`,
  `abstain(top_confidence, threshold=650.0)`,
  `gate(ranked, threshold=650.0)` (research/runners/abstention_gate.py:7-12);
  `tests/test_abstention_gate.py` = 7 tests. Byte-identical + 7/7 throughout.
- **Discipline reference only (NOT imported).** Frozen verdict modules
  `integrated_loop_core.py` / `_core_v2.py` show the fixed-constant + verdict
  function pattern (`_IL_*` bars; `integrated_loop_verdict(rungs)`); the new
  capability-verdict module mirrors this *structure* with its own new
  constants and does not import or change either.

## 5. The capability, the task, and the falsifiable success signature

**Capability under test:** answer a grounded compositional query by reading a
*recent-specific* fact from the hippocampal path and *general/semantic*
structure from the consolidated neocortical path — each in its correct regime
— and abstain rather than confabulate when a regime cannot ground its part.

**Task (fixed, pre-registered).** On the validated concept substrate:
1. Encode a small set of recent-specific relational facts one-shot via the
   hippocampal engram pathway (the validated stim-recall mechanism).
2. Build the order-invariant consolidated semantic schema over a base
   vocabulary via the validated replay-consolidation phase.
3. Pose compositional queries whose correct answer requires *combining* a
   recent-specific fact (hippocampal regime) with general structure (remote
   consolidated regime), read separately then composed (retrieval-augmented,
   per [1][2]) — never demanded of one shared readout.

**Pre-registered fixed-bar success signature (three-state):**
- **Full system** (both paths, each in its own regime, composed): compositional
  answer accuracy ≥ a fixed bar, across a fixed load ladder, ≥ 3 seeds,
  non-decreasing across load within a fixed tolerance.
- **Recent-only ablation** (consolidated schema removed; hippocampus on): the
  *generalization* component collapses while the specific recent fact remains
  retrievable, and the monitor *abstains* on the ungrounded generalization
  (does not confabulate).
- **Remote-only ablation** (hippocampus strict-silenced via the validated
  hippo-OFF protocol): the *recent-specific* component collapses (the
  consolidated store is order/specific-invariant — exactly the five-convergent
  finding), and the monitor *abstains* rather than confabulates.
- **Trustworthiness invariant:** the no-confabulation moat is byte-unchanged
  and never lowered; under every ablation the system abstains rather than
  confabulate. This is the project's distinctive, validated property, here
  shown to *survive composition*.

Ablations here are **diagnostic of the composition** (does each path
contribute its regime-appropriate part; does abstention hold), not a necessity
verdict. There is no single readout whose collapse must be attributed to one
lesion, so the refuted necessity question is not re-posed.

**Honest ceiling (stated up front, never spun):** a clean success = a
biology-grounded two-system composition answers grounded compositional queries
by reading recent-specific and remote-semantic content each in its correct
regime, holding/improving with load, and abstaining rather than confabulating
under ablation. Explicitly **NOT** fluent open-ended language, **NOT** an LLM,
and **NOT** the previously *retracted* transitive-inference claim (that result
was a bug artifact; this design does not resurrect it — it tests the
regime-correct composition + abstention property, which is new and distinct).
All prior validated results and honest boundaries are unaffected.

## 6. Three concrete architectures, with honest ceilings and de-risk cost

- **A — Minimal regime-correct composition (RECOMMENDED, falsify-cheaply
  first).** Recent-specific facts via engram tags; remote-semantic via the
  validated consolidated readout; composition = hippocampal tag retrieval
  feeds the consolidated readout (retrieval-augmented, [1]); abstention moat at
  output. Smallest net-new wiring (a composition+routing controller only;
  every learning rule and subsystem reused unchanged). Cheap NumPy precursor
  de-risks the composition+abstention logic before any GPU spend.
- **B — A + schema-accelerated assimilation.** Add the project's existing
  schema-anchor reinforcement (grounded in [4][5][6]) so a new recent fact
  *congruent* with the consolidated schema is assimilated in one shot and
  remains answerable after hippocampus-OFF. Richer science (tests the causal
  scaffold prediction); more net-new wiring; staged only if A passes.
- **C — B + per-regime metamemory monitors.** Two separate confidence
  monitors, one per path, mirroring the doubly-dissociable recent/remote
  metamemory streams [7]; the system reports which regime it answered from.
  Most faithful, most net-new wiring, hardest to de-risk cheaply; staged only
  if B passes.

**Recommendation:** build A first under a pre-registered fixed-bar gate;
B and C are pre-registered staged follow-ons, each its own fixed-bar test,
pursued autonomously following the biology — the iterate-following-biology
discipline, no hand-back.

## 7. Pre-registered gate, falsify-cheaply-first, and anti-cheat

- **New frozen capability-verdict module** (its own file; standard library +
  typing only; does NOT import or change any existing verdict module or the
  moat). Fixed constants set now and never tuned: full-system accuracy bar,
  ablation-collapse bar, abstention-correctness bar, fixed load ladder, min
  seeds = 3, scale tolerance. Three states plus VOID strictly distinct from
  FAIL; instrument-validity checked first; malformed input → "cannot
  conclude", never a crash; "cannot conclude" never reported as success.
- **Falsify-cheaply-first.** A fast pure-NumPy simulation of the
  composition+abstention logic at minimal load runs before any decisive GPU
  run; its toy numbers are explicitly not reported as a result; it only
  screens for fatal logic flaws.
- **Anti-cheat (non-negotiable).** Mandatory smell-test scrutinising a nominal
  PASS harder than a FAIL; a dedicated adversarial review of the load-bearing
  composition runner + the verdict module *before* the no-harm phase;
  controller trust-but-verify git diffs with the full protected set byte-empty;
  the no-confabulation moat + its 7/7 test byte-identical; **no automatic
  differentiation anywhere** (every learning rule is a reused validated rule);
  GPU/CuPy for every decisive run (NumPy only for the smoke); honest
  propagation of **every** outcome (findings doc + capability pillar +
  schema-green + push both remotes); no configuration-cranking past the
  pre-registered terminus; an honest negative is a real finding, propagated
  without spin, followed by the next biology-identified refinement — autonomous,
  no hand-back.

## 8. Components, data flow, error handling, testing (for the plan)

- **Components:** (i) substrate+hippocampus builder (reused); (ii)
  recent-specific encoder (reused engram API); (iii) consolidated-schema
  builder (reused replay-consolidation); (iv) net-new composition/routing
  controller (the only genuinely new code in A); (v) regime-correct readout
  (reused multitag/consolidated readout + reused hippo-OFF protocol for the
  remote read); (vi) abstention moat at output (reused, byte-unchanged);
  (vii) new frozen capability-verdict module; (viii) kill-safe multi-seed
  runner mirroring the proven runner scaffold.
- **Data flow:** encode recent facts → consolidate base schema → for each
  query: hippocampal retrieval (recent regime) ⊕ consolidated readout (remote
  regime) → composition controller → abstention moat → answer or "I don't
  know" → verdict module scores full + both ablations across the load ladder.
- **Error handling:** instrument-validity-first; any malformed/instrument
  failure → VOID (not FAIL, not a fabricated PASS); kill-safe/resumable via the
  reused checkpoint module.
- **Testing:** ≥ 12-case adversarial matrix on the verdict module
  (full-passes; each ablation collapses its regime part; abstention-correct
  under ablation; non-decreasing across load; threshold-tamper → cannot
  conclude; malformed → cannot conclude not crash; fixed-threshold pins);
  no-harm phase proving the full protected set byte-unchanged and the moat
  still 7/7.

## 9. References

[1] [Hippocampo-neocortical interaction as compressive retrieval-augmented generation](https://consensus.app/papers/details/3d0141cd146156eb9856bda8586cbeff/?utm_source=claude_code) (Spens et al., 2026, bioRxiv)
[2] [A generative model of memory construction and consolidation](https://consensus.app/papers/details/0f8c7aa167aa54da88190dff4ec3f157/?utm_source=claude_code) (Spens et al., 2023, Nature Human Behaviour)
[3] [Details, gist and schema: hippocampal-neocortical interactions underlying recent and remote episodic and spatial memory](https://consensus.app/papers/details/9a04d8e242925fa39de9e4e676bece1e/?utm_source=claude_code) (Robin & Moscovitch, 2017, Current Opinion in Behavioral Sciences)
[4] [Schemas and Memory Consolidation](https://consensus.app/papers/details/98c6a62f1f585d528372a57d705bbe74/?utm_source=claude_code) (Tse et al., 2007, Science)
[5] [Schemas provide a scaffold for neocortical integration of new memories over time](https://consensus.app/papers/details/23ec1d4244a15931a68ae6f5f969d909/?utm_source=claude_code) (Audrain et al., 2022, Nature Communications)
[6] [The Assimilation of Novel Information into Schemata and Its Efficient Consolidation](https://consensus.app/papers/details/c8ffdb98850c5f059f4c019c67fe372d/?utm_source=claude_code) (Sommer et al., 2022, Journal of Neuroscience)
[7] [Causal neural network of metamemory for retrospection in primates](https://consensus.app/papers/details/9c12d1787f425daf8e19fe4d7a6cae5d/?utm_source=claude_code) (Miyamoto et al., 2017, Science)
[8] [Hippocampal ripples and their coordinated dialogue with the default mode network during recent and remote recollection](https://consensus.app/papers/details/83446d6aaa29527683e8fb5d4e89e111/?utm_source=claude_code) (Norman et al., 2021, Neuron)
[9] [Development of the SPEAR Model: Separate Phases of Encoding and Retrieval Are Necessary for Storing Multiple Overlapping Associative Memories](https://consensus.app/papers/details/d046aee19b345362b973ff681c473b64/?utm_source=claude_code) (Hasselmo, 2024, Hippocampus)
[10] [Encoding and retrieval in the CA3 region of the hippocampus: a model of theta-phase separation](https://consensus.app/papers/details/f24ffcdc22595a779e9119983353f1c4/?utm_source=claude_code) (Kunec et al., 2005, Journal of Neurophysiology)
[11] [Septohippocampal acetylcholine and theta oscillations can modulate memory encoding and retrieval: insights from a neural masses network](https://consensus.app/papers/details/8a33ae751c67529982b15add37f1d505/?utm_source=claude_code) (Pirazzini et al., 2025, Brain Research Bulletin)
[12] [Episodic sequence memory is supported by a theta-gamma phase code](https://consensus.app/papers/details/1d1ed424bd64509c8a6515b12139adfb/?utm_source=claude_code) (Heusser et al., 2016, Nature Neuroscience)
[13] [Theta-gamma coupling as a ubiquitous brain mechanism: implications for memory, attention, dreaming, imagination, and consciousness](https://consensus.app/papers/details/e55e34c777be50ca811689e081247b09/?utm_source=claude_code) (Ursino et al., 2024, Current Opinion in Behavioral Sciences)
[14] [A model of working memory for encoding multiple items and ordered sequences exploiting the theta-gamma code](https://consensus.app/papers/details/e17f7fc053485161afe1538750fd0c28/?utm_source=claude_code) (Ursino et al., 2022, Cognitive Neurodynamics)
[15] [Generative replay underlies compositional inference in the hippocampal-prefrontal circuit](https://consensus.app/papers/details/5778d944825e58c4a85e91b410e485b6/?utm_source=claude_code) (Schwartenbeck et al., 2023, Cell)
[16] [Single-neuronal elements of speech production in humans](https://consensus.app/papers/details/31ccee4697f15d69b5934a4566ad9c1f/?utm_source=claude_code) (Khanna et al., 2024, Nature)
[17] [The neural basis of combinatory syntax and semantics](https://consensus.app/papers/details/23bd94d0f58050f1bf41341cd6c07610/?utm_source=claude_code) (Pylkkänen, 2019, Science)

---

**Next:** write the test-driven implementation plan (writing-plans), then
execute it subagent-driven under the pre-registered fixed-bar gate, honest
propagation of every outcome to both remotes, iterating following the biology
— autonomous, no hand-back.
