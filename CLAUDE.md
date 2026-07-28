# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Repository**: https://github.com/danthi123/neural-simulator

## ⭐ ACTIVE MISSION (2026-07-23 PIVOT) — read [`docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md`](docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md) + [`GAP_CLOSURE_MISSION.md`](GAP_CLOSURE_MISSION.md) (CURRENT STATE) FIRST, EVERY session

> **To SKIM (plain-language status surface):** read [`ROADMAP.md`](ROADMAP.md) — what's done, what's in progress, what's left, in prose, no shorthand. Its "Project shorthand" table decodes FHRR/BTSP/BDSP/GNW/gap#N/DR-N/RANK-N/EMERGE/the-moat/the-composer. The MASTER ROADMAP is the forward-looking PLAN; `GAP_CLOSURE_MISSION.md` CURRENT STATE is the live RESUME point.

**MAJOR DIRECTION PIVOT (owner, 2026-07-23).** The north-star is now a sim-brain that **CONVERSES GENUINELY** — reasons
to its OWN conclusions, with an **affective world-model + emotion + self-awareness + curiosity** — NOT fact-recall/RAG,
NOT LLM plausible-text. **Success is defined as TRUE CONSCIOUSNESS on the emergentist bet** (it emerges when a human
brain's full faculties + behavior are emulated COMPLETELY + FAITHFULLY enough), so the job is **completeness +
faithfulness of the biological emulation**, not a benchmark score. It is developed via a **TEMPORARY AI-teacher
scaffold** (accelerates early growth) that then **graduates to real-human interaction**; every scaffold is biologized
toward the one spiking brain (transformer minimized/retired). **HARD RULES: (1) DO NOT DEFER any functionality —
surpass EVERY wall with a mechanism rooted in real biology (no "characterized limit" / "honest negative" as a stopping
point); (2) speed is SECONDARY (slow-but-faithful biology — dendritic credit, seconds-long BTSP plateaus, sleep-replay
consolidation — is explicitly in scope; never trade faithfulness for speed); (3) ONE spiking substrate (host code only
for world + body + teacher-as-social-environment).** The **honesty boundary is a DELIVERABLE, not a caveat**: build +
measure every functional consciousness / self-model / affect CORRELATE, design every self-report as an honest
functional read-out ("my familiarity monitor reads this as novel, so I'm uncertain"), and NEVER assert phenomenal
experience.

**⭐ THE PRIMARY PLAN IS `docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md`** — a LIVING staged faculty roadmap
(complete faculty map · one-brain architecture · 6 developmental stages proto→human-ready · a 14-wall ledger where each
wall carries its biological surpass · a parallelization map). Foundation: `docs/plans/2026-07-22-genuine-conversation-affective-self-aware-brain-plan.md`.
**`GAP_CLOSURE_MISSION.md` remains the session-by-session working board** — its CURRENT STATE opens with this pivot
block + the live compute lanes; read the roadmap for the PLAN and the board's CURRENT STATE for the RESUME point. **The
prior "close the 5-gap cluster" framing is SUBSUMED** — the 5 gaps live on as the roadmap's faculty-map + walls-ledger
(still valid, now a sub-view), no longer the top-level mission.

**THE LAW (unchanged, now applied to every wall in the ledger): a wall/negative is a verdict on a METHOD, never a
license to abandon a CAPABILITY — bank the failing method, take a new biology/spiking/one-brain method, and keep going
until it WORKS. Closure cannot be deferred.** **SESSION START — OR ANY CONTINUATION (resumed from compaction): VERIFY a
within-session anti-stall + RUN-STATE heartbeat Monitor is live (a prior session's died with it; a continuation usually
has NONE), and arm one if not. It must be STATE-CHECKING (emits GPU / running-procs / recent-output every ~15 min) — a
text-only "are you idle?" nudge is insufficient (the 2026-07-24 failure was a live-but-stalled run, not idleness). Exact
recipe in `GAP_CLOSURE_MISSION.md` → "SESSION START". NEVER WAIT on a background run without a live state-heartbeat, and
never trust a subagent-armed Monitor / passive re-invocation to catch a completion. Then resume from CURRENT STATE.**
Cross-session continuation is MANUAL by owner choice (a plain "continue" + the roadmap + that board re-anchors) — no
watchdog/daemon.

## Keep the SUMMARY docs synced when a finding lands (2026-07-24 drift → the `sync-documentation` skill)

**Committing a finding is NOT enough — the summary docs must move WITH it, SAME cycle (deferring the sync IS the drift;
it happened all of 2026-07-24: findings committed, board left stale).** When a committed finding changes a wall/gap
STATUS, the CURRENT FRONTIER, or a "next action", **run the `sync-documentation` skill** — it now does BOTH the mechanical
drift (line counts / runner-test-findings counts / g11 flags / `sim/__init__` exports) AND the semantic summary-doc sync
(roadmap §7 wall-ledger + [`GAP_CLOSURE_MISSION.md`](GAP_CLOSURE_MISSION.md) CURRENT STATE + [`AUTONOMOUS_STATE.md`](research/findings/AUTONOMOUS_STATE.md)
+ `ROADMAP.md` status/frontier/next-action, contradictions, banners, the plain-language header + shorthand glossary). A
PostToolUse hook nudges it on `sim/` / `research/runners/` / findings changes — RUN it, don't just acknowledge the nudge.
Stale pointers are drift #12, the #1 cause of re-deriving concluded work.

## Evolve the workflows themselves (the `evolve-skills` skill)

**When a process lapse RECURS** (the owner had to catch the same *class* of problem twice), at a **session-end /
pre-compaction inflection**, or **when the owner asks** — run the **`evolve-skills`** skill: it reviews (with evidence)
what's WORKING and what's RECURRINGLY FAILING in our workflows, then makes INCREMENTAL updates to the applicable skills
so the workflows compound instead of re-learning the same lessons. Grounded, honest, lean (edits the on-demand skills,
never CLAUDE.md/memory bloat). A caught lapse IS a skill gap — evolve the skill so it can't recur, don't just patch the
instance. (Born 2026-07-24 after the owner caught three process lapses in one session.)

## When Compacting (custom compaction instruction — must survive into every compaction)

When this session auto-compacts or `/compact` runs, the summary MUST preserve (and may drop everything else to fit):
- **The ACTIVE MISSION block + the non-negotiables** (brain-based-only, one-brain, no-defer, speed-secondary, the honesty boundary) and the pointers to [`docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md`](docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md) + [`GAP_CLOSURE_MISSION.md`](GAP_CLOSURE_MISSION.md).
- **The current frontier + the exact next action:** the wall being worked, its GO-gate command + the anti-cheat controls, and the literal next command to run.
- **Live background work:** every running run / workflow / subagent ID + its Monitor, and every uncommitted result awaiting a verdict (so nothing is lost or double-launched).
- **Files created/modified this session + their purpose**, plus any `NO sim/ edit` / additive-default-off scope flags.
- **Owner directives given this session** — verbatim intent, not a paraphrase.

Summarize aggressively (keep only what changes a decision): git log, verbose run logs (error lines only), search-result dumps, exploratory file reads. Preserve any test / benchmark / GO-gate command VERBATIM.

**Context hygiene (2026-07-23):** history lives in [`docs/project-history-archive.md`](docs/project-history-archive.md) (RAG-indexed, `--corpus doc`), NOT inline — retrieve it, don't reload it. Prefer `/clear` between unrelated arcs (nav → conversation → gap#5) over one mega-session; offload heavy reading/search to subagents (their context doesn't count against the main window).

## Project Overview

GPU-accelerated neural network simulator with real-time 3D OpenGL visualization. Uses NVIDIA CUDA/CuPy for massively parallel GPU computation, simulating large-scale networks (10K-100K+ neurons) with biologically-inspired neuron models (Izhikevich, Hodgkin-Huxley, AdEx), synaptic plasticity, and spatial connectivity.

## Standing practice: deep research + catalog review FIRST at roadblocks and new directions

**(2026-06-07, owner directive — make this the default first step, not an afterthought.)** Whenever the project hits a **significant roadblock** (a multiply-confirmed boundary / repeated NEGATIVE) **OR is about to begin work on a new part of the sim**, run a **deep research + reference-catalog review BEFORE committing build/GPU resources.** This has repeatedly been the decisive pivot:
- the conversational decorrelation/whitening blocker → reframed by the Mikulasch-Priesemann point-neuron limit (whitening is analog/pre-spike in biology);
- the navigation action-selection readout boundary → diagnosed as a *missing accumulator* (Wang 2002 NMDA attractor → Lo-Wang commit burst), which fixed it;
- the navigation perceptual cold-start → root-caused as a **wrong-pathway** problem (routed through the position-*invariant* ventral "what" stream / IT instead of the dorsal "where" stream + superior-colliculus orienting + place cells) via the catalog + Kandel + literature.

**The pattern (LOCAL-FIRST — 2026-07-23 repair: the local corpus + RAG had rotted to dead `E:` paths, so the workflow silently fell back to online search; paths fixed + this is the mandatory FIRST move):** the FIRST move is our OWN local corpus via the auto-updating RAG index — `.venv-rag/bin/python tools/rag/rag_search.py "<question>" 5 [--corpus finding|plan|doc|catalog|kandel|paper|all]` (hybrid vector+BM25 → cross-encoder rerank; auto-refreshes on commit; SOMA retired). It spans our findings/plans/docs PLUS the canonical biology catalog (`~/Projects/sim-catalog/references/feature-catalog.md`, ~323 entries across clusters A–Q, the separate `sim-catalog` worktree), Kandel 6e full text (`~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt`), and 7 `.txt`-readable specialty textbooks/papers (Marr, Albus, Buzsáki, O'Keefe-Nadel, Schultz, Sutton-Barto, Bolam/Tepper BG — under `~/Projects/sim-catalog/references/textbooks/`), plus `references/glossary.md`. The RAG LOCATES; then READ the surfaced source in depth (a rerank hit is a pointer, not a paraphrase). ONLY after the local corpus is exhausted go external (WebSearch + the `bio-research` MCP). A read-only research subagent may run this and produce a findings doc: **diagnosis → ranked biologically-grounded options → what existing project machinery is reusable → a recommended cheap-first de-risk → the anti-cheat controls it needs.** The controller reviews it (trust-but-verify the load-bearing claims), pushes the doc, and presents the recommendation before building. Treat this as the standing opening move for roadblocks and new-direction work.

**The research gate — the AUTOMATIC trigger (2026-06-20, owner directive — make it mechanical, not a judgment call I can rationalize past; it failed once because "is this a significant roadblock?" is rationalizable).** Before committing ANY build / GPU / `sim/`-edit effort to *overcome* a difficulty, the gate fires (dispatch the read-only deep-research subagent FIRST, present its ranked options before building) if **ANY** of these objective conditions hold:
- **(a) Confirmed boundary:** an experiment/de-risk returned NEGATIVE / BOUNDARY / NO-GO / "walls" / "can't on this substrate," and the next move is a mechanism to push past it.
- **(b) Known family:** the wall is the same family as a prior documented boundary (the graded-magnitude / divisive-normalization / rate-code / point-neuron-limit / whitening family) — even on the FIRST occurrence in a new place.
- **(c) Blocks a goal:** the difficulty blocks a stated roadmap/goal item (not a side-nicety).
- **(d) New mechanism:** about to design a mechanism *class* not previously built (vs. composing already-proven pieces).
- **(e) `sim/`-to-overcome:** the candidate fix edits protected `sim/` code specifically to push past a limit.
- **(f) Stuck:** ≥2 distinct approaches to the same goal have failed.

**The self-check (the exact failure to prevent):** the moment I write or read a verdict containing NEGATIVE / BOUNDARY / NO-GO / "walls" / "can't" AND my next instinct is "scope/build the fix" — *that instinct IS the trigger.* The next action is the research gate, and the fix I had in mind becomes just ONE option the research ranks (it is never the default).

**The SURPASS sharpening (2026-06-20, owner directive — after a single owner sentence + ONE deep-research round overturned a too-comfortable "closed as a structural primitive" verdict AND found a cheap fix the controller had missed).** The gate fires not only before BUILDING a fix but **before ACCEPTING a boundary** — and "boundary" includes the SOFTER comfortable verdicts that quietly END investigation without a fix: "structural primitive," "honest negative," "not a shortcut," "the cost IS the deliverable," "characterized limit," "defensible," "that's just how the substrate is." Those are **DISGUISED boundaries** and are exactly where over-comfort hides. **Extended self-check:** the moment I write ANY conclusion that ends investigation of a difficulty *without a fix* — the hard NEGATIVE/BOUNDARY/NO-GO *or* the soft it's-a-primitive / honest-negative / not-a-shortcut / defensible — that IS the trigger. **The surpass deep-research round is MANDATORY and has FOUR moves (not just "diagnose + rank options"):** (1) **ISOLATE + QUANTIFY the genuine residual** — how big is the truly-irreducible part? Usually most of the "blocker" is already defensible or solved and the genuine residual is TINY (the FHRR-B "host-designed binding structure" was, on inspection, a single local `conj()` call; the rest was random-developmental codes + learned codes). Never accept a vague "the structure/op is host/hard" — pin down EXACTLY which bytes are the residual and measure them. (2) **REFRAME via "how does REAL biology actually do this?"** — am I testing the WRONG hypothesis? (we'd tested "can the bind be LEARNED from task data," which fails — but biology DEVELOPS the structure from local wiring rules, a different category with a cheap answer). (3) **RANK cheap-first SURPASS mechanisms** — the cheapest path PAST it, not merely a diagnosis. (4) **Verdict: surpassable-and-how-cheaply, vs genuinely-irreducible-and-precisely-why-defensible.** A boundary is accepted ONLY after it SURVIVES this round; the comfortable verdict is the START of the research, never the end.

**⚠️ THE LOOPHOLE THAT DEFEATED THE GATE (2026-07-26): a SEQUENCE of individually-cheap config tests IS a build effort.** Six levers / ~4 GPU-hours were spent against ONE defect without the gate ever subjectively firing, because no single flag felt like "committing build effort"; the research round then resolved in one pass what the sequential guessing had not. **MECHANICAL: ≥2 distinct levers tested against the SAME defect without resolution ⇒ the gate FIRES.** Cheapness of the next step is not an exemption — the quantity that matters is cumulative effort against one difficulty. Details + the measurement-placement rules in `.claude/skills/verify-go/SKILL.md`.

**Does NOT fire (so the gate stays calibrated, not over-triggering) — proceed directly:** routine/mechanical bugs with a clear cause (a backend mismatch, an off-by-one, a crash with an obvious fix); engineering that *composes* already-de-risked mechanisms; the GPU / multi-seed *confirmation* of an already-de-risked result; documentation, refactors, frontend wire-up. When genuinely unsure whether the gate fires, it fires (the read-only research is cheap relative to building the wrong fix).

## Standing standard: BRAIN-BASED ONLY (neurons / synapses / their communication), or it is a shortcut

**(2026-06-08, owner directive — the load-bearing bar for "a proper brain analogue".)** Anything NOT done directly by the simulated brain — **neurons firing, synapses, and the communication between them** — is a **cheat/shortcut, EVEN IF the host-side calculation is biologically correct.** A prediction error computed by a Python formula, a "reflex" that reads pixels and returns a cardinal in code, a reward computed by a distance formula, an argmax over spike counts — all are shortcuts, because the *brain* is not doing them; the simulation's bookkeeping is.

**The boundary — host code is legitimate ONLY for:**
1. **The environment** — the world's state (agent/goal positions, the grid) and rendering the agent's sensory input (the retinal image the neural retina then receives).
2. **The body** — the agent acting on its motor output (moving based on which motor pool fires).

**Everything between sensation and action is the brain's job and MUST be neurons/synapses:** perception/salience, orienting decisions, reward, value, dopamine/neuromodulators, action selection. When a capability is realized by host computation (even biologically-shaped), it is a documented shortcut to be converted to a spiking/synaptic mechanism — and an **honest negative** (the neural version underperforming the host shortcut) **IS the scientific deliverable** (it maps what the substrate can/can't do on its own). Applies **PROJECT-WIDE** (navigation AND the conversational pipeline — e.g. the VSA composer's clean exact-inverse algebra is a host shortcut for what a learned cortex would do; see the "composer-as-idealization" note). **Re-classification:** the recent nav wins (N1 SC reflex, N5 perceived reward, N6 thal/argmax readout, N9-step-1 scalar RPE) are biologically-*shaped* but partly **host-computed → they are now shortcuts**, with their spiking/synaptic versions (a spiking superior colliculus, a neural reward/value system, a spiking SNc, a neural position code, a minimal motor read-out) the real target. The host versions become the *teaching scaffolds* for their neural replacements (the innate-reflex-teaches-a-learned-circuit pattern).

> _Archived: **Recent-arc narratives** (was CLAUDE.md L76-236) → [`docs/project-history-archive.md`](docs/project-history-archive.md); retrieve via `.venv-rag/bin/python tools/rag/rag_search.py "<q>" --corpus doc`._
## Common Commands

```bash
# Run the simulator (GUI mode)
python neural-simulator.py

# Run headless auto-tuning (parameter sweep)
python neural-simulator.py --auto-tune
python neural-simulator.py --auto-tune --quick  # Faster reduced sweep

# Run performance benchmarks
python benchmark.py --output results.json
python benchmark.py --quick  # Reduced configurations

# Run visualization performance benchmark
python viz_benchmark.py --output benchmarks/viz_performance_results.json
python viz_benchmark.py --quick  # Faster test

# Run biological validation suite (Bi & Poo STDP, E/I balance, STP PPR, gamma, homeostasis)
python run_benchmarks.py --benchmark stdp-timing
python run_benchmarks.py --benchmark ei-balance
python run_benchmarks.py --benchmark stp-paired-pulse
python run_benchmarks.py --benchmark gamma-oscillations
python run_benchmarks.py --benchmark homeostasis

# Run a research-gate runner (G1..G11)
python -m research.runners.g11_bg_runner --moving-goal --seed 42 --n-steps 1800 \
    --out research/findings/raw/g11_bg/g11_seed42.json
python -m research.runners.g11_bg_runner --probe-action W   # static cascade probe

# Run headless experiments (4 built-in presets)
python run_experiment_headless.py --preset rl --seed 42

# Parameter sweep
python run_parameter_sweep.py -e associative --sweep "stdp_a_plus=0.004,0.012,0.024"

# Run all tests
pytest tests/ -v

# Targeted test suites
pytest tests/test_determinism.py -v
pytest tests/test_experiment_system.py -v
pytest tests/test_neuromodulators.py -v
pytest tests/test_regions.py -v
```

## Architecture

### Modular Package Layout

The simulator was originally a single ~12K-line `neural-simulator.py`. As of 2026-04 it is refactored into modular packages — `neural-simulator.py` is now just the GUI host (2.2K lines), and the engine lives in `sim/`.

```
neural-simulator.py     # 2.2K lines — DearPyGUI host + main entry point only
sim/                    # 42 modules (+ __init__.py), ~20K lines — core engine
  bridge.py             # 9529 lines — SimulationBridge + GPU state orchestration (incl. transmission_gate, graded inhibition, input-mean adaptation, RF complex-synapse ops)
  config.py             # 1284 lines — all @dataclass configs
  enums.py              #  830 lines — NeuronType (50+ presets), enums, default param managers
  connectivity.py       #  999 lines — spatial/WS/motif connection generators (backend-pluggable)
  kernels.py            #  707 lines — fused @fuse() neuron + plasticity kernels (cupy/numpy)
  profiles.py           #  432 lines — NEURAL_STRUCTURE_PROFILES + CONNECTIVITY_MOTIFS dicts
  regions.py            #  791 lines — BrainRegion + RegionPathway (incl. transmission_gate, graded, input_mean_adapt) + RegionManager
  neuromodulators.py    # 1135 lines — declarative neuromodulator subsystem
  data_bus.py           #   95 lines — DataChannel pub/sub for streaming sim data
  replicas.py           #  243 lines — replicated wiring (multi-bridge support)
  text_embeddings.py    #  273 lines — token embeddings for language regions (2026-05-01)
  visual_cortex.py      #  310 lines — Gabor RFs + retina rendering (Cluster K v2, 2026-05-01)
  bioparameter.py       #  231 lines — biological parameter helpers
  progress.py           #  214 lines — universal [PROGRESS] event format (2026-05-04)
  lineage.py            #  538 lines — BridgeLineage persistent continuous-learning + growth-log + shard export (2026-05-11)
  auto_growth.py        #  357 lines — TierPromoter + weight-transfer (auto-growth Phase A, 2026-05-11)
  backend.py            #  415 lines — pluggable xp abstraction + device helpers + RNG state (cupy/numpy, 2026-05-11)
  synapse_storage.py    #  415 lines — TieredSynapseStore + idle/pressure eviction (tiering Phase 3+4, 2026-05-11)
  bridge_memory.py      #  721 lines — BridgeMemory LLM-callable memory wrapper (Path 3 Phase 3.1.6, 2026-05-11)
  llm_memory_orchestrator.py #  452 lines — MockLLM + LLMMemoryOrchestrator tool-use loop, 5 tool schemas (Phase 3.2, 2026-05-11)
  llm_adapters.py       #  204 lines — OllamaLLM + LlamaCppLLM stub adapters (Phase 3.3 scaffold, 2026-05-11)
  # (plus ~20 newer modules from the language-generation + learned-cortex arc:
  #  surrogate_grad / bptt_snn / bptt_snn_gpu / char_tokenizer / bpe_tokenizer /
  #  tiny_transformer / ngram_* / td_value_critic / predictive_coding /
  #  dendritic_* / compose_temporal_bind / song_hvc / activity_probe / …)
viz/                    # OpenGL renderer, camera, picker, overlays
ui/                     # DearPyGUI panels, callbacks, layout, sweep panel, plots
experiment/             # ExperimentEngine + StimulusManager + ReadoutEngine + TrainingProtocolEngine
research/runners/       # 1000+ headless runners (g1..g11 + cluster/text/k_v2/phase1/phase2/chat/perf_benchmark/bridge_lineage/llm_memory_demo/multibridge_chat/g20_multibridge/g20_sparse/order_intrinsic/generator_S-D-E-F-G/mode-unification/content_selection+content_selection_spiking+dialogue_agent/nested_composition+phasor_associative_memory+phasor_chat+gated_compose/unified_agent_benchmark+spiking_unified_agent/multibridge_graded_derisk+cortex_conversation_ensemble+phase1_composer_ab/one_brain_composer/grounded_lang/bridge_coresidence/longitudinal_develop_loop/_emerge*/_d3_*/etc) for research
research/findings/      # session-by-session findings docs (1800+ files)
tests/                  # 450+ test files (determinism, runners, kernels, plasticity, lineage, tiering, llm orchestrator, multibridge, g20-sparse, generator/BPTT, order-intrinsic, mode-unification, content-selection/dialogue, nested-composition, transmission-gate/gated-compose, unified-agent-benchmark/spiking-unified-agent, core-sim-composition + brain-conversational-agent, learned-graded cortex, one-brain composer/agent, emerge competitive-pooler/taxonomy, etc.)
```

### Thread Model
- **Main Thread**: DearPyGUI event loop + OpenGL rendering
- **Simulation Thread**: GPU-accelerated neural dynamics computation (fully isolated)
- **Communication**: Lock-free queues (`ui_to_sim_queue`, `sim_to_ui_queue`) for inter-thread messaging

### Key Classes

**SimulationBridge** (`sim/bridge.py:217`): Central simulation orchestrator
- Manages all GPU state arrays (CuPy)
- Simulation stepping (`_run_one_simulation_step` at line 6523)
- Initialization (`_initialize_simulation_data` at line 1218)
- Recording/playback to HDF5
- Checkpoint save/restore
- Profiling and performance monitoring

**Configuration Dataclasses** (all in `sim/config.py`):
- `CoreSimConfig` (line 27): Network topology, neuron models, plasticity, biological realism
  - STP fields: `stp_U`, `stp_tau_d`, `stp_tau_f` (global defaults)
  - Per-connection-type STP: `enable_per_type_stp`, `stp_U_per_type[4]`, `stp_tau_d_per_type[4]`, `stp_tau_f_per_type[4]`
  - Structural plasticity: `struct_plast_activity_bias` (0.0–1.0) for activity-dependent synaptogenesis
  - Homeostasis: EMA alpha (~0.0002, tau ~5s) and threshold adapt rate (~0.0005)
  - Inhibitory reversal: `E_inh = -75mV`, propagation scaled 0.7x for driving force compensation
  - HH numerical stability: dt auto-adjusts to 0.05ms when HH model selected
  - **Per-gate Q10**: `hh_q10_m=3.0`, `hh_q10_h=hh_q10_n=1.5` (fixed 2026-04-25 — uniform Q10=3 over-compressed dynamics at 37°C; see Phase A below)
  - **STDP bounds gotcha**: `stdp_w_max=2.0` default. The STDP rule is **soft-bound** (`Δw_LTP = A_plus * (w_max - w) * exp(...)`) so when `weight_mean > stdp_w_max`, every "LTP" event is strongly negative and weights collapse to w_max within ms. Set `cfg.stdp_w_max` above your design weights (e.g. cortex→D1 in Phase B uses `weight_mean=25` → set `stdp_w_max=30`). **⚠️ THIS TRAP IS PER-RULE AND HAS NOW HIT FOUR RULES — STDP (`stdp_w_max`), BDSP (`bdsp_w_max`, below), BTSP (`btsp_w_max` — saturation silently crushed a rank-1 write to a flat null, 2026-07-25) and HEBBIAN (`hebbian_max_weight` **defaults to 1.0**, far below typical design weights: at a 3.015 pathway every "potentiation" was strongly negative and collapsed the TRAINED and UNTRAINED pathways identically, reading as "the rule doesn't help here"). **STANDING PRE-FLIGHT for ANY plasticity rule: compare its bound against the ACTUAL weight (`_mean_gate_weight(bridge, gate)` vs `cfg.<rule>_max_weight`), and verify the trained pathway moves DIFFERENTLY from an untrained control.** A bound below the weights does not merely fail to learn — it destroys weights uniformly, which reads as a substrate limitation.
  - **BDSP clamp-at-lr=0 gotcha** (2026-07-24, commit 6a9a44c3): `fused_bdsp_update` applies `cp.clip(w, bdsp_w_min=-5, bdsp_w_max=5)` **unconditionally — even at `lr=0`** (a frozen/control arm), so any weight outside ±5 is silently flattened to the bound (it collapsed a gap#5 encode store to `bdsp_w_max=5` and plausibly caps gap#4's ±5-bounded FF weights on a 9-way task). Set `bdsp_w_max` above your design weights, and don't assume `lr=0` means "no weight change" for BDSP. A `sim/` clamp-fix (gate the clip by lr / plasticity gain, mirroring the STDP masked-clip) is filed.
- `VisualizationConfig` (line 900): OpenGL rendering and camera parameters
- `RuntimeState` (line 920): Mutable execution state (running, paused, time tracking)
- `GPUConfig` (line 935): GPU features, memory management, recording modes
- Experiment configs (lines 1048–1227): `StimulusPattern`, `StimulusChannel`, `NeuronGroup`, `ReadoutConfig`, `TrainingConfig`, `ExperimentPhase`, `ExperimentConfig`

### GPU Array Naming Conventions
- `cp_*`: CuPy GPU arrays (e.g., `cp_membrane_potential_v`, `cp_firing_states`)
- `gl_*`: OpenGL handles/VBOs
- `fused_*`: GPU kernel functions decorated with `@cp.fuse()`

### Simulation Step Pipeline (in `_run_one_simulation_step()`)
1. STP (Short-Term Plasticity) update – per-connection-type if enabled
2. Synaptic conductance update – uses E_inh = -75mV with 0.7x propagation scaling
2b. **Experiment stimulus injection** – if ExperimentEngine is running, adds stimulus current
3. Background noise (OU process)
4. Neuron dynamics (model-specific: Izhikevich/HH/AdEx)
5. Plasticity updates (Hebbian, STDP, reward modulation, structural with activity bias, homeostasis)
6. Visualization updates
7. Recording (if active)

**Note on dt Auto-Adjustment**: When switching to Hodgkin–Huxley model, dt is automatically
reduced to 0.05ms for numerical stability of voltage-gated kinetics. When switching to Izhikevich
or AdEx, dt restores to 0.5ms. This occurs in `apply_simulation_configuration_core()`.

### Fused CUDA Kernels (`sim/kernels.py`)
Performance-critical GPU operations decorated with `@cp.fuse()`:
- `fused_izhikevich2007_dynamics_update()`: 9-parameter Izhikevich model
- `fused_izhikevich_legacy_dynamics_update()`: Legacy 4-param Izhikevich
- `fused_hodgkin_huxley_dynamics_update()`: Temperature-dependent HH
- `fused_adex_dynamics_update()`: Adaptive Exponential IF
- `fused_hh_m_current_update()`, `fused_hh_CaT_current_update()`, etc.: Extended HH currents
- `fused_hh_h_current_update()`: HH h-current (Ih)
- `fused_hh_NaP_current_update()`: HH persistent sodium current
- `fused_conductance_decay_and_current()`: Synaptic dynamics
- `fused_nmda_update_and_current()`: NMDA voltage-dependent Mg2+ block
- `fused_stp_decay_recovery()`: Short-term plasticity Tsodyks-Markram
- `fused_stdp_weight_update()`: Spike-timing dependent plasticity
- `fused_homeostasis_update()`: Homeostatic firing rate regulation
- `fused_eligibility_trace_decay()`: Reward modulation eligibility traces

### Hodgkin-Huxley Presets (`sim/enums.py`)
Region-specific HH parameter dicts in `DefaultHodgkinHuxleyParams`, derived from `REALISTIC_L5_PYRAMIDAL_RS_37C` base with region overrides. All retuned 2026-04-25 (per-gate Q10 fix):

Cortical: `HH_L5_CORTICAL_PYRAMIDAL_RS`, `HH_PFC_PYRAMIDAL`, `HH_CORTICAL_FS_INTERNEURON`
Hippocampus: `HH_CA1_PYRAMIDAL_BURST`, `HH_CA3_PYRAMIDAL_BURST`
Thalamus: `HH_THALAMIC_RELAY_TBURST`, `HH_TRN_BURST_INHIB`
Basal ganglia: `HH_STRIATAL_MSN`, `HH_STRIATAL_MSN_D1`, `HH_STRIATAL_MSN_D2`, `HH_STRIATAL_TAN`, `HH_STN_BURST`, `HH_GPE_PACEMAKER`, `HH_GPI_OUTPUT`, `HH_DOPAMINE_SNC`
Cerebellum: `HH_CEREBELLAR_PURKINJE`, `HH_CEREBELLAR_GRANULE`
Spinal: `HH_SPINAL_MOTOR`, `HH_SPINAL_INTERNEURON`
Other: `HH_OLFACTORY_MITRAL`, `HH_INFERIOR_OLIVE`

### Izhikevich 2007 Presets (`sim/enums.py`)
9-parameter Izhikevich-2007 presets in `DefaultIzhikevichParamsManager`, used by `cfg.default_neuron_type_izh` and per-region `izh_neuron_type` overrides. Cortical: `IZH2007_RS_CORTICAL_PYRAMIDAL`, `IZH2007_FS_CORTICAL_INTERNEURON`. BG: `IZH2007_STRIATAL_MSN`, `IZH2007_STRIATAL_MSN_D1`, `IZH2007_STRIATAL_MSN_D2`, `IZH2007_STRIATAL_TAN`, `IZH2007_GPE_PACEMAKER`, `IZH2007_GPI_OUTPUT`, `IZH2007_STN_BURST`, `IZH2007_DOPAMINE`. Thalamus: `IZH2007_THALAMIC_RELAY`, `IZH2007_THALAMIC_RETICULAR`. Hippo: `IZH2007_HIPPO_PYRAMIDAL`. (Fixed 2026-04-25: bridge previously ignored `default_neuron_type_izh` because trait-split was always-on; now opt-in only when `num_traits > 1`.)

### AdEx Presets (`sim/enums.py`)
Brette & Gerstner 2005 phenotypes in `DefaultAdExParamsManager`: `ADEX_RS`, `ADEX_FS`, `ADEX_IB`, `ADEX_CH`, `ADEX_LTS`, `ADEX_MSN`, `ADEX_DOPAMINE`. (Fixed 2026-04-25: bridge now overlays preset params onto `cfg.adex_*` fields — previously all 7 presets behaved identically because preset wasn't loaded.)

### Neural Structure Profiles (`sim/profiles.py`)
Region presets that configure traits, connectivity, and default parameters. Defined in the `NEURAL_STRUCTURE_PROFILES` dict:
- `GENERIC_UNSTRUCTURED`
- `CORTEX_L23_RS_FS`, `CORTEX_L4_INPUT_LAYER`, `CORTEX_L5_DEEP_OUTPUT`
- `PREFRONTAL_CORTEX_WM`
- `HIPPOCAMPUS_CA1_RS_FS`, `HIPPOCAMPUS_CA3_RECURRENT`
- `BASAL_GANGLIA_STRIATUM`, `BASAL_GANGLIA_STN_GPE`
- `THALAMUS_TC_TRN`
- `CEREBELLAR_CORTEX_SIMPLE`, `SPINAL_CORD_SEGMENT`
- `OLFACTORY_BULB`, `DOPAMINERGIC_MIDBRAIN`
- `CORTEX_GAMMA_FS_NETWORK`, `INFERIOR_OLIVE`

### Profile Naming Convention
Each brain region has three JSON profile variants in `simulation_profiles/`:
- `{region}_hh.json`: Full biophysics (Hodgkin-Huxley, dt=0.05ms)
- `{region}_adex.json`: Adaptive Exponential (dt=0.5ms, 10-20× faster than HH)
- `{region}_izh.json`: Izhikevich fast testing (dt=1.0ms, fastest)
- Plus `quick_demo_cortex.json` for beginners

### JSON Profile Dropdown System (`ui/`)
Full simulation profiles saved as `.json` in `simulation_profiles/` (47 files). A UI dropdown auto-populates from this directory. Key functions live in the UI package: `_scan_profile_directory()`, `_handle_full_profile_dropdown_change()`, `_refresh_full_profile_dropdown()`.

### UI-Config Roundtrip
Two critical functions must be kept in sync for profile save/load to work correctly:
- `_update_sim_config_from_ui()`: Extracts all parameter values from UI widgets and builds `CoreSimConfig`, `VisualizationConfig`, `RuntimeState`, and `GPUConfig` dataclasses
- `_populate_ui_from_config_dict()`: Takes a configuration dictionary and updates all UI widgets to reflect those values

These are inverse operations: any parameter exposed in the UI must have a corresponding getter and setter to ensure bidirectional sync between UI state and simulation configuration.

### Experiment & Stimulus System (`experiment/` package)
Programmable infrastructure for stimulus injection, I/O neuron group management, training protocols, readout/analysis, and multi-phase experiment orchestration. Configs live in `sim/config.py` (lines 1048–1227); engines live in `experiment/`.

**Key Classes:**
- `StimulusManager` (`experiment/stimulus.py`): Generates per-step GPU current arrays from channel definitions
- `NeuronGroupManager` (`experiment/groups.py`): Manages designated populations (input/output/hidden)
- `ReadoutEngine` (`experiment/readout.py`): Population rates, spike counts, PSD via FFT, Fano synchrony, band power
- `TrainingProtocolEngine` (`experiment/training.py`): Trial state machine for RL/supervised/associative
- `ExperimentEngine` (`experiment/engine.py`): Top-level orchestrator called once per simulation step
- `ExperimentPresets` (`experiment/presets.py`): Factory for 4 common experiment configurations

**Stimulus Pattern Types:** CONSTANT, PULSE_TRAIN, SINUSOIDAL, RAMP, POISSON_SPIKE_TRAIN, GAUSSIAN_NOISE, CUSTOM_WAVEFORM

**Training Modes:** ASSOCIATIVE_PAIRING (Rescorla-Wagner), REINFORCEMENT_LEARNING (R-STDP), SUPERVISED_TARGET, RESERVOIR_READOUT

**Built-in Presets:**
- Basic Stimulus-Response: inject current, measure output transfer function
- Associative Conditioning (CS-US): Pavlovian pairing with STDP learning
- Reinforcement Learning (R-STDP): Three-factor learning with reward/punishment
- Frequency Response Characterization: Sinusoidal sweep for bandpass analysis

**Integration Points:**
- SimulationBridge: `self.experiment_engine` initialized in `apply_simulation_configuration_core()`
- Simulation step: experiment stimulus injected after synaptic current, before OU noise
- Queue messages: LOAD_EXPERIMENT_PRESET, LOAD_EXPERIMENT_CONFIG, START_EXPERIMENT, STOP_EXPERIMENT, GET_EXPERIMENT_STATUS, SAVE_EXPERIMENT_LOG
- Checkpoint: experiment config saved/restored as JSON attribute in HDF5
- UI: "Experiment & Stimulus System" collapsing header with preset selector, controls, status display

**Running Tests:**
```bash
pytest tests/test_experiment_system.py -v
```

### Neuromodulator Subsystem (Session E.1, opt-in)

Declarative framework for hormones / neuromodulators with concentration
dynamics and configurable receptor effects on bridge state. Replaces
the one-off `current_reward_signal` and shelved `cp_synaptic_gain_modulator`
mechanisms. Default OFF for full backward compatibility.

**Module:** `sim/neuromodulators.py`

**Config (in `CoreSimConfig`):**
- `enable_neuromodulator_subsystem: bool = False` — opt-in flag
- `neuromodulators: List[NeuromodulatorConfig]` — list of declared modulators

**Three dataclasses:**
- `NeuromodulatorConfig(name, baseline, decay_tau_ms, concentration_min/max, targets, production_rules)`
- `ModulatorTarget(target_type, scope, sensitivity)` — receptor effect spec
- `ProductionRule(rule_type, sensitivity, threshold, window_ms)` — what drives concentration

**Built-in target types:**
- `synaptic_gain` — multiplies effective synaptic strength (scope=all only)
- `plasticity_rate` — multiplies reward_learning_rate (scope=all)
- `excitability_drive` — adds pA to membrane drive (scope=all, trait:N, group:NAME)

**Built-in production rules:**
- `manual` — only set externally (testing, experiments)
- `from_reward` — adds sensitivity*(current_reward_signal - reward_baseline) per step
- `from_error_persistence` — EMA of |error| > threshold drives sustained tonic increase

**Bridge integration:**
- Manager allocated in `_init_synapse_arrays_with_capacity` when subsystem enabled
- `manager.step(self)` called once per simulation step after C2 reward modulation
- `compute_synaptic_gain_multiplier()` applied in `effective_synaptic_strength`
- `compute_plasticity_rate_multiplier()` applied to `reward_learning_rate`
- `compute_excitability_drive_pA()` + `compute_excitability_drive_per_neuron()` added to `total_input_current_pA`

**Group registration:**
Runners that want `scope="group:NAME"` targets must call
`bridge.neuromodulator_manager.set_group_indices({name: indices})`
after the engine groups are known. G9 runner does this automatically
for the standard input/hidden/motor groups.

**Plan:** `docs/plans/2026-04-24-neuromodulator-subsystem.md`

**Running tests:**
```bash
pytest tests/test_neuromodulators.py -v
```

### Brain-Region Framework (Session E.2, opt-in)

Declarative framework for multiple brain regions (PFC, Motor, Hippocampus,
Striatum, etc.) on a single bridge. Each region owns a contiguous slice
of neuron indices with its own internal connectivity; cross-region
pathways are declared rather than hand-wired. Composes with the
neuromodulator subsystem from E.1 — pathways can declare
`neuromodulator_gates` and regions auto-register as neuromodulator groups.

Default OFF for full backward compatibility.

**Module:** `sim/regions.py`

**Config (in `CoreSimConfig`):**
- `enable_brain_region_framework: bool = False` — opt-in flag
- `brain_regions: List[BrainRegion]` — declared regions
- `region_pathways: List[RegionPathway]` — directed projections

**Two dataclasses:**
- `BrainRegion(name, n_neurons, exc_fraction, internal_density, exc/inh_weight_mean, weight_jitter, plastic_internal, nm_outputs)`
- `RegionPathway(from_region, to_region, density, weight_mean, weight_jitter, plastic, neuromodulator_gates)`

**Manager:** `RegionManager(regions, pathways)`
- `initialize(seed)` — allocate contiguous index slices + deterministic inh selection
- `total_neurons()` — sum across regions (auto-sets `num_neurons`)
- `indices(name)` / `inhibitory_indices(name)` — per-region lookups
- `region_indices_dict()` — for `nm_mgr.set_group_indices()`
- `build_wiring_plan(seed)` — yields plan dict consumed by `inject_explicit_wiring`

**Bridge integration:**
- Bridge allocates `region_manager` BEFORE neuron arrays (so num_neurons
  is set from `region_manager.total_neurons()`).
- Wiring is generated by `build_wiring_plan()` and fed through
  `inject_explicit_wiring()` (replacing legacy motif/WS/spatial paths).
- When BOTH frameworks are on, regions auto-register as neuromodulator
  groups so `ModulatorTarget(scope="group:PFC")` resolves natively.

**Plan:** `docs/plans/2026-04-24-brain-region-framework.md`

**Running tests:**
```bash
pytest tests/test_regions.py -v
```

### Motor Exploration Noise (Session G)

**Purpose:** Defeats the silent-motor trap (motor neurons that never fire in
phase 1 cannot acquire STDP eligibility, so reward-mediated weight updates
never reach them; agent stays glued to phase-1 winners even when reward
flips sign).

**Mechanism:** Inject independent Poisson spike trains into each output
neuron during the stimulus integration window. Each event is a strong
spike-driving current pulse, so every motor fires occasionally regardless
of upstream activity. STDP can then form positive eligibility on
hidden→silent-motor synapses; reward converts those into weight changes.

**Implementation:** Reuses existing `StimulusManager` POISSON_SPIKE_TRAIN
support — no new GPU code. The G9 runner adds a second `StimulusChannel`
alongside the sensor channel when `motor_exploration_rate_hz > 0`.

**Runner kwargs (`research/runners/g9_runner.py`):**
- `motor_exploration_rate_hz` (default 0.0 — backward compatible)
- `motor_exploration_current_pA` (default 1000.0)
- `motor_exploration_spike_ms` (default 2.0)

Typical working range: 5-30 Hz (~0.5-3 spurious spikes per motor per 100 ms
readout window). 0 disables. Above ~50 Hz starts to dominate action selection.

**Relation to ε-greedy:** Equivalent to ε-greedy / entropy regularization /
Boltzmann exploration in tabular RL, just at the spike-event level instead
of the action-distribution level. Biologically grounded in tonic dopamine
driving spontaneous striatal/cortical activity (Schultz 2007).

**Plan / findings:** `research/findings/2026-04-25-session-g-motor-exploration.md`

### Phase B BG Action Selection Module (resolved silent-motor trap)

**Status (2026-04-25):** GO. 3-seed acid test, phase 1 finalQ avg 1.76 vs G9 baseline 6.74 (74% improvement).

**Why it exists:** Sessions D–I tried 7 runner-side variants (V1–V7) of motor exploration / ε-greedy / proportional sampling to break the silent-motor trap. All were NEGATIVE. The trap was structural: a shared 200-neuron reservoir + argmax readout has a dominant-motor bias from random initial weights that no runner-side hack can fix.

**The fix:** Replace reservoir+argmax with a per-action BG cascade. Each action has its own dedicated populations: `cortex_X → str_D1_X / str_D2_X → gpi_X → thal_X → motor_X` with disinhibition gating (D1 inhibits GPi → thal released → motor fires). No shared argmax; selection happens via independent gates.

**Builder:** `research.runners.g11_bg_runner.build_bg_brain_regions(n_cortex=100)` — returns `(regions, pathways)` with 30 regions and 32 pathways (per-action cortex / D1 / D2 / GPe / GPi / Thal / Motor pools + shared STN + dopamine; ~14.5K synapses).

**Two non-obvious bugs that almost killed the architecture** (both fixed 2026-04-25):
1. `n_cortex=400` over-drove D1 to ~220 Hz (saturated, unphysiological), GPi couldn't silence past STN excitation. **Fix:** use `n_cortex=100` (25 cortex/action). The static probe used 100; the moving-goal runner shipped with 400, so the probe "passed" but the deployment failed. Lesson: probes must call the same builder with the same args as deployment.
2. `cortex→D1` weight_mean=25 against default `stdp_w_max=2` collapsed weights from 25→2 in milliseconds via soft-bound STDP. **Fix:** set `cfg.stdp_w_max = 30.0` in the runner.

**Findings:**
- `research/findings/2026-04-25-phase-b-acid-test-real-win.md` — final 3-seed GO result + diagnosis
- `research/findings/2026-04-25-phase-b-cascade-stability-fix.md` — bug 1 (n_cortex)
- `research/findings/2026-04-25-phase-b-honest-correction.md` — early overstated finding
- `research/findings/2026-04-25-phase-b-bg-acid-test.md` — initial (overstated) result kept for trail

> _Archived: **Phase B refinement** (was CLAUDE.md L613-619) → [`docs/project-history-archive.md`](docs/project-history-archive.md); retrieve via `.venv-rag/bin/python tools/rag/rag_search.py "<q>" --corpus doc`._
### 🎉 Plastic-input-layer arc RESOLVED (2026-04-27)

After 7 NEGATIVE attempts on 2026-04-26, the plastic-input-layer
problem was resolved on 2026-04-27 via per-pathway plasticity gating
infrastructure + real curriculum learning. See
[`research/findings/2026-04-27-plastic-input-layer-RESOLVED.md`](research/findings/2026-04-27-plastic-input-layer-RESOLVED.md)
and [`research/findings/2026-04-27-task-adaptive-curriculum.md`](research/findings/2026-04-27-task-adaptive-curriculum.md).

Key new infrastructure:
- `RegionPathway.plasticity_gate: str | None` — tag pathways for runtime gating
- `bridge.set_plasticity_gate(name, value)` — freeze/thaw at runtime
- `cp_plasticity_rate_gain` array — gates STDP, eligibility, Hebbian, synaptic scaling (renamed from `cp_plasticity_gain` 2026-04-29; old name is a deprecated property alias)
- NM-driven gates: `target_type="plasticity_gate", scope="gate:<name>"`

> **GOTCHA — plasticity gate vs synaptic transmission (2026-04-28):**
> `cp_plasticity_rate_gain` and `set_plasticity_gate(...)` freeze weight UPDATES
> only — STDP, eligibility, Hebbian, synaptic scaling. They do NOT freeze
> synaptic CURRENT (`g_syn × (V - E)`). A frozen pathway with non-zero
> `weight_mean` still injects current and affects forward dynamics. To
> staged-introduce a new pathway without disrupting the system before
> the thaw step, initialize it with `weight_mean=0.0` (then let STDP grow
> it from zero after thaw) — OR add a runtime weight scale per gate
> (small bridge change, not yet implemented). The cheat-5 v1 NEGATIVE
> result (2026-04-28) was caused by missing this distinction; v2 fixes
> it via zero-init.
>
> **UPDATE (2026-06-03): the complement now EXISTS — `transmission_gate`.**
> `RegionPathway(transmission_gate="name")` + `bridge.set_transmission_gate(name, value)`
> scale a pathway's effective synaptic **CURRENT** in [0,1] at runtime
> (the `cp_transmission_gain` per-synapse multiplier in `_run_one_simulation_step`,
> mirroring `cp_plasticity_rate_gain` but on current, not weight updates).
> Pre-wire a route with a fixed weight, hold it CLOSED (gate=0, no current,
> no STDP cold-start), OPEN it on command → **thalamocortical dynamical
> gating**: binding = which gate is open, not which weight grew
> (Logiaco-Abbott-Escola 2021). Validated in spikes
> (`tests/test_transmission_gate.py`): closed → target silent; open → target
> fires; re-binding reroutes the same source with **zero weight change**,
> where grown weights could not. Default `None` = always-on (additive, zero
> overhead unused). See `2026-06-03-deep-research-surpassing-the-blockers-synthesis.md`.

Curriculum: phase 1 corticostriatal plastic + input layers frozen; phase 2
cortex frozen (or partial) + input layers thawed. Biologically: real
critical periods close gradually, gated by neuromodulators, allowing
sensory cortex to mature before association cortex.

### Pluggable backend (2026-05-11): NumPy backend SHIPPED end-to-end

**Status:** Phases 1+2 of the tiering design SHIPPED 2026-05-11.
SimulationBridge construction + initialization + simulation steps +
brain region framework + checkpoint save/load + bio_three_factor
training + chat_repl W→A + chat_repl :speak A→W ALL work end-to-end
under `SIM_BACKEND=numpy`. No NVIDIA/CUDA dependency required.

CuPy backend remains the production speed path (4-50× faster than
NumPy depending on workload). NumPy backend is for portability +
verification + CI + low-end hardware.

**Usage:**
```bash
# Default (CuPy if available, else NumPy)
python -m research.runners.chat_repl --mode tier1 --seed 42

# Force NumPy backend (Mac M-series, GPU-less Linux, CI)
SIM_BACKEND=numpy python -m research.runners.chat_repl --mode tier1 --seed 42

# Force CuPy explicitly (or fail if unavailable)
SIM_BACKEND=cupy python -m research.runners.chat_repl --mode tier1 --seed 42
```

Findings:
- `research/findings/2026-05-11-numpy-backend-shipped.md` (Phase 2 milestone)
- `research/findings/2026-05-11-numpy-backend-chat-repl-shipped.md` (full chat pipeline)

Design doc: [`docs/plans/2026-05-11-cpu-ram-ssd-tiering-design.md`](docs/plans/2026-05-11-cpu-ram-ssd-tiering-design.md)
Strategic context: [`docs/plans/2026-05-11-strategic-reevaluation.md`](docs/plans/2026-05-11-strategic-reevaluation.md)

**Pattern for new code:** instead of `import cupy as cp`, use:

```python
from sim.backend import get_backend, fuse, synchronize, to_host
xp, backend_name = get_backend()

@fuse()
def my_kernel(a, b):
    return a + b  # works on both cupy + numpy backends
```

**Backend selection** (in priority order):
1. Explicit `get_backend("cupy")` or `get_backend("numpy")` (test code)
2. `SIM_BACKEND` env var (`cupy` / `numpy` / `auto`)
3. Cached backend from a prior call (sticky)
4. Auto-detect: CuPy if installed AND `cp.cuda.runtime.getDeviceCount() > 0`,
   else NumPy

**Helpers exposed by `sim.backend`:**
- `get_backend()` — returns `(xp_module, backend_name)`
- `get_sparse_module()` — `cupyx.scipy.sparse` or `scipy.sparse`
- `is_gpu_backend()` — True if active backend is CuPy
- `fuse(...)` — decorator that's `cp.fuse()` on CuPy, no-op on NumPy
- `synchronize()` — `cp.cuda.Stream.null.synchronize()` on CuPy, no-op on NumPy
- `to_host(arr)` / `from_host(arr)` — D↔H transfers (passthrough on NumPy)
- `get_memory_pool_used_mb()` — CuPy memory pool stats or None

**Tests:** 27/27 pass on both NumPy and CuPy paths (`tests/test_backend.py`).
The pattern is additive — existing `import cupy as cp` code is unaffected
until refactored. No runtime behavior change for current users.

**Status of bridge.py / connectivity.py / kernels.py refactor (Phase 1 part 2, 2026-05-11):**
- `sim/kernels.py` migrated: `import cupy as cp` → backend-aware import;
  all `@cp.fuse()` decorators → `@fuse()` (no-op on NumPy backend).
- `sim/connectivity.py` migrated: `import cupy as cp` + `cupyx.scipy.sparse`
  → backend-aware via `get_sparse_module()`.
- `sim/bridge.py` migrated (import block only): backend-aware `cp` / `csp`
  / `fuse` / `synchronize`. Defensive fallback preserves CuPy code path
  exactly when `sim.backend` is unavailable (e.g. partial bootstrap).
- 19 GPU-specific call sites in bridge.py (`cp.cuda.*`,
  `cp.get_default_memory_pool()`) remain unmigrated. They work on CuPy
  backend; Phase 2 of the tiering design refactors them behind
  `is_gpu_backend()` guards. Until then, constructing a SimulationBridge
  with `SIM_BACKEND=numpy` will fail at GPU-init time — that's expected
  Phase 1 scope.
- 198 lightweight CPU-only tests pass; kernel smoke (Izhikevich) verified
  on CuPy path. No regression for current users.

### Synapse tiering (2026-05-11): pathway-grained storage + activity tracking

**Status:** Phase 3 Strategies B+C SHIPPED 2026-05-11. The bridge can
mirror its per-pathway CSRs into a `TieredSynapseStore` (`sim/synapse_storage.py`)
and track per-pathway activity each simulation step. Inference still
uses the monolithic `cp_connections`; the store is observational +
foundation for Phase 4 auto-tiering. Per-pathway shards can be
exported alongside the lineage's `current.simstate.h5` for inspection
or future SSD-tiered access.

**Opt-in usage:**

```python
# In a CoreSimConfig:
cfg.enable_brain_region_framework = True   # required (pathway names)
cfg.enable_synapse_tiering = True          # opt-in
cfg.synapse_tiering_evict_idle_steps = 1000
cfg.synapse_tiering_grace_pagein_steps = 100
cfg.synapse_tiering_root = "bridges/synapse_shards/active"

# Bridge auto-initializes self.synapse_store at end of
# _initialize_simulation_data; per-step activity tracked in
# _run_one_simulation_step.

# Inspect at runtime:
print(bridge.synapse_store.stats())
# {'n_pathways': 24, 'n_in_memory': 18, 'n_on_disk': 6,
#  'n_pageins_lifetime': 12, 'n_pageouts_lifetime': 8, ...}
```

**Lineage export (Strategy C, works with or without runtime tiering):**

```python
from sim.lineage import BridgeLineage
lineage = BridgeLineage("main")
n_shards = lineage.export_shards(bridge)
# Writes <lineage>/shards/<pathway_name>.npz per pathway
```

**CLI:**
```bash
# Inspect exported shards for a lineage
python -m research.runners.bridge_lineage list-shards main
```

**Webapp endpoint:** `GET /api/synapse-tiering/{name}` returns shard
inventory + sizes per pathway. (Active after webapp restart.)

**Design:**
- Foundational design: [`docs/plans/2026-05-11-cpu-ram-ssd-tiering-design.md`](docs/plans/2026-05-11-cpu-ram-ssd-tiering-design.md)
- Bridge integration design: [`docs/plans/2026-05-11-tiering-phase3-part2-bridge-integration-design.md`](docs/plans/2026-05-11-tiering-phase3-part2-bridge-integration-design.md)
- 3-strategy incremental plan: C (export only) → B (mirror + activity tracking) → A (per-pathway compute, 3-4 weeks scope, deferred)

**Tests:** 56 across `sim.synapse_storage` + bridge integration
(`tests/test_synapse_storage.py`, `tests/test_numpy_backend_integration.py`).
All PASS, all CPU-only.

### 🎉 OPPONENCY ESCAPED — FHRR-on-bridge composer is the conversational PRODUCTION DEFAULT (2026-06-05)

**`BrainConversationalAgent` now defaults to the FHRR-on-bridge `RFPhasorComposer` (opponency-free).** The composer's
last numpy op — `onoff(bon−boff)` opponency (common-mode removal of a small signed difference of correlated channels)
— was confirmed a FUNDAMENTAL rate-coded SNR wall: 3 independent spiking mechanisms NEGATIVE (simple accumulator
0.41, NEF integrator 0.90-aggregate/0.077-unbind, bipolar/WTA 0.385), because biology removes the common mode in the
ANALOG stage BEFORE spiking — rate codes physically can't (Kandel 6e Ch 22, the retina). Fix: pivot the bound-vector
representation from the ±1 Hadamard to **spiking-phasor FHRR** (Frady-Sommer 2019 resonate-and-fire phasor neurons +
complex synapses) — unit-magnitude, info in PHASE, so there is no common mode and no small signed difference and the
opponency simply does not exist. Realized ON the bridge: new `NeuronModel.RESONATE_AND_FIRE` (complex state
Z=re+i·im reusing v/u; rotate `exp(λ+iω)`; Im zero-crossing spike = phase) + complex synaptic matvec
(`rf_set_complex_weights`, SPARSE) + a dedicated `rf_resonate_steps` fast loop + `rf_kick`/`rf_read_phases` (all
ADDITIVE/guarded protected `sim/` edits — Izhikevich/HH/AdEx byte-unchanged; the bind/unbind/bundle happen THROUGH
complex synapses, Frady-Sommer). `RFPhasorComposer` (`research/runners/rf_phasor_composer.py`) reproduces the full
capability matrix (who/what Q&A, abstention, negation/yes-no, one-attribute, recursive clauses, dialogue, generation)
multi-seed; 320-concept correctness GO (8/8/8). The agent's FULL existing suite (`tests/test_brain_conversational_agent.py`)
passes VERBATIM on the RF default — behavioral parity, no-confab moat intact, ZERO regression (29 GPU tests). Rate
composer = explicit opt-in (`composer_kind='rate'`); the separate 320-concept retrieval pipeline is untouched. The
F=3 two-attribute resonator (which the ±1 scheme provably can't do) is now available to lift the K=5 boundary
(follow-on). Findings: `2026-06-05-fhrr-production-switch-DONE.md`, `-fhrr-layer-{a,b,c}-*`,
`-B-opponency-rate-coded-SNR-wall-CONFIRMED.md`, `-FHRR-pivot-derisk.md`,
`-spiking-opponency-literature-synthesis.md`. Plan: `docs/plans/2026-06-05-full-fhrr-on-bridge-feature-plan.md`.

**Known limitation — composer is a principled idealization, not a functional cortex (2026-06-06):** the
FHRR/VSA composer is a *principled idealization* (Eliasmith Spaun / Semantic Pointer Architecture — a
serious hypothesis that cortex binds VSA-like), NOT a functional reproduction of cortex. Its binding is
a clean, exactly-invertible ALGEBRA that DEMANDS decorrelated full-precision codes (the whole whitening
requirement is downstream of this); a real cortex has LEARNED, lossy, redundant read-outs that learn to
read whatever messy code arrives. The binding OPERATIONS are already on-substrate spiking (FHRR
resonate-and-fire + complex synapses); the residual idealization is the exact-inverse algebra + the
clean-code demand. The spike-native robustness ladder (a phase-encoded handoff, b temporal integration,
c population redundancy + attractor cleanup) makes the scaffold spike-FAITHFUL; the genuine-cortical
conversion (d: learned read-outs replacing the fixed algebra) is **BENCHED** below the planned work
(cheat/shortcut removal → single-brain consolidation → capability addition + scaling). NOT labelled a
"cheat," but stay cognizant it is not functionally identical to the cortex it stands in for. Trade-off:
the algebra buys the no-confab moat + compositional reliability ~free; a learned cortex does not.
See `research/findings/2026-06-06-composer-vsa-idealization-known-limitation.md`.

### Conversational pipeline CONSOLIDATED onto the core sim (2026-06-04)

**The production conversational agent runs ON the core `SimulationBridge` (the brain), not on a
bolted-on numpy simulator.** Per the owner's directive ("the core sim IS the simulated brain;
capabilities realized through it, no bolted-on modules"), the conversational loop —
comprehend / store / recall / who-what Q&A / abstention / negation / clauses / one-attribute /
dialogue planning — was consolidated onto three interacting core-sim bridges:

- **`research/runners/core_sim_composition.py` (`CoreSimComposer`)** — role-filler VSA composition
  computed by **spiking coincidence neurons** on a real ~6400-neuron Izhikevich bridge (the ±1
  Hadamard: `bound_ON=AND(role_ON,fill_ON)+AND(role_OFF,fill_OFF)`), reused for unbind; SVO fact
  memory, who/what Q&A, abstention (the no-confab moat → `None` when no fact's agent matches),
  negation/yes-no (a bound polarity tag). Concept codes are the substrate's own (`denoise64`).
- **`research/runners/brain_conversational_agent.py` (`BrainConversationalAgent`, `BridgeParser`)** —
  the full loop: a **Hebbian-learned parser bridge** (comprehension: `(word-position × voice) → role`,
  voice-invariant — active "dog go north" and its passive frame assign the same agent) + the composer +
  recursive **clauses** + **dialogue planning** (`elaborate(topic)` via the dlPFC spiking
  content-selection Control over an association graph built from the agent's own facts).
- 10 on-brain regression tests pass: `tests/test_core_sim_composition.py` (5) +
  `tests/test_brain_conversational_agent.py` (5). All build a real bridge; they skip gracefully if
  the `denoise64` concept-code cache is absent.

**Honest residual:** the ±1 coincidence scheme cannot invertibly bind two concept codes (adj⊗noun) —
attributes use a feature-binding ATTRIBUTE role-tag: **1-attribute RESOLVES, 2-attribute is a
documented K=5-load BOUNDARY**, and the FHRR F=3 resonator stays a **numpy reference**. Vocab is the
validated probe scale (V=16); production 320-concept on the brain agent is a follow-on.

**COMPOSER CLEANUP SHORTCUT CLEARED — spiking NEF cleanup (2026-06-05):** the composer's last numpy readout
(the `np.argmax([concepts[w]·est])` nearest-concept cleanup in `unbind`/`_render_filler`) now has a validated,
fully-spiking, biology-grounded replacement: the **NEF thresholded cleanup** (Stewart-Tang-Eliasmith 2011, the
cleanup inside Spaun). Opt-in `CoreSimComposer(enable_spiking_cleanup=True)` builds a persistent cleanup bridge from
the codebook (operating point `NEF_CLEANUP_OP`: input-normalized matched filter + per-concept firing threshold placed
so off-target emits ZERO spikes + n_per=12 noise averaging) and routes the cleanup through it; **== numpy on the
capability matrix at production D=2048 multi-seed (27/27 seeds 42/43/44, no regression, NO sim/ edits).** Reached via
owner-steered deep research after 3 hand-tuned mechanisms plateaued/failed (divisive-norm 0.84, two-stage 0.91,
hand-WTA 0.13 — the last violated the Rutishauser α>1 WTA-stability condition). Key insight: a rate readout is a
LINEAR reconstructor (off-target leak caps it ~0.91); a placed threshold discretizes it to argmax parity. The grounded
agent enables it; numpy stays the fast default. Findings: `2026-06-05-composer-cleanup-NEF-GO.md` +
`-spiking-cleanup-memory-literature-synthesis.md`. The deeper **(B) memory shortcut** (the numpy-held bound fact +
numpy superposition/opponency) is the remaining full-clear piece (options: `docs/plans/2026-06-05-composer-B-substrate-held-memory-options.md`).

> **UPDATE (2026-06-20, shortcut-burndown #1):** the 2026-06-05 NEF cleanup above was opt-in on the
> `CoreSimComposer`/`rf` path; the shortcut-burndown inventory found the SHIPPED production `OneBrainComposer`
> (the `--composer onebrain` default) was STILL selecting each recalled word with a host `np.argmax` over the
> cleanup membrane (`one_brain_composer.py`), i.e. the spiking cleanup was never wired into the one-brain default.
> Burndown #1 (`69fd355d`) fixed that: the spiking Izhikevich WTA cleanup is now the DEFAULT on the OneBrain path
> (`consolidated_320` demo + `BrainConversationalAgent`), == host-argmax, moat 0-FA, no `sim/` edit, with an
> `enable_spiking_cleanup=False` escape for the numpy-CPU + test-oracle path. So the cleanup is now genuinely
> spiking in the shipped one-brain conversation, not just available opt-in.

**ONE-BRIDGE UNIFICATION COMPLETE (2026-06-04):** the three conversational regions now run as disjoint
persistent slices on ONE interacting `SimulationBridge` — `research/runners/unified_brain_bridge.py`
(`UnifiedBrainBridge`). Step 1: parser + composer share the bridge (no capability regression at
production D=2048 multi-seed; a `plastic=False` population still drifts under global Hebbian, so the
composer's fixed bind population is frozen by a per-synapse plasticity gate, `cp_plasticity_rate_gain=0`).
Step 2: the parser→composer hand-off is SYNAPTIC — comprehension routes composition in spikes via a
parser-gated transmission route (`hear_synaptic`); a transmission gate coupled to a BURSTY control needs a
working-memory LATCH to hold routing during the downstream read (comprehend→latch→compose). Step 3: the
dlPFC dialogue-planning loop (`enable_dlpfc=True`) merges at dt=1.0 — its NMDA-dependent WM latch survives
dt=1.0 (de-risked at the genuinely NMDA-dependent attractor weight 30, not the saturated 50 = AMPA
ping-pong); a per-region NMDA mask isolates NMDA to the dlPFC slice; `elaborate` reproduces the dlPFC's
validated dialogue-planning function with no regression. QUALIFIED nuance: rank-order (latency) coding
RESOLUTION is dt-bound, so at dt=1.0 equidistant direct neighbours tie and the tie-break may pick a
different-but-equally-valid associate than the dt=0.5 oracle (the GATE asserts the validated function, not
the tie-break). NO `sim/` edits anywhere in the unification (reuse-by-import). Findings:
`2026-06-04-one-bridge-unification-step1-capability.md`, `-step2-DONE.md`, `-step3-dlpfc-dt-survives.md`,
`-step3-dlpfc-MERGED.md`.

**The two standalone numpy phasor simulators are REFERENCE-only, NOT the production substrate:**
`research/runners/spiking_phasor_fhrr.py` + `resonate_fire_fhrr.py` (and the unified agents that import
them — `nested_composition_agent` / `spiking_unified_agent` / `unified_agent_*`) carry a NUMPY-REFERENCE
header and are retained only as the FHRR validation ceiling. Do not treat them as "the brain analogue."

Finding: `research/findings/2026-06-04-conversational-pipeline-consolidated-onto-core-sim.md`.
Audit: `research/findings/2026-06-04-conversational-pipeline-substrate-audit.md`.
Plan: `docs/plans/2026-06-04-consolidate-conversational-pipeline-onto-core-sim-design.md`.

### 🧠✅ NAVIGATION + CONVERSATIONAL merged onto ONE bridge — roadmap step 2 DONE (2026-06-10)

**STATUS: roadmap step 2 COMPLETE (2026-06-10).** The navigation cascade, the conversational parser, the dlPFC
dialogue planner, AND the resonate-and-fire (RF) composer now run as **disjoint neuron-index slices on ONE
`SimulationBridge` with one step loop**, capability-equivalent to the separate brains (STEP 2a + 2b both
COMPLETE, all acceptance gates GREEN — see the per-step bullets below). The remaining frontier is step 3 (the
true learned cortex), deferred to its own arc.

After navigation was fully biologized (every cognitive computation between sensation and action is a
validated neural mechanism — N1 spiking superior colliculus, N5 neural reward, N6/N8/N9 spiking selection +
disinhibition + dopamine RPE, N2/N7 defensible perception), the arc was **consolidating the navigation
brain and the conversational brain onto ONE `SimulationBridge`** (the owner's "one brain" directive). Builder:
`research/runners/nav_conv_merged_bridge.py` (`build_merged_nav_conv_bridge` + `MergedNavConvAgent`). The whole
arc was de-risked cheapest-first BEFORE any protected edit:

- **De-risk 5a (plasticity isolation) — PASS + one characterized gap.** The per-synapse plasticity gate
  (`cp_plasticity_rate_gain=0`) isolates weight UPDATES against the full navigation stressor (reward-STDP +
  the global dopamine `scope="all"` + Hebbian) — a frozen conversational slice stays byte-identical, controls
  change, a conversational read is unchanged across a navigation burst. THE ONE GAP (since CLOSED in code): the
  two global weight CLIPS were UNGATED, so a frozen weight OUTSIDE the active rule's clip bounds was moved by the
  clip. **SUBSEQUENTLY FIXED** — the Hebbian / reward / homeostatic clips are now **gated by plasticity gain**
  (`bridge.py:6673`/`6990`/`7253`: the `_active_syn`/`_active_rw`/`_active_hs` masked-clip paths clip only
  plastic synapses, so a frozen synapse keeps its weight verbatim). The original mitigation (raise `stdp_w_max` +
  `hebbian_max_weight` above the frozen conversational real-valued weight ~300) is now belt-and-suspenders; the RF
  composer's COMPLEX binding weights (`cp_rf_w_re/im`) are array-disjoint from `cp_connections` so they are IMMUNE
  regardless. Findings: `2026-06-10-unification-5a-plasticity-isolation-PASS-with-clip-caveat.md`.
- **De-risk 5b (RF vs Izhikevich) — KILL confirmed → the minimal protected edit.** RF stores its complex
  phasor in the same `v`/`u` arrays Izhikevich uses; one Izhikevich step destroys a phasor (|z| 1.0 → 16.3).
  But the composer is stateless-per-op (re-kicks each op) and stores memory in complex synapses, so the
  minimal edit is to **slice the RF ops** (not a core-step-loop dual-dispatch): `rf_kick(..., neuron_mask=)`
  + `_rf_advance_one` mask all `v`/`u` writes to the RF slice. **Default `None` = byte-identical** (18/18
  conversational tests pass verbatim incl. the no-confab moat); validated co-residence (an RF op on a masked
  slice == a standalone RF bridge exactly, the Izhikevich slice byte-isolated). **OWNER-APPROVED** for the
  strict (RF co-resident) merge. `tests/test_rf_neuron_mask_coexistence.py`. Findings:
  `2026-06-10-unification-5b-*` + `2026-06-10-unification-sliced-RF-ops-edit-byte-review.md`.
- **STEP 2a (merged bridge, RF composer external) — COMPLETE.** The framework path IS a wrapper around
  `inject_explicit_wiring` (`bridge.py:2196`), so the parser + dlPFC are appended as framework regions.
  The conversational gate (b) passes VERBATIM on the merged bridge — `tests/test_nav_conv_merged_agent.py`
  8/8 incl. the three `is None` no-confab assertions (`what_does`/`elaborate`/`describe`). The navigation gate
  (a) uses a HYBRID `run_moving_goal_episode` integration (4 additive no-op-default params + an index-based
  `finalize_conv_for_nav_gate` hook that runs AFTER the V1/SC post-init `set_pathway_weights(add_missing=True)`
  CSR rebuild — which re-sorts the data + stales gate-index maps + the Hebbian decay would erode the fixed
  perception weights; the hook handles all three by masking by index, not gate name). The **nav-on-merged
  smoke PASSES**: the merged bridge navigates AND the conversational populations stay byte-frozen in vivo
  under the live navigation reward-STDP + dopamine stressor. A `stdp_w_max=400` cheap-check confirmed the
  navigation score is byte-identical to 150 (the actor is ceiling-bound, not soft-bound — over-grows to 311 —
  but inert because the spiking WTA readout saturates). **Navigation gate (a) = PASS (GREEN_INERT):** the
  standalone-vs-merged score is BYTE-IDENTICAL (sum 2.0, per-phase `[0.496,0.504,0.496,0.504]`) at every completed
  seed (3/6; the remaining 3 cancelled by owner authorization to free the GPU for 2b — byte-identity is exact +
  mechanistically seed-independent for this inertness/null gate, so 3 byte-identical = conclusive, distinct from
  the standing 6-seed rule for variable effects). Tool: `research/runners/nav_gate2a_aggregate.py` (9 tests).
  Design: `docs/plans/2026-06-10-nav-conv-merge-implementation-design.md` +
  `docs/plans/2026-06-10-nav-episode-integration-design.md`. Findings:
  `2026-06-10-step2a-nav-gate-a-PASS-3of6-byte-identical.md`, `2026-06-10-nav-on-merged-smoke-PASS-*`.
- **STEP 2b (RF composer co-resident on the one bridge) — COMPLETE.** Via the owner-approved masked RF ops (an
  `rf` region with no `cp_connections` out-edges; the composer driven through `rf_kick(neuron_mask=rf_mask)`). Opt-in
  `MergedNavConvAgent(co_resident_composer=True)` (default off = STEP-2a byte-preserved); `MergedRFComposer`
  overrides only `_resonate` to address the rf slice. All three acceptance gates GREEN: (1) CPU bit-exactness +
  byte-isolation `tests/test_merged_rf_composer_coresident.py` 5/5 (== standalone composer to atol 1e-9; the
  co-resident Izhikevich slice byte-identical across the op); (2) the full conversational matrix co-resident at
  production D=128 `tests/test_nav_conv_step2b_coresident.py` 7/7 on GPU (incl. the `is None` no-confab moat + the
  co-residence anti-cheat); (3) nav-not-regressed-with-rf = 2.0 byte-identical (Δ=0). NO sim/ edit (beyond the
  default-off masked op). **⇒ ROADMAP STEP 2 (consolidate nav + conversation onto ONE bridge) DONE** — nav + parser
  + dlPFC + composer all on one `SimulationBridge`, capability-equivalent. HONEST SCOPE: a consolidation of EXISTING
  capabilities, not a new one; the composer's exact-inverse VSA binding stays the principled idealization (= step 3).
  Finding: `2026-06-10-step2b-rf-composer-coresident-COMPLETE.md`.

### 🧠🎉 THE REAL "one brain" — the whole who/what conversational turn FUNCTIONALLY INTEGRATED on ONE persistent bridge (2026-06-18)

**Roadmap step 2 (above) only CO-LOCATED nav+conv as disjoint slices (zero cross-talk); the owner's "everything in one
brain" goal meant FUNCTIONAL integration — the whole conversational pipeline as ONE persistent interacting spiking loop,
ops handing off as spikes through synapses, NO host round-trips between ops** (memory `project_one_brain_integrated_pipeline_and_cleanup`).
That is now BUILT + validated for the who/what conversational core. Built cheapest-first from 6 multi-seed GO de-risks
(CYCLE 172-179): 4-role phase coherence · GAP A (persistent multi-fact synapse-store to **K=32**, register-reset-safe,
zero cross-talk) · GAP B (the **parser front-end** — comprehension is synaptic, the parser's firing selects each bind,
voice-invariant) · the cue-matching scan · A3 (the integrated composer) · the agent wiring + negation + richer caps.

- **`research/runners/one_brain_composer.py` (`OneBrainComposer`)** — an `RFPhasorComposer` API-sibling holding the whole
  pipeline on ONE persistent co-resident `SimulationBridge`: a `BridgeParser` slice (Izhikevich) + resonate-and-fire work
  registers + a **persistent fact-store in complex synapses** + cleanup, all masked-co-resident. `hear(sentence,voice)`
  comprehends (the on-bridge parser) → the parser's role firing selects each word's bind → bundle → append a store block;
  `query_patient`/`query_agent`/`ask_yes_no` (yes/no/unknown via a 4th **polarity** role = negation) / `render_fact`
  (describe) / `query_chain` (multi-hop reason) all run the cue-matching scan; the **no-confab moat** abstains throughout.
- **`BrainConversationalAgent(composer_kind="onebrain")`** — the agent's `hear()` DELEGATES comprehension to the composer
  (ONE parser on the one brain). The full core interface (hear · what/who · yes-no/negation · describe · reason ·
  elaborate) == the rf reference agent == ground truth, **multi-seed**, moat intact. ADDITIVE wiring: the rf/rate default
  is byte-unchanged (regression GREEN). CI guard `tests/test_one_brain_composer_agent.py` (11 tests — see the A5
  cleanup bullet below; the richer features clauses/reconsolidation/multi-turn are now at parity too).
- **SPEED — A5 levers 1+3 DONE (onebrain is now FASTER than the rf reference): 96 ms/query vs rf 416 ms (~4.3×),
  answer-identical, CI-guarded.** The full speed arc: 2680 (reconstruct-per-block) → 605 (lever 1, the batched scan:
  read ALL K stored blocks in 3 resonate windows, composer-layer, no `sim/` edit) → 96 ms (lever 3, the
  masked-megakernel `sim/` edit, CYCLE 185). Lever 3 makes the RF resonate megakernel (`cfg.enable_rf_cudagraph`) honor
  a `neuron_mask` (it used to bail to the per-step loop with a mask, so the co-resident composer couldn't use it; the
  resonate is ~83% of a query). ADDITIVE + default-preserving (`use_mask==0` short-circuits to the byte-identical
  no-mask path); the masked writeback == the masked `_rf_advance_one` loop; `OneBrainComposer(enable_rf_cudagraph=True)`
  default (GPU-only, loop fallback). **GATE:** `tests/test_rf_megakernel.py` 4/4 (incl. the masked golden) +
  `tests/test_one_brain_composer_agent.py` 5/5 (answer-identical with the megakernel). Findings: the CYCLE 181/185
  commits + `docs/plans/2026-06-18-onebrain-A5-speed-cleanup-design.md`.
- **A5 CLEANUP — feature parity ACHIEVED at the validated scale (CYCLE 186-189, 2026-06-18); the production-default
  flip is gated only on a 320-scale run.** The richer rf-composer features are now at parity on the onebrain path, all
  multi-seed/GPU-gated, NO `sim/` edit (reuse-by-import), the no-confab moat never weakened:
  - **recursive embedded CLAUSES** (a fact whose patient is an SVO clause → a 2-level register→register unbind, the
    intermediate composite RE-KICKED as a clean unit phasor per hop == the numpy oracle's fresh-kick-per-hop). CYCLE 186.
  - **RECONSOLIDATION** (`update_on_mismatch` + `count_facts`): a correction reactivates the cued block, computes a
    PHASE-level prediction error vs an auto-calibrated labilization gate, and rewrites the fact IN PLACE (no duplicate);
    a re-statement restabilizes; a never-stored cue abstains. `_store_composite` factored into `_compose_phases` + a
    block-major `_write_block` (replace-or-append). CYCLE 187.
  - **AGENT-LEVEL validation:** a parser-agnostic `BrainConversationalAgent.parse` (the agent's own parser is None on
    the onebrain path — the composer carries the one parser) + `composer_kind` plumbed through `MultiTurnAgentV2` /
    `MultiTurnAgent`; the correction-turn (pronoun-cued 'actually it go south') + multi-turn anaphora both pass on the
    onebrain path; the default rf path is byte-unregressed (`test_reconsolidation_update` + `test_multi_turn_agent`
    18/18). CYCLE 188.
  - **production drop-in:** `OneBrainComposer(grounded_codes=...)` passes the learned-from-conversation codes through to
    the inner composer (== the rf grounded path), so onebrain uses the SAME codes the production conversation depends on;
    the two production demos (`consolidated_320_conversation_demo`, `multi_turn_conversation_demo`) get a `--composer
    {rf,onebrain}` opt-in (default rf = the oracle / numpy-CPU path). CYCLE 189.
  - CI guard `tests/test_one_brain_composer_agent.py` is now **11 tests** (core matrix/moat · negation · describe/reason ·
    batched==per-block · clause parity · agent-clause · reconsolidation parity · grounded-codes drop-in · multi-turn
    correction · multi-turn anaphora), all GREEN with the masked megakernel default-on.
  - **320-SCALE GO 3/3 → the PRODUCTION-DEFAULT FLIP DONE (CYCLE 190, 2026-06-18).** The consolidated-320 demo on the
    stream-learned cortex codes, `--composer onebrain` (onebrain at V=320 ≈ 54K neurons), is **3/3 GO** (seeds
    42/43/44): recall 1.00, abstain 1.00 with **0 false-accepts** (the no-confab moat holds at 320 concepts), yes/no,
    neural-ordered describe, on-topic elaborate — the WHOLE conversational turn on ONE spiking brain, on the codes it
    LEARNED FROM CONVERSATION. So **`consolidated_320_conversation_demo` now defaults to `--composer onebrain`** (the
    flagship production conversation is fully-spiking-one-brain; needs `SIM_BACKEND=cupy`); **rf is retained as the
    TEST ORACLE + the numpy-CPU path** (`--composer rf`). NOT flipped (deliberate, safe): the library constructor
    defaults (`BrainConversationalAgent`/`MultiTurnAgent` `composer_kind="rf"`) + the CPU transcript demo — flipping
    those would force GPU on every default agent and break numpy-CPU portability. The bind stays the exact-inverse FHRR
    idealization (the genuine learned-cortex bind = the separate step-3 frontier). Optional lower-priority follow-ons
    (NOT on the agent's critical path): attributed entities (adj+noun — the parser feeds flat SVO); A4 (fully-spiking
    WTA selection — the host argmax read-out is brain-based-compliant). Finding:
    `2026-06-18-onebrain-320-scale-production-GO.md`.
- Findings: `2026-06-18-one-brain-{multirole-coherence,multifact-store-GAP-A,parser-frontend,composer-A3,agent-wired}-GO.md`,
  `2026-06-18-production-one-brain-composer-scoping.md`, `2026-06-18-onebrain-gapB-parser-frontend-scoping.md`; the A5
  cleanup arc is logged in `research/findings/AUTONOMOUS_STATE.md` (CYCLE 186-189).

- **Step 3 (true cortex) — DE-RISKED to a FORK (2026-06-11); flat-cortex (A) no-confab moat validated.** The
  arc to replace the composer's exact-inverse vector-symbolic-algebra (Fourier Holographic Reduced Representation,
  "FHRR") idealization with a learned spiking-cortical binder was run to ground cheap-first. **Core finding:** the
  brain's own concept codes are CORRELATED (carry semantic similarity), and **four mechanistically-distinct
  brain-based mechanisms FAILED to decorrelate them on the point-neuron substrate** — vanilla Hopfield
  (common-mode collapse), Storkey local covariance (locality wall: only a NON-local matrix inverse removes the
  common mode), spiking dentate-gyrus (sub-reproducible read), and a fixed random expansion / Marr-Albus granule
  recoding (the common mode survives the linear expansion; threshold units flip under realistic noise). All four
  converge on the **documented Mikulasch-Priesemann point-neuron limit: decorrelation/whitening is an ANALOG /
  pre-spike (dendritic) computation a point-neuron substrate fundamentally cannot do** (the project's prior
  conversational whitening blocker, "Standing practice" above). Conversely, on DECORRELATED codes everything
  works: the distributed attractor cleanup recovers 1.000, AND a LEARNED binder generalizes SYSTEMATICALLY to
  never-seen role-filler combinations (Fodor-Pylyshyn held-out test, held-out=1.000=train, 3 seeds, leakage-
  asserted, vs memorization-floor 0.000). **⇒ THE FORK (owner decision, `docs/plans/2026-06-11-cortex-build-plan-decorrelate-then-bind.md`):
  (A) a semantically-FLAT cortex** (generated decorrelated codes + the validated binder + cleanup + no-confab
  gate) is **achievable now** and already passes the full conversational matrix at V=320, but cannot generalize
  across similar concepts; **(B) a semantically-STRUCTURED cortex** (preserve the correlated semantic codes →
  generalization) needs the **deferred dendritic-substrate rewrite** (months-scale, Mikulasch-Priesemann-mandated)
  — the path to a proper, biology-translatable brain analogue that generalizes. **Flat-cortex (A)'s last
  brain-based gap closed:** the no-confab abstention moat (currently a host check) now has a VALIDATED neural
  replacement — the learned Bogacz-Brown familiarity gate matches the host abstention decision at V=320 multi-seed
  (agreement 168/168 every seed, **zero moat-breaches**, zero abstention-floor false-accepts; validated ALONGSIDE
  the host, moat NOT weakened). Findings: `2026-06-11-cortex-{storkey-ca3,dg-ratekwta,fixed-expansion-decorrelation}-*.md`,
  `2026-06-11-cortex-sparse-attractor-poscontrol-GO.md`, `2026-06-11-cortex-learned-binder-systematicity-NEGATIVE-ON-CORRELATED.md`,
  `2026-06-11-familiarity-gate-v320-GO.md`, `2026-06-11-cortex-core-learned-binder-research.md`. The (B)
  dendritic rewrite remains the deepest/highest-variance open problem and a deliberate owner call.

- **UPDATE (2026-06-15) — the GENERALIZING learned cortex is achievable WITHOUT the (B) dendritic rewrite,
  and is REALIZED on the spiking substrate, learned from the conversation stream.** The fork's (B) framing
  ("decorrelate the correlated codes → needs the dendritic rewrite") was superseded by the CYCLE-88 reframe:
  the off-diagonal decorrelation was a **red herring**. A generalizing cortex needs **feedforward LOCAL
  normalization** (PPMI = log + per-hub + per-concept mean-subtraction + threshold, all local ops), NOT
  cross-neuron decorrelation (which would *destroy* generalization). PPMI codes reach host (+0.518) AND
  generalize (held-out 0.86), land in the binding sweet spot, and pass the full who/what + no-confab pipeline
  (CYCLE 88-90, numpy). The biology-faithful **online STREAM** version — a cortex that hears the corpus
  word-by-word (online Hebbian co-occurrence + running-frequency, NO preprocessing, NO global matrix) —
  reaches the target (CYCLE 94, +0.513). And it is now **realized ON THE REAL SPIKING SUBSTRATE** (CYCLE
  95-96): rate-Hebbian co-occurrence learning (6-seed `corr(M,C) +0.686`; STDP is the WRONG rule — measured
  656k events / 0 weight change at `delta_t≈0`, because symmetric co-occurrence has no pre→post order) +
  the **population code** (lifts the single-neuron read-out from 47% → **100-108%** of host-ref, the
  documented rate-code-wall lift) + the full conversation on the **stream-learned** codes (3-seed who/what
  recall **1.00**; no-confab moat **0.96** — 1 tail false-accept on the lowest-fidelity seed = the
  code-fidelity cost, NOT a moat-mechanism weakening; the lever is more stream → wider familiarity gap,
  never a looser gate — CONFIRMED: seed 43 at 70000 windows restores the moat to abstain 1.00 / 0
  false-accepts). HONEST SCOPE: validated at 64 concepts; the on-bridge absolute fidelity is
  window-budget-bounded (a wall-clock cap, not a substrate limit — `corr(M,C) 0.885` shows faithful
  learning). The **320-concept stream-scaling** (needs a corpus-grounded 320-word taxonomy) and the
  **on-bridge log-domain normalization CIRCUIT** (the read-out double-centring is currently a host-side
  scaffold; CYCLE 93b builds it as per-concept feedforward inhibition + per-hub adaptation, POST-f-I) are
  the remaining build. ⇒ a generalizing, biology-faithful, **learned-from-conversation** cortex on the
  point-neuron substrate (with population coding); the months-scale dendritic rewrite is NOT required for
  this generalizing cortex. Finding:
  `research/findings/2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md` (+ the CYCLE
  88-94 PPMI/stream findings: `2026-06-15-off-diagonal-red-herring-ppmi-local-normalization-reaches-host.md`,
  `2026-06-15-biology-faithful-online-stream-cortex-reaches-target.md`).

- **UPDATE (2026-06-16) — biologization sweep COMPLETE + the "learned bind" capability map is SETTLED.**
  (1) The conversational pipeline's four cognitive pieces are biologized/de-risked: the no-confab **moat**
  (learned Bogacz-Brown familiarity gate), **cleanup** (spiking NEF, 0.96), the **binding operation** (±1
  coincidence on the learned codes, 0.92), and **read-out normalization** (neural spike-frequency adaptation +
  feedforward inhibition = 96% of host). Finding: `2026-06-16-biologization-sweep-conversational-pipeline.md`.
  (2) The **learned-bind** frontier (replace the fixed FHRR algebra with a cortex that LEARNS to bind) now has
  a complete, multi-seed capability map: a learned role-filler bind generalizes **single-attribute** bindings
  and is validated on **real LIF spikes** (on-bridge held-out 0.833 = 100% of the numpy reference), but
  multi-attribute **bundling** (a fact = a superposition of bindings) is **not learnable from scratch** on the
  point-neuron substrate — additive has no inverse (0.193), a learned *linear* inverse cannot be a reciprocal
  (0.056, breaks even single-attribute), while a **fixed ±1 self-inverse bind bundles 0.989** on the same
  harness (positive control). ⇒ the conversational bind = **learned representations** (codes + single-attribute
  binding, both substrate-validated) flowing through a **fixed, biology-grounded coincidence/multiplicative
  binding primitive** (= the production composer binding the learned codes; binding-by-coincidence /
  dendritic-multiplication is a STRUCTURAL neural primitive — not a host shortcut, and not learnable from
  scratch on point neurons). Finding:
  `2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE.md`.
  (3) **Sentence GENERATION de-templating (CYCLE 104-105):** the last conversational-output host shortcut — the
  word-ordering f-string `f"{agent} {action} {patient}"` — is now de-risked + wired to a NEURAL mechanism, opt-in.
  Deep-research (controller-verified) re-framed it as serial-order PRODUCTION (a prior closed-loop HVC generator
  failed only because its self-comprehension JUDGE couldn't read order; the fix = the stored fact as an external
  order-teacher). Cheap-first de-risk GO both phases, 6/6 seeds: a **rate-coded competitive-queuing serial-order
  generator** (Grossberg/Bullock-Rhodes; catalog G.07/H.19) — the frame's primacy gradient = graded current →
  per-pool spiking RATE ranking = the emission order — beats the permuted-order + no-learning anti-cheat controls.
  Wired into `BrainConversationalAgent(enable_neural_render=True)` (default OFF, byte-identical): `describe()`'s
  word order is now produced by the spiking CQ read-out, NOT a host literal, with the no-confab moat preserved
  (`test_neural_render_describe` GPU GATE PASS). HONEST SCOPE: the SVO frame's ordering is neural (= what the
  f-string did); the remaining host orders (embedded-clause render, adjective-noun, dialogue replies) and
  MULTI-FRAME order-learning (different orders per frame = real syntax) are bounded follow-ons. Findings:
  `2026-06-16-sentence-generation-biologization-deep-research.md`, `2026-06-16-sentence-generation-serial-order-cheap-first-GO.md`.
  Runners: `neural_serial_order_renderer.py`, `_phaseB_serial_order_{cq,spiking}_derisk.py`.

- **UPDATE (2026-06-17) — the conversational arc is COMPREHENSIVELY COMPLETE: consolidation + multi-hop reasoning
  (production) + multi-turn dialogue (production), all reuse-by-import, NO `sim/` edit, the no-confab moat never
  weakened.** (1) **Consolidation GO** — the production conversational agent (`consolidated_320_conversation_demo.py`,
  `BrainConversationalAgent` + `RFPhasorComposer`) now converses end-to-end on the **320 codes the cortex LEARNED
  FROM CONVERSATION** (the fully-brain-based stream cortex): recall 1.00, 0 false-accepts, yes/no, neural-ordered
  describe, dialogue-plan — 3-seed host read-out + 2-seed fully-brain-based neural read-out (3rd streaming). The
  loop closes: learn word meanings by listening → converse using them. `tests/test_consolidated_320_conversation.py`.
  Finding: `2026-06-17-consolidated-320-production-conversation-GO.md`. (2) **Multi-hop reasoning = PRODUCTION**
  (`RFPhasorComposer.query_chain` / `BrainConversationalAgent.reason_chain`): the role-structured pointer-chase
  iterates the validated `query_patient` (match agent+action, read patient, abstain on miss → no-confab moat at
  EVERY hop), handles MIXED-relation chains. De-risked **unanimous 3-seed × 3-D GO** (2-hop chase 1.00 vs spreading
  floor ~0.08, permuted 0.00, lesion 0.00, moat intact, holds through 4 hops — the cleanup re-discretizes between
  hops so error doesn't compound) with all 5 anti-cheats foregrounding the 2026-05-14 transitive-inference
  RETRACTION. `tests/test_multihop_query_chain.py`. Findings: `2026-06-17-multihop-reasoning-multiturn-dialogue-scoping.md`,
  `-multihop-query-chain-GO.md`. (3) **Multi-turn dialogue = PRODUCTION** (`multi_turn_agent.py` `MultiTurnAgent` +
  `multi_turn_conversation_demo.py`): a persistent `SpikingLoopContextBuffer` holds discourse referents across
  turns → a turn-2 pronoun ("it") resolves to the held concept (de-risked GO 3-seed: reset/lesion break it,
  empty-WM abstains) + a multi-hop chain's intermediate is carried in the SAME loop. `tests/test_multi_turn_agent.py`.
  Finding: `2026-06-17-multiturn-anaphora-derisk-GO.md`. (4) **Two honest negatives that sharpen the map:** recall
  errors under noise are NEAR-random not within-category (the codes' category margin is thin, swamped by noise —
  `2026-06-17-within-category-error-signature-NEGATIVE.md`, incl. a same-session self-correction of an overstated
  mechanism); and multi-REFERENT disambiguation (which of several held referents a bare pronoun binds) needs
  **winner-take-all biased-competition inhibition** between referent attractors — NOT recency, NOT a salience
  boost (two converging NEGATIVEs, `2026-06-17-multireferent-disambiguation-NEGATIVE.md`) — the precise, specified
  next mechanism whenever multi-referent dialogue is prioritized. **⇒ the full conversational stack on the
  validated substrate: parse · store · recall · abstain · negate/yes-no · generate (neural word order) ·
  dialogue-plan · learn-from-conversation · multi-hop-reason · multi-turn-anaphora.**

`SIM_BACKEND=cupy` (GPU) is required for the merged-bridge runs (numpy is a tiny-smoke / CI path only).

### 🧠🔗 Cross-region "one brain" FUNCTIONAL interaction + step-3 COMPOSE-PERCEIVED-CONTENT de-risked (2026-06-16)

**Roadmap step 2 merged nav + conversation onto one bridge but they were CO-LOCATED, not interacting** (owner
challenge [[project_one_brain_substrate_vs_functional]]). The cross-region SYNAPTIC interaction (the real "one
brain") is now BUILT both directions, all milestones 6-seed GO + controller-verified
(`docs/plans/2026-06-10-functional-integration-one-brain-design.md`):
- **(A) LANGUAGE→ACTION — 6-seed GO** (`spoken_instruction_nav.py`, `2026-06-10-spoken-instruction-nav-GO.md`): the
  parser's FIRING opens a synaptic `command_route` gate → the learned word→action route steers the nav body; the
  spoken command is the only goal ⇒ the route is load-bearing + lesion-confirmed.
- **(B) PERCEPTION→MEMORY — 6-seed GO** (`navigate_to_see_then_answer.py`,
  `2026-06-16-navigate-to-see-then-answer.md`): the **navigate-to-see-then-answer** behavioral task — the agent
  navigates a gridworld (the BG cascade selecting each move NEURALLY), PERCEIVES objects rendered into `cortex_it`
  live in-episode, engram-tags them, and afterward RECALLS what it saw via neural reactivation through a TRAINED
  `cortex_it→language_output` route. COUPLED recall 3/3 every seed, LESION 0/3, isolated controls collapse,
  scramble specificity tracks the layout, provenance clean. HONEST SCOPE: this is RECALL, not composition.

**Step-3 = COMPOSE perceived content (dissolve the rate-vs-phasor wall) — COMPREHENSIVELY DE-RISKED, GO**
(`2026-06-16-step3-live-cortex-grounded-compose-cheap-first.md`). The (B) recall can say "I saw the apple" but
cannot algebraically bind a perceived apple into a NEW fact (the navigation perception is a RATE code; the composer
is a PHASOR code). The fix (per the controller-verified scoping `2026-06-16-step3-compose-perceived-content-scoping.md`)
is **shared grounded codes**: a fixed complex projection maps a LIVE `cortex_it` spiking firing-rate vector into a
unit phasor, so the percept enters the validated bind/bundle/unbind/cleanup algebra. Results (all this session):
- **cheap-first GO** (`_step3_live_cortex_grounded_compose_probe.py`, 3 seeds, CPU): live rate → grounded phasor →
  compose; held-out (never-composed) facts recover **1.000** vs a recall baseline's **0.500** memorization floor
  (compose GENERALIZES; a lookup does not).
- **scaled 6-seed GO** (`_step3_..._scale.py`, GPU): holds to 32 objects (chance 0.031, only 4 active neurons) —
  clean 1.000, corrupt 0.92. NOT a small-vocab artifact.
- **production-composer drop-in, 6-seed GO @ D=2048** (`_step3_grounded_codes_production_composer_derisk.py`, GPU):
  the SAME live-perception grounded codes drop into the deployed `RFPhasorComposer` (real 3-way SVO `store`/`query`
  + the no-confab moat) — recall 6/6, moat-abstain 3/3, parity-vs-random every seed. Closes the composer's
  documented "producing *meaningful* grounded codes is the open problem" boundary FOR PERCEIVED OBJECTS.
- **correlation boundary mapped** (`_step3_correlated_percept_boundary.py`): the compose algebra TOLERATES code
  correlation up to code-sim **≈0.98** (the role-binding decorrelates the cross-terms). CAVEAT recorded: this is
  compose-ROBUSTNESS to correlation, **NOT** generalization-across-similar-concepts (the separate dendritic/PPMI
  job; "decorrelation is a red herring", CYCLE 88). "Algebra tolerates correlation" ≠ "correlation buys
  generalization."

⇒ the rate-vs-phasor wall is **dissolved for perceived-object facts** via shared grounded codes, on the live
spiking substrate, drop-in to the production composer, moat intact.

**🎯 INTEGRATION BUILD DONE — "the agent COMPOSES what it perceives" on ONE brain, 6-seed GO (2026-06-16).**
`research/runners/navigate_to_compose_then_answer.py` (finding `2026-06-16-navigate-to-compose-then-answer.md`):
a LIVE merged nav+conv bridge episode where the agent NAVIGATES (the BG cascade selects each move neurally),
PERCEIVES + GROUNDS each encountered object IN-EPISODE (`composer.concepts[o] = angle(M @ live_cortex_it_rate)`),
COMPOSES a novel held-out perceived-object fact on the co-resident `rf` slice, then answers a who/what query +
ABSTAINS on unstored. 6 seeds (42/43/44/100/101/102), GPU: held-out compose **1.000 ≫ mem-floor 0.444** (chance
0.250) every seed, no-confab moat abstains 6/6 + a stored fact retrieves 6/6, **LESION (grounding severed) collapses
the compose** (→ 0.167/0.000), ISO-perception (no body) grounds 0, byte-identity holds. This upgrades the (B)
navigate-to-see RECALL milestone to **COMPOSE** on one bridge (nav cascade + parser + dlPFC + `rf` composer +
perception all co-resident). NO `sim/` edit anywhere in the step-3 arc — two additive default-False RUNNER builder
kwargs (`co_resident_perception`, `enable_spiking_wta_readout`); regression `test_merged_rf_composer_coresident`
5/5 + `test_nav_conv_step2b_coresident` 7/7. HONEST SCOPE: flat-distinct OBJECT facts via the FIXED FHRR algebra
(not a learned bind); NOT generalization across SIMILAR concepts (the separate frontier below).

**🧠⚡ The merged "one brain" nav action-decision is now FULLY-SPIKING by DEFAULT (2026-06-19, roadmap #4 default-on).**
Per the owner's brain-based-purity directive, `run_moving_goal_episode`'s LIBRARY defaults are flipped to the
validated spiking config — `readout_source="spiking_wta"`, `sel_recurrent_weight=0.3`, `n_sel_per_action=n_commit_per_action=40`,
`urgency_max_pA=180.0` — so the action EMERGES from the spiking competition (Wang-2002 accumulator + Lo-Wang
commit-burst threshold-crossing), the host Python argmax RETIRED. Validated 6-seed grid-32/1800 at **1.16× host
(within the 25% deploy bar), 100% commit-burst** (zero argmax fallback) — down from the CYCLE-216 ~1.7× boundary via
two levers (Usher-McClelland accumulator LEAK + finite-size-noise N-scaling; the ~16% residual = the irreducible
commit-timing/finite-size floor, the honest BRAIN-BASED-ONLY deliverable). **The CLI `--readout-source` default stays
`"motor"`** so every documented standalone benchmark reproduces unchanged; `motor`/`thal` = the opt-in host-argmax
ORACLE (the tuned levers are inert under them). NO `sim/` edit (runner-only default flip); the spiking read-out is
array-disjoint from the parser/composer so the conversational no-confab moat is preserved by construction
(`test_nav_conv_merged_agent` 8/8 + `test_nav_conv_step2b_coresident` 7/7 pass with the new default). #4 was the one
cleanly-engineering-closable boundary (#5 place-code + #3 cue-shift = honest substrate/op-point boundaries). Finding:
`research/findings/2026-06-19-spiking-decision-default-on-GO.md`.

**🎯 GENERALIZATION across SIMILAR concepts — DE-RISKED END-TO-END on the point-neuron substrate (2026-06-16),
dendritic NOT required.** The honest open boundary of the compose-perceived arc (it uses flat-distinct codes, so it
can't treat "dog"/"cat" as related) is now de-risked via **cross-modal Hebbian unification** (perception inherits
the conversation PPMI cortex's generalizing codes). Scoping `2026-06-16-generalization-frontier-scoping.md`
(deep-research, controller-verified) → three GO de-risks, all multi-seed, all anti-cheated, NO `sim/` edit:
- **cheap-first GO** (`_genfrontier_crossmodal_unify_derisk.py`, `2026-06-16-generalization-crossmodal-unify-cheap-first.md`):
  cross-modal convergence transfers the word cortex's category-generalization to perception — held-out (never-
  converged) concepts land in their correct category 1.00 (chance 0.25) — but ONLY with similarity-structured
  perception input (**Option B is the PREREQUISITE**: flat-distinct → chance). Category-derangement control
  collapses; moat survives.
- **Option B GO** (`_genfrontier_optionB_visual_similarity_derisk.py`, `2026-06-16-generalization-optionB-visual-similarity.md`):
  legitimate sensory rendering (object shapes with shared visual features → the existing Gabor/V1 bank
  `sim.visual_cortex.build_v1_simple_weights`) produces similarity-structured perception codes — within-cat 0.86 vs
  between-cat 0.08 (flat baseline 0.0), **RSA pixel-provenance r=0.99** (label-free; the structure is VISUAL, not
  injected). ⇒ NO learned projection needed; the visual hierarchy supplies the structure.
- **on-substrate A GO** (`_genfrontier_onsubstrate_convergence_derisk.py`, `2026-06-16-generalization-onsubstrate-convergence.md`):
  the convergence is NEURAL — population-Hebbian co-activation of a structured-perception region + a concept region
  on a real `SimulationBridge` transfers category-generalization on SPIKES (held-out cat-acc 0.92, flat 0.17,
  derangement collapses, moat intact). (First read as graded population depolarization because the point-neuron
  concept assembly can't spike from perception alone — the rate-code-wall — RESOLVED by the next de-risk.)
- **graded-propagation GO** (`_genfrontier_graded_propagation_derisk.py`, `2026-06-16-generalization-graded-propagation.md`):
  the rate-code residual is RESOLVED — with **NMDA on the concept region** (the slow NMDA conductance temporally
  integrates the sparse perception drive past threshold), the converged concept assembly now **SPIKES** (146/cue,
  real `cp_firing_states`) category-correctly for a held-out NOVEL-perceived cue — spike-based cat-acc 0.92 (=the
  graded read, now neural), flat 0.25, derangement collapses, moat intact (controller-reproduced on GPU). The
  concept→readout wiring is block-diagonal FIXED (category structure is LEARNED, not smuggled). ⇒ the novel-
  perception response is now SYNAPTICALLY readable (the who/what+moat pipeline reads the concept's own spikes).
  Honest bounded follow-on: a SECOND downstream relay hop loses fidelity (read-out region's own read 0.25).
- **Decisive call confirmed + EXTENDED to perception:** the months-scale dendritic rewrite is NOT required —
  generalization needs LOCAL normalization (the conversation PPMI cortex) + similarity from shared features (the
  visual hierarchy) + Hebbian convergence, all point-neuron/feedforward. ATL convergence-zone biology
  (Patterson-Lambon Ralph hub-and-spoke; Garagnani-Pulvermüller 2018 spiking precedent).
- **⇒ the generalization MECHANISM is comprehensively de-risked (all 4 pieces GO, multi-seed, anti-cheated, NO
  `sim/` edit). THE CAPSTONE (end-to-end demonstration) — honest status:**
  - **Stage 1 (vision→concept) = GO** (`_genfrontier_capstone_vision_to_concept_derisk.py`,
    `2026-06-16-generalization-capstone-vision-to-concept.md`): a NOVEL object, perceived through the real Gabor/V1
    front end (a shape → top-K structure-preserving perception drive), makes its CONCEPT neurons SPIKE in the
    correct category (cat-acc 0.75, 3× chance, 3 seeds, flat baseline at chance, derangement collapses, moat
    intact). **Generalization from pixels to spiking concepts — DEMONSTRATED.**
  - **Stage 2 (verbalize) = CAPSTONE ACHIEVED VIA THE HYBRID (3-seed)** (`_genfrontier_capstone_verbalize_derisk.py`,
    `2026-06-16-generalization-capstone-verbalize.md`): the HYBRID (option b — the spiking concept-category keys the
    VALIDATED `RFPhasorComposer` recall + its intact moat) = **0.92 3-seed** (0.75/1.00/1.00) → a NOVEL object's
    spiking concept recalls the matched category's fact. The FULLY-SPIKING fact-tag recall (option a) is a robust
    honest BOUNDARY (cat-acc 0.17 ≈ chance, moat breach all 3 seeds — the runner correctly refused to weaken the
    moat). ⇒ "perceive novel → generalize (concept spikes) → answer (recall the matched category's fact, 0.92)"
    works on one brain (brain-based: spiking concept + validated composer recall; host only routes which concept
    spiked). Bounded optional polish: the fully-spiking version (fact-tag WTA + Bogacz-Brown gate) = the all-spiking
    ideal. Reuse-by-import, NO `sim/` edit.
  - **⇒ THE GENERALIZATION ARC IS COMPLETE + comprehensively characterized:** mechanism de-risked (4 GO) + the
    capstone demonstrated end-to-end (perceive a novel object through real vision → its concept neurons generalize +
    fire → recall a fact about the matched category → answer), all on the point-neuron substrate, NO dendritic
    rewrite, NO `sim/` edit. Generalization across similar concepts — the hallmark of a real cortex — is achieved.

### Path 3 LLM-callable memory (2026-05-11): BridgeMemory API

**Status:** Phase 3.1.5 SHIPPED 2026-05-11. The `BridgeMemory` class
in `sim/bridge_memory.py` wraps a SimulationBridge + BridgeLineage as
a key-value memory subsystem that an LLM can call via tool-use.

Design doc: [`docs/plans/2026-05-11-path3-bridge-memory-api-design.md`](docs/plans/2026-05-11-path3-bridge-memory-api-design.md)

**Why:** the strategic re-eval (Path 1/2/3) places this on the most
pragmatic path — a locally-runnable LLM (Phi-3-mini / Llama 3.2 1B /
Qwen2.5) handles language + cognition; the biology-grounded sim
becomes the **memory subsystem** distinguished by continuous learning
across sessions without catastrophic forgetting.

**Usage:**

```python
from sim.bridge_memory import BridgeMemory

mem = BridgeMemory(lineage_name="alice", mode="synonym")

# Bind facts — value must map to N/E/S/W (current 4-motor-pool arch)
mem.store("alice", "north", n_events=50)
# {"key": "alice", "value": "north", "target_action": "N",
#  "confidence": 1.5, "bound_correctly": True, "n_events_run": 50}

# Recall
results = mem.recall("alice", top_k=4)
# [{"action": "N", "value": "north", "confidence": 1.0, "rank": 1,
#   "raw_delta": 317}, ...]

# Extinction-style forgetting (Phase 3.2 real-ops, 2026-05-11)
mem.forget("alice", decay_rate=0.5)
# -> {"key": "alice", "decay_rate": 0.5, "n_active_neurons": 6,
#     "n_synapses_decayed": 60, "mean_weight_pre": 1.0,
#     "mean_weight_post": 0.5, "estimated_retention": 0.5}

# Long-term consolidation (Phase 3.2 real-ops, 2026-05-11)
# Requires hippocampus-enabled bridge (main lineage isn't; bootstrap
# `main_hippo` via research.runners.bootstrap_hippo_lineage)
mem.consolidate(n_sleep_cycles=3)
# -> hippo-enabled: {"n_sleep_cycles_run": 3, "n_swr_events_run": 600,
#                     "elapsed_seconds": 45.2, "hippocampus_enabled": True}
# -> no hippo:     {"n_sleep_cycles_run": 0, "hippocampus_enabled": False,
#                     "note": "Bridge lacks hippocampus..."}

# State
print(mem.stats())
```

**Webapp endpoint:** `GET /api/bridge-memory/{name}` returns memory
state aggregated from lineage growth events: n_bindings, n_forgets,
n_consolidations, the binding history (last 50), current_tier.
(Active after webapp restart; shipped commit `def96d8`.)

**What's Phase 3.2 (deferred):**
- Choose local LLM hosting (vLLM / llama.cpp / ollama)
- Wire BridgeMemory methods to tool-use handlers (OpenAI / Anthropic
  schema)
- 5-turn conversation smoke test
- Multi-session continuity test (Phase 3.3)

**Limitation:** today's bridge has 4 motor pools (N/E/S/W). Values
must map to these. Multi-modal arbitrary k/v bindings need a larger
arch (Phase 3.2+).

**Tests:** 18 across `sim.bridge_memory` (17 in test_bridge_memory.py
+ 1 real-bridge integration test in test_numpy_backend_integration.py).
All PASS.

### Engram-tagging API (P2, 2026-05-11): catalog D.14 / roadmap T1.C SHIPPED

**Status:** SHIPPED commit 29513ac + a3acb9c. 12/12 unit tests pass.
Persistence through save/load validated (2 integration tests skipped
pending fuller test bridge).

**Module:** `sim/bridge.py` (added 9 methods to SimulationBridge,
~200 lines including docstrings)

Tonegawa-style ensemble tagging — "Apple is a CA3 ensemble":

```python
bridge.start_engram_recording("apple")
# Drive lang_input("apple") + run bridge steps for the encoding window
for _ in range(encoding_steps):
    bridge._run_one_simulation_step()  # auto-accumulates spike counts
stats = bridge.commit_engram_tag("apple", top_k=50,
                                    region_filter=["ca3"])
# stats = {"n_tagged": 47, "n_recorded_steps": 100, "window_ms": 100.0,
#          "mean_spike_count": 1.4, ...}

# Later — causal recall by stimulating the tag:
bridge.stimulate_tag("apple", drive_pA=200.0)
# Now run more steps and observe downstream regions
```

Auto-tick wired into `_run_one_simulation_step` (zero overhead when
no active recordings).

Methods:
- `start_engram_recording(name)` — begin accumulating spike counts
- `commit_engram_tag(name, threshold_hz=5.0, top_k=None,
                      region_filter=None)` — finalize tag from
  accumulated counts. Two selection modes: top-K or threshold-Hz.
- `stimulate_tag(name, drive_pA, additive=False)` — drive
  `cp_external_input_current` at tagged indices
- `clear_tag_drive(name=None)` — zero per-tag or globally
- `list_engram_tags()` / `get_engram_tag_indices(name)` / `delete_engram_tag(name)`

Persistence: tags saved as HDF5 `engram_tags/` group in
`save_checkpoint`; restored in `load_checkpoint`. Concepts survive
between sessions, matching the project's continual-learning premise.

Validation: catalog D.14 (Tonegawa engram cells); roadmap T1.C
behavioral check is the Liu 2012 inception-of-fear paradigm (train
context A → reward, tag ensemble, drive ensemble in context B,
verify reward-conditioned behavior emerges). Liu 2012 reproduction
is downstream work; the API is the prerequisite.

### Positional context P4.1 substrate (2026-05-11): catalog D.01+D.02+D.11

**Status:** SUBSTRATE SHIPPED commit 11c7c53 + ea9e439. Multi-seed
validation pending GPU (after P1 two-concept aggregates).

`sim/text_embeddings.py` adds:
  `positional_drive_pattern(position, n_neurons=200, sparsity=0.1,
                              n_max_positions=16)` — deterministic
  sparse code per position. Same band-stride layout as
  `orthogonal_drive_pattern` for maximal separability.

`research/runners/text_minimal_isolation.py` adds:
  `enable_episodic_context` flag → adds `ec_context` region (default
  200 neurons) + `ec_context → dg` plastic pathway (gate
  `ec_context_to_dg`). When enabled, DG receives a combined
  (word, position) drive → distinct CA3 ensembles per (word,
  position) tuple.

`research/runners/validate_positional_binding.py` (Test runner for
P4.1):
  Encodes 4 (word, position) bindings (apple@pos_0/pos_2,
  alice@pos_0/pos_2) and measures pairwise CA3 ensemble cosines.
  PASS criteria:
    - Same word, different position: cos < 0.4
    - Different word, same position: cos < 0.4

After P4.1 PASS, the architecture supports word-order-dependent
meaning. Downstream P5/P6 can learn to distinguish sentences by
their (word, position) ensemble structure.

### Concept replay P3.1 (2026-05-11): catalog D.19 + T1.B SHIPPED

**Status:** SHIPPED commit d569848. 5/5 unit tests pass.

`run_concept_replay_phase(bridge, tag_names, n_replays_per_tag=20)`
added to `research/runners/consolidation_trainer.py`. During NREM,
drives each engram-tagged CA3 ensemble repeatedly so STDP at
ca3→ca1→cortex consolidates the specific concept.

Differs from existing `run_swr_replay_phase` (random sparse CA3
drives): concept replay is SELECTIVE to the day's tagged concepts.
After enough replay cycles, recall works from cortex without needing
hippo state (consolidated).

Graceful error handling: missing tag names + empty tags silently
skipped. Caller manages awake/sleep gate transitions.

P3.2 (sequence replay with 10-20× time compression) deferred until
P4 episodic encoder produces sequences worth replaying.

### Hippocampal trisynaptic loop (P1, 2026-05-11): catalog D.03+D.12+D.13 validated

**Status:** SINGLE-SEED PASS commit 9d9b8f3. Multi-seed (seeds 42,
43, 44) shows D.12 (separation) robust at 3/3; D.13 (completion)
1/3 on the absolute cos > 0.7 threshold (seed 42=0.748, seeds
43=0.676, 44=0.679). Seeds 43/44 within 3% of threshold —
autoassociator working but seed-variable. Two-concept discrimination
test (relative criterion, more biology-faithful per catalog D.13
"too much completion → confused episodes; too little → no
generalization") running 3 seeds; results pending.

**Runner:** `research/runners/validate_trisynaptic_loop.py`.

The trisynaptic architecture was ALREADY built in
`build_biological_brain_regions(enable_hippocampus_consolidation=True)`
(Phase 1.3 consolidation work). P1 validated the catalog's two
characteristic functional properties:

```bash
python -m research.runners.validate_trisynaptic_loop \
    --seed 42 --train-events 400 --ca3-recurrent-weight 5.0 \
    --direct-ca3-drive \
    --out research/findings/raw/g11_bg/trisynaptic_seed42.json
```

- **D.12 pattern separation** (Kandel pp 1357–1360): DG cosine 0.218
  from input cosine 0.800 — 58pp orthogonalization. ✅ PASS
- **D.13 pattern completion** (Kandel pp 1342, 1360–1361; Marr 1971):
  CA3 cosine 0.748 (target > 0.7). ✅ PASS

Methodology note: EC-driven test (drive lang_input, propagate
through trisynaptic chain) FAILED at all parameter combinations.
DIRECT-CA3 test (drive partial of stored CA3 ensemble directly) is
the cleaner Marr autoassociator test and PASSES at train=400 +
ca3_recurrent_weight=5.0.

See `research/findings/2026-05-11-P1-trisynaptic-loop-validation.md`.

> _Archived: **Realigned plan** (was CLAUDE.md L1499-1523) → [`docs/project-history-archive.md`](docs/project-history-archive.md); retrieve via `.venv-rag/bin/python tools/rag/rag_search.py "<q>" --corpus doc`._
> _Archived: **Concept-pool v1->v17 architecture + engram-composition saga** (was CLAUDE.md L1524-2523) → [`docs/project-history-archive.md`](docs/project-history-archive.md); retrieve via `.venv-rag/bin/python tools/rag/rag_search.py "<q>" --corpus doc`._
> _Archived: **160/320-concept G.20 sparse-distributed ensemble + 320 flat-distinct composition** (was CLAUDE.md L2524-2611) → [`docs/project-history-archive.md`](docs/project-history-archive.md); retrieve via `.venv-rag/bin/python tools/rag/rag_search.py "<q>" --corpus doc`._

> _Archived: **Path 3 Phase 3.2** (was CLAUDE.md L2613-2704) → [`docs/project-history-archive.md`](docs/project-history-archive.md); retrieve via `.venv-rag/bin/python tools/rag/rag_search.py "<q>" --corpus doc`._

### Continuous-learning workflow (2026-05-11): Bridge Lineage Manager

**Status:** SHIPPED 2026-05-11. The chat REPL now "lives" between sessions
by default. See
[`research/findings/2026-05-11-bridge-lineage-shipped.md`](research/findings/2026-05-11-bridge-lineage-shipped.md)
for the full shipping notes; design doc at
[`docs/plans/2026-05-10-bridge-lineage-design.md`](docs/plans/2026-05-10-bridge-lineage-design.md).

Persistent training state lives under `bridges/lineage/<name>/`:
`current.simstate.h5` (latest state, auto-loaded), `metadata.json`
(vocab, tier, cumulative events, accuracy_history, growth_events), and
`history/` (last 30 snapshots by default). The `BridgeLineage` class
(`sim/lineage.py`) handles atomic save (`.new` + `os.replace`),
millisecond-precision history timestamps, and schema-version migration.

**Default workflow (continuous mode):**
```bash
# Loads lineage 'main' if it exists, skips ~6-20 min training.
# Saves back on exit; previous state goes to history/.
python -m research.runners.chat_repl --mode synonym
```

**Science mode (multi-seed reproducibility):**
```bash
# Always trains from random init; does NOT touch lineage.
python -m research.runners.chat_repl --mode synonym --from-scratch --seed 42
```

**Branching for experiments:**
```bash
# Fork 'main' into a new lineage; future saves go to the fork.
python -m research.runners.chat_repl --mode synonym --fork-lineage experiment_v3
```

**Inspection / management CLI (`research/runners/bridge_lineage.py`):**
```bash
python -m research.runners.bridge_lineage list
python -m research.runners.bridge_lineage show main
python -m research.runners.bridge_lineage history main
python -m research.runners.bridge_lineage rollback main --to <snapshot_id>
python -m research.runners.bridge_lineage fork main experiment_v3
python -m research.runners.bridge_lineage prune main --keep-last 10
python -m research.runners.bridge_lineage diff main --from <snap_id> --to current
```

**Webapp endpoints (`GET /api/lineages`, `GET /api/lineages/{name}`):**
Surface the lineage data for the future Lineages tab. Endpoints are
wired + tested; frontend tab is the only remaining piece.

**Compatibility:**
- Lineage stores `mode` + arch in metadata. Loading a `tier1` lineage
  with `--mode synonym` triggers a "fallback to fresh training"
  warning — no shape-mismatch crash.
- `save_checkpoint` doesn't preserve firing thresholds / STP /
  eligibility per the CLAUDE.md gotcha above. Self-recovers in ~10ms
  of free running. Fine for inference (REPL chat); documented.
- Batch demos (`chat_demo`, `chat_synonym_demo`, `chat_speak_synonym_demo`)
  default to fresh training; opt-in to lineage via `--lineage NAME`.

**Tests:** 78 across the subsystem (21 BridgeLineage, 13 CLI, 28
chat_repl, 14 chat_demo_aggregate, 2 webapp). All PASS, all CPU-only.

> _Archived: **Recommended configuration** (was CLAUDE.md L2768-2943) → [`docs/project-history-archive.md`](docs/project-history-archive.md); retrieve via `.venv-rag/bin/python tools/rag/rag_search.py "<q>" --corpus doc`._
> _Archived: **Text I/O infrastructure** (was CLAUDE.md L2944-3252) → [`docs/project-history-archive.md`](docs/project-history-archive.md); retrieve via `.venv-rag/bin/python tools/rag/rag_search.py "<q>" --corpus doc`._
**🎯 LATEST BREAKTHROUGH 2026-05-05: G v2.5 + K v2 SCALES to 32×32 at 2.57 ± 0.11 (n=6) — 13.3% BETTER than the 16×16 baseline.**

```bash
# G v2.5 + K v2 — biology-grounded, perception only, scales to 32×32:
python -m research.runners.g11_bg_runner --moving-goal --goal-schedule multi --deterministic \
    --enable-msn-lateral-inhibition --enable-d1-d2-asymmetry --enable-striatal-pv-fsi \
    --enable-cluster-a-closed-loop --enable-cluster-e-topography \
    --enable-dlpfc-wm --enable-pfc-nmda \
    --enable-visual-cortex --visual-cortex-action-warmup-steps 600 \
    --grid-size 32 --seed N --n-steps 1800
```

**Scaling result (2026-05-05 step 3).** ⚠️ **RE-CORRECTED 2026-07-16 (the FIRST correction, written the same day, was
itself WRONG — it declared "all figures are `sum_finalQ`" one line above a figure that is a MEAN, thereby CERTIFYING
the very conflation it was written to kill. It fixed the label without re-checking the number.)** The runner prints
BOTH metrics on one line (`g11_bg_runner.py:8158-8161`): `sum_finalQ` = the SUM over the 4 goal phases of each phase's
final-quarter mean Manhattan distance; `mean_distance_overall` = the mean over all steps. **They differ ~3× at 16×16
and the two headline rows below were quoted from DIFFERENT metrics.** Recomputed from the raw artifacts:
- **32×32 (n=6): `sum_finalQ` 2.75 ± 0.17**, range 2.55–3.04. *(Its `mean_distance_overall` is **2.57 ± 0.11**, range
  2.42–2.72 — that is the number formerly quoted on this line, and it is NOT a sum. Its own per-quarter row below
  averages to 2.575, which proves it.)*
- **16×16 (n=3): `sum_finalQ` 2.97 ± 0.12** (Cluster K v2 baseline). *(Its `mean_distance_overall` is 1.06 ± 0.03.)*
- **Like-for-like on `sum_finalQ`: 32×32 is ~7.5% better than the n=3 baseline (4.3% vs n=6), and 5/6 seeds beat it
  (seed 43 = 3.04 loses).** The retracted "13.3% better / 6-of-6" subtracted a MEAN (2.57) from a SUM (2.97).
- **Variance is WIDER, not tighter** (sum: 0.17 vs 0.12; mean: 0.11 vs 0.03).
- 32×32 random walk baseline: ~21 estimated
- 36.1% of 1800 steps at goal (650 ± 5 per seed)
- Per-quarter (`mean_distance_quarters`, NOT finalQ terms): Q1 ~4.3 (exploration), Q2-Q4 ~1.7-2.3 (stable AT goal)

**What survives: the architecture holds a 4× larger grid at roughly equal `sum_finalQ`** — a real scaling result.
**What is withdrawn: "13.3% better", "6/6 seeds", "TIGHTER variance", and "unexploited capacity."** Finding:
[`2026-07-16-anchor-claim-audit-...`](research/findings/2026-07-16-anchor-claim-audit-10-defects-in-the-record-incl-my-own-correction.md). ⚠️ **CORRECTED 2026-07-16 — the "closes 4 of 5 cheats
(heuristic, (gx,gy), (x,y), beacon)" claim was FALSE and is WITHDRAWN.** This config leaves
`--heuristic-strength` at its **default 1.0** → 800 pA into `cortex_N/E/S/W` derived from **direct
`gy > y` / `gx > x` goal reads**. The flag that actually closes the heuristic is
`--cue-reflex-replaces-heuristic` (`g11_bg_runner.py:7042-7045`), and it is **absent from this run's own
recorded command** (`raw/g11_bg/k_v2_stress_16x16_seed100.cmd.json`). The claim was copied from the
2026-04-27 flagship, which DOES carry that flag (so the "NO heuristic" line further down, for THAT
config, is correct). **The 2.97/2.57 numbers stand as measured — with the heuristic ON;** the visual
pathway's independent contribution is unquantified. Finding:
[`2026-07-16-clusterKv2-NO-heuristic-claim-is-FALSE-the-flag-that-closes-it-is-absent.md`](research/findings/2026-07-16-clusterKv2-NO-heuristic-claim-is-FALSE-the-flag-that-closes-it-is-absent.md). See
[`research/findings/2026-05-05-step3-32x32-scaling-success.md`](research/findings/2026-05-05-step3-32x32-scaling-success.md)
for the smoke result and [`research/findings/2026-05-05-FINAL-autonomous-arc-synthesis.md`](research/findings/2026-05-05-FINAL-autonomous-arc-synthesis.md)
for the full autonomous arc that produced this result.

Earlier breakthrough 2026-05-01 (still valid, now superseded as flagship):
**G v2.5 + K v2 visual-only at 16×16: 2.97 ± 0.12 (n=3)** — closes 4 of
5 original cheats (heuristic, (gx,gy), (x,y), beacon). 5.2× better than
Tier 0 vanilla perception arc at 16×16 (15.47 ± 7.06). Beats the
documented 8×8 perception arc baseline (4.08 ± 0.49) on a 4× larger grid.
38% of 1800 steps spent AT the goal. See
[`research/findings/2026-05-01-cluster-k-v2-breakthrough.md`](research/findings/2026-05-01-cluster-k-v2-breakthrough.md).

> _Archived: **Superseded/earlier nav flagships part 1** (was CLAUDE.md L3306-3405) → [`docs/project-history-archive.md`](docs/project-history-archive.md); retrieve via `.venv-rag/bin/python tools/rag/rag_search.py "<q>" --corpus doc`._
`--deterministic` sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before cupy
import. Tightens seed-to-seed noise floor from ±3-5 to ±0.7. Required
to detect cluster effects below the historical noise floor. ~10-30%
slowdown.
> _Archived: **Superseded/earlier nav flagships part 2** (was CLAUDE.md L3410-3691) → [`docs/project-history-archive.md`](docs/project-history-archive.md); retrieve via `.venv-rag/bin/python tools/rag/rag_search.py "<q>" --corpus doc`._
### Research Runner Ecosystem (`research/runners/`)

Headless runners for the research-gate progression (G1 through G11). Each is invocable as `python -m research.runners.gN_runner [args]` and writes results to `research/findings/raw/gN/`.

| Runner | Purpose | Status |
|--------|---------|--------|
| `g1_runner.py`, `g1_v2_runner.py`, `g1_v3_runner.py` | Encoder-decoder roundtrip | G1 GO (v3, 71.3% test acc) |
| `g2_runner.py` | STDP local learning | NO-GO (no epoch improvement) |
| `g3_runner.py` | Persistence/checkpointing | GO |
| `g5_runner.py`, `g5_v2_runner.py`, `g5_v3_runner.py` | Sensorimotor (signed perceptron) | GO |
| `g6_runner.py` | 2D gridworld | PARTIAL (gate metric needs redesign) |
| `g8_runner.py` | (session 8 work) | — |
| `g9_runner.py` | Moving-goal RL + motor exploration | NO-GO at runner-side |
| `g11_bg_runner.py` | BG cascade + perception arc + sensed reward + curriculum | **GO 2026-04-27/28 — flagship** |
| `aggregate_seeds.py` | Cross-seed result rollup | utility |

Findings docs in `research/findings/` document each session's outcome; **negative results are real findings** and stored alongside positives. A new runner should be added whenever a new architectural variant is being tested.

## File Formats

| Format | Extension | Purpose |
|--------|-----------|---------|
| Profiles | `.json` | Human-readable simulation configuration |
| Checkpoints | `.simstate.h5` | HDF5 compressed full simulation state |
| Recordings | `.simrec.h5` | HDF5 compressed frame-by-frame data |

Directories:
- `simulation_profiles/`: Saved configuration profiles
- `simulation_checkpoints_h5/`: State checkpoints
- `simulation_recordings_h5/`: Recorded simulations

## Units

- Time: milliseconds (ms)
- Voltage: millivolts (mV)
- Current: picoamperes (pA) or microamperes/cm² (µA/cm²)
- Conductance: nanosiemens (nS) or mS/cm²
- Capacitance: picofarads (pF) or µF/cm²

## Reproducibility

All RNG sources (CuPy, NumPy, random) are seeded together for determinism. The `RuntimeState.actual_seed_used` tracks the seed used. Separate seeds exist for heterogeneity and noise (`heterogeneity_seed`, `ou_seed`).

> ### ⛔ **`actual_seed_used` DOES NOT SEED ANYTHING. Set `cfg.seed`.** (a real bug, 2026-07-17 — read this before writing a runner)
>
> **`actual_seed_used` is a REPORTING field. The bridge never reads it.** Heterogeneity is seeded from **`cfg.seed`**
> (`bridge.py:2136`): `het_seed = cfg.heterogeneity_seed if cfg.heterogeneity_seed >= 0 else cfg.seed;
> if het_seed >= 0: cp.random.seed(het_seed)`. **Both default to `-1`**, so if you never set one, **the guard never
> fires** and the per-neuron firing thresholds (`bridge.py:1508`, `cp.random.uniform`) come from the **UNSEEDED GLOBAL
> RNG** — `--seeds 42` will NOT control your substrate.
>
> ```python
> cfg = CoreSimConfig(..., seed=42, ...)   # ✅ correct — what the determinism suite does
> cfg = CoreSimConfig(); cfg.seed = 42     # ✅ also correct
> cfg = CoreSimConfig(); cfg.actual_seed_used = 42   # ⛔ SEEDS NOTHING. Different neurons every run.
> ```
>
> **This cost the deep-credit arc months of confounded results**: two fresh processes at the same seed got different
> neurons; four nets built back-to-back in ONE process differed by up to **18.4 mV** (each build advances the global
> RNG), so every FULL-vs-FROZEN comparison compared **different neurons** — a confound **~3× the effect** being
> measured (`deep_credit_share` read **+0.333 / 0.000 / −0.333** on the *same* seed). **8 of 93 runners had this bug.**
>
> **The engine is fine** — it seeds correctly the moment you pass `seed=`. **Verify, don't assume:** build twice at one
> seed and hash `cp_neuron_firing_thresholds`; identical ⇒ seeded. Pinned by
> `tests/test_determinism.py::TestSubstrateActuallySeeded`. Finding:
> [`2026-07-17-THE-SEED-NEVER-CONTROLLED-THE-SUBSTRATE-...`](research/findings/2026-07-17-THE-SEED-NEVER-CONTROLLED-THE-SUBSTRATE-the-deep-credit-arc-was-confounded-by-unseeded-neurons.md).

## GPU Memory Considerations

- Networks >100K neurons require 20GB+ VRAM
- Use `GPUConfig.memory_pool_limit_fraction` (default 0.8) to control CuPy memory pool
- Connectivity uses CSR sparse matrices to scale with actual connections, not N²

## Agent Style

See `.claude/style.md` for the recommended agent identity and communication style when working on this codebase (computational neuroscience engineer with GPU computing expertise).
