# Roadmap

This is the plain-language capability view of the simulator. It is downstream
of the [2026-08-02 project charter](docs/plans/2026-08-02-PROJECT-CHARTER-grounded-emergence-realignment.md),
the [project handoff](HANDOFF.md), and the live state in
[GAP_CLOSURE_MISSION.md](GAP_CLOSURE_MISSION.md). The persistent execution
board, when checked out, is `research/coordination/workboard.json`; it records
active lanes, resources, blockers, and exact next actions.

Short, medium, and long describe dependency horizons, not promised dates.
Status below reflects the records available at the 2026-08-04 audit.

## Purpose

Build one developing artificial mind: a single simulated brain that learns from
a body, a world, and other people. It should form memories, needs, emotions,
beliefs, and language as parts of one ongoing life. It should speak because it
has something to communicate, express the strength and source of its evidence,
and keep learning from interaction.

The target is not a text generator with brain vocabulary around it. The target
is a continuously running loop:

`perception -> internal state -> action or speech -> consequence -> learning`

The loop must operate in the same brain over time. A component that passes a
small test in isolation is useful evidence, but is not the finished ability.

## Architecture Constraints

- **One brain, one shared substrate.** Dedicated regions are allowed and
  expected, but they must be neural regions of one spiking system. They must
  communicate through modeled neural activity and synapses, not through
  separate programs that exchange cognitive answers.
- **Fully spiking in the causal path.** Between sensation and action, the
  brain must compute perception, salience, value, reward, neuromodulation,
  memory, emotion, reasoning, language, and self-monitoring with neurons,
  synapses, and their local signals.
- **A narrow host boundary.** Ordinary code may create the world, render
  sensory input, enact the body's motor output, and measure or store runs. It
  may not decide what the brain perceives, values, remembers, means, or says.
  A host-side formula is still a shortcut even when the formula is biologically
  plausible.
- **Small first, earned growth.** The system must run locally when small and
  gain neurons, connections, regions, and compute as learning earns them. It
  should not begin as a pre-allocated giant network.
- **Ownable compute.** The design target is a high-end personal machine, not a
  datacenter. Event-driven, sparse, local computation is both a biological
  constraint and the path toward future analog neuromorphic hardware.
- **Temporary scaffolds are explicit.** A scaffold is a shortcut used to make
  progress while its biological replacement is built. Every scaffold needs a
  named replacement, an owner, a removal test, and a burn-down condition in
  the scaffold ledger. It cannot quietly become the permanent faculty.

## What The Evidence Means

The project uses **banked narrow de-risk** for an experiment that reduces risk
or confirms a mechanism under stated conditions. It is not evidence that the
whole brain has the corresponding human ability. A result becomes a capability
claim only after it is integrated into the continuously running brain, survives
its controls and lesions, and is tested at the required seed coverage.

### Supported, but narrow

- **Grounded action selection:** a learned convention with two communication
  intents and two referents has a six-seed positive result. The intrinsic neural
  action selector in Gate A also has a four-seed positive result. These results
  establish small pieces of action and communication, not a self-directed
  conversational mind.
- **Delayed reward is still open:** Gate B v1, which tests whether local neural
  activity can assign delayed credit to the action that caused an outcome, is
  a no-go. Unrelated or yoked reward still creates arbitrary preferences. Gate
  B v2 adds competing spiking action-value populations and is in calibration;
  its development and held-out seeds are locked, so no promotion claim is due.
- **Source and confidence machinery:** a learned seen/heard/self pathway now
  co-resides with episodic memory, anterior prefrontal cortex, and anterior
  cingulate cortex populations. Earlier versions passed some calibration
  checks, but the latest local-competition version failed a preregistered
  no-harm control on one seed. A metadata-based safety floor and trace-based
  confidence hooks are scaffolds, not final biological honesty.
- **Replay and memory:** the record contains narrow evidence that uncued
  hippocampal replay can change cortical weights, and that a localized spiking
  replay mechanism can run beside a conversational slice. Useful
  hippocampus-independent recall and replay-driven consolidation have not
  repeated strongly enough in the current state. The next live memory build is
  selective hippocampal CA1-to-cortex target reinstatement.
- **Perception:** host top-k feature selection has been replaced by competition
  based on spike timing, and its selector and lesion controls work. Fresh
  calibration seeds still fail invariant visual-identity decoding. The next
  step is learned representation and normalization, not another selector
  threshold.
- **Curiosity and metacognition:** isolated learning-progress and confidence
  monitors have useful proxy results in the record. They do not yet show that
  curiosity develops from the brain's own history or that confidence causally
  controls speech and action across the integrated system.
- **Spiking language and local learning:** several sequence, memory, and
  spiking-forward conversion mechanisms have been de-risked at limited scale.
  A current large promotion must not be called positive until its required
  six-seed artifact exists and validates. None of these results establishes
  grounded, open-ended language generated by the brain's own state.

### Not established

- There is not yet a closed, continuous perception-to-action-to-learning loop
  in which all of these pieces work together in one developing brain.
- The existing corpus-trained language machinery is not evidence of grounded
  meaning, self-generated intent, natural conversation, or a lived internal
  world. It remains a temporary development path until grounded message
  selection and neural generation replace its shortcuts.
- A narrow positive test does not establish emotion, consciousness, selfhood,
  curiosity, agency, or a whole-brain faculty. The project has functional
  correlates and mechanisms; it does not have evidence that a person is
  present.
- Deep local credit assignment on real spikes remains an open research
  problem. Rate-level or isolated credit results, and a run that merely reaches
  a target computation, do not close the on-substrate learning requirement.

## Current Blockers

1. **Integration is the main blocker.** The project has more tested parts than
   integrated behavior. The next meaningful milestone is a small world, body,
   social interaction, and grounded reason to communicate running together.
2. **Gate B physiology and delayed credit are unresolved.** The V14 engine now
   compiles and independently verifies pinned SNr candidates, runs intact plus
   four intrinsic-current lesions, recomputes metrics from bound raw traces,
   and records provenance. Production runs write authenticated compact traces
   and stop at 101 spikes or the operational timeout. An exact 512-candidate,
   24-dimensional Sobol screen is filed. The engine materialized all 512
   independently verified candidates, executed all 2,560 GPU arm traces, and
   reduced them by strict noncompensating triage to 2 engineering passes, 101
   inconclusive candidates, and 409 failures. A strict
   aggregator reports only the five resolved subgates; NumPy/CPU confirmation
   remains authoritative. The low endpoint was silent and `UNAVAILABLE`; the
   high endpoint failed Nap-silence and Cav2.2-CV signs. AHP, HCN current-step,
   SK cohort, and Nap voltage protocols remain unavailable; delayed
   action-reward learning and its unrelated-reward control come afterward.
3. **Source monitoring has a real tradeoff.** Improving weak source margins
   must not damage already-correct source judgments. The next version needs a
   biological consistency signal rather than an exact composer metadata read.
4. **Replay consolidation is not repeatable enough.** The next mechanism must
   reinstate the correct cortical target and beat learned-target, replay-order,
   and other control explanations across locked seeds.
5. **Visual invariance is not learned yet.** The spike-latency selector is not
   enough; locally learned, stable representations must handle changes in
   position and appearance.
6. **Language is still too detached from life.** Scaling an isolated corpus
   predictor would improve surface output without solving grounded intent,
   state, source, or social consequence.
7. **The deep-credit and scaling frontier remains open.** The project must
   distinguish a narrow mechanism de-risk from a local learning rule that can
   grow useful structure on the real shared substrate.
8. **Compute is scheduled and measured under controlled conditions.** Early
   V14 performance attempts exposed repeated CuPy compilation, insufficient
   fusion, and unstable host/GPU conditions; each failed candidate remains
   banked. The prospective V3 matrix used persistent source-isolated workers,
   adjacent pairing, fixed CPU/GPU controls, and a host-heavy-work lease. It
   passed the sealed engineering gate: default-off behavior was effectively
   unchanged, the active path was faster, and direct output was about one
   quarter faster than its unfused comparison. This removes V14's performance
   blocker only. Physiology and behavior still require their own preregistered
   validation.

A failed method is a method verdict, not permission to close the capability.
Bank the method, preserve its controls and diagnosis, and choose the next
biology-based spiking method. A capability remains open until it works in the
required integrated form.

## Roadmap By Horizon

### Short term: make a small brain grounded and integrated

- Give the brain a minimal world, body, social interaction, and a reason to
  communicate. Make speech an action selected from internal state and
  expected consequence, not a free-standing text completion.
- Use the V14 Stage B production runner for batched, authenticated candidate
  screening under resolved causal subgates without opening reserved scientific
  partitions early. Run the exact filed 512-candidate Sobol manifest through
  GPU engineering screening, then confirm candidates with authoritative
  NumPy/CPU runs. AHP, HCN current-step, SK cohort, and Nap voltage gates stay
  unavailable until their source-faithful protocols are fully filed.
- Run the next replay build around selective CA1-to-cortex target
  reinstatement, with the learned-target and replay-order controls intact.
- Specify and resolve the source-monitoring no-harm tradeoff before another
  source version is promoted. Replace metadata confidence with a neural source
  consistency signal where the role requires it.
- Build learned visual invariance upstream of spike-latency selection.
- Wire only cleared mechanisms into the persistent development loop. Do not
  scale the conventional language scaffold ahead of grounded message
  selection.

The short-term acceptance test is behavioral and causal: the same brain must
perceive, change internal state, choose speech or action, receive a consequence,
and change later behavior. A collection of connected demos is not enough.

### Medium term: learn, grow, and regulate through interaction

- Close continual learning from lived interaction without catastrophic
  forgetting. A temporary teacher may act as a caregiver, but the teacher is a
  recorded scaffold that must be reduced as ordinary interaction becomes
  possible.
- Grow structure as needed through activity-dependent connections, neuron or
  region growth, pruning, homeostasis, and replay-based consolidation.
- Turn the affect core into graded internal state that changes attention,
  memory, speech, and action. A scalar label or binary mood switch is not an
  emotion claim.
- Make curiosity track learning progress and uncertainty in the brain's own
  experience, rather than rewarding novelty by a host rule.
- Make source, confidence, authorship, and uncertainty influence what the
  shared brain says or withholds. Retire host-side safety floors when their
  neural replacements are verified.
- Use new combinations, lesions, social consequences, and retention tests to
  judge the whole loop rather than collecting more isolated faculty gates.

### Long term: become fluent, deep, and efficient without changing the claim

- Reach open-ended conversation that is genuinely generated by the brain's
  grounded world model, self model, affect, memory, and goals.
- Let the system form and revise beliefs, remember sources, imagine and test
  alternatives, and keep learning after the initial caregiver period.
- Retire the remaining corpus, host-decision, hand-set-structure, and exact
  metadata scaffolds that stand between sensation and action.
- Optimize the same faithful neural mechanisms for the high-end consumer
  hardware envelope. Preserve sparse, event-driven, local computation so the
  design can eventually inform analog neuromorphic hardware.

## Research And RAG Workflow

**RAG** means retrieval-augmented generation: retrieve relevant project records
and scientific sources before proposing or writing a result. In this project,
retrieval prevents redoing refuted work; it does not replace reading the source.

Before building a mechanism:

1. Search the project's findings, plans, biology catalog, and retracted or
   refuted records. Run the local pre-build/corpus check when available.
2. Read the cited biology in depth, then check relevant external engineering,
   machine-learning, and spiking-neuroscience work. A RAG hit is a pointer;
   open and read the load-bearing passage.
3. Write a functional-role specification: what the mechanism must do for the
   whole brain, what a template could fake, and what would count as failure.
4. Produce a ranked set of biology-based, fully spiking, one-brain methods.
   Start with the cheapest rate-level or spike-level de-risk that preserves the
   necessary controls, then move to the real shared substrate.
5. Record every external claim in a structured research packet. External
   evidence may inform a gate only after explicit review and source intake; a
   packet is not automatic permission to call a result solved.

Keep the RAG index fresh on CPU and check both manifest freshness and retrieval
quality, such as labeled top-three hit rate and mean reciprocal rank. Index
maintenance is workflow support, not biological evidence.

## Experiment Engine Workflow

The experiment engine now automates a bounded Stage B screening path, not the
full research loop. It authenticates candidate packets, supports a distinct
authority policy for each candidate, executes readiness traces and four
intrinsic-current lesions, stores compact authenticated traces, binds artifacts
to scoring receipts, and strictly aggregates the five resolved subgates. The
exact 512-point Sobol candidate manifest is deterministic and filed. The GPU
batch path accelerates engineering screening; NumPy/CPU execution remains the
scientific authority.

The exact filed campaign now materializes, dispatches, persists, and triages
end to end. Its first full GPU screen completed 2,560 arm traces and nominated
Sobol candidates 284 and 404 for authority runs. It is not yet a fully
autonomous research loop: authoritative CPU confirmation, recovery across
machines, and generation of a subsequent search from confirmed results are not
integrated end to end. Unspecified biological subgates remain fail-closed
rather than being inferred from partial evidence.

1. **Plan.** Materialize the treatment, controls, lesions, anti-cheats, exact
   variables, seed partitions, expected artifacts, and resource budget.
2. **Seal.** Freeze the command/configuration and record provenance. Keep
   development and held-out seeds mechanically separate.
3. **Dry-run.** Validate the sealed handoff, arm materialization, control set,
   lesion set, receipts, and held-out gates before dispatch is allowed.
4. **Execute.** The controller, not a short-lived research agent, owns decisive
   multi-seed runs. Each seed runs as an independently identifiable process
   when parallelism is scientifically valid.
5. **Verify.** Read the runner's own verdict and raw artifact, check backend,
   seed, configuration, controls, lesions, and provenance, then use
   independent adversarial checks before calling a result positive.
6. **Record.** Append the finding and update the live state, workboard, and
   roadmap in the same cycle when a status, blocker, or next action changes.

Agents may build or audit, but they do not own long sweeps. Independent work
must run concurrently when resources permit, every lane has a next action, and
every blocker has a recovery action. The controller must not fill hardware with
duplicate, unplanned, or scientifically dependent work.

## Compute And Parallelism Rules

- Use the local RTX 3090 with 24 GB VRAM for large coupled simulations. Set
  `SIM_BACKEND=cupy` explicitly for GPU work; do not infer the backend from
  imports or process mappings.
- Use `SIM_BACKEND=numpy` for tests and tiny smoke checks. A runner's default
  may silently select CPU, so the call site must choose the backend explicitly.
- Use local CPU for tests and bounded calibration. Use `pool40`, `pool41`, and
  `pool42` mini PCs for independent CPU seeds when the dispatcher and source
  provenance checks allow it.
- Fan independent seeds out as separate OS processes rather than looping all
  seeds serially in one process. Do not parallelize arms that share mutable
  state or violate the preregistered design.
- GPU work requires the shared lease and an empty running-queue claim. Check
  lane coverage before stocking a queue: keep independent CPU lanes active and
  do not mistake a full GPU for scientific coverage.
- Keep the local model-offload service stopped during GPU experiments. Use it
  only for bounded conservative work in its isolated fallback clone when the
  lease is free. Its end-to-end edit, local commit, exact-session resume, and
  cleanup path are validated; frontier review remains mandatory.
- Long runs need per-seed or per-day checkpoints, resumable output, provenance,
  and a state-checking heartbeat. This roadmap edit does not launch experiments.

## Acceptance And Honesty Boundary

Use **GO** only for the exact test that passed, and **NO-GO** for the tested
method when its controls fail. Neither label alone means that a human faculty
or the whole mind is complete. Generalization claims normally require the six
canonical seeds 42, 43, 44, 100, 101, and 102, plus matched controls, lesions,
and adversarial verification. A gate that uses a different preregistered seed
set must be reported with that scope, not silently upgraded.

Every claim should say:

- what was tested and what was not;
- which computation was neural and which part was a temporary host scaffold;
- which controls, lesions, seeds, backend, and artifact support the result;
- whether the result is a narrow de-risk, an integrated capability, a failed
  method, or an unresolved blocker; and
- what exact evidence would permit the next promotion.

The system may report functional readings such as, "the familiarity monitor
reports this input as novel" or "the confidence signal is weak." It must not
say or imply that it feels, is conscious, has subjective experience, or has a
person inside it. The project measures functional correlates of self-modeling,
affect, memory, agency, and uncertainty. Phenomenal experience is outside what
the experiments can honestly establish.

## Short Glossary

- **Shared substrate:** the common simulated neural network on which regions
  communicate through modeled activity and synapses.
- **Fully spiking:** the causal computation between sensation and action is
  carried by spiking neurons and synapses, not host-side cognitive formulas.
- **Scaffold:** a temporary shortcut with a named biological replacement and a
  removal test.
- **De-risk:** a small experiment that tests feasibility or a mechanism; it is
  narrower than an integrated capability demonstration.
- **Held-out seed:** a reserved random initialization used only after a design
  is fixed, to test generalization without tuning on it.
- **Local credit assignment:** a neural learning rule that assigns a delayed
  consequence to the synapses and actions that caused it without a host answer
  key or nonlocal backpropagation shortcut.
- **Source monitoring:** distinguishing what was experienced, heard, inferred,
  imagined, or is uncertain about, and using that distinction in behavior.
- **Neuromodulation:** brain-wide or regional chemical-like signals that alter
  learning, attention, motivation, or plasticity in the neural model.
- **RAG:** retrieval-augmented generation; here it means retrieving and then
  reading project and scientific sources before research decisions.
- **CuPy and NumPy:** the GPU and CPU numerical backends used by the simulator.
