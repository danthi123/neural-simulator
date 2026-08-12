---
type: design
status: selected
date: 2026-08-08
mechanism: developmental-teacher-loop-corrective-acquisition-synaptic-use
lane: conversation (developmental teacher-loop)
---

# Developmental TEACHER-LOOP — first de-risk DESIGN: corrective interaction -> SYNAPTIC acquisition -> USE in the live loop

Design only (NO build this pass). The deliverable is this protocol + its teeth, committed as the gate the runner is
built against. Discipline: reuse-by-import, `SIM_BACKEND=numpy`, `cfg.seed` (not `actual_seed_used`), additive /
default-off, single-seed SMOKE -> 6-seed claim.

## 1. THE GAP — verified against the record

Today the develop-loop's "converse" is **one-way fact INJECTION**: a fact is written into the composer's knowledge
base by a host call (`RFPhasorComposer.store(agent, action, patient)` -> `self.kb.append(...)`, a NumPy list write).
The develop-loop supervisor (`develop_loop_supervisor.run_resumable`) hears a Claude-authored curriculum and stores
it the same way. Even the opt-in `enable_substrate_store` path writes the **host-computed bound composite** into a
per-fact synapse — a one-time device configuration (a store-write), not learning.

Phase-5 growth (`2026-07-01-fluid-conversation-phase5-growth-GO.md`) already showed "learn a new fact mid-conversation,
answer it, moat holds" — but its own honest scope names the two residuals this de-risk targets: **(i)** the fact is
acquired by the **brain store** (a host `store` write), **not** the brain's own plasticity; **(ii)** growth is over
**pre-allocated concept codes**. So Phase-5 is the injection we already have, dressed as growth.

Children do not learn this way. They learn through **closed-loop corrective interaction**: they PRODUCE, a teacher
CORRECTS (recasts / contrastive feedback / contingent naming), they UPDATE, and they RE-USE. The teacher-loop must
(1) let the host teacher — legitimate as the **social environment**, same status as world/sensory-render — CORRECT the
brain's OWN output or name a NEW concept; (2) drive the brain to ACQUIRE it **synaptically** (its own plasticity moves
its own weights), NOT by a host store-write; (3) have the brain then **USE** the acquired concept in a later turn of
the live loop `research/runners/_stageA_full_integration_derisk.py` (query path `comp.query_patient(agent, action)`).

**What already exists to reuse (the two halves this de-risk BRIDGES):**
- `_a1_teacher_contingent_eprop_derisk.py` (2026-08-01): a **Kuhl-style CONTINGENT teacher** supplies a corrected
  target on the brain's OWN spiking output, and the transport-free e-prop rule (`OnBridgeEpropNet` +`_train_eprop`,
  the production Izhikevich bridge) **moves the FF weights toward it** — a synaptic acquisition atom, with the
  contingency + credit-route lesions already built. It learns cue -> class LABEL; it is **not** wired to the composer
  query nor the live loop, and the acquired thing is a bare label, not a conversational fact.
- The live loop `_stageA_full_integration_derisk.py` (main `3fbe1f0e`): the `CoResidentOneBrainComposer` on ONE
  SimulationBridge, whose `query_patient(agent, action)` runs the no-confab **moat** (abstain -> `None`) then renders
  the stored patient. This is the USE site.

The genuine residual = **close the loop**: the a1 synaptic-acquisition atom must acquire a NEW conversational fact
the brain did NOT know, and the live loop's own query must then USE it — with before/after + lesion teeth that prove
it was LEARNED (plasticity), not WIRED (a host write).

## 2. THE MINIMAL FIRST DE-RISK — teach ONE new fact by correction; show acquire-by-plasticity + USE

**One taught fact.** A NEW referent `dax` (a noisy perceptual prototype the brain has never seen — a small perceptual
category, the a1 construction) and a NEW fact `dax eats grass`. `grass` is an existing patient CODE in the composer
vocab; the FACT (this cue -> this patient) is absent from `kb` and from any plastic map. (Brand-new lexeme CODE
allocation — a novel phonological/word code — is the harder dendritic/allocation frontier; declared in §6, NOT in the
smoke.) This is exactly "the dax eats grass": the words exist, the FACT is new — a child's corrective-learning atom.

**The turn protocol (host teacher = social environment; brain = acquirer):**
1. **PRODUCE / PROBE (before).** Present the `dax` percept + the query context (agent=`dax`, action=`eats`). The brain
   is asked "what does it eat?" Its acquired read-path is untrained -> low readout confidence -> **abstain** (the honest
   "I don't know that yet"). Record `answer_before = None`.
2. **CORRECT (the teacher presents input, like a sensory input).** The teacher pairs the SAME cue the brain is
   responding to with the target answer `grass` — a **contingent recast**. This target enters the brain as an ERROR
   on its OWN readout, NOT a persistent clamp (see §3): `error = softmax(readout_logits) - onehot(grass)`, gated by a
   corrective/reward DA burst. Repeated over the corrective micro-turns (noisy fresh presentations of `dax`, the a1
   `make_referent_task` draw), the brain's OWN weights move.
3. **USE (after, in the live loop).** With NO teacher present, present a FRESH `dax` draw + the query and call the
   live-loop path `comp.query_patient('dax', 'eats')`. The acquired read now clears threshold -> answers **`grass`**.
   Record `answer_after`. The stageA loop uses it exactly as it uses any fact (moat runs first, then render).

**The acquisition read-path (additive, default-off).** The a1 `OnBridgeEpropNet` is instantiated with `K = patient
vocab size` (classes are patient WORDS, not abstract labels). Input = the `dax` percept features ⊕ the query-context
features. The trained readout's argmax = the answer word; its softmax max = the confidence that gates abstention. USE =
this acquired word is returned as the answer to `query_patient` for the taught cue. **Honest seam (declared):** for the
ACQUIRED fact the abstain/answer gate is the e-prop readout's **confidence threshold**, NOT the composer's structural
`kb`-membership moat — a genuinely-learned fact has no `kb` block by construction (that would be a store-write). So the
matched-control moat check in §4 MEASURES false-accepts on untaught cues; a learned confidence gate that leaks is the
boundary the developmental engine must then close (links to `_phaseB_harden_320_learned_moat`).

## 3. THE NEURAL ACQUISITION MECHANISM — how a correction drives synaptic learning WITHOUT a host store-write

**Biology.** Language acquisition runs on **corrective feedback + reinforcement**, not fact-copying. Kuhl's social-
gating result (infants learn a phonetic contrast from a LIVE, CONTINGENT tutor but NOT from non-contingent audio/video)
is the anchor the a1 teacher already cites: **contingency is the learning signal**. Caregiver **recasts / contrastive
discourse** supply the corrective target on the child's own production (Saxton's corrective-input account of negative
evidence). The substrate that turns a contingent correction into a weight change is **dopaminergic reward/predication-
error gating cortico-striatal (three-factor) plasticity**, with fast hippocampal binding for the one/few-shot part. The
teacher's contingent recast = a **phasic DA burst** (a corrective/reward third factor) landing on the eligibility trace
of the co-active cue->answer synapses.

**Mechanism (no host store-write).** The teacher's correction enters as a **learning SIGNAL on the brain's own weights**,
two equivalent framings both already in the codebase:
- **e-prop (validated substrate rule, a1 GO):** `error = softmax(readout_logits) - onehot(target)` is projected as the
  learning signal onto the per-synapse eligibility of `OnBridgeEpropNet` (`train_batch`); the FF weights integrate it.
  The error is **not a clamp** — it VANISHES at match (softmax->onehot), so it can never become the "clamp-as-crutch"
  the 2026-06-08 teacher-correction finding warns against.
- **three-factor (more biological, DA-gated):** `bio_three_factor.update_eligibility_and_weights(...)` with `da_per_action`
  = the corrective DA burst as the third factor; eligibility = the cue×answer co-firing trace. Same signature already
  used by the develop-loop plasticity path.

**The dividing line, made mechanical.** The teacher PRESENTS corrective input (a target current / a DA burst) — like a
sensory input. The brain ACQUIRES it by moving its OWN synaptic weights (`ff_weight_moved > 0`, the a1 anti-cheat). The
answer at USE time is read out of those weights by firing the cue through them. **No `composer.store()` is called for
the taught fact** — that is the injection we are replacing. If the acquisition were a `kb.append`, that is the one-way
injection; declare it, do not sell it as learning.

## 4. THE TEETH — before/after + matched control + two lesions of the LEARNING pathway

<!--derived-->

| teeth | operation | PASS (acquired-by-plasticity) | proves |
|---|---|---|---|
| **BEFORE/AFTER** | `query_patient('dax','eats')` before vs after correction (fresh draws) | before = abstain(`None`); after = `grass` | the fact was not known, then is |
| **WEIGHTS MOVED** | `ff_weight_moved = |W_after - W_before|` | `> 1e-3` | acquisition is a weight change, not a read of a host write |
| **MATCHED CONTROL (moat)** | an UNTAUGHT cue (`dax chases ?`, and a 2nd untaught referent) after teaching | still abstain; 0 false-accepts | the update is SPECIFIC; it did not just "start answering everything" |
| **LESION-1 learning-pathway** | freeze W / `learning_gate=0` during correction; then query | NOT acquired -> still abstain | it was LEARNED, not WIRED (kills the store-write null) |
| **LESION-2 contingency** | NON-CONTINGENT teacher (target drawn at random, uncorrelated with cue) | fresh-draw held-out -> chance/abstain | the CONTINGENCY is the signal (Kuhl); not noise-memorization |
| **LESION-2b credit-route** | SHUFFLE-DFA (eligibility intact, credit mismatched to the example) | held-out -> chance/abstain | the CREDIT ROUTE carried it, not the forward reservoir |

Generalization to **fresh noisy draws** of `dax` (the a1 held-out test set) makes lesion-2 clean: a real teacher signal
generalizes; a scrambled one can only memorize noise -> chance. Attribution is asserted with `tools.lab.attributable_to`
("teacher contingency", main vs non-contingent) — the effect must be the correction, not merely two arms measured.

**GO (single fact, per seed):** `after == grass` AND `before == None` AND `ff_weight_moved > 1e-3` AND untaught cues
0-false-accept AND `frozen-W == abstain` AND `main > non_contingent + 0.15` AND `main > shuffle_dfa + 0.15`.

## 5. HOW THE BRAIN USES IT IN THE LOOP

The acquired fact is exercised through the **unchanged live-loop read** `comp.query_patient('dax','eats')` — the same
call the stageA multi-turn loop already makes (`_colored_answer` / `_colored_answer_graded`: the moat runs FIRST, then
affect colors tone WITHIN the decided band, never flipping abstain->assert). So USE = a later turn where the taught
referent is queried and the brain answers `grass` with its normal grounded+affect-colored rendering, having learned it
this session by correction. The de-risk adds ONE glue shim (additive, default-off): route the taught cue's answer
through the acquired e-prop read-path instead of the `kb` scan; byte-identical when the flag is off (the composer's
neuron firing thresholds are unchanged; the shim only appends a read path).

## 6. RISKS / honest boundaries (each is a first-class deliverable if it fires)

- **HONEST NEGATIVE (the primary expected finding), with teeth:** neural corrective acquisition may **underperform host
  injection** — need many corrective turns for one fact, or generalize worse to fresh `dax` draws, or leak the moat.
  Run the **host-injection baseline** (`composer.store('dax','eats','grass')`) as the reference arm and report
  neural-vs-injection head to head. If neural underperforms, that MAPS what the developmental engine needs (more
  corrective turns / sleep-replay consolidation / a learned abstention gate) — the deliverable, not a failure.
- **Moat trade (declared in §2):** the acquired read replaces the structural `kb` moat with a confidence-threshold
  moat; a leaky threshold on untaught cues is the boundary (-> learned-moat work). Measured by the matched control.
- **Pre-allocated CODE frontier:** the smoke teaches a new FACT over EXISTING word codes. Teaching a brand-new lexeme
  CODE (a novel word the vocab has never held) is the dendritic/allocation frontier — the next de-risk, NOT this one.
- **One-shot vs slow:** a single correction may not suffice; biology uses repeated recasts + overnight consolidation
  (speed is secondary). The protocol allows N corrective micro-turns; if it needs replay to stick, that is in scope.
- **Clamp-crutch regression:** verify the teacher signal is an ERROR that vanishes at match, not a persistent clamp
  (the 2026-06-08 warning) — asserted by checking the learning signal -> 0 as `readout -> target`.
- **Seed reality:** the substrate is seeded via `CoreSimConfig.seed`, NOT `actual_seed_used` (the reporting field
  seeds nothing) — build twice at one seed and hash `cp_neuron_firing_thresholds` to confirm before trusting arms.

## 7. Commands

<!--derived-->

These are the PROSPECTIVE run commands (the runner is not built this pass). `<RAW>` denotes the out directory
`research/findings/raw/lanes/teacherloop` (the artifacts do not exist yet — a placeholder, not a citation).

SMOKE (single seed, one taught fact, all teeth):
```
PYTHONPATH=$PWD SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 \
  python -m research.runners._teacher_loop_corrective_acquire_derisk --seeds 42 \
    --out <RAW>/teacher_loop_corrective_acquire_s42.json
```
6-SEED claim (GO needs 6/6 at 42..47):
```
PYTHONPATH=$PWD SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 \
  python -m research.runners._teacher_loop_corrective_acquire_derisk --seeds 42 43 44 45 46 47 \
    --out <RAW>/teacher_loop_corrective_acquire_6seed.json
```

**Artifacts to build against (reuse-by-import, NO `sim/` edit):** `_a1_teacher_contingent_eprop_derisk.py` +
`_onbridge_eprop_port_derisk.OnBridgeEpropNet`/`_train_eprop` (synaptic acquisition + lesions); `rf_phasor_composer.py`
`query_patient` (USE + moat); `_stageA_full_integration_derisk.py` main `3fbe1f0e` (the live loop);
`bio_three_factor.update_eligibility_and_weights` (the DA-gated variant); `tools.lab.attributable_to`.
