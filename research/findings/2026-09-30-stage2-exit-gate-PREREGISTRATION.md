---
type: finding
status: live
date: 2026-09-30
lane: integration-first (stage exits on the calendar; owner 2026-09-29, acceleration report M1 / section 6.1)
mechanism: PRE-REGISTRATION of the second developmental STAGE-EXIT attempt -- Stage 2 (CHILD) of the master roadmap, run END TO END on the INTEGRATED production brain (the live /api/brain-chat handler at production defaults, pinned revision), never on an isolated de-risk runner; five criteria (grammatical grounded conversation, inference on held-out, curiosity ask-rate >= 2x on gaps + learning-progress, discrete-emotion read-out, private->inner-speech chain) each measured at the REPLY level with the internal read-out secondary, with a null arm, a lesion arm per responsible faculty (repeated), a permuted-exposure arm, and a host-shortcut audit
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION only. Filed before any gate-seed arm was built. No result is claimed here.
runner: research/runners/stage2_exit_gate.py
artifacts:
  - research/findings/raw/_gateB_curiosity_production_verify.json
  - research/findings/raw/_spiking_mouth_recall_soak.json
  - research/runners/stage2_exit_gate.py
  - tests/test_stage2_exit_gate.py
  - docs/drafts/stage2-exit/HANDOFF.md
---

# Stage 2 (CHILD) exit gate on the integrated production brain: PRE-REGISTRATION (2026-09-30)

**Committed alone, before any gate-seed (42/43/44/100/101/102) arm is built.** Branch `research/stage2-exit-gate`.
The gate runs at a revision that contains this file BYTE-UNCHANGED plus the runner (the pin is recorded in
`docs/drafts/stage2-exit/GATE_SHA` and checked by the HANDOFF), provisioned on the pool with
`tools/pool_provision.sh --revision <sha> --isolated`; every job line carries the pin, `--prereg` (the gate-seed
guard, `tools/gate_seed_guard.py`) and `max_age_h=240` (the dispatcher's debris guard, FAILURE_LOG 2026-09-30).

**Why now (owner directives).** Stage exits are on the calendar, not on readiness: Stage 2 by 2026-11-21, "run even
if expected to fail, because a failed exit is the first measurement of the integrated brain the project has ever
had" (`docs/plans/2026-09-29-acceleration-report.md` section 6.1; CLAUDE.md integration-first + pivot dwell). The
roadmap phases: `docs/plans/2026-09-29-roadmap-to-the-primary-goal.md` Phase C's exit ("Stage 1-2 exit gates in the
integrated brain ... curiosity ask-rate >= 2x with learning progress, held-out inference"). This packet is written
2026-09-30 so the local model can queue and score it in November without judgment; it is NOT queued now (the HANDOFF
says when). Four of the five criteria are predicted to FAIL for code-level reasons named below; each predicted FAIL
names the gap it measures, and the build-lane checklist's "do not stage a run whose pre-registration already
predicts failure" is overridden by the owner's calendar rule exactly as in the Stage-1 packet.

**What this is not.** Not a de-risk (every mechanism involved has one, cited in section 2). Not a GO/NO-GO on any
mechanism. The verdict vocabulary is PASS / FAIL / UNDEFINED(INCOMPLETE) per criterion and for the stage; a FAIL is
a measurement of the integrated brain, never a stop (CLAUDE.md non-negotiable 3). Honesty boundary: every "feel",
"curious" and "on my mind" read-out below is a functional read of a spiking organ; nothing here measures or asserts
experience.

**Relation to the Stage-1 packet.** Same harness, same seeds, same arm kinds, same null / lesion-vs-null attribution
/ earned-verdict machinery and the same pool lane (`research/runners/stage1_exit_gate.py`, branch
`research/stage1-exit-gate`, prereg `2026-09-29-stage1-exit-gate-PREREGISTRATION.md`). Stage 1's criteria are not
repeated here; a Stage-2 exit is read on its own five criteria, and the stage ladder (Stage 1 before Stage 2) is the
roadmap's reading of the two results together, not this gate's job.

## 1. The five criteria, operationalized

The roadmap's Stage-2 exit line (section 4, STAGE 2): "grammatical grounded conversation; inference on held-out;
curiosity ask-rate >=2x on gaps + learning-progress; discrete-emotion read-out; private->inner-speech chain". Each
becomes one probe GROUP on the live handler (one session = one fresh `ChatBrain`, one arm = one process), scored by
the registered aggregate (`--aggregate`) and nothing else.

| criterion | group | the brain EXPERIENCES (teacher channel = chat turns) | held-out test (reply-level primary field) | frozen bar |
|---|---|---|---|---|
| C1 grammatical grounded conversation | `conv` | four teach turns: two transitive facts about a new animal (`the fox chases the hare`, `the fox eats the berry`), one PREPOSITIONAL fact (`the deer sleeps in the meadow`), one PASSIVE-voice fact (`the mole is hunted by the owl`) | `what does the fox chase` (a well-formed frame naming the fact), `tell me more` (a grounded elaboration with new content), `where does the deer sleep` (the frame keeps its preposition), `what does the owl hunt` -> mole (the passive comprehended), `what does the mole hunt` -> abstain, a built-in fact, an OOV subject -> abstain | every scored reply grammatical (section 4.1's fixed checker) and grounded, the passive comprehended, the moats held, on 6/6 seeds; the spiking-Broca mouth lesion changes the transitive reply surface 6/6 and reproduces; the role-permuted teaching is tracked 6/6 |
| C2 inference on held-out | `infer` | `the heron hunts the frog`, `the frog eats the gnat` (two premises; the conclusion is never told); `the crow chases the wren`, `the wren chases the moth` (a same-relation chain for the yes/no probe) | `what does the heron's prey eat` -> gnat, framed as DERIVED ("I derived this from: ..."); both premises recalled directly; `what does the heron's prey chase` -> abstain (hop 2 has no fact); `does crow chase moth` reported | the conclusion named and framed as derived on 6/6; the chain-route lesion removes it 6/6 and reproduces; permuted premises (heron hunts gnat; gnat eats frog) give `frog` 6/6 |
| C3 curiosity ask-rate >= 2x on gaps + learning-progress | `ask` | three facts about set A (`the yak eats the grass`, `the moose eats the leaf`, `the goat eats the hay`); set B (eel, crab, newt) is never mentioned | six abstain probes (`what does the <X> chase`): the NOVEL-topic gaps (B) must carry the follow-up question, the TAUGHT-topic moats (A) must not; a learnable gap (`ibex`) asked, answered by the teacher, then recalled and not asked; an unlearnable gap (`toad`) asked once, never answered, probed four times with a built-in recall between probes | ask-rate(gaps) >= 2/3 AND >= 2x ask-rate(moats), the learnable gap closes, the unlearnable gap is NOT asked on its 4th probe, all gap probes abstain, on 6/6; the curiosity-drive lesion removes the asks 6/6, the graded-novelty lesion collapses the discrimination 6/6, each reproduced; the permuted teaching (set B taught, A novel) is tracked 6/6 |
| C4 discrete-emotion read-out | `emo` | four self-relevant situations, each twice, each followed by `how do you feel`: joy (`you win a gift and you are happy and glad and delighted`), fear (`a wolf is attacking you and you are scared and afraid`), anger (`someone cheats you and you are angry and mad and furious`), sadness (`you are alone and lonely and sad and you cry`) | the feel reply NAMES a discrete category (section 4.4's fixed lexicon) that matches the situation, four distinct categories | all four named and correct on 6/6; the ladder lesion changes every feel reply 6/6 and reproduces; the affect-word-swapped situations (joy<->sadness, fear<->anger) are tracked 6/6 |
| C5 private->inner-speech chain | `inner` | a three-link chain (`the lynx chases the vole`, `the vole chases the beetle`, `the beetle chases the mite`) | PRIVATE: `say something` -> the brain voices an internally selected STORED fact in its self-initiated frame; INNER: `what does the lynx chase all the way` -> mite (the terminal, never told, reached by the workspace's own re-entrant chase with its hop count decided by its own ignition), `what does the vole chase all the way` -> mite, the single hop `what does the lynx chase` -> vole, the chain tail -> abstain | all on 6/6; the workspace-recurrence lesion removes the terminal 6/6 and the self-initiation lesion removes the voiced thought 6/6, each reproduced; the permuted chain (lynx -> mite -> vole -> beetle) gives `beetle` 6/6 |

The exact turn tables are frozen in `research/runners/stage2_exit_gate.py` (`GROUPS`); the messages there are the
protocol. "Teacher channel" means the scripted interlocutor typing into the chat, which is the host acting as the
social environment (CLAUDE.md non-negotiable 1). No arm injects an answer, a code, a weight or a store entry: every
exposure is a chat turn the production handler receives as the owner would type it.

## 2. The record this rests on (research-first)

`bash tools/before_you_build.sh` was run 2026-09-30 (logged to the shared corpus-check log) plus five RAG queries,
one per criterion; the sources named below were read, not paraphrased from the hits. The two artifacts the C1 and
C3 predictions rest on are cited here as the record, not as results of this gate:
`research/findings/raw/_spiking_mouth_recall_soak.json` (the recall mouth's 6-seed soak: flag-off byte-identical,
load-bearing, the template-vs-spiking surface pair) and `research/findings/raw/_gateB_curiosity_production_verify.json`
(the curiosity wire-in's real-handler verify: the novel abstain asks, the familiar recall does not, the lesion collapses
the ASK-pool want).

- **C1.** The spiking Broca producer renders reply frames with learned function-word and inflection slots (EMERGE-59,
  `2026-07-03-emerge59-spiking-broca-frame-slots-GO.md`; EMERGE-61 order robustness; EMERGE-60 wired it into the
  flagship console); the RECALL surface of the production turn renders on that mouth, default-ON since 2026-08-26
  (`2026-08-26-spiking-broca-mouth-recall-surface-production-wirein-GO.md`; `spiking_mouth_recall_prod.
  recall_mouth_enabled`, `_RECALL_MOUTH_DEFAULT_ON`), but ONLY for a bounded transitive SVO (single-word roles,
  subject != object, non-copula verb); every other fact falls to the host template stub, whose surface is
  "The <S> <V>s <O>." with a naive inflection (the Stage-1 dev smoke printed `The robin flys sky.` for
  `the robin flies in the sky`). The frame inventory's self-organization is a de-risk, not a production organ
  (`2026-08-21-spiking-broca-frame-structure-self-organizes-slot-inventory-6seed-GO-slot-order-signal.md`; EMERGE-22
  pos-frame grammar). Comprehension of a non-canonical form (object-relatives, passives) is de-risked
  (`2026-07-07-emerge91-passive-comprehension-general-formtorole-GO-airtight.md`, EMERGE-90) but the live handler's
  acquisition is the host 3-content-token SVO extractor (`ChatBrain._maybe_acquire`, position-only), so a passive
  teach is stored with its roles as they appear. The comprehension monitor (D4, `2026-08-12-GateB-comprehension-
  monitor-production-chat.md`) judges transitive ASSERTIONS and is trace-only on well-formed probes (the 2026-09-29
  reply-level rescore); its lesion arm crashed in B2b and was fixed 2026-09-29 (re-measurement pending), so this gate
  reads its trace and does not lesion it. The discourse follow-up ("tell me more") elaborates the held topic with
  genuinely new content (`rich_answer_composer._is_followup`, `gather(followup=True)`).
- **C2.** Two-hop composition over the brain's own facts is wired default-ON as the possessive-chain route
  (`research/runners/compositional_chain_route.py`, `2026-08-25-reasoning-frontier-chain-routing.md`): a host regex
  detects "what does X's ROLE V", a role-noun table proposes hop 1, and the 2026-08-25 hardening pass made each hop a
  `_distinct_patients` scan of the composer's `.kb` (a host list) with the spiking `query_patient` only as the
  fallback for an unscannable composer. So on the production onebrain composer this route is host-owned end to end
  at HEAD (section 5). The derived answer is framed honestly ("I derived this from: ...", `frame_derived_answer`,
  `derived`/`derived_from` in the response). The transitive yes/no route over the spiking workspace chase
  (`webapp/reasoning_transitive_chat.py`, `2026-09-24-reasoning-transitive-chat-PREREGISTRATION.md`) is
  default-OFF (`BRAIN_TRANSITIVE_CHAT`); it is the informational arm here. Inheritance, taxonomy and induction are
  isolated de-risks (EMERGE-26/27/28, rung-3c, the 2026-08-26 inductive-coverage PARTIAL) with no live route; Stage
  1's C1 measures inheritance and is not repeated.
- **C3.** The curiosity crave-drive is wired default-ON: on an abstain, the epistemic gap drives the `from_novelty`
  neuromodulator onto a spiking ASK pool and the follow-up question fires when the pool's rate crosses its calibrated
  threshold (`2026-08-12-GateB-curiosity-followup-production-chat.md`; `curiosity_production_organ.judge`;
  reply-load-bearing 6/6 in B2b). Since 2026-09-05 the novelty fed to the judge is GRADED per topic through a
  Bogacz-Brown anti-Hebbian familiarity projector imprinted with the brain's own vocabulary (`graded_novelty_enabled`,
  `_GRADED_NOVELTY_DEFAULT_ON`; a held word reads LOW novelty, an unknown one HIGH); the Stage-1 dev smoke showed the
  reply-level consequence: `what does the bear chase` asked before the bear was taught, `what does the otter chase`
  and `what does the wolf chase` did NOT ask after those animals had been taught. The learning-progress SELECTOR
  (Oudeyer-Kaplan LP, `2026-08-07-laneB-curiosity-learning-progress-MAXIMIZING-selection-CPU-proxy-6seed-GO.md`) and
  the reward-omission VETO are NOT wired ("neither is wired", the 2026-08-12 finding's declared residual); no
  production organ tracks how often a gap was asked or whether asking produced learning. DR-1's own GO gate names
  the honesty test this measures: "NOISY-CONCEPT (unlearnable -> zero learning-progress -> STOPS asking)".
- **C4.** The production affect organ is a signed SCALAR: a bistable graded-affect ladder whose value per word is the
  DR-2 learned map and whose salience gate is the Warriner norm (`affect_production_organ.appraise_text`); the feel
  read-out prints the sign, the differential and an arousal band and nothing else (`feel_readout`; the Stage-1 C4
  read). Appraisal -> four discrete emotion attractors (SEEKING/CARE/FEAR/RAGE) with a vmPFC reappraisal gate is a
  6-seed cupy de-risk that "does not wire it" (`2026-08-13-spiking-appraisal-discrete-emotion-reappraisal-derisk.md`,
  residual 4: "Runner-only: folding the slice into build_one_brain ... is the production-integration step"). No
  production module emits an emotion category word (grep 2026-09-30 over `webapp/` and the production organs). The
  affective-ToM empathic lead (W5, default-ON) fires only on a named third party and is kept out of these turns.
- **C5.** The self-cued associative chain of thought is de-risked (Tier 2.2, `2026-06-27-tier2.2-chain-of-thought-GO.md`:
  the agent picks each next relation itself). In the live handler the one self-cued multi-hop mechanism is the GNW
  multistep re-entrant chase (`webapp/gnw_multistep_deliberation.py`, default-ON since 2026-08-19, reply-load-bearing
  6/6 in B2b): on an explicit "all the way" question the workspace re-enters its own broadcast as the next cue and
  stops when its own ignition read says the leaf is reached (`n_ignited` off `cp_firing_states`), emitting only the
  terminal; the hops are internal (the trace). The idle turn is the one turn class whose CONTENT is internally
  selected (`2026-08-18-self-initiated-utterance-wired-brain-chat-GO.md`, default-ON, reply-load-bearing 6/6 in B2b):
  "say something" makes the brain voice a stored concept and a question to itself ("Something's been on my mind --
  cat eat fish. What does cat eat?"). No route voices the chase's intermediate hops on request (a "think aloud" or
  "how did you get there" route does not exist at HEAD); the possessive route's "I derived this from" lead is the
  only spoken chain and it is host-owned (C2). Vygotsky's private->inner sequence is therefore operationalized as:
  PRIVATE = a self-directed utterance whose content the brain selected (the idle turn); INNER = a self-cued chain
  that runs silently to a conclusion the brain then speaks (the chase). The timing of the private turn is
  HTTP-triggered (the 2026-08-18 finding's declared scope); the chain's first cue is the user's question and what is
  self-cued is every later hop and the stop.

## 3. The harness (frozen)

Real `webapp.server.brain_chat` handler, called in-process by a worker subprocess exactly as the regression battery,
the load-bearing-fraction runner and the Stage-1 gate do; tiny-demo brain with the shipped LTM tier
(`BRAIN_LTM_SHIP_DEFAULT=1`, its production default made explicit; expected `source` = `tiny-demo +LTM`); stub
renderer; no LLM; numpy/CPU; one BLAS thread; `BRAIN_CHAT_SEED=<seed>`; `PYTHONHASHSEED=0`; the request's `rich`
field OMITTED (the production default path). Session names are per (group, arm, seed); every arm is a fresh process
and a fresh `ChatBrain`. Unlike Stage 1's `lived`, no group forces a cupy-default value: every organ these groups
touch runs its production numpy path (the self-initiation organ's numpy path speaks the curiosity-top decodable
concept, its declared light path; the multistep chase, the curiosity organ, the ladder and the mouth have no
backend-gated default).

**Arms per group.** `intact_a` (production, nothing installed); `intact_b` (an independent same-seed rebuild: the
null); one or two LESION arms (below), each with a `.rep0` repeat in a second fresh process (docs/TERMS.md:
"reproduces against the lesion's own repeat"); `permuted` (the exposure permuted: the teach turns, or for `emo` the
affect words); and for C2 two INFORMATIONAL arms never counted. 31 arms per seed (conv 5, infer 7, ask 7, emo 5,
inner 7), 186 jobs.

| group | lesion arm | what it cuts | how (production knob or in-process patch) |
|---|---|---|---|
| conv | `lesion_mouth` | the spiking Broca mouth: the recall surface falls back to the host template stub | `BRAIN_SPIKING_MOUTH=0` (the master mouth kill, which also forces the recall surface off: the 2026-08-26 finding's literal lesion oracle) |
| infer | `lesion_chain_route` | the possessive-chain 2-hop route (a host dispatch at HEAD; section 5) | `BRAIN_CHAIN_ROUTE=0` (production flag) |
| infer (informational) | `qp_lesion` | the composer's spiking `query_patient` read returns None; does the derived answer need it at all? | in-process: `OneBrainComposer.query_patient` -> None; the arm records the call count |
| infer (informational) | `transitive_chat` | (not a lesion) the default-OFF transitive yes/no route over the spiking workspace chase | `BRAIN_TRANSITIVE_CHAT=1` |
| ask | `lesion_curiosity` | the curiosity drive pathway (`from_novelty` -> the spiking ASK pool) | `BRAIN_CURIOSITY_LESION=1` (production flag) |
| ask | `lesion_graded` | the graded per-topic novelty read (the weights-cleared twin: every topic reads ceiling novelty) | `BRAIN_CURIOSITY_GRADED_NOVELTY_LESION=1` (production flag) |
| emo | `lesion_affect` | the graded-affect ladder read (affect_out clamped to 0) | `BRAIN_AFFECT_LESION=1` (production flag) |
| inner | `lesion_workspace` | the GNW workspace's assembly self-recurrence (the re-entrant chase cannot sustain ignition) | `BRAIN_GNW_MULTISTEP_LESION=1` (production flag) |
| inner | `lesion_selfinit` | the self-initiation store (no-encode: nothing surfaces to voice) | `BRAIN_SELF_INITIATE_LESION=1` (production flag) |

**Declared departures, identical in every arm of the group (including intact_a):** (a) no seed threading into the
three host-built organs that ignore `BRAIN_CHAT_SEED` at HEAD (FAILURE_LOG 2026-09-29): production is left as it is.
(b) `PYTHONHASHSEED=0`. (c) The corpus-learned production organs read `data/corpus/tinystories.txt` and DEGRADE
SILENTLY without it: every arm records `corpus_present`, the aggregate requires it, and the HANDOFF checks each pool
node after provisioning (the Stage-1 lesson). (d) Each arm records the brain's build-time fact list
(`builtin_facts`, read off the session's `ChatBrain.stored_facts` after the first turn) as the grounding set for the
C1 and C5 checks; the worker never writes to it. No `sim/` or `webapp/` file is edited by any arm.

## 4. Per criterion: protocol, measurement, anti-cheats, bar, prediction

### 4.1 C1, grammatical grounded conversation (`conv`)

**Protocol.** Four teach turns (section 1), then seven probes: `what does the fox chase`, `tell me more`, `where does
the deer sleep`, `what does the owl hunt`, `what does the mole hunt` (moat), `what does the dog chase` (built-in),
`what does the zorblat chase` (OOV: abstain, never a confabulation).

**Reply-level measurement: the fixed grammar checker** (`grammatical`, `grounded`, `frames` in the runner). A frame
is `the <S> <V> [in|on|at|to] the <O>` anywhere in the reply text (the discourse-register leads that production
prepends, "Setting the held thread aside -- On fox, then --", are outside the frame and ignored). A reply is
GRAMMATICAL iff it contains at least one frame, every frame's verb is the 3sg form of a lemma in the frozen table
(`VERB_3SG`: chase/eat/hunt/sleep/use/learn/store/fly/swim/bite/win/cheat), a taught prepositional fact keeps its
preposition and a transitive one carries none, and the host template stub's surface (`The X Vs Y.`) appears nowhere.
A reply is GROUNDED iff every frame names a fact the brain holds (the build-time list plus the canonical active-voice
facts the teacher asserted; for the passive teach, what a comprehending brain would store, `owl hunts mole`).
Per probe: `fox_chase` = the reply names `hare`, grammatical, grounded; `more` = grammatical, grounded, and at least
one frame is not `fox chases hare` (new content); `deer_sleep` = a frame `the deer sleeps in the meadow`, grammatical;
`owl_hunt` = the reply names `mole`; `mole_moat`, `oov_abstains` = abstained; `builtin_dog_cat`. `all` = every one.
Sub-bars reported beside it: transitive (fox_chase, builtin, more), prepositional (deer_sleep), passive (owl_hunt,
mole_moat). Internal secondary: `comprehension` (D4 monitor trace) on the OOV and passive turns, `activity`.

**Anti-cheats.** (1) Permuted teaching: `the hare chases the fox`, `the owl is hunted by the mole`; the replies must
follow: `what does the fox chase` abstains, `what does the owl hunt` abstains, `what does the mole hunt` -> owl; the
prepositional and built-in checks unchanged. (2) Lesion `lesion_mouth`: the two transitive recall surfaces
(`p_fox_chase`, `c_dog_chase`) must change (the template stub renders the same SVO in a different surface) on 6/6
and reproduce. (3) Host audit (section 5): the SVO acquisition, the question parse, the k-select first-match and the
discourse leads are host; the surface word order and inflection on the transitive frame are the spiking Broca's
rate read (100% attributable in the 2026-08-26 soak); the template stub is host.

**Bar.** `all` on 6/6; lesion 6/6 reproduced; permuted tracked 6/6.

**Prediction: FAIL on `deer_sleep` and on the passive, PASS on the transitive sub-bar.** The store has no
preposition slot (`the robin flies in the sky` was stored as `[robin, fly, sky]`), so `where does the deer sleep`
renders either the spiking `the deer sleeps the meadow` (the mouth's bounded transitive frame, missing the
preposition) or the template `The deer sleeps meadow.`; both fail the checker. The passive `the mole is hunted by
the owl` is acquired by the position-only extractor as `[mole, hunt, owl]`, so `what does the owl hunt` abstains and
`what does the mole hunt` answers owl: `owl_hunt` and `mole_moat` both fail. The transitive recall and the built-in
render on the spiking mouth (`the fox chases the hare`, the dev-shard form), and the follow-up elaborates with `the
fox eats the berry` (the least-evidenced sub-probe; the dev smoke in section 11 is its first read). A FAIL names two
gaps exactly: the frame inventory beyond the transitive (EMERGE-22 / the 2026-08-21 slot-inventory de-risk, not
wired) and form-to-role comprehension (EMERGE-91, not wired: acquisition is a host positional extractor).

### 4.2 C2, inference on held-out (`infer`)

**Protocol.** Four teach turns (section 1), then `what does the heron's prey eat` (the conclusion), `what does the
heron hunt`, `what does the frog eat` (premises), `what does the heron's prey chase` (moat), `does crow chase moth`
(the transitive yes/no probe: reported in every arm, counted in no arm's pass condition; the informational
`transitive_chat` arm reports whether the default-OFF route answers it as derived), `what does the dog chase`.

**Reply-level measurement.** `conclusion` = the reply names `gnat` (the derived terminal: `derived_from[-1][2]`, or
the last frame in the answer); `derived_flag` = the response's `derived` field is true (a derived answer must never
pass as a perceived fact); `premise1`, `premise2`, `moat_prey_chase`, `builtin_dog_cat`. `all` = every one.
`derived_lead` (the "I derived this from" text) is reported.

**Anti-cheats.** (1) Permuted premises (`the heron hunts the gnat`, `the gnat eats the frog`): the conclusion must
be `frog`, `what does the heron hunt` -> gnat, `what does the frog eat` abstains. (2) Lesion `lesion_chain_route`: the
conclusion vanishes (the question falls to the generic parser, which truncates the third content token and
abstains) on 6/6, reproduced. (3) The informational `qp_lesion` arm: the composer's spiking `query_patient` read is
cut; if the conclusion SURVIVES, the route needed no spiking read (the host kb scan answered), which is the
ownership fact section 5 predicts. (4) Host audit: the whole route is host at HEAD (section 5); a PASS is reported
with that share attached, never as neural ownership.

**Bar.** `all` on 6/6; lesion 6/6 reproduced; permuted tracked 6/6.

**Prediction: PASS, host-owned.** The route was verified on exactly this shape (`the wolf hunts the deer`, `the deer
eats the grass`, "what does the wolf's prey eat?") in the 2026-08-25 finding, with `prey` in the role table; the
premises are single-valued so the ambiguity check passes; the moat (`what does the heron's prey chase`) abstains
because the frog has no chase fact. The `qp_lesion` arm is predicted to leave the conclusion intact (the hop is a kb
scan). `does crow chase moth` at defaults is predicted to mis-parse to the first hop (`the crow chases the wren`, the
documented generic-parser truncation) and to answer `moth` as derived only in the `transitive_chat` arm. The Stage-2
reading of a PASS here is "the integrated brain composes two told facts into a never-told conclusion, through a host
route"; the ownership phase's next method is to hand the hop reads and the dispatch to the substrate (the transitive
route is the wired candidate).

### 4.3 C3, curiosity ask-rate >= 2x on gaps + learning-progress (`ask`)

**Protocol.** Three teach turns about set A; then the six discrimination probes interleaved (novel `eel`, taught
`yak`, novel `crab`, taught `moose`, novel `newt`, taught `goat`; every probe asks `what does the <X> chase`, a fact
never taught for any of them, so all six abstain and the only difference is whether the TOPIC is known); then the
learnable gap (`what does the ibex chase` -> the teacher answers `the ibex chases the marmot` -> the same question);
then the unlearnable gap (`what does the toad chase`, four times, with `what does the dog chase` between probes; the
teacher never answers).

**Reply-level measurement.** `asked` = the reply text carries the follow-up question frame ("what can you tell me
about"); `ask_rate_gap` and `ask_rate_moat` over the three novel and the three taught probes; `discriminates` =
ask_rate_gap >= 2/3 AND ask_rate_gap >= 2 x ask_rate_moat; `lp_learnable` = asked on the first ibex probe, the second
recalls `marmot` and is not asked; `lp_unlearnable` = asked on the first toad probe and NOT asked on the fourth;
`all_gaps_abstain` (every gap probe abstains: the moat is intact under curiosity); `builtin_dog_cat` on all three
controls. `all` = every one. Internal secondary per probe: the organ's `curious`, `want_hz`, `threshold` and the
graded `novelty` value (`curiosity` in the response).

**Anti-cheats.** (1) Permuted teaching: set B taught (`the eel eats the grass`, ...) and set A never mentioned; the
asks must follow the teaching (rate on A >= 2/3 and >= 2 x the rate on B). (2) Lesion `lesion_curiosity`: the follow-up
vanishes on the novel gaps, the ibex first probe and the toad first probe, 6/6, reproduced. (3) Lesion
`lesion_graded`: every topic reads ceiling novelty, so the TAUGHT-topic moats now ask too (the scored turns are the
three moats), 6/6, reproduced: the discrimination rides the learned familiarity projector, not the abstain flag.
(4) The unlearnable probe is the DR-1 honesty test at the reply level: a brain that asks on every abstain
regardless of history keeps asking and fails it. (5) Host audit: the topic word extractor and the wh-frame are
host; the ASK-pool rate and the threshold decision, and the novelty projector, are spiking reads.

**Bar.** `all` on 6/6; both lesions 6/6 reproduced; permuted tracked 6/6.

**Prediction: FAIL on `lp_unlearnable`, PASS on the discrimination and the learnable gap.** The graded novelty
gate (default-ON since 2026-09-05) reads a taught topic as familiar: the Stage-1 dev shard shows the exact
reply-level shape (asked on `bear` before teaching; not asked on `otter`, `wolf`, `canary` after teaching), so
`ask_rate_gap` is predicted 1.0 and `ask_rate_moat` 0.0. The learnable gap closes because the teacher's answer is
acquired and recalled (Stage 1's C2/C3, PASS at dev seed 7). The unlearnable gap is asked on every probe because no
production organ tracks learning progress: the judge's read is a deterministic function of the topic and the
substrate (the OU stream is reset per read), so the 4th probe reads exactly as the 1st. A FAIL names the
integration slot: the Oudeyer-Kaplan LP signal (de-risked as a CPU proxy 2026-08-02/07, on-bridge 1/6) has no
production organ; the next method is an LP trace on the ASK pool's own history (asked-and-nothing-learned lowers
the drive), default-off, re-run at dev seeds before any flip.

### 4.4 C4, discrete-emotion read-out (`emo`)

**Protocol.** A baseline feel; then joy, fear, anger, sadness in that order, each situation twice (the mood EMA,
decay 0.4, sits at 84% of the situation's value after two exposures) followed by `how do you feel`. Every situation
has at least four content words (never a 3-content-word assertion the host SVO extractor would store) and every
affect word is STRONG under the production salience gate (|v-5| >= 2; pinned by the tests). The situations are
second-person (the system) and name no third party, so the W5 empathic lead never fires.

**Reply-level measurement.** `parse_feel` reads the feel reply's sign word, differential, arousal band and the
CATEGORY it names, through the fixed lexicon `EMOTION_LEXICON` (joy: happy/joy/joyful/glad/delighted/pleased/
cheerful/elated; fear: afraid/scared/fear/fearful/anxious/frightened/terrified; anger: angry/anger/mad/furious/
rage/irritated/annoyed; sadness: sad/sadness/sorrow/grief/lonely/unhappy/miserable/down); a reply naming words of
more than one category names none. `all` = the four replies name four distinct categories, each the intended one.
Reported beside it, never counted: `sign_correct_all` (joy positive, the other three negative), the arousal bands,
`n_distinct_2d` (how many distinct (sign, arousal) pairs the four situations produced: the differentiation the
scalar ladder does show), the differentials.

**Anti-cheats.** (1) Permuted affect company: the joy frame carries the sadness words and vice versa, the fear frame
carries the anger words and vice versa; the expected categories follow the words. Tracking is defined only when
the intact arm names categories at all. (2) Lesion `lesion_affect`: every feel reply reads neutral, so all four
scored replies differ from intact (whose signs move), 6/6, reproduced. (3) Host audit: the salience gate and the
map lookup and the EMA are host arithmetic; the differential is the ladder's spiking read; the read-out text is host.

**Bar.** `all` on 6/6; lesion 6/6 reproduced; permuted tracked 6/6.

**Prediction: FAIL, capability absent 6/6, by code.** `feel_readout` prints a sign, a number and an arousal band
and no category word; the discrete-emotion de-risk is runner-only. The 2-D read is predicted differentiated: the
sign correct on all four (the joy words' learned values are positive, the rest negative) and at least two distinct
arousal bands (the sadness words' arousal norms, 4.6-5.4, sit below the fear/anger words', 6.0-7.0), so
`n_distinct_2d` >= 2 while fear and anger read identically. A FAIL names the integration slot exactly: the
2026-08-13 appraisal-dimension -> emotion-category WTA slice folded into the production ladder, its winner given a
word in the read-out, default-off, re-run at dev seeds before any flip.

### 4.5 C5, private -> inner-speech chain (`inner`)

**Protocol.** Three teach turns (the chain), then `say something` (PRIVATE), `what does the lynx chase all the way`,
`what does the vole chase all the way` (INNER), `what does the lynx chase` (single hop), `what does the mite chase
all the way` (chain tail: abstain), `what does the beetle chase all the way` (-> mite; the tail in the permuted
arm), `what does the dog chase`.

**Reply-level measurement.** `private_voices_stored_fact` = the reply carries the self-initiated frame
("Something's been on my mind -- <a> <v> <p>.") and that triple is a fact the brain holds (the build-time list or
a taught fact); the neutral idle line ("Nothing in particular is surfacing ...") fails it. `inner_terminal_lynx`,
`inner_terminal_vole` = the reply names `mite` (the terminal, never told with the lynx or the vole);
`single_hop` = vole; `moat_tail` = abstain; `builtin_dog_cat`. `all` = every one. Internal secondary: the chase
trace (`cycles`, `resolved_hops`, `terminal`; `chain_hops_internal_ge2` reported), the self-initiation organ's
`n_utt`.

**Anti-cheats.** (1) Permuted chain (`lynx -> mite -> vole -> beetle`): the terminal must be `beetle` on both chase
probes, the single hop `mite`, the mite's own chase reaches `beetle`, and the beetle's chase abstains. (2) Lesion
`lesion_workspace`: the terminal is not reached (an abstain or the first hop) on both chase probes, 6/6,
reproduced; the emergent stop is the spiking competition, not a host loop. (3) Lesion `lesion_selfinit`: the idle
turn falls to the neutral idle line, 6/6, reproduced. (4) Host audit: the chase-marker detection and the (agent,
action) parse are host; each hop's `query_patient` and the ignition/stop read are spiking; the self-initiated
frame is a host articulation scaffold around a spiking decode; on numpy the WHICH of the idle concept is the
curiosity-top ranking, not the cupy CA3 wander (declared).

**Bar.** `all` on 6/6; both lesions 6/6 reproduced; permuted tracked 6/6.

**Prediction: PASS on the inner chain, PASS on the private utterance, with one named risk.** The chase is
reply-load-bearing 6/6 in B2b on a taught chain and its de-risk covers 3-hop depth; the self-initiated utterance is
reply-load-bearing 6/6 in B2b. The risk: the multistep chase's over-run guard reads `n_ignited == 0` at the leaf, and
a 3-link chain taught in-session is one link longer than the battery's probe; a FAIL on `inner_terminal_lynx` with
`inner_terminal_vole` passing would name the chase's depth on in-session facts, measurable from the trace. What this
criterion does NOT show, stated now: no route lets the brain voice the chain on request, so "private" and "inner"
are measured on two different mechanisms rather than as one utterance internalizing; that is the named gap for the
faculty ("a think-aloud route over the chase trace") whether the criterion passes or not.

## 5. Host-decision seams these answers pass through (the host-shortcut audit)

The plan (`2026-09-25-prove-who-owns-the-computation-PLAN.md` sections 1-2) requires a host-share beside every
capability number. The host-share instrument (branch `research/host-share-baselines`) reports its own numbers; this
gate REPORTS the seam audit below and computes no F/S/D. Running that instrument on these probe turns is
NEEDS-CLAUDE #10 in the HANDOFF.

| criterion | host code that touches the answer (file) | neural state before it | seam class |
|---|---|---|---|
| C1 surface | `_maybe_acquire` SVO extraction + `lemma_verb` (store side); `_extract_route` / `_neural_question_parse` (question parse); `_fact_shard_first_match` (k-select); the discourse-register leads; `TemplateStubFaculty.render_svo` for non-transitive facts | the recalled block; the Broca rate ranking (order + inflection on the transitive frame) | content k-select (host); surface (spiking on the transitive frame, host template elsewhere) |
| C2 | `_POSSESSIVE_CHAIN_RE` (dispatch), `_ROLE_NOUN_HINTS` (hop-1 relation), `_distinct_patients` (`.kb` scan: the hop read itself at HEAD), `frame_derived_answer` | none on the hop reads at HEAD (the spiking `query_patient` is the fallback for an unscannable composer only) | routing, candidate generation AND content: host-owned end to end |
| C3 | `extract_topic` (which word), `followup_question` (the wh-frame), the abstain that supplies the gap | the graded-novelty projector read; the ASK pool's rate vs its calibrated threshold | routing (host) -> commitment (spiking) |
| C4 | `appraise_text` salience gate + learned-map lookup; `_update_session_mood` EMA; `feel_readout` bands | the ladder's opponent read (`read_differential`) | candidate scoring (host) -> read (spiking) -> composition (host) |
| C5 inner | `detect_chase` (marker), the inner (agent, action) parse | each hop's `query_patient`; the workspace ignition read that decides ADVANCE / COMMIT / ABSTAIN | routing (host) -> content + commitment (spiking) |
| C5 private | the idle-turn regex; `self_initiated_text` frame; on numpy the curiosity-top ranking of decodable concepts | the mouth's resonate-and-fire decode of the concept | routing (host) -> content (spiking decode; the WHICH is a host ranking on numpy) |

Expectation, written before the gate: C2's PASS is host-owned (the `qp_lesion` arm is predicted to show the spiking
read is not needed); C3's discrimination and C5's chain are the two spiking-owned decisions in this gate; C1's
grammar is spiking-owned on the transitive frame only; C4's number is a spiking read of a host-computed drive.

## 6. Mechanisms expected to be NECESSARY (six-seed ablation; CLAUDE.md non-negotiable 4)

| criterion | mechanism | ablation arm | predicted reply-level effect of the ablation |
|---|---|---|---|
| C1 | the spiking Broca mouth (recall surface) | `lesion_mouth` | the transitive replies change surface (template stub: `The fox chases hare.`), content unchanged |
| C2 | the possessive-chain route | `lesion_chain_route` | the conclusion vanishes (generic-parser abstain) |
| C2 | the composer's spiking `query_patient` | `qp_lesion` (informational) | NO effect on the conclusion (host kb scan); the single-hop premise recalls may change |
| C3 | the curiosity drive pathway | `lesion_curiosity` | no follow-up anywhere |
| C3 | the graded novelty projector | `lesion_graded` | the taught-topic moats ask too (discrimination collapses) |
| C4 | the graded-affect ladder | `lesion_affect` | every feel read neutral |
| C5 | the workspace self-recurrence | `lesion_workspace` | the terminal is not reached |
| C5 | the self-initiation store | `lesion_selfinit` | the neutral idle line |

Not ablated here, named for the follow-up: the on-brain BridgeParser question parse (`BRAIN_SUBSTRATE_PARSE_LESION`),
the comprehension monitor (`BRAIN_COMPREHENSION_LESION`, re-measurement pending after the 2026-09-29 fix), the D6
referent store on the discourse follow-up.

## 7. Validity, UNDEFINED handling and the verdict rules (frozen; scored only by `--aggregate`)

- **Defined seed** (per group): every COUNTED arm of the group present and readable (`intact_a`, `intact_b`, each
  lesion arm and its `.rep0` repeat, `permuted`; informational arms excluded); the null clean (every primary reply
  field of every turn equal between `intact_a` and `intact_b`; the primary fields are `answer`, `abstained`,
  `recalled_svo`, `derived`); every response's `source` = `tiny-demo +LTM`; `PYTHONHASHSEED=0` and `corpus_present`
  recorded; for `emo`, all four feel reads parsed. A seed failing any of these is UNDEFINED for that group, with the
  reason recorded; UNDEFINED is never counted as a pass or a fail (a missing arm is an instrument failure, never a
  FAIL clause).
- **Lesion effect** = a primary reply field differs between `intact_a` and the lesion arm on at least one of the
  lesion's scored turns (`SCORED_LABELS` in the runner: C1 the two transitive recalls; C2 the conclusion probe; C3
  the novel gaps + the two first asks for the drive lesion, the three taught-topic moats for the graded lesion; C4
  the four feel turns; C5 the two chase probes for the workspace lesion, the idle turn for the self-initiation
  lesion). **Reproduced** = the lesion arm and its `.rep0` agree on every primary field of every turn. An in-process
  lesion whose patch was never called is `applied=False` and voids that arm. Attribution (`tools.lab.attributable_to`,
  treatment = primary-field differences intact-vs-lesion on the scored turns, control = the same count between the
  two intact builds) is recorded per seed and lesion.
- **Permuted tracks** = the permuted arm's replies follow the permuted exposure (each group's rule in section 4).
- **Group verdict:** INCOMPLETE if fewer than 6 defined seeds; else PASS iff correct 6/6 AND every lesion arm has
  its effect 6/6 AND reproduced 6/6 AND permuted tracks 6/6; else FAIL with the failing clause named ("capability
  absent on k of 6", "not load-bearing", "not exposure-driven"). The sub-clause counts (`sub_clauses_n_true`) are
  reported so a FAIL says which clause failed; they are never a pass condition alone.
- **Criterion verdict** = its group's verdict. Informational arms are reported and never counted.
- **Stage verdict:** the aggregate carries a `preconditions` block (`tools.verdict.Verdict`): no primary shard
  missing, null clean everywhere, expected source everywhere, hash seed, the corpus present in every primary arm,
  every in-process lesion applied, six defined seeds per criterion. Any failed precondition makes the stage
  UNDEFINED (INCOMPLETE), with the per-group numbers still reported; else PASS iff all five criteria PASS, else FAIL.
- **Numbers:** an ask-rate over a probe whose reply errored is UNDEFINED for that seed (the seed is then undefined
  through the missing-field rule); a feel read that does not parse makes the `emo` seed UNDEFINED.
- **Amendments:** any change after the first gate job is dispatched is appended to section 12, dated, before the
  aggregate is read; a changed bar voids the gate.

## 8. What each outcome means

| criterion | PASS means | FAIL means (a first measurement, not a stop) and the gap it names | next method (no-defer) |
|---|---|---|---|
| C1 | every reply in a mixed dialogue is a well-formed frame naming a true fact, a passive is comprehended, the surface is authored by the spiking mouth | which sub-bar fails names it: prepositional (no frame beyond the transitive), passive (positional acquisition), elaboration (the discourse thread) | wire the learned frame inventory (EMERGE-22 / 2026-08-21) behind the recall mouth; a form-to-role comprehension organ (EMERGE-91) in front of `_maybe_acquire`; both default-off, dev seeds first |
| C2 | the integrated brain composes two told facts into a never-told conclusion and frames it as derived (host-owned route) | the route is missing or mis-dispatching on this shape | the transitive route (spiking chase) flipped after its own prereg; hop reads onto the substrate (Phase B content seam) |
| C3 | the brain asks on what it does not know and not on what it knows, closes a gap when answered, and stops asking where nothing is learned | `discriminates` names the novelty read; `lp_unlearnable` (predicted) names the missing learning-progress signal | an LP trace on the ASK pool's own history (default-off), re-run at dev seeds |
| C4 | the feel reply names distinct, correct emotion categories for four appraisals | (predicted) no category in the read-out: the scalar ladder | fold the 2026-08-13 appraisal -> category WTA into the production ladder (default-off), give the winner a word |
| C5 | the brain voices a thought it selected, and runs a self-cued chain to a conclusion it speaks | which half fails names it: the chase depth on in-session chains, or the idle-turn selection | a think-aloud route over the chase trace (the missing private-speech voice of the chain) |
| stage | Stage 2 exit on the integrated brain | the per-criterion table above is the Stage-2 checklist the roadmap's Phase C exit needs | the failing criteria become the next integration slots in ledger order |

## 9. Predictions (written before the gate; expectations, not bars)

| criterion / group | predicted verdict | reason |
|---|---|---|
| C1 `conv` | FAIL (capability absent 6/6: prepositional + passive) | no preposition slot in the store, template stub outside the transitive frame; positional acquisition of the passive |
| C2 `infer` | PASS (host-owned) | the possessive-chain route verified on this shape 2026-08-25; the `qp_lesion` arm predicted to leave it intact |
| C3 `ask` | FAIL (capability absent 6/6: `lp_unlearnable`); `discriminates` and `lp_learnable` 6/6 | graded novelty default-ON; no learning-progress organ in production |
| C4 `emo` | FAIL (capability absent 6/6); signs 4/4, `n_distinct_2d` >= 2 | the feel read-out prints no category |
| C5 `inner` | PASS | chase and self-initiation both reply-load-bearing 6/6 in B2b; risk: the chase depth on a 3-link in-session chain |
| stage | FAIL (C1, C3, C4) | three criteria predicted absent; this FAIL is the measurement the calendar asked for |

## 10. Cost, lane and ETA

**Lane: the pool (numpy), not the GPU queue.** Reasons, in order: (1) comparability with the Stage-1 packet (same
harness, same seeds, same declared departures, same scorer machinery); (2) from 2026-10-06 the local model drives
and its serving (llama-swap on the one 3090) holds the GPU, and a brain job on the GPU queue needs `llm yield`,
which pauses the driver for the gate's wall time, whereas the pool costs the driver nothing; (3) none of the five
groups needs a cupy-only production path (no forced write; every organ's numpy path is its production numpy path);
(4) the reference-hardware rule (numpy on the pool is the lane every battery uses). The runner accepts `--backend
cupy` per shard; a cupy replication through `tools/gpu_queue.sh` is an informational follow-up (HANDOFF NEEDS-CLAUDE
#11), never a gate arm.

**Jobs.** 186 (6 seeds x 31 arms). A tiny-demo+LTM build is ~6.1-7.7 GB RSS, so `mem_gb=7`, `memcap 9`, one to two
jobs per 15 GB pool node. Planning figures (Stage-1 measurements: build + first content turn 4.5-13 min, a teach or
recall turn 10-47 s, a feel turn ~2 s, an exposure turn ~8 s, lazy organ builds 2-8 min on a few turns; a pool node
runs ~1.4x slower than this workstation), refined by the dev smoke in section 11: `conv` ~20 min (11 turns), `infer`
~20 min (10 turns), `ask` ~25 min (19 turns), `emo` ~12 min (13 turns), `inner` ~30 min (10 turns plus the GNW
workspace build). Per seed: 5x20 + 7x20 + 7x25 + 5x12 + 7x30 = 685 min; about 69 job-hours in total. Pool only
(pool41 + pool42, 2-4 concurrent): about 17-35 h of wall time once the lines are reached. Every line carries
`max_age_h=240` (the cap: the queue depth at queue time in November is unknown, and a gate line must never expire as
debris). If the queue is deeper than ~150 lines at queue time the HANDOFF names the AWS CPU lane (r7i.4xlarge, ~16
jobs per node under the $50/day cap, about 4-6 h, about $6-9) as NEEDS-CLAUDE #13, exactly as Stage 1 did.

## 11. Dev verification on the final code (dev seed 7, numpy, not a verdict)

The dev arms (`research/findings/raw/_stage2_exit/dev/s7/`: the five `intact_a` arms, then `ask/lesion_graded`,
`inner/lesion_workspace`, `conv/lesion_mouth`, `emo/lesion_affect`, `infer/transitive_chat`) run AFTER this file is
committed, one brain at a time under `tools/mem_ok.sh 7 4` + `tools/memcap.sh 9` (the workstation is shared with the
B2d mouth-arm GPU job, 10 GB RSS, whose own margin that is; `mem_ok.sh 7` at the default 8 GB margin refused while
it ran, so no dev arm had completed when this file was committed). Their PASS-list lines (no errored turn, source
`tiny-demo +LTM`, hash ok, corpus present, `builtin_facts` recorded, per-turn timings) go to
`docs/drafts/stage2-exit/WORKLOG.md` and the HANDOFF's section 2, never into this file, so the prereg stays
byte-identical from its commit to the gate. Dev seeds govern nothing; the smokes are integrity checks plus the
first reads behind section 9. If a dev arm reveals an INSTRUMENT defect (a turn the handler parses as something
other than the protocol intends, a scorer reading the wrong field, a lesion flag production does not read), the
fix is filed as a dated amendment in section 12 in its own commit before any gate seed; a CAPABILITY result at dev
seed 7 changes nothing here (the predictions in section 9 stand as written from the record).

## 12. Amendments

(Appended dated entries only. The bars, arms and turns above are frozen at the commit that adds this file.)

**2026-09-30, amendment A1 (instrument, filed before any gate job was queued or dispatched; adversarial review of
the packet at 8c934312a; runner fix commit ed956d08b).** No bar, arm, turn, seed, prediction or scored field
changes. (1) Harness: `--shard` spawned its `--worker` without `--allow-gate-seed`, and the gate-seed guard also
applies to `--worker`, so every gate job would have exited rc=2 before building a brain; the shard now hands the
flag and `--prereg` to its worker. (2) Section 7's defined-seed rule is applied to EVERY counted arm (intact_a,
intact_b, each lesion arm and its `.rep0`, permuted), not only the intact pair: every response's `source` =
`tiny-demo +LTM`, `PYTHONHASHSEED=0`, `corpus_present`; and three instrument failures are added to it, each making
the seed UNDEFINED (never a pass, never a FAIL clause): an errored turn in any counted arm (section 7's
"an ask-rate over a probe whose reply errored is UNDEFINED", generalized: an errored scored turn in a lesion arm had
read as "no effect" -> FAIL "not load-bearing"); a caught organ exception (`<organ>.error` in a response, or a
chase trace whose reason is an error) in any counted arm (FAILURE_LOG 2026-09-29, B2b: such a turn was scored as
the lesion's effect); an empty `builtin_facts` in a `conv` or `inner` arm (the grounding set their scorers read).
The aggregate's preconditions gain "no errored turn and no organ error in any counted arm" and "the grounding set
recorded". (3) Section 4.5's internal secondary: the handler stashes the chase trace on the chat
(`_last_gnw_multistep`) and never attaches it to the response, so the worker now records it per turn under
`internal` (object identity: only a trace written that turn); the vole probe's trace is reported beside the lynx
probe's. (4) Each arm also records its peak RSS and the LTM bundle the handler resolved (the shipped default is
`wikidata_100k`, `wikidata_core_15k` the fallback; both give the same `source` string); reported, never scored.

**2026-09-30, amendment A2 (lane, before any gate job; runner commit 8691e84c1).** Section 10's memory figures are
replaced by the dev-seed-7 measurement on the final runner (workstation, numpy): peak RSS 6.3-6.5 GB for `conv`,
`infer`, `ask` and `inner`, 8.1 GB for `emo`; so every gate line declares `mem_gb=9` under `memcap.sh 10` (was 7
under 9, which left ~1 GB before the cap killed an `emo` shard). Measured walls (workstation; a pool node is ~1.4x
slower): conv 876 s, infer 1246 s, ask 913 s, emo 557 s, inner 793 s. No bar, arm, turn, seed, prediction or scored
field changes. The dev-seed-7 capability reads (C3's DA-gated threshold, C5's private utterance from the
self-initiation organ's own fixed lexicon, C1's follow-up and wh-parse) are recorded in the packet's WORKLOG, not
here: per section 11 a dev capability read changes nothing in this file; any protocol change they prompt is a
separate, later amendment by Claude before GATE_SHA is pinned.
