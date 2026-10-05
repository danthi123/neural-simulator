---
type: finding
status: live
date: 2026-09-29
lane: integration-first (stage exits on the calendar; owner 2026-09-29, acceleration report M1 / section 6.1)
mechanism: PRE-REGISTRATION of the first developmental STAGE-EXIT attempt -- Stage 1 (INFANT) of the master roadmap, run END TO END on the INTEGRATED production brain (the live /api/brain-chat handler at production defaults, pinned revision), never on an isolated de-risk runner; four criteria (category held-out inheritance, word->concept recall, lived-fact recall, valence inheritance r>=0.55) each measured at the REPLY level with the internal read-out secondary, with a null arm, a lesion arm per responsible faculty (repeated), a permuted-exposure arm, and a host-shortcut audit
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION only. Filed before any gate-seed arm was built. No result is claimed here.
runner: research/runners/stage1_exit_gate.py
artifacts:
  - research/runners/stage1_exit_gate.py
  - tests/test_stage1_exit_gate.py
  - docs/drafts/stage1-exit/HANDOFF.md
  - research/findings/raw/_stage1_exit/dev/s7/word__intact_a.json
  - research/findings/raw/_stage1_exit/dev/s7/word__lesion_recruit.json
  - research/findings/raw/_stage1_exit/dev/s7/val_heldout__intact_a_2words.json
  - research/findings/raw/_stage1_exit/dev/s7/cat_disc__intact_a.json
  - research/findings/raw/_stage1_exit/dev/s7/lived__intact_a.json
---

# Stage 1 (INFANT) exit gate on the integrated production brain: PRE-REGISTRATION (2026-09-29)

**Committed alone, before any gate-seed (42/43/44/100/101/102) arm is built.** Branch `research/stage1-exit-gate`.
The gate runs at a revision that contains this file BYTE-UNCHANGED plus the runner (the pin is recorded in
`docs/drafts/stage1-exit/GATE_SHA` and checked by the HANDOFF), provisioned on the pool with
`tools/pool_provision.sh --revision <sha> --isolated`; every job line carries the pin and `--prereg` (the gate-seed
guard, `tools/gate_seed_guard.py`).

**Why now (owner directives).** Integration-first with a stage-exit attempt now: "run the Stage 1 exit gate end to
end in October even if it fails" (`docs/plans/2026-09-29-acceleration-report.md` section 4.1 M1); "Stage exits are
scheduled on the calendar, not on readiness ... each run even if it is expected to fail, because a failed exit is
the first measurement of the integrated brain the project has ever had" (section 6.1); Stage 1 by 2026-10-24
(CLAUDE.md, integration-first + pivot dwell). The roadmap phases: `docs/plans/2026-09-29-roadmap-to-the-primary-goal.md`
section 2 ("No developmental stage-exit gate has been passed by the integrated brain") and Phase C's exit ("Stage 1-2
exit gates in the integrated brain"). This document therefore preregisters two sub-tests it EXPECTS TO FAIL for
code-level reasons (section 4.1 and 4.4b): the build-lane checklist's "do not stage a run whose pre-registration
already predicts failure" is overridden here by the owner's calendar rule, and each predicted FAIL names the gap it
measures.

**What this is not.** Not a de-risk (every mechanism involved already has one, cited in section 2). Not a GO/NO-GO
on any mechanism. The verdict vocabulary is PASS / FAIL / UNDEFINED(INCOMPLETE) per criterion and for the stage; a
FAIL is a measurement of the integrated brain, never a stop (CLAUDE.md non-negotiable 3: the next method, not a
deferral). Honesty boundary: every "feel" read-out below is a functional read of a ladder differential; nothing
here measures or asserts experience.

## 1. The four criteria, operationalized

The roadmap's Stage-1 exit line (section 4, STAGE 1): "category held-out inheritance; word->concept recall;
lived-fact recall; valence-inheritance r>=0.55". Each becomes one probe GROUP on the live handler (one session = one
fresh `ChatBrain`, one arm = one process), scored by the registered aggregate (`--aggregate`) and nothing else.

| criterion | group (role) | the brain EXPERIENCES (teacher channel = chat turns) | held-out test (reply-level primary field) | frozen bar |
|---|---|---|---|---|
| C1 category held-out inheritance | `cat_disc` (primary) | 10 assertions: three bird exemplars share a context (`eat worm`, `fly sky`), two fish exemplars share another (`eat bug`, `swim lake`); the held-out `canary` is given ONLY the bird motion context | "what does the canary eat" (strict recall route) and "what might the canary eat" (the brain's own generative route): the reply names `worm` (the bird property never told for the canary) | the reply names the category-consistent patient on 6/6 seeds; the permuted-context arm (canary streamed with the fish context) names `bug` or abstains, never `worm`, 6/6; the draw-likelihood lesion changes the reply 6/6 and reproduces |
| C1 diagnostic | `cat_told` | the same, but the category is TOLD (`robin isa bird`, `bird eats worm`, `canary isa bird` ...) | as above; plus both premises recalled directly (`what is a canary` -> bird; `what does the bird eat` -> worm) | reported beside C1; not a pass condition |
| C2 word->concept recall | `word` (primary) | two assertions with words the brain has never seen in any store (`the otter eats the clam`, `the bear chases the otter`) after a pre-exposure probe | `what does the bear chase` -> otter (the new word decoded as patient); `what does the otter eat` -> clam (the new word as agent); `what does the otter chase` / `what does the clam eat` -> abstain (moat); `what does the dog chase` -> cat (a built-in fact survives). Reported, not counted: `what does it eat` (anaphora through the new word; see 4.2) | all of the counted probes on 6/6 seeds; the pre-exposure probe abstains; the recruit lesion removes every counted new-word answer while the built-in fact survives, 6/6, reproduced; the role-permuted exposure is tracked 6/6 |
| C3 lived-fact recall | `lived` (primary) | a pre-exposure probe and a pre-exposure "did we discuss the wolf"; then `the wolf bites the apple`; then three unrelated turns | `what does the wolf bite` -> apple after the intervening turns; `did we discuss the wolf` -> a disclosure (not the "I don't recall" abstain); `what does the wolf chase` -> abstain; `what does the dog chase` -> cat | all on 6/6; both pre-exposure probes abstain; the substrate-recall lesion removes the recall and the D5 lesion removes the disclosure, each 6/6 and reproduced; `the wolf bites the pear` (permuted) is tracked 6/6 |
| C4 valence inheritance | `val_heldout` (primary) | each of 24 held-out affect words (section 4.4) presented twice, bare, in a seed-specific order, each followed by the production feel query | Pearson r, per seed, between the differential PRINTED in the feel reply and the word's Warriner norm (which the brain never reads at run time; its per-word value is the DR-2 leave-one-out learned map) | r >= 0.55 on every seed (the roadmap's own bar, DR-2's gate); the ladder lesion makes r UNDEFINED (no variance) or |r| < 0.20 on every seed; the permuted learned map gives r < 0.20 on every seed |
| C4 diagnostic | `val_novel` | two words outside every lexicon (`glorp`, `blick`) keep valenced company; each is then presented alone against the mood the other left | the feel reply's differential moves toward the company's valence on both probes | reported beside C4; not a pass condition |

The exact turn tables are frozen in `research/runners/stage1_exit_gate.py` (`GROUPS`); the messages there are the
protocol. "Teacher channel" here means the scripted interlocutor typing into the chat, which is the host acting as
the social environment (CLAUDE.md non-negotiable 1: the teacher is legitimate host code). No arm injects an answer,
a code, a weight or a store entry: every exposure is a chat turn the production handler receives as the owner would
type it.

## 2. The record this rests on (research-first)

`bash tools/before_you_build.sh` was run 2026-09-29 (logged to the shared corpus-check log) plus four RAG queries,
one per criterion. What exists is a set of isolated de-risks, a production integration for C2/C3, a production
value-source for C4, and NO integrated inheritance route for C1:

- **C1.** Inheritance over an is-a hierarchy emerges on the spiking HTM cortex with no inference engine
  (EMERGE-26, `2026-07-02-emerge26-emergent-inheritance-GO.md`, 6/6, deranged-superordinate and dAP-lesion
  collapse); the GNW workspace reasons through a LEARNED category (rung-3c,
  `2026-07-07-GNW-rung3c-learned-category-inheritance-6of6-heldout-5of6-identity.md`, held-out `canary` inherits
  `flies` via a co-occurrence-learned BIRD context, permuted-context follows the permutation); inheritance rides
  categories the brain discovers by clustering its own codes, corpus-general
  (`2026-07-08-knowledge-half-inheritance-over-FULLY-EMERGENT-clusters-corpus-general-GO.md`); real-word category
  inheritance from PPMI codes is linear, so the goal-relevant depth lives in composition
  (`2026-07-08-deep-credit-real-word-inheritance-is-linear-depth-lives-in-composition.md`). In the LIVE handler the
  only multi-hop routes are the possessive chain (`research/runners/compositional_chain_route.py`, default-on, a
  host regex + role-noun table over two `query_patient` hops, `2026-08-25-reasoning-frontier-chain-routing.md`)
  and the transitive-chase route (`webapp/reasoning_transitive_chat.py`, default-OFF, `2026-09-24-reasoning-
  transitive-chat-PREREGISTRATION.md`); neither expresses `isa` inheritance, and the strict recall route is an exact
  (agent, action) match. The generative route (`what might X V`, the #3E channel wired 2026-08-12,
  `2026-08-12-INTEGRATION-open-ended-generation-brain-volunteers-novel-propositions-on-default-chat.md`; the
  vocab-agnostic spiking DRAW, `2026-08-12-vocab-agnostic-spiking-openended-generation-6seed.md`) proposes a
  novel, plausibility-gated proposition over the brain's OWN fact graph, including facts taught in-session: it is
  the one default-on path by which the canary could be assigned a property it was never told.
- **C2 / C3.** The default turn LEARNS a fact heard this conversation and recalls it from the spiking substrate,
  lesion-verified (`2026-08-12-INTEGRATION2-in-loop-learning-substrate-first-recall-the-turn-LEARNS-lesion-
  verified.md`); a never-seen word is bound through recruit-an-assembly (`OneBrainComposer(vocab_headroom=128)`,
  `2026-08-12-onebrain-spiking-store-breaks-runtime-newword-LEARN-de-risk.md`); the onebrain resonate-and-fire recall
  is the production default (`2026-08-12-INTEGRATION-onebrain-is-now-the-production-default-genuinely-spiking-
  recall.md`). The D5 hippocampal dAP episodic recall gate is wired and its store->recall drive is reply-load-bearing
  6/6 in the B2b battery (`2026-08-12-D5-episodic-production-organ-spiking-recall-gate-wireable.md`,
  `2026-09-20-hollow-episodic-drive` diagnosis in `load_bearing_fraction.py`, and
  `2026-09-29-load-bearing-fraction-reply-level-rescore.md`); the BTSP write is cupy-gated on the production numpy
  path (`webapp/server.py::_episodic_store_ok`), forced with `BRAIN_EPISODIC_STORE=1` exactly as the battery does.
- **C4.** Concepts learn their valence from the association graph, held-out r = 0.81 (per-seed 0.77-0.83), permuted
  graph collapses (DR-2, `2026-07-23-DR2-affective-concept-tagging-6seed-GO.md`); the production appraisal VALUE is
  that leave-one-out learned map (`2026-08-12-D1-affect-appraisal-value-learned-DR2-not-hardcoded-lexicon.md`,
  `research/findings/raw/_affect_learned_valence_map.json`), while the salience GATE (which words move the mood at
  all) is the raw Warriner norm; the mood colors the reply through the co-resident spiking graded-affect ladder
  (`2026-08-12-GateB-affect-colors-production-chat.md`) and the `how do you feel` read-out is a functional read of
  that ladder's differential (`affect_production_organ.feel_readout`). The variant whose opponent weights derive
  from the brain's own learned code (that finding's wording) sits at a boundary near r = 0.55
  (`2026-08-13-affect-opponent-weights-self-organized-BOUNDARY.md`). A word outside
  the Warriner list contributes exactly zero at production defaults; the default-OFF learned affect vocabulary
  (`2026-09-24-affect-learned-vocabulary-PREREG.md`, `BRAIN_AFFECT_LEARNED_VOCAB`) is the named candidate. Under
  the stub renderer the affect-manner rows are trace-only (the reply-level rescore); the feel read-out is the one
  affect surface that reaches `answer` at production defaults, which is why C4 reads it.

## 3. The harness (frozen)

Real `webapp.server.brain_chat` handler, called in-process by a worker subprocess exactly as the regression battery,
the load-bearing-fraction runner and the host-share instrument do; tiny-demo brain with the shipped LTM tier
(`BRAIN_LTM_SHIP_DEFAULT=1`, its production default made explicit; expected `source` = `tiny-demo +LTM`); stub
renderer; no LLM; numpy/CPU; one BLAS thread; `BRAIN_CHAT_SEED=<seed>`; `PYTHONHASHSEED=0`; the request's `rich`
field OMITTED (the production default path). Session names are per (group, arm, seed); every arm is a fresh process
and a fresh `ChatBrain`.

**Arms per group.** `intact_a` (production, nothing installed); `intact_b` (an independent same-seed rebuild: the
null); one or two LESION arms (below), each with a `.rep0` repeat in a second fresh process (docs/TERMS.md:
"reproduces against the lesion's own repeat"); `permuted` (the exposure permuted: the teach turns, or for
`val_heldout` the learned map); and for C4 one INFORMATIONAL arm never counted (`dr2_off`: the raw-norm lookup
oracle; `learned_vocab`: the default-off candidate). 34 arms per seed (cat_disc 5, cat_told 5, word 5, lived 7,
val_heldout 6, val_novel 6), 204 jobs.

| group | lesion arm | what it cuts | how (production knob or in-process patch) |
|---|---|---|---|
| cat_disc, cat_told | `lesion_draw` | the vocab-agnostic spiking DRAW's likelihood (uniform drive) | `BRAIN_SPIKING_DRAW_LESION=1` (production flag) |
| word | `lesion_recruit` | recruit-an-assembly: no reserved cleanup slot is ever claimed for a never-seen word | in-process: `OneBrainComposer._recruit_word` returns False (the pool reads as exhausted); the arm records the call count, which must be > 0 |
| lived | `lesion_recall` | the substrate-first spiking recall read | `BRAIN_SUBSTRATE_RECALL_LESION=1` (production flag; `ChatBrain._substrate_recall` returns None) |
| lived | `lesion_episodic` | the D5 hippocampal dAP episodic recall gate | `BRAIN_EPISODIC_LESION=1` (production flag) |
| val_heldout, val_novel | `lesion_affect` | the graded-affect ladder read (affect_out clamped to 0) | `BRAIN_AFFECT_LESION=1` (production flag) |
| val_heldout (permuted) | `permuted` | the learned map's values deranged across the 24 held-out words (seed-specific) | in-process: `affect_production_organ._LEARNED_VALENCE` replaced after load |

**Declared departures, identical in every arm of the group (including intact_a):** (a) group `lived` sets
`BRAIN_EPISODIC_STORE=1`, the D5 write's cupy-default value forced on numpy (the production GPU deployment runs it;
on numpy production defers it as a declared latency residual); the other groups never ask a referential question,
so the deferred write cannot touch their measured replies. (b) No seed threading into the three host-built organs
that ignore `BRAIN_CHAT_SEED` at HEAD (value-choice critic, BG selector, shared salience; FAILURE_LOG 2026-09-29
row): production is left as it is, so those three organs are seed-invariant here and the six seeds vary the
substrate, composer, ladder and every per-session organ. (c) `PYTHONHASHSEED=0` (value choice's `_stable_seed`).
(d) The corpus-learned production organs (the one-brain cross-edge pool, the learned animacy lexicon) read
`data/corpus/tinystories.txt` from the repo root and DEGRADE SILENTLY without it (found in the first dev smoke: a
git worktree has none, the file is git-excluded); every arm records `corpus_present`, the aggregate requires it,
and the HANDOFF checks each pool node's revision directory for the file after provisioning. No `sim/` or
`webapp/` file is edited by any arm.

## 4. Per criterion: protocol, measurement, anti-cheats, bar, prediction

### 4.1 C1, category held-out inheritance (`cat_disc` primary; `cat_told` diagnostic)

**Protocol.** Eleven teach turns interleaving the two categories (`the robin eats the worm`, `the trout eats the
bug`, `the robin flies in the sky`, `the trout swims in the lake`, ... `the finch flies in the sky`) and last `the
canary flies in the sky`. The canary is never told what it eats. Then: `what does the canary eat` (strict route),
`what might the canary eat` (generative route), `what does the robin eat` and `what does the trout eat` (identity
controls), `what does the canary swim` (moat: never taught, not inheritable).

**Reply-level measurement.** The patient the reply names on the two inheritance probes (`recalled_svo[2]` or
`hypothesis_svo[2]`, both rendered in `answer`; on the generative route the reply is a flagged hypothesis, "perhaps
the canary eats the worm"). `inherits` = either route names `worm`. Internal secondary: `activity` (the matched
block), the DRAW organ's trace (`spiking_draw`), the plausibility gate's decision.

**Anti-cheats.** (1) Permuted exposure: the canary streamed with the fish context (`the canary swims in the lake`);
the reply must name `bug` or abstain, never `worm` (rung-3c's "the inference follows the learned co-occurrence").
(2) Lesion `lesion_draw`: the generative reply must change (a uniform likelihood no longer prefers the category-
consistent patient) on 6/6 and reproduce. (3) Host-shortcut audit (section 5): the strict route's (agent, action)
match is host first-match over spiking decodes (k-select); the generative route's plausibility likelihood and
novelty exclusion are host code over the brain's own fact graph; a PASS via the generative route is therefore
reported with that host share attached, never as neural ownership.

**Bar.** 6/6 seeds `inherits`; permuted tracks 6/6; lesion effect 6/6 reproduced. No roadmap number exists for
this criterion; 6/6 follows the six-seed rule and EMERGE-26's own 1.00/0.00 contrast.

**Prediction: FAIL, on both routes.** Strict route: exact-match recall abstains (no `(canary, eat)` block exists;
no route performs `isa` or context inheritance at production defaults). Generative route: `_plausible` requires
`_related(canary, eat)` on the co-occurrence graph built from the brain's facts; `canary` co-occurs only with
`fly`/`sky`, so the proposer most likely abstains; if it does propose, it may pick `worm` by verb->patient
frequency alone (both `eat` patients are equally frequent here, so that shortcut is at chance). A FAIL names the
gap exactly: none of the four de-risked inheritance mechanisms (EMERGE-26, rung-3c, emergent clusters, the
conversation rung) is wired into `/api/brain-chat`; the next method is to wire rung-3c's learned member->context
pathway (or the EMERGE-26 shared superordinate code) behind a default-off flag and re-run this group at dev seeds
before any flip. `cat_told` separates the two halves: if it also fails, the missing piece is the inference step
itself; if `cat_told` passes and `cat_disc` fails, the missing piece is discovery.

### 4.2 C2, word->concept recall (`word`)

**Protocol.** `what does the bear chase` (pre-exposure: must abstain); `the otter eats the clam`; `the bear chases
the otter`; then the six probes in section 1. `otter`, `clam` and `bear` are absent from the tiny-demo vocabulary
and from the shipped LTM core (checked 2026-09-29 against the `wikidata_core_15k` bundle's facts file: 0 hits each).

**Reply-level measurement.** `recalled_svo` / `abstained` / `answer` on each probe. Internal secondary: the
composer's recruited-slot count (`_free_slots` consumed), `activity.roles`, the anaphora referent read.

**The anaphora sub-probe is reported, not counted (a dev-seed design decision, declared).** `what does it eat`
after `the bear chases the otter` should answer clam through the recruited word. On dev seed 7 the intact arm
resolved `it` to the built-in `cat` and answered "the cat eats the fish" (section 11): the discourse referent store
is built from the build-time vocabulary (`_build_tiny_demo`: `referents = [w for w in vocab if w not in actions]`),
so a word recruited in-session is never a discourse referent. That is a discourse-WM property (the roadmap's
Stage-2 comprehension list), not the Stage-1 word->concept binding, so it stays in the turn table (it is also an
intervening turn) and is reported as `anaphora_it_eat` beside C2, never in its pass condition. It is the named
residual: the D6 referent store's handling of runtime vocabulary.

**Anti-cheats.** (1) Permuted exposure (`the clam eats the otter`, `the otter chases the bear`): `what does the bear
chase` must NOT answer otter; `what does the otter chase` -> bear; `what does the clam eat` -> otter; `what does
the otter eat` abstains. (2) Lesion `lesion_recruit`: both counted new-word probes change (the recruit pool reads
exhausted, so the outer cleanup never decodes the word) while `what does the dog chase` still answers cat
(specificity), 6/6, reproduced; the arm records that the patch was called. (3) Host audit: the acquisition's
3-token SVO extraction and verb lemmatization are host parsing (declared scaffolds); the store, the recruit, the
bind and the resonate-and-fire recall are on the substrate; the final block choice is the host first-match (k-select).

**Bar.** the counted probes correct on 6/6; pre-exposure abstains 6/6; lesion 6/6 reproduced; permuted tracked 6/6.

**Prediction: PASS on the counted probes** (dev seed 7: bear->otter, otter->clam, both moats, the pre-exposure
abstain and the built-in fact all read as required, and the recruit lesion removed exactly the two new-word answers
while leaving the built-in one). **The reported anaphora sub-probe is predicted 0/6** for the code-level reason
above.

### 4.3 C3, lived-fact recall (`lived`)

**Protocol.** `what does the wolf bite` and `did we discuss the wolf` (both pre-exposure: abstain); `the wolf bites
the apple`; three intervening QUESTIONS (`how is the weather today`, `what is the weather like tomorrow`, `what
does the cat eat`); then `what does the wolf bite`, `did we discuss the wolf`, `what does the wolf chase` (moat),
`what does the dog chase` (built-in). The distractors are questions on purpose: the dev smoke's declarative
`tell me about the sky` was acquired by the host SVO extractor as a junk fact (section 11), which would make the
group's lived experience two facts instead of one.

**Reply-level measurement.** `recalled_svo` on the recall probe; `abstained` on the disclosure probe (the D5 path
answers "I recall discussing the wolf" with `abstained=False`, or "I don't recall" with `abstained=True`). Internal
secondary: `episodic.in_memory` (the dAP completion), `activity`.

**Anti-cheats.** (1) Permuted exposure: `the wolf bites the pear` -> the recall must say pear. (2) Two lesions, each
specific: `lesion_recall` must remove the recall (the fact was lived but the spiking read is cut) and
`lesion_episodic` must remove the disclosure, each 6/6 and reproduced. (3) The pre-exposure probes are the
within-session "never lived" control; a brain that answered them would be answering from a store, not from
experience. (4) Host audit: k-select first-match on the recall; the referential regex and referent extraction on the
disclosure are host parsing; the completion itself is the spiking dAP gate.

**Bar.** all on 6/6, both lesions 6/6 reproduced, permuted 6/6.

**Prediction: PASS for the recall; PASS for the disclosure with the forced write** (the battery's store->recall
drive reads 6/6 at reply level in B2b; dev seed 7 read every C3 check as required, section 11). Risk: the three
intervening turns (one of them a recall of a built-in fact that notes its own topic) are new relative to the
battery's two-turn drive; a FAIL on the disclosure after distractors would name the D5 topic slot's capacity or
recency bias, a Stage-1 fact worth having.

### 4.4 C4, valence inheritance (`val_heldout` primary; `val_novel` diagnostic)

**Protocol (4.4a).** A baseline `how do you feel`; then, in a seed-specific order, each of the 24 held-out words
presented twice as a bare turn, followed by `how do you feel`. The words (frozen, with their Warriner norms copied
from the production lexicon): nice, home, proud, help, brave, sing, gift, friend, warm, smile, kind, happy;
trouble, lonely, bite, cry, scared, mad, bad, hurt, lost, die, evil, war. Each passes the production salience gate
(|v-5| >= 2) and each has its production VALUE inferred by the DR-2 leave-one-out propagation (its own norm never
seeds it: `_affect_learned_valence_map.json`, `meta.note`). Two exposures put the mood EMA (decay 0.4) at 84% of the
word's value and 16% carry-over from the previous word; the seed-specific order turns the carry-over into noise
that averages across seeds rather than a bias.

**Reply-level measurement.** The differential printed in the feel reply ("valence differential <signed number,
three decimals>") and its sign word; r = Pearson over the 24 (reply differential, norm) pairs per seed. Internal secondary:
`affect.differential`, `affect.appraisal_valence`, the ladder's `pos_rate`/`neg_rate`.

**Anti-cheats.** (1) Lesion `lesion_affect`: the ladder is clamped, so every read is neutral: r is UNDEFINED (zero
variance; never reported as 0) or |r| < 0.20 on every seed, and the null-vs-lesion reply text differs on every feel
turn. (2) Permuted learned map: the 24 words' learned values deranged across words (a seed-specific derangement);
the salience gate still admits every word, so the reply still moves, but r must fall below 0.20: the reply valence
rides the learned map, not the gate. (3) The `dr2_off` informational arm reads the raw norm as the value (a lookup
ceiling); reported only, so the inheritance cost is visible. (4) Host audit (section 5): the salience gate, the map
lookup and the EMA are host arithmetic; the differential is the spiking ladder read; the read-out text bands are
host. (5) Known limitation, stated: the production gate admits only strong words, so this r is over a bimodal set
(easier than DR-2's mid-band stress test); sign agreement is reported beside r.

**Bar.** r >= 0.55 on every seed (the roadmap's number, DR-2's gate); lesion and permuted as above.

**Prediction: PASS.** The learned map correlates with the norm at r = 0.91 over the 78 strong words and at
r = 0.90 over the 24 frozen words, with one sign miss (`war`, learned near-neutral; the D1 finding's own
TinyStories ambiguity) (computed 2026-09-29 from the map and the lexicon, the same two files the runner reads);
the read passes through a monotone EMA and a ladder that is graded at the differential level (Gate-B; the dev
smoke's reply text read +0.059 after `sing` and +0.022 after `help`, whose learned value is mild <!--derived-->);
saturation at the extremes
would keep the sign and lower r but not below 0.55 unless the ladder binarizes, which the lesion arm's
zero-variance rule would also expose. A FAIL would name the ladder's dynamic range or the EMA carry-over, both
measurable from the rows.

**Protocol (4.4b, diagnostic).** `the glorp makes everyone wonderful and happy` x3, feel; `the blick makes everyone
terrible and sad` x3, feel; `glorp` alone, feel (the mood was left negative by blick; an inherited positive valence
must move it up); the glorp company again x3, feel; `blick` alone, feel (must move down). Permuted: the company is
swapped, the probe words and positions are not, so both expected moves reverse. Lesion: every read neutral.
**Prediction: FAIL at production defaults for a code-level reason** (a word outside the Warriner list contributes
exactly zero; a zero-hit turn holds the prior mood), run because the calendar rule says so and because it measures
the integrated brain's in-session affect learning, which the roadmap lists under Stage 1 ("core-affect valence
tagging from reward context"). The `learned_vocab` informational arm shows whether the default-off candidate would
pass; a FAIL names it as the integration.

## 5. Host-decision seams these answers pass through (the host-shortcut audit)

The plan (`2026-09-25-prove-who-owns-the-computation-PLAN.md` sections 1-2) requires a host-share beside every
capability number. The instrument (`research/runners/host_share.py` + `host_share_schema.py`, branch
`research/host-share-baselines`, prereg `2026-09-29-host-share-baselines-PREREGISTRATION.md`) is not on main at
filing, so this gate REPORTS the seam audit below and does not compute F/S/D. Running that instrument's intervention
arms on these probe turns is NEEDS-CLAUDE #8 in the HANDOFF; when it lands, its schema is imported, not re-derived.

| criterion | host code that touches the answer (file) | neural state before it | seam class (host-share prereg) |
|---|---|---|---|
| C1 strict | `OneBrainComposer._fact_shard_first_match` (first match over the spiking decodes); the (agent, action) question parse (`ChatBrain._neural_question_parse` on the BridgeParser, host fallback) | the DG-CA3 shard's decoded blocks | k-select (content) |
| C1 generative | `_parse_open_ended` regex; `_generate_hypothesis`'s `_plausible` likelihood and novelty exclusion (`brain_chat_tui.py`); the hedged-transitive template | the vocab-agnostic spiking soft-WTA DRAW off `cp_firing_states` | candidate admission/filtering (host), winner (spiking) |
| C2, C3 recall | `_maybe_acquire` SVO extraction + `lemma_verb` (store side); k-select first-match; organ-C set-membership vote; metacog band arithmetic; the consensus `argmax` (`norgan_hop`) | the resonate-and-fire recall, the recruited cleanup slot, the D4 sel-pool margins | k-select (content); organ-c, metacog, ignition (commitment, call-graph checks) |
| C3 disclosure | `is_referential` regex + `extract_referent` (first known topic token); `recall_disclosure` text | the D5 dAP completion (`in_memory`) | routing (host) -> commitment (spiking gate) |
| C4 | `appraise_text` salience gate (raw norm) + learned-map lookup; `_update_session_mood` EMA; `feel_readout` bands and the printed number | the graded-affect ladder's opponent read (`read_differential`) | candidate scoring (host) -> read (spiking) -> composition (host) |
| all | `bg_action_selection` pre-filter (content turns never reach the race); `affect_drives` / `da_drives` suffixes (inert at level 0) | -- | commitment (bg-select) |

Expectation, written before the gate: every C2/C3 answer is content-selected by the host first-match over a
spiking decode (the host-share prereg's k-select "match" events are predicted HOST-OWNED); C4's number is a
spiking read of a host-computed drive; C1's generative answer, if any, is host-admitted and spiking-drawn.

## 6. Mechanisms expected to be NECESSARY (six-seed ablation; CLAUDE.md non-negotiable 4)

| criterion | mechanism | ablation arm | predicted reply-level effect of the ablation |
|---|---|---|---|
| C1 | the spiking DRAW likelihood | `lesion_draw` | UNDEFINED if intact abstains (nothing to remove); otherwise the hypothesis changes or vanishes |
| C2 | recruit-an-assembly (reserved cleanup slots) | `lesion_recruit` | every new-word answer becomes an abstain or a wrong recall; built-in answers unchanged |
| C3 | the substrate-first spiking recall | `lesion_recall` | the lived fact is not recalled (abstain or the host router's stale answer) |
| C3 | the D5 dAP episodic gate | `lesion_episodic` | the disclosure becomes "I don't recall" |
| C4 | the graded-affect ladder | `lesion_affect` | every feel read neutral; r UNDEFINED |
| C4 | the DR-2 learned map (inheritance) | `permuted` | r collapses below 0.20 while the reply still moves |

Not ablated here, named for the follow-up: the on-brain BridgeParser question parse (`BRAIN_SUBSTRATE_PARSE_LESION`),
the D6 referent store on the anaphora turn (`BRAIN_MULTIREF_LESION`), the K-way sequencer (off by default).

## 7. Validity, UNDEFINED handling and the verdict rules (frozen; scored only by `--aggregate`)

- **Defined seed** (per group): every COUNTED arm of the group present and readable (`intact_a`, `intact_b`, each
  lesion arm and its `.rep0` repeat, `permuted`; informational arms excluded); the null clean (every primary reply
  field of every turn equal between `intact_a` and `intact_b`); every response's `source` = `tiny-demo +LTM`;
  `PYTHONHASHSEED=0` and `corpus_present` recorded. A seed failing any of these is UNDEFINED for that group, with
  the reason recorded; UNDEFINED is never counted as a pass or a fail (a missing arm is an instrument failure, never
  a FAIL clause).
- **Lesion effect** = a primary reply field differs between `intact_a` and the lesion arm on the scored turn(s)
  (C1: both inheritance probes; C2: the two counted new-word probes; C3: the recall probe for `lesion_recall`, the
  disclosure probe for `lesion_episodic`; C4: the r rule of section 4.4). **Reproduced** = the lesion arm and its
  `.rep0` agree on every primary field of every turn. An in-process lesion whose patch was never called is
  `applied=False` and voids that arm.
- **Permuted tracks** = the permuted arm's replies follow the permuted exposure (the group's rule in section 4).
- **Group verdict:** INCOMPLETE if fewer than 6 defined seeds; else PASS iff correct 6/6 AND every lesion arm has
  its effect 6/6 AND reproduced 6/6 AND permuted tracks 6/6; else FAIL with the failing clause named
  ("capability absent on k of 6", "not load-bearing", "not exposure-driven").
- **Criterion verdict** = its primary group's verdict. Diagnostic groups and informational arms are reported and
  never counted.
- **Stage verdict:** the aggregate carries a `preconditions` block (`tools.verdict.Verdict`): no primary shard
  missing, null clean everywhere, expected source everywhere, hash seed, the corpus present in every primary arm,
  every in-process lesion applied, six defined seeds per primary criterion. Any failed precondition makes the stage UNDEFINED (INCOMPLETE), with the
  per-group numbers still reported; else PASS iff all four criteria PASS, else FAIL.
- **Numbers:** r over fewer than 12 parsed feel reads is UNDEFINED; a zero-variance r is UNDEFINED ("no variance
  in the reply differentials"), never 0; a missing probe field is UNDEFINED for that check.
- **Amendments:** any change after the first gate job is dispatched is appended to section 12, dated, before the
  aggregate is read; a changed bar voids the gate.

## 8. What each outcome means

| criterion | PASS means | FAIL means (a first measurement, not a stop) and the gap it names | next method (no-defer) |
|---|---|---|---|
| C1 | the integrated brain assigns a never-told property from a discovered category, through one of its own routes, and that route is load-bearing and exposure-driven | the inference step (both routes abstain: no inheritance route in the live handler) or the discovery step (`cat_told` passes, `cat_disc` fails) | wire rung-3c's learned member->context pathway (default-off) into the strict route's recall miss; re-run this group at dev seeds |
| C2 | a word never seen is bound to a concept in-session and reused in both roles with the moat intact, the recruit pool is necessary; the reported anaphora sub-probe says whether the discourse WM admits a recruited word (predicted: it does not) | which probe fails names it: the recruit itself, the agent-role decode, or the moat on the new word | the named organ's runtime-vocabulary path; for the reported anaphora residual, the D6 referent store's admission of recruited words |
| C3 | a fact lived this session survives three unrelated turns and the episode is disclosed, both spiking reads necessary | recall after distractors (composer capacity / interference) or the disclosure (D5 topic slot recency) | measure the capacity curve (turns between exposure and probe) |
| C4 | the reply-visible felt valence of held-out words tracks human norms through inherited (never seeded) values, the ladder is necessary, the map is necessary | r below 0.55: the ladder's range or the EMA (rows show which); `val_novel` FAIL (predicted): no in-session inheritance at defaults | flip-candidate `BRAIN_AFFECT_LEARNED_VOCAB` after its own prereg passes; then re-run `val_novel` |
| stage | Stage 1 exit on the integrated brain, first ever | the per-criterion table above is the Stage-1 checklist the roadmap's Phase C exit needs | the failing criteria become the next integration slots in ledger order |

## 9. Predictions (written before the gate; expectations, not bars)

| criterion / group | predicted verdict | reason |
|---|---|---|
| C1 `cat_disc` | FAIL (both routes abstain) | no inheritance route at production defaults; the generative route's plausibility gate needs a `canary`-`eat` co-occurrence that was never given |
| C1 `cat_told` | FAIL (inference step), premises recalled | `isa` facts store and recall directly; nothing composes them |
| C2 `word` | PASS (counted probes); the reported anaphora sub-probe 0/6 | integrated 2026-08-12; dev seed 7 read every counted probe as required and the anaphora miss as a build-time referent store |
| C3 `lived` | PASS | reply-load-bearing 6/6 in B2b for the store->recall drive; distractors are the new element |
| C4 `val_heldout` | PASS (r >= 0.55 every seed) | map-vs-norm r = 0.91 over the strong set; a monotone graded path |
| C4 `val_novel` | FAIL (defaults); the `learned_vocab` arm may inherit | a non-lexicon word contributes zero by code |
| stage | FAIL (C1) | one criterion predicted absent; this FAIL is the measurement the calendar asked for |

## 10. Cost and ETA (pool)

204 jobs (6 seeds x 34 arms). A tiny-demo+LTM build is ~6.1-7.7 GB RSS (this smoke, host-share dev, B2b), so
`mem_gb=7`, `memcap 9`, one to two jobs per 15 GB pool node. Measured on dev seed 7 (numpy, one thread, this
workstation, section 11): the build 150-330 s, the first content turn 120-460 s (lazy organ builds), a teach or
recall turn 10-47 s, a bare-word exposure turn ~8 s, a feel turn ~2 s; in group `lived` the forced BTSP write adds
~500-1100 s to the teach turn and to each later verified recall (a `lived` arm took 2813 s). Planning figure per
job on a pool node (~1.4x slower than this workstation, B2d's figure): `cat_disc` ~25 min, `cat_told` ~20 min,
`word` ~21 min, `lived` ~65 min, `val_heldout` ~17 min, `val_novel` ~11 min; about 95 job-hours in total, half of
it `lived`. Pool only (pool41 + pool42, 2-4 concurrent): about 25-50 h of wall time once the lines are reached, no
cloud cost; the shared queue held 330 lines at filing (215 long-delay-credit sweep lines, 104 host-share gate
lines), so on the pool alone the gate starts one to three days after queueing. AWS (r7i.4xlarge, ~16 jobs per node
under the $50/day cap): about 6 h on one node or 3-4 h on two, about $7-10, the lane to use if the October window
is at risk (HANDOFF NEEDS-CLAUDE #13).
GPU: not needed; a cupy replication of `lived` without the forced write is an optional follow-up through
`tools/gpu_queue.sh`, never a gate arm.

## 11. Dev verification on the final code (dev seed 7, numpy, not a verdict)

Recorded from the smoke runs before this file was committed (`research/findings/raw/_stage1_exit/dev/s7/`; every
shard: source `tiny-demo +LTM`, `pythonhashseed_ok` true, `corpus_present` true, no errored turn, per-turn timings
recorded). Dev seeds govern nothing; these are integrity smokes plus the first reads behind section 9.

- `val_heldout__intact_a_2words.json` (labels `feel_base` .. `help_feel`, the first two words of the seed-7 order):
  305 s wall (build 153 s, first content turn 124 s, an exposure turn ~8 s, a feel turn ~2 s). The baseline feel
  reply reads neutral at +0.00; after `sing` x2 it reads positive at about +0.06; after `help` x2 about +0.02
  (neutral band) <!--derived-->. Both reads parse to a differential and a sign word; the bare-word exposure turns
  abstain with the curiosity follow-up, as expected for a word with no stored fact.
- `word__intact_a.json`: 886 s wall. `what does the bear chase` -> abstain before exposure; after the two teach turns
  -> "the bear chases the otter" (`recalled_svo` [bear, chase, otter]); `what does the otter eat` -> [otter, eat,
  clam]; `what does the otter chase` and `what does the clam eat` -> abstain; `what does the dog chase` -> [dog,
  chase, cat]. `what does it eat` -> "the cat eats the fish" (the anaphora miss of section 4.2). Two turns carried
  lazy organ builds (385 s and 168 s); the others 11-40 s.
- `word__lesion_recruit.json`: 811 s wall; `patch.calls` 21. Both counted new-word probes -> abstain; the moats and
  the built-in fact unchanged; `what does it eat` -> the same built-in "the cat eats the fish" as intact (so that
  turn does not separate the arms, one more reason it is not counted). The lesion contrast on the counted probes:
  `answer`, `abstained` and `recalled_svo` all differ on `p_bear_chase` and `p_otter_eat`; nothing differs on
  `c_dog_chase`, `c_otter_chase`, `c_clam_eat`.
- Corpus finding (first attempt, discarded): the same two `word` arms first ran in the worktree WITHOUT
  `data/corpus/tinystories.txt` and the server logged "ONEBRAIN XEDGE build FAILED -> degrading to standalone
  organs" plus a learned-animacy traceback, with the turns still answering. Those shards were discarded and the
  `corpus_present` precondition added (section 3 (d)).
- `cat_disc__intact_a.json`: 1045 s wall (build + first teach 464 s, the other teach turns 33-47 s, the probes
  10-17 s, one lazy organ build 142 s). All eleven teach turns stored ([robin, eat, worm] ... [canary, fly, sky]).
  `what does the canary eat` -> abstain with the recall path exercised (`activity.roles` = canary/eat, no matched
  block); `what might the canary eat` -> abstain with `activity` null, i.e. `gate()` returned from the open-ended
  branch before the recall path: the generative route fired and proposed nothing (the curiosity follow-up's
  "I haven't learned about might" is the host topic extractor reading the second word of the question, a wording
  wart the scorer never reads). Identity controls robin -> worm and trout -> bug hold; `what does the canary swim`
  abstains. Exactly the predicted C1 FAIL shape (section 4.1): the instrument reaches both routes and both are
  empty at production defaults.
- `lived__intact_a.json`: 2813 s wall (build + first turn 326 s; the teach turn 1075 s and the two later verified
  recalls 689 s and 514 s carry the forced numpy BTSP writes; the disclosure turn 121 s; the rest 9-30 s). Both
  pre-exposure probes abstain; after the three distractors `what does the wolf bite` -> [wolf, bite, apple];
  `did we discuss the wolf` -> "Earlier you brought up wolf -- my hippocampal readout completes its assembly ..."
  (`abstained` false, `episodic.in_memory` true); `what does the wolf chase` abstains; `what does the dog chase`
  -> [dog, chase, cat]. Every C3 check as required. Instrument wart found and fixed before this commit: the
  distractor `tell me about the sky` was acquired as the junk fact [tell, about, sky] ("the tell abouts the sky"),
  so the frozen distractors are now all questions (section 4.3; FAILURE_LOG 2026-09-29).
- `val_novel__intact_a` (dev seed 7) runs after this file is committed; its PASS-list line goes to
  `docs/drafts/stage1-exit/WORKLOG.md` and the HANDOFF's section 2, never into this file, so the prereg stays
  byte-identical from its commit to the gate. If it reveals an instrument defect (not a capability result), the
  fix is filed as a dated amendment in section 12 in its own commit before any gate seed.

## 12. Amendments

(Appended dated entries only. The bars, arms and turns above are frozen at the commit that adds this file.)
