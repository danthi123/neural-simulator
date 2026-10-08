---
type: finding
status: live
date: 2026-09-30
lane: integration-first (Stage-1 exit criterion C1; stage exits on the calendar, Stage 1 by 2026-10-24)
mechanism: PRE-REGISTRATION of Stage-1 criterion C1 (category held-out inheritance) measured on the integrated production brain with a DEFAULT-OFF wiring switched on -- a spiking category-inheritance organ (rung-3c's learned member->context pathway, plus a learned frame->frame association in place of rung-3c's hand-placed context->property projection) read on the strict route's recall miss -- scored by the Stage-1 exit gate's own C1 group and scorer, with a null arm, a reply-level lesion arm of the responsible faculty (repeated), a permuted-exposure arm, and four informational ablations
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION only. Filed before any gate-seed arm was built. No result is claimed here.
runner: research/runners/stage1_c1_category.py
artifacts:
  - research/runners/category_inherit_organ.py
  - webapp/category_inherit_chat.py
  - research/runners/stage1_c1_category.py
  - tests/test_category_inherit_organ.py
  - research/packets/stage1-c1-category/HANDOFF.md
---

# Stage-1 C1 (category held-out inheritance) with the category-inheritance wiring: PRE-REGISTRATION (2026-09-30)

**Committed alone, before any gate-seed (42/43/44/100/101/102) arm is built.** Branch `research/stage1-c1-category`.
The gate runs at a revision that contains this file byte-unchanged plus the runner; the pin is recorded in
`research/packets/stage1-c1-category/PINS.tsv` and checked by the HANDOFF. Every pool job line carries the pin and
`--prereg` (the gate-seed guard, `tools/gate_seed_guard.py`).

**Why.** The Stage-1 exit prereg (`2026-09-29-stage1-exit-gate-PREREGISTRATION.md`, branch
`research/stage1-exit-gate`) predicts C1 FAILS at production defaults: the live `/api/brain-chat` handler has no
inheritance route (its dev seed 7: `what does the canary eat` abstains on the strict route, `what might the canary eat`
abstains on the generative route). Its section 8 names the next method: "wire rung-3c's learned member->context
pathway (default-off) into the strict route's recall miss; re-run this group at dev seeds"; its HANDOFF NEEDS-CLAUDE
#12 asks for the choice between rung-3c and the EMERGE-26 shared superordinate code. Owner rules in force:
integration-first (the unit of progress is a faculty on by default whose lesion changes the reply), the stage-exit
calendar, pivot dwell. This document makes the choice (section 2), freezes the measurement, and predicts it.

**What this is not.** Not a flip: nothing here changes a production default (the anchor
`_CATEGORY_INHERIT_DEFAULT_ON = False`, `webapp/category_inherit_chat.py`). Not a Stage-1 verdict: the Stage-1 gate's
C1 arms run at production defaults on its own branch and are untouched. Not a claim that the organ is part of the one
spiking substrate: it is a co-resident standalone bridge (section 12). The verdict vocabulary is PASS / FAIL /
UNDEFINED(INCOMPLETE); a FAIL is a measurement, never a stop (CLAUDE.md non-negotiable 3).

## 1. The criterion, unchanged

The Stage-1 prereg's C1 group `cat_disc` is imported, not copied (`stage1_exit_gate.GROUPS["cat_disc"]`, turns and
permuted turns; `stage1_exit_gate.score_cat`, `null_clean`, `reply_differs`, `n_field_diffs`). Eleven teach turns
build a bird context (`eat worm`, `fly sky` for robin, sparrow, finch) and a fish context (`eat bug`, `swim lake` for
trout, salmon); last, `the canary flies in the sky`. Probes: `what does the canary eat` (strict route), `what might
the canary eat` (generative route), the identity controls `what does the robin eat` / `what does the trout eat`, the
moat `what does the canary swim`. **Correct** = either inheritance probe names `worm` (`score_cat(...)["inherits"]`).
**Permuted** = the canary is streamed with `the canary swims in the lake`; **tracks** = either probe names `bug`
(`score_cat(..., permuted=True)["inherits"]`), defined only when intact is correct (the Stage-1 aggregate's rule).
The one change from the Stage-1 C1 row is the lesion arm, and it follows the Stage-1 prereg's own rule ("a lesion arm
per responsible faculty"): the responsible faculty here is the category organ, so the counted lesion is
`lesion_inherit`; the Stage-1 arm `lesion_draw` (the generative DRAW) is kept as an informational arm (section 3).

## 2. The choice: rung-3c, extended; not EMERGE-26 (the record, read first)

`bash tools/before_you_build.sh` was run 2026-09-30 (logged to the shared corpus-check log; top hits: the rung-3c,
EMERGE-26 and EMERGE-30 findings) plus a RAG query for production inheritance; the three findings and both runners
were read in full.

- **EMERGE-26** (`2026-07-02-emerge26-emergent-inheritance-GO.md`, runner `_emerge26_emergent_inheritance_derisk.py`):
  inheritance with cancellation on the HTM cortex bridge, 6/6, but the shared superordinate is **host-assigned**: the
  runner's `SUPER = {"robin": "BIRD", ..., "trout": "FISH"}` is a dict literal, and the finding's own honest scope says
  "inference-OVER-structure, NOT acquisition-OF-structure-from-experience". C1's primary group is `cat_disc`, whose
  category is DISCOVERED from shared contexts and never told. Wiring EMERGE-26 into it would need a host categorizer to
  hand the canary its BIRD block, which is the host lookup this wiring must not contain. EMERGE-26 fits the told form
  (`cat_told`, the Stage-1 diagnostic), not the discovered one.
- **EMERGE-30** (`2026-07-02-emerge30-emergent-superordinate-GO.md`) removes EMERGE-26's caveat by learning
  member->context from co-occurrence, i.e. by rung-3c's rule; it then TEACHES the class property onto the emergent
  context cells ("bird-context flies"), a teaching form C1's turns do not contain (properties are taught per member).
- **Rung-3c** (`2026-07-07-GNW-rung3c-learned-category-inheritance-6of6-heldout-5of6-identity.md`, runner
  `_gnw_rung3c_learned_inheritance_derisk.py`): the member->context pathway is LEARNED by the on-bridge rate-Hebbian
  from co-occurrence (floor 0.010 before streaming, canary->BIRD ~16 after, canary->FISH at the floor <!--derived-->,
  quoted from that finding); the held-out
  canary streamed only with the bird context inherits 6/6; the permuted context makes it infer the displaced property
  (exactly C1's permuted arm); no learning abstains. Its residual: the context->property projection is fixed by hand
  (`CONTEXT_TO_REASON_W`).
- **Decision: rung-3c, with its residual closed.** The organ keeps rung-3c's learned member->context pathway and its
  two fixes (zero Hebbian decay, strong co-drive), and replaces the hand-placed context->property projection with a
  frame->frame association learned by the same rule. In C1 the "context" and the "property" are both predicate frames
  (`fly sky`, `eat worm`), so one learned association between frames that the same members carry does the job, and no
  category, member list or property is placed by hand. Pivot-dwell note: this retires nothing and is not a pivot; it is
  the next method the Stage-1 prereg named.

## 3. The wiring and the harness (frozen)

**Organ** (`research/runners/category_inherit_organ.py`, parameters frozen at the revision in PINS.tsv). One
standalone Izhikevich SimulationBridge: 24 member slots and 24 frame slots of 20 neurons; member->frame all-to-all and
frame->frame (different slots) all-to-all, both floor 0.01, each on its own plasticity gate, open only while a heard
fact is rehearsed (12 presentations of 20 steps, member 2500 pA + frame 2200 pA, each from the quiescent state);
on-bridge Hebbian rate 0.05, soft bound 40, decay 0; per-neuron heterogeneity seeded by `cfg.seed`; no OU noise. Read:
the asked member driven 2500 pA for 60 steps, spikes counted per frame slot; frames whose verb is the asked verb are
admitted; the winner must fire >= 0.02 spikes/neuron/step and beat the runner-up by 1.25x, else abstain.

**Wiring** (`webapp/category_inherit_chat.py`, installed by `webapp/server.py::brain_reply` outside every gate wrapper
when `BRAIN_CATEGORY_INHERIT` is truthy). An acquired AFFIRM fact (the `('done', svo)` branch of
`ChatBrain.gate_extract`) is observed. When the production gate stack returns None for a routable, non-anaphora
strict recall (`gate_extract` returned `('route', q, agent, action, False)` this turn), the organ is read; a reached
object returns as an `InheritedSVO`, a `HypothesisSVO` subclass: every surface renders it as a flagged guess, the rich
path reports it as `hypothesis_svo` with `recalled_svo` null. Each flag-on response carries a `category_inherit` trace
(the observe record, the read with admitted rates, the lesion mode, `lesion_held`, and a per-read pathway attribution).

**Harness**: `stage1_exit_gate.BASE_ENV` exactly (numpy/CPU, one BLAS thread, stub renderer, no LLM, tiny-demo brain
with the shipped LTM tier, `PYTHONHASHSEED=0`, `rich` omitted = the production default path), `BRAIN_CHAT_SEED=<seed>`,
one fresh process and one fresh ChatBrain per arm. The flag is set explicitly in every arm (`1`, or `0` in `off`).

| arm | env (beyond the harness) | role |
|---|---|---|
| `intact_a` | `BRAIN_CATEGORY_INHERIT=1` | counted: correctness |
| `intact_b` | same | counted: the null (an independent same-seed rebuild) |
| `lesion_inherit`, `lesion_inherit.rep0` | `+ BRAIN_CATEGORY_INHERIT_LESION=1` | counted: both learned pathways silenced at read time (weights zeroed for the read, restored after; `lesion_held` must read true) |
| `permuted` | flag on, the permuted turns | counted: exposure tracking |
| `off` | `BRAIN_CATEGORY_INHERIT=0` | informational: the production default (the Stage-1 C1 shape) and the specificity check |
| `lesion_lateral_learn` | `+ LESION=lateral_learn` | informational: the frame->frame association never forms (plasticity blockade) |
| `lesion_lateral_read` | `+ LESION=lateral` | informational: frame->frame silenced at read only |
| `lesion_draw` | `+ BRAIN_SPIKING_DRAW_LESION=1` | informational: the Stage-1 prereg's frozen C1 lesion with the wiring on |

Nine arms per seed, 54 jobs. No `sim/` edit; no arm injects an answer, a code, a weight or a store entry.

## 4. Measurement, validity and verdict (frozen; scored only by `--aggregate`)

- **Defined seed**: every counted arm present and readable; the null clean (every primary reply field of every turn
  equal between `intact_a` and `intact_b`); every counted arm with `source = tiny-demo +LTM`, `PYTHONHASHSEED=0`, the
  corpus present, no errored turn; the wiring applied (every flag-on turn carries the `category_inherit` trace; the
  lesion arms' direct-probe read reports the lesion mode with `lesion_held` true). A seed failing any is UNDEFINED
  for C1 with the reason recorded; UNDEFINED is never a pass or a fail.
- **Lesion effect** = a primary reply field differs between `intact_a` and `lesion_inherit` on a C1 probe; **reproduced**
  = `lesion_inherit` equals `lesion_inherit.rep0` on every primary field of every turn. Attribution:
  `tools.lab.attributable_to` of the intact-vs-lesion field-difference count against the intact_a-vs-intact_b count.
- **Verdict**: INCOMPLETE if fewer than 6 defined seeds; PASS iff correct 6/6, lesion effect 6/6, reproduced 6/6,
  permuted tracks 6/6; else FAIL naming the clause ("capability absent on k of 6", "not load-bearing", "not
  exposure-driven"). The earned verdict carries a `tools.verdict.Verdict` preconditions block; a failed precondition
  makes it UNDEFINED (INCOMPLETE).
- **Flip-readiness** (reported beside the verdict, never part of it): READY iff C1 PASS and, on every defined seed,
  (a) safety: robin -> worm and trout -> bug on the identity controls and `what does the canary swim` abstains, and
  (b) specificity: `off` vs `intact_a` differ in reply CONTENT (`abstained`, `recalled_svo`, `hypothesis_svo`) on no
  turn except `p_canary_eat` (answer-text differences on later turns, e.g. the discourse preamble, are reported).
- **Amendments**: any change after the first gate job is dispatched is appended to section 13, dated, before the
  aggregate is read; a changed bar voids the gate.

## 5. Host-decision seams on this route (the host-shortcut audit)

| step | host code | neural state before it | seam class |
|---|---|---|---|
| the fact reaches the organ | `_maybe_acquire` SVO extraction + `lemma_verb`; B3 polarity read; slot recruitment (word -> slot index) | none (store side) | routing / allocation (host) |
| the miss is detected | the gate stack's own veto (3-organ consensus, spiking) read from `gate_extract`'s route tuple | the bus organs' reads | routing (host) after commitment (spiking) |
| which frames the member reaches | none: spikes through the learned member->frame and frame->frame synapses | the organ's firing | -- (spiking) |
| the answer | admission by the asked verb; the floor and the 1.25x margin; the winner's object as the answer | per-frame spike counts | k-select (content) |
| the reply | `render_hypothesis` (the spiking Broca mouth for a supported SVO, else the flagged template) | -- | composition |

Expectation, written before the gate: the answer is host-admitted and spiking-selected, the same class as the strict
route's first match over spiking decodes; the host-share instrument's intervention arms (when on main) should attribute
the frame choice to the organ and the admission to the host.

## 6. Mechanisms expected to be NECESSARY (CLAUDE.md non-negotiable 4)

| mechanism | ablation arm | predicted reply-level effect |
|---|---|---|
| the learned member->frame + frame->frame pathways (read) | `lesion_inherit` (counted) | the direct probe abstains, 6/6, reproduced |
| the learned frame->frame association (learning) | `lesion_lateral_learn` | the direct probe abstains 6/6: without it the canary reaches only `fly sky` |
| the frame->frame pathway at read time | `lesion_lateral_read` | NOT necessary: `worm` survives 6/6, because the canary->`eat worm` link is written during encoding (pattern completion while the canary's fact is rehearsed); the per-read attribution puts about a quarter of the winner's spikes on the lateral read (organ dev, section 11) |
| the generative DRAW | `lesion_draw` | no change on either probe, 6/6 (this route never calls the DRAW) |

## 7. Frozen bars

C1 PASS: correct 6/6, lesion effect 6/6, reproduced 6/6, permuted tracks 6/6, with six defined seeds. The Stage-1
criterion has no roadmap number; 6/6 follows the six-seed rule and rung-3c's own contrast. Flip-readiness: safety 6/6
and specificity 6/6 on top of the PASS.

## 8. Predictions (expectations, not bars)

| arm / quantity | prediction | reason |
|---|---|---|
| `intact_a` direct probe | a flagged guess naming `worm`, 6/6 (`hypothesis_svo` = canary, eat, worm) | organ dev seeds 1-8 (section 11): `worm` 8/8, `eat bug` 0.0 |
| `intact_a` generative probe | abstains, as at production defaults | the open-ended route is untouched |
| C1 verdict | PASS | the organ is deterministic per seed and the rest of the turn is the Stage-1 harness |
| `lesion_inherit` | abstains 6/6, reproduced | organ dev: abstain 8/8; real handler dev seed 7: abstain, only that turn changes |
| `permuted` | a flagged guess naming `bug`, 6/6 | organ dev seeds 1-8 (section 11): `bug` 8/8, `eat worm` 0.0 |
| null | clean 6/6 | same-seed organ reads are identical (tested) |
| `off` | both probes abstain (the Stage-1 C1 FAIL shape) | the production default |
| flip-readiness | READY | identity controls ride the strict route; the moat frame is unreachable |
| `lesion_lateral_learn` / `lesion_lateral_read` / `lesion_draw` | abstain / worm / unchanged, each 6/6 | section 6 |

## 9. What each outcome means, and the next method (no defer)

| outcome | meaning | next |
|---|---|---|
| PASS + READY | the integrated brain assigns a never-told property from a discovered category, the organ's lesion changes the reply, and the reply follows the exposure | Claude: adversarial verify (`verify-go`), then a flip commit on a flip branch and a production-default C1 re-run under an amendment of the Stage-1 prereg that names `lesion_inherit` (HANDOFF NEEDS-CLAUDE) |
| PASS, NOT READY | C1 holds but the wiring changes another turn's content or breaks the moat/identity | the named turn is the defect; fix on this branch with a dated amendment, re-run dev |
| FAIL, capability absent | the organ read or the miss detection does not reach the reply on k seeds | the per-seed `organ_read_intact` names which: organ below floor/tie (organ dynamics under that seed) or no read (the stack did not miss the way assumed) |
| FAIL, not load-bearing / not exposure-driven | the reply is `worm` without the organ, or ignores the permutation | a frequency shortcut somewhere else in the turn; find it before any flip |
| UNDEFINED | an instrument failure (null, source, corpus, wiring, missing shard) | HANDOFF stop conditions |

## 10. Cost, lane, ETA

54 jobs (6 seeds x 9 arms), numpy, `mem_gb=7`, `memcap 9`, ~25-30 min each on a pool node (the Stage-1 `cat_disc`
planning figure ~25 min plus ~2 min of organ rehearsal), ~25 job-hours. Lane: the pool (pool41, pool42), lines carry
`max_age_h=120` (the queue was ~440 lines, three to four days deep, at filing). Not the GPU queue: this criterion was
frozen on numpy in the Stage-1 prereg, and a cupy build changes the organ stack that authors the recall (the two-organ
bus stays inert on cupy without backend-neutral init), so a cupy result would not be the same criterion. No cloud cost.

## 11. Dev verification (dev seeds only; not a verdict)

Dev seeds govern nothing; these are integrity reads plus the first evidence behind section 8. Code at c081ff8a9 /
ce496d36e (the organ and wiring as frozen here).

**Organ level** (`CategoryInheritOrgan` fed the eleven C1 facts directly; numpy; dev seeds 1-8). Design sweeps first
(scratch runs on a since-removed variant, not saved; quoted): a bipartite member<->frame variant with identical
neurons read the held-out frame NON-MONOTONICALLY in the weight bound (0.000 at 40, 0.050 at 60, 0.000 at 90
<!--derived-->: identical assemblies fire in lockstep and a multi-hop read mode-locks; FAILURE_LOG 2026-09-30); 300 pA
OU noise made every link grow during encoding (every frame ~0.25 <!--derived-->). The frozen design (member->frame +
learned frame->frame, seeded heterogeneity, bound 40, 12 rehearsals, 60-step read), recorded in
`research/findings/raw/_stage1_c1_category/dev/organ_dev_seeds1-8.json` (`summary`): canary `eat worm` 0.1042-0.1192
with `eat bug` 0.0 on all eight dev seeds; permuted canary `eat bug` 0.0908-0.1275 with `eat worm` 0.0, deciding `bug`
8/8; the read lesion `all` abstains 8/8; `lateral_learn` abstains 8/8; the lateral read lesion keeps `worm`
(0.0692-0.0875); same-seed reads identical 8/8; the weight audit reads canary->`fly sky` 20.095-25.394, canary->`eat
worm` 7.247-9.402 (written during encoding), `fly sky`->`eat worm` 31.371-33.362, every cross-category block at the
0.05 floor; the per-read attribution puts 0.2204-0.336 <!--derived--> (rounded from `attribution_lateral_at_read`) of the winning rate on the lateral read. A bridging member
(`bat` eats the bug and flies in the sky) made the canary's `eat` read a tie (abstain) on 4 of 8 seeds and `worm` on
4, and `what does the canary swim` reached `lake` on 8 of 8 (section 12). The C2/C3 turn shapes (`the otter eats the
clam`, `the bear chases the otter`, `the wolf bites the apple`) abstained on every never-taught (agent, verb), 8/8:
the organ does not answer the Stage-1 moats.

**Real handler, dev seed 7, `intact_a`** (`research/findings/raw/_stage1_c1_category/dev/s7/cat_disc__intact_a.json`,
1142.8 s wall; source `tiny-demo +LTM`, PYTHONHASHSEED 0, corpus present, no errored turn). All eleven teach turns were
acquired and observed by the organ (the `category_inherit.observe` trace on each). `what does the canary eat` ->
`hypothesis_svo` [canary, eat, worm], `recalled_svo` null, `abstained` false; the reply reads "(My authorship monitor
reads this as self-generated -- a guess of my own, not something I was taught.) perhaps the canary eats the worm [a
guess from what I've learned -- not something I was taught]" after the discourse preamble; organ read `eat worm`
0.1192, `eat bug` 0.0; attribution: learned pathways own 1.0 of the winning rate, the lateral read 0.2657 <!--derived--> (rounded) (the rest
rides the canary->`eat worm` link written during encoding). `what might the canary eat` abstains (unchanged from
production). `what does the robin eat` -> worm and `what does the trout eat` -> bug (strict route); `what does the
canary swim` abstains (organ: `swim lake` 0.0). Timings (rounded from the shard's `timings_s`) <!--derived-->: build + first teach 455 s, teach turns 35-51 s (the organ adds
about 9 s each), probes 13-22 s, one lazy organ build 178 s.

**Real handler, dev seed 7, `lesion_inherit`** (`research/findings/raw/_stage1_c1_category/dev/s7/cat_disc__lesion_inherit.json`, 1199.1 s wall; same
integrity reads). Every teach turn observed exactly as in `intact_a`; `what does the canary eat` -> "I don't know
about that." (`abstained` true; the organ read with lesion `all`, `lesion_held` true, every admitted frame 0.0). The
primary reply fields differ from `intact_a` on `p_canary_eat` ONLY; the identity controls, the moat and the generative
probe are unchanged. The predicted lesion effect, on one dev seed.

The remaining dev-seed-7 arms (`permuted`, `off`, `intact_b`, and the three informational ablations) are listed with
their reads in the HANDOFF section 0 and the WORKLOG as they land; if one reveals an
instrument defect (not a capability result), the fix is filed as a dated amendment in section 13, in its own commit,
before any gate seed.

## 12. Declared residuals

- **Standalone organ, not the one substrate.** The organ is its own bridge (like the plausibility and surprise organs),
  not co-located on the one-brain merged pool; the next rung is co-location with a cross-edge to the composer.
- **Host admission.** The asked verb admits frames by string (section 5); a spiking relation cue is the next rung.
- **Slot recruitment and the frame token are host** (the same scaffolds as the composer's recruit and the SVO
  extractor); therefore this organ is not "self-organized" in the docs/TERMS.md sense.
- **Only facts acquired this session teach the organ**; the tiny-demo built-ins and the LTM tier do not.
- **A bridging member merges categories.** Organ dev (seeds 1-8) with an added `bat` that eats the bug and flies in
  the sky: the canary's `eat` read tied on 4 of 8 seeds (abstain) and read `worm` on 4; `what does the canary swim`
  then reached `lake` on 8 of 8 through the bat (an over-generalization through the merged category). Not in C1's
  turns; a flip decision must weigh it (HANDOFF NEEDS-CLAUDE).
- **The host-escape path** (`BRAIN_GNW_BUS_HOST=1`, the original `ChatBrain.gate`) never calls `gate_extract`, so the
  wiring does not act there.
- **The Stage-1 prereg's frozen C1 lesion is `lesion_draw`.** With the wiring on and flipped, the Stage-1 aggregate
  would read C1 "not load-bearing" unless the Stage-1 re-attempt amends its C1 lesion arm to the responsible faculty
  (`BRAIN_CATEGORY_INHERIT_LESION=1`); that amendment is Claude's (HANDOFF NEEDS-CLAUDE).

## 13. Amendments

(Appended dated entries only. The bars, arms and turns above are frozen at the commit that adds this file.)

### Amendment 1 (2026-10-06, Claude; RULING c1-hop-first): the C1 gate re-measured with the hop fix

bar-sanity: N/A the bars of section 4 are unchanged; this entry registers a second run of the same 54 lines under one added environment variable and one added engagement precondition

**Why.** The gate above read PASS 6/6 (scored 2026-10-06, `research/findings/2026-10-06-stage1-c1-category-inheritance-RESULT.md`)
WITHOUT hop attenuation. The owner ruled that the hop fix (`BRAIN_CATEGORY_INHERIT_HOP`, built on `claude-review/c1-hop`,
`research/findings/2026-10-06-c1-hop-attenuation-bridging-member.md`) lands before the flip, so the configuration the
flip ships is the organ WITH the fix, which the 09-30 run did not measure. This entry is committed before any of its
data and changes no earlier verdict: the 09-30 PASS stands as the no-hop measurement.

**What runs.** The same 54 lines (section 3: nine arms x seeds 42, 43, 44, 100, 101, 102, the same turns, harness and
flags) with `BRAIN_CATEGORY_INHERIT_HOP=1` added to every line's environment, at the revision Hermes pins after
`claude-review/c1-hop` is merged, emitted by `python -m research.runners.stage1_c1_category --emit-pool-jobs <SHA> --hop`
into `research/findings/raw/_stage1_c1_category/gate_hop/`.

**How it is scored.** `python -m research.runners.stage1_c1_category --aggregate research/findings/raw/_stage1_c1_category/gate_hop --hop --seeds 42,43,44,100,101,102`.
Section 4 verbatim, plus one validity precondition: every arm records `hop_env` "1", and every flag-on arm's
direct-probe organ read records `hop: true`; a seed failing it is UNDEFINED for C1 ("hop attenuation not applied").

**What it decides.** The C1 PASS that the flip (PLAN step C) and Stage-1 attempt 2 rely on is this re-measure's
verdict. PASS: the C1 PASS holds for the hop configuration. FAIL or UNDEFINED after one re-run of missing cells: the
flip is off and the fix is reworked on dev seeds (never retuned on these gate seeds); the 09-30 PASS stays the record
of the no-hop organ only.
