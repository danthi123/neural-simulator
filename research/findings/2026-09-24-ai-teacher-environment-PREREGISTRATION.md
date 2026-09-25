---
type: finding
status: live
date: 2026-09-24
lane: D6-learn-and-grow
mechanism: AI-TEACHER social environment (roadmap P2.1) -- a template teacher whose ONLY channel to the brain is the /api/brain-chat text handler teaches K facts (curiosity answers, a paced lesson, a quiz with corrections); after a sleep-depth idle interval the brain is probed with the teacher absent; recall is attributed to the brain's own in-conversation synaptic write (the D6 local Hebbian store) by freeze / zero / teacher-lesion / permuted-teacher / wrong-teacher arms
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION only (filed before any evaluation run). No result is claimed here.
runner: research/runners/ai_teacher_experiment.py
artifacts: []
---

# PRE-REGISTRATION -- the AI teacher teaches the brain through chat only (2026-09-24)

Committed on its own, BEFORE any evaluation run it governs. The code it governs is commit `77aa1cb30` (branch
`research/ai-teacher-environment`; the teacher landed in `153af81bf`) plus the amendments listed at the end of this
file (none at filing). Terms follow
`docs/TERMS.md`: "learns" and "recalls" below mean that a reply's content changes with a synaptic write, measured by
lesion; they are functional read-outs, not claims of experience.

## Why

Owner direction (2026-09-24): the brain's knowledge must be LEARNED by the brain and grow over time like a human's,
never retrieval from a store the language model pulls from; the Qwen mouth stays fact-free; "I don't know" should
become a learning moment. The master roadmap's P2.1 names the AI-teacher scaffold (the teacher is the SOCIAL
ENVIRONMENT, legitimate host code under the brain-based-only rule; everything between hearing and answering is the
brain's job) with anti-cheats teacher-lesion, frozen brain, permuted curriculum and retention.

## What is reused and what is new

Reused unchanged: the `/api/brain-chat` handler (`webapp.server.brain_chat`, `rich=False`, stub renderer, no LLM),
the tiny-demo brain, the brain's own acquisition hook (`ChatBrain._maybe_acquire`), the D6 capability configuration
(`BRAIN_D6_HEBBIAN_STORE=1`, `BRAIN_D6_ENGRAM_VOCAB=1`, `BRAIN_D6_ENGRAM_READTIME=1`, GO 6/6 on its own gate:
`research/findings/2026-09-23-d6-learn-through-use-v3-capability-gate-GO-6of6.md`), its write-freeze lesion
(`BRAIN_D6_HEBBIAN_FREEZE=1`) and experimenter ablation (`d6_hebbian_store.ablate_block`), the spiking curiosity ask
on an abstain ("... what can you tell me about X?"), the server's own idle tick
(`webapp.continuous_engine.tick_idle_sessions`) for the sleep interval.

New (commits `153af81bf`, `77aa1cb30`): `research/runners/ai_teacher.py` (the teacher; stdlib-only),
`research/runners/ai_teacher_guard.py` (the isolation instrument), `research/fixtures/ai_teacher_curriculum_v1.json`
(the vetted curriculum), `research/runners/ai_teacher_experiment.py` (arms + gate + selftest),
`tests/test_ai_teacher_isolation.py`.

## Protocol (per arm: a fresh brain in its own subprocess)

Every arm: `BRAIN_CHAT_SEED=<seed>`, `SIM_BACKEND=numpy`, `SIM_DISABLE_LLM=1`, renderer `stub`,
`BRAIN_LTM_BUNDLE=off` (the brain holds NO wikidata LTM: everything it knows beyond its 5 build-time facts must
come through the conversation), the D6 capability flags above, `BRAIN_AI_TEACHER=1` (harness flag). Session turns:

1. warmup -- experimenter: "what does the cat eat" (builds the brain; a build-time fact).
2. lesson -- TEACHER, for each of the first K curriculum facts: asks "what does the <s> <v>"; if the reply asks
   back about <s>, answers the ask with "the <s> <v>s the <o>"; otherwise tells that sentence.
3. quiz -- TEACHER: asks each question again; judges the reply TEXT (names <o>, not "I don't know"); restates the
   fact when wrong.
4. sleep -- ENVIRONMENT: the session clock is advanced past `SLEEP_IDLE_SEC` and the server's own idle tick runs
   once (`--sleep-ticks 1`) with the production getters; what it did is recorded, not assumed.
5. lesion -- ZERO arm only: the experimenter zeroes every block whose agent is a taught subject.
6. test -- experimenter, TEACHER ABSENT: the K questions, then "what does the cat eat", "what does the dog chase".

The curriculum's teaching order interleaves the tiers; K=4 teaches "the blicket eats the dax", "the selva touches
the osona" (wikidata `shares_border_with`, index 9525 of the `wikidata_100k` bundle's facts file), "the wug chases the toma",
"the fep carries the zorb"; K=8 adds anoia/bages (wikidata), kiki/modi, tulver/pilk, creuse/allier (wikidata). Every
template verb is one the brain's verb lexicon already covers (the ZPD rule, see "What was SEEN"). The teacher's only
access to the brain is `channel(text) -> reply text`.

## Arms (K=4 unless stated)

| arm | what differs from TEACH |
|---|---|
| TEACH | -- (the teacher teaches the vetted facts) |
| NOTEACH | no lesson, no quiz (teacher lesion) |
| FREEZE | `BRAIN_D6_HEBBIAN_FREEZE=1`: eta=0 for in-conversation writes, same input, same encode activity (WRONG on "same input": see Amendment 2, correction a) |
| ZERO | every taught block zeroed after the sleep interval, before the test |
| PERM | the teacher BELIEVES a per-seed derangement of the objects (it teaches and quizzes those) |
| ERR | the teacher is wrong on facts 2 and 4 (distractor objects that appear nowhere else) and right on 1 and 3 |
| TEACH_REP | identical to TEACH, rebuilt (determinism null; seed 7 only) |
| TEACH K=2, K=8 | retention-vs-K sweep (secondary) |

## Pre-registered criteria (per seed; thresholds fixed in `ai_teacher_experiment.py`)

"recalls(arm, s, v, o)" = the test-phase reply to "what does the s v" is not an abstain and its `recalled_svo`
contains s and o. A probe that never ran is UNDEFINED, never a pass.

- **T1 LEARNS** -- TEACH recalls >= 0.75 of the K taught facts. FAILS IF the brain does not acquire invented-noun
  facts told in chat, or loses them across the quiz / sleep / intervening probes.
- **T2 TEACHER-LESION** -- NOTEACH recalls 0 taught facts. FAILS IF a taught object reaches the brain by any route
  other than the conversation (a leak; e.g. an LTM or harness path).
- **T3 FREEZE** -- FREEZE recalls 0 taught facts. DEFINED ONLY IF the lever held: every taught-subject block in
  FREEZE reads mean |w| == 0 after teaching, FREEZE has a block for every subject TEACH wrote (the same write
  episodes ran), and every TEACH taught block reads mean |w| > 0.5 (else UNDEFINED, e.g. a direct-copy
  reconsolidation write bypassed the freeze). FAILS IF recall
  survives the freeze through another per-session state (discourse / working-memory buffer, episodic organ, the host
  kb record).
- **T4 ZERO** -- ZERO recalls 0 taught facts. DEFINED ONLY IF >= K blocks were ablated and every taught block reads
  0 at test. FAILS IF recall survives zeroing the taught synapses (the reply is carried by something else).
- **T5 PERMUTED** -- PERM recalls >= 0.75 of the objects it was TOLD and 0 of the canonical curriculum objects.
  FAILS IF recall follows the vetted source rather than the teacher, or the brain cannot learn a counterfactual.
- **T6 TEACHER ERROR** -- ERR recalls the ground truth of 0 corrupted facts, and >= 0.5 of its clean facts. FAILS
  IF the brain produces a truth it was never told (a leak) or loses the clean facts. The propagation rate (share of
  corrupted facts recalled as the teacher's wrong object) is REPORTED, not gated.
- **T7 CONTROLS** -- in every gated arm both build-time control probes are recalled (cat->fish, dog->cat). FAILS
  IF teaching, a lesion or capacity use breaks the read path.
- **T8 ISOLATION** -- in every gated arm the guard reports 0 violations, 0 teacher-attributed store calls, >= 10
  patched entry points including `OneBrainComposer._write_block`, `ChatBrain._maybe_acquire` and
  `d6_hebbian_store.hebbian_encode`. FAILS IF any store write is reached from a teacher frame without the chat
  handler on the stack (tested in both directions by `tests/test_ai_teacher_isolation.py`).
- **T9 NO TEST-PHASE WRITES** -- 0 `_write_block` calls during the test phase in every gated arm. FAILS IF probing
  writes (e.g. reconsolidation or consolidation during the teacher-absent test).

A seed is GO iff T1-T9 all pass; a seed with a missing or errored arm, or an UNDEFINED lever, is UNDEFINED (never a
pass, never a fail). **Aggregate GO = GO on all six seeds 42 43 44 100 101 102.** Seed 7 is the dev smoke: scored
the same way and reported, not part of the verdict.

## Secondary (reported, never gating)

- retention vs K: TEACH recall at K = 2, 4, 8;
- learned-content fraction of the test phase: the share of TEACH test replies with content whose content is absent
  from the same probe in ZERO and in NOTEACH (lesion-verified learned); with 2 build-time controls its ceiling at K=4
  is 4/6;
- session learned-content fraction: over every question reply with content in the TEACH session, the share whose
  subject is a taught subject;
- `attributable_to` (tools.lab) of TEACH recall vs FREEZE, ZERO and NOTEACH;
- delivery: facts delivered as curiosity answers vs plain tells, pre-known facts, quiz corrections;
- seed 7: TEACH vs TEACH_REP turn-by-turn differences (the determinism null);
- costs: wall time and peak RSS per arm.

## Commands (verbatim)

With `OUT=research/findings/raw/_ai_teacher/v1`, seed-7 dev smoke (pool node, isolated revision of this commit):
`SIM_BACKEND=numpy OMP_NUM_THREADS=1 .venv/bin/python -u -m research.runners.ai_teacher_experiment --seeds 7 --K 4
--arms TEACH NOTEACH FREEZE ZERO PERM ERR TEACH_REP --k-sweep 2 8 --jobs 3 --arm-dir "$OUT" --json
"$OUT"/verdict_s7.json`

Six seeds: the same with `--seeds 42 43 44 100 101 102 --arms TEACH NOTEACH FREEZE ZERO PERM ERR --k-sweep 2 8`
and `--json "$OUT"/verdict_6seed.json` (the controller runs every gated arm before the sweep arms); scoring alone
adds `--score-only`.
Selftest: `.venv/bin/python -m research.runners.ai_teacher_experiment --selftest` (must print SELFTEST PASS; it
checks that the gate passes the capability case and fails or reads UNDEFINED in each failing direction).

## Declared host residuals (not credited to the brain)

- The teacher is template host code (the social environment): it picks facts, renders sentences with one template,
  and judges a reply by a string match on its text. No LLM paraphrase today.
- The sleep interval is a host clock advance that runs the server's own idle tick; which offline processes run in
  it is whatever the production defaults arm, recorded per arm.
- Every D6 residual (a)-(j) declared in `research/runners/d6_hebbian_store.py` is inherited: the host-wired
  instructive pathway, the host phase-lock loop, the W_MAX clamp, the host block->words map, one disjoint block per
  fact with exact host routing (so capacity, not interference, bounds K; `k_max` = 32 blocks).
- Comprehension of a told sentence runs through the brain's parser, but the verb lemmatizer and the B3
  polar-assertion extractor that decide whether a sentence is an acquisition candidate are host rules.
- The ZERO ablation and the lever reads are experimenter instruments, not brain mechanisms.

## What was SEEN before filing

One plumbing run at DEV SEED 3 (not a registered seed; K=2; arms TEACH, FREEZE, ZERO; pool2, commit `153af81bf`
code with the first curriculum; artifacts kept out of the verdict and filed later under
`research/findings/raw/_ai_teacher/plumbing_s3/`, carrying no pre-registered weight):

- "the blicket eats the dax" was delivered as the answer to the brain's own curiosity ask ("... what can you tell me
  about blicket?"), acquired, answered right in the quiz (TEACH, ZERO), and recalled in the test after the sleep tick
  with the teacher absent (TEACH). FREEZE and ZERO abstained on it at test; the FREEZE blocks read 0.0, the ZERO
  block 1.504 before ablation and 0.0 after.  <!--derived-->
- "the selva borders the osona" was REFUSED by the brain's D4 comprehension monitor (verb and both nouns unfamiliar
  to its cue lexicons -> "I followed the shape of that, but I don't know the words 'selva' or 'osona' yet -- what do
  they refer to?"), in all three arms. That is why the template verbs were changed to verbs the brain already knows
  (commit `77aa1cb30`), before this filing. The refused-sentence case is a real boundary, reported here and named in
  the finding: the brain's repair ask has no learning path yet.
- The sleep tick ran (1 session, ~21-26 s): a thought-wander, a Turrigiano scaling pass over the DA-encoded engrams
  (it rescaled the taught block 1.723 -> 1.504), and a DA-mode relax.  <!--derived-->
- Guard: 30 entry points patched, 0 violations, 15-16 brain-attributed and 5-6 other (build + experimenter) calls.
- Cost at pool2 load ~23 on 16 cores: warmup (brain build + first turn) ~1050-1070 s, later turns 9-93 s, ~1420-1480 s
  per K=2 arm, peak RSS ~0.7 GB.

## Amendment log

(none at filing)

**Amendment 1 (2026-09-24 ~15:55 EDT; scorer output format only).** The aggregate verdict file now carries a
`tools.verdict` `preconditions` block and a `status` (GO / NO-GO / UNDEFINED), because
`gates/verdict_preconditions` blocks any committed verdict artifact without one (it blocked `verdict_s7.json`).
The registered preconditions are this document's own definedness conditions, per seed: every gated arm present,
error-free, with the write counter on; every gated test probe ran; the T3 freeze lever held; the T4 ablation held;
every criterion measured.
No threshold, criterion or per-seed rule changes, and the `GO` boolean is the registered rule unchanged.
One label is stricter in an edge case: a probe that never ran was a per-seed fail in the old scorer; the aggregate
`status` is now UNDEFINED there, as "A probe that never ran is UNDEFINED, never a pass" above already says.
The per-seed `go` field is unchanged.
**What was seen when this was filed:** the seed-7 dev verdict (GO, all nine criteria) and an interim score of the
four registered seeds complete at that time (42, 43, 44, 100: all nine criteria pass on each).
Seeds 101 and 102 were still running.
The six-seed controller on pool2 runs the scorer at `c12c0d47e`, so its own `verdict_6seed.json` has no
preconditions block. The verdict committed for this run is a `--score-only` re-score at this amendment's commit,
over the same arm files, with the same command plus `--score-only`.

**Amendment 2 (2026-09-24 ~17:15 EDT; a STRICTER gate, one new gated arm, provenance preconditions).**
Filed after an independent review of `ecac2791a` found that the gate could not read NO-GO when the brain fails to
form the write, and committed BEFORE any arm it newly governs was scored or run.
Every change below makes a NO-GO or an UNDEFINED reachable where it was not; none relaxes a threshold.

*What was seen when this was filed.*
Seeds 7, 42, 43, 44 and 100: scored under the old scorer, GO on T1-T9 each (Amendment 1).
Seed 101 TEACH and NOTEACH: committed in `87242cf47`; not scored as a seed (the seed was void), and the review quoted
their TEACH taught-block mean |w| (minimum 1.596), a lever read, not a criterion outcome. <!--derived-->
Seed 101 FREEZE/ZERO/PERM/ERR and every seed-102 gated arm: landed on pool2; only the directory listing (names, sizes,
times) was seen; none was read or scored before this commit.
No SHAM arm exists yet.

*Change 1: three-valued criteria, and a measured fail is never hidden.*
Each criterion now reads a declared set of arms: T1 TEACH; T2 NOTEACH; T3 FREEZE and TEACH; T4 ZERO and TEACH; T5
PERM; T6 ERR; T7, T8 and T9 every gated arm, one arm at a time; T10 SHAM and TEACH.
A criterion is UNDEFINED when an arm it reads is void (missing, errored, write counter off, or provenance failed:
change 5), when a probe it reads never ran, or when its lever did not hold.
For T7-T9 a void arm is UNDEFINED for itself and a measured fail in any valid arm fails the criterion.
The seed rule becomes: NO-GO if ANY criterion fails; GO if all pass; otherwise UNDEFINED.
So a T1 fail is NO-GO whatever T3 and T4 read, and whether or not another arm is void.
This replaces "a seed with a missing or errored arm, or an UNDEFINED lever, is UNDEFINED (never a pass, never a
fail)", which now holds for the criteria that read that arm or lever, not for the whole seed.
A probe that never ran was a per-seed FAIL in the old scorer; it is now UNDEFINED, as the criteria section says.
Why: under the old rule, a told sentence the brain refused, or a weak write, made T3 or T4 UNDEFINED and hid the T1
fail. The review reproduced three such cases on the module's own synthetic arms (no write in any arm; half the facts
refused; weak writes with no recall): each read UNDEFINED with T1 failing.

*Change 2: T4 is defined by the cut, not by a count.*
T4 is DEFINED iff every taught block ZERO holds at test is in its ablation record and reads 0, every ablation record
reads 0 after the cut, and the cut covers every subject TEACH wrote a block for.
This replaces "DEFINED ONLY IF >= K blocks were ablated", which read UNDEFINED whenever one told fact was refused
(3 of 4 learned: T1 passes at 0.75, and T4 could not be measured).

*Change 3: the aggregate.*
Aggregate NO-GO iff some registered seed is NO-GO; its `tools.verdict` preconditions are exactly what that NO-GO
rests on (the arms each failing criterion reads are valid with clean provenance, and the criterion was measured),
so an incomplete seed elsewhere cannot hide it.
Aggregate GO iff every registered seed is GO (unchanged), with every seed's arms and criteria registered.
Otherwise UNDEFINED, with the same full registration.

*Change 4: a new gated arm, SHAM, and criterion T10 (review issue 6).*
SHAM runs TEACH exactly. After the sleep interval the experimenter runs the T4 cut's own path
(`d6_hebbian_store.ablate_block`: encoding gain off, `_write_block`, the store-CSR / CSR-cache / fact-shard
invalidation) on every taught block, writing each block's OWN weights back (`sham_rewrite_block`; no cut).
The experimenter then zeroes, with `ablate_block` itself, every block whose agent is neither a taught subject nor a
control-probe subject (cat, dog).
At K=4 these off-target blocks are the three build-time "brain" facts: 3 blocks, not K.
No equal-size cut elsewhere is possible in this brain at K=4: it holds 5 build-time blocks and 2 are the controls.
The size match is carried by the sham: the same K blocks, through the same procedure.
The test phase is unchanged.
**T10 SHAM** -- SHAM recalls every taught fact that TEACH recalls (0 lost).
DEFINED ONLY IF all of these hold: SHAM's warmup, lesson and quiz turns equal TEACH's turn for turn (phase,
message, abstain, recalled_svo, reply text), so the two are the same session up to the manipulation; the sham
rewrite ran on every taught block SHAM holds at test, covered every subject TEACH wrote, and moved no weight (max
|dw| == 0 in every block); every taught block reads > 0 at test; at least one off-target block was cut; every
off-target block SHAM holds was cut and reads 0 at test.
FAILS IF the cut's procedure, or losing weight elsewhere, removes taught recall: then T4's loss is not specific to
the taught synapses.
T7, T8 and T9 also read SHAM. Aggregate GO now needs T1-T10 on all six seeds, so the aggregate stays UNDEFINED
until the SHAM arms land.
Before this change, T4's specificity rested only on T7 (the two build-time controls survive the cut in ZERO); no
cut of equal size elsewhere, and no sham, had been run.

*Change 5: provenance preconditions (review issue 8).*
An arm counts only if: P1 its `.prov.json` sidecar exists and names it (artifact basename, and the worker argv's
`--arm`, `--seed`, `--K`); P2 `git_dirty` is False and, for a git-archive revision, the source manifest was verified
at start and at exit; P3 its revision is `c12c0d47e` (this registration) or a descendant of it; P4 every T1-T9 arm
of one seed ran at ONE revision (else all of them are void, since none can be singled out as the leftover).
SHAM, run later at this amendment's commit, is exempt from P4; T10's same-session condition checks it against
TEACH in the data instead.
A failed provenance check voids that arm (change 1). The controller now prints the revision of every arm file it
skips. Before this change, the scorer did not read the sidecars, and a leftover arm from another revision would
have been scored silently (the review checked all 35 arm sidecars by hand: `c12c0d47e`, clean, manifest verified).

*Change 6: the flag (review issue 7).*
`BRAIN_AI_TEACHER` did nothing: only its own test read it, and default-OFF held because nothing in `webapp/`, `sim/`
or `research/runners/` imports the teacher. "Default-OFF flag" in earlier commit messages and in this file
overstated it.
From this commit `AITeacher(...)` raises `AITeacherDisabled` unless the flag is on; the worker sets it per arm, so
behaviour with it on is unchanged (T10's same-session condition checks that for the SHAM arms in the data).
The T1-T9 arms ran at `c12c0d47e`, where the flag was set and not checked.

*Corrections to this document and to the lane report.*
(a) The arms table said FREEZE gets "same input, same encode activity" as TEACH. Wrong on input: the quiz depends
on the reply, the frozen brain fails it, and the teacher restates each fact. On every scored seed (7, 42, 43, 44,
100) FREEZE had 23 turns against TEACH's 19, and 8 write episodes (lesson 4 + quiz 4) against 4.
The difference is conservative (the lesioned arm gets more exposure) and the finding must state it.
The scorer now reports `secondary.exposure_by_arm`.
(b) A lane report (not this file) said "T3 freeze blocks read 0 against taught blocks above 1.7". Wrong.
TEACH taught-block mean |w| after teaching, minimum per seed: 1.756 (s7), 1.868 (s42), 1.460 (s43), 1.454 (s44), <!--derived-->
1.537 (s100), 1.596 (s101). The lever threshold is 0.5, so no verdict changes. The sentence must not enter the <!--derived-->
finding; the scorer now reports `secondary.teach_taught_block_w`. (The minima above are rounded from that field in
research/findings/raw/_ai_teacher/v1/verdict_s7.json and verdict_6seed.json.)
(c) `verdict_s7.json` is the dev seed. Nothing may cite it as the lane's result.

*Commands (verbatim), with `OUT=research/findings/raw/_ai_teacher/v1`.*
SHAM arms (pool node, isolated revision of this amendment's commit):
`SIM_BACKEND=numpy OMP_NUM_THREADS=1 .venv/bin/python -u -m research.runners.ai_teacher_experiment --seeds 7 42 43
44 100 101 102 --K 4 --arms SHAM --jobs 7 --arm-dir "$OUT"`
Dispatch note (resources only, filed before any SHAM arm ran): the SHAM arms were queued on the pool as seven
one-seed jobs (`--seeds <S> --jobs 1`, same `--K 4 --arms SHAM --arm-dir "$OUT"`, thread env vars at 1), pinned to
the isolated revision `d460b4498` provisioned on pool2 only, the node that ran every T1-T9 arm (same CPU model and
numpy 2.2.6, so T10's same-session condition compares like with like). The arms are the same as the one-job form.
Copy the arms home without the pool's own verdicts: `rsync -a --exclude 'verdict_*' <node>:<revision>/"$OUT"/
"$OUT"/`.
The registered verdict: `SIM_BACKEND=numpy .venv/bin/python -m research.runners.ai_teacher_experiment --score-only
--seeds 42 43 44 100 101 102 --K 4 --k-sweep 2 8 --arm-dir "$OUT" --json "$OUT"/verdict_6seed.json`, at this
amendment's commit or later.
The dev seed: the same with `--seeds 7` and `--json "$OUT"/verdict_s7.json`.
Selftest: `--selftest` must print SELFTEST PASS. It now also reproduces the review's three hidden-fail cases (each
must read NO-GO), the 3-of-4 case (T4 defined), every T10 failing and undefined direction, and each provenance
failure (each must read UNDEFINED).
