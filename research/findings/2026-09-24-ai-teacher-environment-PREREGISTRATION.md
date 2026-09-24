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
| FREEZE | `BRAIN_D6_HEBBIAN_FREEZE=1`: eta=0 for in-conversation writes, same input, same encode activity |
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

<!--derived-->
One plumbing run at DEV SEED 3 (not a registered seed; K=2; arms TEACH, FREEZE, ZERO; pool2, commit `153af81bf`
code with the first curriculum; artifacts kept out of the verdict and filed later under
`research/findings/raw/_ai_teacher/plumbing_s3/`, carrying no pre-registered weight):

- "the blicket eats the dax" was delivered as the answer to the brain's own curiosity ask ("... what can you tell me
  about blicket?"), acquired, answered right in the quiz (TEACH, ZERO), and recalled in the test after the sleep tick
  with the teacher absent (TEACH). FREEZE and ZERO abstained on it at test; the FREEZE blocks read 0.0, the ZERO
  block 1.504 before ablation and 0.0 after.
- "the selva borders the osona" was REFUSED by the brain's D4 comprehension monitor (verb and both nouns unfamiliar
  to its cue lexicons -> "I followed the shape of that, but I don't know the words 'selva' or 'osona' yet -- what do
  they refer to?"), in all three arms. That is why the template verbs were changed to verbs the brain already knows
  (commit `77aa1cb30`), before this filing. The refused-sentence case is a real boundary, reported here and named in
  the finding: the brain's repair ask has no learning path yet.
- The sleep tick ran (1 session, ~21-26 s): a thought-wander, a Turrigiano scaling pass over the DA-encoded engrams
  (it rescaled the taught block 1.723 -> 1.504), and a DA-mode relax.
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
