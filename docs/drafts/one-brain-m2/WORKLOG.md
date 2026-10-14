# One brain, many conversations: M2 (Brain / Context) WORKLOG

Branch `claude-build/one-brain-m2` off S main `afeb1bcc9`. Design: `docs/plans/2026-10-13-one-brain-shared-memory-DESIGN.md`
(sections 1, 2, 5; owner rulings section 9). Flag `BRAIN_ONE_BRAIN`, DEFAULT OFF (`webapp/one_brain.py`
`_ONE_BRAIN_DEFAULT_ON = False`). No flip, no persistence flip (M5), no source-monitor organ (M3). Dev seeds 7 and 8
only; no gate seed was run.

## 1. What landed

- `webapp/one_brain.py`: `Context` (one per conversation: the C rows, the typed `interlocutor`, the conversation's
  ChatBrain fields while it is not bound), `ContextRegistry` + `ContextFieldView` (the server's `_BRAIN_CHATS`,
  `_BRAIN_RICH` and every `_SESSION_*` dict is now a dict-shaped view of one Context field, exact dict semantics and
  order), `Brain` (one per process; at ON one shared ChatBrain per (brain, renderer), the bind/swap of the
  conversation-level fields, the tonic mood baseline, the host source tags, the zero instrument), `DeferredMouth`
  + `articulate` (the generative mouth after the lock), `TickAdapter` (sleep as a brain event).
- `webapp/server.py`: the registries are Context field views; `_bkey()` keys the brain-level organs (D5 episodic,
  idle replay `note_turn`/`speak`, causal organ, `recall_disclosure`, persistence) by the brain key at ON and by the
  conversation key at OFF (identity); `_one_brain_turn` (bind, the same `brain_reply`, source tags, the frame, the
  deferred mouth); `one_brain_zero_learned_store()` (experimenter instrument, refused in production and at OFF);
  `BRAIN_ONE_BRAIN_FREEZE_LEARNING` (the shared store's plasticity gate, `_episodic_store_ok` false too);
  `interlocutor` / OpenAI `user` / per-message `name` on the request; at ON the reset route, the startup warm thread
  and the shutdown flush take the turn lock (M0 left them outside it); `BRAIN_SHARED_LTM` is forced on at ON.
- `webapp/continuous_engine.py`: at ON the shared-store steps (sleep replay, persistence save, Turrigiano
  homeostasis, tag-and-capture) are brain events, once per brain, on the WHOLE brain's idleness (sleep: no turn in
  any conversation for >= `SLEEP_IDLE_SEC` = 300 s); the feel, the wander, D5 learn-through-use and the #91/#92
  relaxations stay per conversation.
- `research/runners/affect_production_organ.py`: `MoodConditionedRenderer` hands a manner to a deferred mouth
  (`render_svo_manner`) so the mood conditioning reaches the generative prompt after the lock.
- `webapp/openai_shim.py`: `_last_user_name`.
- Fold-ins (a)-(c) of the M0 review, each with a test (section 2).

## 2. Fold-ins (a)-(c)

- (a) `continuous_engine.last_wandered` returns the true newest wander (`_LAST_WANDERED`, set when a wander is
  recorded, cleared by `forget_session`), even after the mull consumed it. Test:
  `tests/test_brain_turn_lock.py::test_last_wandered_names_the_newest_thought_even_after_it_was_consumed`
  (fails on the base engine: `'cat' == 'dog'`).
- (b) The wander PEEKS at the mull arm; the record step consumes it only for a recorded (not preempted, not reset)
  wander, and only if it is still the arm the wander read. Test: `test_a_stale_wander_leaves_a_fresh_sessions_mull_arm`
  (fails on base: `None == 'otter'`) + a control.
- (c) `cross_session_persistence` reload: the slot record is parsed per word (a non-int slot drops that word only, with
  a warning, never the whole reload); after the facts reload, every free slot whose placeholder code equals a live
  word's code gets a fresh code (`OneBrainComposer.separate_reserve_codes`), then `duplicate_codes()` is asserted
  empty (an ERROR is logged otherwise). Tests: `tests/test_cross_session_persistence.py`
  `test_reload_with_a_damaged_slot_record_keeps_every_code_distinct[non_int|slotless|out_of_range|duplicate]` + the
  control `test_damaged_slot_record_without_the_separation_shares_a_code`.

## 3. Tests

- Brain-free (no lock needed; numpy, no build): `tests/test_one_brain.py` 38 passed (tip): the flag parse (default off),
  Context views vs a plain dict on a fixed op sequence (order, re-set, pop-then-set, a stored None), registry remove,
  the server views being Context fields + eviction removing the Context, the ATTRIBUTE CLASSIFICATION SCAN (every
  `chat.<attr> =` in `webapp/*.py` and every ChatBrain self-attribute written outside `__init__` must be classified
  conversation-level or brain-level; ChatBrain's lazily built organs, caches and the store's word sets are
  brain-level), the bind swap
  (C state per conversation, B state shared, an evicted resident starts clean), the source frame (lead, part note,
  the teller's own conversation untouched, abstain untouched, anonymous), the tonic seed, the real handler at ON with
  the build stubbed (one build for two conversations, B starts clean, B recalls with the frame, the teller does not,
  the interlocutor never leaves its Context, OFF keeps a chat per conversation, the reset route at ON takes the lock
  and forgets the wander state but not what the brain learned), the deferred mouth (generation without the turn lock,
  VERIFY under it, a failed VERIFY keeps the template, a second conversation is admitted while the first one's mouth
  generates, the mood manner reaches the generative prompt, a template renderer gets no deferred mouth), the freeze
  gate, the zero instrument (only post-build blocks, caches invalidated, tags cleared, refused at OFF and in
  production), sleep/persist as ONE brain event only when the whole brain is idle >= 300 s, the OFF tick unchanged,
  the OpenAI shim's interlocutor pass-through, the frozen acquisition reporting no acquisition (the real
  `ChatBrain._maybe_acquire`), zero dropping the organs built after the build, a reset conversation being a new
  conversation for the frame.
- Brain-free regression: `test_brain_turn_lock.py`, `test_battery_session_release.py`,
  `test_tick_pre_m0_equivalence.py` (the pre-M0 golden tick trace), `test_idle_replay_organ.py`,
  `test_sleep_replay_capture.py`, `test_brain_prewarm_scratch_session.py`: 102 passed, 1 skipped.
- Under the brain lock (frozen export): `tests/test_cross_session_persistence.py` 27 passed (log
  `runs/pytest_csp_rerun.log`). The first run (`runs/pytest_csp.log`) failed 2: a local `import logging` in
  `reload_session_facts` shadowed the module import, so the new non-int-slot warning raised `UnboundLocalError` (the
  new test caught it), and the M0 control `test_reload_without_slot_restore_reproduces_the_shared_code` no longer
  reproduced the collision because the new separation also prevents it (the control now removes both defences).
  Fixed in `40764aead`.
- Under the brain lock, memcap 18: `tests/test_webapp_server.py -k "brain_chat_reset_idempotent or
  tiny_demo_answers_and_abstains or rich_false_escape or curiosity_d6_session_isolated or
  source_provenance_explicit_off"`: 4 passed, 1 failed (`test_brain_chat_xedge_curiosity_d6_session_isolated`: the
  xedge curiosity-D6 pool build raises `KeyError: 'cue'` in `sim/regions.py` via
  `_onebrain_crossedge_curiosity_to_d6wm.py:140`, files M2 does not touch). Pre-existing: run alone on the base export
  (afeb1bcc9) it fails the same way (`KeyError: 'cue'`, 1 failed in 547 s). Logged in `research/FAILURE_LOG.md`.
- Run logs are local only (`*.log` is gitignored): `docs/drafts/one-brain-m2/runs/*.log` in the worktree, with the
  bridge's per-build INFO lines stripped.

## 4. OFF hash identity (single conversation)

Driver `docs/drafts/one-brain-m2/conversation_hashes.py`: the real `/api/brain-chat` handler (`S.brain_chat`), 17
turns in one conversation that drive every per-conversation registry (the build buffer, a taught fact and its recall,
the affect EMA, the LTM, the multiref WM and an anaphor, D5 "did we discuss", a contradiction + reconsolidation, the
discourse register fold and before-query, prospective memory, the world-model expectation, an empty turn
(self-initiation), "tell me more" (rich thread), an untaught word (the moat)); sha256 over each reply's WHOLE JSON
body. tiny-demo, stub renderer, numpy CPU, one OMP thread. Base = `git archive afeb1bcc9` (S main); OFF = the branch
export with `BRAIN_ONE_BRAIN` unset. Compare: `compare_hashes.py`.

| run pair | seed | hash seed | identical bodies | the difference |
|---|---|---|---|---|
| base vs OFF | 7 | unset (random) | 16/17 | turn 16: `curiosity.novelty` 0.9748080180058432 vs ...433 |
| base vs OFF | 8 | base unset, OFF 0 | 16/17 | turn 16: the same field, ...433 vs ...432 |
| base vs OFF | 7 | 0 | PENDING | |
| base vs OFF | 8 | 0 | PENDING | |

The one-ulp difference is not the branch: the same value flips between ...432 and ...433 across runs of EITHER code
(base gave ...432 at s7 and ...433 at s8; the branch the reverse), and the topic gate behind it is seeded 42
regardless of the chat seed. It is read after `topic_novelty` imprints `_brain_vocab(chat)`, a Python `set` of
strings, whose iteration order follows the per-process string-hash seed. The project's identity convention is
`PYTHONHASHSEED=0` + one BLAS thread (`docs/drafts/s2p/make_jobs.sh`); the hash script now sets it and both sides
were re-run under it (rows 3-4).

## 5. ON: the M0 probe arms (seeds 7 and 8)

`research.runners.one_brain_probe` (M0, predictions committed before any run), stub renderer, numpy CPU, memcap 14,
from the frozen export. Reads: recall = an SVO with the expected role filler, abstain = abstained and no match,
other = anything else. Raw: `research/findings/raw/_one_brain_probe/m2_s7.json`, `m2_s7_instruments.json`,
`m2_s8.json`; logs `docs/drafts/one-brain-m2/runs/probe_s*.log`.

Code per arm (the export's `FROZEN_SHA`): s7 one_brain / first freeze / first zero at `40764aead`; s7 intact,
restart_persist_on_one_brain and both instrument re-runs (`m2_s7_instruments.json`) and every s8 arm at `38130f5c8`
(the instrument fix `d56e7baf7` + the frame scoping). The probe's own `git_sha` reads "unknown" (an export has no git).

| seed | arm | A_ask | B_ask | A_ask_again | B_ltm | B_build | all match | peak RSS MB | wall s |
|---|---|---|---|---|---|---|---|---|---|
| 7 | one_brain | recall | recall |  | recall | recall | yes | 6502 | 558 |
| 7 | freeze_learning | abstain | abstain |  | recall | recall | yes | 6514 | 530 |
| 7 | zero_learned_store | recall | abstain |  | recall | recall | yes | 6451 | 570 |
| 7 | intact (OFF) | recall | abstain |  | recall | recall | yes | 11927 | 683 |
| 7 | restart_persist_on_one_brain | recall | recall | recall | recall | recall | yes | 6431 | 507+549 |
| 7 | freeze_learning (first run, before the fix) | **other** (pred abstain) | **other** (pred abstain) |  | recall | recall | **no** | 6501 | 529 |
| 7 | zero_learned_store (first run, before the fix) | recall | **other** (pred abstain) |  | recall | recall | **no** | 6498 | 557 |
| 8 | one_brain | recall | recall |  | recall | recall | yes | 6500 | 596 |
| 8 | freeze_learning | abstain | abstain |  | recall | recall | yes | 6492 | 541 |
| 8 | zero_learned_store | recall | abstain |  | recall | recall | yes | 6453 | 584 |
| 8 | intact (OFF) | recall | abstain |  | recall | recall | yes | 11929 | 700 |
| 8 | restart_persist_on_one_brain | recall | recall | recall | recall | recall | yes | 6432 | 516+576 |

Reading it. At ON, conversation B recalls the fact conversation A taught (both seeds); with the shared store's
plasticity gate shut, neither A nor B recalls it; zeroing what was learned since the build removes it from B; the LTM
and build-fact controls recall in every arm (the lesions are specific). At OFF (intact) B abstains, as today. With
persistence on, the brain writes ONE checkpoint under the brain identity
(`facts/__one_brain____tiny-demo__stub-*/checkpoints/*/facts.json`, plus the process-global xedge) and after a restart
both conversations recall. One brain is about half the memory of two per-session brains (6.5 vs 11.9 GB peak RSS).
These are two DEV seeds; no generalization claim (the M2 flip needs the six gate seeds under a committed prereg).

What B hears. s7 (`40764aead`): "Someone told me: the wug eats the dax the dog chases the cat the cat eats the fish";
the rich reply also states two build facts, so the lead attributed build knowledge to "someone". Fixed in
`38130f5c8`: the lead only when everything the reply states was told elsewhere, else the part note. s8: "the wug eats
the dax the dog chases the cat the cat eats the fish (Someone told me part of that.)". Under the fixed freeze the
teach turn itself says "I don't know about that. My curiosity is piqued -- I haven't learned about wug yet: what can
you tell me about wug?" (no acquisition is reported). After a restart B recalls the fact with NO frame (the host
source tags are not checkpointed; open risk 2).

### The first freeze and zero arms failed their prediction, and why (s7, code `40764aead`)

Both arms read `other` on B_ask (and the freeze arm on A_ask too), not `abstain`. The reply in both: "(My authorship
monitor reads this as self-generated -- a guess of my own, not something I was taught.) perhaps wug eat dax [a guess
from what I've learned -- not something I was taught]". The composer store was frozen (freeze) or truncated (zero:
1 block dropped), and the strict recall abstained; the guess came from the default-ON CATEGORY-INHERITANCE organ
(Stage-1 C1, `webapp/category_inherit_chat.py`): a brain-level learned member->frame Hebbian pathway that is FED by
the acquisition verdict (`gate_extract` -> `('done', svo)`) and READ on a strict-recall abstain, returning a flagged
guess. So the one brain holds a told fact in TWO learned places, the composer store (recall) and the C1 pathway (a
flagged guess), and both are shared across conversations. The instruments covered only the first:
- the freeze gate returned None from `inner.hear`, so `_maybe_acquire` still reported the acquisition and C1 observed
  it;
- the zero instrument truncated the composer but left the C1 organ (built after the build) in place.
The honesty boundary held in both: the reply was flagged as a guess and the authorship monitor read it as
self-generated, never as a recalled fact. Fix (`d56e7baf7`): the freeze gate raises `LearningFrozen`, which
`_maybe_acquire` treats as a failed write (no acquisition reported, so no learner observes it); zero also drops every
brain-level learned organ that did not exist at the build (`ZERO_DROP_IF_BUILT_LATER`: the C1 pathway, the #129
provenance trace) and the generative proposer. Re-run as `inst7` (`m2_s7_instruments.json`); seed 8 ran on the fixed
code. Residual (declared): Hebbian updates to an organ that existed at the build (the CA3 assoc graph
`_learned_assoc`, fed by `hear`) are not reverted by zero; under freeze `hear` never runs, so freeze covers it.

## 6. Latency per turn, ON vs OFF

Wall seconds per turn from the hash runs (one process each, serial, numpy CPU, one thread). The workstation is shared
(the CPU pool and the GPU queue may have been running other jobs), so treat differences under ~10% as noise. Turn 0
includes the brain build.

| run | turn 0 (build) OFF / ON | turns 1-16 summed OFF / ON | median per-turn ratio ON/OFF (range) |
|---|---|---|---|
| single conversation s7 | 463.3 / 444.4 | 767.1 / 716.9 | 0.93 (0.90-0.97) |
| single conversation s8 | 482.3 / 459.3 | 799.0 / 783.3 | 0.99 (0.92-1.02) |
| two conversations s7 (pair) | 355.9 / 351.9 | 388.6 / 282.1 | 0.96 (0.18-0.99) |

- The bind (saving one conversation's fields and restoring another's) costs nothing measurable: a single conversation
  at ON is not slower (median ratio 0.93-0.99; the s7 OFF side ran without a hash seed, s8 both under it). Against
  OFF with the shared LTM on (the like-for-like store), s7 turns 1-16: 725.0 s OFF vs 716.9 s ON.
- A NEW conversation is cheaper at ON: in the pair run B's second turn took 114.3 s at OFF (its own brain's first
  rich recall) and 20.5 s at ON (the ratio 0.18); in the probe, B's first ask took 124.7 / 125.9 s (s7 / s8) with its
  own per-session brain (intact) and 25.6 / 27.2 s on the shared brain.
- Memory: 6.5 GB peak RSS for one brain serving two conversations vs 11.9 GB for two per-session brains (probe).
- Not measured here: the deferred Qwen mouth (no GPU run; stub renderer throughout). With a generative renderer the
  turn lock is released before generation, so a second conversation's turn waits only for the first one's decision
  (`tests/test_one_brain.py::test_a_second_conversation_gets_its_turn_while_the_first_one_speaks`), plus the short
  re-take of the lock for VERIFY.

## 7. Every reply change at ON, by design

Measured (hash runs, ON vs OFF, same seed, both under `PYTHONHASHSEED=0` unless noted) and by construction.

A. ONE conversation alone (single script; s8 both hash-seeded, s7 OFF side unseeded). Every `answer`, `abstained`
and `recalled_svo` is identical (17/17 at both seeds). Bodies identical 14/17 at s8 (12/17 at s7, where turn 16 is
also the hash-seed ulp of section 4); the differing fields are numeric read-outs of the block the conversation TAUGHT
("the wug eats the dax"): `activity.rf.mean_magnitude` (s7 83467.6 -> 85884.5), the role-decode margins,
`metacog.mean_role_conf`, `d6_hebbian.held.readout` (404.0 -> 415.7) on turns 1-2 (and the magnitude on turn 3 at
s7), and the idle-replay candidate score of that engram on the empty turn 14. Build facts read identically.
ATTRIBUTED, all of it, to the forced shared LTM: OFF with `BRAIN_SHARED_LTM=1` (`offltm_s7_single.json`) vs ON at s7,
both hash-seeded: 17/17 bodies byte-identical. ON forces `BRAIN_SHARED_LTM`, which forces `BRAIN_LTM_RNG_ISOLATE`, so
the LTM load no longer reseeds the process-global RNG streams the teach turn's write draws its noise from (the
shared-LTM step's documented effect; its own flip battery is separate). The Brain/Context machinery itself (bind,
brain-level keys, the tonic with one conversation, the tick adapter) changes no byte of a single conversation.

B. Two conversations (pair script s7: A teaches, A says something happy, B asks).
1. B recalls what A taught (OFF: "I don't know about that."; ON: the fact). The core change.
2. The source frame: "(Someone told me part of that.)" appended (or "Someone told me: " leading when everything the
   reply states was told in another conversation) and a `one_brain_source` field with `reason: "host_tag"`.
3. The TONIC mood term: B's phasic mood starts from the brain's tonic baseline, which A's happy turn moved (B reports
   `affect.appraisal_valence` 0.0787 / `appraisal_arousal` 0.0597 on every turn instead of 0.0). Below the sign
   threshold here, so no wording changed; a stronger or repeated affect in one conversation will change the tone of
   the next one's first turns.
4. Downstream of 1, in B's OWN conversation state: B now holds the recalled topic, so its next question gets the
   thread-swap lead ("Setting the held thread aside -- On dog, then -- the dog chases the cat"), and "did we discuss
   the wug" routes to the D5 referential read (B now knows the word) and answers "I don't recall us discussing wug --
   no assembly completes for that cue" instead of the unknown-word reply. `swap_drives.held_topic` differs after.
5. A's own replies change only by the numeric read-outs of A1.

C. By construction (no ON run with these conditions here; covered by unit tests):
1. With a generative (Qwen) renderer the brain decides on the template articulation inside the lock and the mouth
   generates after it; prose that fails VERIFY keeps the verified template, and the in-lock `render_svo_regen` retry
   no longer runs. Each reply carries a `one_brain_mouth` record. Mood manner reaches the prompt as before.
2. Sleep replay, the persistence save, Turrigiano homeostasis and tag-and-capture are BRAIN events on whole-brain
   idleness: a conversation idle alone while another is active no longer gets a sleep-replay or a persistence save
   (inner-life notes and the consolidation timing change accordingly).
3. A reset at ON forgets the conversation (its Context and wander state) but not what the brain learned: a reset
   conversation can recall a fact it taught before the reset, framed "Someone told me" (a new conversation id).
4. The lesion instruments, only when set: under `BRAIN_ONE_BRAIN_FREEZE_LEARNING=1` an assertion is not acquired and
   the turn says it has not learned the word; `one_brain_zero_learned_store()` (test/experimenter only) removes what
   was learned since the build.
5. The interlocutor name changes no reply (stored on the Context only).

## 8. Open risks

1. "Did we discuss X?" across conversations. The D5 episodic store is brain-level (design row B) and carries no
   source or WHEN tag yet, so conversation B can be told it discussed something only A discussed. The "Someone told
   me" frame covers RECALLED FACTS (`recalled_svo`, `supporting_facts`, `derived_from`), not D5 referential replies.
   This is M3's job (the source monitor reads "did YOU tell me"); until then a cross-conversation D5 "yes" is an
   honesty gap at ON. Not triggered in the pair run (B's "did we discuss the wug" found no D5 episode and said it
   did not recall), so it is structural, not observed.
2. The source frame reads a HOST tag (declared scaffold, `one_brain_source.reason = "host_tag"`): exact (a, v, p)
   triples that entered the store during a conversation's turn. Not checkpointed: after a restart with persistence
   (M5) a reloaded fact has no tag, so it is recalled with no frame (observed: the restart arm's B_ask, both seeds). A reconsolidation rewrite is tagged to the
   conversation that rewrote it.
3. The tonic mood baseline is not checkpointed (it restarts at neutral).
4. The renderer is part of the brain key `("__one_brain__", brain, renderer)`: a stub conversation and a qwen
   conversation in one process get two brains (two learned stores). "One brain per instance" holds for one renderer
   per process, which is how production runs; a per-Context mouth over one brain is the follow-up.
5. Each new conversation builds its own discourse WM lazily on its first multi-referent turn (the WM is C), so a
   multi-referent turn in a fresh conversation pays one WM build (latency, section 6).
6. Prewarm at ON: the scratch kernel-warm turn runs on THE brain, so its brain-level side effects (a D5 episode of the
   warm-up question, an idle-replay note) remain in the brain; the scratch Context is discarded.
7. Default-off Qwen paths that call the warm faculty directly (`BRAIN_OPEN_ENDED`, `BRAIN_OPEN_ENDED_GATED`) still
   generate inside the turn lock at ON; only `chat.renderer` is deferred.
8. With a generative renderer the in-lock render is the template and the mouth runs once per call after the lock:
   the in-lock `render_svo_regen` retry on a failed VERIFY no longer fires (a failed prose VERIFY keeps the verified
   template). Not exercised on a real Qwen here (every run used the stub renderer).
9. At ON `_BRAIN_CHATS` is empty (the shared chats live on `_ONE_BRAIN.chats`): code that enumerates `_BRAIN_CHATS`
   (status, memory accounting) sees no chat at ON.
10. Reset at ON evicts the Context only; nothing a conversation does resets the brain's learned store (design section
   6). The battery and the stage-1 harness rely on a fresh brain per group/session today, so an ON battery needs a
   brain reset between groups first (the test-only `one_brain_zero_learned_store`, refused under
   `BRAIN_PRODUCTION=1`, is the instrument; wiring it into the harnesses is a follow-up before any ON battery run).
11. The freeze gate shuts the acquisition write (`inner.hear`) and the D5 store; other plastic writes on the shared
   store (reconsolidation in place, D6 Hebbian, tag-and-capture, homeostasis) are not gated. The zero instrument drops
   blocks appended after the build; an in-place rewrite of a BUILD fact is not undone.
12. One process-wide turn lock: N simultaneous conversations serialize (with the stub renderer the whole turn).
13. Unchanged at OFF by construction (OFF must be byte-identical): the reset route still never calls
   `forget_session` at OFF (M0 finding); it does at ON.

## 9. Reproduce

- Frozen export of the commit under test: `git archive <sha> -- . ':(exclude)research/findings'` into a directory,
  `research/findings` and `data` symlinked, `FROZEN_SHA` written.
- `bash docs/drafts/one-brain-m2/queue_runs.sh <export> tests probe7 off7 probe8 off8 on7 on8 pair7 offpair7 ltm7`
  (every step under `flock -w 7200 /home/dant123/Projects/sim-rev/brain.lock` + `tools/mem_ok.sh` +
  `tools/memcap.sh`; numpy CPU, `OMP_NUM_THREADS=1`).
- Base: `run_hashes.sh base <git archive afeb1bcc9 export> <seed> single`.
- Compare: `python docs/drafts/one-brain-m2/compare_hashes.py <A.json> <B.json>`.
