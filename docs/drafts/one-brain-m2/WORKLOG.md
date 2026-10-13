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

PENDING (brain-free + under the lock).

## 4. OFF hash identity (single conversation)

PENDING.

## 5. ON: the M0 probe arms (seeds 7 and 8)

PENDING.

## 6. Latency per turn, ON vs OFF

PENDING.

## 7. Every reply change at ON, by design

PENDING.

## 8. Open risks

PENDING.

## 9. Reproduce

- Frozen export of the commit under test: `git archive <sha> -- . ':(exclude)research/findings'` into a directory,
  `research/findings` and `data` symlinked, `FROZEN_SHA` written.
- `bash docs/drafts/one-brain-m2/queue_runs.sh <export> tests probe7 off7 probe8 off8 on7 on8 pair7 offpair7 ltm7`
  (every step under `flock -w 7200 /home/dant123/Projects/sim-rev/brain.lock` + `tools/mem_ok.sh` +
  `tools/memcap.sh`; numpy CPU, `OMP_NUM_THREADS=1`).
- Base: `run_hashes.sh base <git archive afeb1bcc9 export> <seed> single`.
- Compare: `python docs/drafts/one-brain-m2/compare_hashes.py <A.json> <B.json>`.
