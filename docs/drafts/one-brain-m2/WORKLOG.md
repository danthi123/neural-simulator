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

- Brain-free (no lock needed; numpy, no build): `tests/test_one_brain.py` 35 passed: the flag parse (default off),
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
  the OpenAI shim's interlocutor pass-through.
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
  `_onebrain_crossedge_curiosity_to_d6wm.py:140`, files M2 does not touch). BASE CHECK: PENDING.

## 4. OFF hash identity (single conversation)

PENDING.

## 5. ON: the M0 probe arms (seeds 7 and 8)

`research.runners.one_brain_probe` (M0, predictions committed before any run), stub renderer, numpy CPU, memcap 14,
from the frozen export. Reads: recall = an SVO with the expected role filler, abstain = abstained and no match,
other = anything else. Raw: `research/findings/raw/_one_brain_probe/m2_s7.json`, `m2_s7_instruments.json`,
`m2_s8.json`; logs `docs/drafts/one-brain-m2/runs/probe_s*.log`.

TABLE: PENDING (filled when every arm has run).

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

PENDING.

## 7. Every reply change at ON, by design

PENDING.

## 8. Open risks

1. "Did we discuss X?" across conversations. The D5 episodic store is brain-level (design row B) and carries no
   source or WHEN tag yet, so conversation B can be told it discussed something only A discussed. The "Someone told
   me" frame covers RECALLED FACTS (`recalled_svo`, `supporting_facts`, `derived_from`), not D5 referential replies.
   This is M3's job (the source monitor reads "did YOU tell me"); until then a cross-conversation D5 "yes" is an
   honesty gap at ON.
2. The source frame reads a HOST tag (declared scaffold, `one_brain_source.reason = "host_tag"`): exact (a, v, p)
   triples that entered the store during a conversation's turn. Not checkpointed: after a restart with persistence
   (M5) a reloaded fact has no tag, so it is recalled with no frame. A reconsolidation rewrite is tagged to the
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
