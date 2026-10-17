# One brain, shared memory: WORKLOG

Design: `docs/plans/2026-10-13-one-brain-shared-memory-DESIGN.md` (merge 2e7392f12). Branch
`claude-build/one-brain-m0`, cut from s/main 2e7392f12.

## M0 (2026-10-13)

### Turn lock (as revised after the review, FIX_FIRST on d6b7baa09)

- `webapp/brain_turn_lock.py`: one process-level `FifoRLock` (a ticket queue on a Condition, re-entrant for the
  holder; a timed-out waiter leaves the queue clean). The first version used `threading.RLock`, which is not fair:
  the review measured a turn waiting through all five sessions of a synthetic tick (0.3 s steps) twice.
  `brain_turn(where, timeout)`, escape `BRAIN_TURN_LOCK=0`, wait statistics in `stats()`.
- `webapp/server.py` `brain_chat`: the locked body (backend pin, reset, cache lookup + build + persisted-learning
  reload, `brain_reply`) moved verbatim into `_brain_chat_turn`. `mark_request` runs before the lock (a waiting turn
  reads as active); `finally` stamps `mark_turn_end` (no wander-budget refill).
- Bounded wait: `BRAIN_TURN_WAIT_TIMEOUT_S` (default 300 s; <= 0 = unbounded). A turn not admitted in time gets a
  200 reply in the normal shape with `busy: true` (nothing recalled, `abstained` false: the brain never saw it),
  instead of stacking waits that hold anyio threadpool tokens (which can stall `/reset` and `/v1/models`).
- Placement of the mouth: the Qwen render is called from inside `brain_reply` (`chat.render`, the rich composer), so
  it is INSIDE the lock. Moving it out needs `brain_reply` split into decide / articulate; that is M2 work.
- Wait time: the response has no timing or debug field, so nothing was added to it. It is in
  `brain_turn_lock.stats()` and in each probe phase record (`turn_lock`).

### Idle tick and world steps

- Path: `server.py` startup task `_continuous_state_tick` runs `continuous_engine.tick_idle_sessions` every
  `IDLE_SEC`=20 s in a thread executor, default ON (`BRAIN_CONTINUOUS`). It reads the PROCESS-SHARED affect organ
  and, per session, runs the self-init wander (~55 s on cupy, budget 1, refilled every turn), D5 consolidation and
  sleep replay (write the D5 store), the persistence save, Turrigiano scaling, the DA tag-capture ledger and the
  #91/#92 relax drives. Its only guard was "no request for 20 s".
- The first M0 version held the lock for the whole per-session step, wander included, which reversed the 2026-08-20
  rule that a chat turn never waits on the wander. Revised: per session, (1) under the lock, cheap: idleness (clock
  read inside the lock; measured from the later of turn start and turn end) and the relax + shared-affect read;
  (2) the wander with NO turn lock, under the session's own `selfinit_lock`; (3) under the lock: record the result,
  or, if a turn for that session arrived during the wander (`_REQ_SEQ` moved), drop it and record the drop
  (`wander_dropped` in the inner-life record) and leave the rest of that session for the next idle period; (4) each
  consolidation / persistence / ledger step under its own lock hold. Biology: an exogenous input preempts
  mind-wandering. A waiting turn is served between any two steps (FIFO).
- Correction to the review's premise: the tick is not the only user of the self-init organ. A self-init trigger
  turn ("what are you thinking about?") speaks through the same organ (`brain_reply`, `_get_selfinit_organ(...)
  .speak`), so that call now takes the organ's `selfinit_lock`. Re-review: that turn holds the turn lock, so it
  waits at most 0.25 s for the organ (`continuous_engine.selfinit_speak_or_mid_thought`); if its session's wander is
  still running it answers with an honest mid-thought read-out (naming the last thought on record) instead of making
  every other session wait out a ~55 s wander. The end-of-turn stamp is written inside the lock. A reset during the
  wander drops what the wander wrote (IOR adaptation, ideation counter and store) and records nothing.
- A caller-supplied `now` (the battery's single-threaded world steps) keeps the pre-M0 clock reference (turn start).
  Equivalence with the pre-M0 tick (2e7392f12) on that clock is a test, `tests/test_tick_pre_m0_equivalence.py`: a
  stub scenario with every gated faculty armed (3 sessions x 5 ticks, D5 + sleep replay sharing ticks, an ideation
  tick, IOR) must reproduce a golden captured from the pre-M0 source (`tests/_capture_tick_pre_m0_golden.py`): the
  same 73 faculty calls in order, the same 15 inner-life records, budgets, IOR adaptation and ideation counter. A
  second test recomputes the golden from the pre-M0 commit when the clone has it. Swapping two consolidation steps
  or skipping the budget spend fails it. The battery's direct `da_tag_capture_chat.mark_awake` loop is not locked;
  the battery worker is single-threaded.
- Still outside the lock (other owners' `server.py` regions): the startup warm thread (`_warm_chat_brain`, and with
  `BRAIN_PREWARM` a scratch `brain_reply`), `/api/brain-chat/reset`, and the shutdown persistence flush.

### Tests

`tests/test_brain_turn_lock.py`, 41 tests, brain-free (the real `brain_chat` with build and `brain_reply` stubbed; the
real `tick_idle_sessions` with stub organs). They pin BEHAVIOUR, not reply identity: two sessions never overlap (max
concurrent 1) and the escape restores the overlap (max 2); FIFO arrival order; a releaser cannot jump a waiting
thread; a timed-out waiter leaves the queue clean; the `busy` reply after the bound, then normal service; the
turn-end stamp without a budget refill; a tick behind a same-session turn does not tick it when the turn ends; the
battery clock ignores the end stamp; the wander runs without the turn lock and a turn gets in within 0.2 s; a
same-session turn preempts the wander (dropped, recorded, no consolidation step); with 5 sessions x 0.3 s steps a
waiting turn waits < 0.45 s. Added after the re-review: the end stamp is written with the lock held (also when the
turn body raises); a self-init turn returns the mid-thought read-out within the bound while a wander runs and never
drives the busy organ; a reset during the wander leaves no wander state and no record in the fresh session (control:
without a reset the IOR write stays). Command: `.venv/bin/python -m pytest tests/test_brain_turn_lock.py -q`.

Single-caller byte identity, real brain (`research/runners/turn_lock_identity_check.py`, dev seed 7, numpy, stub
renderer, production LTM default, one session, 5 turns: teach, ask, build-KB ask, LTM ask, re-ask; three fresh
processes): sha256 of every response body equal for `BRAIN_TURN_LOCK=1` vs `=0` on 5/5 turns, and the null
(lock ON vs ON again, a second process) equal on 5/5, so the compare is readable. Verdict byte_identical = true.
`research/findings/raw/_one_brain_probe/turn_lock_identity_s7.json`. Peak RSS 6.5 GB per process; ~570 s per arm.

### Probe: `research/runners/one_brain_probe.py`, dev seed 7

Command (brain lock, memcap 14 GB, numpy, stub renderer, production LTM default = wikidata_100k):
`research/findings/raw/_one_brain_probe/m0_s7.json` (code 5ceec4bc4; the trimmed log `m0_s7.log` is gitignored, local only).

| arm | A_ask | B_ask | A_ask_again | B_ltm | B_build | matches prediction |
|---|---|---|---|---|---|---|
| intact | recall | abstain | - | recall | recall | yes |
| restart_persist_off | recall | abstain | abstain | recall | recall | yes |
| restart_persist_on | recall | abstain | **abstain** | recall | recall | **no** (predicted A_ask_again recall) |
| one_brain, freeze_learning, zero_learned_store, restart_persist_on_one_brain | UNDEFINED | | | | | needs BRAIN_ONE_BRAIN |

- Baseline confirmed: conversation B does not know what A was told (B: "I don't know about that. ... I haven't
  learned about wug yet"); both known-fact checks recall in B (LTM: angora_turkey facts via the `rf` LTM tier;
  build KB: dog chase cat), so B's abstain is readable.
- A restart loses the fact with persistence OFF, as predicted. With persistence ON it is ALSO lost: the prediction
  failed. Cause, from the run's own log: `save_session_facts` raised
  `AttributeError: 'Pool1BoundOneBrainComposer' object has no attribute 'concepts'`
  (`developed_brain_io.extract_grounded_codes` reads `comp.concepts`; the production default composer behind
  `TieredFactStore` has none). The error is logged and swallowed, and the shutdown hook still prints
  "flushed 1 session(s)". Only the xedge pool was written (2 checkpoints, 207,444 bytes); no facts checkpoint; no
  D5 checkpoint (no referential turn built a D5 organ). So on today's production default tiny-demo brain (`onebrain`
  composer), persistence keeps no taught fact for any session identity. The "same identity recovers it" half of the baseline cannot be shown until the
  facts save works. Logged in `research/FAILURE_LOG.md`.
- Contrasts (`tools.lab`): persistence flag -> A recalls after restart: UNCHANGED (abstain/abstain, attribution
  UNDEFINED, a null); persistence flag -> checkpoint files: 0 -> 4 (the flag did act; it wrote the xedge only).
- Disk (design section 4 asks for the number): xedge checkpoint ~104 KB each. D5 and facts checkpoint sizes are not
  measured yet (none was written).
- Turn lock in the probe (single caller): 2-5 acquisitions per phase, max wait 2 microseconds.

Peak RSS (`ru_maxrss` per worker process): one session 6.25 GB (restart phase 0); two sessions in one process
11.93 GB (intact), 11.87 GB (restart phase 1). Wall: intact 717 s; restart arms 522-524 s + 633-672 s. A recall
turn on numpy with `OMP_NUM_THREADS=1` takes ~120-135 s; the first teach turn ~390 s (build included).

### Persistence fix and the restart_persist_on rerun (review item 5)

- Gate first: two strict-xfail tests (061552d7c) on a brain-free stand-in shaped like the default composer (`kb` +
  `.comp.concepts`, no `.concepts`, behind the real `TieredFactStore`) reproduced the `AttributeError` and the empty
  save. Fix (e455d548f): `developed_brain_io.concept_codebook` reads `.comp.concepts` on the onebrain composer, for
  the save and for the reload's code injection; unrecruited `__free<k>__` reserve slots are not persisted. A flush
  counts a session only when its D5 store or facts were written; failures log at ERROR and the shutdown hook prints
  each one. The two gate tests are plain tests now, plus two for the error report and the shutdown count.
- Rerun at dev seed 7 against the committed prediction (code e455d548f,
  `research/findings/raw/_one_brain_probe/m0_review_persist_on_s7.json`): A_ask recall, B_ask abstain,
  **A_ask_again recall**, B_ltm recall, B_build recall; all five reads match the prediction written before any data.
  The shutdown hook printed "flushed 1 session(s); 0 had nothing to save; 0 FAILED". Written: one facts checkpoint
  for session A (61,694 bytes, facts + codes) and two xedge checkpoints (207,444 bytes). Against the first run's
  restart_persist_off (A_ask_again abstain), the persistence flag now moves the after-restart read abstain -> recall.
  Only the same session identity recovers the fact: B, a new identity, still abstains (persistence is keyed per
  session today; the one-brain arms are M2).
- Peak RSS: phase 0 6.25 GB (one session), phase 1 11.86 GB (two sessions). The after-restart A turn took 469 s
  (session build + reload + recall on numpy).

### Reserve slots across a reload (re-review item 1)

- Defect found by the re-review: the facts checkpoint saved codes but not which `__free<k>__` reserve slot each
  recruited word held, so the reload's re-store recruited words in re-store order. Order: teach "crow chase wren"
  (wren takes the first free slot), reconsolidate to "moth" (moth takes the next slot), save, reload: moth went into
  wren's old slot with its saved code while its own slot's placeholder kept the same code, and the next new word
  (heron) took that code. Two words, one code.
- Fix: the checkpoint also saves `slots` (`OneBrainComposer.recruited_slots`), and the reload calls
  `restore_recruit(word, slot, saved_code)` for each saved word, in slot order, before the code injection and the
  re-store. A checkpoint without `slots` (written before this) falls back to the old path.
- Gate: `tests/test_cross_session_persistence.py::test_reload_keeps_each_recruited_word_in_its_own_reserve_slot`
  (real `OneBrainComposer`, numpy, D=64, three reserve slots, that exact order: heron's code differs from moth's
  and wren's, both keep their saved slot and code, and `query_patient("crow", "chase")` reads moth after the
  reload), with a control that removes `restore_recruit` and reproduces the shared code. The round-trip test on the
  stand-in now checks that each reloaded code equals the saved one (it passed before with the codebook stubbed out
  during the reload; each stand-in instance now draws its own codes).

### Identity rerun after the review changes (review item 6)

`research/runners/turn_lock_identity_check.py --seed 7` at code e455d548f (FIFO lock, bounded wait, turn-end stamp,
restructured tick), `research/findings/raw/_one_brain_probe/turn_lock_identity_review_s7.json`. Scope: the real
handler called directly (no uvicorn, no idle tick running), stub renderer, numpy, production LTM default, one
session, five turns, three fresh processes. sha256 per body, first 12 hex: 954f3ae0a5b4, 31c4e0423a79,
e7f59c2f83ac, 3e3245f4b65a, 4a54b989b84b; equal for lock ON, OFF and ON again (the null) on 5/5 turns, and equal to
the first run's hashes (`turn_lock_identity_s7.json`, code 5ceec4bc4). Verdict byte_identical = true, readable.
Peak RSS 6.5 GB per process; ~600 s per arm.

