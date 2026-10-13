# One brain, shared memory: WORKLOG

Design: `docs/plans/2026-10-13-one-brain-shared-memory-DESIGN.md` (merge 2e7392f12). Branch
`claude-build/one-brain-m0`, cut from s/main 2e7392f12.

## M0 (2026-10-13)

### Turn lock

- `webapp/brain_turn_lock.py`: one process-level `threading.RLock`, `brain_turn(where)` context manager,
  escape `BRAIN_TURN_LOCK=0` (read per call), wait-time statistics in `stats()`.
- `webapp/server.py` `brain_chat`: the lock covers `_pin_bridge_backend`, the reset, the cache lookup + build +
  persisted-learning reload, and the whole `brain_reply` pipeline. `mark_request` moved ahead of the lock, so a turn
  that is waiting already reads as active to the idle tick.
- Placement of the mouth: the Qwen render is called from inside `brain_reply` (`chat.render`, the rich composer), so
  it is INSIDE the lock. Moving it out needs `brain_reply` split into decide / articulate; that is M2 work.
- Wait time: the `/api/brain-chat` response has no timing or debug field, so nothing was added to it (an added key
  would change every body). It is recorded in `brain_turn_lock.stats()` and in each probe phase record
  (`turn_lock`: n_acquired, max_wait_s).
- Not built in M0: the design's bounded queue with a `busy` status on timeout. Turns wait without limit.

### Idle tick and world steps (investigation)

- Path: `server.py` startup task `_continuous_state_tick` runs `continuous_engine.tick_idle_sessions` every
  `IDLE_SEC`=20 s in a thread executor (`run_in_executor`), default ON (`BRAIN_CONTINUOUS`). It reads the
  PROCESS-SHARED affect organ (`read_differential`) and, per session, the self-init wander organ, D5 consolidation and
  sleep replay (writes the D5 store), the persistence save, Turrigiano scaling on the composer, the DA tag-capture
  ledger and the #91/#92 relax drives. Its only guard was "no request for 20 s", which does not hold off a turn
  on another session (shared affect organ) or a turn longer than 20 s.
- Fix: `tick_idle_sessions` takes the same lock once per session, with the idleness check inside the lock (a turn
  marked while the tick waited is skipped). A waiting turn gets in between two sessions' tick steps.
- The regression battery's world steps (`_run_world_step`, `_run_vclock_step`) call the same `tick_idle_sessions`,
  so they take the lock too. Their direct `da_tag_capture_chat.mark_awake` loop does not; the battery worker is
  single-threaded, so nothing races there today.
- Still outside the lock (out of M0's scope, `server.py` regions owned elsewhere): the startup warm thread
  (`_warm_chat_brain`: builds the default ChatBrain and the shared organs; with `BRAIN_PREWARM` it runs
  `brain_reply` on a scratch session), `/api/brain-chat/reset` (pops the registries), and the shutdown persistence
  flush. Each can overlap a live turn; the first is the only one that touches organs on a running server.

### Tests

`tests/test_brain_turn_lock.py`, 20 tests, brain-free (the real `brain_chat` with build and `brain_reply` stubbed):
two sessions on two threads never overlap (max concurrent 1); with `BRAIN_TURN_LOCK=0` the same instrument sees them
overlap (max 2); single-caller bodies equal ON vs OFF; a failed build releases the lock; the idle tick reads the
affect organ under the lock, waits for a turn in flight, skips a session marked while it waited, and takes no lock
under the escape. Neighbouring suites green (battery release, sleep replay, persistence, prewarm: 59 passed).

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
