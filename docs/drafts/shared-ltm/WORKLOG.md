# shared-ltm: load the imported knowledge store once per server process (STEP 1 of "one brain, many conversations")

Branch `claude-build/shared-ltm` from `s/main` 6a3bfe247. Workstation, numpy CPU, dev seed 7. Units: rss_kb from
/proc is KiB; GiB = KiB / 1048576.

## Why

`webapp/server.py` `_build_chat_brain` runs once per conversation cache key `(session, brain, renderer)`. On the
tiny-demo path it calls `_load_or_build_ltm_store(_resolve_ltm_bundle(), seed=_brain_chat_seed())` and wraps the
session composer in `TieredFactStore(buffer, ltm)`. The developed-bundle path does the same inside
`developed_brain_io.load_developed_brain(ltm_bundle=...)`. The store is wikidata_100k (395 shards, 78,857 facts,
one 129-neuron `SimulationBridge` per fact under the substrate store). One copy is about 5.44 GiB
(`docs/drafts/battery-retention/WORKLOG.md`), and the server holds one per live conversation, with no limit.

## Step 1: the facts, established before any code

Code read at 6a3bfe247: `research/runners/tiered_fact_store.py`, `sharded_phasor_store.py`,
`rf_phasor_composer.py` (init, `_resonate`, `_filler_phases`, the batched scan, escalation, codebook cache,
substrate store and retrieve, the query methods, `_trace_scan`, `_rf_gauge`, the Izhikevich bank),
`sim/bridge.py` `rf_kick` / `rf_set_complex_weights` / `_rf_advance_one` / `rf_resonate_steps`,
`developed_brain_io.load_developed_brain`, and every caller that reaches `composer.ltm` (grep for `.ltm`,
`"ltm"`, `promote_buffer_to_ltm`, `update_on_mismatch`, `encoding_gain_fn`, `_retrieve_*`, `.shards`).

### (a) Is the LTM store read-only during chat?

Verdict: YES for the stored knowledge. Every write path into the store is unreachable from a chat turn. The
store does carry per-call scratch, lazy caches and trace instrumentation; only the trace instrumentation is
per-conversation state.

Writes (would change what the store holds):

| Path | Reached on a chat turn? |
| --- | --- |
| `ShardedPhasorStore.store` / shard `store` (kb append, `_alias_index` reset) | No. `TieredFactStore.store` writes the buffer only. |
| `TieredFactStore.promote_buffer_to_ltm` | No caller anywhere in the tree except its definition. |
| shard `update_on_mismatch` (reconsolidation, kb slot rewrite) | No. `composer.update_on_mismatch` resolves through `TieredFactStore.__getattr__` to the BUFFER; nothing calls it on `.ltm`. |
| shard `_filler_phases` runtime word growth (`concepts`, `words`, `_growth_rng`) | No. Only `_encode` calls it (store path). Queries compare decoded words to the cue; `_escalate_role_match` returns early on a word not in `concepts`. |
| `encoding_gain_fn`, `homeostatic_scaling`, `apply_homeostatic_scaling` (DA encoding, idle-tick homeostasis) | No. Set and called on `composer`, which `TieredFactStore.__setattr__` / `__getattr__` route to the buffer. |
| D5 learn-through-use, D6 Hebbian store, DA tag capture, reconsolidation organ | Buffer only (`d6_hebbian_chat`, `da_tag_capture_chat` unwrap `TieredFactStore` to `.buffer`). |
| sleep replay / idle ticks (`continuous_engine.tick_idle_sessions`, idle replay organ) | Read `composer.kb` (buffer) or call buffer methods; no `.ltm` access. |
| persistence (`_reload_persisted_learning`, session-learning save) | Re-stores taught facts through `composer.store` (buffer). The LTM is never saved from the server. |

Direct readers of `composer.ltm` (all read-only scans of `shard.kb` fact dicts): `compositional_chain_route.
_distinct_patients`, `gnw_multistep_deliberation` (stored-check), `rich_answer_composer._ltm_facts_about` (flag
`BRAIN_ELABORATE_FROM_LTM_SHARD`, plus `ltm._resolve_alias`).

Per-call state ON the store, and whether it is per-conversation:

| State | What it is | Per-conversation? |
| --- | --- | --- |
| store `_trace`, shard `trace` | the trace recording gate; the server sets `composer.trace = True` every turn and `TieredFactStore.__setattr__` forwards it to `ltm.trace` | YES (wrapped) |
| store `last_trace` | the answering shard's trace; the server never resets it on the LTM (`composer.last_trace = None` reaches the buffer only); `TieredFactStore._propagate_ltm_trace` copies it to the buffer after an LTM answer | YES (wrapped). Example: `chain_of_thought` dead-ends before any hop and leaves it unchanged, so it would show another conversation's trace. |
| shard `last_trace` | each shard's own last trace; `render_fact` and the per-fact `query_agent` loop never reset it, and `_note_trace` copies it up | YES (wrapped). Example: a `render_fact` falls through to a shard another conversation queried and reports that conversation's trace. |
| shard `_last_resonate_n` + the `_bridge_cache` bridges | the reused per-op RF bridge and the gauge pointer `_rf_gauge` reads | Safe to share. `rf_set_complex_weights` replaces all weights, `rf_kick` resets v, u, prev_im, fired, spike_step and the counter, so every op is a pure function of (weights, kick, period, lam, floor). `_rf_gauge` reads the bridge only inside `_trace_scan`, after that same call's resonates. |
| shard `_substrate_scan_bridge` / `_substrate_scan_key` | the consolidated block-diagonal retrieve bridge, keyed on handle identities | Safe: rebuilt only when kb changes (it does not), re-kicked per query. |
| shard `_scan_comps_cache` | the last batched retrieve, reused for the matched winner | Safe: written by `_scan_comps` and read by `_matched_comp` in the same call. |
| per-fact substrate bridges (`kb` handles) | `_retrieve_substrate` kicks the fact's own bridge (`render_fact`, per-fact fallback) | Safe: the kick resets state; the batched path only reads their weights. |
| `_cb_frac` / `_cb_z` / `_concept_row` (codebook cache), `_dg_index` (sparse index, default off), `_alias_index` | lazy caches, pure functions of `words` / `concepts` / `kb` | Safe: built once; nothing grows them during chat. |
| `_retrieve_noise_rng` | stateful read-noise RNG | Safe at the default: drawn only when `_retrieve_noise > 0`, which only de-risk scripts set. Sharing refuses a store with noise on. |
| `_izh_bank_cache` | Izhikevich WTA banks run through `_run_one_simulation_step`; the code resets only v and u | NOT shown pure. Reached only with `enable_spiking_cleanup` (never passed for the LTM) or `BRAIN_METACOG_SPIKING_MARGIN=1` (default off). Sharing refuses both. |

### (b) Thread safety

`/api/brain-chat` is a sync `def brain_chat`, so FastAPI runs each request in its threadpool. There is no lock
around a turn; the only lock is `_WARM_QWEN_LOCK` (warm-model load). The continuous-state tick runs
`tick_idle_sessions` in a thread executor next to live turns. So two conversations can be inside an LTM query at
the same time. Concurrent recall on ONE shared store is NOT safe: two threads would share a shard's
`_bridge_cache` bridge mid-resonate (one `rf_kick` resets the other's state), overwrite `_scan_comps_cache`
between `_scan_comps` and `_matched_comp`, and swap `last_trace`. A lock around every LTM call is needed. Reads of
`shard.kb` fact dicts by the direct readers above need none (kb is never mutated).

### (c) Does anything key the store's behaviour by session?

No. The seed is `_brain_chat_seed()` (process env `BRAIN_CHAT_SEED`), not per session; on the persisted fast
path the manifest's own seed (42), D and n_shards are used and the caller's seed is ignored. The cache key never
reaches the store. The developed path's build fallback seeds with the developed bundle's `brain.json` seed. The
construction env read inside `RFPhasorComposer` is `BRAIN_BATCHED_SUBSTRATE_SCAN`, `BRAIN_SHARD_SPARSE_INDEX`,
`BRAIN_METACOG_SPIKING_MARGIN`; the shared cache key includes them.

### Decision

Build it. The knowledge is read-only on every chat path; the only per-conversation state is the trace
instrumentation, which a per-conversation view keeps; the per-call scratch is shown pure; one lock per shared
store serializes LTM calls.

## Step 2: the mechanism

- `research/runners/shared_ltm.py`: `view_for(cache, key, loader)` loads the store once per key (under a guard, so
  two conversations building at once do not load two copies) and returns a new `SharedLTMView` for each
  conversation. The view has the store's read API; each read runs under the store's RLock with this conversation's
  trace state swapped in (store `trace` + `last_trace`, every shard's `last_trace`) and swapped out after.
  `trace` / `last_trace` are the view's own. Other attributes (`shards`, `shard_for`, `total_facts`, ...) read
  through. Writes (`store`, `update_on_mismatch`, setting any other attribute) raise `SharedLTMWriteError`.
  Copies of a view share the store. `unshareable_reason` refuses a store with spiking cleanup, the spiking recall
  margin or read noise on (the stateful read paths of step 1); such a store is returned uncached, a private copy.
- `webapp/server.py`: `_SHARED_LTM` (the process cache), `_shared_ltm_on()` (`BRAIN_SHARED_LTM`, default on via
  `_SHARED_LTM_DEFAULT_ON`), `_shared_ltm_key(...)` = (realpath of the bundle, seed, D, n_shards, codebook cache,
  decode escalation, substrate store, batched substrate scan, the three composer-construction env values), and
  `_ltm_store_for_conversation(...)`. The tiny-demo branch of `_build_chat_brain` calls it where it called
  `_load_or_build_ltm_store`. With the flag off it makes exactly that call, argument for argument.
- The developed-bundle branch uses the same cache and loader (`_load_or_build_ltm_store` mirrors
  `load_developed_brain`'s own LTM handling; build-fallback seed = the bundle's `brain.json` seed via
  `_developed_ltm_seed`) and hands the view to `load_developed_brain(ltm_store=...)` with `ltm_bundle=None`.
  `ltm_store` is a new keyword, default None. With the flag off the call is the old one (`ltm_bundle=...`,
  `ltm_store=None`).
- The reply's `source` string is unchanged ("tiny-demo +LTM"), so replies can be compared byte for byte.

## Step 3: tests (`tests/test_shared_ltm.py`, brain-free, 21 tests, ~4 s)

- Two conversations get the identical store object: at `_ltm_store_for_conversation` (loader called once) and
  through the real `_build_chat_brain` for the tiny-demo branch (brain build stubbed) and the developed branch
  (`load_developed_brain` stubbed, checks `ltm_store` / `ltm_bundle` and the brain.json seed).
- The escape `BRAIN_SHARED_LTM=0` gives distinct objects, through the pre-change call argument for argument, and
  caches nothing. Flag parsing for unset / blank / on / off values.
- The cache key changes with each of: bundle path, seed, D, n_shards, codebook cache, decode escalation,
  `BRAIN_LTM_SUBSTRATE_STORE`, `BRAIN_BATCHED_SUBSTRATE_SCAN`, `BRAIN_SHARD_SPARSE_INDEX`,
  `BRAIN_METACOG_SPIKING_MARGIN`; a symlinked bundle maps to the same key; a flag changed between conversations
  loads a second store.
- Mutation guard: sha256 over every fact's substrate weights, the codebook, roles, vocab and both RNG states is
  unchanged after every read the chat path issues through two views (unknown words included).
- Equivalence: two conversations' interleaved reads (hits, misses, `render_fact` after another conversation used
  that shard, dead-end chains) on views over ONE store equal two private stores, answer and `last_trace`, op by op.
- The lock: two threads querying through two views stay correct. Writes through a view raise.
- Negative controls (run by hand, scratch script): removing the trace swap fails the equivalence test (at
  `render_fact('cat')`); a read that grows the vocabulary fails the mutation guard; removing the lock fails the
  concurrency test. So each test can fail.
- Existing suites still pass: test_tiered_fact_store, test_sharded_phasor_store, test_arm_renderer_production_path,
  test_battery_session_release, test_rank2_integrated_loop_thread, test_brain_prewarm_scratch_session (76 passed,
  1 skipped with test_shared_ltm), test_communicable_turn_wirein (8 passed).

## Step 4: one real run, three conversations, both arms (numpy CPU, dev seed 7)

Driver `docs/drafts/shared-ltm/three_sessions.py` calls the real `brain_chat` handler (tiny-demo, stub renderer,
`SIM_DISABLE_LLM=1`, `BRAIN_CHAT_SEED=7`) for sessions s1, s2, s3. Turn 1 builds each session back to back; turns 2
and 3 then interleave across sessions (s1, s2, s3), so s2 and s3 read the store after s1 used it. Turns:
"what does the dog chase" (a tiny-demo buffer fact), "what country is chelsea fc from?" (recalls
`[chelsea_fc, country, united_kingom]`, a fact that exists only in wikidata_100k, so the LTM answered),
"what does the dragon do" (untaught: buffer and LTM both abstain). RSS is VmRSS after each session's first turn.
`compare.py` hashes nothing itself; it compares the sha256 each arm recorded over the handler's whole JSON body.

Commands (each arm under the brain lock; `run_arm.sh` = `mem_ok.sh <cap>` then
`memcap.sh <cap> -- env SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 [BRAIN_SHARED_LTM=0] python -u
docs/drafts/shared-ltm/three_sessions.py docs/drafts/shared-ltm/runs/<arm>.json`):

    flock -w 7200 /home/dant123/Projects/sim-rev/brain.lock docs/drafts/shared-ltm/run_arm.sh off 20
    flock -w 7200 /home/dant123/Projects/sim-rev/brain.lock docs/drafts/shared-ltm/run_arm.sh on 14
    python docs/drafts/shared-ltm/compare.py docs/drafts/shared-ltm/runs/off.json docs/drafts/shared-ltm/runs/on.json

The off arm needs a 20 GiB cap, not 14: three private copies peak at 17.80 GiB, so it cannot run under 14. Both
exited 0. Per-turn records: `runs/off.json`, `runs/on.json` (the bodies are included).

| | off (`BRAIN_SHARED_LTM=0`) | on (default) |
| --- | --- | --- |
| RSS after s1 built | 6,192,752 KiB (5.91 GiB) | 6,192,480 KiB (5.91 GiB) |
| RSS after s2 built | 11,732,828 KiB (11.19 GiB) | 6,493,592 KiB (6.19 GiB) |
| RSS after s3 built | 17,328,092 KiB (16.53 GiB) | 6,654,440 KiB (6.35 GiB) |
| added by s2 / by s3 | +5,540,076 / +5,595,264 KiB (5.28 / 5.34 GiB) | +301,112 / +160,848 KiB (0.29 / 0.15 GiB) |
| process peak (VmHWM) | 18,659,232 KiB (17.80 GiB) | 7,071,332 KiB (6.74 GiB) |
| LTM store objects | 3 (`ShardedPhasorStore`) | 1 (`SharedLTMView` x3, one store; one `_SHARED_LTM` entry) |
| s2 / s3 build turn | 141.6 s / 142.6 s | 77.6 s / 79.8 s (no LTM reload) |
| LTM recall turn s1 / s2 / s3 | 29.9 / 29.3 / 30.2 s | 30.4 / 29.3 / 29.4 s |

Byte identity: 9/9 turns have the same sha256 in both arms (whole reply body, per (session, turn)):

| turn | s1 | s2 | s3 |
| --- | --- | --- | --- |
| dog chase | 00e5253cc912b9b8 | e7095a6f4c343036 | e7095a6f4c343036 |
| chelsea fc country (LTM) | 9063a984bb29b739 | 9063a984bb29b739 | 9063a984bb29b739 |
| dragon (abstain) | b243040cff0a2c2e | b243040cff0a2c2e | b243040cff0a2c2e |

(s1's first reply differs from s2/s3's in both arms alike: the first session in a fresh process differs from later
ones, independent of this change.)

## Open risks

- LTM reads now serialize across conversations (one lock per store). Under concurrent chat, a conversation's LTM
  query can wait for another's (~0.5-1 s each on cupy, more on numpy); before, private copies could overlap. Not
  measured here (the run was sequential).
- The shared store lives for the process; there is no eviction (it is the point). Each new key (a changed LTM flag
  or bundle) adds one more ~5.4 GiB entry.
- Direct `shard.kb` readers (chain route, multistep, elaboration) bypass the view's lock. Safe while nothing
  writes kb; a future write path through `view.shards[i]` would not be refused (only the view's API is guarded;
  the mutation-guard test covers the read API).
- The Izhikevich-bank read paths (spiking cleanup, `BRAIN_METACOG_SPIKING_MARGIN=1`) and read noise are refused
  for sharing, not proven safe: with any of them on, each conversation still gets its own copy.
- The developed-bundle branch is covered by the brain-free test only (stubbed `load_developed_brain`); the real
  run covered the tiny-demo path, the production default.
- `docs/plans/2026-10-13-one-brain-shared-memory-DESIGN.md` (landed on s/main 79ba4b0a1, after this branch point)
  names M1's validation as the battery §9.1 reply hashes ON vs OFF plus the per-session RSS drop. This branch ran
  the three-session real-handler check above, not the battery §9.1 pair; that pair is still to run before the merge.
  The design's M0 `turn_lock` will sit outside this per-store lock (this one is held only inside one store call), so
  the two nest without a lock-order cycle.
