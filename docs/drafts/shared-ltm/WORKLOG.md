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
| shard `_last_resonate_n` + the `_bridge_cache` bridges | the reused per-op RF bridge and the gauge pointer `_rf_gauge` reads | Safe to share. `rf_set_complex_weights` replaces all weights, `rf_kick` resets v, u, prev_im, fired, spike_step and the counter, so every op is a pure function of (weights, kick, period, lam, floor). `_rf_gauge` reads the bridge only inside `_trace_scan`, after that same call's resonates. MISSED here, found in review: a COLD build of such a bridge reseeds the process-global RNGs (see Review fixes). |
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
store serializes LTM calls. (Review correction: one side effect was missed. Building a bridge reseeds the
process-global RNGs, so a cold lazy build is not side-effect free; see Review fixes.)

## Step 2: the mechanism (as first built, bf91d5747; the review changes are in Review fixes below)

- `research/runners/shared_ltm.py`: `view_for(cache, key, loader)` loads the store once per key (under a guard, so
  two conversations building at once do not load two copies) and returns a new `SharedLTMView` for each
  conversation. The view has the store's read API; each read runs under the store's RLock with this conversation's
  trace state swapped in (store `trace` + `last_trace`, every shard's `last_trace`) and swapped out after.
  `trace` / `last_trace` are the view's own. Other attributes (`shards`, `shard_for`, `total_facts`, ...) read
  through. Writes (`store`, `update_on_mismatch`, setting any other attribute) raise `SharedLTMWriteError`.
  Copies of a view share the store. `unshareable_reason` refuses a store with spiking cleanup, the spiking recall
  margin or read noise on (the stateful read paths of step 1); such a store is returned uncached, a private copy.
- `webapp/server.py`: `_SHARED_LTM` (the process cache), `_shared_ltm_on()` (`BRAIN_SHARED_LTM`; default on at
  bf91d5747, OFF since the review), `_shared_ltm_key(...)` = (realpath of the bundle, seed, D, n_shards, codebook cache,
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

Ran at the first build (sharing default on, no RNG isolation; driver 62161b040). The review's evidence run with RNG
isolation, seeds 7 and 8 and a longer turn list is Step 5.

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

## Review fixes (review of bf91d5747: FIX_FIRST, 2026-10-13)

1. DEFAULT OFF. `_SHARED_LTM_DEFAULT_ON = False` (design section 5: default-off, then flip). Unset env makes the
   pre-change call. `BRAIN_SHARED_LTM=1` opts in; the coordinator flips it after validation.
2. GLOBAL RNG LEAK (major). `SimulationBridge._initialize_simulation_data` runs `_initialize_rng(cfg.seed)`
   (sim/bridge.py), which calls `cp.random.seed`, `np.random.seed` and `random.seed`. The LTM builds bridges at
   seed 42: one per fact at load, and lazily on a shard's first recall (`_resonate`, `_retrieve_all_substrate`,
   `_spiking_cleanup`, `_izh_bank`). So a cold recall resets every global stream to seed 42, whatever
   BRAIN_CHAT_SEED is, and with a shared store a warm shard skips the reseed, so ON and OFF diverge on later global
   draws (e.g. a chat-path bridge's OU noise). The developed path at bf91d5747 also loaded the shared store BEFORE
   `load_developed_brain`, so its reseed landed before the brain build instead of after it.
   - `research/runners/global_rng_guard.py` `preserved_global_rng()`. numpy: `get_state` / `set_state`. python
     `random`: `getstate` / `setstate`. cupy: `cupy.random.seed` reseeds the CURRENT RandomState object in place,
     so the guard swaps a temporary RandomState into the device slot for the block and puts the original object
     back, untouched. Checked on the cupy backend: same object back, next draw equal to the no-build draw, while
     the unguarded control differs.
   - `RFPhasorComposer._isolate_global_rng` (default False = byte-identical). When set, every bridge the composer
     builds goes through the guard (`_new_rf_bridge`, `_izh_bank`).
   - `webapp/server.py`: `BRAIN_LTM_RNG_ISOLATE` (`_ltm_rng_isolate_on()`). Forced on whenever the shared store is
     on; otherwise default off (`_LTM_RNG_ISOLATE_DEFAULT_ON = False`), so plain OFF is unchanged. When on,
     `_load_ltm_isolated` runs the whole load under the guard and sets every shard's flag. This holds both for the
     shared store and for a private copy (OFF + isolate). The developed path with either flag on loads through the
     same function and passes `ltm_store=`. Plain OFF passes `ltm_bundle=` as before.
   - research/FAILURE_LOG.md row (the reseed defect, named tests).
3. Evidence: Step 5.
4. Minors.
   - A test of the REAL `load_developed_brain(ltm_store=...)` branch on a small saved developed bundle:
     TieredFactStore installed over the given view; an LTM-only fact is recalled; the buffer fact and the moat hold.
   - Cache key for a persisted bundle (manifest with n_shards) = (realpath, the (mtime_ns, size) of manifest.json,
     facts.json and composites.npz), not seed / D / n_shards. `ShardedPhasorStore.load` takes those from the
     manifest, so tiny-demo (BRAIN_CHAT_SEED) and a developed bundle (its own seed) share one copy, and a rebuilt
     bundle is a new key. A facts-only bundle keys on seed, D, n_shards and the facts.json stamp. Both keys also
     carry the four LTM flags and the constructor env.
   - Per-key load locks (`shared_ltm._LOAD_LOCKS`). `_KEYS_GUARD` is held only for dict lookups, never across a
     load, so a cached key never waits behind another key's load. Two conversations on one cold key still load
     once.
   - `test_cache_key_covers_every_env_read_in_the_store_modules` (re-review: was constructors only) scans the WHOLE
     source of rf_phasor_composer.py and sharded_phasor_store.py for env reads, fails on any name missing from
     `_SHARED_LTM_COMPOSER_ENV`, and fails on any read whose name is not a string literal.
   - `research/runners/_knowledge_scale_100k_production_verify.py`: the "EXACT call" docstrings now say that call is
     the default-flags path, and that with either flag on the server passes `ltm_store=`.

Tests after the fixes: `tests/test_shared_ltm.py` 31 passed. Composer / LTM / server suites (test_rf_phasor_composer,
test_decode_escalation_seed44_hole, test_sharded_phasor_store, test_tiered_fact_store, test_shared_ltm,
test_arm_renderer_production_path, test_battery_session_release, test_rank2_integrated_loop_thread,
test_brain_prewarm_scratch_session, test_communicable_turn_wirein): 149 passed, 5 skipped.

## Step 5: review evidence, seeds 7 and 8, three arms (numpy CPU)

Driver `three_sessions.py <out> v2` through the real `brain_chat` handler (tiny-demo, stub renderer,
`SIM_DISABLE_LLM=1`). Sessions s1, s2, s3 are built back to back on turn 0; every later turn interleaves s1, s2, s3.
Turns (v2): t0 "what does the dog chase" (buffer); t1 "what country is chelsea fc from?" (LTM);
t2 "what does the cat eat" (buffer, after an LTM recall); t3 "what sport is chelsea fc in?" (LTM);
t4 "what does the brain use" (buffer, after an LTM recall); t5 "what does the dragon do" (both tiers abstain).
Arms: `on` = BRAIN_SHARED_LTM=1 (isolation forced); `offiso` = BRAIN_SHARED_LTM=0 BRAIN_LTM_RNG_ISOLATE=1; `off` =
BRAIN_SHARED_LTM=0 BRAIN_LTM_RNG_ISOLATE=0 (today's default). Commands, each under the brain lock, chained by
`run_all.sh`:

    flock -w 7200 /home/dant123/Projects/sim-rev/brain.lock docs/drafts/shared-ltm/run_arm.sh <on|offiso|off> 20 <7|8> v2
    # run_arm.sh: mem_ok.sh 14, then memcap.sh 20 -- env SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1
    #   BRAIN_CHAT_SEED=<seed> <arm flags> python -u docs/drafts/shared-ltm/three_sessions.py runs/<arm>_s<seed>_v2.json v2
    python docs/drafts/shared-ltm/compare.py runs/offiso_s7_v2.json runs/on_s7_v2.json    # and the other pairs

All six exited 0. The code was identical across the runs: the recorded SHAs 58053c4c2, 5a97bae78, f287aac58 and
9d1a528a5 differ only by committed run JSONs. `git_dirty` true on the seed-7 runs is the tracked provenance log
`research/findings/raw/_provenance/runs.jsonl`, which runs append to (reverted, not committed).

Result: ON vs OFF+isolate is 18/18 identical at seed 7 and 18/18 at seed 8 (whole reply body, sha256 per (session,
turn)). OFF+isolate vs plain OFF is also 18/18 at both seeds. So the hygiene fix changed no reply on these turns;
the global stream the LTM reseeded is not read by these reply paths in a way that changes a reply. That is not
proof for other paths: the unit tests show the leak itself and that isolation removes it. The hashes are the same
in all three arms, so one table per seed (first 16 hex digits):

Seed 7:

| turn | s1 | s2 | s3 |
| --- | --- | --- | --- |
| t0 dog chase | 00e5253cc912b9b8 | e7095a6f4c343036 | e7095a6f4c343036 |
| t1 chelsea country (LTM) | 9063a984bb29b739 | 9063a984bb29b739 | 9063a984bb29b739 |
| t2 cat eat | d9c8917cc1a050ad | d9c8917cc1a050ad | d9c8917cc1a050ad |
| t3 chelsea sport (LTM) | 3705e909126473e9 | 3705e909126473e9 | 3705e909126473e9 |
| t4 brain use | 0b9655ade8807a33 | 0b9655ade8807a33 | 0b9655ade8807a33 |
| t5 dragon (abstain) | 2870f3efc2c170b9 | 2870f3efc2c170b9 | 2870f3efc2c170b9 |

Seed 8:

| turn | s1 | s2 | s3 |
| --- | --- | --- | --- |
| t0 dog chase | 2e71d2a5480ee265 | ddea2ea8eb24ea1a | ddea2ea8eb24ea1a |
| t1 chelsea country (LTM) | 4278bb69bf72a52a | 4278bb69bf72a52a | 4278bb69bf72a52a |
| t2 cat eat | bf476a16873d9fd8 | bf476a16873d9fd8 | bf476a16873d9fd8 |
| t3 chelsea sport (LTM) | 28b19518773a1986 | 28b19518773a1986 | 28b19518773a1986 |
| t4 brain use | 0e6b5bcaf12abed2 | 0e6b5bcaf12abed2 | 0e6b5bcaf12abed2 |
| t5 dragon (abstain) | ebc3350e7b336164 | ebc3350e7b336164 | ebc3350e7b336164 |

RSS (VmRSS after each session's build turn; KiB, GiB = KiB / 1048576):

| arm | seed | s1 | s2 | s3 | added by s2 / s3 | VmHWM |
| --- | --- | --- | --- | --- | --- | --- |
| on | 7 | 6,195,388 | 6,485,684 | 6,662,660 | +290,296 / +176,976 (0.28 / 0.17 GiB) | 7,403,744 (7.06 GiB) |
| offiso | 7 | 6,196,292 | 11,733,712 | 17,327,368 | +5,537,420 / +5,593,656 (5.28 / 5.33 GiB) | 19,682,444 (18.77 GiB) |
| off | 7 | 6,190,720 | 11,732,652 | 17,325,172 | +5,541,932 / +5,592,520 (5.29 / 5.33 GiB) | 19,681,240 (18.77 GiB) |
| on | 8 | 6,148,808 | 6,480,528 | 6,661,232 | +331,720 / +180,704 (0.32 / 0.17 GiB) | 7,401,540 (7.06 GiB) |
| offiso | 8 | 6,205,368 | 11,730,260 | 17,327,760 | +5,524,892 / +5,597,500 (5.27 / 5.34 GiB) | 19,680,776 (18.77 GiB) |
| off | 8 | 6,114,108 | 11,732,452 | 17,325,816 | +5,618,344 / +5,593,364 (5.36 / 5.33 GiB) | 19,680,676 (18.77 GiB) |

The private-copy arms peak at 18.77 GiB under a 20 GiB cap: three private copies plus six turns leave about 1.2 GiB
of headroom.

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
- The developed-bundle branch is covered by brain-free tests: the server branch with `load_developed_brain`
  stubbed, plus the real `ltm_store=` branch on a small saved bundle. No real-handler run on a developed bundle;
  the runs covered the tiny-demo path, the production default.
- The RNG guard serializes guarded blocks on one module RLock (re-review): two guarded loads on two threads no
  longer interleave their save/restore. RESIDUAL: a thread that draws from a global stream WITHOUT the guard while
  another thread is inside a guarded block (a cold load or a cold lazy build) reads the block's seed-42 values. The
  guard restores the streams afterwards; it cannot hide the window from unguarded readers. A whole cold load holds
  the lock for minutes, so a concurrent unguarded draw in that window is possible under concurrent chat.
- The RNG guard covers bridge BUILDS. Per-call global draws inside an LTM query would not be covered. None exist on
  the default read path (`rf_kick` / `rf_resonate_steps` draw nothing); the Izhikevich-bank paths that step the
  full simulation are refused for sharing.
- `docs/plans/2026-10-13-one-brain-shared-memory-DESIGN.md` (landed on s/main 79ba4b0a1, after this branch point)
  names M1's validation as the battery §9.1 reply hashes ON vs OFF plus the per-session RSS drop. That pair is the
  FLIP validation (before `_SHARED_LTM_DEFAULT_ON` becomes True), not a merge precondition: the branch merged default
  OFF on the three-session real-handler check above. The battery §9.1 pair has not run yet.
  The design's M0 `turn_lock` will sit outside this per-store lock (this one is held only inside one store call), so
  the two nest without a lock-order cycle.
