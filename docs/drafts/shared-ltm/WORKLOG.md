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
