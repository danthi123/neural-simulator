# battery-retention (c5-next(b)): what the battery worker kept per session, and the fix

Branch `claude-build/battery-retention` from `s/main` a41082d54. Workstation, numpy CPU, OMP_NUM_THREADS=1,
`BRAIN_CHAT_SEED=7` (dev seed), memcap 14 GB. One brain process at a time.

## What is retained

- `webapp/server.py` `brain_chat` caches one ChatBrain per `(session, brain, renderer)` key in `_BRAIN_CHATS`
  (plus ~20 per-session registries under the same key) and frees them only on `req.reset`. The worker runs every
  turn in one process, so each distinct session stays alive to the end. The step is per session, not per turn kind:
  bc_a and bc_b share session `bc` and add nothing.
- Inside each ChatBrain, the big part is the tiny-demo's long-term-memory tier. `_build_chat_brain` attaches
  `_load_or_build_ltm_store(wikidata_100k)` to every new ChatBrain as `TieredFactStore(composer, ltm)`. With the
  substrate store on (the default since RANK-6/#211), every LTM fact is held as its own 129-neuron
  `SimulationBridge` (`RFPhasorComposer._store_substrate`).
- `ltm_size_probe.py` loads only that LTM store, with no brain build (memcap 9): 395 shards, 78,857 facts, 78,857
  `SimulationBridge` handles. RSS goes up 5,709,276 KiB (about 72.4 KiB per fact). Deleting the store and running
  gc + malloc_trim gives 5,372,704 KiB back. The battery step it explains is emo to bc_a, +5,690,652 KiB (below).

## Fix (commit fefd940a7)

- `webapp/server.py`: the `req.reset` teardown is moved word for word into `_evict_brain_chat_session(cache_key)`.
  It is the same pops in the same order, and reset calls it. `brain_chat_cache_key(req)` returns the handler's own
  key expression.
- `research/runners/onebrain_regression_battery.py`: `session_release_plan` releases a session once nothing later
  in the turn list can read it. That point is after its last turn, or after the last world step that follows its
  first turn, whichever comes later; world steps tick every live session through `_SESSION_MOOD` and
  `_LAST_REQUEST`. Release means the reset teardown, then gc.collect and malloc_trim.
- `BATTERY_KEEP_SESSIONS=1` restores the old behaviour, where every session is kept to the end.
- No sharing of the LTM between sessions was attempted. The LTM is mutable per session (trace, promote_buffer_to_ltm,
  codebook cache), so sharing could change replies.

## Evidence (turns emo,bc_a,bc_b; per-turn `battery_turn` rss_kb)

| turn | BEFORE (pristine a41082d54) | AFTER (fefd940a7) |
|---|---|---|
| emo | 6679772 | 6677732 |
| bc_a | 12370424 | 6287396 |
| bc_b | 12370600 | 6287384 |
| process hwm | 12454096 | 6834412 |

The two output JSONs are byte-identical in full, with no fields excluded. Both have sha256
`1b6dfe69a2e0ca972c19d6dc58325d4f8e66e21f554458cfee64ae64ac20dd01`. The release of `emo` runs before the `bc`
session builds, so this exercises the mechanism directly. The per-turn RSS sidecars are
`before_seed7_emo_bc_rss.jsonl` and `after_seed7_emo_bc_rss.jsonl`. The two reply JSONs are not committed: the
device-and-cost gate refuses an output without a provenance sidecar, and the runner writes sidecars only under
`research/findings/raw`. Both arms ran with `SIM_BACKEND=numpy`; re-run either one with
`--worker --turns emo,bc_a,bc_b` (plus `BATTERY_KEEP_SESSIONS=1` for the old behaviour).

## Scope

- Shows: the worker no longer holds one LTM-bearing ChatBrain per session, and on this list the replies are unchanged.
- Does not show: byte-identity on lists with world steps, interleaved sessions or rich/selfinit turns. The release
  plan keeps sessions alive through world steps by construction, and the CPU tests cover that, but no brain run did.
- Does not change: the webapp server itself. A live server still builds a 5.7 GB LTM tier per conversation
  session and keeps it until reset. That is a separate production design question, possibly a shared read-only
  LTM, and it is not touched here.
