# Rank-6 synaptic fact-store — VRAM-reduction options (fallback if the flip's VRAM verdict is tight)

**Context:** the owner-APPROVED production flip (`enable_substrate_store=True` default-ON, branch
`research/substrate-store-flip`) is gated on a GPU VRAM measurement (queued). If the substrate-store's ~3.8-5GB
(estimated) VRAM at 78,857-fact scale is too tight alongside the spiking mouth + the rest of the brain on the 24GB
3090, these CONCRETE code-grounded options let the flip land anyway. (Code-grounded research 2026-09-07.)

**Root cause:** `_store_substrate` builds a full `SimulationBridge` PER FACT (`_build_rf_bridge(1+D)`,
`rf_phasor_composer.py:40-59,1461`); `ShardedPhasorStore` (n_shards=4) does NOT pool them — each shard holds its
per-fact bridges eagerly. The 78,857× multiplier is the problem.

## Ranked options
**#1 — Lazy per-shard (de)materialization (RECOMMENDED, lowest-risk).** In `sharded_phasor_store.py`: keep only
the hot shard(s) materialized on cupy; demote cold shards to plain host-side composites (the pre-existing
`enable_substrate_store=False` numpy `kb` representation this class already supports) and rebuild bridges on the
next cue routing there (LRU, e.g. keep 1-2 of 4 resident). VRAM → ~hot/n_shards of today (~25-50% at n_shards=4).
Cost: paid ONLY on a shard cache-miss (topic/agent switch to a cold shard) — a serial rebuild (~tens of s for a
~19,714-fact shard) to HIDE via background pre-warm or more shards. No cross-backend hazard; reuses validated
store/retrieve paths.

**#2 — Dedicated numpy WORKER PROCESS (near-zero VRAM).** A `multiprocessing` worker that sets
`SIM_BACKEND=numpy` BEFORE its first `import sim.bridge`, exposing store/retrieve over a Pipe; `_store_substrate`/
`_retrieve_substrate` gain a `host`-backend IPC branch. ⛔ Do NOT flip the backend IN-PROCESS: `sim.bridge` binds
`cp` at import as a module global; `get_backend("numpy")` only flips a sticky cache and corrupts the live cupy chat
path (documented `webapp/server.py:~3569` hazard). Must be a genuinely separate process. Cost: IPC round-trip per
fact touched (small 128-complex payload) + worker lifecycle engineering.

**#3 — Do REGARDLESS (free, stacks with either): float32 downcast.** `rf_set_complex_weights`
(`sim/bridge.py:7833-7834`) builds `w_re`/`w_im` as float64 → cupy; switch to float32 → halves the weight-matrix
VRAM slice, zero architecture risk. (But per-bridge VRAM is dominated by the ~25-30 n-sized IZH state arrays
`_initialize_simulation_data` allocates, not the D-nonzero weights, so this alone saves ~10-20% total.)

**REJECTED — pooled single bridge across all facts:** disqualified on QUERY LATENCY — `_rf_advance_one` does a
full sparse matvec over the WHOLE `(n,n)` matrix every resonate step; consolidating all facts → nnz≈F·D (~10.1M),
so every query pays O(F·D) × ~208 steps regardless of which one fact is read (no early-exit). Breaks the
query-time-comparable property the flip passed on. Not recommended.

**Combine (#1+#2):** run the cold tier through the numpy worker (fetch from host RAM instead of rebuilding) → hot/cold
split + near-zero VRAM floor for cold facts, at the cost of doing both.

_Decision procedure: (1) run the queued cupy VRAM measurement; (2) if it fits the 24GB budget alongside the brain +
mouth → land the flip as-is; (3) if tight → apply #3 free + #1 lazy-shards (likely enough); (4) if still tight → #2
worker. The flip lands in all cases; only the mechanism differs._
