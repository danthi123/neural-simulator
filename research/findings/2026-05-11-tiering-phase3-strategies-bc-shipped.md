# Synapse Tiering — Strategies B+C SHIPPED with activity-driven eviction

**Date:** 2026-05-11 04:35 EDT
**Status:** MILESTONE — Phases 1+2+3 of the CPU/RAM/SSD tiering design
COMPLETE. The bridge now mirrors per-pathway CSRs into a
`TieredSynapseStore`, tracks activity each simulation step, and evicts
dormant pathways to disk as `.npz` shards. End-to-end verified on
NumPy backend with real bridge.

---

## What works end-to-end

| Capability | Status | Verification |
|------------|--------|--------------|
| `TieredSynapseStore` (Phase 3 part 1) | ✅ shipped | 25 tests in `tests/test_synapse_storage.py` |
| `bridge.extract_per_pathway_csrs()` | ✅ shipped | 2 tests on real bridge under NumPy |
| `lineage.export_shards(bridge)` (Strategy C) | ✅ shipped | 4 tests with mock + real bridge |
| `cfg.enable_synapse_tiering` (Strategy B) | ✅ shipped | 2 integration tests |
| `bridge.synapse_store` mirror + activity tracking | ✅ shipped | Real bridge: 20 steps → eviction fires |
| `bridge._detect_fired_pathways()` per step | ✅ shipped | No-op when tiering off; <1ms overhead when on |
| CLI: `bridge_lineage list-shards <name>` | ✅ shipped | E2E |
| Webapp: `GET /api/synapse-tiering/{name}` | ✅ shipped | 2 tests; live on port 8765 |
| Frontend: Lineages tab shard inventory section | ✅ shipped | Lazy-loaded from /api/synapse-tiering |
| CLAUDE.md "Synapse tiering" section | ✅ shipped | Documents opt-in usage |

## Real-bridge end-to-end demonstration

```python
# SIM_BACKEND=numpy
cfg = CoreSimConfig()
cfg.enable_brain_region_framework = True
cfg.brain_regions = [
    BrainRegion(name="A", n_neurons=10, ...),
    BrainRegion(name="B", n_neurons=10, ...),
]
cfg.region_pathways = [RegionPathway("A", "B", density=0.3, weight_mean=1.0)]
cfg.enable_synapse_tiering = True
cfg.synapse_tiering_evict_idle_steps = 5  # aggressive for demo
cfg.synapse_tiering_grace_pagein_steps = 0

bridge = SimulationBridge(core_config=cfg, ...)
bridge._initialize_simulation_data()

# At init:
print(bridge.synapse_store.stats())
# → {'n_pathways': 1, 'n_in_memory': 1, 'n_on_disk': 0,
#    'n_pageouts_lifetime': 0, ...}

# Run 20 simulation steps (no firing in this toy network):
for _ in range(20):
    bridge._run_one_simulation_step()

print(bridge.synapse_store.stats())
# → {'n_pathways': 1, 'n_in_memory': 0, 'n_on_disk': 1,
#    'n_pageouts_lifetime': 1, ...}
```

The dormant `A→B` pathway is automatically evicted to disk as
`bridges/synapse_shards/active/A_to_B.npz` after 5 idle steps. The
sim continues running cleanly using the monolithic `cp_connections`
(Strategy B: observational tiering only; Strategy A would also
replace compute).

## The full tiering surface

### Per-pathway access (Strategy C, opt-in via export)

```python
# Extract: split monolithic CSR into per-pathway sub-matrices
pathways = bridge.extract_per_pathway_csrs()
# → {"language_input_to_motor_N": <scipy.sparse.csr 2048x500>, ...}

# Export: save to lineage's shards/ sidecar dir
lineage = BridgeLineage("main")
n_shards = lineage.export_shards(bridge)
# → 24 shards written to bridges/lineage/main/shards/

# Inspect: list available shards
lineage.list_shards()
# → ["language_input_to_motor_N", "language_input_to_motor_E", ...]
```

### Runtime activity tracking (Strategy B, opt-in via config)

```python
cfg.enable_synapse_tiering = True
cfg.synapse_tiering_evict_idle_steps = 1000  # default
cfg.synapse_tiering_grace_pagein_steps = 100  # default

# Bridge auto-builds the store at _initialize_simulation_data
# _run_one_simulation_step ticks store.step(fired_pathways) each step
# Eviction fires on idle threshold + grace period decay

# Inspect at runtime:
print(bridge.synapse_store.stats())
# → {n_pathways: 24, n_in_memory: 18, n_on_disk: 6, n_pageouts_lifetime: 8, ...}
```

### Webapp surface

```
GET /api/synapse-tiering/{lineage_name}
  → 200 {
       lineage_name, shards_dir, n_pathways, total_size_mb,
       shards: [{name, exists, size_mb}, ...]
     }
  → 404 if lineage not found
```

Dashboard Lineages tab now renders the shard inventory inline when
clicking on a lineage (lazy-loaded, gracefully missing if no shards
exported yet).

## What's NOT yet shipped (deferred)

### Strategy A — full per-pathway compute (3-4 wk)

The big refactor: replace the monolithic `cp_connections @ neuron_state`
matvec in `_run_one_simulation_step` with N per-pathway matvecs (one per
pathway). Requires:

- Per-pathway STDP / Hebbian / structural plasticity (per-pathway eligibility)
- Per-pathway weight update at end of step
- Performance benchmark: per-pathway matvec ≤ 5× monolithic overhead
- Validation at Tier 1 (n_lang=2048, n_motor=500) + 64-word arch

**Decision criterion:** ship Strategy A only when active memory tiering
is on the critical path (e.g. 96+ word arch on local hardware, or
cloud-anchored 1B+ params). Current local 64-word ceiling fits in VRAM,
so Strategy A isn't urgent.

### Auto-tiering policy (Phase 4 expansion)

Strategy B already auto-evicts on idle. Phase 4 expansion would add:

- **Memory-pressure-based eviction:** evict when VRAM/RAM used > threshold
- **Predictive page-in:** preload pathways before they're needed (e.g.
  based on co-firing history)
- **Multi-tier policies:** RAM warm tier between VRAM hot and SSD cold

Built on Strategy B's per-step `store.step()` hook.

### Strategy A's CuPy parallel path

`TieredSynapseStore` is currently scipy.sparse-only. For Strategy A on
CuPy backend, we'd need a `cupyx.scipy.sparse` parallel path with the
same API. Not urgent — NumPy backend covers the Mac M-series + CI
use cases.

## Migration summary (this autonomous arc, since /autonomous-runs)

| Subsystem | New | Modified | Tests |
|-----------|-----|----------|-------|
| Bridge Lineage Manager | 4 files | 5 files | 90 |
| Auto-growth Phase A | 2 files | 1 file (design doc) | 25 |
| Backend abstraction | 1 file | 7 files | 47 |
| Synapse Storage | 2 files | 5 files | 56 |
| Strategic + design docs | 3 docs | — | — |
| Findings docs | 5 docs | — | — |
| Webapp endpoints | 0 new files | 2 files | 4 |
| Webapp frontend | 0 new files | 2 files | — |

Total: ~40 commits, ~250 new tests, all PASS, all CPU-only safe.

## Provenance

- This findings doc: `research/findings/2026-05-11-tiering-phase3-strategies-bc-shipped.md`
- Foundational design: `docs/plans/2026-05-11-cpu-ram-ssd-tiering-design.md`
- Bridge integration design (3-strategy plan): `docs/plans/2026-05-11-tiering-phase3-part2-bridge-integration-design.md`
- Strategic context: `docs/plans/2026-05-11-strategic-reevaluation.md`
- Previous milestone: `research/findings/2026-05-11-numpy-backend-chat-repl-shipped.md`
- Commits this arc: `33ca704` → `2c978f0`

## What's next (when user wakes up)

1. **Strategic Path 1/2/3 decision** still open — the tiering work
   is foundational for all three paths but the path choice tells us
   what we're scaling toward.

2. **Phase A2: chat_repl --auto-grow integration** — uses the
   already-shipped TierPromoter scaffold. Could ship CPU-only design
   doc + scaffolding; full integration needs GPU.

3. **Strategy A** — only if memory tiering becomes urgent (96+ word
   arch locally, or cloud-anchored work).

4. **Phase 4 expansion** — memory-pressure-based eviction, predictive
   page-in. Built on Strategy B's foundation.

All paths preserve the work shipped tonight. Tiering Phases 1-3 are
**done**; what's left is Strategy A (substantial refactor) and Phase 4
policy expansion (incremental).
