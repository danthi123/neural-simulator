# NumPy backend SHIPPED — hardware-independent sim end-to-end

**Date:** 2026-05-11 03:03 EDT
**Status:** Phase 2 of the CPU/RAM/SSD tiering design SHIPPED in a
single autonomous arc, substantially ahead of the 1-3 week design
estimate. The simulator can now construct, initialize, run simulation
steps, and save/load checkpoints **without any NVIDIA / CUDA / CuPy
dependency.**
**Trigger:** User (2026-05-11) — strategic direction toward CPU-only
+ RAM + SSD tiered architecture for hardware independence + larger
working sets.

---

## What works end-to-end on `SIM_BACKEND=numpy`

| Capability | Status | Verification |
|------------|--------|--------------|
| Backend selection (`SIM_BACKEND=numpy`) | ✅ | `python -c "from sim.backend import get_backend; print(get_backend())"` |
| `SimulationBridge.__init__` | ✅ | 50-neuron bridge constructs with `cfg.num_neurons=50` |
| `_initialize_simulation_data` (Watts-Strogatz) | ✅ | 495 synapses generated, STP + STDP + structural plasticity initialized |
| `_run_one_simulation_step` (single step) | ✅ | Returns without error in <1 ms at n=50 |
| Batch sim stepping | ✅ | 500 steps × 200 neurons in 105ms (0.21 ms/step) |
| Brain region framework | ✅ | 2 BrainRegions + 1 RegionPathway, 70 neurons, 398 synapses, 100 steps in 19ms |
| `save_checkpoint` | ✅ | HDF5 checkpoint written; size ~470 KB for n=50 |
| `load_checkpoint` | ✅ | Loaded into a fresh bridge cleanly |
| Lineage workflow (BridgeLineage) | ✅ (by transitivity) | Lineage uses save_checkpoint + load_checkpoint internally |
| All 195 lightweight tests | ✅ | No regression for existing CuPy users |

## What this unlocks

1. **Mac M-series compatibility** (via NumPy + scipy.sparse; MLX backend stub
   in place for native acceleration later).
2. **Linux servers without GPUs** (containerized deployment, cloud CPUs).
3. **Windows / WSL without RTX cards** (most user-facing scenarios).
4. **CI/CD without GPU runners** — `SIM_BACKEND=numpy pytest` works.
5. **Algorithmic verification** — toy-scale numerical parity between
   CuPy + NumPy backends.
6. **Foundation for Phase 3 (SSD synapse paging)** — the tier-promotion +
   sparse-shard machinery sits on the NumPy code path.

## Performance honesty

NumPy is **dramatically slower than CuPy for sparse-heavy workloads**.
Per the design doc's prediction (10-50× slowdown), early measurements:

| Workload | CuPy | NumPy | Slowdown factor |
|----------|------|-------|-----------------|
| 1 sim step @ n=200 | ~0.05 ms | 0.21 ms | ~4× (small network, Python overhead dominates) |
| Brain region step @ n=70 | ~0.02 ms | 0.19 ms | ~10× |

At larger scales (16-word arch, 28K neurons, 32M synapses) the
slowdown will be more severe (estimated 30-50× per the design doc).
That's acceptable — the use cases for the NumPy backend are
verification + CI + low-end hardware, NOT peak training. The CuPy
backend remains the production speed target.

## Migrations executed across the arc

A single autonomous arc (~3 hours) shipped:

### Phase 1 (design doc: ~1 week)

| Component | Sites | Pattern |
|-----------|-------|---------|
| `sim/backend.py` | new module | `xp` abstraction + 12 helpers (get_backend, fuse, synchronize, to_host, from_host, set_device, get_device_mem_info, get_device_properties, get_memory_pool, get_pinned_memory_pool, get_memory_pool_used_mb, is_gpu_backend) |
| `sim/kernels.py` | 15 sites | `@cp.fuse()` → `@fuse()` (no-op on NumPy) |
| `sim/connectivity.py` | import block | Backend-aware via `get_sparse_module()` |
| `sim/bridge.py` | import block | Backend-aware + defensive bootstrap fallback |

### Phase 2 (design doc: 1-3 weeks)

| Pattern | Count | Replacement |
|---------|-------|-------------|
| `cp.asnumpy(arr)` | 47 | `_backend_to_host(arr)` |
| `.get()` on scalar | 5 | `int(...)` / `bool(...)` wrap |
| `.get()` on CSR fields | 3 | `_backend_to_host(...)` |
| `cp.cuda.Device(0).use()` | 1 | `_backend_set_device(0)` |
| `cp.cuda.Device().mem_info` | 7 | `_backend_get_device_mem_info()` |
| `cp.cuda.runtime.getDeviceProperties()` | 1 | `_backend_get_device_properties()` |
| `cp.get_default_memory_pool()` | 3 | `_backend_get_memory_pool()` |
| `cp.cuda.Device().synchronize()` (profiling) | 7 | `_backend_synchronize()` |
| `cp.cuda.memory.OutOfMemoryError` | 1 | Defensive `getattr` chain |
| `import cupyx.scipy.sparse as csp` (function-local) | 2 | Removed; use module-level csp |

**Total: 77 CuPy-specific call sites migrated.** Plus 37 new tests on
`tests/test_backend.py` (27 abstraction + 10 device-helper).

## Code paths verified

- ✅ Bridge construction + GPU init under both backends
- ✅ Random network generation (Watts-Strogatz on numpy: scipy.sparse coo→csr)
- ✅ Single-region init (50 neurons WS network)
- ✅ Brain region framework init (multi-region + pathways)
- ✅ Simulation step end-to-end (Izhikevich dynamics, conductance, plasticity)
- ✅ STP, STDP, structural plasticity, homeostasis (all init + step on NumPy)
- ✅ Checkpoint save (HDF5 via h5py; backend transparent)
- ✅ Checkpoint load (re-init + apply weights cleanly)
- ✅ Profiling instrumentation (synchronize calls route through backend)

## Code paths not yet exercised on NumPy (likely-clean but unverified)

- Recording playback (`record_current_frame_if_active`, `gpu_playback_cache`)
- Bio three-factor training loop (`bio_three_factor.run_three_factor`)
- Chat REPL (`chat_repl.run_repl`)
- Replica wiring (`sim/replicas.py`)
- Visual cortex pathway (`sim/visual_cortex.py`)
- Neuromodulator subsystem (`sim/neuromodulators.py`)

These will be verified incrementally as they're exercised. Any
remaining CuPy-isms will be patched as they surface (the same
mechanical pattern: `_backend_to_host()`, `int()`-wrap scalars,
avoid `.get()` on arrays).

## Roadmap forward

| Phase | Status | Scope estimate | Actual |
|-------|--------|----------------|--------|
| 1: xp abstraction | ✅ SHIPPED | 1 week | ~1 hour |
| 2: NumPy backend passes tests | ✅ SHIPPED (core paths) | 1-3 weeks | ~2 hours |
| 3: SSD synapse paging | Pending | 2-3 weeks | — |
| 4: Activity-driven auto-tiering | Pending | 1 week | — |

**Phase 3 (SSD paging) can begin from this foundation.** The NumPy
backend's CSR matrices are scipy.sparse.csr_matrix — already
serializable via msgpack/np.savez/h5py without GPU round-trip. The
tier-store design from the tiering doc is directly applicable.

## Honesty caveats

1. **Performance is NOT a goal of this work.** The NumPy backend exists
   for portability + verification + low-end hardware. Production
   training stays on CuPy. Don't measure NumPy backend speed and
   conclude "the sim is slow now" — it's a different code path.

2. **Not every CuPy-ism is patched.** The 77 sites covered are the
   ones the verified code paths hit. Deeper paths (recording, bio
   training, visual cortex) may surface more `cp.x` issues; pattern
   is well-established now.

3. **Numerical drift between backends is expected.** Different BLAS
   implementations, different reduction orders, different RNG. The
   test suite should use tolerance-based comparisons for any
   cross-backend equivalence checks.

4. **No multi-GPU work.** Out of scope for this design; the existing
   single-device model is preserved.

## Provenance

- This findings doc: `research/findings/2026-05-11-numpy-backend-shipped.md`
- Design: `docs/plans/2026-05-11-cpu-ram-ssd-tiering-design.md`
- Strategic context: `docs/plans/2026-05-11-strategic-reevaluation.md`
- Backend module: `sim/backend.py` (37 tests at `tests/test_backend.py`)
- Commits this arc: `ab5500f`, `a568b91`, `49dbaa4`, `d607363`, `7dd147c`, `bd0de5a`
