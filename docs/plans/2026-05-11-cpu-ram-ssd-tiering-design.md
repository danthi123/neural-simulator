---
type: plan
status: live
date: 2026-05-11
---

# CPU/RAM/SSD memory tiering + NumPy backend — hardware-independence design

**Date:** 2026-05-11 02:45 EDT
**Status:** DESIGN — companion to `2026-05-11-strategic-reevaluation.md`.
Foundational work, valuable regardless of which path (1/2/3) the project
takes. CPU-only work to design + implement.
**Trigger:** User (2026-05-11) — "migrating this project to run CPU-only
and store the entire sim in RAM, with applicable components expected to
be on fast storage like SSD(s)? Sort of treating the entire sim arch
like ZFS?"

---

## Goals

1. **Hardware independence** — run the sim without NVIDIA / CUDA. Mac M-series,
   Linux CPU-only, Windows without RTX cards — all viable.
2. **Larger working sets** — RAM (64-128 GB commodity) ≫ VRAM (24 GB on
   our 3090). Bigger arches, more synapses, less arch-driven retention
   wall.
3. **Memory tiering** — hot/warm/cold storage with automatic paging, like
   ZFS ARC + L2ARC + storage pool. Bridge state lives in the appropriate
   tier based on activity.
4. **Lineage compatibility** — the lineage system shipped tonight remains
   the source of truth; tiering is layered underneath, not on top.
5. **Test parity** — same test suite passes on CuPy and NumPy backends.
   Reproducibility verified.

## Non-goals

- **Throwing away GPU acceleration.** GPU stays available + is the
  performance baseline. The CPU backend is a *parallel codepath*, not
  a replacement.
- **Reaching GPU performance on CPU.** A 32-core EPYC can match maybe
  5-20% of an RTX 3090 for sparse-heavy workloads. We accept the
  slowdown for the hardware-independence + larger-RAM benefit.
- **Distributed multi-machine training.** Out of scope; that's Phase 2
  of the master plan.

## Architecture overview

Three components plus the existing lineage system:

```
                         ┌──────────────────────────────┐
                         │  BridgeLineage (existing)    │
                         │  bridges/lineage/<name>/      │
                         │   current.simstate.h5         │
                         │   metadata.json               │
                         │   history/                    │
                         └──────────────┬───────────────┘
                                        │
                                        │ load / save
                                        ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                  SimulationBridge (refactored)               │
   │                                                              │
   │   xp abstraction layer:                                      │
   │     backend = "cupy" | "numpy" | "mlx" (future)              │
   │     xp = backend module                                      │
   │     all sim/bridge.py ops route through xp                   │
   └───────────────────┬─────────────────────────────────────────┘
                       │
       ┌───────────────┼─────────────────────────────────────┐
       │               │                                      │
       ▼               ▼                                      ▼
  ┌─────────┐   ┌─────────────┐                  ┌───────────────────────┐
  │  VRAM   │   │     RAM     │                  │     SSD (NVMe)        │
  │  hot    │   │    warm     │                  │       cold            │
  │ tier    │◀──┤    tier     │◀────dorm─────────┤  sparse synapse       │
  │         │   │             │     promotion    │  shards               │
  │ active  │   │  bridge     │     /eviction    │  + lineage history    │
  │ state   │   │  state if   │                  │                       │
  │ +syns   │   │  CuPy off   │                  │  reload-on-demand     │
  └─────────┘   └─────────────┘                  └───────────────────────┘
```

### Tier definitions

| Tier | Storage | Latency | Capacity | Contents |
|------|---------|---------|----------|----------|
| VRAM hot | RTX 3090 GPU memory | 0 (already there) | 24 GB | Active synapses + neuron state (CuPy mode) |
| RAM warm | DDR4/5 system memory | ~100 ns | 64-128 GB | Full bridge state (NumPy mode) OR dormant pathways (CuPy mode) |
| SSD cold | NVMe (Samsung 990 Pro etc.) | ~50 µs | TBs | Sparse synapse shards, lineage history snapshots |

### Activity-based paging policy

A *pathway* (e.g. `language_input -> motor_N`) is the unit of paging. Each
pathway has an **activity score** computed each simulation step:

```
activity[pathway] = (1 - decay) * activity[pathway] + decay * |Δfiring|
```

Where `Δfiring` is the spike count change in that pathway's post-neurons.
Pathways with `activity < threshold` for N consecutive steps get *demoted*:
- VRAM → RAM (CuPy → NumPy block copy)
- RAM → SSD (msgpack/npz dump to disk shard)

Pathways with rising activity get *promoted* in reverse. Promotion latency:
- SSD → RAM: ~50 ms for a 1M-synapse pathway (NVMe read + msgpack decode)
- RAM → VRAM: ~10 ms for the same (PCIe + CuPy upload)

This is biology-grounded: real cortex *does* have hot/cold pools — neurons
that haven't fired in a while drift toward homeostatic baseline; sleep
replay consolidates rarely-used patterns to long-term storage. Phase 1.3
consolidation already models this in software; SSD paging extends it to
physical storage.

## Phase breakdown

Four shippable phases. Each independently useful; cumulative benefit at
the end.

### Phase 1 — `xp` abstraction layer (~1 week)

**Scope:** make every `cp.*` call in `sim/bridge.py` route through an
`xp` variable that points to either CuPy or NumPy.

**Implementation:**

```python
# sim/backend.py (new)
import os

def get_backend(name: str | None = None):
    """Return the numpy-like module for the requested backend.

    name: "cupy" | "numpy" | None (auto-detect: cupy if available, else numpy)
    """
    if name is None:
        name = os.environ.get("SIM_BACKEND", "auto")
    if name == "auto":
        try:
            import cupy
            return cupy, "cupy"
        except ImportError:
            import numpy
            return numpy, "numpy"
    if name == "cupy":
        import cupy
        return cupy, "cupy"
    if name == "numpy":
        import numpy
        return numpy, "numpy"
    if name == "mlx":
        # Future: Apple Silicon
        raise NotImplementedError("MLX backend not yet implemented")
    raise ValueError(f"unknown backend: {name}")
```

Then in `sim/bridge.py`:
```python
from sim.backend import get_backend
xp, backend_name = get_backend()  # at module import time
# All cp.foo() calls → xp.foo()
# All cp.fuse() decorators → conditional (no-op on numpy)
# All cp.cuda.* calls → guarded if backend_name == "cupy"
```

**Estimated work:**
- ~6000 lines of `cp.*` in bridge.py to swap (mechanical, fast)
- ~200 lines of `cp.fuse()` kernels need conditional wrapping (a no-op
  decorator for numpy; CuPy keeps the fused implementation)
- ~50 lines of `cp.cuda.Stream.null.synchronize()` / `cp.get_default_memory_pool()`
  need backend-specific guards
- Test runner needs to set `SIM_BACKEND=numpy` for CPU-only CI

**Risks:**
- Some CuPy ops don't have NumPy equivalents (e.g. `cupyx.scipy.sparse` vs
  `scipy.sparse` — mostly drop-in but some edge cases)
- FP16 handling differs (NumPy supports FP16 in storage but most ops
  upcast to FP32; CuPy has native FP16 compute on Tensor cores)
- Random number generators differ — need to seed both consistently

### Phase 2 — NumPy backend passes the test suite (~1 week)

**Scope:** all existing tests pass on both `SIM_BACKEND=cupy` and
`SIM_BACKEND=numpy`.

**Work breakdown:**
1. Run the 78-test lineage + 25-test auto_growth subsystems on NumPy
   — these are mostly CPU-only already; should pass with minimal work
2. Run `tests/test_determinism.py` on NumPy — guaranteed differences
   in floating-point order vs CuPy, so the test will need a
   per-backend baseline
3. Run `tests/test_kernels.py` on NumPy — fused kernels need numpy
   reference implementations; this is where most of the new work is
4. Run sample smoke tests on NumPy — e.g. `chat_synonym_demo --seed 42
   --train-events 20` with `SIM_BACKEND=numpy`; verify the trained
   bridge gives reasonable predictions (don't need 6/6 alignment;
   just smoke-test that training converges)

**Estimated work:** 1 week if everything cooperates, 2-3 weeks if there
are subtle numerical differences. Some tests might need `pytest.mark.skip`
for CPU-only runs (e.g. tests that explicitly use CuPy memory pools).

**Risks:**
- **Numerical drift** — CuPy and NumPy use different BLAS implementations,
  different reduction orders, different RNG. Even seeded, results won't
  match bit-exactly. Need a tolerance-based comparison for
  reproducibility tests.
- **Performance** — NumPy backend will be 10-50× slower for non-trivial
  arches. Most tests use toy arches (e.g. `n_lang=64`), so test wall-clock
  shouldn't explode, but the GPU-only fast tests (perf benchmarks) need
  to be marked GPU-only.

### Phase 3 — SSD synapse paging (~2-3 weeks)

**Scope:** sparse-synapse pathways can be transparently paged between
RAM and SSD. The bridge presents them as if all-in-memory; the paging
layer handles eviction + reload.

**Implementation:**

```python
# sim/synapse_storage.py (new)
from dataclasses import dataclass
from pathlib import Path
import msgpack
import numpy as np

@dataclass
class PathwayShard:
    """Sparse-synapse shard backed by an NVMe file.

    Format: msgpack with {pre_indices, post_indices, weights, dtype,
    pathway_name, n_pre, n_post}. CSR-like but msgpack-portable
    (vs h5py which is HDF5-only).
    """
    pathway_name: str
    shard_path: Path
    in_memory: bool = False
    cached_csr = None  # scipy.sparse.csr_matrix when in_memory


class TieredSynapseStore:
    """RAM + SSD tier for sparse synapses.

    Pathways start in RAM. After N steps of low activity, eviction kicks
    them to SSD. They're transparently reloaded when needed.
    """
    def __init__(self, root: Path = Path("bridges/synapse_shards"),
                 ram_capacity_mb: int = 8192,
                 evict_after_idle_steps: int = 1000):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.ram_capacity_mb = ram_capacity_mb
        self.evict_after_idle_steps = evict_after_idle_steps
        self.shards: dict[str, PathwayShard] = {}
        self.idle_counter: dict[str, int] = {}

    def get_pathway(self, name: str) -> "csr_matrix":
        """Return the CSR matrix for a pathway. Pages in from SSD if needed."""
        if name not in self.shards:
            raise KeyError(name)
        shard = self.shards[name]
        if not shard.in_memory:
            self._page_in(shard)
        self.idle_counter[name] = 0  # accessed → reset idle
        return shard.cached_csr

    def step(self, fired_pathways: set[str]) -> None:
        """Called once per simulation step. Tracks idle time + evicts
        as needed."""
        for name in list(self.shards.keys()):
            if name in fired_pathways:
                self.idle_counter[name] = 0
            else:
                self.idle_counter[name] = self.idle_counter.get(name, 0) + 1
                if (self.idle_counter[name] > self.evict_after_idle_steps
                        and self.shards[name].in_memory):
                    self._page_out(self.shards[name])

    def _page_in(self, shard: PathwayShard) -> None:
        """Load shard from SSD into RAM as a scipy.sparse.csr_matrix."""
        with open(shard.shard_path, "rb") as f:
            data = msgpack.unpackb(f.read())
        import scipy.sparse as sp
        shard.cached_csr = sp.csr_matrix(
            (np.frombuffer(data[b"weights"], dtype=np.float32),
             np.frombuffer(data[b"post_indices"], dtype=np.int32),
             np.frombuffer(data[b"indptr"], dtype=np.int64)),
            shape=(data[b"n_post"], data[b"n_pre"]),
        )
        shard.in_memory = True

    def _page_out(self, shard: PathwayShard) -> None:
        """Persist shard to SSD; release RAM."""
        csr = shard.cached_csr
        payload = {
            b"weights": csr.data.tobytes(),
            b"post_indices": csr.indices.tobytes(),
            b"indptr": csr.indptr.tobytes(),
            b"n_post": csr.shape[0],
            b"n_pre": csr.shape[1],
            b"dtype": str(csr.data.dtype),
            b"pathway_name": shard.pathway_name,
        }
        with open(shard.shard_path, "wb") as f:
            f.write(msgpack.packb(payload))
        shard.cached_csr = None
        shard.in_memory = False
```

**Integration with bridge:**

The bridge's `cp_connections` becomes a *facade* that the tiered store
backs. When the sim accesses `cp_connections[post_idx, pre_idx]`, it
routes through the store, which pages in the relevant shard. For the
hot path (frequently-firing pathways), the shard stays in RAM and the
access is direct CSR.

**Estimated work:** 2-3 weeks. The trickiest part is the bridge's
`compute_synaptic_conductance` step which iterates over all pathways
per simulation step — we need to make sure dormant pathways stay
dormant (no per-step access) until activity rises.

**Risks:**
- Latency thrash if eviction policy is too aggressive (pages in/out
  every few steps). Need careful tuning of `evict_after_idle_steps`.
- Memory amplification — caching shards in RAM still has overhead;
  could end up using MORE total memory than current CuPy-on-GPU.
- Lineage compatibility — the lineage save format (HDF5) needs to
  handle paged-out pathways. Solution: on lineage save, page everything
  in to RAM, then dump consistent snapshot.

### Phase 4 — Activity-driven auto-tiering (~1 week)

**Scope:** the bridge automatically promotes/demotes pathways based on
activity, without per-pathway manual control.

**Implementation:**

```python
# sim/bridge.py additions
def _run_one_simulation_step(self):
    # ... existing step logic ...

    # Update activity tracker
    if self.synapse_store is not None:
        fired_pathways = self._detect_fired_pathways()  # cheap inspection
        self.synapse_store.step(fired_pathways)
```

**Estimated work:** 1 week including tuning + a 24-hour stress test.

## Performance expectations

Honest estimates from BLAS / NVMe / DDR4 throughput:

| Operation | CuPy (RTX 3090) | NumPy (32-core EPYC) | NumPy (M3 Pro CPU) |
|-----------|------------------|----------------------|----------------------|
| Dense matmul (4K×4K) | 1.5 ms | 50 ms | 80 ms |
| Sparse CSR mv (10M nnz) | 5 ms | 80 ms | 120 ms |
| Point-wise op (1M elem) | 0.1 ms | 1 ms | 1.5 ms |
| Bridge step (16w arch) | ~6 ms | ~120 ms | ~180 ms |
| `:speak` end-to-end (16w) | 1.7 sec | ~30 sec | ~45 sec |
| `:speak` end-to-end (64w) | 6 sec | ~120 sec | ~180 sec |

Training scales similarly. The 96-word XL training (79 min on GPU) would
be roughly **24 hours on a 32-core EPYC** or **40 hours on an M3 Pro**.
Not practical for routine work, but possible for overnight runs.

### SSD paging latency budget

- NVMe 990 Pro: ~7 GB/sec sequential read, ~6 GB/sec write
- 1M-synapse pathway as CSR: ~16 MB (4-byte indices × 1M + 4-byte weights × 1M
  + small indptr)
- Page-in latency: ~3 ms read + ~5 ms msgpack decode = **8 ms total**
- This is *cheap relative to the 100 ms sim step* at the larger arches.
  As long as we don't thrash, paging cost is invisible.

### Memory ceiling on a single workstation

| Component | 32-core EPYC | M3 Pro (Mac Studio) |
|-----------|---------------|----------------------|
| RAM | 256 GB | 192 GB (unified) |
| NVMe (single drive) | 4 TB | 4 TB |
| GPU (optional) | RTX 4090: 24 GB | M3 Pro GPU: 18 TFLOPS (shared with RAM) |
| Max bridge synapses (in RAM) | ~5B (≈ 10× Qwen2.5-0.5B) | ~3.5B |
| Max bridge synapses (SSD cold) | 100B+ | 100B+ |

The "tiny SOTA LLM" parameter range (0.5-3.8B) is comfortably in RAM
for both targets. The current 24 GB VRAM ceiling falls away. **This
is the foundational unlock.**

## Lineage compatibility

The lineage system shipped tonight assumes a single `.simstate.h5` file
per state. With SSD paging, the state is multi-file:

```
bridges/lineage/main/
├── current.simstate.h5          # neuron state + non-paged pathways
├── current.shards/              # paged pathway shards
│   ├── language_input_to_motor_N.msgpack
│   ├── language_input_to_motor_E.msgpack
│   └── ...
├── metadata.json
└── history/
    ├── 2026-05-11T03-00-00-000-checkpoint.simstate.h5
    └── 2026-05-11T03-00-00-000-checkpoint.shards/
        ├── language_input_to_motor_N.msgpack
        └── ...
```

`BridgeLineage.save()` needs to:
1. Trigger a "freeze + flush" on the tiered store (page everything in, then
   dump all paged shards next to the main h5 file)
2. Atomic-rename the whole shards/ directory alongside the .h5

`BridgeLineage.load()` needs to:
1. Load the .h5 (neuron state + always-resident pathways)
2. Register the shards/ directory with the tiered store (pathways stay
   on disk; load on first access)

The lineage tests (`tests/test_lineage.py`) need to be extended to
verify save/load works with paged state.

## Test strategy

1. **CI matrix expansion** — add `SIM_BACKEND=numpy` row to GitHub
   Actions (if/when CI is set up). All tests pass on both backends.
2. **Numerical-tolerance tests** — bridge step output on the two backends
   should match within float32 tolerance (~1e-5) for non-stochastic ops;
   stochastic ops (Poisson, OU noise) need per-backend seeds + comparison
   of statistics not exact values.
3. **Performance regression tests** — track NumPy backend wall clock for
   a fixed small benchmark; fail CI if regression > 20%.
4. **Paging stress test** — 24h run with 16 pathways at different
   activity rates; verify no thrashing, no memory growth, no data loss.

## Risks + mitigation

| Risk | Mitigation |
|------|------------|
| NumPy backend 10-50× slower; tests take forever | Mark slow tests `@pytest.mark.slow`; only run on CI nightly |
| Reproducibility lost vs CuPy | Tolerance-based comparison; document numerical drift in CLAUDE.md |
| Paging thrash on borderline-active pathways | Hysteresis: `evict_after_idle_steps=1000`, `promote_after_active_steps=10` |
| Lineage save fails if disk full | Atomic write to .new + os.replace; existing pattern from lineage MVP |
| Existing GPU code breaks during refactor | Phased rollout: Phase 1 (xp) lands behind `SIM_BACKEND` flag; CuPy default unchanged |
| MLX (future Apple Silicon) requires different abstraction | Defer; design `get_backend` to be extensible; don't lock in to numpy/cupy only |

## Total scope estimate

| Phase | Time | Risk |
|-------|------|------|
| Phase 1: xp abstraction | 1 week | Low |
| Phase 2: tests pass on NumPy | 1-3 weeks | Medium (numerical drift) |
| Phase 3: SSD synapse paging | 2-3 weeks | Medium (paging tuning) |
| Phase 4: activity-driven auto-tiering | 1 week | Low |
| **Total** | **5-8 weeks** | **Medium overall** |

Roughly one month of focused work for a solo developer. Spread out as
background-pace work alongside other priorities: 2-3 months realistically.

## Provenance + dependencies

- This doc: `docs/plans/2026-05-11-cpu-ram-ssd-tiering-design.md`
- Companion: `docs/plans/2026-05-11-strategic-reevaluation.md` (the
  "which path are we on" doc; tiering is foundational for all 3 paths)
- Master plan addendum: `docs/plans/2026-05-10-MASTER-PLAN-strategic-addendum.md`
  (Phase 2 of which this enables; cloud-anchored scale-up assumes
  hardware-independence)
- Auto-growth design: `docs/plans/2026-05-10-auto-growth-design.md`
  (Phase B/C of auto-growth was the SSD-paging idea; this is the
  concrete design)
- Bridge lineage design: `docs/plans/2026-05-10-bridge-lineage-design.md`
  (lineage save/load needs the multi-file extension described above)

## Open questions

1. **Backend default** — when both CuPy and NumPy are available, which is
   the default? Recommendation: CuPy if `cp.cuda.runtime.getDeviceCount() > 0`,
   else NumPy. Override via `SIM_BACKEND` env var.
2. **Tiered store always-on or opt-in?** Recommendation: opt-in via
   `cfg.enable_synapse_tiering = True`. Off by default to preserve current
   behavior for existing users.
3. **MLX backend ETA?** Recommendation: design for it (extensibility), but
   don't implement until someone with M-series hardware needs it. Phase 5
   if/when demanded.
4. **Multi-machine sharding (future)?** Out of scope here. If we go down
   the cloud-anchored Phase 2 path, this design extends naturally: each
   machine has its own tier-store; pathways shard across machines by
   region. Bookkeep that as a "Phase 6, post-MVP" item.
