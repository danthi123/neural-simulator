# Perf option — Multi-seed batching in one process

**Status:** design only. NOT yet planned for implementation. Needs your review before committing to the refactor.

**Context:** Throughput investigation hit a wall at 4-6 concurrent processes (aggregate ~5-6 step/s). Adding more processes barely helps because CUDA context-switching between them dominates. The biggest unexplored lever is to **eliminate context-switching entirely** by running multiple seeds inside a single CuPy process, vectorising over a leading seed dimension.

This is the only known path to 3-5× aggregate throughput on this hardware. It's also the largest refactor in the codebase. This doc lays out the scope so you can decide whether it's worth the time.

## Goal

Run `n_seeds` independent agents in lockstep within one CuPy process. Each per-step kernel processes `(n_seeds, ...)` instead of `(...,)`. The single CuPy context drives the GPU continuously without context-switching, eliminating the 4-6-concurrent ceiling.

Concretely: a 6-seed validation batch that currently takes ~14h at 4-concurrent (the actual measured Tier 3 budget) might run in ~4-5h at 6-batched.

## What gets vectorised

Almost every per-synapse and per-neuron array on `SimulationBridge` grows from shape `(N,)` or `(nnz,)` to `(n_seeds, N)` or `(n_seeds, nnz)`:

- `cp_membrane_potential_v` (N,) → (n_seeds, N)
- `cp_firing_states` (N,) → (n_seeds, N)
- `cp_external_input_current` (N,) → (n_seeds, N)
- All Izhikevich/HH state arrays (recovery, m, h, n, etc.)
- `cp_synapse_weights` / `cp_connections.data` (nnz,) → (n_seeds, nnz)
- `cp_eligibility_trace` (nnz,) → (n_seeds, nnz)
- `cp_plasticity_gain` (nnz,) → (n_seeds, nnz)
- `cp_synapse_alive` / `cp_synapse_survival` (the new pruning arrays)
- All neuromodulator concentration arrays (per-NM scalars become per-seed)
- All STP state (Tsodyks-Markram)
- All STDP traces (pre/post spike traces)

What stays unchanged:
- Connection topology (`cp_connections.indices`, `cp_connections.indptr`) — same sparse pattern across all seeds (we vary weights, not topology)
- Region/pathway metadata
- Plasticity-gate dictionaries (per-name gate value still scalar; applied uniformly across seeds)

## What changes per kernel

Every `@cp.fuse()` kernel needs review. Three categories:

1. **Trivially vectorisable** (most kernels): elementwise math (e.g. `fused_izhikevich2007_dynamics_update`). CuPy broadcasts naturally — just need the input arrays to have a leading seed dim. Probably no code change.

2. **Reductions over neurons** (e.g. firing rate calc, motor counts): need to collapse the neuron axis but keep the seed axis. Currently `firing.sum()`; would become `firing.sum(axis=-1)`.

3. **Sparse-matrix synaptic current** (`fused_conductance_decay_and_current`): the heavy hitter. Currently uses CSR sparse matmul with weights as a (nnz,) vector. Need to either:
   - Use CuPy's batched sparse-dense matmul (`cupyx.scipy.sparse.csr_matrix @ dense` with broadcasting) — may or may not exist for our use case
   - Custom CUDA kernel with explicit seed dim
   - Loop over seeds (defeats the purpose)

This third point is the technical risk. If batched sparse matmul is awkward in CuPy, the refactor gets much harder.

## What changes per runner

Each runner currently does: build one `SimulationBridge`, run the eval loop. Multi-seed batching means: build one bridge with `n_seeds`-replicated state, run a vectorised eval loop where action selection, reward, agent position update etc. all become `(n_seeds,)` operations.

`run_moving_goal_episode` would need to:
- Take `seeds: list[int]` instead of `seed: int` (or have a wrapper that batches a list of seeds).
- Keep agent position per seed: `x = np.array([1] * n_seeds, dtype=int)`, `y = np.array([1] * n_seeds, ...)`.
- Compute reward per seed (Manhattan delta is per-seed scalar).
- Action selection (argmax of motor counts) becomes per-seed argmax.
- `bridge.core_config.current_reward_signal` becomes a `(n_seeds,)` cupy array.

Per-trial Python overhead drops because what was `n_seeds` separate trial bodies becomes one vectorised body. Likely the biggest single win after the GPU side.

## What stays single-process

- The webapp + run launcher: still spawns one `python -m research.runners.g11_bg_runner` per "batch", just each batch now wraps multiple seeds.
- Output JSON: each seed's results go to its own file — the wrapper writes `n_seeds` separate output JSONs after the run.
- Existing single-seed runners: keep the `seed: int` API for backward compatibility; multi-seed is opt-in via `seeds: list[int]`.

## Scope estimate

| Change | Estimate |
|---|---|
| `sim/bridge.py`: add seed dim to all per-neuron and per-synapse arrays | 1-2 days |
| `sim/kernels.py`: review every fused kernel; verify broadcasting works; handle reductions | 1 day |
| Sparse matmul (the risk): figure out batched sparse for synaptic current | 4h - 2 days (depends on CuPy support) |
| `research/runners/g11_bg_runner.py`: vectorise the trial loop, agent state, action selection | 1 day |
| Tests: update existing tests OR add `n_seeds=1` test mode that's equivalent to single-seed | 0.5-1 day |
| Validation: A/B compare 1-seed-batched vs 1-seed-original to verify behaviour identical | 0.5 day |
| Profiling: measure actual speedup at n_seeds=1, 4, 8 | 0.5 day |

**Total: 4-7 days of focused work.** Conservative estimate is 7 days; aggressive 4. Plus the risk of the sparse-matmul step blowing up.

## Risk profile

- **Behaviour equivalence**: at `n_seeds=1`, the batched code path MUST produce bit-identical (or float-tolerable) results to the original. Comprehensive test required. Without this, every existing finding is suspect.
- **Sparse matmul performance**: if batched sparse-dense matmul in CuPy doesn't broadcast well over a seed dim, we'd need a custom CUDA kernel. Could turn the 4-7 day estimate into 7-14.
- **Memory blow-up**: arrays grow `n_seeds`× in size. At `n_seeds=6`, our 943-neuron / 86909-synapse network goes from ~50 MB to ~300 MB GPU. Still fine on 24 GB. At larger networks (e.g. scaling to 16x16 grid → ~3500 neurons), 6× scaling would cost ~1 GB — still fine.
- **Existing test suite breakage**: 96+ tests across regions, neuromodulators, structural pruning, webapp. All assume single-seed bridges. Either need a compatibility shim (n_seeds=1 path = current behaviour) or test updates.
- **Cross-cutting refactor**: small mistakes in one file (e.g. forgetting to add the seed dim somewhere) cause hard-to-debug correctness bugs. Probably the dominant risk.

## Why this is worth it (eventually)

- **Tier 3 validation goes from overnight to a couple hours.** Instead of running 6 seeds × ~14h, run 1 batch of 6 in ~4-5h.
- **Iteration speed during research.** The expensive part of cheat-5 work is the GPU validation; halving that wall-clock means we can try 2-3× more variants per day.
- **Larger experiments become tractable.** Want to do a 12-seed bootstrap to compute confidence intervals? Currently impossible at 14h × 2 batches = 28h. Batched: ~5h.

## When to do it

NOT now. Reasonable triggers:

- Cheat-5 closure (real or by-design) is settled and we're confident the architecture is stable for at least a few weeks.
- We're about to commit to a multi-week scaling experiment (e.g. 16×16 grid, 32×32 grid, larger BG circuits) where iteration speed compounds.
- Hardware changes — different GPU may shift the math.

If we kept hitting Tier 3-scale validations at 14h each, the refactor pays back within ~2 weeks of typical research pace.

## Alternative path (cheaper, smaller win)

Instead of multi-seed batching in CuPy, **use CUDA streams** to parallelize multiple bridges within one process. Each bridge gets its own stream; CuPy will overlap them on the GPU without explicit batching. Smaller win (probably 1.5-2× rather than 3-5×) but ~1-2 days of work instead of 4-7. Worth exploring as a "if seed-batching is too much" middle ground.

## Decision

I'd recommend NOT doing the seed-batching refactor right now. The structural-plasticity (cheat-5 option-1) experiment is in flight; the v4 closure-by-design is fresh; we don't know yet whether option 1 or 2 will succeed. The right time is after the cheat-5 chapter closes for good. Keep the design doc on hand; revisit when we're committing to a multi-week scaling phase.

If you want a smaller perf win sooner, **profile (#1) and option B (#2) are the right next steps** — they're both ~1 day of work each at most, and they tell us where we actually stand vs where we're guessing.

## Files

- This design: `docs/plans/2026-04-28-perf-seed-batching-design.md` (this file)
- Implementation plan: NOT YET WRITTEN. Should be drafted via `superpowers:writing-plans` if/when we decide to go ahead.
- Implementation files (when ready): `sim/bridge.py`, `sim/kernels.py`, `research/runners/g11_bg_runner.py`, plus a new `tests/test_seed_batching.py`.
