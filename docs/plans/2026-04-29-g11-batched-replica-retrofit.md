---
type: plan
status: live
date: 2026-04-29
---

# g11 batched-replica retrofit plan

**Date:** 2026-04-29
**Status:** sim-side groundwork DONE (commit `f1497e0`); runner retrofit pending dedicated session.
**Goal:** Make `g11_bg_runner.py` accept `--n-replicas N` and run N independent agents in a single Python process via the block-diagonal embedding from `sim/replicas.py`. Expected speedup: ~6× wall-clock vs spawning N subprocess (which is currently CPU-bound on the per-step Python orchestration cost).

## Why this is needed

Profile (from `research/findings/2026-04-24-route-b-profile.md`) found that small-network simulations (~264 neurons) are bottlenecked by CuPy kernel-launch overhead: ~50 kernels/step × ~40-80μs/kernel = 2 ms/step Python dispatch cost.

Current g11 runs are larger (~1500+ neurons with cluster A+E, ~2200 with cluster F added) but the bottleneck still applies once we run N processes in parallel. CPU saturates at ~12-18 processes well before GPU memory does. Empirically: tonight's 18-process Cluster F eval runs at 0.31 steps/sec/process, vs ~1.0 steps/sec/process at n=12.

A batched-replica runner would dispatch ONE set of CuPy kernels per step that processes all N replicas simultaneously (single sparse matmul on the block-diagonal CSR, single fused neuron-dynamics kernel, etc.). Per-replica wall-clock is thus dominated by GPU compute (which scales sublinearly with N) rather than Python dispatch (which scales linearly).

## Pre-work (DONE 2026-04-29 in commit f1497e0)

- `sim/replicas.py` — `ReplicaConfig`, `ReplicaManager`, `replicate_wiring_plan`, `replicate_wiring_plan_with_seeds`. 234 lines.
- `sim/neuromodulators.py` — `set_replica_indices()`, `scope='replica:N'` targeting in `compute_excitability_drive_per_neuron`. +37 lines.
- `tests/test_replicas.py` — 13 tests including a 3-replica end-to-end smoke (G9 reservoir, 50 steps in one bridge process).

The bridge's `inject_explicit_wiring` accepts replicated wiring plans without any modification. The 3-replica smoke test passes immediately on current main with all the recent cluster A/B/D/E/F machinery in place. **The sim-side foundation is operational.**

## Scope of the runner retrofit

`g11_bg_runner.py:run_moving_goal_episode` is the target function. It currently runs ONE agent through one trajectory; we need to make it run N agents through N parallel trajectories.

### Per-step state that must become per-replica (lists/arrays of length N)

Currently single-valued; needs to be per-replica:

1. Position: `x`, `y` → `xs[r]`, `ys[r]`
2. Goal: `gx`, `gy` → `gxs[r]`, `gys[r]` (could be shared if all replicas use same schedule)
3. Logs: `trajectory`, `goal_log`, `action_log`, `reward_log`, `distance_log`, `motor_counts` → list of N lists
4. Phase stats: `phase_stats` → list of N
5. Per-replica RNGs: `action_rng`, `rng` → list of N (different seed per replica)
6. Successful trajectories (sleep replay): per-replica list
7. Per-replica heuristic state machinery: any phase-2 cortex gain ramps need per-replica timing if goals differ

### Drive setting (per-step)

The current pattern is:
```python
bridge.cp_external_input_current[region_indices_cp["cortex_N"]] = HEURISTIC_DRIVE
```

After replication, `region_indices_cp["cortex_N"]` covers ALL replicas' cortex_N neurons (block-diagonal: cortex_N occupies indices `[0..n_cortex_per_action), [N+0..N+n_cortex_per_action), ..., [(B-1)*N..(B-1)*N+n_cortex_per_action)`). The drive needs to be:

- **Same drive across replicas** (e.g., baseline currents for gpe, gpi, stn, snc, dopamine, etc.): no change — broadcasting the scalar value over the union of indices works.
- **Different drive per replica** (e.g., heuristic cortex drive depends on `(xs[r], ys[r], gxs[r], gys[r])`): need to compute a per-neuron drive vector.

For the heuristic case, the simplest implementation is:
```python
heuristic_drive_vector = cp.zeros(total_neurons)
for r in range(n_replicas):
    shift = r * neurons_per_replica
    cortex_dir = decide_action_for_replica(xs[r], ys[r], gxs[r], gys[r])
    heuristic_drive_vector[region_indices_cp[f"cortex_{cortex_dir}"][r*group_size:(r+1)*group_size]] = HEURISTIC_DRIVE
bridge.cp_external_input_current[:] = baseline + heuristic_drive_vector
```

Cost: O(N) Python loop overhead per step. Acceptable for N ≤ 32 or so.

### Action selection (per-step)

Current pattern: read `motor_counts[a]` per action, pick argmax + tie-break. After replication, `motor_counts` is per-replica:

```python
for r in range(n_replicas):
    counts = [int(bridge.cp_motor_pool_spike_counts[motor_indices[r][a]].sum().get())
              for a in ACTION_NAMES]
    action_per_replica[r] = pick_action(counts, action_rng[r])
```

Or vectorized via gather + argmax on GPU. The vectorized form is faster but more complex.

### Reward computation (per-step)

Trivially per-replica — just loop:
```python
for r in range(n_replicas):
    reward_per_replica[r] = compute_reward(xs[r], ys[r], gxs[r], gys[r])
```

### Cluster F CF firing trigger

Current: `bridge.cp_external_input_current[region_indices_cp["inferior_olive"]] = 450 if reward < 0`.

After replication: per-replica IO drive vector. Each replica's IO block gets 450 if its reward < 0.

### Output / saving

Each replica produces its own results JSON. The simplest is to save N separate files at the end:
```python
for r in range(n_replicas):
    out_path_r = out_path.replace(".json", f"_replica{r}.json")
    save_replica_results(out_path_r, replica=r, ...)
```

Or one file with per-replica arrays — depends on downstream consumers.

## Task breakdown

Estimated 4-6 hours total for v1.

### T1 — replica wiring + bridge boot (~45 min)

Add to `run_moving_goal_episode`:
- New kwarg `n_replicas: int = 1`. Default 1 → existing single-agent behavior unchanged.
- When `n_replicas > 1`:
  1. Build the regions+pathways via `build_bg_brain_regions(...)` — this is the template.
  2. Build a wiring plan template via `RegionManager(regions, pathways).build_wiring_plan(seed)`.
  3. Construct `ReplicaConfig(replica_id=r, seed_offset=seed*1000+r) for r in range(n_replicas)`.
  4. Call `replicate_wiring_plan_with_seeds(template, replicas, neurons_per_replica)`.
  5. Build the bridge with `cfg.num_neurons = n_replicas * neurons_per_replica`.
  6. Inject the replicated plan via `bridge.inject_explicit_wiring(replicated_plan)`.
  7. Register `bridge.neuromodulator_manager.set_replica_indices(rmgr.replica_indices_dict())`.

Deliverable: bridge can run `bridge._run_one_simulation_step()` for the replicated network without crashes. Smoke test: `python -m research.runners.g11_bg_runner --n-replicas 2 --n-steps 10 --enable-cluster-f-cerebellum --moving-goal` runs to completion.

### T2 — per-replica state lists (~30 min)

Convert `x`, `y`, `gx`, `gy`, `trajectory`, `goal_log`, `action_log`, `reward_log`, `distance_log`, `motor_counts`, `phase_stats`, `successful_trajectories`, `action_rng`, `rng` to lists of N where applicable. Initialize at start. Default for n_replicas=1: behavior matches current single-agent (just lists of length 1).

### T3 — drive setting (~60 min)

Refactor the per-step drive setting block to handle per-replica heuristic drives. Vectorize where possible; loop where simpler. Verify the replicated baseline currents (gpe, gpi, stn, snc, dopamine, dcn_aip, etc.) all set correctly across all blocks.

### T4 — action selection + reward + IO firing (~45 min)

Convert motor-pool readout to per-replica argmax. Reward computation per replica. Cluster F IO firing drive set per-replica when `reward[r] < 0`.

### T5 — output + per-replica result saving (~30 min)

Save N output files OR one file with per-replica arrays. Make the choice consistent with how the eval aggregator (`research/runners/aggregate_*.py`) consumes results.

### T6 — validation (~60 min)

Acid test: `--n-replicas 6 --seed 42` should produce results that, after some-tolerance comparison, match running 6 separate `--seed 42, 43, 44, 100, 101, 102` subprocesses (assuming seed-offsets are 0, 1, 2, 58, 59, 60 to match — TBD seed convention).

Can't expect exact match because the replicated agents have weight-jittered initial conditions per `seed_offset`, while sequential subprocesses use the per-seed full RNG seeding. But the SUMMARY STATISTICS (mean finalQ, std) should be in the same neighborhood. Define a "match" criterion (e.g., per-phase finalQ within ±15% of the n=6 sequential mean).

### T7 — speedup measurement (~15 min)

Baseline: time `--n-replicas 1` for 100 steps.
Replicated: time `--n-replicas 6` for 100 steps (one process).
Compare to: time 6 subprocesses each running `--n-replicas 1` for 100 steps (current state).

Expected: 6-replica-in-one-process ≈ 1× single-replica time (just dispatch overhead amortized); 6 subprocesses ≈ 4-6× single-replica time (CPU contention).

## Risks

1. **Region-name collision under replication**: if two regions in the template have the same name, after replication their indices overlap. Inspect `RegionManager` to confirm uniqueness; if needed, prepend `r{replica}_` to region names.
2. **CF firing in cluster F**: the IO firing trigger sets `cp_external_input_current[region_indices_cp["inferior_olive"]]` — under replication, this is the union of all replicas' IO blocks. We need to set per-replica drives, not a global broadcast.
3. **Cluster D's CA3 recurrent autoassociator**: the per-replica seed-offset weight jitter should give independent dynamics, but we should verify the `internal_density` patterns don't cause cross-replica leakage.
4. **D1/D2 asymmetry kernel**: `cp.cp_d1_d2_sign` is a per-synapse mask. Should auto-replicate via the wiring plan since each replica's str_D2_X synapses are separate. Verify.
5. **Curriculum gates**: `bridge.set_plasticity_gate(name, value)` operates on ALL synapses tagged with `name`, which includes all replicas' synapses. If we want different curriculum schedules per replica, we'd need per-replica gates (gate_name = f"corticostriatal_r{r}"). For v1, all replicas share the same curriculum.

## Acceptance criteria

- 13/13 existing replica tests still pass (no regression).
- Smoke test `--n-replicas 2 --n-steps 60 --enable-cluster-f-cerebellum` runs to completion, returncode 0.
- Acid test: `--n-replicas 6 --seed 42 --n-steps 1800` produces 6 result files; per-phase finalQ summary matches the equivalent 6-subprocess run within ±15%.
- Speedup: 6-replica-in-one-process is at least 3× faster than 6 subprocesses (target: 6× from theory).

## Out of scope for v1

- Per-replica goal_schedule (different task variants per replica)
- Per-replica concentration arrays (E.3 Task 5 — concentrations remain global scalar)
- Multi-cluster combinations beyond the current six (no new cluster work, just retrofit the existing)
- Replication for `_run_pretraining_phase` — pretraining can stay single-replica until motivated by a use case
