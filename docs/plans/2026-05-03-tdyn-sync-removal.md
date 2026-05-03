# t_dyn GPU-CPU sync removal — proposed micro-opt

**Status:** designed, NOT MERGED. Defer until GPU is free for verification.
**Estimated speedup:** 3-5% on overall step time (~10-25% on t_dyn alone).
**Risk:** off-by-one in refractory timer semantics if not careful.

---

## What

In `_run_one_simulation_step()` around line 4463 of `sim/bridge.py`:

```python
# Current code — has GPU->CPU sync at fired_indices.size check
not_in_refractory = (self.cp_refractory_timers <= 0)
current_spike_thresholds = ...
fired_this_step = (v_new >= current_spike_thresholds) & not_in_refractory
fired_indices = cp.where(fired_this_step)[0]

if fired_indices.size > 0:                              # <-- GPU->CPU sync
    v_new[fired_indices] = self.cp_izh_c_reset[fired_indices]
    u_new[fired_indices] += self.cp_izh_d_increment[fired_indices]
    self.cp_refractory_timers[fired_indices] = cfg.refractory_period_steps

self.cp_membrane_potential_v[:] = v_new
self.cp_recovery_variable_u[:] = u_new
self.cp_refractory_timers[self.cp_refractory_timers > 0] -= 1
```

The `fired_indices.size > 0` check forces CuPy to materialize the
size of `fired_indices` on host before the Python conditional can
proceed. This is a synchronous GPU->CPU transfer per sub-step.

## Proposed replacement

```python
# Replacement — pure GPU operations, no sync
not_in_refractory = (self.cp_refractory_timers <= 0)
current_spike_thresholds = ...
fired_this_step = (v_new >= current_spike_thresholds) & not_in_refractory

# Apply fire-resets via masked-where
v_new = cp.where(fired_this_step, self.cp_izh_c_reset, v_new)
u_new = u_new + cp.where(fired_this_step, self.cp_izh_d_increment,
                         cp.float32(0.0))

# Refractory: combine set+decrement into single operation
# Fired neurons: set to refractory_period_steps - 1 (matching original
#   off-by-one: original sets to N then decrements to N-1)
# Non-fired with timer > 0: decrement
# Non-fired with timer = 0: stays 0
refractory_after = cp.where(
    fired_this_step,
    cp.int32(max(0, cfg.refractory_period_steps - 1)),
    cp.maximum(self.cp_refractory_timers - 1, cp.int32(0))
)
self.cp_membrane_potential_v[:] = v_new
self.cp_recovery_variable_u[:] = u_new
self.cp_refractory_timers[:] = refractory_after
```

## Why it's faster

1. **Eliminates GPU->CPU sync** at `fired_indices.size > 0`. Saves
   ~50-100us per sub-step.
2. **Replaces fancy-index ops with element-wise ops.** Fancy indexing
   has higher launch overhead than element-wise; cp.where is one
   kernel.
3. **Combines `set refractory` + `decrement-if-positive` into one
   op.** Saves one kernel launch.

Expected wall-clock savings: 100-200us per sub-step on the v2
architecture. At 290 sub-steps/sec baseline = ~3.4ms/step, that's
~3-5% speedup.

## Correctness verification needed

1. **Off-by-one check.** Original sets `refractory[fired] = N` then
   decrements to N-1. Replacement sets directly to N-1. Should be
   equivalent.
2. **Numerical equivalence over many steps.** Run identical seeds
   before and after, compare `cp_membrane_potential_v`,
   `cp_recovery_variable_u`, `cp_refractory_timers` at each of the
   first 100 steps. Should match to within numerical precision.
3. **Spike count over a longer run.** Run 5000 steps with both
   versions, compare total spike count per neuron. Should match.

## TDD test (proposed)

```python
# tests/test_bridge_dynamics_equivalence.py
def test_tdyn_micro_opt_numerically_equivalent():
    """Replacing fancy-index spike-reset with cp.where masking
    should produce identical (v, u, refractory) trajectory."""
    from sim.bridge import SimulationBridge
    # Build identical bridges from same seed
    bridge_old = build_test_bridge(seed=42, mode='legacy')
    bridge_new = build_test_bridge(seed=42, mode='cp_where')

    # Step 100 times, compare state
    for step in range(100):
        bridge_old._run_one_simulation_step()
        bridge_new._run_one_simulation_step()

    import cupy as cp
    assert cp.allclose(bridge_old.cp_membrane_potential_v,
                       bridge_new.cp_membrane_potential_v, atol=1e-5)
    assert cp.allclose(bridge_old.cp_recovery_variable_u,
                       bridge_new.cp_recovery_variable_u, atol=1e-5)
    assert cp.array_equal(bridge_old.cp_refractory_timers,
                          bridge_new.cp_refractory_timers)
```

Adding the legacy/new mode flag is itself a small refactor; might be
cleaner to use a feature flag in `cfg` like `cfg.fast_spike_reset`.

## When to do this

- After all current GPU runs finish (tonight's auto-followup)
- Verify on a quick 1-seed v2 baseline run that aligned ratio doesn't
  change (should be 0/1, same as current)
- Then merge

## Why it's not a top priority

- 3-5% speedup vs 2x speedup from dt=1.0 (already validated)
- Risk of subtle numerical change that affects long-term plasticity
- HH model has its own refractory logic that's untouched, so this
  only affects Izh runs

Better path: get tomorrow's bigger experiments running, then come
back to this when the science direction is clearer.
