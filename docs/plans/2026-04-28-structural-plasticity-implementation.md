# Structural Plasticity (axon pruning) — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or subagent-driven-development) to implement this task-by-task.

**Goal:** Add experience-dependent synapse pruning. During pretraining, synapses with negative survival score AND low weight get permanently pruned (weight=0, plasticity gain=0, contributes nothing to forward dynamics). After pretraining ends, pruning auto-stops via the existing plasticity gate.

**Architecture:** Pragmatic insertion. Three new arrays on `SimulationBridge` + one new method (`update_pruning`) called from `_run_pretraining_phase` only when `--enable-structural-pruning` is set. Forward/plasticity gating via simple multiplicative `alive[i]` mask. Default off — flagship behavior unchanged.

**Reference:** [`2026-04-28-structural-plasticity-design.md`](2026-04-28-structural-plasticity-design.md). Survey: [`2026-04-28-cheat5-real-options-survey.md`](2026-04-28-cheat5-real-options-survey.md).

**Tech stack:** Python 3.12, CuPy (GPU arrays), pytest with `pytest.importorskip("cupy")`. Bridge in `sim/bridge.py`; runner in `research/runners/g11_bg_runner.py`.

---

## Task 1: Add `enable_structural_pruning` config field + bridge array allocation

**Files:**
- Modify: `sim/config.py` (add fields to `CoreSimConfig`)
- Modify: `sim/bridge.py` (allocate new arrays in initialize path)
- Test: `tests/test_structural_pruning.py` (new file)

**Step 1: Write the failing test**

Create `tests/test_structural_pruning.py`:

```python
"""Smoke tests for the structural-plasticity (axon pruning) machinery."""
from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_structural_pruning_arrays_allocated_when_enabled():
    """When enable_structural_pruning is True on a bridge with synapses,
    `cp_synapse_alive` (bool) and `cp_synapse_survival` (float32) arrays
    are allocated and have shape (nnz,)."""
    pytest.importorskip("cupy")
    import cupy as cp
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig

    cfg = CoreSimConfig(num_neurons=20, enable_structural_pruning=True,
                        connection_density_input=0.5, connection_density_recurrent=0.5)
    bridge = SimulationBridge(cfg, VisualizationConfig(), RuntimeState(), GPUConfig())
    bridge.apply_simulation_configuration_core()
    nnz = int(bridge.cp_connections.nnz)
    assert hasattr(bridge, "cp_synapse_alive"), "cp_synapse_alive must be allocated"
    assert hasattr(bridge, "cp_synapse_survival"), "cp_synapse_survival must be allocated"
    assert bridge.cp_synapse_alive.shape == (nnz,)
    assert bridge.cp_synapse_survival.shape == (nnz,)
    assert bridge.cp_synapse_alive.dtype == cp.bool_
    assert bridge.cp_synapse_survival.dtype == cp.float32
    # All synapses start alive and with zero survival score
    assert bool(bridge.cp_synapse_alive.all())
    assert float(bridge.cp_synapse_survival.sum()) == 0.0


def test_structural_pruning_default_off():
    """When the flag is not set, the arrays are not allocated. Flagship is bit-identical."""
    pytest.importorskip("cupy")
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig

    cfg = CoreSimConfig(num_neurons=20)
    bridge = SimulationBridge(cfg, VisualizationConfig(), RuntimeState(), GPUConfig())
    bridge.apply_simulation_configuration_core()
    assert not hasattr(bridge, "cp_synapse_alive") or bridge.cp_synapse_alive is None
    assert not hasattr(bridge, "cp_synapse_survival") or bridge.cp_synapse_survival is None
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_structural_pruning.py -v
```

Expected: FAIL — `enable_structural_pruning` is not a `CoreSimConfig` field; `cp_synapse_alive` is not allocated.

**Step 3: Write minimal implementation**

a) In `sim/config.py`, add to `CoreSimConfig` near other plasticity flags (search for `enable_per_type_stp` or similar):

```python
    # ─── Structural plasticity (2026-04-28) ──────────────────────────
    # Cheat #5 closure attempt #5 (option 1 of the post-v4 plan, see
    # docs/plans/2026-04-28-structural-plasticity-design.md). Adds
    # experience-dependent synapse pruning: synapses with negative
    # survival score AND low weight get permanently eliminated.
    enable_structural_pruning: bool = False
    pruning_alpha: float = 0.001
    pruning_threshold: float = -1.0
    pruning_weight_floor: float = 1.0
```

b) In `sim/bridge.py`, after the existing `cp_plasticity_gain` allocation block (search for `cp_plasticity_gain = cp.ones(nnz`), add:

```python
        if self.cfg.enable_structural_pruning:
            # Per-synapse alive mask + survival score for axon pruning.
            # See docs/plans/2026-04-28-structural-plasticity-design.md.
            self.cp_synapse_alive = cp.ones(nnz, dtype=cp.bool_)
            self.cp_synapse_survival = cp.zeros(nnz, dtype=cp.float32)
        else:
            self.cp_synapse_alive = None
            self.cp_synapse_survival = None
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_structural_pruning.py -v
```

Expected: BOTH pass.

**Step 5: Commit**

```bash
git add tests/test_structural_pruning.py sim/config.py sim/bridge.py
git commit -m "feat(sim): allocate structural-pruning arrays (opt-in)

First task of cheat-5 option-1 (structural plasticity). Adds the
enable_structural_pruning config flag + cp_synapse_alive (bool[nnz])
+ cp_synapse_survival (float32[nnz]) arrays on the bridge. Default
off — flagship bit-identical.

Plan: docs/plans/2026-04-28-structural-plasticity-implementation.md Task 1."
git push origin main
```

---

## Task 2: `bridge.update_pruning` method — survival update + pruning rule

**Files:**
- Modify: `sim/bridge.py`
- Test: `tests/test_structural_pruning.py`

**Step 1: Write the failing test**

Append to `tests/test_structural_pruning.py`:

```python
def test_update_pruning_increments_survival():
    """update_pruning(eligibility, reward) updates survival in place by
    alpha * eligibility * reward. Synapses with positive eligibility
    when reward is positive accumulate positive survival; opposite for
    negative reward."""
    pytest.importorskip("cupy")
    import cupy as cp
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig

    cfg = CoreSimConfig(num_neurons=20, enable_structural_pruning=True,
                        connection_density_input=0.5, connection_density_recurrent=0.5,
                        pruning_alpha=1.0)  # large alpha for crisp test signal
    bridge = SimulationBridge(cfg, VisualizationConfig(), RuntimeState(), GPUConfig())
    bridge.apply_simulation_configuration_core()
    nnz = int(bridge.cp_connections.nnz)
    # Set first half of synapses to eligibility +1, second half to -1
    eligibility = cp.zeros(nnz, dtype=cp.float32)
    eligibility[:nnz // 2] = 1.0
    eligibility[nnz // 2:] = -1.0
    bridge.update_pruning(eligibility, reward_signal=1.0, prunable_indices=None)
    # First half should now have positive survival; second half negative
    surv = bridge.cp_synapse_survival.get()
    assert (surv[:nnz // 2] == 1.0).all()
    assert (surv[nnz // 2:] == -1.0).all()


def test_update_pruning_eliminates_low_survival_low_weight():
    """When survival is below threshold AND weight is below floor, the
    synapse gets pruned: alive=False, weight=0."""
    pytest.importorskip("cupy")
    import cupy as cp
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig

    cfg = CoreSimConfig(
        num_neurons=20, enable_structural_pruning=True,
        connection_density_input=0.5, connection_density_recurrent=0.5,
        pruning_threshold=-0.5, pruning_weight_floor=0.5)
    bridge = SimulationBridge(cfg, VisualizationConfig(), RuntimeState(), GPUConfig())
    bridge.apply_simulation_configuration_core()
    nnz = int(bridge.cp_connections.nnz)
    # Set first quarter to (low survival, low weight) — should prune
    # Set second quarter to (low survival, high weight) — should NOT prune
    # Set third quarter to (high survival, low weight) — should NOT prune
    # Set fourth quarter to (high survival, high weight) — should NOT prune
    bridge.cp_synapse_survival[:nnz // 4] = -1.0
    bridge.cp_synapse_survival[nnz // 4:nnz // 2] = -1.0
    bridge.cp_synapse_survival[nnz // 2:3 * nnz // 4] = 1.0
    bridge.cp_synapse_survival[3 * nnz // 4:] = 1.0
    bridge.cp_connections.data[:nnz // 4] = 0.1
    bridge.cp_connections.data[nnz // 4:nnz // 2] = 1.0
    bridge.cp_connections.data[nnz // 2:3 * nnz // 4] = 0.1
    bridge.cp_connections.data[3 * nnz // 4:] = 1.0
    bridge.update_pruning(
        eligibility_trace=cp.zeros(nnz, dtype=cp.float32),
        reward_signal=0.0, prunable_indices=None)
    alive = bridge.cp_synapse_alive.get()
    assert not alive[:nnz // 4].any(), "first quarter (low surv + low weight) should be pruned"
    assert alive[nnz // 4:nnz // 2].all(), "second quarter (low surv + high weight) should survive"
    assert alive[nnz // 2:3 * nnz // 4].all(), "third quarter (high surv + low weight) should survive"
    assert alive[3 * nnz // 4:].all(), "fourth quarter should survive"
    weights = bridge.cp_connections.data.get()
    assert (weights[:nnz // 4] == 0.0).all(), "pruned synapses must have weight==0"


def test_update_pruning_respects_prunable_indices():
    """When prunable_indices is provided, only those synapses are eligible
    for pruning; others are left alone even if they meet the criteria."""
    pytest.importorskip("cupy")
    import cupy as cp
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig

    cfg = CoreSimConfig(
        num_neurons=20, enable_structural_pruning=True,
        connection_density_input=0.5, connection_density_recurrent=0.5,
        pruning_threshold=-0.5, pruning_weight_floor=0.5)
    bridge = SimulationBridge(cfg, VisualizationConfig(), RuntimeState(), GPUConfig())
    bridge.apply_simulation_configuration_core()
    nnz = int(bridge.cp_connections.nnz)
    # Set all synapses to (low survival, low weight) — would prune everything if unprotected
    bridge.cp_synapse_survival[:] = -1.0
    bridge.cp_connections.data[:] = 0.1
    # But only allow pruning of the first half
    prunable = cp.arange(nnz // 2, dtype=cp.int64)
    bridge.update_pruning(
        eligibility_trace=cp.zeros(nnz, dtype=cp.float32),
        reward_signal=0.0, prunable_indices=prunable)
    alive = bridge.cp_synapse_alive.get()
    assert not alive[:nnz // 2].any(), "first half (in prunable set) pruned"
    assert alive[nnz // 2:].all(), "second half (not in prunable set) protected"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_structural_pruning.py -v
```

Expected: AttributeError — `update_pruning` doesn't exist on bridge.

**Step 3: Write minimal implementation**

In `sim/bridge.py`, near the existing `set_plasticity_gate` method:

```python
    def update_pruning(self, eligibility_trace, reward_signal, prunable_indices=None):
        """Structural-plasticity step. Updates survival scores based on
        reward-aligned eligibility, then prunes synapses meeting both
        criteria: survival < pruning_threshold AND weight < pruning_weight_floor.

        Pruned synapses get alive=False, weight=0. Forward dynamics + plasticity
        respect the alive mask via cp_plasticity_gain[i] *= alive[i] (applied
        here as a side effect, since cp_plasticity_gain is already used for
        plasticity gating).

        prunable_indices: optional cupy int64 array. If provided, only synapses
        in this set are eligible for pruning. Used by the runner to restrict
        pruning to cross-projection synapses only.

        See docs/plans/2026-04-28-structural-plasticity-design.md.
        """
        import cupy as cp
        if self.cp_synapse_alive is None:
            return  # not enabled
        # Update survival score for all synapses
        delta = self.cfg.pruning_alpha * eligibility_trace * float(reward_signal)
        self.cp_synapse_survival += delta.astype(cp.float32)
        # Pruning rule
        weights = self.cp_connections.data
        prune_mask = (
            (self.cp_synapse_survival < self.cfg.pruning_threshold) &
            (weights < self.cfg.pruning_weight_floor) &
            self.cp_synapse_alive
        )
        if prunable_indices is not None:
            # Restrict to the prunable set: zero out mask outside of it
            restricted = cp.zeros_like(prune_mask)
            restricted[prunable_indices] = prune_mask[prunable_indices]
            prune_mask = restricted
        # Apply: alive=False, weight=0, plasticity_gain=0
        self.cp_synapse_alive[prune_mask] = False
        weights[prune_mask] = 0.0
        if self.cp_plasticity_gain is not None:
            self.cp_plasticity_gain[prune_mask] = 0.0
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_structural_pruning.py -v
```

Expected: all 5 tests pass.

**Step 5: Commit**

```bash
git add sim/bridge.py tests/test_structural_pruning.py
git commit -m "feat(sim): bridge.update_pruning — survival update + prune rule

Second task of cheat-5 option-1. Adds the per-step pruning rule:
update survival score by alpha * eligibility * reward, then prune
synapses with (survival < threshold) AND (weight < weight_floor),
optionally restricted to a prunable_indices subset.

Plan: docs/plans/2026-04-28-structural-plasticity-implementation.md Task 2."
git push origin main
```

---

## Task 3: Forward-pass + plasticity-gain gating via alive mask

**Files:**
- Modify: `sim/bridge.py` (forward/plasticity hot path)
- Test: `tests/test_structural_pruning.py`

**Goal:** Once a synapse is marked `alive=False`, it should:
- Contribute zero current in the forward pass.
- Receive zero plasticity updates (STDP, eligibility, reward).

We achieve this via the existing `cp_plasticity_gain` array (which we already zero at prune time in Task 2 — so plasticity is handled). For forward dynamics, the cleanest approach is to multiply `weight[i]` by `alive[i]` once per simulation step (idempotent — if alive=True it's a noop; if alive=False, weight stays 0 since prune set it to 0 already).

**Step 1: Write the failing test**

Append:

```python
def test_pruned_synapse_stays_at_zero_after_simulation_steps():
    """After pruning, even if other forces would push the weight up
    (or down), the alive mask keeps it at zero across many sim steps."""
    pytest.importorskip("cupy")
    import cupy as cp
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig

    cfg = CoreSimConfig(
        num_neurons=20, enable_structural_pruning=True,
        connection_density_input=0.5, connection_density_recurrent=0.5,
        pruning_threshold=-0.5, pruning_weight_floor=0.5,
        enable_stdp=True)
    bridge = SimulationBridge(cfg, VisualizationConfig(), RuntimeState(), GPUConfig())
    bridge.apply_simulation_configuration_core()
    nnz = int(bridge.cp_connections.nnz)
    # Force-prune the first 10 synapses
    bridge.cp_synapse_alive[:10] = False
    bridge.cp_connections.data[:10] = 0.0
    if bridge.cp_plasticity_gain is not None:
        bridge.cp_plasticity_gain[:10] = 0.0
    # Even with high external drive, run several sim steps; pruned weights stay 0.
    for _ in range(20):
        bridge._run_one_simulation_step()
    weights = bridge.cp_connections.data.get()
    assert (weights[:10] == 0.0).all(), "pruned synapse weights diverged from zero"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_structural_pruning.py::test_pruned_synapse_stays_at_zero_after_simulation_steps -v
```

Expected: PASS already (because Task 2 set `cp_plasticity_gain[i] = 0` at prune time, and the existing plasticity machinery multiplies updates by `cp_plasticity_gain`). If it fails, we need explicit `weight *= alive` enforcement.

If it passes immediately, that confirms Task 2's gating via `cp_plasticity_gain` is sufficient — no Task 3 code change needed; the test acts as a regression guard.

**Step 3: Implementation — only if Step 2 fails**

If the test fails, add to the per-step routine in `bridge._run_one_simulation_step` (right after weight updates from plasticity):

```python
if self.cp_synapse_alive is not None:
    self.cp_connections.data *= self.cp_synapse_alive  # zeroes pruned weights
```

This is idempotent and cheap (single cupy multiplication per step).

**Step 4: Re-run to verify pass**

```bash
pytest tests/test_structural_pruning.py -v
```

Expected: all 6 tests pass.

**Step 5: Commit (only if Step 3 was needed)**

```bash
git add sim/bridge.py tests/test_structural_pruning.py
git commit -m "feat(sim): enforce alive-mask in forward weight pass

If Task 2's plasticity-gain gating wasn't sufficient (some other path
was modifying weights and bypassing the gain), add an explicit
weight *= alive per sim step. Belt-and-suspenders correctness for the
pruning invariant."
git push origin main
```

If Task 3 wasn't needed, just commit the test as a regression guard:

```bash
git add tests/test_structural_pruning.py
git commit -m "test(sim): regression guard for pruned-synapse-stays-zero"
git push origin main
```

---

## Task 4: CLI flags on `g11_bg_runner.py`

**Files:**
- Modify: `research/runners/g11_bg_runner.py`
- Test: `tests/test_g11_bg_runner_flags.py`

**Step 1: Write the failing test**

Append to `tests/test_g11_bg_runner_flags.py`:

```python
def test_enable_structural_pruning_kwarg_accepted(tmp_out_path):
    """The runner accepts the new pruning kwargs without TypeError."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import run_moving_goal_episode
    run_moving_goal_episode(
        out_path=tmp_out_path, seed=42, n_steps=20, verbose=False,
        enable_bg_cross_projections=True,
        enable_developmental_pretraining=True,
        enable_structural_pruning=True,
        pretraining_n_goals=0, pretraining_steps_per_goal=0,
    )
```

**Step 2: Run test → expect TypeError on unknown kwarg.**

**Step 3: Implementation**

a) Add to `run_moving_goal_episode` signature (near `enable_developmental_pretraining`):

```python
    enable_structural_pruning: bool = False,
```

b) Map onto `cfg`:

```python
    cfg.enable_structural_pruning = enable_structural_pruning
```

(Place this with the other `cfg.X = ...` assignments in the function setup.)

c) Add argparse:

```python
    ap.add_argument("--enable-structural-pruning", action="store_true",
                    help="Cheat-5 option 1: experience-dependent synapse pruning during "
                         "pretraining. Synapses with negative survival score AND low weight "
                         "get permanently eliminated. See "
                         "docs/plans/2026-04-28-structural-plasticity-design.md.")
```

d) Pass-through in `main()`:

```python
            enable_structural_pruning=args.enable_structural_pruning,
```

**Step 4: Run test → PASS.**

**Step 5: Commit:**

```
feat(g11): wire --enable-structural-pruning flag + kwarg
```

---

## Task 5: Hook pruning into `_run_pretraining_phase`

**Files:**
- Modify: `research/runners/g11_bg_runner.py:_run_pretraining_phase`
- Test: `tests/test_g11_bg_runner_flags.py`

**Goal:** during pretraining, after the existing per-trial reward+plasticity update, call `bridge.update_pruning(eligibility_trace, reward, prunable_indices=cross_projection_indices)`. Only cross-projection synapses are prunable.

**Step 1: Write the failing test**

```python
def test_pretraining_with_pruning_smoke(tmp_out_path):
    """End-to-end: tiny pretraining with --enable-structural-pruning. Some
    cross-projection synapses get pruned (alive=False) by the end."""
    pytest.importorskip("cupy")
    import cupy as cp
    from research.runners.g11_bg_runner import run_moving_goal_episode
    import research.runners.g11_bg_runner as runner_mod

    snapshots = {}
    original = runner_mod._run_pretraining_phase

    def wrapped(*args, **kwargs):
        bridge = kwargs.get("bridge", args[0] if args else None)
        result = original(*args, **kwargs)
        if bridge.cp_synapse_alive is not None:
            cross = bridge._plasticity_gate_to_synapses.get("bg_cross_projections")
            if cross:
                idx = cp.asarray(list(cross), dtype=cp.int64)
                snapshots["cross_alive_count"] = int(bridge.cp_synapse_alive[idx].sum())
                snapshots["cross_total"] = int(idx.size)
        return result

    runner_mod._run_pretraining_phase = wrapped
    try:
        run_moving_goal_episode(
            out_path=tmp_out_path, seed=42, n_steps=50, verbose=False,
            enable_bg_cross_projections=True,
            cross_projection_weight=0.0,
            enable_bg_lateral_inhibition=True,
            enable_curriculum=True, curriculum_warmup_steps=10,
            enable_developmental_pretraining=True,
            enable_structural_pruning=True,
            pretraining_n_goals=1, pretraining_steps_per_goal=200,
        )
    finally:
        runner_mod._run_pretraining_phase = original

    # Pruning should kill SOME but not ALL cross-projection synapses.
    # Use loose bounds for the smoke test (signal can vary by seed).
    cross_alive = snapshots["cross_alive_count"]
    cross_total = snapshots["cross_total"]
    assert cross_total > 0, "test config should produce cross-projection synapses"
    assert cross_alive < cross_total, "pruning should eliminate at least 1 synapse"
    assert cross_alive > 0, "pruning should NOT eliminate everything"
```

**Step 2: Run test → expect failure (no pruning yet, all synapses alive).**

**Step 3: Implementation**

In `_run_pretraining_phase`, find the per-trial reward update block. Add after the reward signal is set and reward_hold_steps run, BEFORE the next trial starts:

```python
            # Structural pruning (cheat-5 option-1, 2026-04-28). Only fires
            # during pretraining when enable_structural_pruning is on. Restricted
            # to cross-projection synapses so we don't sparsify the same-action
            # cortex_to_d1 routing.
            if cfg.enable_structural_pruning and bridge.cp_synapse_alive is not None:
                cross_idx = bridge._plasticity_gate_to_synapses.get("bg_cross_projections")
                if cross_idx:
                    bridge.update_pruning(
                        eligibility_trace=bridge.cp_eligibility_trace,
                        reward_signal=delivered_reward,
                        prunable_indices=cp.asarray(list(cross_idx), dtype=cp.int64),
                    )
```

(`cp` is already imported in `_run_pretraining_phase` per Task 5 of v4.)

**Step 4: Run test → PASS.**

**Step 5: Commit:**

```
feat(g11): hook structural pruning into pretraining trial loop
```

---

## Task 6: Full sweep + regression

**Files:** none (verification step)

```bash
pytest tests/test_structural_pruning.py tests/test_g11_bg_runner_flags.py -v
pytest tests/test_webapp_server.py tests/test_regions.py tests/test_neuromodulators.py -v 2>&1 | tail
```

Expected: all green. ~32+ tests in g11_bg_runner_flags + 6 in structural_pruning + 96 webapp/regions/neuromodulators. No regressions.

---

## Task 7: Tier 1 wiring smoke (manual GPU run)

```bash
python -m research.runners.g11_bg_runner --moving-goal \
    --hippocampus --learned-perception --pfc \
    --beacon-perception --beacon-replaces-goal \
    --cue-reflex --cue-reflex-replaces-heuristic \
    --landmarks --landmarks-replace-place \
    --sensed-reward \
    --bg-lateral-inhibition --bg-cross-projections --cross-projection-weight 0.0 \
    --adaptive-da --adaptive-da-ema-decay-negative 0.7 \
    --curriculum --curriculum-warmup-steps 600 \
    --developmental-pretraining --pretraining-n-goals 1 --pretraining-steps-per-goal 1000 \
    --enable-structural-pruning \
    --seed 42 --n-steps 1800 \
    --out research/findings/raw/g11_bg/g11_seed42_v5tier1.json
```

Pass criteria:
- rc=0
- "pretraining complete" line present
- no NaN in cross weights summary
- Some cross-projection synapses pruned (verify by adding a print in pretraining helper that reports alive-count at end, OR by inspecting the output JSON if we extend it)

---

## Task 8: Tier 2 signal check (3 seeds, ~4h batch)

Same flags, seeds 42/43/44, `--pretraining-n-goals 5 --pretraining-steps-per-goal 1000`.

Decision matrix unchanged from v4: ≤4.5 → Tier 3; 4.5-6 → review; >6 → NO-GO option 1, pivot to option 2.

---

## Task 9: Tier 3 6-seed validation (only if Tier 2 promising)

`--pretraining-n-goals 10 --pretraining-steps-per-goal 3000`, 6 seeds, overnight.

---

## Task 10: Findings doc + propagation

Same template as v4: `research/findings/2026-04-28-cheat5-option1-results.md` + propagation to CLAUDE.md / SCIENCE_ROADMAP / INDEX / CHANGELOG / memory.
