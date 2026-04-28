# Cluster B.1 — D1/D2 Plasticity Asymmetry Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement task-by-task.

**Goal:** Add per-synapse plasticity-sign asymmetry between D1 and D2 corticostriatal terminations. Currently the reward-driven weight update at `sim/bridge.py:4309` applies the same sign to all synapses; with this change, synapses terminating on `str_D1_*` get sign=+1 (LTP under +DA) and synapses terminating on `str_D2_*` get sign=-1 (LTP under -DA, LTD under +DA — the inverted D2 rule).

**Architecture:** New `cp_d1_d2_sign` array on `SimulationBridge` (shape (nnz,), default +1). At BG region setup time, mark synapses whose `to_region.startswith("str_D2_")` with sign=-1. The existing weight-update line multiplies by this sign. Default off — flagship behavior unchanged when flag is off.

**Tech stack:** Python 3.12, CuPy, pytest with `pytest.importorskip("cupy")`. Bridge in `sim/bridge.py`; runner in `research/runners/g11_bg_runner.py`; tests in `tests/test_d1_d2_asymmetry.py` (new file).

**Reference:** Cluster B design at [`docs/plans/2026-04-28-cluster-b-striatal-microcircuit-design.md`](2026-04-28-cluster-b-striatal-microcircuit-design.md). Strategy at [`docs/plans/2026-04-28-cheat5-real-options-survey.md`](2026-04-28-cheat5-real-options-survey.md). Reframe context at [`research/findings/2026-04-28-cheat5-post-v4-reframe.md`](../../research/findings/2026-04-28-cheat5-post-v4-reframe.md).

---

## Task 1: Config field + bridge array allocation

**Files:**
- Modify: `sim/config.py` (add field to `CoreSimConfig`)
- Modify: `sim/bridge.py` (allocate `cp_d1_d2_sign` array; populate at BG region setup time)
- Test: `tests/test_d1_d2_asymmetry.py` (new file)

**Step 1: Write the failing test**

Create `tests/test_d1_d2_asymmetry.py`:

```python
"""Tests for Cluster B.1 — D1/D2 plasticity asymmetry."""
from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_bg_bridge(enable_d1_d2: bool):
    """Build a small BG bridge using the runner's region builder."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import build_bg_brain_regions
    from sim import (
        SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig,
    )
    from sim.regions import RegionManager

    regions, pathways = build_bg_brain_regions(n_cortex=20)  # small for speed
    cfg = CoreSimConfig(
        num_neurons=1,  # placeholder; region_manager will override
        enable_brain_region_framework=True,
        brain_regions=regions,
        region_pathways=pathways,
        enable_d1_d2_asymmetry=enable_d1_d2,
    )
    bridge = SimulationBridge(cfg, VisualizationConfig(), RuntimeState(), GPUConfig())
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


def test_d1_d2_sign_array_allocated_when_enabled():
    """cp_d1_d2_sign exists with shape (nnz,) and dtype float32 when enable=True."""
    pytest.importorskip("cupy")
    import cupy as cp
    bridge = _build_bg_bridge(enable_d1_d2=True)
    assert hasattr(bridge, "cp_d1_d2_sign") and bridge.cp_d1_d2_sign is not None
    nnz = int(bridge.cp_connections.nnz)
    assert bridge.cp_d1_d2_sign.shape == (nnz,)
    assert bridge.cp_d1_d2_sign.dtype == cp.float32
    bridge.clear_simulation_state_and_gpu_memory()


def test_d1_d2_sign_default_off():
    """When flag off, cp_d1_d2_sign is None — flagship is bit-identical."""
    pytest.importorskip("cupy")
    bridge = _build_bg_bridge(enable_d1_d2=False)
    assert getattr(bridge, "cp_d1_d2_sign", None) is None
    bridge.clear_simulation_state_and_gpu_memory()


def test_d1_targeted_synapses_have_sign_plus_one():
    """Synapses whose post-region is str_D1_* get sign=+1."""
    pytest.importorskip("cupy")
    bridge = _build_bg_bridge(enable_d1_d2=True)
    # Find synapses targeting str_D1_N region. region_manager exposes indices
    # for each named region.
    d1_n_neurons = bridge.region_manager.indices("str_D1_N")
    # cp_connections is CSR; for each synapse, post-neuron is at row index.
    # sparse storage: data[i] is the weight, indices[i] is the column = post,
    # indptr defines row boundaries (= pre-neuron). For each i, the post is
    # cp_connections.indices[i].
    import cupy as cp
    post = bridge.cp_connections.indices
    d1_n_set = cp.asarray(list(d1_n_neurons), dtype=cp.int64)
    # Mask: which synapses post into str_D1_N
    mask = cp.isin(post, d1_n_set)
    if int(mask.sum()) == 0:
        pytest.skip("No synapses target str_D1_N in this build (probably no D1 inputs)")
    signs_at_d1 = bridge.cp_d1_d2_sign[mask]
    assert (signs_at_d1 == 1.0).all(), \
        f"D1-targeted synapses must have sign=+1, got {signs_at_d1}"
    bridge.clear_simulation_state_and_gpu_memory()


def test_d2_targeted_synapses_have_sign_minus_one():
    """Synapses whose post-region is str_D2_* get sign=-1."""
    pytest.importorskip("cupy")
    bridge = _build_bg_bridge(enable_d1_d2=True)
    d2_n_neurons = bridge.region_manager.indices("str_D2_N")
    import cupy as cp
    post = bridge.cp_connections.indices
    d2_n_set = cp.asarray(list(d2_n_neurons), dtype=cp.int64)
    mask = cp.isin(post, d2_n_set)
    if int(mask.sum()) == 0:
        pytest.skip("No synapses target str_D2_N in this build")
    signs_at_d2 = bridge.cp_d1_d2_sign[mask]
    assert (signs_at_d2 == -1.0).all(), \
        f"D2-targeted synapses must have sign=-1, got {signs_at_d2}"
    bridge.clear_simulation_state_and_gpu_memory()


def test_non_d1_d2_synapses_have_sign_plus_one():
    """Synapses NOT targeting D1 or D2 (e.g. cortex→cortex, gpe→gpi) keep sign=+1.
    This ensures the rest of the network is unaffected by the asymmetry."""
    pytest.importorskip("cupy")
    bridge = _build_bg_bridge(enable_d1_d2=True)
    import cupy as cp
    post = bridge.cp_connections.indices
    # Compute the union of all D1+D2 neuron indices
    d1_d2_neurons = []
    for action in ("N", "E", "S", "W"):
        d1_d2_neurons.extend(bridge.region_manager.indices(f"str_D1_{action}"))
        d1_d2_neurons.extend(bridge.region_manager.indices(f"str_D2_{action}"))
    d1_d2_set = cp.asarray(d1_d2_neurons, dtype=cp.int64)
    mask_outside = ~cp.isin(post, d1_d2_set)
    if int(mask_outside.sum()) == 0:
        pytest.skip("No non-D1/D2-targeted synapses in this build")
    signs_outside = bridge.cp_d1_d2_sign[mask_outside]
    assert (signs_outside == 1.0).all(), \
        f"Non-D1/D2 synapses must keep sign=+1, got {signs_outside.unique()}"
    bridge.clear_simulation_state_and_gpu_memory()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_d1_d2_asymmetry.py -v
```

Expected: FAIL — `enable_d1_d2_asymmetry` is not a `CoreSimConfig` field; `cp_d1_d2_sign` is not allocated.

**Step 3: Write minimal implementation**

a) In `sim/config.py`, add to `CoreSimConfig` near other plasticity flags:

```python
    # ─── Cluster B.1 (2026-04-28): D1/D2 plasticity asymmetry ─────────
    # D1 MSNs LTP under +DA / LTD under -DA; D2 MSNs invert both signs.
    # Implements via per-synapse sign multiplier on the reward-modulated
    # weight update. See docs/plans/2026-04-28-cluster-b1-d1d2-asymmetry-
    # implementation.md.
    enable_d1_d2_asymmetry: bool = False
```

b) In `sim/bridge.py`, allocate the array. Find the existing block where `cp_plasticity_gain` is set up under per-synapse arrays. Add right after:

```python
        if self.core_config.enable_d1_d2_asymmetry:
            # Default +1 (D1-style: LTP under +DA). D2-targeting synapses
            # are flipped to -1 below after pathway iteration.
            self.cp_d1_d2_sign = cp.ones(nnz, dtype=cp.float32)
        else:
            self.cp_d1_d2_sign = None
```

c) After `inject_explicit_wiring` (or wherever the BG regions/pathways are processed), iterate through pathways and flip sign for D2-targeted ones. This may live in `inject_explicit_wiring` or a follow-up pass — find the spot where `_plasticity_gate_to_synapses` is populated; add adjacent:

```python
        # Cluster B.1: tag D2-targeting synapses with sign=-1 for
        # plasticity asymmetry. D1-targeting synapses + everything else
        # stay at +1 (default). Only flips when the flag is on.
        if self.cp_d1_d2_sign is not None:
            for pathway in regions_or_pathways_iterable:  # adjust to actual var name
                if pathway.to_region.startswith("str_D2_"):
                    indices = self._pathway_synapse_indices(pathway)  # or similar lookup
                    if indices:
                        idx_gpu = cp.asarray(indices, dtype=cp.int64)
                        self.cp_d1_d2_sign[idx_gpu] = -1.0
```

NOTE: the exact iteration structure depends on how pathways are tracked. Look at how `_plasticity_gate_to_synapses` is populated and mirror that pattern.

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_d1_d2_asymmetry.py -v
```

Expected: all 5 tests pass.

**Step 5: Commit**

```bash
git add sim/config.py sim/bridge.py tests/test_d1_d2_asymmetry.py
git commit -m "feat(sim): allocate cp_d1_d2_sign + tag D2-targeting synapses

First task of Cluster B.1 (D1/D2 plasticity asymmetry). Adds the
enable_d1_d2_asymmetry config flag + cp_d1_d2_sign array (shape
(nnz,) float32, default +1.0). Synapses with to_region starting
'str_D2_' get sign=-1; D1-targeting + all other synapses keep +1.

The plasticity rule will multiply weight updates by this sign in
Task 2. For now, the array is allocated but unused — flagship
behavior unchanged.

Plan: docs/plans/2026-04-28-cluster-b1-d1d2-asymmetry-implementation.md"
git push origin main
```

---

## Task 2: Plasticity rule integration

**Files:**
- Modify: `sim/bridge.py` (the reward-modulated weight update at line ~4309)
- Test: `tests/test_d1_d2_asymmetry.py`

**Step 1: Write the failing test**

Append to `tests/test_d1_d2_asymmetry.py`:

```python
def test_d1_d2_sign_inverts_weight_change_under_reward():
    """With enable_d1_d2_asymmetry on:
       - D1-targeting synapses' weights move in the SAME direction as reward
       - D2-targeting synapses' weights move in the OPPOSITE direction
    With a fixed positive eligibility trace and positive reward, D1 weights
    grow and D2 weights shrink."""
    pytest.importorskip("cupy")
    import cupy as cp
    bridge = _build_bg_bridge(enable_d1_d2=True)
    nnz = int(bridge.cp_connections.nnz)

    # Set uniform positive eligibility on all synapses
    bridge.cp_eligibility_trace[:nnz] = 1.0
    # Save initial weights
    w_before = bridge.cp_connections.data.copy()
    # Apply reward (positive)
    bridge.core_config.current_reward_signal = 1.0
    bridge.core_config.reward_baseline = 0.0
    bridge.core_config.reward_learning_rate = 0.01
    bridge.core_config.enable_reward_modulation = True
    bridge._run_one_simulation_step()
    w_after = bridge.cp_connections.data
    delta = w_after - w_before

    # Find D1- and D2-targeted synapse indices
    post = bridge.cp_connections.indices
    d1_set = cp.asarray(
        [n for action in ("N", "E", "S", "W") for n in bridge.region_manager.indices(f"str_D1_{action}")],
        dtype=cp.int64,
    )
    d2_set = cp.asarray(
        [n for action in ("N", "E", "S", "W") for n in bridge.region_manager.indices(f"str_D2_{action}")],
        dtype=cp.int64,
    )
    d1_mask = cp.isin(post, d1_set)
    d2_mask = cp.isin(post, d2_set)

    if int(d1_mask.sum()) > 0:
        assert (delta[d1_mask] >= 0).all(), \
            f"D1 weights should grow under +reward; saw deltas {delta[d1_mask].min().get():.4f} to {delta[d1_mask].max().get():.4f}"
    if int(d2_mask.sum()) > 0:
        assert (delta[d2_mask] <= 0).all(), \
            f"D2 weights should shrink under +reward; saw deltas {delta[d2_mask].min().get():.4f} to {delta[d2_mask].max().get():.4f}"
    bridge.clear_simulation_state_and_gpu_memory()
```

**Step 2: Run test to verify it fails**

Currently the weight update doesn't multiply by `cp_d1_d2_sign`, so the test will fail (D2 weights move same direction as D1).

**Step 3: Implementation**

In `sim/bridge.py` at the existing weight-update site (around line 4309), modify:

```python
                    weight_updates = effective_reward_lr * reward_prediction_error * self.cp_eligibility_trace[:actual_nnz]
                    # Per-pathway plasticity gain (Stage 1, 2026-04-27): gate
                    # the eligibility-to-weight conversion.
                    if self.cp_plasticity_gain is not None:
                        weight_updates = weight_updates * self.cp_plasticity_gain[:actual_nnz]
                    # Cluster B.1 (2026-04-28): D1/D2 plasticity asymmetry.
                    # D2-targeting synapses move opposite to reward direction;
                    # D1-targeting + everything else move with reward.
                    if self.cp_d1_d2_sign is not None:
                        weight_updates = weight_updates * self.cp_d1_d2_sign[:actual_nnz]
                    self.cp_connections.data += weight_updates
```

**Step 4: Run tests**

```bash
pytest tests/test_d1_d2_asymmetry.py -v
```

Expected: all 6 tests pass. Run regression sweep:

```bash
pytest tests/test_regions.py tests/test_neuromodulators.py tests/test_structural_pruning.py -v 2>&1 | tail -10
```

Expected: 67+ tests pass, no regressions.

**Step 5: Commit**

```bash
git add sim/bridge.py tests/test_d1_d2_asymmetry.py
git commit -m "feat(sim): D1/D2 asymmetric weight updates via cp_d1_d2_sign

Multiplicative sign factor on the reward-modulated weight update.
D2-targeting synapses move opposite to reward direction; D1 + all
other synapses move with reward.

Single line addition adjacent to the existing cp_plasticity_gain
multiplier — same pattern, different semantics.

Plan: docs/plans/2026-04-28-cluster-b1-d1d2-asymmetry-implementation.md Task 2."
git push origin main
```

---

## Task 3: CLI flag + kwarg plumbing

**Files:**
- Modify: `research/runners/g11_bg_runner.py`
- Test: `tests/test_g11_bg_runner_flags.py`

**Step 1: Write the failing test**

Append to `tests/test_g11_bg_runner_flags.py`:

```python
def test_d1_d2_asymmetry_kwarg_accepted(tmp_out_path):
    """The runner accepts enable_d1_d2_asymmetry without TypeError."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import run_moving_goal_episode
    run_moving_goal_episode(
        out_path=tmp_out_path, seed=42, n_steps=20, verbose=False,
        enable_d1_d2_asymmetry=True,
    )
```

**Step 2: Verify failure (TypeError on unknown kwarg).**

**Step 3: Implementation**

a) Add to `run_moving_goal_episode` signature near other plasticity-related kwargs (around line 990):

```python
    enable_d1_d2_asymmetry: bool = False,
```

b) Set on cfg inside the function body (with the other cfg.X = ... assignments):

```python
    cfg.enable_d1_d2_asymmetry = enable_d1_d2_asymmetry
```

c) Add argparse:

```python
    ap.add_argument("--enable-d1-d2-asymmetry", action="store_true",
                    help="Cluster B.1: D1/D2 plasticity asymmetry — D2-targeting "
                         "synapses' weight updates flip sign vs D1. See "
                         "docs/plans/2026-04-28-cluster-b1-d1d2-asymmetry-implementation.md.")
```

d) Pass-through in `main()`:

```python
            enable_d1_d2_asymmetry=args.enable_d1_d2_asymmetry,
```

**Step 4: Verify pass + regression sweep.**

**Step 5: Commit:**

```
feat(g11): wire --enable-d1-d2-asymmetry CLI + kwarg
```

---

## Task 4: Biology probe (standalone validation)

**Files:**
- Create: `research/probes/d1_d2_asymmetry_probe.py`
- (No new tests — this is a manual verification script.)

**Goal:** Verify the asymmetry produces the EXPECTED biological signature: D1 weights grow under positive reward; D2 weights grow under negative reward. Run on a small standalone simulation, no full BG cascade needed.

**Sketch:**

```python
"""Cluster B.1 biology probe — verify D1/D2 plasticity asymmetry.

Builds a minimal cortex → str_D1 + str_D2 setup, drives with constant
input, applies +reward and -reward conditions, and reports the weight
change distributions. The expected signature is:
- Under +reward: D1 weights ↑, D2 weights ↓
- Under -reward: D1 weights ↓, D2 weights ↑

Run: python -m research.probes.d1_d2_asymmetry_probe
"""
# (~80-line standalone script that builds bridge, runs ~50 trials with
# +reward then ~50 with -reward, reports delta distributions for D1 vs
# D2 synapses.)
```

**Step 5: Commit + run probe + capture output:**

```
feat(probe): D1/D2 plasticity asymmetry biology probe
```

The probe output (delta distributions) should be saved as a finding for the cluster validation record.

---

## Task 5: Cheat-5 multi-goal re-eval (manual GPU runs)

After Tasks 1-4 land:

### 5a — v3 + B.1 baseline (no cross-projections)

3 seeds, multi-goal, no cross. Should be ≤ v3 baseline 7.08:

```bash
for SEED in 42 43 44; do
    python -m research.runners.g11_bg_runner --moving-goal --goal-schedule multi \
        --hippocampus --learned-perception --pfc \
        --beacon-perception --beacon-replaces-goal \
        --cue-reflex --cue-reflex-replaces-heuristic \
        --landmarks --landmarks-replace-place \
        --sensed-reward --bg-lateral-inhibition \
        --adaptive-da --adaptive-da-ema-decay-negative 0.7 \
        --curriculum --curriculum-warmup-steps 600 \
        --enable-d1-d2-asymmetry \
        --seed $SEED --n-steps 1800 \
        --out research/findings/raw/g11_bg/g11_seed${SEED}_v3_b1.json
done
```

Decision: ≤7.08 mean → B.1 is non-regressive; proceed to 5b. >7.5 → B.1 introduced a problem; debug before proceeding.

### 5b — patch-matrix + B.1

3 seeds, multi-goal, sparse cross + B.1. Compare against patch-matrix-alone result (8.76 ± 2.54):

```bash
for SEED in 42 43 44; do
    python -m research.runners.g11_bg_runner --moving-goal --goal-schedule multi \
        ... (full flag set with --enable-d1-d2-asymmetry --bg-cross-projections \
             --cross-projection-weight 5.0 --cross-projection-density 0.25)
done
```

Decision matrix (compared against patch-matrix-alone 8.76 ± 2.54):
- Mean ≤ 7.0 + std < 1.0: **first real cheat-5 partial closure signal.** B.1 stabilized cross-projections. Proceed to Cluster B.2 (FSIs).
- Mean ~8 (close to patch-matrix-alone): no improvement. Move to B.2 + B.3 to add the rest of Cluster B.
- Mean > 10: B.1 made things worse with cross-projections. Debug or rethink.

---

## Task 6: Findings doc + propagation

After Tasks 1-5 complete:

- Create `research/findings/2026-04-28-cluster-b1-d1d2-asymmetry-results.md` with biology probe output + cheat-5 multi-goal numbers.
- Update CLAUDE.md, SCIENCE_ROADMAP §4.7, INDEX, CHANGELOG, memory.
- If B.1 alone showed cheat-5 signal: flag in CLAUDE.md as "first real cluster signal" and recommend continuing to B.2.
- If B.1 alone showed no cheat-5 signal: document it; proceed to B.2 + B.3 as planned (Cluster B is a unit).

## Done criteria

- [ ] All 6 unit tests pass (5 from Task 1 + 1 from Task 2)
- [ ] CLI flag accepted (Task 3)
- [ ] Biology probe output shows D1↑/D2↓ under +reward, D1↓/D2↑ under -reward
- [ ] v3 + B.1 multi-goal baseline ≤ 7.08 (no regression)
- [ ] Patch-matrix + B.1 multi-goal result reported with decision matrix outcome
- [ ] Findings doc + CLAUDE.md / SCIENCE_ROADMAP / INDEX / CHANGELOG / memory propagated
