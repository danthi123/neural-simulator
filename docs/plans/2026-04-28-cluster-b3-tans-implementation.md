# Cluster B.3 — Cholinergic Interneurons (TANs) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement task-by-task.

**Goal:** Add cholinergic interneurons (TANs / Tonically Active Neurons) that release acetylcholine (ACh) modulating corticostriatal plasticity. Real BG TANs are tonically active (~5 Hz baseline) but **pause briefly on salient events (reward, novel stimuli)**. ACh release at corticostriatal synapses creates "plasticity windows" — synapses only consolidate when ACh is paused (low ACh = plasticity-on; high ACh = plasticity-off). This is the third piece of Cluster B (striatal microcircuit).

**Architecture:** Extends the existing neuromodulator subsystem with a new neuromodulator: `acetylcholine`. New per-synapse "TAN-modulation" gate that scales corticostriatal plasticity by `(1 - acetylcholine_concentration)`. ACh concentration tracks reward salience: pauses (drops below baseline) on absolute reward magnitude, recovers slowly to baseline.

**Why this might help cheat-5:** B.2 broke Phase 0 because FSIs broadcast inhibition before agents commit to actions. TANs orthogonally gate plasticity in time — even if FSIs disrupt cortex-MSN dynamics, plasticity ONLY consolidates during ACh-pause windows (around reward events). So learned weights accumulate in correctly-timed windows, mitigating FSI bootstrap disruption.

**Tech stack:** Python 3.12, CuPy. Builder/runner in `research/runners/g11_bg_runner.py`; neuromodulator framework in `sim/neuromodulators.py`; bridge in `sim/bridge.py`.

**Reference:** Cluster B design at [`docs/plans/2026-04-28-cluster-b-striatal-microcircuit-design.md`](2026-04-28-cluster-b-striatal-microcircuit-design.md). B.1 + B.2 results at [`research/findings/2026-04-28-cluster-b1-d1d2-asymmetry-results.md`](../../research/findings/2026-04-28-cluster-b1-d1d2-asymmetry-results.md), [`research/findings/2026-04-28-cluster-b2-striatal-fsis-results.md`](../../research/findings/2026-04-28-cluster-b2-striatal-fsis-results.md).

---

## Task 1: Add `acetylcholine` neuromodulator with `pause_on_reward` production rule

**Files:**
- Modify: `sim/neuromodulators.py` (extend `ProductionRule` to handle pause-on-event dynamics; verify existing framework supports this)
- Modify: `research/runners/g11_bg_runner.py:run_moving_goal_episode` (add `enable_tans: bool = False` kwarg + register the ACh neuromodulator when on)
- Test: `tests/test_tans.py` (new file)

**Background on existing infrastructure:**
The neuromodulator subsystem (`sim/neuromodulators.py`) already supports declaring neuromodulators with `baseline`, `decay_tau_ms`, and production rules like `from_reward`, `from_error_persistence`, `manual`. ACh's "pause on salient event" is the inverse pattern — instead of ramping UP on reward, ACh ramps DOWN on |reward|. Could be implemented as:
- New production rule type: `pause_on_reward` — adds `-sensitivity * |current_reward_signal|` per step (concentration drops when reward fires).
- Or reuse `from_reward` with negative sensitivity — concentration goes negative when reward is positive. But we want pause from a tonic baseline, not absolute negative.

Approach: add a new production rule `pause_on_reward` that does `concentration += sensitivity * (current_reward_signal_abs - reward_baseline)` then clamps to `[concentration_min, concentration_max]`. With negative sensitivity, |reward| above baseline drives concentration DOWN. With baseline = 1.0 and the natural decay back to baseline, this models the "pause then recover" pattern.

**Step 1: Write failing test (declarative behavior)**

Create `tests/test_tans.py` with tests that:
1. ACh neuromodulator can be declared via `NeuromodulatorConfig` with `pause_on_reward` rule.
2. Bridge with TANs enabled has the ACh in its neuromod manager.
3. Driving reward signal causes ACh concentration to drop below baseline.
4. After reward returns to 0, ACh recovers toward baseline (decay).

Specific test code:

```python
"""Tests for Cluster B.3 — Cholinergic Interneurons (TANs)."""
from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_acetylcholine_neuromod_declared_with_pause_rule():
    """Verifies pause_on_reward production rule exists and ACh declares cleanly."""
    from sim.neuromodulators import (
        NeuromodulatorConfig, ProductionRule, ModulatorTarget, NeuromodulatorManager,
    )
    config = NeuromodulatorConfig(
        name="acetylcholine",
        baseline=1.0,
        decay_tau_ms=500.0,
        concentration_min=0.0,
        concentration_max=2.0,
        targets=[ModulatorTarget(target_type="plasticity_window_gate", scope="all")],
        production_rules=[
            ProductionRule(rule_type="pause_on_reward", sensitivity=-2.0, threshold=0.1),
        ],
    )
    # Should construct without error
    assert config.name == "acetylcholine"
    assert config.production_rules[0].rule_type == "pause_on_reward"


def test_acetylcholine_concentration_drops_on_reward():
    """When reward signal fires, ACh concentration should drop below baseline."""
    pytest.importorskip("cupy")
    # Build a minimal bridge with TANs enabled, drive reward, observe ACh
    # concentration dropping. (Exact bridge construction follows the test
    # idiom in test_neuromodulators.py.)
    from sim.neuromodulators import NeuromodulatorManager, NeuromodulatorConfig, ProductionRule, ModulatorTarget
    
    mgr = NeuromodulatorManager(neuromods=[
        NeuromodulatorConfig(
            name="acetylcholine",
            baseline=1.0,
            decay_tau_ms=500.0,
            concentration_min=0.0,
            concentration_max=2.0,
            targets=[ModulatorTarget(target_type="plasticity_window_gate", scope="all")],
            production_rules=[
                ProductionRule(rule_type="pause_on_reward", sensitivity=-1.0, threshold=0.0),
            ],
        ),
    ])
    # ... (mock bridge with current_reward_signal=1.0; step manager; assert ACh < 1.0)


def test_acetylcholine_recovers_to_baseline():
    """After reward stops, ACh decays back to baseline at decay_tau_ms time scale."""
    # ... similar to above but verify recovery dynamics
```

Adjust based on what's actually implementable cleanly in `sim/neuromodulators.py`.

**Step 2: Run test → expect failure** (`pause_on_reward` rule unknown).

**Step 3: Implementation**

a) **In `sim/neuromodulators.py`**, add `pause_on_reward` to the supported rule types in the manager's step function. The rule should compute:
```python
delta = sensitivity * (abs(current_reward_signal) - threshold)
new_concentration = clamp(current_concentration + delta, min, max)
# Decay toward baseline at decay_tau_ms time scale (existing decay machinery)
```

b) **Add `enable_tans: bool = False` kwarg** to `run_moving_goal_episode` near `enable_striatal_fsis`. When on, register the ACh neuromodulator config with the bridge's neuromod manager.

c) **Define a sensible default ACh config** (perhaps in a new helper `_default_acetylcholine_config()`) so the runner doesn't need to inline the config:
```python
def _default_acetylcholine_config():
    return NeuromodulatorConfig(
        name="acetylcholine",
        baseline=1.0,
        decay_tau_ms=500.0,
        concentration_min=0.0,
        concentration_max=2.0,
        targets=[ModulatorTarget(target_type="plasticity_window_gate", scope="all")],
        production_rules=[
            ProductionRule(rule_type="pause_on_reward", sensitivity=-2.0, threshold=0.0),
        ],
    )
```

**Step 4: Verify pass + regression sweep**

```bash
pytest tests/test_tans.py -v
pytest tests/test_neuromodulators.py -v 2>&1 | tail -10
```

**Step 5: Commit**:
```
feat(sim): add pause_on_reward neuromod rule + ACh default config
```

---

## Task 2: Implement `plasticity_window_gate` target type

**Files:**
- Modify: `sim/neuromodulators.py` (extend `ModulatorTarget` to support `plasticity_window_gate` target type)
- Modify: `sim/bridge.py` (apply the gate in the reward-modulated weight update)
- Test: `tests/test_tans.py`

**Goal:** When ACh concentration is high (= baseline, "no salient event"), corticostriatal plasticity is SUPPRESSED. When ACh pauses (concentration below baseline), plasticity is PERMITTED. This is the inverse of `plasticity_rate` modulation that exists already.

**Mechanism:**
Effective plasticity gain = `1 - (acetylcholine_concentration / baseline)` [clamped to [0, 1]]
- ACh = baseline (1.0) → effective gain = 0 (plasticity blocked)
- ACh = 0 (full pause) → effective gain = 1 (plasticity full)
- ACh = above baseline (over-shoot) → effective gain = 0 (plasticity still blocked; clamp to ≥0)

This applies to ALL synapses by default, but should be SCOPE-able (e.g., apply only to corticostriatal synapses, not all). For B.3 first cut: scope="all" — affects all reward-driven plasticity uniformly. Refinement: scope="cortex_to_d1+bg_cross_projections" — apply only to synapses on those plasticity gates. Defer to follow-up if first cut works.

**Implementation outline:**
- In `NeuromodulatorManager`, add a `compute_plasticity_window_gate_multiplier()` method that returns the current modulation factor (1 - concentration/baseline).
- In `bridge._run_one_simulation_step()` reward-modulated update, multiply `weight_updates` by this factor when TAN subsystem is on.

**Tests:**
- ACh at baseline → plasticity gate ~0 (no weight update)
- ACh paused (e.g. 0.1) → plasticity gate ~0.9 (mostly permitted)
- Weight evolution is faster when ACh is paused than when at baseline

**Step 5: Commit**:
```
feat(sim): plasticity_window_gate target type for TAN/ACh modulation
```

---

## Task 3: CLI flag + kwarg plumbing

**Files:**
- Modify: `research/runners/g11_bg_runner.py`
- Test: `tests/test_g11_bg_runner_flags.py`

Same pattern as B.1 / B.2:

a) Add `enable_tans: bool = False` kwarg to `run_moving_goal_episode`.
b) Conditionally enable neuromod subsystem + register ACh config when flag on.
c) `--enable-tans` argparse flag.
d) Pass-through in `main()`.

Test:
```python
def test_tans_kwarg_accepted(tmp_out_path):
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import run_moving_goal_episode
    run_moving_goal_episode(
        out_path=tmp_out_path, seed=42, n_steps=20, verbose=False,
        enable_tans=True,
    )
```

**Step 5: Commit**:
```
feat(g11): wire --enable-tans CLI + kwarg
```

---

## Task 4: Biology probe — ACh pause + plasticity timing

**Files:**
- Create: `research/probes/tan_ach_probe.py`

**Goal:** Verify that ACh concentration drops on reward and that plasticity is gated accordingly.

**Probe design:**
1. Build a minimal bridge with TANs enabled.
2. Set a synapse with non-zero eligibility.
3. Apply reward in a brief window (10ms), then stop reward, then continue simulating.
4. Sample (a) ACh concentration over time, (b) effective plasticity gate over time, (c) weight change at each step.

Expected:
- ACh concentration drops at reward onset, recovers over decay_tau_ms.
- Weight changes during the ACh-paused window; minimal weight changes outside.

Output: stdout summary + JSON.

**Step 5: Commit**:
```
feat(probe): TAN/ACh plasticity-window timing probe
```

---

## Task 5: Cheat-5 multi-goal re-eval (full Cluster B)

After Tasks 1-4 land:

### 5a — v3 + B.1 + B.2 + B.3 baseline (no cross-projections)

3 seeds, multi-goal. Should be ≤ 7.08 baseline. If still bad (e.g., 9+), B.3 didn't fix B.2's Phase-0 issue.

### 5b — patch-matrix + B.1 + B.2 + B.3 (the cluster signal test)

3 seeds, multi-goal, with cross-projections + sparse topology + full Cluster B. **The decisive test** for whether the cluster strategy closes cheat-5.

Decision:
- Mean ≤ 7.0 + std < 0.5 → **CHEAT #5 CLOSURE SIGNAL**. Tier 3 (6-seed) validation next.
- Mean ≤ 7.5 + std < 0.7 → **CLUSTER B PARTIAL CLOSURE**. Tier 3 to confirm.
- Mean 7.5-9 → **MODEST**. Cluster B helps but not enough. Move to Cluster A.
- Mean > 9 → **NO CLOSURE**. Move to Cluster A; B.2 retune may also be needed.

Variance trajectory continuation expected: 0.62 → ~0.4 if pattern holds.

---

## Task 6: Findings doc + propagation

After Task 5:

- Create `research/findings/2026-04-28-cluster-b3-tans-results.md` (mirror B.1/B.2's templates).
- Create `research/findings/2026-04-28-cluster-b-full-results.md` summarizing the cluster as a whole.
- Update CLAUDE.md, SCIENCE_ROADMAP, INDEX, CHANGELOG, memory.
- If Cluster B closes cheat-5: huge milestone, write a special findings entry.
- If not: assess whether to retune B.2 OR move to Cluster A.

## Done criteria

- [ ] ACh neuromodulator + pause_on_reward rule + plasticity_window_gate target implemented (Tasks 1+2)
- [ ] CLI flag wired (Task 3)
- [ ] Biology probe shows ACh pause + plasticity-gating dynamics (Task 4)
- [ ] v3 + full Cluster B baseline ≤ 7.5 (Task 5a)
- [ ] Cluster B + cross-projections multi-goal numbers reported (Task 5b)
- [ ] Findings doc + propagation (Task 6)
- [ ] Decision on next cluster (Cluster A or retune)
