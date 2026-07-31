---
type: plan
status: live
date: 2026-04-24
---

# Brain-Region Framework Implementation Plan (Session E.2)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this task-by-task. TDD discipline.

**Goal:** First-class framework where multiple brain regions (PFC, BG, hippocampus, etc.) coexist as configured submodules on a common substrate. Each region has its own local connectivity, neuron parameters, plasticity rules, and modulator outputs. Cross-region pathways are declared rather than wired bespoke. Builds *on top of* the neuromodulator subsystem (E.1) without changing it.

**Architecture:** A `BrainRegion` dataclass declares a population (slice of neuron indices, neuron model overrides, internal connectivity rules, plasticity gating, modulator production rules). A `RegionPathway` dataclass declares projections between regions (pre region, post region, density, weight, plasticity flag, neuromodulator gating). A `RegionManager` instantiates the wiring during bridge initialization, registers neuron groups with the experiment engine + neuromodulator manager, and exposes per-region state for runners. Default OFF: when no regions are declared, the bridge runs as a single population (today's behavior).

**Tech Stack:** CuPy, Python dataclasses, pytest, sim/bridge.py + sim/regions.py (new). Composes with sim/neuromodulators.py from E.1.

**Scope (MVP):**
- Framework: 1 new module (`sim/regions.py`), config additions (`brain_regions`, `region_pathways`, `enable_brain_region_framework`), bridge integration to inject wiring, manager owning region indices.
- 3 declarative concepts: `BrainRegion`, `RegionPathway`, `RegionManager`.
- 2 working regions configured for validation: **PFC** (small recurrent excitatory pool with persistent activity) + **Motor** (downstream output region receiving PFC + sensory).
- 1 cross-region pathway: PFC → Motor with neuromodulator-gated plasticity.
- Backward-compat: when `brain_regions` is empty, bridge behavior unchanged (today's path runs).

**Out of scope:**
- Full PFC working memory tuning (validation probe is whether the framework runs end-to-end, not whether PFC fixes silent-motor trap)
- Hippocampus / striatum / amygdala (those are *future configs*, this is the framework)
- Region-specific neuron MODELS (all regions use Izhikevich or whatever the global model is); per-region parameter overrides are scope for E.2.5
- HDF5 checkpoint integration (deferred to E.2.5)

---

## Task 1: `sim/regions.py` skeleton with dataclasses

**Files:**
- Create: `sim/regions.py`
- Test: `tests/test_regions.py`

**Step 1: Failing tests**

```python
def test_brain_region_defaults():
    from sim.regions import BrainRegion
    r = BrainRegion(name="PFC", n_neurons=200)
    assert r.name == "PFC"
    assert r.n_neurons == 200
    assert r.exc_fraction == 0.8
    assert r.internal_density == 0.1
    assert r.exc_weight_mean == 0.3
    assert r.inh_weight_mean == 0.8
    assert r.plastic_internal is False  # reservoir-style by default


def test_region_pathway_defaults():
    from sim.regions import RegionPathway
    p = RegionPathway(from_region="PFC", to_region="Motor")
    assert p.from_region == "PFC"
    assert p.to_region == "Motor"
    assert p.density == 0.5
    assert p.weight_mean == 1.0
    assert p.plastic is True  # cross-region projections default plastic
    assert p.neuromodulator_gates == []
```

**Step 2: Verify fails (no module)**

**Step 3: Implement**

```python
# sim/regions.py
"""Brain-region framework (Session E.2).

A first-class framework for declaring multiple cortical / subcortical
populations that share a single bridge. Each BrainRegion owns a slice
of the neuron-index space; each RegionPathway declares cross-region
projections with optional neuromodulator gating.

Default OFF: when CoreSimConfig.brain_regions is empty, the bridge
runs as a single population (today's behavior unchanged).

See:
- docs/plans/2026-04-24-brain-region-framework.md
- research/findings/2026-04-24-session-e1-neuromodulator-subsystem.md (motivation)
- sim/neuromodulators.py (composes with E.1's modulator subsystem)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BrainRegion:
    """One brain region: a population of neurons with local connectivity.

    name: unique identifier (also registered as a NeuronGroup with the
        experiment engine and neuromodulator manager).
    n_neurons: number of neurons. Allocated as a contiguous slice of the
        global neuron-index space; concatenation order matches the order
        in core_config.brain_regions.
    exc_fraction: fraction excitatory (rest inhibitory). 0.8 matches cortex.
    internal_density: fraction of all-pairs internal connections that exist
        (sparse Erdős–Rényi within the region).
    exc_weight_mean: mean weight of internal excitatory connections.
    inh_weight_mean: mean weight of internal inhibitory connections.
    weight_jitter: relative std of normal noise around the means (0.2 = 20%).
    plastic_internal: whether internal synapses are plastic. False (reservoir
        style) for sensorimotor regions; True for cortical learning regions.
    nm_outputs: List[str] of neuromodulator names this region produces. Used
        by future production rules like 'from_region_activity'. Currently
        informational; integrates with E.1.5 sweep work.
    """
    name: str
    n_neurons: int
    exc_fraction: float = 0.8
    internal_density: float = 0.1
    exc_weight_mean: float = 0.3
    inh_weight_mean: float = 0.8
    weight_jitter: float = 0.2
    plastic_internal: bool = False
    nm_outputs: List[str] = field(default_factory=list)


@dataclass
class RegionPathway:
    """Directed projection from one region to another.

    from_region, to_region: BrainRegion.name strings; must exist in
        core_config.brain_regions.
    density: fraction of pre-post pairs that have a synapse.
    weight_mean: mean weight of pathway synapses.
    weight_jitter: relative std (default 0.2).
    plastic: whether pathway synapses are plastic (subject to STDP +
        reward modulation). Cross-region projections default True so
        learning rules can shape them.
    neuromodulator_gates: List[str] of neuromodulator names that gate
        this pathway's plasticity rate (multiplies the effective
        learning rate). Empty = no gating (uses global plasticity rate).
        Integrates with sim.neuromodulators.compute_plasticity_rate_multiplier.
    """
    from_region: str
    to_region: str
    density: float = 0.5
    weight_mean: float = 1.0
    weight_jitter: float = 0.2
    plastic: bool = True
    neuromodulator_gates: List[str] = field(default_factory=list)
```

**Step 4: Verify passes**

**Step 5: Commit**

```bash
git commit -am "feat(regions): BrainRegion + RegionPathway dataclasses"
```

---

## Task 2: `RegionManager` with index allocation

**Files:** `sim/regions.py`, `tests/test_regions.py`

**Step 1: Failing test**

```python
def test_region_manager_allocates_contiguous_indices():
    from sim.regions import BrainRegion, RegionManager
    regions = [
        BrainRegion(name="PFC", n_neurons=100),
        BrainRegion(name="Motor", n_neurons=20),
    ]
    mgr = RegionManager(regions, [])
    mgr.initialize()
    assert mgr.total_neurons() == 120
    assert mgr.indices("PFC") == list(range(0, 100))
    assert mgr.indices("Motor") == list(range(100, 120))
    with pytest.raises(KeyError):
        mgr.indices("Hippocampus")


def test_region_manager_inhibitory_indices():
    from sim.regions import BrainRegion, RegionManager
    regions = [BrainRegion(name="PFC", n_neurons=100, exc_fraction=0.8)]
    mgr = RegionManager(regions, [])
    mgr.initialize(seed=42)
    inh_indices = mgr.inhibitory_indices("PFC")
    assert len(inh_indices) == 20  # 20% inhibitory
    # All inhibitory indices fall inside the PFC range
    for idx in inh_indices:
        assert 0 <= idx < 100
```

**Step 2: Verify fails**

**Step 3: Implement RegionManager.initialize() with deterministic random selection of inhibitory cells**

**Step 4: Verify passes**

**Step 5: Commit**

```bash
git commit -am "feat(regions): RegionManager allocates index ranges + per-region inh assignment"
```

---

## Task 3: Internal connectivity generation

**Files:** `sim/regions.py`, `tests/test_regions.py`

**Step 1: Failing test**

```python
def test_region_manager_internal_wiring_plan():
    pytest.importorskip("cupy")
    from sim.regions import BrainRegion, RegionManager
    regions = [BrainRegion(name="PFC", n_neurons=50, exc_fraction=0.8,
                            internal_density=0.1)]
    mgr = RegionManager(regions, [])
    mgr.initialize(seed=42)
    plan = mgr.build_wiring_plan()
    # Should have a "PFC_internal" group in the plan
    assert "PFC_internal" in plan
    g = plan["PFC_internal"]
    # Sparse: density 0.1 over 50*49 ordered pairs ≈ 245 connections
    assert 200 < g["count"] < 300
    # All endpoints inside [0, 50)
    for pre, post in zip(g["pre_indices"], g["post_indices"]):
        assert 0 <= pre < 50
        assert 0 <= post < 50
        assert pre != post  # no self-loops
```

**Step 2-5: Implement, verify, commit**

---

## Task 4: Cross-region pathway wiring

**Files:** `sim/regions.py`, `tests/test_regions.py`

**Step 1: Failing test**

```python
def test_region_manager_cross_region_pathway_in_plan():
    from sim.regions import BrainRegion, RegionPathway, RegionManager
    regions = [
        BrainRegion(name="PFC", n_neurons=100),
        BrainRegion(name="Motor", n_neurons=20),
    ]
    pathways = [RegionPathway(from_region="PFC", to_region="Motor",
                                density=0.5, weight_mean=1.0)]
    mgr = RegionManager(regions, pathways)
    mgr.initialize(seed=42)
    plan = mgr.build_wiring_plan()
    assert "pathway_PFC_to_Motor" in plan
    g = plan["pathway_PFC_to_Motor"]
    # Density 0.5 over 100*20 = 1000 ordered pairs
    assert 800 < g["count"] < 1200
    # All pre in PFC range, all post in Motor range
    for pre, post in zip(g["pre_indices"], g["post_indices"]):
        assert 0 <= pre < 100
        assert 100 <= post < 120
```

**Step 2-5: Implement, verify, commit**

---

## Task 5: Bridge integration — config flag + RegionManager allocation

**Files:** `sim/config.py`, `sim/bridge.py`, `tests/test_regions.py`

**Step 1: Failing test**

```python
def test_bridge_allocates_region_manager_when_enabled():
    pytest.importorskip("cupy")
    from sim import SimulationBridge, CoreSimConfig, ...
    from sim.regions import BrainRegion, RegionPathway

    cfg = CoreSimConfig()
    cfg.num_neurons = 0  # will be set by region manager
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion(name="PFC", n_neurons=80),
        BrainRegion(name="Motor", n_neurons=20),
    ]
    cfg.region_pathways = [RegionPathway(from_region="PFC", to_region="Motor")]
    sb = SimulationBridge(...)
    sb._initialize_simulation_data(...)
    assert sb.region_manager is not None
    assert sb.core_config.num_neurons == 100
```

Bridge automatically sets `num_neurons` from the region manager's `total_neurons()` if `brain_regions` is non-empty.

**Step 2-5: Implement, verify, commit**

---

## Task 6: Bridge wiring injection — region pathways → cp_connections

**Files:** `sim/bridge.py`, `tests/test_regions.py`

When the framework is enabled, the bridge calls
`region_manager.build_wiring_plan()` and feeds it to `inject_explicit_wiring`
(which already exists from earlier work). All connectivity comes from the
plan; no Watts-Strogatz / spatial / random fallback runs.

**Step 1-5: Test, implement, verify, commit**

---

## Task 7: NeuromodulatorManager group registration from regions

**Files:** `sim/bridge.py`

After both managers are initialized (regions + neuromodulators), the bridge
calls `nm_mgr.set_group_indices(region_mgr.region_indices_dict())` so that
target scope `group:NAME` resolves to the region's neurons.

Backward compat: if no regions, runners still register groups manually.

**Step 1-5: Test, implement, verify, commit**

---

## Task 8: Cross-region pathway plasticity gating by neuromodulator

**Files:** `sim/bridge.py`, `sim/regions.py`

When a pathway has `neuromodulator_gates=["dopamine"]`, the synapses on
that pathway have their effective `reward_learning_rate` multiplied by
`dopamine_concentration / dopamine_baseline`. This implements
DA-dependent corticostriatal LTP-style learning.

Implementation: pathway-aware mask that scopes the multiplier to those
synapses only (extending plasticity_rate scope beyond `all`).

**Step 1-5: Test, implement, verify, commit**

---

## Task 9-12: Higher-level conveniences

- 9: Save final region-aware results in g9-style runners (or new g11_regions runner)
- 10: Per-region readout helpers (`region_firing_rate(name, window_ms)`)
- 11: Region-aware checkpoint / resume (defer if too big)
- 12: `BrainRegionPresets` factory (PFC, Motor, BasalGanglia, Hippocampus stubs)

---

## Task 13: Validation probe — PFC + Motor on silent-motor-trap

**Files:** `research/run_g11_pfc_motor_probe.py`

Two-region setup: PFC (200 neurons, recurrent excitation tuned for slow
persistent activity) + Motor (4 neurons, downstream of PFC + sensory).
PFC has neuromodulator-gated plasticity from a goal-context input.
Run on the relaxed moving-goal scenario with NE.

If readaptation works under this architecture (≥ 2/3 seeds reach P1 PF > 0.3):
the brain-region framework + neuromodulator subsystem together solve
silent-motor trap.

If not: framework is still validated as infrastructure; scope reset.

---

## Task 14: Final validation, push, CLAUDE.md, wiki-sync

Same as E.1 task 18.

---

## Wrap-up

Merge to main if validation probe shows clear improvement. Otherwise leave
on branch with documented findings.
