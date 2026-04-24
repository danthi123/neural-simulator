# Neuromodulator Subsystem Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Each task has its own failing test, minimal impl, passing test, and commit. Use TDD discipline.

**Goal:** First-class neuromodulator framework where each hormone (dopamine, noradrenaline, etc.) is a declared `NeuromodulatorConfig` with a concentration variable, decay dynamics, production rules, and configurable receptor targets that affect synaptic gain, plasticity rate, and neuronal excitability — replacing the one-off `current_reward_signal` and shelved `cp_synaptic_gain_modulator` hacks with a real subsystem.

**Architecture:** A `NeuromodulatorManager` owned by `SimulationBridge` holds GPU concentration arrays (one scalar or per-region array per modulator), production-rule callbacks, and target-application logic. Each step: (1) update each modulator's concentration via decay + production rules, (2) apply effects to bridge state. Default off (`enable_neuromodulator_subsystem=False`) for full backward compatibility.

**Tech Stack:** CuPy, Python dataclasses, pytest, sim/bridge.py integration.

**Scope (MVP):**
- Framework: 1 new module (`sim/neuromodulators.py`), config additions, bridge integration.
- 2 working modulators built on the framework: **Dopamine** (replicates current reward path) and **Noradrenaline** (new, addresses silent-motor trap).
- 3 target types: synaptic_gain, plasticity_rate, excitability_drive.
- 3 production rules: `dopamine_from_reward`, `noradrenaline_from_error_persistence`, `manual`.
- ACh / 5-HT / others: left as configurable (no built-in production rules, reachable via `manual`).

**Out of scope:** Brain-region framework (Session E.2). Per-neuron heterogeneity (only global scalar this round). Endocrine-level kinetics (no separate slow timescale).

---

## Task 1: Set up `sim/neuromodulators.py` skeleton + dataclasses

**Files:**
- Create: `sim/neuromodulators.py`
- Modify: `sim/__init__.py` (export new symbols)
- Test: `tests/test_neuromodulators.py`

**Step 1: Write failing test for NeuromodulatorConfig dataclass**

```python
# tests/test_neuromodulators.py
def test_neuromodulator_config_defaults():
    from sim.neuromodulators import NeuromodulatorConfig

    nm = NeuromodulatorConfig(name="dopamine")
    assert nm.name == "dopamine"
    assert nm.baseline == 0.0
    assert nm.decay_tau_ms == 500.0
    assert nm.concentration_min == 0.0
    assert nm.concentration_max == 5.0
    assert nm.targets == []
    assert nm.production_rules == []


def test_neuromodulator_config_custom_values():
    from sim.neuromodulators import NeuromodulatorConfig
    nm = NeuromodulatorConfig(
        name="noradrenaline",
        baseline=0.2,
        decay_tau_ms=2000.0,
        concentration_min=0.0,
        concentration_max=2.0,
    )
    assert nm.decay_tau_ms == 2000.0
    assert nm.concentration_max == 2.0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_neuromodulators.py::test_neuromodulator_config_defaults -v`
Expected: FAIL with ImportError on `sim.neuromodulators`

**Step 3: Implement minimal NeuromodulatorConfig + ModulatorTarget + ProductionRule dataclasses**

```python
# sim/neuromodulators.py
"""Neuromodulator subsystem (Session E).

Models hormones / neuromodulators as declared entities with concentration
dynamics and effects on neuronal/synaptic state. See
`docs/plans/2026-04-24-neuromodulator-subsystem.md` and
`research/findings/2026-04-24-session-d-part-a.md` §4 for motivation.

Concentration semantics: scalar per modulator (global broadcast). Each
step the concentration decays toward `baseline` with `decay_tau_ms`,
then production rules add to it. Effects are applied multiplicatively
or additively to bridge state depending on target type.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence


@dataclass
class ModulatorTarget:
    """How a modulator's concentration affects bridge state.

    target_type:
        "synaptic_gain"    : multiplies effective_synaptic_strength.
                             effect = 1.0 + sensitivity * (conc - baseline)
        "plasticity_rate"  : multiplies STDP amplitudes and reward_learning_rate.
                             effect = 1.0 + sensitivity * (conc - baseline)
        "excitability_drive": adds current to membrane drive.
                             effect = sensitivity * (conc - baseline)  [pA]

    scope: which neurons/synapses are affected.
        "all"               : every neuron / synapse.
        "trait:<idx>"       : neurons whose cp_traits == idx (e.g. trait:0 for
                              excitatory, trait:1 for inhibitory).
        "group:<name>"      : neuron group registered with the experiment engine.
        "plastic_only"      : (synaptic_gain & plasticity_rate) only synapses
                              with cp_synapse_plastic_mask == True.

    sensitivity: scaling factor (see effect formulas above). 0 = no effect.
    """
    target_type: str
    scope: str = "all"
    sensitivity: float = 1.0


@dataclass
class ProductionRule:
    """How bridge state drives modulator concentration.

    rule_type:
        "manual"                          : externally set; production = 0
                                            unless API call adds it.
        "from_reward"                     : tracks core_config.current_reward_signal.
                                            On reward != 0, adds magnitude*sensitivity
                                            to concentration.
        "from_error_persistence"          : tracks running mean of reward error.
                                            When |running_mean| > threshold for
                                            longer than `window_ms`, produces.
        "from_novelty"                    : (placeholder for future ACh)

    sensitivity, threshold, window_ms: tunable per rule.
    """
    rule_type: str
    sensitivity: float = 1.0
    threshold: float = 0.5
    window_ms: float = 500.0


@dataclass
class NeuromodulatorConfig:
    """Declarative description of one hormone / neuromodulator."""
    name: str
    baseline: float = 0.0
    decay_tau_ms: float = 500.0
    concentration_min: float = 0.0
    concentration_max: float = 5.0
    targets: List[ModulatorTarget] = field(default_factory=list)
    production_rules: List[ProductionRule] = field(default_factory=list)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_neuromodulators.py -v`
Expected: 2 passed.

**Step 5: Commit**

```bash
git add sim/neuromodulators.py tests/test_neuromodulators.py
git commit -m "feat(neuromodulators): NeuromodulatorConfig + ModulatorTarget + ProductionRule dataclasses"
```

---

## Task 2: Add `NeuromodulatorManager` class with state allocation

**Files:**
- Modify: `sim/neuromodulators.py`
- Test: `tests/test_neuromodulators.py`

**Step 1: Write failing test for state allocation**

```python
def test_manager_allocates_concentration_per_modulator():
    pytest.importorskip("cupy")
    import cupy as cp
    from sim.neuromodulators import (
        NeuromodulatorConfig, NeuromodulatorManager,
    )

    nms = [
        NeuromodulatorConfig(name="dopamine", baseline=0.0),
        NeuromodulatorConfig(name="noradrenaline", baseline=0.2),
    ]
    mgr = NeuromodulatorManager(nms, dt_ms=1.0)
    mgr.initialize(n_neurons=100, cp_module=cp)

    # Concentrations start at baseline
    assert mgr.get_concentration("dopamine") == 0.0
    assert abs(mgr.get_concentration("noradrenaline") - 0.2) < 1e-6
    # Unknown modulator
    with pytest.raises(KeyError):
        mgr.get_concentration("serotonin")
```

**Step 2: Run, verify fails**

Expected: ImportError on `NeuromodulatorManager`.

**Step 3: Implement minimal manager**

```python
class NeuromodulatorManager:
    """Owns the per-modulator concentration state and applies effects each step.

    Initialize after the bridge has cp_module + n_neurons. Call step() once
    per simulation step, after the rest of the synaptic state has been
    updated but before applying effects to the next-step quantities.
    """
    def __init__(self, configs: Sequence[NeuromodulatorConfig], dt_ms: float):
        self._configs = list(configs)
        self.dt_ms = float(dt_ms)
        self._concentrations: dict[str, float] = {}
        self._cp = None
        self._n_neurons = 0
        # Per-rule running state (e.g. reward-error EMA) lives here
        self._rule_state: dict[str, dict] = {}

    def initialize(self, n_neurons: int, cp_module):
        self._cp = cp_module
        self._n_neurons = int(n_neurons)
        self._concentrations = {c.name: float(c.baseline) for c in self._configs}
        self._rule_state = {c.name: {"err_ema": 0.0} for c in self._configs}

    def get_concentration(self, name: str) -> float:
        return self._concentrations[name]
```

**Step 4: Run test, verify passes**

**Step 5: Commit**

```bash
git commit -am "feat(neuromodulators): NeuromodulatorManager state allocation"
```

---

## Task 3: Per-step concentration decay toward baseline

**Files:** `sim/neuromodulators.py`, `tests/test_neuromodulators.py`

**Step 1: Failing test**

```python
def test_concentration_decays_toward_baseline():
    pytest.importorskip("cupy")
    import cupy as cp
    from sim.neuromodulators import NeuromodulatorConfig, NeuromodulatorManager

    nm = NeuromodulatorConfig(name="dopamine", baseline=0.0, decay_tau_ms=100.0)
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)

    # Manually perturb
    mgr.set_concentration("dopamine", 1.0)
    # After 100ms (1 tau), expect ~exp(-1) ≈ 0.368
    for _ in range(100):
        mgr.step(bridge=None)
    assert 0.30 < mgr.get_concentration("dopamine") < 0.45

    # After many tau, should converge to baseline
    for _ in range(1000):
        mgr.step(bridge=None)
    assert abs(mgr.get_concentration("dopamine")) < 0.01
```

**Step 2: Run, verify fails**

**Step 3: Implement decay step**

```python
import math

class NeuromodulatorManager:
    ...
    def set_concentration(self, name: str, value: float) -> None:
        self._concentrations[name] = float(value)

    def step(self, bridge) -> None:
        """One simulation step: decay + production + clipping."""
        for cfg in self._configs:
            conc = self._concentrations[cfg.name]
            # Exponential decay toward baseline
            decay_factor = math.exp(-self.dt_ms / max(cfg.decay_tau_ms, 1e-9))
            conc = cfg.baseline + (conc - cfg.baseline) * decay_factor

            # Production rules will be added in Task 5
            # Clip
            conc = max(cfg.concentration_min, min(cfg.concentration_max, conc))
            self._concentrations[cfg.name] = conc
```

**Step 4: Run, verify passes**

**Step 5: Commit**

```bash
git commit -am "feat(neuromodulators): per-step exponential decay toward baseline"
```

---

## Task 4: Implement `from_reward` production rule

**Files:** `sim/neuromodulators.py`, `tests/test_neuromodulators.py`

**Step 1: Failing test**

```python
def test_from_reward_production_rule_pulses_dopamine():
    pytest.importorskip("cupy")
    import cupy as cp
    from sim.neuromodulators import (
        NeuromodulatorConfig, NeuromodulatorManager,
        ModulatorTarget, ProductionRule,
    )

    nm = NeuromodulatorConfig(
        name="dopamine",
        baseline=0.0,
        decay_tau_ms=500.0,
        production_rules=[ProductionRule(rule_type="from_reward", sensitivity=1.0)],
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)

    # Mock bridge with a reward signal
    class _Bridge:
        class _Cfg:
            current_reward_signal = 1.0
            reward_baseline = 0.0
        core_config = _Cfg()

    bridge = _Bridge()
    # First step: reward present, should rise
    mgr.step(bridge)
    assert mgr.get_concentration("dopamine") > 0.5

    # Withdraw reward, several steps: should decay toward 0
    bridge.core_config.current_reward_signal = 0.0
    for _ in range(2000):
        mgr.step(bridge)
    assert mgr.get_concentration("dopamine") < 0.05
```

**Step 2: Run, verify fails**

**Step 3: Implement from_reward inside step()**

```python
def step(self, bridge) -> None:
    for cfg in self._configs:
        conc = self._concentrations[cfg.name]
        decay_factor = math.exp(-self.dt_ms / max(cfg.decay_tau_ms, 1e-9))
        conc = cfg.baseline + (conc - cfg.baseline) * decay_factor

        # Production rules
        for rule in cfg.production_rules:
            production = self._compute_production(rule, cfg, bridge)
            conc += production

        conc = max(cfg.concentration_min, min(cfg.concentration_max, conc))
        self._concentrations[cfg.name] = conc

def _compute_production(self, rule: ProductionRule, cfg: NeuromodulatorConfig, bridge) -> float:
    if rule.rule_type == "manual":
        return 0.0
    if rule.rule_type == "from_reward":
        if bridge is None or not hasattr(bridge, "core_config"):
            return 0.0
        cc = bridge.core_config
        reward = float(getattr(cc, "current_reward_signal", 0.0))
        baseline = float(getattr(cc, "reward_baseline", 0.0))
        return rule.sensitivity * (reward - baseline)
    if rule.rule_type == "from_error_persistence":
        # Implemented in Task 5
        return 0.0
    return 0.0
```

**Step 4: Run, verify passes**

**Step 5: Commit**

```bash
git commit -am "feat(neuromodulators): from_reward production rule (dopamine pulse on reward)"
```

---

## Task 5: Implement `from_error_persistence` production rule (for noradrenaline)

**Files:** `sim/neuromodulators.py`, `tests/test_neuromodulators.py`

**Step 1: Failing test**

```python
def test_from_error_persistence_rises_with_sustained_negative_reward():
    pytest.importorskip("cupy")
    import cupy as cp
    from sim.neuromodulators import (
        NeuromodulatorConfig, NeuromodulatorManager, ProductionRule,
    )
    nm = NeuromodulatorConfig(
        name="noradrenaline",
        baseline=0.1,
        decay_tau_ms=2000.0,
        production_rules=[ProductionRule(
            rule_type="from_error_persistence",
            sensitivity=0.5,
            threshold=0.3,
            window_ms=200.0,
        )],
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)

    class _Bridge:
        class _Cfg:
            current_reward_signal = 0.0
            reward_baseline = 0.0
        core_config = _Cfg()
    bridge = _Bridge()

    # Sustained negative reward: noradrenaline should rise above baseline
    bridge.core_config.current_reward_signal = -1.0
    for _ in range(1000):
        mgr.step(bridge)
    assert mgr.get_concentration("noradrenaline") > 0.5

    # Reward back to zero, NE decays back toward baseline
    bridge.core_config.current_reward_signal = 0.0
    for _ in range(8000):
        mgr.step(bridge)
    assert abs(mgr.get_concentration("noradrenaline") - nm.baseline) < 0.05
```

**Step 2: Verify fails**

**Step 3: Implement error_persistence rule**

```python
def _compute_production(self, rule, cfg, bridge):
    ...
    if rule.rule_type == "from_error_persistence":
        if bridge is None or not hasattr(bridge, "core_config"):
            return 0.0
        cc = bridge.core_config
        reward = float(getattr(cc, "current_reward_signal", 0.0))
        baseline = float(getattr(cc, "reward_baseline", 0.0))
        err = abs(reward - baseline)

        # Update EMA of error magnitude
        state = self._rule_state[cfg.name]
        ema_alpha = self.dt_ms / max(rule.window_ms, 1e-9)
        ema = state.get("err_ema", 0.0)
        ema = ema + ema_alpha * (err - ema)
        state["err_ema"] = ema

        # Produce iff sustained EMA exceeds threshold
        if ema > rule.threshold:
            return rule.sensitivity * (ema - rule.threshold) * (self.dt_ms / 1000.0)
        return 0.0
    ...
```

**Step 4: Verify passes**

**Step 5: Commit**

```bash
git commit -am "feat(neuromodulators): from_error_persistence production rule for noradrenaline"
```

---

## Task 6: `apply_synaptic_gain` target effect

**Files:** `sim/neuromodulators.py`, `tests/test_neuromodulators.py`

**Step 1: Failing test**

```python
def test_apply_synaptic_gain_returns_multiplier():
    pytest.importorskip("cupy")
    import cupy as cp
    from sim.neuromodulators import (
        NeuromodulatorConfig, NeuromodulatorManager,
        ModulatorTarget, ProductionRule,
    )
    nm = NeuromodulatorConfig(
        name="dopamine", baseline=0.0, decay_tau_ms=500.0,
        targets=[ModulatorTarget(target_type="synaptic_gain", scope="all", sensitivity=0.5)],
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    mgr.set_concentration("dopamine", 1.0)

    # synaptic_gain effect formula: 1.0 + sensitivity*(conc - baseline) = 1.5
    multiplier = mgr.compute_synaptic_gain_multiplier()
    # Returns scalar (since scope=all)
    assert abs(multiplier - 1.5) < 1e-6


def test_apply_synaptic_gain_zero_when_subsystem_unused():
    # No modulator with synaptic_gain target -> multiplier is 1.0
    pytest.importorskip("cupy")
    import cupy as cp
    from sim.neuromodulators import NeuromodulatorManager
    mgr = NeuromodulatorManager([], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    assert mgr.compute_synaptic_gain_multiplier() == 1.0
```

**Step 2: Verify fails**

**Step 3: Implement compute_synaptic_gain_multiplier**

```python
def compute_synaptic_gain_multiplier(self) -> float:
    """Aggregate synaptic_gain effects across all modulators (scope=all only).

    Returns a scalar multiplier. Per-trait or per-group scoping is
    deferred (returns the all-scope contribution; trait/group requires
    GPU array support added in a later task).
    """
    multiplier = 1.0
    for cfg in self._configs:
        for tgt in cfg.targets:
            if tgt.target_type != "synaptic_gain":
                continue
            if tgt.scope != "all":
                continue  # trait/group support added later
            conc = self._concentrations[cfg.name]
            multiplier *= 1.0 + tgt.sensitivity * (conc - cfg.baseline)
    return float(max(0.0, multiplier))
```

**Step 4: Verify passes**

**Step 5: Commit**

```bash
git commit -am "feat(neuromodulators): synaptic_gain target multiplier (scope=all)"
```

---

## Task 7: `apply_plasticity_rate` target effect

**Files:** `sim/neuromodulators.py`, `tests/test_neuromodulators.py`

**Step 1: Failing test**

```python
def test_apply_plasticity_rate_returns_multiplier():
    pytest.importorskip("cupy")
    import cupy as cp
    from sim.neuromodulators import (
        NeuromodulatorConfig, NeuromodulatorManager, ModulatorTarget,
    )
    nm = NeuromodulatorConfig(
        name="dopamine", baseline=0.0, decay_tau_ms=500.0,
        targets=[ModulatorTarget(target_type="plasticity_rate", sensitivity=2.0)],
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    mgr.set_concentration("dopamine", 0.5)

    # 1 + 2 * 0.5 = 2.0
    assert abs(mgr.compute_plasticity_rate_multiplier() - 2.0) < 1e-6
```

**Step 2: Verify fails**

**Step 3: Implement, mirroring synaptic_gain**

```python
def compute_plasticity_rate_multiplier(self) -> float:
    multiplier = 1.0
    for cfg in self._configs:
        for tgt in cfg.targets:
            if tgt.target_type != "plasticity_rate":
                continue
            if tgt.scope != "all":
                continue
            conc = self._concentrations[cfg.name]
            multiplier *= 1.0 + tgt.sensitivity * (conc - cfg.baseline)
    return float(max(0.0, multiplier))
```

**Step 4: Verify passes**

**Step 5: Commit**

```bash
git commit -am "feat(neuromodulators): plasticity_rate target multiplier"
```

---

## Task 8: `apply_excitability_drive` target effect (additive current)

**Files:** `sim/neuromodulators.py`, `tests/test_neuromodulators.py`

**Step 1: Failing test**

```python
def test_apply_excitability_drive_returns_current():
    pytest.importorskip("cupy")
    import cupy as cp
    from sim.neuromodulators import (
        NeuromodulatorConfig, NeuromodulatorManager, ModulatorTarget,
    )
    nm = NeuromodulatorConfig(
        name="noradrenaline", baseline=0.1, decay_tau_ms=2000.0,
        targets=[ModulatorTarget(target_type="excitability_drive",
                                 scope="all", sensitivity=50.0)],
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    mgr.set_concentration("noradrenaline", 0.4)

    # additive: 50.0 * (0.4 - 0.1) = 15 pA
    drive = mgr.compute_excitability_drive_pA()
    assert abs(drive - 15.0) < 1e-6


def test_apply_excitability_drive_supports_trait_scope():
    """trait:idx -> per-neuron array with current applied only to those neurons."""
    pytest.importorskip("cupy")
    import cupy as cp
    import numpy as np
    from sim.neuromodulators import (
        NeuromodulatorConfig, NeuromodulatorManager, ModulatorTarget,
    )
    nm = NeuromodulatorConfig(
        name="ne", baseline=0.0, decay_tau_ms=2000.0,
        targets=[ModulatorTarget(target_type="excitability_drive",
                                 scope="trait:1", sensitivity=10.0)],
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=4, cp_module=cp)
    mgr.set_concentration("ne", 1.0)

    traits = cp.asarray([0, 1, 1, 0], dtype=cp.int32)
    drive = mgr.compute_excitability_drive_per_neuron(cp_traits=traits)
    drive_np = cp.asnumpy(drive)
    expected = np.array([0.0, 10.0, 10.0, 0.0], dtype=np.float32)
    assert np.allclose(drive_np, expected, atol=1e-5)
```

**Step 2: Verify fails**

**Step 3: Implement**

```python
def compute_excitability_drive_pA(self) -> float:
    """Scalar additive drive (scope=all only)."""
    drive = 0.0
    for cfg in self._configs:
        for tgt in cfg.targets:
            if tgt.target_type != "excitability_drive":
                continue
            if tgt.scope != "all":
                continue
            conc = self._concentrations[cfg.name]
            drive += tgt.sensitivity * (conc - cfg.baseline)
    return float(drive)


def compute_excitability_drive_per_neuron(self, cp_traits=None, group_indices=None):
    """Per-neuron additive drive array, honoring scope=trait:N or group:NAME.

    Returns None if no per-neuron-scoped targets exist.
    """
    cp = self._cp
    if cp is None:
        return None
    drive = None
    for cfg in self._configs:
        for tgt in cfg.targets:
            if tgt.target_type != "excitability_drive":
                continue
            if tgt.scope == "all":
                continue  # handled by compute_excitability_drive_pA
            conc = self._concentrations[cfg.name]
            value = tgt.sensitivity * (conc - cfg.baseline)
            if drive is None:
                drive = cp.zeros(self._n_neurons, dtype=cp.float32)
            if tgt.scope.startswith("trait:") and cp_traits is not None:
                idx = int(tgt.scope.split(":", 1)[1])
                drive = drive + cp.where(cp_traits == idx, cp.float32(value), cp.float32(0.0))
            elif tgt.scope.startswith("group:") and group_indices is not None:
                gname = tgt.scope.split(":", 1)[1]
                indices = group_indices.get(gname)
                if indices is not None:
                    mask = cp.zeros(self._n_neurons, dtype=cp.bool_)
                    mask[cp.asarray(indices, dtype=cp.int32)] = True
                    drive = drive + cp.where(mask, cp.float32(value), cp.float32(0.0))
    return drive
```

**Step 4: Verify passes**

**Step 5: Commit**

```bash
git commit -am "feat(neuromodulators): excitability_drive target (scope all + trait + group)"
```

---

## Task 9: Add `enable_neuromodulator_subsystem` config flag + plumbing

**Files:**
- Modify: `sim/config.py` (CoreSimConfig)
- Modify: `sim/bridge.py` (allocate manager, call step)
- Test: `tests/test_neuromodulators.py`

**Step 1: Failing test**

```python
def test_bridge_allocates_manager_when_subsystem_enabled():
    pytest.importorskip("cupy")
    from sim import (
        SimulationBridge, CoreSimConfig, VisualizationConfig,
        RuntimeState, GPUConfig,
    )
    from sim.enums import NeuronModel
    from sim.neuromodulators import NeuromodulatorConfig

    cfg = CoreSimConfig()
    cfg.num_neurons = 50
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = 42
    cfg.enable_neuromodulator_subsystem = True
    cfg.neuromodulators = [NeuromodulatorConfig(name="dopamine", baseline=0.0)]

    sb = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    sb._initialize_simulation_data(called_from_playback_init=False)
    assert sb.neuromodulator_manager is not None
    assert sb.neuromodulator_manager.get_concentration("dopamine") == 0.0
    sb.clear_simulation_state_and_gpu_memory()


def test_bridge_no_manager_when_subsystem_disabled():
    pytest.importorskip("cupy")
    from sim import SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.enums import NeuronModel
    cfg = CoreSimConfig()
    cfg.num_neurons = 50
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = 42
    # default: enable_neuromodulator_subsystem = False
    sb = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    sb._initialize_simulation_data(called_from_playback_init=False)
    assert sb.neuromodulator_manager is None
    sb.clear_simulation_state_and_gpu_memory()
```

**Step 2: Verify fails (no enable_neuromodulator_subsystem field on config; no neuromodulator_manager attr)**

**Step 3: Implement**

In `sim/config.py`:
```python
# After reward modulation block:
# Neuromodulator subsystem (Session E) — opt-in framework that subsumes
# the legacy current_reward_signal path when active.
enable_neuromodulator_subsystem: bool = False
neuromodulators: list = field(default_factory=list)  # List[NeuromodulatorConfig]
```

In `sim/bridge.py`:
```python
# In __init__:
self.neuromodulator_manager = None

# In _initialize_simulation_data, after eligibility allocation:
if cfg.enable_neuromodulator_subsystem and cfg.neuromodulators:
    from sim.neuromodulators import NeuromodulatorManager
    self.neuromodulator_manager = NeuromodulatorManager(
        cfg.neuromodulators, cfg.dt_ms,
    )
    self.neuromodulator_manager.initialize(cfg.num_neurons, cp)
else:
    self.neuromodulator_manager = None
```

**Step 4: Verify passes**

**Step 5: Commit**

```bash
git commit -am "feat(bridge): allocate NeuromodulatorManager when subsystem enabled"
```

---

## Task 10: Wire `manager.step(self)` into bridge per-step loop

**Files:** `sim/bridge.py`, test in `tests/test_neuromodulators.py`

**Step 1: Failing test**

```python
def test_bridge_step_advances_modulator_concentration():
    pytest.importorskip("cupy")
    from sim import SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.enums import NeuronModel
    from sim.neuromodulators import NeuromodulatorConfig, ProductionRule

    cfg = CoreSimConfig()
    cfg.num_neurons = 50
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = 42
    cfg.enable_neuromodulator_subsystem = True
    cfg.enable_reward_modulation = True
    cfg.neuromodulators = [
        NeuromodulatorConfig(
            name="dopamine", baseline=0.0, decay_tau_ms=500.0,
            production_rules=[ProductionRule(rule_type="from_reward", sensitivity=1.0)],
        )
    ]
    sb = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    sb._initialize_simulation_data(called_from_playback_init=False)

    sb.core_config.current_reward_signal = 1.0
    sb._run_one_simulation_step()
    sb.runtime_state.current_time_step += 1
    # Concentration should have moved away from 0
    assert sb.neuromodulator_manager.get_concentration("dopamine") > 0.5
    sb.clear_simulation_state_and_gpu_memory()
```

**Step 2: Verify fails (manager.step not called)**

**Step 3: Implement — add a single line in bridge `_run_one_simulation_step` near the end of the plasticity block**

In `sim/bridge.py` (find the end of section 4c, after reward modulation):
```python
# Section 4d: Neuromodulator subsystem update (opt-in)
if cfg.enable_neuromodulator_subsystem and self.neuromodulator_manager is not None:
    self.neuromodulator_manager.step(self)
```

**Step 4: Verify passes**

**Step 5: Commit**

```bash
git commit -am "feat(bridge): call neuromodulator_manager.step() each simulation step"
```

---

## Task 11: Wire synaptic_gain effect into effective_synaptic_strength

**Files:** `sim/bridge.py`, `tests/test_neuromodulators.py`

**Step 1: Failing test**

```python
def test_synaptic_gain_modulates_conductance_when_enabled():
    """When dopamine has a synaptic_gain target, effective synapse strength
    should scale with concentration. Verify by checking propagated conductance."""
    pytest.importorskip("cupy")
    import cupy as cp
    import numpy as np
    from sim import SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.enums import NeuronModel
    from sim.neuromodulators import NeuromodulatorConfig, ModulatorTarget

    cfg = CoreSimConfig()
    cfg.num_neurons = 50
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = 42
    cfg.enable_neuromodulator_subsystem = True
    cfg.neuromodulators = [
        NeuromodulatorConfig(
            name="dopamine", baseline=0.0, decay_tau_ms=10000.0,
            targets=[ModulatorTarget(target_type="synaptic_gain", scope="all", sensitivity=1.0)],
        )
    ]
    sb = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    sb._initialize_simulation_data(called_from_playback_init=False)

    # Set concentration to 1.0 -> multiplier = 2.0
    sb.neuromodulator_manager.set_concentration("dopamine", 1.0)
    # Force a presynaptic firing pattern
    sb.cp_prev_firing_states[:] = False
    sb.cp_prev_firing_states[0] = True
    sb._run_one_simulation_step()
    g_e_with_dopamine = float(cp.sum(sb.cp_conductance_g_e).get())

    # Same with dopamine off
    sb.neuromodulator_manager.set_concentration("dopamine", 0.0)
    sb.cp_conductance_g_e[:] = 0.0
    sb.cp_prev_firing_states[:] = False
    sb.cp_prev_firing_states[0] = True
    sb._run_one_simulation_step()
    g_e_without_dopamine = float(cp.sum(sb.cp_conductance_g_e).get())

    # With dopamine = 1.0 (multiplier 2.0), conductance increment should
    # be roughly 2x. Allow some slack because OU noise + decay also contribute.
    assert g_e_with_dopamine > 1.5 * g_e_without_dopamine, (
        f"Expected ~2x conductance with dopamine, got {g_e_with_dopamine}/{g_e_without_dopamine}"
    )
    sb.clear_simulation_state_and_gpu_memory()
```

**Step 2: Verify fails**

**Step 3: Implement** — find the `effective_synaptic_strength = base_synaptic_weights * stp_u_active * stp_x_active` line and the no-STP branch. Multiply by neuromodulator gain when subsystem is on.

```python
# After computing effective_synaptic_strength in both STP and non-STP branches,
# add (only one site, abstracted):
if (cfg.enable_neuromodulator_subsystem
        and self.neuromodulator_manager is not None):
    nm_gain = self.neuromodulator_manager.compute_synaptic_gain_multiplier()
    if abs(nm_gain - 1.0) > 1e-9:
        effective_synaptic_strength = effective_synaptic_strength * nm_gain
```

**Step 4: Verify passes**

**Step 5: Commit**

```bash
git commit -am "feat(bridge): apply neuromodulator synaptic_gain to effective synapse strength"
```

---

## Task 12: Wire plasticity_rate effect into STDP and reward modulation

**Files:** `sim/bridge.py`, `tests/test_neuromodulators.py`

**Step 1: Failing test**

```python
def test_plasticity_rate_modulates_reward_lr():
    """plasticity_rate target should multiply effective reward_learning_rate."""
    pytest.importorskip("cupy")
    import cupy as cp
    from sim import SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.enums import NeuronModel
    from sim.neuromodulators import NeuromodulatorConfig, ModulatorTarget

    cfg = CoreSimConfig()
    cfg.num_neurons = 50
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = 42
    cfg.enable_stdp = True
    cfg.enable_reward_modulation = True
    cfg.reward_learning_rate = 0.01
    cfg.enable_neuromodulator_subsystem = True
    cfg.neuromodulators = [
        NeuromodulatorConfig(
            name="dopamine", baseline=0.0, decay_tau_ms=10000.0,
            targets=[ModulatorTarget(target_type="plasticity_rate",
                                      scope="all", sensitivity=2.0)],
        )
    ]
    sb = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    sb._initialize_simulation_data(called_from_playback_init=False)
    sb.neuromodulator_manager.set_concentration("dopamine", 0.5)
    # 1 + 2*0.5 = 2.0x

    # Compute effective lr exposed via getter (added in this task)
    eff_lr = sb._effective_reward_learning_rate()
    assert abs(eff_lr - 0.02) < 1e-6
    sb.clear_simulation_state_and_gpu_memory()
```

**Step 2: Verify fails (no `_effective_reward_learning_rate` method)**

**Step 3: Implement** — add helper + use it in the reward modulation path

In `sim/bridge.py`, near the reward modulation block:
```python
def _effective_reward_learning_rate(self) -> float:
    base = float(self.core_config.reward_learning_rate)
    if (self.core_config.enable_neuromodulator_subsystem
            and self.neuromodulator_manager is not None):
        return base * self.neuromodulator_manager.compute_plasticity_rate_multiplier()
    return base
```

Then update the reward-modulation block to use `self._effective_reward_learning_rate()` instead of `cfg.reward_learning_rate`.

Also update the STDP weight update path to multiply `cfg.stdp_a_plus` and `cfg.stdp_a_minus` by the multiplier when subsystem is on:
```python
plasticity_mult = 1.0
if (cfg.enable_neuromodulator_subsystem
        and self.neuromodulator_manager is not None):
    plasticity_mult = self.neuromodulator_manager.compute_plasticity_rate_multiplier()
a_plus_eff = cfg.stdp_a_plus * plasticity_mult
a_minus_eff = cfg.stdp_a_minus * plasticity_mult
```

**Step 4: Verify passes**

**Step 5: Commit**

```bash
git commit -am "feat(bridge): plasticity_rate effect modulates STDP amplitudes + reward lr"
```

---

## Task 13: Wire excitability_drive into membrane current injection

**Files:** `sim/bridge.py`, `tests/test_neuromodulators.py`

**Step 1: Failing test**

```python
def test_excitability_drive_increases_firing_rate():
    """When NE is high and has excitability_drive target, neurons fire more."""
    pytest.importorskip("cupy")
    import cupy as cp
    from sim import SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.enums import NeuronModel
    from sim.neuromodulators import NeuromodulatorConfig, ModulatorTarget

    def run_total_spikes(ne_concentration: float) -> int:
        cfg = CoreSimConfig()
        cfg.num_neurons = 100
        cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
        cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
        cfg.dt_ms = 1.0
        cfg.seed = 42
        cfg.enable_neuromodulator_subsystem = True
        cfg.neuromodulators = [
            NeuromodulatorConfig(
                name="ne", baseline=0.0, decay_tau_ms=100000.0,
                targets=[ModulatorTarget(target_type="excitability_drive",
                                          scope="all", sensitivity=100.0)],
            )
        ]
        sb = SimulationBridge(
            core_config=cfg, viz_config=VisualizationConfig(),
            runtime_state=RuntimeState(), gpu_config=GPUConfig(),
        )
        sb._initialize_simulation_data(called_from_playback_init=False)
        sb.neuromodulator_manager.set_concentration("ne", ne_concentration)
        total = 0
        for _ in range(200):
            sb._run_one_simulation_step()
            sb.runtime_state.current_time_step += 1
            total += int(cp.sum(sb.cp_firing_states).get())
        sb.clear_simulation_state_and_gpu_memory()
        return total

    n_low = run_total_spikes(0.0)
    n_high = run_total_spikes(1.0)
    # 100 pA boost on every cell should produce many more spikes
    assert n_high > n_low + 50, f"Excitability drive not effective: low={n_low}, high={n_high}"
```

**Step 2: Verify fails**

**Step 3: Implement** — find where `total_input_current_pA` is composed and add the drive

```python
# After total_input_current_pA = synaptic_current_I_syn_pA + self.cp_external_input_current
if (cfg.enable_neuromodulator_subsystem
        and self.neuromodulator_manager is not None):
    nm_drive_scalar = self.neuromodulator_manager.compute_excitability_drive_pA()
    if abs(nm_drive_scalar) > 1e-9:
        total_input_current_pA = total_input_current_pA + nm_drive_scalar
    nm_drive_per_n = self.neuromodulator_manager.compute_excitability_drive_per_neuron(
        cp_traits=self.cp_traits,
    )
    if nm_drive_per_n is not None:
        total_input_current_pA = total_input_current_pA + nm_drive_per_n
```

**Step 4: Verify passes**

**Step 5: Commit**

```bash
git commit -am "feat(bridge): excitability_drive target adds to membrane input current"
```

---

## Task 14: Validation — biological benchmarks must not regress with subsystem OFF

**Files:** `tests/test_neuromod_benchmark_compat.py`

**Step 1: Failing test**

```python
def test_tiny_sim_spike_count_unchanged_when_subsystem_off():
    """With subsystem disabled (default), the locked tiny-seeded-sim spike
    count from the drift detector must still be 170 +- 10.
    Mirrors test_benchmark_drift.test_tiny_seeded_sim_spike_count_in_range
    but explicitly verifies the new subsystem code paths are inactive."""
    pytest.importorskip("cupy")
    from tests.test_benchmark_drift import _build_tiny_sim, _run_and_count
    sb, cfg = _build_tiny_sim(seed=42)
    assert cfg.enable_neuromodulator_subsystem is False  # default
    assert sb.neuromodulator_manager is None
    total, _ = _run_and_count(sb, cfg, n_steps=200)
    sb.clear_simulation_state_and_gpu_memory()
    assert 160 <= total <= 180, f"Drift detected: {total}"
```

**Step 2: Verify passes immediately (default-off path)**

**Step 3: No code change needed; just adds a regression guard**

**Step 4: Run full existing test suite**
Run: `python -m pytest tests/ -q --ignore=tests/test_ui_build.py`
Expected: all pre-existing tests still pass.

**Step 5: Commit**

```bash
git commit -am "test(neuromodulators): regression guard — subsystem off must not change tiny-sim drift"
```

---

## Task 15: Validation — replicate dopamine reward learning via subsystem

**Files:** `tests/test_neuromod_dopamine_replay.py`

**Step 1: Failing test**

```python
def test_dopamine_subsystem_replicates_legacy_reward_modulation():
    """Two configurations of the bridge — legacy reward modulation
    (enable_reward_modulation=True, subsystem off) vs new subsystem
    (subsystem on with dopamine modulator using from_reward + plasticity_rate
    target) — should produce roughly equivalent weight changes given the
    same reward signal.
    """
    pytest.importorskip("cupy")
    import cupy as cp
    import numpy as np
    from sim import SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.enums import NeuronModel
    from sim.neuromodulators import NeuromodulatorConfig, ModulatorTarget, ProductionRule

    def _run(use_subsystem: bool) -> float:
        cfg = CoreSimConfig()
        cfg.num_neurons = 100
        cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
        cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
        cfg.dt_ms = 1.0
        cfg.seed = 42
        cfg.enable_stdp = True
        cfg.enable_reward_modulation = True
        cfg.reward_learning_rate = 0.01
        if use_subsystem:
            cfg.enable_neuromodulator_subsystem = True
            cfg.neuromodulators = [
                NeuromodulatorConfig(
                    name="dopamine", baseline=0.0, decay_tau_ms=500.0,
                    production_rules=[ProductionRule(rule_type="from_reward",
                                                       sensitivity=1.0)],
                    targets=[],  # no extra effect; legacy reward path still runs
                ),
            ]
        sb = SimulationBridge(
            core_config=cfg, viz_config=VisualizationConfig(),
            runtime_state=RuntimeState(), gpu_config=GPUConfig(),
        )
        sb._initialize_simulation_data(called_from_playback_init=False)
        # Pulse reward
        sb.core_config.current_reward_signal = 0.5
        for _ in range(100):
            sb._run_one_simulation_step()
            sb.runtime_state.current_time_step += 1
        sb.core_config.current_reward_signal = 0.0
        for _ in range(100):
            sb._run_one_simulation_step()
            sb.runtime_state.current_time_step += 1
        # Snapshot mean weight
        w_mean = float(cp.mean(sb.cp_connections.data).get())
        sb.clear_simulation_state_and_gpu_memory()
        return w_mean

    w_legacy = _run(use_subsystem=False)
    w_subsys = _run(use_subsystem=True)
    # Should differ by < 5% (subsystem with no targets should be no-op)
    assert abs(w_legacy - w_subsys) / max(abs(w_legacy), 1e-6) < 0.05, (
        f"subsystem-on with no targets diverged: legacy={w_legacy} vs subsys={w_subsys}"
    )
```

**Step 2: Verify passes**

**Step 3: No code changes needed — this validates Task 9-13 don't break legacy when targets are empty**

**Step 4: Commit**

```bash
git commit -am "test(neuromodulators): subsystem with no targets matches legacy reward path"
```

---

## Task 16: Add G9-runner integration: optional `nm_configs` kwarg

**Files:** `research/runners/g9_runner.py`, `tests/test_g9_runner_smoke.py`

**Step 1: Failing test**

```python
def test_g9_smoke_with_neuromodulators(tmp_path):
    """G9 runner accepts a list of NeuromodulatorConfig and threads it through."""
    pytest.importorskip("cupy")
    from research.runners.g9_runner import run_g9_episode
    from sim.neuromodulators import (
        NeuromodulatorConfig, ProductionRule, ModulatorTarget,
    )

    nm_configs = [
        NeuromodulatorConfig(
            name="dopamine", baseline=0.0, decay_tau_ms=500.0,
            production_rules=[ProductionRule(rule_type="from_reward", sensitivity=1.0)],
        ),
        NeuromodulatorConfig(
            name="noradrenaline", baseline=0.1, decay_tau_ms=2000.0,
            production_rules=[ProductionRule(
                rule_type="from_error_persistence",
                sensitivity=0.5, threshold=0.3, window_ms=1000.0,
            )],
            targets=[ModulatorTarget(target_type="excitability_drive",
                                       scope="group:motor", sensitivity=30.0)],
        ),
    ]
    out = tmp_path / "g9_nm.json"
    r = run_g9_episode(
        out_path=str(out),
        seed=42, n_steps=30, grid_size=8,
        start_pos=(1, 1), goal_pos=(6, 6),
        learning_rate=0.05,
        action_selection="argmax",
        nm_configs=nm_configs,
        verbose=False,
    )
    import json
    data = json.load(open(out))
    assert "neuromodulator_concentrations" in data
    assert "dopamine" in data["neuromodulator_concentrations"]
    assert "noradrenaline" in data["neuromodulator_concentrations"]
```

**Step 2: Verify fails**

**Step 3: Implement nm_configs param + record concentrations at end**

In `g9_runner.py`:
- Add `nm_configs: list = None` to both `_build_g9_plan` and `run_g9_episode`.
- In `_build_g9_plan`, if non-empty: `core_cfg.enable_neuromodulator_subsystem = True; core_cfg.neuromodulators = nm_configs`.
- At end of `run_g9_episode`, record final concentrations to results dict.
- For `group:NAME` scope: bridge needs to know group indices. Pass via `engine.group_manager` or via a setattr on bridge.

This is the most involved task. Sub-tasks:

  16a. Add `nm_configs` param and pass through to core_cfg.
  16b. Record final concentrations in results.
  16c. Wire group_indices to manager so `group:motor` scope works. (Add a `set_group_indices(self, group_dict)` method on manager, called by runner once groups are known.)

**Step 4: Verify passes**

**Step 5: Commit**

```bash
git commit -am "feat(g9): accept nm_configs and thread to neuromodulator subsystem"
```

---

## Task 17: Build a probe that proves NE excitability_drive escapes the silent-motor trap

**Files:** `research/run_g9_neuromod_probe.py`

**Step 1: Write probe script**

A 1800-step moving-goal scenario (same as the relaxed probe) but with a noradrenaline modulator wired to boost motor-neuron excitability when reward is persistently negative. 3 seeds.

**Step 2: Run it (foreground, ~30-45 min on RTX 3090)**

`python research/run_g9_neuromod_probe.py`

**Step 3: Analyze with gate_metrics.py**

`python research/gate_metrics.py "research/findings/raw/g9/g9_neuromod_relaxed_*.json"`

**Step 4: Document findings**
Either way:
- If P1 acquires for ≥ 2/3 seeds → silent-motor trap dissolved by NE
- If still 0/3 → diagnose; framework still validated, just need different parameters

**Step 5: Commit raw + findings**

```bash
git add research/run_g9_neuromod_probe.py research/findings/raw/g9/g9_neuromod_relaxed*.json research/findings/2026-04-24-neuromod-validation.md
git commit -m "findings: NE excitability_drive on silent-motor-trap (RESULT)"
```

---

## Task 18: Final validation pass + push

**Files:** none changed; just running things

**Step 1: Run full test suite**
`python -m pytest tests/ -q --ignore=tests/test_ui_build.py`
Expected: all green.

**Step 2: Run drift test explicitly**
`python -m pytest tests/test_benchmark_drift.py -v`
Expected: 3 pass + 1 skip.

**Step 3: Push branch**
`git push -u origin neuromodulator-subsystem`

**Step 4: Decide merge based on Task 17 results**
- If NE probe shows clear improvement → recommend merge.
- If neutral → leave on branch as documented framework.

**Step 5: Update CLAUDE.md**
Add a §Neuromodulator-subsystem section listing entry points (`sim/neuromodulators.py`, `enable_neuromodulator_subsystem`, available production rules, target types).

---

## Wrap-up

Wiki-sync the session-E.1 work. Update todo list and report final state to user.
