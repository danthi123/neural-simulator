"""Unit tests for the neuromodulator subsystem (Session E.1).

See docs/plans/2026-04-24-neuromodulator-subsystem.md for the full plan
and research/findings/2026-04-24-session-d-part-a.md §4 for motivation.

The subsystem replaces the one-off `current_reward_signal` /
`cp_synaptic_gain_modulator` hacks with a declarative framework where
each neuromodulator is a `NeuromodulatorConfig` with concentration,
decay tau, baseline, production rules, and configurable receptor targets.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------- Task 1: dataclass shape ----------

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
    assert nm.baseline == 0.2


def test_modulator_target_defaults():
    from sim.neuromodulators import ModulatorTarget

    t = ModulatorTarget(target_type="synaptic_gain")
    assert t.target_type == "synaptic_gain"
    assert t.scope == "all"
    assert t.sensitivity == 1.0


def test_production_rule_defaults():
    from sim.neuromodulators import ProductionRule

    r = ProductionRule(rule_type="from_reward")
    assert r.rule_type == "from_reward"
    assert r.sensitivity == 1.0
    assert r.threshold == 0.5
    assert r.window_ms == 500.0


# ---------- Task 2: manager state allocation ----------

def test_manager_allocates_concentration_per_modulator():
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import NeuromodulatorConfig, NeuromodulatorManager

    nms = [
        NeuromodulatorConfig(name="dopamine", baseline=0.0),
        NeuromodulatorConfig(name="noradrenaline", baseline=0.2),
    ]
    mgr = NeuromodulatorManager(nms, dt_ms=1.0)
    mgr.initialize(n_neurons=100, cp_module=cp)

    assert mgr.get_concentration("dopamine") == 0.0
    assert abs(mgr.get_concentration("noradrenaline") - 0.2) < 1e-6
    with pytest.raises(KeyError):
        mgr.get_concentration("serotonin")


def test_manager_empty_modulator_list_initializes_cleanly():
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import NeuromodulatorManager

    mgr = NeuromodulatorManager([], dt_ms=1.0)
    mgr.initialize(n_neurons=100, cp_module=cp)
    assert mgr.modulator_names() == []


# ---------- Task 3: per-step exponential decay ----------

def test_concentration_decays_toward_baseline():
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import NeuromodulatorConfig, NeuromodulatorManager

    nm = NeuromodulatorConfig(name="dopamine", baseline=0.0, decay_tau_ms=100.0)
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)

    # Perturb to 1.0 manually, then decay over 1 tau (100 steps of 1ms).
    mgr.set_concentration("dopamine", 1.0)
    for _ in range(100):
        mgr.step(bridge=None)
    # exp(-1) ≈ 0.368
    assert 0.30 < mgr.get_concentration("dopamine") < 0.45

    # Far past steady state, converge to baseline.
    for _ in range(2000):
        mgr.step(bridge=None)
    assert abs(mgr.get_concentration("dopamine")) < 0.01


def test_concentration_decays_to_nonzero_baseline():
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import NeuromodulatorConfig, NeuromodulatorManager

    nm = NeuromodulatorConfig(name="ne", baseline=0.3, decay_tau_ms=200.0)
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    mgr.set_concentration("ne", 1.5)
    for _ in range(2000):
        mgr.step(bridge=None)
    assert abs(mgr.get_concentration("ne") - 0.3) < 0.01


def test_concentration_clipped_to_min_max():
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import NeuromodulatorConfig, NeuromodulatorManager

    nm = NeuromodulatorConfig(
        name="da", baseline=0.0, decay_tau_ms=10000.0,
        concentration_min=0.0, concentration_max=2.0,
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    mgr.set_concentration("da", 100.0)  # absurdly high, should clip to max
    mgr.step(bridge=None)
    assert mgr.get_concentration("da") <= 2.0
