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
