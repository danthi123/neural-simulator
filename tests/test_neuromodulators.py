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


# ---------- Task 4: from_reward production rule ----------


class _FakeBridge:
    """Minimal bridge stub for production-rule unit tests."""

    class _Cfg:
        current_reward_signal = 0.0
        reward_baseline = 0.0

    def __init__(self):
        self.core_config = self._Cfg()


def test_from_reward_rule_pulses_dopamine_on_positive_reward():
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import (
        NeuromodulatorConfig,
        NeuromodulatorManager,
        ProductionRule,
    )

    nm = NeuromodulatorConfig(
        name="dopamine",
        baseline=0.0,
        decay_tau_ms=500.0,
        production_rules=[ProductionRule(rule_type="from_reward", sensitivity=1.0)],
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    bridge = _FakeBridge()

    bridge.core_config.current_reward_signal = 1.0
    mgr.step(bridge)
    # 1.0 added (sensitivity * (1 - 0)), tiny decay -> ~0.998
    assert mgr.get_concentration("dopamine") > 0.5

    bridge.core_config.current_reward_signal = 0.0
    for _ in range(2000):
        mgr.step(bridge)
    assert mgr.get_concentration("dopamine") < 0.05


def test_from_reward_rule_negative_reward_reduces_below_baseline():
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import (
        NeuromodulatorConfig,
        NeuromodulatorManager,
        ProductionRule,
    )

    nm = NeuromodulatorConfig(
        name="dopamine",
        baseline=0.5,
        decay_tau_ms=500.0,
        concentration_min=0.0,
        concentration_max=5.0,
        production_rules=[ProductionRule(rule_type="from_reward", sensitivity=1.0)],
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    bridge = _FakeBridge()

    # Sustained negative reward, dopamine drops below baseline (clipped at 0)
    bridge.core_config.current_reward_signal = -0.5
    for _ in range(500):
        mgr.step(bridge)
    # With baseline 0.5 and sustained -0.5 reward production, equilibrium
    # is below 0.5 (suppressed dopamine). Verify it's below baseline but
    # not below the clip floor.
    assert 0.0 <= mgr.get_concentration("dopamine") < 0.5


def test_from_reward_rule_ignores_missing_bridge_config():
    """When bridge has no core_config, rule produces 0 (no-op, no crash)."""
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import (
        NeuromodulatorConfig,
        NeuromodulatorManager,
        ProductionRule,
    )

    nm = NeuromodulatorConfig(
        name="dopamine",
        baseline=0.0,
        decay_tau_ms=500.0,
        production_rules=[ProductionRule(rule_type="from_reward", sensitivity=1.0)],
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    mgr.step(bridge=None)
    assert mgr.get_concentration("dopamine") == 0.0


# ---------- Task 5: from_error_persistence rule (noradrenaline-like) ----------


def test_from_error_persistence_rises_with_sustained_negative_reward():
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import (
        NeuromodulatorConfig,
        NeuromodulatorManager,
        ProductionRule,
    )

    nm = NeuromodulatorConfig(
        name="noradrenaline",
        baseline=0.1,
        decay_tau_ms=2000.0,
        concentration_min=0.0,
        concentration_max=2.0,
        production_rules=[
            ProductionRule(
                rule_type="from_error_persistence",
                sensitivity=2.0,
                threshold=0.3,
                window_ms=200.0,
            )
        ],
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    bridge = _FakeBridge()

    bridge.core_config.current_reward_signal = -1.0
    for _ in range(2000):
        mgr.step(bridge)
    # Persistent error >> threshold -> NE rises well above baseline
    assert mgr.get_concentration("noradrenaline") > 0.5

    bridge.core_config.current_reward_signal = 0.0
    for _ in range(15000):
        mgr.step(bridge)
    # NE returns to baseline
    assert abs(mgr.get_concentration("noradrenaline") - nm.baseline) < 0.05


def test_from_error_persistence_silent_when_below_threshold():
    """Small reward errors below threshold should NOT raise NE above baseline."""
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import (
        NeuromodulatorConfig,
        NeuromodulatorManager,
        ProductionRule,
    )

    nm = NeuromodulatorConfig(
        name="ne",
        baseline=0.1,
        decay_tau_ms=2000.0,
        production_rules=[
            ProductionRule(
                rule_type="from_error_persistence",
                sensitivity=2.0,
                threshold=0.3,
                window_ms=200.0,
            )
        ],
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    bridge = _FakeBridge()

    # Small errors well below threshold
    bridge.core_config.current_reward_signal = 0.1
    for _ in range(2000):
        mgr.step(bridge)
    # Should remain very near baseline
    assert abs(mgr.get_concentration("ne") - nm.baseline) < 0.05


def test_from_error_persistence_responds_to_positive_or_negative_error():
    """Rule uses |reward_error|, so |+0.8| and |-0.8| both elevate NE."""
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import (
        NeuromodulatorConfig,
        NeuromodulatorManager,
        ProductionRule,
    )

    nm_cfg = lambda: NeuromodulatorConfig(
        name="ne",
        baseline=0.0,
        decay_tau_ms=2000.0,
        production_rules=[
            ProductionRule(
                rule_type="from_error_persistence",
                sensitivity=2.0,
                threshold=0.3,
                window_ms=200.0,
            )
        ],
    )

    def _equilibrium(reward_signal: float) -> float:
        mgr = NeuromodulatorManager([nm_cfg()], dt_ms=1.0)
        mgr.initialize(n_neurons=10, cp_module=cp)
        bridge = _FakeBridge()
        bridge.core_config.current_reward_signal = reward_signal
        for _ in range(2000):
            mgr.step(bridge)
        return mgr.get_concentration("ne")

    pos = _equilibrium(+0.8)
    neg = _equilibrium(-0.8)
    # Both should produce non-trivial NE elevation; values should be similar
    # because the rule uses |error|.
    assert pos > 0.1
    assert neg > 0.1
    assert abs(pos - neg) < 0.1
