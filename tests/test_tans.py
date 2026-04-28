"""Tests for Cluster B.3 - Cholinergic Interneurons (TANs).

Real BG TANs are tonically active (~5 Hz baseline) but pause briefly on
salient events (reward, novel stimuli). ACh release at corticostriatal
synapses creates "plasticity windows" - synapses only consolidate when
ACh is paused (low ACh = plasticity-on; high ACh = plasticity-off).

Task 1 introduces:
- A new ProductionRule type ``pause_on_reward`` that drives concentration
  DOWN by ``sensitivity * (|current_reward_signal| - threshold)`` per step.
  With negative sensitivity, |reward| above threshold drives concentration
  below baseline.
- A default ACh NeuromodulatorConfig helper that wires the pause rule to
  a tonic baseline of 1.0 with slow recovery (decay_tau_ms=500ms).

Task 2 will wire the ``plasticity_window_gate`` target type into the
bridge; for Task 1 it is data only (silently ignored by existing
compute_* methods, which is the project's forward-compatibility pattern).

See:
- docs/plans/2026-04-28-cluster-b3-tans-implementation.md
- research/findings/2026-04-28-cluster-b2-striatal-fsis-results.md
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Mirrors the _FakeBridge pattern from tests/test_neuromodulators.py.
class _FakeBridge:
    """Minimal bridge stub for production-rule unit tests."""

    class _Cfg:
        current_reward_signal = 0.0
        reward_baseline = 0.0

    def __init__(self):
        self.core_config = self._Cfg()


# ---------- Task 1: declarative shape ----------


def test_acetylcholine_neuromod_declared_with_pause_rule():
    """Verifies pause_on_reward production rule + plasticity_window_gate
    target both parse cleanly into the existing dataclasses.

    Task 2 will wire plasticity_window_gate into the bridge; for Task 1
    this is just declarative data and should not crash.
    """
    from sim.neuromodulators import (
        NeuromodulatorConfig,
        ProductionRule,
        ModulatorTarget,
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
    assert config.name == "acetylcholine"
    assert config.baseline == 1.0
    assert config.decay_tau_ms == 500.0
    assert config.production_rules[0].rule_type == "pause_on_reward"
    assert config.production_rules[0].sensitivity == -2.0
    assert config.production_rules[0].threshold == 0.1
    assert config.targets[0].target_type == "plasticity_window_gate"
    assert config.targets[0].scope == "all"


def test_default_acetylcholine_config_helper():
    """The runner should not need to inline the ACh config; a helper
    returns a sensible default that we can register directly."""
    from sim.neuromodulators import (
        NeuromodulatorConfig,
        _default_acetylcholine_config,
    )

    cfg = _default_acetylcholine_config()
    assert isinstance(cfg, NeuromodulatorConfig)
    assert cfg.name == "acetylcholine"
    # Tonic baseline > 0 (the "ACh is normally on" state).
    assert cfg.baseline > 0.0
    # Has at least one pause_on_reward rule with negative sensitivity.
    pause_rules = [
        r for r in cfg.production_rules if r.rule_type == "pause_on_reward"
    ]
    assert len(pause_rules) >= 1
    assert pause_rules[0].sensitivity < 0.0
    # Has plasticity_window_gate target wired up (data only for Task 1).
    gate_targets = [
        t for t in cfg.targets if t.target_type == "plasticity_window_gate"
    ]
    assert len(gate_targets) >= 1


# ---------- Task 1: pause_on_reward dynamics ----------


def test_pause_on_reward_drops_concentration_on_positive_reward():
    """When |reward| exceeds threshold and sensitivity is negative,
    concentration should drop below baseline."""
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import (
        NeuromodulatorConfig,
        NeuromodulatorManager,
        ProductionRule,
    )

    nm = NeuromodulatorConfig(
        name="acetylcholine",
        baseline=1.0,
        decay_tau_ms=500.0,
        concentration_min=0.0,
        concentration_max=2.0,
        production_rules=[
            ProductionRule(rule_type="pause_on_reward", sensitivity=-1.0, threshold=0.0),
        ],
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    bridge = _FakeBridge()

    # Sustained positive reward of 1.0 with sensitivity=-1.0 and threshold=0.0
    # produces -1.0 per step; combined with decay toward baseline=1.0, the
    # equilibrium is below baseline.
    bridge.core_config.current_reward_signal = 1.0
    for _ in range(200):
        mgr.step(bridge)

    final = mgr.get_concentration("acetylcholine")
    assert final < 1.0, f"ACh should drop below baseline=1.0, got {final:.3f}"
    # Should NOT clip below concentration_min=0.0.
    assert final >= 0.0


def test_pause_on_reward_responds_to_negative_reward_via_abs():
    """The rule uses |current_reward_signal|, so negative rewards also
    trigger pauses (real BG TANs pause on punishment too)."""
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import (
        NeuromodulatorConfig,
        NeuromodulatorManager,
        ProductionRule,
    )

    def _equilibrium(reward_signal: float) -> float:
        nm = NeuromodulatorConfig(
            name="acetylcholine",
            baseline=1.0,
            decay_tau_ms=500.0,
            concentration_min=0.0,
            concentration_max=2.0,
            production_rules=[
                ProductionRule(rule_type="pause_on_reward", sensitivity=-1.0, threshold=0.0),
            ],
        )
        mgr = NeuromodulatorManager([nm], dt_ms=1.0)
        mgr.initialize(n_neurons=10, cp_module=cp)
        bridge = _FakeBridge()
        bridge.core_config.current_reward_signal = reward_signal
        for _ in range(500):
            mgr.step(bridge)
        return mgr.get_concentration("acetylcholine")

    pos = _equilibrium(+0.8)
    neg = _equilibrium(-0.8)
    # Both should produce roughly equal pause magnitudes.
    assert pos < 1.0
    assert neg < 1.0
    assert abs(pos - neg) < 0.05, (
        f"Pause should be symmetric in |reward|: pos={pos:.3f} neg={neg:.3f}"
    )


def test_pause_on_reward_silent_below_threshold():
    """When |reward| is below threshold, concentration stays at baseline."""
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import (
        NeuromodulatorConfig,
        NeuromodulatorManager,
        ProductionRule,
    )

    nm = NeuromodulatorConfig(
        name="acetylcholine",
        baseline=1.0,
        decay_tau_ms=500.0,
        concentration_min=0.0,
        concentration_max=2.0,
        production_rules=[
            ProductionRule(rule_type="pause_on_reward", sensitivity=-2.0, threshold=0.5),
        ],
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    bridge = _FakeBridge()

    # |reward|=0.2 < threshold=0.5, so no pause production should occur.
    bridge.core_config.current_reward_signal = 0.2
    for _ in range(2000):
        mgr.step(bridge)

    # Should remain very close to baseline.
    assert abs(mgr.get_concentration("acetylcholine") - 1.0) < 0.02


def test_pause_on_reward_recovers_to_baseline_after_reward_stops():
    """After reward returns to 0, ACh should decay back to baseline at the
    decay_tau_ms time scale."""
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import (
        NeuromodulatorConfig,
        NeuromodulatorManager,
        ProductionRule,
    )

    nm = NeuromodulatorConfig(
        name="acetylcholine",
        baseline=1.0,
        decay_tau_ms=200.0,
        concentration_min=0.0,
        concentration_max=2.0,
        production_rules=[
            ProductionRule(rule_type="pause_on_reward", sensitivity=-1.0, threshold=0.0),
        ],
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    bridge = _FakeBridge()

    # Phase 1: drive reward to suppress ACh below baseline.
    bridge.core_config.current_reward_signal = 1.0
    for _ in range(200):
        mgr.step(bridge)
    paused = mgr.get_concentration("acetylcholine")
    assert paused < 0.9, f"Expected pause below baseline, got {paused:.3f}"

    # Phase 2: reward off, observe recovery toward baseline=1.0.
    bridge.core_config.current_reward_signal = 0.0
    for _ in range(2000):
        mgr.step(bridge)
    recovered = mgr.get_concentration("acetylcholine")
    assert abs(recovered - 1.0) < 0.02, (
        f"Should recover to baseline=1.0 after reward stops, got {recovered:.3f}"
    )


def test_pause_on_reward_clipped_to_concentration_min():
    """Even with very strong reward, concentration cannot drop below
    concentration_min."""
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import (
        NeuromodulatorConfig,
        NeuromodulatorManager,
        ProductionRule,
    )

    nm = NeuromodulatorConfig(
        name="acetylcholine",
        baseline=1.0,
        decay_tau_ms=10000.0,  # very slow decay so production dominates
        concentration_min=0.0,
        concentration_max=2.0,
        production_rules=[
            ProductionRule(rule_type="pause_on_reward", sensitivity=-100.0, threshold=0.0),
        ],
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    bridge = _FakeBridge()

    bridge.core_config.current_reward_signal = 5.0
    for _ in range(50):
        mgr.step(bridge)
    # Even with absurd negative drive, must clip at concentration_min.
    assert mgr.get_concentration("acetylcholine") >= 0.0


def test_pause_on_reward_ignores_missing_bridge_config():
    """When bridge has no core_config (or bridge=None), the rule no-ops
    rather than crashing - same forward-compatibility pattern as
    from_reward."""
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import (
        NeuromodulatorConfig,
        NeuromodulatorManager,
        ProductionRule,
    )

    nm = NeuromodulatorConfig(
        name="acetylcholine",
        baseline=1.0,
        decay_tau_ms=500.0,
        concentration_min=0.0,
        concentration_max=2.0,
        production_rules=[
            ProductionRule(rule_type="pause_on_reward", sensitivity=-1.0, threshold=0.0),
        ],
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    # bridge=None -> rule should silently produce 0, decay-only dynamics
    # leave concentration at baseline.
    for _ in range(100):
        mgr.step(bridge=None)
    assert abs(mgr.get_concentration("acetylcholine") - 1.0) < 1e-6
