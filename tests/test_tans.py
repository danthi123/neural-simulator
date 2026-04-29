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


def test_default_acetylcholine_tan_config_helper():
    """The runner should not need to inline the ACh config; a helper
    returns a sensible default that we can register directly.

    Renamed 2026-04-29 (Wave-1 #10): the helper is now
    `_default_acetylcholine_tan_config()` and the modulator name is
    `"acetylcholine_tan"` to specify the source population (striatal TANs;
    distinct from basal-forebrain or brainstem ACh sources)."""
    from sim.neuromodulators import (
        NeuromodulatorConfig,
        _default_acetylcholine_tan_config,
    )

    cfg = _default_acetylcholine_tan_config()
    assert isinstance(cfg, NeuromodulatorConfig)
    assert cfg.name == "acetylcholine_tan"
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
    assert abs(pos - neg) < 1e-6, (
        f"Pause should be bit-symmetric in |reward|: pos={pos:.6f} neg={neg:.6f}"
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


# ---------- Task 2: plasticity_window_gate target -- manager-level ----------


def test_plasticity_window_gate_blocked_at_baseline_ach():
    """ACh at tonic baseline (concentration == baseline) should yield gate ~ 0
    (plasticity blocked while ACh is firing tonically)."""
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import (
        ModulatorTarget,
        NeuromodulatorConfig,
        NeuromodulatorManager,
    )

    nm = NeuromodulatorConfig(
        name="acetylcholine",
        baseline=1.0,
        decay_tau_ms=500.0,
        concentration_min=0.0,
        concentration_max=2.0,
        targets=[ModulatorTarget(target_type="plasticity_window_gate", scope="all")],
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    mgr.set_concentration("acetylcholine", 1.0)
    gate = mgr.compute_plasticity_window_gate_multiplier()
    assert abs(gate) < 1e-6, f"Expected gate ~ 0 at baseline ACh, got {gate}"


def test_plasticity_window_gate_permitted_at_paused_ach():
    """ACh at concentration=0 (full pause) should yield gate = 1.0 (full
    plasticity)."""
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import (
        ModulatorTarget,
        NeuromodulatorConfig,
        NeuromodulatorManager,
    )

    nm = NeuromodulatorConfig(
        name="acetylcholine",
        baseline=1.0,
        decay_tau_ms=500.0,
        concentration_min=0.0,
        concentration_max=2.0,
        targets=[ModulatorTarget(target_type="plasticity_window_gate", scope="all")],
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    mgr.set_concentration("acetylcholine", 0.0)
    gate = mgr.compute_plasticity_window_gate_multiplier()
    assert abs(gate - 1.0) < 1e-6, f"Expected gate = 1.0 at fully-paused ACh, got {gate}"


def test_plasticity_window_gate_clipped_below_zero():
    """ACh above baseline (overshoot) should clip gate at 0, not go negative."""
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import (
        ModulatorTarget,
        NeuromodulatorConfig,
        NeuromodulatorManager,
    )

    nm = NeuromodulatorConfig(
        name="acetylcholine",
        baseline=1.0,
        decay_tau_ms=500.0,
        concentration_min=0.0,
        concentration_max=2.0,
        targets=[ModulatorTarget(target_type="plasticity_window_gate", scope="all")],
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    mgr.set_concentration("acetylcholine", 1.5)  # above baseline=1.0
    gate = mgr.compute_plasticity_window_gate_multiplier()
    assert gate == 0.0, f"Expected gate = 0 (clipped) at over-baseline ACh, got {gate}"


def test_plasticity_window_gate_aggregates_multiplicatively():
    """Multiple modulators with plasticity_window_gate targets should
    combine multiplicatively (matches compute_synaptic_gain_multiplier
    aggregation)."""
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import (
        ModulatorTarget,
        NeuromodulatorConfig,
        NeuromodulatorManager,
    )

    # First modulator: baseline=1.0, conc=0.5 -> gate = 1 - 0.5/1.0 = 0.5
    nm_a = NeuromodulatorConfig(
        name="ach_a",
        baseline=1.0,
        decay_tau_ms=500.0,
        concentration_min=0.0,
        concentration_max=2.0,
        targets=[ModulatorTarget(target_type="plasticity_window_gate", scope="all")],
    )
    # Second modulator: baseline=1.0, conc=0.2 -> gate = 1 - 0.2/1.0 = 0.8
    nm_b = NeuromodulatorConfig(
        name="ach_b",
        baseline=1.0,
        decay_tau_ms=500.0,
        concentration_min=0.0,
        concentration_max=2.0,
        targets=[ModulatorTarget(target_type="plasticity_window_gate", scope="all")],
    )
    mgr = NeuromodulatorManager([nm_a, nm_b], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    mgr.set_concentration("ach_a", 0.5)
    mgr.set_concentration("ach_b", 0.2)
    gate = mgr.compute_plasticity_window_gate_multiplier()
    # Expected: 0.5 * 0.8 = 0.4
    assert abs(gate - 0.4) < 1e-6, (
        f"Expected combined gate = 0.4 (0.5 * 0.8), got {gate}"
    )


def test_plasticity_window_gate_returns_one_when_no_targets():
    """Manager with no plasticity_window_gate targets returns 1.0 (no-op)."""
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import NeuromodulatorManager

    mgr = NeuromodulatorManager([], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    # Empty manager (no modulators at all) -> default 1.0.
    assert mgr.compute_plasticity_window_gate_multiplier() == 1.0


# ---------- Task 2: plasticity_window_gate -- bridge integration ----------


def _make_bridge_with_ach(set_ach_concentration: float):
    """Build a small bridge with the ACh subsystem on for integration tests.

    Mirrors tests/test_neuromodulators.py::_make_bridge but pre-registers
    the default ACh config. Returns (sb, cfg).
    """
    pytest.importorskip("cupy")
    from sim import (
        SimulationBridge,
        CoreSimConfig,
        VisualizationConfig,
        RuntimeState,
        GPUConfig,
    )
    from sim.enums import NeuronModel
    from sim.neuromodulators import _default_acetylcholine_config

    cfg = CoreSimConfig()
    cfg.num_neurons = 50
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = 42
    cfg.enable_neuromodulator_subsystem = True
    cfg.enable_stdp = True
    cfg.enable_reward_modulation = True
    cfg.reward_learning_rate = 0.05
    cfg.neuromodulators = [_default_acetylcholine_config()]

    sb = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    # Override the just-initialized concentration to whatever the test wants.
    sb.neuromodulator_manager.set_concentration("acetylcholine", set_ach_concentration)
    return sb, cfg


def test_bridge_blocks_reward_weight_updates_when_ach_at_baseline():
    """With ACh at tonic baseline (gate=0), reward-modulated weight updates
    should be ZERO. With ACh paused (gate~1), the same setup should produce
    a measurable weight delta.

    Semantic note (post-2026-04-28-bugfix): the bridge step now runs
    `manager.step()` BEFORE reward modulation, so the same step's reward
    signal can drive the same step's NM concentration changes. This means
    a config with `pause_on_reward` would have its ACh dragged DOWN during
    the test step, opening the gate even when ACh started at baseline.
    To isolate the gate mechanic itself (independent of the production
    rule), this test uses an ACh config WITHOUT production rules and
    pokes the concentration directly via set_concentration. The
    pause_on_reward dynamics are exercised by the dedicated unit tests
    above (test_pause_on_reward_*).
    """
    pytest.importorskip("cupy")
    import cupy as cp

    from sim import (
        SimulationBridge,
        CoreSimConfig,
        VisualizationConfig,
        RuntimeState,
        GPUConfig,
    )
    from sim.enums import NeuronModel
    from sim.neuromodulators import (
        ModulatorTarget,
        NeuromodulatorConfig,
    )

    def _delta_with_ach(ach_conc: float) -> float:
        # Inline a minimal ACh config: same gate target but NO production
        # rules. This pins ACh at whatever set_concentration assigns,
        # because manager.step's only effect is decay toward baseline +
        # production (which is zero here). Over one step, decay is
        # negligible (dt=1ms vs decay_tau=500ms -> ~0.2% drift).
        nm_no_production = NeuromodulatorConfig(
            name="acetylcholine",
            baseline=1.0,
            decay_tau_ms=500.0,
            concentration_min=0.0,
            concentration_max=2.0,
            targets=[
                ModulatorTarget(target_type="plasticity_window_gate", scope="all")
            ],
            production_rules=[],  # KEY: no pause_on_reward, gate driven by set_concentration only
        )
        cfg = CoreSimConfig()
        cfg.num_neurons = 50
        cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
        cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
        cfg.dt_ms = 1.0
        cfg.seed = 42
        cfg.enable_neuromodulator_subsystem = True
        cfg.enable_stdp = True
        cfg.enable_reward_modulation = True
        cfg.reward_learning_rate = 0.05
        cfg.neuromodulators = [nm_no_production]

        sb = SimulationBridge(
            core_config=cfg,
            viz_config=VisualizationConfig(),
            runtime_state=RuntimeState(),
            gpu_config=GPUConfig(),
        )
        sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        sb._initialize_simulation_data(called_from_playback_init=False)

        # Set wide weight bounds so the post-update clip doesn't mask the
        # small reward-driven delta we're testing. Same trick B.1 uses.
        sb.core_config.stdp_w_min = -10.0
        sb.core_config.stdp_w_max = 100.0
        sb.core_config.hebbian_min_weight = -10.0
        sb.core_config.hebbian_max_weight = 100.0
        if sb.cp_eligibility_trace is None:
            sb.clear_simulation_state_and_gpu_memory()
            pytest.skip("eligibility trace not allocated")
        actual_nnz = int(sb.cp_connections.nnz)
        # Set up: uniform positive eligibility, no STDP / Hebbian / structural
        # writes to mask the small reward-driven delta.
        sb.core_config.enable_stdp = False
        sb.core_config.enable_hebbian_learning = False
        sb.core_config.enable_homeostasis = False
        sb.core_config.enable_structural_plasticity = False
        sb.core_config.enable_synaptic_scaling = False
        sb.cp_eligibility_trace[:] = 0.0
        sb.cp_eligibility_trace[:actual_nnz] = 0.5
        w_before = sb.cp_connections.data[:actual_nnz].copy()
        # Apply reward AFTER eligibility set-up. Pin ACh to the test value.
        # Without pause_on_reward, set_concentration is the sole driver
        # of the ACh state visited by the reward path this step.
        sb.neuromodulator_manager.set_concentration("acetylcholine", ach_conc)
        sb.core_config.current_reward_signal = 1.0
        sb.core_config.reward_baseline = 0.0
        sb._run_one_simulation_step()
        sb.runtime_state.current_time_step += 1
        w_after = sb.cp_connections.data[:actual_nnz]
        # Sum of absolute weight deltas — robust to per-synapse sign.
        delta = float(cp.sum(cp.abs(w_after - w_before)).get())
        sb.clear_simulation_state_and_gpu_memory()
        return delta

    blocked = _delta_with_ach(1.0)   # at baseline -> gate = 0
    permitted = _delta_with_ach(0.0)  # full pause -> gate = 1

    # When ACh is at baseline, gate=0 should suppress all weight changes.
    assert blocked < 1e-6, (
        f"Expected ZERO weight delta with ACh at baseline (gate=0), got {blocked:.6e}"
    )
    # When ACh is paused, gate~1 should permit normal updates.
    assert permitted > 1e-4, (
        f"Expected nontrivial weight delta with ACh paused (gate=1), got {permitted:.6e}"
    )


# ---------- Regression: single-pulse reward must fire plasticity within step ----------


def test_single_pulse_reward_fires_plasticity_within_step():
    """Regression test for the 2026-04-28 step-order bug.

    Setup: full default ACh config (with pause_on_reward production rule).
    ACh starts at tonic baseline = 1.0 (gate = 0). A SINGLE-step reward
    pulse arrives. The TAN gate must open within that same step so the
    reward-modulated weight update is permitted.

    Pre-fix behavior: `manager.step()` ran AFTER the reward block, so
    the gate was read from the previous step's ACh (still at baseline,
    gate = 0) -> weight_updates *= 0. The reward never produced any
    learning. Then the next step's gate was open, but the reward was
    already gone. Single-pulse rewards never trained the network.

    Post-fix behavior: `manager.step()` runs BEFORE the reward block, so
    the same step's reward drops ACh, opens the gate, and the reward
    update goes through.

    This test would have caught the empirical regression
    (1800-step `--enable-tans` run with full Cluster B + multi-goal:
    sum=19.76 vs baseline 9.50) before the runner ever shipped.
    """
    pytest.importorskip("cupy")
    import cupy as cp

    sb, cfg = _make_bridge_with_ach(set_ach_concentration=1.0)
    # Wide weight bounds so the clip doesn't hide the reward-driven delta.
    sb.core_config.stdp_w_min = -10.0
    sb.core_config.stdp_w_max = 100.0
    sb.core_config.hebbian_min_weight = -10.0
    sb.core_config.hebbian_max_weight = 100.0
    if sb.cp_eligibility_trace is None:
        sb.clear_simulation_state_and_gpu_memory()
        pytest.skip("eligibility trace not allocated")
    actual_nnz = int(sb.cp_connections.nnz)
    # Disable other plasticity to isolate the reward path.
    sb.core_config.enable_stdp = False
    sb.core_config.enable_hebbian_learning = False
    sb.core_config.enable_homeostasis = False
    sb.core_config.enable_structural_plasticity = False
    sb.core_config.enable_synaptic_scaling = False
    # Uniform positive eligibility -> a non-zero reward should produce a
    # measurable weight delta IFF the gate opens.
    sb.cp_eligibility_trace[:] = 0.0
    sb.cp_eligibility_trace[:actual_nnz] = 0.5

    # Snapshot weights, ensure ACh is exactly at baseline (gate = 0 a priori),
    # then deliver one reward pulse.
    sb.neuromodulator_manager.set_concentration("acetylcholine", 1.0)
    w_before = sb.cp_connections.data[:actual_nnz].copy()
    sb.core_config.current_reward_signal = 1.0
    sb.core_config.reward_baseline = 0.0
    sb._run_one_simulation_step()
    sb.runtime_state.current_time_step += 1
    w_after = sb.cp_connections.data[:actual_nnz]
    delta = float(cp.sum(cp.abs(w_after - w_before)).get())

    # ACh must have been pulled below baseline by pause_on_reward in the
    # SAME step (i.e., before the reward block), opening the gate.
    ach_after = sb.neuromodulator_manager.get_concentration("acetylcholine")
    sb.clear_simulation_state_and_gpu_memory()

    # Pre-fix invariant violated: delta would have been ~0 here.
    assert delta > 1e-4, (
        f"Single-pulse reward did NOT open the TAN gate within the same "
        f"step (regression of the 2026-04-28 step-order bug). "
        f"delta={delta:.6e}, ACh after step={ach_after:.4f}. "
        f"Expected delta > 1e-4 indicating the gate opened and reward-driven "
        f"weight updates flowed through."
    )
    # Sanity: ACh should be measurably below tonic baseline 1.0.
    assert ach_after < 0.99, (
        f"pause_on_reward should have dragged ACh below baseline within "
        f"the step; got ACh={ach_after:.4f}"
    )
