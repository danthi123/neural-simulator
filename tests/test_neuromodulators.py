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


# ---------- Task 5b: from_surprise rule (NE-like RPE phasic firing) ----------


def test_from_surprise_silent_after_long_constant_reward():
    """After long convergence to constant reward, NE drops back to baseline.

    Uses threshold > 0 so small EMA-lag-induced RPE doesn't keep firing.
    """
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import (
        NeuromodulatorConfig,
        NeuromodulatorManager,
        ProductionRule,
    )

    nm = NeuromodulatorConfig(
        name="ne",
        baseline=0.0,
        decay_tau_ms=200.0,  # fast-decaying NE
        production_rules=[
            ProductionRule(rule_type="from_surprise", sensitivity=1.0,
                            threshold=0.3, window_ms=300.0)  # threshold > expected RPE-during-convergence
        ],
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    bridge = _FakeBridge()

    bridge.core_config.current_reward_signal = 0.5
    # Long enough for EMA to converge AND NE to decay
    for _ in range(5000):
        mgr.step(bridge)
    # After convergence, RPE small (below threshold), NE decays to baseline
    assert mgr.get_concentration("ne") < 0.05, (
        f"NE should decay after convergence, got {mgr.get_concentration('ne'):.3f}"
    )


def test_from_surprise_fires_on_step_change():
    """Sudden change in reward produces a phasic NE pulse."""
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import (
        NeuromodulatorConfig,
        NeuromodulatorManager,
        ProductionRule,
    )

    nm = NeuromodulatorConfig(
        name="ne",
        baseline=0.0,
        decay_tau_ms=200.0,
        concentration_max=10.0,  # allow high pulses
        production_rules=[
            ProductionRule(rule_type="from_surprise", sensitivity=2.0,
                            threshold=0.1, window_ms=500.0)
        ],
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    bridge = _FakeBridge()

    # Phase 1: stable reward = +0.5, EMA settles
    bridge.core_config.current_reward_signal = 0.5
    for _ in range(2000):
        mgr.step(bridge)
    pre_change = mgr.get_concentration("ne")

    # Phase 2: sudden flip to reward = -0.5 (big RPE)
    bridge.core_config.current_reward_signal = -0.5
    peak = pre_change
    for _ in range(50):
        mgr.step(bridge)
        peak = max(peak, mgr.get_concentration("ne"))

    # NE should spike on the change
    assert peak > pre_change + 0.5, (
        f"NE pulse too small: pre={pre_change:.3f}, peak={peak:.3f}"
    )


def test_from_surprise_threshold_silences_small_rpe():
    """Below-threshold RPE produces no NE."""
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import (
        NeuromodulatorConfig,
        NeuromodulatorManager,
        ProductionRule,
    )

    nm = NeuromodulatorConfig(
        name="ne",
        baseline=0.0,
        decay_tau_ms=300.0,
        production_rules=[
            ProductionRule(rule_type="from_surprise", sensitivity=2.0,
                            threshold=0.5, window_ms=500.0)
        ],
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    bridge = _FakeBridge()

    # Small fluctuations — RPE always < threshold
    for i in range(2000):
        bridge.core_config.current_reward_signal = 0.1 if i % 2 == 0 else 0.2
        mgr.step(bridge)

    # NE should stay near baseline since RPE never crosses threshold
    assert mgr.get_concentration("ne") < 0.1


# ---------- Task 6: synaptic_gain target multiplier ----------


def test_synaptic_gain_multiplier_when_concentration_at_baseline():
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import (
        ModulatorTarget,
        NeuromodulatorConfig,
        NeuromodulatorManager,
    )

    nm = NeuromodulatorConfig(
        name="da",
        baseline=0.0,
        targets=[ModulatorTarget(target_type="synaptic_gain", scope="all", sensitivity=1.0)],
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    # Concentration at baseline -> multiplier == 1.0
    assert mgr.compute_synaptic_gain_multiplier() == 1.0


def test_synaptic_gain_multiplier_with_concentration_above_baseline():
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import (
        ModulatorTarget,
        NeuromodulatorConfig,
        NeuromodulatorManager,
    )

    nm = NeuromodulatorConfig(
        name="da",
        baseline=0.0,
        targets=[ModulatorTarget(target_type="synaptic_gain", scope="all", sensitivity=0.5)],
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    mgr.set_concentration("da", 1.0)
    # 1.0 + 0.5 * (1.0 - 0.0) = 1.5
    assert abs(mgr.compute_synaptic_gain_multiplier() - 1.5) < 1e-6


def test_synaptic_gain_returns_one_when_no_targets():
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import NeuromodulatorManager

    mgr = NeuromodulatorManager([], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    assert mgr.compute_synaptic_gain_multiplier() == 1.0


def test_synaptic_gain_multiplier_clamped_at_zero():
    """Negative concentration shouldn't make multiplier negative (transmission
    can't be reversed). It should clamp at 0.
    """
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import (
        ModulatorTarget,
        NeuromodulatorConfig,
        NeuromodulatorManager,
    )

    nm = NeuromodulatorConfig(
        name="da",
        baseline=1.0,
        concentration_min=-2.0,  # allow negatives just for this test
        targets=[ModulatorTarget(target_type="synaptic_gain", scope="all", sensitivity=2.0)],
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    mgr.set_concentration("da", -2.0)
    # 1 + 2 * (-2 - 1) = 1 - 6 = -5 -> clamped to 0
    assert mgr.compute_synaptic_gain_multiplier() == 0.0


# ---------- Task 7: plasticity_rate target multiplier ----------


def test_plasticity_rate_multiplier_with_concentration():
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import (
        ModulatorTarget,
        NeuromodulatorConfig,
        NeuromodulatorManager,
    )

    nm = NeuromodulatorConfig(
        name="da",
        baseline=0.0,
        targets=[ModulatorTarget(target_type="plasticity_rate", scope="all", sensitivity=2.0)],
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    mgr.set_concentration("da", 0.5)
    # 1 + 2 * 0.5 = 2.0
    assert abs(mgr.compute_plasticity_rate_multiplier() - 2.0) < 1e-6


def test_plasticity_rate_multiplier_default_is_one():
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import NeuromodulatorManager

    mgr = NeuromodulatorManager([], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    assert mgr.compute_plasticity_rate_multiplier() == 1.0


# ---------- Task 8: excitability_drive target ----------


def test_excitability_drive_scalar_scope_all():
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import (
        ModulatorTarget,
        NeuromodulatorConfig,
        NeuromodulatorManager,
    )

    nm = NeuromodulatorConfig(
        name="ne",
        baseline=0.1,
        targets=[ModulatorTarget(target_type="excitability_drive", scope="all", sensitivity=50.0)],
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    mgr.set_concentration("ne", 0.4)
    # 50 * (0.4 - 0.1) = 15 pA
    assert abs(mgr.compute_excitability_drive_pA() - 15.0) < 1e-6


def test_excitability_drive_per_neuron_with_trait_scope():
    pytest.importorskip("cupy")
    import cupy as cp
    import numpy as np

    from sim.neuromodulators import (
        ModulatorTarget,
        NeuromodulatorConfig,
        NeuromodulatorManager,
    )

    nm = NeuromodulatorConfig(
        name="ne",
        baseline=0.0,
        targets=[ModulatorTarget(target_type="excitability_drive", scope="trait:1", sensitivity=10.0)],
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=4, cp_module=cp)
    mgr.set_concentration("ne", 1.0)

    traits = cp.asarray([0, 1, 1, 0], dtype=cp.int32)
    drive = mgr.compute_excitability_drive_per_neuron(cp_traits=traits)
    drive_np = cp.asnumpy(drive)
    # Effect = 10.0 * (1.0 - 0.0) = 10.0 on neurons with trait==1.
    expected = np.array([0.0, 10.0, 10.0, 0.0], dtype=np.float32)
    assert np.allclose(drive_np, expected, atol=1e-5)


def test_excitability_drive_per_neuron_with_group_scope():
    pytest.importorskip("cupy")
    import cupy as cp
    import numpy as np

    from sim.neuromodulators import (
        ModulatorTarget,
        NeuromodulatorConfig,
        NeuromodulatorManager,
    )

    nm = NeuromodulatorConfig(
        name="ne",
        baseline=0.0,
        targets=[ModulatorTarget(target_type="excitability_drive", scope="group:motor", sensitivity=20.0)],
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=6, cp_module=cp)
    mgr.set_group_indices({"motor": [3, 5]})
    mgr.set_concentration("ne", 0.5)

    drive = mgr.compute_excitability_drive_per_neuron()
    drive_np = cp.asnumpy(drive)
    # 20 * 0.5 = 10 on neurons 3 and 5
    expected = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 10.0], dtype=np.float32)
    assert np.allclose(drive_np, expected, atol=1e-5)


def test_excitability_drive_per_neuron_returns_none_if_no_per_neuron_targets():
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import (
        ModulatorTarget,
        NeuromodulatorConfig,
        NeuromodulatorManager,
    )

    nm = NeuromodulatorConfig(
        name="ne",
        baseline=0.0,
        targets=[ModulatorTarget(target_type="excitability_drive", scope="all", sensitivity=10.0)],
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=4, cp_module=cp)
    mgr.set_concentration("ne", 1.0)
    # Only scope=all targets, no per-neuron arrays needed
    assert mgr.compute_excitability_drive_per_neuron() is None


# ---------- Task 9: bridge config flag + allocation ----------


def _make_bridge(extra_cfg_overrides: dict = None):
    """Helper: minimal bridge for integration tests. Returns (sb, cfg)."""
    pytest.importorskip("cupy")
    from sim import (
        SimulationBridge,
        CoreSimConfig,
        VisualizationConfig,
        RuntimeState,
        GPUConfig,
    )
    from sim.enums import NeuronModel

    cfg = CoreSimConfig()
    cfg.num_neurons = 50
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = 42
    if extra_cfg_overrides:
        for k, v in extra_cfg_overrides.items():
            setattr(cfg, k, v)

    sb = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    return sb, cfg


def test_bridge_no_manager_when_subsystem_disabled():
    """Default config (subsystem off) -> bridge.neuromodulator_manager is None."""
    pytest.importorskip("cupy")
    sb, cfg = _make_bridge()
    assert cfg.enable_neuromodulator_subsystem is False
    assert sb.neuromodulator_manager is None
    sb.clear_simulation_state_and_gpu_memory()


def test_bridge_allocates_manager_when_subsystem_enabled():
    """With enable_neuromodulator_subsystem=True and a non-empty list,
    bridge allocates a NeuromodulatorManager and exposes it via attr."""
    pytest.importorskip("cupy")
    from sim.neuromodulators import NeuromodulatorConfig

    sb, cfg = _make_bridge({
        "enable_neuromodulator_subsystem": True,
        "neuromodulators": [NeuromodulatorConfig(name="dopamine", baseline=0.0)],
    })
    assert sb.neuromodulator_manager is not None
    assert sb.neuromodulator_manager.get_concentration("dopamine") == 0.0
    sb.clear_simulation_state_and_gpu_memory()


def test_bridge_no_manager_when_subsystem_enabled_but_empty_list():
    """If enable_neuromodulator_subsystem is True but no modulators are
    configured, manager stays None (no point allocating an empty manager)."""
    pytest.importorskip("cupy")

    sb, cfg = _make_bridge({
        "enable_neuromodulator_subsystem": True,
        "neuromodulators": [],
    })
    assert sb.neuromodulator_manager is None
    sb.clear_simulation_state_and_gpu_memory()


# ---------- Task 10: bridge step integration ----------


def test_bridge_step_advances_concentration_via_from_reward():
    """A bridge running with subsystem on + dopamine + from_reward rule
    + a positive reward signal should see DA concentration rise after
    one simulation step."""
    pytest.importorskip("cupy")
    from sim.neuromodulators import NeuromodulatorConfig, ProductionRule

    sb, cfg = _make_bridge({
        "enable_neuromodulator_subsystem": True,
        "enable_reward_modulation": True,
        "neuromodulators": [
            NeuromodulatorConfig(
                name="dopamine",
                baseline=0.0,
                decay_tau_ms=500.0,
                production_rules=[ProductionRule(rule_type="from_reward", sensitivity=1.0)],
            )
        ],
    })
    assert sb.neuromodulator_manager is not None
    assert sb.neuromodulator_manager.get_concentration("dopamine") == 0.0

    sb.core_config.current_reward_signal = 1.0
    sb._run_one_simulation_step()
    sb.runtime_state.current_time_step += 1

    # After one step: 1.0 added by from_reward, exp(-1/500) decay first ≈ 0.998
    # Should be ~1.0 (clipped at concentration_max default 5.0)
    assert sb.neuromodulator_manager.get_concentration("dopamine") > 0.5
    sb.clear_simulation_state_and_gpu_memory()


def test_bridge_step_no_concentration_change_when_subsystem_off():
    """Subsystem off -> manager.step() never called -> no state change."""
    pytest.importorskip("cupy")

    sb, cfg = _make_bridge()
    assert sb.neuromodulator_manager is None  # default off
    # Just running steps with subsystem off should be a no-op for
    # neuromodulator state. Call a step to make sure nothing crashes.
    sb._run_one_simulation_step()
    sb.runtime_state.current_time_step += 1
    assert sb.neuromodulator_manager is None
    sb.clear_simulation_state_and_gpu_memory()


# ---------- Task 11: synaptic_gain wired into bridge ----------


def test_synaptic_gain_modulates_conductance_via_bridge():
    """When dopamine has a synaptic_gain target with high concentration,
    propagated conductance should be roughly proportional to (1 + sens*conc).
    """
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import (
        ModulatorTarget, NeuromodulatorConfig, ProductionRule,
    )

    def _g_e_after_pulse(set_dopamine: float) -> float:
        sb, cfg = _make_bridge({
            "enable_neuromodulator_subsystem": True,
            "enable_short_term_plasticity": False,
            "enable_ou_process": False,
            "neuromodulators": [
                NeuromodulatorConfig(
                    name="dopamine",
                    baseline=0.0,
                    decay_tau_ms=1e9,  # effectively no decay during test
                    targets=[ModulatorTarget(target_type="synaptic_gain",
                                              scope="all", sensitivity=1.0)],
                )
            ],
        })
        sb.neuromodulator_manager.set_concentration("dopamine", set_dopamine)
        # Force a synthetic prior firing pattern
        sb.cp_prev_firing_states[:] = False
        sb.cp_prev_firing_states[0] = True
        # Reset conductance baseline
        sb.cp_conductance_g_e[:] = 0.0
        sb._run_one_simulation_step()
        g_total = float(cp.sum(sb.cp_conductance_g_e).get())
        sb.clear_simulation_state_and_gpu_memory()
        return g_total

    g0 = _g_e_after_pulse(0.0)  # multiplier 1.0
    g1 = _g_e_after_pulse(1.0)  # multiplier 1 + 1*1 = 2.0

    # With dopamine=1.0 (multiplier 2.0), conductance should be ~2x.
    # Allow up to 30% slack for any non-modulated baseline drift.
    if g0 < 1e-6:
        # If neuron 0 has no outgoing E synapses (sparse network) skip
        pytest.skip("synthetic pulse produced no excitatory conductance to test")
    ratio = g1 / max(g0, 1e-9)
    assert ratio > 1.5, (
        f"Expected ~2x conductance with dopamine, got ratio {ratio:.3f} "
        f"(g0={g0:.4f}, g1={g1:.4f})"
    )


# ---------- Task 12: plasticity_rate wired into bridge ----------


def test_plasticity_rate_target_scales_reward_weight_updates():
    """When dopamine has a plasticity_rate target with concentration > 0,
    reward-modulated weight updates should scale by (1 + sens*conc)."""
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import (
        ModulatorTarget, NeuromodulatorConfig,
    )

    def _weight_delta_after_reward(plasticity_mult_concentration: float) -> float:
        sb, cfg = _make_bridge({
            "enable_stdp": True,
            "enable_reward_modulation": True,
            "enable_neuromodulator_subsystem": True,
            "reward_learning_rate": 0.05,
            "neuromodulators": [
                NeuromodulatorConfig(
                    name="dopamine",
                    baseline=0.0,
                    decay_tau_ms=1e9,  # effectively no decay during test
                    targets=[ModulatorTarget(target_type="plasticity_rate",
                                              scope="all", sensitivity=2.0)],
                )
            ],
        })
        sb.neuromodulator_manager.set_concentration(
            "dopamine", plasticity_mult_concentration,
        )
        # Synthetic eligibility trace
        if sb.cp_eligibility_trace is None:
            sb.clear_simulation_state_and_gpu_memory()
            pytest.skip("eligibility trace not allocated")
        sb.cp_eligibility_trace[:] = 0.0
        actual_nnz = int(sb.cp_connections.nnz)
        sb.cp_eligibility_trace[:actual_nnz] = 0.5  # constant non-zero
        w_before = float(cp.mean(sb.cp_connections.data[:actual_nnz]).get())
        sb.core_config.current_reward_signal = 1.0
        sb._run_one_simulation_step()
        sb.runtime_state.current_time_step += 1
        w_after = float(cp.mean(sb.cp_connections.data[:actual_nnz]).get())
        sb.clear_simulation_state_and_gpu_memory()
        return w_after - w_before

    delta_no_da = _weight_delta_after_reward(0.0)  # multiplier 1.0
    delta_da_05 = _weight_delta_after_reward(0.5)  # multiplier 1 + 2*0.5 = 2.0

    if abs(delta_no_da) < 1e-9:
        pytest.skip("baseline reward update too small to measure")
    ratio = delta_da_05 / delta_no_da
    assert 1.7 < ratio < 2.3, (
        f"Expected ~2x reward-update with DA plasticity_rate, got ratio "
        f"{ratio:.3f} (no_da={delta_no_da:.6f} da_05={delta_da_05:.6f})"
    )


# ---------- Task 13: excitability_drive wired into bridge ----------


def test_excitability_drive_increases_firing_rate():
    """A high-NE config with excitability_drive (scope=all) should produce
    measurably more spikes than the same config with NE at baseline."""
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.neuromodulators import (
        ModulatorTarget, NeuromodulatorConfig,
    )

    def _spike_count(ne_concentration: float, n_steps: int = 200) -> int:
        sb, cfg = _make_bridge({
            "num_neurons": 100,
            "enable_neuromodulator_subsystem": True,
            "neuromodulators": [
                NeuromodulatorConfig(
                    name="ne",
                    baseline=0.0,
                    decay_tau_ms=1e9,
                    targets=[ModulatorTarget(target_type="excitability_drive",
                                              scope="all", sensitivity=200.0)],
                )
            ],
        })
        sb.neuromodulator_manager.set_concentration("ne", ne_concentration)
        total = 0
        for _ in range(n_steps):
            sb._run_one_simulation_step()
            sb.runtime_state.current_time_step += 1
            total += int(cp.sum(sb.cp_firing_states).get())
        sb.clear_simulation_state_and_gpu_memory()
        return total

    n_low = _spike_count(0.0)
    n_high = _spike_count(1.0)  # +200 pA on every neuron
    # Heavy current injection should produce many more spikes
    assert n_high > n_low + 100, (
        f"Excitability drive ineffective: low={n_low}, high={n_high}"
    )


def test_excitability_drive_per_neuron_scope_trait():
    """trait:1 (inhibitory) excitability_drive boost should preferentially
    raise inhibitory firing rate."""
    pytest.importorskip("cupy")
    import cupy as cp
    import numpy as np

    from sim.neuromodulators import (
        ModulatorTarget, NeuromodulatorConfig,
    )

    sb, cfg = _make_bridge({
        "num_neurons": 200,
        "neural_profile_name": "CORTEX_L23_RS_FS",  # has inhibitory traits
        "enable_neuromodulator_subsystem": True,
        "neuromodulators": [
            NeuromodulatorConfig(
                name="ne",
                baseline=0.0,
                decay_tau_ms=1e9,
                targets=[ModulatorTarget(target_type="excitability_drive",
                                          scope="trait:1", sensitivity=200.0)],
            )
        ],
    })
    if sb.cp_traits is None:
        sb.clear_simulation_state_and_gpu_memory()
        pytest.skip("traits not allocated under this profile")

    sb.neuromodulator_manager.set_concentration("ne", 1.0)

    inh_mask = (sb.cp_traits == 1)
    exc_mask = (sb.cp_traits == 0)
    inh_count = 0
    exc_count = 0
    n_inh_neurons = int(cp.sum(inh_mask).get())
    n_exc_neurons = int(cp.sum(exc_mask).get())
    if n_inh_neurons == 0 or n_exc_neurons == 0:
        sb.clear_simulation_state_and_gpu_memory()
        pytest.skip("no inhibitory or excitatory neurons in this profile")

    for _ in range(200):
        sb._run_one_simulation_step()
        sb.runtime_state.current_time_step += 1
        fired = sb.cp_firing_states
        inh_count += int(cp.sum(fired & inh_mask).get())
        exc_count += int(cp.sum(fired & exc_mask).get())
    sb.clear_simulation_state_and_gpu_memory()

    # Inhibitory firing rate per neuron should be much higher than
    # excitatory under this targeted boost.
    inh_rate = inh_count / max(n_inh_neurons, 1)
    exc_rate = exc_count / max(n_exc_neurons, 1)
    assert inh_rate > 2.0 * exc_rate, (
        f"trait:1 boost should bias toward inhibitory; got inh_rate={inh_rate:.2f} "
        f"exc_rate={exc_rate:.2f}"
    )


# ---------- Task 14: drift regression guard ----------


def test_drift_regression_subsystem_off_unchanged():
    """With subsystem disabled (default), the locked tiny-seeded-sim spike
    count from the drift detector must still be 149 +- 10. Verifies the
    new subsystem code paths are inactive when not enabled.

    Mirrors tests/test_benchmark_drift.test_tiny_seeded_sim_spike_count_in_range
    but asserts explicitly that neuromodulator_manager is None.

    Locked at 149 from commit 5fc92c8 (2026-04-25). Was 170 before that
    commit added 8 IZH2007 BG/thalamus/HC/DA presets, which expanded
    `defined_izh2007_types` and shifted the trait-to-preset modulo math
    in bridge.py:958. See research/findings/2026-04-25-rng-drift-from-izh-presets.md.
    """
    pytest.importorskip("cupy")
    from tests.test_benchmark_drift import _build_tiny_sim, _run_and_count

    sb, cfg = _build_tiny_sim(seed=42)
    assert getattr(cfg, "enable_neuromodulator_subsystem", False) is False
    assert sb.neuromodulator_manager is None
    total, _ = _run_and_count(sb, cfg, n_steps=200)
    sb.clear_simulation_state_and_gpu_memory()
    assert 139 <= total <= 159, (
        f"Drift detected: {total} spikes (expected 149 +- 10)"
    )


# ---------- Task 15: legacy parity ----------


def test_subsystem_with_no_targets_matches_legacy_reward_path():
    """Subsystem ON with a dopamine modulator that has only from_reward
    production but ZERO targets should produce mean-weight outcome
    indistinguishable from subsystem OFF (legacy reward modulation only).

    Why: a no-target modulator just tracks its own concentration but
    has no effect on bridge state. The legacy current_reward_signal path
    runs unchanged. This proves the subsystem doesn't accidentally
    interfere with the legacy code path when no effects are wired.
    """
    pytest.importorskip("cupy")
    import cupy as cp
    from sim.neuromodulators import NeuromodulatorConfig, ProductionRule

    def _final_mean_weight(use_subsystem: bool) -> float:
        sb, cfg = _make_bridge({
            "num_neurons": 100,
            "enable_stdp": True,
            "enable_reward_modulation": True,
            "reward_learning_rate": 0.01,
            **(
                {
                    "enable_neuromodulator_subsystem": True,
                    "neuromodulators": [
                        NeuromodulatorConfig(
                            name="dopamine",
                            baseline=0.0,
                            decay_tau_ms=500.0,
                            production_rules=[ProductionRule(rule_type="from_reward",
                                                              sensitivity=1.0)],
                            targets=[],  # KEY: no targets -> no effect on bridge state
                        )
                    ],
                }
                if use_subsystem
                else {}
            ),
        })
        sb.core_config.current_reward_signal = 0.5
        for _ in range(50):
            sb._run_one_simulation_step()
            sb.runtime_state.current_time_step += 1
        sb.core_config.current_reward_signal = 0.0
        for _ in range(50):
            sb._run_one_simulation_step()
            sb.runtime_state.current_time_step += 1
        actual_nnz = int(sb.cp_connections.nnz)
        w_mean = float(cp.mean(sb.cp_connections.data[:actual_nnz]).get())
        sb.clear_simulation_state_and_gpu_memory()
        return w_mean

    w_legacy = _final_mean_weight(use_subsystem=False)
    w_subsys = _final_mean_weight(use_subsystem=True)
    rel_diff = abs(w_legacy - w_subsys) / max(abs(w_legacy), 1e-9)
    # Both paths should give bit-identical results in principle (subsystem
    # with no targets only steps its own concentration). Allow tiny
    # tolerance for any RNG-state ordering shifts caused by the manager
    # constructor.
    assert rel_diff < 0.05, (
        f"subsystem-with-no-targets diverged from legacy: "
        f"legacy={w_legacy:.6f} subsys={w_subsys:.6f} rel_diff={rel_diff:.4f}"
    )
