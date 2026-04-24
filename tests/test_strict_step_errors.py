"""Test the opt-in strict-step-errors mode on SimulationBridge.

Normally `_run_one_simulation_step` catches every exception, logs CRITICAL,
and continues. That design is load-bearing for the biological-experiment
UI — a single bad step shouldn't kill the whole session. But for research
runners it silently masks state-restore bugs: the G3 checkpoint regressed
test accuracy to chance for ~1 hour last night because two missing cached
attributes were caught and swallowed step after step.

Fix: add `bridge.strict_step_errors`. Default False (back-compat). When
True, re-raise so callers see the exception.
"""
import pytest


def test_strict_step_errors_reraises(tmp_path):
    pytest.importorskip("cupy")
    import cupy as cp

    from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
    from sim.config import CoreSimConfig
    from sim.enums import NeuronModel

    cfg = CoreSimConfig()
    cfg.num_neurons = 4
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.seed = 1
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.enable_stdp = False
    cfg.enable_watts_strogatz = False
    cfg.connections_per_neuron = 0

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)
    assert bridge.is_initialized

    # Opt-in strict mode.
    bridge.strict_step_errors = True

    # Force a failure: delete a required cached attribute.
    del bridge._cached_decay_e

    # Should raise rather than swallow.
    with pytest.raises(AttributeError, match="_cached_decay_e"):
        bridge._run_one_simulation_step()


def test_strict_step_errors_default_false_swallows():
    """Back-compat: without setting the flag, the bridge swallows as before.

    This is load-bearing for the biological experiment UI, which relies on
    single-step failures not bringing down the session.
    """
    pytest.importorskip("cupy")
    import cupy as cp

    from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
    from sim.config import CoreSimConfig
    from sim.enums import NeuronModel

    cfg = CoreSimConfig()
    cfg.num_neurons = 4
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.seed = 2
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.enable_stdp = False
    cfg.enable_watts_strogatz = False
    cfg.connections_per_neuron = 0

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)

    # Default: strict mode not set.
    assert getattr(bridge, "strict_step_errors", False) is False

    # Force a failure. Should NOT raise — caught internally.
    del bridge._cached_decay_e
    bridge._run_one_simulation_step()  # returns without raising

    # is_running flipped off as a side-effect of stop_simulation().
    # (Research runners can watch for this as a signal.)
    assert bridge.runtime_state.is_running is False
