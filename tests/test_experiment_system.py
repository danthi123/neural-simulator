"""
Unit tests for the Experiment & Stimulus System.

Tests the experiment system components (StimulusPattern, StimulusChannel,
NeuronGroup, ExperimentConfig, etc.) without requiring GPU/CuPy.
Uses numpy as a mock for CuPy operations.
"""

import sys
import os
import json
import math
import pytest
import numpy as np

# Mock CuPy with NumPy for testing without GPU
class MockCuPy:
    """Minimal CuPy mock using NumPy for CPU-based testing."""
    float32 = np.float32
    int32 = np.int32
    bool_ = np.bool_

    class random:
        @staticmethod
        def random(n):
            return np.random.random(n).astype(np.float32)

        @staticmethod
        def randn(n):
            return np.random.randn(n).astype(np.float32)

    @staticmethod
    def zeros(shape, dtype=np.float32):
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def array(data, dtype=None):
        return np.array(data, dtype=dtype)

    @staticmethod
    def sum(arr):
        class Result:
            def __init__(self, val):
                self.val = val
            def get(self):
                return self.val
        return Result(np.sum(arr))

    @staticmethod
    def mean(arr):
        class Result:
            def __init__(self, val):
                self.val = val
            def get(self):
                return self.val
        return Result(np.mean(arr))

    @staticmethod
    def where(condition, x, y):
        return np.where(condition, x, y)

    @staticmethod
    def maximum(a, b):
        return np.maximum(a, b)

    @staticmethod
    def asarray(data):
        return np.asarray(data)

    @staticmethod
    def asnumpy(data):
        return np.asarray(data)

    @staticmethod
    def is_available():
        return False


# We need to import the experiment system classes from neural-simulator.py
# Since it's a monolithic file, we'll extract and exec the relevant section
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the Enum and dataclass types needed
from enum import Enum
from dataclasses import dataclass, field, asdict, fields
from typing import List, Dict

# Read the experiment system code block directly
experiment_block_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'experiment_system_block.py')
if os.path.exists(experiment_block_path):
    with open(experiment_block_path, 'r') as f:
        experiment_code = f.read()
    # Execute the experiment system code in our namespace
    exec(experiment_code, globals())
else:
    # Fallback: extract from neural-simulator.py
    simulator_path = os.path.join(os.path.dirname(__file__), '..', 'neural-simulator.py')
    with open(simulator_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the experiment system block
    start_marker = "# =============================================================================\n# EXPERIMENT & STIMULUS SYSTEM"
    end_marker = "# --- Simulation Bridge (Core Logic) ---"

    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)

    if start_idx >= 0 and end_idx >= 0:
        experiment_code = content[start_idx:end_idx]
        exec(experiment_code, globals())
    else:
        raise ImportError("Could not find experiment system code in neural-simulator.py")


cp = MockCuPy()


class TestStimulusPatternType:
    """Test stimulus pattern enum definitions."""

    def test_all_pattern_types_defined(self):
        assert hasattr(StimulusPatternType, 'CONSTANT')
        assert hasattr(StimulusPatternType, 'PULSE_TRAIN')
        assert hasattr(StimulusPatternType, 'SINUSOIDAL')
        assert hasattr(StimulusPatternType, 'RAMP')
        assert hasattr(StimulusPatternType, 'POISSON_SPIKE_TRAIN')
        assert hasattr(StimulusPatternType, 'GAUSSIAN_NOISE')
        assert hasattr(StimulusPatternType, 'CUSTOM_WAVEFORM')

    def test_pattern_count(self):
        assert len(StimulusPatternType) == 7


class TestStimulusPattern:
    """Test StimulusPattern dataclass defaults and creation."""

    def test_default_values(self):
        p = StimulusPattern()
        assert p.pattern_type == StimulusPatternType.CONSTANT.name
        assert p.amplitude_pA == 100.0
        assert p.pulse_frequency_hz == 20.0
        assert p.frequency_hz == 10.0

    def test_custom_values(self):
        p = StimulusPattern(
            pattern_type=StimulusPatternType.SINUSOIDAL.name,
            amplitude_pA=250.0,
            frequency_hz=40.0,
        )
        assert p.amplitude_pA == 250.0
        assert p.frequency_hz == 40.0


class TestStimulusManager:
    """Test stimulus current generation."""

    def setup_method(self):
        self.n_neurons = 1000
        self.dt_ms = 0.5
        self.manager = StimulusManager(self.n_neurons, self.dt_ms)

    def test_constant_stimulus(self):
        channel = StimulusChannel(
            name="test_const",
            pattern=StimulusPattern(
                pattern_type=StimulusPatternType.CONSTANT.name,
                amplitude_pA=200.0,
            ),
            target_neuron_indices=list(range(100)),
            onset_ms=0.0,
            duration_ms=1000.0,
        )

        group_mgr = NeuronGroupManager(self.n_neurons)
        self.manager.initialize([channel], group_mgr, cp)

        # At t=50ms (within duration), should get current
        current = self.manager.compute_step_current(50.0, 0.0, cp)
        assert current.shape == (self.n_neurons,)

        # First 100 neurons should have current
        assert np.sum(current[:100]) > 0
        # Rest should be zero
        assert np.sum(current[100:]) == 0.0

    def test_sinusoidal_stimulus(self):
        channel = StimulusChannel(
            name="test_sin",
            pattern=StimulusPattern(
                pattern_type=StimulusPatternType.SINUSOIDAL.name,
                amplitude_pA=100.0,
                frequency_hz=10.0,
            ),
            target_neuron_indices=list(range(50)),
            onset_ms=0.0,
            duration_ms=1000.0,
        )

        group_mgr = NeuronGroupManager(self.n_neurons)
        self.manager.initialize([channel], group_mgr, cp)

        # At t=25ms (quarter cycle of 10Hz = 25ms), sin should be at peak
        current = self.manager.compute_step_current(25.0, 0.0, cp)
        target_current = current[:50]
        assert np.all(target_current > 0)  # Should be positive at quarter cycle

    def test_timing_window(self):
        channel = StimulusChannel(
            name="test_timing",
            pattern=StimulusPattern(
                pattern_type=StimulusPatternType.CONSTANT.name,
                amplitude_pA=150.0,
            ),
            target_neuron_indices=list(range(100)),
            onset_ms=100.0,
            duration_ms=200.0,
        )

        group_mgr = NeuronGroupManager(self.n_neurons)
        self.manager.initialize([channel], group_mgr, cp)

        # Before onset: no current
        current_before = self.manager.compute_step_current(50.0, 0.0, cp)
        assert np.sum(np.abs(current_before)) == 0.0

        # During stimulus: current present
        current_during = self.manager.compute_step_current(200.0, 0.0, cp)
        assert np.sum(current_during[:100]) > 0

        # After offset: no current
        current_after = self.manager.compute_step_current(350.0, 0.0, cp)
        assert np.sum(np.abs(current_after)) == 0.0

    def test_ramp_stimulus(self):
        channel = StimulusChannel(
            name="test_ramp",
            pattern=StimulusPattern(
                pattern_type=StimulusPatternType.RAMP.name,
                start_amplitude_pA=0.0,
                end_amplitude_pA=200.0,
            ),
            target_neuron_indices=[0],
            onset_ms=0.0,
            duration_ms=1000.0,
        )

        group_mgr = NeuronGroupManager(self.n_neurons)
        self.manager.initialize([channel], group_mgr, cp)

        # At start: ~0
        c0 = self.manager.compute_step_current(0.0, 0.0, cp).copy()
        # At midpoint: ~100
        c50 = self.manager.compute_step_current(500.0, 0.0, cp).copy()
        # At end: ~200
        c100 = self.manager.compute_step_current(999.0, 0.0, cp).copy()

        assert c0[0] < c50[0] < c100[0]

    def test_cleanup(self):
        self.manager.initialize([], NeuronGroupManager(self.n_neurons), cp)
        self.manager.cleanup()
        assert self.manager.cp_stimulus_current is None


class TestNeuronGroupManager:
    """Test neuron group management."""

    def test_group_creation(self):
        mgr = NeuronGroupManager(1000)
        groups = [
            NeuronGroup(name="input", role=NeuronGroupRole.INPUT.name,
                       index_start=0, index_end=100),
            NeuronGroup(name="output", role=NeuronGroupRole.OUTPUT.name,
                       index_start=100, index_end=200),
        ]
        mgr.initialize(groups, cp_module=cp)

        assert len(mgr.groups) == 2
        assert len(mgr.get_group("input").neuron_indices) == 100
        assert len(mgr.get_group("output").neuron_indices) == 100

    def test_groups_by_role(self):
        mgr = NeuronGroupManager(1000)
        groups = [
            NeuronGroup(name="in1", role=NeuronGroupRole.INPUT.name,
                       index_start=0, index_end=50),
            NeuronGroup(name="in2", role=NeuronGroupRole.INPUT.name,
                       index_start=50, index_end=100),
            NeuronGroup(name="out1", role=NeuronGroupRole.OUTPUT.name,
                       index_start=100, index_end=200),
        ]
        mgr.initialize(groups, cp_module=cp)

        input_groups = mgr.get_groups_by_role(NeuronGroupRole.INPUT.name)
        assert len(input_groups) == 2

        output_groups = mgr.get_groups_by_role(NeuronGroupRole.OUTPUT.name)
        assert len(output_groups) == 1

    def test_trait_based_population(self):
        mgr = NeuronGroupManager(10)
        traits = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2], dtype=np.int32)

        groups = [
            NeuronGroup(name="trait_0", role=NeuronGroupRole.INPUT.name, trait_index=0),
            NeuronGroup(name="trait_1", role=NeuronGroupRole.OUTPUT.name, trait_index=1),
        ]
        mgr.initialize(groups, cp_traits=traits, cp_module=cp)

        assert sorted(mgr.get_group("trait_0").neuron_indices) == [0, 1, 2]
        assert sorted(mgr.get_group("trait_1").neuron_indices) == [3, 4, 5, 6]

    def test_group_mask(self):
        mgr = NeuronGroupManager(10)
        groups = [
            NeuronGroup(name="test", role=NeuronGroupRole.INPUT.name,
                       neuron_indices=[2, 5, 7]),
        ]
        mgr.initialize(groups, cp_module=cp)

        mask = mgr.get_group_mask("test", cp)
        assert mask[2] == True
        assert mask[5] == True
        assert mask[7] == True
        assert mask[0] == False
        assert np.sum(mask) == 3


class TestReadoutEngine:
    """Test readout measurement engine."""

    def test_firing_rate_computation(self):
        n = 100
        dt = 0.5
        readout = ReadoutEngine(n, dt)

        group_mgr = NeuronGroupManager(n)
        group_mgr.initialize([
            NeuronGroup(name="test_group", role=NeuronGroupRole.OUTPUT.name,
                       index_start=0, index_end=50),
        ], cp_module=cp)

        config = ReadoutConfig(
            rate_window_ms=50.0,
            rate_group_names=["test_group"],
        )
        readout.initialize(config, group_mgr, cp)

        # Simulate spikes
        firing = np.zeros(n, dtype=np.bool_)
        firing[0:10] = True  # 10 of 50 neurons fire
        vm = np.full(n, -65.0, dtype=np.float32)

        # Run several steps
        for _ in range(100):
            readout.update(firing, vm, cp)

        rate = readout.current_rates.get("test_group", 0.0)
        assert rate > 0  # Should have measured a non-zero rate


class TestExperimentPresets:
    """Test experiment preset creation."""

    def test_preset_names(self):
        names = ExperimentPresets.get_preset_names()
        assert len(names) == 4
        assert "Basic Stimulus-Response" in names
        assert "Associative Conditioning (CS-US)" in names
        assert "Reinforcement Learning (R-STDP)" in names
        assert "Frequency Response Characterization" in names

    def test_basic_stimulus_response(self):
        config = ExperimentPresets.basic_stimulus_response()
        assert config.name == "Basic Stimulus-Response"
        assert config.enabled == True
        assert len(config.neuron_groups) == 2
        assert len(config.stimulus_channels) == 1
        assert len(config.phases) == 3

        # Check group roles
        roles = [g.role for g in config.neuron_groups]
        assert NeuronGroupRole.INPUT.name in roles
        assert NeuronGroupRole.OUTPUT.name in roles

    def test_associative_conditioning(self):
        config = ExperimentPresets.associative_conditioning()
        assert "Associative" in config.name
        assert len(config.stimulus_channels) == 2  # CS and US

        channel_names = [ch.name for ch in config.stimulus_channels]
        assert "cs" in channel_names
        assert "us" in channel_names

    def test_reinforcement_learning(self):
        config = ExperimentPresets.reinforcement_learning()
        assert "Reinforcement" in config.name

        # Should have a training phase
        training_phases = [p for p in config.phases
                          if p.phase_type == ExperimentPhaseType.TRAINING.name]
        assert len(training_phases) >= 1

        # Training config should be RL mode
        training_config = training_phases[0].training_config
        assert training_config.mode == TrainingMode.REINFORCEMENT_LEARNING.name

    def test_frequency_response(self):
        config = ExperimentPresets.frequency_response_characterization(
            num_frequencies=5
        )
        assert "Frequency" in config.name
        assert len(config.stimulus_channels) == 5  # One per frequency

        # PSD should be enabled
        assert config.readout.enable_psd == True

    def test_get_preset_by_name(self):
        config = ExperimentPresets.get_preset("Basic Stimulus-Response")
        assert config is not None
        assert config.name == "Basic Stimulus-Response"

        invalid = ExperimentPresets.get_preset("Nonexistent Preset")
        assert invalid is None


class TestExperimentConfigSerialization:
    """Test JSON serialization of experiment configs."""

    def test_serialize_basic(self):
        config = ExperimentPresets.basic_stimulus_response()
        d = experiment_config_to_dict(config)

        assert isinstance(d, dict)
        assert d["name"] == "Basic Stimulus-Response"
        assert isinstance(d["neuron_groups"], list)
        assert isinstance(d["stimulus_channels"], list)
        assert isinstance(d["phases"], list)

    def test_roundtrip(self):
        original = ExperimentPresets.associative_conditioning()

        # Serialize
        d = experiment_config_to_dict(original)
        json_str = json.dumps(d)

        # Deserialize
        d2 = json.loads(json_str)
        restored = experiment_config_from_dict(d2)

        assert restored.name == original.name
        assert len(restored.neuron_groups) == len(original.neuron_groups)
        assert len(restored.stimulus_channels) == len(original.stimulus_channels)
        assert len(restored.phases) == len(original.phases)
        assert restored.enabled == original.enabled

    def test_roundtrip_preserves_training_config(self):
        original = ExperimentPresets.reinforcement_learning()

        d = experiment_config_to_dict(original)
        restored = experiment_config_from_dict(d)

        # Check training phase config
        orig_training = [p for p in original.phases
                        if p.phase_type == ExperimentPhaseType.TRAINING.name][0]
        rest_training = [p for p in restored.phases
                        if p.phase_type == ExperimentPhaseType.TRAINING.name][0]

        assert orig_training.training_config.mode == rest_training.training_config.mode
        assert orig_training.training_config.reward_magnitude == rest_training.training_config.reward_magnitude


class TestExperimentEngine:
    """Test the top-level experiment orchestrator."""

    def setup_method(self):
        self.n = 500
        self.dt = 0.5
        self.engine = ExperimentEngine(self.n, self.dt)

    def test_load_experiment(self):
        config = ExperimentPresets.basic_stimulus_response(input_group_size=50, output_group_size=50)
        self.engine.load_experiment(config)

        assert self.engine.config is not None
        assert len(self.engine.phases) == 3
        assert self.engine.is_experiment_running == False

    def test_initialize(self):
        config = ExperimentPresets.basic_stimulus_response(input_group_size=50, output_group_size=50)
        self.engine.load_experiment(config)
        self.engine.initialize(cp_module=cp)

        assert len(self.engine.group_manager.groups) == 2
        assert len(self.engine.log) == 1  # initialization event

    def test_start_stop(self):
        config = ExperimentPresets.basic_stimulus_response(input_group_size=50, output_group_size=50)
        self.engine.load_experiment(config)
        self.engine.initialize(cp_module=cp)
        self.engine.start(0.0)

        assert self.engine.is_experiment_running == True

        self.engine.stop()
        assert self.engine.is_experiment_running == False

    def test_step_returns_current(self):
        config = ExperimentPresets.basic_stimulus_response(input_group_size=50, output_group_size=50)
        self.engine.load_experiment(config)
        self.engine.initialize(cp_module=cp)
        self.engine.start(0.0)

        firing = np.zeros(self.n, dtype=np.bool_)
        vm = np.full(self.n, -65.0, dtype=np.float32)

        current = self.engine.step(0.0, firing, vm, None, cp)
        assert current.shape == (self.n,)

    def test_phase_transitions(self):
        config = ExperimentConfig(
            name="Phase Test",
            neuron_groups=[
                NeuronGroup(name="g1", role=NeuronGroupRole.INPUT.name,
                           index_start=0, index_end=50),
            ],
            stimulus_channels=[],
            phases=[
                ExperimentPhase(name="p1", phase_type=ExperimentPhaseType.BASELINE.name,
                               duration_ms=10.0),
                ExperimentPhase(name="p2", phase_type=ExperimentPhaseType.STIMULUS.name,
                               duration_ms=10.0),
            ],
            readout=ReadoutConfig(),
            enabled=True,
        )

        self.engine.load_experiment(config)
        self.engine.initialize(cp_module=cp)
        self.engine.start(0.0)

        firing = np.zeros(self.n, dtype=np.bool_)
        vm = np.full(self.n, -65.0, dtype=np.float32)

        # Step through first phase
        for t in np.arange(0, 10.0, self.dt):
            self.engine.step(t, firing, vm, None, cp)

        assert self.engine.current_phase_idx == 0  # Still in first phase

        # Step into second phase
        for t in np.arange(10.0, 15.0, self.dt):
            self.engine.step(t, firing, vm, None, cp)

        assert self.engine.current_phase_idx == 1  # Should have moved to second phase

    def test_experiment_completion(self):
        config = ExperimentConfig(
            name="Completion Test",
            neuron_groups=[],
            stimulus_channels=[],
            phases=[
                ExperimentPhase(name="only", phase_type=ExperimentPhaseType.BASELINE.name,
                               duration_ms=5.0),
            ],
            readout=ReadoutConfig(),
            enabled=True,
        )

        self.engine.load_experiment(config)
        self.engine.initialize(cp_module=cp)
        self.engine.start(0.0)

        firing = np.zeros(self.n, dtype=np.bool_)
        vm = np.full(self.n, -65.0, dtype=np.float32)

        # Step past the only phase
        for t in np.arange(0, 10.0, self.dt):
            self.engine.step(t, firing, vm, None, cp)

        assert self.engine.is_experiment_complete == True
        assert self.engine.is_experiment_running == False

    def test_get_status(self):
        config = ExperimentPresets.basic_stimulus_response(input_group_size=50, output_group_size=50)
        self.engine.load_experiment(config)
        self.engine.initialize(cp_module=cp)

        status = self.engine.get_experiment_status()
        assert "is_running" in status
        assert "is_complete" in status
        assert "total_phases" in status

    def test_cleanup(self):
        config = ExperimentPresets.basic_stimulus_response(input_group_size=50, output_group_size=50)
        self.engine.load_experiment(config)
        self.engine.initialize(cp_module=cp)
        self.engine.start(0.0)
        self.engine.cleanup()

        assert self.engine.is_experiment_running == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
