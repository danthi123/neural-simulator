"""
Determinism and Reproducibility Tests

Tests that simulations with the same seed produce identical results.
This ensures the simulator is deterministic for scientific reproducibility.

Run with: pytest tests/test_determinism.py -v
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cupy as cp
import pytest

from neural_simulator import (
    SimulationBridge, CoreSimConfig, VisualizationConfig,
    RuntimeState, GPUConfig, NeuronModel
)


class TestDeterministicSpikes:
    """Test that spike trains are deterministic given same seed."""
    
    def test_izhikevich_deterministic_spikes(self):
        """Two Izhikevich runs with same seed produce identical spike trains."""
        config = CoreSimConfig(
            num_neurons=100,
            connections_per_neuron=50,
            seed=42,
            neuron_model_type=NeuronModel.IZHIKEVICH.name,
            dt_ms=1.0,
            enable_hebbian_learning=False,  # Disable for strict determinism
            enable_short_term_plasticity=False,
            enable_homeostasis=False
        )
        
        gpu_config = GPUConfig(enable_profiling=False)
        
        # Run 1
        sim1 = SimulationBridge(
            core_config=config,
            viz_config=VisualizationConfig(),
            runtime_state=RuntimeState(),
            gpu_config=gpu_config
        )
        sim1._initialize_simulation_data()
        
        spikes1 = []
        for _ in range(100):
            sim1._run_one_simulation_step()
            spikes1.append(cp.asnumpy(sim1.cp_firing_states).copy())
        
        sim1.clear_simulation_state_and_gpu_memory()
        
        # Run 2
        sim2 = SimulationBridge(
            core_config=config,
            viz_config=VisualizationConfig(),
            runtime_state=RuntimeState(),
            gpu_config=gpu_config
        )
        sim2._initialize_simulation_data()
        
        spikes2 = []
        for _ in range(100):
            sim2._run_one_simulation_step()
            spikes2.append(cp.asnumpy(sim2.cp_firing_states).copy())
        
        sim2.clear_simulation_state_and_gpu_memory()
        
        # Compare
        for i, (s1, s2) in enumerate(zip(spikes1, spikes2)):
            assert np.array_equal(s1, s2), f"Step {i}: Izhikevich spike trains differ"
        
        print(f"✓ Izhikevich deterministic: {len(spikes1)} steps matched")
    
    def test_hodgkin_huxley_deterministic_spikes(self):
        """Two HH runs with same seed produce identical spike trains."""
        config = CoreSimConfig(
            num_neurons=50,  # HH is slower, use fewer neurons
            connections_per_neuron=25,
            seed=123,
            neuron_model_type=NeuronModel.HODGKIN_HUXLEY.name,
            dt_ms=0.025,  # HH needs smaller dt
            enable_hebbian_learning=False,
            enable_short_term_plasticity=False,
            enable_homeostasis=False
        )
        
        gpu_config = GPUConfig(enable_profiling=False)
        
        # Run 1
        sim1 = SimulationBridge(
            core_config=config,
            viz_config=VisualizationConfig(),
            runtime_state=RuntimeState(),
            gpu_config=gpu_config
        )
        sim1._initialize_simulation_data()
        
        spikes1 = []
        for _ in range(50):  # Fewer steps due to smaller dt
            sim1._run_one_simulation_step()
            spikes1.append(cp.asnumpy(sim1.cp_firing_states).copy())
        
        sim1.clear_simulation_state_and_gpu_memory()
        
        # Run 2
        sim2 = SimulationBridge(
            core_config=config,
            viz_config=VisualizationConfig(),
            runtime_state=RuntimeState(),
            gpu_config=gpu_config
        )
        sim2._initialize_simulation_data()
        
        spikes2 = []
        for _ in range(50):
            sim2._run_one_simulation_step()
            spikes2.append(cp.asnumpy(sim2.cp_firing_states).copy())
        
        sim2.clear_simulation_state_and_gpu_memory()
        
        # Compare
        for i, (s1, s2) in enumerate(zip(spikes1, spikes2)):
            assert np.array_equal(s1, s2), f"Step {i}: HH spike trains differ"
        
        print(f"✓ Hodgkin-Huxley deterministic: {len(spikes1)} steps matched")
    
    def test_adex_deterministic_spikes(self):
        """Two AdEx runs with same seed produce identical spike trains."""
        config = CoreSimConfig(
            num_neurons=100,
            connections_per_neuron=50,
            seed=456,
            neuron_model_type=NeuronModel.ADEX.name,
            dt_ms=0.1,  # AdEx benefits from smaller dt
            enable_hebbian_learning=False,
            enable_short_term_plasticity=False,
            enable_homeostasis=False
        )
        
        gpu_config = GPUConfig(enable_profiling=False)
        
        # Run 1
        sim1 = SimulationBridge(
            core_config=config,
            viz_config=VisualizationConfig(),
            runtime_state=RuntimeState(),
            gpu_config=gpu_config
        )
        sim1._initialize_simulation_data()
        
        spikes1 = []
        for _ in range(100):
            sim1._run_one_simulation_step()
            spikes1.append(cp.asnumpy(sim1.cp_firing_states).copy())
        
        sim1.clear_simulation_state_and_gpu_memory()
        
        # Run 2
        sim2 = SimulationBridge(
            core_config=config,
            viz_config=VisualizationConfig(),
            runtime_state=RuntimeState(),
            gpu_config=gpu_config
        )
        sim2._initialize_simulation_data()
        
        spikes2 = []
        for _ in range(100):
            sim2._run_one_simulation_step()
            spikes2.append(cp.asnumpy(sim2.cp_firing_states).copy())
        
        sim2.clear_simulation_state_and_gpu_memory()
        
        # Compare
        for i, (s1, s2) in enumerate(zip(spikes1, spikes2)):
            assert np.array_equal(s1, s2), f"Step {i}: AdEx spike trains differ"
        
        print(f"✓ AdEx deterministic: {len(spikes1)} steps matched")


class TestDeterministicMembranePotential:
    """Test that membrane potential traces are deterministic."""
    
    def test_izhikevich_membrane_potential(self):
        """Two runs produce identical membrane potential traces."""
        config = CoreSimConfig(
            num_neurons=50,
            connections_per_neuron=25,
            seed=789,
            neuron_model_type=NeuronModel.IZHIKEVICH.name,
            enable_hebbian_learning=False,
            enable_short_term_plasticity=False
        )
        
        gpu_config = GPUConfig(enable_profiling=False)
        
        # Run 1
        sim1 = SimulationBridge(core_config=config, gpu_config=gpu_config)
        sim1._initialize_simulation_data()
        
        v_traces1 = []
        for _ in range(50):
            sim1._run_one_simulation_step()
            v_traces1.append(cp.asnumpy(sim1.cp_membrane_potential_v).copy())
        
        sim1.clear_simulation_state_and_gpu_memory()
        
        # Run 2
        sim2 = SimulationBridge(core_config=config, gpu_config=gpu_config)
        sim2._initialize_simulation_data()
        
        v_traces2 = []
        for _ in range(50):
            sim2._run_one_simulation_step()
            v_traces2.append(cp.asnumpy(sim2.cp_membrane_potential_v).copy())
        
        sim2.clear_simulation_state_and_gpu_memory()
        
        # Compare with tolerance for floating-point arithmetic
        for i, (v1, v2) in enumerate(zip(v_traces1, v_traces2)):
            assert np.allclose(v1, v2, rtol=1e-6, atol=1e-6), \
                f"Step {i}: Membrane potential traces differ"
        
        print(f"✓ Membrane potential deterministic: {len(v_traces1)} steps matched")


class TestDeterministicConnectivity:
    """Test that connectivity generation is deterministic."""
    
    def test_connectivity_generation(self):
        """Same seed produces identical connectivity matrices."""
        config = CoreSimConfig(
            num_neurons=200,
            connections_per_neuron=50,
            seed=999,
            neuron_model_type=NeuronModel.IZHIKEVICH.name
        )
        
        gpu_config = GPUConfig(enable_profiling=False)
        
        # Generate 1
        sim1 = SimulationBridge(core_config=config, gpu_config=gpu_config)
        sim1._initialize_simulation_data()
        
        conn1_data = cp.asnumpy(sim1.cp_connections.data).copy()
        conn1_indices = cp.asnumpy(sim1.cp_connections.indices).copy()
        conn1_indptr = cp.asnumpy(sim1.cp_connections.indptr).copy()
        
        sim1.clear_simulation_state_and_gpu_memory()
        
        # Generate 2
        sim2 = SimulationBridge(core_config=config, gpu_config=gpu_config)
        sim2._initialize_simulation_data()
        
        conn2_data = cp.asnumpy(sim2.cp_connections.data).copy()
        conn2_indices = cp.asnumpy(sim2.cp_connections.indices).copy()
        conn2_indptr = cp.asnumpy(sim2.cp_connections.indptr).copy()
        
        sim2.clear_simulation_state_and_gpu_memory()
        
        # Compare connectivity
        assert np.allclose(conn1_data, conn2_data, rtol=1e-6), "Connection weights differ"
        assert np.array_equal(conn1_indices, conn2_indices), "Connection indices differ"
        assert np.array_equal(conn1_indptr, conn2_indptr), "Connection indptr differ"
        
        print(f"✓ Connectivity generation deterministic: {len(conn1_data)} synapses matched")


class TestSeedTracking:
    """Test that actual seed used is tracked correctly."""
    
    def test_explicit_seed_tracked(self):
        """Explicit seed is stored in runtime_state."""
        config = CoreSimConfig(seed=12345)
        sim = SimulationBridge(core_config=config)
        sim._initialize_simulation_data()
        
        assert sim.runtime_state.actual_seed_used == 12345, \
            "Explicit seed not tracked correctly"
        
        sim.clear_simulation_state_and_gpu_memory()
        print(f"✓ Explicit seed tracked: {sim.runtime_state.actual_seed_used}")
    
    def test_random_seed_generated(self):
        """Random seed (-1) generates and stores a seed."""
        config = CoreSimConfig(seed=-1)
        sim = SimulationBridge(core_config=config)
        sim._initialize_simulation_data()
        
        assert sim.runtime_state.actual_seed_used != -1, \
            "Random seed was not generated"
        assert sim.runtime_state.actual_seed_used >= 0, \
            "Generated seed is negative"
        
        sim.clear_simulation_state_and_gpu_memory()
        print(f"✓ Random seed generated and tracked: {sim.runtime_state.actual_seed_used}")


if __name__ == "__main__":
    # Can run directly without pytest
    print("Running determinism tests...")
    
    test_spikes = TestDeterministicSpikes()
    test_spikes.test_izhikevich_deterministic_spikes()
    test_spikes.test_hodgkin_huxley_deterministic_spikes()
    test_spikes.test_adex_deterministic_spikes()
    
    test_v = TestDeterministicMembranePotential()
    test_v.test_izhikevich_membrane_potential()
    
    test_conn = TestDeterministicConnectivity()
    test_conn.test_connectivity_generation()
    
    test_seed = TestSeedTracking()
    test_seed.test_explicit_seed_tracked()
    test_seed.test_random_seed_generated()
    
    print("\n✅ All determinism tests passed!")
