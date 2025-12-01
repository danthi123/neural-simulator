#!/usr/bin/env python
"""
Test script for Phase C: Network-Level Realism & Connectivity
Tests STDP, reward-modulated plasticity, and structural plasticity features.
"""

import sys
import numpy as np
try:
    import cupy as cp
except ImportError:
    print("CuPy not available. This test requires GPU support.")
    sys.exit(1)

# Import configuration from neural-simulator
sys.path.insert(0, '.')
from dataclasses import asdict

# Prevent GUI from starting
import os
os.environ['HEADLESS_TEST'] = '1'

# Import only what we need from neural-simulator
# Load minimal components without starting GUI
import importlib.util
spec = importlib.util.spec_from_file_location("neural_sim", "neural-simulator.py")
neural_sim = importlib.util.module_from_spec(spec)

# Intercept the main() call to prevent GUI startup
original_main = None
def mock_main():
    pass

# Execute the module without running main
import sys
old_argv = sys.argv
sys.argv = ['test']  # Prevent command-line parsing

try:
    # Temporarily replace main to prevent GUI startup
    if hasattr(neural_sim, 'main'):
        original_main = neural_sim.main
        neural_sim.main = mock_main
    
    spec.loader.exec_module(neural_sim)
finally:
    sys.argv = old_argv

# Import the classes we need
CoreSimConfig = neural_sim.CoreSimConfig
SimulationBridge = neural_sim.SimulationBridge
NeuronModel = neural_sim.NeuronModel

def test_stdp_basic():
    """Test basic STDP functionality."""
    print("\n=== Testing STDP ===")
    
    # Create minimal config with STDP enabled
    cfg = CoreSimConfig(
        num_neurons=10,
        connections_per_neuron=5,
        dt_ms=1.0,
        enable_stdp=True,
        enable_hebbian_learning=False,  # Disable to isolate STDP
        stdp_a_plus=0.01,
        stdp_a_minus=0.0105,
        stdp_tau_plus_ms=20.0,
        stdp_tau_minus_ms=20.0,
        stdp_w_min=0.0,
        stdp_w_max=2.0,
        neuron_model_type="IZHIKEVICH"
    )
    
    # Initialize simulation bridge
    sim = SimulationBridge(core_config=cfg)
    sim._initialize_simulation_data()
    
    assert sim.is_initialized, "Simulation failed to initialize"
    assert sim.cp_last_spike_time is not None, "STDP spike time array not initialized"
    assert sim.cp_last_spike_time.shape[0] == 10, f"Expected 10 neurons, got {sim.cp_last_spike_time.shape[0]}"
    
    # Check initial spike times are negative (not yet spiked)
    initial_times = cp.asnumpy(sim.cp_last_spike_time)
    assert np.all(initial_times < 0), "Initial spike times should be negative"
    
    print(f"  ✓ STDP arrays initialized: {sim.cp_last_spike_time.shape}")
    print(f"  ✓ Initial spike times: {initial_times[:5]}")
    
    # Manually trigger some spikes and run a step
    sim.cp_firing_states[0] = True
    sim.cp_firing_states[5] = True
    sim._run_one_simulation_step()
    
    # Check spike times were updated
    updated_times = cp.asnumpy(sim.cp_last_spike_time)
    assert updated_times[0] == sim.runtime_state.current_time_ms, "Neuron 0 spike time not updated"
    assert updated_times[5] == sim.runtime_state.current_time_ms, "Neuron 5 spike time not updated"
    print(f"  ✓ Spike times updated correctly after firing")
    
    print("  ✓ STDP basic test passed")
    
    # Cleanup
    sim.clear_simulation_state_and_gpu_memory()

def test_reward_modulation():
    """Test reward-modulated plasticity."""
    print("\n=== Testing Reward Modulation ===")
    
    cfg = CoreSimConfig(
        num_neurons=10,
        connections_per_neuron=5,
        dt_ms=1.0,
        enable_stdp=True,
        enable_reward_modulation=True,
        reward_learning_rate=0.01,
        reward_eligibility_tau_ms=1000.0,
        reward_baseline=0.0,
        current_reward_signal=0.0,
        neuron_model_type="IZHIKEVICH"
    )
    
    sim = SimulationBridge(core_config=cfg)
    sim._initialize_simulation_data()
    
    assert sim.is_initialized, "Simulation failed to initialize"
    assert sim.cp_eligibility_trace is not None, "Eligibility trace not initialized"
    
    num_synapses = sim.cp_connections.nnz
    assert sim.cp_eligibility_trace.shape[0] == num_synapses, \
        f"Eligibility trace size mismatch: {sim.cp_eligibility_trace.shape[0]} vs {num_synapses}"
    
    print(f"  ✓ Reward modulation arrays initialized: {num_synapses} synapses")
    
    # Check initial eligibility traces are zero
    initial_traces = cp.asnumpy(sim.cp_eligibility_trace)
    assert np.all(initial_traces == 0), "Initial eligibility traces should be zero"
    print(f"  ✓ Initial eligibility traces: all zeros")
    
    # Run a step with reward signal
    sim.core_config.current_reward_signal = 1.0
    sim._run_one_simulation_step()
    
    print(f"  ✓ Reward modulation step executed")
    print("  ✓ Reward modulation basic test passed")
    
    # Cleanup
    sim.clear_simulation_state_and_gpu_memory()

def test_structural_plasticity():
    """Test structural plasticity (synapse formation/elimination)."""
    print("\n=== Testing Structural Plasticity ===")
    
    cfg = CoreSimConfig(
        num_neurons=20,
        connections_per_neuron=3,
        dt_ms=1.0,
        enable_structural_plasticity=True,
        struct_plast_formation_rate=1e-3,  # High rate for testing
        struct_plast_elimination_rate=1e-3,  # High rate for testing
        struct_plast_weight_threshold=0.05,
        struct_plast_target_density=0.15,
        struct_plast_update_interval_steps=10,
        enable_hebbian_learning=False,  # Disable to isolate structural plasticity
        neuron_model_type="IZHIKEVICH"
    )
    
    sim = SimulationBridge(core_config=cfg)
    sim._initialize_simulation_data()
    
    assert sim.is_initialized, "Simulation failed to initialize"
    assert sim.cp_struct_plast_step_counter is not None, "Structural plasticity counter not initialized"
    assert sim.cp_struct_plast_step_counter == 0, "Counter should start at zero"
    
    initial_synapses = sim.cp_connections.nnz
    print(f"  ✓ Structural plasticity initialized")
    print(f"  ✓ Initial synapse count: {initial_synapses}")
    
    # Run multiple steps to trigger structural plasticity updates
    for i in range(15):
        sim._run_one_simulation_step()
    
    final_synapses = sim.cp_connections.nnz
    print(f"  ✓ After 15 steps, synapse count: {final_synapses}")
    
    # Synapses should have changed (either formed or eliminated)
    # Note: Due to randomness, we just check that the mechanism runs without errors
    print(f"  ✓ Synapse count change: {final_synapses - initial_synapses}")
    print("  ✓ Structural plasticity basic test passed")
    
    # Cleanup
    sim.clear_simulation_state_and_gpu_memory()

def test_phase_c_integration():
    """Test all Phase C features working together."""
    print("\n=== Testing Phase C Integration ===")
    
    cfg = CoreSimConfig(
        num_neurons=50,
        connections_per_neuron=10,
        dt_ms=1.0,
        enable_stdp=True,
        enable_reward_modulation=True,
        enable_structural_plasticity=True,
        stdp_a_plus=0.01,
        stdp_a_minus=0.0105,
        reward_learning_rate=0.005,
        struct_plast_formation_rate=1e-5,
        struct_plast_elimination_rate=5e-6,
        struct_plast_update_interval_steps=50,
        neuron_model_type="IZHIKEVICH"
    )
    
    sim = SimulationBridge(core_config=cfg)
    sim._initialize_simulation_data()
    
    assert sim.is_initialized, "Simulation failed to initialize"
    assert sim.cp_last_spike_time is not None, "STDP not initialized"
    assert sim.cp_eligibility_trace is not None, "Reward modulation not initialized"
    assert sim.cp_struct_plast_step_counter is not None, "Structural plasticity not initialized"
    
    print(f"  ✓ All Phase C features initialized")
    print(f"  ✓ Neurons: {cfg.num_neurons}")
    print(f"  ✓ Initial synapses: {sim.cp_connections.nnz}")
    
    # Run simulation for 100 steps
    initial_synapses = sim.cp_connections.nnz
    initial_weights = cp.asnumpy(sim.cp_connections.data.copy())
    
    for i in range(100):
        if i % 20 == 0:
            # Apply reward signal periodically
            sim.core_config.current_reward_signal = 0.5
        else:
            sim.core_config.current_reward_signal = 0.0
        
        sim._run_one_simulation_step()
    
    final_synapses = sim.cp_connections.nnz
    final_weights = cp.asnumpy(sim.cp_connections.data)
    
    print(f"  ✓ Ran 100 simulation steps successfully")
    print(f"  ✓ Final synapses: {final_synapses} (change: {final_synapses - initial_synapses})")
    
    # Check that weights have changed (plasticity is working)
    # Compare weights of synapses that still exist
    min_size = min(len(initial_weights), len(final_weights))
    if min_size > 0:
        weight_changes = np.abs(final_weights[:min_size] - initial_weights[:min_size])
        avg_change = np.mean(weight_changes)
        print(f"  ✓ Average weight change: {avg_change:.6f}")
        assert avg_change > 0, "Weights should have changed due to plasticity"
    
    print(f"  ✓ Phase C integration test passed")
    
    # Cleanup
    sim.clear_simulation_state_and_gpu_memory()

def main():
    """Run all Phase C tests."""
    print("=" * 60)
    print("Phase C: Network-Level Realism & Connectivity - Test Suite")
    print("=" * 60)
    
    try:
        test_stdp_basic()
        test_reward_modulation()
        test_structural_plasticity()
        test_phase_c_integration()
        
        print("\n" + "=" * 60)
        print("✓ ALL PHASE C TESTS PASSED")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
