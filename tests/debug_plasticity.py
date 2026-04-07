"""Quick diagnostic for plasticity test 0-spike issue."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cupy as cp
import numpy as np

from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("neural_simulator", os.path.join(os.path.dirname(__file__), '..', 'neural-simulator.py'))
ns = module_from_spec(spec)
spec.loader.exec_module(ns)

CoreSimConfig = ns.CoreSimConfig
GPUConfig = ns.GPUConfig
VisualizationConfig = ns.VisualizationConfig
RuntimeState = ns.RuntimeState
SimulationBridge = ns.SimulationBridge
NeuronModel = ns.NeuronModel

def inject_external_drive(sim, model_type, strength=1.0):
    n = sim.core_config.num_neurons
    base_mean, base_std = 1500.0, 300.0
    drive = cp.random.normal(base_mean * strength, base_std * strength, n).astype(cp.float32)
    sim.cp_external_input_current[:] = cp.maximum(drive, 0.0)

# Plasticity config (same as test)
config = CoreSimConfig(
    num_neurons=300,
    connections_per_neuron=50,
    seed=42,
    neuron_model_type=NeuronModel.IZHIKEVICH.name,
    dt_ms=1.0,
    enable_hebbian_learning=True,
    hebbian_learning_rate=0.0005,
    hebbian_min_weight=0.05,
    hebbian_max_weight=1.0,
    enable_short_term_plasticity=True,
    stp_U=0.15,
    enable_stdp=True,
    stdp_a_plus=0.01,
    stdp_a_minus=0.0105,
    enable_reward_modulation=True,
    reward_learning_rate=0.01,
    enable_homeostasis=True,
    homeostasis_target_rate=0.02,
    enable_ou_process=False,
)
gpu_config = GPUConfig(enable_profiling=False)

sim = SimulationBridge(
    core_config=config,
    viz_config=VisualizationConfig(),
    runtime_state=RuntimeState(),
    gpu_config=gpu_config
)
sim._initialize_simulation_data()
inject_external_drive(sim, NeuronModel.IZHIKEVICH.name, strength=1.5)

print(f"External drive: mean={float(sim.cp_external_input_current.mean()):.1f} pA, max={float(sim.cp_external_input_current.max()):.1f} pA")
print(f"Initial V: mean={float(sim.cp_membrane_potential_v.mean()):.2f}, range=[{float(sim.cp_membrane_potential_v.min()):.2f}, {float(sim.cp_membrane_potential_v.max()):.2f}]")
if sim.cp_neuron_firing_thresholds is not None:
    print(f"Initial thresholds: mean={float(sim.cp_neuron_firing_thresholds.mean()):.2f}, range=[{float(sim.cp_neuron_firing_thresholds.min()):.2f}, {float(sim.cp_neuron_firing_thresholds.max()):.2f}]")
print(f"STP u: mean={float(sim.cp_stp_u[:sim._synapse_count].mean()):.4f}")
print(f"STP x: mean={float(sim.cp_stp_x[:sim._synapse_count].mean()):.4f}")
print(f"Weights: mean={float(sim.cp_connections.data.mean()):.4f}, nnz={sim.cp_connections.nnz}")

print("\n--- Running step 0 ---")
sim._run_one_simulation_step()
n_spikes_0 = int(sim.cp_firing_states.sum())
print(f"Step 0: spikes={n_spikes_0}")
print(f"  V: mean={float(sim.cp_membrane_potential_v.mean()):.2f}, range=[{float(sim.cp_membrane_potential_v.min()):.2f}, {float(sim.cp_membrane_potential_v.max()):.2f}]")
print(f"  V has NaN: {bool(cp.isnan(sim.cp_membrane_potential_v).any())}")
print(f"  u has NaN: {bool(cp.isnan(sim.cp_recovery_variable_u).any())}")
print(f"  Weights: mean={float(sim.cp_connections.data.mean()):.4f}, NaN={bool(cp.isnan(sim.cp_connections.data).any())}, Inf={bool(cp.isinf(sim.cp_connections.data).any())}")
print(f"  g_e: mean={float(sim.cp_conductance_g_e.mean()):.6f}, NaN={bool(cp.isnan(sim.cp_conductance_g_e).any())}")
print(f"  g_i: mean={float(sim.cp_conductance_g_i.mean()):.6f}, NaN={bool(cp.isnan(sim.cp_conductance_g_i).any())}")
print(f"  ext_drive has NaN: {bool(cp.isnan(sim.cp_external_input_current).any())}")
stp_u_active = sim.cp_stp_u[:sim._synapse_count]
stp_x_active = sim.cp_stp_x[:sim._synapse_count]
print(f"  STP u: mean={float(stp_u_active.mean()):.4f}, NaN={bool(cp.isnan(stp_u_active).any())}, range=[{float(stp_u_active.min()):.4f}, {float(stp_u_active.max()):.4f}]")
print(f"  STP x: mean={float(stp_x_active.mean()):.4f}, NaN={bool(cp.isnan(stp_x_active).any())}, range=[{float(stp_x_active.min()):.4f}, {float(stp_x_active.max()):.4f}]")
eff_strength = sim.cp_connections.data * stp_u_active * stp_x_active
print(f"  Eff strength (w*u*x): NaN={bool(cp.isnan(eff_strength).any())}, Inf={bool(cp.isinf(eff_strength).any())}")
if sim.cp_eligibility_trace is not None:
    et = sim.cp_eligibility_trace[:sim._synapse_count]
    print(f"  Elig trace: mean={float(et.mean()):.6f}, NaN={bool(cp.isnan(et).any())}, max={float(et.max()):.6f}")

print("\n--- Running step 1 ---")
sim._run_one_simulation_step()
n_spikes_1 = int(sim.cp_firing_states.sum())
print(f"Step 1: spikes={n_spikes_1}")
print(f"  V: mean={float(cp.nanmean(sim.cp_membrane_potential_v)):.2f}, NaN count={int(cp.isnan(sim.cp_membrane_potential_v).sum())}/{sim.core_config.num_neurons}")
print(f"  u: NaN count={int(cp.isnan(sim.cp_recovery_variable_u).sum())}")
print(f"  Weights: NaN={bool(cp.isnan(sim.cp_connections.data).any())}, Inf={bool(cp.isinf(sim.cp_connections.data).any())}")
print(f"  g_e: mean={float(cp.nanmean(sim.cp_conductance_g_e)):.6f}, NaN={bool(cp.isnan(sim.cp_conductance_g_e).any())}")
print(f"  g_i: mean={float(cp.nanmean(sim.cp_conductance_g_i)):.6f}, NaN={bool(cp.isnan(sim.cp_conductance_g_i).any())}")
stp_u_active = sim.cp_stp_u[:sim._synapse_count]
stp_x_active = sim.cp_stp_x[:sim._synapse_count]
print(f"  STP u: NaN={bool(cp.isnan(stp_u_active).any())}, range=[{float(cp.nanmin(stp_u_active)):.4f}, {float(cp.nanmax(stp_u_active)):.4f}]")
print(f"  STP x: NaN={bool(cp.isnan(stp_x_active).any())}, range=[{float(cp.nanmin(stp_x_active)):.4f}, {float(cp.nanmax(stp_x_active)):.4f}]")

sim.clear_simulation_state_and_gpu_memory()
