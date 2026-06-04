"""Per-pathway multiplicative transmission gate (thalamocortical dynamical gating, 2026-06-03).

A transmission gate scales a pathway's effective synaptic CURRENT in [0,1] at runtime, WITHOUT changing
weights. Pre-wire a route with a fixed weight, hold it CLOSED (no current flows), and OPEN it on command:
binding = which gate is open, not which weight grew (Logiaco-Abbott-Escola 2021). These tests pin the
primitive: a closed gate blocks downstream firing; opening it lets the route drive the target; re-binding
(close one route, open another) reroutes with zero weight change.

Runs on whatever backend is active (use SIM_BACKEND=numpy for CPU CI).
"""
import numpy as np

from sim.regions import BrainRegion, RegionPathway


def _bridge(regions, pathways, seed=42):
    from sim import SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.enums import NeuronModel
    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = seed
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.ou_std_current_pA = 0.0
    cfg.enable_short_term_plasticity = False
    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False
    cfg.enable_structural_plasticity = False
    cfg.enable_reward_modulation = False
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                          runtime_state=RuntimeState(), gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    return sb


def _rate(sb, drive_idx, read_idx, n_steps=60, drive_pA=1500.0):
    from sim.backend import to_host
    sb.cp_external_input_current[:] = 0.0
    sb.cp_external_input_current[np.asarray(drive_idx)] = drive_pA
    acc = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    for _ in range(n_steps):
        sb._run_one_simulation_step()
        acc += to_host(sb.cp_firing_states).astype(np.float64)
    return float(acc[np.asarray(read_idx)].mean()) / n_steps


def _two_region_gated_bridge(gate="route", weight=300.0):
    regions = [BrainRegion(name="A", n_neurons=40, exc_fraction=1.0, internal_density=0.0),
               BrainRegion(name="B", n_neurons=40, exc_fraction=1.0, internal_density=0.0)]
    pathways = [RegionPathway(from_region="A", to_region="B", density=1.0, weight_mean=weight,
                              weight_jitter=0.0, plastic=False, transmission_gate=gate)]
    return _bridge(regions, pathways)


def test_transmission_gate_is_allocated():
    sb = _two_region_gated_bridge()
    assert sb.cp_transmission_gain is not None            # a transmission_gate pathway exists -> gain allocated
    assert "route" in sb._transmission_gate_to_synapses   # the named gate maps to its synapses
    assert len(sb._transmission_gate_to_synapses["route"]) > 0


def test_closed_gate_blocks_current_open_gate_allows():
    sb = _two_region_gated_bridge()
    a, b = sb.region_manager.indices("A"), sb.region_manager.indices("B")
    sb.set_transmission_gate("route", 0.0)               # CLOSED
    rate_closed = _rate(sb, a, b)
    sb.set_transmission_gate("route", 1.0)               # OPEN (no weight change, just the gate)
    rate_open = _rate(sb, a, b)
    assert rate_open > rate_closed                        # opening the gate lets A drive B
    assert rate_closed < 0.01                             # closed -> B silent (no current despite non-zero weight)
    assert rate_open > 0.05                               # open -> B driven


def test_rebinding_reroutes_without_weight_change():
    # the thalamocortical hypothesis in spikes: bind A->B by opening its gate; RE-BIND A->C by closing the
    # first gate and opening the second -- the SAME source reroutes to a different target with ZERO weight
    # change (a grown-weight model cannot: the A->B weight would persist). Binding = which gate is open.
    regions = [BrainRegion(name="A", n_neurons=40, exc_fraction=1.0, internal_density=0.0),
               BrainRegion(name="B", n_neurons=40, exc_fraction=1.0, internal_density=0.0),
               BrainRegion(name="C", n_neurons=40, exc_fraction=1.0, internal_density=0.0)]
    pathways = [RegionPathway(from_region="A", to_region="B", density=1.0, weight_mean=300.0,
                              weight_jitter=0.0, plastic=False, transmission_gate="r_AB"),
                RegionPathway(from_region="A", to_region="C", density=1.0, weight_mean=300.0,
                              weight_jitter=0.0, plastic=False, transmission_gate="r_AC")]
    sb = _bridge(regions, pathways)
    a = sb.region_manager.indices("A")
    b = sb.region_manager.indices("B")
    c = sb.region_manager.indices("C")
    weights_before = float(np.abs(__import__("sim.backend", fromlist=["to_host"]).to_host(sb.cp_connections.data)).sum())

    sb.set_transmission_gate("r_AB", 1.0)        # bind A -> B
    sb.set_transmission_gate("r_AC", 0.0)
    b1, c1 = _rate(sb, a, b), _rate(sb, a, c)
    assert b1 > 0.05 and c1 < 0.01               # A drives B, not C

    sb.set_transmission_gate("r_AB", 0.0)        # RE-BIND A -> C (no weight change, just the gates)
    sb.set_transmission_gate("r_AC", 1.0)
    b2, c2 = _rate(sb, a, b), _rate(sb, a, c)
    # A now drives C, not B -- rerouted on command. (B shows only a brief membrane transient as it settles
    # from its previous bound state, so assert C is the dominant target and B dropped, not a brittle b2~0.)
    assert c2 > 0.05                             # A now drives C
    assert c2 > 3 * b2                           # C is decisively the target (rerouted)
    assert b2 < b1                               # B dropped from its bound level

    weights_after = float(np.abs(__import__("sim.backend", fromlist=["to_host"]).to_host(sb.cp_connections.data)).sum())
    assert abs(weights_after - weights_before) < 1e-3   # the synaptic WEIGHTS never changed; only the gates did


def test_no_transmission_gate_means_no_gain_array():
    # a pathway WITHOUT a transmission_gate leaves cp_transmission_gain None (additive: zero overhead)
    regions = [BrainRegion(name="A", n_neurons=20, internal_density=0.0),
               BrainRegion(name="B", n_neurons=20, internal_density=0.0)]
    pathways = [RegionPathway(from_region="A", to_region="B", density=0.5, plastic=False)]
    sb = _bridge(regions, pathways)
    assert sb.cp_transmission_gain is None
