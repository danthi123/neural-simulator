"""Wiring helper that builds the G1 network topology.

Returns (CoreSimConfig, wiring_plan). The runner injects the explicit
wiring into SimulationBridge after `_initialize_simulation_data()`.

Topology:
    64 input neurons  (indices 0..63)      — Izhikevich RS, receive rate-coded Poisson
    4 output neurons (indices 64..67)      — Izhikevich RS, receive plastic i->o + fixed lat-inh
    256 plastic input→output synapses     — STDP, initial weights Uniform(0.05, 0.15)
    12 fixed output→output synapses       — lateral inhibition, magnitude 1.0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.config import CoreSimConfig
from sim.enums import NeuronModel


@dataclass
class G1NetworkSpec:
    n_input: int = 64
    n_output: int = 4

    init_weight_min: float = 0.05
    init_weight_max: float = 0.15
    weight_max_cap: float = 1.5
    lateral_inhibition_weight: float = 1.0

    @property
    def n_total(self):
        return self.n_input + self.n_output

    @property
    def input_indices(self):
        return list(range(0, self.n_input))

    @property
    def output_indices(self):
        return list(range(self.n_input, self.n_input + self.n_output))


def build_g1_network_config(seed, spec=None):
    """Produce the CoreSimConfig + explicit wiring plan for the G1 classifier.

    Callers are responsible for invoking `SimulationBridge.inject_explicit_wiring`
    after `_initialize_simulation_data` to overwrite auto-generated connectivity
    with this plan.
    """
    spec = spec or G1NetworkSpec()

    core_cfg = CoreSimConfig()
    core_cfg.num_neurons = spec.n_total
    core_cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    core_cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    core_cfg.seed = int(seed)
    core_cfg.dt_ms = 1.0
    core_cfg.connections_per_neuron = 0  # Overwritten by inject_explicit_wiring.
    core_cfg.num_traits = 2               # 0=excitatory inputs, 1=inhibitory (output lateral path)
    core_cfg.inhibitory_trait_indices = [1]

    # Plasticity
    core_cfg.enable_stdp = True
    core_cfg.enable_hebbian_learning = False
    core_cfg.enable_short_term_plasticity = False
    core_cfg.enable_structural_plasticity = False
    core_cfg.enable_homeostasis = True
    core_cfg.enable_reward_modulation = False
    core_cfg.enable_watts_strogatz = False

    core_cfg.stdp_a_plus = 0.012
    core_cfg.stdp_a_minus = 0.010
    core_cfg.stdp_w_min = 0.0
    core_cfg.stdp_w_max = spec.weight_max_cap

    # Weaker OU noise so the rate-coded Poisson input carries the signal.
    core_cfg.ou_std_current_pA = 30.0

    rng = np.random.default_rng(seed)

    # All-to-all input -> output
    pre_i2o, post_i2o = [], []
    for i in spec.input_indices:
        for o in spec.output_indices:
            pre_i2o.append(i)
            post_i2o.append(o)
    w_i2o = rng.uniform(spec.init_weight_min, spec.init_weight_max,
                        size=len(pre_i2o)).astype(np.float32)

    # All-to-all (minus self) output -> output lateral inhibition
    pre_lat, post_lat = [], []
    for a in spec.output_indices:
        for b in spec.output_indices:
            if a == b:
                continue
            pre_lat.append(a)
            post_lat.append(b)
    w_lat = np.full(len(pre_lat), spec.lateral_inhibition_weight, dtype=np.float32)

    wiring_plan = {
        "input_to_output": {
            "pre_indices": pre_i2o,
            "post_indices": post_i2o,
            "initial_weights": w_i2o,
            "plastic": True,
            "conn_type": "E_TO_E",
            "count": len(pre_i2o),
        },
        "output_lateral_inhibition": {
            "pre_indices": pre_lat,
            "post_indices": post_lat,
            "initial_weights": w_lat,
            "plastic": False,
            "conn_type": "I_TO_E",
            "count": len(pre_lat),
        },
        "spec": spec,
    }
    return core_cfg, wiring_plan
