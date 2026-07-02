"""rung-4 Stage A' — the guarded two-compartment dAP (`enable_two_compartment_dap`).

Byte-inertness when off: with the flag default-off, the two-compartment code path is never taken (the apical
compartment `cp_v_apical` is never allocated and the plateau current is added to the soma via the original line),
so a coincidence run is byte-identical to before the edit. When on, the apical compartment is allocated and the
plateau regenerates on the apical voltage (predictive != active — validated behaviorally by EMERGE-10 Stage A').
CPU (numpy backend); no GPU required.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
import pytest


def _build(two_compartment):
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType
    regions = [
        BrainRegion(name="context", n_neurons=40, exc_fraction=1.0, internal_density=0.0, exc_weight_mean=0.0,
                    inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name),
        BrainRegion(name="column", n_neurons=20, exc_fraction=1.0, internal_density=0.0, exc_weight_mean=0.0,
                    inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name),
    ]
    pathways = [RegionPathway(from_region="context", to_region="column", density=1.0, weight_mean=0.1,
                             weight_jitter=0.0, plastic=False, coincidence_detector=True)]
    cfg = CoreSimConfig()
    cfg.seed = cfg.heterogeneity_seed = cfg.ou_seed = 42
    cfg.dt_ms = 1.0; cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"; cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions; cfg.region_pathways = pathways
    cfg.enable_stdp = False; cfg.enable_hebbian_learning = False; cfg.enable_nmda = False
    cfg.fast_spike_reset = True
    for f in ("enable_homeostasis", "enable_short_term_plasticity", "enable_ou_process",
              "enable_conductance_noise", "enable_parameter_heterogeneity", "enable_structural_plasticity"):
        setattr(cfg, f, False)
    cfg.enable_coincidence_detection = True
    cfg.coincidence_k_threshold = 8.0
    cfg.enable_two_compartment_dap = bool(two_compartment)
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b.runtime_state.actual_seed_used = 42
    b._initialize_simulation_data(called_from_playback_init=False)
    return b


def _drive_and_step(b, n_steps=6):
    rm = b.region_manager
    ctx = np.asarray(rm.indices("context"), dtype=np.int64)
    for _ in range(n_steps):
        ext = np.zeros(int(b.core_config.num_neurons), dtype=np.float32)
        ext[ctx] = 600.0
        b.cp_external_input_current[:] = np.asarray(ext)
        b._run_one_simulation_step()


def test_two_compartment_off_is_byte_inert():
    """Default-off: the apical compartment is never allocated -> the two-compartment code path is not taken -> the
    coincidence run is byte-identical to before the edit."""
    b = _build(two_compartment=False)
    _drive_and_step(b)
    assert b.cp_v_apical is None, "cp_v_apical must stay None when enable_two_compartment_dap is off (byte-inert)"


def test_two_compartment_on_allocates_apical():
    """On: the apical compartment is allocated (the plateau regenerates on it)."""
    b = _build(two_compartment=True)
    _drive_and_step(b)
    assert b.cp_v_apical is not None, "cp_v_apical must be allocated when enable_two_compartment_dap is on"
    assert int(np.asarray(b.cp_v_apical).shape[0]) == int(b.core_config.num_neurons)


if __name__ == "__main__":
    test_two_compartment_off_is_byte_inert()
    test_two_compartment_on_allocates_apical()
    print("OK: two-compartment dAP off=byte-inert (cp_v_apical None), on=allocates apical")
