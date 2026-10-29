"""input_divisive_norm_external_only (step 0d, ruling after c2 step 0c): the divisive-norm stage divides ONLY the external
input current when the flag is on; default OFF is byte-identical to the pre-edit code.

GOLDEN_SHA is the sha256 of (firing, v) over a short seeded run of a divnorm-ON bridge WITH synaptic (excitatory and
inhibitory) input, computed on main BEFORE the sim/ edit (the pre-edit sim/ at commit 2932f1800 (git archive HEAD sim, run from that copy)). The flag-off path must reproduce it."""
import hashlib
import os

import numpy as np
import pytest

os.environ.setdefault("SIM_NO_PROVENANCE", "1")
os.environ.setdefault("SIM_BACKEND", "numpy")

GOLDEN_SHA = "ff430368196b5af2c4899052ba9b9d2f4e34b205ca8c630a4d00735747009f35"


def _bridge(external_only=None, V=4, seed=7):
    from sim import CoreSimConfig, SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
    from sim.enums import NeuronModel
    from sim.regions import BrainRegion, RegionPathway
    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = seed
    cfg.enable_brain_region_framework = True
    cfg.ou_std_current_pA = 0.0
    for flag in ("enable_short_term_plasticity", "enable_hebbian_learning", "enable_homeostasis",
                 "enable_structural_plasticity", "enable_reward_modulation", "enable_stdp"):
        setattr(cfg, flag, False)
    cfg.enable_input_divisive_norm = True
    cfg.input_divisive_sigma = 1.0
    cfg.input_divisive_gain = 0.001
    if external_only is not None:
        cfg.input_divisive_norm_external_only = external_only
    regions = [BrainRegion(name="w%d" % w, n_neurons=10, exc_fraction=1.0, internal_density=0.0,
                           input_divisive_norm=True) for w in range(V)]
    regions.append(BrainRegion(name="pinh", n_neurons=8, exc_fraction=0.0, internal_density=0.0))
    cfg.brain_regions = regions
    P = []
    for w in range(V):
        P.append(RegionPathway(from_region="w%d" % w, to_region="pinh", density=1.0, weight_mean=60.0,
                               weight_jitter=0.0, plastic=False))
        P.append(RegionPathway(from_region="pinh", to_region="w%d" % w, density=1.0, weight_mean=40.0,
                               weight_jitter=0.0, plastic=False))
    cfg.region_pathways = P
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(),
                          gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    return sb


def _run(sb, steps=25):
    from sim.backend import to_host, from_host
    n = sb.core_config.num_neurons
    cur = np.zeros(n)
    cur[0:10] = 6.0e4          # pool 0 (winner), pool 1, pool 2 driven; D = 1 + 0.001*mean ~ 26 -> ~2300 pA post-division
    cur[10:20] = 2.4e4
    cur[20:30] = 1.6e4
    cur = from_host(cur)
    h = hashlib.sha256()
    for _ in range(steps):
        sb.cp_external_input_current[:] = cur
        sb._run_one_simulation_step()
        h.update(np.asarray(to_host(sb.cp_firing_states)).astype(np.uint8).tobytes())
        h.update(np.asarray(to_host(sb.cp_membrane_potential_v)).astype(np.float32).tobytes())
    return h.hexdigest()


def test_flag_default_is_off():
    from sim import CoreSimConfig
    assert CoreSimConfig().input_divisive_norm_external_only is False


def test_flag_off_is_byte_identical_to_pre_edit_golden():
    assert _run(_bridge()) == GOLDEN_SHA
    assert _run(_bridge(external_only=False)) == GOLDEN_SHA


def test_flag_on_leaves_synaptic_current_undivided():
    """Same seeded run; flag on must change the trajectory (inhibition is no longer divided by D), and the divisor must be
    computed from the external drive only."""
    assert _run(_bridge(external_only=True)) != GOLDEN_SHA
    from sim.backend import to_host
    dv = {}
    for flag in (True, False):
        sb = _bridge(external_only=flag)
        n = sb.core_config.num_neurons
        v0 = float(np.asarray(to_host(sb.cp_membrane_potential_v))[:40].mean())
        sb.cp_conductance_g_i[:40] = 200.0          # a strong inhibitory conductance on the flagged pools, no external drive
        sb.cp_external_input_current[:] = 0.0
        sb._run_one_simulation_step()
        dv[flag] = float(np.asarray(to_host(sb.cp_membrane_potential_v))[:40].mean()) - v0
    assert dv[True] < -0.5                         # undivided inhibition hyperpolarizes
    assert dv[True] != dv[False]
