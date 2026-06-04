"""Cleanest possible test: does the brain-region framework support INHIBITION at all?

Two regions, default Izhikevich type (no pacemaker / neuron-type confounds):
  - src: all-inhibitory (exc_fraction=0.0) -> its outgoing synapses should be GABAergic.
  - tgt: excitable (exc_fraction=1.0), held firing by a tonic current.
  - pathway src -> tgt.
Drive src ON vs OFF. If framework inhibition works, tgt firing DROPS when src fires. If tgt firing RISES or
is unchanged, the framework's region-pathway inhibition does not actually inhibit -- a core finding that
explains why a genuine BG disinhibition cascade (cheat-removal #2) cannot be built on region pathways as-is.

  SIM_BACKEND=numpy python -m research.runners._framework_inhibition_minimal_probe
"""
import numpy as np

from sim.regions import BrainRegion, RegionPathway


def _build(seed=42, n=40, w=300.0):
    from sim import SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.enums import NeuronModel
    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = seed
    cfg.enable_brain_region_framework = True
    cfg.ou_std_current_pA = 0.0
    for flag in ("enable_short_term_plasticity", "enable_hebbian_learning", "enable_homeostasis",
                 "enable_structural_plasticity", "enable_reward_modulation"):
        setattr(cfg, flag, False)
    cfg.brain_regions = [
        BrainRegion(name="src", n_neurons=n, exc_fraction=0.0, internal_density=0.0),   # all inhibitory
        BrainRegion(name="tgt", n_neurons=n, exc_fraction=1.0, internal_density=0.0),   # excitable
    ]
    cfg.region_pathways = [
        RegionPathway(from_region="src", to_region="tgt", density=1.0, weight_mean=w,
                      weight_jitter=0.0, plastic=False),
    ]
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                          runtime_state=RuntimeState(), gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    return sb


def _tgt_rate(sb, drive_src, tgt_tonic=400.0, src_pA=1500.0, settle=80):
    from sim.backend import to_host
    sb.cp_external_input_current[:] = 0.0
    sb.cp_external_input_current[np.asarray(sb.region_manager.indices("tgt"))] = tgt_tonic
    if drive_src:
        sb.cp_external_input_current[np.asarray(sb.region_manager.indices("src"))] = src_pA
    acc = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    for _ in range(settle):
        sb._run_one_simulation_step()
        acc += to_host(sb.cp_firing_states).astype(np.float64)
    tgt = acc[np.asarray(sb.region_manager.indices("tgt"))].mean() / settle
    src = acc[np.asarray(sb.region_manager.indices("src"))].mean() / settle
    return tgt, src


def main():
    print("=== Minimal framework inhibition test: does src(inhibitory) -> tgt actually inhibit? ===\n", flush=True)
    sb = _build(seed=42)
    off, _ = _tgt_rate(sb, drive_src=False)
    on, src_rate = _tgt_rate(sb, drive_src=True)
    print(f"  tgt rate (src OFF) = {off:.3f}", flush=True)
    print(f"  tgt rate (src ON, src rate={src_rate:.3f}) = {on:.3f}", flush=True)
    if on < off - 0.02:
        verdict = "INHIBITION WORKS (tgt drops when inhibitory src fires)"
    elif on > off + 0.02:
        verdict = "INHIBITION INVERTED -> EXCITATORY (tgt RISES) -- framework bug for region-pathway inhibition"
    else:
        verdict = "NO EFFECT (inhibition not applied)"
    print(f"\n  => {verdict}", flush=True)


if __name__ == "__main__":
    main()
