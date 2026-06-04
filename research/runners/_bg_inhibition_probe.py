"""Decisive probe: does the VALIDATED g11_bg cascade (build_bg_brain_regions) actually produce GPi-silencing
(genuine D1 -| GPi -| thal disinhibition) in the brain-region framework?

My minimal hand-built cascade (gated_compose_bg_genuine_demo) used the SAME framework and D1->GPi came out
EXCITATORY (gpi rose when D1 fired). The region framework makes every cross-region pathway E_TO_MIX/positive;
inhibition is applied from the presynaptic neuron's *trait* (set via output_inhibitory_indices -> cp_traits=1).
g11_bg navigates successfully, which REQUIRES the cascade to disinhibit -- so either (a) g11_bg's cascade works
(framework inhibition is fine; my minimal config hit a specific issue) or (b) g11_bg selects actions some other
way. This probe settles it: build the default BG, drive str_D1_N directly (the striatal selection signal),
and measure whether gpi_N DROPS and thal_N RISES relative to a no-drive baseline. A clean drop+rise => reuse is
the genuine-disinhibition path (cheat-removal #2); a non-drop => a systemic framework finding.

  SIM_BACKEND=numpy python -m research.runners._bg_inhibition_probe
"""
import numpy as np

from research.runners.g11_bg_runner import build_bg_brain_regions, ACTION_NAMES


def _build_bridge(seed=42):
    from sim import SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.enums import NeuronModel
    regions, pathways = build_bg_brain_regions()   # defaults: the validated navigation cascade
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
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                          runtime_state=RuntimeState(), gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    return sb


def _rates(sb, d1_drive_action=None, d1_pA=1500.0, settle=80, watch=("gpi_N", "thal_N", "gpi_E", "thal_E")):
    """Drive str_D1_<action> directly (the striatal 'go' signal), settle, return mean firing per watched region."""
    from sim.backend import to_host
    sb.cp_external_input_current[:] = 0.0
    if d1_drive_action is not None:
        idx = np.asarray(sb.region_manager.indices(f"str_D1_{d1_drive_action}"))
        sb.cp_external_input_current[idx] = d1_pA
    acc = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    for _ in range(settle):
        sb._run_one_simulation_step()
        acc += to_host(sb.cp_firing_states).astype(np.float64)
    return {nm: acc[np.asarray(sb.region_manager.indices(nm))].mean() / settle for nm in watch}


def main():
    print("=== Probe: does build_bg_brain_regions produce genuine D1 -| GPi -| thal disinhibition? ===\n", flush=True)
    sb = _build_bridge(seed=42)
    base = _rates(sb, d1_drive_action=None)
    sel = _rates(sb, d1_drive_action="N")     # drive striatal D1 for action N
    print(f"  baseline (no D1 drive):  gpi_N={base['gpi_N']:.3f} thal_N={base['thal_N']:.3f}  "
          f"gpi_E={base['gpi_E']:.3f} thal_E={base['thal_E']:.3f}", flush=True)
    print(f"  drive str_D1_N:          gpi_N={sel['gpi_N']:.3f} thal_N={sel['thal_N']:.3f}  "
          f"gpi_E={sel['gpi_E']:.3f} thal_E={sel['thal_E']:.3f}", flush=True)
    d1_silences = sel['gpi_N'] < base['gpi_N'] - 0.02
    thal_released = sel['thal_N'] > base['thal_N'] + 0.03
    other_gpi_stays = abs(sel['gpi_E'] - base['gpi_E']) < 0.05
    verdict = "GENUINE DISINHIBITION (reuse is the path)" if (d1_silences and thal_released) \
        else "NO GPi-SILENCING (systemic framework finding)"
    print(f"\n  -> D1 silences its GPi: {d1_silences}   thal_N released: {thal_released}   "
          f"other channel unchanged: {other_gpi_stays}", flush=True)
    print(f"  => {verdict}", flush=True)


if __name__ == "__main__":
    main()
