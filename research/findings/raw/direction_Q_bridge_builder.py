# research/findings/raw/direction_Q_bridge_builder.py
"""Direction Q standalone test bridge builder.

Builds a fresh SimulationBridge with ONLY dlpfc_wm at the
target scale + a small stimulus input region. Isolates the
Wang 2002 NMDA persistence mechanism from any other substrate
component.

Reuses validated infrastructure byte-unchanged:
- BrainRegion + RegionPathway framework
- NMDA kernel (fused_nmda_update_and_current)
- IZH2007_HIPPO_PYRAMIDAL preset (Direction I baseline; PFC-style)
"""
from __future__ import annotations


def build_q_test_bridge(seed: int, n_dlpfc: int = 1000,
                          dlpfc_density: float = 0.10,
                          n_stim: int = 200,
                          enable_nmda: bool = True,
                          inh_weight_mean: float = 4.0,
                          verbose: bool = False):
    """Construct a standalone Direction Q test bridge.

    Args:
        seed: RNG seed
        n_dlpfc: dlpfc_wm region size (Direction I used 60; Q uses 1000+)
        dlpfc_density: internal recurrent density (Wang 2002 ~0.20;
                       0.10 is conservative starting point)
        n_stim: stimulus input region size
        enable_nmda: NMDA on/off (False = AMPA-only control)
        inh_weight_mean: dlpfc_wm internal inhibitory weight magnitude.
                         Default 4.0 preserves the prior Q-prime
                         scaling-envelope behavior (inh:exc = 2:1).
                         Direction Q-secondary E/I balance sweep varies
                         this parameter to test whether the substrate's
                         inhibition-dominance throttles the recurrent
                         attractor formation; smaller values (2.0, 3.0)
                         relax inhibition toward parity with excitation
                         (exc_weight_mean=2.0 is held fixed).
        verbose: print build info
    """
    from sim.config import (CoreSimConfig, VisualizationConfig,
                              RuntimeState, GPUConfig)
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronType

    regions = [
        BrainRegion(
            name="dlpfc_wm",
            n_neurons=n_dlpfc,
            exc_fraction=0.8,
            internal_density=dlpfc_density,
            exc_weight_mean=2.0,
            inh_weight_mean=float(inh_weight_mean),
            weight_jitter=0.2,
            plastic_internal=False,  # frozen for test (no learning)
            izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name,
            enable_nmda=enable_nmda,
        ),
        BrainRegion(
            name="q_stim_input",
            n_neurons=n_stim,
            exc_fraction=1.0,
            internal_density=0.0,
            exc_weight_mean=0.0,
            inh_weight_mean=0.0,
            weight_jitter=0.0,
            plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ),
    ]

    pathways = [
        RegionPathway(
            from_region="q_stim_input",
            to_region="dlpfc_wm",
            density=0.10,
            weight_mean=3.0,
            weight_jitter=0.3,
            plastic=False,
        ),
    ]

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5
    cfg.seed = seed
    cfg.enable_nmda = enable_nmda
    cfg.nmda_tau_decay = 100.0  # Wang 2002 calibration
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_short_term_plasticity = False
    cfg.fast_spike_reset = True

    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)

    if verbose:
        print("[BUILD-Q] dlpfc_wm n=" + str(n_dlpfc)
              + " density=" + str(dlpfc_density)
              + " inh_w=" + str(inh_weight_mean)
              + " NMDA=" + str(enable_nmda)
              + " stim_input n=" + str(n_stim), flush=True)
    return bridge
