"""Gap #5 — dendritic-plateau BISTABILITY probe: does a single two-compartment dAP neuron HOLD a plateau (a stable
high fixed point) after a transient partial cue, AND stay silent at rest? (The intrinsic bistability that resolves the
completion trilemma: magnitude vs bistability, since a strong point-neuron attractor self-sustains.)

Tests, per neuron (no recurrence -- pure single-cell dendritic dynamics):
  1. LATCH-AND-HOLD: drive the apical with a coincident cue for `cue_steps`, then REMOVE it and run `hold_steps` -> does
     v_apical STAY depolarized (plateau held)? A transient dAP decays; a bistable dendrite holds.
  2. SILENT REST: no cue -> v_apical stays at rest (no self-ignition).
  3. RESET: a hyperpolarizing (inhibitory) kick knocks it out of the plateau back to rest.

Baseline (current transient kernel) is expected to DECAY after cue removal (test 1 fails) -> that is the gap the
self-regenerating-conductance kernel change must close. GPU. Uses the existing enable_two_compartment_dap machinery.
"""
import os, sys
os.environ.setdefault("SIM_BACKEND", "cupy")
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
import numpy as np


def build(seed=42, n=64, k_thresh=6.0, plateau_strength=80.0, mg=1.0, apical_R=0.15, apical_gc=1.0,
          self_regen=0.0):
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion, RegionPathway
    regions = [BrainRegion(name="d", n_neurons=n, exc_fraction=1.0, internal_density=0.0,
                           exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False)]
    pathways = [RegionPathway(from_region="d", to_region="d", density=0.0, weight_mean=0.0, weight_jitter=0.0, plastic=False)]
    cfg = CoreSimConfig(); cfg.seed = cfg.heterogeneity_seed = cfg.ou_seed = int(seed)
    cfg.dt_ms = 1.0; cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions; cfg.region_pathways = pathways
    cfg.enable_stdp = False; cfg.enable_homeostasis = False; cfg.enable_hebbian_learning = False
    cfg.enable_short_term_plasticity = False; cfg.enable_ou_process = False
    cfg.enable_structural_plasticity = False; cfg.fast_spike_reset = True; cfg.enable_nmda = True
    cfg.enable_coincidence_detection = True; cfg.coincidence_weighted_drive = True
    cfg.coincidence_k_threshold = float(k_thresh); cfg.coincidence_plateau_strength = float(plateau_strength)
    cfg.nmda_mg_concentration = float(mg)
    cfg.enable_two_compartment_dap = True
    cfg.apical_R = float(apical_R); cfg.apical_g_couple = float(apical_gc)
    if hasattr(cfg, "coincidence_plateau_self_regen"):
        cfg.coincidence_plateau_self_regen = float(self_regen)   # the kernel-change knob (0 = current transient behavior)
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    return b, cfg


def probe(self_regen=0.0, cue_steps=40, hold_steps=120, cue_c_drive=20.0):
    """Drive the coincidence c_drive high for cue_steps (trigger the plateau), remove it, watch v_apical over hold_steps.
    NOTE: driving c_drive directly requires access to the coincidence path; here we inject a strong somatic current +
    a coincident-input proxy. Placeholder for the full probe once the self-regen kernel knob exists."""
    from sim.backend import to_host
    b, cfg = build(self_regen=self_regen)
    va0 = float(np.mean(np.asarray(to_host(b.cp_v_apical)))) if b.cp_v_apical is not None else float("nan")
    return {"self_regen": self_regen, "v_apical_rest": va0,
            "note": "harness scaffold; full latch-hold measurement wired once the self-regen kernel knob lands"}


if __name__ == "__main__":
    print(probe(self_regen=0.0))
