"""Phase B -- the spiking similarity-matching cortex BRIDGE BUILDER + PPMI-shaped input encoder (Task 1).

This is the substrate for the learned-graded similarity-matching cortex: a 2-region bridge with a
PLASTIC, gated hub -> cortex projection, the dendritic divisive (/marginal) gain, STDP + homeostasis,
and the OU background DISABLED (so a concept's code reflects only the drive-driven firing, not a
spontaneous baseline that would swamp the per-hub co-occurrence marginal the gain normalizes by).

Two public functions:
  build_sm_cortex_bridge(n_hub, n_cortex, seed, ...) -> (bridge, hub_idx, cortex_idx)
  encode_drive(C_row, log=True) -> the Weber-Fechner log1p(max(C,0)) input compression (the /marginal +
      threshold are applied LATER by the bridge's dendritic gain + rheobase, NOT here).

Template: research/runners/dendritic_cortex_forward_codes_derisk._build_cortex_bridge -- same brain-region
framework + dendritic-gain + OU-off pattern; the DIFFERENCE is the plastic gated hub->cortex pathway +
STDP/homeostasis on + the soft-bound stdp_w_max raised above the design weight.

NO new sim/ edits (uses the existing brain-region framework + the Phase-1 dendritic gain).

Run (CPU smoke):
  SIM_BACKEND=numpy python -m research.runners.spiking_sm_cortex
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def build_sm_cortex_bridge(
    n_hub,
    n_cortex,
    seed,
    *,
    density=0.1,
    weight_mean=0.05,
    weight_jitter=0.02,
    sigma=0.05,
    alpha=0.05,
):
    """Build a 2-region similarity-matching cortex bridge.

    hub region   : n_hub excitatory neurons, internal_density 0 (a pure input layer).
    cortex region: n_cortex excitatory neurons, internal_density 0 (the learned read-out layer).
    pathway      : a PLASTIC hub -> cortex projection tagged plasticity_gate="hub_to_cortex" so it can be
                   frozen/thawed at runtime via bridge.set_plasticity_gate("hub_to_cortex", v).

    Enabled: the brain-region framework, the dendritic divisive (/marginal) gain, STDP, and homeostasis.
    The STDP soft-bound stdp_w_max is raised ABOVE the design weight so STDP does not collapse the weights
    (CLAUDE.md soft-bound gotcha). The OU background process is DISABLED.

    Returns (bridge, hub_idx, cortex_idx) where the index arrays are the per-region neuron index slices.
    """
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, GPUConfig, RuntimeState, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion(name="hub", n_neurons=n_hub, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="cortex", n_neurons=n_cortex, exc_fraction=1.0, internal_density=0.0),
    ]
    cfg.region_pathways = [
        RegionPathway(
            from_region="hub",
            to_region="cortex",
            density=density,
            weight_mean=weight_mean,
            weight_jitter=weight_jitter,
            plastic=True,
            plasticity_gate="hub_to_cortex",
        ),
    ]

    cfg.dt = 1.0
    cfg.seed = seed
    cfg.ou_seed = seed
    cfg.heterogeneity_seed = seed

    # the concept code must reflect the DRIVE-driven firing only -- spontaneous OU noise would give every
    # hub a baseline firing rate that swamps the per-hub co-occurrence marginal the gain normalizes by.
    cfg.enable_ou_process = False

    # the dendritic per-presynaptic-source divisive (/marginal) gain (D2 Phase 1): g = sigma/(sigma + a_src).
    cfg.enable_dendritic_divisive_gain = True
    cfg.dendritic_divisive_sigma = sigma
    cfg.dendritic_gain_ema_alpha = alpha

    # the learning rules: STDP (Hebbian timing rule) + homeostasis (intrinsic excitability regulation).
    cfg.enable_stdp = True
    cfg.enable_homeostasis = True
    # soft-bound STDP collapses weights when weight_mean > stdp_w_max (CLAUDE.md gotcha). Raise the cap
    # well above the design weight (weight_mean ~ 0.05) so STDP can grow/adjust without clipping.
    cfg.stdp_w_max = 30.0

    rt = RuntimeState()
    rt.actual_seed_used = seed

    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=rt,
        gpu_config=GPUConfig(),
    )
    bridge._initialize_simulation_data()

    hub_idx = np.asarray(bridge.region_manager.indices("hub"))
    cortex_idx = np.asarray(bridge.region_manager.indices("cortex"))
    return bridge, hub_idx, cortex_idx


def encode_drive(C_row, log=True):
    """PPMI-shaped input encoder: the Weber-Fechner input compression of a concept's count row.

    Returns log1p(max(C_row, 0)) when log=True (the perceptual compression of raw co-occurrence counts),
    else max(C_row, 0) (the clipped raw counts). Same length / dtype-float as the input.

    The /marginal normalization and the threshold are applied LATER by the bridge's dendritic divisive
    gain + the neuron rheobase -- NOT here.
    """
    arr = np.asarray(C_row, dtype=np.float64)
    clipped = np.maximum(arr, 0.0)
    if log:
        return np.log1p(clipped)
    return clipped


if __name__ == "__main__":
    # CPU smoke: build a tiny bridge + show the encoder.
    b, hub, cortex = build_sm_cortex_bridge(n_hub=200, n_cortex=64, seed=42)
    print(f"built sm-cortex bridge: n_hub={len(hub)} n_cortex={len(cortex)} "
          f"total_neurons={int(b.cp_membrane_potential_v.shape[0])}")
    print("encode_drive([0,1,3,7], log=True) =", encode_drive(np.array([0.0, 1.0, 3.0, 7.0])))
