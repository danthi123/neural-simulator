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
    enable_hebbian_learning=False,
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
    # structural plasticity (synaptogenesis) is NOT part of the similarity-matching learn (the hub->cortex
    # topology is fixed; only the WEIGHTS learn) AND it triggers a pre-existing capacity bug on a plasticity-
    # gated pathway (cp_plasticity_rate_gain isn't resized when the synapse nnz grows -> IndexError at
    # bridge.py:6378, which silently corrupts the STDP update). Disable it for this build.
    cfg.enable_structural_plasticity = False
    # Hebbian learning carries a per-sub-step weight DECAY (~1e-5) that, over the 100K+ steps of Phase-B
    # training, collapses STDP-grown weights to the ~0.05 floor (CLAUDE.md text-IO fix #1; every g* / Tier-1
    # learning recipe sets this False and relies on STDP). Default OFF so the learned hub->cortex weights
    # survive extended training; opt-in only.
    cfg.enable_hebbian_learning = enable_hebbian_learning
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


def _set_hub_drive(bridge, hub_idx, drive_row, drive_scale):
    """Set bridge.cp_external_input_current on hub_idx to drive_row * drive_scale (zero elsewhere).

    Handles the cupy/numpy split: on cupy the per-index assignment needs an xp array (mirrors
    dendritic_cortex_forward_codes_derisk._present)."""
    from sim.backend import get_backend

    xp, _ = get_backend()
    hub_idx = np.asarray(hub_idx)
    drive = (np.asarray(drive_row, dtype=np.float64) * float(drive_scale)).astype(np.float32)
    bridge.cp_external_input_current[:] = 0.0
    is_cupy = type(bridge.cp_external_input_current).__module__.startswith("cupy")
    if is_cupy:
        bridge.cp_external_input_current[xp.asarray(hub_idx)] = xp.asarray(drive)
    else:
        bridge.cp_external_input_current[hub_idx] = drive


def train_sm_cortex(
    bridge,
    C_drive,
    hub_idx,
    cortex_idx,
    *,
    n_epochs=2,
    drive_scale=12.0,
    window=20,
    settle=6,
):
    """Train the plastic hub->cortex projection by presenting each concept's drive (plasticity ON).

    For each epoch, for each concept row i: drive the hub neurons with C_drive[i] * drive_scale, run
    settle + window steps so the cortex fires and the plastic hub->cortex STDP potentiates co-active
    (hub, cortex) pairs. The external current is zeroed at the end of each presentation.

    The hub_to_cortex plasticity gate is left at its default (1.0 = full plasticity) -- STDP runs.
    Returns the bridge (mutated in place).
    """
    from sim.backend import to_host  # noqa: F401  (re-export availability check; used by callers)

    Nc = int(np.asarray(C_drive).shape[0])
    for _ in range(int(n_epochs)):
        for i in range(Nc):
            _set_hub_drive(bridge, hub_idx, C_drive[i], drive_scale)
            for _t in range(int(settle) + int(window)):
                bridge._run_one_simulation_step()
            bridge.cp_external_input_current[:] = 0.0
    return bridge


def read_codes(
    bridge,
    C_drive,
    hub_idx,
    cortex_idx,
    *,
    drive_scale=12.0,
    window=20,
    settle=6,
):
    """Read the per-concept cortex spike-count codes with plasticity FROZEN.

    Freezes the hub->cortex pathway (set_plasticity_gate("hub_to_cortex", 0.0)) so the read does not
    perturb the learned weights, then for each concept presents its drive (same protocol as training)
    and accumulates the cortex region's SPIKE COUNTS over the `window` steps (summing
    bridge.cp_firing_states[cortex_idx] each step after `settle`). Restores the gate to 1.0 afterward.

    Returns an [Nc x n_cortex] numpy array of per-concept cortex spike-count codes.
    """
    from sim.backend import to_host

    cortex_idx = np.asarray(cortex_idx)
    Nc = int(np.asarray(C_drive).shape[0])
    codes = np.zeros((Nc, cortex_idx.size), dtype=np.float64)

    bridge.set_plasticity_gate("hub_to_cortex", 0.0)
    try:
        for i in range(Nc):
            _set_hub_drive(bridge, hub_idx, C_drive[i], drive_scale)
            acc = np.zeros(cortex_idx.size, dtype=np.float64)
            for t in range(int(settle) + int(window)):
                bridge._run_one_simulation_step()
                if t >= int(settle):
                    fired = np.asarray(to_host(bridge.cp_firing_states))[cortex_idx]
                    acc += fired.astype(np.float64)
            codes[i] = acc
            bridge.cp_external_input_current[:] = 0.0
    finally:
        bridge.set_plasticity_gate("hub_to_cortex", 1.0)
    return codes


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
