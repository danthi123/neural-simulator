"""G11: Basal-ganglia-style action selection module.

Phase B follow-up to the silent-motor trap arc (Sessions G/H/I, all NEGATIVE).
The trap was diagnosed (V6) as a *reservoir-state bias problem* — random
hidden->motor weights on a shared reservoir naturally favor whichever motor
the input pattern happens to align with. Argmax + reservoir bias = lock-in.

Phase B fix (architectural): replace the shared-reservoir + argmax-readout
with a real basal-ganglia-style circuit. Each motor has its own dedicated
striatum_D1, striatum_D2, GPi, thalamus, and motor populations. Lateral
inhibition between motor populations provides structural winner-take-all
(no shared spike count to bias).

Architecture:
    cortex ─-> str_D1[N,E,S,W]    str_D2[N,E,S,W]
                  │                     │
              direct path          indirect path
                  v                     v
              GPi[N,E,S,W] <-── STN <-── GPe[N,E,S,W]
                  │
                  v (disinhibition)
              thal[N,E,S,W]
                  │
                  v
              motor[N,E,S,W]   (lateral inhibition between)

DA modulation: VTA/SNc DA neurons project to all striatal pools. DA enhances
direct pathway (D1+ sensitivity) and suppresses indirect pathway (D2-).

Built on validated Phase A presets:
- IZH2007_STRIATAL_MSN_D1 / D2 (rest=-80 mV down-state, fires when driven)
- IZH2007_GPE_PACEMAKER, IZH2007_GPI_OUTPUT (high tonic rates)
- IZH2007_STN_BURST (autonomous + scales with input)
- IZH2007_THALAMIC_RELAY (tonic mode)
- IZH2007_RS_CORTICAL_PYRAMIDAL, IZH2007_FS_CORTICAL_INTERNEURON (cortex)
- IZH2007_DOPAMINE (slow tonic + phasic)

Reference: Frank 2005 J Neurosci; Schroll & Hamker 2013 Front Comp Neurosci.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np


ACTION_NAMES = ["N", "E", "S", "W"]
N_ACTIONS = 4


def build_bg_brain_regions(
    n_cortex: int = 100,
    n_striatum_per_action: int = 50,
    n_gpe_per_action: int = 10,
    n_gpi_per_action: int = 10,
    n_stn: int = 20,
    n_thal_per_action: int = 10,
    n_motor_per_action: int = 10,
    n_motor_fs_per_action: int = 5,
    n_dopamine: int = 10,
    enable_motor_lateral_inhibition: bool = False,
    # WTA defaults validated 2026-04-25 on probe_bg_wta_ambiguous: under equal
    # cortex_N/cortex_E drive, asymmetry 1.06x → 1.77x with these weights.
    # Lower values (10/5) leave FS pool subthreshold and inhibition is dead.
    motor_to_fs_weight: float = 50.0,
    fs_to_motor_weight: float = 20.0,
    # Real perception (option #3 in Phase B follow-up): replace heuristic
    # cortex drive with a learned sensory→cortex mapping. Adds a 49-neuron
    # sensory layer tuned to (dx, dy) ∈ [-3, 3]² relative-position pairs.
    # Plastic sensory→cortex pathways must learn position-to-action mapping
    # via STDP+reward.
    enable_learned_perception: bool = False,
    n_sensory: int = 49,  # 7×7 grid of (dx, dy)-tuned neurons
    sensory_to_cortex_weight: float = 10.0,
    # Hippocampal module (option #1 in Phase B follow-up): adds place cells and
    # goal cells, both Gaussian-tuned (sparse). Plastic place+goal → cortex
    # pathways let the agent learn spatial→action associations. Replaces
    # heuristic cortex drive when enabled. Sparse encoding (σ=0.5) avoids
    # cascade saturation that broke earlier dense-encoding attempts.
    enable_hippocampus: bool = False,
    n_hippocampus_per_layer: int = 64,  # 8×8 grid place + 8×8 grid goal cells
    hippocampus_to_cortex_weight: float = 10.0,
):
    """Returns list of BrainRegion + list of RegionPathway for the BG circuit.

    When `enable_motor_lateral_inhibition=True`, adds 4 motor_FS_X regions
    (FS interneuron sub-pools, exc_fraction=0.0) plus pathways:
      - motor_X → motor_FS_X (excitatory; motor's own activity drives its FS)
      - motor_FS_X → motor_Y for Y != X (inhibitory; FS suppresses other motors)
    This creates standard cortical winner-take-all microcircuit dynamics:
    when motor_X fires, motor_FS_X fires, suppressing motor_{Y,Z,W}.
    """
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronType

    regions = []
    pathways = []

    # Hippocampal module (opt-in): place + goal cells with sparse Gaussian tuning.
    # Place cells encode agent (x, y), goal cells encode goal (gx, gy). Both
    # project plastically to all 4 cortex pools so the agent can learn
    # (place, goal) → action associations via STDP+reward.
    # Sparse encoding (σ=0.5 in runner): only 1-3 cells fire per position →
    # avoids cascade saturation that broke previous dense sensory encoding.
    if enable_hippocampus:
        regions.append(BrainRegion(
            name="place_cells",
            n_neurons=n_hippocampus_per_layer,
            exc_fraction=1.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name,
        ))
        regions.append(BrainRegion(
            name="goal_cells",
            n_neurons=n_hippocampus_per_layer,
            exc_fraction=1.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name,
        ))

    # Sensory layer (opt-in): position-tuned input neurons feeding cortex.
    # Replaces heuristic cortex drive when enable_learned_perception=True.
    # Each sensory neuron is tuned to a relative-position (dx, dy) ∈ [-3, 3]².
    # 7×7 grid = 49 neurons. The runner sets per-step drive based on goal offset.
    if enable_learned_perception:
        regions.append(BrainRegion(
            name="sensory",
            n_neurons=n_sensory,
            exc_fraction=1.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))

    # Cortex (input layer for goal-directed signals).
    # Split into per-action pools so different inputs preferentially activate
    # different actions. This is a phenomenological substitute for what
    # learning would produce: differential cortex→striatum weights.
    n_cortex_per_action = n_cortex // N_ACTIONS
    for action in ACTION_NAMES:
        regions.append(BrainRegion(
            name=f"cortex_{action}",
            n_neurons=n_cortex_per_action,
            exc_fraction=1.0,  # All excitatory for cortex inputs
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))

    # Per-action striatal pools (D1 direct, D2 indirect).
    # internal_density=0 (no lateral inhibition) initially — MSNs need
    # strong cortex drive to escape the down-state and lateral inhibition
    # makes that even harder. Add it back later if action selection needs
    # sharpening.
    for action in ACTION_NAMES:
        regions.append(BrainRegion(
            name=f"str_D1_{action}",
            n_neurons=n_striatum_per_action,
            exc_fraction=0.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D1.name,
        ))
        regions.append(BrainRegion(
            name=f"str_D2_{action}",
            n_neurons=n_striatum_per_action,
            exc_fraction=0.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D2.name,
        ))

    # Per-action BG output (GPe / GPi)
    for action in ACTION_NAMES:
        regions.append(BrainRegion(
            name=f"gpe_{action}",
            n_neurons=n_gpe_per_action,
            exc_fraction=0.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_GPE_PACEMAKER.name,
        ))
        regions.append(BrainRegion(
            name=f"gpi_{action}",
            n_neurons=n_gpi_per_action,
            exc_fraction=0.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_GPI_OUTPUT.name,
        ))

    # Single STN (excitatory, projects diffusely to all GPi)
    regions.append(BrainRegion(
        name="stn",
        n_neurons=n_stn,
        exc_fraction=1.0,  # STN is glutamatergic (excitatory)
        internal_density=0.0,
        exc_weight_mean=0.0, inh_weight_mean=0.0,
        weight_jitter=0.0, plastic_internal=False,
        izh_neuron_type=NeuronType.IZH2007_STN_BURST.name,
    ))

    # Per-action thalamic relay + motor cortex
    for action in ACTION_NAMES:
        regions.append(BrainRegion(
            name=f"thal_{action}",
            n_neurons=n_thal_per_action,
            exc_fraction=1.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_THALAMIC_RELAY.name,
        ))
        regions.append(BrainRegion(
            name=f"motor_{action}",
            n_neurons=n_motor_per_action,
            exc_fraction=1.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))

    # Dopamine neurons (single pool, broadcasts via neuromodulator subsystem)
    regions.append(BrainRegion(
        name="dopamine",
        n_neurons=n_dopamine,
        exc_fraction=1.0,
        internal_density=0.0,
        exc_weight_mean=0.0, inh_weight_mean=0.0,
        weight_jitter=0.0, plastic_internal=False,
        izh_neuron_type=NeuronType.IZH2007_DOPAMINE.name,
    ))

    # ---- Pathways (cross-region projections) ----

    # Sensory → cortex (LEARNING site for perception, opt-in).
    # Plastic; agent learns position-to-action mapping via STDP + reward.
    # Each sensory neuron projects to all 4 cortex pools; learning shapes
    # which sensory patterns drive which cortex action pool.
    if enable_learned_perception:
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region="sensory", to_region=f"cortex_{action}",
                density=1.0, weight_mean=sensory_to_cortex_weight,
                weight_jitter=0.2, plastic=True,
            ))

    # Hippocampus → cortex (LEARNING site, opt-in).
    # Plastic; agent learns (place, goal) → action via STDP + reward.
    # Place cells provide spatial context (where am I), goal cells provide
    # task context (where do I want to be). Together they should learn
    # the full position-action mapping.
    if enable_hippocampus:
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region="place_cells", to_region=f"cortex_{action}",
                density=1.0, weight_mean=hippocampus_to_cortex_weight,
                weight_jitter=0.2, plastic=True,
            ))
            pathways.append(RegionPathway(
                from_region="goal_cells", to_region=f"cortex_{action}",
                density=1.0, weight_mean=hippocampus_to_cortex_weight,
                weight_jitter=0.2, plastic=True,
            ))

    # Cortex -> striatum (LEARNING site).
    # Each cortex_X projects strongly to its corresponding str_D1_X / str_D2_X
    # AND weakly to other actions' striatum (cross-projection allows learning
    # to redistribute action representations on goal change).
    for cortex_action in ACTION_NAMES:
        for str_action in ACTION_NAMES:
            same = (cortex_action == str_action)
            # Eliminate cross-projections to avoid confused multi-cortex drive.
            # Each cortex_X projects ONLY to str_D1_X / str_D2_X.
            # Learning-based redistribution can come later.
            if not same:
                continue
            density = 1.0
            weight = 25.0
            pathways.append(RegionPathway(
                from_region=f"cortex_{cortex_action}",
                to_region=f"str_D1_{str_action}",
                density=density, weight_mean=weight, weight_jitter=0.2, plastic=True,
            ))
            pathways.append(RegionPathway(
                from_region=f"cortex_{cortex_action}",
                to_region=f"str_D2_{str_action}",
                density=density, weight_mean=weight, weight_jitter=0.2, plastic=True,
            ))

    # Direct pathway: D1 -> GPi (inhibitory). Strong weight needed to overcome
    # GPi tonic firing (~30-75 Hz baseline).
    for action in ACTION_NAMES:
        pathways.append(RegionPathway(
            from_region=f"str_D1_{action}", to_region=f"gpi_{action}",
            density=1.0, weight_mean=15.0, weight_jitter=0.2, plastic=False,
        ))

    # Indirect pathway: D2 -> GPe -> STN -> GPi
    for action in ACTION_NAMES:
        pathways.append(RegionPathway(
            from_region=f"str_D2_{action}", to_region=f"gpe_{action}",
            density=0.6, weight_mean=2.5, weight_jitter=0.2, plastic=False,
        ))
        pathways.append(RegionPathway(
            from_region=f"gpe_{action}", to_region="stn",
            density=0.3, weight_mean=1.5, weight_jitter=0.2, plastic=False,
        ))

    # STN -> all GPi (diffuse excitation; this is the "hyperdirect"-like
    # contribution that biases against premature action selection)
    for action in ACTION_NAMES:
        pathways.append(RegionPathway(
            from_region="stn", to_region=f"gpi_{action}",
            density=0.4, weight_mean=1.0, weight_jitter=0.2, plastic=False,
        ))

    # GPi -> thalamus (inhibitory). Strong weight + density needed so
    # GPi tonic firing fully suppresses thal, AND so D1-mediated GPi
    # silence cleanly releases the gate.
    for action in ACTION_NAMES:
        pathways.append(RegionPathway(
            from_region=f"gpi_{action}", to_region=f"thal_{action}",
            density=1.0, weight_mean=8.0, weight_jitter=0.2, plastic=False,
        ))

    # Thalamus -> motor cortex (excitatory). Very strong weight needed
    # because thal pool is small (10 cells) and we need ~50 Hz motor output
    # from ~24 Hz thal input.
    for action in ACTION_NAMES:
        pathways.append(RegionPathway(
            from_region=f"thal_{action}", to_region=f"motor_{action}",
            density=1.0, weight_mean=20.0, weight_jitter=0.2, plastic=False,
        ))

    # ---- Motor lateral inhibition (opt-in) ----
    # FS interneuron sub-pool per motor pool. Each motor_X drives its own
    # motor_FS_X (excitatory), which in turn inhibits the other 3 motor pools.
    # This implements the cortical WTA microcircuit: when motor_X fires,
    # motor_FS_X fires, suppressing motor_{Y,Z,W}. Combined with BG gating,
    # this should sharpen action selection in cases where multiple cortex
    # pools drive simultaneously (currently the dominant random-fallback case).
    if enable_motor_lateral_inhibition:
        for action in ACTION_NAMES:
            regions.append(BrainRegion(
                name=f"motor_FS_{action}",
                n_neurons=n_motor_fs_per_action,
                exc_fraction=0.0,  # all-inhibitory → outgoing synapses are inhibitory
                internal_density=0.0,
                exc_weight_mean=0.0, inh_weight_mean=0.0,
                weight_jitter=0.0, plastic_internal=False,
                izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name,
            ))

        # motor_X → motor_FS_X (excitatory drive — motor's own activity drives its FS)
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region=f"motor_{action}", to_region=f"motor_FS_{action}",
                density=1.0, weight_mean=motor_to_fs_weight, weight_jitter=0.2,
                plastic=False,
            ))

        # motor_FS_X → motor_Y for Y != X (inhibitory cross-pool suppression)
        for src_action in ACTION_NAMES:
            for tgt_action in ACTION_NAMES:
                if src_action == tgt_action:
                    continue
                pathways.append(RegionPathway(
                    from_region=f"motor_FS_{src_action}", to_region=f"motor_{tgt_action}",
                    density=1.0, weight_mean=fs_to_motor_weight, weight_jitter=0.2,
                    plastic=False,
                ))

    return regions, pathways


def _position_to_cortex_drive(x, y, n_cortex_per_action, grid_size,
                                rate_peak=400.0, rate_floor=50.0, sigma=1.5):
    """Map (x,y) position to per-action cortex drive amplitudes.

    Each action's cortex pool gets a baseline + position-dependent component.
    For now: uniform baseline drive to all 4 cortex pools (the differential
    selectivity comes from learning the cortex→striatum weights, not from
    input encoding).

    Returns a dict {action: drive_pA}.
    """
    # Simple encoding: drive ALL cortex pools uniformly with a position-
    # dependent total amplitude. The cortex→striatum learning has to
    # discover which action is right for each position.
    return {a: rate_peak for a in ACTION_NAMES}


def run_moving_goal_episode(
    out_path: str,
    seed: int = 42,
    n_steps: int = 1800,
    grid_size: int = 8,
    start_pos=(1, 1),
    goal_pos=(6, 6),
    goal_schedule=None,
    learning_rate: float = 0.01,
    reward_eligibility_tau_ms: float = 500.0,
    reward_hold_steps: int = 10,
    verbose: bool = True,
    enable_motor_lateral_inhibition: bool = False,
    enable_per_action_da_targeting: bool = False,
    enable_adaptive_per_action_da: bool = False,
    adaptive_da_ema_decay: float = 0.9,  # ~tau=10 trials (used for positive reward)
    adaptive_da_ema_decay_negative: float = None,  # if set, separate decay for negative reward (faster = quicker exploration trigger)
    enable_learned_perception: bool = False,
    sensory_drive_max_pA: float = 600.0,
    sensory_drive_sigma: float = 1.5,
    enable_hippocampus: bool = False,
    hippocampus_drive_max_pA: float = 600.0,
    hippocampus_drive_sigma: float = 0.5,  # narrower → sparser firing → 1-3 cells per position
    # Informed init for learned perception: bias initial sensory->cortex_X weights
    # by alignment between sensor's preferred (dx,dy) and action X's direction
    # vector. Solves cold-start failure (random init produces no asymmetry, no
    # learning signal). Plasticity then refines the prior rather than discovers.
    enable_learned_perception_informed_init: bool = False,
    informed_init_alpha: float = 8.0,  # sharper positive-only prior; aligned ~ 24.5 weight (heuristic-equivalent)
    # DA-gated WTA: when both --motor-lateral-inhibition and adaptive DA are on,
    # scale FS→motor inhibition weight per-trial by gating_strength (reward EMA).
    # Implements the user's "DA gate" concept: when winning, WTA strong (commit);
    # when losing, WTA relaxes (explore via reduced inhibition).
    enable_da_gated_wta: bool = False,
    # RPE-scaled reward (NE-like surprise amplification):
    # delivered_reward = reward + alpha * (reward - reward_ema)
    # When reward is unexpectedly negative (after positive EMA), the prediction
    # error is large and amplified — fast adaptation. Expected outcomes get
    # muted. Real biology: DA encodes RPE not raw reward (Schultz 1997).
    enable_rpe_scaled_reward: bool = False,
    rpe_scale_alpha: float = 1.0,  # 1.0 means: delivered = 2*reward - ema
    # Surprise-boosted learning rate: when |RPE| is high (unexpected outcome),
    # temporarily boost reward_learning_rate. Models NE-like fast meta-modulation.
    enable_surprise_lr_boost: bool = False,
    surprise_lr_alpha: float = 2.0,  # max boost factor: 1 + alpha * |RPE|
):
    """Phase B acid test: run BG circuit on G9-style moving-goal scenario.

    If the BG architecture dissolves the silent-motor trap (which V1-V7
    runner-side interventions all failed to do), phase 1 finalQ should
    drop substantially below the G9 baseline of 6.74.
    """
    import cupy as cp
    from sim import (
        SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig,
    )
    from sim.enums import NeuronModel

    if goal_schedule is None:
        goal_schedule = [(0, tuple(goal_pos))]
    goal_schedule_sorted = sorted(
        [(int(s), tuple(g)) for s, g in goal_schedule], key=lambda t: t[0]
    )

    regions, pathways = build_bg_brain_regions(
        n_cortex=100,  # 25 per action — keeps D1 firing in physiological range (~75 Hz)
        enable_motor_lateral_inhibition=enable_motor_lateral_inhibition,
        enable_learned_perception=enable_learned_perception,
        enable_hippocampus=enable_hippocampus,
    )

    # Pre-compute sensory neuron preferred (dx, dy) — 7x7 grid covering [-3, 3]²
    if enable_learned_perception:
        sensory_pref = []
        for iy in range(7):
            for ix in range(7):
                sensory_pref.append((ix - 3, iy - 3))  # dx, dy ∈ [-3, 3]
        sensory_pref_dx = np.array([p[0] for p in sensory_pref], dtype=np.float32)
        sensory_pref_dy = np.array([p[1] for p in sensory_pref], dtype=np.float32)
    else:
        sensory_pref_dx = None
        sensory_pref_dy = None

    # Pre-compute hippocampal cell preferred (x, y) — 8x8 grid covering [0, 7]²
    if enable_hippocampus:
        hippo_pref_x = np.array([i % 8 for i in range(64)], dtype=np.float32)
        hippo_pref_y = np.array([i // 8 for i in range(64)], dtype=np.float32)
    else:
        hippo_pref_x = None
        hippo_pref_y = None
    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.dt_ms = 1.0
    cfg.seed = int(seed)
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.enable_stdp = True
    cfg.enable_reward_modulation = True
    cfg.reward_learning_rate = float(learning_rate)
    cfg.reward_eligibility_tau_ms = float(reward_eligibility_tau_ms)
    # cortex->D1 weight_mean=25 needs w_max above that or soft-bound STDP collapses it
    cfg.stdp_w_max = 30.0
    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_ou_process = False
    cfg.enable_conductance_noise = False
    cfg.enable_parameter_heterogeneity = False
    cfg.enable_structural_plasticity = False  # keep synapse count fixed (per-action DA mask depends on it)

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    # Pre-cache region indices (cupy arrays for fast per-step indexing)
    region_indices_cp = {}
    for r in regions:
        idx = list(bridge.region_manager.indices(r.name))
        if idx:
            region_indices_cp[r.name] = cp.asarray(idx, dtype=cp.int64)
    motor_idx_per_action = {
        a: region_indices_cp[f"motor_{a}"] for a in ACTION_NAMES
    }

    # Per-action DA targeting: pre-compute synapse-post-action mask.
    # For each plastic cortex→str_D1_X synapse, mark which action X it serves.
    # Per-trial: scale eligibility on synapses where post is in str_D1_Y (Y != selected)
    # by (1 - gating_strength), where gating_strength is either fixed at 1.0 (hard
    # mode) or adapted from recent reward stability (adaptive mode).
    # Adaptive: tracks reward EMA; high positive EMA → strength=1 (commit, exploit);
    # low/negative EMA → strength=0 (explore, broadcast credit).
    # We restrict to D1 (direct path); D2 (indirect) keeps broadcast learning.
    use_da_targeting = enable_per_action_da_targeting or enable_adaptive_per_action_da
    if use_da_targeting:
        coo = bridge.cp_connections.tocoo()
        post_neurons_cp = coo.col  # cupy int64
        n_synapses = int(post_neurons_cp.size)
        synapse_post_action = cp.full(n_synapses, -1, dtype=cp.int8)
        for action_idx_setup, action_name in enumerate(ACTION_NAMES):
            d1_indices = region_indices_cp[f"str_D1_{action_name}"]
            mask_d1 = cp.isin(post_neurons_cp, d1_indices)
            synapse_post_action[mask_d1] = action_idx_setup
        # Cache: per-action mask of "synapses NOT going to action X's D1 pool"
        # (used to zero eligibility before reward hold).
        d1_synapse_other_action_masks = {}
        for action_idx_setup, action_name in enumerate(ACTION_NAMES):
            # Mask = is a D1 synapse AND post-action != this action
            other_d1 = (synapse_post_action >= 0) & (synapse_post_action != action_idx_setup)
            d1_synapse_other_action_masks[action_idx_setup] = other_d1
        if verbose:
            n_d1_synapses = int((synapse_post_action >= 0).sum().get())
            mode = "adaptive" if enable_adaptive_per_action_da else "hard"
            print(f"[g11 seed={seed}] per-action DA ({mode}): "
                  f"{n_d1_synapses} synapses are cortex->D1 (will be selectively gated)")
    else:
        d1_synapse_other_action_masks = None

    # Adaptive DA state — reward EMA in [-1, +1]
    reward_ema = 0.0
    da_strength_log = []  # log per-trial gating strength for analysis

    # DA-gated WTA: pre-compute FS->motor synapse indices and save baseline weights.
    # Per-trial we'll scale these weights by gating_strength to make WTA adaptive.
    fs_to_motor_indices = None
    fs_to_motor_baseline_weights = None
    if enable_da_gated_wta and enable_motor_lateral_inhibition:
        # All FS pre-neurons (across 4 actions); all motor post-neurons (across 4 actions)
        fs_indices_all = []
        motor_indices_all = []
        for action in ACTION_NAMES:
            fs_indices_all.extend(region_indices_cp[f"motor_FS_{action}"].get().tolist())
            motor_indices_all.extend(region_indices_cp[f"motor_{action}"].get().tolist())
        fs_set = set(fs_indices_all)
        motor_set = set(motor_indices_all)
        # Find synapse indices where pre in fs_set AND post in motor_set
        coo = bridge.cp_connections.tocoo()
        rows = coo.row.get(); cols = coo.col.get()
        # CSR convention: assume cp_connections[i, j] means i->j (pre->post)
        # We pick the orientation that gives non-zero count.
        mask_a = np.array([r in fs_set and c in motor_set for r, c in zip(rows, cols)])
        mask_b = np.array([c in fs_set and r in motor_set for r, c in zip(rows, cols)])
        if mask_a.sum() > mask_b.sum():
            chosen_mask = mask_a
            convention = "row=pre, col=post"
        else:
            chosen_mask = mask_b
            convention = "row=post, col=pre"
        fs_to_motor_indices = cp.asarray(np.where(chosen_mask)[0], dtype=cp.int64)
        # Snapshot baseline weights (constant since FS->motor is plastic=False)
        fs_to_motor_baseline_weights = bridge.cp_connections.data[fs_to_motor_indices].copy()
        if verbose:
            print(f"[g11 seed={seed}] DA-gated WTA: {int(chosen_mask.sum())} FS->motor synapses "
                  f"({convention}), will scale by gating_strength per trial")

    # Informed initialization for learned perception: bias initial sensory->cortex_X
    # weights by alignment between sensor's preferred (dx, dy) and action X's
    # direction vector. Solves the cold-start problem identified in
    # research/findings/2026-04-26-learned-perception-cold-start-fail.md.
    if (enable_learned_perception
            and enable_learned_perception_informed_init
            and sensory_pref_dx is not None):
        # Action direction vectors (N, E, S, W) — must match ACTION_DELTAS
        action_dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        sensory_indices_list = list(bridge.region_manager.indices("sensory"))
        sensory_set = set(sensory_indices_list)
        sensory_idx_to_pos = {n: i for i, n in enumerate(sensory_indices_list)}
        coo = bridge.cp_connections.tocoo()
        rows_np = coo.row.get(); cols_np = coo.col.get()
        n_modified = 0
        # CSR convention here: rows are pre, cols are post (verified by FS->motor logic above)
        for action_idx, action_name in enumerate(ACTION_NAMES):
            cortex_X_set = set(bridge.region_manager.indices(f"cortex_{action_name}"))
            ax, ay = action_dirs[action_idx]
            # Find synapse indices where pre is in sensory and post is in cortex_X
            new_weights = []
            target_indices = []
            for syn_idx in range(rows_np.size):
                pre = int(rows_np[syn_idx])
                post = int(cols_np[syn_idx])
                if pre in sensory_set and post in cortex_X_set:
                    sensor_layer_idx = sensory_idx_to_pos[pre]
                    dx_pref = float(sensory_pref_dx[sensor_layer_idx])
                    dy_pref = float(sensory_pref_dy[sensor_layer_idx])
                    # Alignment: dot product of sensor's preferred direction with action's direction
                    alignment = dx_pref * ax + dy_pref * ay  # ranges roughly [-3, +3]
                    # SHARP prior: only positive alignment contributes meaningfully.
                    # Orthogonal/anti-aligned sensors get near-zero weight so they don't
                    # drive cortex_X (avoiding cascade saturation across all 4 pools).
                    # Aligned sensors get strong weight (up to ~25 = matches heuristic 800pA equivalent).
                    positive_alignment = max(0.0, alignment)
                    new_w = max(0.5, 0.5 + informed_init_alpha * positive_alignment)
                    new_weights.append(new_w)
                    target_indices.append(syn_idx)
            if target_indices:
                idx_cp = cp.asarray(target_indices, dtype=cp.int64)
                w_cp = cp.asarray(new_weights, dtype=cp.float32)
                bridge.cp_connections.data[idx_cp] = w_cp
                n_modified += len(target_indices)
        if verbose:
            print(f"[g11 seed={seed}] learned perception (informed init): "
                  f"rewrote {n_modified} sensory->cortex weights with directional prior "
                  f"(alpha={informed_init_alpha})")

    # Setup baseline tonic drives that don't change between steps
    bridge.cp_external_input_current[:] = 0.0
    for region_name in [f"gpe_{a}" for a in ACTION_NAMES]:
        bridge.cp_external_input_current[region_indices_cp[region_name]] = cp.float32(150.0)
    for region_name in [f"gpi_{a}" for a in ACTION_NAMES]:
        bridge.cp_external_input_current[region_indices_cp[region_name]] = cp.float32(110.0)
    for region_name in ["stn", "dopamine"]:
        bridge.cp_external_input_current[region_indices_cp[region_name]] = cp.float32(150.0)
    for region_name in [f"thal_{a}" for a in ACTION_NAMES]:
        bridge.cp_external_input_current[region_indices_cp[region_name]] = cp.float32(300.0)

    # Action deltas
    ACTION_DELTAS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # N, E, S, W
    n_motor_per_action = sum(1 for r in regions if r.name.startswith("motor_")) * 0  # placeholder
    # Number of neurons in each motor pool (all same)
    n_motor_pop = next(r.n_neurons for r in regions if r.name.startswith("motor_"))

    x, y = start_pos
    current_schedule_idx = 0
    gx, gy = goal_schedule_sorted[0][1]

    def manhattan(px, py):
        return abs(px - gx) + abs(py - gy)

    trajectory = [(x, y)]
    goal_log = [(gx, gy)]
    motor_counts_log = []
    action_log = []
    reward_log = []
    distance_log = [manhattan(x, y)]
    goal_change_steps = []

    STIMULUS_MS = 100.0
    READOUT_START_MS = 30.0
    READOUT_END_MS = 100.0
    n_stim_steps = int(STIMULUS_MS / cfg.dt_ms)
    readout_start = int(READOUT_START_MS / cfg.dt_ms)
    readout_end = int(READOUT_END_MS / cfg.dt_ms)

    cortex_idx_per_action = {
        a: region_indices_cp[f"cortex_{a}"] for a in ACTION_NAMES
    }

    if verbose:
        print(f"[g11 seed={seed}] BG circuit: {len(regions)} regions, "
              f"{cfg.num_neurons} neurons, {bridge.cp_connections.nnz} synapses",
              flush=True)

    t0 = time.time()
    # Track current gating_strength (used for DA-gated WTA across the whole trial,
    # not just the reward-hold sub-step). Initialized to 1.0 (full WTA on first trial
    # before any reward feedback exists).
    current_gating_strength = 1.0
    for step in range(n_steps):
        # Goal change
        while (current_schedule_idx + 1 < len(goal_schedule_sorted)
               and step >= goal_schedule_sorted[current_schedule_idx + 1][0]):
            current_schedule_idx += 1
            gx, gy = goal_schedule_sorted[current_schedule_idx][1]
            goal_change_steps.append(step)
            if verbose:
                print(f"[g11 seed={seed}] step {step}: GOAL CHANGED to ({gx}, {gy})",
                      flush=True)

        # DA-gated WTA: scale FS->motor synapse weights by current gating_strength.
        # When gating=1 (winning, exploit), full WTA. When gating=0 (losing,
        # explore), WTA disabled (no inhibition). Updated AFTER each trial's
        # reward feedback below.
        if fs_to_motor_indices is not None:
            bridge.cp_connections.data[fs_to_motor_indices] = (
                fs_to_motor_baseline_weights * cp.float32(current_gating_strength)
            )

        dist_before = manhattan(x, y)

        # Sensory input encoding: drive cortex pools based on position.
        # SIMPLE HEURISTIC: drive each cortex_X pool with strength inversely
        # proportional to current direction's distance to goal. This is a
        # phenomenological "goal-direction signal" — what the agent's
        # higher cortex would compute given knowledge of the goal.
        # The BG circuit then has to produce a clean motor output.
        # NOTE: this DOESN'T let the BG demonstrate "discovery" — but it
        # does test whether the BG's per-action structure dissolves the
        # silent-motor trap on phase change.
        # RE-SET ALL baseline drives every trial (defensive against any drift).
        bridge.cp_external_input_current[:] = 0.0
        for rn in [f"gpe_{a}" for a in ACTION_NAMES]:
            bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(150.0)
        for rn in [f"gpi_{a}" for a in ACTION_NAMES]:
            bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(110.0)
        for rn in ["stn", "dopamine"]:
            bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(150.0)
        for rn in [f"thal_{a}" for a in ACTION_NAMES]:
            bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(300.0)
        # Cortex drives — main source: heuristic OR learned perception (mutually exclusive)
        if enable_learned_perception:
            # Drive sensory layer based on relative goal position (dx, dy).
            # Each sensory neuron i has preferred (dx_i, dy_i); rate = max * exp(-d²/2σ²)
            # No direct cortex drive — agent must learn sensory→cortex mapping.
            dx = float(gx - x)
            dy = float(gy - y)
            # Clip to sensor range to handle positions far from goal
            dx_clip = max(-3.0, min(3.0, dx))
            dy_clip = max(-3.0, min(3.0, dy))
            d_sq = (sensory_pref_dx - dx_clip) ** 2 + (sensory_pref_dy - dy_clip) ** 2
            sensory_drive = sensory_drive_max_pA * np.exp(-d_sq / (2.0 * sensory_drive_sigma ** 2))
            bridge.cp_external_input_current[region_indices_cp["sensory"]] = cp.asarray(sensory_drive, dtype=cp.float32)
        else:
            # Heuristic cortex drive: directly drive cortex_X for each goal-relative direction
            if gy > y:
                bridge.cp_external_input_current[region_indices_cp["cortex_N"]] = cp.float32(800.0)
            if gx > x:
                bridge.cp_external_input_current[region_indices_cp["cortex_E"]] = cp.float32(800.0)
            if gy < y:
                bridge.cp_external_input_current[region_indices_cp["cortex_S"]] = cp.float32(800.0)
            if gx < x:
                bridge.cp_external_input_current[region_indices_cp["cortex_W"]] = cp.float32(800.0)

        # Hippocampus drive (ADDITIVE on top of heuristic — provides plastic memory).
        # Real biology: hippocampus augments cortex, doesn't replace it. Place + goal
        # cells learn (place, goal) → action associations via STDP+reward, providing
        # additional cortex drive that should reinforce the correct action over training.
        if enable_hippocampus:
            place_dsq = (hippo_pref_x - float(x)) ** 2 + (hippo_pref_y - float(y)) ** 2
            place_drive = hippocampus_drive_max_pA * np.exp(-place_dsq / (2.0 * hippocampus_drive_sigma ** 2))
            bridge.cp_external_input_current[region_indices_cp["place_cells"]] = cp.asarray(place_drive, dtype=cp.float32)
            goal_dsq = (hippo_pref_x - float(gx)) ** 2 + (hippo_pref_y - float(gy)) ** 2
            goal_drive = hippocampus_drive_max_pA * np.exp(-goal_dsq / (2.0 * hippocampus_drive_sigma ** 2))
            bridge.cp_external_input_current[region_indices_cp["goal_cells"]] = cp.asarray(goal_drive, dtype=cp.float32)

        # Run stimulus window and tally motor spikes
        motor_counts = {a: 0 for a in ACTION_NAMES}
        bridge.core_config.current_reward_signal = 0.0
        for s in range(n_stim_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = (
                bridge.runtime_state.current_time_step * cfg.dt_ms
            )
            if readout_start <= s < readout_end:
                firing = bridge.cp_firing_states.get().astype(bool)
                for a in ACTION_NAMES:
                    motor_counts[a] += int(firing[motor_idx_per_action[a].get()].sum())

        motor_counts_log.append([motor_counts[a] for a in ACTION_NAMES])

        # Argmax action selection (random if all silent)
        if max(motor_counts.values()) > 0:
            action_idx = max(range(N_ACTIONS), key=lambda i: motor_counts[ACTION_NAMES[i]])
        else:
            action_idx = int(np.random.default_rng(seed * 10000 + step).integers(0, N_ACTIONS))
        action_log.append(action_idx)

        dx, dy = ACTION_DELTAS[action_idx]
        new_x = int(np.clip(x + dx, 0, grid_size - 1))
        new_y = int(np.clip(y + dy, 0, grid_size - 1))
        dist_after = manhattan(new_x, new_y)
        x, y = new_x, new_y
        trajectory.append((x, y))
        goal_log.append((gx, gy))
        distance_log.append(dist_after)

        if dist_after < dist_before:
            reward = 1.0
        elif dist_after > dist_before:
            reward = -1.0
        else:
            reward = 0.0
        reward_log.append(float(reward))

        if abs(reward) > 0:
            # Capture EMA BEFORE update (= the agent's prediction at this step)
            reward_ema_pre = reward_ema
            # Update reward EMA (used by adaptive DA mode).
            # If asymmetric decay is configured, use faster decay for negative
            # reward (quicker exploration trigger on goal change / policy break).
            # Models phasic DA biology: dips on negative RPE faster than ramps
            # up on positive (Schultz 1998).
            if reward < 0 and adaptive_da_ema_decay_negative is not None:
                _decay = adaptive_da_ema_decay_negative
            else:
                _decay = adaptive_da_ema_decay
            reward_ema = _decay * reward_ema + (1 - _decay) * float(reward)

            # Compute gating strength for per-action DA targeting:
            #   hard:     always 1.0 (full gating)
            #   adaptive: scales linearly from reward_ema in [-1, +1] to strength in [0, 1]
            #             reward_ema=+1 (consistently winning) → strength=1.0 (full gating, exploit)
            #             reward_ema=-1 (consistently losing)  → strength=0.0 (no gating, explore)
            if enable_adaptive_per_action_da:
                gating_strength = max(0.0, min(1.0, (reward_ema + 1.0) / 2.0))
            elif enable_per_action_da_targeting:
                gating_strength = 1.0
            else:
                gating_strength = 0.0
            da_strength_log.append(float(gating_strength))
            # Cache for next trial's WTA scaling
            current_gating_strength = float(gating_strength)

            # Apply per-action DA: scale eligibility on non-selected pathways by (1 - strength)
            if (gating_strength > 0
                    and d1_synapse_other_action_masks is not None
                    and bridge.cp_eligibility_trace is not None):
                actual_nnz = bridge.cp_connections.nnz
                other_mask = d1_synapse_other_action_masks[action_idx][:actual_nnz]
                scale = float(1.0 - gating_strength)
                # Scale eligibility on non-selected pathways
                trace = bridge.cp_eligibility_trace[:actual_nnz]
                trace[other_mask] = trace[other_mask] * scale

            # RPE-scaled reward (opt-in): amplify surprise (= deviation from expectation)
            # Uses reward_ema_pre (the agent's prediction BEFORE this trial's reward).
            rpe = float(reward) - reward_ema_pre
            if enable_rpe_scaled_reward:
                delivered_reward = float(reward) + rpe_scale_alpha * rpe
            else:
                delivered_reward = float(reward)
            bridge.core_config.current_reward_signal = delivered_reward

            # Surprise-boosted learning rate (opt-in): NE-like fast meta-modulation.
            # When |RPE| is high, temporarily boost reward_learning_rate. Restored
            # after reward hold. Decoupled from per-action DA gating mechanism.
            base_lr = float(learning_rate)
            if enable_surprise_lr_boost:
                surprise = abs(rpe)
                bridge.core_config.reward_learning_rate = base_lr * (1.0 + surprise_lr_alpha * surprise)

            for _ in range(reward_hold_steps):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
                bridge.runtime_state.current_time_ms = (
                    bridge.runtime_state.current_time_step * cfg.dt_ms
                )
            bridge.core_config.current_reward_signal = 0.0
            # Restore base reward_learning_rate (in case surprise-boosted)
            if enable_surprise_lr_boost:
                bridge.core_config.reward_learning_rate = base_lr

        if verbose and (step + 1) % 100 == 0:
            recent_dist = float(np.mean(distance_log[-100:]))
            print(f"[g11 seed={seed}] step {step+1}/{n_steps}  pos=({x},{y})  "
                  f"goal=({gx},{gy})  recent_dist={recent_dist:.2f}  "
                  f"actions={action_log[-100:].count(0):>3d}N/{action_log[-100:].count(1):>3d}E/"
                  f"{action_log[-100:].count(2):>3d}S/{action_log[-100:].count(3):>3d}W",
                  flush=True)

    elapsed = time.time() - t0
    dist_arr = np.asarray(distance_log[1:])
    quarters = [float(dist_arr[i*len(dist_arr)//4:(i+1)*len(dist_arr)//4].mean())
                for i in range(4)]

    # Per-phase stats
    phase_stats = []
    phase_boundaries = [0] + goal_change_steps + [n_steps]
    for phase_idx in range(len(phase_boundaries) - 1):
        p_start = phase_boundaries[phase_idx]
        p_end = phase_boundaries[phase_idx + 1]
        p_dist = dist_arr[p_start:p_end]
        p_actions = action_log[p_start:p_end]
        if len(p_dist) == 0:
            continue
        p_goal = goal_log[p_start + 1] if p_start + 1 < len(goal_log) else goal_log[-1]
        phase_stats.append({
            "phase": phase_idx,
            "step_start": p_start, "step_end": p_end,
            "goal": list(p_goal),
            "mean_distance": float(p_dist.mean()),
            "final_quarter_mean_distance": float(p_dist[len(p_dist)*3//4:].mean())
                if len(p_dist) >= 4 else float(p_dist.mean()),
            "n_steps_at_goal": int((p_dist == 0).sum()),
            "n_steps": len(p_dist),
            "action_counts": [int((np.asarray(p_actions) == a).sum())
                              for a in range(N_ACTIONS)],
        })

    results = {
        "seed": seed, "n_steps": n_steps, "grid_size": grid_size,
        "start_pos": list(start_pos), "goal_pos": list(goal_pos),
        "goal_schedule": [[s, list(g)] for s, g in goal_schedule_sorted],
        "goal_change_steps": goal_change_steps,
        "phase_stats": phase_stats,
        "reward_learning_rate": learning_rate,
        "trajectory": trajectory, "goal_log": goal_log,
        "motor_counts": motor_counts_log,
        "action_log": action_log, "reward_log": reward_log,
        "distance_log": distance_log,
        "mean_distance_overall": float(dist_arr.mean()),
        "mean_distance_quarters": quarters,
        "n_steps_at_goal": int((dist_arr == 0).sum()),
        "elapsed_seconds": elapsed,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(f"\n[g11 seed={seed}] DONE in {elapsed:.0f}s. "
              f"Phase stats:")
        for p in phase_stats:
            print(f"  phase {p['phase']} goal={p['goal']} "
                  f"meanD={p['mean_distance']:.2f} "
                  f"finalQ={p['final_quarter_mean_distance']:.2f} "
                  f"actions={p['action_counts']}")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke test: build + 50 steps at rest")
    ap.add_argument("--probe-action", type=str, default=None,
                    choices=ACTION_NAMES,
                    help="Drive cortex toward this action and measure motor output")
    ap.add_argument("--moving-goal", action="store_true",
                    help="Run G9-style moving-goal scenario (Phase B.T6 acid test)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-steps", type=int, default=1800)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--motor-lateral-inhibition", action="store_true",
                    help="Enable FS-mediated motor pool lateral inhibition (WTA microcircuit)")
    ap.add_argument("--per-action-da", action="store_true",
                    help="Enable per-action dopamine targeting (hard): reward only credits chosen action's cortex->D1 synapses")
    ap.add_argument("--adaptive-da", action="store_true",
                    help="Enable ADAPTIVE per-action DA: gating strength scales with recent reward EMA (low reward -> broadcast)")
    ap.add_argument("--adaptive-da-ema-decay", type=float, default=0.9,
                    help="EMA decay for adaptive DA (default 0.9, tau~10 trials; lower = faster reaction)")
    ap.add_argument("--adaptive-da-ema-decay-negative", type=float, default=None,
                    help="Separate (faster) EMA decay for negative reward (asymmetric ramp; biologically: phasic DA dip)")
    ap.add_argument("--learned-perception", action="store_true",
                    help="Enable learned sensory->cortex mapping (49-neuron sensory layer, plastic to cortex)")
    ap.add_argument("--informed-init", action="store_true",
                    help="Bias initial sensory->cortex weights by directional alignment (requires --learned-perception)")
    ap.add_argument("--informed-init-alpha", type=float, default=8.0,
                    help="Strength of positive-only directional prior (default 8.0; aligned weight ~24.5, orthogonal ~0.5)")
    ap.add_argument("--hippocampus", action="store_true",
                    help="Enable hippocampal module: 64 place cells + 64 goal cells with sparse Gaussian tuning, plastic to cortex")
    ap.add_argument("--da-gated-wta", action="store_true",
                    help="Scale motor FS->motor inhibition by reward-EMA gating_strength (the 'DA gate'). Requires --motor-lateral-inhibition + --adaptive-da")
    ap.add_argument("--goal-schedule", type=str, default="default",
                    help="'default' = (6,6) -> (1,6) at step 300. 'multi' = 4 goal changes across the corners.")
    ap.add_argument("--rpe-scaled-reward", action="store_true",
                    help="Scale reward by prediction error: delivered = reward + alpha * (reward - reward_ema). Surprise gets amplified.")
    ap.add_argument("--rpe-alpha", type=float, default=1.0)
    ap.add_argument("--surprise-lr-boost", action="store_true",
                    help="Boost reward_learning_rate when |RPE| is high (NE-like fast meta-modulation)")
    ap.add_argument("--surprise-lr-alpha", type=float, default=2.0)
    args = ap.parse_args()

    if args.moving_goal:
        out_path = args.out or f"research/findings/raw/g11_bg/g11_seed{args.seed}.json"
        if args.goal_schedule == "multi":
            # 4 corners cycle, goal changes every 450 steps
            goal_schedule = [(0, (6, 6)), (450, (1, 6)), (900, (1, 1)), (1350, (6, 1))]
        else:
            goal_schedule = [(0, (6, 6)), (300, (1, 6))]
        run_moving_goal_episode(
            out_path=out_path,
            seed=args.seed,
            n_steps=args.n_steps,
            goal_schedule=goal_schedule,
            enable_motor_lateral_inhibition=args.motor_lateral_inhibition,
            enable_per_action_da_targeting=args.per_action_da,
            enable_adaptive_per_action_da=args.adaptive_da,
            adaptive_da_ema_decay=args.adaptive_da_ema_decay,
            adaptive_da_ema_decay_negative=args.adaptive_da_ema_decay_negative,
            enable_learned_perception=args.learned_perception,
            enable_learned_perception_informed_init=args.informed_init,
            informed_init_alpha=args.informed_init_alpha,
            enable_hippocampus=args.hippocampus,
            enable_da_gated_wta=args.da_gated_wta,
            enable_rpe_scaled_reward=args.rpe_scaled_reward,
            rpe_scale_alpha=args.rpe_alpha,
            enable_surprise_lr_boost=args.surprise_lr_boost,
            surprise_lr_alpha=args.surprise_lr_alpha,
        )
        return 0

    from sim import (
        SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig,
    )
    from sim.enums import NeuronModel
    import cupy as cp

    print(f"\n{'='*72}")
    print(f"  G11 BG Action Selection Module — Smoke Test")
    print(f"{'='*72}\n", flush=True)

    regions, pathways = build_bg_brain_regions()
    n_total = sum(r.n_neurons for r in regions)
    print(f"  Built {len(regions)} regions with {n_total} total neurons")
    print(f"  Built {len(pathways)} pathways")
    print()

    # Verify no name collisions
    names = [r.name for r in regions]
    assert len(set(names)) == len(names), "Region name collision!"

    cfg = CoreSimConfig()
    cfg.num_neurons = 0  # Set by region framework
    cfg.dt_ms = 1.0
    cfg.seed = int(args.seed)
    cfg.num_traits = 1  # Force single neuron type per region
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.enable_stdp = False  # Smoke test: no plasticity
    cfg.enable_reward_modulation = False
    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_ou_process = False
    cfg.enable_conductance_noise = False
    cfg.enable_parameter_heterogeneity = False

    print(f"  Initializing bridge...", flush=True)
    t0 = time.time()
    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    elapsed = time.time() - t0
    print(f"  Bridge initialized in {elapsed:.1f}s", flush=True)
    print(f"  Total neurons: {cfg.num_neurons}")
    print(f"  Total synapses: {bridge.cp_connections.nnz}")

    if not args.smoke and not args.probe_action:
        return 0

    # Quick 30-step smoke run with no input — should show GPe/GPi tonic firing
    if bridge.cp_external_input_current is not None:
        bridge.cp_external_input_current[:] = 0.0
    n_steps = 50
    n_motor_total = sum(r.n_neurons for r in regions if r.name.startswith("motor_"))

    spike_counts = np.zeros(cfg.num_neurons, dtype=np.int32)
    print(f"\n  Running {n_steps} steps with no input (rest dynamics)...", flush=True)
    for s in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * cfg.dt_ms
        firing = bridge.cp_firing_states.get().astype(np.int32)
        spike_counts += firing

    # Per-region firing rate
    print(f"\n  Per-region firing rates (Hz over {n_steps}ms with no input):")
    for r in regions:
        idx = bridge.region_manager.indices(r.name)
        rate_hz = spike_counts[list(idx)].sum() / r.n_neurons / (n_steps * cfg.dt_ms / 1000.0)
        print(f"    {r.name:<24s} ({r.izh_neuron_type or 'default':<32s}): {rate_hz:.1f} Hz")

    print(f"\n  Smoke test PASSED — {len(regions)} regions, "
          f"{bridge.cp_connections.nnz} synapses initialized cleanly.")

    # ---- Phase B.T4 / T5: action selection probe ----
    if args.probe_action:
        print(f"\n{'='*72}")
        print(f"  Action selection probe: drive cortex -> {args.probe_action} pathway")
        print(f"{'='*72}\n", flush=True)

        # Inject strong current into a SUBSET of cortex neurons. The cortex->D1/D2
        # weights are random — so the input pattern preferentially activates
        # whichever D1/D2 happens to have stronger weights from these inputs.
        # For a clean probe, manually override: inject ONLY into cortex neurons
        # whose hash maps to the target action.
        # Apply tonic baseline drive to BG output nuclei (mimics intrinsic
        # depolarizing conductance that makes real GPe/GPi/STN autonomously
        # fire 30-80 Hz). Without this, our Izh presets sit at rest because
        # Izh doesn't model intrinsic Ca pacemaker currents.
        bridge.cp_external_input_current[:] = 0.0
        # Per-region tonic drive levels:
        for region_name in [f"gpe_{a}" for a in ACTION_NAMES]:
            idx = list(bridge.region_manager.indices(region_name))
            if idx:
                bridge.cp_external_input_current[cp.asarray(idx, dtype=cp.int64)] = cp.float32(150.0)
        for region_name in [f"gpi_{a}" for a in ACTION_NAMES]:
            idx = list(bridge.region_manager.indices(region_name))
            if idx:
                # Lower baseline for GPi → easier to silence by D1 inhibition
                bridge.cp_external_input_current[cp.asarray(idx, dtype=cp.int64)] = cp.float32(110.0)
        for region_name in ["stn", "dopamine"]:
            idx = list(bridge.region_manager.indices(region_name))
            if idx:
                bridge.cp_external_input_current[cp.asarray(idx, dtype=cp.int64)] = cp.float32(150.0)
        # Thalamus baseline drive — set such that GPi inhibition (when active)
        # keeps thal silent, AND when GPi drops to 0 (D1 suppression),
        # thal fires actively.
        for region_name in [f"thal_{a}" for a in ACTION_NAMES]:
            idx = list(bridge.region_manager.indices(region_name))
            if idx:
                bridge.cp_external_input_current[cp.asarray(idx, dtype=cp.int64)] = cp.float32(300.0)

        # Drive ONLY the target action's cortex pool
        cortex_idx = list(bridge.region_manager.indices(f"cortex_{args.probe_action}"))
        cortex_cp = cp.asarray(cortex_idx, dtype=cp.int64)

        bridge.runtime_state.current_time_step = 0
        bridge.runtime_state.current_time_ms = 0.0

        drive_pA = 800.0
        n_probe_steps = 500
        target_cortex = cortex_idx
        spike_counts = np.zeros(cfg.num_neurons, dtype=np.int32)
        for s in range(n_probe_steps):
            bridge.cp_external_input_current[cortex_cp] = cp.float32(drive_pA)
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * cfg.dt_ms
            firing = bridge.cp_firing_states.get().astype(np.int32)
            spike_counts += firing

        # Per-region firing rate
        print(f"  Driving {len(target_cortex)}/{len(cortex_idx)} cortex neurons "
              f"with {drive_pA} pA for {n_probe_steps}ms")
        print(f"\n  Per-region firing rates over {n_probe_steps}ms:")
        ordered_groups = [f"cortex_{a}" for a in ACTION_NAMES]
        for a in ACTION_NAMES:
            ordered_groups += [f"str_D1_{a}", f"str_D2_{a}", f"gpe_{a}",
                                f"gpi_{a}", f"thal_{a}", f"motor_{a}"]
        ordered_groups += ["stn", "dopamine"]
        for region_name in ordered_groups:
            r = next((reg for reg in regions if reg.name == region_name), None)
            if r is None:
                continue
            idx = bridge.region_manager.indices(r.name)
            if not idx:
                continue
            rate_hz = spike_counts[list(idx)].sum() / r.n_neurons / (n_probe_steps / 1000.0)
            marker = " <-" if (region_name.endswith(f"_{args.probe_action}") and
                              region_name.startswith(("str_D1_", "thal_", "motor_"))) else ""
            print(f"    {r.name:<15s} {rate_hz:>6.1f} Hz{marker}")

        # Quick check: did the right motor pop fire most?
        motor_rates = {}
        for a in ACTION_NAMES:
            idx = bridge.region_manager.indices(f"motor_{a}")
            n = len(idx)
            r = spike_counts[list(idx)].sum() / max(n, 1) / (n_probe_steps / 1000.0)
            motor_rates[a] = r
        winner = max(motor_rates, key=motor_rates.get)
        print(f"\n  Motor rates: {motor_rates}")
        print(f"  Winner: {winner}  (target: {args.probe_action})")
        if winner == args.probe_action and motor_rates[winner] > 5:
            print(f"  [OK] BG circuit selected the correct motor")
        else:
            print(f"  -> BG circuit did not produce a clean winner (rates may be too low/noisy)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
