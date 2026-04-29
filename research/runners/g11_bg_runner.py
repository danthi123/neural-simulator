"""G11: Basal ganglia action selection module.

Phase B follow-up to the silent-motor trap arc (Sessions G/H/I, all NEGATIVE).
The trap was diagnosed (V6) as a *reservoir-state bias problem* — random
hidden->motor weights on a shared reservoir naturally favor whichever motor
the input pattern happens to align with. Argmax + reservoir bias = lock-in.

Phase B fix (architectural): replace the shared-reservoir + argmax-readout
with a per-action basal-ganglia cascade. Each motor has its own dedicated
D1 MSN pool, D2 MSN pool, GPi, thalamus, and motor populations. Lateral
inhibition between motor populations provides structural winner-take-all
(no shared spike count to bias).

Architecture:
    cortex ─-> str_D1[N,E,S,W]    str_D2[N,E,S,W]
                  │                     │
            direct pathway       indirect pathway
                  v                     v
              GPi[N,E,S,W] <-── STN <-── GPe[N,E,S,W]
                  │
                  v (disinhibition)
              thal[N,E,S,W]
                  │
                  v
              motor[N,E,S,W]   (lateral inhibition between)

DA modulation: midbrain DA neurons (A9 SNc / A10 VTA, collapsed in this
model) project to all striatal pools. DA enhances the direct pathway
(D1-class receptor, Gs-coupled, LTP-biased) and suppresses the indirect
pathway (D2-class receptor, Gi-coupled, LTD-biased). Per Kandel ch 43.

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
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ───────────────────────────────────────────────────────────────────────
# CUDA determinism (must be set BEFORE cupy/cuBLAS init).
# Triggered by --deterministic flag in argv. Tightens seed-to-seed noise
# floor (per the 2026-04-29 finding that A+E single-goal det gave
# 3.31 +/- 0.74 vs documented 4.08 +/- 0.49 — same code, +/-3-5 noise
# without determinism). ~10-30% slowdown.
# ───────────────────────────────────────────────────────────────────────
if "--deterministic" in sys.argv:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np


ACTION_NAMES = ["N", "E", "S", "W"]
N_ACTIONS = 4


def build_bg_brain_regions(
    n_cortex: int = 100,
    n_striatum_per_action: int = 50,
    n_gpe_per_action: int = 10,
    n_gpe_arky_per_action: int = 4,  # R3.7: arkypallidal (PV-) subpool
    n_str_patch_per_action: int = 8,  # R3.11: striosome (patch) subpool
    enable_cluster_a_closed_loop: bool = False,  # Cluster A: hyperdirect + thal->cortex
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
    # Cortex-level WTA (Phase B follow-up to plastic-input-layer cold-start).
    # Adds per-pool FS interneurons that mediate cross-pool inhibition.
    # Mirrors motor WTA pattern. Goal: enforce one-cortex-pool-wins regardless
    # of how noisy the input drive is. Lets hippocampus / sensory plastic layers
    # add drive on top of heuristic without washing out cascade selectivity.
    enable_cortex_lateral_inhibition: bool = False,
    n_cortex_fs_per_action: int = 5,
    # Scaled down 2.5x from motor WTA values: cortex pools are 25 neurons each
    # (vs 10 for motor), so density=1.0 gives 2.5x more synapses. Compensating
    # keeps total drive into/from FS comparable to motor case.
    cortex_to_fs_weight: float = 20.0,
    fs_to_cortex_weight: float = 8.0,
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
    # Working memory in PFC (Item 3, 2026-04-27).
    # Adds a prefrontal cortex region with recurrent internal connectivity
    # to support persistent activity (working memory). Real PFC neurons
    # show sustained firing across delay periods to maintain task-relevant
    # information. With this region, goal_cells project to PFC (plastic),
    # PFC has dense recurrent connectivity (plastic), PFC projects to
    # cortex (plastic). Tests whether PFC can hold goal info across delays.
    enable_pfc: bool = False,
    n_pfc: int = 60,
    pfc_internal_density: float = 0.2,  # recurrent connectivity for persistence
    goal_to_pfc_weight: float = 8.0,
    pfc_to_cortex_weight: float = 8.0,
    # Cheat #5: BG cross-projections (2026-04-27).
    # Default: cortex_X → str_D1_X only (same-action). Real biology has
    # cross-projections (cortex_E might also project weakly to str_D1_W,
    # learnable). With cross-projections enabled, all 16 cortex×D1 pairs
    # exist, but with cross-projections starting weak. Plasticity should
    # learn to weaken/strengthen them appropriately.
    enable_bg_cross_projections: bool = False,
    cross_projection_weight: float = 5.0,  # weak vs same-action 25.0
    cross_projection_density: float = 1.0,  # 1.0 = dense (24 cross-pathways); 0.25 = patch-matrix-like (6 of 24)
    cross_projection_topology_seed: int = 0,  # deterministic pathway selection when density < 1.0
    # Goal-beacon perception (Item 1 Stage 1, 2026-04-27 skeleton).
    # Replaces direct (gx, gy) goal access with beacon sensors that detect
    # beacon strength + direction (modeling biological cue perception).
    # Skeleton only — full wiring in trial loop deferred to next session.
    # See docs/plans/2026-04-27-perception-arc-plan.md for the full plan.
    enable_beacon_perception: bool = False,
    n_beacon_sensors: int = 8,  # 8 directional sensors (cardinal + diagonal)
    beacon_to_goal_weight: float = 8.0,
    # Landmark perception (Item 1 Stage 2, 2026-04-27).
    # Adds landmark_sensors region perceiving a FIXED-position landmark
    # (typically grid center). Used to self-organize place_cells via
    # plastic landmark_sensors → place_cells pathway. With a known fixed
    # landmark at L and 8 directional sensors, the (distance, bearing)
    # to L uniquely identifies agent position — place cells can learn to
    # fire at specific positions based on this multi-cell sensor pattern.
    enable_landmarks: bool = False,
    n_landmark_sensors: int = 8,
    landmark_to_place_weight: float = 8.0,
    # v3 (2026-04-28): MSN cross-pool lateral inhibition. Real BG sharpens
    # action selection via GABAergic collaterals between MSNs (within and
    # between action pools), striatal FS interneurons, and pallidal
    # center-surround. v3 adds the cross-pool MSN→MSN piece (the simplest
    # and most impactful). Without this, cross-projections (cheat #5)
    # corrupt the cascade because there's nothing to suppress cross-talk.
    # Static (plastic=False). MSN regions are GABAergic (exc_fraction=0.05)
    # so the projection is inhibitory.
    enable_bg_lateral_inhibition: bool = False,
    lateral_inhibition_density: float = 0.3,
    lateral_inhibition_weight: float = 2.0,
    # Cluster B.2 (2026-04-28): striatal fast-spiking interneurons.
    # Real BG striatum has ~1% PV-positive FSIs that provide fast convergent
    # GABAergic broadcast inhibition. Different from v3 MSN-MSN lateral
    # (slower, more local) — FSIs broadcast indiscriminately on a
    # millisecond timescale to bias which action's MSN pool wins.
    # Per-action FS pool receives same-action cortex drive, then inhibits
    # ALL striatal MSN pools (D1+D2, every action including same-action).
    # All FS pathways plastic=False (static gating, not plastic).
    # NOTE: kwargs are prefixed `cortex_to_str_fs_*` / `str_fs_to_msn_*` to
    # avoid collision with the cortex-WTA `cortex_to_fs_weight` (line 84)
    # and `fs_to_cortex_weight` (line 85) — different microcircuit.
    enable_striatal_fsis: bool = False,
    n_striatal_fs_per_action: int = 5,
    cortex_to_str_fs_weight: float = 30.0,
    # Cluster B.2 retune (2026-04-28 evening): initial guess of 8.0 caused
    # over-suppression — winner pool got suppressed by 35% (12.8 Hz drop)
    # while loser only got 1.6 Hz drop. With density=1.0 and 4 FS source
    # pools, effective inhibition was 32 (vs v3 lateral inhibition ~7).
    # Lowering to 2.0 → effective ~8, comparable to v3 lateral.
    str_fs_to_msn_weight: float = 2.0,
    # Cluster D v1 (2026-04-29): hippocampus trisynaptic loop.
    # Adds 5 regions (ec, dg, dg_fs, ca3, ca1) and ~10 pathways implementing
    # the canonical Cajal trisynaptic loop:
    #   sensory + landmark_sensors -> ec
    #   ec -> dg (perforant path), ec -> dg_fs (FFi recruitment)
    #   dg_fs -> dg (strong feedforward inhibition for sparsity)
    #   ec -> ca1 (direct cortical bypass)
    #   dg -> ca3 (mossy fibers; sparse but strong)
    #   ca3 -> ca3 (recurrent autoassociator; via region.internal_density)
    #   ca3 -> ca1 (Schaffer collaterals)
    #   ca1 -> place_cells (readout into existing perception arc, when
    #     enable_hippocampus is on; otherwise CA1 still exists but its
    #     readout pathway into place_cells is omitted).
    # Composition: ADDS to existing perception arc; does NOT replace
    # place_cells/goal_cells regions or landmark_sensors -> place_cells.
    # See docs/plans/2026-04-29-cluster-d-hippocampus-design.md.
    enable_cluster_d_hippocampus: bool = False,
    # Cluster E v1 (2026-04-29): topographic maps + distance-dependent
    # connection probability. When enabled:
    #   - cortex_X / str_D1_X / str_D2_X regions get 2D coordinates anchored
    #     to a corner of the unit square (N=(0.5,1.0), E=(1.0,0.5),
    #     S=(0.5,0.0), W=(0.0,0.5)).
    #   - cortex_X -> str_D1_X / str_D2_X pathways are sampled with
    #     Gaussian-weighted probability (sigma=0.3 by default).
    # Default off — backward compatible.
    # See docs/plans/2026-04-29-cluster-e-topographic-maps-design.md.
    enable_cluster_e_topography: bool = False,
    cluster_e_distance_sigma: float = 0.3,
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

    # Goal-beacon perception (Item 1 Stage 1 skeleton, 2026-04-27). Replaces
    # direct (gx, gy) goal access with directional beacon sensors. Each sensor
    # has a preferred bearing; activation is proportional to beacon intensity
    # × cosine alignment with sensor direction. Plastic beacon → goal_cells
    # pathway lets goal_cells learn to integrate sensor patterns into spatial
    # representations. Full trial-loop wiring deferred to next session.
    if enable_beacon_perception:
        regions.append(BrainRegion(
            name="beacon_sensors",
            n_neurons=n_beacon_sensors,
            exc_fraction=1.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))

    # Landmark perception (Item 1 Stage 2, 2026-04-27). Fixed-position
    # landmark with 8 directional sensors. Plastic landmark_sensors →
    # place_cells pathway lets place cells self-organize from the unique
    # (distance, bearing) pattern at each agent position. Replaces direct
    # (x, y) place cell access with biologically-grounded localization.
    if enable_landmarks:
        regions.append(BrainRegion(
            name="landmark_sensors",
            n_neurons=n_landmark_sensors,
            exc_fraction=1.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))

    # PFC working memory (Item 3, 2026-04-27): recurrent prefrontal region.
    # Internal density > 0 enables recurrent connections that can sustain
    # activity across delay periods (persistent activity / attractor dynamics).
    # PFC pyramidal preset has biophysical features for sustained firing.
    if enable_pfc:
        regions.append(BrainRegion(
            name="dlpfc_wm",
            n_neurons=n_pfc,
            exc_fraction=0.8,
            internal_density=pfc_internal_density,
            exc_weight_mean=2.0,  # moderate self-excitation for persistence
            inh_weight_mean=4.0,
            weight_jitter=0.2,
            plastic_internal=True,  # plastic recurrence supports learning
            izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name,
            # IZH2007_HIPPO_PYRAMIDAL works for PFC-style dynamics; can switch
            # to dedicated PFC preset (HH_PFC_PYRAMIDAL) for full biophysics.
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
    # Cluster C v2 (2026-04-29): action_index stamped on action-specific
    # regions so cp_synapse_action_tag can resolve per-synapse DA targeting.
    # Cluster E v1 (2026-04-29): topographic 2D coordinates per action
    # corner of unit square when enable_cluster_e_topography is on.
    n_cortex_per_action = n_cortex // N_ACTIONS
    # Cardinal-direction corners of the unit square (Cluster E v1).
    _action_corner = {
        "N": (0.5, 1.0),
        "E": (1.0, 0.5),
        "S": (0.5, 0.0),
        "W": (0.0, 0.5),
    }
    _topo_kw = (
        {"coordinate_dim": 2, "coordinate_extent": (1.0, 1.0)}
        if enable_cluster_e_topography
        else {}
    )
    # cortex_{N,E,S,W}: per-action motor-cortex (M1-equivalent) pools.
    # Anatomy: regular-spiking pyramidal neurons (RS preset). The "cortex_"
    # prefix is project shorthand; biologically these stand in for primary
    # motor cortex columns wired in topographic action channels (cf.
    # Cluster E catalog, Kandel 6e Ch 38). Each pool drives the
    # corresponding striatal D1/D2 channel (cortex -> str_d1_X / str_d2_X).
    for action_idx, action in enumerate(ACTION_NAMES):
        kw = dict(_topo_kw)
        if enable_cluster_e_topography:
            kw["coordinate_center"] = _action_corner[action]
        regions.append(BrainRegion(
            name=f"cortex_{action}",
            n_neurons=n_cortex_per_action,
            exc_fraction=1.0,  # All excitatory for cortex inputs
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
            action_index=action_idx,
            **kw,
        ))

    # Cortex WTA microcircuit (opt-in). Per-pool FS interneurons that mediate
    # cross-pool inhibition: cortex_X drives cortex_FS_X, which inhibits
    # cortex_{Y,Z,W}. Standard cortical WTA pattern, mirror of motor WTA.
    # Goal: enforce clean pool selectivity even when plastic input layers
    # (hippocampus, learned-perception) add noisy drive across all 4 pools.
    if enable_cortex_lateral_inhibition:
        for action_idx, action in enumerate(ACTION_NAMES):
            regions.append(BrainRegion(
                name=f"cortex_FS_{action}",
                n_neurons=n_cortex_fs_per_action,
                exc_fraction=0.0,  # all-inhibitory → outgoing synapses are inhibitory
                internal_density=0.0,
                exc_weight_mean=0.0, inh_weight_mean=0.0,
                weight_jitter=0.0, plastic_internal=False,
                izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name,
                action_index=action_idx,
            ))

    # Per-action striatal pools (D1 direct, D2 indirect).
    # internal_density=0 (no lateral inhibition) initially — MSNs need
    # strong cortex drive to escape the down-state and lateral inhibition
    # makes that even harder. Add it back later if action selection needs
    # sharpening.
    for action_idx, action in enumerate(ACTION_NAMES):
        # Striatal MSNs: ECl ~−60 mV (PBR-160 ch 6, gramicidin perforated patch).
        # IPSPs are shunting near rest, hyperpolarizing only near AP threshold.
        # Cluster E v1: same per-action corner as cortex for topographic
        # cortex_X -> str_D{1,2}_X mapping.
        msn_kw = dict(_topo_kw)
        if enable_cluster_e_topography:
            msn_kw["coordinate_center"] = _action_corner[action]
        regions.append(BrainRegion(
            name=f"str_D1_{action}",
            n_neurons=n_striatum_per_action,
            exc_fraction=0.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D1.name,
            syn_reversal_potential_i_override=-60.0,
            action_index=action_idx,
            **msn_kw,
        ))
        regions.append(BrainRegion(
            name=f"str_D2_{action}",
            n_neurons=n_striatum_per_action,
            exc_fraction=0.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D2.name,
            syn_reversal_potential_i_override=-60.0,
            action_index=action_idx,
            **msn_kw,
        ))

    # Cluster B.2 (2026-04-28): striatal fast-spiking interneurons (FSIs).
    # ~1% of striatal cells; PV-positive; broadcast inhibition. One small
    # str_PV_FSI_{N,E,S,W}: per-action striatal fast-spiking interneurons.
    # Strict naming: this is the **PV-FSI** class (parvalbumin-positive
    # fast-spiking) — one of EIGHT distinct striatal GABAergic interneuron
    # classes catalogued in Tepper-2018 (the others are NPY-LTS, NPY-NGF,
    # CR, TH/THIN, FAI, SABI, plus the cholinergic ChI/TAN). The "str_FS"
    # prefix in this codebase models PV-FSI specifically — it is NOT a
    # generic "all striatal interneurons" pool. The class is named "FS"
    # for its short-AP / high-rate firing (Tepper 2018 ch 8). Catalog
    # ref: TK-2017 ch 8; Tepper 2018 §"Functional Significance".
    # FS pool per action, all GABAergic (exc_fraction=0.0) so the outgoing
    # synapses are auto-derived inhibitory by the bridge. No internal
    # recurrence: FSIs just receive cortex drive and broadcast to all MSNs.
    if enable_striatal_fsis:
        for action_idx, action in enumerate(ACTION_NAMES):
            regions.append(BrainRegion(
                name=f"str_PV_FSI_{action}",
                n_neurons=n_striatal_fs_per_action,
                exc_fraction=0.0,  # all-inhibitory → outgoing synapses are inhibitory
                internal_density=0.0,
                exc_weight_mean=0.0, inh_weight_mean=0.0,
                weight_jitter=0.0, plastic_internal=False,
                izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name,
                action_index=action_idx,
            ))

    # Per-action BG output (GPe / GPi)
    # R3.7 (2026-04-29): GPe is split into PV+ (prototypic) and PV-
    # (arkypallidal) subpools per Mallet 2008 / Kita 2007 (PBR-160 ch 7).
    # gpe_X = prototypic (PV+), forming the canonical GPe -> STN/GPi/SNr
    # projection. gpe_arky_X = arkypallidal (PV-), forming the
    # GPe -> striatum feedback (broadcasts onto FSIs, "stop-signal"
    # role per Mallet 2012). Sizes: PV+ at the original n_gpe_per_action
    # (10), PV- at n_gpe_arky_per_action (4) — consistent with Kita's
    # observation that PV-negative cells form ~1/3 of GPe.
    for action_idx, action in enumerate(ACTION_NAMES):
        regions.append(BrainRegion(
            name=f"gpe_{action}",  # prototypic (PV+); existing alias preserved
            n_neurons=n_gpe_per_action,
            exc_fraction=0.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_GPE_PACEMAKER.name,
            action_index=action_idx,
        ))
        regions.append(BrainRegion(
            name=f"gpe_arky_{action}",  # arkypallidal (PV-); R3.7 new pool
            n_neurons=n_gpe_arky_per_action,
            exc_fraction=0.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_GPE_PACEMAKER.name,
            action_index=action_idx,
        ))
        # gpi_{N,E,S,W}: BG-output complex per action (GPi/SNr in primates;
        # predominantly SNr in rodents — internal-pallidal cells are sparse
        # in rats/mice and SNr carries most output-nucleus work). Tonic
        # 40-80 Hz GABAergic projection neurons. Disinhibition via direct
        # pathway (D1 MSN -> GPi/SNr) is the canonical "go" mechanism.
        # Catalog refs: Kandel 6e Ch 38 p 935-943; PBR-160 ch 9 Deniau.
        regions.append(BrainRegion(
            name=f"gpi_{action}",
            n_neurons=n_gpi_per_action,
            exc_fraction=0.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_GPI_OUTPUT.name,
            action_index=action_idx,
        ))
        # R3.11 (2026-04-29): striosome (patch) compartment.
        # Per PBR-160 ch 9 / ch 11: striosomes are D1-MSN-rich patches
        # that project to BOTH SNc (canonical, drives DA) and SNr (gpi)
        # in addition to the matrix-pathway. The patch/matrix split
        # aligns with SNc/SNr at the output level. Real input is limbic
        # (vmPFC, amygdala, ventral hippocampus); we use cortex_X as a
        # placeholder until a limbic source is added (Cluster O work).
        # E_inh override -60 mV is inherited via the same MSN-class
        # convention applied to str_D1/D2.
        regions.append(BrainRegion(
            name=f"str_patch_{action}",
            n_neurons=n_str_patch_per_action,
            exc_fraction=0.05,  # MSN is GABAergic with sparse glutamatergic spillover
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D1.name,
            syn_reversal_potential_i_override=-60.0,  # MSN GABA_A reversal (R1.1)
            action_index=action_idx,
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
    for action_idx, action in enumerate(ACTION_NAMES):
        regions.append(BrainRegion(
            name=f"thal_{action}",
            n_neurons=n_thal_per_action,
            exc_fraction=1.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_THALAMIC_RELAY.name,
            action_index=action_idx,
        ))
        regions.append(BrainRegion(
            name=f"motor_{action}",
            n_neurons=n_motor_per_action,
            exc_fraction=1.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
            action_index=action_idx,
        ))

    # SNc dopamine neurons (single pool, broadcasts via neuromodulator subsystem).
    # Anatomy note: this region is the project's A9-equivalent — SNc
    # dopaminergic neurons that drive nigrostriatal projections. The
    # mesolimbic A10/VTA → NAc/PFC arms are NOT separately modeled; the
    # single `snc` pool collapses A9 + A10 into one broadcast modulator.
    # With Cluster C v2 (`--enable-compartmentalized-da`), per-action DA
    # channels (dopamine_{N,E,S,W}) decompose this into per-action
    # targeting, though still A9-typed. The transmitter (`dopamine`
    # neuromodulator) keeps its canonical chemistry name; only the
    # *region* renamed from "dopamine" → "snc" 2026-04-29 (Wave-1 #3).
    # Catalog refs: Kandel 6e Ch 11 (DA system); PBR-160 ch 11 (Tepper & Lee).
    # SNc DA neurons lack KCC2 → ECl ~−55 mV (PBR-160 ch 11). GABA_A is
    # depolarizing or even excitatory at rest in adult SNc; override the
    # cortical-pyramidal default of −75 mV.
    regions.append(BrainRegion(
        name="snc",
        n_neurons=n_dopamine,
        exc_fraction=1.0,
        internal_density=0.0,
        exc_weight_mean=0.0, inh_weight_mean=0.0,
        weight_jitter=0.0, plastic_internal=False,
        izh_neuron_type=NeuronType.IZH2007_DOPAMINE.name,
        syn_reversal_potential_i_override=-55.0,
    ))

    # Cluster D v1 (2026-04-29): hippocampus trisynaptic loop.
    # Five new regions implementing the canonical Cajal loop. See
    # docs/plans/2026-04-29-cluster-d-hippocampus-design.md.
    #   ec (entorhinal cortex stub) — receives sensory + landmark, projects
    #     to DG, CA1; bridges perception to hippocampus proper.
    #   dg (dentate gyrus) — pattern separation via FFi-driven sparsity;
    #     internal_density=0 (no recurrence — DG granule cells fire sparsely).
    #   dg_fs — fast-spiking interneurons providing strong feedforward
    #     inhibition (exc_fraction=0.0 → outputs auto-derived inhibitory).
    #   ca3 — pattern completion; recurrent autoassociator core
    #     (internal_density=0.30 generates the dense recurrent collaterals).
    #   ca1 — readout integrating direct EC input + CA3 output; projects
    #     into existing place_cells region when enable_hippocampus is on.
    if enable_cluster_d_hippocampus:
        regions.append(BrainRegion(
            name="ec",
            n_neurons=80,
            exc_fraction=0.8,
            internal_density=0.05,
            exc_weight_mean=0.3, inh_weight_mean=0.8,
            weight_jitter=0.2, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))
        regions.append(BrainRegion(
            name="dg",
            n_neurons=200,
            exc_fraction=0.95,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name,
        ))
        regions.append(BrainRegion(
            name="dg_fs",
            n_neurons=60,
            exc_fraction=0.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name,
        ))
        regions.append(BrainRegion(
            name="ca3",
            n_neurons=100,
            exc_fraction=0.85,
            internal_density=0.30,  # dense recurrent autoassociator
            exc_weight_mean=1.5, inh_weight_mean=2.0,
            weight_jitter=0.2, plastic_internal=True,  # recurrent CA3 plasticity
            izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name,
        ))
        regions.append(BrainRegion(
            name="ca1",
            n_neurons=120,
            exc_fraction=0.85,
            internal_density=0.05,
            exc_weight_mean=0.3, inh_weight_mean=0.8,
            weight_jitter=0.2, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name,
        ))

    # ---- Pathways (cross-region projections) ----

    # Sensory → cortex (LEARNING site for perception, opt-in).
    # Plastic; agent learns position-to-action mapping via STDP + reward.
    # Each sensory neuron projects to all 4 cortex pools; learning shapes
    # which sensory patterns drive which cortex action pool.
    # Tagged with plasticity_gate="sensory_to_cortex" so curriculum can
    # stage perceptual learning: frozen during cortex warmup, thawed
    # during phase 2 to learn position→action mapping with the heuristic
    # as teacher (additive, not mutually exclusive).
    if enable_learned_perception:
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region="sensory", to_region=f"cortex_{action}",
                density=1.0, weight_mean=sensory_to_cortex_weight,
                weight_jitter=0.2, plastic=True,
                plasticity_gate="sensory_to_cortex",
            ))

    # Hippocampus → cortex (LEARNING site, opt-in).
    # Plastic; agent learns (place, goal) → action via STDP + reward.
    # Place cells provide spatial context (where am I), goal cells provide
    # task context (where do I want to be). Together they should learn
    # the full position-action mapping.
    # Tagged with plasticity_gate="hippo_to_cortex" so runners can
    # implement curriculum: freeze during cortex-warmup, thaw later.
    if enable_hippocampus:
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region="place_cells", to_region=f"cortex_{action}",
                density=1.0, weight_mean=hippocampus_to_cortex_weight,
                weight_jitter=0.2, plastic=True,
                plasticity_gate="hippo_to_cortex",
            ))
            pathways.append(RegionPathway(
                from_region="goal_cells", to_region=f"cortex_{action}",
                density=1.0, weight_mean=hippocampus_to_cortex_weight,
                weight_jitter=0.2, plastic=True,
                plasticity_gate="hippo_to_cortex",
            ))

    # Beacon perception pathway (Item 1 Stage 1 skeleton, 2026-04-27).
    # Beacon sensors → goal_cells: tagged plasticity_gate="beacon_to_goal"
    # for curriculum-staged learning. With curriculum, this pathway is frozen
    # during cortex warmup (when heuristic provides selectivity) and thawed
    # in phase 2 to learn beacon-pattern → goal-cell-position mapping.
    # NOTE: full trial-loop wiring (driving beacon_sensors based on beacon
    # position) is deferred to next session. Currently the region exists
    # but isn't driven, so enable_beacon_perception is a no-op until the
    # trial loop is updated.
    if enable_beacon_perception and enable_hippocampus:
        pathways.append(RegionPathway(
            from_region="beacon_sensors", to_region="goal_cells",
            density=1.0, weight_mean=beacon_to_goal_weight,
            weight_jitter=0.2, plastic=True,
            plasticity_gate="beacon_to_goal",
        ))

    # Landmark → place cells pathway (Item 1 Stage 2, 2026-04-27).
    # Plastic; place cells self-organize from landmark sensor patterns.
    # Each unique (distance, bearing) to landmark gives a unique sensor
    # activation pattern, so place cells learn to fire at specific positions.
    if enable_landmarks and enable_hippocampus:
        pathways.append(RegionPathway(
            from_region="landmark_sensors", to_region="place_cells",
            density=1.0, weight_mean=landmark_to_place_weight,
            weight_jitter=0.2, plastic=True,
            plasticity_gate="landmark_to_place",
        ))

    # PFC working memory pathways (Item 3, 2026-04-27):
    #   goal_cells → PFC: goal info enters working memory
    #   PFC → cortex_X: PFC drives cortex selection across delays
    # Both tagged with plasticity_gate="dlpfc_wm_pathways" so curriculum can
    # stage PFC learning. Internal PFC connectivity is plastic_internal=True
    # for recurrent learning (gated by "pfc_internal" if needed).
    if enable_pfc:
        if enable_hippocampus:
            # goal_cells → PFC for working memory of goal
            pathways.append(RegionPathway(
                from_region="goal_cells", to_region="dlpfc_wm",
                density=0.5, weight_mean=goal_to_pfc_weight,
                weight_jitter=0.2, plastic=True,
                plasticity_gate="dlpfc_wm_pathways",
            ))
        # PFC → cortex (action selection driven by working memory)
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region="dlpfc_wm", to_region=f"cortex_{action}",
                density=0.5, weight_mean=pfc_to_cortex_weight,
                weight_jitter=0.2, plastic=True,
                plasticity_gate="dlpfc_wm_pathways",
            ))

    # Cortex -> striatum (LEARNING site).
    # Each cortex_X projects strongly to its corresponding str_D1_X / str_D2_X
    # AND (if enable_bg_cross_projections) weakly to other actions' striatum.
    # Same-action paths are tagged with plasticity_gate="corticostriatal" so the
    # curriculum can freeze cortex→striatum once mature.
    # Cross-projections are tagged with plasticity_gate="bg_cross_projections"
    # (separate gate, 2026-04-28) so the curriculum can stage them
    # independently — keep them frozen during phase 1+2 (don't accumulate
    # phase-0 motor bias), thaw post-goal-change in phase 3 so STDP+reward
    # can shape cross-action routing symmetrically.
    # Patch-matrix sparsity (2026-04-28, option 2): if cross_projection_density < 1.0,
    # randomly skip cross-pathways at build time to mirror real BG patch-matrix
    # anatomy (~10-25% cross-projection density). Selection is deterministic
    # given cross_projection_topology_seed so reruns reproduce the same topology.
    import random as _random
    _topology_rng = _random.Random(cross_projection_topology_seed)
    _all_cross_pairs = [(c, s) for c in ACTION_NAMES for s in ACTION_NAMES if c != s]
    _n_keep = max(0, int(round(len(_all_cross_pairs) * cross_projection_density)))
    _selected_cross = set(_topology_rng.sample(_all_cross_pairs, _n_keep))

    # R3.5 (2026-04-29): cortex->MSN density tightened to 0.20 (was 1.0)
    # per Bolam-2000 / Kincaid 1998 (catalog ref). At our scale (25 cortex
    # x 50 MSN per pool) density 0.20 ~ 5 cortex inputs per MSN, ~10 MSN
    # targets per cortex axon — matches "sparse + decorrelated" biological
    # convergence. Original density=1.0 was anatomically dense (every
    # cortex neuron synapsing every MSN). Re-tunable via runner kwarg
    # cortex_to_msn_density if needed; weight_mean kept at 25.0 to
    # maintain net excitatory drive given the sparser fan-in.
    cortex_to_msn_density_same = float(locals().get("cortex_to_msn_density_same_override", 0.20))
    cortex_to_msn_density_cross = 0.10  # sparser still per Bolam
    # R3.5 follow-up (2026-04-29 morning diagnostic): density 1.0 -> 0.20 reduced
    # cortex->MSN drive ~5x, which empirically silenced motor pools (1798/1800
    # trials all-zero motor counts at seed 42). To preserve effective drive
    # while honoring Bolam-2000 "few synapses per pair" biology, scale weight
    # inversely with density. Original (density=1.0, weight=25) -> default scaled
    # weight at density=0.20 is 25/0.2 = 125. Override via cortex_to_msn_weight_override
    # kwarg if needed.
    if cortex_to_msn_density_same < 1.0:
        # Scale weight to compensate density reduction. Original (density=1.0, weight=25)
        # gives 25 weight-units per cortex-MSN pair on average. After R3.5's density=0.20,
        # naive weight=25 gives 5 weight-units (5x weaker drive). Compensating gives
        # weight = 25 / density = 125 at density 0.20.
        cortex_to_msn_weight_same = 25.0 / cortex_to_msn_density_same
    else:
        cortex_to_msn_weight_same = 25.0
    cortex_to_msn_weight_same = float(locals().get("cortex_to_msn_weight_same_override", cortex_to_msn_weight_same))
    # When using sparse density (post-R3.5 default 0.20), scale weight to recover drive.
    # Setting cortex_to_msn_weight_same_override=25.0 reverts to the broken
    # weak-cascade behavior; setting density_same=1.0 reverts to pre-R3.5.
    # Cluster E v1 (2026-04-29): when topography is on, cortex_X -> str_D{1,2}_X
    # pathways carry distance_sigma so connections are Gaussian-weighted by
    # 2D corner distance. Same-action pairs share the same corner (distance=0,
    # full density); cross-action pairs are 1.0 unit apart (heavily attenuated
    # at sigma=0.3). Falls back to uniform Bernoulli when the flag is off.
    _cluster_e_sigma = (
        float(cluster_e_distance_sigma)
        if enable_cluster_e_topography
        else None
    )
    for cortex_action in ACTION_NAMES:
        for str_action in ACTION_NAMES:
            same = (cortex_action == str_action)
            if same:
                density = cortex_to_msn_density_same
                weight = cortex_to_msn_weight_same
                gate = "corticostriatal"
            elif enable_bg_cross_projections and (cortex_action, str_action) in _selected_cross:
                density = cortex_to_msn_density_cross
                weight = cross_projection_weight
                gate = "bg_cross_projections"
            else:
                continue
            pathways.append(RegionPathway(
                from_region=f"cortex_{cortex_action}",
                to_region=f"str_D1_{str_action}",
                density=density, weight_mean=weight, weight_jitter=0.2, plastic=True,
                plasticity_gate=gate,
                distance_sigma=_cluster_e_sigma,
            ))
            pathways.append(RegionPathway(
                from_region=f"cortex_{cortex_action}",
                to_region=f"str_D2_{str_action}",
                density=density, weight_mean=weight, weight_jitter=0.2, plastic=True,
                plasticity_gate=gate,
                distance_sigma=_cluster_e_sigma,
            ))

    # v3 (2026-04-28): MSN cross-pool lateral inhibition.
    # Adds str_D1_X → str_D1_Y and str_D2_X → str_D2_Y for X != Y. MSNs are
    # GABAergic (exc_fraction=0.05), so these projections IS inhibitory —
    # firing in pool X suppresses firing in pool Y, sharpening action
    # selection. Real BG has GABAergic MSN collaterals plus FS interneurons
    # for stronger feed-forward inhibition. v3 covers the MSN-collateral
    # piece. FS interneurons + pallidal center-surround are v3.5 if needed.
    # Static (plastic=False): lateral inhibition is a structural feature.
    # 4 cortex actions × 3 cross targets × 2 (D1/D2) = 24 new pathways.
    if enable_bg_lateral_inhibition:
        for src_action in ACTION_NAMES:
            for dst_action in ACTION_NAMES:
                if src_action == dst_action:
                    continue
                for d_type in ("D1", "D2"):
                    pathways.append(RegionPathway(
                        from_region=f"str_{d_type}_{src_action}",
                        to_region=f"str_{d_type}_{dst_action}",
                        density=lateral_inhibition_density,
                        weight_mean=lateral_inhibition_weight,
                        weight_jitter=0.2,
                        plastic=False,
                    ))

    # Cluster B.2 (2026-04-28, R1.2 rewire 2026-04-29): striatal FSI pathways.
    # (a) cortex_X → str_PV_FSI_X (excitatory, dense, plastic=False, same-action only).
    #     FS pool gets driven only by its same-action cortex pool.
    # (b) str_PV_FSI_X → str_D{1,2}_Y for X != Y ONLY (cross-action feedforward
    #     inhibition; auto-derived inhibitory because str_FS regions have
    #     exc_fraction=0.0). 4 FS × 3 cross D-pool × 2 D-types = 24 paths.
    #
    # Biological grounding (Tepper-2018 pp 8–9; Tepper, Koós & Wilson, TK-2017
    # pp 161–163): paired-recording studies show MSN→MSN collaterals deliver
    # only ~0.5 mV unitary IPSPs at 14–25% connection probability with high
    # failure rates and short-term depression — i.e., MSN-MSN lateral
    # inhibition is functionally weak. By contrast, FSI→MSN feedforward
    # IPSPs are significantly larger and more reliable, and FSIs preferentially
    # innervate MSNs of OTHER action channels (cross-action). This makes the
    # FSI cross-action projection the dominant biological substrate for the
    # striatal WTA microcircuit. The previous (R1.1) within-action broadcast
    # was anatomically inaccurate; we now restrict FS_X to MSN_Y for Y != X.
    # The v3 `--bg-lateral-inhibition` MSN→MSN flag is now redundant with
    # this cross-action FSI WTA but is kept opt-in for backward compatibility.
    if enable_striatal_fsis:
        # (a) cortex_X → str_PV_FSI_X (excitatory drive, same-action)
        for cortex_action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region=f"cortex_{cortex_action}",
                to_region=f"str_PV_FSI_{cortex_action}",
                density=1.0,
                weight_mean=cortex_to_str_fs_weight,
                weight_jitter=0.2,
                plastic=False,
            ))
        # (b) str_PV_FSI_X → str_D{1,2}_Y for X != Y only (cross-action WTA;
        # FSIs do NOT inhibit their own action's MSN pool).
        for fs_action in ACTION_NAMES:
            for str_action in ACTION_NAMES:
                if fs_action == str_action:
                    continue  # skip within-action — FSIs target other channels
                for d_type in ("D1", "D2"):
                    pathways.append(RegionPathway(
                        from_region=f"str_PV_FSI_{fs_action}",
                        to_region=f"str_{d_type}_{str_action}",
                        density=1.0,  # dense within-pool
                        weight_mean=str_fs_to_msn_weight,
                        weight_jitter=0.2,
                        plastic=False,
                    ))

    # Direct pathway: D1 -> GPi (inhibitory). Strong weight needed to overcome
    # GPi tonic firing (~30-75 Hz baseline).
    for action in ACTION_NAMES:
        pathways.append(RegionPathway(
            from_region=f"str_D1_{action}", to_region=f"gpi_{action}",
            density=1.0, weight_mean=15.0, weight_jitter=0.2, plastic=False,
        ))

    # Indirect pathway: D2 -> GPe (PV+) -> STN -> GPi
    for action in ACTION_NAMES:
        pathways.append(RegionPathway(
            from_region=f"str_D2_{action}", to_region=f"gpe_{action}",
            density=0.6, weight_mean=2.5, weight_jitter=0.2, plastic=False,
        ))
        pathways.append(RegionPathway(
            from_region=f"gpe_{action}", to_region="stn",
            density=0.3, weight_mean=1.5, weight_jitter=0.2, plastic=False,
        ))

    # R3.7 (2026-04-29): arkypallidal (PV-) GPe subpool. D2 also drives
    # arky cells; arky projects back to striatal FSIs broadcasting a
    # "stop signal" (Mallet 2012). Per Kita 2007 / Tepper-2018, PV-
    # cells rarely collateralize to STN/GPi -- their canonical target
    # is the striatum. Modeling as broadcast to all str_PV_FSI_Y so a single
    # action's D2 activation can feedback-inhibit the entire striatal
    # FSI population, halting ongoing motor commitments.
    if enable_striatal_fsis:  # arky->FSI requires FSI population
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region=f"str_D2_{action}", to_region=f"gpe_arky_{action}",
                density=0.5, weight_mean=2.0, weight_jitter=0.2, plastic=False,
            ))
            for fs_action in ACTION_NAMES:
                pathways.append(RegionPathway(
                    from_region=f"gpe_arky_{action}", to_region=f"str_PV_FSI_{fs_action}",
                    density=0.3, weight_mean=1.5, weight_jitter=0.2, plastic=False,
                ))
    else:
        # Without FSI population, arky has no striatal target. Still
        # receive D2 input so dynamics are correct; outputs are dropped.
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region=f"str_D2_{action}", to_region=f"gpe_arky_{action}",
                density=0.5, weight_mean=2.0, weight_jitter=0.2, plastic=False,
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

    # R3.11 (2026-04-29): striosome (patch) pathways.
    # cortex_X -> str_patch_X: placeholder for limbic input (vmPFC/amygdala/
    # ventral hippocampus per PBR-160 ch 9). Plastic so patch can learn
    # cortical-to-patch mapping. Same density as matrix per Bolam.
    # str_patch_X -> snc: canonical striosome->SNc projection driving
    # phasic DA (Tepper & Lee PBR-160 ch 11 p 191).
    # str_patch_X -> gpi_X: secondary striosome->SNr projection (PBR-160
    # ch 9 Deniau p 160 — striosomes contribute substantial direct input
    # to SNr in addition to the canonical SNc target). Smaller weight
    # than matrix's str_D1->gpi to reflect minor contribution.
    for action in ACTION_NAMES:
        pathways.append(RegionPathway(
            from_region=f"cortex_{action}", to_region=f"str_patch_{action}",
            density=cortex_to_msn_density_same, weight_mean=cortex_to_msn_weight_same,
            weight_jitter=0.2, plastic=True, plasticity_gate="corticostriatal",
        ))
        pathways.append(RegionPathway(
            from_region=f"str_patch_{action}", to_region="snc",
            density=0.4, weight_mean=2.5, weight_jitter=0.2, plastic=False,
        ))
        pathways.append(RegionPathway(
            from_region=f"str_patch_{action}", to_region=f"gpi_{action}",
            density=0.3, weight_mean=1.5, weight_jitter=0.2, plastic=False,
        ))

    # R3.10 (2026-04-29): GPi/SNr -> snc collateral disinhibition
    # (PBR-160 ch 11 Tepper & Lee pp 192-193, 199; Tepper et al. 1995).
    # SNr GABA neurons project to SNc DA neurons via axon collaterals;
    # the major in-vivo drive of spontaneous DA burst firing is the
    # SNr -> SNc disinhibition (when D1-mediated SNr silencing releases
    # tonic GABA suppression of DA cells, DA neurons burst). Combined
    # with R1.1 (E_inh = -55 mV on the snc region, since SNc lacks KCC2),
    # this gives a biologically grounded substrate for phasic DA without
    # external injection. NOTE: in our cascade we conflate SNr with GPi
    # (both GABAergic BG output nuclei); this is the standard rodent vs
    # primate naming difference rather than a separate population.
    for action in ACTION_NAMES:
        pathways.append(RegionPathway(
            from_region=f"gpi_{action}", to_region="snc",
            density=0.3, weight_mean=2.0, weight_jitter=0.2, plastic=False,
        ))

    # Thalamus -> motor cortex (excitatory). Very strong weight needed
    # because thal pool is small (10 cells) and we need ~50 Hz motor output
    # from ~24 Hz thal input.
    for action in ACTION_NAMES:
        pathways.append(RegionPathway(
            from_region=f"thal_{action}", to_region=f"motor_{action}",
            density=1.0, weight_mean=20.0, weight_jitter=0.2, plastic=False,
        ))

    # Cluster A (2026-04-29): closed BG loop.
    # (a) Hyperdirect pathway: cortex_X -> stn (Nambu 2002). ~30% of cortex
    #     pyramids project directly to STN, bypassing striatum. Sparse
    #     excitatory drive provides a fast global "stop" signal that
    #     biases against premature action commitment when multiple
    #     cortex pools fire simultaneously. Static (plastic=False) since
    #     anatomical projection is genetically specified, not learned.
    # (b) Thalamo-cortical feedback: thal_X -> cortex_X. Closes the
    #     cortex -> BG -> thal -> cortex loop. Action-specific (not
    #     cross-action) per VA/VL topographic organization. Provides the
    #     post-synaptic activity that lets STDP shape useful cross-action
    #     weights (the "teaching signal" missing for cross-projection
    #     learning per CLAUDE.md cheat-5 reframe). Static.
    if enable_cluster_a_closed_loop:
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region=f"cortex_{action}", to_region="stn",
                density=0.10, weight_mean=3.0, weight_jitter=0.2,
                plastic=False,
            ))
            pathways.append(RegionPathway(
                from_region=f"thal_{action}", to_region=f"cortex_{action}",
                density=0.50, weight_mean=5.0, weight_jitter=0.2,
                plastic=False,
            ))

    # ---- Motor lateral inhibition (opt-in) ----
    # FS interneuron sub-pool per motor pool. Each motor_X drives its own
    # motor_FS_X (excitatory), which in turn inhibits the other 3 motor pools.
    # This implements the cortical WTA microcircuit: when motor_X fires,
    # motor_FS_X fires, suppressing motor_{Y,Z,W}. Combined with BG gating,
    # this should sharpen action selection in cases where multiple cortex
    # pools drive simultaneously (currently the dominant random-fallback case).
    if enable_motor_lateral_inhibition:
        for action_idx, action in enumerate(ACTION_NAMES):
            regions.append(BrainRegion(
                name=f"motor_FS_{action}",
                n_neurons=n_motor_fs_per_action,
                exc_fraction=0.0,  # all-inhibitory → outgoing synapses are inhibitory
                internal_density=0.0,
                exc_weight_mean=0.0, inh_weight_mean=0.0,
                weight_jitter=0.0, plastic_internal=False,
                izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name,
                action_index=action_idx,
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

    # Cortex WTA pathways (opt-in). Mirror of motor WTA structure.
    if enable_cortex_lateral_inhibition:
        # cortex_X → cortex_FS_X (excitatory: cortex pool drives its FS)
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region=f"cortex_{action}", to_region=f"cortex_FS_{action}",
                density=1.0, weight_mean=cortex_to_fs_weight, weight_jitter=0.2,
                plastic=False,
            ))
        # cortex_FS_X → cortex_Y for Y != X (inhibitory: FS suppresses other pools)
        for src_action in ACTION_NAMES:
            for tgt_action in ACTION_NAMES:
                if src_action == tgt_action:
                    continue
                pathways.append(RegionPathway(
                    from_region=f"cortex_FS_{src_action}", to_region=f"cortex_{tgt_action}",
                    density=1.0, weight_mean=fs_to_cortex_weight, weight_jitter=0.2,
                    plastic=False,
                ))

    # ---- Cluster D v1 (2026-04-29): hippocampus trisynaptic loop pathways ----
    # See docs/plans/2026-04-29-cluster-d-hippocampus-design.md.
    # Pathways added when --enable-cluster-d-hippocampus is on:
    #   sensory -> ec (perceptual entry; only if --learned-perception)
    #   landmark_sensors -> ec (only if --landmarks; landmark_sensors region
    #     only exists in that case)
    #   ec -> dg (perforant path; main excitatory drive to DG)
    #   ec -> dg_fs (FFi recruitment)
    #   dg_fs -> dg (strong feedforward inhibition for sparsity)
    #   ec -> ca1 (direct cortical bypass)
    #   dg -> ca3 (mossy fibers; sparse but strong)
    #   ca3 -> ca3 (recurrent autoassociator — handled by region.internal_density)
    #   ca3 -> ca1 (Schaffer collaterals)
    #   ca1 -> place_cells (readout; only if --hippocampus, since place_cells
    #     region only exists then; coexists with landmark_sensors->place_cells)
    if enable_cluster_d_hippocampus:
        # sensory -> ec (only when learned-perception layer exists)
        if enable_learned_perception:
            pathways.append(RegionPathway(
                from_region="sensory", to_region="ec",
                density=0.40, weight_mean=4.0, weight_jitter=0.2,
                plastic=True, plasticity_gate="sensory_to_ec",
            ))
        # landmark_sensors -> ec (only when landmark_sensors region exists)
        if enable_landmarks:
            pathways.append(RegionPathway(
                from_region="landmark_sensors", to_region="ec",
                density=0.40, weight_mean=4.0, weight_jitter=0.2,
                plastic=True, plasticity_gate="sensory_to_ec",
            ))
        # ec -> dg (perforant path)
        pathways.append(RegionPathway(
            from_region="ec", to_region="dg",
            density=0.40, weight_mean=6.0, weight_jitter=0.2,
            plastic=True, plasticity_gate="ec_to_dg",
        ))
        # ec -> dg_fs (FFi recruitment, static)
        pathways.append(RegionPathway(
            from_region="ec", to_region="dg_fs",
            density=0.40, weight_mean=5.0, weight_jitter=0.2,
            plastic=False,
        ))
        # dg_fs -> dg (strong feedforward inhibition; static)
        pathways.append(RegionPathway(
            from_region="dg_fs", to_region="dg",
            density=1.00, weight_mean=6.0, weight_jitter=0.2,
            plastic=False,
        ))
        # ec -> ca1 (direct cortical bypass)
        pathways.append(RegionPathway(
            from_region="ec", to_region="ca1",
            density=0.30, weight_mean=3.0, weight_jitter=0.2,
            plastic=True, plasticity_gate="ec_to_ca1",
        ))
        # dg -> ca3 (mossy fibers)
        pathways.append(RegionPathway(
            from_region="dg", to_region="ca3",
            density=0.10, weight_mean=8.0, weight_jitter=0.2,
            plastic=True, plasticity_gate="dg_to_ca3",
        ))
        # ca3 -> ca3 recurrent: handled via ca3 region.internal_density=0.30 above.
        # ca3 -> ca1 (Schaffer collaterals)
        pathways.append(RegionPathway(
            from_region="ca3", to_region="ca1",
            density=0.30, weight_mean=4.0, weight_jitter=0.2,
            plastic=True, plasticity_gate="ca3_to_ca1",
        ))
        # ca1 -> place_cells: only when --hippocampus is on (place_cells region
        # only exists in that case). Coexists with landmark_sensors->place_cells.
        if enable_hippocampus:
            pathways.append(RegionPathway(
                from_region="ca1", to_region="place_cells",
                density=0.50, weight_mean=5.0, weight_jitter=0.2,
                plastic=False,
            ))

    return regions, pathways


def _warn_motor_lateral_inhibition_deprecated(value: bool) -> bool:
    """Emit a one-time DeprecationWarning if --motor-lateral-inhibition was
    used. The flag is NEGATIVE on cheat-5 (2026-04-26 evaluation) and the
    biology is wrong (real motor-pool WTA = spinal Renshaw, not cortical-FS).
    Slated for removal in a future cleanup."""
    if value:
        import warnings
        warnings.warn(
            "--motor-lateral-inhibition is DEPRECATED (NEGATIVE on cheat-5 "
            "evaluation; biology is wrong — real motor-pool WTA is spinal "
            "Renshaw inhibition per Kandel ch 35, not cortical-FS-like "
            "inhibition). Slated for removal in a future cleanup. If you "
            "need motor-WTA dynamics, plan to use spinal Renshaw modeling "
            "instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    return value


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


# Plasticity gates we expect to find on the runner's pathways. Pretraining
# thaws all of these; absence means a runner-side typo in plasticity_gate=
# (or a flag that doesn't add the pathway). Error early before GPU work.
_PRETRAINING_THAWED_GATES = (
    "corticostriatal",
    "sensory_to_cortex",
    "hippo_to_cortex",
    "beacon_to_goal",
    "landmark_to_place",
    "dlpfc_wm_pathways",
    "bg_cross_projections",
)


def _sample_pretraining_goal(rng, grid_size, start_pos, prev_goal):
    """Uniform random (gx, gy) on the grid with Manhattan >= 3 from start_pos
    and != prev_goal. Re-samples on rejection. The grid is small enough
    (8x8 → 16 valid cells given start (1,1)) that rejection sampling is
    trivially fast."""
    sx, sy = start_pos
    while True:
        gx = rng.randrange(grid_size)
        gy = rng.randrange(grid_size)
        if abs(gx - sx) + abs(gy - sy) < 3:
            continue
        if prev_goal is not None and (gx, gy) == prev_goal:
            continue
        return (gx, gy)


def _run_pretraining_phase(
    bridge,
    cfg,
    regions,
    n_goals: int,
    steps_per_goal: int,
    grid_size: int,
    start_pos,
    seed: int,
    enable_bg_cross_projections: bool = True,
    verbose: bool = True,
) -> dict:
    """Critical-period analog. Thaws ALL declared plasticity gates and runs
    the agent through n_goals random goals for steps_per_goal trials each.

    Returns a summary dict: {n_trials, n_goal_changes, cross_weights_mean,
    cross_weights_std}. See docs/plans/2026-04-28-cheat5-v4-design.md."""
    available = set(bridge.list_plasticity_gates())
    missing = [g for g in _PRETRAINING_THAWED_GATES
               if g not in available
               and _gate_required(g, regions,
                                  enable_bg_cross_projections=enable_bg_cross_projections)]
    if missing:
        raise KeyError(
            f"_run_pretraining_phase: gate(s) not declared on any pathway: "
            f"{missing!r}. Available: {sorted(available)!r}. "
            f"Either spell-check the gate name in build_bg_brain_regions, "
            f"or enable the flag that adds the pathway."
        )

    # Thaw every gate that IS declared. Gates not declared (e.g. learned
    # perception is off, so sensory_to_cortex doesn't exist) are silently
    # skipped — the corresponding pathway just isn't there.
    for gate in _PRETRAINING_THAWED_GATES:
        if gate in available:
            bridge.set_plasticity_gate(gate, 1.0)

    if verbose:
        print(f"[g11 seed={seed}] pretraining: all {len(available)} declared gates "
              f"thawed to 1.0; running {n_goals} goals × {steps_per_goal} steps each",
              flush=True)

    # Capture cross-projection synapse indices once (constant after build) so
    # we can compute weight stats at the end of pretraining. Empty if the
    # gate isn't declared (e.g. --bg-cross-projections off).
    cross_indices_cpu = []
    if "bg_cross_projections" in getattr(bridge, "_plasticity_gate_to_synapses", {}):
        cross_indices_cpu = list(bridge._plasticity_gate_to_synapses["bg_cross_projections"])

    # Early-out: zero goals → nothing to drive. Useful for tests that only
    # exercise the gate-thaw / signature path. Fall through to the summary.
    if n_goals == 0 or steps_per_goal == 0:
        return {
            "n_trials": 0,
            "n_goal_changes": 0,
            "cross_weights_mean": float("nan"),
            "cross_weights_std": float("nan"),
        }

    # Imports kept inside the helper to match the file's existing style and
    # avoid touching the top-of-file import block (Task 5 constraint).
    import random
    import numpy as np
    import cupy as cp

    # Reconstruct GPU-index arrays for the regions we drive in the inner
    # loop. The eval loop pre-caches these in run_moving_goal_episode; we
    # rebuild here so the helper stays self-contained (no extra kwargs).
    region_indices_cp = {}
    for r in regions:
        idx = list(bridge.region_manager.indices(r.name))
        if idx:
            region_indices_cp[r.name] = cp.asarray(idx, dtype=cp.int64)
    motor_idx_per_action = {
        a: region_indices_cp[f"motor_{a}"] for a in ACTION_NAMES
    }

    # Stimulus / readout window (mirrors eval). The fused dynamics
    # accumulate in 0.5 ms ticks, so 100 ms = 200 sub-steps.
    STIMULUS_MS = 100.0
    READOUT_START_MS = 30.0
    READOUT_END_MS = 100.0
    n_stim_steps = int(STIMULUS_MS / cfg.dt_ms)
    readout_start = int(READOUT_START_MS / cfg.dt_ms)
    readout_end = int(READOUT_END_MS / cfg.dt_ms)
    reward_hold_steps = 10  # matches eval default

    # Action geometry (must mirror ACTION_DELTAS in run_moving_goal_episode)
    ACTION_DELTAS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # N, E, S, W

    # Lock baseline tonic drives once. The eval loop re-sets these every
    # trial as a defensive measure; for pretraining we accept the drift
    # tradeoff in exchange for simpler code and equivalent biology (basal
    # ganglia tonic drives are biologically slow-varying).
    bridge.cp_external_input_current[:] = 0.0
    for rn in [f"gpe_{a}" for a in ACTION_NAMES]:
        bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(150.0)
    for rn in [f"gpe_arky_{a}" for a in ACTION_NAMES]:
        if rn in region_indices_cp:
            bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(120.0)
    for rn in [f"gpi_{a}" for a in ACTION_NAMES]:
        bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(110.0)
    for rn in ["stn", "snc"]:
        bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(150.0)
    for rn in [f"thal_{a}" for a in ACTION_NAMES]:
        bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(300.0)

    rng = random.Random(seed * 7919)  # deterministic, distinct from eval RNGs
    # Action-selection RNG must NOT collide with the eval loop's per-step
    # RNG seeds (which use seed*10000 + step). Use a different prime offset.
    action_rng = np.random.default_rng(seed * 13_417)

    prev_goal = None
    n_goal_changes = 0
    trial_counter = 0
    x, y = start_pos

    HEURISTIC_DRIVE_PA = cp.float32(800.0)

    for goal_idx in range(n_goals):
        gx, gy = _sample_pretraining_goal(rng, grid_size, start_pos, prev_goal)
        prev_goal = (gx, gy)
        n_goal_changes += 1
        if verbose:
            print(f"[g11 seed={seed}] pretraining goal {goal_idx + 1}/{n_goals}: "
                  f"({gx},{gy})", flush=True)

        # Reset agent to start at each new pretraining-goal episode
        x, y = start_pos

        for trial in range(steps_per_goal):
            # ── Heuristic cortex drive: directly drive cortex_X for each
            # goal-relative direction. Pretraining always uses the
            # heuristic — no opt-in perception modes here. The point is to
            # evolve weights under varied goals using the simplest possible
            # input pathway.
            #
            # Zero cortex pools first so the prior trial's drive doesn't
            # leak across direction transitions.
            for a in ACTION_NAMES:
                bridge.cp_external_input_current[region_indices_cp[f"cortex_{a}"]] = cp.float32(0.0)
            if gy > y:
                bridge.cp_external_input_current[region_indices_cp["cortex_N"]] = HEURISTIC_DRIVE_PA
            if gx > x:
                bridge.cp_external_input_current[region_indices_cp["cortex_E"]] = HEURISTIC_DRIVE_PA
            if gy < y:
                bridge.cp_external_input_current[region_indices_cp["cortex_S"]] = HEURISTIC_DRIVE_PA
            if gx < x:
                bridge.cp_external_input_current[region_indices_cp["cortex_W"]] = HEURISTIC_DRIVE_PA

            # ── Run stimulus window and tally motor spikes
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

            # ── Argmax action selection (random if all silent)
            if max(motor_counts.values()) > 0:
                action_idx = max(range(N_ACTIONS),
                                 key=lambda i: motor_counts[ACTION_NAMES[i]])
            else:
                action_idx = int(action_rng.integers(0, N_ACTIONS))

            # ── Position update + reward (Manhattan-delta only; sensed
            # reward is an eval-time refinement and adds no value during
            # pretraining where we just want weight evolution)
            dist_before = abs(x - gx) + abs(y - gy)
            dxa, dya = ACTION_DELTAS[action_idx]
            new_x = int(np.clip(x + dxa, 0, grid_size - 1))
            new_y = int(np.clip(y + dya, 0, grid_size - 1))
            x, y = new_x, new_y
            dist_after = abs(x - gx) + abs(y - gy)

            if dist_after < dist_before:
                reward = 1.0
            elif dist_after > dist_before:
                reward = -1.0
            else:
                reward = 0.0

            # ── Reward signal hold: drive plasticity for reward_hold_steps
            # extra sim ticks. This is the actual learning step — STDP
            # eligibility built up during the stimulus window gets
            # converted to weight updates here.
            if abs(reward) > 0:
                bridge.core_config.current_reward_signal = float(reward)
                for _ in range(reward_hold_steps):
                    bridge._run_one_simulation_step()
                    bridge.runtime_state.current_time_step += 1
                    bridge.runtime_state.current_time_ms = (
                        bridge.runtime_state.current_time_step * cfg.dt_ms
                    )
                bridge.core_config.current_reward_signal = 0.0

            # Structural pruning (cheat-5 option-1, 2026-04-28). Only fires
            # during pretraining when enable_structural_pruning is on. Restricted
            # to cross-projection synapses so we don't sparsify the same-action
            # corticostriatal routing. cp_eligibility_trace is allocated at capacity
            # (which can exceed nnz to leave room for structural plasticity), so
            # we slice it down to nnz before handing to update_pruning.
            if cfg.enable_structural_pruning and bridge.cp_synapse_alive is not None:
                cross_idx_list = bridge._plasticity_gate_to_synapses.get("bg_cross_projections")
                if cross_idx_list:
                    nnz = int(bridge.cp_connections.nnz)
                    bridge.update_pruning(
                        eligibility_trace=bridge.cp_eligibility_trace[:nnz],
                        reward_signal=reward,
                        prunable_indices=cp.asarray(list(cross_idx_list), dtype=cp.int64),
                    )

            trial_counter += 1

    # ── Cross-projection weight summary
    if cross_indices_cpu:
        cross_w = bridge.cp_connections.data[cp.asarray(cross_indices_cpu)].get()
        if np.isnan(cross_w).any():
            raise RuntimeError(
                "pretraining produced NaN cross-projection weights — likely STDP "
                "instability. Lower learning rate or shorten "
                "pretraining_steps_per_goal."
            )
        cross_mean = float(cross_w.mean())
        cross_std = float(cross_w.std())
    else:
        cross_mean = float("nan")
        cross_std = float("nan")

    if verbose:
        print(f"[g11 seed={seed}] pretraining complete: {trial_counter} trials, "
              f"{n_goal_changes} goal changes; cross weights mean={cross_mean:.3f} "
              f"std={cross_std:.3f} -> handing off to eval (curriculum will freeze "
              f"bg_cross_projections)", flush=True)

    return {
        "n_trials": trial_counter,
        "n_goal_changes": n_goal_changes,
        "cross_weights_mean": cross_mean,
        "cross_weights_std": cross_std,
    }


def _gate_required(name: str, regions, enable_bg_cross_projections: bool = True) -> bool:
    """Return True iff the gate must exist regardless of which flags are on.

    `regions` is accepted for forward-compatibility — Task 3 will inspect it
    to derive the full required-set from the active flag combination.
    Currently unused; we hard-code the gates known to always exist or whose
    presence is gated by a known flag.

    `enable_bg_cross_projections` softens the bg_cross_projections requirement:
    when False, that gate is not expected (--bg-cross-projections is off, so
    the pathway isn't built). Pretraining still runs but won't shape any
    cross-projection weights — Task 7 emits a warning at that path.
    """
    if name == "corticostriatal":
        return True
    if name == "bg_cross_projections":
        return enable_bg_cross_projections
    return False


def run_moving_goal_episode(
    out_path: str,
    seed: int = 42,
    n_steps: int = 1800,
    grid_size: int = 8,
    start_pos=(1, 1),
    goal_pos=(6, 6),
    goal_schedule=None,
    n_hippocampus_per_layer: int = 64,  # default 8×8 grid; should be roughly grid_size²
    sensory_to_cortex_weight: float = 10.0,
    hippocampus_to_cortex_weight: float = 10.0,
    enable_pfc: bool = False,
    n_pfc: int = 60,
    pfc_internal_density: float = 0.2,
    goal_to_pfc_weight: float = 8.0,
    pfc_to_cortex_weight: float = 8.0,
    enable_bg_cross_projections: bool = False,
    cross_projection_weight: float = 5.0,
    cross_projection_density: float = 1.0,
    cross_projection_topology_seed: int = 0,
    # v3 (2026-04-28) — see build_bg_brain_regions docstring.
    enable_bg_lateral_inhibition: bool = False,
    lateral_inhibition_density: float = 0.3,
    lateral_inhibition_weight: float = 2.0,
    # Interactive runtime control (2026-04-28). When set to a writable JSON
    # file path, the runner polls the file at the start of each trial and
    # applies the contents:
    #   { "paused": bool, "goal": [gx, gy] | null, "inject_reward": float | null }
    # - paused: blocks the trial loop until cleared
    # - goal: overrides the scheduled goal (persistent until set again)
    # - inject_reward: one-shot additive reward applied this trial; runner
    #   clears it back to null after consuming
    # Used by the webapp's World-tab live mode for click-to-teleport-goal,
    # pause/resume, and reward-injection. Default None = no polling, no
    # behavior change (ie. fully backwards compatible).
    interactive_control_file: str = None,
    # Progress print frequency (steps). Default 100 keeps validation runs
    # quiet; webapp interactive runs override to 1 so the dashboard's live
    # mode can animate per-step instead of jumping every 100 steps.
    progress_print_interval: int = 100,
    # Optional throttle (ms) between trials. Lets a human watch the agent
    # learn in real time without GPU saturation outpacing the eye. Default
    # 0 = full speed.
    trial_sleep_ms: float = 0.0,
    enable_beacon_perception: bool = False,
    n_beacon_sensors: int = 8,
    beacon_to_goal_weight: float = 8.0,
    beacon_max_intensity: float = 600.0,  # peak sensor drive (pA) when on top of beacon
    beacon_falloff: float = 1.0,           # intensity = peak / (1 + falloff*distance)
    beacon_replaces_goal: bool = False,    # if True, beacon→goal_cells is the ONLY goal info (true Stage 1 test)
    # Landmark perception (Item 1 Stage 2, 2026-04-27).
    enable_landmarks: bool = False,
    n_landmark_sensors: int = 8,
    landmark_to_place_weight: float = 8.0,
    landmark_position: tuple = None,  # default to grid center
    landmark_max_intensity: float = 600.0,
    landmark_falloff: float = 1.0,
    landmarks_replace_place: bool = False,
    # Cheat #4: sensed reward (2026-04-27).
    # Default reward = +1 if Manhattan distance decreased, -1 if increased.
    # This computes from raw (gx, gy, x, y) coordinates — a cheat. Sensed
    # reward instead computes reward from beacon-intensity GRADIENT
    # (intensity_after - intensity_before): the agent "feels warmer" as it
    # approaches and "cooler" as it retreats. Same information content as
    # distance-based, but operates on the agent's perceptual signal.
    enable_sensed_reward: bool = False,
    # Cue-following reflex (Item 1 Stage 3, 2026-04-27).
    # Replaces the heuristic with a hand-tuned innate reflex that computes
    # cortex drive from beacon sensor activations. Models a real animal's
    # "approach attractive cue" reflex (e.g., phototaxis). The reflex is
    # non-plastic — it represents innate sensorimotor wiring like vestibular
    # reflexes or looming detection. Plastic layers (sensory, hippo, beacon
    # → goal_cells) layer on top to refine the behavior.
    enable_cue_reflex: bool = False,
    cue_reflex_strength: float = 800.0,  # peak reflex drive matching heuristic
    cue_reflex_replaces_heuristic: bool = False,  # if True, heuristic disabled when reflex on
    learning_rate: float = 0.01,
    reward_eligibility_tau_ms: float = 500.0,
    reward_hold_steps: int = 10,
    verbose: bool = True,
    enable_motor_lateral_inhibition: bool = False,
    enable_cortex_lateral_inhibition: bool = False,
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
    # Curriculum learning (Option B from plastic-input-layer arc):
    # In phase 1 (steps 0..curriculum_warmup_steps), suppress hippocampus drive
    # so the heuristic+WTA builds up cortex→D1 selectivity in isolation. Then
    # in phase 2, enable hippo drive — hippo plastic weights learn given that
    # cortex→D1 is already mature.
    #
    # Stage 3 (2026-04-27): real curriculum uses bridge plasticity_gate
    # infrastructure — cortex→D1 frozen at warmup, hippo→cortex thawed.
    # Stage 5 (2026-04-27): ramp_steps>0 enables smooth critical-period
    # closure: gate values interpolate linearly from phase-1 to phase-2
    # values over `ramp_steps` centered on warmup_steps. Biologically
    # grounded: real critical periods close gradually via PV interneuron
    # maturation (~weeks), not as instantaneous step functions. Smoother
    # transition reduces variance from abrupt cascade disruption.
    enable_curriculum: bool = False,
    curriculum_warmup_steps: int = 600,  # phase 1 length: cortex→D1 builds without hippo noise
    curriculum_ramp_steps: int = 0,      # 0 = abrupt step; >0 = smooth ramp window
    # Stage 5 (2026-04-27): partial freeze allows cortex→D1 to keep
    # adapting at reduced rate during phase 2. 0.0 = full freeze (default,
    # cortex locked); 1.0 = no freeze (combo A). Intermediate values let
    # cortex slowly track changing reward landscape while hippo learns
    # primary input mapping. Biologically: cortical plasticity doesn't
    # halt absolutely with maturation — it slows but persists, especially
    # under top-down attention or unexpected reward (DA-modulated).
    curriculum_phase2_cortex_gain: float = 0.0,
    curriculum_phase2_hippo_gain: float = 1.0,
    # Cheat #5 closure (2026-04-28): cross-projections (cortex_X → str_D1_Y / str_D2_Y
    # for X != Y) are tagged with a separate plasticity gate "bg_cross_projections"
    # so the curriculum can stage them later than same-action pathways. The
    # naive approach (cross-projections on same gate as same-action) failed
    # 2026-04-27 because phase-0 motor activations reinforced cross-projections
    # to all D1 pools, locking in N/E motor bias before goal change.
    # Phase 3 thaws cross-projections AFTER goal change, when the agent has
    # experienced both regimes and STDP+reward can shape cross-action routing
    # symmetrically. -1 = stay frozen forever (default for safety).
    bg_cross_thaw_step: int = -1,
    # Plasticity gain for bg_cross_projections in phase 3. 1.0 = full plastic,
    # 0.5 = half-rate (slower than same-action), 0.0 = stay frozen.
    bg_cross_phase3_gain: float = 0.5,
    # ─── v4 (2026-04-28): developmental pretraining ────────────────────
    # Run a critical-period analog before the standard eval: N random
    # goals × M trials per goal with all plasticity gates open. At the
    # transition, the existing curriculum init naturally freezes
    # bg_cross_projections (line 1220 of this file). See
    # docs/plans/2026-04-28-cheat5-v4-design.md.
    enable_developmental_pretraining: bool = False,
    pretraining_n_goals: int = 10,
    pretraining_steps_per_goal: int = 3000,
    enable_structural_pruning: bool = False,
    # Cluster B.1 (2026-04-28): D1/D2 plasticity asymmetry — D2-targeting
    # synapses' weight updates flip sign vs D1. Default off.
    enable_d1_d2_asymmetry: bool = False,
    # Cluster B.2 (2026-04-28): striatal fast-spiking interneurons —
    # 4 str_PV_FSI_X pools providing broadcast inhibition to all D1/D2 MSN
    # pools. Default off. See
    # docs/plans/2026-04-28-cluster-b2-striatal-fsis-implementation.md.
    enable_striatal_fsis: bool = False,
    # Cluster B.3 (2026-04-28): cholinergic interneurons (TANs). Adds an
    # acetylcholine neuromodulator with the `pause_on_reward` rule that
    # transiently drops corticostriatal plasticity_window_gate on salient
    # reward events. Default off. See
    # docs/plans/2026-04-28-cluster-b3-tans-implementation.md.
    enable_tans: bool = False,
    enable_bg_neuropeptides: bool = False,  # R3.6: D1/D2 neuropeptide arms
    enable_cluster_a_closed_loop: bool = False,  # Cluster A: hyperdirect + thal->cortex
    enable_tonic_da: bool = False,  # Cluster C v1: dopamine as a real neuromodulator
    enable_compartmentalized_da: bool = False,  # Cluster C v2: per-action DA channels
    enable_cluster_d_hippocampus: bool = False,  # Cluster D v1: trisynaptic loop (ec+dg+ca3+ca1)
    enable_cluster_e_topography: bool = False,  # Cluster E v1: 2D coords + Gaussian-weighted cortex->striatum
    cluster_e_distance_sigma: float = 0.3,
    # Structural-pruning hyperparameters (cheat-5 option-1, 2026-04-28).
    # Defaults match CoreSimConfig but can be overridden from the runner's
    # CLI / kwargs to tune the pruning aggressiveness for short pretraining
    # windows (e.g. smoke tests). None preserves the cfg default.
    pruning_alpha: float = None,
    pruning_threshold: float = None,
    pruning_weight_floor: float = None,
    # Heuristic decay (Stage 6, 2026-04-27): scales the heuristic cortex
    # drive (800 pA per aligned pool) by this factor. Default 1.0 keeps
    # full heuristic. Set to 0.0 to disable heuristic entirely (tests
    # whether learned hippo weights alone can navigate). Useful for
    # validating that hippo actually learned something vs. just being
    # along for the ride.
    heuristic_strength: float = 1.0,
    # Step at which heuristic_strength changes from heuristic_strength to
    # post_curriculum_heuristic_strength. -1 = no change (default).
    heuristic_decay_after_step: int = -1,
    post_curriculum_heuristic_strength: float = 0.0,
    # Sleep-replay memory consolidation (Stage 7, 2026-04-27).
    # During sleep phases: no external goal, hippo cells fire in random
    # replay patterns (modeling NREM sharp-wave ripples), corticostriatal
    # is thawed (consolidation), hippo_to_cortex is frozen (preserve
    # learned weights). The replayed hippo signal drives cortex pools
    # via the learned hippo→cortex weights, and STDP between cortex_X
    # and D1_X consolidates the pattern into the cortex→D1 cascade.
    # After sleep, the cortex→D1 weights should encode hippo's learned
    # mapping, enabling navigation with reduced hippo dependency.
    # Biologically: episodic→semantic memory consolidation during NREM.
    # -1 = no sleep replay (default).
    sleep_replay_after_step: int = -1,
    sleep_replay_steps: int = 300,
    sleep_replay_rate_hz: float = 200.0,  # high rate (sharp-wave ripples)
    # NREM/REM stages (Item 7, 2026-04-27). When sleep_nrem_rem_alternate=True,
    # the sleep period alternates between NREM (trajectory replay, slow ripples)
    # and REM (random replay, faster). NREM cycle dominates first half, REM
    # second half, modeling sleep-stage progression.
    sleep_nrem_rem_alternate: bool = False,
    # PFC Stage 2: delayed-response test. Silence goal_cells during a delay
    # window to test whether PFC maintains goal info via persistent activity.
    # If PFC works as working memory, agent should still navigate toward goal
    # during the silence period (PFC remembers). Without PFC, agent should
    # drift (no goal info available).
    goal_silence_after_step: int = -1,
    goal_silence_duration: int = 0,
):
    """Phase B acid test: run BG circuit on G9-style moving-goal scenario.

    If the BG architecture dissolves the silent-motor trap (which V1-V7
    runner-side interventions all failed to do), phase 1 finalQ should
    drop substantially below the G9 baseline of 6.74.
    """
    # v4 (2026-04-28): conflict check. v4 keeps cross-projections frozen
    # during eval; v3.1 thaws them at bg_cross_thaw_step. Both at once is
    # meaningless. Fail loud instead of silent priority resolution.
    if enable_developmental_pretraining and bg_cross_thaw_step >= 0:
        raise ValueError(
            "--developmental-pretraining (v4) is incompatible with "
            "--bg-cross-thaw-step (v3.1). v4 keeps cross-projections frozen "
            "throughout eval; v3.1 thaws them mid-eval. Use one or the other, "
            f"not both. Got bg_cross_thaw_step={bg_cross_thaw_step}."
        )
    if enable_developmental_pretraining and not enable_bg_cross_projections:
        print(
            "[g11 warning] --developmental-pretraining without "
            "--bg-cross-projections: pretraining will run but won't shape any "
            "bg_cross_projections gate (no cross pathways exist). Did you "
            "mean to also pass --bg-cross-projections?",
            flush=True,
        )
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
        enable_cortex_lateral_inhibition=enable_cortex_lateral_inhibition,
        enable_learned_perception=enable_learned_perception,
        enable_hippocampus=enable_hippocampus,
        n_hippocampus_per_layer=n_hippocampus_per_layer,
        sensory_to_cortex_weight=sensory_to_cortex_weight,
        hippocampus_to_cortex_weight=hippocampus_to_cortex_weight,
        enable_pfc=enable_pfc,
        n_pfc=n_pfc,
        pfc_internal_density=pfc_internal_density,
        goal_to_pfc_weight=goal_to_pfc_weight,
        pfc_to_cortex_weight=pfc_to_cortex_weight,
        enable_bg_cross_projections=enable_bg_cross_projections,
        cross_projection_weight=cross_projection_weight,
        cross_projection_density=cross_projection_density,
        cross_projection_topology_seed=cross_projection_topology_seed,
        enable_bg_lateral_inhibition=enable_bg_lateral_inhibition,
        lateral_inhibition_density=lateral_inhibition_density,
        lateral_inhibition_weight=lateral_inhibition_weight,
        enable_striatal_fsis=enable_striatal_fsis,
        enable_cluster_a_closed_loop=enable_cluster_a_closed_loop,
        enable_cluster_d_hippocampus=enable_cluster_d_hippocampus,
        enable_cluster_e_topography=enable_cluster_e_topography,
        cluster_e_distance_sigma=cluster_e_distance_sigma,
        enable_beacon_perception=enable_beacon_perception,
        n_beacon_sensors=n_beacon_sensors,
        beacon_to_goal_weight=beacon_to_goal_weight,
        enable_landmarks=enable_landmarks,
        n_landmark_sensors=n_landmark_sensors,
        landmark_to_place_weight=landmark_to_place_weight,
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

    # Pre-compute hippocampal cell preferred (x, y) — covering full grid.
    # Layout: square grid of side = ceil(sqrt(n_hippocampus_per_layer)) with
    # cells spaced to span the full grid range. For 8×8 grid with 64 cells,
    # one cell per position. For 16×16 grid with 256 cells, also one per
    # position. For mismatched cases, cells space out uniformly.
    if enable_hippocampus:
        side = int(round(n_hippocampus_per_layer ** 0.5))
        scale = (grid_size - 1) / max(1, side - 1) if side > 1 else 1.0
        hippo_pref_x = np.array([(i % side) * scale for i in range(n_hippocampus_per_layer)], dtype=np.float32)
        hippo_pref_y = np.array([(i // side) * scale for i in range(n_hippocampus_per_layer)], dtype=np.float32)
    else:
        hippo_pref_x = None
        hippo_pref_y = None

    # Pre-compute beacon sensor preferred directions (Item 1 Stage 1).
    # Sensors evenly distributed in 2D — for n=8: N, NE, E, SE, S, SW, W, NW.
    # Each sensor responds maximally when beacon is in its preferred direction
    # (cosine alignment), with intensity falling off with distance.
    # Models biological directional cue detection (e.g., bilateral hearing
    # estimating sound source direction from intensity differences).
    if enable_beacon_perception:
        beacon_pref_x = np.zeros(n_beacon_sensors, dtype=np.float32)
        beacon_pref_y = np.zeros(n_beacon_sensors, dtype=np.float32)
        for i in range(n_beacon_sensors):
            angle = 2.0 * np.pi * i / n_beacon_sensors
            beacon_pref_x[i] = np.cos(angle)
            beacon_pref_y[i] = np.sin(angle)
    else:
        beacon_pref_x = None
        beacon_pref_y = None

    # Pre-compute landmark sensor preferred directions (Item 1 Stage 2).
    # Same structure as beacon sensors; landmark is at fixed position.
    if enable_landmarks:
        landmark_pref_x = np.zeros(n_landmark_sensors, dtype=np.float32)
        landmark_pref_y = np.zeros(n_landmark_sensors, dtype=np.float32)
        for i in range(n_landmark_sensors):
            angle = 2.0 * np.pi * i / n_landmark_sensors
            landmark_pref_x[i] = np.cos(angle)
            landmark_pref_y[i] = np.sin(angle)
        # Default landmark position: grid center
        if landmark_position is None:
            landmark_position = (grid_size / 2.0, grid_size / 2.0)
    else:
        landmark_pref_x = None
        landmark_pref_y = None

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
    # cortex->D1 weight_mean needs w_max above that or soft-bound STDP collapses it.
    # When R3.5's density reduction triggers weight scaling (e.g. weight=125 at
    # density=0.20), w_max must be ABOVE that — otherwise LTP events drive weights
    # negative, collapsing the cascade silently. Recompute the post-R3.5 weight
    # locally (mirrors build_bg_brain_regions logic).
    _ctx_msn_density = 0.20  # R3.5 default
    _ctx_msn_weight = (25.0 / _ctx_msn_density) if _ctx_msn_density < 1.0 else 25.0
    cfg.stdp_w_max = max(30.0, _ctx_msn_weight * 1.2)
    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_ou_process = False
    cfg.enable_conductance_noise = False
    cfg.enable_parameter_heterogeneity = False
    cfg.enable_structural_plasticity = False  # keep synapse count fixed (per-action DA mask depends on it)
    cfg.enable_structural_pruning = enable_structural_pruning
    cfg.enable_d1_d2_asymmetry = enable_d1_d2_asymmetry
    # Cluster B.3 (2026-04-28): cholinergic TANs. Turn the neuromod
    # subsystem ON cumulatively (no other flag in this runner enables it
    # today, but `|=` keeps it future-proof if one starts to) and append
    # the default acetylcholine config to whatever the cfg already has.
    if enable_tans:
        from sim.neuromodulators import _default_acetylcholine_tan_config
        cfg.enable_neuromodulator_subsystem = True
        cfg.neuromodulators = list(cfg.neuromodulators) + [
            _default_acetylcholine_tan_config()
        ]
    # R3.6 (2026-04-29): D1/D2 neuropeptide arms — dynorphin (D1, KOR
    # plasticity-rate brake), substance P (D1, NK-1 ACh boost), enkephalin
    # (D2, DOR plasticity-rate boost). All three opt-in together.
    if enable_bg_neuropeptides:
        from sim.neuromodulators import (
            _default_dynorphin_config,
            _default_substance_p_config,
            _default_enkephalin_config,
        )
        cfg.enable_neuromodulator_subsystem = True
        cfg.neuromodulators = list(cfg.neuromodulators) + [
            _default_dynorphin_config(),
            _default_substance_p_config(),
            _default_enkephalin_config(),
        ]
    # Cluster C v1 (2026-04-29): tonic dopamine via neuromodulator framework.
    # Replaces signed-scalar reward modulation with a real DA concentration
    # (tonic baseline + phasic activation/depression). Unlocks B.3 ACh
    # window-gating (which is otherwise a no-op without tonic DA-driven
    # plasticity to gate). Composes with --enable-tans and
    # --enable-bg-neuropeptides.
    #
    # Precedence: when both --enable-tonic-da and --enable-compartmentalized-da
    # are set, only the per-action channels are registered (the global
    # `dopamine` modulator would double-count with the per-synapse path).
    if enable_tonic_da and not enable_compartmentalized_da:
        from sim.neuromodulators import _default_dopamine_config
        cfg.enable_neuromodulator_subsystem = True
        cfg.neuromodulators = list(cfg.neuromodulators) + [
            _default_dopamine_config()
        ]

    # Cluster C v2 (2026-04-29): compartmentalized DA — per-action channels.
    # Registers 4 modulators (dopamine_N, dopamine_E, dopamine_S, dopamine_W),
    # each targeting only synapses with matching action_index via
    # scope='action:{idx}'. Production rule: from_action_specific_reward
    # gates concentration update by last_selected_action. Implies tonic-DA
    # at the per-action level (the single global dopamine modulator is NOT
    # registered when this flag is on).
    # See docs/plans/2026-04-29-cluster-c-v2-compartmentalized-da-design.md.
    if enable_compartmentalized_da:
        from sim.neuromodulators import _default_per_action_dopamine_config
        cfg.enable_neuromodulator_subsystem = True
        cfg.neuromodulators = list(cfg.neuromodulators) + [
            _default_per_action_dopamine_config(action, idx)
            for idx, action in enumerate(ACTION_NAMES)
        ]
        if verbose:
            print(f"[g11 seed={seed}] Cluster C v2 compartmentalized DA: "
                  f"4 modulators registered "
                  f"(dopamine_{{{','.join(ACTION_NAMES)}}})")
    if pruning_alpha is not None:
        cfg.pruning_alpha = float(pruning_alpha)
    if pruning_threshold is not None:
        cfg.pruning_threshold = float(pruning_threshold)
    if pruning_weight_floor is not None:
        cfg.pruning_weight_floor = float(pruning_weight_floor)

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
    for region_name in [f"gpe_arky_{a}" for a in ACTION_NAMES]:
        if region_name in region_indices_cp:
            bridge.cp_external_input_current[region_indices_cp[region_name]] = cp.float32(120.0)
    for region_name in [f"gpi_{a}" for a in ACTION_NAMES]:
        bridge.cp_external_input_current[region_indices_cp[region_name]] = cp.float32(110.0)
    for region_name in ["stn", "snc"]:
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

    # Sleep-replay trajectory log: stores (x, y, gx, gy) tuples from
    # waking trials where the agent successfully approached goal
    # (reward > 0). During sleep, these are replayed instead of random
    # patterns, modeling biological hippocampal replay of successful
    # episodic memories.
    # Bounded to recent ~200 entries so sleep replays mostly the
    # current-goal patterns, not stale patterns from earlier goals
    # (which can bias consolidation toward old goal directions).
    # Biologically: hippocampal trace decay ensures replay reflects
    # recent experience, not arbitrary old episodes.
    successful_trajectories: List = []
    SUCCESSFUL_TRAJ_MAX = 200

    # Curriculum: real plasticity gating (Stage 3, 2026-04-27).
    # The hippo→cortex pathways are tagged "hippo_to_cortex" and cortex→D1/D2
    # are tagged "corticostriatal" in build_bg_brain_regions. We use these gates
    # to implement true developmental staging:
    #   Phase 1 (warmup): cortex→D1 plastic, hippo→cortex frozen
    #     → cortex builds correct cortex→D1 mapping under heuristic alone
    #   Phase 2 (mature): cortex→D1 frozen, hippo→cortex plastic
    #     → hippo learns place→action given that cascade is locked-in
    # This addresses the architectural ceiling identified in the 6-NEGATIVE
    # plastic-input-layer arc: the cascade depends on a single clean cortex
    # input source. By staging plasticity, we let cortex selectivity
    # establish itself, then add the plastic input layer with the cascade
    # protected against further drift.
    #
    # Stage 5 ramping: when ramp_steps>0, transitions are smooth (linear
    # interpolation of gate values over ramp window centered on warmup).
    # This matches biology — critical periods close gradually via PV
    # maturation, not as step functions — and reduces variance from
    # abrupt cascade disruption.
    # Curriculum gates: corticostriatal, hippo_to_cortex, sensory_to_cortex,
    # beacon_to_goal. In phase 1, all input layers (hippo, sensory, beacon→goal)
    # are frozen and only corticostriatal is plastic. Cortex builds D1 mapping
    # under the heuristic teacher. In phase 2, corticostriatal freezes and the
    # input layers thaw, learning their mappings with cortex as the locked target.

    # v4 developmental pretraining (2026-04-28). Runs only if enabled.
    # Inserted BEFORE curriculum init so the init's phase-1 gate values
    # naturally freeze bg_cross_projections at eval start (line 1220).
    pretraining_summary = None
    if enable_developmental_pretraining:
        pretraining_summary = _run_pretraining_phase(
            bridge=bridge, cfg=cfg, regions=regions,
            n_goals=pretraining_n_goals,
            steps_per_goal=pretraining_steps_per_goal,
            grid_size=grid_size, start_pos=start_pos,
            seed=seed,
            enable_bg_cross_projections=enable_bg_cross_projections,
            verbose=verbose,
        )

    available_gates = bridge.list_plasticity_gates() if enable_curriculum else []
    has_hippo_gate = enable_curriculum and "hippo_to_cortex" in available_gates
    has_cortex_gate = enable_curriculum and "corticostriatal" in available_gates
    has_sensory_gate = enable_curriculum and "sensory_to_cortex" in available_gates
    has_beacon_gate = enable_curriculum and "beacon_to_goal" in available_gates
    has_landmark_gate = enable_curriculum and "landmark_to_place" in available_gates
    has_bg_cross_gate = enable_curriculum and "bg_cross_projections" in available_gates
    bg_cross_thawed = False  # tracks the phase-3 thaw event for verbose logging
    if enable_curriculum:
        # Phase 1: input plasticity OFF, corticostriatal plasticity ON,
        # bg_cross_projections OFF (stays off until phase 3 if configured)
        if has_hippo_gate:
            bridge.set_plasticity_gate("hippo_to_cortex", 0.0)
        if has_sensory_gate:
            bridge.set_plasticity_gate("sensory_to_cortex", 0.0)
        if has_beacon_gate:
            bridge.set_plasticity_gate("beacon_to_goal", 0.0)
        if has_landmark_gate:
            bridge.set_plasticity_gate("landmark_to_place", 0.0)
        if has_cortex_gate:
            bridge.set_plasticity_gate("corticostriatal", 1.0)
        if has_bg_cross_gate:
            bridge.set_plasticity_gate("bg_cross_projections", 0.0)
        if verbose:
            ramp_msg = (f", ramp={curriculum_ramp_steps}" if curriculum_ramp_steps > 0
                       else " (abrupt)")
            gates_msg = ", ".join(filter(None, [
                "hippo_to_cortex" if has_hippo_gate else None,
                "sensory_to_cortex" if has_sensory_gate else None,
            ]))
            print(f"[g11 seed={seed}] curriculum phase 1: corticostriatal plastic, "
                  f"input gates frozen [{gates_msg}]{ramp_msg}", flush=True)
    last_logged_phase = 1  # for verbose phase-2 announcement on first ramp tick

    def _curriculum_gate_values(step_idx):
        """Return (cortex_gate, hippo_gate) for the given step under the
        current curriculum schedule. Linear ramp centered on warmup boundary
        when ramp_steps > 0; abrupt step otherwise.

        Phase 1 values: cortex=1.0, hippo=0.0 (cortex plastic, hippo frozen)
        Phase 2 values: cortex=curriculum_phase2_cortex_gain (default 0.0),
                        hippo=curriculum_phase2_hippo_gain (default 1.0).
        Partial-freeze configs (e.g. cortex=0.3) let cortex slowly track
        changing reward landscape while hippo learns the primary input
        mapping — biologically: cortical plasticity slows but persists.
        """
        c_phase1, h_phase1 = 1.0, 0.0
        c_phase2 = curriculum_phase2_cortex_gain
        h_phase2 = curriculum_phase2_hippo_gain
        if curriculum_ramp_steps <= 0:
            # Abrupt: phase 1 until warmup, phase 2 after
            if step_idx < curriculum_warmup_steps:
                return c_phase1, h_phase1
            return c_phase2, h_phase2
        # Smooth: ramp over [warmup - half, warmup + half]
        half = curriculum_ramp_steps // 2
        ramp_start = curriculum_warmup_steps - half
        ramp_end = curriculum_warmup_steps + (curriculum_ramp_steps - half)
        if step_idx < ramp_start:
            return c_phase1, h_phase1
        if step_idx >= ramp_end:
            return c_phase2, h_phase2
        # In ramp window: linear interpolation between phase 1 and phase 2 values
        progress = (step_idx - ramp_start) / float(curriculum_ramp_steps)
        c_val = c_phase1 + (c_phase2 - c_phase1) * progress
        h_val = h_phase1 + (h_phase2 - h_phase1) * progress
        return c_val, h_val

    t0 = time.time()
    # Track current gating_strength (used for DA-gated WTA across the whole trial,
    # not just the reward-hold sub-step). Initialized to 1.0 (full WTA on first trial
    # before any reward feedback exists).
    current_gating_strength = 1.0
    for step in range(n_steps):
        # Curriculum gate update — for ramp mode, update every step during
        # the ramp window; for abrupt mode, only at the warmup boundary.
        # Sensory and hippo input layers share phase-2 gain (they're peer
        # input pathways being thawed together).
        if enable_curriculum and (has_cortex_gate or has_hippo_gate or has_sensory_gate):
            target_cortex, target_hippo = _curriculum_gate_values(step)
            target_sensory = target_hippo  # input layers transition together
            if curriculum_ramp_steps > 0:
                if has_cortex_gate:
                    bridge.set_plasticity_gate("corticostriatal", float(target_cortex))
                if has_hippo_gate:
                    bridge.set_plasticity_gate("hippo_to_cortex", float(target_hippo))
                if has_sensory_gate:
                    bridge.set_plasticity_gate("sensory_to_cortex", float(target_sensory))
                if has_beacon_gate:
                    bridge.set_plasticity_gate("beacon_to_goal", float(target_sensory))
                if has_landmark_gate:
                    bridge.set_plasticity_gate("landmark_to_place", float(target_sensory))
                if (last_logged_phase == 1 and target_hippo > 0.0):
                    last_logged_phase = 2
                    if verbose:
                        print(f"[g11 seed={seed}] step {step}: CURRICULUM RAMP "
                              f"BEGINNING (cortex {target_cortex:.2f}, inputs {target_hippo:.2f})",
                              flush=True)
            else:
                if last_logged_phase == 1 and step >= curriculum_warmup_steps:
                    last_logged_phase = 2
                    if has_cortex_gate:
                        bridge.set_plasticity_gate("corticostriatal", float(curriculum_phase2_cortex_gain))
                    if has_hippo_gate:
                        bridge.set_plasticity_gate("hippo_to_cortex", float(curriculum_phase2_hippo_gain))
                    if has_sensory_gate:
                        bridge.set_plasticity_gate("sensory_to_cortex", float(curriculum_phase2_hippo_gain))
                    if has_beacon_gate:
                        bridge.set_plasticity_gate("beacon_to_goal", float(curriculum_phase2_hippo_gain))
                    if has_landmark_gate:
                        bridge.set_plasticity_gate("landmark_to_place", float(curriculum_phase2_hippo_gain))
                    if verbose:
                        print(f"[g11 seed={seed}] step {step}: CURRICULUM PHASE 2 -- "
                              f"corticostriatal={curriculum_phase2_cortex_gain:.2f}, "
                              f"inputs={curriculum_phase2_hippo_gain:.2f}", flush=True)

        # Phase 3 (Cheat #5 closure, 2026-04-28): thaw bg_cross_projections.
        # Cross-projection cortex_X → str_D1_Y / str_D2_Y pathways stay frozen
        # through phases 1 and 2 (so they don't accumulate phase-0 motor bias),
        # then thaw at bg_cross_thaw_step. By this point the agent has typically
        # experienced both pre- and post-goal-change regimes (default thaw=1200
        # is ~300 steps after the default goal change at 900), so STDP+reward
        # can shape cross-action routing symmetrically rather than locking in
        # phase-0 winners.
        if (
            has_bg_cross_gate and not bg_cross_thawed
            and bg_cross_thaw_step >= 0 and step >= bg_cross_thaw_step
        ):
            bridge.set_plasticity_gate("bg_cross_projections", float(bg_cross_phase3_gain))
            bg_cross_thawed = True
            if verbose:
                print(f"[g11 seed={seed}] step {step}: CURRICULUM PHASE 3 -- "
                      f"bg_cross_projections gain={bg_cross_phase3_gain:.2f}",
                      flush=True)

        # Sleep-replay phase (Stage 7, 2026-04-27): biological memory consolidation.
        # During sleep, hippo cells fire in random replay patterns (sharp-wave ripples),
        # corticostriatal is thawed (consolidation), hippo_to_cortex is frozen.
        # Hippo's already-learned weights drive cortex via existing connections;
        # STDP between cortex and D1 then consolidates the pattern.
        in_sleep = (sleep_replay_after_step >= 0
                   and step >= sleep_replay_after_step
                   and step < sleep_replay_after_step + sleep_replay_steps)
        if in_sleep:
            # Set gates for consolidation: corticostriatal plastic, hippo_to_cortex frozen
            if has_cortex_gate:
                bridge.set_plasticity_gate("corticostriatal", 1.0)
            if has_hippo_gate:
                bridge.set_plasticity_gate("hippo_to_cortex", 0.0)
            if has_sensory_gate:
                bridge.set_plasticity_gate("sensory_to_cortex", 0.0)
            # Mark phase entry for verbose output
            if step == sleep_replay_after_step and verbose:
                print(f"[g11 seed={seed}] step {step}: ENTERING SLEEP REPLAY "
                      f"(corticostriatal=1, hippo/sensory frozen, replay rate={sleep_replay_rate_hz:.0f}Hz)",
                      flush=True)
        elif sleep_replay_after_step >= 0 and step == sleep_replay_after_step + sleep_replay_steps and verbose:
            print(f"[g11 seed={seed}] step {step}: EXITING SLEEP REPLAY",
                  flush=True)
            # Restore phase-2 gates
            if has_cortex_gate:
                bridge.set_plasticity_gate("corticostriatal", float(curriculum_phase2_cortex_gain))
            if has_hippo_gate:
                bridge.set_plasticity_gate("hippo_to_cortex", float(curriculum_phase2_hippo_gain))

        # Interactive runtime control (2026-04-28). Polls a JSON file every
        # trial for paused / goal / inject_reward overrides from the webapp.
        # See webapp/static/world.js for the click-to-control wiring.
        manual_reward_injection = 0.0
        if interactive_control_file:
            try:
                with open(interactive_control_file) as _cf:
                    _ctrl = json.load(_cf)
            except (FileNotFoundError, OSError, json.JSONDecodeError):
                _ctrl = {}
            # Pause loop — block while paused, re-reading the file periodically
            while _ctrl.get("paused"):
                time.sleep(0.1)
                try:
                    with open(interactive_control_file) as _cf:
                        _ctrl = json.load(_cf)
                except (FileNotFoundError, OSError, json.JSONDecodeError):
                    break
            # Goal override (persistent until set again)
            _new_goal = _ctrl.get("goal")
            if _new_goal is not None and len(_new_goal) == 2:
                _ng = (int(_new_goal[0]), int(_new_goal[1]))
                if (gx, gy) != _ng:
                    gx, gy = _ng
                    goal_change_steps.append(step)
                    if verbose:
                        print(f"[g11 seed={seed}] step {step}: INTERACTIVE GOAL "
                              f"-> ({gx}, {gy})", flush=True)
            # One-shot reward injection (consumed by clearing the field)
            _inj = _ctrl.get("inject_reward")
            if _inj is not None:
                manual_reward_injection = float(_inj)
                _ctrl["inject_reward"] = None
                try:
                    with open(interactive_control_file, "w") as _cf:
                        json.dump(_ctrl, _cf)
                except OSError:
                    pass

        # Goal change (scheduled)
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
        for rn in ["stn", "snc"]:
            bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(150.0)
        for rn in [f"thal_{a}" for a in ACTION_NAMES]:
            bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(300.0)
        # Cortex drives — both heuristic AND learned perception can be active
        # simultaneously (additive). The heuristic represents innate
        # sensorimotor primitives; the sensory layer learns refined
        # position→action mappings on top. With curriculum, the sensory
        # layer learns via STDP+reward using the heuristic as teacher.
        # Heuristic cortex drive: directly drive cortex_X for each goal-relative direction.
        # Heuristic strength can decay post-curriculum to test pure-learned navigation.
        # During sleep replay: heuristic disabled so consolidation runs purely
        # on hippo-driven cortex activity.
        # During goal_silence (PFC Stage 2): also silence heuristic to test
        # whether PFC + already-learned input layers maintain navigation.
        in_goal_silence_step = (goal_silence_after_step >= 0
                                and step >= goal_silence_after_step
                                and step < goal_silence_after_step + goal_silence_duration)
        if in_sleep or in_goal_silence_step:
            h_strength = 0.0
        elif heuristic_decay_after_step >= 0 and step >= heuristic_decay_after_step:
            h_strength = post_curriculum_heuristic_strength
        elif enable_cue_reflex and cue_reflex_replaces_heuristic:
            # Stage 3: reflex replaces heuristic. The reflex below computes
            # cortex drive from beacon sensor activations instead of (gx,gy).
            h_strength = 0.0
        else:
            h_strength = heuristic_strength
        h_drive = cp.float32(800.0 * h_strength)
        if h_strength > 0:
            if gy > y:
                bridge.cp_external_input_current[region_indices_cp["cortex_N"]] = h_drive
            if gx > x:
                bridge.cp_external_input_current[region_indices_cp["cortex_E"]] = h_drive
            if gy < y:
                bridge.cp_external_input_current[region_indices_cp["cortex_S"]] = h_drive
            if gx < x:
                bridge.cp_external_input_current[region_indices_cp["cortex_W"]] = h_drive

        # Cue-following reflex (Item 1 Stage 3, 2026-04-27).
        # Innate reflex: computes cortex drive from beacon sensor activations
        # instead of from raw (gx, gy) coordinates. Each cortex pool gets
        # drive proportional to the integrated beacon strength in its
        # preferred cardinal direction. Models "approach attractive cue"
        # reflex like phototaxis. Non-plastic (innate sensorimotor wiring).
        # Direction-normalized: reflex strength is independent of beacon
        # distance (real biological reflexes operate on direction once
        # stimulus is detected, not on absolute intensity).
        if enable_cue_reflex and enable_beacon_perception and not (in_sleep or in_goal_silence_step):
            bdx = float(gx - x); bdy = float(gy - y)
            distance = (bdx * bdx + bdy * bdy) ** 0.5
            if distance > 1e-6:
                bearing_x = bdx / distance
                bearing_y = bdy / distance
                # Direction-only sensor pattern: cosine alignment, half-rectified
                sensor_dir = np.maximum(0.0, beacon_pref_x * bearing_x + beacon_pref_y * bearing_y)
                # Normalize so total activation sums to 1 (direction representation)
                total = sensor_dir.sum() + 1e-6
                sensor_norm = sensor_dir / total
                # Each cortex pool integrates sensors aligned with its cardinal direction
                drive_N = float(np.sum(sensor_norm * np.maximum(0, beacon_pref_y)))
                drive_E = float(np.sum(sensor_norm * np.maximum(0, beacon_pref_x)))
                drive_S = float(np.sum(sensor_norm * np.maximum(0, -beacon_pref_y)))
                drive_W = float(np.sum(sensor_norm * np.maximum(0, -beacon_pref_x)))
                # Scale to match heuristic strength regardless of distance
                # (the reflex is "go this direction at full strength" once
                # the cue direction is detected, like phototaxis)
                if drive_N > 1e-3:
                    bridge.cp_external_input_current[region_indices_cp["cortex_N"]] = cp.float32(drive_N * cue_reflex_strength)
                if drive_E > 1e-3:
                    bridge.cp_external_input_current[region_indices_cp["cortex_E"]] = cp.float32(drive_E * cue_reflex_strength)
                if drive_S > 1e-3:
                    bridge.cp_external_input_current[region_indices_cp["cortex_S"]] = cp.float32(drive_S * cue_reflex_strength)
                if drive_W > 1e-3:
                    bridge.cp_external_input_current[region_indices_cp["cortex_W"]] = cp.float32(drive_W * cue_reflex_strength)
        # Sensory layer drive (opt-in, additive on top of heuristic).
        # Each sensory neuron i has preferred (dx_i, dy_i); rate = max * exp(-d²/2σ²)
        # The sensory→cortex pathway is plastic — agent learns mapping via STDP+reward.
        if enable_learned_perception:
            dx = float(gx - x)
            dy = float(gy - y)
            dx_clip = max(-3.0, min(3.0, dx))
            dy_clip = max(-3.0, min(3.0, dy))
            d_sq = (sensory_pref_dx - dx_clip) ** 2 + (sensory_pref_dy - dy_clip) ** 2
            sensory_drive = sensory_drive_max_pA * np.exp(-d_sq / (2.0 * sensory_drive_sigma ** 2))
            bridge.cp_external_input_current[region_indices_cp["sensory"]] = cp.asarray(sensory_drive, dtype=cp.float32)

        # Hippocampus drive (ADDITIVE on top of heuristic — provides plastic memory).
        # Real biology: hippocampus augments cortex, doesn't replace it. Place + goal
        # cells learn (place, goal) → action associations via STDP+reward, providing
        # additional cortex drive that should reinforce the correct action over training.
        # Curriculum gate: during the warmup phase, suppress hippo drive so the
        # heuristic (+WTA if enabled) builds up cortex→D1 selectivity in isolation.
        # After the warmup, hippo drive turns on and learns via STDP+reward.
        # SLEEP REPLAY: drive place + goal cells to simulate sharp-wave
        # ripples. The replayed pattern, via existing learned hippo→cortex
        # weights, drives cortex pools, which then strengthens cortex→D1
        # weights via STDP (corticostriatal thawed).
        # Trajectory replay (preferred): sample from successful_trajectories
        # log (built during wake from positive-reward steps). Models
        # biological replay of episodic memories. Falls back to random
        # patterns if no trajectories logged yet.
        # NREM/REM (Item 7): if sleep_nrem_rem_alternate, first half of sleep
        # is NREM-style (trajectory replay, biological consolidation), second
        # half is REM-style (random patterns, less structured).
        if in_sleep and enable_hippocampus:
            sleep_progress = (step - sleep_replay_after_step) / max(1, sleep_replay_steps)
            in_rem_phase = sleep_nrem_rem_alternate and sleep_progress >= 0.5
            if successful_trajectories and not in_rem_phase:
                # NREM: trajectory replay from logged successful steps
                idx = int(np.random.randint(0, len(successful_trajectories)))
                replay_x, replay_y, replay_gx, replay_gy = successful_trajectories[idx]
                replay_x = float(replay_x); replay_y = float(replay_y)
                replay_gx = float(replay_gx); replay_gy = float(replay_gy)
            else:
                # REM (or fallback): random patterns, less structured
                replay_x = float(np.random.randint(0, grid_size))
                replay_y = float(np.random.randint(0, grid_size))
                replay_gx = float(np.random.randint(0, grid_size))
                replay_gy = float(np.random.randint(0, grid_size))
            place_dsq = (hippo_pref_x - replay_x) ** 2 + (hippo_pref_y - replay_y) ** 2
            place_drive = hippocampus_drive_max_pA * np.exp(-place_dsq / (2.0 * hippocampus_drive_sigma ** 2))
            bridge.cp_external_input_current[region_indices_cp["place_cells"]] = cp.asarray(place_drive, dtype=cp.float32)
            goal_dsq = (hippo_pref_x - replay_gx) ** 2 + (hippo_pref_y - replay_gy) ** 2
            goal_drive = hippocampus_drive_max_pA * np.exp(-goal_dsq / (2.0 * hippocampus_drive_sigma ** 2))
            bridge.cp_external_input_current[region_indices_cp["goal_cells"]] = cp.asarray(goal_drive, dtype=cp.float32)
            hippo_active = False  # skip the normal-flow hippo drive below
        else:
            hippo_active = enable_hippocampus and (
                not enable_curriculum or step >= curriculum_warmup_steps
            )
        if hippo_active:
            if enable_landmarks and landmarks_replace_place:
                # Stage 2: don't drive place_cells directly. They get input
                # only via the plastic landmark_sensors → place_cells pathway.
                pass
            else:
                place_dsq = (hippo_pref_x - float(x)) ** 2 + (hippo_pref_y - float(y)) ** 2
                place_drive = hippocampus_drive_max_pA * np.exp(-place_dsq / (2.0 * hippocampus_drive_sigma ** 2))
                bridge.cp_external_input_current[region_indices_cp["place_cells"]] = cp.asarray(place_drive, dtype=cp.float32)
            # Goal cells silencing test (PFC Stage 2): during the silence
            # window, goal_cells are forced to 0 — tests whether PFC working
            # memory holds the goal info during the delay.
            in_goal_silence = (goal_silence_after_step >= 0
                              and step >= goal_silence_after_step
                              and step < goal_silence_after_step + goal_silence_duration)
            if in_goal_silence:
                bridge.cp_external_input_current[region_indices_cp["goal_cells"]] = cp.float32(0.0)
            elif enable_beacon_perception and beacon_replaces_goal:
                # Replace mode: don't drive goal_cells directly. The
                # beacon → goal_cells pathway must learn to drive them
                # from sensor patterns.
                pass  # goal_cells gets only the plastic beacon→goal drive
            else:
                goal_dsq = (hippo_pref_x - float(gx)) ** 2 + (hippo_pref_y - float(gy)) ** 2
                goal_drive = hippocampus_drive_max_pA * np.exp(-goal_dsq / (2.0 * hippocampus_drive_sigma ** 2))
                bridge.cp_external_input_current[region_indices_cp["goal_cells"]] = cp.asarray(goal_drive, dtype=cp.float32)
        elif enable_hippocampus:
            # Curriculum phase 1: keep hippo neurons silent (zero drive) so they
            # don't fire and don't accumulate STDP eligibility. Cortex→D1 trains
            # without hippo noise.
            bridge.cp_external_input_current[region_indices_cp["place_cells"]] = cp.float32(0.0)
            bridge.cp_external_input_current[region_indices_cp["goal_cells"]] = cp.float32(0.0)

        # Landmark perception drive (Item 1 Stage 2, 2026-04-27).
        # Drives landmark_sensors based on agent's bearing+distance to a
        # FIXED landmark position. Each unique (distance, bearing) gives a
        # unique sensor activation pattern, so place_cells can self-organize
        # to fire at specific positions via the plastic landmark→place pathway.
        if enable_landmarks:
            in_goal_silence_step_lm = (goal_silence_after_step >= 0
                                       and step >= goal_silence_after_step
                                       and step < goal_silence_after_step + goal_silence_duration)
            if in_sleep or in_goal_silence_step_lm:
                bridge.cp_external_input_current[region_indices_cp["landmark_sensors"]] = cp.float32(0.0)
            else:
                lx, ly = landmark_position
                ldx = float(lx - x); ldy = float(ly - y)
                ldist = (ldx * ldx + ldy * ldy) ** 0.5
                if ldist < 1e-6:
                    sensor_act = np.full(n_landmark_sensors, landmark_max_intensity, dtype=np.float32)
                else:
                    bearing_x = ldx / ldist
                    bearing_y = ldy / ldist
                    intensity = landmark_max_intensity / (1.0 + landmark_falloff * ldist)
                    cos_alignment = landmark_pref_x * bearing_x + landmark_pref_y * bearing_y
                    sensor_act = intensity * np.maximum(0.0, cos_alignment)
                bridge.cp_external_input_current[region_indices_cp["landmark_sensors"]] = (
                    cp.asarray(sensor_act, dtype=cp.float32)
                )

        # Beacon perception drive (Item 1 Stage 1, 2026-04-27).
        # The beacon emits intensity that falls off with distance from goal.
        # Each sensor has a preferred direction; activation is intensity ×
        # max(0, cosine_alignment) — modeling biological directional cue
        # detection (e.g., bilateral hearing inferring sound source direction).
        # During goal silence (PFC Stage 2 test) and sleep, beacon is also
        # silenced — these tests assume no external goal info available.
        if enable_beacon_perception:
            in_goal_silence_step = (goal_silence_after_step >= 0
                                    and step >= goal_silence_after_step
                                    and step < goal_silence_after_step + goal_silence_duration)
            if in_sleep or in_goal_silence_step:
                bridge.cp_external_input_current[region_indices_cp["beacon_sensors"]] = cp.float32(0.0)
            else:
                # Compute beacon-to-agent vector
                bdx = float(gx - x)
                bdy = float(gy - y)
                distance = (bdx * bdx + bdy * bdy) ** 0.5
                if distance < 1e-6:
                    # On top of beacon: all sensors max
                    sensor_act = np.full(n_beacon_sensors,
                                         beacon_max_intensity,
                                         dtype=np.float32)
                else:
                    bearing_x = bdx / distance
                    bearing_y = bdy / distance
                    intensity = beacon_max_intensity / (1.0 + beacon_falloff * distance)
                    cos_alignment = beacon_pref_x * bearing_x + beacon_pref_y * bearing_y
                    sensor_act = intensity * np.maximum(0.0, cos_alignment)
                bridge.cp_external_input_current[region_indices_cp["beacon_sensors"]] = (
                    cp.asarray(sensor_act, dtype=cp.float32)
                )

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
        # Cluster C v2 (2026-04-29): expose selected action so per-action DA
        # production rules can fire only for the matching channel.
        bridge.core_config.last_selected_action = int(action_idx)

        dx, dy = ACTION_DELTAS[action_idx]
        # During sleep, agent does not move (consolidation phase, no behavior)
        if in_sleep:
            new_x, new_y = x, y
        else:
            new_x = int(np.clip(x + dx, 0, grid_size - 1))
            new_y = int(np.clip(y + dy, 0, grid_size - 1))
        dist_after = manhattan(new_x, new_y)
        x, y = new_x, new_y
        trajectory.append((x, y))
        goal_log.append((gx, gy))
        distance_log.append(dist_after)

        # Reward computation. Default uses Manhattan distance change (cheat:
        # uses raw (gx, gy)). Sensed reward instead uses beacon-intensity
        # gradient (the agent "feels warmer" as it approaches), which operates
        # on the perceptual signal — biologically grounded.
        if enable_sensed_reward and enable_beacon_perception:
            # Compute beacon intensity at old vs new position
            d_before = float(((gx - (x - dx)) ** 2 + (gy - (y - dy)) ** 2) ** 0.5) if not in_sleep else 0.0
            d_after = float(((gx - x) ** 2 + (gy - y) ** 2) ** 0.5)
            intensity_before = beacon_max_intensity / (1.0 + beacon_falloff * d_before)
            intensity_after = beacon_max_intensity / (1.0 + beacon_falloff * d_after)
            intensity_diff = intensity_after - intensity_before
            # Threshold to avoid noise; sign-only output
            if intensity_diff > 1e-3:
                reward = 1.0
            elif intensity_diff < -1e-3:
                reward = -1.0
            else:
                reward = 0.0
        else:
            if dist_after < dist_before:
                reward = 1.0
            elif dist_after > dist_before:
                reward = -1.0
            else:
                reward = 0.0
        # Interactive reward injection (2026-04-28): additive on top of
        # the natural reward. Lets the user "click +reward" from the webapp
        # to test conditioning / exploration in real time.
        if manual_reward_injection != 0.0:
            reward = float(reward) + manual_reward_injection
            if verbose:
                print(f"[g11 seed={seed}] step {step}: INTERACTIVE REWARD "
                      f"injection {manual_reward_injection:+.2f} -> reward={reward:+.2f}",
                      flush=True)
        reward_log.append(float(reward))

        # Log successful (place, goal) tuples during wake for sleep-replay.
        # When reward > 0 (agent moved toward goal), the (place_before, goal)
        # pairing is biologically meaningful and should be replayed during
        # sleep for memory consolidation. Only logged during wake (not sleep).
        if reward > 0 and not in_sleep:
            successful_trajectories.append((x, y, gx, gy))
            if len(successful_trajectories) > SUCCESSFUL_TRAJ_MAX:
                # Drop oldest to keep memory bounded
                successful_trajectories.pop(0)

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

        if verbose and progress_print_interval > 0 and (step + 1) % progress_print_interval == 0:
            recent_dist = float(np.mean(distance_log[-100:]))
            print(f"[g11 seed={seed}] step {step+1}/{n_steps}  pos=({x},{y})  "
                  f"goal=({gx},{gy})  recent_dist={recent_dist:.2f}  "
                  f"actions={action_log[-100:].count(0):>3d}N/{action_log[-100:].count(1):>3d}E/"
                  f"{action_log[-100:].count(2):>3d}S/{action_log[-100:].count(3):>3d}W",
                  flush=True)

        # Optional throttle for human-watchable speed in interactive mode.
        if trial_sleep_ms > 0:
            time.sleep(trial_sleep_ms / 1000.0)

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
    ap.add_argument("--grid-size", type=int, default=8,
                    help="Side length of square gridworld (default 8). Larger grids stress-test the architecture.")
    ap.add_argument("--n-hippocampus-per-layer", type=int, default=64,
                    help="Number of place + goal cells per layer (should be ~grid_size² for one cell per position; default 64 = 8×8).")
    ap.add_argument("--sensory-to-cortex-weight", type=float, default=10.0,
                    help="Initial mean weight for sensory→cortex pathway (default 10). Higher values let input layer drive cortex more strongly during phase 2.")
    ap.add_argument("--hippocampus-to-cortex-weight", type=float, default=10.0,
                    help="Initial mean weight for hippocampus→cortex pathway (default 10). Higher = stronger plastic input contribution.")
    # Canonical: --enable-dlpfc-wm. The implementation is a single recurrent
    # attractor modeling dlPFC working-memory persistent activity (catalog
    # G.06 / G.08), not the whole prefrontal cortex (dlPFC + vmPFC + OFC + ACC).
    # Legacy --pfc kept as alias for one release cycle (2026-04-29 Wave-1 #2).
    ap.add_argument("--enable-dlpfc-wm", "--pfc", action="store_true",
                    dest="pfc",
                    help="Enable a dlPFC working-memory module (one recurrent "
                         "attractor pool implementing persistent activity, NOT "
                         "the whole prefrontal cortex). Catalog G.06 / G.08.")
    ap.add_argument("--n-dlpfc-wm", "--n-pfc", type=int, default=60,
                    dest="n_pfc",
                    help="Number of dlPFC working-memory neurons (default 60).")
    ap.add_argument("--pfc-internal-density", type=float, default=0.2,
                    help="PFC recurrent connection density (default 0.2; higher = more persistent activity).")
    ap.add_argument("--goal-to-pfc-weight", type=float, default=8.0)
    ap.add_argument("--pfc-to-cortex-weight", type=float, default=8.0)
    ap.add_argument("--beacon-perception", action="store_true",
                    help="Item 1 Stage 1: enable beacon_sensors region with directional tuning. Sensors are driven each step based on perceived beacon strength + bearing.")
    ap.add_argument("--n-beacon-sensors", type=int, default=8,
                    help="Number of beacon sensors (default 8 = cardinal+diagonal).")
    ap.add_argument("--beacon-to-goal-weight", type=float, default=8.0)
    ap.add_argument("--beacon-max-intensity", type=float, default=600.0,
                    help="Peak sensor drive (pA) when on top of beacon (default 600).")
    ap.add_argument("--beacon-falloff", type=float, default=1.0,
                    help="Intensity = peak / (1 + falloff*distance). Higher = faster falloff.")
    ap.add_argument("--beacon-replaces-goal", action="store_true",
                    help="If set, beacon → goal_cells is the ONLY goal info source (true perception test). Otherwise beacon adds info on top of direct goal_cells drive.")
    ap.add_argument("--cue-reflex", action="store_true",
                    help="Item 1 Stage 3: cue-following reflex computes cortex drive from beacon sensors (innate sensorimotor wiring like phototaxis). Augments heuristic by default; use --cue-reflex-replaces-heuristic to fully replace.")
    ap.add_argument("--cue-reflex-strength", type=float, default=800.0,
                    help="Peak reflex drive (pA) — matches heuristic strength by default.")
    ap.add_argument("--cue-reflex-replaces-heuristic", action="store_true",
                    help="If set with --cue-reflex, the heuristic is fully disabled; only reflex provides cortex drive.")
    # Canonical name: --enable-landmark-sensor (the implementation is a sensor
    # abstraction, not landmark-cell biology). --landmarks is the legacy alias
    # kept for one release cycle (2026-04-29 Wave-1 rename).
    ap.add_argument("--enable-landmark-sensor", "--landmarks", action="store_true",
                    dest="enable_landmark_sensor",
                    help="Item 1 Stage 2: enable fixed-position landmark with directional sensors. Plastic landmark_sensors → place_cells pathway lets place cells self-organize from sensor patterns.")
    ap.add_argument("--n-landmark-sensors", type=int, default=8)
    ap.add_argument("--landmark-to-place-weight", type=float, default=8.0)
    ap.add_argument("--landmark-x", type=float, default=None,
                    help="Landmark x position (default = grid_size/2)")
    ap.add_argument("--landmark-y", type=float, default=None,
                    help="Landmark y position (default = grid_size/2)")
    ap.add_argument("--landmark-max-intensity", type=float, default=600.0)
    ap.add_argument("--landmark-falloff", type=float, default=1.0)
    ap.add_argument("--landmarks-replace-place", action="store_true",
                    help="If set, place_cells receive ONLY landmark-derived input (no direct (x,y) cheat). True Stage 2 perception test.")
    ap.add_argument("--sensed-reward", action="store_true",
                    help="Cheat #4: compute reward from beacon-intensity gradient (sensed signal) instead of Manhattan distance change (cheat). Requires --beacon-perception.")
    ap.add_argument("--bg-cross-projections", action="store_true",
                    help="Cheat #5: enable cortex × str_D1 cross-projections (e.g. cortex_E → str_D1_W) at weak initial weight. Plasticity learns the right cross-strengths instead of hand-coded same-action-only.")
    ap.add_argument("--cross-projection-weight", type=float, default=5.0,
                    help="Initial weight for BG cross-projections (default 5.0 vs 25.0 same-action).")
    ap.add_argument("--cross-projection-density", type=float, default=1.0,
                    help="Cheat-5 option 2: pathway-level density of cross-projections at build time. "
                         "1.0=dense (24 cross-pathways, current default); 0.25=patch-matrix-like (6 of 24).")
    ap.add_argument("--cross-projection-topology-seed", type=int, default=0,
                    help="Cheat-5 option 2: deterministic RNG seed for which cross-pathways survive when density<1.0. "
                         "Vary independently from --seed to test topology-conditional reproducibility.")
    # Canonical: --enable-msn-lateral-inhibition (specifies MSN-MSN, not BG-wide).
    # Legacy --bg-lateral-inhibition kept as alias for one release cycle
    # (2026-04-29 Wave-1 rename #8). Note: catalog B.04 supplemental flags
    # this implementation as anatomically backwards — real cross-pool WTA in
    # striatum is FSI feedforward, not MSN-MSN feedback (Wilson 2007 PBR-160
    # ch 6). Kept as v3 default per 2026-04-28 evaluation; future biology
    # buildout should replace with FSI-mediated form.
    ap.add_argument("--enable-msn-lateral-inhibition", "--bg-lateral-inhibition",
                    action="store_true", dest="bg_lateral_inhibition",
                    help="v3 (2026-04-28): add MSN cross-pool lateral inhibition (24 GABAergic pathways). Sharpens action selection regardless of cheat #5; required prerequisite for cross-projection closure.")
    ap.add_argument("--lateral-inhibition-density", type=float, default=0.3,
                    help="Density of MSN cross-pool inhibitory pathways (default 0.3).")
    ap.add_argument("--lateral-inhibition-weight", type=float, default=2.0,
                    help="Weight of MSN cross-pool inhibitory connections (default 2.0).")
    ap.add_argument("--interactive-control-file", type=str, default=None,
                    help="If set, runner polls this JSON file at the start of "
                         "each trial for runtime control: paused (bool), "
                         "goal ([gx, gy] override, persistent), inject_reward "
                         "(one-shot additive). Used by webapp World-tab live "
                         "mode for click-to-teleport-goal etc.")
    ap.add_argument("--progress-print-interval", type=int, default=100,
                    help="Print a progress line every N steps (default 100). "
                         "Webapp interactive mode sets this to 1 for per-step "
                         "live animation.")
    ap.add_argument("--trial-sleep-ms", type=float, default=0.0,
                    help="Sleep this many ms between trials (default 0 = full "
                         "speed). Use 50-200 to watch the agent learn at "
                         "human-readable speed in interactive mode.")
    ap.add_argument("--bg-cross-thaw-step", type=int, default=-1,
                    help="Cheat #5 closure (2026-04-28): step at which bg_cross_projections "
                         "gate thaws to its phase-3 value. -1 = stay frozen. Recommended 1200 "
                         "for default 1800-step moving-goal episodes (~300 steps after goal "
                         "change at step 900). Requires --bg-cross-projections + --curriculum.")
    ap.add_argument("--bg-cross-phase3-gain", type=float, default=0.5,
                    help="Plasticity gain for bg_cross_projections in phase 3. 1.0 = full plastic, "
                         "0.5 = half-rate (slower than same-action; default), 0.0 = stay frozen.")
    # v4 (2026-04-28): developmental pretraining
    ap.add_argument("--developmental-pretraining", action="store_true",
                    help="v4 cheat-5 closure: run a critical-period analog "
                         "(all plasticity gates open) on N random goals before "
                         "the standard eval. Cross-projections freeze at eval "
                         "start. Requires --bg-cross-projections.")
    ap.add_argument("--pretraining-n-goals", type=int, default=10,
                    help="Number of random goal positions during pretraining (default 10).")
    ap.add_argument("--pretraining-steps-per-goal", type=int, default=3000,
                    help="Trials per pretraining goal (default 3000). 10x3000=30K "
                         "default total; reduce for tier-2 smoke (e.g. 1000) or "
                         "tier-1 wiring check (e.g. 1 goal x 1000).")
    ap.add_argument("--enable-structural-pruning", action="store_true",
                    help="Cheat-5 option 1: experience-dependent synapse pruning during "
                         "pretraining. Synapses with negative survival score AND low weight "
                         "get permanently eliminated. See "
                         "docs/plans/2026-04-28-structural-plasticity-design.md.")
    ap.add_argument("--enable-d1-d2-asymmetry", action="store_true",
                    help="Cluster B.1: D1/D2 plasticity asymmetry — D2-targeting "
                         "synapses' weight updates flip sign vs D1. See "
                         "docs/plans/2026-04-28-cluster-b1-d1d2-asymmetry-implementation.md.")
    # Canonical: --enable-striatal-pv-fsi (specifies PV+ FSI; per Tepper-2018
    # this is one of EIGHT distinct striatal GABAergic interneuron classes —
    # NPY-LTS, NPY-NGF, CR, TH/THIN, FAI, SABI, ChI/TAN are NOT modeled).
    # Legacy --enable-striatal-fsis kept as alias for one release cycle
    # (2026-04-29 Wave-1 rename #9). Region naming: str_PV_FSI_X (canonical)
    # with str_PV_FSI_X retained as a region-name alias via RegionManager.
    ap.add_argument("--enable-striatal-pv-fsi", "--enable-striatal-fsis",
                    action="store_true", dest="enable_striatal_fsis",
                    help="Cluster B.2: striatal PV-FSI fast-spiking interneurons "
                         "(broadcast inhibition). One of 8 striatal GABAergic "
                         "interneuron classes per Tepper-2018; the others are NOT "
                         "modeled. See "
                         "docs/plans/2026-04-28-cluster-b2-striatal-fsis-implementation.md.")
    ap.add_argument("--enable-bg-neuropeptides", action="store_true",
                    help="R3.6 (2026-04-29): D1/D2 neuropeptide co-release. "
                         "Registers dynorphin (D1, KOR plasticity-rate brake), "
                         "substance P (D1, NK-1 ACh boost), and enkephalin "
                         "(D2, DOR plasticity-rate boost) neuromodulators. "
                         "Per PBR-160 ch 16 McGinty.")
    ap.add_argument("--enable-cluster-a-closed-loop", action="store_true",
                    help="Cluster A (2026-04-29): closed BG loop. Adds "
                         "cortex_X -> stn (hyperdirect, sparse) and "
                         "thal_X -> cortex_X (action-specific feedback). "
                         "Provides the teaching signal missing for "
                         "cross-projection learning. See "
                         "docs/plans/2026-04-29-cluster-a-closed-bg-loop-design.md.")
    ap.add_argument("--enable-tonic-da", action="store_true",
                    help="Cluster C v1 (2026-04-29): replace signed-scalar "
                         "reward modulation with a real `dopamine` "
                         "neuromodulator (tonic baseline + phasic "
                         "activation/depression). Unlocks B.3 TANs by "
                         "providing tonic DA-driven plasticity for ACh "
                         "to gate. See "
                         "docs/plans/2026-04-29-cluster-c-tonic-da-design.md.")
    ap.add_argument("--enable-compartmentalized-da", action="store_true",
                    help="Cluster C v2 (2026-04-29): replace single-channel "
                         "DA with 4 per-action DA modulators "
                         "(dopamine_{N,E,S,W}). Each targets only synapses "
                         "with matching action_index; production rule fires "
                         "only when last_selected_action matches. Implies "
                         "tonic DA at the per-action level (the global "
                         "`dopamine` modulator is NOT registered when this "
                         "flag is on, even if --enable-tonic-da is set). "
                         "See docs/plans/2026-04-29-cluster-c-v2-"
                         "compartmentalized-da-design.md.")
    ap.add_argument("--enable-cluster-d-hippocampus", action="store_true",
                    help="Cluster D v1 (2026-04-29): hippocampus trisynaptic "
                         "loop. Adds 5 regions (ec, dg, dg_fs, ca3, ca1) and "
                         "~10 pathways implementing the canonical Cajal loop "
                         "(EC -> DG -> CA3 -> CA1 + EC -> CA1 direct + CA3 "
                         "recurrent autoassociator). Composes with --hippocampus "
                         "(adds ca1 -> place_cells readout) and --landmarks "
                         "(adds landmark_sensors -> ec). See "
                         "docs/plans/2026-04-29-cluster-d-hippocampus-design.md.")
    ap.add_argument("--enable-cluster-e-topography", action="store_true",
                    help="Cluster E v1 (2026-04-29): topographic maps + "
                         "distance-dependent connection probability. "
                         "cortex_X / str_D1_X / str_D2_X regions get 2D coords "
                         "anchored to corners of unit square (N=(0.5,1.0), "
                         "E=(1.0,0.5), S=(0.5,0.0), W=(0.0,0.5)); cortex_X -> "
                         "str_D{1,2}_Y pathways are sampled with Gaussian-"
                         "weighted probability (sigma=0.3 default, set via "
                         "--cluster-e-distance-sigma). See "
                         "docs/plans/2026-04-29-cluster-e-topographic-maps-design.md.")
    ap.add_argument("--cluster-e-distance-sigma", type=float, default=0.3,
                    help="Cluster E v1 Gaussian-kernel sigma for distance-"
                         "weighted cortex -> striatum connectivity. Default 0.3 "
                         "(at corner-to-corner distance ~1.0, cross-action prob "
                         "drops to ~0.4%% of same-action). Larger -> looser "
                         "spatial selectivity.")
    ap.add_argument("--enable-tans", action="store_true",
                    help="Cluster B.3: cholinergic interneurons (TANs). Adds "
                         "an acetylcholine_tan neuromodulator (the striatal-TAN-"
                         "specific ACh source) that pauses on reward and gates "
                         "corticostriatal plasticity windows. See "
                         "docs/plans/2026-04-28-cluster-b3-tans-implementation.md.")
    ap.add_argument("--pruning-alpha", type=float, default=None,
                    help="Cheat-5 option-1 pruning rate. Default: cfg.pruning_alpha (0.001 = conservative). "
                         "Try 0.05 for a 5K-trial pretraining smoke; 0.005 for 30K validation.")
    ap.add_argument("--pruning-threshold", type=float, default=None,
                    help="Cheat-5 option-1: survival score below which pruning is eligible. Default: -1.0.")
    ap.add_argument("--pruning-weight-floor", type=float, default=None,
                    help="Cheat-5 option-1: weight below which pruning is eligible. Default: 1.0.")
    ap.add_argument("--out", type=str, default=None)
    # DEPRECATED 2026-04-29 (Wave-1 rename master plan #11). NEGATIVE on
    # cheat-5 evaluation; biology is wrong (real motor-pool WTA is via spinal
    # Renshaw cells / reciprocal inhibition per Kandel ch 35, not cortical-FS-
    # like inhibition). Slated for removal in a future cleanup. The
    # motor_FS_X regions and motor_X→motor_FS_X→motor_Y plumbing remain for
    # archival reproducibility of 2026-04-26 findings.
    ap.add_argument("--motor-lateral-inhibition", "--enable-motor-pool-wta",
                    action="store_true", dest="motor_lateral_inhibition",
                    help="DEPRECATED (NEGATIVE on cheat-5; slated for removal). "
                         "Enable FS-mediated motor pool lateral inhibition "
                         "(WTA microcircuit). Real motor-pool WTA biology is "
                         "spinal Renshaw, not cortical-FS-like inhibition.")
    ap.add_argument("--cortex-wta", action="store_true",
                    help="Enable cortex-level WTA: per-pool FS interneurons enforce one-cortex-pool-wins. Tools plastic input layers (hippocampus, learned-perception) to coexist with heuristic.")
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
    ap.add_argument("--deterministic", action="store_true",
                    help="Set CUBLAS_WORKSPACE_CONFIG=:4096:8 BEFORE cupy import for "
                         "deterministic cuBLAS algos. Tightens seed-to-seed noise "
                         "(2026-04-29 result: A+E det single-goal 3.31 +/- 0.74 vs "
                         "non-det 7.28 +/- 1.76 multi-goal). ~10-30% slowdown. "
                         "Note: this flag is read at module-import time (top of file), "
                         "not parsed here — argparse just suppresses 'unrecognized arg'.")
    ap.add_argument("--rpe-scaled-reward", action="store_true",
                    help="Scale reward by prediction error: delivered = reward + alpha * (reward - reward_ema). Surprise gets amplified.")
    ap.add_argument("--rpe-alpha", type=float, default=1.0)
    ap.add_argument("--surprise-lr-boost", action="store_true",
                    help="Boost reward_learning_rate when |RPE| is high (NE-like fast meta-modulation)")
    ap.add_argument("--surprise-lr-alpha", type=float, default=2.0)
    ap.add_argument("--curriculum", action="store_true",
                    help="Curriculum learning: suppress hippocampus drive for first N steps (cortex→D1 builds without hippo noise), then enable. Requires --hippocampus.")
    ap.add_argument("--curriculum-warmup-steps", type=int, default=600,
                    help="Steps to keep hippo silent at start of curriculum (default 600).")
    ap.add_argument("--curriculum-ramp-steps", type=int, default=0,
                    help="Smooth gate ramp window centered on warmup boundary (default 0 = abrupt step). Biologically grounded: critical periods close gradually via PV maturation.")
    ap.add_argument("--curriculum-phase2-cortex-gain", type=float, default=0.0,
                    help="Phase 2 plasticity gain for cortex→D1 (default 0.0 = full freeze). Biologically: cortical plasticity slows but doesn't fully halt.")
    ap.add_argument("--curriculum-phase2-hippo-gain", type=float, default=1.0,
                    help="Phase 2 plasticity gain for hippo→cortex (default 1.0 = full plasticity).")
    ap.add_argument("--heuristic-strength", type=float, default=1.0,
                    help="Heuristic cortex drive strength multiplier (default 1.0). 0.0 disables heuristic.")
    ap.add_argument("--heuristic-decay-after-step", type=int, default=-1,
                    help="Step after which heuristic_strength changes to --post-curriculum-heuristic-strength (default -1 = no decay).")
    ap.add_argument("--post-curriculum-heuristic-strength", type=float, default=0.0,
                    help="Heuristic strength after decay step (default 0.0 = full off).")
    ap.add_argument("--sleep-replay-after-step", type=int, default=-1,
                    help="Step at which to enter sleep-replay phase (default -1 = no sleep). During sleep, hippo replays random place/goal patterns, corticostriatal thaws for consolidation.")
    ap.add_argument("--sleep-replay-steps", type=int, default=300,
                    help="Number of steps in sleep-replay phase.")
    ap.add_argument("--sleep-replay-rate-hz", type=float, default=200.0,
                    help="Replay drive rate (Hz) — biologically: sharp-wave ripples ~150-250Hz.")
    ap.add_argument("--sleep-nrem-rem-alternate", action="store_true",
                    help="Alternate between NREM (trajectory replay, first half) and REM (random replay, second half) during sleep.")
    ap.add_argument("--goal-silence-after-step", type=int, default=-1,
                    help="PFC Stage 2 delayed-response test: silence goal_cells AND heuristic at this step. PFC working memory should maintain goal info.")
    ap.add_argument("--goal-silence-duration", type=int, default=0,
                    help="How long to keep goal_cells/heuristic silenced.")
    args = ap.parse_args()

    if args.moving_goal:
        out_path = args.out or f"research/findings/raw/g11_bg/g11_seed{args.seed}.json"
        # Scale goal positions to grid size — keeps relative spacing the same
        # so the same task structure works at any grid scale. Defaults are
        # ~75% and ~12% of grid extent (matches the 8×8 (6,6) and (1,6)).
        gs = args.grid_size
        far = (max(0, gs - 2), max(0, gs - 2))            # was (6, 6)
        far_west = (max(0, 1), max(0, gs - 2))            # was (1, 6)
        sw = (max(0, 1), max(0, 1))                        # was (1, 1)
        far_se = (max(0, gs - 2), max(0, 1))              # was (6, 1)
        if args.goal_schedule == "multi":
            goal_schedule = [(0, far), (450, far_west), (900, sw), (1350, far_se)]
        elif args.goal_schedule == "curriculum":
            flip = max(1200, args.curriculum_warmup_steps + 600)
            goal_schedule = [(0, far), (flip, far_west)]
        else:
            goal_schedule = [(0, far), (300, far_west)]
        run_moving_goal_episode(
            out_path=out_path,
            seed=args.seed,
            n_steps=args.n_steps,
            grid_size=args.grid_size,
            n_hippocampus_per_layer=args.n_hippocampus_per_layer,
            sensory_to_cortex_weight=args.sensory_to_cortex_weight,
            hippocampus_to_cortex_weight=args.hippocampus_to_cortex_weight,
            enable_pfc=args.pfc,
            n_pfc=args.n_pfc,
            pfc_internal_density=args.pfc_internal_density,
            goal_to_pfc_weight=args.goal_to_pfc_weight,
            pfc_to_cortex_weight=args.pfc_to_cortex_weight,
            enable_beacon_perception=args.beacon_perception,
            n_beacon_sensors=args.n_beacon_sensors,
            beacon_to_goal_weight=args.beacon_to_goal_weight,
            beacon_max_intensity=args.beacon_max_intensity,
            beacon_falloff=args.beacon_falloff,
            beacon_replaces_goal=args.beacon_replaces_goal,
            enable_cue_reflex=args.cue_reflex,
            cue_reflex_strength=args.cue_reflex_strength,
            cue_reflex_replaces_heuristic=args.cue_reflex_replaces_heuristic,
            enable_landmarks=args.enable_landmark_sensor,
            n_landmark_sensors=args.n_landmark_sensors,
            landmark_to_place_weight=args.landmark_to_place_weight,
            landmark_position=(args.landmark_x, args.landmark_y) if args.landmark_x is not None and args.landmark_y is not None else None,
            landmark_max_intensity=args.landmark_max_intensity,
            landmark_falloff=args.landmark_falloff,
            landmarks_replace_place=args.landmarks_replace_place,
            enable_sensed_reward=args.sensed_reward,
            enable_bg_cross_projections=args.bg_cross_projections,
            cross_projection_weight=args.cross_projection_weight,
            cross_projection_density=args.cross_projection_density,
            cross_projection_topology_seed=args.cross_projection_topology_seed,
            bg_cross_thaw_step=args.bg_cross_thaw_step,
            bg_cross_phase3_gain=args.bg_cross_phase3_gain,
            enable_bg_lateral_inhibition=args.bg_lateral_inhibition,
            enable_developmental_pretraining=args.developmental_pretraining,
            pretraining_n_goals=args.pretraining_n_goals,
            pretraining_steps_per_goal=args.pretraining_steps_per_goal,
            enable_structural_pruning=args.enable_structural_pruning,
            enable_d1_d2_asymmetry=args.enable_d1_d2_asymmetry,
            enable_striatal_fsis=args.enable_striatal_fsis,
            enable_tans=args.enable_tans,
            enable_bg_neuropeptides=args.enable_bg_neuropeptides,
            enable_cluster_a_closed_loop=args.enable_cluster_a_closed_loop,
            enable_tonic_da=args.enable_tonic_da,
            enable_compartmentalized_da=args.enable_compartmentalized_da,
            enable_cluster_d_hippocampus=args.enable_cluster_d_hippocampus,
            enable_cluster_e_topography=args.enable_cluster_e_topography,
            cluster_e_distance_sigma=args.cluster_e_distance_sigma,
            pruning_alpha=args.pruning_alpha,
            pruning_threshold=args.pruning_threshold,
            pruning_weight_floor=args.pruning_weight_floor,
            lateral_inhibition_density=args.lateral_inhibition_density,
            lateral_inhibition_weight=args.lateral_inhibition_weight,
            interactive_control_file=args.interactive_control_file,
            progress_print_interval=args.progress_print_interval,
            trial_sleep_ms=args.trial_sleep_ms,
            goal_schedule=goal_schedule,
            enable_motor_lateral_inhibition=_warn_motor_lateral_inhibition_deprecated(args.motor_lateral_inhibition),
            enable_cortex_lateral_inhibition=args.cortex_wta,
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
            enable_curriculum=args.curriculum,
            curriculum_warmup_steps=args.curriculum_warmup_steps,
            curriculum_ramp_steps=args.curriculum_ramp_steps,
            curriculum_phase2_cortex_gain=args.curriculum_phase2_cortex_gain,
            curriculum_phase2_hippo_gain=args.curriculum_phase2_hippo_gain,
            heuristic_strength=args.heuristic_strength,
            heuristic_decay_after_step=args.heuristic_decay_after_step,
            post_curriculum_heuristic_strength=args.post_curriculum_heuristic_strength,
            sleep_replay_after_step=args.sleep_replay_after_step,
            sleep_replay_steps=args.sleep_replay_steps,
            sleep_replay_rate_hz=args.sleep_replay_rate_hz,
            sleep_nrem_rem_alternate=args.sleep_nrem_rem_alternate,
            goal_silence_after_step=args.goal_silence_after_step,
            goal_silence_duration=args.goal_silence_duration,
        )
        return 0

    from sim import (
        SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig,
    )
    from sim.enums import NeuronModel
    import cupy as cp

    print(f"\n{'='*72}")
    print(f"  G11 BG Action Selection Module -- Smoke Test")
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

    # Cluster C v2 (2026-04-29) smoke compatibility: register per-action DA
    # modulators if --enable-compartmentalized-da is set. Smoke run will
    # exercise the registration path; reward modulation is disabled so the
    # DA signal is not actually consumed but the array allocations and
    # registration are validated.
    if args.enable_compartmentalized_da:
        from sim.neuromodulators import _default_per_action_dopamine_config
        cfg.enable_neuromodulator_subsystem = True
        cfg.neuromodulators = list(cfg.neuromodulators) + [
            _default_per_action_dopamine_config(action, idx)
            for idx, action in enumerate(ACTION_NAMES)
        ]
        print(f"  Cluster C v2: registered {len(ACTION_NAMES)} per-action DA modulators "
              f"(dopamine_{{{','.join(ACTION_NAMES)}}})")

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

    print(f"\n  Smoke test PASSED -- {len(regions)} regions, "
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
        for region_name in ["stn", "snc"]:
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
        ordered_groups += ["stn", "snc"]
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
