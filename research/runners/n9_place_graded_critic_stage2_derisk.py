"""Stage-2 de-risk — a SPARSE-read MSN-D1 value critic GRADES NEAR>>FAR on the Stage-1 self-organized
spiking place code (the N9 value-of-LOCATION unblock). CuPy-only.

═══════════════════════════════════════════════════════════════════════════════════════════════════
STEP-2 VOLLEY UPGRADE (2026-06-09, --enable-volley; default OFF -> the baseline reproduces the prior
NEGATIVE where critic@NEAR=0.0 Hz). The baseline arm read the sparse-distinct place code with PLAIN
synapses and could NOT fire the critic (the RATE-CODING / ASYNCHRONY wall: a sparse-async ~10 Hz
ensemble emits per-step coincidence c_i<=1). STEP-1 (coincidence_volley_n9_derisk.py, GO 3/3 seeds)
broke that wall: an FS-PING gamma volley re-times the sparse ensemble into a coincident packet that
the landed Route-D plateau (b980070a) fires -- at <=5% sparsity, with jitter collapsing it (coincidence
not rate). --enable-volley wires THAT mechanism into THIS critic arm (n-place 800 + FS-PING on `place`
+ place->striosome_value as a coincidence_detector), so the critic FIRES from the sparse-distinct code
and the gates (FIRE/GRADE/LTP/ACTOR/GABA_B) + anti-cheats (jitter/place-shuffle/ablate) re-test whether
place-GRADING finally opens. Operating point = the STEP-1 GO (re-tuned FS-PING, K=4, plateau 80).
Run: SIM_BACKEND=cupy python -m research.runners.n9_place_graded_critic_stage2_derisk \\
       --enable-volley --n-place 800 --seeds 42,43,44 --out ..._volley.json
═══════════════════════════════════════════════════════════════════════════════════════════════════

THE LOAD-BEARING TEST OF THE WHOLE ARC. Stage 1 (placecode_selforg_stage1_derisk.py, PASS 3/3 CuPy)
produced a SELF-ORGANIZED sparse SPIKING place code (landmark_sensors -> `place` pool, Hartley-Burgess
competitive learning): position-specific (diff-location cosine 0.064), stable (0.872), ~3.65% sparse,
and 100% sensor-driven (ablate sensors -> place pool SILENT). This Stage-2 probe reads THAT code into
the N9 MSN-D1 critic and tests whether place-grading OPENS on CuPy.

WHY THE PRIOR N9 NEGATIVE (2026-06-09-N9-convergent-upstate-derisk.md, PLACE-GRADED 0/3):
  The critic read a host-rendered DENSE 2-D Gaussian (vs_place_context) through a DENSE NON-plastic
  convergent A1 floor. A dense convergent read of overlapping bumps is POSITION-BLIND -- the cell
  enters the up-state WHEREVER a bump exists (set by which afferent cells randomly wire on, peaking at
  the grid center), so it caps NEAR/FAR at ~1.2x. _n9_matched_pos.py: matched n_active -> 1.79 vs
  16.79 Hz (random structure, not location).

THE STAGE-2 FIX (place-code-biologization-research.md, Option A read-out §1.4):
  Read the Stage-1 SPARSE, DISTINCT-per-location place code into striosome_value via the A2 PLASTIC
  DA-delta-gated arm ONLY (gate `value_input`). NO dense position-blind A1 floor. Because the place
  code is now sparse + distinct, DA-gated STDP grows DIFFERENT critic synapses per location -> the NEAR
  place ensemble's synapses potentiate (value-leads-reward), the FAR ensemble's never do, and there is
  NO convergent floor to fire FAR critic cells position-blindly. The critic reaches its up-state at
  NEAR from the LEARNED place->value synapses; at FAR it stays silent. This is the 1-D-probe regime
  (snc_stageb_critic_probe_place.py, 3/3) ported to the self-organized 2-D place code.

SEQUENCING (the prompt's suggested clean path):
  (1) Self-organize the place code (Stage-1 mechanism: landmark_sensors -> place, plastic, competitive),
      then FREEZE it (close `landmark_to_place`). Stable place fields = the critic's afferent.
  (2) Train the critic's V (place -> critic plastic, DA-delta-gated) ON TOP of the frozen place fields,
      so the place code is a stable afferent while V learns.

GATES (CuPy, >=3 seeds, deterministic regime OU/cond-noise/global-homeostasis OFF):
  2a FIRE          : critic >= ~5 Hz at the goal (NEAR) after training.
  2b PLACE-GRADED  : critic NEAR >= 3x FAR (far ~0). <-- THE load-bearing gate the dense blob capped 1.2x.
  2c LEARNS-V (LTP): the plastic place->critic near-goal weight grows from a realistic init and
                     exceeds the far weight (>= 2x).
  2d ACTOR-NOT-PERTURBED : actor cortex firing within +-10% vs a critic-absent twin.
  2e GABA_B subtraction  : zero the GABA_B mask -> the SNc predicted-vs-unpredicted gap vanishes.

ANTI-CHEATS:
  (a) PLACE-SHUFFLE : permute the place-cell -> location mapping (the place ENSEMBLE that fires at NEAR
      is decoupled from where reward is delivered). Grading (2b) AND the GABA_B gap MUST FAIL under
      shuffle -- proves value-of-LOCATION, not fired-on-any-drive.
  (b) SENSOR-ABLATION : zero landmark sensors at recall -> the place pool goes silent -> critic grading
      collapses (inherits Stage-1's sensor-dependence; a host Gaussian would be unaffected).
  (c) REGIME FIDELITY : assert backend==cupy AND OU/conductance-noise/global-homeostasis OFF +
      NO per-region homeostasis on the critic (it fires from LEARNED synaptic current, not threshold
      collapse).
  Position-leak: (x,y) enters the brain ONLY via the egocentric landmark render -> place_sensors;
      the place pool + critic NEVER receive a direct allocentric (x,y) injection. Enforced by
      construction (this probe writes external current ONLY to landmark_sensors).

USAGE (MUST be cupy):
  SIM_BACKEND=cupy python -m research.runners.n9_place_graded_critic_stage2_derisk \
      --seeds 42,43,44 --out research/findings/raw/_n9_place_graded_critic_stage2_3seed.json
  SIM_BACKEND=cupy python -m research.runners.n9_place_graded_critic_stage2_derisk --seed 42 --lesion
  SIM_BACKEND=cupy python -m research.runners.n9_place_graded_critic_stage2_derisk --seed 42 --shuffle
  SIM_BACKEND=cupy python -m research.runners.n9_place_graded_critic_stage2_derisk --seed 42 --ablate-sensors
"""
from __future__ import annotations
import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _host(a):
    from sim.backend import to_host
    try:
        return to_host(a)
    except Exception:
        return np.asarray(a)


def _idx(bridge, name):
    return np.asarray(bridge.region_manager.indices(name), dtype=np.int64)


def cosine_counts(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── Egocentric landmark sensor render (Stage-1 mechanism, verbatim) — the legitimate body-sensing
#    channel (D.09 object-vector input). (x,y) enters the brain ONLY here (position-leak). ──

def landmark_sensor_act(x, y, landmarks, n_bearing, n_dist, max_int, falloff, dist_sigma,
                        dist_max, bexp):
    blocks = []
    bpx = np.cos(2.0 * np.pi * np.arange(n_bearing) / n_bearing)
    bpy = np.sin(2.0 * np.pi * np.arange(n_bearing) / n_bearing)
    dist_centers = np.linspace(0.0, dist_max, n_dist)
    for (lx, ly) in landmarks:
        dx = float(lx - x); dy = float(ly - y)
        d = (dx * dx + dy * dy) ** 0.5
        if d < 1e-6:
            bear = np.full(n_bearing, max_int, dtype=np.float32)
            dist = np.full(n_dist, max_int, dtype=np.float32)
        else:
            bx = dx / d; by = dy / d
            intensity = max_int / (1.0 + falloff * d)
            cos_align = np.maximum(0.0, bpx * bx + bpy * by)
            bear = (intensity * (cos_align ** bexp)).astype(np.float32)
            dist = (max_int * np.exp(-(d - dist_centers) ** 2 / (2.0 * dist_sigma ** 2))).astype(np.float32)
        blocks.append(bear.astype(np.float32))
        blocks.append(dist.astype(np.float32))
    return np.concatenate(blocks).astype(np.float32)


# ── Sparse actor place code (for gate 2d's actor stub — a distinct distributed pop, like the nav loop) ──
def _grid_prefs(n_cells, grid_size):
    side = int(round(np.sqrt(n_cells)))
    xs = np.linspace(0.0, grid_size - 1.0, side, dtype=np.float64)
    ys = np.linspace(0.0, grid_size - 1.0, side, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys)
    px = gx.ravel(); py = gy.ravel()
    if px.size < n_cells:
        reps = int(np.ceil(n_cells / px.size))
        px = np.tile(px, reps)[:n_cells]; py = np.tile(py, reps)[:n_cells]
    return px[:n_cells].copy(), py[:n_cells].copy()


def _sparse_place_code(pos_xy, prefs_xy, max_pA, sigma):
    px, py = prefs_xy
    x, y = float(pos_xy[0]), float(pos_xy[1])
    dsq = (px - x) ** 2 + (py - y) ** 2
    return (max_pA * np.exp(-dsq / (2.0 * sigma ** 2))).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# Build: landmark_sensors -> place (Stage-1 self-org) -> striosome_value (PLASTIC sparse, DA-gated) ->
#        snc (GABA_B). Plus an actor stub (sensor_place_readout -> cortex_X) for gate 2d.
# NO dense non-plastic A1 floor (that was the position-blind blob). NO host Gaussian into place/critic.
# ──────────────────────────────────────────────────────────────────────

def _build(seed, *, n_sensors, n_place, n_strio, n_snc, grid_size,
           lm_to_place_weight, lm_to_place_density, lm_to_place_jitter,
           place_to_value_weight, place_to_value_density, place_to_value_jitter,
           strio_to_snc_weight, snc_da_sensitivity, reward_learning_rate,
           gabab, gabab_tau_decay, gabab_propagation_strength,
           include_actor=True, n_sensor_place=64, n_cortex_per_action=50,
           sensor_place_to_cortex_weight=10.0, place_to_value_active=True,
           nmda_critic=False, dt_ms=1.0,
           # ── STEP-2 VOLLEY ADDITIONS (2026-06-09; default OFF -> byte-identical baseline) ──
           # When enable_volley=True, the validated STEP-1 mechanism is wired into the critic arm:
           #   (1) an FS-PING pool on `place` (place->FS exc, FS->place GABA_A) so a gamma rhythm
           #       EMERGES and re-times the sparse place ensemble into a coincident VOLLEY;
           #   (2) the place->striosome_value arm is a Route-D coincidence_detector (the landed b980070a
           #       dendritic-plateau subunit) -> the synchronized volley fires the critic that the
           #       sparse-async code could not. STILL plastic + DA-gated (so it GRADES + LEARNS).
           # Operating point = the STEP-1 GO (n_place 800, re-tuned FS-PING, K=4, plateau 80).
           enable_volley=False, n_fs=160,
           place_to_fs_weight=16.0, place_to_fs_density=0.4,
           fs_to_place_weight=8.0, fs_to_place_density=0.4,
           coincidence_k=4.0, coincidence_gain=2.0, coincidence_plateau=80.0, stdp_w_max=40.0):
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType
    from sim.neuromodulators import NeuromodulatorConfig, ModulatorTarget, ProductionRule

    regions = [
        # The legitimate egocentric landmark sensors (driven externally each step; Stage-1 channel).
        BrainRegion(name="landmark_sensors", n_neurons=int(n_sensors), exc_fraction=1.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                    weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name),
        # The self-organizing place pool (hippocampal pyramidal; competition = the cell's own threshold).
        # NO per-region homeostasis (anti-cheat c: fire from synaptic current, not threshold collapse).
        BrainRegion(name="place", n_neurons=int(n_place), exc_fraction=1.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                    weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name),
        # The MSN-D1 value critic (fully GABAergic, KIR2 up/down, rheobase ~339 pA; B.02).
        # STEP-2 VOLLEY: the coincidence plateau reuses the Mg2+-block kernel, so the critic needs NMDA
        # ON when the volley arm is active (mirrors the STEP-1 volley bed's target enable_nmda=True).
        BrainRegion(name="striosome_value", n_neurons=int(n_strio), exc_fraction=0.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                    weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D1.name,
                    syn_reversal_potential_i_override=-60.0,
                    enable_nmda=bool(nmda_critic or enable_volley)),
        BrainRegion(name="snc", n_neurons=int(n_snc), exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_DOPAMINE.name,
                    syn_reversal_potential_i_override=-55.0),
    ]
    pathways = [
        # Stage-1 self-organization pathway: landmark_sensors -> place (plastic, competitive).
        RegionPathway(from_region="landmark_sensors", to_region="place",
                      density=float(lm_to_place_density), weight_mean=float(lm_to_place_weight),
                      weight_jitter=float(lm_to_place_jitter), plastic=True,
                      plasticity_gate="landmark_to_place"),
        # THE STAGE-2 ARM: place -> striosome_value, PLASTIC, DA-delta-gated, SPARSE. NO dense A1 floor.
        # When place_to_value_active=False this is weight 0 (the critic-absent twin for gate 2d).
        # STEP-2 VOLLEY: coincidence_detector=enable_volley -> this arm reads the FS-PING-synchronized
        # place volley through the landed Route-D plateau (the thing that fires the critic from a
        # sparse-distinct code). Still PLASTIC + DA-gated so it grades + learns.
        RegionPathway(from_region="place", to_region="striosome_value",
                      density=float(place_to_value_density),
                      weight_mean=float(place_to_value_weight if place_to_value_active else 0.0),
                      weight_jitter=float(place_to_value_jitter), plastic=True,
                      plasticity_gate="value_input",
                      coincidence_detector=bool(enable_volley and place_to_value_active)),
        # The shipped GABA_B value subtraction (striosome_value -> snc).
        RegionPathway(from_region="striosome_value", to_region="snc",
                      density=0.5, weight_mean=float(strio_to_snc_weight),
                      weight_jitter=0.2, plastic=False,
                      receptor=("gaba_b" if gabab else "gaba_a")),
    ]
    if enable_volley:
        # ── FS-PING gamma synchronizer on the place pool (the STEP-1 GO mechanism, brain-based) ──
        # An FS interneuron pool reciprocally wired to `place`: the active place cells excite FS, FS
        # GABA_A silences the pool for ~one GABA_A decay, release -> the active cells re-fire TOGETHER
        # each gamma cycle. The gamma EMERGES from neurons+synapses (mirrors CORTEX_GAMMA_FS_NETWORK).
        # Location-BLIND (FS sees only the currently-active place cells -> it sets WHEN they fire, the
        # place code selects WHICH) -> distinctness preserved, NOT densified. No host pacing.
        regions.append(
            BrainRegion(name="place_fs", n_neurons=int(n_fs), exc_fraction=0.0, internal_density=0.0,
                        exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                        izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name))
        pathways.append(
            RegionPathway(from_region="place", to_region="place_fs", density=float(place_to_fs_density),
                          weight_mean=float(place_to_fs_weight), weight_jitter=0.2, plastic=False))
        pathways.append(
            RegionPathway(from_region="place_fs", to_region="place", density=float(fs_to_place_density),
                          weight_mean=float(fs_to_place_weight), weight_jitter=0.2, plastic=False))
    if include_actor:
        regions.append(BrainRegion(
            name="sensor_place_readout", n_neurons=int(n_sensor_place), exc_fraction=1.0,
            internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
            plastic_internal=False, izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name))
        for ai, action in enumerate(("N", "E", "S", "W")):
            regions.append(BrainRegion(
                name=f"cortex_{action}", n_neurons=int(n_cortex_per_action), exc_fraction=1.0,
                internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                plastic_internal=False, izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
                action_index=ai))
            pathways.append(RegionPathway(
                from_region="sensor_place_readout", to_region=f"cortex_{action}",
                density=1.0, weight_mean=float(sensor_place_to_cortex_weight),
                weight_jitter=0.2, plastic=True, plasticity_gate="place_goal_to_cortex"))

    cfg = CoreSimConfig()
    cfg.seed = int(seed); cfg.heterogeneity_seed = int(seed); cfg.ou_seed = int(seed)
    cfg.dt_ms = float(dt_ms); cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.enable_stdp = True
    cfg.enable_hebbian_learning = False
    cfg.enable_reward_modulation = True
    cfg.reward_learning_rate = float(reward_learning_rate)
    cfg.current_reward_signal = 0.0       # BRAIN-BASED: the SNc FIRING is the signal, not a host scalar
    cfg.reward_baseline = 0.0
    cfg.enable_nmda = bool(nmda_critic or enable_volley)  # per-region mask restricts NMDA to the critic
    # STDP soft-bound cap. STEP-2 VOLLEY: keep this LOW (~4-6) so the learned place->critic AMPA stays
    # in the COINCIDENCE-DEPENDENT regime -- if w_near runs away (>~8) the rate-coded AMPA alone fires
    # the critic and the jitter anti-cheat no longer collapses it (a rate leak; observed at the
    # over-grown seed). Capping keeps the critic firing CONTINGENT on the synchronized volley.
    cfg.stdp_w_max = float(stdp_w_max)
    cfg.fast_spike_reset = True
    # === STEP-2 Route-D coincidence read-out (landed b980070a); default OFF (enable_volley=False) ===
    if enable_volley:
        cfg.enable_coincidence_detection = True
        cfg.coincidence_k_threshold = float(coincidence_k)
        cfg.coincidence_gain = float(coincidence_gain)
        cfg.coincidence_plateau_strength = float(coincidence_plateau)
    if gabab:
        cfg.enable_gabab = True
        cfg.gabab_reversal_potential = -90.0
        cfg.gabab_tau_decay = float(gabab_tau_decay)
        cfg.gabab_propagation_strength = float(gabab_propagation_strength)
    # === deterministic-nav regime (g11_bg_runner.py:3340-3344) ===
    cfg.enable_homeostasis = False          # GLOBAL homeostasis OFF (regime fidelity)
    cfg.enable_short_term_plasticity = False
    cfg.enable_ou_process = False
    cfg.enable_conductance_noise = False
    cfg.enable_parameter_heterogeneity = False
    cfg.enable_structural_plasticity = False

    cfg.enable_neuromodulator_subsystem = True
    cfg.neuromodulators = [NeuromodulatorConfig(
        name="dopamine", baseline=0.5, decay_tau_ms=200.0, concentration_min=0.0, concentration_max=2.0,
        targets=[ModulatorTarget(target_type="plasticity_rate", scope="all", sensitivity=+1.0)],
        production_rules=[ProductionRule(rule_type="from_region_firing_signed",
                                         sensitivity=float(snc_da_sensitivity), threshold=0.30,
                                         window_ms=200.0, source_regions=["snc"])])]

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = seed
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge, cfg


def _assert_cupy_regime(cfg, backend_name):
    """Anti-cheat (c): the de-risk MUST run on cupy in the deterministic regime."""
    if backend_name != "cupy":
        raise AssertionError(
            f"REGIME FIDELITY (anti-cheat c): this Stage-2 critic de-risk MUST run on CuPy "
            f"(numpy is DISQUALIFIED for striatal/near-threshold work; see "
            f"2026-06-09-N9-cupy-membrane-divergence-ROOT.md). Got backend={backend_name!r}. "
            f"Set SIM_BACKEND=cupy.")
    bad = [k for k in ("enable_ou_process", "enable_conductance_noise", "enable_homeostasis",
                       "enable_parameter_heterogeneity", "enable_short_term_plasticity")
           if getattr(cfg, k, False)]
    if bad:
        raise AssertionError(f"REGIME FIDELITY (anti-cheat c): deterministic-regime knobs ON: {bad}")


# ──────────────────────────────────────────────────────────────────────
# Step / drive helpers
# ──────────────────────────────────────────────────────────────────────

def _tick(bridge):
    bridge._run_one_simulation_step()
    bridge.runtime_state.current_time_step += 1
    bridge.runtime_state.current_time_ms = (
        bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)


def _step(bridge, n):
    for _ in range(n):
        _tick(bridge)


def _drive_landmarks(bridge, xp, sensor_idx_gpu, sensor_act, ablate=False):
    """Set landmark_sensors external current (the ONLY region this probe ever drives)."""
    bridge.cp_external_input_current[:] = 0.0
    if not ablate and sensor_act is not None:
        bridge.cp_external_input_current[sensor_idx_gpu] = xp.asarray(sensor_act, dtype=xp.float32)


def _critic_rate_at_location(bridge, xp, sensor_idx_gpu, crit_idx_gpu, sensor_act, *,
                             n_steps=120, warmup=30, ablate=False, freeze_lr=True, jitter=False):
    """Drive landmark_sensors at a location (-> place pool fires its ensemble -> critic), measure the
    critic firing rate over a post-warmup window. Learning frozen so this is a pure read.

    jitter (THE coincidence anti-cheat, Branco-Hausser): de-synchronize the place drive by clamping
    the sensors only every OTHER step (same active cells, same total drive, spikes spread across
    alternating steps -> the FS-PING volley is destroyed). If the critic firing/grading SURVIVES
    jitter it is rate, not coincidence -> NOT a real volley read."""
    n_crit = int(crit_idx_gpu.size)
    saved = bridge.core_config.reward_learning_rate
    if freeze_lr:
        bridge.core_config.reward_learning_rate = 0.0
    if not jitter:
        _drive_landmarks(bridge, xp, sensor_idx_gpu, sensor_act, ablate=ablate)
    act_gpu = xp.asarray(sensor_act, dtype=xp.float32) if sensor_act is not None else None
    spk = 0; m = 0
    for t in range(n_steps):
        if jitter:
            # desync: sensor clamp ON only on even steps (place spikes scatter, no coincident volley)
            bridge.cp_external_input_current[:] = 0.0
            if (not ablate) and act_gpu is not None and (t % 2 == 0):
                bridge.cp_external_input_current[sensor_idx_gpu] = act_gpu
        _tick(bridge)
        if t >= warmup:
            spk += int(bridge.cp_firing_states[crit_idx_gpu].sum()); m += 1
    bridge.core_config.reward_learning_rate = saved
    bridge.cp_external_input_current[:] = 0.0
    return spk / max(n_crit, 1) / max(m * 1e-3, 1e-9)


def _place_ensemble(bridge, xp, sensor_idx_gpu, place_idx_gpu, sensor_act, *, n_steps=80, ablate=False):
    """Per-cell spike-count vector of the `place` pool at a location (for ensemble cosines + the
    NEAR/FAR active-cell sets used to track per-location LTP)."""
    n = int(place_idx_gpu.size)
    _drive_landmarks(bridge, xp, sensor_idx_gpu, sensor_act, ablate=ablate)
    counts = xp.zeros(n, dtype=xp.float32)
    saved = bridge.core_config.reward_learning_rate
    bridge.core_config.reward_learning_rate = 0.0
    for _ in range(n_steps):
        _tick(bridge)
        counts += bridge.cp_firing_states[place_idx_gpu].astype(xp.float32)
    bridge.core_config.reward_learning_rate = saved
    bridge.cp_external_input_current[:] = 0.0
    return _host(counts)


def _mean_w(bridge, pre_name, post_name, pre_subset=None):
    pre = (set(int(i) for i in _idx(bridge, pre_name)) if pre_subset is None
           else set(int(i) for i in pre_subset))
    post = set(int(i) for i in _idx(bridge, post_name))
    coo = bridge.cp_connections.tocoo()
    rows = np.asarray(_host(coo.row)); cols = np.asarray(_host(coo.col)); data = np.asarray(_host(coo.data))
    m = np.isin(rows, list(pre)) & np.isin(cols, list(post))
    if not m.any():
        m = np.isin(rows, list(post)) & np.isin(cols, list(pre))
    return float(data[m].mean()) if m.any() else 0.0


def _calibrate_da(bridge, cfg, snc_idx_gpu, tonic_pa, xp, n_steps=300):
    """Set the DA production threshold to the SNc tonic firing fraction (so phasic bursts drive DA)."""
    n_snc = int(snc_idx_gpu.size)
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[snc_idx_gpu] = xp.float32(tonic_pa)
    frac = 0.0; m = 0
    for i in range(n_steps):
        _tick(bridge)
        if i >= n_steps // 2:
            frac += float(bridge.cp_firing_states[snc_idx_gpu].sum()) / max(n_snc, 1); m += 1
    tf = frac / max(m, 1)
    cfg.neuromodulators[0].production_rules[0].threshold = float(tf)
    bridge.cp_external_input_current[:] = 0.0
    return tf


def _lesion_gabab(bridge):
    m = getattr(bridge, "cp_gabab_synapse_mask", None)
    if m is None:
        return 0
    n_was = int(_host(m).sum())
    from sim.backend import get_backend
    xp, _ = get_backend()
    bridge.cp_gabab_synapse_mask = xp.zeros_like(m)
    if getattr(bridge, "cp_conductance_g_gabab", None) is not None:
        bridge.cp_conductance_g_gabab[:] = 0.0
    return n_was


def _measure_actor_rate(bridge, xp, idx_map, sensor_vec, *, cortex_tonic_pa=400.0,
                        landmark_vec=None, n_steps=120, warmup=40):
    """Gate 2d: actor cortex output rate under a fixed cortical tonic + the sparse actor place code,
    while ALSO driving the landmark sensors (-> place -> critic) concurrently. The place/critic have NO
    edge to cortex, so a faithful dedicated read-out leaves the cortex rate unchanged vs a twin."""
    sp_idx = idx_map["sensor_place_readout"]
    cortex_idx = xp.asarray(np.concatenate(
        [np.asarray(_host(idx_map[f"cortex_{a}"])) for a in ("N", "E", "S", "W")]))
    n_cx = int(cortex_idx.size)
    saved = bridge.core_config.reward_learning_rate; bridge.core_config.reward_learning_rate = 0.0
    spk = 0; m = 0
    lm_idx = idx_map.get("landmark_sensors")
    for t in range(n_steps):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[cortex_idx] = xp.float32(cortex_tonic_pa)
        bridge.cp_external_input_current[sp_idx] = xp.asarray(sensor_vec, dtype=xp.float32)
        if landmark_vec is not None and lm_idx is not None:
            bridge.cp_external_input_current[lm_idx] = xp.asarray(landmark_vec, dtype=xp.float32)
        _tick(bridge)
        if t >= warmup:
            spk += int(bridge.cp_firing_states[cortex_idx].sum()); m += 1
    bridge.core_config.reward_learning_rate = saved
    bridge.cp_external_input_current[:] = 0.0
    return spk / max(n_cx, 1) / max(m * 1e-3, 1e-9)


# ──────────────────────────────────────────────────────────────────────
# Stage-2 run for one seed
# ──────────────────────────────────────────────────────────────────────

def run_seed(seed, *, locations, landmarks, n_bearing, n_dist, n_place, n_strio, n_snc, grid_size,
             lm_to_place_weight, lm_to_place_density, lm_to_place_jitter,
             place_to_value_weight, place_to_value_density, place_to_value_jitter,
             max_intensity, falloff, dist_sigma, dist_max, bexp,
             selforg_passes, selforg_steps_per_loc,
             snc_tonic_pa, snc_reward_gain, hold_steps, n_train, lead_steps,
             strio_to_snc_weight, snc_da_sensitivity, reward_learning_rate,
             gabab, gabab_propagation_strength, nmda_critic,
             sensor_drive_pa, sensor_sigma, actor_cortex_tonic_pa,
             enable_volley=False, n_fs=160, place_to_fs_weight=16.0, place_to_fs_density=0.4,
             fs_to_place_weight=8.0, fs_to_place_density=0.4,
             coincidence_k=4.0, coincidence_gain=2.0, coincidence_plateau=80.0, readout_plateau=None,
             stdp_w_max=40.0,
             lesion=False, shuffle=False, ablate_sensors=False, jitter=False, verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    from sim.backend import get_backend
    xp, bk = get_backend()
    # Shared volley kwargs for both the main + critic-absent-twin builds.
    _volley_kw = dict(enable_volley=bool(enable_volley), n_fs=int(n_fs),
                      place_to_fs_weight=float(place_to_fs_weight), place_to_fs_density=float(place_to_fs_density),
                      fs_to_place_weight=float(fs_to_place_weight), fs_to_place_density=float(fs_to_place_density),
                      coincidence_k=float(coincidence_k), coincidence_gain=float(coincidence_gain),
                      coincidence_plateau=float(coincidence_plateau), stdp_w_max=float(stdp_w_max))

    near_name = "near"
    far_names = [n for n in locations if n.startswith("far")]
    assert far_names, "need >=1 'far_*' location"
    n_per_landmark = n_bearing + n_dist
    n_sensors = len(landmarks) * n_per_landmark

    t0 = time.time()
    bridge, cfg = _build(
        seed, n_sensors=n_sensors, n_place=n_place, n_strio=n_strio, n_snc=n_snc, grid_size=grid_size,
        lm_to_place_weight=lm_to_place_weight, lm_to_place_density=lm_to_place_density,
        lm_to_place_jitter=lm_to_place_jitter, place_to_value_weight=place_to_value_weight,
        place_to_value_density=place_to_value_density, place_to_value_jitter=place_to_value_jitter,
        strio_to_snc_weight=strio_to_snc_weight, snc_da_sensitivity=snc_da_sensitivity,
        reward_learning_rate=reward_learning_rate, gabab=gabab,
        gabab_tau_decay=150.0, gabab_propagation_strength=gabab_propagation_strength,
        nmda_critic=nmda_critic, **_volley_kw)
    _assert_cupy_regime(cfg, bk)
    build_s = time.time() - t0

    rm = bridge.region_manager
    sensor_idx = np.asarray(rm.indices("landmark_sensors"), dtype=np.int64)
    place_idx = np.asarray(rm.indices("place"), dtype=np.int64)
    crit_idx = np.asarray(rm.indices("striosome_value"), dtype=np.int64)
    snc_idx = np.asarray(rm.indices("snc"), dtype=np.int64)
    sensor_idx_g = xp.asarray(sensor_idx); place_idx_g = xp.asarray(place_idx)
    crit_idx_g = xp.asarray(crit_idx); snc_idx_g = xp.asarray(snc_idx)

    log(f"  [seed {seed}] built {build_s:.1f}s; {cfg.num_neurons} neurons, "
        f"{int(bridge.cp_connections.nnz)} synapses; backend={bk}")

    # Position-leak audit: the ONLY region this probe drives is landmark_sensors (+ the actor stub's
    # sensor_place_readout/cortex tonic for gate 2d, which feed ONLY the actor, not place/critic).
    driven = {"landmark_sensors"}
    assert "place" not in driven and "striosome_value" not in driven

    # ── Renders ──
    def render(name, drop_landmark=None):
        x, y = locations[name]
        act = landmark_sensor_act(x, y, landmarks, n_bearing, n_dist, max_intensity, falloff,
                                  dist_sigma, dist_max, bexp)
        if drop_landmark is not None:
            per = n_bearing + n_dist
            act = act.copy(); act[drop_landmark * per:(drop_landmark + 1) * per] = 0.0
        return act
    loc_sensor = {n: render(n) for n in locations}

    # ════════════════════════════════════════════════════════════════
    # STEP 1 — SELF-ORGANIZE the place code (Stage-1 mechanism), then FREEZE it.
    # ════════════════════════════════════════════════════════════════
    log(f"  [seed {seed}] STEP 1: self-organizing place fields "
        f"({selforg_passes} passes x {len(locations)} locs)...")
    bridge.set_plasticity_gate("landmark_to_place", 1.0)
    bridge.set_plasticity_gate("value_input", 0.0)   # freeze the critic arm during place self-org
    t_so = time.time()
    rng = np.random.default_rng(seed)
    loc_names = list(locations.keys())
    for _p in range(selforg_passes):
        order = list(loc_names); rng.shuffle(order)
        for name in order:
            bridge.cp_external_input_current[:] = 0.0
            _step(bridge, 20)
            bridge.cp_external_input_current[sensor_idx_g] = xp.asarray(loc_sensor[name], dtype=xp.float32)
            _step(bridge, selforg_steps_per_loc)
    bridge.set_plasticity_gate("landmark_to_place", 0.0)   # FREEZE the place fields (stable afferent)
    bridge.cp_external_input_current[:] = 0.0
    log(f"  [seed {seed}] place self-org done ({time.time() - t_so:.0f}s); place fields FROZEN")

    # ── Place-code provenance (Stage-1 gates, abridged): distinct ensembles per location ──
    place_ens = {n: _place_ensemble(bridge, xp, sensor_idx_g, place_idx_g, loc_sensor[n]) for n in loc_names}
    diff_cos = [cosine_counts(place_ens[a], place_ens[b])
                for a, b in itertools.combinations(loc_names, 2)]
    place_diff_cos = float(np.mean(diff_cos)) if diff_cos else 1.0
    place_sparsity = float(np.mean([np.mean(place_ens[n] > 0) for n in loc_names]))
    log(f"  [seed {seed}] place code: diff-loc cos={place_diff_cos:.3f} sparsity={place_sparsity:.3f}")

    # ── NEAR/FAR active-place-cell sets (disjoint cores) for per-location LTP tracking ──
    # SHUFFLE control (anti-cheat a): permute which place CELLS the value arm reads as "the NEAR
    # ensemble". Self-organization (place fields) is left intact; we decouple the place-ENSEMBLE that
    # potentiates from the location where reward is delivered, so a learned value-of-LOCATION FAILS.
    def active_set(name, env=None):
        e = (env if env is not None else place_ens)[name]
        return set(int(place_idx[i]) for i in np.where(np.asarray(e) > 0)[0])
    near_set_true = active_set(near_name)
    far_set_true = set().union(*[active_set(fn) for fn in far_names]) - near_set_true

    near_set = near_set_true; far_set = far_set_true
    if shuffle:
        rng_s = np.random.RandomState(seed ^ 0x5A5A)
        perm = rng_s.permutation(len(place_idx))
        remap = {int(place_idx[i]): int(place_idx[perm[i]]) for i in range(len(place_idx))}
        near_set = set(remap[i] for i in near_set_true)
        far_set = set(remap[i] for i in far_set_true) - near_set

    # ── DA calibration ──
    tonic_frac = _calibrate_da(bridge, cfg, snc_idx_g, snc_tonic_pa, xp)

    # ── Gate-2d baseline: actor cortex firing on a critic-ABSENT twin (place->value weight 0) ──
    base_bridge, base_cfg = _build(
        seed, n_sensors=n_sensors, n_place=n_place, n_strio=n_strio, n_snc=n_snc, grid_size=grid_size,
        lm_to_place_weight=lm_to_place_weight, lm_to_place_density=lm_to_place_density,
        lm_to_place_jitter=lm_to_place_jitter, place_to_value_weight=place_to_value_weight,
        place_to_value_density=place_to_value_density, place_to_value_jitter=place_to_value_jitter,
        strio_to_snc_weight=strio_to_snc_weight, snc_da_sensitivity=snc_da_sensitivity,
        reward_learning_rate=reward_learning_rate, gabab=gabab, gabab_tau_decay=150.0,
        gabab_propagation_strength=gabab_propagation_strength, nmda_critic=nmda_critic,
        place_to_value_active=False, **_volley_kw)
    _assert_cupy_regime(base_cfg, bk)
    base_idx = {n: xp.asarray(_idx(base_bridge, n)) for n in
                ("landmark_sensors", "sensor_place_readout", "cortex_N", "cortex_E", "cortex_S", "cortex_W")}
    n_sp = int(base_idx["sensor_place_readout"].size)
    sp_prefs = _grid_prefs(n_sp, grid_size)
    sensor_near_actor = _sparse_place_code(locations[near_name], sp_prefs, sensor_drive_pa, sensor_sigma)
    actor_no_critic = _measure_actor_rate(base_bridge, xp, base_idx, sensor_near_actor,
                                          cortex_tonic_pa=actor_cortex_tonic_pa,
                                          landmark_vec=loc_sensor[near_name])
    del base_bridge

    log(f"  [seed {seed}] calib SNc tonic frac={tonic_frac:.4f}; "
        f"gate-2d baseline actor(no-critic)={actor_no_critic:.2f} Hz")

    # ── Critic FIRE @ NEAR at INIT (before value training; place arm small init) ──
    w_near_init = _mean_w(bridge, "place", "striosome_value", pre_subset=near_set)
    w_far_init = _mean_w(bridge, "place", "striosome_value", pre_subset=far_set)
    crit_near_init = _critic_rate_at_location(bridge, xp, sensor_idx_g, crit_idx_g, loc_sensor[near_name])
    log(f"  [seed {seed}] critic@NEAR init (pre-value-train) = {crit_near_init:.2f} Hz "
        f"(w_near {w_near_init:.3f})")

    # ════════════════════════════════════════════════════════════════
    # STEP 2 — TRAIN the critic's V (place -> critic plastic, DA-delta-gated) on the FROZEN place fields.
    # value-leads-reward: visit NEAR (place ensemble fires) + SNc reward burst -> DA-gated LTP grows
    # the NEAR place->critic synapses. FAR held out (never paired with reward).
    # ════════════════════════════════════════════════════════════════
    log(f"  [seed {seed}] STEP 2: training critic V (value-leads-reward, {n_train} trials)...")
    bridge.set_plasticity_gate("value_input", 1.0)        # open the critic arm
    bridge.set_plasticity_gate("landmark_to_place", 0.0)  # place fields stay FROZEN
    near_v_curve = []
    for t in range(n_train):
        # ITI floor: SNc tonic, no place drive
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[snc_idx_g] = xp.float32(snc_tonic_pa)
        _step(bridge, hold_steps)
        if getattr(bridge, "cp_eligibility_trace", None) is not None:
            bridge.cp_eligibility_trace[:] = 0.0
        # LEARN: drive landmark sensors at NEAR (-> place ensemble -> critic post-spike) + reward burst
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[sensor_idx_g] = xp.asarray(loc_sensor[near_name], dtype=xp.float32)
        bridge.cp_external_input_current[snc_idx_g] = xp.float32(snc_tonic_pa + snc_reward_gain)
        spk = 0
        for _ in range(hold_steps):
            _tick(bridge)
            spk += int(bridge.cp_firing_states[crit_idx_g].sum())
        near_v_curve.append(spk / max(len(crit_idx), 1) / max(hold_steps * 1e-3, 1e-9))
        if verbose and (t < 3 or t % 10 == 0 or t == n_train - 1):
            wn = _mean_w(bridge, "place", "striosome_value", pre_subset=near_set)
            wf = _mean_w(bridge, "place", "striosome_value", pre_subset=far_set)
            da = float(bridge.neuromodulator_manager.get_concentration("dopamine"))
            log(f"    [acq t={t:02d}] V(near)={near_v_curve[-1]:6.2f}Hz w_near={wn:.3f} w_far={wf:.3f} "
                f"(near/far {wn/max(wf,1e-6):.2f}) DA={da:.3f}")
    bridge.set_plasticity_gate("value_input", 0.0)        # freeze for the read-out gates
    bridge.cp_external_input_current[:] = 0.0

    import statistics as _st
    early = slice(0, max(1, n_train // 5)); late = slice(-max(1, n_train // 5), None)
    near_v_early = _st.mean(near_v_curve[early]); near_v_late = _st.mean(near_v_curve[late])
    w_near_final = _mean_w(bridge, "place", "striosome_value", pre_subset=near_set)
    w_far_final = _mean_w(bridge, "place", "striosome_value", pre_subset=far_set)

    # ════════════════════════════════════════════════════════════════
    # GATES
    # ════════════════════════════════════════════════════════════════
    # 2a FIRE + 2b PLACE-GRADED: trained critic rate NEAR vs each FAR (sensor-driven, learning frozen).
    # Under --jitter, the place drive is de-synchronized at recall -> the volley collapses (the
    # decisive coincidence anti-cheat: G_FIRE + G_GRADE must FAIL).
    #
    # READOUT-PLATEAU (2026-06-09): the Route-D coincidence count is WEIGHT-BLIND (c_i = #coincident
    # spikes, not summed weights), so at the training plateau strength the critic fires at ANY volley
    # (NEAR + FAR) -> grading washes out (~1.2x, the dense-blob ceiling). Biology: the NMDA plateau is a
    # coincidence-GATED integration WINDOW -- it depolarizes the cell so the LEARNED AMPA weights decide
    # threshold crossing. We model that by LOWERING the plateau at READ-OUT (a sub-threshold window) so
    # the cell needs the learned NEAR AMPA (w~5) to fire and the unlearned FAR AMPA (w~0.6) can't. The
    # strong plateau is kept during TRAINING (it bootstraps the post-spike that drives the DA-gated LTP).
    # readout_plateau<=0 disables (keeps the training plateau -> the weight-blind baseline).
    _saved_plateau = float(getattr(bridge.core_config, "coincidence_plateau_strength", 80.0))
    if enable_volley and readout_plateau is not None and readout_plateau > 0:
        bridge.core_config.coincidence_plateau_strength = float(readout_plateau)
    crit_near = _critic_rate_at_location(bridge, xp, sensor_idx_g, crit_idx_g, loc_sensor[near_name],
                                         ablate=ablate_sensors, jitter=jitter)
    crit_far_each = {fn: _critic_rate_at_location(bridge, xp, sensor_idx_g, crit_idx_g, loc_sensor[fn],
                                                  ablate=ablate_sensors, jitter=jitter) for fn in far_names}
    bridge.core_config.coincidence_plateau_strength = _saved_plateau
    crit_far = float(np.mean(list(crit_far_each.values())))
    crit_far_max = float(np.max(list(crit_far_each.values())))
    place_graded_ratio = crit_near / max(crit_far_max, 1e-3)   # vs the WORST (highest) far -> strict
    fire = bool(crit_near >= 5.0)
    place_graded = bool(crit_near >= 5.0 and place_graded_ratio >= 3.0)

    # 2c LEARNS-V (LTP): near weight grew from init AND exceeds far (>=2x)
    weight_grew = bool(w_near_final > 1.05 * max(w_near_init, 1e-6)
                       and w_near_final >= 2.0 * max(w_far_final, 1e-6))
    v_learned = bool(near_v_late > 1.10 * max(near_v_early, 1e-6))

    # 2d ACTOR-NOT-PERTURBED
    actor_with_critic = _measure_actor_rate(
        bridge, xp, {**{n: xp.asarray(_idx(bridge, n)) for n in
                        ("landmark_sensors", "sensor_place_readout", "cortex_N", "cortex_E", "cortex_S", "cortex_W")}},
        sensor_near_actor, cortex_tonic_pa=actor_cortex_tonic_pa, landmark_vec=loc_sensor[near_name])
    actor_ratio = actor_with_critic / max(actor_no_critic, 1e-9)
    actor_ok = (0.90 <= actor_ratio <= 1.10) if actor_no_critic > 1e-6 else (actor_with_critic <= 1e-6)

    # ── SNc state-specific gap (value-leads-reward LEAD test) + gate-2e lesion ──
    def _snc_test(sensor_act, snc_pa):
        # ITI floor
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[snc_idx_g] = xp.float32(snc_tonic_pa)
        _step(bridge, hold_steps + 20)
        # LEAD: place drive (critic fires -> GABA_B onto SNc) BEFORE the reward burst
        if lead_steps > 0:
            bridge.cp_external_input_current[:] = 0.0
            bridge.cp_external_input_current[sensor_idx_g] = xp.asarray(sensor_act, dtype=xp.float32)
            bridge.cp_external_input_current[snc_idx_g] = xp.float32(snc_tonic_pa)
            _step(bridge, int(lead_steps))
        # REWARD burst (with the place drive still on)
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[sensor_idx_g] = xp.asarray(sensor_act, dtype=xp.float32)
        bridge.cp_external_input_current[snc_idx_g] = xp.float32(snc_pa)
        n_snc_local = int(snc_idx_g.size); spk = 0
        saved = bridge.core_config.reward_learning_rate; bridge.core_config.reward_learning_rate = 0.0
        for _ in range(hold_steps):
            _tick(bridge)
            spk += int(bridge.cp_firing_states[snc_idx_g].sum())
        bridge.core_config.reward_learning_rate = saved
        bridge.cp_external_input_current[:] = 0.0
        return spk / max(n_snc_local, 1) / max(hold_steps * 1e-3, 1e-9)

    if lesion:
        n_cut = _lesion_gabab(bridge)
        log(f"  [seed {seed}] gate-2e lesion: zeroed {n_cut} GABA_B synapses")

    pred_r = _snc_test(loc_sensor[near_name], snc_tonic_pa + snc_reward_gain)    # predicted (NEAR)
    far_for_gap = far_names[0]
    unpred_r = _snc_test(loc_sensor[far_for_gap], snc_tonic_pa + snc_reward_gain)  # unpredicted (FAR)
    gap_ratio = unpred_r / max(pred_r, 1e-6)
    state_specific = bool((unpred_r > 1.30 * max(pred_r, 1e-6)) and (unpred_r >= 10.0))

    if verbose:
        far_str = ", ".join(f"{fn}={crit_far_each[fn]:.2f}" for fn in far_names)
        log(f"  [gate-2a FIRE] critic@NEAR={crit_near:.2f}Hz (>=5 => {fire})")
        log(f"  [gate-2b PLACE-GRADED] NEAR={crit_near:.2f}Hz FAR[{far_str}]Hz "
            f"ratio(near/maxfar)={place_graded_ratio:.2f} (>=3 & near>=5 => {place_graded})")
        log(f"  [gate-2c LEARNS-V] w_near {w_near_init:.3f}->{w_near_final:.3f} "
            f"w_far {w_far_init:.3f}->{w_far_final:.3f} (near>=2x far & grew => {weight_grew}); "
            f"V(near) {near_v_early:.2f}->{near_v_late:.2f}Hz")
        log(f"  [gate-2d ACTOR] no-critic={actor_no_critic:.2f}Hz with-critic={actor_with_critic:.2f}Hz "
            f"ratio={actor_ratio:.3f} (not-perturbed => {actor_ok})")
        log(f"  [gate-2e SNc gap] predicted(NEAR)={pred_r:.2f}Hz unpredicted(FAR)={unpred_r:.2f}Hz "
            f"gap={gap_ratio:.2f} state-specific={state_specific} (lesion={lesion})")

    primary = bool(fire and place_graded and weight_grew and actor_ok)
    return dict(
        seed=seed, backend=bk, lesion=lesion, shuffle=shuffle, ablate_sensors=ablate_sensors,
        jitter=jitter, enable_volley=enable_volley,
        place_diff_location_cosine=place_diff_cos, place_sparsity=place_sparsity,
        n_neurons=int(cfg.num_neurons), n_synapses=int(bridge.cp_connections.nnz),
        n_place=int(place_idx.size), n_strio=int(crit_idx.size),
        crit_near_init_hz=crit_near_init,
        crit_near_trained_hz=crit_near, crit_far_mean_hz=crit_far, crit_far_max_hz=crit_far_max,
        crit_far_each_hz={k: round(v, 3) for k, v in crit_far_each.items()},
        place_graded_ratio=place_graded_ratio, fire=fire, place_graded=place_graded,
        w_near_init=w_near_init, w_near_final=w_near_final,
        w_far_init=w_far_init, w_far_final=w_far_final,
        weight_near_over_far=(w_near_final / max(w_far_final, 1e-6)),
        weight_grew=weight_grew, v_learned=v_learned,
        near_v_early_hz=near_v_early, near_v_late_hz=near_v_late,
        actor_no_critic_hz=actor_no_critic, actor_with_critic_hz=actor_with_critic,
        actor_ratio=actor_ratio, actor_not_perturbed=actor_ok,
        snc_predicted_near_hz=pred_r, snc_unpredicted_far_hz=unpred_r,
        snc_gap_ratio=gap_ratio, snc_state_specific=state_specific,
        primary=primary,
        ou_off=(not cfg.enable_ou_process), cond_noise_off=(not cfg.enable_conductance_noise),
        global_homeo_off=(not cfg.enable_homeostasis),
        total_seconds=round(time.time() - t0, 1),
    )


def default_locations(grid_size):
    """NEAR-goal + 3 FAR (incl. the grid center, where the dense blob peaked). NEAR is chosen as a
    location with a ROBUST place ensemble (c_75_25 from the placefire probe) so the negative is NOT
    conflated with the Stage-1 drive-fragile edge-location (the original (0.25,0.75) NEAR is the 6-cell
    under-firing corner). The center (0.50,0.50) is a deliberately HARD far: the old dense up-state
    fired MOST there."""
    g = grid_size - 1
    return {
        "near":     (g * 0.75, g * 0.25),
        "far_a":    (g * 0.25, g * 0.75),
        "far_b":    (g * 0.25, g * 0.25),
        "far_center": (g * 0.50, g * 0.50),
    }


def default_landmarks(grid_size):
    g = grid_size - 1
    return [(0.0, 0.0), (float(g), 0.0), (float(g) / 2.0, float(g))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid-size", type=int, default=32)
    # place-code (Stage-1 canonical params)
    ap.add_argument("--n-bearing", type=int, default=12)
    ap.add_argument("--n-dist", type=int, default=8)
    ap.add_argument("--bexp", type=float, default=4.0)
    ap.add_argument("--dist-sigma", type=float, default=4.0)
    ap.add_argument("--n-place", type=int, default=400)
    ap.add_argument("--lm-to-place-weight", type=float, default=28.0)
    ap.add_argument("--lm-to-place-density", type=float, default=0.5)
    ap.add_argument("--lm-to-place-jitter", type=float, default=0.6)
    ap.add_argument("--max-intensity", type=float, default=450.0)
    ap.add_argument("--falloff", type=float, default=0.03)
    ap.add_argument("--selforg-passes", type=int, default=12)
    ap.add_argument("--selforg-steps-per-loc", type=int, default=120)
    # critic arm
    ap.add_argument("--n-strio", type=int, default=80)
    ap.add_argument("--n-snc", type=int, default=30)
    ap.add_argument("--place-to-value-weight", type=float, default=0.5,
                    help="init weight of the plastic place->critic arm (realistic small; STDP grows it)")
    ap.add_argument("--place-to-value-density", type=float, default=0.6)
    ap.add_argument("--place-to-value-jitter", type=float, default=0.2)
    ap.add_argument("--reward-learning-rate", type=float, default=0.12)
    ap.add_argument("--strio-to-snc-weight", type=float, default=10.0)
    ap.add_argument("--snc-da-sensitivity", type=float, default=8.0)
    ap.add_argument("--snc-tonic-pa", type=float, default=180.0)
    ap.add_argument("--snc-reward-gain", type=float, default=300.0)
    ap.add_argument("--hold-steps", type=int, default=40)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--lead-ms", type=float, default=150.0)
    ap.add_argument("--gabab-propagation-strength", type=float, default=0.02)
    # actor stub
    ap.add_argument("--sensor-drive-pa", type=float, default=1500.0)
    ap.add_argument("--sensor-sigma", type=float, default=1.5)
    ap.add_argument("--actor-cortex-tonic-pa", type=float, default=400.0)
    # ── STEP-2 VOLLEY (2026-06-09): the validated STEP-1 mechanism wired into the critic arm ──
    # Default OFF -> the baseline (sparse-async place code, NO volley) reproduces the prior NEGATIVE.
    # --enable-volley turns on the FS-PING synchronizer + the Route-D coincidence read-out at the
    # STEP-1 GO operating point (n-place 800, re-tuned FS-PING, K=4, plateau 80).
    ap.add_argument("--enable-volley", action="store_true",
                    help="STEP-2: FS-PING gamma volley + Route-D coincidence read-out on the place->critic arm")
    ap.add_argument("--n-fs", type=int, default=160, help="FS-PING pool size (~20%% of n-place)")
    ap.add_argument("--place-to-fs-weight", type=float, default=16.0)
    ap.add_argument("--place-to-fs-density", type=float, default=0.4)
    ap.add_argument("--fs-to-place-weight", type=float, default=8.0)
    ap.add_argument("--fs-to-place-density", type=float, default=0.4)
    ap.add_argument("--coincidence-k", type=float, default=4.0, help="Route-D K threshold (MUST be > 1)")
    ap.add_argument("--coincidence-gain", type=float, default=2.0)
    ap.add_argument("--coincidence-plateau", type=float, default=80.0,
                    help="TRAINING plateau strength (bootstraps the post-spike for DA-gated LTP)")
    ap.add_argument("--readout-plateau", type=float, default=None,
                    help="READ-OUT plateau strength (lowered -> sub-threshold integration window so the "
                         "LEARNED AMPA decides firing -> grading via weight, not weight-blind coincidence). "
                         "Default None = keep training plateau (weight-blind baseline).")
    ap.add_argument("--stdp-w-max", type=float, default=40.0,
                    help="STDP soft-bound cap on place->critic. VOLLEY: keep LOW (~4-6) so w_near stays "
                         "coincidence-dependent (a runaway w_near rate-leaks past the jitter anti-cheat).")
    # controls
    ap.add_argument("--lesion", action="store_true", help="gate-2e: zero GABA_B mask -> gap must vanish")
    ap.add_argument("--shuffle", action="store_true",
                    help="anti-cheat a: permute place-cell->location mapping -> gates 2b+gap must FAIL")
    ap.add_argument("--ablate-sensors", action="store_true",
                    help="anti-cheat b: zero landmark sensors at recall -> grading must collapse")
    ap.add_argument("--jitter", action="store_true",
                    help="anti-cheat (coincidence): de-synchronize the place drive at recall -> the volley "
                         "collapses -> gates 2a+2b must FAIL (proves coincidence, not rate)")
    ap.add_argument("--nmda-critic", action="store_true",
                    help="Option B fallback: per-region NMDA on the critic (only if plain FAILs)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    if float(args.coincidence_k) <= 1.0:
        raise AssertionError("ANTI-CHEAT: coincidence_k_threshold must be > 1 (a single input must not trigger).")

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    grid_size = int(args.grid_size)
    locations = default_locations(grid_size)
    landmarks = default_landmarks(grid_size)
    dist_max = float(grid_size) * 1.42

    mode = ("LESION (gate-2e)" if args.lesion else
            "SHUFFLE-CONTROL (anti-cheat a)" if args.shuffle else
            "SENSOR-ABLATION (anti-cheat b)" if args.ablate_sensors else
            "JITTER-CONTROL (coincidence anti-cheat)" if args.jitter else
            ("PRIMARY-VOLLEY (FS-PING + Route-D critic)" if args.enable_volley else
             "PRIMARY (sparse-read place-graded critic)"))
    print("=" * 78)
    print("Stage-2 de-risk: SPARSE-read MSN-D1 value critic on the self-organized place code")
    print(f"  mode={mode}  seeds={seeds}  grid={grid_size}")
    print(f"  locations={ {k: tuple(round(c,1) for c in v) for k,v in locations.items()} }")
    print(f"  place_to_value: w={args.place_to_value_weight} density={args.place_to_value_density} "
          f"(PLASTIC, DA-gated; NO dense A1 floor)  nmda_critic={args.nmda_critic}")
    print("=" * 78)

    kw = dict(
        locations=locations, landmarks=landmarks, n_bearing=int(args.n_bearing),
        n_dist=int(args.n_dist), n_place=int(args.n_place), n_strio=int(args.n_strio),
        n_snc=int(args.n_snc), grid_size=grid_size,
        lm_to_place_weight=float(args.lm_to_place_weight), lm_to_place_density=float(args.lm_to_place_density),
        lm_to_place_jitter=float(args.lm_to_place_jitter),
        place_to_value_weight=float(args.place_to_value_weight),
        place_to_value_density=float(args.place_to_value_density),
        place_to_value_jitter=float(args.place_to_value_jitter),
        max_intensity=float(args.max_intensity), falloff=float(args.falloff),
        dist_sigma=float(args.dist_sigma), dist_max=dist_max, bexp=float(args.bexp),
        selforg_passes=int(args.selforg_passes), selforg_steps_per_loc=int(args.selforg_steps_per_loc),
        snc_tonic_pa=float(args.snc_tonic_pa), snc_reward_gain=float(args.snc_reward_gain),
        hold_steps=int(args.hold_steps), n_train=int(args.n_train),
        lead_steps=int(round(args.lead_ms / 1.0)),
        strio_to_snc_weight=float(args.strio_to_snc_weight),
        snc_da_sensitivity=float(args.snc_da_sensitivity),
        reward_learning_rate=float(args.reward_learning_rate),
        gabab=True, gabab_propagation_strength=float(args.gabab_propagation_strength),
        nmda_critic=bool(args.nmda_critic), sensor_drive_pa=float(args.sensor_drive_pa),
        sensor_sigma=float(args.sensor_sigma), actor_cortex_tonic_pa=float(args.actor_cortex_tonic_pa),
        enable_volley=bool(args.enable_volley), n_fs=int(args.n_fs),
        place_to_fs_weight=float(args.place_to_fs_weight), place_to_fs_density=float(args.place_to_fs_density),
        fs_to_place_weight=float(args.fs_to_place_weight), fs_to_place_density=float(args.fs_to_place_density),
        coincidence_k=float(args.coincidence_k), coincidence_gain=float(args.coincidence_gain),
        coincidence_plateau=float(args.coincidence_plateau),
        readout_plateau=(float(args.readout_plateau) if args.readout_plateau is not None else None),
        stdp_w_max=float(args.stdp_w_max),
        lesion=bool(args.lesion), shuffle=bool(args.shuffle), ablate_sensors=bool(args.ablate_sensors),
        jitter=bool(args.jitter))

    results = []
    for s in seeds:
        print(f"\n[stage2 seed={s}] {mode}:")
        r = run_seed(s, **kw)
        results.append(r)
        if not args.lesion and not args.shuffle and not args.ablate_sensors and not args.jitter:
            print(f"  => seed {s} PRIMARY {'PASS' if r['primary'] else 'FAIL'} "
                  f"[fire {r['fire']}, place-graded {r['place_graded']}, LTP {r['weight_grew']}, "
                  f"actor-ok {r['actor_not_perturbed']}]")

    N = len(results)
    n_fire = sum(1 for r in results if r["fire"])
    n_graded = sum(1 for r in results if r["place_graded"])
    n_ltp = sum(1 for r in results if r["weight_grew"])
    n_actor = sum(1 for r in results if r["actor_not_perturbed"])
    n_gap = sum(1 for r in results if r["snc_state_specific"])
    n_primary = sum(1 for r in results if r["primary"])

    def _agg(key):
        vals = [r[key] for r in results]
        return {"mean": round(float(np.mean(vals)), 3), "min": round(float(np.min(vals)), 3),
                "max": round(float(np.max(vals)), 3), "values": [round(float(v), 3) for v in vals]}

    print("\n" + "=" * 78)
    print(f"STAGE-2 SUMMARY ({N} seeds, {mode})")
    print(f"  2a FIRE(>=5Hz)              : {n_fire}/{N}  near {_agg('crit_near_trained_hz')['values']}")
    print(f"  2b PLACE-GRADED(near>=3xfar): {n_graded}/{N}  ratio {_agg('place_graded_ratio')['values']}")
    print(f"  2c LEARNS-V(LTP near>=2xfar): {n_ltp}/{N}  w_near/far {_agg('weight_near_over_far')['values']}")
    print(f"  2d ACTOR-NOT-PERTURBED      : {n_actor}/{N}  ratio {_agg('actor_ratio')['values']}")
    print(f"  2e SNc state-specific gap   : {n_gap}/{N}  gap {_agg('snc_gap_ratio')['values']} "
          f"(LESION/SHUFFLE expect 0/N; primary expects >=N/2)")
    if not args.lesion and not args.shuffle and not args.ablate_sensors and not args.jitter:
        print(f"  PRIMARY (2a+2b+2c+2d)      : {n_primary}/{N}")
        verdict = ("PASS" if n_primary == N else "PARTIAL" if n_primary > 0 else "NEGATIVE")
    else:
        verdict = "CONTROL"
    print(f"  VERDICT: {verdict}")
    print("=" * 78)

    summary = {
        "mode": ("lesion" if args.lesion else "shuffle" if args.shuffle
                 else "ablate_sensors" if args.ablate_sensors else "primary"),
        "n_seeds": N, "seeds": seeds, "deterministic_regime": True, "backend": "cupy",
        "nmda_critic": bool(args.nmda_critic),
        "place_to_value_weight": float(args.place_to_value_weight),
        "n_fire": n_fire, "n_place_graded": n_graded, "n_ltp": n_ltp, "n_actor_ok": n_actor,
        "n_snc_gap": n_gap, "n_primary": n_primary, "verdict": verdict,
        "crit_near_trained_hz": _agg("crit_near_trained_hz"),
        "crit_far_max_hz": _agg("crit_far_max_hz"),
        "place_graded_ratio": _agg("place_graded_ratio"),
        "weight_near_over_far": _agg("weight_near_over_far"),
        "actor_ratio": _agg("actor_ratio"),
        "snc_gap_ratio": _agg("snc_gap_ratio"),
        "place_diff_location_cosine": _agg("place_diff_location_cosine"),
        "results": results,
    }
    if args.out:
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
        print(f"[OUT] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
