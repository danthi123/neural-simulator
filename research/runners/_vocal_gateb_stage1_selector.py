"""Gate B Stage 1: continuous center-surround basal-ganglia selector.

This builds on the Stage-0 autonomous tonic-output substrate
(`research/findings/2026-08-06-gateB-stage0-tonic-output-seed-robustness-and-stage1-reanchor.md`).
It starts from the Gate A v2 populations
(`research/runners/_vocal_action_selector_gate.py`) but:

  * GPi/SNr runs on an immutable region-scoped ``intrinsic_current_pA`` from
    step 0 (the Stage-0 scaffold) -- there is NO host GPi tonic current;
  * ``selector_reset`` and its reset pathways are removed -- there is NO reset
    current and NO host stop-on-winner;
  * two missing center-surround pathways are added in mechanism order:
      1. proposal/cortex -> shared STN (fast hyperdirect hold; Nambu 2002),
      2. GPe -> same-channel GPi/SNr (direct pallidal control of output).

The circuit alone decides when an action begins and ends. The host presents a
shared proposal-onset (practice-arousal) drive at fixed times and a constant
motor-thalamus afferent drive; it never injects a channel-specific current,
never drives GPi/SNr, and never breaks on a winner or resets state.

All weights are immutable (no learning). This is a CONSTRUCTION gate scored from
step 0 on a single seed, both backends. Reward/credit is Stage 2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from research.runners._vocal_action_selector_gate import (
    CHANNELS,
    _indices,
    _region,
)
from sim import (
    CoreSimConfig,
    GPUConfig,
    NeuronModel,
    RuntimeState,
    SimulationBridge,
    VisualizationConfig,
)
from sim.backend import get_backend, to_host
from sim.enums import NeuronType
from sim.regions import BrainRegion, RegionPathway
from tools.lab import assert_backend
from tools.verdict import Verdict


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "research/findings/raw/gateb_stage1_selector"

# Construction seed (fresh; not a sealed capability seed -- this is a
# single-seed construction gate, not a multiseed capability partition).
CONSTRUCTION_SEED = 730501

# Autonomous GPi/SNr pacemaker drive. 140 pA gives ~45 Hz tonic output in this
# 20-neuron GPi population (in the Stage-0 40-80 Hz band), strong enough to
# clamp the motor thalamus at rest yet transiently pausable by the direct
# pathway. Immutable, region-scoped; the reduced Izhikevich pacemaker
# representation is entered in docs/SCAFFOLD-LEDGER.md.
GPI_INTRINSIC_PA = 140.0

# The commit/motor read-out is NON-recurrent (internal_density 0) so it tracks
# the thalamic relay and terminates the moment GPi/SNr re-clamps the thalamus,
# rather than self-sustaining after an action ends.
COMMIT_INTERNAL_DENSITY = 0.0


# --- construction weights (symmetric across channels; immutable at runtime) ---
# Operating point found by transfer-function characterisation (see finding):
# GPi tonic ~45 Hz clamps thalamus (gpi_to_thal 25) at the 300 pA afferent; a
# proposal drives D1 (proposal_to_msn 40) which pauses GPi 45->~7 Hz, releasing
# the thalamus. Hyperdirect proposal->STN gives an early global hold without
# saturating GPi.
ENABLE_STRIATAL_FSI = True

W = {
    "arousal_to_proposal": 10.0,
    "proposal_to_msn": 40.0,
    "proposal_to_msn_density": 1.0,
    # striatal FSI feed-forward lateral inhibition (center-surround at the
    # striatum): each channel's proposal drives its FSI, which inhibits the
    # OTHER channel's near-threshold MSNs. The slightly-more-driven channel's
    # D1 crosses first, suppressing the competitor -> sharp winner-take-all.
    "proposal_to_fsi": 100.0,
    "fsi_to_msn": 32.0,
    # hyperdirect: both channels' proposals -> shared STN
    "proposal_to_stn": 2.0,
    "proposal_to_stn_density": 1.0,
    # direct: D1 -> GPi (inhibitory)
    "d1_to_gpi": 15.0,
    # indirect: D2 -> GPe -> STN
    "d2_to_gpe": 2.5,
    "d2_to_gpe_density": 0.60,
    "gpe_to_stn": 1.5,
    "gpe_to_stn_density": 0.30,
    # NEW GPe -> same-channel GPi (inhibitory pallidal control)
    "gpe_to_gpi": 2.0,
    "gpe_to_gpi_density": 0.5,
    # STN -> GPi (excitatory)
    "stn_to_gpi": 1.0,
    "stn_to_gpi_density": 0.40,
    # GPi -> thalamus (inhibitory)
    "gpi_to_thal": 25.0,
    "thal_to_commit": 40.0,
    "commit_to_fsi": 30.0,
    "commit_fsi_cross": 60.0,
    "commit_to_motor": 80.0,
}

# region sizes (reuse Gate A v2 shapes)
N = {
    "proposal": 60, "striatum": 36, "gpe": 16, "gpi": 20, "stn": 20,
    "thal": 24, "commit": 30, "commit_fs": 16, "motor": 30, "practice": 24,
}

# runtime protocol
OU_SIGMA_PA = 40.0
PRACTICE_PA = 1000.0
THALAMUS_TONIC_PA = 300.0
# Disclosed one-time initialisation into the continuous-brain baseline attractor
# (GPi tonic, thalamus clamped, motor silent). This is NOT a per-action reset:
# there is exactly one settle before scoring begins, and NO reset between the
# two scored actions. The brain runs uninterrupted across both actions. The
# cold-start transient from rest is separately measured and reported.
SETTLE_STEPS = 150
BASELINE_STEPS = 150
ONSET_STEPS = 200
GAP_STEPS = 200
N_ACTIONS = 4          # attempt 4 windows; gate needs >=2 clean (margin)
EARLY_WINDOW = 40      # steps after onset counted as the hyperdirect "hold"
MOTOR_THRESHOLD = 12   # spikes in an action window to count as a motor action
THAL_RELEASE_THRESHOLD = 12  # winner-thalamus spikes to count as a release
LOSER_RATIO = 0.25     # behavioral: competitor MOTOR must be <=25% of winner
# Thalamic focus: the winner's thalamic release must more than double the
# loser's (loser < 50% of winner). Looser than the motor gate because the
# unselected channel's thalamus is expected to leak partially -- the clean
# BEHAVIORAL output is the motor pool, which the strict LOSER_RATIO governs.
# (A tighter thalamic cutoff only tracks a backend's exact noise realisation,
# not the selection phenotype.)
THAL_LOSER_RATIO = 0.5
TONIC_LO_HZ = 25.0     # GPi tonic band (per-neuron over a phase window)
TONIC_HI_HZ = 90.0


def _hash(value) -> str:
    array = np.ascontiguousarray(np.asarray(to_host(value)))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _gpi_region(name: str, n: int) -> BrainRegion:
    return BrainRegion(
        name=name, n_neurons=int(n), exc_fraction=0.0, internal_density=0.1,
        exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
        plastic_internal=False,
        izh_neuron_type=NeuronType.IZH2007_GPI_OUTPUT.name,
        intrinsic_current_pA=float(GPI_INTRINSIC_PA),
        enable_nmda=False, enable_homeostasis=False, enable_heterogeneity=True,
    )


def build_stage1_bridge(seed: int) -> SimulationBridge:
    rs = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL
    fs = NeuronType.IZH2007_FS_CORTICAL_INTERNEURON
    d1 = NeuronType.IZH2007_STRIATAL_MSN_D1
    d2 = NeuronType.IZH2007_STRIATAL_MSN_D2
    gpe = NeuronType.IZH2007_GPE_PACEMAKER
    stn = NeuronType.IZH2007_STN_BURST
    thal = NeuronType.IZH2007_THALAMIC_RELAY

    regions = [
        _region("practice_arousal", N["practice"], exc_fraction=1.0, neuron_type=rs),
        _region("selector_stn", N["stn"], exc_fraction=1.0, neuron_type=stn),
    ]
    for c in CHANNELS:
        regions.append(
            _region(f"proposal_{c}", N["proposal"], exc_fraction=1.0, neuron_type=rs))
        if ENABLE_STRIATAL_FSI:
            regions.append(
                _region(f"str_fsi_{c}", N["commit_fs"], exc_fraction=0.0, neuron_type=fs))
        regions.extend([
            _region(f"str_d1_{c}", N["striatum"], exc_fraction=0.0, neuron_type=d1),
            _region(f"str_d2_{c}", N["striatum"], exc_fraction=0.0, neuron_type=d2),
            _region(f"gpe_{c}", N["gpe"], exc_fraction=0.0, neuron_type=gpe),
            _gpi_region(f"gpi_{c}", N["gpi"]),
            _region(f"thal_{c}", N["thal"], exc_fraction=1.0, neuron_type=thal),
            _region(f"commit_{c}", N["commit"], exc_fraction=1.0, neuron_type=rs,
                    internal_density=COMMIT_INTERNAL_DENSITY,
                    internal_weight=0.5, enable_nmda=False),
            _region(f"commit_fs_{c}", N["commit_fs"], exc_fraction=0.0, neuron_type=fs),
            _region(f"motor_{c}", N["motor"], exc_fraction=1.0, neuron_type=rs),
        ])

    pathways = []
    for c in CHANNELS:
        other = 1 - c
        if ENABLE_STRIATAL_FSI:
            pathways.extend([
                RegionPathway(from_region=f"proposal_{c}", to_region=f"str_fsi_{c}",
                              density=1.0, weight_mean=W["proposal_to_fsi"],
                              weight_jitter=0.0, plastic=False),
                RegionPathway(from_region=f"str_fsi_{c}", to_region=f"str_d1_{other}",
                              density=1.0, weight_mean=W["fsi_to_msn"],
                              weight_jitter=0.0, plastic=False, receptor="gaba_a"),
                RegionPathway(from_region=f"str_fsi_{c}", to_region=f"str_d2_{other}",
                              density=1.0, weight_mean=W["fsi_to_msn"],
                              weight_jitter=0.0, plastic=False, receptor="gaba_a"),
            ])
        pathways.extend([
            RegionPathway(from_region="practice_arousal", to_region=f"proposal_{c}",
                          density=1.0, weight_mean=W["arousal_to_proposal"],
                          weight_jitter=0.0, plastic=False),
            RegionPathway(from_region=f"proposal_{c}", to_region=f"str_d1_{c}",
                          density=W["proposal_to_msn_density"],
                          weight_mean=W["proposal_to_msn"], weight_jitter=0.05,
                          plastic=False),
            RegionPathway(from_region=f"proposal_{c}", to_region=f"str_d2_{c}",
                          density=W["proposal_to_msn_density"],
                          weight_mean=W["proposal_to_msn"], weight_jitter=0.05,
                          plastic=False),
            # hyperdirect: proposal -> shared STN (fast global hold)
            RegionPathway(from_region=f"proposal_{c}", to_region="selector_stn",
                          density=W["proposal_to_stn_density"],
                          weight_mean=W["proposal_to_stn"], weight_jitter=0.05,
                          plastic=False),
            # direct: D1 -> GPi (inhibitory; sign from exc_fraction=0 source)
            RegionPathway(from_region=f"str_d1_{c}", to_region=f"gpi_{c}",
                          density=1.0, weight_mean=W["d1_to_gpi"],
                          weight_jitter=0.05, plastic=False, receptor="gaba_a"),
            # indirect: D2 -> GPe
            RegionPathway(from_region=f"str_d2_{c}", to_region=f"gpe_{c}",
                          density=W["d2_to_gpe_density"], weight_mean=W["d2_to_gpe"],
                          weight_jitter=0.05, plastic=False),
            RegionPathway(from_region=f"gpe_{c}", to_region="selector_stn",
                          density=W["gpe_to_stn_density"], weight_mean=W["gpe_to_stn"],
                          weight_jitter=0.05, plastic=False, receptor="gaba_a"),
            # NEW GPe -> same-channel GPi (direct pallidal inhibition of output)
            RegionPathway(from_region=f"gpe_{c}", to_region=f"gpi_{c}",
                          density=W["gpe_to_gpi_density"], weight_mean=W["gpe_to_gpi"],
                          weight_jitter=0.05, plastic=False, receptor="gaba_a"),
            RegionPathway(from_region="selector_stn", to_region=f"gpi_{c}",
                          density=W["stn_to_gpi_density"], weight_mean=W["stn_to_gpi"],
                          weight_jitter=0.05, plastic=False),
            RegionPathway(from_region=f"gpi_{c}", to_region=f"thal_{c}",
                          density=1.0, weight_mean=W["gpi_to_thal"],
                          weight_jitter=0.05, plastic=False, receptor="gaba_a"),
            RegionPathway(from_region=f"thal_{c}", to_region=f"commit_{c}",
                          density=1.0, weight_mean=W["thal_to_commit"],
                          weight_jitter=0.05, plastic=False),
            RegionPathway(from_region=f"commit_{c}", to_region=f"commit_fs_{c}",
                          density=1.0, weight_mean=W["commit_to_fsi"],
                          weight_jitter=0.0, plastic=False),
            RegionPathway(from_region=f"commit_fs_{c}", to_region=f"commit_{other}",
                          density=1.0, weight_mean=W["commit_fsi_cross"],
                          weight_jitter=0.0, plastic=False, receptor="gaba_a"),
            RegionPathway(from_region=f"commit_{c}", to_region=f"motor_{c}",
                          density=1.0, weight_mean=W["commit_to_motor"],
                          weight_jitter=0.05, plastic=False),
        ])

    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = int(seed)
    cfg.enable_brain_region_framework = True
    cfg.enable_ou_process = True
    cfg.ou_mean_current_pA = 0.0
    cfg.ou_std_current_pA = float(OU_SIGMA_PA)
    cfg.ou_tau_ms = 15.0
    for flag in (
        "enable_short_term_plasticity", "enable_hebbian_learning",
        "enable_homeostasis", "enable_structural_plasticity",
        "enable_reward_modulation", "enable_stdp", "enable_inhibitory_stdp",
    ):
        setattr(cfg, flag, False)
    cfg.brain_regions = regions
    cfg.region_pathways = pathways

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(enable_profiling=False),
    )
    bridge.strict_step_errors = True
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    if not bridge.is_initialized:
        raise RuntimeError("bridge initialization failed")

    # Undo the wiring layer's 0.01 floor on declared zero-weight (internal
    # recurrent) edges, matching the Stage-0 substrate. Every DECLARED pathway
    # weight here is >> 0.011, so only the spurious floor edges are removed.
    xp0, _ = get_backend()
    bridge.cp_connections.data[:] = xp0.where(
        bridge.cp_connections.data <= xp0.float32(0.011),
        xp0.float32(0.0), bridge.cp_connections.data,
    )

    # OU noise only on the proposal populations (symmetry-breaking); everything
    # else is deterministic given the seed.
    xp, _ = get_backend()
    proposal_idx = np.concatenate([_indices(bridge, f"proposal_{c}") for c in CHANNELS])
    bridge.cp_ou_neuron_mask = xp.zeros(int(cfg.num_neurons), dtype=bool)
    bridge.cp_ou_neuron_mask[xp.asarray(proposal_idx)] = True

    # Initialise the GPi/SNr pacemaker neurons at desynchronised points in their
    # sub-threshold cycle (uniform in [vr, vt]) instead of all at rest. A
    # continuously-running brain's output pacemaker is never "off"; a random
    # phase both suppresses the cold-start synchronised volley and prevents an
    # artificial synchronised first spike. Seeded for determinism.
    gpi_idx = np.concatenate([_indices(bridge, f"gpi_{c}") for c in CHANNELS])
    rng = np.random.default_rng(int(seed))
    v_host = np.asarray(to_host(bridge.cp_membrane_potential_v)).copy()
    v_host[gpi_idx] = rng.uniform(-65.0, -50.0, size=gpi_idx.size).astype(np.float32)
    bridge.cp_membrane_potential_v[:] = xp.asarray(v_host, dtype=xp.float32)
    return bridge


def _apply_afferents(bridge, *, arousal: bool):
    """Constant motor-thalamus afferent drive; optional shared proposal onset.

    GPi/SNr is NEVER driven here -- it runs on intrinsic current alone.
    """
    xp, _ = get_backend()
    bridge.cp_external_input_current[:] = xp.float32(0.0)
    for c in CHANNELS:
        bridge.cp_external_input_current[
            xp.asarray(_indices(bridge, f"thal_{c}"))
        ] = xp.float32(THALAMUS_TONIC_PA)
    if arousal:
        bridge.cp_external_input_current[
            xp.asarray(_indices(bridge, "practice_arousal"))
        ] = xp.float32(PRACTICE_PA)


def _region_idx(bridge):
    names = ["practice_arousal", "selector_stn"]
    for c in CHANNELS:
        names += [f"proposal_{c}", f"str_d1_{c}", f"str_d2_{c}", f"gpe_{c}",
                  f"gpi_{c}", f"thal_{c}", f"commit_{c}", f"motor_{c}"]
    return {name: np.asarray(_indices(bridge, name), dtype=np.int64) for name in names}


def _phase_rate(raster_slice, indices, n) -> float:
    steps = raster_slice.shape[0]
    if steps == 0:
        return 0.0
    return float(raster_slice[:, indices].sum() / n / (steps / 1000.0))


def run_stage1(seed: int, *, verbose: bool = False) -> dict:
    bridge = build_stage1_bridge(seed)
    ridx = _region_idx(bridge)
    xp, _ = get_backend()
    gpi_ext_indices = np.concatenate([_indices(bridge, f"gpi_{c}") for c in CHANNELS])

    weight_hash_before = _hash(bridge.cp_connections.data)
    intrinsic_before = _hash(bridge.cp_intrinsic_current_pA)

    n = int(bridge.core_config.num_neurons)
    gpi_ext_zero = True

    # ---- disclosed one-time settle into the continuous-brain baseline ----
    # Runs intrinsic GPi drive + thalamic afferent, NO arousal, NO reset. This
    # is NOT a per-action reset (there is one settle, then two+ actions run
    # uninterrupted). The cold-start transient from rest is measured here and
    # reported as an initialisation caveat, not hidden.
    settle_raster = np.zeros((SETTLE_STEPS, n), dtype=bool)
    for step in range(SETTLE_STEPS):
        _apply_afferents(bridge, arousal=False)
        gpi_ext_zero = gpi_ext_zero and bool(
            np.all(np.asarray(to_host(bridge.cp_external_input_current[
                xp.asarray(gpi_ext_indices)])) == 0.0)
        )
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        settle_raster[step] = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
    coldstart_motor = int(sum(settle_raster[:, ridx[f"motor_{c}"]].sum() for c in CHANNELS))
    coldstart_thal = int(sum(settle_raster[:, ridx[f"thal_{c}"]].sum() for c in CHANNELS))
    # steady baseline reached by the end of settle (last 50 steps)
    settle_tail_motor = int(sum(settle_raster[-50:][:, ridx[f"motor_{c}"]].sum() for c in CHANNELS))

    total_steps = BASELINE_STEPS + N_ACTIONS * (ONSET_STEPS + GAP_STEPS)
    raster = np.zeros((total_steps, n), dtype=bool)

    # phase schedule
    phases = []  # (label, start, end, arousal)
    t = 0
    phases.append(("baseline", t, t + BASELINE_STEPS, False)); t += BASELINE_STEPS
    for a in range(N_ACTIONS):
        phases.append((f"onset_{a}", t, t + ONSET_STEPS, True)); t += ONSET_STEPS
        phases.append((f"gap_{a}", t, t + GAP_STEPS, False)); t += GAP_STEPS

    arousal_by_step = np.zeros(total_steps, dtype=bool)
    for _, s, e, ar in phases:
        arousal_by_step[s:e] = ar

    for step in range(total_steps):
        _apply_afferents(bridge, arousal=bool(arousal_by_step[step]))
        # GPi/SNr must never receive external drive
        gpi_ext_zero = gpi_ext_zero and bool(
            np.all(np.asarray(to_host(bridge.cp_external_input_current[
                xp.asarray(gpi_ext_indices)])) == 0.0)
        )
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        raster[step] = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)

    weight_hash_after = _hash(bridge.cp_connections.data)
    intrinsic_after = _hash(bridge.cp_intrinsic_current_pA)

    def rate(sl, region, n_region):
        return _phase_rate(raster[sl[0]:sl[1]], ridx[region], n_region)

    # baseline metrics
    bl = (0, BASELINE_STEPS)
    baseline_gpi = float(np.mean([rate(bl, f"gpi_{c}", N["gpi"]) for c in CHANNELS]))
    baseline_thal = int(sum(raster[bl[0]:bl[1]][:, ridx[f"thal_{c}"]].sum() for c in CHANNELS))
    baseline_motor = int(sum(raster[bl[0]:bl[1]][:, ridx[f"motor_{c}"]].sum() for c in CHANNELS))

    actions = []
    for a in range(N_ACTIONS):
        o_s = BASELINE_STEPS + a * (ONSET_STEPS + GAP_STEPS)
        o_e = o_s + ONSET_STEPS
        g_s, g_e = o_e, o_e + GAP_STEPS
        early = (o_s, o_s + EARLY_WINDOW)

        stn_baseline = rate(bl, "selector_stn", N["stn"])
        stn_early = rate(early, "selector_stn", N["stn"])
        gpi_early = float(np.mean([rate(early, f"gpi_{c}", N["gpi"]) for c in CHANNELS]))

        motor_spikes = [int(raster[o_s:o_e][:, ridx[f"motor_{c}"]].sum()) for c in CHANNELS]
        thal_spikes = [int(raster[o_s:o_e][:, ridx[f"thal_{c}"]].sum()) for c in CHANNELS]
        gpi_action = [rate((o_s, o_e), f"gpi_{c}", N["gpi"]) for c in CHANNELS]

        # The BG selector's OUTPUT is thalamic disinhibition: the winner is the
        # channel whose motor thalamus is released. The motor pool is the
        # read-out that must AGREE with the thalamic winner.
        winner = int(np.argmax(thal_spikes))
        loser = 1 - winner
        motor_winner = int(np.argmax(motor_spikes))
        winner_spk = motor_spikes[winner]
        loser_spk = motor_spikes[loser]
        loser_ratio = float(loser_spk / max(1, winner_spk))
        thal_loser_ratio = float(thal_spikes[loser] / max(1, thal_spikes[winner]))
        gpi_pause_winner = gpi_action[winner]

        # autonomous return: score the LATTER half of the gap so the action has
        # a physiological wind-down before silence is required.
        rg_s = g_s + GAP_STEPS // 2
        gpi_return = float(np.mean([rate((rg_s, g_e), f"gpi_{c}", N["gpi"]) for c in CHANNELS]))
        motor_return = int(sum(raster[rg_s:g_e][:, ridx[f"motor_{c}"]].sum() for c in CHANNELS))
        thal_return = int(sum(raster[rg_s:g_e][:, ridx[f"thal_{c}"]].sum() for c in CHANNELS))

        clean = bool(
            motor_winner == winner                            # read-out agrees with BG
            and winner_spk >= MOTOR_THRESHOLD                 # a real motor action
            and loser_ratio <= LOSER_RATIO                    # competitor motor suppressed
            and thal_spikes[winner] >= THAL_RELEASE_THRESHOLD # winner thalamus released
            and thal_loser_ratio <= THAL_LOSER_RATIO          # winner thalamus dominates loser >2x
            and gpi_action[winner] < gpi_action[loser]        # winner GPi more paused (focused)
            and gpi_pause_winner < baseline_gpi               # and below its own tonic baseline
            and TONIC_LO_HZ <= gpi_return <= TONIC_HI_HZ      # autonomous return to tonic
            and motor_return == 0                             # motor silent again
        )
        action = {
            "action_index": a, "winner": winner, "motor_winner": motor_winner,
            "motor_spikes": motor_spikes, "thal_spikes": thal_spikes,
            "winner_spikes": winner_spk, "loser_spikes": loser_spk,
            "loser_ratio": loser_ratio, "thal_loser_ratio": thal_loser_ratio,
            "stn_baseline_hz": stn_baseline, "stn_early_hz": stn_early,
            "gpi_early_hz": gpi_early,
            "gpi_action_hz": gpi_action, "gpi_pause_winner_hz": gpi_pause_winner,
            "gpi_return_hz": gpi_return, "motor_return_spikes": motor_return,
            "thal_return_spikes": thal_return,
            "early_stn_rise": bool(stn_early > stn_baseline),
            "early_gpi_hold": bool(gpi_early >= baseline_gpi * 0.8),
            "clean": clean,
        }
        actions.append(action)
        if verbose:
            print(f"[{os.environ.get('SIM_BACKEND')}] action {a}: winner={winner} "
                  f"motor={motor_spikes} thal={thal_spikes} loser_ratio={loser_ratio:.2f} "
                  f"thal_lr={thal_loser_ratio:.2f} gpi_w/l={gpi_action[winner]:.1f}/{gpi_action[loser]:.1f} "
                  f"base={baseline_gpi:.1f} stn {stn_baseline:.1f}->{stn_early:.1f} "
                  f"gpi_ret={gpi_return:.1f} motor_ret={motor_return} clean={clean}",
                  flush=True)

    n_clean = int(sum(a["clean"] for a in actions))
    clean_actions = [a for a in actions if a["clean"]]
    checks = {
        "tonic_gpi_baseline": bool(TONIC_LO_HZ <= baseline_gpi <= TONIC_HI_HZ),
        "thalamus_inhibited_baseline": bool(baseline_thal == 0),
        "motor_silent_baseline": bool(baseline_motor == 0),
        "early_stn_rise": bool(clean_actions and all(a["early_stn_rise"] for a in clean_actions)),
        "focused_gpi_pause": bool(clean_actions and all(
            a["gpi_action_hz"][a["winner"]] < a["gpi_action_hz"][1 - a["winner"]]
            for a in clean_actions)),
        "winner_thalamus_released": bool(clean_actions and all(
            a["thal_spikes"][a["winner"]] >= THAL_RELEASE_THRESHOLD for a in clean_actions)),
        "competitor_thalamus_suppressed": bool(clean_actions and all(
            a["thal_loser_ratio"] <= THAL_LOSER_RATIO for a in clean_actions)),
        "one_clean_motor_action": bool(n_clean >= 1),
        "competitor_motor_suppressed": bool(clean_actions and all(
            a["loser_ratio"] <= LOSER_RATIO for a in clean_actions)),
        "autonomous_return": bool(clean_actions and all(
            a["motor_return_spikes"] == 0
            and TONIC_LO_HZ <= a["gpi_return_hz"] <= TONIC_HI_HZ
            for a in clean_actions)),
        "at_least_two_clean_actions": bool(n_clean >= 2),
        "weights_immutable": bool(weight_hash_before == weight_hash_after),
        "intrinsic_immutable": bool(intrinsic_before == intrinsic_after),
        "zero_gpi_external_current": bool(gpi_ext_zero),
    }
    bridge.clear_simulation_state_and_gpu_memory()
    return {
        "seed": int(seed),
        "settle_steps": SETTLE_STEPS,
        "coldstart_motor_spikes": coldstart_motor,
        "coldstart_thal_spikes": coldstart_thal,
        "settle_tail_motor_spikes": settle_tail_motor,
        "baseline_gpi_hz": baseline_gpi,
        "baseline_thal_spikes": baseline_thal,
        "baseline_motor_spikes": baseline_motor,
        "n_clean_actions": n_clean,
        "actions": actions,
        "checks": checks,
        "pass": bool(all(checks.values())),
        "weight_hash": weight_hash_after,
        "raster_hash": _hash(raster),
    }


def _backend_info() -> dict:
    requested = os.environ.get("SIM_BACKEND")
    if requested not in ("numpy", "cupy"):
        raise ValueError("SIM_BACKEND must be explicitly set to numpy or cupy")
    assert_backend(requested, note="Gate B Stage 1 selector")
    xp, actual = get_backend()
    if actual != requested:
        raise RuntimeError(f"requested {requested}, resolved {actual}")
    info = {"backend": actual, "device": "CPU (NumPy backend)", "host": platform.node()}
    if actual == "cupy":
        name = xp.cuda.runtime.getDeviceProperties(0)["name"]
        info["device"] = name.decode() if isinstance(name, bytes) else str(name)
    return info


def _earned_verdict(result: dict) -> dict:
    verdict = Verdict("Gate B Stage 1 continuous BG selector")
    for name, ok in result["checks"].items():
        verdict.require(name, bool(ok), expect=True)
    decided = verdict.decide(go=bool(result["pass"]), verbose=True)
    return {
        "verdict_status": decided["status"],
        "preconditions": decided["preconditions"],
        "undefined_reasons": decided["undefined_reasons"],
        "go": decided["go"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=CONSTRUCTION_SEED)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    backend = _backend_info()
    started = time.perf_counter()
    result = run_stage1(args.seed, verbose=args.verbose)
    verdict = _earned_verdict(result)
    outcome = ("STAGE1_GO" if verdict["go"] else "STAGE1_NO_GO")
    if verdict["verdict_status"] == "UNDEFINED":
        outcome = "STAGE1_UNDEFINED"
    artifact = {
        "probe": "gateB_stage1_continuous_bg_selector",
        "stage": "stage1_construction",
        "backend": backend["backend"],
        "device": backend["device"],
        "backend_info": backend,
        "gpi_intrinsic_pA": GPI_INTRINSIC_PA,
        "construction_weights": W,
        "protocol": {
            "baseline_steps": BASELINE_STEPS, "onset_steps": ONSET_STEPS,
            "gap_steps": GAP_STEPS, "n_actions": N_ACTIONS,
            "practice_pA": PRACTICE_PA, "thalamus_tonic_pA": THALAMUS_TONIC_PA,
            "motor_threshold": MOTOR_THRESHOLD, "loser_ratio": LOSER_RATIO,
        },
        **result,
        **verdict,
        "outcome": outcome,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    out = Path(args.out) if args.out else OUT_DIR / f"{backend['backend']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({
        "outcome": outcome, "pass": result["pass"],
        "n_clean_actions": result["n_clean_actions"],
        "checks": result["checks"], "output": str(out),
    }, indent=2))
    if verdict["verdict_status"] == "UNDEFINED":
        return 1
    return 0 if verdict["go"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
