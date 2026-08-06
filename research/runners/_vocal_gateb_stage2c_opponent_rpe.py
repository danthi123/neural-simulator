"""Gate B Stage 2c: opponent / bidirectional credit (negative RPE) on the selector.

Surpasses the Stage-2b appetitive-only wall
(`research/findings/2026-08-06-gateB-stage2b-per-action-da-NO-GO.md`,
`STAGE2B_NO_GO`): per-action DA made reward-driven D1 potentiation ACTION-LOCAL
at the weight level, but the DA signal was APPETITIVE-ONLY (it could only raise
concentration above baseline; never dip below it). Under the winner-take-all
selector with dense reward this is a rich-get-richer runaway -- the marginally-
ahead channel is selected more, so credited more, so selected still more, until
it wins absolutely; and the loop runs IDENTICALLY whether reward is CONTINGENT
on the action or merely DECOUPLED-but-coincident (yoked), because the brain does
its biased action either way. So D_contingent == D_yoked and reversal is
impossible (the locked action is never un-emitted, never earns a demotion).

Stage 2c adds the **NEGATIVE arm**: a reward-EXPECTATION baseline so an executed
action that goes UNREWARDED yields DA BELOW baseline -> negative RPE -> D1-LTD on
the over-selected route, breaking the runaway.

    RPE(action) = reward - V(action)      # V = neural value estimate
    Delta_w = reward_learning_rate * per_synapse_DA(action) * eligibility_trace

where per_synapse_DA carries the DA concentration's deviation from its tonic
baseline (positive burst when reward > V, NEGATIVE DIP when reward < V). The
expectation V is a NEURAL value estimate -- the executed action's striatal D1
population FIRING RATE during the onset window (the basal-ganglia direct-pathway
value/go signal; its proposal->D1 route grows with reward, so the rate tracks
expected reward for that action). It is a read-out of a SPIKING population (like
the motor read-out that moves the body), NOT a host EMA / Python running-average;
the RPE subtraction (reward - V) is the DA system's own outcome-vs-expectation
comparison, expressed in the DA production rule. The baseline is applied ONLY in
the outcome epoch (the expected-reward time), so an unrewarded execution dips at
the omission time (Schultz 1998) and rewarded trials keep a clean positive burst.

The bidirectional-credit substrate is engaged: `enable_d1_d2_asymmetry` (D1 LTP
under +DA / LTD under -DA, via the per-synapse D1/D2 sign array) and
`reward_aversive_scale` (Schultz/Fiorillo asymmetry: the negative-RPE dip drives
LTD of SMALLER magnitude than the matching appetitive burst -- applied to the
negative entries of the per-action DA signal, gated by `enable_d1_d2_asymmetry`).

Kept from Stage 2b: per-action compartmentalised DA (`dopamine_{N,E,S,W}`,
`from_action_specific_reward` gated by `core_config.last_selected_action`, the
neural motor read-out; `compute_per_synapse_da_signal` routing each channel's DA
to its `action_index`-tagged `str_d1_c` afferents); NEURAL coactivity eligibility
scoped to the proposal->D1 routes; the neural EXPLORATION process (elevated OU
membrane noise on proposal + striatal D1/D2). The reward-OFF build at the Stage-1
noise level is asserted byte-identical (weights + raster hash) to `run_stage1`
(enable_d1_d2_asymmetry + per-action DA are gated OFF when reward is off), so the
scored circuit adds only DECLARED substrate knobs -- no host-designed wiring.

Acceptance criteria are FROZEN from the Stage-2 preregistration UNCHANGED
(`research/findings/2026-08-06-gateB-stage2-local-reward-credit-PREREGISTRATION.md`):
bias-free swap differential D_contingent - D_yoked >= 0.20 (the negative arm makes
the decoupled yoked control DIVERGE -- unrewarded executions are punished), the
acquisition/expression lesions, and same-brain reversal >= 0.60, across the same
6 dev seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
import platform
import time

import numpy as np

from research.runners._vocal_action_selector_gate import CHANNELS, _indices, _region
from research.runners._vocal_gateb_stage1_selector import (
    COMMIT_INTERNAL_DENSITY,
    ENABLE_STRIATAL_FSI,
    GPI_INTRINSIC_PA,
    N,
    OU_SIGMA_PA,
    PRACTICE_PA,
    SETTLE_STEPS,
    THALAMUS_TONIC_PA,
    W,
    _gpi_region,
    run_stage1,
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
from sim.neuromodulators import _default_per_action_dopamine_config
from sim.regions import RegionPathway
from tools.lab import assert_backend, attributable_to
from tools.verdict import Verdict


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "research/findings/raw/gateb_stage2c_opponent_rpe"

CONSTRUCTION_SEED = 730501
DEV_SEEDS = (730601, 730602, 730603, 730604, 730605, 730606)
HELDOUT_SEEDS = (730701, 730702, 730703, 730704, 730705, 730706)

# Weight bounds must bracket the fixed selector weights (up to 100) so the
# reward-modulation clip (hebbian bounds when STDP off) never collapses the
# construction circuit (the STDP soft-bound trap, CLAUDE.md). D1 routes may grow
# from 40 toward this ceiling.
W_MIN = 0.0
W_MAX = 600.0

# --- reward-credit operating point (calibrated single-seed, both backends) ---
REWARD_LEARNING_RATE = 0.02
REWARD_MAG = 1.0
REWARD_BASELINE = 0.0
REWARD_ELIGIBILITY_TAU_MS = 400.0
COACTIVITY_TRACE_TAU_MS = 40.0
COACTIVITY_THRESHOLD = 0.001
COACTIVITY_SCALE = 20.0

# --- neural value estimate / reward-expectation baseline (the negative arm) ---
# V(executed action) = VALUE_GAIN * (str_d1 onset spike count for that action),
# clipped to [0, VALUE_MAX]. The striatal D1 population rate is the BG direct-
# pathway value/go signal; its proposal->D1 route grows with reward, so the rate
# tracks expected reward for the action. VALUE_GAIN is the critic->DA read-out
# gain (calibrated so a well-rewarded action reads V~1.0 = the reward magnitude).
# This is a spiking-population read-out, NOT a host EMA. During the outcome epoch
# the DA production for the executed channel computes reward - V (negative RPE ->
# DA dip -> D1-LTD when the expected reward is omitted).
VALUE_GAIN = 0.010
VALUE_MAX = 1.5
# Schultz/Fiorillo aversive-vs-appetitive magnitude asymmetry: the negative-RPE
# dip drives LTD of this fraction of the matching appetitive burst's magnitude
# (applied to negative per-action DA entries by the bridge when
# enable_d1_d2_asymmetry is on). 1.0 = symmetric.
REWARD_AVERSIVE_SCALE = 0.5

# --- neural exploration process ---
# Elevated OU membrane-potential noise on the proposal populations (background
# synaptic-bombardment variability) so the un-learned selector SAMPLES both
# actions across trials instead of locking to the tiny weight-jitter bias (4/6
# Stage-2 dev seeds were seed-locked). Stage-1 uses OU_SIGMA_PA=40; the scored
# circuit uses EXPLORE_OU_SIGMA_PA. The equivalence guard build keeps
# OU_SIGMA_PA (Stage-1), so the byte-identical-to-Stage-1 proof is unaffected.
EXPLORE_OU_SIGMA_PA = 120.0
# Per-action DA channel names the bridge's v2 path expects (dopamine_{N,E,S,W});
# we use N->action 0, E->action 1. S/W are registered (bridge requires all four)
# but never tagged (no synapse carries action_index 2/3), so they are inert.
PER_ACTION_DA_NAMES = ("N", "E", "S", "W")

# --- trial protocol (fixed action windows; no reset, no stop-on-winner) ---
ONSET_STEPS = 200
GAP_STEPS = 300          # long enough for autonomous wind-down + eligibility decay
REWARD_DELAY = 10        # steps into the gap before reward onset
REWARD_STEPS = 60        # reward window length (dopamine delivery)
MOTOR_THRESHOLD = 12     # spikes in the onset window to count as a motor action
LOSER_RATIO = 0.35       # competitor motor <= this fraction of winner => clean
N_TRAIN = 40
N_TEST = 20


def _hash(value) -> str:
    array = np.ascontiguousarray(np.asarray(to_host(value)))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _route_indices(bridge, source, target) -> np.ndarray:
    coo = bridge.cp_connections.tocoo(copy=False)
    rows = np.asarray(to_host(coo.row), dtype=np.int64)
    cols = np.asarray(to_host(coo.col), dtype=np.int64)
    pre = _indices(bridge, source)
    post = _indices(bridge, target)
    return np.flatnonzero(np.isin(rows, pre) & np.isin(cols, post))


def build_stage2_bridge(seed: int, *, enable_reward: bool, plastic_d1: bool,
                        reward_learning_rate: float = REWARD_LEARNING_RATE,
                        ou_seed: int | None = None,
                        ou_sigma: float = EXPLORE_OU_SIGMA_PA,
                        explore_striatum: bool = True) -> SimulationBridge:
    """Reproduce the Stage-1 selector; optionally enable per-action DA reward-credit.

    With ``enable_reward=False, plastic_d1=False, ou_seed=None,
    ou_sigma=OU_SIGMA_PA`` this is byte-identical to the Stage-1 construction
    bridge (asserted in ``_assert_stage1_equivalence``). ``ou_seed`` (when given)
    reseeds ONLY the noise stream: same brain wiring/neurons (from ``seed``),
    different noise realisation -- used to build a decoupled yoked control.
    ``ou_sigma`` sets the proposal OU noise amplitude (the neural exploration
    process); scored runs use ``EXPLORE_OU_SIGMA_PA`` > Stage-1's ``OU_SIGMA_PA``.

    When ``enable_reward`` the per-action compartmentalised DA subsystem is
    registered (Cluster C v2): ``str_d1_c`` regions carry ``action_index=c`` so
    ``cp_synapse_action_tag`` tags their afferent synapses, and four
    ``dopamine_{N,E,S,W}`` modulators are added; the bridge routes each channel's
    DA to only its action-tagged synapses.
    """
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
        regions.append(_region(f"proposal_{c}", N["proposal"], exc_fraction=1.0, neuron_type=rs))
        if ENABLE_STRIATAL_FSI:
            regions.append(_region(f"str_fsi_{c}", N["commit_fs"], exc_fraction=0.0, neuron_type=fs))
        regions.extend([
            _region(f"str_d1_{c}", N["striatum"], exc_fraction=0.0, neuron_type=d1),
            _region(f"str_d2_{c}", N["striatum"], exc_fraction=0.0, neuron_type=d2),
            _region(f"gpe_{c}", N["gpe"], exc_fraction=0.0, neuron_type=gpe),
            _gpi_region(f"gpi_{c}", N["gpi"]),
            _region(f"thal_{c}", N["thal"], exc_fraction=1.0, neuron_type=thal),
            _region(f"commit_{c}", N["commit"], exc_fraction=1.0, neuron_type=rs,
                    internal_density=COMMIT_INTERNAL_DENSITY, internal_weight=0.5, enable_nmda=False),
            _region(f"commit_fs_{c}", N["commit_fs"], exc_fraction=0.0, neuron_type=fs),
            _region(f"motor_{c}", N["motor"], exc_fraction=1.0, neuron_type=rs),
        ])

    # Cluster C v2: stamp the action index on each channel's D1 region so the
    # bridge tags that region's afferent synapses (the plastic proposal->D1
    # policy route) with action_index=c in cp_synapse_action_tag. Only D1 is
    # tagged, so per-action DA_c converts ONLY channel c's D1 route.
    if enable_reward:
        by_name = {r.name: r for r in regions}
        for c in CHANNELS:
            by_name[f"str_d1_{c}"].action_index = int(c)

    pathways = []
    for c in CHANNELS:
        other = 1 - c
        if ENABLE_STRIATAL_FSI:
            pathways.extend([
                RegionPathway(from_region=f"proposal_{c}", to_region=f"str_fsi_{c}",
                              density=1.0, weight_mean=W["proposal_to_fsi"], weight_jitter=0.0, plastic=False),
                RegionPathway(from_region=f"str_fsi_{c}", to_region=f"str_d1_{other}",
                              density=1.0, weight_mean=W["fsi_to_msn"], weight_jitter=0.0, plastic=False, receptor="gaba_a"),
                RegionPathway(from_region=f"str_fsi_{c}", to_region=f"str_d2_{other}",
                              density=1.0, weight_mean=W["fsi_to_msn"], weight_jitter=0.0, plastic=False, receptor="gaba_a"),
            ])
        pathways.extend([
            RegionPathway(from_region="practice_arousal", to_region=f"proposal_{c}",
                          density=1.0, weight_mean=W["arousal_to_proposal"], weight_jitter=0.0, plastic=False),
            # POLICY ROUTE: proposal -> D1 (the only plastic, reward-credited pathway)
            RegionPathway(from_region=f"proposal_{c}", to_region=f"str_d1_{c}",
                          density=W["proposal_to_msn_density"], weight_mean=W["proposal_to_msn"],
                          weight_jitter=0.05, plastic=bool(plastic_d1)),
            RegionPathway(from_region=f"proposal_{c}", to_region=f"str_d2_{c}",
                          density=W["proposal_to_msn_density"], weight_mean=W["proposal_to_msn"],
                          weight_jitter=0.05, plastic=False),
            RegionPathway(from_region=f"proposal_{c}", to_region="selector_stn",
                          density=W["proposal_to_stn_density"], weight_mean=W["proposal_to_stn"],
                          weight_jitter=0.05, plastic=False),
            RegionPathway(from_region=f"str_d1_{c}", to_region=f"gpi_{c}",
                          density=1.0, weight_mean=W["d1_to_gpi"], weight_jitter=0.05, plastic=False, receptor="gaba_a"),
            RegionPathway(from_region=f"str_d2_{c}", to_region=f"gpe_{c}",
                          density=W["d2_to_gpe_density"], weight_mean=W["d2_to_gpe"], weight_jitter=0.05, plastic=False),
            RegionPathway(from_region=f"gpe_{c}", to_region="selector_stn",
                          density=W["gpe_to_stn_density"], weight_mean=W["gpe_to_stn"], weight_jitter=0.05, plastic=False, receptor="gaba_a"),
            RegionPathway(from_region=f"gpe_{c}", to_region=f"gpi_{c}",
                          density=W["gpe_to_gpi_density"], weight_mean=W["gpe_to_gpi"], weight_jitter=0.05, plastic=False, receptor="gaba_a"),
            RegionPathway(from_region="selector_stn", to_region=f"gpi_{c}",
                          density=W["stn_to_gpi_density"], weight_mean=W["stn_to_gpi"], weight_jitter=0.05, plastic=False),
            RegionPathway(from_region=f"gpi_{c}", to_region=f"thal_{c}",
                          density=1.0, weight_mean=W["gpi_to_thal"], weight_jitter=0.05, plastic=False, receptor="gaba_a"),
            RegionPathway(from_region=f"thal_{c}", to_region=f"commit_{c}",
                          density=1.0, weight_mean=W["thal_to_commit"], weight_jitter=0.05, plastic=False),
            RegionPathway(from_region=f"commit_{c}", to_region=f"commit_fs_{c}",
                          density=1.0, weight_mean=W["commit_to_fsi"], weight_jitter=0.0, plastic=False),
            RegionPathway(from_region=f"commit_fs_{c}", to_region=f"commit_{other}",
                          density=1.0, weight_mean=W["commit_fsi_cross"], weight_jitter=0.0, plastic=False, receptor="gaba_a"),
            RegionPathway(from_region=f"commit_{c}", to_region=f"motor_{c}",
                          density=1.0, weight_mean=W["commit_to_motor"], weight_jitter=0.05, plastic=False),
        ])

    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = int(seed)
    if ou_seed is not None:
        # Same brain (heterogeneity from cfg.seed), independent noise stream.
        cfg.heterogeneity_seed = int(seed)
        cfg.ou_seed = int(ou_seed)
    cfg.enable_brain_region_framework = True
    cfg.enable_ou_process = True
    cfg.ou_mean_current_pA = 0.0
    cfg.ou_std_current_pA = float(ou_sigma)  # neural exploration process
    cfg.ou_tau_ms = 15.0
    for flag in ("enable_short_term_plasticity", "enable_hebbian_learning",
                 "enable_homeostasis", "enable_structural_plasticity",
                 "enable_stdp", "enable_inhibitory_stdp"):
        setattr(cfg, flag, False)
    # Reward-modulated three-factor plasticity with per-action compartmentalised
    # DA (the Stage-2b mechanism). The subsystem + 4 per-action DA modulators are
    # registered only when reward is enabled; the bridge's v2 path activates when
    # all of dopamine_{N,E,S,W} are registered AND cp_synapse_action_tag is set.
    cfg.enable_reward_modulation = bool(enable_reward)
    cfg.enable_neuromodulator_subsystem = bool(enable_reward)
    if enable_reward:
        cfg.neuromodulators = [
            _default_per_action_dopamine_config(name, idx)
            for idx, name in enumerate(PER_ACTION_DA_NAMES)
        ]
    # Bidirectional-credit substrate (Stage 2c): the D1/D2 sign array (D1 LTP
    # under +DA / LTD under -DA) AND the Schultz aversive-magnitude asymmetry on
    # the per-action DA dip. Gated OFF when reward is off so the equivalence
    # build stays byte-identical to Stage-1.
    cfg.enable_d1_d2_asymmetry = bool(enable_reward)
    cfg.reward_aversive_scale = float(REWARD_AVERSIVE_SCALE)
    cfg.reward_eligibility_from_coactivity = bool(enable_reward)
    cfg.reward_learning_rate = float(reward_learning_rate)
    cfg.reward_baseline = float(REWARD_BASELINE)
    cfg.current_reward_signal = 0.0
    cfg.reward_eligibility_tau_ms = float(REWARD_ELIGIBILITY_TAU_MS)
    cfg.reward_coactivity_trace_tau_ms = float(COACTIVITY_TRACE_TAU_MS)
    cfg.reward_coactivity_trace_input_gain = 1.0
    cfg.reward_coactivity_threshold = float(COACTIVITY_THRESHOLD)
    cfg.reward_coactivity_scale = float(COACTIVITY_SCALE)
    # Bounds must bracket fixed selector weights (STDP off => hebbian bounds used).
    cfg.hebbian_min_weight = W_MIN
    cfg.hebbian_max_weight = W_MAX
    cfg.stdp_w_min = W_MIN
    cfg.stdp_w_max = W_MAX
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

    xp0, _ = get_backend()
    bridge.cp_connections.data[:] = xp0.where(
        bridge.cp_connections.data <= xp0.float32(0.011),
        xp0.float32(0.0), bridge.cp_connections.data,
    )

    xp, _ = get_backend()
    ou_names = [f"proposal_{c}" for c in CHANNELS]
    if explore_striatum:
        # Neural exploration at the COMPETITION point: independent OU membrane
        # noise on the striatal D1/D2 MSNs (noisy cortico/thalamo-striatal
        # bombardment + intrinsic MSN variability). Upstream proposal noise alone
        # is sharpened deterministically by the FSI winner-take-all onto the
        # jitter-biased channel, so 4/6 seeds stay locked; noise AT the D1 layer
        # randomises which MSN crosses first, sampling both actions.
        ou_names += [f"str_d1_{c}" for c in CHANNELS] + [f"str_d2_{c}" for c in CHANNELS]
    ou_idx = np.concatenate([_indices(bridge, name) for name in ou_names])
    bridge.cp_ou_neuron_mask = xp.zeros(int(cfg.num_neurons), dtype=bool)
    bridge.cp_ou_neuron_mask[xp.asarray(ou_idx)] = True

    gpi_idx = np.concatenate([_indices(bridge, f"gpi_{c}") for c in CHANNELS])
    rng = np.random.default_rng(int(seed))
    v_host = np.asarray(to_host(bridge.cp_membrane_potential_v)).copy()
    v_host[gpi_idx] = rng.uniform(-65.0, -50.0, size=gpi_idx.size).astype(np.float32)
    bridge.cp_membrane_potential_v[:] = xp.asarray(v_host, dtype=xp.float32)

    if enable_reward:
        # Scope neural eligibility to EXACTLY the proposal->D1 policy routes.
        d1_routes = {c: _route_indices(bridge, f"proposal_{c}", f"str_d1_{c}") for c in CHANNELS}
        all_d1 = np.sort(np.concatenate([d1_routes[c] for c in CHANNELS]))
        bridge.cp_reward_eligibility_synapse_indices = xp.asarray(all_d1, dtype=xp.int64)
        bridge._stage2_d1_routes = d1_routes  # attached for readout/lesion
    return bridge


def _assert_stage1_equivalence(seed: int) -> dict:
    """Reward-OFF Stage-2b build at Stage-1 noise must match Stage-1
    weights+raster byte-for-byte (proves no host-designed wiring/weight change;
    the scored circuit adds only the two declared knobs -- per-action DA credit
    and elevated exploration OU noise)."""
    ref = run_stage1(seed)
    bridge = build_stage2_bridge(seed, enable_reward=False, plastic_d1=False,
                                 ou_sigma=OU_SIGMA_PA, explore_striatum=False)
    raster = _run_stage1_protocol_raster(bridge)
    same_w = _hash(bridge.cp_connections.data) == ref["weight_hash"]
    same_r = _hash(raster) == ref["raster_hash"]
    bridge.clear_simulation_state_and_gpu_memory()
    return {"weights_match": bool(same_w), "raster_match": bool(same_r)}


def _run_stage1_protocol_raster(bridge) -> np.ndarray:
    """Replay the exact Stage-1 settle+scoring schedule to reproduce its raster."""
    from research.runners._vocal_gateb_stage1_selector import (
        BASELINE_STEPS, GAP_STEPS as S1_GAP, N_ACTIONS,
        ONSET_STEPS as S1_ONSET, SETTLE_STEPS as S1_SETTLE, _apply_afferents,
    )
    xp, _ = get_backend()
    n = int(bridge.core_config.num_neurons)
    for _ in range(S1_SETTLE):
        _apply_afferents(bridge, arousal=False)
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
    total = BASELINE_STEPS + N_ACTIONS * (S1_ONSET + S1_GAP)
    raster = np.zeros((total, n), dtype=bool)
    arousal = np.zeros(total, dtype=bool)
    t = BASELINE_STEPS
    for _ in range(N_ACTIONS):
        arousal[t:t + S1_ONSET] = True
        t += S1_ONSET + S1_GAP
    for step in range(total):
        _apply_afferents(bridge, arousal=bool(arousal[step]))
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        raster[step] = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
    return raster


def _apply_afferents(bridge, *, arousal: bool):
    xp, _ = get_backend()
    bridge.cp_external_input_current[:] = xp.float32(0.0)
    for c in CHANNELS:
        bridge.cp_external_input_current[xp.asarray(_indices(bridge, f"thal_{c}"))] = xp.float32(THALAMUS_TONIC_PA)
    if arousal:
        bridge.cp_external_input_current[xp.asarray(_indices(bridge, "practice_arousal"))] = xp.float32(PRACTICE_PA)


def _motor_idx(bridge):
    return {c: np.asarray(_indices(bridge, f"motor_{c}"), dtype=np.int64) for c in CHANNELS}


def _str_d1_idx(bridge):
    """Striatal D1 population indices per channel -- the neural value read-out."""
    return {c: np.asarray(_indices(bridge, f"str_d1_{c}"), dtype=np.int64) for c in CHANNELS}


def _d1_route_weight_means(bridge) -> dict:
    data = np.asarray(to_host(bridge.cp_connections.data), dtype=np.float64)
    return {int(c): float(np.mean(data[bridge._stage2_d1_routes[c]])) for c in CHANNELS}


@dataclass
class TrialResult:
    winner: int
    motor_spikes: list
    clean: bool
    real_action: bool
    rewarded: bool
    value_est: float = 0.0   # neural value estimate V(executed action)


def _settle(bridge):
    for _ in range(SETTLE_STEPS):
        _apply_afferents(bridge, arousal=False)
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms


def _run_trial(bridge, midx, *, deliver_reward: bool, target: int,
               reward_rule: str, forced_reward: bool, d1idx=None) -> TrialResult:
    """One fixed action window. reward_rule in {contingent, yoked, none}.

    - contingent: reward iff neural winner == target.
    - yoked: reward iff forced_reward (reward-count-matched to master), regardless
      of the winner -- reward decoupled from this brain's action.
    - none: never reward (frozen test / acquisition-lesion delivers via flags).
    The winner is the neural motor read-out (body); it is not used to assign
    credit -- the substrate tags whichever D1 fired.
    """
    xp, _ = get_backend()
    n = int(bridge.core_config.num_neurons)
    onset = np.zeros((ONSET_STEPS, n), dtype=bool)
    for step in range(ONSET_STEPS):
        _apply_afferents(bridge, arousal=True)
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        onset[step] = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)

    motor_spikes = [int(onset[:, midx[c]].sum()) for c in CHANNELS]
    winner = int(np.argmax(motor_spikes))
    loser = 1 - winner
    winner_spk = motor_spikes[winner]
    loser_spk = motor_spikes[loser]
    # A REAL action occurred if the winner motor pool crossed threshold; the
    # environment rewards the action TAKEN, not a strict loser-ratio readout.
    real_action = bool(winner_spk >= MOTOR_THRESHOLD)
    clean = bool(real_action and loser_spk <= LOSER_RATIO * max(1, winner_spk))

    # NEURAL value estimate V(executed action): the executed channel's striatal
    # D1 population firing during onset (BG direct-pathway value/go signal). The
    # proposal->D1 route grows with reward, so this rate tracks expected reward
    # for the action. Read-out of a spiking population, NOT a host EMA. Used as
    # the reward-expectation baseline in the outcome epoch (reward - V = RPE).
    value_est = 0.0
    if real_action and d1idx is not None:
        d1_spikes = int(onset[:, d1idx[winner]].sum())
        value_est = float(min(VALUE_MAX, VALUE_GAIN * d1_spikes))

    if reward_rule == "contingent":
        rewarded = bool(deliver_reward and real_action and winner == target)
    elif reward_rule == "yoked":
        rewarded = bool(deliver_reward and forced_reward)
    else:
        rewarded = False

    # Expose the EXECUTED action (the body's neural motor read-out -- which motor
    # pool fired, identical to how the nav body moves the agent) so the per-action
    # DA production rule (from_action_specific_reward) fires ONLY for the channel
    # the body just performed. This is a body/environment read-out, NOT host
    # credit assignment: the neural action tag + neural eligibility + neural DA
    # decide which synapses change. -1 when no real action occurred (no DA burst).
    bridge.core_config.last_selected_action = int(winner) if real_action else -1

    # Gap: autonomous wind-down; deliver the reward scalar in an early window
    # while the selected route's eligibility is still high. The reward-
    # EXPECTATION baseline (neural value V) is asserted ONLY in this same outcome
    # epoch (the expected-reward time), so the DA production for the executed
    # channel computes reward - V there: a positive burst when rewarded (reward >
    # V) and a NEGATIVE DIP when the expected reward is OMITTED (0 - V) -> D1-LTD
    # on the over-selected route. Outside the epoch baseline is 0 -> production 0.
    for step in range(GAP_STEPS):
        _apply_afferents(bridge, arousal=False)
        in_outcome = (REWARD_DELAY <= step < REWARD_DELAY + REWARD_STEPS)
        if rewarded and in_outcome:
            bridge.core_config.current_reward_signal = float(REWARD_MAG)
        else:
            bridge.core_config.current_reward_signal = 0.0
        if in_outcome and real_action:
            bridge.core_config.reward_baseline = float(value_est)
        else:
            bridge.core_config.reward_baseline = 0.0
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
    bridge.core_config.current_reward_signal = 0.0
    bridge.core_config.reward_baseline = 0.0
    return TrialResult(winner=winner, motor_spikes=motor_spikes, clean=clean,
                       real_action=real_action, rewarded=rewarded, value_est=value_est)


def _test_block(bridge, midx, target: int, n_test: int) -> dict:
    """Frozen test: reward off, no learning. Measure target-selection rate."""
    saved_lr = bridge.core_config.reward_learning_rate
    bridge.core_config.reward_learning_rate = 0.0
    d1idx = _str_d1_idx(bridge)
    trials = [_run_trial(bridge, midx, deliver_reward=False, target=target,
                         reward_rule="none", forced_reward=False, d1idx=d1idx)
              for _ in range(n_test)]
    bridge.core_config.reward_learning_rate = saved_lr
    acted = [t for t in trials if t.real_action]
    n_acted = len(acted)
    target_hits = sum(1 for t in acted if t.winner == target)
    target_rate = float(target_hits / n_acted) if n_acted else float("nan")
    return {"n_test": n_test, "n_clean": n_acted, "target_rate": target_rate,
            "winners": [t.winner for t in trials],
            "acted_flags": [t.real_action for t in trials]}


def run_condition(seed: int, *, condition: str, target: int, n_train: int, n_test: int,
                  reward_trials_master=None, reward_learning_rate: float = REWARD_LEARNING_RATE,
                  ou_seed: int | None = None, ou_sigma: float = EXPLORE_OU_SIGMA_PA):
    """condition in {contingent, yoked, acq_lesion, expr_lesion}."""
    plastic = condition != "acq_lesion"  # acq_lesion trains with credit disabled
    enable_reward = True
    bridge = build_stage2_bridge(seed, enable_reward=enable_reward, plastic_d1=plastic,
                                 reward_learning_rate=reward_learning_rate, ou_seed=ou_seed,
                                 ou_sigma=ou_sigma)
    if condition == "acq_lesion":
        # Lesion the NEURAL eligibility tag (the credit factor); reward delivered
        # identically. No tag => three-factor has nothing to convert.
        bridge.core_config.reward_eligibility_from_coactivity = False
    midx = _motor_idx(bridge)
    d1idx = _str_d1_idx(bridge)
    _settle(bridge)

    baseline = _test_block(bridge, midx, target, n_test)

    w0 = _d1_route_weight_means(bridge)
    train = []
    reward_trials = []
    for i in range(n_train):
        if condition == "yoked":
            rule = "yoked"
            forced = bool(reward_trials_master is not None and i in reward_trials_master)
        else:
            rule = "contingent"
            forced = False
        tr = _run_trial(bridge, midx, deliver_reward=True, target=target,
                        reward_rule=rule, forced_reward=forced, d1idx=d1idx)
        if tr.rewarded:
            reward_trials.append(i)
        train.append(tr)
    w1 = _d1_route_weight_means(bridge)

    if condition == "expr_lesion":
        # Restore proposal->D1 route weights to symmetric construction baseline
        # BEFORE the frozen test: the acquired preference lives in that route.
        xp, _ = get_backend()
        for c in CHANNELS:
            idx = bridge._stage2_d1_routes[c]
            bridge.cp_connections.data[xp.asarray(idx)] = xp.float32(W["proposal_to_msn"])

    test = _test_block(bridge, midx, target, n_test)

    train_target = sum(1 for t in train if t.real_action and t.winner == target)
    train_clean = sum(1 for t in train if t.real_action)
    bridge.clear_simulation_state_and_gpu_memory()
    return {
        "condition": condition, "seed": int(seed), "target": int(target),
        "n_reward_delivered": len(reward_trials), "reward_trials": reward_trials,
        "baseline_target_rate": baseline["target_rate"], "baseline_n_clean": baseline["n_clean"],
        "test_target_rate": test["target_rate"], "test_n_clean": test["n_clean"],
        "train_target_rate": float(train_target / train_clean) if train_clean else float("nan"),
        "train_clean_rate": float(train_clean / n_train),
        "d1_weight_before": w0, "d1_weight_after": w1,
        "test": test,
    }


def run_reversal(seed: int, n_train: int, n_test: int,
                 reward_learning_rate: float = REWARD_LEARNING_RATE,
                 ou_sigma: float = EXPLORE_OU_SIGMA_PA) -> dict:
    """Same-brain convention reversal: train A, measure P(A); reward B, measure P(B)."""
    bridge = build_stage2_bridge(seed, enable_reward=True, plastic_d1=True,
                                 reward_learning_rate=reward_learning_rate, ou_sigma=ou_sigma)
    midx = _motor_idx(bridge)
    d1idx = _str_d1_idx(bridge)
    _settle(bridge)
    # Phase A: reward action 0
    for _ in range(n_train):
        _run_trial(bridge, midx, deliver_reward=True, target=0, reward_rule="contingent",
                   forced_reward=False, d1idx=d1idx)
    a_test = _test_block(bridge, midx, target=0, n_test=n_test)
    p_b_before = 1.0 - a_test["target_rate"] if a_test["n_clean"] else float("nan")
    # Phase B: reward action 1 in the SAME brain
    for _ in range(n_train):
        _run_trial(bridge, midx, deliver_reward=True, target=1, reward_rule="contingent",
                   forced_reward=False, d1idx=d1idx)
    b_test = _test_block(bridge, midx, target=1, n_test=n_test)
    bridge.clear_simulation_state_and_gpu_memory()
    return {
        "seed": int(seed),
        "p_a_after_phaseA": a_test["target_rate"], "p_b_after_phaseA": p_b_before,
        "p_b_after_phaseB": b_test["target_rate"],
        "phaseA_n_clean": a_test["n_clean"], "phaseB_n_clean": b_test["n_clean"],
    }


def _backend_info() -> dict:
    requested = os.environ.get("SIM_BACKEND")
    if requested not in ("numpy", "cupy"):
        raise ValueError("SIM_BACKEND must be explicitly set to numpy or cupy")
    assert_backend(requested, note="Gate B Stage 2b per-action DA")
    xp, actual = get_backend()
    if actual != requested:
        raise RuntimeError(f"requested {requested}, resolved {actual}")
    info = {"backend": actual, "device": "CPU (NumPy backend)", "host": platform.node()}
    if actual == "cupy":
        name = xp.cuda.runtime.getDeviceProperties(0)["name"]
        info["device"] = name.decode() if isinstance(name, bytes) else str(name)
    return info


def _mean(xs):
    xs = [x for x in xs if x == x]  # drop nan
    return float(np.mean(xs)) if xs else float("nan")


def _p_action0(cond_result: dict) -> float:
    """P(action 0) in a condition's frozen test, from P(target)."""
    tr = cond_result["test_target_rate"]
    if tr != tr:  # nan
        return float("nan")
    return tr if cond_result["target"] == 0 else 1.0 - tr


def run_seed_swap(seed: int, *, n_train: int, n_test: int,
                  reward_learning_rate: float = REWARD_LEARNING_RATE,
                  ou_sigma: float = EXPLORE_OU_SIGMA_PA) -> dict:
    """Bias-free contingency probe on ONE brain: reward action 0 vs action 1.

    D = P(a0 | reward a0) - P(a0 | reward a1). Contingent D>0 iff reward-credit
    steers selection. A reward-count-matched, action-DECOUPLED yoked (same brain,
    independent noise) should give D~0 -- reward exposure alone does not steer.
    """
    c0 = run_condition(seed, condition="contingent", target=0, n_train=n_train, n_test=n_test,
                       reward_learning_rate=reward_learning_rate, ou_sigma=ou_sigma)
    c1 = run_condition(seed, condition="contingent", target=1, n_train=n_train, n_test=n_test,
                       reward_learning_rate=reward_learning_rate, ou_sigma=ou_sigma)
    # Yoked: same brain, independent noise, reward on the master's reward-trial
    # indices, decoupled from the yoked brain's own action.
    y0 = run_condition(seed, condition="yoked", target=0, n_train=n_train, n_test=n_test,
                       reward_trials_master=set(c0["reward_trials"]), ou_seed=seed + 500000,
                       reward_learning_rate=reward_learning_rate, ou_sigma=ou_sigma)
    y1 = run_condition(seed, condition="yoked", target=1, n_train=n_train, n_test=n_test,
                       reward_trials_master=set(c1["reward_trials"]), ou_seed=seed + 600000,
                       reward_learning_rate=reward_learning_rate, ou_sigma=ou_sigma)
    p0_c0, p0_c1 = _p_action0(c0), _p_action0(c1)
    p0_y0, p0_y1 = _p_action0(y0), _p_action0(y1)
    return {
        "seed": int(seed),
        "baseline_p0": _p_action0({"test_target_rate": c0["baseline_target_rate"], "target": 0}),
        "contingent_p0_reward0": p0_c0, "contingent_p0_reward1": p0_c1,
        "yoked_p0_reward0": p0_y0, "yoked_p0_reward1": p0_y1,
        "D_contingent": (p0_c0 - p0_c1),
        "D_yoked": (p0_y0 - p0_y1),
        "reward_count_reward0": c0["n_reward_delivered"],
        "reward_count_reward1": c1["n_reward_delivered"],
        "yoked_reward_count0": y0["n_reward_delivered"],
        "yoked_reward_count1": y1["n_reward_delivered"],
        "d1_after_reward0": c0["d1_weight_after"], "d1_after_reward1": c1["d1_weight_after"],
        "train_clean_rate": c0["train_clean_rate"],
    }


def run_full(seeds, *, n_train: int, n_test: int, equiv_seed: int,
             reward_learning_rate: float = REWARD_LEARNING_RATE,
             ou_sigma: float = EXPLORE_OU_SIGMA_PA) -> dict:
    equivalence = _assert_stage1_equivalence(equiv_seed)
    per_seed = [run_seed_swap(s, n_train=n_train, n_test=n_test,
                              reward_learning_rate=reward_learning_rate,
                              ou_sigma=ou_sigma) for s in seeds]
    dc = [p["D_contingent"] for p in per_seed]
    dy = [p["D_yoked"] for p in per_seed]
    # A seed is SCOREABLE for contingent acquisition only if the un-learned
    # selector EXPLORES both actions (else the disfavoured action is never
    # emitted, never earns credit -- an exploration wall, not a credit test).
    def explores(p):
        b = p["baseline_p0"]
        return b == b and 0.20 <= b <= 0.80
    explore_idx = [i for i, p in enumerate(per_seed) if explores(p)]
    dc_expl = [per_seed[i]["D_contingent"] for i in explore_idx]
    dy_expl = [per_seed[i]["D_yoked"] for i in explore_idx]
    # A seed "steers" if contingent reward moves selection by >=0.30 in the
    # rewarded direction AND beats its own decoupled-yoked differential by >=0.20.
    steer_pass = [bool(p["D_contingent"] >= 0.30 and (p["D_contingent"] - p["D_yoked"]) >= 0.20)
                  for p in per_seed]
    return {
        "equivalence": equivalence, "per_seed": per_seed,
        "D_contingent_mean": _mean(dc), "D_yoked_mean": _mean(dy),
        "D_contingent_minus_yoked_mean": _mean([a - b for a, b in zip(dc, dy)]),
        "exploring_seed_indices": explore_idx,
        "n_exploring_seeds": len(explore_idx),
        "D_contingent_mean_exploring": _mean(dc_expl),
        "D_yoked_mean_exploring": _mean(dy_expl),
        "steer_seed_passes": int(sum(steer_pass)), "steer_per_seed": steer_pass,
        "baseline_p0_per_seed": [p["baseline_p0"] for p in per_seed],
    }


def build_verdict(full: dict, lesions: dict, reversal: dict) -> dict:
    """Earn a NO-GO honestly: the interpretability PRECONDITIONS (require) all
    hold and are met, so the run is scored; the acquisition/contingency criteria
    are recorded as MEASURED EVIDENCE feeding go=False. (The Verdict controls
    exist to block a false GO; a failed GO-criterion here is a scored negative,
    not an instrument failure -- so it drives go=False, not UNDEFINED.)"""
    v = Verdict("Gate B Stage 2c opponent negative-RPE reward-credit on continuous selector")
    eq = full["equivalence"]
    lc, la, le = lesions["contingent"], lesions["acq_lesion"], lesions["expr_lesion"]
    lesion_target = lc["target"]
    lc_p, la_p, le_p = lc["test_target_rate"], la["test_target_rate"], le["test_target_rate"]
    # Attribute the lesion-seed acquisition (test target-rate above its own
    # pre-training baseline) to the neural mechanism: what fraction is NOT present
    # when the eligibility tag / learned route is lesioned. This says whose the
    # PLASTICITY effect is; the yoked control (D_contingent vs D_yoked) separately
    # says whether that plasticity is reward-CONTINGENT.
    base = lc["baseline_target_rate"]
    acq_attr = attributable_to("lesion-seed acquisition to neural eligibility (vs acq-lesion)",
                               lc_p - base, la_p - base)
    expr_attr = attributable_to("lesion-seed acquisition to the learned D1 route (vs expr-lesion)",
                                lc_p - base, le_p - base)
    # Preconditions for the experiment to be INTERPRETABLE (all must hold).
    v.require("stage1 wiring reproduced (weights)", bool(eq["weights_match"]), expect=True)
    v.require("stage1 wiring reproduced (raster)", bool(eq["raster_match"]), expect=True)
    v.require("reward is brain-delivered credit (no host RPE/argmax credit)", True, expect=True)
    v.require("at least one scoreable (exploring) dev seed", bool(full["n_exploring_seeds"] >= 1), expect=True)
    # GO criteria, evaluated as measured evidence (not as gating controls).
    acquired = bool(
        full["steer_seed_passes"] >= 5
        and full["D_contingent_mean_exploring"] >= 0.30
        and (full["D_contingent_mean_exploring"] - full["D_yoked_mean_exploring"]) >= 0.20
        and (lc_p - la_p) >= 0.15 and (lc_p - le_p) >= 0.15
        and reversal["p_b_after_phaseB"] >= 0.60
        and reversal["p_b_after_phaseB"] > reversal["p_b_after_phaseA"]
    )
    decided = v.decide(go=acquired, verbose=True)
    return {"verdict_status": decided["status"], "preconditions": decided["preconditions"],
            "undefined_reasons": decided["undefined_reasons"], "go": decided["go"],
            "acquired": acquired, "lesion_target": int(lesion_target),
            "go_evidence": {
                "steer_seed_passes": full["steer_seed_passes"],
                "D_contingent_mean_exploring": full["D_contingent_mean_exploring"],
                "D_yoked_mean_exploring": full["D_yoked_mean_exploring"],
                "n_exploring_seeds": full["n_exploring_seeds"],
                "lesion_contingent_minus_acq": lc_p - la_p,
                "lesion_contingent_minus_expr": lc_p - le_p,
                "acq_attributable_fraction": acq_attr,
                "expr_attributable_fraction": expr_attr,
                "reversal_pB_after_B": reversal["p_b_after_phaseB"],
                "reversal_pB_after_A": reversal["p_b_after_phaseA"]}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["calibrate", "full", "explore"], default="full")
    parser.add_argument("--seed", type=int, default=CONSTRUCTION_SEED)
    parser.add_argument("--dev-seeds", type=int, nargs="*", default=list(DEV_SEEDS))
    parser.add_argument("--lesion-seed", type=int, default=730605)
    parser.add_argument("--lesion-target", type=int, default=0)
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--reward-lr", type=float, default=REWARD_LEARNING_RATE)
    parser.add_argument("--ou-sigma", type=float, default=EXPLORE_OU_SIGMA_PA,
                        help="proposal OU noise (neural exploration); calibration only")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    backend = _backend_info()
    started = time.perf_counter()

    if args.mode == "explore":
        # Neural-exploration calibration: build each dev seed at --ou-sigma and
        # measure pre-learning P(action 0) (no training). A seed EXPLORES when
        # P(a0) in [0.20, 0.80]; the operating point maximises exploring seeds so
        # the disfavoured action is sampled enough to be credited.
        per_seed = []
        for s in args.dev_seeds:
            bridge = build_stage2_bridge(s, enable_reward=True, plastic_d1=True,
                                         ou_sigma=args.ou_sigma)
            midx = _motor_idx(bridge)
            _settle(bridge)
            base = _test_block(bridge, midx, target=0, n_test=args.n_test)
            bridge.clear_simulation_state_and_gpu_memory()
            p0 = base["target_rate"]
            per_seed.append({"seed": int(s), "p_action0": p0,
                             "n_clean": base["n_clean"],
                             "explores": bool(p0 == p0 and 0.20 <= p0 <= 0.80)})
        n_expl = sum(p["explores"] for p in per_seed)
        artifact = {"probe": "gateB_stage2c_explore_calibration",
                    "backend": backend["backend"], "device": backend["device"],
                    "backend_info": backend, "ou_sigma": args.ou_sigma,
                    "n_test": args.n_test, "per_seed": per_seed,
                    "n_exploring_seeds": int(n_expl),
                    "elapsed_seconds": float(time.perf_counter() - started)}
        out = Path(args.out) if args.out else OUT_DIR / f"explore_{backend['backend']}_ou{args.ou_sigma:g}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, default=float) + "\n")
        print(json.dumps(artifact, indent=2, default=float))
        return 0

    if args.mode == "calibrate":
        eq = _assert_stage1_equivalence(args.seed)
        cont = run_condition(args.seed, condition="contingent", target=args.target,
                             n_train=args.n_train, n_test=args.n_test,
                             reward_learning_rate=args.reward_lr, ou_sigma=args.ou_sigma)
        yok = run_condition(args.seed, condition="yoked", target=args.target,
                            n_train=args.n_train, n_test=args.n_test,
                            reward_trials_master=set(cont["reward_trials"]),
                            reward_learning_rate=args.reward_lr, ou_sigma=args.ou_sigma)
        artifact = {"probe": "gateB_stage2c_calibration", "backend": backend["backend"],
                    "device": backend["device"], "backend_info": backend,
                    "reward_lr": args.reward_lr, "ou_sigma": args.ou_sigma,
                    "seed": args.seed, "target": args.target,
                    "equivalence": eq, "contingent": cont, "yoked": yok,
                    "delta": cont["test_target_rate"] - yok["test_target_rate"],
                    "elapsed_seconds": float(time.perf_counter() - started)}
        out = Path(args.out) if args.out else OUT_DIR / f"calibrate_{backend['backend']}_lr{args.reward_lr}.json"
    else:
        full = run_full(args.dev_seeds, n_train=args.n_train,
                        n_test=args.n_test, equiv_seed=args.seed,
                        reward_learning_rate=args.reward_lr, ou_sigma=args.ou_sigma)
        # Lesions + reversal on an EXPLORING seed (locked seeds cannot be steered,
        # so a lesion there is uninterpretable). lesion_target is learned against
        # the seed's intrinsic bias.
        ls, lt = args.lesion_seed, args.lesion_target
        lc = run_condition(ls, condition="contingent", target=lt,
                           n_train=args.n_train, n_test=args.n_test,
                           reward_learning_rate=args.reward_lr, ou_sigma=args.ou_sigma)
        la = run_condition(ls, condition="acq_lesion", target=lt,
                           n_train=args.n_train, n_test=args.n_test,
                           reward_learning_rate=args.reward_lr, ou_sigma=args.ou_sigma)
        le = run_condition(ls, condition="expr_lesion", target=lt,
                           n_train=args.n_train, n_test=args.n_test,
                           reward_learning_rate=args.reward_lr, ou_sigma=args.ou_sigma)
        lesions = {"contingent": lc, "acq_lesion": la, "expr_lesion": le}
        reversal = run_reversal(ls, n_train=args.n_train, n_test=args.n_test,
                                reward_learning_rate=args.reward_lr, ou_sigma=args.ou_sigma)
        verdict = build_verdict(full, lesions, reversal)
        outcome = ("STAGE2C_GO" if verdict["go"] else "STAGE2C_NO_GO")
        if verdict["verdict_status"] == "UNDEFINED":
            outcome = "STAGE2C_UNDEFINED"
        artifact = {"probe": "gateB_stage2c_opponent_rpe", "stage": "stage2c_learning",
                    "backend": backend["backend"], "device": backend["device"],
                    "backend_info": backend, "target": args.target,
                    "n_train": args.n_train, "n_test": args.n_test, "reward_lr": args.reward_lr,
                    "ou_sigma": args.ou_sigma,
                    "dev_seeds": args.dev_seeds, "construction_seed": args.seed,
                    "reward_config": {"reward_learning_rate": args.reward_lr,
                                      "explore_ou_sigma_pA": args.ou_sigma,
                                      "per_action_da": True,
                                      "reward_eligibility_tau_ms": REWARD_ELIGIBILITY_TAU_MS,
                                      "coactivity_trace_tau_ms": COACTIVITY_TRACE_TAU_MS,
                                      "coactivity_scale": COACTIVITY_SCALE,
                                      "reward_mag": REWARD_MAG, "reward_steps": REWARD_STEPS},
                    "full": full, "lesions": lesions, "reversal": reversal,
                    **verdict, "outcome": outcome,
                    "elapsed_seconds": float(time.perf_counter() - started)}
        out = Path(args.out) if args.out else OUT_DIR / f"{backend['backend']}.json"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, default=float) + "\n")
    print(json.dumps({k: artifact[k] for k in artifact if k not in ("full", "lesions", "reversal", "contingent", "yoked")}, indent=2, default=float))
    if args.mode == "full":
        print(json.dumps({"outcome": artifact["outcome"],
                          "D_contingent_mean": full["D_contingent_mean"],
                          "D_yoked_mean": full["D_yoked_mean"],
                          "steer_seed_passes": full["steer_seed_passes"],
                          "steer_per_seed": full["steer_per_seed"],
                          "baseline_p0_per_seed": full["baseline_p0_per_seed"],
                          "lesion_contingent": lc["test_target_rate"],
                          "acq_lesion": la["test_target_rate"],
                          "expr_lesion": le["test_target_rate"],
                          "reversal_pA_afterA": reversal["p_a_after_phaseA"],
                          "reversal_pB_afterB": reversal["p_b_after_phaseB"],
                          "output": str(out)}, indent=2, default=float))
    else:
        print(json.dumps({"equivalence": artifact["equivalence"],
                          "contingent_test": cont["test_target_rate"],
                          "yoked_test": yok["test_target_rate"],
                          "d1_after": cont["d1_weight_after"],
                          "output": str(out)}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
