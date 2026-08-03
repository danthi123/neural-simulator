"""Calibration-only Gate B v3 for neural vocal action credit.

V2 learned the rewarded action, but yoked reward produced an arbitrary
saturated policy, and omission did not lower dopamine. V3 keeps the
executed-action-local eligibility and the v2 value-to-SNc expectation route.
It adds two circuits:

* action-local striatal fast-spiking inhibition keeps each value population in
  a graded operating range; and
* an expectation-gated lateral-habenula (LHb) -> RMTg -> SNc path turns an
  expected but absent reward into a neural negative prediction error.

The host supplies a shared cue, generic outcome timing, and whether a sensory
reward occurred. It does not stimulate an action channel, assign eligibility,
set dopamine, or apply a channel-specific weight update.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from research.runners._vocal_action_credit_gate import (
    ACTION_COLLATERAL_GATE,
    CREDIT_CUE,
    CREDIT_PLASTICITY_GATE,
    DOPAMINE_PATH_GATE,
    OUTCOME_ONSET,
    REWARD_US,
    SNC,
    VALUE_PLASTICITY_GATE,
    VALUE_TO_SNC_GATE,
    CALIBRATION_SEEDS as V1_V2_CALIBRATION_SEEDS,
    DEVELOPMENT_SEEDS as V1_V2_DEVELOPMENT_SEEDS,
    HELD_OUT_SEEDS as V1_V2_HELD_OUT_SEEDS,
    CreditConfig,
    CreditRoutes,
    _actor,
    _core_updates,
    _credit_pathways,
    _credit_regions,
    _cue_led_rate,
    _eligibility_means,
    _preference,
    _route_change,
    _route_synapses,
    _route_weight_means,
    _set_learning_gates,
    _set_trial_current,
    _value,
    _yoked_schedule,
    credit_config,
)
from research.runners._vocal_action_selector_gate import (
    CHANNELS,
    _indices,
    _region,
    _set_equal_tonic_current,
    _step,
    build_selector_bridge,
    selector_config,
)
from sim.backend import get_backend, to_host
from sim.enums import NeuronType
from sim.regions import RegionPathway
from tools.verdict import UNDEFINED, Verdict


VALUE_FS_PREFIX = "vocal_credit_value_fs_"
OMISSION_GATE = "vocal_credit_omission_gate"
LATERAL_HABENULA = "vocal_credit_lateral_habenula"
REWARD_VETO = "vocal_credit_reward_veto"
RMTG = "vocal_credit_rmtg"

CRITIC_NORMALIZATION_GATE = "vocal_credit_local_value_normalization"
EXPECTATION_TO_OMISSION_GATE = "vocal_credit_expectation_to_omission"
OMISSION_PATH_GATE = "vocal_credit_lhb_rmtg_omission_path"
REWARD_VETO_GATE = "vocal_credit_reward_veto_path"

OPEN_PHASES = ("calibration",)
CALIBRATION_SEEDS = (401, 409)
DEVELOPMENT_SEEDS = (419, 421, 431, 433)
HELD_OUT_SEEDS = (439, 443)
SMOKE_SEED = 0
OTHER_LANE_FORMAL_SEEDS = (
    224, 225, 226, 227,  # visual calibration/development
    228, 229, 230, 231,  # replay calibration/development
    232, 233, 234, 235,  # source calibration/development
    322, 323, 324, 325,  # visual held-out
    326, 327, 328, 329,  # replay held-out
    330, 331, 332, 333,  # source held-out
)

MIN_DA_BURST = 0.001
MIN_DA_DIP = 0.001
MIN_LESION_DA_DELTA = 0.0005
MIN_NORMALIZATION_EFFECT = 1.20

YOKED_MODES = {"yoked", "critic_lesion", "omission_path_lesion"}
CONTINGENT_MODES = {
    "contingent",
    "collateral_lesion",
    "da_lesion",
    "normalization_lesion",
}
CONTROL_MODES = (
    "yoked",
    "collateral_lesion",
    "da_lesion",
    "critic_lesion",
    "omission_path_lesion",
    "normalization_lesion",
)


@dataclass(frozen=True)
class VocalCreditConfigV3(CreditConfig):
    """V2 operating point plus neural normalization and omission circuitry."""

    enable_action_value_critic: bool = True
    snc_tonic_pA: float = 350.0
    snc_tonic_calibration_steps: int = 1000
    reward_to_snc_weight: float = 100.0
    gabab_propagation_strength: float = 0.00004
    reward_steps: int = 40
    cue_pA: float = 6000.0
    actor_plasticity_gain: float = 0.3
    dopamine_decay_tau_ms: float = 30.0
    dopamine_window_ms: float = 20.0
    dopamine_sensitivity: float = 40.0

    n_value_fs: int = 12
    n_omission_gate: int = 20
    n_lhb: int = 24
    n_reward_veto: int = 16
    n_rmtg: int = 20
    motor_to_value_fs_weight: float = 20.0
    value_fs_to_value_weight: float = 12.0
    value_to_omission_gate_weight: float = 8.0
    omission_gate_tonic_pA: float = 350.0
    omission_gate_to_lhb_weight: float = 16.0
    outcome_to_lhb_weight: float = 16.0
    reward_to_veto_weight: float = 20.0
    reward_veto_to_lhb_weight: float = 20.0
    lhb_to_rmtg_weight: float = 18.0
    rmtg_to_snc_weight: float = 18.0


def v3_config() -> VocalCreditConfigV3:
    return VocalCreditConfigV3()


def _value_fs(channel: int) -> str:
    return f"{VALUE_FS_PREFIX}{int(channel)}"


def _v3_regions(config: VocalCreditConfigV3):
    fs = NeuronType.IZH2007_FS_CORTICAL_INTERNEURON
    gpe = NeuronType.IZH2007_GPE_PACEMAKER
    rs = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL
    regions = list(_credit_regions(config))
    regions.extend(
        _region(
            _value_fs(channel),
            config.n_value_fs,
            exc_fraction=0.0,
            neuron_type=fs,
        )
        for channel in CHANNELS
    )
    regions.extend([
        _region(
            OMISSION_GATE,
            config.n_omission_gate,
            exc_fraction=0.0,
            neuron_type=gpe,
        ),
        _region(
            LATERAL_HABENULA,
            config.n_lhb,
            exc_fraction=1.0,
            neuron_type=rs,
        ),
        _region(
            REWARD_VETO,
            config.n_reward_veto,
            exc_fraction=0.0,
            neuron_type=fs,
        ),
        _region(
            RMTG,
            config.n_rmtg,
            exc_fraction=0.0,
            neuron_type=fs,
        ),
    ])
    return regions


def _v3_pathways(config: VocalCreditConfigV3):
    pathways = list(_credit_pathways(config))
    for channel in CHANNELS:
        pathways.extend([
            RegionPathway(
                from_region=f"motor_{channel}",
                to_region=_value_fs(channel),
                density=1.0,
                weight_mean=config.motor_to_value_fs_weight,
                weight_jitter=0.0,
                plastic=False,
                transmission_gate=CRITIC_NORMALIZATION_GATE,
            ),
            RegionPathway(
                from_region=_value_fs(channel),
                to_region=_value(channel),
                density=1.0,
                weight_mean=config.value_fs_to_value_weight,
                weight_jitter=0.0,
                plastic=False,
                receptor="gaba_a",
                transmission_gate=CRITIC_NORMALIZATION_GATE,
            ),
            RegionPathway(
                from_region=_value(channel),
                to_region=OMISSION_GATE,
                density=0.60,
                weight_mean=config.value_to_omission_gate_weight,
                weight_jitter=0.05,
                plastic=False,
                receptor="gaba_b",
                transmission_gate=EXPECTATION_TO_OMISSION_GATE,
            ),
        ])
    pathways.extend([
        RegionPathway(
            from_region=OMISSION_GATE,
            to_region=LATERAL_HABENULA,
            density=0.70,
            weight_mean=config.omission_gate_to_lhb_weight,
            weight_jitter=0.05,
            plastic=False,
            receptor="gaba_a",
        ),
        RegionPathway(
            from_region=OUTCOME_ONSET,
            to_region=LATERAL_HABENULA,
            density=0.70,
            weight_mean=config.outcome_to_lhb_weight,
            weight_jitter=0.05,
            plastic=False,
        ),
        RegionPathway(
            from_region=REWARD_US,
            to_region=REWARD_VETO,
            density=0.70,
            weight_mean=config.reward_to_veto_weight,
            weight_jitter=0.05,
            plastic=False,
            transmission_gate=REWARD_VETO_GATE,
        ),
        RegionPathway(
            from_region=REWARD_VETO,
            to_region=LATERAL_HABENULA,
            density=0.70,
            weight_mean=config.reward_veto_to_lhb_weight,
            weight_jitter=0.05,
            plastic=False,
            receptor="gaba_a",
            transmission_gate=REWARD_VETO_GATE,
        ),
        RegionPathway(
            from_region=LATERAL_HABENULA,
            to_region=RMTG,
            density=0.70,
            weight_mean=config.lhb_to_rmtg_weight,
            weight_jitter=0.05,
            plastic=False,
            transmission_gate=OMISSION_PATH_GATE,
        ),
        RegionPathway(
            from_region=RMTG,
            to_region=SNC,
            density=0.70,
            weight_mean=config.rmtg_to_snc_weight,
            weight_jitter=0.05,
            plastic=False,
            receptor="gaba_a",
            transmission_gate=OMISSION_PATH_GATE,
        ),
    ])
    return pathways


def build_v3_bridge(seed: int, config: VocalCreditConfigV3 | None = None):
    config = config or v3_config()
    bridge = build_selector_bridge(
        int(seed),
        selector_config("v2"),
        extra_regions=_v3_regions(config),
        extra_pathways=_v3_pathways(config),
        core_config_updates=_core_updates(config),
    )
    xp, _ = get_backend()
    actor_routes = {
        channel: _route_synapses(bridge, CREDIT_CUE, _actor(channel))
        for channel in CHANNELS
    }
    value_routes = {
        channel: _route_synapses(
            bridge, f"motor_{channel}", _value(channel)
        )
        for channel in CHANNELS
    }
    routes = CreditRoutes(actor=actor_routes, value=value_routes)
    bridge.set_global_plasticity_gain(0.0)
    _set_learning_gates(
        bridge, config, actor_enabled=True, value_enabled=True,
    )
    bridge.cp_reward_eligibility_synapse_indices = xp.asarray(
        routes.all_indices(), dtype=xp.int64
    )
    return bridge, routes


def validate_phase_seeds(phase: str, seeds) -> tuple[int, ...]:
    phase = str(phase)
    checked = tuple(int(seed) for seed in seeds)
    if phase not in OPEN_PHASES:
        raise ValueError(
            f"Gate B v3 phase {phase!r} is locked; open phases={OPEN_PHASES}"
        )
    if not checked:
        raise ValueError("at least one calibration seed is required")
    invalid = [seed for seed in checked if seed not in CALIBRATION_SEEDS]
    if invalid:
        raise ValueError(
            f"Gate B v3 accepts calibration seeds {CALIBRATION_SEEDS} only; "
            f"development={DEVELOPMENT_SEEDS}, held_out={HELD_OUT_SEEDS}, "
            f"rejected={invalid}"
        )
    return checked


def _pathway_matches(
    pathway,
    source: str,
    target: str,
    *,
    receptor: str | None = None,
    gate: str | None = None,
) -> bool:
    if pathway.from_region != source or pathway.to_region != target:
        return False
    if receptor is not None and pathway.receptor != receptor:
        return False
    if gate is not None and pathway.transmission_gate != gate:
        return False
    return True


def _structural_preconditions(
    bridge,
    routes: CreditRoutes,
    config: VocalCreditConfigV3,
) -> dict[str, bool]:
    pathways = list(bridge.region_manager.pathways())
    region_names = {region.name for region in bridge.region_manager.regions()}
    expected_regions = {
        CREDIT_CUE,
        OUTCOME_ONSET,
        REWARD_US,
        SNC,
        OMISSION_GATE,
        LATERAL_HABENULA,
        REWARD_VETO,
        RMTG,
    }
    expected_regions.update(_value(channel) for channel in CHANNELS)
    expected_regions.update(_value_fs(channel) for channel in CHANNELS)

    plastic = np.asarray(
        to_host(bridge.cp_synapse_plastic_mask), dtype=bool
    )
    expected_plastic = np.zeros(plastic.size, dtype=bool)
    expected_plastic[routes.all_indices()] = True

    channel_targets = {
        *(_actor(channel) for channel in CHANNELS),
        *(_value(channel) for channel in CHANNELS),
        *(f"motor_{channel}" for channel in CHANNELS),
    }
    generic_outcome_only = not any(
        pathway.from_region in {OUTCOME_ONSET, REWARD_US}
        and pathway.to_region in channel_targets
        for pathway in pathways
    )
    v2 = credit_config("v2")
    old_seeds = set(
        V1_V2_CALIBRATION_SEEDS
        + V1_V2_DEVELOPMENT_SEEDS
        + V1_V2_HELD_OUT_SEEDS
    )
    new_seeds = set(CALIBRATION_SEEDS + DEVELOPMENT_SEEDS + HELD_OUT_SEEDS)
    other_lane_seeds = set(OTHER_LANE_FORMAL_SEEDS)

    return {
        "all_v3_populations_share_one_bridge": expected_regions.issubset(
            region_names
        ),
        "only_actor_and_critic_routes_are_plastic": bool(
            np.array_equal(plastic, expected_plastic)
        ),
        "outcome_afferents_are_generic_not_channel_specific": (
            generic_outcome_only
        ),
        "v2_value_to_snc_gabab_gain_is_unchanged": bool(
            config.gabab_propagation_strength
            == v2.gabab_propagation_strength
            and config.value_to_snc_weight == v2.value_to_snc_weight
        ),
        "local_fs_normalization_exists_for_both_critics": all(
            any(_pathway_matches(
                pathway,
                f"motor_{channel}",
                _value_fs(channel),
                gate=CRITIC_NORMALIZATION_GATE,
            ) for pathway in pathways)
            and any(_pathway_matches(
                pathway,
                _value_fs(channel),
                _value(channel),
                receptor="gaba_a",
                gate=CRITIC_NORMALIZATION_GATE,
            ) for pathway in pathways)
            for channel in CHANNELS
        ),
        "action_value_reaches_omission_comparator": all(
            any(_pathway_matches(
                pathway,
                _value(channel),
                OMISSION_GATE,
                receptor="gaba_b",
                gate=EXPECTATION_TO_OMISSION_GATE,
            ) for pathway in pathways)
            for channel in CHANNELS
        ),
        "generic_outcome_reaches_lhb": any(_pathway_matches(
            pathway, OUTCOME_ONSET, LATERAL_HABENULA
        ) for pathway in pathways),
        "reward_recruits_neural_lhb_veto": bool(
            any(_pathway_matches(
                pathway,
                REWARD_US,
                REWARD_VETO,
                gate=REWARD_VETO_GATE,
            ) for pathway in pathways)
            and any(_pathway_matches(
                pathway,
                REWARD_VETO,
                LATERAL_HABENULA,
                receptor="gaba_a",
                gate=REWARD_VETO_GATE,
            ) for pathway in pathways)
        ),
        "lhb_rmtg_path_inhibits_snc": bool(
            any(_pathway_matches(
                pathway,
                LATERAL_HABENULA,
                RMTG,
                gate=OMISSION_PATH_GATE,
            ) for pathway in pathways)
            and any(_pathway_matches(
                pathway,
                RMTG,
                SNC,
                receptor="gaba_a",
                gate=OMISSION_PATH_GATE,
            ) for pathway in pathways)
        ),
        "v3_seed_partitions_are_disjoint_and_fresh": bool(
            not (old_seeds & new_seeds)
            and not (other_lane_seeds & new_seeds)
            and SMOKE_SEED not in new_seeds
            and len(new_seeds)
            == len(CALIBRATION_SEEDS)
            + len(DEVELOPMENT_SEEDS)
            + len(HELD_OUT_SEEDS)
        ),
    }


def _lesion_gate_values(mode: str) -> dict[str, float]:
    if mode not in {"contingent", *CONTROL_MODES}:
        raise ValueError(mode)
    values = {
        ACTION_COLLATERAL_GATE: 1.0,
        DOPAMINE_PATH_GATE: 1.0,
        VALUE_TO_SNC_GATE: 1.0,
        EXPECTATION_TO_OMISSION_GATE: 1.0,
        OMISSION_PATH_GATE: 1.0,
        CRITIC_NORMALIZATION_GATE: 1.0,
        REWARD_VETO_GATE: 1.0,
    }
    if mode == "collateral_lesion":
        values[ACTION_COLLATERAL_GATE] = 0.0
    elif mode == "da_lesion":
        values[DOPAMINE_PATH_GATE] = 0.0
    elif mode == "critic_lesion":
        values[VALUE_TO_SNC_GATE] = 0.0
        values[EXPECTATION_TO_OMISSION_GATE] = 0.0
    elif mode == "omission_path_lesion":
        values[OMISSION_PATH_GATE] = 0.0
    elif mode == "normalization_lesion":
        values[CRITIC_NORMALIZATION_GATE] = 0.0
    return values


def _apply_lesions(bridge, mode: str) -> dict[str, float]:
    values = _lesion_gate_values(mode)
    for gate, value in values.items():
        bridge.set_transmission_gate(gate, value)
    return values


def _lesion_telemetry_matches(condition: dict) -> bool:
    expected = _lesion_gate_values(condition["mode"])
    observed = condition.get("lesion_gate_values", {})
    return observed == expected


def _set_trial_current_v3(
    bridge,
    selector,
    config: VocalCreditConfigV3,
    *,
    arousal: bool = False,
    cue: bool = False,
    reward: bool = False,
    outcome: bool = False,
) -> None:
    _set_trial_current(
        bridge,
        selector,
        config,
        arousal=arousal,
        cue=cue,
        reward=reward,
        outcome=outcome,
    )
    xp, _ = get_backend()
    bridge.cp_external_input_current[
        xp.asarray(_indices(bridge, OMISSION_GATE))
    ] = xp.float32(config.omission_gate_tonic_pA)


def _reset_trial_v3(bridge, selector, config: VocalCreditConfigV3) -> None:
    xp, _ = get_backend()
    _set_equal_tonic_current(bridge, selector)
    bridge.cp_external_input_current[
        xp.asarray(_indices(bridge, "selector_reset"))
    ] = xp.float32(selector.reset_pA)
    _step(bridge, selector.reset_steps)
    _set_trial_current_v3(bridge, selector, config)
    _step(bridge, selector.washout_steps + config.post_reward_steps)


def _calibrate_snc_tonic_v3(
    bridge, selector, config: VocalCreditConfigV3
) -> float:
    snc = _indices(bridge, SNC)
    firing_fraction_sum = 0.0
    n_measured = 0
    _set_trial_current_v3(bridge, selector, config)
    for step in range(int(config.snc_tonic_calibration_steps)):
        _step(bridge)
        if step < int(config.snc_tonic_calibration_steps) // 2:
            continue
        firing = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        firing_fraction_sum += float(firing[snc].mean())
        n_measured += 1
    tonic_fraction = firing_fraction_sum / max(1, n_measured)
    dopamine = bridge.neuromodulator_manager._config_by_name("dopamine")
    dopamine.production_rules[0].threshold = float(tonic_fraction)
    bridge.neuromodulator_manager._rule_state["dopamine"][
        "signed_rate_ema"
    ] = float(tonic_fraction)
    bridge.neuromodulator_manager.set_concentration(
        "dopamine", dopamine.baseline
    )
    return float(tonic_fraction)


def _spike_count(firing: np.ndarray, bridge, region: str) -> int:
    return int(firing[_indices(bridge, region)].sum())


def _run_trial(
    bridge,
    routes: CreditRoutes,
    selector,
    config: VocalCreditConfigV3,
    *,
    scheduled_reward: bool = False,
    reward_action: int | None = None,
) -> dict:
    lead_actor_spikes = np.zeros(2, dtype=np.int64)
    lead_motor_spikes = np.zeros(2, dtype=np.int64)
    motor_counts = np.zeros(2, dtype=np.int64)
    decision_actor_spikes = np.zeros(2, dtype=np.int64)
    first_crossing = None
    decision_step = None
    decision_phase = None
    simultaneous = False
    cue_lead_steps_elapsed = 0

    _set_trial_current_v3(bridge, selector, config, cue=True)
    for step in range(int(config.cue_lead_steps)):
        _step(bridge)
        firing = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        previous = motor_counts.copy()
        for channel in CHANNELS:
            lead_actor_spikes[channel] += _spike_count(
                firing, bridge, _actor(channel)
            )
            spikes = _spike_count(firing, bridge, f"motor_{channel}")
            lead_motor_spikes[channel] += spikes
            motor_counts[channel] += spikes
        cue_lead_steps_elapsed = step + 1
        crossed = [
            channel
            for channel in CHANNELS
            if previous[channel] < selector.commit_threshold_spikes
            <= motor_counts[channel]
        ]
        if len(crossed) == 1:
            first_crossing = int(crossed[0])
            decision_step = int(step)
            decision_phase = "cue_lead"
            break
        if len(crossed) > 1:
            simultaneous = True
            decision_step = int(step)
            decision_phase = "cue_lead"
            break

    if first_crossing is None and not simultaneous:
        _set_trial_current_v3(
            bridge, selector, config, arousal=True, cue=True
        )
        for step in range(int(selector.action_steps)):
            _step(bridge)
            firing = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
            previous = motor_counts.copy()
            for channel in CHANNELS:
                motor_counts[channel] += _spike_count(
                    firing, bridge, f"motor_{channel}"
                )
                decision_actor_spikes[channel] += _spike_count(
                    firing, bridge, _actor(channel)
                )
            crossed = [
                channel
                for channel in CHANNELS
                if previous[channel] < selector.commit_threshold_spikes
                <= motor_counts[channel]
            ]
            if len(crossed) == 1:
                first_crossing = int(crossed[0])
                decision_step = int(step)
                decision_phase = "arousal"
                break
            if len(crossed) > 1:
                simultaneous = True
                decision_step = int(step)
                decision_phase = "arousal"
                break

    winner = None
    if first_crossing is not None and not simultaneous:
        loser = 1 - first_crossing
        loser_ratio = motor_counts[loser] / max(
            1, motor_counts[first_crossing]
        )
        if loser_ratio <= selector.clean_loser_ratio:
            winner = int(first_crossing)

    actor_spikes = np.zeros(2, dtype=np.int64)
    value_spikes = np.zeros(2, dtype=np.int64)
    value_fs_spikes = np.zeros(2, dtype=np.int64)
    _set_trial_current_v3(bridge, selector, config, cue=True)
    for _ in range(int(config.collateral_steps)):
        _step(bridge)
        firing = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        for channel in CHANNELS:
            actor_spikes[channel] += _spike_count(
                firing, bridge, _actor(channel)
            )
            value_spikes[channel] += _spike_count(
                firing, bridge, _value(channel)
            )
            value_fs_spikes[channel] += _spike_count(
                firing, bridge, _value_fs(channel)
            )

    _set_trial_current_v3(bridge, selector, config)
    _step(bridge, config.reward_delay_steps)
    eligibility = _eligibility_means(bridge, routes)
    ratio = None
    if winner is not None:
        ratio = float(
            eligibility[winner] / max(eligibility[1 - winner], 1e-12)
        )

    reward = bool(scheduled_reward)
    if reward_action is not None:
        reward = winner == int(reward_action)

    dopamine_before = float(
        bridge.neuromodulator_manager.get_concentration("dopamine")
    )
    dopamine_peak = dopamine_before
    dopamine_min = dopamine_before
    ach_before = float(
        bridge.neuromodulator_manager.get_concentration(
            "acetylcholine_outcome"
        )
    )
    ach_min = ach_before
    outcome_counts = {
        "outcome_onset_spikes": 0,
        "reward_us_spikes": 0,
        "reward_veto_spikes": 0,
        "omission_gate_spikes": 0,
        "lhb_spikes": 0,
        "rmtg_spikes": 0,
        "snc_spikes": 0,
    }
    _set_trial_current_v3(
        bridge, selector, config, reward=reward, outcome=True
    )
    for _ in range(int(config.reward_steps)):
        _step(bridge)
        firing = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        outcome_counts["outcome_onset_spikes"] += _spike_count(
            firing, bridge, OUTCOME_ONSET
        )
        outcome_counts["reward_us_spikes"] += _spike_count(
            firing, bridge, REWARD_US
        )
        outcome_counts["reward_veto_spikes"] += _spike_count(
            firing, bridge, REWARD_VETO
        )
        outcome_counts["omission_gate_spikes"] += _spike_count(
            firing, bridge, OMISSION_GATE
        )
        outcome_counts["lhb_spikes"] += _spike_count(
            firing, bridge, LATERAL_HABENULA
        )
        outcome_counts["rmtg_spikes"] += _spike_count(
            firing, bridge, RMTG
        )
        outcome_counts["snc_spikes"] += _spike_count(
            firing, bridge, SNC
        )
        dopamine = float(
            bridge.neuromodulator_manager.get_concentration("dopamine")
        )
        dopamine_peak = max(dopamine_peak, dopamine)
        dopamine_min = min(dopamine_min, dopamine)
        ach_min = min(
            ach_min,
            float(bridge.neuromodulator_manager.get_concentration(
                "acetylcholine_outcome"
            )),
        )

    _reset_trial_v3(bridge, selector, config)
    return {
        "winner": winner,
        "first_crossing": first_crossing,
        "decision_step": decision_step,
        "decision_phase": decision_phase,
        "simultaneous_crossing": bool(simultaneous),
        "motor_spikes": motor_counts.tolist(),
        "cue_lead_steps_elapsed": int(cue_lead_steps_elapsed),
        "cue_lead_motor_spikes": lead_motor_spikes.tolist(),
        "cue_lead_actor_spikes": lead_actor_spikes.tolist(),
        "decision_actor_spikes": decision_actor_spikes.tolist(),
        "actor_spikes": actor_spikes.tolist(),
        "value_spikes": value_spikes.tolist(),
        "value_fs_spikes": value_fs_spikes.tolist(),
        "eligibility_means_before_reward": {
            str(channel): eligibility[channel] for channel in CHANNELS
        },
        "executed_to_losing_eligibility_ratio": ratio,
        "reward_delivered": bool(reward),
        **outcome_counts,
        "dopamine_before_outcome": dopamine_before,
        "dopamine_peak": float(dopamine_peak),
        "dopamine_min": float(dopamine_min),
        "dopamine_burst_depth": float(dopamine_peak - dopamine_before),
        "dopamine_dip_depth": float(dopamine_before - dopamine_min),
        "ach_before_outcome": ach_before,
        "ach_min_during_outcome": float(ach_min),
    }


def _mean_training_spikes(condition: dict, key: str) -> float:
    rows = condition["training"]["rows"]
    return float(
        np.mean([sum(row[key]) for row in rows]) if rows else 0.0
    )


def _winner_local_value_rate(condition: dict) -> float:
    rows = [
        row
        for row in condition["training"]["rows"]
        if row["winner"] is not None and sum(row["value_spikes"]) > 0
    ]
    return float(
        sum(
            row["value_spikes"][row["winner"]]
            > row["value_spikes"][1 - row["winner"]]
            for row in rows
        )
        / max(1, len(rows))
    )


def run_condition(
    seed: int,
    *,
    mode: str,
    training_trials: int,
    baseline_trials: int,
    evaluation_trials: int,
    yoked_schedule=None,
    config: VocalCreditConfigV3 | None = None,
) -> dict:
    if mode not in {"contingent", *CONTROL_MODES}:
        raise ValueError(mode)
    if mode in YOKED_MODES and yoked_schedule is None:
        raise ValueError(f"{mode} requires a complete yoked reward schedule")

    config = config or v3_config()
    selector = selector_config("v2")
    bridge, routes = build_v3_bridge(seed, config)
    lesion_gates = _apply_lesions(bridge, mode)
    structural = _structural_preconditions(bridge, routes, config)

    _set_learning_gates(
        bridge, config, actor_enabled=False, value_enabled=False
    )
    tonic_fraction = _calibrate_snc_tonic_v3(bridge, selector, config)
    _set_trial_current_v3(bridge, selector, config)
    _step(bridge, selector.warmup_steps)
    _reset_trial_v3(bridge, selector, config)

    initial_weights = np.asarray(
        to_host(bridge.cp_connections.data), dtype=np.float32
    ).copy()
    initial_actor_weights = _route_weight_means(bridge, routes.actor)
    initial_value_weights = _route_weight_means(bridge, routes.value)

    baseline = [
        _run_trial(bridge, routes, selector, config)
        for _ in range(int(baseline_trials))
    ]
    _set_learning_gates(
        bridge,
        config,
        actor_enabled=True,
        value_enabled=mode != "critic_lesion",
    )

    training = []
    reward_schedule = []
    for trial in range(int(training_trials)):
        contingent = mode in CONTINGENT_MODES
        scheduled_reward = False
        if not contingent:
            if trial >= len(yoked_schedule):
                raise ValueError(f"{mode} received an incomplete yoked schedule")
            scheduled_reward = bool(yoked_schedule[trial])
        row = _run_trial(
            bridge,
            routes,
            selector,
            config,
            scheduled_reward=scheduled_reward,
            reward_action=0 if contingent else None,
        )
        training.append(row)
        reward_schedule.append(bool(row["reward_delivered"]))

    _set_learning_gates(
        bridge, config, actor_enabled=False, value_enabled=False
    )
    omission_probe = _run_trial(
        bridge, routes, selector, config, scheduled_reward=False
    )
    rewarded_probe = _run_trial(
        bridge, routes, selector, config, scheduled_reward=True
    )
    evaluation = [
        _run_trial(bridge, routes, selector, config)
        for _ in range(int(evaluation_trials))
    ]

    final_weights = np.asarray(
        to_host(bridge.cp_connections.data), dtype=np.float32
    )
    changed = np.abs(final_weights - initial_weights) > 1e-7
    declared = np.zeros(changed.size, dtype=bool)
    declared[routes.all_indices()] = True
    final_actor_weights = _route_weight_means(bridge, routes.actor)
    final_value_weights = _route_weight_means(bridge, routes.value)
    rewarded_ratios = [
        row["executed_to_losing_eligibility_ratio"]
        for row in training
        if row["reward_delivered"]
        and row["executed_to_losing_eligibility_ratio"] is not None
    ]

    result = {
        "seed": int(seed),
        "phase": "calibration",
        "mode": mode,
        "config": asdict(config),
        "structural_preconditions": structural,
        "lesion_gate_values": lesion_gates,
        "snc_tonic_firing_fraction": float(tonic_fraction),
        "baseline": {
            "trials": int(baseline_trials),
            "clean_commit_rate": float(sum(
                row["winner"] is not None for row in baseline
            ) / max(1, len(baseline))),
            "action_0_preference": _preference(baseline),
            "rows": baseline,
        },
        "training": {
            "trials": int(training_trials),
            "n_rewards": int(sum(reward_schedule)),
            "reward_schedule": reward_schedule,
            "rewarded_eligibility_ratio_ge_10_rate": float(sum(
                ratio >= 10.0 for ratio in rewarded_ratios
            ) / max(1, len(rewarded_ratios))),
            "rewarded_eligibility_ratios": rewarded_ratios,
            "winner_local_value_rate": _winner_local_value_rate({
                "training": {"rows": training}
            }),
            "mean_value_spikes_per_trial": float(np.mean([
                sum(row["value_spikes"]) for row in training
            ]) if training else 0.0),
            "mean_value_fs_spikes_per_trial": float(np.mean([
                sum(row["value_fs_spikes"]) for row in training
            ]) if training else 0.0),
            "rows": training,
        },
        "outcome_probes": {
            "omission": omission_probe,
            "rewarded": rewarded_probe,
        },
        "evaluation": {
            "trials": int(evaluation_trials),
            "clean_commit_rate": float(sum(
                row["winner"] is not None for row in evaluation
            ) / max(1, len(evaluation))),
            "action_0_preference": _preference(evaluation),
            "cue_led_rate": float(sum(
                row["winner"] is not None
                and row["decision_phase"] == "cue_lead"
                for row in evaluation
            ) / max(1, sum(
                row["winner"] is not None for row in evaluation
            ))),
            "rows": evaluation,
        },
        "weights": {
            "initial_route_means": {
                str(channel): initial_actor_weights[channel]
                for channel in CHANNELS
            },
            "final_route_means": {
                str(channel): final_actor_weights[channel]
                for channel in CHANNELS
            },
            "initial_value_route_means": {
                str(channel): initial_value_weights[channel]
                for channel in CHANNELS
            },
            "final_value_route_means": {
                str(channel): final_value_weights[channel]
                for channel in CHANNELS
            },
            "changed_declared_synapses": int(np.count_nonzero(
                changed & declared
            )),
            "changed_outside_declared_routes": int(np.count_nonzero(
                changed & ~declared
            )),
        },
    }
    del bridge
    return result


def _calibration_verdict(row: dict) -> dict:
    contingent = row["contingent"]
    controls = row["controls"]
    yoked = controls["yoked"]
    critic_lesion = controls["critic_lesion"]
    omission_lesion = controls["omission_path_lesion"]
    normalization_lesion = controls["normalization_lesion"]
    conditions = [contingent, *controls.values()]

    contingent_pref = contingent["evaluation"]["action_0_preference"]
    yoked_pref = yoked["evaluation"]["action_0_preference"]
    omission = yoked["outcome_probes"]["omission"]
    rewarded = yoked["outcome_probes"]["rewarded"]
    critic_omission = critic_lesion["outcome_probes"]["omission"]
    pathway_omission = omission_lesion["outcome_probes"]["omission"]

    changed_outside = sum(
        condition["weights"]["changed_outside_declared_routes"]
        for condition in conditions
    )
    rewarded_ratios = contingent["training"][
        "rewarded_eligibility_ratios"
    ]
    intact_value_spikes = contingent["training"][
        "mean_value_spikes_per_trial"
    ]
    lesion_value_spikes = normalization_lesion["training"][
        "mean_value_spikes_per_trial"
    ]

    checks = {
        "contingent_action_zero_acquired": bool(
            contingent_pref >= 0.90
            and contingent["evaluation"]["cue_led_rate"] >= 0.90
        ),
        "yoked_reward_does_not_create_either_dominant_action": bool(
            0.25 <= yoked_pref <= 0.75
        ),
        "executed_action_eligibility_is_local": bool(
            rewarded_ratios
            and sum(ratio >= 10.0 for ratio in rewarded_ratios)
            / len(rewarded_ratios)
            >= 0.90
        ),
        "critic_activity_is_executed_action_local": bool(
            contingent["training"]["winner_local_value_rate"] >= 0.90
        ),
        "collateral_path_is_load_bearing": bool(
            controls["collateral_lesion"]["evaluation"][
                "action_0_preference"
            ]
            < 0.80
            and _route_change(controls["collateral_lesion"]) <= 1e-6
        ),
        "reward_to_snc_path_is_load_bearing": bool(
            controls["da_lesion"]["evaluation"]["action_0_preference"]
            < 0.80
        ),
        "expected_omission_creates_lhb_rmtg_dopamine_dip": bool(
            omission["lhb_spikes"] > 0
            and omission["rmtg_spikes"] > 0
            and omission["dopamine_dip_depth"] >= MIN_DA_DIP
        ),
        "reward_veto_suppresses_negative_path_and_preserves_burst": bool(
            rewarded["reward_veto_spikes"] > 0
            and rewarded["lhb_spikes"] < omission["lhb_spikes"]
            and rewarded["rmtg_spikes"] < omission["rmtg_spikes"]
            and rewarded["dopamine_burst_depth"] >= MIN_DA_BURST
        ),
        "critic_lesion_removes_expected_omission_dip": bool(
            critic_omission["rmtg_spikes"] < omission["rmtg_spikes"]
            and critic_omission["dopamine_dip_depth"]
            + MIN_LESION_DA_DELTA
            <= omission["dopamine_dip_depth"]
        ),
        "omission_path_lesion_removes_rmtg_dopamine_dip": bool(
            pathway_omission["rmtg_spikes"] == 0
            and pathway_omission["dopamine_dip_depth"]
            + MIN_LESION_DA_DELTA
            <= omission["dopamine_dip_depth"]
        ),
        "local_fs_normalization_is_load_bearing": bool(
            contingent["training"]["mean_value_fs_spikes_per_trial"] > 0
            and normalization_lesion["training"][
                "mean_value_fs_spikes_per_trial"
            ]
            == 0
            and lesion_value_spikes
            >= MIN_NORMALIZATION_EFFECT * max(intact_value_spikes, 1e-12)
        ),
        "plasticity_stays_in_declared_routes": changed_outside == 0,
    }

    earned = Verdict("vocal action credit Gate B v3 calibration")
    earned.require(
        "all required control arms are present",
        set(controls) == set(CONTROL_MODES),
        expect=True,
    )
    earned.require(
        "all structural preconditions pass in every arm",
        all(
            all(condition["structural_preconditions"].values())
            for condition in conditions
        ),
        expect=True,
    )
    earned.require(
        "all baseline selectors produce clean neural commits",
        all(
            condition["baseline"]["clean_commit_rate"] >= 0.90
            for condition in conditions
        ),
        expect=True,
    )
    earned.require(
        "all yoked arms preserve the contingent reward count",
        all(
            controls[mode]["training"]["n_rewards"]
            == contingent["training"]["n_rewards"]
            for mode in YOKED_MODES
        ),
        expect=True,
    )
    earned.require(
        "tonic SNc calibration is finite and positive",
        all(
            np.isfinite(condition["snc_tonic_firing_fraction"])
            and condition["snc_tonic_firing_fraction"] > 0.0
            for condition in conditions
        ),
        expect=True,
    )
    earned.require(
        "generic outcome onset reaches the circuit in every frozen probe",
        all(
            condition["outcome_probes"][probe][
                "outcome_onset_spikes"
            ]
            > 0
            for condition in conditions
            for probe in ("omission", "rewarded")
        ),
        expect=True,
    )
    earned.require(
        "rewarded probes activate the neural reward-veto population",
        all(
            condition["outcome_probes"]["rewarded"][
                "reward_veto_spikes"
            ]
            > 0
            for condition in conditions
        ),
        expect=True,
    )
    earned.require(
        "plasticity scope telemetry is present for every arm",
        all(
            "changed_outside_declared_routes" in condition["weights"]
            for condition in conditions
        ),
        expect=True,
    )
    earned.require(
        "every lesion arm reports exactly its preregistered gate cut",
        all(_lesion_telemetry_matches(condition) for condition in conditions),
        expect=True,
    )
    decided = earned.decide(go=all(checks.values()), verbose=False)
    status = (
        "UNDEFINED"
        if decided["status"] == UNDEFINED
        else "CALIBRATION_PASS" if decided["go"] else "CALIBRATION_FAIL"
    )
    return {
        "status": status,
        "checks": checks,
        "preconditions": decided["preconditions"],
        "disabled_processes": decided["disabled_processes"],
        "undefined_reasons": decided["undefined_reasons"],
        "metrics": {
            "contingent_evaluation_action_0": contingent_pref,
            "contingent_evaluation_cue_led": contingent["evaluation"][
                "cue_led_rate"
            ],
            "yoked_evaluation_action_0": yoked_pref,
            "yoked_preference_distance_from_balanced": abs(
                yoked_pref - 0.5
            ),
            "yoked_omission_dopamine_dip": omission[
                "dopamine_dip_depth"
            ],
            "yoked_reward_dopamine_burst": rewarded[
                "dopamine_burst_depth"
            ],
            "critic_lesion_omission_dopamine_dip": critic_omission[
                "dopamine_dip_depth"
            ],
            "omission_path_lesion_dopamine_dip": pathway_omission[
                "dopamine_dip_depth"
            ],
            "intact_mean_value_spikes_per_trial": intact_value_spikes,
            "normalization_lesion_mean_value_spikes_per_trial": (
                lesion_value_spikes
            ),
            "changed_outside_declared_routes": changed_outside,
        },
    }


def run_seed(
    seed: int,
    *,
    training_trials: int = 40,
    baseline_trials: int = 20,
    evaluation_trials: int = 40,
    config: VocalCreditConfigV3 | None = None,
) -> dict:
    config = config or v3_config()
    contingent = run_condition(
        seed,
        mode="contingent",
        training_trials=training_trials,
        baseline_trials=baseline_trials,
        evaluation_trials=evaluation_trials,
        config=config,
    )
    schedule = _yoked_schedule(contingent["training"]["reward_schedule"])
    controls = {}
    for mode in CONTROL_MODES:
        controls[mode] = run_condition(
            seed,
            mode=mode,
            training_trials=training_trials,
            baseline_trials=baseline_trials,
            evaluation_trials=evaluation_trials,
            yoked_schedule=schedule if mode in YOKED_MODES else None,
            config=config,
        )
    row = {
        "seed": int(seed),
        "phase": "calibration",
        "contingent": contingent,
        "controls": controls,
    }
    row["verdict"] = _calibration_verdict(row)
    return row


def schema_smoke() -> dict:
    """Build v3 anatomy and validate output contracts without a science run."""

    config = v3_config()
    bridge, routes = build_v3_bridge(SMOKE_SEED, config)
    structural = _structural_preconditions(bridge, routes, config)
    result = {
        "probe": "vocal_action_credit_gate_b_v3",
        "version": "v3",
        "status": "SCHEMA_SMOKE",
        "science_seed_executed": False,
        "smoke_seed": SMOKE_SEED,
        "structural_preconditions": structural,
        "required_controls": list(CONTROL_MODES),
        "required_condition_fields": [
            "seed",
            "phase",
            "mode",
            "config",
            "structural_preconditions",
            "lesion_gate_values",
            "snc_tonic_firing_fraction",
            "baseline",
            "training",
            "outcome_probes",
            "evaluation",
            "weights",
        ],
        "seed_policy": {
            "open_phases": list(OPEN_PHASES),
            "calibration": list(CALIBRATION_SEEDS),
            "development_reserved": list(DEVELOPMENT_SEEDS),
            "held_out_reserved": list(HELD_OUT_SEEDS),
        },
        "host_boundary": {
            "generic_cue_timing": True,
            "generic_outcome_timing": True,
            "sensory_reward_presence": True,
            "desired_channel_current": False,
            "host_eligibility_assignment": False,
            "host_dopamine_assignment": False,
            "host_weight_update": False,
        },
    }
    del bridge
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run calibration-only vocal credit Gate B v3."
    )
    parser.add_argument(
        "--phase",
        choices=("calibration", "development", "held_out"),
        default="calibration",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(CALIBRATION_SEEDS)
    )
    parser.add_argument("--training-trials", type=int, default=40)
    parser.add_argument("--baseline-trials", type=int, default=20)
    parser.add_argument("--evaluation-trials", type=int, default=40)
    parser.add_argument("--schema-smoke", action="store_true")
    parser.add_argument(
        "--output",
        default=(
            "research/findings/raw/vocal_action_credit_gate_v3/"
            "calibration.json"
        ),
    )
    args = parser.parse_args(argv)

    if args.schema_smoke:
        smoke = schema_smoke()
        print(json.dumps(smoke, indent=2))
        return 0 if all(smoke["structural_preconditions"].values()) else 1

    seeds = validate_phase_seeds(args.phase, args.seeds)
    config = v3_config()
    rows = [
        run_seed(
            seed,
            training_trials=args.training_trials,
            baseline_trials=args.baseline_trials,
            evaluation_trials=args.evaluation_trials,
            config=config,
        )
        for seed in seeds
    ]
    result = {
        "probe": "vocal_action_credit_gate_b_v3",
        "version": "v3",
        "phase": args.phase,
        "seeds": list(seeds),
        "backend": get_backend()[1],
        "seed_policy": {
            "open_phases": list(OPEN_PHASES),
            "calibration": list(CALIBRATION_SEEDS),
            "development_reserved": list(DEVELOPMENT_SEEDS),
            "held_out_reserved": list(HELD_OUT_SEEDS),
        },
        "fixed_criteria": {
            "contingent_action_zero_min": 0.90,
            "contingent_cue_led_min": 0.90,
            "yoked_action_zero_interval": [0.25, 0.75],
            "local_eligibility_ratio_min": 10.0,
            "minimum_dopamine_burst": MIN_DA_BURST,
            "minimum_dopamine_dip": MIN_DA_DIP,
            "minimum_lesion_dopamine_delta": MIN_LESION_DA_DELTA,
            "minimum_normalization_effect": MIN_NORMALIZATION_EFFECT,
            "all_checks_required_per_seed": True,
        },
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "phase": result["phase"],
        "seeds": result["seeds"],
        "statuses": [row["verdict"]["status"] for row in rows],
        "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
