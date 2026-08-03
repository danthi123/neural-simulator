"""Gate B calibration for executed-action-local vocal reward credit.

This probe extends the validated Gate A v2 selector with two corticostriatal
actor routes. A shared cue projects to both actors through plastic synapses.
Only the executed motor population returns an excitatory collateral to its own
actor, creating local coactivity eligibility before a delayed sensory reward
drives a shared spiking SNc dopamine broadcast.

The host presents cue and reward events and observes motor threshold crossings.
It never assigns eligibility, dopamine, current, or a weight update by channel.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import numpy as np

from research.runners._vocal_action_selector_gate import (
    CHANNELS,
    DIRECT_PATH_GATE,
    _indices,
    _region,
    _set_equal_tonic_current,
    _step,
    build_selector_bridge,
    selector_config,
)
from sim.backend import get_backend, to_host
from sim.enums import NeuronType
from sim.neuromodulators import (
    ModulatorTarget,
    NeuromodulatorConfig,
    ProductionRule,
)
from sim.regions import RegionPathway
from tools.lab import attributable_to
from tools.verdict import UNDEFINED, Verdict


CREDIT_CUE = "vocal_credit_cue"
CREDIT_ACTOR_PREFIX = "vocal_credit_actor_"
CREDIT_VALUE_PREFIX = "vocal_credit_value_"
REWARD_US = "vocal_credit_reward_us"
OUTCOME_ONSET = "vocal_credit_outcome_onset"
SNC = "vocal_credit_snc"
CREDIT_PLASTICITY_GATE = "vocal_credit_learning"
VALUE_PLASTICITY_GATE = "vocal_value_learning"
ACTION_COLLATERAL_GATE = "vocal_action_collateral"
DOPAMINE_PATH_GATE = "vocal_credit_reward_to_snc"
VALUE_TO_SNC_GATE = "vocal_value_to_snc"

CALIBRATION_SEEDS = (7, 11)
DEVELOPMENT_SEEDS = (42, 43, 44, 100)
HELD_OUT_SEEDS = (101, 102)


@dataclass(frozen=True)
class CreditConfig:
    n_cue: int = 24
    n_actor: int = 24
    n_reward_us: int = 24
    n_outcome: int = 24
    n_snc: int = 20
    n_value: int = 24
    cue_pA: float = 1000.0
    reward_pA: float = 1000.0
    outcome_pA: float = 1000.0
    cue_to_actor_weight: float = 0.1
    motor_to_actor_weight: float = 20.0
    actor_to_gpi_weight: float = 60.0
    reward_to_snc_weight: float = 15.0
    motor_to_value_weight: float = 20.0
    value_to_snc_weight: float = 10.0
    cue_lead_steps: int = 100
    collateral_steps: int = 30
    reward_delay_steps: int = 100
    reward_steps: int = 20
    post_reward_steps: int = 500
    reward_learning_rate: float = 0.06
    actor_plasticity_gain: float = 1.0
    eligibility_tau_ms: float = 300.0
    coactivity_trace_tau_ms: float = 80.0
    coactivity_threshold: float = 0.001
    coactivity_scale: float = 20.0
    weight_max: float = 80.0
    dopamine_decay_tau_ms: float = 30.0
    dopamine_window_ms: float = 8.0
    dopamine_sensitivity: float = 40.0
    enable_action_value_critic: bool = False
    snc_tonic_pA: float = 0.0
    snc_tonic_calibration_steps: int = 300
    gabab_propagation_strength: float = 0.22
    gabab_tau_decay_ms: float = 150.0
    ach_decay_tau_ms: float = 20.0
    ach_window_ms: float = 5.0
    ach_pause_sensitivity: float = -1000.0


def credit_config(version):
    if version == "v1":
        return CreditConfig()
    if version == "v2":
        return replace(
            CreditConfig(),
            enable_action_value_critic=True,
            snc_tonic_pA=350.0,
            snc_tonic_calibration_steps=1000,
            reward_to_snc_weight=100.0,
            gabab_propagation_strength=0.00004,
            reward_steps=40,
            cue_pA=6000.0,
            actor_plasticity_gain=0.3,
            dopamine_decay_tau_ms=30.0,
            dopamine_window_ms=20.0,
            dopamine_sensitivity=40.0,
        )
    raise ValueError(f"unknown credit version: {version}")


def validate_calibration_seeds(seeds):
    checked = tuple(int(seed) for seed in seeds)
    invalid = [seed for seed in checked if seed not in CALIBRATION_SEEDS]
    if invalid:
        raise ValueError(
            f"Gate B accepts calibration seeds {CALIBRATION_SEEDS} only; "
            f"development={DEVELOPMENT_SEEDS}, held_out={HELD_OUT_SEEDS}, rejected={invalid}"
        )
    if not checked:
        raise ValueError("at least one calibration seed is required")
    return checked


@dataclass(frozen=True)
class CreditRoutes:
    actor: dict
    value: dict

    def all_indices(self):
        groups = list(self.actor.values()) + list(self.value.values())
        if not groups:
            return np.array([], dtype=np.int64)
        return np.concatenate(groups)


def _actor(channel):
    return f"{CREDIT_ACTOR_PREFIX}{channel}"


def _value(channel):
    return f"{CREDIT_VALUE_PREFIX}{channel}"


def _credit_regions(config):
    rs = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL
    d1 = NeuronType.IZH2007_STRIATAL_MSN_D1
    dopamine = NeuronType.IZH2007_DOPAMINE
    regions = [
        _region(CREDIT_CUE, config.n_cue, exc_fraction=1.0, neuron_type=rs),
        _region(
            REWARD_US, config.n_reward_us,
            exc_fraction=1.0, neuron_type=rs,
        ),
        _region(SNC, config.n_snc, exc_fraction=1.0, neuron_type=dopamine),
    ]
    regions.extend(
        _region(
            _actor(channel), config.n_actor,
            exc_fraction=0.0, neuron_type=d1,
        )
        for channel in CHANNELS
    )
    if config.enable_action_value_critic:
        regions.append(_region(
            OUTCOME_ONSET, config.n_outcome,
            exc_fraction=1.0, neuron_type=rs,
        ))
        regions.extend(
            _region(
                _value(channel), config.n_value,
                exc_fraction=0.0, neuron_type=d1,
            )
            for channel in CHANNELS
        )
    return regions


def _credit_pathways(config):
    pathways = [RegionPathway(
        from_region=REWARD_US,
        to_region=SNC,
        density=0.60,
        weight_mean=config.reward_to_snc_weight,
        weight_jitter=0.05,
        plastic=False,
        transmission_gate=DOPAMINE_PATH_GATE,
    )]
    for channel in CHANNELS:
        pathways.extend([
            RegionPathway(
                from_region=CREDIT_CUE,
                to_region=_actor(channel),
                density=1.0,
                weight_mean=config.cue_to_actor_weight,
                weight_jitter=0.0,
                plastic=True,
                plasticity_gate=CREDIT_PLASTICITY_GATE,
            ),
            RegionPathway(
                from_region=f"motor_{channel}",
                to_region=_actor(channel),
                density=1.0,
                weight_mean=config.motor_to_actor_weight,
                weight_jitter=0.05,
                plastic=False,
                transmission_gate=ACTION_COLLATERAL_GATE,
            ),
            RegionPathway(
                from_region=_actor(channel),
                to_region=f"gpi_{channel}",
                density=1.0,
                weight_mean=config.actor_to_gpi_weight,
                weight_jitter=0.05,
                plastic=False,
                transmission_gate=DIRECT_PATH_GATE,
            ),
            RegionPathway(
                from_region="selector_reset",
                to_region=_actor(channel),
                density=0.70,
                weight_mean=16.0,
                weight_jitter=0.0,
                plastic=False,
                receptor="gaba_a",
            ),
        ])
        if config.enable_action_value_critic:
            pathways.extend([
                RegionPathway(
                    from_region=f"motor_{channel}",
                    to_region=_value(channel),
                    density=1.0,
                    weight_mean=config.motor_to_value_weight,
                    weight_jitter=0.0,
                    plastic=True,
                    plasticity_gate=VALUE_PLASTICITY_GATE,
                    transmission_gate=ACTION_COLLATERAL_GATE,
                ),
                RegionPathway(
                    from_region=_value(channel),
                    to_region=SNC,
                    density=0.50,
                    weight_mean=config.value_to_snc_weight,
                    weight_jitter=0.05,
                    plastic=False,
                    receptor="gaba_b",
                    transmission_gate=VALUE_TO_SNC_GATE,
                ),
                RegionPathway(
                    from_region="selector_reset",
                    to_region=_value(channel),
                    density=0.70,
                    weight_mean=16.0,
                    weight_jitter=0.0,
                    plastic=False,
                    receptor="gaba_a",
                ),
            ])
    return pathways


def _core_updates(config):
    dopamine = NeuromodulatorConfig(
        name="dopamine",
        baseline=0.5,
        decay_tau_ms=config.dopamine_decay_tau_ms,
        concentration_min=0.0,
        concentration_max=2.0,
        targets=[],
        production_rules=[ProductionRule(
            rule_type="from_region_firing_signed",
            sensitivity=config.dopamine_sensitivity,
            threshold=(
                0.30 if config.enable_action_value_critic else 0.0
            ),
            window_ms=config.dopamine_window_ms,
            source_regions=[SNC],
        )],
    )
    neuromodulators = [dopamine]
    if config.enable_action_value_critic:
        neuromodulators.append(NeuromodulatorConfig(
            name="acetylcholine_outcome",
            baseline=1.0,
            decay_tau_ms=config.ach_decay_tau_ms,
            concentration_min=0.0,
            concentration_max=1.0,
            targets=[ModulatorTarget(
                target_type="plasticity_window_gate", scope="all",
            )],
            production_rules=[ProductionRule(
                rule_type="from_region_firing_signed",
                sensitivity=config.ach_pause_sensitivity,
                threshold=0.0,
                window_ms=config.ach_window_ms,
                source_regions=[OUTCOME_ONSET],
            )],
        ))
    updates = {
        "enable_reward_modulation": True,
        "enable_neuromodulator_subsystem": True,
        "neuromodulators": neuromodulators,
        "reward_defer_stdp_weight_update": True,
        "reward_learning_rate": config.reward_learning_rate,
        "reward_eligibility_tau_ms": config.eligibility_tau_ms,
        "reward_eligibility_from_coactivity": True,
        "reward_coactivity_trace_tau_ms": config.coactivity_trace_tau_ms,
        "reward_coactivity_threshold": config.coactivity_threshold,
        "reward_coactivity_scale": config.coactivity_scale,
        "stdp_w_min": 0.0,
        "stdp_w_max": config.weight_max,
        "hebbian_min_weight": 0.0,
        "hebbian_max_weight": config.weight_max,
    }
    if config.enable_action_value_critic:
        updates.update({
            "enable_gabab": True,
            "gabab_reversal_potential": -90.0,
            "gabab_tau_decay": config.gabab_tau_decay_ms,
            "gabab_propagation_strength": (
                config.gabab_propagation_strength
            ),
            "gabab_conductance_max": 0.0,
        })
    return updates


def _route_synapses(bridge, source, target):
    coo = bridge.cp_connections.tocoo(copy=False)
    rows = np.asarray(to_host(coo.row), dtype=np.int64)
    cols = np.asarray(to_host(coo.col), dtype=np.int64)
    pre = _indices(bridge, source)
    post = _indices(bridge, target)
    return np.flatnonzero(np.isin(rows, pre) & np.isin(cols, post))


def build_credit_bridge(seed, config=CreditConfig()):
    bridge = build_selector_bridge(
        seed,
        selector_config("v2"),
        extra_regions=_credit_regions(config),
        extra_pathways=_credit_pathways(config),
        core_config_updates=_core_updates(config),
    )
    xp, _ = get_backend()
    actor_routes = {
        channel: _route_synapses(bridge, CREDIT_CUE, _actor(channel))
        for channel in CHANNELS
    }
    value_routes = {}
    if config.enable_action_value_critic:
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
        routes.all_indices(),
        dtype=xp.int64,
    )
    return bridge, routes


def _set_learning_gates(
    bridge, config, *, actor_enabled, value_enabled,
):
    bridge.set_plasticity_gate(
        CREDIT_PLASTICITY_GATE,
        config.actor_plasticity_gain if actor_enabled else 0.0,
    )
    if config.enable_action_value_critic:
        bridge.set_plasticity_gate(
            VALUE_PLASTICITY_GATE, 1.0 if value_enabled else 0.0,
        )


def _set_trial_current(bridge, selector, config, *, arousal=False,
                       cue=False, reward=False, outcome=False):
    xp, _ = get_backend()
    _set_equal_tonic_current(bridge, selector)
    if config.snc_tonic_pA:
        bridge.cp_external_input_current[
            xp.asarray(_indices(bridge, SNC))
        ] = xp.float32(config.snc_tonic_pA)
    if arousal:
        bridge.cp_external_input_current[
            xp.asarray(_indices(bridge, "practice_arousal"))
        ] = xp.float32(selector.practice_pA)
    if cue:
        bridge.cp_external_input_current[
            xp.asarray(_indices(bridge, CREDIT_CUE))
        ] = xp.float32(config.cue_pA)
    if reward:
        bridge.cp_external_input_current[
            xp.asarray(_indices(bridge, REWARD_US))
        ] = xp.float32(config.reward_pA)
    if outcome and config.enable_action_value_critic:
        bridge.cp_external_input_current[
            xp.asarray(_indices(bridge, OUTCOME_ONSET))
        ] = xp.float32(config.outcome_pA)


def _eligibility_means(bridge, routes):
    trace = np.asarray(to_host(bridge.cp_eligibility_trace), dtype=np.float64)
    return {
        channel: float(np.mean(np.abs(trace[routes.actor[channel]])))
        for channel in CHANNELS
    }


def _route_weight_means(bridge, route_group):
    weights = np.asarray(to_host(bridge.cp_connections.data), dtype=np.float64)
    return {
        channel: float(np.mean(weights[route]))
        for channel, route in route_group.items()
    }


def _calibrate_snc_tonic(bridge, selector, config):
    if not config.enable_action_value_critic:
        return None
    snc = _indices(bridge, SNC)
    firing_fraction_sum = 0.0
    n_measured = 0
    _set_trial_current(bridge, selector, config)
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


def _reset_trial(bridge, selector, config):
    xp, _ = get_backend()
    _set_equal_tonic_current(bridge, selector)
    bridge.cp_external_input_current[
        xp.asarray(_indices(bridge, "selector_reset"))
    ] = xp.float32(selector.reset_pA)
    _step(bridge, selector.reset_steps)
    _set_trial_current(bridge, selector, config)
    _step(bridge, selector.washout_steps + config.post_reward_steps)


def _run_trial(
    bridge,
    routes,
    selector,
    config,
    *,
    scheduled_reward=False,
    reward_action=None,
):
    lead_actor_spikes = np.zeros(2, dtype=np.int64)
    lead_motor_spikes = np.zeros(2, dtype=np.int64)
    motor_counts = np.zeros(2, dtype=np.int64)
    decision_actor_spikes = np.zeros(2, dtype=np.int64)
    first_crossing = None
    decision_step = None
    decision_phase = None
    simultaneous = False
    cue_lead_steps_elapsed = 0
    _set_trial_current(bridge, selector, config, cue=True)
    for step in range(int(config.cue_lead_steps)):
        _step(bridge)
        firing = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        previous = motor_counts.copy()
        for channel in CHANNELS:
            lead_actor_spikes[channel] += int(
                firing[_indices(bridge, _actor(channel))].sum()
            )
            spikes = int(
                firing[_indices(bridge, f"motor_{channel}")].sum()
            )
            lead_motor_spikes[channel] += spikes
            motor_counts[channel] += spikes
        cue_lead_steps_elapsed = step + 1
        crossed = [
            channel for channel in CHANNELS
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
        _set_trial_current(
            bridge, selector, config, arousal=True, cue=True,
        )
        for step in range(int(selector.action_steps)):
            _step(bridge)
            firing = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
            previous = motor_counts.copy()
            for channel in CHANNELS:
                motor_counts[channel] += int(
                    firing[_indices(bridge, f"motor_{channel}")].sum()
                )
                decision_actor_spikes[channel] += int(
                    firing[_indices(bridge, _actor(channel))].sum()
                )
            crossed = [
                channel for channel in CHANNELS
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
        loser_ratio = motor_counts[loser] / max(1, motor_counts[first_crossing])
        if loser_ratio <= selector.clean_loser_ratio:
            winner = int(first_crossing)

    actor_spikes = np.zeros(2, dtype=np.int64)
    value_spikes = np.zeros(2, dtype=np.int64)
    _set_trial_current(bridge, selector, config, cue=True)
    for _ in range(int(config.collateral_steps)):
        _step(bridge)
        firing = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        for channel in CHANNELS:
            actor_spikes[channel] += int(
                firing[_indices(bridge, _actor(channel))].sum()
            )
            if config.enable_action_value_critic:
                value_spikes[channel] += int(
                    firing[_indices(bridge, _value(channel))].sum()
                )

    _set_trial_current(bridge, selector, config)
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

    dopamine_before = bridge.neuromodulator_manager.get_concentration(
        "dopamine"
    )
    ach_before = None
    ach_min = None
    if config.enable_action_value_critic:
        ach_before = bridge.neuromodulator_manager.get_concentration(
            "acetylcholine_outcome"
        )
        ach_min = ach_before
    gabab_before = None
    gabab_peak_before = None
    if config.enable_action_value_critic:
        snc_idx = _indices(bridge, SNC)
        gabab = np.asarray(
            to_host(bridge.cp_conductance_g_gabab), dtype=np.float64
        )[snc_idx]
        gabab_before = float(gabab.mean())
        gabab_peak_before = float(gabab.max())
    dopamine_peak = dopamine_before
    dopamine_min = dopamine_before
    snc_spikes = 0
    reward_us_spikes = 0
    _set_trial_current(
        bridge, selector, config, reward=reward, outcome=True,
    )
    for _ in range(int(config.reward_steps)):
        _step(bridge)
        firing = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        snc_spikes += int(firing[_indices(bridge, SNC)].sum())
        reward_us_spikes += int(firing[_indices(bridge, REWARD_US)].sum())
        dopamine_peak = max(
            dopamine_peak,
            bridge.neuromodulator_manager.get_concentration("dopamine"),
        )
        dopamine_min = min(
            dopamine_min,
            bridge.neuromodulator_manager.get_concentration("dopamine"),
        )
        if config.enable_action_value_critic:
            ach_min = min(
                ach_min,
                bridge.neuromodulator_manager.get_concentration(
                    "acetylcholine_outcome"
                ),
            )

    _reset_trial(bridge, selector, config)
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
        "eligibility_means_before_reward": {
            str(channel): eligibility[channel] for channel in CHANNELS
        },
        "executed_to_losing_eligibility_ratio": ratio,
        "reward_delivered": bool(reward),
        "snc_spikes": int(snc_spikes),
        "reward_us_spikes": int(reward_us_spikes),
        "snc_gabab_mean_before_reward": gabab_before,
        "snc_gabab_max_before_reward": gabab_peak_before,
        "dopamine_before_reward": float(dopamine_before),
        "dopamine_peak": float(dopamine_peak),
        "dopamine_min": float(dopamine_min),
        "ach_before_outcome": ach_before,
        "ach_min_during_outcome": ach_min,
    }


def _preference(rows, action=0):
    winners = [row["winner"] for row in rows if row["winner"] is not None]
    return float(winners.count(action) / max(1, len(winners)))


def _yoked_schedule(contingent_schedule):
    schedule = list(bool(value) for value in contingent_schedule)
    if len(schedule) < 2:
        return schedule
    offset = max(1, len(schedule) // 3)
    return schedule[offset:] + schedule[:offset]


def _cue_led_rate(condition):
    rows = [row for row in condition["evaluation"]["rows"] if row["winner"] is not None]
    return float(sum(row["decision_phase"] == "cue_lead" for row in rows) / max(1, len(rows)))


def _route_change(condition, route_key="final_route_means"):
    initial_key = "initial_" + route_key.removeprefix("final_")
    initial = condition["weights"][initial_key]
    final = condition["weights"][route_key]
    return max(abs(float(final[str(channel)]) - float(initial[str(channel)])) for channel in CHANNELS)


def _calibration_verdict(row, config):
    contingent = row["contingent"]
    controls = row["controls"]
    yoked = controls["yoked"]
    contingent_pref = contingent["evaluation"]["action_0_preference"]
    yoked_pref = yoked["evaluation"]["action_0_preference"]
    conditions = [contingent, *controls.values()]
    changed_outside = sum(
        condition["weights"]["changed_outside_declared_routes"]
        for condition in conditions
    )
    rewarded_ratios = contingent["training"]["rewarded_eligibility_ratios"]
    checks = {
        "contingent_action_zero_acquired": bool(
            contingent_pref >= 0.90 and _cue_led_rate(contingent) >= 0.90
        ),
        "yoked_reward_does_not_create_either_dominant_action": bool(
            0.25 <= yoked_pref <= 0.75
        ),
        "executed_action_eligibility_is_local": bool(
            rewarded_ratios
            and sum(ratio >= 10.0 for ratio in rewarded_ratios) / len(rewarded_ratios) >= 0.90
        ),
        "collateral_path_is_load_bearing": bool(
            controls["collateral_lesion"]["evaluation"]["action_0_preference"] < 0.80
            and _route_change(controls["collateral_lesion"]) <= 1e-6
        ),
        "reward_to_snc_path_is_load_bearing": bool(
            controls["da_lesion"]["evaluation"]["action_0_preference"] < 0.80
        ),
        "plasticity_stays_in_declared_routes": changed_outside == 0,
    }
    attributable_to(
        "contingent reward over yoked reward",
        contingent_pref,
        yoked_pref,
    )
    earned = Verdict("vocal action-conditioned reward credit calibration")
    earned.require(
        "all baseline selectors produce clean neural commits",
        all(condition["baseline"]["clean_commit_rate"] >= 0.90 for condition in conditions),
        expect=True,
    )
    earned.require(
        "yoked schedule preserves the contingent reward count",
        yoked["training"]["n_rewards"] == contingent["training"]["n_rewards"],
        expect=True,
    )
    earned.require(
        "plasticity scope telemetry is present for every arm",
        all("changed_outside_declared_routes" in condition["weights"] for condition in conditions),
        expect=True,
    )
    if config.enable_action_value_critic:
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
            "outcome onset produces the declared ACh pause",
            all(
                any(
                    trial["ach_min_during_outcome"] < trial["ach_before_outcome"]
                    for trial in condition["training"]["rows"]
                )
                for condition in conditions
            ),
            expect=True,
        )
    decided = earned.decide(go=all(checks.values()), verbose=False)
    return {
        "status": (
            "UNDEFINED"
            if decided["status"] == UNDEFINED
            else "CALIBRATION_PASS" if decided["go"] else "CALIBRATION_FAIL"
        ),
        "checks": checks,
        "preconditions": decided["preconditions"],
        "undefined_reasons": decided["undefined_reasons"],
        "metrics": {
            "contingent_evaluation_action_0": contingent_pref,
            "contingent_evaluation_cue_led": _cue_led_rate(contingent),
            "yoked_evaluation_action_0": yoked_pref,
            "yoked_preference_distance_from_balanced": abs(yoked_pref - 0.5),
            "changed_outside_declared_routes": changed_outside,
        },
    }


def run_condition(
    seed,
    *,
    mode,
    training_trials,
    baseline_trials,
    evaluation_trials,
    yoked_schedule=None,
    config=CreditConfig(),
):
    allowed_modes = {
        "contingent",
        "yoked",
        "collateral_lesion",
        "da_lesion",
        "yoked_value_lesion",
    }
    if mode not in allowed_modes:
        raise ValueError(mode)
    if mode == "yoked_value_lesion" and not config.enable_action_value_critic:
        raise ValueError("yoked_value_lesion requires credit version v2")
    selector = selector_config("v2")
    bridge, routes = build_credit_bridge(seed, config)
    bridge.set_transmission_gate(
        ACTION_COLLATERAL_GATE, 0.0 if mode == "collateral_lesion" else 1.0
    )
    bridge.set_transmission_gate(
        DOPAMINE_PATH_GATE, 0.0 if mode == "da_lesion" else 1.0
    )
    if config.enable_action_value_critic:
        bridge.set_transmission_gate(
            VALUE_TO_SNC_GATE,
            0.0 if mode == "yoked_value_lesion" else 1.0,
        )
    _set_learning_gates(
        bridge, config, actor_enabled=False, value_enabled=False,
    )
    tonic_fraction = _calibrate_snc_tonic(bridge, selector, config)
    _set_trial_current(bridge, selector, config)
    _step(bridge, selector.warmup_steps)
    _reset_trial(bridge, selector, config)
    initial_weights = np.asarray(
        to_host(bridge.cp_connections.data), dtype=np.float32
    ).copy()
    initial_route_weights = _route_weight_means(bridge, routes.actor)
    initial_value_weights = _route_weight_means(bridge, routes.value)

    _set_learning_gates(
        bridge, config, actor_enabled=False, value_enabled=False,
    )
    baseline = [
        _run_trial(bridge, routes, selector, config)
        for _ in range(int(baseline_trials))
    ]
    _set_learning_gates(
        bridge,
        config,
        actor_enabled=True,
        value_enabled=mode != "yoked_value_lesion",
    )

    training = []
    reward_schedule = []
    for trial in range(int(training_trials)):
        contingent = mode in {
            "contingent", "collateral_lesion", "da_lesion"
        }
        scheduled_reward = False
        if not contingent:
            if yoked_schedule is None or trial >= len(yoked_schedule):
                raise ValueError("yoked mode requires a complete reward schedule")
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
        bridge, config, actor_enabled=False, value_enabled=False,
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
    for route in list(routes.actor.values()) + list(routes.value.values()):
        declared[route] = True
    final_route_weights = _route_weight_means(bridge, routes.actor)
    final_value_weights = _route_weight_means(bridge, routes.value)
    rewarded_ratios = [
        row["executed_to_losing_eligibility_ratio"]
        for row in training
        if row["reward_delivered"]
        and row["executed_to_losing_eligibility_ratio"] is not None
    ]
    return {
        "seed": int(seed),
        "mode": mode,
        "config": asdict(config),
        "snc_tonic_firing_fraction": tonic_fraction,
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
            "rows": training,
        },
        "evaluation": {
            "trials": int(evaluation_trials),
            "clean_commit_rate": float(sum(
                row["winner"] is not None for row in evaluation
            ) / max(1, len(evaluation))),
            "action_0_preference": _preference(evaluation),
            "rows": evaluation,
        },
        "weights": {
            "initial_route_means": {
                str(channel): initial_route_weights[channel]
                for channel in CHANNELS
            },
            "final_route_means": {
                str(channel): final_route_weights[channel]
                for channel in CHANNELS
            },
            "initial_value_route_means": {
                str(channel): value
                for channel, value in initial_value_weights.items()
            },
            "final_value_route_means": {
                str(channel): value
                for channel, value in final_value_weights.items()
            },
            "changed_declared_synapses": int(np.count_nonzero(
                changed & declared
            )),
            "changed_outside_declared_routes": int(np.count_nonzero(
                changed & ~declared
            )),
        },
    }


def run_seed(seed, *, training_trials=40, baseline_trials=20,
             evaluation_trials=40, config=CreditConfig()):
    contingent = run_condition(
        seed,
        mode="contingent",
        training_trials=training_trials,
        baseline_trials=baseline_trials,
        evaluation_trials=evaluation_trials,
        config=config,
    )
    schedule = _yoked_schedule(contingent["training"]["reward_schedule"])
    control_modes = ["yoked", "collateral_lesion", "da_lesion"]
    if config.enable_action_value_critic:
        control_modes.append("yoked_value_lesion")
    controls = {
        mode: run_condition(
            seed,
            mode=mode,
            training_trials=training_trials,
            baseline_trials=baseline_trials,
            evaluation_trials=evaluation_trials,
            yoked_schedule=(
                schedule if mode in {"yoked", "yoked_value_lesion"} else None
            ),
            config=config,
        )
        for mode in control_modes
    }
    row = {"seed": int(seed), "contingent": contingent, "controls": controls}
    row["verdict"] = _calibration_verdict(row, config)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--credit-version", choices=("v1", "v2"), default="v1",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11])
    parser.add_argument("--training-trials", type=int, default=40)
    parser.add_argument("--baseline-trials", type=int, default=20)
    parser.add_argument("--evaluation-trials", type=int, default=40)
    parser.add_argument(
        "--output",
        default="research/findings/raw/vocal_action_credit_gate/calibration.json",
    )
    args = parser.parse_args(argv)
    validate_calibration_seeds(args.seeds)
    config = credit_config(args.credit_version)
    rows = [
        run_seed(
            seed,
            training_trials=args.training_trials,
            baseline_trials=args.baseline_trials,
            evaluation_trials=args.evaluation_trials,
            config=config,
        )
        for seed in args.seeds
    ]
    result = {
        "probe": "vocal_action_credit_gate_b_calibration",
        "credit_version": args.credit_version,
        "seeds": list(args.seeds),
        "backend": get_backend()[1],
        "seed_policy": {
            "calibration": list(CALIBRATION_SEEDS),
            "development_reserved": list(DEVELOPMENT_SEEDS),
            "held_out_reserved": list(HELD_OUT_SEEDS),
        },
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"seeds": result["seeds"], "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
