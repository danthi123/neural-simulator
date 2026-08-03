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
from dataclasses import asdict, dataclass
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
from sim.neuromodulators import NeuromodulatorConfig, ProductionRule
from sim.regions import RegionPathway


CREDIT_CUE = "vocal_credit_cue"
CREDIT_ACTOR_PREFIX = "vocal_credit_actor_"
REWARD_US = "vocal_credit_reward_us"
SNC = "vocal_credit_snc"
CREDIT_PLASTICITY_GATE = "vocal_credit_learning"
ACTION_COLLATERAL_GATE = "vocal_action_collateral"
DOPAMINE_PATH_GATE = "vocal_credit_reward_to_snc"


@dataclass(frozen=True)
class CreditConfig:
    n_cue: int = 24
    n_actor: int = 24
    n_reward_us: int = 24
    n_snc: int = 20
    cue_pA: float = 1000.0
    reward_pA: float = 1000.0
    cue_to_actor_weight: float = 0.1
    motor_to_actor_weight: float = 20.0
    actor_to_gpi_weight: float = 60.0
    reward_to_snc_weight: float = 15.0
    cue_lead_steps: int = 100
    collateral_steps: int = 30
    reward_delay_steps: int = 100
    reward_steps: int = 20
    post_reward_steps: int = 500
    reward_learning_rate: float = 0.06
    eligibility_tau_ms: float = 300.0
    coactivity_trace_tau_ms: float = 80.0
    coactivity_threshold: float = 0.001
    coactivity_scale: float = 20.0
    weight_max: float = 80.0
    dopamine_decay_tau_ms: float = 30.0
    dopamine_window_ms: float = 8.0
    dopamine_sensitivity: float = 40.0


def _actor(channel):
    return f"{CREDIT_ACTOR_PREFIX}{channel}"


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
            threshold=0.0,
            window_ms=config.dopamine_window_ms,
            source_regions=[SNC],
        )],
    )
    return {
        "enable_reward_modulation": True,
        "enable_neuromodulator_subsystem": True,
        "neuromodulators": [dopamine],
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
    routes = {
        channel: _route_synapses(bridge, CREDIT_CUE, _actor(channel))
        for channel in CHANNELS
    }
    bridge.set_global_plasticity_gain(0.0)
    bridge.set_plasticity_gate(CREDIT_PLASTICITY_GATE, 1.0)
    bridge.cp_reward_eligibility_synapse_indices = xp.asarray(
        np.concatenate([routes[channel] for channel in CHANNELS]),
        dtype=xp.int64,
    )
    return bridge, routes


def _set_trial_current(bridge, selector, config, *, arousal=False,
                       cue=False, reward=False):
    xp, _ = get_backend()
    _set_equal_tonic_current(bridge, selector)
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


def _eligibility_means(bridge, routes):
    trace = np.asarray(to_host(bridge.cp_eligibility_trace), dtype=np.float64)
    return {
        channel: float(np.mean(np.abs(trace[routes[channel]])))
        for channel in CHANNELS
    }


def _route_weight_means(bridge, routes):
    weights = np.asarray(to_host(bridge.cp_connections.data), dtype=np.float64)
    return {
        channel: float(np.mean(weights[routes[channel]]))
        for channel in CHANNELS
    }


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
    _set_trial_current(bridge, selector, config, cue=True)
    for _ in range(int(config.collateral_steps)):
        _step(bridge)
        firing = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        for channel in CHANNELS:
            actor_spikes[channel] += int(
                firing[_indices(bridge, _actor(channel))].sum()
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
    dopamine_peak = dopamine_before
    snc_spikes = 0
    _set_trial_current(bridge, selector, config, reward=reward)
    for _ in range(int(config.reward_steps)):
        _step(bridge)
        firing = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        snc_spikes += int(firing[_indices(bridge, SNC)].sum())
        dopamine_peak = max(
            dopamine_peak,
            bridge.neuromodulator_manager.get_concentration("dopamine"),
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
        "eligibility_means_before_reward": {
            str(channel): eligibility[channel] for channel in CHANNELS
        },
        "executed_to_losing_eligibility_ratio": ratio,
        "reward_delivered": bool(reward),
        "snc_spikes": int(snc_spikes),
        "dopamine_before_reward": float(dopamine_before),
        "dopamine_peak": float(dopamine_peak),
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
    if mode not in {"contingent", "yoked", "collateral_lesion", "da_lesion"}:
        raise ValueError(mode)
    selector = selector_config("v2")
    bridge, routes = build_credit_bridge(seed, config)
    bridge.set_transmission_gate(
        ACTION_COLLATERAL_GATE, 0.0 if mode == "collateral_lesion" else 1.0
    )
    bridge.set_transmission_gate(
        DOPAMINE_PATH_GATE, 0.0 if mode == "da_lesion" else 1.0
    )
    bridge.set_plasticity_gate(CREDIT_PLASTICITY_GATE, 0.0)
    _set_trial_current(bridge, selector, config)
    _step(bridge, selector.warmup_steps)
    _reset_trial(bridge, selector, config)
    initial_weights = np.asarray(
        to_host(bridge.cp_connections.data), dtype=np.float32
    ).copy()
    initial_route_weights = _route_weight_means(bridge, routes)

    bridge.set_plasticity_gate(CREDIT_PLASTICITY_GATE, 0.0)
    baseline = [
        _run_trial(bridge, routes, selector, config)
        for _ in range(int(baseline_trials))
    ]
    bridge.set_plasticity_gate(CREDIT_PLASTICITY_GATE, 1.0)

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

    bridge.set_plasticity_gate(CREDIT_PLASTICITY_GATE, 0.0)
    evaluation = [
        _run_trial(bridge, routes, selector, config)
        for _ in range(int(evaluation_trials))
    ]
    final_weights = np.asarray(
        to_host(bridge.cp_connections.data), dtype=np.float32
    )
    changed = np.abs(final_weights - initial_weights) > 1e-7
    declared = np.zeros(changed.size, dtype=bool)
    for route in routes.values():
        declared[route] = True
    final_route_weights = _route_weight_means(bridge, routes)
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
    controls = {
        mode: run_condition(
            seed,
            mode=mode,
            training_trials=training_trials,
            baseline_trials=baseline_trials,
            evaluation_trials=evaluation_trials,
            yoked_schedule=schedule if mode == "yoked" else None,
            config=config,
        )
        for mode in ("yoked", "collateral_lesion", "da_lesion")
    }
    return {"seed": int(seed), "contingent": contingent, "controls": controls}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11])
    parser.add_argument("--training-trials", type=int, default=40)
    parser.add_argument("--baseline-trials", type=int, default=20)
    parser.add_argument("--evaluation-trials", type=int, default=40)
    parser.add_argument(
        "--output",
        default="research/findings/raw/vocal_action_credit_gate/calibration.json",
    )
    args = parser.parse_args(argv)
    rows = [
        run_seed(
            seed,
            training_trials=args.training_trials,
            baseline_trials=args.baseline_trials,
            evaluation_trials=args.evaluation_trials,
        )
        for seed in args.seeds
    ]
    result = {
        "probe": "vocal_action_credit_gate_b_calibration",
        "seeds": list(args.seeds),
        "backend": get_backend()[1],
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"seeds": result["seeds"], "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
