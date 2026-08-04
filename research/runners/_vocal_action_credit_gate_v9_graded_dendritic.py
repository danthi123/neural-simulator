"""Sealed Gate B v9 graded-dendritic expectation engagement smoke.

Reserved seed zero may test whether the existing learned action-context route
can recruit MSN-D1 expectation spikes through the bridge's graded dendritic
plateau. No formal seed is assigned or executable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from research.runners import _vocal_action_credit_gate_v3 as v3
from research.runners import _vocal_action_credit_gate_v5 as v5
from research.runners import _vocal_action_credit_gate_v5_learning as v5l
from research.runners import _vocal_action_credit_gate_v7_dense_convergence as v7
from research.runners._vocal_action_credit_gate import (
    CREDIT_CUE,
    CREDIT_PLASTICITY_GATE,
    CreditRoutes,
    _actor,
    _route_synapses,
)
from research.runners._vocal_action_selector_gate import (
    CHANNELS,
    _indices,
    build_selector_bridge,
    selector_config,
)
from sim.backend import get_backend, to_host
from tools.lab import attributable_to


OPEN_PHASES: tuple[str, ...] = ()
SMOKE_SEED = 0
TRACE_SIZE = 200
PLATEAU_CENTER_LADDER = (16.0, 8.0, 4.0, 2.0)
PLATEAU_SLOPE = 0.33
PLATEAU_STRENGTH = 2.0
PLATEAU_TAU_RISE_MS = 2.0
PLATEAU_TAU_DECAY_MS = 500.0

HOST_BOUNDARY = {
    **v7.HOST_BOUNDARY,
    "host_reads_plateau_to_calculate_dopamine": False,
    "host_injects_expectation_current": False,
    "host_selects_expectation_channel": False,
    "expectation_output_open_during_engagement": False,
}


@dataclass(frozen=True)
class VocalCreditConfigV9(v7.VocalCreditConfigV7):
    n_value: int = TRACE_SIZE
    action_tag_center: float = PLATEAU_CENTER_LADDER[0]
    action_tag_slope: float = PLATEAU_SLOPE
    action_tag_strength: float = PLATEAU_STRENGTH
    action_tag_tau_rise_ms: float = PLATEAU_TAU_RISE_MS
    action_tag_tau_decay_ms: float = PLATEAU_TAU_DECAY_MS


def v9_config(center: float = PLATEAU_CENTER_LADDER[0]) -> VocalCreditConfigV9:
    checked = float(center)
    if checked not in PLATEAU_CENTER_LADDER:
        raise ValueError(
            f"plateau center must be in preregistered ladder "
            f"{PLATEAU_CENTER_LADDER}; received={checked}"
        )
    return VocalCreditConfigV9(action_tag_center=checked)


def validate_smoke_seed(seed: int) -> int:
    checked = int(seed)
    if checked != SMOKE_SEED:
        raise ValueError(f"v9 smoke accepts reserved seed {SMOKE_SEED} only")
    return checked


def validate_phase(phase: str) -> str:
    raise ValueError(
        "Gate B v9 formal phases are sealed and unassigned; "
        f"open phases={OPEN_PHASES}, requested={phase!r}"
    )


def run_formal_seed(seed: int):
    raise ValueError(f"Gate B v9 formal execution is sealed; received={int(seed)}")


def _is_expectation_route(pathway) -> bool:
    return bool(
        pathway.from_region in {v5l._trace(channel) for channel in CHANNELS}
        and pathway.to_region in {
            v5l._expectation(channel) for channel in CHANNELS
        }
    )


def _v9_pathways(
    config: VocalCreditConfigV9,
    *,
    action_permutation: tuple[int, int] = (0, 1),
):
    pathways = v7._v7_pathways(
        config, action_permutation=action_permutation
    )
    return [
        replace(pathway, coincidence_detector=True)
        if _is_expectation_route(pathway)
        else pathway
        for pathway in pathways
    ]


def _output_route_indices(bridge) -> np.ndarray:
    routes = []
    for channel in CHANNELS:
        routes.extend([
            _route_synapses(
                bridge, v5l._expectation(channel), v5l.SNC
            ),
            _route_synapses(
                bridge, v5l._expectation(channel), v3.OMISSION_GATE
            ),
        ])
    return np.sort(np.concatenate(routes))


def build_v9_bridge(
    seed: int = SMOKE_SEED,
    config: VocalCreditConfigV9 | None = None,
    *,
    action_permutation: tuple[int, int] = (0, 1),
):
    validate_smoke_seed(seed)
    config = config or v9_config()
    bridge = build_selector_bridge(
        seed,
        selector_config("v2"),
        extra_regions=v5l._learning_regions(config),
        extra_pathways=_v9_pathways(
            config, action_permutation=action_permutation
        ),
        core_config_updates=v5._v5_core_updates(config),
    )
    actor_routes = {
        channel: _route_synapses(bridge, CREDIT_CUE, _actor(channel))
        for channel in CHANNELS
    }
    expectation_routes = {
        channel: _route_synapses(
            bridge, v5l._trace(channel), v5l._expectation(channel)
        )
        for channel in CHANNELS
    }
    routes = CreditRoutes(actor=actor_routes, value=expectation_routes)
    xp, _ = get_backend()
    bridge.set_global_plasticity_gain(0.0)
    bridge.set_plasticity_gate(
        CREDIT_PLASTICITY_GATE, config.actor_plasticity_gain
    )
    bridge.set_plasticity_gate(v5l.EXPECTATION_PLASTICITY_GATE, 1.0)
    bridge.set_transmission_gate(v5l.EXPECTATION_OUTPUT_GATE, 0.0)
    bridge.cp_reward_eligibility_synapse_indices = xp.asarray(
        routes.all_indices(), dtype=xp.int64
    )
    return bridge, {
        "routes": routes,
        "actor_routes": actor_routes,
        "expectation_routes": expectation_routes,
        "expectation_output_routes": _output_route_indices(bridge),
        "action_permutation": tuple(action_permutation),
    }


def _expected_coincidence_indices(bridge, handles: dict) -> np.ndarray:
    expected = []
    permutation = handles["action_permutation"]
    for source_channel in CHANNELS:
        target_channel = int(permutation[source_channel])
        expected.append(_route_synapses(
            bridge, f"commit_{source_channel}", v5l._trace(target_channel)
        ))
    for channel in CHANNELS:
        expected.extend([
            _route_synapses(
                bridge, "practice_arousal", v5l._trace(channel)
            ),
            handles["expectation_routes"][channel],
        ])
    return np.sort(np.concatenate(expected))


def structural_audit(bridge, handles: dict) -> dict[str, object]:
    base = v5l._structural_audit(bridge, handles)
    coincidence = np.asarray(
        to_host(bridge.cp_coincidence_synapse_mask), dtype=bool
    )
    expected = np.zeros(coincidence.shape, dtype=bool)
    expected[_expected_coincidence_indices(bridge, handles)] = True
    output_gate = np.sort(np.asarray(to_host(
        bridge._transmission_gate_indices_gpu[v5l.EXPECTATION_OUTPUT_GATE]
    ), dtype=np.int64))
    base["coincidence"] = {
        "enabled_synapses": int(coincidence.sum()),
        "intended_synapses": int(expected.sum()),
        "enabled_outside_intended_routes": int(
            np.logical_and(coincidence, ~expected).sum()
        ),
        "disabled_inside_intended_routes": int(
            np.logical_and(~coincidence, expected).sum()
        ),
    }
    base["expectation_output_gate_matches_routes"] = bool(
        np.array_equal(output_gate, handles["expectation_output_routes"])
    )
    base["expectation_output_gate_value"] = float(
        bridge._transmission_gate_values[v5l.EXPECTATION_OUTPUT_GATE]
    )
    return base


def lesion_expectation_plateau(bridge, handles: dict) -> dict[str, object]:
    indices = np.sort(np.concatenate(
        list(handles["expectation_routes"].values())
    ))
    before = np.asarray(
        to_host(bridge.cp_coincidence_synapse_mask), dtype=bool
    ).copy()
    weights_before = np.asarray(
        to_host(bridge.cp_connections.data), dtype=np.float32
    ).copy()
    xp, _ = get_backend()
    bridge.cp_coincidence_synapse_mask[
        xp.asarray(indices, dtype=xp.int64)
    ] = False
    after = np.asarray(
        to_host(bridge.cp_coincidence_synapse_mask), dtype=bool
    )
    weights_after = np.asarray(
        to_host(bridge.cp_connections.data), dtype=np.float32
    )
    changed = np.flatnonzero(before != after)
    return {
        "expected_route_synapses": int(indices.size),
        "changed_mask_synapses": int(changed.size),
        "changed_only_expected_routes": bool(np.array_equal(changed, indices)),
        "expected_routes_disabled": bool(not after[indices].any()),
        "other_dendritic_routes_unchanged": bool(
            np.array_equal(
                np.delete(before, indices), np.delete(after, indices)
            )
        ),
        "weights_unchanged": bool(np.array_equal(weights_before, weights_after)),
    }


def _region_spikes(firing: np.ndarray, bridge, region: str) -> int:
    return v3._spike_count(firing, bridge, region)


def _plateau_means(state: np.ndarray, bridge, prefix) -> list[float]:
    return [
        float(state[_indices(bridge, prefix(channel))].mean())
        for channel in CHANNELS
    ]


def _observe_window(bridge, steps: int) -> dict[str, object]:
    totals = {
        "commit": [0, 0],
        "motor": [0, 0],
        "actor": [0, 0],
        "trace": [0, 0],
        "expectation": [0, 0],
        "graded_plateau_trace_integral": [0.0, 0.0],
        "graded_plateau_expectation_integral": [0.0, 0.0],
        "graded_plateau_trace_peak": [0.0, 0.0],
        "graded_plateau_expectation_peak": [0.0, 0.0],
        "snc": 0,
        "lhb": 0,
        "rmtg": 0,
    }
    dopamine = []
    for _ in range(int(steps)):
        v5._step(bridge)
        firing = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        plateau = np.asarray(
            to_host(bridge.cp_conductance_g_graded_plateau), dtype=np.float64
        )
        trace_plateau = _plateau_means(plateau, bridge, v5l._trace)
        expectation_plateau = _plateau_means(
            plateau, bridge, v5l._expectation
        )
        for channel in CHANNELS:
            totals["commit"][channel] += _region_spikes(
                firing, bridge, f"commit_{channel}"
            )
            totals["motor"][channel] += _region_spikes(
                firing, bridge, f"motor_{channel}"
            )
            totals["actor"][channel] += _region_spikes(
                firing, bridge, _actor(channel)
            )
            totals["trace"][channel] += _region_spikes(
                firing, bridge, v5l._trace(channel)
            )
            totals["expectation"][channel] += _region_spikes(
                firing, bridge, v5l._expectation(channel)
            )
            totals["graded_plateau_trace_integral"][channel] += (
                trace_plateau[channel]
            )
            totals["graded_plateau_expectation_integral"][channel] += (
                expectation_plateau[channel]
            )
            totals["graded_plateau_trace_peak"][channel] = max(
                totals["graded_plateau_trace_peak"][channel],
                trace_plateau[channel],
            )
            totals["graded_plateau_expectation_peak"][channel] = max(
                totals["graded_plateau_expectation_peak"][channel],
                expectation_plateau[channel],
            )
        totals["snc"] += _region_spikes(firing, bridge, v5l.SNC)
        totals["lhb"] += _region_spikes(
            firing, bridge, v3.LATERAL_HABENULA
        )
        totals["rmtg"] += _region_spikes(firing, bridge, v3.RMTG)
        dopamine.append(float(
            bridge.neuromodulator_manager.get_concentration("dopamine")
        ))
    totals["dopamine_min"] = float(min(dopamine, default=0.0))
    totals["dopamine_max"] = float(max(dopamine, default=0.0))
    totals["dopamine_last"] = float(dopamine[-1] if dopamine else 0.0)
    return totals


def _run_trial(
    bridge,
    handles: dict,
    config: VocalCreditConfigV9,
    *,
    reward_action: int | None,
    scheduled_reward: bool | None = None,
) -> dict[str, object]:
    selector = selector_config("v2")
    v3._set_trial_current_v3(bridge, selector, config, cue=True)
    cue = _observe_window(bridge, config.cue_lead_steps)
    v3._set_trial_current_v3(
        bridge, selector, config, cue=True, arousal=True
    )
    action = _observe_window(bridge, config.smoke_action_steps)
    cue_winner = v5._winner_from_neural_counts(cue["commit"])
    arousal_winner = v5._winner_from_neural_counts(action["commit"])
    winner = arousal_winner

    v3._set_trial_current_v3(bridge, selector, config)
    delay = _observe_window(bridge, config.reward_delay_steps)
    snc_indices = _indices(bridge, v5l.SNC)
    gabab_state = np.asarray(
        to_host(bridge.cp_conductance_g_gabab), dtype=np.float64
    )[snc_indices]
    eligibility = np.asarray(
        to_host(bridge.cp_eligibility_trace), dtype=np.float64
    )
    eligibility_means = {
        "actor": [
            float(np.mean(np.abs(eligibility[handles["actor_routes"][c]])))
            for c in CHANNELS
        ],
        "expectation": [
            float(np.mean(np.abs(
                eligibility[handles["expectation_routes"][c]]
            )))
            for c in CHANNELS
        ],
    }
    reward = (
        bool(scheduled_reward)
        if scheduled_reward is not None
        else bool(reward_action is not None and winner == int(reward_action))
    )
    dopamine_before = float(
        bridge.neuromodulator_manager.get_concentration("dopamine")
    )
    v3._set_trial_current_v3(
        bridge, selector, config, outcome=True, reward=reward
    )
    outcome = _observe_window(bridge, config.reward_steps)
    v3._reset_trial_v3(bridge, selector, config)
    v7._clear_girk_and_target_state(bridge)
    return {
        "winner": winner,
        "cue_winner": cue_winner,
        "arousal_winner": arousal_winner,
        "cue_matches_executed_action": bool(
            cue_winner is not None and cue_winner == arousal_winner
        ),
        "reward_delivered": reward,
        "cue": cue,
        "action": action,
        "delay": delay,
        "outcome": outcome,
        "eligibility_before_outcome": eligibility_means,
        "gabab_snc_before_outcome_mean": float(gabab_state.mean()),
        "gabab_snc_before_outcome_peak": float(gabab_state.max()),
        "dopamine_before_outcome": dopamine_before,
        "dopamine_burst_depth": float(
            outcome["dopamine_max"] - dopamine_before
        ),
        "dopamine_dip_depth": float(
            dopamine_before - outcome["dopamine_min"]
        ),
    }


def run_engagement_condition(
    mode: str,
    config: VocalCreditConfigV9,
    *,
    seed: int = SMOKE_SEED,
) -> dict[str, object]:
    if mode not in {
        "intact", "expectation_learning_lesion", "expectation_plateau_lesion"
    }:
        raise ValueError(mode)
    validate_smoke_seed(seed)
    bridge, handles = build_v9_bridge(seed=seed, config=config)
    selector = selector_config("v2")
    bridge.set_plasticity_gate(CREDIT_PLASTICITY_GATE, 0.0)
    bridge.set_plasticity_gate(v5l.EXPECTATION_PLASTICITY_GATE, 0.0)
    plateau_lesion = None
    if mode == "expectation_plateau_lesion":
        plateau_lesion = lesion_expectation_plateau(bridge, handles)
    v3._calibrate_snc_tonic_v3(bridge, selector, config)
    v3._set_trial_current_v3(bridge, selector, config)
    v5._step(bridge, selector.warmup_steps)
    v7._reset_measured_trial(bridge, config)

    baseline_probe = _run_trial(
        bridge, handles, config, reward_action=None, scheduled_reward=False
    )
    weights_before = np.asarray(
        to_host(bridge.cp_connections.data), dtype=np.float32
    ).copy()
    expectation_before = v5l._weight_means(
        bridge, handles["expectation_routes"]
    )
    bridge.set_plasticity_gate(
        CREDIT_PLASTICITY_GATE, config.actor_plasticity_gain
    )
    bridge.set_plasticity_gate(v5l.EXPECTATION_PLASTICITY_GATE, 1.0)
    if mode == "expectation_learning_lesion":
        bridge.set_plasticity_gate(v5l.EXPECTATION_PLASTICITY_GATE, 0.0)

    rows = [
        _run_trial(bridge, handles, config, reward_action=0)
        for _ in range(config.smoke_training_trials)
    ]
    weights_after = np.asarray(
        to_host(bridge.cp_connections.data), dtype=np.float32
    )
    changed = np.abs(weights_after - weights_before) > 1e-7
    declared = np.zeros(changed.shape, dtype=bool)
    declared[handles["routes"].all_indices()] = True
    return {
        "mode": mode,
        "plateau_center": config.action_tag_center,
        "baseline_probe": baseline_probe,
        "expectation_weight_before": expectation_before,
        "expectation_weight_after": v5l._weight_means(
            bridge, handles["expectation_routes"]
        ),
        "expectation_output_gate": float(
            bridge._transmission_gate_values[v5l.EXPECTATION_OUTPUT_GATE]
        ),
        "plateau_lesion_audit": plateau_lesion,
        "clean_trials": int(sum(row["winner"] is not None for row in rows)),
        "rewarded_trials": int(sum(row["reward_delivered"] for row in rows)),
        "changed_synapses": int(changed.sum()),
        "changed_outside_declared_routes": int(
            np.logical_and(changed, ~declared).sum()
        ),
        "rows": rows,
    }


def _late_rewarded(condition: dict[str, object]) -> list[dict[str, object]]:
    return [
        row for row in condition["rows"] if row["reward_delivered"]
    ][-4:]


def _late_delay_expectation(condition: dict[str, object]) -> list[int]:
    late = _late_rewarded(condition)
    return [
        int(sum(row["delay"]["expectation"][channel] for row in late))
        for channel in CHANNELS
    ]


def _window_sum(
    condition: dict[str, object], key: str, windows=("delay",)
) -> list[float]:
    rows = condition["rows"]
    return [
        float(sum(
            row[window][key][channel]
            for row in rows
            for window in windows
        ))
        for channel in CHANNELS
    ]


def run_engagement_smoke(
    center: float,
    *,
    seed: int = SMOKE_SEED,
) -> dict[str, object]:
    config = v9_config(center)
    conditions = {
        mode: run_engagement_condition(mode, config, seed=seed)
        for mode in (
            "intact",
            "expectation_learning_lesion",
            "expectation_plateau_lesion",
        )
    }
    return _finish_engagement_smoke(center, seed, config, conditions)


def run_output_condition(
    mode: str,
    probe: str,
    *,
    seed: int = SMOKE_SEED,
) -> dict[str, object]:
    if mode not in {
        "output_intact", "expectation_output_lesion",
        "expectation_learning_lesion",
    }:
        raise ValueError(mode)
    if probe not in {"reward", "omission"}:
        raise ValueError(probe)
    validate_smoke_seed(seed)
    config = v9_config(2.0)
    bridge, handles = build_v9_bridge(seed=seed, config=config)
    selector = selector_config("v2")
    bridge.set_plasticity_gate(CREDIT_PLASTICITY_GATE, 0.0)
    bridge.set_plasticity_gate(v5l.EXPECTATION_PLASTICITY_GATE, 0.0)
    bridge.set_transmission_gate(v5l.EXPECTATION_OUTPUT_GATE, 0.0)
    v3._calibrate_snc_tonic_v3(bridge, selector, config)
    v3._set_trial_current_v3(bridge, selector, config)
    v5._step(bridge, selector.warmup_steps)
    v7._reset_measured_trial(bridge, config)

    baseline_probe = _run_trial(
        bridge, handles, config, reward_action=None, scheduled_reward=False
    )
    weights_before = np.asarray(
        to_host(bridge.cp_connections.data), dtype=np.float32
    ).copy()
    expectation_before = v5l._weight_means(
        bridge, handles["expectation_routes"]
    )
    bridge.set_plasticity_gate(
        CREDIT_PLASTICITY_GATE, config.actor_plasticity_gain
    )
    bridge.set_plasticity_gate(v5l.EXPECTATION_PLASTICITY_GATE, 1.0)
    if mode == "expectation_learning_lesion":
        bridge.set_plasticity_gate(v5l.EXPECTATION_PLASTICITY_GATE, 0.0)
    output_gate_during_training = float(
        bridge._transmission_gate_values[v5l.EXPECTATION_OUTPUT_GATE]
    )
    training_rows = [
        _run_trial(bridge, handles, config, reward_action=0)
        for _ in range(config.smoke_training_trials)
    ]
    expectation_after = v5l._weight_means(
        bridge, handles["expectation_routes"]
    )
    weights_after_training = np.asarray(
        to_host(bridge.cp_connections.data), dtype=np.float32
    ).copy()

    bridge.set_plasticity_gate(CREDIT_PLASTICITY_GATE, 0.0)
    bridge.set_plasticity_gate(v5l.EXPECTATION_PLASTICITY_GATE, 0.0)
    v7._reset_measured_trial(bridge, config)
    bridge.set_transmission_gate(
        v5l.EXPECTATION_OUTPUT_GATE,
        0.0 if mode == "expectation_output_lesion" else 1.0,
    )
    probe_rows = [
        _run_trial(
            bridge,
            handles,
            config,
            reward_action=0 if probe == "reward" else None,
            scheduled_reward=None if probe == "reward" else False,
        )
        for _ in range(4)
    ]
    weights_after_probe = np.asarray(
        to_host(bridge.cp_connections.data), dtype=np.float32
    )
    changed_training = np.abs(weights_after_training - weights_before) > 1e-7
    declared = np.zeros(changed_training.shape, dtype=bool)
    declared[handles["routes"].all_indices()] = True
    return {
        "mode": mode,
        "probe": probe,
        "seed": int(seed),
        "science_seed_executed": False,
        "plateau_center": config.action_tag_center,
        "expectation_output_gate_during_training": output_gate_during_training,
        "expectation_output_gate_during_probe": float(
            bridge._transmission_gate_values[v5l.EXPECTATION_OUTPUT_GATE]
        ),
        "expectation_weight_before": expectation_before,
        "expectation_weight_after_training": expectation_after,
        "baseline_probe": baseline_probe,
        "clean_training_trials": int(sum(
            row["winner"] is not None for row in training_rows
        )),
        "rewarded_training_trials": int(sum(
            row["reward_delivered"] for row in training_rows
        )),
        "changed_training_synapses": int(changed_training.sum()),
        "changed_training_outside_declared_routes": int(
            np.logical_and(changed_training, ~declared).sum()
        ),
        "probe_weights_unchanged": bool(
            np.array_equal(weights_after_training, weights_after_probe)
        ),
        "training_rows": training_rows,
        "probe_rows": probe_rows,
    }


def _fraction_reduction(treatment: float, control: float) -> float | None:
    if control <= 0.0:
        return None
    return float((control - treatment) / control)


def _mean_path(rows: list[dict[str, object]], *path: object) -> float:
    values = []
    for row in rows:
        value = row
        for key in path:
            value = value[key]
        values.append(float(value))
    return float(np.mean(values)) if values else 0.0


def _selected_rows(
    conditions: dict[str, dict[str, object]],
    probe: str,
    mode: str,
    action: int,
) -> list[dict[str, object]]:
    return [
        row
        for row in conditions[f"{mode}_{probe}"]["probe_rows"]
        if row["winner"] == action
    ]


def run_output_smoke(*, seed: int = SMOKE_SEED) -> dict[str, object]:
    validate_smoke_seed(seed)
    modes = (
        "output_intact",
        "expectation_output_lesion",
        "expectation_learning_lesion",
    )
    probes = ("reward", "omission")
    conditions = {
        f"{mode}_{probe}": run_output_condition(
            mode, probe, seed=seed
        )
        for mode in modes
        for probe in probes
    }
    action_sequences = {
        probe: {
            mode: [
                row["winner"]
                for row in conditions[f"{mode}_{probe}"]["probe_rows"]
            ]
            for mode in modes
        }
        for probe in probes
    }
    selected = {
        probe: {
            mode: {
                action: _selected_rows(conditions, probe, mode, action)
                for action in CHANNELS
            }
            for mode in modes
        }
        for probe in probes
    }

    def mean(probe: str, mode: str, action: int, *path: object) -> float:
        return _mean_path(selected[probe][mode][action], *path)

    reward_burst = {
        mode: mean("reward", mode, 0, "dopamine_burst_depth")
        for mode in modes
    }
    reward_suppression = {
        control: _fraction_reduction(
            reward_burst["output_intact"], reward_burst[control]
        )
        for control in (
            "expectation_output_lesion", "expectation_learning_lesion"
        )
    }
    omission_dip = {
        mode: mean("omission", mode, 0, "dopamine_dip_depth")
        for mode in modes
    }
    omission_dip_retained = {
        control: (
            omission_dip[control]
            / max(omission_dip["output_intact"], 1e-12)
        )
        for control in (
            "expectation_output_lesion", "expectation_learning_lesion"
        )
    }
    matched_sequences = all(
        all(
            action_sequences[probe][mode]
            == action_sequences[probe]["output_intact"]
            for mode in modes[1:]
        )
        for probe in probes
    )
    both_actions_sampled = all(
        all(
            selected[probe][mode][action]
            for action in CHANNELS
        )
        for probe in probes
        for mode in modes
    )

    def expectation(probe: str, mode: str, action: int, channel: int) -> float:
        return mean(probe, mode, action, "delay", "expectation", channel)

    def output_retains_expectation(probe: str) -> bool:
        intact = expectation(probe, "output_intact", 0, 0)
        lesioned = expectation(probe, "expectation_output_lesion", 0, 0)
        return bool(
            intact > 0.0
            and 0.80 * intact <= lesioned <= 1.20 * intact
        )

    action_conditioned_means = {
        probe: {
            mode: {
                str(action): {
                    "delay_expectation": [
                        expectation(probe, mode, action, channel)
                        for channel in CHANNELS
                    ],
                    "gabab_snc_before_outcome": mean(
                        probe, mode, action,
                        "gabab_snc_before_outcome_mean",
                    ),
                    "dopamine_burst_depth": mean(
                        probe, mode, action, "dopamine_burst_depth"
                    ),
                    "dopamine_dip_depth": mean(
                        probe, mode, action, "dopamine_dip_depth"
                    ),
                    "outcome": {
                        region: mean(
                            probe, mode, action, "outcome", region
                        )
                        for region in ("snc", "lhb", "rmtg")
                    },
                }
                for action in CHANNELS
            }
            for mode in modes
        }
        for probe in probes
    }

    checks = {
        "reserved_seed_only": int(seed) == SMOKE_SEED,
        "formal_execution_remains_sealed": OPEN_PHASES == (),
        "locked_center_two_only": all(
            condition["plateau_center"] == 2.0
            for condition in conditions.values()
        ),
        "all_conditions_train_with_output_closed": all(
            condition["expectation_output_gate_during_training"] == 0.0
            for condition in conditions.values()
        ),
        "probe_output_gates_match_lesions": all(
            condition["expectation_output_gate_during_probe"]
            == (0.0 if condition["mode"] == "expectation_output_lesion" else 1.0)
            for condition in conditions.values()
        ),
        "baseline_expectation_is_zero": all(
            condition["baseline_probe"]["delay"]["expectation"] == [0, 0]
            for condition in conditions.values()
        ),
        "all_training_actions_are_clean": all(
            condition["clean_training_trials"]
            >= 0.9 * v9_config(2.0).smoke_training_trials
            for condition in conditions.values()
        ),
        "fixed_four_probe_blocks": all(
            len(condition["probe_rows"]) == 4
            for condition in conditions.values()
        ),
        "reward_probe_is_delivered_and_omission_is_withheld": bool(
            all(
                row["reward_delivered"] == (row["winner"] == 0)
                for mode in modes
                for row in conditions[f"{mode}_reward"]["probe_rows"]
            )
            and not any(
                row["reward_delivered"]
                for mode in modes
                for row in conditions[f"{mode}_omission"]["probe_rows"]
            )
        ),
        "intact_expectation_is_action_specific_before_both_outcomes": bool(
            all(
                expectation(probe, "output_intact", 0, 0) > 0.0
                and expectation(probe, "output_intact", 0, 0)
                >= 3.0 * expectation(probe, "output_intact", 0, 1)
                for probe in probes
            )
        ),
        "learning_lesion_removes_80pct_of_probe_expectation": all(
            expectation(probe, "expectation_learning_lesion", 0, 0)
            <= 0.20 * expectation(probe, "output_intact", 0, 0)
            for probe in probes
        ),
        "intact_expectation_creates_snc_gabab_before_outcomes": bool(
            all(
                mean(
                    probe, "output_intact", 0,
                    "gabab_snc_before_outcome_mean",
                ) > 0.0
                and mean(
                    probe, "expectation_output_lesion", 0,
                    "gabab_snc_before_outcome_mean",
                ) == 0.0
                for probe in probes
            )
        ),
        "output_lesion_retains_expectation_within_20pct": all(
            output_retains_expectation(probe) for probe in probes
        ),
        "dopamine_reward_burst_is_suppressed_20pct": all(
            value is not None and value >= 0.20
            for value in reward_suppression.values()
        ),
        "intact_snc_outcome_spikes_do_not_increase": all(
            mean("reward", "output_intact", 0, "outcome", "snc")
            <= mean("reward", control, 0, "outcome", "snc")
            for control in (
                "expectation_output_lesion", "expectation_learning_lesion"
            )
        ),
        "intact_omission_recruits_lhb_rmtg_and_dopamine_dip": bool(
            mean("omission", "output_intact", 0, "outcome", "lhb") > 0.0
            and mean("omission", "output_intact", 0, "outcome", "rmtg") > 0.0
            and omission_dip["output_intact"] > 0.0
        ),
        "omission_lesions_remove_80pct_lhb_and_rmtg": all(
            mean("omission", control, 0, "outcome", region)
            <= 0.20 * mean(
                "omission", "output_intact", 0, "outcome", region
            )
            for control in (
                "expectation_output_lesion", "expectation_learning_lesion"
            )
            for region in ("lhb", "rmtg")
        ),
        "omission_lesions_remove_half_dopamine_dip": all(
            value <= 0.50 for value in omission_dip_retained.values()
        ),
        "untrained_action_expectation_does_not_exceed_trained_action": all(
            expectation(probe, "output_intact", 1, 1)
            <= expectation(probe, "output_intact", 0, 0)
            for probe in probes
        ),
        "training_plasticity_is_confined": all(
            condition["changed_training_outside_declared_routes"] == 0
            for condition in conditions.values()
        ),
        "probe_weights_are_frozen": all(
            condition["probe_weights_unchanged"]
            for condition in conditions.values()
        ),
    }
    preconditions = [
        {
            "name": "all_six_conditions_completed",
            "ok": len(conditions) == 6,
            "observed": sorted(conditions),
            "expected_count": 6,
        },
        {
            "name": "matched_four_probe_action_sequences",
            "ok": matched_sequences,
            "observed": action_sequences,
            "expected": "identical across lesions within each probe type",
        },
        {
            "name": "both_actions_sampled_in_every_condition",
            "ok": both_actions_sampled,
            "observed": {
                probe: {
                    mode: {
                        str(action): len(selected[probe][mode][action])
                        for action in CHANNELS
                    }
                    for mode in modes
                }
                for probe in probes
            },
            "expected": "at least one row for each action",
        },
    ]
    prerequisites_hold = all(item["ok"] for item in preconditions)
    protocol_check_names = (
        "reserved_seed_only",
        "formal_execution_remains_sealed",
        "locked_center_two_only",
        "all_conditions_train_with_output_closed",
        "probe_output_gates_match_lesions",
        "baseline_expectation_is_zero",
        "all_training_actions_are_clean",
        "fixed_four_probe_blocks",
        "reward_probe_is_delivered_and_omission_is_withheld",
        "training_plasticity_is_confined",
        "probe_weights_are_frozen",
    )
    protocol_holds = all(checks[name] for name in protocol_check_names)
    xp, _ = get_backend()
    return {
        "artifact_schema_version": 1,
        "probe": "vocal_action_credit_gate_b_v9_gabab_output_smoke",
        "seed": int(seed),
        "science_seed_executed": False,
        "backend": "cupy" if xp.__name__ == "cupy" else "numpy",
        "device": (
            xp.cuda.runtime.getDeviceProperties(0)["name"].decode("utf-8")
            if xp.__name__ == "cupy"
            else platform.processor() or platform.machine() or "CPU"
        ),
        "host_boundary": dict(HOST_BOUNDARY),
        "conditions": conditions,
        "probe_action_sequences": action_sequences,
        "action_conditioned_means": action_conditioned_means,
        "reward_suppression_fraction": reward_suppression,
        "omission_dip_retained_fraction": omission_dip_retained,
        "preconditions": preconditions,
        "protocol_check_names": list(protocol_check_names),
        "checks": checks,
        "status": (
            "OUTPUT_FAIL"
            if not protocol_holds
            else "UNDEFINED"
            if not prerequisites_hold
            else "OUTPUT_PASS"
            if all(checks.values())
            else "OUTPUT_FAIL"
        ),
    }


def _finish_engagement_smoke(
    center: float,
    seed: int,
    config: VocalCreditConfigV9,
    conditions: dict[str, dict[str, object]],
) -> dict[str, object]:
    intact = conditions["intact"]
    learning_lesion = conditions["expectation_learning_lesion"]
    plateau_lesion = conditions["expectation_plateau_lesion"]
    initial = float(intact["expectation_weight_before"][0])
    intact_delta = float(intact["expectation_weight_after"][0]) - initial
    intact_separation = (
        float(intact["expectation_weight_after"][0])
        - float(intact["expectation_weight_after"][1])
    )
    learning_lesion_delta = max(
        abs(float(after) - float(before))
        for before, after in zip(
            learning_lesion["expectation_weight_before"],
            learning_lesion["expectation_weight_after"],
        )
    )
    delays = {
        mode: _late_delay_expectation(condition)
        for mode, condition in conditions.items()
    }
    intact_delay = delays["intact"]
    expectation_plateau = {
        mode: _window_sum(
            condition, "graded_plateau_expectation_integral"
        )
        for mode, condition in conditions.items()
    }
    trace_plateau = {
        mode: _window_sum(
            condition,
            "graded_plateau_trace_integral",
            windows=("action", "delay"),
        )
        for mode, condition in conditions.items()
    }
    lesion_audit = plateau_lesion["plateau_lesion_audit"] or {}
    checks = {
        "reserved_seed_only": int(seed) == SMOKE_SEED,
        "formal_execution_remains_sealed": OPEN_PHASES == (),
        "at_least_90pct_clean_action_epochs": all(
            condition["clean_trials"] >= 0.9 * config.smoke_training_trials
            for condition in conditions.values()
        ),
        "upstream_action_trace_fires_before_outcome": all(
            sum(sum(row["delay"]["trace"]) for row in condition["rows"]) > 0
            for condition in conditions.values()
        ),
        "baseline_expectation_is_zero": all(
            sum(conditions[mode]["baseline_probe"]["delay"]["expectation"])
            == 0
            for mode in ("intact", "expectation_learning_lesion")
        ),
        "rewarded_route_learns_and_separates": bool(
            intact_delta >= 0.25 * initial
            and intact_separation >= 0.20 * initial
        ),
        "responsible_expectation_fires_before_outcome": bool(
            intact_delay[0] > 0
            and intact_delay[0] >= 3 * max(intact_delay[1], 1)
        ),
        "learning_lesion_blocks_route_change": learning_lesion_delta <= 1e-7,
        "learning_lesion_removes_80pct_of_delay_expectation": bool(
            intact_delay[0] > 0
            and delays["expectation_learning_lesion"][0]
            <= 0.20 * intact_delay[0]
        ),
        "plateau_lesion_removes_80pct_of_delay_expectation": bool(
            intact_delay[0] > 0
            and delays["expectation_plateau_lesion"][0]
            <= 0.20 * intact_delay[0]
        ),
        "expectation_plateau_is_present_and_causally_lesioned": bool(
            expectation_plateau["intact"][0] > 0.0
            and expectation_plateau["expectation_plateau_lesion"][0] == 0.0
            and lesion_audit.get("changed_only_expected_routes", False)
            and lesion_audit.get("other_dendritic_routes_unchanged", False)
            and lesion_audit.get("weights_unchanged", False)
        ),
        "upstream_trace_plateau_survives_expectation_lesion": bool(
            trace_plateau["expectation_plateau_lesion"][0] > 0.0
        ),
        "plasticity_confined_to_declared_routes": all(
            condition["changed_outside_declared_routes"] == 0
            for condition in conditions.values()
        ),
        "expectation_output_remains_closed": all(
            condition["expectation_output_gate"] == 0.0
            for condition in conditions.values()
        ),
    }
    preconditions = [
        {
            "name": "reserved_smoke_seed_only",
            "ok": int(seed) == SMOKE_SEED,
            "observed": int(seed),
            "expected": SMOKE_SEED,
        },
        {
            "name": "formal_execution_remains_sealed",
            "ok": OPEN_PHASES == (),
            "observed": list(OPEN_PHASES),
            "expected": [],
        },
        {
            "name": "all_conditions_completed_fixed_trial_count",
            "ok": all(
                len(condition["rows"]) == config.smoke_training_trials
                for condition in conditions.values()
            ),
            "observed": {
                mode: len(condition["rows"])
                for mode, condition in conditions.items()
            },
            "expected": config.smoke_training_trials,
        },
        {
            "name": "late_window_has_four_rewarded_trials",
            "ok": all(
                condition["rewarded_trials"] >= 4
                for condition in conditions.values()
            ),
            "observed": {
                mode: condition["rewarded_trials"]
                for mode, condition in conditions.items()
            },
            "expected_minimum": 4,
        },
        {
            "name": "learning_lesion_arm_is_inert",
            "ok": learning_lesion_delta <= 1e-7,
            "observed_max_absolute_weight_change": learning_lesion_delta,
            "expected_max_absolute_weight_change": 1e-7,
        },
        {
            "name": "plateau_lesion_changes_only_expectation_route_mask",
            "ok": bool(
                lesion_audit.get("changed_only_expected_routes", False)
                and lesion_audit.get("expected_routes_disabled", False)
                and lesion_audit.get("other_dendritic_routes_unchanged", False)
                and lesion_audit.get("weights_unchanged", False)
            ),
            "observed": lesion_audit,
            "expected": "only learned expectation-route dendritic bits cleared",
        },
        {
            "name": "expectation_output_is_closed_in_all_conditions",
            "ok": all(
                condition["expectation_output_gate"] == 0.0
                for condition in conditions.values()
            ),
            "observed": {
                mode: condition["expectation_output_gate"]
                for mode, condition in conditions.items()
            },
            "expected": 0.0,
        },
    ]
    prerequisites_hold = all(item["ok"] for item in preconditions)
    xp, _ = get_backend()
    config_payload = json.dumps(
        asdict(config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "artifact_schema_version": 1,
        "probe": "vocal_action_credit_gate_b_v9_graded_dendritic_engagement",
        "plateau_center": float(center),
        "seed": int(seed),
        "science_seed_executed": False,
        "backend": "cupy" if xp.__name__ == "cupy" else "numpy",
        "device": (
            xp.cuda.runtime.getDeviceProperties(0)["name"].decode("utf-8")
            if xp.__name__ == "cupy"
            else platform.processor() or platform.machine() or "CPU"
        ),
        "config_sha256": hashlib.sha256(config_payload).hexdigest(),
        "host_boundary": dict(HOST_BOUNDARY),
        "preconditions": preconditions,
        "config": asdict(config),
        "conditions": conditions,
        "late_rewarded_delay_expectation_spikes": delays,
        "expectation_plateau_integral": expectation_plateau,
        "trace_plateau_integral": trace_plateau,
        "delay_expectation_attributable_to_learning": attributable_to(
            "pre-outcome expectation from local trace learning",
            intact_delay[0],
            delays["expectation_learning_lesion"][0],
            warn_below=0.80,
        ),
        "delay_expectation_attributable_to_graded_plateau": attributable_to(
            "pre-outcome expectation from learned-route graded dendrite",
            intact_delay[0],
            delays["expectation_plateau_lesion"][0],
            warn_below=0.80,
        ),
        "checks": checks,
        "status": (
            "ENGAGEMENT_PASS"
            if prerequisites_hold and all(checks.values())
            else "ENGAGEMENT_FAIL"
            if prerequisites_hold
            else "UNDEFINED"
        ),
    }


def run_construction_smoke(
    center: float = PLATEAU_CENTER_LADDER[0],
    *,
    seed: int = SMOKE_SEED,
) -> dict[str, object]:
    config = v9_config(center)
    bridge, handles = build_v9_bridge(seed=seed, config=config)
    audit = structural_audit(bridge, handles)
    pathways = _v9_pathways(config)
    learned = [pathway for pathway in pathways if _is_expectation_route(pathway)]
    snc_outputs = [
        pathway for pathway in pathways
        if pathway.from_region in {
            v5l._expectation(channel) for channel in CHANNELS
        }
        and pathway.to_region == v5l.SNC
    ]
    checks = {
        "reserved_seed_only": int(seed) == SMOKE_SEED,
        "formal_execution_unassigned_and_sealed": OPEN_PHASES == (),
        "plateau_center_is_preregistered": float(center) in PLATEAU_CENTER_LADDER,
        "trace_population_is_locked_to_200": config.n_value == TRACE_SIZE,
        "learned_routes_are_local_plastic_and_dendritic": bool(
            len(learned) == len(CHANNELS)
            and all(pathway.plastic for pathway in learned)
            and all(pathway.density == 0.50 for pathway in learned)
            and all(pathway.coincidence_detector for pathway in learned)
        ),
        "shared_plateau_parameters_match_preregistration": bool(
            bridge.core_config.enable_graded_dendritic_plateau
            and bridge.core_config.coincidence_plateau_strength == 0.0
            and bridge.core_config.graded_plateau_center == float(center)
            and bridge.core_config.graded_plateau_slope == PLATEAU_SLOPE
            and bridge.core_config.graded_plateau_strength == PLATEAU_STRENGTH
            and bridge.core_config.graded_plateau_tau_rise_ms
            == PLATEAU_TAU_RISE_MS
            and bridge.core_config.graded_plateau_tau_decay_ms
            == PLATEAU_TAU_DECAY_MS
        ),
        "expectation_outputs_use_gabab": bool(
            len(snc_outputs) == len(CHANNELS)
            and all(pathway.receptor == "gaba_b" for pathway in snc_outputs)
        ),
        "plasticity_confined_to_declared_routes": bool(
            audit["plastic_synapses"] == audit["declared_plastic_synapses"]
            and audit["plastic_outside_declared_routes"] == 0
            and audit["fixed_inside_declared_routes"] == 0
        ),
        "runtime_ownership_matches_declared_routes": bool(
            audit["reward_eligibility_matches_declared_routes"]
            and audit["actor_gate_matches_actor_routes"]
            and audit["expectation_gate_matches_expectation_routes"]
        ),
        "coincidence_routes_match_loaders_plus_learned_routes": bool(
            audit["coincidence"]["enabled_synapses"]
            == audit["coincidence"]["intended_synapses"]
            and audit["coincidence"]["enabled_outside_intended_routes"] == 0
            and audit["coincidence"]["disabled_inside_intended_routes"] == 0
        ),
        "expectation_output_gate_is_closed_and_owned": bool(
            audit["expectation_output_gate_matches_routes"]
            and audit["expectation_output_gate_value"] == 0.0
        ),
    }
    preconditions = [
        {
            "name": "reserved_smoke_seed_only",
            "ok": int(seed) == SMOKE_SEED,
            "observed": int(seed),
            "expected": SMOKE_SEED,
        },
        {
            "name": "formal_execution_remains_sealed",
            "ok": OPEN_PHASES == (),
            "observed": list(OPEN_PHASES),
            "expected": [],
        },
        {
            "name": "plateau_center_is_preregistered",
            "ok": float(center) in PLATEAU_CENTER_LADDER,
            "observed": float(center),
            "expected": list(PLATEAU_CENTER_LADDER),
        },
    ]
    prerequisites_hold = all(item["ok"] for item in preconditions)
    config_payload = json.dumps(
        asdict(config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    xp, _ = get_backend()
    return {
        "artifact_schema_version": 1,
        "probe": "vocal_action_credit_gate_b_v9_graded_dendritic_construction",
        "plateau_center": float(center),
        "seed": int(seed),
        "science_seed_executed": False,
        "backend": "cupy" if xp.__name__ == "cupy" else "numpy",
        "device": (
            xp.cuda.runtime.getDeviceProperties(0)["name"].decode("utf-8")
            if xp.__name__ == "cupy"
            else platform.processor() or platform.machine() or "CPU"
        ),
        "config": asdict(config),
        "config_sha256": hashlib.sha256(config_payload).hexdigest(),
        "host_boundary": dict(HOST_BOUNDARY),
        "preconditions": preconditions,
        "audit": audit,
        "checks": checks,
        "status": (
            "CONSTRUCTION_PASS"
            if prerequisites_hold and all(checks.values())
            else "CONSTRUCTION_FAIL"
            if prerequisites_hold
            else "UNDEFINED"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SMOKE_SEED)
    parser.add_argument(
        "--plateau-center",
        type=float,
        choices=PLATEAU_CENTER_LADDER,
        default=PLATEAU_CENTER_LADDER[0],
    )
    parser.add_argument("--engagement", action="store_true")
    parser.add_argument("--output-smoke", action="store_true")
    parser.add_argument("--formal-phase")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.formal_phase is not None:
        validate_phase(args.formal_phase)
    validate_smoke_seed(args.seed)
    if args.engagement and args.output_smoke:
        parser.error("--engagement and --output-smoke are mutually exclusive")
    if args.output_smoke and args.plateau_center != 16.0:
        parser.error("Phase-2 center is locked internally at 2; omit --plateau-center")
    result = (
        run_output_smoke(seed=args.seed)
        if args.output_smoke
        else run_engagement_smoke(args.plateau_center, seed=args.seed)
        if args.engagement
        else run_construction_smoke(args.plateau_center, seed=args.seed)
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0 if result["status"] in {
        "CONSTRUCTION_PASS", "ENGAGEMENT_PASS", "OUTPUT_PASS"
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
