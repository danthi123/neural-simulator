"""Sealed Gate B v10 corticostriatal policy-eligibility smoke.

Reserved seed zero may test whether the unchanged Gate A v2 selector leaves
action-local eligibility on its proposal-to-D1/D2 policy routes. This phase
delivers no reward, applies no dopamine teaching signal, and changes no weight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

from research.runners._vocal_action_credit_gate import _route_synapses
from research.runners._vocal_action_selector_gate import (
    CHANNELS,
    _indices,
    _set_equal_tonic_current,
    build_selector_bridge,
    selector_config,
)
from sim.backend import get_backend, to_host
from tools.lab import attributable_to


OPEN_PHASES: tuple[str, ...] = ()
SMOKE_SEED = 0
POLICY_GATE = "vocal_selector_policy_learning"
TRIALS = 12
WARMUP_STEPS = 80
ACTION_STEPS = 600
REWARD_DELAY_STEPS = 100
RESET_STEPS = 35
WASHOUT_STEPS = 3000
ELIGIBILITY_TAU_MS = 300.0
COACTIVITY_TAU_MS = 80.0
COACTIVITY_THRESHOLD = 0.001
COACTIVITY_SCALE = 20.0
WEIGHT_MIN = 0.0
WEIGHT_MAX = 600.0

HOST_BOUNDARY = {
    "host_selects_action": False,
    "host_argmax_or_fallback": False,
    "host_forces_action": False,
    "host_labels_eligible_channel": False,
    "host_edits_eligibility": False,
    "host_edits_coactivity_trace": False,
    "host_delivers_reward": False,
    "host_calculates_teaching_signal": False,
    "symmetric_fixed_phase_measurement_gates": True,
    "formal_execution_open": False,
}


def validate_smoke_seed(seed: int) -> int:
    checked = int(seed)
    if checked != SMOKE_SEED:
        raise ValueError(f"v10 engagement accepts reserved seed {SMOKE_SEED} only")
    return checked


def validate_phase(phase: str) -> str:
    raise ValueError(
        "Gate B v10 formal phases are sealed and unassigned; "
        f"open phases={OPEN_PHASES}, requested={phase!r}"
    )


def run_formal_seed(seed: int):
    raise ValueError(f"Gate B v10 formal execution is sealed; received={int(seed)}")


def _core_updates(*, coactivity: bool, diagnostic_signal: float = 0.0):
    return {
        "enable_reward_modulation": True,
        "enable_neuromodulator_subsystem": False,
        "enable_d1_d2_asymmetry": True,
        "reward_defer_stdp_weight_update": True,
        "reward_learning_rate": 0.0,
        "reward_eligibility_tau_ms": ELIGIBILITY_TAU_MS,
        "reward_baseline": 0.0,
        "current_reward_signal": float(diagnostic_signal),
        "reward_eligibility_from_coactivity": bool(coactivity),
        "reward_coactivity_trace_tau_ms": COACTIVITY_TAU_MS,
        "reward_coactivity_trace_input_gain": 0.0,
        "reward_coactivity_threshold": COACTIVITY_THRESHOLD,
        "reward_coactivity_scale": COACTIVITY_SCALE,
        "enable_stdp": False,
        "enable_hebbian_learning": False,
        "enable_homeostasis": False,
        "enable_structural_plasticity": False,
        "stdp_w_min": WEIGHT_MIN,
        "stdp_w_max": WEIGHT_MAX,
        "hebbian_min_weight": WEIGHT_MIN,
        "hebbian_max_weight": WEIGHT_MAX,
    }


def _policy_routes(bridge) -> dict[str, dict[int, np.ndarray]]:
    return {
        route_class: {
            channel: _route_synapses(
                bridge, f"proposal_{channel}", f"str_{route_class}_{channel}"
            )
            for channel in CHANNELS
        }
        for route_class in ("d1", "d2")
    }


def _all_policy_indices(routes: dict[str, dict[int, np.ndarray]]) -> np.ndarray:
    return np.sort(np.concatenate([
        routes[route_class][channel]
        for route_class in ("d1", "d2")
        for channel in CHANNELS
    ]))


def build_v10_bridge(
    seed: int = SMOKE_SEED,
    *,
    coactivity: bool = True,
    diagnostic_signal: float = 0.0,
):
    validate_smoke_seed(seed)
    bridge = build_selector_bridge(
        seed,
        selector_config("v2"),
        policy_plasticity_gate=POLICY_GATE,
        core_config_updates=_core_updates(
            coactivity=coactivity, diagnostic_signal=diagnostic_signal
        ),
    )
    bridge.strict_step_errors = True
    routes = _policy_routes(bridge)
    all_policy = _all_policy_indices(routes)
    xp, _ = get_backend()
    bridge.cp_reward_eligibility_synapse_indices = xp.asarray(
        all_policy, dtype=xp.int64
    )
    bridge.set_plasticity_gate(POLICY_GATE, 0.0)
    bridge.core_config.reward_coactivity_trace_input_gain = 0.0
    return bridge, {
        "policy_routes": routes,
        "all_policy": all_policy,
        "expected_coactivity": bool(coactivity),
    }


def _weights(bridge) -> np.ndarray:
    return np.asarray(to_host(bridge.cp_connections.data), dtype=np.float32).copy()


def _hash_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _route_means(bridge, handles: dict) -> dict[str, dict[str, float]]:
    trace = np.asarray(to_host(bridge.cp_eligibility_trace), dtype=np.float64)
    return {
        route_class: {
            str(channel): float(np.mean(np.abs(
                trace[handles["policy_routes"][route_class][channel]]
            )))
            for channel in CHANNELS
        }
        for route_class in ("d1", "d2")
    }


def _max_policy_eligibility(bridge, handles: dict) -> float:
    trace = np.asarray(to_host(bridge.cp_eligibility_trace), dtype=np.float64)
    return float(np.max(np.abs(trace[handles["all_policy"]]), initial=0.0))


def _max_coactivity_trace(bridge) -> float:
    trace = getattr(bridge, "cp_reward_coactivity_trace", None)
    if trace is None:
        return 0.0
    host = np.asarray(to_host(trace), dtype=np.float64)
    return float(np.max(np.abs(host), initial=0.0))


def _coactivity_trace_snapshot(bridge) -> dict[str, object]:
    trace = getattr(bridge, "cp_reward_coactivity_trace", None)
    allocated = trace is not None
    host = (
        np.asarray(to_host(trace), dtype=np.float32)
        if allocated else np.zeros(int(bridge.core_config.num_neurons), dtype=np.float32)
    )
    return {
        "allocated": allocated,
        "values": host.tolist(),
        "hash": _hash_array(host),
        "max_absolute": float(np.max(np.abs(host), initial=0.0)),
    }


def _snapshot(bridge, handles: dict) -> dict[str, object]:
    return {
        "route_means": _route_means(bridge, handles),
        "max_policy_eligibility": _max_policy_eligibility(bridge, handles),
    }


def structural_audit(bridge, handles: dict) -> dict[str, object]:
    cfg = bridge.core_config
    policy = handles["all_policy"]
    plastic = np.asarray(to_host(bridge.cp_synapse_plastic_mask), dtype=bool)
    eligible = np.sort(np.asarray(to_host(
        bridge.cp_reward_eligibility_synapse_indices
    ), dtype=np.int64))
    gated = np.sort(np.asarray(to_host(
        bridge._plasticity_gate_indices_gpu[POLICY_GATE]
    ), dtype=np.int64))
    expected_mask = np.zeros(plastic.shape, dtype=bool)
    expected_mask[policy] = True
    signs = np.asarray(to_host(bridge.cp_d1_d2_sign), dtype=np.float32)
    route_signs = {
        route_class: sorted(set(float(value) for value in signs[np.concatenate([
            handles["policy_routes"][route_class][channel] for channel in CHANNELS
        ])]))
        for route_class in ("d1", "d2")
    }
    region_names = {region.name for region in bridge.region_manager.regions()}
    forbidden_tokens = (
        "vocal_credit", "credit_cue", "expectation", "reward_us", "snc",
        "habenula", "rmtg", "omission", "outcome",
    )
    policy_pathways = [
        pathway for pathway in cfg.region_pathways
        if pathway.from_region.startswith("proposal_")
        and pathway.to_region.startswith(("str_d1_", "str_d2_"))
    ]
    weights = _weights(bridge)
    policy_weights = weights[policy]
    route_sizes = {
        f"{route_class}_{channel}": int(
            handles["policy_routes"][route_class][channel].size
        )
        for route_class in ("d1", "d2") for channel in CHANNELS
    }
    checks = {
        "selector_v2_has_600_neurons": int(cfg.num_neurons) == 600,
        "selector_has_36_declared_pathways": len(cfg.region_pathways) == 36,
        "no_actor_or_reward_regions": not any(
            token in name.lower() for name in region_names for token in forbidden_tokens
        ),
        "exactly_four_policy_pathways": len(policy_pathways) == 4,
        "policy_pathway_declarations_are_locked": all(
            pathway.density == 1.0
            and pathway.weight_mean == 400.0
            and pathway.weight_jitter == 0.05
            and pathway.plastic is True
            and pathway.plasticity_gate == POLICY_GATE
            for pathway in policy_pathways
        ),
        "each_policy_route_has_2160_synapses": all(
            size == 60 * 36 for size in route_sizes.values()
        ),
        "policy_pathways_keep_ordinary_transmission": all(
            pathway.transmission_gate is None for pathway in policy_pathways
        ),
        "plastic_set_equals_policy": np.array_equal(
            np.flatnonzero(plastic), policy
        ),
        "eligible_set_equals_policy": np.array_equal(eligible, policy),
        "policy_gate_equals_policy": np.array_equal(gated, policy),
        "policy_gate_starts_closed": (
            bridge._plasticity_gate_values[POLICY_GATE] == 0.0
        ),
        "d1_signs_are_positive": route_signs["d1"] == [1.0],
        "d2_signs_are_negative": route_signs["d2"] == [-1.0],
        "policy_weights_strictly_inside_bounds": bool(
            np.all(policy_weights > WEIGHT_MIN)
            and np.all(policy_weights < WEIGHT_MAX)
        ),
        "reward_learning_rate_is_zero": cfg.reward_learning_rate == 0.0,
        "dt_is_one_ms": cfg.dt_ms == 1.0,
        "reward_modulation_and_deferred_update_are_on": (
            cfg.enable_reward_modulation
            and cfg.reward_defer_stdp_weight_update
        ),
        "eligibility_constants_are_locked": (
            cfg.reward_eligibility_tau_ms == ELIGIBILITY_TAU_MS
            and cfg.reward_coactivity_trace_tau_ms == COACTIVITY_TAU_MS
            and cfg.reward_coactivity_threshold == COACTIVITY_THRESHOLD
            and cfg.reward_coactivity_scale == COACTIVITY_SCALE
        ),
        "coactivity_condition_matches_construction": (
            cfg.reward_eligibility_from_coactivity
            == handles["expected_coactivity"]
            and (bridge.cp_reward_coactivity_trace is not None)
            == handles["expected_coactivity"]
        ),
        "reward_signal_and_baseline_are_zero": (
            cfg.current_reward_signal == 0.0 and cfg.reward_baseline == 0.0
        ),
        "neuromodulator_subsystem_is_off": (
            not cfg.enable_neuromodulator_subsystem
            and bridge.neuromodulator_manager is None
        ),
        "nonreward_plasticity_is_off": not any((
            cfg.enable_stdp,
            cfg.enable_hebbian_learning,
            cfg.enable_homeostasis,
            cfg.enable_structural_plasticity,
        )),
        "hebbian_bounds_are_locked": (
            cfg.hebbian_min_weight == WEIGHT_MIN
            and cfg.hebbian_max_weight == WEIGHT_MAX
        ),
        "inactive_stdp_bounds_are_locked": (
            cfg.stdp_w_min == WEIGHT_MIN and cfg.stdp_w_max == WEIGHT_MAX
        ),
        "coactivity_input_gate_starts_closed": (
            cfg.reward_coactivity_trace_input_gain == 0.0
        ),
    }
    return {
        "checks": checks,
        "all_checks_pass": bool(all(checks.values())),
        "policy_synapses": int(policy.size),
        "plastic_synapses": int(plastic.sum()),
        "initial_policy_weight_min": float(policy_weights.min()),
        "initial_policy_weight_max": float(policy_weights.max()),
        "route_signs": route_signs,
        "route_sizes": route_sizes,
        "region_names": sorted(region_names),
    }


def _phase_gate_telemetry(bridge, *, phase: str) -> dict[str, object]:
    policy = bridge._plasticity_gate_indices_gpu[POLICY_GATE]
    gains = np.asarray(to_host(
        bridge.cp_plasticity_rate_gain[policy]
    ), dtype=np.float32)
    transmission = (
        np.asarray(to_host(bridge.cp_transmission_gain[policy]), dtype=np.float32)
        if bridge.cp_transmission_gain is not None
        else np.ones(policy.size, dtype=np.float32)
    )
    return {
        "phase": phase,
        "declared_policy_gain": float(
            bridge._plasticity_gate_values[POLICY_GATE]
        ),
        "actual_policy_gain_values": sorted(set(float(v) for v in gains)),
        "coactivity_trace_input_gain": float(
            bridge.core_config.reward_coactivity_trace_input_gain
        ),
        "policy_transmission_gain_values": sorted(
            set(float(v) for v in transmission)
        ),
    }


def _set_phase_gates(bridge, *, action: bool, phase: str) -> dict[str, object]:
    value = float(bool(action))
    bridge.set_plasticity_gate(POLICY_GATE, value)
    bridge.core_config.reward_coactivity_trace_input_gain = value
    return _phase_gate_telemetry(bridge, phase=phase)


def _step_and_hash(bridge, digest, *, n: int = 1):
    firing = None
    for _ in range(int(n)):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        firing = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        digest.update(np.packbits(firing).tobytes())
    return firing


def _diagnostic_regions() -> list[str]:
    names = ["practice_arousal", "selector_stn", "selector_reset"]
    for channel in CHANNELS:
        names.extend([
            f"proposal_{channel}", f"str_d1_{channel}", f"str_d2_{channel}",
            f"gpe_{channel}", f"gpi_{channel}", f"thal_{channel}",
            f"commit_{channel}", f"commit_fs_{channel}", f"motor_{channel}",
        ])
    return names


def _run_trial(bridge, handles: dict, selector, digest) -> dict[str, object]:
    xp, _ = get_backend()
    baseline = _snapshot(bridge, handles)
    pretrace = _coactivity_trace_snapshot(bridge)
    _set_equal_tonic_current(bridge, selector)
    bridge.cp_external_input_current[
        xp.asarray(_indices(bridge, "practice_arousal"))
    ] = xp.float32(selector.practice_pA)
    gate_schedule = {
        "pre_action": _phase_gate_telemetry(bridge, phase="pre_action"),
        "action": _set_phase_gates(bridge, action=True, phase="action"),
    }

    regions = _diagnostic_regions()
    indices = {name: _indices(bridge, name) for name in regions}
    region_spikes = {name: 0 for name in regions}
    motor_counts = np.zeros(2, dtype=np.int64)
    first_crossing = None
    decision_step = None
    decision_counts = None
    decision_snapshot = None
    simultaneous = False

    for step in range(ACTION_STEPS):
        firing = _step_and_hash(bridge, digest)
        for name in regions:
            region_spikes[name] += int(firing[indices[name]].sum())
        previous = motor_counts.copy()
        for channel in CHANNELS:
            motor_counts[channel] += int(
                firing[indices[f"motor_{channel}"]].sum()
            )
        crossed = [
            channel for channel in CHANNELS
            if previous[channel] < selector.commit_threshold_spikes
            <= motor_counts[channel]
        ]
        if decision_step is None and crossed:
            decision_step = int(step)
            decision_counts = motor_counts.copy()
            decision_snapshot = _snapshot(bridge, handles)
            if len(crossed) == 1:
                first_crossing = int(crossed[0])
            else:
                simultaneous = True

    action_end = _snapshot(bridge, handles)
    gate_schedule["reward_delay"] = _set_phase_gates(
        bridge, action=False, phase="reward_delay"
    )
    _set_equal_tonic_current(bridge, selector)
    _step_and_hash(bridge, digest, n=REWARD_DELAY_STEPS)
    pre_outcome = _snapshot(bridge, handles)

    winner = None
    cleanliness_reason = "no_unique_crossing"
    loser_ratio = None
    if first_crossing is not None and not simultaneous:
        loser = 1 - first_crossing
        loser_ratio = float(
            decision_counts[loser] / max(1, decision_counts[first_crossing])
        )
        if loser_ratio > selector.clean_loser_ratio:
            cleanliness_reason = "loser_ratio_at_crossing"
        elif motor_counts[loser] >= selector.commit_threshold_spikes:
            cleanliness_reason = "loser_crossed_later"
        else:
            winner = int(first_crossing)
            cleanliness_reason = "clean"

    _set_equal_tonic_current(bridge, selector)
    bridge.cp_external_input_current[
        xp.asarray(_indices(bridge, "selector_reset"))
    ] = xp.float32(selector.reset_pA)
    gate_schedule["reset"] = _phase_gate_telemetry(bridge, phase="reset")
    _step_and_hash(bridge, digest, n=RESET_STEPS)
    _set_equal_tonic_current(bridge, selector)
    gate_schedule["washout"] = _phase_gate_telemetry(bridge, phase="washout")
    _step_and_hash(bridge, digest, n=WASHOUT_STEPS)
    return {
        "winner": winner,
        "first_crossing": first_crossing,
        "decision_step": decision_step,
        "simultaneous_crossing": simultaneous,
        "cleanliness_reason": cleanliness_reason,
        "loser_ratio_at_crossing": loser_ratio,
        "decision_motor_counts": (
            decision_counts.tolist() if decision_counts is not None else None
        ),
        "motor_counts_full_window": motor_counts.tolist(),
        "region_spikes": region_spikes,
        "pretrial": baseline,
        "preaction_coactivity_trace": pretrace,
        "phase_gate_schedule": gate_schedule,
        "snapshots": {
            "decision": decision_snapshot,
            "action_end": action_end,
            "pre_outcome": pre_outcome,
        },
    }


def _with_net_values(row: dict) -> dict:
    baseline = row["pretrial"]["route_means"]
    elapsed = {
        "decision": (
            None if row["decision_step"] is None
            else float(row["decision_step"] + 1)
        ),
        "action_end": float(ACTION_STEPS),
        "pre_outcome": float(ACTION_STEPS + REWARD_DELAY_STEPS),
    }
    for snapshot_name, snapshot in row["snapshots"].items():
        if snapshot is None:
            continue
        decay_ms = elapsed[snapshot_name]
        decay = math.exp(-decay_ms / ELIGIBILITY_TAU_MS)
        snapshot["net_route_means"] = {
            route_class: {
                str(channel): float(max(
                    snapshot["route_means"][route_class][str(channel)]
                    - baseline[route_class][str(channel)] * decay,
                    0.0,
                ))
                for channel in CHANNELS
            }
            for route_class in ("d1", "d2")
        }
    return row


def run_condition(*, mode: str, seed: int = SMOKE_SEED) -> dict[str, object]:
    if mode not in {"intact", "coactivity_lesion"}:
        raise ValueError(f"unknown v10 engagement mode: {mode}")
    coactivity = mode == "intact"
    bridge, handles = build_v10_bridge(seed, coactivity=coactivity)
    selector = selector_config("v2")
    structure = structural_audit(bridge, handles)
    initial = _weights(bridge)
    digest = hashlib.sha256()
    warmup_gate = _set_phase_gates(
        bridge, action=False, phase="warmup"
    )
    _set_equal_tonic_current(bridge, selector)
    _step_and_hash(bridge, digest, n=WARMUP_STEPS)
    rows = [
        _with_net_values(_run_trial(bridge, handles, selector, digest))
        for _ in range(TRIALS)
    ]
    final = _weights(bridge)
    return {
        "mode": mode,
        "seed": int(seed),
        "structure": structure,
        "initial_weight_hash": _hash_array(initial),
        "final_weight_hash": _hash_array(final),
        "weights_byte_identical": bool(np.array_equal(initial, final)),
        "firing_sequence_hash": digest.hexdigest(),
        "warmup_gate": warmup_gate,
        "rows": rows,
    }


def run_clip_path_control(seed: int = SMOKE_SEED) -> dict[str, object]:
    bridge, handles = build_v10_bridge(
        seed, coactivity=False, diagnostic_signal=1.0
    )
    bridge.set_plasticity_gate(POLICY_GATE, 1.0)
    initial = _weights(bridge)
    bridge._run_one_simulation_step()
    bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
    final = _weights(bridge)
    policy = initial[handles["all_policy"]]
    return {
        "diagnostic_reward_signal": float(bridge.core_config.current_reward_signal),
        "reward_learning_rate": float(bridge.core_config.reward_learning_rate),
        "coactivity_enabled": bool(
            bridge.core_config.reward_eligibility_from_coactivity
        ),
        "initial_policy_weights_inside_hebbian_bounds": bool(
            np.all(policy > WEIGHT_MIN) and np.all(policy < WEIGHT_MAX)
        ),
        "initial_weight_hash": _hash_array(initial),
        "final_weight_hash": _hash_array(final),
        "weights_byte_identical": bool(np.array_equal(initial, final)),
    }


def _rows_match(intact: dict, lesion: dict) -> bool:
    keys = (
        "winner", "first_crossing", "decision_step", "simultaneous_crossing",
        "cleanliness_reason", "decision_motor_counts", "motor_counts_full_window",
    )
    return all(
        all(left[key] == right[key] for key in keys)
        for left, right in zip(intact["rows"], lesion["rows"], strict=True)
    )


def _washout_passes(rows: list[dict]) -> bool:
    for trial_index, row in enumerate(rows):
        route_means = row["pretrial"]["route_means"]
        if trial_index == 0:
            if row["pretrial"]["max_policy_eligibility"] != 0.0:
                return False
            if any(
                route_means[kind][str(channel)] != 0.0
                for kind in ("d1", "d2") for channel in CHANNELS
            ):
                return False
            continue
        previous = rows[trial_index - 1]["snapshots"]["pre_outcome"][
            "route_means"
        ]
        if row["pretrial"]["max_policy_eligibility"] > 0.01:
            return False
        for kind in ("d1", "d2"):
            for channel in CHANNELS:
                current = route_means[kind][str(channel)]
                prior = previous[kind][str(channel)]
                if prior == 0.0:
                    if current != 0.0:
                        return False
                elif current > 0.001 * prior:
                    return False
    return True


def _pretrace_passes(rows: list[dict]) -> bool:
    return bool(
        rows[0]["preaction_coactivity_trace"]["max_absolute"] == 0.0
        and all(
            row["preaction_coactivity_trace"]["max_absolute"] <= 1e-6
            for row in rows[1:]
        )
    )


def _gate_row_matches(telemetry: dict, expected: float) -> bool:
    return bool(
        telemetry["declared_policy_gain"] == expected
        and telemetry["actual_policy_gain_values"] == [expected]
        and telemetry["coactivity_trace_input_gain"] == expected
        and telemetry["policy_transmission_gain_values"] == [1.0]
    )


def _phase_schedule_passes(condition: dict) -> bool:
    if not _gate_row_matches(condition["warmup_gate"], 0.0):
        return False
    expected = {
        "pre_action": 0.0,
        "action": 1.0,
        "reward_delay": 0.0,
        "reset": 0.0,
        "washout": 0.0,
    }
    return all(
        _gate_row_matches(row["phase_gate_schedule"][phase], value)
        for row in condition["rows"]
        for phase, value in expected.items()
    )


def _group_metrics(rows: list[dict], *, snapshot: str, kind: str, action: int):
    values = []
    for row in rows:
        if row["winner"] != action:
            continue
        means = row["snapshots"][snapshot]["net_route_means"][kind]
        selected = float(means[str(action)])
        loser = float(means[str(1 - action)])
        values.append({
            "selected": selected,
            "loser": loser,
            "margin": selected - loser,
            "ratio": selected / max(loser, 1e-12),
        })
    selected = np.asarray([value["selected"] for value in values], dtype=float)
    loser = np.asarray([value["loser"] for value in values], dtype=float)
    margins = selected - loser
    ratios = np.asarray([value["ratio"] for value in values], dtype=float)
    checks = {
        "every_selected_is_nonzero": bool(values and np.all(selected > 0.0)),
        "median_ratio_at_least_4": bool(
            values and float(np.median(ratios)) >= 4.0
        ),
        "mean_loser_at_most_25pct_selected": bool(
            values and float(loser.mean()) <= 0.25 * float(selected.mean())
        ),
        "selected_exceeds_loser_on_80pct": bool(
            values and float(np.mean(selected > loser)) >= 0.80
        ),
        "mean_margin_is_positive": bool(values and float(margins.mean()) > 0.0),
    }
    return {
        "n_trials": len(values),
        "selected_mean": float(selected.mean()) if values else None,
        "loser_mean": float(loser.mean()) if values else None,
        "margin_mean": float(margins.mean()) if values else None,
        "ratio_median": float(np.median(ratios)) if values else None,
        "local_win_rate": float(np.mean(selected > loser)) if values else None,
        "checks": checks,
        "pass": bool(all(checks.values())),
        "values": values,
    }


def _mean_policy_eligibility(condition: dict) -> float:
    values = [
        float(snapshot["route_means"][kind][str(channel)])
        for row in condition["rows"]
        for snapshot in row["snapshots"].values()
        if snapshot is not None
        for kind in ("d1", "d2")
        for channel in CHANNELS
    ]
    return float(np.mean(values)) if values else 0.0


def evaluate(intact: dict, lesion: dict, clip: dict, *, backend: str):
    groups = {
        f"{snapshot}_{kind}_action{action}": _group_metrics(
            intact["rows"], snapshot=snapshot, kind=kind, action=action
        )
        for snapshot in ("decision", "pre_outcome")
        for kind in ("d1", "d2")
        for action in CHANNELS
    }
    clean = {
        mode: sum(row["winner"] is not None for row in condition["rows"])
        for mode, condition in (("intact", intact), ("coactivity_lesion", lesion))
    }
    winner_counts = {
        str(channel): sum(row["winner"] == channel for row in intact["rows"])
        for channel in CHANNELS
    }
    intact_selected = {
        kind: [
            row["snapshots"][snapshot]["route_means"][kind][str(row["winner"])]
            for row in intact["rows"] if row["winner"] is not None
            for snapshot in ("decision", "pre_outcome")
        ]
        for kind in ("d1", "d2")
    }
    lesion_max = {
        kind: max((
            row["snapshots"][snapshot]["route_means"][kind][str(channel)]
            for row in lesion["rows"]
            for snapshot in ("decision", "action_end", "pre_outcome")
            for channel in CHANNELS
            if row["snapshots"][snapshot] is not None
        ), default=0.0)
        for kind in ("d1", "d2")
    }
    lesion_collapse = all(
        lesion_max[kind] <= 0.01 * float(np.mean(intact_selected[kind]))
        for kind in ("d1", "d2") if intact_selected[kind]
    ) and all(intact_selected.values())
    eligibility_attribution = attributable_to(
        "policy eligibility from coactivity (intact vs coactivity lesion)",
        _mean_policy_eligibility(intact),
        _mean_policy_eligibility(lesion),
        warn_below=0.99,
    )
    validity = {
        "cupy_backend": backend == "cupy",
        "reserved_seed_only_and_formal_sealed": (
            intact["seed"] == SMOKE_SEED
            and lesion["seed"] == SMOKE_SEED and not OPEN_PHASES
        ),
        "structures_match_preregistration": (
            intact["structure"]["all_checks_pass"]
            and lesion["structure"]["all_checks_pass"]
        ),
        "matched_initial_weights": (
            intact["initial_weight_hash"] == lesion["initial_weight_hash"]
        ),
        "whole_run_firing_hashes_match": (
            intact["firing_sequence_hash"] == lesion["firing_sequence_hash"]
        ),
        "trial_telemetry_matches": _rows_match(intact, lesion),
        "at_least_11_clean_trials": min(clean.values()) >= 11,
        "both_actions_have_three_clean_trials": min(winner_counts.values()) >= 3,
        "washout_is_valid": (
            _washout_passes(intact["rows"])
            and _washout_passes(lesion["rows"])
        ),
        "preaction_coactivity_trace_is_clean": (
            _pretrace_passes(intact["rows"])
            and _pretrace_passes(lesion["rows"])
        ),
        "fixed_symmetric_phase_schedule_observed": (
            _phase_schedule_passes(intact)
            and _phase_schedule_passes(lesion)
        ),
        "all_engagement_weights_are_immutable": (
            intact["weights_byte_identical"]
            and lesion["weights_byte_identical"]
        ),
        "coactivity_lesion_collapses_eligibility_99pct": bool(lesion_collapse),
        "clip_path_control_is_immutable": clip["weights_byte_identical"],
    }
    if not validity["cupy_backend"]:
        verdict = "UNDEFINED_BACKEND"
    elif not validity["reserved_seed_only_and_formal_sealed"]:
        verdict = "UNDEFINED_PHASE_OR_SEED"
    elif not validity["structures_match_preregistration"]:
        verdict = "UNDEFINED_STRUCTURE"
    elif not (
        validity["all_engagement_weights_are_immutable"]
        and validity["clip_path_control_is_immutable"]
    ):
        verdict = "UNDEFINED_WEIGHT_MOVEMENT"
    elif not (
        validity["matched_initial_weights"]
        and validity["whole_run_firing_hashes_match"]
        and validity["trial_telemetry_matches"]
    ):
        verdict = "UNDEFINED_MATCHED_DYNAMICS"
    elif not validity["washout_is_valid"]:
        verdict = "UNDEFINED_WASHOUT"
    elif not validity["preaction_coactivity_trace_is_clean"]:
        verdict = "UNDEFINED_PRETRACE"
    elif not validity["fixed_symmetric_phase_schedule_observed"]:
        verdict = "UNDEFINED_PHASE_GATES"
    elif not validity["both_actions_have_three_clean_trials"]:
        verdict = "UNDEFINED_ACTION_COVERAGE"
    elif not validity["coactivity_lesion_collapses_eligibility_99pct"]:
        verdict = "UNDEFINED_LESION"
    elif not validity["at_least_11_clean_trials"]:
        verdict = "UNDEFINED_CLEAN_COMMIT"
    elif all(group["pass"] for group in groups.values()):
        verdict = "ENGAGEMENT_GO"
    else:
        verdict = "ENGAGEMENT_FAIL"
    return {
        "verdict": verdict,
        "go": verdict == "ENGAGEMENT_GO",
        "validity": validity,
        "clean_trials": clean,
        "winner_counts": winner_counts,
        "lesion_max_eligibility": lesion_max,
        "eligibility_attributable_to_coactivity": eligibility_attribution,
        "groups": groups,
    }


def _performance(bridge_result_seconds: float) -> dict[str, object]:
    xp, backend = get_backend()
    result = {
        "elapsed_seconds": float(bridge_result_seconds),
        "wall_ms_per_trial_condition": float(
            1000.0 * bridge_result_seconds / (2 * TRIALS)
        ),
        "backend": backend,
    }
    if backend == "cupy":
        props = xp.cuda.runtime.getDeviceProperties(0)
        device = props["name"]
        if isinstance(device, bytes):
            device = device.decode("utf-8", errors="replace")
        pool = xp.get_default_memory_pool()
        result.update({
            "device": str(device),
            "memory_pool_used_bytes": int(pool.used_bytes()),
            "memory_pool_reserved_bytes": int(pool.total_bytes()),
        })
    return result


def run_engagement_smoke(seed: int = SMOKE_SEED) -> dict[str, object]:
    validate_smoke_seed(seed)
    started = time.perf_counter()
    intact = run_condition(mode="intact", seed=seed)
    lesion = run_condition(mode="coactivity_lesion", seed=seed)
    clip = run_clip_path_control(seed)
    elapsed = time.perf_counter() - started
    _, backend = get_backend()
    evaluation = evaluate(intact, lesion, clip, backend=backend)
    return {
        "probe": "vocal_action_credit_gate_b_v10_policy_eligibility",
        "seed": int(seed),
        "formal_phases_open": list(OPEN_PHASES),
        "host_boundary": HOST_BOUNDARY,
        "fixed_protocol": {
            "trials": TRIALS,
            "warmup_steps": WARMUP_STEPS,
            "action_steps": ACTION_STEPS,
            "reward_delay_steps": REWARD_DELAY_STEPS,
            "reset_steps": RESET_STEPS,
            "washout_steps": WASHOUT_STEPS,
            "eligibility_tau_ms": ELIGIBILITY_TAU_MS,
            "coactivity_trace_tau_ms": COACTIVITY_TAU_MS,
            "coactivity_threshold": COACTIVITY_THRESHOLD,
            "coactivity_scale": COACTIVITY_SCALE,
            "weight_bounds": [WEIGHT_MIN, WEIGHT_MAX],
        },
        "conditions": {"intact": intact, "coactivity_lesion": lesion},
        "clip_path_control": clip,
        "evaluation": evaluation,
        "performance": _performance(elapsed),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }


def run_construction_smoke(seed: int = SMOKE_SEED) -> dict[str, object]:
    bridge, handles = build_v10_bridge(seed, coactivity=True)
    return {
        "probe": "vocal_action_credit_gate_b_v10_policy_construction",
        "seed": int(seed),
        "formal_phases_open": list(OPEN_PHASES),
        "host_boundary": HOST_BOUNDARY,
        "structure": structural_audit(bridge, handles),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SMOKE_SEED)
    parser.add_argument("--construction-only", action="store_true")
    parser.add_argument(
        "--output",
        default=(
            "research/findings/raw/vocal_action_credit_gate_v10/"
            "engagement_seed0_cupy.json"
        ),
    )
    args = parser.parse_args(argv)
    result = (
        run_construction_smoke(args.seed)
        if args.construction_only else run_engagement_smoke(args.seed)
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {
        "output": str(output),
        "construction_only": bool(args.construction_only),
        "all_checks_pass": (
            result["structure"]["all_checks_pass"]
            if args.construction_only else result["evaluation"]["go"]
        ),
        "verdict": (
            "CONSTRUCTION_GO" if args.construction_only
            and result["structure"]["all_checks_pass"]
            else "CONSTRUCTION_FAIL" if args.construction_only
            else result["evaluation"]["verdict"]
        ),
    }
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
