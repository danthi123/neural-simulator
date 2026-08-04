"""Sealed Gate B v8 dual-afferent expectation engagement smoke.

Reserved seed zero may test whether a fixed subthreshold state afferent and a
separate plastic action trace jointly engage the existing MSN-D1 expectation
population. No formal seed is assigned or executable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
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
    _region,
    build_selector_bridge,
    selector_config,
)
from sim.backend import get_backend, to_host
from sim.enums import NeuronType
from sim.regions import RegionPathway
from tools.lab import attributable_to


OPEN_PHASES: tuple[str, ...] = ()
SMOKE_SEED = 0
TRACE_SIZE = 200
FIXED_WEIGHT_LADDER = (2.0, 4.0, 8.0, 12.0, 16.0)
FIXED_OUTPUT_GATE = "vocal_reward_fixed_upstate_output"
FIXED_TRACE_PREFIX = "vocal_credit_fixed_state_trace_"

HOST_BOUNDARY = {
    **v7.HOST_BOUNDARY,
    "host_fixed_state_trace_injection": False,
    "host_selects_fixed_state_channel": False,
}


@dataclass(frozen=True)
class VocalCreditConfigV8(v7.VocalCreditConfigV7):
    n_value: int = TRACE_SIZE
    n_fixed_trace: int = TRACE_SIZE
    fixed_trace_to_expectation_density: float = 0.80
    fixed_trace_to_expectation_weight: float = FIXED_WEIGHT_LADDER[0]


def v8_config(fixed_weight: float = FIXED_WEIGHT_LADDER[0]) -> VocalCreditConfigV8:
    checked = float(fixed_weight)
    if checked not in FIXED_WEIGHT_LADDER:
        raise ValueError(
            f"fixed weight must be in preregistered ladder {FIXED_WEIGHT_LADDER}; "
            f"received={checked}"
        )
    return VocalCreditConfigV8(fixed_trace_to_expectation_weight=checked)


def _fixed_trace(channel: int) -> str:
    return f"{FIXED_TRACE_PREFIX}{int(channel)}"


def validate_smoke_seed(seed: int) -> int:
    checked = int(seed)
    if checked != SMOKE_SEED:
        raise ValueError(f"v8 smoke accepts reserved seed {SMOKE_SEED} only")
    return checked


def validate_phase(phase: str) -> str:
    raise ValueError(
        "Gate B v8 formal phases are sealed and unassigned; "
        f"open phases={OPEN_PHASES}, requested={phase!r}"
    )


def run_formal_seed(seed: int):
    raise ValueError(f"Gate B v8 formal execution is sealed; received={int(seed)}")


def _v8_regions(config: VocalCreditConfigV8):
    regions = list(v5l._learning_regions(config))
    rs = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL
    regions.extend(
        _region(
            _fixed_trace(channel),
            config.n_fixed_trace,
            exc_fraction=1.0,
            neuron_type=rs,
        )
        for channel in CHANNELS
    )
    return regions


def _v8_pathways(
    config: VocalCreditConfigV8,
    *,
    action_permutation: tuple[int, int] = (0, 1),
) -> list[RegionPathway]:
    if tuple(action_permutation) not in ((0, 1), (1, 0)):
        raise ValueError("action permutation must be identity or the fixed swap")
    pathways = list(v7._v7_pathways(
        config, action_permutation=action_permutation
    ))
    for source_channel in CHANNELS:
        target_channel = int(action_permutation[source_channel])
        pathways.append(RegionPathway(
            from_region=f"commit_{source_channel}",
            to_region=_fixed_trace(target_channel),
            density=1.0,
            weight_mean=config.commit_to_value_weight,
            weight_jitter=0.0,
            plastic=False,
            coincidence_detector=True,
        ))
    for channel in CHANNELS:
        pathways.extend([
            RegionPathway(
                from_region="practice_arousal",
                to_region=_fixed_trace(channel),
                density=1.0,
                weight_mean=config.arousal_to_value_weight,
                weight_jitter=0.0,
                plastic=False,
                coincidence_detector=True,
            ),
            RegionPathway(
                from_region=_fixed_trace(channel),
                to_region=v5l._expectation(channel),
                density=config.fixed_trace_to_expectation_density,
                weight_mean=config.fixed_trace_to_expectation_weight,
                weight_jitter=0.0,
                plastic=False,
                transmission_gate=FIXED_OUTPUT_GATE,
            ),
        ])
    return pathways


def build_v8_bridge(
    seed: int = SMOKE_SEED,
    config: VocalCreditConfigV8 | None = None,
    *,
    action_permutation: tuple[int, int] = (0, 1),
):
    validate_smoke_seed(seed)
    config = config or v8_config()
    bridge = build_selector_bridge(
        seed,
        selector_config("v2"),
        extra_regions=_v8_regions(config),
        extra_pathways=_v8_pathways(
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
    fixed_routes = {
        channel: _route_synapses(
            bridge, _fixed_trace(channel), v5l._expectation(channel)
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
    bridge.set_transmission_gate(FIXED_OUTPUT_GATE, 1.0)
    bridge.cp_reward_eligibility_synapse_indices = xp.asarray(
        routes.all_indices(), dtype=xp.int64
    )
    return bridge, {
        "routes": routes,
        "actor_routes": actor_routes,
        "expectation_routes": expectation_routes,
        "fixed_routes": fixed_routes,
        "action_permutation": tuple(action_permutation),
    }


def _expected_coincidence_indices(bridge, handles: dict) -> np.ndarray:
    expected = []
    permutation = handles["action_permutation"]
    for source_channel in CHANNELS:
        target_channel = int(permutation[source_channel])
        expected.extend([
            _route_synapses(
                bridge, f"commit_{source_channel}", v5l._trace(target_channel)
            ),
            _route_synapses(
                bridge, f"commit_{source_channel}", _fixed_trace(target_channel)
            ),
        ])
    for channel in CHANNELS:
        expected.extend([
            _route_synapses(bridge, "practice_arousal", v5l._trace(channel)),
            _route_synapses(bridge, "practice_arousal", _fixed_trace(channel)),
        ])
    return np.sort(np.concatenate(expected))


def structural_audit(bridge, handles: dict) -> dict[str, object]:
    plastic = np.asarray(to_host(bridge.cp_synapse_plastic_mask), dtype=bool)
    expected_plastic = np.zeros(plastic.shape, dtype=bool)
    expected_plastic[handles["routes"].all_indices()] = True
    eligible = np.sort(np.asarray(
        to_host(bridge.cp_reward_eligibility_synapse_indices), dtype=np.int64
    ))
    expected_eligible = np.sort(handles["routes"].all_indices())
    actor_gate = np.sort(np.asarray(
        to_host(bridge._plasticity_gate_indices_gpu[CREDIT_PLASTICITY_GATE]),
        dtype=np.int64,
    ))
    expectation_gate = np.sort(np.asarray(
        to_host(
            bridge._plasticity_gate_indices_gpu[v5l.EXPECTATION_PLASTICITY_GATE]
        ),
        dtype=np.int64,
    ))
    fixed_gate = np.sort(np.asarray(
        to_host(bridge._transmission_gate_indices_gpu[FIXED_OUTPUT_GATE]),
        dtype=np.int64,
    ))
    expected_actor = np.sort(np.concatenate(list(handles["actor_routes"].values())))
    expected_expectation = np.sort(np.concatenate(
        list(handles["expectation_routes"].values())
    ))
    expected_fixed = np.sort(np.concatenate(list(handles["fixed_routes"].values())))
    coincidence = np.asarray(
        to_host(bridge.cp_coincidence_synapse_mask), dtype=bool
    )
    expected_coincidence = np.zeros(coincidence.shape, dtype=bool)
    expected_coincidence[_expected_coincidence_indices(bridge, handles)] = True
    return {
        "plastic_synapses": int(plastic.sum()),
        "declared_plastic_synapses": int(expected_plastic.sum()),
        "plastic_outside_declared_routes": int(
            np.logical_and(plastic, ~expected_plastic).sum()
        ),
        "fixed_inside_declared_routes": int(
            np.logical_and(~plastic, expected_plastic).sum()
        ),
        "fixed_upstate_routes_are_nonplastic": bool(
            not plastic[expected_fixed].any()
        ),
        "reward_eligibility_matches_declared_routes": bool(
            np.array_equal(eligible, expected_eligible)
        ),
        "actor_gate_matches_actor_routes": bool(
            np.array_equal(actor_gate, expected_actor)
        ),
        "expectation_gate_matches_expectation_routes": bool(
            np.array_equal(expectation_gate, expected_expectation)
        ),
        "fixed_output_gate_matches_fixed_routes": bool(
            np.array_equal(fixed_gate, expected_fixed)
        ),
        "coincidence": {
            "enabled_synapses": int(coincidence.sum()),
            "intended_synapses": int(expected_coincidence.sum()),
            "enabled_outside_intended_routes": int(
                np.logical_and(coincidence, ~expected_coincidence).sum()
            ),
            "disabled_inside_intended_routes": int(
                np.logical_and(~coincidence, expected_coincidence).sum()
            ),
        },
    }


def _region_spikes(firing: np.ndarray, bridge, region: str) -> int:
    return v3._spike_count(firing, bridge, region)


def _observe_window(bridge, steps: int) -> dict[str, object]:
    totals = {
        "commit": [0, 0],
        "motor": [0, 0],
        "actor": [0, 0],
        "trace": [0, 0],
        "fixed_trace": [0, 0],
        "expectation": [0, 0],
        "snc": 0,
        "lhb": 0,
        "rmtg": 0,
    }
    dopamine = []
    for _ in range(int(steps)):
        v5._step(bridge)
        firing = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
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
            totals["fixed_trace"][channel] += _region_spikes(
                firing, bridge, _fixed_trace(channel)
            )
            totals["expectation"][channel] += _region_spikes(
                firing, bridge, v5l._expectation(channel)
            )
        totals["snc"] += _region_spikes(firing, bridge, v5l.SNC)
        totals["lhb"] += _region_spikes(firing, bridge, v3.LATERAL_HABENULA)
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
    config: VocalCreditConfigV8,
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
    eligibility = np.asarray(
        to_host(bridge.cp_eligibility_trace), dtype=np.float64
    )
    eligibility_means = {
        "actor": [
            float(np.mean(np.abs(eligibility[handles["actor_routes"][channel]])))
            for channel in CHANNELS
        ],
        "expectation": [
            float(np.mean(np.abs(
                eligibility[handles["expectation_routes"][channel]]
            )))
            for channel in CHANNELS
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
    config: VocalCreditConfigV8,
    *,
    seed: int = SMOKE_SEED,
) -> dict[str, object]:
    if mode not in {
        "intact", "expectation_learning_lesion", "fixed_arm_lesion"
    }:
        raise ValueError(mode)
    validate_smoke_seed(seed)
    bridge, handles = build_v8_bridge(seed=seed, config=config)
    selector = selector_config("v2")
    bridge.set_plasticity_gate(CREDIT_PLASTICITY_GATE, 0.0)
    bridge.set_plasticity_gate(v5l.EXPECTATION_PLASTICITY_GATE, 0.0)
    if mode == "fixed_arm_lesion":
        bridge.set_transmission_gate(FIXED_OUTPUT_GATE, 0.0)
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
        "fixed_weight": config.fixed_trace_to_expectation_weight,
        "baseline_probe": baseline_probe,
        "expectation_weight_before": expectation_before,
        "expectation_weight_after": v5l._weight_means(
            bridge, handles["expectation_routes"]
        ),
        "fixed_route_weight": v5l._weight_means(
            bridge, handles["fixed_routes"]
        ),
        "fixed_output_gate": float(
            bridge._transmission_gate_values[FIXED_OUTPUT_GATE]
        ),
        "clean_trials": int(sum(row["winner"] is not None for row in rows)),
        "rewarded_trials": int(sum(row["reward_delivered"] for row in rows)),
        "changed_synapses": int(changed.sum()),
        "changed_outside_declared_routes": int(
            np.logical_and(changed, ~declared).sum()
        ),
        "rows": rows,
    }


def _late_delay_expectation(condition: dict[str, object]) -> list[int]:
    rewarded = [
        row for row in condition["rows"] if row["reward_delivered"]
    ]
    late = rewarded[-4:]
    return [
        int(sum(row["delay"]["expectation"][channel] for row in late))
        for channel in CHANNELS
    ]


def _delay_trace_total(condition: dict[str, object], key: str) -> int:
    return int(sum(
        sum(row["delay"][key]) for row in condition["rows"]
    ))


def run_engagement_smoke(
    fixed_weight: float,
    *,
    seed: int = SMOKE_SEED,
) -> dict[str, object]:
    config = v8_config(fixed_weight)
    conditions = {
        mode: run_engagement_condition(mode, config, seed=seed)
        for mode in (
            "intact", "expectation_learning_lesion", "fixed_arm_lesion"
        )
    }
    intact = conditions["intact"]
    learning_lesion = conditions["expectation_learning_lesion"]
    fixed_lesion = conditions["fixed_arm_lesion"]
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
    checks = {
        "reserved_seed_only": int(seed) == SMOKE_SEED,
        "formal_execution_remains_sealed": OPEN_PHASES == (),
        "at_least_90pct_clean_action_epochs": all(
            condition["clean_trials"] >= 0.9 * config.smoke_training_trials
            for condition in conditions.values()
        ),
        "both_trace_populations_engage_before_outcome": all(
            _delay_trace_total(condition, "trace") > 0
            and _delay_trace_total(condition, "fixed_trace") > 0
            for condition in conditions.values()
        ),
        "fixed_arm_is_subthreshold_before_learning": all(
            sum(conditions[mode]["baseline_probe"]["delay"]["expectation"]) == 0
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
        "fixed_arm_lesion_removes_80pct_of_delay_expectation": bool(
            intact_delay[0] > 0
            and delays["fixed_arm_lesion"][0] <= 0.20 * intact_delay[0]
        ),
        "plasticity_confined_to_declared_routes": all(
            condition["changed_outside_declared_routes"] == 0
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
            "name": "fixed_arm_lesion_gate_is_zero",
            "ok": fixed_lesion["fixed_output_gate"] == 0.0,
            "observed": fixed_lesion["fixed_output_gate"],
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
        "probe": "vocal_action_credit_gate_b_v8_dual_afferent_engagement",
        "fixed_weight": float(fixed_weight),
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
        "delay_expectation_attributable_to_learning": attributable_to(
            "pre-outcome expectation from local trace learning",
            intact_delay[0],
            delays["expectation_learning_lesion"][0],
            warn_below=0.80,
        ),
        "delay_expectation_attributable_to_fixed_state": attributable_to(
            "pre-outcome expectation from fixed convergent state",
            intact_delay[0],
            delays["fixed_arm_lesion"][0],
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
    fixed_weight: float = FIXED_WEIGHT_LADDER[0],
    *,
    seed: int = SMOKE_SEED,
) -> dict[str, object]:
    config = v8_config(fixed_weight)
    bridge, handles = build_v8_bridge(seed=seed, config=config)
    audit = structural_audit(bridge, handles)
    pathways = _v8_pathways(config)
    learned = [
        pathway for pathway in pathways
        if pathway.from_region in {v5l._trace(c) for c in CHANNELS}
        and pathway.to_region in {v5l._expectation(c) for c in CHANNELS}
    ]
    fixed = [
        pathway for pathway in pathways
        if pathway.from_region in {_fixed_trace(c) for c in CHANNELS}
        and pathway.to_region in {v5l._expectation(c) for c in CHANNELS}
    ]
    snc_outputs = [
        pathway for pathway in pathways
        if pathway.from_region in {v5l._expectation(c) for c in CHANNELS}
        and pathway.to_region == v5l.SNC
    ]
    checks = {
        "reserved_seed_only": int(seed) == SMOKE_SEED,
        "formal_execution_unassigned_and_sealed": OPEN_PHASES == (),
        "fixed_weight_is_preregistered": float(fixed_weight) in FIXED_WEIGHT_LADDER,
        "trace_populations_are_distinct_and_locked_to_200": bool(
            config.n_value == TRACE_SIZE
            and config.n_fixed_trace == TRACE_SIZE
            and all(_fixed_trace(c) != v5l._trace(c) for c in CHANNELS)
        ),
        "plastic_routes_match_preregistration": bool(
            len(learned) == len(CHANNELS)
            and all(pathway.plastic for pathway in learned)
            and all(pathway.density == 0.50 for pathway in learned)
        ),
        "fixed_routes_match_preregistration": bool(
            len(fixed) == len(CHANNELS)
            and all(not pathway.plastic for pathway in fixed)
            and all(pathway.density == 0.80 for pathway in fixed)
            and all(
                pathway.weight_mean == float(fixed_weight)
                for pathway in fixed
            )
            and all(pathway.transmission_gate == FIXED_OUTPUT_GATE for pathway in fixed)
        ),
        "expectation_outputs_use_gabab": bool(
            len(snc_outputs) == len(CHANNELS)
            and all(pathway.receptor == "gaba_b" for pathway in snc_outputs)
        ),
        "plasticity_confined_to_declared_routes": bool(
            audit["plastic_synapses"] == audit["declared_plastic_synapses"]
            and audit["plastic_outside_declared_routes"] == 0
            and audit["fixed_inside_declared_routes"] == 0
            and audit["fixed_upstate_routes_are_nonplastic"]
        ),
        "runtime_ownership_matches_declared_routes": bool(
            audit["reward_eligibility_matches_declared_routes"]
            and audit["actor_gate_matches_actor_routes"]
            and audit["expectation_gate_matches_expectation_routes"]
            and audit["fixed_output_gate_matches_fixed_routes"]
        ),
        "coincidence_confined_to_both_fixed_trace_loaders": bool(
            audit["coincidence"]["enabled_synapses"]
            == audit["coincidence"]["intended_synapses"]
            and audit["coincidence"]["enabled_outside_intended_routes"] == 0
            and audit["coincidence"]["disabled_inside_intended_routes"] == 0
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
            "name": "fixed_weight_is_preregistered",
            "ok": float(fixed_weight) in FIXED_WEIGHT_LADDER,
            "observed": float(fixed_weight),
            "expected": list(FIXED_WEIGHT_LADDER),
        },
    ]
    prerequisites_hold = all(item["ok"] for item in preconditions)
    config_payload = json.dumps(
        asdict(config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    xp, _ = get_backend()
    return {
        "artifact_schema_version": 1,
        "probe": "vocal_action_credit_gate_b_v8_dual_afferent_construction",
        "fixed_weight": float(fixed_weight),
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
        "--fixed-weight", type=float, choices=FIXED_WEIGHT_LADDER,
        default=FIXED_WEIGHT_LADDER[0],
    )
    parser.add_argument("--engagement", action="store_true")
    parser.add_argument("--formal-phase")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.formal_phase is not None:
        validate_phase(args.formal_phase)
    validate_smoke_seed(args.seed)
    result = (
        run_engagement_smoke(args.fixed_weight, seed=args.seed)
        if args.engagement
        else run_construction_smoke(args.fixed_weight, seed=args.seed)
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0 if result["status"] in {"CONSTRUCTION_PASS", "ENGAGEMENT_PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
