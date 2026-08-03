"""Smoke-only Gate B v5 neural action-tag prototype.

The selected commit population and the shared practice/arousal population
converge on each action-value pool through the graded dendritic route.  The
plateau transfer is centered above either input's measured solo drive, so the
tag is loaded by their coincidence during a fixed action epoch.  The host does
not inspect a winner to open or close any route.

At outcome time one generic population excites both value pools and both local
FS pools symmetrically.  The latter provide feedforward inhibition.  This file
is smoke-only: seed zero is the only executable seed and every formal phase is
sealed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from research.runners import _vocal_action_credit_gate_v3 as v3
from research.runners._vocal_action_credit_gate import (
    CREDIT_CUE,
    CREDIT_PLASTICITY_GATE,
    OUTCOME_ONSET,
    VALUE_PLASTICITY_GATE,
    CreditRoutes,
    _actor,
    _core_updates,
    _route_synapses,
    _set_learning_gates,
    _value,
)
from research.runners._vocal_action_selector_gate import (
    CHANNELS,
    _indices,
    _step,
    build_selector_bridge,
    selector_config,
)
from sim.backend import get_backend, to_host
from sim.regions import RegionPathway
from tools.lab import attributable_to


OPEN_PHASES: tuple[str, ...] = ()
SMOKE_SEED = 0
HOST_BOUNDARY = {
    **v3.HOST_BOUNDARY,
    "host_expected_value_state": False,
    "host_action_winner_latch": False,
    "host_action_timed_transmission_window": False,
    "fixed_action_epoch": True,
    "shared_outcome_afferent_identical_across_channels": True,
}

OUTCOME_EXCITATION_GATE = "vocal_credit_v5_outcome_excitation"
OUTCOME_INHIBITION_GATE = "vocal_credit_v5_outcome_inhibition"


@dataclass(frozen=True)
class VocalCreditConfigV5(v3.VocalCreditConfigV3):
    commit_to_value_weight: float = 20.0
    arousal_to_value_weight: float = 20.0
    outcome_to_value_weight: float = 2.0
    outcome_to_value_fs_weight: float = 14.0
    action_tag_center: float = 400.0
    action_tag_slope: float = 0.20
    action_tag_strength: float = 3.0
    action_tag_tau_decay_ms: float = 500.0
    action_tag_tau_rise_ms: float = 2.0
    smoke_warmup_steps: int = 120
    smoke_action_steps: int = 100
    smoke_gap_steps: int = 100
    smoke_outcome_steps: int = 40


def v5_config() -> VocalCreditConfigV5:
    return VocalCreditConfigV5()


def validate_smoke_seed(seed: int) -> int:
    checked = int(seed)
    if checked != SMOKE_SEED:
        raise ValueError(f"v5 smoke accepts reserved seed {SMOKE_SEED} only")
    return checked


def validate_phase(phase: str) -> str:
    raise ValueError(
        f"v5 formal phases are sealed; open phases={OPEN_PHASES}, requested={phase!r}"
    )


def validate_formal_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    checked = tuple(int(seed) for seed in seeds)
    raise ValueError(f"v5 formal seeds are sealed; received={checked}")


def _is_replaced_v3_pathway(pathway) -> bool:
    return bool(
        pathway.from_region in {f"motor_{channel}" for channel in CHANNELS}
        and pathway.to_region in {
            _value(channel) for channel in CHANNELS
        } | {v3._value_fs(channel) for channel in CHANNELS}
    )


def _v5_pathways(
    config: VocalCreditConfigV5,
    *,
    action_permutation: tuple[int, int] = (0, 1),
) -> list[RegionPathway]:
    if tuple(action_permutation) not in ((0, 1), (1, 0)):
        raise ValueError("action permutation must be identity or the fixed swap")
    pathways = [
        pathway for pathway in v3._v3_pathways(config)
        if not _is_replaced_v3_pathway(pathway)
    ]
    for source_channel in CHANNELS:
        target_channel = int(action_permutation[source_channel])
        pathways.append(RegionPathway(
            from_region=f"commit_{source_channel}",
            to_region=_value(target_channel),
            density=1.0,
            weight_mean=config.commit_to_value_weight,
            weight_jitter=0.0,
            plastic=True,
            plasticity_gate=VALUE_PLASTICITY_GATE,
            coincidence_detector=True,
        ))
    for channel in CHANNELS:
        pathways.extend([
            RegionPathway(
                from_region="practice_arousal",
                to_region=_value(channel),
                density=1.0,
                weight_mean=config.arousal_to_value_weight,
                weight_jitter=0.0,
                plastic=False,
                coincidence_detector=True,
            ),
            RegionPathway(
                from_region=OUTCOME_ONSET,
                to_region=_value(channel),
                density=1.0,
                weight_mean=config.outcome_to_value_weight,
                weight_jitter=0.0,
                plastic=False,
                transmission_gate=OUTCOME_EXCITATION_GATE,
            ),
            RegionPathway(
                from_region=OUTCOME_ONSET,
                to_region=v3._value_fs(channel),
                density=1.0,
                weight_mean=config.outcome_to_value_fs_weight,
                weight_jitter=0.0,
                plastic=False,
                transmission_gate=OUTCOME_INHIBITION_GATE,
            ),
        ])
    return pathways


def _v5_core_updates(config: VocalCreditConfigV5) -> dict:
    updates = dict(_core_updates(config))
    updates.update({
        "enable_coincidence_detection": True,
        "coincidence_weighted_drive": True,
        "coincidence_plateau_strength": 0.0,
        "enable_graded_dendritic_plateau": True,
        "graded_plateau_center": config.action_tag_center,
        "graded_plateau_slope": config.action_tag_slope,
        "graded_plateau_strength": config.action_tag_strength,
        "graded_plateau_tau_decay_ms": config.action_tag_tau_decay_ms,
        "graded_plateau_tau_rise_ms": config.action_tag_tau_rise_ms,
    })
    return updates


def build_v5_bridge(
    seed: int = SMOKE_SEED,
    config: VocalCreditConfigV5 | None = None,
    *,
    action_permutation: tuple[int, int] = (0, 1),
):
    validate_smoke_seed(seed)
    config = config or v5_config()
    bridge = build_selector_bridge(
        seed,
        selector_config("v2"),
        extra_regions=v3._v3_regions(config),
        extra_pathways=_v5_pathways(
            config, action_permutation=action_permutation
        ),
        core_config_updates=_v5_core_updates(config),
    )
    actor_routes = {
        channel: _route_synapses(bridge, CREDIT_CUE, _actor(channel))
        for channel in CHANNELS
    }
    tag_routes = {
        source: _route_synapses(
            bridge,
            f"commit_{source}",
            _value(action_permutation[source]),
        )
        for source in CHANNELS
    }
    routes = CreditRoutes(actor=actor_routes, value=tag_routes)
    xp, _ = get_backend()
    bridge.set_global_plasticity_gain(0.0)
    _set_learning_gates(bridge, config, actor_enabled=True, value_enabled=True)
    bridge.cp_reward_eligibility_synapse_indices = xp.asarray(
        routes.all_indices(), dtype=xp.int64
    )
    return bridge, {
        "routes": routes,
        "tag_routes": tag_routes,
        "action_permutation": tuple(action_permutation),
    }


def _region_spikes(firing, bridge, region: str) -> int:
    return int(firing[_indices(bridge, region)].sum())


def _plateau_means(bridge) -> list[float]:
    state = np.asarray(
        to_host(bridge.cp_conductance_g_graded_plateau), dtype=np.float64
    )
    return [
        float(state[_indices(bridge, _value(channel))].mean())
        for channel in CHANNELS
    ]


def _observe_counts(bridge, steps: int) -> dict[str, object]:
    totals = {
        "arousal": 0,
        "commit": [0, 0],
        "motor": [0, 0],
        "value": [0, 0],
        "value_fs": [0, 0],
    }
    for _ in range(int(steps)):
        _step(bridge)
        firing = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        totals["arousal"] += _region_spikes(firing, bridge, "practice_arousal")
        for channel in CHANNELS:
            totals["commit"][channel] += _region_spikes(
                firing, bridge, f"commit_{channel}"
            )
            totals["motor"][channel] += _region_spikes(
                firing, bridge, f"motor_{channel}"
            )
            totals["value"][channel] += _region_spikes(
                firing, bridge, _value(channel)
            )
            totals["value_fs"][channel] += _region_spikes(
                firing, bridge, v3._value_fs(channel)
            )
    return totals


def _winner_from_neural_counts(motor_counts: list[int]) -> int | None:
    crossed = [
        channel for channel in CHANNELS
        if motor_counts[channel] >= selector_config("v2").commit_threshold_spikes
    ]
    if len(crossed) != 1:
        return None
    return int(crossed[0])


def _rates(counts: list[int], config: VocalCreditConfigV5) -> list[float]:
    seconds = config.smoke_outcome_steps / 1000.0
    return [float(count / config.n_value / seconds) for count in counts]


def run_smoke_condition(
    control: str = "intact",
    config: VocalCreditConfigV5 | None = None,
) -> dict:
    controls = {
        "intact", "arousal_lesion", "plateau_lesion",
        "outcome_excitation_lesion", "outcome_inhibition_lesion",
        "action_channel_permutation",
    }
    if control not in controls:
        raise ValueError(control)
    config = config or v5_config()
    permutation = (1, 0) if control == "action_channel_permutation" else (0, 1)
    bridge, handles = build_v5_bridge(
        config=config, action_permutation=permutation
    )
    if control == "plateau_lesion":
        bridge.cp_coincidence_synapse_mask[:] = False
    if control == "outcome_excitation_lesion":
        bridge.set_transmission_gate(OUTCOME_EXCITATION_GATE, 0.0)
    if control == "outcome_inhibition_lesion":
        bridge.set_transmission_gate(OUTCOME_INHIBITION_GATE, 0.0)

    selector = selector_config("v2")
    v3._set_trial_current_v3(bridge, selector, config)
    warmup = _observe_counts(bridge, config.smoke_warmup_steps)
    plateau_before_action = _plateau_means(bridge)

    v3._set_trial_current_v3(
        bridge,
        selector,
        config,
        arousal=control != "arousal_lesion",
        cue=True,
    )
    action = _observe_counts(bridge, config.smoke_action_steps)
    plateau_after_action = _plateau_means(bridge)
    winner = _winner_from_neural_counts(action["motor"])

    v3._set_trial_current_v3(bridge, selector, config)
    delay = _observe_counts(bridge, config.smoke_gap_steps)
    plateau_before_outcome = _plateau_means(bridge)

    v3._set_trial_current_v3(bridge, selector, config, outcome=True)
    outcome = _observe_counts(bridge, config.smoke_outcome_steps)
    expected = None if winner is None else int(permutation[winner])
    result = {
        "control": control,
        "seed": SMOKE_SEED,
        "scientific_seed_executed": False,
        "config": asdict(config),
        "winner_observed_after_fixed_action_epoch": winner,
        "expected_value_channel": expected,
        "warmup": warmup,
        "action": action,
        "delay": delay,
        "outcome": outcome,
        "plateau_before_action": plateau_before_action,
        "plateau_after_action": plateau_after_action,
        "plateau_before_outcome": plateau_before_outcome,
        "outcome_value_rate_hz_per_cell": _rates(outcome["value"], config),
        "host_boundary": dict(HOST_BOUNDARY),
        "route_gains": {
            "outcome_excitation": float(
                bridge._transmission_gate_values[OUTCOME_EXCITATION_GATE]
            ),
            "outcome_inhibition": float(
                bridge._transmission_gate_values[OUTCOME_INHIBITION_GATE]
            ),
        },
    }
    del bridge
    return result


def run_smoke(config: VocalCreditConfigV5 | None = None) -> dict:
    rows = {
        control: run_smoke_condition(control, config)
        for control in (
            "intact", "arousal_lesion", "plateau_lesion",
            "outcome_excitation_lesion", "outcome_inhibition_lesion",
            "action_channel_permutation",
        )
    }
    intact = rows["intact"]
    expected = intact["expected_value_channel"]
    permuted = rows["action_channel_permutation"]
    permuted_expected = permuted["expected_value_channel"]
    if expected is None:
        attribution = {}
    else:
        attribution = {
            "action_tag_to_arousal_coincidence": attributable_to(
                "v5 action tag from commit/arousal coincidence",
                intact["plateau_after_action"][expected],
                rows["arousal_lesion"]["plateau_after_action"][expected],
            ),
            "outcome_read_to_plateau": attributable_to(
                "v5 outcome read from persistent action tag",
                intact["outcome"]["value"][expected],
                rows["plateau_lesion"]["outcome"]["value"][expected],
            ),
            "outcome_read_to_excitation": attributable_to(
                "v5 outcome read from generic excitation",
                intact["outcome"]["value"][expected],
                rows["outcome_excitation_lesion"]["outcome"]["value"][expected],
            ),
            "outcome_suppression_to_feedforward_inhibition": attributable_to(
                "v5 excess outcome firing removed by feedforward inhibition",
                rows["outcome_inhibition_lesion"]["outcome"]["value"][expected],
                intact["outcome"]["value"][expected],
            ),
        }
    checks = {
        "reserved_smoke_seed_only": all(
            row["seed"] == SMOKE_SEED and not row["scientific_seed_executed"]
            for row in rows.values()
        ),
        "no_host_winner_or_timing_latch": bool(
            not HOST_BOUNDARY["host_action_winner_latch"]
            and not HOST_BOUNDARY["host_action_timed_transmission_window"]
        ),
        "selector_commits": expected is not None,
        "coincidence_adds_selected_plateau": bool(
            expected is not None
            and intact["plateau_after_action"][expected]
            > intact["plateau_before_action"][expected]
        ),
        "arousal_is_load_bearing_for_tag": bool(
            expected is not None
            and intact["plateau_after_action"][expected]
            > rows["arousal_lesion"]["plateau_after_action"][expected]
        ),
        "tag_is_channel_selective": bool(
            expected is not None
            and intact["plateau_before_outcome"][expected]
            > intact["plateau_before_outcome"][1 - expected]
        ),
        "permutation_moves_neural_tag": bool(
            permuted_expected is not None
            and permuted_expected == 1 - permuted["winner_observed_after_fixed_action_epoch"]
            and permuted["plateau_before_outcome"][permuted_expected]
            > permuted["plateau_before_outcome"][1 - permuted_expected]
        ),
        "plateau_lesion_removes_tag": max(
            rows["plateau_lesion"]["plateau_before_outcome"]
        ) == 0.0,
        "plateau_is_load_bearing_for_outcome_read": bool(
            expected is not None
            and intact["outcome"]["value"][expected]
            > rows["plateau_lesion"]["outcome"]["value"][expected]
        ),
        "generic_outcome_drives_both_fs_pools": all(
            count > 0 for count in intact["outcome"]["value_fs"]
        ),
        "outcome_excitation_is_load_bearing": bool(
            expected is not None
            and intact["outcome"]["value"][expected]
            > rows["outcome_excitation_lesion"]["outcome"]["value"][expected]
        ),
        "outcome_inhibition_is_load_bearing": bool(
            expected is not None
            and rows["outcome_inhibition_lesion"]["outcome"]["value"][expected]
            > intact["outcome"]["value"][expected]
        ),
        "selected_outcome_rate_is_bounded_1_to_20_hz_per_cell": bool(
            expected is not None
            and 1.0 <= intact["outcome_value_rate_hz_per_cell"][expected] <= 20.0
        ),
    }
    return {
        "probe": "vocal_action_credit_gate_b_v5",
        "version": "v5-smoke-only",
        "science_seed_executed": False,
        "open_phases": list(OPEN_PHASES),
        "host_boundary": dict(HOST_BOUNDARY),
        "attribution": attribution,
        "checks": checks,
        "status": "SMOKE_PASS" if all(checks.values()) else "SMOKE_FAIL",
        "conditions": rows,
    }


def run_formal_seed(seed: int, config: VocalCreditConfigV5 | None = None):
    validate_formal_seeds((seed,))
