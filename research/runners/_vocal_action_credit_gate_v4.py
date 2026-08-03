"""Smoke-only Gate B v4 successor for delayed vocal action credit.

V4 preserves the v3 selector, actor/critic plasticity, local FS normalization,
value-to-SNc GABA-B subtraction, reward veto, and LHb -> RMTg -> SNc omission
chain.  Its one mechanistic change is to route the existing plastic
motor-to-action-value afferent through the existing graded dendritic plateau
substrate.  A fixed, symmetric outcome afferent then supplies a shared somatic
read pulse to both value populations.  The residual action-local dendritic
state can bias firing at outcome time without a host-side winner latch.

This file is deliberately smoke-only.  Seed zero may exercise anatomy and
short physiology probes.  Fresh formal partitions are named but every
scientific phase and direct formal entry point is sealed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from research.runners import _vocal_action_credit_gate_v3 as v3
from research.runners._vocal_action_credit_gate import (
    ACTION_COLLATERAL_GATE,
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


OPEN_PHASES: tuple[str, ...] = ()
RETIRED = True
SMOKE_SEED = 0
CALIBRATION_SEEDS = (70001, 70003)
DEVELOPMENT_SEEDS = (70009, 70019, 70039, 70051)
HELD_OUT_SEEDS = (70061, 70067)

SHARED_OUTCOME_READ_GATE = "vocal_credit_shared_outcome_read"
EXPECTATION_ROUTE_CONTROL = "vocal_credit_expectation_route"

FORMAL_CONTROLS = (
    "contingent",
    "reward_count_matched_shifted_yoked",
    "collateral_lesion",
    "reward_to_snc_lesion",
    "critic_output_lesion",
    "omission_path_lesion",
    "normalization_lesion",
    "frozen_expectation_route",
    "plateau_lesion",
    "shared_outcome_read_lesion",
    "action_channel_permutation",
)
SMOKE_CONTROLS = (
    "intact",
    "frozen_expectation_route",
    "plateau_lesion",
    "shared_outcome_read_lesion",
    "action_channel_permutation",
)

HOST_BOUNDARY = {
    **v3.HOST_BOUNDARY,
    "host_expected_value_state": False,
    "host_action_winner_latch": True,
    "host_action_timed_transmission_window": True,
    "shared_outcome_afferent_identical_across_channels": True,
}


@dataclass(frozen=True)
class VocalCreditConfigV4(v3.VocalCreditConfigV3):
    """The single preregistered smoke operating point."""

    outcome_to_value_weight: float = 18.0
    expectation_plateau_center: float = 400.0
    expectation_plateau_slope: float = 0.025
    expectation_plateau_strength: float = 900.0
    expectation_plateau_tau_decay_ms: float = 500.0
    expectation_plateau_tau_rise_ms: float = 2.0
    smoke_warmup_steps: int = 120
    smoke_action_steps: int = 100
    smoke_load_steps: int = 30
    smoke_outcome_steps: int = 40
    smoke_gap_steps: tuple[int, int, int] = (60, 100, 160)


def v4_config() -> VocalCreditConfigV4:
    return VocalCreditConfigV4()


def validate_exact_config(config: VocalCreditConfigV4) -> VocalCreditConfigV4:
    if type(config) is not VocalCreditConfigV4 or asdict(config) != asdict(v4_config()):
        raise ValueError("Gate B v4 requires the exact preregistered configuration.")
    return config


def validate_smoke_seed(seed: int) -> int:
    checked = int(seed)
    if checked != SMOKE_SEED:
        raise ValueError(
            f"Gate B v4 smoke accepts reserved non-scientific seed {SMOKE_SEED} "
            f"only; refusing {checked}."
        )
    return checked


def validate_phase(phase: str) -> str:
    checked = str(phase).strip().lower()
    raise ValueError(
        f"Gate B v4 formal phases are sealed; open phases={OPEN_PHASES}, "
        f"requested={checked!r}."
    )


def validate_formal_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    checked = tuple(int(seed) for seed in seeds)
    raise ValueError(
        "Gate B v4 formal seeds are sealed; named calibration partition "
        f"{CALIBRATION_SEEDS}, received={checked}."
    )


def _is_v3_value_afferent(pathway) -> bool:
    return bool(
        pathway.from_region in {f"motor_{channel}" for channel in CHANNELS}
        and pathway.to_region in {_value(channel) for channel in CHANNELS}
        and pathway.plasticity_gate == VALUE_PLASTICITY_GATE
    )


def _is_v3_normalization_afferent(pathway) -> bool:
    return bool(
        pathway.from_region in {f"motor_{channel}" for channel in CHANNELS}
        and pathway.to_region in {
            v3._value_fs(channel) for channel in CHANNELS
        }
        and pathway.transmission_gate == v3.CRITIC_NORMALIZATION_GATE
    )


def _v4_pathways(
    config: VocalCreditConfigV4,
    *,
    action_permutation: tuple[int, int] = (0, 1),
) -> list[RegionPathway]:
    if tuple(action_permutation) not in ((0, 1), (1, 0)):
        raise ValueError("action permutation must be identity or the fixed swap")

    pathways = [
        pathway
        for pathway in v3._v3_pathways(config)
        if not _is_v3_value_afferent(pathway)
        and not _is_v3_normalization_afferent(pathway)
    ]
    for source_channel in CHANNELS:
        target_channel = int(action_permutation[source_channel])
        pathways.extend([
            RegionPathway(
                from_region=f"motor_{source_channel}",
                to_region=_value(target_channel),
                density=1.0,
                weight_mean=config.motor_to_value_weight,
                weight_jitter=0.0,
                plastic=True,
                plasticity_gate=VALUE_PLASTICITY_GATE,
                transmission_gate=ACTION_COLLATERAL_GATE,
                coincidence_detector=True,
            ),
            RegionPathway(
                from_region=f"motor_{source_channel}",
                to_region=v3._value_fs(target_channel),
                density=1.0,
                weight_mean=config.motor_to_value_fs_weight,
                weight_jitter=0.0,
                plastic=False,
                transmission_gate=v3.CRITIC_NORMALIZATION_GATE,
            ),
        ])
    for channel in CHANNELS:
        pathways.append(RegionPathway(
            from_region=OUTCOME_ONSET,
            to_region=_value(channel),
            density=1.0,
            weight_mean=config.outcome_to_value_weight,
            weight_jitter=0.0,
            plastic=False,
            transmission_gate=SHARED_OUTCOME_READ_GATE,
        ))
    return pathways


def _v4_core_updates(config: VocalCreditConfigV4) -> dict:
    updates = dict(_core_updates(config))
    updates.update({
        # The graded substrate reuses the coincidence routing mask.  The
        # all-or-none contribution is held at zero so only the graded sibling
        # can inject plateau current.
        "enable_coincidence_detection": True,
        "coincidence_weighted_drive": True,
        "coincidence_plateau_strength": 0.0,
        "enable_graded_dendritic_plateau": True,
        "graded_plateau_center": config.expectation_plateau_center,
        "graded_plateau_slope": config.expectation_plateau_slope,
        "graded_plateau_strength": config.expectation_plateau_strength,
        "graded_plateau_tau_decay_ms": (
            config.expectation_plateau_tau_decay_ms
        ),
        "graded_plateau_tau_rise_ms": config.expectation_plateau_tau_rise_ms,
    })
    return updates


def build_v4_bridge(
    seed: int,
    config: VocalCreditConfigV4 | None = None,
    *,
    action_permutation: tuple[int, int] = (0, 1),
):
    validate_smoke_seed(seed)
    config = validate_exact_config(config or v4_config())
    action_permutation = tuple(int(value) for value in action_permutation)
    bridge = build_selector_bridge(
        seed,
        selector_config("v2"),
        extra_regions=v3._v3_regions(config),
        extra_pathways=_v4_pathways(
            config, action_permutation=action_permutation
        ),
        core_config_updates=_v4_core_updates(config),
    )
    actor_routes = {
        channel: _route_synapses(bridge, CREDIT_CUE, _actor(channel))
        for channel in CHANNELS
    }
    expectation_routes = {
        source_channel: _route_synapses(
            bridge,
            f"motor_{source_channel}",
            _value(action_permutation[source_channel]),
        )
        for source_channel in CHANNELS
    }
    routes = CreditRoutes(actor=actor_routes, value=expectation_routes)
    xp, _ = get_backend()
    bridge.set_global_plasticity_gain(0.0)
    _set_learning_gates(
        bridge, config, actor_enabled=True, value_enabled=True
    )
    bridge.cp_reward_eligibility_synapse_indices = xp.asarray(
        routes.all_indices(), dtype=xp.int64
    )
    handles = {
        "routes": routes,
        "expectation_routes": expectation_routes,
        "action_permutation": action_permutation,
    }
    return bridge, handles


def _count_enabled(mask, indices) -> int:
    xp, _ = get_backend()
    device_indices = xp.asarray(indices, dtype=xp.int64)
    return int(np.asarray(to_host(mask[device_indices]), dtype=bool).sum())


def _set_expectation_plateau_route(bridge, handles: dict, enabled: bool) -> None:
    xp, _ = get_backend()
    indices = np.concatenate(list(handles["expectation_routes"].values()))
    bridge.cp_coincidence_synapse_mask[
        xp.asarray(indices, dtype=xp.int64)
    ] = bool(enabled)


def _set_expectation_transmission_window(
    bridge, handles: dict, enabled: bool
) -> None:
    xp, _ = get_backend()
    indices = np.concatenate(list(handles["expectation_routes"].values()))
    bridge.cp_transmission_gain[
        xp.asarray(indices, dtype=xp.int64)
    ] = xp.float32(1.0 if enabled else 0.0)


def _route_telemetry(bridge, handles: dict) -> dict:
    mask = bridge.cp_coincidence_synapse_mask
    by_source = {}
    for source, indices in handles["expectation_routes"].items():
        by_source[str(source)] = {
            "target": int(handles["action_permutation"][source]),
            "total": int(indices.size),
            "plateau_enabled": _count_enabled(mask, indices),
        }
    expected = np.concatenate(list(handles["expectation_routes"].values()))
    enabled_total = int(np.asarray(to_host(mask), dtype=bool).sum())
    return {
        "by_source": by_source,
        "total": int(expected.size),
        "enabled": _count_enabled(mask, expected),
        "all_enabled_coincidence_synapses_belong_to_expectation_route": bool(
            enabled_total == _count_enabled(mask, expected)
        ),
    }


def _shared_outcome_routes_are_symmetric(bridge, config) -> bool:
    pathways = list(bridge.region_manager.pathways())
    routes = [
        pathway
        for pathway in pathways
        if pathway.from_region == OUTCOME_ONSET
        and pathway.to_region in {_value(channel) for channel in CHANNELS}
    ]
    return bool(
        len(routes) == len(CHANNELS)
        and {pathway.to_region for pathway in routes}
        == {_value(channel) for channel in CHANNELS}
        and all(
            pathway.density == 1.0
            and pathway.weight_mean == config.outcome_to_value_weight
            and pathway.weight_jitter == 0.0
            and not pathway.plastic
            and pathway.transmission_gate == SHARED_OUTCOME_READ_GATE
            for pathway in routes
        )
    )


def structural_preconditions(bridge, handles: dict, config) -> dict[str, bool]:
    inherited = v3._structural_preconditions(
        bridge, handles["routes"], config
    )
    # V3 forbade all outcome-to-value afferents.  V4 replaces that check with
    # the stricter requirement that one fixed generic afferent reaches both
    # value pools with identical anatomy and no action information.
    inherited.pop("outcome_afferents_are_generic_not_channel_specific", None)
    inherited.pop("local_fs_normalization_exists_for_both_critics", None)
    formal = set(CALIBRATION_SEEDS + DEVELOPMENT_SEEDS + HELD_OUT_SEEDS)
    old = set(
        v3.CALIBRATION_SEEDS + v3.DEVELOPMENT_SEEDS + v3.HELD_OUT_SEEDS
    )
    route_state = _route_telemetry(bridge, handles)
    inherited.update({
        "shared_outcome_read_is_fixed_symmetric_and_action_agnostic": (
            _shared_outcome_routes_are_symmetric(bridge, config)
        ),
        "local_fs_normalization_follows_each_action_value_microchannel": all(
            any(
                pathway.from_region == f"motor_{source}"
                and pathway.to_region
                == v3._value_fs(handles["action_permutation"][source])
                and pathway.transmission_gate
                == v3.CRITIC_NORMALIZATION_GATE
                for pathway in bridge.region_manager.pathways()
            )
            for source in CHANNELS
        ),
        "graded_plateau_uses_only_plastic_action_value_routes": bool(
            route_state[
                "all_enabled_coincidence_synapses_belong_to_expectation_route"
            ]
            and route_state["enabled"] in (0, route_state["total"])
            and route_state["total"] > 0
        ),
        "all_or_none_plateau_contribution_is_zero": bool(
            bridge.core_config.coincidence_plateau_strength == 0.0
        ),
        "graded_plateau_state_is_on_substrate": bool(
            bridge.cp_conductance_g_graded_plateau is not None
            and bridge.cp_conductance_g_graded_plateau_rise is not None
        ),
        "fresh_formal_partitions_are_disjoint_and_sealed": bool(
            OPEN_PHASES == ()
            and SMOKE_SEED not in formal
            and not (formal & old)
            and len(formal)
            == len(CALIBRATION_SEEDS)
            + len(DEVELOPMENT_SEEDS)
            + len(HELD_OUT_SEEDS)
        ),
    })
    return inherited


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


def _rates(spikes: np.ndarray, n_cells: int, steps: int, dt_ms: float) -> list[float]:
    seconds = max(float(steps) * float(dt_ms) / 1000.0, 1e-12)
    return [float(count / max(1, n_cells) / seconds) for count in spikes]


def _select_action_and_load(bridge, handles, config, selector) -> dict:
    motor_counts = np.zeros(len(CHANNELS), dtype=np.int64)
    value_spikes = np.zeros(len(CHANNELS), dtype=np.int64)
    fs_spikes = np.zeros(len(CHANNELS), dtype=np.int64)
    winner = None

    # Listen for an action only during selection. The actor collateral remains
    # untouched; only the expectation afferents close outside this window.
    _set_expectation_transmission_window(bridge, handles, False)
    v3._set_trial_current_v3(bridge, selector, config)
    _step(bridge, config.smoke_warmup_steps)
    v3._set_trial_current_v3(
        bridge, selector, config, arousal=True, cue=True
    )
    tag_window_open = False
    for _ in range(config.smoke_action_steps):
        _step(bridge)
        firing = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        for channel in CHANNELS:
            motor_counts[channel] += _region_spikes(
                firing, bridge, f"motor_{channel}"
            )
        if not tag_window_open:
            # The substrate consumes previous-step spikes. Opening after the
            # first action step excludes the final pre-action warmup state.
            _set_expectation_transmission_window(bridge, handles, True)
            tag_window_open = True
        crossed = [
            channel
            for channel in CHANNELS
            if motor_counts[channel] >= selector.commit_threshold_spikes
        ]
        if len(crossed) == 1:
            winner = int(crossed[0])
            # Consume the just-observed committed motor volley once, then
            # close before any subsequent motor activity can retag the trace.
            _step(bridge)
            break

    _set_expectation_transmission_window(bridge, handles, False)
    v3._set_trial_current_v3(bridge, selector, config, cue=True)
    for _ in range(config.smoke_load_steps):
        _step(bridge)
        firing = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        for channel in CHANNELS:
            value_spikes[channel] += _region_spikes(
                firing, bridge, _value(channel)
            )
            fs_spikes[channel] += _region_spikes(
                firing, bridge, v3._value_fs(channel)
            )

    return {
        "winner": winner,
        "motor_spikes": motor_counts.tolist(),
        "load_value_spikes": value_spikes.tolist(),
        "load_value_fs_spikes": fs_spikes.tolist(),
        "plateau_after_load": _plateau_means(bridge),
    }


def run_smoke_condition(
    control: str,
    *,
    gap_steps: int,
    config: VocalCreditConfigV4 | None = None,
) -> dict:
    if control not in SMOKE_CONTROLS:
        raise ValueError(f"unknown smoke control {control!r}")
    config = validate_exact_config(config or v4_config())
    permutation = (1, 0) if control == "action_channel_permutation" else (0, 1)
    bridge, handles = build_v4_bridge(
        SMOKE_SEED, config, action_permutation=permutation
    )
    selector = selector_config("v2")
    if control == "plateau_lesion":
        _set_expectation_plateau_route(bridge, handles, False)
    if control == "shared_outcome_read_lesion":
        bridge.set_transmission_gate(SHARED_OUTCOME_READ_GATE, 0.0)
    if control == "frozen_expectation_route":
        bridge.set_plasticity_gate(VALUE_PLASTICITY_GATE, 0.0)

    structural = structural_preconditions(bridge, handles, config)
    route_before = _route_telemetry(bridge, handles)
    weights_before = np.asarray(
        to_host(bridge.cp_connections.data), dtype=np.float32
    ).copy()
    action = _select_action_and_load(bridge, handles, config, selector)

    delay_spikes = np.zeros(len(CHANNELS), dtype=np.int64)
    v3._set_trial_current_v3(bridge, selector, config)
    for _ in range(int(gap_steps)):
        _step(bridge)
        firing = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        for channel in CHANNELS:
            delay_spikes[channel] += _region_spikes(
                firing, bridge, _value(channel)
            )
    plateau_before_outcome = _plateau_means(bridge)

    outcome_spikes = np.zeros(len(CHANNELS), dtype=np.int64)
    outcome_onset_spikes = 0
    v3._set_trial_current_v3(bridge, selector, config, outcome=True)
    for _ in range(config.smoke_outcome_steps):
        _step(bridge)
        firing = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        outcome_onset_spikes += _region_spikes(firing, bridge, OUTCOME_ONSET)
        for channel in CHANNELS:
            outcome_spikes[channel] += _region_spikes(
                firing, bridge, _value(channel)
            )

    weights_after = np.asarray(
        to_host(bridge.cp_connections.data), dtype=np.float32
    )
    changed = np.abs(weights_after - weights_before) > 1e-7
    actor_indices = np.concatenate(
        list(handles["routes"].actor.values())
    )
    expectation_indices = np.concatenate(
        list(handles["expectation_routes"].values())
    )
    declared = np.zeros(changed.size, dtype=bool)
    declared[actor_indices] = True
    declared[expectation_indices] = True
    n_cells = int(config.n_value)
    result = {
        "control": control,
        "seed": SMOKE_SEED,
        "scientific_seed_executed": False,
        "gap_steps": int(gap_steps),
        "config": asdict(config),
        "action": action,
        "expected_value_channel": (
            None
            if action["winner"] is None
            else int(permutation[action["winner"]])
        ),
        "plateau_before_outcome": plateau_before_outcome,
        "delay_value_spikes": delay_spikes.tolist(),
        "delay_value_rate_hz_per_cell": _rates(
            delay_spikes, n_cells, gap_steps, bridge.core_config.dt_ms
        ),
        "outcome_onset_spikes": int(outcome_onset_spikes),
        "outcome_value_spikes": outcome_spikes.tolist(),
        "outcome_value_rate_hz_per_cell": _rates(
            outcome_spikes,
            n_cells,
            config.smoke_outcome_steps,
            bridge.core_config.dt_ms,
        ),
        "route_telemetry_before_action": route_before,
        "route_telemetry_after_outcome": _route_telemetry(bridge, handles),
        "shared_outcome_read_gain": float(
            bridge._transmission_gate_values[SHARED_OUTCOME_READ_GATE]
        ),
        "expectation_plasticity_gain": float(
            bridge._plasticity_gate_values[VALUE_PLASTICITY_GATE]
        ),
        "actor_plasticity_gain": float(
            bridge._plasticity_gate_values[CREDIT_PLASTICITY_GATE]
        ),
        "changed_synapses": int(np.count_nonzero(changed)),
        "changed_actor_synapses": int(np.count_nonzero(changed[actor_indices])),
        "changed_expectation_synapses": int(np.count_nonzero(
            changed[expectation_indices]
        )),
        "changed_outside_declared_routes": int(np.count_nonzero(
            changed & ~declared
        )),
        "structural_preconditions": structural,
        "host_expected_value_state": False,
    }
    del bridge
    return result


def _smoke_checks(rows: dict[str, dict]) -> dict[str, bool]:
    intact = rows["intact"]
    plateau_lesion = rows["plateau_lesion"]
    outcome_lesion = rows["shared_outcome_read_lesion"]
    permuted = rows["action_channel_permutation"]
    frozen = rows["frozen_expectation_route"]
    winner = intact["action"]["winner"]
    expected = intact["expected_value_channel"]
    permuted_expected = permuted["expected_value_channel"]
    intact_outcome = (
        0.0 if expected is None else intact["outcome_value_rate_hz_per_cell"][expected]
    )
    lesion_outcome = (
        0.0
        if expected is None
        else plateau_lesion["outcome_value_rate_hz_per_cell"][expected]
    )
    outcome_read_lesion_rate = (
        0.0
        if expected is None
        else outcome_lesion["outcome_value_rate_hz_per_cell"][expected]
    )
    return {
        "all_anatomy_and_scope_checks_pass": all(
            all(row["structural_preconditions"].values())
            for row in rows.values()
        ),
        "shared_selector_commits_without_host_channel_drive": bool(
            winner is not None
            and all(row["action"]["winner"] == winner for row in rows.values())
        ),
        "intact_action_tag_maps_to_executed_channel": bool(expected == winner),
        "intact_action_tag_is_neurally_channel_selective": bool(
            expected is not None
            and intact["plateau_before_outcome"][expected]
            > intact["plateau_before_outcome"][1 - expected]
            and intact["outcome_value_rate_hz_per_cell"][expected]
            > intact["outcome_value_rate_hz_per_cell"][1 - expected]
        ),
        "permutation_moves_tag_to_opposite_value_channel": bool(
            winner is not None and permuted_expected == 1 - winner
        ),
        "permuted_action_tag_is_neurally_channel_selective": bool(
            permuted_expected is not None
            and permuted["plateau_before_outcome"][permuted_expected]
            > permuted["plateau_before_outcome"][1 - permuted_expected]
            and permuted["outcome_value_rate_hz_per_cell"][permuted_expected]
            > permuted["outcome_value_rate_hz_per_cell"][1 - permuted_expected]
        ),
        "intact_plateau_persists_to_outcome": bool(
            expected is not None and intact["plateau_before_outcome"][expected] > 0.0
        ),
        "plateau_lesion_removes_dendritic_state": bool(
            max(plateau_lesion["plateau_before_outcome"]) == 0.0
        ),
        "shared_outcome_afferent_is_active": bool(
            intact["outcome_onset_spikes"] > 0
        ),
        "intact_value_firing_is_bounded_1_to_20_hz_per_cell": bool(
            1.0 <= intact_outcome <= 20.0
        ),
        "delay_value_firing_is_near_zero": bool(
            max(intact["delay_value_rate_hz_per_cell"]) <= 0.5
        ),
        "plateau_is_load_bearing_for_outcome_read": bool(
            intact_outcome > lesion_outcome
        ),
        "shared_outcome_read_is_load_bearing": bool(
            intact_outcome > outcome_read_lesion_rate
        ),
        "frozen_expectation_route_preserves_actor_scope": bool(
            frozen["expectation_plasticity_gain"] == 0.0
            and frozen["actor_plasticity_gain"] == v4_config().actor_plasticity_gain
            and frozen["changed_expectation_synapses"] == 0
            and frozen["changed_actor_synapses"] > 0
            and frozen["changed_outside_declared_routes"] == 0
        ),
    }


def run_smoke(config: VocalCreditConfigV4 | None = None) -> dict:
    config = validate_exact_config(config or v4_config())
    short_gap, middle_gap, long_gap = config.smoke_gap_steps
    rows = {
        control: run_smoke_condition(
            control, gap_steps=middle_gap, config=config
        )
        for control in SMOKE_CONTROLS
    }
    decay_rows = {
        str(gap): run_smoke_condition("intact", gap_steps=gap, config=config)
        for gap in (short_gap, middle_gap, long_gap)
    }
    checks = _smoke_checks(rows)
    expected = rows["intact"]["expected_value_channel"]
    plateau_by_gap = [
        decay_rows[str(gap)]["plateau_before_outcome"][expected]
        if expected is not None else 0.0
        for gap in (short_gap, middle_gap, long_gap)
    ]
    response_by_gap = [
        decay_rows[str(gap)]["outcome_value_rate_hz_per_cell"][expected]
        if expected is not None else 0.0
        for gap in (short_gap, middle_gap, long_gap)
    ]
    checks.update({
        "plateau_state_decays_across_preregistered_gaps": bool(
            plateau_by_gap[0] > plateau_by_gap[1] > plateau_by_gap[2] >= 0.0
        ),
        "outcome_read_response_is_recorded_at_every_gap": bool(
            len(response_by_gap) == 3
            and all(np.isfinite(value) for value in response_by_gap)
        ),
    })
    xp, backend_name = get_backend()
    if backend_name == "cupy":
        properties = xp.cuda.runtime.getDeviceProperties(0)
        device_name = properties["name"]
        if isinstance(device_name, bytes):
            device_name = device_name.decode("utf-8", errors="replace")
    else:
        device_name = "CPU (NumPy)"
    preconditions = [
        {
            "name": "reserved non-scientific seed zero only",
            "ok": all(
                row["seed"] == SMOKE_SEED
                and row["scientific_seed_executed"] is False
                for row in rows.values()
            ),
        },
        {
            "name": "all structural anatomy and scope checks measured",
            "ok": all(
                all(row["structural_preconditions"].values())
                for row in rows.values()
            ),
        },
        {
            "name": "all smoke measurements finite",
            "ok": all(
                np.isfinite(row["plateau_before_outcome"]).all()
                and np.isfinite(row["outcome_value_rate_hz_per_cell"]).all()
                for row in rows.values()
            ),
        },
    ]
    return {
        "probe": "vocal_action_credit_gate_b_v4",
        "version": "v4-smoke-only",
        "backend": backend_name,
        "device": str(device_name),
        "preconditions": preconditions,
        "retired": RETIRED,
        "retirement_reasons": [
            "load-bearing host winner latch controls expectation-tag duration",
            "CuPy outcome firing exceeds the locked physiology ceiling",
            "late motor activity can drive the wrong value-FS microchannel",
        ],
        "status": "SMOKE_PASS" if all(checks.values()) else "SMOKE_FAIL",
        "science_seed_executed": False,
        "smoke_seed": SMOKE_SEED,
        "config": asdict(config),
        "seed_policy": {
            "open_phases": list(OPEN_PHASES),
            "calibration_sealed": list(CALIBRATION_SEEDS),
            "development_sealed": list(DEVELOPMENT_SEEDS),
            "held_out_sealed": list(HELD_OUT_SEEDS),
        },
        "formal_controls_preregistered": list(FORMAL_CONTROLS),
        "host_boundary": dict(HOST_BOUNDARY),
        "checks": checks,
        "conditions": rows,
        "gap_decay": {
            "gaps_steps": [short_gap, middle_gap, long_gap],
            "plateau_expected_channel": plateau_by_gap,
            "outcome_rate_expected_channel_hz_per_cell": response_by_gap,
            "rows": decay_rows,
        },
        "interpretation_limit": (
            "Seed-zero smoke checks mechanism engagement and falsifiers only; "
            "it is not evidence of learned vocal credit or a Gate B verdict."
        ),
    }


def run_formal_seed(seed: int, config: VocalCreditConfigV4 | None = None):
    validate_formal_seeds((seed,))
    raise AssertionError("unreachable")


def resolve_cli_request(*, smoke: bool, phase: str | None, seeds) -> str:
    if smoke:
        if phase is not None or seeds is not None:
            raise ValueError("--smoke does not accept --phase or --seeds")
        return "smoke"
    validate_phase(phase or "calibration")
    return "formal"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the sealed Gate B v4 seed-zero physiology smoke."
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--phase", choices=("calibration", "development", "held_out")
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    resolve_cli_request(smoke=args.smoke, phase=args.phase, seeds=args.seeds)
    result = run_smoke()
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    print(rendered, end="")
    return 0 if result["status"] == "SMOKE_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
