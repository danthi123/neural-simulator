"""Bounded v4 calibration for replay-driven cortical consolidation.

V3 reached the learned cortical index during uncued replay but did not recruit
the cortical target strongly enough to engage its fast-spiking loop. V4 changes
only that synaptic boundary: existing index-to-target synapses also feed the
weighted dendritic-coincidence mask. Their ordinary AMPA current remains
active, and the inherited CA1-to-index plateau route is unchanged.

The target-plateau lesion removes only the index-output synapses from the
coincidence mask during sleep. All v3 weights, thresholds, plateau dynamics,
timing, populations, controls, and scientific criteria are retained.

Seed 216 is construction/dynamics smoke only and can never produce a
scientific verdict. Its dynamics check failed before formal execution, so all
scientific phases and named seed partitions are mechanically closed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.runners import _replay_cortical_consolidation_gate_v3 as v3  # noqa: E402
from tools.verdict import UNDEFINED, Verdict  # noqa: E402


OPEN_PHASES = ()
SMOKE_SEED = 216
CALIBRATION_SEEDS = (451, 457)
DEVELOPMENT_SEEDS = (461, 463, 467)
HELD_OUT_SEEDS = (479, 487, 491)
CONDITIONS = v3.CONDITIONS + ("target_plateau_lesion",)

GateConfig = v3.GateConfig
CA3_GATE = v3.CA3_GATE
INDEX_CUE_GATE = v3.INDEX_CUE_GATE
INDEX_TARGET_GATE = v3.INDEX_TARGET_GATE
CORTICAL_GATE = v3.CORTICAL_GATE
SCHAFFER_GATE = v3.SCHAFFER_GATE
TARGET_INHIBITION_GATE = v3.TARGET_INHIBITION_GATE
INDEX_OUTPUT_GATE = v3.INDEX_OUTPUT_GATE
INDEX_BALANCE_GATE = v3.INDEX_BALANCE_GATE

TARGET_ACTIVITY_PRESERVATION = 0.75
TARGET_ACTIVITY_REMAINING = 0.75
MIN_TARGET_SPIKE_REDUCTION = 1


def smoke_config() -> GateConfig:
    """Use v3's unchanged full configuration for an informative dynamics smoke."""
    return GateConfig()


def validate_phase(phase: str) -> str:
    checked = str(phase).strip().lower()
    if checked not in OPEN_PHASES:
        raise ValueError(
            f"This bounded v4 runner opens {OPEN_PHASES} only; refusing phase {phase!r}."
        )
    return checked


def validate_calibration_seed(seed: int) -> int:
    checked = int(seed)
    raise ValueError(
        "Replay v4 was retired after its smoke dynamics check; all scientific "
        f"seeds are locked, including requested seed {checked}."
    )


def validate_calibration_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    checked = tuple(int(seed) for seed in seeds)
    raise ValueError(
        "Replay v4 was retired after its smoke dynamics check; the named "
        f"calibration partition {CALIBRATION_SEEDS} is locked, received {checked}."
    )


def validate_smoke_seed(seed: int) -> int:
    checked = int(seed)
    if checked != SMOKE_SEED:
        raise ValueError(
            f"Smoke execution accepts non-scientific seed {SMOKE_SEED} only; "
            f"refusing seed {checked}."
        )
    return checked


def _target_route_indices(bridge):
    return bridge._transmission_gate_indices_gpu[INDEX_OUTPUT_GATE]


def _index_route_indices(bridge):
    return bridge._plasticity_gate_indices_gpu[INDEX_TARGET_GATE]


def _count_enabled(mask, indices) -> int:
    from sim.backend import to_host

    return int(np.asarray(to_host(mask[indices])).sum())


def _set_target_plateau_route(bridge, enabled: bool) -> None:
    """Toggle only index-output coincidence routing; AMPA gating is untouched."""
    bridge.cp_coincidence_synapse_mask[_target_route_indices(bridge)] = bool(enabled)


def _coincidence_route_telemetry(bridge) -> dict[str, int]:
    mask = bridge.cp_coincidence_synapse_mask
    target_indices = _target_route_indices(bridge)
    index_indices = _index_route_indices(bridge)
    return {
        "target_route_total": int(target_indices.size),
        "target_route_enabled": _count_enabled(mask, target_indices),
        "ca1_index_route_total": int(index_indices.size),
        "ca1_index_route_enabled": _count_enabled(mask, index_indices),
    }


def build_bridge(seed: int, config: GateConfig) -> tuple[object, dict]:
    """Build v3 unchanged, then add its index-output synapses to coincidence routing."""
    bridge, handles = v3.build_bridge(seed, config)
    target_indices = _target_route_indices(bridge)
    if target_indices.size == 0:
        raise RuntimeError("Index-output route has no synapses to coincidence-route.")
    if _count_enabled(bridge.cp_coincidence_synapse_mask, target_indices) != 0:
        raise RuntimeError("V3 index-output synapses unexpectedly already use coincidence routing.")
    _set_target_plateau_route(bridge, True)
    handles["coincidence_routes"] = _coincidence_route_telemetry(bridge)
    return bridge, handles


def _target_plateau_snapshot(bridge, handles: dict) -> dict[str, dict[str, float]]:
    from sim.backend import to_host

    conductance = bridge.cp_conductance_g_coincidence
    snapshots: dict[str, dict[str, float]] = {}
    for memory in ("A", "B"):
        values = np.asarray(
            to_host(conductance[handles["device_patterns"][memory]["target"]]),
            dtype=np.float64,
        )
        snapshots[memory] = {
            "peak": float(values.max(initial=0.0)),
            "mean": float(values.mean()) if values.size else 0.0,
        }
    return snapshots


def _sleep(bridge, handles: dict, condition: str, seed: int, config: GateConfig) -> dict:
    from sim.backend import get_backend

    xp, _ = get_backend()
    v3.v1._clear_dynamics(bridge)
    v3._clear_event_dynamics(bridge, preserve_slow_balance=False)
    cortical_on = condition != "cortical_plasticity_off"
    v3._set_phase_gates(bridge, sleep=True, cortical=cortical_on)
    bridge.core_config.hebbian_learning_rate = float(config.cortical_sleep_learning_rate)
    if condition == "ca3_ca1_lesion":
        bridge.set_transmission_gate(SCHAFFER_GATE, 0.0)

    target_inhibition_gain = 0.0 if condition == "target_inhibition_lesion" else 1.0
    relay_gain = 0.0 if condition == "index_relay_lesion" else 1.0
    balance_gain = 0.0 if condition == "index_balance_lesion" else 1.0
    target_plateau_enabled = condition != "target_plateau_lesion"
    bridge.set_transmission_gate(TARGET_INHIBITION_GATE, target_inhibition_gain)
    bridge.set_transmission_gate(INDEX_OUTPUT_GATE, relay_gain)
    bridge.set_transmission_gate(INDEX_BALANCE_GATE, balance_gain)
    _set_target_plateau_route(bridge, target_plateau_enabled)
    routes_during_sleep = _coincidence_route_telemetry(bridge)
    index_output_transmission_gain = float(
        bridge._transmission_gate_values[INDEX_OUTPUT_GATE]
    )
    shuffled_edges = (
        v3.v1._shuffle_target_index(bridge, seed)
        if condition == "shuffled_target_index"
        else 0
    )

    regions = handles["regions"]
    events = v3.v2._ordered_sleep_events(
        seed,
        config,
        regions["ca3"],
        shuffle=condition == "shuffled_replay_order",
    )
    region_devices = {
        name: xp.asarray(regions[name], dtype=xp.int64)
        for name in (
            "ca3",
            "ca1",
            "cortical_cue",
            "cortical_target",
            "cortical_target_fs",
            "cortical_index",
            "cortical_index_fs",
        )
    }
    ca3_a = handles["device_patterns"]["A"]["ca3"]
    ca3_b = handles["device_patterns"]["B"]["ca3"]
    event_winners: list[str] = []
    spike_totals = {name: 0 for name in region_devices}
    balance_trace: list[dict[str, float]] = []
    plateau_peak = {memory: 0.0 for memory in ("A", "B")}
    plateau_area = {memory: 0.0 for memory in ("A", "B")}

    for event in events:
        v3._clear_event_dynamics(bridge, preserve_slow_balance=True)
        event_a = event_b = 0
        if condition == "no_sleep":
            v3.v1._step(bridge, config.sleep_noise_steps + config.sleep_free_steps)
        else:
            background_dev = xp.asarray(event, dtype=xp.int64)
            for step in range(config.sleep_noise_steps + config.sleep_free_steps):
                v3.v1._zero_current(bridge)
                if step < config.sleep_noise_steps:
                    bridge.cp_external_input_current[background_dev] = config.sleep_drive_pA
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
                firing = bridge.cp_firing_states
                event_a += int(firing[ca3_a].sum())
                event_b += int(firing[ca3_b].sum())
                for name, indices in region_devices.items():
                    spike_totals[name] += int(firing[indices].sum())
                plateau = _target_plateau_snapshot(bridge, handles)
                for memory in ("A", "B"):
                    plateau_peak[memory] = max(
                        plateau_peak[memory], plateau[memory]["peak"]
                    )
                    plateau_area[memory] += plateau[memory]["mean"]
        if event_a == event_b == 0:
            event_winners.append("none")
        elif event_a > event_b:
            event_winners.append("A")
        elif event_b > event_a:
            event_winners.append("B")
        else:
            event_winners.append("tie")
        balance_trace.append(v3._balance_conductance(bridge, handles))

    v3.v1._zero_current(bridge)
    final_balance = v3._balance_conductance(bridge, handles)
    bridge.set_transmission_gate(TARGET_INHIBITION_GATE, 1.0)
    bridge.set_transmission_gate(INDEX_OUTPUT_GATE, 1.0)
    bridge.set_transmission_gate(INDEX_BALANCE_GATE, 1.0)
    _set_target_plateau_route(bridge, True)
    return {
        "events": int(len(events)),
        "event_winners": event_winners,
        "reactivated_events": int(sum(winner != "none" for winner in event_winners)),
        "replayed_A": int(sum(winner == "A" for winner in event_winners)),
        "replayed_B": int(sum(winner == "B" for winner in event_winners)),
        "spikes": spike_totals,
        "shuffled_edges": int(shuffled_edges),
        "event_content_multiset_digest": v3.v2._event_digest(events, order_sensitive=False),
        "event_order_digest": v3.v2._event_digest(events, order_sensitive=True),
        "mean_adjacent_input_overlap": v3.v2._mean_adjacent_overlap(events),
        "target_inhibition_gain_during_sleep": target_inhibition_gain,
        "index_relay_gain_during_sleep": relay_gain,
        "index_balance_gain_during_sleep": balance_gain,
        "index_output_transmission_gain_during_sleep": index_output_transmission_gain,
        "coincidence_routes_during_sleep": routes_during_sleep,
        "target_plateau_peak": plateau_peak,
        "target_plateau_area": plateau_area,
        "target_plateau_peak_overall": max(plateau_peak.values()),
        "target_plateau_area_overall": float(sum(plateau_area.values())),
        "index_balance_conductance_final": final_balance,
        "index_balance_conductance_peak": {
            memory: max((row[memory] for row in balance_trace), default=0.0)
            for memory in ("A", "B")
        },
        "host_selected_episode_for_replay": False,
        "host_selected_target_drive": False,
    }


def run_condition(
    seed: int,
    condition: str,
    config: GateConfig | None = None,
    *,
    smoke: bool = False,
) -> dict:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; expected one of {CONDITIONS}.")
    if smoke:
        validate_smoke_seed(seed)
    else:
        validate_calibration_seed(seed)
    cfg = config or GateConfig()
    bridge, handles = build_bridge(seed, cfg)
    bridge_ids = [id(bridge)]
    phase_trace: list[str] = []

    bridge.set_transmission_gate(TARGET_INHIBITION_GATE, 0.0)
    before = {
        "ca3": v3.v1._path_weights(bridge, CA3_GATE),
        "index_cue": v3.v1._path_weights(bridge, INDEX_CUE_GATE),
        "index_target": v3.v1._path_weights(bridge, INDEX_TARGET_GATE),
        "cortical": v3.v1._path_weights(bridge, CORTICAL_GATE),
    }
    encode_a = v3._encode_memory(bridge, handles, "A", cfg.encode_a_events, cfg)
    phase_trace.append("encode_A")
    bridge_ids.append(id(bridge))
    after_a = {
        "ca3": v3.v1._path_weights(bridge, CA3_GATE),
        "cortical": v3.v1._path_weights(bridge, CORTICAL_GATE),
    }
    encode_b = v3._encode_memory(bridge, handles, "B", cfg.encode_b_events, cfg)
    phase_trace.append("encode_B")
    bridge_ids.append(id(bridge))
    after_b = {
        "ca3": v3.v1._path_weights(bridge, CA3_GATE),
        "index_cue": v3.v1._path_weights(bridge, INDEX_CUE_GATE),
        "index_target": v3.v1._path_weights(bridge, INDEX_TARGET_GATE),
        "cortical": v3.v1._path_weights(bridge, CORTICAL_GATE),
    }
    sleep = _sleep(bridge, handles, condition, seed, cfg)
    phase_trace.append("sleep")
    bridge_ids.append(id(bridge))
    after_sleep = {
        "ca3": v3.v1._path_weights(bridge, CA3_GATE),
        "index_cue": v3.v1._path_weights(bridge, INDEX_CUE_GATE),
        "index_target": v3.v1._path_weights(bridge, INDEX_TARGET_GATE),
        "cortical": v3.v1._path_weights(bridge, CORTICAL_GATE),
    }

    v3._set_phase_gates(bridge)
    v3._clear_event_dynamics(bridge, preserve_slow_balance=False)
    recall = {
        memory: v3.v1._probe_memory(bridge, handles, memory, cfg)
        for memory in ("A", "B")
    }
    phase_trace.append("retest")
    bridge_ids.append(id(bridge))

    def mean_delta(later: np.ndarray, earlier: np.ndarray) -> float:
        return float(np.mean(later - earlier))

    return {
        "seed": int(seed),
        "seed_partition": "smoke" if smoke else "calibration",
        "scientific_partition": not smoke,
        "condition": condition,
        "config": asdict(cfg),
        "phase_trace": phase_trace,
        "single_bridge_persisted": len(set(bridge_ids)) == 1,
        "wiring_counts": handles["wiring_counts"],
        "neutral_index_fan_in": handles["neutral_index_fan_in"],
        "inhibitory_neuron_count": int(len(handles["inhibitory_indices"])),
        "coincidence_routes_after_build": handles["coincidence_routes"],
        "encode_A": encode_a,
        "encode_B": encode_b,
        "sleep": sleep,
        "recall": recall,
        "weight_deltas": {
            "ca3_during_encode_A": mean_delta(after_a["ca3"], before["ca3"]),
            "ca3_during_encode_B": mean_delta(after_b["ca3"], after_a["ca3"]),
            "ca3_during_sleep": mean_delta(after_sleep["ca3"], after_b["ca3"]),
            "index_cue_during_wake": mean_delta(after_b["index_cue"], before["index_cue"]),
            "index_target_during_wake": mean_delta(
                after_b["index_target"], before["index_target"]
            ),
            "index_cue_during_sleep": mean_delta(
                after_sleep["index_cue"], after_b["index_cue"]
            ),
            "index_target_during_sleep": mean_delta(
                after_sleep["index_target"], after_b["index_target"]
            ),
            "cortical_during_wake": mean_delta(after_b["cortical"], before["cortical"]),
            "cortical_during_sleep": mean_delta(
                after_sleep["cortical"], after_b["cortical"]
            ),
        },
    }


def _materially_reduced(intact: int, lesion: int) -> bool:
    return (
        intact - lesion >= MIN_TARGET_SPIKE_REDUCTION
        and lesion <= TARGET_ACTIVITY_REMAINING * intact
    )


def _preserved(intact: int, lesion: int) -> bool:
    return lesion > 0 and lesion >= TARGET_ACTIVITY_PRESERVATION * intact


def _calibration_verdict(conditions: dict[str, dict]) -> dict:
    inherited = v3._calibration_verdict(conditions)
    intact = conditions["intact"]
    lesion = conditions["target_plateau_lesion"]
    intact_sleep = intact["sleep"]
    lesion_sleep = lesion["sleep"]
    intact_routes = intact_sleep["coincidence_routes_during_sleep"]
    lesion_routes = lesion_sleep["coincidence_routes_during_sleep"]

    route_gate = Verdict("replay-driven cortical consolidation v4 route isolation")
    route_gate.require(
        "intact enables every index-output coincidence synapse",
        intact_routes["target_route_total"] > 0
        and intact_routes["target_route_enabled"] == intact_routes["target_route_total"],
        expect=True,
    )
    route_gate.require(
        "target plateau lesion removes only coincidence while preserving AMPA",
        lesion_routes["target_route_enabled"] == 0
        and lesion_sleep["index_output_transmission_gain_during_sleep"] == 1.0,
        expect=True,
    )
    route_gate.require(
        "target plateau lesion preserves the CA1-to-index plateau route",
        lesion_routes["ca1_index_route_total"] > 0
        and lesion_routes["ca1_index_route_enabled"]
        == lesion_routes["ca1_index_route_total"],
        expect=True,
    )
    route_decision = route_gate.decide(go=True, verbose=False)

    target_checks = {
        "intact_target_plateau_nonzero": (
            intact_sleep["target_plateau_peak_overall"] > 0.0
            and intact_sleep["target_plateau_area_overall"] > 0.0
        ),
        "intact_target_and_fs_recruited": (
            intact_sleep["spikes"]["cortical_target"] > 0
            and intact_sleep["spikes"]["cortical_target_fs"] > 0
        ),
        "target_plateau_lesion_preserves_index": _preserved(
            intact_sleep["spikes"]["cortical_index"],
            lesion_sleep["spikes"]["cortical_index"],
        ),
        "target_plateau_lesion_preserves_index_fs": _preserved(
            intact_sleep["spikes"]["cortical_index_fs"],
            lesion_sleep["spikes"]["cortical_index_fs"],
        ),
        "target_plateau_is_load_bearing_for_target": _materially_reduced(
            intact_sleep["spikes"]["cortical_target"],
            lesion_sleep["spikes"]["cortical_target"],
        ),
        "target_plateau_is_load_bearing_for_target_fs": _materially_reduced(
            intact_sleep["spikes"]["cortical_target_fs"],
            lesion_sleep["spikes"]["cortical_target_fs"],
        ),
    }
    checks = dict(inherited["checks"])
    checks.update(target_checks)
    if (
        inherited["calibration_status"] == "UNDEFINED"
        or route_decision["status"] == UNDEFINED
    ):
        status = "UNDEFINED"
    elif all(checks.values()):
        status = "CALIBRATION_PROMISING"
    else:
        status = "CALIBRATION_NEEDS_REVISION"

    return {
        **inherited,
        "calibration_status": status,
        "preconditions": inherited["preconditions"] + route_decision["preconditions"],
        "undefined_reasons": (
            inherited["undefined_reasons"] + route_decision["undefined_reasons"]
        ),
        "checks": checks,
        "target_plateau_checks": target_checks,
        "target_plateau_telemetry": {
            "intact": {
                "peak": intact_sleep["target_plateau_peak_overall"],
                "area": intact_sleep["target_plateau_area_overall"],
                "spikes": intact_sleep["spikes"],
            },
            "lesion": {
                "peak": lesion_sleep["target_plateau_peak_overall"],
                "area": lesion_sleep["target_plateau_area_overall"],
                "spikes": lesion_sleep["spikes"],
            },
        },
    }


def run_seed(seed: int, config: GateConfig | None = None) -> dict:
    checked = validate_calibration_seed(seed)
    cfg = config or GateConfig()
    conditions = {
        condition: run_condition(checked, condition, cfg) for condition in CONDITIONS
    }
    verdict = _calibration_verdict(conditions)
    return {
        "seed": checked,
        "seed_partition": "calibration",
        "scientific_partition": True,
        "conditions": conditions,
        "calibration": verdict,
        "calibration_status": verdict["calibration_status"],
    }


def run_calibration(seeds: Iterable[int], config: GateConfig | None = None) -> dict:
    validate_phase("calibration")
    checked = validate_calibration_seeds(seeds)
    started = time.time()
    rows = [run_seed(seed, config) for seed in checked]
    statuses = [row["calibration_status"] for row in rows]
    if any(status == "UNDEFINED" for status in statuses):
        aggregate_status = "UNDEFINED"
    elif all(status == "CALIBRATION_PROMISING" for status in statuses):
        aggregate_status = "CALIBRATION_PROMISING"
    else:
        aggregate_status = "CALIBRATION_NEEDS_REVISION"
    return {
        "gate": "replay_cortical_consolidation_v4",
        "phase": "calibration",
        "scientific_partition": True,
        "calibration_status": aggregate_status,
        "seeds": list(checked),
        "reserved_seeds_inspected": False,
        "rows": rows,
        "remaining_scaffolds": [
            "host-defined wake episode populations and partial probe cues",
            "fixed cortical relay, teacher, and inhibitory channel membership",
            "host-scheduled sleep boundaries and episode-agnostic CA3 background current",
            "host measurement against known calibration assemblies",
            "rate-window Hebbian plasticity and fixed assembly anatomy",
            "single coincidence subunit per point neuron",
        ],
        "elapsed_seconds": time.time() - started,
    }


def run_smoke(config: GateConfig | None = None) -> dict:
    """Exercise v4 construction and dynamics without computing a verdict."""
    seed = validate_smoke_seed(SMOKE_SEED)
    cfg = config or smoke_config()
    started = time.time()
    conditions = {
        condition: run_condition(seed, condition, cfg, smoke=True)
        for condition in CONDITIONS
    }
    intact = conditions["intact"]["sleep"]
    lesion = conditions["target_plateau_lesion"]["sleep"]
    return {
        "gate": "replay_cortical_consolidation_v4",
        "phase": "smoke",
        "seed": seed,
        "seed_partition": "smoke",
        "scientific_partition": False,
        "calibration_verdict_computed": False,
        "conditions": conditions,
        "structural_checks": {
            "all_conditions_executed": set(conditions) == set(CONDITIONS),
            "fixed_phase_sequence": all(
                row["phase_trace"] == ["encode_A", "encode_B", "sleep", "retest"]
                for row in conditions.values()
            ),
            "single_bridge_persisted": all(
                row["single_bridge_persisted"] for row in conditions.values()
            ),
            "no_scientific_seed_used": all(
                row["seed_partition"] == "smoke"
                and row["scientific_partition"] is False
                for row in conditions.values()
            ),
            "target_route_added": (
                intact["coincidence_routes_during_sleep"]["target_route_enabled"]
                == intact["coincidence_routes_during_sleep"]["target_route_total"]
            ),
            "target_lesion_preserves_ampa": (
                lesion["coincidence_routes_during_sleep"]["target_route_enabled"] == 0
                and lesion["index_output_transmission_gain_during_sleep"] == 1.0
            ),
            "target_lesion_preserves_ca1_index_plateau": (
                lesion["coincidence_routes_during_sleep"]["ca1_index_route_enabled"]
                == lesion["coincidence_routes_during_sleep"]["ca1_index_route_total"]
            ),
        },
        "dynamics_checks": {
            "intact_target_plateau_nonzero": (
                intact["target_plateau_peak_overall"] > 0.0
                and intact["target_plateau_area_overall"] > 0.0
            ),
            "intact_target_recruited": intact["spikes"]["cortical_target"] > 0,
            "intact_target_fs_recruited": intact["spikes"]["cortical_target_fs"] > 0,
            "lesion_preserves_index_activity": _preserved(
                intact["spikes"]["cortical_index"],
                lesion["spikes"]["cortical_index"],
            ),
            "lesion_preserves_index_fs_activity": _preserved(
                intact["spikes"]["cortical_index_fs"],
                lesion["spikes"]["cortical_index_fs"],
            ),
            "lesion_reduces_target_activity": _materially_reduced(
                intact["spikes"]["cortical_target"],
                lesion["spikes"]["cortical_target"],
            ),
            "lesion_reduces_target_fs_activity": _materially_reduced(
                intact["spikes"]["cortical_target_fs"],
                lesion["spikes"]["cortical_target_fs"],
            ),
        },
        "elapsed_seconds": time.time() - started,
    }


def resolve_cli_request(
    *,
    smoke: bool,
    phase: str | None,
    seeds: Iterable[int] | None,
) -> tuple[str, tuple[int, ...]]:
    requested_phase = phase or ("smoke" if smoke else "calibration")
    supplied = None if seeds is None else tuple(int(seed) for seed in seeds)
    if smoke:
        if requested_phase != "smoke":
            raise ValueError("--smoke cannot be combined with --phase calibration")
        checked = (SMOKE_SEED,) if supplied is None else supplied
        if checked != (SMOKE_SEED,):
            raise ValueError(f"--smoke accepts --seeds {SMOKE_SEED} only")
        return "smoke", checked
    if requested_phase == "smoke":
        raise ValueError("--phase smoke requires --smoke")
    validate_phase(requested_phase)
    checked = CALIBRATION_SEEDS if supplied is None else supplied
    return "calibration", validate_calibration_seeds(checked)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("calibration", "smoke"), default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    try:
        requested_phase, seeds = resolve_cli_request(
            smoke=args.smoke,
            phase=args.phase,
            seeds=args.seeds,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if requested_phase == "smoke":
        payload = run_smoke(smoke_config())
    else:
        payload = run_calibration(seeds, GateConfig())
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
