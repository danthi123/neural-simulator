"""Preregistered Gate B v12 motor-disinhibited boundary construction.

Construction seed 997 is the only executable seed. Engagement seed 2 remains
sealed until a cross-backend construction GO and a separate causal runner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np

from research.runners._vocal_action_credit_gate_v10_policy import (
    POLICY_GATE,
    _all_policy_indices,
    _core_updates,
    _hash_array,
    _policy_routes,
    _weights,
    structural_audit as v10_structural_audit,
)
from research.runners._vocal_action_credit_gate_v11_boundary import (
    _local_synapse_coordinates,
    _pathway,
    _region,
    _step,
    _step_hash,
)
from research.runners._vocal_action_selector_gate import (
    CHANNELS,
    _indices,
    _set_equal_tonic_current,
    build_selector_bridge,
    selector_config,
)
from sim.backend import get_backend, to_host


CONSTRUCTION_SEED = 997
RESERVED_SEED = 2
WARMUP_STEPS = 1000
CATCH_STEPS = 600
ACTION_STEPS = 600
RECOVERY_STEPS = 3000
QUIET_TAIL_STEPS = 1000
BASELINE_PRACTICE_PA = 250.0
ACTION_PRACTICE_PA = 1000.0
MOTOR_THRESHOLD = 12
CLEAN_LOSER_RATIO = 0.25
GUARD_BIN_STEPS = 100
GUARD_SUPPRESSION_RATIO_MAX = 0.50

BACKGROUND_GATE = "boundary_background"
DISINHIBITOR_DRIVE_GATE = "boundary_disinhibitor_drive"
DISINHIBITION_GATE = "boundary_disinhibition"
GUARD_GATE = "boundary_guard"
MOTOR_COPY_GATE = "boundary_motor_copy"
PROPOSAL_STOP_GATE = "boundary_proposal_stop"
COMMIT_STOP_GATE = "boundary_commit_stop"
BOUNDARY_GATES = (
    BACKGROUND_GATE,
    DISINHIBITOR_DRIVE_GATE,
    DISINHIBITION_GATE,
    GUARD_GATE,
    MOTOR_COPY_GATE,
    PROPOSAL_STOP_GATE,
    COMMIT_STOP_GATE,
)

STOP_REGIONS = (
    "proposal_stop_fs_0",
    "proposal_stop_fs_1",
    "commit_stop_fs_0",
    "commit_stop_fs_1",
)
QUIET_BOUNDARY_REGIONS = ("boundary_vip", "action_corollary", *STOP_REGIONS)
BOUNDARY_REGIONS = (
    "action_corollary",
    "boundary_guard_som",
    "boundary_vip",
    *STOP_REGIONS,
)
NEW_REGION_SET = frozenset(BOUNDARY_REGIONS)

HOST_BOUNDARY = {
    "host_selects_action": False,
    "host_argmax_or_fallback": False,
    "host_forces_action": False,
    "host_reads_threshold_to_change_neural_state": False,
    "host_drives_new_boundary_population": False,
    "host_drives_selector_reset": False,
    "host_clears_state_between_phases": False,
    "reward_or_weight_update": False,
    "engagement_execution_open": False,
}


def validate_construction_seed(seed: int) -> int:
    checked = int(seed)
    if checked != CONSTRUCTION_SEED:
        raise ValueError(
            f"V12 construction accepts seed {CONSTRUCTION_SEED} only; "
            f"reserved seed {RESERVED_SEED} remains sealed"
        )
    return checked


def validate_engagement_seed(seed: int):
    raise ValueError(
        "Gate B v12 engagement execution is sealed until cross-backend "
        f"construction qualifies and its causal runner is committed; received={int(seed)}"
    )


def _boundary_topology():
    regions = [
        _region("action_corollary", 24, excitatory=True),
        _region("boundary_guard_som", 24, excitatory=False),
        _region("boundary_vip", 16, excitatory=False),
    ]
    for channel in CHANNELS:
        regions.extend([
            _region(f"proposal_stop_fs_{channel}", 16, excitatory=False),
            _region(f"commit_stop_fs_{channel}", 16, excitatory=False),
        ])

    pathways = [
        _pathway(
            "practice_arousal", "boundary_guard_som",
            density=1.0, weight=8.0, gate=BACKGROUND_GATE,
        ),
        _pathway(
            "boundary_vip", "boundary_guard_som",
            density=1.0, weight=8.0, gate=DISINHIBITION_GATE,
            receptor="gaba_a",
        ),
        _pathway(
            "boundary_guard_som", "action_corollary",
            density=1.0, weight=8.0, gate=GUARD_GATE,
            receptor="gaba_a",
        ),
    ]
    for channel in CHANNELS:
        pathways.extend([
            _pathway(
                f"motor_{channel}", "boundary_vip",
                density=1.0, weight=30.0,
                gate=DISINHIBITOR_DRIVE_GATE,
            ),
            _pathway(
                f"motor_{channel}", "action_corollary",
                density=1.0, weight=30.0, gate=MOTOR_COPY_GATE,
            ),
            _pathway(
                "action_corollary", f"proposal_stop_fs_{channel}",
                density=1.0, weight=30.0, gate=PROPOSAL_STOP_GATE,
            ),
            _pathway(
                "action_corollary", f"commit_stop_fs_{channel}",
                density=1.0, weight=30.0, gate=COMMIT_STOP_GATE,
            ),
            _pathway(
                f"proposal_stop_fs_{channel}", f"proposal_{channel}",
                density=1.0, weight=16.0, receptor="gaba_a",
            ),
            _pathway(
                f"commit_stop_fs_{channel}", f"commit_{channel}",
                density=1.0, weight=16.0, receptor="gaba_a",
            ),
            _pathway(
                f"commit_stop_fs_{channel}", f"motor_{channel}",
                density=1.0, weight=16.0, receptor="gaba_a",
            ),
        ])
    return regions, pathways


def build_v12_bridge(
    seed: int,
    *,
    coactivity: bool = True,
    gate_values: dict[str, float] | None = None,
):
    if int(seed) not in (CONSTRUCTION_SEED, RESERVED_SEED):
        raise ValueError(f"unassigned V12 seed: {int(seed)}")
    extra_regions, extra_pathways = _boundary_topology()
    updates = _core_updates(coactivity=coactivity)
    updates.update({"enable_nmda": False, "enable_gabab": False})
    bridge = build_selector_bridge(
        int(seed),
        selector_config("v2"),
        policy_plasticity_gate=POLICY_GATE,
        commit_enable_nmda=False,
        extra_regions=extra_regions,
        extra_pathways=extra_pathways,
        core_config_updates=updates,
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
    values = {gate: 1.0 for gate in BOUNDARY_GATES}
    values.update(gate_values or {})
    for gate, value in values.items():
        bridge.set_transmission_gate(gate, float(value))
    return bridge, {
        "policy_routes": routes,
        "all_policy": all_policy,
        "expected_coactivity": bool(coactivity),
        "boundary_indices": {
            name: _indices(bridge, name) for name in BOUNDARY_REGIONS
        },
        "gate_values": values,
    }


def _expected_new_pathways() -> dict[tuple[str, str], tuple]:
    expected = {
        ("practice_arousal", "boundary_guard_som"):
            (1.0, 8.0, 0.0, BACKGROUND_GATE, "ampa", "gaba_a"),
        ("boundary_vip", "boundary_guard_som"):
            (1.0, 8.0, 0.0, DISINHIBITION_GATE, "ampa", "gaba_a"),
        ("boundary_guard_som", "action_corollary"):
            (1.0, 8.0, 0.0, GUARD_GATE, "ampa", "gaba_a"),
    }
    for channel in CHANNELS:
        expected.update({
            (f"motor_{channel}", "boundary_vip"):
                (1.0, 30.0, 0.0, DISINHIBITOR_DRIVE_GATE, "ampa", "gaba_a"),
            (f"motor_{channel}", "action_corollary"):
                (1.0, 30.0, 0.0, MOTOR_COPY_GATE, "ampa", "gaba_a"),
            ("action_corollary", f"proposal_stop_fs_{channel}"):
                (1.0, 30.0, 0.0, PROPOSAL_STOP_GATE, "ampa", "gaba_a"),
            ("action_corollary", f"commit_stop_fs_{channel}"):
                (1.0, 30.0, 0.0, COMMIT_STOP_GATE, "ampa", "gaba_a"),
            (f"proposal_stop_fs_{channel}", f"proposal_{channel}"):
                (1.0, 16.0, 0.0, None, "ampa", "gaba_a"),
            (f"commit_stop_fs_{channel}", f"commit_{channel}"):
                (1.0, 16.0, 0.0, None, "ampa", "gaba_a"),
            (f"commit_stop_fs_{channel}", f"motor_{channel}"):
                (1.0, 16.0, 0.0, None, "ampa", "gaba_a"),
        })
    return expected


def _pathway_signature(pathway) -> tuple:
    return (
        float(pathway.density),
        float(pathway.weight_mean),
        float(pathway.weight_jitter),
        pathway.transmission_gate,
        getattr(pathway, "exc_receptor", "ampa"),
        getattr(pathway, "receptor", "gaba_a"),
    )


def boundary_structural_audit(bridge, handles: dict) -> dict[str, object]:
    regions = {region.name: region for region in bridge.core_config.brain_regions}
    pathways = list(bridge.core_config.region_pathways)
    pairs = [(path.from_region, path.to_region) for path in pathways]
    new_pathways = {
        (path.from_region, path.to_region): path
        for path in pathways
        if path.from_region in NEW_REGION_SET or path.to_region in NEW_REGION_SET
    }
    expected = _expected_new_pathways()

    symmetry_pairs = [
        (("motor_0", "boundary_vip"), ("motor_1", "boundary_vip")),
        (("motor_0", "action_corollary"), ("motor_1", "action_corollary")),
        (("action_corollary", "proposal_stop_fs_0"),
         ("action_corollary", "proposal_stop_fs_1")),
        (("action_corollary", "commit_stop_fs_0"),
         ("action_corollary", "commit_stop_fs_1")),
        (("proposal_stop_fs_0", "proposal_0"),
         ("proposal_stop_fs_1", "proposal_1")),
        (("commit_stop_fs_0", "commit_0"),
         ("commit_stop_fs_1", "commit_1")),
        (("commit_stop_fs_0", "motor_0"),
         ("commit_stop_fs_1", "motor_1")),
    ]
    symmetric_coordinates = all(
        _local_synapse_coordinates(bridge, *left)
        == _local_synapse_coordinates(bridge, *right)
        for left, right in symmetry_pairs
    )

    v10_audit = v10_structural_audit(bridge, handles)
    inherited = {
        name: value for name, value in v10_audit["checks"].items()
        if name not in (
            "selector_v2_has_600_neurons",
            "selector_has_36_declared_pathways",
        )
    }
    inhibitory_paths = [
        path for path in new_pathways.values()
        if regions[path.from_region].exc_fraction == 0.0
    ]
    excitatory_paths = [
        path for path in new_pathways.values()
        if regions[path.from_region].exc_fraction == 1.0
    ]
    nmda_regions = sorted(
        name for name, region in regions.items() if region.enable_nmda
    )
    xp, _ = get_backend()
    new_current = np.concatenate([
        np.asarray(to_host(bridge.cp_external_input_current[
            xp.asarray(_indices(bridge, name))
        ]), dtype=np.float64)
        for name in BOUNDARY_REGIONS
    ])
    checks = {
        "v10_policy_structure": bool(all(inherited.values())),
        "v12_has_728_neurons": int(bridge.core_config.num_neurons) == 728,
        "v12_has_53_declared_pathways": len(pathways) == 53,
        "unique_ordered_region_pairs": len(pairs) == len(set(pairs)),
        "exact_seventeen_new_pathways": set(new_pathways) == set(expected),
        "new_pathway_declarations_are_locked": (
            set(new_pathways) == set(expected)
            and all(
                _pathway_signature(new_pathways[pair]) == signature
                for pair, signature in expected.items()
            )
        ),
        "new_inhibitory_sources_use_gaba_a": all(
            getattr(path, "receptor", "gaba_a") == "gaba_a"
            for path in inhibitory_paths
        ),
        "new_excitatory_sources_use_ampa": all(
            getattr(path, "exc_receptor", "ampa") == "ampa"
            for path in excitatory_paths
        ),
        "new_channel_routes_are_exactly_symmetric": symmetric_coordinates,
        "no_nmda_region_or_substrate": (
            not bridge.core_config.enable_nmda and nmda_regions == []
        ),
        "no_gabab_substrate": not bridge.core_config.enable_gabab,
        "no_boundary_route_targets_bg_or_reset": not any(
            path.from_region in NEW_REGION_SET
            and (
                path.to_region.startswith(("str_d1_", "str_d2_", "gpi_", "thal_"))
                or path.to_region == "selector_reset"
            )
            for path in pathways
        ),
        "new_populations_start_without_external_current": bool(
            np.all(new_current == 0.0)
        ),
        "host_boundary_is_closed": not any(HOST_BOUNDARY.values()),
    }
    return {
        "checks": checks,
        "pass": bool(all(checks.values())),
        "nmda_regions": nmda_regions,
        "new_pathway_pairs": sorted([list(pair) for pair in new_pathways]),
        "v10": v10_audit,
    }


def _set_drive(bridge, practice_pA: float):
    xp, _ = get_backend()
    _set_equal_tonic_current(bridge, selector_config("v2"))
    bridge.cp_external_input_current[
        xp.asarray(_indices(bridge, "practice_arousal"))
    ] = xp.float32(practice_pA)
    new_max = max(
        float(np.max(np.abs(np.asarray(to_host(
            bridge.cp_external_input_current[xp.asarray(_indices(bridge, name))]
        ), dtype=np.float64)), initial=0.0))
        for name in BOUNDARY_REGIONS
    )
    reset_max = float(np.max(np.abs(np.asarray(to_host(
        bridge.cp_external_input_current[
            xp.asarray(_indices(bridge, "selector_reset"))
        ]
    ), dtype=np.float64)), initial=0.0))
    if new_max != 0.0 or reset_max != 0.0:
        raise AssertionError("V12 new-boundary and selector-reset current must remain zero")
    return new_max, reset_max


def _region_names() -> list[str]:
    names = [
        "practice_arousal", "selector_reset", "action_corollary",
        "boundary_guard_som", "boundary_vip", *STOP_REGIONS,
    ]
    for stem in ("proposal", "str_d1", "str_d2", "commit", "motor"):
        names.extend(f"{stem}_{channel}" for channel in CHANNELS)
    return names


def _run_epoch(bridge, *, steps: int, practice_pA: float) -> dict[str, object]:
    new_current_max, reset_current_max = _set_drive(bridge, practice_pA)
    names = _region_names()
    indices = {name: _indices(bridge, name) for name in names}
    region_steps = {name: [] for name in names}
    g_i_max = {name: 0.0 for name in names}
    hashes = []
    motor_counts = np.zeros(2, dtype=np.int64)
    first_motor_spike = None
    first_vip_spike = None
    first_corollary_spike = None
    first_stop_spike = None
    first_crossing = None
    decision_step = None
    decision_counts = None
    simultaneous = False

    for step in range(int(steps)):
        firing = _step(bridge)
        hashes.append(_step_hash(firing))
        g_i = np.asarray(to_host(bridge.cp_conductance_g_i), dtype=np.float64)
        for name in names:
            region_steps[name].append(int(firing[indices[name]].sum()))
            g_i_max[name] = max(
                g_i_max[name],
                float(np.max(np.abs(g_i[indices[name]]), initial=0.0)),
            )
        if first_motor_spike is None and any(
            region_steps[f"motor_{channel}"][-1] > 0 for channel in CHANNELS
        ):
            first_motor_spike = int(step)
        if first_vip_spike is None and region_steps["boundary_vip"][-1] > 0:
            first_vip_spike = int(step)
        if (
            first_corollary_spike is None
            and region_steps["action_corollary"][-1] > 0
        ):
            first_corollary_spike = int(step)
        if first_stop_spike is None and any(
            region_steps[name][-1] > 0 for name in STOP_REGIONS
        ):
            first_stop_spike = int(step)

        previous = motor_counts.copy()
        for channel in CHANNELS:
            motor_counts[channel] += region_steps[f"motor_{channel}"][-1]
        crossed = [
            channel for channel in CHANNELS
            if previous[channel] < MOTOR_THRESHOLD <= motor_counts[channel]
        ]
        if decision_step is None and crossed:
            decision_step = int(step)
            decision_counts = motor_counts.copy()
            if len(crossed) == 1:
                first_crossing = int(crossed[0])
            else:
                simultaneous = True

    winner = None
    loser_ratio = None
    cleanliness_reason = "no_unique_crossing"
    if first_crossing is not None and not simultaneous:
        loser = 1 - first_crossing
        loser_ratio = float(
            decision_counts[loser] / max(1, decision_counts[first_crossing])
        )
        if loser_ratio > CLEAN_LOSER_RATIO:
            cleanliness_reason = "loser_ratio_at_crossing"
        elif motor_counts[loser] >= MOTOR_THRESHOLD:
            cleanliness_reason = "loser_crossed_later"
        else:
            winner = int(first_crossing)
            cleanliness_reason = "clean"

    guard = region_steps["boundary_guard_som"]
    guard_pre_rate = None
    guard_post_rate = None
    guard_ratio = None
    if first_vip_spike is not None:
        pre = guard[max(0, first_vip_spike - 100):first_vip_spike]
        post = guard[first_vip_spike + 1:min(steps, first_vip_spike + 101)]
        if len(pre) == 100 and len(post) == 100:
            guard_pre_rate = float(sum(pre) / (100 * 24))
            guard_post_rate = float(sum(post) / (100 * 24))
            if guard_pre_rate > 0.0:
                guard_ratio = float(guard_post_rate / guard_pre_rate)

    return {
        "steps": int(steps),
        "practice_pA": float(practice_pA),
        "winner": winner,
        "first_crossing": first_crossing,
        "decision_step": decision_step,
        "decision_motor_counts": (
            decision_counts.tolist() if decision_counts is not None else None
        ),
        "motor_counts": motor_counts.tolist(),
        "loser_ratio": loser_ratio,
        "cleanliness_reason": cleanliness_reason,
        "first_motor_spike": first_motor_spike,
        "first_vip_spike": first_vip_spike,
        "first_corollary_spike": first_corollary_spike,
        "first_stop_spike": first_stop_spike,
        "guard_pre_vip_rate": guard_pre_rate,
        "guard_post_vip_rate": guard_post_rate,
        "guard_suppression_ratio": guard_ratio,
        "region_spikes": {
            name: int(sum(values)) for name, values in region_steps.items()
        },
        "region_step_counts": region_steps,
        "max_gaba_a_conductance_by_region": g_i_max,
        "new_boundary_external_current_max": float(new_current_max),
        "selector_reset_current_max": float(reset_current_max),
        "step_firing_hashes": hashes,
    }


def _guard_bins_active(row: dict, *, tail_steps: int | None = None) -> bool:
    values = row["region_step_counts"]["boundary_guard_som"]
    if tail_steps is not None:
        values = values[-int(tail_steps):]
    return bool(values) and len(values) % GUARD_BIN_STEPS == 0 and all(
        sum(values[start:start + GUARD_BIN_STEPS]) > 0
        for start in range(0, len(values), GUARD_BIN_STEPS)
    )


def _boundary_quiet(row: dict, *, tail_steps: int | None = None) -> bool:
    values = row["region_step_counts"]
    start = -int(tail_steps) if tail_steps is not None else 0
    return all(sum(values[name][start:]) == 0 for name in QUIET_BOUNDARY_REGIONS)


def _action_triggered(row: dict) -> bool:
    times = (
        row["first_motor_spike"], row["first_vip_spike"],
        row["first_corollary_spike"], row["first_stop_spike"],
    )
    if any(value is None for value in times):
        return False
    motor, vip, corollary, stop = times
    all_stops = all(
        row["region_spikes"][name] > 0 for name in STOP_REGIONS
    )
    return bool(
        motor < vip
        and motor < corollary
        and vip < stop
        and corollary < stop
        and row["guard_suppression_ratio"] is not None
        and row["guard_suppression_ratio"] <= GUARD_SUPPRESSION_RATIO_MAX
        and all_stops
    )


def _combined_hashes(warmup: dict, action: dict) -> list[str]:
    return list(warmup["step_firing_hashes"]) + list(action["step_firing_hashes"])


def _first_source_step(warmup: dict, action: dict, sources: tuple[str, ...]):
    combined = {
        name: (
            warmup["region_step_counts"][name]
            + action["region_step_counts"][name]
        )
        for name in sources
    }
    for step in range(WARMUP_STEPS + ACTION_STEPS):
        if any(combined[name][step] > 0 for name in sources):
            return int(step)
    return None


def _target_total_after(
    warmup: dict,
    action: dict,
    targets: tuple[str, ...],
    start: int,
) -> int:
    return int(sum(
        sum((
            warmup["region_step_counts"][name]
            + action["region_step_counts"][name]
        )[start:])
        for name in targets
    ))


def _max_target_gaba(
    warmup: dict, action: dict, targets: tuple[str, ...]
) -> float:
    if not targets:
        raise ValueError("inhibitory audit requires at least one target")
    return float(max(
        max(
            warmup["max_gaba_a_conductance_by_region"][name],
            action["max_gaba_a_conductance_by_region"][name],
        )
        for name in targets
    ))


INHIBITORY_AUDIT_SPECS = (
    ("guard", GUARD_GATE, ("boundary_guard_som",), ("action_corollary",)),
    ("disinhibition", DISINHIBITION_GATE, ("boundary_vip",), ("boundary_guard_som",)),
    ("proposal_stop", PROPOSAL_STOP_GATE, ("proposal_stop_fs_0", "proposal_stop_fs_1"), ("proposal_0", "proposal_1")),
    ("commit_stop", COMMIT_STOP_GATE, ("commit_stop_fs_0", "commit_stop_fs_1"), ("commit_0", "commit_1", "motor_0", "motor_1")),
)


def run_inhibitory_audit(*, seed: int, spec: tuple) -> dict[str, object]:
    validate_construction_seed(seed)
    name, gate, sources, targets = spec
    rows = {}
    for arm, gate_value in (("source_on", 1.0), ("source_off", 0.0)):
        bridge, _ = build_v12_bridge(seed, gate_values={gate: gate_value})
        initial_weights = _weights(bridge)
        warmup = _run_epoch(
            bridge, steps=WARMUP_STEPS, practice_pA=BASELINE_PRACTICE_PA
        )
        action = _run_epoch(
            bridge, steps=ACTION_STEPS, practice_pA=ACTION_PRACTICE_PA
        )
        final_weights = _weights(bridge)
        rows[arm] = {
            "warmup": warmup,
            "action": action,
            "initial_weight_hash": _hash_array(initial_weights),
            "final_weight_hash": _hash_array(final_weights),
            "weights_are_byte_identical": bool(
                np.array_equal(initial_weights, final_weights)
            ),
        }

    on = rows["source_on"]
    off = rows["source_off"]
    source_step = _first_source_step(on["warmup"], on["action"], sources)
    start = (source_step + 1) if source_step is not None else 0
    on_total = _target_total_after(on["warmup"], on["action"], targets, start)
    off_total = _target_total_after(off["warmup"], off["action"], targets, start)
    on_hashes = _combined_hashes(on["warmup"], on["action"])
    off_hashes = _combined_hashes(off["warmup"], off["action"])
    first_divergence = next((
        step for step, (left, right) in enumerate(zip(on_hashes, off_hashes))
        if left != right
    ), None)
    prefix_end = source_step if source_step is not None else 0
    prefix_match = on_hashes[:prefix_end] == off_hashes[:prefix_end]
    max_gaba = _max_target_gaba(on["warmup"], on["action"], targets)
    checks = {
        "source_event_exists": source_step is not None,
        "precausal_firing_prefix_matches": bool(prefix_match),
        "target_gaba_a_conductance_is_nonzero": max_gaba > 0.0,
        "source_on_does_not_increase_target_firing": on_total <= off_total,
        "source_on_and_off_start_from_identical_weights": (
            on["initial_weight_hash"] == off["initial_weight_hash"]
        ),
        "source_on_and_off_weights_remain_identical": (
            on["weights_are_byte_identical"]
            and off["weights_are_byte_identical"]
        ),
    }
    return {
        "name": name,
        "gate": gate,
        "sources": list(sources),
        "targets": list(targets),
        "first_source_spike": source_step,
        "first_firing_divergence": first_divergence,
        "source_on_target_spikes": on_total,
        "source_off_target_spikes": off_total,
        "source_on_max_target_gaba_a": max_gaba,
        "checks": checks,
        "pass": bool(all(checks.values())),
        "arms": rows,
    }


def _backend_info() -> dict[str, object]:
    xp, backend = get_backend()
    info = {"backend": backend, "host": platform.node()}
    if backend == "cupy":
        props = xp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        info["device"] = str(name)
    return info


def run_construction(seed: int = CONSTRUCTION_SEED) -> dict[str, object]:
    validate_construction_seed(seed)
    started = time.perf_counter()
    bridge, handles = build_v12_bridge(seed)
    initial_weights = _weights(bridge)
    audit = boundary_structural_audit(bridge, handles)
    warmup = _run_epoch(
        bridge, steps=WARMUP_STEPS, practice_pA=BASELINE_PRACTICE_PA
    )
    catch = _run_epoch(
        bridge, steps=CATCH_STEPS, practice_pA=BASELINE_PRACTICE_PA
    )
    recovery_0 = _run_epoch(
        bridge, steps=RECOVERY_STEPS, practice_pA=BASELINE_PRACTICE_PA
    )
    action_1 = _run_epoch(
        bridge, steps=ACTION_STEPS, practice_pA=ACTION_PRACTICE_PA
    )
    recovery_1 = _run_epoch(
        bridge, steps=RECOVERY_STEPS, practice_pA=BASELINE_PRACTICE_PA
    )
    action_2 = _run_epoch(
        bridge, steps=ACTION_STEPS, practice_pA=ACTION_PRACTICE_PA
    )
    recovery_2 = _run_epoch(
        bridge, steps=RECOVERY_STEPS, practice_pA=BASELINE_PRACTICE_PA
    )
    final_weights = _weights(bridge)
    inhibitory_audits = [
        run_inhibitory_audit(seed=seed, spec=spec)
        for spec in INHIBITORY_AUDIT_SPECS
    ]

    checks = {
        "structure": audit["pass"],
        "warmup_boundary_is_quiet": _boundary_quiet(warmup),
        "warmup_guard_is_continuously_active": _guard_bins_active(warmup),
        "warmup_motor_stays_subthreshold": all(
            count < MOTOR_THRESHOLD for count in warmup["motor_counts"]
        ),
        "catch_boundary_is_quiet": _boundary_quiet(catch),
        "catch_guard_is_continuously_active": _guard_bins_active(catch),
        "catch_motor_stays_subthreshold": all(
            count < MOTOR_THRESHOLD for count in catch["motor_counts"]
        ),
        "catch_recovery_is_quiet": _boundary_quiet(
            recovery_0, tail_steps=QUIET_TAIL_STEPS
        ),
        "catch_recovery_guard_is_active": _guard_bins_active(recovery_0),
        "first_action_is_motor_triggered": _action_triggered(action_1),
        "first_action_is_clean": action_1["winner"] in CHANNELS,
        "first_recovery_is_quiet": _boundary_quiet(
            recovery_1, tail_steps=QUIET_TAIL_STEPS
        ),
        "first_recovery_guard_is_active": _guard_bins_active(recovery_1),
        "second_action_is_motor_triggered": _action_triggered(action_2),
        "second_action_is_clean": action_2["winner"] in CHANNELS,
        "second_recovery_is_quiet": _boundary_quiet(
            recovery_2, tail_steps=QUIET_TAIL_STEPS
        ),
        "second_recovery_guard_is_active": _guard_bins_active(recovery_2),
        "inhibitory_source_on_off_audits_pass": all(
            row["pass"] for row in inhibitory_audits
        ),
        "new_boundary_current_is_always_zero": all(
            row["new_boundary_external_current_max"] == 0.0
            for row in (
                warmup, catch, recovery_0, action_1,
                recovery_1, action_2, recovery_2,
            )
        ),
        "selector_reset_current_is_always_zero": all(
            row["selector_reset_current_max"] == 0.0
            for row in (
                warmup, catch, recovery_0, action_1,
                recovery_1, action_2, recovery_2,
            )
        ),
        "weights_are_byte_identical": bool(
            np.array_equal(initial_weights, final_weights)
        ),
    }
    clean_keys = ("first_action_is_clean", "second_action_is_clean")
    nonclean_pass = all(value for key, value in checks.items() if key not in clean_keys)
    too_short = (
        nonclean_pass
        and all(
            row["cleanliness_reason"] in ("clean", "loser_crossed_later")
            for row in (action_1, action_2)
        )
        and not all(checks[key] for key in clean_keys)
    )
    if all(checks.values()):
        outcome = "CONSTRUCTION_GO"
    elif too_short:
        outcome = "CONSTRUCTION_QUALIFIED_BOUNDARY_TOO_SHORT"
    else:
        outcome = "CONSTRUCTION_NO_GO"

    return {
        "probe": "vocal_action_credit_gate_v12_disinhibitory_construction",
        "stage": "construction_backend",
        "seed": int(seed),
        "reserved_seed": RESERVED_SEED,
        "engagement_execution_open": False,
        "outcome": outcome,
        "construction_go": outcome == "CONSTRUCTION_GO",
        "checks": checks,
        "audit": audit,
        "initial_weight_hash": _hash_array(initial_weights),
        "final_weight_hash": _hash_array(final_weights),
        "warmup": warmup,
        "catch": catch,
        "recovery_after_catch": recovery_0,
        "action_1": action_1,
        "recovery_1": recovery_1,
        "action_2": action_2,
        "recovery_2": recovery_2,
        "inhibitory_audits": inhibitory_audits,
        "topology": {
            "neurons": int(bridge.core_config.num_neurons),
            "synapses": int(bridge.cp_connections.nnz),
            "regions": len(bridge.core_config.brain_regions),
            "pathways": len(bridge.core_config.region_pathways),
        },
        "backend_info": _backend_info(),
        "host_boundary": HOST_BOUNDARY,
        "config": {
            "warmup_steps": WARMUP_STEPS,
            "catch_steps": CATCH_STEPS,
            "action_steps": ACTION_STEPS,
            "recovery_steps": RECOVERY_STEPS,
            "baseline_practice_pA": BASELINE_PRACTICE_PA,
            "action_practice_pA": ACTION_PRACTICE_PA,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def merge_construction_artifacts(numpy_path: Path, cupy_path: Path):
    artifacts = [json.loads(path.read_text()) for path in (numpy_path, cupy_path)]
    by_backend = {item["backend_info"]["backend"]: item for item in artifacts}
    if set(by_backend) != {"numpy", "cupy"}:
        raise ValueError(f"expected numpy and cupy artifacts, got {sorted(by_backend)}")
    for item in artifacts:
        if item["seed"] != CONSTRUCTION_SEED:
            raise ValueError("construction artifact consumed an unassigned seed")
        if item["reserved_seed"] != RESERVED_SEED:
            raise ValueError("construction artifact changed the reserved seed")
        if item["engagement_execution_open"]:
            raise ValueError("construction artifact illegally opened engagement")
    outcomes = {backend: item["outcome"] for backend, item in by_backend.items()}
    if all(value == "CONSTRUCTION_GO" for value in outcomes.values()):
        outcome = "CONSTRUCTION_GO"
    elif all(
        value == "CONSTRUCTION_QUALIFIED_BOUNDARY_TOO_SHORT"
        for value in outcomes.values()
    ):
        outcome = "CONSTRUCTION_QUALIFIED_BOUNDARY_TOO_SHORT"
    else:
        outcome = "CONSTRUCTION_NO_GO"
    return {
        "probe": "vocal_action_credit_gate_v12_disinhibitory_construction",
        "stage": "construction_cross_backend",
        "seed": CONSTRUCTION_SEED,
        "reserved_seed": RESERVED_SEED,
        "engagement_execution_open": False,
        "backend_artifacts": {
            item["backend_info"]["backend"]: str(path)
            for item, path in zip(artifacts, (numpy_path, cupy_path))
        },
        "backend_outcomes": outcomes,
        "outcome": outcome,
        "construction_go": outcome == "CONSTRUCTION_GO",
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=CONSTRUCTION_SEED)
    parser.add_argument("--construction-only", action="store_true")
    parser.add_argument("--merge-construction", nargs=2, metavar=("NUMPY", "CUPY"))
    parser.add_argument(
        "--out",
        default="research/findings/raw/vocal_action_credit_gate_v12/construction.json",
    )
    args = parser.parse_args(argv)
    if args.merge_construction:
        result = merge_construction_artifacts(
            Path(args.merge_construction[0]), Path(args.merge_construction[1])
        )
    elif args.construction_only:
        result = run_construction(args.seed)
    else:
        validate_engagement_seed(args.seed)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "stage": result["stage"],
        "seed": result["seed"],
        "outcome": result["outcome"],
        "output": str(output),
    }, indent=2))
    return 0 if result["construction_go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
