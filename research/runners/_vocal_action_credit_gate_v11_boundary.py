"""Preregistered Gate B v11 neural action-boundary construction.

Construction seed 991 is the only executable seed in this stage. The reserved
engagement seed remains sealed until one locked recurrence weight qualifies on
both NumPy and CuPy and the formal causal runner is committed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
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
from research.runners._vocal_action_selector_gate import (
    CHANNELS,
    _indices,
    _set_equal_tonic_current,
    build_selector_bridge,
    selector_config,
)
from sim.backend import get_backend, to_host
from sim.enums import NeuronType
from sim.regions import BrainRegion, RegionPathway


CONSTRUCTION_SEED = 991
RESERVED_SEED = 1
RECURRENCE_WEIGHTS = (0.25, 0.50, 1.00, 2.00)
WARMUP_STEPS = 80
CATCH_STEPS = 600
ACTION_STEPS = 600
RECOVERY_STEPS = 3000
RECOVERY_QUIET_TAIL_STEPS = 1000
CATCH_PRACTICE_PA = 250.0
ACTION_PRACTICE_PA = 1000.0
MOTOR_THRESHOLD = 12
CLEAN_LOSER_RATIO = 0.25
BOUNDARY_LATE_STEP = 550
BOUNDARY_DECISION_CUTOFF = 450
RECOVERY_NMDA_MAX = 1e-6

MOTOR_COPY_GATE = "boundary_motor_copy"
RECURRENCE_GATE = "boundary_recurrence"
PROPOSAL_STOP_GATE = "boundary_proposal_stop"
COMMIT_STOP_GATE = "boundary_commit_stop"
BOUNDARY_GATES = (
    MOTOR_COPY_GATE,
    RECURRENCE_GATE,
    PROPOSAL_STOP_GATE,
    COMMIT_STOP_GATE,
)

BOUNDARY_REGIONS = (
    "action_corollary",
    "proposal_stop_fs_0",
    "proposal_stop_fs_1",
    "commit_stop_fs_0",
    "commit_stop_fs_1",
)

HOST_BOUNDARY = {
    "host_selects_action": False,
    "host_argmax_or_fallback": False,
    "host_forces_action": False,
    "host_reads_threshold_to_change_neural_state": False,
    "host_drives_selector_reset": False,
    "host_clears_state_between_trials": False,
    "reward_or_weight_update": False,
    "formal_execution_open": False,
}


def validate_construction_seed(seed: int) -> int:
    checked = int(seed)
    if checked != CONSTRUCTION_SEED:
        raise ValueError(
            f"V11 construction accepts seed {CONSTRUCTION_SEED} only; "
            f"reserved seed {RESERVED_SEED} remains sealed"
        )
    return checked


def validate_formal_seed(seed: int):
    raise ValueError(
        "Gate B v11 formal execution is sealed until cross-backend "
        f"construction qualifies; received={int(seed)}"
    )


def _region(
    name: str,
    n_neurons: int,
    *,
    excitatory: bool,
    enable_nmda: bool = False,
) -> BrainRegion:
    neuron_type = (
        NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL
        if excitatory
        else NeuronType.IZH2007_FS_CORTICAL_INTERNEURON
    )
    return BrainRegion(
        name=name,
        n_neurons=int(n_neurons),
        exc_fraction=1.0 if excitatory else 0.0,
        internal_density=0.0,
        exc_weight_mean=0.0,
        inh_weight_mean=0.0,
        weight_jitter=0.0,
        plastic_internal=False,
        izh_neuron_type=neuron_type.name,
        enable_nmda=bool(enable_nmda),
        enable_homeostasis=False,
    )


def _pathway(
    source: str,
    target: str,
    *,
    density: float,
    weight: float,
    gate: str | None = None,
    receptor: str = "ampa",
) -> RegionPathway:
    kwargs = {}
    if receptor != "ampa":
        kwargs["receptor"] = receptor
    return RegionPathway(
        from_region=source,
        to_region=target,
        density=float(density),
        weight_mean=float(weight),
        weight_jitter=0.0,
        plastic=False,
        transmission_gate=gate,
        **kwargs,
    )


def _boundary_topology(recurrence_weight: float):
    regions = [
        _region("action_corollary", 24, excitatory=True, enable_nmda=True),
    ]
    for channel in CHANNELS:
        regions.extend([
            _region(f"proposal_stop_fs_{channel}", 16, excitatory=False),
            _region(f"commit_stop_fs_{channel}", 16, excitatory=False),
        ])

    pathways = []
    for channel in CHANNELS:
        pathways.append(_pathway(
            f"motor_{channel}",
            "action_corollary",
            density=1.0,
            weight=30.0,
            gate=MOTOR_COPY_GATE,
        ))
    pathways.append(_pathway(
        "action_corollary",
        "action_corollary",
        density=0.35,
        weight=float(recurrence_weight),
        gate=RECURRENCE_GATE,
    ))
    for channel in CHANNELS:
        pathways.extend([
            _pathway(
                "action_corollary",
                f"proposal_stop_fs_{channel}",
                density=1.0,
                weight=30.0,
                gate=PROPOSAL_STOP_GATE,
            ),
            _pathway(
                "action_corollary",
                f"commit_stop_fs_{channel}",
                density=1.0,
                weight=30.0,
                gate=COMMIT_STOP_GATE,
            ),
            _pathway(
                f"proposal_stop_fs_{channel}",
                f"proposal_{channel}",
                density=1.0,
                weight=16.0,
                receptor="gaba_a",
            ),
            _pathway(
                f"commit_stop_fs_{channel}",
                f"commit_{channel}",
                density=1.0,
                weight=16.0,
                receptor="gaba_a",
            ),
            _pathway(
                f"commit_stop_fs_{channel}",
                f"motor_{channel}",
                density=1.0,
                weight=16.0,
                receptor="gaba_a",
            ),
        ])
    return regions, pathways


def build_v11_bridge(
    seed: int,
    *,
    recurrence_weight: float,
    coactivity: bool = True,
    gate_values: dict[str, float] | None = None,
):
    if int(seed) not in (CONSTRUCTION_SEED, RESERVED_SEED):
        raise ValueError(f"unassigned V11 seed: {int(seed)}")
    extra_regions, extra_pathways = _boundary_topology(recurrence_weight)
    updates = _core_updates(coactivity=coactivity)
    updates.update({
        "enable_nmda": True,
        "nmda_ratio": 0.4,
        "nmda_tau_decay": 100.0,
        "nmda_tau_rise": 3.0,
        "nmda_mg_concentration": 1.0,
    })
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
        "boundary_indices": {
            name: _indices(bridge, name) for name in BOUNDARY_REGIONS
        },
        "gate_values": values,
        "recurrence_weight": float(recurrence_weight),
        "expected_coactivity": bool(coactivity),
    }


def _local_synapse_coordinates(bridge, source: str, target: str):
    coo = bridge.cp_connections.tocoo(copy=False)
    rows = np.asarray(to_host(coo.row), dtype=np.int64)
    cols = np.asarray(to_host(coo.col), dtype=np.int64)
    source_indices = np.asarray(_indices(bridge, source), dtype=np.int64)
    target_indices = np.asarray(_indices(bridge, target), dtype=np.int64)
    source_local = {int(value): i for i, value in enumerate(source_indices)}
    target_local = {int(value): i for i, value in enumerate(target_indices)}
    mask = np.isin(rows, source_indices) & np.isin(cols, target_indices)
    return sorted(
        (source_local[int(pre)], target_local[int(post)])
        for pre, post in zip(rows[mask], cols[mask])
    )


def boundary_structural_audit(bridge, handles: dict) -> dict[str, object]:
    regions = {region.name: region for region in bridge.core_config.brain_regions}
    pathways = list(bridge.core_config.region_pathways)
    pairs = [(p.from_region, p.to_region) for p in pathways]
    inhibitory_sources_use_gaba = all(
        getattr(p, "receptor", None) in ("gaba_a", "gaba_b")
        for p in pathways
        if regions[p.from_region].exc_fraction == 0.0
    )
    corollary_inputs_excitatory = all(
        regions[p.from_region].exc_fraction == 1.0
        for p in pathways
        if p.to_region == "action_corollary"
    )
    nmda_regions = sorted(
        name for name, region in regions.items() if region.enable_nmda
    )

    symmetric_pairs = []
    for stem, target_stem in (
        ("proposal_stop_fs", "proposal"),
        ("commit_stop_fs", "commit"),
        ("commit_stop_fs", "motor"),
    ):
        left = _local_synapse_coordinates(
            bridge, f"{stem}_0", f"{target_stem}_0"
        )
        right = _local_synapse_coordinates(
            bridge, f"{stem}_1", f"{target_stem}_1"
        )
        symmetric_pairs.append(left == right)

    v10_audit = v10_structural_audit(bridge, handles)
    inherited_policy_checks = {
        name: value for name, value in v10_audit["checks"].items()
        if name not in (
            "selector_v2_has_600_neurons",
            "selector_has_36_declared_pathways",
        )
    }
    checks = {
        "v10_policy_structure": bool(all(inherited_policy_checks.values())),
        "unique_ordered_region_pairs": len(pairs) == len(set(pairs)),
        "inhibitory_sources_use_gaba_receptors": bool(
            inhibitory_sources_use_gaba
        ),
        "corollary_inputs_are_excitatory": bool(corollary_inputs_excitatory),
        "only_corollary_nmda_enabled": nmda_regions == ["action_corollary"],
        "new_local_routes_are_exactly_symmetric": all(symmetric_pairs),
        "no_boundary_route_targets_msn": not any(
            p.from_region in BOUNDARY_REGIONS
            and p.to_region.startswith(("str_d1_", "str_d2_"))
            for p in pathways
        ),
        "host_boundary_is_closed": not any(HOST_BOUNDARY.values()),
    }
    return {
        "checks": checks,
        "pass": bool(all(checks.values())),
        "nmda_regions": nmda_regions,
        "v10": v10_audit,
    }


def _set_drive(bridge, practice_pA: float):
    selector = selector_config("v2")
    xp, _ = get_backend()
    _set_equal_tonic_current(bridge, selector)
    bridge.cp_external_input_current[
        xp.asarray(_indices(bridge, "practice_arousal"))
    ] = xp.float32(practice_pA)
    reset = np.asarray(
        to_host(bridge.cp_external_input_current[
            xp.asarray(_indices(bridge, "selector_reset"))
        ]),
        dtype=np.float64,
    )
    if np.any(reset != 0.0):
        raise AssertionError("V11 selector_reset current must remain zero")


def _step(bridge):
    bridge._run_one_simulation_step()
    bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
    return np.asarray(to_host(bridge.cp_firing_states), dtype=bool)


def _step_hash(firing: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(firing).tobytes()).hexdigest()


def _run_epoch(bridge, handles: dict, *, steps: int, practice_pA: float):
    _set_drive(bridge, practice_pA)
    names = [
        "practice_arousal", "selector_reset", "action_corollary",
        "proposal_stop_fs_0", "proposal_stop_fs_1",
        "commit_stop_fs_0", "commit_stop_fs_1",
        "proposal_0", "proposal_1", "str_d1_0", "str_d1_1",
        "str_d2_0", "str_d2_1", "commit_0", "commit_1",
        "motor_0", "motor_1",
    ]
    indices = {name: _indices(bridge, name) for name in names}
    region_steps = {name: [] for name in names}
    hashes = []
    motor_counts = np.zeros(2, dtype=np.int64)
    first_motor_spike = None
    first_corollary_spike = None
    first_stop_spike = None
    final_boundary_spike = None
    first_crossing = None
    decision_step = None
    decision_counts = None
    simultaneous = False

    for step in range(int(steps)):
        firing = _step(bridge)
        hashes.append(_step_hash(firing))
        for name in names:
            region_steps[name].append(int(firing[indices[name]].sum()))
        if first_motor_spike is None and any(
            region_steps[f"motor_{channel}"][-1] > 0 for channel in CHANNELS
        ):
            first_motor_spike = int(step)
        if first_corollary_spike is None and region_steps["action_corollary"][-1] > 0:
            first_corollary_spike = int(step)
        stop_spikes = sum(
            region_steps[name][-1] for name in BOUNDARY_REGIONS[1:]
        )
        if first_stop_spike is None and stop_spikes > 0:
            first_stop_spike = int(step)
        boundary_spikes = region_steps["action_corollary"][-1] + stop_spikes
        if boundary_spikes > 0:
            final_boundary_spike = int(step)

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
        "first_corollary_spike": first_corollary_spike,
        "first_stop_spike": first_stop_spike,
        "final_boundary_spike": final_boundary_spike,
        "region_spikes": {
            name: int(sum(values)) for name, values in region_steps.items()
        },
        "region_step_counts": region_steps,
        "step_firing_hashes": hashes,
    }


def _run_recovery(bridge, handles: dict, *, steps: int):
    _set_drive(bridge, 0.0)
    boundary_indices = handles["boundary_indices"]
    tail = []
    total = 0
    for _ in range(int(steps)):
        firing = _step(bridge)
        count = int(sum(
            firing[indices].sum() for indices in boundary_indices.values()
        ))
        total += count
        tail.append(count)
        if len(tail) > RECOVERY_QUIET_TAIL_STEPS:
            tail.pop(0)
    corollary = boundary_indices["action_corollary"]
    nmda = np.asarray(
        to_host(bridge.cp_conductance_g_nmda[corollary]), dtype=np.float64
    )
    reset = np.asarray(
        to_host(bridge.cp_external_input_current[
            _indices(bridge, "selector_reset")
        ]),
        dtype=np.float64,
    )
    return {
        "steps": int(steps),
        "boundary_spikes": int(total),
        "boundary_spikes_final_1000": int(sum(tail)),
        "corollary_nmda_max": float(np.max(np.abs(nmda), initial=0.0)),
        "selector_reset_current_max": float(
            np.max(np.abs(reset), initial=0.0)
        ),
    }


def _action_passes(row: dict) -> bool:
    order = (
        row["first_motor_spike"] is not None
        and row["first_corollary_spike"] is not None
        and row["first_stop_spike"] is not None
        and row["first_motor_spike"] < row["first_corollary_spike"]
        < row["first_stop_spike"]
    )
    late = (
        row["decision_step"] is not None
        and (
            row["decision_step"] > BOUNDARY_DECISION_CUTOFF
            or (
                row["final_boundary_spike"] is not None
                and row["final_boundary_spike"] >= BOUNDARY_LATE_STEP
            )
        )
    )
    return bool(row["winner"] in CHANNELS and order and late)


def _recovery_passes(row: dict) -> bool:
    return bool(
        row["boundary_spikes_final_1000"] == 0
        and row["corollary_nmda_max"] <= RECOVERY_NMDA_MAX
        and row["selector_reset_current_max"] == 0.0
    )


def run_construction_point(*, seed: int, recurrence_weight: float):
    validate_construction_seed(seed)
    bridge, handles = build_v11_bridge(
        seed, recurrence_weight=float(recurrence_weight)
    )
    initial_weights = _weights(bridge)
    audit = boundary_structural_audit(bridge, handles)
    warmup = _run_recovery(bridge, handles, steps=WARMUP_STEPS)
    catch = _run_epoch(
        bridge, handles, steps=CATCH_STEPS, practice_pA=CATCH_PRACTICE_PA
    )
    recovery_0 = _run_recovery(bridge, handles, steps=RECOVERY_STEPS)
    action_1 = _run_epoch(
        bridge, handles, steps=ACTION_STEPS, practice_pA=ACTION_PRACTICE_PA
    )
    recovery_1 = _run_recovery(bridge, handles, steps=RECOVERY_STEPS)
    action_2 = _run_epoch(
        bridge, handles, steps=ACTION_STEPS, practice_pA=ACTION_PRACTICE_PA
    )
    recovery_2 = _run_recovery(bridge, handles, steps=RECOVERY_STEPS)
    final_weights = _weights(bridge)

    checks = {
        "structure": audit["pass"],
        "weak_catch_has_no_motor_crossing": catch["first_crossing"] is None,
        "weak_catch_has_no_boundary_spike": sum(
            catch["region_spikes"][name] for name in BOUNDARY_REGIONS
        ) == 0,
        "first_action_is_clean_bounded": _action_passes(action_1),
        "first_recovery_is_autonomous": _recovery_passes(recovery_1),
        "same_brain_second_action_is_clean_bounded": _action_passes(action_2),
        "second_recovery_is_autonomous": _recovery_passes(recovery_2),
        "no_reset_current_anywhere": all(
            row["selector_reset_current_max"] == 0.0
            for row in (warmup, recovery_0, recovery_1, recovery_2)
        ),
        "weights_are_byte_identical": bool(
            np.array_equal(initial_weights, final_weights)
        ),
    }
    return {
        "seed": int(seed),
        "recurrence_weight": float(recurrence_weight),
        "checks": checks,
        "pass": bool(all(checks.values())),
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
        "topology": {
            "neurons": int(bridge.core_config.num_neurons),
            "synapses": int(bridge.cp_connections.nnz),
            "regions": len(bridge.core_config.brain_regions),
            "pathways": len(bridge.core_config.region_pathways),
        },
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


def run_construction_ladder(seed: int = CONSTRUCTION_SEED):
    validate_construction_seed(seed)
    started = time.perf_counter()
    rows = [
        run_construction_point(seed=seed, recurrence_weight=weight)
        for weight in RECURRENCE_WEIGHTS
    ]
    selected = next(
        (row["recurrence_weight"] for row in rows if row["pass"]), None
    )
    return {
        "probe": "vocal_action_credit_gate_v11_boundary_construction",
        "stage": "construction_backend",
        "seed": int(seed),
        "reserved_seed": RESERVED_SEED,
        "formal_execution_open": False,
        "recurrence_weights": list(RECURRENCE_WEIGHTS),
        "selected_weight_this_backend": selected,
        "backend_info": _backend_info(),
        "host_boundary": HOST_BOUNDARY,
        "config": {
            "warmup_steps": WARMUP_STEPS,
            "catch_steps": CATCH_STEPS,
            "action_steps": ACTION_STEPS,
            "recovery_steps": RECOVERY_STEPS,
            "catch_practice_pA": CATCH_PRACTICE_PA,
            "action_practice_pA": ACTION_PRACTICE_PA,
        },
        "rows": rows,
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
        if item["recurrence_weights"] != list(RECURRENCE_WEIGHTS):
            raise ValueError("construction ladder differs from preregistration")
    rows = {
        backend: {row["recurrence_weight"]: row for row in item["rows"]}
        for backend, item in by_backend.items()
    }
    selected = next((
        weight for weight in RECURRENCE_WEIGHTS
        if rows["numpy"][weight]["pass"] and rows["cupy"][weight]["pass"]
    ), None)
    return {
        "probe": "vocal_action_credit_gate_v11_boundary_construction",
        "stage": "construction_cross_backend",
        "seed": CONSTRUCTION_SEED,
        "reserved_seed": RESERVED_SEED,
        "formal_execution_open": False,
        "recurrence_weights": list(RECURRENCE_WEIGHTS),
        "selected_recurrence_weight": selected,
        "construction_go": selected is not None,
        "backend_artifacts": {
            backend: str(path)
            for backend, path in zip(
                (artifacts[0]["backend_info"]["backend"], artifacts[1]["backend_info"]["backend"]),
                (numpy_path, cupy_path),
            )
        },
        "backend_selected_weights": {
            backend: item["selected_weight_this_backend"]
            for backend, item in by_backend.items()
        },
        "shared_pass_by_weight": [
            {
                "recurrence_weight": weight,
                "numpy_pass": bool(rows["numpy"][weight]["pass"]),
                "cupy_pass": bool(rows["cupy"][weight]["pass"]),
            }
            for weight in RECURRENCE_WEIGHTS
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=CONSTRUCTION_SEED)
    parser.add_argument("--construction-only", action="store_true")
    parser.add_argument("--merge-construction", nargs=2, metavar=("NUMPY", "CUPY"))
    parser.add_argument(
        "--out",
        default="research/findings/raw/vocal_action_credit_gate_v11/construction.json",
    )
    args = parser.parse_args(argv)
    if args.merge_construction:
        result = merge_construction_artifacts(
            Path(args.merge_construction[0]), Path(args.merge_construction[1])
        )
    elif args.construction_only:
        result = run_construction_ladder(args.seed)
    else:
        validate_formal_seed(args.seed)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "stage": result["stage"],
        "seed": result["seed"],
        "selected": result.get(
            "selected_recurrence_weight",
            result.get("selected_weight_this_backend"),
        ),
        "output": str(output),
    }, indent=2))
    if result["stage"] == "construction_cross_backend":
        return 0 if result["construction_go"] else 1
    return 0 if result["selected_weight_this_backend"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
