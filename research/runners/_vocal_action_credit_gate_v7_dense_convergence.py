"""Sealed Gate B v7 dense-convergence engagement smoke.

Reserved seed zero may test whether a larger neural action trace provides the
convergent excitation needed by the existing MSN-D1 expectation population.
No formal seed is assigned or executable.
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


OPEN_PHASES: tuple[str, ...] = ()
SMOKE_SEED = 0
TRACE_SIZE_LADDER = (24, 64, 128, 200)

HOST_BOUNDARY = {
    **v5l.HOST_BOUNDARY,
    "host_dense_trace_injection": False,
    "host_clears_residual_girk_between_measured_trials": True,
}


@dataclass(frozen=True)
class VocalCreditConfigV7(v5l.VocalCreditConfigV5Learning):
    n_value: int = TRACE_SIZE_LADDER[0]
    trace_to_expectation_density: float = 0.50


def v7_config(trace_size: int = TRACE_SIZE_LADDER[0]) -> VocalCreditConfigV7:
    checked = int(trace_size)
    if checked not in TRACE_SIZE_LADDER:
        raise ValueError(
            f"trace size must be in preregistered ladder {TRACE_SIZE_LADDER}; "
            f"received={checked}"
        )
    return VocalCreditConfigV7(n_value=checked)


def validate_smoke_seed(seed: int) -> int:
    checked = int(seed)
    if checked != SMOKE_SEED:
        raise ValueError(f"v7 smoke accepts reserved seed {SMOKE_SEED} only")
    return checked


def validate_phase(phase: str) -> str:
    raise ValueError(
        "Gate B v7 formal phases are sealed and unassigned; "
        f"open phases={OPEN_PHASES}, requested={phase!r}"
    )


def run_formal_seed(seed: int):
    raise ValueError(f"Gate B v7 formal execution is sealed; received={int(seed)}")


def _v7_pathways(
    config: VocalCreditConfigV7,
    *,
    action_permutation: tuple[int, int] = (0, 1),
):
    pathways = v5l._learning_pathways(
        config, action_permutation=action_permutation
    )
    updated = []
    for pathway in pathways:
        is_learned_expectation = bool(
            pathway.from_region in {
                v5l._trace(channel) for channel in CHANNELS
            }
            and pathway.to_region in {
                v5l._expectation(channel) for channel in CHANNELS
            }
        )
        is_expectation_to_snc = bool(
            pathway.from_region in {
                v5l._expectation(channel) for channel in CHANNELS
            }
            and pathway.to_region == v5l.SNC
        )
        if is_learned_expectation:
            pathway = replace(
                pathway, density=config.trace_to_expectation_density
            )
        if is_expectation_to_snc:
            pathway = replace(pathway, receptor="gaba_b")
        updated.append(pathway)
    return updated


def build_v7_bridge(
    seed: int = SMOKE_SEED,
    config: VocalCreditConfigV7 | None = None,
    *,
    action_permutation: tuple[int, int] = (0, 1),
):
    validate_smoke_seed(seed)
    config = config or v7_config()
    bridge = build_selector_bridge(
        seed,
        selector_config("v2"),
        extra_regions=v5l._learning_regions(config),
        extra_pathways=_v7_pathways(
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
    bridge.cp_reward_eligibility_synapse_indices = xp.asarray(
        routes.all_indices(), dtype=xp.int64
    )
    return bridge, {
        "routes": routes,
        "actor_routes": actor_routes,
        "expectation_routes": expectation_routes,
        "action_permutation": tuple(action_permutation),
    }


def _clear_girk_and_target_state(bridge) -> None:
    conductance = getattr(bridge, "cp_conductance_g_gabab", None)
    if conductance is not None:
        conductance[:] = 0.0
    target_names = [v5l.SNC, v3.OMISSION_GATE]
    target_names.extend(v5l._expectation(channel) for channel in CHANNELS)
    target = np.concatenate([
        np.asarray(_indices(bridge, name), dtype=np.int64)
        for name in target_names
    ])
    xp, _ = get_backend()
    target_gpu = xp.asarray(target, dtype=xp.int64)
    if (
        getattr(bridge, "cp_membrane_potential_v", None) is not None
        and getattr(bridge, "cp_izh_vr", None) is not None
    ):
        bridge.cp_membrane_potential_v[target_gpu] = bridge.cp_izh_vr[target_gpu]
    if getattr(bridge, "cp_recovery_variable_u", None) is not None:
        bridge.cp_recovery_variable_u[target_gpu] = 0.0


def _reset_measured_trial(bridge, config: VocalCreditConfigV7) -> None:
    v3._reset_trial_v3(bridge, selector_config("v2"), config)
    _clear_girk_and_target_state(bridge)


def _run_trial(
    bridge,
    handles: dict,
    config: VocalCreditConfigV7,
    *,
    reward_action: int | None,
    scheduled_reward: bool | None = None,
) -> dict[str, object]:
    row = v5l._run_fixed_trial(
        bridge,
        handles,
        config,
        reward_action=reward_action,
        scheduled_reward=scheduled_reward,
    )
    # The inherited helper already performs the standard trial reset. Add only
    # the slow-conductance and target-state isolation required by v7.
    _clear_girk_and_target_state(bridge)
    return row


def run_engagement_condition(
    mode: str,
    config: VocalCreditConfigV7,
    *,
    seed: int = SMOKE_SEED,
) -> dict[str, object]:
    if mode not in {"intact", "expectation_learning_lesion"}:
        raise ValueError(mode)
    validate_smoke_seed(seed)
    bridge, handles = build_v7_bridge(seed=seed, config=config)
    selector = selector_config("v2")
    bridge.set_plasticity_gate(CREDIT_PLASTICITY_GATE, 0.0)
    bridge.set_plasticity_gate(v5l.EXPECTATION_PLASTICITY_GATE, 0.0)
    v3._calibrate_snc_tonic_v3(bridge, selector, config)
    v3._set_trial_current_v3(bridge, selector, config)
    v5._step(bridge, selector.warmup_steps)
    _reset_measured_trial(bridge, config)
    bridge.set_plasticity_gate(
        CREDIT_PLASTICITY_GATE, config.actor_plasticity_gain
    )
    bridge.set_plasticity_gate(v5l.EXPECTATION_PLASTICITY_GATE, 1.0)
    if mode == "expectation_learning_lesion":
        bridge.set_plasticity_gate(v5l.EXPECTATION_PLASTICITY_GATE, 0.0)

    weights_before = np.asarray(
        to_host(bridge.cp_connections.data), dtype=np.float32
    ).copy()
    expectation_before = v5l._weight_means(
        bridge, handles["expectation_routes"]
    )
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
        "trace_size": config.n_value,
        "expectation_weight_before": expectation_before,
        "expectation_weight_after": v5l._weight_means(
            bridge, handles["expectation_routes"]
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


def run_engagement_smoke(
    trace_size: int,
    *,
    seed: int = SMOKE_SEED,
) -> dict[str, object]:
    config = v7_config(trace_size)
    conditions = {
        mode: run_engagement_condition(mode, config, seed=seed)
        for mode in ("intact", "expectation_learning_lesion")
    }
    intact = conditions["intact"]
    lesion = conditions["expectation_learning_lesion"]
    initial = float(intact["expectation_weight_before"][0])
    intact_delta = float(intact["expectation_weight_after"][0]) - initial
    intact_separation = (
        float(intact["expectation_weight_after"][0])
        - float(intact["expectation_weight_after"][1])
    )
    lesion_delta = max(
        abs(float(after) - float(before))
        for before, after in zip(
            lesion["expectation_weight_before"],
            lesion["expectation_weight_after"],
        )
    )
    intact_delay = _late_delay_expectation(intact)
    lesion_delay = _late_delay_expectation(lesion)
    checks = {
        "reserved_seed_only": int(seed) == SMOKE_SEED,
        "formal_execution_remains_sealed": OPEN_PHASES == (),
        "at_least_90pct_clean_action_epochs": all(
            condition["clean_trials"] >= 0.9 * config.smoke_training_trials
            for condition in conditions.values()
        ),
        "rewarded_route_learns_and_separates": bool(
            intact_delta >= 0.25 * initial
            and intact_separation >= 0.20 * initial
        ),
        "responsible_expectation_fires_before_outcome": bool(
            intact_delay[0] > 0
            and intact_delay[0] >= 3 * max(intact_delay[1], 1)
        ),
        "learning_lesion_blocks_route_change": lesion_delta <= 1e-7,
        "learning_lesion_removes_80pct_of_delay_expectation": bool(
            intact_delay[0] > 0
            and lesion_delay[0] <= 0.20 * intact_delay[0]
        ),
        "plasticity_confined_to_declared_routes": all(
            condition["changed_outside_declared_routes"] == 0
            for condition in conditions.values()
        ),
    }
    return {
        "artifact_schema_version": 1,
        "probe": "vocal_action_credit_gate_b_v7_dense_engagement",
        "seed": int(seed),
        "science_seed_executed": False,
        "backend": "cupy" if get_backend()[0].__name__ == "cupy" else "numpy",
        "config": asdict(config),
        "conditions": conditions,
        "late_rewarded_delay_expectation_spikes": {
            "intact": intact_delay,
            "expectation_learning_lesion": lesion_delay,
        },
        "checks": checks,
        "status": "ENGAGEMENT_PASS" if all(checks.values()) else "ENGAGEMENT_FAIL",
    }


def run_construction_smoke(
    trace_size: int = TRACE_SIZE_LADDER[0],
    *,
    seed: int = SMOKE_SEED,
) -> dict[str, object]:
    config = v7_config(trace_size)
    bridge, handles = build_v7_bridge(seed=seed, config=config)
    audit = v5l._structural_audit(bridge, handles)
    pathways = _v7_pathways(config)
    learned = [
        pathway for pathway in pathways
        if pathway.from_region in {v5l._trace(c) for c in CHANNELS}
        and pathway.to_region in {v5l._expectation(c) for c in CHANNELS}
    ]
    snc_outputs = [
        pathway for pathway in pathways
        if pathway.from_region in {v5l._expectation(c) for c in CHANNELS}
        and pathway.to_region == v5l.SNC
    ]
    config_payload = json.dumps(
        asdict(config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    xp, _ = get_backend()
    checks = {
        "reserved_seed_only": int(seed) == SMOKE_SEED,
        "formal_execution_unassigned_and_sealed": OPEN_PHASES == (),
        "trace_size_is_preregistered": config.n_value in TRACE_SIZE_LADDER,
        "dense_routes_are_local_and_plastic": bool(
            len(learned) == len(CHANNELS)
            and all(pathway.plastic for pathway in learned)
            and all(
                pathway.density == config.trace_to_expectation_density
                for pathway in learned
            )
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
        "coincidence_confined_to_fixed_trace_loader": bool(
            audit["coincidence"]["enabled_synapses"]
            == audit["coincidence"]["intended_synapses"]
            and audit["coincidence"]["enabled_outside_intended_routes"] == 0
            and audit["coincidence"]["disabled_inside_intended_routes"] == 0
        ),
    }
    return {
        "artifact_schema_version": 1,
        "probe": "vocal_action_credit_gate_b_v7_dense_construction",
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
        "audit": audit,
        "checks": checks,
        "status": "CONSTRUCTION_PASS" if all(checks.values()) else "CONSTRUCTION_FAIL",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SMOKE_SEED)
    parser.add_argument(
        "--trace-size", type=int, choices=TRACE_SIZE_LADDER,
        default=TRACE_SIZE_LADDER[0],
    )
    parser.add_argument("--engagement", action="store_true")
    parser.add_argument("--formal-phase")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.formal_phase is not None:
        validate_phase(args.formal_phase)
    validate_smoke_seed(args.seed)
    result = (
        run_engagement_smoke(args.trace_size, seed=args.seed)
        if args.engagement
        else run_construction_smoke(args.trace_size, seed=args.seed)
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
