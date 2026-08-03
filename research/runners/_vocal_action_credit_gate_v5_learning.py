"""Sealed Gate B v5 learning candidate with separate trace and expectation.

Reserved seed zero may construct this circuit for implementation smoke. Formal
seed partitions are committed but remain non-executable until the smoke and its
independent audit pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np

from research.runners import _vocal_action_credit_gate_v3 as v3
from research.runners import _vocal_action_credit_gate_v5 as v5
from research.runners._vocal_action_credit_gate import (
    CREDIT_CUE,
    CREDIT_PLASTICITY_GATE,
    OUTCOME_ONSET,
    REWARD_US,
    SNC,
    CreditRoutes,
    _actor,
    _route_synapses,
    _value,
)
from research.runners._vocal_action_selector_gate import (
    CHANNELS,
    _region,
    build_selector_bridge,
    selector_config,
)
from sim.backend import get_backend, to_host
from sim.enums import NeuronType
from sim.regions import RegionPathway
from tools.assign_vocal_credit_v5_learning_seeds import validate_manifest


OPEN_PHASES: tuple[str, ...] = ()
SMOKE_SEED = 0
SEED_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "seed_manifests"
    / "vocal_action_credit_gate_v5_learning.json"
)

EXPECTATION_PREFIX = "vocal_credit_reward_expectation_"
EXPECTATION_PLASTICITY_GATE = "vocal_reward_expectation_learning"
EXPECTATION_OUTPUT_GATE = "vocal_reward_expectation_output"

HOST_BOUNDARY = {
    **v5.HOST_BOUNDARY,
    "host_copies_action_to_outcome": False,
    "host_prediction_error_calculation": False,
    "host_weight_update": False,
    "formal_execution_open": False,
}


def _load_seed_partitions() -> dict[str, tuple[int, ...]]:
    payload = json.loads(SEED_MANIFEST.read_text(encoding="utf-8"))
    validate_manifest(payload)
    return {
        name: tuple(int(seed) for seed in payload["partitions"][name])
        for name in ("calibration", "development", "held_out")
    }


_SEED_PARTITIONS = _load_seed_partitions()


@dataclass(frozen=True)
class VocalCreditConfigV5Learning(v5.VocalCreditConfigV5):
    n_expectation: int = 24
    trace_to_expectation_weight: float = 0.1
    outcome_to_expectation_weight: float = 2.0
    expectation_to_snc_weight: float = 10.0
    expectation_to_omission_weight: float = 8.0
    expectation_reset_weight: float = 16.0


def learning_config() -> VocalCreditConfigV5Learning:
    return VocalCreditConfigV5Learning()


def _trace(channel: int) -> str:
    """The historical v5 value pool is a fixed action trace, not a critic."""
    return _value(channel)


def _expectation(channel: int) -> str:
    return f"{EXPECTATION_PREFIX}{int(channel)}"


def validate_smoke_seed(seed: int) -> int:
    checked = int(seed)
    if checked != SMOKE_SEED:
        raise ValueError(f"v5 learning smoke accepts reserved seed {SMOKE_SEED} only")
    return checked


def validate_phase(phase: str) -> str:
    raise ValueError(
        "Gate B v5 learning formal phases are sealed; "
        f"open phases={OPEN_PHASES}, requested={phase!r}"
    )


def validate_formal_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    checked = tuple(int(seed) for seed in seeds)
    raise ValueError(f"Gate B v5 learning formal seeds are sealed; received={checked}")


def run_formal_seed(seed: int):
    raise ValueError(f"Gate B v5 learning formal execution is sealed; received={int(seed)}")


def _learning_regions(config: VocalCreditConfigV5Learning):
    regions = list(v3._v3_regions(config))
    d1 = NeuronType.IZH2007_STRIATAL_MSN_D1
    regions.extend(
        _region(
            _expectation(channel),
            config.n_expectation,
            exc_fraction=0.0,
            neuron_type=d1,
        )
        for channel in CHANNELS
    )
    return regions


def _is_trace_critic_output(pathway: RegionPathway) -> bool:
    return bool(
        pathway.from_region in {_trace(channel) for channel in CHANNELS}
        and pathway.to_region in {SNC, v3.OMISSION_GATE}
    )


def _learning_pathways(
    config: VocalCreditConfigV5Learning,
    *,
    action_permutation: tuple[int, int] = (0, 1),
) -> list[RegionPathway]:
    inherited = [
        pathway
        for pathway in v5._v5_pathways(
            config, action_permutation=action_permutation
        )
        if not _is_trace_critic_output(pathway)
    ]
    # The action-tag loader is fixed. Only the downstream trace-to-expectation
    # route is allowed to learn reward prediction.
    pathways = [
        replace(pathway, plastic=False, plasticity_gate=None)
        if (
            pathway.from_region.startswith("commit_")
            and pathway.to_region in {_trace(channel) for channel in CHANNELS}
        )
        else pathway
        for pathway in inherited
    ]

    for channel in CHANNELS:
        pathways.extend([
            RegionPathway(
                from_region=_trace(channel),
                to_region=_expectation(channel),
                density=1.0,
                weight_mean=config.trace_to_expectation_weight,
                weight_jitter=0.0,
                plastic=True,
                plasticity_gate=EXPECTATION_PLASTICITY_GATE,
            ),
            RegionPathway(
                from_region=OUTCOME_ONSET,
                to_region=_expectation(channel),
                density=1.0,
                weight_mean=config.outcome_to_expectation_weight,
                weight_jitter=0.0,
                plastic=False,
            ),
            RegionPathway(
                from_region=_expectation(channel),
                to_region=SNC,
                density=0.50,
                weight_mean=config.expectation_to_snc_weight,
                weight_jitter=0.05,
                plastic=False,
                receptor="gaba_b",
                transmission_gate=EXPECTATION_OUTPUT_GATE,
            ),
            RegionPathway(
                from_region=_expectation(channel),
                to_region=v3.OMISSION_GATE,
                density=0.60,
                weight_mean=config.expectation_to_omission_weight,
                weight_jitter=0.05,
                plastic=False,
                receptor="gaba_b",
                transmission_gate=EXPECTATION_OUTPUT_GATE,
            ),
            RegionPathway(
                from_region="selector_reset",
                to_region=_expectation(channel),
                density=0.70,
                weight_mean=config.expectation_reset_weight,
                weight_jitter=0.0,
                plastic=False,
                receptor="gaba_a",
            ),
        ])
    return pathways


def build_learning_bridge(
    seed: int = SMOKE_SEED,
    config: VocalCreditConfigV5Learning | None = None,
    *,
    action_permutation: tuple[int, int] = (0, 1),
):
    validate_smoke_seed(seed)
    config = config or learning_config()
    bridge = build_selector_bridge(
        seed,
        selector_config("v2"),
        extra_regions=_learning_regions(config),
        extra_pathways=_learning_pathways(
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
            bridge, _trace(channel), _expectation(channel)
        )
        for channel in CHANNELS
    }
    routes = CreditRoutes(actor=actor_routes, value=expectation_routes)
    xp, _ = get_backend()
    bridge.set_global_plasticity_gain(0.0)
    bridge.set_plasticity_gate(CREDIT_PLASTICITY_GATE, config.actor_plasticity_gain)
    bridge.set_plasticity_gate(EXPECTATION_PLASTICITY_GATE, 1.0)
    bridge.cp_reward_eligibility_synapse_indices = xp.asarray(
        routes.all_indices(), dtype=xp.int64
    )
    return bridge, {
        "routes": routes,
        "actor_routes": actor_routes,
        "expectation_routes": expectation_routes,
        "action_permutation": tuple(action_permutation),
    }


def _structural_audit(bridge, handles: dict) -> dict[str, object]:
    plastic = np.asarray(to_host(bridge.cp_synapse_plastic_mask), dtype=bool)
    expected_plastic = np.zeros(plastic.shape, dtype=bool)
    expected_plastic[handles["routes"].all_indices()] = True
    eligible = np.asarray(
        to_host(bridge.cp_reward_eligibility_synapse_indices), dtype=np.int64
    )
    expected_eligible = np.sort(handles["routes"].all_indices())
    actor_gate = np.sort(np.asarray(
        to_host(bridge._plasticity_gate_indices_gpu[CREDIT_PLASTICITY_GATE]),
        dtype=np.int64,
    ))
    expectation_gate = np.sort(np.asarray(
        to_host(bridge._plasticity_gate_indices_gpu[EXPECTATION_PLASTICITY_GATE]),
        dtype=np.int64,
    ))
    expected_actor = np.sort(np.concatenate(list(handles["actor_routes"].values())))
    expected_expectation = np.sort(np.concatenate(
        list(handles["expectation_routes"].values())
    ))
    coincidence = v5._coincidence_route_audit(
        bridge, handles["action_permutation"]
    )
    return {
        "plastic_synapses": int(plastic.sum()),
        "declared_plastic_synapses": int(expected_plastic.sum()),
        "plastic_outside_declared_routes": int(
            np.logical_and(plastic, ~expected_plastic).sum()
        ),
        "fixed_inside_declared_routes": int(
            np.logical_and(~plastic, expected_plastic).sum()
        ),
        "reward_eligibility_matches_declared_routes": bool(
            np.array_equal(np.sort(eligible), expected_eligible)
        ),
        "actor_gate_matches_actor_routes": bool(
            np.array_equal(actor_gate, expected_actor)
        ),
        "expectation_gate_matches_expectation_routes": bool(
            np.array_equal(expectation_gate, expected_expectation)
        ),
        "actor_gate_gain": float(
            bridge._plasticity_gate_values[CREDIT_PLASTICITY_GATE]
        ),
        "expectation_gate_gain": float(
            bridge._plasticity_gate_values[EXPECTATION_PLASTICITY_GATE]
        ),
        "coincidence": coincidence,
        "trace_expectation_populations_distinct": all(
            _trace(channel) != _expectation(channel) for channel in CHANNELS
        ),
    }


def run_construction_smoke(seed: int = SMOKE_SEED) -> dict[str, object]:
    config = learning_config()
    bridge, handles = build_learning_bridge(seed=seed, config=config)
    audit = _structural_audit(bridge, handles)
    xp, _ = get_backend()
    config_payload = json.dumps(
        asdict(config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    checks = {
        "reserved_seed_only": int(seed) == SMOKE_SEED,
        "formal_execution_sealed": OPEN_PHASES == (),
        "trace_and_expectation_are_separate": bool(
            audit["trace_expectation_populations_distinct"]
        ),
        "plasticity_confined_to_declared_routes": bool(
            audit["plastic_synapses"] == audit["declared_plastic_synapses"]
            and audit["plastic_outside_declared_routes"] == 0
            and audit["fixed_inside_declared_routes"] == 0
        ),
        "runtime_learning_ownership_matches_declared_routes": bool(
            audit["reward_eligibility_matches_declared_routes"]
            and audit["actor_gate_matches_actor_routes"]
            and audit["expectation_gate_matches_expectation_routes"]
            and audit["actor_gate_gain"] == config.actor_plasticity_gain
            and audit["expectation_gate_gain"] == 1.0
        ),
        "coincidence_confined_to_fixed_trace_loader": bool(
            audit["coincidence"]["enabled_synapses"]
            == audit["coincidence"]["intended_synapses"]
            and audit["coincidence"]["enabled_outside_intended_routes"] == 0
            and audit["coincidence"]["disabled_inside_intended_routes"] == 0
        ),
    }
    result = {
        "artifact_schema_version": 1,
        "probe": "vocal_action_credit_gate_b_v5_learning_construction",
        "seed": int(seed),
        "science_seed_executed": False,
        "backend": "cupy" if xp.__name__ == "cupy" else "numpy",
        "device": (
            xp.cuda.runtime.getDeviceProperties(0)["name"].decode("utf-8")
            if xp.__name__ == "cupy"
            else platform.processor() or platform.machine() or "CPU"
        ),
        "config_sha256": hashlib.sha256(config_payload).hexdigest(),
        "formal_seed_partitions_assigned_but_sealed": {
            name: list(values) for name, values in _SEED_PARTITIONS.items()
        },
        "host_boundary": dict(HOST_BOUNDARY),
        "audit": audit,
        "checks": checks,
        "status": "CONSTRUCTION_PASS" if all(checks.values()) else "CONSTRUCTION_FAIL",
    }
    del bridge
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SMOKE_SEED)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--formal-phase")
    args = parser.parse_args(argv)
    if args.formal_phase is not None:
        validate_phase(args.formal_phase)
    result = run_construction_smoke(args.seed)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0 if result["status"] == "CONSTRUCTION_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
