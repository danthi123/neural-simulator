#!/usr/bin/env python3
"""Deterministic adaptive experiment proposals above the sealed experiment harness.

This module proposes parameter/fidelity batches only. It deliberately cannot run a
scientific command. Proposed points must be materialized in a preregistered spec
and pass tools.experiment's normal seal, job expansion, and execution path.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import scipy
    from scipy.stats import qmc
except ImportError:  # pragma: no cover - SciPy is a project dependency.
    scipy = None
    qmc = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.experiment import HarnessError, load_experiment_spec  # noqa: E402


SCHEMA = "sim-adaptive-experiment-v1"
BATCH_SCHEMA = "sim-adaptive-experiment-batch-v1"
CANONICALIZATION = "json-sort-keys-compact-ascii-v1"
OBJECTIVE_CATEGORIES = frozenset(
    ("physiology", "behavior", "robustness", "compute", "scaffold_penalty")
)
FIDELITY_KINDS = ("cpu_screen", "gpu", "replication")
PARAMETER_TYPES = frozenset(("continuous", "discrete", "categorical"))


class AdaptiveExperimentError(ValueError):
    """Raised when a design cannot produce a trustworthy adaptive proposal."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdaptiveExperimentError(message)


def _text(value: Any, field: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{field} must be a non-empty string")
    return value.strip()


def _number(value: Any, field: str) -> float:
    _require(not isinstance(value, bool) and isinstance(value, (int, float)), f"{field} must be numeric")
    result = float(value)
    _require(math.isfinite(result), f"{field} must be finite")
    return result


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    _require(not isinstance(value, bool) and isinstance(value, int), f"{field} must be an integer")
    _require(value >= minimum, f"{field} must be >= {minimum}")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _held_out(name: str) -> bool:
    return "".join(character for character in name.lower() if character.isalnum()).startswith("heldout")


def _safe_repo_path(root: Path, value: Any, field: str) -> tuple[str, Path]:
    text = _text(value, field)
    relative = PurePosixPath(text)
    _require(not relative.is_absolute() and relative.name and ".." not in relative.parts,
             f"{field} must be a safe repository-relative path")
    root = root.resolve(strict=True)
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if not os.path.lexists(current):
            continue
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise AdaptiveExperimentError(f"cannot inspect {field}: {current}: {exc}") from exc
        _require(not stat.S_ISLNK(mode), f"{field} cannot contain a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise AdaptiveExperimentError(f"{field} is missing or escapes the repository: {text}") from exc
    _require(candidate.is_file() and not candidate.is_symlink(), f"{field} must be a regular non-symlink file")
    return relative.as_posix(), resolved


def _normalize_scalar(value: Any, field: str) -> Any:
    _require(value is not None and isinstance(value, (str, int, float, bool)),
             f"{field} must be a JSON scalar")
    if isinstance(value, float):
        _require(math.isfinite(value), f"{field} must be finite")
    return value


def _validate_parameter_space(raw: Any) -> dict[str, dict[str, Any]]:
    _require(isinstance(raw, dict) and raw, "parameter_space must be a non-empty object")
    result: dict[str, dict[str, Any]] = {}
    for name in sorted(raw):
        _text(name, "parameter name")
        spec = raw[name]
        _require(isinstance(spec, dict), f"parameter {name!r} must be an object")
        kind = spec.get("type")
        _require(kind in PARAMETER_TYPES, f"parameter {name!r} has unsupported type {kind!r}")
        if kind == "continuous":
            _require(set(spec) <= {"type", "low", "high", "transform"},
                     f"continuous parameter {name!r} has unknown fields")
            low = _number(spec.get("low"), f"parameter {name!r} low")
            high = _number(spec.get("high"), f"parameter {name!r} high")
            _require(low < high, f"parameter {name!r} requires low < high")
            transform = spec.get("transform", "linear")
            _require(transform in ("linear", "log"), f"parameter {name!r} transform must be linear or log")
            _require(transform != "log" or low > 0, f"log parameter {name!r} requires low > 0")
            result[name] = {"type": kind, "low": low, "high": high, "transform": transform}
        else:
            _require(set(spec) == {"type", "values"}, f"parameter {name!r} requires exactly type and values")
            values = spec.get("values")
            _require(isinstance(values, list) and len(values) >= 2,
                     f"parameter {name!r} values must contain at least two choices")
            normalized = [_normalize_scalar(item, f"parameter {name!r} value") for item in values]
            _require(len({_canonical_bytes(item) for item in normalized}) == len(normalized),
                     f"parameter {name!r} values must be unique")
            if kind == "discrete":
                _require(all(not isinstance(item, bool) and isinstance(item, (int, float)) for item in normalized),
                         f"discrete parameter {name!r} values must be numeric")
            result[name] = {"type": kind, "values": normalized}
    return result


def _value_expression(node: Any, parameters: Mapping[str, Any]) -> Any:
    _require(isinstance(node, dict), "constraint value expressions must be objects")
    if set(node) == {"param"}:
        name = _text(node["param"], "constraint parameter")
        _require(name in parameters, f"constraint references unknown parameter {name!r}")
        return parameters[name]
    if set(node) == {"value"}:
        return node["value"]
    op = node.get("op")
    args = node.get("args")
    _require(set(node) == {"op", "args"} and op in ("add", "sub", "mul", "div", "min", "max"),
             "unsupported constraint value expression")
    _require(isinstance(args, list) and len(args) >= 1, f"constraint operation {op!r} needs arguments")
    values = [_number(_value_expression(item, parameters), f"constraint {op} operand") for item in args]
    if op == "add":
        return sum(values)
    if op == "sub":
        _require(len(values) == 2, "constraint sub requires two operands")
        return values[0] - values[1]
    if op == "mul":
        return math.prod(values)
    if op == "div":
        _require(len(values) == 2 and values[1] != 0, "constraint div requires two operands and nonzero divisor")
        return values[0] / values[1]
    return min(values) if op == "min" else max(values)


def _predicate(node: Any, parameters: Mapping[str, Any]) -> bool:
    _require(isinstance(node, dict), "constraint predicate must be an object")
    op = node.get("op")
    if op in ("and", "or"):
        args = node.get("args")
        _require(set(node) == {"op", "args"} and isinstance(args, list) and len(args) >= 2,
                 f"constraint {op} requires at least two predicates")
        values = [_predicate(item, parameters) for item in args]
        return all(values) if op == "and" else any(values)
    if op == "not":
        _require(set(node) == {"op", "arg"}, "constraint not requires one arg")
        return not _predicate(node["arg"], parameters)
    _require(op in ("lt", "le", "gt", "ge", "eq", "ne", "in", "not_in"),
             f"unsupported constraint predicate {op!r}")
    _require(set(node) == {"op", "left", "right"}, f"constraint {op!r} requires left and right")
    left = _value_expression(node["left"], parameters)
    if op in ("in", "not_in"):
        right = node["right"]
        _require(isinstance(right, list), f"constraint {op!r} right side must be a list")
        result = left in right
        return result if op == "in" else not result
    right = _value_expression(node["right"], parameters)
    try:
        if op == "lt":
            return left < right
        if op == "le":
            return left <= right
        if op == "gt":
            return left > right
        if op == "ge":
            return left >= right
        if op == "eq":
            return left == right
        return left != right
    except TypeError as exc:
        raise AdaptiveExperimentError(f"constraint {op!r} compares incompatible values") from exc


def _validate_constraints(raw: Any, space: Mapping[str, Any]) -> list[dict[str, Any]]:
    _require(isinstance(raw, list), "constraints must be a list")
    result = []
    seen = set()
    midpoint = _midpoint(space)
    for index, item in enumerate(raw):
        _require(isinstance(item, dict) and set(item) == {"id", "source", "predicate"},
                 f"constraint[{index}] must contain exactly id, source, and predicate")
        identifier = _text(item["id"], f"constraint[{index}] id")
        _require(identifier not in seen, f"duplicate constraint id {identifier!r}")
        seen.add(identifier)
        source = _text(item["source"], f"constraint {identifier!r} source")
        _predicate(item["predicate"], midpoint)  # Structural/type validation without requiring feasibility.
        result.append({"id": identifier, "source": source, "predicate": item["predicate"]})
    return result


def _validate_objectives(raw: Any) -> list[dict[str, Any]]:
    _require(isinstance(raw, list) and len(raw) >= 2, "objectives must contain at least two objectives")
    result = []
    names = set()
    categories = set()
    total_weight = 0.0
    for index, item in enumerate(raw):
        _require(isinstance(item, dict), f"objective[{index}] must be an object")
        _require(set(item) <= {"name", "category", "direction", "weight", "range", "target"}
                 and {"name", "category", "direction", "weight", "range"} <= set(item),
                 f"objective[{index}] has missing or unknown fields")
        name = _text(item["name"], f"objective[{index}] name")
        _require(name not in names, f"duplicate objective name {name!r}")
        names.add(name)
        category = item["category"]
        _require(category in OBJECTIVE_CATEGORIES, f"objective {name!r} has unknown category {category!r}")
        categories.add(category)
        direction = item["direction"]
        _require(direction in ("maximize", "minimize"), f"objective {name!r} direction is invalid")
        weight = _number(item["weight"], f"objective {name!r} weight")
        _require(weight > 0, f"objective {name!r} weight must be positive")
        scale = item["range"]
        _require(isinstance(scale, list) and len(scale) == 2, f"objective {name!r} range must have two values")
        low = _number(scale[0], f"objective {name!r} range low")
        high = _number(scale[1], f"objective {name!r} range high")
        _require(low < high, f"objective {name!r} range requires low < high")
        target = item.get("target")
        if target is not None:
            target = _number(target, f"objective {name!r} target")
            _require(low <= target <= high, f"objective {name!r} target must lie inside its range")
        total_weight += weight
        result.append({"name": name, "category": category, "direction": direction,
                       "weight": weight, "range": [low, high], "target": target})
    missing = sorted(OBJECTIVE_CATEGORIES - categories)
    _require(not missing, f"objectives must cover all required categories; missing {missing}")
    for item in result:
        item["weight"] /= total_weight
    return result


def _validate_policy(raw: Any) -> dict[str, Any]:
    _require(isinstance(raw, dict), "policy must be an object")
    allowed = {
        "seed", "batch_size", "candidate_pool_size", "initial_design_size",
        "min_surrogate_observations", "exploration_weight", "promotion_slots",
        "promotion_quantile", "min_completed_for_promotion", "max_completed_observations",
        "plateau_window", "min_improvement", "min_feasible_fraction",
        "research_after_observations", "max_model_uncertainty", "stop_on_replicated_targets",
    }
    _require(set(raw) <= allowed, f"policy has unknown fields: {sorted(set(raw) - allowed)}")
    policy = {
        "seed": _integer(raw.get("seed", 0), "policy seed"),
        "batch_size": _integer(raw.get("batch_size", 4), "policy batch_size", minimum=1),
        "candidate_pool_size": _integer(raw.get("candidate_pool_size", 256), "policy candidate_pool_size", minimum=16),
        "initial_design_size": _integer(raw.get("initial_design_size", 8), "policy initial_design_size", minimum=1),
        "min_surrogate_observations": _integer(raw.get("min_surrogate_observations", 6),
                                                "policy min_surrogate_observations", minimum=3),
        "exploration_weight": _number(raw.get("exploration_weight", 0.25), "policy exploration_weight"),
        "promotion_slots": _integer(raw.get("promotion_slots", 1), "policy promotion_slots"),
        "promotion_quantile": _number(raw.get("promotion_quantile", 0.75), "policy promotion_quantile"),
        "min_completed_for_promotion": _integer(raw.get("min_completed_for_promotion", 3),
                                                  "policy min_completed_for_promotion", minimum=1),
        "max_completed_observations": _integer(raw.get("max_completed_observations", 100),
                                                 "policy max_completed_observations", minimum=1),
        "plateau_window": _integer(raw.get("plateau_window", 8), "policy plateau_window", minimum=2),
        "min_improvement": _number(raw.get("min_improvement", 0.005), "policy min_improvement"),
        "min_feasible_fraction": _number(raw.get("min_feasible_fraction", 0.01),
                                          "policy min_feasible_fraction"),
        "research_after_observations": _integer(raw.get("research_after_observations", 8),
                                                  "policy research_after_observations", minimum=1),
        "max_model_uncertainty": _number(raw.get("max_model_uncertainty", 0.08),
                                          "policy max_model_uncertainty"),
        "stop_on_replicated_targets": raw.get("stop_on_replicated_targets", True),
    }
    _require(policy["exploration_weight"] >= 0, "policy exploration_weight must be nonnegative")
    _require(0 <= policy["promotion_quantile"] <= 1, "policy promotion_quantile must be in [0, 1]")
    _require(0 <= policy["min_feasible_fraction"] <= 1, "policy min_feasible_fraction must be in [0, 1]")
    _require(policy["min_improvement"] >= 0 and policy["max_model_uncertainty"] >= 0,
             "policy improvement and uncertainty thresholds must be nonnegative")
    _require(isinstance(policy["stop_on_replicated_targets"], bool),
             "policy stop_on_replicated_targets must be Boolean")
    return policy


def _midpoint(space: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    point = {}
    for name, spec in space.items():
        if spec["type"] == "continuous":
            point[name] = math.sqrt(spec["low"] * spec["high"]) if spec["transform"] == "log" \
                else (spec["low"] + spec["high"]) / 2
        else:
            point[name] = spec["values"][len(spec["values"]) // 2]
    return point


def _valid_parameter_value(name: str, value: Any, spec: Mapping[str, Any]) -> Any:
    if spec["type"] == "continuous":
        number = _number(value, f"observation parameter {name!r}")
        _require(spec["low"] <= number <= spec["high"], f"observation parameter {name!r} is out of bounds")
        return number
    _require(value in spec["values"], f"observation parameter {name!r} is not an allowed value")
    return value


def _validate_design(raw: Any, *, root: Path) -> dict[str, Any]:
    _require(isinstance(raw, dict), "adaptive design must be a JSON object")
    expected = {"schema", "id", "experiment", "parameter_space", "constraints", "objectives",
                "fidelity_tiers", "observations", "policy"}
    _require(set(raw) == expected, f"adaptive design must contain exactly {sorted(expected)}")
    _require(raw["schema"] == SCHEMA, f"unsupported adaptive design schema {raw['schema']!r}")
    design_id = _text(raw["id"], "design id")

    experiment = raw["experiment"]
    _require(isinstance(experiment, dict) and set(experiment) == {"spec_path"},
             "experiment must contain exactly spec_path")
    relative_spec, spec_path = _safe_repo_path(root, experiment["spec_path"], "experiment spec_path")
    try:
        experiment_spec = load_experiment_spec(spec_path)
    except HarnessError as exc:
        raise AdaptiveExperimentError(f"experiment spec is invalid: {exc}") from exc
    held_out = {name for name in experiment_spec["partitions"] if _held_out(name)}
    _require(held_out, "experiment spec must retain an explicit held-out partition")

    space = _validate_parameter_space(raw["parameter_space"])
    constraints = _validate_constraints(raw["constraints"], space)
    objectives = _validate_objectives(raw["objectives"])
    policy = _validate_policy(raw["policy"])

    tiers_raw = raw["fidelity_tiers"]
    _require(isinstance(tiers_raw, list) and len(tiers_raw) == 3,
             "fidelity_tiers must declare cpu_screen, gpu, and replication")
    tiers = []
    names = set()
    for index, item in enumerate(tiers_raw):
        _require(isinstance(item, dict) and set(item) == {"name", "kind", "backend", "partition", "cost"},
                 f"fidelity_tiers[{index}] has invalid fields")
        name = _text(item["name"], f"fidelity_tiers[{index}] name")
        _require(name not in names, f"duplicate fidelity name {name!r}")
        names.add(name)
        kind = item["kind"]
        _require(kind in FIDELITY_KINDS, f"fidelity tier {name!r} has invalid kind {kind!r}")
        backend = _text(item["backend"], f"fidelity tier {name!r} backend")
        partition = _text(item["partition"], f"fidelity tier {name!r} partition")
        _require(backend in experiment_spec["backends"], f"fidelity tier {name!r} uses undeclared backend")
        _require(partition in experiment_spec["partitions"], f"fidelity tier {name!r} uses undeclared partition")
        _require(partition not in held_out, f"fidelity tier {name!r} cannot use held-out partition {partition!r}")
        cost = _number(item["cost"], f"fidelity tier {name!r} cost")
        _require(cost > 0, f"fidelity tier {name!r} cost must be positive")
        tiers.append({"name": name, "kind": kind, "backend": backend, "partition": partition, "cost": cost})
    _require([item["kind"] for item in tiers] == list(FIDELITY_KINDS),
             "fidelity tiers must be ordered cpu_screen, gpu, replication")
    _require(tiers[0]["backend"] == "numpy", "cpu_screen fidelity must use the numpy backend")
    _require(tiers[1]["backend"] == "cupy" and tiers[2]["backend"] == "cupy",
             "gpu and replication fidelity must use the cupy backend")
    _require(tiers[0]["cost"] <= tiers[1]["cost"] <= tiers[2]["cost"],
             "fidelity tier costs must be nondecreasing")

    observations_raw = raw["observations"]
    _require(isinstance(observations_raw, list), "observations must be a list")
    tier_by_name = {item["name"]: item for item in tiers}
    objective_names = {item["name"] for item in objectives}
    observations = []
    observation_ids = set()
    seen_cells = set()
    for index, item in enumerate(observations_raw):
        _require(isinstance(item, dict) and set(item) == {"id", "status", "parameters", "fidelity",
                                                          "partition", "objectives"},
                 f"observation[{index}] has invalid fields")
        identifier = _text(item["id"], f"observation[{index}] id")
        _require(identifier not in observation_ids, f"duplicate observation id {identifier!r}")
        observation_ids.add(identifier)
        _require(item["status"] == "complete", f"observation {identifier!r} must be complete")
        fidelity = _text(item["fidelity"], f"observation {identifier!r} fidelity")
        _require(fidelity in tier_by_name, f"observation {identifier!r} has unknown fidelity")
        partition = _text(item["partition"], f"observation {identifier!r} partition")
        _require(not _held_out(partition) and partition not in held_out,
                 f"observation {identifier!r} attempts to use held-out data")
        _require(partition == tier_by_name[fidelity]["partition"],
                 f"observation {identifier!r} partition does not match its fidelity")
        parameters = item["parameters"]
        _require(isinstance(parameters, dict) and set(parameters) == set(space),
                 f"observation {identifier!r} parameters must exactly match parameter_space")
        normalized_parameters = {name: _valid_parameter_value(name, parameters[name], space[name])
                                 for name in sorted(space)}
        _require(all(_predicate(rule["predicate"], normalized_parameters) for rule in constraints),
                 f"observation {identifier!r} violates a hard biological constraint")
        values = item["objectives"]
        _require(isinstance(values, dict) and set(values) == objective_names,
                 f"observation {identifier!r} must report every objective")
        normalized_values = {name: _number(value, f"observation {identifier!r} objective {name!r}")
                             for name, value in values.items()}
        cell = (fidelity, _digest(normalized_parameters))
        _require(cell not in seen_cells, f"duplicate completed observation for fidelity {fidelity!r}")
        seen_cells.add(cell)
        observations.append({"id": identifier, "status": "complete", "parameters": normalized_parameters,
                             "fidelity": fidelity, "partition": partition, "objectives": normalized_values})

    return {"schema": SCHEMA, "id": design_id,
            "experiment": {"spec_path": relative_spec, "id": experiment_spec["id"],
                           "held_out_partitions": sorted(held_out)},
            "parameter_space": space, "constraints": constraints, "objectives": objectives,
            "fidelity_tiers": tiers, "observations": observations, "policy": policy}


def load_adaptive_design(path: str | Path, *, root: str | Path = ROOT) -> dict[str, Any]:
    """Load and validate an adaptive design without reading any held-out results."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdaptiveExperimentError(f"cannot read adaptive design {path}: {exc}") from exc
    _validate_design(value, root=Path(root).resolve())
    return value


def _point_from_unit(unit: Sequence[float], space: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    point = {}
    for coordinate, (name, spec) in zip(unit, space.items()):
        value = min(max(float(coordinate), 0.0), np.nextafter(1.0, 0.0))
        if spec["type"] == "continuous":
            if spec["transform"] == "log":
                raw = math.exp(math.log(spec["low"]) + value * (math.log(spec["high"]) - math.log(spec["low"])))
            else:
                raw = spec["low"] + value * (spec["high"] - spec["low"])
            point[name] = float(f"{raw:.12g}")
        else:
            point[name] = spec["values"][min(int(value * len(spec["values"])), len(spec["values"]) - 1)]
    return point


def _fallback_halton(count: int, dimensions: int, seed: int) -> np.ndarray:
    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)
    _require(dimensions <= len(primes), "SciPy is unavailable and fallback supports at most 16 parameters")
    offset = seed * 104729 + 1
    result = np.empty((count, dimensions), dtype=float)
    for row in range(count):
        index = row + offset
        for column, base in enumerate(primes[:dimensions]):
            fraction = 0.0
            factor = 1.0 / base
            n = index
            while n:
                fraction += factor * (n % base)
                n //= base
                factor /= base
            result[row, column] = fraction
    return result


def _finite_grid(space: Mapping[str, Mapping[str, Any]], limit: int) -> list[dict[str, Any]] | None:
    if any(spec["type"] == "continuous" for spec in space.values()):
        return None
    size = math.prod(len(spec["values"]) for spec in space.values())
    if size > limit:
        return None
    names = list(space)
    return [dict(zip(names, values)) for values in itertools.product(*(space[name]["values"] for name in names))]


def _candidate_pool(design: Mapping[str, Any]) -> tuple[list[dict[str, Any]], float, str]:
    space = design["parameter_space"]
    count = design["policy"]["candidate_pool_size"]
    finite = _finite_grid(space, count)
    if finite is not None:
        raw = finite
        method = "exhaustive-finite-grid"
    else:
        if qmc is not None:
            exponent = math.ceil(math.log2(count))
            unit = qmc.Sobol(d=len(space), scramble=True, seed=design["policy"]["seed"]).random_base2(exponent)[:count]
            method = "scipy.stats.qmc.Sobol-scrambled"
        else:  # pragma: no cover
            unit = _fallback_halton(count, len(space), design["policy"]["seed"])
            method = "deterministic-Halton-fallback"
        raw = [_point_from_unit(row, space) for row in unit]
    unique = {_digest(point): point for point in raw}
    feasible = [point for _, point in sorted(unique.items())
                if all(_predicate(rule["predicate"], point) for rule in design["constraints"])]
    ratio = len(feasible) / max(1, len(unique))
    return feasible, ratio, method


def _encode(point: Mapping[str, Any], space: Mapping[str, Mapping[str, Any]]) -> np.ndarray:
    values = []
    for name, spec in space.items():
        value = point[name]
        if spec["type"] == "continuous":
            if spec["transform"] == "log":
                normalized = (math.log(value) - math.log(spec["low"])) / (math.log(spec["high"]) - math.log(spec["low"]))
            else:
                normalized = (value - spec["low"]) / (spec["high"] - spec["low"])
            values.append(normalized)
        elif spec["type"] == "discrete":
            values.append(spec["values"].index(value) / (len(spec["values"]) - 1))
        else:
            values.extend(1.0 if value == choice else 0.0 for choice in spec["values"])
    return np.asarray(values, dtype=float)


def _utility(observation: Mapping[str, Any], objectives: Sequence[Mapping[str, Any]]) -> float:
    total = 0.0
    for objective in objectives:
        value = observation["objectives"][objective["name"]]
        low, high = objective["range"]
        score = (value - low) / (high - low)
        if objective["direction"] == "minimize":
            score = 1.0 - score
        total += objective["weight"] * min(1.0, max(0.0, score))
    return total


def _targets_met(observation: Mapping[str, Any], objectives: Sequence[Mapping[str, Any]]) -> bool:
    targeted = [item for item in objectives if item["target"] is not None]
    if not targeted:
        return False
    return all((observation["objectives"][item["name"]] >= item["target"])
               if item["direction"] == "maximize"
               else (observation["objectives"][item["name"]] <= item["target"])
               for item in targeted)


class _RbfSurrogate:
    def __init__(self, points: np.ndarray, values: np.ndarray):
        self.points = points
        pairwise = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        positive = pairwise[pairwise > 0]
        self.length = float(np.median(positive)) if positive.size else 1.0
        self.length = max(self.length, 1e-6)
        kernel = np.exp(-0.5 * (pairwise / self.length) ** 2)
        self.inverse = np.linalg.pinv(kernel + np.eye(len(points)) * 1e-6, hermitian=True)
        self.coefficients = self.inverse @ values

    def predict(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        distances = np.linalg.norm(points[:, None, :] - self.points[None, :, :], axis=2)
        kernel = np.exp(-0.5 * (distances / self.length) ** 2)
        mean = kernel @ self.coefficients
        variance = 1.0 - np.einsum("ij,jk,ik->i", kernel, self.inverse, kernel)
        return np.clip(mean, 0.0, 1.0), np.sqrt(np.clip(variance, 0.0, 1.0))


def _maximin(points: list[dict[str, Any]], count: int, space: Mapping[str, Any],
             existing: Sequence[Mapping[str, Any]]) -> list[tuple[dict[str, Any], float]]:
    available = list(points)
    selected = []
    anchors = [_encode(item, space) for item in existing]
    midpoint = _encode(_midpoint(space), space)
    while available and len(selected) < count:
        encoded = np.stack([_encode(item, space) for item in available])
        if anchors:
            distances = np.min(np.linalg.norm(encoded[:, None, :] - np.stack(anchors)[None, :, :], axis=2), axis=1)
        else:
            distances = np.linalg.norm(encoded - midpoint, axis=1)
        best = max(range(len(available)), key=lambda index: (float(distances[index]), _digest(available[index])))
        point = available.pop(best)
        score = float(distances[best])
        selected.append((point, score))
        anchors.append(_encode(point, space))
    return selected


def _surrogate_diagnostics(model: _RbfSurrogate | None, candidates: Sequence[Mapping[str, Any]],
                           space: Mapping[str, Any], observation_count: int) -> dict[str, Any]:
    if model is None or len(candidates) < 4:
        return {"status": "insufficient_data", "observations": observation_count,
                "sensitivity": [], "interactions": []}
    sample = list(candidates[: min(128, len(candidates))])
    base = np.stack([_encode(point, space) for point in sample])
    base_prediction, _ = model.predict(base)
    rotated = sample[1:] + sample[:1]
    sensitivity = []
    changed_predictions = {}
    for name in space:
        changed = [{**point, name: alternate[name]} for point, alternate in zip(sample, rotated)]
        prediction, _ = model.predict(np.stack([_encode(point, space) for point in changed]))
        changed_predictions[name] = prediction
        sensitivity.append({"parameter": name, "mean_absolute_effect": float(np.mean(np.abs(prediction - base_prediction)))})
    sensitivity.sort(key=lambda item: (-item["mean_absolute_effect"], item["parameter"]))
    interactions = []
    for left, right in itertools.combinations(space, 2):
        changed = [{**point, left: alternate[left], right: alternate[right]}
                   for point, alternate in zip(sample, rotated)]
        joint, _ = model.predict(np.stack([_encode(point, space) for point in changed]))
        residual = joint - changed_predictions[left] - changed_predictions[right] + base_prediction
        interactions.append({"parameters": [left, right], "root_mean_square_interaction": float(np.sqrt(np.mean(residual ** 2)))})
    interactions.sort(key=lambda item: (-item["root_mean_square_interaction"], item["parameters"]))
    return {"status": "available", "observations": observation_count,
            "method": "deterministic cyclic-permutation effects on RBF surrogate",
            "sensitivity": sensitivity, "interactions": interactions}


def _pareto_ids(observations: Sequence[Mapping[str, Any]], objectives: Sequence[Mapping[str, Any]]) -> list[str]:
    result = []
    for candidate in observations:
        dominated = False
        for other in observations:
            if other is candidate:
                continue
            no_worse = []
            better = []
            for objective in objectives:
                a = other["objectives"][objective["name"]]
                b = candidate["objectives"][objective["name"]]
                sign = 1 if objective["direction"] == "maximize" else -1
                no_worse.append(sign * a >= sign * b)
                better.append(sign * a > sign * b)
            if all(no_worse) and any(better):
                dominated = True
                break
        if not dominated:
            result.append(candidate["id"])
    return sorted(result)


def _promotion_candidates(
    design: Mapping[str, Any],
) -> list[tuple[dict[str, Any], dict[str, Any], float, float]]:
    tiers = design["fidelity_tiers"]
    observations = design["observations"]
    completed = {(item["fidelity"], _digest(item["parameters"])) for item in observations}
    result = []
    for index in range(len(tiers) - 1):
        source = tiers[index]
        destination = tiers[index + 1]
        source_observations = [item for item in observations if item["fidelity"] == source["name"]]
        if len(source_observations) < design["policy"]["min_completed_for_promotion"]:
            continue
        utilities = np.asarray([_utility(item, design["objectives"]) for item in source_observations])
        threshold = float(np.quantile(utilities, design["policy"]["promotion_quantile"]))
        for observation, utility in zip(source_observations, utilities):
            fingerprint = _digest(observation["parameters"])
            if (destination["name"], fingerprint) in completed:
                continue
            if float(utility) >= threshold and (_targets_met(observation, design["objectives"]) or not any(
                    item["target"] is not None for item in design["objectives"])):
                efficiency = float(utility) / destination["cost"]
                result.append((observation, destination, float(utility), efficiency))
    return sorted(result, key=lambda item: (-item[3], -item[2], item[1]["kind"],
                                            _digest(item[0]["parameters"])))


def propose_next_batch(design: Mapping[str, Any], *, root: str | Path = ROOT) -> dict[str, Any]:
    """Validate *design* and deterministically propose a non-executing next batch."""
    validated = _validate_design(dict(design), root=Path(root).resolve())
    observations = validated["observations"]
    policy = validated["policy"]
    tiers = validated["fidelity_tiers"]
    screen = tiers[0]
    screen_observations = [item for item in observations if item["fidelity"] == screen["name"]]
    utilities = [_utility(item, validated["objectives"]) for item in screen_observations]
    promotions = _promotion_candidates(validated)

    stop_reasons = []
    research_reasons = []
    budget_reached = len(observations) >= policy["max_completed_observations"]
    if budget_reached:
        stop_reasons.append("maximum completed-observation budget reached")
    replicated = [item for item in observations if item["fidelity"] == tiers[-1]["name"]]
    replicated_targets_met = (policy["stop_on_replicated_targets"]
                              and any(_targets_met(item, validated["objectives"]) for item in replicated))
    if replicated_targets_met:
        stop_reasons.append("all declared targets were met at replication fidelity")
    window = policy["plateau_window"]
    plateau_reached = False
    if len(utilities) >= 2 * window:
        prior_best = max(utilities[:-window])
        recent_best = max(utilities[-window:])
        if recent_best - prior_best < policy["min_improvement"] and not promotions:
            plateau_reached = True
            stop_reasons.append("screen-fidelity utility plateau reached the preregistered threshold")

    candidates, feasible_fraction, pool_method = _candidate_pool(validated)
    observed_screen = {_digest(item["parameters"]) for item in screen_observations}
    unobserved = [item for item in candidates if _digest(item) not in observed_screen]
    exhausted = not unobserved
    no_feasible_candidates = not candidates
    exhausted_without_promotion = exhausted and not promotions
    if no_feasible_candidates:
        research_reasons.append("hard biological constraints leave no feasible candidate")
    elif exhausted_without_promotion:
        research_reasons.append("bounded parameter space is exhausted without a promotable result")
    low_feasible_fraction = (len(observations) >= policy["research_after_observations"]
                             and feasible_fraction < policy["min_feasible_fraction"])
    if low_feasible_fraction:
        research_reasons.append("feasible-space fraction is below the research escalation threshold")

    model = None
    model_uncertainty = None
    ranked_new: list[tuple[dict[str, Any], float, float, float]] = []
    surrogate_threshold = max(policy["initial_design_size"], policy["min_surrogate_observations"])
    model_exhausted = False
    if len(screen_observations) >= surrogate_threshold and unobserved:
        train_x = np.stack([_encode(item["parameters"], validated["parameter_space"])
                            for item in screen_observations])
        model = _RbfSurrogate(train_x, np.asarray(utilities))
        candidate_x = np.stack([_encode(item, validated["parameter_space"]) for item in unobserved])
        mean, uncertainty = model.predict(candidate_x)
        acquisition = mean + policy["exploration_weight"] * uncertainty
        ranked_new = sorted(zip(unobserved, acquisition, mean, uncertainty),
                            key=lambda item: (-float(item[1]), _digest(item[0])))
        model_uncertainty = float(max(uncertainty))
        if (not promotions and len(observations) >= policy["research_after_observations"]
                and model_uncertainty <= policy["max_model_uncertainty"]
                and float(max(mean)) <= (max(utilities) + policy["min_improvement"])):
            model_exhausted = True
            research_reasons.append("surrogate is confident that the bounded space offers no material improvement")

    if stop_reasons:
        decision = "stop"
    elif research_reasons:
        decision = "escalate_to_research"
    else:
        decision = "propose"

    proposals = []
    used = set()
    active_promotions = promotions if decision == "propose" else []
    for observation, tier, utility, efficiency in active_promotions[
            : min(policy["promotion_slots"], policy["batch_size"])]:
        fingerprint = _digest(observation["parameters"])
        used.add((tier["name"], fingerprint))
        proposals.append({"parameters": observation["parameters"], "fidelity": tier["name"],
                          "fidelity_kind": tier["kind"], "backend": tier["backend"],
                          "partition": tier["partition"], "reason": "promote", "source_observation": observation["id"],
                          "acquisition": {"utility": utility, "utility_per_fidelity_cost": efficiency,
                                          "fidelity_cost": tier["cost"], "predicted_utility": None,
                                          "uncertainty": None}})

    remaining = policy["batch_size"] - len(proposals)
    if decision == "propose" and remaining:
        if ranked_new:
            choices = [(point, float(acquisition), float(mean), float(uncertainty))
                       for point, acquisition, mean, uncertainty in ranked_new[:remaining]]
        else:
            initial = _maximin(unobserved, remaining, validated["parameter_space"],
                               [item["parameters"] for item in screen_observations])
            choices = [(point, score, None, None) for point, score in initial]
        for point, acquisition, mean, uncertainty in choices:
            fingerprint = _digest(point)
            if (screen["name"], fingerprint) in used:
                continue
            used.add((screen["name"], fingerprint))
            proposals.append({"parameters": point, "fidelity": screen["name"],
                              "fidelity_kind": screen["kind"], "backend": screen["backend"],
                              "partition": screen["partition"],
                              "reason": "surrogate_acquisition" if model is not None else "space_filling",
                              "source_observation": None,
                              "acquisition": {"utility": float(acquisition),
                                              "utility_per_fidelity_cost": float(acquisition) / screen["cost"],
                                              "fidelity_cost": screen["cost"], "predicted_utility": mean,
                                              "uncertainty": uncertainty}})

    design_digest = _digest(validated)
    for index, proposal in enumerate(proposals):
        proposal["candidate_id"] = hashlib.sha256(_canonical_bytes(
            {"design_sha256": design_digest, "fidelity": proposal["fidelity"],
             "parameters": proposal["parameters"]}
        )).hexdigest()[:20]
        proposal["order"] = index

    diagnostics = _surrogate_diagnostics(model, candidates, validated["parameter_space"],
                                          len(screen_observations))
    diagnostics["pareto_observation_ids_by_fidelity"] = {
        tier["name"]: _pareto_ids(
            [item for item in observations if item["fidelity"] == tier["name"]], validated["objectives"]
        )
        for tier in tiers
    }
    diagnostics["feasible_candidate_fraction"] = feasible_fraction
    diagnostics["surrogate_max_uncertainty"] = model_uncertainty
    batch = {
        "schema": BATCH_SCHEMA,
        "design_id": validated["id"],
        "design_sha256": design_digest,
        "decision": decision,
        "reasons": stop_reasons if decision == "stop" else research_reasons,
        "algorithm": {
            "candidate_design": pool_method,
            "surrogate": "regularized radial-basis kernel with distance-derived uncertainty" if model else None,
            "acquisition": "normalized weighted utility + exploration_weight * uncertainty" if model else "maximin distance",
            "deterministic_seed": policy["seed"],
            "library_versions": {"numpy": np.__version__, "scipy": scipy.__version__ if scipy else None},
        },
        "completed_observation_count": len(observations),
        "candidates": proposals,
        "decision_conditions": {
            "stop": [
                {"id": "observation_budget", "triggered": budget_reached,
                 "threshold": policy["max_completed_observations"]},
                {"id": "replicated_targets", "triggered": replicated_targets_met,
                 "enabled": policy["stop_on_replicated_targets"]},
                {"id": "utility_plateau", "triggered": plateau_reached,
                 "window": window, "minimum_improvement": policy["min_improvement"]},
            ],
            "escalate_to_research": [
                {"id": "no_feasible_candidates", "triggered": no_feasible_candidates},
                {"id": "finite_space_exhausted", "triggered": exhausted_without_promotion},
                {"id": "low_feasible_fraction", "triggered": low_feasible_fraction,
                 "threshold": policy["min_feasible_fraction"]},
                {"id": "surrogate_exhausted", "triggered": model_exhausted,
                 "maximum_uncertainty": policy["max_model_uncertainty"],
                 "minimum_improvement": policy["min_improvement"]},
            ],
        },
        "diagnostics": diagnostics,
        "experiment_handoff": {
            "authority": "tools/experiment.py",
            "experiment_id": validated["experiment"]["id"],
            "spec_path": validated["experiment"]["spec_path"],
            "required_sequence": [
                "materialize candidate parameters as preregistered experiment arms",
                "create_experiment_seal",
                "expand_experiment_jobs for the candidate's named non-held-out partition",
                "execute only the emitted digest-bound job contract",
            ],
            "direct_runner_commands_emitted": False,
            "seal_required": True,
            "digest_bound_job_expansion_required": True,
            "held_out_partitions_accessed": [],
        },
        "sha256": "",
    }
    batch["sha256"] = _digest({key: value for key, value in batch.items() if key != "sha256"})
    return batch


def write_next_batch(path: str | Path, batch: Mapping[str, Any]) -> Path:
    """Create a read-only batch file and refuse any overwrite."""
    path = Path(path).expanduser().absolute()
    _require(batch.get("schema") == BATCH_SCHEMA, "batch has the wrong schema")
    body = {key: value for key, value in batch.items() if key != "sha256"}
    _require(batch.get("sha256") == _digest(body), "batch self-digest is invalid")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if not os.path.lexists(current):
            continue
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise AdaptiveExperimentError(f"cannot inspect batch path: {current}: {exc}") from exc
        _require(not stat.S_ISLNK(mode), f"batch path cannot contain a symlink: {current}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(batch, handle, sort_keys=True, indent=2, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise AdaptiveExperimentError(f"refusing to replace existing batch {path}") from None
    path.chmod(0o444)
    return path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("design", help="adaptive design JSON")
    parser.add_argument("--output", required=True, help="new create-only batch JSON")
    parser.add_argument("--root", default=str(ROOT), help="repository root")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        design = load_adaptive_design(args.design, root=args.root)
        batch = propose_next_batch(design, root=args.root)
        write_next_batch(args.output, batch)
    except AdaptiveExperimentError as exc:
        print(f"adaptive-experiment: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": args.output, "decision": batch["decision"],
                      "candidate_count": len(batch["candidates"]), "sha256": batch["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
