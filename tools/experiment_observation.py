"""Compile sealed executor receipts into adaptive-experiment observations.

This bridge only authenticates and reduces completed evidence. It cannot choose
parameters, dispatch work, read held-out partitions, or issue a verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping, Sequence

from tools.adaptive_experiment import AdaptiveExperimentError, _predicate, _validate_design
from tools.experiment_executor import (
    ExecutorError,
    RECEIPT_SCHEMA,
    _validate_plan,
    _validate_provenance,
)


CONTRACT_SCHEMA = "sim-observation-contract-v1"
OUTPUT_SCHEMA = "sim-observation-output-v1"
CANONICALIZATION = "json-sort-keys-compact-ascii-v1"


class ObservationCompilerError(ValueError):
    """Raised when completed evidence cannot be authenticated mechanically."""


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ObservationCompilerError("value is not canonical JSON data") from exc


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _fail(message: str) -> None:
    raise ObservationCompilerError(message)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or "\x00" in value:
        _fail(f"{field} must be non-empty trimmed text")
    return value


def _sha(value: Any, field: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _self_digested(document: Mapping[str, Any], field: str) -> None:
    expected = _sha(document.get("sha256"), f"{field} self digest")
    if expected != digest({key: value for key, value in document.items() if key != "sha256"}):
        _fail(f"{field} self digest is invalid")


def _file_state(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _safe_path(root: Path, value: Any, field: str, *, require_file: bool = True) -> tuple[str, Path]:
    if not isinstance(value, (str, os.PathLike)):
        _fail(f"{field} must be a path string")
    supplied = Path(os.fspath(value)).expanduser()
    candidate = supplied if supplied.is_absolute() else root / supplied
    root = root.resolve(strict=True)
    try:
        relative = candidate.absolute().relative_to(root)
    except ValueError:
        _fail(f"{field} escapes repository root")
    pure = PurePosixPath(relative.as_posix())
    if not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        _fail(f"{field} is not a canonical repository-relative path")
    current = root
    for part in pure.parts:
        current /= part
        if os.path.lexists(current) and stat.S_ISLNK(current.lstat().st_mode):
            _fail(f"{field} cannot contain a symlink")
    try:
        resolved = candidate.resolve(strict=require_file)
        resolved.relative_to(root)
    except (OSError, ValueError):
        _fail(f"{field} is missing or escapes repository root")
    if require_file and (not resolved.is_file() or resolved.is_symlink()):
        _fail(f"{field} must be a regular file")
    return pure.as_posix(), resolved


def _read_bytes(path: Path, field: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ObservationCompilerError(f"cannot open {field}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _fail(f"{field} is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read()
        after = os.fstat(descriptor)
        named = path.lstat()
        if (_file_state(before) != _file_state(after) or stat.S_ISLNK(named.st_mode)
                or _file_state(named) != _file_state(after)):
            _fail(f"{field} changed while being read")
        return raw
    finally:
        os.close(descriptor)


def _json_file(root: Path, value: Any, field: str) -> tuple[dict[str, str], dict[str, Any], Path]:
    relative, path = _safe_path(root, value, field)
    raw = _read_bytes(path, field)
    try:
        document = json.loads(raw, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ObservationCompilerError(f"{field} is not valid finite JSON") from exc
    if not isinstance(document, dict):
        _fail(f"{field} must contain a JSON object")
    return {"path": relative, "sha256": hashlib.sha256(raw).hexdigest()}, document, path


def _bound_json(root: Path, binding: Any, field: str) -> tuple[dict[str, str], dict[str, Any], Path]:
    if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
        _fail(f"{field} binding must contain exactly path and sha256")
    expected = _sha(binding.get("sha256"), f"{field} binding digest")
    reference, document, path = _json_file(root, binding.get("path"), field)
    if reference["sha256"] != expected:
        _fail(f"{field} digest does not match")
    return reference, document, path


def _contract(root: Path, path: str | Path) -> tuple[dict[str, str], dict[str, Any]]:
    reference, contract, _ = _json_file(root, path, "observation contract")
    _self_digested(contract, "observation contract")
    required = {
        "schema", "id", "status", "bindings", "objectives", "fidelity_mapping",
        "required_seeds", "held_out_seeds", "output_path", "sha256",
    }
    if set(contract) != required or contract.get("schema") != CONTRACT_SCHEMA \
            or contract.get("status") != "preregistered":
        _fail("observation contract has an invalid shape or status")
    _text(contract.get("id"), "contract id")
    bindings = contract.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {"adaptive_design", "executor_manifest"}:
        _fail("observation contract bindings are incomplete")
    objectives = contract.get("objectives")
    if not isinstance(objectives, Mapping) or not objectives:
        _fail("observation contract objectives must be non-empty")
    for name, source in objectives.items():
        _text(name, "objective name")
        if (not isinstance(source, Mapping) or set(source) != {"arm", "path", "reducer"}
                or source.get("reducer") != "mean" or not isinstance(source.get("path"), list)
                or not source["path"] or any(not isinstance(part, str) or not part for part in source["path"])):
            _fail(f"objective source {name!r} is invalid")
        _text(source.get("arm"), f"objective {name!r} arm")
    mappings = contract.get("fidelity_mapping")
    if (not isinstance(mappings, list) or not mappings
            or any(not isinstance(row, Mapping) or set(row) != {"backend", "partition", "fidelity"}
                   for row in mappings)):
        _fail("fidelity_mapping is invalid")
    mapping_keys = [(row["backend"], row["partition"]) for row in mappings]
    if len(set(mapping_keys)) != len(mapping_keys):
        _fail("fidelity_mapping contains duplicate backend/partition pairs")
    for field in ("required_seeds", "held_out_seeds"):
        values = contract.get(field)
        if (not isinstance(values, list) or any(type(value) is not int or value < 0 for value in values)
                or len(set(values)) != len(values)):
            _fail(f"{field} must contain unique non-negative integers")
    if not contract["required_seeds"] or set(contract["required_seeds"]) & set(contract["held_out_seeds"]):
        _fail("required seeds are empty or overlap held-out seeds")
    _text(contract.get("output_path"), "output_path")
    return reference, contract


def _direct_value(document: Mapping[str, Any], parts: Sequence[str]) -> Any:
    current: Any = document
    for part in parts:
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(part)
        current = current[part]
    return current


def _finite_scalar(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _engineering_only(value: Any) -> bool:
    return isinstance(value, str) and "".join(
        character for character in value.lower() if character.isalnum()
    ).startswith("engineering")


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    if os.path.lexists(path):
        _fail(f"refusing to replace existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(canonical_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ObservationCompilerError(f"refusing to replace existing output: {path}") from exc
    path.chmod(0o444)


def compile_executor_receipts_to_observations(
    contract_path: str | Path,
    executor_manifest_path: str | Path,
    receipt_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Authenticate exact executor receipts and produce deterministic observations."""
    root = Path(repository_root).expanduser().resolve(strict=True)
    contract_ref, contract = _contract(root, contract_path)
    design_ref, design_raw, _ = _bound_json(root, contract["bindings"]["adaptive_design"], "adaptive design")
    manifest_ref, plan, _ = _bound_json(root, contract["bindings"]["executor_manifest"], "executor manifest")
    supplied_manifest_ref, supplied_plan, _ = _json_file(root, executor_manifest_path, "supplied executor manifest")
    if supplied_manifest_ref != manifest_ref or supplied_plan != plan:
        _fail("supplied executor manifest does not equal the contract binding")
    try:
        _validate_plan(plan)
        design = _validate_design(design_raw, root=root)
    except (ExecutorError, AdaptiveExperimentError) as exc:
        raise ObservationCompilerError(str(exc)) from exc
    if set(contract["objectives"]) != {item["name"] for item in design["objectives"]}:
        _fail("contract objective mapping does not equal adaptive design objectives")
    fidelity_by_pair = {
        (row["backend"], row["partition"]): row["fidelity"] for row in contract["fidelity_mapping"]
    }
    tiers = {item["name"]: item for item in design["fidelity_tiers"]}
    for pair, fidelity in fidelity_by_pair.items():
        if fidelity not in tiers or pair != (tiers[fidelity]["backend"], tiers[fidelity]["partition"]):
            _fail("fidelity mapping disagrees with adaptive design")

    jobs = {job["job_id"]: job for job in plan["jobs"]}
    expected_arms = {job["arm"] for job in plan["jobs"]}
    objective_arms = {source["arm"] for source in contract["objectives"].values()}
    if not objective_arms <= expected_arms:
        _fail("objective source names an arm outside the executor manifest")
    loaded: dict[str, tuple[dict[str, str], Mapping[str, Any], Mapping[str, Any]]] = {}
    blocked: list[dict[str, Any]] = []
    for receipt_path in receipt_paths:
        receipt_ref, receipt, _ = _json_file(root, receipt_path, "executor receipt")
        _self_digested(receipt, "executor receipt")
        job_id = receipt.get("job_id")
        job = jobs.get(job_id)
        if (receipt.get("schema") != RECEIPT_SCHEMA or job is None
                or receipt.get("executor_manifest_sha256") != plan["sha256"]
                or receipt.get("job_sha256") != job["job_sha256"]):
            _fail("executor receipt is invalid or belongs to another manifest")
        if job_id in loaded:
            _fail(f"duplicate executor receipt for job {job_id}")
        if receipt.get("status") != "succeeded":
            blocked.append({"receipt": receipt_ref, "reason": f"job status is {receipt.get('status')!r}"})
            continue
        result = receipt.get("result")
        if not isinstance(result, Mapping) or result.get("exit_code") != 0:
            _fail("successful executor receipt has an invalid result")
        output_ref, output, _ = _json_file(root, job["output"], "executor output")
        provenance_ref, _, provenance_path = _json_file(root, job["provenance"], "executor provenance")
        if (result.get("output_sha256") != output_ref["sha256"]
                or result.get("provenance_sha256") != provenance_ref["sha256"]):
            _fail("executor output or provenance changed after receipt completion")
        try:
            provenance = _validate_provenance(provenance_path, job)
        except ExecutorError as exc:
            raise ObservationCompilerError(str(exc)) from exc
        if provenance.get("run_id") != result.get("provenance_run_id"):
            _fail("executor provenance run identity changed after completion")
        if output.get("engineering_only") is True or _engineering_only(job.get("partition")):
            blocked.append({"receipt": receipt_ref, "reason": "engineering-only output cannot enter optimization"})
            continue
        loaded[job_id] = (receipt_ref, job, output)

    required_seeds = set(contract["required_seeds"])
    if any(job["seed"] in set(contract["held_out_seeds"]) for _, job, _ in loaded.values()):
        _fail("held-out seed receipt cannot enter observation compilation")
    groups: dict[tuple[str, str], list[tuple[dict[str, str], Mapping[str, Any], Mapping[str, Any]]]] = {}
    for item in loaded.values():
        _, job, _ = item
        pair = (job["backend"], job["partition"])
        if pair not in fidelity_by_pair or not job.get("candidate_id"):
            blocked.append({"receipt": item[0], "reason": "job is outside the adaptive fidelity mapping"})
            continue
        groups.setdefault((job["candidate_id"], fidelity_by_pair[pair]), []).append(item)

    observations = []
    evidence = []
    for (candidate_id, fidelity), rows in sorted(groups.items()):
        jobs_by_cell = {(job["arm"], job["seed"]): (ref, job, output) for ref, job, output in rows}
        expected_cells = {(arm, seed) for arm in expected_arms for seed in required_seeds}
        if set(jobs_by_cell) != expected_cells:
            blocked.append({"candidate_id": candidate_id, "fidelity": fidelity,
                            "reason": "incomplete or widened arm/seed evidence set"})
            continue
        parameter_documents = [job.get("parameter_document") for _, job, _ in rows]
        candidates = [item for item in parameter_documents if isinstance(item, Mapping)]
        if (len(candidates) != len(rows)
                or any(item.get("candidate_id") != candidate_id for item in candidates)
                or len({digest(item.get("candidate_parameters")) for item in candidates}) != 1):
            _fail("group has inconsistent adaptive parameter bindings")
        parameters = dict(candidates[0]["candidate_parameters"])
        if not all(_predicate(rule["predicate"], parameters) for rule in design["constraints"]):
            _fail("executor evidence violates a hard biological constraint")
        objective_values = {}
        missing = None
        for name, source in contract["objectives"].items():
            values = []
            for seed in sorted(required_seeds):
                output = jobs_by_cell[(source["arm"], seed)][2]
                try:
                    value = _direct_value(output, source["path"])
                except KeyError:
                    missing = f"missing objective {name!r}"
                    break
                if not _finite_scalar(value):
                    missing = f"objective {name!r} is not a finite scalar"
                    break
                values.append(float(value))
            if missing:
                break
            objective_values[name] = math.fsum(values) / len(values)
        if missing:
            blocked.append({"candidate_id": candidate_id, "fidelity": fidelity, "reason": missing})
            continue
        observation_id = f"{candidate_id}--{fidelity}"
        observations.append({
            "id": observation_id,
            "status": "complete", "parameters": parameters, "fidelity": fidelity,
            "partition": tiers[fidelity]["partition"], "objectives": objective_values,
        })
        evidence.append({
            "observation_id": observation_id,
            "receipts": sorted((item[0] for item in rows), key=lambda ref: ref["path"]),
        })

    output_relative, output_file = _safe_path(root, output_path, "output", require_file=False)
    if output_relative != contract["output_path"]:
        _fail("output path does not match the preregistered contract")
    body = {
        "schema": OUTPUT_SCHEMA, "canonicalization": CANONICALIZATION,
        "contract": contract_ref, "adaptive_design": design_ref, "executor_manifest": manifest_ref,
        "observations": observations, "evidence": evidence,
        "blocked": sorted(blocked, key=lambda row: canonical_bytes(row)),
        "scientific_verdict": None,
    }
    compiled = {**body, "sha256": digest(body)}
    _write_once(output_file, compiled)
    return compiled


compile_observations = compile_executor_receipts_to_observations


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--executor-manifest", required=True)
    parser.add_argument("--receipt", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository-root", required=True)
    args = parser.parse_args(argv)
    try:
        result = compile_executor_receipts_to_observations(
            args.contract, args.executor_manifest, args.receipt, args.output,
            repository_root=args.repository_root,
        )
    except ObservationCompilerError as exc:
        print(f"experiment-observation: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps({"complete": len(result["observations"]), "blocked": len(result["blocked"]),
                      "output": args.output, "sha256": result["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
